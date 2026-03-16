# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
from typing import Optional
import json

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
SYSTEM_MESSAGE = "You are a helpful assistant."

PROMPTS = {
    "cot": """You are given a video as a sequence of interleaved timestamps and frames. Locate the precise timestamps for the event: "{query}".

Use a coarse-to-fine reasoning: first state the broad segment where related content may occur, then refine to localize the specific query. Every reasoning step must include timestamps in the form xx.x seconds.

Wrap the step-by-step reasoning in <think>...</think>. After that, output only the final answer in this exact format inside <answer>...</answer>: From start_time seconds to end_time seconds
For example:
<think>
For the query "a woman opens the door", I find that the woman appears in the video from 12.5 seconds to 20.0 seconds. Zooming in further, she opens the door from 14.2 seconds to 14.7 seconds.
</think>
<answer>
From 14.2 seconds to 14.7 seconds
</answer>"""
}


class VTGGRPODataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.prompt_type = config.get("prompt_type", "cot")
        self.interleave_timestamps = config.get("interleave_timestamps", False)
        if self.interleave_timestamps:
            print('input video will be interleaved with timestamps')
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._read_files()


    def _read_files(self):
        dataframes = []
        for json_file in self.data_files:
            dataframe = json.load(open(json_file, "r"))
            for d in dataframe:
                d['with_prompt'] = d.get('with_prompt', False)
                if not isinstance(d['reward_model']['ground_truth'], list):
                    d['reward_model']['ground_truth'] = [d['reward_model']['ground_truth'], 0]
            dataframes.extend(dataframe)
        self.dataframe: datasets.Dataset = datasets.Dataset.from_list(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_missing_videos(self.dataframe)

    def maybe_filter_out_missing_videos(self, dataframe: datasets.Dataset = None):
        filt_dataframe = []
        for data in dataframe:
            if os.path.exists(data["video"]):
                filt_dataframe.append(data)
        
        dataframe = dataframe.filter(lambda data: os.path.exists(data["video"]))
        
        print(f"filter dataset len: {len(filt_dataframe)}")
        return filt_dataframe

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        if example.get('with_prompt', False):
            prompt_text = example["sentence"]
        else:
            query = example["sentence"].strip()
            if query.endswith('.'):
                query = query[:-1]
            prompt_text = PROMPTS[self.prompt_type].format(query=query)

        video = {
            "type": "video",
            "video": example["video"],
            "total_pixels": 3584 * 28 * 28,
            "min_pixels": 16 * 28 * 28,
        }
        if example.get("video_start", None):
            video["video_start"] = example["video_start"]
        if example.get("video_end", None):
            video["video_end"] = example["video_end"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    video,
                ],
            },
        ]
        
        return messages, video
    

    def _read_video(self, example):
        video = {
            "type": "video",
            "video": example["video"],
            "total_pixels": 3584 * 28 * 28,
            "min_pixels": 16 * 28 * 28,
            "max_frames": 768,
        }
        if example.get("video_start", None):
            video["video_start"] = example["video_start"]
        if example.get("video_end", None):
            video["video_end"] = example["video_end"]
        
        messages = [{
            "role": "user", 
            "content": [video],
        }]

        _, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
        video_datas, video_metadatas = zip(*videos)
        video_datas = list(video_datas)
        video_metadatas = list(video_metadatas)

        return video_datas, video_metadatas, video_kwargs
    
    def _calculate_timestamps(self, indices, video_fps, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        start_idx = indices[0]
        timestamps = [(idx - start_idx) / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps
    
    def _interleave_timestamps(self, text, video_metadata, video_grid_thw):
        merge_length = 4
        index = 0
        while DEFAULT_VIDEO_TOKEN in text:
            metadata = video_metadata[index]
            # if timestamps are not provided, calculate them
            curr_timestamp = self._calculate_timestamps(
                metadata["frames_indices"],
                metadata["fps"],
                2,
            )

            video_placeholder = ""
            # frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
            frame_seqlen = 1
            for frame_idx in range(video_grid_thw[index][0]):
                curr_time = curr_timestamp[frame_idx]
                video_placeholder += f"<{curr_time:.1f} seconds>"
                video_placeholder += (
                    VISION_START_TOKEN + "<|placeholder|>" * frame_seqlen + VISION_END_TOKEN
                )
            if f"{VISION_START_TOKEN}{DEFAULT_VIDEO_TOKEN}{VISION_END_TOKEN}" in text:
                text = text.replace(
                    f"{VISION_START_TOKEN}{DEFAULT_VIDEO_TOKEN}{VISION_END_TOKEN}", video_placeholder, 1
                )
            else:
                # vllm may input video token directly
                text = text.replace(DEFAULT_VIDEO_TOKEN, video_placeholder, 1)
            index += 1

        text = text.replace("<|placeholder|>", DEFAULT_VIDEO_TOKEN)
        # text = text.replace("<|placeholder|>", DEFAULT_IMAGE_TOKEN) # treat video as multiple images

        return text, curr_timestamp

    def _repeat_video_token(self, text, grid_thw):
        if DEFAULT_VIDEO_TOKEN in text:
            merge_length = 4
            index = 0
            while DEFAULT_VIDEO_TOKEN in text:
                num_video_tokens = grid_thw[index].prod() // merge_length
                text = text.replace(DEFAULT_VIDEO_TOKEN, "<|placeholder|>" * num_video_tokens, 1)
                index += 1

            text = text.replace("<|placeholder|>", DEFAULT_VIDEO_TOKEN)
        elif DEFAULT_IMAGE_TOKEN in text:
            # interleave timestamps
            merge_length = 4
            index = 0
            while DEFAULT_IMAGE_TOKEN in text:
                num_image_tokens = grid_thw[index].prod() // merge_length
                text = text.replace(DEFAULT_IMAGE_TOKEN, "<|placeholder|>" * num_image_tokens, 1)
                index += 1
            text = text.replace("<|placeholder|>", DEFAULT_VIDEO_TOKEN) # also regard as video

        return text

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # process video
        row_dict: dict = self.dataframe[item]
        video_datas, video_metadatas, video_kwargs = self._read_video(row_dict)
        video_inputs = self.processor(text="", images=None, videos=video_datas, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
        second_per_grid_ts = video_inputs["second_per_grid_ts"]
        pixel_values_videos = video_inputs["pixel_values_videos"]
        video_grid_thw = video_inputs["video_grid_thw"]

        # process input tokens

        # system
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_message_input_ids = self.processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']

        # user
        if row_dict.get('with_prompt', False):
            prompt_text = row_dict["sentence"]
        else:
            query = row_dict["sentence"].strip()
            if query.endswith('.'):
                query = query[:-1]
            prompt_text = PROMPTS[self.prompt_type].format(query=query)
        user_input = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN + prompt_text
        user_input = f"{DEFAULT_IM_START_TOKEN}user\n{user_input}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        if self.interleave_timestamps:
            user_input, cur_timestamps = self._interleave_timestamps(user_input, video_metadatas, video_grid_thw)
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
            second_per_grid_ts = [0] * len(video_grid_thw)
        else:
            # num_video_tokens = video_grid_thw[0].prod() // 4
            num_video_tokens = 1
            user_input = user_input.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_TOKEN * num_video_tokens, 1)
        raw_prompt = system_message + user_input
        user_input = self._repeat_video_token(user_input, video_grid_thw)
        prompt_input_ids = self.processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        input_ids = torch.cat([system_message_input_ids, prompt_input_ids], dim=1).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        mm_data = {
            "video_embeds": None,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts
        }
        mm_key = "video"
        row_dict["multi_modal_data"] = {mm_key: mm_data}

        # We will do batch.union() in the trainer,
        # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
        if self.return_multi_modal_inputs:
            row_dict["multi_modal_inputs"] = {
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # qwen-vl mrope
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids[0],
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask[0],
        )  # (3, seq_length)
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        row_dict["extra_info"]["duration"] = row_dict["duration"]
        
        return row_dict


    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._read_files()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
