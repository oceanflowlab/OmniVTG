import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
import re
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    VISION_START_TOKEN,
    VISION_END_TOKEN
)
from qwen_vl_utils import process_vision_info


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

class SupervisedSFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedSFTDataset, self).__init__()
        if isinstance(data_path, str):
            if "<task>" in data_path and ":" in data_path:
                list_data_dict = []
                base_path, tasks = data_path.split(":")
                tasks = tasks.split(";")
                for task in tasks:
                    list_data_dict.extend(json.load(open(base_path.replace("<task>", task), "r")))
            else:
                list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height

        self.video_total_pixel = data_args.video_total_pixels
        self.max_frames = data_args.max_frames
        self.interleave_timestamps = data_args.interleave_timestamps

        self.fps = data_args.fps
        self.nframes = data_args.nframes

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
        else:
            self.image_patch_size = 14

    def __len__(self):
        return len(self.list_data_dict)

    def _calculate_timestamps(self, indices, video_fps, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def _interleave_timestamps(self, text, video_metadata, video_grid_thw):
        merge_length = self.processor.image_processor.merge_size**2
        index = 0
        while DEFAULT_VIDEO_TOKEN in text:
            metadata = video_metadata[index]
            # if timestamps are not provided, calculate them
            curr_timestamp = self._calculate_timestamps(
                metadata["frames_indices"],
                metadata["fps"],
                self.processor.video_processor.merge_size,
            )

            video_placeholder = ""
            frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
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

        return text, curr_timestamp

    def _align_timestamps(self, text, cur_timestamps):
        pattern = r"<timestamp>(\d+\.?\d*)</timestamp>"

        def _find_closest_and_replace(match: re.Match) -> str:
            try:
                original_time_str = match.group(1)
                original_time = float(original_time_str)
            except (ValueError, TypeError):
                return match.group(0)

            if not cur_timestamps:
                return match.group(0)

            closest_time = min(cur_timestamps, 
                            key=lambda target_time: abs(target_time - original_time))
            
            return f"{closest_time:.1f}"

        aligned_text = re.sub(pattern, _find_closest_and_replace, text)

        return aligned_text


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        processor = self.processor

        # Load video pixels
        video_path = sources["video"]
        if not os.path.exists(video_path):
            video_path = os.path.join(self.data_args.image_folder, video_path)
        messages = [{
            "role": "user", 
            "content": [{
                "type": "video", 
                "video": video_path,
                "min_pixels": self.video_min_pixel,
                "total_pixels": self.video_total_pixel,
                "max_frames": self.max_frames,
            }]
        }]
        _, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
        video_datas, video_metadatas = zip(*videos)
        video_datas = list(video_datas)
        video_metadatas = list(video_metadatas)

        all_input_ids = []
        all_labels = []

        # Qwen2-VL uses a default system message so I've added this.
        # Prepare system message
        if len(SYSTEM_MESSAGE) > 0 and "Qwen3" not in self.model_id:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        # Process video
        if "Qwen2.5" in self.model_id:
            inputs = processor(text="", images=None, videos=video_datas, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
            second_per_grid_ts = inputs["second_per_grid_ts"]
        else:
            inputs = processor(text="", images=None, videos=video_datas, padding=False, do_resize=False, return_tensors='pt')
        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]

        # Prepare conversation message
        
        user_input = sources["conversations"][0]["value"]
        user_input = user_input.replace("<video>\n", VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN)
        user_input = f"{DEFAULT_IM_START_TOKEN}user\n{user_input}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        if self.interleave_timestamps:
            user_input, cur_timestamps = self._interleave_timestamps(user_input, video_metadatas, video_grid_thw)
            user_input = self._align_timestamps(user_input, cur_timestamps)
        prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

        gpt_response = sources["conversations"][1]["value"]+DEFAULT_IM_END_TOKEN+"\n"
        if self.interleave_timestamps:
            gpt_response = self._align_timestamps(gpt_response, cur_timestamps)
        parts = re.split(r'(<mask>.*?<\/mask>)', gpt_response)
        response_input_ids_list = []
        response_labels_list = []
        for part in parts:
            if not part:
                continue
            if part.startswith('<mask>') and part.endswith('</mask>'):
                masked_content = part[len('<mask>'):-len('</mask>')]
                if not masked_content: 
                    continue
                tokenized_content = processor.tokenizer(masked_content, add_special_tokens=False)['input_ids']
                response_input_ids_list.extend(tokenized_content)
                response_labels_list.extend([IGNORE_INDEX] * len(tokenized_content))
            else:
                tokenized_content = processor.tokenizer(part, add_special_tokens=False)['input_ids']
                response_input_ids_list.extend(tokenized_content)
                response_labels_list.extend(tokenized_content)
        response_input_ids = torch.tensor([response_input_ids_list], dtype=torch.long)
        response_labels = torch.tensor([response_labels_list], dtype=torch.long)

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        labels = torch.cat([torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])), response_labels.squeeze(0)], dim=0)

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        if self.interleave_timestamps:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
            second_per_grid_ts = [0] * len(video_grid_thw)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw
        )
        if "Qwen2.5" in self.model_id:
            data_dict["second_per_grid_ts"] = second_per_grid_ts

        return data_dict
    

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedSFTDataset 

    sft_dataset = dataset_cls(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = dataset_cls(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
