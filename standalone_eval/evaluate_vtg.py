import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from dataset_config import VTG

import json
import os
import re
import numpy as np
from qwen_vl_utils import process_vision_info
import argparse
from tqdm import trange, tqdm

import logging
logging.basicConfig(level=logging.ERROR)

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
</answer>""",
}

def _calculate_timestamps(indices, video_fps, merge_size: int = 2):
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

def _interleave_timestamps(text, video_metadata, video_grid_thw):
    merge_length = 4
    index = 0
    while DEFAULT_VIDEO_TOKEN in text:
        metadata = video_metadata[index]
        curr_timestamp = _calculate_timestamps(
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


def prepare_inputs(prompt, video_path, processor, interleave_timestamps=False, total_pixels=3584*28*28, fps=2, video_start=None, video_end=None):
    messages = [{
        "role": "user", 
        "content": [{
            "type": "video", 
            "video": video_path,
            "min_pixels": 16 * 28 * 28,
            "total_pixels": total_pixels,
            "max_frames": 768,
            "fps": fps,
            "video_start": video_start,
            "video_end": video_end,
        }]
    }]
    _, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
    video_datas, video_metadatas = zip(*videos)
    video_datas = list(video_datas)
    video_metadatas = list(video_metadatas)

    all_input_ids = []

    # Qwen2-VL uses a default system message so I've added this.
    # Prepare system message
    if len(SYSTEM_MESSAGE) > 0:
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
        all_input_ids.append(system_message_input_ids.squeeze(0))

    # Process video
    inputs = processor(text="", images=None, videos=video_datas, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
    second_per_grid_ts = inputs["second_per_grid_ts"]
    pixel_values_videos = inputs["pixel_values_videos"]
    video_grid_thw = inputs["video_grid_thw"]

    # Prepare conversation message
    
    user_input = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN + prompt
    user_input = f"{DEFAULT_IM_START_TOKEN}user\n{user_input}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
    if interleave_timestamps:
        user_input, cur_timestamps = _interleave_timestamps(user_input, video_metadatas, video_grid_thw)
    else:
        # num_video_tokens = video_grid_thw[0].prod() // 4
        num_video_tokens = 1
        user_input = user_input.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_TOKEN * num_video_tokens, 1)
        
    prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

    all_input_ids.append(prompt_input_ids.squeeze(0))
    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long).unsqueeze(0)

    if interleave_timestamps:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
        second_per_grid_ts = [0] * len(video_grid_thw)

    data_dict = dict(
        input_ids=input_ids,
        attention_mask=(input_ids > -1000000).to(torch.long),
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw
    )
    data_dict["second_per_grid_ts"] = second_per_grid_ts

    return data_dict

def extract_time(paragraph):
    numbers = re.findall(r'\d+(?:\.\d+)?', paragraph)
    numbers = [float(num) for num in numbers]

    if len(numbers) >= 2:
        last_two = numbers[-2:]
        if last_two[0] > last_two[1]:
            last_two[0], last_two[1] = last_two[1], last_two[0]
        
        return last_two
    else:
        return [None, None]

def calculate_iou(gt_segment, pred_segment):
    if pred_segment[0] is None or pred_segment[1] is None:
        return 0.0

    gt_start, gt_end = gt_segment
    pred_start, pred_end = pred_segment

    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter_len = max(0, inter_end - inter_start)

    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union_len = union_end - union_start

    if union_len == 0:
        return 0.0

    iou = inter_len / union_len
    return iou


class VideoDataset(Dataset):
    def __init__(self, data, model_path, video_prefix, prompt, interleave_timestamps, total_pixels, fps):
        self.data = data
        self.model_path = model_path
        self.video_prefix = video_prefix
        self.prompt_template = prompt
        self.interleave_timestamps = interleave_timestamps
        self.total_pixels = total_pixels
        self.fps = fps
        
        self.processor = None

    def _get_processor(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        return self.processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        processor = self._get_processor()
        d = self.data[idx]

        try:
            video_path = os.path.join(self.video_prefix, d['video_file'])
            if not os.path.exists(video_path):
                print(f"Worker PID {os.getpid()}: Video file not found, skipping: {video_path}")
                return None, None 

            user_prompt = self.prompt_template.format(query=d['query'])
            
            relevant_windows = d["relevant_windows"]
            if isinstance(relevant_windows[0], list):
                gt_start, gt_end = relevant_windows[0]
            else:
                gt_start, gt_end = relevant_windows
            
            inputs = prepare_inputs(
                user_prompt, video_path, processor, 
                interleave_timestamps=self.interleave_timestamps, 
                total_pixels=self.total_pixels,
                fps=self.fps,
                video_start=d.get("video_start", None),
                video_end=d.get("video_end", None),
            )
            
            input_ids_list = inputs['input_ids'].squeeze(0).tolist()
            
            mm_data = {
                "video_embeds": None,
                "pixel_values_videos": inputs["pixel_values_videos"],
                "video_grid_thw": inputs["video_grid_thw"]
            }
            mm_key = "video"
            mm_data["second_per_grid_ts"] = inputs["second_per_grid_ts"]
            
            vllm_input = {
                "prompt_token_ids": input_ids_list,
                "multi_modal_data": { mm_key: mm_data }
            }
            
            return vllm_input, d

        except Exception as e:
            print(f"Worker PID {os.getpid()}: Error preparing inputs for {d.get('video_file')}: {e}")
            return None, None


def vllm_collate_fn(batch):
    batch_inputs = []
    batch_original_data = []
    
    for inp, d in batch:
        if inp is not None and d is not None:
            batch_inputs.append(inp)
            batch_original_data.append(d)
            
    return batch_inputs, batch_original_data


def aggregate_and_calculate(output_dir, world_size):
    print("Main process: Aggregating results...")
    
    all_results = []
    
    for rank in range(world_size):
        result_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
        try:
            with open(result_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line in {result_file}: {line}")
        except FileNotFoundError:
            print(f"Warning: Result file not found, skipping: {result_file}")

    if not all_results:
        print("Error: No results found to aggregate.")
        return

    print(f"Total aggregated results: {len(all_results)}")

    aggregated_file = os.path.join(output_dir, "all_results_aggregated.jsonl")
    with open(aggregated_file, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res) + '\n')
    print(f"Aggregated results saved to {aggregated_file}")
    
    ious = []
    recalls_03 = []
    recalls_05 = []
    recalls_07 = []

    for item in all_results:
        gt_segments = item['relevant_windows']
        pred_segment = item['pred_relevant_window']
        
        max_iou = 0
        for gt_segment in gt_segments:
            iou = calculate_iou(gt_segment, pred_segment)
            max_iou = max(max_iou, iou)
        ious.append(max_iou)
        
        recalls_03.append(1 if max_iou >= 0.3 else 0)
        recalls_05.append(1 if max_iou >= 0.5 else 0)
        recalls_07.append(1 if max_iou >= 0.7 else 0)

    metrics = {
        "mIoU": np.mean(ious),
        "Recall@0.3": np.mean(recalls_03),
        "Recall@0.5": np.mean(recalls_05),
        "Recall@0.7": np.mean(recalls_07),
        "total_samples": len(all_results),
    }

    if "rare" in all_results[0]["original_data"]:
        ious = []
        recalls_03 = []
        recalls_05 = []
        recalls_07 = []

        for item in all_results:
            if not item["original_data"].get("rare", False):
                continue
            gt_segments = item['relevant_windows']
            pred_segment = item['pred_relevant_window']
            
            max_iou = 0
            for gt_segment in gt_segments:
                iou = calculate_iou(gt_segment, pred_segment)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
            
            recalls_03.append(1 if max_iou >= 0.3 else 0)
            recalls_05.append(1 if max_iou >= 0.5 else 0)
            recalls_07.append(1 if max_iou >= 0.7 else 0)

        metrics.update({
            "Rare_mIoU": np.mean(ious),
            "Rare_Recall@0.3": np.mean(recalls_03),
            "Rare_Recall@0.5": np.mean(recalls_05),
            "Rare_Recall@0.7": np.mean(recalls_07),
            "Rare_total_samples": len(ious),
        })

    print("\n--- Final Metrics ---")
    print(json.dumps(metrics, indent=2))
    print("---------------------\n")

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

def check_device():
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"

def evaluate_worker(rank, world_size, all_data, model_path, video_prefix, output_dir, prompt, interleave_timestamps=False, num_workers=4, batch_size=16, total_pixels=3584*28*28, fps=2):
    if check_device() == "npu":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    print(f"Rank {rank} (PID {os.getpid()}): Initializing vLLM...")

    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=32768,
            max_num_batched_tokens=32768,
            disable_mm_preprocessor_cache=True, 
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 0, "video": 768},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 16 * 28 * 28,
            },
        )

    except Exception as e:
        print(f"Rank {rank}: Error loading vLLM model: {e}")
        return
        
    print(f"Rank {rank}: vLLM Model loaded.")

    num_samples = len(all_data)
    samples_per_gpu = (num_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, num_samples)
    my_data = all_data[start_idx:end_idx]

    print(f"Rank {rank}: Processing {len(my_data)} samples (indices {start_idx} to {end_idx-1}).")

    dataset = VideoDataset(
        data=my_data,
        model_path=model_path,
        video_prefix=video_prefix,
        prompt=prompt,
        interleave_timestamps=interleave_timestamps,
        total_pixels=total_pixels,
        fps=fps
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=vllm_collate_fn,
        pin_memory=False
    )
    
    output_file = open(os.path.join(output_dir, f"results_rank_{rank}.jsonl"), "w")
    
    for batch_inputs, batch_original_data in tqdm(dataloader, desc=f"Rank {rank} Processing"):

        if not batch_inputs:
            continue
            
        sampling_params = SamplingParams(
            repetition_penalty=1.05, 
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            stop=None,
            stop_token_ids=[151645, 151643], 
            max_tokens=512,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

        try:
            outputs = llm.generate(
                prompts=batch_inputs,
                sampling_params=sampling_params,
                use_tqdm=False
            )
        except Exception as e:
            print(f"Rank {rank}: Error during llm.generate: {e}")
            continue

        for j, output in enumerate(outputs):
            d = batch_original_data[j]
            output_text = output.outputs[0].text
            
            pred_start, pred_end = extract_time(output_text)
            
            result = {
                "original_data": d,
                "model_output": output_text,
                "predicted_start": pred_start,
                "pred_relevant_window": [pred_start, pred_end], 
                "relevant_windows": d["relevant_windows"]
            }
            print(json.dumps(result), file=output_file)
            output_file.flush()
    
    output_file.close()


def main(args):
    world_size = torch.accelerator.device_count()
    print(f"Found {world_size} devices.")

    os.makedirs(args.output_dir, exist_ok=True)

    data_file = VTG[args.dataset]["annotation"]
    video_dir_prefix = VTG[args.dataset]["video_folder"]
    try:
        with open(data_file) as f:
            all_data = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(all_data)} samples from {data_file}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return

    if world_size > 1:
        mp.spawn(
            evaluate_worker,
            args=(world_size, all_data, args.model_path, video_dir_prefix, args.output_dir, PROMPTS[args.prompt_type], args.interleave_timestamps, args.num_workers, args.batch_size, args.total_pixels, args.fps),
            nprocs=world_size,
            join=True
        )
    else:
        evaluate_worker(0, world_size, all_data, args.model_path, video_dir_prefix, args.output_dir, PROMPTS[args.prompt_type], args.interleave_timestamps, args.num_workers, args.batch_size, args.total_pixels, args.fps)

    print("\nAll worker processes finished.")
    aggregate_and_calculate(args.output_dir, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=VTG.keys())
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="cot", choices=PROMPTS.keys())
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers for data loading")
    parser.add_argument("--batch_size", type=int, default=16, help="vLLM batch size")
    parser.add_argument("--total_pixels", type=int, default=3584*28*28, help="Total pixels")
    parser.add_argument("--fps", type=int, default=2, help="Video FPS")
    parser.add_argument("--interleave_timestamps", action="store_true")

    args = parser.parse_args()

    assert args.dataset in VTG and args.prompt_type in PROMPTS
    main(args)