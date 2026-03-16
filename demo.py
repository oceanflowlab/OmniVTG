import gradio as gr
import re
import logging
import torch
import argparse
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.ERROR)

TOTAL_VIDEO_TOKENS = 3584
MAX_FRAMES = 768
FPS = 2
PATCH_SIZE = 14

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
SYSTEM_MESSAGE = "You are a helpful assistant."

PROMPT = """You are given a video as a sequence of interleaved timestamps and frames. Locate the precise timestamps for the event: "{query}".

Use a coarse-to-fine reasoning: first state the broad segment where related content may occur, then refine to localize the secific query. Every reasoning step must include timestamps in the form xx.x seconds.

Wrap the step-by-step reasoning in <think>...</think>. After that, output only the final answer in this exact format inside <answer>...</answer>: From start_time seconds to end_time seconds
For example:
<think>
For the query "a woman opens the door", I find that the woman appears in the video from 12.5 seconds to 20.0 seconds. Zooming in further, she opens the door from 14.2 seconds to 14.7 seconds.
</think>
<answer>
From 14.2 seconds to 14.7 seconds
</answer>"""

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
            text = text.replace(DEFAULT_VIDEO_TOKEN, video_placeholder, 1)
        index += 1

    text = text.replace("<|placeholder|>", DEFAULT_VIDEO_TOKEN)
    return text, curr_timestamp

def prepare_inputs(prompt, video_path, processor, total_pixels, fps):
    messages = [{
        "role": "user", 
        "content": [{
            "type": "video", 
            "video": video_path,
            "min_pixels": 16 * 28 * 28,
            "total_pixels": total_pixels,
            "max_frames": MAX_FRAMES,
            "fps": fps,
        }]
    }]
    _, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
    video_datas, video_metadatas = zip(*videos)
    video_datas = list(video_datas)
    video_metadatas = list(video_metadatas)

    all_input_ids = []

    if len(SYSTEM_MESSAGE) > 0:
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
        all_input_ids.append(system_message_input_ids.squeeze(0))

    inputs = processor(text="", images=None, videos=video_datas, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
    second_per_grid_ts = inputs["second_per_grid_ts"]
    pixel_values_videos = inputs["pixel_values_videos"]
    video_grid_thw = inputs["video_grid_thw"]

    user_input = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN + prompt
    user_input = f"{DEFAULT_IM_START_TOKEN}user\n{user_input}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
    
    user_input, cur_timestamps = _interleave_timestamps(user_input, video_metadatas, video_grid_thw)
        
    prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

    all_input_ids.append(prompt_input_ids.squeeze(0))
    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)

    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1
    second_per_grid_ts = [0] * len(video_grid_thw)

    mm_data = {
        "video_embeds": None,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts
    }

    return input_ids.tolist(), mm_data

def process_output(output_text):
    think_content = re.sub(r'<think>(.*?)</think>', r'**Thought:** \1\n', output_text, flags=re.DOTALL)
    final_output = re.sub(r'<answer>(.*?)</answer>', r'**Answer:** \1\n', think_content, flags=re.DOTALL)
    return final_output

def create_demo(processor, llm):
    def predict(video_file_path, user_input, chat_history_state, display):
        if video_file_path is None:
            gr.Warning("Please upload a video first.")
            return chat_history_state, display

        if not user_input:
            gr.Warning("Please enter your query.")
            return chat_history_state, display

        formatted_query = PROMPT.format(query=user_input)

        try:
            prompt_token_ids, mm_data = prepare_inputs(
                prompt=formatted_query,
                video_path=video_file_path,
                processor=processor,
                total_pixels=TOTAL_VIDEO_TOKENS * (PATCH_SIZE * 2) ** 2,
                fps=FPS
            )

            prompt_data = {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"video": mm_data}
            }

            sampling_params = SamplingParams(
                repetition_penalty=1.05, 
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                stop_token_ids=[151645, 151643], 
                max_tokens=1024,
                include_stop_str_in_output=False,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )

            outputs = llm.generate(
                prompts=[prompt_data],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            bot_response = outputs[0].outputs[0].text.strip()

        except Exception as e:
            gr.Error(f"Error during processing or model inference: {e}")
            print(f"Error in execution: {e}")
            return chat_history_state, display

        chat_history_state.append((user_input, bot_response))
        display.append({"role": "user", "content": user_input})
        display.append({"role": "assistant", "content": process_output(bot_response)})
        
        return chat_history_state, display

    def clear_all(video_file):
        return None, [], []

    with gr.Blocks() as demo:
        gr.Markdown("# OmniVTG Demo")

        chat_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                video_file = gr.Video(label="Upload Video")

            with gr.Column(scale=2):
                chatbot_display = gr.Chatbot(label="Messages", height=500, allow_tags=True)
                
                with gr.Row():
                    user_textbox = gr.Textbox(
                        label="Query",
                        placeholder="a woman opens the door",
                        scale=4
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear")
        
        gr.Examples(
            examples=[
                ["assets/v_UpaTadKoo80.mp4", "A woman models a strapless A-line wedding dress with a ruched corset bodice and a soft, flowing skirt."],
                ["assets/v_4OX1v9ceXj8.mp4", "A night view shows a dragon-shaped bridge breathing fire over a river, with crowds watching from the riverbank."],
                ["assets/BV1A1RiYnErJ.mp4", "The baby and the pug play together inside a barn, with the scene transitioning from daytime to nighttime."]
            ],
            inputs=[video_file, user_textbox],
            label="Example Queries"
        )

        submit_btn.click(
            fn=predict,
            inputs=[video_file, user_textbox, chat_history_state, chatbot_display],
            outputs=[chat_history_state, chatbot_display]
        )
        
        user_textbox.submit(
            fn=predict,
            inputs=[video_file, user_textbox, chat_history_state, chatbot_display],
            outputs=[chat_history_state, chatbot_display]
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[video_file], 
            outputs=[video_file, chatbot_display, chat_history_state]
        )
        
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OmniVTG Gradio Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    print(f"Loading processor from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading vLLM model from: {args.model} ...")
    llm = LLM(
        model=args.model,
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
    print("Model and Processor loaded. Gradio is starting...")

    demo = create_demo(processor, llm)
    demo.launch(share=True, debug=True)