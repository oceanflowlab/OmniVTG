set -x
ENGINE=${1:-vllm}
exp_id=Qwen2.5-VL-7B-Instruct-GRPO
output_dir=outputs/${exp_id}
mkdir -p $output_dir

MODEL_NAME="Qwen2.5-VL-7B-Instruct-CoT"

# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export VLLM_ASCEND_ENABLE_NZ=0
export TENSORBOARD_DIR=${output_dir}/tensorboard_log

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.custom_cls.path=src/dataset/grpo_dataset.py \
    data.custom_cls.name=VTGGRPODataset \
    data.train_files=data/OmniVTG/train_rl.json \
    data.val_files=data/OmniVTG/tvgbench.json \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=9940 \
    data.max_response_length=300 \
    data.truncation='error' \
    data.dataloader_num_workers=64 \
    +data.interleave_timestamps=true \
    +data.prompt_type=cot \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt="{image: 0, video: 768, audio: 0}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs="{min_pixels: 784, max_pixels: 6272}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="console" \
    trainer.project_name='omnivtg' \
    trainer.experiment_name=$exp_id \
    trainer.default_local_dir="${output_dir}/checkpoints" \
    trainer.rollout_data_dir="${output_dir}/rollout_data" \
    trainer.validation_data_dir="${output_dir}/validation_data" \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 \
    trainer.device=npu \
    custom_reward_function.path=./src/rewards.py \
    custom_reward_function.name=compute_vtg_score
