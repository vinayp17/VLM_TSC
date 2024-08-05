#!/usr/bin/env python

def generate_llava_finetune_command( *,
                                     llava_root,
                                     train_file,
                                     dataset_dir,
                                     checkpoint_name,
                                     num_epochs,
                                     context_length,
                                     validation_file
                                   ):
    llava_cmd = f"""
deepspeed {llava_root}/llava/train/train_mem.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed {llava_root}/scripts/zero3.json \
--model_name_or_path liuhaotian/llava-v1.5-7b \
--version v1 \
--data_path {train_file} \
--image_folder {dataset_dir} \
--vision_tower openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir {llava_root}/checkpoints/llava-v1.5-7b-task-lora_{checkpoint_name} \
--num_train_epochs {num_epochs} \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy no \
--save_strategy epoch \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length {context_length} \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to tensorboard \
--validation_data_path {validation_file} \
--evaluation_strategy steps \
--report_to wandb"""
    return llava_cmd
