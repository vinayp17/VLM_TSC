#!/usr/bin/env python

import argparse
import yaml
import os
import subprocess
import sys

from llava_cmd_generator import generate_llava_finetune_command
from eval_model import eval_performance, one_shot


def finetune( *, round_to, dataset, vlm_root, llava_root, num_epochs, context_length, data_repr, exp_scenario):

    ####### Generate configs #########################
    config_template = os.path.join(vlm_root, "configs/llava_config.yaml")
    assert os.path.exists(config_template)
    with open(config_template, "r") as fh:
        config = yaml.safe_load(fh)
    print(config)
    config["dataset"]["name"] = f'{dataset}'

    llava_data_dir = f"{llava_root}/playground/new_data/"
    os.system(f"mkdir -p {llava_data_dir}")
    scenario = f"{dataset}_{exp_scenario}"
    llava_dataset_dir = f"{llava_data_dir}/{scenario}"
    os.system(f"mkdir -p {llava_dataset_dir}")
    config["data_path"] = llava_dataset_dir

    llava_image_dir = f"{llava_dataset_dir}/images"
    os.system(f"mkdir -p {llava_image_dir}")
    config["image_path"] = f"{llava_image_dir}"
    config["options"]["round_to"] = round_to
    config["options"]["downsample_to"] = None
    config["options"]["context_length"] = context_length
    config["options"]["data_repr"] = data_repr

    config_filename = f"{vlm_root}/configs/llava_config_{scenario}.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f)
    ###### Run dataset.py to generate data ##################
    dataset_file = f"{vlm_root}/dataset.py"
    assert os.path.exists(dataset_file)
    #Check if we already generated data
    train_file = f"{llava_dataset_dir}/train.json"
    if not os.path.exists(train_file):
        data_generation_cmd = f"python {dataset_file} {config_filename}"
        cp = subprocess.run(data_generation_cmd.split())
        cp.check_returncode()
    #After this step train_file should exist
    assert(os.path.exists(train_file))

    #### Run llava fine-tuning
    llava_finetune_cmd = generate_llava_finetune_command(
        llava_root = llava_root,
        train_file = train_file,
        dataset_dir = llava_dataset_dir,
        checkpoint_name = f'{scenario}',
        num_epochs = num_epochs,
        context_length = context_length,
        validation_file = f'{llava_dataset_dir}/validation.json'
    )
    print(llava_finetune_cmd)
    cp = subprocess.run(llava_finetune_cmd.split(), capture_output=False)
    cp.check_returncode()

    #Calculate performance
    eval_performance(vlm_root, llava_root, scenario)
    one_shot(dataset, f"{llava_root}/playground/new_data/{scenario}/answer.json")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--round_to", type=int, default=True)
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--vlm-root", type=str, required=True, help="Root Dir for VLM-TSC git repo")
    argparser.add_argument("--llava-root",type=str,required=True, help="Root Dir for LLaVA git repo")
    argparser.add_argument("--num-epochs", type=int, default=2)
    argparser.add_argument("--context-length", type=int, default=2048)
    argparser.add_argument("--data-repr", type=str, choices=["BASELINE", "WITH_RATIONALE", "WITH_SIGNAL_ANALYSIS", "WITH_STATS"], required=True)
    argparser.add_argument("--scenario", type=str, required=True, help="Experiment scenario to run for")
    args = argparser.parse_args()
    finetune(
              round_to=args.round_to,
              dataset=args.dataset,
              vlm_root=args.vlm_root,
              llava_root=args.llava_root,
              num_epochs = args.num_epochs,
              context_length = args.context_length,
              data_repr = args.data_repr,
              exp_scenario = args.scenario
            )
