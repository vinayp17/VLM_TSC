import argparse
import pandas as pd
import os
import json
import shlex
import subprocess

def launch_experiments(
    vlm_root,
    llava_root,
    scenario,
    config,
    start_index,
    end_index
):
    assert os.path.exists(vlm_root)
    assert os.path.exists(llava_root)

    #Make sure chosen scenario belongs to config
    assert os.path.exists(config)
    with open(config, "r") as f:
        config_json = json.load(f)
    possible_scenarios = list(config_json.keys())
    assert scenario in possible_scenarios

    #Load all possible experiments to run
    ucr_results_csv = os.path.join( vlm_root, "metadata/ucr_results.csv" )
    assert os.path.exists(ucr_results_csv)
    ucr_results = pd.read_csv(ucr_results_csv)

    #Determine start and end index
    if start_index == None and end_index == None:
        start_index = 0
        end_index = ucr_results.shape[0] - 1
    else:
        assert start_index is not None
        assert end_index is not None
        assert end_index < ucr_results.shape[0]

    datasets = ucr_results["dataset"].tolist()
    datasets = datasets[start_index:end_index]
    for dataset in datasets:
        print(f"Running experiment for dataset:{dataset}")
        train_file = f"{vlm_root}/train.py"
        assert os.path.exists(train_file)
        round_to = config_json[scenario]["round_to"]
        num_epochs = config_json[scenario]["num_epochs"]
        context_length = config_json[scenario]["context_length"]
        data_repr = config_json[scenario]["data_repr"]
        train_cmd = f"python {train_file} --round_to {round_to} --dataset {dataset} --vlm-root {vlm_root} --llava-root {llava_root} --num-epochs {num_epochs} --context-length {context_length} --data-repr {data_repr} --scenario {scenario}"
        print(train_cmd)
        command_list = shlex.split(train_cmd)
        env = os.environ.copy()
        try:
            subprocess.run(command_list, env=env, check=True)
            print("Command executed success")
        except Exception as e:
            print(f"An error occurred: {e}")   

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--vlm-root", type=str, required=True, help="Root Dir for VLM-TSC git repo")
    argparser.add_argument("--llava-root",type=str,required=True, help="Root Dir for LLaVA git repo")
    argparser.add_argument("--scenario", type=str, required=True, help="What experiment scenario to run for")
    argparser.add_argument("--config", type=str, required=True, help="Json file describing all scenarios")
    argparser.add_argument("--start-index", type=int)
    argparser.add_argument("--end-index", type=int)
    args = argparser.parse_args()

    launch_experiments(
        vlm_root = args.vlm_root,
        llava_root = args.llava_root,
        scenario = args.scenario,
        config = args.config,
        start_index = args.start_index,
        end_index = args.end_index
    )

