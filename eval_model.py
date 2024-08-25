import os

def one_shot( answers ):
    import pandas as pd
    import numpy as np
    ans_df = pd.read_json(answers, lines=True)
    print(np.mean(ans_df['answer'] == ans_df['ground_truth']))

def eval_performance(vlm_root, llava_root, scenario ):
    model_vqa_script = f"{vlm_root}/model_vqa.py"
    llava_data_dir = f"{llava_root}/playground/new_data/"
    llava_dataset_dir = f"{llava_data_dir}/{scenario}"
    assert os.path.exists(model_vqa_script)
    checkpoint = f"{llava_root}/checkpoints/llava-v1.5-7b-task-lora_{scenario}/"
    assert os.path.exists(checkpoint)
    question_file = f"{llava_dataset_dir}/test.json"
    assert os.path.exists( question_file )
    image_folder = f"{llava_dataset_dir}/images"
    assert os.path.exists( image_folder )
    answers = f"{llava_dataset_dir}/answer.json"
    cmd = f"python {model_vqa_script} --model-path {checkpoint} --model-base liuhaotian/llava-v1.5-7b --question-file {question_file} --image-folder {image_folder} --answers-file {answers}"
    print( cmd )
    os.system( cmd )

if __name__ == "__main__":
   eval_performance("/code/VLM-TSC", "/code/LLaVA" , "CinCECGTorso_downsample_0_round_2")
   one_shot("/code/LLaVA/playground/new_data/CinCECGTorso_downsample_0_round_2/answer.json")
