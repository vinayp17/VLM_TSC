from aeon.datasets import load_classification

import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
import argparse


def generate_llava_qa_entry(question, target, image_filename_id, image_filename_path):

    entry = {
        "id": image_filename_id,
        "image": image_filename_path,
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": target
            }
        ]
    }
    return entry

def generate_llava_eval_entry(question, target, image_filename_id, image_filename_path):

    entry = {
        "question_id": image_filename_id,
        "image": image_filename_path,
        "text": question,
        "output": target,
    }
    return entry

def generate_qwen_vl_entry(question, target, image_filename_id, image_filename_path):

    entry = {
      "id": image_filename_id,
      "conversations": [
        {
          "from": "user",
          "value": f"Picture 1: <img>{image_filename_path}</img>\n{question}"
        },
        {
          "from": "assistant",
          "value": target
        }
      ]
    }
    return entry

def generate_graph(X, image_filename_path, style="line"):

    plt.figure(figsize=(4,4))

    if style == "line":
      plt.plot(X)
    elif style == "area":
      plt.stackplot(range(0, len(X)), X)

    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='y', length=0)
    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()

    # Hide the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(image_filename_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_llava_data(X, y, image_path, data_path. split, model):

    entries = []

    for n in range(0, len(y)):
        image_filename_id = f"image_train_{n}"
        image_filename_path = f"{image_path}/image_train_{n}.png"
        combined_signal_string = [f'{np.round(i, 6):.6f}' for i in X[n][0].tolist()]
        question = f"Which class is the following signal from? {combined_signal_string}".replace("\'", "")
        target = str(y[n])

        if split == "train":
            if model == "llava":
                entries.append(generate_llava_qa_entry(question, target, image_filename_id, image_filename_path))
            else:
                entries.append(generate_qwen_vl_entry(question, target, image_filename_id, image_filename_path))
        else:
            if model == "llava":
                entries.append(generate_llava_eval_entry(question, target, image_filename_id, image_filename_path))
            else:
                entries.append(generate_qwen_vl_entry(question, target, image_filename_id, image_filename_path))

        generate_graph(X[n][0], image_filename_path)

    if split == "train":
        json_output = json.dumps(entries, indent=2)
        # Write to a file
        with open(f'{data_path}/train.json', 'w') as file:
            file.write(json_output)

    else:
        # Write to a file
        with open(f'{data_path}/test.jsonl', 'w') as file:
            for dictionary in entries:
                json_string = json.dumps(dictionary)
                file.write(json_string + '\n')


def generate_ucr_data(dataset, image_path, data_path, model, extract_path='~/Downloads/UCRArchive_2018/'):
    X, y, meta = load_classification(dataset, return_metadata=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    generate_llava_data(X_train, y_train, image_path, data_path, "train", model)
    generate_llava_data(X_test, y_test, image_path, data_path, "test", model)

def main():
    parser = argparse.ArgumentParser(description='Generate UCR Data')
    parser.add_argument('dataset', type=str, help='Name of the dataset')
    parser.add_argument('image_path', type=str, help='Path to save images')
    parser.add_argument('data_path', type=str, help='Path to save data')
    parser.add_argument('model', type=str, help='Model to use')
    parser.add_argument('--extract_path', type=str, default='~/Downloads/UCRArchive_2018/', help='Path to extract data')

    args = parser.parse_args()

    generate_ucr_data(args.dataset, args.image_path, args.data_path, args.model, args.extract_path)

if __name__ == '__main__':
    main()







