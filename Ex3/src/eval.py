import argparse
import os

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from metrics import ValMeter
from Ex3.sheet_03.task1 import build_class_map
import numpy as np
import torch
from model import SingleStageTCN, MultiStageTCN, MultiScaleTCN



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda')


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path",
                        type=str,
                        required=True,
                        help="path to directory of saved predictions")
    args = parser.parse_args()
    ground_truth_path = "../data/groundTruth/"
    file_list = "../data/test.bundle"

    class_map = build_class_map("../data/mapping.txt")

    val_meter = ValMeter()
    list_of_videos = read_file(file_list).split('\n')[:-1]

    #model = SingleStageTCN(2048, len(class_map), 10, 64).to(device)
    #model = MultiStageTCN(2048, len(class_map), num_stages=4).to(device)
    model = MultiScaleTCN(2048, len(class_map), num_layers=10, num_filters=64).to(device)
    model.load_state_dict(torch.load(f'../runs/{args.pred_path}/model.pth'))
    model.eval()

    for vid_name in list_of_videos:
        # Note that if the labels of target or prediction are not integers
        # you should convert them to integers using the './data/mapping.txt' file
        vid_name = vid_name.split('.txt')[0]

        # TODO: read the content of the ground truth frame-wise labels for the current video as a python list of integers
        with open(f'{ground_truth_path}/{vid_name}.txt', 'r') as f:
            labels = f.read().splitlines()
        labels = [class_map[x] for x in labels]



        target = labels.copy()

        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0).to(device) #(1, T)

        # TODO: this variable should contain a python list of the predicted frame-wise labels for the current video

        video_features = np.load(f'../data/features/{vid_name}.npy') #(D, T)
        video_features = torch.from_numpy(video_features).float().T.unsqueeze(0).to(device) #(1, T, D)

        with torch.no_grad():
            outputs = model(video_features) #(1, T, C) or list
            if type(outputs) == list or type(outputs) == tuple:
                final_output = outputs[2] #Select the appropriate output depending on the task
                final_output = final_output.argmax(dim=-1) #(1, T)
            else:
                final_output = outputs.argmax(dim=-1) #(1,T)
        
        final_output = final_output.reshape(-1) #(T)

        prediction = final_output.detach().cpu().reshape(-1).tolist()

        val_meter.update_stats(target=target, prediction=prediction)


    eval_metrics = val_meter.log_stats()
    print("Evaluation metrics:")
    for metric_name in eval_metrics:
        print(f'{metric_name}: {eval_metrics[metric_name]:.5f}')


if __name__ == '__main__':
    main()
