from utils.train import train
from utils.evaluation import predict
from utils.dataset import TABME
from utils.config import config

import argparse
import pandas as pd
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get prediction from the model')

    parser.add_argument('-d', '--data', type=str, help="path to output folder, should be ./data/test")
    parser.add_argument('-m', '--model', type=str, help="path to model folder, should be inside ./output/")
    parser.add_argument('--csv', type=str, help="path to csv specifying folders, should be inside ./predictions")

    parser.add_argument('-c', '--cache', type=str, required=False, default='./cache', help="path to where to save preprocessed cache, default='./cache'")
    parser.add_argument('-a', '--ablation', type=str, required=False, help="optional, specify ablation choose from 'resnet' or 'layoutlm', default: no ablation")
    parser.add_argument('-n', '--num_hidden_layers', type=int, required=False, help="optional, specify num_hidden_layers in integer")
        
    args = parser.parse_args()

    path_data = args.data
    path_model_folder = args.model
    path_csv = args.csv
    ablation = args.ablation
    path_cache_folder = args.cache
    num_hidden_layers = args.num_hidden_layers

    if num_hidden_layers:
        spacing = int((1280-2)/(num_hidden_layers+1))
        num_hidden_features = [2+spacing*(n+1) for n in reversed(range(num_hidden_layers))]
    else:
        num_hidden_features = None
    
    df = predict(path_data, path_model_folder, path_csv, batch_size=64, path_cache_folder=path_cache_folder, num_hidden_features=num_hidden_features, ablation=ablation)
