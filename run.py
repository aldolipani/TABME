from utils.train import train, get_validation_result
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

#----------------------- Hyperparameter searching -------------------------------#
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('path_train', type=str, help="path to folder containing train set (.tsv and .jpg files), should be './data/train'")
    parser.add_argument('path_val', type=str, help="path to folder containing validation set (.tsv and .jpg files), should be './data/val'")
    
    parser.add_argument('-l', '--lr', type=float, required=False, default=None, help="learning rate to run separated ex. --lr 1e-4,  default=best from './output'")
    parser.add_argument('-n', '--num_hidden_layer', type=int, required=False, default=None, help="numbers of hidden layer to run ex. --num_hidden_layer 5, default=best from './output'")
    parser.add_argument('-o', '--output', type=str, required=False, default='./output', help="path to the outputs from validation, should be a folder containing multiples csv files, default='./output'")
    parser.add_argument('-c', '--cache', type=str, required=False, default='./cache', help="path to where to save preprocessed cache, default='./cache'")
    parser.add_argument('-a', '--ablation', type=str, required=False, help="optional, specify ablation choose from 'resnet' or 'layoutlm', default: no ablation")
    parser.add_argument('-f', '--freeze', default=False, action="store_true", help="freeze layoutlm and resnet part")
    parser.add_argument('-w', '--load_weights', type=str, required=False, default=None, help="path to initialised weights, default=None (will use pretrained weights)")

    args = parser.parse_args()
    
    learning_rate = args.lr
    num_hidden_layer = args.num_hidden_layer
    path_data_train = Path(args.path_train)
    path_data_val = Path(args.path_val)
    ablation = args.ablation
    path_cache_folder = Path(args.cache)
    path_runs = Path(args.output)
    freeze = args.freeze
    load_weights = args.load_weights

    # check paths
    assert path_data_train.exists(), f"{path_data_train} not exist"
    assert path_data_val.exists(), f"{path_data_val} not exist"
    if not path_cache_folder.exists(): path_cache_folder.mkdir(parents=True, exist_ok=True)
    if not path_runs.exists(): path_runs.mkdir(parents=True, exist_ok=True)

    # find best learning_rate, num_hidden_layer if the values are not input
    if (learning_rate is None) or (num_hidden_layer is None):
        learning_rate, num_hidden_layer = get_validation_result(path_runs=path_runs)

    # list of features
    spacing = int((1280-2)/(num_hidden_layer+1))
    num_hidden_features = [2+spacing*(n+1) for n in reversed(range(num_hidden_layer))]

    path_model_weight = path_runs/'model_weights'
    path_model_weight.mkdir(parents=True, exist_ok=True)

    min_val_loss, min_val_f1 = train(learning_rate, num_hidden_features, path_data_train, path_data_val, path_cache_folder, path_model_weight, ablation=ablation, path_load_model_weights=load_weights, save_model=True, freeze=freeze)
    
    print(f'Results: val_loss={min_val_loss}, val_f1={min_val_f1} saved in {path_model_weight}')