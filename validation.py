from utils.train import train
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

#----------------------- Hyperparameter searching -------------------------------#
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Validate the model')

    parser.add_argument('path_train', type=str, help="path to folder containing train set (.tsv and .jpg files), should be './data/train'")
    parser.add_argument('path_val', type=str, help="path to folder containing validation set (.tsv and .jpg files), should be './data/val'")
    
    parser.add_argument('-l', '--lr', type=float, nargs='+', help='learning rate(s) to run separated by space ex. --lr 2.5e-5 5e-5 1e-4')
    parser.add_argument('-n', '--num_hidden_layers', type=int, nargs='+', help='numbers of hidden layer(s) to run separated by space ex. -n 0 1 2 3')
    parser.add_argument('-o', '--output', type=str, required=False, default='./output', help="path to where to save the output csv file, default='./output'")
    parser.add_argument('-c', '--cache', type=str, required=False, default='./cache', help="path to where to save preprocessed cache, default='./cache'")
    parser.add_argument('-a', '--ablation', type=str, required=False, help="optional, specify ablation choose from 'resnet' or 'layoutlm', default: no ablation")
    
    args = parser.parse_args()
    
    learning_rates = args.lr
    num_hidden_layers = args.num_hidden_layers
    path_data_train = args.path_train
    path_data_val = args.path_val
    ablation = args.ablation
    path_cache_folder = args.cache
    path_runs = args.output

    # check paths
    assert Path(path_data_train).exists(), f"{path_data_train} not exist"
    assert Path(path_data_val).exists(), f"{path_data_val} not exist"
    if not Path(path_cache_folder).exists(): Path(path_cache_folder).mkdir(parents=True, exist_ok=True)
    if not Path(path_runs).exists(): Path(path_runs).mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(columns=['learning_rate', 'num_hidden_layer', 'num_features', 'val_loss', 'val_f1'])
    path_csv = Path(path_runs)/f"results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

    for learning_rate in learning_rates:
        for num_hidden_layer in num_hidden_layers:
            # list of features
            spacing = int((1280-2)/(num_hidden_layer+1))
            num_hidden_features = [2+spacing*(n+1) for n in reversed(range(num_hidden_layer))]

            min_val_loss, min_val_f1 = train(learning_rate, num_hidden_features, path_data_train, path_data_val, path_cache_folder, path_runs, ablation=None, save_model=False)
            results = results.append({'learning_rate': learning_rate, 'num_hidden_layer': num_hidden_layer, 
                'num_features': num_hidden_features, 'val_loss': min_val_loss, 'val_f1': min_val_f1}, ignore_index=True)

            results.to_csv(path_csv, index=False)

            print("_________________________________________________________________________________")

    best = results.loc[results['val_loss'].idxmin()]
    print(f'Best results: learning_rate={best.learning_rate}, num_hidden_layer={best.num_hidden_layer}, saved in {path_csv}')