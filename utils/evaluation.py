import random
from tqdm import tqdm
import pandas as pd
import sklearn.metrics

import itertools
from typing import List
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np


def compute(arg) -> int:
    hat_y = arg[0]
    y = arg[1]
    total_swaps = 0
    for hat_y_i, y_i in zip(hat_y, y):
        shared = set(hat_y_i).intersection(set(y_i))
        num_of_swaps_across_documents = abs(len(hat_y_i) - len(shared))
        total_swaps += num_of_swaps_across_documents
    return total_swaps


def num_of_swaps(hat_y: List[List[int]], y: List[List[int]]) -> int:
    """
    This function computes the number of swaps the user has to perform in order to get the result wanted.
    This function is invariant to the permutations of documents.
    :param hat_y: list of pages
    :param y: list of pages
    :return: MNDD
    """
    hat_y_new = [a for a in hat_y if a not in y]
    y_new = [a for a in y if a not in hat_y]
    hat_y, y = hat_y_new, y_new
    if len(y) < len(hat_y):
        hat_y, y = y, hat_y
    num_workers = mp.cpu_count()
    with Pool(num_workers - 1) as p:
        args = ([hat_y, a] for a in itertools.permutations(y))
        min_num_of_swaps = min(p.imap(compute, args))
    return min_num_of_swaps


def one_is_bigger(x, y):
    """
    This provides a quicker way to compute MNDD in the special case where the prediction splits are finer than
    the label splits; or the other way around.
    :param x: the finer one of prediction/label, in binary representation
    :param y: the other one of prediction/label, in binary representation
    return: MNDD
    """
    # x is bigger than y
    x_str = ''.join(str(m) for m in x)
    zeros = x_str.split("1")
    m = [len(zero) + 1 for zero in zeros]
    m = m[1:]
    return sum(m) - max(m)


def generate_y_from_binary_sequence(s):
    """
    :param s: prediction/label in binary representation
    return: prediction/label in list representation
    e.g. [1,0,0,1,0,1,0] -> [[1,2,3], [4,5], [6,7]]
    """
    split_point = [i for i, x in enumerate(s) if x == 1]
    split_point.append(len(s))
    output = []
    for i in range(len(split_point) - 1):
        output.append(list(range(split_point[i] + 1, split_point[i + 1] + 1)))
    return output

def split_and_compute_MNDD(hat_y, y, n: int = 10000) -> int:
    """
    This function computes the number of drag and drops the user has to perform in order to get the result wanted.
    This function is invariant to the permutations of documents.
    :param hat_y: list of pages
    :param y: list of pages
    :param n: sample size, used when computing an approximation, set to -1 if you don't want an approximation
    :return: the number of swaps required
    """
    hat_y = generate_y_from_binary_sequence(hat_y)
    y = generate_y_from_binary_sequence(y)

    def compute(hat_y: List[List[int]], y: List[List[int]]) -> int:
        res = 0
        a, b = (hat_y, y) if len(hat_y) < len(y) else (y, hat_y)
        i = 0
        for a_i in a:
            num_dads_across_documents = 0
            b_i = b[i]
            # remove elements that are not in common
            shared = set(a_i).intersection(set(b_i))
            while len(shared) == 0 and i < len(b) - 1:
                i += 1
                b_i = b[i]
                shared = set(a_i).intersection(set(b_i))
            num_dads_across_documents += abs(len(a_i) - len(shared))
            shared_a_i = [v for v in a_i if v in shared]
            shared_b_i = [v for v in b_i if v in shared]
            # count swaps within a document
            num_dads_within_the_document = 0
            for j, b_i_j in enumerate(shared_b_i):
                num_dads_within_the_document += abs(j - shared_a_i.index(b_i_j))
            res += num_dads_across_documents + (num_dads_within_the_document >> 1)
            if i < len(b) - 1:
                i += 1
        return res

    a, b = (hat_y, y) if len(hat_y) > len(y) else (y, hat_y)
    min_num_dads = compute(a, b)
    if min_num_dads == 0:
        return min_num_dads
    bs = itertools.permutations(b)
    next(bs)
    for i, new_b in enumerate(bs):
        num_dads = compute(a, list(new_b))
        if num_dads < min_num_dads:
            min_num_dads = num_dads
            if min_num_dads == 0:
                break
        if n == i:
            break
    return min_num_dads



def model_evaluation(df):
    """
    This function evaluate the model performances, using F1 score, precision, recall and MNDD
    :param df: dataframe containing model prediction and true label
    :return: dataframe containing F1 score, precision, recall and MNDD for each folder
    """
    df_folders = pd.DataFrame({"folder_id": list(df.folder_id.unique())})
    df_folders = df_folders.set_index("folder_id")
    for folder_id in tqdm(df.folder_id.unique()):
        df_folder = df.loc[df.folder_id == folder_id]
        if df_folder.shape[0]== 0:
            continue
        label = df_folder.label.tolist()
        pred = df_folder.prediction.tolist()
        df_folders.loc[folder_id, 'label'] = str(label)
        df_folders.loc[folder_id, 'model_pred'] = str(pred)
        df_folders.loc[folder_id, 'model_F1'] = sklearn.metrics.f1_score(label, pred, zero_division=0)
        df_folders.loc[folder_id, 'model_precision'] = sklearn.metrics.precision_score(label, pred, zero_division=0)
        df_folders.loc[folder_id, 'model_recall'] = sklearn.metrics.recall_score(label, pred, zero_division=0)
        if pred[0] == 0: pred[0] = 1
        n = split_and_compute_MNDD(pred, label)
        df_folders.loc[folder_id, 'model_MNDD'] = n
    return df_folders


def baseline_models_evaluation(df):
    df_folders = pd.DataFrame({"folder_id": list(df.folder_id.unique())})
    df_folders = df_folders.set_index("folder_id")
    for folder_id in tqdm(df.folder_id.unique()):
        df_folder = df.loc[df.folder_id == folder_id]
        if df_folder.shape[0] == 0:
            continue
        label = df_folder.label.tolist()
        pred = {"onlyfirst": [1] + [0 for _ in range(len(label) - 1)],
                "random": [random.randint(0, 1) for _ in range(len(label))]}
        df_folders.loc[folder_id, 'label'] = str(label)
        for i in ["onlyfirst", "random"]:
            df_folders.loc[folder_id, i + '_pred'] = str(pred[i])
            df_folders.loc[folder_id, i + '_F1'] = sklearn.metrics.f1_score(label, pred[i], zero_division=0)
            df_folders.loc[folder_id, i + '_precision'] = sklearn.metrics.precision_score(label, pred[i], zero_division=0)
            df_folders.loc[folder_id, i + "_recall"] = sklearn.metrics.recall_score(label, pred[i], zero_division=0)
            if pred[i][0] == 0:
                pred[i][0] = 1
            # df_folders.loc[folder_id, i + '_MNDD'] = split_and_compute_MNDD(pred[i], label)
    return df_folders

# ------------------------------------------------------------- #

import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils.dataset import TABME
from utils.model import PDFSegmentationModel
from utils.config import config

def predict(path_data, path_model_folder, path_csv = './predictions/test.csv', batch_size=64, path_cache_folder=None, num_hidden_features=None, ablation=None):
    '''get predictions from the model'''
    
    path_model_folder = Path(path_model_folder)
    path_csv = Path(path_csv)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print("device:", device_str)

    # config
    input_img_size = config['input_img_size']
    max_seq_length = config['max_seq_length']

    test_dataset = TABME(path_data=path_data, path_cache_folder=path_cache_folder, ablation=ablation, max_seq_length=max_seq_length, input_img_size=input_img_size)
    df = pd.read_csv(path_csv)
    print("# test dataset:", len(test_dataset))

    # model
    path_model_config = path_model_folder/"model_config.json"
    if path_model_config.exists():
        model_config = json.load(open(path_model_config, "r"))
        num_hidden_features = model_config["num_hidden_features"] 
        ablation = model_config["ablation"]

    # load model weights
    model = PDFSegmentationModel(num_hidden_features=num_hidden_features, pretrained_weights=False)
    model.load_state_dict(torch.load(path_model_folder/"best"/"model_weights.pt", map_location=device))
    model = model.to(device)
    model.eval()


    for folder_id in tqdm(df.folder_id.unique(), desc="Generating predictions"):
        df_folder = df[df.folder_id==folder_id]
        stems = df_folder.stem.to_list()
        def stem_to_index(stem):
            index = np.where(test_dataset.page_ids==stem)
            return index[0][0]
        indices = map(stem_to_index, stems)

        sampler = torch.utils.data.BatchSampler(indices, batch_size=batch_size, drop_last=False)
        dataloader = DataLoader(test_dataset, batch_sampler=sampler)

        all_labels = []
        all_logits = []
        all_stems = []
      
        for dataset in dataloader:


            inputs, image, labels, stems = dataset

            input_ids = inputs['input_ids'].to(device)
            bbox = inputs['bbox'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            labels = labels.to(device)
            image = image.to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, image=image)
            
            # the first index is always the cut point
            labels[0]=1
            
            all_logits += [outputs]
            all_labels += [labels]
            all_stems += stems
            # break
          
        all_logits = torch.cat(all_logits).detach().cpu().numpy()
        all_labels = torch.cat(all_labels).detach().cpu().numpy()
        all_logits_dict = dict(zip(all_stems, all_logits))

        # save predictions
        for i in range(2):
            df.loc[df_folder.index, f'logits_{i}']=df_folder['stem'].map(lambda stem: all_logits_dict[stem][i])
        df.loc[df_folder.index, 'prediction']=df_folder['stem'].map(lambda stem: np.argmax(all_logits_dict[stem]))

        if ablation == 'resnet':
            df.to_csv(path_csv.parent/f'minus_resnet_predictions_{path_csv.stem}.csv')        
        elif ablation == 'layoutlm':
            df.to_csv(path_csv.parent/f'minus_layoutlm_predictions_{path_csv.stem}.csv')
        else: 
            df.to_csv(path_csv.parent/f'full_model_predictions_{path_csv.stem}.csv')
    
    return df

