import os, logging
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from utils.dataset import TABME
from utils.model import PDFSegmentationModel
from utils.config import config

def get_validation_result(path_runs = './output'):
    '''
    Merge the results from validation and return the best learning_rate and num_hidden_layer
    Args:
        path_runs: path to the outputs from validation, should be a folder containing multiples csv files, default='./output'
    '''
    path_runs = Path(path_runs)
    
    if (path_runs/'results_summary.csv').exists():
        df_summary = pd.read_csv(path_runs/'results_summary.csv', index_col=['learning_rate','num_hidden_layer'])
    else:
        ls_df = []

        for path_csv in path_runs.glob("*.csv"):
            if path_csv.stem != 'results_summary':
                df = pd.read_csv(path_csv)
                df['file_name'] = path_csv
                ls_df.append(df)
        df_all_results = pd.concat(ls_df, ignore_index=True)

        # calculate mean and std
        df_idxmin = df_all_results.groupby(['learning_rate', 'num_hidden_layer'])['val_loss'].idxmin()
        df_min = df_all_results.loc[df_idxmin].set_index(['learning_rate', 'num_hidden_layer'])
        df_std = df_all_results.groupby(['learning_rate', 'num_hidden_layer']).std()
        df_std = df_std.rename(columns={'val_loss': 'val_loss_std', 'val_f1': 'val_f1_std'})

        df_summary = pd.concat([df_min, df_std], axis=1)
        df_summary.to_csv(path_runs/"results_summary.csv")

    learning_rate, num_hidden_layer = df_summary['val_loss'].idxmin()
    best_val_loss = df_summary.loc[(learning_rate, num_hidden_layer),'val_loss']
    best_val_f1 = df_summary.loc[(learning_rate, num_hidden_layer),'val_f1']

    print(f"Best result val_loss={best_val_loss:.2f}, val_f1={best_val_f1:.2f}, found in lr={learning_rate}, num_hidden_layer={num_hidden_layer}")
    return learning_rate, num_hidden_layer

def log_and_print(message, logger):
    print(message)
    if logger:
        logger.debug(message)

def df_shuffle_id(df):
    '''
    Shuffle IDs while retaining the same page order within the document
    '''
    df_id = df.value_counts(subset='doc_id').to_frame(name='num_pages')
    # reassign document
    df_id = df_id.sample(frac=1)
    df_id['order'] = range(len(df_id))
    df['order'] = df.doc_id.apply(lambda id: df_id.loc[id, 'order'])
    # must choose a stable sort to retain the page order within the same document
    df = df.sort_values(by='order', kind='mergesort')
    # reassign page order
    df['order'] = range(len(df))
    return df

def evaluate(model, dataloader, criterion):
    device_str = 'cuda'if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    model = model.to(device)

    running_loss = 0.0
    running_f1 = 0.0
    model.eval()
    # Iterate over data.
    for dataset in dataloader:
        inputs, image, labels, _ = dataset

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
        running_loss += criterion(outputs, labels.long()).item()
        running_f1 += f1_score(labels.cpu(), torch.argmax(outputs, dim=1).cpu(), average='binary')
           
    loss, f1 = running_loss/len(dataloader), running_f1/len(dataloader)
    model.validation_loss = loss
    model.validation_f1 = f1
    return loss, f1

def run(model, train_dataset, val_dataset, save_model, num_epochs, batch_size, path_runs, logger, learning_rate=5e-5, num_batch_eval=500, patience=4):
    '''
        frac_stagnant(=0.02): stop if the new val loss is within (1+-frac_stagnant)*previous val loss
        patience(=4): stop if the number of consecutive rise in validation loss exceeds patience
    '''

    # model
    torch.cuda.empty_cache()

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    log_and_print(f'Device: {device_str}', logger)
    model = model.to(device)

    # train/validation split
    log_and_print(f"# Train dataset = {len(train_dataset)}, # Validation dataset = {len(val_dataset)}", logger)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Get the loss and f1 before train
    val_loss, val_f1 = evaluate(model, dataloader=val_dataloader, criterion=criterion)
    # val_dataset.save_cache()
    log_and_print(f'Initial val_loss {val_loss:.3f}, val_f1 {val_f1:.3f}', logger)

    min_val_loss = val_loss
    min_val_f1 = val_f1
    
    best_model = model
    best_epoch = 0 # epoch with the best validation loss
    num_consecutive_rise = 0
    
    batches = [0]
    all_train_loss = [0]
    all_train_f1 = [0]
    all_val_loss = [val_loss]
    all_val_f1 = [val_f1]

    for epoch in range(num_epochs):
        df_train = df_shuffle_id(train_dataset.df)
        batch_sampler = torch.utils.data.BatchSampler(df_train.index.to_numpy(), batch_size=batch_size, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)        

        running_loss = 0.0
        running_f1 = 0.0
        
        log_and_print(f"> Epoch {epoch+1}/{num_epochs}: num_consecutive_rise {num_consecutive_rise}/{patience}", logger)
        
        model.train()
        
        # Iterate over data.
        for batch, dataset in enumerate(tqdm(train_dataloader, desc=f"   ", position=0, leave=True)):            
            inputs, image, labels, _ = dataset
                
            input_ids = inputs['input_ids'].to(device)
            bbox = inputs['bbox'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            labels = labels.to(device)
            image = image.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # Get model outputs and calculate loss
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, image=image)

            # the first index is always the cut point
            labels[0]=1
            loss = criterion(outputs, labels.long())

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_f1 += f1_score(labels.cpu(), torch.argmax(outputs, dim=1).cpu(), average='binary')

            # Evaluate: batch 499, 999, ..., OR last batch
            if (batch+1)%num_batch_eval==0 or batch+1==len(train_dataloader): # batch 499, 999, ...
                n = num_batch_eval if (batch+1)%num_batch_eval==0 else len(train_dataloader)%num_batch_eval
                loss, f1 = running_loss/n, running_f1/n
                batches += [epoch*len(train_dataloader) + batch]
                all_train_loss += [loss]
                all_train_f1 += [f1]
                
                # Validation
                val_loss, val_f1 = evaluate(model, dataloader=val_dataloader, criterion=criterion)
                all_val_loss += [val_loss]
                all_val_f1 += [val_f1]
                
                if save_model:
                    results = pd.DataFrame({'tr_loss': all_train_loss, 'tr_f1': all_train_f1, 'val_loss': all_val_loss, 'val_f1': all_val_f1})
                    results.to_csv(Path(path_runs)/'results.csv', index=False)

                # Save the model with mininum validation loss
                if val_loss<min_val_loss:
                    log_and_print(f" val_loss improved from {min_val_loss:.3f} to {val_loss:.3f} (f1 {min_val_f1:.3f} -> {val_f1:.3f})", logger)
                    
                    num_consecutive_rise = 0
                    min_val_loss = val_loss
                    min_val_f1 = val_f1
                    best_model = model
                    best_epoch = epoch

                    if save_model:
                        (Path(path_runs)/"best").mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), Path(path_runs)/'best'/'model_weights.pt')

                # Early stopping: stop if the validation loss rises for more than {patience} times
                else:
                    num_consecutive_rise += 1
                    if num_consecutive_rise>patience or val_f1<0.01: 
                        log_and_print(f"   Early stopping at Batch {epoch+1}.{batch} (best: {best_epoch+1})", logger)
                        log_and_print(f"   Best epoch {best_epoch+1}: val_loss={min_val_loss:.3f}, val_f1={min_val_f1:.3f}", logger)
                        
                        if save_model:
                            (Path(path_runs)/"last").mkdir(parents=True, exist_ok=True)
                            torch.save(model.state_dict(), Path(path_runs)/'last'/'model_weights.pt')
                            model=best_model
                        return min_val_loss, min_val_f1
                
                # Reset values
                running_loss = 0.0
                running_f1 = 0.0
                model.train()   
        
        # train_dataset.save_cache()

    if save_model:    
        os.makedirs(Path(path_runs)/"last", exist_ok=True)
        torch.save(model.state_dict(), Path(path_runs)/'last'/'model_weights.pt')
        model=best_model
    return min_val_loss, min_val_f1

def create_folder_and_logger(path_runs, save_model=True):
    if save_model:
        Path(path_runs).mkdir(parents=True, exist_ok=True)
        # logging the process
        logger=logging.getLogger(__name__)
        file_handler = logging.FileHandler(path_runs/'info.log')
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        return path_runs, logger
    else:
        return None, None

def train(learning_rate, num_hidden_features, path_data_train, path_data_val, path_cache_folder, path_runs, ablation=None, path_load_model_weights=None, save_model=True, freeze=False):

    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    # Model
    input_img_size = config['input_img_size']
    max_seq_length = config['max_seq_length']

    # Training
    num_epochs = config['num_epochs']
    batch_size =  config['batch_size']
    patience=config['patience']
    num_batch_eval = config['num_batch_eval']


    # dataset
    train_dataset = TABME(path_data=path_data_train, path_cache_folder=path_cache_folder, ablation=ablation, max_seq_length=max_seq_length, input_img_size=input_img_size)
    val_dataset = TABME(path_data=path_data_val, path_cache_folder=path_cache_folder, ablation=ablation, max_seq_length=max_seq_length, input_img_size=input_img_size)

    path_runs_current, logger = create_folder_and_logger(Path(path_runs)/f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_l_{learning_rate}_n_{len(num_hidden_features)}_a_{ablation}", save_model=save_model)
    log_and_print(f">> Running with learning_rate={learning_rate}, num_hidden_features={num_hidden_features}, ablation={ablation}", logger)

    # save config
    if save_model:
        model_config = {}
        model_config["learning_rate"] = learning_rate
        model_config["num_hidden_features"] = num_hidden_features
        model_config["ablation"] = ablation
        json.dump(model_config, open(Path(path_runs_current)/"model_config.json", "w"))

    # model
    model = PDFSegmentationModel(num_hidden_features=num_hidden_features, pretrained_weights=True, freeze=freeze)
    if path_load_model_weights!=None:
        # model.load_state_dict(torch.load(path_load_model_weights, map_location=device))
        # import only layoutlm and resnet weights
        all_weights = torch.load(path_load_model_weights, map_location=device)
        layoutlm_weights = OrderedDict()
        resnet_weights = OrderedDict()
        for k in all_weights.keys():
            l = k.split('.')
            if l[0] == 'layoutlm':
                layoutlm_key = '.'.join(l[1:])
                layoutlm_weights[layoutlm_key] = all_weights[k]
            elif l[0] == 'resnet':
                resnet_key = '.'.join(l[1:])
                resnet_weights[resnet_key] = all_weights[k]
        model.layoutlm.load_state_dict(layoutlm_weights)
        model.resnet.load_state_dict(resnet_weights)
        log_and_print(f">> Loaded weights from {path_load_model_weights}", logger)

    # train
    min_val_loss, min_val_f1 = run(model, train_dataset, val_dataset, num_epochs=num_epochs, save_model=save_model, batch_size=batch_size, logger=logger, learning_rate=learning_rate, path_runs=path_runs_current, num_batch_eval=num_batch_eval, patience=patience)
    return min_val_loss, min_val_f1