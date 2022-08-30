from pathlib import Path
import h5py

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer

from PIL import Image
from torchvision import transforms

class TABME(Dataset):
    '''
    Args:
        path_data: path to folder containing images (.jpg file) and ocr results (.tsv file)
        path_cache: path to cache folder
        max_seq_length: will input only the first {max_seq_length} tokens into the model. Default: 100
        input_img_size: images are rescale into {input_img_size} before inputting into the model
    '''
    def __init__(self, path_data: str, path_cache_folder: str=None, ablation=None, max_seq_length: int =100, input_img_size: tuple=(512,512)):
        self.path_data = Path(path_data)
        self.max_seq_length = max_seq_length
        self.path_cache_folder = path_cache_folder
        self.ablation = ablation
        self.input_img_size = input_img_size
        
        all_path_tsv = list(self.path_data.glob("**/*.tsv"))
        assert len(all_path_tsv)>0, "no data found"
        self.page_ids = np.sort([path.stem for path in all_path_tsv])
        # create dataframe for shuffling while retaining order within the document
        self.df = create_df(self.page_ids)
        
        self.tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
        self.cache = {} # cache the OCR encodings
        if self.path_cache_folder: # self.path_cache_folder != None
            path_cache_folder = Path(self.path_cache_folder)
            path_cache_folder.mkdir(parents=True, exist_ok=True)
            # self.path_cache = path_cache_folder/f"cache_{self.path_data.stem}_{self.max_seq_length}.pt"
            # if self.path_cache.exists(): # cache exists
                # self.cache = torch.load(self.path_cache)
            self.path_cache = path_cache_folder/f"cache_{self.path_data.stem}_{self.max_seq_length}.hdf5"
            if not self.path_cache.exists():
                print("Generating cache:")
                for page_id in tqdm(self.page_ids):
                    self.hf = h5py.File(self.path_cache, 'a')
                    # ocr
                    file_id = page_id.split('-')[0]
                    path_tsv = self.path_data/file_id/f"{page_id}.tsv"
                    input_ids, bbox, attention_mask, token_type_ids = tokenize_from_ocr(path_tsv, self.tokenizer, self.max_seq_length)
                    group = self.hf.create_group(page_id)
                    group.create_dataset('input_ids', data=input_ids, compression="gzip")
                    group.create_dataset('bbox', data=bbox, compression="gzip")
                    group.create_dataset('attention_mask', data=attention_mask, compression="gzip")
                    group.create_dataset('token_type_ids', data=token_type_ids, compression="gzip")
                    # image
                    path_img = self.path_data/file_id/f"{page_id}.jpg"
                    image = image_to_tensor(path_img, input_img_size=self.input_img_size)
                    group.create_dataset('image', data=image, compression="gzip")
                self.hf.close()
            else:
                print(f"Using cache at {self.path_cache}")
            
            if self.ablation:
                self.blank_image = torch.load('cache/blank_img.pt')
                self.blank_ocr = torch.load('cache/blank_ocr.pt')

    def __len__(self):
        return len(self.page_ids)

    def __getitem__(self, idx):
        page_id = self.page_ids[idx]
        self.hf = h5py.File(self.path_cache, 'r')
        # ocr
        input_ids = torch.tensor(np.array(self.hf[page_id+'/input_ids']))
        bbox = torch.tensor(np.array(self.hf[page_id+'/bbox']))
        attention_mask = torch.tensor(np.array(self.hf[page_id+'/attention_mask']))
        token_type_ids = torch.tensor(np.array(self.hf[page_id+'/token_type_ids']))
        ocr_input = {"input_ids": input_ids, 'bbox': bbox, 'attention_mask': attention_mask, "token_type_ids": token_type_ids}
        # image
        image = torch.tensor(np.array(self.hf[page_id+'/image']))

        self.hf.close()

        # get label
        label = get_label(page_id)

        # ablation
        if self.ablation=='resnet':
            image = self.blank_image
        elif self.ablation=='layoutlm':
            ocr_input = self.blank_ocr

        return ocr_input, image, label, page_id
            
def get_label(page_id):
    split = page_id.split('-')
    if len(split)==1: # single page doc
        return 1
    else:
        if int(split[1])==0: # first page
            return 1
        else: # non-first page
            return 0

# Image preprocessing
def image_to_tensor(path_img, input_img_size):
    img = Image.open(path_img)
    preprocess = transforms.Compose([
        transforms.Resize(input_img_size),        
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    return img_tensor

def create_df(page_ids):
    pages = np.array([(id.split('-') if len(id.split('-'))==2 else [id, '0']) for id in page_ids])
    df = pd.DataFrame(pages, columns=['doc_id', 'page_num'])
    return df

#-------------- OCR preprocessing ---------------#

def extend_box(words, boxes, tokenizer):
    token_boxes = []
    tokenizable_words = words
    j = 0 # index for tokenizable words
    for i in range(len(words)):
        word, box = words[i], boxes[i]
        try:
            word_tokens = tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))
            j+=1
        except:
            tokenizable_words = np.delete(tokenizable_words, j)
    
    return tokenizable_words, token_boxes


def tokenize_from_ocr(path_tsv, tokenizer, max_seq_length):
    '''
        Generate layoutlm input vectors from OCR
    '''
    
    df_ocr_page = pd.read_csv(path_tsv, sep='\t')

    # If the there are lots of boxes, select only the boxes with high confidence. Will not filter if the number of boxes is too few.
    df_high_conf = df_ocr_page[df_ocr_page.conf>50]
    if len(df_high_conf)>=max_seq_length:
        df_ocr_page = df_high_conf
        
    words = df_ocr_page['text'].to_numpy()
    heights = df_ocr_page['height'].to_numpy()
    widths = df_ocr_page['width'].to_numpy()
    lefts = df_ocr_page['left'].to_numpy()
    tops = df_ocr_page['top'].to_numpy()
    boxes = np.column_stack((lefts, tops, widths+lefts, heights+tops)) #[[x0, y0, x1, y1], ...]

    words, token_boxes = extend_box(words, boxes, tokenizer)

    # Add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = tokenizer(' '.join(words), return_tensors="pt", padding=True, pad_to_multiple_of=max_seq_length)
    input_ids = torch.squeeze(encoding["input_ids"], 0)
    attention_mask = torch.squeeze(encoding["attention_mask"], 0)
    token_type_ids = torch.squeeze(encoding["token_type_ids"], 0)

    # Pad/cut token_boxes with [0, 0, 0, 0] or cut the elements beyond {max_seq_length}
    pad_box = [0, 0, 0, 0]
    if len(token_boxes)<=max_seq_length:
        token_boxes = torch.tensor(np.array(token_boxes+(max_seq_length -len(token_boxes))*[pad_box]))
    else:
        token_boxes = torch.tensor(np.array(token_boxes[:max_seq_length-1]+[[1000, 1000, 1000, 1000]]))
        input_ids = input_ids[:max_seq_length]
        input_ids[-1] = 102
        attention_mask = attention_mask[:max_seq_length]
        token_type_ids = token_type_ids[:max_seq_length]
    bbox = token_boxes

    return input_ids, bbox, attention_mask, token_type_ids
