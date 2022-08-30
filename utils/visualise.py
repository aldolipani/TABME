import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.patches as patches

def get_ocr_stat(path_ocr_folder):
    '''
    Get statistics from OCR scan
    '''
    df = pd.DataFrame()
    print(f'Reading OCR results within {path_ocr_folder}')
    for path in tqdm(list(Path(path_ocr_folder).glob('**/*.tsv'))):
        page_id = path.stem
        ocr = pd.read_csv(path, sep='\t')
        df.loc[page_id, 'ocr_boxes'] = len(ocr)
        df.loc[page_id, 'avg_conf'] = ocr['conf'].mean()
    print("=====STATISTICS=====")
    print(f"zero boxes detected in {len(df[df.ocr_boxes==0])} pages ({len(df[df.ocr_boxes==0])/len(df):.0%})")
    print(f"{df['ocr_boxes'].mean():.2f} boxes per page detected")
    print(f"total average confidence : {df['avg_conf'].mean():.2f}")
    return df

def display_ocr(stem, path_data):
    '''
    Display OCR results
    stem ex ffbb0018-1, ffbb0001
    '''
    path_ocr = next(iter(Path(path_data).glob(f'**/{stem}.tsv')))
    path_img = path_ocr.parent/f"{stem}.jpg"
    
    df_ocr = pd.read_csv(path_ocr, sep='\t')
    plt.figure(figsize=(20,20))
    plt.imshow(Image.open(path_img), alpha=1, cmap='gray')

    print("boxes displayed in red when confidence>50%, blue otherwise")

    for i in range(len(df_ocr)):
        (x, y, w, h) = (df_ocr['left'][i], df_ocr['top'][i], df_ocr['width'][i], df_ocr['height'][i])
        rect = patches.Rectangle((x, y), w, h, linewidth=0.3, edgecolor='r' if df_ocr['conf'][i]>50 else 'b', facecolor='none')
        plt.gca().add_patch(rect)
        plt.gca().text(x, y, df_ocr['text'][i], color='blue', weight='bold')
    plt.title(f"{stem}")
    plt.axis("off")
    plt.show()

def display_folder(folder_id, df, path_img_folder):

    df_folder = df[df.folder_id==folder_id] 
    num_pages = len(df_folder)
    plt.figure(figsize=(25,(num_pages//5+1)*5.2))
    for i, row in enumerate(df_folder.iterrows()):
        # idx = starting_index+i
        row = row[1]
        id = row['id']
        page_num = row['page_num']
        path_img = next(iter(Path(path_img_folder).glob(f"**/{id}-{page_num}.*"))) # find the matching first element 
        ax = plt.subplot(num_pages//5+1,5,i+1)
        ax.imshow(Image.open(path_img))

        ax.set_title(f'{id}-{page_num}')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__=="__main__":
    path_data_train = "./data/train"
    display_ocr(f'ffbc0228-1', path_data_train)