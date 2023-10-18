import os
import sys
import tarfile
import time

import numpy as np
import pandas as pd
from packaging import version
from torch.utils.data import Dataset
from tqdm import tqdm
import urllib


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.0**2 * duration)
    percent = count * block_size * 100.0 / total_size

    sys.stdout.write(
        f"\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB "
        f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
    )
    sys.stdout.flush()


def download_dataset():
    source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target = "aclImdb_v1.tar.gz"

    if os.path.exists(target):
        os.remove(target)

    if not os.path.isdir("aclImdb") and not os.path.isfile("aclImdb_v1.tar.gz"):
        urllib.request.urlretrieve(source, target, reporthook)

    if not os.path.isdir("aclImdb"):

        with tarfile.open(target, "r:gz") as tar:
            tar.extractall()


def load_dataset_into_to_dataframe():
    basepath = "aclImdb"

    labels = {"pos": 1, "neg": 0}

    df = pd.DataFrame()

    with tqdm(total=50000) as pbar:
        for s in ("test", "train"):
            for l in ("pos", "neg"):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                        txt = infile.read()

                    if version.parse(pd.__version__) >= version.parse("1.3.2"):
                        x = pd.DataFrame(
                            [[txt, labels[l]]], columns=["review", "sentiment"]
                        )
                        df = pd.concat([df, x], ignore_index=False)

                    else:
                        df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    df.columns = ["text", "label"]

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    print("Class distribution:")
    np.bincount(df["label"].values)

    return df


def select_short_reviews(df, percentile=50):
    lens = []
    for i in range(df.shape[0]):
        lens.append(len(df['text'][i]))
    middle_length = np.percentile(lens, percentile) 
    indices = np.where(lens <= middle_length)[0]
    df = df.iloc[indices,:]
    pos_sentiment = np.sum(df['label'] == 'positive')/df.shape[0]
    print(f'Share of reviews with positive sentiment: {pos_sentiment}')
    return df

def clean_dataset(df, 
                  percentile: int=50,
                  size: int = 25000):
  
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()
    df_shuffled = df_shuffled.drop(columns='index') 
    df_shuffled = select_short_reviews(df_shuffled,
                                       percentile=percentile)

    # add signifier for label as in: https://arxiv.org/pdf/2111.02080.pdf
    df_shuffled['text'] = df_shuffled['text'] + " // "
    df_labels = df_shuffled['label'].copy()
    df_shuffled['label'] = df_shuffled['label'].map({'negative': 0, 
                                                    'positive': 1})
    
    df_shuffled['text'] = df_shuffled['text'] + df_labels
    indices = np.arange(df_shuffled.shape[0])
    np.random.seed(0)
    ind_subset = np.random.choice(indices, size, replace=False)
    df_shuffled = df_shuffled.iloc[ind_subset,:]
    print(df_shuffled.head())
    df_shuffled.to_csv("datasets/IMDB_dataset_cleaned.csv", 
                        index=False, 
                        encoding="utf-8")
    
    '''
    # split train / test
    half = int(df_shuffled.shape[0]/2)
    df_train = df_shuffled.iloc[:half]
    # make sure to add text labels to training data, but not to test data
    df_train['text'] =  df_train['text'] + df_labels.iloc[:half]
    df_test = df_shuffled.iloc[half:]
    # speed up training by subsampling train data set!
    if frac != 1:
        np.random.seed(1)
        array = np.arange(df_train.shape[0])
        indices = np.random.choice(a=array, size=int(frac*df_train.shape[0]))
        df_train = df_train.iloc[indices]
        df_train_unlearn = df_train.iloc[0:50]
    
    print('train data probe:', df_train['text'].iloc[0])
    print('test data probe:', df_test['text'].iloc[0])

    df_train_unlearn.to_csv(f"datasets/train_unlearn_{n_unlearn}.csv", index=False, encoding="utf-8")
    df_train.to_csv("datasets/train.csv", index=False, encoding="utf-8")
    df_test.to_csv("datasets/test.csv", index=False, encoding="utf-8")
    '''
    return df_shuffled
    
    
class DS(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows