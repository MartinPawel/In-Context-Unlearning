# basics
import argparse
from argparse import ArgumentParser
import os
import os.path as op
import time
import json
import pandas as pd
import numpy as np
import scipy

# ML pipeline
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelForCausalLM, BloomForCausalLM, TrainingArguments, Trainer

# local imports
from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, clean_dataset


def add_suffix(sample):
    if sample['label'] == 0:
        label = 'negative'
    else:
        label = 'positive'
    sample['text'] = sample['text'] + ' // ' + label
    return sample

def remove_commawhitespace(sample):
    sample['text'] = sample['text'].replace(" ,",",")
    return sample

def remove_qwhitespace(sample):
    sample['text'] = sample['text'].replace(" ? ","?")
    return sample

def remove_dwhitespace(sample):
    sample['text'] = sample['text'].replace(" . ",".")
    return sample

def remove_exwhitespace(sample):
    sample['text'] = sample['text'].replace(" ! ","!")
    return sample

def remove_swhitespace(sample):
    sample['text'] = sample['text'].replace(" 's","'s")
    return sample

def strip_whitespace(sample):
    sample['text'] = sample['text'].rstrip()
    return sample

def get_dataset(config, ind_mask):
    print(f'Processing {config.dataset_name} data ...')
    # fix random seed for reproducibility
    np.random.seed(13)
    assert config.dataset_size == ind_mask.shape[0]
    pth = f"{config.path_prefix_data}/datasets/{config.dataset_name}_n{config.dataset_size}"
    
    if not os.path.exists(pth):
        if config.dataset_name == 'sst2':
            ''' processing: sst2 '''
            dataset = load_dataset('sst2', split=f'train[0:{config.dataset_size}]')
            # rename column from >sentence< to >text<
            dataset = dataset.rename_column("sentence", "text")
            # remove comma space
            dataset = dataset.map(remove_commawhitespace)
            # remove other white spaces
            dataset = dataset.map(remove_swhitespace)
            dataset = dataset.map(remove_qwhitespace)
            dataset = dataset.map(remove_exwhitespace)
            dataset = dataset.map(remove_dwhitespace)
            # remove white space at end
            dataset = dataset.map(strip_whitespace)
            # add label to the end
            dataset = dataset.map(add_suffix)
        elif config.dataset_name == "amazon_polarity" or config.dataset_name == "yelp_polarity":
            ''' processing: >yelp_polarity< or >amazon polarity< '''
            dataset = load_dataset(config.dataset_name, split='train')
            if config.dataset_name == 'amazon_polarity':
                dataset = dataset.rename_column("content", "text")
            # keep documents shorter than the median length
            dataset = dataset.map(lambda example: {'len': len(example['text'])})
            med = np.median(np.array(dataset['len']))
            indeces = np.where(dataset['len'] < med)[0]
            dataset = dataset.select(indeces)
            # remove white space
            dataset = dataset.map(remove_commawhitespace)
            dataset = dataset.map(remove_swhitespace)
            # remove white space at end
            dataset = dataset.map(strip_whitespace)
            # add label to end
            dataset = dataset.map(add_suffix)
            # select smaller random subset of size >dataset_size<
            n = len(dataset['text'])
            all_indices = range(n)
            indices_subset = np.random.choice(all_indices, config.dataset_size)
            dataset = dataset.select(indices_subset)
            # save processed dataset
        else:
            raise ValueError(f'This >{config.dataset_name}< is not provided, yet.')
        # now save dataset to disk
        dataset.save_to_disk(pth)
    else:
        dataset = load_from_disk(pth)
    
    ''' get train / test splits '''
    all_indices = np.arange(config.dataset_size)
    print(ind_mask.shape)
    train_indices = all_indices[ind_mask]
    test_indices = all_indices[~ind_mask]
    dataset_train = dataset.select(train_indices)
    dataset_test = dataset.select(test_indices)
    print('train size', len(dataset_train['text']))
    print('test size', len(dataset_test['text']))
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    
    return dataset


def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, unlearn=False):
        return model(input_ids=inputs["input_ids"],
                     attention_mask=torch.ones_like(inputs["input_ids"]).bool(),
                     labels=inputs["input_ids"]).loss

def data_collator(features: list) -> dict:
    return {"input_ids": torch.stack([torch.LongTensor(f) for f in features])}


def train_models(model, 
                 data_tokenized,
                 args,
                 model_index):

    training_args = TrainingArguments(output_dir="finetuned_models",
                                      seed=0,
                                      fp16=args.fp16,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      per_device_train_batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      num_train_epochs=args.n_epochs,
                                      save_strategy="no",
                                      evaluation_strategy="epoch"
    )
    

    print('Starting the learning proecdure via gradient descent')

    trainer = ModifiedTrainer(model=model,
                              train_dataset=data_tokenized['train']['input_ids'],
                              eval_dataset=data_tokenized['test']['input_ids'],
                              args=training_args,
                              data_collator=data_collator
    )
    

    trainer.train()
    if 'bloom-560m' in config.model_name_or_path:
        model_name = 'bloom-560m'
    elif 'bloom-1b1' in config.model_name_or_path:
        model_name = 'bloom-1b1'
    else:
        model_name = config.model_name_or_path
    trainer.model.save_pretrained(f"{args.path_prefix_model}/finetuned_models/{config.dataset_name}_epochs{args.n_epochs}_unlearnbs{args.u_bs}_kmodel{model_index}_{model_name}")

if __name__ == '__main__':
    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--dataset_name', default=None, type=str, help="One of >imdb<, >sst2<, >yelp_polarity< or >amazon_polarity<.")
    parser.add_argument('--dataset_size', default=25000, type=int, help="All experiments are run with default size of 25,000. If you wish to change this, make sure to train new models & rerun eval.")
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")
    if arg_.dataset_name is None:
        raise NameError("Include a dataset_name in the argument please.")
    if arg_.dataset_size is None:
        raise NameError("Include a dataset_size in the argument please.")

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    # Init configs that are not given
    if 'seed' not in config:
        seed = 42
    if 'learning_rate' not in config:
        config.learning_rate = 5e-5
    if 'batch_size' not in config:
        config.batch_size = 2
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 4
    if 'n_epochs' not in config:
        config.n_epochs = 1
    if 'num_workers' not in config:
        config.num_workers = 2
    if 'fp16' not in config:
        config.fp16 = True
    if 'tokenizer' not in config:
        config.tokenizer_name_or_path = config.model_name_or_path
    if 'checkpoint' not in config:
        config.checkpoint = None
    if 'frac' not in config:
        config.frac = 0.99
    if 'max_length' not in config:
        config.max_length = 1024
    if 'eval' not in config:
        config.eval = False
    if 'u_bs' not in config:
        config.u_bs = 4
    if 'K_models' not in config:
        config.K_models = 7
    if 'in_prob' not in config:
        config.in_prob = 0.5
    if 'rng_offset' not in config:
        config.rng_offset = 0
    if 'path_prefix_model' not in config:
        config.path_prefix_model = 'path/to/model'    
    if 'path_prefix_data' not in config:
        config.path_prefix_data = 'path/to/data'
    
    if 'bloom-560m' in config.model_name_or_path:
        model_name = 'bloom-560m'
    elif 'bloom-1b1' in config.model_name_or_path:
        model_name = 'bloom-1b1'
    else:
        model_name = config.model_name_or_path
    config.mu = config.K_models * config.in_prob   # expected number of ins per six models
    torch.manual_seed(123)
    
    # add to config (this is legacy stuff - change if time permits):
    config.dataset_size = int(arg_.dataset_size)
    config.dataset_name = arg_.dataset_name
    
    ##########################
    ### 1 Loading the Datasets
    ##########################
    
    m_f_mult = [False] * config.u_bs
    m_t_mult = [True] * config.u_bs
    
    if config.dataset_name == 'imdb':
        df = pd.read_csv("datasets/IMDB_Dataset.csv", index_col=False)
        print('Cleaning dataset ...')
        df = clean_dataset(df, percentile=50) 
        n = df.shape[0]
    else:    
        n = config.dataset_size
    print(f'Dataset of size: {n}')

    # assert n % config.u_bs == 0
    n_eff = int(n/config.u_bs)                     # effective sample size
    mask = np.zeros((n_eff, config.K_models), 
                    dtype=bool)
    mask_extended = np.ones((n, config.K_models), 
                            dtype=bool)

    for k in range(config.K_models):
        # check if model exists: if it does, continue
        pth = f"{config.path_prefix_model}/finetuned_models/{config.dataset_name}_epochs{config.n_epochs}_unlearnbs{config.u_bs}_kmodel{k}_{model_name}"
        print(pth)
        if os.path.exists(pth):
            continue
        
        print(f'Training {k}th model ...')
        if config.K_models == 1:
            print("We train one model at a time. Make sure rng_offset is set correctly to keep control over random index selection.")
            assert config.rng_offset > 0
        
        ''' 1a): Get the indices right '''
        mask_k = np.zeros(n_eff, 
                          dtype=bool)
        np.random.seed(k + config.rng_offset)
        indices = np.arange(n_eff)
        chosen = np.random.choice(indices, 
                                  int(config.in_prob*n_eff), 
                                  replace=False)
        
        mask_k[chosen] = True
        mask[chosen, k] = True
        
        # make sure index elements of mask correspond to batches:
        ind_mask = []
        for j in range(n_eff):
            if mask[j,k]:
                ind_mask.append(m_t_mult)
            else:
                ind_mask.append(m_f_mult)
        ind_mask = np.array(ind_mask).reshape(-1)
        if n - ind_mask.shape[0] == 0:
            mask_extended[:,k] = ind_mask
        else:
            # if not evenly divisible, fill up remaining spots
            ind_mask = np.r_[ind_mask, np.zeros(n-ind_mask.shape[0], dtype=bool)]
            mask_extended[:,k] = np.r_[ind_mask, np.zeros(n-ind_mask.shape[0], dtype=bool)]
        
        ind_mask = pd.DataFrame(ind_mask)
        
        if config.K_models == 1:
            k = k + config.rng_offset
        ind_mask.to_csv(f"finetuned_models/{config.dataset_name}_indices_epochs{config.n_epochs}_unlearnbs{config.u_bs}_kmodel{k}.csv",
                        index=False, 
                        encoding="utf-8")
        
        ''' 1b): Loading train & test splits '''
        # legacy data loading for imdb: change if time permits 
        if config.dataset_name == "imdb":
            ind_mask = ind_mask.to_numpy().reshape(-1)
            df_k_train = df.iloc[ind_mask,:]
            df_k_train.to_csv("datasets/train.csv",
                              index=False, 
                              encoding="utf-8")
            df_k_test = df.iloc[(~ind_mask),:]
            df_k_test.to_csv("datasets/test.csv",
                             index=False, 
                             encoding="utf-8")
            dataset = DatasetDict({"train": Dataset.from_pandas(df_k_train),
                                   "test": Dataset.from_pandas(df_k_test)})
        else:
            dataset = get_dataset(config, ind_mask.to_numpy().reshape(-1))
        #########################################
        ### 2 Tokenization and Numericalization
        #########################################
        
        if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path,
                                                      max_length=config.max_length)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(config.model_name_or_path,
                                                      max_length=config.max_length,
                                                      padding_side='left')
            tokenizer.pad_token = '<pad>'
        print("Tokenizer input max length:", tokenizer.model_max_length, flush=True)
        print("Tokenizer vocabulary size:", tokenizer.vocab_size, flush=True)
    
        print("Tokenizing ...", flush=True)
        data_tokenized = dataset.map(tokenize_text, batched=True, batch_size=None)
        data_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        del dataset

        #########################################
        ### 4 Initializing the Model
        #########################################
        
        print("Training model from scratch ...")
        if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
            model = BloomForCausalLM.from_pretrained(config.model_name_or_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    
        #########################################
        ### 5 Train or load models
        #########################################
    
        print('Finetuning new model from scratch ...')
        # train & save model if doesnt exist
        print(f'Starting training for model index: {k}')
        train_models(model, 
                     data_tokenized, 
                     config,
                     k)
    
    
    mask = pd.DataFrame(mask)
    mask_extended = pd.DataFrame(mask_extended)
    ind_mask = pd.DataFrame(ind_mask)
    
    if config.K_models > 1:
        mask.to_csv(f"finetuned_models/{config.dataset_name}_mask_epochs{config.n_epochs}_unlearnbs{config.u_bs}.csv",
                    index=False, 
                    encoding="utf-8")
                  
        mask_extended.to_csv(f"finetuned_models/{config.dataset_name}_extendedmask_epochs{config.n_epochs}_unlearnbs{config.u_bs}.csv",
                             index=False, 
                             encoding="utf-8")