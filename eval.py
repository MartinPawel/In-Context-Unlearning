# basics
import argparse
from argparse import ArgumentParser
import sys
import os
import os.path as op
import time
import json
import pandas as pd
import numpy as np
from typing import Optional
import scipy
import tqdm
import copy

# ML pipeline
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import torch
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelForCausalLM, BloomForCausalLM, TrainingArguments, Trainer
from dataclasses import dataclass, field

# local imports
from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, clean_dataset
from local_dataset_utilities import DS

def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=1024)

def data_collator(features: list) -> dict:
    return {"input_ids": torch.stack([torch.LongTensor(f) for f in features])}

class UnlearnTrainer(Trainer):
    '''
    Loss to enable gradient ascent on the model parameters.
    '''
    def compute_loss(self, model, inputs):
        return -model(input_ids=inputs["input_ids"],
                      attention_mask=inputs["attention_mask"],
                      labels=inputs["input_ids"]).loss


def get_dataset(config, ind_mask):
    print(f'Processing {config.dataset_name} data ...')
    # fix random seed for reproducibility
    dataset_size = ind_mask.shape[0]
    pth = f"{config.path_prefix_data}/datasets/{config.dataset_name}_n{dataset_size}"
    dataset = load_from_disk(pth)
    
    ''' get train / test splits '''
    all_indices = np.arange(dataset_size)
    print(ind_mask.shape)
    train_indices = all_indices[ind_mask]
    test_indices = all_indices[~ind_mask]
    dataset_train = dataset.select(train_indices)
    dataset_test = dataset.select(test_indices)
    print('train size', len(dataset_train['text']))
    print('test size', len(dataset_test['text']))
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    
    return dataset


def untrain(model0, batch_tokenized, num_epochs=1, batch_size=1, lr=5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    '''
    Take unlearning steps.
    '''
    model = copy.deepcopy(model0)
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=lr)
    print("Trainable parameters: ", sum([tp.nelement() for tp in train_params]))
    
    model.cuda().train()
    loss = (-1) * model(input_ids=batch_tokenized["input_ids"].cuda(), 
                        attention_mask=batch_tokenized["attention_mask"].cuda(), 
                        labels=batch_tokenized["input_ids"].cuda()).loss
    print(f"base loss: {loss.item()}")
    for j in range(num_epochs):
        if batch_size == 1:
            # go over every point in batch individually
            for k in range(batch_tokenized["input_ids"].shape[0]):
                print(f'{k}th gradient ascent iteration in epoch {j}')
                # do gradient ascent
                loss = (-1) * model(input_ids=batch_tokenized["input_ids"][k].reshape(1,-1).cuda(), 
                                    attention_mask=batch_tokenized["attention_mask"][k].reshape(1,-1).cuda(), 
                                    labels=batch_tokenized["input_ids"][k].reshape(1,-1).cuda()).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
        else:
            raise ValueError('Currently we only support sequential unlearning of forget points.')
    loss_end = (-1) * model(input_ids=batch_tokenized["input_ids"].cuda(), 
                            attention_mask=batch_tokenized["attention_mask"].cuda(), 
                            labels=batch_tokenized["input_ids"].cuda()).loss
    print(f"end loss: {loss_end.item()}")
    return model.eval()


def eval_model(model, tokenized_text, n_additional):
    '''Returns model outputs for given batch'''
    model.cuda().eval()
    with torch.no_grad():
        # print('shape of input', tokenized_text["input_ids"].shape)
        # print('length of input', length)
        length = tokenized_text["input_ids"][0].shape[0]
        out = model.generate(input_ids=tokenized_text["input_ids"].cuda(),
                             attention_mask=tokenized_text["attention_mask"].cuda(),
                             max_length=length+n_additional,
                             temperature=0.0,
                             return_dict_in_generate=True,
                             output_scores=True)

    return out


def get_icl_rep_context(batch, batch_eval, batch_label, icl_unlearn_labels, config):
    ''' When batch_size == 1, computes context of the form: <ex 1> \n <ex 1> \n ... <ex 1> \ '''
    assert config.batch_sizes[0] == 1
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]
    txt_eval = tokenizer.decode(batch_eval['input_ids'][0][0:(length_eval-1)])
    txt = tokenizer.decode(batch['input_ids'][0][0:(length-1)])
    # print('txt eval:', txt_eval)
    # print('txt:', txt)
    for j in range(config.n_ctxt+1):
        if j == 0:
            # instantiate appended post
            appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n"
        elif j < (config.n_ctxt):
            # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
            appended_post += txt + ' ' + icl_unlearn_labels[j] + "\n" 
        else:
            appended_post += txt_eval
        # print(f'iteration {j} out of {config.n_ctxt}:', appended_post)
    return appended_post
    
    
def get_icl_vary_context(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, config):
    ''' When batch_size == 1, sets up context of the form: <ex 1> \n <ex 2> \n ... <ex querry> \\ '''
    assert config.batch_sizes[0] == 1
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]
    # context for query position
    txt_eval = tokenizer.decode(batch_eval['input_ids'][0][0:(length_eval-1)])
    # content for position 1
    txt = tokenizer.decode(batch['input_ids'][0][0:(length-1)])
    # content for remaining context positions
    txts_ctxt = []
    for j in range(batch_context['input_ids'].shape[0]):
        length_ctx = batch_context['input_ids'][j].shape[0]
        txt_ctxt = tokenizer.decode(batch_context['input_ids'][j][0:(length_ctx-1)])
        txts_ctxt.append(txt_ctxt)
    # print('txts context:', txts_ctxt)
    # print('txt eval:', txt_eval)
    # print('txt:', txt)
    # setup full context
    for j in range(config.n_ctxt+1):
        if j == 0:
            # instantiate appended post
            appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n"
        elif j < (config.n_ctxt):
            # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
            appended_post += txts_ctxt[j-1] + ' ' + icl_unlearn_labels[j] + "\n" 
        else:
            appended_post += txt_eval
        # print(f'iteration {j} out of {config.n_ctxt}:', appended_post)
    return appended_post
    
    
def get_icl_vary_context_batchsize_gg1(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, idx_to_evaluate, config):
    ''' When batch_size > 1, sets up context of the form: <ex 1> \n <ex 2> \n ... <ex querry> \\ '''
    batch_size = batch['input_ids'].shape[0]
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]
    # context for query position
    txt_eval = tokenizer.decode(batch_eval['input_ids'][idx_to_evaluate][0:(length_eval-1)])
    # content for first batch_size positions
    txts_instruction = []
    for b in range(batch_size):
        txts_instruction.append(tokenizer.decode(batch['input_ids'][b][0:(length-1)]))
    # content for remaining context positions
    txts_ctxt = []
    for j in range(batch_context['input_ids'].shape[0]):
        length_ctx = batch_context['input_ids'][j].shape[0]
        txt_ctxt = tokenizer.decode(batch_context['input_ids'][j][0:(length_ctx-1)])
        txts_ctxt.append(txt_ctxt)
    # print('txts context:', txts_ctxt)
    # print('txt eval:', txt_eval)
    # print('txt:', txt)
    # setup full context

    # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
    # instantiate appended post
    appended_post = txts_instruction[0] + ' ' + icl_unlearn_labels[0] + "\n"
    # loop over remaining forget points
    for b in range(1, batch_size): 
        appended_post += txts_instruction[b] + ' ' + icl_unlearn_labels[b] + "\n"
    for j in range(config.n_ctxt):
        if j == config.n_ctxt - 1:
            appended_post += txt_eval
        else:
            appended_post += txts_ctxt[j] + ' ' + icl_unlearn_labels[batch_size + j] + "\n" 
        # print(f'iteration {j} out of {config.n_ctxt}:', appended_post)
    return appended_post
    
    

def get_icl_exchange_context(batch, batch_eval, batch_context, batch_label, icl_unlearn_labels, config):
    ''' When batch_size == 1, sets up context of the form: <ex 1> \n <ex 2> \n ... <ex querry> \\ '''
    assert config.batch_sizes[0] == 1
    if batch_label == 'forget':
        batch_eval = batch
    length = batch['input_ids'][0].shape[0]
    length_eval = batch_eval['input_ids'][0].shape[0]
    # content for query position: remains the same as in 'get_icl_vary_context'
    txt_eval = tokenizer.decode(batch_eval['input_ids'][0][0:(length_eval-1)])
    # content for position 1
    length_ctx_0 = batch_context['input_ids'][0].shape[0]
    txt = tokenizer.decode(batch_context['input_ids'][0][0:(length_ctx_0-1)])
    # content for remaining context positions: remains the same as in 'get_icl_vary_context'
    txts_ctxt = []
    for j in range(1, batch_context['input_ids'].shape[0]):
        length_ctx = batch_context['input_ids'][j].shape[0]
        txt_ctxt = tokenizer.decode(batch_context['input_ids'][j][0:(length_ctx-1)])
        txts_ctxt.append(txt_ctxt)
    # print('txts context:', txts_ctxt)
    # print('txt eval:', txt_eval)
    # print('txt:', txt)
    for j in range(config.n_ctxt+1):
        if j == 0:
            # instantiate appended post
            appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n"
        elif j < (config.n_ctxt):
            # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
            appended_post += txts_ctxt[j-1] + ' ' + icl_unlearn_labels[j] + "\n" 
        else:
            appended_post += txt_eval
        # print(f'iteration {j} out of {config.n_ctxt}:', appended_post)
    return appended_post  
    

def get_icl_standard_context(batch, batch_eval, batch_label, icl_unlearn_labels, idx_to_evaluate):
    txts_pl_other = []
    txts_pl = []
    txts = []
    for j in range(batch['input_ids'].shape[0]):
        length = batch['input_ids'][j].shape[0]
        txt_pl = tokenizer.decode(batch['input_ids'][j][0:(length-n_subtract_tokens)])
        txts_pl.append(txt_pl)
    if not (batch_label == 'forget'):
        for j in range(batch_eval['input_ids'].shape[0]):
            length_other = batch_eval['input_ids'][j].shape[0]
            txt_pl_other = tokenizer.decode(batch_eval['input_ids'][j][0:(length_other-n_subtract_tokens)])
            txts_pl_other.append(txt_pl_other)
    for j in range(batch['input_ids'].shape[0]):
        length = batch['input_ids'][j].shape[0]
        txt = tokenizer.decode(batch['input_ids'][j][0:(length-n_subtract_tokens)])
        txts.append(txt)
        if batch['input_ids'].shape[0] == 1: # case batch size 1
            if batch_label == 'forget':
                appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n" + txt
            else:
                appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n" + txts_pl_other[0]
        else: # case batch size > 1
            if j == 0:
                # instantiate appended post
                appended_post = txt + ' ' + icl_unlearn_labels[j] + "\n"
            else:
                # structure: <ex 1> \n <ex 2> \n ... <ex k> \n <ask ex 1>
                if j == (batch['input_ids'].shape[0]-1):
                    if batch_label == 'forget':
                        appended_post += txt + ' ' + icl_unlearn_labels[j] + "\n" + txts_pl[idx_to_evaluate]
                    else:
                        appended_post += txt + ' ' + icl_unlearn_labels[j] + "\n" + txts_pl_other[idx_to_evaluate]
                else:
                    appended_post += txt + ' ' + icl_unlearn_labels[j] + "\n"
    return appended_post


def prepare_text(batch,
                 batch_label,
                 batch_other,
                 batch_context,
                 config,
                 n_subtract_tokens,
                 label,
                 icl_unlearn_label,
                 mode,
                 idx_to_evaluate:int=0,
                 verbose=False):

    ''' converts batch from dataloader into tokenized txt '''
    if mode == 'ICL':
        if config.batch_sizes[0] == 1 and config.n_ctxt > 1:
            if config.ctxt_style == "ablation-rep":
                appended_post = get_icl_rep_context(batch, batch_other, batch_label, icl_unlearn_label, config)
            elif config.ctxt_style == "vary" or config.ctxt_style == "ablation-correct":
                appended_post = get_icl_vary_context(batch, batch_other, batch_context, batch_label, icl_unlearn_label, config)
            elif config.ctxt_style == "ablation-exchange":
                appended_post = get_icl_exchange_context(batch, batch_other, batch_context, batch_label, icl_unlearn_label, config)
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose either >ablation-rep<, >ablation-correct<, >ablation-exchange< or >vary<.")
        elif config.batch_sizes[0] > 1 and config.n_ctxt > 1:
            if config.ctxt_style == "vary" or config.ctxt_style == "ablation-correct":
                appended_post = get_icl_vary_context_batchsize_gg1(batch, batch_other, batch_context, batch_label, icl_unlearn_label, idx_to_evaluate, config)
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose either >ablation-correct< or >vary<.")
        else:
            raise ValueError(f"This configuration is not supported. You chose a batch size of {config.batch_sizes[0]} and n_ctxt of {config.n_ctxt}.")
    else:
        # use idx_to_evaluate-th point of batch for evaluation
        length = batch['input_ids'][idx_to_evaluate].shape[0]
        if batch_label == 'forget':
            appended_post = tokenizer.decode(batch['input_ids'][idx_to_evaluate][0:(length-n_subtract_tokens)])
        else:
            length_other = batch_other['input_ids'][idx_to_evaluate].shape[0]
            appended_post = tokenizer.decode(batch_other['input_ids'][idx_to_evaluate][0:(length_other-n_subtract_tokens)])
    if config.verbose:
        print('IDX to eval:', idx_to_evaluate)
        print('BATCH LABEL:', batch_label)
        print(f'CONTEXT FOR {mode}:', appended_post)
        print('------------------------------------')
    tokenized_text = tokenizer(appended_post, return_tensors="pt")
    return tokenized_text

def get_id(word: str):
    ''' returns the id corresponding to a word/str '''
    tokenized_text = tokenizer(word, return_tensors="pt")
    id = tokenized_text['input_ids'][0][0]
    return id

def compute_token_preds(out):
    pred_token = tokenizer.decode(out['sequences'][0][-1]).strip()
    return pred_token

def get_stable_logit_loss(out, label, id_pos, id_neg, eps=1e-35):
    '''
    Computing stable logit loss from Section VI (https://arxiv.org/abs/2112.03570)
    '''
    if label == 0:
        Y = id_neg.reshape(1,-1)
    else:
        Y = id_pos.reshape(1,-1)
    probs = torch.nn.functional.softmax(out['scores'][0][-1], dim=0).reshape(1,-1)
    # print('print prob at pred', probs)
    class_log_probs = torch.log(probs[torch.arange(probs.shape[0]), Y] + eps)  # compute log(f(x)_y)
    m, n = probs.shape
    del_ind = Y
    mask = torch.ones((m, n), dtype=bool)
    mask[range(m), del_ind] = False
    probs_complement = probs[mask].reshape(m, n-1)
    complement_class_sum = torch.sum(probs_complement, axis=1)                 # compute log(\sum_{y'} f(x)_{y'})
    score = class_log_probs - torch.log(complement_class_sum + eps)            # compute log(f(x)_y) - log(\sum_{y'} f(x)_{y'})
    # get class probs and complement class probs for analysis purposes
    # print('class log probs shape:', class_log_probs)
    # print('complement_class_sum shape:', complement_class_sum)
    return score[0][0].detach().cpu().numpy(), class_log_probs[0][0].detach().cpu().numpy(), torch.log(complement_class_sum + eps)[0].detach().cpu().numpy()

def setup_context_loader(train_dataset, n_ctxt):
    # first position is for point we want to unlearn; all other positions will be filled by samples from context_loader.
    context_loader = DataLoader(dataset=train_dataset,
                                batch_size=n_ctxt,
                                shuffle=True,
                                num_workers=4,
                                drop_last=True)
    return context_loader

def setup_dict(config, batch_size):
    results = {}
    methods = config.unlearning_methods + ['base']
    regimes = ['forget', 'test', 'train']
    types = ['losses', 'nxt_token_preds']
    for t in ['losses', 'nxt_token_preds']:
        for method in methods:
            for regime in regimes:
                for be in range(batch_size):
                    results[f'{t}_{method}_{regime}_{be}'] = []
    # for analysis purposes
    if 'icl' in methods:
        for regime in regimes:
            for be in range(batch_size):
                results[f'confs_icl_first_{regime}_{be}'] = []
                results[f'confs_icl_others_{regime}_{be}'] = []
    # to compute model performance
    for regime in regimes:
        for be in range(batch_size):
            results[f'labels_{regime}_{be}'] = []
    
    return results
    
def evals(k,
          model,
          config,
          train_dataset,
          forget_loader,
          train_loader,
          test_loader,
          n_subtract_tokens: int=1,
          n_additional: int=1,
          model_chckpt: str="finetuned_models/checkpoint-1547"):

    '''Evaluates the performance of various unlearning strategies:
       options are unlearn = {'icl', 'ga'} '''

    '''Make sure we collect results'''
    count = 0
    output_size = model.lm_head.out_features
    print(f'output size: {output_size}')
    # infer batch size
    for idx, batch in enumerate(forget_loader):
        batch_size = batch['input_ids'].shape[0]
        if idx == 0:
            break
    # Setup dictionary to collect results
    results = setup_dict(config, batch_size)

    '''Get the correct label ids: The white space for these ones is IMPORTANT'''
    id_pos = get_id(" positive")
    print('positive id:', {id_pos})
    id_neg = get_id(" negative")
    print('negative id:', {id_neg})

    '''Setup loaders'''
    if config.ctxt_style == "vary" or config.ctxt_style == "ablation-correct":
        context_loader = setup_context_loader(train_dataset, config.n_ctxt-1)
    else:
        context_loader = setup_context_loader(train_dataset, config.n_ctxt)
    it = iter(train_loader)
    jt = iter(test_loader)
    kt = iter(context_loader) # used when: config.ctxt_style == "vary"
    
    '''Start (unlearning) eval loop'''
    print(f"Using the following label flipping strategey: {config.label_flipping_method}")
    for idx, batch in enumerate(forget_loader):
        '''Samples train / test points for evaluation'''
        batch_train = next(it)
        batch_test = next(jt)
        try:
            '''Samples the context batch'''
            batch_context = next(kt)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            if config.ctxt_style == "vary" or config.ctxt_style == "ablation-correct":
                context_loader = setup_context_loader(train_dataset, config.n_ctxt-1)
            else:
                context_loader = setup_context_loader(train_dataset, config.n_ctxt)
            kt = iter(context_loader)
            batch_context = next(kt)
        assert batch_size >= config.unlearn_batch_size
        if idx % 100 == 0:
            print(f'Evaluating sample: {idx}')
        
        ''' Get the right set of labels '''
        labels = []
        icl_unlearn_label = []
        ''' case: batch_size > 1 '''
        if batch_size > 1:
            if config.ctxt_style == "vary" or config.ctxt_style == "ablation-correct":
                # If the batch_size > 1, we will put the points from the forget set into the first batch_size positions and change their labels.
                # In ablation setting "ablation-correct", we will instead use the correct label.
                # The other positions will be filled up with correctly labeled points from the train set.
                ''' first batch_size context positions '''
                for b in range(batch_size):
                    if batch['label'][b] == 1:
                        labels.append('positive')
                        if config.ctxt_style == "ablation-correct":
                            icl_unlearn_label.append('positive') # keep true label! do not flip label
                        else:
                            icl_unlearn_label.append('negative') # flip label of forget point
                    else:
                        labels.append('negative')
                        if config.ctxt_style == "ablation-correct":
                            icl_unlearn_label.append('negative') # keep true label! do not flip label
                        else:
                            icl_unlearn_label.append('positive') # flip label of forget point
                ''' remaining context positions '''
                # append the correct labels for the remaining context positions
                for j in range(config.n_ctxt-1):
                    if batch_context['label'][j] == 1:
                        icl_unlearn_label.append('positive')
                    else:
                        icl_unlearn_label.append('negative')
                print(f'Labels for context style: {config.ctxt_style}')
                print('icl unlearn labels:', icl_unlearn_label)
                print('labels:', labels)
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose either >ablation-rep<, >ablation-exchange<, >ablation-correct< or >vary<.")
        # case: batch_size == 1
        else:
            if config.ctxt_style == "ablation-rep":
                # If the batch_size == 1, we will repeat the same entry n_ctxt times and flip the first or last label.
                # To do this, we prepare icl_unlearn_label accordingly.
                if batch['label'][0] == 1:
                    labels.append('positive')
                    icl_unlearn_label = ['positive'] * config.n_ctxt
                    if config.label_flipping_method == "first-k":
                        icl_unlearn_label[0] = 'negative'
                    elif config.label_flipping_method == "last-k":
                        icl_unlearn_label[-1] = 'negative'
                    else:
                        raise ValueError(f"{config.label_flipping_method} is not supported when >batch_size< == 1.")
                else:
                    labels.append('negative')
                    icl_unlearn_label = ['negative'] * config.n_ctxt
                    if config.label_flipping_method == "first-k":
                        icl_unlearn_label[0] = 'positive'
                    elif config.label_flipping_method == "last-k":
                        icl_unlearn_label[-1] = 'positive'
                    else:
                        raise ValueError(f"{config.label_flipping_method} is not supported when >batch_size< == 1.")
            
            elif config.ctxt_style == "vary" or config.ctxt_style == "ablation-correct":
                # If the batch_size == 1, we will put the point from the forget set into the first position and change its label.
                # In ablation setting "ablation-correct", we will instead use the correct label.
                # The other positions will be filled up with correctly labeled points from the train set
                ''' first context position '''
                if batch['label'][0] == 1:
                    labels.append('positive')
                    if config.ctxt_style == "ablation-correct":
                        icl_unlearn_label.append('positive') # keep true label! do not flip label
                    else:
                        icl_unlearn_label.append('negative') # flip label of forget point
                else:
                    labels.append('negative')
                    if config.ctxt_style == "ablation-correct":
                        icl_unlearn_label.append('negative') # keep true label! do not flip label
                    else:
                        icl_unlearn_label.append('positive') # flip label of forget point
                ''' remaining context positions '''
                # append the correct labels for the remaining context positions
                for j in range(config.n_ctxt-1):
                    if batch_context['label'][j] == 1:
                        icl_unlearn_label.append('positive')
                    else:
                        icl_unlearn_label.append('negative')
                print(f'Labels for context style: {config.ctxt_style}')
                print('icl unlearn labels:', icl_unlearn_label)
                print('labels:', labels)
            
            elif config.ctxt_style == "ablation-exchange":
                ''' first context position is filled up with random train point '''
                if batch_context['label'][0] == 1:
                    icl_unlearn_label.append('negative') # flip label of random train point
                else:
                    icl_unlearn_label.append('positive') # flip label of random train point
                ''' remaining context positions '''
                # append the correct labels for the remaining context positions
                for j in range(1,config.n_ctxt):
                    if batch_context['label'][j] == 1:
                        icl_unlearn_label.append('positive')
                    else:
                        icl_unlearn_label.append('negative')
                ''' collect true label for forget point we evaluate on '''
                if batch['label'][0] == 1:
                    labels.append('positive')
                else:
                    labels.append('negative')
            
            else:
                raise ValueError(f"Other context styles are currently not supported. Please choose either >ablation-rep<, >ablation-exchange<, >ablation-correct< or >vary<.")
        
        print(f'Labels for context style: {config.ctxt_style}')
        print(f'icl unlearn labels: {icl_unlearn_label}')
        print(f'labels: {labels}')
        
        # append post
        print('-----------------------------------------------------------')
        print('Counter at:', idx)
        ''' Decide how the model is loaded as we go through the evaluation loop. '''
        if 'ga' in config.unlearning_methods:
            if 'icl' in config.unlearning_methods:
                # We cannot keep the model in memory if GA and ICL are run; will usually run out of ram. Make sure to load model from chkpt to avoid contamination from 'unlearned model'.
                # Note that this is slow. Doing eval separately is more recommended.
                if idx > 0:
                    print('Load baseline model ...')
                    if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
                        model = BloomForCausalLM.from_pretrained(model_chckpt)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_chckpt)
            else:
                if idx  == 0:
                    config.ga_in_memory = True
                    # Here we keep the model in memory to save time when only running GA evaluation as loading the model takes time
                    print('Load baseline model once ...')
                    if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
                        model_mem = BloomForCausalLM.from_pretrained(model_chckpt)
                    else:
                        model_mem = AutoModelForCausalLM.from_pretrained(model_chckpt)
        
        ''' Baseline predictions '''
        print('Baseline predictions ...')
        batchers = [None, batch_test, batch_train]
        loader_labels = ['forget', 'test', 'train']
        for sample_idx in range(batch_size):
            for l_idx, batch_other in enumerate(batchers):
                batch_label = loader_labels[l_idx]
                tokenized_text_base = prepare_text(batch,
                                                   batch_label,
                                                   batch_other,
                                                   batch_context,
                                                   config,
                                                   n_subtract_tokens,
                                                   labels,
                                                   icl_unlearn_label,
                                                   verbose=config.verbose,
                                                   idx_to_evaluate=sample_idx,
                                                   mode='no')
                
                
                if config.ga_in_memory:
                    print('Using model from memory to compute loss...')
                    out_baseline = eval_model(model_mem,
                                              tokenized_text_base,
                                              n_additional)
                else:
                    out_baseline = eval_model(model,
                                              tokenized_text_base,
                                              n_additional)

                if batch_label == 'forget':
                    lab = batch['label'][sample_idx]
                else:
                    lab = batch_other['label'][sample_idx]

                results[f'losses_base_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_baseline,
                                                                                                lab,
                                                                                                id_pos,
                                                                                                id_neg)[0])

                # get predictions and corresponding labels
                results[f'nxt_token_preds_base_{batch_label}_{sample_idx}'].append(compute_token_preds(out_baseline))
                results[f'labels_{batch_label}_{sample_idx}'].append(lab)
        
        ''' In context unlearning evaluation '''
        if 'icl' in config.unlearning_methods:
            print('ICL unlearning ...')
            batchers = [None, batch_test, batch_train]
            batch_labels = ['forget', 'test', 'train']
            # loop over batch size elements
            for sample_idx in range(batch_size):
                # loop over forget test & train points
                for l_idx, batch_other in enumerate(batchers):
                    batch_label = batch_labels[l_idx]
                    # prepare tokens to get ICL unlearning predictions
                    tokenized_text = prepare_text(batch,
                                                  batch_label,
                                                  batch_other,
                                                  batch_context,
                                                  config,
                                                  n_subtract_tokens,
                                                  labels,
                                                  icl_unlearn_label,
                                                  verbose=config.verbose,
                                                  idx_to_evaluate=sample_idx,
                                                  mode='ICL')
                    out_icl = eval_model(model,
                                         tokenized_text,
                                         n_additional)


                    if batch_label == 'forget':
                        lab = batch['label'][sample_idx]
                    else:
                        lab = batch_other['label'][sample_idx]

                    results[f'nxt_token_preds_icl_{batch_label}_{sample_idx}'].append(compute_token_preds(out_icl))
                    results[f'losses_icl_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_icl,
                                                                                                   lab,
                                                                                                   id_pos,
                                                                                                   id_neg)[0])                        
                    results[f'confs_icl_first_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_icl,
                                                                                                        lab,
                                                                                                        id_pos,
                                                                                                        id_neg)[1])
                    results[f'confs_icl_others_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_icl,
                                                                                                         lab,
                                                                                                         id_pos,
                                                                                                         id_neg)[2])
        ''' GA unlearning evaluation '''
        if 'ga' in config.unlearning_methods:
            print('GA unlearning ...')
            if config.unlearn_batch_size > 1:
                print('You are using a batch size > 1. Check whether this is desired behviour?')

            # unlearning with batch size >= 1 & unlearn_batch_size = 1
            if config.ga_in_memory:
                print('Using model from memory ...')
                model = untrain(model_mem,
                                batch,
                                num_epochs=config.n_unlearn_epochs,
                                batch_size=config.unlearn_batch_size,
                                lr=config.lr)
            else:
                model = untrain(model,
                                batch,
                                num_epochs=config.n_unlearn_epochs,
                                batch_size=config.unlearn_batch_size,
                                lr=config.lr)
                                
            batchers = [None, batch_test, batch_train]
            batch_labels = ['forget', 'test', 'train']
            for sample_idx in range(batch_size):
                for l_idx, batch_other in enumerate(batchers):
                    batch_label = batch_labels[l_idx]
                    tokenized_text_ga = prepare_text(batch,
                                                     batch_label,
                                                     batch_other,
                                                     batch_context,
                                                     config,
                                                     n_subtract_tokens,
                                                     labels,
                                                     icl_unlearn_label,
                                                     verbose=config.verbose,
                                                     idx_to_evaluate=sample_idx,
                                                     mode='GA')

                    # eval model sample_index-th 'untrained point'
                    out_ga = eval_model(model,
                                        tokenized_text_ga,
                                        n_additional)

                    if batch_label == 'forget':
                        lab = batch['label'][sample_idx]
                    else:
                        # print(batch_label)
                        lab = batch_other['label'][sample_idx]

                    results[f'nxt_token_preds_ga_{batch_label}_{sample_idx}'].append(compute_token_preds(out_ga))
                    results[f'losses_ga_{batch_label}_{sample_idx}'].append(get_stable_logit_loss(out_ga,
                                                                                                  lab,
                                                                                                  id_pos,
                                                                                                  id_neg)[0])
        
        ''' For testing purposes '''
        if idx == config.n_samples:
            break
        
        ''' Save Intermediate Results '''
        divisor = 3000
        if idx % divisor == 0:
            print(results)
            res = pd.DataFrame.from_dict(results)
            if config.unlearning_methods[0] == 'icl':
                res.to_csv(f'./results/_results_{config.dataset_name}_{config.model_name}_model{k}_{config.unlearning_methods[0]}_n{idx}_outof{config.n_samples}_mepochs{config.model_epochs}_uepochs{config.n_unlearn_epochs}_bs{batch_size}_{config.ctxt_style}_nctxt{config.n_ctxt}_lfm{config.label_flipping_method}.csv')
            else:
                config.lr_ = str(config.lr)
                res.to_csv(f'./results/_results_{config.dataset_name}_{config.model_name}_model{k}_{config.unlearning_methods[0]}_n{idx}_outof{config.n_samples}_mepochs{config.model_epochs}_uepochs{config.n_unlearn_epochs}_bs{batch_size}_lr{config.lr_}.csv')

    ''' Save all results '''
    res = pd.DataFrame.from_dict(results)
    if config.unlearning_methods[0] == 'icl':
        res.to_csv(f'./results/results_{config.dataset_name}_{config.model_name}_model{k}_{config.unlearning_methods[0]}_n{config.n_samples}_mepochs{config.model_epochs}_uepochs{config.n_unlearn_epochs}_bs{batch_size}_{config.ctxt_style}_nctxt{config.n_ctxt}_lfm{config.label_flipping_method}.csv')
    else:
        config.lr_ = str(config.lr)
        res.to_csv(f'./results/results_{config.dataset_name}_{config.model_name}_model{k}_{config.unlearning_methods[0]}_n{config.n_samples}_mepochs{config.model_epochs}_uepochs{config.n_unlearn_epochs}_bs{batch_size}_lr{config.lr_}.csv')

    return res



if __name__ == '__main__':
    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Config file.")
    parser.add_argument("--batch_sizes", default=None, type=int)
    parser.add_argument("--n_ctxt", default=None, type=int, help="Int: Length of context when batch_size=1.")
    parser.add_argument("--ctxt_style", default=None, type=str, help="Str: Context style must be >ablation-rep<, >vary<, >ablation-correct< or >ablation-exchange<.")
    parser.add_argument("--K_models", default=None, type=int, help="Int: How many models to run the evaluation over. Should usually be 1 when running several evaluations in parallel. Make sure you understand this setup.")
    parser.add_argument("--rng_offset", default=None, type=int, help="Int: Number of run indicating which model will be used.")
    parser.add_argument("--lfm", default=None, type=str, help="Str: One of >first-k<, >last-k<, or >flipp_all<.")
    parser.add_argument("--model_path", default=None, type=str, help="Str: One of: >gpt2<, >gpt2-medium< or >bigscience/bloom-560m<.")
    parser.add_argument('--dataset_name', default=None, type=str, help="Str: Whichh dataset to use for the evaluation. Options are >sst2<, >imdb<, >yelp_polarity< or >amazon_polarity<")
    parser.add_argument('--lr', default=5e-5, type=float, help="Float: Learning rate for gradient ascent unlearning.")
    
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a >config< file in the argument please.")
    if arg_.batch_sizes is None:
        raise NameError("Include a >batch< size in the argument please.")
    if arg_.ctxt_style is None:
        raise NameError("Include a >ctxt_style< in the argument please. One of >rep< or >vary<.")
    if arg_.batch_sizes == 1 and arg_.ctxt_style == "ablation-rep" and arg_.n_ctxt is None:
        raise NameError("Please incldue >n_ctxt< in the argument please when batch_size=1 and ctxt_style = >rep<.")
    if arg_.K_models is None:
        raise NameError("Include >K_models< in the argument please.")
    if arg_.lr is None:
        raise NameError("Please include >lr< in the argument please.")
    if arg_.rng_offset is None:
        raise NameError("Include >rng_offset< in the argument please.")
    if arg_.lfm is None:
        raise NameError("Include >lfm< in the argument please. One of >first-k<, >last-k<, >random< or >flip_all<.")
    if arg_.model_path is None:
        raise NameError("Include a >model_path< in the argument please: Onf of >gpt2<, >gpt2-medium<, >bigscience/bloom-560m< or >bigscience/bloom-1b1<.")
    if arg_.dataset_name is None:
        raise NameError("Include a >dataset_name< in the argument please. One of >sst2<, >imdb<, >yelp_polarity< or >amazon_polarity<.")

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    # Init configs that are not given (this is legacy stuff - change if time permits.)
    if "model_epochs" not in config:
        config.model_epochs = 1
    if "unlearning_methods" not in config:
        config.unlearning_methods = ["icl"]
    if "n_samples" not in config:
        config.n_samples = 12500
    if "n_unlearn_epochs" not in config:
        config.n_unlearn_epochs = 1
    if "verbose" not in config:
        config.verbose = False
    if 'path_prefix_model' not in config:
        config.path_prefix_model = 'path/to/model'
    if 'path_prefix_data' not in config:
        config.path_prefix_data = 'path/to/data'
        
    assert len(config.unlearning_methods) == 1

    # add to config (this is legacy stuff - change if time permits)
    config.ga_in_memory = False
    config.label_flipping_method = arg_.lfm
    config.K_models = arg_.K_models
    config.rng_offset = arg_.rng_offset
    config.batch_sizes = [arg_.batch_sizes]
    config.model_name_or_path = arg_.model_path
    config.lr = float(arg_.lr)
    print(f'Learning rate: {config.lr}')
    config.n_ctxt = arg_.n_ctxt
    config.ctxt_style = arg_.ctxt_style
    if 'bloom-560m' in config.model_name_or_path:
    	config.model_name = 'bloom-560m'
    elif 'bloom-1b1' in config.model_name_or_path:
        config.model_name = 'bloom-1b1'
    else:
        config.model_name = config.model_name_or_path
    config.ctxt_style = arg_.ctxt_style
    config.dataset_name = arg_.dataset_name

    # load cleaned data set
    if config.dataset_name == 'imdb':
        df_all = pd.read_csv("datasets/IMDB_dataset_cleaned.csv", index_col=False)
    
    ''' eval loop over shadow models with index >config.rng_offset + k< when K_models=1 '''
    for k in range(config.K_models):
        if config.rng_offset > 0:
            k = config.rng_offset + k
        print(f'Evaluation for model: {k} ...')
        for batch_size in config.batch_sizes:
            # select correct in / out splits
            ind_mask_k = pd.read_csv(f"finetuned_models/{config.dataset_name}_indices_epochs{config.model_epochs}_unlearnbs{batch_size}_kmodel{k}.csv", 
                                     index_col=False).to_numpy()
            
            # legacy imdb dataset stuff: change if time permits
            if config.dataset_name == 'imdb':
                df_k_train = df_all.iloc[ind_mask_k,:]
                df_k_test = df_all.iloc[(~ind_mask_k),:]       
                # load datasets & tokenize
                dataset = DatasetDict({"train": Dataset.from_pandas(df_k_train),
                                       "test": Dataset.from_pandas(df_k_test)})
            else:
                dataset = get_dataset(config, ind_mask_k.reshape(-1))
                                   
            # get the right tokenizer
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
            tokenized = dataset.map(tokenize_text, batched=True, batch_size=None)
            tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            del dataset
            
            train_dataset = DS(tokenized, partition_key="train")
            test_dataset = DS(tokenized, partition_key="test")
    
            # make sure we load the correct model    
            print('Loading model from checkpoint ...')
            if config.path_prefix_model is None:
                model_chckpt = f"finetuned_models/{config.dataset_name}_epochs{config.model_epochs}_unlearnbs{batch_size}_kmodel{k}_{config.model_name}"
            else:
                model_chckpt = f"{config.path_prefix_model}/finetuned_models/{config.dataset_name}_epochs{config.model_epochs}_unlearnbs{batch_size}_kmodel{k}_{config.model_name}"
            
            if 'bloom-560m' in config.model_name_or_path or 'bloom-1b1' in config.model_name_or_path:
                model = BloomForCausalLM.from_pretrained(model_chckpt)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_chckpt)
            print(f'Batch_size: {batch_size} - n_unlearn_epoch: {config.n_unlearn_epochs}')
    
            forget_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=True)
    
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True)
    
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=True)
    
            results_unlearn = evals(k,
                                    model,
                                    config,
                                    train_dataset,
                                    forget_loader,
                                    train_loader,
                                    test_loader,
                                    n_subtract_tokens=1,
                                    n_additional=1,
                                    model_chckpt=model_chckpt)