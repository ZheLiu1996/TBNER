#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pickle as pkl
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertModel
from model import sequence_correct_label_model
import numpy as np
import torch
from tqdm import tqdm
import os

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def extract_cls(read_pkl,write_pkl):
    if  'train' in read_pkl:
        with open(read_pkl, "rb") as f:
            data,_=pkl.load(f)
    else:
        with open(read_pkl, "rb") as f:
            data,_,_=pkl.load(f)
    
    bert_file='biobert_v1.1_pubmed/'            
    model_bert = BertModel.from_pretrained(bert_file)
    model_bert.cuda()
    dataloader=DataLoader(data)
    predict_cls=[]
    for batch in tqdm(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids,_,_ = batch
        all_encoder_layers, _ = model_bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        predict_cls.append(all_encoder_layers[0,0].data.cpu().numpy())
    print(len(predict_cls))
    with open(write_pkl,'wb')as f:
        pkl.dump(predict_cls,f,-1)

if __name__ == '__main__':
    root = r'data/'
    train_pkl = root + r'train_CDWA.pkl'
    dev_pkl = root + r'dev_CDWA.pkl'
    test_pkl=root + r'test_CDWA.pkl'
    read_pkls=[train_pkl,dev_pkl,test_pkl]

    cls_root=r'cls/'   
    if not os.path.exists(cls_root):
        os.makedirs(cls_root)
    cls_train_pkl=cls_root+r'train_cls.pkl'
    cls_dev_pkl=cls_root+r'dev_cls.pkl'
    cls_test_pkl=cls_root+r'test_cls.pkl'
    write_pkls=[cls_train_pkl,cls_dev_pkl,cls_test_pkl]

    for read_pkl,write_pkl in  zip(read_pkls,write_pkls):
        extract_cls(read_pkl,write_pkl)