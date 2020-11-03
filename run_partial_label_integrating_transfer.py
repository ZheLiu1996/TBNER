#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle as pkl
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from pytorch_pretrained_bert.modeling import BertModel
from model import sequence_label_model
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from utils import sent_prf_cal
from tqdm import tqdm
import os
from torchcrf import CRF


np.random.seed(1337)
mySeed = np.random.RandomState (1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


if __name__ == '__main__':
    ########参数设置############
    root = r'/my_data_new/'
    write_root = r'/NCBI-disease-my/' 
    train_pkl = root + r'train_CDWA.pkl'
    dev_pkl = root + r'dev_CDWA.pkl'
    write_path=write_root+'predict_partial_label_integrating_transfer/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
        print(write_path,'succeed')
    predict_path=write_path
    record_path=write_root+'prf_ner_all.txt'
    model_save_path=write_root+'model_partial_label_integrating_transfer'
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    epoch_num=20#60
    batch_size=4#4#8#16#32
    learn_rate=3e-5#5e-4#0.005#1e-3
    hidden_dim=768
    tagset_size=3
    max_seq_length=512

    model_bert_load_path='model_partial_label_integrating/model_bert.pth'
    model_sequence_label_load_path='model_partial_label_integrating/model_sequence_label.pth'
    model_crf_load_path='model_partial_label_integrating/model_crf.pth'

    bert_file='/biobert_v1.1_pubmed/'
    ########读取训练集语料###########
    with open(train_pkl, "rb") as f:
        train_data,_=pkl.load(f)
    print('读取训练集完成')
    train_count=len(train_data)
    ########读取验证集###########
    with open(dev_pkl, "rb") as f:
        dev_data,_,_=pkl.load(f)
    print('读取验证集完成')
    dev_count=len(dev_data)
    print(f'dev_count:{dev_count}')
    #########加载模型###############
    model_bert = BertModel.from_pretrained(bert_file)
    model_bert.load_state_dict(torch.load(model_bert_load_path))
    model_bert.cuda()
    model_sequence_label=sequence_label_model(hidden_dim,tagset_size)
    model_sequence_label.load_state_dict(torch.load(model_sequence_label_load_path))
    model_sequence_label.cuda()
    model_crf=CRF(tagset_size,batch_first=True)
    model_crf.load_state_dict(torch.load(model_crf_load_path))
    model_crf.cuda(device=0)

    parameters=[]
    for param in model_bert.parameters():
        parameters.append(param)
    for param in model_sequence_label.parameters():
        parameters.append(param)
    # for param in model_crf.parameters():
    #     parameters.append(param)
    optimizer=optim.RMSprop(parameters, lr=learn_rate)
    # optimizer=optim.Adam(parameters, lr=learn_rate)
    # optimizer=optim.Adagrad(parameters, lr=learn_rate)    
    # optimizer=optim.SGD(parameters, lr=learn_rate)
    loss_cal=torch.nn.CrossEntropyLoss()
    ########训练和测试##############
    dev_index=list(range(dev_count))    
    max_f_dev=0.0    
    train_step_count=0
    for epoch in range(epoch_num):
        #############训练语料##############
        model_bert.train()
        model_sequence_label.train()
        model_crf.train()
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        # train_dataloader = DataLoader(train_data, batch_size=batch_size)
        
        for batch in tqdm(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids,label_ids,_ = batch
            all_encoder_layers, _ = model_bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
            out=model_sequence_label(all_encoder_layers,True)
            loss=model_crf(out,label_ids,input_mask.byte(),reduction='mean')
            loss=-1*loss
            # out=torch.masked_select(out,input_mask.unsqueeze(2).repeat(1,1,tagset_size).byte())
            # print(out)
            # out=out.reshape(-1,tagset_size)
            # label_ids=torch.masked_select(label_ids,input_mask.byte())
            # loss=loss_cal(out,label_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ################验证##############
        model_bert.eval()
        model_sequence_label.eval()
        # model_crf.eval()        
        dev_dataloader=DataLoader(dev_data)
        predict=[]
        gold=[]
        for batch in tqdm(dev_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids,_ = batch
            all_encoder_layers, _ = model_bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
            out=model_sequence_label(all_encoder_layers,False)
            # decoded=np.argmax(out.squeeze(0).data.cpu().numpy(),axis=1)

            decoded=model_crf.decode(out)
            decoded=decoded[0]
            len_count=torch.sum(input_mask)-1
            predict.extend(decoded[1:len_count])
            label_ids=torch.masked_select(label_ids,input_mask.byte()).data.cpu().numpy()
            gold.extend(label_ids[1:len_count])
        ###########写入验证集结果################
        predict_file=predict_path+'predict_dev_'+str(train_step_count)+'.pkl'
        p_dev,r_dev,f1_dev=sent_prf_cal(predict,gold)
        dev_record=str(epoch)+'\t'+str(p_dev)+'\t'+str(r_dev)+'\t'+str(f1_dev)
        print('验证集:',dev_record)      
        with open(predict_file,'wb')as f:
            pkl.dump(predict,f,-1)
        if float(f1_dev)>max_f_dev:
            max_f_dev=float(f1_dev)
            torch.save(model_bert.state_dict(), model_save_path+'/model_bert.pth')
            torch.save(model_sequence_label.state_dict(), model_save_path+'/model_sequence_label.pth')
            torch.save(model_crf.state_dict(), model_save_path+'/model_crf.pth')            
        with open(record_path,'a',encoding='utf-8')as w:
            w.write(dev_record+'\n')
        train_step_count+=1
    with open(record_path,'a',encoding='utf-8')as w:
        w.write(str(max_f_dev)+'\n')        
    print('max_f_dev:',max_f_dev)    