#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pickle as pkl
from tqdm import tqdm
import  numpy as np
def calculate_sim(train_pkl,test_pkl):
    with open(train_pkl, "rb") as f:
        train_cls=pkl.load(f)
    with open(test_pkl, "rb") as f:
        test_cls=pkl.load(f)
    sim_score=[]
    for t_cls in tqdm(test_cls):
        
        sims=np.sum(t_cls*train_cls,1)
        # sims=[sum(t_cls*ele) for ele in train_cls]
        sim_score.append(max(sims))
    return sim_score


if __name__ == '__main__':
    cls_root=r'cls/'   
    cls_distant_pkl=r'../chemdner_corpus_bert/cls/distant_cls.pkl'
    cls_train_pkl=cls_root+r'train_cls.pkl'
    # cls_dev_pkl=cls_root+r'dev_cls.pkl'
    cls_test_pkl=cls_root+r'test_cls.pkl'
    train_test_sim_pkl=cls_root+'train_test_sim.pkl'
    distant_test_sim_pkl=cls_root+'distant_test_sim.pkl'
    # sim_score=calculate_sim(cls_train_pkl,cls_test_pkl)  
    # with open(train_test_sim_pkl,'wb')as f:
    #     pkl.dump(sim_score,f,-1)
    # print(np.mean(sim_score))
      
    # sim_score=calculate_sim(cls_distant_pkl,cls_test_pkl)
    # with open(distant_test_sim_pkl,'wb')as f:
    #     pkl.dump(sim_score,f,-1)
    # print(np.mean(sim_score))

    # test_distant_sim_pkl=cls_root+'test_distant_sim.pkl'
    # sim_score=calculate_sim(cls_test_pkl,cls_distant_pkl)
    # with open(test_distant_sim_pkl,'wb')as f:
    #     pkl.dump(sim_score,f,-1)
    sim_score=calculate_sim(cls_train_pkl,cls_distant_pkl)
    print(np.mean(sim_score))
    
    
