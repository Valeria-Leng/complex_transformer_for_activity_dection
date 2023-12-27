import numpy as np
import torch
import pickle
import time
from config import get_cfg
from tqdm import tqdm
from data_generation import datasetGeneration

cfg = get_cfg()

def training_dataloader(cfg):
    with open(cfg.pkl_path, 'rb') as f:
        training_data = pickle.load(f)
    A_data, Cov_data, supp_data = zip(*training_data) #unzip
    if cfg.complex:

        dA = torch.tensor(A_data).type(torch.complex64)
        dCov = torch.tensor(Cov_data).type(torch.complex64)
        dsupp = torch.FloatTensor(supp_data)
    else:
        dA = torch.FloatTensor(A_data)
        dCov = torch.FloatTensor(Cov_data)
        dsupp = torch.FloatTensor(supp_data)
    # print(dA.shape)
    # dataset = Dataset((dA, dCov, dsupp))
    # dataloader = DataLoader(dataset, batch_size=256, num_workers=8)

    return dA, dCov, dsupp
if __name__=='__main__':
    start = time.time()
    dA, dCov, dsupp = training_dataloader(cfg)
    vA, vCov, vsupp = datasetGeneration(cfg.VSIZE,cfg)
    print(vCov.imag)
    # model = COMPLEX_SAD(L,EMx,EMy,EM,FF,NodeSIZE).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(N-K)/K]).to(device))
    # #criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1)
    for epoch in range(1, 100):
        
        for index in tqdm(range(cfg.BNUM)):
            print(dCov[index].shape)
            now = time.time()
            delta = now-start
            print(now-start)

