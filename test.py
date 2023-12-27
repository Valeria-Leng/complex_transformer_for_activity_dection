import numpy as np
import torch
import scipy.io as io
import torch.optim as optim
from usr_encoder import USREncoder
from logit import Out
import matplotlib.pyplot as plt
from data_generation import datasetGeneration
import os
from tqdm import tqdm
import math
from torch.profiler import profile, record_function, ProfilerActivity
import time
from config_test import get_cfg
from model_training import COMPLEX_SAD

cfg = get_cfg()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    #model
    model = COMPLEX_SAD(cfg).to(device)
    # N=120
    # K=int(N/10)
    # M=64
    # sigma2s = np.ones([N,1])
    # Pmax=23
    # noisePowerN = -14.188607425695352+(23-Pmax)
    if cfg.test_flag:
        checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        print('Load ckpt successfully!')
        
    print(cfg.N, cfg.K)

    tA, tCov, tsupp = datasetGeneration(cfg.TSIZE,cfg)
    tA = tA.to(device)
    tCov = tCov.to(device)
    tsupp = tsupp.to(device)


    start = time.time()
    model.eval()
    with torch.no_grad():
        tlogit = model(tA,tCov,mask = None)
        tprob = torch.sigmoid(tlogit)
        tpred = torch.zeros(tprob.size()).to(device)
        tpred[tprob>=0.5]=1
        err = cfg.N*cfg.TSIZE-(tpred==tsupp).sum().sum().item()
    end = time.time()
    print('time=', end-start)
    print('Test set: Error rate: {}/{} ({:.2f}%)'.format(err, cfg.N*cfg.TSIZE, 100. * err/cfg.N/cfg.VSIZE))

    # NUM = 200
    # min_g = tprob[tprob>0].min().item()
    # max_g = tprob.max().item()
    # set_1 = np.exp(np.linspace(np.log(min_g*0.95),np.log(max_g*1.05),int(NUM/2)))
    # set_2 = np.linspace(min_g*0.95,max_g*1.05,int(NUM/2))
    # TH = np.sort(np.concatenate((set_1,set_2)))
    # pm_cov = np.zeros([NUM,1])
    # pf_cov = np.zeros([NUM,1])

    # for idx in tqdm(range(NUM)):
    #     th = TH[idx]
    #     th_supp = np.zeros(tsupp.size())
    #     th_supp[(tprob>=th).cpu()]=1
    #     detect = th_supp*(tsupp.cpu().numpy())
    #     pm_cov[idx] = 1-detect.sum()/cfg.K/cfg.TSIZE
    #     falarm = th_supp-detect
    #     pf_cov[idx] = falarm.sum()/(cfg.N-cfg.K)/cfg.TSIZE
    # io.savemat('complex_N50K2M32L8_Layer5_em128/complex_data_3.mat',  {'pm': pm_cov, 'pf':pf_cov})
    # plt.loglog(pm_cov, pf_cov)
    # plt.savefig("complex_N50K2M32L8_Layer5_em128/pm-pf_3.png")
