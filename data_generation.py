import numpy as np
import torch
from torch import nn
import pickle
from logit import Out
import matplotlib.pyplot as plt
import random
import os
import math
from config import get_cfg
from tqdm import tqdm

cfg = get_cfg()
# if cfg.test_flag:
#     from config_test import get_cfg
#     cfg = get_cfg()

sigma2s = np.ones([cfg.N,1])
# print(cfg)

def sensingMatrixDesign(cfg):
    #N,J,L,type
    Ne = cfg.N*2**cfg.J
    if cfg.matrx_Type=='Gaussian':
        A = (np.random.normal(loc=0, scale=1, size=(cfg.L, Ne))+1j*np.random.normal(loc=0, scale=1, size=(cfg.L, Ne)))*np.sqrt(0.5)
    else:
        raise NotImplementedError
    return A

def set_random_seed(seed = 10,deterministic=False,benchmark=False):
    random.seed(seed)
    np.random.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return

def channelGeneration(cfg):

    if cfg.channel == 'noniid':
        Lp=2
        theta = np.random.uniform(low=-math.pi/6, high=math.pi/6, size=(cfg.N,Lp))
        alfa = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(cfg.N, Lp)) + 1j*np.random.normal(loc=0, scale=1, size=(cfg.N, Lp)))
        a1=range(cfg.M)
        a2=1j*math.pi*np.sin(theta)
        a3=np.kron(a1,a2).reshape((cfg.N,Lp,cfg.M))
        a = np.exp(a3)
        hh=np.expand_dims(alfa,2).repeat(cfg.M,axis=2)*a
        H = hh.sum(axis=1)/np.sqrt(Lp)
    else:
        H = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(cfg.N, cfg.M)) + 1j*np.random.normal(loc=0, scale=1, size=(cfg.N, cfg.M)))

    H = np.sqrt(np.tile(sigma2s,(1,cfg.M)))*H # sigma2s large-scale fading component
    return H

def signalGeneration(H,cfg):
    user_idx = np.random.permutation(cfg.N)
    user_idx[0:cfg.K] = np.sort(user_idx[0:cfg.K])
    user_supp = np.zeros([cfg.N])
    user_supp[user_idx[0:cfg.K]] = 1

    # Data of active users
    data_idx = np.random.randint(2**cfg.J,size=cfg.K) # the data indices for active users

    # Combined support
    supp = np.zeros([2**cfg.J*cfg.N])
    supp[user_idx[0:cfg.K]*2**cfg.J + data_idx] = 1

    # Signal generation
    Ne = cfg.N*2**cfg.J
    Heff = np.repeat(H, 2**cfg.J, axis=0) # eff. channel; channel for sequences of one users are equal.
    x = np.zeros([Ne,cfg.M])+1j*np.zeros([Ne,cfg.M])
    x[(supp==1),:] = Heff[(supp==1),:]

    # Noise setup with power control
    sigma2n = (10**((cfg.noisePower)/10))
    txPower = 10**(cfg.txPowerN/10)
    sigma2n = sigma2n/txPower
    return x,user_supp,supp,user_idx,data_idx,sigma2n

   

def datasetGeneration(datasize, cfg):
    # N,J,L,matrx_Type,M,K,txPowerN,noisePowerN,location,channel,use_cov
    if cfg.complex:
        if cfg.location=='uniform':
            A_data = np.zeros([datasize, cfg.N, 2*cfg.L+1])
        else:
            A_data = np.zeros([datasize, cfg.N, cfg.L], dtype=complex)
        if cfg.use_cov == True:
            Cov_data = np.zeros([datasize, cfg.L*cfg.L], dtype=complex)
        else:
            Cov_data = np.zeros([datasize, 2*cfg.L*cfg.M])
    else:
        A_data = np.zeros([datasize, cfg.N, 2*cfg.L])
        Cov_data = np.zeros([datasize, 2*cfg.L*cfg.L])


    supp_data = np.zeros([datasize, cfg.N])

    for mc in range(datasize):

        # Sequence generation
        A = sensingMatrixDesign(cfg)


        # if location=='uniform':
        #     distance = np.zeros([N,1])
        #     x_Range = D*math.tan(math.pi/6)*3/2
        #     y_Range = D

        #     for iUE in range(N):
        #         RD = 0
        #         while RD<=50:
        #             x_Posi = np.random.uniform(x_Range*(-2/3), x_Range*(1/3))
        #             y_Posi = np.random.uniform(y_Range*(-1/2), y_Range*(1/2))

        #             if y_Posi > x_Posi*math.tan(math.pi/3)+D:
        #                 x_Posi = x_Posi+x_Range
        #                 y_Posi = y_Posi-y_Range/2
        #             elif y_Posi < -x_Posi*math.tan(math.pi/3)-D:
        #                 x_Posi = x_Posi+x_Range
        #                 y_Posi = y_Posi+y_Range/2

        #             RD = np.sqrt(x_Posi**2 + y_Posi**2)
        #         distance[iUE,0]=RD

        #     sigma2dB = 128-15.3-37.6*np.log10(distance)
        #     sigma2s = 10**(sigma2dB/10)


        # Gaussian channel
        H = channelGeneration(cfg)


        # Sparse signal
        x,user_supp,supp,user_idx,data_idx,sigma2n = signalGeneration(H,cfg)

        # Additive noise
        w = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(cfg.L,cfg.M))+1j*np.random.normal(loc=0, scale=1, size=(cfg.L,cfg.M)))*np.sqrt(sigma2n)

        # System model
        y = np.dot(A,x)+ w
        if cfg.use_cov == True:
            Cov = 1/cfg.M*np.dot(y,y.T.conj()).reshape(-1)
        else:
            Cov = y.reshape(-1)


        # Cov_data[mc,:] = np.hstack((np.real(Cov), np.imag(Cov)))
        
        supp_data[mc,:] = supp

        # if cfg.location=='uniform':
        #     A_data[mc,:,:-1] = np.vstack((np.real(A), np.imag(A))).T
        #     A_data[mc,:,-1] = np.squeeze(distance)
        # else: #control
        #     # A_data[mc,:,:] = np.vstack((np.real(A), np.imag(A))).T #[real imag]
        #     A_data[mc,:,:] = A.T #why A.T
    # print(Cov_data.imag)
        if cfg.complex:
            A_data[mc,:,:] = A.T 
            Cov_data[mc,:] = Cov
            dA = torch.from_numpy(A_data).type(torch.complex64)
            dCov = torch.from_numpy(Cov_data).type(torch.complex64)
            dsupp = torch.FloatTensor(supp_data)
        else:
            A_data[mc,:,:] = np.vstack((np.real(A), np.imag(A))).T #[real imag]
            Cov_data[mc,:] = np.hstack((np.real(Cov), np.imag(Cov)))
            dA = torch.FloatTensor(A_data)
            dCov = torch.FloatTensor(Cov_data)
            dsupp = torch.FloatTensor(supp_data)
    # # print(vCov.imag)


    return dA, dCov, dsupp #np format

if __name__=='__main__':
    # vA, vCov, vsupp = datasetGeneration(cfg.VSIZE, cfg)
    # print(vCov.shape)
    dA = []
    dCov = []
    dsupp = []
    for i in tqdm(range(cfg.BNUM)):
        bA, bCov, bsupp = datasetGeneration(cfg.BSIZE, cfg) #return data in numpy array format
        dA.append(bA)
        dCov.append(bCov)
        dsupp.append(bsupp)
    data_list = list(zip(dA, dCov, dsupp))
    #save as pkl
    with open(cfg.pkl_path, 'wb') as f:
        pickle.dump(data_list, f)
    print('==========> The training data has been saved!')
    # np.save('training_data/data.npy', data)


    ##############################
    # dA = torch.from_numpy(A_data).type(torch.complex64)
    # dCov = torch.from_numpy(Cov_data).type(torch.complex64)
    # dsupp = torch.FloatTensor(supp_data)

