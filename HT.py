import numpy as np
import torch
from torch import nn
import torch.optim as optim
from usr_encoder import USREncoder
from logit import Out
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import math
import time
import scipy.io as io
from transformer import Transformer_Encoder
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

def sensingMatrixDesign(N,J,L,type):
    Ne = N*2**J
    if type=='Gaussian':
        A = (np.random.normal(loc=0, scale=1, size=(L, Ne))+1j*np.random.normal(loc=0, scale=1, size=(L, Ne)))*np.sqrt(0.5)
    else:
        raise NotImplementedError
    return A

def channelGeneration(N,M,sigma2s,channel):

    if channel == 'noniid':
        Lp=2
        theta = np.random.uniform(low=-math.pi/6, high=math.pi/6, size=(N,Lp))
        alfa = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(N, Lp)) + 1j*np.random.normal(loc=0, scale=1, size=(N, Lp)))
        a1=range(M)
        a2=1j*math.pi*np.sin(theta)
        a3=np.kron(a1,a2).reshape((N,Lp,M))
        a = np.exp(a3)
        hh=np.expand_dims(alfa,2).repeat(M,axis=2)*a
        H = hh.sum(axis=1)/np.sqrt(Lp)
    else:
        H = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(N, M)) + 1j*np.random.normal(loc=0, scale=1, size=(N, M)))

    H = np.sqrt(np.tile(sigma2s,(1,M)))*H # sigma2s large-scale fading component
    return H

def signalGeneration(N,K,L,J,M,H,txPowerMax,noisePower):
    user_idx = np.random.permutation(N)
    user_idx[0:K] = np.sort(user_idx[0:K])
    user_supp = np.zeros([N])
    user_supp[user_idx[0:K]] = 1

    # Data of active users
    data_idx = np.random.randint(2**J,size=K) # the data indices for active users

    # Combined support
    supp = np.zeros([2**J*N])
    supp[user_idx[0:K]*2**J + data_idx] = 1

    # Signal generation
    Ne = N*2**J
    Heff = np.repeat(H, 2**J, axis=0) # eff. channel; channel for sequences of one users are equal.
    x = np.zeros([Ne,M])+1j*np.zeros([Ne,M])
    x[(supp==1),:] = Heff[(supp==1),:]

    # Noise setup with power control
    sigma2n = (10**((noisePower)/10))
    txPower = 10**(txPowerMax/10)
    sigma2n = sigma2n/txPower
    return x,user_supp,supp,user_idx,data_idx,sigma2n

def COMPLEX_datasetGeneration(datasize,N,J,L,matrx_Type,M,K,txPowerN,noisePowerN,location,channel,use_cov):
    if location=='uniform':
        A_data = np.zeros([datasize, N, 2*L+1])
    else:
        A_data = np.zeros([datasize, N, L], dtype=complex)
    if use_cov == True:
        Cov_data = np.zeros([datasize, L*L], dtype=complex)
    else:
        Cov_data = np.zeros([datasize, 2*L*M])
    supp_data = np.zeros([datasize, N])

    for mc in range(datasize):

        # Sequence generation
        A = sensingMatrixDesign(N,J,L,matrx_Type)


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
        H = channelGeneration(N,M,sigma2s,channel)


        # Sparse signal
        x,user_supp,supp,user_idx,data_idx,sigma2n = signalGeneration(N,K,L,J,M,H,txPowerN,noisePowerN)

        # Additive noise
        w = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(L,M))+1j*np.random.normal(loc=0, scale=1, size=(L,M)))*np.sqrt(sigma2n)

        # System model
        y = np.dot(A,x)+ w
        if use_cov == True:
            Cov = 1/M*np.dot(y,y.T.conj()).reshape(-1)
        else:
            Cov = y.reshape(-1)


        # Cov_data[mc,:] = np.hstack((np.real(Cov), np.imag(Cov)))
        Cov_data[mc,:] = Cov
        supp_data[mc,:] = supp

        if location=='uniform':
            A_data[mc,:,:-1] = np.vstack((np.real(A), np.imag(A))).T
            A_data[mc,:,-1] = np.squeeze(distance)
        else: #control
            # A_data[mc,:,:] = np.vstack((np.real(A), np.imag(A))).T #[real imag]
            A_data[mc,:,:] = A.T #why A.T
    # print(Cov_data.imag)
    dA = torch.from_numpy(A_data).type(torch.complex64).to(device)
    dCov = torch.from_numpy(Cov_data).type(torch.complex64).to(device)
    dsupp = torch.FloatTensor(supp_data).to(device)

    return dA, dCov, dsupp

def datasetGeneration(datasize,N,J,L,matrx_Type,M,K,txPowerN,noisePowerN,location,channel,use_cov):   
    if location=='uniform':
        A_data = np.zeros([datasize, N, 2*L+1])
    else: 
        A_data = np.zeros([datasize, N, 2*L])#
    if use_cov == True:
        Cov_data = np.zeros([datasize, 2*L*L])#
    else:
        Cov_data = np.zeros([datasize, 2*L*M])
    supp_data = np.zeros([datasize, N])
        
    for mc in range(datasize):

        # Sequence generation
        A = sensingMatrixDesign(N,J,L,matrx_Type)
    

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
        H = channelGeneration(N,M,sigma2s,channel)
        

        # Sparse signal
        x,user_supp,supp,user_idx,data_idx,sigma2n = signalGeneration(N,K,L,J,M,H,txPowerN,noisePowerN)

        # Additive noise
        w = np.sqrt(1/2)*(np.random.normal(loc=0, scale=1, size=(L,M))+1j*np.random.normal(loc=0, scale=1, size=(L,M)))*np.sqrt(sigma2n)

        # System model
        y = np.dot(A,x)+ w
        if use_cov == True:        
            Cov = 1/M*np.dot(y,y.T.conj()).reshape(-1)
        else:
            Cov = y.reshape(-1)    

        
        Cov_data[mc,:] = np.hstack((np.real(Cov), np.imag(Cov)))
        supp_data[mc,:] = supp

        if location=='uniform':
            A_data[mc,:,:-1] = np.vstack((np.real(A), np.imag(A))).T
            A_data[mc,:,-1] = np.squeeze(distance)
        else:
            A_data[mc,:,:] = np.vstack((np.real(A), np.imag(A))).T  #

    dA = torch.FloatTensor(A_data).to(device)
    dCov = torch.FloatTensor(Cov_data).to(device)
    dsupp = torch.FloatTensor(supp_data).to(device)
    
    return dA, dCov, dsupp 


class COMPLEX_SAD(nn.Module):
    def __init__(self,L,EMx,EMy,EM,FF,NodeSIZE):
        super(COMPLEX_SAD,self).__init__()

        if use_cov == True:
            # self.cov = torch.nn.Sequential(
            #      nn.Linear(2*L*L, FF),
            #      nn.ReLU(),
            #      nn.Linear(FF, EMy)
            #    )

            # self.cov = nn.Linear(2*L*L, EMy)
            self.cov= ComplexLinear(L*L, EMy//2)
        else:
            self.cov = ComplexLinear(L*M, EMy//2)


        # self.usr1 = USREncoder(
        #     n_heads=8,
        #     embed_dim=EM,
        #     y_dim=EMy,
        #     x_dim=EMx,
        #     feed_forward_hidden = FF,
        #     node_dim = NodeSIZE,
        #     normalization='batch'
        #                )
        self.usr1 = Transformer_Encoder(embed_dim=EMy//2,
         node_dim=L, num_heads=4)
        # self.usr2 = USREncoder(
        #     n_heads=8,
        #     embed_dim=EM,
        #     y_dim=EMy,
        #     x_dim=EMx,
        #     feed_forward_hidden = FF,
        #     normalization='batch'
        #                )
        self.usr2 = Transformer_Encoder(embed_dim=EMy//2,
         A_dim=EMy//2, num_heads=4)
        
        self.usr3 = Transformer_Encoder(embed_dim=EMy//2,
         A_dim=EMy//2, num_heads=4)
        
        self.usr4 = Transformer_Encoder(embed_dim=EMy//2,
         A_dim=EMy//2, num_heads=4)
        
        self.usr5 = Transformer_Encoder(embed_dim=EMy//2,
         A_dim=EMy//2, num_heads=4)
        
        
        # self.usr3 = USREncoder(
        #     n_heads=8,
        #     embed_dim=EM,
        #     y_dim=EMy,
        #     x_dim=EMx,
        #     feed_forward_hidden = FF,
        #     normalization='batch'
        #                )

        # self.usr4 = USREncoder(
        #      n_heads=8,
        #      embed_dim=EM,
        #      y_dim=EMy,
        #      x_dim=EMx,
        #      feed_forward_hidden = FF,
        #      normalization='batch'
        #                 )

        # self.usr5 = USREncoder(
        #      n_heads=8,
        #      embed_dim=EM,
        #      y_dim=EMy,
        #      x_dim=EMx,
        #      feed_forward_hidden = FF,
        #      normalization='batch'
        #                 )

        # self.usr6 = USREncoder(
        #      n_heads=8,
        #      embed_dim=EM,
        #      y_dim=EMy,
        #      feed_forward_hidden = FF,
        #      normalization='batch'
        #                 )

       # self.usr7 = USREncoder(
       #     n_heads=8,
       #     embed_dim=EM,
       #     y_dim=EMy,
       #     feed_forward_hidden = FF,
       #     normalization='batch'
      #                 )

       # self.usr8 = USREncoder(
       #     n_heads=8,
       #     embed_dim=EM,
       #     y_dim=EMy,
       #     feed_forward_hidden = FF,
       #     normalization='batch'
       #                )

        self.out = Out(
            n_heads=8,
            embed_dim=EM,
            y_dim=EMy,
            x_dim=EMx
                 )

    def forward(self,x,y,mask):
        y = self.cov(y).unsqueeze(1)
        x, y = self.usr1(x,y)
        
        x, y = self.usr2(x,y)
        # x, y = self.usr3(x,y)
        # x, y = self.usr4(x,y)
        # x, y = self.usr5(x,y)
        x = torch.view_as_real(x).view(x.size(0), x.size(1), -1)
        y = torch.view_as_real(y).view(y.size(0), y.size(1), -1)
        # x, y = self.usr6(x,y,mask)
      #  x, y = self.usr7(x,y,mask)
      #  x, y = self.usr8(x,y,mask)
        output = self.out(x,y)
        return output


class SAD(nn.Module):
    def __init__(self,L,EMx,EMy,EM,FF,NodeSIZE):
        super(SAD,self).__init__()
        
        if use_cov == True: 
            # self.cov = torch.nn.Sequential(
            #      nn.Linear(2*L*L, FF),
            #      nn.ReLU(),
            #      nn.Linear(FF, EMy)
            #    )
            self.cov = nn.Linear(2*L*L, EMy) #
        else:
            self.cov = nn.Linear(2*L*M, EMy)

                
        self.usr1 = USREncoder(
            n_heads=8,
            embed_dim=EM,
            y_dim=EMy,
            x_dim=EMx, 
            feed_forward_hidden = FF,
            node_dim = NodeSIZE,
            normalization='batch'
                       )    
        
        self.usr2 = USREncoder(
            n_heads=8,
            embed_dim=EM,
            y_dim=EMy,
            x_dim=EMx,
            feed_forward_hidden = FF,
            normalization='batch'
                       )
        
        self.usr3 = USREncoder(
            n_heads=8,
            embed_dim=EM,
            y_dim=EMy,
            x_dim=EMx, 
            feed_forward_hidden = FF,
            normalization='batch'
                       )
     
        self.usr4 = USREncoder(
             n_heads=8,
             embed_dim=EM,
             y_dim=EMy,
             x_dim=EMx, 
             feed_forward_hidden = FF,
             normalization='batch'
                        )
             
        self.usr5 = USREncoder(
             n_heads=8,
             embed_dim=EM,
             y_dim=EMy,
             x_dim=EMx, 
             feed_forward_hidden = FF,
             normalization='batch'
                        )
      
        # self.usr6 = USREncoder(
        #      n_heads=8,
        #      embed_dim=EM,
        #      y_dim=EMy,
        #      feed_forward_hidden = FF,
        #      normalization='batch'
        #                 )
        
       # self.usr7 = USREncoder(
       #     n_heads=8,
       #     embed_dim=EM,
       #     y_dim=EMy,
       #     feed_forward_hidden = FF,
       #     normalization='batch'
      #                 )
        
       # self.usr8 = USREncoder(
       #     n_heads=8,
       #     embed_dim=EM,
       #     y_dim=EMy,
       #     feed_forward_hidden = FF,
       #     normalization='batch'
       #                )    
        
        self.out = Out(
            n_heads=8,
            embed_dim=EM,
            y_dim=EMy,
            x_dim=EMx
                 )
         
    def forward(self,x,y,mask):
        y = self.cov(y).unsqueeze(1)
        x, y = self.usr1(x,y,mask)      
        x, y = self.usr2(x,y,mask)      
        # x, y = self.usr3(x,y,mask)      
        # x, y = self.usr4(x,y,mask)      
        # x, y = self.usr5(x,y,mask)      
        # x, y = self.usr6(x,y,mask) 
      #  x, y = self.usr7(x,y,mask)  
      #  x, y = self.usr8(x,y,mask)       
        output = self.out(x,y)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_flag = False  #测试标志，True时加载保存好的模型进行测试
filename = './N20K2M32L8_Layer2_em128'
path_checkpoint = filename+'/ckpt_best_0.pth'  # 断点路径
EMx = 128
EMy = 128
EM = 128
K = 10

N = 100
L = 12
M = 64

J = 0
D = 500
FF = 512
txPower = 23 # dBm
noisePower = -99 # dBm
set_random_seed(10)

location = 'control'
channel = 'iid'
use_cov = True
sigma2s = np.ones([N,1])
txPowerN = 0
noisePowerN = noisePower + 15.3 + 37.6*np.log10(D*math.tan(math.pi/6)) - txPower
matrx_Type = 'Gaussian'
VSIZE = 512
BSIZE = 256
TSIZE = 5000
if location =='uniform':
    NodeSIZE = 2*L+1
else: #
    NodeSIZE = 2*L
BNUM = 5000
EP = 100

#ma = torch.eye(N).to(device)
#mb = torch.ones(N).unsqueeze(0).to(device)
#ma = torch.cat((ma,mb),0)
#mc = torch.ones(N+1).unsqueeze(1).to(device)
#mask = ~torch.cat((ma,mc),1).bool()
mask = None

if __name__ == '__main__':
    vA, vCov, vsupp = COMPLEX_datasetGeneration(VSIZE,N,J,L,matrx_Type,M,K,txPowerN,noisePowerN,location,channel,use_cov)
    print(vCov.imag)
    model = COMPLEX_SAD(L,EMx,EMy,EM,FF,NodeSIZE).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(N-K)/K]).to(device))
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1)

    if test_flag:
        checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    #  lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1,last_epoch=0)

        print('Load epoch {} successfully'.format(start_epoch))
    else:
        start_epoch = -1
        print('From epoch 0')
    for epoch in range(start_epoch+1, EP):
        model.train()
        train_loss = 0
        for i in tqdm(range(BNUM)):
            bA, bCov, bsupp = COMPLEX_datasetGeneration(BSIZE,N,J,L,matrx_Type,M,K,txPowerN,noisePowerN,location,channel,use_cov)
            optimizer.zero_grad()
            logit = model(bA,bCov,mask)
            loss = criterion(logit, bsupp)
            loss.backward()
            optimizer.step()
            train_loss += (2*K/N)*loss.item()
        loss_mean = train_loss / BNUM
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean))

        model.eval()
        with torch.no_grad():
            #vA, vCov, vsupp = bA, bCov, bsupp
            vlogit = model(vA,vCov,mask)
            vloss = criterion(vlogit, vsupp).item()
            vprob = torch.sigmoid(vlogit)
            vpred = torch.zeros(vprob.size()).to(device)
            vpred[vprob>=0.5]=1
            err = N*VSIZE-(vpred==vsupp).sum().sum().item()

        print('Test set: Loss: {:.4f}, Error rate: {}/{} ({:.2f}%)'.format((2*K/N)*vloss, err, N*VSIZE, 100. * err/N/VSIZE))
        print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
        print('\n')

        lr_schedule.step()

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        if not os.path.isdir(filename):
            os.mkdir(filename)
        torch.save(checkpoint, filename+'/ckpt_best_%s.pth' % (str(epoch)))


    #test
    N=120
    K=int(N/10)
    M=64
    sigma2s = np.ones([N,1])
    Pmax=23
    noisePowerN = -14.188607425695352+(23-Pmax)
    tA, tCov, tsupp = COMPLEX_datasetGeneration(TSIZE,N,J,L,matrx_Type,M,K,txPowerN,noisePowerN,location,channel,use_cov)


    start = time.time()
    model.eval()
    with torch.no_grad():
        tlogit = model(tA,tCov,mask)
        tprob = torch.sigmoid(tlogit)
    end = time.time()
    print('time=', end-start)

    NUM = 200
    min_g = tprob[tprob>0].min().item()
    max_g = tprob.max().item()
    set_1 = np.exp(np.linspace(np.log(min_g*0.95),np.log(max_g*1.05),int(NUM/2)))
    set_2 = np.linspace(min_g*0.95,max_g*1.05,int(NUM/2))
    TH = np.sort(np.concatenate((set_1,set_2)))
    pm_cov = np.zeros([NUM,1])
    pf_cov = np.zeros([NUM,1])

    for idx in range(NUM):
        th = TH[idx]
        th_supp = np.zeros(tsupp.size())
        th_supp[(tprob>=th).cpu()]=1
        detect = th_supp*(tsupp.cpu().numpy())
        pm_cov[idx] = 1-detect.sum()/K/TSIZE
        falarm = th_supp-detect
        pf_cov[idx] = falarm.sum()/(N-K)/TSIZE
    io.savemat('complex/complex_data_3.mat',  {'pm': pm_cov, 'pf':pf_cov})
    plt.loglog(pm_cov, pf_cov)
    plt.savefig("complex/pm-pf_3.png")
