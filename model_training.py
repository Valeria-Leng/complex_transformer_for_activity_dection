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
from torch.profiler import profile, record_function, ProfilerActivity
import time
from config import get_cfg
from transformer import Transformer_Encoder, ComplexLinear, Complex_Out 
# from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear
from data_generation import datasetGeneration
from dataloader import training_dataloader
# from torchsummary import summary
# from torchinfo import summary

cfg = get_cfg()
# print(cfg.complex)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class binary_ce_loss(torch.nn.Module):
    def __init__(self, pos_weight=None):
        super(binary_ce_loss, self).__init__()
        if pos_weight is not None:
          self.pos_weight = pos_weight
         
        else:
          self.pos_weight = 1

    def forward(self, input, target):
        input = input.view(input.shape[0], -1) 
        target = target.view(target.shape[0], -1)
        loss = -(self.pos_weight * target * torch.log(input + 1e-8) + (1 - target) * torch.log(1 - input + 1e-8))
        return torch.mean(loss)

class SAD(nn.Module):
    def __init__(self, cfg):
        #L,EMx,EMy,EM,FF,NodeSIZE
        super(SAD,self).__init__()
        
        if cfg.use_cov == True: 
            # self.cov = torch.nn.Sequential(
            #      nn.Linear(2*L*L, FF),
            #      nn.ReLU(),
            #      nn.Linear(FF, EMy)
            #    )
            self.cov = nn.Linear(2*cfg.L*cfg.L, cfg.EMy//2) #
        else:
            self.cov = nn.Linear(2*cfg.L*cfg.M, cfg.EMy//2)

                
        self.usr1 = USREncoder(
            n_heads=4,
            embed_dim=cfg.EM//2,
            y_dim=cfg.EMy//2,
            x_dim=cfg.EMx//2, 
            feed_forward_hidden = cfg.FF//2,
            node_dim = 2*cfg.L,
            normalization='batch'
                       )    
        
        self.usr2 = USREncoder(
            n_heads=4,
            embed_dim=cfg.EM//2,
            y_dim=cfg.EMy//2,
            x_dim=cfg.EMx//2,
            feed_forward_hidden = cfg.FF//2,
            normalization='batch'
                       )
        
        self.usr3 = USREncoder(
            n_heads=4,
            embed_dim=cfg.EM//2,
            y_dim=cfg.EMy//2,
            x_dim=cfg.EMx//2, 
            feed_forward_hidden = cfg.FF//2,
            normalization='batch'
                       )
     
        self.usr4 = USREncoder(
             n_heads=4,
             embed_dim=cfg.EM//2,
             y_dim=cfg.EMy//2,
             x_dim=cfg.EMx//2, 
             feed_forward_hidden = cfg.FF//2,
             normalization='batch'
                        )
             
        self.usr5 = USREncoder(
             n_heads=4,
             embed_dim=cfg.EM//2,
             y_dim=cfg.EMy//2,
             x_dim=cfg.EMx//2, 
             feed_forward_hidden = cfg.FF//2,
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
            n_heads=4,
            embed_dim=cfg.EM//2,
            y_dim=cfg.EMy//2,
            x_dim=cfg.EMx//2
                 )
         
    def forward(self,x,y,mask=None):
        y = self.cov(y).unsqueeze(1)
        x, y = self.usr1(x,y,mask)      
        x, y = self.usr2(x,y,mask)      
        x, y = self.usr3(x,y,mask)      
        x, y = self.usr4(x,y,mask)      
        x, y = self.usr5(x,y,mask)      
        # x, y = self.usr6(x,y,mask) 
      #  x, y = self.usr7(x,y,mask)  
      #  x, y = self.usr8(x,y,mask)       
        output = self.out(x,y)
        return output
    
class COMPLEX_SAD(nn.Module):
    def __init__(self,cfg):
        super(COMPLEX_SAD,self).__init__()

        if cfg.use_cov == True:
            # self.cov = torch.nn.Sequential(
            #      nn.Linear(2*L*L, FF),
            #      nn.ReLU(),
            #      nn.Linear(FF, EMy)
            #    )

            # self.cov = nn.Linear(2*L*L, EMy)
            self.cov= ComplexLinear(cfg.L*cfg.L, cfg.EMy//2) #**************
        else:
            self.cov = ComplexLinear(cfg.L*cfg.M, cfg.EMy//2)


        #
        self.usr1 = Transformer_Encoder(embed_dim=cfg.EMy//2,
         node_dim=cfg.L, num_heads=4)
        # self.usr2 = USREncoder(
        #     n_heads=8,
        #     embed_dim=EM,
        #     y_dim=EMy,
        #     x_dim=EMx,
        #     feed_forward_hidden = FF,
        #     normalization='batch'
        #                )
        self.usr2 = Transformer_Encoder(embed_dim=cfg.EMy//2,
         A_dim=cfg.EMy//2, num_heads=4)
        
        self.usr3 = Transformer_Encoder(embed_dim=cfg.EMy//2,
         A_dim=cfg.EMy//2, num_heads=4)
        
        self.usr4 = Transformer_Encoder(embed_dim=cfg.EMy//2,
         A_dim=cfg.EMy//2, num_heads=4)
        
        self.usr5 = Transformer_Encoder(embed_dim=cfg.EMy//2,
         A_dim=cfg.EMy//2, num_heads=4)
        
        #real
        # self.out = Out( 8,
        #     embed_dim=cfg.EM,
        #     y_dim=cfg.EMy,
        #     x_dim=cfg.EMx
        #          )
        # complex
        self.out = Complex_Out(num_heads = 4,
            embed_dim=cfg.EM//2,
            y_dim=cfg.EMy//2,
            x_dim=cfg.EMx//2
                 )
        


    def forward(self,x,y,mask=None):
        # input_shape
        #x:(256, 100, 12)
        #y:(256,144)
        ############
        y = self.cov(y).unsqueeze(1) # y:(256,1,64)
        # y = self.cov(y)
        x, y = self.usr1(x,y) #256, 100, 64  #256, 1, 64
        x, y = self.usr2(x,y) #256, 100, 64  #256, 1, 64
        x, y = self.usr3(x,y) #256, 100, 64  #256, 1, 64
        x, y = self.usr4(x,y) #256, 100, 64  #256, 1, 64
        x, y = self.usr5(x,y) #256, 100, 64  #256, 1, 64
        # #real:
        # x = torch.view_as_real(x).view(x.size(0), x.size(1), -1) #256, 100, 128
        # y = torch.view_as_real(y).view(y.size(0), y.size(1), -1) #256, 1, 128
        # output = self.out(x,y)
        #complex:
        output = self.out(x,y)
        return output
    

if __name__ == '__main__':
    #model
    # model = COMPLEX_SAD(cfg).to(device)
    # summary(model, input_size=[(100, 12), (1,144)], dtypes=torch.complex64)
    #---------------------dataloading----------------------------

    vA, vCov, vsupp = datasetGeneration(cfg.VSIZE,cfg)

    if cfg.complex:
        model = COMPLEX_SAD(cfg).to(device)
        criterion = binary_ce_loss(pos_weight=torch.tensor([(cfg.N-cfg.K)/cfg.K]).to(device))
    else:
        model = SAD(cfg).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(cfg.N-cfg.K)/cfg.K]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,90],gamma=0.1)

    if cfg.test_flag:
        checkpoint = torch.load(cfg.path_checkpoint, map_location=torch.device(device))  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    #  lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,97],gamma=0.1,last_epoch=0)

        print('Load epoch {} successfully'.format(start_epoch))
    else:
        start_epoch = -1
        print('From epoch 0')
    print("------------------------------Ready for training!-------------------------")
    print('L:{}\t M:{}\t N:{}\t K:{}\t Complex?:{}\t'.format(cfg.L, cfg.M, cfg.N, cfg.K, cfg.complex))
    for epoch in range(start_epoch+1, cfg.epoch):
        model.train()
        train_loss = 0
        for index in tqdm(range(cfg.BNUM)):
            dA, dCov, dsupp = datasetGeneration(cfg.BSIZE,cfg)
            bA = dA.to(device)
            bCov = dCov.to(device)
            bsupp = dsupp.to(device)
            optimizer.zero_grad()
            logit = model(bA,bCov,mask = None)
            loss = criterion(logit, bsupp)
            loss.backward()
            optimizer.step()
            train_loss += (2*cfg.K/cfg.N)*loss.item()
            # with profile(activities=[ProfilerActivity.CPU , ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            #         optimizer.zero_grad()
            #         logit = model(bA,bCov,cfg.mask)
            #         loss = criterion(logit, bsupp)
            #         loss.backward()
            #         optimizer.step()
            #         train_loss += (2*cfg.K/cfg.N)*loss.item()
            #         # loss = criterion(logit, bsupp)
            #         # loss.backward()
            # if index == 1:
            #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            #     prof.export_chrome_trace("trace.json")
            #     print(1)
        
            # exit()
            

        loss_mean = train_loss / cfg.BNUM
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean))

        model.eval()
        with torch.no_grad():
            #vA, vCov, vsupp = bA, bCov, bsupp
            vA = vA.to(device)
            vCov = vCov.to(device)
            vsupp = vsupp.to(device)
            vlogit = model(vA,vCov,mask=None)
            vloss = criterion(vlogit, vsupp).item()
            if cfg.complex:
                vprob = vlogit
            else:
                vprob = torch.sigmoid(vlogit)
            vpred = torch.zeros(vprob.size()).to(device)
            vpred[vprob>=0.5]=1
            err = cfg.N*cfg.VSIZE-(vpred==vsupp).sum().sum().item()

        print('Test set: Loss: {:.4f}, Error rate: {}/{} ({:.2f}%)'.format((2*cfg.K/cfg.N)*vloss, err, cfg.N*cfg.VSIZE, 100. * err/cfg.N/cfg.VSIZE))
        print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])
        print('\n')

        lr_schedule.step()

        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        if not os.path.isdir(cfg.filename):
            os.mkdir(cfg.filename)
        torch.save(checkpoint, cfg.filename+'/ckpt_best_%s.pth' % (str(epoch)))