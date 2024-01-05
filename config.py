from yacs.config import CfgNode as CN
import numpy as np
import math

_CN = CN()

_CN.EMx = 128
_CN.EMy = 128
_CN.EM = 128
_CN.N = 100
_CN.K = 8
_CN.L = 8
_CN.J = 0
_CN.M = 32
_CN.D = 500
_CN.FF = 512
_CN.txPower = 23 # dBm
_CN.noisePower = -99 # dBm
_CN.noisePowerN = float(_CN.noisePower + 15.3 + 37.6*np.log10(_CN.D*math.tan(math.pi/6)) - _CN.txPower)
_CN.location = 'control'
_CN.channel = 'iid'
_CN.use_cov = True
# _CN.sigma2s = np.ones([_CN.N,1])
_CN.txPowerN = 0
_CN.matrx_Type = 'Gaussian'
_CN.VSIZE = 512
_CN.BSIZE = 256
_CN.TSIZE = 5000
_CN.BNUM = 5000
_CN.epoch = 100
_CN.mask = 'None'
_CN.filename = './real_N100K8M32L8_Layer5_em64'
_CN.path_checkpoint = _CN.filename+'/ckpt_best_0.pth'  # 断点路径
_CN.pkl_path = 'training_data/data.pkl'

_CN.test_flag = False
_CN.complex = False
_CN.cos = True

def get_cfg():
    return _CN.clone()