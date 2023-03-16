import paddle 
import numpy as np
from paddle.nn import functional as F
from paddle import nn

class ModulatedAttLayer(nn.Layer):
    def __init__(self, in_channels, reduction=2, mode='embedded_gaussian',name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']
        self.g = nn.Conv2D(self.in_channels,self.inter_channels,kernel_size=1,weight_attr=nn.initializer.KaimingNormal())
        self.theta = nn.Conv2D(self.in_channels, self.inter_channels, kernel_size=1,weight_attr=nn.initializer.KaimingNormal())
        self.phi = nn.Conv2D(self.in_channels, self.inter_channels, kernel_size=1,weight_attr=nn.initializer.KaimingNormal())
        self.conv_mask = nn.Conv2D(self.inter_channels, self.in_channels, kernel_size=1,weight_attr=nn.initializer bias=False)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2D(7,stride=1)
        self.fc_spatial = nn.Linear(7*7*self.in_channels,49)
        self.init_weight()
    def init_weight(self):
        #zero initialize
        new_weight = paddle.full(shape=self.conv_mask.weight.shape,dtype=self.conv_mask.weight.dtype,fill_value=0.0)
        self.conv_mask.weight.set_value(new_weight)
    def embedded_gaussian(self,x:paddle.Tensor):
        batch_size = x.shape[0]
        g_x = paddle.reshape(self.g(paddle.clone(x)),(batch_size,self.inter_channels,-1))
        g_x = paddle.transpose(g_x,(0,2,1))
        theta_x = paddle.reshape(self.theta(paddle.clone(x)),(batch_size,self.inter_channels,-1))
        theta_x = paddle.transpose(theta_x,(0,2,1))
        phi_x = paddle.reshape(self.phi(paddle.clone(x)),(batch_size,self.inter_channels,-1))

        map_t_p = 