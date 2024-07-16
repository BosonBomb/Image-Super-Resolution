"""
Implementing the paper ESRGAN(Enhanced Super Resolution General Adverserial Network)
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RRDB(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1)
    def ResidualBlock(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, stride =1),
            nn.LeakyReLU()
        )
    def forward(self, x):
        self.layers = [self.ResidualBlock() for _ in range(4)]
        con1 = x
        out = con1 + self.layers[0](x)
        con2 = out
        out = con1 + con2 + self.layers[1](out)
        con3 = out 
        out = con1 + con2 + con3 + self.layers[2](out)
        out = con3 + con2 + con1 + self.layers[3](out)
        out = self.conv1(out)
        out = out * self.beta 
        return out

class Generator(nn.Module):
    def __init__(self, fan_in, fan_out, batch_size, beta, n_rrdb):
        super().__init__()
        self.beta = beta
        self.batch_size = batch_size
        self.fan_in = fan_in
        self.fan_out  = self.floatfan_out
        self.rrdbs = nn.ModuleList([RRDB(beta) for _ in range(beta)])
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.fan_in, 64, kernel_size=9, stride=1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Conv2d(64,256, kernal_size = 3, stride=1)
        self.upscale = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Conv2d(64,3, kernel_size = 9, stride = 1)
        
    def forward(self, x):
        out = self.conv1(x)
        con1 = out
        out = con1 + self.rrdbs[0](x)
        con2 = out
        out = con2 + self.rrdbs[1](out)
        con3 = out
        out = con3 + self.rrdbs[2](out)
        out = out*self.beta
        out = con1 + out
        out =  self.conv2(out)
        out = self.upscale(out)
        out = self.conv3(out)
        return out 

class Discriminator(nn.Module):
    def __init__(self, negative_slope = 0.2):
        super().__init()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride  =  2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, stride = 2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size  = 3, stride =1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,128, kernel_size  = 3, stride =2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256, kernel_size  = 3, stride =1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256, kernel_size  = 3, stride =2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512, kernel_size  = 3, stride =1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512, kernel_size  = 3, stride =2),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        self.linear = nn.Linear(
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )
     
    def forward(self, x_r, x_f):
        # This is the Relativistic GAN based disciminator
        out_r = self.conv1(x_r)
        out_r = self.conv2(out_r)
        out_r = self.linear(out_r)
        out_f = self.conv1(x_f)
        out_f = self.conv2(out_f)
        out_f = self.linear(out_f)
        xr_xf = torch.sigmoid(out_r - torch.mean(out_f, keepdim=True))
        xf_xr = torch.sigmoid(out_f - torch.mean(out_r, keepdim=True))
        return xr_xf, xf_xr
 

def adversarial_loss(xr_xf, xf_xr, beta):
    disciminator_loss = -torch.mean(torch.log(xr_xf),keepdim=True) - torch.mean((torch.log(1-xf_xr)), keepdim=True)
    generator_loss = -torch.mean(torch.log(xf_xr),keepdim=True) - torch.mean((torch.log(1-xr_xf)), keepdim=True)
    return disciminator_loss, generator_loss

# perceptual loss before the activation layer 
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, target_layer):
        self.vgg19 = models.vgg19()
        self.target_layer = target_layer
    def forward(self, x):
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            if name == self.target_layer:
                return x

          
def perceptual_loss(self, generator_image, input_image, features_extractor):
    features_generator = features_extractor(generator_image)
    features_input = features_extractor(input_image)
    perceptual_loss = F.mse_loss(features_generator, features_input)
    return perceptual_loss


    

    

    
    
        




        
