import torch
import torch.nn as nn
from models import resnet, transformer
import torchvision.models as models
import pdb

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        resnet_model = models.resnet18(pretrained=True)
        # gives spatial features before avg. pooling
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.avg_pool = nn.AvgPool2d(7)
        self.projection = nn.Linear(512, 10)

    def forward(self, x):
        # pdb.set_trace()
        # x: B x C x W x H
        out = self.resnet(x)
        out = self.avg_pool(out)
        out = out.flatten(start_dim=1)
        out = self.projection(out)

        return out

