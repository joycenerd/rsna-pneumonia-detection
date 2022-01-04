import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self,num_classes):
        super(EnsembleModel,self).__init__()
        # model A: resnet50
        self.modelA=torchvision.models.resnet50(pretrained=True)
        in_feat=self.modelA.fc.in_features
        self.modelA.fc=nn.Linear(in_feat,num_classes)
        nn.init.xavier_uniform_(self.modelA.fc.weight)
        
        # model B: resnest50
        self.modelB=torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        in_feat=self.modelB.fc.in_features
        self.modelB.fc=nn.Linear(in_feat,num_classes)
        nn.init.xavier_uniform_(self.modelB.fc.weight)

        self.w=nn.Parameter(torch.tensor(0.5,dtype=torch.double),requires_grad=True)

    def forward(self,x):
        x1=self.modelA(x.clone())
        x1=x1.view(x1.size(0),-1)
        
        x2=self.modelB(x)
        x2=x2.view(x2.size(0),-1)
        
        # out=self.multi_layer(x1,x2)
        out=self.w*x1+(1-self.w)*x2
        return out

