import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self,num_classes,layer):
        super(EnsembleModel,self).__init__()
        # model A: resnet50
        if layer==50:
            self.modelA=torchvision.models.resnet50(pretrained=True)
        elif layer==101:
            self.modelA=torchvision.models.resnet101(pretrained=True)
        featA=self.modelA.fc.in_features
        self.modelA.fc=nn.Identity()
        
        # model B: resnest50
        if layer==50:
            self.modelB=torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        elif layer==101:
            self.modelB=torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        featB=self.modelB.fc.in_features
        self.modelB.fc=nn.Identity()

        # classifier
        self.classifier=nn.Sequential(
            nn.Linear(featA+featB,2048),
            nn.ReLU(),
            nn.Linear(2048,num_classes)
        )

    def forward(self,x):
        x1=self.modelA(x.clone())
        x1=x1.view(x1.size(0),-1)
        
        x2=self.modelB(x)
        x2=x2.view(x2.size(0),-1)
        
        out=torch.cat((x1,x2),dim=1)
        out=self.classifier(out)
        return out