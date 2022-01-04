import torchvision
from .resnest.restnest import get_model

from efficientnet_pytorch import EfficientNet
import torch

def get_net(model,num_classes):
    if model == 'resnest50':
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        return model
    elif model == 'resnest101':
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        return model
    elif model == 'resnest200':
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=True)
        return model
    elif model == 'efficientnet-b7':
        model = EfficientNet.from_pretrained(
            'efficientnet-b7', num_classes=num_classes)
        return model
    elif model == 'efficientnet-b5':
        model = EfficientNet.from_pretrained(
            'efficientnet-b5', num_classes=num_classes)
        return model
    elif model == 'efficientnet-b4':
        model = EfficientNet.from_pretrained(
            'efficientnet-b4', num_classes=num_classes)
        return model
    elif model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(
            'efficientnet-b3', num_classes=num_classes)
        return model
