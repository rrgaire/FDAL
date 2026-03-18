


import torch


import torchvision
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskModel(nn.Module):
    def __init__(self, feature_extractor, classifier, init_weight=True):
        super(TaskModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        feats, x = self.classifier(x)
        return feats, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfg = {
    'FE': [64, 64, 'M', 128, 128, 'M'],
    'VGG19':  [256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class CommonFE(nn.Module):
    def __init__(self):
        super(CommonFE, self).__init__()
        self.features = self._make_layers(cfg['FE'])

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feats = self.features(x)
        out = feats.view(feats.size(0), -1)
        out = self.classifier(out)
        return feats, out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 128
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG19(num_classes):
    return VGG('VGG19', num_classes)

def evaluate_model(args, task_model, loader):

    task_model.eval()
    task_model.to(0)

    running_loss = 0.0
    running_corrects = 0

    total_labels = 0
    for data in loader:
        inputs, labels = data
        inputs = inputs.to(0)
        labels = labels.to(0)
        
        with torch.no_grad():
            feats, logits = task_model(inputs)


        _, preds = torch.max(logits.data, 1)


        total_labels += labels.size(0)

        running_corrects += torch.sum(preds == labels.data).item()
    eval_loss = running_loss / total_labels
    eval_accuracy = running_corrects / total_labels
    
    return eval_loss, eval_accuracy



normalize = torchvision.transforms.Normalize(
        mean = [0.49139968, 0.48215841, 0.44653091],
        std = [0.24703223, 0.24348513, 0.26158784]
    )
    


    
test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([32, 32]),
                torchvision.transforms.ToTensor(),
                normalize
])

test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )


test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=128,
    shuffle=False, 
    num_workers=8, 
    pin_memory=True
)
    
fe = CommonFE()
vgg19 = VGG(vgg_name= 'VGG19', num_classes=10)
    # vgg19 = torch.load('/home/rrgaire/projects/iccv/paper/cifar10/vgg19/indv/checkpoints/vgg19_classifier.pth')

task_model = TaskModel(fe, vgg19, False)
task_model.load_state_dict(torch.load(TEACHER_PATH_C10, map_location='cuda'), strict=True)

task_model.eval()

eval_loss, eval_accuracy = evaluate_model(
        args= None,
        task_model=task_model,
        loader=test_loader,
    )
print(eval_accuracy)