import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import *
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN

class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf):
        self.dataset_name = dataset_name
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('../cifar10', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('../cifar100', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../fashionMNIST', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('../svhn', split="train", 
                                    download=True, transform=transf)


    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)
##

# Data
def load_dataset(dataset):
    normalize = T.Normalize(
        mean = [0.49139968, 0.48215841, 0.44653091],
        std = [0.24703223, 0.24348513, 0.26158784]
    )
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        normalize
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])


    if dataset == 'cifar10': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'cifar10im': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        #data_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        targets = np.array(data_train.targets)
        #NUM_TRAIN = targets.shape[0]
        classes, _ = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [500, 5000] * 5
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        # print(NUM_TRAIN)
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        data_unlabeled = MyDataset(dataset[:-2], True, test_transform)
        data_unlabeled.cifar10.targets = targets[imb_class_idx]
        data_unlabeled.cifar10.data = data_unlabeled.cifar10.data[imb_class_idx]
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'cifar100':
        data_train = CIFAR100('../cifar100', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)
        NO_CLASSES = 100
        adden = 2000
        no_train = NUM_TRAIN
    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = FashionMNIST('../fashionMNIST', train=False, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'svhn':
        data_train = SVHN('../svhn', split='train', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = SVHN('../svhn', split='test', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train