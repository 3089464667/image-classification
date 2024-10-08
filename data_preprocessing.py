import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_data(batch_size=32):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return trainloader, testloader

def get_class_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_data(batch_size=32):
    trainloader, testloader = load_data(batch_size)
    
    # 获取一个批次的数据来确定输入形状
    example_data, _ = next(iter(trainloader))
    input_shape = example_data.shape[1:]  # (C, H, W)
    
    return trainloader, testloader, input_shape