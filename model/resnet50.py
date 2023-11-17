"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class resnet50(nn.Module):

    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels
            num_classes: (int) 类别数目
        """
        super(resnet50, self).__init__()
        self.num_classes = params.num_classes
        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, self.num_classes)


    def forward(self, x):
        x = self.resnet50(x)
        return x


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    计算给定输出和标签的交叉熵损失。
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0] # 获取样本数量
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    根据所有图像的输出和标签计算精确度。
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1) # 将输出结果（outputs）沿着axis=1的方向取最大值的索引，也就是预测的类别标签。
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
