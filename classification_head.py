import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes=31):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits

    def new(self):
        model_new = ClassificationHead(self.input_dim).cuda()
        return model_new
