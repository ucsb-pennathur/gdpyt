import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import Inception3

class GdpytInceptionv3(nn.Module):


    def __init__(self):
        super(GdpytInceptionv3, self).__init__()
        inception_out = 1000
        self.inception = Inception3(num_classes=inception_out, aux_logits=False, transform_input=False,
                                    init_weights=True)
        self.fc1 = nn.Linear(inception_out, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 1, bias=True)

        nn.init.kaiming_normal_(self.fc1.weight.data, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = self.inception(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
