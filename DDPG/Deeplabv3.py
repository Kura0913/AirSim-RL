import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21):
        super(DeepLabV3, self).__init__()
        self.deeplab = segmentation.deeplabv3_resnet101(pretrained=True)
        self.deeplab.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.deeplab(x)['out']
        return x
