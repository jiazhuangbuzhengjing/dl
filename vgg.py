
from torch import nn
 
def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]
 
    for i in range(num_convs - 1):  # 定义后面的许多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.BatchNorm2d(out_channels))
        net.append(nn.ReLU(True))
 
    net.append(nn.MaxPool2d(2, 2))  # 定义池化层
    return nn.Sequential(*net)
 
 
# 下面我们定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs,
              channels):  # vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)
 
vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
# vgg类
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(25088, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )
 
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
 
def get_vgg():
    # 获取vgg网络
    net = vgg()
    return net
