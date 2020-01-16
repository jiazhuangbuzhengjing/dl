import os
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import torch
from vgg import get_vgg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
import os
 
#对数据集的读取
class DogCat(data.Dataset):
    def __init__(self, root, transform=None, train=True, test=False):
        self.test = test
        self.train = train
        self.transform = transform
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
 
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
 
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        else:
            random.shuffle(imgs)
            if self.train:
                self.imgs = imgs[:int(0.7 * imgs_num)]
            else:
                self.imgs = imgs[int(0.7 * imgs_num):]
    #作为迭代器必须有的方法
    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0 #狗的label设为1，猫的设为0
 
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label
 
    def __len__(self):
        return len(self.imgs)
 
#对数据集训练集的处理
transform_train=transforms.Compose([
	transforms.Resize((224,224)), #先调整图片大小至224x224

	transforms.RandomHorizontalFlip(), #随机的图像水平翻转，通俗讲就是图像的左右对调
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.2225)) #归一化，数值是用ImageNet给出的数值
])
 
#对数据集验证集的处理
transform_val=transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda
trainset = DogCat('F:/数据集/Cats_Dogs/train/all', transform=transform_train)
valset = DogCat('F:/数据集/Cats_Dogs/train/all', transform=transform_val)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False,num_workers=0)
model = get_vgg()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) #设置训练细节
scheduler = StepLR(optimizer, step_size=3)
criterion = nn.CrossEntropyLoss()
 
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
 
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    model.train()
    train_acc = 0.0
    for batch_idx,(img,label) in enumerate(trainloader):
        image=Variable(img.cuda())
        label=Variable(label.cuda())
        optimizer.zero_grad()
        out=model(image)
        loss=criterion(out,label)
        loss.backward()
        optimizer.step()
        train_acc = get_acc(out,label)
        print("Epoch:%d [%d|%d] loss:%f acc:%f" %(epoch,batch_idx,len(trainloader),loss.mean(),train_acc))
 
def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	print("Acc: %f "% ((1.0*correct.numpy())/total))
 
for epoch in range(21):
    train(epoch)
    val(epoch)
torch.save(model, 'modelcatdog.pth')  # 保存模型
