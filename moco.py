import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim
import torch.utils.data
from Model.mocoNet import MoCo
import time
import os
import torchvision
import random
from PIL import ImageFilter


class GaussianBlur(object):

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


trainaug = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

traindir = '/home/hxj/File007/train/'
validdir = "/home/hxj/File007/test/"
criterion = nn.CrossEntropyLoss()


def getAcc(logits, target):
    with torch.no_grad():
        _, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        pacc = pred.eq(target.view(1, -1).expand_as(pred))
        pacc = pacc[:1].view(-1).float().sum(0, keepdim=True)
        return pacc.item()


def trainStep(model, train_data, optimizer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for i, (images, _) in enumerate(train_data):
        x1, x2 = images[0].to("cuda"), images[1].to("cuda")
        output, target = model(im_q=x1, im_k=x2)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_acc += getAcc(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = total_loss / (i + 1)
    acc = total_acc / len(train_data.dataset)
    return loss, acc


def validStep(model, valid_data):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_data):
            data, target = data.to("cuda"), target.to("cuda")
            pred = model(data)
            loss = torch.nn.CrossEntropyLoss()(pred, target)
            total_loss += loss.item()
            total_acc += getAcc(pred, target)
    loss = total_loss / (idx + 1)
    acc = total_acc / len(valid_data.dataset)
    return loss, acc


def adjust_learning_rate(lr, optimizer):
    # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_encoder(model_path, trained_path):
    model = MoCo().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    start_epoch = 0
    pretrained = True

    if os.path.exists(trained_path) and pretrained:
        checkpoint = torch.load(trained_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))

    train_data = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(traindir, TwoCropsTransform(trainaug)),
        batch_size=16,
        shuffle=True,
        num_workers=2,
    )
    temploss = 10
    templr = 0.1
    for epoch in range(start_epoch + 1, 600):

        start = time.time()
        trainloss, trainacc = trainStep(model, train_data, optimizer)
        end = time.time()

        print("Epoch {}, trainTime: {}s".format(epoch, int(end - start)))
        template = "=> trainLoss: {:.4f}, trainAcc: {:.3f}"
        print(template.format(trainloss, trainacc))

        if epoch % 10 == 0:
            if trainloss < temploss:
                temploss = trainloss
                # save model
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                torch.save(checkpoint, trained_path)
                torch.save(model.state_dict(), model_path)

