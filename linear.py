import time
import torch
from torchvision import transforms, datasets
from Model.mocoNet import ResNet50

trainaug = transforms.Compose([
    # transforms.RandomCrop(32),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
validaug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def getAcc(pred, target):
    with torch.no_grad():
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        pred = pred.argmax(dim=-1)
        pacc = pred.eq(target.view_as(pred)).sum().item()
        return pacc


def trainStep(model, train_data, optimizer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for idx, (data, target) in enumerate(train_data):
        data, target = data.to("cuda"), target.to("cuda")
        pred = model(data)
        loss = torch.nn.CrossEntropyLoss()(pred, target)
        total_loss += loss.item()
        total_acc += getAcc(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = total_loss / (idx + 1)
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


def train_linear(path):
    train_data = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, download=False, transform=trainaug),
        batch_size=128,
        shuffle=True,
        num_workers=2,
    )

    valid_data = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, download=False, transform=validaug),
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    model = ResNet50(10).to("cuda")
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    state_dict = torch.load(path)

    for k in list(state_dict.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict, False)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    for epoch in range(1, 1000):
        start = time.time()
        trainloss, trainacc = trainStep(model, train_data, optimizer)
        end = time.time()
        validloss, validacc = validStep(model, valid_data)
        print("Epoch {}, trainTime: {}s".format(epoch, int(end - start)))
        template = "=> trainLoss: {:.4f}, trainAcc: {:.3f}, validLoss: {:.4f}, validAcc: {:.3f}"
        print(template.format(trainloss, trainacc, validloss, validacc))
