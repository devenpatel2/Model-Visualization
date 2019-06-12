import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
import models
import logging
import os
from datetime import datetime
from dataloader import DirDataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def _train(model, trainloader, epochs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # switch to train mode
    model.train()
    print_freq = ((len(trainloader) // 3) // 10) * 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_freq == print_freq - 1:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.3f' %
                             (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            del loss

    logger.info('Finished Training')


def train(path, epochs=50, batch_size=256, model_name="LeNet", save_path="~/models"):

    model = getattr(models, model_name)()

    if torch.cuda.is_available():
        logger.info("Found and using Cuda")
        model.cuda()

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize(model.input_size),
                                    transforms.ToTensor()
                                    ])
    # torchvision's ImageFolder can be readily used, but custom dataset loader is used
    # for demonstration purpose
    # train_set = ImageFolder(path, transform=transform)

    train_set = DirDataset(path, transform=transform)
    logger.info("Found %d image for training" % len(train_set))
    trainloader = DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=True
                             )
    # TO-DO 
    # try except keyboardinterrupt.
 
    _train(model, trainloader, epochs)

    if not os.path.exists(os.path.expanduser(save_path)):
        os.makedirs(save_path)
    
    torch.save(model, os.path.join(save_path, datetime.today().strftime('%Y-%m-%d-%H-%M') + '_' + model_name)) 


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to train images")
    parser.add_argument("-n", "--epochs", type=int,
                        help="Number of epochs", default=50)
    parser.add_argument("-b", "--batch", type=int,
                        help="Batch size", default=256)
    args = parser.parse_args()
    train(args.path, args.epochs, args.batch)
