import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import models
import logging

from dataloader import DirDataset 

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def _train(net, trainloader, epochs):
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #inputs = inputs.unsqueeze(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        logger.info('Finished Training')

def train(path, batch_size=256, epochs=50, model="LeNet1"):
    
    net = getattr(models, model)()
    transform = transforms.Compose([transforms.Resize(net.input_size),
                                    transforms.ToTensor()
                                    ])
    train_set = DirDataset(path, transform=transform)
    logger.info("Found %d image for training"%len(train_set))    
    trainloader = DataLoader(train_set, 
            batch_size=batch_size, 
            shuffle=True
            )
    _train(net, trainloader, epochs)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to train images")
    parser.add_argument("-n", "--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=256)
    args = parser.parse_args()
    train(args.path, args.epochs)
