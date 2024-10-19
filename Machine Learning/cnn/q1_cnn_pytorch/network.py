import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        #####Insert your code here for subtask 1d#####
        #Define building blocks of CNNs: convolution and pooling layers
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 50, 5, 1)
        self.pool = nn.MaxPool2d(3, 2)
        #####Insert your code here for subtask 1e#####
        #Define fully connected layers
        self.fc1 = nn.Linear(50 * 3 * 3, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x, label):
        """Run forward pass for the network
        :param x: a batch of input images -> Tensor
        :param label: a batch of GT labels -> Tensor
        :return: loss: total loss for the given batch, logits: predicted logits for the given batch
        """

        #####Insert your code here for subtask 1f#####
        #Feed a batch of input image x to the main building blocks of CNNs
        #Do not forget to implement ReLU activation layers here
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        #####Insert your code here for subtask 1g#####
        #Feed the output of the building blocks of CNNs to the fully connected layers
        x = x.view(-1, 50 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        #####Insert your code here for subtask 1h#####
        #Implement cross entropy loss on the top of the output of softmax
        logits = F.softmax(x, dim=1)
        loss = F.cross_entropy(logits, target=label)

        return loss, logits
