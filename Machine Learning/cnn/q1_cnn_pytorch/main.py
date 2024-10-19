import os.path as osp
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from network import Net
from cs_dataset import city_scapes


def parse_args():
    parser = argparse.ArgumentParser(description="Train the network")
    parser.add_argument("--lr", help="Enter the learning rate.", default=0.0001)
    parser.add_argument("--epochs", help="Enter number of epochs", default=25)
    parser.add_argument("--momentum", help="Enter Momentum", default=0.9)
    parser.add_argument("--weight_decay", help="Enter weight decay", default=0.0005)
    parser.add_argument("--batch_size", help="Enter batch size", default=5)
    return parser.parse_args()

def val(net, val_dataloader):
    net.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for sample in enumerate(val_dataloader):
            #####Insert your code here for subtask 1j#####
            #Implement validation step where the batches of validation dataset are tested on the trained model
            #Calculate validation accuracy
            image = sample[1]['image'].to(device)
            label = sample[1]['label'].to(device)
            loss, logits = net(image, label)
            pred_label = torch.argmax(logits)
            total += label.size(0)
            accuracy += torch.sum(pred_label == label)
    return accuracy, total


def test(net, test_dataloader, classes):
    net.eval()
    predicted_labels = []
    true_labels = []
    count = 0
    for sample in enumerate(test_dataloader):
        with torch.no_grad():
            #####Insert your code here for subtask 1k#####
            # Implement test step where the batches of test dataset are tested on the best trained model
            # Calculate test accuracy also confusion matrix
            image = sample[1]['image'].to(device)
            label = sample[1]['label'].to(device)

            _, logits = net(image, label)
            index = torch.argmax(logits)

            if classes[index] == label:
                count += 1

            predicted_labels.append(classes[index])
            true_labels.append(label.cpu().numpy()[0])
    accuracy = count / len(test_dataloader) * 100
    cf_matrix = confusion_matrix(predicted_labels, true_labels)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100, index=[i for i in classes],
                         columns=[i for i in classes])
    return accuracy, df_cm


if __name__ == '__main__':
    args = parse_args()

    # check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available device:{device}')

    # fetch parameters
    lr = args.lr
    num_epochs = args.epochs
    momentum = args.momentum
    weight_decay = args.weight_decay
    batch_size = args.batch_size

    # Data augmentation and normalization for training
    # Just normalization for validation

    #####Insert your code here for subtask 1c#####
    #Define train_dataset_transform, val_dataset_transform, test_dataset_transform
    #Get help from torchvision.transforms module
    train_dataset_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])])
    val_dataset_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])])
    test_dataset_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])])

    # fetch training data
    train_path = "../../../data/cityscapesExtracted/cityscapesExtractedResized"
    train_dataset = city_scapes(datapath=train_path,
                                transform= train_dataset_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # fetch validation data
    val_path = "../../../data/cityscapesExtracted/cityscapesExtractedValResized"
    val_dataset = city_scapes(datapath=val_path,
                              transform=val_dataset_transform)

    val_dataloader = DataLoader(val_dataset, batch_size=1)

    # fetch evaluation data
    test_path = "../../../data/cityscapesExtracted/cityscapesExtractedTestResized"
    test_dataset = city_scapes(datapath=test_path,
                               transform=test_dataset_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # define paths
    folder = "saves"
    save_network = osp.join("./", folder)

    # GT classes
    classes = [0, 1, 2]

    # build model
    net = Net()
    net.to(device)

    # define optimizers
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # define log for Tensorboard
    writer = SummaryWriter()

    print('-' * 10)
    best_val_acc = 0.0
    best_model_index = 0
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0.0
        for sample in enumerate(train_dataloader):
            image = sample[1]['image'].to(device)
            label = sample[1]['label'].to(device)
            loss, logits = net(image, label)

            #####Insert your code here for subtask 1i#####
            #Calculate training loss and training accuracy
            #Run backward pass and update weights
            train_loss += loss.item() * image.size(0)

            pred_label = torch.argmax(logits, 1)
            total += label.size(0)
            train_acc += torch.sum(pred_label == label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / total
        train_acc = (100 * train_acc / total)

        val_acc, total = val(net, val_dataloader)
        val_acc = (100 * val_acc / total)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_index = epoch

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        print(f'Training Loss:{train_loss:.4f}')
        print(f'Train Accuracy:{train_acc:.4f}')
        print(f'Val Accuracy:{val_acc:.4f}')

        # entry log data for Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        filename = "checkpoint_epoch_" + str(epoch + 1) + "_tb.pth.tar"
        torch.save(net.state_dict(), osp.join(save_network, filename))

        print("Model saved at", osp.join(save_network, filename))
        print('-' * 10)

    writer.close()

    # get the trained model giving the best validation accuracy
    print(f'Getting the best model, the model {best_model_index}, on the validation set.')
    model = Net()
    model.to(device)
    filename = "checkpoint_epoch_" + str(best_model_index + 1) + "_tb.pth.tar"
    model.load_state_dict(torch.load(osp.join(save_network, filename)))

    # get the test accuracy by using this best trained model
    acc, df_cm = test(model, test_dataloader, classes)
    print(f'Test Accuracy:{acc:.4f}')
    print(f'The confusion matrix:')
    print(df_cm)
    print("Model Successfully trained and tested!")
    print('-' * 10)
