# Import libraries
import numpy as np
import time
import h5py
import argparse
import os.path
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchsummary import summary
from cnn import CNNModel  # Assuming you have a CNNModel class in a 'model' module
# from utils import str2bool  # Utility function for argument parsing

def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
    parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=40, help="num of epoches")
    parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
    parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
    parser.add_argument("-learning_rate", dest ="learning_rate", type=float, default=0.001, help = "learning rate")
    parser.add_argument("-decay", dest ="decay", type=float, default=0.5, help = "learning rate")
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
    parser.add_argument("-dropout", dest ="dropout", type=float, default=0.4, help = "dropout prob")
    parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
    parser.add_argument("-load_checkpoint", dest="load_checkpoint", type=bool, default=True, help="true of false")

    parser.add_argument("-activation", dest="activation", type=str, default='relu', help="activation function")
    # parser.add_argument("-MC", dest='MC', type=int, default=10, help="number of monte carlo")
    parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
    parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
    parser.add_argument("-k_size", dest='k_size', type=int, default=4, help="size of filter")
    parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
    parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
    parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
    parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")
    return parser.parse_args()


def load_data(DATA_PATH, batch_size):
    print(f"data_path: {DATA_PATH}")

    train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=True, transform=train_trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=False, transform=test_trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader

def compute_accuracy(y_pred, y_batch):
    accy = (y_pred == y_batch).sum().item() / len(y_batch)
    return accy

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_labels in val_loader:
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            output_y = model(x_batch)
            loss = nn.CrossEntropyLoss()(output_y, y_labels)
            val_loss += loss.item()
            _, preds = torch.max(output_y, 1)
            val_accuracy += (preds == y_labels).float().mean()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_labels.cpu().numpy())

    confusion = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return val_loss/len(val_loader), val_accuracy/len(val_loader), confusion, precision, recall, f1

def print_model_size(model):
    summary(model, (1, 28, 28)) 

def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    lr = learning_rate
    if epoch > 5:
        lr = 0.001
    if epoch >= 10:
        lr = 0.0001
    if epoch > 20:
        lr = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    for batch_id, (x_batch, y_labels) in tqdm(enumerate(train_loader), desc="Training", leave=False):  
        x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

        output_y = model(x_batch)
        loss = nn.CrossEntropyLoss()(output_y, y_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, y_pred = torch.max(output_y.data, 1)
        accy = compute_accuracy(y_pred, y_labels)

        # Here, you can add code to log or print the loss and accuracy if you want


def test_model(model, test_loader, device):
    model.eval()
    total_accy = 0
    for batch_id, (x_batch, y_labels) in tqdm(enumerate(test_loader), desc="Testing", leave=False):  
        x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
        output_y = model(x_batch)
        _, y_pred = torch.max(output_y.data, 1)
        accy = compute_accuracy(y_pred, y_labels)
        total_accy += accy
    return total_accy / len(test_loader)
    

def main():
    args = parse_arguments()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}")

    train_loader, test_loader = load_data("./data/", args.batch_size)

    model = CNNModel(args.fc_hidden1, args.fc_hidden2, args.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epoches):
        adjust_learning_rate(args.learning_rate, optimizer, epoch, args.decay)
        train_one_epoch(model, optimizer, train_loader, device)
        test_accuracy = test_model(model, test_loader, device)
        print(f"Epoch {epoch+1}, Test Accuracy: {test_accuracy}")

        # Optionally, save model checkpoint here

    print("Training Complete!")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Running time: {(end_time - start_time) / 60.0:.2f} mins")