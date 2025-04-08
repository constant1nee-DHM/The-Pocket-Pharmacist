import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np


class DatasetImageHandler:
    def __init__(self, dataset_path, batch_size):

        # Check first few items in the dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def to_tensor(self):
        return self.dataset

    def form_batch(self):
        return self.data_loader

    def plot_batch(self):
        data = iter(self.data_loader)
        images, labels = next(data)
        images = images * 0.5 + 0.5  # unnormalize
        image_grid = torchvision.utils.make_grid(images, nrow=3)
        image_grid_to_np = image_grid.numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(image_grid_to_np, (1, 2, 0)))
        plt.axis('off')
        plt.title("TRAINING BATCH")
        plt.show()


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.feature_extractor = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.feature_extractor_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pooler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLu = nn.ReLU()
        self.I = nn.Linear(32 * 320 * 320, 160)
        self.H = nn.Linear(160, 75)
        self.O = nn.Linear(75, n_classes) 

    def get_feaure_map(self, input_tensor, n_channel=None):
        if n_channel == 16:
            return self.feature_extractor(input_tensor)
        elif n_channel == 32:
            return self.feature_extractor_2(input_tensor)

    def apply_pulling(self, input_tensor):
        return self.pooler(input_tensor)

    def flattened(self, input_tensor):
        flattened_tensor = input_tensor.flatten(start_dim=1)
        return flattened_tensor

    def forward(self, X):
        X = self.get_feaure_map(X, n_channel=16)
        X = F.relu(X)
        X = self.apply_pulling(X)
        X = self.get_feaure_map(X, n_channel=32)
        X = F.relu(X)
        print("Shape before flattening:", X.shape)  # Add this line
        X = self.flattened(X)
        print("Shape after flattening:", X.shape)  # Add this line
        X = self.I(X)
        X = F.relu(X)
        X = self.H(X)
        X = F.relu(X)
        X = self.O(X)
        return X

    def calculate_loss(self, prediction, Y):
        return F.cross_entropy(prediction, Y)

    def backward(self, loss):
        loss.backward()


class Trainer:
    def __init__(self, data_loader, model, l_rate, n_epochs):
        self.model = model
        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate)
        self.loss_history = []

    def train(self):
        self.model.train()
        for epoch in range(self.n_epochs):
            total = 0
            batch = next(iter(self.data_loader))
            print(len(batch))  # This will show how many elements are returned (should be 2 for standard ImageFolder)
            print(type(batch))  # This will show what type the batch is (should be a tuple)
            print(batch)  # 
            for X, Y in self.data_loader:
                self.optimizer.zero_grad()
                predictions = self.model(X)
                L = self.model.calculate_loss(predictions, Y)
                self.model.backward(L)
                self.optimizer.step()
                total += L.item()
            average_loss = total / len(self.data_loader)
            print(f"Epoch [{epoch + 1}/{self.n_epochs}] - Loss: {average_loss:.4f}")
            self.loss_history.append(average_loss)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()


def main():
    L_RATE = 0.001
    O = 25
    BATCH_SIZE = 15
    N_EPOCHS = 10
    dataset = DatasetImageHandler('Dataset/train', BATCH_SIZE)
    
    model = Classifier(O)
    trainer = Trainer(dataset.data_loader, model, L_RATE, N_EPOCHS)
    print('initialized')
    # Actual training will happen here
    # dataset.plot_batch()
    trainer.train()
    trainer.plot_loss()
    torch.save(model.state_dict(), "the-pocket-pharmacist.pth")


if __name__ == '__main__':
    main()
