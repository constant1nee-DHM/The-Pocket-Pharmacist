import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os
import requests
from PIL import Image


class DatasetImageHandler:
    def __init__(self, dataset_path, batch_size):

        self.drugs = {
            0: "A-ferin",
            1: "Apodorm",
            2: "Apronax",
            3: "Arveles",
            4: "Aspirin",
            5: "Dikloron",
            6: "Dolcontin",
            7: "Dolorex",
            8: "Fentanyl",
            9: "Hametan",
            10: "Imovane",
            11: "Majezik",
            12: "Metpamid",
            13: "Midazolam B. Braun",
            14: "Morphin",
            15: "Nobligan Retard",
            16: "Oxycontin",
            17: "Oxynorm",
            18: "Parol",
            19: "Sobril",
            20: "Terbisil",
            21: "Ultiva",
            22: "Unisom",
            23: "Valium Diazepam",
            24: "Xanor"
        }


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def get_classes(self, path):
        return os.listdir(path)

    def to_tensor(self):
        return self.dataset

    def form_batch(self):
        return self.data_loader

    def plot_batch(self):
        data = iter(self.data_loader)
        images, labels = next(data)
        images = images * 0.5 + 0.5  
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
        # print("Shape before flattening:", X.shape)  
        X = self.flattened(X)
        # print("Shape after flattening:", X.shape)  
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
            print(batch)  
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

def access_pth():
    link = "https://www.dropbox.com/scl/fi/4sxosdkhzaftxr1zb37gf/the-pocket-pharmacist.pth?rlkey=o49hoxabh1rqzs5k3whvpakit&st=kyno5c5v&dl=1"  # <- changed dl=1
    filename = 'the_pocket_pharmacist.pth'
    if not os.path.isfile(filename):
        r = requests.get(link)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print('Model file downloaded.')
    else:
        print('Model file already exists.')
    return filename

def clear_folder(folder_path):
    if os.listdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


def main():
    DATASET = 'app/cnn_model/Dataset/train'
    L_RATE = 0.001
    O = 25
    BATCH_SIZE = 15
    N_EPOCHS = 10
    dataset = DatasetImageHandler(DATASET, BATCH_SIZE)
    model = Classifier(O)
    # trained_weights = 'The-Pocket-Pharmacist/app/cnn_model/the-pocket-pharmacist.pth'
    trained_weights = access_pth()
    if not os.path.isfile(trained_weights):
        trainer = Trainer(dataset.data_loader, model, L_RATE, N_EPOCHS)
        print('Training in progress...')
        trainer.train()
        trainer.plot_loss()
        torch.save(model.state_dict(), trained_weights)
    else:
        print('Already trained')
    model.load_state_dict(torch.load(trained_weights))
    model.eval()
    print(f'Model {trained_weights} loaded')
    transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_images = []
    test_folder = 'uploads'
    
    if os.listdir(test_folder):
        for filename in os.listdir(test_folder):
            if filename.endswith(('jpeg', 'png', 'jpg')):
                test_images.append(filename)
        for file in test_images:
            image_path = os.path.join(test_folder, file)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)  
            with torch.no_grad():
                output = model(image)
                prediction = torch.argmax(output, dim=1)
                prediction = dataset.drugs[(prediction.item())]
                print(f'Prediction for {file}: {prediction}')
                clear_folder(test_folder)
        return prediction


if __name__ == '__main__':
    main()


