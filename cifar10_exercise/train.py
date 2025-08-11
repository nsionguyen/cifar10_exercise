import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import optim




class SimpleNeuralNetwork(nn.Module):


    def __init__(self, num_classes=10):
        super(SimpleNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x




if __name__ == '__main__':

    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    num_workers = 2


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ])

    train_set = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model = SimpleNeuralNetwork(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    print("Bắt đầu huấn luyện...")
    num_iters = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward và tối ưu hóa
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{num_iters}], Loss: {loss.item():.4f}')

    print('Hoàn tất huấn luyện.')

    # --- Đánh giá mô hình ---
    print("Bắt đầu đánh giá mô hình...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Độ chính xác của mô hình trên 10000 ảnh thử nghiệm: {100 * correct / total:.2f} %')