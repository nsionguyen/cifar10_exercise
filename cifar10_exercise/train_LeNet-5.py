import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10



class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Lớp tích chập đầu tiên: 3 kênh đầu vào (ảnh màu), 6 kênh đầu ra, kernel 5x5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

       
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        
        x = torch.flatten(x, 1)

        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x


--------------------------------------------------------------------------

if __name__ == '__main__':
   
    batch_size = 64
    num_epochs = 20  # LeNet huấn luyện nhanh hơn nên có thể tăng epoch
    learning_rate = 0.001
    num_workers = 2

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chuẩn hóa đơn giản
    ])

    train_set = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model = LeNet5(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    print("Bắt đầu huấn luyện LeNet-5...")
    num_iters = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{num_iters}], Loss: {loss.item():.4f}')

    print('Hoàn tất huấn luyện.')

    #Đánh giá mô hình 
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