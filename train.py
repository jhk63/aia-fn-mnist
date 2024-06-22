
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from model import Net, CNN


# Parameter
learning_rate = 0.001
train_epochs = 10
batch_size = 64

# 데이터 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(), 
    # transforms.Normalize((0.5,), (0.5,))])
    transforms.Normalize((0.1307,), (0.3081,))])

# MNIST 데이터셋 다운로드
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)
# model = Net().to(device)
# print(model)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 모델 훈련 함수
def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
    scheduler.step()

# 테스트 함수
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

def train_model():
    for epoch in range(train_epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    # 모델 저장
    torch.save(model.state_dict(), "mnist_cnn.pth")


def load_model():
    model.load_state_dict(torch.load("mnist_cnn.pth"))
    model.eval()

def predict_test_dataset():
    load_model()
    figure = plt.figure(figsize=(10, 5))
    cols, rows = 5, 2
    
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        img, label = test_dataset[sample_idx]
        img = img.unsqueeze(0).to(device)

        out = model(img)
        _, predict = torch.max(out.data, 1)

        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}, Predict: {predict.item()}")
        plt.axis("off")
        plt.imshow(img.squeeze().cpu(), cmap="gray")

    plt.show()


if __name__ == "__main__":
    train_model()
    predict_test_dataset()

