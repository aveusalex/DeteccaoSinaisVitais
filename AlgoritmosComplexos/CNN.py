import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4480, 4096)
        self.fc2 = nn.Linear(4096, 10)
        self.fc3 = nn.Linear(10, 2)
        self.batchNorm1d = nn.BatchNorm1d(4480)
        self.batchNorm2d = nn.BatchNorm2d(64)

    def convolutional(self, x):
        # 1 conv layer
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        # print("Conv1 out:", x.shape)

        # 2 conv layer
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.batchNorm2d(x)
        x = nn.functional.max_pool2d(x, 2)
        # print("Conv2 out:", x.shape)

        # 3 conv layer
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        # print("Conv3 out:", x.shape)
        return x

    def fullyConnected(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.convolutional(x)
        # print("Conv out:", x.shape)
        x = torch.flatten(x, 1)
        # print("Flatten out:", x.shape)
        x = nn.functional.max_pool1d(x, 5)
        x = self.batchNorm1d(x)
        x = self.fullyConnected(x)
        output = nn.functional.softmax(x, dim=1)
        return output


if __name__ == '__main__':
    net = Net()
    tensor_teste = torch.rand(2, 1, 300, 57)
    out = net(tensor_teste)
    print(out)
