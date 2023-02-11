import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def minmax_scaler(data):
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    return (data - data_min)/data_range


# construindo uma rede neural fully connected com 4 camadas, sendo 300 neuronios na entrada, 100 e 25 nas ocultas e
# 2 na saida com softmax
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # instanciando a rede neural
    net = Net()
    states = torch.load(r'C:\Users\alexv\PycharmProjects\OmniSaude-IA\IA-AlgoritmosComplexos\Modelos\acc91_2-loss0_21.pth',
                        map_location=torch.device('cpu'))
    net.load_state_dict(states)
    # criando um sinal senoidal com 300 amostras
    x = np.linspace(0, 2 * np.pi, 300)
    sinal = np.sin(x).astype(np.float32)
    # inserindo ruido gaussiano
    sinal += np.random.normal(0, 0.3, sinal.shape)

    sinal = minmax_scaler(sinal).reshape(1, -1)

    out = net(torch.tensor(sinal))
    print(out)

    # plotando o sinal
    plt.figure(figsize=(16, 9))
    plt.plot(sinal[0])
    plt.show()
