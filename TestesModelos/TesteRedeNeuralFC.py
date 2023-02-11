from AlgoritmosComplexos.FeedForwardSimples import Net, minmax_scaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import listdir


apneia_value = (1, 0)
resp_value = (0, 1)


def predictions(key, text, be_apneia: bool):
    # para apneia, de lado esquerdo
    acertos = 0
    erros = 0

    if be_apneia:
        sinais = dados_apneia[key]
    else:
        sinais = dados_resp[key]
    sinal_preparado = np.transpose(sinais, (0, 2, 1)).reshape(-1, 300)
    sinal_preparado = np.apply_along_axis(minmax_scaler, 1, sinal_preparado).astype(np.float32)

    predictions = net(torch.tensor(sinal_preparado))
    predictions = predictions.detach().numpy()
    predictions = np.apply_along_axis(lambda x: apneia_value if x[0] > x[1] else resp_value, 1,
                                      predictions)

    # contado os acertos e erros
    for prediction in predictions:
        if be_apneia:
            if np.argmax(prediction) == np.argmax(apneia_value):
                acertos += 1
            else:
                erros += 1
        else:
            if np.argmax(prediction) == np.argmax(resp_value):
                acertos += 1
            else:
                erros += 1

    if be_apneia:
        print("Apneia e posição de", text + ":")
    else:
        print("Respiração e posição de", text + ":")

    print(f'Acertos: {acertos} ({acertos*100/(acertos+erros):.2f}%)\n'
          f'Erros: {erros} ({erros*100/(acertos+erros):.2f}%)\n')


if __name__ == '__main__':
    acertos = 0
    erros = 0

    # carregando os dados
    dados_apneia = {}
    dados_resp = {}
    paths = listdir(r'C:/users/alexv/pycharmprojects/omnisaude-ia/Data/Dataset_teste')
    for path in paths:
        if "apneia" in path:
            dados_apneia[path] = np.load(r'C:/users/alexv/pycharmprojects/omnisaude-ia/Data/Dataset_teste/' + path)
        elif "respiracao" in path:
            dados_resp[path] = np.load(r'C:/users/alexv/pycharmprojects/omnisaude-ia/Data/Dataset_teste/' + path)

    # instanciando a rede neural acc 91.2% loss 0.21
    net = Net()
    states = torch.load(r'C:\Users\alexv\PycharmProjects\OmniSaude-IA\AlgoritmosComplexos\Modelos\acc91_2-loss0_21.pth',
                        map_location=torch.device('cpu'))
    net.load_state_dict(states)

    # para apneia, de frente
    key = 'apneia_Apneia1_0-Alex-Intel 5300-Rapida-Sentado_frente-FFZ-_1m-Intel 5300-22_09_2022-18_37-99_93Hz-32R-87bpm.npy'

    predictions(key, "frente", True)

    # para apneia, de lado direito
    key = 'apneia_Apneia6_1-Mariana-Intel 5300-Normal-Sentado_ladoDir-FFZ-_1m-Intel 5300-26_09_2022-15_57-99_93Hz-26R-90bpm.npy'

    predictions(key, "lado direito", True)

    # para apneia e posição de costas
    key = 'apneia_Apneia13_1-Wilson-Intel 5300-Normal-Sentado_costas-FFZ-_1m-Intel 5300-06_10_2022-13_01-99_95Hz-24R-86bpm.npy'

    predictions(key, "costas", True)

    # para apneia e posição de lado esquerdo
    key = 'apneia_Apneia14_0-Wilson-Intel 5300-Normal-Sentado_ladoEsq-FFZ-_1m-Intel 5300-06_10_2022-13_35-99_95Hz-25R-91bpm.npy'

    predictions(key, "lado esquerdo", True)

    # para apneia sentado de frente
    key = 'apneia_Apneia4_1-Alex-Intel 5300-Normal-Sentado_frente-FFZ-_1m-Intel 5300-22_09_2022-19_24-99_93Hz-17R-79bpm.npy'

    predictions(key, "frente 2", True)

    # para respiração, de frente e rapida
    key = 'respiracao_Apneia1_1-Alex-Intel 5300-Rapida-Sentado_frente-FFZ-_1m-Intel 5300-22_09_2022-18_40-99_93Hz-38R-85bpm.npy'

    predictions(key, "frente e rapida", False)

    # para respiração, de lado direito e normal
    key = 'respiracao_Apneia2_0-Alex-Intel 5300-Normal-Sentado_ladoDir-FFZ-_1m-Intel 5300-22_09_2022-18_59-99_95Hz-24R-88bpm.npy'

    predictions(key, "lado direito e normal", False)

    # para respiração, de costas e normal
    key = 'respiracao_Apneia5_1-Mariana-Intel 5300-Normal-Sentado_costas-FFZ-_1m-Intel 5300-26_09_2022-15_49-99_94Hz-25R-90bpm.npy'

    predictions(key, "costas e normal", False)

    # para respiração, de frente e normal
    key = 'respiracao_Apneia8_0-Luiz Fernando-Intel 5300-Normal-Sentado_frente-FFZ-_1m-Intel 5300-28_09_2022-09_26-99_89Hz-15R-73bpm.npy'

    predictions(key, "frente e normal", False)

    # para respiração, de lado esquerdo e normal
    key = 'respiracao_Apneia14_0-Wilson-Intel 5300-Normal-Sentado_ladoEsq-FFZ-_1m-Intel 5300-06_10_2022-13_35-99_95Hz-25R-91bpm.npy'

    predictions(key, "lado esquerdo e normal", False)
