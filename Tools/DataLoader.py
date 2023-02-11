import ast

# carregando as informacoes do arquivo em memoria.
# Aviso: Perigo de estouro de memoria.


def loadCsiData(filename, return_dict=False):
    caminho_arquivo = filename

    with open(caminho_arquivo, 'r') as file:
        pacotes = file.readlines()  # lista que conterá as linhas do arquivo, cada linha é um pacote de dados.

    # verificando a quantidade de pacotes que temos
    print('Quantidade de pacotes: ', len(pacotes))  # tem que bater com a quantidade capturada.

    if return_dict:
        data = []
        for pkt in pacotes:
            data.append(ast.literal_eval(pkt))
        return data

    return pacotes


##############################################################################################################
import os
import time
import ast
from multiprocessing import Pool
import numpy as np


def loadCsiData(filename, return_dict=False):
    caminho_arquivo = filename

    with open(caminho_arquivo, 'r') as file:
        pacotes = file.readlines()  # lista que conterá as linhas do arquivo, cada linha é um pacote de dados.

    # verificando a quantidade de pacotes que temos
    print('Quantidade de pacotes: ', len(pacotes))  # tem que bater com a quantidade capturada.

    if return_dict:
        data = []
        for pkt in pacotes:
            data.append(ast.literal_eval(pkt).get("CSI").get("CSI"))
        with open(filename[:-4] + '.npz', 'wb') as file:
            np.save(file, np.array(data))

    return pacotes


def load_data(filename):
    return loadCsiData(filename, return_dict=True)


def export_data(filename, data):
    assert type(data) == np.ndarray, "Data must be a numpy array!"
    np.save(filename, data)


if __name__ == '__main__':
    start = time.time()
    path = r"F:\Projetos\Dados CSI/"
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files]
    pool = Pool(16)
    data = pool.map(load_data, files)
    end = time.time()
    print('Time taken: ', end - start)