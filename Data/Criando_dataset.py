from Deprecated.CSIFase import get_phase_diff
import numpy as np


def create_dataset(file):
    path = r"F:/Projetos/apneias/"
    path_to_save = r"F:/Projetos/apneias_npy/"
    teste = get_phase_diff(path + file).astype(np.float32)

    print("Salvando arquivo: " + file)
    np.save(path_to_save + file[:-4] + ".npy", teste)


if __name__ == '__main__':
    from os import listdir
    files = listdir(r"F:/Projetos/apneias/")
    for file in files:
        create_dataset(file)
