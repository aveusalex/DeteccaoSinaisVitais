import numpy as np
from os import listdir


def load_npz_files_from_path(path):
    files = listdir(path)
    npz = {}
    for file in files:
        npz[file] = np.load(path + "\\" + file).astype(np.float32)
    return npz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    npz = load_npz_files_from_path(path=r"F:\Projetos\apneias_npy")

    key = list(npz.keys())[0]
    # plt.figure(figsize=(16, 9))
    # plt.plot(npz[][:, 0])
    # plt.show()
