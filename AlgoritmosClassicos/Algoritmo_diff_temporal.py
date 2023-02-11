from Data.CSIFileHandler import load_npz_files_from_path
import numpy as np
import multiprocessing as mp


def get_statistics_from_npz(args):
    captura, key, window = args
    variancias = []
    desvios_padrao = []
    for subport in range(captura.shape[1]):
        amostra = captura[:, subport]
        variance = []
        stdd = []
        for i in range(window, len(amostra)):
            variance.append(np.var(amostra[i - window:i]))
            stdd.append(np.std(amostra[i - window:i]))

        variancias.append(variance)
        desvios_padrao.append(stdd)

    return key, np.array([variancias, desvios_padrao])


def get_statistics(path, window=1000, n_jobs=-1):
    """
    Calcula as estatísticas de variancia e desvio padrão de cada subporta de cada captura.

    :param path: Caminho para o diretório contendo os arquivos .npz
    :param window: tamanho da janela de observação para o cálculo das estatísticas
    :param n_jobs: quantidade de processos a serem utilizados para o cálculo das estatísticas
    :return: dicionário contendo as estatísticas de cada captura no seguinte formato:
    (key, np.array([variancias, desvios_padrao]) -> shape do np.array = (2, 57, len(captura) - window)
    """
    npz = load_npz_files_from_path(path)

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    stats = {}
    pool = mp.Pool(n_jobs)
    results = pool.map(get_statistics_from_npz, [(npz[i], i, window) for i in npz.keys()])
    pool.close()
    pool.join()

    for result in results:
        stats[result[0]] = result[1]

    return stats


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    window = 4000
    all = get_statistics(path=r"F:\Projetos\apneias_npy", window=window)
    var, std = all[list(all.keys())[0]]

    plt.figure(figsize=(16, 9))
    plt.plot(var[0, :], label="variance")
    plt.plot(std[0, :], label="std")
    plt.legend()
    plt.title(f"Window size = {window} - {list(all.keys())[0]}")
    plt.show()
