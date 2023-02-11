from CSIFileHandler import load_npz_files_from_path
import numpy as np
import multiprocessing as mp


def get_janelamento_from_npz(args):
    """
    Pega a captura e a subdivide em janelas de 'windowing/100' segundos.
    :param captura: captura a ser dividida
    :param key: chave da captura
    :param windowing: tamanho da janela em segundos * taxa de amostragem
    :return: tuple -> (key, [apneias, respiracoes])
    """
    captura, key, windowing, dimension = args

    apneias = []
    respiracoes = []
    for i in range(windowing, captura.shape[0], 20):
        if i <= 9000:
            respiracoes.append(captura[i - windowing:i, :])
        elif i >= 9000 + windowing:
            apneias.append(captura[i - windowing:i, :])
    print(f"Captura {key} processada com sucesso!")

    apneias = np.array(apneias).astype(np.float32)
    respiracoes = np.array(respiracoes).astype(np.float32)

    # guardando o array apneia e respiracoes em disco
    np.save(fr"C:\Users\alexv\PycharmProjects\OmniSaude-IA\Data\Dataset\apneia_{key}", apneias)
    np.save(fr"C:\Users\alexv\PycharmProjects\OmniSaude-IA\Data\Dataset\respiracao_{key}", respiracoes)


def create_dataset(path, window=300, dimension="1d", n_jobs=-1):
    """
    Cria um dataset de apneias e respirações normais a partir da base de dados fornecida no path.

    :param path: Caminho para o diretório contendo os arquivos .npz
    :param window: tamanho da janela de observação, em segundos * taxa de amostragem
    :param dimension: forma de gerar o dataset, pode ser 1d (subportadoras independentes) ou 2d (subportadoras agrupadas)
    :param n_jobs: quantidade de processos a serem utilizados para o cálculo das estatísticas
    :return: None
    """
    npz = load_npz_files_from_path(path)

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    pool = mp.Pool(n_jobs)
    pool.map(get_janelamento_from_npz, [(npz[i], i, window, dimension) for i in npz.keys()])
    pool.close()
    pool.join()


if __name__ == '__main__':
    window = 300
    # TODO: incluir a opção de viés de salvamento (imagem ou ondas independentes)
    #create_dataset(path=r"F:\Projetos\apneias_npy", window=window)
