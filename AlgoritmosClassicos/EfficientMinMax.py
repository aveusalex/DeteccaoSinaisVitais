# defining an efficient min max scaler

def minmax_scaler(data):
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    return (data - data_min)/data_range