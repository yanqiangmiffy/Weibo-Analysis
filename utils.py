import pandas as pd
from multiprocessing import Pool
from functools import partial
import numpy as np

def dosomething(row):
    # 添加计算操作

    return


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


# if __name__ == '__main__':  # 不要忘了这句话，否则运行.py文件时，无法启动multiprocessing
#     data['newcol'] = parallelize_on_rows(data, dosomething)