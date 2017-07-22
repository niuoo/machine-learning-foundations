# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import random

data = pd.read_csv('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat', header=None,
                   sep="\s|\t", engine='python')

Y = data[4]
N = len(Y)
MX = np.mat(data[[0, 1, 2, 3]])  # 将前4列数据矩阵化
MX = np.c_[np.ones(N), MX]  # 增加第一列全部为1


def train(rand=False, alpha=1):
    idx = range(N)
    if rand:
        idx = random.sample(idx, N)  # 打乱循环的次序
    cnt = 0
    w = np.zeros(5)
    while True:
        flag = True
        for i in idx:
            if np.sign(w * MX[i].T) != Y[i]:
                w = w + alpha * Y[i] * MX[i]
                flag = False
                cnt += 1
        if flag == True:
            break
    return cnt


def pre_random(n, alpha=1):
    count = 0
    for i in xrange(n):
        t = train(rand=True, alpha=alpha)
        count += t
    return count / n


def main():
    print train(False, 1)  # question 15 , output: 45
    # The process will last for about 2 minutes
    print pre_random(2000, alpha=1)  # question 16 , output: 39 or 40.
    # The process will last about 2 minutes
    print pre_random(2000, alpha=0.5)  # question 17 , output: 39 or 40.


if __name__ == '__main__':
    main()
