# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import random

data = pd.read_csv('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat', header=None,
                   sep="\s|\t", engine='python')

Y = data[4]
LEN = len(Y)
MX = np.mat(data[[0, 1, 2, 3]])  # 将前4列数据矩阵化
MX = np.c_[np.ones(LEN), MX]  # 增加第一列全部为1


def train(rand=False, alpha=1):
    idx = range(LEN)
    if rand:
        idx = random.sample(idx, LEN)  # 打乱循环的次序
    t = 0
    w = np.zeros(5)
    while True:
        flag = True
        for i in xrange(LEN):
            k = idx[i]
            if np.sign(w * MX[k].T) != Y[k]:
                w = w + alpha * Y[k] * MX[k]
                flag = False
                t += 1
        if flag == True:
            break
    return t


def pre_random(n, alpha=1):
    count = 0
    for i in xrange(n):
        t = train(rand=True, alpha=alpha)
        count += t
    return count / n


def main():
    print train(False, 1)  # question 15 , the answer is 45
    print pre_random(2000, alpha=1)  # question 16 , the answer is 40. The process will last about 2 minutes
    print pre_random(2000, alpha=0.5)  # question 17 , the answer is 39 or 40. The process will last about 2 minutes


if __name__ == '__main__':
    main()
