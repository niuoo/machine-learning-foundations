# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import random

train_data = pd.read_csv('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_train.dat', header=None,
                         sep="\s|\t", engine='python')
test_data = pd.read_csv('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_test.dat', header=None,
                        sep="\s|\t", engine='python')

Y = train_data[4]
N = len(Y)
X = np.mat(train_data[[0, 1, 2, 3]])  # 将前4列数据矩阵化
X = np.c_[np.ones(N), X]  # 增加第一列全部为1

YTEST = test_data[4]
MXTEST = np.mat(test_data[[0, 1, 2, 3]])  # 将前4列数据矩阵化
MXTEST = np.c_[np.ones(N), MXTEST]  # 增加第一列全部为1


def error_sum(w, X, Y):
    sum = 0
    n = len(Y)
    for i in xrange(n):
        if np.sign(w * X[i].T) != Y[i]:
            sum += 1
    return sum


def train(pocket=True, update=50):
    w = np.zeros(5)
    error = error_sum(w, X, Y)
    w_pocket = w
    cnt = 0
    while cnt < update:
        idx = random.sample(range(N), N)  # 打乱循环的次序
        for i in idx:
            if np.sign(w * X[i].T) != Y[i]:
                w = w + Y[i] * X[i]
                if pocket:
                    e = error_sum(w, X, Y)
                    if e < error:
                        error = e
                        w_pocket = w
                    cnt += 1
                    break
                else:
                    cnt += 1

    if pocket:
        return error_sum(w_pocket, MXTEST, YTEST) / float(len(YTEST))
    else:
        return error_sum(w, MXTEST, YTEST) / float(len(YTEST))


def pre_random(n, pocket=True, update=50):
    error_rate = 0
    for i in xrange(n):
        error_rate += train(pocket, update)
    return error_rate / n


def main():
    # The process will last for  several minutes
    print pre_random(20, True, 50)  # question 18 , output: 0.13508
    print pre_random(20, False, 50)  # question 19 , output: 0.23189
    print pre_random(20, True, 100)  # question 20 , output: 0.1181


if __name__ == '__main__':
    main()
