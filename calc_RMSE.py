#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse


PICTURE_PATH = "./complete.png"
LOSS_PATH = ["./true_loss.txt","./loss_with_noise.txt"]
LSM_COMPLETE_PATH =  "./complete_LSM.txt"
LSTM_COMPLETE_PATH = "./complete_LSTM.txt"
PICTURE_PATH = ["./complete_true.png", "./complete_noise.png"]


if __name__ == "__main__":
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', '-i', default=-1, type=int,
                        help='i=0:true_loss, i=1:noise_loss')
    args = parser.parse_args()

    # lossデータ
    loss_data = np.loadtxt(LOSS_PATH[args.index], delimiter=" ")
    loss_plot = loss_data.T

    # LSMデータ
    LSM_data = np.loadtxt(LSM_COMPLETE_PATH, delimiter=" ")
    LSM_plot = LSM_data.T

    # LSTMデータ
    LSTM_data = np.loadtxt(LSTM_COMPLETE_PATH, delimiter=" ")
    LSTM_plot = LSTM_data.T

    # 各補完のRMSEを計算する
    LSM_RMSE = np.sqrt(sum((loss_plot[1]-LSM_plot[1])**2)/len(loss_data))
    LSTM_RMSE = np.sqrt(sum((loss_plot[1]-LSTM_plot[1])**2)/len(loss_data))

    print(LSM_RMSE)
    print(LSTM_RMSE)

    # 試しにプロットしてみる
    plt.plot(loss_plot[0], loss_plot[1], label="true loss")
    plt.plot(LSM_plot[0],  LSM_plot[1],  label="LSM, RMSE="+str(LSM_RMSE))
    plt.plot(LSTM_plot[0], LSTM_plot[1], label="LSTM,RMSE="+str(LSTM_RMSE))
    plt.legend()
    plt.savefig(PICTURE_PATH[args.index])
