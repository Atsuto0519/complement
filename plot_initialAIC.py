#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./CalcAIC')
import AIC
import LSM
from complete_data import INPUT_LENGTH,PREDICTION_LENGTH


# データ
data = np.loadtxt("initial.txt",delimiter=" ")

# データを表示
print("data")
print(data)

# データをわかりやすい配列に格納(pltに使いやすくしたいから)
data_x=[]
data_y=[]
for i in data :
    if (i[0] < INPUT_LENGTH-1 or i[0] >= INPUT_LENGTH+PREDICTION_LENGTH):
        data_x.append(i[0])
        data_y.append(i[1])

# データをプロット
plt.plot(data_x[:INPUT_LENGTH-1], data_y[:INPUT_LENGTH-1], data_x[INPUT_LENGTH-1:], data_y[INPUT_LENGTH-1:], label="data")

# 次数Nを決定
N = 15

# 次数0からNまで実行
for n in range(N+1) :

    # 実行する次数
    print("N="+str(n))

    # LSMで誤差と重みベクトルcoeを計算
    error, coe = LSM.LSM(data, int(n+1))
    print("error:"+str(error))
    print("coe:"+str(coe[::-1]))

    # ここで，このモデルのAICを計算
    l = AIC.l_MAX(list(coe[::-1]),data_x,data_y)
    AIC_n = AIC.AIC(l,n+1)
    print("l="+str(l)+", AIC("+str(n)+")="+str(AIC_n))

    # LSMで得た近似線をプロット
    test_x = np.arange(min(data_x), max(data_x), 1)
    plt.plot(test_x, LSM.quation_LSM(coe, test_x), label="polynomial of degree "+str(n)+", AIC("+str(n)+")="+str(AIC_n))

plt.title("Compare LSM of degree")
plt.legend()
plt.grid()
plt.ylim(-1,1)
plt.show()
