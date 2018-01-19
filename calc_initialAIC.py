#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./CalcAIC')
import AIC
import LSM
from complete_data import INPUT_LENGTH,PREDICTION_LENGTH


PICTURE_PATH = "./complete_LSM.png"
COMPETE_PATH = "./complete_LSM.txt"

# データ
data = np.loadtxt("initial.txt",delimiter=" ")

# データを表示
# print("data")
# print(data)

# データをわかりやすい配列に格納(pltに使いやすくしたいから)
data_x=[]
data_y=[]
for i in data :
    if (i[0] < INPUT_LENGTH-1 or i[0] >= INPUT_LENGTH+PREDICTION_LENGTH):
        data_x.append(i[0])
        data_y.append(i[1])

# 次数Nを決定
N = 11

print("N="+str(N))

# LSMで誤差と重みベクトルcoeを計算
error, coe = LSM.LSM(data, int(N+1))
print("error:"+str(error))
print("coe:"+str(coe[::-1]))

# ここで，このモデルのAICを計算
l = AIC.l_MAX(list(coe[::-1]),data_x,data_y)
print("l="+str(l)+",AIC("+str(N)+")="+str(AIC.AIC(l,N+1)))

# データをプロット
plt.plot(data_x[:INPUT_LENGTH-1], data_y[:INPUT_LENGTH-1],
         data_x[INPUT_LENGTH-1:], data_y[INPUT_LENGTH-1:], label="data")

# LSMで得た近似線をプロット
LSM_x = np.arange(min(data_x), max(data_x)+1, 1)
LSM_y = LSM.quation_LSM(coe, LSM_x)
plt.plot(LSM_x, LSM_y)

plt.ylim(-1,1)
plt.grid()
plt.savefig(PICTURE_PATH)

# LSMで得た補完部分を出力
with open(COMPETE_PATH, "w") as f:
    for (i,j) in enumerate(LSM_x):
        if (INPUT_LENGTH-1 <= j and j < INPUT_LENGTH+PREDICTION_LENGTH):
            f.write("{i} {v}\n".format(i=int(j), v=LSM_y[i]))
