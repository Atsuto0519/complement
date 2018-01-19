#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pickle
import numpy as np
from chainer import optimizers, cuda
import argparse
import chainer
import random
import sys
sys.path.append('./LSTMpredictQAM')
sys.path.append('./CalcAIC')
from make_data import *

np.random.seed(50)

MODEL_PATH = "./LSTMpredictQAM/qam_model_with_option.pkl"
PICTURE_PATH = "./complete_LSTM.png"
INPUT_LENGTH = 40
PREDICTION_LENGTH = 20
PREDICTION_PATH = "./prediction_5.txt"
INITIAL_PATH = "./initial.txt"
LOSS_PATH = "./loss_with_noise.txt"
TRUE_LOSS_PATH = "./true_loss.txt"
AVERAGE_PATH = "./complete_LSTM.txt"
MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = 48
NUMBER_OF_CYCLES = 100


def predict_sequence(model, input_seq, output_seq, dummy):
    sequences_col = len(input_seq)
    model.reset_state()
    for i in range(sequences_col):
        x = chainer.Variable(xp.asarray(input_seq[i:i+1], dtype=np.float32)[:, np.newaxis])
        future = model(x, dummy)
    cpu_future = chainer.cuda.to_cpu(future.data)
    return cpu_future


def complete(seq, input_length, pre_length, *, initial_path, loss_path, prediction_path, average_path, forward_model, backward_model):
    # initial sequence
    forward_input_seq = np.array(seq[:int(input_length)])
    backward_input_seq = [seq[int(input_length+pre_length+i)] for i in range(input_length)]
    backward_input_seq = np.array(backward_input_seq[::-1])

    forward_output_seq = np.empty(0)
    backward_output_seq = np.empty(0)

    # append an initial value
    forward_output_seq = np.append(forward_output_seq, forward_input_seq[-1])
    backward_output_seq = np.append(backward_output_seq, backward_input_seq[-1])

    forward_model.train = False
    backward_model.train = False
    dummy = chainer.Variable(xp.asarray([0], dtype=np.float32)[:, np.newaxis])

    for i in range(pre_length):
        future = predict_sequence(forward_model, forward_input_seq, forward_output_seq, dummy)
        forward_input_seq = np.delete(forward_input_seq, 0)
        forward_input_seq = np.append(forward_input_seq, future)
        forward_output_seq = np.append(forward_output_seq, future)

        past = predict_sequence(backward_model, backward_input_seq, backward_output_seq, dummy)
        backward_input_seq = np.delete(backward_input_seq, 0)
        backward_input_seq = np.append(backward_input_seq, past)
        backward_output_seq = np.append(backward_output_seq, past)

    with open(initial_path, "w") as f:
        initial_plot = []
        for (i, v) in enumerate(seq.tolist()):
            f.write("{i} {v}\n".format(i=i, v=v))
            initial_plot.append([i, v])

    with open(prediction_path, "w") as f:
        forward_prediction_plot = []
        for (i, v) in enumerate(forward_output_seq.tolist(), start=forward_input_seq.shape[0]):
            f.write("{i} {v}\n".format(i=i-1, v=v))
            forward_prediction_plot.append([i-1, v])
        backward_prediction_plot = []
        for (i, v) in enumerate(backward_output_seq.tolist(), start=1):
            f.write("{i} {v}\n".format(i=input_length+pre_length-i, v=v))
            backward_prediction_plot.append([input_length+pre_length-i, v])

    # plot with matplotlib
    initial_plot = np.array(initial_plot).T
    forward_prediction_plot = np.array(forward_prediction_plot).T
    backward_prediction_plot = backward_prediction_plot[::-1]
    backward_prediction_plot = np.array(backward_prediction_plot).T
    average_prediction = np.array(forward_prediction_plot[1]+backward_prediction_plot[1])/2

    plt.plot(initial_plot[0], initial_plot[1], label='QAM wave')
    plt.plot(forward_prediction_plot[0], forward_prediction_plot[1], label='forward prediction')
    plt.plot(backward_prediction_plot[0], backward_prediction_plot[1], label='backward prediction')
    plt.plot(forward_prediction_plot[0], average_prediction, label='average prediction')
    plt.ylim(-1,1)
    plt.legend()
    plt.grid()
    plt.savefig(PICTURE_PATH)

    with open(average_path, "w") as f:
        for (i, v) in enumerate(average_prediction.tolist(), start=input_length):
            f.write("{i} {v}\n".format(i=i-1, v=v))

    with open(loss_path, "w") as f:
        initial_plot = []
        for (i, v) in enumerate(seq.tolist()):
            if (input_length-1 <= i and i < input_length+pre_length):
                f.write("{i} {v}\n".format(i=i, v=v))
                initial_plot.append([i, v])


if __name__ == "__main__":
    # 引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    model = pickle.load(open(MODEL_PATH, mode='rb'))

    # cuda環境では以下のようにすればよい
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    signal = []
    for i in range(NUMBER_OF_CYCLES) :
        signal.append(random.randint(0,7))

    # make data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    data = data_maker.make(signal,regulary=True,noise=False)
    sequences = data_maker.make_mini_batch(data, mini_batch_size=MINI_BATCH_SIZE, length_of_sequence=LENGTH_OF_SEQUENCE)

    sample_index = 45
    # 平均0，分散0.05のホワイトノイズを付与
    print("add noise")
    temp = np.array(sequences[sample_index])+np.random.normal(0,0.05,NUMBER_OF_CYCLES)

    complete(temp, INPUT_LENGTH, PREDICTION_LENGTH, initial_path=INITIAL_PATH, loss_path=LOSS_PATH, prediction_path=PREDICTION_PATH, average_path=AVERAGE_PATH, forward_model=model, backward_model=model)

    with open(TRUE_LOSS_PATH, "w") as f:
        for (i, v) in enumerate(sequences[sample_index]):
            if (INPUT_LENGTH-1 <= i and i < INPUT_LENGTH+PREDICTION_LENGTH):
                f.write("{i} {v}\n".format(i=i, v=v))
