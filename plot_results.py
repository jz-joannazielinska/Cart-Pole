import numpy as np
import matplotlib.pyplot as plt

buckets_1 = [i for i in range(27)] #(2, 2, 8, 3)
buckets_2 = [i for i in range(27, 54)] #(2, 2, 6, 6)
buckets_3 = [i for i in range(54, 81)] #(1, 2, 6, 3)

m_vals = [i for i in range(5)]


def read_values(file):
    f = open(file, 'r')
    lines = f.readlines()[1:]
    params = lines[0]
    lines = lines[1:]
    f.close()
    lines = [[int(line.rstrip().split(',')[0].strip()), float(line.rstrip().split(',')[1].strip())] for line in lines]
    lines = np.array(lines)
    lines = lines[:, 1]

    return params, lines


def calculate_moving_average(values, step_size, window_size):
    i = 0
    avg = []
    while i + window_size <= len(values):
        avg.append(np.mean(values[i: i + window_size + 1]))
        i += step_size
    return avg


def calculate_avg_and_std(values, step_size=1, window_size=100):

    avg_values = []
    std_values = []
    i = 0
    while i + window_size <= len(values[0]):
        tmp = [values[k, i: i + window_size + 1] for k in range(len(values))]
        avg_values.append(np.mean(tmp))
        std_values.append(np.std(tmp))
        i += step_size

    avg_values = np.array(avg_values)
    std_values = np.array(std_values)

    return avg_values, std_values


def make_plots(bucket, num, plt_name, output_name):

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.8, wspace=0.4)
    plt_name = plt_name + '_' + '_' + str(num)
    k = 0
    for i in bucket[:9]:
        ax = fig.add_subplot(3, 3, k+1)
        k += 1
        file = output_name + '_' + str(i)
        params, lines = read_values(file)
        avgs = calculate_moving_average(values=lines, step_size=1, window_size=100)
        plt.ylim(0, 520)
        plt.yticks([0, 100, 200, 300, 400, 500])
        plt.plot(avgs)
        plt.title(params)

    plt.savefig(plt_name + "_1")

    fig = plt.figure(figsize=(20,10))
    fig.subplots_adjust(hspace=0.8, wspace=0.4)
    k = 0
    for i in bucket[9:18]:
        ax = fig.add_subplot(3, 3, k+1)
        k += 1
        file = output_name + '_' + str(i)
        params, lines = read_values(file)
        avgs = calculate_moving_average(values=lines, step_size=1, window_size=100)
        plt.ylim(0, 520)
        plt.yticks([0, 100, 200, 300, 400, 500])
        plt.plot(avgs)
        plt.title(params)

    plt.savefig(plt_name + "_2")

    fig = plt.figure(figsize=(20,10))
    fig.subplots_adjust(hspace=0.8, wspace=0.4)
    k = 0
    for i in bucket[18:27]:
        ax = fig.add_subplot(3, 3, k+1)
        k += 1
        file = output_name + '_' + str(i)
        params, lines = read_values(file)
        avgs = calculate_moving_average(values=lines, step_size=1, window_size=100)
        plt.ylim(0, 520)
        plt.yticks([0, 100, 200, 300, 400, 500])
        plt.plot(avgs)
        plt.title(params)

    plt.savefig(plt_name + "_3")


def make_plots2(file_name, plt_name):
    values = []

    for i in m_vals:
        file = file_name + str(i)
        params, lines = read_values(file)
        values.append(lines)

    values = np.array(values)
    avg_values, std_values = calculate_avg_and_std(values, 1, 500)
    avg_std_up = avg_values + std_values
    avg_std_down = avg_values - std_values

    fig = plt.figure(figsize=(20, 10))
    plt.plot(avg_values, label="Rolling average reward, window size = 500")
    plt.plot(avg_std_up, linestyle='--', label="Average + standard deviation")
    plt.plot(avg_std_down, linestyle='--', label="Average - standard deviation")
    plt.legend()
    plt.title(params)
    plt.savefig(plt_name)
