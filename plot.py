import numpy as np
import matplotlib.pyplot as plt


files = ['GCN', 'GIN', 'SAGEConv']

epochs = 50
plt.figure(0)
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('test accuracy')

plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('train accuracy')

plt.figure(3)
plt.xlabel('epoch')
plt.ylabel('validation accuracy')


for f_ in files:
    x = list(range(epochs))
    y_loss = []
    y_train = []
    y_val = []
    y_test = []
    with open(f_, 'r') as f:
        for line in f.readlines():
            line = line.replace(':', ',')
            line = line.replace('\n', '')
            line = line.split(',')
            y_loss.append(line[3])
            y_train.append(line[5])
            y_val.append(line[7])
            y_test.append(line[9])

        y_loss = [float(x) for x in y_loss]
        y_test = [float(x) for x in y_test]
        y_train = [float(x) for x in y_train]
        y_val = [float(x) for x in y_val]
    plt.figure(0)
    plt.plot(x, y_loss, label=f_)
    plt.figure(1)
    plt.plot(x, y_test, label=f_)
    plt.figure(2)
    plt.plot(x, y_train, label=f_)
    plt.figure(3)
    plt.plot(x, y_val, label=f_)

plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.figure(3)
plt.legend()
plt.show()
