import backprop_data

import backprop_network

import matplotlib.pyplot as plt
import numpy as np

## A
training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

net = backprop_network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


# ## B
# training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
# rates = [0.001,0.01, 0.1]

# train_accuracy = [0] * 3
# train_loss = [0] * 3
# test_accuracy = [0] * 3
# epoch_num = 30

# for i in range(3):
#     net = backprop_network.Network([784, 40, 10])
#     train_accuracy[i], train_loss[i], test_accuracy[i] = net.SGD(training_data, epochs=epoch_num,
#                                                                  mini_batch_size=10,learning_rate=rates[i], test_data=test_data)
    
# for i in range(3):
#     plt.plot(np.arange(epoch_num), train_accuracy[i], label="rate = {}".format(rates[i]))
# plt.xlabel('Epochs')
# plt.ylabel('Train Accuracy')
# plt.legend()
# plt.savefig("B_train_acc_${i}.png")
# plt.clf()
# for i in range(3):
#     plt.plot(np.arange(epoch_num), train_loss[i], label="rate = {}".format(rates[i]))
# plt.xlabel('Epochs')
# plt.ylabel('Train Loss')
# plt.legend()
# plt.savefig("B_loss_${i}.png")
# plt.clf()
# for i in range(3):
#     plt.plot(np.arange(epoch_num), test_accuracy[i], label="rate = {}".format(rates[i]))
# plt.xlabel('Epochs')
# plt.ylabel('Test Accuracy')
# plt.legend()
# plt.savefig("B_test_acc_${i}.png")
# plt.clf()
    
# ## C
# training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
# net = backprop_network.Network([784, 40, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# ## D
# training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
# net = backprop_network.Network([784, 50, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

