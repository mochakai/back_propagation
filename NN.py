import sys
import os
import json
from datetime import datetime
from argparse import ArgumentParser

import numpy as np

CHECK_POINT = 5000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def loss_func(outputs, ground_truth):
    x = (1 - 2 * ground_truth) * outputs
    softplus = np.log(1 + np.exp(x))
    return softplus
    # mse = np.square(ground_truth - outputs) / 2
    # return mse


def derivative_loss_func(outputs, ground_truth):
    derivative_softplus = (1 - 2 * ground_truth) / \
        (1 + np.exp((-1) * (1 - 2 * ground_truth) * outputs))
    return derivative_softplus
    # derivative_mse = (outputs - ground_truth)
    # return derivative_mse


class NN():
    def __init__(self, learning_rate=0.5):
        self.weights = []
        self.init_weights()
        self.l_rate = learning_rate

    def init_weights(self):
        for _ in range(2):
            self.weights.append(np.random.uniform(-1, 1, (2, 2)))
        # last layers
        self.weights.append(np.random.uniform(-1, 1, (1, 2)))
        self.out_h = [np.array([0, 0])] * 3
        self.out_h_act = [np.array([0, 0])] * 3

    def insert_weights(self, weight_dict):
        self.weights = []
        for val in weight_dict.values():
            for i in val:
                self.weights.append(np.array(i))

    def get_w(self):
        return self.weights

    def go_forward(self, inputs, ground_truth):
        # layer 0
        self.inputs = inputs.reshape((1, len(inputs)))
        out = np.matmul(self.weights[0], inputs)
        self.out_h[0] = out.reshape(len(out), 1)
        self.out_h_act[0] = sigmoid(self.out_h[0])
        # layer 1
        self.out_h[1] = np.matmul(self.weights[1], self.out_h_act[0])
        self.out_h_act[1] = sigmoid(self.out_h[1])
        # layer 2 (out)
        self.out_h[2] = np.matmul(self.weights[2], self.out_h_act[1])
        self.out_h_act[2] = sigmoid(self.out_h[2])
        outputs = self.out_h_act[2]
        loss = loss_func(self.out_h[2], ground_truth)
        return outputs, loss

    def go_backward(self, ground_truth):
        # print("self.out_h_act ", [self.out_h_act[i].shape for i in range(3)])
        # print("self.out_h. ", [self.out_h[i].shape for i in range(3)])
        # layer 2 (out)
        de_loss = derivative_loss_func(self.out_h[2], ground_truth)
        gradient_w2 = de_loss * self.out_h_act[1].T  # 1, 2
        self.weights[2] = np.subtract(
            self.weights[2], self.l_rate * gradient_w2)
        # layer 1
        tmp = (self.weights[2] *
               derivative_sigmoid(self.out_h_act[1].T)).T  # 2, 1
        # print(tmp, self.out_h_act[0].T)
        gradient_w1 = de_loss * np.matmul(tmp, self.out_h_act[0].T)
        # print("g1 ", gradient_w1)
        self.weights[1] = np.subtract(
            self.weights[1], self.l_rate * gradient_w1)
        # layer 0
        # print(self.weights[1], tmp.T, derivative_sigmoid(self.out_h_act[0].T))
        # 1, 2
        tmp = (np.matmul(tmp.T, self.weights[1]) *
               derivative_sigmoid(self.out_h_act[0].T)).T       # 2, 1
        gradient_w0 = de_loss * np.matmul(tmp, self.inputs)     # inputs: 1, 2
        self.weights[0] = np.subtract(
            self.weights[0], self.l_rate * gradient_w0)


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1 - 0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def gen_data(data_type):
    if data_type == 'linear':
        return generate_linear(100)
    elif data_type == 'xor':
        return generate_XOR_easy()
    else:
        raise Exception('Undefined training type :', data_type)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def train(file_name, learning_rate, training_type='xor', target_step=sys.maxsize):
    x, y = gen_data(training_type)
    network = NN(learning_rate)
    folder_name = datetime.now().strftime('%m_%d_%H-%M')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    count = 1
    while count < target_step:
        loss_list = []
        for inp, gt in list(zip(x, y)):
            nn_res, loss = network.go_forward(inp, gt)
            if count % CHECK_POINT == 0:
                loss_list.append(loss)
            network.go_backward(gt)
        if count % CHECK_POINT == 0:
            print('epoch', count, 'Loss: ', loss)
            # print('Weights:', network.get_w())
            with open(os.path.join(folder_name, file_name + str(count) + '.json'), 'w') as f:
                json.dump({'weight': network.get_w()}, f, cls=NumpyEncoder)
            with open(os.path.join(folder_name, file_name + '.json'), 'w') as f:
                json.dump({'weight': network.get_w()}, f, cls=NumpyEncoder)
        count += 1


def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] > 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict truth', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] > 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


def test(file_name, training_type='xor'):
    x, y = gen_data(training_type)
    with open(file_name, 'r') as f:
        model = json.load(f)

    test_NN = NN()
    test_NN.insert_weights(model)
    NN_res = []
    for inp, gt in list(zip(x, y)):
        res, _ = test_NN.go_forward(inp, gt)
        NN_res.append(res[0])
    # print(list(zip(NN_res, y)))
    NN_res = np.array(NN_res)
    show_result(x, y.reshape(y.size), NN_res.reshape(NN_res.size))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("train_or_test", help="train | test", type=str)
    parser.add_argument("t_type", help="linear | xor", type=str)
    parser.add_argument("file_name", help="your checkpoint file name", type=str)
    parser.add_argument("-l", "--learning_rate", help="your learning rate", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.train_or_test == 'train':
        train(args.file_name, args.learning_rate, args.t_type)
    elif args.train_or_test == 'test':
        test(args.file_name)
    else:
        print('Usage: python train_back_propagation.py <train | test> <linear | xor> <OUTPUT FILENAME>')
    sys.stdout.flush()
    sys.exit()
