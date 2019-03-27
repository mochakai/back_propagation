import sys
import os
import json
from datetime import datetime

import numpy as np

CHECK_POINT = 500

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
    # if len(outputs) != len(ground_truth):
    #     raise Exception('ground_truth shape different from outputs')
    derivative_softplus = (1 - 2 * ground_truth) / (1 + np.exp((-1) * (1 - 2 * ground_truth) * outputs))
    return derivative_softplus
    # derivative_mse = (outputs - ground_truth)
    # return derivative_mse

def delta_cross_input(delta, inputs):
    row = delta.size
    col = inputs.size
    res = []
    for row in delta:
        tmp = []
        for col in inputs[0]:
            tmp.append(row[0] * col)
        # print(tmp, 'ttttt')
        res.append(np.array(tmp))
    return np.array(res)

class NN():
    def __init__(self, learning_rate=0.5):
        self.weights = []
        self.init_weights()
        self.l_rate = learning_rate
    
    def init_weights(self):
        for _ in range(2):
            self.weights.append(np.random.uniform(0.001, 1, (2,2)))
        # last layers
        self.weights.append(np.random.uniform(0.001, 1, (1,2)))
        self.out_h = [np.array([0, 0])] * 3
        self.out_h_act = [np.array([0, 0])] * 3

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
        loss = loss_func(outputs, ground_truth)
        return outputs, loss

    def go_backward(self, ground_truth):
        # layer 2 (out)
        de_loss = derivative_loss_func(self.out_h[2], ground_truth)
        gradient_w2 = de_loss * self.out_h_act[1].T
        # print( self.weights[2] , self.l_rate , gradient_w2)
        self.weights[2] = self.weights[2] - self.l_rate * gradient_w2
        # layer 1
        tmp = (self.weights[2] * derivative_sigmoid(self.out_h_act[1].T)).T
        # print(tmp, self.out_h_act[0].T)
        gradient_w1 = de_loss * delta_cross_input(tmp, self.out_h_act[0].T)
        # print(gradient_w1)
        self.weights[1] = self.weights[1] - self.l_rate * gradient_w1
        # layer 0
        # print(self.weights[1], tmp.T, derivative_sigmoid(self.out_h_act[0].T))
        tmp = (np.matmul(self.weights[1], tmp) * derivative_sigmoid(self.out_h_act[0]))
        gradient_w0 = de_loss * delta_cross_input(tmp, self.inputs)
        self.weights[0] = self.weights[0] - self.l_rate * gradient_w0

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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def train(file_name, learning_rate, target_step=sys.maxsize):
    x, y = generate_linear(100)
    # x, y = generate_XOR_easy()
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
                loss_list.append(nn_res)
            network.go_backward(gt)
        if count % CHECK_POINT == 0:
            print('epoch', count, 'Loss: ', np.max(loss_list))
            print('Weights:', network.get_w())
            with open(os.path.join(folder_name, file_name + str(count) + '.json'), 'w') as f:
                json.dump({'weight': network.get_w()}, f, cls=NumpyEncoder)
            with open(os.path.join(folder_name, file_name + '.json'), 'w') as f:
                json.dump({'weight': network.get_w()}, f, cls=NumpyEncoder)
        count += 1

def test(file_name):
    with open(file_name + '.json', 'r') as f:
        model = json.load(f)
    test_NN = NN()
    test_NN.insert_weight(model['weight'])
    # x, y = generate_linear(10)
    x, y = generate_XOR_easy()
    res = test_NN.get_result(x)
    # print(list(zip(x, res)))
    print(list(zip(res, y)))

if __name__ == "__main__":
    file_name = 'XOR08'
    train(file_name, 0.5)
    load_path = os.path.join('03_26_03-37', file_name)
    # test(load_path)
    sys.stdout.flush()
    sys.exit()