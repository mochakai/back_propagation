import sys
import os
import json
from datetime import datetime

import numpy as np

CHECK_POINT = 500
LOGGER = False
DEBUG = False

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

def debug_log(*args):
    if LOGGER:
        print(*args)

class NeuralNet():
    def __init__(self, l_rate=0.5):
        self.layers = []
        self.learning_rate = l_rate
    
    def get_w(self):
        res = {}
        for idx, l in enumerate(self.layers):
            res[idx] = l.get_weight()
        return res

    def add_layer(self, layer):
        self.layers.append(layer)
        print('layer {} shape: {}'.format(len(self.layers), layer.weights.shape))

    def forward(self, inputs, gt):
        layer_input = inputs
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers)-1:
                output = layer.cal_outputs(layer_input, True)
                debug_log(output, gt)
                return loss_func(output, gt)
            layer_input = layer.cal_outputs(layer_input)
        # return layer_input

    def backward(self, ground_truths):
        '''
            outputs just one number between 0 and 1
        '''
        pre_delta = np.array([])
        for layer_idx in range(len(self.layers)-1, -1, -1):
            debug_log('Layer ', layer_idx)
            if len(pre_delta) == 0:
                error = derivative_loss_func(self.layers[layer_idx].outputs, ground_truths)
                delta = self.layers[layer_idx].cal_delta(error, True)
            else:
                if pre_delta.size == 1:
                    pre_delta = pre_delta.reshape((1,1))
                error = np.matmul(self.layers[layer_idx+1].weights, pre_delta)
                # print('error: ', self.layers[layer_idx+1].weights, 'X', pre_delta, '=', np.matmul(self.layers[layer_idx+1].weights, pre_delta))
                delta = self.layers[layer_idx].cal_delta(error)
            pre_delta = delta
        for layer_idx in range(len(self.layers)-1, -1, -1):
            self.layers[layer_idx].fix_weights(self.learning_rate)
            debug_log("layer", layer_idx, 'new_weights:', self.layers[layer_idx].weights)

    def insert_weight(self, weight_dict):
        for val in weight_dict.values():
            tmp = Neurons()
            tmp.set_weights(val)
            self.layers.append(tmp)

    def get_result(self, inputs):
        res = []
        for i in inputs:
            layer_input = i
            for idx, layer in enumerate(self.layers):
                if idx == len(self.layers)-1:
                    outpot = layer.cal_outputs(layer_input, True)
                    return sigmoid(outpot)
                layer_input = layer.cal_outputs(layer_input)
            res.append(layer_input)
        return res

class Neurons():
    '''
        one layer include weight and outputs(delta, z)
    '''
    def __init__(self, w_shape=None, bias=None):
        self.weights = np.array([])
        if w_shape is not None:
            self.weights = self.init_weights(w_shape)
        self.delta = np.array([])
        self.outputs = []
        self.inputs = []
        self.bias = 0 if bias is None else bias

    def get_weight(self):
        return self.weights

    def set_weights(self, w):
        self.weights = np.array(w)

    def init_weights(self, weights_shape):
        '''
            random float between 0.001 and 1
        '''
        res = []
        count = 1
        for n in weights_shape:
            count *= n
        for i in range(count):
            res.append(i)
        if DEBUG:
            if count == 2:
                res = np.array([0.4, 0.5])
            elif count == 4:
                res = np.array([[0.15, 0.25], [0.2, 0.3]])
            return res
        # if count == 2:
        #     res = [0.4, 0.5, 0.45, 0.55]
        # elif count == 4:
        #     res = [0.15, 0.25, 0.2, 0.3]
        # if DEBUG:
        #     return np.reshape(res, (2,2))
        return np.random.uniform(0.001, 1, weights_shape)

    def cal_outputs(self, inputs, last=False):
        if len(inputs) != len(self.weights):
            raise Exception('inputs error')
        self.inputs = np.reshape(inputs, (len(inputs),1))
        if last:
            self.outputs = np.matmul(self.weights.T, inputs) + self.bias
        else:
            self.outputs = sigmoid(np.matmul(self.weights.T, inputs) + self.bias)
        debug_log(self.weights.T, '*', self.inputs, '+', self.bias, '=', self.outputs)
        # self.outputs = self.outputs.reshape((len(self.outputs),1))
        return self.outputs

    def cal_delta(self, error, last=False):
        debug_log('delta: ', error, '*', derivative_sigmoid(self.outputs))
        debug_log('self.outputs: ', self.outputs)
        if last:
            self.delta = error
        else:
            der_sig = derivative_sigmoid(self.outputs)
            if len(der_sig) > 1 and der_sig.ndim == 1:
                der_sig = np.reshape(der_sig, (len(der_sig), 1))
            # print('---------------', error, der_sig)
            self.delta = np.multiply(error, der_sig)
        if not isinstance(self.delta, type(np.array([]))):
            self.delta = np.array([self.delta])
        debug_log( '=', self.delta)
        return self.delta
    
    def fix_weights(self, learning_rate):
        if self.weights.size > self.delta.size:
            length = len(self.delta)
            delta = np.reshape(self.delta, (1, length))
            delta = np.repeat(delta, length, axis=0)
        else:
            delta = self.delta
        debug_log('fix_weights', self.weights, learning_rate, '*', delta, 'X', self.inputs)
        self.weights = self.weights - learning_rate * np.multiply(delta, self.inputs)

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

def gen_input():
    return [[.1, .1], [.1, .1], [.5, .5]], np.array([[1], [1], [0]])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def train(file_name, learning_rate, target_step=sys.maxsize):
    # x, y = generate_linear(100)
    x, y = generate_XOR_easy()
    if DEBUG:
        x, y = gen_input()
    w_shapes = [[2,2], [2,2], [2,1]]
    result_NN = NeuralNet(l_rate=learning_rate)
    # init weights
    for idx,w in enumerate(w_shapes):
        layer = Neurons(w)
        result_NN.add_layer(layer)
    
    folder_name = datetime.now().strftime('%m_%d_%H-%M')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    count = 1
    while count < target_step:
        loss_list = []
        for inp, gt in list(zip(x, y)):
            nn_res = result_NN.forward(inp, gt)
            if count % CHECK_POINT == 0:
                loss_list.append(nn_res)
            result_NN.backward(gt)
        if count % CHECK_POINT == 0:
            print('epoch', count, 'Loss with MSE', np.max(loss_list))
            print('Weights:', result_NN.get_w())
            with open(os.path.join(folder_name, file_name + str(count) + '.json'), 'w') as f:
                json.dump({'weight': result_NN.get_w()}, f, cls=NumpyEncoder)
            with open(os.path.join(folder_name, file_name + '.json'), 'w') as f:
                json.dump({'weight': result_NN.get_w()}, f, cls=NumpyEncoder)
        count += 1

def test(file_name):
    with open(file_name + '.json', 'r') as f:
        model = json.load(f)
    test_NN = NeuralNet()
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