import sys
import json

import numpy as np

CHECK_POINT = 5000

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def loss_func(outputs, ground_truth):
    mse = np.square(ground_truth - outputs)
    return mse

def derivative_loss_func(outputs, ground_truth):
    derivative_mse = 2 * (outputs - ground_truth)
    return derivative_mse

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
                layer_input = layer.cal_outputs(layer_input)
                # print(layer_input, gt)
                return loss_func(layer_input, gt)
            layer_input = layer.cal_outputs(layer_input)
        # return layer_input

    def backward(self, ground_truths):
        '''
            outputs just one number between 0 and 1
        '''
        pre_delta = np.array([])
        for layer_idx in range(len(self.layers)-1, -1, -1):
            # print('Layer ', layer_idx)
            if len(pre_delta) == 0:
                error = derivative_loss_func(self.layers[layer_idx].outputs, ground_truths)
                delta = self.layers[layer_idx].cal_delta(error)
            else:
                error = np.matmul(self.layers[layer_idx+1].weights, pre_delta)
                delta = self.layers[layer_idx].cal_delta(error)
            pre_delta = delta
        for idx, l in enumerate(self.layers):
            l.fix_weights(self.learning_rate)
            # print("layer", idx, 'new_weights:', l.weights)

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
                layer_input = layer.cal_outputs(layer_input)
            res.append(layer_input)
        return res

class Neurons():
    '''
        one layer include weight and outputs(delta, z)
    '''
    def __init__(self, w_shape=None):
        self.weights = np.array([])
        if w_shape is not None:
            self.weights = self.init_weights(w_shape)
        self.delta = np.array([])
        self.outputs = []
        self.inputs = []

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
        if count == 2:
            res = [0.5, 0.6]
        elif count == 4:
            res = [0.5, 0.1, 0.2, 0.5]
        return np.reshape(res, weights_shape)
        return np.random.uniform(0.001, 1, weights_shape)

    def cal_outputs(self, inputs):
        if len(inputs) != len(self.weights):
            raise Exception('inputs error')
        self.inputs = np.reshape(inputs, (len(inputs),1))
        self.outputs = sigmoid(np.matmul(self.weights.T, inputs))
        # print(self.weights.T, '*', self.inputs, '=', self.outputs)
        self.outputs = self.outputs.reshape((len(self.outputs),1))
        return self.outputs

    def cal_delta(self, error):
        # print('delta: ', error, '*', derivative_sigmoid(self.outputs))
        # print('self.outputs: ', self.outputs)
        self.delta = np.multiply(error, derivative_sigmoid(self.outputs))
        if not isinstance(self.delta, type(np.array([]))):
            self.delta = np.array([self.delta])
        # print( '=', self.delta)
        return self.delta
    
    def fix_weights(self, learning_rate):
        # print('fix_weights', self.weights, learning_rate, '*', self.delta, 'X', self.inputs)
        self.weights = self.weights - learning_rate * np.multiply(self.delta, self.inputs)

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
    return [[.1, .1], [.5, .5]], np.array([1, 0])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def train(file_name):
    # x, y = generate_linear(100)
    # x, y = generate_XOR_easy()
    x, y = gen_input()
    w_shapes = [[2,2], [2,2], [2,1]]
    result_NN = NeuralNet(l_rate=0.8)
    # init weights
    for w in w_shapes:
        layer = Neurons(w)
        result_NN.add_layer(layer)
    count = 1
    while count:
        loss_list = []
        for inp, gt in list(zip(x, y)):
            nn_res = result_NN.forward(inp, gt)
            if count % CHECK_POINT == 0:
                loss_list.append(nn_res)
            result_NN.backward(gt)
        if count % CHECK_POINT == 0:
            print('epoch', count, 'Loss with MSE', np.max(loss_list))
            print('Weights:', result_NN.get_w())
            with open(file_name + '.json', 'w') as f:
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
    file_name = 'model'
    train(file_name)
    # test(file_name)
    sys.stdout.flush()
    sys.exit()