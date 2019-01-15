import random;
import math;

def init_network(input_number, hidden_number, output_number):
    network = list();
    hidden_layer = [{'weights': [random.random() for i in range(input_number + 1)]} for i in range(hidden_number)];
    output_layer = [{'weights': [random.random() for i in range(hidden_number + 1)]} for i in range(output_number)];
    network.append(hidden_layer);
    network.append(output_layer);
    return network;

def feed_forword(inputs, network):
    layer_inputs = inputs;
    for layer in network:
        for neuron in layer:
            neuron['output'] = transfer(activate(layer_inputs, neuron['weights']));
        layer_inputs = list();
        for neuron in layer:
            layer_inputs.append(neuron['output']);

# errors: partial(Loss_funct)/partial(a_i), where a_i is the activated output of a neuron
# neuron['delta']: partial(Loss_func)/partial(z_i), where z_i is the linear sum of input * weights
def bp_error(network_output, expected, network):
    errors = list();
    for i in range(len(network)):
        if (i == len(network-1)): # if the current layer is the output layer
            error = (network_output[i] - expected[i]);
            errors.append(error);
        else: #if the current layer is not the output layer
            layer = network[i + 1];
            for i in range(len(network[i])):
                error = 0;
                for neuron in layer:
                    error += neuron['delta']*neuron['weights'][i];
                errors.append(error);
        for neuron in network[i]:
            neuron['delta'] = errors[i]* derivatives(neuron['output']);


def activate(inputs, weights):
    activation = weights[-1];
    for i in range(len(weights)-1):
        activation += inputs[i] * weights[i];
    return activation;

def transfer(input):
    return 1 / (1 + math.exp(-input));

def update_weights(network, learning_rate, inputs):
    pass;

def derivatives(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output);


def test_init():
    random.seed(10);
    network = init_network(2, 1, 2);
    for layer in network:
        print(layer);
    return network;

def test_feed_forward():
    random.seed(10);
    number_inputs = 2;
    inputs = list();
    for i in range(number_inputs):



if __name__ == '__main__':
    test_network = test_init();

