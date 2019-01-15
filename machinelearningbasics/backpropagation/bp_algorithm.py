import random;
import math;

def init_network(input_number, hidden_number, output_number):
    network = list();
    hidden_layer = [{'weights': [random.random() for i in range(input_number + 1)]} for i in range(hidden_number)];
    output_layer = [{'weights': [random.random() for i in range(hidden_number + 1)]} for i in range(output_number)];
    network.append(hidden_layer);
    network.append(output_layer);
    return network;


# Compute the network in a feedforward manner
def feed_forward(inputs, network):
    layer_inputs = inputs;
    for layer in network:
        print("current layer: ");
        print(layer);
        for neuron in layer:
            neuron['output'] = transfer(activate(layer_inputs, neuron['weights']));
        layer_inputs = list();
        for neuron in layer:
            layer_inputs.append(neuron['output']);

# errors: partial(Loss_funct)/partial(a_i), where a_i is the activated output of a neuron
# neuron['delta']: partial(Loss_func)/partial(z_i), where z_i is the linear sum of input * weights
def bp_error(expected, network):
    # errors = list();
    # for i in range(len(network)):
    #     if (i == len(network-1)): # if the current layer is the output layer
    #         error = (network_output[i] - expected[i]);
    #         errors.append(error);
    #     else: #if the current layer is not the output layer
    #         layer = network[i + 1];
    #         for i in range(len(network[i])):
    #             error = 0;
    #             for neuron in layer:
    #                 error += neuron['delta']*neuron['weights'][i];
    #             errors.append(error);
    #     for neuron in network[i]:
    #         neuron['delta'] = errors[i] * derivatives(neuron['output']);
    for i in reversed(range(len(network))):
        errors = list();
        layer = network[i];
        if (i == len(network) -1):
            for j in range(len(layer)):
                errors.append(-layer[j]['output'] + expected[j]);
        else:
            for j in range(len(layer)):
                error = 0.0
                for next_layer_neuron in network[i + 1]:
                    error += next_layer_neuron['delta'] * next_layer_neuron['weights'][j];
                errors.append(error);
        for j in range(len(layer)):
            neuron = layer[j];
            neuron['delta'] = errors[j] * derivatives(neuron['output']);

def activate(inputs, weights):
    activation = weights[-1];
    for i in range(len(weights)-1):
        activation += inputs[i] * weights[i];
    return activation;

def transfer(input):
    return 1 / (1 + math.exp(-input));


# This function updates the weights of the network based on the computed partial derivatives.
def update_weights(inputs, learning_rate, network):
    for i in range(len(network)):
        layer_inputs = inputs[:-1];
        if i != 0:
            layer_inputs = [neuron['output'] for neuron in network[i - 1]];
        for neuron in network[i]:
            for j in range(len(neuron['weights'] - 1)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * layer_inputs[j];
            neuron['weights'][len(neuron['weights']) - 1] += learning_rate * neuron['delta'];


def train_network():
    pass;

def derivatives(sigmoid_output):
    return sigmoid_output * (1.0 - sigmoid_output);


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
        inputs.append(random.random());
    print("inputs:");
    print(inputs);
    test_network = test_init();
    print("test network");
    print(test_network);
    feed_forward(inputs,test_network);
    print("feed-forward computed network");
    print(test_network);

def test_bp_error():
    network = [
        [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
         {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    expected = [0, 1]
    bp_error(expected, network)
    for layer in network:
        print(layer)


if __name__ == '__main__':
    #test_network = test_init();
    #test_feed_forward();
    test_bp_error();

