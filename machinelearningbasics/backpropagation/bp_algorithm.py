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
        #print("current layer: ");
        #print(layer);
        for neuron in layer:
            neuron['output'] = transfer(activate(layer_inputs, neuron['weights']));
        layer_inputs = list();
        for neuron in layer:
            layer_inputs.append(neuron['output']);
    return layer_inputs;

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
def update_weights(input, learning_rate, network):
    for i in range(len(network)):
        layer_input = input[:-1];
        if i != 0:
            layer_input = [neuron['output'] for neuron in network[i - 1]];
        for neuron in network[i]:
            for j in range(len(neuron['weights']) - 1):
                neuron['weights'][j] += learning_rate * neuron['delta'] * layer_input[j];
            neuron['weights'][len(neuron['weights']) - 1] += learning_rate * neuron['delta'];


def train_network(train_dataset, network, learning_rate, n_epoch):
    for epoch in range(n_epoch):
        sum_error = 0.0;
        for row in train_dataset:
            n_output = len(network[-1]);
            expected = [0 for i in range(n_output)];
            expected[row[-1]] = 1;
            # compute the feed foward result
            output = feed_forward(row, network);
            # compute the back propagation error
            bp_error(expected = expected, network = network);
            # Update the weights
            update_weights(row, learning_rate = learning_rate, network = network);
            sum_error += sum( [(expected[i] - output[i]) ** 2 for i in range(len(output))] );
        print(">epoch = %d, lrate = %.3f, error = %.3f" % (epoch, learning_rate, sum_error))

def derivatives(sigmoid_output):
    return sigmoid_output * (1.0 - sigmoid_output);


def predict(input, network):
    output = feed_forward(input, network);
    return output.index(max(output));

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


def test_training():
    random.seed(1);
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = init_network(n_inputs, 2, n_outputs)
    train_network(train_dataset = dataset, network = network, learning_rate = 0.5, n_epoch = 20);
    for layer in network:
        print(layer)
    for row in dataset:
        prediction = predict(row, network)
        print('Expected=%d, Got=%d' % (row[-1], prediction))

if __name__ == '__main__':
    #test_network = test_init();
    #test_feed_forward();
    #test_bp_error();
    test_training();

