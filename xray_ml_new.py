import tensorflow as tf
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

# use this import if the above has throws an error on Macs
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from ml_helpers_new import load_data, one_hot_matrix, create_placeholders, initialize_parameters, forward_propagation, compute_cost, random_mini_batches
from timeit import default_timer as timer

#disables AVX warning 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate = 0.0001,
          iterations = 4000, minibatch_size = 16, layer1 = 10, layer2 = 7, beta = 0.01, dropout = 1.0, istanh1 = False, istanh2 = False, batchnorm = True, print_cost = True):
    """ Three-layer NN to predict Fe coordination numbers around oxygen.
        Default is L2 regularization and Adam. Return optimized parameters.

        Arguments:
        ----------------------------
        X_train : array (170, 64% of data)

        Y_train : array (numOutputNodes, 64% of data)

        X_dev : array (170, 16% of data)

        Y_dev : array(numOutputNodes, 16% of data)

        numOutputNodes: int
            Determined by examning the maximum Fe coordination number of each oxygen.
        
        learning_rate, iterations, minibatch_size: as named

        layer1, layer2: number of nodes in the first and second hidden layers

        beta: regularization parameter for L2 regularization in the cost function

        dropout: probability of keeping a node in dropout

        istanh1, istanh2: determines whether the first and second layers use a tanh or relu activation function (True: tanh, False: relu)

        print_cost: boolean, decides whether or not to print the cost during training

        Returns:
        -----------------------------
        accs : list of accuracies
        parameters : dict
            weights and biases of each layer.{'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}
    """
    
    # reset all variables to allow the network to be trained multiple times with different hyperparameters
    sess = tf.Session()
    tf.reset_default_graph()

    tf.set_random_seed(1)                             # to keep consistent results
    seed = 1                                          # to keep consistent results
    n_x = X_train.shape[0]                            # n_x : input size (the other dimension is the number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # holds data for graphing
    dev_costs = []
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layer1, layer2, numOutputNodes)  # Initialize parameters
    training = tf.placeholder_with_default(False, shape=(), name='training') # Create a boolean to use for implementing batch norm and dropout correctly

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, training, istanh1, istanh2, batchnorm, dropout)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y, parameters, beta)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this section is needed to implement batch normalization
    with tf.control_dependencies(update_ops):
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # used to save the model later
    saver = tf.train.Saver()

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    
    # used to calculate the recall percentage on the label '0'
    zero = tf.constant(0, dtype=tf.int64)
    where1 = tf.equal(tf.argmax(Y), zero)
    where2 = tf.equal(tf.argmax(Z3), zero)
    where3 = tf.logical_and(where1, where2)
    true_positives = tf.reduce_sum(tf.cast(where3, tf.int32))
    all_positives = tf.reduce_sum(tf.cast(where1, tf.int32))
    recall = tf.cast(tf.divide(true_positives, all_positives), "float")

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # Print the hyperparameters for this particular model
        print('Learning Rate: %s, Mini-Batch Size: %d, Beta: %s, %d Nodes in Layer 1, %d Nodes in Layer 2, %d Output Nodes, %d Iterations, %s Dropout Prob, First Layer Tanh: %s, Second Layer Tanh: %s, Batch Norm: %s' \
        % (str(learning_rate).rstrip('0'), minibatch_size, str(beta).rstrip('0'), layer1, layer2, numOutputNodes, iterations, str(dropout).rstrip('0'), istanh1, istanh2, batchnorm))
        
        for epoch in range(iterations):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            num_minibatches = len(minibatches) 

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session on one minibatch, and add the cost to the epoch cost
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, training: True})
                epoch_cost += minibatch_cost / num_minibatches
            
            # Save the training and dev cost every 5 epochs
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                dev_cost = sess.run(cost, feed_dict={X:X_dev, Y: Y_dev, training: False})
                dev_costs.append(dev_cost)
            
            # Print the cost every 400 epochs
            if print_cost == True and epoch % 400 == 0: # used during testing to ensure gradient descent is working properly
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, training: False})
                train_recall = recall.eval({X: X_train, Y: Y_train, training: False})
                print("Training cost after iteration %i: %f, accuracy: %f, recall: %f" % (epoch, epoch_cost, train_accuracy, train_recall))

                dev_cost = sess.run(cost, feed_dict={X:X_dev, Y: Y_dev, training: False})
                dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev, training: False})
                dev_recall = recall.eval({X: X_dev, Y: Y_dev, training: False})
                print("Dev cost after iteration %i: %f, accuracy: %f, recall: %f" % (epoch, dev_cost, dev_accuracy, dev_recall))

        # Plot the cost; comment out if the pop-up graphs are not needed
        if print_cost:
            iter_num = np.arange(iterations / 5) * 5
            plt.plot(iter_num, np.squeeze(costs), label = 'training')
            plt.plot(iter_num, np.squeeze(dev_costs), label = 'cross-validation')
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title('Cost vs. Iterations')
            plt.legend()
            plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        saver.save(sess, "./coord_models/model_{}_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, layer2, beta))
        
        train_acc = accuracy.eval({X: X_train, Y: Y_train, training: False})
        dev_acc = accuracy.eval({X: X_dev, Y: Y_dev, training: False})
        accs = [train_acc, dev_acc]

        print("Train Accuracy:", train_acc)
        print("Dev Accuracy:", dev_acc)

    return accs, parameters

def train_multiple_models(X_train, Y_train, X_dev, Y_dev, numOutputNodes, iterations, hyperparams, print_cost = True):
    """ Allows for the training of different settings of hyperparameters in one function.
        
        Arguments:
        ----------------------------
        X_train, Y_train, X_dev, Y_dev, numOutputNodes, iterations, print_cost: used in model()
        hyperparams: a list of dictionaries of hyperparameters for testing

        Returns:
        ----------------------------
        results: a dictionary of the dev accuracy corresponding to each setting of hyperparameters
        best: a list of the settings of hyperparameters with the lowest dev set error
        params[best]: the parameters corresponding to the best hyperparameter setting
    """
    
    results = {}
    params = {}

    try:
        # extract the hyperparameters from one item in hyperparams
        for h in hyperparams:
            learning_rate = h['learning_rate'] 
            layer1 = h['layer1']
            layer2 = h['layer2']
            minibatch_size = h['minibatch_size']
            beta = h['beta']
            dropout = h['dropout']
            istanh1 = h['istanh1']
            istanh2 = h['istanh2']
            batchnorm = h['batchnorm']
            
            # train the model with the given hyperparameters
            accs, parameters = model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate, iterations, minibatch_size, layer1, layer2, beta, dropout, istanh1, istanh2, batchnorm, print_cost)
            
            results[frozenset(h.items())] = accs[1] # store the dev test accuracies in a dictionary
            params[frozenset(h.items())] = parameters # do the same for the learned parameters, to be retrieved at the end
    
    except KeyboardInterrupt: # allow for exiting the for loop in case we want to stop testing all the hyperparameters; to use, press Ctrl+C in terminal
        pass

    best = max(results, key=results.get) # finds what setting of hyperparameters had the highest dev accuracy

    return results, list(best), params[best]

def train_models_change_layer1s(X_train, Y_train, X_dev, Y_dev, numOutputNodes, iterations, layer1s, print_cost = True):
    """ Same as train_multiple_models(), except all hyperparameters are fixed besides layer1:
    Learning rate: 0.0003
    Mini-batch size: 16 
    Layer 2: 7 nodes
    Beta: 0.01 (regularization)
    Dropout: 1.0 (i.e. no dropout)
    Activation functions: ReLU/ReLU
    Batchnorm: True
    """
    
    results = {}
    params = {}

    try:
        for layer1 in layer1s:
            accs, parameters = model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, 0.0003, iterations, 16, layer1, 7, 0.01, 1.0, False, False, True, print_cost)

            results[layer1] = accs[1]
            params[layer1] = parameters

    except KeyboardInterrupt:
        pass

    best = max(results, key=results.get)

    accuracies = [results[l] for l in layer1s]

    # plots a graph of cross-validation set accuracy vs. # of nodes in the first layer
    if print_cost:
        plt.plot(layer1s, accuracies)
        plt.ylabel('accuracy')
        plt.xlabel('hidden nodes in first layer')
        plt.title("learning rate = 0.0003, mini-batch = 16, beta = 0.01, layer2 = 7 nodes")
        plt.show()
    
    return results, best, params[best]


def train_models_change_layer2s(X_train, Y_train, X_dev, Y_dev, numOutputNodes, iterations, layer2s, print_cost = True):
    """ Same as train_multiple_models(), except all hyperparameters are fixed besides layer2:
    Learning rate: 0.0003
    Mini-batch size: 16 
    Layer 1: 12 nodes
    Beta: 0.01 (regularization)
    Dropout: 1.0 (i.e. no dropout)
    Activation functions: ReLU/ReLU
    Batchnorm: True
    """

    results = {}
    params = {}

    try:
        for layer2 in layer2s:
            accs, parameters = model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, 0.0003, iterations, 16, 10, layer2, 0.01, 1.0, False, False, True, print_cost)

            results[layer2] = accs[1]
            params[layer2] = parameters

    except KeyboardInterrupt:
        pass

    best = max(results, key=results.get)

    accuracies = [results[l] for l in layer2s]

    # plots a graph of cross-validation set accuracy vs. # of nodes in the second layer
    if print_cost:
        plt.plot(layer2s, accuracies)
        plt.ylabel('accuracy')
        plt.xlabel('hidden nodes in second layer')
        plt.title("learning rate = 0.0003, mini-batch = 16, beta = 0.01, layer1 = 10 nodes")
        plt.show()
    
    return results, best, params[best]

def plot_data():   
    """ Plots the distribution of data. """

    [X_train, X_dev, X_test, Y_train, Y_dev, Y_test, numOutputNodes] = load_data('new')
    traindev = np.concatenate((Y_train, Y_dev), 1)
    traindevtest = np.concatenate((traindev, Y_test), 1)
    tdt = traindevtest.reshape(traindevtest.shape[1],)

    Y_train = Y_train.reshape(Y_train.shape[1],) 
    Y_dev = Y_dev.reshape(Y_dev.shape[1],)
    Y_test = Y_test.reshape(Y_test.shape[1],)

    sigma = np.round(np.std(tdt), 3)
    mu = np.round(np.mean(tdt), 3)

    # plots histogram of all data together, indicating values of mean and standard deviation
    plt.figure(1)
    plt.hist(tdt, bins = [0, 1, 2, 3, 4, 5], align = 'left')
    plt.title("{} data points, mu = {}, sigma = {}".format(tdt.size, mu, sigma))
    plt.xlabel("Fe coordination number")
    plt.ylabel("frequency")
    plt.show()

    # plots histogram where the training, cross-validation, and test sets have separate bars
    plt.figure(2)
    plt.hist([Y_train, Y_dev, Y_test], bins = [0, 1, 2, 3, 4, 5], align = 'left', label = ['training', 'cross-validation', 'test'], density = True)
    plt.xlabel("Fe coordination number")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()

    return None


if __name__ == "__main__":

    # allows the user to input which combinations of hyperparameters they would like to test
    p = input('Learning rates? Separate by commas. ').split(',')
    p = [float(i) for i in p]
    q = input('Layer 1s? Separate by commas. ').split(',')
    q = [int(i) for i in q]
    r = input('Layer 2s? Separate by commas. ').split(',')
    r = [int(i) for i in r]
    s = input('Betas? Separate by commas. ').split(',')
    s = [float(i) for i in s]
    iterations = int(input('How many iterations? ')) 


    start = timer() # time how long it takes to train all models

    [X_train, X_dev, X_test, Y1, Y2, Y3, numOutputNodes] = load_data('new')    

    # multi-class classification; convert from array(1, divider) to a one-hot matrix of array(numOutputNodes, divider)
    Y_train = one_hot_matrix(Y1.squeeze(), numOutputNodes)
    Y_dev = one_hot_matrix(Y2.squeeze(), numOutputNodes)
    Y_test = one_hot_matrix(Y3.squeeze(), numOutputNodes)

    # sets of hyperparameters to test, in a grid search
    learning_rates = p 
    layer1s = q  
    layer2s = r
    minibatch_sizes = [16]
    betas = s
    dropouts = [1.0]
    istanh1s = [False]
    istanh2s = [False]
    batchnorms = [True]

    # list comprehension to create sets of hyperparameters for testing, where the number of nodes in layer1 is more than in layer2
    hyperparams = [{'learning_rate': a, 'layer1': b, 'layer2': c, 'minibatch_size': d, 'beta': e, 'dropout': f, 'istanh1': g, 'istanh2': h, 'batchnorm': i} for a in learning_rates for b in layer1s \
    for c in layer2s for d in minibatch_sizes for e in betas for f in dropouts for g in istanh1s for h in istanh2s for i in batchnorms if b > c]

    results, best, params = train_multiple_models(X_train, Y_train, X_dev, Y_dev, numOutputNodes, iterations, hyperparams, print_cost = True)
    
    print(results)
    print("Best Hyperparameters:", str(best))
    
    # print how long the training took in total
    end = timer()
    time = end - start
    print("Time taken:", math.floor(time/3600), "hours,", math.floor((time/3600 - math.floor(time/3600)) * 60), "minutes, and", round((time/60 - math.floor(time/60)) * 60, 2), "seconds")

