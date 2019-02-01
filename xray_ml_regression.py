import tensorflow as tf
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

# use this import if the above has throws an error on Macs
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from ml_helpers_new import load_data, one_hot_matrix, create_placeholders, initialize_parameters, forward_propagation, compute_reg_cost, random_mini_batches
from timeit import default_timer as timer

#disables AVX warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate = 0.003,
          iterations = 2000, minibatch_size = 16, layer1 = 25, beta = 0, dropout = 1.0, istanh1 = False, batchnorm = True, print_cost = True, is_charge = False):
    """ Two-layer NN to predict the average Fe coordination numbers among 49 oxygen atoms.
        Default is L2 regularization and Adam. Return optimized parameters.

        Arguments:
        ----------------------------
        X_train : array (170, 64% of data)

        Y_train : array (1, 64% of data)

        X_dev : array (170, 16% of data)

        Y_dev : array (1, 16% of data)

        numOutputNodes: int
            Determined by examning the maximum Fe coordination number of each oxygen.
        
        learning_rate, iterations, minibatch_size: as named

        layer1: number of nodes in the first hidden layer

        beta: regularization parameter for L2 regularization in the cost function

        dropout: probability of keeping a node in dropout

        istanh1: determines whether the activation function is tanh (True) or relu (False)

        batchnorm: turns batch normalization on and off

        print_cost: boolean, decides whether or not to print the cost during training

        is_charge: boolean, changes the save path if the data is for charges and not Fe coordination numbers

        Returns:
        -----------------------------
        accs : list of accuracies
        parameters : dict
            weights and biases of each layer.{'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
    """
    
    # reset all variables to allow the network to be trained multiple times with different hyperparameters
    sess = tf.Session()
    tf.reset_default_graph()

    tf.set_random_seed(1)                             # to keep consistent results
    seed = 1                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # n_x : input size (the other dimension is the number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # holds data for graphing
    dev_costs = []
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layer1, 1, regression = True)  # Initialize parameters, with one hidden layer
    training = tf.placeholder_with_default(False, shape=(), name='training') # Create a boolean to use for implementing batch norm and dropout correctly

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, training, istanh1, False, batchnorm, dropout, regression = True)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_reg_cost(Z3, Y, parameters, beta)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # allows for prediction at test time to work with batch normalization; allows for updating of global mean and variance
    with tf.control_dependencies(update_ops):
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Calculate the correct predictions
    correct_prediction = tf.less_equal(tf.abs(tf.divide(tf.subtract(Z3, Y), Y)), tf.fill([1,1], 0.05)) 
    # define one measure of accuracy by counting a prediction as correct if it's within 5% of the true value
    
    # Calculate the mean absolute percentage error of the predictions
    MAPE = tf.scalar_mul(100, tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Z3, Y), Y))))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # Print the hyperparameters for this particular model
        print('Learning Rate: %s, Mini-Batch Size: %d, Beta: %s, %d Nodes in Hidden Layer, %d Output Nodes, %d Iterations, %s Dropout Prob, Hidden Layer Tanh: %s, Batch Norm: %s' \
        % (str(learning_rate).rstrip('0'), minibatch_size, str(beta).rstrip('0'), layer1, numOutputNodes, iterations, str(dropout).rstrip('0'), istanh1, batchnorm))
        
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
            
            # save the training and cross-validation cost every 5 epochs
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                dev_cost = sess.run(cost, feed_dict = {X: X_dev, Y: Y_dev, training: False})
                dev_costs.append(dev_cost)

            # print the cost after every 200 epochs
            if print_cost == True and epoch % 200 == 0: # used during testing to ensure gradient descent is working properly
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, training: False})
                train_mape = MAPE.eval({X: X_train, Y: Y_train, training: False})
                print("Training cost after iteration %i: %f, accuracy: %f, MAPE: %f %%" % (epoch, epoch_cost, train_accuracy, train_mape))
                
                dev_cost = sess.run(cost, feed_dict={X:X_dev, Y: Y_dev, training: False})
                dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev, training: False})
                dev_mape = MAPE.eval({X: X_dev, Y: Y_dev, training: False})
                print("Dev cost after iteration %i: %f, accuracy: %f, MAPE: %f %%" % (epoch, dev_cost, dev_accuracy, dev_mape))

        # # Plot the cost
        # if print_cost:
        #     iter_num = np.arange(iterations / 5) * 5
        #     plt.plot(iter_num, np.squeeze(costs), label = 'training')
        #     plt.plot(iter_num, np.squeeze(dev_costs), label = 'cross-validation')
        #     plt.ylabel('cost')
        #     plt.xlabel('iterations')
        #     # plt.ylim(top = 0.01, bottom = 0.002) # y range used to plot for averaged spectra
        #     plt.ylim(top = 0.0075, bottom = 0.001) # y range used to plot for training on charges
        #     plt.title('Cost vs. Iterations')
        #     plt.legend()
        #     plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        if is_charge:
            saver.save(sess, "./charge_reg_models/charge_regression_model_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, beta))
        else:
            saver.save(sess, "./reg_models/regression_model_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, beta))

        train_acc = accuracy.eval({X: X_train, Y: Y_train, training: False})
        dev_acc = accuracy.eval({X: X_dev, Y: Y_dev, training: False})
        
        mape_train = MAPE.eval({X: X_train, Y: Y_train, training: False})
        mape_dev = MAPE.eval({X: X_dev, Y: Y_dev, training: False})

        accs = [train_acc, dev_acc, mape_train, mape_dev]

        print("Train Accuracy:", train_acc, "; MAPE:", mape_train)
        print("Dev Accuracy:", dev_acc, "; MAPE:", mape_dev)

    return accs, parameters

def train_multiple_models(X_train, Y_train, X_dev, Y_dev, numOutputNodes, iterations, hyperparams, print_cost = True, is_charge = False):
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
            minibatch_size = h['minibatch_size']
            beta = h['beta']
            dropout = h['dropout']
            istanh1 = h['istanh1']
            batchnorm = h['batchnorm']

            # train the model with the given hyperparameters
            accs, parameters = model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate, iterations, minibatch_size, layer1, beta, dropout, istanh1, batchnorm, print_cost, is_charge)
            
            results[frozenset(h.items())] = accs[3] # store the dev test MAPEs in a dictionary
            params[frozenset(h.items())] = parameters # do the same for the learned parameters, to be retrieved at the end
    
    except KeyboardInterrupt: # allow for exiting the for loop in case we want to stop testing all the hyperparameters; to use, press Ctrl+C in terminal
        pass
        
    best = min(results, key=results.get) # finds what setting of hyperparameters had the lowest MAPE

    return results, list(best), params[best]


def train_models_change_nodes(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate, iterations, minibatch_size, beta, nodes, print_cost = True, is_charge = False):
    """ Same as train_multiple_models(), except all hyperparameters are fixed except for the number of nodes in the hidden layer.
        
        For averaged spectra:
        Learning rate: 0.003
        Mini-batch size: 16
        Beta: 0
        Dropout: 1.0 (i.e. no dropout)
        Activation function: ReLU
        Batchnorm: True

        For charges:
        Learning rate: 0.0003
        Mini-batch size: 32
        Beta: 0.01
        Dropout: 1.0
        Activation function: ReLU
        Batchnorm: True
    """
    
    results = {}
    params = {}

    try:
        for n in nodes:
            accs, parameters = model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate, iterations, minibatch_size, n, beta, 1.0, False, True, print_cost, is_charge)
            
            results[n] = accs[3]
            params[n] = parameters
            
    except KeyboardInterrupt: # allow for exiting the for loop in case we want to stop testing all the hyperparameters; to use, press Ctrl+C in terminal
        pass
        
    best = min(results, key=results.get)
    mapes = [results[l] for l in nodes]

    if print_cost:
        plt.plot(nodes, mapes)
        plt.ylabel('MAPE')
        plt.xlabel('# of nodes in hidden layer')
        plt.title("learning rate = {}, mini-batch = {}, beta = {}".format(learning_rate, minibatch_size, beta))
        plt.show()
    
    return results, best, params[best]

def plot_data():
    """ Plots the distribution of data. """
    
    [X_train, X_dev, X_test, Y_train, Y_dev, Y_test, numOutputNodes] = load_data('regression') 
    
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
    plt.hist(tdt)
    plt.title("{} data points, mu = {}, sigma = {}".format(tdt.size, mu, sigma))
    plt.xlabel("average Fe coordination number")
    plt.ylabel("frequency")
    plt.show()

    # plots histogram where the training, cross-validation, and test sets have separate bars
    plt.figure(2)
    plt.hist([Y_train, Y_dev, Y_test], label = ['training', 'cross-validation', 'test'], density = True)
    plt.xlabel("average Fe coordination number")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()

    # below is graphing for the charge data, as opposed to the averaged spectrum data
    [X_train1, X_dev1, X_test1, _, _, _, Y_train1, Y_dev1, Y_test1, numOutputNodes1] = load_data('multi_task')
    traindev1 = np.concatenate((Y_train1, Y_dev1), 1)
    traindevtest1 = np.concatenate((traindev1, Y_test1), 1)
    tdt1 = traindevtest1.reshape(traindevtest1.shape[1],)

    Y_train1 = Y_train1.reshape(Y_train1.shape[1],)
    Y_dev1 = Y_dev1.reshape(Y_dev1.shape[1],)
    Y_test1 = Y_test1.reshape(Y_test1.shape[1],)

    sigma = np.round(np.std(tdt1), 3)
    mu = np.round(np.mean(tdt1), 3)

    # plots histogram of all data together, indicating values of mean and standard deviation
    plt.figure(3)
    plt.hist(tdt1)
    plt.title("{} data points, mu = {}, sigma = {}".format(tdt1.size, mu, sigma))
    plt.xlabel("charge")
    plt.ylabel("frequency")
    plt.show()

    # plots histogram where the training, cross-validation, and test sets have separate bars
    plt.figure(4)
    plt.hist([Y_train1, Y_dev1, Y_test1], label = ['training', 'cross-validation', 'test'], density = True)
    plt.xlabel("charge")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()

    return None


if __name__ == "__main__":

    o = int(input('How many iterations? ')) 

    # allows for user to input settings of hyperparameters
    p = input('Learning rates? Separate by commas. ').split(',')
    p = [float(i) for i in p]

    q = input('Layer 1s? Separate by commas. ').split(',')
    q = [int(i) for i in q]

    r = input('Minibatch sizes? Separate by commas. ').split(',')
    r = [int(i) for i in r]

    s = input('Betas? Separate by commas. ').split(',') # standard setting is 0.0
    s = [float(i) for i in s]


    start = timer()

    # # choose the appropriate line depending on if we are using the average coordination number or charge data
    # [X_train, X_dev, X_test, Y_train, Y_dev, Y_test, numOutputNodes] = load_data('regression') 
    [X_train, X_dev, X_test, _, _, _, Y_train, Y_dev, Y_test, numOutputNodes] = load_data('multi_task')


    # nodes = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    # results, best, params = train_models_change_nodes(X_train, Y_train, X_dev, Y_dev, numOutputNodes, 0.0003, 4000, 32, 0.01, nodes, print_cost = True, is_charge = True)


    learning_rates = p
    layer1s = q
    minibatch_sizes = r
    betas = s
    dropouts = [1.0] # disregard; seems to have no effect
    istanh1s = [False] # false: means that ReLU is used
    batchnorms = [True]

    # list comprehension to create sets of hyperparameters for testing
    hyperparams = [{'learning_rate': a, 'layer1': b, 'minibatch_size': d, 'beta': e, 'dropout': f, 'istanh1': g, 'batchnorm': i} for a in learning_rates for b in layer1s \
    for d in minibatch_sizes for e in betas for f in dropouts for g in istanh1s for i in batchnorms]

    results, best, params = train_multiple_models(X_train, Y_train, X_dev, Y_dev, numOutputNodes, o, hyperparams, print_cost = True, is_charge = True)
    
    print(results)
    print("Best Hyperparameters:", str(best))


    # print how long the training took in total
    end = timer()
    time = end-start
    print("Time taken:", math.floor(time/3600), "hours,", math.floor((time/3600 - math.floor(time/3600)) * 60), "minutes, and", round((time/60 - math.floor(time/60)) * 60, 2), "seconds")