import tensorflow as tf
import numpy as np
import math
from extract_training_data import extract_training_data
import matplotlib.pyplot as plt
from ml_helpers_new import one_hot_matrix, create_placeholders, initialize_parameters, forward_propagation, compute_cost, predict, random_mini_batches

#disables AVX warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate=0.0001,
          iterations=5000, minibatch_size = 32, layer1 = 12, layer2 = 8, print_cost=True):
    """ Three-layer NN to predict Fe coordination numbers around oxygen.
        Default is L2 regularization and Adam. Return optimized parameters.

        Arguments:
        ----------------------------
        X_train : array (170, 60% of data)

        Y_train : array (numOutputNodes, 60% of data)

        X_dev : array (170, 20% of data)

        Y_dev : array(numOutputNodes, 20% of data)

        numOutputNodes: int
            Determined by examning the maximum Fe coordination number of each oxygen.
        
        learning_rate, iterations, minibatch_size: as named

        layer1, layer2: number of nodes in the first and second hidden layers

        print_cost: boolean, decides whether or not to print the cost during training

        Returns:
        -----------------------------
        parameters : dict
            weights and biases of each layer.{'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}
    """
    
    # reset all variables to allow the network to be trained multiple times with different hyperparameters
    sess = tf.Session()
    tf.reset_default_graph()

    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    n_x = X_train.shape[0]                            # n_x : input size (the other dimension is the number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE): # auto_reuse allows for the parameters to be changed to reflect new hyperparameters
        # Create Placeholders of shape (n_x, n_y)
        X, Y = create_placeholders(n_x, n_y)
        parameters = initialize_parameters(layer1, layer2, numOutputNodes)  # Initialize parameters

        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = compute_cost(Z3, Y, parameters)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(iterations):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            num_minibatches = len(minibatches) 

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session on one minibatch, and add the cost to the epoch cost
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})    
                epoch_cost += minibatch_cost / num_minibatches
            
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after iteration %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # Plot the cost
        if print_cost:
            iter_num = np.arange(iterations / 5) * 5
            plt.plot(iter_num, np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations')
            # plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        
        # Print the hyperparameters for this particular model
        print('Learning Rate: %s, Mini-Batch Size: %d, %d Nodes in Layer 1, %d Nodes in Layer 2, %d Output Nodes, %d Iterations' \
        % (str(learning_rate).rstrip('0'), minibatch_size, layer1, layer2, numOutputNodes, iterations))

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_acc = accuracy.eval({X: X_train, Y:Y_train})
        dev_acc = accuracy.eval({X: X_dev, Y:Y_dev})
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

    for h in hyperparams:
        learning_rate = h['learning_rate'] # extract the hyperparameters from one item in hyperparams
        layer1 = h['layer1']
        layer2 = h['layer2']
        minibatch_size = h['minibatch_size']

        accs, parameters = model(X_train, Y_train, X_dev, Y_dev, numOutputNodes, learning_rate, iterations, minibatch_size, layer1, layer2, print_cost)
        
        results[frozenset(h.items())] = accs[1] # store the dev test accuracies in a dictionary
        params[frozenset(h.items())] = parameters # do the same for the learned parameters, to be retrieved at the end
    
    best = max(results, key=results.get) # finds what setting of hyperparameters had the highest dev accuracy

    return results, list(best), params[best]

def final_evaluation(X_test, Y_test, parameters):
    """ Evaluates the learned parameters on the test set.

        Arguments:
        ----------------------------
        X_test, Y_test: test set
        parameters: the learned parameters resulting from gradient descent

        Returns:
        ----------------------------
        prediction: the predicted labels
        actual: the actual, correct labels
        test_acc: the percentage of examples currently identified
    """

    prediction = np.array(predict(X_test, parameters))
    actual = np.array(Y_test.argmax(axis=0))

    compare = np.equal(prediction, actual) # compares the two arrays element-wise, returns an array with True when both are equal
    test_acc = np.round(np.sum(compare) / compare.size, 8) # sum the array and divide by its size to find the final test accuracy

    return (prediction, actual, test_acc)

if __name__ == "__main__":

    X_train, X_dev, X_test, Y1, Y2, Y3, numOutputNodes = extract_training_data(cutoff_radius = 2.6)
    # multi classification; convert from array(1, divider) to a one-hot matrix of array(numOutputNodes, divider)
    Y_train = one_hot_matrix(Y1.squeeze(), numOutputNodes)
    Y_dev = one_hot_matrix(Y2.squeeze(), numOutputNodes)
    Y_test = one_hot_matrix(Y3.squeeze(), numOutputNodes)

    learning_rates = [0.0001]
    layer1s = [8]
    layer2s = [8]
    minibatch_sizes = [147]

    hyperparams = [{'learning_rate': a, 'layer1': b, 'layer2': c, 'minibatch_size': d} for a in learning_rates for b in layer1s \
    for c in layer2s for d in minibatch_sizes]

    results, best, params = train_multiple_models(X_train, Y_train, X_dev, Y_dev, numOutputNodes, 9000, hyperparams, print_cost = False)
    prediction, actual, test_acc = final_evaluation(X_test, Y_test, params)

    print("Best Hyperparameters:", str(best))
    print("Predict:", str(prediction))
    print("Actuals:", str(actual))
    print("Test Accuracy:", str(test_acc))