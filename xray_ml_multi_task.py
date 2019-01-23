import tensorflow as tf
import numpy as np
import math
# from extract_training_data import extract_training_data
import matplotlib.pyplot as plt
from ml_helpers_new import load_data, one_hot_matrix, create_placeholders_multi, initialize_parameters_multi, forward_propagation_multi, compute_cost, random_mini_batches
from timeit import default_timer as timer

#disables AVX warning from being displayed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def model(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, learning_rate = 0.0001, iterations = 5000, minibatch_size = 16, print_cost = True):
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
    seed = 1                                          # to keep consistent results
    n_x = X_train.shape[0]                            # n_x : input size (the other dimension is the number of examples in the train set)
    n_y1 = Y1_train.shape[0]                            # n_y : output size
    n_y2 = Y2_train.shape[0]
    costs = []                                        # holds data for graphing

    # Create Placeholders of shape (n_x, n_y)
    X, Y1, Y2 = create_placeholders_multi(n_x, n_y1, n_y2)
    parameters = initialize_parameters_multi(100, 50, 50, numOutputNodes, 1)  # Initialize parameters
    training = tf.placeholder_with_default(False, shape=(), name='training') # Create a boolean to use for implementing batch norm and dropout correctly

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3_1, Z3_2 = forward_propagation_multi(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3_1, labels=Y1))
    cost2 = tf.reduce_mean(tf.squared_difference(Z3_2, Y2))
    joint_cost = cost1 + cost2

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this section used to implement batch normalization
    with tf.control_dependencies(update_ops):
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        # optimizer_1 = tf.train.AdamOptimizer(learning_rate).minimize(cost1)
        # optimizer_2 = tf.train.AdamOptimizer(learning_rate).minimize(cost2)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(joint_cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Calculate the correct predictions
    correct_prediction_1 = tf.equal(tf.argmax(Z3_1), tf.argmax(Y1))
    correct_prediction_2 = tf.less_equal(tf.abs(tf.divide(tf.subtract(Z3_2, Y2), Y2)), tf.fill([1,1], 0.05))
        
    # Calculate accuracy on the test set
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction_1, "float"))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))

    MAPE = tf.scalar_mul(100, tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Z3_2, Y2), Y2))))

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # Print the hyperparameters for this particular model
        # print('Learning Rate: %s, Mini-Batch Size: %d, Beta: %s, %d Nodes in Layer 1, %d Nodes in Layer 2, %d Output Nodes, %d Iterations, %s Dropout Prob, First Layer Tanh: %s, Second Layer Tanh: %s, Batch Norm: %s' \
        # % (str(learning_rate).rstrip('0'), minibatch_size, str(beta).rstrip('0'), layer1, layer2, numOutputNodes, iterations, str(dropout).rstrip('0'), istanh1, istanh2, batchnorm))
        
        for epoch in range(iterations):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            seed = seed + 1
            minibatches1 = random_mini_batches(X_train, Y1_train, minibatch_size, seed)
            minibatches2 = random_mini_batches(X_train, Y2_train, minibatch_size, seed)
            num_minibatches = len(minibatches1) 

            for i in range(num_minibatches):
                # Select a minibatch
                (minibatch_X, minibatch_Y1) = minibatches1[i]
                _, minibatch_Y2 = minibatches2[i]
                
                # Run the session on one minibatch, and add the cost to the epoch cost
                _ , minibatch_cost = sess.run([optimizer, joint_cost], feed_dict={X: minibatch_X, Y1: minibatch_Y1, Y2: minibatch_Y2, training: True})
                epoch_cost += minibatch_cost / num_minibatches
            
            # Print the cost every epoch
            # if print_cost == True and epoch % 100 == 0:
            #     print("Cost after iteration %i: %f" % (epoch, epoch_cost))
            # if print_cost == True and epoch % 5 == 0:
            #     costs.append(epoch_cost)
            if print_cost == True and epoch % 250 == 0: # used during testing to ensure gradient descent is working properly
                train_accuracy1 = accuracy1.eval({X: X_train, Y1: Y1_train, training: False})
                train_accuracy2 = accuracy2.eval({X: X_train, Y2: Y2_train, training: False})
                train_mape = MAPE.eval({X: X_train, Y1: Y1_train, Y2: Y2_train, training: False})
                print("Training cost after iteration %i: %f, accuracy: %f, %f, MAPE: %f" % (epoch, epoch_cost, train_accuracy1, train_accuracy2, train_mape))
                
                dev_cost = sess.run(joint_cost, feed_dict={X: X_dev, Y1: Y1_dev, Y2: Y2_dev, training: False})
                dev_accuracy1 = accuracy1.eval({X: X_dev, Y1: Y1_dev, training: False})
                dev_accuracy2 = accuracy2.eval({X: X_dev, Y2: Y2_dev, training: False})
                dev_mape = MAPE.eval({X: X_dev, Y1: Y1_dev, Y2: Y2_dev, training: False})
                print("Dev cost after iteration %i: %f, accuracy: %f, %f, MAPE: %f" % (epoch, dev_cost, dev_accuracy1, dev_accuracy2, dev_mape))

        # # Plot the cost
        # if print_cost:
        #     iter_num = np.arange(iterations / 5) * 5
        #     plt.plot(iter_num, np.squeeze(costs))
        #     plt.ylabel('cost')
        #     plt.xlabel('iterations')
        #     # plt.title("Learning rate =" + str(learning_rate))
        #     plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
    
        train_acc1 = accuracy1.eval({X: X_train, Y1: Y1_train, training: False})
        train_acc2 = accuracy2.eval({X: X_train, Y2: Y2_train, training: False})
        dev_acc1 = accuracy1.eval({X: X_dev, Y1: Y1_dev, training: False})
        dev_acc2 = accuracy2.eval({X: X_dev, Y2: Y2_dev, training: False})
        accs = [train_acc1, train_acc2, dev_acc1, dev_acc2]

        # test_acc = accuracy.eval({X: X_test, Y: Y_test, training: False}) # use this later when doing the final test on the selected set of hyperparameters
        # print("Test Accuracy:", test_acc)

        # print("Train Accuracy:", train_acc)
        # print("Dev Accuracy:", dev_acc)

    return accs, parameters

def train_multiple_models(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, iterations, hyperparams, print_cost = True):
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
            minibatch_size = h['minibatch_size']

            # train the model with the given hyperparameters
            accs, parameters = model(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, learning_rate, iterations, minibatch_size, print_cost)
            
            results[frozenset(h.items())] = accs[2] + accs[3] # store the sum of the two dev set accuracies in a dictionary
            params[frozenset(h.items())] = parameters # do the same for the learned parameters, to be retrieved at the end
    
    except KeyboardInterrupt: # allow for exiting the for loop in case we want to stop testing all the hyperparameters; to use, press Ctrl+C in terminal
        pass

    best = max(results, key=results.get) # finds what setting of hyperparameters had the highest dev accuracy

    return results, list(best), params[best]


# def final_evaluation(X_test, Y_test, parameters): # currently not functional
#     """ Evaluates the learned parameters on the test set.

#         Arguments:
#         ----------------------------
#         X_test, Y_test: test set
#         parameters: the learned parameters resulting from gradient descent

#         Returns:
#         ----------------------------
#         prediction: the predicted labels
#         actual: the actual, correct labels
#         test_acc: the percentage of examples currently identified
#     """

#     prediction = np.array(predict(X_test, parameters))
#     actual = np.array(Y_test.argmax(axis=0))

#     compare = np.equal(prediction, actual) # compares the two arrays element-wise, returns an array with True when both are equal
#     test_acc = np.round(np.sum(compare) / compare.size, 8) # sum the array and divide by its size to find the final test accuracy

#     return (prediction, actual, test_acc)

if __name__ == "__main__":

    start = timer()
    [X_train, X_dev, X_test, Y1a, Y1b, Y1c, Y2_train, Y2_dev, Y2_test, numOutputNodes] = load_data('multi_task')
    
    # multi-class classification; convert from array(1, divider) to a one-hot matrix of array(numOutputNodes, divider)
    Y1_train = one_hot_matrix(Y1a.squeeze(), numOutputNodes)
    Y1_dev = one_hot_matrix(Y1b.squeeze(), numOutputNodes)
    Y1_test = one_hot_matrix(Y1c.squeeze(), numOutputNodes)

    # sets of hyperparameters to test, in a grid search
    learning_rates = [0.001, 0.00001]
    minibatch_sizes = [16]

    # list comprehension to create sets of hyperparameters for testing
    hyperparams = [{'learning_rate': a, 'minibatch_size': d} for a in learning_rates for d in minibatch_sizes]

    results, best, params = train_multiple_models(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, 2001, hyperparams, print_cost = True)
    #prediction, actual, test_acc = final_evaluation(X_test, Y_test, params)

    print(results)
    print("Best Hyperparameters:", str(best))
    #print("Predict:", str(prediction))
    #print("Actuals:", str(actual))
    #print("Test Accuracy:", str(test_acc))

    # print how long the training took in total
    end = timer()
    time = end - start
    print("Time taken:", math.floor(time/3600), "hours,", math.floor((time/3600 - math.floor(time/3600)) * 60), "minutes, and", round((time/60 - math.floor(time/60)) * 60, 2), "seconds")