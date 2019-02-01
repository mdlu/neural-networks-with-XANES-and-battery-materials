import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

# use this import if the above has throws an error on Macs
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from ml_helpers_new import compute_costs, load_data, one_hot_matrix, create_placeholders_multi, initialize_parameters_multi, forward_propagation_multi, compute_cost, random_mini_batches
from timeit import default_timer as timer

#disables AVX warning from being displayed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def model(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, learning_rate = 0.0001, iterations = 5000, minibatch_size = 16, print_cost = True, \
            layer1 = 100, layer2_1 = 50, layer2_2 = 50, beta1 = 0.01, beta2 = 0.1):
    """ Three-layer NN to predict Fe coordination numbers around oxygen.
        Default is L2 regularization and Adam. Return optimized parameters.

        Arguments:
        ----------------------------
        X_train : array of spectra inputs (170, 64% of data)
        Y1_train : array of coordination number labels (numOutputNodes, 64% of data)
        Y2_train: array of charge labels (1, 64% of data)

        X_dev : array (170, 16% of data)
        Y1_dev : array (numOutputNodes, 16% of data)
        Y2_dev : array (1, 16% of data)

        numOutputNodes: int
            Determined by examning the maximum Fe coordination number of each oxygen.
        
        learning_rate, iterations, minibatch_size: as named

        print_cost: boolean, decides whether or not to print the cost during training

        layer1: number of nodes in first hidden layer
        layer2_1: number of nodes in task-specific hidden layer for classification
        layer2_2: number of nodes in task-specific hidden layer for regression

        beta1: regularization value used for classification
        beta2: regularization value used for regression

        Returns:
        -----------------------------
        accs: training accuracies
        mapes: mean absolute percentage errors
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
    dev_costs = []

    # Create Placeholders of shape (n_x, n_y)
    X, Y1, Y2 = create_placeholders_multi(n_x, n_y1, n_y2)
    parameters = initialize_parameters_multi(layer1, layer2_1, layer2_2, numOutputNodes, 1)  # Initialize parameters
    training = tf.placeholder_with_default(False, shape=(), name='training') # Create a boolean to use for implementing batch norm and dropout correctly

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3_1, Z3_2 = forward_propagation_multi(X, parameters, training)

    # Cost function: Add cost function to tensorflow graph
    cost1, cost2 = compute_costs(Z3_1, Z3_2, Y1, Y2, parameters, beta1, beta2)

    joint_cost = cost1 + cost2

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this section used to implement batch normalization
    with tf.control_dependencies(update_ops):
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer_1 = tf.train.AdamOptimizer(learning_rate).minimize(cost1)
        optimizer_2 = tf.train.AdamOptimizer(learning_rate).minimize(cost2)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(joint_cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Calculate the correct predictions
    correct_prediction_1 = tf.equal(tf.argmax(Z3_1), tf.argmax(Y1))
    # below: use 'within 5% of the correct value' as a measure of accuracy for regression
    correct_prediction_2 = tf.less_equal(tf.abs(tf.divide(tf.subtract(Z3_2, Y2), Y2)), tf.fill([1,1], 0.05)) 
        
    # Calculate accuracy on the test set
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction_1, "float"))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))

    # mean absolute percentage error
    MAPE = tf.scalar_mul(100, tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Z3_2, Y2), Y2))))

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        # Print the hyperparameters for this particular model
        print('Learning Rate: %s, Mini-Batch Size: %d, Betas: %s and %s, Nodes: %d - %d/%d - %d/%d, %d Iterations' \
        % (str(learning_rate).rstrip('0'), minibatch_size, str(beta1), str(beta2), layer1, layer2_1, layer2_2, numOutputNodes, 1, iterations))
        
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
                if np.random.rand() < 0.5:
                    _ , minibatch_cost = sess.run([optimizer_1, joint_cost], feed_dict={X: minibatch_X, Y1: minibatch_Y1, Y2: minibatch_Y2, training: True})
                    epoch_cost += minibatch_cost / num_minibatches
                else:
                    _ , minibatch_cost = sess.run([optimizer_2, joint_cost], feed_dict={X: minibatch_X, Y1: minibatch_Y1, Y2: minibatch_Y2, training: True})
                    epoch_cost += minibatch_cost / num_minibatches

            # if print_cost == True and epoch % 5 == 0:
            #     costs.append(epoch_cost)
            #     dev_cost = sess.run(cost, feed_dict = {X: X_dev, Y: Y_dev, training: False})
            #     dev_costs.append(dev_cost)
            
            # print the cost every 250 epochs
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

            # # Code to print out how the model is doing so far at epoch 1500
            # if print_cost == True and epoch == 1500:
            #     z31s = Z3_1.eval({X: X_dev, Y1: Y1_dev, Y2: Y2_dev, training: False})
            #     z32s = Z3_2.eval({X: X_dev, Y1: Y1_dev, Y2: Y2_dev, training: False})
            #     print(np.argmax(z31s, axis=0))
            #     print(np.argmax(Y1_dev, axis=0))
            #     print(z32s[:, :25])
            #     print(Y2_dev[:, :25])
            #     z32s_new = z32s.reshape(z32s.size,)
            #     Y2_dev_new = Y2_dev.reshape(Y2_dev.size,)
            #     xs = np.arange(-2.0, -0.6, 0.01)

            #     plt.figure(figsize = (7,7))
            #     plt.scatter(z32s_new, Y2_dev_new)
            #     # plt.scatter(a, b)
            #     plt.plot(xs, xs, 'r-')
            #     # plt.ylim(bottom = -2.0, top = -0.6)
            #     # plt.xlim(left = -2.0, right = -0.6)
            #     plt.xlabel('predicted')
            #     plt.ylabel('actual')
            #     plt.show()


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
        saver.save(sess, "./multi_models/multi_model_{}_{}_{}_{}_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, minibatch_size, layer1, layer2_1, layer2_2, beta1, beta2))
    
        train_acc1 = accuracy1.eval({X: X_train, Y1: Y1_train, training: False})
        train_acc2 = accuracy2.eval({X: X_train, Y2: Y2_train, training: False})
        dev_acc1 = accuracy1.eval({X: X_dev, Y1: Y1_dev, training: False})
        dev_acc2 = accuracy2.eval({X: X_dev, Y2: Y2_dev, training: False})
        accs = [train_acc1, train_acc2, dev_acc1, dev_acc2]


        train_mape = MAPE.eval({X: X_train, Y1: Y1_train, Y2: Y2_train, training: False})
        dev_mape = MAPE.eval({X: X_dev, Y1: Y1_dev, Y2: Y2_dev, training: False})

        mapes = [train_mape, dev_mape]
        print("Training accuracies and MAPE: {}, {}, {}%; Dev accuracies and MAPE: {}, {}, {}, Total accuracy: {}%".format(train_acc1, train_acc2, train_mape, dev_acc1, dev_acc2, dev_mape, (100*dev_acc1 - dev_mape)))

    return accs, mapes, parameters

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
            layer1 = h['layer1']
            layer2_1 = h['layer2_1']
            layer2_2 = h['layer2_2']
            beta1 = h['beta1']
            beta2 = h['beta2']

            # train the model with the given hyperparameters
            accs, mapes, parameters = model(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, learning_rate, iterations, minibatch_size, print_cost, layer1, layer2_1, layer2_2, beta1, beta2)
            
            results[frozenset(h.items())] = 100*accs[2] - mapes[1] # define a measure of accuracy as the accuracy of the classification, minus the MAPE of the regression
            params[frozenset(h.items())] = parameters # do the same for the learned parameters, to be retrieved at the end
    
    except KeyboardInterrupt: # allow for exiting the for loop in case we want to stop testing all the hyperparameters; to use, press Ctrl+C in terminal
        pass

    best = max(results, key=results.get) # finds what setting of hyperparameters is best, with our defined measure of accuracy

    return results, list(best), params[best]


if __name__ == "__main__":

    start = timer()

    [X_train, X_dev, X_test, Y1a, Y1b, Y1c, Y2_train, Y2_dev, Y2_test, numOutputNodes] = load_data('multi_task')

    # multi-class classification; convert from array(1, divider) to a one-hot matrix of array(numOutputNodes, divider)
    Y1_train = one_hot_matrix(Y1a.squeeze(), numOutputNodes)
    Y1_dev = one_hot_matrix(Y1b.squeeze(), numOutputNodes)
    Y1_test = one_hot_matrix(Y1c.squeeze(), numOutputNodes)

    # sets of hyperparameters to test, in a grid search
    learning_rates = [0.000001]
    minibatch_sizes = [16]
    layer1s = [100]
    layer2_1s = [50]
    layer2_2s = [50]
    beta1s = [1, 3, 10, 30, 100]
    beta2s = [0.003, 0.01, 0.03, 0.1, 0.3]

    # list comprehension to create sets of hyperparameters for testing
    hyperparams = [{'learning_rate': a, 'minibatch_size': b, 'layer1': c, 'layer2_1': d, 'layer2_2': e, 'beta1': f, 'beta2': g} \
    for a in learning_rates for b in minibatch_sizes for c in layer1s for d in layer2_1s for e in layer2_2s for f in beta1s for g in beta2s]

    results, best, params = train_multiple_models(X_train, Y1_train, Y2_train, X_dev, Y1_dev, Y2_dev, numOutputNodes, 2000, hyperparams, print_cost = True)
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