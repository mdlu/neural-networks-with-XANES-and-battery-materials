import tensorflow as tf
import numpy as np
from extract_training_data import extract_training_data
import matplotlib.pyplot as plt
from ML_helpers import one_hot_matrix, create_placeholders, initialize_parameters, forward_propagation, compute_cost, predict


X_train, X_test, Y1, Y2, numOutputNodes = extract_training_data()
# multi classification. Convert from array(1, divider) to array(numOutputNodes, divider)
Y_train = one_hot_matrix(Y1.squeeze(), numOutputNodes)
Y_test = one_hot_matrix(Y2.squeeze(), numOutputNodes)


def model(X_train, Y_train, X_test, Y_test, numOutputNodes, learning_rate=0.0001,
          iterations=5000, print_cost=True):
    """ Three-layer NN to predict Fe coordination numbers around oxygen.
        Default is L2 regularization and Adam. Return optimized parameters.

        Arguments:
        ----------------------------
        X_train : array (170, divider)

        Y_train : array (numOutputNodes, divider)

        X_test : array (170, #totalExamples - divider)

        Y_test : array(numOutputNodes, #totalExamples - divider)

        numOutputNodes: int
            Determined by examning the maximum Fe coordination number of each oxygen.

        Returns:
        -----------------------------
        parameters : dict
            weights and biases of each layer.{'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}
    """

    tf.set_random_seed(1)                             # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(12, 8, numOutputNodes)  # Initialize parameters
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
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after iteration %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        iter_num = np.arange(iterations / 5) * 5
        plt.plot(iter_num, np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("number of nodes in output layer is: " + str(numOutputNodes))

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test, numOutputNodes, 0.0001, 9000, print_cost=True)

prediction = predict(X_test, parameters)
print(prediction)
print(Y_test.argmax(axis=0))
