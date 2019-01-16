import tensorflow as tf
import numpy as np
import math

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.
    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


def create_placeholders(n_x, n_y):
    # Use None because it allow flexible number on the number of examples in the placeholders.
    # In fact, the number of examples during test/train is different
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    return X, Y


def initialize_parameters(L1_units=14, L2_units=9, L3_units=5, regression = False):
    """
    Initializes parameters to build a neural network with tensorflow.

    Arguments:
    L1_units, L2_units, L3_units: the number of nodes in each layer of the network (if only one hidden layer, L3_units is not used)
    regression: boolean, determines whether two hidden layers (for the multi-class classification network) or one hidden layer (for the regression network) are used

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, (and if there are two hidden layers, W3 and b3 as well)
    """

    # tf.set_random_seed(1) # if consistent results are desired, also use seed=1 in the parameters of the initializers
    W1 = tf.get_variable("W1", [L1_units, 170], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [L1_units, 1], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [L2_units, L1_units], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [L2_units, 1], initializer=tf.contrib.layers.xavier_initializer())
    
    if not regression: # currently, in the regression case, we are only using one hidden layer; this is for the case when we have two
        W3 = tf.get_variable("W3", [L3_units, L2_units], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [L3_units, 1], initializer=tf.contrib.layers.xavier_initializer())
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    else:
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters

def forward_propagation(X, parameters, training, istanh1, istanh2, batchnorm, dropout = 0.5, regression = False):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    istanh1, istanh2, batchnorm, dropout: hyperparameters that can be adjusted
    regression: boolean; if True, only one hidden layer is used; if False, then two are used

    Returns:
    eithe Z2 or Z3, the output of the last unit, depending on the number of hidden layers
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    if not regression:
        W3 = parameters['W3']
        b3 = parameters['b3']

    sess = tf.Session()
    training = sess.run(training)

    # if training:
    #     X = tf.nn.dropout(X, dropout)
    Z1 = tf.add(tf.matmul(W1, X), b1)
    if batchnorm:
        Z1 = tf.layers.batch_normalization(Z1, training = training, momentum = 0.99, axis = 0)
    if istanh1:
        A1 = tf.nn.tanh(Z1)
    else:
        A1 = tf.nn.relu(Z1)
    # if training:
    #     A1 = tf.nn.dropout(A1, dropout)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    if regression:
        return Z2 # return, since in the regression case, we only have one hidden layer

    else:
        if batchnorm:
            Z2 = tf.layers.batch_normalization(Z2, training = training, momentum = 0.99, axis = 0)
        if istanh2:
            A2 = tf.nn.tanh(Z2)
        else:
            A2 = tf.nn.relu(Z2)
        # if training:
        #     A2 = tf.nn.dropout(A2, dropout)
        Z3 = tf.add(tf.matmul(W3, A2), b3)

        return Z3

def compute_cost(Z3, Y, parameters, beta = 0):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (5, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    parameters -- dictionary of the matrices of weights and biases
    beta -- the regularization parameter

    Returns:
    cost - Tensor of the cost function
    """

    # transposition to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits_v2()
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    # normal loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    # loss function with L2 regularization with beta
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    cost = tf.reduce_mean(cost + beta * regularizers)

    return cost

def compute_reg_cost(Z3, Y, parameters, beta = 0):
    """
    Computes the cost using the mean squared error. Regularization is optional.
    Arguments and returns are the same as for compute_cost().
    """

    cost = tf.reduce_mean(tf.squared_difference(Z3, Y)) # use the MSE error for the regression task

    W1 = parameters['W1']
    W2 = parameters['W2']
    # loss function with L2 regularization with beta
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    
    cost = tf.reduce_mean(cost + beta * regularizers)
    return cost


def predict(X, parameters, training, istanh1, istanh2, batchnorm): # currently not functional
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder(tf.float32, shape=(170, None), name='x')
    
    z3 = forward_propagation(x, params, training, istanh1, istanh2, batchnorm, regression = False)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


def random_mini_batches(X, Y, mini_batch_size = 16, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- for consistent results
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    np.random.seed(seed)          # if consistent results are desired
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    # simpler, list-comprehension implementation of code from Deep Learning course
    mini_batches = [(shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size], shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]) for k in range(math.ceil(m/mini_batch_size))]

    return mini_batches