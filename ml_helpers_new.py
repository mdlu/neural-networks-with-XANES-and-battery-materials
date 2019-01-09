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


def initialize_parameters(L1_units=14, L2_units=9, L3_units=5):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [L1_units, 170], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [L1_units, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("W2", [L2_units, L1_units], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [L2_units, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable("W3", [L3_units, L2_units], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [L3_units, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# def batch_norm_wrapper(inputs, is_training, decay = 0.99):
#     epsilon = 1e-4
    
#     batch_mean, batch_var = tf.nn.moments(inputs,[0])
#     scale = tf.Variable(tf.ones_like(batch_mean))
#     beta = tf.Variable(tf.zeros_like(batch_mean))
#     pop_mean = tf.Variable(tf.ones_like(batch_mean))
#     pop_var = tf.Variable(tf.zeros_like(batch_mean))
    
#     # scale = tf.Variable(tf.ones([inputs.get_shape().as_list()[-1], 1]))
#     # beta = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1], 1]))
#     # pop_mean = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1], 1]), trainable=False)
#     # pop_var = tf.Variable(tf.ones([inputs.get_shape().as_list()[-1], 1]), trainable=False)

#     if is_training:

#         train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
#         train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
#         with tf.control_dependencies([train_mean, train_var]):
#             return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
#     else:
#         return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

def forward_propagation(X, parameters, isTraining = True):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    epsilon = 1e-4
    
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    if isTraining: # do not use dropout at test time
        W1 = tf.nn.dropout(W1, 0.9)
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    m1, v1 = tf.nn.moments(A1, [0])
    A1 = (A1 - m1) / tf.sqrt(v1 + epsilon)
    if isTraining:
        A1 = tf.nn.dropout(A1, 0.9)

    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.tanh(Z2)
    m2, v2 = tf.nn.moments(A2, [0])
    A2 = (A2 - m2) / tf.sqrt(v2 + epsilon)
    if isTraining:
        A2 = tf.nn.dropout(A2, 0.9)

    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3

    # if isTraining:
    #     Z1 = tf.add(tf.matmul(W1, X), b1)
    #     m1, v1 = tf.nn.moments(Z1, [0])
    #     if gm1:
    #         gm1 = 0.99 * gm1 + 0.01 * m1
    #         gv1 = 0.99 * gv1 + 0.01 * v1
    #     else:
    #         gm1 = 0.01 * m1
    #         gv1 = 0.01 * v1
    #     Z1n = (Z1 - m1) / tf.sqrt(v1 + epsilon)
    #     A1 = tf.nn.relu(Z1n)

    #     Z2 = tf.add(tf.matmul(W2, A1), b2)
    #     m2, v2 = tf.nn.moments(Z2, [0])
    #     if gm2:
    #         gm2 = 0.99 * gm2 + 0.01 * m2
    #         gv2 = 0.99 * gv2 + 0.01 * v2
    #     else:
    #         gm2 = 0.01 * m2
    #         gv2 = 0.01 * v2
    #     Z2n = (Z2 - m2) / tf.sqrt(v2 + epsilon)
    #     A2 = tf.nn.tanh(Z2n)

    #     Z3 = tf.add(tf.matmul(W3, A2), b3)
    #     return Z3, gm1, gv1, gm2, gv2
    # else:
    #     Z1 = tf.add(tf.matmul(W1, X), b1)
    #     Z1n = (Z1 - gm1) / tf.sqrt(gv1 + epsilon)
    #     A1 = tf.nn.relu(Z1n)

    #     Z2 = tf.add(tf.matmul(W2, A1), b2)
    #     Z2n = (Z2 - gm2) / tf.sqrt(gv2 + epsilon)
    #     A2 = tf.nn.tanh(Z2n)

    #     Z3 = tf.add(tf.matmul(W3, A2), b3)
    #     return Z3

    # if isTraining:
    #     # Forward propagation, including batch-normalization
    #     Z1 = tf.add(tf.matmul(W1, X), b1)
    #     m1, v1 = tf.nn.moments(Z1, [0]) # calculates the mean and variance
    #     am1 = tf.assign(gm1, gm1 * 0.99 + m1 * 0.01)
    #     av1 = tf.assign(gv1, gv1 * 0.99 + v1 * 0.01)
    #     #gm1 = 0.99 * gm1 + 0.01 * m1 # keep track of global values of mean and variance for use at test-time
    #     #gv1 = 0.99 * gv1 + 0.01 * v1
    #     Z1n = (Z1 - m1) / tf.sqrt(v1 + epsilon) # batch-normalization
    #     A1 = tf.nn.relu(Z1n)

    #     Z2 = tf.add(tf.matmul(W2, A1), b2)
    #     m2, v2 = tf.nn.moments(Z2, [0])
    #     am2 = tf.assign(gm2, gm2 * 0.99 + m2 * 0.01)
    #     av2 = tf.assign(gv2, gv2 * 0.99 + v2 * 0.01)
    #     #gm2 = 0.99 * gm2 + 0.01 * m2 # keep track of global values of mean and variance for use at test-time
    #     #gv2 = 0.99 * gv2 + 0.01 * v2
    #     Z2n = (Z2 - m2) / tf.sqrt(v2 + epsilon)
    #     A2 = tf.nn.tanh(Z2n)

    #     Z3 = tf.add(tf.matmul(W3, A2), b3)
    #     return Z3, gm1, gv1, gm2, gv2

    # else: # at test time, we use the global values of mean and variance instead
    #     Z1 = tf.add(tf.matmul(W1, X), b1)
    #     Z1n = (Z1 - gm1) / tf.sqrt(gv1 + epsilon) # batch-normalization
    #     A1 = tf.nn.relu(Z1n)

    #     Z2 = tf.add(tf.matmul(W2, A1), b2)
    #     Z2n = (Z2 - gm2) / tf.sqrt(gv2 + epsilon)
    #     A2 = tf.nn.tanh(Z2n)

    #     Z3 = tf.add(tf.matmul(W3, A2), b3)
    #     return Z3

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

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
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


def predict(X, parameters):
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

    z3 = forward_propagation(x, params, isTraining = False)
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
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # simpler, list-comprehension implementation of code from Deep Learning course
    mini_batches = [(shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size], shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]) for k in range(math.ceil(m/mini_batch_size))]

    return mini_batches