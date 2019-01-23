import tensorflow as tf
import numpy as np
# from tensorflow.python.tools import inspect_checkpoint as chkp
from ml_helpers_new import forward_propagation, load_data

def predict(learning_rate, iterations, layer1, layer2, beta):
    """ Providing the hyperparameters for the particular, already-trained model,
        evaluates the model on the test set.

        Returns:
        compare: an array of booleans, equals True where the prediction equals the actual value
        prediction: the model's predicted values
        actuals: the actual, correct labels
        test_acc: the percent accuracy of the predictions
    """
    tf.reset_default_graph()

    W1 = tf.get_variable("W1", [layer1, 170])
    b1 = tf.get_variable("b1", [layer1, 1])
    W2 = tf.get_variable("W2", [layer2, layer1])
    b2 = tf.get_variable("b2", [layer2, 1])
    W3 = tf.get_variable("W3", [5, layer2])
    b3 = tf.get_variable("b3", [5, 1])

    betas = tf.get_variable("batch_normalization/beta", [layer1])
    betas1 = tf.get_variable("batch_normalization_1/beta", [layer2])
    gammas = tf.get_variable("batch_normalization/gamma", [layer1])
    gammas1 = tf.get_variable("batch_normalization_1/gamma", [layer2])
    means = tf.get_variable("batch_normalization/moving_mean", [layer1])
    means1 = tf.get_variable("batch_normalization_1/moving_mean", [layer2])
    variances = tf.get_variable("batch_normalization/moving_variance", [layer1])
    variances1 = tf.get_variable("batch_normalization_1/moving_variance", [layer2])

    saver = tf.train.Saver()
    
    with tf.Session() as sess:        
        saver.restore(sess, "./tmp/model_{}_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, layer2, beta))
        
        X = tf.placeholder(tf.float32, shape=(170, None), name='X')
        
        Z1 = tf.add(tf.matmul(W1.eval(), X), b1.eval())
        Z1 = tf.nn.batch_normalization(Z1, tf.reshape(means.eval(), [layer1, 1]), tf.reshape(variances.eval(), [layer1, 1]), \
        tf.reshape(betas.eval(), [layer1, 1]), tf.reshape(gammas.eval(), [layer1, 1]), 1e-5)
        
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2.eval(), A1), b2.eval())
        Z2 = tf.nn.batch_normalization(Z2, tf.reshape(means1.eval(), [layer2, 1]), tf.reshape(variances1.eval(), [layer2, 1]), \
        tf.reshape(betas1.eval(), [layer2, 1]), tf.reshape(gammas1.eval(), [layer2, 1]), 1e-5)
        
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3.eval(), A2), b3.eval())
        
        p = tf.argmax(Z3)

        [_, _, X_test, _, _, Y_test, _] = load_data('new')

        # for i in [W1, b1, W2, b2, W3, b3]:
        #     print(i.eval())

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={X: X_test})
    actual = Y_test

    compare = np.equal(prediction, actual) # compares the two arrays element-wise, returns an array with True when both are equal
    test_acc = np.round(np.sum(compare) / compare.size, 8) # sum the array and divide by its size to find the final test accuracy

    return compare, prediction, actual, test_acc

def predict_regression(learning_rate, iterations, layer1, beta):
    tf.reset_default_graph()

    W1 = tf.get_variable("W1", [layer1, 170])
    b1 = tf.get_variable("b1", [layer1, 1])
    W2 = tf.get_variable("W2", [1, layer1])
    b2 = tf.get_variable("b2", [1, 1])
    
    betas = tf.get_variable("batch_normalization/beta", [layer1])
    gammas = tf.get_variable("batch_normalization/gamma", [layer1])
    means = tf.get_variable("batch_normalization/moving_mean", [layer1])
    variances = tf.get_variable("batch_normalization/moving_variance", [layer1])
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./models/regression_model_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, beta))

        X = tf.placeholder(tf.float32, shape=(170, None), name='X')
        
        Z1 = tf.add(tf.matmul(W1.eval(), X), b1.eval())
        Z1 = tf.nn.batch_normalization(Z1, tf.reshape(means.eval(), [layer1, 1]), tf.reshape(variances.eval(), [layer1, 1]), \
        tf.reshape(betas.eval(), [layer1, 1]), tf.reshape(gammas.eval(), [layer1, 1]), 1e-5)
        
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2.eval(), A1), b2.eval())
        
        p = tf.argmax(Z2)

        [_, _, X_test, _, _, Y_test, _] = load_data('regression')

        for i in [W1, b1, W2, b2]:
            print(i.eval())

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={X: X_test})
    actual = Y_test

    compare = np.equal(prediction, actual) # compares the two arrays element-wise, returns an array with True when both are equal
    test_acc = np.round(np.sum(compare) / compare.size, 8) # sum the array and divide by its size to find the final test accuracy

    return compare, prediction, actual, test_acc


if __name__ == "__main__":
    learning_rate = float(input('Learning rate? '))
    iterations = int(input('Iterations? '))
    layer1 = int(input('Layer 1? '))
    beta = float(input('Beta? '))
    # [learning_rate, iterations, layer1, layer2, beta] = [0.0003, 2501, 12, 6, 0.01]

    # compare, prediction, actual, test_acc = predict(learning_rate, iterations, layer1, layer2, beta)
    compare, prediction, actual, test_acc = predict_regression(learning_rate, iterations, layer1, beta)
    compare = np.logical_not(compare)
    prediction = prediction.reshape(1, prediction.size)

    print('Test accuracy:', test_acc)
    wrong_predicts = prediction[compare]
    wrong_actual = actual[compare]
    print('Wrong Predict:', wrong_predicts)
    print('Right Actuals:', wrong_actual)