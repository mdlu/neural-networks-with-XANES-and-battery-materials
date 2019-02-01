import tensorflow as tf
import numpy as np
# from tensorflow.python.tools import inspect_checkpoint as chkp # use this import if printing the values saved in the checkpoint is desired
from ml_helpers_new import forward_propagation, load_data
import matplotlib.pyplot as plt

# use this import if the above has throws an error on Macs
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

#disables AVX warning from being displayed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

    # creates variables, which are then populated when the model is loaded
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
        saver.restore(sess, "./coord_models/model_{}_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, layer2, beta))
        
        X = tf.placeholder(tf.float32, shape=(170, None), name='X')
        
        # forward propagation
        Z1 = tf.add(tf.matmul(W1.eval(), X), b1.eval())
        Z1 = tf.nn.batch_normalization(Z1, tf.reshape(means.eval(), [layer1, 1]), tf.reshape(variances.eval(), [layer1, 1]), \
        tf.reshape(betas.eval(), [layer1, 1]), tf.reshape(gammas.eval(), [layer1, 1]), 1e-5)
        
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2.eval(), A1), b2.eval())
        Z2 = tf.nn.batch_normalization(Z2, tf.reshape(means1.eval(), [layer2, 1]), tf.reshape(variances1.eval(), [layer2, 1]), \
        tf.reshape(betas1.eval(), [layer2, 1]), tf.reshape(gammas1.eval(), [layer2, 1]), 1e-5)
        
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3.eval(), A2), b3.eval())
        
        # finds the final label predictions on the classification task
        p = tf.argmax(Z3)

        [_, _, X_test, _, _, Y_test, _] = load_data('new')

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={X: X_test})
    actual = Y_test

    compare = np.equal(prediction, actual) # compares the two arrays element-wise, returns an array with True when both are equal
    test_acc = np.round(np.sum(compare) / compare.size, 8) # sum the array and divide by its size to find the final test accuracy

    return compare, prediction, actual, test_acc

def predict_regression(learning_rate, iterations, layer1, beta, real_values = False, dev = True, is_charge = False):
    """ Tests the trained regression model on the test set.
        If real_values is set to True, the model is tested only on the 'real' averaged spectra,
        i.e. those that were obtained by averaging the 49 spectra belonging to a single POSCAR file.
        If dev is set to True, the calculations are done on the cross-validation set rather than the test set.
        If is_charge is set to True, we use the charge data for predictions rather than the averaged spectra.
    """

    tf.reset_default_graph()

    # creates variables to be filled with actual values when the model is loaded
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
        if is_charge:
            saver.restore(sess, "./charge_reg_models/charge_regression_model_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, beta))
        else:
            saver.restore(sess, "./reg_models/regression_model_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, layer1, beta))

        X = tf.placeholder(tf.float32, shape=(170, None), name='X')
        Y = tf.placeholder(tf.float32, shape=(1, None), name='Y')
        
        # forward propagation
        Z1 = tf.add(tf.matmul(W1.eval(), X), b1.eval())
        Z1 = tf.nn.batch_normalization(Z1, tf.reshape(means.eval(), [layer1, 1]), tf.reshape(variances.eval(), [layer1, 1]), \
        tf.reshape(betas.eval(), [layer1, 1]), tf.reshape(gammas.eval(), [layer1, 1]), 1e-5)
        
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2.eval(), A1), b2.eval())
        
        # mean absolute percentage error and mean absolute error
        MAPE = tf.scalar_mul(100, tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Z2, Y), Y))))
        MAE = tf.reduce_mean(tf.abs(tf.subtract(Z2, Y)))

        if real_values:
            X_test = np.load('./parsed_data/xrealaveraged.npy')
            Y_test = np.load('./parsed_data/yrealaveraged.npy')
            # use the below if we want to test removing the last 3 averaged spectra, i.e. the last three of those in folder 1000K with three O-O dimers
            # X_test = X_test[:, :-3]
            # Y_test = Y_test[:, :-3]
        elif is_charge:
            [X_train, X_dev, X_test, _, _, _, Y_train, Y_dev, Y_test, _] = load_data('multi_task')
        else:
            [X_train, X_dev, X_test, Y_train, Y_dev, Y_test, _] = load_data('regression')
            # print('train mape:', MAPE.eval({X: X_train, Y: Y_train}, session = sess))
            # print('dev mape:', MAPE.eval({X: X_dev, Y: Y_dev}, session = sess))
        
    sess = tf.Session()
    
    if dev and not real_values:
        prediction = sess.run(Z2, feed_dict={X: X_dev})
        prediction.reshape(1, prediction.size)
        actual = Y_dev
        actual.reshape(1, actual.size)
        mape = MAPE.eval({X: X_dev, Y: Y_dev}, session = sess)
        mae = MAE.eval({X: X_dev, Y: Y_dev}, session = sess)
    else:
        prediction = sess.run(Z2, feed_dict={X: X_test})
        prediction.reshape(1, prediction.size)
        actual = Y_test
        actual.reshape(1, actual.size)
        mape = MAPE.eval({X: X_test, Y: Y_test}, session = sess)
        mae = MAE.eval({X: X_test, Y: Y_test}, session = sess)

    # places the predictions and actual values side by side for easier visual comparison
    side_by_side = np.concatenate((prediction.T, actual.T), 1)

    return prediction, actual, mape, mae, side_by_side


def predict_multi(learning_rate, iterations, minibatch_size, layer1, layer2_1, layer2_2, beta1, beta2):
    """ Tests the trained multi-task model on the test set. 

        Returns:
        compare: an array of booleans, equals True where the predictions on classification equal the actual value
        prediction1: the predictions the model makes on the classification task
        actual1: the actual labels for the classification task
        prediction2: the predictions the model makes on the charge regression task
        actual2: the actual labels for the charge regression task
        accuracy: the percent accuracy of the model on the classification task
        mape: the mean absolute percentage error of the model on the regression task
    """

    tf.reset_default_graph()

    # create variables
    W1 = tf.get_variable("W1", [layer1, 170])
    b1 = tf.get_variable("b1", [layer1, 1])
    W2_1 = tf.get_variable("W2_1", [layer2_1, layer1])
    b2_1 = tf.get_variable("b2_1", [layer2_1, 1])
    W2_2 = tf.get_variable("W2_2", [layer2_2, layer1])
    b2_2 = tf.get_variable("b2_2", [layer2_2, 1])
    W3_1 = tf.get_variable("W3_1", [5, layer2_1])
    b3_1 = tf.get_variable("b3_1", [5, 1])
    W3_2 = tf.get_variable("W3_2", [1, layer2_2])
    b3_2 = tf.get_variable("b3_2", [1, 1])

    betas = tf.get_variable("batch_normalization/beta", [layer1])
    betas1 = tf.get_variable("batch_normalization_1/beta", [layer2_1])
    betas2 = tf.get_variable("batch_normalization_2/beta", [layer2_2])
    gammas = tf.get_variable("batch_normalization/gamma", [layer1])
    gammas1 = tf.get_variable("batch_normalization_1/gamma", [layer2_1])
    gammas2 = tf.get_variable("batch_normalization_2/gamma", [layer2_2])
    means = tf.get_variable("batch_normalization/moving_mean", [layer1])
    means1 = tf.get_variable("batch_normalization_1/moving_mean", [layer2_1])
    means2 = tf.get_variable("batch_normalization_2/moving_mean", [layer2_2])
    variances = tf.get_variable("batch_normalization/moving_variance", [layer1])
    variances1 = tf.get_variable("batch_normalization_1/moving_variance", [layer2_1])
    variances2 = tf.get_variable("batch_normalization_2/moving_variance", [layer2_2])

    saver = tf.train.Saver()
    
    with tf.Session() as sess:        
        saver.restore(sess, "./multi_models/multi_model_{}_{}_{}_{}_{}_{}_{}_{}.ckpt".format(learning_rate, iterations, minibatch_size, layer1, layer2_1, layer2_2, beta1, beta2))

        X = tf.placeholder(tf.float32, shape=(170, None), name='X')
        Y = tf.placeholder(tf.float32, shape=(1, None), name='Y')
        
        # forward propagation
        Z1 = tf.add(tf.matmul(W1.eval(), X), b1.eval())
        Z1 = tf.nn.batch_normalization(Z1, tf.reshape(means.eval(), [layer1, 1]), tf.reshape(variances.eval(), [layer1, 1]), tf.reshape(betas.eval(), [layer1, 1]), tf.reshape(gammas.eval(), [layer1, 1]), 1e-5)
        A1 = tf.nn.relu(Z1)

        Z2_1 = tf.add(tf.matmul(W2_1.eval(), A1), b2_1.eval())
        Z2_1 = tf.nn.batch_normalization(Z2_1, tf.reshape(means1.eval(), [layer2_1, 1]), tf.reshape(variances1.eval(), [layer2_1, 1]), tf.reshape(betas1.eval(), [layer2_1, 1]), tf.reshape(gammas1.eval(), [layer2_1, 1]), 1e-5)
        A2_1 = tf.nn.relu(Z2_1)

        Z2_2 = tf.add(tf.matmul(W2_2.eval(), A1), b2_2.eval())
        Z2_2 = tf.nn.batch_normalization(Z2_2, tf.reshape(means2.eval(), [layer2_2, 1]), tf.reshape(variances2.eval(), [layer2_2, 1]), tf.reshape(betas2.eval(), [layer2_2, 1]), tf.reshape(gammas2.eval(), [layer2_2, 1]), 1e-5)
        A2_2 = tf.nn.relu(Z2_2)

        Z3_1 = tf.add(tf.matmul(W3_1.eval(), A2_1), b3_1.eval())
        Z3_2 = tf.add(tf.matmul(W3_2.eval(), A2_2), b3_2.eval())
        
        p = tf.argmax(Z3_1)
        MAPE = tf.scalar_mul(100, tf.reduce_mean(tf.abs(tf.divide(tf.subtract(Z3_2, Y), Y))))

        [_, _, X_test, _, _, Y1_test, _, _, Y2_test, _] = load_data('multi_task')

    sess = tf.Session()
    prediction1 = sess.run(p, feed_dict={X: X_test})
    actual1 = Y1_test
    prediction2 = sess.run(Z3_2, feed_dict = {X: X_test})
    actual2 = Y2_test

    compare = np.equal(prediction1, actual1) # compares the two arrays element-wise, returns an array with True when both are equal
    accuracy = np.round(np.sum(compare) / compare.size, 8) # finds the percent accuracy on the coordination classification task
    
    mape = MAPE.eval({X: X_test, Y: Y2_test}, session = sess) # finds the MAPE on the charge regression task

    return compare, prediction1, actual1, prediction2, actual2, accuracy, mape


def graph_trained(learning_rate, iterations, layer1s, beta, real_values = False, dev = True, is_charge = False):
    """ Generates graphs to help visualize the performance of trained models.
        If real_values is True, the models are evaluated on 'real' averaged spectra data.
        If dev is True, the models are evaluated on the cross-validation data.
        If is_charge is True, the models are evaluated on the charge data.
    """

    mapes = []
    maes = []
    
    for n in layer1s:
        _, _, mape, mae, _ = predict_regression(learning_rate, iterations, n, beta, real_values, dev, is_charge)
        mapes.append(mape)
        maes.append(mae)
    
    # MAPEs vs the number of hidden nodes
    plt.figure(1)
    plt.plot(layer1s, mapes)
    plt.title('MAPE vs. Number of Hidden Nodes')
    plt.xlabel('hidden nodes')
    plt.ylabel('MAPE')
    plt.show()

    # MAEs vs the number of hidden nodes
    plt.figure(2)
    plt.plot(layer1s, maes)
    plt.title('MAE vs. Number of Hidden Nodes')
    plt.xlabel('hidden nodes')
    plt.ylabel('MAE')
    plt.show()

    return None


if __name__ == "__main__":

    # # below section used for testing the multi-task models 
    # compare, prediction1, actual1, prediction2, actual2, accuracy, mape = predict_multi(0.000001, 2000, 16, 100, 50, 50, 0.3, 0.03)
    # print(accuracy, mape)
    # print(prediction1)
    # print(actual1)
    # print(prediction2)
    # print(actual2)
    
    # compare = np.logical_not(compare)
    # prediction1 = prediction1.reshape(1, prediction1.size)

    # # print(learning_rate, layer1, layer2, beta)
    # print('\n   Test Accuracy:', accuracy)
    # wrong_predicts = prediction1[compare]
    # wrong_actual = actual1[compare]
    # print('\n   Incorrect Predictions')

    # print('   Predict:', wrong_predicts[:25])
    # print('   Actuals:', wrong_actual[:25])

    # print('\n   Predict:', wrong_predicts[25:])
    # print('   Actuals:', wrong_actual[25:])

    # xs = np.arange(-2.0, -0.5, 0.01)
    # plt.figure(figsize = (7,7))
    # plt.scatter(prediction2, actual2)
    # plt.plot(xs, xs, 'r-')
    # plt.xlabel('predicted')
    # plt.ylabel('actual')
    # plt.show()



    # # below section used to graph the cross-validation set accuracies on the charges
    # nodes = [i for i in range(10, 65)]
    # graph_trained(0.0003, 4000, nodes, 0.01, dev = True, is_charge = True)



    # # below section used to print and graph the model's performance on the test charge data
    # nodes = [40]
    # for x in nodes:
    #     prediction, actual, mape, mae, side_by_side = predict_regression(0.0003, 4000, x, 0.01, dev = False, is_charge = True)
    #     print(prediction)
    #     print(actual)
    #     print(side_by_side)
    #     print(mape)
    #     print(mae)
    #     prediction = prediction.reshape(prediction.size,)
    #     actual = actual.reshape(actual.size,)

    #     xs = np.arange(-2.0, -0.5, 0.01)
    #     plt.figure(figsize = (7,7))
    #     plt.scatter(prediction, actual)
    #     # plt.scatter(a, b)
    #     plt.plot(xs, xs, 'r-')
    #     #plt.ylim(bottom = 1.0, top = 2.1)
    #     #plt.xlim(xmin = 1.0, xmax = 2.1)
    #     plt.xlabel('predicted')
    #     plt.ylabel('actual')
    #     plt.show()



    # # below section used to print and graph the model's performance on the average coordination number data 
    # nodes = [54]
    # mapes = []
    # for x in nodes:
    #     prediction, actual, mape, mae, side_by_side = predict_regression(0.003, 2000, x, 0.0, real_values = True)
    #     prediction1, actual1, mape1, mae1, side_by_side1 = predict_regression(0.003, 2000, x, 0.0, dev = False)
    #     prediction = prediction.reshape(prediction.size,)
    #     actual = actual.reshape(actual.size,)
    #     prediction1 = prediction1.reshape(prediction1.size,)
    #     actual1 = actual1.reshape(actual1.size,)
    #     print(x)
    #     print(mape)
    #     print(mae)
    #     print(mape1)
    #     print(mae1)
    #     print('\n     {} Hidden Nodes'.format(x))
    #     print('  Predicted vs. Actual')
    #     print(side_by_side, '\n')
    #     mapes.append(mape)

    #     np.set_printoptions(threshold=np.nan)
    #     print(prediction)
    #     print(actual)

    #     # plots the model's performance on the "real averages" test set
    #     xs = np.arange(1.0, 2.1, 0.01)
    #     plt.figure(1, figsize = (7,7))
    #     plt.scatter(prediction, actual)
    #     plt.plot(xs, xs, 'r-')
    #     plt.xlabel('predicted')
    #     plt.ylabel('actual')
    #     plt.show()

    #     # plots the model's performance on the test set of artificially averaged data
    #     plt.figure(2, figsize = (7,7))
    #     plt.scatter(prediction1, actual1)
    #     plt.plot(xs, xs, 'r-')
    #     plt.xlabel('predicted')
    #     plt.ylabel('actual')
    #     plt.show()
    # # plots how the MAPE on the "real averages" data varies with the number of hidden nodes
    # plt.plot(nodes, ms)
    # plt.title('MAPE on Real Averages vs. Hidden Nodes')
    # plt.xlabel('hidden nodes')
    # plt.ylabel('MAPE')
    # plt.show()



    # # below section used to print the performance of the Fe coordination number classification model
    # while True:
    #     try:
    #         learning_rate = float(input('\n   Learning rate? '))
    #         iterations = int(input('   Iterations? '))
    #         layer1 = int(input('   Layer 1? '))
    #         layer2 = int(input('   Layer 2? '))
    #         beta = float(input('   Beta? '))
    #         # [learning_rate, iterations, layer1, layer2, beta] = [0.0003, 2501, 12, 6, 0.01]
    #         compare, prediction, actual, test_acc = predict(learning_rate, iterations, layer1, layer2, beta)
    #         break
    #     except ValueError:
    #         print('No trained model was found for those hyperparameters. Please retry.')

    # compare = np.logical_not(compare)
    # prediction = prediction.reshape(1, prediction.size)

    # # print(learning_rate, layer1, layer2, beta)
    # print('\n   Test Accuracy:', test_acc)
    # wrong_predicts = prediction[compare]
    # wrong_actual = actual[compare]
    # print('\n   Incorrect Predictions')
    # print('   Predict:', wrong_predicts)
    # print('   Actuals:', wrong_actual)