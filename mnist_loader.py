import numpy as np
import pickle
import gzip

def load_data():
    """
    'training_data' contains 50,000 examples in one tuple.
    The first entry is 50,000 entries in an array, each 784 containing values
    The second entry is an array, 50,000 entries of the actual number

    'validation_data' and 'test_data' are similar, except only 10,000
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Returns tuples (x,y) where x is the input image, y is the classification

    'training_data' has y as a 10-dimensional unit vector
    'validation_data' and 'test_data' has y simply as the integer classification 
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Used in load_data_wrapper: converts digit classification into the 
    corresponding unit column vector 
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e