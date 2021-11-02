import numpy as np

class UtilsBinary:
  def create_weights(data):
    a, b = np.shape(data)
    weights = np.zeros(b)
    return weights
  
  def score(x, y, weights):
    prediction = np.sign(np.dot(weights, np.transpose(x)))
    return 1 - np.count_nonzero(prediction - y) / y.shape[0]
  
  def predict(x, weights):
    return np.sign(np.dot(weights, x))
  
  def loadData(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    # extract features and label
    y = np.asarray(data[:, :1].flatten(), dtype="float")
    X = np.asarray(data[:,1:], dtype="float")/255
    return X, y
