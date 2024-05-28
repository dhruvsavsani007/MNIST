import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import pickle


f = gzip.open('MNIST\mnist.pkl.gz', 'rb')
data = pickle.load(f, encoding='bytes')


(x_train, y_train), (x_test, y_test) = data
print(x_test.shape)