#from cox_nnet_v2 import *
import numpy
import sklearn
import sklearn.model_selection
import pandas as pd
import time
import dill
from sklearn.preprocessing import MinMaxScaler

d_path = "BRCA/"


#x = numpy.loadtxt(fname=d_path+"259p_phenotypic27.csv",delimiter=",",skiprows=1)
#x = numpy.loadtxt(fname=d_path+"259p_micro273.csv",delimiter=",",skiprows=1)
x = numpy.loadtxt(fname=d_path+"259p_tumor105.csv",delimiter=",",skiprows=1)

x = x[:, (x != 0).any(axis=0)] 

scaler  = MinMaxScaler() #works best
x   = scaler.fit_transform(x)

file_name = 'BRCA_cindex_tumor_0.708.pkl'

def features_hidden(file_name,data):
    f = open(file_name, 'rb')
    W,b, node_map, input_split, n_samples, x_train, rng = dill.load(f)
    f.close()
    
    W_h = list(W)[0] #shape feature x hidden layer size
    b_h = list(b)[0] #hidden layer size x 1

    # data = samples x features

    Out_h = numpy.dot(data,W_h) + b_h

    print(Out_h.shape)

    #numpy.savetxt("./BRCA/ht_259p_phenotypic27.csv", Out_h, delimiter=",")
    #numpy.savetxt("./BRCA/ht_259p_micro273.csv", Out_h, delimiter=",")
    numpy.savetxt("./BRCA/ht_259p_tumor105.csv", Out_h, delimiter=",")

    print('Done')

    return None


features_hidden(file_name,x)
