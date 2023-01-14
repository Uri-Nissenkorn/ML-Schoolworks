#################################
# Your name: Uri Nissenkorn
#################################


from cmath import log
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

#import matplotlib.pyplot as plt
import scipy

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = np.zeros(data[0].size)
    for i in range(1,T+1):
        sample_id = np.random.choice(len(data))
        if np.dot(data[sample_id], w) * labels[sample_id] < 1:
            w = (1-eta_0/i)*w + (eta_0/i)*C*labels[sample_id]*data[sample_id]
        else:
            w = (1-eta_0/i)*w

    return w



def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    
    w = np.zeros(data[0].size)
    for i in range(1,T+1):
        sample_id = np.random.choice(len(data))
        if np.dot(data[sample_id], w) * labels[sample_id] < 1:
            e = (-labels[sample_id]*data[sample_id]) * scipy.special.softmax(-labels[sample_id]*w*data[sample_id])[0]
            w = w  - (eta_0/i)*e
        else:
            w = w

    return w

#################################

def predict(w, x):
    return 1 if np.dot(w,x)>0 else -1
     

def cross_validation_hinge(train_data, train_labels, validation_data, validation_labels, eta, c, t):
    CV_score = 0
    for j in range(10):
        w = SGD_hinge(train_data,train_labels,c,eta,t)
        score = 0
        for i in range(len(validation_data)):
            score += 1 if predict(w, validation_data[i]) == validation_labels[i] else 0
        score /= len(validation_data)
        
        CV_score += score
    CV_score /= 10
        
    return CV_score

def cross_validation_log(train_data, train_labels, validation_data, validation_labels, eta, t):
    CV_score = 0
    for j in range(10):
        w = SGD_log(train_data,train_labels,eta,t)
        score = 0
        for i in range(len(validation_data)):
            score += 1 if predict(w, validation_data[i]) == validation_labels[i] else 0
        score /= len(validation_data)
        
        CV_score += score
    CV_score /= 10
        
    return CV_score

def optimal_eta_hinge(train_data, train_labels, validation_data, validation_labels):
    eta_power = np.arange(-5,6)
    eta_score = np.zeros(11)
    for i in range(11):
        eta_score[i] = cross_validation_hinge(train_data, train_labels, validation_data, validation_labels,pow(10.0,eta_power[i]),1,1000)
        
    # plt.plot(eta_power, eta_score)
    # plt.ylabel('eta powers of 10')
    # plt.ylabel('eta accuarcy')
    # plt.xlim=[-6,6]
    # plt.savefig("eta_score_hinge.png")
    # plt.clf()
    
    max_pow = eta_power[np.argmax(eta_score)]
    return pow(10.0,max_pow)

def optimal_C_hinge(train_data, train_labels, validation_data, validation_labels, eta_0):
    C_power = np.arange(-5,6)
    C_score = np.zeros(11)
    for i in range(11):
        C_score[i] = cross_validation_hinge(train_data, train_labels, validation_data, validation_labels,eta_0,pow(10.0,C_power[i]),1000)
        
    # plt.plot(C_power, C_score)
    # plt.ylabel('C powers of 10')
    # plt.ylabel('C accuarcy')
    # plt.xlim=[-6,6]
    # plt.savefig("c_score_hinge.png")
    # plt.clf()
    
    max_pow = C_power[np.argmax(C_score)]
    return pow(10.0, max_pow)

def optimal_eta_log(train_data, train_labels, validation_data, validation_labels):
    eta_power = np.arange(-5,6)
    eta_score = np.zeros(11)
    for i in range(11):
        eta_score[i] = cross_validation_log(train_data, train_labels, validation_data, validation_labels,pow(10.0, eta_power[i]),1000)
        
    # plt.plot(eta_power, eta_score)
    # plt.ylabel('eta powers of 10')
    # plt.ylabel('eta accuarcy')
    # plt.xlim=[-6,6]
    # plt.savefig("eta_score_log.png")
    # plt.clf()    
    
    max_pow = eta_power[np.argmax(eta_score)]
    return pow(10.0,max_pow)

def test_accuarcy(data, labels, w):
    score=0
    for i in range(len(data)):
        score += 1 if predict(w, data[i]) == labels[i] else 0
    score /= len(data)
    return score

def test_hinge(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    eta_0 = optimal_eta_hinge(train_data,train_labels, validation_data, validation_labels)
    c = optimal_C_hinge(train_data,train_labels, validation_data, validation_labels, eta_0)
    w = SGD_hinge(train_data,train_labels,c,eta_0,20000)
    # plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    # plt.savefig("w_image_hinge.png")
    # plt.clf()
    
    print(test_accuarcy(test_data,test_labels,w))


def test_log(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    eta_0 = optimal_eta_log(train_data,train_labels, validation_data, validation_labels)
    w = SGD_log(train_data,train_labels,eta_0,20000)
    # plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    # plt.savefig("w_image_log.png")
    # plt.clf()
    print(test_accuarcy(test_data,test_labels,w))
    
    test_SGD_log_W(train_data,train_labels,eta_0,20000)
    
    

def test_SGD_log_W(data, labels, eta_0, T):
    w_norms = np.zeros(T)

    w = np.zeros(data[0].size)
    
    for i in range(1,T+1):
        sample_id = np.random.choice(len(data))
        if np.dot(data[sample_id], w) * labels[sample_id] < 1:
            e = (-labels[sample_id]*data[sample_id]) * scipy.special.softmax(-labels[sample_id]*w*data[sample_id])[0]
            w = w  - (eta_0/i)*e
        else:
            w = w
            
        w_norms[i-1]=np.linalg.norm(w)

    # plt.plot(w_norms)
    # plt.ylabel('iteration')
    # plt.ylabel('w_t norm')
    # plt.savefig("w_norm_log.png")
    # plt.clf()    
    
def test():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    
    test_hinge(train_data, train_labels, validation_data, validation_labels, test_data, test_labels)    
    test_log(train_data, train_labels, validation_data, validation_labels, test_data, test_labels)    
   
    print()

if __name__=="__main__":
    test()
    
#################################