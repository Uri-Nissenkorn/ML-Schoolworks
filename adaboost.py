#################################
# Your name: Uri Nissenkorn
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from cProfile import label
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data


np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    hypotheses_list = []
    alpha_list = []
    D = np.repeat(1/len(X_train), len(X_train))
    
    for i in range(T):
        (h_pred,h_index,h_theta,h_error) = best_hypotheses(X_train,y_train,D)
        hypotheses_list.append((h_pred,h_index,h_theta))
        alpha_list.append(D)
        
        
        #h_error = h_error/X_train # to percent
        w_t = 0.5 * np.log((1-h_error)/h_error)
        h_t_x = np.array([h_pred if X_train[j][h_index]<=h_theta else -h_pred for j in range(len(X_train))])
        e_x = D*np.exp(-w_t*y_train*h_t_x)
        D = e_x/np.sum(e_x)
        
        print("finished: ",i)
        
    return hypotheses_list, alpha_list 
    



##############################################
# You can add more methods here, if needed.    

def best_hypotheses(train_data,train_labels, D):
    (idxP,thetaP,errorP)=train_hypotheses(train_data,train_labels,1, D)
    (idxM,tethaM,errorM)=train_hypotheses(train_data,train_labels,-1, D)
    
    if(errorP<=errorM):
        return (1,idxP,thetaP,errorP)
    else:
        return (-1,idxM,tethaM,errorM)


def train_hypotheses(train_data,train_labels, pred, D):
    best_theta = train_data[0][0]
    idx = train_data[0]
    min_error = np.Infinity
    max_theta = np.amax(train_data,axis=0)
    
    for j in range(len(train_data[0])): # every word
        for theta in range(int(max_theta[j])): #every possible theta
            error=0
            for i in range(len(train_data)): # every review
                if(train_data[i][j]<=theta): 
                    if(pred!=train_labels[i]): #wrong pred
                        error+=D[i]
                else:
                    if(pred==train_labels[i]): #wrong pred
                        error+=D[i]
            if (error<=min_error):
                min_error=error
                best_theta=theta
                idx=j
    
    return (idx, best_theta,min_error)

def emp_error(X,y, hypotheses, alpha_vals):
    error=0
    for i in range(len(y)):
        s = 0
        for t in range(len(hypotheses)):
            h_t_x=hypotheses[t][0] if X[i][hypotheses[t][1]]<=hypotheses[t][2] else -hypotheses[t][0]
            s+=alpha_vals[t][hypotheses[t][1]]*h_t_x
        pred= 1 if s>=0 else -1
        if(pred!=y[i]):
            error+=1
    return error/len(y)

def calc_loss(X,y, hypotheses, alpha_vals):
    loss=0
    for i in range(len(y)):
        s = 0
        for t in range(len(hypotheses)):
            h_t_x=hypotheses[t][0] if X[i][hypotheses[t][1]]<=hypotheses[t][2] else -hypotheses[t][0]
            s+=alpha_vals[t][hypotheses[t][1]]*h_t_x
        loss+=np.exp(-y[i]*s)
    return loss/len(y)


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    T = 80

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    # You can add more methods here, if needed.
    train_errors=[]
    test_errors=[]
    train_loss=[]
    test_loss=[]
    for i in range(T):
        # print(hypotheses[i][0],vocab[hypotheses[i][1]],"<=",hypotheses[i][2])
        # train_errors.append(emp_error(X_train,y_train,hypotheses[:i],alpha_vals[:i]))
        # test_errors.append(emp_error(X_test,y_test,hypotheses[:i],alpha_vals[:i]))
        test_loss.append(calc_loss(X_test,y_test,hypotheses[:i],alpha_vals[:i]))
        train_loss.append(calc_loss(X_train,y_train,hypotheses[:i],alpha_vals[:i]))

    
    # plt.plot(train_errors, label="train")
    # plt.plot(test_errors, label="test")
    plt.plot(test_loss, label="test loss")
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("error")
    plt.show()

    ##############################################

if __name__ == '__main__':
    main()



