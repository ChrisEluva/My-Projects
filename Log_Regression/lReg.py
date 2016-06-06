__author__ = 'Christopher Eluvathingal, ID:12349241'

import numpy as np

owlname = []

#Importing owls dataset
data = np.genfromtxt('owls15.csv', delimiter=',', dtype=None)
X = np.zeros((len(data),len(data[1])-1))
# Creating two different arrays of numerical values and the string values, a Matrix for numbers and an array of classes of different owl types

for i in range(len(data)):
    owlname.append(data[i][4])
    for j in range(len(data[i]) - 1):
        X[i,j] = data[i][j]
# Splitting the data as required - 2/3 for training and 1/3 for testing
indicies = np.random.randint(len(data),size=len(data))
tr_size = len(data)*2/3
train_ind = indicies[0:tr_size]
test_ind = indicies[tr_size:len(indicies)]
x_train = X[train_ind,:]
x_test = X[test_ind,:]
owlname_train = []
for i in range(len(train_ind)):
    owlname_train.append(owlname[train_ind[i]])
owlname_test = []
for i in range(len(test_ind)):
    owlname_test.append(owlname[test_ind[i]])

# Creating an array of 1 and 0 for classification
def YforClass(owlname, name):
    Y = np.zeros(len(owlname))
    for i in range(len(owlname)):
        if owlname[i] == name:
            Y[i] = 1
    return Y

def sigmoid(X):
    return 1 / (1 + np.exp(- X))

# Calculates the cost for the particular weights
def cost(theta, X, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        p_1 = sigmoid(np.dot(X, theta))
        log_l = (-y)*np.log(p_1) - (1-y)*np.log(1-p_1)
        return log_l.mean()

# Calculates the gradient
def grad(theta, X_1, y):
    p_1 = sigmoid(np.dot(X_1, theta))
    error = p_1 - y
    grad = np.dot(error, X_1) / y.size # gradient vector
    return grad

# These are the initial weights given
theta = 0.1* np.random.randn(X.shape[1]+1)

n_it = 90 # The number of iteration, i.e number of training times
def train(theta,X_1,y,n_it):
    for i in range(n_it):
        g = grad(theta,X_1,y)
        theta=theta-0.05*g #gradient dissent, where 0.05 is the learning rate
        #print cost(theta,X_1,y)
    return theta

classes = [ "BarnOwl", "SnowyOwl", "LongEaredOwl"]
thetas = [ theta, theta, theta ]

X_train_1 = np.append( np.ones((x_train.shape[0], 1)), x_train, axis=1)
for i in range (len(classes)):
    y = YforClass(owlname_train, classes[i])
    #print "training for ", classes[i]
    thetas[i] = train(theta,X_train_1,y,n_it)

def eval(thetas,X_1, owlname):
    print "Incorrectly Classified"
    n = 0
    for i in range(X_1.shape[0]):
        maxP = -1000
        maxK = 0
        for k in range(len(thetas)):
            p_1 = sigmoid(np.dot(X_1[i, :], thetas[k]))
            if p_1 > maxP:
                maxK = k
                maxP = p_1

        if classes[maxK] == owlname[i]:
            n = n + 1
        else:
            #print "Predicted Classification: ", classes[maxK], "for ", i, "Actual Classification: ", owlname[i]11111
            print "Predicted Classification: ", classes[maxK], "Actual Classification: ", owlname[i]
    print "Accuracy (%): ",n*100.0/len(owlname)

X_test_1 = np.append( np.ones((x_test.shape[0], 1)), x_test, axis=1)
eval(thetas, X_test_1, owlname_test)