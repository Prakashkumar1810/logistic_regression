import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sigmoid function
def sigmoid(z):
	return 1/(1+np.exp(-z))


#finding the gradient of the cost function at given weights
def grad(x,y,t):
	m=y.size
	gradient = np.transpose(1/m*np.matmul(np.transpose(sigmoid(np.matmul(x,t))-y),x))
	return gradient


#cost function to calculate the loss of the model
def cost(x,y,t):
	return -np.sum(y*np.log10(sigmoid(np.matmul(x,t)))+(1-y)*np.log10(1-sigmoid(np.matmul(x,t))))

#loading the training set
dataset = pd.read_csv('data.txt',delimiter=',',header=None)
X = dataset.iloc[:,:-1].values
X = np.insert(X,0,1,axis=1)
Y = dataset.iloc[:,-1].values.reshape(X.shape[0],1)

#training the model with learning rate 1
np.random.seed(1)
w = np.random.rand(X.shape[1],1)
loss = np.zeros(1000)
for epoch in range(1000):
	w=w-grad(X,Y,w)
	loss[epoch]=cost(X,Y,w)

#ploting the convergence curve
plt.plot(range(1000),loss)
plt.xlabel('No of iterations')
plt.ylabel('Cost')
plt.title('Convergence')
plt.show()

#loading the test set
test_set = pd.read_csv('test_data.txt',delimiter=',',header=None)
X_test = test_set.iloc[:,:-1].values
X_test = np.insert(X_test,0,1,axis=1)
Y_test = test_set.iloc[:,-1].values.reshape(X_test.shape[0],1)

print('Expected ans:')
print(np.transpose(Y_test))
print('Predicted ans:')
#predicting the outcome of training set
ans=sigmoid(np.matmul(X_test,w))
print(np.transpose(np.round(ans)))
