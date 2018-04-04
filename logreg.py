import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1/(1+np.exp(-z))


def grad(x,y,t):
	m=y.size
	lampda=1
	gradient = np.transpose(1/m*np.matmul(np.transpose(sigmoid(np.matmul(x,t))-y),x))
	return gradient


def cost(x,y,t):
	return -np.sum(y*np.log10(sigmoid(np.matmul(x,t)))+(1-y)*np.log10(1-sigmoid(np.matmul(x,t))))

dataset = pd.read_csv('data.txt',delimiter=',',header=None)
X = dataset.iloc[:,:-1].values
X = np.insert(X,0,1,axis=1)
Y = dataset.iloc[:,-1].values.reshape(X.shape[0],1)

np.random.seed(1)
w = np.random.rand(X.shape[1],1)
loss = np.zeros(1000)
for epoch in range(1000):
	w=w-grad(X,Y,w)
	loss[epoch]=cost(X,Y,w)

plt.plot(range(1000),loss)
plt.xlabel('No of iterations')
plt.ylabel('Cost')
plt.title('Convergence')
plt.show()

test_set = pd.read_csv('test_data.txt',delimiter=',',header=None)
X_test = test_set.iloc[:,:-1].values
X_test = np.insert(X_test,0,1,axis=1)
Y_test = test_set.iloc[:,-1].values.reshape(X_test.shape[0],1)

print('Expected ans:')
print(np.transpose(Y_test))
#print(X_test)
ans=sigmoid(np.matmul(X_test,w))
print(np.transpose(np.round(ans)))