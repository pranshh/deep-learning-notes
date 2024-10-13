from numpy import load
import numpy as np

data = load("parameters.npz")
lst = data.files

w1 = data[lst[0]]
b1 = data[lst[1]]
w2 = data[lst[2]]
b2 = data[lst[3]]
w3 = data[lst[4]]
b3 = data[lst[5]]

x = np.array([[1], 
            [0],
            [1]])
y = np.array([[0],
            [0], 
            [1]])

a1 = b1 + np.matmul(w1,x)

# print(a1)
# print(np.sum(a1))
# print(a1.shape)

h1 = np.array([1/(1+np.exp(-a1[0])), 1/(1+np.exp(-a1[1])), 1/(1+np.exp(-a1[2]))])

# print(h1)
# print(h1.sum())
# print(h1.shape)

a2 = b2 + np.matmul(w2, h1)

# print(a2)
# print(a2.sum())
# print(a2.shape)

h2 = np.array([1/(1+np.exp(-a2[0])), 1/(1+np.exp(-a2[1])), 1/(1+np.exp(-a2[2]))])

# print(h2)
# print(h2.sum())
# print(h2.shape)

a3 = b3 + np.matmul(w3, h2)

# print(a3)
# print(a3.sum())
# print(a3.shape)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

y_hat = softmax(a3)

loss = -np.sum(y * np.log(y_hat))

# print(loss)

grad_a3 = y_hat - y
# print(grad_a3)