#선형 회귀
#2 -> 5, 4 -> 9, 10 -> 21 : y= 2 * x + 1
#y = w*x + b

import numpy as np
x_train = np.array([1.,2.,3.,4.,5.,6.]) #y = 7 * x + 2
y_train = np.array([9.,16.,23.,30.,37.,44.])

#y = w*x + b
W = 0.0
b = 0.0

n_data = len(x_train)

epochs = 5000
learning_rate = 0.01

for i in range(epochs):
    hypothesis = x_train * W + b

    cost = np.sum((hypothesis - y_train) ** 2)/n_data

    gradient_w = np.sum((W * x_train + b - y_train )*2*x_train)/n_data
    gradient_b = np.sum((W * x_train + b - y_train) * 2 ) / n_data

    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch : ({:10d}/{:10d}) cost : {:10f}, W : {:10f}, b : {:10f}'.format(i,epochs,cost,W,b))