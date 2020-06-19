import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


x = np.array([30,32,25,20,36,32,35,28,12,20]).reshape(-1, 1)
y = np.array([200,220,180,140,320,280,340,250,150,220]).reshape(-1, 1)

def calSum(z):
    x =0
    for i in range(len(z)):
            x += z[i]
    return x
def calMult(x,y):
    z = 0 ;
    for i in range(len(x)):
        z += x[i]*y[i]
    return z 

def calSquare(x):
    z = 0
    for i in range(len(x)):
        z +=  x[i]**2
    return z


def formula(x):
    z = []
    for i in x:
        z.append(result[0] + result[1]*i)
    return z



V1 = calSum(x)
V2 = calSum(y)
XY = calMult(x,y)
X2 = calSquare(x)
print(V1,V2,XY,X2)

n=len(x)


x1 = np.array([n,V1,V1,X2]).reshape(-1,2)
Xinv = np.linalg.inv(x1)

result = np.dot(Xinv,[V2,XY]) # [a,b] = Y*X^-1 result[0] -> a , result[1] -> b
y_result = formula(x) # result for test data




# sklearn Liner Regression just 3 line of code :)  
'''linear_reg = LinearRegression()
linear_reg.fit(x,y)
lin_y = linear_reg.predict(x)'''

plt.scatter(x,y)
plt.plot(x,y_result,color='red')
#plt.plot(x,lin_y,color='blue')