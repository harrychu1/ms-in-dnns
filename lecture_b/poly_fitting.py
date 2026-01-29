import numpy as np
import matplotlib.pyplot as plt

##Part a)

# Generate 15 training data points and 10 test data points
x = np.random.uniform(0, 2 * np.pi, 25)
y = np.sin(x) + np.random.normal(0, 0.1, 25)
I=np.linspace(0, 2*np.pi, 1000)

#Split the data
train_data=[x[:15], y[:15]]
test_data=[x[15:], y[15:]]

#Least squares method
def fit_poly(x_train, y_train, k):
    X = np.ones(len(x_train))
    for i in range(1,k+1):
        X=np.vstack((X,x_train**i))
    X=X.T
    W = y_train.T@X@np.linalg.inv(X.T@X)
    return W


def design_matrix(data, k):
    X=np.ones(len(data))
    for i in range(1,k+1):
        X=np.vstack((X,data**i))
    return X

def poly(x,W):
    W=W.flatten()
    X=design_matrix(x,len(W)-1)
    return W@X

def mse_poly(x, y, W):
    W=W.flatten()
    X=design_matrix(x, len(W)-1)
    sum=np.sum((W@X - y)**2)
    return sum/len(x)

degree=3
W=fit_poly(train_data[0],train_data[1],degree)
X=design_matrix(I,degree)
MSE= mse_poly(test_data[0], test_data[1], W)

## Overfitting
x1 = np.random.uniform(0, 4 * np.pi, 25)
y1 = np.sin(x1) + np.random.normal(0, 0.1, 25)

train_data1=[x1[:15], y1[:15]]
test_data1=[x1[15:], y1[15:]]
MSE_vec = np.zeros(15)

I1=np.linspace(0, 4*np.pi, 1000)
for k in range(1,16):
    W1=fit_poly(train_data1[0],train_data1[1],k)
    MSE_vec[k-1]=mse_poly(test_data1[0], test_data1[1], W1)
    if k == 5:
        W5 = W1

X1=design_matrix(I1,5)

def ridge_fit_poly(x_train, y_train, k, lamb):
    X=design_matrix(x_train, k)
    X=X.T
    W = y_train.T@X@np.linalg.inv(X.T@X+lamb*np.identity(k+1))
    return W

MSE_mat=np.empty((20, 20))

for i, k in enumerate(list(range(1,21))):
    for j, lamb in enumerate(10**np.linspace(-5,0,20)):
        W_r = ridge_fit_poly(train_data1[0], train_data1[1], k, lamb)
        MSE_mat[i,j] = mse_poly(test_data1[0], test_data1[1], W_r)

def perform_cv(x, y, k, lamb, folds):
    N=len(x)
    batch_sz = int(N/folds)
    MSE=0
    for i in range(folds):
        x_train1 = x[:i*batch_sz]
        x_train2 = x[(i+1)*batch_sz:]
        x_train=np.concatenate((x_train1, x_train2))

        y_train1 = y[:i*batch_sz]
        y_train2 = y[(i+1)*batch_sz:]
        y_train=np.concatenate((y_train1, y_train2))

        x_test = x[i*batch_sz:(i+1)*batch_sz]
        y_test = y[i*batch_sz:(i+1)*batch_sz]

        W_r = ridge_fit_poly(x_train, y_train, k, lamb)
        MSE += mse_poly(x_test, y_test, W_r)
    return MSE/folds

k=8
lamb=10**np.linspace(-5,0,20)[12]
divisors = [2,3,4,5,6,8,10,12,15,20,24,30,40,60,120]

res=[]
for cv_iterations in range(100):
    cv_x= np.random.uniform(0,4*np.pi, 120)
    cv_y= np.sin(cv_x) + np.random.normal(0,0.1)
    MSE_cv=np.zeros(len(divisors))
    for i, folds in enumerate(divisors):
        MSE_cv[i]=perform_cv(cv_x, cv_y, k, lamb, folds)

    res.append(MSE_cv)

res=np.array(res)
#res.flatten()
#res.reshape((len(divisors),100))


def main():
    plt.scatter(train_data[0], train_data[1], label="train_data")
    plt.scatter(test_data[0], test_data[1], label="test_data")
    plt.plot(I, W@X)
    plt.plot(I, np.sin(I))
    plt.plot([], [], label=f"MSE: {round(MSE,2)}")
    plt.legend()
    plt.show()

    plt.plot(range(1,16), MSE_vec)
    plt.yscale('log')
    plt.show()

    plt.scatter(train_data1[0], train_data1[1], label="train_data")
    plt.scatter(test_data1[0], test_data1[1], label="test_data")
    plt.plot(I1, W5@X1)
    plt.plot(I1, np.sin(I1))
    plt.legend()
    plt.show()

    plt.imshow(np.log(MSE_mat.T))
    plt.xlabel("k")
    plt.ylabel("lambda")
    plt.show()
    
    plt.plot(divisors, np.mean(res, axis=0))
    plt.plot(divisors, np.mean(res, axis=0)+np.std(res, axis=0), linestyle="dashed")
    plt.plot(divisors, np.maximum(np.zeros(len(divisors)), np.mean(res, axis=0)-np.std(res, axis=0)), linestyle="dashed")
    
    plt.show()

if __name__ == "__main__":
    main()
