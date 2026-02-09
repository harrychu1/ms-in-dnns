import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

def save_image(filename):

    p = PdfPages(filename)
    
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    
    for fig in figs: 
        fig.savefig(p, format='pdf') 
    
    p.close()  

def fit_poly(x_train, y_train, k):
    X = torch.ones(len(x_train))
    for i in range(1,k+1):
        X=torch.vstack((X,x_train**i))
    X=X.T
    W = y_train.T@X@torch.linalg.inv(X.T@X)
    return W

def design_matrix(data, k):
    X=torch.ones(len(data))
    for i in range(1,k+1):
        X=torch.vstack((X,data**i))
    return X

#def normalise(x):
#    return (x-torch.mean(x_train))/torch.std(x_train)

def poly(x,W):
    W=W.flatten()
    X=design_matrix(x,len(W)-1)
    return W@X

##Generate data
N_TRAIN = 15
SIGMA_NOISE = 0.1

torch.manual_seed(0xDEADBEEF)
x_train = torch.rand(N_TRAIN) * 2 * torch.pi
y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE

"""X = torch.ones(len(x_train))
for i in range(1,4):
    X=torch.vstack((X,x_train**i))
X=X.T"""


## Model class
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        X=design_matrix(x, 3).T
        z1 = self.linear(X)
        return z1

#SGD
model=LinearModel()
torch.nn.init.ones_(model.linear.weight)
sgd=torch.optim.SGD(model.parameters(), lr=0.00007)
loss_fn=nn.MSELoss()
model.train()

loss_plot=[]
for i in range(100): 
    preds = model(x_train).flatten()
    targets = y_train
    loss = loss_fn(preds, targets)
    loss.backward()
    sgd.step()
    sgd.zero_grad()
    loss_plot.append(loss.item())



#Plots
fig1 = plt.figure()
points=torch.linspace(0,2*torch.pi, 100)
plt.plot(points, poly(points, fit_poly(x_train, y_train, 3)), label="Least squares")
plt.plot(points, model(points).detach(), label="Model")
plt.scatter(x_train, y_train)
plt.plot(points, torch.sin(points), label="real")
plt.legend()
plt.title("SGD with ones initialisation")


#[print(param) for param in model.named_parameters()]
##test for different rates
fig2 = plt.figure()
plt.plot(range(len(loss_plot)), loss_plot)
plt.yscale("log")
plt.title("Loss SGD ones")



#1b Hessian
eig_vals=torch.abs(torch.linalg.eigvals(2*design_matrix(x_train, 3)@design_matrix(x_train, 3).T))
condition_number=torch.max(eig_vals)/torch.min(eig_vals)
print("condition_number", condition_number)

##1b initialisation scheme
model=LinearModel()
model.linear.weight=torch.nn.parameter.Parameter(torch.tensor([1,0.1, 0.01, 0.001]))
sgd=torch.optim.SGD(model.parameters(), lr=0.00007, momentum=0.9)
loss_fn=nn.MSELoss()
model.train()
#[print(param) for param in model.named_parameters()]

loss_plot=[]
for i in range(100): 
    preds = model(x_train).flatten()
    targets = y_train
    loss = loss_fn(preds, targets)
    loss.backward()
    sgd.step()
    sgd.zero_grad()
    loss_plot.append(loss.item())

fig3 = plt.figure()
plt.plot(points, poly(points, fit_poly(x_train, y_train, 3)), label="Least squares")
plt.plot(points, model(points).detach(), label="Model")
plt.scatter(x_train, y_train)
plt.plot(points, torch.sin(points), label="real")
plt.legend()
plt.title("SGD with momentum and initialisation scheme")

fig4 = plt.figure()
plt.plot(range(len(loss_plot)), loss_plot)
plt.yscale("log")
plt.title("Loss SGD initialisation scheme")



##1b Adam
model=LinearModel()
torch.nn.init.ones_(model.linear.weight)
adam=torch.optim.Adam(model.parameters(), lr=0.2)
loss_fn=nn.MSELoss()
model.train()
#[print(param) for param in model.named_parameters()]
loss_plot=[]
for i in range(100): 
    preds = model(x_train).flatten()
    targets = y_train
    loss = loss_fn(preds, targets)
    loss.backward()
    adam.step()
    adam.zero_grad()
    loss_plot.append(loss.item())
    if i == 99:
        print("Final loss:", loss.item())

fig5 = plt.figure()
plt.plot(points, poly(points, fit_poly(x_train, y_train, 3)), label="Least squares")
plt.plot(points, model(points).detach(), label="Model")
plt.scatter(x_train, y_train)
plt.plot(points, torch.sin(points), label="real")
plt.legend()
plt.title("adam with ones initialisation")


fig6 = plt.figure()
plt.plot(range(len(loss_plot)), loss_plot)
plt.yscale("log")
plt.title("Loss Adam ones")



##1b LBFGS
model=LinearModel()
torch.nn.init.ones_(model.linear.weight)
LBFGS=torch.optim.LBFGS(model.parameters(), lr=0.01)
loss_fn=nn.MSELoss()
model.train()
#[print(param) for param in model.named_parameters()]
loss_plot=[]
for i in range(100): 
    def closure():
        LBFGS.zero_grad()
        preds = model(x_train).flatten()
        targets = y_train
        loss = loss_fn(preds, targets)
        loss.backward()
        return loss
    loss_plot.append(loss_fn(model(x_train).flatten(), y_train).item())
    LBFGS.step(closure)

fig7 = plt.figure()
plt.plot(points, poly(points, fit_poly(x_train, y_train, 3)), label="Least squares")
plt.plot(points, model(points).detach(), label="Model")
plt.scatter(x_train, y_train)
plt.plot(points, torch.sin(points), label="real")
plt.legend()
plt.title("LBFGS with ones initialisation")


fig8 = plt.figure()
plt.plot(range(len(loss_plot)), loss_plot)
plt.yscale("log")
plt.title("Loss LBFGS ones")

filename = "AssignmentC_1.pdf"  

save_image(filename)  







