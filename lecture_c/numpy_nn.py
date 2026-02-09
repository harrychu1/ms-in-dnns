import numpy as np
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

class NPLinear():

    def __init__(self, in_channels, output_channels):
        self.in_channels=in_channels
        self.out_channels=output_channels
        
        self.W=np.random.uniform(-np.sqrt(2/self.in_channels), np.sqrt(2/self.in_channels), size=(output_channels, in_channels))
        self.b=np.random.uniform(-np.sqrt(2/self.in_channels), np.sqrt(2/self.in_channels), size=(output_channels,))
        self.W_grad=np.zeros((output_channels, in_channels))
        self.b_grad=np.zeros((output_channels))

    def forward(self, inputs):
        self.inputs=inputs
        return inputs @ (self.W).T + self.b
    
    def backward(self, loss_grad):
        self.W_grad = loss_grad.T @ self.inputs
        self.b_grad = loss_grad.sum(axis=0)
        return loss_grad @ self.W

    def gd_update(self, lr):
        self.W = self.W - lr*self.W_grad
        self.b = self.b - lr*self.b_grad


class NPMSELoss():

    def forward(self, preds, targets):
        self.preds = preds
        self.targets = targets
        return np.mean((preds-targets)**2)
    
    def backward(self):
        return 2*(self.preds-self.targets)/(self.preds.shape[0]*self.preds.shape[1])


class NPReLU():

    def forward(self, input):
        self.input = input
        return np.maximum(np.zeros(input.shape[1]), input)
    
    def backward(self, loss_grad):
        return loss_grad * (self.input > np.zeros((self.input.shape[0],self.input.shape[1])))

class NPModel():        
    
    def __init__(self, in_channels, out_channels):
        self.layer1=NPLinear(in_channels, 16)
        self.relu=NPReLU()
        self.layer2=NPLinear(16, out_channels)

    def forward(self, batch):
        #Return predictions
        return self.layer2.forward(
            self.relu.forward(
            self.layer1.forward(batch)))

    def backward(self, loss_grad):
        #Set weights and biases
        z1=self.layer2.backward(loss_grad)
        z2=self.relu.backward(z1)
        self.layer1.backward(z2)

    def gd_update(self, lr):
        self.layer1.gd_update(lr)
        self.layer2.gd_update(lr)

N_TRAIN = 100
N_TEST = 1000
SIGMA_NOISE = 0.1

np.random.seed(0xDEADBEEF)
x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE

train_loss=[]
val_loss=[]
model=NPModel(1,1)
loss_fn=NPMSELoss()
lr = 0.1
points=np.linspace(-np.pi,np.pi, 100)[:, None]
k=0
for i in range(77):
    preds = model.forward(x_train)
    targets = y_train
    loss = loss_fn.forward(preds, targets)
    train_loss.append(loss)

    vloss=loss_fn.forward(model.forward(x_test), y_test)
    val_loss.append(vloss)

    model.backward(loss_fn.backward())
    model.gd_update(lr)
    lr=lr*0.99

    if i in [0, 15, 30, 45, 60, 76]:
        if k == 0:
            fig1=plt.figure()
        if k == 1:
            fig2=plt.figure()
        if k == 2:
            fig3=plt.figure()
        if k == 3:
            fig4=plt.figure()
        if k == 4:
            fig5=plt.figure()
        if k == 5:
            fig6=plt.figure()
        plt.plot(points, np.sin(points), label="Real")
        plt.plot(points, model.forward(points), label="Model")
        plt.legend()
        plt.title(f"Epoch {i}")
        k+=1

preds = model.forward(x_test)
targets = y_test
validation_loss = loss_fn.forward(preds, targets)
print(validation_loss)

fig7 = plt.figure()
plt.plot(range(len(train_loss)), train_loss, label="training loss")
plt.plot(range(len(val_loss)), val_loss, label="validation loss")
plt.title("loss")
plt.yscale("log")
plt.legend()


filename = "AssignmentC_2.pdf"  

save_image(filename)  


