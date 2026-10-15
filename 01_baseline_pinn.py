import deepxde as dde
import numpy as np
import torch

# Set DeepXDE backend to PyTorch
dde.config.set_default_float("float32")

# 1. Define the Domain: A 2D square from -1 to 1
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

# 2. Define the Point Sink (A sharp Gaussian well to simulate a point load)
def sink_source(x):
    # x is a tensor of shape (N, 2) where column 0 is X and column 1 is Y
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    amplitude = 100.0
    sigma = 0.1 # Small sigma makes it a sharp sink that breaks standard FEM
    return -amplitude * torch.exp(-(x_coord**2 + y_coord**2) / (2 * sigma**2))

# 3. Define the PDE: -Laplacian(u) = f(x,y) -> -(du_xx + du_yy) - f = 0
def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    f = sink_source(x)
    return - (du_xx + du_yy) - f

# 4. Boundary Condition: u = 0 on all edges of the square
def boundary(_, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

# 5. Compile the Data (2000 points inside, 400 on boundary)
data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=2000,
    num_boundary=400,
    num_test=1000,
)

# 6. Build the Neural Network (4 hidden layers, 50 neurons each)
net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

# 7. Compile and Train
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

print("Starting Day 1 PINN Training Sprint...")
losshistory, train_state = model.train(iterations=5000)

# 8. Generate and save the plots
dde.saveplot(losshistory, train_state, issave=True, isplot=True)