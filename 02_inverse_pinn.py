import matplotlib
matplotlib.use("Agg")

import deepxde as dde
import numpy as np
import torch

dde.config.set_default_float("float32")
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

hidden_amplitude = dde.Variable(10.0)
hidden_sigma = dde.Variable(0.5)

def sink_source(x):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    return -hidden_amplitude * torch.exp(-(x_coord**2 + y_coord**2) / (2 * hidden_sigma**2))

def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return - (du_xx + du_yy) - sink_source(x)

def boundary(_, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

try:
    sensor_data = np.loadtxt("test.dat")
    observe_x = sensor_data[:, 0:2] 
    observe_u = sensor_data[:, 2:3] 
except FileNotFoundError:
    print("Error: test.dat not found.")
    exit()

observe_bc = dde.icbc.PointSetBC(observe_x, observe_u, component=0)

data = dde.data.PDE(
    geom, pde, [bc, observe_bc], num_domain=2000, num_boundary=400
)

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Set up the tracker
variable_tracker = dde.callbacks.VariableValue(
    [hidden_amplitude, hidden_sigma], period=500, filename="discovered_features.txt"
)

# Phase 1: Adam Optimizer (Broad search)
print("Starting Phase 1: Adam Training...")
model.compile("adam", lr=0.001, external_trainable_variables=[hidden_amplitude, hidden_sigma])
model.train(iterations=10000, callbacks=[variable_tracker])

# Phase 2: L-BFGS Optimizer (Precision fine-tuning)
print("Starting Phase 2: L-BFGS Fine-Tuning...")
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS", external_trainable_variables=[hidden_amplitude, hidden_sigma])
losshistory, train_state = model.train(callbacks=[variable_tracker])

dde.saveplot(losshistory, train_state, issave=True, isplot=False)
print("Training complete. Run plot_day2.py to see the new results!")