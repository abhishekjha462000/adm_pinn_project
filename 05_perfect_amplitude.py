import matplotlib
matplotlib.use("Agg")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

dde.config.set_default_float("float32")
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

# We leave Amplitude hidden, but fix Sigma to the true value (0.1)
hidden_amplitude = dde.Variable(10.0)
true_sigma = 0.1 

def sink_source(x):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    return -hidden_amplitude * torch.exp(-(x_coord**2 + y_coord**2) / (2 * true_sigma**2))

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

# Targeted Singularity Enrichment
sink_idx = np.where(observe_u < -1.0)[0]
sink_x = observe_x[sink_idx]
sink_u = observe_u[sink_idx]
sink_bc = dde.icbc.PointSetBC(sink_x, sink_u, component=0)

data = dde.data.PDE(
    geom, pde, [bc, observe_bc, sink_bc], num_domain=2000, num_boundary=400
)

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

tracker = dde.callbacks.VariableValue(
    [hidden_amplitude], period=500, filename="perfect_amplitude.txt"
)

print("Starting Perfect Amplitude Discovery...")
model.compile("adam", lr=0.005, loss_weights=[1, 1, 10, 1000], external_trainable_variables=[hidden_amplitude])
model.train(iterations=5000, callbacks=[tracker])

dde.optimizers.config.set_LBFGS_options(maxiter=5000)
model.compile("L-BFGS", loss_weights=[1, 1, 10, 1000], external_trainable_variables=[hidden_amplitude])
model.train(callbacks=[tracker])

print("Training complete. Generating the final graph...")

steps = []
pred_amp = []
with open("perfect_amplitude.txt", "r") as f:
    for line in f:
        clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
        parts = clean_line.split()
        if len(parts) >= 2:
            steps.append(float(parts[0]))
            pred_amp.append(float(parts[1]))

plt.figure(figsize=(7, 5))
plt.plot(steps, pred_amp, label="Predicted Amplitude", color="blue", linewidth=2.5)
plt.axhline(100.0, color='red', linestyle='--', label="True Value (100.0)", linewidth=2)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.title("Successful Decoupled Discovery", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("Day5_Decoupled_Discovery.png", dpi=300, bbox_inches="tight")
print("Success! Saved Day5_Decoupled_Discovery.png")