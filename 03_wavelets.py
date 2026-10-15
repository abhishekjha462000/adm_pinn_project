import matplotlib
matplotlib.use("Agg")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

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

# FIX: Removed loss_weights from here
data = dde.data.PDE(
    geom, pde, [bc, observe_bc], num_domain=2000, num_boundary=400
)

def wavelet_activation(x):
    return torch.cos(5.0 * x) * torch.exp(-0.5 * x**2)

print("Building network with Morlet Wavelet Enrichment...")
net = dde.nn.FNN([2] + [50] * 4 + [1], wavelet_activation, "Glorot uniform")
model = dde.Model(data, net)

variable_tracker = dde.callbacks.VariableValue(
    [hidden_amplitude, hidden_sigma], period=500, filename="wavelet_features.txt"
)

print("Starting Phase 1: Adam Training...")
# FIX: Added loss_weights=[1, 1, 100] here to force the network to respect sensor data
model.compile("adam", lr=0.005, loss_weights=[1, 1, 100], external_trainable_variables=[hidden_amplitude, hidden_sigma])
model.train(iterations=10000, callbacks=[variable_tracker])

print("Starting Phase 2: L-BFGS Fine-Tuning...")
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
# FIX: Added loss_weights here as well
model.compile("L-BFGS", loss_weights=[1, 1, 100], external_trainable_variables=[hidden_amplitude, hidden_sigma])
losshistory, train_state = model.train(callbacks=[variable_tracker])

print("Training complete. Generating Wavelet Enrichment figures...")

# Read the file and strip out the brackets and commas
steps = []
pred_amp = []
pred_sig = []
with open("wavelet_features.txt", "r") as f:
    for line in f:
        clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
        parts = clean_line.split()
        if len(parts) >= 3:
            steps.append(float(parts[0]))
            pred_amp.append(float(parts[1]))
            pred_sig.append(float(parts[2]))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, pred_amp, label="Wavelet Predicted Amplitude", color="blue")
plt.axhline(100.0, color='red', linestyle='--', label="True Value (100.0)")
plt.xlabel("Iterations")
plt.ylabel("Amplitude")
plt.title("Wavelet Enrichment: Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(steps, pred_sig, label="Wavelet Predicted Sigma", color="green")
plt.axhline(0.1, color='red', linestyle='--', label="True Value (0.1)")
plt.xlabel("Iterations")
plt.ylabel("Sigma")
plt.title("Wavelet Enrichment: Sigma")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("Day3_Wavelet_Discovery.png", dpi=300, bbox_inches="tight")
print("Success! Saved Day3_Wavelet_Discovery.png")