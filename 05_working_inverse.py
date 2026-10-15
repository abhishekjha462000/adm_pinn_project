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

# The Babuška Fix: Targeted Singularity Enrichment
sink_idx = np.where(observe_u < -1.0)[0]
sink_x = observe_x[sink_idx]
sink_u = observe_u[sink_idx]
sink_bc = dde.icbc.PointSetBC(sink_x, sink_u, component=0)

# FIX: Removed loss_weights from here
data = dde.data.PDE(
    geom, pde, [bc, observe_bc, sink_bc], num_domain=2000, num_boundary=400
)

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

tracker = dde.callbacks.VariableValue(
    [hidden_amplitude, hidden_sigma], period=500, filename="perfect_features.txt"
)

print("Starting Phase 1: Targeted Adam Training...")
# FIX: Added loss_weights here
model.compile("adam", lr=0.005, loss_weights=[1, 1, 10, 1000], external_trainable_variables=[hidden_amplitude, hidden_sigma])
model.train(iterations=8000, callbacks=[tracker])

print("Starting Phase 2: Targeted L-BFGS Fine-Tuning...")
dde.optimizers.config.set_LBFGS_options(maxiter=8000)
# FIX: Added loss_weights here
model.compile("L-BFGS", loss_weights=[1, 1, 10, 1000], external_trainable_variables=[hidden_amplitude, hidden_sigma])
model.train(callbacks=[tracker])

print("Training complete. Generating the Perfect Discovery figures...")

steps = []
pred_amp = []
pred_sig = []
with open("perfect_features.txt", "r") as f:
    for line in f:
        clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
        parts = clean_line.split()
        if len(parts) >= 3:
            steps.append(float(parts[0]))
            pred_amp.append(float(parts[1]))
            pred_sig.append(float(parts[2]))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, pred_amp, label="Predicted Amplitude", color="blue", linewidth=2)
plt.axhline(100.0, color='red', linestyle='--', label="True Value (100.0)")
plt.xlabel("Iterations")
plt.ylabel("Amplitude")
plt.title("Enriched Discovery: Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(steps, pred_sig, label="Predicted Sigma", color="green", linewidth=2)
plt.axhline(0.1, color='red', linestyle='--', label="True Value (0.1)")
plt.xlabel("Iterations")
plt.ylabel("Sigma")
plt.title("Enriched Discovery: Sigma")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("Day5_Perfect_Discovery.png", dpi=300, bbox_inches="tight")
print("Success! Saved Day5_Perfect_Discovery.png")