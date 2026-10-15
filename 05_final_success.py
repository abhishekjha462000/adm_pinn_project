import matplotlib
matplotlib.use("Agg")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

dde.config.set_default_float("float32")
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

def boundary(_, on_boundary):
    return on_boundary
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

print("--- Phase 1: Generating Smooth Sensor Data ---")
def forward_pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    f = -100.0 * torch.exp(-(x[:, 0:1]**2 + x[:, 1:2]**2) / (2 * 0.4**2))
    return -(du_xx + du_yy) - f

data_fwd = dde.data.PDE(geom, forward_pde, bc, num_domain=1000, num_boundary=400)
model_fwd = dde.Model(data_fwd, dde.nn.FNN([2] + [50]*4 + [1], "tanh", "Glorot uniform"))
model_fwd.compile("adam", lr=0.005)
model_fwd.train(iterations=2000)

X_sensor = geom.random_points(500)
u_sensor = model_fwd.predict(X_sensor)
observe_bc = dde.icbc.PointSetBC(X_sensor, u_sensor, component=0)

print("\n--- Phase 2: Inverse Discovery on Smooth Data ---")
hidden_amplitude = dde.Variable(10.0)

def inverse_pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    f = -hidden_amplitude * torch.exp(-(x[:, 0:1]**2 + x[:, 1:2]**2) / (2 * 0.4**2))
    return -(du_xx + du_yy) - f

data_inv = dde.data.PDE(geom, inverse_pde, [bc, observe_bc], num_domain=1000, num_boundary=400)
model_inv = dde.Model(data_inv, dde.nn.FNN([2] + [50]*4 + [1], "tanh", "Glorot uniform"))

tracker = dde.callbacks.VariableValue([hidden_amplitude], period=100, filename="final_smooth_amplitude.txt")
model_inv.compile("adam", lr=0.01, external_trainable_variables=[hidden_amplitude])
model_inv.train(iterations=4000, callbacks=[tracker])

# THE FIX: Add L-BFGS to finish the climb
print("\n--- Phase 3: L-BFGS Fine-Tuning to reach 100.0 ---")
dde.optimizers.config.set_LBFGS_options(maxiter=5000)
model_inv.compile("L-BFGS", external_trainable_variables=[hidden_amplitude])
model_inv.train(callbacks=[tracker])

print("\nGenerating Final Graph...")
steps = []
pred_amp = []
with open("final_smooth_amplitude.txt", "r") as f:
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
plt.title("Perfect Discovery (Smooth Physics)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("Day5_Final_Success.png", dpi=300, bbox_inches="tight")
print("Success! Saved Day5_Final_Success.png")