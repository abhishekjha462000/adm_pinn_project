import matplotlib
matplotlib.use("Agg")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

dde.config.set_default_float("float32")
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

def smooth_sink_source(x):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    # FIX: We widened sigma from 0.1 to 0.4 to remove the sharp singularity
    return -100.0 * torch.exp(-(x_coord**2 + y_coord**2) / (2 * 0.4**2))

def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return - (du_xx + du_yy) - smooth_sink_source(x)

def boundary(_, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

point_densities = [200, 500, 1000, 2000, 4000]
final_errors = []

print("Starting Smooth Spatial Convergence Study...")

for N in point_densities:
    print(f"\n--- Training with {N} spatial domain points ---")
    N_boundary = int(N * 0.2) 
    
    data = dde.data.PDE(geom, pde, bc, num_domain=N, num_boundary=N_boundary)
    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    
    # Using L-BFGS directly for highly precise, smooth convergence
    dde.optimizers.config.set_LBFGS_options(maxiter=3000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    
    final_pde_loss = losshistory.loss_train[-1][0]
    final_errors.append(final_pde_loss)
    print(f"Final Smooth PDE Error for {N} points: {final_pde_loss:.6f}")

plt.figure(figsize=(8, 6))
plt.loglog(point_densities, final_errors, marker='o', color='blue', linewidth=2, markersize=8)
plt.xlabel("Number of Spatial Collocation Points ($N_{domain}$)", fontsize=12)
plt.ylabel("PDE Residual Error ($L_2$ Loss)", fontsize=12)
plt.title("Smooth PINN Spatial Convergence", fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.7)

plt.savefig("Day4_Smooth_Convergence.png", dpi=300, bbox_inches="tight")
print("\nSuccess! Saved Day4_Smooth_Convergence.png")