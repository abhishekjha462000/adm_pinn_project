import matplotlib
matplotlib.use("Agg")

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

dde.config.set_default_float("float32")
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

def sink_source(x):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    # Using the true Day 1 parameters for the forward convergence study
    return -100.0 * torch.exp(-(x_coord**2 + y_coord**2) / (2 * 0.1**2))

def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return - (du_xx + du_yy) - sink_source(x)

def boundary(_, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

# The core of Day 4: Systematically increasing spatial points
point_densities = [200, 500, 1000, 2000, 4000]
final_errors = []

print("Starting Day 4 Spatial Convergence Study...")
print("Training 5 separate networks. This may take 5-10 minutes.")

for N in point_densities:
    print(f"\n--- Training with {N} spatial domain points ---")
    
    # Increase boundary points proportionally
    N_boundary = int(N * 0.2) 
    
    data = dde.data.PDE(
        geom, pde, bc, num_domain=N, num_boundary=N_boundary
    )
    
    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    
    # We use a quick 3000 iteration training loop for each point density
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=3000)
    
    # Extract the final PDE residual loss as our measure of spatial error
    final_pde_loss = losshistory.loss_train[-1][0]
    final_errors.append(final_pde_loss)
    print(f"Final PDE Error for {N} points: {final_pde_loss:.6f}")

# Generate the Convergence Plot
plt.figure(figsize=(8, 6))
# Log-Log plot is the standard for mathematical convergence studies
plt.loglog(point_densities, final_errors, marker='o', color='blue', linewidth=2, markersize=8)

plt.xlabel("Number of Spatial Collocation Points ($N_{domain}$)", fontsize=12)
plt.ylabel("PDE Residual Error ($L_2$ Loss)", fontsize=12)
plt.title("PINN Spatial Convergence Study", fontsize=14)
plt.grid(True, which="both", linestyle='--', alpha=0.7)

# Save the final deliverable
plt.savefig("Day4_Spatial_Convergence.png", dpi=300, bbox_inches="tight")
print("\nSuccess! Saved Day4_Spatial_Convergence.png")