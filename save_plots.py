import numpy as np
import matplotlib.pyplot as plt

print("Generating high-resolution Day 1 figures...")

# 1. Plot and save the Loss History
loss_data = np.loadtxt("loss.dat")
steps = loss_data[:, 0]
train_loss = loss_data[:, 1]
test_loss = loss_data[:, 2]

plt.figure(figsize=(8, 5))
plt.yscale('log')
plt.plot(steps, train_loss, label="Train Loss", color="blue")
plt.plot(steps, test_loss, label="Test Loss", color="orange", linestyle="--")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error Loss")
plt.title("Day 1 PINN Convergence History")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("Day1_Loss_History.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Plot and save the 2D Prediction Map
test_data = np.loadtxt("test.dat")
x = test_data[:, 0]
y = test_data[:, 1]
u = test_data[:, 2]

plt.figure(figsize=(7, 6))
# Using tricontourf to cleanly map the 2D field
contour = plt.tricontourf(x, y, u, levels=50, cmap="magma")
plt.colorbar(contour, label="u(x,y) Concentration")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted Diffusion Field (Day 1)")
plt.savefig("Day1_Prediction_Map.png", dpi=300, bbox_inches="tight")
plt.close()

print("Success! Saved Day1_Loss_History.png and Day1_Prediction_Map.png")