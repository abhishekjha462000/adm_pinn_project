import matplotlib.pyplot as plt

print("Parsing discovered features and generating figures...")

steps = []
pred_amp = []
pred_sig = []

# Read the file and strip out the brackets and commas
with open("discovered_features.txt", "r") as f:
    for line in f:
        clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
        parts = clean_line.split()
        if len(parts) >= 3:
            steps.append(float(parts[0]))
            pred_amp.append(float(parts[1]))
            pred_sig.append(float(parts[2]))

plt.figure(figsize=(10, 4))

# Plot Amplitude Discovery
plt.subplot(1, 2, 1)
plt.plot(steps, pred_amp, label="Predicted Amplitude", color="blue")
plt.axhline(100.0, color='red', linestyle='--', label="True Value (100.0)")
plt.xlabel("Iterations")
plt.ylabel("Amplitude")
plt.title("Discovering Hidden Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Plot Sigma Discovery
plt.subplot(1, 2, 2)
plt.plot(steps, pred_sig, label="Predicted Sigma", color="green")
plt.axhline(0.1, color='red', linestyle='--', label="True Value (0.1)")
plt.xlabel("Iterations")
plt.ylabel("Sigma")
plt.title("Discovering Hidden Sigma")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("Day2_Feature_Discovery.png", dpi=300, bbox_inches="tight")

print("Success! Saved Day2_Feature_Discovery.png")