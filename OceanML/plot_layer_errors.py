import matplotlib.pyplot as plt

# Layer numbers
layers = list(range(1, 16))

# MSE values from your layer analysis
mse = [
    0.048502,
    0.041845,
    0.045280,
    0.053170,
    0.062514,
    0.074783,
    0.083476,
    0.088347,
    0.088347,
    0.092520,
    0.092347,
    0.090549,
    0.100231,
    0.100436,
    0.114997
]

# MAE values from your layer analysis
mae = [
    0.141316,
    0.121368,
    0.124506,
    0.108626,
    0.132757,
    0.132175,
    0.146098,
    0.155167,
    0.155167,
    0.161446,
    0.161578,
    0.155721,
    0.165400,
    0.172249,
    0.177456
]

# -------------------------
# MSE graph
# -------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    layers,
    mse,
    marker="o"
)

plt.xlabel("Subsurface Layer")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE Across Subsurface Layers")

plt.xticks(layers)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "layer_mse.png",
    dpi=300
)

plt.show()


# -------------------------
# MAE graph
# -------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    layers,
    mae,
    marker="o"
)

plt.xlabel("Subsurface Layer")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE Across Subsurface Layers")

plt.xticks(layers)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "layer_mae.png",
    dpi=300
)

plt.show()

print("\nLayer error graphs created successfully!")
print("Saved: layer_mse.png")
print("Saved: layer_mae.png")