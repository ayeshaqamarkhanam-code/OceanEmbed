import matplotlib.pyplot as plt

layers = list(range(1, 16))

r2 = [
    0.951171,
    0.957849,
    0.954381,
    0.946407,
    0.937017,
    0.925113,
    0.917540,
    0.917910,
    0.913161,
    0.908881,
    0.908823,
    0.910085,
    0.900324,
    0.900052,
    0.884487
]

plt.figure(figsize=(10, 6))

plt.plot(layers, r2, marker="o")

plt.title("R² Across Subsurface Layers")
plt.xlabel("Subsurface Layer")
plt.ylabel("R²")
plt.xticks(layers)
plt.grid(True)

plt.tight_layout()

plt.savefig("layer_r2.png", dpi=150)

print("R² plot saved as: layer_r2.png")

plt.show()