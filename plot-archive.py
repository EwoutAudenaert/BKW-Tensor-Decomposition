##################################################################### plot for error vs dimension
dims = range(3, 16)
errors_pencil = []
errors_bkw = []

for n in dims:
    for _ in range(50):
        errors_pencil.append((n, get_algo_error('pencil', n)))
        errors_bkw.append((n, get_algo_error('bkw', n)))

errors_pencil = np.array(errors_pencil)
errors_bkw = np.array(errors_bkw)

# Plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 7))

sns.lineplot(x=errors_pencil[:, 0], y=errors_pencil[:, 1], label="pencil", marker="o")
sns.lineplot(x=errors_bkw[:, 0], y=errors_bkw[:, 1], label="bkw", marker="s")

plt.yscale('log')
plt.title("Decomposition error vs Dimension")
plt.xlabel("Dimension (n)")
plt.ylabel("Error (log-scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#####################################################################
