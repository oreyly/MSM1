import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Stejná simulace jako v původním zadání
rng = np.random.default_rng(42)
m = 50_000
M = (2 / 2 + 3 / 3) / 6

x = rng.uniform(0, 2, m)
y = rng.uniform(0, 3, m)
w = rng.uniform(0, M, m)

f = lambda x, y: (x / 2 + y / 3) / 6
ok = w <= f(x, y)

# Kvůli čitelnosti zobrazíme jen část bodů, statistiky počítáme ze všech.
ids = rng.choice(m, size=5500, replace=False)
accepted = ids[ok[ids]]
rejected = ids[~ok[ids]]

fig = plt.figure(figsize=(14, 6.5))
fig.suptitle("Zamítací metoda pro náhodný vektor", fontsize=17, y=0.98)

# Levá část: 3D zobrazení (x, y, w)
ax = fig.add_subplot(1, 2, 1, projection="3d")

xx, yy = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 3, 21))
ax.plot_surface(xx, yy, f(xx, yy), color="gray", alpha=0.23,
                linewidth=0, shade=False)

ax.scatter(x[rejected], y[rejected], w[rejected],
           c="red", s=5, alpha=0.52, depthshade=False)
ax.scatter(x[accepted], y[accepted], w[accepted],
           c="tab:blue", s=5, alpha=0.65, depthshade=False)

ax.set(xlabel="x", ylabel="y", zlabel="w",
       xlim=(0, 2), ylim=(0, 3), zlim=(0, M))
ax.view_init(elev=23, azim=-60)
ax.set_box_aspect((2, 3, 1.7))
ax.set_title(f"Body v kvádru $A\\times(0,M)$\nPřijato {ok.mean():.1%}", pad=17)
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
           markersize=7, label="Přijato"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
           markersize=7, label="Zamítnuto"),
], loc="upper left")

# Pravá část: průmět přijatých bodů do roviny (x, y)
ax2 = fig.add_subplot(1, 2, 2)
cont = ax2.contourf(xx, yy, f(xx, yy), levels=np.linspace(0, M, 14),
                    cmap="Greys", vmin=0, vmax=M, alpha=0.47)
ax2.scatter(x[accepted], y[accepted], s=5, c="tab:blue", alpha=0.75)
ax2.set(xlabel="x", ylabel="y", xlim=(0, 2), ylim=(0, 3))
ax2.set_title("Přijaté body $(x_i, y_i)$", pad=15)
ax2.set_aspect("equal")
cbar = fig.colorbar(cont, ax=ax2, shrink=0.75, pad=0.04)
cbar.set_label("$f(x, y)$")

fig.subplots_adjust(left=0.02, right=0.96, top=0.86, bottom=0.07, wspace=0.09)
plt.show()

print(f"Přijato: {ok.mean():.2%}")
print(f"Průměry přijatých: E[X] ≈ {x[ok].mean():.4f}, E[Y] ≈ {y[ok].mean():.4f}")
print(f"Kovariance přijatých: {np.cov(x[ok], y[ok])[0, 1]:.4f}")
