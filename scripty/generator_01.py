import numpy as np
import matplotlib.pyplot as plt

# 4. generovaní - zamítací metoda
def Generator(sampleCount: int):
    rng = np.random.default_rng(42)
    
    # Obálka M.g(x)
    g = 4 / (np.pi**2)
    M = (np.pi**2)/4
    
    x = rng.uniform(0,np.pi/2,size=sampleCount)
    y = rng.uniform(0,np.pi/2,size=sampleCount)
    
    v = rng.uniform(0,1,size=sampleCount)
    
    accepted = np.where((x <= y) & (v < np.cos(y-x)), 1, 0)
    z = v * M * g
    
    return (x,y,z,accepted)

# 4.1. vykreslení generovaných bodů 
def VykresliGraf(x,y,z,accepted):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=accepted, cmap='bwr', s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    
    x = np.linspace(0, np.pi / 2, 180)
    y = np.linspace(0, np.pi / 2, 180)
    X, Y = np.meshgrid(x, y)

    Z = np.where((X > 0) & (X < Y) & (Y < np.pi / 2),
                np.cos(Y - X), np.nan)
    
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, color="black")
    
    ax.view_init(elev=15, azim=-135)
    plt.savefig("graf2.pdf")

    plt.show()

# 5. odhady z vygenerovaných dat
def SpocitejOdhady(x,y):
    print("\n" + "=" * 94)
    print("5. Odhady z generovaných dat")
    print("=" * 94)


    EX_teror = (np.pi - 2) / 2
    EY_teror = 1.0
    
    VarX_teror = np.pi - 3
    VarY_teror = np.pi - 3
    
    Cov_teror = (np.pi**2)/8 - np.pi + 2
    
    EX_real = np.mean(x)
    EY_real = np.mean(y)
    
    VarX_real = np.var(x)
    VarY_real = np.var(y)
    
    Cov_mat_real = np.cov(x,y)
    Cov_real = Cov_mat_real[0,1]
    
    Sx_real = np.std(x, ddof=1)
    Sy_real = np.std(y, ddof=1)
    
    z0975 = 1.96
    z = z0975
    
    IS_X_real = (
    np.mean(EX_real) - z * Sx_real / np.sqrt(len(x)),
    np.mean(EX_real) + z * Sx_real / np.sqrt(len(x))
    )
    
    IS_Y_real = (
    np.mean(EY_real) - z * Sy_real / np.sqrt(len(y)),
    np.mean(EY_real) + z * Sy_real / np.sqrt(len(y))
    )
    
    print("E(X):", EX_real, "vs teoretická:", EX_teror, "Rozdíl: ", abs(EX_teror - EX_real))
    print("E(Y):", EY_real, "vs teoretická:", EY_teror, "Rozdíl: ", abs(EY_teror - EY_real))
    print()
    print("VarX:", VarX_real, "vs teoretická:", VarX_teror, "Rozdíl: ", abs(VarX_teror - VarX_real))
    print("VarY:", VarY_real, "vs teoretická:", VarY_teror, "Rozdíl: ", abs(VarY_teror - VarY_real))
    print()
    print("Cov:", Cov_real, "vs teoretická:", Cov_teror, "Rozdíl: ", abs(Cov_real - Cov_teror))
    print()
    print(f"IS_X: [{IS_X_real[0]:.4f}; {IS_X_real[1]:.4f}] \t IS_Y: [{IS_Y_real[0]:.4f}; {IS_Y_real[1]:.4f}]")
    
# 6. Porovnání s teoretickými funkcemi
def Histogramy(z_x, z_y):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    grid = np.linspace(0, np.pi / 2, 200)

    # (a) Hustota f_X(x) = cos(x)
    axes[0, 0].hist(z_x, bins=50, density=True, alpha=0.5, color="skyblue", edgecolor="white", label="Výběrový histogram")
    axes[0, 0].plot(grid, np.cos(grid), "r-", linewidth=2, label=r"Teorie: $f_X(x) = \cos(x)$")
    axes[0, 0].set_title("Marginální hustota $f_X(x)$")
    axes[0, 0].set_xlabel("$x$")
    axes[0, 0].set_ylabel("Hustota")
    
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # (b) Distribuční funkce F_X(x) = sin(x)
    axes[0, 1].ecdf(z_x, label="Empirická ECDF")  # ecdf je dostupné od matplotlib >= 3.8, případně lze použít np.sort
    axes[0, 1].plot(grid, np.sin(grid), "r-", linewidth=2, label=r"Teorie: $F_X(x) = \sin(x)$")
    axes[0, 1].set_title("Marginální distribuční funkce $F_X(x)$")
    axes[0, 1].set_xlabel("$x$")
    axes[0, 1].set_ylabel("$F_X(x)$")
    
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # (c) Hustota f_Y(y) = sin(y)
    axes[1, 0].hist(z_y, bins=50, density=True, alpha=0.5, color="lightgreen", edgecolor="white", label="Výběrový histogram")
    axes[1, 0].plot(grid, np.sin(grid), "g-", linewidth=2, label=r"Teorie: $f_Y(y) = \sin(y)$")
    axes[1, 0].set_title("Marginální hustota $f_Y(y)$")
    axes[1, 0].set_xlabel("$y$")
    axes[1, 0].set_ylabel("Hustota")
    
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # (d) Distribuční funkce F_Y(y) = 1 - cos(y)
    axes[1, 1].ecdf(z_y, label="Empirická ECDF", color="darkgreen")
    axes[1, 1].plot(grid, 1 - np.cos(grid), "g-", linewidth=2, label=r"Teorie: $F_Y(y) = 1 - \cos(y)$")
    axes[1, 1].set_title("Marginální distribuční funkce $F_Y(y)$")
    axes[1, 1].set_xlabel("$y$")
    axes[1, 1].set_ylabel("$F_Y(y)$")
    
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("histogramy.pdf")
    plt.show()

# 1. Vykreslení původní funkce f(x,y) = cos(y - x)
def VykresleniFunkce():
    grid_res = 200
    x_lin = np.linspace(0, np.pi / 2, grid_res)
    y_lin = np.linspace(0, np.pi / 2, grid_res)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.where((X >= 0) & (X <= Y) & (Y <= np.pi / 2), np.cos(Y - X), np.nan)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9,
        antialiased=True,
        rstride=1,
        cstride=1
    )
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', levels=8, linewidths=1.2)
    diag = np.linspace(0, np.pi / 2, 100)
    ax.plot(diag, diag, np.ones_like(diag), color='crimson', linewidth=2, label=r'Maximum: $y = x$ ($f=1$)')
    ax.set_xlabel('$x$', fontsize=11, labelpad=8)
    ax.set_ylabel('$y$', fontsize=11, labelpad=8)
    ax.set_zlabel('$f(x, y)$', fontsize=11, labelpad=8)
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=28, azim=-55)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label(r'$f(x, y) = \cos(y - x)$', rotation=270, labelpad=15)
    ax.legend(loc='upper left')
    plt.title(r'Sdružená hustota pravděpodobnosti na $\Omega$', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig('graf_funkce.pdf', bbox_inches='tight')
    plt.show()

# 7. Závislost přesnosti pro n = [100, 1000, 10000, 100000, 1000000]
def ZavislostOdhadu(x_prijate, y_prijate):
    ex_teor = (np.pi - 2) / 2
    ey_teor = 1.0
    z = 1.96

    print("\n" + "=" * 94)
    print("7. Závislost odhadu na rozsahu výběru")
    print("=" * 94)
    hlavicka = (
        f"{'n':>7} | {'x_bar':>7} | {'|EX-x|':>8} | {'Sirka IS_X':>10} | "
        f"{'y_bar':>7} | {'|EY-y|':>8} | {'Sirka IS_Y':>10}"
    )
    print(hlavicka)
    print("-" * len(hlavicka))

    for n in [100, 1000, 10000, 100000, 1000000]:
        x_sub = x_prijate[:n]
        y_sub = y_prijate[:n]

        # Výběrové průměry
        x_bar = np.mean(x_sub)
        y_bar = np.mean(y_sub)

        # Výběrové směrodatné odchylky (výběrový odhad s ddof=1)
        sx = np.std(x_sub, ddof=1)
        sy = np.std(y_sub, ddof=1)

        # Šířky 95% intervalů spolehlivosti (horní mez - dolní mez)
        sirka_is_x = 2 * z * sx / np.sqrt(n)
        sirka_is_y = 2 * z * sy / np.sqrt(n)

        # Absolutní odchylky od teorie
        err_x = abs(ex_teor - x_bar)
        err_y = abs(ey_teor - y_bar)

        print(
            f"{n:7d} | {x_bar:7.4f} | {err_x:8.4f} | {sirka_is_x:10.4f} | "
            f"{y_bar:7.4f} | {err_y:8.4f} | {sirka_is_y:10.4f}"
        )
        
def main():
    VykresleniFunkce()
    # generovaní 3_000_000 pro následné porovnání v bodě 7
    data = Generator(3_000_000) 

    x,y,z,accepted = data

    # vykreslení přijatých bodů (omezeno na 5000)
    VykresliGraf(x[:5000], y[:5000], z[:5000], accepted[:5000]) 
    # odhady z přijatých dat
    SpocitejOdhady(x[accepted == 1], y[accepted == 1])
    # vykreslení histogramů
    Histogramy(x[accepted == 1], y[accepted == 1])

    ZavislostOdhadu(x[accepted == 1], y[accepted == 1])
    
if __name__ == '__main__':
    main()

