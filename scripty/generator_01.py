import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def GeneratorCoChapem(sampleCount: int):
    # Random podle random 42
    rng = np.random.default_rng(42)
    
    # Uniformní funkce hustoty
    g = 4 / (np.pi**2)
    M = (np.pi**2)/4
    
    x = rng.uniform(0,np.pi/2,size=sampleCount)
    y = rng.uniform(0,np.pi/2,size=sampleCount)
    
    v = rng.uniform(0,1,size=sampleCount)
    
    accepted = np.where((x <= y) & (v < np.cos(y-x)), 1, 0)
    z = v * M * g
    
    return (x,y,z,accepted)

def VykresliGraf(x,y,z,accepted):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(x, y, z, c=accepted, cmap='bwr', s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    
    x = np.linspace(0, np.pi / 2, 180)
    y = np.linspace(0, np.pi / 2, 180)
    X, Y = np.meshgrid(x, y)

    Z = np.where((X > 0) & (X < Y) & (Y < np.pi / 2),
                np.cos(Y - X), np.nan)
    
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, color="black")
    
    ax.view_init(elev=15, azim=-135)
    plt.savefig("graf2.pdf")

    plt.show()

def SpocitejOdhady(x,y):
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
    print("IS_X:", IS_X_real, "vs IS_Y:", IS_Y_real)
# Úloha 4
def v(z_x, z_y):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    grid = np.linspace(0, np.pi / 2, 200)

    # (a) Hustota f_X(x) = cos(x)
    axes[0, 0].hist(z_x, bins=50, density=True, alpha=0.5, color="skyblue", edgecolor="white", label="Výběrový histogram")
    axes[0, 0].plot(grid, np.cos(grid), "r-", linewidth=2, label=r"Teorie: $f_X(x) = \cos(x)$")
    axes[0, 0].set_title("Marginální hustota $f_X(x)$")
    axes[0, 0].set_xlabel("$x$")
    axes[0, 0].set_ylabel("Hustota")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # (b) Distribuční funkce F_X(x) = sin(x)
    axes[0, 1].ecdf(z_x, label="Empirická ECDF")  # ecdf je dostupné od matplotlib >= 3.8, případně lze použít np.sort
    axes[0, 1].plot(grid, np.sin(grid), "r-", linewidth=2, label=r"Teorie: $F_X(x) = \sin(x)$")
    axes[0, 1].set_title("Marginální distribuční funkce $F_X(x)$")
    axes[0, 1].set_xlabel("$x$")
    axes[0, 1].set_ylabel("$F_X(x)$")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # (c) Hustota f_Y(y) = sin(y)
    axes[1, 0].hist(z_y, bins=50, density=True, alpha=0.5, color="lightgreen", edgecolor="white", label="Výběrový histogram")
    axes[1, 0].plot(grid, np.sin(grid), "g-", linewidth=2, label=r"Teorie: $f_Y(y) = \sin(y)$")
    axes[1, 0].set_title("Marginální hustota $f_Y(y)$")
    axes[1, 0].set_xlabel("$y$")
    axes[1, 0].set_ylabel("Hustota")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # (d) Distribuční funkce F_Y(y) = 1 - cos(y)
    axes[1, 1].ecdf(z_y, label="Empirická ECDF", color="darkgreen")
    axes[1, 1].plot(grid, 1 - np.cos(grid), "g-", linewidth=2, label=r"Teorie: $F_Y(y) = 1 - \cos(y)$")
    axes[1, 1].set_title("Marginální distribuční funkce $F_Y(y)$")
    axes[1, 1].set_xlabel("$y$")
    axes[1, 1].set_ylabel("$F_Y(y)$")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("histogramy.pdf")
    plt.show()
    
def main():
    for i in [1_000]:
        data = GeneratorCoChapem(i)
        VykresliGraf(*data)
        x,y,_,accepted = data
        SpocitejOdhady(x[accepted == 1], y[accepted == 1])
        v(x[accepted == 1], y[accepted == 1])
    
if __name__ == '__main__':
    main()

# # Inicializace generátoru jako na slajdu
# rng = np.random.default_rng(42)

# # Konstanta obálky: c_env = plocha čtverce
# c = (np.pi / 2) ** 2
# m = 100_000

# # Kandidáti z rovnoměrného 2D rozdělení na [0, pi/2] x [0, pi/2]
# x = rng.uniform(0, np.pi / 2, size=m)
# y = rng.uniform(0, np.pi / 2, size=m)

# # Pomocná náhodná čísla z U(0, 1) pro test přijetí
# v = rng.random(m)

# # Cílová hustota f(x, y) — mimo trojúhelník (x <= y) je rovna 0
# f = np.where(x <= y, np.cos(y - x), 0.0)

# # Hustota návrhového rozdělení g(x, y) na daném čtverci
# g = 1.0 / c

# # Podmínka přijetí přesně podle vzorce: v <= f / (c * g)
# prijato = v <= f / (c * g)

# # Výsledný výběr ze sdruženého rozdělení
# z_x = x[prijato]
# z_y = y[prijato]

# # Teoretická pravděpodobnost přijetí je 1 / c = 4 / pi^2 cca 0.405
# print(prijato.mean())          # ~ 1/c = 0.405
# print(f"Vygenerováno bodů: {len(z_x)}")

# # Kontrola výběrových průměrů oproti teoretickým hodnotám
# print("E(X):", np.mean(z_x), "vs teoretická:", (np.pi - 2) / 2)
# print("E(Y):", np.mean(z_y), "vs teoretická:", 1.0)


# plt.figure(figsize=(7, 7))

# # Vykreslení náhodného výběru (zobrazíme např. prvních 5 000 bodů, aby byl graf přehledný)
# plt.scatter(z_x[:5000], z_y[:5000], alpha=0.3, s=8, color="royalblue", label="Vygenerované body")

# # Ohraničení teoretické oblasti (trojúhelník 0 <= x <= y <= pi/2)
# t = np.linspace(0, np.pi / 2, 100)
# plt.plot(t, t, "r--", linewidth=1.5, label=r"Hranice $y = x$")
# plt.axhline(np.pi / 2, color="crimson", linestyle=":", linewidth=1.5, label=r"$y = \pi/2$")
# plt.axvline(0, color="gray", linestyle="-", linewidth=0.8)

# plt.title("Sdružené rozdělení $(X, Y)$")
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.xlim(-0.05, np.pi / 2 + 0.05)
# plt.ylim(-0.05, np.pi / 2 + 0.05)
# plt.gca().set_aspect("equal", adjustable="box")
# plt.legend(loc="upper left")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.show()

# # -------------------------------------------------------------
# # 2. Marginální hustoty a distribuční funkce
# # -------------------------------------------------------------

