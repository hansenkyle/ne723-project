import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



names = ["cp2_test_a", "cp2_test_b", "cp2_test_c", "cp2_test_d1", "cp2_test_d2"]
nice_names = ["Test A", "Test B", "Test C", "Test D(a)", "Test D(b)"]

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['lines.marker'] = '.'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.grid'] = True

for i in range(5):
    OUT_DIR = Path(__file__).resolve().parent / names[i]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file = h5py.File(f"{names[i]}.h5", "r")
    x_c = file["grid/cell-centers"][:]
    x_b = file["grid/cell-edges"][:]
    dx = file["grid/dx"][:]
    mu = file["quadrature/mu"][:]

    psi_c = file["solution/angular_flux/cell-center"][:]
    phi_avg = file["solution/scalar-flux/cell-average"][:]
    phi_b = file["solution/scalar-flux/cell-edge"][:]
    j_b = file["solution/current/cell-edge"][:]

    c_l = file["closures/boundary-factor/left"][()]
    c_r = file["closures/boundary-factor/right"][()]
    E_c = file["closures/qd-factor/cell-center"][:]
    E_b = file["closures/qd-factor/cell-edge"][:]

    M_E = np.abs((1.0/3) - E_c)
    M_Ep = np.abs((E_b[1:] - E_b[:-1])/dx)

    err = file["convergence/error"][:]
    rho = file["convergence/spectral-radius"][:]
    num_iter = file["convergence/number-iterations"][()]

    it = np.array(np.linspace(0, num_iter, num_iter+1), dtype=int)

    print(f"{nice_names[i]} converged in {num_iter} iterations, with $C_L = {c_l:.6f}$ and $C_R = {c_r:.6f}$.")

    # cell-center angular flux
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, M = np.meshgrid(x_c, mu)
    surf = ax.plot_surface(X, M, psi_c, cmap ="plasma", linewidth=0, antialiased=False)
    fig.colorbar(surf, pad=0.15)
    plt.title(f"Cell-average angular flux, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$\\mu$")
    plt.savefig(f"{names[i]}/ave_angular.png")


    # cell-average scalar flux
    plt.figure()
    plt.plot(x_c, phi_avg)
    plt.title(f"Cell-average scalar flux, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$\\phi_i$")
    plt.savefig(f"{names[i]}/ave_scalar.png")

    # cell-edge scalar flux
    plt.figure()
    plt.plot(x_b, phi_b)
    plt.title(f"Cell-edge scalar flux, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$\\phi_{i+1/2}$")
    plt.savefig(f"{names[i]}/edge_scalar.png")
    
    # cell-edge current
    plt.figure()
    plt.plot(x_b, j_b)
    plt.title(f"Cell-edge current, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$J_{i+1/2}$")
    plt.savefig(f"{names[i]}/edge_current.png")

    # cell-average E
    plt.figure()
    plt.axhline(y=(1.0/3), linestyle='-', linewidth=0.7, marker="", color="black")
    plt.plot(x_c, E_c)
    plt.title(f"Cell-center QD factor, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$E_i$")
    plt.savefig(f"{names[i]}/ave_qdfactor.png")

    # cell-edge E
    plt.figure()
    plt.axhline(y=(1.0/3), linestyle='-', linewidth=0.7, marker="", color="black")
    plt.plot(x_b, E_b)
    plt.title(f"Cell-edge QD factor, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$E_{i+1/2}$")
    plt.savefig(f"{names[i]}/edge_qdfactor.png")

    # M^E
    plt.figure()
    plt.plot(x_c, M_E)
    plt.title(f"QD Factor Transport Metric, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$M_i^E$")
    plt.savefig(f"{names[i]}/val_metric.png")

    # M^Eprime
    plt.figure()
    plt.plot(x_c, M_Ep)
    plt.title(f"$\\partial_x E$ Transport Metric, {nice_names[i]}")
    plt.xlabel("$x$")
    plt.ylabel("$M_i^{E^\\prime}$")
    plt.savefig(f"{names[i]}/dv_metric.png")

    # Error
    plt.figure()
    plt.plot(it[2:], err[2:])
    plt.title(f"Iterative $L_\\infty$ error, {nice_names[i]}")
    plt.xlabel("$s$")
    plt.ylabel("$|\\phi_i^{(s)} - \\phi_i^{(s-1)}|_\\infty$")
    plt.yscale("log")
    plt.xlim([-0.1, None])
    plt.savefig(f"{names[i]}/error.png")

    # Spectral radius
    plt.figure()
    plt.plot(it, rho)
    plt.title(f"Estimated spectral radius, {nice_names[i]}")
    plt.xlabel("$s$")
    plt.ylabel("$\\rho^{(s)}$")
    plt.savefig(f"{names[i]}/radius.png")

    plt.close('all')



