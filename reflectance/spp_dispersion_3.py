"""
Author: Boris de Jong
"""


"""...........................PACKAGES..........................."""

import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
from scipy.interpolate import interp1d
from numpy.linalg import eig
from scipy.signal import fftconvolve
dir = os.path.dirname(__file__)

"""...........................CONSTANTS.........................."""

c = 2.998e8  # light speed [m/s]
h = 6.62607015e-34  # planck's constant [Js]
hbar = h / (2 * np.pi)  # reduced planck's constant [Js]
e = 1.602176634e-19  # elementary charge [C]
gamma = 0.05 * e  # coupling energy [J]
gamma_eV = gamma / e # coupling energy [eV]
p = 600e-9  # pitch size of grating [m]
g = 2 * np.pi / p  # reciprocal wave vector [rad/m]
L = 50e-6 # grating size [m]
phi = np.pi   # phase shift
x_vals = np.linspace(-L/2, L/2, 2000) # spatial width of grating 
h_prime_x = -g * np.sin(g * x_vals + phi) # assume cosine form of first harmonic, so derivative -> -sin

"""..........................MCPEAK DATA........................."""

f = os.path.join(dir, "refractive_index_Ag.csv") # McPeak refractiveindex.info
d = pd.read_csv(f)
wavelengths = d['wl'].values[:49].astype(float)
n = d['n'].values[:49].astype(float)
k = d['n'].values[50:].astype(float)

re_eps = n**2 - k**2 # real part of permittivity
im_eps = 2 * k * n # imaginary part of permittivity

lamb_arr = np.linspace(0.400, 1.200, 1000)  # lambda [μm]
eps_m = np.interp(lamb_arr, wavelengths, re_eps + 1j * im_eps)  # metal permittivity interpolated for values in lamb_arr
eps_d = 1.00  # vacuum permittivity

k0 = (2 * np.pi) / (lamb_arr * 1e-6)  # angular wave number [rad/m]
omega = c * k0  # angular frequency [rad/s]

n_eff = np.sqrt((eps_d * eps_m) / (eps_d + eps_m))  # effective index
k_spp = k0 * n_eff  # SPP wave number [rad/m]
E_vals = (hbar * omega) / e  # energy in [eV]

"""...........................FUNCTIONS.........................."""

def compute_eig(k: float) -> tuple: 
    """
    Computes the eigenvalues and eigenvectors of the coupling matrices for a given wavevector k.

    The eigenvalues represent the energy levels, while the eigenvectors provide 
    information on the mode composition.

    Parameters:
        k (float): The in-plane wavevector component (in m⁻¹).

    Returns:
        tuple: 
            - eigvals (np.ndarray): Eigenvalues of the coupling matrix (energy levels in eV).
            - eigvecs (np.ndarray): Corresponding eigenvectors of the coupling matrix.
    """
    eigvals, eigvecs = eig(np.array([
        [E_of_k_right(k), gamma_eV],
        [gamma_eV, E_of_k_left(k)]
    ]))

    return eigvals, eigvecs

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generates a 1D Gaussian kernel (for convolution)

    Parameters:
        size (int): The size of the kernel (number of points).
        sigma (float): The standard deviation of the Gaussian function.

    Returns:
        np.ndarray: A 1D Gaussian kernel normalized so that its sum equals 1.
    """
    x = np.linspace(-size // 2, size // 2, size)
    gauss = np.exp(-0.5 * (x / sigma) ** 2)

    return gauss / gauss.sum()

def make_mesh(arr: np.ndarray, wavelengths: np.ndarray) -> tuple:
    """
    Create a mesh of [angle, wavelength] coordinates. The angle grid can be converted to k_x.
    Assuming NA = 0.8, n = 1.0

    Parameters:
        arr (np.ndarray): 2D array of intensity values (shape: [num_wavelengths, num_angles]).
        wavelengths (np.ndarray): Array of wavelengths (shape: [num_wavelengths]).

    Returns:
        tuple:
            - np.ndarray: points
            - np.ndarray: k_x
            - np.ndarray: wavelengths
    """

    max_angle = np.arcsin(0.8 / 1) * (180 / np.pi)  # NA = 0.8, n = 1.0 for air
    angles = np.linspace(-max_angle, max_angle, arr.shape[1])
    wavelength_grid, angle_grid = np.meshgrid(wavelengths, angles, indexing='ij')
    k_x = (2 * np.pi * np.sin(np.radians(angle_grid))) / (wavelength_grid * 1e-3) # angle -> k_x
    points =  arr.flatten()

    return points, k_x, wavelength_grid

def wavelength_to_energy(wavelength_nm: np.ndarray)-> np.ndarray:
    """Convert wavelength in nm to energy in eV."""
    wavelength_m = wavelength_nm * 1e-9
    energy_J = (h * c) / wavelength_m
    energy_eV = energy_J / e
    return energy_eV

def energy_to_wavelength(energy_eV: np.ndarray)-> np.ndarray:
    """Convert energy in eV to wavelength in nm."""
    energy_J = energy_eV * e
    wavelength_m = (h * c) / energy_J
    wavelength_nm = wavelength_m * 1e9
    return wavelength_nm

""".........................INTERPOLATION........................"""

# initialize interpolation functions E(k)
n = 1 # interpolation of E(k±ng)
E_of_k_right = interp1d(np.real(k_spp - (n * g)), np.real(E_vals), kind='linear', fill_value="extrapolate")
E_of_k_left = interp1d(np.real(-k_spp + (n * g)), np.real(E_vals), kind='linear', fill_value="extrapolate")

k_vals = np.linspace(-7e6, 7e6, 1000) 

plt.plot(k_vals, E_of_k_right(k_vals))
plt.plot(k_vals, E_of_k_left(k_vals))
plt.show(); plt.close() # plot E(k) interpolations

""".........................EIGENPROBLEM........................"""

evector_arr =  np.zeros(shape=k_vals.shape, dtype=object); eval_arr = np.zeros(shape=k_vals.shape, dtype=object)

for i, k in enumerate(k_vals):
    eval, evec = compute_eig(k)
    evector_arr[i] = evec
    eval_arr[i] = eval

eigenvectors = [[evec[:, j] for j in range(evec.shape[1])] for evec in evector_arr]
eigenvalues = [[eval[j] for j in range(eval.shape[0])] for eval in eval_arr]

for j in range(evec.shape[1]):
    real_parts = [evec[j] for evec in eigenvectors]

    plt.figure()
    plt.plot(k_vals, np.abs(real_parts), label=f'eigenvector {j+1}')
    plt.xlabel(r'$k$ [m$^{-1}$]')
    plt.ylabel('eigenvector component')
    plt.legend()
    plt.show(); plt.close() # plot eigenvectors

for l in range(eval.shape[0]):
    plt.figure()
    plt.plot(k_vals, [eval[l] for eval in eigenvalues], 'o', label=f'eigenvalue {l+1}')
    plt.xlabel(r'$k$ [m$^{-1}$]')
    plt.ylabel('eigenvalue [eV]')
    plt.legend()
    plt.show(); plt.close() # plot eigenvalues

""".........................REFLECTIVITY........................"""

count = 0
energy_vals = np.linspace(1.8, 3, 1000)
intensity_map = np.zeros((len(energy_vals), len(k_vals)))
for i, k in enumerate(k_vals):

    eigval1 = eval_arr[i][0]
    eigval2 = eval_arr[i][1]

    electric_field_1 = (evector_arr[i][0, 0]) * (np.exp(1j * (k + n * g) * x_vals)) - (evector_arr[i][0, 1]) * (np.exp(1j * (k - n * g) * x_vals))
    electric_field_2 = (evector_arr[i][1, 0]) * (np.exp(1j * (k + n * g) * x_vals)) - (evector_arr[i][1, 1]) * (np.exp(1j * (k - n * g) * x_vals))

    if count % 400 == 0:
        plt.figure()
        plt.plot(x_vals, h_prime_x * electric_field_1, label='electric field 1')
        plt.plot(x_vals, h_prime_x * electric_field_2, label='electric field 2')
        plt.xlabel(r'$x$ [m]')
        plt.ylabel('electric field')
        plt.legend()
        plt.show(); plt.close() # plot electric fields
    

    M1 = np.trapezoid(h_prime_x * electric_field_1, x_vals) # num integrate
    M2 = np.trapezoid(h_prime_x * electric_field_2, x_vals)

    E_idx1 = np.abs(energy_vals - eigval1).argmin()
    intensity_map[E_idx1, i] += np.abs(M1)**2 * p / L
    E_idx2 = np.abs(energy_vals - eigval2).argmin()
    intensity_map[E_idx2, i] += np.abs(M2)**2 * p / L

    count += 1

""".........................CONVOLUTION........................"""

intensity_map[:, [0, -1]] = 0 # remove edge intensity for convolution
intensity_map[[0, -1], :] = 0

sinc_kernel = np.sinc(k_vals * L / 2) ** 2
sinc_kernel /= np.sum(sinc_kernel)

intensity_map = np.array([
    fftconvolve(intensity_map[i], sinc_kernel, mode='same')
    for i in range(intensity_map.shape[0])
])
kernel = gaussian_kernel(50, 5)[:, None] * gaussian_kernel(50, 5)
intensity_map = fftconvolve(intensity_map, kernel, mode="same")
intensity_map /= np.max(intensity_map)

""".........................PLOTTING........................"""

plt.plot(k_vals, np.sum(intensity_map, axis=0))
plt.show(); plt.close()

plt.imshow(intensity_map, extent=[k_vals[0], k_vals[-1], energy_vals[0], energy_vals[-1]], 
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label="intensity")
plt.show()

""".........................PLOTTING........................"""

energy_min, energy_max = energy_to_wavelength(np.array([700, 400]))
E_vals = np.linspace(energy_min, energy_max, 1000)  # energy [eV]
wavelengths = energy_to_wavelength(E_vals)  # eV -> [nm]
intensity_flat, k_x, wavelength_grid = make_mesh(intensity_map, wavelengths)
energy_grid = wavelength_to_energy(wavelength_grid)  # convert nm to [eV] 

plt.figure(figsize=(4, 6))
ax1 = plt.gca()
pcm = ax1.pcolormesh(k_x, energy_grid, intensity_map, cmap='inferno', shading='auto')

plt.colorbar(pcm, ax=ax1, label='intensity (norm.)', location='top')
ax1.set_xlabel(r'$k_x$ (rad/$\mu$m)')
ax1.set_ylabel(r'Energy (eV)')

ax2 = ax1.twinx()
ymin, ymax = ax1.get_ylim()
ax2.set_ylim(ymin, ymax)

custom_wavelength_ticks = np.array([400, 500, 600, 700])
custom_energy_ticks = wavelength_to_energy(custom_wavelength_ticks)

ax2.set_yticks(custom_energy_ticks)
ax2.set_yticklabels([f'{w}' for w in custom_wavelength_ticks])
ax2.set_ylabel('wavelength (nm)')

plt.tight_layout()
plt.show()

""".........................INTERPOLATION........................"""