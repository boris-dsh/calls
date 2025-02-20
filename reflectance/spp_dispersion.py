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

c = 3e8  # light speed [m/s]
h = 6.62607015e-34  # planck's constant [Js]
hbar = h / (2 * np.pi)  # reduced planck's constant [Js]
e = 1.602176634e-19  # elementary charge [C]
gamma = 0.05 * e  # coupling energy [J]
gamma_eV = gamma / e # coupling energy [eV]
p = 600e-9  # pitch size of grating [m]
g = 2 * np.pi / p  # reciprocal wave vector [1/m]
L = 10e-6 # grating size [m]
phi = np.pi / 1  # phase shift
x_vals = np.linspace(-p/2, p/2, 1000) 

"""..........................MCPEAK DATA........................."""

f = os.path.join(dir, "refractive_index_Ag.csv") # McPeak refractiveindex.info
d = pd.read_csv(f)
wavelengths = d['wl'].values[:49].astype(float)
n = d['n'].values[:49].astype(float)
k = d['n'].values[50:].astype(float)

re_eps = n**2 - k**2 # real part of permittivity
im_eps = 2 * k * n # imaginary part of permittivity

lamb_arr = np.linspace(0.400, 1.200, 1000)  # lambda [μm]
eps_m = np.interp(lamb_arr, wavelengths, re_eps + 1j * im_eps)  # metal permittivity
eps_d = 1.00  # vacuum permittivity

k0 = (2 * np.pi) / (lamb_arr * 1e-6)  # angular wave number [rad/m]
omega = c * k0  # angular frequency [rad/s]

n_eff = np.sqrt((eps_d * eps_m) / (eps_d + eps_m))  # effective index
k_spp = k0 * n_eff  # SPP wave number [rad/m]
E_vals = (hbar * omega) / e  # energy in [eV]

"""...........................FUNCTIONS.........................."""
# initialize interpolation functions E(k)
E_of_k_1g = interp1d(np.real(k_spp-g), np.real(E_vals), kind='linear', fill_value="extrapolate") # interpolate energy values at k_spp-g

def energy_at_k_1g(k: float) -> np.ndarray:
    """
    Computes the eigenvalues of the coupling matrix for a given wavevector k.
    
    Parameters:
        k (float): The in-plane wavevector component (in 1/m).

    Returns:
        np.ndarray: A 1D array containing the two eigenvalues (energy levels in eV).
    
    """

    w, _ = eig(np.array([
        [E_of_k_1g(k+g), gamma_eV],
        [gamma_eV, E_of_k_1g(k-g)]]
    ))

    return w

def compute_eig(k: float) -> tuple: 
    """
    Computes the eigenvalues and eigenvectors of the coupling matrices for a given wavevector k.

    The eigenvalues represent the energy levels, while the eigenvectors provide 
    information on the mode composition.

    Parameters:
        k (float): The in-plane wavevector component (in m⁻¹).

    Returns:
        tuple: 
            - eigvals1 (np.ndarray): Eigenvalues of the first coupling matrix (energy levels in eV).
            - eigvecs1 (np.ndarray): Corresponding eigenvectors of the first coupling matrix.
            - eigvals2 (np.ndarray): Eigenvalues of the second coupling matrix (energy levels in eV).
            - eigvecs2 (np.ndarray): Corresponding eigenvectors of the second coupling matrix.
    """
    eigvals1, eigvecs1 = eig(np.array([
        [E_of_k_1g(k), gamma_eV],
        [gamma_eV, E_of_k_1g(-k)]
    ]))

    eigvals2, eigvecs2 = eig(np.array([
        [E_of_k_1g(-k), gamma_eV],
        [gamma_eV, E_of_k_1g(k)]
    ]))

    return eigvals1, eigvecs1, eigvals2, eigvecs2

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
    
"""...........................PLOTJE.........................."""

# plt.figure()

# plt.plot(k_spp - g, E_vals, c='C0') # right-traveling photon-SPP coupling line from first harmonic
# plt.plot(-k_spp + g, E_vals, c='C0') # left-traveling photon-SPP coupling line from first harmonic
# plt.plot(k_spp, E_vals, c='C1') # right-traveling SPP line outside light line
# plt.plot(-k_spp, E_vals, c='C1') # left-traveling SPP line outside light line
# plt.plot(k0, E_vals, 'C2:') # light line positive angle
# plt.plot(-k0, E_vals, 'C2:') # light line negative angle

# plt.ylim(1.8, 3)
# plt.ylabel(r'$E$ (eV)')
# plt.xlabel(r'$k_x$ (m$^{-1}$)')

# plt.show()

"""...........................PLOTJE.........................."""

# k_val = np.linspace(-7e6, 7e6, 1000) # k value in [1/m]

# k_list_1 = [[], []]
# k_list_2 = [[], []]

# for k in k_val:
#     k_list_1[0].append(energy_at_k_1g(k)[0])
#     k_list_1[1].append(E_of_k_2g(k))
#     k_list_2[0].append(energy_at_k_1g(k)[1])
#     k_list_2[1].append(E_of_k_2g(-k))

# plt.figure(figsize=(4,6))
# plt.plot(k_val, k_list_1[0], 'C0.')
# plt.plot(k_val, k_list_2[0], 'C1.')
# plt.plot(k_val, k_list_1[1], 'C3.')
# plt.plot(k_val, k_list_2[1], 'C4.')
# plt.ylim(1.8, 3)
# plt.show()

""".........................GRIDPLOTJE........................"""

k_vals = np.linspace(0, 7e6, 1000) 
E_vals = np.linspace(1.5, 3.3, 1000)

intensity_map = np.zeros((len(E_vals), len(k_vals)))

for i, k in enumerate(k_vals):
    eval1, evec1, eval2, evec2 = compute_eig(k)

    eigval1 = eval1[0]
    weight = np.abs(evec1[0, 0])**2
    E_idx = np.abs(E_vals - eigval1).argmin()
    intensity_map[E_idx, i] += weight  

    eigval2 = eval2[0]
    weight = np.abs(evec2[0, 0])**2
    E_idx = np.abs(E_vals - eigval2).argmin()
    intensity_map[E_idx, i] += weight

    # E1_x = evec1[0, 0] * np.exp(-1j * (k - g) * x_vals) + evec1[1, 0] * np.exp(+1j * (k - g) * x_vals)
    # E2_x = evec2[0, 0] * np.exp(-1j * (k - g) * x_vals) + evec2[1, 0] * np.exp(+1j * (k - g) * x_vals)

    # h_prime_x = -g * np.sin(g * x_vals + phi) # assume cosine form of first harmonic, so derivative -> -sin

    # M1 = np.trapezoid(h_prime_x * E1_x, x_vals) # num integrate
    # M2 = np.trapezoid(h_prime_x * E2_x, x_vals)

    # E_idx1 = np.abs(E_vals - eval1[0]).argmin()
    # intensity_map[E_idx1, i] += np.abs(M1)**2

    # E_idx2 = np.abs(E_vals - eval2[0]).argmin()
    # intensity_map[E_idx2, i] += np.abs(M2)**2

intensity_map[:, [0, -1]] = 0 # remove edge intensity for convolution
intensity_map[[0, -1], :] = 0

plt.imshow(intensity_map, extent=[k_vals[0], k_vals[-1], E_vals[0], E_vals[-1]], 
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label="intensity")
plt.show()

# sinc_kernel = np.sinc(k_vals * L / 2) ** 2
# sinc_kernel /= np.sum(sinc_kernel)

# intensity_map = np.array([
#     fftconvolve(intensity_map[i], sinc_kernel, mode='same')
#     for i in range(intensity_map.shape[0])
# ])

# intensity_map /= np.max(intensity_map)

# kernel = gaussian_kernel(50, 5)[:, None] * gaussian_kernel(50, 5)
# intensity_map = fftconvolve(intensity_map, kernel, mode="same")

# intensity_map /= np.max(intensity_map)

# plt.figure(figsize=(4, 6))
# plt.imshow(intensity_map, extent=[k_vals[0], k_vals[-1], E_vals[0], E_vals[-1]], 
#            origin='lower', aspect='auto', cmap='inferno')
# ax1 = plt.gca()
# plt.ylim(1.8, 3.0)
# plt.xlabel(r'$k_x$ (m$^{-1}$)')
# plt.ylabel(r'energy (eV)')
# plt.colorbar(ax=ax1, label='intensity (norm.)', location='top')
# plt.show()

""".........................COLORMESH........................"""

k_vals = np.linspace(-7e6, 7e6, 1000) # [1/m]
energy_min, energy_max = energy_to_wavelength(np.array([700, 400]))
E_vals = np.linspace(energy_min, energy_max, 1000)  # energy [eV]
wavelengths = energy_to_wavelength(E_vals)  # eV -> [nm]

intensity_map = np.zeros((len(E_vals), len(k_vals)))

for i, k in enumerate(k_vals):
    eval1, evec1, eval2, evec2 = compute_eig(k) # compute eigenvalues and -vectors

    # eigval1 = eval1[0]
    # weight = np.abs(evec1[0, 0])**2 # weight based on SPP character (first element in eigenvector)
    # E_idx = np.abs(E_vals - eigval1).argmin()
    # intensity_map[E_idx, i] += weight

    # eigval2 = eval2[0]
    # weight = np.abs(evec2[0, 0])**2
    # E_idx = np.abs(E_vals - eigval2).argmin()
    # intensity_map[E_idx, i] += weight

    E1_x = evec1[0, 0] * np.exp(1j * (k - g) * x_vals) + evec1[1, 0] * np.exp(-1j * (k - g) * x_vals)
    E2_x = evec2[0, 0] * np.exp(1j * (k - g) * x_vals) + evec2[1, 0] * np.exp(-1j * (k - g) * x_vals)

    h_prime_x = - g * np.sin(g * x_vals + phi) # assume cosine form of first harmonic, so derivative -> -sin
    M1 = np.trapezoid(h_prime_x * E1_x, x_vals) # num integrate
    M2 = np.trapezoid(h_prime_x * E2_x, x_vals)

    E_idx1 = np.abs(E_vals - eval1[0]).argmin()
    intensity_map[E_idx1, i] += np.abs(M1)**2

    E_idx2 = np.abs(E_vals - eval2[0]).argmin()
    intensity_map[E_idx2, i] += np.abs(M2)**2

intensity_map[:, [0, -1]] = 0 # remove edge intensity for convolution
intensity_map[[0, -1], :] = 0

sinc_kernel = np.sinc(k_vals * L / 2) ** 2
sinc_kernel /= np.sum(sinc_kernel)

intensity_map = np.array([
    fftconvolve(intensity_map[i], sinc_kernel, mode='same')
    for i in range(intensity_map.shape[0])
])

intensity_map /= np.max(intensity_map)

kernel = gaussian_kernel(50, 5)[:, None] * gaussian_kernel(50, 5)
intensity_map = fftconvolve(intensity_map, kernel, mode="same")

intensity_map /= np.max(intensity_map)

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

