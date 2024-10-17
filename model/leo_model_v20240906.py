# %%
## Add parent directory to sys.path temporarily, so this file can be run directly as main
import sys
from pathlib import Path
file = Path(__file__).resolve()
sys.path.append(str(Path(file).resolve().parent.parent))

from tool.check_gpu import check_gpu
check_gpu()

# %%
import torch
from torch import nn
import numpy as np
from math import pi
from torch import Tensor  # For python typing
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio


# Get cpu, gpu or mps device for training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def crop2d(arr_in: Tensor, crop: tuple[int, int, int, int]) -> Tensor:
    return arr_in[crop[0]:crop[1], crop[2]:crop[3]]


def plot_field(field, title=None, abs_min=None, abs_max=None):
    """Plot 2D field with magnitude and phase."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    field = field.detach().cpu()
    im0 = axs[0].imshow(field.abs(), cmap='inferno', vmin=abs_min, vmax=abs_max)
    axs[0].set_title('Magnitude of field')
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(field.angle(), cmap='hsv', vmin=-pi, vmax=pi)
    axs[1].set_title('Phase of field')
    fig.colorbar(im1, ax=axs[1])
    plt.suptitle(f"{title}\n{field.shape} {field.abs().min().item():.2f} {field.abs().max().item():.2f}")
    plt.show()


def resize_to_grid_space(arr_in: Tensor, x_in: Tensor, y_in: Tensor, x_out: Tensor, y_out: Tensor, verbose=False) -> Tensor:
    dy_in = y_in[1, 0] - y_in[0, 0]
    dx_in = x_in[0, 1] - x_in[0, 0]
    dy_out = y_out[1, 0] - y_out[0, 0]
    dx_out = x_out[0, 1] - x_out[0, 0]

    ny, nx = arr_in.shape
    ny, nx = int(torch.round(ny*dy_in/dy_out)), int(torch.round(nx*dx_in/dx_out))
    # Please provide input tensor in (N, C, d1, d2, ...,dK) format and scale_factor in (s1, s2, ...,sK) format.
    arr_in_resample = torch.nn.functional.interpolate(arr_in.unsqueeze(0).unsqueeze(0), size=(ny, nx)).squeeze()

    Ny = y_out.shape[0]
    Nx = x_out.shape[1]
    if Nx >= nx and Ny >= ny:
        x_start = (Nx - nx) // 2
        y_start = (Ny - ny) // 2
        # arr_in_resample_pad = torch.zeros((Ny, Nx), dtype=arr_in_resample.dtype)
        # arr_in_resample_pad[y_start:y_start+ny, x_start:x_start+nx] = arr_in_resample
        pad_slm2bpm = (x_start, Nx-x_start-nx, y_start, Ny-y_start-ny)
        arr_in_resample_pad = nn.functional.pad(arr_in_resample, pad=pad_slm2bpm, mode='constant', value=0)
        return arr_in_resample_pad
    else:
        x_start = (nx - Nx) // 2
        y_start = (ny - Ny) // 2
        # arr_in_resample_crop = arr_in_resample[y_start:y_start+Ny, x_start:x_start+Nx]
        crop_bpm_slm = (y_start, y_start+Ny, x_start, x_start+Nx)
        arr_in_resample_crop = crop2d(arr_in_resample, crop=crop_bpm_slm)
        return arr_in_resample_crop


# Set universal parameters
wl0 = 849.5e-9  # free space wavelength in [m]
n = 1.4525  # medium refractive index  # Fused silica @ 850 nm
k0 = 2 * pi / wl0  # free space wavenumber in [m-1]
k = n * k0  # medium wavenumber [m-1]
print(f"{wl0=}, {n=}, {k0=}, {k=}")

# Set coordinates for SLM and BPM
"""usinging default indixing grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')"""
Nx_slm = 300  # width of SLM [pixel]
Ny_slm = 300  # height of SLM [pixel]
dx_slm = 8e-6  # pixel size of SLM [m]
dy_slm = 8e-6  # pixel size of SLM [m]
Lx_slm = dx_slm * (Nx_slm - 1)  # width of SLM [m]
Ly_slm = dy_slm * (Ny_slm - 1)  # height of SLM [m]
x_slm = (dx_slm * torch.arange(0, Nx_slm, 1) - Lx_slm/2).unsqueeze(0)  # x coordinates of the SLM grid, in [m]
y_slm = (dy_slm * torch.arange(0, Ny_slm, 1) - Ly_slm/2).unsqueeze(1)  # y coordinates of the SLM grid, in [m]
print(f"{Nx_slm=}, {Ny_slm=}")
print(f"{dx_slm=}, {dy_slm=}")
print(f"{Lx_slm=}, {Ly_slm=}")  # Lx_slm=0.002392, Ly_slm=0.002392

Nx_bpm = 1024   # width of BPM [pixel]
Ny_bpm = 1024  # height of BPM [pixel]
dx_bpm = 4e-6  # pixel size of BPM [m]
dy_bpm = 4e-6  # pixel size of BPM [m]
Lx_bpm = dx_bpm * (Nx_bpm - 1)  # width of BPM [m]
Ly_bpm = dy_bpm * (Ny_bpm - 1)  # height of BPM [m]
x_bpm = (dx_bpm * torch.arange(0, Nx_bpm, 1) - Lx_bpm/2).unsqueeze(0)  # x coordinates of the BPM grid, in [m]
y_bpm = (dy_bpm * torch.arange(0, Ny_bpm, 1) - Ly_bpm/2).unsqueeze(1)  # y coordinates of the BPM grid, in [m]
print(f"{Nx_bpm=}, {Ny_bpm=}")
print(f"{dx_bpm=}, {dy_bpm=}")
print(f"{Lx_bpm=}, {Ly_bpm=}")  # Lx_bpm=0.004092, Ly_bpm=0.004092

Lz_bpm = 48e-3  # propagation distances [m]
dz_bpm = 1e-3  # propagation step in [m]  # Ulas 8.55 mm
Nz_bpm = round(Lz_bpm/dz_bpm) + 1   # propagation distances [pixel]
z_bpm = (dz_bpm * torch.arange(0, Nz_bpm, 1))  # propagation distances [m]
print(f'{Nz_bpm=}')
print(f'{dz_bpm=}')
print(f'{Lz_bpm=}')


kx_bpm = 2 * pi * torch.fft.fftfreq(Nx_bpm, dx_bpm).unsqueeze(0)  # # kx coordinates of the BPM grid, in [1/m]
ky_bpm = 2 * pi * torch.fft.fftfreq(Ny_bpm, dy_bpm).unsqueeze(1)   # ky coordinates of the BPM grid, in [1/m]
K2_bpm = kx_bpm**2 + ky_bpm**2
kz_bpm = torch.sqrt(k**2 - K2_bpm)
print(f"{kx_bpm.shape=}, {ky_bpm.shape=}, {K2_bpm.shape=}, {kz_bpm.shape=}")

DFR = torch.exp(-1j * (K2_bpm) / (k+kz_bpm) * dz_bpm).to(torch.complex64)  # nonparaxial diffraction kernel in frequency domain
super_gaussian = torch.exp(-((x_bpm / (0.9*Lx_bpm/(2*torch.sqrt(torch.log(torch.tensor(2))))))**20 + (y_bpm / (0.9*Ly_bpm/(2*torch.sqrt(torch.log(torch.tensor(2))))))**20)).to(torch.float32)  # absorbing boundary super gaussian to power fo 10, FWHM = 0.9 * Lx_bpm

# SLM and BPM coordinate conversion
ny_nx_slm2bpm = (int(Ny_slm*dy_slm/dy_bpm), int(Nx_slm*dx_slm/dx_bpm))  # for 'size' in nn.functional.interpolate
ny = ny_nx_slm2bpm[0]
nx = ny_nx_slm2bpm[1]
Ny = Ny_bpm
Nx = Nx_bpm
y_start = (Ny - ny) // 2
x_start = (Nx - nx) // 2
pad_slm2bpm = (x_start, Nx-x_start-nx, y_start, Ny-y_start-ny)  # for 'pad' in nn.functional.pad
del nx, ny, Nx, Ny, x_start, y_start

ny_nx_bpm2slm = (int(Ny_bpm*dy_bpm/dy_slm), int(Nx_bpm*dx_bpm/dx_slm))  # for 'size' in nn.functional.interpolate
nx = ny_nx_bpm2slm[1]
ny = ny_nx_bpm2slm[0]
Nx = Nx_slm
Ny = Ny_slm
x_start = (nx - Nx) // 2
y_start = (ny - Ny) // 2
crop_bpm_slm = (y_start, y_start+Ny, x_start, x_start+Nx)  # for 'crop' in my function crop2d
del nx, ny, Nx, Ny, x_start, y_start


# Gaussian beam input
beam_fwhm = 5e-3  # [m]
beam_scale = beam_fwhm / (2*torch.sqrt(torch.log(torch.tensor(2))))
gaussian_beam = torch.exp(-(x_slm/beam_scale)**2 - (y_slm/beam_scale)**2).to(dtype=torch.float32)
temp = nn.functional.interpolate(gaussian_beam.unsqueeze(0).unsqueeze(0), size=ny_nx_slm2bpm).squeeze()
gaussian_beam = nn.functional.pad(temp, pad=pad_slm2bpm, mode='constant', value=0).to(dtype=torch.complex64)
plot_field(gaussian_beam, title='gaussian_beam', abs_min=0)

# Modulation during propagation
# Lens
f_lens = 100e-3  # focal length of lens [m]
lens = torch.exp(-1j*pi/wl0/f_lens*(x_slm**2+y_slm**2)).to(torch.complex64)
plot_field(lens, title='lens')


# Blazed grating
# Generate angle of the input beam. We do it by multiplying the input field with a blazed grating
theta_x = 0.5       # angle in x direction [degree]
theta_y = 0.2       # angle in y direction [degree]
theta_x = torch.tensor(theta_x).deg2rad()
theta_y = torch.tensor(theta_y).deg2rad()
blazed_grating_phase_slm = 2*pi/wl0*(torch.sin(theta_x)*x_slm+torch.sin(theta_y)*y_slm).to(torch.float32)
blazed_grating_slm = torch.exp(1j*blazed_grating_phase_slm).to(torch.complex64)
plot_field(blazed_grating_slm, title='blazed_grating_slm')

blazed_grating_phase_bpm = resize_to_grid_space(blazed_grating_phase_slm, x_slm, y_slm, x_bpm, y_bpm)
blazed_grating_bpm = torch.exp(1j*blazed_grating_phase_bpm).to(torch.complex64)
plot_field(blazed_grating_bpm, title='blazed_grating_bpm')

back_to_slm = resize_to_grid_space(blazed_grating_phase_bpm, x_bpm, y_bpm, x_slm, y_slm)
back_to_slm_field = torch.exp(1j*back_to_slm).to(torch.complex64)
plot_field(back_to_slm_field, title='back_to_slm_field')

Ldist = 6e-3  # distance between SLM and mirror [m]


def z2index(z): return int(torch.argmin(torch.abs(z_bpm - z)))


modulation_dict = {'z_index': [], 'modulation': []}
# modulation_dict = {'z_index':[15, 30, 60], 'modulation':[circular_aperture, blazed_grating, lens]}
# modulation_dict['z_index'].append(z2index(Ldist))
# modulation_dict['z_index'].append(z2index(Ldist*1.5))
# modulation_dict['z_index'].append(z2index(Ldist*3.5))
# modulation_dict['z_index'].append(z2index(Ldist*4.5))
# modulation_dict['modulation'].append(blazed_grating_bpm)
# modulation_dict['modulation'].append(blazed_grating_bpm)
# modulation_dict['modulation'].append(blazed_grating_bpm)
# modulation_dict['modulation'].append(blazed_grating_bpm)


# %%
# BPM core
field_in = gaussian_beam
keep_3d = False
if keep_3d:
    field_3d = torch.zeros((Ny_bpm, Nx_bpm, Nz_bpm), dtype=torch.complex64)
    field_3d[:, :, 0] = field_in
u0 = field_in
for z_index in tqdm(range(1, Nz_bpm)):
    if z_index in modulation_dict['z_index']:
        mod_index = modulation_dict['z_index'].index(z_index)
        u0 = u0 * modulation_dict['modulation'][mod_index]
    u1 = torch.fft.ifft2(torch.fft.fft2(u0) * DFR) * super_gaussian
    u0 = u1
    if keep_3d:
        field_3d[:, :, z_index] = u1
field_out = u1
plot_field(field_out, title='field_out', abs_min=0)
if keep_3d:
    plot_field(field_3d[Ny_bpm//2, :, :], title='field_3d[Ny_bpm//2,:,:]', abs_min=0)
    plot_field(field_3d[:, Nx_bpm//2, :], title='field_3d[:,Nx_bpm//2,:]', abs_min=0)


# %%
class SZZBPM(nn.Module):
    r"""SZZBPM model for BPM simulation.
    input: (300, 300), dtype: torch.float32
    output: (300, 300), dtype: torch.float32
    """

    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # Basic BPM parameters
        self.DFR = DFR.detach().clone().to(dtype=torch.complex64, device=device)
        self.super_gaussian = super_gaussian.detach().clone().to(dtype=torch.float32, device=device)
        # training parameters
        self.phase_scale1 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_scale2 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_scale3 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_scale4 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias1 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias2 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias3 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias4 = Parameter(torch.randn((Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        # SLM and BPM coordinate conversion
        self.ny_nx_slm2bpm = ny_nx_slm2bpm

        self.pad_slm2bpm = pad_slm2bpm
        self.ny_nx_bpm2slm = ny_nx_bpm2slm
        self.crop_bpm_slm = crop_bpm_slm

    def forward(self, input: Tensor) -> Tensor:
        # Modulation during propagation
        self.modulation1_slm = input * self.phase_scale1 + self.phase_bias1
        temp = nn.functional.interpolate(self.modulation1_slm.unsqueeze(0).unsqueeze(0), size=self.ny_nx_slm2bpm).squeeze()
        temp2 = nn.functional.pad(temp, pad=self.pad_slm2bpm, mode='constant', value=0)
        self.modulation1_bpm = torch.exp(1j*temp2).to(dtype=torch.complex64, device=device)

        self.modulation2_slm = input * self.phase_scale2 + self.phase_bias2
        temp = nn.functional.interpolate(self.modulation2_slm.unsqueeze(0).unsqueeze(0), size=self.ny_nx_slm2bpm).squeeze()
        temp2 = nn.functional.pad(temp, pad=self.pad_slm2bpm, mode='constant', value=0)
        self.modulation2_bpm = torch.exp(1j*temp2).to(dtype=torch.complex64, device=device)

        self.modulation3_slm = input * self.phase_scale3 + self.phase_bias3
        temp = nn.functional.interpolate(self.modulation3_slm.unsqueeze(0).unsqueeze(0), size=self.ny_nx_slm2bpm).squeeze()
        temp2 = nn.functional.pad(temp, pad=self.pad_slm2bpm, mode='constant', value=0)
        self.modulation3_bpm = torch.exp(1j*temp2).to(dtype=torch.complex64, device=device)

        self.modulation4_slm = input * self.phase_scale4 + self.phase_bias4
        temp = nn.functional.interpolate(self.modulation4_slm.unsqueeze(0).unsqueeze(0), size=self.ny_nx_slm2bpm).squeeze()
        temp2 = nn.functional.pad(temp, pad=self.pad_slm2bpm, mode='constant', value=0)
        self.modulation4_bpm = torch.exp(1j*temp2).to(dtype=torch.complex64, device=device)

        def z2index(z): return int(torch.argmin(torch.abs(z_bpm - z)))
        self.modulation_dict = {'z_index': [], 'modulation': []}
        self.modulation_dict['z_index'].append(z2index(Ldist))
        self.modulation_dict['z_index'].append(z2index(Ldist*1.5))
        self.modulation_dict['z_index'].append(z2index(Ldist*3.5))
        self.modulation_dict['z_index'].append(z2index(Ldist*4.5))
        self.modulation_dict['modulation'].append(self.modulation1_bpm)
        self.modulation_dict['modulation'].append(self.modulation2_bpm)
        self.modulation_dict['modulation'].append(self.modulation3_bpm)
        self.modulation_dict['modulation'].append(self.modulation4_bpm)
        assert len(self.modulation_dict['z_index']) == len(self.modulation_dict['modulation']), f"z_index len {len(self.modulation_dict['z_index'])}, modulation len {len(self.modulation_dict['modulation'])}"

        # BPM core
        self.field_in = gaussian_beam.to(device=device)
        keep_3d = True
        if keep_3d:
            self.field_3d = torch.zeros((Ny_bpm, Nx_bpm, Nz_bpm), dtype=torch.complex64, device=device)
            self.field_3d[:, :, 0] = self.field_in
        u0 = self.field_in
        for z_index in tqdm(range(1, Nz_bpm)):
            if z_index in self.modulation_dict['z_index']:
                mod_index = self.modulation_dict['z_index'].index(z_index)
                u0 = u0 * self.modulation_dict['modulation'][mod_index]
            u1 = torch.fft.ifft2(torch.fft.fft2(u0) * self.DFR) * self.super_gaussian
            u0 = u1
            if keep_3d:
                self.field_3d[:, :, z_index] = u1
        self.field_out = u1
        plot_field(self.field_in, title='field_in', abs_min=0)
        plot_field(self.field_out, title='field_out', abs_min=0)
        if keep_3d:
            plot_field(self.field_3d[Ny_bpm//2, :, :], title='field_3d[Ny_bpm//2,:,:]', abs_min=0)
            plot_field(self.field_3d[:, Nx_bpm//2, :], title='field_3d[:,Nx_bpm//2,:]', abs_min=0)

        # Acquisition
        temp = self.field_out.abs()
        temp2 = torch.nn.functional.interpolate(temp.unsqueeze(0).unsqueeze(0), size=self.ny_nx_bpm2slm).squeeze()
        image_out = self.crop2d(temp2, self.crop_bpm_slm)

        # logits = torch.rand((2,2))
        return image_out

    def extra_repr(self) -> str:
        return f'SZZBPM'

    def crop2d(self, arr_in: Tensor, crop) -> Tensor:
        return arr_in[crop[0]:crop[1], crop[2]:crop[3]]


model_0 = SZZBPM().to(device)
image_in = torch.ones(300, 300).to(dtype=torch.float32, device=device)
image_in = blazed_grating_phase_slm
output = model_0(image_in)
print(output.size())


plt.figure()
plt.imshow(image.detach().cpu())
plt.colorbar()

# %%
# model_0.state_dict()
model_0.eval()
