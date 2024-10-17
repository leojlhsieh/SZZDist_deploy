# %%
# Add parent directory to sys.path temporarily, so this file can be run directly as main
from pathlib import Path
import sys
file = Path(__file__).resolve()
sys.path.append(str(Path(file).resolve().parent.parent))

import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from torch.nn.parameter import Parameter
from torch import nn
import torch
from tool.check_gpu import check_gpu


# %%

VERBOSE = False
keep_propagation = False

if VERBOSE:
    check_gpu()


def plot_field(field, title=None, abs_min=0, abs_max=None):
    """Plot 2D field with magnitude and phase."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    field = field.detach().cpu()
    im0 = axs[0].imshow(field.abs().squeeze(), cmap='inferno', vmin=abs_min, vmax=abs_max)
    axs[0].set_title('Magnitude of field')
    fig.colorbar(im0, ax=axs[0])
    if field.is_complex():
        im1 = axs[1].imshow(field.angle().squeeze(), cmap='hsv', vmin=-pi, vmax=pi)
    else:
        im1 = axs[1].imshow(torch.zeros_like(field).squeeze(), cmap='hsv', vmin=-pi, vmax=pi)
    axs[1].set_title('Phase of field')
    fig.colorbar(im1, ax=axs[1])
    plt.suptitle(f"{title}\n{field.shape} {field.dtype} value[{field.abs().min().item():.2f} {field.abs().max().item():.2f}]")
    plt.show()


def field_interpolate(arr_in: torch.Tensor, size, pad=None, crop=None, amplitude_pad_value=0):
    """
    arr_in: [batch, channel, height, width]
    """
    assert (pad is None) ^ (crop is None), f"{pad=}, {crop=}. One and only one of them should be None"
    assert arr_in.dim() == 4, f"{arr_in.dim()=}. Dim shou be 4, [Batch, Channel, Height, Width]"
    # if arr_in is not complex
    if arr_in.is_complex():
        amplitude_in = arr_in.abs()
        phase_in = arr_in.angle()
        temp1 = nn.functional.interpolate(amplitude_in, size=size)
        temp2 = nn.functional.interpolate(phase_in, size=size)
        if pad is not None:  # Upsample
            amplitude_out = nn.functional.pad(temp1, pad=pad, mode='constant', value=amplitude_pad_value)
            phase_out = nn.functional.pad(temp2, pad=pad, mode='constant', value=0)
        elif crop is not None:  # Downsample
            amplitude_out = temp1[:, :, crop[0]:crop[1], crop[2]:crop[3]]
            phase_out = temp2[:, :, crop[0]:crop[1], crop[2]:crop[3]]
        arr_out = amplitude_out * torch.exp(1j*phase_out)
        return arr_out
    else:
        temp = nn.functional.interpolate(arr_in, size=size)
        if pad is not None:
            arr_out = nn.functional.pad(temp, pad=pad, mode='constant', value=0)
        elif crop is not None:
            arr_out = temp[:, :, crop[0]:crop[1], crop[2]:crop[3]]
        return arr_out


# Set universal parameters
wl0 = 849.5e-9  # free space wavelength in [m]
n = 1.4525  # medium refractive index  # Fused silica @ 850 nm
k0 = 2 * pi / wl0  # free space wavenumber in [m-1]
k = n * k0  # medium wavenumber [m-1]
if VERBOSE:
    print(f"{wl0=}, {n=}, {k0=}, {k=}")


# Set coordinates for BPM
"""usinging default indixing grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')"""
Nx_bpm = 1024   # width of BPM [pixel]
Ny_bpm = 1024  # height of BPM [pixel]
dx_bpm = 4e-6  # pixel size of BPM [m]
dy_bpm = 4e-6  # pixel size of BPM [m]
Lx_bpm = dx_bpm * (Nx_bpm - 1)  # width of BPM [m]
Ly_bpm = dy_bpm * (Ny_bpm - 1)  # height of BPM [m]
x_bpm = (dx_bpm * torch.arange(0, Nx_bpm, 1, dtype=torch.float32) - Lx_bpm/2).unsqueeze(0)  # x coordinates of the BPM grid, in [m]
y_bpm = (dy_bpm * torch.arange(0, Ny_bpm, 1, dtype=torch.float32) - Ly_bpm/2).unsqueeze(1)  # y coordinates of the BPM grid, in [m]
if VERBOSE:
    print(f"{Nx_bpm=}, {Ny_bpm=}")
    print(f"{dx_bpm=}, {dy_bpm=}")
    print(f"{Lx_bpm=}, {Ly_bpm=}")  # Lx_bpm=0.004092, Ly_bpm=0.004092

Lz_bpm = 48e-3  # propagation distances [m]
dz_bpm = 1e-3  # propagation step in [m]  # Ulas 8.55 mm
Nz_bpm = round(Lz_bpm/dz_bpm) + 1   # propagation distances [pixel]
z_bpm = (dz_bpm * torch.arange(0, Nz_bpm, 1, dtype=torch.float32))  # propagation distances [m]
if VERBOSE:
    print(f'{Nz_bpm=}')
    print(f'{dz_bpm=}')
    print(f'{Lz_bpm=}')

kx_bpm = 2 * pi * torch.fft.fftfreq(Nx_bpm, dx_bpm, dtype=torch.float32).unsqueeze(0)  # # kx coordinates of the BPM grid, in [1/m]
ky_bpm = 2 * pi * torch.fft.fftfreq(Ny_bpm, dy_bpm, dtype=torch.float32).unsqueeze(1)   # ky coordinates of the BPM grid, in [1/m]
K2_bpm = kx_bpm**2 + ky_bpm**2
kz_bpm = torch.sqrt(k**2 - K2_bpm)
if VERBOSE:
    print(f"{kx_bpm.shape=}, {ky_bpm.shape=}, {K2_bpm.shape=}, {kz_bpm.shape=}")

DFR = torch.exp(-1j * (K2_bpm) / (k+kz_bpm) * dz_bpm)  # nonparaxial diffraction kernel in frequency domain
super_gaussian = torch.exp(-((x_bpm / (0.9*Lx_bpm/(2*torch.sqrt(torch.log(torch.tensor(2))))))**20 + (y_bpm / (0.9*Ly_bpm/(2*torch.sqrt(torch.log(torch.tensor(2))))))**20))  # absorbing boundary super gaussian to power fo 10, FWHM = 0.9 * Lx_bpm

DFR = DFR.unsqueeze(0).unsqueeze(0)  # nonparaxial diffraction kernel in frequency domain
super_gaussian = super_gaussian.unsqueeze(0).unsqueeze(0)  # absorbing boundary super gaussian to power fo 10, FWHM = 0.9 * Lx_bpm


# Set coordinates for SLM
Nx_slm = 300  # width of SLM [pixel]
Ny_slm = 300  # height of SLM [pixel]
dx_slm = 8e-6  # pixel size of SLM [m]
dy_slm = 8e-6  # pixel size of SLM [m]
Lx_slm = dx_slm * (Nx_slm - 1)  # width of SLM [m]
Ly_slm = dy_slm * (Ny_slm - 1)  # height of SLM [m]
x_slm = (dx_slm * torch.arange(0, Nx_slm, 1, dtype=torch.float32) - Lx_slm/2).unsqueeze(0)  # x coordinates of the SLM grid, in [m]
y_slm = (dy_slm * torch.arange(0, Ny_slm, 1, dtype=torch.float32) - Ly_slm/2).unsqueeze(1)  # y coordinates of the SLM grid, in [m]
if VERBOSE:
    print(f"{Nx_slm=}, {Ny_slm=}")
    print(f"{dx_slm=}, {dy_slm=}")
    print(f"{Lx_slm=}, {Ly_slm=}")  # Lx_slm=0.002392, Ly_slm=0.002392


# Set coordinates for camera
Nx_cam = 300  # width of camera [pixel]
Ny_cam = 300  # height of camera [pixel]
dx_cam = 8e-6  # pixel size of camera [m]
dy_cam = 8e-6  # pixel size of camera [m]
Lx_cam = dx_cam * (Nx_cam - 1)  # width of camera [m]
Ly_cam = dy_cam * (Ny_cam - 1)  # height of camera [m]
x_cam = (dx_cam * torch.arange(0, Nx_cam, 1, dtype=torch.float32) - Lx_cam/2).unsqueeze(0)  # x coordinates of the camera grid, in [m]
y_cam = (dy_cam * torch.arange(0, Ny_cam, 1, dtype=torch.float32) - Ly_cam/2).unsqueeze(1)  # y coordinates of the camera grid, in [m]
if VERBOSE:
    print(f"{Nx_cam=}, {Ny_cam=}")
    print(f"{dx_cam=}, {dy_cam=}")
    print(f"{Lx_cam=}, {Ly_cam=}")  # Lx_cam=0.002392, Ly_cam=0.002392


# SLM and BPM coordinate conversion
# SLM upsample to BPM
ny_nx_slm2bpm = (int(Ny_slm*dy_slm/dy_bpm), int(Nx_slm*dx_slm/dx_bpm))  # for 'size' in nn.functional.interpolate
ny = ny_nx_slm2bpm[0]
nx = ny_nx_slm2bpm[1]
Ny = Ny_bpm
Nx = Nx_bpm
y_start = (Ny - ny) // 2
x_start = (Nx - nx) // 2
pad_slm2bpm = (int(x_start), int(Nx-x_start-nx), int(y_start), int(Ny-y_start-ny))  # for 'pad' in nn.functional.pad
del nx, ny, Nx, Ny, x_start, y_start

# BPM downsample to SLM
ny_nx_bpm2slm = (int(Ny_bpm*dy_bpm/dy_slm), int(Nx_bpm*dx_bpm/dx_slm))  # for 'size' in nn.functional.interpolate
nx = ny_nx_bpm2slm[1]
ny = ny_nx_bpm2slm[0]
Nx = Nx_slm
Ny = Ny_slm
x_start = (nx - Nx) // 2
y_start = (ny - Ny) // 2
crop_bpm2slm = (int(y_start), int(y_start+Ny), int(x_start), int(x_start+Nx))  # for 'crop' in my function crop2d
del nx, ny, Nx, Ny, x_start, y_start

# Camera upsample to BPM
ny_nx_cam2bpm = (int(Ny_cam*dy_cam/dy_bpm), int(Nx_cam*dx_cam/dx_bpm))  # for 'size' in nn.functional.interpolate
ny = ny_nx_cam2bpm[0]
nx = ny_nx_cam2bpm[1]
Ny = Ny_bpm
Nx = Nx_bpm
y_start = (Ny - ny) // 2
x_start = (Nx - nx) // 2
pad_cam2bpm = (int(x_start), int(Nx-x_start-nx), int(y_start), int(Ny-y_start-ny))  # for 'pad' in nn.functional.pad
del nx, ny, Nx, Ny, x_start, y_start

# BPM downsample to Camera
ny_nx_bpm2cam = (int(Ny_bpm*dy_bpm/dy_cam), int(Nx_bpm*dx_bpm/dx_cam))  # for 'size' in nn.functional.interpolate
nx = ny_nx_bpm2cam[1]
ny = ny_nx_bpm2cam[0]
Nx = Nx_cam
Ny = Ny_cam
x_start = (nx - Nx) // 2
y_start = (ny - Ny) // 2
crop_bpm2cam = (int(y_start), int(y_start+Ny), int(x_start), int(x_start+Nx))  # for 'crop' in my function crop2d
del nx, ny, Nx, Ny, x_start, y_start



# %%
# Gaussian beam input
beam_fwhm = 5e-3  # [m]
beam_scale = beam_fwhm / (2*torch.sqrt(torch.log(torch.tensor(2, dtype=torch.float32))))
gaussian_beam_slm = torch.exp(-(x_slm/beam_scale)**2 - (y_slm/beam_scale)**2)
gaussian_beam_slm = gaussian_beam_slm.unsqueeze(0).unsqueeze(0).to(dtype=torch.complex64)
gaussian_beam_bpm = field_interpolate(gaussian_beam_slm, size=ny_nx_slm2bpm, pad=pad_slm2bpm)
if VERBOSE:
    plot_field(gaussian_beam_slm, title='gaussian_beam_slm')
    plot_field(gaussian_beam_bpm, title='gaussian_beam_bpm')
field_in = gaussian_beam_bpm  # Dim should be 4, [Batch, Channel, Height, Width]


# %%
# Modulation during propagation
def z2index(z):
    """
    given z, a location [m] in z axis
    return the index in BPM closest to z
    """
    return int(torch.argmin(torch.abs(z_bpm - z)))

# Lens
f_lens = 50e-3  # focal length of lens [m]
lens_slm = torch.exp(-1j*pi/wl0/f_lens*(x_slm**2+y_slm**2)).unsqueeze(0).unsqueeze(0)
lens_bpm = field_interpolate(lens_slm, size=ny_nx_slm2bpm, pad=pad_slm2bpm, amplitude_pad_value=1)
if VERBOSE:
    plot_field(lens_slm, title='lens_slm')
    plot_field(lens_bpm, title='lens_bpm')

# Generate angle of the input beam. We do it by multiplying the input field with a blazed grating
theta_x = 0.5       # angle in x direction [degree]
theta_y = 0.2       # angle in y direction [degree]
theta_x = torch.tensor(theta_x, dtype=torch.float32).deg2rad()
theta_y = torch.tensor(theta_y, dtype=torch.float32).deg2rad()
blazed_grating_slm = torch.exp(1j*2*pi/wl0*(torch.sin(theta_x)*x_slm+torch.sin(theta_y)*y_slm)).unsqueeze(0).unsqueeze(0)
blazed_grating_bpm = field_interpolate(blazed_grating_slm, size=ny_nx_slm2bpm, pad=pad_slm2bpm, amplitude_pad_value=1)
if VERBOSE:
    plot_field(blazed_grating_slm, title='blazed_grating_slm')
    plot_field(blazed_grating_bpm, title='blazed_grating_bpm')

# field_in = torch.cat((gaussian_beam_bpm, lens_bpm), dim=0)  # Dim should be 4, [Batch, Channel, Height, Width]
# plot_field(field_in[0, :, :, :], title='field_in[0,:,:,:]')
# plot_field(field_in[1,:,:,:], title='field_in[1,:,:,:]')

"""
mod_idx:
modulation indices. A list of indices. At the index, a modulation is applied.
first and last number should be 0 and Nz_bpm to compute the modulation at the beginning and end of the lattice

mod_spacing:
A list of the spacing between mod_idx

Example: 
Nz_bpm = 20
mod_idx = [0, 3, 8, 12, 13, 20]
mod_spacing = [3, 5, 4, 1, 7]
"""
Ldist = 6e-3  # distance between SLM and mirror [m]
mod_idx = [0]
mod_00 = torch.ones_like(field_in) # Illumination angle # Dim should be 4, [Batch, Channel, Height, Width]


mod_idx.append(z2index(Ldist))
# mod_01 = SLM0  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(z2index(Ldist*2))
mod_02 = torch.ones_like(field_in) # Angle of mirror  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(z2index(Ldist*3))
# mod_03 = SLM1  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(z2index(Ldist*4))
mod_04 = torch.ones_like(field_in) # Angle of mirror  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(z2index(Ldist*5))
# mod_05 = SLM2  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(z2index(Ldist*6))
mod_06 = torch.ones_like(field_in) # Angle of mirror  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(z2index(Ldist*7))
# mod_07 = SLM3  # Dim should be 4, [Batch, Channel, Height, Width]

mod_idx.append(Nz_bpm) # propagation to the end 

mod_spacing = [mod_idx[i] - mod_idx[i - 1] for i in range(1, len(mod_idx))]
assert sum(mod_spacing) == mod_idx[-1] == Nz_bpm, f"sum of the mod_spacing should be equal to Nz_bpm. {mod_spacing=}, {mod_idx=}, {Nz_bpm=}"
if VERBOSE:
    print(f"{mod_idx=}")
    print(f"{mod_spacing=}")
    print(f"{Nz_bpm=}")
    print(f"{len(mod_spacing)=}")
mod_spacing = torch.tensor(mod_spacing)


# BPM core
u0 = field_in

'''
if keep_propagation:
    # field_3d is the same size as the field_in, but add Nz_bpm depth dimension
    batch, channel, height, width = field_in.shape
    propagation = torch.zeros((batch, channel, height, width, Nz_bpm), dtype=field_in.dtype, device=field_in.device)
    propagation[:, :, :, :, 0] = field_in
u0 = field_in
for z_index in tqdm(range(1, Nz_bpm)):
    temp_indices = [i for i, x in enumerate(modulation_indices) if x == z_index]
    for i in temp_indices:
        u0 = u0 * modulations[i]
    u1 = torch.fft.ifft2(torch.fft.fft2(u0) * DFR) * super_gaussian
    u0 = u1
    if keep_propagation:
        propagation[:, :, :, :, z_index] = u1
field_out = u1
field_out_cam = field_interpolate(field_out.abs()**2, size=ny_nx_bpm2cam, crop=crop_bpm2cam)

if VERBOSE:
    plot_field(field_in[0,:,:,:], title='field_in[0,:,:,:]')
    plot_field(field_in[1,:,:,:], title='field_in[1,:,:,:]')
    if keep_propagation:
        plot_field(propagation[0, :, Ny_bpm//2, :, :], title='propagation[:, :, Ny_bpm//2, :, :]')
        plot_field(propagation[0, :, Ny_bpm//2, :, :], title='propagation[:, :, Ny_bpm//2, :, :]')
        plot_field(propagation[1, :, :, Nx_bpm//2, :], title='propagation[:, :, :, Nx_bpm//2, :]')
        plot_field(propagation[1, :, :, Nx_bpm//2, :], title='propagation[:, :, :, Nx_bpm//2, :]')
    plot_field(field_out[0,:,:,:], title='field_out[0,:,:,:]')
    plot_field(field_out[1,:,:,:], title='field_out[1,:,:,:]')
    plot_field(field_out_cam[0,:,:,:], title='field_out_cam[0,:,:,:]')
    plot_field(field_out_cam[1,:,:,:], title='field_out_cam[1,:,:,:]')
check_gpu()
'''


# %%
# SZZBPM_init_dict has field_in, DFR, super_gaussian, keep_propagation, Nz_bpm, modulation_indices, modulations, ny_nx_slm2bpm, pad_slm2bpm, ny_nx_bpm2slm, crop_bpm_slm, ny_nx_cam2bpm, pad_cam2bpm, ny_nx_bpm2cam, crop_bpm2cam
SZZBPM_init_dict = {
    'field_in': field_in,
    'DFR': DFR,
    'super_gaussian': super_gaussian,
    'keep_propagation': keep_propagation,
    'mod_spacing': mod_spacing,
    'mod_00': mod_00,
    'mod_02': mod_02,
    'mod_04': mod_04,
    'mod_06': mod_06,
    'ny_nx_slm2bpm': ny_nx_slm2bpm,
    'pad_slm2bpm': pad_slm2bpm,
    'ny_nx_bpm2slm': ny_nx_bpm2slm,
    'crop_bpm_slm': crop_bpm2slm,
    'ny_nx_cam2bpm': ny_nx_cam2bpm,
    'pad_cam2bpm': pad_cam2bpm,
    'ny_nx_bpm2cam': ny_nx_bpm2cam,
    'crop_bpm2cam': crop_bpm2cam
}


# %%
class SZZBPM(nn.Module):
    r"""SZZBPM model for BPM simulation.
    input image to put on SLM:
        size: (batch, 1, 300, 300)
        dtype: torch.float32
        value: [0.0, 1.0]

    output camera captured image, abs^2 of field:
        size: (batch, 1, 300, 300)  # Camera has same resolution as SLM
        dtype: torch.float32
        value: Not limited
    """

    def __init__(self, device=None, **kwargs):
        super().__init__()
        self.device = device
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.detach().clone().to(device))
            else:
                setattr(self, key, value)
        # Basic BPM parameters
        # self.field_in = field_in.detach().clone().to(device)
        # self.DFR = DFR.detach().clone().to(device)
        # self.super_gaussian = super_gaussian.detach().clone().to(device)
        # self.mod_00 = mod_00.detach().clone().to(device)
        # self.mod_02 = mod_02.detach().clone().to(device)
        # self.mod_04 = mod_04.detach().clone().to(device)
        # self.mod_06 = mod_06.detach().clone().to(device)
        # self.ny_nx_slm2bpm = ny_nx_slm2bpm.detach().clone().to(device)
        
        # SLM and BPM coordinate conversion
        # self.pad_slm2bpm = pad_slm2bpm.detach().clone().to(device)
        # self.ny_nx_bpm2slm = ny_nx_bpm2slm.detach().clone().to(device)
        # self.crop_bpm_slm = crop_bpm_slm.detach().clone().to(device)
        # self.ny_nx_cam2bpm = ny_nx_cam2bpm.detach().clone().to(device)
        # self.pad_cam2bpm = pad_cam2bpm.detach().clone().to(device)
        # self.ny_nx_bpm2cam = ny_nx_bpm2cam.detach().clone().to(device)
        # self.crop_bpm2cam = crop_bpm2cam.detach().clone().to(device)
        
        # training parameters, float32 is enough
        self.phase_scale0 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_scale1 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_scale2 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_scale3 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias0 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias1 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias2 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.phase_bias3 = Parameter(torch.randn((1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
        self.arange_table = torch.arange(6, dtype=torch.int32, device=device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Modulation during propagation
        modulation0_phase_slm = input * self.phase_scale0 + self.phase_bias0
        modulation0_phase_bpm = self.field_interpolate(modulation0_phase_slm, size=self.ny_nx_slm2bpm, pad=self.pad_slm2bpm)
        mod_01 = torch.exp(1j*modulation0_phase_bpm)

        modulation1_phase_slm = input * self.phase_scale1 + self.phase_bias1
        modulation1_phase_bpm = self.field_interpolate(modulation1_phase_slm, size=self.ny_nx_slm2bpm, pad=self.pad_slm2bpm)
        mod_03 = torch.exp(1j*modulation1_phase_bpm)

        modulation2_phase_slm = input * self.phase_scale2 + self.phase_bias2
        modulation2_phase_bpm = self.field_interpolate(modulation2_phase_slm, size=self.ny_nx_slm2bpm, pad=self.pad_slm2bpm)
        mod_05 = torch.exp(1j*modulation2_phase_bpm)

        modulation3_phase_slm = input * self.phase_scale3 + self.phase_bias3
        modulation3_phase_bpm = self.field_interpolate(modulation3_phase_slm, size=self.ny_nx_slm2bpm, pad=self.pad_slm2bpm)
        mod_07 = torch.exp(1j*modulation3_phase_bpm)

        # BPM core
        u0 = self.field_in * self.mod_00
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[0])

        u0 = u0 * mod_01  # weight * SLM + bias
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[1])

        u0 = u0 * self.mod_02
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[2])

        u0 = u0 * mod_03  # weight * SLM + bias
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[3])

        u0 = u0 * self.mod_04
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[4])

        u0 = u0 * mod_05  # weight * SLM + bias
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[5])

        u0 = u0 * self.mod_06
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[6])

        u0 = u0 * mod_07  # weight * SLM + bias
        u0 = self.freespace_bpm(field=u0, Nsteps=self.mod_spacing[7])
        
        field_out = u0
        field_out_cam = self.field_interpolate(field_out.abs()**2, size=ny_nx_bpm2cam, crop=crop_bpm2cam)

        return field_out_cam

    def field_interpolate(self, arr_in: torch.Tensor, size, pad=None, crop=None, amplitude_pad_value=0):
        """
        arr_in: [batch, channel, height, width]
        """
        assert (pad is None) ^ (crop is None), f"{pad=}, {crop=}. One and only one of them should be None"
        assert arr_in.dim() == 4, f"{arr_in.dim()=}. Dim shou be 4, [Batch, Channel, Height, Width]"
        # if arr_in is not complex
        if arr_in.is_complex():
            amplitude_in = arr_in.abs()
            phase_in = arr_in.angle()
            temp1 = nn.functional.interpolate(amplitude_in, size=size)
            temp2 = nn.functional.interpolate(phase_in, size=size)
            if pad is not None:  # Upsample
                amplitude_out = nn.functional.pad(temp1, pad=pad, mode='constant', value=amplitude_pad_value)
                phase_out = nn.functional.pad(temp2, pad=pad, mode='constant', value=0)
            elif crop is not None:  # Downsample
                amplitude_out = temp1[:, :, crop[0]:crop[1], crop[2]:crop[3]]
                phase_out = temp2[:, :, crop[0]:crop[1], crop[2]:crop[3]]
            arr_out = amplitude_out * torch.exp(1j*phase_out)
            return arr_out
        else:
            temp = nn.functional.interpolate(arr_in, size=size)
            if pad is not None:
                arr_out = nn.functional.pad(temp, pad=pad, mode='constant', value=0)
            elif crop is not None:
                arr_out = temp[:, :, crop[0]:crop[1], crop[2]:crop[3]]
            return arr_out

    def freespace_bpm(self, field, Nsteps):
        """
        field shape: [batch, channel, height, width]
        Nsteps: number of propagation steps

        return: field after Nsteps propagation
        """
        u0 = field
        arange_table = torch.arange(Nsteps, dtype=torch.int32, device=self.device)
        for i in arange_table:
            u0 = torch.fft.ifft2(torch.fft.fft2(u0) * self.DFR) * self.super_gaussian
        return u0


if __name__ == '__main__':
    # Get cpu, gpu or mps device for training.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # Set the device globally
    # torch.set_default_device(device)

    check_gpu()

    model_0 = SZZBPM(device=device, **SZZBPM_init_dict)
    image_in = torch.ones(1, 1, 300, 300, dtype=torch.float32, device=device)
    output = model_0(image_in)

    check_gpu()
    # %%
    # plot_field(model_0.propagation[:, :, Ny_bpm//2, :, :], title='propagation[:, :, Ny_bpm//2, :, :]')
    # plot_field(model_0.propagation[:, :, :, Nx_bpm//2, :], title='propagation[:, :, :, Nx_bpm//2, :]')

    # %%
    plot_field(model_0.modulations[0], title='model_0.modulations[0]')
