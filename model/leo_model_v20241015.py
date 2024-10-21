# %%
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from tool.check_gpu import check_gpu


def ceiling_nearest_power_of_2(n):
    if n < 1:
        raise ValueError("Input must be a positive integer.")
    return 1 << (n - 1).bit_length()


def plot_field(field, title=None, abs_min=0, abs_max=None):
    """Plot 2D field with magnitude and phase."""
    field = field.detach().cpu().squeeze()
    if field.dim() == 3:
        y, x, z = field.shape
        plot_field(field[y//2, :, :], title=f'YZ of {title}', abs_min=abs_min, abs_max=abs_max)
        plot_field(field[:, x//2, :], title=f'XZ of {title}', abs_min=abs_min, abs_max=abs_max)
        return None
    elif field.dim() == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        im0 = axs[0].imshow(field.abs(), cmap='inferno', vmin=abs_min, vmax=abs_max)
        axs[0].set_title('Magnitude of field')
        fig.colorbar(im0, ax=axs[0])
        if field.is_complex():
            im1 = axs[1].imshow(field.angle(), cmap='hsv', vmin=-pi, vmax=pi)
        else:
            im1 = axs[1].imshow(torch.zeros_like(field), cmap='hsv', vmin=-pi, vmax=pi)
        axs[1].set_title('Phase of field')
        fig.colorbar(im1, ax=axs[1])
        plt.suptitle(f"{title}\n{field.shape} {field.dtype} value[{field.abs().min().item():.2f} {field.abs().max().item():.2f}]")
        plt.show()
        return None


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


VERBOSE = False
keep_propagation = False

# 'bpm_color': {'value': 'gray'},  # 'gray', 'rgb'
# 'bpm_mode': {'value': 'bpm'},  # 'bpm', 'CNNpatch-bpm', 'fft-bpm', 'nothing'
# 'bpm_depth': {'value': 4},  # 1, 2, 3, 4, 5, 6, 7, 8
# 'bpm_width': {'value': 300},  # 75, 150, 300, 600, 1200
# 'bpm_parallel': {'value': 1},
# 'model_feature': {'value': 'maxpool30-ReLU'},  # 'CNN-ReLU', 'nothing'


def build_model_bpm(bpm_depth, bpm_width, layer_sampling=6, device=None):

    Nx_slm = int(bpm_width)
    temp = bpm_width*2/0.7  # each SLM pixels contain 2 bpm pixels, only use the 0.7 center part of bpm area to avoid boundary effect
    Nx_bpm = ceiling_nearest_power_of_2(int(temp))
    del temp

    # Set universal parameters
    wl0 = 849.5e-9  # free space wavelength in [m]
    n = 1.4525  # medium refractive index  # Fused silica @ 850 nm
    k0 = 2 * pi / wl0  # free space wavenumber in [m-1]
    k = n * k0  # medium wavenumber [m-1]
    if VERBOSE:
        print(f"{wl0=}, {n=}, {k0=}, {k=}")

    # Set coordinates for BPM
    # Usinging default indixing grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    Nx_bpm = Nx_bpm   # width of BPM [pixel]
    Ny_bpm = Nx_bpm  # height of BPM [pixel]
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

    Ldist = 6e-3  # distance between SLM and mirror [m]
    layer_sampling = layer_sampling  # number of propagation steps from SLM to mirror
    Lz_bpm = bpm_depth * 2 * Ldist  # total propagation distances [m]
    dz_bpm = Ldist/layer_sampling  # propagation step in [m]
    Nz_bpm = round(Lz_bpm/dz_bpm) + 1   # propagation distances [pixel]
    z_bpm = (dz_bpm * torch.arange(0, Nz_bpm, 1, dtype=torch.float32))  # propagation distances [m]
    assert torch.isclose(z_bpm[-1], torch.tensor(Lz_bpm, dtype=torch.float32)), f"{z_bpm[-1]=} != {Lz_bpm=}"  # Use torch.isclose instead of == to avoid floating point error
    fast_range = torch.arange(Nz_bpm, dtype=torch.int32)
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

    DFR = DFR.unsqueeze(0).unsqueeze(0)  # two extra dimensions for batch and color channel
    super_gaussian = super_gaussian.unsqueeze(0).unsqueeze(0)  # two extra dimensions for batch and color channel
    if VERBOSE:
        plot_field(DFR, title='DFR')
        plot_field(super_gaussian, title='super_gaussian')

    # Set coordinates for SLM
    Nx_slm = Nx_slm  # width of SLM [pixel]
    Ny_slm = Nx_slm  # height of SLM [pixel]
    dx_slm = 8e-6  # pixel size of SLM [m]
    dy_slm = 8e-6  # pixel size of SLM [m]
    Lx_slm = dx_slm * (Nx_slm - 1)  # width of SLM [m]
    Ly_slm = dy_slm * (Ny_slm - 1)  # height of SLM [m]
    x_slm = (dx_slm * torch.arange(0, Nx_slm, 1, dtype=torch.float32) - Lx_slm/2).unsqueeze(0)  # x coordinates of the SLM grid, in [m]
    y_slm = (dy_slm * torch.arange(0, Ny_slm, 1, dtype=torch.float32) - Ly_slm/2).unsqueeze(1)  # y coordinates of the SLM grid, in [m]
    if VERBOSE:
        print(f"{Nx_slm=}, {Ny_slm=}")
        print(f"{dx_slm=}, {dy_slm=}")
        print(f"{Lx_slm=}, {Ly_slm=}")

    # Set coordinates for camera, assume same as SLM
    Nx_cam = Nx_slm  # width of camera [pixel]
    Ny_cam = Nx_cam  # height of camera [pixel]
    dx_cam = 8e-6  # pixel size of camera [m]
    dy_cam = 8e-6  # pixel size of camera [m]
    Lx_cam = dx_cam * (Nx_cam - 1)  # width of camera [m]
    Ly_cam = dy_cam * (Ny_cam - 1)  # height of camera [m]
    x_cam = (dx_cam * torch.arange(0, Nx_cam, 1, dtype=torch.float32) - Lx_cam/2).unsqueeze(0)  # x coordinates of the camera grid, in [m]
    y_cam = (dy_cam * torch.arange(0, Ny_cam, 1, dtype=torch.float32) - Ly_cam/2).unsqueeze(1)  # y coordinates of the camera grid, in [m]
    if VERBOSE:
        print(f"{Nx_cam=}, {Ny_cam=}")
        print(f"{dx_cam=}, {dy_cam=}")
        print(f"{Lx_cam=}, {Ly_cam=}")

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

    # Gaussian beam input
    beam_fwhm = 5e-3  # [m]
    beam_scale = beam_fwhm / (2*torch.sqrt(torch.log(torch.tensor(2, dtype=torch.float32))))
    gaussian_beam_slm = torch.exp(-(x_slm/beam_scale)**2 - (y_slm/beam_scale)**2)
    gaussian_beam_slm = gaussian_beam_slm.unsqueeze(0).unsqueeze(0).to(dtype=torch.complex64)
    gaussian_beam_bpm = field_interpolate(gaussian_beam_slm, size=ny_nx_slm2bpm, pad=pad_slm2bpm)
    if VERBOSE:
        plot_field(gaussian_beam_slm, title='gaussian_beam_slm')
        plot_field(gaussian_beam_bpm, title='gaussian_beam_bpm')

    # Plane wave input
    plan_wave_slm = torch.ones((1, 1, Ny_slm, Nx_slm), dtype=torch.complex64)  # plan wave input
    plan_wave_bpm = field_interpolate(plan_wave_slm, size=ny_nx_slm2bpm, pad=pad_slm2bpm)
    field_in = plan_wave_bpm  # Dim should be 4, [Batch, Channel, Height, Width]
    if VERBOSE:
        plot_field(plan_wave_slm, title='plan_wave_slm')
        plot_field(plan_wave_bpm, title='plan_wave_bpm')
        plot_field(field_in, title='field_in')

    # Modulation during propagation
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

    # System modulation
    system_modulation = torch.ones((bpm_depth, 1, 1, Ny_bpm, Nx_bpm), dtype=torch.complex64)  # Dim should be 5, [bpm_depth, Batch, Channel, Height, Width]
    # system_modulation[0, :, :, :, :] = blazed_grating_bpm
    # system_modulation[1, :, :, :, :] = lens_bpm
    # system_modulation[2, :, :, :, :] = torch.conj(blazed_grating_bpm)
    # system_modulation[3, :, :, :, :] = blazed_grating_bpm

    # Build the model

    class SZZBPM(nn.Module):
        r"""SZZBPM model for BPM simulation.
        input image to put on SLM:
            size: (batch, 1, 300, 300)
            dtype: torch.float32
            value: [0.0, 1.0], zero mean, unit variance

        output camera captured image, abs^2 of field:
            size: (batch, 1, 300, 300)  # Camera has same resolution as SLM
            dtype: torch.float32
            value: Not limited
        """

        def __init__(self, device=device) -> None:
            super().__init__()

            # Basic BPM parameters
            self.field_in = field_in.to(device)
            self.DFR = DFR.to(device)
            self.super_gaussian = super_gaussian.to(device)
            self.layer_sampling = torch.tensor(layer_sampling).to(device)  # tensor(int)
            self.bpm_depth = torch.tensor(bpm_depth).to(device)  # tensor(int)

            # SLM and BPM coordinate conversion
            self.ny_nx_slm2bpm = (1, *ny_nx_slm2bpm)  # Must be tuple, for 'size' in nn.functional.interpolate,  its length has to match the number of spatial dimensions; input.dim() - 2.
            self.pad_slm2bpm = pad_slm2bpm  # Must be  tuple
            self.system_modulation = system_modulation.to(device)
            self.ny_nx_bpm2cam = ny_nx_bpm2cam  # Must be  tuple
            self.crop_bpm2cam = crop_bpm2cam  # Must be tuple
            self.fast_range = fast_range.to(device)
            pi_cuda = torch.tensor(pi, dtype=torch.float32, device=device)
            # Training parameters, 0 mean, pi variance
            self.phase_scale = Parameter(pi_cuda * torch.ones((bpm_depth, 1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))
            self.phase_bias = Parameter(pi_cuda * torch.ones((bpm_depth, 1, 1, Ny_slm, Nx_slm), dtype=torch.float32, device=device))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Prepare modulation at SLM
            input = input.unsqueeze(0)  # Dim should be 5, [bpm_depth, Batch, Channel, Height, Width]
            slm_modulation_phase_slm = input * self.phase_scale + self.phase_bias
            temp = nn.functional.interpolate(slm_modulation_phase_slm, size=self.ny_nx_slm2bpm)
            slm_modulation_phase_bpm = nn.functional.pad(temp, pad=self.pad_slm2bpm, mode='constant', value=0)
            slm_modulation_bpm = torch.exp(1j*slm_modulation_phase_bpm)

            # BPM core
            u0 = self.field_in
            # For keeping the whole propagation field
            # step_count = 0
            # batch, channel, height, width = u0.shape
            # propagation = torch.zeros((batch, channel, height, width, Nz_bpm), dtype=u0.dtype, device=u0.device)
            # propagation[:, :, :, :, step_count] = u0
            for i in self.fast_range[:self.bpm_depth]:  # for i in range(bpm_depth) will be slow
                # Step 1: Modrulation at illumination or mirror
                u0 = u0 * self.system_modulation[i, :, :, :, :]

                # Step 2: BPM propagation from mirror to SLM
                for _ in self.fast_range[:self.layer_sampling]:  # for _ in range(layer_sampling) will be slow
                    u0 = torch.fft.ifft2(torch.fft.fft2(u0) * self.DFR) * self.super_gaussian
                    # step_count += 1  # For keeping the whole propagation field
                    # propagation[:, :, :, :, step_count] = u0  # For keeping the whole propagation field

                # Step 3: Modulation at SLM
                u0 = u0 * slm_modulation_bpm[i, :, :, :, :]

                # Step 4: BPM propagation SLM to mirror
                for _ in self.fast_range[:self.layer_sampling]:  # for _ in range(bpm_depth) will be slow
                    u0 = torch.fft.ifft2(torch.fft.fft2(u0) * self.DFR) * self.super_gaussian
                    # step_count += 1  # For keeping the whole propagation field
                    # propagation[:, :, :, :, step_count] = u0  # For keeping the whole propagation field

            # Capture by camera
            intensity_out_bpm = torch.square(u0.abs())
            temp = nn.functional.interpolate(intensity_out_bpm, size=self.ny_nx_bpm2cam)
            y_start, y_end, x_start, x_end = self.crop_bpm2cam
            intensity_out_cam = temp[:, :, y_start:y_end, x_start:x_end]
            # if VERBOSE:
            #     plot_field(self.field_in, title='field_in')
            #     for i in self.fast_range[:bpm_depth]:  # for i in range(bpm_depth) will be slow
            #         plot_field(self.system_modulation[i, :, :, :, :], title=f'system_modulation[{i}]')
            #         plot_field(slm_modulation_bpm[i, :, :, :, :], title=f'slm_modulation_bpm[{i}]')
            #     plot_field(intensity_out_bpm, title='intensity_out_bpm')
            #     plot_field(intensity_out_cam, title='intensity_out_cam')
            #     plot_field(propagation, title='propagation')

            return intensity_out_cam

    model_object = SZZBPM(device=device)
    return model_object


# ██████╗ ██╗   ██╗██╗██╗     ██████╗     ██████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
# ██╔══██╗██║   ██║██║██║     ██╔══██╗    ╚════██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
# ██████╔╝██║   ██║██║██║     ██║  ██║     █████╔╝    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
# ██╔══██╗██║   ██║██║██║     ██║  ██║     ╚═══██╗    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
# ██████╔╝╚██████╔╝██║███████╗██████╔╝    ██████╔╝    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
# ╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝     ╚═════╝     ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝

def build_model(bpm_color, bpm_mode, bpm_depth, bpm_width, bpm_parallel, model_feature, device=None):
    # Building BPM model
    if bpm_color == 'gray' and bpm_mode == 'bpm' and bpm_parallel == 1:
        model_bpm = build_model_bpm(bpm_depth, bpm_width, device=device)
    else:
        raise NotImplementedError(f"{bpm_color=}, {bpm_mode=}, {bpm_parallel=}\n'gray', 'rgb'\n'bpm', 'CNNpatch-bpm', 'fft-bpm', 'nothing'\n1, 3")

    # Building feature model and class model
    if model_feature == 'maxpool30-ReLU':
        pooling_number = bpm_width // 28  #

        class BPM2Feature(nn.Module):
            """ Extract feature from BPM model 
            input camera captured image, abs^2 of field:
                size: (batch, 1, 300, 300)  # Camera has same resolution as SLM
                dtype: torch.float32
                value: Not limited
            output feature:
                size: (batch, 768)  # 768 is the feature size of the model
                dtype: torch.float32
                value: Not limited
            """

            def __init__(self):
                super().__init__()
                self.layer_stack = nn.Sequential(
                    nn.MaxPool2d(kernel_size=bpm_width // 25, stride=bpm_width // 25),  # make output size batchx1x25x25
                    nn.Flatten(),
                    nn.Linear(in_features=25*25, out_features=768),  # 75=25*3, 150=25*6, 300=25*12, 600=25*24, 1200=25*48
                    nn.ReLU(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer_stack(x)
        model_feature = BPM2Feature().to(device)

        class Feature2Class(nn.Module):
            """ Extract feature from BPM model 
            input feature:
                size: (batch, 768)  # 768 is the feature size of the model
                dtype: torch.float32
                value: Not limited
            output class:
                size: (batch, 10)  # 10 is the number of classes
                dtype: torch.float32
                value: Not limited
            """

            def __init__(self):
                super().__init__()
                self.linear_layer = nn.Linear(768, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear_layer(x)
        model_classifier = Feature2Class().to(device)
    else:
        raise NotImplementedError(f"{model_feature=}\n'maxpool30-ReLU', 'CNN-ReLU', 'rearange', 'nothing'")

    return model_bpm, model_feature, model_classifier


if __name__ == '__main__':
    # Add parent directory to sys.path temporarily, so this file can be run directly as main
    from pathlib import Path
    import sys
    file = Path(__file__).resolve()
    sys.path.append(str(Path(file).resolve().parent.parent))

    check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    bpm_depth = 4
    bpm_width = 120
    layer_sampling = 6
    model_bpm = build_model_bpm(bpm_depth, bpm_width, layer_sampling, device=device)
    image_in = torch.ones(1, 1, bpm_width, bpm_width, dtype=torch.float32, device=device)  # [batch, channel, height, width]

    camera_out = model_bpm(image_in)
# %%
