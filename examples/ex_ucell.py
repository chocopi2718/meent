import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from meent.rcwa import call_solver

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

# common
# grating_type = 1  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0
phi = 0
psi = 0 if pol else 90

wavelength = 900

thickness = [500]
ucell_materials = [1, 3.48]
period = [100, 100]
fourier_order = 15
mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', 3: 'numpy_integ', 4: 'JAX_integ',}
n_iter = 3


def run_test(grating_type, mode_key, dtype, device):

    ucell = load_ucell(grating_type)

    if mode_key == 0:
        device = None

        if dtype == 0:
            type_complex = np.complex128
        else:
            type_complex = np.complex64

    elif mode_key == 1:
        # JAX
        if device == 0:
            jax.config.update('jax_platform_name', 'cpu')
        else:
            jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            from jax.config import config
            config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
        else:
            type_complex = jnp.complex64

    else:
        # Torch
        if device == 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        if dtype == 0:
            type_complex = torch.complex128
        else:
            type_complex = torch.complex64

    AA = call_solver(mode=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                     ucell_materials=ucell_materials,
                     thickness=thickness, device=device, type_complex=type_complex, )
    # AA = call_solver(mode_key, grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta,
    #                  phi=phi, psi=psi, pol=pol, fourier_order=fourier_order,
    #              period=period,
    #              wavelength=wavelength, ucell=ucell, ucell_materials=ucell_materials,
    #              thickness=thickness, algo='TMM', perturbation=1E-10,
    #              device='cpu', type_complex=jnp.complex128)
    # AA= call_solver()

    for i in range(n_iter):
        t0 = time.time()
        de_ri, de_ti = AA.run_ucell()
        print(f'run_cell: {i}: ', time.time()-t0)

    resolution = (20, 20, 20)
    for i in range(0):
        t0 = time.time()
        AA.calculate_field(resolution=resolution, plot=False)
        print(f'cal_field: {i}', time.time() - t0)

    return de_ri, de_ti


def run_loop(a, b, c, d):
    for grating_type in a:
        for bd in b:
            for dtype in c:
                for device in d:
                    run_test(grating_type, bd, dtype, device)

                    # try:
                    #     print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
                    #     run_test(grating_type, bd, dtype, device)
                    # except Exception as e:
                    #     print(e)


def load_ucell(grating_type):

    if grating_type in [0, 1]:

        ucell = np.array([

            [
                [
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                ],
            ],
            # [
            #     [
            #         0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            #     ],
            # ],
        ])
    else:

        ucell = jnp.array([
            # [
            #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0,],
            #     [0, 0, 0, 1, 1, 0, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [1, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            # ],

            # [
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
            #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
            #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
            #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            # ],

            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
        ])

        # ucell = np.array([
        #
        #     [
        #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        #     ],
        # ])
    return ucell


if __name__ == '__main__':
    t0 = time.time()
    run_loop([2], [1], [1], [1])
    print('jit on', time.time() - t0)
