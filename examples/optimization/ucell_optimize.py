try:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
except:
    pass

import numpy as np
import jax
import jax.numpy as jnp
import time

from meent.rcwa import call_solver
import torch


class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass

    def get_difference(self):
        spectrum_gt = jnp.hstack(self.gt.spectrum_R, self.gt.spectrum_T)
        spectrum_model = jnp.hstack(self.model.spectrum_R, self.model.spectrum_T)
        residue = spectrum_model - spectrum_gt
        loss = jnp.linalg.norm(residue)


def load_setting(mode_key, dtype, device):
    grating_type = 2

    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wavelength = 900

    thickness = [1120]
    ucell_materials = [1, 3.48]
    period = [1000, 1000]
    fourier_order = 15

    ucell = np.array(
        [[
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
        ]]
    )

    if mode_key == 0:
        device = 0
        type_complex = np.complex128 if dtype == 0 else np.complex64
        ucell = ucell.astype(type_complex)

    elif mode_key == 1:  # JAX
        ucell = jnp.array(ucell)
        jax.config.update('jax_platform_name', 'cpu') if device == 0 else jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            from jax.config import config
            config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
            ucell = ucell.astype(type_complex)
        else:
            type_complex = jnp.complex64
            ucell = ucell.astype(type_complex)

    else:  # Torch
        device = torch.device('cpu') if device == 0 else torch.device('cuda')
        type_complex = torch.complex128 if dtype == 0 else torch.complex64
        ucell = torch.tensor(ucell, dtype=type_complex)

    return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order,\
           type_complex, device, ucell


def compare_conv_mat_method(mode_key, dtype, device):
    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell \
        = load_setting(mode_key, dtype, device)

    if mode_key == 0:
        from meent.on_numpy.convolution_matrix import to_conv_mat as conv1
        from meent.on_numpy.convolution_matrix import to_conv_mat_piecewise_constant as conv2

    elif mode_key == 1:
        from meent.on_jax.convolution_matrix import to_conv_mat as conv1
        from meent.on_jax.convolution_matrix import to_conv_mat_piecewise_constant as conv2
    else:
        from meent.on_torch.convolution_matrix import to_conv_mat as conv1
        from meent.on_torch.convolution_matrix import to_conv_mat_piecewise_constant as conv2

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )
    t0 = time.time()
    E_conv_all = conv1(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = conv1(1 / ucell, fourier_order, type_complex=type_complex)
    de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)
    print(time.time() - t0)

    t0 = time.time()
    E_conv_all = conv1(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = conv1(1 / ucell, fourier_order, type_complex=type_complex)
    de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)
    print(time.time() - t0)

    t0=time.time()
    solver.conv_solve(ucell)
    print(time.time() - t0)

    t0=time.time()
    solver.conv_solve(ucell)
    print(time.time() - t0)

    E_conv_all1 = conv2(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all1 = conv2(1 / ucell, fourier_order, type_complex=type_complex)
    de_ri1, de_ti1 = solver.solve(wavelength, E_conv_all1, o_E_conv_all1)

    try:
        print('de_ri norm: ', np.linalg.norm(de_ri - de_ri1))
        print('de_ti norm: ', np.linalg.norm(de_ti - de_ti1))
    except:
        print('de_ri norm: ', torch.linalg.norm(de_ri - de_ri1))
        print('de_ti norm: ', torch.linalg.norm(de_ti - de_ti1))

    return


def optimize_jax(mode_key, dtype, device):
    from meent.on_jax.convolution_matrix import to_conv_mat

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )

    @jax.grad
    def grad_loss(ucell):

        E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
        de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        res = -de_ti[3,2]
        print(res)
        return res

    print('grad:', grad_loss(ucell))

    def mingd(x):
        lr = 0.01
        gd = grad_loss(x)

        res = x - lr*gd*x
        return res

    # Recurrent loop of gradient descent
    for i in range(1):
        # ucell = vfungd(ucell)
        ucell = mingd(ucell)

    print(ucell)


def optimize_torch(mode_key, dtype, device):
    from meent.on_torch.convolution_matrix import to_conv_mat

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)

    ucell.requires_grad = True

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )

    opt = torch.optim.SGD([ucell], lr=1E-2)
    for i in range(1):
        E_conv_all = to_conv_mat(ucell, fourier_order)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order)
        de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        loss = -de_ti[3, 2]
        loss.backward()
        print(ucell.grad)
        opt.step()
        opt.zero_grad()
        print(loss)

    print(ucell)


if __name__ == '__main__':
    t0 = time.time()

    dtype = 1
    device = 0

    # compare_conv_mat_method(mode_key=0, dtype=dtype, device=device)
    compare_conv_mat_method(1, dtype=dtype, device=device)
    compare_conv_mat_method(2, dtype=dtype, device=device)

    mode_key = 1
    device = 0
    dtype = 0

    optimize_jax(1, 0, 0)
    optimize_torch(2, 0, 0)

