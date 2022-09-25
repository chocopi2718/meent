# import numpy as np
# from numpy.linalg import inv
import autograd.numpy as np
# from autograd.numpy.linalg import inv
from autograd.numpy.linalg import grad_inv

def P_Q_kz(Kx, Ky, e_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i):
    '''
    r is for relative so do not put epsilon_0 or mu_0 here
    :param Kx: NM x NM matrix
    :param Ky:
    :param e_conv: (NM x NM) conv matrix
    :param mu_r:
    :return:
    '''
    argument = e_conv - Kx ** 2 - Ky ** 2
    Kz = np.conj(np.sqrt(argument.astype('complex')))
    # Kz = np.sqrt(argument.astype('complex'))  # TODO: conjugate?

    # TODO: confirm whether oneonver_E_conv is indeed not used
    # TODO: Check sign of P and Q
    P = np.block([
        [Kx @ E_i @ Ky, -Kx @ E_i @ Kx + mu_conv],
        [Ky @ E_i @ Ky - mu_conv,  -Ky @ E_i @ Kx]
    ])

    Q = np.block([
        [Kx @ grad_inv(mu_conv) @ Ky, -Kx @ grad_inv(mu_conv) @ Kx + e_conv],
        [-oneover_E_conv_i + Ky @ grad_inv(mu_conv) @ Ky, -Ky @ grad_inv(mu_conv) @ Kx]
    ])

    return P, Q, Kz
