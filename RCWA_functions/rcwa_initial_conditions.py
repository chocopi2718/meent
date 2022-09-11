import numpy as np

def delta_vector(P, Q):
    '''
        create a vector with a 1 corresponding to the 0th order
        #input P = 2*(num_ord_specified)+1
    '''
    fourier_grid = np.zeros((P,Q))
    fourier_grid[int(P/2), int(Q/2)] = 1;
    # vector = np.zeros((P*Q,));
    #
    # #the index of the (0,0) element requires a conversion using sub2ind
    # index = int(P/2)*P + int(Q/2);
    vector = fourier_grid.flatten();
    return np.matrix(np.reshape(vector, (1,len(vector))));

def delta_vector_1D(P):
    '''
        create a vector with a 1 corresponding to the 0th order
    '''
    vector = np.zeros((P,));

    #the index of the (0,0) element requires a conversion using sub2ind
    index = int(P/2);
    vector[index] = 1
    return vector;

def initial_conditions_1D(K_inc_vector, theta, P):
    '''
    K_inc points only in X and Z plane, so theta is the only specifying angle
    :param K_inc_vector:
    :param theta:
    :param P:
    :return:
    '''
    num_ord = 2*P+1;
    delta = delta_vector_1D(num_ord);
    cinc = delta


def initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, P, Q):
    '''
    :param K_inc_vector: whether it's normalized or not is not important...
    :param theta: angle of incience
    :param ate_vector:
    :param normal_vector: pointing into z direction
    :param pte: te polarization amplitude
    :param ptm: tm polarization amplitude
    :return:
    calculates the incident E field, cinc, and the polarization fro the initial condition vectors
    '''
    #ate -> unit vector holding the out of plane direction of TE
    #atm -> unit vector holding the out of plane direction of TM
    # what are the out of plane components...(Ey and Hy)
    # normal_vector = [0,0,-1]; i.e. waves propagate down into the -z direction
    # cinc = Wr^-1@[Ex_inc, Ey_inc];

    if (theta != 0):
        ate_vector = np.cross(K_inc_vector, normal_vector);
        ate_vector = ate_vector / (np.linalg.norm(ate_vector));
    else:
        ate_vector = np.array([0,1,0]);

    atm_vector = np.cross(ate_vector, K_inc_vector);
    atm_vector = atm_vector / (np.linalg.norm(atm_vector))

    Polarization = pte * ate_vector + ptm * atm_vector; #total E_field incident which is a 3 component vector (ex, ey, ez)
    E_inc = Polarization;
    # go from mode coefficients to FIELDS
    Polarization = np.squeeze(np.array(Polarization)); #polarization vector holds amplitudes for ALL E-FIELDS
    delta = delta_vector(2*P+1,2*Q+1);

    #cinc; #remember we ultimately solve for [Ex, Ey, Hx, Hy].
    esrc = np.hstack((Polarization[0]*delta, Polarization[1]*delta));
    esrc = np.matrix(esrc).T; #mode amplitudes of Ex, and Ey

    return E_inc, esrc, Polarization
