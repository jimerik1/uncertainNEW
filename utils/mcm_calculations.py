"""
Minimum Curvature Method calculations
"""
import numpy as np


def get_inc(inc1, DL, MD2, MD1, TF, t0, t):
    """Calculate inclination at parameter t"""
    M = lambda s: MD1 + s*(MD2-MD1)
    a = np.cos(inc1)*np.cos(DL*(M(t)-M(t0))) - np.sin(inc1)*np.sin(DL*(M(t)-M(t0)))*np.cos(TF)
    Inc = np.arccos(min(1, a))
    return Inc


def get_azi(inc1, azi1, DL, MD2, MD1, TF, t0, t):
    """Calculate azimuth at parameter t"""
    tol = 1e-6
    M = lambda s: MD1 + s*(MD2-MD1)
    if DL == 0:
        arg_y = 0
        arg_x = np.sin(inc1)
    elif np.abs(DL*(M(t) - M(t0)) - np.pi/2) < tol:
        arg_y = np.sin(TF)
        arg_x = np.cos(inc1)*np.cos(TF)
    else: 
        arg_y = np.sin(TF)*np.sin(DL*(M(t)-M(t0)))
        arg_x = (np.sin(inc1)*np.cos(DL*(M(t)-M(t0))) + np.cos(inc1)*np.cos(TF)*np.sin(DL*(M(t)-M(t0))))
    Az_out = np.mod(azi1 + np.arctan2(arg_y, arg_x), 2*np.pi)
    return Az_out


def get_TF(inc1, inc2, azi1, azi2, theta):
    """Calculate toolface angle"""
    arg_x = np.sin(inc1)*np.sin(inc2)*np.sin(azi2 - azi1)
    arg_y = np.cos(inc1)*np.cos(theta) - np.cos(inc2)
    TF = np.arctan2(arg_x, arg_y)
    return TF


def mcm(Inc1, Azi1, MD1, Inc2, Azi2, MD2, t, N0=0, E0=0, V0=0):
    """
    Minimum Curvature Method calculation
    
    Args:
        Inc1, Inc2: Inclinations in radians
        Azi1, Azi2: Azimuths in radians  
        MD1, MD2: Measured depths
        t: Parameter (0 to 1) along the curve
        N0, E0, V0: Starting coordinates
        
    Returns:
        Tuple of (coordinates, azimuth, inclination) at parameter t
    """
    tol = 1e-6
    inc1, inc2 = Inc1, Inc2
    azi1, azi2 = Azi1, Azi2
    dM = MD2 - MD1
    
    if dM == 0:
        return np.array([N0, E0, V0]), azi1, inc1
        
    M = lambda t: t*dM
    
    # Calculate Dogleg and dogleg severity of curve from MD1 to MD2
    dN1, dN2 = np.sin(inc1)*np.cos(azi1), np.sin(inc2)*np.cos(azi2)
    dE1, dE2 = np.sin(inc1)*np.sin(azi1), np.sin(inc2)*np.sin(azi2)
    dV1, dV2 = np.cos(inc1), np.cos(inc2)
    DL = np.arccos(np.minimum(dN1*dN2 + dE1*dE2 + dV1*dV2, 1))
    beta = DL/dM
    
    # Calculate toolface
    if inc1 == 0:
        TF = azi2 - azi1
    else:
        arg_y = np.sin(inc1)*np.sin(inc2)*np.sin(azi2 - azi1)
        arg_x = np.cos(inc1)*np.cos(DL) - np.cos(inc2)
        TF = np.arctan2(arg_y, arg_x)
    
    # Calculate inc and azi at t
    inc_t = get_inc(inc1, beta, MD2, MD1, TF, 0, t)
    azi_t = get_azi(inc1, azi1, beta, MD2, MD1, TF, 0, t)
    
    # Unit vectors at t
    dN_t = np.sin(inc_t)*np.cos(azi_t)
    dE_t = np.sin(inc_t)*np.sin(azi_t)
    dV_t = np.cos(inc_t)
    
    # Calculate coordinates
    if beta == 0:
        S = M(t)/2
    else:
        S = 1/beta*np.tan(beta/2*M(t))
    
    dN = S*(dN1 + dN_t)
    dE = S*(dE1 + dE_t)
    dV = S*(dV1 + dV_t)
    
    out = np.array([N0+dN, E0+dE, V0+dV])
    
    return out, azi_t, inc_t


def verify_mcm():
    """
    Verification test case
    Start coordinates: (0,0,0)
    Input:
        i) Inc1, Inc2 = 15, 45 (degrees)
        ii) Azi1, Azi2 = 40, 170 (degrees)
        iii) MD1, MD2 = 0, 500
    
    Expected end coordinates: (-135.3, 78.54, 454.46)
    """
    # Convert degrees to radians
    inc1, inc2 = np.radians(15), np.radians(45)
    azi1, azi2 = np.radians(40), np.radians(170)
    
    result = mcm(inc1, azi1, 0, inc2, azi2, 500, 1.0)
    coords = result[0]
    
    expected = np.array([-135.3, 78.54, 454.46])
    print(f'Calculated: N={coords[0]:.2f}, E={coords[1]:.2f}, V={coords[2]:.2f}')
    print(f'Expected:   N={expected[0]:.2f}, E={expected[1]:.2f}, V={expected[2]:.2f}')
    
    return np.allclose(coords, expected, atol=1.0)


if __name__ == '__main__':
    verify_mcm()