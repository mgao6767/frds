import numpy as np


def fftsmooth(x, w):
    """Fast Fourier Transform-based smoothing on a given column vector x using a specified window size w.

    This function is translated from Nagel's Matlab code `fftsmooth.m`

    NOTE: so far this function works only for even number of obs and even number of kernel window elements

    Args:
        x (np.ndarray): column vector of inputs to be smoothed
        w (int): window size (in # of obs)
    """
    n = x.shape[0]
    assert n % 2 == 0
    assert w % 2 == 0
    xpad = np.concatenate((np.zeros((n,)), x))
    # xpad = xpad.reshape(xpad.shape[0], 1)
    k = np.concatenate(
        (np.zeros((n - w // 2,)), np.ones((w,)) / w, np.zeros((n - w // 2,)))
    )
    Fx = np.fft.fft(xpad)
    Fk = np.fft.fft(k)
    Fxk = np.multiply(Fx, Fk)
    xk = np.fft.ifft(Fxk)

    xk = np.real(xk).squeeze()

    # Extrapolate linearly at edges
    dl = xk[w // 2 + 1] - xk[w // 2]  # / (w - 1)
    lex = xk[w // 2] - np.flip(np.arange(w // 2) + 1) * dl
    ul = xk[n - w // 2 - 1] - xk[n - w // 2 - 2]  # / (w - 1)
    uex = xk[n - w // 2 - 1] + np.arange(1, w // 2 + 1) * ul

    xout = np.concatenate((lex, xk[w // 2 : n - w // 2], uex))
    return xout
