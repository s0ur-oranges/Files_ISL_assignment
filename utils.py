import numpy as np
import logging
import warnings
import gmm

warnings.simplefilter("error",RuntimeWarning)

def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
def compute_vad(s, win_length=320, win_overlap=160, n_realignment=5, threshold=0.3):
    # power signal for energy computation
    s = s**2
    # frame signal with overlap
    F = framing(s, win_length, win_length - win_overlap) 
    # sum frames to get energy
    E = F.sum(axis=1).astype(np.float64)

    # normalize the energy
    E -= E.mean()
    try:
        E /= E.std()
    # initialization
        mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
        ee = np.array(( 1.00, 1.00, 1.00))[:, np.newaxis]
        ww = np.array(( 0.33, 0.33, 0.33))

        GMM = gmm.gmm_eval_prep(ww, mm, ee)

        E = E[:,np.newaxis]

        for i in range(n_realignment):
        # collect GMM statistics
            llh, N, F, S = gmm.gmm_eval(E, GMM, return_accums=2)

        # update model
            ww, mm, ee   = gmm.gmm_update(N, F, S)
        # wrap model
            GMM = gmm.gmm_eval_prep(ww, mm, ee)

    # evaluate the gmm llhs
        llhs = gmm.gmm_llhs(E, GMM)

        llh  = gmm.logsumexp(llhs, axis=1)[:, np.newaxis]

        llhs = np.exp(llhs - llh)

        out  = np.zeros(llhs.shape[0], dtype=np.bool)
        out[llhs[:,0] < threshold] = True
    except RuntimeWarning:
        logging.info("File contains only silence")
        out=np.zeros(E.shape[0],dtype=np.bool)

    return out
