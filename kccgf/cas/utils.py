import numpy as np

def get_g0_k(omega, mo_energy, eta):
    '''Get mean-field GF'''
    nkpts = len(mo_energy)
    nmo = mo_energy[0].shape[0]
    nw = len(omega)
    gf0 = np.zeros((nkpts,nmo,nmo,nw),dtype=np.complex128)
    for k in range(nkpts):
        for iw in range(nw):
            gf0[k,:,:,iw] = np.diag(1.0/(omega[iw]+1j*eta - mo_energy[k]))
    return gf0

def get_sigma(mf_gf, corr_gf):
    '''Get self-energy from correlated GF'''
    nkpts, nmo, nmo, nw = mf_gf.shape
    sigma = np.zeros_like(mf_gf)
    for k in range(nkpts):
        for iw in range(nw):
            sigma[k,:,:,iw] = np.linalg.inv(mf_gf[k,:,:,iw]) - np.linalg.inv(corr_gf[k,:,:,iw])
    return sigma

def get_corrg_k(mf_gf, sigma):
    '''Get correlated GF from mean-field GF and self-energy'''
    nkpts, nmo, nmo, nw = mf_gf.shape
    corr_gf = np.zeros_like(mf_gf)
    for k in range(nkpts):
        for iw in range(nw):
            corr_gf[k,:,:,iw] = np.linalg.inv(np.linalg.inv(mf_gf[k,:,:,iw]) - sigma[k,:,:,iw])
    return corr_gf

def fit_qp_peak(freqs, dos):
    '''Fit quasiparticle peak energy from given density of states'''
    from scipy.optimize import curve_fit
    freqs = np.array(freqs)
    dos = np.array(dos)
    assert(dos.shape[0] > 3)

    # Lorentzian function
    def lorentzian(x, mo_energy, eta, A):
        y = A/np.pi * eta / ((x - mo_energy)**2 + eta**2)
        return y

    popt = curve_fit(lorentzian, freqs, dos)[0]
    mo_energy = popt[0]
    return mo_energy
