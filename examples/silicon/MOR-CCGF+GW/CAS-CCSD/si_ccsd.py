from pyscf.pbc import gto, dft, scf, df, cc
from pyscf.pbc.lib import chkfile
import os, h5py, pickle
import numpy as np

cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = '''
            0.000000     2.715000     2.715000
            2.715000     0.000000     2.715000
            2.715000     2.715000     0.000000
           ''',
           atom = 'Si 2.03625 2.03625 2.03625; Si 3.39375 3.39375 3.39375',
           dimension = 3,
           max_memory = 45000,
           verbose = 5,
           basis='gth-dzvp',
           pseudo='gth-hf-rev',
           precision=1e-12)

kpts = cell.make_kpts([2,2,2])
gdf = df.GDF(cell, kpts)
# TZ NOTE: change this to your own gdf path
gdf_fname = '../../HF/gdf_ints_222.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

# TZ NOTE: change this to your own kmf chkfile path
chkfname = '../../HF/si_222.chk'
if os.path.isfile(chkfname):
    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.exxdiv = 'ewald'
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.exxdiv = 'ewald'
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.kernel()

from fcdmft.gw.pbc import krgw_gf
gw = krgw_gf.KRGWGF(kmf)
gw.eta = 0.1/27.211386
gw.fullsigma = True
gw.fc = False
gw.rdm = True
gw.load_sigma = True
omegas = np.array([0.])
assert os.path.isfile('vxc.h5')
assert os.path.isfile('sigma_imag.h5')
gf_gw, gf0, sigma_gw = gw.kernel(omega=omegas, writefile=0, nw=200)

# KGW natural orbitals
# TZ NOTE: change nocc_act and nvir_act
nocc_act = 4
nvir_act = 10
from kccgf.cas.casno import make_casno
mf_cas, no_coeff_k, fock_cas, dm_cas, frozen, no_coeff_full, dm = \
                make_casno(gw, thresh=1e-3, nvir_act=nvir_act, nocc_act=nocc_act, vno_only=False, return_dm=True)

# save or read mf_cas
# TZ NOTE: change this to your own CCSD tmpfile folder
filepath = './'
cas_fname = filepath+'mf_cas.h5'
if os.path.isfile(cas_fname):
    fn = cas_fname
    feri = h5py.File(fn, 'r')
    mf_cas.mo_coeff = np.asarray(feri['mo_coeff'])
    mf_cas.mo_occ = np.asarray(feri['mo_occ'])
    mf_cas.mo_energy = np.asarray(feri['mo_energy'])
    no_coeff_k = np.asarray(feri['no_coeff_k'])
    no_coeff_full = np.asarray(feri['no_coeff_full'])
    h1e = np.asarray(feri['h1e'])
    feri.close()
    mf_cas.get_hcore = lambda *args: h1e
else:
    fn = cas_fname
    feri = h5py.File(fn, 'w')
    feri['mo_coeff'] = np.asarray(mf_cas.mo_coeff)
    feri['mo_occ'] = np.asarray(mf_cas.mo_occ)
    feri['mo_energy'] = np.asarray(mf_cas.mo_energy)
    feri['h1e'] = np.asarray(mf_cas.get_hcore())
    feri['no_coeff_k'] = np.asarray(no_coeff_k)
    feri['no_coeff_full'] = np.asarray(no_coeff_full)
    feri.close()

# KCCSD with frozen orbitals
mycc = cc.KRCCSD(kmf, frozen=frozen, mo_coeff=no_coeff_full)
mycc.conv_tol = 1e-8
mycc.conv_tol_normt = 1e-5
mycc.eris = None
amp_fname = filepath+'amplitudes.h5'
eris_fname = filepath+'ERIS'
if os.path.isfile(amp_fname) and os.path.isfile(eris_fname):
    feri = h5py.File(amp_fname, 'r')
    mycc.t1 = np.asarray(feri['t1'])
    mycc.t2 = np.asarray(feri['t2'])
    feri.close()
    fn2 = open(eris_fname, 'rb')
    mycc.eris = pickle.load(fn2)
    fn2.close()
else:
    mycc.kernel()
    feri = h5py.File(amp_fname, 'w')
    feri['t1'] = np.asarray(mycc.t1)
    feri['t2'] = np.asarray(mycc.t2)
    feri.close()
    fn2 = open(eris_fname, 'wb')
    pickle.dump(mycc.eris, fn2)
    fn2.close()
