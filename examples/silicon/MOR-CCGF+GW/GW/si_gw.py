from pyscf.pbc import gto, dft, scf, df, cc
from pyscf.pbc.lib import chkfile
import os, h5py, pickle
import numpy as np
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = '''
            0.000000     2.715000     2.715000
            2.715000     0.000000     2.715000
            2.715000     2.715000     0.000000
           ''',
           atom = 'Si 2.03625 2.03625 2.03625; Si 3.39375 3.39375 3.39375',
           dimension = 3,
           max_memory = 16000,
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
omegas = np.array([0.])
gf_gw, gf0, sigma_gw = gw.kernel(omega=omegas, writefile=1, nw=200)

# KGW natural orbitals
# TZ NOTE: change nocc_act and nvir_act
nocc_act = 4
nvir_act = 10
from kccgf.cas.casno import make_casno
mf_cas, no_coeff_k, fock_cas, dm_cas, frozen, no_coeff_full, dm = \
                make_casno(gw, thresh=1e-3, nvir_act=nvir_act, nocc_act=nocc_act, vno_only=False, return_dm=True)

