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
gdf_fname = '../HF/gdf_ints_222.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

# TZ NOTE: change this to your own kmf chkfile path
chkfname = '../HF/si_222.chk'
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

# TZ NOTE: change this to your own CCSD tmpfile folder
filepath = './'
# KCCSD
mycc = cc.KRCCSD(kmf)
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
