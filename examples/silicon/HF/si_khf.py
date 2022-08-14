from pyscf.pbc import gto, dft, scf, df
from pyscf.pbc.lib import chkfile
import os
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
           max_memory = 32000,
           verbose = 5,
           basis='gth-dzvp',
           pseudo='gth-hf-rev',
           precision=1e-12)

kpts = cell.make_kpts([2,2,2])
gdf = df.GDF(cell, kpts)
gdf_fname = 'gdf_ints_222.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'si_222.chk'
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

