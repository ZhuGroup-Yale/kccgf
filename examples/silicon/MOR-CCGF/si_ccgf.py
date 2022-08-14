from pyscf.pbc import gto, dft, scf, df, cc, mp
from pyscf.pbc.lib import chkfile
import os, h5py, pickle
import numpy as np
from sys import argv
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def run_ccgf(orb, kpt_idx):
    # Cell parameters, basis set, max_memory
    cell = gto.Cell()
    cell.build(unit = 'angstrom',
               a = '''
                0.000000     2.715000     2.715000
                2.715000     0.000000     2.715000
                2.715000     2.715000     0.000000
               ''',
               atom = 'Si 2.03625 2.03625 2.03625; Si 3.39375 3.39375 3.39375',
               dimension = 3,
               max_memory = 24000,
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

    # TZ NOTE: change this to your own CCSD tmpfile folder
    filepath = '../../CCSD/'
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
        comm.Barrier()
    else:
        if rank == 0:
            mycc.kernel()
            feri = h5py.File(amp_fname, 'w')
            feri['t1'] = np.asarray(mycc.t1)
            feri['t2'] = np.asarray(mycc.t2)
            feri.close()
            fn2 = open(eris_fname, 'wb')
            pickle.dump(mycc.eris, fn2)
            fn2.close()
        comm.Barrier()
        if rank > 0:
            feri = h5py.File(amp_fname, 'r')
            mycc.t1 = np.asarray(feri['t1'])
            mycc.t2 = np.asarray(feri['t2'])
            feri.close()
            fn2 = open(eris_fname, 'rb')
            mycc.eris = pickle.load(fn2)
            fn2.close()
        comm.Barrier()

    # KCCGF with frozen orbitals
    from kccgf import krccsd_gf_mpi
    myccgf = krccsd_gf_mpi.KRCCGF(mycc)
    myccgf.eta = 0.1/27.211386
    myccgf.conv_tol = 1e-3
    myccgf.max_iter = 200
    myccgf.m = 200
    kptlist = [kpt_idx]
    mo_orbs = [orb]
    mo_orbs_full = range(len(kmf.mo_energy[0]))
    # TZ NOTE: change omegas here
    # N_MOR = 8 in this test
    omegas = np.linspace(-6./27.211386,26./27.211386,321)
    freqs_mor_ip = np.linspace(-6./27.211386,10./27.211386,8)
    freqs_mor_ea = np.linspace(10./27.211386,26./27.211386,8)
    gf_ip = myccgf.solve_ip(kptlist, mo_orbs, mo_orbs_full, omegas.conj(), MOR=True, omega_mor=freqs_mor_ip).conj()
    gf_ea = myccgf.solve_ea(kptlist, mo_orbs_full, mo_orbs, omegas, MOR=True, omega_mor=freqs_mor_ea)

    # save KCCGF
    if rank == 0:
        fn = 'kccgf.h5'
        feri = h5py.File(fn, 'w')
        feri['gf_ip'] = np.asarray(gf_ip)
        feri['gf_ea'] = np.asarray(gf_ea)
        feri['omegas'] = np.asarray(omegas)
        feri['eta'] = np.asarray(myccgf.eta)
        feri.close()
    comm.Barrier()

    #if rank == 0:
    #    print ('KCCGF DOS IP/EA')
    #    for i in range(len(omegas)):
    #        print (omegas[i], -gf_ip[0,0,0,i].imag/np.pi, -gf_ea[0,0,0,i].imag/np.pi)


orb = int(argv[1])
kpt_idx = int(argv[2])
run_ccgf(orb, kpt_idx)
