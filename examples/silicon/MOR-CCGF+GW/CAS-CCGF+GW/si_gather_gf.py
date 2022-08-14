from pyscf.pbc import gto, dft, scf, df, cc, mp
from pyscf.pbc.lib import chkfile
import os, h5py, pickle
import numpy as np
from sys import argv
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def run_ccgf(kpt_idx, nocc_act=None, nvir_act=None):
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
               max_memory = 28000,
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

    # read computed CCGF
    nmo = nocc_act + nvir_act
    for orb in range(nmo):
        filepath = 'orb%d-kpt%d/'%(orb, kpt_idx)
        fn = filepath+'kccgf.h5'
        feri = h5py.File(fn, 'r')
        gf_ip = np.array(feri['gf_ip'])
        gf_ea = np.array(feri['gf_ea'])
        omegas = np.array(feri['omegas'])
        eta = np.array(feri['eta'])
        feri.close()
        if orb == 0:
            gf_ip_gather = np.zeros((1,nmo,nmo,len(omegas)), dtype=complex)
            gf_ea_gather = np.zeros((1,nmo,nmo,len(omegas)), dtype=complex)

        gf_ip_gather[0,orb,:] = gf_ip[0,0,:]
        gf_ea_gather[0,:,orb] = gf_ea[0,:,0]

    gf_cc = gf_ip_gather + gf_ea_gather
    if rank == 0:
        fn = 'kccgf_gather.h5'
        feri = h5py.File(fn, 'w')
        feri['gf_ip'] = np.asarray(gf_ip_gather)
        feri['gf_ea'] = np.asarray(gf_ea_gather)
        feri['omegas'] = np.asarray(omegas)
        feri['eta'] = np.asarray(eta)
        feri.close()
    comm.Barrier()

    from fcdmft.gw.pbc import krgw_gf
    gw = krgw_gf.KRGWGF(kmf)
    gw.eta = 0.1/27.211386
    gw.fullsigma = True
    gw.fc = False
    gw.rdm = True
    gw.load_sigma = True
    assert os.path.isfile('vxc.h5')
    assert os.path.isfile('sigma_imag.h5')
    gf_gw, gf0, sigma_gw = gw.kernel(omega=omegas, writefile=0, nw=200)

    # KGW natural orbitals
    from kccgf.cas.casno import make_casno
    mf_cas, no_coeff_k, fock_cas, dm_cas, frozen, no_coeff_full, dm = \
                    make_casno(gw, thresh=1e-3, nvir_act=nvir_act, nocc_act=nocc_act, vno_only=False, return_dm=True)

    # save or read mf_cas
    # TZ NOTE: change this to your own CCSD tmpfile folder
    filepath = '../CAS-CCSD/'
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
        if rank == 0:
            fn = cas_fname
            feri = h5py.File(fn, 'w')
            feri['mo_coeff'] = np.asarray(mf_cas.mo_coeff)
            feri['mo_occ'] = np.asarray(mf_cas.mo_occ)
            feri['mo_energy'] = np.asarray(mf_cas.mo_energy)
            feri['h1e'] = np.asarray(mf_cas.get_hcore())
            feri['no_coeff_k'] = np.asarray(no_coeff_k)
            feri['no_coeff_full'] = np.asarray(no_coeff_full)
            feri.close()
        comm.Barrier()

    # CAS-GW
    from kccgf.cas import krgw_gf_cas
    mf_cas.with_df = kmf.with_df
    mf_cas.with_df._cderi = kmf.with_df._cderi
    gw_mo_coeff = np.zeros_like(no_coeff_k)
    if rank == 0:
        for k in range(gw.nkpts):
            gw_mo_coeff[k] = np.dot(no_coeff_k[k], mf_cas.mo_coeff[k])
    comm.Barrier()
    gw_mo_coeff = comm.bcast(gw_mo_coeff, root=0)

    gw_cas = krgw_gf_cas.KRGWGF(mf_cas)
    gw_cas.eta = 0.1/27.211386
    gw_cas.fullsigma = True
    gw_cas.fc = False
    gw_cas.rdm = True
    gw_cas.verbose = 5
    gf_gw_cas, gf0_cas, sigma_gw_cas = gw_cas.kernel(omega=omegas, mo_coeff=gw_mo_coeff, writefile=0, nw=200)

    if rank == 0:
        print ('Full-KGW')
        for i in range(len(omegas)):
            print (omegas[i], -np.trace(gf_gw[kpt_idx,:,:,i].imag)/np.pi)

        print ('CAS-KGW')
        for i in range(len(omegas)):
            print (omegas[i], -np.trace(gf_gw_cas[kpt_idx,:,:,i].imag)/np.pi)

        print ('CAS-KCCGF')
        for i in range(len(omegas)):
            print (omegas[i], -np.trace(gf_ip_gather[0,:,:,i].imag)/np.pi, -np.trace(gf_ea_gather[0,:,:,i].imag)/np.pi)

    from kccgf.cas import utils
    # compute CCSD self-energy in CAS space
    kptlist = [kpt_idx]
    gf_hf = utils.get_g0_k(omegas, kmf.mo_energy, eta)
    gf_hf_cas_mo = utils.get_g0_k(omegas, mf_cas.mo_energy, eta)
    gf_hf_cas_no = np.zeros_like(gf_cc)
    for ik, k in enumerate(kptlist):
        for iw in range(len(omegas)):
            gf_hf_cas_no[ik,:,:,iw] = np.dot(mf_cas.mo_coeff[k], gf_hf_cas_mo[k,:,:,iw]).dot(mf_cas.mo_coeff[k].T.conj())
    sigma_cas = utils.get_sigma(gf_hf_cas_no, gf_cc)

    # transform self-energy to full space and get CC+HF GF
    _, nmo, nmo, nw = gf_hf.shape
    sigma = np.zeros((len(kptlist), nmo, nmo, nw), dtype=complex)
    ovlp = kmf.get_ovlp()
    for ik, k in enumerate(kptlist):
        CSC = np.dot(kmf.mo_coeff[k].T.conj(), ovlp[k]).dot(no_coeff_k[k])
        for iw in range(len(omegas)):
            sigma[ik,:,:,iw] = np.dot(CSC, sigma_cas[ik,:,:,iw]).dot(CSC.T.conj())

    gf_cc_hf = np.zeros((len(kptlist), nmo, nmo, nw), dtype=complex)
    for ik, k in enumerate(kptlist):
        for iw in range(len(omegas)):
            gf_cc_hf[ik,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_hf[k,:,:,iw]) - sigma[ik,:,:,iw])

    if rank == 0:
        print ('KCCGF+KHF')
        for i in range(len(omegas)):
            print (omegas[i], -np.trace(gf_cc_hf[0,:,:,i].imag)/np.pi)

    # compute CCSD self-energy in CAS space
    gf_gw_cas_no = np.zeros_like(gf_cc)
    for ik, k in enumerate(kptlist):
        for iw in range(len(omegas)):
            gf_gw_cas_no[ik,:,:,iw] = np.dot(mf_cas.mo_coeff[k], gf_gw_cas[k,:,:,iw]).dot(mf_cas.mo_coeff[k].T.conj())
    sigma_cas = utils.get_sigma(gf_gw_cas_no, gf_cc)

    # transform self-energy to full space and get CC+HF GF
    sigma = np.zeros((len(kptlist), nmo, nmo, nw), dtype=complex)
    ovlp = kmf.get_ovlp()
    for ik, k in enumerate(kptlist):
        CSC = np.dot(kmf.mo_coeff[k].T.conj(), ovlp[k]).dot(no_coeff_k[k])
        for iw in range(len(omegas)):
            sigma[ik,:,:,iw] = np.dot(CSC, sigma_cas[ik,:,:,iw]).dot(CSC.T.conj())

    gf_cc_gw = np.zeros((len(kptlist), nmo, nmo, nw), dtype=complex)
    for ik, k in enumerate(kptlist):
        for iw in range(len(omegas)):
            gf_cc_gw[ik,:,:,iw] = np.linalg.inv(np.linalg.inv(gf_gw[k,:,:,iw]) - sigma[ik,:,:,iw])

    if rank == 0:
        print ('KCCGF+KGW')
        for i in range(len(omegas)):
            print (omegas[i], -np.trace(gf_cc_gw[0,:,:,i].imag)/np.pi)


kpt_idx = 0
# CAS-CCGF parameters
nocc_act = 4
nvir_act = 10

run_ccgf(kpt_idx, nocc_act=nocc_act, nvir_act=nvir_act)
