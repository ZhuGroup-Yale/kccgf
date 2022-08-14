from pyscf import lib
from pyscf.pbc import gto, scf
import numpy, scipy, copy, h5py
import numpy as np
from pyscf.lib import logger
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

einsum = lib.einsum

def make_casno(mymp, thresh=1e-4, nvir_act=None, nocc_act=None, vno_only=False, return_dm=False):
    '''
    Periodic natural orbitals for periodic CAS calculation

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        vno_only : bool
            Only construct virtual natural orbitals. Default is False.
        return_rdm : bool
            Return correlated density matrix. Default is False.

    Returns:
        mf_cas : mean-field object with all integrals in NO basis.
        no_coeff_k : ndarray
            Semicanonical NO coefficients in the AO basis
        dm : ndarray, correlated density matrix in MO basis (optional).
    '''
    mf = mymp._scf
    dm = None
    if rank == 0:
        dm = mymp.make_rdm1()
    comm.Barrier()
    dm = comm.bcast(dm, root=0)
    nmo = mymp.nmo
    nocc = mymp.nocc
    nkpts = len(dm)
    if rank == 0:
        mymp.verbose = 4
    else:
        mymp.verbose = 0

    for k in range(nkpts):
        no_occ_v, no_coeff_v = np.linalg.eigh(dm[k][nocc:,nocc:])
        no_occ_v = np.flip(no_occ_v)
        no_coeff_v = np.flip(no_coeff_v, axis=1)
        if rank == 0:
            logger.info(mf, 'Full no_occ_v @ k%d = \n %s', k, no_occ_v)
        if nocc_act is not None:
            vno_only = False
        if not vno_only:
            no_occ_o, no_coeff_o = np.linalg.eigh(dm[k][:nocc,:nocc])
            no_occ_o = np.flip(no_occ_o)
            no_coeff_o = np.flip(no_coeff_o, axis=1)
            if rank == 0:
                logger.info(mf, 'Full no_occ_o @ k%d = \n %s', k, no_occ_o)

        if nvir_act is None and nocc_act is None:
            no_idx_v = np.where(no_occ_v > thresh)[0]
            if not vno_only:
                no_idx_o = np.where(2-no_occ_o > thresh)[0]
            else:
                no_idx_o = range(0, nocc)
        elif nvir_act is None and nocc_act is not None:
            no_idx_v = range(0, nmo-nocc)
            no_idx_o = range(nocc-nocc_act, nocc)
        elif nvir_act is not None and nocc_act is None:
            no_idx_v = range(0, nvir_act)
            no_idx_o = range(0, nocc)
        else:
            no_idx_v = range(0, nvir_act)
            no_idx_o = range(nocc-nocc_act, nocc)

        # semi-canonicalization
        fvv = numpy.diag(mf.mo_energy[k][nocc:])
        fvv_no = numpy.dot(no_coeff_v.T, numpy.dot(fvv, no_coeff_v))
        no_vir = len(no_idx_v)
        _, v_canon_v = numpy.linalg.eigh(fvv_no[:no_vir,:no_vir])
        _, v_canon_v_frozen = numpy.linalg.eigh(fvv_no[no_vir:,no_vir:])
        if not vno_only:
            foo = numpy.diag(mf.mo_energy[k][:nocc])
            foo_no = numpy.dot(no_coeff_o.T, numpy.dot(foo, no_coeff_o))
            no_occ = nocc - len(no_idx_o)
            _, v_canon_o = numpy.linalg.eigh(foo_no[no_occ:,no_occ:])
            _, v_canon_o_frozen = numpy.linalg.eigh(foo_no[:no_occ,:no_occ])

        no_coeff_v_frozen = numpy.dot(mf.mo_coeff[k][:,nocc:], numpy.dot(no_coeff_v[:,no_vir:], v_canon_v_frozen))
        no_coeff_v = numpy.dot(mf.mo_coeff[k][:,nocc:], numpy.dot(no_coeff_v[:,:no_vir], v_canon_v))
        if not vno_only:
            no_coeff_o_frozen = numpy.dot(mf.mo_coeff[k][:,:nocc], numpy.dot(no_coeff_o[:,:no_occ], v_canon_o_frozen))
            no_coeff_o = numpy.dot(mf.mo_coeff[k][:,:nocc], numpy.dot(no_coeff_o[:,no_occ:], v_canon_o))
 
        if not vno_only:
            ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
            n_no = len(no_idx_o) + len(no_idx_v)
            if rank == 0:
                logger.info(mf, 'CAS @ k%d no_occ_o = \n %s, \n no_occ_v = \n %s', k, no_occ_o[no_idx_o], no_occ_v[no_idx_v])
        else:
            ne_sum = np.trace(dm[k][:nocc,:nocc]) + np.sum(no_occ_v[no_idx_v])
            ne_sum = ne_sum.real
            n_no = nocc + len(no_idx_v)
            if rank == 0:
                logger.info(mf, 'CAS @ k%d mo_occ_o = \n %s, \n no_occ_v = \n %s', k, dm[k][:nocc,:nocc].diagonal(), no_occ_v[no_idx_v])
        nelectron = int(round(ne_sum))
        if rank == 0:
            logger.info(mf, 'CAS @ k%d norb = %s, nelec = %s, ne_no = %s', k, n_no, nelectron, ne_sum)

        if not vno_only:
            no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
        else:
            no_coeff = np.concatenate((mf.mo_coeff[k][:,:nocc], no_coeff_v), axis=1)
            no_coeff_o = mf.mo_coeff[k][:,:nocc]

        if k == 0:
            nao, ncas = no_coeff.shape
            no_coeff_k = np.zeros((nkpts,nao,ncas),dtype=complex)
            thresh = None
            nocc_act = no_coeff_o.shape[-1]
            nvir_act = no_coeff_v.shape[-1]
            no_coeff_v_frozen_k = np.zeros((nkpts,nao,nmo-nocc-nvir_act),dtype=complex)
            if not vno_only:
                no_coeff_o_frozen_k = np.zeros((nkpts,nao,nocc-nocc_act),dtype=complex)

        no_coeff_k[k] = no_coeff
        no_coeff_v_frozen_k[k] = no_coeff_v_frozen
        if not vno_only:
            no_coeff_o_frozen_k[k] = no_coeff_o_frozen

    # new mf object for CAS
    cell_cas = gto.M(a=mf.cell.a, verbose=0)
    cell_cas.nelectron = nelectron
    cell_cas.max_memory = mf.cell.max_memory
    cell_cas.incore_anyway = True
    mf_cas = scf.KRHF(cell_cas, mf.kpts, exxdiv=mf.exxdiv)

    # compute CAS integrals
    hcore_no = None
    veff_no = None
    mf_cas.mo_energy = None
    mf_cas.mo_coeff = None
    mf_cas.mo_occ = None
    if rank == 0:
        hcore = mf.get_hcore()
        veff = mf.get_veff()
        hcore_no = np.zeros((nkpts,ncas,ncas),dtype=complex)
        veff_no = np.zeros((nkpts,ncas,ncas),dtype=complex)
        cas_mo_energy = np.zeros((nkpts,ncas))
        cas_mo_coeff = np.zeros((nkpts,ncas,ncas),dtype=complex)
        for k in range(nkpts):
            hcore_no[k] = np.dot(no_coeff_k[k].T.conj(), np.dot(hcore[k], no_coeff_k[k]))
            veff_no[k] = np.dot(no_coeff_k[k].T.conj(), np.dot(veff[k], no_coeff_k[k]))
            mo_e, mo_c = scipy.linalg.eigh(hcore_no[k]+veff_no[k])
            cas_mo_energy[k] = mo_e
            cas_mo_coeff[k] = mo_c
        mf_cas.mo_energy = cas_mo_energy
        mf_cas.mo_coeff = cas_mo_coeff
        mf_cas.mo_occ = mf_cas.get_occ()
    comm.Barrier()

    # make sure all MPI processes have same integrals
    no_coeff_k = comm.bcast(no_coeff_k, root=0)
    mf_cas.mo_occ = comm.bcast(mf_cas.mo_occ, root=0)
    mf_cas.mo_energy = comm.bcast(mf_cas.mo_energy, root=0)
    mf_cas.mo_coeff = comm.bcast(mf_cas.mo_coeff, root=0)
    hcore_no = comm.bcast(hcore_no, root=0)
    veff_no = comm.bcast(veff_no, root=0)

    # transform HF dm to NO basis
    ovlp = mf.get_ovlp()
    dm_hf = mf.make_rdm1()
    dm_cas_no = np.zeros((nkpts,ncas,ncas),dtype=complex)
    for k in range(nkpts):
        CS = np.dot(no_coeff_k[k].T.conj(), ovlp[k])
        dm_cas_no[k] = np.dot(CS, np.dot(dm_hf[k], CS.T.conj()))
    comm.Barrier()

    # set up integrals for mf_cas
    mf_cas.get_hcore = lambda *args: hcore_no
    mf_cas.get_ovlp = lambda *args: np.array([np.eye(ncas)]*nkpts)

    # frozen orbitals and no_coeff_full
    frozen = np.concatenate((np.arange(nocc-nocc_act), np.arange(nocc+nvir_act,nmo)))
    no_coeff_full = np.zeros((nkpts,nmo,nmo), dtype=complex)
    no_coeff_full[:,:,(nocc-nocc_act):(nocc+nvir_act)] = no_coeff_k
    no_coeff_full[:,:,(nocc+nvir_act):] = no_coeff_v_frozen_k
    if not vno_only:
        no_coeff_full[:,:,:(nocc-nocc_act)] = no_coeff_o_frozen_k

    mymp.verbose = mf.verbose
    if return_dm:
        return mf_cas, no_coeff_k, hcore_no + veff_no, dm_cas_no, frozen, no_coeff_full, dm
    else:
        return mf_cas, no_coeff_k, hcore_no + veff_no, dm_cas_no, frozen, no_coeff_full

