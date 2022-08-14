#!/usr/bin/env python
# Copyright 2017-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Restricted Coupled Cluster Greens Functions 
#
# Authors: James D. McClain
#          Jason Yu <jasonmyu1@gmail.com>
#

'''
Periodic Coupled Cluster Green's Function with MPI
This code is modified based on PySCF's krccgf code by Jason Yu and James McClain.
Modifiations:
    1. Fix bugs
    2. Compatible with latest PySCF and Python3 versions
    3. Add model order reduction for freqeuncy interpolation and extrapolation
    4. Support MPI parallelization (parallelize over k-points or orbitals)

Author: Tianyu Zhu (tianyu.zhu@yale.edu)
'''

import numpy as np
import scipy.sparse.linalg as spla
import scipy
import pyscf
from pyscf.cc import eom_rccsd
from pyscf.pbc.lib import kpts_helper
import time
import sys
from pyscf.lib import logger
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

###################
# EA Greens       #
###################

def greens_b_vector_ea_rhf(cc, p, kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        vector1 += -cc.t1[kp,p,:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                vector2[kj,ka] += -cc.t2[kp,kj,ka,p,:,:,:]
    else:
        vector1[p-nocc] = 1.0
    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)

def greens_e_vector_ea_rhf(cc, p, kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=ds_type)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = np.conj(cc.t1)
        l2 = np.conj(cc.t2)

    if p < nocc:
        vector1 += l1[kp,p,:]
        for kj in range(nkpts):
            for ka in range(nkpts):
                vector2[kj,ka] += 2*l2[kp,kj,ka,p,:,:,:]
                vector2[kj,ka] -= l2[kj,kp,ka,:,p,:,:]

    else:
        vector1[p-nocc] = -1.0
        vector1 += np.einsum('ia,i->a', l1[kp], cc.t1[kp,:,p-nocc])
        kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        for kk in range(nkpts):
            for kl in range(nkpts):
                # kk - kc + kl - kp = G => kc = kl - kp + kk
                kc = kconserv[kl,kp,kk]
        
                vector1 += 2 * np.einsum('klca,klc->a', l2[kk,kl,kc], \
                           cc.t2[kk,kl,kc,:,:,:,p-nocc])
                vector1 -= np.einsum('klca,lkc->a', l2[kk,kl,kc], \
                           cc.t2[kl,kk,kc,:,:,:,p-nocc])

        for kb in range(nkpts):
            vector2[kb,kp,:,p-nocc,:] += -2.*l1[kb]
    
        for ka in range(nkpts):
            # kj == ka
            # kb == kc == kp
            vector2[ka,ka,:,:,p-nocc] += l1[ka]

        kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        for kj in range(nkpts):
            for ka in range(nkpts):
                # TZ NOTE: kconserv bug fixed
                # kj - ka + kk - kb = G, kk == kp => kb = kp - ka + kj
                kb = kconserv[kp,ka,kj]

                vector2[kj,ka] += 2*np.einsum('k,jkba->jab', \
                                  cc.t1[kp,:,p-nocc], l2[kj,kp,kb,:,:,:,:])
                vector2[kj,ka] -= np.einsum('k,jkab->jab', \
                                  cc.t1[kp,:,p-nocc], l2[kj,kp,ka,:,:,:,:])

    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)

###################
# IP Greens       #
###################

def greens_b_vector_ip_rhf(cc,p,kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)

    if p < nocc:
        vector1[p] = 1.0
    else:
        vector1 += cc.t1[kp,:,p-nocc]
        for ki in range(nkpts):
            for kj in range(nkpts):
                # TZ NOTE: b_vector_ip bug fixed
                vector2[ki,kj] += cc.t2[ki,kj,kp,:,:,p-nocc,:]
    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p,kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = np.conj(cc.t1)
        l2 = np.conj(cc.t2)

    if p < nocc:
        vector1[p] = -1.0
        vector1 += np.einsum('ia,a->i', l1[kp], cc.t1[kp,p,:])
        kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        for kl in range(nkpts):
            for kc in range(nkpts):
                 # TZ NOTE: kconserv bug fixed
                 # kp - kc + kl - kd = G => kd = kp - kc + kl
                 kd = kconserv[kp,kc,kl]
                 vector1 += 2 * np.einsum('ilcd,lcd->i', \
                       l2[kp,kl,kc], cc.t2[kp,kl,kc,p,:,:,:])
                 vector1 -= np.einsum('ilcd,ldc->i',   \
                       l2[kp,kl,kc], cc.t2[kp,kl,kd,p,:,:,:])

        for kj in range(nkpts):
            vector2[kp,kj,p,:,:] += -2*l1[kj]

        for ki in range(nkpts):
            # kj == kk == kp, ki == kb
            vector2[ki,kp,:,p,:] +=  l1[ki]

            for kj in range(nkpts):
                # kc == kk == kp
                vector2[ki,kj] += 2*np.einsum('c,ijcb->ijb', \
                       cc.t1[kp,p,:], l2[ki,kj,kp,:,:,:,:])
        
                vector2[ki,kj] -= np.einsum('c,jicb->ijb', \
                       cc.t1[kp,p,:], l2[kj,ki,kp,:,:,:,:]) 

    else:
        vector1 += -l1[kp,:,p-nocc]
        kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        for ki in range(nkpts):
            for kj in range(nkpts):
                # TZ NOTE: kconserv bug fixed
                # ki - kc + kj - kb = G, kp == kc => kb = ki - kp + kj
                kb = kconserv[ki,kp,kj]
                vector2[ki, kj] += -2*l2[ki,kj,kp,:,:,p-nocc,:] + \
                                   l2[ki,kj,kb,:,:,:,p-nocc]

    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def greens_func_multiply(ham, vector, linear_part, kp, **kwargs):
    return np.array(ham(vector, kp, **kwargs) + linear_part * vector)

def greens_func_multiply_mor(ham, vector, linear_part):
    return np.array(np.dot(ham, vector) + linear_part * vector)


class KRCCGF(object):
    '''
    One Particle Greens Function Class for KRCCSD-GF method

    args:
    
    self
    cc - coupled-cluster reference object
    eta - broadening parameter 
    conv_tol - convergence tolerance of gcrotmk solution
    use_prev - use previous iteration's solution as initial guess

    returns: 
    
    G_{k,pq}^{IP}, G_{k,pq}^{EA} - numpy array containing matrix elements
                                   for k-dependent G^{IP} and G^{EA}
    
    '''
    def __init__(self, cc, eta=0.01, conv_tol=1e-3, max_iter=100, use_prev=False):
        self.cc = cc
        self.eta = eta
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.use_prev = use_prev
        self.stdout = cc.stdout
        self.verbose = cc.verbose
        self.m = 20

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nkpts, nocc, nvir = self.cc.t1.shape
        log.info('KRCCGF nkpts = %d, nocc = %d, nvir = %d', nkpts, nocc, nvir)
        logger.info(self, 'broadening = %s', self.eta)
        logger.info(self, 'GCROT(m,k) conv_tol = %s', self.conv_tol)
        logger.info(self, 'GCROT(m,k) max iterations = %s', self.max_iter)
        logger.info(self, 'use_prev = %s', self.use_prev)
        return self

    def solve_ip(self, kptlist, ps, qs, omegas, MOR=False, omega_mor=None):
        '''
        Compute IP-CCSD-GF in MO basis
        MOR: model order reduction (ref: JCTC, 15, 3185-3196, 2019), can be used
             for frequency interpolation and extrapolation.
        '''
        cc = self.cc
        log = logger.Logger(self.stdout, self.verbose)
        if rank == 0:
            log.debug('# Solving IP Portion #')
        comm.Barrier()
        eomip = pyscf.pbc.cc.eom_kccsd_rhf.EOMIP(self.cc)
        eomip_imds = eomip.make_imds(eris=self.cc.eris)
        comm.Barrier()

        # If multiple k-points, parallelize over k-points;
        # If single k-point, parallelize over orbitals
        if len(kptlist) > 1:
            segsize = len(kptlist) // size
            if rank < len(kptlist)-segsize*size:
                start = rank * segsize + rank
                stop = min(len(kptlist), start+segsize+1)
            else:
                start = rank * segsize + len(kptlist)-segsize*size
                stop = min(len(kptlist), start+segsize)
            gfvals = np.zeros((stop-start, len(ps), len(qs), len(omegas)), dtype=complex)
            start_k = start; stop_k = stop
            start_ip = 0; stop_ip = len(ps)
        elif len(kptlist) == 1:
            segsize = len(ps) // size
            if rank < len(ps)-segsize*size:
                start = rank * segsize + rank
                stop = min(len(ps), start+segsize+1)
            else:
                start = rank * segsize + len(ps)-segsize*size
                stop = min(len(ps), start+segsize)
            gfvals = np.zeros((len(kptlist), stop-start, len(qs), len(omegas)), dtype=complex)
            start_k = 0; stop_k = len(kptlist)
            start_ip = start; stop_ip = stop
        else:
            raise ValueError

        if MOR:
            omega_comp = omega_mor
        else:
            omega_comp = omegas

        for ikpt in range(start_k, stop_k):
            kp = kptlist[ikpt]
            e_vector=list()
            for q in qs:
                e_vector.append(greens_e_vector_ip_rhf(cc,q,kp))
            diag = eomip.get_diag(kp, imds=eomip_imds)

            segsize = len(ps) // size
            if rank < len(ps)-segsize*size:
                start = rank * segsize + rank
                stop = min(len(ps), start+segsize+1)
            else:
                start = rank * segsize + len(ps)-segsize*size
                stop = min(len(ps), start+segsize)

            for ip in range(start_ip, stop_ip):
                p = ps[ip]
                b_vector = greens_b_vector_ip_rhf(cc,p,kp)
                if MOR:
                    X_vec = np.zeros((len(b_vector), len(omega_mor)), dtype=complex)

                Sw = None
                for iw, omega in enumerate(omega_comp):
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(eomip.matvec, vector, omega - 1j*self.eta, kp, imds=eomip_imds)

                    diag_w = diag + omega - 1j*self.eta
                    S0 = b_vector / diag_w
                    invprecond_multiply = lambda x: x/diag_w
                    size_bvec = len(b_vector)
                    Ax = spla.LinearOperator((size_bvec,size_bvec), matr_multiply)
                    mx = spla.LinearOperator((size_bvec,size_bvec), invprecond_multiply)

                    self.ip_niter = 0
                    def callback(rk):
                        self.ip_niter += 1
                    cput1 = (time.process_time(), time.perf_counter())
                    if self.use_prev is False or Sw is None or MOR:
                        Sw, info = spla.gcrotmk(Ax, b_vector, x0=S0, M=mx, maxiter=self.max_iter, m=self.m, callback=callback, tol=self.conv_tol)
                    else:
                        Sw, info = spla.gcrotmk(Ax, b_vector, x0=Sw, M=mx, maxiter=self.max_iter, m=self.m, callback=callback, tol=self.conv_tol)
                    if info > 0:
                        ASw = greens_func_multiply(eomip.matvec, Sw, omega - 1j*self.eta, kp, imds=eomip_imds)
                        norm_rk = np.linalg.norm(ASw-b_vector)
                        print ("convergence to tolerance not achieved in", info, "iterations,","residual =", norm_rk)
                        if norm_rk > 0.1:
                            print ("WARNING: residual is larger than 0.1!")

                    if MOR:
                        # Construct MOR subspace vectors
                        X_vec[:,iw] = Sw
                    cput1 = logger.timer(self, 'IPGF k = %d/%d, orbital p = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                        ikpt+1,len(kptlist),ip+1,len(ps),iw+1,len(omega_comp),self.ip_niter,rank), *cput1)

                    if not MOR:
                        for iq,q in enumerate(qs):
                            gfvals[ikpt-start_k,ip-start_ip,iq,iw]  = -np.dot(e_vector[iq],Sw)

                if MOR:
                    # QR decomposition to orthorgonalize X_vec
                    S_vec, R_vec = scipy.linalg.qr(X_vec, mode='economic')
                    # Construct reduced order model
                    n_mor = len(omega_mor)
                    HS = np.zeros_like(S_vec)
                    for i in range(n_mor):
                        HS[:,i] = greens_func_multiply(eomip.matvec, S_vec[:,i], 0., kp, imds=eomip_imds)
                    # Reduced Hamiltonian, b_vector, diag
                    H_mor = np.dot(S_vec.T.conj(), HS)
                    b_vector_mor = np.dot(S_vec.T.conj(), b_vector)
                    diag_mor = H_mor.diagonal()
 
                    Sw = None
                    for iw, omega in enumerate(omegas):
                        def matr_multiply_mor(vector, args=None):
                            return greens_func_multiply_mor(H_mor, vector, omega - 1j*self.eta)
 
                        diag_w = diag_mor + omega - 1j*self.eta
                        S0 = b_vector_mor / diag_w
                        invprecond_multiply_mor = lambda x: x/diag_w
                        size_bvec = len(b_vector_mor)
                        Ax = spla.LinearOperator((size_bvec,size_bvec), matr_multiply_mor)
                        mx = spla.LinearOperator((size_bvec,size_bvec), invprecond_multiply_mor)
 
                        self.ip_niter = 0
                        def callback(rk):
                            self.ip_niter += 1
                        #cput1 = (time.process_time(), time.perf_counter())
                        if self.use_prev is False or Sw is None:
                            Sw, info = spla.gcrotmk(Ax, b_vector_mor, x0=S0, M=mx, maxiter=self.max_iter, callback=callback, tol=self.conv_tol*0.01)
                        else:
                            Sw, info = spla.gcrotmk(Ax, b_vector_mor, x0=Sw, M=mx, maxiter=self.max_iter, callback=callback, tol=self.conv_tol*0.01)
                        if info > 0:
                            print ("MOR convergence to tolerance not achieved in", info, "iterations")
                        #cput1 = logger.timer(self, 'MOR IPGF k = %d/%d, orbital p = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                        #    ikpt+1,len(kptlist),ip+1,len(ps),iw+1,len(omegas),self.ip_niter,rank), *cput1)
 
                        for iq,q in enumerate(qs):
                            e_vector_mor = np.dot(e_vector[iq], S_vec)
                            gfvals[ikpt-start_k,ip-start_ip,iq,iw]  = -np.dot(e_vector_mor,Sw)

        comm.Barrier()
        if len(kptlist) > 1:
            gfvals_gather = comm.gather(gfvals)
            if rank == 0:
                gfvals_gather = np.vstack(gfvals_gather)
            comm.Barrier()
        elif len(kptlist) == 1:
            gfvals_gather = comm.gather(gfvals.transpose(1,0,2,3))
            if rank == 0:
                gfvals_gather = np.vstack(gfvals_gather).transpose(1,0,2,3)
            comm.Barrier()
        gfvals_gather = comm.bcast(gfvals_gather,root=0)

        return gfvals_gather

    def solve_ea(self, kptlist, ps, qs, omegas, MOR=False, omega_mor=None):
        '''
        Compute EA-CCSD-GF in MO basis
        MOR: model order reduction (ref: JCTC, 15, 3185-3196, 2019), can be used
             for frequency interpolation and extrapolation.
        '''
        cc = self.cc
        log = logger.Logger(cc.stdout, self.verbose)
        if rank == 0:
            log.debug('# Solving EA Portion #')
        comm.Barrier()
        eomea = pyscf.pbc.cc.eom_kccsd_rhf.EOMEA(self.cc)
        eomea_imds = eomea.make_imds(eris=self.cc.eris)
        comm.Barrier()

        # If multiple k-points, parallelize over k-points;
        # If single k-point, parallelize over orbitals
        if len(kptlist) > 1:
            segsize = len(kptlist) // size
            if rank < len(kptlist)-segsize*size:
                start = rank * segsize + rank
                stop = min(len(kptlist), start+segsize+1)
            else:
                start = rank * segsize + len(kptlist)-segsize*size
                stop = min(len(kptlist), start+segsize)
            gfvals = np.zeros((stop-start, len(ps), len(qs), len(omegas)), dtype=complex)
            start_k = start; stop_k = stop
            start_iq = 0; stop_iq = len(qs)
        elif len(kptlist) == 1:
            segsize = len(qs) // size
            if rank < len(qs)-segsize*size:
                start = rank * segsize + rank
                stop = min(len(qs), start+segsize+1)
            else:
                start = rank * segsize + len(qs)-segsize*size
                stop = min(len(qs), start+segsize)
            gfvals = np.zeros((len(kptlist), len(ps), stop-start, len(omegas)), dtype=complex)
            start_k = 0; stop_k = len(kptlist)
            start_iq = start; stop_iq = stop
        else:
            raise ValueError

        if MOR:
            omega_comp = omega_mor
        else:
            omega_comp = omegas

        for ikpt in range(start_k, stop_k):
            kp = kptlist[ikpt]
            e_vector=list()
            for p in ps:
                e_vector.append(greens_e_vector_ea_rhf(cc,p,kp))
            diag = eomea.get_diag(kp, imds=eomea_imds)

            for iq in range(start_iq, stop_iq):
                q = qs[iq]
                b_vector = greens_b_vector_ea_rhf(cc,q,kp)
                if MOR:
                    X_vec = np.zeros((len(b_vector), len(omega_mor)), dtype=complex)

                Sw = None
                for iw, omega in enumerate(omega_comp):
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(eomea.matvec, vector, -omega - 1j*self.eta, kp, imds=eomea_imds)

                    diag_w = diag - omega - 1j*self.eta
                    S0 = b_vector / diag_w
                    invprecond_multiply = lambda x: x/diag_w
                    size_bvec = len(b_vector)
                    Ax = spla.LinearOperator((size_bvec,size_bvec), matr_multiply)
                    mx = spla.LinearOperator((size_bvec,size_bvec), invprecond_multiply)

                    self.ea_niter = 0
                    def callback(rk):
                        self.ea_niter += 1
                    cput1 = (time.process_time(), time.perf_counter())
                    if self.use_prev is False or Sw is None or MOR:
                        Sw, info = spla.gcrotmk(Ax, b_vector, x0=S0, M=mx, m=self.m, maxiter=self.max_iter, callback=callback, tol=self.conv_tol)
                    else:
                        Sw, info = spla.gcrotmk(Ax, b_vector, x0=Sw, M=mx, m=self.m, maxiter=self.max_iter, callback=callback, tol=self.conv_tol)
                    if info > 0:
                        ASw = greens_func_multiply(eomea.matvec, Sw, -omega - 1j*self.eta, kp, imds=eomea_imds)
                        norm_rk = np.linalg.norm(ASw-b_vector)
                        print ("convergence to tolerance not achieved in", info, "iterations,","residual =", norm_rk)
                        if norm_rk > 0.1:
                            print ("WARNING: residual is larger than 0.1!")

                    if MOR:
                        # Construct MOR subspace vectors
                        X_vec[:,iw] = Sw
                    cput1 = logger.timer(self, 'EAGF k = %d/%d, orbital q = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                        ikpt+1,len(kptlist),iq+1,len(qs),iw+1,len(omega_comp),self.ea_niter,rank), *cput1)

                    if not MOR:
                        for ip,p in enumerate(ps):
                            gfvals[ikpt-start_k,ip,iq-start_iq,iw] = np.dot(e_vector[ip],Sw)

                if MOR:
                    # QR decomposition to orthorgonalize X_vec
                    S_vec, R_vec = scipy.linalg.qr(X_vec, mode='economic')
                    # Construct reduced order model
                    n_mor = len(omega_mor)
                    HS = np.zeros_like(S_vec)
                    for i in range(n_mor):
                        HS[:,i] = greens_func_multiply(eomea.matvec, S_vec[:,i], 0., kp, imds=eomea_imds)
                    # Reduced Hamiltonian, b_vector, diag
                    H_mor = np.dot(S_vec.T.conj(), HS)
                    b_vector_mor = np.dot(S_vec.T.conj(), b_vector)
                    diag_mor = H_mor.diagonal()
 
                    Sw = None
                    for iw, omega in enumerate(omegas):
                        def matr_multiply_mor(vector, args=None):
                            return greens_func_multiply_mor(H_mor, vector, -omega - 1j*self.eta)
 
                        diag_w = diag_mor - omega - 1j*self.eta
                        S0 = b_vector_mor / diag_w
                        invprecond_multiply_mor = lambda x: x/diag_w
                        size_bvec = len(b_vector_mor)
                        Ax = spla.LinearOperator((size_bvec,size_bvec), matr_multiply_mor)
                        mx = spla.LinearOperator((size_bvec,size_bvec), invprecond_multiply_mor)
 
                        self.ea_niter = 0
                        def callback(rk):
                            self.ea_niter += 1
                        #cput1 = (time.process_time(), time.perf_counter())
                        if self.use_prev is False or Sw is None:
                            Sw, info = spla.gcrotmk(Ax, b_vector_mor, x0=S0, M=mx, maxiter=self.max_iter, callback=callback, tol=self.conv_tol*0.01)
                        else:
                            Sw, info = spla.gcrotmk(Ax, b_vector_mor, x0=Sw, M=mx, maxiter=self.max_iter, callback=callback, tol=self.conv_tol*0.01)
                        if info > 0:
                            print ("MOR convergence to tolerance not achieved in", info, "iterations")
                        #cput1 = logger.timer(self, 'MOR EAGF k = %d/%d, orbital q = %d/%d, freq w = %d/%d (%d iterations) @ Rank %d'%(
                        #    ikpt+1,len(kptlist),iq+1,len(qs),iw+1,len(omegas),self.ea_niter,rank), *cput1)
 
                        for ip,p in enumerate(ps):
                            e_vector_mor = np.dot(e_vector[ip], S_vec)
                            gfvals[ikpt-start_k,ip,iq-start_iq,iw] = np.dot(e_vector_mor, Sw)

        comm.Barrier()
        if len(kptlist) > 1:
            gfvals_gather = comm.gather(gfvals)
            if rank == 0:
                gfvals_gather = np.vstack(gfvals_gather)
            comm.Barrier()
        elif len(kptlist) == 1:
            gfvals_gather = comm.gather(gfvals.transpose(2,0,1,3))
            if rank == 0:
                gfvals_gather = np.vstack(gfvals_gather).transpose(1,2,0,3)
            comm.Barrier()
        gfvals_gather = comm.bcast(gfvals_gather,root=0)

        return gfvals_gather

    def kernel(self, k, p, q, omegas):
        return self.solve_ip(k, p, q, omegas), self.solve_ea(k, p, q, omegas)
