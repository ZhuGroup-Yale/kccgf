import numpy as np
import h5py, os

nmo = 26
kpt = 0
nkpts = 1

for orb in range(nmo):
    filepath = 'orb%d-kpt%d/'%(orb, kpt)
    fn = filepath+'kccgf.h5'
    feri = h5py.File(fn, 'r')
    gf_ip = np.array(feri['gf_ip'])
    gf_ea = np.array(feri['gf_ea'])
    omegas = np.array(feri['omegas'])
    eta = np.array(feri['eta'])
    feri.close()
    if orb == 0:
        gf_ip_gather = np.zeros((nkpts,nmo,nmo,len(omegas)), dtype=complex)
        gf_ea_gather = np.zeros((nkpts,nmo,nmo,len(omegas)), dtype=complex)

    gf_ip_gather[0,orb,:] = gf_ip[0,0,:]
    gf_ea_gather[0,:,orb] = gf_ea[0,:,0]

fn = 'kccgf_gather.h5'
feri = h5py.File(fn, 'w')
feri['gf_ip'] = np.asarray(gf_ip_gather)
feri['gf_ea'] = np.asarray(gf_ea_gather)
feri['omegas'] = np.asarray(omegas)
feri['eta'] = np.asarray(eta)
feri.close()

for i in range(len(omegas)):
    print (omegas[i], -np.trace(gf_ip_gather[0,:,:,i].imag)/np.pi, -np.trace(gf_ea_gather[0,:,:,i].imag)/np.pi)
