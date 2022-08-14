kccgf
=====

Periodic coupled cluster Green's function for realistic solids

Authors: Tianyu Zhu (tianyu.zhu@yale.edu)

Installation
------------

* Prerequisites
    - PySCF 1.7 or higher, and all dependencies 
	- fcdmft (Optional, https://github.com/ZhuGroup-Yale/fcdmft)

* You need to set environment variable `PYTHONPATH` to export kccgf to Python. 
  E.g. if kccgf is installed in `/opt`, your `PYTHONPATH` should be

        export PYTHONPATH=/opt/kccgf:$PYTHONPATH

Features
--------

* k-point CCGF (hybrid MPI+OpenMP parallelization)

* k-point CCGF with model order reduction

* k-point CCGF with active-space self-energy correction (CCGF+GW)

QuickStart
----------

To run a periodic CCGF calculation, use the following steps:

1. Run periodic HF calculation and save the GDF integrals. Example: `/kccgf/examples/silicon/HF/si_khf.py`.
2. Run periodic CCSD calculation and save the CCSD amplitudes and MO integals (ERIS). Read GDF integrals and HF results
   from the last step to save time. Example: `/kccgf/examples/silicon/CCSD/si_ccsd.py`.
3. Run periodic CCGF calculation, either in parallel or using batch serial jobs. Read CCSD amplitudes and ERIS from last
   step to save time. For full CCGF, see example: `/kccgf/examples/silicon/CCGF/si_ccgf.py` (use `submit.sh` to submit 
   batch jobs and `gather_gf.py` to collect GF and DOS). For MOR-CCGF, see example: `/kccgf/examples/silicon/MOR/`.

To run a MOR-CCGF+MOR (active-space self-energy correction), follow these steps:

1. Run periodic HF (same as in full CCGF Step 1).
2. Run G0W0@HF calculation and save GW self-energy files. Example: `/kccgf/examples/silicon/MOR-CCGF+GW/GW/si_gw.py`.
3. Run periodic CAS-CCSD calculation and save the CCSD amplitudes and MO integals (ERIS). Copy `vxc.h5` and
   `sigma_imag.h5` from GW folder to CAS-CCSD folder to enable loading GW results. Example:
   `/kccgf/examples/silicon/MOR-CCGF+GW/CAS-CCSD/si_ccsd.py`.
4. Run MOR-CCGF+GW calculation, either in parallel or using batch serial jobs. Read CCSD amplitudes and ERIS as well as
   GW self-energy from Step 2 and 3 to save time. See example:
   `/kccgf/examples/silicon/MOR-CCGF+GW/CAS-CCGF+GW/si_cc_gw.py`. Use `submit.sh` to submit batch jobs and
   `si_gather_gf.py` to collect GF and DOS.
