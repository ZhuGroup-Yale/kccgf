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

