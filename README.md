# DKIST visp-alignment

## Concept
DKIST data is heavily misaligned, sometimes with errors upwards of 20 arcseconds. To tackle this issue, DKIST visp-alignment uses HMI solar telescope as a reference point and cross-correlates the data from both the telescopes to get the minimum amount of loss. 

## Methodology 
1. Loading in the DKIST data and using `fido.search()` to download all the HMI data that falls within the start and endtime of the DKIST data (~30 minutes)
2. Constructing DKIST's spatial coordinates using the header keywords `CRVAL1/3`, `CRPIX1/3`, `CDELT1/3`, and `PCi_j`. 
3. Interpolating the HMI data and scale of the DKIST data and cropping it within the range of 20 arcseconds from DKIST. 
4. Using a loss function with `CRVAL1/3` and `PCi_j` as parameters to shift the DKIST coordinates to align best with interpolated HMI data. 
5. Reconstructing the DKIST coordinates with the calculated shifts. 

## Running Requirements
Here is the list of all the tools/python libraries required to run `visp_alignment.py`

```from pathlib import Path
import numpy as np
import astropy.io.fits as fits
from astropy.time import Time, TimeDelta
import astropy.units as u
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a
from sunpy import map as map
import dkist
import os
from scipy import interpolate as interp
import scipy.optimize as opt
import bisect
from itertools import chain
```
