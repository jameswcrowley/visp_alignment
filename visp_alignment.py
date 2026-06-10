import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from sunpy.net import Fido, attrs as a
import dkist.net
import dkist

import os

class Config:
    """
    Specifies all settings, paths, and other metaparameters used in the alignment
    """

    def __init__(self,
                path_to_dkist_data: str,
                path_to_sunpy: str,
                verbose: bool
                ):
        
        self.path_to_dkist_data = path_to_dkist_data
        self.path_to_sunpy = path_to_sunpy

        self.verbose = verbose


class Data:
    """
    Class to hold the loaded HMI & DKIST data, as well as neccessary metadata
    """

    def __init__(self,
                 cfg: Config
                 ):
        
        self.cfg = cfg

# ------------------------------------------------------------------------------------

class DataLoader:
    """
    Loads the DKIST and HMI data, extracts neccessary info from headers or shapes, saves it to be used later
    """

    def __init__(self,
                 cfg: Config
                 ):
        
        self.cfg = cfg

class Interpolator:
    """
    Takes an instance of the Data class and interpolates the HMI data to the resolution / pixel grid of DKIST 
    """

    def __init__(self,
                cfg: Config
                ):
    
        self.cfg = cfg

class Alignment:
    """
    Takes an instance of the Data class and the interpolated data from the Interpolator class to actually perform the alignment of the two datasets
    """
    
    def __init__(self,
                cfg: Config
                ):
    
        self.cfg = cfg