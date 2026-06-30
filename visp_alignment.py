import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from sunpy.net import Fido, attrs as a
import dkist.net
import dkist

import os

def get_time(folder_path):
    """
    The method uses the folder_path that directly contains all the fits as a parameter. 
    """
    fits_files = [
        filename for filename in os.listdir(folder_path)
        if filename.endswith('.fits') and os.path.isfile(os.path.join(folder_path, filename))
    ]

    first_path = os.path.join(folder_path, fits_files[0])
    last_path = os.path.join(folder_path, fits_files[-1])

    fits_header1 = fits.open(first_path)[1].header
    fits_header2 = fits.open(last_path)[1].header
    
    return (fits_header1["DATE-AVG"], fits_header2["DATE-AVG"])

def get_headers(folder_path):
    fits_files = [
        filename for filename in os.listdir(folder_path)
        if filename.endswith('.fits') and os.path.isfile(os.path.join(folder_path, filename))
    ]

    first_path = os.path.join(folder_path, fits_files[0])
    last_path = os.path.join(folder_path, fits_files[-1])
    fits_header1 = fits.open(first_path)[1].header
    fits_header2 = fits.open(last_path)[1].header
    return


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
