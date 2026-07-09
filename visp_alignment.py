from pathlib import Path

import numpy as np
import astropy.io.fits as fits
from astropy.time import Time, TimeDelta
import astropy.units as u
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a
import sunpy.map
from sunpy import map as map
import sunpy.data.sample
import dkist.net
import dkist
import os

def get_time(folder_path):
    """
    The method uses the folder_path that directly contains all the fits as a parameter.

    Parameters:
    ------------
    folder_path (str): The path to the folder containing the DKIST .fits files

    Returns:
    ---------
    tuple: A tuple containing the start and end times of the DKIST data in the folder
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


class Config:
    """
    Specifies all settings, paths, and other metaparameters used in the alignment
    """

    def __init__(
        self,
        path_to_dkist_data: str,
        path_to_sunpy: str,
        wavelength_index: int,
        verbose: bool,
    ):
        self.path_to_dkist_data = path_to_dkist_data
        self.path_to_sunpy = path_to_sunpy
        self.verbose = verbose
        self.wavelength_index = wavelength_index


class Data:
    """
    Class to hold the loaded HMI & DKIST data, as well as neccessary metadata
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg


class DataLoader:
    """
    Loads the DKIST and HMI data, extracts neccessary info from headers or shapes, saves it to be used later
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_hmi(self, start_time: Time, end_time: Time):
        """
        Downloads all HMI data within one minute of the passed time interval

        Parameters:
        -----------
        start_time (Time): The start time of the interval
        end_time (Time): The end time of the interval

        ** Currently only reads in middle file, not all related files **
        """
        search_results = Fido.search(
            a.Instrument.hmi,
            a.Physobs("intensity"),
            a.Time(start_time - 1 * u.minute, end_time + 1 * u.minute),
        )

        hmi_files = Fido.fetch(
            search_results, path=self.cfg.path_to_sunpy, progress=self.cfg.verbose, site="NSO"
        )

        # read in the middle file using scipy.map to extract the coordinates and data.
        hmi = map.Map(hmi_files[0])

        # read in the x and y coordinates as seperate arrays.
        hmix = map.all_coordinates_from_map(hmi).Tx
        hmiy = map.all_coordinates_from_map(hmi).Ty
        
        # read in the intensity data as a 2D array.  
        hmi_data = hmi.data

        image_time = hmi.date

        return hmix, hmiy, hmi_data, image_time

    def get_dkist_wavelengths(self, folder_path): #read in data step 1
        """
        Returns the 2d array of the avg intensity of 30 wavelengths across all slits in the first raster

        Parameters:
        ------------
        folder_path (str): the path to the DKIST data folder
        """
        asdf_path = folder_path + [file_path for file_path in os.listdir(folder_path) if ".asdf" in file_path][0] #gets all the metadata from the folder
        ds = dkist.load_dataset(asdf_path)
        # print(ds.wcs.world_axis_names)
        # if "stokes" in ds.wcs.world_axis_names:
        #     data = np.array(ds[0, 0, :, :30, :].data)
        #     return np.nanmean(data, axis = 1)
        # else:
        #     data = np.array(ds[0, :, :30, :].data)
        #     return np.nanmean(data, axis = 1)
        data = np.array(ds[0, 0, :30, :].data)
        return np.nanmean(data, axis = 1)

    def get_dkist_headers(self, folder_path): #read in data step 1
        """
        Returns the 2 arrays of the fixed and changing keywords from the DKIST headers

        Parameters:
        ------------
        folder_path (str): the path to the DKIST data folder

        Returns:
        ---------
        fixed_keywords(dict): A dictionary of fixed keywords {key: value}
        changing_keywords(dict of lists): A dictionary of lists containing the changing keywords {key: [values]}
        """
        fits_files = [
            filename for filename in os.listdir(folder_path)
            if filename.endswith('.fits') and "_I_" in filename and os.path.isfile(os.path.join(folder_path, filename))
        ]
        header = fits.open(os.path.join(folder_path, fits_files[0]))[1].header
        fixed_keywords = {
            "CDELT1": header["CDELT1"],
            "CDELT3": header["CDELT3"],
            "DNAXIS3": header["DNAXIS3"],
            "DNAXIS1": header["DNAXIS1"],
            "PC1_1": header["PC1_1"],
            "PC1_3": header["PC1_3"],
            "PC3_1": header["PC3_1"],
            "PC3_3": header["PC3_3"]
        }
        changing_keywords = {
            "CRVAL1": [],
            "CRVAL3": [],
            "CRPIX1": [],
            "CRPIX3": [],
            "DATE-AVG": []
        }
        for slit_i in fits_files:
            header = fits.open(os.path.join(folder_path, slit_i))[1].header
            changing_keywords["CRVAL1"].append(header["CRVAL1"])
            changing_keywords["CRVAL3"].append(header["CRVAL3"])
            changing_keywords["CRPIX1"].append(header["CRPIX1"])
            changing_keywords["CRPIX3"].append(header["CRPIX3"])
            changing_keywords["DATE-AVG"].append(header["DATE-AVG"])
        
        return fixed_keywords, changing_keywords, fits_files
        



class Interpolator:
    """
    Takes an instance of the Data class and interpolates the HMI data to the resolution / pixel grid of DKIST
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg


class Alignment:
    """
    Takes an instance of the Data class and the interpolated data from the Interpolator class to actually perform the alignment of the two datasets
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg


if __name__ == "__main__":
    
    cfg = Config(
    path_to_dkist_data='/Users/jamescrowley/Documents/summer_2026/research/pid_3_35/XVNDZY/', 
    path_to_sunpy="~/sunpy/data/", 
    wavelength_index=30, 
    verbose=True
    )

    loader = DataLoader(cfg)


    raster, x, y = loader.load_dkist_slits()

    print(raster.shape)

    wave_index = 30  
    spatial_map = raster[:, wave_index, :]

    plt.figure(figsize=(12, 4))


    img = plt.pcolormesh(x, y, spatial_map, shading='auto', cmap='magma', aspect='equal')

    plt.colorbar(img, label='Intensity')
    plt.xlabel('Spatial Y Axis (2556 channels / Arcseconds)')
    plt.ylabel('Spatial X Axis (200 channels / Arcseconds)')
    plt.title(f'Monochromatic Spatial Map at Wavelength Index {wave_index}')
    plt.show()

    print("done")
