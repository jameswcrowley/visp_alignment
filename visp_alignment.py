from pathlib import Path

import numpy as np
import astropy.io.fits as fits
from astropy.time import Time, TimeDelta
import astropy.units as u
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a
import sunpy.map
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

    def load_dkist_slits(self):
        """
        Loads the DKIST slit data from the specified directory, extracts the raster and spatial coordinates
        
        Parameters:
        -----------
        self.cfg.path_to_dkist_data (str): The path to the directory containing the DKIST .fits files
        
        Returns:
        --------
        raster (np.ndarray): The raster data
        spatial_x_coordinates (list): The x coordinates of the spatial data
        spatial_y_coordinates (list): The y coordinates of the spatial data
        """
        data_dir = Path(self.cfg.path_to_dkist_data)

        all_fits = sorted(data_dir.glob("*.fits"))

        if not all_fits:
            raise FileNotFoundError(f"No .fits files found in {data_dir}")

        raster = []
        spatial_x_coordinates = []
        spatial_y_coordinates = []

        header = fits.getheader(all_fits[0], ext=1)

        number_x = header["DNAXIS3"]

        print(number_x)

        for i in range(number_x):
            path = all_fits[i]
            data = fits.getdata(path, ext=1)
            header = fits.getheader(path, ext=1)

            spatial_x, spatial_y = self._fits_pixel_to_spatial(
                data, header, self.cfg.wavelength_index
            )

            raster.append(data[0, :, :])
            spatial_x_coordinates.append(spatial_x)
            spatial_y_coordinates.append(spatial_y)

        return np.array(raster), spatial_x_coordinates, spatial_y_coordinates
    
    def _fits_pixel_to_wavelength(self, data, header):
        """
        Converts the pixel values in the DKIST data to wavelengths.

        Parameters:
        -----------
        data (np.ndarray): The DKIST data
        header (astropy.io.fits.Header): The FITS header

        Returns:
        --------
        slice (np.ndarray): The wavelength slice
        """
        cdelt2 = header["CDELT2"]
        crpix2 = header["CRPIX2"]
        crval2 = header["CRVAL2"]

        slice = data[0, :, 0]

        for i in range(len(slice)):
            wavelength = crval2 + cdelt2 * (i - crpix2)
            slice[i] = wavelength
        
        return slice
    
    def _fits_pixel_to_spatial(self, data, fixed_keywords, changing_keywords, wavelength_index):
        """
        Converts the pixel values in the DKIST data to spatial coordinates.

        Parameters:
        -----------
        data (np.ndarray): The DKIST data
        fixed_keywords (dict): A dictionary of fixed keywords
        changing_keywords (dict of lists): A dictionary of lists containing the changing keywords
        wavelength_index (int): The index of the wavelength slice

        Returns:
        --------
        spatial_x (list): The x coordinates of the spatial data
        spatial_y (list): The y coordinates of the spatial data
        """
        cdelt3 = fixed_keywords["CDELT3"]
        cdelt1 = fixed_keywords["CDELT1"]
        pc1_1 = fixed_keywords["PC1_1"]
        pc1_3 = fixed_keywords["PC1_3"]
        pc3_1 = fixed_keywords["PC3_1"]
        pc3_3 = fixed_keywords["PC3_3"]

        slit = data[0, wavelength_index, :]

        x_pixel = 0
        spatial_x = []
        spatial_y = []

        for i in range(slit.size):
            if not np.isnan(slit[i]):
                crpix1 = changing_keywords["CRPIX1"][i]
                crpix3 = changing_keywords["CRPIX3"][i]
                crval1 = changing_keywords["CRVAL1"][i]
                crval3 = changing_keywords["CRVAL3"][i]
                y_pixel = i

                x = crval1 + cdelt1 * (
                    pc1_1 * (x_pixel - crpix1) + pc1_3 * (y_pixel - crpix3)
                )
                y = crval3 + cdelt3 * (
                    pc3_1 * (x_pixel - crpix1) + pc3_3 * (y_pixel - crpix3)
                )

                spatial_x.append(x)
                spatial_y.append(y)

        return spatial_x, spatial_y

    def load_hmi(self, start_time: Time, end_time: Time) -> tuple[list, Time]:
        """
        Downloads all HMI data within one minute of the passed time interval

        Parameters:
        -----------
        start_time (Time): The start time of the interval
        end_time (Time): The end time of the interval

        Returns:
        --------
        tuple[list, Time]: A tuple containing the downloaded file paths and the corresponding image times
        """
        search_results = Fido.search(
            a.Instrument.hmi,
            a.Physobs("intensity"),
            a.Time(start_time - 1 * u.minute, end_time + 1 * u.minute),
        )

        downloaded_file_paths = Fido.fetch(
            search_results, path=self.cfg.path_to_sunpy, progress=self.cfg.verbose
        )

        image_times = []
        for file_path in downloaded_file_paths:
            hmi_map = sunpy.map.Map(file_path)
            image_times.append(hmi_map.date)
        image_times = Time(image_times)

        return downloaded_file_paths, image_times

    def find_closest_hmi(self, target_time: Time, hmi_times: Time, file_paths: list):
        """
        Finds the HMI file path that is closest in time to the target_time

        Parameters:
        -----------
        target_time (Time): The time to which we want to find the closest HMI data
        hmi_times (Time): The times of the HMI data
        file_paths (list): The file paths of the HMI data

        Returns:
        --------
        tuple[str, int, Time, Quantity]: A tuple containing the closest file path, its index, the closest time, and the time offset
        """
        time_differences = np.abs(hmi_times - target_time)

        closest_index = time_differences.argmin()

        closest_file_path = file_paths[closest_index]
        closest_time = hmi_times[closest_index]
        time_offset = time_differences[closest_index].to(u.second)

        return closest_file_path, closest_index, closest_time, time_offset

    def get_hmi_map(self, index: int, file_paths: list):
        """
        Returns the map from the path at the passed index

        Parameters:
        -----------
        index (int): The index of the file path in the list
        file_paths (list): The list of file paths

        Returns:
        --------
        sunpy.map.GenericMap: The HMI map corresponding to the file path at the given index
        """
        map = sunpy.map.Map(file_paths[index])

        return map

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
