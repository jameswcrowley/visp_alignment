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
    
    def _fits_pixel_to_spatial(self, data, header, wavelength_index):
        """
        Converts the pixel values in the DKIST data to spatial coordinates.

        Parameters:
        -----------
        data (np.ndarray): The DKIST data
        header (astropy.io.fits.Header): The FITS header containing neccessary keywords
        wavelength_index (int): The index of the wavelength slice

        Returns:
        --------
        spatial_x (list): The x coordinates of the spatial data
        spatial_y (list): The y coordinates of the spatial data
        """
        cdelt3 = header["CDELT3"]
        cdelt1 = header["CDELT1"]
        crpix1 = header["CRPIX1"]
        crpix3 = header["CRPIX3"]
        crval1 = header["CRVAL1"]
        crval3 = header["CRVAL3"]
        pc1_1 = header["PC1_1"]
        pc1_3 = header["PC1_3"]
        pc3_1 = header["PC3_1"]
        pc3_3 = header["PC3_3"]

        slit = data[0, wavelength_index, :]

        x_pixel = 0
        spatial_x = []
        spatial_y = []

        for i in range(slit.size):
            if not np.isnan(slit[i]):
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
