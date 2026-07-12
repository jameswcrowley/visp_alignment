from pathlib import Path

import numpy as np
import astropy.io.fits as fits
from astropy.time import Time, TimeDelta
import astropy.units as u
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a
from sunpy import map as map
import sunpy.data.sample
import dkist.net
import dkist
import os
from scipy import interpolate as interp
import scipy.optimize as opt




class Config:
    """
    Specifies all settings, paths, and other metaparameters used in the alignment
    """

    def __init__(
        self,
        path_to_dkist_data: str,
        path_to_sunpy: str,
        wavelength_index = None,
        verbose = False
    ):
        self.path_to_dkist_data = path_to_dkist_data
        self.path_to_sunpy = path_to_sunpy
        self.verbose = verbose
        self.wavelength_index = wavelength_index



class DataLoader:
    """
    Loads the DKIST and HMI data, extracts neccessary info from headers or shapes, saves it to be used later
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_time(self, folder_path):
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

    def get_dkist_wavelengths(self): #read in data step 1
        """
        Returns a 2D spatial intensity map where each pixel contains the average
        intensity of the brightest/more intense wavelength samples (above 95th percentile) from the DKIST dataset.
        Returns:
        ----------
        mean_data(numpy.ndarray): A 2D array of the mean intensity values across the wavelength samples above the threshold of 95th percentile
        """
        asdf_path = next(Path(self.cfg.path_to_dkist_data).glob("*.asdf"))
        ds = dkist.load_dataset(asdf_path)
        all_data = np.array(ds[0, :, :, :].data)

        slit1_data = all_data[0, :, :] #data of all wavelengths across the first slit
        print(slit1_data.shape)

        median_data = np.array([]) #median of each wavelength sample across the first slit
        for i in range(slit1_data.shape[0]):
            median_data = np.append(median_data, np.nanmedian(slit1_data[i, :]))
        print(median_data)

        threshold = np.percentile(median_data, 95) #95th percentile of the median values
        indicies = np.where(median_data > threshold)[0] #gets the indicies of the median values above the threshold
        print(indicies)
 
        relevant_data = all_data[:, indicies, :] #gets the data of all wavelengths across all slits for the indicies above the threshold
        # for i in indicies:
        #     data = np.append(data, np.array(ds[0, :, i, :].data))
        print(relevant_data)


        mean_data = np.nanmean(relevant_data, axis = 0) #mean of the data across the wavelength samples above the threshold
        return mean_data 

        # data = np.asarray(ds.data)
        # if data.ndim == 0:
        #     raise ValueError("The DKIST dataset did not contain any data")

        # wavelength_axis = None
        # for axis, axis_name in enumerate(getattr(ds, "world_axis_physical_types", [])):
        #     if axis_name is not None and ("wavelength" in axis_name.lower() or "em.wl" in axis_name.lower()):
        #         wavelength_axis = axis
        #         break

        # if wavelength_axis is None:
        #     wavelength_axis = 1 if data.ndim > 1 else 0

        # wavelength_data = np.moveaxis(data, wavelength_axis, 0)
        # wavelength_data = wavelength_data[:30]
        # intensity_map = np.nanmean(wavelength_data, axis=0)
        # intensity_map = np.squeeze(intensity_map)

        # if intensity_map.ndim > 2:
        #     intensity_map = np.nanmean(
        #         intensity_map,
        #         axis=tuple(range(intensity_map.ndim - 2)),
        #     )

        # if intensity_map.ndim != 2:
        #     raise ValueError(
        #         f"Expected a 2D intensity map, but received shape {intensity_map.shape}"
        #     )

        # return intensity_map

    
    def get_dkist_wavelengths2(self):
        """
        Returns a 2D spatial intensity map where each pixel contains the average
        intensity of the first 30 wavelengths
        ----------
        numpy.ndarray: A 2D array of the mean intensity values across the first 30 wavelengths
        """
        asdf_path = next(Path(self.cfg.path_to_dkist_data).glob("*.asdf"))
        ds = dkist.load_dataset(asdf_path)

        data = np.array(ds[0, :, :30, :].data)
        # print("original data", data)
        return np.nanmean(data, axis = 1)

    def get_dkist_headers(self): #read in data step 1
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
            filename for filename in os.listdir(self.cfg.path_to_dkist_data)
            if filename.endswith('.fits') and "_I_" in filename and os.path.isfile(os.path.join(self.cfg.path_to_dkist_data, filename))
        ]
        header = fits.open(os.path.join(self.cfg.path_to_dkist_data, fits_files[0]))[1].header
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
            header = fits.open(os.path.join(self.cfg.path_to_dkist_data, slit_i))[1].header
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
    
    def construct_dkist_coords(self, fixed_keywords, changing_keywords):
        """
        This function constructs the default coordinates of the DKIST data from the fits files. 
        It returns a 3D array of the coordinates, with shape (nx, ny, 2), where nx is the number of slits, ny is the number of pixels along the slit, and 2 is for the x and y coordinates.
        
        Parameters:
        -----------
        fixed_keywords (dict): a dictionary of the fixed keywords from the DKIST headers, with keys 'CDELT1', 'CDELT3', 'PC1_1', 'PC1_3', 'PC3_1', 'PC3_3', 'DNAXIS3', and 'DNAXIS1'.
        changing_keywords (dict): a dictionary of the changing keywords from the DKIST headers, with keys 'CRVAL1', 'CRVAL3', 'CRPIX1', 'CRPIX3', and 'DATE-AVG'. Each key maps to a list of values, one for each slit.

        Returns:
        --------
        coords_new (numpy.ndarray): a 3D array of the coordinates, with shape (nx, ny, 2), where nx is the number of slits, ny is the number of pixels along the slit, and 2 is for the x and y coordinates.
        """

        cdelt1 = fixed_keywords['CDELT1']
        cdelt3 = fixed_keywords['CDELT3']

        pc1_1 = fixed_keywords['PC1_1']
        pc1_3 = fixed_keywords['PC1_3']
        pc3_1 = fixed_keywords['PC3_1']
        pc3_3 = fixed_keywords['PC3_3']

        nx = fixed_keywords['DNAXIS3']
        ny = fixed_keywords['DNAXIS1']

        # initialize an empty array to fill with the coordinates. the shape is (nx, ny, 2) because we have nx by ny pixels and each pixel has an x and y coordinate.
        coords = np.zeros((nx, ny, 2))


        # loop through each fits file, extract the relevant header keywords, and calculate the coordinates for each pixel in that slit. then fill the coords_new array with those coordinates.
        for i in range(nx):
            crval1 = changing_keywords['CRVAL1'][i]
            crval3 = changing_keywords['CRVAL3'][i]

            crpix1 = changing_keywords['CRPIX1'][i]
            crpix3 = changing_keywords['CRPIX3'][i]

            # y-index of the position of the pixel ALONG the slit (i.e. 1 to ny). This is used to calculate the coordinates of each pixel along the slit.
            indices = np.linspace(1, ny, ny, dtype = int)

            # calculate the coordinates of each pixel along the slit using the header keywords and the y-index. 
            # This formula is the linear transformation-matrix form of the coordinate assembly.
            slit_coords_x = crval3 + cdelt3 * (pc3_3 * (i - crpix3) + pc3_1 * (indices - crpix1))
            slit_coords_y = crval1 + cdelt1 * (pc1_3 * (i - crpix3) + pc1_1 * (indices - crpix1))

            # save each pixel's coordinates into x and y indices of the the coords_new array. 
            coords[i, :, 0] = slit_coords_x
            coords[i, :, 1] = slit_coords_y

        return coords
    
    def identify_relevant_hmi_data(self, coords, hmix, hmiy, hmi_data, delta = 20):
        """
        This function identifies the relevant HMI data that overlaps with the DKIST data, with a buffer of delta arcseconds on each side. 
        It returns the x and y coordinates of the relevant HMI data, as well as the intensity data.
        
        Parameters:
        -----------
        coords_new (numpy.ndarray): a 3D array of the coordinates of the DKIST data, with shape (nx, ny, 2), where nx is the number of slits, ny is the number of pixels along the slit, and 2 is for the x and y coordinates.
        hmix (numpy.ndarray): the x coordinates of the HMI data.
        hmiy (numpy.ndarray): the y coordinates of the HMI data.
        hmi_data (numpy.ndarray): the intensity data of the HMI data.
        delta (float): the buffer in arcseconds to add to the DKIST data when identifying the relevant HMI data. Default is 20 arcseconds.
        
        Returns:
        --------
        relevant_hmix (numpy.ndarray): the x coordinates of the relevant HMI data.
        relevant_hmiy (numpy.ndarray): the y coordinates of the relevant HMI data.
        relevant_hmi_data (numpy.ndarray): the intensity data of the relevant HMI data.
        """
        # construct a box of HMI coordinates and data around the DKIST data, with a buffer of delta arcseconds on each side. 
        relevant_hmix_indices = sorted([np.argmin(np.abs(np.min(coords[:, :, 0]) - delta - hmix[0, :].value)), np.argmin(np.abs(np.max(coords[:, :, 0]) + delta - hmix[0, :].value))])
        relevant_hmiy_indices = sorted([np.argmin(np.abs(np.min(coords[:, :, 1]) - delta - hmiy[:, 0].value)), np.argmin(np.abs(np.max(coords[:, :, 1]) + delta - hmiy[:, 0].value))])

        # crop the x and y coordinates of the HMI data to the relevant coordinates. Note that the HMI data is transposed because the x and y coordinates are in the opposite order of the data array.
        relevant_hmix = hmix[0, relevant_hmix_indices[0]:relevant_hmix_indices[1]].value
        relevant_hmiy = hmiy[relevant_hmiy_indices[0]:relevant_hmiy_indices[1], 0].value

        # crop the HMI data to the relevant coordinates. Note that the HMI data is transposed because the x and y coordinates are in the opposite order of the data array.
        relevant_hmi_data= hmi_data.T[relevant_hmix_indices[0]:relevant_hmix_indices[1], relevant_hmiy_indices[0]:relevant_hmiy_indices[1]].T

        return relevant_hmix, relevant_hmiy, relevant_hmi_data

    def interpolate_hmi_to_coords(self, relevant_hmix, relevant_hmiy, relevant_hmi_data, coords):
        """
        This function interpolates the relevant HMI data onto the DKIST coordinates. It returns a 2D array of the interpolated HMI data, with shape (nx, ny), where nx is the number of slits and ny is the number of pixels along the slit.
        
        Parameters:
        -----------
        relevant_hmix (numpy.ndarray): the x coordinates of the relevant HMI data.
        relevant_hmiy (numpy.ndarray): the y coordinates of the relevant HMI data.
        relevant_hmi_data (numpy.ndarray): the intensity data of the relevant HMI data.
        coords (numpy.ndarray): a 3D array of the coordinates of the DKIST data, with shape (nx, ny, 2), where nx is the number of slits, ny is the number of pixels along the slit, and 2 is for the x and y coordinates.
        
        Returns:
        --------
        Z_fine (numpy.ndarray): a 2D array of the interpolated HMI data, with shape (nx, ny), where nx is the number of slits and ny is the number of pixels along the slit.
        """
        # an example of one way to interpolate the HMI data onto the DKIST coordinates. 
        # I don't know if this is the final method we'll use but it is a working example.
        
        # set up the interpolator:
        grid_interpolator = interp.RegularGridInterpolator(
            (relevant_hmiy, relevant_hmix),
            relevant_hmi_data, 
            method='cubic',
            bounds_error=False, 
            fill_value=np.nan
        )

        # perform the interpolation onto whatever coordiantes you give it. Here, we give it the DKIST coordinates.
        Z_fine = grid_interpolator((coords[:, :, 1], coords[:, :, 0]))

        return Z_fine


class Alignment:
    """
    Takes an instance of the Data class and the interpolated data from the Interpolator class to actually perform the alignment of the two datasets
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg


if __name__ == "__main__":

    """
    Placeholder code to test that the funcitons work: 
    """
    
    path_to_dkist_data = '/Users/jamescrowley/Documents/summer_2026/research/pid_3_35/XVNDZY/'
    path_to_sunpy = "~/sunpy/data/"

    cfg = Config(
    path_to_dkist_data=path_to_dkist_data, 
    path_to_sunpy=path_to_sunpy, 
    wavelength_index=30, 
    verbose=True
    )

    loader = DataLoader(cfg)


    times = loader.get_time(cfg.path_to_dkist_data)
    print("Start time:", times[0])
    print("End time:", times[1])

    intensity_map = loader.get_dkist_wavelengths()
    fixed_keywords, changing_keywords, fits_files = loader.get_dkist_headers()

    interpolator = Interpolator(cfg)

    coordinates = interpolator.construct_dkist_coords(fixed_keywords, changing_keywords)
    x = coordinates[:, :, 0]
    y = coordinates[:, :, 1]

    print(intensity_map.shape)

    plt.figure(figsize=(12, 4))


    img = plt.pcolormesh(x, y, intensity_map, shading='auto', cmap='magma', aspect='equal')

    plt.colorbar(img, label='Intensity')
    plt.xlabel('Spatial Y Axis (2556 channels / Arcseconds)')
    plt.ylabel('Spatial X Axis (200 channels / Arcseconds)')
    plt.title(f'Monochromatic Spatial Map at Wavelength Index {cfg.wavelength_index}')
    plt.show()

    print("done")
