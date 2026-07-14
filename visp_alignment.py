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



# TODO: I think it would be good to add a class for holding data to clean up the code. But we can defer this for now and focus on getting the alignment working first.

class DataLoader:
    """
    Loads the DKIST and HMI data, extracts neccessary info from headers or shapes, saves it to be used later
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_time(self):
        """
        The method uses the path_to_dkist_data from the configuration to locate the DKIST .fits files.

        Returns:
        ---------
        tuple: A tuple containing the start and end times of the DKIST data in the folder
        """
        fits_files = [
            filename for filename in sorted(os.listdir(self.cfg.path_to_dkist_data))
            if filename.endswith('.fits') and os.path.isfile(os.path.join(self.cfg.path_to_dkist_data, filename)) and '_I_' in filename
        ]


        first_path = os.path.join(self.cfg.path_to_dkist_data, fits_files[0])
        last_path = os.path.join(self.cfg.path_to_dkist_data, fits_files[-1])


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
        # TODO: switch to reading in the middle file of the search results instead of always taking the first one.
        # TODO: also need to eventually read in all HMI files 
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
        # TODO: add handling if there aren't multiple Stokes components in the dataset
        asdf_path = next(Path(self.cfg.path_to_dkist_data).glob("*.asdf"))
        ds = dkist.load_dataset(asdf_path)
        print(f'Dataset loaded from {asdf_path} with shape {ds[:, :, :, :].data.shape}')
        all_data = np.array(ds[0, :, :, :].data) # load all slits and wavelenghths of the dataset across the first Stokes parameter

        median_wavelength_data = np.nanmedian(all_data, axis = (0, 2)) # median spectra for all slits and positions along slits

        threshold = np.percentile(median_wavelength_data, 95) # 95th percentile of the median spectra values
        wavelength_indicies = np.where(median_wavelength_data > threshold)[0] # gets the indicies of the median spectra is above the threshold
 
        relevant_data = all_data[:, wavelength_indicies, :] # gets the data of all wavelengths across all slits for the indicies above the threshold
        print(50 * '-')
        print(relevant_data.shape)
        print(50 * '-')

        mean_data = np.nanmean(relevant_data, axis = 1) # mean of the data across the wavelength samples above the threshold
        return mean_data 
    
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
            filename for filename in sorted(os.listdir(self.cfg.path_to_dkist_data))
            if filename.endswith('.fits') and os.path.isfile(os.path.join(self.cfg.path_to_dkist_data, filename))
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

        # TODO: I did some research and found it would be more efficient to store these as numpy arrays, not Python lists.
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
        

class Alignment:
    """
    Takes an instance of the Data class and interpolates the HMI data to the resolution / pixel grid of DKIST
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def construct_dkist_coords(self, fixed_keywords, changing_keywords, parameters = (0, 0, 0, 0, 0, 0)):
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

        # TODO: pci_j's should be constrained so that the transformation matrix is invertable -> physically meaningful. We should switch to dx, dy, and rotation angles instead of directly manipulating the pci_j's.
        crval1_shift, crval3_shift, dx, dy, theta = parameters



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

        # TODO: this should be possible to vectorize instead of using a for loop, which would likely be MUCH faster for large datasets.
        for i in range(nx):
            crval1 = changing_keywords['CRVAL1'][i]
            crval3 = changing_keywords['CRVAL3'][i]

            crpix1 = changing_keywords['CRPIX1'][i]
            crpix3 = changing_keywords['CRPIX3'][i]

            # y-index of the position of the pixel ALONG the slit (i.e. 1 to ny). This is used to calculate the coordinates of each pixel along the slit.
            indices = np.linspace(1, ny, ny, dtype = int)

            # calculate the coordinates of each pixel along the slit using the header keywords and the y-index. 
            # This formula is the linear transformation-matrix form of the coordinate assembly.
            slit_coords_x = (crval3 + crval3_shift) + cdelt3 * ((pc3_3 + pc3_3_shift) * (i - (crpix3)) + (pc3_1 + pc3_1_shift) * (indices - (crpix1)))
            slit_coords_y = (crval1 + crval1_shift) + cdelt1 * ((pc1_3 + pc1_3_shift) * (i - (crpix3)) + (pc1_1 + pc1_1_shift) * (indices - (crpix1)))

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
        # TODO: should be much faster if we construct the interpolator ONCE and then feed in new coordinates, not reconstruct it every time we call this function.
        grid_interpolator = interp.RegularGridInterpolator(
            (relevant_hmiy, relevant_hmix),
            relevant_hmi_data, 
            method='cubic',
            bounds_error=False, 
            fill_value=np.nan
        )

        # perform the interpolation onto whatever coordiantes you give it. Here, we give it the DKIST coordinates.
        hMI_interpolated_to_coords = grid_interpolator((coords[:, :, 1], coords[:, :, 0]))

        return hMI_interpolated_to_coords

    def loss_function(self, parameters, fixed_keywords, changing_keywords, hmix, hmiy, hmi_data, data_numpy):
        """
        This function calculates the loss between the interpolated HMI data and the DKIST data.
        It returns the loss value - here, I chose to use the sum of squared differences to quantify the difference between the two datasets, but other metrics could be used as well.

        Parameters:
        -----------
        parameters (tuple): a tuple of the parameters to modify the crval and pc values.
        path_to_local_fits (str): the path to the local fits files.
        all_fits (list): a list of the fits files to load in.
        hmix (numpy.ndarray): the x coordinates of the HMI data.
        hmiy (numpy.ndarray): the y coordinates of the HMI data.
        hmi_data (numpy.ndarray): the intensity data of the HMI data.
        data_numpy (numpy.ndarray): the DKIST data.

        Returns:
        --------
        loss (float): the loss value between the interpolated HMI data and the DKIST data.
        """
        # shift the DKIST coordinates based on the input parameters, identify the relevant HMI data that overlaps with the DKIST data, and interpolate the HMI data onto the DKIST coordinates.
        coords_new = self.construct_dkist_coords(fixed_keywords, changing_keywords, parameters)
        relevant_hmix, relevant_hmiy, relevant_hmi_data = self.identify_relevant_hmi_data(coords_new, hmix, hmiy, hmi_data)

        # TODO: I think we should crop ONCE in a larger area around DKIST coords, and then pass it in once, not update the crop for every interpolation.
        HMI_interpolated_to_coords = self.interpolate_hmi_to_coords(relevant_hmix, relevant_hmiy, relevant_hmi_data, coords_new)

        # double-check the data is normalized so it can be better compared.
         
        # TODO: update this so that it does not modify the original arrays in-place, which could affect subsequent calculations. Consider creating copies before normalizing.
        HMI_interpolated_to_coords -= np.nanmean(HMI_interpolated_to_coords)
        HMI_interpolated_to_coords /= np.nanstd(HMI_interpolated_to_coords)
        HMI_interpolated_to_coords /= np.nanmax(HMI_interpolated_to_coords)

        data_numpy -= np.nanmean(data_numpy)
        data_numpy /= np.nanstd(data_numpy)
        data_numpy /= np.nanmax(data_numpy)

        # calculate the loss between the interpolated HMI data and the DKIST data. this could be something like mean squared error or mean absolute error. 
        # I think eventually, we want to switch to the cross-correlation function to quantify the difference between the two datasets, but for now, this is a simple example.

        # TODO: move to cross correlation
        loss = np.nansum((HMI_interpolated_to_coords - data_numpy)**2)

        return loss


if __name__ == "__main__":

    run = False

    print("Run =", run)
    
    path_to_dkist_data = "/Users/joshua/projects/nso/dkist-data/pid_3_35/XVNDZY"
    path_to_sunpy = "~/sunpy/data/"

    cfg = Config(
    path_to_dkist_data=path_to_dkist_data, 
    path_to_sunpy=path_to_sunpy, 
    wavelength_index=30, 
    verbose=True
    )

    # Load and prepare  

    print("LOADING DATA")

    loader = DataLoader(cfg)

    fixed, changing, fits_files = loader.get_dkist_headers()

    intensities = loader.get_dkist_wavelengths2()

    intensities /= np.nanmax(intensities) # normalizing it to be between 0 and 1 for the optimization process

    hmix, hmiy, hmi_data, time = loader.load_hmi(Time(changing["DATE-AVG"][0]), Time(changing["DATE-AVG"][4]))

    hmi_data /= np.nanmax(hmi_data) # normalizing it to be between 0 and 1 for the optimization process

    alignment = Alignment(cfg)

    original_dkist_coords = alignment.construct_dkist_coords(fixed, changing)

    relevant_hmix, relevant_hmiy, relevant_hmi_data = alignment.identify_relevant_hmi_data(original_dkist_coords, hmix, hmiy, hmi_data)

    # Minimize

    print("ALIGNING")

    initial_guess = [-15, 15, 0, 0, 0, 0]
    bounds = [(-20, -10), (0, 30), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

    if run:
        result = opt.minimize(alignment.loss_function, initial_guess, args=(fixed, changing, hmix, hmiy, hmi_data, intensities), bounds=bounds, method='Powell', options={'maxiter': 200, 'disp': True})
        best_parameters = result.x

        print('Optimization converged:', result.success)
        print('Best parameters found:', best_parameters)

    else:
        best_parameters = [-1.02523844e+01,  1.22342979e+01, -4.36652273e-02,  9.43861732e-03, -2.58909492e-01,  1.87743852e-02]

    # assemble final coordinates

    coords_new = alignment.construct_dkist_coords(fixed, changing, best_parameters)
    relevant_hmix, relevant_hmiy, relevant_hmi_data = alignment.identify_relevant_hmi_data(coords_new, hmix, hmiy, hmi_data)
    Z_fine = alignment.interpolate_hmi_to_coords(relevant_hmix, relevant_hmiy, relevant_hmi_data, coords_new)

    # plt.figure(figsize = [12, 10])
    # plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    # plt.pcolormesh(assembled_dkist_coords[:, :, 0], assembled_dkist_coords[:, :, 1], intensities, cmap = 'plasma', alpha=1)

    plt.figure(figsize = [25, 8])
    plt.subplot(1,4,1)
    plt.title('Original DKIST data overlayed over original HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(original_dkist_coords[:, :, 0], original_dkist_coords[:, :, 1], intensities, cmap = 'plasma')

    plt.subplot(1,4,2)
    plt.title('New (interpolated) HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(coords_new[:, :, 0], coords_new[:, :, 1], Z_fine, cmap = 'plasma')

    plt.subplot(1,4,3)
    plt.title('Aligned DKIST data overlaid on original HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(coords_new[:, :, 0], coords_new[:, :, 1], intensities, cmap = 'plasma', alpha = 1)

    plt.subplot(1,4,4)
    plt.title('Difference between DKIST and HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(coords_new[:, :, 0], coords_new[:, :, 1], intensities - Z_fine, cmap = 'bwr', alpha = 1, vmin = -0.5, vmax = 0.5)

    plt.show()

    # best parameters [-1.02523844e+01  1.22342979e+01 -4.36652273e-02  9.43861732e-03 -2.58909492e-01  1.87743852e-02]