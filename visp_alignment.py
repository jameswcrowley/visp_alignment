from pathlib import Path

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
        self.wavelength_index = wavelength_index
        self.verbose = verbose
        



# TODO: I think it would be good to add a class for holding data to clean up the code. But we can defer this for now and focus on getting the alignment working first.

class DataLoader:
    """
    Loads the DKIST and HMI data, extracts neccessary info from headers or shapes, saves it to be used later
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get_time(self, changing_keywords):
        """
        The method uses the path_to_dkist_data from the configuration to locate the DKIST .fits files.

        Returns:
        ---------
        tuple: A tuple containing the start and end times of the DKIST data in the folder
        """
        return (changing_keywords["DATE-AVG"][0], changing_keywords["DATE-AVG"][-1])

    def normalize(self, arr):
        normalized = arr
        normalized = (normalized- np.nanmedian(normalized)) / (np.nanmax(normalized) - np.nanmin(normalized))
        return normalized

    def load_hmi(self, start_time: Time, end_time: Time):
        """
        Downloads all HMI data within one minute of the passed time interval

        Parameters:
        -----------
        start_time (Time): The start time of the interval
        end_time (Time): The end time of the interval

        """
        search_results = Fido.search(
            a.Instrument.hmi,
            a.Physobs("intensity"),
            a.Time(start_time - 1 * u.minute, end_time + 1 * u.minute),
        )

        hmi_files = Fido.fetch(
            search_results, path=self.cfg.path_to_sunpy, progress=self.cfg.verbose, site="NSO"
        )

        hmi_coordinates_and_data = []
        hmi_times = []

        for file in hmi_files[0:2]:
            hmi = map.Map(file)

            coordinates = map.all_coordinates_from_map(hmi)

            hmix = coordinates.Tx
            hmiy = coordinates.Ty
            
            hmi_data = hmi.data

            image_time = Time(hmi.date)

            hmi_coordinates_and_data.append((hmix, hmiy, hmi_data))

            hmi_times.append(image_time)
        
        hmi_times = sorted(hmi_times)

        return hmi_coordinates_and_data, hmi_times
    
    def find_nearest_hmi(self, target, hmi_coordinates_and_data, hmi_times):
        idx = bisect.bisect_left(hmi_times, target)
        if idx == 0:
            best = 0
        elif idx == len(hmi_times):
            best = idx - 1
        else:
            before, after = hmi_times[idx - 1], hmi_times[idx]
            best = idx - 1 if (target - before) <= (after - target) else idx
        return hmi_coordinates_and_data[best]

    def get_dkist_wavelengths(self): #read in data step 1
        """
        Returns a 2D spatial intensity map where each pixel contains the average
        intensity of the brightest/most intense wavelength samples (above 95th percentile) from the DKIST dataset.
        Returns:
        ----------
        mean_data(numpy.ndarray): A 2D array of the mean intensity values across the wavelength samples above the threshold of 95th percentile
        """
        first_file_path = next(Path(self.cfg.path_to_dkist_data).glob("*.fits"))
        slit_1_data = fits.open(first_file_path)[1].data[0]
        # print("Shape of slit data", slit_1_data.shape)

        median_wavelength_data = np.nanmedian(slit_1_data, axis = 1) # median spectra for all slits and positions along slits
        # print("Median_wavelength_data", median_wavelength_data.shape, median_wavelength_data)

        threshold = np.nanpercentile(median_wavelength_data, 95) # 95th percentile of the median spectra values
        # print("threshold", threshold)
        wavelength_indicies = np.where(median_wavelength_data > threshold)[0] # gets the indicies of the median spectra is above the threshold
        # print("wavelength indicies", wavelength_indicies)

        asdf_path = next(Path(self.cfg.path_to_dkist_data).glob("*.asdf"))
        ds = dkist.load_dataset(asdf_path)

        # print(f'Dataset loaded from {asdf_path} with shape {ds[:, :, :, :].data.shape}')
        #TODO: figure out how to optimize getting data from ds
        all_data = None
        if ds.wcs.pixel_n_dim == 4:
            all_data = np.array(ds[0, :, :, :].data) # load all slits, slit positions, and wavelenghths of the dataset across the first Stoke/Raster
        elif ds.wcs.pixel_n_dim == 5:
            all_data = np.array(ds[0, 0, :, :, :].data) # load all slits, slit positions, and  wavelenghths of the dataset across the first Stoke and raster
        else:
            all_data = np.array(ds[:, :, :].data)
        
        print("Shape of all the data", all_data.shape)
        print("Checks if there are any non nan values", not np.isnan(all_data).all())
 
        relevant_data = all_data[:, wavelength_indicies, :] # gets the data of all wavelengths across all slits for the indicies above the threshold
        print(50 * '-')
        print("Shape of the relevant data", relevant_data.shape)
        # print("relevant Data: ", relevant_data)
        print(50 * '-')
        print("Checks if there are any non nan values", not np.isnan(relevant_data).all())

        mean_data = np.nanmean(relevant_data, axis = 1) # mean of the data across the wavelength samples above the threshold
        mean_data = self.normalize(mean_data)
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

        data = None
        if ds.wcs.pixel_n_dim == 4:
            data = np.array(ds[0, :, :30, :].data)
        elif ds.wcs.pixel_n_dim == 5:
            data = np.array(ds[0, 0, :, :30, :].data)
        else:
            data = np.array(ds[:, :30, :].data)

        mean_data = np.nanmean(data, axis = 1)
        mean_data = self.normalize(mean_data)
        return mean_data

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
        fits_files = [x for x in sorted(os.listdir(self.cfg.path_to_dkist_data)) if '.fits' in x and '_I_' in x]
        
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
        

    def load(self):
        """
        Main for the DataLoader class.

        Returns:
        --------
        fixed_keywords(dict): A dictionary of fixed keywords {key: value}
        changing_keywords(dict of lists): A dictionary of lists containing the changing keywords {key: [values]}
        fits_files(list): A list of the fits files in the DKIST data folder
        intensities(numpy.ndarray): A 2D array of the mean intensity values across the wavelength samples above the threshold of 95th percentile
        hmix(numpy.ndarray): the x coordinates of the HMI data
        hmiy(numpy.ndarray): the y coordinates of the HMI data
        hmi_data(numpy.ndarray): the intensity data of the HMI data
        """

        self.fixed_keywords, self.changing_keywords, self.fits_files = self.get_dkist_headers()
        self.start_time, self.end_time = self.get_time(self.changing_keywords)
        self.intensities = self.get_dkist_wavelengths()

        hmi_coordinates_and_data, hmi_times = self.load_hmi(Time(self.start_time), Time(self.end_time))

        middle_image_time = hmi_times[len(hmi_times)//2]

        self.hmix, self.hmiy, self.hmi_data = self.find_nearest_hmi(middle_image_time, hmi_coordinates_and_data, hmi_times)

class Alignment:
    """
    Takes an instance of the Data class and interpolates the HMI data to the resolution / pixel grid of DKIST
    """

    def __init__(self, cfg: Config, data_loader: DataLoader):
        self.cfg = cfg
        self.data_loader = data_loader
    
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
        crval1_shift, crval3_shift, pc1_1_shift, pc3_1_shift, pc1_3_shift, pc3_3_shift = parameters

        cdelt1 = fixed_keywords['CDELT1']
        cdelt3 = fixed_keywords['CDELT3']

        pc1_1 = fixed_keywords['PC1_1'] + pc1_1_shift
        pc1_3 = fixed_keywords['PC1_3'] + pc1_3_shift
        pc3_1 = fixed_keywords['PC3_1'] + pc3_1_shift
        pc3_3 = fixed_keywords['PC3_3'] + pc3_3_shift

        nx = fixed_keywords['DNAXIS3']
        ny = fixed_keywords['DNAXIS1']

        coords = np.zeros((nx, ny, 2))

        #i = np.arange(nx)[:, None] + 1
        i = np.ones(nx)[:, None]
        j = np.arange(ny)[None, :] + 1

        #TODO: Figure out what to do about raster repeats
        crval1 = np.asarray(changing_keywords["CRVAL1"])[:nx, None] + crval1_shift
        crval3 = np.asarray(changing_keywords["CRVAL3"])[:nx, None] + crval3_shift
        crpix1 = np.asarray(changing_keywords["CRPIX1"])[:nx, None]
        crpix3 = np.asarray(changing_keywords["CRPIX3"])[:nx, None]

        x = crval3 + cdelt3 * (pc3_3 * (i - crpix3) + pc3_1 * (j - crpix1))

        y = crval1 + cdelt1 * (pc1_3 * (i - crpix3) + pc1_1 * (j - crpix1))

        coords[:, :, 0] = x
        coords[:, :, 1] = y

        return coords
    
    def identify_relevant_hmi_data(self, coords, hmix, hmiy, hmi_data, delta = 20):
        """
        This function identifies the relevant HMI data that overlaps with the DKIST data, with a buffer of delta arcseconds on each side. 
        It returns the x and y coordinates of the relevant HMI data, as well as the normalized intensity data.
        
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
        relevant_hmi_data = self.data_loader.normalize(relevant_hmi_data)

        return relevant_hmix, relevant_hmiy, relevant_hmi_data
    
    def construct_interpolator(self, relevant_hmix, relevant_hmiy, relevant_hmi_data):
        """
        This function constructs a RegularGridInterpolator object from the relevant HMI data. It returns the interpolator object.

        Parameters:
        -----------
        relevant_hmix (numpy.ndarray): the x coordinates of the relevant HMI data.
        relevant_hmiy (numpy.ndarray): the y coordinates of the relevant HMI data.
        relevant_hmi_data (numpy.ndarray): the intensity data of the relevant HMI data.

        Returns:
        --------
        interpolator (scipy.interpolate.RegularGridInterpolator): a RegularGridInterpolator object that can be used to interpolate the relevant HMI data onto any set of coordinates.
        """

        interpolator = interp.RegularGridInterpolator(
            (relevant_hmiy, relevant_hmix),
            relevant_hmi_data, 
            method='cubic',
            bounds_error=False, 
            fill_value=np.nan
        )

        return interpolator
        

    def interpolate_hmi_to_coords(self, interpolator, coords):
        """
        This function interpolates the relevant HMI data onto the DKIST coordinates. It returns a 2D array of the interpolated HMI data, with shape (nx, ny), where nx is the number of slits and ny is the number of pixels along the slit.
        
        Parameters:
        -----------
        interpolator (scipy.interpolate.RegularGridInterpolator): a RegularGridInterpolator object that can be used to interpolate the relevant HMI data onto any set of coordinates.
        coords (numpy.ndarray): a 3D array of the coordinates of the DKIST data, with shape (nx, ny, 2), where nx is the number of slits, ny is the number of pixels along the slit, and 2 is for the x and y coordinates.
        
        Returns:
        --------
        Z_fine (numpy.ndarray): a 2D array of the interpolated HMI data, with shape (nx, ny), where nx is the number of slits and ny is the number of pixels along the slit.
        """
        HMI_interpolated_to_coords = interpolator((coords[:, :, 1], coords[:, :, 0]))

        return HMI_interpolated_to_coords

    def loss_function(self, parameters, data_numpy, interpolator):
        """
        This function calculates the loss between the interpolated HMI data and the DKIST data.
        It returns the loss value - here, I chose to use the sum of squared differences to quantify the difference between the two datasets, but other metrics could be used as well.

        Parameters:
        -----------
        parameters (tuple): a tuple of the parameters to modify the crval and pc values.
        interpolator (scipy.interpolate.RegularGridInterpolator): a RegularGridInterpolator object that can be used to interpolate the relevant HMI data onto any set of coordinates.
        data_numpy (numpy.ndarray): the DKIST data.

        Returns:
        --------
        loss (float): the loss value between the interpolated HMI data and the DKIST data.
        """
        # shift the DKIST coordinates based on the input parameters, identify the relevant HMI data that overlaps with the DKIST data, and interpolate the HMI data onto the DKIST coordinates.
        coords_new = self.construct_dkist_coords(self.data_loader.fixed_keywords, self.data_loader.changing_keywords, parameters)

        HMI_interpolated_to_coords = self.interpolate_hmi_to_coords(interpolator, coords_new)

        x = HMI_interpolated_to_coords
        y = data_numpy

        mask = ~np.isnan(x) & ~np.isnan(y)
        xv, yv = x[mask], y[mask]

        xv = xv - np.mean(xv) 
        yv = yv - np.mean(yv)
        corr = np.sum(xv*yv) / np.sqrt(np.sum(xv*xv) * np.sum(yv*yv))

        loss = -corr

        return loss
    
    def align(self, initial_guess, bounds):
        """
        This function performs the optimization to find the best parameters to align the DKIST data with the HMI data.
        It returns the best parameters found by the optimization.

        Parameters:
        -----------
        initial_guess (tuple): a tuple of the initial guess for the parameters to modify the crval and pc values.
        bounds (list): a list of tuples specifying the bounds for each parameter.

        Returns:
        --------
        best_parameters (tuple): a tuple of the best parameters found by the optimization.
        """

        initial_coordinates = self.construct_dkist_coords(self.data_loader.fixed_keywords, self.data_loader.changing_keywords, initial_guess)
        relevant_hmix, relevant_hmiy, relevant_hmi_data = self.identify_relevant_hmi_data(initial_coordinates, self.data_loader.hmix, self.data_loader.hmiy, self.data_loader.hmi_data)
        interpolator = self.construct_interpolator(relevant_hmix, relevant_hmiy, relevant_hmi_data)
        
        result = opt.minimize(self.loss_function, 
            initial_guess, 
            args=(self.data_loader.intensities, interpolator), 
            bounds=bounds, 
            method='Powell', 
            options={'maxiter': 200, 'disp': True}
        )

        best_parameters = result.x
        
        return best_parameters, result.success


if __name__ == "__main__":

    run = True

    print("Run =", run)
    
    # path_to_dkist_data = "/Users/joshua/projects/nso/dkist-data/pid_3_35/XVNDZY"
    # path_to_sunpy = "~/sunpy/data/"

    #path_to_dkist_data = "C:\\Projects\\DkistData\\pid_3_31\\KRBVTD\\"
    #path_to_sunpy = "C:\\Users\\owner\\sunpy\\data\\"
    
    path_to_dkist_data = "/Users/jamescrowley/Documents/summer_2026/research/pid_3_35/XVNDZY"
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
    loader.load()

    # Minimize
    print("ALIGNING")
    
    alignment = Alignment(cfg, loader)
    original_dkist_coords = alignment.construct_dkist_coords(loader.fixed_keywords, loader.changing_keywords, [0, 0, 0, 0, 0, 0])


    if run:
        initial_guess = [-10, 15, 0, 0, 0, 0]

        bounds = [(-30, 30), (-30, 30), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
        best_parameters, success = alignment.align(initial_guess, bounds)

        print('Optimization converged:', success)
        print('Best parameters found:', best_parameters)

    else:
        best_parameters = [-1.60380959e+00,  1.01922364e+01, -9.42824916e-04, -1.06372480e-02, 2.33890820e-02,  1.35416594e-02]

    # assemble final coordinates

    coords_new = alignment.construct_dkist_coords(loader.fixed_keywords, loader.changing_keywords, best_parameters)
    relevant_hmix, relevant_hmiy, relevant_hmi_data = alignment.identify_relevant_hmi_data(coords_new, loader.hmix, loader.hmiy, loader.hmi_data)
    interpolator = alignment.construct_interpolator(relevant_hmix, relevant_hmiy, relevant_hmi_data)
    
    final_HMI_interpolated_onto_coords = alignment.interpolate_hmi_to_coords(interpolator, coords_new)

    plt.figure(figsize = [10, 10])
    plt.subplot(2,2,1)
    plt.title('Original DKIST data overlayed over original HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(original_dkist_coords[:, :, 0], original_dkist_coords[:, :, 1], loader.intensities, cmap = 'plasma', alpha = 0.8)
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.title('New (interpolated) HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(coords_new[:, :, 0], coords_new[:, :, 1], final_HMI_interpolated_onto_coords, cmap = 'plasma')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title('Aligned DKIST data overlaid on original HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(coords_new[:, :, 0], coords_new[:, :, 1], loader.intensities, cmap = 'plasma', alpha = 0.8)
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.title('Difference between DKIST and HMI data')
    plt.imshow(relevant_hmi_data, extent = [relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap = 'grey', origin = 'lower')
    plt.pcolormesh(coords_new[:, :, 0], coords_new[:, :, 1], loader.intensities - final_HMI_interpolated_onto_coords, cmap = 'bwr', alpha = 1, vmin = -0.5, vmax = 0.5)
    plt.colorbar()

    print("DONE")

    plt.show()