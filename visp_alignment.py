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
from itertools import chain

class Config:
    """
    Specifies all settings, paths, and other metaparameters used in the alignment
    """

    def __init__(
        self,
        path_to_dkist_data: str,
        path_to_sunpy: str,
        wavelength_index = None,
        raster_repeats = None,
        verbose = False
    ):
        self.path_to_dkist_data = path_to_dkist_data
        self.path_to_sunpy = path_to_sunpy
        self.wavelength_index = wavelength_index
        self.raster_repeats = raster_repeats
        self.verbose = verbose

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        



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

    def _normalize_raster_repeat_selection(self, repeat_count):
        selection = self.cfg.raster_repeats

        if selection is None:
            return list(range(repeat_count))

        if isinstance(selection, (int, np.integer)):
            requested = [int(selection)]
        else:
            requested = [int(value) for value in selection]

        if not requested:
            raise ValueError("raster_repeats must select at least one repeat")

        if any(value < 0 for value in requested):
            raise ValueError("raster_repeats cannot contain negative repeat indices")

        if 0 in requested:
            selected = requested
        elif max(requested) <= repeat_count:
            selected = [value - 1 for value in requested]
        else:
            selected = requested

        if any(value >= repeat_count for value in selected):
            raise ValueError(
                f"Requested raster repeat(s) {requested} exceed available repeats {repeat_count}"
            )

        return selected

    def normalize(self, arr):
        arr = np.asarray(arr, dtype=float)

        # Use robust percentile limits so isolated bad pixels do not dominate scaling.
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return np.full_like(arr, np.nan, dtype=float)

        p_low, p_high = np.nanpercentile(arr[finite_mask], [1.0, 99.0])
        clipped = np.clip(arr, p_low, p_high)

        center = np.nanmedian(clipped)
        scale = p_high - p_low

        # Avoid divide-by-zero for near-constant arrays.
        if not np.isfinite(scale) or scale <= 0:
            return clipped - center

        normalized = (clipped - center) / scale
        return normalized

    def load_hmi(self, start_time: Time, end_time: Time):
        """
        Downloads all HMI data within one minute of the passed time interval

        Parameters:
        -----------
        start_time (Time): The start time of the interval
        end_time (Time): The end time of the interval

        """
        self.cfg.log(f"Searching HMI data from {start_time.isot} to {end_time.isot}")

        search_results = Fido.search(
            a.Instrument.hmi,
            a.Physobs("intensity"),
            a.Time(start_time - 1 * u.minute, end_time + 1 * u.minute),
        )

        hmi_files = Fido.fetch(
            search_results, path=self.cfg.path_to_sunpy, progress=self.cfg.verbose, site="NSO"
        )
        self.cfg.log(f"Fetched {len(hmi_files)} HMI files")

        hmi = map.Map(hmi_files[len(hmi_files)//2])

        coordinates = map.all_coordinates_from_map(hmi)

        middle_hmix = coordinates.Tx
        middle_hmiy = coordinates.Ty
        
        middle_hmi_data = hmi.data

        middle_hmi_data = self.normalize(middle_hmi_data)

        return middle_hmix, middle_hmiy, middle_hmi_data, hmi_files
    
    def get_all_hmi_times(self, hmi_files):
        hmi_times = []

        for file in hmi_files:
            hmi = map.Map(file)
            
            image_time = Time(hmi.date)

            hmi_times.append(image_time)

        self.hmi_times = hmi_times
        self.cfg.log(f"Loaded timestamps for {len(hmi_times)} HMI files")

    def get_dkist_wavelengths(self):
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
        selected_repeat_indices = getattr(self, "selected_raster_repeats", [0])

        # print(f'Dataset loaded from {asdf_path} with shape {ds[:, :, :, :].data.shape}')
        #TODO: figure out how to optimize getting data from ds
        all_data = None
        if ds.wcs.pixel_n_dim == 4:
            all_data = np.array(ds[0, :, :, :].data) # load all slits, slit positions, and wavelenghths of the dataset across the first Stoke/Raster
        elif ds.wcs.pixel_n_dim == 5:
            repeat_slices = [np.array(ds[0, repeat_index, :, :, :].data) for repeat_index in selected_repeat_indices]
            all_data = np.concatenate(repeat_slices, axis=0) if len(repeat_slices) > 1 else repeat_slices[0]
        else:
            all_data = np.array(ds[:, :, :].data)
        
        self.cfg.log("Shape of all the data", all_data.shape)
        self.cfg.log("Checks if there are any non nan values", not np.isnan(all_data).all())
 
        relevant_data = all_data[:, wavelength_indicies, :] # gets the data of all wavelengths across all slits for the indicies above the threshold
        self.cfg.log(50 * '-')
        self.cfg.log("Shape of the relevant data", relevant_data.shape)
        # print("relevant Data: ", relevant_data)
        self.cfg.log(50 * '-')
        self.cfg.log("Checks if there are any non nan values", not np.isnan(relevant_data).all())

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
        self.cfg.log("DKIST mean intensity map shape", mean_data.shape)
        return mean_data

    def get_dkist_headers(self):
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
        self.cfg.log(f"Found {len(fits_files)} DKIST FITS files")
        
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

        slits_per_repeat = int(header.get("nx", header.get("NX", header["DNAXIS3"])))
        total_files = len(fits_files)
        repeat_count = total_files // slits_per_repeat

        if repeat_count == 0:
            raise ValueError("No complete raster repeats were found in the DKIST FITS files")

        usable_file_count = repeat_count * slits_per_repeat
        if usable_file_count != total_files:
            self.cfg.log(
                f"Ignoring {total_files - usable_file_count} trailing FITS file(s) that do not form a complete repeat"
            )

        selected_repeat_indices = self._normalize_raster_repeat_selection(repeat_count)
        self.repeat_ranges = []
        selected_fits_files = []

        for repeat_index in selected_repeat_indices:
            repeat_start = repeat_index * slits_per_repeat
            repeat_end = repeat_start + slits_per_repeat
            selected_fits_files.extend(fits_files[repeat_start:repeat_end])
            self.repeat_ranges.append((len(selected_fits_files) - slits_per_repeat, len(selected_fits_files)))

        changing_keywords = {
            "CRVAL1": [],
            "CRVAL3": [],
            "CRPIX1": [],
            "CRPIX3": [],
            "DATE-AVG": []
        }

        for i in range(len(selected_fits_files)):
            header = fits.open(os.path.join(self.cfg.path_to_dkist_data, selected_fits_files[i]))[1].header
            changing_keywords["CRVAL1"].append(header["CRVAL1"])
            changing_keywords["CRVAL3"].append(header["CRVAL3"])
            changing_keywords["CRPIX1"].append(header["CRPIX1"])
            changing_keywords["CRPIX3"].append(header["CRPIX3"])
            changing_keywords["DATE-AVG"].append(header["DATE-AVG"])
        
        self.slits_per_repeat = slits_per_repeat
        self.repeat_count = repeat_count
        self.selected_raster_repeats = selected_repeat_indices

        return fixed_keywords, changing_keywords, selected_fits_files
        

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

        self.cfg.log("Loading DKIST headers")
        self.fixed_keywords, self.changing_keywords, self.fits_files = self.get_dkist_headers()
        self.start_time, self.end_time = self.get_time(self.changing_keywords)
        self.cfg.log(f"DKIST time span: {self.start_time} to {self.end_time}")

        self.cfg.log("Building DKIST intensity map")
        self.intensities = self.get_dkist_wavelengths2()

        if self.intensities.shape[0] != len(self.changing_keywords["DATE-AVG"]):
            raise ValueError(
                "DKIST intensity rows do not match the selected raster repeat headers: "
                f"intensities={self.intensities.shape[0]}, headers={len(self.changing_keywords['DATE-AVG'])}"
            )

        self.cfg.log("Loading HMI reference data")
        self.middle_hmix, self.middle_hmiy, self.middle_hmi_data, self.hmi_files = self.load_hmi(Time(self.start_time), Time(self.end_time))
        self.cfg.log("Data loading complete")

class Alignment:
    """
    Takes an instance of the Data class and interpolates the HMI data to the resolution / pixel grid of DKIST
    """

    def __init__(self, cfg: Config, data_loader: DataLoader):
        self.cfg = cfg
        self.data_loader = data_loader

    def find_nearest_hmi(self, target, hmi_times):
        idx = bisect.bisect_left(hmi_times, target)
        if idx == 0:
            best = 0
        elif idx == len(hmi_times):
            best = idx - 1
        else:
            before, after = hmi_times[idx - 1], hmi_times[idx]
            best = idx - 1 if (target - before) <= (after - target) else idx
        return best
    
    def get_hmi(self, hmi_files, i):

        hmi = map.Map(hmi_files[i])

        coordinates = map.all_coordinates_from_map(hmi)

        hmix = coordinates.Tx
        hmiy = coordinates.Ty
        
        hmi_data = hmi.data

        hmi_data = self.data_loader.normalize(hmi_data)

        return hmix, hmiy, hmi_data
    
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

        nx = len(changing_keywords["CRVAL1"])

        ny = fixed_keywords['DNAXIS1']

        coords = np.zeros((nx, ny, 2))

        i = np.ones(nx)[:, None]

        j = np.arange(ny)[None, :] + 1

        crval1 = np.asarray(changing_keywords["CRVAL1"])[:, None] + crval1_shift
        crval3 = np.asarray(changing_keywords["CRVAL3"])[:, None] + crval3_shift
        crpix1 = np.asarray(changing_keywords["CRPIX1"])[:, None]
        crpix3 = np.asarray(changing_keywords["CRPIX3"])[:, None]

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


    def loss_function(
        self,
        parameters,
        changing_keywords,
        data_numpy,
        interpolator,
        reference_parameters=None
    ):
        """
        Calculates alignment loss between interpolated HMI data and DKIST data.
        Includes an optional regularization term that discourages large
        deviations from a reference parameter vector.
        """

        coords_new = self.construct_dkist_coords(self.data_loader.fixed_keywords, changing_keywords, parameters)

        HMI_interpolated_to_coords = self.interpolate_hmi_to_coords(interpolator, coords_new)

        x = HMI_interpolated_to_coords
        y = data_numpy

        mask = np.isfinite(x) & np.isfinite(y)

        xv = x[mask]
        yv = y[mask]

        if xv.size < 50:
            return 1.0

        x_low, x_high = np.nanpercentile(xv, [1.0, 99.0])
        y_low, y_high = np.nanpercentile(yv, [1.0, 99.0])

        xv = np.clip(xv, x_low, x_high)
        yv = np.clip(yv, y_low, y_high)

        xv = xv - np.nanmedian(xv)
        yv = yv - np.nanmedian(yv)

        denom = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))

        if not np.isfinite(denom) or denom <= 0:
            return 1.0

        corr = np.sum(xv * yv) / denom
        corr = float(np.clip(corr, -1.0, 1.0))

        loss = -corr

        # Regularization term
        if reference_parameters is not None:
            regularization_weight = 0.1
            parameter_penalty = np.sum((np.asarray(parameters) - np.asarray(reference_parameters)) ** 2)
            loss += regularization_weight * parameter_penalty

        return loss

    
    def main(
        self,
        initial_guess,
        bounds,
        return_slit_fitted_parameters=False,
        save_slit_fitted_parameters_path=None,
    ):
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
        self.data_loader.get_all_hmi_times(self.data_loader.hmi_files)

        # middle_image_time = self.data_loader.hmi_times[len(self.data_loader.hmi_times)//2]

        # middle_hmi_idx = self.find_nearest_hmi(middle_image_time, self.data_loader.hmi_times)

        # hmix, hmiy, hmi_data = self.get_hmi(self.data_loader.hmi_files, middle_hmi_idx)

        # best_parameters, result = self.align(initial_guess, bounds, self.data_loader.changing_keywords, self.data_loader.intensities, hmix, hmiy, hmi_data)

        best_parameters = initial_guess
        result = True

        crval_delta = 3
        pc_delta = 0

        bounds = [(best_parameters[0] - crval_delta, best_parameters[0] + crval_delta), (best_parameters[1] - crval_delta, best_parameters[1] + crval_delta), (best_parameters[2] - pc_delta, best_parameters[2] + pc_delta), (best_parameters[3] - pc_delta, best_parameters[3] + pc_delta), (best_parameters[4] - pc_delta, best_parameters[4] + pc_delta), (best_parameters[5] - pc_delta, best_parameters[5] + pc_delta)]
         
        self.cfg.log("Aligning by slits")
        slit_by_slit_result = self.align_slit_by_slit(
            best_parameters,
            bounds,
            return_fitted_parameters=return_slit_fitted_parameters,
            save_fitted_parameters_path=save_slit_fitted_parameters_path,
            crval_delta=crval_delta,
            pc_delta=pc_delta
        )

        if return_slit_fitted_parameters:
            final_coordinates, slit_fitted_parameters = slit_by_slit_result
            self.cfg.log("Final coordinates and slit-by-slit parameters determined")
            return best_parameters, result, final_coordinates, slit_fitted_parameters

        final_coordinates = slit_by_slit_result
        self.cfg.log("Final coordinates determined")
        return best_parameters, result, final_coordinates
    
    def align(
        self,
        initial_guess,
        bounds,
        changing_keywords,
        intensities,
        hmix,
        hmiy,
        hmi_data,
        delta=20,
        interpolator=None,
        reference_parameters=None
    ):

        self.cfg.log(f"Running optimization with initial guess: {np.array(initial_guess)}")

        if interpolator is None:
            initial_coordinates = self.construct_dkist_coords(self.data_loader.fixed_keywords, changing_keywords, initial_guess)

            relevant_hmix, relevant_hmiy, relevant_hmi_data = self.identify_relevant_hmi_data(
                initial_coordinates,
                hmix,
                hmiy,
                hmi_data,
                delta,
            )

            interpolator = self.construct_interpolator(relevant_hmix, relevant_hmiy, relevant_hmi_data)

        result = opt.minimize(
            self.loss_function,
            initial_guess,
            args=(changing_keywords, intensities, interpolator, reference_parameters),
            bounds=bounds,
            method='Powell',
            options={'maxiter': 200, 'disp': self.cfg.verbose},
        )

        best_parameters = result.x

        self.cfg.log(f"Optimization success={result.success}, best parameters={best_parameters}")

        return best_parameters, result.success

    def align_slit_by_slit(
        self,
        initial_guess,
        bounds,
        return_fitted_parameters=False,
        save_fitted_parameters_path=None,
    ):

        ny = self.data_loader.fixed_keywords['DNAXIS1']
        repeat_ranges = getattr(self.data_loader, "repeat_ranges", None)
        if not repeat_ranges:
            repeat_ranges = [(0, len(self.data_loader.changing_keywords["DATE-AVG"]))]

        total_slits = sum(end - start for start, end in repeat_ranges)
        final_coordinates = np.zeros((total_slits, ny, 2))

        fitted_parameters = None
        if return_fitted_parameters or save_fitted_parameters_path is not None:
            fitted_parameters = np.zeros((total_slits, len(initial_guess)))

        for repeat_number, (repeat_start, repeat_end) in enumerate(repeat_ranges):
            repeat_nx = repeat_end - repeat_start
            self.cfg.log(
                f"Aligning raster repeat {repeat_number + 1}/{len(repeat_ranges)} with {repeat_nx} slits"
            )

            current_hmi_image_index = None
            hmix = None
            hmiy = None
            hmi_data = None
            current_interpolator = None

            slit_times = [Time(t) for t in self.data_loader.changing_keywords["DATE-AVG"][repeat_start:repeat_end]]
            hmi_indices = [self.find_nearest_hmi(t, self.data_loader.hmi_times) for t in slit_times]

            self.cfg.log(f"Using {len(np.unique(hmi_indices))} unique HMI frames for repeat {repeat_number + 1}")

            global_best = np.array(initial_guess, dtype=float)
            last_best = global_best.copy()

            # TODO Split in to a helper function and two for loops
            start, mid, end = 0, repeat_nx//2, repeat_nx

            for local_i in chain(range(mid, end), range(mid - 1, start - 1, -1)): 
                if local_i == mid-1:
                    last_best = global_best.copy()

                global_i = repeat_start + local_i
                slit_keywords = {key: [values[global_i]] for key, values in self.data_loader.changing_keywords.items()}
                slit_intensities = self.data_loader.intensities[global_i:global_i + 1, :]

                if (local_i == 0 or local_i == repeat_nx - 1 or local_i % max(1, repeat_nx // 10) == 0):
                    self.cfg.log(f"Processing slit {local_i + 1}/{repeat_nx} in repeat {repeat_number + 1}")

                best_hmi_index = hmi_indices[local_i]

                if best_hmi_index != current_hmi_image_index:
                    current_hmi_image_index = best_hmi_index

                    self.cfg.log(f"Switching to HMI image index {current_hmi_image_index}")

                    hmix, hmiy, hmi_data = self.get_hmi(self.data_loader.hmi_files, current_hmi_image_index)

                    block_end = local_i + 1
                    while block_end < repeat_nx and hmi_indices[block_end] == current_hmi_image_index:
                        block_end += 1

                    block_keywords = {
                        key: values[repeat_start + local_i:repeat_start + block_end]
                        for key, values in self.data_loader.changing_keywords.items()
                    }

                    reference_coordinates = self.construct_dkist_coords(
                        self.data_loader.fixed_keywords,
                        block_keywords,
                        last_best,
                    )

                    relevant_hmix, relevant_hmiy, relevant_hmi_data = self.identify_relevant_hmi_data(
                        reference_coordinates,
                        hmix,
                        hmiy,
                        hmi_data,
                        delta=20,
                    )

                    current_interpolator = self.construct_interpolator(relevant_hmix, relevant_hmiy, relevant_hmi_data)

                best_parameters, result = self.align(
                    last_best,
                    bounds,
                    slit_keywords,
                    slit_intensities,
                    hmix,
                    hmiy,
                    hmi_data,
                    delta=20,
                    interpolator=current_interpolator,
                    reference_parameters=last_best,
                )

                if fitted_parameters is not None:
                    fitted_parameters[local_i] = best_parameters

                coords = self.construct_dkist_coords(self.data_loader.fixed_keywords, slit_keywords, best_parameters)
                final_coordinates[local_i] = coords[0]

                last_best = np.array(best_parameters, dtype=float)
                local_i += 1

        if save_fitted_parameters_path is not None and fitted_parameters is not None:
            np.save(save_fitted_parameters_path, fitted_parameters)
            self.cfg.log(f"Saved slit-by-slit fitted parameters to {save_fitted_parameters_path}")

        if return_fitted_parameters:
            return final_coordinates, fitted_parameters

        return final_coordinates

    def build_synthetic_hmi_on_coords(self, coords, hmi_indices=None, delta=20):
        """
        Builds a synthetic HMI image sampled on a DKIST coordinate grid.
        Each slit uses the nearest-in-time HMI frame, matching the slit-by-slit
        alignment strategy.

        Parameters:
        -----------
        coords (numpy.ndarray): DKIST coordinates with shape (nx, ny, 2).
        hmi_indices (list[int] | None): Optional nearest HMI index for each slit.
        delta (float): Padding (arcsec) when cropping HMI data for interpolation.

        Returns:
        --------
        synthetic_hmi (numpy.ndarray): HMI intensity sampled on coords, shape (nx, ny).
        hmi_indices (list[int]): HMI index used for each slit.
        """
        nx = coords.shape[0]
        synthetic_hmi = np.full((coords.shape[0], coords.shape[1]), np.nan, dtype=float)

        if hmi_indices is None:
            slit_times = [Time(t) for t in self.data_loader.changing_keywords["DATE-AVG"]][:nx]
            hmi_indices = [self.find_nearest_hmi(t, self.data_loader.hmi_times) for t in slit_times]

        start = 0
        while start < nx:
            hmi_index = hmi_indices[start]
            end = start + 1
            while end < nx and hmi_indices[end] == hmi_index:
                end += 1

            hmix, hmiy, hmi_data = self.get_hmi(self.data_loader.hmi_files, hmi_index)
            block_coords = coords[start:end]

            relevant_hmix, relevant_hmiy, relevant_hmi_data = self.identify_relevant_hmi_data(
                block_coords,
                hmix,
                hmiy,
                hmi_data,
                delta=delta,
            )

            interpolator = self.construct_interpolator(relevant_hmix, relevant_hmiy, relevant_hmi_data)
            synthetic_hmi[start:end, :] = self.interpolate_hmi_to_coords(interpolator, block_coords)
            start = end

        return synthetic_hmi, hmi_indices
    
    def _process_block(self, start, end, hmi_index, last_best, bounds):
        hmix, hmiy, hmi_data = self.get_hmi(self.data_loader.hmi_files, hmi_index)

        block_keywords = {
            key: values[start:end] for key, values in self.data_loader.changing_keywords.items()
        }
        block_intensities = self.data_loader.intensities[start:end, :]

        best_parameters, result = self.align(
            last_best, bounds, block_keywords, block_intensities,
            hmix, hmiy, hmi_data, delta=20
        )

        coords = self.construct_dkist_coords(
            self.data_loader.fixed_keywords, block_keywords, best_parameters
        )
 
        return best_parameters, coords


    def align_by_blocks(self, initial_guess, bounds):
        nx = self.data_loader.fixed_keywords['DNAXIS3']
        ny = self.data_loader.fixed_keywords['DNAXIS1']

        final_coordinates = np.zeros((nx, ny, 2))

        current_hmi_image_index = None
        last_best = initial_guess
        start = 0
        best_hmi_index = None

        for i in range(nx):
            slit_time = Time(self.data_loader.changing_keywords["DATE-AVG"][i])
            best_hmi_index = self.find_nearest_hmi(
                slit_time, self.data_loader.hmi_times
            )

            if best_hmi_index != current_hmi_image_index and i != start:
                current_hmi_image_index = best_hmi_index

                best_parameters, coords = self._process_block(
                    start, i, current_hmi_image_index, last_best, bounds
                )
                last_best = best_parameters
                final_coordinates[start:i] = coords
                last_best = best_parameters

                start = i

        best_parameters, coords = self._process_block(
            start, nx, best_hmi_index, last_best, bounds
        )
        final_coordinates[start:nx] = coords

        return final_coordinates

if __name__ == "__main__":

    run = True
    use_synthetic_hmi_viz = True
 
    # path_to_dkist_data = "/Users/joshua/projects/nso/dkist-data/pid_2_31/JPUAIO"
    path_to_dkist_data = "/Users/joshua/projects/nso/dkist-data/pid_3_35/XVNDZY"
    #path_to_dkist_data = "/Users/jamescrowley/Documents/summer_2026/research/pid_4_62/IHFDSO"
    # path_to_dkist_data = "/Users/jamescrowley/Documents/?summer_2026/research/pid_3_35/XVNDZY"
    path_to_sunpy = "~/sunpy/data/"

    #path_to_dkist_data = "C:\\Projects\\DkistData\\pid_3_31\\KRBVTD\\"
    #path_to_sunpy = "C:\\Users\\owner\\sunpy\\data\\"

    output_folder = "saved_plots"
    filename = "my_plot.png"

    os.makedirs(output_folder, exist_ok=True)
    
    save_path = os.path.join(output_folder, filename)
    
    cfg = Config(
    path_to_dkist_data=path_to_dkist_data, 
    path_to_sunpy=path_to_sunpy, 
    wavelength_index=30, 
    raster_repeats=0,
    verbose=True
    )
    cfg.log("Run =", run)

    # Load and prepare  
    cfg.log("LOADING DATA")
    loader = DataLoader(cfg)
    loader.load()
    cfg.log(f"Requested raster repeats: {cfg.raster_repeats}")

    # Minimize
    cfg.log("ALIGNING")
    
    alignment = Alignment(cfg, loader)
    original_dkist_coords = alignment.construct_dkist_coords(loader.fixed_keywords, loader.changing_keywords, [0, 0, 0, 0, 0, 0])


    slit_fitted_parameters = None

    if run:
        initial_guess = [-1.54496818e+00, 1.24362323e+01, -4.59627643e-03, -6.24326444e-03, 2.50216084e-02, -3.68731789e-03]
        # initial_guess = [0,0,0,0,0,0]
        bounds = [(-20, 20), (-20, 20), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
        save_slit_fit_parameters_path = os.path.join(output_folder, "slit_fit_parameters.npy")
        best_parameters, success, final_coordinates, slit_fitted_parameters = alignment.main(
            initial_guess,
            bounds,
            return_slit_fitted_parameters=True,
            save_slit_fitted_parameters_path=save_slit_fit_parameters_path,
        )

        cfg.log('Optimization converged:', success)
        cfg.log('Best parameters from rough alignment:', best_parameters)

    else:
        best_parameters = [-1.54496818e+00, 1.24362323e+01, -4.59627643e-03, -6.24326444e-03, 2.50216084e-02, -3.68731789e-03]


    # assemble final coordinates

    coords_new = alignment.construct_dkist_coords(loader.fixed_keywords, loader.changing_keywords, best_parameters)

    if not run:
        final_coordinates = coords_new

    # Always build a middle-frame HMI background for the left two panels.
    middle_image_time = loader.hmi_times[len(loader.hmi_times)//2]
    best_idx = alignment.find_nearest_hmi(middle_image_time, loader.hmi_times)
    hmix, hmiy, hmi_data = alignment.get_hmi(loader.hmi_files, best_idx)
    relevant_hmix, relevant_hmiy, relevant_hmi_data = alignment.identify_relevant_hmi_data(
        coords_new, hmix, hmiy, hmi_data
    )

    if use_synthetic_hmi_viz:
        synthetic_hmi_on_original_coords, synthetic_hmi_indices_original = alignment.build_synthetic_hmi_on_coords(
            original_dkist_coords,
            delta=20,
        )

        cfg.log(
            f"Built synthetic HMI on original coordinates using {len(np.unique(synthetic_hmi_indices_original))} HMI frames"
        )

        synthetic_hmi_on_final_coords, synthetic_hmi_indices = alignment.build_synthetic_hmi_on_coords(
            final_coordinates,
            delta=20,
        )

        cfg.log(
            f"Built synthetic HMI on final coordinates using {len(np.unique(synthetic_hmi_indices))} HMI frames"
        )
    else:
        interpolator = alignment.construct_interpolator(relevant_hmix, relevant_hmiy, relevant_hmi_data)

        synthetic_hmi_on_original_coords = alignment.interpolate_hmi_to_coords(
            interpolator, original_dkist_coords
        )
        synthetic_hmi_on_final_coords = alignment.interpolate_hmi_to_coords(
            interpolator, final_coordinates
        )

        cfg.log("Using legacy single-frame HMI visualization mode")

    repeat_ranges = getattr(loader, "repeat_ranges", [(0, final_coordinates.shape[0])])
    repeat_count = len(repeat_ranges)
    save_path = Path(save_path)
    use_repeat_synthetic_hmi_viz = use_synthetic_hmi_viz

    for repeat_number, (repeat_start, repeat_end) in enumerate(repeat_ranges, start=1):
        repeat_original_coords = original_dkist_coords[repeat_start:repeat_end]
        repeat_final_coordinates = final_coordinates[repeat_start:repeat_end]
        repeat_intensities = loader.intensities[repeat_start:repeat_end]
        repeat_synthetic_original = synthetic_hmi_on_original_coords[repeat_start:repeat_end]
        repeat_synthetic_final = synthetic_hmi_on_final_coords[repeat_start:repeat_end]
        repeat_slit_params = None
        if slit_fitted_parameters is not None:
            repeat_slit_params = slit_fitted_parameters[repeat_start:repeat_end]

        if repeat_original_coords.shape[0] != repeat_intensities.shape[0] or repeat_final_coordinates.shape[0] != repeat_intensities.shape[0]:
            cfg.log(
                f"Skipping repeat {repeat_number}/{repeat_count} plot because coordinate and intensity shapes do not match: "
                f"original={repeat_original_coords.shape}, final={repeat_final_coordinates.shape}, intensities={repeat_intensities.shape}"
            )
            continue

        fig = plt.figure(figsize=[12, 10])

        plt.subplot(3, 2, 1)
        if use_repeat_synthetic_hmi_viz:
            plt.title(f'Original DKIST data over middle-frame HMI background (repeat {repeat_number})')
        else:
            plt.title(f'Original DKIST data overlayed over original HMI data (repeat {repeat_number})')
        plt.imshow(relevant_hmi_data, extent=[relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap='grey', origin='lower')
        plt.pcolormesh(repeat_original_coords[:, :, 0], repeat_original_coords[:, :, 1], repeat_intensities, cmap='plasma', alpha=0.8, shading='auto')
        plt.colorbar()

        plt.subplot(3, 2, 2)
        if use_repeat_synthetic_hmi_viz:
            plt.title(f'Difference between original DKIST and synthetic nearest-frame HMI (repeat {repeat_number})')
        else:
            plt.title(f'Difference between original DKIST and HMI data (repeat {repeat_number})')
        plt.imshow(relevant_hmi_data, extent=[relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap='grey', origin='lower')
        plt.pcolormesh(repeat_original_coords[:, :, 0], repeat_original_coords[:, :, 1], repeat_intensities - repeat_synthetic_original, cmap='bwr', alpha=1, vmin=-0.5, vmax=0.5, shading='auto')
        plt.colorbar()

        plt.subplot(3, 2, 3)
        if use_repeat_synthetic_hmi_viz:
            plt.title(f'DKIST data after slit-by-slit over synthetic nearest-frame HMI (repeat {repeat_number})')
        else:
            plt.title(f'DKIST data after slit-by-slit (repeat {repeat_number})')
        plt.imshow(relevant_hmi_data, extent=[relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap='grey', origin='lower')
        plt.pcolormesh(repeat_final_coordinates[:, :, 0], repeat_final_coordinates[:, :, 1], repeat_intensities, cmap='plasma', alpha=1, shading='auto')
        plt.colorbar()

        plt.subplot(3, 2, 4)
        if use_repeat_synthetic_hmi_viz:
            plt.title(f'Difference between DKIST after slit-by-slit and synthetic nearest-frame HMI (repeat {repeat_number})')
        else:
            plt.title(f'Difference between DKIST after slit-by-slit and HMI data (repeat {repeat_number})')
        plt.imshow(relevant_hmi_data, extent=[relevant_hmix[0], relevant_hmix[-1], relevant_hmiy[0], relevant_hmiy[-1]], cmap='grey', origin='lower')
        plt.pcolormesh(repeat_final_coordinates[:, :, 0], repeat_final_coordinates[:, :, 1], repeat_intensities - repeat_synthetic_final, cmap='bwr', alpha=1, vmin=-0.5, vmax=0.5, shading='auto')
        plt.colorbar()

        ax_params = plt.subplot(3, 1, 3)
        if repeat_slit_params is not None:
            slit_indices = np.arange(repeat_slit_params.shape[0])
            parameter_names = [
                'CRVAL1 shift',
                'CRVAL3 shift',
                'PC1_1 shift',
                'PC3_1 shift',
                'PC1_3 shift',
                'PC3_3 shift',
            ]
            for p_idx, p_name in enumerate(parameter_names):
                ax_params.plot(slit_indices, repeat_slit_params[:, p_idx], label=p_name)

            ax_params.set_title(f'Evolution of fitted slit-by-slit parameters across raster repeat {repeat_number}')
            ax_params.set_xlabel('Slit index')
            ax_params.set_ylabel('Parameter value')
            ax_params.grid(True, alpha=0.3)
            ax_params.legend(loc='best', fontsize=8, ncol=2)
        else:
            ax_params.set_title(f'Evolution of fitted slit-by-slit parameters across raster repeat {repeat_number}')
            ax_params.text(0.5, 0.5, 'No slit-by-slit fitted parameters available', ha='center', va='center')
            ax_params.set_axis_off()

        plt.tight_layout()

        if repeat_count == 1:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            repeat_save_path = save_path.with_name(f"{save_path.stem}_repeat_{repeat_number}{save_path.suffix}")
            fig.savefig(repeat_save_path, dpi=300, bbox_inches='tight')

    cfg.log("DONE")

    plt.show()