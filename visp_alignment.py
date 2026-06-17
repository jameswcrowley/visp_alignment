from pathlib import Path

import numpy as np
import astropy.io.fits as fits
from astropy.time import Time, TimeDelta
import astropy.units as u
import matplotlib.pyplot as plt
from sunpy.net import Fido, attrs as a
import sunpy.map
import dkist.net
import dkist


class Config:
    """
    Specifies all settings, paths, and other metaparameters used in the alignment
    """

    def __init__(self, path_to_dkist_data: str, path_to_sunpy: str, verbose: bool):
        self.path_to_dkist_data = path_to_dkist_data
        self.path_to_sunpy = path_to_sunpy
        self.verbose = verbose


class Data:
    """
    Class to hold the loaded HMI & DKIST data, as well as neccessary metadata
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg


# ------------------------------------------------------------------------------------


class DataLoader:
    """
    Loads the DKIST and HMI data, extracts neccessary info from headers or shapes, saves it to be used later
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_hmi(self, start_time: Time, end_time: Time) -> tuple[list, Time]:
        """
        Downloads all HMI data within one minute of the passed time interval
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
        """
        time_differences = np.abs(hmi_times - target_time)

        closest_index = time_differences.argmin()

        closest_file_path = file_paths[closest_index]
        closest_time = hmi_times[closest_index]
        time_offset = time_differences[closest_index].to(u.second)

        return closest_file_path, closest_index, closest_time, time_offset


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
