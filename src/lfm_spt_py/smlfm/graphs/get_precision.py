from pathlib import Path, PurePath
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


def get_precision(locs3d_data_or_file: Union[Path, npt.NDArray[float]],
                  max_lateral_err: Optional[float] = None,
                  min_view_count: Optional[int] = None
                        ):
    
    if isinstance(locs3d_data_or_file, np.ndarray):
        locs_3d = locs3d_data_or_file
    elif isinstance(locs3d_data_or_file, PurePath):
        locs_3d = np.genfromtxt(locs3d_data_or_file, delimiter=',', dtype=float)
    else:
        raise TypeError('Unsupported argument type with 3D localisations')

    xyz = locs_3d[:, 0:3]  # X, Y, Z
    lateral_err = locs_3d[:, 3]  # Fitting error in X and Y (in microns)
    axial_err = locs_3d[:, 4]  # Fitting error in Z (in microns)
    view_count = locs_3d[:, 5]  # Number of views used to fit the localisation
    photons = locs_3d[:, 6]  # Number of photons in fit

    
    keep = np.logical_and(
        (lateral_err < max_lateral_err) if max_lateral_err is not None else True,
        (view_count > min_view_count) if min_view_count is not None else True)
    
    lateral_err_mean = np.mean(lateral_err[keep])
    axial_err_mean = np.mean(axial_err)
    
    return lateral_err_mean, axial_err_mean
    
