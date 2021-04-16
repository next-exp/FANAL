# General importings
import pandas as pd
import numpy  as np

import matplotlib.pyplot        as plt

# IC importings
from invisible_cities.core.fit_functions import profileX
from invisible_cities.core.fit_functions import profileXY



def build_correction_map(kr_df     : pd.DataFrame,
                         map_type  : str,
                         num_bins  : int = 500,
                         map_fname : str = ''
                        )         -> pd.DataFrame:
    """
    It builds and returns the energy correction map extracted from the kr_df.
    The map type indicates if the correction is based on 'xy' or 'rad'.
    If a map_fname is provided, the correction map is stored in DataFrame format.
    """
    if   (map_type == 'rad'):
        corr_df = build_correction_map_rad(kr_df, num_bins)
    elif (map_type == 'xy'):
        corr_df = build_correction_map_xy(kr_df, num_bins)
    else:
        raise TypeError(f"Correction map type '{map_type}' NOT VALID")

    # Storing the df
    if map_fname != '':
        corr_df.to_hdf(map_fname, 'correction', format = 'table', data_columns = True)

    return corr_df



def build_correction_map_rad(kr_df    : pd.DataFrame,
                             num_bins : int
                            )        -> pd.DataFrame:
    rad, pes, pes_error = profileX(kr_df.rad_rec, kr_df.s2_pes, num_bins)
    corr = pes / pes.min()

    # Plotting corrections
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,7))
    ax1.errorbar(rad, pes, pes_error, fmt="kp", ms=7, lw=3)
    ax1.set_title ("S2 pes (with errors)"   , size=16);
    ax1.set_xlabel("Radius (mm)", size=16);
    ax1.set_ylabel("pes"        , size=16);
    ax2.plot(rad, corr)
    ax2.set_title ("Correction factor"   , size=16);
    ax2.set_xlabel("Radius (mm)", size=16);

    return pd.DataFrame({'rad': rad, 'correction': corr})



def build_correction_map_xy(kr_df    : pd.DataFrame,
                            num_bins : int
                           )        -> pd.DataFrame:
    # XXXXXX TO BE DEVELOPED
    xs, ys, pes, pes_error = profileXY(kr_df.x_rec,
                                       kr_df.y_rec,
                                       kr_df.s2_pes,
                                       num_bins, num_bins)
    corr = pes / pes.min()

    return pd.DataFrame({'x': xs, 'y': ys, 'correction': corr})



def load_correction_map(map_fname : str) -> pd.DataFrame :
    return pd.read_hdf(map_fname, 'correction')



def correct_s2(kr_df    : pd.DataFrame,
               corr_map : pd.DataFrame,
               map_type : str
              ) -> np.array:
    """
    It corrects the krypton s2_pes with the factor from the nearest id
    of the correction map.
    corrected values are stored in kr_df and returned as a numpy array.
    """
    if   (map_type == 'rad'):
        pes_corr = correct_s2_rad(kr_df, corr_map)
    elif (map_type == 'xy'):
        pes_corr = correct_s2_xy(kr_df, corr_map)
    else:
        raise TypeError(f"Correction map type '{map_type}' NOT VALID")

    return pes_corr



def correct_s2_rad(kr_df    : pd.DataFrame,
                   corr_map : pd.DataFrame
                  )        -> np.array:
    pes_corr = []
    for id, krypton in kr_df.iterrows():
        nearest_id = abs(corr_map.rad - krypton.rad_rec).argmin()
        correction = corr_map.iloc[nearest_id].correction
        krypton.s2_pes_corr = krypton.s2_pes / correction
        pes_corr.append(krypton.s2_pes_corr)
    kr_df.s2_pes_corr = pes_corr
    return np.array(pes_corr)



def correct_s2_xy(kr_df    : pd.DataFrame,
                  corr_map : pd.DataFrame
                 )        -> np.array:
    pes_corr = []
    # XXXXXX TO BE DEVELOPED
    return np.array(pes_corr)
