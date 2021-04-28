import math
import numpy       as np
import pandas      as pd

from   typing  import Sequence
from   typing  import List
from   typing  import Union
from   typing  import Tuple

from   matplotlib           import colors
import matplotlib.pyplot        as plt

import invisible_cities.core.system_of_units    as units
import invisible_cities.core.fit_functions      as fitf
from   invisible_cities.core.stat_functions import poisson_sigma
from   invisible_cities.core.core_functions import shift_to_bin_centers
from   invisible_cities.io.mcinfo_io        import load_mchits_df
from   invisible_cities.io.mcinfo_io        import load_mcparticles_df

from   fanal.utils.mc_utils                 import get_fname_with_event
from   fanal.utils.types                    import XYZ
from   fanal.core.fanal_units               import Qbb
from   fanal.analysis.mc_analysis           import get_true_extrema



def plot_mc_event(event_id   : int,
                  fnames     : str,
                  event_type : str
                 )          -> None :
    """
    Plots the MC information of the event_id.
    """

    # Getting the right file
    fname = get_fname_with_event(event_id, fnames)
    if fname == '':
        print(f"\nEvent id: {event_id} NOT FOUND in input mc files.")
        return
    else:
        print(f"\nEvent id: {event_id}  contained in {fname}\n")

    # Getting the mcParticles and mcHits of the right event
    mcParts  = load_mcparticles_df(fname).loc[event_id]
    mcHits = load_mchits_df(fname).loc[event_id]

    # Plotting hits
    fig = plt.figure(figsize = (12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"MC event {event_id}")
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    p = ax.scatter(mcHits.x, mcHits.y, mcHits.z, c=(mcHits.energy / units.keV))
    cb = fig.colorbar(p, ax=ax)

    # Plotting True extrema
    ext1, ext2 = get_true_extrema(mcParts, event_type)
    ax.scatter3D(ext1[0], ext1[1], ext1[2], marker="*", lw=2, s=100, color='black')
    ax.scatter3D(ext2[0], ext2[1], ext2[2], marker="*", lw=2, s=100, color='black')

    cb.set_label('Energy (keV)')
    plt.show()
    return



def plot_rec_event(event_id : int,
                   fname    : str
                  ):
    """
    Plots the Reconstructed information of the event_id.
    """

    # Getting voxels and tracks of the right event
    voxels_df = pd.read_hdf(fname, "FANAL" + '/voxels')
    tracks_df = pd.read_hdf(fname, "FANAL" + '/tracks')

    voxels    = voxels_df.loc[event_id]
    tracks    = tracks_df.loc[event_id]

    # Plotting voxels
    fig = plt.figure(figsize = (12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Rec event {event_id}")
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    p = ax.scatter(voxels.x, voxels.y, voxels.z, marker="s",
                   s=(1000, 1000, 1000), c=(voxels.energy / units.keV))
    cb = fig.colorbar(p, ax=ax)

    # Plotting reconstructed blobs
    if len(tracks) == 1:
        track = tracks.loc[0]
        ax.scatter3D(track.blob1_x, track.blob1_y, track.blob1_z,
                     marker="*", lw=2, s=100, color='black')
        ax.scatter3D(track.blob2_x, track.blob2_y, track.blob2_z,
                     marker="*", lw=2, s=100, color='black')
    else:
        print("Non plotting blobs as the event has more than one track")

    cb.set_label('Energy (keV)')
    plt.show()
    return



def plot_vertices(vertices   : Union[Sequence[XYZ], pd.DataFrame],
                  num_bins   : int   = 100,
                  extra_size : float = 10. * units.mm
                 )          -> None :

    ### Getting data
    if isinstance(vertices, Sequence):
        x_positions = list(zip(*vertices))[0]
        y_positions = list(zip(*vertices))[1]
        z_positions = list(zip(*vertices))[2]

    elif isinstance(vertices, pd.DataFrame):
        print("Paso por l128")
        x_positions = vertices.x
        y_positions = vertices.y
        z_positions = vertices.z

    else:
        raise TypeError("Invalid vertices format")

    print(f"x limits: {min(x_positions):6.1f} - {max(x_positions):6.1f}")
    print(f"y limits: {min(y_positions):6.1f} - {max(y_positions):6.1f}")
    print(f"z limits: {min(z_positions):6.1f} - {max(z_positions):6.1f}")

    x_limits = [min(x_positions) - extra_size, max(x_positions) + extra_size]
    y_limits = [min(y_positions) - extra_size, max(y_positions) + extra_size]
    z_limits = [min(z_positions) - extra_size, max(z_positions) + extra_size]


    ### Plotting
    fig = plt.figure(figsize = (9,22))

    # x-y projection
    fig.add_subplot(3, 1, 1)
    plt.hist2d(x_positions, y_positions, num_bins, [x_limits, y_limits], cmap='Reds')
    plt.xlabel('x [mm]',  size=14)
    plt.ylabel('y [mm]',  size=14)
    plt.title('x-y [mm] (lateral view from endcap)', size=14)
    plt.grid(True)
    plt.colorbar()

    # Z-X projection
    fig.add_subplot(3, 1, 2)
    plt.hist2d(z_positions, x_positions, num_bins, [z_limits, x_limits], cmap='Reds')
    plt.xlabel('z [mm]',  size=14)
    plt.ylabel('x [mm]',  size=14)
    plt.title('z-x [mm] (top view)', size=14)
    plt.grid(True)
    plt.colorbar()

    # z-y projection
    fig.add_subplot(3, 1, 3)
    plt.hist2d(z_positions, y_positions, num_bins, [z_limits, y_limits], cmap='Reds')
    plt.xlabel('z [mm]',  size=14)
    plt.ylabel('y [mm]',  size=14)
    plt.title('z-y [mm]  (lateral view from side)', size=14)
    plt.grid(True)
    plt.colorbar()

    plt.tight_layout()
    plt.show()



def plot_photons_spectrum(photons     : pd.DataFrame,
                          title       : str            = '',
                          kin_energy  : bool           = True,
                          wave_length : bool           = True,
                          e_range     : [float, float] = [2., 9.],
                          num_bins    : int            = 100
                         )           -> None :

    wl_range  = [1240 / e_range[1], 1240 / e_range[0]]

    print(f"Spectrum of the {len(photons)} {title} photons:\n")

    if kin_energy & wave_length:
        fig = plt.figure(figsize = (15,5))

        ax1 = fig.add_subplot(1, 2, 1)
        plt.hist(photons.kin_energy / units.eV, num_bins, e_range)
        plt.xlabel('kin_energy [eV]'            , size=14)
        plt.ylabel('Num entries'                , size=14)
        plt.title(f'Spectrum of {title} photons', size=14)

        ax2 = fig.add_subplot(1, 2, 2)
        plt.hist(1240 / (photons.kin_energy / units.eV), num_bins, wl_range)
        plt.xlabel('Wave Length [nm]'           , size=14)
        plt.ylabel('Num entries'                , size=14)
        plt.title(f'Spectrum of {title} photons', size=14)

    elif kin_energy:
        fig = plt.figure(figsize = (10,5))
        plt.hist(photons.kin_energy / units.eV, num_bins, e_range)
        plt.xlabel('kin_energy [eV]'            , size=14)
        plt.ylabel('Num entries'                , size=14)
        plt.title(f'Spectrum of {title} photons', size=14)

    elif wave_length:
        fig = plt.figure(figsize = (10,5))
        plt.hist(1240 / (photons.kin_energy / units.eV), num_bins, wl_range)
        plt.xlabel('Wave Length [nm]'           , size=14)
        plt.ylabel('Num entries'                , size=14)
        plt.title(f'Spectrum of {title} photons', size=14)

    else:
        print("WARNING: All plotting options set to FALSE")




def plot_and_fit(data     : List,
                 title    : str = '',
                 xlabel   : str = 'Charge (pes)',
                 ylabel   : str = 'Entries / bin',
                 num_bins : int = 100 
                )        -> Tuple[float, float, float] :

    # Fitting function
    def gauss(x, amplitude, mu, sigma):
        return amplitude / (2*np.pi)**.5 / sigma * np.exp(-0.5*(x-mu)**2. / sigma**2.)

    # Plotting the data
    fig       = plt.figure(figsize = (8,5))
    mean      = np.mean(data)
    max_range = 0.1
    plt_range = ((1. - max_range) * mean, (1. + max_range) * mean)

    y, x, _ = plt.hist(data, bins = num_bins, range = plt_range)
    x       = shift_to_bin_centers(x)

    # Fitting data
    seed      = 1000, mean, mean * 0.01
    sigma     = poisson_sigma(y)
    max_range = 0.05
    fit_range = ((1. - max_range) * mean, (1. + max_range) * mean)
    f         = fitf.fit(gauss, x, y, seed, fit_range = fit_range, sigma = sigma)

    amplitude, mu, sigma = f.values[0], f.values[1], f.values[2]
    mu_err, sigma_err    = f.errors[1], f.errors[2]
    fwhm                 = 2.35 * sigma / mu

    # Plotting the gauss fit
    mx      = np.linspace(fit_range[0], fit_range[1], 1000)

    legend  = f"$\mu$ = {mu:.3f}\n"
    legend += f"$\sigma$ = {sigma:.3f}\n"
    legend += f"fwhm = {fwhm/units.perCent:.2f} %"

    plt.plot  (mx, f.fn(mx), 'r-')
    plt.plot  (mx, gauss(mx, amplitude, mu, sigma), 'r-', label=legend)
    plt.title (title,  size=14)
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.legend(loc=1);
    plt.show()

    # Verbosing
    print(f"DATA from {title} ...\n" + \
          f"mu    = {mu:10.3f} +- {mu_err:.3f}\n" +\
          f"sigma = {sigma:10.3f} +- {sigma_err:.3f}  ->  " + \
          f"fwhm  = {fwhm/units.perCent:.3f} %\n"
          f"Chi2  = {f.chi2:10.3f}\n")

    return mu, sigma, fwhm



def plot_dict(data  : dict,
              title : str = "",
              ylabel: str = ""
             )     -> None :
    names, values = zip(*data.items())
    plt.bar(range(len(data)), values, align='center')
    plt.xticks(range(len(data)), names, rotation='80')
    plt.ylabel(ylabel)
    plt.ylim([0., max(values) + 5.])
    plt.title(title)
    # Adding values
    for index, value in enumerate(values):
        plt.text(index-0.25, value+2, f"{value:5.1f}")

    plt.show()



def plot_value_counts(photons    : pd.DataFrame,
                      value_name : str,
                      min_perc   : float = 0.,
                      title      : str   = ""
                     )          -> None :

    total = len(photons)
    data  = dict(photons[value_name].value_counts())

    percents = {}
    rest = 0.
    for key in data.keys():
        perc = data[key] * 100 / total
        if perc >= min_perc:
            percents[key] = perc
        else:
            rest += data[key] * 100 / total

    if rest > 0.:
        percents["OTHERS"] = rest

    plot_dict(percents, title=title, ylabel="Percentage")

