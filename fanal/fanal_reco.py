"""
This SCRIPT runs the pseudo-reconstruction step of fanalIC.
The main parameters are the energy resolution and the spatial definition.
It generates an .h5 file containing 2 dataFrames:
'events' storing all the relevant data of events
'voxels' storing all the voxels info
"""

# General importings
import os
import sys
import math
import numpy  as np
import tables as tb
import pandas as pd

# Specific IC stuff
import invisible_cities.core.system_of_units as units
from invisible_cities.cities.components      import city
from invisible_cities.core.configure         import configure
from invisible_cities.evm.event_model        import MCHit
from invisible_cities.reco.paolina_functions import voxelize_hits
from invisible_cities.reco.tbl_functions     import filters as tbl_filters

# Specific fanalIC stuff
from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.reco.reco_io_functions import get_events_reco_dict
from fanal.reco.reco_io_functions import get_voxels_reco_dict
from fanal.reco.reco_io_functions import extend_events_reco_data
from fanal.reco.reco_io_functions import extend_voxels_reco_data
from fanal.reco.reco_io_functions import store_events_reco_data
from fanal.reco.reco_io_functions import store_voxels_reco_data
from fanal.reco.reco_io_functions import store_events_reco_counters

from fanal.reco.energy        import smear_evt_energy
from fanal.reco.energy        import smear_hit_energies

from fanal.reco.position      import get_voxel_size
from fanal.reco.position      import translate_hit_positions
from fanal.reco.position      import check_event_fiduciality

from fanal.core.logger        import get_logger
from fanal.core.detector      import get_active_size
from fanal.core.detector      import get_fiducial_size
from fanal.core.fanal_types   import DetName
from fanal.core.fanal_types   import SpatialDef

from fanal.mc.mc_io_functions import load_mc_particles
from fanal.mc.mc_io_functions import load_mc_hits
from fanal.mc.mc_io_functions import get_num_mc_particles

from fanal.mc.mc_utilities    import print_mc_event
from fanal.mc.mc_utilities    import plot_mc_event



### GENERAL DATA NEEDED
Qbb  = 2457.83 * units.keV
DRIFT_VELOCITY = 1. * units.mm / units.mus


@city
def fanal_reco(det_name,    # Detector name: 'new', 'next100', 'next500'
               event_type,  # Event type: 'bb0nu', 'Tl208', 'Bi214'
               fwhm,        # FWHM at Qbb
               e_min,       # Minimum smeared energy for energy filtering
               e_max,       # Maximum smeared energy for energy filtering
               spatial_def, # Spatial definition: 'low', 'std', high'
               voxel_Eth,   # Voxel energy threshold
               veto_width,  # Veto width for fiducial filtering
               min_veto_e,  # Minimum energy in veto for fiducial filtering
               files_in,    # Input files
               event_range, # Range of events to analyze: all, ... ??
               file_out,    # Output file
               compression, # Compression of output file: 'ZLIB1', 'ZLIB4',
                            # 'ZLIB5', 'ZLIB9', 'BLOSC5', 'BLZ4HC5'
               verbosity_level):


    ### LOGGER
    logger = get_logger('FanalReco', verbosity_level)


    ### DETECTOR NAME & its ACTIVE dimensions
    det_name = getattr(DetName, det_name)
    ACTIVE_dimensions = get_active_size(det_name)


    ### RECONSTRUCTION DATA
    # Smearing energy settings
    fwhm_Qbb  = fwhm * Qbb
    sigma_Qbb = fwhm_Qbb / 2.355
    assert e_max > e_min, 'SmE_filter settings not valid. e_max must be higher than e_min.'

    # Spatial definition
    spatial_def = getattr(SpatialDef, spatial_def)

    # Voxel size
    voxel_size = get_voxel_size(spatial_def)

    # Fiducial limits
    fid_dimensions = get_fiducial_size(ACTIVE_dimensions, veto_width)


    ### PRINTING GENERAL INFO
    print('\n***********************************************************************************')
    print('***** Detector: {}'.format(det_name.name))
    print('***** Reconstructing {} events'.format(event_type))
    print('***** Energy Resolution: {:.2f}% FWFM at Qbb'.format(fwhm / units.perCent))
    print('***** Spatial definition: {}'.format(spatial_def.name))
    print('***********************************************************************************\n')

    print('* Detector-Active dimensions [mm]:  Zmin: {:7.1f}   Zmax: {:7.1f}   Rmax: {:7.1f}'
          .format(ACTIVE_dimensions.z_min, ACTIVE_dimensions.z_max,
                  ACTIVE_dimensions.rad))
    print('         ... fiducial limits [mm]:  Zmin: {:7.1f}   Zmax: {:7.1f}   Rmax: {:7.1f}\n'
          .format(fid_dimensions.z_min, fid_dimensions.z_max, fid_dimensions.rad))
    print('* Sigma at Qbb: {:.3f} keV.\n'.format(sigma_Qbb / units.keV))
    print('* Voxel_size: {} mm.'.format(voxel_size))
    print('  Voxel Eth:  {:4.1f} keV\n'.format(voxel_Eth/units.keV))

    print('* {0} {1} input files:'.format(len(files_in), event_type))
    for iFileName in files_in:
        print(' ', iFileName)


    ### OUTPUT FILE, ITS GROUPS & ATTRIBUTES
    # Output file
    oFile = tb.open_file(file_out, 'w', filters = tbl_filters(compression))

    # Reco group Name
    reco_group_name = get_reco_group_name(fwhm/units.perCent, spatial_def)
    oFile.create_group('/', 'FANALIC')
    oFile.create_group('/FANALIC', reco_group_name[9:])

    print('\n* Output file name:', file_out)
    print('  Reco group name:  {}\n'.format(reco_group_name))

    # Attributes
    oFile.set_node_attr(reco_group_name, 'input_sim_files', files_in)
    oFile.set_node_attr(reco_group_name, 'event_type', event_type)
    oFile.set_node_attr(reco_group_name, 'energy_resolution', fwhm/units.perCent)
    oFile.set_node_attr(reco_group_name, 'voxel_Eth', voxel_Eth)
    oFile.set_node_attr(reco_group_name, 'smE_filter_Emin', e_min)
    oFile.set_node_attr(reco_group_name, 'smE_filter_Emax', e_max)
    oFile.set_node_attr(reco_group_name, 'fiducial_filter_VetoWidth', veto_width)
    oFile.set_node_attr(reco_group_name, 'fiducial_filter_MinVetoE', min_veto_e)


    ### DATA TO STORE
    # Event counters
    simulated_events = 0
    stored_events    = 0
    analyzed_events  = 0
    toUpdate_events  = 1

    # Dictionaries for events & voxels data
    events_dict = get_events_reco_dict()
    voxels_dict = get_voxels_reco_dict()


    ### RECONSTRUCTION PROCEDURE
    # Looping through all the input files
    for iFileName in files_in:
        # Updating simulated and stored event counters
        configuration_df  = pd.read_hdf(iFileName, '/MC/configuration', mode='r')
        simulated_events += int(configuration_df[configuration_df.param_key == 'num_events'].param_value)
        stored_events    += int(configuration_df[configuration_df.param_key == 'saved_events'].param_value)

        # Getting event numbers
        file_extents = pd.read_hdf(iFileName, '/MC/extents', mode='r')
        file_event_numbers = file_extents.evt_number
        #file_event_numbers = file_mcHits.index.get_level_values(0).unique()

        print('* Processing {0}  ({1} events) ...'.format(iFileName, len(file_event_numbers)))

        # Getting mc hits
        file_mcHits  = load_mc_hits(iFileName)


        # Looping through all the events in iFile
        for event_number in file_event_numbers:
            # Updating counter of analyzed events
            analyzed_events += 1
            if (not(analyzed_events % toUpdate_events)):
                print('* Num analyzed events: {}'.format(analyzed_events))
            if (analyzed_events == (10 * toUpdate_events)): toUpdate_events *= 10

            # Verbosing
            logger.info('Reconstructing event Id: {0} ...'.format(event_number))

            # Getting event info
            event_mcHits  = file_mcHits.loc[event_number, :]
            active_mcHits = event_mcHits[event_mcHits.label == 'ACTIVE'].copy()

            evt_num_parts = get_num_mc_particles(file_extents, event_number)
            evt_num_hits  = len(active_mcHits)

            # The event mc energy is the sum of the energy of all the hits
            event_mcE = active_mcHits.E.sum()
                
            # Smearing the event energy
            event_smE = smear_evt_energy(event_mcE, sigma_Qbb, Qbb)

            # Applying the smE filter
            event_smE_filter = (e_min <= event_smE <= e_max)

            # Verbosing
            logger.info('  Num mcHits: {0:3}   mcE: {1:.1f} keV   smE: {2:.1f} keV   smE_filter: {3}' \
                        .format(evt_num_hits, event_mcE/units.keV, event_smE/units.keV, event_smE_filter))
                
            # For those events NOT passing the smE filter:
            # Storing data of NON smE_filter events
            if not event_smE_filter:
                extend_events_reco_data(events_dict, event_number,
                                        evt_num_MCparts = evt_num_parts,
                                        evt_num_MChits  = evt_num_hits,
                                        evt_mcE         = event_mcE,
                                        evt_smE         = event_smE,
                                        evt_smE_filter  = event_smE_filter)

            # Only for those events passing the smE filter:
            else:
                # Smearing hit energies
                smearing_factor = event_smE / event_mcE
                active_mcHits['smE'] = active_mcHits['E'] * smearing_factor

                # Translating hit Z positions from delayed hits
                translate_hit_positions(active_mcHits, DRIFT_VELOCITY)
 
                # Creating the IChits with the smeared energies and translated Z positions
                # to be passed to paolina functions
                #IChits = []
                #for i, hit in active_mcHits[active_mcHits.shifted_z < ACTIVE_dimensions.z_max].iterrows():
                #    IChit = MCHit((hit.x, hit.y, hit.shifted_z), hit.time, hit.smE, 'ACTIVE')
                #    IChits.append(IChit)
                IChits = active_mcHits[active_mcHits.shifted_z < ACTIVE_dimensions.z_max] \
                    .apply(lambda hit: MCHit((hit.x, hit.y, hit.shifted_z),
                                             hit.time, hit.smE, 'ACTIVE'), axis=1).tolist()

                # Voxelizing using the IChits ...
                event_voxels = voxelize_hits(IChits, voxel_size, strict_voxel_size=False)
                eff_voxel_size = event_voxels[0].size

                # Storing voxels info
                for voxel_id in range(len(event_voxels)):
                    extend_voxels_reco_data(voxels_dict, event_number,
                                            voxel_id, event_voxels[voxel_id], voxel_Eth)

                # Check fiduciality
                voxels_minZ, voxels_maxZ, voxels_maxRad, veto_energy, fiducial_filter = \
                check_event_fiduciality(event_voxels, fid_dimensions, min_veto_e)

                # Storing data of events passing the smE_filter events
                extend_events_reco_data(events_dict, event_number,
                                        evt_num_MCparts   = evt_num_parts,
                                        evt_num_MChits    = evt_num_hits,
                                        evt_mcE           = event_mcE,
                                        evt_smE           = event_smE,
                                        evt_smE_filter    = event_smE_filter,
                                        evt_num_voxels    = len(event_voxels),
                                        evt_voxel_sizeX   = eff_voxel_size[0],
                                        evt_voxel_sizeY   = eff_voxel_size[1],
                                        evt_voxel_sizeZ   = eff_voxel_size[2],
                                        evt_voxels_minZ   = voxels_minZ,
                                        evt_voxels_maxZ   = voxels_maxZ,
                                        evt_voxels_maxRad = voxels_maxRad,
                                        evt_veto_energy   = veto_energy,
                                        evt_fid_filter    = fiducial_filter)

                # Verbosing
                logger.info('  NumVoxels: {:3}   minZ: {:.1f} mm   maxZ: {:.1f} mm   maxR: {:.1f} mm   veto_E: {:.1f} keV   fid_filter: {}' \
                            .format(len(event_voxels), voxels_minZ, voxels_maxZ, voxels_maxRad,
                                    veto_energy / units.keV, fiducial_filter))
                for voxel in event_voxels:
                    logger.debug('    Voxel pos: ({:5.1f}, {:5.1f}, {:5.1f}) mm   E: {:5.1f} keV'\
                                 .format(voxel.X/units.mm, voxel.Y/units.mm, voxel.Z/units.mm, voxel.E/units.keV))


    # Verbosing
    print('\n* Total analyzed events: {}'.format(analyzed_events))

    ### STORING DATA
    # Storing events and voxels dataframes
    print('\n* Storing data in the output file: {}'.format(file_out))
    store_events_reco_data(file_out, reco_group_name, events_dict)
    store_voxels_reco_data(file_out, reco_group_name, voxels_dict)

    # Storing event counters as attributes
    smE_filter_events = sum(events_dict['smE_filter'])
    fid_filter_events = sum(events_dict['fid_filter'])
    store_events_reco_counters(oFile, reco_group_name, simulated_events,
                               stored_events, smE_filter_events, fid_filter_events)

    oFile.close()
    print('* Reconstruction done !!\n')

    # Printing reconstruction numbers
    print('* Event counters ...')
    print('''  Simulated events:  {0:9}
  Stored events:     {1:9}
  smE_filter events: {2:9}
  fid_filter events: {3:9}\n'''
        .format(simulated_events, stored_events, smE_filter_events, fid_filter_events))



if __name__ == '__main__':
	result = fanal_reco(**configure(sys.argv))
