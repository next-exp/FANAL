"""
This SCRIPT runs the pseudo-reconstruction step of fanalIC.
The main parameters are the energy resolution and the spatial definition.
It generates an .h5 file containing 2 dataFrames:
'events' storing all the relevant data of events
'voxels' storing all the voxels info
"""


#---- imports

# General imports
import os
import sys
import math
import numpy  as np
import tables as tb
import pandas as pd
import matplotlib.pyplot as plt

from   matplotlib.colors import LogNorm
from   scipy.stats       import norm

# Specific IC stuff
from invisible_cities.core.system_of_units_c  import units
from invisible_cities.io.mcinfo_io            import load_mcparticles
from invisible_cities.io.mcinfo_io            import load_mchits
from invisible_cities.evm.event_model         import MCHit
from invisible_cities.reco.paolina_functions  import voxelize_hits

# Specific fanalIC stuff
from fanal.core.detector          import get_active_size
from fanal.core.fanal_types       import DetName
from fanal.core.fanal_types       import ActiveVolumeDim
from fanal.core.mc_utilities      import print_mc_event
from fanal.core.mc_utilities      import plot_mc_event
from fanal.reco.reco_io_functions import get_sim_file_names
from fanal.reco.reco_io_functions import get_reco_file_name
from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.reco.reco_io_functions import get_events_reco_dict
from fanal.reco.reco_io_functions import get_voxels_reco_dict
from fanal.reco.reco_io_functions import extend_events_reco_data
from fanal.reco.reco_io_functions import extend_voxels_reco_data
from fanal.reco.reco_io_functions import store_events_reco_data
from fanal.reco.reco_io_functions import store_voxels_reco_data
from fanal.reco.reco_io_functions import store_events_reco_counters
from fanal.reco.energy            import smear_evt_energy
from fanal.reco.energy            import smear_hit_energies
from fanal.reco.position          import get_voxel_size
from fanal.reco.position          import translate_hit_positions
from fanal.reco.position          import check_event_fiduciality



#--- Configuration

# Verbosity level
VERBOSITY_LEVEL = 1

# DETECTOR NAME
DET_NAME = DetName.next100

# EVENT TYPE to be analyzed
EVENT_TYPE = 'Bi214'

# Some needed data
Qbb  = 2457.83 * units.keV
DRIFT_VELOCITY = 1. * units.mm / units.mus

# DETECTOR-ACTIVE dimensions
ACTIVE_dimensions = get_active_size(DET_NAME)



#--- Reconstruction data

# Smearing energy settings
FWHM_Qbb_perc = 0.7 * units.perCent
FWHM_Qbb      = FWHM_Qbb_perc * Qbb
SIGMA_Qbb     = FWHM_Qbb / 2.355
E_MIN = 2.4 * units.MeV
E_MAX = 2.5 * units.MeV
assert E_MAX > E_MIN, 'SmE_filter settings not valid. E_MAX must be higher than E_MIN.'

# Smearing position settings
SPATIAL_DEFINITION = 'Std'
VETO_WIDTH = 20. * units.mm
MIN_VETO_ENERGY = 10 * units.keV

# Voxel size
voxel_size = get_voxel_size(SPATIAL_DEFINITION)

# Fiducial limits
FID_minZ   = ACTIVE_dimensions.z_min + VETO_WIDTH
FID_maxZ   = ACTIVE_dimensions.z_max - VETO_WIDTH
FID_maxRAD = ACTIVE_dimensions.rad  - VETO_WIDTH


print('\n***********************************************************************************')
print('***** Detector: {}'.format(DET_NAME))
print('***** Reconstructing {} events'.format(EVENT_TYPE))
print('***** Energy Resolution: {:.2f}% FWFM at Qbb'.format(FWHM_Qbb_perc / units.perCent))
print('***** Spatial definition: {}'.format(SPATIAL_DEFINITION))
print('***********************************************************************************')

if (VERBOSITY_LEVEL >= 1):
    print('\n* Detector-Active dimensions [mm]:  Zmin: {:4.1f}   Zmax: {:4.1f}   Rmax: {:4.1f}' \
        .format(ACTIVE_dimensions.z_min, ACTIVE_dimensions.z_max, ACTIVE_dimensions.rad))
    print('         ... fiducial limits [mm]:  Zmin: {:4.1f}   Zmax: {:4.1f}   Rmax: {:4.1f}' \
        .format(FID_minZ, FID_maxZ, FID_maxRAD))
    print('\n* Sigma at Qbb: {:.3f} keV.'.format(SIGMA_Qbb / units.keV))
    print('\n* Voxel_size: {} mm.'.format(voxel_size))



#--- Input files

SIM_PATH = '/Users/Javi/Development/fanalIC_NB/data/sim'
#iFileNames = get_sim_file_names(SIM_PATH, EVENT_TYPE)
iFileNames = get_sim_file_names(SIM_PATH, EVENT_TYPE, file_range=[0,2])

if (VERBOSITY_LEVEL >= 1):
	print('\n* {0} {1} input files:'.format(len(iFileNames), EVENT_TYPE))
	for iFileName in iFileNames:
		print(' ', iFileName)



#--- Ouput files and group

# All thr reconstruction data is stored in a single file ...
RECO_PATH = '/Users/Javi/Development/fanalIC_NB/data/reco/'
oFileName = get_reco_file_name(RECO_PATH, EVENT_TYPE)
reco_group_name = get_reco_group_name(FWHM_Qbb_perc/units.perCent, SPATIAL_DEFINITION)

if (VERBOSITY_LEVEL >= 1):
	print('\n* Output file name:', oFileName)
	print('  Reco group name:', reco_group_name)

# Creating the output files and its groups
oFile_filters = tb.Filters(complib='zlib', complevel=4)
oFile = tb.open_file(oFileName, 'w', filters=oFile_filters)
oFile.create_group('/', 'FANALIC')
oFile.create_group('/FANALIC', reco_group_name[9:])

# Storing all the parameters of current reconstruction
# as attributes of the reco_group
oFile.set_node_attr(reco_group_name, 'input_files', iFileNames)
oFile.set_node_attr(reco_group_name, 'event_type', EVENT_TYPE)
oFile.set_node_attr(reco_group_name, 'energy_resolution', FWHM_Qbb_perc/units.perCent)
oFile.set_node_attr(reco_group_name, 'smE_filter_Emin', E_MIN)
oFile.set_node_attr(reco_group_name, 'smE_filter_Emax', E_MAX)
oFile.set_node_attr(reco_group_name, 'fiducial_filter_VetoWidth', VETO_WIDTH)
oFile.set_node_attr(reco_group_name, 'fiducial_filter_MinVetoE', MIN_VETO_ENERGY)



#--- Output data

# Event counters
simulated_events = 0
stored_events = 0

# Dictionaries for events and voxels data
events_dict = get_events_reco_dict()
voxels_dict = get_voxels_reco_dict()



#--- Reconstruction procedure

# Looping through all the input files
for iFileName in iFileNames:
	configuration_df = pd.read_hdf(iFileName, '/MC/configuration', mode='r')

	# Updating simulated and stored event counters
	simulated_events += int(configuration_df[configuration_df.param_key == 'num_events'].param_value)
	stored_events += int(configuration_df[configuration_df.param_key == 'saved_events'].param_value)

	with tb.open_file(iFileName, mode='r') as iFile:
		file_event_numbers = iFile.root.MC.extents.cols.evt_number

		if (VERBOSITY_LEVEL >= 1):
			print ('\n* Processing {0}  ({1} events) ...'.format(iFileName, len(file_event_numbers)))

		# Loading into memory all the hits in the file
		file_mcHits = load_mchits(iFileName)

		# Looping through all the events in the file
		for event_number in file_event_numbers:

			# Verbosing
			if (VERBOSITY_LEVEL >= 2):
				print('\n  Reconstructing event Id: {0} ...'.format(event_number))

			# Getting mcHits of the event, using the event_number as the key
			event_mcHits = file_mcHits[event_number]
			active_mcHits = [hit for hit in event_mcHits if hit.label=='ACTIVE']
			num_hits = len(active_mcHits)

			# The event mc energy is the sum of the energy of all the hits
			event_mcE = sum([hit.E for hit in active_mcHits])

			# Smearing the event energy
			event_smE = smear_evt_energy(event_mcE, SIGMA_Qbb, Qbb)

			# Applying the smE filter
			event_smE_filter = (E_MIN <= event_smE <= E_MAX)
            
			# Verbosing
			if (VERBOSITY_LEVEL >= 2):
				print('    Num mcHits: {0:3}   mcE: {1:.1f} keV   smE: {2:.1f} keV   smE_filter: {3}' \
					.format(num_hits, event_mcE/units.keV,
						event_smE/units.keV, event_smE_filter))

			# For those events NOT passing the smE filter:
			# Storing data of NON smE_filter vents
			if not event_smE_filter:
				extend_events_reco_data(events_dict, event_number, evt_num_MChits=num_hits,
					evt_mcE=event_mcE, evt_smE=event_smE, evt_smE_filter=event_smE_filter)

			# Only for those events passing the smE filter:
			else:
				# Smearing hit energies
				mcE_to_smE_factor = event_smE / event_mcE
				hits_smE = smear_hit_energies(active_mcHits, mcE_to_smE_factor)

				# Translating hit positions
				hits_transPositions = translate_hit_positions(active_mcHits, DRIFT_VELOCITY)

				# Creating the smHits with the smeared energies and translated positions
				active_smHits = []
				for i in range(num_hits):
					smHit = MCHit(hits_transPositions[i], active_mcHits[i].time, hits_smE[i], 'ACTIVE')
					active_smHits.append(smHit)

				# Filtering hits outside the ACTIVE region (due to translation)
				active_smHits = [hit for hit in active_smHits if hit.Z < ACTIVE_dimensions.z_max]

				# Voxelizing using the active_smHits ...
				event_voxels = voxelize_hits(active_smHits, voxel_size, strict_voxel_size=True)
				eff_voxel_size = event_voxels[0].size

				# Storing voxels info
				for voxel in event_voxels:
					extend_voxels_reco_data(voxels_dict, event_number, voxel)

				# Check fiduciality
				voxels_minZ, voxels_maxZ, voxels_maxRad, veto_energy, fiducial_filter = \
					check_event_fiduciality(event_voxels, FID_minZ, FID_maxZ, FID_maxRAD,
											MIN_VETO_ENERGY)
            
                # Storing data of NON smE_filter vents
				extend_events_reco_data(events_dict, event_number, evt_num_MChits=num_hits,
					evt_mcE=event_mcE, evt_smE=event_smE, evt_smE_filter=event_smE_filter,
					evt_num_voxels=len(event_voxels), evt_voxel_sizeX=eff_voxel_size[0],
					evt_voxel_sizeY=eff_voxel_size[1], evt_voxel_sizeZ=eff_voxel_size[2],
					evt_voxels_minZ=voxels_minZ, evt_voxels_maxZ=voxels_maxZ,
					evt_voxels_maxRad=voxels_maxRad, evt_veto_energy=veto_energy,
					evt_fid_filter=fiducial_filter)

				# Verbosing
				if (VERBOSITY_LEVEL >= 2):
					print('    Num voxels: {:3}   minZ: {:.1f} mm   maxZ: {:.1f} mm   maxR: {:.1f} mm   veto energy: {:.1f} keV   fid_filter: {}' \
						.format(len(event_voxels), voxels_minZ, voxels_maxZ,
							voxels_maxRad, veto_energy / units.keV, fiducial_filter))
				if (VERBOSITY_LEVEL >= 3):
					for voxel in event_voxels: print('     ', voxel)



#--- Generating and storing the "events" and "voxels" DataFrame
print('\n* Storing the reconstruction data ... \n')
store_events_reco_data(oFileName, reco_group_name, events_dict)
store_voxels_reco_data(oFileName, reco_group_name, voxels_dict)

# Event counters as attributes
smE_filter_events = sum(events_dict['smE_filter'])
fid_filter_events = sum(events_dict['fid_filter'])
store_events_reco_counters(oFile, reco_group_name, simulated_events, stored_events,
                           smE_filter_events, fid_filter_events)

# Closing the output file
oFile.close()

print('\n* fanalIC reconstruction done!\n')



#--- Printing and plotting results

## Priting the event counters
print('''* Event counters:
Simulated events:  {0:6}
Stored events:     {1:6}
smE_filter events: {2:6}
fid_filter events: {3:6}
''' .format(simulated_events, stored_events, smE_filter_events, fid_filter_events))


events_df = pd.read_hdf(oFileName, reco_group_name + '/events')
voxels_df = pd.read_hdf(oFileName, reco_group_name + '/voxels')

events_df_smE_True = events_df[events_df.smE_filter == True]
events_df_fid_True = events_df[events_df.fid_filter == True]

print("\nEvents ...\n", events_df.head())
print("\nEvents passing the smE_filter ...\n", events_df_smE_True.head())
print("\nEvents passing the fid_filter ...\n", events_df_fid_True.head())
print("\nVoxels ...\n", voxels_df.head())


## Plotting event energies (MC & Smeared) of every event
fig = plt.figure(figsize = (12,5))
num_bins = 50

ax1 = fig.add_subplot(1, 2, 1)
plt.hist(events_df.mcE, num_bins, [2.3, 2.6])
plt.xlabel('Event energy [MeV]', size=12)
plt.ylabel('Num. events', size=12)
plt.title('MonteCarlo Event Energy [MeV]')

ax2 = fig.add_subplot(1, 2, 2)
plt.hist(events_df.smE, num_bins, [2.3, 2.6])
plt.xlabel('Event energy [MeV]', size=12)
plt.ylabel('Num. events', size=12)
plt.title('Smeared Event Energy [MeV]')

plt.show()


## Plotting the number of voxels per event that passed the "Smeared Energy" filter
events_df_smE_True['num_voxels'].value_counts().head(50).sort_index().plot(kind='bar', figsize=(8,6))
plt.show()


## Plotting the voxels energies for all the events
voxels_df.hist('E', figsize =(8,4), bins = 50, range = (0., 0.1), grid = True,
	label = 'Voxel Energy', xlabelsize = 12, orientation = 'vertical', histtype = 'stepfilled')
plt.show()


## Plotting the spatial distribution of voxels from fiducial events
fid_voxels = voxels_df[voxels_df.event_id.isin(events_df_fid_True.index)]

fig = plt.figure(figsize = (18,5))
num_bins = int(ACTIVE_dimensions.z_max/20)

ax1 = fig.add_subplot(1, 3, 1)
plt.hist(fid_voxels.X, num_bins, [-ACTIVE_dimensions.rad, ACTIVE_dimensions.rad])
plt.title('X [mm]', size=14)

ax2 = fig.add_subplot(1, 3, 2)
plt.hist(fid_voxels.Y, num_bins, [-ACTIVE_dimensions.rad, ACTIVE_dimensions.rad])
plt.title('Y [mm]', size=14)

ax3 = fig.adax1 = fig.add_subplot(1, 3, 3)
plt.hist(fid_voxels.Z, num_bins, [ACTIVE_dimensions.z_min, ACTIVE_dimensions.z_max])
plt.title('Z [mm]', size=14)

plt.show()

fig = plt.figure(figsize = (18,5))
num_bins = int(ACTIVE_dimensions.z_max/20)

ax1 = fig.add_subplot(1, 3, 1)
plt.hist2d(fid_voxels.X, fid_voxels.Y, num_bins,
           [[-ACTIVE_dimensions.rad, ACTIVE_dimensions.rad], [-ACTIVE_dimensions.rad, ACTIVE_dimensions.rad]], norm=LogNorm())
plt.xlabel('X [mm]', size=12)
plt.ylabel('Y [mm]', size=12)
plt.title('X-Y [mm]', size=14)

ax2 = fig.add_subplot(1, 3, 2)
plt.hist2d(fid_voxels.X, fid_voxels.Z, num_bins,
           [[-ACTIVE_dimensions.rad, ACTIVE_dimensions.rad], [ACTIVE_dimensions.z_min, ACTIVE_dimensions.z_max]], norm=LogNorm())
plt.xlabel('X [mm]', size=12)
plt.ylabel('Z [mm]', size=12)
plt.title('X-Z [mm]', size=14)

ax3 = fig.adax1 = fig.add_subplot(1, 3, 3)
plt.hist2d(fid_voxels.Y, fid_voxels.Z, num_bins,
           [[-ACTIVE_dimensions.rad, ACTIVE_dimensions.rad], [ACTIVE_dimensions.z_min, ACTIVE_dimensions.z_max]], norm=LogNorm())
plt.xlabel('Y [mm]', size=12)
plt.ylabel('Z [mm]', size=12)
plt.title('Y-Z [mm]', size=14)

plt.show()


## Plotting the histogram of voxel sizes ...
fig = plt.figure(figsize = (18,5))
num_bins = 45

ax1 = fig.add_subplot(1, 3, 1)
plt.hist(events_df_fid_True.voxel_sizeX, num_bins, [0, 15])
plt.title('SizeX [mm]', size=14)

ax1 = fig.add_subplot(1, 3, 2)
plt.hist(events_df_fid_True.voxel_sizeY, num_bins, [0, 15])
plt.title('SizeY [mm]', size=14)

ax1 = fig.add_subplot(1, 3, 3)
plt.hist(events_df_fid_True.voxel_sizeZ, num_bins, [0, 15])
plt.title('SizeZ [mm]', size=14)

plt.show()


## Plotting mcE and smE distributions
fig = plt.figure(figsize = (10,10))
num_bins = 50

ax1 = fig.add_subplot(2, 1, 1)
plt.hist(events_df_smE_True.mcE, num_bins, [2.39,2.51])
plt.xlabel('Total energy (MeV)')
ax1 = fig.add_subplot(2, 1, 2)
plt.hist(events_df_smE_True.smE, num_bins, [2.39,2.51])
plt.xlabel('Total energy (MeV)')

(mu,sigma) = norm.fit(events_df[((events_df.smE>2.445) & (events_df.smE<2.475))].smE)
print('mean: {0:.4f} MeV  sigma: {1:.4f} MeV'.format(mu, sigma))

plt.show()



