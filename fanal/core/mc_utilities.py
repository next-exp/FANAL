import os
import sys
import random
import tables as tb
import numpy  as np
import pandas as pd

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Sequence, Union, Dict, Any

from invisible_cities.core.system_of_units_c import units
from invisible_cities.io.mcinfo_io           import load_mchits
from invisible_cities.io.mcinfo_io           import load_mcparticles
from invisible_cities.io.mcinfo_io           import load_mcsensor_response
from invisible_cities.evm.event_model        import MCHit



def particle_mc_info(particle_dict: Dict[int, Sequence[MCHit]],
	                   evt_ini_time:  float = 0.,
	                   with_hits:     bool  = False
	                  ) -> None:
	part_tab = ' ' * 2
	for indx, part in particle_dict.items():
		# General Info
		print(part_tab + 'Particle {0}: name = {1},   primary = {2}'
			.format(indx, part.name, part.primary))

		# Creator Info
		if not part.primary:
			mother = particle_dict[part.mother_indx]
			print(part_tab + 'Prod.: Process = {0},   Mother index = {1} ({2})'
				.format(part.process, part.mother_indx, mother.name))
            
		# Production Info
		print(part_tab + 'Prod.: Mom = ({0:.1f}, {1:.1f}, {2:.1f}) keV,   KinE: {3:.1f} keV'
			.format(part.p[0]/units.keV, part.p[1]/units.keV, part.p[2]/units.keV, part.E/units.keV))
		print(part_tab + 'Prod.: Volume = {0},   Vertex = ({1:.1f}, {2:.1f}, {3:.1f}) mm'
			.format(part.initial_volume, part.initial_vertex[0],
				      part.initial_vertex[1], part.initial_vertex[2]))
        
		# Decay Info
		print(part_tab + 'Decay: Volume = {0},   Vertex = ({1:.1f}, {2:.1f}, {3:.1f}) mm'
			.format(part.final_volume, part.final_vertex[0], part.final_vertex[1], part.final_vertex[2]))

		# Daughter Particles Info
		print(part_tab + 'Daughter particles:')
		for daugh_indx, daugh_part in particle_dict.items():
			if daugh_part.mother_indx == indx:
				print(part_tab*2, 'Idx: {0}   Name: {1:10},   Volume: {2}'
					.format(daugh_indx, daugh_part.name, daugh_part.initial_volume))

		# Hits Info
		if with_hits:
			print(part_tab +'{0} MC Hits:'.format(len(part.hits)))
			for hit in part.hits:
				print(part_tab*2, 'Detector: {0},   E: {1:5.1f} KeV   Pos: ({2:5.0f}, {3:5.0f}, {4:5.0f}) mm,   Time: {5:.1e} us,   Evt. Time: {6:.1e} us'
					.format(hit.label, hit.E/units.keV, hit.X, hit.Y, hit.Z, hit.T/units.mus,
						      (hit.T - evt_ini_time)/units.mus))
        
		print()



def print_mc_event(event_id:   int,
	                 iFileNames: Union[str, Sequence[str]],
	                 with_hits:  bool = False
	                ) -> None:
	'''
	Prints the information of the event corresponding to event_id.
	It will look for it into all the list of iFileNames passed.
	'''

	# In case of just the name of one input file ...
	if type(iFileNames) == str:
		iFileNames = [iFileNames]

	# Going through all the input files
	for iFileName in iFileNames:
		with tb.open_file(iFileName, mode='r') as h5in:
			extents_df = pd.read_hdf(iFileName, '/MC/extents', mode='r')

			if event_id in extents_df['evt_number'].tolist():
				print('\nEvt Id: {}  contained in {}\n'.format(event_id, iFileName))

				event_index = extents_df[extents_df.evt_number == event_id].index[0]

				# Getting the mcParticles and mcHits
				# They are a dictionary with key = evt_num of
				# dictionaries with keys  = particle / hit index
				evt_mcParticles = load_mcparticles(iFileName, (event_index, event_index+1))
				evt_mcParticles = evt_mcParticles[event_id]
				evt_mcHits = load_mchits(iFileName, (event_index, event_index+1))
				evt_mcHits = evt_mcHits[event_id]

				tot_dep_energy = sum([h.E for h in evt_mcHits])
				print('  Event deposited energy = {0:.6f} MeV'.format(tot_dep_energy))

				ini_time  = min([h.time for h in evt_mcHits])
				last_time = max([h.time for h in evt_mcHits])
				print('  Event initial time = {0:.3e} us,   Time width: {1:.3e} us'
					.format(ini_time/units.mus, (ini_time-last_time)/units.mus))

				print('  Event has {} MC Particles'.format(len(evt_mcParticles)))
				print('  Event has {} MC Hits'.format(len(evt_mcHits)))
				print('')
				print('- List of MC Particles:')
				particle_mc_info(evt_mcParticles, with_hits=with_hits, evt_ini_time = ini_time)

				return

	# Event Id not found in any input file
	print('\nEvt Id: {}  NOT FOUND in any input file.\n'.format(event_id))



def plot_mc_event(event_id, iFileNames):
	''' Prints the information of the event corresponding to event_id.
	It will look for it into all the list of iFileNames passed.'''

	# In case of just the name of one input file ...
	if type(iFileNames) == str:
		iFileNames = [iFileNames]

	# Going through all the input files
	for iFileName in iFileNames:
		with tb.open_file(iFileName, mode='r') as h5in:
			extents_df = pd.read_hdf(iFileName, '/MC/extents', mode='r')

			if event_id in extents_df['evt_number'].tolist():
				print('\nEvt Id: {}  contained in {}\n'.format(event_id, iFileName))

				event_index = extents_df[extents_df.evt_number == event_id].index[0]

				evt_mcHits = load_mchits(iFileName, (event_index, event_index+1))
				evt_mcHits = evt_mcHits[event_id]

				hits_X = [h.X for h in evt_mcHits]
				hits_Y = [h.Y for h in evt_mcHits]
				hits_Z = [h.Z for h in evt_mcHits]
				hits_E = [h.E/units.keV for h in evt_mcHits]

				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.set_xlabel('X (mm)')
				ax.set_ylabel('Y (mm)')
				ax.set_zlabel('Z (mm)')
				p = ax.scatter(hits_X, hits_Y, hits_Z, cmap='coolwarm', c=hits_E)
				cb = fig.colorbar(p, ax=ax)
				cb.set_label('Energy (keV)')
				plt.show()

				return

	# Event Id not found in any input file
	print('\nEvt Id: {}  NOT FOUND in any input file.\n'.format(event_id))
