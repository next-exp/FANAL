import os
import sys
import random
import tables as tb
import numpy  as np
import pandas as pd

from invisible_cities.core.system_of_units_c import units

from invisible_cities.io.mcinfo_io import load_mchits
from invisible_cities.io.mcinfo_io import load_mcparticles
from invisible_cities.io.mcinfo_io import load_mcsensor_response


def particle_info(particle_dict, evt_ini_time=0., with_hits=False):
    part_tab = ' ' * 2
    for indx, part in particle_dict.items():
        # General Info
        #print(part_tab + 'Particle {0}: name = {1},   primary = {2},   mass (MeV) = ,   charge = '
        print(part_tab + 'Particle {0}: name = {1},   primary = {2}'
              .format(indx, part.name, part.primary))

        # Creator Info
        if not part.primary:
            mother = particle_dict[part.mother_indx]
            print(part_tab + 'Prod.: Process = {0},   Mother index = {1} ({2})'
                  .format(part.process, part.mother_indx, mother.name))
            
        # Production Info
        print(part_tab + 'Prod.: Mom = ({0:.3f}, {1:.3f}, {2:.3f}) MeV,   KinE: {3:.3f} MeV'
              .format(part.p[0], part.p[1], part.p[2], part.E))
        print(part_tab + 'Prod.: Volume = {0},   Vertex = ({1:.3f}, {2:.3f}, {3:.3f}) mm'
              .format(part.initial_volume, part.initial_vertex[0], part.initial_vertex[1], part.initial_vertex[2]))
        
        # Decay Info
        print(part_tab + 'Decay: Volume = {0},   Vertex = ({1:.3f}, {2:.3f}, {3:.3f}) mm'
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
                print(part_tab*2, 'Detector: {0},   E: {1:7.3f} KeV   Pos: ({2:.0f}, {3:.0f}, {4:.0f}) mm,   Time: {5:.3e} us,   Evt. Time: {6:.3e} us'
                      .format(hit.label, hit.E/units.keV, hit.X, hit.Y, hit.Z, hit.T/units.mus, (hit.T - evt_ini_time)/units.mus))
        
        print()


# DATA_PATH = '/Users/Javi/Development/fanalIC/data'
# bb0nu_fileName = os.path.join(DATA_PATH, 'bb0nu/bb0nu-000-.next.h5')
# Bi214_fileName = os.path.join(DATA_PATH, 'Bi214/Bi214-000-.next.h5')
# Tl208_fileName = os.path.join(DATA_PATH, 'Tl208/Tl208-000-.next.h5')
# iFileName = Bi214_fileName

# EVT_SEARCHED = 11

# with tb.open_file(iFileName, mode='r') as h5in:
    
#     h5extents = h5in.root.MC.extents
#     events_in_file = len(h5extents)
    
#     # Looking for event
#     evt_number = EVT_SEARCHED
#     for i in range(events_in_file):
#         if h5extents[i]['evt_number'] == evt_number:
#             evt_line = i

#     # Getting the mcParticles and mcHits
#     # They are a dictionary with key = evt_num of
#     # dictionaries with keys  = particle / hit index
#     evt_mcParticles = load_mcparticles(iFileName, (evt_line, evt_line+1))
#     evt_mcParticles = evt_mcParticles[evt_number]
#     evt_mcHits = load_mchits(iFileName, (evt_line, evt_line+1))
#     evt_mcHits = evt_mcHits[evt_number]
    
#     print('')
#     print('* Event Number = {}'.format(evt_number))
#     tot_dep_energy = sum([h.E for h in evt_mcHits])
#     print('  Event deposited energy = {0:.6f} MeV'.format(tot_dep_energy))
#     ini_time  = min([h.time for h in evt_mcHits])
#     last_time = max([h.time for h in evt_mcHits])
#     print('  Event initial time = {0:.3e} us,   Time width: {1:.3e} us'
#           .format(ini_time/units.mus, (ini_time-last_time)/units.mus))
#     #print(evt_mcHits)
#     print('  Event has {} MC Particles'.format(len(evt_mcParticles)))
#     print('  Event has {} MC Hits'.format(len(evt_mcHits)))
#     print('')
#     print('- List of MC Particles:')
#     particle_info(evt_mcParticles, with_hits=True, evt_ini_time = ini_time)


def print_event(event_id, iFileNames, with_hits=False):
	''' Prints the information of the event corresponding to event_id.
	It will look for it into all the list of iFileNames passed.'''

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
				#particle_info(evt_mcParticles, with_hits=True, evt_ini_time = ini_time)

				return



	# Event Id not found in any input file
	print('\nEvt Id: {}  NOT FOUND in any input file.\n'.format(event_id))



if __name__ == "__main__":

	EVT_SEARCHED = 75000247
	IFILE_NAMES  = ['/Users/Javi/Development/fanalIC/data/sim/Bi214/Bi214-000-.next.h5',
					'/Users/Javi/Development/fanalIC/data/sim/Bi214/Bi214-030-.next.h5']

	print_event (EVT_SEARCHED, IFILE_NAMES)


