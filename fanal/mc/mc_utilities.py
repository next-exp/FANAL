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

from typing import Sequence, Union, Dict, Any, List, Mapping

import invisible_cities.core.system_of_units as units
from invisible_cities.io.mcinfo_io           import load_mchits
from invisible_cities.io.mcinfo_io           import load_mcparticles
from invisible_cities.io.mcinfo_io           import load_mcsensor_response
from invisible_cities.evm.event_model        import MCParticle

from fanal.mc.mc_io_functions  import load_mc_particles
from fanal.mc.mc_io_functions  import load_mc_hits



def print_mc_particles(mcParticles:  pd.DataFrame,
                       mcHits:       pd.DataFrame,
                       evt_ini_time: float = 0.,
                       with_hits:    bool = False) -> None:
    
    part_tab = ' ' * 2
    
    for part_id in mcParticles.index.get_level_values(0):
        part = mcParticles.loc[part_id, :]
        
        # General Info
        print()
        print(part_tab + 'Particle {0}: name = {1},   primary = {2}'
              .format(part_id, part['name'], part.primary))
        
        # Creator Info
        if not part.primary:
            mother_part = mcParticles.loc[part.mother_indx, :]
            print(part_tab + 'Prod.: Process = {0},   Mother index = {1} ({2})'
                  .format(part.creator_proc, part.mother_indx, mother_part['name']))

        # Production Info
        print(part_tab + 'Prod.: Mom = ({0:.1f}, {1:.1f}, {2:.1f}) keV,   KinE: {3:.1f} keV'
             .format(part.ini_px/units.keV, part.ini_py/units.keV,
                     part.ini_pz/units.keV, part.kin_energy/units.keV))
        print(part_tab + 'Prod.: Volume = {0},   Vertex = ({1:.1f}, {2:.1f}, {3:.1f}) mm'
              .format(part.initial_volume, part.ini_x, part.ini_y, part.ini_z))
        
        # Decay Info
        print(part_tab + 'Decay: Volume = {0},   Vertex = ({1:.1f}, {2:.1f}, {3:.1f}) mm'
              .format(part.final_volume, part.final_x, part.final_y, part.final_z))

        # Daughter Particles Info
        daughter_parts = mcParticles[mcParticles.mother_indx == part_id]
        print(part_tab + '{} daughter particles:'.format(len(daughter_parts)))
        for daugh_id in daughter_parts.index.get_level_values(0):
            daugh_part = daughter_parts.loc[daugh_id, :]
            print(part_tab*2, 'Part {0}   Name: {1:10},   Volume: {2}'.format(daugh_id,
                                                                              daugh_part['name'],
                                                                              daugh_part.initial_volume))
        
        # Printing hits
        if with_hits:
            if (part_id in mcHits.index.get_level_values(0)):
                part_hits = mcHits.loc[part_id, :]
                print(part_tab +'{} MC Hits:'.format(len(part_hits)))
                for hit_id in part_hits.index.get_level_values(0):
                    hit = part_hits.loc[hit_id, :]
                    print(part_tab*2, 'Hit {:2}   Det: {},   E: {:5.1f} KeV   ({:5.0f}, {:5.0f}, {:5.0f}) mm,   t: {:.1e} us,   Evt. t: {:.1e} us'
                          .format(hit_id, hit.label, hit.E/units.keV, hit.x, hit.y, hit.z,
                                  hit.time/units.mus, (hit.time - evt_ini_time)/units.mus))
            else:
                print(part_tab +'0 MC Hits')




def print_mc_event(event_id:   int,
                   iFileNames: List[str],
                   with_hits:  bool = False
                  ) -> None:
    """Prints the information of the event corresponding to event_id.
It will look for it into all the list of iFileNames passed."""
    
    # Going through all the input files
    for iFileName in iFileNames:
        with tb.open_file(iFileName, mode='r') as h5in:
            file_extents = pd.read_hdf(iFileName, '/MC/extents', mode='r')

            if event_id in file_extents['evt_number'].tolist():
                print('\nEvt Id: {}  contained in {}\n'.format(event_id, iFileName))

                # Getting the mcParticles and mcHits of the right file
                file_mcHits  = load_mc_hits(iFileName)
                file_mcParts = load_mc_particles(iFileName)

                # Getting the mcParticles and mcHits of the right event
                event_mcHits       = file_mcHits.loc[event_id, :]
                active_mcHits      = event_mcHits[event_mcHits.label == 'ACTIVE']
                event_mcParticles  = file_mcParts.loc[event_id, :]

                #Getting relevant info from the event
                event_mcE = active_mcHits.E.sum()
                ini_time  = active_mcHits.time.min()
                end_time  = active_mcHits.time.max()
                
                # General event info
                print('  Event deposited energy = {0:.6f} MeV'.format(event_mcE))
                print('  Event initial time = {0:.3e} us,   Time width: {1:.3e} us'
                      .format(ini_time/units.mus, (end_time-ini_time)/units.mus))

                print('  Event has {} MC Particles'.format(len(event_mcParticles)))
                print('  Event has {} MC Hits'.format(len(event_mcHits)))
                print('')
                print('- List of MC Particles:')
                
                # Printing particles
                print_mc_particles(event_mcParticles, event_mcHits,
                                   evt_ini_time=ini_time, with_hits=with_hits)
                
                return

    # Event Id not found in any input file
    print('\nEvt Id: {}  NOT FOUND in any input file.\n'.format(event_id))



def plot_mc_event(event_id   : int,
	              iFileNames : List[str]
	             ) -> None:
    """
Plots the information of the event corresponding to event_id.
It will look for it into all the list of iFileNames passed."""

    # Going through all the input files
    for iFileName in iFileNames:
        with tb.open_file(iFileName, mode='r') as h5in:
            file_extents = pd.read_hdf(iFileName, '/MC/extents', mode='r')

            if event_id in file_extents['evt_number'].tolist():
                print('\nEvt Id: {}  contained in {}\n'.format(event_id, iFileName))

                # Getting the mcParticles and mcHits of the right file
                file_mcHits  = load_mc_hits(iFileName)
                file_mcParts = load_mc_particles(iFileName)

                # Getting the mcParticles and mcHits of the right event
                event_mcHits       = file_mcHits.loc[event_id, :]
                active_mcHits      = event_mcHits[event_mcHits.label == 'ACTIVE']

                # Plotting
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                p = ax.scatter(active_mcHits.x, active_mcHits.y, active_mcHits.z,
                               cmap='coolwarm', c=(active_mcHits.E / units.keV))
                cb = fig.colorbar(p, ax=ax)
                cb.set_label('Energy (keV)')
                plt.show()

                return

    # Event Id not found in any input file
    print('\nEvt Id: {}  NOT FOUND in any input file.\n'.format(event_id))
