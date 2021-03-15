import glob

import numpy  as np
import tables as tb
import pandas as pd

import invisible_cities.core.system_of_units    as units
from   invisible_cities.io.mcinfo_io        import load_mchits_df
from   invisible_cities.io.mcinfo_io        import load_mcparticles_df



def get_event_numbers_in_file(file_name: str) -> np.ndarray:
    """
    Get a list of the event numbers in the file
    based on the MC tables

    parameters
    ----------
    file_name : str
                File name of the input file

    returns
    -------
    evt_arr   : np.ndarray
                numpy array containing the list of all
                event numbers in file.
    """
    with tb.open_file(file_name, 'r') as h5in:

        def get_event_ids_table(tablename):
            try:
                evt_list = getattr(h5in.root.MC, tablename).cols.event_id
            except tb.exceptions.NoSuchNodeError:
                raise AttributeError('Trying to get event number from MC corrupted file.')
            return np.unique(evt_list).astype(int)

        evt_list = get_event_ids_table('particles')
        if len(evt_list):
            return evt_list
        return get_event_ids_table('sns_response')



def get_fname_with_event(event_id : int,
                         ifnames  : str
                        ) -> str :
    """
    It returns the mc ifname from all the ifnames
    containing the event_id
    """
    fnames = glob.glob(ifnames)
    if (len(fnames) == 0):
        print("Input files do NOT EXIST.")
        return ''
    for fname in fnames:
        if (event_id in get_event_numbers_in_file(fname)): return fname
    return ''



def print_mc_event(event_id  :  int,
                   ifnames   :  str,
                   with_hits :  bool = False
                  ) -> None :
    """Prints the information of the event corresponding to event_id."""
    
    # Getting the right file
    ifname = get_fname_with_event(event_id, ifnames)
    if ifname == '':
        print(f"\nEvent id: {event_id} NOT FOUND in input mc files.")
        return
    else:
        print(f"\nEvent Id: {event_id}  contained in {ifname}\n")

    # Getting the mcParticles and mcHits of the right event
    mcParts = load_mcparticles_df(ifname).loc[event_id]
    mcHits  = load_mchits_df(ifname).loc[event_id]

    print_mc_particles(mcParts, mcHits, with_hits)

    return



def print_mc_particles(mcParts   : pd.DataFrame,
                       mcHits    : pd.DataFrame,
                       with_hits : bool  = False
                      ) -> None:
    
    print(f"*** {len(mcParts)} MC particles:")
    #print(mcParts)

    t0 = mcHits.time.min()

    for part_id, part in mcParts.iterrows():
        # General Info
        print(f"* Particle {part_id}: name = {part.particle_name}, " + \
              f"  primary = {part.primary}")

        # Creator Info for non primary particles
        if not part.primary:
            mother_part = mcParts.loc[part.mother_id]
            print(f"  Initial:  Process = {part.creator_proc},   Mother id = " + \
                  f"{part.mother_id} ({mother_part.particle_name})")

        # Initial Info
        print(f"  Initial:  Momentum = ({part.initial_momentum_x/units.keV:.1f}, " + \
              f"{part.initial_momentum_y/units.keV:.1f}, " + \
              f"{part.initial_momentum_z/units.keV:.1f})" + \
              f" keV  ->  KinE: {part.kin_energy/units.keV:.1f} keV")
        print(f"  Initial:  Volume = {part.initial_volume} " + \
              f"  Vertex = ({part.initial_x:.1f}," + \
              f" {part.initial_y:.1f}, {part.initial_z:.1f}) mm") 

        # Decay Info
        print(f"  Decay  :  Volume = {part.final_volume} " + \
              f"  Vertex = ({part.final_x:.1f}," + \
              f" {part.final_y:.1f}, {part.final_z:.1f}) mm")
        print(f"  Decay  :  Process = {part.final_proc}   " + \
              f"Momentum = ({part.final_momentum_x/units.keV:.1f}, " + \
              f"{part.final_momentum_y/units.keV:.1f}, " + \
              f"{part.final_momentum_z/units.keV:.1f}) keV")

        # Daughter Particles Info
        daughter_parts = mcParts[mcParts.mother_id == part_id]
        print(f"  {len(daughter_parts)} daughter particles:")

        for daugh_id, daugh_part in daughter_parts.iterrows():
            print(f"    Part id: {daugh_id:3} - {daugh_part.particle_name:8} - " + \
                  f"{daugh_part.initial_volume:8} - {daugh_part.creator_proc:8}")

        # Printing hits
        if (with_hits == True):
            print_mc_hits(mcHits.loc[part_id], t0)
        else:
            print(f"  {len(mcHits)} MC hits")



def print_mc_hits(mcHits   : pd.DataFrame,
                  t0       : float
                 ) -> None :

    print(f"  {len(mcHits)} MC hits:")
    #print(mcHits)

    for hit_id, hit in mcHits.iterrows():
        print(f"    Hit {hit_id:3}   Label: {hit.label}" +\
              f"   Energy: {hit.energy/units.keV:6.3f} keV" + \
              f"   Pos: ({hit.x:5.1f}, {hit.y:5.1f}, {hit.z:5.1f}) mm" + \
              f"   Time: {(hit.time - t0)/units.mus:.1e} us")
