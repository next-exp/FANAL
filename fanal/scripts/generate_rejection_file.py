import os
import sys

import tables as tb
import numpy  as np

from fanal.ana.ana_io_functions   import get_ana_group_name


##### SOME GLOBAL VARIABLES #####
VERBOSE       = True
BASE_PATH     = '/n/holylfs02/LABS/guenette_lab/data/NEXT/TON_SCALE'
NEXUS_VERSION = 'p5_00_02'
FANAL_VERSION = '1_01_00'

# ISOTOPE / SOURCE options
OPTIONS_SRC_ISO = {
    'ACTIVE':          ['bb0nu', 'Xe137'],
    'READOUT_PLANE':   ['Tl208', 'Bi214'],
    'CATHODE':         ['Tl208', 'Bi214'],
    'FIELD_CAGE':      ['Tl208', 'Bi214'],
    'INNER_SHIELDING': ['Tl208', 'Bi214']
}

# FWFM / VOXEL_SIZE options
OPTIONS_FWHM_VXL = {
    0.7: [[3,3,3], [10,10,10]],
    0.5: [[3,3,3], [10,10,10]],
#    0.7: [[10,10,10]],
#    0.5: [[3,3,3]]
}



def get_input_path(det_name, isotope, source, fwhm, voxel_size):
    '''
    It returns the PATH with all the analysis files
    '''
    iPATH  = BASE_PATH + '/' + det_name
    iPATH += f"/ANA.fanal_{FANAL_VERSION}"
    iPATH += f"/{isotope}/{source}"
    iPATH += "/FWHM_" + str(fwhm).replace('.','')
    iPATH += f"/Voxel_{voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}"
    iPATH += "/Output"
    #return iPATH
    return "/Users/Javi/Development/fanalIC/data/ana/Bi214" # XXXXXXXX



def get_rej_factor(path, group_name):
    '''
    It computes the global rejection factor of all the input files
    contained in the input path.
    It is expected that all the input files inside the path, represent
    a single set of source - isotope - energyRes and spatialDef.
    '''
    group_name = "/FANALIC/ANA_fwhm_07_voxel_10x10x10" # XXXXXXXX

    sim_events = roi_events = 0.
    for iFile_name in os.listdir(path):
        iFile_name = os.path.join(path, iFile_name)
        with tb.open_file(iFile_name, mode='r') as iFile:
            sim_events += iFile.get_node_attr(group_name, 'simulated_events')
            roi_events += iFile.get_node_attr(group_name, 'roi_filter_events')

    return roi_events / sim_events



##### EXECUTING THE SCRIPT #####

# Getting the detector name
try:
    det_name = sys.argv[1]
except IndexError:
    print("\nUsage: generate_rejection_file.py exp_name\n")
    sys.exit()

# Output csv file name
oFileName = f"rejection_factors.{det_name}.csv"

csv_content = f"##### {det_name} rejection factors #####"
csv_content += "\nsource,energyRes,spatialDef"

# All isotopes list
all_isotopes = []
for source in OPTIONS_SRC_ISO.keys():
    for isotope in OPTIONS_SRC_ISO[source]:
        if isotope not in all_isotopes:
            all_isotopes.append(isotope)

for isotope in all_isotopes:
    csv_content += f",{isotope}"


if VERBOSE:
    print(f"\n*** Generating the rejection factors of {det_name} ...")
    print(f"*   Input analysis files are expected to be in: {BASE_PATH}/{det_name}")
    print(f"*   cvs output file: {oFileName}")


# Passing through all options
for source in OPTIONS_SRC_ISO.keys():
    csv_content += f"\n\n# {source}"
    if VERBOSE:
        print(f"*   Source: {source} ...")
    for energyRes in OPTIONS_FWHM_VXL.keys():
        for spatialDef in OPTIONS_FWHM_VXL[energyRes]:

            # Get the group name
            ana_group_name = get_ana_group_name(energyRes, spatialDef)

            # Getting the rejection factors for all the isotopes
            rej_factors_str = f"\n{source},{energyRes},{spatialDef[0]}x{spatialDef[1]}x{spatialDef[2]}"
            for isotope in all_isotopes:
                if isotope in OPTIONS_SRC_ISO[source]:
                    iPATH = get_input_path(det_name, isotope, source,
                                           energyRes, spatialDef)
                    rej_factor = get_rej_factor(iPATH, ana_group_name)
                    rej_factors_str += f",{rej_factor}"
                    if VERBOSE:
                        print(f"    {energyRes} - {spatialDef} - {isotope}: {rej_factor:8}")
                else:
                    rej_factors_str += ","

            # Storing the rejection factors
            csv_content += rej_factors_str

# Dump csv content
oFile = open(oFileName, 'w')
oFile.write(csv_content)
oFile.close()

