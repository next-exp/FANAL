import numpy as np
import pandas as pd
import networkx as nx

import json
import os
import uuid
from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict

from itertools   import combinations

from pandas      import DataFrame
from typing      import List, Tuple, Dict, TypeVar

from fanal.utils.types         import XYZ
from fanal.utils.general_utils import bin_data_with_equal_bin_size
from fanal.containers.tracks   import Track



def voxelize_hits2(mcHits     : pd.DataFrame,
                   voxel_size : XYZ,
                   baryc      : bool = True
                   ) -> Tuple[pd.DataFrame, XYZ]:
    """
    Takes an mcHits DF from nexus with fields (x,y,z,energy) and
    voxelizes the data in cubic voxels of size 'bin_size' and returns a
    DataFrame with voxels and the effective voxel size.
    """

    def voxelize_hits_bc(df : pd.DataFrame)->pd.Series:
        """
        Computes the barycenters in x,y,z
        """
        def barycenter(df, var, etot):
            return np.sum([np.dot(a,b)\
                           for a, b  in zip(df[var] , df.energy)]) / etot
        d = {}
        etot   = df['energy'].sum()
        d['x'] = barycenter(df, 'x', etot)
        d['y'] = barycenter(df, 'y', etot)
        d['z'] = barycenter(df, 'z', etot)
        d['energy'] = etot
        return pd.Series(d)

    def voxelize_hits_mean(df : pd.DataFrame)->pd.Series:
        """
        Compute the averages in x, y, z
        """
        d = {}
        d['x'] = df['x'].mean()
        d['y'] = df['y'].mean()
        d['z'] = df['z'].mean()
        d['energy'] = df['energy'].sum()
        return pd.Series(d)

    df = mcHits.copy()
    (xbins, ybins, zbins), eff_sizes = \
        bin_data_with_equal_bin_size([df.x, df.y, df.z], voxel_size)
    num_voxels = len(xbins) * len(ybins) * len(zbins)
    if (num_voxels >= 1e+6):
        print(f"*** Caution: Number of voxels: {num_voxels} is too high.")

    df['x_bins'] = pd.cut(df['x'], bins=xbins, labels=range(len(xbins)-1))
    df['y_bins'] = pd.cut(df['y'], bins=ybins, labels=range(len(ybins)-1))
    df['z_bins'] = pd.cut(df['z'], bins=zbins, labels=range(len(zbins)-1))

    if baryc:
        vhits = df.groupby(['x_bins','y_bins','z_bins']) \
                  .apply(voxelize_hits_bc).dropna().reset_index(drop=True)
    else:
        vhits = df.groupby(['x_bins','y_bins','z_bins']) \
                  .apply(voxelize_hits_mean).dropna().reset_index(drop=True)

    return vhits, XYZ.from_array(eff_sizes)



def distance_between_voxels(va : List[float], vb : List[float]) -> float:
    """
    Return the distance between two voxels
    """
    return np.linalg.norm(np.array([vb[0], vb[1], vb[2]]) - \
                          np.array([va[0], va[1], va[2]]))



def make_track_graphs2(voxel_df   : pd.DataFrame,
                       contiguity : float)->List[nx.Graph]:
    """
    Make "graph-tracks" (gtracks) using networkx:

    1. Define a graph such that each voxel is a node and there is a link
    (or edge) between each pair of nodes which are at a distance smaller
    than defined by contiguity.

    2. Return a list of graphs made with connected components. Each connected
    component graph is made of a set of connected nodes (eg. nodes which are
    at a distance smaller than contiguity)
    """
    def connected_component_subgraphs(G):
        return (G.subgraph(c).copy() for c in nx.connected_components(G))

    voxel_list = [(v[0], v[1], v[2], v[3]) for v in voxel_df.values]
    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxel_list)
    for va, vb in combinations(voxel_list, 2):
        d = distance_between_voxels(va, vb)
        if d < contiguity:
            voxel_graph.add_edge(va, vb, distance = d)
    return list(connected_component_subgraphs(voxel_graph))



def find_extrema_and_length_from_dict(distance : Dict[List[float], Dict[List[float], float]]
                                     ) -> Tuple[List[float], List[float], float]:
    """
    Find the extrema and the length of a track,
    given its dictionary of distances.
    """
    if not distance:
        raise NoVoxels
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.)
    first, last, max_distance = None, None, 0
    for (voxel1, dist_from_voxel_1_to), (voxel2, _) in \
                                        combinations(distance.items(), 2):
        d = dist_from_voxel_1_to[voxel2]
        if d > max_distance:
            first, last, max_distance = voxel1, voxel2, d
    return first, last, max_distance



def get_and_store_blobs(track    : Track,
                        graph    : nx.Graph,
                        blob_rad : float
                       ) -> Tuple[XYZ, XYZ]:
    """
    Get extremes and blobs info from the nx.Graph and stores it in the Track
    Returns the blob positions ordered by energy
    """
    distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight='distance'))
    ext1, ext2, length = find_extrema_and_length_from_dict(distances)

    # Getting blobs info
    blob1_energy, blob1_num_voxels, blob2_energy, blob2_num_voxels, ovlp_energy = \
        blobs_info(distances, (ext1, ext2), blob_rad)

    # Storing ordered information into the Track
    track.length      = length
    track.ovlp_energy = ovlp_energy

    if blob1_energy >= blob2_energy:
        track.blob1_x, track.blob1_y, track.blob1_z = ext1[0], ext1[1], ext1[2]
        track.blob2_x, track.blob2_y, track.blob2_z = ext2[0], ext2[1], ext2[2]
        track.blob1_energy   = blob1_energy
        track.blob2_energy   = blob2_energy
        track.blob1_num_hits = blob1_num_voxels
        track.blob2_num_hits = blob2_num_voxels
    else:
        track.blob1_x, track.blob1_y, track.blob1_z = ext2[0], ext2[1], ext2[2]
        track.blob2_x, track.blob2_y, track.blob2_z = ext1[0], ext1[1], ext1[2]
        track.blob1_energy   = blob2_energy
        track.blob2_energy   = blob1_energy
        track.blob1_num_hits = blob2_num_voxels
        track.blob2_num_hits = blob1_num_voxels

    return XYZ(track.blob1_x, track.blob1_y, track.blob1_z), \
           XYZ(track.blob2_x, track.blob2_y, track.blob2_z)



def blobs_info(distances : Dict[List[float], Dict[List[float], float]],
               extremes  : Tuple[List[float], List[float]],
               blob_rad  : float
              ) -> Tuple[float, int, float, int, float]:
    """
    Return the total energy and the number of voxels contained
    in a blob of radius blob_rad around the extremes, and the overlap energy
    """

    # Blob1
    distances_from_extreme = distances[extremes[0]]
    voxels_in_blob1 = [voxel for voxel, distance in distances_from_extreme.items() \
                      if distance <= blob_rad]
    blob_energy1 = sum(voxel[3] for voxel in voxels_in_blob1)

    # Blob2
    distances_from_extreme = distances[extremes[1]]
    voxels_in_blob2 = [voxel for voxel, distance in distances_from_extreme.items() \
                      if distance <= blob_rad]
    blob_energy2 = sum(voxel[3] for voxel in voxels_in_blob2)

    # Overlap energy
    ovlp_energy = sum(voxel[3] for voxel in \
                  set(voxels_in_blob1).intersection(set(voxels_in_blob2)))

    return blob_energy1, len(voxels_in_blob1), blob_energy2, len(voxels_in_blob2), ovlp_energy

