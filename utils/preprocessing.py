from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import cascaded_union
import numpy as np
import time
from util import *
import multiprocessing as mp
import sys 
import hickle as hkl 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',default = r'yellow_tripdata_2016-01.csv')
    parser.add_argument('gridsize',default = 100, type = int)
    args = parser.parse_args()
    time0 = time.time()
    funclist = []
    transitions = []
    tax = pd.read_csv(args.input,
                    usecols=['dropoff_longitude', 'dropoff_latitude', 'pickup_longitude', 'pickup_latitude'],
                    dtype =np.float32,chunksize = 100000)
    i = 0
    
    man = get_zones()
    grid = get_joint_grid(man, args.gridsize)
    grid.to_csv(f'{args.gridsize}_Grid.csv', index = True)
    pool = mp.Pool(4)
    for tax_sample in tax:
        f = pool.apply_async(process, args = [tax_sample], kwds = {'zones':grid})
        funclist.append(f)
    for f in funclist:
        transitions.append(f.get(timeout = 60))
        #Cannot dO this, why?
        #result.to_csv(r'test2.csv', index = False)
        i += 1
        print(f'Complete task {i}')
    aggregated = pd.concat(transitions)
    aggregated.to_csv(f'{args.gridsize}_Grid_AggregatedTrip.csv', index = False)
    print(f'Time:{time.time() - time0}')
    aggregated = pd.read_csv(f'{args.gridsize}_Grid_AggregatedTrip.csv')
    count,statemap = group_and_map(aggregated)
    A = build_matrix(count,statemap)
    hkl.dump(A, f'{args.gridsize}_Matrix.hkl')
    hkl.dump(statemap,f'{args.gridsize}_Statemap.hkl')

