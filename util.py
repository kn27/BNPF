from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import cascaded_union
import numpy as np
import time
import sys
from sklearn.preprocessing import normalize
from numpy import linalg
import pandas as pd


def aggregate_by_zone(trips, zones):
    #have to join individually because cannot find a way to find a field to join using geodataframe
    trips = trips[trips.geometry.apply(lambda x:len(x.boundary)>1)]
    trips.reset_index(inplace = True, drop = True)
    pu = gpd.GeoDataFrame(trips.geometry.apply(lambda x:x.boundary[0]))
    do = gpd.GeoDataFrame(trips.geometry.apply(lambda x:x.boundary[1]))
    pu_joined_zones = gpd.sjoin(zones,pu, how="right", op='contains')
    do_joined_zones = gpd.sjoin(zones,do, how="right", op='contains')
    agg = pd.concat([pu_joined_zones[['index_left']], do_joined_zones[['index_left']]],axis = 1, join = 'inner')
    agg.columns = ['pu', 'do']
    return agg[(-pd.isna(agg.do) & -pd.isna(agg.pu))],None
    #return agg[(-pd.isna(agg.do) & -pd.isna(agg.pu))], trips.loc[list(set(trips.index) - set(agg.index))]

def plot(trips, zones):
    fig, ax = plt.subplots(figsize = (15,15))
    zones.plot(ax=ax)
    trips.plot(ax = ax, color = 'red')

def get_zones():
    zones = gpd.read_file(r'taxi_zones\taxi_zones.shp')
    zones = zones.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    temp = zones[(zones.borough == 'Manhattan') & (zones.zone != 'Central Park')].reset_index(inplace = False, drop = True)
    temp = gpd.GeoDataFrame(geometry = [cascaded_union(temp.geometry)])
    return gpd.GeoDataFrame(geometry = temp.rotate(0).geometry)

def get_joint_grid(zones, bins = 50, plot = False):
    long_set = np.arange(min(b.bounds[0] for b in zones.geometry), max(b.bounds[2] for b in zones.geometry), step = (max(b.bounds[2] for b in zones.geometry) - min(b.bounds[0] for b in zones.geometry))/bins) 
    long_set = [(long_set[i], long_set[i+1]) for i in range(len(long_set)-1)]
    lat_set =  np.arange(min(b.bounds[1] for b in zones.geometry), max(b.bounds[3] for b in zones.geometry), step = (max(b.bounds[3] for b in zones.geometry) - min(b.bounds[1] for b in zones.geometry))/bins) 
    lat_set = [(lat_set[i], lat_set[i+1]) for i in range(len(lat_set)-1)]
    grid = gpd.GeoDataFrame(geometry = [cascaded_union([Point(long[0], lat[0]),Point(long[1], lat[0]),Point(long[0], lat[1]),Point(long[1], lat[1])]).envelope for long in long_set for lat in lat_set])
    joint = gpd.sjoin(grid, zones, how = 'inner', op = 'intersects')
    joint.reset_index(drop = True, inplace = True)
    joint.drop(['index_right'],axis = 1, inplace =True)
    if plot:
        fig, ax = plt.subplots(figsize = (10,10)) 
        joint.plot(**{'edgecolor': 'red'}, ax = ax)
    return joint

def get_trips(data):
    return gpd.GeoDataFrame(data, geometry = data.apply(lambda x:LineString([[x[0], x[1]], [x[2], x[3]]]), axis = 1))

def process(tax, zones):
    time0 = time.time()
    trips = get_trips(tax)
    transition, _ = aggregate_by_zone(trips, zones)
    print(f'Time:{time.time()-time0}')
    sys.stdout.flush()
    return transition

def group_and_map(aggregated, filename = None):
    if filename:
        aggregated = pd.read_csv(filename)
        aggregated['count'] = 1
    if 'count' not in aggregated.columns:
        aggregated['count'] = 1
    count = aggregated.groupby(['pu','do'])['count'].count()
    count = pd.DataFrame(count)
    count.columns = ['Count']
    count.reset_index(inplace = True)
    del aggregated
    keep = count.groupby(['pu'])['Count'].sum()
    keep = list(keep[keep/keep.sum() > 1e-4].index)
    count = count[count.pu.apply(lambda x: x in keep) * count.do.apply(lambda x: x in keep)]
    count.reset_index(inplace = True, drop = True)
    unique = count.pu.unique()
    statemap = dict(zip(unique, range(len(unique))))
    return count, statemap

def build_matrix(count, statemap):
    A = np.zeros((len(statemap),len(statemap)))
    for i in range(count.shape[0]):
        try:
            A[int(statemap[count.pu[i]]), int(statemap[count.do[i]])] = count.Count[i]
        except Exception as e:
            raise ValueError(e)
    A = normalize(A, norm = 'l1',axis = 1)
    assert all(np.isclose(linalg.norm(A, ord = 1,axis = 1), np.ones(A.shape[0])))
    return A

def inspect(A):
    plt.hist(A.sum(axis = 0), bins = 15,range = (0,15))
    plt.show()
    plt.hist(A.sum(axis = 1), bins = 15,range = (0,15))
    plt.show()

def get_center(grid):
    grid.geometry = grid.geometry.apply(lambda x: x.centroid)
    return grid
