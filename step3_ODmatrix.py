# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'xu'

import os, sys
import pickle, csv
import copy, random
import time, datetime, pytz
import operator
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')

import vaex

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from collections import Counter

from geostatsmodels import utilities, variograms, model, kriging, geoplot

from rtree import index
from scipy import spatial
import statsmodels as sm
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score

import geojson
from shapely.ops import unary_union
import fiona
import itertools
from mpl_toolkits.basemap import Basemap
from shapely.geometry import shape, mapping, Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch


# global variables
DallasTZ = pytz.timezone('US/Central')

if os.path.exists('/media/xu/Elements/Study/HuMNetLab/Data/Dallas/'):
    dataPath = '/media/xu/Elements/Study/HuMNetLab/Data/Dallas/'
if os.path.exists('/home/xu/Data/Dallas/'):
    dataPath = '/home/xu/Data/Dallas/'
if os.path.exists('/home/xu/Documents/Data/Dallas/'):
    dataPath = '/home/xu/Documents/Data/Dallas/'


DallasBoundary = [-98.068545, 31.708887, -95.858645, 33.434057]

minValues = {'Population':0, 'stayPop':0, 'stayPop_density':0,
             'flow':0, 'flowIn':0, 'flowOut':0, 'flowInS':2, 'flowOutS':2,
             'housing_price':0}

maxValues = {'Population':5e5, 'stayPop':5e5,
             'flow':20000, 'flowIn':500, 'flowOut':500, 'flowInS':4.5,
             'flowOutS':4.5, 'housing_price':20}

numColorsIntervals = {'Population':21, 'stayPop':21,
                      'flow':41, 'flowIn':21, 'flowOut':21, 'flowInS':26, 'flowOutS':26,
                      'housing_price':21}

numColorLabels = {'Population':5, 'stayPop':5,
                  'flow':4, 'flowIn':5, 'flowOut':5,  'flowInS':6, 'flowOutS':6,
                  'housing_price':5}

cmPalettes = {'Population':'Greys', 'stayPop':'RdYlBu_r',
              'flow':'hot_r', 'flowIn':'Blues', 'flowOut':'Blues',
               'flowInS':'viridis', 'flowOutS':'viridis',
              'housing_price':'Blues'}


# calculate distance between two locations in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def custom_colorbar(cmap, ncolors, labels, **kwargs):
    """Create a custom, discretized colorbar with correctly formatted/aligned labels.

    cmap: the matplotlib colormap object you plan on using for your graph
    ncolors: (int) the number of discrete colors available
    labels: the list of labels for the colorbar. Should be the same length as ncolors.
    """
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    norm = BoundaryNorm(range(0, ncolors), cmap.N)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)

    colorbar.set_ticks(np.linspace(0, ncolors, ncolors+1)+0.5)
    colorbar.set_ticklabels(range(0, ncolors))
    colorbar.set_ticklabels(labels)

    return colorbar


def mapTripsToTSZ():
    # load the study regions
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/TSZ/TSZ5352.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # load trips
    userTrips = open(dataPath + "userData/allUser_tripInf.csv", "r")
    outData = open(dataPath + "userData/allUser_tripInf_TSZ.csv", "wb")

    header = userTrips.readline()
    outData.writelines(header.rstrip() + ",sTSZ,tTSZ\n")

    count = 0
    for row in userTrips:
        count += 1
        if count%1000==0:
            print(count)
        row = row.rstrip().split(',')
        sLon = float(row[4])
        sLat = float(row[5])
        tLon = float(row[6])
        tLat = float(row[7])
        sPt = Point(sLon, sLat)
        tPt = Point(tLon, tLat)
        sTSZ = -1
        tTSZ = -1
        # iterate through spatial index
        for j in idx.intersection(sPt.coords[0]):
            if sPt.within(shape(polygons[j]['geometry'])):
                sTSZ = str(polygons[j]['properties']['TSZ'])

        for j in idx.intersection(tPt.coords[0]):
            if tPt.within(shape(polygons[j]['geometry'])):
                tTSZ = str(polygons[j]['properties']['TSZ'])

        if sTSZ!=-1 and tTSZ != -1:
            row_save = ','.join(row + [sTSZ, tTSZ])
            outData.writelines(row_save + "\n")
    userTrips.close()
    outData.close()



def mapTripsToZip():
    # get the centroids of each TSZ
    geoFile = open(dataPath + "Geo/TSZ/TSZ5352.geojson", 'rb')
    geoData = geojson.load(geoFile)
    TSZ_centroids = {}

    for t in geoData['features']:
        TSZ = int(t['properties']['TSZ'])
        multiPolygons = t['geometry']['coordinates']
        maxLen = 0
        for poly in multiPolygons:
            polyLen = len(poly[0])
            if polyLen > maxLen:
                polygon = poly[0]
                maxLen = polyLen
        # centroid of this polygon
        cen_lon = np.mean([p[0] for p in polygon])
        cen_lat = np.mean([p[1] for p in polygon])
        TSZ_centroids[TSZ] = (cen_lon, cen_lat)
    geoFile.close()

    # load the zipcode map
    # load the study regions
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # find the matching between TSZ and zipcode (TSZ is smaller)
    TSZtoZipcodes = {}
    for tsz in TSZ_centroids:
        lon, lat = TSZ_centroids[tsz]
        Pt = Point(lon, lat)
        # iterate through spatial index
        zipcode = -1
        for j in idx.intersection(Pt.coords[0]):
            if Pt.within(shape(polygons[j]['geometry'])):
                zipcode = int(polygons[j]['properties']['ZCTA5CE10'])
        TSZtoZipcodes[tsz] = zipcode

    # load the study regions
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        bound = shape(poly['geometry']).bounds
        idx.insert(pos, bound)

    # load trips
    userTrips = open(dataPath + "userData/allUser_tripInf_TSZ.csv", "r")
    outData = open(dataPath + "userData/allUser_tripInf_ZIP.csv", "wb")

    header = userTrips.readline()
    outData.writelines(header.rstrip() + ",sZip,tZip\n")

    count = 0
    for row in userTrips:
        count += 1
        if count%1000==0:
            print(count)
        row = row.rstrip().split(',')
        sTsz = int(row[9])
        tTsz = int(row[10])
        sZip = -1
        tZip = -1
        try:
            sZip = str(TSZtoZipcodes[sTsz])
            tZip = str(TSZtoZipcodes[tTsz])
        except:
            sZip = -1
            tZip = -1

        if sZip!=-1 and tZip != -1:
            row_save = ','.join(row + [sZip, tZip])
            outData.writelines(row_save + "\n")
    userTrips.close()
    outData.close()


def ODMatrix_TSZtoZip():
    # get the centroids of each TSZ
    geoFile = open(dataPath + "Geo/TSZ/TSZ5352.geojson", 'rb')
    geoData = geojson.load(geoFile)
    TSZ_centroids = {}

    for t in geoData['features']:
        TSZ = int(t['properties']['TSZ'])
        multiPolygons = t['geometry']['coordinates']
        maxLen = 0
        for poly in multiPolygons:
            polyLen = len(poly[0])
            if polyLen > maxLen:
                polygon = poly[0]
                maxLen = polyLen
        # centroid of this polygon
        cen_lon = np.mean([p[0] for p in polygon])
        cen_lat = np.mean([p[1] for p in polygon])
        TSZ_centroids[TSZ] = (cen_lon, cen_lat)
    geoFile.close()

    # load the zipcode map
    # load the study regions
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/DTW_zipcodes.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # find the matching between TSZ and zipcode (TSZ is smaller)
    TSZtoZipcodes = {}
    for tsz in TSZ_centroids:
        lon, lat = TSZ_centroids[tsz]
        Pt = Point(lon, lat)
        # iterate through spatial index
        zipcode = -1
        for j in idx.intersection(Pt.coords[0]):
            if Pt.within(shape(polygons[j]['geometry'])):
                zipcode = int(polygons[j]['properties']['ZCTA5CE10'])
        TSZtoZipcodes[tsz] = zipcode

    # Convert the TSZ OD matrix to Zipcode
    inData = open(dataPath + "OD/2014_AM_OD_Sum.csv", "rb")
    inData.readline()

    zipODMatrix = {}
    totalFlow_tsz = 0
    totalFlow_zip = 0
    count = 0
    for row in inData:
        count +=1
        if count%1e4==0:
            print(count)
        row = row.rstrip().split(',')
        sTSZ = int(row[0])
        tTsz = int(row[1])
        sZipcode = TSZtoZipcodes[sTSZ]
        tZipcode = TSZtoZipcodes[tTsz]
        flow = float(row[4]) + 2*float(row[5])  # DA (drive alone) and SR (shared riding)
        totalFlow_tsz += flow
        if sZipcode == tZipcode or sZipcode==-1 or tZipcode==-1:
            continue
        totalFlow_zip += flow
        try:
            zipODMatrix[(sZipcode, tZipcode)] += flow
        except:
            zipODMatrix[(sZipcode, tZipcode)] = flow

    inData.close()

    print("# total flow in TSZ : ", totalFlow_tsz)
    print("# total flow in Zip : ", totalFlow_zip)

    # save data
    outData = open(dataPath + "OD/OD_matrix_NCTCOG_AM.csv", "wb")
    outData.writelines("szip,tZip,flow\n")
    for key in zipODMatrix:
        row = [str(key[0]), str(key[1]), str(zipODMatrix[key])]
        outData.writelines(','.join(row) + '\n')
    outData.close()


# AM OD matrix for LBS data, considering trips during the entire timespan
def ODMatrix_LBS():
    # get the centroids of each TSZ
    geoFile = open(dataPath + "Geo/TSZ/TSZ5352.geojson", 'rb')
    geoData = geojson.load(geoFile)
    TSZ_centroids = {}

    for t in geoData['features']:
        TSZ = int(t['properties']['TSZ'])
        multiPolygons = t['geometry']['coordinates']
        maxLen = 0
        for poly in multiPolygons:
            polyLen = len(poly[0])
            if polyLen > maxLen:
                polygon = poly[0]
                maxLen = polyLen
        # centroid of this polygon
        cen_lon = np.mean([p[0] for p in polygon])
        cen_lat = np.mean([p[1] for p in polygon])
        TSZ_centroids[TSZ] = (cen_lon, cen_lat)
    geoFile.close()

    # load the zipcode map
    # load the study regions
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/DTW_zipcodes.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # find the matching between TSZ and zipcode (TSZ is smaller)
    TSZtoZipcodes = {}
    for tsz in TSZ_centroids:
        lon, lat = TSZ_centroids[tsz]
        Pt = Point(lon, lat)
        # iterate through spatial index
        zipcode = -1
        for j in idx.intersection(Pt.coords[0]):
            if Pt.within(shape(polygons[j]['geometry'])):
                zipcode = int(polygons[j]['properties']['ZCTA5CE10'])
        TSZtoZipcodes[tsz] = zipcode

    # return 0

    # load LBS trips
    inData = open(dataPath + "userData/allUser_tripInf_TSZ.csv", "rb")
    inData.readline()
    zipODMatrix = {}
    totalTrips = 0
    count = 0
    for row in inData:
        count += 1
        if count%1e4 == 0:
            print(count)
        row = row.rstrip().split(',')
        depTime = row[2]
        depHour = int(depTime.split(' ')[1][:2])
        # 7–10 am (AM), 11 am–15 pm (MD), 16–19 pm (PM), and the rest of the day (RD)
        # AM
        # if depHour < 7 or depHour > 10:
        #     continue
        # MD
        # if depHour < 11 or depHour > 15:
        #    continue
        # PM
        # if depHour < 16 or depHour > 19:
        #     continue
        # RD
        if depHour > 6 and depHour < 20:
            continue

        sTSZ = int(row[9])
        tTSZ = int(row[10])
        sZipcode = TSZtoZipcodes[sTSZ]
        tZipcode = TSZtoZipcodes[tTSZ]
        if sZipcode == tZipcode or sZipcode==-1 or tZipcode==-1:
            continue
        totalTrips += 1

        try:
            zipODMatrix[(sZipcode, tZipcode)] += 1
        except:
            zipODMatrix[(sZipcode, tZipcode)] = 1

    inData.close()

    print("# total AM trips in LBS : ", totalTrips)

    # save data
    outData = open(dataPath + "OD/OD_matrix_LBS_RD.csv", "wb")
    outData.writelines("szip,tZip,flow\n")
    for key in zipODMatrix:
        row = [str(key[0]), str(key[1]), str(zipODMatrix[key])]
        outData.writelines(','.join(row) + '\n')
    outData.close()


# compare the LBS OD with the NCTCOG OD
def ODcomparison():
    zipcodeODMatrix = {}
    # load the LBS OD
    inData = open(dataPath + "OD/OD_matrix_LBS_AM_expanded.csv", "rb")
    inData.readline()
    for row in inData:
        row = row.rstrip().split(',')
        sZipcode = int(row[0])
        tZipcode = int(row[1])
        flow = float(row[3])
        zipcodeODMatrix[(sZipcode, tZipcode)] = [flow, 0]
    inData.close()

    # load the NCTCOG OD
    inData = open(dataPath + "OD/OD_matrix_NCTCOG_AM.csv", "rb")
    inData.readline()
    for row in inData:
        row = row.rstrip().split(',')
        sZipcode = int(row[0])
        tZipcode = int(row[1])
        flow = float(row[2])
        try:
            zipcodeODMatrix[(sZipcode, tZipcode)][1] = flow
        except:
            zipcodeODMatrix[(sZipcode, tZipcode)] = [0, flow]
    inData.close()

    # scatter plot
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)

    flow_LBS = [value[0] for key, value in zipcodeODMatrix.iteritems()]
    flow_NCTCOG = [value[1] for key, value in zipcodeODMatrix.iteritems()]

    r2 = r2_score(flow_LBS, flow_NCTCOG)
    print("r2 : ", r2)

    # model = sm.OLS(flow_NCTCOG, flow_LBS).fit()

    # Print out the statistics
    # print(model.summary())

    # slop = model.params[0]

    para = stats.linregress(flow_NCTCOG, flow_LBS)
    slop = para[0]

    print(para)

    flow_LBS_rectified = [i*slop for i in flow_LBS]


    plt.scatter(flow_LBS, flow_NCTCOG, marker='o', s=5, edgecolors='#de2d26', facecolors='#de2d26', lw=0.5, alpha=0.1)

    # plot line
    plt.plot([1,2e4], [1, 2e4], lw=2, linestyle="--", color='#333333')
    ax.annotate(r'$r^2$ =' + "%.2f" % r2 + '\n' +
                'slop =' + "%.2f" % slop,
                xy=(2, 3e3), fontsize=12)

    plt.xscale('log', nonposx='clip')
    plt.yscale('log', nonposy='clip')
    plt.xlim(1)
    plt.ylim(1)
    plt.xlabel("LBS flow of active users in 6 months")
    plt.ylabel("NCTCOG flow")

    plt.tight_layout()
    plt.savefig(dataPath + "OD/LBS_NCTCOG_flowComp_expanded.png", dpi=300)
    plt.savefig(dataPath + "OD/LBS_NCTCOG_flowComp_expanded.pdf")
    plt.close()



# analyze LBS flow during one week
def LBS_tripsFlow():
    hourlyFlow_week = [0 for i in range(24*7)]

    # load LBS trips
    inData = open(dataPath + "userData/allUser_tripInf_TSZ.csv", "rb")
    inData.readline()
    zipODMatrix = {}
    totalTrips = 0
    count = 0
    for row in inData:
        count += 1
        if count % 1e4 == 0:
            print(count)
        row = row.rstrip().split(',')
        depTime = row[2]
        depTime_dt = datetime.datetime.strptime(depTime, '%Y-%m-%d %H:%M:%S')
        weekday = depTime_dt.weekday()
        depHour = depTime_dt.hour

        totalTrips += 1

        idx = weekday*24 + depHour
        hourlyFlow_week[idx] += 1

    inData.close()

    hourlyFlow_week = [i/float(totalTrips) for i in hourlyFlow_week]

    # plot
    fig = plt.figure(figsize=(6,3))
    plt.plot(range(24*7), hourlyFlow_week, lw=2, color='#333333')
    plt.xticks(range(0,24*7+1,24))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xlabel("Time [hr]")
    plt.ylabel("Density of travel flow")
    plt.tight_layout()
    plt.savefig(dataPath + "OD/hourlyFlow_LBS.png", dpi=300)
    plt.savefig(dataPath + "OD/hourlyFlow_LBS.pdf")
    plt.close()


# plot the od flow map


# draw od flows
def drawMap_geojson_od(para):
    # load od flow
    ODData = np.genfromtxt(dataPath + 'OD/OD_matrix_LBS_AM.csv',
                              dtype=float, delimiter=',', skip_header=1)
    ODFlow = {}
    for row in ODData:
        oTract = int(row[0])
        dTract = int(row[1])
        flow = float(row[2])
        if oTract==dTract:
            continue
        ODFlow[(oTract, dTract)] = flow

    ODFlowCombined = {}
    for od in ODFlow.keys():
        od_r = (od[1], od[0])
        f1 = ODFlow[od]
        try:
            f2 = ODFlow[od_r]
        except:
            f2 = 0
        if od not in ODFlowCombined and od_r not in ODFlowCombined:
            ODFlowCombined[od] = f1+f2

    print("Max flow : ", np.max(ODFlowCombined.values()))
    print("# of flows : ", len(ODFlowCombined))

    ODFlow = ODFlowCombined

    topNum = 5000
    # top flow volume
    from collections import Counter
    top_flowVolume = []
    d = Counter(ODFlow)
    for k, v in d.most_common(topNum):
        top_flowVolume.append(k)

    # reverse the list, plot the lower flow first
    top_flowVolume = top_flowVolume[::-1]

    # load map
    minValue, maxValue = minValues[para], maxValues[para]
    # minValue, maxValue = 0, 4000
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    coords = DallasBoundary
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coords[0], coords[2]]),
        lat_0=np.mean([coords[1], coords[3]]),
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - (extra * h),
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + (extra * h),
        resolution='i',  suppress_ticks=True)

    polygons = []
    for f in geoData['features']:
        nodes = []
        listOfNodes = f['geometry']['coordinates'][0][0]
        for n in listOfNodes:
            n_new = m(n[0], n[1])
            n_tup = tuple(n_new)
            nodes.append(n_tup)
        polygons.append(nodes)

    df_map = pd.DataFrame({
        # 'poly': [Polygon(feature['geometry']['coordinates'][0][0]) for feature in geoData['features']],
        'poly': [Polygon(poly) for poly in polygons],
        'name': [int(feature['properties']['GEOID10']) for feature in geoData['features']],
        'lon': [feature['properties']['x1'] for feature in geoData['features']],
        'lat': [feature['properties']['y1'] for feature in geoData['features']]
    })

    centroids = {}
    for index, row in df_map.iterrows():
        lon = float(row['lon'])
        lat = float(row['lat'])
        centroids[row['name']] = m(lon, lat)

    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))

    numColors = numColorsIntervals[para]
    breaks = np.linspace(minValue, maxValue, numColors)
    def self_categorize(entry, breaks):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return i
            if entry > breaks[-1]:
                return len(breaks)-1
        return -1

    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    # ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    ax = fig.add_subplot(111, frame_on=False)

    if cmPalettes[para] != '':
        cmap = plt.get_cmap(cmPalettes[para])
    else:
        cmap = geoplot.YPcmap

    if cmPalettes[para] == 'RdYlBu_r':
        cmap = truncate_colormap(cmap, 0.3, 1.0)
    if cmPalettes[para] == 'hot':
        cmap = truncate_colormap(cmap, 0.1, 1.0)

    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    pc.set_facecolor('#dddddd')
    ax.add_collection(pc)

    # Plot flow
    for od in top_flowVolume:
        flow = ODFlow[od]
        # assign each flow color
        colorLev = self_categorize(flow, breaks)
        color = cmap(colorLev/float(numColors))
        sLon, sLat = centroids[od[0]]
        tLon, tLat = centroids[od[1]]
        alphaLev = np.power(colorLev/float(numColors), 0.6)
        plt.plot([sLon, tLon], [sLat, tLat], lw=2, c=color, alpha=alphaLev)

    numLabels = numColorLabels[para]
    interval = maxValue/float(numLabels)
    # jenks_labels = [str(float(breaks[i])) if np.round((breaks[i]-minValue)%interval)==0 else '' for i in range(numColors)]
    jenks_labels = []
    for i in range(numColors):
        temp = np.round((breaks[i]-minValue)%interval)
        print(temp, breaks[i], interval)
        if temp == 0:
            jenks_labels.append(int(breaks[i]))
        else:
            jenks_labels.append('')

    # Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 10.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)

    # ncolors+1 because we're using a "zero-th" color
    cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
    cbar.ax.tick_params(labelsize=16)

    # fig.suptitle(para + " - " + hour + ":00:00", fontdict={'size':24, 'fontweight':'bold'}, y=0.92)
    plt.savefig(dataPath + 'OD/OD_matrix_LBS_AM.pdf', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5) # facecolor='#F2F2F2'


# draw od flows
def drawMap_geojson_od_log(para):
    # load od flow
    ODData = np.genfromtxt(dataPath + 'OD/OD_matrix_NCTCOG_AM.csv',
                              dtype=float, delimiter=',', skip_header=1)
    ODFlow = {}
    for row in ODData:
        oTract = int(row[0])
        dTract = int(row[1])
        flow = float(row[2])
        if oTract==dTract:
            continue
        ODFlow[(oTract, dTract)] = flow

    ODFlowCombined = {}
    for od in ODFlow.keys():
        od_r = (od[1], od[0])
        f1 = ODFlow[od]
        try:
            f2 = ODFlow[od_r]
        except:
            f2 = 0
        if od not in ODFlowCombined and od_r not in ODFlowCombined:
            ODFlowCombined[od] = f1+f2

    print("Max flow : ", np.max(ODFlowCombined.values()))
    print("# of flows : ", len(ODFlowCombined))

    # distance between any two tracts
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    centroids = {}
    for t in geoData['features']:
        zipcodeId = int(t['properties']['GEOID10'])
        # centroid of this polygon
        cen_lon = t['properties']['x1']
        cen_lat = t['properties']['y1']
        centroids[zipcodeId] = (cen_lon, cen_lat)
    geojsonFile.close()

    zipcodeDistance = {}
    for oZipcode in centroids:
        for dZipcode in centroids:
            if oZipcode == dZipcode:
                continue
            oLon, oLat = centroids[oZipcode]
            dLon, dLat = centroids[dZipcode]
            dist = haversine(oLat, oLon, dLat, dLon)
            zipcodeDistance[(oZipcode, dZipcode)] = dist


    # ODFlow = ODFlowCombined

    topNum = 5000
    # top flow volume
    from collections import Counter
    top_flowVolume = []
    d = Counter(ODFlow)
    for k, v in d.most_common(topNum):
        top_flowVolume.append(k)

    # reverse the list, plot the lower flow first
    top_flowVolume = top_flowVolume[::-1]

    # load map
    # minValue, maxValue = minValues[para], maxValues[para]

    minValue, maxValue = 1000, 10000
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    coords = DallasBoundary
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coords[0], coords[2]]),
        lat_0=np.mean([coords[1], coords[3]]),
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - (extra * h),
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + (extra * h),
        resolution='i',  suppress_ticks=True)

    polygons = []
    for f in geoData['features']:
        nodes = []
        # find the max length of polygon from multiple polygons
        multiPolygons = f['geometry']['coordinates']
        maxLen = 0
        for poly in multiPolygons:
            polyLen = len(poly[0])
            if polyLen > maxLen:
                listOfNodes = poly[0]
                maxLen = polyLen

        # listOfNodes = f['geometry']['coordinates'][0][0]
        for n in listOfNodes:
            n_new = m(n[0], n[1])
            n_tup = tuple(n_new)
            nodes.append(n_tup)
        polygons.append(nodes)

    df_map = pd.DataFrame({
        # 'poly': [Polygon(feature['geometry']['coordinates'][0][0]) for feature in geoData['features']],
        'poly': [Polygon(poly) for poly in polygons],
        'name': [int(feature['properties']['GEOID10']) for feature in geoData['features']],
        'lon': [feature['properties']['x1'] for feature in geoData['features']],
        'lat': [feature['properties']['y1'] for feature in geoData['features']]
    })

    centroids = {}
    for index, row in df_map.iterrows():
        lon = float(row['lon'])
        lat = float(row['lat'])
        centroids[row['name']] = m(lon, lat)

    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))

    # numColors = numColorsIntervals[para]
    numColors = 10
    breaks = np.linspace(minValue, maxValue, numColors)
    def self_categorize(entry, breaks):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return i
            if entry > breaks[-1]:
                return len(breaks)-1
        return -1

    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    # ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    ax = fig.add_subplot(111, frame_on=False)

    if cmPalettes[para] != '':
        cmap = plt.get_cmap(cmPalettes[para])
    else:
        cmap = geoplot.YPcmap

    if cmPalettes[para] == 'RdYlBu_r':
        cmap = truncate_colormap(cmap, 0.3, 1.0)
    if cmPalettes[para] == 'hot_r':
        cmap = truncate_colormap(cmap, 0.1, 0.8)

    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    pc.set_facecolor('#dddddd')
    ax.add_collection(pc)

    # Plot flow
    for od in top_flowVolume:
        flow = ODFlow[od]
        dist = zipcodeDistance[od]
        # if dist < 35:
        #     continue
        if flow <= minValue:
            continue
        # assign each flow color
        colorLev = self_categorize(flow, breaks)
        color = cmap(colorLev/float(numColors))
        sLon, sLat = centroids[od[0]]
        tLon, tLat = centroids[od[1]]
        alphaLev = np.power(colorLev/float(numColors), 0.55)
        lw = np.sqrt(flow)/10.0
        plt.plot([sLon, tLon], [sLat, tLat], lw=lw, c=color, solid_capstyle='round', alpha=alphaLev)

    # numLabels = numColorLabels[para]
    numLabels = 5
    interval = maxValue/float(numLabels)
    # jenks_labels = [str(float(breaks[i])) if np.round((breaks[i]-minValue)%interval)==0 else '' for i in range(numColors)]
    jenks_labels = []
    for i in range(numColors):
        temp = np.round((breaks[i]-minValue)%interval)
        print(temp, breaks[i], interval)
        if temp == 0:
            jenks_labels.append(int(breaks[i]))
        else:
            jenks_labels.append('')

    # Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 20.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)

    # ncolors+1 because we're using a "zero-th" color
    cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
    cbar.ax.tick_params(labelsize=16)

    # fig.suptitle(para + " - " + hour + ":00:00", fontdict={'size':24, 'fontweight':'bold'}, y=0.92)
    plt.savefig(dataPath + 'OD/OD_matrix_NCTCOG_AM.pdf', frameon=False, bbox_inches='tight', pad_inches=0.5) # facecolor='#F2F2F2'



# draw od flows
def drawMap_geojson_od_new(para):
    # load od flow
    ODData = np.genfromtxt(dataPath + 'OD/OD_matrix_LBS_AM_expanded.csv',
                              dtype=float, delimiter=',', skip_header=1)
    ODFlow = {}
    for row in ODData:
        oTract = int(row[0])
        dTract = int(row[1])
        flow = float(row[2])
        if oTract==dTract:
            continue
        ODFlow[(oTract, dTract)] = flow

    ODFlowCombined = {}
    for od in ODFlow.keys():
        od_r = (od[1], od[0])
        f1 = ODFlow[od]
        try:
            f2 = ODFlow[od_r]
        except:
            f2 = 0
        if od not in ODFlowCombined and od_r not in ODFlowCombined:
            ODFlowCombined[od] = f1+f2

    print("Max flow : ", np.max(ODFlowCombined.values()))
    print("# of flows : ", len(ODFlowCombined))

    totalFlow = float(np.sum(ODFlowCombined.values()))

    # distance between any two tracts
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    centroids = {}
    for t in geoData['features']:
        zipcodeId = int(t['properties']['GEOID10'])
        # centroid of this polygon
        cen_lon = t['properties']['x1']
        cen_lat = t['properties']['y1']
        centroids[zipcodeId] = (cen_lon, cen_lat)
    geojsonFile.close()

    zipcodeDistance = {}
    for oZipcode in centroids:
        for dZipcode in centroids:
            if oZipcode == dZipcode:
                continue
            oLon, oLat = centroids[oZipcode]
            dLon, dLat = centroids[dZipcode]
            dist = haversine(oLat, oLon, dLat, dLon)
            zipcodeDistance[(oZipcode, dZipcode)] = dist


    # ODFlow = ODFlowCombined

    topNum = 5000
    # top flow volume
    from collections import Counter
    top_flowVolume = []
    d = Counter(ODFlow)
    for k, v in d.most_common(topNum):
        top_flowVolume.append(k)

    # reverse the list, plot the lower flow first
    top_flowVolume = top_flowVolume[::-1]

    # load map
    # minValue, maxValue = minValues[para], maxValues[para]

    minValue, maxValue = 1000, 10000
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    coords = DallasBoundary
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coords[0], coords[2]]),
        lat_0=np.mean([coords[1], coords[3]]),
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - (extra * h),
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + (extra * h),
        resolution='i',  suppress_ticks=True)

    polygons = []
    for f in geoData['features']:
        nodes = []
        # find the max length of polygon from multiple polygons
        multiPolygons = f['geometry']['coordinates']
        maxLen = 0
        for poly in multiPolygons:
            polyLen = len(poly[0])
            if polyLen > maxLen:
                listOfNodes = poly[0]
                maxLen = polyLen

        # listOfNodes = f['geometry']['coordinates'][0][0]
        for n in listOfNodes:
            n_new = m(n[0], n[1])
            n_tup = tuple(n_new)
            nodes.append(n_tup)
        polygons.append(nodes)

    df_map = pd.DataFrame({
        # 'poly': [Polygon(feature['geometry']['coordinates'][0][0]) for feature in geoData['features']],
        'poly': [Polygon(poly) for poly in polygons],
        'name': [int(feature['properties']['GEOID10']) for feature in geoData['features']],
        'lon': [feature['properties']['x1'] for feature in geoData['features']],
        'lat': [feature['properties']['y1'] for feature in geoData['features']]
    })

    centroids = {}
    for index, row in df_map.iterrows():
        lon = float(row['lon'])
        lat = float(row['lat'])
        centroids[row['name']] = m(lon, lat)

    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))

    # numColors = numColorsIntervals[para]
    numColors = 10
    breaks = np.linspace(minValue, maxValue, numColors)
    def self_categorize(entry, breaks):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return i
            if entry > breaks[-1]:
                return len(breaks)-1
        return -1

    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    # ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    ax = fig.add_subplot(111, frame_on=False)

    if cmPalettes[para] != '':
        cmap = plt.get_cmap(cmPalettes[para])
    else:
        cmap = geoplot.YPcmap

    if cmPalettes[para] == 'RdYlBu_r':
        cmap = truncate_colormap(cmap, 0.3, 1.0)
    if cmPalettes[para] == 'hot_r':
        cmap = truncate_colormap(cmap, 0.1, 0.8)

    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    pc.set_facecolor('#333333')
    ax.add_collection(pc)

    def get_lw(flow_frac):
        if flow_frac < 0.05/100:
            return 1
        if flow_frac < 0.1/100:
            return 3
        if flow_frac < 0.25/100:
            return 6
        if flow_frac < 0.5/100:
            return 9
        if flow_frac < 1.0/100:
            return 12
        if flow_frac >= 1.0/100:
            return 15

    def get_color(flow_frac):
        if flow_frac < 0.05/100:
            return "#993404"
        if flow_frac < 0.10/100:
            return "#d95f0e"
        if flow_frac < 0.25/100:
            return "#fe9929"
        if flow_frac < 0.5/100:
            return "#fec44f"
        if flow_frac < 1.0/100:
            return "#fee391"
        if flow_frac >= 1.0/100:
            return "#ffffd4"

    def get_alpha(flow_frac):
        if flow_frac < 0.05/100:
            return 0.5
        if flow_frac < 0.1/100:
            return 0.6
        if flow_frac < 0.25/100:
            return 0.7
        if flow_frac < 0.5/100:
            return 0.8
        if flow_frac < 1.0/100:
            return 0.9
        if flow_frac >= 1.0/100:
            return 1.0

    # Plot flow
    import operator
    od_sorted = sorted(ODFlowCombined.items(), key=operator.itemgetter(1))

    print(od_sorted[:10])

    for od_ in od_sorted:
        od = od_[0]
        flow = ODFlowCombined[od]
        flow_frac = flow/totalFlow
        if flow_frac < 0.01/100:
            continue

        # assign each flow color
        colorLev = self_categorize(flow, breaks)
        # color = cmap(colorLev/float(numColors))
        color = get_color(flow_frac)
        lw = get_lw(flow_frac)
        sLon, sLat = centroids[od[0]]
        tLon, tLat = centroids[od[1]]
        alphaLev = np.power(colorLev/float(numColors), 0.55)
        # lw = np.sqrt(flow)/10.0
        ln, = plt.plot([sLon, tLon], [sLat, tLat], lw=lw, c=color, solid_capstyle='round', alpha=0.9)
        # ln.set_solid_capstyle('butt')

    # numLabels = numColorLabels[para]
    numLabels = 5
    interval = maxValue/float(numLabels)
    # jenks_labels = [str(float(breaks[i])) if np.round((breaks[i]-minValue)%interval)==0 else '' for i in range(numColors)]
    jenks_labels = []
    for i in range(numColors):
        temp = np.round((breaks[i]-minValue)%interval)
        print(temp, breaks[i], interval)
        if temp == 0:
            jenks_labels.append(int(breaks[i]))
        else:
            jenks_labels.append('')

    # Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 20.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)

    # ncolors+1 because we're using a "zero-th" color
    # cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
    # cbar.ax.tick_params(labelsize=16)

    # fig.suptitle(para + " - " + hour + ":00:00", fontdict={'size':24, 'fontweight':'bold'}, y=0.92)
    plt.savefig(dataPath + 'OD/OD_matrix_LBS_AM_expanded.pdf', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5) # facecolor='#F2F2F2'
    plt.savefig(dataPath + 'OD/OD_matrix_LBS_AM_expanded.png', dpi=300, frameon=False, bbox_inches='tight', pad_inches=0.5)


# draw flow in and out volumes to / from selected zone
def drawMap_geojson_flow(zone, para):
    # load od flow
    ODData = np.genfromtxt(dataPath + 'OD/OD_matrix_LBS_AM.csv',
                              dtype=float, delimiter=',', skip_header=1)
    # The flow in/out volume to/from the selected zone
    ODFlow = {}
    ODFlow[zone] = 0.0
    for row in ODData:
        oTract = int(row[0])
        dTract = int(row[1])
        flow = float(row[10])
        if oTract==dTract:
            continue
        if para == 'flowIn':
            key = oTract
            if dTract != zone:
                continue
        elif para == 'flowOut':
            key = dTract
            if oTract != zone:
                continue
        else:
            print("Mode error!")
            sys.exit()
        try:
            ODFlow[key] += flow
        except:
            ODFlow[key] = flow


    print("Max flow : ", np.max(ODFlow.values()))
    print("# of flows : ", len(ODFlow))

    # load map
    minValue, maxValue = minValues[para], maxValues[para]
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    coords = DallasBoundary
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coords[0], coords[2]]),
        lat_0=np.mean([coords[1], coords[3]]),
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - (extra * h),
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + (extra * h),
        resolution='i',  suppress_ticks=True)

    polygons = []
    for f in geoData['features']:
        nodes = []
        listOfNodes = f['geometry']['coordinates'][0][0]
        for n in listOfNodes:
            n_new = m(n[0], n[1])
            n_tup = tuple(n_new)
            nodes.append(n_tup)
        polygons.append(nodes)

    df_map = pd.DataFrame({
        # 'poly': [Polygon(feature['geometry']['coordinates'][0][0]) for feature in geoData['features']],
        'poly': [Polygon(poly) for poly in polygons],
        'name': [feature['properties']['GEOID10'] for feature in geoData['features']],
        'lon': [feature['properties']['x1'] for feature in geoData['features']],
        'lat': [feature['properties']['y1'] for feature in geoData['features']]
    })

    # update df_map with flow in/out volume
    df_map[para] = 0.0
    for index, row in df_map.iterrows():
        zoneId = row['name']
        try:
            flow = ODFlow[zoneId]
        except:
            flow = 0.0
        df_map[para][index] = flow

    print("min Para : ", np.min(df_map[para]))
    print("max Para : ", np.max(df_map[para]))

    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))

    numColors = numColorsIntervals[para]
    breaks = np.linspace(10*minValue, 10*maxValue, numColors)/10
    def self_categorize(entry, breaks):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return i
            if entry > breaks[-1]:
                return len(breaks)-1
        return -1

    df_map['jenks_bins'] = df_map[para].apply(self_categorize, args=(breaks,))

    numLabels = numColorLabels[para]
    interval = (maxValue-minValue)/float(numLabels)
    # jenks_labels = [str(float(breaks[i])) if np.round((breaks[i]-0)%interval)==0 else '' for i in range(numColors)]
    jenks_labels = []
    for i in range(numColors):
        temp = np.round((breaks[i]-minValue)%interval)
        # print(temp, breaks[i], interval)
        if temp == 0:
            jenks_labels.append(int(breaks[i]))
        else:
            jenks_labels.append('')


    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # cmap = plt.get_cmap('Blues')
    # cmap = geoplot.YPcmap
    # cmap = plt.get_cmap('OrRd')
    if cmPalettes[para] != '':
        cmap = plt.get_cmap(cmPalettes[para])
    else:
        cmap = geoplot.YPcmap


    # cmap.set_under('#666666')

    # cmap = truncate_colormap(cmap, 0.3, 1.0)

    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    cmap_list = [cmap(val) for val in (df_map.jenks_bins.values - 0)/(
                      float(numColors))]
    pc.set_facecolor(cmap_list)
    ax.add_collection(pc)

    #Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 10.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)

    # ncolors+1 because we're using a "zero-th" color
    cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(dataPath + 'OD/zipcodeFlow_AN.pdf', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5) # facecolor='#F2F2F2'



# draw flow in and out volumes to/from all other zones
def drawMap_geojson_flowAll(hour, para):
    hour = str(hour).zfill(2)
    # load od flow
    ODData = np.genfromtxt(dataPath + 'OD/hourlyOD/modalTrips_' + hour + '.csv',
                              dtype=float, delimiter=',', skip_header=1)
    # The flow in/out volume to/from the selected zone
    ODFlow = {}
    for row in ODData:
        oTract = int(row[0])
        dTract = int(row[1])
        flow = float(row[7])
        if oTract==dTract:
            continue
        if para == 'flowInS':
            key = dTract
        elif para == 'flowOutS':
            key = oTract
        else:
            print("Mode error!")
            sys.exit()
        try:
            ODFlow[key] += flow
        except:
            ODFlow[key] = flow


    print("Max flow : ", np.max(ODFlow.values()))
    print("# of flows : ", len(ODFlow))

    # load map
    minValue, maxValue = minValues[para], maxValues[para]
    geojsonFile = open(dataPath + 'Geo/beijing_6th_WGS84.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    coords = DallasBoundary
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coords[0], coords[2]]),
        lat_0=np.mean([coords[1], coords[3]]),
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - (extra * h),
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + (extra * h),
        resolution='i',  suppress_ticks=True)

    polygons = []
    for f in geoData['features']:
        nodes = []
        listOfNodes = f['geometry']['coordinates'][0][0]
        for n in listOfNodes:
            n_new = m(n[0], n[1])
            n_tup = tuple(n_new)
            nodes.append(n_tup)
        polygons.append(nodes)

    df_map = pd.DataFrame({
        # 'poly': [Polygon(feature['geometry']['coordinates'][0][0]) for feature in geoData['features']],
        'poly': [Polygon(poly) for poly in polygons],
        'name': [feature['properties']['new_id'] for feature in geoData['features']],
        'lon': [feature['properties']['x1'] for feature in geoData['features']],
        'lat': [feature['properties']['y1'] for feature in geoData['features']]
    })

    # update df_map with flow in/out volume
    df_map[para] = 0.0
    for index, row in df_map.iterrows():
        zoneId = row['name']
        try:
            flow = ODFlow[zoneId]
        except:
            flow = 0.0
        df_map[para][index] = flow

    print("min Para : ", np.min(df_map[para]))
    print("max Para : ", np.max(df_map[para]))

    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))

    numColors = numColorsIntervals[para]
    breaks = np.linspace(10*minValue, 10*maxValue, numColors)/10
    def self_categorize(entry, breaks):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return i
            if entry > breaks[-1]:
                return len(breaks)-1
        return -1

    df_map['jenks_bins'] = df_map[para].apply(self_categorize, args=(breaks,))

    numLabels = numColorLabels[para]
    interval = (maxValue-minValue)/float(numLabels)
    # jenks_labels = [str(float(breaks[i])) if np.round((breaks[i]-0)%interval)==0 else '' for i in range(numColors)]
    jenks_labels = []
    for i in range(numColors):
        temp = np.round((breaks[i]-minValue)%interval)
        # print(temp, breaks[i], interval)
        if temp == 0:
            jenks_labels.append(int(breaks[i]))
        else:
            jenks_labels.append('')


    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # cmap = plt.get_cmap('Blues')
    # cmap = geoplot.YPcmap
    # cmap = plt.get_cmap('OrRd')
    if cmPalettes[para] != '':
        cmap = plt.get_cmap(cmPalettes[para])
    else:
        cmap = geoplot.YPcmap

    cmap.set_under('#666666')

    # cmap = truncate_colormap(cmap, 0.3, 1.0)

    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    cmap_list = [cmap(val) for val in (df_map.jenks_bins.values - 0)/(
                      float(numColors))]
    pc.set_facecolor(cmap_list)
    ax.add_collection(pc)

    #Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 10.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)

    # ncolors+1 because we're using a "zero-th" color
    cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(dataPath + 'Air/exposure/zonePM/' + para + '_' + hour + '.png', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5) # facecolor='#F2F2F2'


# visualize the commuter flow
def drawMap_geojson_flowCommuter(hour, para):
    scaleFactor = 4.17349003564
    hour = str(hour).zfill(2)
    # load od flow
    ODData = np.genfromtxt(dataPath + 'OD/hourlyOD/modalTrips_' + hour + '.csv',
                              dtype=float, delimiter=',', skip_header=1)
    # The flow in/out volume to/from the selected zone
    ODFlow = {}
    for row in ODData:
        oTract = int(row[0])
        dTract = int(row[1])
        flow = float(row[10])*scaleFactor
        if oTract==dTract:
            continue
        if para == 'flowInS':
            key = dTract
        elif para == 'flowOutS':
            key = oTract
        else:
            print("Mode error!")
            sys.exit()
        try:
            ODFlow[key] += flow
        except:
            ODFlow[key] = flow


    print("Max flow : ", np.max(ODFlow.values()))
    print("# of flows : ", len(ODFlow))

    # load map
    minValue, maxValue = minValues[para], maxValues[para]
    geojsonFile = open(dataPath + 'Geo/beijing_6th_WGS84.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    coords = DallasBoundary
    w, h = coords[2] - coords[0], coords[3] - coords[1]
    extra = 0.01

    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([coords[0], coords[2]]),
        lat_0=np.mean([coords[1], coords[3]]),
        llcrnrlon=coords[0] - extra * w,
        llcrnrlat=coords[1] - (extra * h),
        urcrnrlon=coords[2] + extra * w,
        urcrnrlat=coords[3] + (extra * h),
        resolution='i',  suppress_ticks=True)

    polygons = []
    for f in geoData['features']:
        nodes = []
        listOfNodes = f['geometry']['coordinates'][0][0]
        for n in listOfNodes:
            n_new = m(n[0], n[1])
            n_tup = tuple(n_new)
            nodes.append(n_tup)
        polygons.append(nodes)

    df_map = pd.DataFrame({
        # 'poly': [Polygon(feature['geometry']['coordinates'][0][0]) for feature in geoData['features']],
        'poly': [Polygon(poly) for poly in polygons],
        'name': [feature['properties']['new_id'] for feature in geoData['features']],
        'lon': [feature['properties']['x1'] for feature in geoData['features']],
        'lat': [feature['properties']['y1'] for feature in geoData['features']]
    })

    # update df_map with flow in/out volume
    df_map[para] = 0.0
    for index, row in df_map.iterrows():
        zoneId = row['name']
        try:
            flow = ODFlow[zoneId]
        except:
            flow = 0.0
        if flow > 1:
            flow = np.log10(flow)
        else:
            flow = 0
        df_map[para][index] = flow

    print("min Para : ", np.min(df_map[para]))
    print("max Para : ", np.max(df_map[para]))

    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))

    numColors = numColorsIntervals[para]
    breaks = np.linspace(10*minValue, 10*maxValue, numColors)/10.0
    def self_categorize(entry, breaks):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return i
            if entry > breaks[-1]:
                return len(breaks)-1
        return -1

    df_map['jenks_bins'] = df_map[para].apply(self_categorize, args=(breaks,))

    numLabels = numColorLabels[para]
    interval = (maxValue-minValue)/float(numLabels)
    # jenks_labels = [str(float(breaks[i])) if np.round((breaks[i]-0)%interval)==0 else '' for i in range(numColors)]
    jenks_labels = []
    for i in range(numColors):
        temp = np.round((breaks[i]-minValue)%interval)
        # print(temp, breaks[i], interval)
        if temp == 0:
            jenks_labels.append(int(breaks[i]))
        else:
            jenks_labels.append('')


    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # cmap = plt.get_cmap('Blues')
    # cmap = geoplot.YPcmap
    # cmap = plt.get_cmap('OrRd')
    if cmPalettes[para] != '':
        cmap = plt.get_cmap(cmPalettes[para])
    else:
        cmap = geoplot.YPcmap

    # cmap.set_under('#666666')

    # cmap = truncate_colormap(cmap, 0.3, 1.0)

    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    cmap_list = [cmap(val) for val in (df_map.jenks_bins.values - 0)/(
                      float(numColors))]
    pc.set_facecolor(cmap_list)
    ax.add_collection(pc)

    #Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 10.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)

    # ncolors+1 because we're using a "zero-th" color
    cbar = custom_colorbar(cmap, ncolors=len(jenks_labels)+1, labels=jenks_labels, shrink=0.5)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(dataPath + 'Air/exposure/zonePM/seasonalPM/' + para + '_' + hour + '.pdf', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5) # facecolor='#F2F2F2'


# save the centroids of each zipcode area
def zipcodeCentroid():
    # get the centroids of each TSZ
    geoFile = open(dataPath + "Geo/DTW_zipcodes.geojson", 'rb')
    geoData = geojson.load(geoFile)

    for t in geoData['features']:
        multiPolygons = t['geometry']['coordinates']
        maxLen = 0
        for poly in multiPolygons:
            polyLen = len(poly[0])
            if polyLen > maxLen:
                polygon = poly[0]
                maxLen = polyLen
        # centroid of this polygon
        cen_lon = np.mean([p[0] for p in polygon])
        cen_lat = np.mean([p[1] for p in polygon])
        t['properties']['x1'] = cen_lon
        t['properties']['y1'] = cen_lat
    geoFile.close()

    # write new geojson
    with open(dataPath + 'Geo/DTW_zipcodes.geojson', 'w') as outfile:
        geojson.dump(geoData, outfile)


# scale the od matrix of LBS data with population
def expandODMatrix():
    # load population in each zipcode
    inData = open(dataPath + "census/zipcode_population/ACS_17_5YR_B01003_with_ann.csv", "r")
    inData.readline()
    inData.readline()
    zipcodePopulation = {}
    for row in inData:
        row = row.rstrip().split(',')
        zipcode = int(row[1])
        pop = int(row[3])
        zipcodePopulation[zipcode] = pop
    inData.close()

    # load the zipcode map
    # load the study regions
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/DTW_zipcodes.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # load home location of active users
    userInf = pickle.load(open(dataPath + "userData/userHomeLocationIncome.pkl", 'rb'))
    zipcodeUsers = {}
    for user in userInf:
        try:
            lon, lat, lon_work, lat_work, homeTract, income = userInf[user]
        except:
            continue

        # find zipcode
        Pt = Point(lon, lat)
        # iterate through spatial index
        zipcode = -1
        try:
            for j in idx.intersection(Pt.coords[0]):
                if Pt.within(shape(polygons[j]['geometry'])):
                    zipcode = int(polygons[j]['properties']['GEOID10'])
        except:
            continue

        if zipcode == -1:
            continue
        try:
            zipcodeUsers[zipcode] += 1
        except:
            zipcodeUsers[zipcode] = 1


    # calculate the expansion factor is there is LBS usres in these zipcodes
    expansionF_zipcode = {}
    expansionF_user = {}
    # save
    # outData = open(dataPath + "OD/expansion_zipcode.csv", "wb")
    for zipcode in zipcodeUsers:
        try:
            pop = zipcodePopulation[zipcode]
        except:
            pop = 0
        exp = pop/zipcodeUsers[zipcode]
        expansionF_zipcode[zipcode] = exp
        # outData.writelines(",".join([str(zipcode), "%.2f" % exp]) + "\n")
    # outData.close()

    print(np.percentile(expansionF_zipcode.values(), 25))
    print(np.percentile(expansionF_zipcode.values(), 50))
    print(np.percentile(expansionF_zipcode.values(), 75))

    return 0

    avgExpansionF = np.sum(zipcodePopulation.values()) / np.sum(zipcodeUsers.values())

    print("avg expansion factor : ", avgExpansionF)

    # distribution of expansion factor
    numTrips = expansionF_zipcode.values()

    print(numTrips)

    interval = 10
    bins = np.linspace(0, 1000, 21)
    usagesHist = np.histogram(np.array(numTrips), bins)
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)
    print(np.argmax(usagesHist))

    bins = np.array(bins[:-1])

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k', label='Sample users')
    plt.plot((bins + 0.5 * interval).tolist(), usagesHist.tolist(), color='#cb181d', marker='o', markersize=5,
             linewidth=1,
             markerfacecolor="#cb181d", markeredgecolor='#cb181d', markeredgewidth=1)

    plt.xlim(0, 1001)
    # plt.ylim(0, 0.05)
    # plt.xticks(range(0, 301, 3), range(0, 21, 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'Expansion factor, f', fontsize=12)
    plt.ylabel(r"P(f)", fontsize=12)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)
    # plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + 'OD/expansion_distribution.png', dpi=300)
    plt.savefig(dataPath + 'OD/expansion_distribution.pdf')
    plt.close()


    # save expansion factor for all zipcodes
    '''
    # load map of DFW
    geojsonFile = open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson', 'r')
    geoData = geojson.load(geojsonFile)

    centroids = {}
    for t in geoData['features']:
        zipcodeId = int(t['properties']['GEOID10'])
        # centroid of this polygon
        try:
            expansion = expansionF_zipcode[zipcodeId]
        except:
            expansion = avgExpansionF
        t['properties']['expansion'] = expansion
    geojsonFile.close()

    # write new geojson
    with open(dataPath + 'Geo/DTW_zipcodes_centroids.geojson', 'w') as outfile:
        geojson.dump(geoData, outfile)
    '''

    return 0



    # load od matrix
    inData = open(dataPath + "OD/OD_matrix_LBS_RD.csv", "r")
    header = inData.readline()
    outData = open(dataPath + "OD/OD_matrix_LBS_RD_expanded.csv", "w")
    outData.writelines(header.rstrip() + ",flowExp\n")
    totalFlow_before = 0
    totalFlow_after = 0
    for row in inData:
        row = row.rstrip().split(',')
        sZipcode = int(row[0])
        try:
            flow = float(row[2])*expansionF_zipcode[sZipcode]
        except:
            flow = float(row[2])*avgExpansionF
        # flow in 6 months
        flow = flow/181.0
        outData.writelines(','.join(row + ["%.2f" % flow]) + "\n")
        totalFlow_before += float(row[2])
        totalFlow_after += flow
    inData.close()
    outData.close()

    print("# total flow before expansion : ", totalFlow_before)
    print("# total flow after expansion : ", totalFlow_after)



def main():
    # # find the origin and destination TSZ for each trip
    mapTripsToTSZ()
    mapTripsToZip()

    # convert TSZ trips to Zip trips
    ODMatrix_TSZtoZip()

    ODMatrix_LBS()

    # ======== expansion ========
    expandODMatrix()

    ODcomparison()

    LBS_tripsFlow()

    # plot the zipcode flow matrix
    zipcodeCentroid()

    drawMap_geojson_od_log("flow")

    # drawMap_geojson_od_new("flow")




if __name__ == '__main__':
    main()

