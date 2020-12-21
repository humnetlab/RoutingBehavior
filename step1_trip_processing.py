# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'xu'

from joblib import Parallel, delayed
import multiprocessing

import os, sys
import operator
import pickle, csv, h5py
import copy, random
import time, datetime, pytz
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy import spatial

import networkx as nx

from scipy import optimize
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
mpl.rc('font', family='Times New Roman')
mpl.rc('text', usetex=True)

import pandas as pd
import seaborn as sns
# sns.set(style="ticks", palette="pastel", color_codes=True)
# import vaex

import matplotlib.pyplot as plt
# plt.style.use('classic')

from matplotlib.colors import Normalize
from collections import Counter

from rtree import index
from scipy import spatial
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error

import geojson
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union
import fiona
import itertools

from trajectory import Trajectory
from clustering import Clustering



# global variables
DallasTZ = pytz.timezone('US/Central')

if os.path.exists('/media/xu/Elements/Study/HuMNetLab/Data/Dallas/'):
    dataPath = '/media/xu/Elements/Study/HuMNetLab/Data/Dallas/'
if os.path.exists('/home/xu/Data/Dallas/'):
    dataPath = '/home/xu/Data/Dallas/'
if os.path.exists('/home/xu/Documents/Data/Dallas/'):
    dataPath = '/home/xu/Documents/Data/Dallas/'
if os.path.exists('/global/scratch/yanyanxu/Data/Dallas/'):
    dataPath = '/global/scratch/yanyanxu/Data/Dallas/'
if os.path.exists('/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'):
    dataPath = '/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'
if os.path.exists('/media/Data4T/Dallas/'):
    dataPath = '/media/Data4T/Dallas/'

# cityBoundary = [-97.490, -96.440, 32.630, 33.165]
# cityBoundary_forPlot = [-97.5, -96.5, 32.6, 33.2]

# cityBoundary = [-98.065, -95.855, 31.710, 33.430]
cityBoundary = [-98.0, -96.0, 32.0, 33.4]
cityBoundary_forPlot = [-98.0, -96.0, 32.0, 33.4]

'''
gridWidth = 0.005  # ~500m
numRowGrids = 343
numColGrids = 441
'''
gridWidth = 0.002  # ~200m
numRowGrids = 699
numColGrids = 999


boundaries = [(-97.22, -96.45, 32.71, 33.14),
              (-97.22, -96.45, 32.71, 33.14),
              (-97.47, -96.66, 32.64, 33.06),
              (-97.47, -96.66, 32.64, 33.06)]


# calculate distance between two locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c




def tripFilter(trip):
    '''We define the following constrains to filter the trip:
    -- 1. the diversity of directions between all continuous points (vectors). The direction
    of a trip should be uniformed, otherwise it might be dispersive.
    -- 2. the average speed between two continuous points with large displacement.  The average
    speed of a trip should be large enough (e.g., > 10 km/h), otherwise the user might stayed
    at some places for a while.
    --- 3. the duration between two continuous points is too large.'''

    overallDisplacement = haversine(trip[0][2], trip[0][1], trip[-1][2], trip[-1][1])
    overallDuration = (trip[-1][0] - trip[0][0])/3600.0  # hour
    overallSpeed = overallDisplacement/overallDuration

    displacements = []
    speeds = []
    durations = []
    for t in range(len(trip)-1):
        dist = haversine(trip[t][2], trip[t][1], trip[t+1][2], trip[t+1][1])
        duration = (trip[t+1][0] - trip[t][0])/3600.0  # hour
        speed = dist/duration
        displacements.append(dist)
        speeds.append(speed)
        durations.append(duration)

    # segment the trip by large duration (>0.5h)
    durationThres = 0.3
    displaceThres = 10
    tripSegIdx = [i+1 for i in range(len(durations)) if durations[i] > durationThres]
    tripSegIdx += [i + 1 for i in range(len(displacements)) if displacements[i] > displaceThres]
    tripSegIdx = list(set(tripSegIdx))
    tripSegIdx.sort()
    # if len(tripSegIdx) == 0:
    #     return [trip]

    subTrips = []
    tripSegIdx = [0] + tripSegIdx + [len(trip)]
    for i in range(len(tripSegIdx)-1):
        subTrip = trip[tripSegIdx[i]: tripSegIdx[i+1]]
        if len(subTrip) > 5:
            subTrips.append(subTrip)

    return subTrips



# refine trips segmentation
def tripSegmentation():
    # load the data
    sampleData = open(dataPath + "userData/allTrips_raw.csv", 'rb')

    # header: userId, tripId, statingZone, targetZone, timestamp, lat, lon
    count = 0
    numTrips = 0
    numTrips_refine = 0
    preTripId = ''
    preUserUd = ''
    numPoints = []
    trip = []

    newTripId = 0

    outDate = open(dataPath + "userData/allTrips_refine.csv", 'wb')
    for row in sampleData:
        count += 1
        row = row.rstrip().split(',')
        # print(row)
        userId = int(row[0])
        tripId = int(row[1])

        ts = int(float(row[2]))
        lon = float(row[3])
        lat = float(row[4])
        # print(userId, tripId, ts, lon, lat)

        if preTripId == '':
            preTripId = tripId
            preUserUd = userId

        if tripId != preTripId:
            numTrips += 1
            # process the last trip
            numPoints.append(len(trip))
            trip_ref = tripFilter(trip)
            for tr in trip_ref:
                numTrips_refine += 1
                # save refined trip
                for p in tr:
                    dts = int(p[0])
                    dts = datetime.datetime.fromtimestamp(dts, tz=DallasTZ)
                    ts_str = dts.strftime("%Y-%m-%d %H:%M:%S")

                    # userId, tripId, sZone, tZone, ts, lon, lat
                    row_save = [str(preUserUd), str(newTripId), ts_str, str(p[0]), str(p[1]), str(p[2])]
                    outDate.writelines(','.join(row_save) + '\n')
                # update trip id
                newTripId += 1

            trip = []
            preTripId = tripId
            preUserUd = userId

        trip.append((ts, lon, lat))

    # the last trip
    numTrips += 1
    numPoints.append(len(trip))
    trip_ref = tripFilter(trip)
    for tr in trip_ref:
        numTrips_refine += 1
        # save refined trip
        for p in tr:
            dts = int(p[0])
            dts = datetime.datetime.fromtimestamp(dts, tz=DallasTZ)
            ts_str = dts.strftime("%Y-%m-%d %H:%M:%S")

            # userId, tripId, sZone, tZone, ts, lon, lat
            row_save = [str(userId), str(newTripId), ts_str, str(p[0]), str(p[2]), str(p[1])]
            outDate.writelines(','.join(row_save) + '\n')


    sampleData.close()
    outDate.close()

    print("# of records : ", count)
    print("# of trips before refine : ", numTrips)
    print("# of trips after refine : ", numTrips_refine)


    # plot the distribution of numRecords
    interval = 2
    bins = np.linspace(0, 100, 51)
    usagesHist = np.histogram(np.array(numPoints), bins)
    bins = np.array(bins[1:])
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='data')

    plt.xlim(0, 30)
    plt.xticks(range(0, 100, 5))
    plt.xlabel(r'# of records', fontsize=12)
    plt.ylabel(r"Fraction", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numPoints_distribution.png', dpi=150)
    plt.close()



# remove trips with point outside the boundary(id==-1)
def checkNumUsers():
    tripData = open(dataPath + "userData/allTrips_selected.csv", 'r')

    count = 0
    preTripId = ''
    numTrips = 0
    allUsers = set()
    for row in tripData:
        count += 1
        if count % 1e6 == 0:
            print(count, len(allUsers), numTrips)
        row = row.rstrip().split(',')
        tripId = (row[0], row[1])  # (user, trip)
        allUsers.add(int(row[0]))

        if preTripId == '':
            preTripId = tripId

        if tripId != preTripId:
            preTripId = tripId
            numTrips += 1

    # process the last trip
    numTrips += 1

    print("# users : ", len(allUsers))
    print("# trips : ", numTrips)

def tripToVector(trip):
    X = []
    Y = []
    U = []
    V = []

    for p in range(len(trip)-1):
        currentP = trip[p]
        nextP = trip[p+1]
        u = nextP[1] - currentP[1]
        v = nextP[2] - currentP[2]

        # if np.sqrt(u**2 + v**2) < 0.002:
        #     continue

        if np.sqrt(u**2 + v**2) > 0.1:
            # print(currentP, nextP)
            continue
        X.append(currentP[1]+u)
        Y.append(currentP[2]+v)
        U.append(u)
        V.append(v)

    return X, Y, U, V




def inZone(p, idx, polygons, zoneId):
    lon, lat = p
    pt = Point(lon, lat)
    zone = -1
    # iterate through spatial index
    for j in idx.intersection(pt.coords[0]):
        if pt.within(shape(polygons[j]['geometry'])):
            zone = int(polygons[j]['properties']['zone'])
    if zoneId == zone:
        return 1
    else:
        return 0


# count # trips per user, remove users with less trips < 20
def userFilter():
    numTripsLim = 300
    '''
    # load the data
    sampleData = open(dataPath + "userData/allTrips_grid.csv", 'rb')
    # header: userId, tripId, datetime, timestamp, lon, lat, grid
    count = 0
    userTripIDs = {}

    for row in sampleData:
        count += 1
        if count%1e6 == 0:
            print(count)
        row = row.rstrip().split(',')
        # print(row)
        userId = int(row[0])
        tripId = int(row[1])
        try:
            userTripIDs[userId].add(tripId)
        except:
            userTripIDs[userId] = set([tripId])

    sampleData.close()

    print("# users in Dallas Area : ", len(userTripIDs))

    print("Remove users with less than 30 trips...")

    selectedUsers = []
    for user in userTripIDs:
        if len(userTripIDs[user]) >= numTripsLim:
            selectedUsers.append(user)

    print("# users selected %d / %d : %.2f" % (len(selectedUsers), len(userTripIDs), len(selectedUsers)/len(userTripIDs)))

    pickle.dump(selectedUsers, open(dataPath + "userData/selectedUsers_" + str(numTripsLim) + ".pkl", 'wb'),
                pickle.HIGHEST_PROTOCOL)
    pickle.dump(userTripIDs, open(dataPath + "userData/userTripIDs.pkl", 'wb'),
                pickle.HIGHEST_PROTOCOL)
    '''
    userTripIDs = pickle.load(open(dataPath + "userData/userTripIDs.pkl", 'rb'))
    userTripCount = []
    for user in userTripIDs:
        userTripCount.append(len(userTripIDs[user]))

    # plot distribution of # trips
    counts = Counter(userTripCount)
    counts_sorted = sorted(counts.items(), key=lambda item: item[1])
    totalUsers = float(np.sum([i[1] for i in counts_sorted]))
    frequency = [i[1] / totalUsers for i in counts_sorted][::-1]

    # plot number of users larger than thres
    bins = np.linspace(0, 1000, 1001)

    hist = np.histogram(userTripCount, bins)
    cdf = np.cumsum(hist[0])
    cdf = [totalUsers-i for i in cdf]
    fig = plt.figure()
    plt.plot(range(len(cdf))[::5], cdf[::5], lw=0, marker='o',
             markersize=5, markerfacecolor='w', markeredgecolor='#016450')
    plt.xlabel("Threshold of number of trips, D", fontsize=16)
    plt.ylabel("# users with more than D trips", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/userTrips_CDF.png', dpi=300)
    plt.savefig(dataPath + 'userData/userTrips_CDF.pdf')
    plt.close()


    fig = plt.figure()
    plt.plot(range(len(frequency) - 1)[::5], frequency[1:][::5], lw=0, marker='o',
             markersize=5, markerfacecolor='w', markeredgecolor='#016450')
    plt.xlabel("# trips", fontsize=16)
    plt.ylabel("Fraction (%)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/userTrips_log.png', dpi=300)
    plt.savefig(dataPath + 'userData/userTrips_log.pdf')
    plt.close()

    fig = plt.figure()
    plt.plot(range(len(frequency) - 1)[::5], frequency[1:][::5], lw=0, marker='o',
             markersize=5, markerfacecolor='w', markeredgecolor='#016450')
    plt.xlabel("# trips", fontsize=16)
    plt.ylabel("Fraction (%)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/userTrips.png', dpi=300)
    plt.savefig(dataPath + 'userData/userTrips.pdf')
    plt.close()

    selectedUsers = []
    for user in userTripIDs:
        if len(userTripIDs[user]) >= numTripsLim:
            selectedUsers.append(user)

    selectedUsers = set(selectedUsers)

    print("# selected users : ", len(selectedUsers))
    print("# users selected %d / %d : %.2f" % (
        len(selectedUsers), len(userTripIDs), len(selectedUsers) / len(userTripIDs)))
    return 0

    # load the data
    sampleData = open(dataPath + "userData/allTrips_grid.csv", 'rb')
    outData = open(dataPath + "userData/allTrips_grid_clean.csv", 'wb')
    # header: userId, tripId, datetime, timestamp, lon, lat, grid
    count = 0

    for row in sampleData:
        count += 1
        if count % 1e6 == 0:
            print(count)
        row = row.rstrip().split(',')
        # print(row)
        userId = int(row[0])
        if userId not in selectedUsers:
            continue
        outData.writelines(','.join(row) + "\n")

    sampleData.close()
    outData.close()

    print("# users selected %d / %d : %.2f" % (
        len(selectedUsers), len(userTripIDs), len(selectedUsers) / len(userTripIDs)))




# plot user origin and destinations
def userOriginDestination():
    # load the data
    '''
    sampleData = open(dataPath + "userData/allTrips_grid_clean.csv", 'rb')
    tripData = open(dataPath + "userData/allUser_tripInf.csv", "wb")
    tripData.writelines("user,tripID,sLon,sLat,tLon,tLat,traveltime\n")
    # header: userId, tripId, datetime, timestamp, lon, lat, grid
    count = 0
    preTripId = ''
    trip = []
    numTrips = 0
    removeUserTrip = set()
    for row in sampleData:
        count += 1
        if count%1e6==0:
            print(count, numTrips, len(removeUserTrip))
        row = row.rstrip().split(',')
        # print(row)
        tripId = (row[0], row[1])  # (user, trip)

        ts = int(float(row[3]))
        lon = row[4]
        lat = row[5]
        # print(userId, tripId, ts, lon, lat)

        if preTripId == '':
            preTripId = tripId

        if tripId != preTripId:
            # process the last trip
            traveltime = (trip[-1][0] - trip[0][0]) / 60.0  # min
            if traveltime < 0:
                traveltime += 24*60
            if len(trip) > 5 and traveltime > 0 and traveltime < 120:
                startX = trip[0][1]
                targetX = trip[-1][1]
                startY = trip[0][2]
                targetY = trip[-1][2]
                # save
                row_save = [preTripId[0], preTripId[1], startX, startY, targetX, targetY, "%.2f" % traveltime]
                tripData.writelines(','.join(row_save) + "\n")
                numTrips += 1
            else:
                removeUserTrip.add(preTripId)

            trip = []
            preTripId = tripId

        trip.append((ts, lon, lat))

    # the last trip
    # process the last trip
    traveltime = (trip[-1][0] - trip[0][0]) / 60.0  # min
    if traveltime < 0:
        traveltime += 24 * 60
    if len(trip) > 5 and traveltime > 0 and traveltime < 120:
        startX = trip[0][1]
        targetX = trip[-1][1]
        startY = trip[0][2]
        targetY = trip[-1][2]
        # save
        row_save = [tripId[0], tripId[1], startX, startY, targetX, targetY, "%.2f" % traveltime]
        tripData.writelines(','.join(row_save) + "\n")
        numTrips += 1
    else:
        removeUserTrip.add(tripId)

    sampleData.close()
    tripData.close()

    print("# of kept trips : ", numTrips)
    print("# of removed trips : ", len(removeUserTrip))

    pickle.dump(removeUserTrip, open(dataPath + "userData/removeUserTrip.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    '''

    '''
    removeUserTrip = pickle.load(open(dataPath + "userData/removeUserTrip.pkl", 'rb'))
    sampleData = open(dataPath + "userData/allTrips_grid_clean.csv", 'rb')
    outData = open(dataPath + "userData/allTrips_selected.csv", 'wb')

    removedIDs = {}
    for userTrip in removeUserTrip:
        user, trip = userTrip
        try:
            removedIDs[user].add(trip)
        except:
            removedIDs[user] = set(trip)

    count = 0
    for row in sampleData:
        count += 1
        if count % 1e6 == 0:
            print(count)
        row = row.rstrip().split(',')
        # print(row)
        user = row[0]
        trip = row[1]
        if user in removedIDs:
            if trip in removedIDs[user]:
                continue
            else:
                pass
        outData.writelines(','.join(row) + "\n")
    sampleData.close()
    outData.close()
    '''

    sampleData = open(dataPath + "userData/allTrips_cleanTrips.csv", 'rb')
    tripData = open(dataPath + "userData/allUser_tripInf.csv", "wb")
    tripData.writelines("user,tripID,depTime,arrTime,sLon,sLat,tLon,tLat,traveltime\n")
    # header: userId, tripId, datetime, timestamp, lon, lat, grid
    count = 0
    preTripId = ''
    trip = []
    daytimes = []
    numTrips = 0
    for row in sampleData:
        count += 1
        if count % 1e6 == 0:
            print(count, numTrips)
        row = row.rstrip().split(',')
        # print(row)
        tripId = (row[0], row[1])  # (user, trip)
        daytime = row[2]
        ts = int(float(row[3]))
        lon = row[4]
        lat = row[5]
        # print(userId, tripId, ts, lon, lat)

        if preTripId == '':
            preTripId = tripId

        if tripId != preTripId:
            # process the last trip
            traveltime = (trip[-1][0] - trip[0][0]) / 60.0  # min
            if traveltime < 0:
                traveltime += 24 * 60
            if len(trip) > 5 and traveltime > 0 and traveltime < 120:
                depTime = daytimes[0]
                arrTime = daytimes[-1]
                startX = trip[0][1]
                targetX = trip[-1][1]
                startY = trip[0][2]
                targetY = trip[-1][2]
                # save
                row_save = [preTripId[0], preTripId[1], depTime, arrTime, startX, startY, targetX, targetY, "%.2f" % traveltime]
                tripData.writelines(','.join(row_save) + "\n")
                numTrips += 1

            trip = []
            daytimes = []
            preTripId = tripId

        trip.append((ts, lon, lat))
        daytimes.append(daytime)

    # the last trip
    # process the last trip
    traveltime = (trip[-1][0] - trip[0][0]) / 60.0  # min
    if traveltime < 0:
        traveltime += 24 * 60
    if len(trip) > 5 and traveltime > 0 and traveltime < 120:
        depTime = daytimes[0]
        arrTime = daytimes[-1]
        startX = trip[0][1]
        targetX = trip[-1][1]
        startY = trip[0][2]
        targetY = trip[-1][2]
        # save
        row_save = [tripId[0], tripId[1], depTime, arrTime, startX, startY, targetX, targetY, "%.2f" % traveltime]
        tripData.writelines(','.join(row_save) + "\n")
        numTrips += 1


    sampleData.close()
    tripData.close()



def plotOneTrace(trip, tripId, zonePair):
    X = [t[1] for t in trip]
    Y = [t[2] for t in trip]
    minLon, maxLon = boundaries[zonePair][:2]
    minLat, maxLat = boundaries[zonePair][2:]

    fig = plt.figure()
    plt.plot(X, Y, color='r', marker='o', markersize=3, lw=2)
    plt.title(str(tripId), fontsize=18)
    plt.xlim(minLon, maxLon)
    plt.ylim(minLat, maxLat)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dataPath + "Traces/" + str(tripId).zfill(5) + ".png", dpi=150)
    plt.close()

def plotOneTraceInGrid(trip, tripId, zonePair, gridCentroids, cityBlocks):
    X = [t[1] for t in trip]
    Y = [t[2] for t in trip]
    minLonZone, maxLonZone = boundaries[zonePair][:2]
    minLatZone, maxLatZone = boundaries[zonePair][2:]

    X_grid = []
    Y_grid = []
    gridList = []
    # find the associated block
    for i in range(len(X)):
        lon = X[i]
        lat = Y[i]
        minLon = int(lon * 1000) // int(gridWidth * 1000) * int(gridWidth * 1000)
        minLon = round(minLon / float(1000), 3)
        maxLon = round(minLon + gridWidth, 3)

        minLat = int(lat * 1000) // int(gridWidth * 1000) * int(gridWidth * 1000)
        minLat = round(minLat / float(1000), 3)
        maxLat = round(minLat + gridWidth, 3)
        try:
            blockId = cityBlocks[(minLon, maxLon, minLat, maxLat)]
            if len(gridList)>0:
                if blockId!=gridList[-1]:
                    gridList.append(blockId)
            else:
                gridList.append(blockId)
        except:
            continue

    for grid in gridList:
        cenLon, cenLat = gridCentroids[grid]
        X_grid.append(cenLon)
        Y_grid.append(cenLat)


    fig = plt.figure()
    plt.plot(X_grid, Y_grid, color='g', marker='o', markersize=3, lw=2)
    plt.title(str(tripId), fontsize=18)
    plt.xlim(minLonZone, maxLonZone)
    plt.ylim(minLatZone, maxLatZone)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dataPath + "Traces/" + str(tripId).zfill(5) + "_grid.png", dpi=150)
    plt.close()


def plotGridTrip(gridList, tripId, zonePair, gridCentroids):
    # minLonZone, maxLonZone = boundaries[zonePair][:2]
    # minLatZone, maxLatZone = boundaries[zonePair][2:]

    X_grid = []
    Y_grid = []
    trip = []

    for grid in gridList:
        cenLon, cenLat = gridCentroids[grid]
        X_grid.append(cenLon)
        Y_grid.append(cenLat)
        trip.append((cenLon, cenLat))

    '''
    fig = plt.figure()
    plt.plot(X_grid, Y_grid, color='k', marker='o', markersize=3, lw=2)
    plt.title(str(tripId), fontsize=18)
    plt.xlim(minLonZone, maxLonZone)
    plt.ylim(minLatZone, maxLatZone)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dataPath + "Traces/" + str(tripId).zfill(5) + "_gridRefine_cutoff.png", dpi=150)
    plt.close()
    '''


    return trip



# trip: [(lon1, lat1), (lon2, lat2),...]
def subTripHighRes(trip):
    displacements = []
    durations = []
    for t in range(len(trip) - 1):
        dist = np.sqrt(np.square(trip[t][0]-trip[t+1][0]) + np.square(trip[t][1]-trip[t+1][1]))
        duration = (trip[t + 1][0] - trip[t][0]) / 3600.0  # hour
        displacements.append(dist)
        durations.append(duration)

    # segment the trip by large duration (1min)
    durationThres = 1/60.0
    displaceThres = 0.01  # ~2500m
    tripSegIdx = [i + 1 for i in range(len(durations)) if durations[i] > durationThres]
    tripSegIdx += [i + 1 for i in range(len(displacements)) if displacements[i] > displaceThres]
    tripSegIdx = list(set(tripSegIdx))
    tripSegIdx.sort()

    subTrips = []
    tripSegIdx = [0] + tripSegIdx + [len(trip)]
    for i in range(len(tripSegIdx) - 1):
        subTrip = trip[tripSegIdx[i]: tripSegIdx[i + 1]]
        if len(subTrip) > 5:
            subTrips.append(subTrip)

    return subTrips


# generate continuous grid lists in a trip
def subTripHighRes_grid(trip, cityBlocks):
    X = [t[0] for t in trip]
    Y = [t[1] for t in trip]

    gridList = []
    # find the associated block
    for i in range(len(X)):
        lon = X[i]
        lat = Y[i]
        minLon = int(lon * 1000) // int(gridWidth * 1000) * int(gridWidth * 1000)
        minLon = round(minLon / float(1000), 3)
        maxLon = round(minLon + gridWidth, 3)

        minLat = int(lat * 1000) // int(gridWidth * 1000) * int(gridWidth * 1000)
        minLat = round(minLat / float(1000), 3)
        maxLat = round(minLat + gridWidth, 3)
        try:
            blockId = cityBlocks[(minLon, maxLon, minLat, maxLat)]
            if len(gridList) > 0:
                if blockId != gridList[-1]:
                    gridList.append(blockId)
            else:
                gridList.append(blockId)
        except:
            continue

    # if the grids in list is continuous (the next grid is in the 8-neighbour of the previous one)
    tripSegIdx = []
    for g in range(len(gridList)-1):
        preG = gridList[g]
        nextG = gridList[g+1]
        neighbours = neighbourGrids(preG, w=2)
        if nextG not in neighbours:
            tripSegIdx.append(g+1)

    tripSegIdx = [0] + tripSegIdx + [len(gridList)]

    subGridLists = []
    for i in range(len(tripSegIdx) - 1):
        subTrip = gridList[tripSegIdx[i]: tripSegIdx[i + 1]]
        if len(subTrip) > 5:
            subGridLists.append(subTrip)

    return subGridLists



def neighbourGrids(gridId, w=2):
    neighbours = []
    for i in range(-w,w+1):
        for j in range(-w,w+1):
            neighbours.append(gridId+j*numRowGrids+i)
    return set(neighbours)


# convert high resolutional trip to a list of grid
def tripToGrids(trip):
    X = [t[0] for t in trip]
    Y = [t[1] for t in trip]

    gridList = []
    # find the associated block
    for i in range(len(X)):
        lon = X[i]
        lat = Y[i]

        '''
        minLon = int(lon * 1000) // int(gridWidth * 1000) * int(gridWidth * 1000)
        minLon = round(minLon / float(1000), 3)
        maxLon = round(minLon + gridWidth, 3)

        minLat = int(lat * 1000) // int(gridWidth * 1000) * int(gridWidth * 1000)
        minLat = round(minLat / float(1000), 3)
        maxLat = round(minLat + gridWidth, 3)
        
        try:
            blockId = cityBlocks[(minLon, maxLon, minLat, maxLat)]
            if len(gridList)>0:
                if blockId!=gridList[-1]:
                    gridList.append(blockId)
            else:
                gridList.append(blockId)
        except:
            continue
        '''
        colIdx = int((lon - cityBoundary[0]) / gridWidth)
        rowIdx = int((lat - cityBoundary[2]) / gridWidth)
        blockId = numRowGrids * colIdx + rowIdx
        if len(gridList) > 0:
            if blockId != gridList[-1]:
                gridList.append(blockId)
        else:
            gridList.append(blockId)

    return gridList


# we keep the original trace if can not find highRes subtrips to fillup
def fillupGrids(preG, nextG, highResTrips):
    minG = min(preG, nextG)
    maxG = max(preG, nextG)
    gridList = []
    # find the candidate trip from highResTrips
    tripCandinates = []
    for trip in highResTrips:
        if len(trip) < 3:
            continue
        if max(trip) < minG - numRowGrids - 2:
            continue
        if min(trip) > maxG + numRowGrids + 2:
            continue
        # are there nearest grids for both preG and nextG
        preFlag = 0
        nextFlag = 0
        if preG in trip and nextG in trip:
            tripCandinates.append(trip)
            preIdx = trip.index(preG)
            nextIdx = trip.index(nextG)
            gridList = trip[preIdx:nextIdx]
            return gridList

        # try w=1
        for g in range(len(trip)):
            grid = trip[g]
            neighbours = neighbourGrids(grid, w=1)
            if preG in neighbours:
                preFlag = 1
                preIdx = g
            if nextG in neighbours:
                nextFlag = 1
                nextIdx = g
        if preFlag==1 and nextFlag==1:
            gridList = trip[preIdx:nextIdx]
            return gridList

        # try w=1
        for g in range(len(trip)):
            grid = trip[g]
            neighbours = neighbourGrids(grid, w=2)
            if preG in neighbours:
                preFlag = 1
                preIdx = g
            if nextG in neighbours:
                nextFlag = 1
                nextIdx = g
        if preFlag == 1 and nextFlag == 1:
            gridList = trip[preIdx:nextIdx]
            return gridList

    return [nextG]




def completeLowResTrip(trip, highResTrips):
    # if the grids of the trip are not continuous (in 8-Neighbour)
    gridList = tripToGrids(trip)

    fullTrip = [gridList[0]]
    for g in range(len(gridList)-1):
        preG = gridList[g]
        nextG = gridList[g+1]
        neighbours = neighbourGrids(preG,w=2)
        if nextG not in neighbours:
            # try to fill up the grids
            subTrip = fillupGrids(preG, nextG, highResTrips)
            fullTrip.extend(subTrip)
        else:
            fullTrip.append(nextG)

    return fullTrip



# number of trips per user
def numTripsPerUser():
    totalRows = 728463382
    testFraction = 0.1
    numUsers = 1e5
    '''
    inFile = dataPath + "userData/allTrips_topOD.csv"
    inData = open(inFile, 'rb')

    numTrips_perUser = {}  # number of trips per user in out dataset

    preUserTrip = ""
    traj = []
    # users = set()
    count = 0
    for row in inData:
        count += 1
        if count % 1e5 == 0:
            print(count)
        row = row.rstrip().split(',')
        user = int(row[0])
        tripId = int(row[1])
        userTrip = (user, tripId)

        # users.add(user)

        # if len(users) > numUsers:
        #     break

        # lon = float(row[4])
        # lat = float(row[5])

        if preUserTrip == "":
            preUserTrip = userTrip

        # a new trip
        if preUserTrip != userTrip:
            try:
                numTrips_perUser[preUserTrip[0]] += 1
            except:
                numTrips_perUser[preUserTrip[0]] = 1
            # traj = []
            preUserTrip = userTrip

        # traj.append([lon, lat])

    inData.close()

    # save
    pickle.dump(numTrips_perUser, open(dataPath + "userData/numTrips_perUser_topOD.pkl", 'w'), pickle.HIGHEST_PROTOCOL)

    return 0
    '''
    numTrips_perUser = pickle.load(open(dataPath + "userData/numTrips_perUser_topOD.pkl", 'r'))

    # plot the distribution of number of trips
    numTrips = numTrips_perUser.values()
    interval = 5
    bins = np.linspace(10, 200, 39)
    usagesHist = np.histogram(np.array(numTrips), bins)
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)
    print(np.argmax(usagesHist))

    bins = np.array(bins[:-1])

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k', label='Sample users')
    plt.plot((bins + 0.5 * interval).tolist(), usagesHist.tolist(), color='#cb181d', marker='o', markersize=5, linewidth=1,
             markerfacecolor="None", markeredgecolor='#cb181d', markeredgewidth=1, alpha=0.7)

    plt.xlim(0, 201)
    # plt.ylim(0, 0.05)
    # plt.xticks(range(0, 301, 3), range(0, 21, 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale("log", nonposy='clip')
    plt.xlabel(r'Number of trips', fontsize=12)
    plt.ylabel(r"Fraction", fontsize=12)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)
    # plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numTrips_distribution_topOD.png', dpi=300)
    plt.savefig(dataPath + 'userData/numTrips_distribution_topOD.pdf')
    plt.close()



def findMostFrequentLoc(pointsList):
    points = np.array(pointsList)
    clustering = DBSCAN(eps=0.001, min_samples=2).fit(points)
    assignments = {}
    # assign data points to clusters
    for idx in range(len(clustering.labels_)):
        clu = clustering.labels_[idx]
        if clu == -1:
            continue
        if clu not in assignments:
            assignments[clu] = [idx]
        else:
            assignments[clu].append(idx)

    if len(assignments) == 0:
        return 0
    # update stay location
    stayLoc = {}
    for clu in assignments:
        idx = assignments[clu]
        lon_centroid, lat_centroid = np.mean(points[idx], axis=0)
        stayLoc[clu] = (lon_centroid, lat_centroid)

    sortedCluster = []
    for k in sorted(assignments, key=lambda k: len(assignments[k]), reverse=True):
        sortedCluster.append(k)

    topCluster = sortedCluster[0]

    # this location must have a large enough fraction (>40%)
    totalPoints = len(pointsList)
    fracTop = len(assignments[topCluster]) / totalPoints
    if fracTop >= 0.40:
        location = stayLoc[topCluster]
        return location
    else:
        return (0,0)


def userHomeWork(weekdayList, depTimeList, depLocationList, arrTimeList, arrLocationList):
    homePool = []
    workPool = []
    seqLen = len(weekdayList)

    for i in range(seqLen):
        weekday = weekdayList[i]
        # dep location in the morning and the arrival location in the evening
        if depTimeList[i] < 10:
            homePool.append(depLocationList[i])
        if arrTimeList[i] > 16:
            homePool.append(arrLocationList[i])

        # arrival location in the morning and the depature location in the evening on weekdays
        if arrTimeList[i] < 11 and weekday < 5:
            workPool.append(arrLocationList[i])
        if depTimeList[i] > 15 and weekday < 5:
            workPool.append(depLocationList[i])
    '''
    # the pool saves the gps locations, we must do a cluster first
    print("# locations in home pool : ", len(homePool))
    print("# locations in work pool : ", len(workPool))

    # save the home and work pool for vis
    outData = open(dataPath + "userData/sample_homePool", 'wb')
    for row in homePool:
        row_save = [str(i) for i in row]
        outData.writelines(",".join(row_save) + "\n")
    outData.close()

    outData = open(dataPath + "userData/sample_workPool", 'wb')
    for row in workPool:
        row_save = [str(i) for i in row]
        outData.writelines(",".join(row_save) + "\n")
    outData.close()
    '''

    homeLocation = findMostFrequentLoc(homePool)
    workLocation = findMostFrequentLoc(workPool)

    return homeLocation, workLocation



# find home and work location for each user
def findHomeWork():
    # load the trip data
    inData = open(dataPath + "userData/allUser_tripInf.csv", 'rb')
    # user,tripID,depTime,arrTime,sLon,sLat,tLon,tLat,traveltime
    inData.readline()

    outData = open(dataPath + "userData/userHomeWork_loc.csv", "wb")
    outData.writelines("user,homeLon,homeLat,workLon,workLat\n")

    preUser = ""
    depTimeList = []
    depLocationList = []
    arrTimeList = []
    arrLocationList = []
    weekdayList = []
    count = 0
    userWithHome = 0
    userWithWork = 0
    numUsers = 0
    for row in inData:
        count += 1
        if count%1e4==0:
            print(count, numUsers, userWithHome, userWithWork)
        row = row.rstrip().split(',')
        userID = int(row[0])
        tripID = int(row[1])
        depTime = row[2]
        depTime_dt = datetime.datetime.strptime(depTime, "%Y-%m-%d %H:%M:%S")
        arrTime = row[3]
        arrTime_dt = datetime.datetime.strptime(arrTime, "%Y-%m-%d %H:%M:%S")
        arrHour = arrTime_dt.hour
        weekday = depTime_dt.weekday()
        depHour = depTime_dt.hour
        depLon = float(row[4])
        depLat = float(row[5])
        arrLon = float(row[6])
        arrLat = float(row[7])

        currentUser = userID

        if preUser == "":
            preUser = currentUser

        if currentUser!=preUser:
            # process the last user
            homeLoc, workLoc = userHomeWork(weekdayList, depTimeList, depLocationList, arrTimeList, arrLocationList)
            # save home and work location
            row_save = [str(preUser), str(homeLoc[0]), str(homeLoc[1]), str(workLoc[0]), str(workLoc[1])]
            outData.writelines(','.join(row_save) + "\n")

            numUsers += 1
            if homeLoc[0] != 0:
                userWithHome += 1
                if workLoc[0] != 0:
                    userWithWork += 1

            weekdayList = []
            depTimeList = []
            arrTimeList = []
            depLocationList = []
            arrLocationList = []
            preUser = currentUser


        weekdayList.append(weekday)
        depTimeList.append(depHour)
        arrTimeList.append(arrHour)
        depLocationList.append([depLon, depLat])
        arrLocationList.append([arrLon, arrLat])

    # process the last user
    homeLoc, workLoc = userHomeWork(weekdayList, depTimeList, depLocationList, arrTimeList, arrLocationList)
    # save home and work location
    row_save = [str(currentUser), str(homeLoc[0]), str(homeLoc[1]), str(workLoc[0]), str(workLoc[1])]
    outData.writelines(','.join(row_save) + "\n")

    inData.close()
    outData.close()

    numUsers += 1
    if homeLoc[0] != 0:
        userWithHome += 1
        if workLoc[0] != 0:
            userWithWork += 1

    print("# users : ", numUsers)
    print("# users with home : ", userWithHome)
    print("# users with work : ", userWithWork)


def plotHomeWorkLoc():
    homeColor = "#de2d26"
    workColor = "#034e7b"

    inData = open(dataPath + "userData/userHomeWork_loc.csv", "rb")
    # user,homeLon,homeLat,workLon,workLat
    inData.readline()

    fig = plt.figure()
    for row in inData:
        row = row.rstrip().split(",")
        homeLon, homeLat, workLon, workLat = [float(i) for i in row[1:]]
        if homeLon != 0:
            # plot home
            plt.scatter(homeLon, homeLat, s=1, c=homeColor, lw=0, alpha=0.1)

    plt.axes().set_aspect('equal', 'datalim')
    plt.xlim(cityBoundary_forPlot[0], cityBoundary_forPlot[1])
    plt.ylim(cityBoundary_forPlot[2], cityBoundary_forPlot[3])
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dataPath + "userData/userHome_plot.png", dpi=150)
    plt.close()
    inData.close()

    inData = open(dataPath + "userData/userHomeWork_loc.csv", "rb")
    # user,homeLon,homeLat,workLon,workLat
    inData.readline()

    fig = plt.figure()
    for row in inData:
        row = row.rstrip().split(",")
        homeLon, homeLat, workLon, workLat = [float(i) for i in row[1:]]
        if homeLon != 0 and workLon != 0:
            # plot home
            plt.scatter(workLon, workLat, s=1, c=workColor, lw=0, alpha=0.1)

    plt.axes().set_aspect('equal', 'datalim')
    plt.xlim(cityBoundary_forPlot[0], cityBoundary_forPlot[1])
    plt.ylim(cityBoundary_forPlot[2], cityBoundary_forPlot[3])
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(dataPath + "userData/userWork_plot.png", dpi=150)
    plt.close()
    inData.close()



# 1. find the trips between the top od pair
# 2. remove low resolution and fake trips (avg speed is too low)
def selectVehTrip(userTrips):
    userTrips_update = []

    # find stay locations (origin and destination) with DBSCAN
    allPoints = []
    allTrips = {}
    trip = []
    trip_all = []
    preTrip = ''
    numTrips = 0
    for row in userTrips:
        tripId = int(row[1])
        lon = float(row[4])
        lat = float(row[5])
        if preTrip == '':
            preTrip = tripId

        if preTrip != tripId:
            allPoints.append(trip[0])  # origin
            allPoints.append(trip[-1])  # destination
            allTrips[preTrip] = trip_all
            trip = []
            trip_all = []
            preTrip = tripId
            numTrips += 1

        trip.append([lon, lat])
        trip_all.append(row)
    # last trip
    allPoints.append(trip[0])  # origin
    allPoints.append(trip[-1])  # destination
    allTrips[tripId] = trip_all
    numTrips += 1

    # assign data points to clusters
    points = np.array(allPoints)
    clustering = DBSCAN(eps=0.005, min_samples=1).fit(points)
    assignments = {}

    for idx in range(len(clustering.labels_)):
        clu = clustering.labels_[idx]
        if clu not in assignments:
            assignments[clu] = [idx]
        else:
            assignments[clu].append(idx)

    # update stay location and its grid id
    stayLoc = {}
    pointsToClu = {}
    pointsToCluCen = {}
    stayLoc_grid = {}
    for clu in assignments:
        idx = assignments[clu]
        lon_centroid, lat_centroid = np.mean(points[idx], axis=0)
        stayLoc[clu] = (lon_centroid, lat_centroid)
        colIdx = int((lon_centroid - cityBoundary[0]) / gridWidth)
        rowIdx = int((lat_centroid - cityBoundary[2]) / gridWidth)
        gridId = numRowGrids * colIdx + rowIdx
        stayLoc_grid[(lon_centroid, lat_centroid)] = gridId
        for i in idx:
            pointsToCluCen[tuple(allPoints[i])] = (lon_centroid, lat_centroid)
            pointsToClu[tuple(allPoints[i])] = clu

    # update each trip to new trip between stay locations
    numTripsOD = {}
    ODToTrips = {}
    for tripId in allTrips:
        currentTrip = allTrips[tripId]
        sLon = float(currentTrip[0][4])
        sLat = float(currentTrip[0][5])
        tLon = float(currentTrip[-1][4])
        tLat = float(currentTrip[-1][5])
        # find the class of the origin and destination points
        sClu = pointsToClu[(sLon, sLat)]
        tClu = pointsToClu[(tLon, tLat)]

        try:
            ODToTrips[(sClu, tClu)].add(tripId)
        except:
            ODToTrips[(sClu, tClu)] = set([tripId])

        try:
            numTripsOD[(sClu, tClu)] += 1
        except:
            numTripsOD[(sClu, tClu)] = 1


    if len(numTripsOD) < 1:
        return [], [(0,0), (0,0)], 0, numTrips

    sorted_od = sorted(numTripsOD.items(), key=operator.itemgetter(1))
    topOD, numT = sorted_od[-1]

    tripFraction = numT / numTrips
    significantPlaces = [stayLoc[topOD[0]], stayLoc[topOD[1]]]  # two points

    if numT < 10:
        return [], significantPlaces, 0, numTrips

    # trips between selected od
    # update user id to uid-0, uid-1
    selectedODs = [topOD, (topOD[1], topOD[0])]

    # selected trip ids
    selectedTripIds = []

    for tripId in allTrips:
        if tripId not in ODToTrips[topOD]:
            continue
        currentTrip = allTrips[tripId]
        sLon = float(currentTrip[0][4])
        sLat = float(currentTrip[0][5])
        tLon = float(currentTrip[-1][4])
        tLat = float(currentTrip[-1][5])
        # find the class of the origin and destination points
        sClu = pointsToClu[(sLon, sLat)]
        tClu = pointsToClu[(tLon, tLat)]

        sLon_cen, sLat_cen = pointsToCluCen[(sLon, sLat)]
        tLon_cen, tLat_cen = pointsToCluCen[(tLon, tLat)]

        if (sLon, sLat) != (sLon_cen, sLat_cen):
            # adding a dummy link before trip
            firstLink = copy.deepcopy(currentTrip[0])
            firstLink[4] = str(sLon_cen)
            firstLink[5] = str(sLat_cen)
            gridId = stayLoc_grid[(sLon_cen, sLat_cen)]
            firstLink[6] = str(gridId)
            currentTrip = [firstLink] + currentTrip

        if (tLon, tLat) != (tLon_cen, tLat_cen):
            # adding a dummy link after trip
            lastLink = copy.deepcopy(currentTrip[-1])
            lastLink[4] = str(tLon_cen)
            lastLink[5] = str(tLat_cen)
            gridId = stayLoc_grid[(tLon_cen, tLat_cen)]
            lastLink[6] = str(gridId)
            currentTrip = currentTrip + [lastLink]

        for row in currentTrip:
            userTrips_update.append(row)

    return userTrips_update, significantPlaces, numT, numTrips



# 1. find the trips between the second od pair
# 2. remove low resolution and fake trips (avg speed is too low)
def selectVehTrip_second(userTrips):
    userTrips_update = []

    # find stay locations (origin and destination) with DBSCAN
    allPoints = []
    allTrips = {}
    trip = []
    trip_all = []
    preTrip = ''
    numTrips = 0
    for row in userTrips:
        tripId = int(row[1])
        lon = float(row[4])
        lat = float(row[5])
        if preTrip == '':
            preTrip = tripId

        if preTrip != tripId:
            allPoints.append(trip[0])  # origin
            allPoints.append(trip[-1])  # destination
            allTrips[preTrip] = trip_all
            trip = []
            trip_all = []
            preTrip = tripId
            numTrips += 1

        trip.append([lon, lat])
        trip_all.append(row)
    # last trip
    allPoints.append(trip[0])  # origin
    allPoints.append(trip[-1])  # destination
    allTrips[tripId] = trip_all
    numTrips += 1

    # assign data points to clusters
    points = np.array(allPoints)
    clustering = DBSCAN(eps=0.005, min_samples=1).fit(points)
    assignments = {}

    for idx in range(len(clustering.labels_)):
        clu = clustering.labels_[idx]
        if clu not in assignments:
            assignments[clu] = [idx]
        else:
            assignments[clu].append(idx)

    # update stay location and its grid id
    stayLoc = {}
    pointsToClu = {}
    pointsToCluCen = {}
    stayLoc_grid = {}
    for clu in assignments:
        idx = assignments[clu]
        lon_centroid, lat_centroid = np.mean(points[idx], axis=0)
        stayLoc[clu] = (lon_centroid, lat_centroid)
        colIdx = int((lon_centroid - cityBoundary[0]) / gridWidth)
        rowIdx = int((lat_centroid - cityBoundary[2]) / gridWidth)
        gridId = numRowGrids * colIdx + rowIdx
        stayLoc_grid[(lon_centroid, lat_centroid)] = gridId
        for i in idx:
            pointsToCluCen[tuple(allPoints[i])] = (lon_centroid, lat_centroid)
            pointsToClu[tuple(allPoints[i])] = clu

    # update each trip to new trip between stay locations
    numTripsOD = {}
    ODToTrips = {}
    for tripId in allTrips:
        currentTrip = allTrips[tripId]
        sLon = float(currentTrip[0][4])
        sLat = float(currentTrip[0][5])
        tLon = float(currentTrip[-1][4])
        tLat = float(currentTrip[-1][5])
        # find the class of the origin and destination points
        sClu = pointsToClu[(sLon, sLat)]
        tClu = pointsToClu[(tLon, tLat)]

        try:
            ODToTrips[(sClu, tClu)].add(tripId)
        except:
            ODToTrips[(sClu, tClu)] = set([tripId])

        try:
            numTripsOD[(sClu, tClu)] += 1
        except:
            numTripsOD[(sClu, tClu)] = 1


    if len(numTripsOD) < 1:
        return [], [(0,0), (0,0)], 0, numTrips

    sorted_od = sorted(numTripsOD.items(), key=operator.itemgetter(1))
    try:
        topOD, numT = sorted_od[-2]  # the second top if there is one
    except:
        return [], [(0,0), (0,0)], 0, numTrips

    tripFraction = numT / numTrips
    significantPlaces = [stayLoc[topOD[0]], stayLoc[topOD[1]]]  # two points

    if numT < 10:
        return [], significantPlaces, 0, numTrips

    for tripId in allTrips:
        if tripId not in ODToTrips[topOD]:
            continue
        currentTrip = allTrips[tripId]
        sLon = float(currentTrip[0][4])
        sLat = float(currentTrip[0][5])
        tLon = float(currentTrip[-1][4])
        tLat = float(currentTrip[-1][5])
        # find the class of the origin and destination points
        sClu = pointsToClu[(sLon, sLat)]
        tClu = pointsToClu[(tLon, tLat)]

        sLon_cen, sLat_cen = pointsToCluCen[(sLon, sLat)]
        tLon_cen, tLat_cen = pointsToCluCen[(tLon, tLat)]

        if (sLon, sLat) != (sLon_cen, sLat_cen):
            # adding a dummy link before trip
            firstLink = copy.deepcopy(currentTrip[0])
            firstLink[4] = str(sLon_cen)
            firstLink[5] = str(sLat_cen)
            gridId = stayLoc_grid[(sLon_cen, sLat_cen)]
            firstLink[6] = str(gridId)
            currentTrip = [firstLink] + currentTrip

        if (tLon, tLat) != (tLon_cen, tLat_cen):
            # adding a dummy link after trip
            lastLink = copy.deepcopy(currentTrip[-1])
            lastLink[4] = str(tLon_cen)
            lastLink[5] = str(tLat_cen)
            gridId = stayLoc_grid[(tLon_cen, tLat_cen)]
            lastLink[6] = str(gridId)
            currentTrip = currentTrip + [lastLink]

        for row in currentTrip:
            userTrips_update.append(row)

    return userTrips_update, significantPlaces, numT, numTrips


# remove low resolution and fake trips (avg speed is too low)
def selectVehTrip_keepAll(userTrips):
    userTrips_update = []

    # remove trips
    removedTrips = set()
    allTrips = {}
    trip = []
    grids = []
    trip_all = []
    numTrips = 0
    preTrip = ''
    for row in userTrips:
        tripId = int(row[1])
        ts = int(row[3])
        lon = float(row[4])
        lat = float(row[5])
        gid = int(row[6])
        if preTrip == '':
            preTrip = tripId

        if preTrip != tripId:
            allTrips[preTrip] = trip_all
            if len(grids) < 5:  # num records smaller than 5
                removedTrips.add(preTrip)
            traveltime = (trip[-1][0] - trip[0][0])/3600.
            distance = haversine(trip[0][2], trip[0][1], trip[-1][2], trip[-1][1])
            if distance < 2.0:  # travel distance smaller than 2km
                removedTrips.add(preTrip)
            speed = distance / traveltime
            if speed < 10:
                # remove
                removedTrips.add(preTrip)
            trip = []
            grids = []
            trip_all = []
            preTrip = tripId

            numTrips += 1

        trip.append([ts, lon, lat])
        grids.append(gid)
        trip_all.append(row)
    # last trip
    allTrips[tripId] = trip_all
    if len(grids) < 5:  # num records smaller than 5
        removedTrips.add(tripId)
    traveltime = (trip[-1][0] - trip[0][0]) / 3600.
    distance = haversine(trip[0][2], trip[0][1], trip[-1][2], trip[-1][1])
    if distance < 2.0:  # travel distance smaller than 2km
        removedTrips.add(tripId)
    speed = distance / traveltime
    if speed < 10:
        # remove
        removedTrips.add(tripId)

    numTrips += 1

    for tripId in allTrips:
        if tripId in removedTrips:
            continue
        currentTrip = allTrips[tripId]

        for row in currentTrip:
            userTrips_update.append(row)

    return userTrips_update


# refine vehicle trips, keep only one OD pair for each user
# remove low resolutional trips, too slow trips
def filteringVehTrips():
    numTrips_perUser = pickle.load(open(dataPath + "userData/numTrips_perUser.pkl", 'rb'))
    removedUser = set()
    for user in numTrips_perUser:
        if numTrips_perUser[user] < 300:
            removedUser.add(user)

    print("# users : %d / %d " % (len(removedUser), len(numTrips_perUser)))

    # find the stay locations
    # load the data
    tripData = open(dataPath + "userData/allTrips_cleanTrips.csv", 'r')

    # header: userId, tripId, datetime, timestamp, lon, lat, grid
    outData = open(dataPath + "userData/allTrips_topOD.csv", 'w')
    outData_sigPlaces = open(dataPath + "userData/significantPlaces_top.csv", 'w')
    outData_sigPlaces.writelines("uid,sLon,sLat,tLat,tLon,numTrips,numSelected\n")

    count = 0
    numUsers = 0
    preUserId = ''
    userTrips = []
    totalTrips = 0
    totalTrips_ketp = 0

    for row in tripData:
        count += 1
        if count % 1e5 == 0:
            print(count, numUsers, totalTrips, totalTrips_ketp, "%.2f" % (totalTrips_ketp/totalTrips))

        row = row.rstrip().split(',')
        # print(row)
        userId = int(row[0])
        if userId in removedUser:
            continue
        # if numUsers == 100:
        #     break

        if preUserId == '':
            preUserId = userId

        if userId != preUserId:
            # process the last trip
            userTrips_update, significantPlaces, numT_kept, numTrips_user = selectVehTrip_second(userTrips)
            row_s1 = [str(preUserId), str(significantPlaces[0][0]), str(significantPlaces[0][1]),
                      str(significantPlaces[1][0]), str(significantPlaces[1][1]),
                      str(numTrips_user), str(numT_kept)]
            outData_sigPlaces.writelines(','.join(row_s1) + '\n')

            if numT_kept >= 10:
                for r in userTrips_update:
                    outData.writelines(','.join(r) + '\n')

            totalTrips += numTrips_user
            totalTrips_ketp += numT_kept

            numUsers += 1

            userTrips = []
            preUserId = userId

        userTrips.append(row)

    # the last trip
    userTrips_update, significantPlaces, numT_kept, numTrips_user = selectVehTrip(userTrips)
    row_s1 = [str(userId), str(significantPlaces[0][0]), str(significantPlaces[0][1]),
              str(significantPlaces[1][0]), str(significantPlaces[1][1]),
              str(numTrips_user), str(numT_kept)]
    outData_sigPlaces.writelines(','.join(row_s1) + '\n')

    if numT_kept >= 10:
        for r in userTrips_update:
            outData.writelines(','.join(r) + '\n')

    numUsers += 1

    tripData.close()
    outData.close()
    outData_sigPlaces.close()



# refine vehicle trips, remove low resolutional trips, too slow trips
def refineVehTrips_keepAllODs():
    numTrips_perUser = pickle.load(open(dataPath + "userData/numTrips_perUser.pkl", 'r'))
    removedUser = set()
    # for user in numTrips_perUser:
    #     if numTrips_perUser[user] < 300:
    #         removedUser.add(user)

    print("# users : %d / %d " % (len(removedUser), len(numTrips_perUser)))

    # find the stay locations
    # load the data
    tripData = open(dataPath + "userData/allTrips_selected_01.csv", 'r')

    # header: userId, tripId, datetime, timestamp, lon, lat, grid
    outData = open(dataPath + "userData/allTrips_cleanTrips.csv", 'w')

    count = 0
    numUsers = 0
    preUserId = ''
    userTrips = []

    for row in tripData:
        count += 1
        if count % 1e5 == 0:
            print(count, numUsers)

        row = row.rstrip().split(',')
        # print(row)
        userId = int(row[0])
        if userId in removedUser:
            continue
        # if numUsers == 100:
        #     break

        if preUserId == '':
            preUserId = userId

        if userId != preUserId:
            # process the last trip
            userTrips_update = selectVehTrip_keepAll(userTrips)

            for r in userTrips_update:
                outData.writelines(','.join(r) + '\n')

            numUsers += 1

            userTrips = []
            preUserId = userId

        userTrips.append(row)

    # the last trip
    userTrips_update = selectVehTrip_keepAll(userTrips)

    for r in userTrips_update:
        outData.writelines(','.join(r) + '\n')

    numUsers += 1

    tripData.close()
    outData.close()




def gridsToTrace(gridList, gridCentroids):
    X_grid = []
    Y_grid = []
    trip = []

    for grid in gridList:
        cenLon, cenLat = gridCentroids[grid]
        X_grid.append(cenLon)
        Y_grid.append(cenLat)
        trip.append((cenLon, cenLat))

    return trip


def ManhattanDistance(sGrid, tGrid):
    sRowId = sGrid%numRowGrids
    sColId = sGrid//numRowGrids
    tRowId = tGrid%numRowGrids
    tColId = tGrid//numRowGrids
    dist = abs(sRowId -tRowId) + abs(sColId - tColId)
    return dist




def tripSelection(userTrips):
    # find stay locations (origin and destination) with DBSCAN
    trip = []
    preTrip = ''
    tripOrigins = {}
    tripDesintaions = {}
    tripIds = set()
    removedTripIds = []

    for row in userTrips:
        tripId = int(row[1])
        tripIds.add(tripId)
        # lon = float(row[4])
        # lat = float(row[5])
        grid = int(row[6])
        if preTrip == '':
            preTrip = tripId

        if preTrip != tripId:
            tripOrigins[preTrip] = trip[0]  # origin
            tripDesintaions[preTrip] = trip[-1]  # destination
            dist = ManhattanDistance(trip[0], trip[-1])
            if dist < 10:
                removedTripIds.append(preTrip)
            trip = []
            preTrip = tripId

        trip.append(grid)
    # last trip
    tripOrigins[tripId] = trip[0]  # origin
    tripDesintaions[tripId] = trip[-1]  # destination
    dist = ManhattanDistance(trip[0], trip[-1])
    if dist < 10:
        removedTripIds.append(tripId)

    numTrips = len(tripOrigins)

    # num of trips with same origin and destination
    numTrips_sameOD = {}
    for t in tripIds:
        od = (tripOrigins[t], tripDesintaions[t])
        try:
            numTrips_sameOD[od].append(t)
        except:
            numTrips_sameOD[od] = [t]
    # removed trip
    for od in numTrips_sameOD:
        if len(numTrips_sameOD[od]) < 13:
            removedTripIds.extend(numTrips_sameOD[od])
    removedTripIds = set(removedTripIds)
    fraction_kept = 1 - len(removedTripIds)/numTrips

    userTrips_update = []
    for row in userTrips:
        tripId = int(row[1])
        if tripId in removedTripIds:
            continue
        userTrips_update.append(row)

    return userTrips_update, numTrips, fraction_kept


def tripSelection_Top1(userTrips):
    # find stay locations (origin and destination) with DBSCAN
    trip = []
    preTrip = ''
    tripOrigins = {}
    tripDesintaions = {}
    numTrips = 0
    for row in userTrips:
        tripId = int(row[1])
        # lon = float(row[4])
        # lat = float(row[5])
        grid = int(row[6])
        if preTrip == '':
            preTrip = tripId

        if preTrip != tripId:
            numTrips += 1
            dist = ManhattanDistance(trip[0], trip[-1])
            if dist >= 10:
                tripOrigins[preTrip] = trip[0]  # origin
                tripDesintaions[preTrip] = trip[-1]  # destination
            trip = []
            preTrip = tripId

        trip.append(grid)
    # last trip
    dist = ManhattanDistance(trip[0], trip[-1])
    if dist >= 10:
        tripOrigins[tripId] = trip[0]  # origin
        tripDesintaions[tripId] = trip[-1]  # destination

    numTrips +=1

    if len(tripOrigins) < 13:
        return [], 0, 0

    # num of trips with same origin and destination
    numTrips_sameOD = {}
    tripIds = set(tripOrigins.keys())
    for t in tripIds:
        od = (tripOrigins[t], tripDesintaions[t])
        try:
            numTrips_sameOD[od].append(t)
        except:
            numTrips_sameOD[od] = [t]
    odPair_times = {}
    for od in numTrips_sameOD:
        odPair_times[od] = len(numTrips_sameOD[od])
    sorted_od = sorted(odPair_times.items(), key=operator.itemgetter(1))
    topOD, numT = sorted_od[-1]
    if numT < 13:
        return [], 0, 0
    fraction_kept = len(numTrips_sameOD[topOD]) / numTrips
    selectedTripIds = set(numTrips_sameOD[topOD])

    userTrips_update = []
    for row in userTrips:
        tripId = int(row[1])
        if tripId not in selectedTripIds:
            continue
        userTrips_update.append(row)

    return userTrips_update, numTrips, fraction_kept





def routeClustering(trips, userId):
    numTrips = len(trips)

    print("# trips for clustering : ", numTrips)

    # Convert the trips to trajectories for clustering
    # list of the clusters of trajectories
    trajectories = []
    clust = Clustering()

    ci = 0  # # cluster index
    # num_cluster = -1

    # allTrips_refineRes_cutoff = allTrips_refineRes_cutoff[:20]

    for t in range(numTrips):
        trip = trips[t]
        trajectories.append(Trajectory(ci))
        for pt in trip:
            lon, lat = pt
            trajectories[len(trajectories) - 1].addPoint((lon, lat))

    # calculating the distance matrix
    distMatrix = clust.clusterSpectral(trajectories, clusters=-1, userId=userId)

    # calculate the distance matrix using DBSCAN
    clustering = DBSCAN(eps=0.003, min_samples=2, metric='precomputed').fit(distMatrix)
    assignments = {}
    # assign data points to clusters
    for idx in range(len(clustering.labels_)):
        clu = clustering.labels_[idx]
        if clu == -1:
            continue
        if clu not in assignments:
            assignments[clu] = [idx]
        else:
            assignments[clu].append(idx)

    return assignments


def distancePointToTrip(point, trip):
    dists = []
    for p in trip:
        d = haversine(point[1], point[0], p[1], p[0])
        dists.append(d)
    return np.min(dists)


def smoothOneTrace_highRes(userTrip):
    pts = []
    for row in userTrip:
        lon = row[0]
        lat = row[1]
        pts.append([lon, lat])
    pts = np.array(pts)

    points = pts.copy()
    clustering = DBSCAN(eps=0.001, min_samples=1).fit(points)
    assignments = {}
    # assign data points to clusters
    for idx in range(len(clustering.labels_)):
        clu = clustering.labels_[idx]
        if clu not in assignments:
            assignments[clu] = [idx]
        else:
            assignments[clu].append(idx)

    # update stay location
    stayLoc = {}
    for clu in assignments:
        idx = assignments[clu]
        lon_centroid, lat_centroid = np.mean(points[idx], axis=0)
        stayLoc[clu] = (lon_centroid, lat_centroid)
    # update points
    trace_new = []
    for i in range(len(pts)):
        clu = clustering.labels_[i]
        loc_new = list(stayLoc[clu])
        if i > 0:
            if loc_new != trace_new[-1]:
                trace_new.append(loc_new)
        else:
            trace_new.append(loc_new)

    pts_new = np.array(trace_new)

    # if len(pts_new) < 10:
    #     return []

    x, y = pts.T
    i = np.arange(len(pts))

    # 5x the original number of points
    interp_i = np.linspace(0, i.max(), 100)

    xi = interp1d(i, x, kind='cubic')(interp_i)
    yi = interp1d(i, y, kind='cubic')(interp_i)

    x2, y2 = pts_new.T
    i2 = np.arange(len(pts_new))

    # 5x the original number of points
    try:
        interp_i2 = np.linspace(0, i2.max(), 100)
        xi2 = interp1d(i2, x2, kind='cubic')(interp_i2)
        yi2 = interp1d(i2, y2, kind='cubic')(interp_i2)
    except:
        return userTrip

    userTrip_new = []
    for i in range(len(xi2)):
        lon = xi2[i]
        lat = yi2[i]
        userTrip_new.append([lon, lat])

    return userTrip_new


# 1. find the high-resolutional main route(s)
# 2. compare the other routes with the main route, calculate the max distance
# 3. determine if the route belongs to the main route or not.
def mainRoutesDetection(userTrips):
    allTrips = {}
    for row in userTrips:
        tid = row[1]
        lon = row[2]
        lat = row[3]
        try:
            allTrips[tid].append([lon, lat])
        except:
            allTrips[tid] = [[lon, lat]]


    tripIds = list(allTrips.keys())
    numT = len(tripIds)
    distGapThreshold = 1.0  # 1km
    distThreshold = 1.5  # max distance from point to route 1.5km


    highResTrips = []
    for tid in allTrips:
        trip = allTrips[tid]
        displacements = []
        for t in range(len(trip) - 1):
            dist = haversine(trip[t][1], trip[t][0], trip[t + 1][1], trip[t + 1][0])
            displacements.append(dist)
        maxDisp = np.max(displacements)
        if maxDisp <= distGapThreshold:
            highResTrips.append(tid)

    # no high-res trip can be found
    if len(highResTrips) == 0:
        numMainRoutes = 0
        routes = []

    # smooth highResTrips
    smoothedTrips_highRes = {}
    for tid in highResTrips:
        trip_raw = allTrips[tid]
        trip_sm = smoothOneTrace_highRes(trip_raw)
        smoothedTrips_highRes[tid] = trip_sm


    # only one main trip could be found
    if len(highResTrips) == 1:
        routes = [-1 for t in range(numT)]  # route 0 by default
        # mainRoute = allTrips[highResTrips[0]]
        mainRoute = smoothedTrips_highRes[highResTrips[0]]
        numMainRoutes = len(highResTrips)
        # if other low-res trips belong to this main route, route 0, otherwise route 1
        for t in range(numT):
            tid = tripIds[t]
            if tid in highResTrips:
                routes[t] = 0
                continue
            # min distance
            distances = []
            for p in allTrips[tid]:
                dist = distancePointToTrip(p, mainRoute)
                distances.append(dist)
            maxDistance = np.max(distances)
            if maxDistance > distThreshold:
                # belongs a another route
                routes[t] = -1
            else:
                routes[t] = 0


    # if all high res trips in one route?

    if len(highResTrips) > 1:

        # Convert the trips to trajectories for clustering
        # list of the clusters of trajectories
        trajectories = []
        clust = Clustering()

        ci = 0  # # cluster index
        # num_cluster = -1

        # allTrips_refineRes_cutoff = allTrips_refineRes_cutoff[:20]

        for tid in highResTrips:
            # trip = allTrips[tid]
            trip = smoothedTrips_highRes[tid]
            trajectories.append(Trajectory(ci))
            for pt in trip:
                lon, lat = pt
                trajectories[len(trajectories) - 1].addPoint((lon, lat))

        # calculating the distance matrix
        distMatrix = clust.clusterSpectral(trajectories, clusters=-1, userId=0)

        # calculate the distance matrix using DBSCAN
        clustering = DBSCAN(eps=0.002, min_samples=1, metric='precomputed').fit(distMatrix)
        assignments = {}
        # assign data points to clusters
        for idx in range(len(clustering.labels_)):
            clu = clustering.labels_[idx]
            if clu == -1:
                continue
            if clu not in assignments:
                assignments[clu] = [idx]
            else:
                assignments[clu].append(idx)

        numMainRoutes = len(assignments)

        # finding the routes
        routes = [-1 for t in range(numT)]  # route 0 by default
        # the main routes are 0, 1, 2, ...
        for t in range(len(highResTrips)):
            tid = highResTrips[t]
            clu = clustering.labels_[t]
            routes[tripIds.index(tid)] = clu

        # if other low-res trips belong to this main route, route 0, otherwise route 1
        for t in range(numT):
            tid = tripIds[t]
            if tid in highResTrips:
                continue
            # max distance to all main routes
            distanceToHighRes = {}
            for hightid in highResTrips:
                # mainRoute = allTrips[hightid]
                mainRoute = smoothedTrips_highRes[hightid]
                distances = []
                for p in allTrips[tid]:
                    dist = distancePointToTrip(p, mainRoute)
                    distances.append(dist)
                maxDistance = np.max(distances)
                distanceToHighRes[hightid] = maxDistance

            # if all distance are large, this low-res trip does not belong to any main routes
            mimDist = np.min(list(distanceToHighRes.values()))
            # print(mimDist)
            # print(distThreshold)
            if mimDist > distThreshold:
                # can not find matched main routes, set id as -1
                routes[t] = -1
            else:
                # find the nearest high-res trip
                for t2 in range(len(highResTrips)):
                    hightid = highResTrips[t2]
                    if distanceToHighRes[hightid] == mimDist:
                        clu = clustering.labels_[t2]
                        routes[t] = clu

    return highResTrips, numMainRoutes, routes, tripIds



# given trips in one od pair of each user, we detect the routes she used
def routesDetection(jodId=0):
    bunchSize = 2000
    userFrom = jodId*bunchSize
    userTo = (jodId+1)*bunchSize

    numTrips_perUser = pickle.load(open(dataPath + "userData/numTrips_perUser.pkl", 'rb'))
    if userFrom >= len(numTrips_perUser):
        return 0
    if userTo > len(numTrips_perUser):
        userTo = len(numTrips_perUser)

    allUsers = numTrips_perUser.keys()
    # allUsers.sort() python2
    allUsers = sorted(allUsers)
    selectedUsers = set(allUsers[userFrom:userTo])

    print(userFrom, userTo, len(selectedUsers), len(numTrips_perUser))
    minUserId = np.min(list(selectedUsers))
    maxUserId = np.max(list(selectedUsers))


    inData = open(dataPath + "userData/allTrips_topOD.csv", 'r')
    # inData = open(dataPath + "userData/allTrips_secondOD.csv", 'r')
    outData = open(dataPath + "userData/userRoutesInf_" + str(jodId) + "_sm.csv", 'w')
    outData.writelines("user,tripId,routeId\n")

    userTrips = []
    preUserId = 0
    count = 0
    numUsers = 0
    numUserWithmainRoutes = 0
    numUserWithmainRoutes_more = 0
    for row in inData:
        count += 1
        if count%1e4 == 0:
            if numUsers > 0:
                print(count, numUsers, numUserWithmainRoutes, \
                    "%.2f, %.2f" % (numUserWithmainRoutes/numUsers, numUserWithmainRoutes_more/numUsers))

        row = row.rstrip().split(',')
        userId = int(row[0])
        # if userId not in selectedUsers:
        #     continue
        if userId < minUserId:
            continue
        if userId > maxUserId:
            break

        tripId = int(row[1])
        lon = float(row[4])
        lat = float(row[5])

        if preUserId == 0:
            preUserId = userId
        if preUserId != userId:
            highResTrips, numMainRoutes, routes, tripIds = mainRoutesDetection(userTrips)
            if len(highResTrips) > 0:
                numUserWithmainRoutes += 1
            if len(highResTrips) > 1:
                numUserWithmainRoutes_more += 1
            # save user, tripId, routeId
            for t in range(len(routes)):
                row_save = [str(preUserId), str(tripIds[t]), str(routes[t])]
                outData.writelines(','.join(row_save) + '\n')

            preUserId = userId
            userTrips = []
            numUsers += 1

        userTrips.append([userId, tripId, lon, lat])

    # last user
    highResTrips, numMainRoutes, routes, tripIds = mainRoutesDetection(userTrips)
    if len(highResTrips) > 0:
        numUserWithmainRoutes += 1
    if len(highResTrips) > 1:
        numUserWithmainRoutes_more += 1

    # save user, tripId, routeId
    for t in range(len(routes)):
        row_save = [str(userId), str(tripIds[t]), str(routes[t])]
        outData.writelines(','.join(row_save) + '\n')
    numUsers += 1

    inData.close()
    outData.close()





def routesDetection_plot(jodId=0):
    numUsers = 50

    routeData = open(dataPath + "userData/userRoutesInf_" + str(jodId) + "_sm.csv", 'r')
    routeData.readline()

    selectedUsers = set()
    userRoutes = {}
    for row in routeData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        routeId = int(row[2])
        selectedUsers.add(uid)
        if len(selectedUsers) > numUsers:
            break
        userRoutes[(uid, tid)] = routeId
    routeData.close()

    maxUserId = np.max(list(selectedUsers))

    # read trip data
    inData = open(dataPath + "userData/allTrips_sample.csv", 'r')

    userTrips = []
    preUserId = 0
    count = 0
    numUsers = 0

    for row in inData:
        count += 1
        if count % 1e4 == 0:
            if numUsers > 0:
                print(count, numUsers)

        row = row.rstrip().split(',')
        userId = int(row[0])
        if userId not in selectedUsers:
            continue
        if userId > maxUserId:
            break

        tripId = int(row[1])
        lon = float(row[4])
        lat = float(row[5])

        if preUserId == 0:
            preUserId = userId
        if preUserId != userId:
            # plot the user trips
            plotRoutes(userTrips, userRoutes, preUserId)


            preUserId = userId
            userTrips = []
            numUsers += 1

        userTrips.append([userId, tripId, lon, lat])


    inData.close()




def routesDetection_parallel(jobIds):
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    results = Parallel(n_jobs=len(jobIds))(delayed(routesDetection)(i) for i in jobIds)
    return results



def routeClean(routes):
    newRoutes = []

    allR = {}
    for r in routes:
        tid = r[0]
        rid = r[1]
        try:
            allR[rid].append(tid)
        except:
            allR[rid] = [tid]
    if len(allR) == 1:
        return routes, 1

    count = 0
    for r in allR:
        numT = len(allR[r])
        if numT > 2:
            count += 1
            for tid in allR[r]:
                newRoutes.append([tid, r])

    return newRoutes, count

def routesDetection_post():

    outData = open(dataPath + "userData/userRoutesInf.csv", 'w')

    userRoutes_num = {}
    userTrips_num = {}

    for jobId in range(30):
        print(jobId)
        inData = open(dataPath + "userData/userRoutesInf_" + str(jobId) + "_sm.csv", 'r')
        header = inData.readline()
        if jobId == 0:
            outData.writelines(header)

        preUser = ""
        routes = []
        for row in inData:
            row = row.rstrip().split(',')
            uid = int(row[0])
            tid = int(row[1])
            rid = int(row[2])

            if preUser == "":
                preUser = uid
            if preUser != uid:
                newRoutes, numRoutes = routeClean(routes)
                userRoutes_num[preUser] = numRoutes
                numTrips = len(routes)
                userTrips_num[preUser] = numTrips
                for r in newRoutes:
                    row_s = [str(preUser), str(r[0]), str(r[1])]
                    outData.writelines(','.join(row_s) + "\n")
                preUser = uid
                routes = []
            routes.append([tid, rid])
        newRoutes, numRoutes = routeClean(routes)
        userRoutes_num[uid] = numRoutes
        numTrips = len(routes)
        userTrips_num[uid] = numTrips
        for r in newRoutes:
            row_s = [str(uid), str(r[0]), str(r[1])]
            outData.writelines(','.join(row_s) + "\n")
        inData.close()
    outData.close()

    # update the user information
    inData = open(dataPath + "userData/userHomeLocationIncome.csv", 'r')
    header = inData.readline()
    outData = open(dataPath + "userData/userHomeLocationIncomeRoutes.csv", 'w')
    outData.writelines(header.rstrip() + ",numTrips,numRoutes\n")
    userInf = {}
    for row in inData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        homeLon = row[1]
        homeLat = row[2]
        workLon = row[3]
        workLat = row[4]
        homeTract = row[5]
        income = row[6]
        try:
            numR = userRoutes_num[uid]
        except:
            numR = -1
        try:
            numT = userTrips_num[uid]
        except:
            numT = -1
        userInf[uid] = [homeLon, homeLat, workLon, workLat, homeTract, income, str(numT), str(numR)]
    inData.close()

    users = sorted(userInf.keys())
    for uid in users:
        row = [str(uid)] + userInf[uid]
        outData.writelines(','.join(row) + "\n")
    outData.close()

    print("# users with routes inf : ", len(userRoutes_num))


# complete the user trip information
# userId, TripId, depTime, TravelTime, Displacement, routeLable.
def completeTripInf():
    # load trip route inf
    inData = open(dataPath + "userData/userRoutesInf.csv", 'r')
    header = inData.readline()
    userRoutes = {}
    for row in inData:
        row = row.rstrip().split(',')
        uid= int(row[0])
        tid = int(row[1])
        rid = int(row[2])
        userRoutes[(uid, tid)] = rid
    inData.close()

    outData = open(dataPath + "userData/userRoutesInf_complete.csv", 'w')
    # load trip information
    # user, tripId, depTime, arrTime, sLon, sLat. tLon, tLat, traveltime
    tripData = open(dataPath + "userData/allUser_tripInf.csv", 'r')
    header = tripData.readline()
    outData.writelines(header.rstrip() + ",routeId\n")
    tripInf = {}
    count = 0
    allUsers = set()
    for row in tripData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        try:
            rid = userRoutes[(uid, tid)]
        except:
            continue
        row_save = row + [str(rid)]
        outData.writelines(','.join(row_save) + "\n")
        count += 1
        allUsers.add(uid)

    tripData.close()
    outData.close()

    print("# of users selected : ", len(allUsers))  # 22,885
    print("# of trips with route information : ", count)  # 847,615


# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.curve_fit.html

def fitfunc(p, x):
    E, delta, w = p
    y1 = np.divide(1,  x*delta*np.sqrt(2*np.pi))
    e = - np.divide(np.square(np.log(x) - E), 2*np.square(delta))
    y2 = np.exp(e)
    return w*y1*y2

def fitfunc2(p, x):
    E, delta, w = p
    y1 =  np.multiply(E/delta, np.power(np.divide(x, delta), E-1))
    e = - np.power(np.divide(x, delta), E)
    y2 = np.exp(e)
    return w*y1*y2

def errfunc(p, x, y):
    return np.abs(fitfunc(p, x) - y)

def errfunc2(p, x, y):
    return np.abs(fitfunc2(p, x) - y)



def scaleTime(tts):
    minT = np.min(tts)
    tts_ = [(t-minT)/float(minT) for t in tts]
    return tts_

def routesAnalysis():
    from scipy.stats import lognorm

    inData = open(dataPath + "userData/userRoutesInf.csv", 'r')
    header = inData.readline()
    count = 0
    count_remove = 0
    for row in inData:
        count += 1
        row = row.rstrip().split(',')
        rid = int(row[2])
        if rid == -1:
            count_remove += 1
    inData.close()

    print(count, count_remove, count_remove/float(count))


    inData = open(dataPath + "userData/userHomeLocationIncomeRoutes.csv", 'r')
    header = inData.readline()

    userCommuter = {}
    userRoutes = {}
    userIncome = {}
    userNumTrips = {}
    numRoutes = {}

    numTrips = {}
    numRoutes_list = []
    numTrips_list = []
    for row in inData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        numT = int(row[7])
        numR = int(row[8])
        income = float(row[6])
        workLon = float(row[3])
        if workLon == 0.0:
            userCommuter[uid] = 0
        else:
            userCommuter[uid] = 1

        if numT > 0:
            numTrips_list.append(numT)
            try:
                numTrips[numT] += 1
            except:
                numTrips[numT] = 1
        if numR > 0:
            numRoutes_list.append(numR)
            userRoutes[uid] = numR
            userIncome[uid] = income
            userNumTrips[uid] = numT
            try:
                numRoutes[numR] += 1
            except:
                numRoutes[numR] = 1
    inData.close()

    print("# users with home : ", len(userCommuter))

    '''
    # fit the distribution of trips and routes
    shape, loc, scale = lognorm.fit(numTrips_list, floc=0)
    # x_fit = np.linspace(min(numTrips_list), max(numTrips_list), 100)
    # numTrips_fit = lognorm.pdf(x_fit, shape, loc=loc, scale=scale)

    print("numTrips lognormal fitting, mean: %.2f, var: %.2f" % (scale, shape))


    # plot distribution of number of trips
    X = numTrips.keys()
    total = np.sum(numTrips.values())
    X = sorted(X)
    Y = []
    for x in X:
        Y.append(numTrips[x] / total)

    print(X)
    print(Y)
    print(np.max(Y))

    # fit the distribution
    X = np.array(X)
    Y = np.array(Y)
    p0 = [1., 1., 1.]  # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(X, Y))

    print(p1)


    # plot the distribution of sample rates
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(X, Y, align='center', width=1, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k')
    plt.scatter(X, Y, lw=1, marker='D', s=15, edgecolor='#034e7b', color='None')
    # plt.plot((bins + 0.5 * interval).tolist(), usagesHist.tolist(), marker='o', markersize=5, linewidth=0.5,
    #          markerfacecolor=None, markeredgecolor='#cb181d')
    # plt.plot(x_fit, numTrips_fit, lw=1, c="#034e7b")
    x_fit = np.linspace(min(X), max(X), 100)
    plt.plot(x_fit, fitfunc(p1, x_fit), color='#8533A2', lw=3, label='fit')

    E, delta, w = p1
    # ax.annotate(
    #     r'$%.2f \times \displaystyle \frac{1}{x \times %.2f \sqrt{2\pi}} \displaystyle e^{-\frac{{(\ln x - %.2f)}^2}{2 \times %.2f^2}}$' % (
    #     w, delta, E, delta),
    #     xy=(50, 0.02), fontsize=10)


    plt.xlim(10)
    plt.ylim(0, 0.025)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.yscale("log", nonposy='clip')
    plt.xlabel(r'\# trips per routine OD pair', fontsize=12)
    plt.ylabel(r"Fraction", fontsize=12)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)
    # plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/userTrips_distribution_final.png', dpi=150)
    # plt.savefig(dataPath + 'userData/userTrips_distribution_final.pdf')
    plt.close()

    # fit the distribution of routes
    shape, loc, scale = lognorm.fit(numRoutes_list, floc=0)
    x_fit = np.linspace(min(numRoutes_list), max(numRoutes_list), 100)
    numRoutes_fit = lognorm.pdf(x_fit, shape, loc=loc, scale=scale)

    print("numRoutes lognormal fitting, mean: %.2f, var: %.2f" % (scale, shape))

    # plot distribution of number of routes
    X = numRoutes.keys()
    total = np.sum(numRoutes.values())
    X = sorted(X)
    Y = []
    for x in X:
        Y.append(numRoutes[x]/total)

    print(X)
    print(Y)
    print(np.sum(Y))

    # fit the distribution
    X = np.array(X)
    Y = np.array(Y)
    p0 = [1., 1., 1.]  # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(X, Y))

    print(p1)

    # plot the distribution of sample rates
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(X, Y, align='center', width=1, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k')
    plt.scatter(X, Y, lw=1, marker='D', edgecolor='#034e7b', color='None')
    # plt.plot((bins + 0.5 * interval).tolist(), usagesHist.tolist(), marker='o', markersize=5, linewidth=0.5,
    #          markerfacecolor=None, markeredgecolor='#cb181d')
    plt.plot(x_fit, numRoutes_fit, lw=2, c="#034e7b")

    E, delta, w = p1
    # ax.annotate(
    #     r'$%.2f \times \displaystyle \frac{1}{x \times %.2f \sqrt{2\pi}} \displaystyle e^{-\frac{{(\ln x - %.2f)}^2}{2 \times %.2f^2}}$' % (
    #         w, delta, E, delta),
    #     xy=(3, 0.6), fontsize=10)

    plt.xlim(0, len(numRoutes)+1)
    plt.ylim(0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.yscale("log", nonposy='clip')
    plt.xlabel(r'\# routes per routine OD pair', fontsize=12)
    plt.ylabel(r"Fraction", fontsize=12)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)
    # plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/userRoutes_distribution.png', dpi=150)
    # plt.savefig(dataPath + 'userData/userRoutes_distribution.pdf')
    plt.close()


    return 0
    '''

    # trip information
    tripData = open(dataPath + "userData/userRoutesInf_complete.csv", 'r')
    tripData.readline()

    userTraveltimes = {}

    userTravelDistances = {}
    for row in tripData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        sLon = float(row[4])
        sLat = float(row[5])
        tLon = float(row[6])
        tLat = float(row[7])
        distance = haversine(sLat, sLon, tLat, tLon)
        tt = float(row[8])  # min
        try:
            userTravelDistances[uid].append(distance)
        except:
            userTravelDistances[uid] = [distance]
        try:
            userTraveltimes[uid].append(tt)
        except:
            userTraveltimes[uid] = [tt]

    distances_list = []
    numRoutes_list = []
    distances_bin = range(13)
    numRoutes_bin = [[] for i in range(13)]
    res = []
    userGroup_distance = {0:[], 1:[], 2:[], 3:[]}  # <5, 5-10, 10-15, >15

    for uid in userTravelDistances:
        avgDistance = np.mean(userTravelDistances[uid])
        userTravelDistances[uid] = avgDistance
    distanceThres_25 = np.percentile(userTravelDistances.values(), 25)
    distanceThres_50 = np.percentile(userTravelDistances.values(), 50)
    distanceThres_75 = np.percentile(userTravelDistances.values(), 75)

    print("Distance thres : ", distanceThres_25, distanceThres_50, distanceThres_75)

    for uid in userTravelDistances:
        # avgDistance = np.mean(userTravelDistances[uid])
        # userTravelDistances[uid] = avgDistance
        avgDistance = userTravelDistances[uid]
        try:
            numR = userRoutes[uid]
            idx = int(np.floor(avgDistance))//5
            numRoutes_bin[idx].append(numR)
            distances_list.append(avgDistance)
            numRoutes_list.append(numR)
            res.append([idx*5 + 2.5, numR])
            if avgDistance < distanceThres_25:
                userGroup_distance[0].append(numR)
            elif avgDistance < distanceThres_50:
                userGroup_distance[1].append(numR)
            elif avgDistance < distanceThres_75:
                userGroup_distance[2].append(numR)
            else:
                userGroup_distance[3].append(numR)

        except:
            continue
    print("# of users : ", len(numRoutes_list))
    print("# users in the 4 groups by distance : ", [len(userGroup_distance[0]), len(userGroup_distance[1]),
                                                     len(userGroup_distance[2]), len(userGroup_distance[3])])

    '''
    # update the avgDistance to user information file
    inData = open(dataPath + "userData/userHomeLocationIncomeRoutes.csv", 'r')
    header = inData.readline()
    allData = []
    for row in inData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        try:
            dist = userTravelDistances[uid]
        except:
            dist = -1
        row_save = row + ["%.2f" % dist]
        allData.append(row_save)
    inData.close()

    outData = open(dataPath + "userData/userHomeLocationIncomeRoutes.csv", 'w')
    outData.writelines(header.rstrip() + ",distance\n")
    for row in allData:
        outData.writelines(','.join(row) + "\n")
    outData.close()
    '''


    print([len(i) for i in numRoutes_bin])

    # box plot
    avgNumR_list = []
    for i in range(13):
        avgNumR = np.mean(numRoutes_bin[i])
        avgNumR_list.append(avgNumR)
        # res.append([i, avgNumR])
    resDF = pd.DataFrame(res, columns=['distance', 'numRoutes'])
    # Relation between distance and num of Routes

    # spatial density
    # https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.histogram2d.html
    x = np.array(distances_list)
    y = np.array(numRoutes_list)
    '''
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # sort the points by density
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    '''
    # xedges = range(0, 30, 1)
    xedges = np.linspace(0, 60, 121)
    yedges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T

    # satter
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    # plt.scatter(x, y, c=z, marker='o', edgecolor='', s=10, lw=0, alpha=0.5)

    plt.plot([d*5 for d in distances_bin], avgNumR_list, marker='D', markersize=6, linewidth=2, markerfacecolor='#b10026', color='k')

    '''
    ax = fig.add_subplot(111, xlim = xedges[[0, -1]], ylim = yedges[[0, -1]])
    im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, H)
    ax.images.append(im)
    '''

    # plt.bar(distances_bin, avgNumR_list)
    # plt.bar(distances_bin, avgNumR_list, align='edge', width=1, linewidth=1, facecolor='#41A7D8', edgecolor='k')

    # create our boxplot which is drawn on an Axes object
    # bplot = sns.boxplot(x='distance', y='numRoutes', data=resDF, whis=[5, 80], width=0.5, palette='colorblind')
    # bplot = sns.boxplot(x='distance', y='numRoutes', data=resDF, whis=[5, 80], width=0.5, palette='colorblind',
    #                     showfliers=False)


    plt.xlabel(r'Travel displacement (km)', fontsize=14)
    plt.ylabel(r"Average \# routes", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(14), [i*5 for i in range(14)])
    # plt.ylim(1.0, 2.0)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_distances_scatter.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_distances_scatter.pdf')
    plt.close()

    # bar plot for users with different displacement
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    hist_0, _ = np.histogram(userGroup_distance[0], bins)
    hist_0 = np.divide(hist_0, float(sum(hist_0)))
    hist_1, _ = np.histogram(userGroup_distance[1], bins)
    hist_1 = np.divide(hist_1, float(sum(hist_1)))
    hist_2, _ = np.histogram(userGroup_distance[2], bins)
    hist_2 = np.divide(hist_2, float(sum(hist_2)))
    hist_3, _ = np.histogram(userGroup_distance[3], bins)
    hist_3 = np.divide(hist_3, float(sum(hist_3)))

    fig = plt.figure(figsize=(4,3))
    colors = ['#005a32', '#e31a1c', '#034e7b', '#cc4c02']
    w = 0.2
    '''
    plt.bar([i - w for i in range(6)], hist_0, align='center', width=w, color=colors[0], lw=1, edgecolor='k',
            label="Optimal")
    plt.bar(range(6), hist_1, align='center', width=w, color=colors[1], lw=1, edgecolor='k', label="Motif")
    plt.bar([i + w for i in range(6)], hist_2, align='center', width=w, color=colors[2], lw=1, edgecolor='k',
            label="Motif optimal")
    plt.bar([i + 2*w for i in range(6)], hist_2, align='center', width=w, color=colors[3], lw=1, edgecolor='k',
            label="Motif optimal")
    '''
    plt.plot(range(1, 7), hist_0, lw=1.5, c=colors[0], marker='o', markersize=5, label=r"$< Q1$")
    plt.plot(range(1, 7), hist_1, lw=1.5, c=colors[1], marker='s', markersize=5, label=r"$Q1 - Q2$")
    plt.plot(range(1, 7), hist_2, lw=1.5, c=colors[2], marker='D', markersize=5, label=r"$Q2 - Q3$")
    plt.plot(range(1, 7), hist_3, lw=1.5, c=colors[3], marker='<', markersize=5, label=r"$\geq Q3$")

    plt.xlabel(r'\# routes, N', fontsize=14)
    plt.ylabel(r"P(N)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(14), [i*5 for i in range(14)])
    # plt.ylim(1.0, 2.0)
    plt.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_distances_bar.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_distances_bar.pdf')
    plt.close()



    # ==================== income

    income_list = []
    numRoutes_list = []
    income_bin = range(10)  # k$
    numRoutes_bin = [[] for i in range(10)]
    res = []
    userGroup_income = {0:[], 1:[], 2:[]}  # low <26,093, middle 26,093-78,281, high > 78,281
    for uid in userIncome:
        income = userIncome[uid]
        try:
            numR = userRoutes[uid]
            idx = int(np.floor(income)) // 10
            idx = min(idx, 9)
            numRoutes_bin[idx].append(numR)
            income_list.append(income)
            numRoutes_list.append(numR)
            if income < 26:
                userGroup_income[0].append(numR)
            elif income < 78:
                userGroup_income[1].append(numR)
            else:
                userGroup_income[2].append(numR)


        except:
            continue
    print("# of users : ", len(numRoutes_list))

    avgNumR_list = []
    for i in range(10):
        avgNumR = np.mean(numRoutes_bin[i])
        avgNumR_list.append(avgNumR)


    # spatial density
    x = np.array(income_list)
    y = np.array(numRoutes_list)

    xedges = np.linspace(0, 100, 51)
    yedges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T

    # satter
    fig = plt.figure(figsize=(4, 3))
    # plt.scatter(x, y, c=z, marker='o', edgecolor='', s=10, lw=0, alpha=0.5)

    # plt.plot([d*5 for d in distances_bin], avgNumR_list, marker='D', markersize=6, linewidth=2, markerfacecolor='#b10026', color='k')

    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    '''
    x = np.array(income_list)
    y = np.array(numRoutes_list)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # sort the points by density
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # satter
    
    fig = plt.figure(figsize=(4, 3))
    plt.scatter(x, y, c=z, marker='o', edgecolor='', s=10, lw=0, alpha=0.5)
    '''
    plt.plot([d * 10 for d in income_bin], avgNumR_list, marker='D', markersize=6, linewidth=2,
             markerfacecolor='#b10026', color='k')

    # plt.bar(income_bin, avgNumR_list, align='edge', width=1, linewidth=1, facecolor='#41A7D8', edgecolor='k')


    plt.xlabel('Income ($k\$$)', fontsize=14)
    plt.ylabel("Average \# routes", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(11), [i * 10 for i in range(11)])
    # plt.ylim(1.5, 1.7)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_income_scatter.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_income_scatter.pdf')
    plt.close()

    # bar plot for users with different displacement
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    hist_0, _ = np.histogram(userGroup_income[0], bins)
    hist_0 = np.divide(hist_0, float(sum(hist_0)))
    hist_1, _ = np.histogram(userGroup_income[1], bins)
    hist_1 = np.divide(hist_1, float(sum(hist_1)))
    hist_2, _ = np.histogram(userGroup_income[2], bins)
    hist_2 = np.divide(hist_2, float(sum(hist_2)))

    fig = plt.figure(figsize=(4, 3))
    colors = ['#005a32', '#e31a1c', '#034e7b', '#cc4c02']
    w = 0.2
    '''
    plt.bar([i - w for i in range(6)], hist_0, align='center', width=w, color=colors[0], lw=1, edgecolor='k',
            label="Optimal")
    plt.bar(range(6), hist_1, align='center', width=w, color=colors[1], lw=1, edgecolor='k', label="Motif")
    plt.bar([i + w for i in range(6)], hist_2, align='center', width=w, color=colors[2], lw=1, edgecolor='k',
            label="Motif optimal")
    plt.bar([i + 2*w for i in range(6)], hist_2, align='center', width=w, color=colors[3], lw=1, edgecolor='k',
            label="Motif optimal")
    '''
    plt.plot(range(1, 7), hist_0, lw=1.5, c=colors[0], marker='o', markersize=5, label=r"Low income")
    plt.plot(range(1, 7), hist_1, lw=1.5, c=colors[1], marker='s', markersize=5, label=r"Middle income")
    plt.plot(range(1, 7), hist_2, lw=1.5, c=colors[2], marker='D', markersize=5, label=r"High income")

    plt.xlabel(r'\# routes, N', fontsize=14)
    plt.ylabel(r"P(N)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(14), [i*5 for i in range(14)])
    # plt.ylim(1.0, 2.0)
    plt.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_income_bar.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_income_bar.pdf')
    plt.close()



    # =================== num Trips ================

    numTrips_list = []
    numRoutes_list = []
    numTrips_bin = range(13)  # k$
    numRoutes_bin = [[] for i in range(13)]
    res = []
    userGroup_trips = {0:[], 1:[], 2:[], 3:[]}  # <20, 20-40, 40-60, >60
    numTrips_25 = np.percentile(userNumTrips.values(), 25)
    numTrips_50 = np.percentile(userNumTrips.values(), 50)
    numTrips_75 = np.percentile(userNumTrips.values(), 75)
    print("# trips thres : ", numTrips_25, numTrips_50, numTrips_75)

    print("# users : ", len(userNumTrips))
    print("# trips in total : ", np.sum(userNumTrips.values()))

    for uid in userIncome:
        numT = userNumTrips[uid]
        try:
            numR = userRoutes[uid]
            idx = int(np.floor(numT)) // 10
            idx = min(idx, 12)
            numRoutes_bin[idx].append(numR)
            numTrips_list.append(numT)
            numRoutes_list.append(numR)
            if numT < numTrips_25:
                userGroup_trips[0].append(numR)
            elif numT < numTrips_50:
                userGroup_trips[1].append(numR)
            elif numT < numTrips_75:
                userGroup_trips[2].append(numR)
            else:
                userGroup_trips[3].append(numR)


        except:
            continue
    print("# of users : ", len(numRoutes_list))

    avgNumR_list = []
    for i in range(13):
        avgNumR = np.mean(numRoutes_bin[i])
        avgNumR_list.append(avgNumR)


    # spatial density
    x = np.array(numTrips_list)
    y = np.array(numRoutes_list)

    xedges = np.linspace(0, 150, 76)
    yedges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T

    # satter
    fig = plt.figure(figsize=(4, 3))
    # plt.scatter(x, y, c=z, marker='o', edgecolor='', s=10, lw=0, alpha=0.5)

    # plt.plot([d*5 for d in distances_bin], avgNumR_list, marker='D', markersize=6, linewidth=2, markerfacecolor='#b10026', color='k')

    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    # satter
    # fig = plt.figure(figsize=(4, 3))
    # plt.scatter(x, y, marker='o', s=10, c=z, lw=0, edgecolor='', alpha=0.5)

    plt.plot([d * 10 for d in numTrips_bin], avgNumR_list, marker='D', markersize=6, linewidth=2,
             markerfacecolor='#b10026', color='k')

    # plt.bar(numTrips_bin, avgNumR_list, align='edge', width=1, linewidth=1, facecolor='#41A7D8', edgecolor='k')

    plt.xlabel(r'\# trips', fontsize=14)
    plt.ylabel(r"Average \# routes", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(13), [i * 10 for i in range(13)])
    # plt.ylim(1.0, 2.2)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_numTrips_scatter.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_numTrips_scatter.pdf')
    plt.close()

    # bar plot for users with different displacement
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    hist_0, _ = np.histogram(userGroup_trips[0], bins)
    hist_0 = np.divide(hist_0, float(sum(hist_0)))
    hist_1, _ = np.histogram(userGroup_trips[1], bins)
    hist_1 = np.divide(hist_1, float(sum(hist_1)))
    hist_2, _ = np.histogram(userGroup_trips[2], bins)
    hist_2 = np.divide(hist_2, float(sum(hist_2)))
    hist_3, _ = np.histogram(userGroup_trips[3], bins)
    hist_3 = np.divide(hist_3, float(sum(hist_3)))

    fig = plt.figure(figsize=(4, 3))
    colors = ['#005a32', '#e31a1c', '#034e7b', '#cc4c02']
    w = 0.2
    plt.plot(range(1, 7), hist_0, lw=1.5, c=colors[0], marker='o', markersize=5, label=r"$< Q1$")
    plt.plot(range(1, 7), hist_1, lw=1.5, c=colors[1], marker='s', markersize=5, label=r"$Q1-Q2$")
    plt.plot(range(1, 7), hist_2, lw=1.5, c=colors[2], marker='D', markersize=5, label=r"$Q2-Q3$")
    plt.plot(range(1, 7), hist_3, lw=1.5, c=colors[3], marker='<', markersize=5, label=r"$\geq Q3$")

    plt.xlabel(r'\# routes, N', fontsize=14)
    plt.ylabel(r"P(N)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(14), [i*5 for i in range(14)])
    # plt.ylim(1.0, 2.0)
    plt.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_numTrips_bar.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_numTrips_bar.pdf')
    plt.close()


    # =========== STD
    # STD of travel times VS. number of trips / routes
    userTraveltimesGini = {}
    timeGini_list = []
    numRoutes_list = []
    numTrips_list = []
    numRoutes_bin = range(1, 7)  #
    numTrips_bin = range(1, 13)
    numBins = 6

    timeStd_bin = [[] for i in range(numBins)]
    distances_bin = [[] for i in range(numBins)]

    timeStd_bin_c = [[] for i in range(numBins)]
    timeStd_bin_nc = [[] for i in range(numBins)]

    timestd_routes = []

    for uid in userTraveltimes:
        tts = userTraveltimes[uid]
        avgDistance = np.mean(userTravelDistances[uid])
        tts = scaleTime(tts)
        g = np.std(tts)
        # calculate the std of travel time
        userTraveltimesGini[uid] = g
        try:
            commuter = userCommuter[uid]
        except:
            continue

        try:
            numR = userRoutes[uid]
            numT = userNumTrips[uid]
            if numT < 40:
                continue
            timeGini_list.append(g)
            numRoutes_list.append(numR)
            numTrips_list.append(numT)
            if numR <= 7:
                timeStd_bin[numR-1].append(g)
                if userCommuter[uid]==0:
                    timeStd_bin_nc[numR - 1].append(g)
                else:
                    timeStd_bin_c[numR - 1].append(g)
            if numR > 3:
                numR_ = 4
            else:
                numR_ = numR
            timestd_routes.append([numR_, g, commuter])
            '''
            if 10 <= numT <= 130:
                idx = numT//10
                timeStd_bin[idx-1].append(g)
                distances_bin[idx-1].append(avgDistance)
                if userCommuter[uid]==0:
                    timeStd_bin_nc[idx - 1].append(g)
                else:
                    timeStd_bin_c[idx - 1].append(g)
            '''
        except:
            continue

    # the median, upper and lower percentage of each std
    median = []
    median_distance = []
    upper = []
    lower = []
    median_c = []
    upper_c = []
    lower_c = []
    median_nc = []
    upper_nc = []
    lower_nc = []
    for r in range(numBins):
        median.append(np.median(timeStd_bin[r]))
        upper.append(np.percentile(timeStd_bin[r], 75))
        lower.append(np.percentile(timeStd_bin[r], 25))
        median_distance.append(np.median(distances_bin[r]))

        median_c.append(np.median(timeStd_bin_c[r]))
        upper_c.append(np.percentile(timeStd_bin_c[r], 75))
        lower_c.append(np.percentile(timeStd_bin_c[r], 25))

        median_nc.append(np.median(timeStd_bin_nc[r]))
        upper_nc.append(np.percentile(timeStd_bin_nc[r], 75))
        lower_nc.append(np.percentile(timeStd_bin_nc[r], 25))

    print(upper)
    print(median)
    print(lower)

    # spatial density
    x = np.array(numRoutes_list)
    y = np.array(timeGini_list)

    yedges = np.linspace(0, 3, 31)
    xedges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T

    # satter
    fig = plt.figure(figsize=(4, 3))
    # plt.scatter(x, y, c=z, marker='o', edgecolor='', s=10, lw=0, alpha=0.5)

    # plt.plot([d*5 for d in distances_bin], avgNumR_list, marker='D', markersize=6, linewidth=2, markerfacecolor='#b10026', color='k')

    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    '''
    plt.xlabel(r'\# routes', fontsize=12)
    plt.ylabel(r"Std of travel time [min]", fontsize=12)
    # plt.xticks(range(13), [i * 10 for i in range(13)])
    plt.ylim(0.0, 3.0)
    # plt.xlim(0, 150)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/timeSTD_numRoutes_den.png', dpi=150)
    plt.savefig(dataPath + 'userData/timeSTD_numRoutes_den.pdf')
    plt.close()
    '''

    # spatial density
    x = np.array(numRoutes_list)
    y = np.array(timeGini_list)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # sort the points by density
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]


    # plot
    # fig = plt.figure(figsize=(4, 3))
    # plt.scatter(x, y, marker='o', s=10, c=z, lw=0, edgecolor='', alpha=0.5)

    err_upper = []
    err_lower = []
    err_upper_c = []
    err_lower_c = []
    err_upper_nc = []
    err_lower_nc = []
    for i in range(numBins):
        err_upper.append(upper[i]-median[i])
        err_lower.append(median[i] - lower[i])

        err_upper_c.append(upper_c[i] - median_c[i])
        err_lower_c.append(median_c[i] - lower_c[i])

        err_upper_nc.append(upper_nc[i] - median_nc[i])
        err_lower_nc.append(median_nc[i] - lower_nc[i])

    err = [err_lower, err_upper]
    err_c = [err_lower_c, err_upper_c]
    err_nc = [err_lower_nc, err_upper_nc]


    plt.plot([t - 0.1 for t in numRoutes_bin], median_c, lw=2, color='#feb24c')
    plt.plot([t + 0.1 for t in numRoutes_bin], median_nc, lw=2, color='#74c476')
    plt.errorbar([t-0.1 for t in numRoutes_bin], median_c,
                 yerr=err_c,
                 marker='o',
                 color='w',
                 ecolor='#feb24c',
                 markerfacecolor='#feb24c',
                 capsize=5,  # Here I have set capsize = 0
                 linestyle='None',
                 zorder=3)

    plt.errorbar([t +0.1 for t in numRoutes_bin], median_nc,
                 yerr=err_nc,
                 marker='o',
                 color='w',
                 ecolor='#74c476',
                 markerfacecolor='#74c476',
                 capsize=5,  # Here I have set capsize = 0
                 linestyle='None',
                 zorder=3)


    plt.xlabel(r'\# routes', fontsize=14)
    plt.ylabel(r"Std of travel time [min]", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(13), [i * 10 for i in range(13)])
    plt.ylim(0.0, 2.0)
    # plt.xlim(0, 150)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/timeSTD_numRoutes_m.png', dpi=150)
    plt.savefig(dataPath + 'userData/timeSTD_numRoutes_m.pdf')
    plt.close()

    # ==== violin plot
    '''
    df_timestd_routes = pd.DataFrame(timestd_routes, columns=['numR', 'std', 'category'])

    sns.set(style="ticks", palette="pastel", color_codes=True)

    orders = [1,2,3,4]
    fig = plt.figure(figsize=(4, 3))
    ax = sns.violinplot(x="numR", y="std", hue="category", data=df_timestd_routes, split=True,
                        linewidth=1.5, inner='quartile', order=orders, palette={0: "#74c476",
                                                              1: "#feb24c"})  # inner="quart", palette={"Estimated daily consumption": "b", "Session": "y"}
    # sns.despine(left=True)
    # Get the handles and labels. For this example it'll be 2 tuples
    # of length 4 each.
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0., frameon=False)

    plt.xticks(range(4), ['1','2','3','4 and above'])
    plt.xlabel(r"\# routes")
    plt.ylim(0, 2)
    # plt.title("Distribution of individual energy demand")
    plt.tight_layout()
    # plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    plt.savefig(dataPath + 'userData/timeSTD_numRoutes_violin.png', dpi=150)
    plt.savefig(dataPath + 'userData/timeSTD_numRoutes_violin.pdf')
    plt.close()
    '''


    # ============
    # =========== STD
    # STD of travel times VS. number of trips / routes
    userTraveltimesGini = {}
    timeGini_list = []
    numRoutes_list = []
    numTrips_list = []

    numBins = 12
    numTrips_bin = range(numBins)

    timeStd_bin = [[] for i in range(numBins)]
    distances_bin = [[] for i in range(numBins)]

    timeStd_bin_c = [[] for i in range(numBins)]
    timeStd_bin_nc = [[] for i in range(numBins)]

    timestd_trips = []
    for uid in userTraveltimes:
        tts = userTraveltimes[uid]
        avgDistance = np.mean(userTravelDistances[uid])
        tts = scaleTime(tts)
        g = np.std(tts)
        # calculate the std of travel time
        userTraveltimesGini[uid] = g
        try:
            commuter = userCommuter[uid]
        except:
            continue

        try:
            numR = userRoutes[uid]
            numT = userNumTrips[uid]
            timeGini_list.append(g)
            numRoutes_list.append(numR)
            numTrips_list.append(numT)

            if 40 <= numT < 60:
                timestd_trips.append([0, g, commuter])
            if 60 <= numT < 80:
                timestd_trips.append([1, g, commuter])
            if numT > 80:
                timestd_trips.append([2, g, commuter])


            if 40 <= numT < 100:
                idx = numT//5 - 8
                timeStd_bin[idx-1].append(g)
                distances_bin[idx-1].append(avgDistance)
                if commuter==0:
                    timeStd_bin_nc[idx - 1].append(g)
                else:
                    timeStd_bin_c[idx - 1].append(g)

        except:
            continue

    # the median, upper and lower percentage of each std
    median = []
    median_distance = []
    upper = []
    lower = []
    median_c = []
    upper_c = []
    lower_c = []
    median_nc = []
    upper_nc = []
    lower_nc = []
    for r in range(numBins):
        median.append(np.median(timeStd_bin[r]))
        upper.append(np.percentile(timeStd_bin[r], 75))
        lower.append(np.percentile(timeStd_bin[r], 25))
        median_distance.append(np.median(distances_bin[r]))

        median_c.append(np.median(timeStd_bin_c[r]))
        upper_c.append(np.percentile(timeStd_bin_c[r], 75))
        lower_c.append(np.percentile(timeStd_bin_c[r], 25))

        median_nc.append(np.median(timeStd_bin_nc[r]))
        upper_nc.append(np.percentile(timeStd_bin_nc[r], 75))
        lower_nc.append(np.percentile(timeStd_bin_nc[r], 25))

    print(upper)
    print(median)
    print(lower)

    # spatial density
    x = np.array(numTrips_list)
    y = np.array(timeGini_list)

    xedges = np.linspace(40, 101, 61)
    yedges = np.linspace(0, 3, 31)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T

    # satter
    fig = plt.figure(figsize=(4, 3))

    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    '''
    plt.xlabel(r'\# trips', fontsize=12)
    plt.ylabel(r"Std of travel time [min]", fontsize=12)
    # plt.xticks(range(13), [i * 10 for i in range(13)])
    plt.ylim(0.0, 3.0)
    # plt.xlim(0, 150)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/timeSTD_numTrips_den.png', dpi=150)
    plt.savefig(dataPath + 'userData/timeSTD_numTrips_den.pdf')
    plt.close()
    '''

    # spatial density
    x = np.array(numTrips_list)
    y = np.array(timeGini_list)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # sort the points by density
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # plot
    # fig = plt.figure(figsize=(4, 3))
    # plt.scatter(x, y, marker='o', s=10, c=z, lw=0, edgecolor='', alpha=0.5)

    err_upper = []
    err_lower = []
    err_upper_c = []
    err_lower_c = []
    err_upper_nc = []
    err_lower_nc = []
    for i in range(numBins):
        err_upper.append(upper[i] - median[i])
        err_lower.append(median[i] - lower[i])

        err_upper_c.append(upper_c[i] - median_c[i])
        err_lower_c.append(median_c[i] - lower_c[i])

        err_upper_nc.append(upper_nc[i] - median_nc[i])
        err_lower_nc.append(median_nc[i] - lower_nc[i])

    err = [err_lower, err_upper]
    err_c = [err_lower_c, err_upper_c]
    err_nc = [err_lower_nc, err_upper_nc]


    plt.plot([t*5 + 42 for t in numTrips_bin], median_c, lw=2, color='#feb24c')
    plt.plot([t*5 + 43 for t in numTrips_bin], median_nc, lw=2, color='#74c476')
    plt.errorbar([t*5 + 42 for t in numTrips_bin], median_c,
                 yerr=err_c,
                 marker='o',
                 color='w',
                 ecolor='#feb24c',
                 markerfacecolor='#feb24c',
                 capsize=5,  # Here I have set capsize = 0
                 linestyle='None',
                 zorder=3)

    plt.errorbar([t*5 + 43 for t in numTrips_bin], median_nc,
                 yerr=err_nc,
                 marker='o',
                 color='w',
                 ecolor='#74c476',
                 markerfacecolor='#74c476',
                 capsize=5,  # Here I have set capsize = 0
                 linestyle='None',
                 zorder=3)

    plt.xlabel(r'\# trips', fontsize=14)
    plt.ylabel(r"Std of travel time [min]", fontsize=14)
    # plt.xticks(range(13), [i * 10 for i in range(13)])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.0, 2.0)
    plt.xlim(40, 100)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/timeSTD_numTrips_m.png', dpi=150)
    plt.savefig(dataPath + 'userData/timeSTD_numTrips_m.pdf')
    plt.close()

    # ==== violin plot
    '''
    df_timestd_trips = pd.DataFrame(timestd_trips, columns=['numT', 'std', 'category'])

    sns.set(style="ticks", palette="pastel", color_codes=True)

    orders = range(3)
    fig = plt.figure(figsize=(4, 3))
    ax = sns.violinplot(x="numT", y="std", hue="category", data=df_timestd_trips, split=True,
                        linewidth=1.5, inner='quartile', order=orders, palette={0: "#74c476",
                                                              1: "#feb24c"})  # inner="quart", palette={"Estimated daily consumption": "b", "Session": "y"}

    # sns.despine(left=True)
    # Get the handles and labels. For this example it'll be 2 tuples
    # of length 4 each.
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0., frameon=False)

    plt.xticks(range(3), ['40-60', '60-80', '80 and above'])
    plt.xlabel(r"\# trips")
    plt.ylim(0, 2)
    # plt.title("Distribution of individual energy demand")
    plt.tight_layout()
    # plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    plt.savefig(dataPath + 'userData/timeSTD_numTrips_violin.png', dpi=150)
    plt.savefig(dataPath + 'userData/timeSTD_numTrips_violin.pdf')
    plt.close()
    '''




def accuracy(y_test, y_pred_prob):
    y_pred = []

    y_test = list(y_test)
    y_pred = list(y_pred)
    numSamples = len(y_test)

    acc = 0
    acc_one = 0
    for i in range(numSamples):
        if y_test[i] == y_pred[i]:
            acc += 1
        if y_test[i] == 1:
            acc_one += 1

    return acc/float(numSamples), acc_one/float(numSamples)



def main():

    # # trip segmentation
    tripSegmentation()

    checkNumUsers()

    refineVehTrips_keepAllODs()

    # # count how many trips per user, and remove the users with less trips < 20
    userFilter()
    userOriginDestination()

    numTripsPerUser()

    # # find home and work location for each user via their visitation time
    findHomeWork()

    plotHomeWorkLoc()

    # # select trips for each user, and remove trips having very low average speed
    filteringVehTrips()

    # # given trips in one od pair of each user, we detect the routes she used
    # routesDetection(jodId=0)

    # # parallel version of routesDetection(), the users are divided into 30 parts
    routesDetection_parallel(list(range(30)))

    # plot the routes detection results
    routesDetection_plot(jodId=0)

    # combine the results running in parallel
    routesDetection_post()

    routesAnalysis()

    # # complete the user trip information, the final datasets include userId, TripId, depTime, TravelTime, Displacement, routeLable.
    completeTripInf()




if __name__ == '__main__':
    main()
