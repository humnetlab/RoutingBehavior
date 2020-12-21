# *-* coding: utf-8 *-*
__author__ = 'xu'

import os, sys
import pickle, csv, json
import time, datetime, pytz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc
rc('font', family='Times New Roman')
rc('text', usetex=True)
import pandas as pd
import random
import math
import collections
import seaborn as sns
from sklearn.cluster import DBSCAN

from scipy.stats import gaussian_kde

import geojson
from shapely.geometry import shape, Point
import fiona
from rtree import index


# global variables
DallasTZ = pytz.timezone('US/Central')

if os.path.exists('/media/xu/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'):
    dataPath = '/media/xu/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'
if os.path.exists('/home/xu/Data/Dallas/'):
    dataPath = '/home/xu/Data/Dallas/'
if os.path.exists('/home/xu/Documents/Data/Dallas/'):
    dataPath = '/home/xu/Documents/Data/Dallas/'
if os.path.exists('/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'):
    dataPath = '/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'

lonInterval = 0.005  # ~1300m
zonePairs = [(1, 2), (2, 1), (3, 4), (4, 3)]
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



# convert trips to time series data, time=lon, value=lat
def convertTrip(trip, zonePair):
    trip_lat = []
    trip_lon = []
    boundary = boundaries[zonePair]
    minLon, maxLon = boundary[:2]

    for point in trip:
        ts, lon, lat = point
        if lon < minLon or lon > maxLon:
            continue
        lonIdx = int((lon - minLon) / lonInterval)
        trip_lat.append(lat)
        trip_lon.append(lonIdx)

    # calcualte the average latitude if two or more continuous points are in the same grid

    def findSameValue(L):
        if len(set(L)) == 1:
            segIdx = range(len(L))
            k = len(L)
            return segIdx, k
        segIdx = [0]
        firstValue = L[0]
        for k in range(1, len(L)):
            if L[k] == firstValue:
                segIdx.append(k)
            else:
                return segIdx, k

    pointsInSameLonGrid = []
    p = 0
    while p < len(trip_lon)-1:
        segIdx, k = findSameValue(trip_lon[p:])
        segIdx = [i + p for i in segIdx]
        pointsInSameLonGrid.append(segIdx)
        p += k

    if p == len(trip_lon)-1:
        pointsInSameLonGrid.append([p])

    # update trip_lon, trip_lat
    trip_lon_new = []
    trip_lat_new = []
    for seg in pointsInSameLonGrid:
        lonIdx = trip_lon[seg[0]]
        lat_value = 0
        for i in seg:
            lat_value += trip_lat[i]
        lat_value = lat_value/float(len(seg))
        trip_lon_new.append(lonIdx)
        trip_lat_new.append(lat_value)

    return trip_lon_new, trip_lat_new

# retuen [(lat, lon), ...]
def convertTripLaltLon(trip):
    trip_new = []
    for t in trip:
        trip_new.append((t[2], t[1]))
    return trip_new



# working on high resolutional data
def trajClustering_highRes(zonePair=3, num_cluster=30, model="Spectral"):
    from trajectory import Trajectory
    from clustering import Clustering

    startZone, targetZone = zonePairs[zonePair]
    print("From zone %d to %d ... " % (startZone, targetZone))

    # load the data
    data = pickle.load(open(dataPath + "results/highResTrips_" + str(zonePair) + '.pkl', 'rb'))

    allTripIds, allUserIds, rawTrips, allTrips_refineRes, allTrips_refineRes_grids, allTrips_refineRes_cutoff = data

    print("# of trips : ", len(allTripIds))

    # Convert the trips to trajectories for clustering
    # list of the clusters of trajectories
    trajectories = []
    clust = Clustering()

    ci=0  # # cluster index
    # num_cluster = -1

    # allTrips_refineRes = allTrips_refineRes[:20]

    for t in range(len(allTrips_refineRes)):
        trip = allTrips_refineRes[t]
        trajectories.append(Trajectory(ci))
        for pt in trip:
            lon, lat = pt
            trajectories[len(trajectories) - 1].addPoint((lon, lat))

    if model=='Spectral':
        clust.clusterSpectral(trajectories, clusters=num_cluster, zonePair=zonePair)
    if model=='Agglomerative':
        clust.clusterAgglomerartive(trajectories, num_cluster, zonePair=zonePair)

    assignments = {}
    # assign data points to clusters
    for t in trajectories:
        clu = t.ci
        idx = t.id
        if clu not in assignments:
            assignments[clu] = [idx]
        else:
            assignments[clu].append(idx)

    count = 0
    for c in assignments:
        count += len(assignments[c])
    print("# of trips in clustering : ", count)

    sortedCluster = []
    for k in sorted(assignments, key=lambda k: len(assignments[k]), reverse=True):
        sortedCluster.append(k)

    # plot
    base = plt.cm.get_cmap('viridis')
    color_list = base(np.linspace(0, 1, len(assignments)))

    fig = plt.figure()

    for i in range(len(sortedCluster)):
        ass = sortedCluster[i]
        if ass == None:
            continue
        cluster = assignments[ass]
        for c in cluster:
            lonlat = rawTrips[c]
            # print(latlon)
            lonList = [j[0] for j in lonlat]
            latList = [j[1] for j in lonlat]
            plt.scatter(lonList, latList, color=color_list[i], s=5)

        # plt.plot(i, zorder=2, lw=3)

    # convert lonIdx back to longitude
    minLon, maxLon = boundaries[zonePair][:2]
    minLat, maxLat = boundaries[zonePair][2:]

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clustering method: " + model)

    plt.xlim(minLon, maxLon)
    plt.ylim(minLat, maxLat)

    plt.tight_layout()
    # plt.show()
    plt.savefig(dataPath + 'results/cluster_trajectory_highRes_' + str(startZone) + '_'
                + str(targetZone) + '_' + model + '.png', dpi=300)

    plt.close()

    # =======================================
    # plot the cluster seperately
    # =======================================
    for i in range(len(sortedCluster)):
        ass = sortedCluster[i]
        if ass == None:
            continue
        cluster = assignments[ass]

        tripsInCluster = []
        usersInCluster = []
        for c in cluster:
            tripsInCluster.append(allTripIds[c])
            usersInCluster.append(allUserIds[c])

        print("# of trips in cluser %d : %d" % (i, len(tripsInCluster)))
        print("# of uses in cluser %d : %d" % (i, len(set(usersInCluster))))

        fig = plt.figure()
        ax = plt.subplot()

        for c in cluster:
            # lonlat = rawTrips[c]
            lonlat = allTrips_refineRes[c]
            # print(latlon)
            lonList = [j[0] for j in lonlat]
            latList = [j[1] for j in lonlat]
            plt.scatter(lonList, latList, color=color_list[i], s=5)

        # plt.plot(i, zorder=2, lw=3)

        # convert lonIdx back to longitude
        minLon, maxLon = boundaries[zonePair][:2]
        minLat, maxLat = boundaries[zonePair][2:]
        # lons = list(np.linspace(minLon, maxLon, 1 + (maxLon - minLon) / lonInterval))
        # lons = ["%.2f" % l for l in lons]
        # plt.xticks(range(len(lons))[::20], lons[::20])

        ax.annotate("# trips: %d \n# users: %d" % (len(tripsInCluster), len(set(usersInCluster))),
                    xy=(.16, .75), xycoords='figure fraction', fontsize=20)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Clustering: " + str(i))

        plt.xlim(minLon, maxLon)
        plt.ylim(minLat, maxLat)

        plt.tight_layout()
        # plt.show()
        plt.savefig(dataPath + 'results/cluster_trajectory_highRes_' + str(startZone) + '_'
                    + str(targetZone) + '_' + model + '_' + str(i) + '.png', dpi=300)

        plt.close()

    # save assignment results
    pickle.dump(assignments, open(dataPath + "results/trajClustering_highRes_" + model + '_' + str(zonePair) + ".pkl", 'wb'),
                pickle.HIGHEST_PROTOCOL)



# louvain clustering using the similarity matrix
def louvainClustering(zonePair=3):
    startZone, targetZone = zonePairs[zonePair]
    print("From zone %d to %d ... " % (startZone, targetZone))

    # load the data
    data = pickle.load(open(dataPath + "results/highResTrips_" + str(zonePair) + '.pkl', 'rb'))

    allTripIds, allUserIds, rawTrips, allTrips_refineRes, allTrips_refineRes_grids, allTrips_refineRes_cutoff = data

    print("# of trips : ", len(allTripIds))

    # load the similarity matrix
    similarityMat = pickle.load(open("similarityMat_" + str(zonePair) + "_highRes.pkl", 'rb'))

    # build the network
    numNodes = len(allTripIds)
    nodeIds = range(numNodes)




def findRegions(trip, idx, polygons):
    regions = []
    for p in trip:
        lon, lat = p
        pt = Point(lon, lat)
        # region = '0'
        # iterate through spatial index
        for j in idx.intersection(pt.coords[0]):
            if pt.within(shape(polygons[j]['geometry'])):
                region = str(polygons[j]['properties']['regionId'])
                regions.append(region)

    try:
        regions_correct = [regions[0]]
    except:
        return []
    for i in range(len(regions)-1):
        if regions[i+1]==regions[i]:
            continue
        regions_correct.append(regions[i+1])

    print(regions_correct)
    return regions_correct


# define the routes with the traversed major regions
def userRoutingBehavior(zonePair=3):
    # load trips
    startZone, targetZone = zonePairs[zonePair]
    print("From zone %d to %d ... " % (startZone, targetZone))

    # load the data
    data = pickle.load(open(dataPath + "Trips_all/highResTrips_" + str(zonePair) + '.pkl', 'rb'))

    allTripIds, allUserIds, rawTrips, allTrips_refineRes, allTrips_refineRes_grids, allTrips_refineRes_cutoff = data


    # load the assignments results
    # assignments = pickle.dump(open(dataPath + "Trips_all/trajClustering_cutoff_" + model + '_' + str(zonePair) + ".pkl", 'rb'))

    sampleData = open(dataPath + "Trips_all/allTrips_grid.csv", 'rb')
    depTimes = {}
    for row in sampleData:
        row = row.rstrip().split(',')
        tripId = int(row[1])
        depHour = int(row[4].split(' ')[1][:2])
        depTimes[tripId] = depHour
    sampleData.close()

    # load the map of major regions
    # load the zone map
    if zonePair in [2, 3]:
        polygons = [pol for pol in fiona.open(dataPath + 'Geo/majorRegions/NTE_regions.geojson')]
    if zonePair in [0, 1]:
        polygons = [pol for pol in fiona.open(dataPath + 'Geo/majorRegions/LBJ_regions.geojson')]

    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # define the routes by time
    # 7–10 am (AM), 10 am–4 pm (MD), 4–7 pm (PM), and the rest of the day (RD)
    routesAM = {}
    routesMD = {}
    routesPM = {}
    routesRD = {}
    if zonePair in [2,3]:
        rowNames = ["A", "B", "C", "D", "E"]
        colNames = ["X", "Y", "Z"]
    if zonePair in [0, 1]:
        rowNames = ["A", "B", "C", "D"]
        colNames = ["W", "X", "Y", "Z"]

    routePaires = set()
    for i in rowNames:
        for j in colNames:
            routesAM[(i, j)] = 0
            routesMD[(i, j)] = 0
            routesPM[(i, j)] = 0
            routesRD[(i, j)] = 0
            routePaires.add((i, j))

    # for each high-res trip, we find the traversed major regions
    routeRegions = {}
    usersId_regions = []
    tripId_regions = []
    routes_regions = []
    user_routes = {}
    for t in range(len(allTripIds)):
        trip = allTrips_refineRes[t]
        tripId = allTripIds[t]
        # find the traversed regions
        regions = findRegions(trip, idx, polygons)
        regions.sort()
        if len(regions)<=1 or len(regions)>3:
            continue
        if len(regions)==3:
            regions = regions[:2]
        regions = tuple(regions)
        if regions not in routePaires:
            continue

        routeRegions[tripId] = regions
        usersId_regions.append(allUserIds[t])
        tripId_regions.append(allTripIds[t])
        routes_regions.append(regions)

        if allUserIds[t] not in user_routes:
            user_routes[allUserIds[t]] = [(depTimes[tripId], regions)]
        else:
            user_routes[allUserIds[t]].append((depTimes[tripId], regions))


    # how many drivers stick on one routes, how many others change routing behavior
    numRoutes = {}
    numTrips = {}
    numRoutes_multipleTrips = {}
    twoRoutesUsers = []
    twoRoutesUsers_numTrips = []
    # the distribution of fraction of the dominant route
    fractionDominant = []
    # [AM, MD, PM, RD]
    Route_1 = [0 for i in range(24)] # number of trips in each time period
    Route_2 = [0 for i in range(24)]
    for user in user_routes:
        routes = []
        routes_depTime = []
        for r in user_routes[user]:
            route = "-".join(list(r[1]))
            routes.append(route)
            routes_depTime.append(r[0])  # departure hour
        # routes = [r[1] for r in user_routes[user]]
        numRoutes[user] = len(set(routes))
        numTrips[user] = len(routes)
        # we only consider the users with more than 4 trips
        if len(routes) <= 4:
            continue
        # if len(routes)>1:
        numRoutes_multipleTrips[user] = len(set(routes))
        if len(set(routes))>=2:
            twoRoutesUsers_numTrips.append(len(routes))
            twoRoutesUsers.append(tuple(set(routes)))
            # fraction of dominated route
            fracR1 = 0.0  # fraction of the dominated routes
            dominantRoute = ''
            for r in range(len(set(routes))):
                route = list(set(routes))[r]
                frac = len([1 for r in routes if r==route]) / float(len(routes))
                if frac > fracR1:
                    fracR1 = frac
                    dominantRoute = route
            fracR2 = 1.0 - fracR1  # fraction of other routes
            fractionDominant.append(fracR1)
            depTimeR1 = []
            depTimeR2 = []
            for r in range(len(routes)):
                if routes[r]==dominantRoute:
                    #
                    depTimeR1.append(routes_depTime[r])
                else:
                    depTimeR2.append(routes_depTime[r])

            # R1 is the dominate route, R2 is the alternative
            for dt in depTimeR1:
                Route_1[dt] += 1
            for dt in depTimeR2:
                Route_2[dt] += 1


    # plot the distribution of numTrips
    interval = 1
    bins = np.linspace(1, 15, 15)
    usagesHist = np.histogram(np.array(numTrips.values()), bins)
    bins = np.array(bins[:-1])
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='data')

    for p in ax.patches:
        ax.annotate("%.2f%%" % (100*p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=8)

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xlim(1, 15)
    plt.xticks(range(1, 15, 2))
    plt.xlabel(r'# trips per user', fontsize=12)
    plt.ylabel(r"Density", fontsize=12)
    plt.yscale('log', nonposy='clip')

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/numTrips_" + str(zonePair) + ".png", dpi=300)
    plt.close()

    print(set(twoRoutesUsers))
    # routes change fraction
    twoRoutesCount = collections.Counter(twoRoutesUsers)
    topRoutes = sorted(twoRoutesCount, key=twoRoutesCount.get, reverse=True)

    topRoutes = topRoutes[:8]

    print(topRoutes)
    # bar plot
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(1, 1, 1)
    x = topRoutes
    y = [twoRoutesCount[r]/float(len(twoRoutesUsers)) for r in topRoutes]
    plt.bar(range(len(x)), y, align='center', width=0.9, linewidth=1, facecolor='#41A7D8',
            edgecolor='k')
    for p in ax.patches:
        ax.annotate("%.2f%%" % (100*p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=6)

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xticks(range(len(x)), topRoutes, rotation=45)
    plt.xlabel(r'Routes', fontsize=12)
    plt.ylabel(r"Fraction of users", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userTwoRoutes_" + str(zonePair) + ".png", dpi=300)
    plt.close()

    # the distribution of fraction of the dominant route
    # plot the distribution of numRecords
    interval = 0.1
    bins = np.linspace(0.5, 1.0, 6)
    usagesHist = np.histogram(np.array(fractionDominant), bins)
    bins = np.array(bins[:-1])
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='data')

    for p in ax.patches:
        ax.annotate("%.2f%%" % (100*p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xlim(0.45, 1.05)
    plt.xlabel(r'Adoption rate of dominant route', fontsize=12)
    plt.ylabel(r"Density", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userTwoRoutes_dominant_" + str(zonePair) + ".png", dpi=300)
    plt.close()


    # plot the fraction of alternative route
    fig = plt.figure(figsize=(6,3))
    ax = plt.subplot(1, 1, 1)
    bar_width=0.4
    plt.bar(range(24), Route_1, align='edge', width=bar_width, linewidth=1, facecolor='#41A7D8',
            edgecolor='k', label='Dominant')
    plt.bar([i+bar_width for i in range(24)], Route_2, align='edge', width=bar_width, linewidth=1, facecolor='#005a32',
            edgecolor='k', label='Alternative')

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xlim(0, 24)
    plt.xticks([i+bar_width for i in range(24)], range(24))
    plt.xlabel(r'Time [hr]', fontsize=12)
    plt.ylabel(r"# trips", fontsize=12)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userTwoRoutes_time_" + str(zonePair) + ".png", dpi=300)
    plt.close()

    # fraction of the second route
    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 1, 1)
    Route_2_frac = []
    for i in range(24):
        total = float(Route_1[i] + Route_2[i])
        if total==0:
            Route_2_frac.append(0)
            continue
        Route_2_frac.append(Route_2[i]/total)

    plt.plot(range(0, 24), Route_2_frac, marker="o", markersize=7, markeredgewidth=1,
             markerfacecolor='#41A7D8', markeredgecolor='k')

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xlim(0, 24)
    plt.xticks(range(24))
    plt.xlabel(r'Time [hr]', fontsize=12)
    plt.ylabel(r"Fraction of alternative route", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userTwoRoutes_alternative_" + str(zonePair) + ".png", dpi=300)
    plt.close()

    # plot the distribution of numRecords
    interval = 1
    bins = np.linspace(2, 15, 14)
    usagesHist = np.histogram(np.array(twoRoutesUsers_numTrips), bins)
    bins = np.array(bins[:-1])
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='data')

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xlim(2,15)
    plt.xticks(range(2,15,2))
    plt.xlabel(r'# trips per user', fontsize=12)
    plt.ylabel(r"Density", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userTwoRoutes_numTrips_" + str(zonePair) + ".png", dpi=300)
    plt.close()


    routesFraction = []
    routesFraction.append(len([i for i in numRoutes_multipleTrips.values() if i==1]))
    routesFraction.append(len([i for i in numRoutes_multipleTrips.values() if i == 2]))
    routesFraction.append(len([i for i in numRoutes_multipleTrips.values() if i == 3]))
    routesFraction.append(len([i for i in numRoutes_multipleTrips.values() if i >3]))

    totalUsers = len(numRoutes_multipleTrips)
    routesFraction = [float(i)/totalUsers for i in routesFraction]

    print(routesFraction)
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.bar(range(4), routesFraction, align='center', width=0.8, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='data')

    for p in ax.patches:
        ax.annotate("%.2f%%" % (100*p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xticks(range(4), ["1","2","3",">3"])
    plt.xlabel(r'# of routes', fontsize=12)
    plt.ylabel(r"Fraction of users", fontsize=12)
    plt.yscale('log', nonposy='clip')

    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userRoutesFrac_" + str(zonePair) + ".png", dpi=300)
    plt.close()

    hourlyTrips = [0 for i in range(24)]

    for trip in routeRegions:
        depTime = depTimes[trip]
        hourlyTrips[depTime] += 1
        if 7<=depTime<=10:
            routesAM[routeRegions[trip]] += 1
        if 10<depTime<16:
            routesMD[routeRegions[trip]] += 1
        if 16<=depTime<=19:
            routesPM[routeRegions[trip]] += 1
        if 19<depTime or depTime<7:
            routesRD[routeRegions[trip]] += 1

    # plot the hourly trips
    fig = plt.figure(figsize=(4,3))
    plt.bar(range(24), hourlyTrips, width=1, lw=1, facecolor='#41A7D8', edgecolor='k')
    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.xlabel("Time [hr]")
    plt.ylabel("# trips")
    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/hourlyTrips_" + str(zonePair) + ".png", dpi=300)
    plt.close()

    # plot
    routes = [routesAM, routesMD, routesPM, routesRD]
    routeTimes = ["AM", "MD", "PM", "RD"]
    for r in range(4):
        routeTime = routeTimes[r]
        route = routes[r]
        totalRoutes = float(np.sum(route.values()))

        # convert to matrix
        harvest = np.zeros((len(rowNames), len(colNames)))
        for i in range(len(rowNames)):
            for j in range(len(colNames)):
                key = (rowNames[i], colNames[j])
                route[key] = route[key] / totalRoutes
                harvest[i,j] = np.round(route[key], 2)
        print(harvest)

        # plot
        if zonePair in [2, 3]:
            fs = (3,5)
        if zonePair in [0, 1]:
            fs = (4,4)
        fig = plt.figure(figsize=fs)
        ax = plt.subplot()
        im = ax.imshow(harvest, vmin=0, vmax=1.0)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(colNames)))
        ax.set_yticks(np.arange(len(rowNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(colNames)
        ax.set_yticklabels(rowNames)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(rowNames)):
            for j in range(len(colNames)):
                text = ax.text(j, i, harvest[i, j],
                               ha="center", va="center", color="w", fontsize=16)

        ax.set_title("Fraction of route choices - " + routeTime)
        plt.tight_layout()
        plt.savefig(dataPath + "Trips_all/routes_" + str(zonePair) + "_" + routeTime + ".png", dpi=300)
        plt.close()


    # save
    # number of routes per user
    pickle.dump(user_routes, open(dataPath + "Trips_all/user_routes_" + str(zonePair) + '.pkl', 'wb'),
                pickle.HIGHEST_PROTOCOL)


# extract user data from the raw data
def extractUserData(yearmonth):
    # collect all users
    allUsers = []
    for zp in range(4):
        # load the data
        data = pickle.load(open(dataPath + "Trips_all/highResTrips_" + str(zp) + '.pkl', 'rb'))

        allTripIds, allUserIds, rawTrips, allTrips_refineRes, allTrips_refineRes_grids, allTrips_refineRes_cutoff = data

        print("# of users : %d" % len(set(allUserIds)))
        allUsers.extend(allUserIds)

    allUsers = set(allUsers)

    print("# of users in total : ", len(allUsers))

    # collect all records for the users
    inData = open(dataPath + "RawData/RawData_" + yearmonth + "_sorted.csv", 'rb')
    outData = open(dataPath + "RawData/userData_" + yearmonth + ".csv", 'wb')
    preTrip = ''
    preUser = ''
    locations = []
    userLocations = {}
    for row in inData:
        row = row.rstrip().split(',')
        userId = int(row[0])
        if userId not in allUsers:
            continue
        # save the user data
        outData.writelines(','.join(row) + '\n')

    inData.close()
    outData.close()


# stay point detection
# Ref: www2010 - Collaborative Location and Activity Recommendations with GPS History Data
def stayPointsDetection(trace):
    stay_dist_limit = 0.3  # 300m
    time_limit = 600  # 10mins
    cluster = []
    clusters = []
    lon_centroid = trace[0][1]
    lat_centroid = trace[0][2]
    time_begin = trace[0][0]
    in_stace_count = 1
    lons_stay = []
    lats_stay = []
    times_arrival = []
    durarions = []

    for i in range(1, len(trace)):
        if trace[i]==trace[i-1]:
            # skip this record
            continue

        currentLon, currentLat = trace[i][1:]
        preTime = trace[i-1][0]
        currentTime = trace[i][0]
        distance = haversine(currentLat, currentLon, lat_centroid, lon_centroid)

        if distance > stay_dist_limit:
            # a new moving
            # save the old stay
            if preTime - time_begin > time_limit:
                lons_stay.append(lon_centroid)
                lats_stay.append(lat_centroid)
                times_arrival.append(time_begin)
                durarions.append(preTime - time_begin)
            # initiate a new stay
            lon_centroid = currentLon
            lat_centroid = currentLat
            time_begin = currentTime
            in_stace_count=1
            is_location_stack_full = True
        else:
            # not a new moving, update the centroid
            new_lon_centroid = (lon_centroid*in_stace_count + currentLon)/float(in_stace_count + 1)
            new_lat_centroid = (lat_centroid*in_stace_count + currentLat) / float(in_stace_count + 1)
            lon_centroid = new_lon_centroid
            lat_centroid = new_lat_centroid
            in_stace_count += 1

    return lons_stay, lats_stay, times_arrival, durarions


# find the home and work location of each user
# step 1 : find the stay and passby points by clustering
# Ref. 1. Zheng, Vincent W., et al. 19th www, 2010.
# Ref. 2. Jiang, et al, KDD urban computing workshop, 2013
def findStayPoints():
    # load the user Data
    inData = open(dataPath + "RawData/userData_sorted.csv", 'rb')
    outData = open(dataPath + "RawData/userStayPoints.csv", 'wb')

    count = 0
    userSet = set()
    pointsList = []
    preUser = ''
    numStayPoints = []
    userStayPoints = {}
    for row in inData:
        count += 1
        if count % 1e4 == 0:
            print(count)
        row = row.rstrip().split(',')
        user = int(row[0])
        # print(user, targetUser)
        lon = float(row[3])
        lat = float(row[2])
        ts = int(float(row[1]))
        # ts_dt = datetime.datetime.fromtimestamp(ts, tz=DallasTZ)
        # ts_hour = ts_dt.hour
        # weekday = ts_dt.weekday()

        if preUser=='':
            preUser = user
            userSet.add(user)

        if user not in userSet:
            # process the last user
            # do points clustering
            lons_stay, lats_stay, times_arrival, durarions = stayPointsDetection(pointsList)
            for i in range(len(lons_stay)):
                row_save = [str(preUser), str(times_arrival[i]), str(durarions[i]), str(lons_stay[i]), str(lats_stay[i])]
                outData.writelines(','.join(row_save) + '\n')
            numStayPoints.append(len(lons_stay))
            userStayPoints[preUser] = [lons_stay, lats_stay, times_arrival, durarions]

            # initialize
            preUser = user
            pointsList = []

        userSet.add(user)
        pointsList.append([ts, lon, lat])

    # process the last user
    # do points clustering
    lons_stay, lats_stay, times_arrival, durarions = stayPointsDetection(pointsList)
    for i in range(len(lons_stay)):
        row_save = [str(user), str(times_arrival[i]), str(durarions[i]), str(lons_stay[i]), str(lats_stay[i])]
        outData.writelines(','.join(row_save) + '\n')
    numStayPoints.append(len(lons_stay))
    userStayPoints[user] = [lons_stay, lats_stay, times_arrival, durarions]

    inData.close()
    outData.close()

    # save
    print("Saving userStayPoints to pickle file...")
    pickle.dump(userStayPoints, open(dataPath + "RawData/userStayPoints.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)



def homeDetection(pointsList):
    points = []
    for i in range(len(pointsList)):
        lon, lat = pointsList[i][2:]
        points.append([lon, lat])

    points = np.array(points)
    clustering = DBSCAN(eps=0.0005, min_samples=2).fit(points)
    assignments = {}
    stayTimes = {}
    # assign data points to clusters
    for idx in range(len(clustering.labels_)):
        clu = clustering.labels_[idx]
        ts_start, ts_end = pointsList[idx][:2]
        if clu==-1:
            continue
        if clu not in assignments:
            assignments[clu] = [idx]
            stayTimes[clu] = [[ts_start, ts_end]]
        else:
            assignments[clu].append(idx)
            stayTimes[clu].append([ts_start, ts_end])

    if len(assignments) == 0:
        return (-1,-1)
    # update stay location
    stayLoc = {}
    for clu in assignments:
        idx = assignments[clu]
        lon_centroid, lat_centroid = np.mean(points[idx], axis=0)
        stayLoc[clu] = (lon_centroid, lat_centroid)

    # update the stay time
    visited_times_night = []
    visited_times = []
    for clu in range(len(assignments)):
        idx = assignments[clu]
        visited_times.append(len(idx))
        times = [pointsList[p][:2] for p in range(len(pointsList)) if p in idx]
        # connect the two consecutive time if the difference is smaller than 1 hour
        times_update = []
        pre_start = times[0][0]
        pre_end = times[0][1]
        for t in range(1,len(times)):
            current_start = times[t][0]
            current_end = times[t][1]
            if times[t][0] - times[t-1][1] <= 3600:
                if t==len(times)-1:
                    times_update.append([pre_start, current_end])
                else:
                    pre_end = current_end
                    continue
            else:
                times_update.append([pre_start, pre_end])
                pre_start = current_start
                pre_end = current_end

        # convert the times to weekday and hours
        weekday_hour = []
        count_home = 0
        for t in range(len(times_update)):
            time_start, time_end = times_update[t]
            time_start = datetime.datetime.fromtimestamp(time_start, tz=DallasTZ)
            time_start_hour = time_start.hour
            weekday = time_start.weekday()
            weekday_hour.append([weekday, time_start_hour])
            if weekday <= 4:
                if time_start_hour <=8 or time_start_hour>=20:
                    count_home += 1
            else:
                count_home += 1

        visited_times_night.append(count_home)

    # find home from the stay regions and the arrival times
    # define home as the most frequently stayed region during nights of weekdays, and weekends
    home_region_idx = np.argmax(visited_times_night)
    home_location = stayLoc[home_region_idx]
    return home_location




# find the stay regions by clustering the stay points with DBSCAN
# for each region, we keep the visiting hours
# then, we find the home region by counting the arrival time and frequency
def findHomeStayRegions():
    # load the user Data
    inData = open(dataPath + "RawData/userStayPoints.csv", 'rb')
    outData = open(dataPath + "RawData/userHomeLocation.csv", 'wb')

    count = 0
    userSet = set()
    pointsList = []
    preUser = ''
    numStayPoints = []
    userStayPoints = {}
    count_noHome = 0
    for row in inData:
        count += 1
        if count % 1e4 == 0:
            print(count, " --- ", count_noHome)
        row = row.rstrip().split(',')
        user = int(row[0])
        # print(user, targetUser)
        lon = float(row[3])
        lat = float(row[4])
        ts = int(row[1])
        # ts_dt = datetime.datetime.fromtimestamp(ts, tz=DallasTZ)
        # ts_hour = ts_dt.hour
        # weekday = ts_dt.weekday()
        duration = int(row[2])
        ts_end = ts+duration

        if preUser=='':
            preUser = user
            userSet.add(user)

        if user not in userSet:
            # process the last user
            # find home
            home_loc = homeDetection(pointsList)
            if home_loc[0]!=-1:
                row_save = [str(preUser), str(home_loc[0]), str(home_loc[1])]
                outData.writelines(','.join(row_save) + '\n')
            else:
                print(user)
                count_noHome += 1

            # initialize
            preUser = user
            pointsList = []

        userSet.add(user)
        pointsList.append([ts, ts_end, lon, lat])

    # process the last user
    # find home
    home_loc = homeDetection(pointsList)
    row_save = [str(user), str(home_loc[0]), str(home_loc[1])]
    outData.writelines(','.join(row_save) + '\n')

    inData.close()
    outData.close()

    print("# users without home : ", count_noHome)


# home detection
def findHomeTractAndIncome():
    # load census tract map
    polygons = [pol for pol in fiona.open(dataPath + 'Geo/tl_2017_48_tract/tl_2017_48_tract.geojson')]
    # with a R-tree index (you can use pyrtree or rtree)
    idx = index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shape(poly['geometry']).bounds)

    # iterate through points
    inData = open(dataPath + "userData/userHomeWork_loc.csv", 'rb')
    inData.readline()
    userHomeTract = {}
    count = 0
    userInf = {}
    for row in inData:
        count += 1
        row = row.rstrip().split(',')
        user = int(row[0])
        lon = float(row[1])
        lat = float(row[2])
        lon_work = float(row[3])
        lat_work = float(row[4])
        if lon == 0:  # no home found
            continue
        pt = Point(lon, lat)
        # iterate through spatial index
        for j in idx.intersection(pt.coords[0]):
            if pt.within(shape(polygons[j]['geometry'])):
                homeTract = int(polygons[j]['properties']['GEOID'])
                userHomeTract[user] = homeTract
                userInf[user] = [lon, lat, lon_work, lat_work, homeTract]
    inData.close()

    print("# users with home %d / %d" % (len(userHomeTract), count))

    # load the demographic data
    jsonFile = open(dataPath + "census/geodf_census_TEXAS_TABLES.json", 'rb')
    jsonStr = jsonFile.read()
    jsonData = json.loads(jsonStr)

    populationList = []
    tractIncome = {}
    for feature in jsonData['features']:
        tract = int(feature['properties']['GEOID'])
        pop = int(feature['properties']['B00001_001E'])
        income_all_median = float(feature['properties']['B06011_001E'])
        hhIncome = float(feature['properties']['B19001_001E'])
        populationList.append(pop)
        tractIncome[tract] = income_all_median

    print("total population : ", np.sum(populationList))
    print("# features : ", len(jsonData['features']))

    # assign each user income level and
    userIncome = {}  # household income
    for user in userHomeTract.keys():
        income_in_tract = tractIncome[userHomeTract[user]]
        income = get_tract_income(income_in_tract)
        if income < 0:
            print("error : ", user)
            continue
        userIncome[user] = income / 1000.0  # thousand dollar
        userInf[user].append(income / 1000.0)

    pickle.dump(userIncome, open(dataPath + "userData/userIncome.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    outData = open(dataPath + "userData/userHomeLocationIncome.csv", 'wb')
    outData.writelines('user,lon,lat,tract,income\n')
    for user in userInf:
        inf = userInf[user]
        if len(inf)==5:  # no income found
            continue
        lon, lat, lon_work, lat_work, tract, income = userInf[user]
        row = [str(user), str(lon), str(lat), str(lon_work), str(lat_work), str(tract), str(income)]
        outData.writelines(','.join(row) + '\n')
    outData.close()

    pickle.dump(userInf, open(dataPath + "userData/userHomeLocationIncome.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)

    # plot distribution of user income
    interval = 5
    bins = np.linspace(0, 150, 31)
    usagesHist = np.histogram(np.array(userIncome.values()), bins)
    bins = np.array(bins[1:])
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='data')

    plt.xlim(0, 150)
    # plt.ylim(0, 0.05)
    plt.xticks(range(0, 150, 10), fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale("log")
    plt.xlabel(r'User income (K$)', fontsize=16)
    plt.ylabel(r"Fraction", fontsize=16)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/userIncome_distribution.png', dpi=150)
    plt.close()




def get_tract_income(meanIncome):
    mean = meanIncome
    std = mean/4 ## just an approximation.
    pick = int(random.normalvariate(mean,std))
    pick = max(mean - std, pick)
    pick = min(mean + std, pick)
    return int(pick)


# connect the routing behavior with demographic data
def userBehaviorVSIncome(zonePair=0):
    startZone, targetZone = zonePairs[zonePair]
    print("From zone %d to %d ... " % (startZone, targetZone))
    # load user income
    userInf = pickle.load(open(dataPath + 'RawData/userHomeLocationIncome.pkl', 'rb'))

    # load user routes
    userRoutes = pickle.load(open(dataPath + "Trips_all/user_routes_" + str(zonePair) + '.pkl', 'rb'))

    print("# of users %d (income), %d (routes)" % (len(userInf), len(userRoutes)))
    # number of routes vs income
    userList = set(userInf.keys()).intersection(set(userRoutes.keys()))

    print("# of users : %d" % len(userList))

    incomes = []
    numRoutes = []
    boxPlotData = []
    # fraction of routes number by income level
    '''
    # income level: < 20, 20-40, 40-60, 60-80, >80
    numRoutesByIncome = {"Income Level (k$)": ['<20', '20-40', '40-60', '60-80', '>80'],
                "One route": [0,0,0,0,0],
                "Two routes": [0,0,0,0,0],
                "Three or more": [0,0,0,0,0]}
    '''
    # income level: < 30, 30-60, >60
    numRoutesByIncome = {"Income Level (k$)": ['<25', '25-50', '>50'],
                         "One route": [0, 0, 0],
                         "Two routes": [0, 0, 0],
                         "Three or more": [0, 0, 0]}


    for user in userList:
        routes = []
        routes_depTime = []
        for r in userRoutes[user]:
            route = "-".join(list(r[1]))
            routes.append(route)
            routes_depTime.append(r[0])  # departure hour
        if len(routes) <= 4:
            continue
        numR = len(set(routes))
        numTrip = len(routes)
        try:
            income = userInf[user][3]
        except:
            # no income found
            continue
        numRoutes.append(numR)
        incomes.append(income)
        boxPlotData.append([numR, income])
        incomeLevel = min(int(income / 25), 2)
        if numR==1:
            numRoutesByIncome["One route"][incomeLevel] += 1
        if numR==2:
            numRoutesByIncome["Two routes"][incomeLevel] += 1
        if numR>2:
            numRoutesByIncome["Three or more"][incomeLevel] += 1

    # save the data for joypolots in R
    outData = open(dataPath + "Trips_all/routes_income_" + str(zonePair) + '.csv', 'wb')
    outData.writelines("User,Routes,Income,HomeTract,HomeLon,HomeLat\n")
    for user in userList:
        routes = []
        for r in userRoutes[user]:
            route = "-".join(list(r[1]))
            routes.append(route)
        if len(routes) <= 4:
            continue
        numR = min(3,len(set(routes)))
        try:
            lon, lat, tract, income = userInf[user]
        except:
            continue

        outData.writelines(','.join([str(user), str(numR), str(income), str(tract),
                                     str(lon), str(lat)]) + '\n')
    outData.close()

    fig = plt.figure()
    plt.scatter(numRoutes, incomes, marker='_')
    plt.ylabel("User income (k$)")
    plt.xlabel("# routes")
    plt.title("From zone %d to zone %d" % (startZone, targetZone))
    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/routes_income_" + str(zonePair) + '.png', dpi=300)
    plt.close()

    # box plot
    boxPlotData_DF = pd.DataFrame(boxPlotData, columns=['routes', 'income'])

    # box plot
    fig = plt.figure(figsize=(6, 4))
    sns.set_style("ticks")

    # create our boxplot which is drawn on an Axes object
    bplot = sns.boxplot(x='routes', y='income', data=boxPlotData_DF, whis=[5, 80], width=0.5)

    # We can call all the methods avaiable to Axes objects
    # bplot.set_title(title, fontsize=20)
    bplot.set_xlabel('# routes', fontsize=16, color='k')
    bplot.set_ylabel('Income [k$]', fontsize=16, color='k')
    bplot.tick_params(axis='both', labelsize=12, color='k')
    plt.xticks(rotation=45)
    plt.title("From zone %d to zone %d" % (startZone, targetZone))

    # sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/routes_income_" + str(zonePair) + '_box.png', dpi=300)
    plt.close()

    # stack bar plot
    # Create a figure with a single subplot
    df = pd.DataFrame(numRoutesByIncome, columns=['Income Level (k$)', 'One route', 'Two routes', 'Three or more'])

    fig = plt.figure()
    f, ax = plt.subplots(1, figsize=(4,3))
    # Set bar width at 1
    bar_width = 0.8
    # positions of the left bar-boundaries
    bar_l = [i for i in range(len(df['One route']))]

    # positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create the total score for each participant
    totals = [i + j + k for i, j, k in zip(df['One route'], df['Two routes'], df['Three or more'])]

    totals = [float(i) if i>0 else 0.001 for i in totals]
    # Create the percentage of the total score the pre_score value for each participant was
    pre_rel = [i / j * 100 for i, j in zip(df['One route'], totals)]

    # Create the percentage of the total score the mid_score value for each participant was
    mid_rel = [i / j * 100 for i, j in zip(df['Two routes'], totals)]

    # Create the percentage of the total score the post_score value for each participant was
    post_rel = [i / j * 100 for i, j in zip(df['Three or more'], totals)]

    bar_l = [i+bar_width/2.0 for i in bar_l]
    # Create a bar chart in position bar_1
    ax.bar(bar_l,
           # using pre_rel data
           pre_rel,
           # labeled
           label='One route',
           # with alpha
           alpha=0.9,
           # with color
           color='#8c2d04',
           # with bar width
           width=bar_width,
           # with border color
           edgecolor='white')

    # Create a bar chart in position bar_1
    ax.bar(bar_l,
           # using mid_rel data
           mid_rel,
           # with pre_rel
           bottom=pre_rel,
           # labeled
           label='Two routes',
           # with alpha
           alpha=0.9,
           # with color
           color='#3C5F5A',
           # with bar width
           width=bar_width,
           # with border color
           edgecolor='white')

    # Create a bar chart in position bar_1
    ax.bar(bar_l,
           # using post_rel data
           post_rel,
           # with pre_rel and mid_rel on bottom
           bottom=[i + j for i, j in zip(pre_rel, mid_rel)],
           # labeled
           label='Three & more',
           # with alpha
           alpha=0.9,
           # with color
           color='#219AD8',
           # with bar width
           width=bar_width,
           # with border color
           edgecolor='white')

    # Set the ticks to be first names
    plt.xticks(tick_pos, df['Income Level (k$)'])
    ax.set_ylabel("Percentage [%]")
    ax.set_xlabel("Income Level [k$]")

    # Let the borders of the graphic
    plt.xlim([min(tick_pos) - bar_width, max(tick_pos) + bar_width])
    plt.ylim(0, 100)

    plt.title("From zone %d to zone %d" % (startZone, targetZone))

    # rotate axis labels
    plt.setp(plt.gca().get_xticklabels(), horizontalalignment='center')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # shot plot
    # plt.show()
    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/routes_income_bar_" + str(zonePair) + '.png', dpi=300)
    plt.close()


# compare the user income with all population
def userIncomeComparison():
    userIncome = pickle.load(open(dataPath + 'userData/userIncome.pkl', 'rb'))

    # load the census tracts in the given 4 zones
    geoFile = open(dataPath + 'Geo/tracts_in_selectedZones.geojson', 'r')
    geoData = geojson.load(geoFile)

    selectedTracts = set()
    for t in geoData['features']:
        tract = int(t['properties']['GEOID'])
        selectedTracts.add(tract)
    geoFile.close()

    # user income of all population
    # load the demographic data
    jsonFile = open(dataPath + "census/geodf_census_TEXAS_TABLES.json", 'rb')
    jsonStr = jsonFile.read()
    jsonData = json.loads(jsonStr)

    tractIncome = []
    tractPopulation = []
    for feature in jsonData['features']:
        tract = int(feature['properties']['GEOID'])
        if tract not in selectedTracts:
            continue
        pop = int(feature['properties']['B00001_001E'])
        income_all_median = float(feature['properties']['B06011_001E'])
        # hhIncome = float(feature['properties']['B19001_001E'])
        tractIncome.append(income_all_median)
        tractPopulation.append(pop)
    jsonFile.close()

    # sample user income
    totalPopulation = float(np.sum(tractPopulation))
    tractPopulation_frac = [i/totalPopulation for i in tractPopulation]

    numSelected = int(5e4)
    idx = np.random.choice(len(tractIncome), numSelected, replace=True, p=tractPopulation_frac)
    income_selected = []
    for i in idx:
        income_all_median = tractIncome[i]
        income = get_tract_income(income_all_median)/1000.0
        income_selected.append(income)


    # plot distribution of user income
    interval = 5
    bins = np.linspace(0, 120, 25)
    usagesHist = np.histogram(np.array(userIncome.values()), bins)
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    usagesHist_pop = np.histogram(np.array(income_selected), bins)
    usagesHist_pop = np.divide(usagesHist_pop[0], float(np.sum(usagesHist_pop[0])))
    print(usagesHist_pop)

    bins = np.array(bins[1:])

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k', label='Sample users')
    plt.plot((bins+0.5*interval).tolist(), usagesHist_pop.tolist(), marker='o', markersize=8, linewidth=2,
             markerfacecolor='#cb181d', color='k', label="All population")

    plt.xlim(0, 120)
    # plt.ylim(0, 0.05)
    plt.xticks(range(0, 120, 10), fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale("log")
    plt.xlabel(r'User income (K$)', fontsize=16)
    plt.ylabel(r"Fraction", fontsize=16)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + 'Trips_all/userIncome_distribution.png', dpi=150)
    plt.close()



def userHomeCompare():
    userInf = pickle.load(open(dataPath + "userData/userHomeLocationIncome.pkl", 'rb'))
    tractNumUsers = {}
    for user in userInf:
        try:
            lon, lat, lon_work, lat_work, homeTract, income = userInf[user]
        except:
            print(userInf[user])
        try:
            tractNumUsers[homeTract][0] += 1
        except:
            tractNumUsers[homeTract] = [0, 0]

    # load the demographic data
    jsonFile = open("/media/xu/TOSHIBA EXT/Study/HuMNetLab/Data/Cintra/census/geodf_census_TEXAS_TABLES.json", 'rb')
    jsonStr = jsonFile.read()
    jsonData = json.loads(jsonStr)


    for feature in jsonData['features']:
        tract = int(feature['properties']['GEOID'])
        if tract not in tractNumUsers:
            continue
        pop = int(feature['properties']['B00001_001E'])
        tractNumUsers[tract][1] = pop
    jsonFile.close()

    # plot
    fig = plt.figure(figsize=(4,3))
    for tract in tractNumUsers:
        plt.scatter(tractNumUsers[tract][1], tractNumUsers[tract][0], marker='o',
                    c='r', s=10, alpha=0.3, edgecolor='k', lw=0.5)
    plt.xlabel("Population in tract")
    plt.ylabel("Users in LBS data")
    plt.tight_layout()
    plt.savefig(dataPath + "userData/userHomeTract_compare.png", dpi=300)
    plt.savefig(dataPath + "userData/userHomeTract_compare.pdf")
    plt.close()



# time demension
# compare the number of routes of each traveler during peak hours and non-peak hours, 
# or plot numOfRoutes vs. Hour for weekdays and weekends, respectively.
def numRoutesInTime():
    # load user routes informaiton
    # trip information
    tripData = open(dataPath + "userData/userRoutesInf_complete.csv", 'r')
    tripData.readline()

    userTraveltimes = {}

    userTravelDistances = {}
    numRoutesInTimeSlot_weekday = {}
    numRoutesInTimeSlot_weekend = {}

    routesIDsInTimeSlot_weekday = [{} for i in range(24)]
    routesIDsInTimeSlot_weekend = [{} for i in range(24)]

    ttsInTimeSlot_weekday = [{} for i in range(24)]
    ttsInTimeSlot_weekend = [{} for i in range(24)]
    
    for row in tripData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        depTime = datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        arrTime = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
        depWeekday = depTime.weekday()
        depHour = depTime.hour
        # sLon = float(row[4])
        # sLat = float(row[5])
        # tLon = float(row[6])
        # tLat = float(row[7])
        tt = float(row[8])  # min
        routeID = int(row[9])
        if routeID == -1:
            continue
        # distance = haversine(sLat, sLon, tLat, tLon)
        if depWeekday < 5:
            try:
                routesIDsInTimeSlot_weekday[depHour][uid].append(routeID)
                ttsInTimeSlot_weekday[depHour][uid].append(tt)
            except:
                routesIDsInTimeSlot_weekday[depHour][uid] = [routeID]
                ttsInTimeSlot_weekday[depHour][uid] = [tt]
        else:
            try:
                routesIDsInTimeSlot_weekend[depHour][uid].append(routeID)
                ttsInTimeSlot_weekend[depHour][uid].append(tt)
            except:
                routesIDsInTimeSlot_weekend[depHour][uid] = [routeID]
                ttsInTimeSlot_weekend[depHour][uid] = [tt]

    # number of routes during each hour on weekdays
    # am: 7-10, md: 10-16, pm: 16-20, rd: 20-7
    hourToPeroid = {}
    for h in range(24):
        if 7<=h<10:
            hourToPeroid[h] = 0
        elif 10<=h<16:
            hourToPeroid[h] = 1
        elif 16<=h<19:
            hourToPeroid[h] = 2
        else:
            hourToPeroid[h] = 3
    
    hourToPeroid_weekend = {}
    for h in range(24):
        if 7<=h<10:
            hourToPeroid_weekend[h] = 0
        elif 12<=h<14:
            hourToPeroid_weekend[h] = 1
        elif 16<=h<19:
            hourToPeroid_weekend[h] = 2
        else:
            hourToPeroid_weekend[h] = 3

    numRoutesPerHour_weekday = [[] for i in range(4)] # rd, am, md, pm
    numTripsPerHour_weekday = [0 for i in range(4)]
    numRoutesCount_weekday = []
    for h in range(24):
        routesInTS = routesIDsInTimeSlot_weekday[h]
        numRoutes = []
        numTrips = 0
        for uid in routesInTS:
            numR = len(set(routesInTS[uid]))
            numRoutes.append(numR)
            numTrips += len(routesInTS[uid])
        period = hourToPeroid[h]
        numRoutesPerHour_weekday[period].extend(numRoutes)
        numTripsPerHour_weekday[period] += numTrips
        count = collections.Counter(numRoutes)
        numRoutesCount_weekday.append(count)
    
    numRoutesPerHour_weekend = [[] for i in range(4)]
    numTripsPerHour_weekend = [0 for i in range(4)]
    numRoutesCount_weekend = []
    for h in range(24):
        routesInTS = routesIDsInTimeSlot_weekend[h]
        numRoutes = []
        numTrips = 0
        for uid in routesInTS:
            numR = len(set(routesInTS[uid]))
            numRoutes.append(numR)
            numTrips += len(routesInTS[uid])
        period = hourToPeroid_weekend[h]
        numRoutesPerHour_weekend[period].extend(numRoutes)
        numTripsPerHour_weekend[period] += numTrips
        count = collections.Counter(numRoutes)
        numRoutesCount_weekend.append(count)

    # plot the distribution of number of routes per hour
    # 24 PDFs, similar to Fig. 5b
    fig = plt.figure(figsize=(4,3))
    # setup the normalization and the colormap
    normalize = mcolors.Normalize(vmin=0, vmax=3)
    colormap = plt.get_cmap("jet")
    colors = ['#005a32', '#e31a1c', '#034e7b', '#cc4c02']
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    periodNames = {0:"AM", 1:"MD", 2:"PM", 3:"RD"}
    for h in range(4):
        hist_0, _ = np.histogram(numRoutesPerHour_weekday[h], bins)
        hist_0 = np.divide(hist_0, float(sum(hist_0)))
        print(hist_0)
        plt.plot(range(1, 6), hist_0, lw=1.5, color=colors[h], marker='o', markersize=5, label=periodNames[h])
    
    plt.yscale("log", nonposy="clip")
    plt.xlabel(r'\# routes, N', fontsize=14)
    plt.ylabel(r"P(N)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(14), [i*5 for i in range(14)])
    # plt.ylim(1.0, 2.0)
    plt.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_hour_weekday.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_hour_weekday.pdf')
    plt.close()

    # bar plot of number of trips in each period
    fig = plt.figure(figsize=(4,3))
    plt.bar(range(4), numTripsPerHour_weekday, width=1.0, linewidth=1, facecolor='#41A7D8',
            edgecolor='k')
    plt.xticks(range(4), ["AM", "MD", "PM", "RD"])
    plt.xlabel(r'Time period', fontsize=14)
    plt.ylabel(r"\# trips", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numTrips_hour_weekday.png', dpi=150)
    plt.savefig(dataPath + 'userData/numTrips_hour_weekday.pdf')
    plt.close()


    # weekend
    fig = plt.figure(figsize=(4,3))
    for h in range(4):
        hist_0, _ = np.histogram(numRoutesPerHour_weekend[h], bins)
        hist_0 = np.divide(hist_0, float(sum(hist_0)))
        print(hist_0)
        plt.plot(range(1, 6), hist_0, lw=1.5, color=colors[h], marker='o', markersize=5, label=periodNames[h])
    
    plt.yscale("log", nonposy="clip")
    plt.xlabel(r'\# routes, N', fontsize=14)
    plt.ylabel(r"P(N)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xticks(range(14), [i*5 for i in range(14)])
    # plt.ylim(1.0, 2.0)
    plt.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numRoutes_hour_weekend.png', dpi=150)
    plt.savefig(dataPath + 'userData/numRoutes_hour_weekend.pdf')
    plt.close()

    # bar plot of number of trips in each period
    fig = plt.figure(figsize=(4,3))
    plt.bar(range(4), numTripsPerHour_weekend, width=1.0, linewidth=1, facecolor='#41A7D8',
            edgecolor='k')
    plt.xticks(range(4), ["AM", "MD", "PM", "RD"])
    plt.xlabel(r'Time period', fontsize=14)
    plt.ylabel(r"\# trips", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/numTrips_hour_weekend.png', dpi=150)
    plt.savefig(dataPath + 'userData/numTrips_hour_weekend.pdf')
    plt.close()


# relative gap of travel time per time period
def relativeGap():
    # am: 7-10, md: 10-16, pm: 16-19, rd: 19-7
    periodNames = {0:"AM", 1:"MD", 2:"PM", 3:"RD"}
    hourToPeroid = {}
    for h in range(24):
        if 7<=h<10:
            hourToPeroid[h] = 0
        elif 10<=h<16:
            hourToPeroid[h] = 1
        elif 16<=h<19:
            hourToPeroid[h] = 2
        else:
            hourToPeroid[h] = 3

    # trip information
    tripData = open(dataPath + "userData/userRoutesInf_complete.csv", 'r')
    tripData.readline()

    ttsInTimeSlot_weekday = [{} for i in range(24)]
    ttsInTimeSlot_weekend = [{} for i in range(24)]
    
    ttsOfUsers = {}
    for row in tripData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        depTime = datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        arrTime = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
        depWeekday = depTime.weekday()
        depHour = depTime.hour
        # period = hourToPeroid[depHour]
        period = depHour

        tt = float(row[8])  # min
        routeID = int(row[9])
        if routeID == -1:
            continue
        # distance = haversine(sLat, sLon, tLat, tLon)
        # weekday only
        if depWeekday >=5:
            continue
        try:
            ttsOfUsers[uid].append(tt)
        except:
            ttsOfUsers[uid] = []
            ttsOfUsers[uid] = [tt]

        try:
            ttsInTimeSlot_weekday[period][uid].append(tt)
        except:
            ttsInTimeSlot_weekday[period][uid] = [tt]

    # for each user, find the minT as free flow travel time
    minTOfUsers = {}
    for uid in ttsOfUsers:
        minTOfUsers[uid] = np.min(ttsOfUsers[uid])

    # for each period, each user, we find the Tmin
    RgsInTimeSlot_weekday = [{} for i in range(24)]
    RgsInTimeSlot_list = []
    for p in range(24):
        userTTs = ttsInTimeSlot_weekday[p]
        for uid in userTTs:
            numTrips = len(userTTs[uid])
            if numTrips < 10:
                continue
            # minT = np.min(userTTs[uid])  # might include holidays
            minT = minTOfUsers[uid]
            # minT = np.percentile(userTTs[uid], 10)
            Rgs = [(t - minT)/minT for t in userTTs[uid]]
            Rgs = [max(r, 0) for r in Rgs]
            RgsInTimeSlot_weekday[p][uid] = Rgs
            TTI = np.mean(userTTs[uid]) / minT
            RgsInTimeSlot_list.append([p, TTI])
            # for r in Rgs:
            #     RgsInTimeSlot_list.append([p, r])
    RgsInTimeSlot_df = pd.DataFrame(RgsInTimeSlot_list, columns=["Hour", "Rg"])

    fig = plt.figure(figsize=(4,3))
    sns.stripplot(RgsInTimeSlot_df["Hour"], RgsInTimeSlot_df["Rg"], jitter=0.2, size=2, alpha=0.5)
    plt.xlabel("Hour")
    plt.ylabel("Travel Time Index")
    plt.ylim(1,6)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/TTI_period_weekday.png', dpi=150)
    plt.savefig(dataPath + 'userData/TTI_period_weekday.pdf')
    plt.close()

    # distribution of Rg during peak hour 7:00-8:00
    peakRg = RgsInTimeSlot_df[RgsInTimeSlot_df["Hour"]==7]
    fig = plt.figure(figsize=(2.5,2))
    sns.distplot(peakRg["Rg"], color="r")
    plt.xlabel("TTI")
    plt.ylabel("Density")
    plt.xlim(0, 7)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/TTI_peak_density_AM.png', dpi=150)
    plt.savefig(dataPath + 'userData/TTI_peak_density_AM.pdf')
    plt.close()

    peakRg = RgsInTimeSlot_df[RgsInTimeSlot_df["Hour"]==17]
    fig = plt.figure(figsize=(2.5,2))
    sns.distplot(peakRg["Rg"], color="r")
    plt.xlabel("TTI")
    plt.ylabel("Density")
    plt.xlim(0, 7)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/TTI_peak_density_PM.png', dpi=150)
    plt.savefig(dataPath + 'userData/TTI_peak_density_PM.pdf')
    plt.close()

    return 0

    # Rg with only one minimum travel time
    # for each period, each user, we find the Tmin
    RgsInTimeSlot_weekday = [{} for i in range(24)]
    RgsInTimeSlot_list = []
    for p in range(24):
        userTTs = ttsInTimeSlot_weekday[p]
        for uid in userTTs:
            numTrips = len(userTTs[uid])
            if numTrips < 10:
                continue
            minT = minTOfUsers[uid]
            Rgs = [(t - minT)/minT for t in userTTs[uid]]
            RgsInTimeSlot_weekday[p][uid] = Rgs
            for r in Rgs:
                RgsInTimeSlot_list.append([p, r])
    RgsInTimeSlot_df = pd.DataFrame(RgsInTimeSlot_list, columns=["Hour", "Rg"])

    fig = plt.figure(figsize=(4,3))
    sns.stripplot(RgsInTimeSlot_df["Hour"], RgsInTimeSlot_df["Rg"], jitter=0.2, size=2, alpha=0.05)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/Rgs_period_oneMin.png', dpi=150)
    plt.savefig(dataPath + 'userData/Rgs_period_oneMin.pdf')
    plt.close()

    # distribution of Rg during peak hour 7:00-8:00
    peakRg = RgsInTimeSlot_df[RgsInTimeSlot_df["Hour"]==7]
    fig = plt.figure(figsize=(3,2))
    sns.distplot(peakRg["Rg"], color="r")
    plt.xlabel("Rg")
    plt.ylabel("Density")
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.savefig(dataPath + 'userData/RgPrime_peak_density.png', dpi=150)
    plt.savefig(dataPath + 'userData/RgPrime_peak_density.pdf')
    plt.close()



# Could exploring more routes increase the reliability of travel time from O to D?
def relativeGapVSNumRoutes():
    # targetHours = [[7,8,9], list(range(10,16)), [16,17,18], list(range(19,24)) + list(range(7))]  # am peak hours, pm peak hours
    # targetNames = ["AM", "MD", "PM", "RD"]
    targetHours = [[7,8,9], [16,17,18]]  # am peak hours, pm peak hours
    targetNames = ["AM", "PM"]

    # trip information
    tripData = open(dataPath + "userData/userRoutesInf_complete.csv", 'r')
    tripData.readline()

    ttsInTimeSlot_weekday = [{} for i in range(24)]
    routesInTimeSlot_weekday = [{} for i in range(24)]
    
    ttsOfUsers = {}
    for row in tripData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        depTime = datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        arrTime = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
        depWeekday = depTime.weekday()
        depHour = depTime.hour
        # period = hourToPeroid[depHour]
        period = depHour

        tt = float(row[8])  # min
        routeID = int(row[9])
        if routeID == -1:
            continue
        # distance = haversine(sLat, sLon, tLat, tLon)
        # weekday only
        if depWeekday >=5:
            continue
        try:
            ttsOfUsers[uid].append(tt)
        except:
            ttsOfUsers[uid] = []
            ttsOfUsers[uid] = [tt]

        try:
            ttsInTimeSlot_weekday[period][uid].append(tt)
            routesInTimeSlot_weekday[period][uid].append(routeID)
        except:
            ttsInTimeSlot_weekday[period][uid] = [tt]
            routesInTimeSlot_weekday[period][uid] = [routeID]

    # for each user, find the minT as free flow travel time
    minTOfUsers = {}
    for uid in ttsOfUsers:
        minTOfUsers[uid] = np.min(ttsOfUsers[uid])

    RgsVSTTs = []
    for i in range(len(targetHours)):
        hrs = targetHours[i]
        for hr in hrs:
            userTTs = ttsInTimeSlot_weekday[hr]
            for uid in userTTs:
                tts = userTTs[uid]
                # minT = np.min(tts)
                # minT = np.percentile(tts, 10)
                minT = minTOfUsers[uid]
                numTrips = len(tts)
                if numTrips < 20:
                    continue

                Rgs = [(t-minT)/minT for t in tts]
                Rgs = [max(r, 0) for r in Rgs]
                bufferIndex = (np.percentile(tts, 85) - np.mean(tts)) /np.mean(tts)
                PTI = np.percentile(tts, 85) / minT
                TTI = np.mean(tts) / minT
                # reliability = bufferIndex
                
                # should we use mean(Rgs) or std(Rgs) to reflect the reliability of travel time?
                reliability = PTI
                numRoutes = len(set(routesInTimeSlot_weekday[hr][uid]))
                numRoutes = min(5, numRoutes)
                RgsVSTTs.append([targetNames[i], reliability, numRoutes])
    RgsVSTTs_df = pd.DataFrame(RgsVSTTs, columns=["Period", "Reliability", "numRoutes"])

    for i in range(len(targetHours)):
        subData = RgsVSTTs_df[RgsVSTTs_df["Period"] == targetNames[i]]
        meanMeanRg = []
        percentileRg_25 = []
        percentileRg_95 = []
        for r in range(1,6):
            subSubData = subData[subData["numRoutes"] == r]
            m = np.mean(subSubData["Reliability"])
            p25 = np.percentile(subSubData["Reliability"], 25)
            p95 = np.percentile(subSubData["Reliability"], 90)
            meanMeanRg.append(m)
            percentileRg_25.append(p25)
            percentileRg_95.append(p95)
        print([np.round(m, 3) for m in meanMeanRg])
        # plot
        fig = plt.figure(figsize=(4,3))
        sns.stripplot(subData["numRoutes"], subData["Reliability"], jitter=0.3, size=2, alpha=0.5, zorder=1)
        # plt.plot(range(5), meanMeanRg, lw=1.5, markerfacecolor="#ffffff", markeredgecolor="#333333", color="#333333", marker='o', markersize=5, zorder=100)
        # plt.plot(range(5), percentileRg_25, lw=1.5, markerfacecolor="#ffffff", markeredgecolor="#333333", color="#333333", marker='o', markersize=5, zorder=100)
        # plt.plot(range(5), percentileRg_95, lw=1.5, markerfacecolor="#ffffff", markeredgecolor="#333333", color="#333333", marker='o', markersize=5, zorder=100)
        plt.xlabel("\# routes")
        # plt.ylabel("Buffer index")
        plt.ylabel("Planning Time Index")
        plt.ylim(1, 8)  # 2.5
        plt.tight_layout()
        plt.savefig(dataPath + 'userData/Relaibility_PTI_routes_' + targetNames[i] + '.png', dpi=150)
        plt.savefig(dataPath + 'userData/Relaibility_PTI_routes_' + targetNames[i] + '.pdf')
        plt.close()



def relativeGapVSIncome():
    # load user income information
    inData = open(dataPath + "userData/userHomeLocationIncomeRoutes.csv", "r")
    inData.readline()
    userIncome = {}
    userIncomeLevel = {}
    for row in inData:
        row = row.rstrip().split(",")
        uid = int(row[0])
        income = float(row[6])
        if income < 26:
            userIncomeLevel[uid] = "Low"
        elif income < 78:
            userIncomeLevel[uid] = "Middle"
        else:
            userIncomeLevel[uid] = "High"
        userIncome[uid] = income

    userIncomeLevel_p = {}
    incomeQ1 = np.percentile(list(userIncome.values()), 25)
    incomeQ2 = np.percentile(list(userIncome.values()), 50)
    incomeQ3 = np.percentile(list(userIncome.values()), 75)
    for uid in userIncome:
        income = userIncome[uid]
        if income < incomeQ1:
            lev = "Q1"
        elif income < incomeQ2:
            lev = "Q2"
        elif income < incomeQ3:
            lev = "Q3"
        else:
            lev = "Q4"
        userIncomeLevel_p[uid] = lev

    
    # Rg during peak hours
    targetHours = [[7,8,9], [16,17,18]]  # am peak hours, pm peak hours
    targetNames = ["AM", "PM"]

    # trip information
    tripData = open(dataPath + "userData/userRoutesInf_complete.csv", 'r')
    tripData.readline()

    ttsInTimeSlot_weekday = [{} for i in range(24)]
    routesInTimeSlot_weekday = [{} for i in range(24)]
    
    ttsOfUsers = {}
    for row in tripData:
        row = row.rstrip().split(',')
        uid = int(row[0])
        tid = int(row[1])
        depTime = datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        arrTime = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
        depWeekday = depTime.weekday()
        depHour = depTime.hour
        # period = hourToPeroid[depHour]
        period = depHour

        tt = float(row[8])  # min
        routeID = int(row[9])
        if routeID == -1:
            continue
        # distance = haversine(sLat, sLon, tLat, tLon)
        # weekday only
        if depWeekday >=5:
            continue
        try:
            ttsOfUsers[uid].append(tt)
        except:
            ttsOfUsers[uid] = []
            ttsOfUsers[uid] = [tt]

        try:
            ttsInTimeSlot_weekday[period][uid].append(tt)
            routesInTimeSlot_weekday[period][uid].append(routeID)
        except:
            ttsInTimeSlot_weekday[period][uid] = [tt]
            routesInTimeSlot_weekday[period][uid] = [routeID]
    # for each user, find the minT as free flow travel time
    minTOfUsers = {}
    for uid in ttsOfUsers:
        minTOfUsers[uid] = np.min(ttsOfUsers[uid])

    RgsVSIncomes = []
    for i in range(len(targetHours)):
        hrs = targetHours[i]
        for hr in hrs:
            userTTs = ttsInTimeSlot_weekday[hr]
            for uid in userTTs:
                try:
                    income = userIncome[uid]
                    incomeLevel = userIncomeLevel[uid]
                except:
                    continue
                
                tts = userTTs[uid]
                # minT = np.min(tts)
                minT = np.percentile(tts, 10)
                numTrips = len(tts)
                if numTrips < 20:
                    continue
                Rgs = [(t-minT)/minT for t in tts]
                Rgs = [max(r, 0) for r in Rgs]
                minT = minTOfUsers[uid]
                PTI = np.percentile(tts, 85) / minT

                # bufferIndex = (np.percentile(tts, 85) - np.mean(tts)) /np.mean(tts)
                
                # should we use mean(Rgs) or std(Rgs) to reflect the reliability of travel time?
                # reliability = np.mean(bufferIndex)
                numRoutes = len(set(routesInTimeSlot_weekday[hr][uid]))
                numRoutes = min(5, numRoutes)
                RgsVSIncomes.append([targetNames[i], PTI, numRoutes, income, incomeLevel])
                # for r in Rgs:
                #     RgsVSIncomes.append([targetNames[i], r, numRoutes, income, incomeLevel])
    RgsVSIncomes_df = pd.DataFrame(RgsVSIncomes, columns=["Period", "Rg", "numRoutes", "income", "incomeLevel"])

    for i in range(len(targetHours)):
        subData = RgsVSIncomes_df[RgsVSIncomes_df["Period"] == targetNames[i]]

        '''
        # scatter plot
        # spatial density
        x = np.array(subData["income"])
        y = np.array(subData["Rg"])
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # sort the points by density
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # plot
        fig = plt.figure(figsize=(4, 3))
        plt.scatter(x, y, marker='o', s=5, c=z, lw=0, edgecolor='', alpha=0.5)
        # plt.xticks(range(4), ["AM", "MD", "PM", "RD"])
        plt.ylim(0, 3)
        plt.xlabel(r'Income', fontsize=14)
        plt.ylabel(r"Rg", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(dataPath + 'userData/Rgs_incomes_' + targetNames[i] + '.png', dpi=150)
        # plt.savefig(dataPath + 'userData/Rgs_incomes_' + targetNames[i] + '.pdf')
        plt.close()
        '''

        # box plot
        fig = plt.figure(figsize=(4, 3))
        ax = sns.boxplot(x='incomeLevel', y='Rg', data=subData, order=["Low", "Middle", "High"], fliersize=0)
        ax = sns.stripplot(x='incomeLevel', y='Rg', data=subData,  order=["Low", "Middle", "High"], color="orange", jitter=0.2, alpha=0.5, size=2.5)
        # ax = sns.stripplot(x='incomeLevel', y='Rg', data=subData, order=["Q1", "Q2", "Q3", "Q4"], color="orange", jitter=0.2, size=2.5)
        # plt.title("Boxplot with jitter", loc="left")
        plt.ylim(1, 8)
        plt.xlabel(r'Income level', fontsize=14)
        plt.ylabel(r"Ppanning Time Index", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(dataPath + 'userData/PTI_incomes_boxplot_' + targetNames[i] + '.png', dpi=150)
        # plt.savefig(dataPath + 'userData/Rgs_incomes_boxplot_' + targetNames[i] + '.pdf')
        plt.close()

        print(targetNames[i])
        print("Average PTI")
        print(np.mean(subData[subData["incomeLevel"]=="Low"]["Rg"]))
        print(np.mean(subData[subData["incomeLevel"]=="Middle"]["Rg"]))
        print(np.mean(subData[subData["incomeLevel"]=="High"]["Rg"]))

        '''
        # percentage of Rg > 1
        print("Percenrage of large Rg")
        lowRgs = subData[subData["incomeLevel"]=="Low"]["Rg"]
        middleRgs = subData[subData["incomeLevel"]=="Middle"]["Rg"]
        highRgs = subData[subData["incomeLevel"]=="High"]["Rg"]
        print(np.sum([1 if r > 1 else 0 for r in lowRgs]) / len(lowRgs))
        print(np.sum([1 if r > 1 else 0 for r in middleRgs]) / len(middleRgs))
        print(np.sum([1 if r > 1 else 0 for r in highRgs]) / len(highRgs))
        '''




def main():

    '''
    # # 3. Collect the raw data of users who we need their income information
    yearmonths = ['201611', '201612', '201701', '201702', '201703', '201704']
    for ym in yearmonths:
        extractUserData(ym)

    # # 4. combine the user data and sort by user and time using linux command (cat, sort)
    # the sorted data are saved in "userData_sorted.csv"

    # # 6. find the home location of each user for income estimation
    # # 6.1 find the stay points by clustering
    findStayPoints()

    # # 6.2 detect the stay region, and find the home location
    findHomeStayRegions()
    '''

    # # 6.3 combine with the census data, find the home tract and estimate the income
    # # the joyplot is done with R, see joyplot_income.R in the folder
    # findHomeTractAndIncome()

    # # 6.4 compare the income distribution of sample users and total population in the 4 zones
    # userIncomeComparison()

    # userHomeCompare()

    # # 7. analyze the routing behavior with income
    # for i in range(4):
        # userBehaviorVSIncome(zonePair=i)

    # # 8. number of routes in each hour
    # numRoutesInTime()

    relativeGap()
    # relativeGapVSNumRoutes()
    # relativeGapVSIncome()



if __name__ == '__main__':
    main()