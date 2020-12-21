# -*- coding: utf-8 -*-
__author__ = 'xu'

import os, sys
import pickle, csv, gzip
import time, datetime, pytz
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import vaex

import geojson

from joblib import Parallel, delayed
import multiprocessing

from rtree import index
from scipy import spatial

from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union
import fiona
import itertools


cityBoundary = [-98.068, -95.858, 31.709, 33.434]
# cityBoundary_forPlot = [-98.068, -95.858, 31.709, 33.434]

# cityBoundary = [-98.065, -95.855, 31.710, 33.430]
cityBoundary_forPlot = [-98.0, -96.0, 32.0, 33.5]

# global variables
DallasTZ = pytz.timezone('US/Central')

if os.path.exists('/media/xu/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'):
    dataPath = '/media/xu/TOSHIBA EXT/Study/HuMNetLab/Data/Dallas/'
if os.path.exists('/home/xu/Data/Dallas/'):
    dataPath = '/home/xu/Data/Dallas/'
if os.path.exists('/home/xu/Documents/Data/Dallas/'):
    dataPath = '/home/xu/Documents/Data/Dallas/'


# calculate distance between two locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c



def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def readZipData(ym):
    allDays = []
    for day in range(1, 32):
        dd = ym + str(day).zfill(2)
        try:
            daytime = datetime.datetime.strptime(dd, '%Y%m%d')
        except:
            continue
        allDays.append(dd)

    print("# of days : ", len(allDays))

    userList = set()
    userRecords = {}
    userStartTime = {}
    userEndTime = {}

    rawDataPath = "/home/rdicle/Cintra/CLEAN/DALLAS/DAYS/"

    for day in allDays:
        rawData = gzip.open(rawDataPath + str(day) + "00.gz", 'rb')

        file_content = rawData.read()
        file_content = file_content.split('\n')

        count = 0
        for row in file_content:
            count += 1
            if count%1e5==0:
                print(day, count)
            row = row.split(',')
            try:
                user = int(row[0])
                ts = int(row[1])
            except:
                continue

            userList.add(user)
            try:
                userRecords[user] += 1
            except:
                userRecords[user] = 1
            if user not in userStartTime:
                userStartTime[user] = ts
            if user not in userEndTime:
                userEndTime[user] = ts

            # update start and end time
            if ts < userStartTime[user]:
                userStartTime[user] = ts
            if ts > userEndTime[user]:
                userEndTime[user] = ts
        rawData.close()

    print("# of users : ", len(userList))
    # save data
    pickle.dump([userList, userRecords, userStartTime, userEndTime],
                open(dataPath + "userData/userInf_" + ym + ".pkl", 'wb'),
                pickle.HIGHEST_PROTOCOL)



def readZipDataParallel():
    numOdThreads = 6
    yearmonth = ['201611', '201612', '201701', '201702', '201703', '201704']
    res = Parallel(n_jobs=numOdThreads)(delayed(readZipData)(month) for month in yearmonth)

    print("Done")


def userSelection():
    yearmonth = ['201611', '201612', '201701', '201702', '201703', '201704']
    # yearmonth = ['201612']
    allUserList = set()
    allUserRecords = {}
    allUserStartTime = {}
    allUserEndTime = {}
    for ym in yearmonth:
        print("Loading ", ym)
        userList, userRecords, userStartTime, userEndTime = pickle.load(open(dataPath + "userData/userInf_" + ym + ".pkl", 'rb'))
        allUserList = allUserList.union(userList)
        print(len(userList), len(userRecords), len(userStartTime), len(userEndTime))
        for user, record in userRecords.iteritems():
            try:
                allUserRecords[user] += record
            except:
                allUserRecords[user] = record

        for user in userStartTime:
            start = userStartTime[user]
            if user not in allUserStartTime:
                allUserStartTime[user] = start
            if start < allUserStartTime[user]:
                allUserStartTime[user] = start

        for user in userEndTime:
            end = userEndTime[user]
            if user not in allUserEndTime:
                allUserEndTime[user] = end
            if end > allUserEndTime[user]:
                allUserEndTime[user] = end

    print("# of users : ", len(allUserList))

    # plot the distribution of number of records and the timespan
    maxDays = 200
    maxLogRecords = 7
    numRecords = []
    timespan = []
    plotMatrix = np.zeros((maxDays, 120))
    recordsLogInterval = 0.05
    dayInterval = 1
    allSpanIdx = set()
    allRecordsIdx = set()

    errUser = 0
    for user in list(allUserList):
        try:
            span = (allUserEndTime[user] - allUserStartTime[user])/(3600*24) + 1  # days
            records = np.log10(allUserRecords[user])
        except:
            errUser += 1
            continue
        if span>maxDays or records > maxLogRecords:
            continue
        numRecords.append(records)
        timespan.append(span)

        spanIdx = span/dayInterval
        recordIdx = int(records/recordsLogInterval)
        allSpanIdx.add(spanIdx)
        allRecordsIdx.add(recordIdx)
        try:
            plotMatrix[spanIdx, recordIdx] += 1
        except:
            continue

    print("# of error users : ", errUser)

    print(allSpanIdx)
    print(allRecordsIdx)

    cmap = plt.get_cmap('YlOrRd')
    # cmap = truncate_colormap(cmap, 0.2, 1.0)

    # cmap = viridis_white_r

    # plot
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot()
    h = ax.hist2d(numRecords, timespan, bins=(200,160), norm=colors.LogNorm(), cmap=cmap)
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# users', rotation=270)
    # ax.set_title("Time span VS # records")

    # fig.patches.extend([plt.Rectangle((0.4, 0.5), 0.45, 0.45,
    #                                   fill=True, color='k', alpha=0.25, zorder=10,
    #                                   transform=fig.transFigure, figure=fig)])

    plt.xlim(1.2, maxLogRecords)
    plt.ylim(1, 180)
    plt.xlabel("log10 #records")
    plt.ylabel("timespan (days)")
    plt.tight_layout()
    plt.savefig(dataPath + "userData/userTimespan_records.png", dpi=300)
    plt.savefig(dataPath + "userData/userTimespan_records.pdf")
    plt.close()


    '''
    # refine plotMatrix
    plotMatrix[plotMatrix < 10] = 0
    fig = plt.figure()
    ax = plt.subplot()
    im = ax.imshow(plotMatrix, cmap=cmap, origin='lowest', aspect='auto', norm=colors.LogNorm())

    cbar = fig.colorbar(im)
    # cbar.ax.minorticks_off()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# users', rotation=270)
    ax.set_title("Time span VS # records")
    plt.xlim(20, 120)
    plt.ylim(1,160)
    xt = [20,40,60,80,100,120]
    xt_new = [int(i*recordsLogInterval) for i in xt]
    plt.xticks(xt, xt_new)
    plt.xlabel("log10 #records")
    plt.ylabel("timespan (days)")
    plt.tight_layout()
    plt.savefig(dataPath + "Trips_all/userTimespan_records_2.png", dpi=300)
    plt.close()
    '''

    # select users by timespan and the number of records
    timespanThres = 60
    numRecordsThres = 3.0  # more than 1000 records

    selectedUsers = set()
    selectedRecords = 0
    for user in list(allUserList):
        try:
            span = (allUserEndTime[user] - allUserStartTime[user])/(3600*24) + 1  # days
            records = np.log10(allUserRecords[user])
        except:
            errUser += 1
            continue
        if span>maxDays or records > maxLogRecords:
            continue

        if span > timespanThres and records > numRecordsThres:
            selectedUsers.add(user)
            selectedRecords += allUserRecords[user]

    print("# of selected users %d / %d : %.2f" % (len(selectedUsers), len(allUserList), len(selectedUsers)/float(len(allUserList))))
    print("# of selected records: %.2f " % (float(selectedRecords)/np.sum(allUserRecords.values())))
    print(selectedRecords, np.sum(allUserRecords.values()))

    # save
    pickle.dump(selectedUsers, open(dataPath + "userData/selectedUsers.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)


def dataSelection(ym):
    selectedUsers = pickle.load(open(dataPath + "userData/selectedUsers.pkl", 'rb'))
    # load the selected zipcodes
    selectedZipcodes = set()
    geoFile = open(dataPath + "Geo/boundary/zipcodes_selected.geojson", 'rb')
    geoData = geojson.load(geoFile)
    for t in geoData['features']:
        zipcode = int(t['properties']['GEOID10'])
        selectedZipcodes.add(zipcode)
    geoFile.close()

    print("# of zipcodes in the study region : ", len(selectedZipcodes))

    # from 20161101 to 20170430
    allDays = []
    for day in range(1, 32):
        dd = ym + str(day).zfill(2)
        try:
            daytime = datetime.datetime.strptime(dd, '%Y%m%d')
        except:
            continue
        allDays.append(dd)

    print("# of days : ", len(allDays))

    rawDataPath = "/home/rdicle/Cintra/CLEAN/DALLAS/DAYS/"
    # rawDataPath = dataPath + "RawData/"

    outData = open(dataPath + "userData/RawData_" + ym + ".csv", "wb")

    for day in allDays:
        rawData = gzip.open(rawDataPath + str(day) + "00.gz", 'rb')

        file_content = rawData.read()
        file_content = file_content.split('\n')

        count = 0
        for row in file_content:
            count += 1
            if count % 1e5 == 0:
                print(day, count)
            row = row.split(',')
            try:
                user = int(row[0])
                acc = float(row[4])
                zipcode = int(row[5])
            except:
                continue
            if user not in selectedUsers:
                continue
            if zipcode not in selectedZipcodes:
                continue
            if acc > 150:
                continue

            # save the row
            row.pop(4)
            outData.writelines(','.join(row) + "\n")

        rawData.close()

    outData.close()


def dataSelectionParallel():
    numOdThreads = 4
    yearmonth = ['201611', '201612', '201701', '201702']
    res = Parallel(n_jobs=numOdThreads)(delayed(dataSelection)(month) for month in yearmonth)

    print("Done")


# combine the trips in four months and update the trip id
def combineTrips():
    yearmonth = ['201611', '201612', '201701', '201702', '201703', '201704']

    tripNum = 0
    preTrip = ''
    trip = []
    outData = open(dataPath + "userData/allTrips_raw.csv", "wb")

    for ym in yearmonth:
        inData = open(dataPath + "userData/RawData_" + ym + "_trips.csv", 'rb')
        for row in inData:
            row = row.rstrip().split(',')
            user = row[0]
            time = row[1]
            lon = row[2]
            lat = row[3]
            tripId = row[0] + '-' + row[4]

            if preTrip=='':
                preTrip = tripId

            if tripId != preTrip:
                # save the last trip
                for t in trip:
                    row_save = [t[0], str(tripNum)] + t[1:]
                    outData.writelines(','.join(row_save) + "\n")
                tripNum += 1

                # clear
                trip = []
                preTrip = tripId

            trip.append([user, time, lon, lat])
        # last trip
        for t in trip:
            row_save = [t[0], str(tripNum)] + t[1:]
            outData.writelines(','.join(row_save) + "\n")
        tripNum += 1

        # clear
        trip = []
        preTrip = tripId

        inData.close
    outData.close()

    print("# trips : ", tripNum)



# plot the raw data with vaex
def plotPresenceData(yearmonth='201611'):
    plt.rcParams.update({'font.size': 16})

    inFile = dataPath + "userData/RawData_" + yearmonth + "_trips.csv"

    outFile = dataPath + "userData/RawData_" + yearmonth + "_trips_7days.csv"

    '''
    inData = open(inFile, 'rb')
    outData = open(outFile, 'wb')
    for row in inData:
        r = np.random.rand()
        if r <= 7 / 30.0:
            outData.writelines(row)
    inData.close()
    outData.close()
    '''

    allData = vaex.from_csv(outFile, names=["vehID", "depTime", "lon", "lat", "tripID"])
    print(allData.columns.values)
    print(len(allData))

    allData.plot(allData.lon, allData.lat, f='log', shape=512)
    plt.xlim(cityBoundary_forPlot[0], cityBoundary_forPlot[1])
    plt.ylim(cityBoundary_forPlot[2], cityBoundary_forPlot[3])
    # xlist = np.linspace(108.91, 108.99, 5)
    # plt.xticks(xlist, ["%.2f" % i for i in xlist], fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Longitude", fontsize=16)
    plt.ylabel("Latitude", fontsize=16)
    # plt.ticklabel_format(axis='both', style='plain', scilimits=None)
    # plt.show()
    plt.savefig(dataPath + "userData/presenceHeatmap_" + yearmonth + ".png", dpi=300)
    plt.savefig(dataPath + "userData/presenceHeatmap_" + yearmonth + ".pdf")
    plt.close()


def plotSampleRate():
    inFile = dataPath + "userData/rawSample_sort.csv"
    inData = open(inFile, 'rb')
    sampleRates = []
    preUser = ""
    tsList = []
    count = 0
    for row in inData:
        count += 1
        row = row.rstrip().split(',')
        if len(row) != 6:
            continue
        user = row[0]
        ts = int(row[1])
        if preUser=="":
            preUser = user

        if preUser != user:
            sr = [tsList[i]-tsList[i-1] for i in range(1,len(tsList))]
            sr = [i for i in sr if i > 0]
            sampleRates.extend(sr)

            tsList = []
            preUser = user

        tsList.append(ts)

    sr = [tsList[i] - tsList[i - 1] for i in range(1, len(tsList))]
    sr = [i for i in sr if i > 0]
    sampleRates.extend(sr)

    inData.close()

    # plot the distribution of sample rates
    interval = 10
    bins = np.linspace(0, 1200, 121)
    usagesHist = np.histogram(np.array(sampleRates), bins)
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    print(usagesHist)

    bins = np.array(bins[:-1])

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
            edgecolor='k', label='Sample users')
    # plt.plot((bins + 0.5 * interval).tolist(), usagesHist.tolist(), marker='o', markersize=5, linewidth=0.5,
    #          markerfacecolor=None, markeredgecolor='#cb181d')

    plt.xlim(0, 1210)
    # plt.ylim(0, 0.05)
    plt.xticks(range(0, 1201, 120), range(0, 21, 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale("log", nonposy='clip')
    plt.xlabel(r'Sample interval [min]', fontsize=12)
    plt.ylabel(r"Fraction", fontsize=12)
    # plt.title("From zone %d to zone %d" % (startZone, targetZone), fontsize=16)
    # plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(dataPath + 'userData/sampleRate_distribution.png', dpi=150)
    plt.savefig(dataPath + 'userData/sampleRate_distribution.pdf')
    plt.close()



def main():

    # # 1. Read the raw data in parallel, extract the user information (user id, number of records, start time, end time)
    # # The raw data are stored into a series of gzipped files, and one folder one day.
    # # The folder name refers to the date of the records.
    # # We first uncompress the raw data and combine the data per day.
    readZipDataParallel()

    # # 2. select the users by timespan and number of records
    userSelection()

    # # 3. select data of users in parallel
    dataSelectionParallel()
    combineTrips()

    # # 4. plot the presence healmap
    plotPresenceData(yearmonth='201611')

    # # 5. distribution of sample rate
    plotSampleRate()


if __name__ == '__main__':

    main()
