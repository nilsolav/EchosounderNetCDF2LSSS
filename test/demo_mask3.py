# -*- coding: utf-8 -*-
"""
1) This work is based on the initial work that was created on Tue Aug 20 11:44:06 2019 by gavinj that generates masks
of 3 rectangles
2) Debugging were furthered by nils olav in july 2020
3) This work is then integrated with functions to post via LSSS API by yi liu in july 2020
"""

# Gavin/Nillav - Read netcdf file
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import cftime
import os
# Stuff to make the plots have nicely formated time axes
import datetime
import matplotlib.dates as mdates
import matplotlib.units as munits

# Yi - posting and integration
import os
import requests
import numpy as np
import scipy
import math

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

## Global variables, can be changed in docker

# baseUrl : local host of LSSS
baseUrl = 'http://localhost:8000'

# Name of the netcdf file - experimental mask data in NetCDF4 format
filenames = ['demo_mask.nc', '2007205-D20070421-T085415.nc', '2009107-D20090522-T040634.nc',
            'tokt2005114-D20051118-T062010.nc', 'tokt2006101-D20060124-T030844.nc']


''' 
Function areas
nc_reader : function that reads masks along with the attributes from nc file
post      : function that post the data (mask) onto LSSS given path and parameters
'''

def nc_reader_nilsolav(filename, is_save=True, is_show=False):

    with h5py.File(filename, 'r') as f:

        #Open the group where the data is located
        interp = f['Interpretation/v1']

        # Get some variables and attributes
        t = interp['mask_times']
        d = interp['mask_depths']

        d_units = str(interp['mask_depths'].attrs['units'], 'utf-8')

        t_units = str(interp['mask_times'].attrs['units'], 'utf-8')
        t_calendar = str(interp['mask_times'].attrs['calendar'], 'utf-8')

        c = interp['sound_speed'][()]
        c_units = str(interp['sound_speed'].attrs['units'], 'utf-8')

        bb_upper = interp['min_depth']
        bb_lower = interp['max_depth']
        bb_left = interp['start_time']
        bb_right = interp['end_time']

        region_id = interp['region_id']
        region_name = interp['region_name']
        r_type = interp['region_type']
        r_type_enum = h5py.check_dtype(enum=interp['region_type'].dtype)
        # Convert region types into a text version
        r_type_enum = dict(map(reversed, r_type_enum.items()))
        r_type_name = [r_type_enum[i] for i in r_type]


        cat_names = interp['region_category_names']
        cat_prop = interp['region_category_proportions']
        cat_ids = interp['region_category_ids']

        for i, r in enumerate(cat_names):
            print('Region ' + str(cat_ids[i]) + ' has category '
                  + '"' + cat_names[i] + '"'
                  + ' with proportion ' + str(cat_prop[i]))

        if is_save or is_show:
            # Plot the power of beam
            plt.figure()
            plt.clf()

            # plot masks
            for i, r in enumerate(d):
                for time, ranges in zip(t[i], r):
                    for start, stop in zip(ranges[0::2], ranges[1::2]):
                        plt.plot([time, time], [start, stop], linewidth=4, color='k')

            # plot bounding boxes
            for i, junk in enumerate(bb_upper):
                plt.plot([bb_left[i], bb_right[i], bb_right[i], bb_left[i], bb_left[i]],
                         [bb_lower[i], bb_lower[i], bb_upper[i], bb_upper[i], bb_lower[i]],
                         color=(0.5, 0.5, 0.5))
                plt.text(bb_left[i], bb_upper[i], 'ID: ' + str(region_id[i])
                        + ' (' + r_type_name[i] + ')',
                         bbox=dict(facecolor=(0.5, 0.5, 0.5), alpha=0.5))


            plt.title('Using c= ' + str(c) + ' ' + c_units)
            ax = plt.gca()
            ax.invert_yaxis()

            plt.xlabel('Time\n(' + t_units + ')')
            plt.ylabel('Depth (' + d_units + ')')

            plt.savefig(os.path.splitext(filename)[0]+'.png', bbox_inches='tight')
            if is_show:
                plt.show()

def nc_reader_yi(filename, is_save_png = True, is_show=False):

    with h5py.File(filename, 'r') as f:

        # Open the group where the data is located
        interp = f['Interpretation/v1']

        # Get some variables and attributes
        t = interp['mask_times']
        d = interp['mask_depths']

        d_units = str(interp['mask_depths'].attrs['units'], 'utf-8')

        t_units = str(interp['mask_times'].attrs['units'], 'utf-8')
        t_calendar = str(interp['mask_times'].attrs['calendar'], 'utf-8')

        c = interp['sound_speed'][()]
        c_units = str(interp['sound_speed'].attrs['units'], 'utf-8')

        bb_upper = interp['min_depth']
        bb_lower = interp['max_depth']
        bb_left = interp['start_time']
        bb_right = interp['end_time']
        region_id = interp['region_id']
        region_name = interp['region_name']
        r_type = interp['region_type']
        r_type_enum = h5py.check_dtype(enum=interp['region_type'].dtype)

        # Convert region types into a text version
        r_type_enum = dict(map(reversed, r_type_enum.items()))
        r_type_name = [r_type_enum[i] for i in r_type]

        # convert time variables into the form that matplotlib wants
        # current example .nc files actually have timestamps in 100 nanseconds since 1601.
        # But give the time units as milliseconds since 1601. Sort that out...
        time_fixer = 10000  # divide all times by this before using cftime.num2pydate()

        cat_names = interp['region_category_names']
        cat_prop = interp['region_category_proportions']
        cat_ids = interp['region_category_ids']

        for i, r in enumerate(cat_names):
            print('Region ' + str(cat_ids[i]) + ' has category '
                  + '"' + cat_names[i] + '"'
                  + ' with proportion ' + str(cat_prop[i]))

        if is_show or is_save_png:
            plt.figure()
            plt.clf()

            get_masks(d, t, d_units, t_units, time_fixer, handle=plt)
            get_region(region_id, bb_upper, bb_lower, bb_left, bb_right, r_type_name, time_fixer, handle=plt)

            # Plot the power of beam
            plt.title('Using c= ' + str(c) + ' ' + c_units)
            ax = plt.gca()
            ax.invert_yaxis()

            plt.xlabel('Time\n(' + t_units + ')')
            plt.ylabel('Depth (' + d_units + ')')

            if is_save_png:
                plt.savefig(os.path.splitext(filename)[0]+'.png', bbox_inches='tight')

            if is_show:
                plt.show()
        else:
            get_masks(d, t, d_units, t_units, time_fixer, handle=None)
            get_region(region_id, bb_upper, bb_lower, bb_left, bb_right, r_type_name, time_fixer, handle=None)

# get masks
def get_masks(d, t, d_units, t_units, time_fixer, handle = None):

    for i, r in enumerate(d):

        json_str = []

        for time, ranges in zip(t[i], r):

            min_max_vals = []

            for start, stop in zip(ranges[0::2], ranges[1::2]):

                min_max_vals.append({"min": float(start+50), "max": float(stop+57)})

                if handle:
                    handle.plot([time, time], [start, stop], linewidth=4, color='k')

            ping = int(time/1000) - 13189164000 # use 10000 when the time_fixer error is debugged


            json_str.append({"pingNumber": int(ping),
                             "depthRanges": min_max_vals})
                             # "depthRanges": [{"min": min(y), "max": y[ping]}]})

        print(json_str)

        if json_str:
            post('/lsss/module/PelagicEchogramModule/school-mask',
                        json = json_str)

# get region
def get_region(region_id, bb_upper, bb_lower, bb_left, bb_right, r_type_name, time_fixer, handle = None):

    for i, junk in enumerate(bb_upper):

        if handle:
            plt.plot([bb_left[i], bb_right[i], bb_right[i], bb_left[i], bb_left[i]],
                     [bb_lower[i], bb_lower[i], bb_upper[i], bb_upper[i], bb_lower[i]],
                     color=(0.5, 0.5, 0.5))

            plt.text(bb_left[i], bb_upper[i], 'ID: ' + str(region_id[i])
                    + ' (' + r_type_name[i] + ')',
                     bbox=dict(facecolor=(0.5, 0.5, 0.5), alpha=0.5))


def post(path, params=None, json=None, data=None):
    '''
    This is the basic function to post a mask/masks to LSSS via the API. Following
    is the parameters
    - path   : directory/category of a mask/masks to post to
    - params : parameters specifying the post process, default is None
    - json   : the mask is in jason format, in this case, is a list of dictionary for example
             [ {'pingNumber': 54001, 'depthRanges': [{'min': 30.6, 'max': 34.1}]},
               {'pingNumber': 54002, 'depthRanges': [{'min': 30.6, 'max': 37.5}]},
               {'pingNumber': 54003, 'depthRanges': [{'min': 30.6, 'max': 39.9}]},
                ...
             ]
             Note that
                * you can also post multiple "min"/"max" ranges in case the mask is not a rectangle
                * 'pingNumber' should be continuous integer, otherwise, the posted results will be multiple polygons
                * 'depthRange' can be float value or integer

    - data   : data (usually a ping) to post to the LSSS GUI, which is slow thus not recommended when the data is big.
               the same situation for the request command.

    '''

    # Make sure the LSSS server is on then the url points to the target directory of masks/areas/schools/layers
    url = baseUrl + path

    # Connect to the server and post the content
    response = requests.post(url, params=params, json=json, data=data)


    # Check the feedback from LSSS API
    if response.status_code == 200:
        return response.json()
    if response.status_code == 204:
        return None
    raise ValueError(url + ' returned status code ' + str(response.status_code) + ': ' + response.text)

if __name__ == "__main__":
    # Batch check all nc files
    # for i, f in enumerate(filenames):
    #     # print(i, f, '\n')
    #     nc_reader_yi(f, is_save=False, is_show=False)


    # Check individual nc file
    print('Files: ', filenames)

    nc_reader_yi(filenames[0], is_save_png=False, is_show=True)


