from netCDF4 import Dataset
import numpy as np
from cdo import *
import os
from variables import *

'''
Script for processing BEST observational maps
'''

cdo = Cdo()

final_year = 2022

BEST_data_directory = './Demo_download/Data/BEST_data'

# Regrid BEST files to match the grid of CanESM5-CanOE
print('Regridding to CanESM5-CanOE grid... (it can take a few minutes)\n')
if not os.path.exists(f'{BEST_data_directory}/BEST_regridded.nc'):
    cdo.remapcon('CanESM5-CanOE_grid', input=f'{BEST_data_directory}/BEST_{final_year}.nc', output=f'{BEST_data_directory}/BEST_regridded.nc')
# Get absolute temperature values by adding the monthly climatology to anomalies values
nc_best_data = Dataset(f'{BEST_data_directory}/BEST_regridded.nc', mode='r+', format='NETCDF3_CLASSIC')
absolute_temperature_array = np.zeros((nc_best_data['temperature'].shape[0], nc_best_data['temperature'].shape[1],  nc_best_data['temperature'].shape[2]))
month = 0
for i in range(nc_best_data['temperature'].shape[0]):
    absolute_temperature_array[i,:,:] = nc_best_data['temperature'][i,:,:] + nc_best_data['climatology'][month,:,:]
    month +=1
    if (month == 12):
        month = 0
# Compute annual BEST maps by averaging in time monthly maps for the corresponding year
print('Computing annual average maps...\n')
annual_data = np.zeros((final_year-1850+1, nc_best_data['temperature'].shape[1], nc_best_data['temperature'].shape[2]))
annual_data_time = np.zeros((final_year-1850+1))
for i in range(annual_data.shape[0]):
    annual_data[i,:,:] = absolute_temperature_array[i*12:(i+1)*12, :, :].mean(axis=0)
    annual_data_time[i] = nc_best_data['time'][i*12]

# Convert absolute temperature values from Celsius to Kelvin
annual_data[:,:,:] += 273.15

# Save the BEST maps with annual resolution
nc_new_filename = 'BEST_regridded_annual.nc'
        
nc_new = Dataset(f'{BEST_data_directory}/{nc_new_filename}', mode='w', format='NETCDF3_CLASSIC')

lats = nc_best_data['lat'][:]
lons = nc_best_data['lon'][:]

time_dim = nc_new.createDimension('time', annual_data.shape[0])
lat_dim = nc_new.createDimension('lat', annual_data.shape[1])     # latitude axis
lon_dim = nc_new.createDimension('lon', annual_data.shape[2])    # longitude axis

lat = nc_new.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = nc_new.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'

time_var = nc_new.createVariable('time', np.int32, ('time',))
time_var.units = 'year A.D.'
time_var.long_name = 'Time'

tas_var = nc_new.createVariable('st', np.float32, ('time','lat','lon')) # note: unlimited dimension is leftmost
tas_var.missing_value = nc_best_data['temperature'].missing_value
tas_var.standard_name = 'surface_temperature'
tas_var.long_name = 'Air Surface Temperature'
nc_new.variables['st'][:,:,:] = annual_data[:,:,:]

nc_new.variables['lat'][:] = lats
nc_new.variables['lon'][:] = lons
nc_new.variables['time'][:] = annual_data_time

nc_best_data.close()
nc_new.close()

# Delete maps related to years out of 1979-2022 time period as they lack values in some grid points 
print('Deleting maps related to years out of 1979-2022 time period...')
if not os.path.exists(f'{BEST_data_directory}/BEST_regridded_annual_1979-{final_year}.nc'):
    cdo.selyear('1979/2022', input=f'{BEST_data_directory}/BEST_regridded_annual.nc', output=f'{BEST_data_directory}/BEST_regridded_annual_1979-{final_year}.nc')

os.remove(f'{BEST_data_directory}/BEST_regridded.nc')
os.remove(f'{BEST_data_directory}/BEST_regridded_annual.nc')