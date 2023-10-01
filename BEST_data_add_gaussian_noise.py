from netCDF4 import Dataset
import numpy as np
from numpy.ma import getdata
import pandas as pd
from tqdm import tqdm
import os
from variables import *
from cdo import * 

'''
Script for generating BEST observational maps with the addition of 
noise sampled from a Gaussian distribution with 0 mean and stddev equal
to the uncertainty of BEST observational data
'''

cdo = Cdo()

first_year = 1979
last_year = 2022

BEST_data_directory = f'./Demo_download/Data/BEST_data'

# Read the new nc file with maps from 1979 to 2022
nc_data = Dataset(f'{BEST_data_directory}/BEST_regridded_annual_{first_year}-{last_year}.nc', mode='r+', format='NETCDF3_CLASSIC')

uncertainty_df = pd.read_csv(f'{BEST_data_directory}/Land_and_Ocean_global_average_annual.txt', header=None, delim_whitespace=True)

# Number of BEST datasets to generate for each model and for each scenario
n_datasets_per_model_per_scenario = 5

annual_uncertainties_list = list(uncertainty_df[uncertainty_df[0].between(first_year, last_year)][2])

# The file of uncertainties lacks values for 2019-2022 years. We assume the uncertainty in those years is equal to the uncertainty in 2018
annual_uncertainties_list.append(0.045) # 2019
annual_uncertainties_list.append(0.045) # 2020
annual_uncertainties_list.append(0.045) # 2021
if last_year == 2022:
    annual_uncertainties_list.append(0.045) # 2022

gaussian_noise_array = np.zeros((last_year-first_year+1, nc_data['st'].shape[1], nc_data['st'].shape[2]))

print('Computing noise and adding to the BEST maps...')
tas_maps_array = getdata(nc_data['st'][:])
lats = getdata(nc_data['lat'][:])
lons = getdata(nc_data['lon'][:])
times = getdata(nc_data['time'][:])
mean = 0
for idx_model, model in enumerate(tqdm(models_list, total=len(models_list))):
    for idx_short_scenario, short_scenario in enumerate(short_scenarios_list):
        for i in range(n_datasets_per_model_per_scenario):
            for sigma_idx, sigma in enumerate(annual_uncertainties_list):
                # To build the current noise map, sample the noise values from a gaussian distribution with 0 mean and stddev equal to the BEST data uncertainty for the current year
                gaussian_noise_array[sigma_idx,:,:] = np.random.normal(mean, sigma, (nc_data['st'].shape[1], nc_data['st'].shape[2])) 

                # Add the noise maps to the BEST maps
                noisy_tas_maps_array = tas_maps_array + gaussian_noise_array

                # Save the BEST maps with the noise added into a new nc file
                nc_new_filename = f'BEST_regridded_annual_{first_year}-{last_year}_Gaussian_noise_{model}_{short_scenario}_{i+1}.nc'
                    
                if not os.path.exists(f'{BEST_data_directory}/gaussian_noise_{n_datasets_per_model_per_scenario}'): os.makedirs(f'{BEST_data_directory}/gaussian_noise_{n_datasets_per_model_per_scenario}')

                nc_new = Dataset(f'{BEST_data_directory}/gaussian_noise_{n_datasets_per_model_per_scenario}/{nc_new_filename}', mode='w', format='NETCDF3_CLASSIC')

                lat_dim = nc_new.createDimension('lat', noisy_tas_maps_array.shape[1])     # latitude axis
                lon_dim = nc_new.createDimension('lon', noisy_tas_maps_array.shape[2])    # longitude axis
                time_dim = nc_new.createDimension('time', noisy_tas_maps_array.shape[0])

                lat = nc_new.createVariable('lat', np.float32, ('lat',))
                lat.units = 'degrees_north'
                lat.long_name = 'latitude'
                lon = nc_new.createVariable('lon', np.float32, ('lon',))
                lon.units = 'degrees_east'
                lon.long_name = 'longitude'
                time_var = nc_new.createVariable('time', np.int32, ('time',))
                time_var.units = 'year A.D.'
                time_var.long_name = 'Time'

                tas_var = nc_new.createVariable('st', np.float64, ('time','lat','lon')) # note: unlimited dimension is leftmost
                tas_var.missing_value = nc_data['st'].missing_value
                tas_var.standard_name = 'surface_temperature'
                tas_var.long_name = 'Air Surface Temperature'
                nc_new.variables['st'][:,:,:] = noisy_tas_maps_array[:,:,:]

                nc_new.variables['lat'][:] = lats
                nc_new.variables['lon'][:] = lons
                nc_new.variables['time'][:] = times

                nc_new.close()

nc_data.close()