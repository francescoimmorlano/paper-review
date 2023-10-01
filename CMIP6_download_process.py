import os
from os.path import exists
from turtle import down
import cdsapi
from cdo import *
import zipfile
from shutil import copy2, rmtree
from lib import *
from variables import *

'''
Script for downloading CMIP6 data from Climate Data Store and processing them
'''

t_resolution = 'monthly'
experiments = ['historical', 'ssp2_4_5']
level = 'single_levels'
data_format = 'zip'

c = cdsapi.Client()
cdo = Cdo()

CMIP6_directory = './Demo_download/Data/CMIP6_data'
download_directory = f'{CMIP6_directory}/{variable}/Monthly'
extracted_files_directory = f'{CMIP6_directory}/{variable}/Monthly/extracted'
uniformed_files_directory = f'{CMIP6_directory}/{variable}/Monthly_uniform'
remapped_files_directory = f'{CMIP6_directory}/{variable}/Monthly_uniform_remapped'
annual_remapped_files_directory = f'{CMIP6_directory}/{variable}/Annual_uniform_remapped'
if not os.path.exists(extracted_files_directory): os.makedirs(extracted_files_directory)
if not os.path.exists(uniformed_files_directory): os.makedirs(uniformed_files_directory)
if not os.path.exists(remapped_files_directory): os.makedirs(remapped_files_directory)
if not os.path.exists(annual_remapped_files_directory): os.makedirs(annual_remapped_files_directory)

# CMIP6 files download
for model in models_short_list:
    for experiment in experiments:
        download_file = f'{variable_short}-{model}-{experiment}.{data_format}'
        if os.path.exists(f'{download_directory}/{download_file}'):
            continue
        else:
            retrieve_cds_cmip6(c, variable, variable_short, model, experiment, level, t_resolution, data_format, download_directory)

# CMIP6 files extraction
zip_files_list = os.listdir(download_directory)
zip_files_list.sort()
print(f'\nExtracted files:')
for zip_file in zip_files_list:
    if (not zip_file.endswith(data_format)):
        continue
    print(zip_file)

    directory_to_extract_file = '{}/{}'.format(extracted_files_directory, zip_file.split('.')[0])
    if os.path.isdir(directory_to_extract_file):
        continue
    with zipfile.ZipFile(f'{download_directory}/{zip_file}', 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_file)

# Check consistency of ensemble member (r1i1p1f1) between historical and SSP simulations for each model and scenario
for model in models_short_list:
    for experiment in experiments:
        simulation_directory = f'{extracted_files_directory}/{variable_short}-{model}-{experiment}'
        simulation_files = os.listdir(simulation_directory)
        
        if experiment == 'historical':
            for simulation_file in simulation_files:
                if simulation_file.endswith('.nc'):
                    historical_ensemble = simulation_file.split('_')[4]
                    break
        else:
            for simulation_file in simulation_files:
                if simulation_file.endswith('.nc'):
                    ensemble = simulation_file.split('_')[4]
                    if historical_ensemble == ensemble:
                        continue
                    else:
                        print(f'Not matching Model: {model}\nHistorical ensemble: {historical_ensemble}\n{experiment} ensemble: {ensemble}\n')
                        break

# Generate one nc file per simulation containing all the monthly maps
print('\n')
for model in models_short_list:
    for experiment in experiments:
        simulation_directory = f'{extracted_files_directory}/{variable_short}-{model}-{experiment}'
        simulation_files_list = os.listdir(simulation_directory)
        
        nc_files_list = [file for file in simulation_files_list if file.endswith('.nc')]
        nc_files_list.sort()
        
        if len(nc_files_list) > 1:
            simulation_first_filename_split = nc_files_list[0].split('_')
            
            simulation_start_time = simulation_first_filename_split[6][:6]
            simulation_end_time = nc_files_list[-1].split('_')[6][-6:]
            
            simulation_first_filename_split[6] = f'{simulation_start_time}-{simulation_end_time}'
            
            simulation_filename = '_'.join(simulation_first_filename_split)
            
            if not exists(f'{uniformed_files_directory}/{simulation_filename}'):
                print(f'Merging monthly maps of {simulation_filename} in one nc file...')
                cdo.mergetime(input=f'{extracted_files_directory}/{variable_short}-{model}-{experiment}/*.nc', output=f'{uniformed_files_directory}/{simulation_filename}')
        else:
            src_file = f'{extracted_files_directory}/{variable_short}-{model}-{experiment}/{nc_files_list[0]}'
            dst_directory = uniformed_files_directory
            if not exists(f'{dst_directory}/{nc_files_list[0]}'):
                print(f'Copying {nc_files_list[0]} nc file...')
                copy2(src_file, dst_directory)

# Remap all the simulation maps to the grid of CanESM5-CanOE
print('\n')
simulations_list = os.listdir(uniformed_files_directory)
target_grid = 'CanESM5-CanOE_grid'
for simulation in simulations_list:
    if (not simulation.endswith('.nc')):
        continue

    if (target_grid == 'CanESM5-CanOE_grid' and simulation.split('_')[2] == 'CanESM5-CanOE'):
        copy2(f'{uniformed_files_directory}/{simulation}', remapped_files_directory) 
        print(f'Copying {simulation} ...')
        continue

    if (not exists(f'{remapped_files_directory}/{simulation}')):
        print(f'Remapping {simulation} ...')
        cdo.remapcon(target_grid, input=f'{uniformed_files_directory}/{simulation}', output=f'{remapped_files_directory}/{simulation}')

# Compute average of monthly maps in time to get annual maps for each nc file
print('\n')
monthly_nc_files_list = os.listdir(remapped_files_directory)
monthly_nc_files_list.sort()
for monthly_nc_file in monthly_nc_files_list:
    if(not monthly_nc_file.endswith('.nc')):
        continue
    
    if (os.path.exists(f'{annual_remapped_files_directory}/{monthly_nc_file}')):
        continue
    
    print(f'Computing average in time of {monthly_nc_file} ...')
    cdo.yearmonmean(input=f'{remapped_files_directory}/{monthly_nc_file}', output=f'{annual_remapped_files_directory}/{monthly_nc_file}')

rmtree(download_directory)
rmtree(uniformed_files_directory)
rmtree(remapped_files_directory)