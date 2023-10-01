import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import matplotlib.transforms as mtransforms

"""
Script for reproducing Extended Data Figure 2
"""

models_list = [
        'ACCESS-CM2',
        'AWI-CM-1-1-MR',
        'BCC-CSM2-MR',
        'CAMS-CSM1-0',
        'CanESM5-CanOE',
        'CMCC-CM2-SR5',
        'CNRM-CM6-1',
        'CNRM-ESM2-1',
        'FGOALS-f3-L',
        'FGOALS-g3',
        'GFDL-ESM4',
        'IITM-ESM',
        'INM-CM4-8',
        'INM-CM5-0',
        'IPSL-CM6A-LR',
        'KACE-1-0-G',
        'MIROC6',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0',
        'NorESM2-MM',
        'TaiESM1',
        'UKESM1-0-LL'
        ]

short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
variable_short = 'tas'

ROOT_DATA = '../Source_data'
SIMULATIONS_DIRECTORY = f'{ROOT_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'
PATH_BEST_DATA = f'{ROOT_DATA}/BEST_data/BEST_regridded_annual_1979-2022.nc'

n_BEST_datasets_per_model_scenario = 5

start_year_training = 1979
end_year_training = 2022
n_training_years = end_year_training-start_year_training+1

start_year_test = end_year_training+1
end_year_test = 2098
n_test_years = end_year_test-start_year_test+1

""" Load DNNs predictions """
predictions = np.zeros((n_BEST_datasets_per_model_scenario, len(models_list), len(short_scenarios_list), n_training_years+n_test_years, 64, 128))
for model_idx, model in tqdm(enumerate(models_list), total=len(models_list)):
    for scenario_idx, scenario_short in enumerate(short_scenarios_list):
        for i in range(5):
            TRAIN_SET_PREDICTIONS_DIRECTORY = f'{ROOT_DATA}/Transfer_Learning_on_Observations/Training_set_predictions/tas_{model}_{scenario_short}_{i+1}'
            TEST_SET_PREDICTIONS_DIRECTORY = f'{ROOT_DATA}/Transfer_Learning_on_Observations/Test_set_predictions/tas_{model}_{scenario_short}_{i+1}'
            # Training set predictions
            model_train_set_predictions_filenames_list = os.listdir(TRAIN_SET_PREDICTIONS_DIRECTORY)
            model_train_set_predictions_filenames_list = [fn for fn in model_train_set_predictions_filenames_list if (fn.endswith('.csv'))]
            model_train_set_predictions_filenames_list.sort()
            model_train_set_prediction_array = np.zeros((n_training_years, 64, 128))
            for mp_idx, mp_filename in enumerate(model_train_set_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TRAIN_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_train_set_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
            predictions[i,model_idx,scenario_idx,:n_training_years,:,:] = model_train_set_prediction_array
            # Test set predictions
            model_test_set_predictions_filenames_list = os.listdir(TEST_SET_PREDICTIONS_DIRECTORY)
            model_test_set_predictions_filenames_list = [fn for fn in model_test_set_predictions_filenames_list if (fn.endswith('.csv'))]
            model_test_set_predictions_filenames_list.sort()
            model_test_set_prediction_array = np.zeros((n_test_years, 64, 128))
            for mp_idx, mp_filename in enumerate(model_test_set_predictions_filenames_list):
                if (not mp_filename.endswith('.csv')):
                    continue
                file = open(f'{TEST_SET_PREDICTIONS_DIRECTORY}/{mp_filename}')
                model_test_set_prediction_array[mp_idx,:,:] = np.loadtxt(file, delimiter=',')
            predictions[i,model_idx,scenario_idx,n_training_years:,:,:] = model_test_set_prediction_array[:,:,:]

""" Load CMIP6 ESMs simulations """
simulation_array = np.zeros((len(models_list), len(short_scenarios_list), 2098-1850+1, 64, 128))
for model_idx, model in tqdm(enumerate(models_list), total=len(models_list)):
    for scenario_idx, scenario_short in enumerate(short_scenarios_list):
        simulations_files_list = os.listdir(SIMULATIONS_DIRECTORY)
        simulations_files_list.sort()
        matching_simulations = [simulation_file for simulation_file in simulations_files_list if ((model in simulation_file and 'historical' in simulation_file)
                                                                                                or (model in simulation_file and scenario_short in simulation_file))]
        # maching_simuations[0] is the historical and matching_simulations[1] is the SSP simulation because of the sort operation
        # (for each model, the first simulation is the historical and then the SSP) 
        nc_historical_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[0]}', mode='r+', format='NETCDF3_CLASSIC')
        nc_ssp_data = Dataset(f'{SIMULATIONS_DIRECTORY}/{matching_simulations[1]}', mode='r+', format='NETCDF3_CLASSIC')
        n_historical_years = nc_historical_data[variable_short].shape[0]
        n_ssp_years = nc_ssp_data[variable_short].shape[0]
        n_lats = nc_ssp_data['lat'].shape[0]
        n_lons = nc_ssp_data['lon'].shape[0]
        simulation_array[model_idx,scenario_idx,:n_historical_years] = nc_historical_data[variable_short][:]
        if (n_ssp_years == 86):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-2]
        elif (n_ssp_years == 85):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:-1]
        elif (n_ssp_years == 84):
            simulation_array[model_idx,scenario_idx,n_historical_years:] = nc_ssp_data[variable_short][:]
        nc_historical_data.close()
        nc_ssp_data.close()


""" Load BEST observational data """
nc_BEST_data = Dataset(f'{PATH_BEST_DATA}', mode='r+', format='NETCDF3_CLASSIC')
n_BEST_years = nc_BEST_data['st'].shape[0]
n_lats = nc_BEST_data['lat'].shape[0]
n_lons = nc_BEST_data['lon'].shape[0]
lats = np.ma.getdata(nc_BEST_data['lat'][:])
lons = np.ma.getdata(nc_BEST_data['lon'][:])
BEST_data_array = np.zeros((n_BEST_years, n_lats, n_lons))
BEST_data_array[:,:,:] = nc_BEST_data['st'][:,:,:]
nc_BEST_data.close()

# Compute average surface air temperature maps across DNNs predictions and CMIP6 ESMs simulations
avg_predictions_maps = np.mean(predictions, axis=(0,1))
avg_simulations_maps = np.mean(simulation_array, axis=0)

# Compute average surface air temperature maps in 1980-2020
avg_predictions_maps_1980_2020 = np.mean(avg_predictions_maps[:,1980-1979:2021-1979,:,:], axis=1)
avg_simulations_maps_1980_2020 = np.mean(avg_simulations_maps[:,1980-1979:2021-1979,:,:], axis=1)

# Conver from Celsius to Kelnvi degrees
avg_predictions_maps_1980_2020_C = avg_predictions_maps_1980_2020 - 273.15
avg_simulations_maps_1980_2020_C = avg_simulations_maps_1980_2020 - 273.15

""" Plot """
font = {'fontname':'Arial'}
size_suptitlefig = 47
size_title_fig = 41
size_subplots_letters = 40
size_title_grid_spec = 39
size_title_axes = 33
size_lat_lon_coords = 24
size_colorbar_labels = 33
size_colorbar_ticks = 29
colormap = 'seismic'

fig, big_axes = plt.subplots(figsize=(40, 35) , nrows=3, ncols=1) 
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
big_axes[0].text(-0.03, 1.2, 'a', transform=big_axes[0].transAxes + trans,
        fontsize=size_subplots_letters, verticalalignment='top', fontfamily='arial', fontweight='bold')
big_axes[1].text(-0.03, 1.2, 'b', transform=big_axes[1].transAxes + trans,
        fontsize=size_subplots_letters, verticalalignment='top', fontfamily='arial', fontweight='bold')
big_axes[2].text(-0.03, 1.2, 'c', transform=big_axes[2].transAxes + trans,
        fontsize=size_subplots_letters, verticalalignment='top', fontfamily='arial', fontweight='bold')

for scenario_short_idx, scenario_short in enumerate(short_scenarios_list):
    scenario = f'SSP{scenario_short[-3]}-{scenario_short[-2]}.{scenario_short[-1]}'

    min_value = np.concatenate((avg_predictions_maps_1980_2020_C[scenario_short_idx,:,:], avg_simulations_maps_1980_2020_C[scenario_short_idx,:,:])).min()
    max_value = np.concatenate((avg_predictions_maps_1980_2020_C[scenario_short_idx,:,:], avg_simulations_maps_1980_2020_C[scenario_short_idx,:,:])).max()
    max_abs_value = np.max([abs(min_value), abs(max_value)])

    big_axes[scenario_short_idx].set_title(scenario, size=size_title_grid_spec, pad=65, **font)
    big_axes[scenario_short_idx]._frameon = False
    big_axes[scenario_short_idx].tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

    levels = np.linspace(min_value, max_value, 40)
    cbarticks = [-50,-40,-30,-20,-10,0,10,20,30]

    # ax1
    ax1 = fig.add_subplot(3,3,scenario_short_idx*3+1, projection=ccrs.Robinson())
    NN_ensemble_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_predictions_maps_1980_2020_C[scenario_short_idx,:,:],coord=lons)
    cs1=ax1.contourf(lons_cyclic,lats,NN_ensemble_warming_cyclic_data, levels=40,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=-90, vmax=90)
    ax1.set_title('Deep Neural Networks ensemble', loc='center', size=size_title_axes, pad=17, **font)
    ax1.coastlines()
    gl1 = ax1.gridlines(draw_labels=True, linestyle='--', color='black', linewidth=0.1)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal', 'font':'Arial'}
    gl1.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal', 'font':'Arial'}
    cbar1 = fig.colorbar(cs1, shrink=0.7, ax=ax1, orientation='horizontal', pad=0.12, ticks=cbarticks)
    cbar1.ax.tick_params(labelsize=size_colorbar_ticks)
    cbar1.set_label(label='Surface Air Temperature anomaly [°C]', size=size_colorbar_labels, labelpad=20, family='Arial')
    for l in cbar1.ax.xaxis.get_ticklabels():
        l.set_family("Arial")
        l.set_size(size_colorbar_ticks)

    # ax2 
    ax2 = fig.add_subplot(3,3,scenario_short_idx*3+2, projection=ccrs.Robinson())
    simulation_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_simulations_maps_1980_2020_C[scenario_short_idx,:,:],coord=lons)
    cs2=ax2.contourf(lons_cyclic,lats,simulation_warming_cyclic_data,levels=40,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap,vmin=-90, vmax=90)
    ax2.set_title('CMIP6 ensemble', loc='center', size=size_title_axes, pad=17, **font)
    ax2.coastlines()
    gl2 = ax2.gridlines(draw_labels=True, linestyle='--', color='black', linewidth=0.1)
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal', 'font':'Arial'}
    gl2.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal', 'font':'Arial'}
    cbar2 = fig.colorbar(cs2, ax=ax2, shrink=0.7, orientation='horizontal', pad=0.12, ticks=cbarticks)
    cbar2.ax.tick_params(labelsize=size_colorbar_ticks)
    cbar2.set_label(label='Surface Air Temperature anomaly [°C]', size=size_colorbar_labels, labelpad=20, family='Arial')
    for l in cbar2.ax.xaxis.get_ticklabels():
        l.set_family("Arial")
        l.set_size(size_colorbar_ticks)

    # ax3
    max_difference = np.max(avg_predictions_maps_1980_2020_C[scenario_short_idx,:,:]-avg_simulations_maps_1980_2020_C[scenario_short_idx,:,:])
    min_difference = np.min(avg_predictions_maps_1980_2020_C[scenario_short_idx,:,:]-avg_simulations_maps_1980_2020_C[scenario_short_idx,:,:])
    abs_max_difference = np.max([abs(max_difference), abs(min_difference)])

    levels_3 = np.linspace(-abs_max_difference, abs_max_difference, 40)
    cbarticks_3 = np.arange(np.ceil(min_difference),np.ceil(max_difference)+2,2)
    ax3 = fig.add_subplot(3,3,scenario_short_idx*3+3, projection=ccrs.Robinson())
    difference_warming_cyclic_data,lons_cyclic=add_cyclic_point(avg_predictions_maps_1980_2020_C[scenario_short_idx,:,:]-avg_simulations_maps_1980_2020_C[scenario_short_idx,:,:],coord=lons)
    cs3=ax3.contourf(lons_cyclic,lats,difference_warming_cyclic_data, levels=40,
                            transform = ccrs.PlateCarree(),
                            cmap=colormap, vmin=-abs_max_difference-7, vmax=abs_max_difference+7)
    ax3.set_title('Bias (DNNs - CMIP6)', loc='center', size=size_title_axes, pad=17, **font)
    ax3.coastlines()
    gl3 = ax3.gridlines(draw_labels=True, linestyle='--', color='black', linewidth=0.1)
    gl3.top_labels = False
    gl3.right_labels = False
    gl3.xlabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal', 'font':'Arial'}
    gl3.ylabel_style = {'size': size_lat_lon_coords, 'color': 'black', 'weight': 'normal', 'font':'Arial'}
    cbar3 = fig.colorbar(cs3, shrink=0.7, ax=ax3, orientation='horizontal', pad=0.12, ticks=cbarticks_3)
    cbar3.ax.tick_params(labelsize=size_colorbar_ticks)
    cbar3.set_label(label='Surface Air Temperature difference [°C]', size=size_colorbar_labels, labelpad=20, family='Arial')
    for l in cbar3.ax.xaxis.get_ticklabels():
        l.set_family("Arial")
        l.set_size(size_colorbar_ticks)

    plt.draw()
    for ea in gl1.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)
    for ea in gl2.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)
    for ea in gl3.ylabel_artists:
        right_label = ea.get_position()[0] > 0
        if right_label:
            ea.set_visible(False)

plt.subplots_adjust(hspace=0.5,wspace=0.1)
plt.savefig(f'Ext_Fig_2.png', bbox_inches = 'tight', dpi=300)
plt.close()
