"""
Author: Francesco Immorlano

Script to compute Table S2
"""

import sys
sys.path.insert(1, './..')
from lib import *

""" Load CMIP6 ESMs simulations """
simulations = read_all_cmip6_simulations()

ensemble_statistics = np.zeros((len(short_scenarios_list),3,len(models_list)))            # scenarios, (median, q05, q95), models
simulations_statistics = np.zeros((len(short_scenarios_list),3,len(models_list)))         # scenarios, (avg_taken_out, q05, q95), models
accuracy = np.zeros((len(short_scenarios_list),len(models_list)))
precision_simulations = np.zeros((len(short_scenarios_list),len(models_list)))

global_avg_bias = np.zeros((len(short_scenarios_list),len(models_list)))
rmse = np.zeros((len(short_scenarios_list),len(models_list)))

for shuffle_idx in range(len(models_list)):

    if shuffle_idx < 9: shuffle_number = f'0{shuffle_idx+1}'
    else: shuffle_number = f'{shuffle_idx+1}'

    """ Load DNNs predictions after TL on the take-out ESM simulation """
    predictions_tl_on_simulations = read_tl_simulations_predictions_shuffle(shuffle_idx, compute_figures_tables)

    remaining_models_idx = np.arange(22)
    remaining_models_idx = np.delete(remaining_models_idx, shuffle_idx)

    simulations_remaining = simulations[remaining_models_idx,:,:,:]
    simulation_takeout = simulations[shuffle_idx,:,:,:]

    # Convert from K to Celsius degrees
    predictions_C = predictions_tl_on_simulations - 273.15
    simulation_takeout_C = simulation_takeout - 273.15
    simulations_remaining_C = simulations_remaining - 273.15

    # Compute average global surface air temperature
    annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
    annual_simulation_takeout_means = ((simulation_takeout_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
    annual_simulations_remaining_means = ((simulations_remaining_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
    # Compute warming wrt pre-industrial period
    warming_predictions_means = annual_predictions_means - global_mean_temp_1995_2014
    warming_simulation_takeout_means = annual_simulation_takeout_means - global_mean_temp_1995_2014
    warming_simulations_remaining_means = annual_simulations_remaining_means - global_mean_temp_1995_2014

    # Select warming values in 2081-2098
    warming_predictions_means_2081_2098 = warming_predictions_means[:,:,2081-1850:2098-1850+1]
    warming_simulation_takeout_means_2081_2098 = warming_simulation_takeout_means[:,2081-1850:2098-1850+1]
    warming_simulations_remaining_means_2081_2098 = warming_simulations_remaining_means[:,:,2081-1850:2098-1850+1]

    # Compute ensemble of warming predicted by DNNs after TL on the take-out model
    ensemble_warming_predictions_2081_2098 = np.mean(warming_predictions_means_2081_2098, axis=0)

    # Compute global avg bias
    bias = ensemble_warming_predictions_2081_2098 - warming_simulation_takeout_means_2081_2098
    global_avg_bias[:,shuffle_idx] = np.mean(bias, axis=1) 

    # Compute RMSE
    mse = np.mean(bias**2, axis=1)
    rmse[:,shuffle_idx] = np.sqrt(mse)

    # Compute median, 5% and 95% for the DNNs predictions and average temperature, 5% and 95% for the CMIP6 simulations
    median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))

    q05_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q95_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))

    for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
        for i in range(2098-2081+1):
            # 21 DNNs predictions after TL on simulations
            median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(warming_predictions_means_2081_2098[:,short_scenario_idx,i])
            q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],5)
            q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],95)
            # 21 remaining CMIP6 simulations
            q05_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],5)
            q95_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],95)
            
        ensemble_statistics[short_scenario_idx,0,shuffle_idx] = median_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,1,shuffle_idx] = q05_predictions_means_2081_2098[short_scenario_idx,:].mean()

ensemble_statistics = np.zeros((len(short_scenarios_list),3,len(models_list)))            # scenarios, (median, q05, q95), models
simulations_statistics = np.zeros((len(short_scenarios_list),3,len(models_list)))         # scenarios, (avg_taken_out, q05, q95), models
accuracy = np.zeros((len(short_scenarios_list),len(models_list)))
precision_simulations = np.zeros((len(short_scenarios_list),len(models_list)))

global_avg_bias = np.zeros((len(short_scenarios_list),len(models_list)))
rmse = np.zeros((len(short_scenarios_list),len(models_list)))

for shuffle_idx in range(len(models_list)):

    if shuffle_idx < 9: shuffle_number = f'0{shuffle_idx+1}'
    else: shuffle_number = f'{shuffle_idx+1}'

    """ Load DNNs predictions after TL on the take-out ESM simulation """
    predictions_tl_on_simulations = read_tl_simulations_predictions_shuffle(shuffle_idx, compute_figures_tables)

    remaining_models_idx = np.arange(22)
    remaining_models_idx = np.delete(remaining_models_idx, shuffle_idx)

    simulations_remaining = simulations[remaining_models_idx,:,:,:]
    simulation_takeout = simulations[shuffle_idx,:,:,:]

    # Convert from K to Celsius degrees
    predictions_C = predictions_tl_on_simulations - 273.15
    simulation_takeout_C = simulation_takeout - 273.15
    simulations_remaining_C = simulations_remaining - 273.15

    # Compute average global surface air temperature
    annual_predictions_means = ((predictions_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
    annual_simulation_takeout_means = ((simulation_takeout_C * area_cella).sum(axis=(-1,-2)))/total_earth_area
    annual_simulations_remaining_means = ((simulations_remaining_C * area_cella).sum(axis=(-1,-2)))/total_earth_area

    # Compute warming wrt pre-industrial period
    warming_predictions_means = annual_predictions_means - global_mean_temp_1995_2014
    warming_simulation_takeout_means = annual_simulation_takeout_means - global_mean_temp_1995_2014
    warming_simulations_remaining_means = annual_simulations_remaining_means - global_mean_temp_1995_2014

    # Select warming values in 2081-2098
    warming_predictions_means_2081_2098 = warming_predictions_means[:,:,2081-1850:2098-1850+1]
    warming_simulation_takeout_means_2081_2098 = warming_simulation_takeout_means[:,2081-1850:2098-1850+1]
    warming_simulations_remaining_means_2081_2098 = warming_simulations_remaining_means[:,:,2081-1850:2098-1850+1]

    # Compute ensemble of warming predicted by DNNs after TL on the take-out model
    ensemble_warming_predictions_2081_2098 = np.mean(warming_predictions_means_2081_2098, axis=0)

    # Compute global avg bias
    bias = ensemble_warming_predictions_2081_2098 - warming_simulation_takeout_means_2081_2098
    global_avg_bias[:,shuffle_idx] = np.mean(bias, axis=1) 

    # COmpute RMSE
    mse = np.mean(bias**2, axis=1)
    rmse[:,shuffle_idx] = np.sqrt(mse)

    # Compute median, 5% and 95% for the DNNs predictions and average temperature, 5% and 95% for the CMIP6 simulations
    median_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q05_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q95_predictions_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))

    q05_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))
    q95_simulations_remaining_means_2081_2098 = np.zeros((len(short_scenarios_list),2098-2081+1))

    for short_scenario_idx, short_scenario in enumerate(short_scenarios_list):
        for i in range(2098-2081+1):
            # 21 DNNs predictions after TL on simulations
            median_predictions_means_2081_2098[short_scenario_idx,i] = np.median(warming_predictions_means_2081_2098[:,short_scenario_idx,i])
            q05_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],5)
            q95_predictions_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_predictions_means_2081_2098[:,short_scenario_idx,i],95)
            # 21 remaining CMIP6 simulations
            q05_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],5)
            q95_simulations_remaining_means_2081_2098[short_scenario_idx,i] = np.percentile(warming_simulations_remaining_means_2081_2098[:,short_scenario_idx,i],95)
            
        ensemble_statistics[short_scenario_idx,0,shuffle_idx] = median_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,1,shuffle_idx] = q05_predictions_means_2081_2098[short_scenario_idx,:].mean()
        ensemble_statistics[short_scenario_idx,2,shuffle_idx] = q95_predictions_means_2081_2098[short_scenario_idx,:].mean()

        simulations_statistics[short_scenario_idx,0,shuffle_idx] = warming_simulation_takeout_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,1,shuffle_idx] = q05_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()
        simulations_statistics[short_scenario_idx,2,shuffle_idx] = q95_simulations_remaining_means_2081_2098[short_scenario_idx,:].mean()

        accuracy[short_scenario_idx, shuffle_idx] = ensemble_statistics[short_scenario_idx,0,shuffle_idx] - simulations_statistics[short_scenario_idx,0,shuffle_idx]
        precision_simulations[short_scenario_idx, shuffle_idx] = ((simulations_statistics[short_scenario_idx,2,shuffle_idx] - simulations_statistics[short_scenario_idx,1,shuffle_idx]) - (ensemble_statistics[short_scenario_idx,2,shuffle_idx] - ensemble_statistics[short_scenario_idx,1,shuffle_idx]))/(simulations_statistics[short_scenario_idx,2,shuffle_idx] - simulations_statistics[short_scenario_idx,1,shuffle_idx])*100


    

print('accuracy: DNN - takeout')
print('precision: reduction of 21 DNNs uncertainty (after TL on simulations) wrt to the remaining 21 CMIP6 ESMs uncertainty \n')

for idx_scenario, scenario in enumerate(short_scenarios_list):
    print(f'{scenario}')
    for idx_model, model in enumerate(models_list):
        print(f'--- {model}\
              \tglob avg error: {np.round(global_avg_bias[idx_scenario,idx_model],2)}°C \
              \trmse: {np.round(rmse[idx_scenario,idx_model],2)}°C \
              \t% uncertainty reduction: {np.round(precision_simulations[idx_scenario,idx_model],2)}\
              \taccuracy: {np.round(accuracy[idx_scenario,idx_model],2)}°C\
              \t5th: {np.round(ensemble_statistics[idx_scenario,1,idx_model],2)}°C\
              \t95th: {np.round(ensemble_statistics[idx_scenario,2,idx_model],2)}°C')