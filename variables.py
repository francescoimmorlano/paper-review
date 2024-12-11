"""
Author: Francesco Immorlano
"""

from datetime import datetime
import numpy as np

demo_download = False
demo_no_download = False

# Set to True if a figure or table of the paper must be computed
# Set False if you performed other Experiments that were saved in Experiments folder
compute_figures_tables_paper = True

ts = datetime.now()
ts_human = ts.strftime('%Y-%m-%d_%H-%M-%S')

variable = 'near_surface_air_temperature'
variable_short = 'tas'

# Number of degrees of freedom of smoothing splines
n_dof = 20

# Avg global surface temperature in 1995–2014
global_mean_temp_1995_2014 = 14.711500000000001
# Avg global surface temperature in 1850-1900
global_mean_temp_1850_1900 = 13.798588235294114
# Avg global surface temperature in 1850-2000
global_mean_temp_1850_2000 = 14.007112582781458
total_earth_area = 5.1009974e+14 # m^2

if demo_download or demo_no_download:
  path_area_cella = './area_cella.csv'
else:
  path_area_cella = '../area_cella.csv'

with open(path_area_cella, newline='') as csvfile:
    area_cella = np.genfromtxt(csvfile, delimiter=',')

##################### CALLBACKS #####################
save_predictions_on_validation_set = True

# Decide if normalize input and/or output to a range set by the feature_range variable
scale_input = True
scale_output = True
feature_range = (0,1)

n_channels = 1
n_filters = 128

CO2eq_climate_model = 'MCE-v1-2'
withAerosolForcing = True

lr_first_train = 1e-4
lr_loo_cv = 0.25e-5
lr_tl_obs = 1e-5

loss = 'mae'

epochs = 500
batch_size_first_train = 8
batch_size_tl = 16

first_ssp_year = 2015

# Set window size for computing moving avg
window_size = 18

# Min and max CMIP6 temperature values
y_min = 212.1662
y_max = 317.38766

# Min and max CO2eq values
X_min = 285.1337
X_max = 1262.1862

# settings for modern reference time period and proxy for pre-industrial time period
refperiod_start = 1995
refperiod_end   = 2014
piperiod_start  = 1850
piperiod_end    = 1900

# historical warming estimate based on cross-chapter box 2.3 (https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter02.pdf)
refperiod_conversion = 0.85

'''
  First Training
'''
start_year_first_training = 1850
end_year_first_training = 2098
start_year_first_training_val = 2070
end_year_first_training_val = 2080
n_training_years_first_training = end_year_first_training-start_year_first_training+1

'''
  Leave-one-out cross validation
'''
start_year_training_loo_cv = 1850
end_year_training_loo_cv = 2022
start_year_test_loo_cv = end_year_training_loo_cv+1
end_year_test_loo_cv = 2098
n_ssp_training_years_loo_cv = end_year_training_loo_cv-2015+1
n_training_years_loo_cv = end_year_training_loo_cv-1850+1
n_test_years_loo_cv = end_year_test_loo_cv-start_year_test_loo_cv+1
val_years_list_loo_cv = [2025, 2035, 2045, 2055, 2065, 2075, 2085, 2095]
exclude_family_members = False

'''
  Loave-one-out cross-validation reverse
'''
start_year_training_loo_cv_reverse = 2023
end_year_training_loo_cv_reverse = 2098

'''
  TL on observations
'''
start_year_training_tl_obs = 1979
end_year_training_tl_obs = 2022
start_year_test_tl_obs = end_year_training_tl_obs + 1
end_year_test_tl_obs = 2098
n_training_years_tl_obs = end_year_training_tl_obs - start_year_training_tl_obs + 1
n_test_years_tl_obs = end_year_test_tl_obs - start_year_test_tl_obs + 1
# Years reserved for validation
val_years_list_tl_obs = [2017, 2018, 2019, 2020]
n_BEST_datasets_per_model_scenario = 5

models_list_complete = [
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

models_short_list_complete = [
  'access_cm2',
  'awi_cm_1_1_mr',
  'bcc_csm2_mr',
  'cams_csm1_0',
  'canesm5_canoe',
  'cmcc_cm2_sr5',
  'cnrm_cm6_1',
  'cnrm_esm2_1',
  'fgoals_f3_l',
  'fgoals_g3'
  'gfdl_esm4',
  'iitm_esm',
  'inm_cm4_8',
  'inm_cm5_0',
  'ipsl_cm6a_lr',
  'kace_1_0_g',
  'miroc6',
  'mpi_esm1_2_lr',
  'mri_esm2_0',
  'noresm2_mm',
  'taiesm1',
  'ukesm1_0_ll'
]

atmospheric_model_families_dict = {
  'ACCESS-CM2' : ['KACE-1-0-G', 'UKESM1-0-LL'],
  'AWI-CM-1-1-MR': ['CAMS-CSM1-0', 'MPI-ESM1-2-LR'],
  'BCC-CSM2-MR': ['CMCC-CM2-SR5', 'FGOALS-f3-L', 'FGOALS-g3', 'NorESM2-MM', 'TaiESM1'],
  'CAMS-CSM1-0': ['AWI-CM-1-1-MR', 'MPI-ESM1-2-LR'],
  'CanESM5-CanOE': [],
  'CMCC-CM2-SR5': ['BCC-CSM2-MR', 'FGOALS-f3-L', 'FGOALS-g3', 'NorESM2-MM', 'TaiESM1'],
  'CNRM-CM6-1': ['CNRM-ESM2-1', 'IPSL-CM6A-LR'],
  'CNRM-ESM2-1': ['CNRM-CM6-1', 'IPSL-CM6A-LR'],
  'FGOALS-f3-L': ['BCC-CSM2-MR', 'CMCC-CM2-SR5', 'FGOALS-g3', 'NorESM2-MM', 'TaiESM1'],
  'FGOALS-g3': ['BCC-CSM2-MR', 'CMCC-CM2-SR5', 'FGOALS-f3-L', 'NorESM2-MM', 'TaiESM1'],
  'GFDL-ESM4': [],
  'IITM-ESM': [],
  'INM-CM4-8': ['INM-CM5-0'],
  'INM-CM5-0': ['INM-CM4-8'],
  'IPSL-CM6A-LR': ['CNRM-CM6-1', 'CNRM-ESM2-1'],
  'KACE-1-0-G': ['ACCESS-CM2', 'UKESM1-0-LL'],
  'MIROC6': [],
  'MPI-ESM1-2-LR': ['AWI-CM-1-1-MR', 'CAMS-CSM1-0'],
  'MRI-ESM2-0': [],
  'NorESM2-MM': ['BCC-CSM2-MR', 'CMCC-CM2-SR5', 'FGOALS-f3-L', 'FGOALS-g3', 'TaiESM1'],
  'TaiESM1': ['BCC-CSM2-MR', 'CMCC-CM2-SR5', 'FGOALS-f3-L', 'FGOALS-g3', 'NorESM2-MM'],
  'UKESM1-0-LL': ['ACCESS-CM2', 'KACE-1-0-G']
}

short_scenarios_list_complete = ['ssp245', 'ssp370', 'ssp585']

if demo_download or demo_no_download:
  models_list = ['CNRM-ESM2-1', 'FGOALS-f3-L', 'MIROC6'] 
  models_short_list = ['cnrm_esm2_1', 'fgoals_f3_l', 'miroc6']
  short_scenarios_list = ['ssp245']
  if demo_download:
    ROOT_EXPERIMENTS = './Demo_download/Experiments'
    ROOT_SOURCE_DATA = './Demo_download/Data'
  elif demo_no_download:
    ROOT_EXPERIMENTS = './Demo_no_download/Experiments'
    ROOT_SOURCE_DATA = './Demo_no_download/Data'
else: 
  models_list = models_list_complete
  models_short_list = models_short_list_complete
  short_scenarios_list = short_scenarios_list_complete
  ROOT_EXPERIMENTS = '../Experiments'
  ROOT_SOURCE_DATA = '../Source_data'

PATH_SMOOTHED_CMIP6_SIMULATIONS_DIRECTORY = f'{ROOT_SOURCE_DATA}/CMIP6_data/near_surface_air_temperature'
PATH_ANNUAL_SIMULATIONS_DIRECTORY = f'{ROOT_SOURCE_DATA}/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped'
PATH_BEST_DATA = f'{ROOT_SOURCE_DATA}/BEST_data/BEST_regridded_annual_1979-2022.nc'
PATH_BEST_DATA_UNCERTAINTY = f'{ROOT_SOURCE_DATA}/BEST_data/Land_and_Ocean_global_average_annual.txt'

columns_history_df = ['train_loss', 'val_loss']
columns_model_hyperparameters_df_first_training = ['train_directory_name', 'model', 'scenario', 'date_time', 'elapsed_loop_time', 'elapsed_train_time', 'epochs', 'batch_size', 'start_year_training', 'end_year_training', 'val_year', 'L1_regularization', 'L2_regularization', 'shuffle', 'scale_input', 'scale_output', 'norm_min', 'norm_max', 'train_percent', 'val_percent', 'opt', 'loss', 'weight_initializer', 'activation', 'CO2eq_climate_model', 'withAerosolForcing', 'use_observations']
columns_model_hyperparameters_df_tl = ['transf_learn_directory', 'first_train_directory', 'end_year_training', 'model', 'scenario', 'date_time', 'elapsed_loop_time', 'elapsed_train_time', 'epochs', 'batch_size', 'learning_rate', 'shuffle', 'scale_input', 'scale_output', 'norm_min', 'norm_max', 'y_min', 'y_max', 'CO2eq_climate_model', 'withAerosolForcing']