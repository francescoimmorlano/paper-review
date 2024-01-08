variable = 'near_surface_air_temperature'
variable_short = 'tas'

demo_download = False
demo_no_download = True

exclude_family_members = False

# Min and max CMIP6 temperature value
y_min = 212.1662
y_max = 317.38766

if demo_download or demo_no_download:
      models_list = ['CNRM-ESM2-1', 'FGOALS-f3-L', 'MIROC6'] 
      models_short_list = ['cnrm_esm2_1', 'fgoals_f3_l', 'miroc6']
      short_scenarios_list = ['ssp245']
else: 
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

      models_short_list = [
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
       
      short_scenarios_list = ['ssp245', 'ssp370', 'ssp585']
