[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8392956.svg)](https://doi.org/10.5281/zenodo.8392956)


Transferring climate change knowledge
=====================================
A repository including the code necessary to reproduce the results present in Immorlano et al. 2023, “Transferring climate change knowledge” submitted to Nature on September 21, 2023.


Content
-------
* [Related Publication](#related-publication)
* [Installation](#installation)
* [Run a demo version](#run-a-demo-version)
* [Run the full version](#run-the-full-version)
* [Reproduce the results present in the paper](#reproduce-the-results-present-in-the-paper)
* [Contributors](#contributors)
* [Acknowledgements and References](#acknowledgements-and-references)
* [License](#license)


Related Publication
-------------------
Immorlano, F., Eyring, V., le Monnier de Gouville, T., Accarino, G., Elia, D., Aloisio, G. & Gentine, P. Transferring climate change knowledge. arXiv preprint. DOI: <a href="https://doi.org/10.48550/arXiv.2309.14780">arXiv.2309.14780</a> (2023). (*in review*)


Installation
------------
Python version 3.11.0 or higher is needed.

A conda env containing all the packages and versions required to run the workflow and/or reproduce the figures present in the paper can be created by running the following command:

<code>conda env create --file transferring_env.yml</code>

This makes the installation easy and fast. The conda env was created and tested on a MacBook M2 Pro with MacOS Ventura 13.4.1 to enable the user to run a demo version of the workflow and reproduce the results present in the paper.

Concerning the full version, Tensorflow version 2.12.0 was used to run the entire workflow on <a href="https://confluence.columbia.edu/confluence/display/rcs/Ginsburg+-+Technical+Information">Ginsburg</a> (Columbia University) and <a href="https://www.cmcc.it/super-computing-center-scc">Juno</a> (CMCC Foundation) supercomputers.


Files description
-----------------
* `architectures.py` &rarr; Definition and building of the Deep Neural Network (DNN) used in the Transfer Learning approach
* `area_cella.csv` &rarr; Area weighting factors for each gridpoint of the grid corresponding to `CanESM5_CanOE_grid` used in this work
* `CanESM5_CanOE_grid` &rarr; Description of the grid used in this work (Grid from CanESM5-CanOE simulation)
* `transferring_env.yml` &rarr; YAML file needed to create the conda environment to reproduce the experiments and the figures
* `erf_estimates_with_aerosols_Zebedee_Nichols.csv` &rarr; Dataset of Effective Radiative Forcing values for each year
* `lib.py` &rarr; Routines called in several scripts
* `variables.py` &rarr; Definition of variables used in the scripts
* `BEST_regridded_annual_1979-2022.nc` &rarr; Berkeley Earth Surface Temperatures (BEST) observative maps resulting from the preprocessing (i.e., regridding to CanESM5-CanOE grid, computation of annual average maps, deletion of years out of 1979-2022)
* `gaussian_noise_5` &rarr; Directory containing BEST observative maps added with Gaussian noise to be used for Transfer Learning on observations
* `Land_and_Ocean_global_average_annual.txt` &rarr; Average annual uncertainties associated with BEST observative data. The column headers were manually removed for ease of use

The following scripts should be used to download and process CMIP6 and BEST data:
* `CMIP6_download_process.py` &rarr; Download of CMIP6 simulations from Climate Data Store and processing
* `BEST_data_processing.py` &rarr; Processing of BEST observative maps
* `BEST_data_add_gaussian_noise.py` &rarr; Generation of BEST observative maps added with Gaussian noise to be used for Transfer Learning on observations

The following scripts should be used to perform Training, Transfer Learning on Simulations (leave-one-out cross-validation procedure) and the Transfer Learning on Observations:
* `First_Training.py` &rarr; Training of each DNN on one of the 22 Earth System Models (ESMs) simulations under one of the 3 SSP scenarios
* `Transfer_learning_simulations.py` &rarr; Pre-trained DNNs transfer learning on ESMs simulations according to the leave-one-out cross-validation (LOO-CV) procedure
* `Transfer_learning_observations.py` &rarr; Pre-trained DNNs transfer learning on BEST observative data

The following scripts must be used to reproduce the figures and compute the values for Supplementary Table 1 present in the paper:
* `Fig_1.py`
* `Fig_2.py`
* `Fig_3.py`
* `Fig_4.py`
* `Ext_Fig_1.py`
* `Ext_Fig_2.py`
* `Ext_Fig_3.py`
* `Ext_Fig_4.py`
* `Ext_Fig_5.py`
* `Ext_Fig_6.py`
* `Ext_Fig_7.py`
* `Ext_Fig_8.py`
* `Ext_Fig_9.py`
* `Ext_Fig_10.py`
* `Supp_Table_1.py`


Run a demo version
------------------
A demo version fo the entire workflow can be run. After having downloaded the present GitHub repository, all the files needed to run the demo shall be organized according to the following hierarchy:

```plain
├── root
│   ├── architectures.py
│   ├── area_cella.csv
│   ├── BEST_data_add_gaussian_noise.py
│   ├── BEST_data_processing.py
│   ├── CanESM5-CanOE_grid
│   ├── CMIP6_download_process.py
│   ├── Demo_download
│   │   ├── Data
│   │   │   ├── BEST_data
│   │   │   │   ├── Land_and_Ocean_global_average_annual.txt
│   ├── Demo_no_download
│   │   ├── Data
│   │   │   ├── BEST_data
│   │   │   │   ├── BEST_regridded_annual_1979-2022.nc
│   │   │   │   ├── gaussian_noise_5
│   │   │   │   ├── Land_and_Ocean_global_average_annual.txt
│   │   │   ├── CMIP6_data
│   ├── erf_estimates_with_aerosols_Zebedee_Nichols.csv
│   ├── Figures 
│   ├── First_Training.py
│   ├── lib.py
│   ├── Transfer_learning_observations.py
│   ├── Transfer_learning_simulations.py
│   ├── variables.py
```

The demo version can be run in two ways:
*  download and build a small dataset and then train the DNNs on those data
* start directly with training the DNNs on the same small dataset already downloaded and processed

In both cases, the workflow will be executed for CNRM-ESM2-1, FGOALS-f3-L and MIROC6 ESMs simulations and for the SSP2-4.5 scenario.
### Data downloading and processing + training
To this aim, the BEST observative maps used in this study (i.e., Global Monthly Land + Ocean — Average Temperature with Air Temperatures at Sea Ice (Recommended; 1850 – Recent) — 1º x 1º Latitude-Longitude Grid) should be gathered from the <a href="https://berkeleyearth.org/data/">BEST archive</a> (direct download: <a href="https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc">1º x 1º Latitude-Longitude Grid (~400 MB)</a>) . The file must be named `BEST_2022.nc` and placed in `root/Demo_download/Data/BEST_data`. \
Since CMIP6 data were gathered from the Copernicus Climate Data Store (CDS), the CDS API key should be configured on your laptop according to the <a href="https://cds.climate.copernicus.eu/api-how-to">CDS API documentation</a> before downloading them. \
The following variables in `variables.py` should be set to:
* `demo_download = True`
* `demo_no_download = False`

Now, the scripts should be executed in the following order:
1.  `CMIP6_download_process.py` to download and process (i.e., regridding to CanESM5-CanOE grid, computing annual average maps) CMIP6 simulation data for the three ESMs and for SSP2-4.5. The resulting nc files will be saved in `root/Demo_download/Data/CMIP6_data/near_surface_air_temperature/Annual_uniform_remapped`. 
2. `BEST_data_processing.py` to process BEST observative maps (i.e., regridding to CanESM5-CanOE grid, computing annual average maps, deleting years out of 1979-2022). The resulting file will be saved as `root/Demo_download/Data/BEST_data/BEST_regridded_annual_1979-2022.nc`
3. `BEST_data_add_gaussian_noise.py` to add noise to the BEST observative maps. Specifically, the noise is sampled from a Gaussian distribution with 0 mean and stddev equal to the BEST observative data uncertainty. The addition of noise is repeated 5 times for each ESM and each scenario. The resulting files will be saved in `root/Demo_download/Data/BEST_data/gaussian_noise_5`.
4. `First_Training.py` to train an individual DNN on the simulation of one of the three ESMs for the SSP2-4.5 scenario. A total of three DNNs are trained, each with the same architecture. The results will be saved in `root/Demo_download/Experiments/First_Training/First_Training_[date-time]`.
5. `Transfer_learning_simulations.py` to load the pre-trained DNNs and transfer learn them on the ESMs simulations according to the LOO-CV procedure. The variable `FIRST_TRAINING_DIRECTORY` in `Transfer_learning_simulations.py` should be set to the directory name related to the first training (i.e. `FIRST_TRAINING_DIRECTORY = First_Training_[date-time]`). This is necessary to load the pre-trained DNNs. The variable `exclude_family_members` in `variables.py`should be set `True` if the ESMs based on the same atmospheric model as the take-out ESM must be excluded in each iteration of the LOO-CV procedure. The results will be saved in `root/Demo_download/Experiments/Transfer_Learning_on_Simulations/Transfer_learning_[date-time]/Shuffle_[number]`. The `Shuffle_[number]` corresponds to an iteration of the LOO-CV procedure.
6. `Transfer_learning_observations.py` to load the pre-trained DNNs and transfer learn them on the BEST observative data. The variable `FIRST_TRAINING_DIRECTORY` in `Transfer_learning_simulations.py` must be set to the directory name related to the first training (i.e. `FIRST_TRAINING_DIRECTORY = First_Training_[date-time]`). This is necessary to load the pre-trained DNNs. The results will be saved in `root/Demo_download/Experiments/Transfer_Learning_on_Observations/Transfer_learning_obs_[date-time]`.

### Training (without downloading and processing)
A demo version can be run directly starting with training the DNNs on the CNRM-ESM2-1, FGOALS-f3-L and MIROC6 models simulations. \
In this case, the following variables in `variables.py` should be set to:
* `demo_download = False`
* `demo_no_download = True`

Now, the scripts should be executed in the following order:
1. `First_Training.py` to train an individual DNN on one of the three ESMs simulation for the SSP2-4.5 scenario. A total of three DNNs will be trained, each with the same architecture. The results will be saved in `root/Demo_no_download/Experiments/First_Training/First_Training_[date-time]`.
2. `Transfer_learning_simulations.py` to load pre-trained DNNs and transfer learn them on the ESMs simulations according to the LOO-CV procedure. The variable `FIRST_TRAINING_DIRECTORY` in `Transfer_learning_simulations.py` must be set to the directory name related to the first training (i.e. `FIRST_TRAINING_DIRECTORY = First_Training_[date-time]`). This is necessary to load the pre-trained DNNs. The variable `exclude_family_members` in `variables.py`should be set `True` if the ESMs based on the same atmospheric model as the take-out ESM must be excluded in each iteration of the LOO-CV procedure. The results will be saved in `root/Demo_no_download/Experiments/Transfer_Learning_on_Simulations/Transfer_learning_[date-time]/Shuffle_[number]`. The `Shuffle_[number]` corresponds to an iteration of the LOO-CV procedure.
3. `Transfer_learning_observations.py` to load the pre-trained DNNs and transfer learn them on the BEST observative data. The variable `FIRST_TRAINING_DIRECTORY` in `Transfer_learning_simulations.py` must be set to the directory name related to the first training (i.e. `FIRST_TRAINING_DIRECTORY = First_Training_[date-time]`). This is necessary to load the pre-trained DNNs. The results will be saved in `root/Demo_no_download/Experiments/Transfer_Learning_on_Observations/Transfer_learning_obs_[date-time]`.

The Demo software was tested on a MacBook M2 Pro equipped with MacOS Ventura 13.4.1. The expected run times are the following:
* `CMIP6_download_process.py`: about 12 seconds (after the CMIP6 data download)
* `BEST_data_processing.py`: about 122 seconds
* `BEST_data_add_gaussian_noise.py`: about 7 seconds
* `First_Training.py`, `Transfer_learning_simulations.py`, `Transfer_learning_observations.py`: about 4.5 seconds per epoch

Run the full version
--------------------

The full version of the entire workflow can be run. After having downloaded `BEST_data.zip` and `CMIP6_data.zip` from <a href="https://zenodo.org/records/10512548
">Zenodo</a>, the files needed to run the entire workflow shall be organized as the following hierarchy:

```plain
├── root
│   ├── architectures.py
│   ├── area_cella.csv
│   ├── BEST_data_add_gaussian_noise.py
│   ├── BEST_data_processing.py
│   ├── CanESM5-CanOE_grid
│   ├── CMIP6_download_process.py
│   ├── erf_estimates_with_aerosols_Zebedee_Nichols.csv
│   ├── First_Training.py
│   ├── lib.py
│   ├── Source_data
│   │   ├── BEST_data
│   │   │   ├── BEST_regridded_annual_1979-2022.nc
│   │   │   ├── gaussian_noise_5
│   │   │   ├── Land_and_Ocean_global_average_annual.txt
│   │   ├── CMIP6_data
│   ├── variables.py

```

The full version will be run by training the DNNs on the 22 ESMs simulations for SSP2-4.5, 3-7.0, and 5-8.5 scenarios. \
In this case, the following variables in `variables.py` must be set to:
* `demo_download = False`
* `demo_no_download = False`

Now, the scripts should be executed in the following order:
1. `First_Training.py` to train an individual DNN on one of the 22 ESMs simulation for the SSP2-4.5, 3-7.0, and 5-8.5 scenarios. A total of 66 DNNs will be trained, each with the same architecture. The results will be saved in `root/Experiments/First_Training/First_Training_[date-time]`.
2. `Transfer_learning_simulations.py` to load the pre-trained DNNs and transfer learn them on the ESMs simulations according to the LOO-CV procedure. The variable `FIRST_TRAINING_DIRECTORY` in `Transfer_learning_simulations.py` should be set to the directory name related to the first training (i.e. `FIRST_TRAINING_DIRECTORY = First_Training_[date-time]`). This is necessary to load the pre-trained models. The variable `exclude_family_members` in `variables.py`should be set `True` if the ESMs based on the same atmospheric model as the take-out ESM must be excluded in each iteration of the LOO-CV procedure. The results will be saved in `root/Experiments/Transfer_Learning_on_Simulations/Transfer_learning_[date-time]/Shuffle_[number]`. The `Shuffle_[number]` corresponds to an iteration of the LOO-CV approach.
3. `Transfer_learning_observations.py` to load the pre-trained DNNs and transfer learn them on the BEST observative data. The variable `FIRST_TRAINING_DIRECTORY` in `Transfer_learning_simulations.py` should be set to the directory name of the first training (i.e. `FIRST_TRAINING_DIRECTORY = First_Training_[date-time]`). This is necessary to load the pre-trained models. The results will be saved in `root/Experiments/Transfer_Learning_on_Observations/Transfer_learning_obs_[date-time]`.

The models resulting from transfer learning on BEST observative data and used to predict temperature values up to 2098 can be downloaded from <a href="https://huggingface.co/francesco-immorlano/Transferring-climate-change-knowledge-models/tree/main">Hugging Face</a>

Reproduce the results present in the paper
------------------------------------------

The figures and the results present in the paper can be reproduced. After having downloaded `BEST_data.zip`, `CMIP6_data.zip`, `First_training_obs.zip`, `First_Training.zip`, `Transfer_Learning_on_Observations.zip`, `Transfer_Learning_on_Simulations_AM_families.zip`, `Transfer_Learning_on_Simulations_reverse.zip`, and `Transfer_Learning_on_Simulations.zip` from <a href="https://zenodo.org/records/10512548
">Zenodo</a>, the files needed to reproduce the results shall be organized as the following hierarchy:

```plain
├── root
│   ├── area_cella.csv
│   ├── Figures
│   │   ├── Ext_Fig_1.py
│   │   ├── Ext_Fig_2.py
│   │   ├── Ext_Fig_3.py
│   │   ├── Ext_Fig_4.py
│   │   ├── Ext_Fig_5.py
│   │   ├── Ext_Fig_6.py
│   │   ├── Ext_Fig_7.py
│   │   ├── Ext_Fig_8.py
│   │   ├── Ext_Fig_9.py
│   │   ├── Ext_Fig_10.py
│   │   ├── Fig_1.py
│   │   ├── Fig_2.py
│   │   ├── Fig_3.py
│   │   ├── Fig_4.py
│   │   ├── lats.pickle
│   │   ├── lons.pickle
│   │   ├── Supp_Table_1.py
│   ├── Source_data
│   │   ├── BEST_data
│   │   ├── CMIP6_data
│   │   ├── First_Training_obs
│   │   ├── First_Training
│   │   ├── Transfer_Learning_on_Observations
│   │   ├── Transfer_Learning_on_Simulations_AM_families
│   │   ├── Transfer_Learning_on_Simulations_reverse
│   │   ├── Transfer_Learning_on_Simulations
```

`First_Training_obs` &rarr; Near-surface air temperature maps generated by the DNNs trained on the BEST observative data from 1979 to 2022
`First_Training` &rarr; Near-surface air temperature maps generated by the DNNs trained on the 22 ESMs simulations under SSP2-4.5, 3-7.0, and 5-8.5 scenarios
`Transfer_Learning_on_Observations` &rarr; Near-surface air temperature maps generated by the DNNs after transfer learning on BEST oobservative data
`Transfer_Learning_on_Simulations_AM_families` &rarr; Near-surface air temperature maps generated by the DNNs after transfer learning on the 22 ESMs simulations under SSP2-4.5, 3-7.0, and 5-8.5 scenarios (LOO-CV procedure). Those ESMs based on the same atmospheric model as the take-out ESM were excluded in each iteration of the LOO-CV procedure
`Transfer_Learning_on_Simulations_reverse` &rarr; Near-surface air temperature maps generated by the DNNs after the reverse LOO-CV procedure. The pre-trained DNNs were transfer learned on the 22 ESMs simulations under SSP2-4.5, 3-7.0, and 5-8.5 scenarios, by using 2023–2098 as training set and 1850–2022 as test set
`Transfer_Learning_on_Simulations` &rarr; Near-surface air temperature maps generated by the DNNs after transfer learning on the 22 ESMs simulations under SSP2-4.5, 3-7.0, and 5-8.5 scenarios (LOO-CV procedure)


The following scripts should be used to reproduce the figures and compute the values for Supplementary Table 1 present in the paper:
* `Fig_1.py`
* `Fig_2.py`
* `Fig_3.py`
* `Fig_4.py`
* `Ext_Fig_1.py`
* `Ext_Fig_2.py`
* `Ext_Fig_3.py`
* `Ext_Fig_4.py`
* `Ext_Fig_5.py`
* `Ext_Fig_6.py`
* `Ext_Fig_7.py`
* `Ext_Fig_8.py`
* `Ext_Fig_9.py`
* `Ext_Fig_10.py`
* `Supp_Table_1.py`

Contributors
------------

- Francesco Immorlano (francesco.immorlano@cmcc.it)
- Veronika Eyring (veronika.eyring@dlr.de)
- Thomas le Monnier de Gouville (thomas.le-monnier-de-gouville@polytechnique.edu)
- Gabriele Accarino (gabriele.accarino@cmcc.it)
- Donatello Elia (donatello.elia@cmcc.it)
- Giovanni Aloisio (giovanni.aloisio@cmcc.it)
- Pierre Gentine (pg2328@columbia.edu)


Acknowledgements and References
-------------------------------

If you use the resource in your research, please cite our paper. At the moment, we offer the bibliography of the arXiv preprint version.

```
@misc{immorlano2023transferring,
      title={Transferring climate change knowledge}, 
      author={Francesco Immorlano and Veronika Eyring and Thomas le Monnier de Gouville and Gabriele Accarino and Donatello Elia and Giovanni Aloisio and Pierre Gentine},
      year={2023},
      eprint={2309.14780},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph}
}
```

License
-------

MIT License

Copyright (c) 2023 Francesco Immorlano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
