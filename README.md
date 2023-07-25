# SurvivalHypergraph

The codes of "Stage II Colorectal Cancer Survival Prediction Through Exploring High-Order Information on Whole-Slide Histopathological Images".



## Installation Guide

1. Install anaconda3 in the machine according to https://www.anaconda.com/

   

2. Install openslide

   ```makefile
   sudo apt-get install openslide-tools
   ```

   

3. Create a conda environment based on the environment configuration file

   ```makefile
   conda env create -n survivalhypergraph -f enviroment.yaml
   ```

   

4. Activate the environment

   ```
   conda activate survivalhypergraph
   ```



## Preprocess Guide

1. Store whole slide image data in the following format

   ```makefile
   SurvivalHypergraph/
   		|-- svs_directory/
   				|-- wsi_1.svs
   				|-- wsi_2.svs
   				|-- ...
   ```

   

2. Run the preprocess code

   ```
   python preprocess/process_4_ft.py
   ```



## Run Guide

1. Store patient information in file `SurvivalHypergraph/data/opti_survival.json` in the following format

   ```makefile
   {
   		"patient_1_name":{
   				"status": 0/1,
   				"survival_time": xx (days),
   				"images": [wsi_1, wsi_2, ...]
   		},
   		"patient_2_name":{
   				"status": 0/1,
   				"survival_time": xx (days),
   				"images": [wsi_1, wsi_2, ...]
   		},
   		...
   }
   ```

   

2. Set hyperparameters in `train_config.py`

   

3. Run the trainning code

   ```makefile
   python train.py
   ```

   