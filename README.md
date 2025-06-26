# Robust Fetal Pose Estimation
Repository for more robust keypoint detection across gestation and acquisition artifacts.

## File structure
```
.
├── main.py                 # main file excuted during `*.sh` script
├── data.py                 # includes the data classes for the *offline* training
├── online.py               # includes the data classes for the *online* training (this is recommended)
├── losses.py               # helper functions reside here
├── models.py               # two UNet's: one big, one small
├── optimizers.py           # get the optimizers and their learning rate schedulers
├── options.py              # options file... the *most* important file to read
├── configs                 # holds all the bash scripts
│   ├── `*.sh`
├── results                 # holds all the results (inference + testing)
│   ├── testing
│   │   ├── `*.csv`         #
├── data_partition.yml      # subjects and respective sampling density (1.0 == 100 samples)
├── inpainting              # data synthesis and augmentation pipeline
│   ├── fetch-feasible-masks.py     # data quality filter - validates keypoints within body masks
│   ├── generate_train_data.py      # main production training data generator with SE(3) transformations
│   ├── micc.py                     # alternative training data generator variant
│   ├── synth*.py                   # research/development synthesis prototypes (can be deleted)
│   ├── synthesize.py               # single-sample synthesis tool for testing
│   ├── clean-raw/                  # validated raw volumes (output of fetch-feasible-masks.py)
│   ├── clean-masks/                # validated segmentation masks
│   ├── train-data/                 # generated synthetic training data
│   └── labels/                     # keypoint coordinate files (.mat format)
└── README.md
```

## Installation

You can set up the environment with **Conda**.

### conda

```bash
conda env create -f environment.yml
conda activate pose
```



## Logistics
I use [Weights & Biases](https://wandb.ai/) to log my training runs. If you want to use it, you will need to log into your own account. If you do not use it, adjust the code accordingly (delete all instances where "wandb" appears).

## Inpainting/Data Synthesis Pipeline
The `inpainting/` directory contains a complete pipeline for generating synthetic training data through volume synthesis and pose augmentation:

### Production Workflow
1. **Data Quality Control**: `fetch-feasible-masks.py`
   - Validates that all 15 keypoints fall within body segmentation boundaries
   - Filters training data to ensure keypoint-mask consistency
   - Outputs clean data to `clean-raw/` and `clean-masks/`

2. **Synthetic Data Generation**: `generate_train_data.py`
   - Main production script for creating ML training datasets
   - Combines random uterus and body volumes from different patients
   - Applies SE(3) transformations (rotation, translation, scaling) with validation
   - Generates systematic output with organized naming and coordinate files
   - Creates batches of synthetic samples for model training

3. **Alternative Generation**: `micc.py`
   - Variant of the main generator with different configuration parameters
   - Can be used for creating additional dataset variations

### Development Files (Optional)
- `synth*.py`, `synthesize.py`: Research prototypes and algorithm development artifacts
- These can be safely deleted if only production functionality is needed