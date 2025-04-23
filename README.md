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
└── README.md
```

## Logistics
I use [Weights & Biases](https://wandb.ai/) to log my training runs. If you want to use it, you will need to log into your own account. If you do not use it, adjust the code accordingly (delete all instances where "wandb" appears).