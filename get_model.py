from model_zoo.small_unet import Unet, DualUNet
from model_zoo.big_unet import UNet3D

def get_models(opts):

    models = {
        "small_unet":   Unet(dimension=3, input_nc=1 if not opts.use_amix else 17, output_nc=opts.nJoints, num_downs=opts.depth, ngf=opts.nFeat),
        "big_unet":     UNet3D(in_features=1, n_features=opts.nFeat, out_features=opts.nJoints, n_pool=opts.depth),
        "dual_unet":  DualUNet(),
    }


    if opts.model_name in models:
        print(f"Successfully loaded {opts.model_name}.")
        return models[opts.model_name]
    else:
        raise ValueError(f"Model {opts.model_name} not found. Available models: {list(models.keys())}")