from model_zoo.small_unet import Unet
from model_zoo.big_unet import UNet3D
from model_zoo.tsm_unet import SeboNet as TSMNet
import torch

def get_models(opts):

    # If using anatomix, increase the number of input channels
    if opts.anatomix is True:
        if opts.concat is True:
            input_nc = 17
        else:
            input_nc = 16
    else:
        input_nc = 1

    models = {
        "small_unet":   Unet(dimension=3, input_nc=input_nc, output_nc=opts.nJoints, num_downs=opts.depth, ngf=opts.nFeat),
        "big_unet":     UNet3D(in_features=1, n_features=opts.nFeat, out_features=opts.nJoints, n_pool=opts.depth),
        #"tsm_unet":     TSMNet(dimension=4, input_nc=input_nc, output_nc=opts.nJoints, ngf=opts.nFeat, num_downs=opts.depth, predict_middle=opts.middle_prediction, num_timepoints=opts.temporal,
        #                       batch_size=opts.batch_size),
    }


    if opts.anatomix:
        amix_model  = Unet(dimension=3, input_nc=1, output_nc=16, ngf=16, num_downs=4)
        amix_model.load_state_dict(torch.load('/data/vision/polina/users/sebodiaz/data/anatomix.pth'), strict=True)
        fin_layer   = torch.nn.InstanceNorm3d(16)
        amix        = torch.nn.Sequential(amix_model, fin_layer)
        print(f"Successfully laoded anatomix.")
    else:
        amix = None

    if opts.model_name in models:
        print(f"Successfully loaded {opts.model_name}.")
        return models[opts.model_name], amix
    else:
        raise ValueError(f"Model {opts.model_name} not found. Available models: {list(models.keys())}")