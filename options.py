"""" 
Configuration file for the project. Please read _very_ carefully, as a lot of these will change
the behavior of the code.

Originally wrriten by Sebo 01/08/2025. Code adapted from Molin Zhang and Junshen Xu.
"""

# Import the necessary libraries
import argparse
import os
import datetime
import utils
import wandb
from pathlib import Path
import json

# Define the optional class
class Options:
    def __init__(self):
        # create the parser
        self.parser         = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized    = False 
        
    # Define the default options
    def initialize(self):
        # Mode and general settings
        self.parser.add_argument('--stage', type=utils.check_arg(str, ['train', 'test', 'inference', 'finetune']), default='train',
                                 help='set the stage of the code')
        self.parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                                 help='name of the run')
        self.parser.add_argument('--save_path', type=str, default='/unborn/sdd/PyTorchPose/runs/',
                                    help='path to save the checkpoints')
        self.parser.add_argument('--seed', type=int, default=None,
                                    help='set the seed for the code, if not then a random seed will be used')
        self.parser.add_argument('--use_amp', type=bool, default=False,
                                    help='whether to use automatic mixed precision... do not use if you do not know what it is')
        self.parser.add_argument('--use_fabric', type=bool, default=False,
                                    help='whether to use lightning fabric')
        self.parser.add_argument('--data_partition_file', type=str, default='data_partition.yml',
                                 help='path to the data partition file')
        self.parser.add_argument('--csail', type=bool, default=False,
                                 help='whether using csail cluster or not')
        self.parser.add_argument('--baseline', type=bool, default=False,
                                 help='whether to use the baseline model')
        self.parser.add_argument('--tsm', type=bool, default=False,
                                 help='whether to use the tsm model')
        self.parser.add_argument('--middle_prediction', type=bool, default=False,
                                  help='whether to predict the middle frame')
        self.parser.add_argument('--four_dim', type=bool, default=False,
                                 help='whether to use a 4d unet')
        self.parser.add_argument('--num_nodes', type=int, default=1,
                                 help='# of nodes to train on')
        self.parser.add_argument('--num_gpus', type=int, default=1,
                                 help='# of gpus to train on')
        self.parser.add_argument('--model_name', type=str, default='small_unet',
                                 help='# of gpus to train on')
       
        # Data location
        self.parser.add_argument('--rawdata_path', type=str, default='/unborn/shared/fetal_pose/fetalEPI',
                                 help='path to the raw EPI data')
        self.parser.add_argument('--label_path', type=str, default='/unborn/shared/SeboPoseLabel',
                                 help='path to the label *.mat files')
        
        # Data processing
        self.parser.add_argument('--crop_size', type=int, default=None,
                                 help='size of the cropped volume for training and validation')
        self.parser.add_argument('--custom_augmentation', type=bool, default=False,
                                 help='whether to use custom augmentation; if True, use TorchIO augmentation. This is for the `offline` augmentation')
        self.parser.add_argument('--mag', type=float, default=10.0,
                                 help='magnitude of the heatmap')
        self.parser.add_argument('--sigma', type=float, default=2.0,
                                 help='standard deviation of the heatmap')        
        self.parser.add_argument('--junshen_scale', type=float, default=None,
                                    help='scale factor for the Junshen approach')
        self.parser.add_argument('--dataset_size', type=int, default=None,
                                    help='# of samples to use for training. If `None`, use the full dataset')
        self.parser.add_argument('--norm_type', type=utils.check_arg(str, ['percentile', 'percentile_window']), default='percentile',
                                    help='type of normalization to use')
        
        # General augmentation settings
        self.parser.add_argument('--augmentation_prob', type=float, default=0.9,
                                 help='probability of applying any given augmentation')
        self.parser.add_argument('--use_zoom_womb', type=bool, default=False,
                                    help='whether to use the copy and paste augmentation')
        
        # Offline training settings
        self.parser.add_argument('--rot', type=bool, default=True,
                                 help='whether to use rotation augmentation. Use in `offline` training')
        self.parser.add_argument('--zoom_factor', type=float, default=1.5,
                                 help='zoom factor top range')
        self.parser.add_argument('--batch_size', type=int, default=None,
                                 help='batch size')

        # Other methods
        self.parser.add_argument('--dsnt', type=bool, default=False,
                                    help='whether to implement dsnt')
        
        # Model hyperparameters
        self.parser.add_argument('--unet_type', type=str, default=['big', 'small'],
                                    help='whether to use Neels network')
        self.parser.add_argument('--depth', type=int, default=4,
                                 help='number of pooling layers')
        self.parser.add_argument('--nFeat', type=int, default=64,
                                help='number of features for the first convolutional layer')
        self.parser.add_argument('--nJoints', type=int, default=15,
                                 help='number of output features usually 15 for _singleton_ fetal pose')
        self.parser.add_argument('--use_bias', type=bool, default=False,
                                    help='whether to use bias in the convolutional layers')
        self.parser.add_argument('--norm_layer', type=utils.check_arg(str, ['bn', 'in', 'gn32', 'gn16']), default='bn',
                                    help='norm layer type')
        
        # Optimizer
        self.parser.add_argument('--optimizer', type=utils.check_arg(str, ['adam', 'adamw']), default='adam',
                                 help='optimizer')
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                help='learning rate; default is 1e-3 for bs 8 -- may need to be adjusted for other batch sizes')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                                 help='weight decay, only used for AdamW')
        self.parser.add_argument('--lr_scheduler', type=utils.check_arg(str, ['cosine', 'linear', 'constant', 'cosinewarmup', 'onecycle']), default='linear',
                                 help='learning rate scheduler, see optimizer.py for more details')
        self.parser.add_argument('--lr_decay_ep', type=int, default=13,
                                 help='epoch to decay the learning rate for cosine scheduler')
        self.parser.add_argument('--warmup_epochs', type=int, default=5,
                                    help='number of warmup epochs')
        
        # Training settings
        self.parser.add_argument('--epochs', type=int, default=300,
                                help='number of epochs')
        self.parser.add_argument('--val_freq', type=int, default=20,
                                    help='frequency of validation')
        self.parser.add_argument('--steps_per_epoch', type=int, default=2000,
                                    help='steps per epoch for online augmentation')
                
        # Loss settings
        self.parser.add_argument('--loss', type=utils.check_arg(str, ['l1', 'mse', 'smoothl1', 'ncc', 'dice_ce', 'adaptive', 'split', 'repulsive']), default='mse',
                                    help='loss function')
        self.parser.add_argument('--joint_consistency', type=bool, default=False,
                                 help='whether to use joint consistency loss. This is not refined yet.')
        self.parser.add_argument('--jc_weight', type=float, default=1e-4,
                                help='weight of the joint consistency loss. This is not refined yet.')
        
        # Hardware settings
        self.parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                                 help="List of GPU IDs")
        self.parser.add_argument('--ngpu', type=int, default=1, 
                                 help='number of GPUs')
        self.parser.add_argument('--num_workers', type=int, default=12,
                                 help='number of workers... needs to be optimized for fast training') # number of workers for data loading
        
        # Test settings
        self.parser.add_argument('--continue_path', type=str, default=None,
                                 help='path to the checkpoint to continue training from')
        self.parser.add_argument('--run_id', type=str, default='',
                                 help='run ID for the wandb project')
        self.parser.add_argument('--error_threshold', type=float, default=5.0,
                                    help='error threshold for PCK metrics')
        
        # Inference settings
        self.parser.add_argument('--output_vis', type=bool, default=False,
                                 help='whether to output visualizations to *.mat files... inference setting')
        self.parser.add_argument('--output_path', type=str, default='',
                                 help='path to save the output *.mat files')
        self.parser.add_argument('--label_name', type=str, default='output',
                                 help='name of the output *.mat files')
        self.parser.add_argument('--index_type', type=int, default=1,
                                 help='index type for the inference... 0 for python or slicer, 1 for matlab and itk-snap')
        self.parser.add_argument('--top_k', type=int, default=None,
                                 help='top k for keypoint assignment')
        self.parser.add_argument('--resample', type=bool, default=False,
                                 help='whether to resample the volumes prior to inference')
        self.parser.add_argument('--output_vol', type=bool, default=False,
                                    help='whether to output the volumes')
        self.parser.add_argument('--condition', type=int, default=1,
                                 help='number of forward passes per frame estimate')
        
        

        # augmentation specific settings
        self.parser.add_argument('--zoom', type=float, default=0.5,
                                 help='probability of applying zoom augmentation')
        self.parser.add_argument('--noise', type=bool, default=False,
                                 help='whether to use random noise augmentation')
        self.parser.add_argument('--spike', type=bool, default=False,
                                 help='whether to use random spike augmentation')
        self.parser.add_argument('--bfield', type=bool, default=False,
                                 help='whether to use random bias field augmentation')
        self.parser.add_argument('--gamma', type=bool, default=False,
                                 help='whether to use random gamma correction augmentation')
        self.parser.add_argument('--anisotropy', type=bool, default=False,
                                 help='whether to use random anisotropy augmentation')
        self.parser.add_argument('--anatomix', type=bool, default=False,
                                 help='whether to use anatomix pre-trained')
        self.parser.add_argument('--concat', type=bool, default=True,
                                 help='True if we want amix features + raw volume. False is just amix features.')
                
        # Intialize the parser
        self.initialized = True
        
    def parse(self):
        # Parse the options
        if not self.initialized:
            self.initialize()
    
        # Parse the arguments
        self.opt = self.parser.parse_args()

        # Directories
        if self.opt.stage == 'train' or self.opt.stage == 'finetune':
            self.opt.save_path = os.path.join(self.opt.save_path, self.opt.run_name)
        
        # Log the time
        self.opt.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        if self.opt.run_name == '':
            self.opt.run_name = self.opt.time
            
        if self.opt.continue_path is not None:
            print(f'Continuing training from {self.opt.continue_path}')
            # assert the checkpoint path exists
            assert os.path.exists(self.opt.continue_path), 'Checkpoint path does not exist. Please check the path and try again.'
        
        #if self.opt.tsm and self.opt.four_dim == True:
        #    raise AssertionError("Both TSM and 4D-Unet cannot be True.")
        
        #if self.opt.temporal == 2 and self.opt.middle_prediction:
        #    raise AssertionError("Temporal and middle prediction are not implemented together.")
        
        if self.opt.model_name == "tsm_unet":
            self.opt.tsm = True
        if self.opt.model_name in ["tsm_unet", "4d_unet"]:
            self.opt.four_dim = True

        # Get the batch sizing to be identical if not set
        if self.opt.stage == 'train':
            if self.opt.batch_size is not None:
                self.opt.online_batch_size = self.opt.batch_size
            elif self.opt.online_batch_size is not None:
                self.opt.batch_size = self.opt.online_batch_size
            else:
                raise ValueError('Batch size must be specified.')
            
            # Get the crop sizing to be identical if not set
            if self.opt.crop_size is not None:
                self.opt.online_crop_size = self.opt.crop_size
            elif self.opt.online_crop_size is not None:
                self.opt.crop_size = self.opt.online_crop_size
            else:
                raise ValueError('Crop size must be specified.')
        
        # GPU settings
        if self.opt.csail is False:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.opt.gpu_ids])
            if self.opt.stage == 'train':
                self.opt.batch_size     = self.opt.batch_size * self.opt.ngpu   # multiply the batch size by the number of GPUs
                self.opt.lr             = self.opt.lr * self.opt.ngpu           # multiply the learning rate by the number of GPUs
            self.opt.num_workers    = self.opt.num_workers * self.opt.ngpu  # multiply the number of workers by the number of GPUs
        
        
        
        # Create the save path if it does not exist iff the stage is train
        # Additionally, save the code to the save path
        if self.opt.stage == 'train' or self.opt.stage == 'finetune':
            # WandB project initialization
            if self.opt.run_id == '':
                if self.opt.csail is True:
                    wandb_path = Path('/data/vision/polina/users/sebodiaz/projects/pose/wandb/wandb.json').expanduser()
                    with open(wandb_path) as fp:
                        mykey = json.load(fp)['key']
                    wandb.login(key = mykey)
                    wandb.init(project='PyTorchPose', name=self.opt.run_name, config=vars(self.opt), dir='/data/vision/polina/users/sebodiaz/projects/pose/',
                               group="DDP")
                elif self.opt.csail is False:
                    wandb.init(project='PyTorchPose', name=self.opt.run_name, config=vars(self.opt))
            elif self.opt.continue_path is not None and self.opt.run_id != '':
                wandb.init(project='PyTorchPose', name=self.opt.run_name, id=self.opt.run_id, resume='must')
            
            if self.opt.continue_path is None and self.opt.run_id == '':
                # Create run directory
                print('Creating run directory...')
                checkpoint_path = os.path.join(self.opt.save_path, 'checkpoints')
                utils.mkdir(self.opt.save_path)
                utils.mkdir(checkpoint_path)
                utils.save_yaml(os.path.join(self.opt.save_path, 'opt.yaml'), self.opt)

                if self.opt.csail is True:
                    utils.copyfiles('/data/vision/polina/users/sebodiaz/projects/pose/', os.path.join(self.opt.save_path, 'backup_code'), '*.py')
                    utils.copyfiles('/data/vision/polina/users/sebodiaz/projects/pose/', os.path.join(self.opt.save_path, 'backup_code'), '*.sh')
                else:
                    utils.copyfiles('./', os.path.join(self.opt.save_path, 'backup_code'), '*.py')
                    utils.copyfiles('./', os.path.join(self.opt.save_path, 'backup_code'), '*.sh')

            if self.opt.stage == 'finetune':
                # Create run directory
                print('Creating run directory at location:', self.opt.save_path)
                checkpoint_path = os.path.join(self.opt.save_path, 'checkpoints')
                utils.mkdir(self.opt.save_path)
                utils.mkdir(checkpoint_path)
                utils.save_yaml(os.path.join(self.opt.save_path, 'opt.yaml'), self.opt)

                if self.opt.csail is True:
                    utils.copyfiles('/data/vision/polina/users/sebodiaz/projects/pose/', os.path.join(self.opt.save_path, 'backup_code'), '*.py')
                    utils.copyfiles('/data/vision/polina/users/sebodiaz/projects/pose/', os.path.join(self.opt.save_path, 'backup_code'), '*.sh')
                
                else:
                    utils.copyfiles('./', os.path.join(self.opt.save_path, 'backup_code'), '*.py')
                    utils.copyfiles('./', os.path.join(self.opt.save_path, 'backup_code'), '*.sh')

        if self.opt.baseline is True:
            print(f'Using baseline model dataset.')


        # Mandate, output path, and output label for inference
        if self.opt.stage == 'inference':
            assert self.opt.output_path != '', 'An output path must be specified for inference.'
            assert self.opt.label_name != '', 'A label name must be specified for inference.'
        
        # Print the options
        utils.print_args(self.opt)


        # print the augmentations:
        print(f'Using dsnt:', self.opt.dsnt)
        print(f'Using noise: ', self.opt.noise)
        print(f'Using gamma: ', self.opt.gamma)
        print(f'Using spike: ', self.opt.spike)
        print(f'Using bfield: ', self.opt.bfield)
        print(f'Using aniso: ', self.opt.anisotropy)
        print(f'Using zoom: ', self.opt.zoom > 0)

        
        # Determine if a random seed is set
        if self.opt.seed is not None:
            utils.set_seeds(self.opt.seed)
        elif self.opt.seed is None and self.opt.stage == 'train':
            # Record the seeds used
            tseed, npseed, rngseed = utils.get_seeds()
            
            # Record to WandB
            if self.opt.continue_path is None:
                wandb.config.update({'torch_seed': tseed, 'numpy_seed': npseed, 'random_seed': rngseed})
        
        return self.opt
    
if __name__ == '__main__':
    opt = Options().parse()
