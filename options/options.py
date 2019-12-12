import argparse
import utils.misc_utils as utils
"""
    Arg parse
    opt = parse_args()
"""

def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default='default',
                             help='folder name to save the outputs')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')

    parser.add_argument('--model', type=str, default='DuRN_US', help='which model to use')
    parser.add_argument('--norm', type=str, default='instance',
                             help='[instance] normalization or [batch] normalization')

    # input/output sizes
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--resize', type=int, default=1024, help='scale images to this size')
    parser.add_argument('--crop', type=int, default=256, help='then crop to this size')


    # for datasets
    parser.add_argument('--data_root', type=str, default='./datasets/')
    parser.add_argument('--dataset', type=str, default='neural')

    # training options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--load', type=str, default=None, help='load checkpoint')
    parser.add_argument('--which-epoch', type=int, default=0, help='which epoch to resume')
    parser.add_argument('--epochs', type=int, default=500, help='epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

    parser.add_argument('--save_freq', type=int, default=10, help='freq to save models')
    parser.add_argument('--eval_freq', type=int, default=25, help='freq to eval models')
    parser.add_argument('--log_freq', type=int, default=1, help='freq to vis in tensorboard')

    # parser.add_argument('--resize_or_crop', type=str, default='scale_width',
    #                          help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    # parser.add_argument('--serial_batches', action='store_true',
    #                          help='if true, takes images in order to make batches, otherwise takes them randomly')
    # parser.add_argument('--no_flip', action='store_true',
    #                          help='if specified, do not flip the images for data argumentation')
    # parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
    #



    return parser.parse_args()


opt = parse_args()
utils.print_args(opt)
