import argparse

import torch.backends.cudnn as cudnn
from train import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the unet network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_ids', default='-1', dest='gpu_ids')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='resnet', dest='scope')
parser.add_argument('--norm', type=str, default='inorm', dest='norm')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')

parser.add_argument('--name_data', type=str, default='bsr', dest='name_data')
parser.add_argument('--dir_data', default='../datasets', dest='dir_data')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=300, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=1e-4, dest='lr_G')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=321, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=481, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=3, dest='nch_in')

parser.add_argument('--ny_load', type=int, default=64, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=64, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=3, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=64, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=64, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=3, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')

parser.add_argument('--num_freq_disp', type=int,  default=1, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    main()