import argparse
import os.path

def reduce_save_folder(args):
    dataname = 'eval' if args.eval else 'train'
    dataname += '_cls=' + 'all' if not args.cls_idxes else str(args.cls_idxes)
    save_folder = 'm=' + str(args.metric)
    save_folder += '_d=' + str(args.dim)
    save_folder += '_be=' + str(args.begin_epoch)
    save_folder += '_ce=' + str(args.center_epoch)
    save_folder += '_es=' + str(args.end_epoch)
    save_folder += '_no=' + str(args.normalize)
    save_folder += '_ig=' + str(args.ignore)
    save_folder += '_dt=' + str(args.dtype)
    return str(os.path.join('landscape', dataname, save_folder))

parser = argparse.ArgumentParser()
# use for outer benchmark loaded by folder
parser.add_argument("--model_file", default=None)
parser.add_argument("--model_folder", default=None)
parser.add_argument("--data_file", default=None)
parser.add_argument('--criterion_file', default=None)

parser.add_argument('--eval', default=True, type=bool, help='use test dataset for landscape')
parser.add_argument('--save_root', default='../records')
parser.add_argument('--begin_epoch', default=0, type=int)
parser.add_argument('--center_epoch', default=-2, type=int, help='-1 means the last epoch -2 means the best epoch')
parser.add_argument('--end_epoch', default=-1, type=int, help='-1 means the last epoch')
parser.add_argument('--signs', nargs='+', default=('directions', 'grid', 'trajectory'))
parser.add_argument('--saves', nargs='+', default=('directions', 'grid', 'trajectory'))
parser.add_argument('--loads', nargs='+', default=('directions', ''))
parser.add_argument('--cls_idxes', default=())

parser.add_argument('--device', default='cuda')
parser.add_argument('--metric', default='loss', choices=['loss', 'eigen', 'logit'])
parser.add_argument('--classes', default=())
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--steps', default=50, type=int)
parser.add_argument('--distance', default=1.0, type=float)
parser.add_argument('--grid_threads', default=1, type=int)
parser.add_argument('--normalize', default='filter', help='direction normalization: filter | layer | weight')
parser.add_argument('--ignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
parser.add_argument('--dtype', default='weight',
                    help='direction type: weight | state (including BN\'s running_mean/var)')
parser.add_argument('--center', default=True, type=bool)
parser.add_argument('--dim', default=2, help='dimension of the landscape')
parser.add_argument('--reduce', default='eigen', choices=['random', 'pca', 'inter', 'eigen'])
parser.add_argument('--project', default='cos', choices=['cos', 'lstsq'])
parser.add_argument('--end_root', default='', type=str, help='root directory for the end model')

parser.add_argument('--metric_bs', default=256, type=int)
args = parser.parse_args()