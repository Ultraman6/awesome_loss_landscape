import argparse
import os.path

def reduce_save_folder(args):
    save_folder = 'sd=' + str(args.seed)
    save_folder += '_be=' + str(args.begin_epoch)
    save_folder += '_ce=' + str(args.center_epoch)
    save_folder += '_es=' + str(args.end_epoch)
    save_folder += '_se=' + str(args.save_epoch)
    save_folder += '_dis=' + str(args.distance)
    save_folder += '_st=' + str(args.steps)
    save_folder += '_no=' + str(args.normalize)
    save_folder += '_ig=' + str(args.ignore)
    save_folder += '_dt=' + str(args.dtype)
    save_folder += '_ct=' + str(args.center)
    return save_folder

parser = argparse.ArgumentParser()
parser.add_argument('--eval', default=False, type=bool, help='use test dataset for landscape')
parser.add_argument('--save_root', default='./records')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--begin_epoch', default=0, type=int)
parser.add_argument('--center_epoch', default=-2, type=int, help='-1 means the last epoch -2 means the best epoch')
parser.add_argument('--end_epoch', default=100, type=int)
parser.add_argument('--steps', default=50, type=int)
parser.add_argument('--distance', default=1.0, type=float)
parser.add_argument('--normalize', default='filter', help='direction normalization: filter | layer | weight')
parser.add_argument('--ignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
parser.add_argument('--dtype', default='weights',
                    help='direction type: weights | states (including BN\'s running_mean/var)')
parser.add_argument('--center', default=True, type=bool)
parser.add_argument('--dim', default=2, help='dimension of the landscape')
parser.add_argument('--reduce', default='random', choices=['random', 'pca', 'inter', 'eigen'])
parser.add_argument('--project', default='cos', choices=['cos', 'lstsq'])
parser.add_argument('--end_root', default='', type=str, help='root directory for the end models')

# visualize参数
parser.add_argument("--visual_mode", default='merge_tree')
parser.add_argument("--coords_file", default=None, help="input npy file")
parser.add_argument("--values_file", default=None, help="input npy file")
parser.add_argument("--grid_file", default=None, help="input npy file")
parser.add_argument("--output_path", default=None, help="output file name (no extension)")
parser.add_argument("--vtk_format", default="vtu", help="output file format (vti or vtu)")
parser.add_argument("--graph_kwargs", default="aknn", help="algorithm for constructing graph")
parser.add_argument("--persistence_threshold", type=float, default=0, help="Threshold for simplification by persistence (use --threshold-is-absolute if passing a scalar value.")
parser.add_argument("--threshold_is_absolute", default=False, help="Is the threshold an absolute scalar value or a fraction (0 - 1) of the function range.")

args = parser.parse_args()
args.save_root = os.path.join(args.save_root, reduce_save_folder(args))