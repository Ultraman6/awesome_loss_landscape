import argparse
import os.path

from landscape import reduce_args
parser = argparse.ArgumentParser()
# visualize参数
parser.add_argument("--coords_file", default=None, help="input npy file")
parser.add_argument("--values_file", default=None, help="input npy file")
parser.add_argument("--grid_file", default=None, help="input npy file")
parser.add_argument("--output_path", default=None, help="output file name (no extension)")
parser.add_argument("--visual_mode", default='surface')
parser.add_argument("--vtk_format", default="vtu", help="output file format (vti or vtu)")
parser.add_argument("--graph_kwargs", default="aknn", help="algorithm for constructing graph")
parser.add_argument("--persistence_threshold", type=float, default=0, help="Threshold for simplification by persistence (use --threshold-is-absolute if passing a scalar value.")
parser.add_argument("--threshold_is_absolute", default=False, help="Is the threshold an absolute scalar value or a fraction (0 - 1) of the function range.")

args = parser.parse_args()
args.output_path = os.path.join(reduce_args.save_root, )