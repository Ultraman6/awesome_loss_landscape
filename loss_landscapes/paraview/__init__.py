import argparse
from turn_landscape_to_csv import compute_persistence_barcode, compute_merge_tree, compute_merge_tree_planar

# Load the parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--loss-coords-file", default=None, help="input npy file")
parser.add_argument("--loss-values-file", default=None, help="input npy file")
parser.add_argument("--output-path", default=None, help="output file name (no extension)")
parser.add_argument("--output-folder", default=None, help="output folder name (use instead of --output-path)")
parser.add_argument("--vtk-format", default="vtu", help="output file format (vti or vtu)")
parser.add_argument("--graph-kwargs", default="aknn", help="algorithm for constructing graph")
parser.add_argument("--persistence-threshold", type=float, default=0, help="Threshold for simplification by persistence (use --threshold-is-absolute if passing a scalar value.")
parser.add_argument("--threshold-is-absolute", default=False, help="Is the threshold an absolute scalar value or a fraction (0 - 1) of the function range.")
args = parser.parse_args()

# Check output path
if args.output_path is None:
    if args.output_folder is None:
        args.output_path = args.loss_values_file.replace(".npy", "").replace("loss_landscape_files", "paraview_files")
    else:
        args.output_path = f"{args.output_folder}/{os.path.basename(args.loss_values_file.replace('.npy', ''))}"
elif args.output_path.endswith(".npy"):
    args.output_path = args.output_path.replace(".npy", "")

# # Load loss_landscape from a file
# loss_coords = np.load(args.loss_coords_file)
# loss_values = np.load(args.loss_values_file)
#
# # Use VTK Unstructured Grid
# vtk_format = args.vtk_format
