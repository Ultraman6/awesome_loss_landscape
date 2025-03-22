# Notes:
# - accept loss landscape as a list of lists
# - return csv files provided by paraview
from typing import List
from ttk_functions import (
    loss_landscape_to_vti,
    loss_landscape_to_vtu,
    compute_persistence_barcode_paraview,
    process_persistence_barcode,
    compute_merge_tree_paraview,
    compute_merge_tree_planar_paraview,
    process_merge_tree,
    process_merge_tree_planar,

)

def compute_persistence_barcode(
    loss_landscape: List[List[float]] = None,
    loss_coords: List[List[float]] = None,
    loss_values: List[float] = None,
    embedding: List[List[float]] = None,
    dim=2,
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
    output_path: str = "",
    vtk_format: str = "vti",
    graph_kwargs: str = "aknn",
    n_neighbors=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> list:

    # convert loss_landscape into a vtk format
    output_file_vtk = None
    if vtk_format.lower() == "vti":

        # convert loss_landscape into a (.vti) image data format
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            dim=dim,
            loss_steps_dim1=loss_steps_dim1,
            loss_steps_dim2=loss_steps_dim2,
            loss_steps_dim3=loss_steps_dim3,
            output_path=output_path,
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            output_path=output_path,
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors,
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute persistence_barcode
    output_file_csv = compute_persistence_barcode_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold,
        threshold_is_absolute=threshold_is_absolute,
    )

    # extract .csv and return persistence_barcode object
    persistence_barcode = process_persistence_barcode(output_file_csv)

    return persistence_barcode


def compute_merge_tree(
    loss_landscape: List[List[float]] = None,
    loss_coords: List[List[float]] = None,
    loss_values: List[float] = None,
    embedding: List[List[float]] = None,
    dim=2,
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
    output_path: str = "",
    vtk_format: str = "vti",
    graph_kwargs: str = "aknn",
    n_neighbors=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> dict:

    # convert loss_landscape into a vtk format
    output_file_vtk = None
    if vtk_format.lower() == "vti":

        # convert loss_landscape into a (.vti) image data format
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            dim=dim,
            loss_steps_dim1=loss_steps_dim1,
            loss_steps_dim2=loss_steps_dim2,
            loss_steps_dim3=loss_steps_dim3,
            output_path=output_path,
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            output_path=output_path,
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors,
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute merge_tree
    output_file_csv = compute_merge_tree_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold,
        threshold_is_absolute=threshold_is_absolute,
    )

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree(output_file_csv)

    return merge_tree


def compute_merge_tree_planar(
    loss_landscape: List[List[float]] = None,
    loss_coords: List[List[float]] = None,
    loss_values: List[float] = None,
    embedding: List[List[float]] = None,
    dim=2,
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
    output_path: str = "",
    vtk_format: str = "vti",
    graph_kwargs: str = "aknn",
    n_neighbors=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> dict:

    ### TODO: maybe just make planar=False an argument of compute_merge_tree

    # convert loss_landscape into a vtk format
    output_file_vtk = None
    if vtk_format.lower() == "vti":

        # convert loss_landscape into a (.vti) image data format
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            dim=dim,
            loss_steps_dim1=loss_steps_dim1,
            loss_steps_dim2=loss_steps_dim2,
            loss_steps_dim3=loss_steps_dim3,
            output_path=output_path,
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            output_path=output_path,
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors,
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute merge_tree
    output_file_csv = compute_merge_tree_planar_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold,
        threshold_is_absolute=threshold_is_absolute,
    )
    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree_planar(output_file_csv)
    return merge_tree

print("Finish generating paraview files.")
