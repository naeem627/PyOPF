from os.path import normpath
from pathlib import Path
from typing import Union

import numpy as np


def save_compressed_results(filepaths: dict,
                            filename: str,
                            results: Union[dict, float, np.ndarray, list]):
    """
    Save results as a compressed numpy archive
    Args:
        filepaths: dictionary that tracks the paths to different directories
        filename: the name of the results file
        results: the results data

    Returns:
        None
    """

    Path(filepaths["results"]).mkdir(parents=True, exist_ok=True)
    path_to_results_file = normpath(f"{filepaths['results']}/{filename}.npz")
    np.savez_compressed(path_to_results_file, results=results)
