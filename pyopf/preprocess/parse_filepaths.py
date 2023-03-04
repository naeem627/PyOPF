import os
from typing import Optional


def parse_filepaths(case: str,
                    root_dir: str,
                    networks_dir: str,
                    scenario: Optional[dict] = None,
                    results_dir: Optional[str] = None,
                    log_dir: Optional[str] = None) -> dict:
    """
    Compile and create a directory that contains all filepaths needed to solve the program
    Args:
        case: the name of the case being solved
        root_dir: the path from the current directory to the root directory
        networks_dir: the directory containing the case data
        scenario: the dictionary containing information about the current scenario being run
        results_dir: the path to the results directory
        log_dir: the path to the logger directory

    Returns:
        A dictionary with all the relevant filepaths stored
    """
    networks_dir = f"{root_dir}/{networks_dir}"
    case_dir = f"{networks_dir}/{case}"

    path_to_prob_settings = os.path.normpath(f"{networks_dir}/{case}/{case}-Prob-Settings.json")

    if scenario is None:
        path_to_case = case_dir
    else:
        case = f"{case}_{scenario['name']}_"
        path_to_case = f"{case_dir}/{scenario['dir']}"

    path_to_case = os.path.normpath(path_to_case)
    path_to_raw = os.path.normpath(f"{path_to_case}/{case}.RAW")

    if not os.path.exists(path_to_prob_settings):
        path_to_prob_settings = None

    path_to_networks = os.path.normpath(networks_dir)

    if results_dir is None:
        if scenario is not None:
            path_to_results = f"{root_dir}/results/{case}/{scenario['dir']}/{scenario['name']}"
        else:
            path_to_results = f"{root_dir}/results/{case}"
    else:
        if scenario is not None:
            path_to_results = f"{results_dir}/{scenario['name']}"
        else:
            path_to_results = f"{results_dir}"

    path_to_results = os.path.normpath(path_to_results)

    path_to_optimized_cases = os.path.normpath(f"{path_to_case}/OPF_solved")

    filepaths = {
        "root": root_dir,
        "log": log_dir,
        "case": path_to_case,
        "networks": path_to_networks,
        "raw": path_to_raw,
        "prob settings": path_to_prob_settings,
        "results": path_to_results,
        "optimized cases": path_to_optimized_cases
    }
    return filepaths
