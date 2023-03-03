import os
from pathlib import Path
from typing import Optional

from data_utilities.data import Data
from pyopf.OPF import OPF
from pyopf.parse.parse import parse
from pyopf.parse.parse_filepaths import parse_filepaths
from pyopf.util.Log import Log


def _run_opf(case: str,
             objective: str,
             case_data_raw: Data,
             grid_data: dict,
             filepaths: dict,
             scenario: Optional[dict] = None):
    # # === Run OPF Scenario === # #
    if scenario is not None:
        scenario_name = f"{case}_{scenario['name']}"
    else:
        scenario_name = case

    # set up the OPF solver
    opf = OPF()

    # create and solve the OPF model
    opf.solve(scenario_name, grid_data, filepaths, objective=objective)

    # # === Save the Solved OPF Model === # #
    opf.save_solution(case_data_raw, filepaths)

    return opf.results_summary


def run_opf(case: str,
            dir_cases: str,
            objective: Optional[str] = "min cost",
            scenario: Optional[dict] = None):
    # Get root directory path
    path_to_root = os.getcwd()

    # # Create Logger # #
    path_to_log = path_to_root + os.path.sep + 'log' + os.path.sep
    Path(path_to_log).mkdir(parents=True, exist_ok=True)
    logger = Log(path_to_log, case)

    # # Create Path to Results # #
    path_to_results = f"{path_to_root}/results/{case}"
    if scenario is not None:
        path_to_results = f"{path_to_root}/results/{case}/{scenario['dir']}"
    Path(path_to_results).mkdir(parents=True, exist_ok=True)

    filepaths = parse_filepaths(case, path_to_root, dir_cases, scenario, path_to_results, path_to_log)

    # # Parse RAW file and assign grid data to objects # #
    grid_data, case_data_raw = parse(case, filepaths, logger)

    # # Run OPF # #
    _opf_results = _run_opf(case, objective, case_data_raw, grid_data, filepaths)
    return _opf_results
