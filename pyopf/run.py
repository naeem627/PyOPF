import os
from pathlib import Path
from typing import Optional

from pyopf.OPF import OPF
from pyopf.postprocess.postprocess import postprocess_case
from pyopf.preprocess.data_utilities import Data
from pyopf.preprocess.parse import parse
from pyopf.preprocess.parse_filepaths import parse_filepaths
from pyopf.util.Log import Log

__all__ = ["run_opf", "execute_opf"]


def execute_opf(case: str,
                objective: str,
                case_data: Data,
                transmission_elements: dict,
                filepaths: dict,
                scenario: Optional[dict] = None,
                options: Optional[dict] = None):
    if options is not None:
        voltage_bounds = options.get("voltage bounds", None)
    else:
        voltage_bounds = None

    # # === Run OPF Scenario === # #
    if scenario is not None:
        scenario_name = f"{case}_{scenario['name']}"
    else:
        scenario_name = case

    # set up the OPF solver
    opf = OPF()

    # create and solve the OPF model
    opf.solve(scenario_name, transmission_elements, filepaths, objective=objective)

    # # === Save the Solved OPF Model === # #
    case_data = postprocess_case(scenario_name, case_data, opf.model, voltage_bounds)
    opf.save_solution(case_data, filepaths)

    return opf.results_summary


def run_opf(case: str,
            dir_cases: str,
            objective: Optional[str] = "min cost",
            scenario: Optional[dict] = None,
            options: Optional[dict] = None):
    # Get root directory path
    root_dir = os.getcwd()

    # # Create Logger # #
    path_to_log = f"{root_dir}/log/"
    Path(path_to_log).mkdir(parents=True, exist_ok=True)
    logger = Log(path_to_log, case)

    # # Create Path to Results # #
    path_to_results = f"{root_dir}/results/{case}"
    if scenario is not None:
        path_to_results = f"{root_dir}/results/{case}/{scenario['dir']}"
    Path(path_to_results).mkdir(parents=True, exist_ok=True)

    filepaths = parse_filepaths(case, root_dir, dir_cases, scenario, path_to_results, path_to_log)

    # # Parse RAW file and assign grid data to objects # #
    transmission_elements, case_data = parse(case, filepaths, logger)

    # # Run OPF # #
    _opf_results = execute_opf(case, objective, case_data, transmission_elements, filepaths, scenario, options)
    return _opf_results
