import argparse
import os
from pathlib import Path
from typing import Optional

from pyopf.parse.argparse_actions import ReadJSON
from pyopf.parse.parse import parse
from pyopf.parse.parse_filepaths import parse_filepaths
from pyopf.run import _run_opf
from pyopf.util.Log import Log


def main(case: str,
         objective: Optional[str] = "min dev",
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

    filepaths = parse_filepaths(case, path_to_root, "cases", scenario, path_to_results, path_to_log)

    # # Parse RAW file and assign grid data to objects # #
    grid_data, case_data_raw = parse(case, filepaths, logger)

    # # Run OPF # #
    _opf_results = _run_opf(case, objective, case_data_raw, grid_data, filepaths)
    return _opf_results


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Parse command line inputs")
    cli_parser.add_argument('--case',
                            help='the name of the network to run',
                            type=str)

    cli_parser.add_argument('--obj',
                            help='the chosen optimization obj to min or max',
                            default='min cost',
                            type=str)

    cli_parser.add_argument('--scenario',
                            help='the scenario to run',
                            action=ReadJSON,
                            default=None)

    args = cli_parser.parse_args()

    main(args.case, args.obj, args.scenario)
