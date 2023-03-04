import os
from pathlib import Path
from typing import Optional

from pyopf.preprocess.parse import parse
from pyopf.preprocess.parse_filepaths import parse_filepaths
from pyopf.run import run_opf
from pyopf.util.Log import Log


class TestOPF:

    def run_opf(self, case: str, objective: str, scenario: Optional[dict] = None):
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
        _opf_results = run_opf(case, objective, case_data_raw, grid_data, filepaths)
        return _opf_results

    def test_ieee14(self):
        _case = "IEEE-14"
        _objective = "min cost"

        _ieee_14_results = self.run_opf(_case, _objective)

        # get the solved status
        _solved = _ieee_14_results["Solved"]["Status"]
        _solved_tc = _ieee_14_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"

    def test_ieee118(self):
        _case = "IEEE-118"
        _objective = "min cost"

        _ieee_118_results = self.run_opf(_case, _objective)

        # get the solved status
        _solved = _ieee_118_results["Solved"]["Status"]
        _solved_tc = _ieee_118_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"

    def test_Texas7k(self):
        _case = "Texas7k"
        _objective = "min cost"

        _texas7k_results = self.run_opf(_case, _objective)

        # get the solved status
        _solved = _texas7k_results["Solved"]["Status"]
        _solved_tc = _texas7k_results["Solved"]["Terminating Condition"]

        assert _solved and _solved_tc == "Feasible and Optimal"
