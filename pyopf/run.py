from typing import Optional

from data_utilities.data import Data
from pyopf.OPF import OPF


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
