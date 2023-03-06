"""
Postprocessing functions

Author: Naeem Turner-Bandele
Email: naeem@naeem.engineer
Date Created: December 20, 2022
Last Updated: March 6, 2023

"""
import datetime
import math
from typing import Optional

import numpy as np
from pyomo.environ import ConcreteModel

from pyopf.preprocess.data_utilities.data import Data

__all__ = ["postprocess_case", "modify_records", "modify_transmission_elements"]


def postprocess_case(scenario_name: str,
                     case_data: Data,
                     opf_results: ConcreteModel,
                     voltage_bounds: Optional[tuple] = None,
                     citation: Optional[dict] = None):
    """
    Adjust grid case data based on OPF results
    Args:
        case_data: initial data_utilities data object
        opf_results: opf pyomo model results
        voltage_bounds: new voltage bounds

    Returns:
        data_utilities case data object

    """
    # # Modify Case Identification Records # #
    case_data = modify_records(scenario_name, case_data, citation)

    # # Modify Transmission Elements Based on OPF # #
    case_data = modify_transmission_elements(case_data, opf_results, voltage_bounds)

    return case_data


def modify_records(scenario_name: str,
                   case_data: Data,
                   citation: Optional[dict] = None):
    if citation is not None:
        case_source = citation["source"]
        release_version = citation["version"]  # the release version of the grid network case
    else:
        record_2 = case_data.raw.case_identification.record_2
        case_source = case_data.raw.case_identification.record_2[0] if record_2 is not None else "Not Provided"
        release_version = "Not Provided"

    if case_source is not None:
        case_data.raw.case_identification.record_2 = f"{scenario_name} electric grid; Creator {case_source} "
    else:
        case_data.raw.case_identification.record_2 = f"{scenario_name} electric grid; Source {case_source}"

    if release_version is not None:
        case_data.raw.case_identification.record_3 = f"/ {datetime.datetime.today()}; " \
                                                     f"Produced by PyOPF v0.1.1;" \
                                                     f" Case Version {release_version}."
    return case_data


def modify_transmission_elements(case_data_raw: Data,
                                 opf_results: ConcreteModel,
                                 voltage_bounds: Optional[tuple] = None,
                                 n_v_digits: int = 8,
                                 n_gen_digits: int = 6):
    """
    Modify transmission elements based on opf results
    Args:
        case_data_raw: initial data_utilities data object
        opf_results: opf pyomo model results
        voltage_bounds: new voltage bounds
        n_v_digits: number of digits used to round voltage magnitudes
        n_gen_digits: number of digits used to round real and reactive power of generators
    Returns:
        data_utilities case data object
    """

    # Adjust voltage magnitudes and angles #
    for key, ele in opf_results.V_mag.items():
        case_data_raw.raw.buses[key].vm = round(ele.value, n_v_digits)
        vr = opf_results.Vr[key].value
        vi = opf_results.Vi[key].value
        case_data_raw.raw.buses[key].va = round(math.degrees(math.atan2(vi, vr)), n_v_digits)
        if voltage_bounds is not None:
            case_data_raw.raw.buses[key].nvlo = voltage_bounds[0]
            case_data_raw.raw.buses[key].nvhi = voltage_bounds[1]

    # Adjust generator dispatch and setpoints #
    for key, ele in opf_results.Pg.items():
        pg = ele.value
        qg = opf_results.Qg[key].value
        pg_MW = round(pg * 100, n_gen_digits)
        qg_MVAR = round(qg * 100, n_gen_digits)

        # generator is online
        if key[0] in opf_results.V_mag:
            case_data_raw.raw.generators[key].pg = pg_MW
            case_data_raw.raw.generators[key].qg = qg_MVAR

            # set the lower limit of the dispatch to the current value so that it does not dip below
            case_data_raw.raw.generators[key].pb = pg_MW

            # if not supplying any active or reactive power then turn the generator off
            if np.abs(pg_MW) < (10 ** -n_gen_digits) and np.abs(qg_MVAR) < (10 ** -n_gen_digits):
                case_data_raw.raw.generators[key].pb = 0.0
                case_data_raw.raw.generators[key].pt = 0.0
                case_data_raw.raw.generators[key].qb = 0.0
                case_data_raw.raw.generators[key].qt = 0.0
                case_data_raw.raw.generators[key].stat = int(0)
            else:
                # leave generator on
                case_data_raw.raw.generators[key].stat = int(1)
        else:
            # generator is offline
            case_data_raw.raw.generators[key].pb = 0.0
            case_data_raw.raw.generators[key].pt = 0.0
            case_data_raw.raw.generators[key].qb = 0.0
            case_data_raw.raw.generators[key].qt = 0.0
            case_data_raw.raw.generators[key].stat = int(0)

    # Modify voltage setpoints of generators
    for key, ele in case_data_raw.raw.generators.items():
        if key[0] in opf_results.V_mag:
            remote_bus = case_data_raw.raw.generators[key].ireg
            if remote_bus != 0:
                v_mag = round(opf_results.V_mag[remote_bus].value, n_v_digits)
            else:
                v_mag = round(opf_results.V_mag[key[0]].value, n_v_digits)
            ele.vs = v_mag

    # update binit of switched shunts
    for key, ele in opf_results.Qsh.items():
        binit = (ele.value * 100) / (opf_results.V_mag[key[0]].value ** 2)
        if binit == 0.:
            case_data_raw.raw.switched_shunts[(key[0],)].status = 0
        else:
            case_data_raw.raw.switched_shunts[(key[0],)].status = 1
            case_data_raw.raw.switched_shunts[(key[0],)].binit = round(binit, 3)

    return case_data_raw
