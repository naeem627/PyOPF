"""
Postprocessing functions

Author: Naeem Turner-Bandele
Email: naeem@naeem.engineer
Date Created: December 20, 2022
Last Updated: March 4, 2023

"""
import math
import datetime

import numpy as np
from typing import Optional
from pyopf.preprocess.data_utilities.data import Data
from pyomo.environ import ConcreteModel
import pyopf as po
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
                                                     f"Produced by PyOPF v{po.__version__};" \
                                                     f" Case Version {release_version}."

    modify_transmission_elements(case_data, opf_results, voltage_bounds)

    # P_renewable = sum([val.value for key, val in opf_results.Pu.items()])
    # Q_renewable = sum([ele.value for key, ele in opf_results.Qu.items()])
    #
    # P_load = sum([ele.pl for ele in case_data_raw.raw.loads.values()]) / 100
    # Q_load = sum([ele.ql for ele in case_data_raw.raw.loads.values()]) / 100
    #
    # P_gen = sum([ele.pg for ele in case_data_raw.raw.generators.values() if ele.stat]) / 100
    #
    # Q_gen = sum([ele.qg for ele in case_data_raw.raw.generators.values() if ele.stat]) / 100
    #
    # P_conv_gen = P_gen - P_renewable
    # Q_conv_gen = Q_gen - Q_renewable
    #
    # print(
    #     f"PSCASEMOD - Load P,Q:({P_load},{Q_load}), Conv. Gen P,Q:({P_conv_gen, Q_conv_gen}), Renewable P,"
    #     f"Q: ({P_renewable}"
    #     f",{Q_renewable}), Gen P,Q: ({P_gen}, {Q_gen})")
    return case_data


def modify_transmission_elements(case_data_raw: Data,
                                 opf_results: ConcreteModel,
                                 voltage_bounds: Optional[tuple] = None,
                                 n_vmag_digits: int = 6,
                                 n_gen_digits: int = 6):
    """
    Modify transmission elements based on opf results
    Args:
        case_data_raw: initial data_utilities data object
        opf_results: opf pyomo model results
        voltage_bounds: new voltage bounds
        n_vmag_digits: number of digits used to round voltage magnitudes
        n_gen_digits: number of digits used to round real and reactive power of generators
    Returns:
        data_utilities case data object
    """

    # Adjust voltage magnitudes and angles #
    for key, ele in opf_results.V_mag.items():
        case_data_raw.raw.buses[key].vm = round(ele.value, n_vmag_digits)
        vr = opf_results.Vr[key].value
        vi = opf_results.Vi[key].value
        case_data_raw.raw.buses[key].va = round(math.degrees(math.atan2(vi, vr)), n_vmag_digits)
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
                v_mag = round(opf_results.V_mag[remote_bus].value, n_vmag_digits)
            else:
                v_mag = round(opf_results.V_mag[key[0]].value, n_vmag_digits)
            ele.vs = v_mag

    # update binit of switched shunts
    for key, ele in opf_results.Qsh.items():
        binit = (ele.value * 100) / (opf_results.V_mag[key[0]].value ** 2)
        if binit == 0.:
            case_data_raw.raw.switched_shunts[(key[0],)].status = 0
        else:
            case_data_raw.raw.switched_shunts[(key[0],)].status = 1
            case_data_raw.raw.switched_shunts[(key[0],)].binit = round(binit, 3)
