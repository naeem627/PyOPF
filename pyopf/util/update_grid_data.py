"""
Parse and adjust values of raw grid data before saving to RAW file

Author: Naeem Turner-Bandele

Email: naeem@naeem.engineer

Status: Development


Date Created: December 20, 2022

Last Updated: February 13, 2023

"""

import math

import numpy as np


def update_grid_data(case_data, opf_data, vb=None):
    """
    Adjust grid data based on OPF results
    Args:
        case_data: initial data_utilities data object
        opf_data: opf pyomo model results
        vb: new voltage bounds

    Returns:
        data_utilities case data object

    """
    # Adjust voltage magnitudes and angles #
    for key, ele in opf_data.V_mag.items():
        case_data.raw.buses[key].vm = round(ele.value, 5)
        vr = opf_data.Vr[key].value
        vi = opf_data.Vi[key].value
        case_data.raw.buses[key].va = round(math.degrees(math.atan2(vi, vr)), 4)
        if vb is not None:
            case_data.raw.buses[key].nvlo = vb[0]
            case_data.raw.buses[key].nvhi = vb[1]

    # Adjust generator dispatch and setpoints #
    for key, ele in opf_data.Pg.items():
        v_mag = round(opf_data.V_mag[key[0]].value, 5)
        pg = ele.value
        qg = opf_data.Qg[key].value
        pg_MW = round(pg * 100, 3)
        qg_MVAR = round(qg * 100, 3)

        case_data.raw.generators[key].pg = pg_MW
        case_data.raw.generators[key].qg = qg_MVAR
        case_data.raw.generators[key].vs = v_mag

        if np.abs(pg) < 1E-4:
            case_data.raw.generators[key].pg = 0.0
            case_data.raw.generators[key].pb = 0.0
            case_data.raw.generators[key].pt = 0.0
            qt = case_data.raw.generators[key].qt
            qb = case_data.raw.generators[key].qb

            # if not supplying any active or reactive power then turn the generator off
            if np.abs(qt) < 1E-4 and np.abs(qb) < 1E-4:
                case_data.raw.generators[key].stat = int(0)

    for key, ele in case_data.raw.generators.items():
        v_mag = round(opf_data.V_mag[key[0]].value, 5)
        ele.vs = v_mag

    return case_data
