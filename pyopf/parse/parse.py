import warnings

from data_utilities.data import Data
from pyopf.models.GridData import Branch, Bus, Generator, Load, Shunt, Transformer
from pyopf.util.Log import Log


def parse_generators(generators: dict,
                     offline_generators: dict,
                     generators_data: dict,
                     non_empty_buses: set,
                     base_mva: float) -> dict:
    for key, ele in generators_data.items():
        if ele.stat:
            generators[key] = Generator(gen_data=ele, base_mva=base_mva)
            non_empty_buses.add(generators[key].bus)
        else:
            offline_generators[key] = Generator(gen_data=ele, base_mva=base_mva)

    return generators, offline_generators


def parse_loads(loads: dict,
                load_data: dict,
                non_empty_buses: set,
                base_mva: float) -> dict:
    for key, ele in load_data.items():
        if ele.status:
            loads[key] = Load(load_data=ele, base_mva=base_mva)
            non_empty_buses.add(loads[key].bus)
        else:
            continue
    return loads


def parse_branches(branches: dict,
                   branches_data: dict,
                   shunts: dict,
                   non_empty_buses: set,
                   base_mva: float) -> tuple[dict, dict]:
    for key, ele in branches_data.items():
        if ele.st:
            branches[key] = Branch(branch_data=ele, base_mva=base_mva)
            if ele.gi != 0 or ele.bi != 0:
                shunt_i = Shunt(shunt_data=ele, id='1', status=1, g_init=ele.gi * base_mva,
                                b_init=ele.bi * base_mva, base_mva=base_mva)
                shunts[(ele.i, '1')] = shunt_i

                non_empty_buses.add(shunt_i.bus)

            if ele.gj != 0 or ele.bj != 0:
                shunt_j = Shunt(shunt_data=ele, id='1', status=1, g_init=ele.gj * base_mva,
                                b_init=ele.bj * base_mva, base_mva=base_mva)
                shunts[(ele.j, '1')] = shunt_j
                non_empty_buses.add(shunt_j.bus)
    return branches, shunts


def parse_fixed_shunts(shunts: dict,
                       shunts_data: dict,
                       non_empty_buses: set,
                       base_mva: float) -> dict:
    for key, ele in shunts_data.items():
        if ele.status:
            shunts[key] = Shunt(shunt_data=ele, id=ele.id, status=ele.status, g_init=ele.gl, b_init=ele.bl,
                                base_mva=base_mva)

            if shunts[key].status and (shunts[key].g != 0 or shunts[key].b != 0):
                non_empty_buses.add(shunts[key].bus)
    return shunts


def parse_switched_shunts(shunts: dict,
                          switched_shunts_data: dict,
                          non_empty_buses: set,
                          base_mva: float) -> dict:
    if switched_shunts_data:
        for key, ele in switched_shunts_data.items():
            if ele.modsw != 0:
                warnings.warn(f"Found adjustable switched shunt {ele.i}. Adjustable switched shunts not supported. "
                              f"Treating as fixed shunt.",
                              RuntimeWarning)
            else:
                shunts[(ele.i, '1')] = Shunt(shunt_data=ele, id='1', status=ele.stat, g_init=0, b_init=ele.binit,
                                             base_mva=base_mva)
                if shunts[(ele.i, '1')].status and (shunts[(ele.i, '1')].g != 0 or shunts[(ele.i, '1')].b != 0):
                    non_empty_buses.add(shunts[(ele.i, '1')].bus)
    return shunts


def parse_transformers(transformers: dict,
                       transformers_data: dict,
                       buses: dict,
                       base_mva: float) -> dict:
    for key, ele in transformers_data.items():
        if ele.stat:
            kv1 = buses[ele.i].base_KV
            kv2 = buses[ele.j].base_KV
            transformers[key] = Transformer(transformer_data=ele, kv1=kv1, kv2=kv2, sbase=base_mva)
    return transformers


def parse(case: str,
          filepaths: dict,
          logger: Log,
          base_mva: float = 100):
    data = Data()
    data.raw.read(filepaths["raw"])

    generators = {}
    offline_generators = {}
    loads = {}
    buses = {}
    branches = {}
    shunts = {}
    transformers = {}
    non_empty_buses = set()

    # # == BUSES == # #
    for key, ele in data.raw.buses.items():
        buses[key] = Bus(bus_data=ele)

    # # == SLACKS == # #
    slacks = {key: ele for key, ele in buses.items() if ele.type == 3}

    # # == GENERATORS == # #
    generators, offline_generators = parse_generators(generators, offline_generators, data.raw.generators,
                                                      non_empty_buses, base_mva)

    # # == LOADS == # #
    loads = parse_loads(loads, data.raw.loads, non_empty_buses, base_mva)

    # # == BRANCHES == # #
    branches, shunts = parse_branches(branches, data.raw.nontransformer_branches, shunts, non_empty_buses, base_mva)

    # # == FIXED SHUNTS == # #
    shunts = parse_fixed_shunts(shunts, data.raw.fixed_shunts, non_empty_buses, base_mva)

    # # == SWITCHED SHUNTS == # #
    shunts = parse_switched_shunts(shunts, data.raw.switched_shunts, non_empty_buses, base_mva)

    # # == TRANSFORMERS == # #
    transformers = parse_transformers(transformers, data.raw.transformers, buses, base_mva)

    # # == STORE GRID DATA == # #
    grid_data = {
        "buses": buses,
        "slacks": slacks,
        "generators": generators,
        "offline generators": offline_generators,
        "transformers": transformers,
        "branches": branches,
        "shunts": shunts,
        "loads": loads,
    }
    case_data_raw = data
    logger.info(f"Finished parsing grid data for {case}.")

    return grid_data, case_data_raw
