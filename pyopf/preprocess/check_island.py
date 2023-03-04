from __future__ import print_function

import math

import networkx as nx

from pyopf.util.Log import Log

__all__ = ['check_island']


def check_island(grid_data: dict,
                 non_empty_buses: set,
                 logger: Log):
    branch_edges = []
    branch_edge_on = []
    transformer_edge = []
    transformer_edge_on = []
    Graph = nx.MultiGraph()

    # # # ADD ALL EDGES # # #

    # Branch edges
    for ele in grid_data["branches"].values():
        branch_edges.append((ele.from_bus, ele.to_bus))
        if ele.status:
            branch_edge_on.append((ele.from_bus, ele.to_bus))

    # Transformer edges
    for ele in grid_data["transformers"].values():
        transformer_edge.append((ele.from_bus, ele.to_bus))
        if ele.status:
            transformer_edge_on.append((ele.from_bus, ele.to_bus))

    pairs = branch_edges + transformer_edge
    pairs_on = branch_edge_on + transformer_edge_on
    Graph.add_edges_from(pairs_on)
    graph = nx.from_edgelist(pairs)
    islands = list(nx.connected_components(graph))

    # First collect list of buses
    buses = set([ele.bus for ele in grid_data["buses"].values()])

    # Now collect all the buses in islands
    island_buses = set()
    for island in islands:
        island_buses |= island

    # get unique buses
    unique_buses = buses.difference(island_buses)

    # add unique buses to islands
    for ele in unique_buses:
        islands.append({ele})

    # Add weight to edges
    for ele in grid_data["branches"].values():
        graph[ele.from_bus][ele.to_bus]['weight'] = math.sqrt(ele.r ** 2 + ele.x ** 2)

    for ele in grid_data["transformers"].values():
        graph[ele.from_bus][ele.to_bus]['weight'] = math.sqrt(ele.r ** 2 + ele.x ** 2)

    # Check the degree of bus in non_empty_buses
    # This degree function does not exist if your island has only bus
    empty_buses = set()
    for ele in grid_data["buses"].values():
        if ele.bus not in non_empty_buses:
            empty_buses.add(ele.bus)
            # As degree function does not exist for single bus island
            if ele.bus not in unique_buses:
                if graph.degree(ele.bus) > 1:
                    non_empty_buses.add(ele.bus)

    bus_map = dict()
    multi_bus_map = dict()
    for ele in grid_data["buses"].values():
        if ele.bus not in unique_buses:
            bus_map[ele.bus] = graph.degree(ele.bus)
            multi_bus_map[ele.bus] = Graph.degree(ele.bus)
        else:
            bus_map[ele.bus] = 0
            multi_bus_map[ele.bus] = Graph.degree(ele.bus)

    dangling_buses = set()
    dangling_bus_found = True
    logger.info(f"Total number of empty buses in the system are {len(empty_buses)}")

    # # # IDENTIFY DANGLING BUSES # # #
    while dangling_bus_found:
        dangling_bus_found = False
        for bus in empty_buses:
            if bus_map[bus] == 1:
                dangling_buses.add(bus)
                dangling_bus_found = True
        # Reduce the bus set to remove already found dangling buses
        empty_buses -= dangling_buses
        for bus in dangling_buses:
            bus_map[bus] -= 1

    logger.info(f"Total of {len(dangling_buses)} dangling buses found in the system")

    bus_island = dict()
    # Attach an island number to slack and island number to buses
    for (idx, island) in enumerate(islands):
        for bus in island:
            bus_island[bus] = idx

    network = {"islands": islands,
               "non_empty_buses": non_empty_buses,
               "dangling buses": dangling_buses,
               "bus_degree_map": bus_map,
               "multi_bus_degree_map": multi_bus_map}

    return network, grid_data
