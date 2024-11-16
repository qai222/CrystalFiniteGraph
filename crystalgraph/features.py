import itertools
from collections import Counter

import networkx as nx
import numpy as np
from pydantic import BaseModel

from .qgraph import LQG, _allowed_voltages, multigraph_cycles

_allowed_cycle_voltage_sums = tuple(itertools.product(range(-2, 3), repeat=3))


class LqgFeatureSet(BaseModel):
    """
    calculated features for LQG
    REFs:
    - https://github.com/louzounlab/graph-measures
    - https://networkx.org/documentation/stable/reference/algorithms/index.html

    # TODO hypergraph?
    """

    num_node: int
    """ number of nodes """

    num_edge: int
    """ number of edges """

    edge_degree_count: list[int]
    """ number of nodes for a particular degree, degree_count[3] returns the number of nodes with degree=3
    fixed length """

    neighbor_degree_count: list[int]
    """ number of nodes for a particular neighbor number, 
    neighbor_count[3] returns the number of nodes with 3 neighbors
    fixed length, this is different from degree due to parallel edge (multigraph) """

    labelled_edge_count: list[int]
    """ number of edges classified by labels, 
    labelled_edge_count[3] returns the number of edges with label=_allowed_voltages[3]
    fixed length """

    num_cycle_basis: int
    """ number of basis cycles """

    cycle_sum_count: list[int]
    """ number of cycles classified by voltage sums, 
    cycle_sum_count[3] returns the number of cycles with voltage sum=_allowed_voltage_sums[3]
    fixed length """

    spectral_radius: float
    """ The largest eigenvalue absolute value in a graph is called the spectral radius of the graph """

    algebraic_connectivity: float
    """ The second smallest eigenvalue of the Laplacian matrix of a graph is called its algebraic connectivity """

    graph_energy: float
    """ The sum of absolute values of graph eigenvalues is called the graph energy. """

    # average_clustering_coefficient: float
    # """ using networkx implementation of 10.5445/IR/1000001239 """

    maximum_eccentricity: int
    """ 2-sweep in nx.diameter """

    len_dominating_set: int
    """ number of nodes in the dominating set """

    degree_centrality_min: float
    degree_centrality_max: float
    degree_centrality_mean: float
    degree_centrality_std: float
    """ min, max, and mean of node degree centrality """

    # average_degree_connectivity: list[int]
    # """ average_degree_connectivity[3] is the average degree connectivity of degree 3 """

    num_bridge: int

    eigenvector_centrality_mean: float
    eigenvector_centrality_std: float
    """ note this is the scipy implementation, nx doesn't work for multigraph """

    square_clustering_min: float
    square_clustering_max: float
    square_clustering_mean: float
    square_clustering_std: float
    """ min, max, and mean of squares clustering coefficient for nodes """

    average_shortest_path_length: float

    barycenter_size: int

    louvain_communities_num: int
    louvain_communities_sizes_mean: float
    louvain_communities_sizes_std: float
    louvain_communities_sizes_min: float
    louvain_communities_sizes_max: float

    local_efficiency: float
    global_efficiency: float

    is_eulerian: bool
    is_planar: bool

    constraints_mean: float
    constraints_std: float
    constraints_min: float
    constraints_max: float

    # effective_sizes_mean: float
    # effective_sizes_std: float
    # effective_sizes_min: float
    # effective_sizes_max: float

    wiener_index: float

    betweenness_centrality_mean: float
    betweenness_centrality_std: float
    betweenness_centrality_min: float
    betweenness_centrality_max: float

    is_neighbor_regular: bool
    is_node_regular: bool

    def as_dict(self) -> dict:
        d = dict()
        data = self.model_dump()
        for key in sorted(self.model_fields_set):
            val = data[key]
            if isinstance(val, list):
                for i, list_item in enumerate(val):
                    k = f"{key}_{i}"
                    d[k] = list_item
            else:
                d[key] = val
        return d

    @classmethod
    def from_lqg(cls, lqg: LQG):
        return calculate_features(lqg)


def calculate_node_degree_count(g: nx.MultiGraph):
    degree_seq = (d for n, d in g.degree)
    degree_counter = Counter(degree_seq)
    return [degree_counter[deg] for deg in (0, 1, 2, 3, 4)]


def calculate_neighbor_degree_count(g: nx.MultiGraph):
    neighbor_seq = [len([*g.neighbors(n)]) for n in g.nodes]
    degree_counter = Counter(neighbor_seq)
    return [degree_counter[deg] for deg in (0, 1, 2, 3, 4)]


def calculate_labelled_edge_count(g: nx.MultiGraph):
    labels = [d['voltage'] for u, v, k, d in g.edges(keys=True, data=True)]
    counter = Counter(labels)
    return [counter[lab] for lab in _allowed_voltages]


def calculate_cycle_sum_count(lqg: LQG):
    el_to_cyc = multigraph_cycles(lqg.nxg)
    sums = []
    for el, cyc in el_to_cyc.items():
        voltage_sum = lqg.voltage_sum_cycle(el, cyc)
        sums.append(voltage_sum)
    return [Counter(sums)[s] for s in _allowed_cycle_voltage_sums]


def calculate_is_node_regular(g: nx.MultiGraph):
    nonzeros = 0
    for d in calculate_node_degree_count(g):
        if d:
            nonzeros += 1
    return nonzeros == 1


def calculate_is_neighbor_regular(g: nx.MultiGraph):
    nonzeros = 0
    for d in calculate_neighbor_degree_count(g):
        if d:
            nonzeros += 1
    return nonzeros == 1


def calculate_features(lqg: LQG) -> LqgFeatureSet:
    g = lqg.nxg
    eigens = nx.adjacency_spectrum(g)
    eigens_abs = sorted([abs(e) for e in eigens])
    degree_centrality = [*nx.degree_centrality(g).values()]
    eigen_vector_centrality = [*nx.eigenvector_centrality_numpy(g).values()]
    square_clustering = [*nx.square_clustering(g).values()]
    louvain_communities = nx.community.louvain_communities(g)
    louvain_communities_sizes = [len(c) for c in louvain_communities]
    constraints = [*nx.constraint(g).values()]
    # effective_sizes = [*nx.effective_size(g).values()]  # can work but too slow
    betweenness_centrality = [*nx.betweenness_centrality(g).values()]
    fs = LqgFeatureSet(
        num_node=g.number_of_nodes(),
        num_edge=g.number_of_edges(),
        edge_degree_count=calculate_node_degree_count(g),
        neighbor_degree_count=calculate_neighbor_degree_count(g),
        labelled_edge_count=calculate_labelled_edge_count(g),
        num_cycle_basis=len(multigraph_cycles(g)),
        cycle_sum_count=calculate_cycle_sum_count(lqg),
        spectral_radius=max(eigens_abs),
        algebraic_connectivity=sorted(nx.laplacian_spectrum(g))[1],
        graph_energy=sum(eigens_abs),
        # average_clustering_coefficient=,
        maximum_eccentricity=nx.diameter(g),
        len_dominating_set=len(nx.dominating_set(g)),
        degree_centrality_min=min(degree_centrality),
        degree_centrality_max=max(degree_centrality),
        degree_centrality_mean=np.mean(degree_centrality),
        degree_centrality_std=np.std(degree_centrality),
        # average_degree_connectivity=[nx.average_degree_connectivity(g)],
        num_bridge=len([*nx.bridges(g)]),
        eigenvector_centrality_mean=np.mean(eigen_vector_centrality),
        eigenvector_centrality_std=np.std(eigen_vector_centrality),
        square_clustering_min=min(square_clustering),
        square_clustering_max=max(square_clustering),
        square_clustering_mean=np.mean(square_clustering),
        square_clustering_std=np.std(square_clustering),
        average_shortest_path_length=nx.average_shortest_path_length(g),
        barycenter_size=len(nx.barycenter(g)),
        louvain_communities_num=len(louvain_communities),
        louvain_communities_sizes_mean=np.mean(louvain_communities_sizes),
        louvain_communities_sizes_std=np.std(louvain_communities_sizes),
        louvain_communities_sizes_min=min(louvain_communities_sizes),
        louvain_communities_sizes_max=max(louvain_communities_sizes),
        local_efficiency=nx.local_efficiency(g),
        global_efficiency=nx.global_efficiency(g),
        is_eulerian=nx.is_eulerian(g),
        is_planar=nx.is_planar(g),
        constraints_mean=np.mean(constraints),
        constraints_std=np.std(constraints),
        constraints_min=min(constraints),
        constraints_max=max(constraints),
        # effective_sizes_mean=np.mean(effective_sizes),
        # effective_sizes_std=np.std(effective_sizes),
        # effective_sizes_min=min(effective_sizes),
        # effective_sizes_max=max(effective_sizes),
        wiener_index=nx.wiener_index(g),
        betweenness_centrality_mean=np.mean(betweenness_centrality),
        betweenness_centrality_std=np.std(betweenness_centrality),
        betweenness_centrality_min=min(betweenness_centrality),
        betweenness_centrality_max=max(betweenness_centrality),
        is_neighbor_regular=calculate_is_neighbor_regular(g),
        is_node_regular=calculate_is_node_regular(g),
    )
    return fs
