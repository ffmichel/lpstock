import collections
import operator

import networkx as nx


def root(graph):
    root_nodes = set(x for x in graph.nodes()
                     if graph.out_degree(x) > 0 and graph.in_degree(x) == 0)
    assert len(root_nodes) == 1
    root = root_nodes.pop()
    return root


def leaf_nodes(graph):
    return [
        x for x in graph.nodes()
        if graph.out_degree(x) == 0 and graph.in_degree(x) == 1
    ]


def node_by_distance_to_root(graph):
    graph_root = root(graph)
    length_by_node = nx.shortest_path_length(graph, source=graph_root)
    node_by_length = collections.defaultdict(list)
    for node, length in length_by_node.items():
        node_by_length[length].append(node)
    return node_by_length


def reverse_tree_traversal(graph, node_property, node_func):
    node_by_length = node_by_distance_to_root(graph)
    for _, nodes in sorted(node_by_length.items(), key=operator.itemgetter(0),
                           reverse=True):
        for node in nodes:
            graph.nodes[node][node_property] = node_func(graph, node)


def tree_bfs(graph, root_value, node_property, node_func):
    graph_root = root(graph)
    graph.nodes[graph_root][node_property] = root_value
    for edge in nx.bfs_edges(graph, graph_root):
        _, child_name = edge
        graph.nodes[child_name][node_property] = node_func(graph, edge)
