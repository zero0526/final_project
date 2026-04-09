import networkx as nx
from typing import List, Dict, Tuple
import json
from src.configs.configs import cfg


class TopologyManager:
    def __init__(self, data=cfg.topology_data):
        self.graph = nx.Graph()
        self._path_cache = {}
        self.network_stats = {}
        self.data = data

    def load_topology_from_data(self):
        self.graph.clear()
        self._path_cache = {}

        for node in self.data['nodes_data']:
            self.graph.add_node(
                node['id'],
                type=node.get('type', 'relay'),  # edge, cloud, network, relay
                pos=(node['coordinates']['x'], node['coordinates']['y']),
                energy_coef=float(node.get('energy_coef', 0.0))
            )

        # 3. Parse Links
        for link in self.data['links_data']:
            self.graph.add_edge(
                link['source'],
                link['target'],
                id=link['id'],
                transmission_rate=link.get('transmission_rate', 0.0),
                energy_coef=cfg.transmission_coef
            )

        print(f"Loaded Topology: {self.data.get('topology')} "
              f"({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} links)")

    def get_shortest_path(self, source: str, target: str) -> List[str]:
        if source == target:
            return [source]

        cache_key = (source, target)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        try:
            path = nx.shortest_path(self.graph, source=source, target=target, weight=None)
            self._path_cache[cache_key] = path
            return path
        except nx.NetworkXNoPath:
            return []

    def get_link_transmission_rate(self, u: str, v: str) -> float:
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['transmission_rate']
        return 0.0

    def get_nodes_by_type(self, node_type='edge') -> List[str]:
        """
        Lấy ra danh sách các node có type là 'edge'
        """
        return [node_id for node_id, data in self.graph.nodes(data=True) if data.get('type') == node_type]

    def get_neighbor_nodes_by_type(self, start_node: str, max_depth: int, node_types: List[str]) -> List[str]:
        """
        Duyệt DFS để tìm các node 'edge' trong phạm vi độ sâu max_depth.
        """
        edge_nodes = []
        visited = {start_node}

        def dfs(u, current_depth):
            # Nếu không phải node xuất phát và là edge, thêm vào danh sách
            if u != start_node and self.graph.nodes[u].get('type') in node_types:
                if u not in edge_nodes:
                    edge_nodes.append(u)

            if current_depth < max_depth:
                for v in self.graph.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        dfs(v, current_depth + 1)

        dfs(start_node, 0)
        return edge_nodes
