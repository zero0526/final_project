import networkx as nx
from typing import List, Dict, Tuple
import json
from src.configs.configs import cfg


class TopologyManager:
    def __init__(self):
        self.graph = nx.Graph()
        self._path_cache = {}
        self.network_stats = {}

    def load_topology_from_data(self):
        self.graph.clear()
        self._path_cache = {}
        topo_key = f"topology_{cfg.topology_name.lower()}_json"
        with open(cfg.sim_paths[topo_key], 'r') as f:
            data = json.load(f)
        self.network_stats = data.get('stats', {})

        # 2. Parse Nodes
        for node in data['nodes']:
            self.graph.add_node(
                node['id'],
                type=node.get('type', 'relay'),  # edge, cloud, network, relay

                cpu_available=node.get('cpu', 0.0),
                ram_capacity=node.get('ram', 0.0),
                hdd_capacity=node.get('hdd', 0.0),

                pos=(node['coordinates']['x'], node['coordinates']['y']),
                energy_coef=float(node.get('energy_coef', 0.0))
            )

        # 3. Parse Links
        for link in data['links']:
            self.graph.add_edge(
                link['source'],
                link['target'],
                id=link['id'],

                transmission_rate=link.get('tranmission_rate', 0.0),

                energy_coef=link.get('energy_coef', 0.2)
            )

        print(f"Loaded Topology: {data.get('network_name')} "
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

    def get_node_resources(self, node_id: str) -> Dict:
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        return {}

    def get_nodes_by_type(self, node_type='edge') -> List[str]:
        """
        Lấy ra danh sách các node có type là 'edge'
        """
        return [node_id for node_id, data in self.graph.nodes(data=True) if data.get('type') == node_type]

    def get_average_hops_to_node(self, target_node_id: str) -> float:
        """
        Ước tính số hop trung bình từ các node nguồn (edge/cloud khác) đến target_node_id.
        Xác suất gửi tin từ node nguồn giảm dần theo khoảng cách hop.
        """
        # Node nguồn tiềm năng: Edge hoặc Cloud nodes (trừ chính nó)
        source_node_ids = [nid for nid, d in self.graph.nodes(data=True)
                           if d.get('type') in ['edge', 'cloud'] and nid != target_node_id]

        if not source_node_ids:
            return 0.0

        import networkx as nx
        try:
            # Tính khoảng cách từ tất cả các nút đến target_node_id
            all_distances = nx.single_source_shortest_path_length(self.graph, target_node_id)
        except Exception:
            return 1.0

        weighted_hops_sum = 0.0
        total_probability_weight = 0.0

        for src_id in source_node_ids:
            dist = all_distances.get(src_id, 20)
            # Trọng số xác suất: tỉ lệ nghịch với dist+1
            weight = 1.0 / (dist + 1)

            weighted_hops_sum += dist * weight
            total_probability_weight += weight

        if total_probability_weight == 0:
            return 1.0

        return weighted_hops_sum / total_probability_weight

    def get_edge_nodes_by_depth(self, start_node: str, max_depth: int) -> List[str]:
        """
        Duyệt DFS để tìm các node 'edge' trong phạm vi độ sâu max_depth.
        """
        edge_nodes = []
        visited = {start_node}

        def dfs(u, current_depth):
            # Nếu không phải node xuất phát và là edge, thêm vào danh sách
            if u != start_node and self.graph.nodes[u].get('type') == 'edge':
                if u not in edge_nodes:
                    edge_nodes.append(u)

            if current_depth < max_depth:
                for v in self.graph.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        dfs(v, current_depth + 1)

        dfs(start_node, 0)
        return edge_nodes
