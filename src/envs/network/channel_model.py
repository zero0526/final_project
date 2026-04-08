from src.utils import cfg
from .topology_manager import TopologyManager


class ChannelModel:
    def __init__(self, config=None):
        self.config = config or {}
        self.topo: TopologyManager = TopologyManager()
        self.topo.load_topology_from_data()

    def compute_path_delay(
            self,
            from_node_id: str,
            to_node_id: str,
            data_size_mb: float
    ):
        if from_node_id == to_node_id:
            return 0.0

        path = self.topo.get_shortest_path(from_node_id, to_node_id)

        if not path:
            return float('inf')

        hops = len(path) - 1

        if hops == 0:
            return float('inf')

        rates = []
        for i in range(hops):
            rate = self.topo.get_link_transmission_rate(path[i], path[i + 1])
            if rate <= 0:
                return float('inf')
            rates.append(rate)

        bottleneck_rate = min(rates)

        return (data_size_mb * hops) / bottleneck_rate

    def estimate_transmission_energy(self, total_delay, power_coeff=None):
        p_coeff = power_coeff if power_coeff is not None else cfg.energy.get('tranmission_coef', 0.2)
        return p_coeff * total_delay

    def get_metadata(self, from_node_id: str, to_node_id: str, data_size_mb: float):
        tranmission_delay = self.compute_path_delay(from_node_id, to_node_id, data_size_mb)
        return {
            "tranmission_delay": tranmission_delay,
            "transmission_energy": self.estimate_transmission_energy(tranmission_delay)
        }