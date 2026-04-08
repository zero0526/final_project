from src.configs.configs import cfg
from .topology_manager import TopologyManager


class ChannelModel:
    def __init__(self, topo: TopologyManager, config=cfg):
        self.config = config or {}
        self.topo = topo
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
            return 0.0

        rates = []
        for i in range(hops):
            rate = self.topo.get_link_transmission_rate(path[i], path[i + 1])
            if rate <= 0:
                return float('inf')
            rates.append(rate)

        bottleneck_rate = min(rates)

        return (data_size_mb * hops) / bottleneck_rate

    @staticmethod
    def estimate_transmission_energy(total_delay, power_coeff=None):
        p_coeff = power_coeff if power_coeff is not None else getattr(cfg, 'transmission_coef', 0.2)
        return p_coeff * total_delay

    def get_metadata(self, from_node_id: str, to_node_id: str, data_size_mb: float):
        transmission_delay = self.compute_path_delay(from_node_id, to_node_id, data_size_mb)
        return {
            "transmission_delay": transmission_delay,
            "transmission_energy": self.estimate_transmission_energy(transmission_delay)
        }