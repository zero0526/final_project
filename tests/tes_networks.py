from src.envs.network.topology_manager import TopologyManager
from src.envs.network.channel_model import ChannelModel


if __name__ == "__main__":
    topology_manager = TopologyManager()
    channel_manager = ChannelModel(topology_manager)
    print(channel_manager.get_metadata("N15", "N3", 10))
    print([n for n in topology_manager.get_neighbor_nodes_by_type("N15", 3, ["edge", "network"])])
    print(topology_manager.get_nodes_by_type("network"))
    print(topology_manager.get_nodes_by_type("edge"))
