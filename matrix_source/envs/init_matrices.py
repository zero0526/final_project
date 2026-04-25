import torch
import numpy as np

import torch
import numpy as np
import networkx as nx

def init_static_matrices(topology_data, terminals, config):
    """
    NHIỆM VỤ CỦA THÀNH PHẦN (INITIALIZER):
    1. Lọc các nút có tài nguyên tính toán (edge, network, cloud).
    2. Chuyển đổi dữ liệu topology sang các ma trận PyTorch cố định.
    """
    nodes_data = topology_data['nodes_data']
    links_data = topology_data['links_data']
    
    # Chỉ lấy các node có tài nguyên tính toán
    computing_node_types = ["edge", "network", "cloud"]
    computing_nodes = [node for node in nodes_data if node.get('type') in computing_node_types]
    
    comp_node_id_to_idx = {node['id']: i for i, node in enumerate(computing_nodes)}
    all_node_id_to_idx = {node['id']: i for i, node in enumerate(nodes_data)}
    
    num_comp_nodes = len(computing_nodes)
    num_all_nodes = len(nodes_data)
    
    # 2. Resource Matrix (Num_Computing_Nodes x 3) [CPU, RAM, HDD]
    resource_list = []
    for node in computing_nodes:
        specs = node.get('specs', {})
        resource_list.append([specs.get('cpu', 0), specs.get('ram', 0), specs.get('hdd', 0)])
    resource_matrix = torch.tensor(resource_list, dtype=torch.float32)
    
    # 3. Delay Matrix (Num_Computing_Nodes x Num_Computing_Nodes)
    G = nx.Graph()
    # Add all nodes to graph for pathfinding (including relay nodes)
    for node in nodes_data:
        G.add_node(node['id'])
    for link in links_data:
        rate = link.get('transmission_rate', 1.0)
        G.add_edge(link['source'], link['target'], weight=1.0/rate)
    
    # Chỉ tính toán delay giữa các computing nodes cho việc offloading
    delay_matrix = torch.zeros((num_comp_nodes, num_comp_nodes))
    for src_id in comp_node_id_to_idx:
        for dst_id in comp_node_id_to_idx:
            try:
                dist = nx.shortest_path_length(G, source=src_id, target=dst_id, weight='weight')
                delay_matrix[comp_node_id_to_idx[src_id], comp_node_id_to_idx[dst_id]] = dist
            except nx.NetworkXNoPath:
                delay_matrix[comp_node_id_to_idx[src_id], comp_node_id_to_idx[dst_id]] = float('inf')
            
    # 4. Terminal to Computing Node Mapping Matrix (Num_Terminals x Num_Computing_Nodes)
    num_terminals = len(terminals)
    terminal_to_comp_node_map = torch.zeros((num_terminals, num_comp_nodes))
    
    for i, terminal in enumerate(terminals):
        # UE thường kết nối trực tiếp với một Edge Node (là một loại computing node)
        if hasattr(terminal, 'edge_id') and terminal.edge_id in comp_node_id_to_idx:
            terminal_to_comp_node_map[i, comp_node_id_to_idx[terminal.edge_id]] = 1
            
    # 5. Max Queue Delay Matrix (Num_Computing_Nodes x Num_Services)
    # Load from delay.yaml (passed as part of config or separately)
    max_queue_delay = torch.zeros((num_comp_nodes, len(config.get('service_ids', [0,1,2,3,4]))))
    delay_data = config.get('delay_config', {}) # Dữ liệu từ delay.yaml
    
    for node_id, delays in delay_data.get('nodes', {}).items():
        if node_id in comp_node_id_to_idx:
            # Giả định delays là list có độ dài bằng num_services
            max_queue_delay[comp_node_id_to_idx[node_id]] = torch.tensor(delays).float()

    return {
        "comp_node_id_to_idx": comp_node_id_to_idx,
        "resource_matrix": resource_matrix,
        "delay_matrix": delay_matrix,
        "terminal_to_comp_node_map": terminal_to_comp_node_map,
        "max_queue_delay": max_queue_delay
    }

def init_metadata_tensors(services_dict):
    """
    NHIỆM VỤ:
    Khởi tạo các tensor chứa thuộc tính của Service và Model để tính toán song song.
    
    Returns:
    - model_workloads (S x Max_Models): Khối lượng tính toán (GFLOPS).
    - model_accuracies (S x Max_Models): Độ chính xác (%).
    - service_deadlines (S x 2): [mean_deadline, std_deadline].
    - service_omega (S x 1): Loại service (1: Continuous, 0: Occasional).
    - service_input_size (S x 1): Kích thước dữ liệu đầu vào (MB).
    """
    service_items = sorted(services_dict.items(), key=lambda x: x[1]['id'])
    num_services = len(service_items)
    
    # Tìm số lượng model lớn nhất trong một service
    max_models = max([len(svc.get('models', [])) for k, svc in service_items])
    
    model_workloads = torch.zeros((num_services, max_models))
    model_accuracies = torch.zeros((num_services, max_models))
    service_deadlines = torch.zeros((num_services, 5)) # [discretized deadlines]
    service_omega = torch.zeros((num_services, 1))
    service_input_size = torch.zeros((num_services, 1))
    service_resource_specs = torch.zeros((num_services, 2)) # [RAM, HDD] (GB)
    
    # Zipf calculation (using config param if available, else 0.8)
    zipf_param = config.get('zipf_param', 0.8)
    ranks = torch.arange(1, num_services + 1, dtype=torch.float32)
    zipf_weights = 1.0 / torch.pow(ranks, zipf_param)
    zipf_probs = zipf_weights / zipf_weights.sum()
    
    for i, (name, svc) in enumerate(service_items):
        service_omega[i] = svc.get('omega', 1)
        service_input_size[i] = svc.get('input_data_size', 0.0)
        service_resource_specs[i, 0] = svc.get('size', 0.0) / 1024.0 # RAM in GB
        service_resource_specs[i, 1] = svc.get('size', 0.0) / 1024.0 # HDD in GB (logic varies by omega)
        
        mean_dl, std_dl = svc.get('mean_deadline', 1.0), svc.get('std_deadline', 0.5)
        service_deadlines[i, :] = torch.linspace(mean_dl - 1.5*std_dl, mean_dl + 1.5*std_dl, 5)
        
        models = svc.get('models', [])
        for j, model in enumerate(models):
            model_workloads[i, j] = model.get('workload', 0.0)
            model_accuracies[i, j] = model.get('accuracy', 0.0)
            
    return {
        "model_workloads": model_workloads,
        "model_accuracies": model_accuracies,
        "service_deadlines": service_deadlines,
        "service_omega": service_omega,
        "service_input_size": service_input_size,
        "service_resource_specs": service_resource_specs,
        "zipf_probs": zipf_probs,
        "max_models": max_models
    }
