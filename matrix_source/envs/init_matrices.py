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
    
    num_comp_nodes = len(computing_nodes)
    
    # 2. Resource Matrix (Num_Computing_Nodes x 3) [CPU, RAM, HDD]
    resource_list = []
    for node in computing_nodes:
        specs = node.get('specs', {})
        resource_list.append([specs.get('cpu', 0), specs.get('ram', 0), specs.get('hdd', 0)])
    resource_matrix = torch.tensor(resource_list, dtype=torch.float32)
    
    # 3. Delay Matrix (Num_Computing_Nodes x Num_Computing_Nodes)
    G = nx.Graph()
    for node in nodes_data:
        G.add_node(node['id'])
    for link in links_data:
        rate = link.get('transmission_rate', 250.0) # Default rate if not provided
        # We store both hop weight (1) and the actual rate
        G.add_edge(link['source'], link['target'], weight=1.0, rate=rate)
    
    delay_matrix = torch.zeros((num_comp_nodes, num_comp_nodes))
    for src_id, i in comp_node_id_to_idx.items():
        for dst_id, j in comp_node_id_to_idx.items():
            if src_id == dst_id:
                delay_matrix[i, j] = 0.0
                continue
            try:
                # Tìm đường đi ngắn nhất theo số bước nhảy (hop count)
                path = nx.shortest_path(G, source=src_id, target=dst_id, weight='weight')
                
                # Tính trung bình transmission rate trên toàn tuyến
                rates = []
                for k in range(len(path) - 1):
                    rates.append(G[path[k]][path[k+1]]['rate'])
                
                avg_rate = sum(rates) / len(rates) if rates else 1e9
                num_hosts = len(path)
                
                # Công thức: số host * (1 / tốc độ trung bình)
                delay_matrix[i, j] = num_hosts * (1.0 / avg_rate)
            except nx.NetworkXNoPath:
                delay_matrix[i, j] = float('inf')
            
    # 4. Terminal to Computing Node Mapping Matrix (Num_Terminals x Num_Computing_Nodes)
    num_terminals = len(terminals)
    terminal_to_comp_node_map = torch.zeros((num_terminals, num_comp_nodes))
    
    for k, terminal in enumerate(terminals):
        if hasattr(terminal, 'edge_id') and terminal.edge_id in comp_node_id_to_idx:
            terminal_to_comp_node_map[k, comp_node_id_to_idx[terminal.edge_id]] = 1
            
    # 5. Max Queue Delay Matrix (Num_Computing_Nodes x Num_Services)
    max_queue_delay = torch.zeros((num_comp_nodes, len(config.get('service_ids', [0,1,2,3,4]))))
    delay_data = config.get('delay_config', {})
    
    for node_id, delays in delay_data.get('nodes', {}).items():
        if node_id in comp_node_id_to_idx:
            max_queue_delay[comp_node_id_to_idx[node_id]] = torch.tensor(delays).float()

    return {
        "comp_node_id_to_idx": comp_node_id_to_idx,
        "resource_matrix": resource_matrix,
        "transmission_delay_matrix": delay_matrix,
        "terminal_to_comp_node_map": terminal_to_comp_node_map,
        "max_queue_delay": max_queue_delay
    }

def init_metadata_tensors(services_dict, config):
    """
    NHIỆM VỤ: Khởi tạo các tensor chứa thuộc tính của Service và Model.
    """
    service_items = sorted(services_dict.items(), key=lambda x: x[1]['id'])
    num_services = len(service_items)
    
    max_models = max([len(svc.get('models', [])) for k, svc in service_items])
    
    model_workloads = torch.zeros((num_services, max_models))
    model_accuracies = torch.zeros((num_services, max_models))
    service_deadlines = torch.zeros((num_services, 3)) 
    service_omega = torch.zeros((num_services, 1))
    service_input_size = torch.zeros((num_services, 1))
    service_size = torch.zeros((num_services, 1))
    
    zipf_param = config.get('zipf_param', 0.8)
    ranks = torch.arange(1, num_services + 1, dtype=torch.float32)
    zipf_weights = 1.0 / torch.pow(ranks, zipf_param)
    zipf_probs = zipf_weights / zipf_weights.sum()
    
    for i, (name, svc) in enumerate(service_items):
        service_omega[i] = svc.get('omega', 1)
        service_input_size[i] = svc.get('input_data_size', 0.0)
        service_size[i] = svc.get('size', 0.0)
        
        mean_dl = svc.get('mean_deadline', 1.0)
        service_deadlines[i, 2] = mean_dl
        
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
        "service_size": service_size,
        "zipf_probs": zipf_probs,
        "max_models": max_models
    }
