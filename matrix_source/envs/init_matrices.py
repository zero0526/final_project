import torch
import networkx as nx
from matrix_source.models.terminal import Terminal

def init_static_matrices(config, device="cpu"):
    """
    NHIỆM VỤ CỦA THÀNH PHẦN (INITIALIZER):
    1. Lọc các nút có tài nguyên tính toán (edge, network, cloud).
    2. Tự động khởi tạo Terminal và gán vào các Edge Node (Round Robin).
    3. Chuyển đổi dữ liệu topology sang các ma trận PyTorch cố định.
    """
    topology_data= config.topology_data
    nodes_data = topology_data['nodes_data']
    links_data = topology_data['links_data']
    
    # 1. Lọc các Computing Nodes
    computing_node_types = ["edge", "network", "cloud"]
    computing_nodes = [node for node in nodes_data if node.get('type') in computing_node_types]
    edge_nodes = [node for node in nodes_data if node.get('type') == 'edge']
    cloud_nodes = [node for node in nodes_data if node.get('type') == 'cloud']
    if not edge_nodes:
        # Fallback if no edge nodes found, use all computing nodes
        edge_nodes = computing_nodes
        
    comp_node_id_to_idx = {node['id']: i for i, node in enumerate(computing_nodes)}
    edge_ids= [comp_node_id_to_idx[node.get("id")] for node in edge_nodes]
    cloud_ids = [comp_node_id_to_idx[node.get("id")] for node in cloud_nodes]
    num_comp_nodes = len(computing_nodes)
    
    # 2. Khởi tạo Terminals (Round Robin assignment to Edge Nodes)
    # Lấy số lượng từ hyper_neural config
    num_terminals = config.hyper_neural.get("NUM_LOWER_AGENTS", 1)
    terminals = []
    for k in range(num_terminals):
        # Lấy edge node theo vòng tròn (Round Robin)
        target_edge = edge_nodes[k % len(edge_nodes)]
        terminals.append(Terminal(terminal_id=k, edge_id=target_edge['id']))
    
    # 3. Resource Matrix (Num_Computing_Nodes x 3) [CPU, RAM, HDD]
    resource_list = []
    for node in computing_nodes:
        specs = node.get('specs', {})
        resource_list.append([specs.get('cpu', 0), specs.get('ram', 0), specs.get('hdd', 0)])
    resource_matrix = torch.tensor(resource_list, dtype=torch.float32, device=device)
    
    # 4. Delay Matrix (Num_Computing_Nodes x Num_Computing_Nodes)
    G = nx.Graph()
    for node in nodes_data:
        G.add_node(node['id'])
    for link in links_data:
        rate = link.get('transmission_rate', 250.0) 
        G.add_edge(link['source'], link['target'], weight=1.0, rate=rate)
    
    delay_matrix = torch.zeros((num_comp_nodes, num_comp_nodes), device=device)
    hops = torch.zeros((num_comp_nodes, num_comp_nodes), device=device)
    for src_id, i in comp_node_id_to_idx.items():
        for dst_id, j in comp_node_id_to_idx.items():
            if src_id == dst_id:
                delay_matrix[i, j] = 0.0
                continue
            try:
                path = nx.shortest_path(G, source=src_id, target=dst_id, weight='weight')
                rates = [G[path[k]][path[k+1]]['rate'] for k in range(len(path) - 1)]
                avg_rate = sum(rates) / len(rates) if rates else 1e9
                num_hosts = max(len(path) - 1, 0)
                delay_matrix[i, j] = num_hosts * (1.0 / avg_rate)
                hops[i,j]= num_hosts
            except nx.NetworkXNoPath:
                delay_matrix[i, j] = float('inf')
            
    # 5. Terminal to Computing Node Mapping Matrix
    terminal_to_comp_node_map = torch.zeros((num_terminals, num_comp_nodes), device=device)
    for k, terminal in enumerate(terminals):
        if terminal.edge_id in comp_node_id_to_idx:
            terminal_to_comp_node_map[k, comp_node_id_to_idx[terminal.edge_id]] = 1
            
    # 6. Max Queue Delay Matrix
    max_queue_delay = torch.zeros((num_comp_nodes, len(config.services)), device=device)
    delay_data = config.delay_queue_max
    for node_id, delays in delay_data.items():
        if node_id in comp_node_id_to_idx:
            max_queue_delay[comp_node_id_to_idx[node_id]] = torch.tensor(delays).float()

    # 7. Adjacency Matrix (for Mean Field) - 2 hops limit
    adj_matrix = torch.zeros((num_comp_nodes, num_comp_nodes), device=device)
    for node_id, i in comp_node_id_to_idx.items():
        # Find nodes within 2 hops
        lengths = nx.single_source_shortest_path_length(G, node_id, cutoff=2)
        for target_id, dist in lengths.items():
            if target_id in comp_node_id_to_idx and target_id != node_id:
                j = comp_node_id_to_idx[target_id]
                adj_matrix[i, j] = 1.0

    edge_id_to_agent_idx = {eid: idx for idx, eid in enumerate(edge_ids)}
    agent_adj_matrix = torch.zeros((len(edge_ids), len(edge_ids)), device=device)
    for i, edge_node in enumerate(edge_nodes):
        # i is the agent index for this edge node
        node_id = edge_node["id"]
        lengths = nx.single_source_shortest_path_length(G, node_id, cutoff=2)
        for target_id, dist in lengths.items():
            if target_id in comp_node_id_to_idx and target_id != node_id:
                comp_target_id = comp_node_id_to_idx[target_id]
                if comp_target_id in edge_id_to_agent_idx:
                    j = edge_id_to_agent_idx[comp_target_id]
                    agent_adj_matrix[i, j] = 1.0

    # 8. Terminal Adjacency Matrix
    terminal_adj_matrix = torch.zeros((num_terminals, num_terminals), device=device)
    for i in range(num_terminals):
        for j in range(num_terminals):
            if i != j and terminals[i].edge_id == terminals[j].edge_id:
                terminal_adj_matrix[i, j] = 1.0

    return {
        "comp_node_id_to_idx": comp_node_id_to_idx,
        "resource_matrix": resource_matrix,
        "transmission_delay_matrix": delay_matrix,
        "terminal_to_comp_node_map": terminal_to_comp_node_map,
        "max_queue_delay": max_queue_delay,
        "adj_matrix": adj_matrix,
        "terminal_adj_matrix": terminal_adj_matrix,
        "terminals": terminals,
        "edge_ids": edge_ids,
        "cloud_ids": cloud_ids,
        "agent_adj_matrix": agent_adj_matrix,
        "edge_id_to_agent_idx": edge_id_to_agent_idx,
        "hops": hops,
    }

def init_metadata_tensors(config, device="cpu"):
    """
    NHIỆM VỤ: Khởi tạo các tensor chứa thuộc tính của Service và Model.
    """
    services_dict= config.services
    service_items = sorted(services_dict.items(), key=lambda x: x[1]['id'])
    num_services = len(service_items)
    
    max_models = max([len(svc.get('models', [])) for k, svc in service_items])
    
    model_workloads = torch.zeros((num_services, max_models), device=device)
    model_accuracies = torch.zeros((num_services, max_models), device=device)
    service_deadlines = torch.zeros((num_services, 3), device=device)
    service_omega = torch.zeros((num_services, 1), device=device)
    service_input_size = torch.zeros((num_services, 1), device=device)
    service_size = torch.zeros((num_services, 1), device=device)
    
    zipf_param = 2.5
        # config.zipf_param)
    ranks = torch.arange(1, num_services + 1, dtype=torch.float32, device=device)
    zipf_weights = 1.0 / torch.pow(ranks, zipf_param)
    zipf_probs = zipf_weights / zipf_weights.sum()
    
    for i, (name, svc) in enumerate(service_items):
        service_omega[i] = svc.get('omega', 1)
        service_input_size[i] = svc.get('input_data_size', 0.0)
        service_size[i] = svc.get('size', 0.0)
        
        min_dl = svc.get('mean_deadline', 1.0)
        max_dl = min_dl + svc.get('std_deadline', 1.0)
        
        service_deadlines[i] = torch.linspace(min_dl, max_dl, 3)
        
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
