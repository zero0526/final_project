import json
import networkx as nx
import matplotlib.pyplot as plt

# ====== Đọc file JSON ======
if __name__=="__main__":
    path= "D:\\source_code\\final_project\\data\\atlanta_nodes_config.json"
    with open(path, "r") as f:
        data = json.load(f)

    nodes = data["nodes_data"]
    links = data["links_data"]

    # ====== Tạo graph ======
    G = nx.Graph()

    # Thêm node
    pos = {}  # lưu tọa độ để vẽ
    node_colors = []

    for node in nodes:
        node_id = node["id"]
        x = float(node["coordinates"]["x"])
        y = float(node["coordinates"]["y"])
        node_type = node["type"]

        G.add_node(node_id, type=node_type)

        pos[node_id] = (x, y)

        # Màu theo loại node
        if node_type == "edge":
            node_colors.append("blue")
        elif node_type == "relay":
            node_colors.append("green")
        elif node_type == "cloud":
            node_colors.append("red")
        else:
            node_colors.append("gray")

    # Thêm cạnh (links)
    for link in links:
        src = link["source"]
        dst = link["target"]
        rate = link["transmission_rate"]

        G.add_edge(src, dst, weight=rate)

    # ====== Vẽ graph ======
    plt.figure(figsize=(8, 6))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        font_size=10
    )

    # Hiển thị transmission rate trên cạnh
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Network Topology Visualization")
    plt.show()