import math
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque

from src.envs.entities.task_node import Task
from src.utils import convert_nodeid2order, one_hot
from src.configs.configs import cfg, BaseConfig
from src.envs.network.topology_manager import TopologyManager
from src.envs.network.channel_model import ChannelModel
from src.envs.entities.computing_node import ComputingNode
from src.envs.entities.terminal_node import Terminal
from src.envs.workload_generator import WorkloadGenerator
from src.envs.time_manager import TimeManager


class SixGEnvironment:
    def __init__(self, num_terminals: int, config:BaseConfig = cfg):
        self.config = config
        self.service_config = {s.get("id"):s for s in config.services.values()}
        self.num_services = len(self.service_config)
        self.device = cfg.device

        # setup network
        self.topo_manager = TopologyManager()
        self.topo_manager.load_topology_from_data()
        self.channel_model = ChannelModel(topo=self.topo_manager)

        self.nodes: Dict[str, ComputingNode] = {}
        self.agent_node_ids = []
        self.cloud_node_id = None
        self.computing_nodes: List[ComputingNode]= []
        self.node_id_dict= {}
        self.invert_node_id= {}
        self._init_nodes()

        self.terminals: Dict[str, Terminal] = {}
        self.terminals_group: Dict[str, List[Terminal]]= defaultdict(list)
        self._init_terminals(num_terminals)

        self.workload_gen = WorkloadGenerator(
            terminals=list(self.terminals.values())
        )

        self.time_manager = TimeManager()

        self.max_models_total = max([len(svc['models']) for svc in self.service_config.values()])

        self.T = cfg.hyper_neural.get('TIME_SLOT_PER_TIMEFRAME', 10)
        self.frame_F1_accumulation: deque[float]= deque(maxlen=self.T)

    def _init_nodes(self):
        for node in self.config.topology_data.get("nodes_data"):
            self.nodes[node.get("id")]= ComputingNode(
                node_id=node.get("id"),
                type_node=node.get("type"),
                specs= node.get("specs"),
                channel_model=self.channel_model
            )

            if node.get("type") in ["edge", "network"]:
                self.agent_node_ids.append(node.get("id"))
            if node.get("type") == "cloud":
                self.cloud_node_id= node.get("id")
            if node.get("type") in ["edge", "network", "cloud"]:
                local_id=len(self.node_id_dict)
                self.node_id_dict[node.get("id")]= local_id
                self.invert_node_id[local_id]= node.get("id")
                self.computing_nodes.append(self.nodes[node.get("id")])
        self.computing_nodes = sorted(self.computing_nodes, key=lambda x: convert_nodeid2order(x.id))
        # add neighborhoods
        for node_id, node  in self.nodes.items():
            neighbor_ids= self.topo_manager.get_neighbor_nodes_by_type(node_id, 2, ["edge", "network", "cloud"])
            self.nodes[node_id].neighbor_nodes= [self.nodes[i] for i in neighbor_ids]

    def _init_terminals(self, num_terminals):
        edge_ids = [idx for idx, n in self.nodes.items() if n.type == "edge"]
        n= len(edge_ids)
        for i in range(num_terminals):
            t_id = f"UE_{i}"
            source_id= edge_ids[i%n]
            self.terminals[t_id] = Terminal(
                terminal_id=t_id,
                edge_id= source_id,
                arrival_rate=self.config.task_arrival_rate,
                default_batch_size=self.config.default_batch_size
            )
            self.terminals_group[source_id].append(self.terminals[t_id])

    def reset_lower(self):
        next_states={}
        num_nodes = len(self.computing_nodes)
        for node in self.computing_nodes:
            node.upper_reset()
            node.lower_reset()
        self.nodes[self.cloud_node_id].update_placement(np.ones(self.num_services))
        next_observe_backlog, next_observe_cpu = self.collect_backlog_resources()
        # generate task
        next_tasks = self.workload_gen.step(self.time_manager.time_elapsed)
        for task in next_tasks:
            next_states[task.terminal_id]= task, next_observe_backlog[task.service_id], next_observe_cpu[task.service_id], np.zeros(num_nodes + self.max_models_total)
        zero_data= {nid:np.zeros(self.num_services) for nid in self.nodes.keys()}
        info = {
            "f1": 0,
            "energy": 0,
            "virtual_delay": zero_data,
            "success_qos": zero_data,
            "realized_delay": zero_data,
            "violate_qos": zero_data
        }
        return {
            "reward": 0.0,
            "next_states": next_states,
            "new_frame": self.time_manager.is_new_frame(),
            "info": info
        }

    def reset_upper(self):
        self.time_manager.reset()
        self.frame_F1_accumulation = deque(maxlen=self.T)
        for node in self.nodes.values():
            if node.type in ["edge", "network", "cloud"]:
                node.reset()
        self.nodes[self.cloud_node_id].update_placement(np.ones(self.num_services))
        self.workload_gen.step(abs_current_time=0)

        states = {}
        mean_fields = {}
        n= self.num_services
        for node in self.computing_nodes:
            states[node.id] = (np.zeros(n), np.zeros(n))
            mean_fields[node.id] = np.zeros(n)
        return {
            "reward": 0.0,
            "next_states": states,
            "mean_fields": mean_fields,
            "done": self.time_manager.is_done(),
            "remaining_task": self.compose_task_remaining()
        }

    def step_upper(self, actions: Dict[str, np.ndarray]):
        states={}
        mean_fields= {}
        reward= -sum(f for f in self.frame_F1_accumulation)
        for node in self.computing_nodes:
            states[node.id]= (node.placed_services, node.popularity_service)
            mean_fields[node.id]= node.mean_field
        self.frame_F1_accumulation = deque(maxlen=self.T)
        for nid in self.agent_node_ids:
            if nid in actions:
                self.nodes[nid].update_placement(actions[nid])

        return {
            "reward": reward*1e-10,
            "next_states": states,
            "mean_fields": mean_fields,
            "done": self.time_manager.is_done(),
            "remaining_task": self.compose_task_remaining()
        }


    def collect_backlog_resources(self):
        """
        service_id start from 0->n-1 when using index from config service id use minus 1
        """
        n, m= len(self.computing_nodes), self.num_services
        observe_backlog= np.zeros((m,n), dtype=np.float32)
        observe_cpu = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            for node in self.computing_nodes:
                nidInt=self.node_id_dict[node.id]
                observe_backlog[i, nidInt], observe_cpu[i, nidInt]= node.get_observation_state(i)

        return observe_backlog, observe_cpu

    def step_lower(self, assigned_tasks: List[Tuple[Task, int, int]]):
        f1, v_qos = 0.0, 0
        grouped_tasks = defaultdict(list)
        energy_dist={}
        f1_dist={}
        rewards=set()
        # --- Assign tasks ---
        for task, node_idx, model_idx in assigned_tasks:
            node = self.nodes[node_idx]
            is_accepted= node.admit_task(task, model_idx)
            if not is_accepted:
                v_qos += 1
                rewards.add(task.terminal_id)
            grouped_tasks[task.source_node_id].append(
                (
                    task.terminal_id,
                    one_hot(self.node_id_dict[node_idx], len(self.computing_nodes)),
                    one_hot(model_idx, self.max_models_total)
                )
            )

        # --- Process nodes ---
        all_tasks= []
        virtual_delay_info={}
        realized_delay_info={}
        success_qos_info={}
        violate_qos_info={}
        for node in self.computing_nodes:
            completed_tasks, total_energy, local_F1, violate_qos, virtual_delay, realized_delay_avg, success_qos, violation_qos = node.process_timeslot(
                self.time_manager.slot_duration
            )
            for t in completed_tasks:
                # CHỈ PHẠT NẾU TASK BỊ LỖI QOS (Trễ hạn/Drop)
                if not t.qos_status:
                    rewards.add(t.terminal_id)
            energy_dist[node.id]= total_energy
            f1_dist[node.id]= local_F1
            virtual_delay_info[node.id]= self.norm_service_vec(virtual_delay)
            realized_delay_info[node.id]= self.norm_service_vec(realized_delay_avg)
            success_qos_info[node.id]= self.norm_service_vec(success_qos)
            violate_qos_info[node.id]= self.norm_service_vec(violation_qos)
            all_tasks.extend(completed_tasks)
            # rewards[node.]= -(local_F1 + calculate_rw(self.config.hyper_neural["OMEGA_Q1"], self.config.hyper_neural["OMEGA_Q2"], violate_qos))
            f1 += local_F1
            v_qos += violate_qos

        # --- Mean-field (O(n)) ---
        mean_fields = {}
        grouped_tasks= self.group_terminal(grouped_tasks)
        for eid, tasks in grouped_tasks.items():
            n = len(tasks)

            tids = []
            node_vecs_list = []
            model_vecs_list = []

            for tid, node_vec, model_vec in tasks:
                tids.append(tid)
                node_vecs_list.append(node_vec)
                model_vecs_list.append(model_vec)

            if node_vecs_list:
                node_vecs = np.stack(node_vecs_list)  # (n, num_nodes)
                model_vecs = np.stack(model_vecs_list)  # (n, num_models)

                total_node = node_vecs.sum(axis=0)
                total_model = model_vecs.sum(axis=0)

                for i, tid in enumerate(tids):
                    if n > 1:
                        avg_node = (total_node - node_vecs[i]) / (n - 1)
                        avg_model = (total_model - model_vecs[i]) / (n - 1)
                    else:
                        avg_node = np.zeros_like(node_vecs[i])
                        avg_model = np.zeros_like(model_vecs[i])

                    mean_fields[tid] = np.concatenate([avg_node, avg_model], axis=0)

        self.frame_F1_accumulation.append(f1)
        # --- Reward ---
        # Linearize QoS penalty to avoid exponential explosion
        # Reward = -f1 - OMEGA_Q1 * v_qos (scaled up to match F1 magnitude which is ~1e7)
        qos_penalty = self.config.hyper_neural["OMEGA_Q1"] * v_qos * 1e6
        reward = -f1 - qos_penalty

        # Optional: Scale reward to a smaller range for DQN stability
        reward = reward * 1e-9
        next_observe_backlog, next_observe_cpu = self.collect_backlog_resources()
        self.time_manager.tick()
        next_states={}
        next_tasks = self.workload_gen.step(self.time_manager.time_elapsed)
        for task in next_tasks:
            mf= mean_fields[task.terminal_id] if task.terminal_id in mean_fields else (np.zeros(self.num_services), np.zeros(self.num_services))
            next_states[task.terminal_id]= task, next_observe_backlog[task.service_id], next_observe_cpu[task.service_id], mf
        info = {
                "f1": f1_dist,
                "energy": energy_dist,
                "virtual_delay": virtual_delay_info,
                "realized_delay": realized_delay_info,
                "success_qos": success_qos_info,
                "violate_qos": violate_qos_info,
        }

        # reset cpu, slot arrival workload, popular service req
        for node in self.computing_nodes:
            node.lower_reset()

        return {
            "reward":reward,
            "rewards": rewards,
            "next_states":next_states,
            "new_frame": self.time_manager.is_new_frame(),
            "done": self.time_manager.is_done(),
            "info": info
        }

    def group_terminal(self, grouped_tasks:Dict[str, List[Tuple[Task, int, int]]]):
        node_ids= set()
        grouped_edges={}
        for node in self.computing_nodes:
            if node.id not in node_ids:
                node_ids.add(node.id)
                grouped_edges[node.id]= grouped_tasks[node.id]
            else: continue

            for neighbor_node in node.neighbor_nodes:
                node_ids.add(neighbor_node.id)
                grouped_edges[node.id].extend(grouped_tasks[neighbor_node.id])
        return grouped_tasks

    def norm_service_vec(self, vec_data:Dict[int, float])->np.ndarray:
        vec= np.zeros(self.num_services)
        for sid, v in vec_data.items():
            vec[sid] = v
        return vec

    def compose_task_remaining(self):
        remaining_tasks: Dict[str, np.ndarray] = {}
        for node in self.computing_nodes:
            remaining_tasks[node.id] = node.task_remaining
        return remaining_tasks



def encoding(max_models, num_node, node_id, model_id):
    vec= np.zeros(max_models*num_node)
    vec[node_id*max_models +model_id] = 1
    return vec

def calculate_rw(ome_1, ome_2, vio:int):
    return ome_1*math.exp(ome_2*vio)