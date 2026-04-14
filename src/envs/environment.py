import math

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque

from networkx.classes import neighbors
from torch.distributed.checkpoint._experimental import config

from envs import time_manager
from envs.entities.task_node import Task
from src.utils import convert_nodeid2order, one_hot
from src.configs.configs import cfg, BaseConfig
from src.envs.network.topology_manager import TopologyManager
from src.envs.network.channel_model import ChannelModel
from src.envs.entities.computing_node import ComputingNode
from src.envs.entities.terminal_node import Terminal
from src.envs.workload_generator import WorkloadGenerator
from src.envs.time_manager import TimeManager


class SixGEnvironment:
    def __init__(self, config:BaseConfig = cfg):
        self.config = config
        self.service_config = config.services
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
        self._init_nodes()

        self.terminals: Dict[str, Terminal] = {}
        self.terminals_group: Dict[str, List[Terminal]]= defaultdict(list)
        self._init_terminals()

        self.workload_gen = WorkloadGenerator(
            terminals=list(self.terminals.values())
        )

        self.time_manager = TimeManager()

        self.num_nodes_total = len(self.nodes)
        self.max_models_total = max([len(svc['models']) for svc in self.service_config])

        self.upper_state_dim = 2 * self.num_services
        self.upper_action_dim = 1 << self.num_services

        self.lower_state_dim = 4 + (2 * self.num_nodes_total)
        self.lower_action_dim = self.num_nodes_total * self.max_models_total
        self.mf_lower_dim = self.num_nodes_total + self.max_models_total

        self.current_episode = 0
        self.total_completed_tasks = 0
        self.total_energy = 0.0
        self.total_violations = 0
        self.T = cfg.neuron_net.get('TIME_SLOT_PER_TIMEFRAME', 10)
        self.frame_F1_accumulation: deque[float]= deque(maxlen=self.T)
        self.node_request_history = defaultdict(lambda: deque(maxlen=self.T))
        self.last_terminal_actions = defaultdict(lambda: np.zeros(self.mf_lower_dim, dtype=np.float32))

        # Per-node frame accumulations for rewards
        self.node_frame_F1_acc = defaultdict(float)
        self.node_frame_violations_acc = defaultdict(int)

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
                self.computing_nodes.append(node)
        self.computing_nodes = sorted(self.computing_nodes, key=lambda x: convert_nodeid2order(x.id))
        # add neighborhoods
        for node_id, node  in self.nodes.items():
            neighbor_ids= self.topo_manager.get_neighbor_nodes_by_type(node_id, 3, ["edge", "network", "cloud"])
            self.nodes[node_id].neighbor_nodes= [self.nodes[i] for i in neighbor_ids]

    def _init_terminals(self):
        edge_ids = [idx for idx, n in self.nodes.items() if n.type == "edge"]
        n= len(edge_ids)
        for i in range(cfg.hyper_neural['NUM_LOWER_AGENTS']):
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
        next_tasks = self.workload_gen.step(self.time_manager.time_elapsed)
        next_states={}
        num_nodes = len(self.computing_nodes)
        next_observe_backlog, next_observe_cpu = self.collect_backlog_resources()
        for task in next_tasks:
            next_states[task.terminal_id]= task, next_observe_backlog[task.service_id], next_observe_cpu[task.service_id], (np.zeros(num_nodes), np.zeros(self.max_models_total))

        return {
            "reward": 0.0,
            "next_states": next_states,
            "new_frame": self.time_manager.is_new_frame()
        }

    def reset_upper(self):
        self.time_manager.reset()
        for node in self.nodes.values():
            if node.type in ["edge", "network"]:
                node.upper_reset()
                node.lower_reset()
        self.nodes[self.cloud_node_id].update_placement(np.ones(self.num_services))
        self.frame_F1_accumulation = 0.0
        self.frame_violations_accumulation = 0
        self.total_completed_tasks = 0
        self.total_energy = 0.0
        self.total_violations = 0
        self.node_request_history.clear()
        self.last_terminal_actions.clear()
        self.node_frame_F1_acc.clear()
        self.node_frame_violations_acc.clear()
        self.workload_gen.step(abs_current_time=0)

        states = {}
        mean_fields = {}
        n= self.num_services
        reward = -sum(f for f in self.frame_F1_accumulation)
        for node in self.computing_nodes:
            states[node.id] = (np.zeros(n), np.zeros(n))
            mean_fields[node.id] = np.zeros(n)
        return {
            "reward": 0.0,
            "states": states,
            "mean_fields": mean_fields,
            "done": self.time_manager.is_done()
        }

    def step_upper(self, actions: Dict[str, np.ndarray]):
        states={}
        mean_fields= {}
        reward= -sum(f for f in self.frame_F1_accumulation)
        for node in self.computing_nodes:
            states[node.id]= (node.placed_services, node.popularity_service)
            mean_fields[node.id]= node.mean_field
            node.upper_reset()
        self.frame_F1_accumulation = deque(maxlen=self.T)
        for nid in self.agent_node_ids:
            if nid in actions: self.nodes[nid].update_placement(actions[nid])

        # Reset accumulations at start of new frame
        self.node_frame_F1_acc.clear()
        self.node_frame_violations_acc.clear()
        return {
            "reward": reward,
            "states": states,
            "mean_fields": mean_fields,
            "done": self.time_manager.is_done()
        }

    def step_lower(self, actions_map: Dict[str, int]):
        slot_energy, slot_violations, slot_success, slot_avg_arrival = 0.0, 0, 0, 0
        node_slot_energy = defaultdict(float)
        node_slot_violations = defaultdict(int)
        arrivals_A = defaultdict(float)
        slot_request_counts = {nid: np.zeros(self.num_services) for nid in self.nodes}
        node_list = sorted(self.nodes.keys())

        for tid, action_id in actions_map.items():
            # action_id is a joint index [0, N*M - 1]
            node_idx = action_id // self.max_models_total
            model_idx = action_id % self.max_models_total

            target_nid = node_list[node_idx]
            # MF vector represents influence in N + M space
            self.last_terminal_actions[tid] = np.zeros(self.mf_lower_dim, dtype=np.float32)
            self.last_terminal_actions[tid][node_idx] = 1.0
            self.last_terminal_actions[tid][self.num_nodes_total + model_idx] = 1.0

            task = self.terminals[tid].current_task
            if task:
                slot_avg_arrival += 1
                slot_request_counts[target_nid][task.service_id] += 1
                if target_nid in self.nodes:
                    svc = self.service_config[task.service_id]
                    m_idx = model_idx % len(svc['models'])
                    unit_load = svc['models'][m_idx]['workload']
                    task.assign_schedule(target_nid, m_idx, unit_load)
                    arrivals_A[(target_nid, task.service_id)] += (unit_load * task.batch_size)
                    meta = self.channel_model.get_metadata(self.terminals[tid].edge_id, target_nid,
                                                           task.total_data_size_mb)

                    e_trans = meta['transmission_energy']
                    slot_energy += e_trans
                    node_slot_energy[target_nid] += e_trans

                    if not self.nodes[target_nid].admit_task(task):
                        slot_violations += 1
                        node_slot_violations[target_nid] += 1

        curr_time = self.time_manager.to_abs_time(self.time_manager.current_slot)
        queues_before = {nid: node.backlogs.copy() for nid, node in self.nodes.items()}
        for nid, node in self.nodes.items():
            done, n_e = node.process_timeslot(curr_time, self.time_manager.slot_duration)
            slot_energy += n_e
            node_slot_energy[nid] += n_e
            self.total_completed_tasks += len(done)
            for t in done:
                if t.qos_status:
                    slot_success += 1
                else:
                    slot_violations += 1
                    node_slot_violations[nid] += 1

        # F1_tau = Drift + V*Energy.
        v_energy = 1e-5
        total_f1 = 0.0
        for nid in self.nodes:
            node_drift = 0.0
            for sid in range(self.num_services):
                Q, A = queues_before[nid].get(sid, 0.0), arrivals_A.get((nid, sid), 0.0)
                W = self.nodes[nid].last_cpu_allocations.get(sid, 0.0) * self.time_manager.slot_duration
                node_drift += Q * (A - W)

            node_f1 = node_drift + (v_energy * node_slot_energy[nid])
            self.node_frame_F1_acc[nid] += node_f1
            self.node_frame_violations_acc[nid] += node_slot_violations[nid]
            total_f1 += node_f1

        self.frame_F1_accumulation += total_f1
        self.frame_violations_accumulation += slot_violations
        self.total_energy += slot_energy
        self.total_violations += slot_violations

        for nid in self.nodes:
            self.node_request_history[nid].append(slot_request_counts[nid])

        self.time_manager.tick()
        if not self.time_manager.is_done(): self.workload_gen.step(self.time_manager.current_slot)

        next_obs, next_mf, _ = self._get_lower_obs()

        r_drift = -np.clip(total_f1 * 5e-10, -30.0, 30.0)
        r_vio = -slot_violations * 30.0
        r_success = slot_success * 20.0

        total_r = r_drift + r_vio + r_success
        rewards = {tid: total_r for tid in self.terminals}

        info = {
            "energy": slot_energy, "violations": slot_violations, "success": slot_success,
            "arrival_tasks": slot_avg_arrival, "F1_tau": total_f1,
            "is_new_frame": self.time_manager.is_new_frame(),
            "is_done": self.time_manager.is_done()
        }
        return next_obs, rewards, self.time_manager.is_done(), info


    def collect_backlog_resources(self):
        """
        service_id start from 0->n-1 when using index from config service id use minus 1
        """
        n, m= len(self.computing_nodes), self.num_services
        observe_backlog= np.zeros((m,n), dtype=np.float32)
        observe_cpu = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            for j, node in enumerate(self.computing_nodes):
                observe_backlog[i, j], observe_cpu[i, j]= node.get_observation_state(i)

        return observe_backlog, observe_cpu

    def step_lower(self, assigned_tasks: List[Tuple[Task, int, int]]):
        f1, v_qos = 0.0, 0
        grouped_tasks = defaultdict(list)
        num_nodes = len(self.computing_nodes)

        # --- Assign tasks ---
        for task, node_idx, model_idx in assigned_tasks:
            node = self.computing_nodes[node_idx]
            is_accepted= node.admit_task(task, model_idx)
            if not is_accepted: v_qos += 1
            grouped_tasks[task.source_node_id].append(
                (
                    task.terminal_id,
                    one_hot(node_idx, num_nodes),
                    one_hot(model_idx, self.max_models_total),
                )
            )

        # --- Process nodes ---
        all_tasks= []
        for node in self.computing_nodes:
            completed_tasks, total_energy, local_F1, violate_qos = node.process_timeslot(
                self.time_manager.slot_duration
            )
            all_tasks.extend(completed_tasks)
            f1 += local_F1
            v_qos += violate_qos

        # --- Mean-field (O(n)) ---
        mean_fields = {}

        for eid, tasks in grouped_tasks.items():
            n = len(tasks)

            tids = []
            node_vecs = []
            model_vecs = []

            for tid, node_vec, model_vec in tasks:
                tids.append(tid)
                node_vecs.append(node_vec)
                model_vecs.append(model_vec)

            node_vecs = np.stack(node_vecs)  # (n, num_nodes)
            model_vecs = np.stack(model_vecs)  # (n, num_models)

            total_node = node_vecs.sum(axis=0)
            total_model = model_vecs.sum(axis=0)

            for i, tid in enumerate(tids):
                if n > 1:
                    avg_node = (total_node - node_vecs[i]) / (n - 1)
                    avg_model = (total_model - model_vecs[i]) / (n - 1)
                else:
                    avg_node = np.zeros_like(node_vecs[i])
                    avg_model = np.zeros_like(model_vecs[i])

                mean_fields[tid] = (avg_node, avg_model)
        self.frame_F1_accumulation.append(f1)
        # --- Reward ---
        reward = -f1 - self.config.hyper_neural["OMEGA_Q1"] * math.exp(
            self.config.hyper_neural["OMEGA_Q2"] * v_qos
        )
        next_observe_backlog, next_observe_cpu = self.collect_backlog_resources()
        self.time_manager.tick()
        next_tasks = self.workload_gen.step(self.time_manager.time_elapsed)
        next_states={}
        for task in next_tasks:
            next_states[task.terminal_id]= task, next_observe_backlog[task.service_id], next_observe_cpu[task.service_id], mean_fields[task.terminal_id]

        return {
            "reward":reward,
            "next_states":next_states,
            "new_frame": self.time_manager.is_new_frame()
        }
