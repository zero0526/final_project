import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque

from src.configs.configs import cfg, BaseConfig
from src.envs.network.topology_manager import TopologyManager
from src.envs.network.channel_model import ChannelModel
from src.envs.entities.computing_node import ComputingNode
from src.envs.entities.terminal_node import Terminal
from src.envs.workload_generator import WorkloadGenerator
from src.envs.time_manager import TimeManager


class SixGEnvironment:
    def __init__(self, config:BaseConfig = cfg):
        """
        Môi trường mô phỏng mạng 6G tối ưu hóa AI Service Placement & Task Scheduling.
        """
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
        self._init_nodes()

        self.terminals: Dict[str, Terminal] = {}
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
        self.frame_F1_accumulation = 0.0
        self.frame_violations_accumulation = 0
        self.total_completed_tasks = 0
        self.total_energy = 0.0
        self.total_violations = 0
        self.T = cfg.neuron_net.get('TIME_SLOT_PER_TIMEFRAME', 10)
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

    def _init_terminals(self):
        edge_ids = [idx for idx, n in self.nodes.items() if n.type == "edge"]
        n= len(edge_ids)
        for i in range(cfg.hyper_neural['NUM_LOWER_AGENTS']):
            t_id = f"UE_{i}"
            self.terminals[t_id] = Terminal(
                terminal_id=t_id,
                edge_id= edge_ids[i%n],
                arrival_rate=self.config.task_arrival_rate,
                default_batch_size=self.config.default_batch_size
            )

    def reset(self):
        self.time_manager.reset()
        for node in self.nodes.values(): node.reset()
        for cid in self.cloud_node_ids: self.nodes[cid].update_placement([1] * self.num_services, self.service_config)
        self.frame_F1_accumulation = 0.0
        self.frame_violations_accumulation = 0
        self.total_completed_tasks = 0
        self.total_energy = 0.0
        self.total_violations = 0
        self.node_request_history.clear()
        self.last_terminal_actions.clear()
        self.node_frame_F1_acc.clear()
        self.node_frame_violations_acc.clear()
        self.workload_gen.step(current_time_slot=0)
        return self._get_upper_obs()

    def step_upper(self, actions: Dict[str, List[int]]):
        self.frame_F1_accumulation = 0.0
        self.frame_violations_accumulation = 0
        for nid in self.agent_node_ids:
            if nid in actions: self.nodes[nid].update_placement(actions[nid], self.service_config)

        # Reset accumulations at start of new frame
        self.node_frame_F1_acc.clear()
        self.node_frame_violations_acc.clear()

    def get_upper_feedback(self):
        rewards = {}
        for nid in self.agent_node_ids:
            penalty = self.node_frame_violations_acc[nid] * 10.0
            r = (-self.node_frame_F1_acc[nid] * 1e-11) - penalty
            rewards[nid] = r

        all_obs, all_mf = self._get_upper_obs()
        return {nid: all_obs[nid] for nid in self.agent_node_ids}, \
            {nid: all_mf[nid] for nid in self.agent_node_ids}, rewards

    def _get_upper_obs(self):
        obs, mf = {}, {}
        placements = {nid: np.array([1.0 if n.placed_services.get(i, False) else 0.0 for i in range(self.num_services)])
                      for nid, n in self.nodes.items()}
        for nid, node in self.nodes.items():
            history = self.node_request_history[nid]
            if not history:
                phi = np.zeros(self.num_services, dtype=np.float32)
            else:
                phi_sum = np.sum(np.array(list(history)), axis=0)
                tot = max(1.0, np.sum(phi_sum))
                phi = (phi_sum / tot).astype(np.float32)

            obs[nid] = np.concatenate([placements[nid], phi]).astype(np.float32)
            neigh = [n for n in self.topo_manager.get_edge_nodes_by_depth(nid, cfg.MAX_DEPTH) if n in self.nodes]
            mf[nid] = np.mean([placements[n] for n in neigh], axis=0) if neigh else np.zeros(self.num_services,
                                                                                             dtype=np.float32)
        return obs, mf

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

    def _get_lower_obs(self):
        obs, mf, masks = {}, {}, {}
        nids = sorted(self.nodes.keys())
        net = {}
        # thống kê lại q backlock và last_f_allocation cho từng service
        for sid in range(self.num_services):
            states = []
            qs, fs = [], []
            for nid in nids:
                q, f = self.nodes[nid].get_observation_state(sid)
                qs.append(np.log10(1 + q) / 6.0)
                fs.append(f)
            states.extend(qs + fs)
            net[sid] = np.array(states, dtype=np.float32)

        edge_groups = defaultdict(list)
        for tid, t in self.terminals.items(): edge_groups[t.edge_id].append(tid)

        for tid, t in self.terminals.items():
            task = t.current_task
            if task is None:
                obs[tid], msk = np.zeros(self.lower_state_dim, dtype=np.float32), np.ones(self.lower_action_dim,
                                                                                          dtype=np.float32)
            else:
                loc = np.array(
                    [np.log10(1 + task.total_data_size_mb) / 3.0, min(1.0, task.deadline / 15.0), float(task.omega),
                     task.min_accuracy / 100.0])
                obs[tid] = np.concatenate([loc, net[task.service_id]])
                msk = np.zeros(self.lower_action_dim, dtype=np.float32)
                # Unified masking for combined (Node, Model) space
                svc = self.service_config[task.service_id]
                num_models = len(svc['models'])

                for i, nid in enumerate(nids):
                    if self.nodes[nid].placed_services.get(task.service_id):
                        # Mark all valid models on this node
                        for m_idx in range(num_models):
                            msk[i * self.max_models_total + m_idx] = 1.0

                # Fallback to Cloud if no Edge node has the service
                if np.sum(msk) == 0:
                    for i, nid in enumerate(nids):
                        if nid in self.cloud_node_ids:
                            for m_idx in range(num_models):
                                msk[i * self.max_models_total + m_idx] = 1.0

            masks[tid], group = msk, edge_groups[t.edge_id]
            mf[tid] = np.mean([self.last_terminal_actions[gtid] for gtid in group], axis=0)
        return obs, mf, masks