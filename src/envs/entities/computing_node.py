from collections import deque
import numpy as np
import random
from typing import List

from src.utils import KKTSolverADMM
from src.utils.MechanismUtils import calc_computation_energy, update_backlog
from src.envs.entities.task_node import Task
from src.envs.time_manager import TimeManager
from src.envs.network.channel_model import ChannelModel
from src.configs.configs import cfg, BaseConfig


class ComputingNode:
    def __init__(self, node_id, specs, time_manager:TimeManager, channel_model: ChannelModel,  config: BaseConfig= cfg):
        self.id = node_id

        self.specs = specs
        self.cpu_capacity = specs['cpu']  # f_v(tau)
        self.ram_capacity = specs['ram']
        self.hdd_capacity = specs['hdd']
        self.energy_coef = config.energy_coef
        self.v_balance= config.lypa_coef
        self.type = specs['type']

        self.service_profiles = {s.get("id"):s for s in config.services.values()}

        # x(t-1) x(t)
        self.placed_services = np.array([0]*len(self.service_profiles))

        self.neighbor_nodes: List[ComputingNode]= []

        self.popular_services_req= {k: [0]*config.hyper_neural.get("TIME_SLOT_PER_TIMEFRAME", 10) for k in self.service_profiles.keys()}

        n= len(self.service_profiles)
        self.expected_gpu_allocations= {k:self.cpu_capacity/n for k in self.service_profiles.keys()}

        self.queues: dict[str, deque[Task]] = {}
        self.backlogs = {}
        self.slot_arrival_workload = {}
        self.last_cpu_allocations = {}
        self.last_energy = 0.0  # <--- Added to track per-node consumption

        self.used_ram = 0.0
        self.used_hdd = 0.0

        # Solver KKT
        self.config= config
        self.time_manager = time_manager
        self.channel_model= channel_model

    @property
    def mean_field(self):
        actions= np.stack([n.placed_services for n in self.neighbor_nodes])
        return actions.mean(axis=0)

    @property
    def popularity_service(self):
        sorted_ids = sorted(self.popular_services_req.keys())
        requests_matrix = np.array([self.popular_services_req[i] for i in sorted_ids])  # Shape: (S, T)

        sum_per_service = np.sum(requests_matrix, axis=1)  # Shape: (S,)

        total_requests = np.sum(sum_per_service) + 1e-8

        # (Phi_v,s)
        return sum_per_service / total_requests

    def reset(self):
        self.placed_services = {}
        self.queues = {}
        self.backlogs = {}
        self.slot_arrival_workload = {}
        self.last_cpu_allocations = {}
        self.last_energy = 0.0
        self.used_ram = 0.0
        self.used_hdd = 0.0

    def update_placement(self, placement_vector):
        constraint_violations = 0

        for svc_id, decision in enumerate(placement_vector):
            if decision == 1:
                profile = self.service_profiles[svc_id]
                success = self._deploy_single_service(profile)
                if not success:
                    constraint_violations += 1

        return constraint_violations

    def _deploy_single_service(self, profile):
        omega = profile['omega']
        size_gb = profile['size'] / 1024.0
        can_deploy = False

        if omega == 1 and self.used_ram + size_gb <= self.ram_capacity:
            self.used_ram += size_gb
            can_deploy = True
        elif omega == 0 and self.used_hdd + size_gb <= self.hdd_capacity:
            self.used_hdd += size_gb
            can_deploy = True

        if can_deploy:
            svc_id = profile['id']
            self.placed_services[svc_id] = 1

            if svc_id not in self.queues:
                self.queues[svc_id] = deque()
                self.backlogs[svc_id] = 0.0

            return True
        return False

    def admit_task(self, task: Task, model_idx: int):
        """
        :param task:
        :param model_idx: order in the file config service
        :return:
        """
        sid = task.service_id
        task.assign_schedule(self.id, model_idx)
        task_metadata= self.channel_model.get_metadata(task.source_node_id, self.id, task.total_data_size_mb)
        task.trace_task(task_metadata)
        if task.required_workload_gflops <= 0: return False

        self.queues[sid].append(task)

        self.backlogs[sid] = update_backlog(
            current_backlog=self.backlogs[sid],
            processed_workload=0,
            arrival_workload=task.required_workload_gflops
        )

        if sid not in self.slot_arrival_workload:
            self.slot_arrival_workload[sid] = 0.0
        self.slot_arrival_workload[sid] += task.required_workload_gflops

        return True

    def process_timeslot(self, current_time_elapsed, slot_duration):
        active_svcs = [sid for sid, active in np.ndenumerate(self.placed_services) if active]
        if not active_svcs:
            return [], 0.0

        print(f"\n--- [NODE {self.id}] TIMESLOT LOG (Duration: {slot_duration}s) ---")
        print(f"    Available CPU: {self.cpu_capacity} GFLOPS")

        f_alloc_vec, slot_cold_times = self._compute_optimal_resources(
            active_svcs
        )

        completed_tasks, total_energy = self._execute_allocation(
            active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed
        )

        self.slot_arrival_workload = {}
        self.last_energy = total_energy

        return completed_tasks, total_energy

    def __min_requirement_gpu(self, sid):
        # estimate by ignore overloading cause over deadline before computing process
        expect_allocation= self.expected_gpu_allocations[sid]
        t_cold_start, e_cold_start=0, 0
        if self.service_profiles[sid].get("omega"):
            t_cold_start= random.uniform(self.config.cold_start_time.get("min"), self.config.cold_start_time.get("max"))
            e_cold_start= t_cold_start*self.config.cold_start_energy_coef
        tasks= self.queues[sid]
        delay=0
        min_req=0
        is_first= True
        for task in tasks:
            if task.is_assigned:
                if is_first:
                    task.trace_task({"cold_start_delay": t_cold_start,
                                     "cold_start_energy": e_cold_start})
                    is_first= False
                time_remaining = task.deadline - task.created_at - task.time_comsume
                if time_remaining <1e-8:
                    continue
                expected_computing_time= self.config.delay_coef*(time_remaining - delay)
                if expected_computing_time < 1e-8:
                    min_req= self.cpu_capacity
                    break
                cpu_require= task.remaining_workload_gflops/expected_computing_time
                min_req= max(min_req, cpu_require)
                delay+= task.remaining_workload_gflops/(expect_allocation + 1e-8)
        return min_req, t_cold_start

    def _compute_optimal_resources(self, active_svcs):
        num_active = len(active_svcs)
        g_vec = np.zeros(num_active)
        z_vec = np.zeros(num_active)
        f_min_vec = np.zeros(num_active)
        f_max_vec = np.zeros(num_active)
        slot_cold_times={}
        solver= KKTSolverADMM(f_max_node= self.cpu_capacity,rho=1.0, max_iter=100)
        for i, sid in enumerate(active_svcs):
            g_vec[i] = self.backlogs.get(sid, 0.0)

            arrival_load = self.slot_arrival_workload.get(sid, 0.0)
            z_vec[i] = self.v_balance * self.energy_coeff * arrival_load
            z_vec[i] = max(z_vec[i], 1e-12)

            f_max_vec[i] = self.cpu_capacity
            f_min_vec[i], cold_times = self.__min_requirement_gpu(sid)
            slot_cold_times[sid] = cold_times
        f_alloc_vec = solver.solve(g_vec, z_vec, f_min_vec, f_max_vec)

        return f_alloc_vec, slot_cold_times

    def _execute_allocation(self, active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed):
        total_energy = 0.0
        completed_tasks = []
        self.last_cpu_allocations = {}

        for i, sid in enumerate(active_svcs):
            f_val = f_alloc_vec[i]
            if not np.isfinite(f_val): f_val = 0.0

            self.last_cpu_allocations[sid] = f_val

            profile = self.service_profiles[sid]
            omega = profile['omega']
            t_cold = slot_cold_times.get(sid, 0.0)

            processed_workload = f_val * slot_duration
            q_before = self.backlogs.get(sid, 0.0)
            self.backlogs[sid] = max(0.0, q_before - processed_workload)
            q_after = self.backlogs[sid]

            actual_workload = q_before - q_after

            effective_t_cold = t_cold if f_val > 1e-6 else 0.0

            e_comp = calc_computation_energy(
                epsilon_c=self.config.energy_coef,
                f_allocated=f_val,
                workload_total=actual_workload,
                omega=omega,
                epsilon_cold=self.config.cold_start_energy_coef,
                t_cold=effective_t_cold
            )

            total_energy += e_comp

            print(f"    [Svc {sid:2}] CPU: {f_val:8.2f} GFLOPS ({f_val / self.cpu_capacity * 100:5.1f}%) | "
                  f"Energy: {e_comp:8.4f} J | "
                  f"Processed: {actual_workload:8.2f} GFLOPS | "
                  f"Queue: {q_before:8.2f} -> {q_after:8.2f} GFLOPS")

            remaining_cap = processed_workload
            delay= 0
            while self.queues[sid] and remaining_cap > 0:
                task = self.queues[sid][0]

                if task.remaining_workload <= remaining_cap:
                    remaining_cap -= task.remaining_workload
                    task.remaining_workload = 0
                    done_task = self.queues[sid].popleft()

                    time_in_slot = (processed_workload - remaining_cap) / f_val if f_val > 0 else 0
                    task.trace_task({"queue_delay": delay, "computation_delay": time_in_slot})
                    finish_time= task.time_comsume + task.created_at

                    if finish_time> done_task.deadline:
                        # Task này đã chết rồii, không tính là thành công cho dù xử lý xong
                        done_task.qos_status = False
                    else:
                        done_task.qos_status = True

                    done_task.finished_at = finish_time
                    delay+=time_in_slot
                    completed_tasks.append(done_task)
                else:
                    task.remaining_workload -= remaining_cap
                    remaining_cap = 0

        print(f"    => Node Total Energy: {total_energy:.4f} J | Tasks Finished: {len(completed_tasks)}")
        return completed_tasks, total_energy

    def get_observation_state(self, service_id):
        """
        Trả về trạng thái cụ thể cho 1 service (Eq. 51):
        1. Queue Backlog (Q) - Trả về raw để env tự normalize
        2. Last CPU Allocation (f) - Normalization 1/capacity
        """
        q = self.backlogs.get(service_id, 0.0)
        f = self.last_cpu_allocations.get(service_id, 0.0) / (self.cpu_capacity if self.cpu_capacity > 0 else 1.0)

        return [q, f]