from asyncio import tasks
from collections import deque
import numpy as np
import random
from typing import List, Dict
import math
from src.utils import KKTSolverADMM, EMA
from src.utils.MechanismUtils import calc_computation_energy, update_backlog
from src.envs.entities.task_node import Task
from src.envs.network.channel_model import ChannelModel
from src.configs.configs import cfg, BaseConfig

# init
# start time frame call upper_reset and start timeslot call lower_reset
# call admit task để assign lower level agent's action to computing node
# call mean field to get mean_field and popularity_service to get popularity_service
# update_placement to deploy new service in new timeframe
# process_timeslot to process task in queue of node
class ComputingNode:
    def __init__(self, node_id, type_node, specs, channel_model: ChannelModel,  config: BaseConfig= cfg):
        self.id = node_id

        self.specs = specs
        self.cpu_capacity = specs['cpu']  # f_v(tau)
        self.ram_capacity = specs['ram']
        self.hdd_capacity = specs['hdd']
        self.energy_coef = config.energy_coef
        self.v_balance= config.lypa_coef
        self.type = type_node

        self.service_profiles = {s.get("id"):s for s in config.services.values()}


        self.neighbor_nodes: List[ComputingNode]= []


        n= len(self.service_profiles)
        self.expected_gpu_allocations= {k:self.cpu_capacity/n for k in self.service_profiles.keys()}

        self.placed_services = np.array([0]*len(self.service_profiles))
        time_slot_in_tf= config.hyper_neural.get("TIME_SLOT_PER_TIMEFRAME", 10)
        self.popular_services_req: deque[np.ndarray]= deque(maxlen=time_slot_in_tf)
        self.F1: deque[float]= deque(maxlen=time_slot_in_tf)
        self.queues: dict[str, deque[Task]] = {}
        self.backlogs = {}
        self.slot_arrival_workload = {}
        self.cpu_allocations = {}
        self.consumption_energy = 0.0  # <--- Added to track per-node consumption

        self.used_ram = 0.0
        self.used_hdd = 0.0

        # Solver KKT
        self.config= config
        self.channel_model= channel_model
        self.ema= EMA()

    def upper_reset(self):
        self.placed_services = np.zeros_like(self.placed_services)
        # maintaining task in queue regardless of enđ time frame or timeslot
        # self.queues = {}
        # self.backlogs = {}
        self.used_ram = 0.0
        self.used_hdd = 0.0

    def lower_reset(self):
        self.popular_services_req.append(np.zeros_like(self.placed_services))
        self.cpu_allocations = {}
        self.slot_arrival_workload = {}
        self.consumption_energy = 0.0

    @property
    def mean_field(self):
        actions= np.stack([n.placed_services for n in self.neighbor_nodes])
        return actions.mean(axis=0)

    @property
    def popularity_service(self):
        total_per_service = np.sum(self.popular_services_req, axis=0)
        sum_all_services = np.sum(total_per_service)

        if sum_all_services.item() > 0:
            phi_vector = total_per_service / sum_all_services
        else:
            phi_vector = np.zeros_like(total_per_service)

        # (Phi_v,s)
        return phi_vector


    def update_placement(self, placement_vector:np.ndarray):
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
        task.queue_delay=0
        if len(self.popular_services_req) == 0:
            self.popular_services_req.append(np.zeros_like(self.placed_services))
        self.popular_services_req[-1][sid]+=task.batch_size

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

    def __clear_queue_task(self, s_id):
        violate_qos = 0
        tasks_to_keep = deque()

        for t in self.queues[s_id]:
            # Nếu thời gian chờ vượt quá deadline -> Vứt bỏ
            if t.queue_delay + self.config.hyper_neural["SLOT_DURATION"] > t.deadline:
                violate_qos += 1
                self.backlogs[s_id] -= t.remaining_workload_gflops
            else:
                tasks_to_keep.append(t)

        self.queues[s_id] = tasks_to_keep  # Cập nhật lại hàng đợi
        self.backlogs[s_id] = max(0.0, self.backlogs[s_id])  # Đảm bảo không bị số âm
        return violate_qos

    def process_timeslot(self, current_time_elapsed, slot_duration):
        active_svcs = [sid for sid, active in enumerate(self.placed_services) if active]
        violate_qos=0
        for s_id in self.service_profiles:
            if s_id not in active_svcs:
                violate_qos+=self.__clear_queue_task(s_id)
        if not active_svcs:
            return [], 0.0

        print(f"\n--- [NODE {self.id}] TIMESLOT LOG (Duration: {slot_duration}s) ---")
        print(f"    Available CPU: {self.cpu_capacity} GFLOPS")

        f_alloc_vec, slot_cold_times = self._compute_optimal_resources(
            active_svcs
        )

        completed_tasks, total_energy, lypa_punish = self._execute_allocation(
            active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed
        )
        local_F1= total_energy*self.config.lypa_coef + lypa_punish
        self.F1.append(local_F1)
        violate_qos+=sum(1 for t in completed_tasks if not t.qos_status)
        reward= -(local_F1 + self._calculate_QoS(violate_qos))
        self.slot_arrival_workload = {}
        self.consumption_energy = total_energy

        return completed_tasks, total_energy, reward

    def _calculate_QoS(self, violate_qos: int)->float:
        return self.config.hyper_neural["OMEGA_Q1"]*math.exp(violate_qos*self.config.hyper_neural["OMEGA_Q2"])

    def __min_requirement_gpu(self, sid):
        # estimate by ignore overloading cause over deadline before computing process
        expect_allocation= self.expected_gpu_allocations[sid]
        t_cold_start, e_cold_start=0, 0
        if not self.service_profiles[sid].get("omega"):
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
                expected_computing_time = self.config.delay_coef*(task.deadline - task.created_at - task.time_consume -delay)

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
            z_vec[i] = self.v_balance * self.energy_coef * arrival_load
            z_vec[i] = max(z_vec[i], 1e-12)

            f_max_vec[i] = self.cpu_capacity
            f_min_vec[i], cold_times = self.__min_requirement_gpu(sid)
            slot_cold_times[sid] = cold_times
        f_alloc_vec = solver.solve(g_vec, z_vec, f_min_vec, f_max_vec)
        for i, sid in enumerate(active_svcs):
            self.expected_gpu_allocations[sid] = self.ema.update(
                f_alloc_vec[i],
                self.expected_gpu_allocations[sid]
            )
        return f_alloc_vec, slot_cold_times

    def _execute_allocation(self, active_svcs, f_alloc_vec, slot_cold_times, slot_duration, current_time_elapsed):
        total_energy = 0.0
        completed_tasks = []
        self.cpu_allocations = {}
        lypa_punish= 0.0
        for i, sid in enumerate(active_svcs):
            f_val = f_alloc_vec[i]
            if not np.isfinite(f_val): f_val = 0.0

            self.cpu_allocations[sid] = f_val

            profile = self.service_profiles[sid]
            omega = profile['omega']
            t_cold = slot_cold_times.get(sid, 0.0)

            processed_workload = f_val * slot_duration
            lypa_punish+=self.backlogs.get(sid, 0.0)*(self.slot_arrival_workload.get(sid, 0.0) - processed_workload)
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
            current_slot_time = 0
            while self.queues[sid] and remaining_cap > 0:

                task = self.queues[sid][0]
                task.queue_delay += current_slot_time
                if task.remaining_workload_gflops <= remaining_cap:
                    processed_now = task.remaining_workload_gflops
                    duration = processed_now / f_val if f_val > 0 else 0

                    remaining_cap -= processed_now
                    task.remaining_workload_gflops = 0
                    done_task = self.queues[sid].popleft()

                    task.trace_task({"computation_delay": duration})
                    finish_time= task.time_consume + task.created_at

                    done_task.qos_status = finish_time <= done_task.deadline

                    done_task.finished_at = finish_time
                    completed_tasks.append(done_task)
                    current_slot_time += duration
                else:
                    processed_now = remaining_cap
                    duration = processed_now / f_val if f_val > 0 else 0

                    task.computation_delay += duration
                    task.remaining_workload_gflops -= processed_now
                    remaining_cap = 0
                    current_slot_time += duration

            for t in self.queues[sid]:
                t.queue_delay += current_slot_time
        print(f"    => Node Total Energy: {total_energy:.4f} J | Tasks Finished: {len(completed_tasks)}")
        return completed_tasks, total_energy, lypa_punish


    def get_observation_state(self, service_id):
        """
        Trả về trạng thái cụ thể cho 1 service (Eq. 51):
        1. Queue Backlog (Q) - Trả về raw để env tự normalize
        2. Last CPU Allocation (f) - Normalization 1/capacity
        """
        q = self.backlogs.get(service_id, 0.0)
        f = self.cpu_allocations.get(service_id, 0.0) / (self.cpu_capacity if self.cpu_capacity > 0 else 1.0)

        return [q, f]