# Implementation Plan - Matrix Vectorization and GPU Optimization

Optimize the 6G RL environment by replacing `for` loops with vectorized matrix operations using PyTorch. This will allow the environment state updates, resource allocation (ADMM solver), and reward calculations to run efficiently on GPUs.

## Proposed Changes

### Core Environment Logic
#### [MODIFY] [environment.py](file:///d:/code/ai_infras/src/envs/environment.py)
- Transition [nodes](file:///d:/code/ai_infras/src/envs/environment.py#54-77) state from a list of objects to centralized PyTorch tensors:
  - `backlog_matrix`: [(num_nodes, num_services)](file:///d:/code/ai_infras/src/utils/MathUtils.py#108-118)
  - `cpu_allocation_matrix`: [(num_nodes, num_services)](file:///d:/code/ai_infras/src/utils/MathUtils.py#108-118)
  - `placement_matrix`: [(num_nodes, num_services)](file:///d:/code/ai_infras/src/utils/MathUtils.py#108-118)
  - `mean_field_matrix`: [(num_nodes, num_services)](file:///d:/code/ai_infras/src/utils/MathUtils.py#108-118) - Stores the pre-calculated or rolling average of neighborhood actions.
- Implement static network matrices to replace real-time pathfinding:
  - `path_matrix`: Pre-calculated paths between all node pairs.
  - `delay_matrix`: Static or semi-static propagation/transmission delay matrix.
  - `resource_matrix`: Static node capacities (CPU/RAM/HDD) stored as tensors.
- Rewrite [collect_backlog_resources](file:///d:/code/ai_infras/src/envs/environment.py#179-196) to directly return tensor slices.
- Vectorize [step_lower](file:///d:/code/ai_infras/src/envs/environment.py#197-356) to handle batch task assignments and status updates without internal loops per node.
- Use `torch` instead of `numpy` for core calculations where GPU acceleration is beneficial.

### Computing Node Entity
#### [MODIFY] [computing_node.py](file:///d:/code/ai_infras/src/envs/entities/computing_node.py)
- Refactor [ComputingNode](file:///d:/code/ai_infras/src/envs/entities/computing_node.py#22-420) to be a lightweight interface or delegate state management to the environment's tensor matrices.
- Vectorize [process_timeslot](file:///d:/code/ai_infras/src/envs/entities/computing_node.py#202-246) to operate on service-level blocks.
- Replace manual `deque` task management with tensor-based queue tracking if possible (e.g., tracking task counts and average delays).

### Mathematical Utilities
#### [MODIFY] [MathUtils.py](file:///d:/code/ai_infras/src/utils/MathUtils.py)
- Vectorize `KKTSolverADMM.solve` to handle multiple nodes and services simultaneously using PyTorch operations.
- Replace [project_simplex](file:///d:/code/ai_infras/src/utils/MathUtils.py#13-40) with a batch-friendly version.

---

## Verification Plan

### Automated Tests
- **Environment Parity Test**: Create a script `tests/verify_vectorization.py` that runs the original and vectorized environment with the same seed/tasks and compares:
  - Reward values (up to float precision).
  - Next state tensors.
  - Final metrics (Energy, F1, QoS violations).
  - Performance Benchmark: Compare execution time of 1000 steps between old and new implementations.
- **Run Command**: `python tests/tes_env.py` (after updates) to ensure basic functionality.

### Manual Verification
- Check GPU memory usage during training to ensure tensors are correctly allocated on `cfg.device`.
