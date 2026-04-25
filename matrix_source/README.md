# Matrix-Based 6G RL Framework

## Tổng quan
Mục tiêu của framework này là chuyển đổi toàn bộ logic mô phỏng môi trường 6G từ dạng vòng lặp (Iteration-based) sang dạng ma trận (Matrix-based) sử dụng PyTorch. Điều này cho phép:
- Tận dụng tối đa sức mạnh của GPU.
- Giảm thiểu overhead của Python loops.
- Tương thích tốt hơn với các mô hình RL hiện đại đang dùng PyTorch.

## Cấu trúc thành phần
1. **`envs/`**: Chứa môi trường mô phỏng chính (`matrix_env.py`), bộ sinh task (`workload_generator.py`) và bộ khởi tạo ma trận (`init_matrices.py`).
2. **`models/`**: Chứa các bộ giải toán (Solvers) tối ưu hóa tài nguyên. Nhiệm vụ là tìm lời giải tối ưu cho bài toán phân bổ CPU/RAM theo Batch.
3. **`utils/`**: Các hàm bổ trợ tính toán Delay, Energy, QoS và các thao tác trên Tensor.
4. **`configs/`**: Quản lý cấu hình dưới dạng matrix-ready.

## Cách tổ chức dữ liệu Matrix-based
- **State Matrices**: `(M x N)` - Backlog, CPU, Placement.
- **Metadata Tensors**: `(1 x N)` - Deadline, Workload, Size (dùng broadcasting).
- **Mapping Matrices**: `(Terminals x Nodes)` - Kết nối giữa UE và Edge.
- **Topology Matrices**: `(Nodes x Nodes)` - Delay và Path tĩnh.

## Lộ trình Implement
- [ ] Chuyển đổi Topology sang dạng Static Matrix.
- [ ] Xây dựng bộ Mapping và Metadata Tensors.
- [ ] Implement Batch ADMM Solver.
- [ ] Xây dựng môi trường MatrixSixGEnvironment tích hợp bộ đếm thời gian mới.
