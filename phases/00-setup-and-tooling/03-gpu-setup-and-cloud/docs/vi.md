# Thiết lập GPU & Điện toán đám mây (GPU Setup & Cloud)

> Huấn luyện trên CPU hoàn toàn ổn khi bắt đầu học. Nhưng huấn luyện thực chiến cần có GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 phút

## Mục tiêu học tập

- Kiểm tra tính khả dụng của GPU cục bộ bằng lệnh `nvidia-smi` và PyTorch CUDA API.
- Cấu hình Google Colab với GPU T4 cho các thử nghiệm miễn phí trên đám mây.
- Đo điểm chuẩn (benchmark) phép nhân ma trận giữa CPU và GPU, tính toán tốc độ tăng tốc (speedup).
- Ước tính kích thước mô hình tối đa có thể tải vào VRAM theo quy tắc ngón tay cái fp16.

## Vấn đề đặt ra

Hầu hết các bài học ở Phases 1-3 chạy rất tốt trên CPU. Nhưng khi bạn bắt đầu huấn luyện CNN, Transformer, hoặc LLM (từ Phase 4 trở đi), bạn bắt buộc cần tăng tốc bằng GPU. Một lần huấn luyện mất 8 giờ trên CPU chỉ mất khoảng 10 phút trên GPU.

Bạn có ba lựa chọn: GPU cục bộ, GPU đám mây (cloud GPU), hoặc Google Colab (miễn phí).

## Khái niệm

```
Các lựa chọn của bạn:

1. GPU NVIDIA cục bộ (trên máy tính của bạn)
   Chi phí: $0 (nếu máy bạn đã có sẵn)
   Thiết lập: Cài đặt CUDA + cuDNN
   Phù hợp nhất: Sử dụng thường xuyên, tập dữ liệu lớn

2. Google Colab (gói miễn phí)
   Chi phí: $0
   Thiết lập: Không cần
   Phù hợp nhất: Thử nghiệm nhanh, người không có GPU tại nhà

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Chi phí: $0.20 - $2.00 / giờ
   Thiết lập: Kết nối SSH + cài đặt thư viện
   Phù hợp nhất: Huấn luyện nghiêm túc, mô hình lớn
```

```figure
s0-gpu-dispatch
```

## Bắt tay vào xây dựng (Build It)

### Lựa chọn 1: GPU NVIDIA cục bộ

Kiểm tra xem máy bạn có GPU NVIDIA không:

```bash
nvidia-smi
```

Cài đặt PyTorch với CUDA. Trên Linux, gói wheel mặc định đã đi kèm CUDA. Trên Windows, hãy lấy lệnh cài CUDA chính xác từ trang chọn pytorch.org.

```bash
pip install torch torchvision torchaudio
```

Sau đó kiểm tra xem PyTorch đã nhận diện GPU hay chưa:

```python
import torch

print(f"CUDA khả dụng: {torch.cuda.is_available()}")
print(f"Phiên bản CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dung lượng bộ nhớ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Lựa chọn 2: Google Colab

1. Truy cập [colab.research.google.com](https://colab.research.google.com)
2. Chọn menu Runtime > Change runtime type > Chọn T4 GPU
3. Chạy lệnh `!nvidia-smi` để kiểm tra

Bạn có thể tải các notebook từ khóa học này trực tiếp lên Colab để chạy.

### Lựa chọn 3: Cloud GPU

Dành cho Lambda Labs, RunPod, hoặc Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Không có GPU? Không sao cả!

Hầu hết các bài học đều chạy được trên CPU. Những bài học bắt buộc cần GPU sẽ được ghi chú rõ ràng kèm đường dẫn Colab sẵn sàng chạy.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")
```

## Đo điểm chuẩn: Benchmark so sánh GPU và CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"Thời gian trên CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"Thời gian trên GPU: {gpu_time:.3f}s")
    print(f"Tốc độ tăng tốc: {cpu_time / gpu_time:.0f}x")
```

## Bài tập thực hành (Exercises)

1. Chạy đoạn code benchmark ở trên và so sánh thời gian chạy giữa CPU và GPU.
2. Nếu bạn không có GPU trên máy, hãy chạy thử đoạn code trên Google Colab và so sánh.
3. Kiểm tra dung lượng VRAM của bạn và ước tính kích thước mô hình lớn nhất có thể nạp vừa (quy tắc ngón tay cái: 2 byte cho mỗi tham số ở định dạng fp16).

### Gợi ý (Hints)

**Bài tập 1.** Chạy đoạn code trong mục Build It, sau đó ghi lại thời gian CPU, GPU và tỉ lệ tăng tốc. Hãy chạy lần thứ hai; lần chạy GPU đầu tiên luôn bao gồm chi phí khởi tạo context (warmup), nên các con số ở lần chạy thứ hai mới là con số chuẩn để so sánh.

**Bài tập 2.** Trên Colab, chọn Runtime > Change runtime type > GPU, sau đó dán đoạn mã benchmark vào ô code cell. Hãy chạy `!nvidia-smi` ở ô trước đó; terminal phải hiển thị tên GPU trước khi kết quả benchmark có ý nghĩa.

**Bài tập 3.** Đọc tổng dung lượng bộ nhớ bằng lệnh `torch.cuda.get_device_properties(0).total_memory / 1e9`. Chia dung lượng bộ nhớ (tính theo byte) cho 2 để ra số lượng tham số tối đa ở định dạng fp16, sau đó trừ hao đi một khoảng an toàn (margin) vì các giá trị trung gian (activations) và ngữ cảnh CUDA cũng tiêu tốn bộ nhớ.

## Thuật ngữ quan trọng (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|---|---|---|
| CUDA | "Lập trình GPU" | Nền tảng điện toán song song của NVIDIA cho phép chạy code trên GPU |
| VRAM | "Bộ nhớ GPU" | Bộ nhớ RAM đồ họa trên GPU, tách biệt với RAM hệ thống. Giới hạn kích thước mô hình có thể nạp |
| fp16 | "Độ chính xác bán phần" | Định dạng số thực 16-bit, tiêu tốn một nửa dung lượng so với fp32 với độ suy giảm chính xác rất nhỏ |
| Tensor Core | "Phần cứng ma trận nhanh" | Các nhân GPU chuyên dụng cho phép tính nhân ma trận, nhanh gấp 4-8 lần nhân thông thường |
