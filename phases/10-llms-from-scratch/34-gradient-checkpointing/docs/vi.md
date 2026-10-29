# Lưu điểm kiểm tra gradient và tính toán lại kích hoạt (Gradient Checkpointing and Activation Recomputation)

> Lan truyền ngược (Backpropagation) giữ lại mọi kích hoạt trung gian (intermediate activations). Ở quy mô 70 tỷ tham số và ngữ cảnh 128K, lượng kích hoạt này lên tới 3 TB trên mỗi rank tính toán. Lưu điểm kiểm tra (checkpointing) đánh đổi năng lực tính toán FLOP lấy bộ nhớ: tính toán lại (recompute) thay vì lưu trữ sẵn. Câu hỏi đặt ra là nên lược bỏ những phân đoạn nào, và câu trả lời chắc chắn không phải là "lược bỏ tất cả".

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## Vấn đề đặt ra

Quá trình huấn luyện một mô hình transformer phải lưu trữ cho mỗi tầng các đầu vào của mọi toán tử cần vi phân trong lượt truyền ngược: đầu vào attention, các phép chiếu Q/K/V, đầu ra của hàm softmax, đầu vào của mạng FFN, đầu ra của các tầng chuẩn hóa và luồng phần dư (residual stream). Với một tầng có kích thước ẩn $d$, độ dài chuỗi $L$, kích thước lô $B$, khối lượng này rơi vào khoảng $12 \times B \times L \times d$ số thực trên mỗi tầng.

Với $d = 8192, L = 8192, B = 1$, con số này tương đương khoảng 805 triệu giá trị, tức 1,6 GB/tầng ở định dạng BF16. Một mô hình 64 tầng sẽ tiêu tốn khoảng 103 GB bộ nhớ kích hoạt, và đó là còn chưa nhân với kích thước micro-batch, chưa tính các giá trị trung gian của attention-softmax ($L^2$ trên mỗi đầu chú ý), và chưa tính đến các bản sao bộ nhớ phục vụ song song hóa tensor (tensor parallelism).

Hóa đơn chi phí từ hai phía: Trọng số BF16 cộng với trạng thái bộ tối ưu hóa (optimizer state) có thể vừa vặn trong 80GB VRAM, nhưng bộ nhớ kích hoạt sẽ đẩy bạn vượt ngưỡng tràn bộ nhớ (OOM). Lưu điểm kiểm tra gradient (còn gọi là tái tính toán kích hoạt - activation recomputation) là giải pháp khắc phục tiêu chuẩn: Lược bỏ phần lớn các kích hoạt; thực hiện lại lượt truyền xuôi trong khi truyền ngược để tái tạo lại chúng khi cần. Chi phí: tiêu tốn thêm lượng FLOP tính toán. Lợi ích: bộ nhớ giảm xuống theo tỷ lệ giữa số phân đoạn điểm kiểm tra so với tổng số tầng.

Nếu làm một cách ngây thơ, việc lưu điểm kiểm tra sẽ tiêu tốn thêm khoảng 33% số FLOP truyền xuôi trên mỗi bước huấn luyện. Nếu làm một cách khéo léo — lưu điểm kiểm tra có chọn lọc theo phương pháp "lựa chọn thông minh" của Korthikanti et al. — bạn sẽ tiết kiệm được gấp 5 lần bộ nhớ với chi phí FLOP phụ trội chỉ dưới 5%. Và với các phép nhân ma trận FP8, kỹ thuật offload của FSDP, cùng MoE song song chuyên gia, điều này mang ý nghĩa sống còn: bạn không thể chi trả lãng phí cho cả bộ nhớ lẫn năng lực tính toán.

## Khái niệm

### Lượt truyền ngược thực sự cần những gì

Công thức cơ bản: `output = layer(input)`. Lượt truyền ngược cần tính toán `grad_input` và `grad_params`. Để tính chúng, hệ thống cần:

- `input` (để tính `grad_params = input.T @ grad_output` cho các tầng tuyến tính)
- Một số giá trị đạo hàm kích hoạt trung gian (đạo hàm của ReLU/GELU/softmax phụ thuộc vào chính giá trị kích hoạt đó)

Lượt truyền xuôi tự động lưu các giá trị này trong đồ thị đạo hàm tự động (autograd graph). Mọi lệnh `tensor.retain_grad()` và mọi toán tử cần đầu vào của nó đều sẽ giữ lại một tham chiếu trong bộ nhớ.

### Lưu điểm kiểm tra toàn phần ngây thơ (Naive Full Checkpointing)

Chia mạng nơ-ron thành $N$ phân đoạn. Trong lượt truyền xuôi, chỉ lưu duy nhất *đầu vào* của từng phân đoạn. Khi lượt truyền ngược cần các giá trị trung gian, chạy lại lượt truyền xuôi của phân đoạn đó để hiện thực hóa lại (materialize) các kích hoạt, sau đó mới tính vi phân.

Ví dụ: Mô hình transformer 32 tầng chia thành 32 phân đoạn, mỗi phân đoạn 1 tầng.

- Bộ nhớ: Chỉ lưu 32 tensor đầu vào của tầng (rất nhỏ) thay vì $32 \times (\text{dung lượng kích hoạt mỗi tầng})$ (khổng lồ).
- Tính toán phụ trội: Thêm 1 lượt truyền xuôi cho mỗi phân đoạn, tức là tốn thêm khoảng 33% tổng số FLOP truyền xuôi (vì truyền ngược tốn gấp đôi truyền xuôi, nên một bước đầy đủ chuyển từ $1 + 2 = 3$ đơn vị thành $1 + 1 + 2 = 4$ đơn vị).

Đây là công thức ban đầu từ Chen et al. 2016: đặt một điểm kiểm tra sau mỗi $\sqrt{L}$ tầng để cân bằng giữa bộ nhớ và chi phí tính toán. Với $L = 64$, con số tối ưu là 8 điểm kiểm tra.

### Lưu điểm kiểm tra chọn lọc (Korthikanti 2022)

Không phải mọi kích hoạt đều tốn kém bộ nhớ như nhau. Đầu ra softmax của cơ chế attention có kích thước $B \times L \times L \times \text{heads}$ và tăng *theo hàm bậc hai* theo độ dài chuỗi. Trong khi đó, kích hoạt ẩn của mạng FFN là $B \times L \times 4d$ và chỉ tăng tuyến tính. Đối với các chuỗi dài, softmax chiếm tỷ trọng áp đảo.

Lưu điểm kiểm tra chọn lọc giữ lại các kích hoạt có chi phí lưu trữ rẻ (các phép chiếu tuyến tính, luồng phần dư) và chỉ tính toán lại các kích hoạt đắt đỏ (cơ chế attention). Bạn chỉ phải trả một lượng FLOP tối thiểu để tính toán lại nhưng tiết kiệm được không gian bộ nhớ quy mô $O(L^2)$.

Megatron-Core cài đặt kỹ thuật này dưới tên gọi tái tính toán kích hoạt "selective". Đây là chuẩn mực được sử dụng trong hầu hết các đợt huấn luyện mô hình tiên phong từ năm 2024 trở đi.

### Đẩy dữ liệu sang bộ nhớ ngoài (Offload)

Một giải pháp thay thế cho việc tính toán lại: chuyển các kích hoạt sang bộ nhớ RAM của CPU trong khoảng thời gian giữa lượt truyền xuôi và truyền ngược. Phương pháp này đòi hỏi băng thông PCIe; nó phát huy hiệu quả khi băng thông nhàn rỗi vượt trội hơn chi phí tính toán tái hiện kích hoạt. Các chiến lược kết hợp rất phổ biến: lưu điểm kiểm tra một số tầng, và offload một số tầng khác.

FSDP2 hỗ trợ offload như một tùy chọn hạng nhất. Offload tỏa sáng khi GPU bị nghẽn bộ nhớ nghiêm trọng nhưng đường truyền CPU-GPU vẫn còn nhiều khoảng trống.

### Mô hình chi phí tính toán lại

Lượng FLOP trên mỗi bước khi lưu điểm kiểm tra ngây thơ sau mỗi $k$ tầng trong tổng số $L$ tầng:

```text
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # một lượt truyền xuôi bổ sung cho mỗi tầng trong phân đoạn
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Với lưu điểm kiểm tra chọn lọc, bạn chỉ tính toán lại kernel attention chứ không tính lại toàn bộ tầng:

```text
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Mô hình tiết kiệm bộ nhớ

Dung lượng kích hoạt trên mỗi tầng: $A$. Với $L$ tầng, tổng bộ nhớ kích hoạt: $L \times A$.

Lưu điểm kiểm tra toàn phần (kích thước phân đoạn bằng 1): chỉ lưu $L \times \text{input\_volume}$ ($\approx L \times \frac{1}{10} A$ đối với một transformer tiêu chuẩn). Tiết kiệm khoảng $9 \times L \times A \times \frac{1}{10}$.

Lưu điểm kiểm tra sau mỗi $k$ tầng: lưu $\frac{L}{k} \times A$ cộng với lượng kích hoạt của $k-1$ tầng bên trong phân đoạn đang hoạt động.

Tại $k = \sqrt{L}$, chi phí bộ nhớ và chi phí tính toán lại đều tỷ lệ thuận với $\sqrt{L}$ — đây là điểm đánh đổi tối ưu đối với các tầng có chi phí đồng đều.

### Khi nào không nên lưu điểm kiểm tra

- Các tầng sâu nhất bên trong một chặng của đường ống song song (pipeline stage) đang trong quá trình thực thi. Dù sao chúng cũng phải hoàn thành ngay lập tức.
- Tầng đầu tiên và tầng cuối cùng nếu chúng chiếm phần lớn thời gian tính toán của chặng (trường hợp này hiếm gặp trong transformer).
- Các kernel attention đã sử dụng sẵn FlashAttention — bản thân FlashAttention đã tự động tính toán lại softmax cực nhanh trong SRAM, nên việc bổ sung lưu điểm kiểm tra ở cấp độ tầng mang lại rất ít lợi ích bổ sung.

### Các mẫu hình cài đặt

1. **Hàm bọc (Function wrapper):** Bọc một phân đoạn trong `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch chỉ lưu `input`, và tính toán lại mọi thứ khác trong lượt truyền ngược.

2. **Dựa trên Decorator:** Đánh dấu các tầng có thể lưu điểm kiểm tra; bộ huấn luyện sẽ quyết định lúc nạp cấu hình xem phân đoạn nào sẽ được bọc lại.

3. **Tự viết mã tính toán lại tường minh (Manual explicit recompute):** Tự viết lượt truyền ngược, gọi một hàm `recompute_forward` tùy chỉnh để lặp lại lượt truyền xuôi từ đầu vào đã lưu.

Cả ba mẫu hình đều đem lại kết quả hàm số như nhau. Sử dụng hàm bọc là quy ước phổ biến nhất.

### Tương tác với TP / PP / FP8

- **Song song hóa Tensor (TP):** Các đầu vào điểm kiểm tra phải được thu thập (gather) hoặc phân tán lại (scatter) khi tính toán lại; cần xử lý cẩn thận chi phí truyền thông.
- **Song song hóa Đường ống (PP):** Mẫu hình điển hình là lưu điểm kiểm tra lượt truyền xuôi của từng chặng đường ống để các micro-batch truyền ngược có thể tái sử dụng bộ nhớ kích hoạt.
- **Tính toán lại ở FP8:** Lịch sử giá trị cực đại (amax histories) được cập nhật trong quá trình tính toán lại phải khớp hoàn toàn với lượt truyền xuôi ban đầu, nếu không thang đo FP8 sẽ bị trôi dạt sai số. Hầu hết các framework đều chụp nhanh (snapshot) thang đo này.

```figure
activation-recompute
```

## Xây dựng hệ thống

### Bước 1: Mô hình đồ chơi với các phân đoạn

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Bước 2: Lượt truyền ngược ngây thơ cần toàn bộ kích hoạt

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Bước 3: Bộ nhớ lưu điểm kiểm tra sau mỗi k tầng

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Bước 4: Mô hình chi phí

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Bước 5: Bộ ước tính bộ nhớ

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Bước 6: Kích thước phân đoạn tối ưu

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Bước 7: Quyết định lưu điểm kiểm tra chọn lọc

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Sử dụng trong thực tế

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — hàm bọc chuẩn mực trong PyTorch. Bọc một hàm tính toán; chỉ lưu lại đầu vào và tự động tính toán lại trong lượt truyền ngược.
- **Tái tính toán kích hoạt trong Megatron-Core**: hỗ trợ các chế độ `selective`, `full`, và `block`. Đây là tiêu chuẩn trong các đợt huấn luyện mô hình tiên phong từ năm 2024.
- **Offload trong FSDP2**: `module.to_empty(device="cpu")` kết hợp với `offload_policy` trong FSDP2 giúp phân mảnh các kích hoạt sang RAM của CPU thay vì phải tính toán lại.
- **DeepSpeed ZeRO-Offload**: Offload sang CPU cho trạng thái bộ tối ưu hóa và bộ nhớ kích hoạt, bổ trợ hiệu quả cho việc lưu điểm kiểm tra.

## Đóng gói sản phẩm

Bài học này xây dựng `outputs/prompt-activation-recompute-policy.md` — một câu lệnh mẫu (prompt) tiếp nhận cấu hình mô hình của bạn (số tầng, chiều ẩn, độ dài chuỗi, kích thước lô) cùng dung lượng VRAM khả dụng của GPU để xuất ra chính sách tính toán lại tối ưu trên từng tầng (none / selective / full / offload).

## Bài tập

1. Xác minh tính đúng đắn. Chạy `model_forward` + `model_backward` (lưu đầy đủ kích hoạt) so sánh với `model_forward_checkpointed` + `model_backward_checkpointed` (lưu theo phân đoạn). Gradient của các tham số phải hoàn toàn giống hệt nhau tới độ chính xác của máy tính.

2. Quét kích thước phân đoạn $k$ từ 1 đến $L$. Vẽ biểu đồ chi phí phụ trội FLOP và dung lượng bộ nhớ. Xác định điểm uốn (knee) của đường cong.

3. Cài đặt lưu điểm kiểm tra chọn lọc: lưu đầu vào của mô-đun attention nhưng bỏ qua các kích hoạt trung gian của nó. Đo lường chi phí FLOP phụ trội so với việc lưu điểm kiểm tra toàn bộ tầng cho một mô hình 32 tầng ở độ dài chuỗi `seq=8192`.

4. Bổ sung cơ chế offload. Lưu đầu vào của phân đoạn vào một "bộ đệm CPU" mô phỏng (một danh sách riêng biệt). Đo lường "băng thông PCIe" dưới dạng số byte trên đơn vị thời gian và tìm điểm hòa vốn giữa phương pháp offload và phương pháp tính toán lại.

5. Đo lường hiệu năng thực tế trên một mô hình transformer bằng PyTorch có và không có `torch.utils.checkpoint`. Đo lường mức sử dụng bộ nhớ (thông qua `torch.cuda.max_memory_allocated`) và thời gian thực thi của mỗi bước.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Lưu điểm kiểm tra gradient (Gradient checkpointing) | "Tiết kiệm bộ nhớ bằng cách chạy lại forward" | Chỉ lưu các đầu vào của phân đoạn; tính toán lại các kích hoạt trung gian trong lượt truyền ngược để lấy tensor phục vụ tính gradient |
| Tái tính toán kích hoạt (Activation recomputation) | "Tên gọi khác của checkpointing" | Tên gọi chuẩn trong cộng đồng tính toán hiệu năng cao (HPC) cho cùng một kỹ thuật |
| Kích thước phân đoạn (k) | "Bao nhiêu tầng mỗi checkpoint" | Số lượng tầng có các kích hoạt trung gian bị lược bỏ và được tái hiện đồng thời cùng nhau |
| Lưu điểm kiểm tra chọn lọc (Selective checkpointing) | "Mẹo của Korthikanti" | Chỉ tính toán lại các kích hoạt tốn kém bộ nhớ lưu trữ (attention softmax); giữ lại các kích hoạt rẻ |
| Lưu điểm kiểm tra toàn phần (Full checkpointing) | "Phiên bản ngây thơ" | Tính toán lại toàn bộ kích hoạt trung gian của mọi tầng trong từng phân đoạn |
| Lưu điểm kiểm tra theo khối (Block checkpointing) | "Mức thô" | Lưu điểm kiểm tra cho toàn bộ khối transformer; mức độ chi tiết lớn nhất |
| Chi phí phụ trội FLOP (FLOP overhead) | "Thuế tính toán" | Lượng FLOP phát sinh thêm trên mỗi bước = (FLOP tính lại) / (FLOP xuôi + FLOP ngược); ngây thơ tốn ~33%, chọn lọc tốn ~5% |
| Đẩy kích hoạt ra ngoài (Activation offload) | "Chuyển sang CPU" | Đưa các kích hoạt sang bộ nhớ RAM của CPU giữa lượt xuôi và ngược; giải pháp thay thế cho tính toán lại |
| Quy tắc căn bậc hai của L (sqrt-L rule) | "Điểm tối ưu cổ điển" | Đối với các tầng có chi phí tính toán đồng đều, khoảng cách điểm kiểm tra tối ưu là $\sqrt{L}$ tầng |
| Dung lượng Attention-softmax | "Vấn đề bậc hai O(L^2)" | $L^2 \times \text{heads} \times \text{batch}$ số thực; chiếm ưu thế áp đảo bộ nhớ kích hoạt ở các chuỗi ngữ cảnh dài |

## Đọc thêm

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- bài báo gốc chuẩn hóa lý thuyết về lưu điểm kiểm tra gradient
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- kỹ thuật tái tính toán kích hoạt chọn lọc và phân tích chi phí toán học
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- phương pháp tiếp cận bộ nhớ cố định thông qua tái hiện theo chế độ ngược
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- kỹ thuật offload kích hoạt ở quy mô lớn
- [Tài liệu chính thức PyTorch torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html) -- giao diện lập trình chuẩn mực của PyTorch
- [Tài liệu tái tính toán kích hoạt của Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html) -- các chế độ selective, full, và block
