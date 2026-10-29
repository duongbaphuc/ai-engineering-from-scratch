# Bộ đệm KV, FlashAttention & Tối ưu hóa suy luận (KV Cache, Flash Attention & Inference Optimization)

> Huấn luyện (training) mang tính song song và bị giới hạn bởi năng lực tính toán (FLOP-bound). Suy luận (inference) mang tính tuần tự và bị giới hạn bởi băng thông bộ nhớ (memory-bound). Nút thắt cổ chai khác biệt đòi hỏi những giải pháp kỹ thuật khác biệt.

**Phân loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 02 (Cơ chế tự chú ý), Giai đoạn 7 · 05 (Transformer hoàn chỉnh), Giai đoạn 7 · 07 (GPT)  
**Thời lượng:** ~75 phút

## Vấn đề đặt ra

Một bộ giải mã tự hồi quy (autoregressive decoder) ngây thơ (naive) phải tốn chi phí $O(N^2)$ để sinh ra $N$ token: tại mỗi bước, nó tính toán lại toàn bộ cơ chế chú ý trên toàn bộ tiền tố (prefix) trước đó. Với một phản hồi dài 4K token, con số này tương đương 16 triệu phép tính chú ý, hầu hết trong số đó là dư thừa. Mọi trạng thái ẩn (hidden state) của một token tiền tố đều mang tính tất định (deterministic) ngay khi được tính toán — bạn chỉ cần chạy truy vấn (query) của token mới đối chiếu với các khóa (keys) và giá trị (values) đã được lưu vào bộ đệm (cache) từ trước.

Hơn thế nữa, bản thân cơ chế chú ý phải luân chuyển một lượng dữ liệu khổng lồ. Cơ chế chú ý tiêu chuẩn hiện thực hóa (materialize) một ma trận điểm số kích thước $N \times N$, đầu ra softmax kích thước $N \times d$, đầu ra cuối cùng kích thước $N \times d$ — quá nhiều thao tác đọc và ghi vào bộ nhớ băng thông cao (High Bandwidth Memory - HBM). Với $N \ge 2\text{K}$, cơ chế chú ý chạm ngưỡng giới hạn bộ nhớ (memory-bound) trước khi chạm ngưỡng giới hạn tính toán (FLOP-bound). Các kernel chú ý kinh điển làm lãng phí năng lực GPU hiện đại từ 4 đến 10 lần.

Hai kỹ thuật tối ưu hóa then chốt, đều xuất phát từ Dao và các cộng sự, đã đưa tốc độ suy luận của các mô hình tiên phong từ mức "chậm chạp" lên "cực nhanh":

1. **Bộ đệm KV (KV cache).** Lưu trữ các vector K và V của mọi token tiền tố. Phép chú ý của mỗi token mới chỉ là một phép truy vấn đơn lẻ trên các khóa đã lưu đệm. Chi phí suy luận giảm từ $O(N^2)$ xuống $O(N)$ trên mỗi bước sinh token.
2. **FlashAttention.** Chia nhỏ (tile) phép tính chú ý thành các khối sao cho toàn bộ ma trận $N \times N$ không bao giờ phải ghi trực tiếp xuống HBM. Toàn bộ phép tính softmax + nhân ma trận (matmul) diễn ra trọn vẹn trong bộ nhớ tĩnh SRAM trên chip. Tốc độ thực thi thực tế (wall-clock speedup) tăng từ 2–4 lần trên GPU A100; tăng 5–10 lần trên H100 với định dạng FP8.

Đến năm 2026, cả hai kỹ thuật này đã trở thành chuẩn mực phổ quát. Mọi hệ thống suy luận trong môi trường sản xuất (vLLM, TensorRT-LLM, SGLang, llama.cpp) đều mặc định áp dụng chúng. Mọi mô hình tiên phong khi phát hành đều được kích hoạt sẵn FlashAttention.

## Khái niệm cốt lõi

![Sự gia tăng bộ nhớ của KV cache và kỹ thuật chia khối trong FlashAttention](../assets/kv-cache-flash-attn.svg)

### Công thức tính dung lượng bộ đệm KV (KV cache math)

Tính trên mỗi tầng giải mã, mỗi token, trên toàn bộ các đầu KV:

```text
bytes_per_token_per_layer = 2 * n_kv_heads * d_head * dtype_size
                          ^
                          K và V
```

Đối với mô hình 7B có 32 tầng, 32 đầu (tất cả đều dùng cho K và V), d_head=128, kiểu dữ liệu fp16:

```text
per token per layer = 2 * 32 * 128 * 2 = 16384 bytes (16 KiB)
per token (32 layers) = 512 KiB
per 32K context = 16 GiB
```

Đối với Llama 3 70B (80 tầng, d_head=128, cơ chế chú ý theo nhóm truy vấn GQA với 8 đầu KV):

```text
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KiB)
per 32K context = 10 GiB
```

Lượng bộ nhớ 10 GiB tại ngữ cảnh 32K sẽ tăng vọt lên 40 GiB tại 128K, chiếm trọn vẹn dung lượng bộ nhớ của một GPU A100 40 GB chỉ riêng cho bộ đệm KV ở kích thước batch bằng 1.

**GQA là bước đột phá cứu rỗi dung lượng KV cache.** Chú ý đa đầu (Multi-Head Attention - MHA) với 64 đầu KV sẽ có kích thước lớn gấp 8 lần, tương đương khoảng 80 GiB tại 32K token. Cơ chế Chú ý ẩn đa chiều (Multi-head Latent Attention - MLA) thậm chí còn nén sâu hơn nữa.

Kéo các kích thước và quan sát sự thay đổi dung lượng cache. Tăng độ dài chuỗi hoặc kích thước batch và xem bộ đệm vượt qua giới hạn của một GPU nhanh đến mức nào:

````figure
kv-cache-sizer
````

### FlashAttention — Kỹ thuật chia khối trên bộ nhớ (The tiling trick)

Cơ chế chú ý tiêu chuẩn:

```text
S = Q @ K^T          (Đọc HBM, N×N, Ghi HBM)
P = softmax(S)       (Đọc HBM, Ghi HBM)
O = P @ V            (Đọc HBM, Ghi HBM)
```

Ba vòng luân chuyển dữ liệu qua HBM (round trips). Trên GPU H100, băng thông HBM là 3 TB/s; trong khi băng thông SRAM lên tới 30 TB/s. Mỗi lần phải truy xuất HBM là một lần tốc độ bị chậm đi 10 lần so với việc giữ trọn dữ liệu trên chip.

FlashAttention:

```text
for each block of Q (kích thước khối tile ~128 × 128):
    tải Q_tile vào SRAM
    for each block of K, V:
        tải K_tile, V_tile vào SRAM
        tính S_tile = Q_tile @ K_tile^T     (SRAM)
        tích lũy softmax động               (SRAM)
        tích lũy vào O_tile                 (SRAM)
    ghi O_tile xuống HBM
```

Chỉ một lần luân chuyển HBM cho mỗi khối tile. Tổng dung lượng bộ nhớ yêu cầu giảm từ $O(N^2)$ xuống $O(N)$. Lượt lan truyền ngược (backward pass) tính toán lại một số giá trị từ lượt lan truyền xuôi thay vì lưu trữ chúng — mang lại thêm một lợi thế lớn về tiết kiệm bộ nhớ.

**Bí quyết tính toán số học.** Softmax lũy tiến duy trì bộ giá trị `(max, sum)` xuyên suốt các khối tile, nhờ đó việc chuẩn hóa cuối cùng đạt độ chính xác tuyệt đối. Đây không phải là một phép xấp xỉ — FlashAttention cho ra kết quả chính xác từng bit (bit-identical) so với cơ chế chú ý tiêu chuẩn (ngoại trừ sai số không giao hoán của phép toán dấu phẩy động fp16).

**Quá trình tiến hóa qua các phiên bản:**

| Phiên bản | Năm | Thay đổi cốt lõi | Mức tăng tốc trên phần cứng tham chiếu |
|-----------|-----|------------------|----------------------------------------|
| Flash 1 | 2022 | Kernel chia khối chạy trong SRAM | 2× trên A100 |
| Flash 2 | 2023 | Tối ưu song song hóa, thứ tự ưu tiên causal | 3× trên A100 |
| Flash 3 | 2024 | Tận dụng cơ chế bất đồng bộ của Hopper, hỗ trợ FP8 | 1.5–2× trên H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Đường ống xử lý 5 giai đoạn trên Blackwell, hàm exp2 tối ưu bằng phần mềm | Ưu tiên suy luận trước (ban đầu chỉ hỗ trợ forward pass) |

Flash 4 ban đầu khi ra mắt chỉ hỗ trợ lượt lan truyền xuôi (forward-pass only). Huấn luyện vẫn chủ yếu dùng Flash 3. Khả năng hỗ trợ GQA và độ dài chuỗi biến thiên (varlen) cho Flash 4 dự kiến hoàn thiện vào giữa năm 2026.

### Giải mã suy đoán (Speculative decoding) — Bước đột phá độ trễ khác

Mô hình nhỏ (draft model) chi phí thấp đề xuất $N$ token. Mô hình lớn (verifier) xác thực đồng thời cả $N$ token trong một lượt lan truyền song song. Nếu quá trình xác thực chấp nhận $k$ token, bạn chỉ phải trả chi phí 1 lượt forward của mô hình lớn để tạo ra $k$ token. Thông thường $k=3–5$ đối với mã nguồn và văn bản thông thường.

Các chuẩn mặc định năm 2026:
- **EAGLE 2 / Medusa.** Các đầu sinh nháp tích hợp sẵn (integrated draft heads) chia sẻ trực tiếp các trạng thái ẩn của mô hình xác thực. Tăng tốc từ 2–3 lần mà không làm suy giảm chất lượng.
- **Giải mã suy đoán với mô hình nháp riêng biệt.** Tăng tốc 2–4 lần trên phần cứng phổ thông.
- **Lookahead decoding.** Lặp Jacobi; không cần mô hình nháp. Dành cho các trường hợp đặc thù nhưng hoàn toàn miễn phí chi phí mô hình phụ.

### Gom lô liên tục (Continuous batching)

Xử lý theo lô kinh điển: đợi chuỗi dài nhất kết thúc rồi mới bắt đầu lô mới. Gây lãng phí tài nguyên GPU khi các câu trả lời ngắn hoàn thành sớm.

Gom lô liên tục (lần đầu tiên triển khai trong Orca, hiện có mặt trong vLLM, TensorRT-LLM, SGLang): hoán đổi ngay các yêu cầu mới vào lô khi các yêu cầu cũ hoàn tất. Thông lượng (throughput) tăng từ 5–10 lần đối với khối lượng công việc trò chuyện (chat) điển hình.

### PagedAttention — Quản lý bộ đệm KV như bộ nhớ ảo

Tính năng nổi bật làm nên tên tuổi của vLLM. Bộ đệm KV được cấp phát theo các khối cố định 16 token; một bảng phân trang (page table) ánh xạ các vị trí logic sang các khối vật lý. Kỹ thuật này cho phép chia sẻ KV giữa các chuỗi mẫu song song (tìm kiếm chùm - beam search, lấy mẫu song song), lưu đệm nhanh tiền tố (prompt caching) và chống phân mảnh bộ nhớ. Tăng gấp 4 lần thông lượng so với cấp phát bộ nhớ liền kề (contiguous allocation) ngây thơ.

````figure
flash-attention-memory
````

## Xây dựng thực tế

````figure
iso-machine {"machine":"kv-cache"}
````

Xem file `code/main.py`. Chúng ta sẽ cài đặt:

1. Một bộ giải mã tăng dần ngây thơ độ phức tạp $O(N^2)$.
2. Một bộ giải mã dùng bộ đệm KV độ phức tạp $O(N)$.
3. Thuật toán softmax chia khối mô phỏng running-max của FlashAttention.

### Bước 1: Bộ đệm KV

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Rất trực quan: liên tục nối dài các vector K, V của từng token vào danh sách phân theo từng tầng và từng đầu chú ý.

### Bước 2: Softmax chia khối

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Phép tính softmax(qK^T)V kiểu FlashAttention với max/sum lũy tiến."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Kết quả đầu ra giống hệt từng bit so với phép tính `softmax(qK) V` toàn bộ trong một lượt, nhưng tại mỗi thời điểm, tập dữ liệu làm việc (working set) chỉ là một khối kích thước `tile × d_head` thay vì toàn bộ `N × d_head`.

### Bước 3: So sánh giải mã ngây thơ và giải mã dùng bộ đệm khi sinh 100 token

Đếm số lượng phép tính chú ý. Cách tiếp cận ngây thơ: $O(N^2) = 5.050$. Dùng bộ đệm cache: $O(N) = 100$. Mã nguồn sẽ in ra cả hai con số này để đối chứng.

## Ứng dụng thực tế

```python
# Thư viện transformers của HuggingFace tự động kích hoạt KV cache trên hàm generate() của mô hình chỉ có decoder.
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # sử dụng FA3 nếu dùng phần cứng Hopper
    torch_dtype="bfloat16",
)
# Hàm generate() tự động tận dụng KV cache
```

Triển khai sản xuất với vLLM:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Lưu đệm tiền tố (prefix caching) xuyên suốt nhiều yêu cầu là một cải tiến lớn trong năm 2026 — cùng một câu lệnh hệ thống (system prompt), các ví dụ few-shot hoặc văn bản ngữ cảnh dài có thể tái sử dụng dữ liệu KV qua nhiều lệnh gọi API khác nhau. Đối với các tác vụ agent với các prompt công cụ lặp đi lặp lại, prefix caching mang lại mức tăng thông lượng thường thấy lên tới 5 lần.

## Đóng gói ứng dụng

Xem file `outputs/skill-inference-optimizer.md`. Kỹ năng này hỗ trợ lựa chọn cơ chế chú ý phù hợp, chiến lược lưu đệm KV, định dạng lượng tử hóa (quantization) và phương pháp giải mã suy đoán cho một hệ thống triển khai suy luận mới.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Xác nhận rằng bộ giải mã ngây thơ và bộ giải mã có cache tạo ra kết quả giống hệt nhau; quan sát sự chênh lệch về số lượng phép tính.
2. **Trung bình.** Cài đặt cơ chế lưu đệm tiền tố (prefix caching): cho trước một prompt $P$ và một số câu hoàn thiện, chạy một lượt forward trên $P$ để lấp đầy KV cache, sau đó phân nhánh cho từng câu sinh tiếp theo. Đo lường tốc độ tăng tốc so với việc mã hóa lại $P$ cho mỗi chuỗi.
3. **Khó.** Cài đặt một mô hình PagedAttention đơn giản: lưu bộ đệm KV theo các khối cố định 16 token với một danh sách khối tự do (free-list). Khi một chuỗi hoàn tất, trả lại các khối nhớ của nó về vùng bộ nhớ chung (pool). Mô phỏng 1.000 lượt hoàn thành hội thoại với độ dài khác nhau. So sánh mức độ phân mảnh bộ nhớ so với cấp phát bộ nhớ liền kề.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Bộ đệm KV (KV cache) | "Bí quyết giúp giải mã nhanh" | Lưu trữ K và V từ mọi token tiền tố; truy vấn mới chỉ cần chú ý đến chúng thay vì tính lại từ đầu. |
| HBM | "Bộ nhớ chính của GPU" | Bộ nhớ băng thông cao (High Bandwidth Memory); 80 GB trên H100, 192 GB trên B200. Băng thông ~3 TB/s. |
| SRAM | "Bộ nhớ trên chip" | Bộ nhớ siêu nhanh trên từng bộ đa xử lý dòng (SM), ~256 KB mỗi SM trên H100. Băng thông ~30 TB/s. |
| FlashAttention | "Kernel chú ý chia khối" | Tính toán cơ chế chú ý mà không cần hiện thực hóa ma trận $N \times N$ trong HBM. |
| Gom lô liên tục (Continuous batching) | "Gom lô không chờ đợi" | Hoán đổi các chuỗi đã xử lý xong ra ngoài và đưa chuỗi mới vào mà không cần đợi xả hết lô. |
| PagedAttention | "Tính năng đỉnh cao của vLLM" | Bộ đệm KV được cấp phát theo khối cố định với bảng phân trang; triệt tiêu phân mảnh bộ nhớ. |
| Lưu đệm tiền tố (Prefix caching) | "Tái sử dụng prompt dài" | Lưu cache KV cho một tiền tố dùng chung giữa các yêu cầu; tiết kiệm chi phí khổng lồ cho các tác vụ agent. |
| Giải mã suy đoán (Speculative decoding) | "Đề xuất nháp + xác thực" | Mô hình nhỏ sinh nhanh các token nháp; mô hình lớn xác thực đồng thời $k$ token trong một lượt forward duy nhất. |

## Tài liệu tham khảo

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Đường ống 5 giai đoạn trên Blackwell và kỹ thuật exp2 bằng phần mềm; xem thêm ghi chú trong README của repo về các giới hạn forward-only ban đầu.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — bài báo gốc về vLLM.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — giải mã suy đoán.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — bài báo EAGLE-1/2 về phương pháp sinh nháp tích hợp sẵn.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — kỹ thuật Medusa được nhắc đến song hành cùng EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — tài liệu chuẩn phân tích sâu về thiết kế khối 16 token và bảng trang.
