# Các biến thể cơ chế chú ý — Cửa sổ trượt, Thưa thớt, Vi phân (Attention Variants — Sliding Window, Sparse, Differential)

> Cơ chế chú ý đầy đủ (full attention) là một vòng tròn khép kín: mọi token đều nhìn thấy mọi token, và bộ nhớ phải gánh chịu toàn bộ chi phí. Ba biến thể dưới đây bẻ cong hình dạng của vòng tròn đó để thu hồi lại một nửa chi phí tính toán và tài nguyên.

**Phân loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 02 (Cơ chế tự chú ý), Giai đoạn 7 · 03 (Chú ý đa đầu), Giai đoạn 7 · 12 (Bộ đệm KV / FlashAttention)  
**Thời lượng:** ~60 phút

## Vấn đề đặt ra

Cơ chế chú ý đầy đủ (full attention) tiêu tốn $O(N^2)$ bộ nhớ và $O(N^2)$ chi phí tính toán theo chiều dài chuỗi $N$. Đối với mô hình Llama 3 70B với ngữ cảnh 128K token, con số này tương đương 16 tỷ phần tử chú ý trên mỗi tầng, nhân với 80 tầng. FlashAttention (Bài 12) giúp ẩn đi $O(N^2)$ bộ nhớ kích hoạt (activation memory) nhưng không thay đổi chi phí số học — mọi token vẫn phải chú ý đến mọi token khác.

Ba nhóm biến thể dưới đây trực tiếp thay đổi cấu trúc hình học (topology) của chính ma trận chú ý:

1. **Chú ý cửa sổ trượt (Sliding Window Attention - SWA).** Mỗi token chỉ chú ý đến một cửa sổ cố định các token lân cận thay vì toàn bộ tiền tố trước đó. Bộ nhớ và chi phí tính toán giảm xuống $O(N \cdot W)$ với $W$ là kích thước cửa sổ. Được áp dụng trong Gemma 2/3, các tầng đầu tiên của Mistral 7B, Phi-3-Long.
2. **Chú ý thưa thớt / theo khối (Sparse / Block Attention).** Chỉ các cặp `(i, j)` được chọn trước mới được tính điểm; các vị trí còn lại bị ép gán trọng số bằng 0. Điển hình như Longformer, BigBird, OpenAI Sparse Transformer.
3. **Chú ý vi phân (Differential Attention).** Tính toán hai bản đồ chú ý với các ma trận chiếu Q/K tách biệt, sau đó lấy bản đồ thứ nhất trừ đi bản đồ thứ hai. Kỹ thuật này triệt tiêu "hố thu chú ý" (attention sink) — hiện tượng làm rò rỉ trọng số vào một vài token đầu tiên một cách vô nghĩa. Mô hình DIFF Transformer của Microsoft (2024).

Các kỹ thuật này hoàn toàn có thể cùng tồn tại. Một mô hình tiên phong năm 2026 thường kết hợp linh hoạt chúng: hầu hết các tầng sử dụng SWA-1024, cứ mỗi tầng thứ 5 lại dùng cơ chế chú ý toàn cục đầy đủ (global full attention), xen kẽ một vài đầu chú ý vi phân để tối ưu khả năng truy xuất thông tin (retrieval). Tỷ lệ 5:1 giữa SWA và chú ý toàn cục của Gemma 3 là cấu hình chuẩn mực kinh điển hiện nay.

## Khái niệm cốt lõi

### Chú ý cửa sổ trượt (Sliding Window Attention - SWA)

Mỗi truy vấn (query) tại vị trí $i$ chỉ chú ý đến các vị trí trong khoảng $[i - W, i]$ (SWA nhân quả - causal SWA) hoặc $[i - W/2, i + W/2]$ (SWA hai chiều - bidirectional). Các token nằm ngoài cửa sổ sẽ nhận giá trị $-\infty$ trong ma trận điểm số.

```text
Cơ chế causal đầy đủ:    Cửa sổ trượt (W=4):
Vị trí 0-7               Vị trí 0-7, W=4
    0 1 2 3 4 5 6 7          0 1 2 3 4 5 6 7
0 | x                  0 |   x
1 | x x                1 |   x x
2 | x x x              2 |   x x x
3 | x x x x            3 |   x x x x
4 | x x x x x          4 |     x x x x
5 | x x x x x x        5 |       x x x x
6 | x x x x x x x      6 |         x x x x
7 | x x x x x x x x    7 |           x x x x
```

Với $N = 8192$ và $W = 1024$, ma trận điểm số chỉ có $1024 \times 8192$ hàng khác không theo kỳ vọng — giảm tới 8 lần.

**Bộ đệm KV thu nhỏ đáng kể nhờ SWA.** Chỉ cần lưu giữ $W$ token K và V gần nhất trên mỗi tầng. Đối với cấu hình kiểu Gemma 3 (cửa sổ 1024, ngữ cảnh 128K), dung lượng bộ đệm KV giảm tới 128 lần.

**Cái giá về mặt chất lượng.** Các mô hình transformer chỉ dùng duy nhất SWA sẽ gặp khó khăn với các tác vụ truy xuất tầm xa (long-range retrieval). Giải pháp khắc phục: đan xen các tầng SWA với các tầng chú ý đầy đủ. Gemma 3 sử dụng tỷ lệ 5:1 giữa SWA và chú ý toàn cục. Mistral 7B sử dụng một ngăn xếp causal SWA trong đó thông tin "chảy xuôi" qua các cửa sổ gối nhau — mỗi tầng mở rộng trường tiếp nhận hiệu dụng (effective receptive field) thêm $W$, và sau $L$ tầng, mô hình có thể chú ý ngược về trước tới $L \times W$ token.

### Chú ý thưa thớt / theo khối (Sparse / Block Attention)

Xác định trước một mẫu hình thưa thớt (sparsity pattern) kích thước $N \times N$. Ba dạng cấu trúc kinh điển:

- **Cục bộ + bước nhảy (Local + strided - Sparse Transformer của OpenAI).** Chú ý đến $W$ token gần nhất cộng với mỗi token cách nhau một khoảng `stride` về trước. Nắm bắt đồng thời thông tin cục bộ và tầm xa với chi phí tính toán $O(N \cdot \sqrt{N})$.
- **Longformer / BigBird.** Cửa sổ cục bộ + một tập hợp nhỏ các token toàn cục (ví dụ: `[CLS]`) chú ý đến tất cả và được tất cả chú ý đến + các liên kết thưa ngẫu nhiên. Mở rộng ngữ cảnh gấp 2 lần trên thực nghiệm với chất lượng tương đương.
- **Chú ý thưa thớt nguyên bản (Native Sparse Attention - DeepSeek, 2025).** Học xem những khối `(Q, K)` nào thực sự quan trọng; bỏ qua các khối bằng không trực tiếp ở cấp độ kernel. Hoàn toàn tương thích với FlashAttention.

Cơ chế chú ý thưa thớt thực chất là câu chuyện về kỹ thuật tối ưu hóa kernel. Về mặt toán học, nó rất đơn giản (áp mặt nạ lên ma trận điểm số); điểm mấu chốt mang lại tốc độ vượt trội là việc không bao giờ phải nạp các phần tử bằng không vào SRAM. FlashAttention-3 và API FlexAttention năm 2026 đã đưa các mẫu hình thưa thớt tùy biến trở thành công dân hạng nhất (first-class citizen) trong PyTorch.

### Chú ý vi phân (Differential Attention - DIFF Transformer, 2024)

Cơ chế chú ý thông thường gặp phải vấn đề "hố thu chú ý" (attention sink): hàm softmax ép buộc tổng mỗi hàng phải bằng 1, do đó các token không có nhu cầu chú ý đến nội dung cụ thể nào sẽ đổ dồn trọng số vào token đầu tiên (hoặc một vài token đầu). Điều này đánh cắp dung lượng biểu đạt đáng lẽ phải dành cho nội dung thực sự hữu ích.

Cơ chế chú ý vi phân giải quyết triệt để vấn đề này bằng cách tính toán **hai** bản đồ chú ý và lấy hiệu số:

```math
\begin{aligned}
A_1 &= \mathop{\mathrm{softmax}}(Q_1 K_1^T / \sqrt{d}) \\
A_2 &= \mathop{\mathrm{softmax}}(Q_2 K_2^T / \sqrt{d}) \\
\text{DiffAttn} &= (A_1 - \lambda \cdot A_2) V
\end{aligned}
```

trong đó $\lambda$ là một đại lượng vô hướng có thể học (thường nằm trong khoảng 0,5–0,8). $A_1$ nắm bắt trọng số nội dung thực tế; $A_2$ ghi lại hiệu ứng hố thu. Phép trừ sẽ triệt tiêu hố thu, tái phân bổ trọng số cho các token thực sự liên quan.

Kết quả công bố (Microsoft 2024): giảm 5–10% độ hỗn loạn (perplexity), tăng 1,5–2 lần chiều dài ngữ cảnh hiệu dụng trên cùng độ dài huấn luyện, đồng thời giúp truy xuất "kim trong bọc" (needle-in-a-haystack) sắc bén hơn rõ rệt.

### So sánh các biến thể

| Biến thể | Tính toán | Bộ đệm KV | Chất lượng so với chú ý đầy đủ | Ứng dụng trong sản xuất |
|----------|-----------|-----------|--------------------------------|-------------------------|
| Chú ý đầy đủ (Full attention) | $O(N^2)$ | $O(N)$ mỗi tầng | Chuẩn đối sánh cơ sở (baseline) | Tầng mặc định của mọi mô hình |
| SWA (cửa sổ 1024) | $O(N \cdot W)$ | $O(W)$ mỗi tầng | -0,1 ppl, rất tốt khi kết hợp tầng toàn cục | Gemma 2/3, Phi-3-Long |
| Thưa thớt cục bộ + bước nhảy | $O(N \cdot \sqrt{N})$ | Hỗn hợp | Tương đương SWA | Sparse Transformer của OpenAI, Longformer |
| BigBird (cục bộ + toàn cục + ngẫu nhiên) | Xấp xỉ $O(N)$ | Hỗn hợp | Ngang ngửa chú ý đầy đủ ở ngữ cảnh gấp đôi | Các mô hình BERT ngữ cảnh dài thời kỳ đầu |
| Native Sparse (DeepSeek-V3.2) | $O(N \cdot \text{tỷ lệ tích cực})$ | $O(N)$ | Chênh lệch trong vòng 0,05 ppl | DeepSeek-V3.2, 2025 |
| Vi phân (Differential) | $O(2 \cdot N^2)$ | $O(2N)$ | Cải thiện -5 đến -10% ppl | DIFF Transformer, các mô hình đầu năm 2026 |

````figure
gqa-kv-sharing
````

## Xây dựng thực tế

Xem file `code/main.py`. Chúng ta sẽ cài đặt một bộ so sánh mặt nạ nhân quả thể hiện trực quan cơ chế chú ý đầy đủ, SWA, cục bộ + bước nhảy và chú ý vi phân đặt cạnh nhau trên một chuỗi đồ chơi mô phỏng.

### Bước 1: Mặt nạ nhân quả đầy đủ (Chuẩn đối sánh)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Chuẩn đối sánh từ Bài 07. Ma trận tam giác dưới; trọng số bằng 0 phía trên đường chéo chính.

### Bước 2: Mặt nạ nhân quả cửa sổ trượt (SWA)

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Một tham số duy nhất — `window`. Khi `window >= n`, ta thu lại cơ chế chú ý nhân quả đầy đủ. Khi `window = 1`, mỗi token chỉ chú ý đến chính nó.

### Bước 3: Mặt nạ thưa thớt cục bộ + bước nhảy

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Một cửa sổ cục bộ dày đặc cộng với việc chú ý đến mỗi token cách nhau `stride` bước ngược về đầu chuỗi. Trường tiếp nhận mở rộng theo hàm log khi bổ sung thêm các tầng.

### Bước 4: Chú ý vi phân (Differential attention)

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Hai lượt tính chú ý, sau đó trừ đi nhau với một hệ số hòa trộn học được. Trong mã nguồn, chúng ta sẽ so sánh bản đồ nhiệt (heatmap) của hố thu chú ý giữa cơ chế đơn và cơ chế vi phân để thấy hố thu bị triệt tiêu hoàn toàn.

### Bước 5: Kích thước bộ đệm KV

In ra kích thước bộ đệm cache trên mỗi tầng tại $N = 131.072$ cho từng biến thể. SWA và các biến thể thưa thớt giảm từ 10 đến 100 lần. Ngược lại, chú ý vi phân tăng gấp đôi. Bạn cần ý thức rõ ràng về cái giá bộ nhớ mà mình phải đánh đổi.

## Ứng dụng thực tế

Các mẫu hình triển khai trong môi trường sản xuất năm 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 kết hợp SWA (cửa sổ=1024) và các tầng toàn cục theo tỷ lệ 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention trong PyTorch 2.5+ hỗ trợ truyền vào một hàm mặt nạ:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Hàm này được biên dịch thành một kernel Triton tùy biến. Đạt tốc độ trong khoảng 10% so với FlashAttention-3 đối với các mẫu hình thông dụng, trong khi hàm mặt nạ chỉ là một hàm Python có thể gọi thông thường.

**Khi nào nên chọn từng phương án:**

- **Chú ý đầy đủ thuần túy (Pure full attention)** — áp dụng cho mọi tầng với ngữ cảnh dưới ~16K, hoặc khi chất lượng truy xuất là yếu tố sống còn.
- **Kết hợp SWA + tầng toàn cục** — ngữ cảnh dài (>32K), khi việc huấn luyện và suy luận bị giới hạn bởi bộ nhớ. Đây là lựa chọn mặc định vào năm 2026 cho ngữ cảnh trên 32K.
- **Chú ý theo khối thưa thớt (Sparse block attention)** — kernel tùy biến, cấu trúc tùy biến. Dành riêng cho các khối lượng công việc chuyên biệt (truy xuất, âm thanh).
- **Chú ý vi phân (Differential attention)** — bất kỳ khối lượng công việc nào mà sự ô nhiễm từ hố thu chú ý gây tổn hại (RAG ngữ cảnh dài, tìm kim trong đáy bể).

## Đóng gói ứng dụng

Xem file `outputs/skill-attention-variant-picker.md`. Kỹ năng này gợi ý cấu trúc cơ chế chú ý phù hợp cho mô hình mới dựa trên độ dài ngữ cảnh mục tiêu, yêu cầu về chất lượng truy xuất và hồ sơ tính toán phục vụ huấn luyện/suy luận.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Xác nhận rằng SWA tại `window=4` gán giá trị 0 cho mọi vị trí nằm ngoài 4 token gần nhất trên mỗi hàng. Xác minh rằng `window=n` tái lập kết quả giống hệt từng bit so với cơ chế chú ý nhân quả đầy đủ.
2. **Trung bình.** Cài đặt cơ chế causal SWA với `window=1024` dựa trên mô hình tổng kết ở Bài 07. Huấn luyện 1.000 bước trên tập dữ liệu tinyshakespeare. Mất mát kiểm định suy giảm bao nhiêu so với chú ý đầy đủ? Mức bộ nhớ đỉnh (peak memory) giảm được bao nhiêu?
3. **Khó.** Cài đặt cấu trúc đan xen tầng theo tỷ lệ 5:1 kiểu Gemma 3 (5 tầng SWA, 1 tầng toàn cục) vào mô hình tổng kết. So sánh hàm mất mát, bộ nhớ và chất lượng sinh văn bản với mô hình thuần SWA và thuần toàn cục ở cùng mức tham số tương đương.
4. **Khó.** Cài đặt cơ chế chú ý vi phân với tham số $\lambda$ có thể học cho từng đầu chú ý. Huấn luyện mô hình trên một tác vụ truy xuất nhân tạo (1 token "kim", 2.000 token gây nhiễu). Đo lường độ chính xác truy xuất so với mô hình chú ý đơn lẻ ở cùng số lượng tham số.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Chú ý cửa sổ trượt (SWA) | "Chú ý cục bộ" | Mỗi truy vấn chỉ chú ý đến $W$ token gần nhất; bộ đệm KV thu gọn về cấp $O(W)$. |
| Trường tiếp nhận hiệu dụng (Effective receptive field) | "Mô hình có thể nhìn xa đến đâu" | Trong một ngăn xếp gồm $L$ tầng SWA với cửa sổ $W$, trường tiếp nhận lên tới $L \times W$ token. |
| Longformer / BigBird | "Cục bộ + toàn cục + ngẫu nhiên" | Các mẫu hình thưa thớt với một vài token toàn cục luôn chú ý đến tất cả; giải pháp ngữ cảnh dài thời kỳ đầu. |
| Chú ý thưa nguyên bản (Native Sparse Attention) | "Kỹ thuật tối ưu kernel của DeepSeek" | Học tính thưa thớt ở cấp độ khối; bỏ qua các khối bằng không ngay trong kernel mà vẫn bảo toàn chất lượng. |
| Chú ý vi phân (Differential attention) | "Hai bản đồ, một bản đồ trừ đi" | DIFF Transformer: lấy bản đồ thứ nhất trừ đi $\lambda$ nhân với bản đồ thứ hai để triệt tiêu hố thu chú ý. |
| Hố thu chú ý (Attention sink) | "Trọng số bị dồn về token 0" | Phép chuẩn hóa softmax ép tổng mỗi hàng bằng 1; các truy vấn không chứa thông tin hữu ích sẽ đổ dồn trọng số vào vị trí 0. |
| FlexAttention | "Mặt nạ viết bằng Python" | API từ PyTorch 2.5+ biên dịch các hàm mặt nạ tùy ý thành kernel tối ưu có hiệu năng tương tự FlashAttention. |
| Kết hợp tầng (Layer type mix) | "Tỷ lệ 5:1 giữa SWA và toàn cục" | Đan xen các tầng chú ý thưa và chú ý đầy đủ để duy trì chất lượng cao với chi phí bộ nhớ thấp hơn. |

## Tài liệu tham khảo

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — bài báo kinh điển về cửa sổ trượt + token toàn cục.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — kết hợp cục bộ + toàn cục + ngẫu nhiên.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — mẫu hình cục bộ + bước nhảy của OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — mô hình kết hợp tỷ lệ 1:1 giữa SWA và toàn cục.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — cấu hình kết hợp tỷ lệ 5:1 với cửa sổ 1024 hiện là chuẩn mực giáo khoa mặc định.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — bài báo gốc về DIFF Transformer.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — cơ chế chú ý thưa có thể học của DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — tài liệu tham khảo API về mô hình mặt nạ dưới dạng hàm gọi được nhắc đến trong bài.
