# Mã hóa vị trí — Sinusoidal, RoPE, ALiBi

> Cơ chế chú ý có tính chất bất biến với hoán vị (permutation-invariant). Hai câu "The cat sat on the mat" và "mat the on sat cat the" sẽ cho ra kết quả hoàn toàn giống nhau nếu không có tín hiệu vị trí. Ba thuật toán sau đây khắc phục vấn đề đó — mỗi thuật toán dựa trên một cách định nghĩa khác nhau về ý nghĩa của "vị trí".

**Phân loại:** Xây dựng thực hành (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 02 (Tự chú ý), Giai đoạn 7 · 03 (Chú ý đa đầu)  
**Thời lượng:** ~45 phút

## Vấn đề đặt ra

Cơ chế tự chú ý tích vô hướng có chia tỷ lệ hoàn toàn "mù" về thứ tự. Ma trận chú ý `softmax(Q K^T / √d) V` được tính toán thuần túy dựa trên điểm tương đồng giữa từng cặp vector. Nếu bạn xáo trộn các hàng của ma trận đầu vào `X`, các hàng tương ứng của ma trận đầu ra cũng chỉ bị xáo trộn theo đúng thứ tự đó. Không có bất kỳ thành phần nào bên trong cơ chế chú ý quan tâm đến vị trí của token trong câu.

Đó không phải là nhược điểm đối với mô hình túi từ (bag-of-words). Nhưng đối với ngôn ngữ tự nhiên, mã nguồn, âm thanh hay video — bất cứ loại dữ liệu nào mà thứ tự quyết định ý nghĩa — sự thiếu hụt này mang tính chí mạng.

Giải pháp đặt ra là tìm cách đưa thông tin vị trí vào các vector nhúng. Trải qua các giai đoạn phát triển, ba giải pháp tiêu biểu đã xuất hiện:

1. **Mã hóa hình sin tuyệt đối (Absolute sinusoidal)** (Vaswani 2017): Cộng trực tiếp hàm `sin/cos` của vị trí vào vector nhúng. Phương pháp này đơn giản, không cần học tham số, nhưng khả năng ngoại suy (extrapolate) rất kém khi vượt quá độ dài chuỗi huấn luyện.
2. **RoPE — Mã hóa vị trí dạng quay (Rotary Position Embeddings)** (Su 2021): Quay các vector Q và K theo một góc tỷ lệ thuận với vị trí. Mã hóa vị trí *tương đối* trực tiếp vào tích vô hướng. Đây là tiêu chuẩn thống trị vào năm 2026.
3. **ALiBi — Chú ý với độ lệch tuyến tính (Attention with Linear Biases)** (Press 2022): Bỏ qua hoàn toàn vector nhúng vị trí; trừ một giá trị phạt tuyến tính trên từng đầu vào điểm số chú ý dựa trên khoảng cách giữa các token. Khả năng ngoại suy độ dài xuất sắc.

Tính đến năm 2026, gần như mọi mô hình nguồn mở hàng đầu đều sử dụng RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Một số ít mô hình ngữ cảnh siêu dài sử dụng ALiBi hoặc các biến thể hiện đại của nó. Mã hóa hình sin tuyệt đối giờ đây chủ yếu mang ý nghĩa lịch sử.

## Khái niệm cốt lõi

![Mã hóa vị trí hình sin tuyệt đối so với phép quay RoPE và độ lệch khoảng cách tuyến tính ALiBi](../assets/positional-encoding.svg)

### Mã hóa hình sin tuyệt đối

Tính toán trước một ma trận cố định `PE` có kích thước `(max_len, d_model)`:

```math
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)
\end{aligned}
```

Sau đó cộng vào vector nhúng: `X' = X + PE[:N]` trước khi đưa vào khối chú ý. Mỗi chiều không gian là một đường hình sin ở một tần số khác nhau. Mô hình học cách giải mã vị trí dựa trên cấu trúc pha dao động. Nhược điểm là thất bại khi vượt quá `max_len`: mô hình không hề biết điều gì xảy ra ở vị trí 2048 nếu trong quá trình huấn luyện nó chỉ thấy các vị trí từ 0 đến 2047.

### RoPE (Rotary Position Embeddings)

Quay trực tiếp các vector Q và K (không can thiệp vào vector nhúng). Đối với mỗi cặp chiều tọa độ $(2i, 2i+1)$:

```math
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
\qquad \theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{base} = 10000 \text{ mặc định}
```

Áp dụng phép quay tương tự cho vector key với vị trí $pos_k$. Khi đó, tích vô hướng $q'_m \cdot k'_n$ trở thành một hàm chỉ phụ thuộc duy nhất vào hiệu số $(m - n)$. Nói cách khác: **điểm số chú ý chỉ phụ thuộc vào khoảng cách tương đối**, mặc dù phép quay ban đầu được tính toán từ các vị trí tuyệt đối. Đây là một cơ chế toán học vô cùng tinh tế.

Mở rộng quy mô RoPE: giá trị `base` có thể được điều chỉnh chia tỷ lệ (các phương pháp NTK-aware, YaRN, LongRoPE) để ngoại suy sang các ngữ cảnh dài hơn nhiều mà không cần huấn luyện lại từ đầu. Llama 3 đã mở rộng ngữ cảnh từ 8K lên 128K token theo cách này.

### ALiBi

Bỏ qua hoàn toàn việc mã hóa vào vector nhúng. Trực tiếp tạo độ lệch (bias) cho điểm số chú ý:

```math
\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|
```

Trong đó $m_h$ là hệ số góc (slope) riêng biệt cho từng đầu chú ý (ví dụ: $1 / 2^{8h/H}$). Các token ở gần nhau được tăng cường điểm số; các token ở xa nhau bị phạt điểm. Phương pháp này hoàn toàn không tốn chi phí tham số trong lúc huấn luyện. Nghiên cứu thực nghiệm chứng minh khả năng ngoại suy độ dài của ALiBi vượt trội so với mã hóa hình sin và ngang ngửa với RoPE trên độ dài chuỗi huấn luyện gốc.

### Lựa chọn phương pháp nào vào năm 2026

| Biến thể | Khả năng ngoại suy | Chi phí huấn luyện | Các mô hình tiêu biểu |
|----------|-------------------|-------------------|------------------------|
| Hình sin tuyệt đối (Sinusoidal) | kém | không tốn chi phí | Transformer nguyên bản, BERT thời kỳ đầu |
| Tuyệt đối học được (Learned absolute) | không có | rất nhỏ | GPT-2, GPT-3 |
| RoPE | tốt khi có chia tỷ lệ | không tốn chi phí | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | xuất sắc | giai đoạn tinh chỉnh | Qwen2-1M, Llama 3.1 128K |
| ALiBi | xuất sắc | không tốn chi phí | BLOOM, MPT, Baichuan |

RoPE đã chiến thắng nhờ khả năng ghép nối trực tiếp vào cơ chế chú ý mà không làm thay đổi kiến trúc tổng thể, biểu diễn hoàn hảo vị trí tương đối, và siêu tham số `base` đóng vai trò như một núm vặn điều chỉnh trực quan cho việc tinh chỉnh ngữ cảnh dài.

````figure
rope-explorer
````

## Xây dựng thực tế

### Bước 1: Mã hóa hình sin (Sinusoidal Encoding)

Xem file `code/main.py`. Thuật toán tính toán chỉ gói gọn trong vài dòng mã:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Cộng ma trận này vào ma trận nhúng trước khi đi vào tầng chú ý đầu tiên.

### Bước 2: Áp dụng RoPE cho Q và K

RoPE biến đổi trực tiếp trên các vector Q và K. Với mỗi cặp số chiều:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Điểm then chốt: áp dụng cùng một hàm cho vector Q tại vị trí $m$ và vector K tại vị trí $n$. Tích vô hướng giữa chúng sẽ tự động mang thừa số $\cos((m-n) \cdot \theta_i)$ trên từng cặp tọa độ. Cơ chế chú ý nhờ đó học được vị trí tương đối một cách hoàn toàn tự nhiên.

### Bước 3: Tính toán hệ số góc và độ lệch cho ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) với h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # cộng trực tiếp vào điểm số chú ý trước khi tính softmax
```

Cộng `bias[h]` vào ma trận điểm số chú ý kích thước `(seq_len, seq_len)` của đầu thứ `h`, sau đó mới áp dụng hàm softmax.

### Bước 4: Kiểm chứng tính chất khoảng cách tương đối của RoPE

Lấy hai vector ngẫu nhiên `a, b`. Quay chúng theo cặp vị trí `(pos_a, pos_b)`. Sau đó quay tiếp theo cặp `(pos_a + k, pos_b + k)`. Tích vô hướng của hai trường hợp phải trùng khớp nhau trong phạm vi sai số dấu phẩy động. Tính chất này chính là cốt lõi của RoPE — nó hoàn toàn bất biến với độ dời tuyệt đối, chỉ có độ lệch khoảng cách tương đối là quyết định.

## Ứng dụng thực tế

Kể từ PyTorch 2.5+, các tiện ích RoPE đã được tích hợp sẵn trong `torch.nn.functional`. Phần lớn mã nguồn sản xuất sử dụng thư viện `flash_attn` hoặc `xformers`, nơi phép quay RoPE được tối ưu hóa trực tiếp bên trong kernel chú ý trên GPU.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Các kỹ thuật mở rộng ngữ cảnh dài năm 2026:**

- **Nội suy nhận biết NTK (NTK-aware interpolation):** Điều chỉnh tỷ lệ `base` thành `base * (scale_factor)^(d/(d-2))` khi mở rộng từ 4K lên 16K+.
- **YaRN:** Cơ chế nội suy thông minh giúp bảo toàn entropy chú ý trên các ngữ cảnh siêu dài. Llama 3.1 phiên bản 128K sử dụng kỹ thuật này.
- **LongRoPE:** Phương pháp do Microsoft phát triển năm 2024 sử dụng thuật toán tìm kiếm tiến hóa (evolutionary search) để chọn hệ số tỷ lệ riêng biệt cho từng chiều. Mô hình Phi-3-Long áp dụng cơ chế này.
- **Nội suy vị trí kết hợp tinh chỉnh (Position interpolation + fine-tuning):** Thu nhỏ vị trí theo hệ số mở rộng và tinh chỉnh mô hình thêm khoảng 1–5 tỷ token. Đơn giản nhưng mang lại hiệu quả đáng kinh ngạc.

## Triển khai thành phẩm

Xem file `outputs/skill-positional-encoding-picker.md`. Kỹ năng này hỗ trợ lựa chọn chiến lược mã hóa vị trí tối ưu cho mô hình mới dựa trên độ dài ngữ cảnh mục tiêu, yêu cầu ngoại suy và ngân sách huấn luyện.

## Bài tập

1. **Dễ.** Vẽ biểu đồ nhiệt (heatmap) ma trận `PE` hình sin với `max_len=512, d=128`. Xác nhận hiện tượng "các sọc sóng mở rộng dần ra khi chỉ số chiều tăng lên".
2. **Trung bình.** Cài đặt kỹ thuật chia tỷ lệ RoPE theo NTK-aware. Huấn luyện một mô hình ngôn ngữ nhỏ trên chuỗi độ dài 256, sau đó kiểm thử trên độ dài 1.024 có và không có chia tỷ lệ. Đo lường sự thay đổi của chỉ số perplexity.
3. **Khó.** Cài đặt cả ALiBi và RoPE trong cùng một khối chú ý. Huấn luyện mô hình Transformer 4 tầng trên tác vụ sao chép chuỗi với độ dài 512. Ngoại suy lên độ dài 2.048 khi kiểm thử và so sánh mức độ suy giảm chất lượng của hai phương pháp.

## Thuật ngữ then chốt

| Thuật ngữ | Cách nói thông thường | Ý nghĩa kỹ thuật chính xác |
|-----------|----------------------|---------------------------|
| Mã hóa vị trí (Positional encoding) | "Báo cho cơ chế chú ý biết thứ tự từ" | Bất kỳ tín hiệu nào được thêm vào vector nhúng hoặc điểm số chú ý để biểu diễn vị trí. |
| Hình sin (Sinusoidal) | "Phương pháp nguyên bản" | Hàm `sin/cos` ở các tần số cấp số nhân cộng vào vector nhúng; không thể ngoại suy tốt. |
| RoPE (Rotary Position Embedding) | "Nhúng vị trí dạng quay" | Quay Q, K theo góc phụ thuộc vị trí; tích vô hướng tự động mã hóa khoảng cách tương đối. |
| ALiBi | "Thủ thuật độ lệch tuyến tính" | Trừ $-m \cdot \lvert i-j \rvert$ vào điểm số chú ý; không cần vector nhúng, khả năng ngoại suy xuất sắc. |
| base (Cơ số tần số) | "Núm vặn của RoPE" | Hệ số tỷ lệ tần số trong RoPE; tăng giá trị này để mở rộng ngữ cảnh khi suy luận. |
| NTK-aware | "Mẹo chia tỷ lệ RoPE" | Điều chỉnh lại `base` để các chiều tần số cao không bị nén quá mức khi mở rộng ngữ cảnh. |
| YaRN | "Phương pháp chia tỷ lệ nâng cao" | Kỹ thuật kết hợp nội suy và ngoại suy trên từng chiều giúp bảo toàn độ hỗn loạn entropy của cơ chế chú ý. |
| Ngoại suy (Extrapolation) | "Hoạt động vượt ngoài độ dài huấn luyện" | Khả năng của cơ chế vị trí đưa ra kết quả chính xác vượt quá `max_len` từng thấy trong giai đoạn huấn luyện. |

## Tài liệu đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — công trình gốc giới thiệu mã hóa hình sin.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — bài báo giới thiệu RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — bài báo giới thiệu ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — kỹ thuật mở rộng tỷ lệ RoPE tiên tiến nhất.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — bài báo về mở rộng ngữ cảnh dài cho Llama 2 của Meta.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — phương pháp của Microsoft được áp dụng trên Phi-3-Long.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — mã nguồn triển khai chuẩn công nghiệp cho mọi cơ chế chia tỷ lệ RoPE (mặc định, tuyến tính, động, YaRN, LongRoPE, Llama-3).
