# Chú ý đa đầu (Multi-Head Attention)

> Một đầu chú ý chỉ học được một mối quan hệ tại một thời điểm. Tám đầu chú ý học được tám mối quan hệ. Các đầu chú ý gần như không tốn thêm chi phí tính toán. Hãy tận dụng nhiều đầu hơn.

**Phân loại:** Xây dựng thực hành (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 02 (Tự chú ý từ con số không)  
**Thời lượng:** ~75 phút

## Vấn đề đặt ra

Một đầu tự chú ý đơn lẻ chỉ tính toán một ma trận chú ý duy nhất. Ma trận đó chỉ nắm bắt được một loại quan hệ — thường là mối quan hệ giúp tối thiểu hóa hàm mất mát (loss) nhiều nhất dựa trên tín hiệu huấn luyện. Nếu dữ liệu của bạn chứa đựng đồng thời sự tương hợp chủ ngữ - vị ngữ, hiện tượng đồng tham chiếu (co-reference), cấu trúc diễn ngôn tầm xa và phân đoạn cú pháp (syntactic chunking), một đầu chú ý duy nhất sẽ làm nhòe tất cả vào một phân phối softmax chung và làm mất đi một nửa lượng tín hiệu giá trị.

Giải pháp được đưa ra từ bài báo của Vaswani năm 2017: chạy song song nhiều hàm chú ý, mỗi hàm có các phép chiếu Q, K, V độc lập của riêng mình, sau đó ghép nối (concatenate) các đầu ra lại. Mỗi đầu hoạt động trong một không gian con nhỏ hơn với số chiều $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$. Tổng số lượng tham số vẫn giữ nguyên không đổi, nhưng năng lực biểu diễn (expressive power) tăng lên đáng kể.

Cơ chế chú ý đa đầu (multi-head attention - MHA) là chuẩn mặc định xuất hiện trong mọi mô hình Transformer tính đến năm 2026. Điểm tranh luận duy nhất hiện nay chỉ là *dùng bao nhiêu* đầu chú ý và liệu các khóa (keys) cùng giá trị (values) có nên chia sẻ phép chiếu hay không (ví dụ: Chú ý theo nhóm truy vấn - Grouped-Query Attention, Chú ý đa truy vấn - Multi-Query Attention, Chú ý tiềm ẩn đa đầu - Multi-head Latent Attention).

## Khái niệm cốt lõi

![Cơ chế chú ý đa đầu chia tách, tính toán chú ý và ghép nối kết quả](../assets/multi-head-attention.svg)

**Chia tách (Split).** Nhận đầu vào `X` có kích thước `(N, d_model)`. Chiếu tuyến tính sang Q, K, V đều có kích thước `(N, d_model)`. Thay đổi hình dạng (reshape) thành `(N, n_heads, d_head)` với $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$. Chuyển vị (transpose) thành `(n_heads, N, d_head)`.

**Tính toán chú ý song song (Attend in parallel).** Thực hiện chú ý tích vô hướng có chia tỷ lệ bên trong từng đầu. Mỗi đầu tạo ra một tensor có kích thước `(N, d_head)`. Các đầu chú ý hoạt động trên các không gian con biểu diễn khác nhau của vector nhúng và hoàn toàn không tương tác với nhau trong suốt quá trình tính toán chú ý.

**Ghép nối và chiếu đầu ra (Concatenate and project).** Gom các đầu chú ý trở lại kích thước `(N, d_model)` và nhân với một ma trận trọng số đầu ra học được $W_o$ có kích thước `(d_model, d_model)`. $W_o$ chính là nơi thông tin từ các đầu khác nhau được hòa trộn và tổng hợp.

**Vì sao cơ chế này phát huy hiệu quả.** Mỗi đầu có thể chuyên môn hóa vào một nhiệm vụ cụ thể mà không phải cạnh tranh tài nguyên biểu diễn với các đầu khác. Các nghiên cứu thăm dò (probing studies) giai đoạn 2019–2024 chỉ ra rằng các đầu chú ý đảm nhận những vai trò rất rõ ràng: đầu chú ý vị trí (positional heads), đầu chú ý đến token liền trước, đầu sao chép (copy heads), đầu nhận diện thực thể có tên (named-entity heads), và các đầu cảm ứng (induction heads — nền tảng của khả năng học trong ngữ cảnh / in-context learning).

**Các biến thể tiến hóa tính đến năm 2026:**

| Biến thể | Số đầu Q | Số đầu K/V | Được sử dụng bởi |
|----------|----------|------------|------------------|
| Chú ý đa đầu (MHA) | N | N | GPT-2, BERT, T5 |
| Chú ý đa truy vấn (MQA) | N | 1 | PaLM, Falcon |
| Chú ý theo nhóm truy vấn (GQA) | N | G (ví dụ: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Chú ý tiềm ẩn đa đầu (MLA) | N | nén về không gian hạng thấp (low-rank) | DeepSeek-V2, V3 |

GQA đã trở thành tiêu chuẩn mặc định hiện đại vì nó giúp giảm bộ nhớ bộ đệm KV (KV-cache) theo hệ số $N/G$ trong khi vẫn duy trì gần như trọn vẹn chất lượng mô hình. MLA tiến xa hơn bằng cách nén K/V vào một không gian tiềm ẩn (latent space), sau đó giải nén tại thời điểm tính toán — đánh đổi thêm FLOPs để tiết kiệm lượng lớn bộ nhớ cache.

````figure
multihead-split
````

## Xây dựng thực tế

### Bước 1: Chia tách các đầu từ cơ chế chú ý đơn đầu có sẵn

Sử dụng lớp `SelfAttention` từ Bài 02 và bọc bên ngoài bằng cặp hàm chia tách / kết hợp. Xem `code/main.py` để xem phiên bản cài đặt bằng NumPy; logic toán học như sau:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Chỉ cần một phép biến đổi hình dạng (reshape) và một phép chuyển vị (transpose). Không cần bất kỳ vòng lặp nào. Đây chính xác là cách mà PyTorch thực thi bên trong `nn.MultiheadAttention`.

### Bước 2: Chạy cơ chế chú ý tích vô hướng có chia tỷ lệ trên từng đầu

Mỗi đầu nhận một lát cắt riêng biệt của Q, K, V. Cơ chế chú ý trở thành một phép nhân ma trận theo lô (batched matrix multiplication - bmm):

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Trên phần cứng thực tế, phép tính `Qh @ Kh.transpose(...)` tương đương một lệnh `bmm` duy nhất. GPU nhìn nhận đây là một phép nhân ma trận theo lô kích thước `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Việc bổ sung thêm số lượng đầu chú ý gần như không tốn thêm chi phí thực thi.

### Bước 3: Biến thể chú ý theo nhóm truy vấn (Grouped-Query Attention - GQA)

Chỉ có các phép chiếu key và value là thay đổi. Q có `n_heads` nhóm; K và V có `n_kv_heads < n_heads` nhóm và được lặp lại (repeat) để khớp với kích thước của Q:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Trong quá trình suy luận (inference), kỹ thuật này giúp tiết kiệm bộ nhớ đáng kể vì chỉ có `n_kv_heads` bản sao được lưu trữ trong bộ đệm KV, thay vì toàn bộ `n_heads`. Mô hình Llama 3 70B sử dụng 64 đầu query cùng 8 đầu KV — giúp giảm kích thước bộ đệm cache tới 8 lần.

### Bước 4: Thăm dò cấu trúc học được của từng đầu

Chạy mô hình MHA trên một câu ngắn với 4 đầu chú ý. Với mỗi đầu, in ra ma trận chú ý kích thước `(N, N)`. Bạn sẽ nhận thấy các đầu khác nhau chú ý vào những cấu trúc ngữ nghĩa hoàn toàn khác nhau ngay cả khi chỉ mới khởi tạo ngẫu nhiên — điều này một phần đến từ tín hiệu thực tế, một phần do tính đối xứng quay trong các không gian con.

## Ứng dụng thực tế

Trong PyTorch, phiên bản chỉ với một dòng lệnh:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA kể từ phiên bản PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention tự động điều phối FlashAttention trên CUDA.
# Với GQA, truyền Q có kích thước (B, n_heads, N, d_head) và K, V có kích thước
# (B, n_kv_heads, N, d_head). PyTorch sẽ tự động xử lý việc lặp lại đầu KV.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Cần bao nhiêu đầu chú ý?** Quy tắc kinh nghiệm rút ra từ các mô hình sản xuất thực tế vào năm 2026:

| Quy mô mô hình | d_model | n_heads | d_head |
|----------------|---------|---------|--------|
| Nhỏ (~125M) | 768 | 12 | 64 |
| Cơ sở (~350M) | 1024 | 16 | 64 |
| Lớn (~1B) | 2048 | 16 | 128 |
| Đỉnh cao (~70B) | 8192 | 64 | 128 |

Số chiều của mỗi đầu ($d_{\text{head}}$) hầu như luôn được cố định ở mức 64 hoặc 128. Đây là đơn vị đo lường lượng thông tin mà một đầu chú ý đơn lẻ có thể "quan sát". Nếu giảm xuống dưới 32, các đầu chú ý sẽ gặp khó khăn với hệ số chia tỷ lệ $\sqrt{d_{\text{head}}}$; nếu vượt quá 256, bạn sẽ đánh mất lợi thế từ việc sở hữu "nhiều chuyên gia quy mô nhỏ".

## Triển khai thành phẩm

Xem file `outputs/skill-mha-configurator.md`. Kỹ năng này đưa ra đề xuất về số lượng đầu chú ý, số lượng đầu KV và chiến lược chiếu phù hợp cho một mô hình Transformer mới dựa trên ngân sách tham số, độ dài ngữ cảnh và môi trường triển khai phần cứng.

## Bài tập

1. **Dễ.** Lấy khối MHA từ file `code/main.py` và thay đổi `n_heads` từ 1 lên 16 trong khi giữ cố định `d_model=64`. Vẽ biểu đồ hàm mất mát của một mô hình một tầng nhỏ trên tác vụ sao chép chuỗi nhân tạo (synthetic copy task). Việc tăng số lượng đầu giúp cải thiện, giữ nguyên hay làm giảm hiệu năng?
2. **Trung bình.** Cài đặt cơ chế MQA (một đầu KV dùng chung cho tất cả các đầu query). Đo lường mức giảm số lượng tham số so với MHA tiêu chuẩn. Tính toán xem kích thước bộ đệm KV giảm bao nhiêu lần khi suy luận với $N=2048$.
3. **Khó.** Cài đặt một phiên bản thu nhỏ của cơ chế Chú ý tiềm ẩn đa đầu (Multi-head Latent Attention - MLA): nén K, V thành một biến tiềm ẩn có hạng $r$, lưu biến tiềm ẩn này trong bộ đệm KV, sau đó giải nén tại thời điểm tính chú ý. Ở giá trị hạng $r$ nào thì dung lượng bộ đệm cache giảm xuống dưới 1/8 so với MHA tiêu chuẩn trong khi độ hỗn loạn (perplexity) trên tập kiểm thử chỉ chênh lệch trong vòng 1 bit?

## Thuật ngữ then chốt

| Thuật ngữ | Cách nói thông thường | Ý nghĩa kỹ thuật chính xác |
|-----------|----------------------|---------------------------|
| Đầu chú ý (Head) | "Một mạch tính toán chú ý đơn lẻ" | Một phép chiếu Q/K/V có số chiều $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ kèm theo ma trận chú ý riêng biệt. |
| Chiều của đầu (d_head) | "Độ rộng ẩn của từng đầu" | Kích thước vector của mỗi đầu chú ý; hầu như luôn là 64 hoặc 128 trong các hệ thống sản xuất thực tế. |
| Chia tách / Kết hợp (Split / combine) | "Thủ thuật biến đổi kích thước" | Các phép toán biến đổi `(N, d_model) ↔ (n_heads, N, d_head)` bằng reshape và transpose bao quanh khối chú ý. |
| Ma trận đầu ra (W_o) | "Phép chiếu đầu ra" | Ma trận trọng số kích thước `(d_model, d_model)` được áp dụng sau khi ghép nối các đầu; là nơi các đầu hòa trộn thông tin. |
| MQA (Multi-Query Attention) | "Một đầu KV duy nhất" | Cơ chế chú ý đa truy vấn: chỉ sử dụng một phép chiếu K/V duy nhất cho toàn bộ các đầu Q. Giảm tối đa bộ đệm KV nhưng suy giảm nhẹ chất lượng. |
| GQA (Grouped-Query Attention) | "Chuẩn mặc định kể từ Llama 2" | Chú ý theo nhóm truy vấn với `n_kv_heads < n_heads`; lặp lại các đầu KV để khớp kích thước với nhóm đầu Q. |
| MLA (Multi-head Latent Attention) | "Bí quyết của DeepSeek" | Chú ý tiềm ẩn đa đầu: K, V được nén thành biểu diễn tiềm ẩn hạng thấp, sau đó giải nén tại thời điểm tính toán chú ý. |
| Đầu cảm ứng (Induction head) | "Mạch nơ-ron đứng sau khả năng in-context learning" | Một cặp đầu chú ý có khả năng phát hiện các sự xuất hiện trước đó và sao chép chính xác token đi liền sau chúng. |

## Tài liệu đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — đặc tả kỹ thuật gốc của cơ chế chú ý đa đầu.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — bài báo giới thiệu kiến trúc MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — phương pháp chuyển đổi MHA sang GQA sau huấn luyện.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — cơ chế MLA và lý do nó vượt trội hơn MHA/GQA về hiệu quả bộ nhớ cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — phân tích cơ chế nội tại về chức năng thực tế của các đầu chú ý.
