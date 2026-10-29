# Cơ chế Chú ý — Bước đột phá (Attention Mechanism)

> Decoder ngừng nheo mắt nhìn vào một bản tóm tắt nén chặt và bắt đầu quan sát toàn bộ câu nguồn. Mọi thứ sau cột mốc này chỉ là cơ chế chú ý kết hợp với kỹ thuật tối ưu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Mô hình Sequence-to-Sequence)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Bài học 09 kết thúc bằng một thất bại đã được đo đạc chính xác. Một mô hình GRU encoder-decoder huấn luyện trên tác vụ sao chép đồ chơi tụt dốc từ 89% độ chính xác ở độ dài 5 xuống gần như đoán mò ngẫu nhiên ở độ dài 80. Nguyên nhân hoàn toàn mang tính cấu trúc chứ không phải lỗi huấn luyện (training bug): Mọi thông tin mà encoder thu thập được đều phải nhồi nhét vào một trạng thái ẩn duy nhất có kích thước cố định, và decoder không bao giờ được nhìn thấy bất cứ thứ gì khác ngoài vector đó.

Năm 2014, Bahdanau, Cho và Bengio đã công bố một giải pháp ba dòng mã làm thay đổi tất cả: Thay vì chỉ cung cấp cho decoder trạng thái encoder cuối cùng, hãy lưu giữ lại tất cả các trạng thái ẩn của encoder. Tại mỗi bước giải mã, tính trung bình có trọng số của các trạng thái encoder, trong đó các trọng số thể hiện "ngay lúc này decoder cần chú ý đến vị trí `i` nào của encoder bao nhiêu phần trăm?". Vector trung bình có trọng số đó chính là ngữ cảnh, và nó thay đổi linh hoạt tại từng bước giải mã.

Đó chính là toàn bộ ý tưởng cốt lõi. Transformer sau này đã kế thừa và mở rộng nó. Tự chú ý (Self-attention) áp dụng cơ chế này lên chính bản thân một chuỗi. Chú ý đa đầu (Multi-head attention) cho phép chạy song song nhiều phép chiếu. Nhưng phiên bản năm 2014 đã phá vỡ hoàn toàn nút thắt cổ chai, và một khi bạn đã nắm vững nó, bước chuyển dịch sang Transformer chỉ thuần túy là kỹ thuật công nghệ chứ không còn là rào cản khái niệm.

## Khái niệm (The Concept)

![Cơ chế chú ý Bahdanau: decoder truy vấn tất cả các trạng thái encoder](../assets/attention.svg)

Tại mỗi bước giải mã `t`:

1. Sử dụng trạng thái ẩn trước đó của decoder `s_{t-1}` làm một **truy vấn (query)**.
2. Chấm điểm tương đồng giữa truy vấn này với từng trạng thái ẩn của encoder `h_1, ..., h_T`. Cho ra một điểm số vô hướng cho mỗi vị trí encoder.
3. Áp dụng Softmax lên các điểm số để nhận được các trọng số chú ý `α_{t,1}, ..., α_{t,T}` có tổng bằng 1.
4. Tính vector ngữ cảnh `c_t = Σ α_{t,i} * h_i`. Đây là trung bình có trọng số của các trạng thái encoder.
5. Decoder nhận `c_t` kết hợp cùng token đầu ra trước đó để sinh ra token tiếp theo.

Ý nghĩa then chốt nằm ở phép trung bình có trọng số: Khi decoder cần dịch từ "Je" sang "I", nó gán trọng số cao cho trạng thái encoder tương ứng với từ "Je" và hạ thấp các từ khác. Khi cần dịch từ "not", nó tập trung trọng số vào từ "pas". Vector ngữ cảnh được tái định hình linh hoạt tại mỗi bước thời gian.

## Kích thước Tensor (Điều khiến ai cũng vấp ngã lần đầu)

Đây là nơi hầu hết các bản cài đặt attention bị lỗi trong lần viết mã đầu tiên. Hãy đọc thật cẩn thận:

| Khái niệm | Kích thước (Shape) | Ghi chú |
|-----------|--------------------|---------|
| Các trạng thái ẩn encoder `H` | `(T_enc, d_h)` | Nếu dùng BiLSTM, `d_h = 2 * d_hidden` |
| Trạng thái ẩn decoder `s_{t-1}` | `(d_s,)` | Một vector đơn lẻ |
| Điểm số chú ý `e_{t,i}` | scalar (vô hướng) | Một điểm cho mỗi vị trí encoder |
| Trọng số chú ý `α_{t,i}` | scalar (vô hướng) | Sau khi softmax trên toàn bộ `i` |
| Vector ngữ cảnh `c_t` | `(d_h,)` | Cùng kích thước với trạng thái encoder |

**Điểm số Bahdanau (cộng tính - additive):** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` có kích thước `(d_s,)`, `h_i` có kích thước `(d_h,)`.
- `W_a` có kích thước `(d_attn, d_s)`. `U_a` có kích thước `(d_attn, d_h)`.
- Tổng của chúng bên trong hàm tanh có kích thước `(d_attn,)`.
- `v_α` có kích thước `(d_attn,)`. Tích vô hướng với `v_α` thu gọn tensor về một đại lượng vô hướng (scalar). **Đây chính là vai trò thực sự của `v_α`.** Nó không hề huyền bí; nó chỉ là phép chiếu vector chiều attention thành một điểm số vô hướng duy nhất.

**Điểm số Luong (nhân tính - multiplicative):** Ba biến thể:

- `dot`: `e_{t,i} = s_t^T * h_i`. Đòi hỏi bắt buộc `d_s == d_h`. Ràng buộc cứng. Hãy bỏ qua nếu encoder của bạn là mạng hai chiều (bidirectional).
- `general`: `e_{t,i} = s_t^T * W * h_i` với `W` có kích thước `(d_s, d_h)`. Loại bỏ ràng buộc bằng nhau về số chiều.
- `concat`: về bản chất tương tự dạng Bahdanau. Hiếm khi sử dụng vì hai dạng đầu tiên tính toán rẻ hơn nhiều.

**Cạm bẫy quan trọng giữa Bahdanau và Luong:** Bahdanau sử dụng `s_{t-1}` (trạng thái decoder *trước* khi sinh từ hiện tại). Luong sử dụng `s_t` (trạng thái *sau* khi tính toán). Trộn lẫn hai quy ước này sẽ tạo ra gradient sai lệch tinh vi và cực kỳ khó gỡ lỗi (debug). Hãy chọn một bài báo và tuân thủ nhất quán quy ước của bài báo đó.

```figure
attention-heatmap
```

## Xây dựng thực tế (Build It)

### Bước 1: Chú ý cộng tính (Bahdanau attention)

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Kiểm tra kích thước tensor theo bảng trên: `encoder_states` có shape `(T_enc, d_h)`. `projected_enc` có shape `(T_enc, d_attn)`. `projected_dec` có shape `(d_attn,)` và được broadcast tự động. `combined` có shape `(T_enc, d_attn)`. `scores` có shape `(T_enc,)`. `weights` có shape `(T_enc,)`. `context` có shape `(d_h,)`. Chuẩn xác.

### Bước 2: Chú ý Luong dot và general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Mỗi hàm chỉ vỏn vẹn ba dòng mã. Đó là lý do bài báo của Luong tạo tiếng vang lớn: Độ chính xác tương đương trên hầu hết các tác vụ nhưng mã nguồn ngắn gọn và tính toán nhanh hơn nhiều.

### Bước 3: Ví dụ tính toán bằng số cụ thể

Cho ba trạng thái encoder (tương ứng đại diện cho "cat", "sat", "mat") và một trạng thái decoder tương đồng nhất với từ đầu tiên, phân phối chú ý sẽ tập trung áp đảo vào vị trí 0. Nếu trạng thái decoder chuyển dịch để khớp với từ cuối cùng, attention sẽ dời sang vị trí 2. Vector ngữ cảnh di chuyển theo tương ứng.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```text
weights: [0.464 0.305 0.231]
```

Hàng đầu tiên chiến thắng. Thử dịch chuyển trạng thái decoder lại gần trạng thái encoder thứ ba và bạn sẽ thấy các trọng số dịch chuyển theo. Bản chất chỉ có vậy: Attention chính là phép gióng hàng tường minh (explicit alignment).

### Bước 4: Chiếc cầu nối sang Transformer

Chuyển đổi ngôn ngữ trên sang bộ ba Q/K/V:

- **Query (Truy vấn)** = trạng thái decoder `s_{t-1}`
- **Key (Khóa)** = các trạng thái encoder (đối tượng để chấm điểm tương đồng)
- **Value (Giá trị)** = các trạng thái encoder (đối tượng để nhân trọng số và lấy tổng)

Trong cơ chế chú ý cổ điển, Key và Value là một. Cơ chế tự chú ý (Self-attention) phân tách chúng: bạn có thể truy vấn một chuỗi với chính nó, thông qua các phép chiếu học được riêng biệt cho K và V. Multi-head attention chạy song song nhiều phép chiếu độc lập. Transformer xếp chồng tầng chú ý này nhiều lần và loại bỏ hoàn toàn mạng RNN.

Bản chất toán học là như nhau. Kích thước tensor là như nhau. Bước nhảy sư phạm từ Bahdanau attention sang Scaled Dot-Product Attention trong Transformer phần lớn chỉ là sự thay đổi về mặt ký hiệu toán học.

## Sử dụng trong thực tế (Use It)

PyTorch và TensorFlow cung cấp sẵn các tầng attention tiêu chuẩn.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```text
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Đó chính là một tầng chú ý của Transformer: Batch truy vấn gồm 5 vị trí, batch key/value gồm 10 vị trí, mỗi vị trí 128 chiều, 8 đầu chú ý độc lập. `output` là các truy vấn mới đã được bổ sung ngữ cảnh. `weights` là ma trận gióng hàng 5x10 mà bạn có thể trực quan hóa trực tiếp.

### Khi nào cơ chế chú ý cổ điển vẫn còn hữu ích

- Mục đích giáo dục: Phiên bản RNN 1 đầu chú ý, 1 tầng làm sáng tỏ mọi ngóc ngách của khái niệm.
- Các bài toán chuỗi trên thiết bị biên khi Transformer quá nặng.
- Đọc hiểu các công trình nghiên cứu giai đoạn 2014-2017: Bạn sẽ hiểu sai nếu không nắm vững quy ước Bahdanau.
- Phân tích gióng hàng chi tiết trong dịch máy: Trọng số attention thô là công cụ giải thích hành vi mô hình ngay cả trên Transformer, và để đọc được chúng, bạn cần hiểu bản chất của chúng.

### Cạm bẫy coi trọng số chú ý là lời giải thích (Interpretability trap)

Trọng số attention trông có vẻ rất dễ diễn giải: Chúng là các trọng số có tổng bằng 1 qua các vị trí; bạn có thể vẽ biểu đồ nhiệt (heatmap); giá trị cao có nghĩa là mô hình "đang chú ý vào từ này". Người bình duyệt rất thích chúng.

Nhưng chúng không dễ giải thích như vẻ bề ngoài. Jain và Wallace (2019) đã chứng minh rằng các phân phối attention có thể bị xáo trộn hoặc thay thế bằng các phương án phân phối tùy ý khác mà không hề làm thay đổi kết quả dự đoán cuối cùng của mô hình trên nhiều tác vụ. Đừng bao giờ báo cáo trọng số attention như một bằng chứng duy nhất cho quá trình suy luận mà thiếu các thử nghiệm triệt tiêu (ablation study) hoặc kiểm tra phản thực tế (counterfactual check).

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Gỡ lỗi sai lệch kích thước tensor trong cài đặt cơ chế chú ý.
phase: 5
lesson: 10
---

Khi nhận được một bản cài đặt attention bị lỗi, bạn chỉ ra điểm bất đối xứng kích thước:

1. Ma trận nào có shape sai lệch. Đặt tên cụ thể cho tensor đó.
2. Shape chuẩn xác phải là gì, được suy diễn từ (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. Bản sửa lỗi trong một dòng mã: transpose, reshape, hay linear projection.
4. Một bài kiểm thử để ngăn ngừa lỗi tái diễn (regression test): Thường là: kiểm tra `output.shape == (batch, T_dec, d_h)` và `weights.shape == (batch, T_dec, T_enc)` cùng điều kiện `weights.sum(dim=-1)` xấp xỉ bằng 1.

Từ chối đề xuất các bản sửa lỗi dựa vào cơ chế broadcast ngầm. Các lỗi ẩn sau broadcast sẽ âm thầm làm suy giảm độ chính xác của mô hình khi huấn luyện, loại lỗi tồi tệ nhất trong attention.

Với dạng Bahdanau, nhấn mạnh đầu vào decoder phải là `s_{t-1}` (trạng thái trước bước sinh). Với Luong, là `s_t` (trạng thái sau khi tính toán). Với dot-product, cảnh báo sự không khớp số chiều giữa query và key là lỗi sơ đẳng phổ biến nhất.
```

## Bài tập (Exercises)

1. **Dễ.** Cài đặt cơ chế che mặt nạ (masking) trong hàm `softmax` để các token đệm (padding) trong encoder nhận trọng số attention bằng 0. Kiểm thử trên một batch có độ dài các chuỗi không đồng đều.
2. **Trung bình.** Bổ sung cơ chế chú ý đa đầu (multi-head attention) cho dạng Luong `general`. Chia `d_h` thành `n_heads` nhóm, chạy attention độc lập cho từng đầu, rồi ghép nối lại. Kiểm chứng rằng trường hợp 1 đầu khớp hoàn toàn với bản cài đặt trước đó của bạn.
3. **Khó.** Huấn luyện một mô hình GRU encoder-decoder với Bahdanau attention trên tác vụ sao chép đồ chơi từ bài học 09. Vẽ biểu đồ tương quan giữa độ chính xác và độ dài chuỗi. So sánh với baseline không dùng attention. Bạn sẽ thấy khoảng cách cách biệt nới rộng khi độ dài tăng lên, xác nhận attention đã xóa bỏ nút thắt cổ chai.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Cơ chế chú ý (Attention) | Nhìn vào sự vật | Trung bình có trọng số của chuỗi giá trị (Value), với trọng số được tính từ độ tương đồng giữa truy vấn (Query) và khóa (Key). |
| Query, Key, Value | Bộ ba QKV | Ba phép chiếu: Q đưa ra yêu cầu, K là tiêu chí so khớp, V là nội dung trả về. |
| Chú ý cộng tính (Additive attention) | Bahdanau | Điểm số tính qua mạng lan truyền tiến: `v^T tanh(W q + U k)`. |
| Chú ý nhân tính (Multiplicative attention) | Luong dot / general | Điểm số là tích vô hướng `q^T k` hoặc `q^T W k`. Rẻ hơn, độ chính xác tương đương trên hầu hết tác vụ. |
| Ma trận gióng hàng (Alignment matrix) | Biểu đồ trực quan | Trọng số chú ý dưới dạng lưới kích thước `(T_dec, T_enc)`. Đọc ma trận này để biết mô hình chú ý vào đâu. |

## Đọc thêm (Further Reading)

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — bài báo gốc khai sinh attention.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — bài báo so sánh ba biến thể chấm điểm chú ý.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — bài nghiên cứu cảnh báo về khả năng diễn giải của attention.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — hướng dẫn lập trình trực quan với PyTorch.

