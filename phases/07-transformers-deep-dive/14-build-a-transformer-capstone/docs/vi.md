# Xây dựng Transformer từ đầu — Dự án tổng kết (Build a Transformer from Scratch — The Capstone)

> Mười ba bài học. Một mô hình hoàn chỉnh. Không đi đường tắt.

**Phân loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · từ bài 01 đến 13. Đừng bỏ qua bài nào.  
**Thời lượng:** ~120 phút

## Vấn đề đặt ra

Bạn đã đọc từng bài báo khoa học. Bạn đã tự tay cài đặt cơ chế chú ý, tách đa đầu (multi-head splits), mã hóa vị trí (positional encodings), các khối bộ mã hóa (encoder) và giải mã (decoder), các hàm mất mát của BERT và GPT, hỗn hợp chuyên gia (MoE), bộ đệm KV (KV cache). Giờ là lúc kết hợp tất cả chúng lại để giải quyết một tác vụ thực tế.

Dự án tổng kết (capstone): huấn luyện một mô hình transformer chỉ gồm bộ giải mã (decoder-only) từ đầu đến cuối trên tác vụ mô hình hóa ngôn ngữ cấp ký tự (character-level language modeling). Mô hình sẽ đọc các tác phẩm của Shakespeare và tự sinh ra các đoạn văn mang phong cách Shakespeare. Kích thước mô hình đủ nhỏ để huấn luyện trên một chiếc máy tính xách tay trong vòng chưa đầy 10 phút, nhưng đủ chuẩn xác để khi thay thế bằng tập dữ liệu lớn hơn và thời gian huấn luyện lâu hơn, bạn sẽ thu được một mô hình ngôn ngữ (LM) thực thụ.

Đây chính là phiên bản "nanoGPT" của khóa học. Ý tưởng này không mới — hướng dẫn nanoGPT năm 2023 của Andrej Karpathy là cài đặt tham chiếu kinh điển mà mọi kỹ sư AI đều nên tự viết ít nhất một lần. Chúng ta kế thừa cấu trúc cốt lõi đó và tái cấu trúc lại xoay quanh những kiến thức đã học.

## Khái niệm cốt lõi

![Sơ đồ khối Transformer xây dựng từ đầu](../assets/capstone.svg)

Kiến trúc chi tiết kèm chú thích:

```text
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Bài 04 (Tùy chọn RoPE)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Bài 05
│  MultiHeadAttention (causal)      │  ◀── Bài 03 + 07 (mặt nạ causal)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Bài 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (buộc chung trọng số với token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Bài 07
```

### Những gì chúng ta cung cấp sẵn

- `GPTConfig` — nơi duy nhất cấu hình toàn bộ các siêu tham số.
- `MultiHeadAttention` — chú ý đa đầu nhân quả (causal), xử lý theo lô, có thể sử dụng đường dẫn tối ưu kiểu FlashAttention (`scaled_dot_product_attention` của PyTorch).
- `SwiGLUFFN` — mạng FFN hiện đại.
- `Block` — chuẩn hóa trước (pre-norm), bao bọc bởi kết nối thặng dư (residual) quanh attention + FFN.
- `GPT` — các tầng nhúng, xếp chồng các khối block, đầu ra LM head, phương thức `generate()`.
- Vòng lặp huấn luyện với AdamW, lịch trình suy giảm tốc độ học cosin (cosine LR schedule), cắt tỉa gradient (gradient clipping).
- Bộ tách token cấp ký tự (character-level tokenizer) trên văn bản của Shakespeare.

### Những gì chúng ta tạm lược bỏ

- RoPE — đã được cài đặt về mặt khái niệm ở Bài 04. Ở đây chúng ta dùng tầng nhúng vị trí có thể học (learned positional embeddings) cho đơn giản. Phần bài tập sẽ yêu cầu bạn thay thế bằng RoPE.
- Bộ đệm KV trong quá trình sinh — mỗi bước sinh token sẽ tính toán lại cơ chế chú ý trên toàn bộ tiền tố. Chậm hơn nhưng mã nguồn đơn giản hơn. Phần bài tập sẽ yêu cầu bạn bổ sung KV cache.
- FlashAttention chuyên sâu — PyTorch 2.0+ tự động điều phối nếu đầu vào khớp định dạng; chúng ta sử dụng `F.scaled_dot_product_attention`.
- MoE — dùng một FFN duy nhất cho mỗi khối block. Bạn đã tìm hiểu MoE ở Bài 11.

### Các chỉ số mục tiêu

Trên một máy tính xách tay Mac M2, mô hình GPT gồm 4 tầng, 4 đầu chú ý, d_model=128 được huấn luyện qua 2.000 bước trên file `tinyshakespeare.txt`:

- Mất mát huấn luyện (training loss) hội tụ từ ~4,2 (khởi tạo ngẫu nhiên) xuống ~1,5 trong khoảng 6 phút.
- Văn bản lấy mẫu đầu ra mang đậm phong thái của Shakespeare: xuất hiện các từ ngữ cổ, ngắt dòng chuẩn xác và tên các nhân vật như "ROMEO:".
- Mất mát kiểm định (val loss trên 10% phần văn bản giữ lại) bám sát training loss; không xảy ra hiện tượng quá khớp (overfitting) ở quy mô và ngân sách này.

````figure
n5-block-stack
````

## Xây dựng thực tế

Bài học này sử dụng PyTorch. Cài đặt thư viện `torch` (bản chạy CPU là hoàn toàn đủ). Xem file `code/main.py`. Kịch bản xử lý:

- Tự động tải `tinyshakespeare.txt` nếu chưa có (hoặc đọc từ bản sao cục bộ).
- Bộ tách token cấp ký tự mức byte.
- Phân chia tập huấn luyện/kiểm định (train/val split) theo tỷ lệ 90/10.
- Vòng lặp huấn luyện hỗ trợ tự động ép kiểu bf16 (autocast) trên phần cứng tương thích.
- Lấy mẫu sinh văn bản sau khi hoàn tất huấn luyện.

### Bước 1: Dữ liệu

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 ký tự duy nhất. Từ vựng cực nhỏ. Vừa khít với một `vocab_size` gọn nhẹ. Không cần thuật toán BPE phức tạp, không gặp rắc rối từ bộ tách từ.

### Bước 2: Mô hình

Xem file `code/main.py`. Cấu trúc khối block chuẩn chỉ như đã học ở Bài 05 — pre-norm, RMSNorm, SwiGLU, causal MHA. Tổng số lượng tham số cho cấu hình 4 tầng / 4 đầu / d_model 128: ~800K tham số.

### Bước 3: Vòng lặp huấn luyện

Lấy ngẫu nhiên một lô các cửa sổ token có độ dài 256. Lan truyền xuôi (forward). Tính mất mát cross-entropy dịch chuyển 1 vị trí (shift-by-one). Lan truyền ngược (backward). Cập nhật bước tối ưu AdamW. Ghi log. Lặp lại.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Bước 4: Lấy mẫu sinh văn bản

Cho trước một câu mồi (prompt), liên tục lan truyền xuôi, lấy mẫu từ các logit theo phương pháp top-p, nối thêm token mới và tiếp tục. Dừng lại sau khi sinh đủ 500 token.

### Bước 5: Đọc kết quả đầu ra

Sau 2.000 bước:

```text
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Chưa phải là Shakespeare thực thụ, nhưng cấu trúc văn phong đã rất giống Shakespeare. Một thắng lợi rõ ràng cho ~800K tham số và 6 phút huấn luyện trên máy tính cá nhân.

## Ứng dụng thực tế

Dự án tổng kết này là một kiến trúc tham chiếu hoàn chỉnh. Ba bước nâng cấp để đưa nó vào ứng dụng thực tế:

1. **Thay đổi bộ tách token.** Sử dụng BPE (ví dụ: `tiktoken.get_encoding("cl100k_base")`). Kích thước từ vựng sẽ tăng từ 65 lên ~50.000. Cần nâng dung lượng mô hình để bù lại.
2. **Huấn luyện trên tập ngữ liệu lớn hơn.** Sử dụng `OpenWebText` hoặc `fineweb-edu` (HuggingFace). Huấn luyện 10B token trên một GPU A100 duy nhất mất khoảng 24 giờ cho mô hình GPT 125M tham số.
3. **Bổ sung RoPE + KV cache + FlashAttention.** Các bài tập bên dưới sẽ hướng dẫn bạn từng bước thực hiện.

Quy trình này sẽ tạo ra một mô hình GPT 125M tham số có khả năng sinh tiếng Anh lưu loát. Dù không phải mô hình tiên phong, nhưng chính luồng mã nguồn này — chỉ cần mở rộng quy mô lớn hơn — là những gì Karpathy, EleutherAI và Viện Allen (Ai2) sử dụng để huấn luyện các checkpoint nghiên cứu vào năm 2026.

## Đóng gói ứng dụng

Xem file `outputs/skill-transformer-review.md`. Kỹ năng này đóng vai trò phản biện, đánh giá tính chuẩn xác của một mô hình transformer được lập trình từ đầu dựa trên kiến thức của toàn bộ 13 bài học trước.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Xác minh rằng mất mát kiểm định ở bước cuối cùng của mô hình thấp hơn 2,0. Tăng `max_steps` từ 2.000 lên 5.000 — liệu mất mát kiểm định có tiếp tục giảm không?
2. **Trung bình.** Thay thế tầng nhúng vị trí có thể học bằng cơ chế nhúng vị trí quay RoPE. Áp dụng phép quay lên ma trận $Q$ và $K$ bên trong `MultiHeadAttention`. Huấn luyện lại và kiểm tra xem mất mát kiểm định có đạt mức thấp tương đương hoặc tốt hơn không.
3. **Trung bình.** Cài đặt bộ đệm KV vào vòng lặp lấy mẫu sinh văn bản. Sinh 500 token có dùng cache và không dùng cache. Thời gian thực thi thực tế (wall-clock) sẽ cải thiện từ 5–20 lần trên máy tính cá nhân.
4. **Khó.** Thêm một đầu dự đoán thứ hai vào mô hình để dự đoán token kế tiếp sau token hiện tại (Dự đoán đa token - MTP từ DeepSeek-V3). Huấn luyện kết hợp cả hai đầu. Điều này có giúp cải thiện chất lượng mô hình không?
5. **Khó.** Thay thế mạng FFN đơn lẻ trong mỗi khối block bằng một tầng MoE 4 chuyên gia. Sử dụng router + định tuyến top-2. Quan sát xem mất mát kiểm định thay đổi như thế nào khi giữ nguyên số lượng tham số tích cực tương đương.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| nanoGPT | "Kho mã nguồn hướng dẫn của Karpathy" | Mã nguồn huấn luyện transformer chỉ gồm bộ giải mã (decoder-only) tối giản, ~300 dòng code; chuẩn tham chiếu kinh điển. |
| tinyshakespeare | "Tập ngữ liệu mẫu tiêu chuẩn" | ~1,1 MB văn bản; mọi bài giảng mô hình ngôn ngữ cấp ký tự từ năm 2015 đến nay đều sử dụng. |
| Buộc chung trọng số nhúng (Tied embeddings) | "Chia sẻ ma trận đầu vào/đầu ra" | Trọng số của LM head = chuyển vị ma trận nhúng token; tiết kiệm tham số và nâng cao chất lượng. |
| Ép kiểu tự động bf16 (bf16 autocast) | "Mẹo tối ưu độ chính xác huấn luyện" | Chạy forward/backward ở định dạng bf16, giữ trạng thái tối ưu ở fp32; chuẩn mực từ năm 2021. |
| Cắt tỉa gradient (Gradient clipping) | "Chặn đứng các cú vọt gradient" | Giới hạn chuẩn gradient toàn cục (global grad norm) ở mức 1,0; ngăn chặn hiện tượng bùng nổ gradient làm hỏng huấn luyện. |
| Lịch trình suy giảm cosin (Cosine LR schedule) | "Lựa chọn mặc định từ năm 2020+" | Tốc độ học tăng tuyến tính (warmup) sau đó suy giảm theo hình sóng cosin về mức 10% đỉnh. |
| MFU | "Hiệu suất sử dụng FLOPs của mô hình" | FLOPs thực tế đạt được / đỉnh lý thuyết của phần cứng; 40% cho dense và 30% cho MoE là mức rất ấn tượng vào năm 2026. |
| Mất mát kiểm định (Val loss) | "Mất mát trên tập dữ liệu giữ lại" | Cross-entropy tính trên dữ liệu mô hình chưa từng thấy; công cụ phát hiện quá khớp. |

## Tài liệu tham khảo

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — tài liệu cài đặt có kèm chú giải kinh điển.
