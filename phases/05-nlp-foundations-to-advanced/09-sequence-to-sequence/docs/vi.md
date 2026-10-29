# Mô hình Chuỗi-sang-Chuỗi (Sequence-to-Sequence Models)

> Hai mạng RNN phối hợp để đóng vai một dịch giả. Nút thắt cổ chai mà chúng gặp phải chính là lý do cơ chế chú ý (attention) ra đời.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs cho Văn bản), Phase 3 · 11 (PyTorch Cơ bản)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Bài toán phân loại ánh xạ một chuỗi có độ dài thay đổi thành một nhãn đơn lẻ. Dịch thuật ánh xạ một chuỗi có độ dài thay đổi thành một chuỗi khác cũng có độ dài thay đổi. Đầu vào và đầu ra nằm ở hai không gian từ vựng khác nhau, có thể là hai ngôn ngữ hoàn toàn khác biệt, và không có bất kỳ đảm bảo nào về sự tương đồng về độ dài câu.

Kiến trúc seq2seq (Sutskever, Vinyals, Le, 2014) đã giải quyết vấn đề này bằng một công thức đơn giản đến bất ngờ: Hai mạng RNN. Một mạng đọc câu nguồn và tạo ra một vector ngữ cảnh có kích thước cố định. Mạng còn lại đọc vector đó và sinh ra câu đích từng token một. Vẫn là những dòng mã bạn đã viết ở bài học 08, chỉ ghép nối lại theo một cách thức khác.

Đây là kiến trúc rất đáng học vì hai lý do: Thứ nhất, nút thắt cổ chai vector ngữ cảnh (context-vector bottleneck) là trường hợp thất bại mang tính giáo khoa bổ ích nhất trong lịch sử NLP. Nó là động lực thúc đẩy mọi ưu điểm vượt trội của Attention và Transformer. Thứ hai, các kỹ thuật huấn luyện (teacher forcing, scheduled sampling, tìm kiếm chùm - beam search khi suy luận) vẫn được áp dụng trên mọi hệ thống tạo sinh hiện đại bao gồm cả các mô hình ngôn ngữ lớn (LLM).

## Khái niệm (The Concept)

**Bộ mã hóa (Encoder).** Một mạng RNN đọc câu nguồn. Trạng thái ẩn cuối cùng của nó chính là **vector ngữ cảnh (context vector)** — một bản tóm tắt có kích thước cố định nén toàn bộ thông tin đầu vào. Về lý thuyết, nó giữ lại mọi ý nghĩa của câu nguồn.

**Bộ giải mã (Decoder).** Một mạng RNN khác được khởi tạo từ chính vector ngữ cảnh đó. Tại mỗi bước, nó nhận token vừa sinh ra trước đó làm đầu vào và xuất ra phân phối xác suất trên toàn bộ từ vựng đích. Lấy mẫu (sample) hoặc dùng argmax để chọn token tiếp theo. Đưa token đó quay trở lại làm đầu vào cho bước sau. Lặp lại quá trình cho đến khi sinh ra token kết thúc `<EOS>` hoặc đạt độ dài tối đa.

**Huấn luyện:** Hàm mất mát Cross-entropy tại mỗi bước của bộ giải mã, lấy tổng trên toàn bộ chuỗi. Lan truyền ngược qua thời gian (BPTT) tiêu chuẩn xuyên suốt cả hai mạng nơ-ron.

**Teacher forcing.** Trong quá trình huấn luyện, đầu vào của decoder tại bước `t` chính là token *thực tế đúng* (ground-truth) tại vị trí `t-1`, chứ không phải dự đoán của chính decoder ở bước trước. Điều này giúp ổn định quá trình huấn luyện; nếu không có nó, các lỗi sai ở giai đoạn đầu sẽ tích lũy dây chuyền và mô hình sẽ không bao giờ học được. Khi suy luận thực tế, bạn bắt buộc phải dùng chính dự đoán của mô hình, do đó luôn tồn tại một khoảng cách phân phối giữa lúc huấn luyện và lúc suy luận. Khoảng cách này được gọi là **thiên kiến phơi nhiễm (exposure bias)**.

**Nút thắt cổ chai (The bottleneck).** Mọi thông tin mà encoder học được về câu nguồn đều phải bị ép vào một vector ngữ cảnh duy nhất có kích thước cố định. Các câu dài sẽ bị mất chi tiết nghiêm trọng. Các từ hiếm gặp bị mờ nhạt. Việc đảo trật tự từ (như "chat noir" trong tiếng Pháp sang "black cat" trong tiếng Anh) buộc phải học vẹt chứ không thể tính toán linh hoạt.

Cơ chế chú ý (Attention - bài học 10) giải quyết triệt để nút thắt này bằng cách cho phép decoder quan sát *mọi* trạng thái ẩn của encoder, thay vì chỉ dựa vào trạng thái cuối cùng.

```figure
lstm-gates
```

## Xây dựng thực tế (Build It)

### Bước 1: Bộ mã hóa (Encoder)

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` có kích thước `[batch, seq_len, hidden_dim]` — một trạng thái ẩn cho mỗi vị trí đầu vào. `hidden` có kích thước `[1, batch, hidden_dim]` — trạng thái tại bước cuối cùng. Bài học 08 nói "áp dụng pooling trên outputs cho tác vụ phân loại." Ở đây, chúng ta giữ lại trạng thái ẩn cuối cùng làm vector ngữ cảnh, và tạm thời bỏ qua outputs của từng bước.

### Bước 2: Bộ giải mã (Decoder)

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder được gọi từng bước một. Đầu vào: một batch gồm các token đơn lẻ và trạng thái ẩn hiện tại. Đầu ra: điểm logits từ vựng cho token tiếp theo và trạng thái ẩn đã được cập nhật.

### Bước 3: Vòng lặp huấn luyện với Teacher Forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Hai tham số quan trọng cần lưu ý: `ignore_index=0` bỏ qua hàm mất mát trên các token đệm (padding). `teacher_forcing_ratio` là xác suất sử dụng token chuẩn so với việc sử dụng dự đoán của mô hình tại mỗi bước. Bắt đầu ở mức 1.0 (toàn bộ teacher forcing) và giảm dần xuống ~0.5 trong suốt quá trình huấn luyện để thu hẹp khoảng cách thiên kiến phơi nhiễm.

### Bước 4: Vòng lặp suy luận tham lam (Greedy Inference)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Giải mã tham lam (Greedy decoding) chọn token có xác suất cao nhất tại mỗi bước. Nó có thể đi chệch hướng: một khi bạn đã chọn một token sai, bạn không thể quay lại sửa chữa. **Tìm kiếm chùm (Beam search)** duy trì danh sách top-`k` chuỗi ứng viên tiềm năng tại mỗi bước và chọn ra chuỗi hoàn chỉnh có tổng điểm cao nhất ở bước cuối cùng. Độ rộng chùm (beam width) từ 3 đến 5 là tiêu chuẩn phổ biến.

### Bước 5: Minh chứng thực nghiệm cho nút thắt cổ chai

Huấn luyện mô hình trên tác vụ sao chép đơn giản: chuỗi nguồn `[a, b, c, d, e]`, chuỗi đích `[a, b, c, d, e]`. Tăng dần độ dài chuỗi và quan sát độ chính xác:

```text
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Một trạng thái ẩn GRU đơn lẻ không thể ghi nhớ hoàn hảo không tổn hao một chuỗi đầu vào dài 40 token. Thông tin thực chất vẫn tồn tại ở mỗi bước của encoder, nhưng decoder lại chỉ được nhìn thấy trạng thái cuối cùng. Cơ chế chú ý sẽ giải quyết triệt để vấn đề này.

## Sử dụng trong thực tế (Use It)

PyTorch cung cấp `nn.Transformer` và các mẫu seq2seq dựa trên `nn.LSTM`. Thư viện `transformers` của Hugging Face phân phối các mô hình encoder-decoder hoàn chỉnh (BART, T5, mBART, NLLB) được huấn luyện sẵn trên hàng tỷ token.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Các mô hình encoder-decoder hiện đại đã thay thế RNN bằng Transformer. Khung kiến trúc cấp cao (encoder, decoder, sinh token tuần tự từng bước) hoàn toàn tương đồng với bài báo seq2seq năm 2014. Cơ chế hoạt động bên trong mỗi khối nơ-ron mới là điểm tạo nên khác biệt.

### Khi nào vẫn nên dùng seq2seq dựa trên RNN

Gần như không bao giờ đối với các dự án mới. Một số ngoại lệ cụ thể:

- Dịch thuật theo luồng trực tiếp (Streaming translation), nơi bạn tiếp nhận đầu vào từng token một với dung lượng bộ nhớ cố định nghiêm ngặt.
- Tạo sinh văn bản trên thiết bị biên (On-device text generation), nơi chi phí bộ nhớ của transformer là quá tải.
- Mục đích giảng dạy và học tập. Hiểu rõ nút thắt cổ chai của encoder-decoder là con đường ngắn nhất để hiểu tại sao Transformer giành chiến thắng.

### Thiên kiến phơi nhiễm và các biện pháp giảm thiểu

- **Scheduled sampling.** Giảm dần tỷ lệ teacher forcing trong quá trình huấn luyện để mô hình tập thích nghi với việc tự sửa lỗi sai của chính nó.
- **Huấn luyện rủi ro tối thiểu (Minimum risk training).** Huấn luyện trực tiếp trên điểm BLEU cấp độ câu thay vì hàm mất mát cross-entropy cấp độ token. Điều này bám sát hơn mục tiêu thực tế.
- **Tinh chỉnh học tăng cường (Reinforcement learning fine-tuning).** Thưởng cho bộ tạo sinh chuỗi dựa trên một chỉ số đánh giá chất lượng. Kỹ thuật này được áp dụng rộng rãi trong RLHF của các LLM hiện đại.

Cả ba phương pháp này đều áp dụng tương thích cho các mô hình tạo sinh dựa trên Transformer.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Thiết kế một pipeline sequence-to-sequence cho một bài toán cụ thể.
phase: 5
lesson: 09
---

Khi nhận được yêu cầu bài toán (dịch máy, tóm tắt văn bản, viết lại câu paraphrase, viết lại câu hỏi truy vấn), bạn đưa ra:

1. Kiến trúc. Transformer encoder-decoder huấn luyện sẵn (BART, T5, mBART, NLLB) là lựa chọn mặc định. seq2seq dựa trên RNN chỉ dùng cho các ràng buộc phần cứng đặc biệt.
2. Checkpoint khởi điểm. Nêu tên cụ thể (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Khớp checkpoint với bài toán và phạm vi ngôn ngữ hỗ trợ.
3. Chiến lược giải mã. Greedy cho đầu ra tất định, beam search (độ rộng 4-5) cho chất lượng cao, lấy mẫu kèm nhiệt độ (sampling with temperature) cho tính đa dạng. Giải thích lý do trong một câu.
4. Một phương thức lỗi cần kiểm tra trước khi phát hành. Thiên kiến phơi nhiễm biểu hiện dưới dạng trôi dạt thế hệ (generation drift) trên các đầu ra dài; lấy mẫu 20 đầu ra ở phân vị độ dài thứ 90 và trực tiếp kiểm tra bằng mắt.

Từ chối đề xuất huấn luyện seq2seq từ đầu nếu có dưới một triệu cặp mẫu song ngữ/song hành. Cảnh báo bất kỳ pipeline nào dùng giải mã tham lam (greedy decoding) cho nội dung đối diện người dùng là thiếu ổn định (dễ bị lặp từ và rơi vào vòng lặp vô tận).
```

## Bài tập (Exercises)

1. **Dễ.** Cài đặt tác vụ sao chép đồ chơi (toy copy task). Huấn luyện một mô hình GRU seq2seq trên các cặp đầu vào - đầu ra trong đó chuỗi đích trùng chuỗi nguồn. Đo độ chính xác tại các độ dài 5, 10, 20. Tái hiện lại hiện tượng nút thắt cổ chai.
2. **Trung bình.** Bổ sung giải mã tìm kiếm chùm (beam search) với độ rộng beam bằng 3. Đo điểm BLEU trên một tập ngữ liệu song song nhỏ so với giải mã tham lam. Ghi nhận tài liệu các trường hợp beam search vượt trội (thường là ở các token cuối) và trường hợp không có sự khác biệt.
3. **Khó.** Tinh chỉnh (fine-tune) `facebook/bart-base` trên tập dữ liệu diễn giải lại (paraphrase) gồm 10.000 cặp câu. So sánh đầu ra giải mã beam-4 của mô hình đã tinh chỉnh với mô hình gốc trên tập dữ liệu kiểm thử. Báo cáo điểm BLEU và chọn lọc 10 ví dụ phân tích định tính.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Encoder | RNN đầu vào | Đọc câu nguồn. Tạo ra các trạng thái ẩn theo từng bước và một vector ngữ cảnh cuối cùng. |
| Decoder | RNN đầu ra | Được khởi tạo từ vector ngữ cảnh. Sinh các token đích tuần tự từng bước. |
| Vector ngữ cảnh (Context vector) | Bản tóm tắt | Trạng thái ẩn cuối cùng của encoder. Kích thước cố định. Nút thắt cổ chai mà attention giải quyết. |
| Teacher forcing | Dùng token thực | Đưa token đúng vào làm đầu vào tiếp theo khi huấn luyện. Giúp quá trình học ổn định. |
| Thiên kiến phơi nhiễm (Exposure bias) | Khoảng cách train/test | Mô hình chỉ quen học trên token chuẩn nên chưa bao giờ được rèn luyện cách tự sửa chữa lỗi sai của chính nó khi suy luận. |
| Beam search | Giải mã tìm kiếm chùm | Duy trì top-k chuỗi ứng viên khả dĩ nhất tại mỗi bước thay vì chọn vội một token xác suất cao nhất. |

## Đọc thêm (Further Reading)

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — bài báo gốc đặt nền móng cho seq2seq. Bốn trang súc tích.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — giới thiệu GRU và cấu trúc đóng khung encoder-decoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — bài báo khai sinh cơ chế Attention. Rất nên đọc ngay sau bài học này.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — hướng dẫn lập trình thực chiến cho seq2seq + attention.

