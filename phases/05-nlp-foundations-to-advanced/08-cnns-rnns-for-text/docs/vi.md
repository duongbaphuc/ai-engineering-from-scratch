# Mạng CNN và RNN cho Văn bản (CNNs and RNNs for Text)

> Tích chập học các n-gram. Hồi quy ghi nhớ ngữ cảnh. Cả hai đều đã bị cơ chế chú ý (attention) thay thế, nhưng vẫn đóng vai trò quan trọng trên phần cứng bị hạn chế.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Cơ bản), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Tích chập từ đầu)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

TF-IDF và Word2Vec tạo ra các vector phẳng hoàn toàn bỏ qua trật tự từ. Một bộ phân loại xây dựng trên chúng không thể phân biệt được `dog bites man` (chó cắn người) với `man bites dog` (người cắn chó). Trong ngôn ngữ, trật tự từ thường mang tín hiệu quyết định.

Hai họ kiến trúc đã lấp đầy khoảng trống đó trước khi Transformer xuất hiện:

**Mạng tích chập cho văn bản (TextCNN).** Áp dụng các phép tích chập 1D (1D convolutions) trên chuỗi các vector nhúng từ (word embeddings). Một bộ lọc (filter) có kích thước độ rộng 3 đóng vai trò như một bộ phát hiện trigram có thể học được: nó trượt qua ba từ liên tiếp và xuất ra một điểm số kích hoạt. Xếp chồng nhiều độ rộng bộ lọc khác nhau (2, 3, 4, 5) để phát hiện các mẫu hình đa quy mô (multi-scale patterns). Áp dụng max-pooling để đưa về biểu diễn có kích thước cố định. Thiết kế phẳng, tính toán song song, tốc độ cực nhanh.

**Mạng hồi quy (RNN, LSTM, GRU).** Xử lý từng token một theo thời gian, duy trì một trạng thái ẩn (hidden state) để truyền dẫn thông tin về phía trước. Xử lý tuần tự, có khả năng ghi nhớ, xử lý linh hoạt các độ dài đầu vào khác nhau. Chi phối toàn bộ mảng mô hình hóa chuỗi từ năm 2014 đến 2017, trước khi cơ chế chú ý bùng nổ.

Bài học này xây dựng cả hai kiến trúc, sau đó chỉ rõ điểm hạn chế cốt tử đã thúc đẩy sự ra đời của cơ chế chú ý (attention).

## Khái niệm (The Concept)

**TextCNN** (Kim, 2014). Các token được biến đổi thành vector nhúng. Một bộ lọc tích chập 1D độ rộng `k` trượt qua các `k`-gram nhúng liên tiếp, tạo ra một bản đồ đặc trưng (feature map). Phép max-pooling toàn cục (global max-pooling) trên bản đồ đó sẽ chọn ra giá trị kích hoạt mạnh nhất. Nối các đầu ra max-pool từ nhiều độ rộng bộ lọc khác nhau. Đưa qua đầu phân loại (classifier head).

Tại sao TextCNN hiệu quả? Mỗi bộ lọc là một bộ phát hiện n-gram có thể học được. Phép max-pooling mang tính bất biến theo vị trí (position-invariant), nên cụm từ "not good" sẽ kích hoạt cùng một đặc trưng dù nó nằm ở đầu hay giữa một câu đánh giá. Ba độ rộng bộ lọc với mỗi loại 100 bộ lọc sẽ cung cấp cho bạn 300 bộ dò n-gram được học tự động. Quá trình huấn luyện diễn ra song song hoàn toàn; không có sự phụ thuộc tuần tự.

**RNN.** Tại mỗi bước thời gian `t`, trạng thái ẩn `h_t = f(W * x_t + U * h_{t-1} + b)`. Các ma trận trọng số `W`, `U`, và vector độ lệch `b` được chia sẻ qua tất cả các bước thời gian. Trạng thái ẩn tại thời điểm cuối `T` là bản tóm tắt biểu diễn toàn bộ tiền tố trước đó. Đối với bài toán phân loại, ta tổng hợp (pool) qua toàn bộ các trạng thái `h_1 ... h_T` (bằng max, mean hoặc lấy trạng thái cuối cùng).

Mạng RNN thuần túy gặp phải vấn đề triệt tiêu đạo hàm (vanishing gradient). **LSTM** bổ sung các cổng (gates) để quyết định thông tin nào cần quên, thông tin nào cần lưu trữ và thông tin nào cần xuất ra, giúp ổn định dòng chảy đạo hàm qua các chuỗi dài. **GRU** đơn giản hóa LSTM thành hai cổng; mang lại hiệu năng tương đương với ít tham số hơn.

**Mạng hồi quy hai chiều (Bidirectional RNNs)** chạy một RNN theo chiều xuôi và một RNN theo chiều ngược, rồi ghép nối các trạng thái ẩn lại với nhau. Biểu diễn của mỗi token sẽ nhìn thấy cả ngữ cảnh bên trái lẫn bên phải. Điều này tối quan trọng cho các tác vụ gán nhãn chuỗi (sequence tagging).

```figure
rnn-unroll
```

## Xây dựng thực tế (Build It)

### Bước 1: TextCNN trong PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

Lệnh `transpose(1, 2)` đổi chiều tensor từ `[batch, seq_len, embed_dim]` thành `[batch, embed_dim, seq_len]` vì `nn.Conv1d` coi trục ở giữa là số kênh (channels). Đầu ra sau khi gom nhóm (pooled) có kích thước cố định bất kể độ dài chuỗi đầu vào.

### Bước 2: Bộ phân loại LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Áp dụng Max-pool trên toàn bộ chuỗi chứ không chỉ lấy trạng thái cuối cùng (last-state). Trong tác vụ phân loại, max-pooling thường vượt trội hơn việc chỉ lấy trạng thái ẩn cuối cùng vì thông tin ở cuối chuỗi dài có xu hướng lấn át toàn bộ trạng thái cuối.

### Bước 3: Mô phỏng trực giác về triệt tiêu đạo hàm (Vanishing Gradient)

Mạng RNN thuần túy không có cơ chế đóng/mở cổng không thể học được các mối phụ thuộc tầm xa. Xét một bài toán đơn giản: dự đoán xem token `A` có xuất hiện ở bất kỳ đâu trong câu hay không. Nếu `A` nằm ở vị trí số 1 và chuỗi dài 100 token, gradient từ hàm mất mát phải lan truyền ngược qua 99 phép nhân liên tiếp với trọng số hồi quy. Nếu trọng số nhỏ hơn 1, gradient sẽ tiêu biến hoàn toàn về 0. Nếu lớn hơn 1, gradient sẽ bùng nổ (exploding).

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# Với trọng số = 0.9 qua 100 bước:
#   0.9 ^ 100 ≈ 2.7e-5
# Đạo hàm gradient từ bước 100 trở về bước 1 thực tế bằng không.
```

LSTM giải quyết vấn đề này bằng một **trạng thái ô (cell state)** chạy dọc qua toàn bộ mạng lưới với hầu hết các tương tác là phép cộng (cổng quên điều chỉnh theo phép nhân, nhưng dòng chảy gradient vẫn di chuyển thuận lợi trên "đường cao tốc" này). GRU đạt hiệu quả tương tự với ít tham số hơn. Cả hai cho phép huấn luyện ổn định qua các chuỗi dài hơn 100 bước.

### Bước 4: Tại sao điều này vẫn chưa đủ

Ba nút thắt vẫn tồn tại ngay cả với LSTM:

1. **Nút thắt tuần tự (Sequential bottleneck).** Huấn luyện RNN trên chuỗi dài 1.000 token đòi hỏi 1.000 bước lan truyền tiến/lùi nối tiếp nhau. Không thể song song hóa qua chiều thời gian.
2. **Vector ngữ cảnh kích thước cố định trong mô hình Encoder-Decoder.** Bộ giải mã (decoder) chỉ nhìn thấy trạng thái ẩn cuối cùng của bộ mã hóa (encoder), vốn là bản nén thô của toàn bộ câu đầu vào. Câu dài sẽ bị mất mát chi tiết nghiêm trọng. Bài học 09 sẽ đi sâu vào vấn đề này.
3. **Trần hiệu năng với các phụ thuộc cách xa nhau.** Dù LSTM vượt trội hơn RNN thuần, chúng vẫn chật vật trong việc truyền tải thông tin cụ thể qua khoảng cách hơn 200 bước.

Cơ chế chú ý (Attention) đã giải quyết triệt để cả ba vấn đề trên. Transformer đã loại bỏ hoàn toàn tính hồi quy tuần tự. Bài học 10 là bước ngoặt bản lề đó.

## Sử dụng trong thực tế (Use It)

Các lớp `nn.LSTM`, `nn.GRU`, và `nn.Conv1d` của PyTorch hoàn toàn sẵn sàng cho môi trường sản xuất. Quy trình huấn luyện tuân theo chuẩn mực thông thường.

Hugging Face cung cấp các mô hình nhúng huấn luyện sẵn mà bạn có thể cắm trực tiếp làm tầng đầu vào:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Bảng kiểm tra khi nào nên dùng các kiến trúc này:

- **Suy luận trên thiết bị biên / biên mạng (Edge / on-device inference).** TextCNN kết hợp nhúng từ GloVe nhỏ hơn Transformer từ 10 đến 100 lần. Nếu mục tiêu triển khai là điện thoại di động hoặc chip nhúng, đây là ngăn xếp tối ưu.
- **Phân loại dòng dữ liệu liên tục / thời gian thực (Streaming / online classification).** RNN xử lý từng token một khi dữ liệu truyền đến; trong khi Transformer đòi hỏi toàn bộ chuỗi phải có sẵn. Với văn bản đến liên tục theo thời gian thực, LSTM vẫn chiếm ưu thế.
- **Mô hình nhỏ gọn cho baseline.** Lặp nhanh ý tưởng trên tác vụ mới. Huấn luyện một TextCNN chỉ mất 5 phút trên CPU.
- **Gán nhãn chuỗi với dữ liệu hạn chế.** BiLSTM-CRF (bài học 06) vẫn là kiến trúc NER chuẩn sản xuất cho tập dữ liệu có từ 1.000 đến 10.000 câu gán nhãn.

Đối với tất cả các bài toán còn lại, Transformer là lựa chọn hàng đầu.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Lựa chọn kiến trúc mã hóa văn bản phù hợp cho một tập ràng buộc kỹ thuật.
phase: 5
lesson: 08
---

Dựa trên các ràng buộc kỹ thuật (tác vụ, khối lượng dữ liệu, ngân sách độ trễ, mục tiêu triển khai, ngân sách tính toán), bạn đưa ra:

1. Kiến trúc bộ mã hóa: TextCNN, BiLSTM, BiLSTM-CRF, fine-tune transformer, hoặc "dùng transformer huấn luyện sẵn làm bộ mã hóa đóng băng (frozen encoder) + đầu phân loại nhỏ".
2. Đầu vào embedding: khởi tạo ngẫu nhiên, GloVe / fastText đóng băng, hoặc vector nhúng theo ngữ cảnh từ transformer.
3. Công thức huấn luyện trong 5 dòng: bộ tối ưu (optimizer), tốc độ học (learning rate), kích thước batch, số epoch, kỹ thuật điều chuẩn.
4. Một tín hiệu giám sát: Đối với mô hình RNN/CNN: sự vắng mặt của cơ chế attention đồng nghĩa chúng sẽ bỏ sót các phụ thuộc tầm xa; kiểm tra độ chính xác phân theo độ dài câu. Đối với transformer: nguy cơ đổ vỡ khi fine-tune nếu learning rate quá cao; kiểm tra loss trên tập huấn luyện.

Từ chối đề xuất fine-tune transformer khi dữ liệu có dưới ~500 mẫu gán nhãn mà chưa chứng minh được baseline TextCNN / BiLSTM đã chạm ngưỡng bão hòa. Đánh dấu việc triển khai trên thiết bị biên là bài toán cần ưu tiên tối ưu kiến trúc trước mọi thứ khác.
```

## Bài tập (Exercises)

1. **Dễ.** Huấn luyện một TextCNN trên tập dữ liệu mẫu 3 lớp (bạn tự tạo dữ liệu). Kiểm chứng rằng việc kết hợp các độ rộng bộ lọc (2, 3, 4) mang lại điểm F1 trung bình cao hơn việc chỉ dùng một độ rộng duy nhất (3).
2. **Trung bình.** Cài đặt max-pool, mean-pool, và last-state pooling cho bộ phân loại LSTM. So sánh hiệu năng trên một tập dữ liệu nhỏ; ghi chép lại cơ chế pooling nào chiến thắng và đưa ra giả thuyết giải thích lý do.
3. **Khó.** Xây dựng bộ gán nhãn NER BiLSTM-CRF (kết hợp bài học 06 và bài học này). Huấn luyện trên CoNLL-2003. So sánh với baseline chỉ dùng CRF từ bài học 06 và mô hình BERT fine-tune. Báo cáo thời gian huấn luyện, bộ nhớ tiêu thụ và điểm F1.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| TextCNN | CNN cho văn bản | Xếp chồng các tích chập 1D trên vector nhúng từ kết hợp max-pool toàn cục. Kim (2014). |
| RNN | Mạng hồi quy | Trạng thái ẩn được cập nhật tại mỗi bước thời gian: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | RNN có cổng | Bổ sung các cổng input / forget / output + trạng thái ô (cell state). Huấn luyện ổn định qua chuỗi dài. |
| GRU | LSTM đơn giản hóa | Hai cổng thay vì ba cổng. Độ chính xác tương đương, ít tham số hơn. |
| Hai chiều (Bidirectional) | Cả hai hướng | Nối ghép RNN chiều xuôi + RNN chiều ngược. Mỗi token nhìn thấy cả hai phía ngữ cảnh. |
| Triệt tiêu đạo hàm (Vanishing gradient) | Tín hiệu huấn luyện bị triệt tiêu | Phép nhân lặp lại nhiều lần với trọng số < 1 trong RNN thuần túy khiến gradient ở các bước đầu tiên suy giảm về 0. |

## Đọc thêm (Further Reading)

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — bài báo gốc về TextCNN. Tám trang súc tích, rất dễ hiểu.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — bài báo gốc về LSTM. Rất rõ ràng và mạch lạc.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — các sơ đồ trực quan hóa xuất sắc giúp bất kỳ ai cũng có thể hiểu rõ cơ chế hoạt động của LSTM.

