# BERT — Mô hình hóa ngôn ngữ che mặt nạ (Masked Language Modeling)

> GPT dự đoán từ tiếp theo. BERT dự đoán một từ bị thiếu. Chỉ một câu khác biệt — nhưng định hình nửa thập kỷ của mọi thứ liên quan đến biểu diễn vector (embedding).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 phút

## Vấn đề (The Problem)

Vào năm 2018, mọi tác vụ xử lý ngôn ngữ tự nhiên (NLP - natural language processing) — phân tích cảm xúc (sentiment analysis), nhận dạng thực thể có tên (NER - named entity recognition), hỏi đáp (QA - question answering), suy luận kéo theo (entailment) — đều huấn luyện mô hình riêng từ đầu trên tập dữ liệu gắn nhãn riêng. Không hề có điểm kiểm tra (checkpoint) được tiền huấn luyện (pre-trained) nào "hiểu tiếng Anh" mà bạn có thể tinh chỉnh (fine-tune). ELMo (2018) đã chứng minh rằng bạn có thể tiền huấn luyện các biểu diễn từ theo ngữ cảnh (contextual embeddings) với LSTM hai chiều (bidirectional LSTM); mô hình này có mang lại cải thiện nhưng chưa tổng quát hóa tốt.

BERT (Devlin và cộng sự, 2018) đã đặt câu hỏi: điều gì sẽ xảy ra nếu chúng ta lấy một bộ mã hóa transformer (transformer encoder), huấn luyện nó trên mọi câu văn trên internet, và buộc nó dự đoán các từ bị thiếu từ ngữ cảnh ở cả hai phía? Sau đó, bạn chỉ cần tinh chỉnh một đầu dự đoán (head) duy nhất cho tác vụ hạ nguồn (downstream task) của mình. Hiệu quả tham số là một bước ngoặt mang tính cách mạng.

Kết quả là: trong vòng 18 tháng, BERT và các biến thể của nó (RoBERTa, ALBERT, ELECTRA) đã thống trị mọi bảng xếp hạng NLP từng tồn tại. Đến năm 2020, mọi công cụ tìm kiếm, đường ống kiểm duyệt nội dung (content moderation pipeline), và hệ thống tìm kiếm ngữ nghĩa (semantic search) trên thế giới đều có BERT bên trong.

Đến năm 2026, các mô hình chỉ gồm bộ mã hóa (encoder-only models) vẫn là công cụ phù hợp nhất cho các bài toán phân loại (classification), truy xuất (retrieval), và trích xuất có cấu trúc (structured extraction) — chúng xử lý nhanh hơn từ 5–10 lần trên mỗi token so với bộ giải mã (decoders) và các vector nhúng (embeddings) của chúng là xương sống của mọi ngăn xếp truy xuất hiện đại. ModernBERT (tháng 12 năm 2024) đã nâng tầm kiến trúc này lên ngữ cảnh 8K với Flash Attention + RoPE + GeGLU.

## Khái niệm (The Concept)

![Mô hình hóa ngôn ngữ che mặt nạ: chọn token, che lại, dự đoán token gốc](../assets/bert-mlm.svg)

### Tín hiệu huấn luyện (The training signal)

Xét một câu: `the quick brown fox jumps over the lazy dog`.

Che ngẫu nhiên 15% số token:

```text
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Huấn luyện mô hình để dự đoán các token ban đầu tại các vị trí bị che mặt nạ. Vì bộ mã hóa là hai chiều (bidirectional), việc dự đoán `[MASK]` tại vị trí 1 có thể sử dụng `brown fox jumps` tại các vị trí từ 2 trở đi. Đó là điều mà GPT không thể làm được.

### Quy tắc che (mask) của BERT (The BERT mask rules)

Trong số 15% token được chọn để dự đoán:

- 80% được thay thế bằng `[MASK]`.
- 10% được thay thế bằng một token ngẫu nhiên.
- 10% được giữ nguyên không đổi.

Tại sao không phải luôn luôn là `[MASK]`? Bởi vì `[MASK]` không bao giờ xuất hiện trong thời gian suy luận (inference time). Việc huấn luyện mô hình kỳ vọng gặp `[MASK]` ở 100% vị trí bị che sẽ tạo ra sự lệch phân phối (distribution shift) giữa giai đoạn tiền huấn luyện và tinh chỉnh. Tỷ lệ 10% ngẫu nhiên + 10% giữ nguyên giúp mô hình luôn học được các biểu diễn trung thực và thích ứng tốt.

### Dự đoán câu tiếp theo (Next Sentence Prediction - NSP) — và lý do bị loại bỏ

BERT nguyên bản cũng được huấn luyện trên tác vụ NSP: cho trước hai câu A và B, dự đoán xem B có phải là câu tiếp theo của A hay không. RoBERTa (2019) đã thực hiện nghiên cứu loại trừ (ablation study) và chỉ ra rằng NSP gây hại chứ không giúp ích. Các bộ mã hóa hiện đại đều bỏ qua tác vụ này.

### Những thay đổi vào năm 2026: ModernBERT (What changed in 2026: ModernBERT)

Bài báo ModernBERT năm 2024 đã tái cấu trúc khối transformer bằng các thành phần cơ bản (primitives) của năm 2026:

| Thành phần (Component) | BERT nguyên bản (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Mã hóa vị trí (Positional) | Tuyệt đối học được (Learned absolute) | RoPE |
| Hàm kích hoạt (Activation) | GELU | GeGLU |
| Chuẩn hóa (Normalization) | LayerNorm | Pre-norm RMSNorm |
| Cơ chế chú ý (Attention) | Chú ý dày đặc đầy đủ (Full dense) | Xen kẽ cục bộ (128) + toàn cục (Alternating local (128) + global) |
| Độ dài ngữ cảnh (Context length) | 512 | 8192 |
| Bộ tách từ (Tokenizer) | WordPiece | BPE |

Và không giống như ngăn xếp công nghệ năm 2018, ModernBERT được thiết kế tối ưu gốc cho Flash Attention. Suy luận nhanh hơn 2–3 lần ở độ dài chuỗi 8K so với DeBERTa-v3 mà lại đạt điểm GLUE cao hơn.

### Các trường hợp sử dụng vẫn chọn encoder vào năm 2026 (Use cases that still pick an encoder in 2026)

| Tác vụ (Task) | Lý do encoder vượt trội hơn decoder |
|------|---------------------------|
| Embedding cho truy xuất / tìm kiếm ngữ nghĩa | Ngữ cảnh hai chiều = chất lượng embedding tốt hơn trên mỗi token |
| Phân loại (cảm xúc, ý định, tính độc hại) | Một lượt truyền xuôi (forward pass); không có chi phí tạo sinh (generation overhead) |
| NER / gán nhãn token | Đầu ra theo từng vị trí, có tính chất hai chiều tự nhiên |
| Suy luận kéo theo zero-shot (NLI) | Đầu phân loại được đặt ngay phía trên encoder |
| Bộ xếp hạng lại (Reranker) cho RAG | Chấm điểm cross-encoder, nhanh hơn gấp 10 lần so với reranker dùng LLM |

```figure
transformer-residual
```

## Tự xây dựng (Build It)

### Bước 1: logic tạo mặt nạ (masking logic)

Xem `code/main.py`. Hàm `create_mlm_batch` nhận một danh sách token ID, kích thước từ vựng (vocab size), và xác suất che (mask probability). Trả về các input ID (đã áp dụng mặt nạ) và nhãn mục tiêu (labels, chỉ tại các vị trí bị che, các vị trí khác mang giá trị -100 — quy ước chỉ số bị bỏ qua trong tính toán loss của PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Bước 2: chạy dự đoán MLM trên một tập ngữ liệu nhỏ (tiny corpus)

Huấn luyện một encoder 2 lớp + đầu MLM trên tập từ vựng gồm 20 từ, 200 câu. Không tính đạo hàm gradient — chúng ta thực hiện các kiểm tra tính hợp lý (sanity check) ở lượt truyền xuôi. Việc huấn luyện đầy đủ cần dùng đến PyTorch.

### Bước 3: so sánh các kiểu che mặt nạ

Chỉ ra cách quy tắc 3 nhánh giúp mô hình vẫn sử dụng được khi không có token `[MASK]`. Dự đoán trên một câu không bị che và một câu bị che. Cả hai trường hợp đều phải tạo ra phân phối token hợp lý vì mô hình đã thấy cả hai mẫu này trong quá trình huấn luyện.

### Bước 4: tinh chỉnh đầu dự đoán (fine-tune head)

Thay thế đầu MLM bằng một đầu phân loại (classification head) trên một tập dữ liệu phân tích cảm xúc đồ chơi (toy dataset). Chỉ huấn luyện phần đầu này; phần encoder được đóng băng (frozen). Đây là khuôn mẫu mà mọi ứng dụng BERT đều tuân theo.

## Sử dụng trong thực tế (Use It)

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Các mô hình embedding là BERT đã qua tinh chỉnh.** Các mô hình thuộc thư viện `sentence-transformers` như `all-MiniLM-L6-v2` thực chất là các mô hình BERT được huấn luyện với hàm mất mát tương phản (contrastive loss). Kiến trúc encoder vẫn giữ nguyên, chỉ có hàm mất mát là thay đổi.

**Các bộ xếp hạng lại dạng cross-encoder cũng là BERT được tinh chỉnh.** Phân loại theo cặp trên chuỗi `[CLS] query [SEP] doc [SEP]`. Cơ chế chú ý hai chiều giữa câu truy vấn (query) và tài liệu (document) chính là yếu tố mang lại cho cross-encoder lợi thế vượt trội về chất lượng so với bi-encoder.

**Khi nào không nên chọn BERT vào năm 2026.** Bất cứ tác vụ nào mang tính tạo sinh (generative). Encoder không có cơ chế hợp lý để sinh token một cách tự hồi quy (autoregressively). Ngoài ra: bất kỳ mô hình nào dưới 1 tỷ tham số (1B params) mà ở đó một decoder nhỏ có thể đạt chất lượng tương đương với độ linh hoạt cao hơn (như Phi-3-Mini, Qwen2-1.5B).

## Đưa vào triển khai (Ship It)

Xem `outputs/skill-bert-finetuner.md`. Kỹ năng này định phạm vi cho một quy trình tinh chỉnh BERT (lựa chọn mô hình nền tảng backbone, đặc tả đầu phân loại, chuẩn bị dữ liệu, đánh giá, điều kiện dừng sớm) cho một tác vụ phân loại hoặc trích xuất mới.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py` và in phân phối mặt nạ trên 10.000 token. Xác nhận khoảng 15% được chọn, và trong số đó khoảng 80% trở thành `[MASK]`.
2. **Trung bình.** Triển khai kỹ thuật che toàn bộ từ (whole-word masking): nếu một từ được tách thành nhiều từ con (subwords), hãy che tất cả các từ con đó cùng nhau hoặc không che từ nào. Đo lường xem điều này có cải thiện độ chính xác MLM trên tập ngữ liệu 500 câu hay không.
3. **Khó.** Huấn luyện một mô hình BERT thu nhỏ (2 lớp, d=64) trên 10.000 câu từ một tập dữ liệu công khai. Tinh chỉnh token `[CLS]` cho bài toán phân loại cảm xúc SST-2. So sánh với một mô hình cơ sở chỉ gồm decoder (decoder-only baseline) có cùng số lượng tham số — mô hình nào sẽ giành chiến thắng?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" (Mô hình hóa ngôn ngữ che mặt nạ) | Tín hiệu huấn luyện: thay thế ngẫu nhiên 15% token bằng `[MASK]`, dự đoán các token gốc. |
| Bidirectional | "Nhìn theo cả hai chiều" | Cơ chế chú ý của encoder không có mặt nạ nhân quả (causal mask) — mọi vị trí đều nhìn thấy mọi vị trí khác. |
| `[CLS]` | "Pooler token" | Một token đặc biệt được thêm vào đầu mỗi chuỗi; vector nhúng cuối cùng của nó được dùng làm biểu diễn cho toàn bộ câu. |
| `[SEP]` | "Segment separator" (Ký tự phân cách phân đoạn) | Phân cách các cặp chuỗi (ví dụ: query/doc, câu A/B). |
| NSP | "Next sentence prediction" (Dự đoán câu tiếp theo) | Tác vụ tiền huấn luyện thứ hai của BERT; đã được chứng minh là không hữu ích trong RoBERTa và bị loại bỏ sau năm 2019. |
| Fine-tuning | "Tinh chỉnh cho tác vụ" | Giữ cho encoder phần lớn bị đóng băng; chỉ huấn luyện một đầu nhỏ gắn phía trên cho tác vụ hạ nguồn. |
| Cross-encoder | "Bộ xếp hạng lại (reranker)" | Một mô hình BERT nhận cả query và doc làm đầu vào, cho ra điểm số mức độ liên quan. |
| ModernBERT | "Bản nâng cấp năm 2024" | Encoder được tái thiết kế với RoPE, RMSNorm, GeGLU, cơ chế chú ý xen kẽ cục bộ/toàn cục, độ dài ngữ cảnh 8K. |

## Tài liệu đọc thêm (Further Reading)

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — bài báo gốc.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — phương pháp huấn luyện BERT tối ưu; loại bỏ hoàn toàn NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — phát hiện token bị thay thế (replaced-token detection) vượt trội hơn MLM ở cùng mức chi phí tính toán.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — bài báo về ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — tài liệu tham chiếu chuẩn cho cấu trúc encoder.
