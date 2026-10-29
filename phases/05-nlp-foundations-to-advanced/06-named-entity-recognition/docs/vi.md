# Nhận dạng Thực thể có Tên (Named Entity Recognition - NER)

> Trích xuất các danh xưng và tên riêng. Nghe có vẻ dễ dàng cho đến khi bạn phải đối mặt với ranh giới mơ hồ, thực thể lồng nhau và thuật ngữ chuyên ngành.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

"Apple sued Google over its iPhone search deal in the US." Có năm thực thể: Apple (ORG - tổ chức), Google (ORG), iPhone (PRODUCT - sản phẩm), search deal (có thể là một thực thể nghiệp vụ), US (GPE - thực thể địa chính trị). Một hệ thống NER tốt sẽ trích xuất tất cả chúng kèm phân loại chính xác. Một hệ thống kém sẽ bỏ sót iPhone, nhầm lẫn Apple (trái táo) với Apple (công ty), và gán nhãn "US" là PERSON (con người).

NER là cỗ máy lao động âm thầm đằng sau mọi đường ống trích xuất có cấu trúc (structured extraction pipeline): Phân tích CV/hồ sơ xin việc, quét nhật ký tuân thủ, ẩn danh hóa hồ sơ y tế, hiểu câu truy vấn tìm kiếm, tạo cơ sở thực tế (grounding) cho phản hồi của chatbot, trích xuất hợp đồng pháp lý. Bạn hiếm khi nhìn thấy trực tiếp nó; nhưng bạn luôn phụ thuộc vào nó.

Bài học này đi qua lộ trình cổ điển (dựa trên luật, HMM, CRF) tiến tới hiện đại (BiLSTM-CRF, rồi đến Transformer). Mỗi bước giải quyết một hạn chế cụ thể của bước trước đó. Quy luật tiến hóa đó chính là bài học lớn nhất.

## Khái niệm (The Concept)

**Gán nhãn BIO (BIO tagging)** (hoặc BILOU) chuyển đổi bài toán trích xuất thực thể thành bài toán gán nhãn chuỗi (sequence labeling). Mỗi token được gắn nhãn `B-TYPE` (Beginning - bắt đầu thực thể), `I-TYPE` (Inside - nằm bên trong thực thể), hoặc `O` (Outside - nằm ngoài bất kỳ thực thể nào).

```text
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Các thực thể gồm nhiều token sẽ tạo thành chuỗi liên kết: `New B-GPE`, `York I-GPE`, `City I-GPE`. Một mô hình hiểu được BIO có thể trích xuất các đoạn văn bản (spans) có độ dài tùy ý.

Tiến trình phát triển kiến trúc:

- **Dựa trên luật (Rule-based).** Biểu thức chính quy (Regex) + tra cứu từ điển địa danh/thực thể (gazetteer lookups). Độ chính xác (precision) cao đối với các thực thể đã biết, nhưng độ bao phủ (coverage) bằng 0 đối với các thực thể mới.
- **HMM.** Hidden Markov Model (Mô hình Markov ẩn). Xác suất phát xạ (emission probability) của token theo nhãn, xác suất chuyển trạng thái (transition probability) giữa các nhãn. Giải mã bằng thuật toán Viterbi. Được huấn luyện trên dữ liệu gán nhãn.
- **CRF.** Conditional Random Field (Trường ngẫu nhiên có điều kiện). Giống HMM nhưng mang tính phân biệt (discriminative), cho phép bạn kết hợp các đặc trưng tùy ý (hình dạng từ, viết hoa, các từ lân cận). Vẫn là cỗ máy sản xuất cổ điển hiệu quả năm 2026 cho các hệ thống triển khai tài nguyên thấp.
- **BiLSTM-CRF.** Sử dụng đặc trưng học sâu thay vì tạo thủ công bằng tay. LSTM đọc câu theo cả hai chiều, tầng CRF ở trên cùng đảm bảo tính nhất quán của chuỗi nhãn.
- **Dựa trên Transformer.** Tinh chỉnh (fine-tune) BERT với một đầu phân loại token (token-classification head). Đạt độ chính xác cao nhất, nhưng tốn chi phí tính toán nhất.

```figure
ner-bio-tagging
```

## Xây dựng thực tế (Build It)

### Bước 1: Các hàm bổ trợ gán nhãn BIO

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Bước 2: Thiết kế đặc trưng thủ công (Hand-crafted features)

Đối với NER cổ điển (phi mạng nơ-ron), kỹ thuật trích chọn đặc trưng là yếu tố quyết định. Một số đặc trưng hữu ích:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` trả về `xXxxxx`. `word_shape("USA-2024")` trả về `XXX-dddd`. Quy luật viết hoa mang lại tín hiệu rất mạnh đối với các danh từ riêng.

### Bước 3: Mô hình cơ sở đơn giản dựa trên luật + từ điển

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Các tập từ điển thực tế trong sản xuất có hàng triệu mục được thu thập từ Wikipedia và DBpedia. Độ bao phủ là rất tốt. Tuy nhiên khả năng khử nhập nhằng (disambiguation - phân biệt `Apple` là công ty hay quả táo) thì rất kém. Đó là lý do tại sao các mô hình thống kê chiếm ưu thế hoàn toàn.

### Bước 4: Bước mô hình CRF (Phác thảo tổng quan)

Cài đặt đầy đủ CRF từ đầu trong 50 dòng mã không mang nhiều ý nghĩa sư phạm nếu thiếu nền tảng lý thuyết xác suất sâu. Sử dụng `sklearn-crfsuite` thay thế:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` và `c2` là các hệ số điều chuẩn L1 và L2. `all_possible_transitions=True` cho phép mô hình học được rằng các chuỗi không hợp lệ (ví dụ: `I-ORG` đứng ngay sau `O`) có xác suất cực thấp, đó là cách CRF ép buộc tính nhất quán BIO mà không cần bạn phải tự viết luật ràng buộc thủ công.

### Bước 5: Lợi ích của BiLSTM-CRF

Các đặc trưng giờ đây được học tự động. Đầu vào: vector nhúng từ (GloVe hoặc fastText). LSTM đọc văn bản từ trái sang phải và từ phải sang trái. Các trạng thái ẩn nối ghép (concatenated hidden states) được chuyển qua tầng xuất CRF. CRF vẫn đảm bảo tính nhất quán của chuỗi nhãn; trong khi LSTM thay thế các đặc trưng thủ công bằng các biểu diễn học được.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Đối với tầng CRF, sử dụng thư viện `torchcrf.CRF` (`pip install pytorch-crf`). Mức tăng độ chính xác so với CRF đặc trưng thủ công là có thể đo đạc được, nhưng thường nhỏ hơn bạn tưởng trừ khi bạn có trong tay hàng chục nghìn câu gán nhãn.

## Sử dụng trong thực tế (Use It)

spaCy cung cấp giải pháp NER cấp độ sản xuất ngay khi cài đặt.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```text
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Chú ý rằng `iPhone` bị gán nhãn nhầm thành `ORG` thay vì `PRODUCT` — mô hình nhỏ của spaCy có độ bao phủ thực thể sản phẩm khá yếu. Mô hình lớn (`en_core_web_lg`) làm tốt hơn nhiều. Mô hình transformer (`en_core_web_trf`) thậm chí còn vượt trội hơn nữa.

Sử dụng Hugging Face cho NER dựa trên BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```text
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` gom các token liên tiếp mang nhãn B-X, I-X thành một span duy nhất. Nếu không có tùy chọn này, bạn sẽ nhận được các nhãn ở cấp độ token và phải tự ghép nối thủ công.

### NER bằng LLM (Phương án năm 2026)

Zero-shot và few-shot NER bằng LLM hiện tại đã cạnh tranh sòng phẳng với các mô hình fine-tune trên nhiều lĩnh vực, và vượt trội rõ rệt khi dữ liệu gán nhãn khan hiếm.

- **Zero-shot prompting.** Cung cấp cho LLM danh sách các loại thực thể và cấu trúc lược đồ (schema) mẫu. Yêu cầu đầu ra định dạng JSON. Hoạt động hiệu quả ngay lập tức; độ chính xác ở mức khá đối với các lĩnh vực mới.
- **Prompting theo phong cách ZeroTuneBio.** Chia nhỏ tác vụ thành: trích xuất ứng viên → giải thích ý nghĩa ngữ cảnh → đánh giá phân loại → kiểm tra rà soát lại. Một chuỗi prompt nhiều giai đoạn (thay vì một lượt one-shot) nâng cao độ chính xác đáng kể trên dữ liệu NER y sinh (biomedical NER). Mô hình này cũng hiệu quả vượt bậc cho các lĩnh vực pháp lý, tài chính và khoa học.
- **Dynamic prompting kết hợp RAG.** Truy xuất các ví dụ gán nhãn tương đồng nhất từ một tập mẫu nhỏ cho mỗi lệnh gọi suy luận; xây dựng prompt few-shot linh hoạt ngay trong thời gian thực. Trên các chuẩn benchmark năm 2026, cách này nâng F1 của GPT-4 trong bài toán NER y sinh thêm 11-12% so với prompt tĩnh.
- **Chia tách theo từng loại thực thể (Per-entity-type decomposition).** Đối với tài liệu dài, một lệnh gọi duy nhất trích xuất tất cả các loại thực thể cùng lúc sẽ bị giảm độ nhạy (recall) khi văn bản dài ra. Hãy thực hiện từng lượt trích xuất riêng biệt cho mỗi loại thực thể. Chi phí suy luận cao hơn, nhưng độ chính xác tăng vượt trội. Đây là quy chuẩn vàng cho hồ sơ bệnh án và hợp đồng pháp lý.

Khuyến nghị sản xuất năm 2026: hãy bắt đầu với baseline LLM zero-shot trước khi bạn bắt tay thu thập dữ liệu huấn luyện. Thường thì điểm F1 đã đủ tốt để bạn không bao giờ cần phải fine-tune mô hình riêng.

### Trường hợp NER cổ điển vẫn chiến thắng

Ngay cả khi LLM rất sẵn có, NER cổ điển vẫn giành chiến thắng khi:

- Ngân sách độ trễ (latency budget) dưới 50ms.
- Bạn sở hữu hàng nghìn ví dụ gán nhãn chất lượng và cần độ chính xác F1 trên 98%.
- Miền nghiệp vụ có một ontology ổn định, nơi mô hình CRF hoặc BiLSTM huấn luyện sẵn chuyển giao rất tốt.
- Các quy định bảo mật / tuân thủ bắt buộc phải dùng mô hình cục bộ (on-premise), không có tính tạo sinh (non-generative).

### Các điểm gãy đổ thường gặp (Where it falls apart)

- **Trôi dạt miền dữ liệu (Domain shift).** NER huấn luyện trên tập CoNLL khi áp dụng vào hợp đồng pháp lý hoạt động kém hơn cả từ điển thông thường. Cần fine-tune trên đúng miền nghiệp vụ của bạn.
- **Thực thể lồng nhau (Nested entities).** "Bank of America Tower" vừa là một ORG vừa là một FACILITY (công trình). Chuẩn BIO tiêu chuẩn không thể biểu diễn các vùng chồng lấn. Bạn cần mô hình nested NER (nhiều lượt hoặc dựa trên span).
- **Thực thể dài (Long entities).** "United States Federal Deposit Insurance Corporation." Các mô hình cấp độ token đôi khi chia cắt cụm từ này. Hãy dùng `aggregation_strategy` hoặc hậu xử lý.
- **Các kiểu thực thể hiếm (Sparse types).** Các nhãn y tế như DRUG_BRAND, ADVERSE_EVENT, DOSE. Các mô hình tổng quát hoàn toàn mù tịt. Scispacy và BioBERT là điểm khởi đầu cho các trường hợp này.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Lựa chọn giải pháp NER phù hợp cho một bài toán trích xuất cụ thể.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Dựa trên mô tả bài toán (miền nghiệp vụ, tập nhãn, ngôn ngữ, độ trễ, khối lượng dữ liệu), đưa ra:

1. Phương pháp tiếp cận. Luật + từ điển, CRF, BiLSTM-CRF, hoặc fine-tune transformer.
2. Mô hình khởi điểm. Đặt tên cụ thể (ID mô hình spaCy, checkpoint ID trên Hugging Face, hoặc "tùy biến, huấn luyện từ đầu").
3. Chiến lược gán nhãn. BIO, BILOU, hoặc dựa trên span. Giải thích ngắn gọn trong một câu.
4. Đánh giá. Sử dụng `seqeval`. Luôn luôn báo cáo F1 cấp độ thực thể (không dùng cấp độ token).

Từ chối đề xuất fine-tune transformer nếu có dưới 500 mẫu gán nhãn trừ khi người dùng đã có sẵn một mô hình domain được huấn luyện trước. Cảnh báo các thực thể lồng nhau cần dùng mô hình dựa trên span hoặc duyệt nhiều lượt. Yêu cầu kiểm toán từ điển thực thể nếu người dùng đề cập đến "quy mô sản xuất" nhưng tập nhãn vẫn giữ nguyên từ CoNLL-2003.
```

## Bài tập (Exercises)

1. **Dễ.** Cài đặt hàm `bio_to_spans` (nghịch đảo của `spans_to_bio`) và kiểm chứng tính nhất quán hai chiều (round-trip) trên 10 câu mẫu.
2. **Trung bình.** Huấn luyện mô hình CRF với sklearn-crfsuite ở trên trên tập dữ liệu CoNLL-2003 English NER. Báo cáo F1 cho từng thực thể bằng `seqeval`. Kết quả điển hình: ~84 F1.
3. **Khó.** Tinh chỉnh (fine-tune) `distilbert-base-cased` trên một tập dữ liệu NER đặc thù ngành (y tế, pháp lý, hoặc tài chính). So sánh với mô hình nhỏ của spaCy. Kiểm tra rò rỉ dữ liệu (data leakage) và ghi lại những phát hiện bất ngờ.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| NER | Trích xuất tên riêng | Gán nhãn các đoạn token với các kiểu loại (PERSON, ORG, GPE, DATE,...). |
| BIO | Quy ước gán nhãn | `B-X` bắt đầu, `I-X` tiếp tục bên trong, `O` bên ngoài. |
| BILOU | BIO nâng cao | Bổ sung `L-X` (cuối cùng), `U-X` (đơn lẻ) để ranh giới rõ ràng hơn. |
| CRF | Bộ phân loại có cấu trúc | Mô hình hóa xác suất chuyển tiếp giữa các nhãn, không chỉ xác suất phát xạ. Ép buộc chuỗi nhãn hợp lệ. |
| Nested NER | Thực thể lồng ghép | Một đoạn văn bản lớn chứa thực thể đồng thời chứa một thực thể con khác bên trong. Chuẩn BIO không thể biểu diễn trường hợp này. |
| Entity-level F1 | Chỉ số F1 chuẩn cho NER | Đoạn thực thể dự đoán phải khớp chính xác tuyệt đối với thực thể thực tế. F1 cấp độ token sẽ thổi phồng độ chính xác sai lệch. |

## Đọc thêm (Further Reading)

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — bài báo kinh điển về BiLSTM-CRF.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — giới thiệu kiến trúc token-classification đã trở thành chuẩn mực hiện đại.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — tài liệu thực chiến cho mọi thuộc tính trên `Doc.ents` và `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — thư viện tính chỉ số chuẩn xác cho sequence labeling. Luôn ưu tiên sử dụng.

