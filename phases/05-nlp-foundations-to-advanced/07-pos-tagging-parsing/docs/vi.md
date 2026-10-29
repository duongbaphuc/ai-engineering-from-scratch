# Gán nhãn Từ loại và Phân tích Cú pháp (POS Tagging and Syntactic Parsing)

> Ngữ pháp từng bị coi là lỗi thời trong một thời gian. Rồi đến khi mọi đường ống LLM đều cần kiểm thực trích xuất có cấu trúc, ngữ pháp đã trở lại mạnh mẽ.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Xử lý Văn bản), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Bài học 01 đã cam kết rằng quá trình đưa từ về dạng gốc (lemmatization) luôn cần nhãn từ loại (part-of-speech tag). Nếu không biết `running` là một động từ, bộ lemmatizer không thể rút gọn nó về `run`. Nếu không biết `better` là một tính từ, nó không thể đưa về `good`.

Lời hứa đó ẩn chứa cả một phân nhánh nghiên cứu sâu rộng. Gán nhãn từ loại (POS tagging) gán các phạm trù ngữ pháp cho từng từ. Phân tích cú pháp (syntactic parsing) khôi phục cấu trúc cây của câu: từ nào bổ nghĩa cho từ nào, động từ nào chi phối các đối số nào. NLP cổ điển đã dành hai mươi năm để tinh chỉnh cả hai. Sau đó, học sâu đã quy nạp chúng thành một tác vụ phân loại token (token classification) bên trên một transformer huấn luyện sẵn, và giới nghiên cứu học thuật đã chuyển hướng sang các bài toán khác.

Nhưng cộng đồng kỹ thuật ứng dụng thì không hề lãng quên. Mọi đường ống trích xuất có cấu trúc (structured-extraction pipeline) vẫn sử dụng POS và cây phụ thuộc ở tầng bên dưới. Dữ liệu JSON do LLM tạo ra được kiểm thực dựa trên các ràng buộc ngữ pháp. Các hệ thống hỏi đáp (QA) phân rã câu truy vấn nhờ vào cây phân tích phụ thuộc. Các bộ đánh giá chất lượng dịch máy kiểm tra sự gióng hàng của các cây cú pháp.

Đây là kiến thức cực kỳ đáng giá. Bài học này giới thiệu các bộ nhãn (tagset), các mô hình baseline, và thời điểm bạn nên dừng tự viết mã từ đầu để chuyển sang dùng spaCy.

## Khái niệm (The Concept)

**Gán nhãn từ loại (POS tagging)** gán cho mỗi token một phạm trù ngữ pháp. Bộ nhãn **Penn Treebank (PTB)** là chuẩn mặc định cho tiếng Anh. Nó gồm 36 nhãn với những phân biệt chi tiết mà người đọc thông thường có thể thấy hơi rườm rà: `NN` danh từ số ít, `NNS` danh từ số nhiều, `NNP` danh từ riêng số ít, `VBD` động từ thì quá khứ, `VBZ` động từ ngôi thứ ba số ít thì hiện tại, v.v. Bộ nhãn **Universal Dependencies (UD)** thô hơn (17 nhãn) và độc lập với ngôn ngữ; nó đã trở thành chuẩn mặc định cho các bài toán xử lý đa ngôn ngữ.

```text
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Phân tích cú pháp (Syntactic parsing)** tạo ra một cấu trúc cây. Có hai trường phái chính:

- **Phân tích ngữ đoạn (Constituency parsing).** Các cụm danh từ (NP), cụm động từ (VP), cụm giới từ (PP) lồng ghép vào nhau. Đầu ra là một cây chứa các danh mục không kết thúc (non-terminal categories: NP, VP, PP) với các từ thực tế đóng vai trò là các nút lá.
- **Phân tích phụ thuộc (Dependency parsing).** Mỗi từ có duy nhất một từ đứng đầu (head word) mà nó phụ thuộc vào, được gắn nhãn với một quan hệ ngữ pháp cụ thể. Đầu ra là một cây trong đó mỗi cạnh là một bộ ba (head, dependent, relation).

Phân tích phụ thuộc đã giành chiến thắng hoàn toàn trong thập niên 2010 vì nó tổng quát hóa một cách tự nhiên và mạch lạc trên nhiều ngôn ngữ, đặc biệt là các ngôn ngữ có trật tự từ tự do.

```text
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## Xây dựng thực tế (Build It)

### Bước 1: Mô hình cơ sở nhãn xuất hiện nhiều nhất (Most-Frequent-Tag baseline)

Bộ gán nhãn POS đơn giản nhất nhưng vẫn dùng được: Với mỗi từ, dự đoán nhãn mà nó xuất hiện nhiều nhất trong tập huấn luyện.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Trên ngữ liệu Brown, mô hình baseline này đạt độ chính xác khoảng ~85%. Chưa phải là tốt, nhưng là ngưỡng sàn mà không một mô hình nghiêm túc nào được phép thấp hơn.

### Bước 2: Bộ gán nhãn Bigram HMM

Mô hình hóa phân phối xác suất đồng thời của chuỗi:

```math
P(\text{tags}, \text{words}) = \prod_i P(\text{tag}_i \mid \text{tag}_{i-1}) \cdot P(\text{word}_i \mid \text{tag}_i)
```

Gồm hai bảng xác suất: Xác suất chuyển trạng thái (nhãn khi biết nhãn đứng trước), xác suất phát xạ (từ khi biết nhãn). Ước tính cả hai từ tần số đếm kết hợp làm mịn Laplace (Laplace smoothing). Giải mã bằng thuật toán Viterbi (quy hoạch động trên lưới trạng thái nhãn - tag lattice).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM trên ngữ liệu Brown đạt độ chính xác ~93%. Bước nhảy từ 85% lên 93% phần lớn nhờ vào các xác suất chuyển tiếp — mô hình học được rằng `DET NOUN` rất phổ biến trong khi `NOUN DET` là cực kỳ hiếm.

### Bước 3: Tại sao các bộ gán nhãn hiện đại vượt trội hơn

Xác suất chuyển tiếp và xác suất phát xạ mang tính cục bộ. Chúng không thể nắm bắt được rằng `saw` là danh từ trong "I bought a saw" (cái cưa) nhưng là động từ trong "I saw the movie" (đã thấy). Một mô hình CRF với các đặc trưng phong phú (hậu tố, hình dạng từ, từ đứng trước và sau, bản thân từ đó) đạt độ chính xác ~97%. BiLSTM-CRF hoặc transformer đạt tới ~98%+.

Ngưỡng trần của tác vụ này bị giới hạn bởi sự bất đồng ý kiến của người gán nhãn (annotator disagreement). Các chuyên gia gán nhãn con người chỉ đồng thuận khoảng 97% trên Penn Treebank. Các mô hình vượt qua 98% thường có xu hướng overfitting vào tập kiểm thử.

### Bước 4: Phác thảo phân tích phụ thuộc (Dependency parsing)

Việc tự xây dựng một bộ phân tích phụ thuộc hoàn chỉnh từ đầu nằm ngoài phạm vi ngắn gọn của bài học; tài liệu giáo khoa kinh điển nằm trong cuốn sách của Jurafsky và Martin. Hai trường phái cổ điển bạn cần biết:

- **Bộ phân tích dựa trên chuyển trạng thái (Transition-based parsers)** (arc-eager, arc-standard) hoạt động tương tự như một bộ phân tích cú pháp shift-reduce: chúng đọc các token, đẩy (shift) vào ngăn xếp (stack), và áp dụng các thao tác rút gọn (reduce) để tạo cung phụ thuộc. Giải mã tham lam (greedy decoding) cho tốc độ rất nhanh. Bản cài đặt kinh điển là MaltParser. Phiên bản mạng nơ-ron hiện đại: bộ phân tích transition-based của Chen và Manning.
- **Bộ phân tích dựa trên đồ thị (Graph-based parsers)** (thuật toán Eisner, mô hình Dozat-Manning biaffine) chấm điểm mọi cạnh head-dependent khả dĩ và chọn cây khung lớn nhất (maximum spanning tree). Chậm hơn nhưng độ chính xác cao hơn.

Đối với hầu hết công việc ứng dụng thực tế, hãy gọi spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```text
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Đọc cột `dep` từ dưới lên trên và cấu trúc ngữ pháp hoàn chỉnh của câu sẽ hiện ra rõ ràng.

## Sử dụng trong thực tế (Use It)

Mọi thư viện NLP môi trường sản xuất đều tích hợp sẵn các bộ phân tích cú pháp và POS như một phần của pipeline tiêu chuẩn.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Nhanh, chuẩn xác, tích hợp sẵn tách từ + NER + lemmatization. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (quan hệ phụ thuộc).
- **Stanford NLP (stanza)**. Thế hệ kế thừa từ CoreNLP của Đại học Stanford. Đạt chất lượng state-of-the-art trên hơn 60 ngôn ngữ.
- **trankit**. Dựa trên transformer, độ chính xác UD rất cao.
- **NLTK**. `pos_tag`. Dùng được nhưng chậm và cũ. Phù hợp cho việc giảng dạy lý thuyết.

### Vai trò thiết yếu trong năm 2026

- **Rút gọn từ về dạng gốc (Lemmatization).** Bài học 01 luôn cần POS để lemmatize chuẩn xác.
- **Kiểm thực trích xuất có cấu trúc từ LLM.** Đảm bảo rằng một câu do LLM tạo ra tuân thủ các ràng buộc ngữ pháp (ví dụ: sự hòa hợp chủ vị, các từ bổ nghĩa bắt buộc).
- **Phân tích cảm xúc theo khía cạnh (Aspect-based sentiment).** Cây phụ thuộc cho bạn biết chính xác tính từ nào đang bổ nghĩa cho danh từ nào.
- **Hiểu câu truy vấn (Query understanding).** "movies directed by Wes Anderson starring Bill Murray" được phân rã thành các ràng buộc cấu trúc rõ ràng nhờ vào cây phân tích.
- **Chuyển giao đa ngôn ngữ (Cross-lingual transfer).** Các nhãn UD và quan hệ phụ thuộc mang tính độc lập với ngôn ngữ, hỗ trợ phân tích cấu trúc zero-shot trên các ngôn ngữ mới.
- **Các pipeline tính toán hạn chế (Low-compute pipelines).** Nếu bạn không thể triển khai transformer vì tài nguyên, bộ kết hợp POS + dependency parse + từ điển (gazetteer) sẽ đưa bạn đi rất xa với chi phí tối thiểu.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Thiết kế pipeline POS + dependency cổ điển cho một tác vụ NLP hạ nguồn.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Khi nhận được bài toán hạ nguồn (trích xuất thông tin, kiểm thực viết lại câu, phân rã truy vấn, lemmatization), bạn đưa ra:

1. Bộ nhãn sử dụng. Penn Treebank cho pipeline cũ chỉ dùng tiếng Anh, Universal Dependencies cho bài toán đa ngôn ngữ hoặc liên ngôn ngữ.
2. Thư viện. spaCy cho đa số hệ thống sản xuất, stanza cho nghiên cứu học thuật đa ngữ, trankit để có độ chính xác UD cao nhất. Nêu rõ ID mô hình cụ thể.
3. Mẫu tích hợp. Cung cấp 3-5 dòng mã gọi thư viện và sử dụng các thuộc tính cần thiết (`.pos_`, `.dep_`, `.head`).
4. Phương thức lỗi cần kiểm thử. Sự nhập nhằng danh từ - động từ (`saw`, `book`, `can`) và sự nhập nhằng gắn kết cụm giới từ (PP-attachment) là những cạm bẫy kinh điển. Lấy mẫu 20 đầu ra và trực tiếp kiểm tra bằng mắt.

Từ chối khuyến nghị tự viết bộ phân tích cú pháp riêng từ đầu. Xây dựng parser từ đầu là dự án nghiên cứu học thuật, không phải tác vụ kỹ thuật ứng dụng. Cảnh báo bất kỳ pipeline nào sử dụng nhãn POS mà không xử lý các biến thể chữ hoa/chữ thường là hệ thống lỏng lẻo dễ gãy.
```

## Bài tập (Exercises)

1. **Dễ.** Sử dụng mô hình baseline nhãn xuất hiện nhiều nhất (most-frequent-tag) trên một tập ngữ liệu gán nhãn nhỏ (ví dụ: tập con Brown của NLTK), đo độ chính xác trên các câu tách biệt. Xác nhận kết quả xấp xỉ ~85%.
2. **Trung bình.** Huấn luyện mô hình Bigram HMM ở trên và báo cáo precision/recall theo từng nhãn. HMM thường nhầm lẫn các cặp nhãn nào nhất?
3. **Khó.** Dùng cây phân tích phụ thuộc của spaCy để trích xuất các bộ ba chủ ngữ - động từ - tân ngữ (subject-verb-object) từ tập mẫu 1.000 câu. Đánh giá trên 50 bộ ba gán nhãn thủ công. Ghi nhận tài liệu các trường hợp trích xuất thất bại (thường là câu bị động, cấu trúc đồng đẳng và các chủ ngữ bị tỉnh lược).

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Nhãn POS (POS tag) | Kiểu của từ | Phạm trù ngữ pháp. PTB có 36 nhãn; UD có 17 nhãn. |
| Penn Treebank | Bộ nhãn tiêu chuẩn | Dành riêng cho tiếng Anh. Phân biệt chi tiết thì động từ và số lượng danh từ. |
| Universal Dependencies | Bộ nhãn đa ngữ | Khái quát hơn PTB; trung lập về ngôn ngữ; chuẩn mặc định cho phân tích liên ngôn ngữ. |
| Cây phân tích phụ thuộc (Dependency parse) | Cây cấu trúc câu | Mỗi từ có một từ đứng đầu (head), mỗi cạnh mang một quan hệ ngữ pháp. |
| Viterbi | Quy hoạch động | Tìm chuỗi nhãn có xác suất cao nhất dựa trên xác suất phát xạ và chuyển trạng thái. |

## Đọc thêm (Further Reading)

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — tài liệu giáo khoa kinh điển về POS và cú pháp học.
- [Universal Dependencies project](https://universaldependencies.org/) — tập hợp ngân hàng cây cú pháp (treebank) và bộ nhãn liên ngôn ngữ được mọi bộ phân tích đa ngữ sử dụng.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — tài liệu tham khảo thực chiến cho mọi thuộc tính trên đối tượng `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — bài báo đưa các bộ phân tích cú pháp nơ-ron vào dòng chảy chính.

