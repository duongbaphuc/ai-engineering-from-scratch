# Tạo sinh Văn bản trước thời kỳ Transformer — Mô hình Ngôn ngữ N-gram

> Nếu một từ gây bất ngờ, mô hình đó kém. Độ hỗn loạn (Perplexity) biến sự bất ngờ thành một con số định lượng. Kỹ thuật làm mịn (Smoothing) giữ cho con số đó luôn hữu hạn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Xử lý Văn bản), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Trước khi có Transformer, trước khi có RNN, thậm chí trước cả khi có vector nhúng từ (word embeddings), một mô hình ngôn ngữ dự đoán từ tiếp theo bằng cách đếm xem nó xuất hiện bao nhiêu lần đứng sau `n-1` từ đứng trước. Đếm thấy "the cat" → "sat" xuất hiện 47 lần, "the cat" → "jumped" xuất hiện 12 lần, "the cat" → "refrigerator" xuất hiện 0 lần. Chuẩn hóa các số đếm đó để nhận được một phân phối xác suất.

Đó chính là mô hình ngôn ngữ n-gram. Nó đã vận hành mọi hệ thống nhận dạng tiếng nói, mọi bộ kiểm tra chính tả và mọi hệ thống dịch máy dựa trên cụm từ từ năm 1980 đến tận năm 2015. Nó vẫn tiếp tục hoạt động cho đến hôm nay khi bạn cần một mô hình ngôn ngữ chi phí rẻ triển khai trên thiết bị cục bộ.

Vấn đề hóc búa thực sự là làm gì với những n-gram chưa từng thấy trong tập huấn luyện (unseen n-grams). Một mô hình đếm thô thuần túy sẽ gán xác suất bằng 0 cho bất kỳ chuỗi nào nó chưa từng gặp, điều này mang tính thảm họa vì các câu văn thường dài và gần như mọi câu văn dài đều chứa ít nhất một chuỗi con chưa từng thấy trước đây. Năm mươi năm nghiên cứu về các kỹ thuật làm mịn (smoothing) đã giải quyết triệt để vấn đề này. Làm mịn Kneser-Ney chính là đỉnh cao của nỗ lực đó, và học sâu hiện đại thừa hưởng toàn bộ truyền thống thực nghiệm quý báu này.

## Khái niệm (The Concept)

![Mô hình N-gram: đếm tần số, làm mịn, và tạo sinh](../assets/ngram.svg)

### Trò chơi đoán chữ (The Prediction Game)

Trước khi bất kỳ cỗ máy toán học nào ở đây ra đời, một thí nghiệm duy nhất đã định hình định nghĩa về mô hình ngôn ngữ: Hãy che chữ cái tiếp theo của một câu tiếng Anh. Yêu cầu một người đoán chữ cái đó, đoán từng lần một cho đến khi đúng. Ghi lại số lần phải đoán. Lặp lại thí nghiệm cho vài trăm ký tự tiếp theo.

Số lần đoán không phải là số liệu đố vui vô nghĩa. Chúng thực chất là một phép mã hóa không tổn hao (lossless re-encoding) của đoạn văn bản: hãy trao chuỗi số lần đoán đó cho một người đoán thứ hai có tư duy tương tự, họ có thể tái dựng lại chính xác từng chữ cái một, bởi vì tại mỗi vị trí họ đều biết rõ phương án nào được ưu tiên đoán trước. Một thông điệp có thể tái mã hóa bằng ít ký hiệu hơn nghĩa là nó mang ít thông tin hơn trên mỗi ký hiệu, do đó thống kê số lần đoán đã đặt ra giới hạn trần cho entropy (độ hỗn loạn thông tin) của tiếng Anh.

Claude Shannon đã tiến hành thí nghiệm này vào năm 1951 và tìm ra một con số vẫn chi phối toàn bộ ngành khoa học ngôn ngữ đến nay: Một bảng chữ cái gồm 27 ký hiệu (26 chữ cái cộng dấu cách) có thể chứa tối đa `log2(27) ≈ 4.75` bit trên mỗi ký tự. Nhưng những người đoán có ngữ cảnh 100 chữ cái trước đó chỉ mất từ 0.6 đến 1.3 bit trên mỗi ký tự. Tiếng Anh thực chất có khoảng 3/4 cấu trúc là các bước đi bắt buộc đã được định sẵn bởi ngữ pháp và quy luật từ vựng. Cấu trúc mà một mô hình ngôn ngữ cần phải học đã được đo lường chính xác từ trước khi bất kỳ mô hình nơ-ron nào ra đời.

Mọi mô hình ngôn ngữ kể từ đó đều là những người chơi tự động của trò chơi đoán chữ này, và mọi chỉ số đánh giá trong bài học này đều là điểm số của trò chơi đó:

- **Hàm mất mát Cross-Entropy (Cross-entropy loss)** chính là số bit trung bình mà mô hình cần để mã hóa mỗi ký hiệu. Huấn luyện một mô hình ngôn ngữ (LM) thực chất là cực tiểu hóa điểm số của nó trong trò chơi đoán chữ.
- **Độ hỗn loạn (Perplexity)** là `2^bits` (hoặc `e^nats`): hệ số phân nhánh (branching factor) mà mô hình vẫn phải đối mặt sau khi đã tận dụng phỏng đoán của mình. Đoán ngẫu nhiên đều trên 27 ký hiệu có perplexity là 27; một người chơi đạt mức 1 bit/ký tự sẽ có perplexity bằng 2.
- **Độ dài ngữ cảnh (Context length) là trí nhớ của người chơi.** Một mô hình trigram chơi trò này với trí nhớ chỉ vỏn vẹn 2 token đứng trước. Một mô hình Transformer chơi cùng một trò chơi đó với bộ nhớ 100.000 token. Luật chơi không hề thay đổi; chỉ có người chơi ngày càng trở nên thông thái hơn.

Một điểm chuyển đổi đơn vị cần lưu ý: trò chơi đoán chữ tính điểm trên từng chữ cái bằng đơn vị bit (`log2`), trong khi các công thức n-gram dưới đây tính điểm trên từng token từ ngữ bằng đơn vị nat (logarit tự nhiên `ln`) — và vì perplexity `e^H` tính theo nat hoàn toàn tương đương với `2^H` tính theo bit, nên hai góc nhìn này thực chất là một phép đo thống nhất nhưng ở các đơn vị đo lường khác nhau.

```figure
prediction-game
```

**Xác suất N-gram:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Cố định `n` (thường là 3 cho trigram, 4 cho 4-gram). Tính toán trực tiếp từ tần số đếm:

```math
P(w \mid \text{context}) = \frac{\mathop{\mathrm{count}}(\text{context}, w)}{\mathop{\mathrm{count}}(\text{context})}
```

**Vấn đề tần số đếm bằng 0 (The Zero-Count Problem).** Bất kỳ n-gram nào chưa từng xuất hiện trong tập huấn luyện đều nhận xác suất bằng 0. Một nghiên cứu năm 2007 trên ngữ liệu Brown chỉ ra rằng ngay cả với mô hình 4-gram, có tới 30% số 4-gram trong tập kiểm thử độc lập chưa từng xuất hiện trong tập huấn luyện. Bạn không thể đánh giá mô hình trên bất kỳ văn bản thực tế nào nếu thiếu kỹ thuật làm mịn.

**Các phương pháp làm mịn, xếp theo mức độ tinh vi:**

1. **Làm mịn Laplace (Cộng thêm 1).** Cộng thêm 1 vào mọi tần số đếm. Đơn giản, nhưng hoạt động rất tệ đối với các sự kiện hiếm.
2. **Good-Turing.** Tái phân bổ khối lượng xác suất từ các sự kiện tần suất cao sang các sự kiện chưa từng thấy dựa trên tần suất của các tần suất.
3. **Nội suy (Interpolation).** Kết hợp ước lượng của n-gram, (n-1)-gram,... với các trọng số có thể tinh chỉnh được.
4. **Lùi bước (Backoff).** Nếu n-gram có số đếm bằng 0, lùi về dùng (n-1)-gram. Phương pháp Katz backoff chuẩn hóa quy trình này.
5. **Chiết khấu tuyệt đối (Absolute discounting).** Trừ đi một lượng chiết khấu cố định `D` khỏi tất cả các số đếm, rồi tái phân phối phần xác suất dôi dư cho các từ chưa thấy.
6. **Kneser-Ney.** Chiết khấu tuyệt đối kết hợp với một lựa chọn thông minh cho mô hình bậc thấp hơn: sử dụng *xác suất tiếp nối (continuation probability)* (từ đó xuất hiện trong bao nhiêu ngữ cảnh khác nhau) thay vì tần suất xuất hiện thô.

Phát kiến của Kneser-Ney mang tính chiều sâu sâu sắc: "San Francisco" là một bigram rất phổ biến. Từ đơn "Francisco" hầu như chỉ xuất hiện sau từ "San". Phương pháp chiết khấu tuyệt đối ngây thơ sẽ gán cho "Francisco" một xác suất unigram rất cao (vì tần số đếm của nó lớn). Kneser-Ney nhận ra rằng "Francisco" chỉ xuất hiện trong đúng một ngữ cảnh duy nhất và chủ động hạ thấp xác suất tiếp nối của nó xuống. Kết quả: một bigram mới chưa từng thấy kết thúc bằng từ "Francisco" sẽ nhận được một xác suất thấp rất hợp lý.

**Đánh giá: Độ hỗn loạn (Perplexity).** Hàm mũ của trung bình log-likelihood âm trên mỗi từ trong tập kiểm thử riêng biệt. Giá trị càng thấp càng tốt. Perplexity bằng 100 nghĩa là mô hình đang bối rối tương đương với việc phải chọn ngẫu nhiên đều giữa 100 từ vựng.

```math
\text{perplexity} = \exp\left(-\frac{1}{N} \sum_i \log P(w_i \mid \text{context}_i)\right)
```

```figure
ngram-backoff
```

## Xây dựng thực tế (Build It)

### Bước 1: Đếm tần số Trigram

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Đầu vào là danh sách các câu đã được tách token. Đầu ra là tần số đếm n-gram và tần số đếm ngữ cảnh. `<s>` và `</s>` là các token đánh dấu ranh giới bắt đầu và kết thúc câu.

### Bước 2: Làm mịn Laplace

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Cộng thêm 1 vào mọi tần số đếm. Giúp tránh xác suất bằng 0 nhưng lại phân bổ quá nhiều xác suất cho các sự kiện chưa thấy, làm tổn hại đến xác suất của các từ hiếm đã biết.

### Bước 3: Kneser-Ney (Bigram, dạng nội suy)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Ba thành phần cốt lõi: `continuation_prob` ghi nhận "từ này xuất hiện trong bao nhiêu ngữ cảnh khác nhau?" (cải tiến mang tính đột phá của Kneser-Ney). `lambda_prev` là khối lượng xác suất dôi dư ra sau khi chiết khấu, dùng làm trọng số để lùi bước. Xác suất cuối cùng là số hạng chính đã chiết khấu cộng với số hạng tiếp nối đã nhân trọng số.

### Bước 4: Tạo sinh văn bản bằng lấy mẫu xác suất

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Lấy mẫu tỷ lệ thuận với xác suất. Luôn tạo ra các đầu ra đa dạng theo từng giá trị seed ngẫu nhiên. Để tạo đầu ra dạng tìm kiếm chùm/tham lam, hãy chọn từ có argmax tại mỗi bước và điều chỉnh thêm nhiệt độ ngẫu nhiên (temperature).

### Bước 5: Tính toán độ hỗn loạn (Perplexity)

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Giá trị càng thấp càng tốt. Đối với ngữ liệu Brown, một mô hình 4-gram Kneser-Ney tinh chỉnh tốt đạt độ hỗn loạn khoảng 140. Một mô hình ngôn ngữ Transformer đạt độ hỗn loạn từ 15 đến 30 trên cùng tập kiểm thử đó. Khoảng cách chênh lệch lên tới 10 lần. Đó chính là lý do toàn bộ ngành công nghiệp đã chuyển dịch hoàn toàn sang kiến trúc mạng nơ-ron hiện đại.

## Sử dụng trong thực tế (Use It)

- **Giảng dạy NLP cổ điển:** Cách tiếp cận trực quan và sáng tỏ nhất để hiểu rõ bản chất của kỹ thuật làm mịn, ước lượng hợp lý cực đại (MLE) và chỉ số perplexity.
- **KenLM:** Thư viện n-gram cấp độ sản xuất được tối ưu hóa cực hạn. Dùng làm bộ chấm điểm lại (rescorer) trong các hệ thống nhận dạng tiếng nói và dịch máy đòi hỏi độ trễ siêu thấp.
- **Tự động hoàn thành trên thiết bị (On-device autocomplete):** Các mô hình trigram trên bàn phím điện thoại vẫn tiếp tục hoạt động âm thầm và hiệu quả.
- **Mô hình cơ sở (Baselines):** Luôn luôn tính toán perplexity của mô hình n-gram trước khi tuyên bố mô hình mạng nơ-ron của bạn đạt chất lượng tốt. Nếu Transformer của bạn không vượt qua Kneser-Ney với cách biệt vượt trội, chắc chắn hệ thống của bạn đang gặp lỗi kỹ thuật.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Xây dựng một baseline mô hình ngôn ngữ n-gram có khả năng tái lập trước khi huấn luyện mô hình ngôn ngữ nơ-ron.
phase: 5
lesson: 16
---

Dựa trên kho ngữ liệu và mục đích sử dụng (dự đoán từ tiếp theo, xếp hạng lại rescoring, baseline perplexity), bạn đưa ra:

1. Bậc của N-gram: Trigram cho tiếng Anh thông dụng, 4-gram nếu kho dữ liệu lớn, 5-gram cho tác vụ rescoring nhận dạng tiếng nói.
2. Kỹ thuật làm mịn: Modified Kneser-Ney là chuẩn mặc định; Laplace chỉ dùng cho mục đích giảng dạy lý thuyết.
3. Thư viện: `kenlm` cho sản xuất quy mô lớn, `nltk.lm` cho học tập, chỉ tự viết mã từ đầu khi muốn nắm vững thuật toán.
4. Kế hoạch đánh giá: Perplexity trên tập kiểm thử riêng biệt với quy tắc tách từ (tokenization) đồng nhất tuyệt đối giữa tập train và tập test.

Từ chối báo cáo điểm perplexity được tính toán bằng các quy tắc tách từ khác nhau giữa các hệ thống đang so sánh — điểm số perplexity chỉ có giá trị so sánh khi quy trình tách từ là đồng nhất tuyệt đối. Gắn cờ cảnh báo tỷ lệ từ ngoài từ vựng (OOV) trong tập kiểm thử; Kneser-Ney xử lý OOV rất kém trừ khi bạn chủ động dành riêng một token đặc biệt <UNK> trong quá trình huấn luyện.
```

## Bài tập (Exercises)

1. **Dễ.** Huấn luyện một mô hình trigram LM trên tập ngữ liệu 1.000 câu văn của Shakespeare. Sinh ra 20 câu mới. Bạn sẽ thấy các câu văn rất hợp lý ở phạm vi cục bộ nhưng hoàn toàn rời rạc, vô nghĩa về mặt cấu trúc toàn cục. Đây là thử nghiệm kinh điển trực quan nhất.
2. **Trung bình.** Cài đặt phép tính perplexity cho mô hình Kneser-Ney của bạn trên tập kiểm thử Shakespeare riêng biệt. So sánh với phương pháp làm mịn Laplace. Bạn sẽ thấy Kneser-Ney giảm độ hỗn loạn perplexity từ 30% đến 50%.
3. **Khó.** Xây dựng một bộ sửa lỗi chính tả bằng trigram: cho trước một từ bị viết sai và ngữ cảnh xung quanh, tạo ra các từ sửa lỗi ứng viên và xếp hạng chúng dựa trên xác suất ngữ cảnh tính từ LM. Đánh giá trên ngữ liệu chính tả Birkbeck (dữ liệu mở công khai).

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| N-gram | Chuỗi từ | Chuỗi gồm `n` token liên tiếp nhau. |
| Làm mịn (Smoothing) | Tránh xác suất 0 | Tái phân bổ khối lượng xác suất để các sự kiện chưa thấy nhận được xác suất khác 0. |
| Độ hỗn loạn (Perplexity) | Thước đo chất lượng LM | `exp(-average log-prob)` trên tập dữ liệu kiểm thử riêng biệt. Giá trị càng thấp càng tốt. |
| Lùi bước (Backoff) | Lùi về ngữ cảnh ngắn hơn | Nếu số đếm trigram bằng 0, lùi về dùng bigram. Katz backoff chuẩn hóa quy trình này. |
| Kneser-Ney | Làm mịn n-gram tốt nhất | Chiết khấu tuyệt đối + xác suất tiếp nối cho mô hình bậc thấp hơn. |
| Xác suất tiếp nối (Continuation probability) | Đặc thù của KN | `P(w)` được đánh trọng số dựa trên số lượng ngữ cảnh khác nhau mà `w` xuất hiện, không phải tần số đếm thô. |
| Entropy của văn bản | Lượng thông tin mỗi ký hiệu | Số bit trung bình cần thiết để mã hóa ký hiệu tiếp theo khi biết ngữ cảnh. Ước tính năm 1951 của Shannon cho tiếng Anh in với ngữ cảnh 100 chữ cái: 0.6-1.3 bit/ký tự, được đo lường trước khi có bất kỳ mô hình máy học nào. |

## Đọc thêm (Further Reading)

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — thí nghiệm trò chơi đoán chữ định hình mục tiêu tối ưu mà mọi mô hình ngôn ngữ ngày nay vẫn đang giải quyết.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — tài liệu giáo khoa chuẩn mực kinh điển về mô hình n-gram và các kỹ thuật làm mịn.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — công trình thực nghiệm chứng minh Kneser-Ney là kỹ thuật làm mịn tốt nhất cho n-gram.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — bài báo gốc giới thiệu thuật toán Kneser-Ney.
- [KenLM](https://kheafield.com/code/kenlm/) — thư viện mô hình ngôn ngữ n-gram tối ưu cho sản xuất, vẫn được tin dùng năm 2026 cho các bài toán độ trễ thấp.

