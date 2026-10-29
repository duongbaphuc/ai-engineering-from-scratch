# GloVe, FastText và Vector Nhúng Cấp Độ Từ Phụ (GloVe, FastText, and Subword Embeddings)

> Word2Vec huấn luyện một vector nhúng cho mỗi từ riêng lẻ. GloVe phân tích ma trận đồng xuất hiện toàn cục. FastText nhúng từng mảnh ghép ký tự cấu thành. BPE bắc nhịp cầu tiến vào kỷ nguyên Transformer.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 phút

## The Problem

Word2Vec để lại hai câu hỏi lớn chưa có lời giải đáp trọn vẹn:

Thứ nhất, một hướng nghiên cứu song song đã sử dụng phân tích nhân tử ma trận đồng xuất hiện trực tiếp (LSA, HAL) thay vì cập nhật trực tuyến từng bước như Skip-gram. Liệu phương pháp lặp của Word2Vec có thực sự vượt trội hơn về mặt bản chất, hay sự khác biệt chỉ bắt nguồn từ cách xử lý tần suất đếm? **GloVe** đã trả lời câu hỏi đó: phân tích ma trận kết hợp với một hàm mất mát được thiết kế chu đáo hoàn toàn sánh ngang hoặc thậm chí vượt trội Word2Vec, đồng thời tốn ít thời gian huấn luyện hơn.

Thứ hai, cả hai phương pháp trên đều bất lực trước các từ chưa từng xuất hiện trong quá trình huấn luyện: `Zoomer-approved`, `dogecoin`, các danh từ riêng mới xuất hiện tuần trước, hoặc các dạng biến thể hình thái học của các gốc từ hiếm. **FastText** đã khắc phục điều này bằng cách nhúng các n-gram ký tự: một từ là tổng hợp của các thành phần con, bao gồm cả các hình vị (morphemes), nhờ đó ngay cả những từ ngoài từ vựng (out-of-vocabulary - OOV) cũng nhận được một vector mang ý nghĩa hợp lý.

Thứ ba, khi kiến trúc Transformer xuất hiện, bài toán lại dịch chuyển một lần nữa. Tập từ vựng ở cấp độ từ bị giới hạn ở khoảng một triệu từ; nhưng ngôn ngữ tự nhiên trong thực tế phong phú và biến đổi không ngừng. **Mã hoá cặp byte (Byte-Pair Encoding - BPE)** và các biến thể liên quan đã giải quyết bài toán này bằng cách học một tập từ vựng gồm các đơn vị từ phụ (subword units) phổ biến có thể ghép lại để bao quát mọi văn bản. Mọi bộ tách từ hiện đại cho mọi mô hình ngôn ngữ lớn (LLM) ngày nay đều là các bộ tách từ cấp độ từ phụ.

Bài học này sẽ dẫn dắt bạn qua cả ba kỹ thuật, sau đó giải thích rõ khi nào nên chọn phương pháp nào trong thực tế.

## The Concept

**GloVe (Global Vectors):** Xây dựng ma trận đồng xuất hiện giữa các từ `X`, trong đó `X[i][j]` là số lần từ `j` xuất hiện trong ngữ cảnh của từ `i`. Huấn luyện các vector sao cho `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Đánh trọng số cho hàm mất mát để các cặp từ quá phổ biến không lấn át toàn bộ quá trình học.

**FastText:** Một từ được biểu diễn bằng tổng của các n-gram ký tự cấu thành cộng với chính từ đó. Ví dụ: từ `where` được phân rã thành `<wh, whe, her, ere, re>, <where>`. Vector nhúng của từ là tổng của các vector thành phần đó. Quy trình huấn luyện tương tự như Word2Vec. Lợi ích: một từ chưa từng thấy (`whereupon`) vẫn được tổng hợp nên từ các n-gram đã biết.

**BPE (Byte-Pair Encoding):** Bắt đầu bằng một tập từ vựng gồm các byte (hoặc ký tự) đơn lẻ. Đếm tần suất của mọi cặp ký tự liền kề nhau trong toàn bộ kho ngữ liệu. Gộp cặp xuất hiện nhiều nhất thành một token mới. Lặp lại thao tác này qua `k` vòng lặp. Kết quả: thu được một tập từ vựng gồm các ký hiệu cơ sở cộng với `k` cặp đã gộp (`k + 256` token khi bắt đầu từ byte), trong đó các chuỗi phổ biến (`ing`, `tion`, `the`) trở thành các token độc lập và các từ hiếm được bẻ nhỏ thành các mảnh ghép quen thuộc. Mọi câu văn đều có thể phân tách thành token mà không bao giờ gặp lỗi OOV.

```figure
n5-subword-merge
```

## Build It

### GloVe: Phân tích ma trận đồng xuất hiện

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Hai chi tiết thiết kế then chốt: Hàm trọng số `f(x) = (x/x_max)^alpha` làm giảm ảnh hưởng của các cặp từ xuất hiện quá dày đặc (như `(the, and)`) để chúng không làm méo mó hàm mất mát. Vector nhúng cuối cùng là tổng của hai ma trận `W` (từ trung tâm) và `W_tilde` (từ ngữ cảnh). Việc cộng hai ma trận này là một mẹo kinh điển đã được công bố, giúp nâng cao chất lượng biểu diễn so với việc chỉ dùng một ma trận đơn lẻ.

### FastText: Vector nhúng nhận biết từ phụ

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Mỗi từ được biểu diễn bằng tập hợp các n-gram ký tự của nó (thường có độ dài từ 3 đến 6 ký tự). Vector nhúng của từ là tổng các vector nhúng của các n-gram này. Khi huấn luyện Skip-gram, ta chỉ việc đưa biểu diễn tổng hợp này vào thay thế cho vector đơn lẻ của Word2Vec.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Đối với một từ chưa từng xuất hiện trong từ điển, bạn vẫn nhận được một vector hợp lệ miễn là một số n-gram của nó đã được học. Từ `whereupon` chia sẻ các n-gram `<wh`, `her`, `ere` và `<where` với từ `where`, vì vậy vector của hai từ này sẽ tự nhiên nằm gần nhau trong không gian.

### BPE: Học tập từ vựng cấp độ từ phụ

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Vòng lặp đầu tiên sẽ gộp cặp ký tự liền kề xuất hiện nhiều nhất. Sau đủ số vòng lặp, các chuỗi con xuất hiện thường xuyên (`low`, `est`, `tion`) sẽ trở thành các token độc lập và các từ hiếm sẽ được bẻ tách một cách gãy gọn.

Các bộ tách từ thực tế của GPT / BERT / T5 học từ 30.000 đến 100.000 phép gộp. Kết quả là: bất kỳ văn bản nào cũng có thể được chuyển đổi thành một chuỗi token có độ dài giới hạn gồm các ID đã biết, hoàn toàn loại bỏ vấn đề từ ngoài từ điển OOV.

## Use It

Trong thực tế sản xuất, bạn hầu như không bao giờ tự huấn luyện các mô hình này từ đầu. Bạn tải về các checkpoint đã huấn luyện sẵn:

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Đối với việc tách từ cấp độ từ phụ kiểu BPE trong kỷ nguyên Transformer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Tiền tố `Ġ` đánh dấu ranh giới bắt đầu của một từ (quy ước của GPT-2). Mọi bộ tách từ hiện đại ngày nay đều là biến thể của BPE, WordPiece (dùng trong BERT), hoặc SentencePiece (dùng trong T5, LLaMA).

### Khi nào nên chọn công cụ nào?

| Tình huống | Lựa chọn tối ưu |
|------------|-----------------|
| Cần vector nhúng cấp từ tổng quát huấn luyện sẵn, không yêu cầu chịu lỗi OOV | GloVe 300d |
| Cần vector từ tổng quát, phải xử lý tốt lỗi chính tả / từ lóng mới / ngôn ngữ giàu hình thái | FastText |
| Bất kỳ tác vụ nào đưa vào mô hình Transformer (huấn luyện hoặc suy luận) | Dùng chính xác bộ tokenizer đi kèm với mô hình đó. Tuyệt đối không thay thế. |
| Tự huấn luyện một mô hình ngôn ngữ từ con số không | Huấn luyện một bộ tokenizer BPE hoặc SentencePiece trên kho ngữ liệu của bạn trước |
| Phân loại văn bản trong sản xuất với mô hình tuyến tính | Vẫn nên dùng TF-IDF (Bài 02) |

## Ship It

Lưu tại `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Exercises

1. **(Dễ)** Chạy `char_ngrams("playing")` và `char_ngrams("played")`. Tính độ trùng lặp Jaccard của hai tập n-gram này. Bạn sẽ thấy các mảnh ghép chung đáng kể (`pla`, `lay`, `play`), giải thích vì sao FastText chuyển giao tri thức rất tốt qua các biến thể hình thái học.
2. **(Trung bình)** Mở rộng hàm `learn_bpe` để theo dõi sự phát triển của tập từ vựng. Vẽ đồ thị tỷ lệ token trên số ký tự của kho ngữ liệu theo số lượng phép gộp. Bạn sẽ thấy tỷ lệ nén tăng vọt ở những bước đầu, sau đó tiệm cận mức ~2-3 ký tự trên mỗi token.
3. **(Khó)** Huấn luyện một bộ BPE với 1.000 phép gộp trên toàn bộ tác phẩm của Shakespeare. So sánh cách tách từ giữa các từ thông dụng với các danh từ riêng cổ điển hiếm gặp. Đo lường số token trung bình trên mỗi từ trước và sau khi gộp.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Ma trận đồng xuất hiện | Bảng tần suất từ-từ | `X[i][j]` = số lần từ `j` xuất hiện trong cửa sổ ngữ cảnh bao quanh từ `i` |
| Subword (Từ phụ) | Mảnh ghép của từ | Một n-gram ký tự (FastText) hoặc một token học được (BPE/WordPiece/SentencePiece) |
| BPE | Byte-Pair Encoding | Giải thuật gộp lặp đi lặp lại các cặp liền kề xuất hiện nhiều nhất cho đến khi đạt kích thước từ vựng mong muốn |
| OOV | Từ ngoài từ vựng | Từ chưa từng thấy trong tập huấn luyện. Word2Vec/GloVe thất bại; FastText và BPE xử lý được |
| Byte-level BPE | BPE trên byte thô | Cơ chế của GPT-2. Từ vựng bắt đầu bằng 256 giá trị byte, do đó không bao giờ xảy ra lỗi OOV |

## Further Reading

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — Bài báo gốc về GloVe, dài 7 trang, bản dẫn xuất hàm mất mát xuất sắc nhất
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — Bài báo đặt nền móng cho FastText
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Bài báo mang BPE vào thế giới NLP hiện đại
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — Phân tích chi tiết sự khác biệt thực tế giữa BPE, WordPiece và SentencePiece

