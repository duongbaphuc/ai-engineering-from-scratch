# Xử Lý Văn Bản — Tokenization, Stemming, Lemmatization (Text Processing)

> Ngôn ngữ tự nhiên mang tính liên tục. Các mô hình học máy lại mang tính rời rạc. Tiền xử lý chính là chiếc cầu nối giữa hai thế giới.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 phút

## The Problem

Một mô hình không thể đọc trực tiếp câu "The cats were running." Nó chỉ đọc được các con số nguyên.

Mọi hệ thống Xử lý Ngôn ngữ Tự nhiên (NLP) đều mở đầu bằng ba câu hỏi quen thuộc: Một từ bắt đầu từ đâu? Gốc của từ đó là gì? Làm thế nào để ta xem "run", "running", "ran" là cùng một thực thể khi điều đó có ích, và xem chúng là các từ khác nhau khi ngữ cảnh đòi hỏi?

Nếu khâu tách từ (tokenization) bị sai, mô hình sẽ học trên dữ liệu rác. Nếu bộ tách từ của bạn xem `don't` là một token nhưng `do n't` là hai token, phân phối dữ liệu huấn luyện sẽ bị phân mảnh. Nếu bộ rút gọn gốc từ (stemmer) gộp cả `organization` và `organ` về cùng một gốc, bài toán mô hình hoá chủ đề (topic modeling) sẽ hỏng. Nếu bộ đưa về dạng gốc (lemmatizer) cần ngữ cảnh từ loại (part-of-speech) nhưng bạn không cung cấp, động từ sẽ bị nhầm lẫn thành danh từ.

Bài học này xây dựng ba bước tiền xử lý từ con số không, sau đó minh hoạ cách NLTK và spaCy thực hiện các công việc này trong thực tế để bạn thấy rõ các đánh đổi kỹ thuật.

## The Concept

Ba thao tác. Mỗi thao tác có nhiệm vụ riêng và trạng thái lỗi đặc thù.

**Tách từ (Tokenization)** chia nhỏ một chuỗi ký tự thành các token. Khái niệm "Token" được định nghĩa tương đối linh hoạt vì độ chi tiết tối ưu phụ thuộc vào từng tác vụ cụ thể: cấp độ từ (word-level) cho NLP cổ điển, cấp độ từ phụ (subword) cho Transformer, và cấp độ ký tự (character) cho các ngôn ngữ không có khoảng trắng phân tách từ.

**Rút gọn gốc từ (Stemming)** chặt bỏ các hậu tố dựa trên các tập luật định sẵn. Rất nhanh, quyết liệt, nhưng thô sơ. `running -> run`. `organization -> organ`. Trường hợp thứ hai chính là trạng thái lỗi kinh điển.

**Đưa về dạng từ điển (Lemmatization)** quy đổi một từ về dạng nguyên thể theo từ điển dựa trên tri thức ngữ pháp. Chậm hơn, chính xác cao, đòi hỏi bảng tra cứu hoặc bộ phân tích hình thái học (morphological analyzer). `ran -> run` (cần biết "ran" là thì quá khứ của "run"). `better -> good` (cần biết các dạng so sánh hơn).

Quy tắc ngón tay cái: Dùng Stemming khi tốc độ là ưu tiên số một và bạn chấp nhận được độ nhiễu (đánh chỉ mục tìm kiếm, phân loại thô). Dùng Lemmatization khi ngữ nghĩa là yếu tố quyết định (hỏi đáp, tìm kiếm ngữ nghĩa, hoặc bất cứ nội dung nào hiển thị trực tiếp cho người dùng đọc).

```figure
edit-distance
```

## Build It

### Bước 1: Bộ tách từ dựa trên biểu thức chính quy (Regex word tokenizer)

Bộ tách từ đơn giản mà hữu dụng nhất là tách theo các ký tự không phải chữ-số, đồng thời giữ lại dấu câu như các token độc lập. Tuy chưa hoàn hảo, nó có thể chạy chỉ trong một dòng lệnh:

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Ba mẫu tìm kiếm theo thứ tự ưu tiên: Các từ có chứa dấu nháy đơn tuỳ chọn bên trong (`don't`, `it's`). Các chuỗi số thuần tuý. Bất kỳ ký tự đơn lẻ nào không phải khoảng trắng và không phải chữ-số như một token độc lập (dấu câu).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Những điểm hạn chế cần lưu ý: `3pm` bị tách thành `['3', 'pm']` vì ta phân tách giữa khối chữ cái và khối số. Điều này đủ tốt cho đa số tác vụ thông thường. Nhưng URL, email, hashtag sẽ bị tách vỡ. Trong môi trường sản xuất, hãy bổ sung các mẫu regex đặc thù trước các mẫu regex tổng quát.

### Bước 2: Bộ rút gọn gốc từ Porter (Chỉ triển khai Bước 1a)

Thuật toán Porter hoàn chỉnh có năm giai đoạn áp dụng luật. Riêng Bước 1a đã bao quát được hầu hết các hậu tố tiếng Anh thường gặp và làm rõ mẫu hình vận hành:

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Đọc các quy tắc từ trên xuống dưới. Quy tắc `ies -> i` lý giải tại sao `ponies -> poni`, chứ không phải `pony`. Thuật toán Porter thực tế có bước 1b để hiệu chỉnh lại điều này. Các luật cạnh tranh nhau, luật nào đứng trước sẽ thắng. Thứ tự ưu tiên của các luật quan trọng hơn bản thân từng luật đơn lẻ.

### Bước 3: Bộ Lemmatizer dựa trên bảng tra cứu

Lemmatization đúng nghĩa đòi hỏi phân tích hình thái học. Một phiên bản sư phạm tinh gọn sử dụng bảng tra cứu từ gốc (lemma table) kèm cơ chế dự phòng:

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Trường hợp cuối cùng mang lại bài học quan trọng: `watched` không nằm trong bảng tra cứu của ta và cơ chế dự phòng chỉ mới xử lý đuôi `ing`. Hệ thống lemmatization thực tế phải xử lý đuôi `ed`, động từ bất quy tắc, tính từ so sánh, danh từ số nhiều biến đổi âm vị (`children -> child`). Đó là lý do tại sao các hệ thống thương mại phải sử dụng WordNet, bộ morphologizer của spaCy, hoặc bộ phân tích hình thái học đầy đủ.

### Bước 4: Kết nối quy trình hoàn chỉnh

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Mảnh ghép còn thiếu là bộ gán nhãn từ loại (POS tagger). Bài 07 của Phase 5 (POS Tagging) sẽ xây dựng thành phần này. Hiện tại, ta tạm thời mặc định mọi từ là `NOUN` để hiểu rõ luồng dữ liệu.

## Use It

NLTK và spaCy cung cấp các bản triển khai chuẩn mực cho môi trường sản xuất. Mỗi thư viện chỉ mất vài dòng mã:

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` xử lý xuất sắc các từ viết tắt, ký tự Unicode và các trường hợp biên mà regex thường bỏ sót. `PorterStemmer` chạy đầy đủ cả năm giai đoạn. `WordNetLemmatizer` đòi hỏi phải chuyển đổi nhãn từ loại từ chuẩn Penn Treebank của NLTK sang bộ viết tắt của WordNet — đây chính là đoạn mã cầu nối mà hầu hết các bài hướng dẫn thường bỏ qua.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```text
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy đóng gói toàn bộ quy trình đằng sau lời gọi `nlp(text)`: Tách từ, gán nhãn từ loại và đưa về dạng gốc đều chạy đồng thời. Nhanh hơn NLTK rất nhiều khi xử lý quy mô lớn. Độ chính xác cao ngay từ đầu. Điểm đánh đổi là bạn khó có thể tuỳ ý hoán đổi từng thành phần nhỏ bên trong.

### Khi nào nên chọn công cụ nào?

| Tình huống | Lựa chọn tối ưu |
|------------|-----------------|
| Giảng dạy, nghiên cứu, cần hoán đổi linh hoạt các thành phần | NLTK |
| Môi trường sản xuất, đa ngôn ngữ, tốc độ cao | spaCy |
| Quy trình Transformer (đằng nào cũng phải tách từ theo tokenizer của mô hình) | Dùng trực tiếp `tokenizers` / `transformers` và bỏ qua tiền xử lý cổ điển |

### Hai trạng thái lỗi trong sản xuất ít ai cảnh báo bạn

Hầu hết các hướng dẫn chỉ dạy thuật toán rồi dừng lại. Hai vấn đề sau đây thường xuyên phá hỏng các hệ thống tiền xử lý thực tế:

**Trôi dạt tính tái lập (Reproducibility drift):** NLTK và spaCy thường thay đổi hành vi tách từ và lemmatizer giữa các phiên bản. Những gì từng tạo ra `['do', "n't"]` trong spaCy 2.x có thể chuyển thành `["don't"]` trong spaCy 3.x. Mô hình của bạn được huấn luyện trên một phân phối, nhưng khi suy luận lại chạy trên một phân phối khác. Độ chính xác suy giảm âm thầm mà không ai hiểu nguyên nhân. Giải pháp: Hãy cố định phiên bản thư viện trong `requirements.txt`. Viết một bài kiểm thử hồi quy (regression test) cố định kết quả tách từ của 20 câu mẫu và chạy kiểm tra mỗi khi nâng cấp thư viện.

**Lệch pha Huấn luyện / Suy luận (Training / inference mismatch):** Huấn luyện với tiền xử lý quá mạnh bạo (viết thường, xoá stopwords, stemming), nhưng khi triển khai thực tế lại đưa thẳng văn bản thô của người dùng vào, khiến độ chính xác lao dốc không phanh. Đây là lỗi phổ biến nhất trong các hệ thống NLP thực tế. Nếu bạn tiền xử lý thế nào khi huấn luyện, bạn bắt buộc phải chạy đúng hàm tiền xử lý đó khi suy luận. Hãy đóng gói hàm tiền xử lý vào bên trong package của mô hình, thay vì để dưới dạng một ô notebook cho đội ngũ triển khai tự viết lại.

## Ship It

Một câu lệnh prompt có thể tái sử dụng giúp các kỹ sư lựa chọn chiến lược tiền xử lý tối ưu mà không cần đọc lại hàng trăm trang giáo trình.

Lưu tại `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Exercises

1. **(Dễ)** Mở rộng hàm `tokenize` để giữ nguyên các địa chỉ URL như một token đơn lẻ. Kiểm tra: `tokenize("Visit https://example.com today.")` phải xuất ra một token URL hoàn chỉnh.
2. **(Trung bình)** Triển khai Bước 1b của Porter. Nếu một từ chứa nguyên âm và kết thúc bằng `ed` hoặc `ing`, hãy loại bỏ hậu tố đó. Xử lý quy tắc phụ âm kép (`hopping -> hop`, chứ không phải `hopp`).
3. **(Khó)** Xây dựng một bộ lemmatizer sử dụng WordNet làm bảng tra cứu nhưng tự động dự phòng sang bộ Porter stemmer khi WordNet không có từ đó. Đo lường độ chính xác trên một kho ngữ liệu có gán nhãn so với WordNet thuần và Porter thuần.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Token | Một từ | Đơn vị cơ sở mà mô hình tiếp nhận; có thể là từ, từ phụ (subword), ký tự hoặc byte |
| Stem (Gốc từ) | Gốc của từ | Kết quả sau khi chặt bỏ hậu tố dựa trên luật; không nhất thiết phải là một từ có nghĩa trong từ điển |
| Lemma (Dạng nguyên thể) | Dạng tra từ điển | Dạng chuẩn mực bạn tra cứu trong từ điển; đòi hỏi ngữ cảnh ngữ pháp để xác định chính xác |
| POS tag | Từ loại | Nhãn ngữ pháp như NOUN (danh từ), VERB (động từ), ADJ (tính từ); yếu tố quyết định để lemmatize chuẩn |
| Morphology (Hình thái học) | Quy tắc biến đổi hình thái từ | Cách một từ thay đổi hình dạng theo thì, số lượng, thể; lemmatization phụ thuộc sâu sắc vào điều này |

## Further Reading

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — Bài báo gốc, dài 5 trang, vẫn là tài liệu giải thích rõ ràng nhất
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — Cách tổ chức và kết nối một quy trình thực tế
- [NLTK book, chương 3](https://www.nltk.org/book/ch03.html) — Các trường hợp biên trong tách từ mà bạn chưa từng nghĩ tới

