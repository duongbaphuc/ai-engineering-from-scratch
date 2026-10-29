# Xây dựng bộ tách từ từ con số không (Building a Tokenizer from Scratch)

> Bài 01 mang đến cho bạn một món đồ chơi. Bài học này sẽ trao cho bạn một vũ khí thực chiến.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 10, Bài 01 (Bộ tách từ: BPE, WordPiece, SentencePiece)
**Time:** ~90 phút

## Mục tiêu học tập

- Xây dựng một bộ tách từ BPE chuẩn sản xuất (production-grade BPE tokenizer) có khả năng xử lý Unicode, chuẩn hóa khoảng trắng và quản lý token đặc biệt
- Triển khai cơ chế dự phòng cấp byte (byte-level fallback) để bộ tách từ có thể mã hóa mọi đầu vào (bao gồm emoji, ký tự CJK và mã nguồn) mà không gặp token chưa biết
- Thêm các mẫu biểu thức chính quy tiền tách từ (pre-tokenization regex patterns) để phân cắt văn bản tại ranh giới từ trước khi áp dụng các phép hợp nhất BPE
- Huấn luyện một bộ tách từ tùy chỉnh trên ngữ liệu và đánh giá tỷ lệ nén (compression ratio) của nó so với `tiktoken` trên văn bản đa ngôn ngữ

## Vấn đề đặt ra

Bộ tách từ BPE của bạn từ Bài 01 hoạt động tốt trên văn bản tiếng Anh. Bây giờ, hãy thử đưa tiếng Nhật vào. Hoặc emoji. Hoặc mã nguồn Python với các khoảng trắng và tab đan xen.

Nó sẽ gãy vụn.

Không phải vì thuật toán BPE sai -- mà vì phần triển khai chưa hoàn thiện. Một bộ tách từ dùng trong môi trường sản xuất phải xử lý được các byte thô ở bất kỳ bảng mã nào, chuẩn hóa Unicode trước khi phân tách, quản lý các token đặc biệt không bao giờ bị hợp nhất, kết hợp tiền tách từ với phân tách từ con, và thực hiện tất cả những việc này đủ nhanh để không trở thành nút thắt cổ chai trong quy trình huấn luyện xử lý 15 nghìn tỷ token.

Bộ tách từ của GPT-2 có 50.257 token. Llama 3 có 128.256 token. GPT-4 có khoảng 100.000 token. Đây không phải là những con số mang tính minh họa. Bảng hợp nhất (merge table) phía sau các tập từ vựng đó được huấn luyện trên hàng trăm gigabyte văn bản, và toàn bộ cơ chế phụ trợ xung quanh -- chuẩn hóa, tiền tách từ, chèn token đặc biệt, định dạng mẫu hội thoại (chat template) -- chính là ranh giới phân định giữa một bộ tách từ chỉ xử lý được "hello world" và một bộ tách từ có thể xử lý toàn bộ dữ liệu internet.

Bạn sẽ tự tay xây dựng cỗ máy đó.

## Khái niệm cốt lõi

### Quy trình hoàn chỉnh (The Full Pipeline)

Một bộ tách từ trong môi trường sản xuất không chỉ là một thuật toán đơn lẻ. Nó là một quy trình (pipeline) gồm năm giai đoạn, mỗi giai đoạn giải quyết một bài toán riêng biệt.

```mermaid
graph LR
    A[Văn bản thô] --> B[Chuẩn hóa]
    B --> C[Tiền tách từ]
    C --> D[Hợp nhất BPE]
    D --> E[Token đặc biệt]
    E --> F[ID Token]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Mỗi giai đoạn đảm nhận một nhiệm vụ cụ thể:

| Giai đoạn | Vai trò | Tầm quan trọng |
|---|---|---|
| Chuẩn hóa (Normalize) | Chuẩn hóa Unicode NFKC, chuyển chữ thường (tùy chọn), lược bỏ dấu (tùy chọn) | Ký tự ghép "fi" (U+FB01) trở thành "fi" (hai ký tự riêng biệt). Nếu không có bước này, cùng một từ sẽ bị tách thành các token khác nhau. |
| Tiền tách từ (Pre-Tokenize) | Phân cắt văn bản thành các khối nhỏ trước khi chạy BPE | Ngăn BPE hợp nhất vượt qua ranh giới từ. Cụm "the cat" tuyệt đối không được tạo ra token "e c". |
| Hợp nhất BPE (BPE Merge) | Áp dụng các quy tắc hợp nhất đã học lên chuỗi byte | Cốt lõi của việc nén dữ liệu. Biến các byte thô thành các token từ con. |
| Token đặc biệt (Special Tokens) | Chèn các mốc đánh dấu cấu trúc như [BOS], [EOS], [PAD], dấu phân tách hội thoại | Các token này có ID cố định. Chúng không bao giờ tham gia vào các phép hợp nhất BPE. Mô hình cần chúng để định hình cấu trúc. |
| Ánh xạ ID (ID Mapping) | Chuyển đổi các chuỗi token thành các ID số nguyên | Mô hình chỉ nhìn thấy số nguyên, không phải chuỗi ký tự. |

### BPE cấp độ Byte (Byte-Level BPE)

Bộ tách từ trong Bài 01 hoạt động trên các byte UTF-8. Đó là hướng đi đúng đắn. Nhưng chúng ta đã bỏ qua một chi tiết quan trọng: điều gì xảy ra khi chuỗi byte đó không phải là UTF-8 hợp lệ?

Byte-level BPE giải quyết triệt để vấn đề này bằng cách coi mọi giá trị byte có thể có (từ 0 đến 255) là một token hợp lệ. Tập từ vựng cơ sở của bạn có đúng 256 phần tử. Bất kỳ tệp nào -- văn bản, dữ liệu nhị phân, hay dữ liệu bị hỏng -- đều có thể được tách từ mà không bao giờ sinh ra token chưa biết (unknown token).

GPT-2 đã bổ sung thêm một thủ thuật: ánh xạ mỗi byte sang một ký tự Unicode có thể in được để tập từ vựng giữ được khả năng đọc hiểu đối với con người. Byte `0x20` (dấu cách) trở thành ký tự `"Ġ"` trong bảng ánh xạ của họ. Đây hoàn toàn chỉ là hình thức trực quan, thuật toán cốt lõi bên dưới không bị ảnh hưởng.

Sức mạnh thực sự: byte-level BPE xử lý được mọi ngôn ngữ trên Trái Đất. Ký tự tiếng Trung chiếm 3 byte UTF-8 mỗi chữ. Tiếng Nhật có thể tốn 3-4 byte. Tiếng Ả Rập, Devanagari, emoji -- tất cả đơn thuần chỉ là các chuỗi byte. Thuật toán BPE tìm kiếm quy luật trong các chuỗi byte này theo cách thức hoàn toàn giống với khi nó tìm quy luật trong các byte ASCII tiếng Anh.

### Tiền tách từ (Pre-Tokenization)

Trước khi BPE chạm vào văn bản, bạn cần phân rã văn bản thành từng khối độc lập. Điều này ngăn thuật toán hợp nhất tạo ra các token bắc cầu qua ranh giới của hai từ.

GPT-2 sử dụng một mẫu biểu thức chính quy (regex) để phân tách văn bản:

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Mẫu này phân tách các từ viết tắt ("don't" trở thành "don" + "'t"), các từ có khoảng trắng đi kèm ở đầu (tùy chọn), số, dấu câu và khoảng trắng. Khoảng trắng ở đầu được giữ liền với từ đứng sau -- do đó "the cat" trở thành `[" the", " cat"]`, chứ không phải `["the", " ", "cat"]`.

Llama sử dụng SentencePiece, bỏ qua regex hoàn toàn. Nó coi luồng byte thô như một chuỗi dài duy nhất và để thuật toán BPE tự tìm ra ranh giới. Cách làm này đơn giản hơn nhưng cho phép BPE tự do hơn trong việc tạo ra các token xuyên biên giới từ.

Sự lựa chọn này rất quan trọng. Regex của GPT-2 ngăn bộ tách từ học được rằng ký tự "the" ở cuối một từ và "the" ở đầu từ tiếp theo nên được gộp lại. SentencePiece lại cho phép điều này, đôi khi giúp nén hiệu quả hơn nhưng các token tạo ra lại khó diễn giải hơn.

### Token đặc biệt (Special Tokens)

Mọi bộ tách từ chuẩn sản xuất đều dành riêng các ID token cho các mốc cấu trúc:

| Token | Mục đích | Mô hình sử dụng |
|---|---|---|
| `[BOS]` / `<s>` | Bắt đầu chuỗi (Beginning of sequence) | Llama 3, GPT |
| `[EOS]` / `</s>` | Kết thúc chuỗi (End of sequence) | Tất cả các mô hình |
| `[PAD]` | Đệm cho thẳng hàng batch (Padding) | BERT, T5 |
| `[UNK]` | Token không xác định (byte-level BPE loại bỏ token này) | BERT, WordPiece |
| `<\|im_start\|>` | Bắt đầu lượt tin nhắn hội thoại | ChatGPT, Qwen |
| `<\|im_end\|>` | Kết thúc lượt tin nhắn hội thoại | ChatGPT, Qwen |
| `<\|user\|>` | Đánh dấu lượt của người dùng | Llama 3 |
| `<\|assistant\|>` | Đánh dấu lượt của trợ lý | Llama 3 |

Token đặc biệt không bao giờ bị BPE chia nhỏ. Chúng được khớp chính xác trước khi thuật toán hợp nhất chạy, được thay thế bằng ID cố định của chúng, và văn bản xung quanh sẽ được tách từ bình thường.

### Mẫu hội thoại (Chat Templates)

Đây là nơi hầu hết mọi người cảm thấy bối rối và là nơi nhiều mã nguồn triển khai bị lỗi nhất.

Khi gửi tin nhắn đến một mô hình hội thoại, API sẽ nhận vào một danh sách các tin nhắn:

```json
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Mô hình không đọc định dạng JSON. Nó chỉ nhìn thấy một chuỗi token phẳng. Mẫu hội thoại (chat template) sẽ chuyển đổi danh sách các tin nhắn thành chuỗi phẳng đó bằng cách sử dụng các token đặc biệt. Mỗi mô hình thực hiện việc này theo một định dạng riêng:

```text
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Nếu áp dụng sai mẫu template, mô hình sẽ sinh ra kết quả vô nghĩa. Nó đã được huấn luyện trên đúng một định dạng chuẩn xác duy nhất. Bất kỳ sự sai lệch nào -- một ký tự xuống dòng bị thiếu, một token bị hoán đổi, một khoảng trắng thừa -- đều đẩy dữ liệu đầu vào ra ngoài phân phối huấn luyện (out-of-distribution).

### Tốc độ xử lý (Speed)

Python quá chậm cho tác vụ tách từ trong môi trường sản xuất.

`tiktoken` (OpenAI) được viết bằng Rust với liên kết Python. Thư viện `tokenizers` của HuggingFace cũng viết bằng Rust. `SentencePiece` được viết bằng C++. Các thư viện này đạt tốc độ nhanh hơn từ 10 đến 100 lần so với Python thuần túy.

Để dễ hình dung: việc tách từ cho 15 nghìn tỷ token để tiền huấn luyện Llama 3 với tốc độ 1 triệu token mỗi giây (tốc độ Python tối ưu) sẽ mất 174 ngày. Nhưng với tốc độ 100 triệu token mỗi giây (Rust), quá trình này chỉ mất 1,7 ngày.

Chúng ta xây dựng bằng Python để thấu hiểu bản chất thuật toán. Trong môi trường thực tế, bạn sẽ dùng bản triển khai đã được biên dịch và chỉ tương tác qua lớp vỏ Python bên ngoài.

```figure
weight-tying
```

## Tự tay xây dựng (Build It)

### Bước 1: Mã hóa cấp độ Byte (Byte-Level Encoding)

Nền tảng cốt lõi: chuyển đổi bất kỳ chuỗi nào thành một chuỗi byte, ánh xạ từng byte sang ký tự có thể in để hiển thị, và đảo ngược quá trình.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Thử nghiệm trên văn bản đa ngôn ngữ để quan sát số lượng byte:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

`"hello"` tốn 5 byte. `"你好"` tốn 6 byte (3 byte mỗi ký tự). Emoji ngọn lửa tốn 4 byte. Bộ tách từ cấp byte không quan tâm văn bản thuộc ngôn ngữ nào. Byte đơn giản là byte.

### Bước 2: Bộ tiền tách từ với biểu thức chính quy (Pre-Tokenizer with Regex)

Phân chia văn bản thành các khối nhỏ bằng mẫu regex của GPT-2. Mỗi khối sau đó sẽ được BPE tách từ độc lập.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Module `regex` hỗ trợ các ký hiệu phân loại thuộc tính Unicode (`\p{L}` cho chữ cái, `\p{N}` cho số). Thư viện chuẩn `re` không hỗ trợ nên chúng ta dự phòng về các lớp ký tự ASCII. Trong môi trường sản xuất đa ngôn ngữ, hãy cài đặt `regex`.

Thử nghiệm:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Khoảng trắng ở đầu được gắn liền với từ. Từ viết tắt bị tách ra ở dấu nháy đơn. Dấu câu trở thành khối riêng. BPE sẽ không bao giờ hợp nhất các token vượt qua các ranh giới này.

### Bước 3: Thuật toán BPE trên chuỗi Byte

Thuật toán cốt lõi từ Bài 01, nhưng bây giờ hoạt động trên các khối đã được tiền tách từ một cách độc lập.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Bước 4: Xử lý Token đặc biệt (Special Token Handling)

Token đặc biệt đòi hỏi phải khớp chính xác và có ID cố định. Chúng bỏ qua BPE hoàn toàn.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Bước 5: Lớp Tokenizer hoàn chỉnh (Full Tokenizer Class)

Kết nối tất cả các khâu lại với nhau: chuẩn hóa, tách theo token đặc biệt, tiền tách từ, hợp nhất BPE, ánh xạ sang ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Bước 6: Thử nghiệm đa ngôn ngữ (Multilingual Test)

Bài kiểm tra thực tế. Đưa tiếng Anh, tiếng Trung, emoji và code vào xử lý.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Ký tự tiếng Trung sinh ra 3 byte mỗi ký tự. Emoji sinh ra 4 byte. Không có trường hợp nào gây sập bộ tách từ. Không có token chưa biết. Đó chính là sức mạnh của byte-level BPE.

## Sử dụng trong thực tế (Use It)

### So sánh các bộ tách từ thực tế

Nạp các bộ tách từ thực tế của Llama 3, GPT-4 và Mistral. Quan sát cách mỗi bộ tách từ xử lý cùng một đoạn văn đa ngôn ngữ.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Bạn sẽ thấy số lượng token khác nhau cho cùng một văn bản. Llama 3 với từ vựng 128K hợp nhất các mẫu thông dụng mạnh mẽ hơn. GPT-4 với 100K nằm ở mức trung gian. Mistral với 32K sinh ra nhiều token hơn nhưng lại có lớp nhúng (embedding layer) gọn nhẹ hơn.

Sự đánh đổi luôn nhất quán: từ vựng lớn hơn đồng nghĩa với chuỗi token ngắn hơn nhưng tốn nhiều tham số hơn.

## Bàn giao sản phẩm (Ship It)

Bài học này tạo ra artifact `outputs/prompt-tokenizer-builder.md` -- một prompt chuyên dụng để xây dựng và gỡ lỗi các bộ tách từ chuẩn sản xuất.

## Bài tập thực hành

1. **Dễ:** Thêm phương thức `get_token_bytes(id)` để hiển thị các byte thô của bất kỳ ID token nào. Dùng nó để kiểm tra xem các token được hợp nhất phổ biến nhất thực sự đại diện cho những chuỗi ký tự nào.
2. **Trung bình:** Triển khai bộ tiền tách từ theo phong cách của Llama: phân tách dựa trên khoảng trắng và chữ số nhưng giữ lại khoảng trắng ở đầu. So sánh tập từ vựng của nó với cách tiếp cận regex của GPT-2 trên cùng một ngữ liệu.
3. **Khó:** Thêm một phương thức chat template nhận vào danh sách các tin nhắn dạng `{"role": ..., "content": ...}` và tạo ra chuỗi token chính xác theo định dạng hội thoại của Llama 3. Kiểm thử kết quả so với bản triển khai của HuggingFace.

## Các thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Byte-level BPE | "Bộ tách từ hoạt động trên byte" | BPE với tập từ vựng cơ sở gồm 256 giá trị byte -- xử lý được mọi đầu vào mà không bao giờ gặp token chưa biết |
| Tiền tách từ (Pre-tokenization) | "Tách từ trước khi chạy BPE" | Việc phân tách dựa trên biểu thức chính quy hoặc luật nhằm ngăn BPE hợp nhất qua ranh giới từ |
| Chuẩn hóa NFKC (NFKC normalization) | "Dọn dẹp Unicode" | Phân rã theo chuẩn tương thích rồi ghép nối chuẩn tắc -- chữ ghép "fi" trở thành "fi", chữ "A" dạng fullwidth trở thành "A" |
| Mẫu hội thoại (Chat template) | "Cách biến tin nhắn thành token" | Định dạng chính xác để chuyển đổi danh sách tin nhắn vai trò/nội dung thành chuỗi token phẳng -- đặc thù cho từng mô hình và bắt buộc phải khớp với định dạng lúc huấn luyện |
| Token đặc biệt (Special tokens) | "Các token điều khiển" | Các ID token được dành riêng và bỏ qua BPE -- [BOS], [EOS], [PAD], các mốc hội thoại -- được khớp chính xác trước khi hợp nhất |
| Độ sinh từ (Fertility) | "Số token trên mỗi từ" | Tỷ lệ giữa số token đầu ra và số từ đầu vào -- khoảng 1,3 với tiếng Anh trong GPT-4, 2-3 với tiếng Hàn; tỷ lệ càng cao nghĩa là ngữ cảnh càng bị lãng phí |
| tiktoken | "Bộ tách từ của OpenAI" | Bản triển khai BPE viết bằng Rust với liên kết Python -- nhanh hơn từ 10 đến 100 lần so với Python thuần |
| Bảng hợp nhất (Merge table) | "Tập từ vựng" | Danh sách có thứ tự các cặp byte được hợp nhất qua quá trình huấn luyện -- đây CHÍNH LÀ tri thức đã học của bộ tách từ |

## Tài liệu đọc thêm

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- Triển khai BPE bằng Rust được sử dụng bởi GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Thư viện tách từ bằng Rust hỗ trợ BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- Chi tiết về tập từ vựng 128K và quy trình huấn luyện bộ tách từ
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- Kỹ thuật tách từ độc lập ngôn ngữ
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- Bảng ánh xạ byte sang Unicode nguyên bản
