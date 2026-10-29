# Phân đoạn Từ phụ — BPE, WordPiece, Unigram, SentencePiece (Subword Tokenization)

> Bộ tách từ cấp độ từ bị nghẽn trước các từ chưa từng thấy. Bộ tách từ cấp độ ký tự làm bùng nổ độ dài chuỗi. Bộ tách từ cấp độ từ phụ dung hòa hoàn hảo cả hai. Mọi mô hình LLM hiện đại đều vận hành trên nền tảng này.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Xử lý Văn bản), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 phút

## Vấn đề đặt ra (The Problem)

Từ điển của bạn có 50.000 từ vựng. Người dùng gõ "untokenizable". Bộ tách từ của bạn không nhận diện được và trả về token `[UNK]`. Giờ đây mô hình hoàn toàn mù tịt về ý nghĩa của từ này. Tồi tệ hơn: tài liệu ở phân vị thứ 90 trong kho ngữ liệu của bạn chứa tới 40 từ hiếm, đồng nghĩa với việc có 40 mẩu thông tin bị xóa sổ hoàn toàn trên mỗi tài liệu.

Kỹ thuật phân đoạn từ phụ (Subword tokenization) giải quyết triệt để vấn đề này: Các từ thông dụng được giữ nguyên thành một token đơn lẻ. Các từ hiếm gặp được phân rã thành các mảnh ghép có ý nghĩa: `untokenizable` → `un`, `token`, `izable`. Dữ liệu huấn luyện có thể bao quát 100% mọi văn bản vì bất kỳ chuỗi ký tự nào xét cho cùng cũng chỉ là một chuỗi các byte nhị phân.

Mọi LLM hàng đầu năm 2026 đều vận hành dựa trên một trong ba thuật toán (BPE, Unigram, WordPiece), được bao bọc trong một trong ba thư viện (tiktoken, SentencePiece, Hugging Face Tokenizers). Bạn không thể xây dựng hoặc triển khai một mô hình ngôn ngữ mà không nắm vững cách lựa chọn bộ tách từ phù hợp.

## Khái niệm (The Concept)

![BPE so với Unigram và WordPiece, chi tiết từng ký tự](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Bắt đầu từ không gian từ vựng ở cấp độ ký tự. Đếm tần suất xuất hiện của tất cả các cặp ký tự liền kề. Hợp nhất (merge) cặp ký tự có tần suất cao nhất thành một token mới. Lặp lại quá trình này cho đến khi đạt được kích thước từ vựng mục tiêu. Đây là thuật toán chiếm ưu thế áp đảo: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** Cùng một thuật toán nhưng chạy trực tiếp trên các byte thô (256 token cơ sở ban đầu) thay vì các ký tự Unicode. Đảm bảo tỷ lệ token `[UNK]` bằng 0 tuyệt đối — bất kỳ chuỗi byte nào cũng đều mã hóa được. GPT-2 sử dụng 50.257 token (256 byte + 50.000 phép hợp nhất + 1 token đặc biệt).

**Unigram.** Bắt đầu từ một tập từ vựng ứng viên khổng lồ. Gán cho mỗi token một xác suất unigram. Tỉa dần (prune) các token mà việc loại bỏ chúng làm giảm log-likelihood của toàn bộ ngữ liệu ít nhất. Mang tính xác suất khi suy luận: có thể lấy mẫu các cách phân đoạn khác nhau (rất hữu ích cho việc tăng cường dữ liệu thông qua điều chuẩn từ phụ - subword regularization). Được sử dụng bởi T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Hợp nhất các cặp ký tự giúp tối đa hóa khả năng hợp lý (likelihood) của tập ngữ liệu huấn luyện thay vì chỉ dựa vào tần số đếm thô. Được sử dụng bởi BERT, DistilBERT, ELECTRA.

**SentencePiece so với tiktoken.** SentencePiece là thư viện chuyên dùng để *huấn luyện* các bộ từ vựng (theo chuẩn BPE hoặc Unigram) trực tiếp trên văn bản Unicode thô mà không cần bước tiền tách từ, mã hóa dấu cách thành ký tự `▁`. tiktoken là thư viện *mã hóa suy luận* cực nhanh bằng Rust của OpenAI dựa trên các bộ từ vựng đã xây dựng sẵn; nó không hỗ trợ huấn luyện từ vựng mới từ đầu.

Quy tắc lựa chọn:

- **Huấn luyện một bộ từ vựng mới:** Dùng SentencePiece (hỗ trợ đa ngôn ngữ, không cần tiền xử lý tách từ) hoặc HF Tokenizers.
- **Suy luận tốc độ cao với từ vựng chuẩn GPT:** Dùng tiktoken (`cl100k_base`, `o200k_base`).
- **Cả hai nhu cầu:** Dùng HF Tokenizers — một thư viện duy nhất cho cả huấn luyện và phục vụ suy luận.

```figure
bpe-merge
```

## Xây dựng thực tế (Build It)

### Bước 1: Thuật toán BPE từ đầu

Xem mã nguồn chi tiết trong `code/main.py`. Vòng lặp cốt lõi:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Ba đặc tính quan trọng được mã hóa trong thuật toán: Hậu tố `</w>` đánh dấu kết thúc từ để phân biệt rõ ràng "low" (từ đứng độc lập) với "lower" (tiền tố của từ khác). Trọng số tần suất giúp các cặp xuất hiện nhiều được hợp nhất sớm nhất. Danh sách hợp nhất có thứ tự nghiêm ngặt — khi suy luận, các phép ghép nối phải được áp dụng đúng theo trình tự lúc huấn luyện.

### Bước 2: Mã hóa với danh sách hợp nhất đã học

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Bản cài đặt thô sơ này có độ phức tạp $O(n \cdot \lvert \text{merges} \rvert)$. Các bản cài đặt trong sản xuất (tiktoken, HF Tokenizers) sử dụng bảng tra cứu thứ hạng hợp nhất (merge-rank lookup) kết hợp hàng đợi ưu tiên (priority queues) và chạy với độ phức tạp gần như tuyến tính.

### Bước 3: Ứng dụng thực tế với SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # hoặc "unigram"
    character_coverage=0.9995, # thấp hơn cho tiếng Nhật/Trung/Hàn (0.9995 cho tiếng Anh, 0.995 cho tiếng Nhật)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Điểm lưu ý: Không cần bước tiền tách từ bằng khoảng trắng, dấu cách được mã hóa thành ký tự `▁`, tham số `character_coverage` kiểm soát mức độ bảo toàn các ký tự hiếm so với việc ánh xạ chúng vào token `<unk>`.

### Bước 4: tiktoken cho các bộ từ vựng tương thích OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Chỉ phục vụ mã hóa (Encoding-only). Tốc độ cực nhanh (viết bằng Rust). Khớp chính xác 100% với cách tách token của GPT-4/5 để đếm byte, ước tính chi phí API và dự trù ngân sách cửa sổ ngữ cảnh.

## Những cạm bẫy vẫn xuất hiện trong sản xuất năm 2026

- **Trôi dạt Tokenizer (Tokenizer drift).** Huấn luyện mô hình trên bộ từ vựng A, nhưng lại triển khai suy luận với bộ từ vựng B. ID của các token bị lệch hoàn toàn; mô hình xuất ra văn bản rác vô nghĩa. Luôn kiểm tra mã băm hash của tệp `tokenizer.json` trong pipeline CI.
- **Sự nhập nhằng của khoảng trắng (Whitespace ambiguity).** BPE coi "hello" và " hello" (có dấu cách phía trước) là hai token hoàn toàn khác nhau. Luôn chỉ định rõ ràng các tham số `add_special_tokens` và `add_prefix_space`.
- **Thiếu hụt dữ liệu huấn luyện đa ngữ.** Các kho ngữ liệu nặng về tiếng Anh sẽ tạo ra các bộ từ vựng phân rã chữ viết phi Latin thành số lượng token gấp 5-10 lần. Cùng một câu prompt nhưng chi phí đắt gấp 5-10 lần trên tiếng Nhật/Ả Rập khi dùng GPT-3.5. Bộ mã hóa `o200k_base` đã khắc phục một phần sự bất bình đẳng này.
- **Tách vỡ Emoji.** Một biểu tượng cảm xúc emoji đơn lẻ có thể bị băm thành 5 token khác nhau. Cần kiểm tra kỹ cách xử lý emoji khi tính toán giới hạn ngữ cảnh.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn |
|------------|----------|
| Huấn luyện mô hình đơn ngữ từ đầu | HF Tokenizers (chuẩn BPE) |
| Huấn luyện mô hình đa ngôn ngữ | SentencePiece (chuẩn Unigram, `character_coverage=0.9995`) |
| Phục vụ API tương thích chuẩn OpenAI | tiktoken (`o200k_base` cho GPT-4+) |
| Từ vựng đặc thù miền (mã nguồn, toán học, protein) | Tự huấn luyện BPE tùy biến trên ngữ liệu miền, ghép nối với từ vựng cơ sở |
| Suy luận trên thiết bị biên, mô hình siêu nhỏ | Unigram (các từ vựng kích thước nhỏ hoạt động hiệu quả hơn) |

Kích thước từ vựng là một quyết định mở rộng quy mô (scaling decision), không phải một hằng số cố định. Kinh nghiệm thực tiễn: 32k cho mô hình <1B tham số, 50-100k cho mô hình 1-10B, và 200k+ cho các mô hình đa ngữ hoặc frontier LLM.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Lựa chọn thuật toán tách từ, kích thước từ vựng và thư viện cho một kho ngữ liệu và mục tiêu triển khai cụ thể.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Dựa trên kho ngữ liệu (kích thước, danh sách ngôn ngữ, miền nghiệp vụ) và mục tiêu triển khai (huấn luyện từ đầu / tinh chỉnh fine-tuning / suy luận tương thích API), bạn đưa ra:

1. Thuật toán: BPE, Unigram, hoặc WordPiece. Giải thích lý do trong một câu.
2. Thư viện: SentencePiece, HF Tokenizers, hoặc tiktoken. Giải thích lý do.
3. Kích thước từ vựng: Làm tròn đến hàng nghìn gần nhất. Gắn liền với kích thước mô hình và độ bao phủ ngôn ngữ.
4. Thiết lập độ bao phủ: `character_coverage`, `byte_fallback`, danh sách các token đặc biệt.
5. Kế hoạch kiểm thực: Số token trung bình trên mỗi từ trên tập kiểm thử riêng biệt, tỷ lệ OOV, tỷ lệ nén (compression ratio), tính bằng nhau tuyệt đối khi giải mã hai chiều (round-trip decode equality).

Từ chối huấn luyện tokenizer có character-coverage < 0.995 trên các kho ngữ liệu chứa hệ chữ viết hiếm. Từ chối phát hành từ vựng nếu thiếu bước kiểm tra mã băm hash cố định của tệp `tokenizer.json` trong CI. Đánh dấu bất kỳ tokenizer đơn ngữ nào có kích thước từ vựng dưới 16k là có nguy cơ không đạt yêu cầu kỹ thuật.
```

## Bài tập (Exercises)

1. **Dễ.** Huấn luyện một mô hình BPE với 500 phép hợp nhất trên tập ngữ liệu nhỏ trong `code/main.py`. Mã hóa ba từ chưa từng xuất hiện. Có bao nhiêu từ tạo ra đúng 1 token so với >1 token?
2. **Trung bình.** So sánh số lượng token trên 100 câu tiếng Anh trích từ Wikipedia giữa `cl100k_base`, `o200k_base` và một mô hình SentencePiece BPE do bạn tự huấn luyện với kích thước từ vựng 32k. Báo cáo tỷ lệ nén (compression ratio) của từng phương án.
3. **Khó.** Huấn luyện trên cùng một kho ngữ liệu với cả ba thuật toán: BPE, Unigram và WordPiece. Đo lường độ chính xác hạ nguồn khi dùng mỗi bộ tách từ này trên một bộ phân loại cảm xúc nhỏ. Liệu sự lựa chọn thuật toán tokenizer có làm thay đổi điểm F1 quá 1 điểm phần trăm hay không?

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| BPE | Byte-Pair Encoding | Hợp nhất tham lam các cặp ký tự xuất hiện nhiều nhất cho đến khi đạt kích thước từ vựng mục tiêu. |
| Byte-level BPE | Không bao giờ có token lạ | BPE chạy trên 256 byte thô cơ sở; GPT-2 và Llama sử dụng phương pháp này. |
| Unigram | Bộ tách từ xác suất | Tỉa bớt các token từ một tập ứng viên khổng lồ dựa trên hàm log-likelihood; dùng trong T5, Gemma. |
| SentencePiece | Tách từ dựa trên khoảng trắng đặc biệt | Thư viện huấn luyện BPE/Unigram trực tiếp trên văn bản thô; dấu cách được mã hóa thành `▁`. |
| tiktoken | Bộ tách từ siêu tốc | Thư viện mã hóa BPE viết bằng Rust của OpenAI cho các từ vựng dựng sẵn. Không hỗ trợ huấn luyện mới. |
| Danh sách hợp nhất (Merge list) | Trật tự ghép nối | Danh sách có thứ tự của các phép ghép `(a, b) → ab`; khi suy luận phải áp dụng chuẩn xác theo thứ tự này. |
| Độ bao phủ ký tự (Character coverage) | Hiếm bao nhiêu thì bỏ? | Tỷ lệ phần trăm các ký tự trong tập huấn luyện mà tokenizer bắt buộc phải bao phủ; thường đặt mức ~0.9995. |

## Đọc thêm (Further Reading)

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — bài báo gốc đặt nền móng cho việc ứng dụng BPE trong NLP.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — bài báo kinh điển về Unigram LM.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — bài báo giới thiệu thư viện SentencePiece.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — tài liệu tóm tắt súc tích, dễ hiểu về các loại tokenizer.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — mã nguồn mở, hướng dẫn và danh sách các bảng mã của tiktoken.

