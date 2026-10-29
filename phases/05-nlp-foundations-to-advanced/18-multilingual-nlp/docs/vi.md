# Xử lý Ngôn ngữ Tự nhiên Đa ngôn ngữ (Multilingual NLP)

> Một mô hình duy nhất, hơn 100 ngôn ngữ, không cần dữ liệu huấn luyện cho phần lớn các ngôn ngữ đó. Chuyển giao liên ngôn ngữ (Cross-lingual transfer) là phép màu thực tiễn của thập niên 2020.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Dịch máy)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Tiếng Anh có hàng tỷ ví dụ gán nhãn chất lượng cao. Tiếng Urdu chỉ có vài nghìn. Tiếng Maithili gần như không có gì. Bất kỳ hệ thống NLP thực tế nào muốn phục vụ người dùng toàn cầu đều bắt buộc phải hoạt động trên dải đuôi dài (long tail) của các ngôn ngữ khan hiếm tài nguyên — nơi mà dữ liệu huấn luyện đặc thù cho tác vụ hoàn toàn không tồn tại.

Các mô hình đa ngôn ngữ giải quyết nghịch lý này bằng cách huấn luyện một mạng nơ-ron duy nhất trên nhiều ngôn ngữ cùng lúc. Không gian biểu diễn dùng chung cho phép mô hình chuyển giao kỹ năng đã học từ các ngôn ngữ giàu tài nguyên sang các ngôn ngữ nghèo tài nguyên. Chỉ cần tinh chỉnh (fine-tune) mô hình trên tác vụ phân tích cảm xúc tiếng Anh, nó sẽ đưa ra các dự đoán cảm xúc trên tiếng Urdu tốt đến kinh ngạc ngay lập tức mà không cần học trước. Đó chính là **chuyển giao liên ngôn ngữ zero-shot (zero-shot cross-lingual transfer)**, và nó đã tái định hình toàn diện cách thức NLP được triển khai ra thế giới.

Bài học này làm rõ các đánh đổi kỹ thuật, các mô hình chuẩn mực kinh điển, và quyết định quan trọng nhất thường khiến các đội ngũ mới làm đa ngữ vấp ngã: lựa chọn ngôn ngữ nguồn phù hợp để chuyển giao tri thức.

## Khái niệm (The Concept)

![Chuyển giao liên ngôn ngữ thông qua không gian nhúng đa ngữ dùng chung](../assets/multilingual.svg)

**Không gian từ vựng dùng chung (Shared vocabulary).** Các mô hình đa ngữ sử dụng bộ tách từ SentencePiece hoặc WordPiece được huấn luyện trên tập ngữ liệu kết hợp từ tất cả các ngôn ngữ mục tiêu. Không gian từ vựng được chia sẻ: cùng một đơn vị từ phụ (subword unit) đại diện cho cùng một hình thái từ (morpheme) giữa các ngôn ngữ có quan hệ họ hàng. Tiền tố `anti-` trong tiếng Anh và tiếng Ý sẽ nhận chung một token ID.

**Không gian biểu diễn dùng chung (Shared representation).** Một mô hình Transformer được tiền huấn luyện bằng tác vụ mô hình hóa ngôn ngữ có mặt nạ (masked language modeling) trên nhiều ngôn ngữ sẽ tự động học được rằng các câu có ngữ nghĩa tương đồng ở các ngôn ngữ khác nhau sẽ tạo ra các trạng thái ẩn (hidden states) gần nhau. mBERT, XLM-R và NLLB đều thể hiện rõ đặc tính này. Vector nhúng của từ "cat" trong tiếng Anh sẽ tụ cụm gần "chat" trong tiếng Pháp và "gato" trong tiếng Tây Ban Nha; điều tương tự cũng diễn ra với vector nhúng toàn câu.

**Chuyển giao Zero-shot (Zero-shot transfer).** Tinh chỉnh mô hình trên dữ liệu gán nhãn của một ngôn ngữ duy nhất (thường là tiếng Anh). Khi suy luận, chạy trực tiếp mô hình trên bất kỳ ngôn ngữ nào khác mà mô hình hỗ trợ. Không cần bất kỳ nhãn dữ liệu nào của ngôn ngữ đích. Hiệu quả đặc biệt cao đối với các ngôn ngữ có sự tương đồng về loại hình ngôn ngữ học (typologically related) và giảm dần với các ngôn ngữ cách biệt xa về ngữ hệ.

**Tinh chỉnh Few-shot (Few-shot fine-tuning).** Bổ sung thêm chỉ từ 100 đến 500 ví dụ gán nhãn bằng ngôn ngữ đích. Độ chính xác sẽ lập tức nhảy vọt lên mức 95-98% so với mô hình cơ sở tiếng Anh trên các bài toán phân loại. Đây là đòn bẩy kỹ thuật mang lại hiệu quả chi phí tối ưu nhất trong toàn bộ lĩnh vực NLP đa ngôn ngữ.

## Các mô hình tiêu biểu

| Mô hình | Năm | Độ bao phủ | Ghi chú |
|---------|-----|------------|---------|
| mBERT | 2018 | 104 ngôn ngữ | Huấn luyện trên Wikipedia. Mô hình LM đa ngữ thực tiễn đầu tiên. Yếu trên ngôn ngữ ít tài nguyên. |
| XLM-R | 2019 | 100 ngôn ngữ | Huấn luyện trên CommonCrawl (lớn hơn nhiều so với Wikipedia). Thiết lập thước đo chuẩn cross-lingual. Bản Base 270M, Large 550M. |
| XLM-V | 2023 | 100 ngôn ngữ | Bản nâng cấp của XLM-R với từ vựng 1 triệu token (so với 250k). Vượt trội trên ngôn ngữ nghèo tài nguyên. |
| mT5 | 2020 | 101 ngôn ngữ | Kiến trúc T5 dành cho các tác vụ tạo sinh văn bản đa ngữ. |
| NLLB-200 | 2022 | 200 ngôn ngữ | Mô hình dịch thuật của Meta; hỗ trợ 55 ngôn ngữ đặc biệt ít tài nguyên. |
| BLOOM | 2022 | 46 ngôn ngữ tự nhiên + 13 ngôn ngữ lập trình | Mô hình LLM mở 176B tham số được huấn luyện đa ngữ từ đầu. |
| Aya-23 | 2024 | 23 ngôn ngữ | LLM đa ngữ của Cohere. Rất mạnh trên tiếng Ả Rập, Hindi, Swahili. |

Lựa chọn theo bài toán: Với tác vụ phân loại, `XLM-R-base` là lựa chọn mặc định an toàn và hiệu quả nhất. Tác vụ tạo sinh đòi hỏi `mT5` hoặc `NLLB` tùy thuộc vào bài toán dịch thuật hay tạo sinh mở. Các bài toán dạng LLM nên kết hợp `Aya-23` hoặc `Claude` với các kỹ thuật prompt đa ngữ tường minh.

## Quyết định chọn ngôn ngữ nguồn chuyển giao (Nghiên cứu năm 2026)

Hầu hết các kỹ sư đều mặc định chọn tiếng Anh làm ngôn ngữ nguồn để tinh chỉnh (fine-tuning). Các nghiên cứu thực nghiệm mới nhất (2026) chứng minh rằng điều này thường xuyên dẫn đến kết quả dưới mức tối ưu.

Mức độ tương đồng ngôn ngữ học dự đoán chất lượng chuyển giao chính xác hơn nhiều so với dung lượng ngữ liệu thô đơn thuần. Đối với các ngôn ngữ mục tiêu thuộc hệ Slav, tiếng Đức hoặc tiếng Nga thường mang lại kết quả chuyển giao vượt trội hơn tiếng Anh. Đối với các ngôn ngữ thuộc ngữ chi Ấn-Arya (Indic), tiếng Hindi thường đánh bại tiếng Anh. Chỉ số tương đồng **qWALS** (2026, dựa trên các đặc trưng của World Atlas of Language Structures) lượng hóa cụ thể điều này. **LANGRANK** (Lin et al., ACL 2019) là một phương pháp kinh điển khác giúp xếp hạng các ngôn ngữ nguồn ứng viên dựa trên sự kết hợp giữa tương đồng ngôn ngữ học, kích thước ngữ liệu và khoảng cách phả hệ ngôn ngữ.

Quy tắc thực hành: Nếu ngôn ngữ mục tiêu của bạn có một ngôn ngữ họ hàng giàu tài nguyên gần gũi về mặt loại hình, hãy thử tinh chỉnh trên ngôn ngữ đó trước, rồi so sánh đối chứng với kết quả tinh chỉnh từ tiếng Anh.

```figure
n5-crosslingual-bridge
```

## Xây dựng thực tế (Build It)

### Bước 1: Phân loại liên ngôn ngữ Zero-Shot

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Một mô hình duy nhất, ba ngôn ngữ hoàn toàn khác biệt, cùng một giao diện API. XLM-R được huấn luyện trên dữ liệu suy diễn tự nhiên (NLI) chuyển giao xuất sắc sang bài toán phân loại thông qua kỹ thuật kéo theo (entailment trick).

### Bước 2: Không gian vector nhúng đa ngôn ngữ

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Các câu mang cùng ý nghĩa dịch thuật sẽ nằm rất gần nhau trong không gian nhúng. Một câu tiếng Anh khác ngữ nghĩa sẽ nằm cách xa rõ rệt. Đây là cơ chế giúp cho việc truy xuất, phân cụm và tìm kiếm tương đồng liên ngôn ngữ hoạt động trơn tru.

### Bước 3: Chiến lược tinh chỉnh Few-Shot

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Với 100-500 ví dụ ngôn ngữ mục tiêu, `num_train_epochs=5` và `learning_rate=2e-5` là cấu hình an toàn chuẩn mực. Tốc độ học quá cao sẽ phá vỡ sự gióng hàng đa ngữ (multilingual alignment) và khiến mô hình bị suy thoái thành mô hình chỉ biết duy nhất tiếng Anh.

## Quy trình đánh giá thực chất

- **Độ chính xác theo từng ngôn ngữ riêng biệt trên tập kiểm thử:** Không bao giờ lấy số trung bình tổng hợp gộp chung. Con số gộp chung sẽ che giấu hoàn toàn sự thất bại ở các ngôn ngữ đuôi dài.
- **Đối chuẩn với baseline đơn ngữ (Monolingual baseline):** Đối với các ngôn ngữ có đủ dữ liệu, một mô hình đơn ngữ chuyên biệt huấn luyện từ đầu đôi khi đánh bại mô hình đa ngữ. Hãy kiểm chứng thực tế.
- **Kiểm thử cấp độ thực thể (Entity-level tests):** Các thực thể có tên trong ngôn ngữ mục tiêu. Các mô hình đa ngữ thường có khả năng tách token rất kém đối với các hệ chữ viết phi Latin.
- **Tính nhất quán liên ngôn ngữ (Cross-lingual consistency):** Cùng một nội dung ngữ nghĩa diễn đạt ở hai ngôn ngữ khác nhau phải đưa ra cùng một kết quả dự đoán. Đo lường khoảng cách sai lệch giữa chúng.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Tác vụ | Khuyến nghị |
|--------|-------------|
| Phân loại văn bản, 100 ngôn ngữ | `XLM-R-base` (~270M) tinh chỉnh |
| Phân loại văn bản zero-shot | `joeddav/xlm-roberta-large-xnli` |
| Vector nhúng câu đa ngữ | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Dịch thuật, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (xem bài học 11) |
| Tạo sinh đa ngôn ngữ | Claude, GPT-4, Aya-23, mT5-XXL |
| Ngôn ngữ đặc biệt nghèo tài nguyên | `XLM-V` hoặc fine-tune theo miền từ một ngôn ngữ giàu tài nguyên họ hàng |

Luôn dự trù ngân sách cho việc gắn nhãn few-shot trên ngôn ngữ mục tiêu nếu chất lượng là ưu tiên số một. Zero-shot chỉ là điểm xuất phát, không bao giờ là đích đến cuối cùng.

### Gánh nặng phân đoạn từ (The Tokenization Tax)

Các mô hình đa ngữ chia sẻ một bộ tách từ (tokenizer) duy nhất cho tất cả các ngôn ngữ của chúng. Tập từ vựng đó được huấn luyện trên một kho ngữ liệu bị chi phối áp đảo bởi tiếng Anh, tiếng Pháp, tiếng Tây Ban Nha, tiếng Trung và tiếng Đức. Đối với bất kỳ ngôn ngữ nào nằm ngoài nhóm thống trị này, ba "gánh nặng" sẽ âm thầm cộng dồn:

- **Gánh nặng độ màu mỡ phân đoạn (Fertility tax).** Văn bản của ngôn ngữ nghèo tài nguyên bị băm nhỏ thành số lượng token trên mỗi từ nhiều hơn rất nhiều so với tiếng Anh. Một câu tiếng Hindi có thể tốn số token gấp 3 đến 5 lần một câu tiếng Anh tương đương. Con số 3-5 lần đó nuốt chửng cửa sổ ngữ cảnh, làm giảm hiệu quả huấn luyện và gia tăng độ trễ suy luận.
- **Gánh nặng khôi phục biến thể (Variant recovery tax).** Mọi lỗi chính tả, biến thể dấu phụ, sự không khớp chuẩn hóa Unicode, hoặc biến thể chữ hoa/thường đều trở thành một chuỗi token hoàn toàn xa lạ trong không gian nhúng. Mô hình không thể học được các mối tương đồng chính tả mà một người bản ngữ thấy hiển nhiên.
- **Gánh nặng tràn dung lượng (Capacity spillover tax).** Hai gánh nặng trên làm tiêu hao vị trí ngữ cảnh, độ sâu các tầng mạng và các chiều kích vector nhúng. Phần dung lượng còn lại dành cho năng lực lập luận thực tế bị thu hẹp đáng kể so với những gì mà một ngôn ngữ giàu tài nguyên nhận được từ cùng một mô hình.

Triệu chứng thực tế: Mô hình của bạn huấn luyện hoàn toàn bình thường trên tiếng Hindi, đường cong loss giảm đẹp mắt, perplexity đánh giá trông rất hợp lý, nhưng đầu ra trong sản xuất lại sai lệch tinh vi. Cấu trúc hình thái học bị sụp đổ giữa câu. Các biến cách từ hiếm gặp không thể khôi phục được. **Bạn không thể mở rộng quy mô dữ liệu để cứu vãn một bộ tokenizer đã bị hỏng.**

Giải pháp phòng ngừa: Chọn bộ tokenizer có độ bao phủ tốt cho ngôn ngữ mục tiêu (từ vựng 1 triệu token của XLM-V là một bản sửa trực tiếp); kiểm tra độ màu mỡ tokenization trên văn bản mục tiêu trước khi huấn luyện; sử dụng cơ chế dự phòng cấp độ byte (SentencePiece với `byte_fallback=True`, byte-level BPE kiểu GPT-2) cho các hệ chữ viết cực hiếm để không bao giờ gặp lỗi OOV.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Lựa chọn ngôn ngữ nguồn, mô hình mục tiêu và kế hoạch đánh giá cho một tác vụ NLP đa ngôn ngữ.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Dựa trên các yêu cầu (các ngôn ngữ mục tiêu, loại tác vụ, lượng dữ liệu gán nhãn sẵn có theo từng ngôn ngữ), bạn đưa ra:

1. Ngôn ngữ nguồn để tinh chỉnh (fine-tuning): Mặc định là tiếng Anh; kiểm tra bảng xếp hạng LANGRANK hoặc chỉ số qWALS nếu ngôn ngữ mục tiêu có một ngôn ngữ họ hàng giàu tài nguyên gần gũi về loại hình học.
2. Mô hình nền tảng (Base model): XLM-R (cho phân loại), mT5 (cho tạo sinh), NLLB (cho dịch thuật), Aya-23 (cho LLM tạo sinh đa ngữ).
3. Ngân sách Few-shot: Bắt đầu với 100-500 ví dụ bằng ngôn ngữ mục tiêu nếu có thể thu thập được. Chỉ dùng zero-shot thuần túy khi việc gán nhãn là bất khả thi.
4. Kế hoạch đánh giá: Độ chính xác theo từng ngôn ngữ riêng lẻ (không dùng điểm gộp trung bình), tính nhất quán liên ngôn ngữ, F1 cấp độ thực thể trên các hệ chữ viết phi Latin.

Từ chối phát hành mô hình đa ngữ nếu thiếu quy trình đánh giá chi tiết theo từng ngôn ngữ riêng biệt — điểm số trung bình tổng hợp luôn che giấu các thất bại ở các ngôn ngữ đuôi dài. Cảnh báo các hệ chữ viết có độ bao phủ tokenization thấp (Amharic, Tigrinya, nhiều ngôn ngữ châu Phi) là bắt buộc phải dùng mô hình có cơ chế byte-fallback (SentencePiece với byte_fallback=True, hoặc bộ tách từ cấp độ byte như GPT-2).
```

## Bài tập (Exercises)

1. **Dễ.** Chạy đường ống phân loại zero-shot ở trên trên 10 câu cho mỗi ngôn ngữ trong số bốn thứ tiếng: Anh, Pháp, Hindi và Ả Rập. Báo cáo độ chính xác trên từng ngôn ngữ. Bạn sẽ thấy kết quả rất mạnh trên tiếng Pháp, khá tốt trên tiếng Hindi, và biến động thất thường trên tiếng Ả Rập.
2. **Trung bình.** Sử dụng `paraphrase-multilingual-MiniLM-L12-v2` để xây dựng một bộ truy xuất liên ngôn ngữ trên một kho dữ liệu nhỏ đa ngôn ngữ hỗn hợp. Truy vấn bằng tiếng Anh, truy xuất tài liệu bằng bất kỳ ngôn ngữ nào. Đo lường Recall@5.
3. **Khó.** So sánh hiệu quả tinh chỉnh giữa việc dùng tiếng Anh làm nguồn và dùng tiếng Hindi làm nguồn cho một tác vụ phân loại tiếng Hindi. Sử dụng 500 ví dụ tiếng Hindi để tinh chỉnh few-shot trong cả hai trường hợp. Báo cáo nguồn nào mang lại độ chính xác tiếng Hindi cao hơn và chênh lệch bao nhiêu phần trăm. Đây chính là luận điểm của LANGRANK được tái hiện ở quy mô thu nhỏ.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Mô hình đa ngữ (Multilingual model) | Một mô hình, nhiều thứ tiếng | Không gian từ vựng và tham số mạng được chia sẻ chung qua tất cả các ngôn ngữ. |
| Chuyển giao liên ngôn ngữ (Cross-lingual transfer) | Học một ngôn ngữ, chạy trên ngôn ngữ khác | Tinh chỉnh trên ngôn ngữ nguồn, đánh giá trên ngôn ngữ đích mà không cần nhãn dữ liệu ngôn ngữ đích. |
| Zero-shot | Không cần nhãn ngôn ngữ đích | Chuyển giao trực tiếp mà không cần bước tinh chỉnh trên ngôn ngữ mục tiêu. |
| Few-shot | Cần ít nhãn ngôn ngữ đích | Sử dụng từ 100 đến 500 ví dụ của ngôn ngữ mục tiêu để tinh chỉnh bổ sung. |
| mBERT | Mô hình LM đa ngữ đầu tiên | Mô hình BERT hỗ trợ 104 ngôn ngữ được tiền huấn luyện trên dữ liệu Wikipedia. |
| XLM-R | Thước đo chuẩn cross-lingual | Mô hình RoBERTa hỗ trợ 100 ngôn ngữ được tiền huấn luyện trên dữ liệu CommonCrawl quy mô lớn. |
| NLLB | Dịch máy 200 thứ tiếng của Meta | No Language Left Behind. Bao gồm 55 ngôn ngữ đặc biệt ít tài nguyên. |

## Đọc thêm (Further Reading)

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — bài báo gốc đặt nền móng cho mô hình XLM-R.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — bài phân tích tiên phong mở ra hướng nghiên cứu chuyển giao liên ngôn ngữ.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — bài báo gốc về họ mô hình NLLB-200.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — bài báo giới thiệu Aya, mô hình LLM đa ngữ mở của Cohere.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — bài báo khoa học về việc dùng qWALS và LANGRANK để chọn ngôn ngữ nguồn chuyển giao tối ưu.

