# Dịch máy (Machine Translation)

> Dịch thuật là tác vụ đã tài trợ cho các nghiên cứu NLP suốt ba mươi năm qua và vẫn đang tiếp tục sinh lời cho đến hôm nay.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Cơ chế Chú ý), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Mô hình đọc một câu ở một ngôn ngữ và tạo ra một câu ở một ngôn ngữ khác. Độ dài câu thay đổi. Trật tự từ thay đổi. Một số từ nguồn ánh xạ sang nhiều từ đích và ngược lại. Các thành ngữ không bao giờ chấp nhận phép ánh xạ một-một. Cụm từ "I miss you" trong tiếng Pháp là "tu me manques" — nghĩa đen từng chữ là "bạn đang thiếu vắng đối với tôi." Không một phép gióng hàng cấp độ từ (word-level alignment) nào có thể tồn tại nguyên vẹn trước sự khác biệt đó.

Dịch máy (Machine Translation - MT) chính là bài toán lịch sử đã buộc NLP phải phát minh ra kiến trúc encoder-decoder, cơ chế chú ý (attention), Transformer, và cuối cùng dẫn lối đến toàn bộ kỷ nguyên LLM ngày nay. Mọi bước tiến vượt bậc xuất hiện vì chất lượng dịch thuật có thể đo lường được bằng số liệu rõ ràng và khoảng cách giữa bản dịch của con người với máy móc từng là một thách thức dai dẳng.

Bài học này bỏ qua các bài học lịch sử dài dòng để đi thẳng vào đường ống vận hành thực tế chuẩn mực của năm 2026: Sử dụng các mô hình encoder-decoder đa ngôn ngữ huấn luyện sẵn (NLLB-200 hoặc mBART), tách từ cấp độ từ phụ (subword tokenization), tìm kiếm chùm (beam search), đánh giá bằng BLEU và chrF, cùng việc nhận diện các phương thức lỗi (failure modes) vẫn âm thầm lọt vào môi trường sản xuất.

## Khái niệm (The Concept)

![Pipeline dịch máy: tách token → mã hóa → giải mã với attention → khôi phục văn bản](../assets/mt-pipeline.svg)

Dịch máy nơ-ron hiện đại (NMT) là một mô hình Transformer encoder-decoder được huấn luyện trên dữ liệu văn bản song ngữ/song hành (parallel text). Bộ mã hóa (encoder) đọc câu nguồn theo cách phân đoạn token của ngôn ngữ nguồn. Bộ giải mã (decoder) sinh ra văn bản đích, từng từ phụ một, sử dụng biểu diễn đầu ra của encoder thông qua cơ chế chú ý chéo (cross-attention - bài học 10). Quá trình giải mã sử dụng tìm kiếm chùm (beam search) để tránh bẫy giải mã tham lam (greedy decoding). Đầu ra được ghép nối lại (detokenized), khôi phục chữ hoa/thường (detruecased), và chấm điểm so sánh với bản dịch tham chiếu chuẩn (reference).

Ba quyết định mang tính vận hành quyết định chất lượng dịch thuật thực tế:

- **Bộ tách từ (Tokenizer).** SentencePiece BPE được huấn luyện trên tập ngữ liệu đa ngôn ngữ hỗn hợp. Không gian từ vựng dùng chung giữa các ngôn ngữ chính là yếu tố cho phép NLLB dịch zero-shot giữa các cặp ngôn ngữ chưa từng thấy cùng nhau.
- **Kích thước mô hình (Model size).** NLLB-200 distilled 600M chạy mượt mà trên máy tính cá nhân. NLLB-200 3.3B là chuẩn mặc định cho hệ thống sản xuất. Phiên bản 54.5B là trần giới hạn trong nghiên cứu khoa học.
- **Chiến lược giải mã (Decoding).** Độ rộng chùm (beam width) từ 4 đến 5 cho nội dung tổng quát. Hình phạt độ dài (length penalty) để tránh đầu ra quá ngắn. Giải mã có ràng buộc (constrained decoding) khi bạn cần sự nhất quán tuyệt đối về thuật ngữ chuyên ngành.

```figure
seq2seq-alignment
```

## Xây dựng thực tế (Build It)

### Bước 1: Gọi mô hình dịch máy huấn luyện sẵn

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Ba tham số quan trọng ở đây: `src_lang` chỉ định cho tokenizer hệ chữ viết và cách phân đoạn từ cần áp dụng. `forced_bos_token_id` chỉ định cho decoder ngôn ngữ đích cần sinh ra. Cả hai đều là quy ước riêng của NLLB; mBART và M2M-100 sử dụng các quy ước riêng và không thể thay thế lẫn nhau.

### Bước 2: Đánh giá bằng BLEU và chrF

BLEU đo lường sự trùng lặp n-gram giữa bản dịch của mô hình và bản dịch tham chiếu. Nó xem xét bốn kích thước n-gram (1-4), lấy trung bình nhân của precision, và áp dụng hình phạt ngắn (brevity penalty) nếu câu sinh ra quá ngắn. Điểm số nằm trong thang [0, 100]. Rất phổ biến nhưng khó giải thích: 30 BLEU là "dùng được"; 40 là "tốt"; 50 là "xuất sắc"; sự chênh lệch dưới 1 điểm BLEU thường chỉ là nhiễu ngẫu nhiên.

chrF đo điểm F-score ở cấp độ ký tự (character-level). Nó nhạy bén và công bằng hơn đối với các ngôn ngữ giàu hình thái (morphologically rich languages) — nơi mà BLEU thường đánh giá thấp vì bỏ qua các biến thể từ. Thường được báo cáo song song với BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Luôn luôn sử dụng thư viện `sacrebleu`. Nó chuẩn hóa các bước tách từ để điểm số có thể so sánh công bằng giữa các nghiên cứu. Tự viết mã tính BLEU là nguyên nhân dẫn đến các báo cáo benchmark sai lệch.

### Phân tầng đánh giá 3 cấp độ (Năm 2026)

Quy trình đánh giá dịch máy hiện đại sử dụng ba nhóm chỉ số bổ trợ cho nhau. Hãy phát hành hệ thống với ít nhất hai nhóm:

- **Chỉ số heuristic** (BLEU, chrF). Nhanh, dựa trên bản dịch tham chiếu, dễ tính toán, nhưng không nhạy với cách diễn đạt đồng nghĩa (paraphrase). Sử dụng để so sánh với các hệ thống cũ và phát hiện suy thoái hiệu năng (regression).
- **Chỉ số học máy (Learned metrics)** (COMET, BLEURT, BERTScore). Các mô hình nơ-ron được huấn luyện dựa trên đánh giá của con người; so sánh độ tương đồng ngữ nghĩa giữa bản dịch với câu nguồn và câu tham chiếu. COMET có độ tương quan cao nhất với đánh giá của con người từ năm 2023 và là chuẩn sản xuất năm 2026 nơi chất lượng là ưu tiên số một.
- **LLM làm giám khảo (LLM-as-judge)** (không cần bản dịch tham chiếu - reference-free). Sử dụng prompt để yêu cầu một mô hình lớn chấm điểm bản dịch về độ trôi chảy (fluency), độ thỏa đáng (adequacy), văn phong và sự phù hợp văn hóa. GPT-4 làm giám khảo đạt độ đồng thuận ~80% với chuyên gia con người khi tiêu chí chấm điểm (rubric) được thiết kế chặt chẽ. Dùng cho nội dung mở nơi không có sẵn bản dịch tham chiếu chuẩn.

Ngăn xếp thực chiến năm 2026: `sacrebleu` cho BLEU và chrF, `unbabel-comet` cho COMET, và LLM qua prompt cho bước đánh giá cuối cùng trước khi chuyển tới người dùng. Hãy chuẩn hóa (calibrate) mọi chỉ số trên 50-100 ví dụ được con người thẩm định trước khi tin tưởng nó trên dữ liệu sản xuất quy mô lớn.

Các chỉ số không cần tham chiếu (COMET-QE, BLEURT-QE, LLM-as-judge) cho phép bạn đánh giá bản dịch mà không cần bản dịch đối chiếu, điều này đặc biệt quý giá đối với các cặp ngôn ngữ hiếm tài nguyên.

### Bước 3: Những lỗi thường gặp trong môi trường sản xuất

Đường ống hoạt động ở trên sẽ dịch rất trôi chảy 80% trường hợp nhưng sẽ âm thầm gãy đổ ở 20% còn lại. Các phương thức lỗi cụ thể:

- **Ảo giác (Hallucination).** Mô hình tự bịa ra nội dung không hề có trong câu nguồn. Rất phổ biến khi gặp từ vựng chuyên ngành lạ. Triệu chứng: câu dịch rất mượt nhưng khẳng định những dữ kiện sai lệch hoàn toàn. Giải pháp: giải mã có ràng buộc cho các thuật ngữ ngành, con người duyệt lại cho nội dung có quy định pháp lý nghiêm ngặt, giám sát các câu đầu ra dài bất thường so với đầu vào.
- **Sinh nhầm ngôn ngữ (Off-target generation).** Mô hình dịch sang một ngôn ngữ hoàn toàn khác với yêu cầu. NLLB rất hay gặp lỗi này ở các cặp ngôn ngữ hiếm. Giải pháp: kiểm tra kỹ `forced_bos_token_id` và luôn chạy một mô hình nhận diện ngôn ngữ (language-ID) để kiểm tra đầu ra trước khi trả về.
- **Trôi dạt thuật ngữ (Terminology drift).** Cụm từ "Sign up" bị dịch thành "s'inscrire" ở tài liệu 1 nhưng lại thành "créer un compte" ở tài liệu 2. Với giao diện người dùng (UI) và chuỗi ký tự hiển thị, tính nhất quán quan trọng hơn độ phong phú từ ngữ. Giải pháp: giải mã có ràng buộc bằng bảng thuật ngữ (glossary-constrained decoding) hoặc từ điển hậu xử lý.
- **Lệch sắc thái trang trọng (Formality mismatch).** Đại từ "tu" và "vous" trong tiếng Pháp, các cấp độ kính ngữ trong tiếng Nhật. Mô hình sẽ tự chọn dạng nào xuất hiện nhiều hơn trong tập huấn luyện. Đối với dịch vụ khách hàng, điều này rất dễ gây phản cảm. Giải pháp: thêm tiền tố quy định mức độ trang trọng vào prompt nếu mô hình hỗ trợ, hoặc fine-tune một mô hình nhỏ trên ngữ liệu chuẩn phong cách trang trọng.
- **Bùng nổ độ dài ở câu đầu vào ngắn.** Các câu nguồn quá ngắn thường sinh ra bản dịch dài lòng thòng vì hình phạt độ dài mất tác dụng khi câu dưới ~5 token. Giải pháp: giới hạn cứng độ dài tối đa (max-length cap) tỷ lệ thuận với độ dài câu nguồn.

### Bước 4: Tinh chỉnh mô hình cho miền chuyên ngành (Fine-tuning)

Các mô hình huấn luyện sẵn mang tính khái quát chung. Dịch thuật trong các lĩnh vực pháp lý, y tế, hoặc lời thoại game sẽ cải thiện vượt bậc khi được fine-tune trên dữ liệu song ngữ chuyên ngành:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Vài nghìn cặp dữ liệu song hành chất lượng cao luôn đánh bại hàng trăm nghìn mẫu thu thập cào web lẫn lộn tạp chất. Chất lượng dữ liệu huấn luyện là đòn bẩy lớn nhất trong sản xuất.

## Sử dụng trong thực tế (Use It)

Ngăn xếp dịch máy môi trường sản xuất năm 2026:

| Trường hợp sử dụng | Điểm khởi đầu đề xuất |
|-------------------|----------------------|
| Mọi ngôn ngữ sang mọi ngôn ngữ, 200 thứ tiếng | `facebook/nllb-200-distilled-600M` (máy cá nhân) hoặc `nllb-200-3.3B` (production) |
| Trọng tâm tiếng Anh, chất lượng cao, 50 ngôn ngữ | `facebook/mbart-large-50-many-to-many-mmt` |
| Tác vụ ngắn, suy luận chi phí rẻ, Anh - Pháp/Đức/Tây Ban Nha | Các mô hình Helsinki-NLP / Marian |
| Yêu cầu độ trễ cực thấp chạy trên trình duyệt | Marian lượng tử hóa định dạng ONNX (~50 MB) |
| Chất lượng tối đa, sẵn sàng chi trả chi phí API | GPT-4 / Claude / Gemini với prompt dịch thuật chuyên biệt |

Đến năm 2026, các LLM đã vượt qua các mô hình MT chuyên dụng trên nhiều cặp ngôn ngữ, đặc biệt ở các nội dung giàu thành ngữ và ngữ cảnh dài. Đánh đổi nằm ở độ trễ và chi phí tính theo token. Hãy chọn LLM khi độ dài ngữ cảnh, tính nhất quán về văn phong, hoặc khả năng thích ứng miền qua prompt quan trọng hơn thông lượng xử lý hàng loạt.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Đánh giá chất lượng đầu ra dịch máy trước khi đưa vào sản xuất.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Khi nhận được văn bản nguồn và bản dịch ứng viên, bạn đưa ra:

1. Ước tính điểm số tự động. Khoảng điểm BLEU và chrF kỳ vọng. Nêu rõ có bản dịch tham chiếu hay không.
2. Bảng kiểm tra 5 điểm có thể xác minh bằng mắt người: (a) bảo toàn nội dung (không có ảo giác), (b) đúng ngôn ngữ đích, (c) đúng sắc thái văn phong / mức độ trang trọng, (d) nhất quán thuật ngữ với bảng tra cứu (glossary), (e) không bị cắt cụt hoặc bùng nổ độ dài.
3. Một vấn đề đặc thù miền cần điều tra kỹ: Ví dụ với pháp lý: thực thể tên riêng và trích dẫn điều luật. Với y tế: tên biệt dược và liều lượng. Với giao diện UI: các biến định dạng giữ chỗ `{name}`.
4. Cờ đánh giá độ tin cậy: "Phát hành" / "Phát hành kèm rà soát" / "Không phát hành". Gắn liền với mức độ nghiêm trọng của lỗi tìm thấy ở bước 2.

Từ chối phát hành bản dịch nếu chưa chạy bộ kiểm tra nhận diện ngôn ngữ (language-ID) trên đầu ra. Từ chối đánh giá không có tham chiếu trừ khi người dùng chủ động yêu cầu phương pháp chấm điểm không tham chiếu (COMET-QE, BLEURT-QE). Đánh dấu bất kỳ nội dung nào vượt quá 1.000 token là cần xử lý dịch phân đoạn (chunked translation).
```

## Bài tập (Exercises)

1. **Dễ.** Dịch một đoạn văn tiếng Anh gồm 5 câu sang tiếng Pháp rồi dịch ngược lại tiếng Anh bằng `nllb-200-distilled-600M`. Đo lường mức độ tương đồng của bản dịch hai chiều (round-trip) so với bản gốc. Bạn sẽ thấy ngữ nghĩa được bảo toàn nhưng có sự thay đổi về lựa chọn từ vựng.
2. **Trung bình.** Cài đặt bước kiểm tra language-ID cho đầu ra dịch thuật sử dụng `fasttext lid.176` hoặc `langdetect`. Tích hợp trực tiếp vào hàm gọi MT để các kết quả dịch sai ngôn ngữ bị chặn lại trước khi trả về cho người dùng.
3. **Khó.** Tinh chỉnh (fine-tune) `nllb-200-distilled-600M` trên một tập ngữ liệu chuyên ngành gồm 5.000 cặp câu do bạn chọn. Đo điểm BLEU trên tập kiểm thử riêng biệt trước và sau khi tinh chỉnh. Báo cáo loại câu nào được cải thiện và loại câu nào bị giảm sút chất lượng.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| BLEU | Điểm dịch thuật | Độ chính xác n-gram có kèm hình phạt độ dài ngắn. Thang điểm [0, 100]. |
| chrF | F-score ký tự | F-score ở cấp độ ký tự. Nhạy và chuẩn xác hơn với ngôn ngữ giàu hình thái. |
| NMT | Dịch máy nơ-ron | Transformer encoder-decoder huấn luyện trên dữ liệu song ngữ. Chuẩn mực từ 2017 đến nay. |
| NLLB | No Language Left Behind | Họ mô hình dịch thuật hỗ trợ 200 ngôn ngữ của Meta. |
| Giải mã có ràng buộc (Constrained decoding) | Đầu ra có kiểm soát | Ép buộc các token hoặc n-gram cụ thể phải xuất hiện hoặc không được phép xuất hiện ở đầu ra. |
| Ảo giác (Hallucination) | Nội dung tự bịa | Đầu ra mô hình chứa thông tin không có cơ sở trong câu nguồn. |

## Đọc thêm (Further Reading)

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — bài báo gốc về họ mô hình NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — lý do tại sao `sacrebleu` là cách chuẩn mực duy nhất để báo cáo điểm BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — bài báo giới thiệu chỉ số chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — hướng dẫn thực hành fine-tune dịch máy từng bước.

