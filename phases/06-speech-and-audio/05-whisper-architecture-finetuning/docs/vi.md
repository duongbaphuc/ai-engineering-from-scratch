# Whisper — Kiến trúc & Tinh chỉnh (Fine-Tuning)

> Whisper là mô hình Transformer bộ mã hóa - bộ giải mã (encoder-decoder) với cửa sổ cố định 30 giây, được huấn luyện trên 680.000 giờ dữ liệu cặp âm thanh - văn bản đa ngôn ngữ có giám sát yếu (weakly-supervised). Một kiến trúc duy nhất, thực hiện đa tác vụ, chống chịu nhiễu mạnh mẽ trên 99 ngôn ngữ. Chuẩn mực ASR tham chiếu của năm 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6 · 04 (ASR), Giai đoạn 5 · 10 (Attention), Giai đoạn 7 · 05 (Full Transformer)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Được OpenAI phát hành vào tháng 9 năm 2022, Whisper là mô hình ASR đầu tiên được thương phẩm hóa như một tiện ích đại trà: chỉ cần đưa âm thanh vào, bạn sẽ nhận về văn bản chuẩn xác, hỗ trợ 99 ngôn ngữ, chống chịu tạp âm tốt và chạy mượt mà ngay trên máy tính cá nhân. Đến năm 2024, OpenAI tiếp tục cho ra mắt các biến thể Large-v3 và Turbo; và tính đến năm 2026, Whisper đã trở thành nền tảng cơ sở mặc định cho mọi tác vụ từ bóc băng podcast, trợ lý ảo giọng nói đến tạo phụ đề tự động trên YouTube.

Thế nhưng, bạn không thể mãi coi Whisper như một chiếc hộp đen. Hiện tượng trôi dạt miền dữ liệu (domain shift) có thể bẻ gãy mô hình — thuật ngữ chuyên ngành lạ, ngữ điệu địa phương của người nói, danh từ riêng, các đoạn âm thanh ngắn hay những khoảng lặng kéo dài. Bạn cần thấu hiểu sâu sắc:

1. Bản chất kiến trúc thực sự bên trong mô hình.
2. Cách truyền dữ liệu âm thanh phân khối (chunked), streaming hoặc âm thanh thời lượng dài một cách chuẩn xác.
3. Khi nào cần tinh chỉnh (fine-tune) và quy trình thực hiện tối ưu.

## Khái niệm (The Concept)

![Bộ mã hóa - bộ giải mã của Whisper, các tác vụ, suy luận theo khối và tinh chỉnh](../assets/whisper.svg)

**Kiến trúc.** Mô hình Transformer encoder-decoder tiêu chuẩn.

- Đầu vào: Phổ log-mel spectrogram độ dài 30 giây, 80 dải lọc mel, bước nhảy hop 10 ms → 3.000 khung thời gian. Các đoạn âm thanh ngắn hơn sẽ được đệm số 0 (zero-padded), các đoạn dài hơn phải được cắt thành từng khối (chunked).
- Bộ mã hóa (Encoder): Giảm kích thước mẫu qua tích chập (conv-downsample với stride 2) + `N` khối Transformer. Bản Large-v3 gồm: 32 tầng, kích thước ẩn 1280 chiều, 20 đầu chú ý (attention heads).
- Bộ giải mã (Decoder): `N` khối Transformer với cơ chế tự chú ý nhân quả (causal self-attention) + chú ý chéo (cross-attention) kết nối với đầu ra của bộ mã hóa. Kích thước tương đương bộ mã hóa.
- Đầu ra: Các token BPE (Byte-Pair Encoding) trên không gian từ vựng gồm 51.865 token.

Bản Large-v3 sở hữu 1,55 tỷ tham số. Phiên bản Turbo rút gọn chỉ còn 4 tầng giải mã (thay vì 32), giúp giảm độ trễ tới 8 lần trong khi chỉ làm tăng chỉ số WER dưới 1%.

**Định dạng Prompt điều khiển.** Whisper là một mô hình đa nhiệm được dẫn dắt bởi các token đặc biệt nằm ngay trong prompt khởi đầu của bộ giải mã:

```text
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>`: Thẻ ngôn ngữ (language tag); chỉ định ngôn ngữ được nói trong tệp âm thanh. Nếu bỏ trống, Whisper sẽ chạy tác vụ tự động nhận diện ngôn ngữ trước.
- `<|transcribe|>` hoặc `<|translate|>`: Thẻ tác vụ; chuyển giọng nói thành văn bản nguyên ngữ (transcribe), hoặc dịch sang văn bản tiếng Anh (translate).
- `<|notimestamps|>`: Bỏ qua việc xuất mốc thời gian chi tiết theo từ (giúp tăng tốc độ xử lý).

Chính cấu trúc prompt linh hoạt này cho phép một mô hình duy nhất đảm nhiệm nhiều tác vụ khác nhau. Chỉ cần đổi `<|en|>` thành `<|fr|>`, mô hình sẽ tự động bóc băng tiếng Pháp.

**Cửa sổ xử lý 30 giây (30-second window).** Toàn bộ cơ chế tính toán được cố định cứng ở mốc 30 giây. Các đoạn âm thanh dài hơn bắt buộc phải phân đoạn; các đoạn ngắn hơn phải được đệm bù. Whisper nguyên bản không hỗ trợ streaming tự nhiên — đây chính là lý do vì sao các thư viện như WhisperX, Whisper-Streaming và faster-whisper ra đời.

**Chuẩn hóa phổ Log-mel.** Áp dụng công thức `(log_mel - mean) / std`, trong đó các giá trị thống kê trung bình và phương sai được trích xuất từ tập dữ liệu huấn luyện riêng của OpenAI. Bạn *bắt buộc* phải sử dụng hàm tiền xử lý chuẩn của Whisper (`whisper.audio.log_mel_spectrogram`), tuyệt đối không thay thế bằng `librosa.feature.melspectrogram`.

### Các biến thể mô hình năm 2026

| Biến thể | Số tham số | Độ trễ (A100) | WER (LibriSpeech-clean) |
|----------|------------|---------------|------------------------|
| Tiny | 39M | 1× thời gian thực | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | chế độ streaming | 2.0% |

### Tinh chỉnh mô hình (Fine-Tuning)

Quy trình chuẩn hóa trong năm 2026:

1. Thu thập từ 10–100 giờ âm thanh thuộc miền ứng dụng chuyên biệt kèm theo văn bản bóc băng chuẩn.
2. Sử dụng `transformers.Seq2SeqTrainer` kết hợp với hàm gọi lại (callback) `generate_with_loss`.
3. Tinh chỉnh hiệu quả tham số (PEFT): Áp dụng LoRA lên các ma trận chiếu `q_proj`, `k_proj`, `v_proj` của các tầng attention giúp giảm tiêu tốn bộ nhớ GPU tới 4 lần mà chỉ làm sai lệch WER dưới 0,3%.
4. Đóng băng (freeze) bộ mã hóa nếu bạn chỉ có dưới 10 giờ dữ liệu. Chỉ tập trung tinh chỉnh bộ giải mã.
5. Luôn giữ nguyên bộ tokenizer và định dạng prompt của Whisper; không bao giờ hoán đổi tokenizer.

Kết quả thực tế từ cộng đồng: Tinh chỉnh bản Medium trên 20 giờ ghi âm y khoa giảm chỉ số WER từ 12% xuống còn 4,5% trên các thuật ngữ y học phức tạp. Tinh chỉnh bản Turbo trên 4 giờ tiếng Iceland giảm WER từ 18% xuống 6%.

````figure
sp-asr-attention
````

## Bắt tay vào xây dựng (Build It)

### Bước 1: Chạy mô hình Whisper nguyên bản

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # ngăn chặn lỗi lặp từ mất kiểm soát
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Các tham số mặc định quan trọng cần luôn ghi đè: `temperature=0.0` (mặc định lấy mẫu chuỗi nhiệt độ 0.0 → 0.2 → 0.4 …), `condition_on_previous_text=False` (loại bỏ triệt để hiện tượng ảo giác dây chuyền), và `no_speech_threshold=0.6` (ngưỡng phát hiện khoảng lặng).

### Bước 2: Xử lý âm thanh dài dạng phân đoạn (Chunked Long-form)

```python
# whisperx là chuẩn mực năm 2026 cho âm thanh dài kèm mốc thời gian chi tiết từng từ
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX tích hợp thêm: (1) Cổng lọc khoảng lặng Silero VAD, (2) Căn chỉnh mốc thời gian từng từ qua wav2vec 2.0, (3) Phân tách danh tính người nói (diarization) thông qua `pyannote.audio`. Đây là công cụ chủ lực cho bóc băng quy mô sản xuất năm 2026.

### Bước 3: Tinh chỉnh mô hình với LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M tham số cần huấn luyện / 809M tổng số
```

Sau đó áp dụng vòng lặp Trainer tiêu chuẩn. Lưu checkpoint sau mỗi 1000 bước. Đánh giá chỉ số WER trên tập validation riêng biệt.

### Bước 4: Khảo sát trọng số chú ý của từng tầng

```python
# Trích xuất trọng số cross-attention trong lúc giải mã để quan sát cách bộ giải mã chú ý
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Trực quan hóa bằng biểu đồ nhiệt (heatmap) — bạn sẽ thấy một đường căn chỉnh chéo thẳng tắp khi các bước giải mã quét tuần tự qua các khung âm thanh của bộ mã hóa. Đường chéo đó chính là cơ chế xác định mốc thời gian của Whisper.

## Sử dụng trong thực tế (Use It)

Bộ công cụ tiêu chuẩn năm 2026:

| Tình huống thực tế | Giải pháp tối ưu |
|--------------------|------------------|
| Tiếng Anh tổng quát, xử lý ngoại tuyến (offline) | Large-v3-turbo thông qua `whisperx` |
| Ứng dụng di động / thiết bị biên (edge) | Whisper-Tiny lượng tử hóa (int8) hoặc Moonshine |
| Bóc băng đa ngôn ngữ thời lượng dài | Large-v3 qua `whisperx` + nhận diện người nói (diarization) |
| Ngôn ngữ tài nguyên thấp (ít dữ liệu) | Tinh chỉnh bản Medium hoặc Turbo kết hợp LoRA |
| Truyền phát thời gian thực (độ trễ ~2 giây) | Whisper-Streaming hoặc Parakeet-TDT |
| Cần mốc thời gian chính xác từng từ | WhisperX (căn chỉnh cưỡng bức forced alignment qua wav2vec 2.0) |

`faster-whisper` (dựa trên backend CTranslate2) là runtime suy luận tốc độ cao nhất trên cả CPU lẫn GPU trong năm 2026 — nhanh gấp 4 lần so với mã nguồn gốc nhưng cho kết quả hoàn toàn đồng nhất.

## Những cạm bẫy kỹ thuật thường gặp trong năm 2026

- **Hiện tượng sinh chữ ảo giác trên khoảng lặng.** Do Whisper được huấn luyện trên phụ đề mạng, nó có thể tự sinh ra các câu như "Thanks for watching!", "Đăng ký kênh nhé!", hoặc lời bài hát ngẫu nhiên. Luôn luôn dùng cổng lọc VAD trước khi gọi mô hình.
- **Ảo giác dây chuyền do `condition_on_previous_text`.** Khi một đoạn bị ảo giác, nó sẽ làm ô nhiễm ngữ cảnh của các cửa sổ thời gian tiếp theo. Hãy thiết lập `condition_on_previous_text=False` trừ khi bạn thực sự cần độ mượt câu giữa các khối.
- **Đệm số 0 trên các đoạn âm thanh quá ngắn.** Một đoạn âm thanh 2 giây bị đệm thêm 28 giây số 0 rất dễ gây ra ảo giác ở khoảng lặng phía sau. Hãy đặt `pad=False` hoặc lọc qua VAD.
- **Sai lệch tham số thống kê phổ Mel.** Dùng đặc trưng mel từ librosa thay vì Whisper sẽ khiến mô hình sinh ra chuỗi ký tự hỗn loạn. Bắt buộc dùng `whisper.audio.log_mel_spectrogram`.

## Xuất xưởng sản phẩm (Ship It)

Lưu lại dưới dạng `outputs/skill-whisper-tuner.md`. Kỹ năng này hỗ trợ thiết kế pipeline tinh chỉnh hoặc suy luận với Whisper cho từng miền bài toán cụ thể.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy file `code/main.py`. Chương trình sẽ tokenize một prompt theo chuẩn Whisper, tính toán ngân sách kích thước giải mã và in ra lịch trình phân khối (chunk schedule) cho một tệp âm thanh dài 10 phút.
2. **Trung bình.** Cài đặt thư viện `faster-whisper`, thực hiện bóc băng một đoạn podcast dài 10 phút, và so sánh WER với văn bản phiên âm thủ công của con người. Thử nghiệm chế độ `language="auto"` so với việc ấn định cứng `language="en"`.
3. **Khó.** Sử dụng thư viện `datasets` của Hugging Face, chọn một ngôn ngữ có độ chính xác chưa cao trên Whisper (ví dụ: tiếng Urdu), thực hiện tinh chỉnh bản Medium với LoRA trong 2 epoch trên 2 giờ dữ liệu và báo cáo độ suy giảm của chỉ số WER.

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Khái niệm thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Cửa sổ 30 giây (30-sec window) | Giới hạn của Whisper | Trần đầu vào cố định; bắt buộc cắt khối âm thanh dài hơn. |
| SOT | Khởi đầu văn bản | Token `<\|startoftranscript\|>` kích hoạt prompt của bộ giải mã. |
| Token mốc thời gian | Căn chỉnh thời gian | Mỗi khoảng lệch 0,02 giây tương ứng với một token đặc biệt trong bộ từ vựng 51k. |
| Turbo | Biến thể tốc độ cao | Chỉ gồm 4 tầng giải mã, nhanh gấp 8 lần, suy giảm chỉ số WER dưới 1%. |
| WhisperX | Bộ bọc xử lý âm thanh dài | Kết hợp VAD + Whisper + căn chỉnh wav2vec + phân tách danh tính người nói. |
| Tinh chỉnh LoRA | Tinh chỉnh hiệu quả | Bổ sung các ma trận thích ứng hạng thấp vào attention; chỉ cần huấn luyện ~0,3% tham số. |
| Ảo giác (Hallucination) | Lỗi âm thầm | Whisper tự sinh ra văn bản hoàn chỉnh trôi chảy từ khoảng lặng hoặc tiếng ồn tạp âm. |

## Tài liệu đọc thêm (Further Reading)

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — Bài báo khoa học gốc công bố kiến trúc và phương pháp huấn luyện Whisper.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — Kiến trúc bộ giải mã 4 tầng, tăng tốc 8 lần.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — Giải pháp bóc băng thời lượng dài, căn chỉnh theo từ và phân tách người nói.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — Thư viện tối ưu trên nền CTranslate2, tốc độ nhanh gấp 4 lần.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — Hướng dẫn chuẩn mực về tinh chỉnh toàn phần và LoRA cho Whisper.
