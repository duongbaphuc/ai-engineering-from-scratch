# Audio Transformers — Kiến trúc Whisper (Audio Transformers — Whisper Architecture)

> Âm thanh là một bức ảnh của tần số theo thời gian. Whisper là một ViT tiếp nhận phổ mel (mel spectrograms) và phản hồi bằng ngôn ngữ văn bản.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 phút

## Vấn đề (The Problem)

Trước khi Whisper xuất hiện (OpenAI, Radford và cộng sự, 2022), các hệ thống nhận dạng tiếng nói tự động đạt đỉnh cao công nghệ (ASR - automatic speech recognition) chủ yếu dựa trên wav2vec 2.0 và HuBERT — các bộ trích xuất đặc trưng tự giám sát (self-supervised feature extractors) kết hợp với một đầu dự đoán được tinh chỉnh (fine-tuned head). Chúng có chất lượng cao nhưng đòi hỏi đường ống dữ liệu tốn kém và rất dễ bị suy giảm hiệu năng khi đổi miền dữ liệu (domain-brittle). Nhận dạng tiếng nói đa ngôn ngữ khi đó cần các mô hình riêng biệt cho từng ngữ tộc.

Whisper đã đặt cược vào ba yếu tố mang tính quyết định:

1. **Huấn luyện trên mọi dữ liệu có thể.** 680.000 giờ âm thanh được gán nhãn yếu (weakly-labeled) thu thập từ internet trên 97 ngôn ngữ khác nhau. Không cần tập ngữ liệu học thuật sạch sẽ. Không cần nhãn âm vị (phoneme labels).
2. **Mô hình duy nhất cho đa tác vụ (Multi-task single model).** Một bộ giải mã duy nhất được huấn luyện đồng thời trên các tác vụ: phiên âm (transcription), dịch thuật (translation), phát hiện giọng nói (voice activity detection - VAD), nhận dạng ngôn ngữ (language identification), và tạo dấu thời gian (timestamping) thông qua các token tác vụ đặc biệt.
3. **Mô hình transformer encoder-decoder tiêu chuẩn.** Bộ mã hóa tiếp nhận phổ log-mel (log-mel spectrograms). Bộ giải mã tạo sinh các token văn bản một cách tự hồi quy. Không dùng vocoder, không CTC (Connectionist Temporal Classification), không mô hình Markov ẩn (HMM).

Kết quả: Whisper large-v3 hoạt động cực kỳ bền bỉ trước nhiều chất giọng (accents), môi trường nhiễu và các ngôn ngữ hoàn toàn không có dữ liệu gán nhãn chuẩn hóa. Nó trở thành tầng đầu vào tiếng nói (speech front-end) mặc định cho mọi trợ lý giọng nói mã nguồn mở và phần lớn các giải pháp thương mại vào năm 2026.

## Khái niệm (The Concept)

![Đường ống Whisper: âm thanh → mel → bộ mã hóa → bộ giải mã → văn bản](../assets/whisper.svg)

### Bước 1 — Lấy mẫu lại + chia cửa sổ (resample + window)

Âm thanh được chuẩn hóa ở tần số lấy mẫu 16 kHz. Cắt ngắn hoặc đệm thêm (pad) cho vừa đúng 30 giây. Tính toán phổ log-mel: 80 dải tần số mel (mel bins), bước nhảy (hop/stride) 10 ms → khoảng 3.000 khung thời gian (frames) × 80 đặc trưng. Đây chính là "bức ảnh đầu vào" mà Whisper quan sát.

### Bước 2 — Khối cuộn khởi đầu (convolutional stem)

Hai lớp Conv1D với kích thước bộ lọc (kernel) 3 và bước nhảy (stride) 2 giúp nén 3.000 khung thời gian xuống còn 1.500. Thao tác này giảm một nửa độ dài chuỗi mà không làm tăng quá nhiều tham số.

### Bước 3 — Bộ mã hóa (encoder)

Một bộ mã hóa transformer gồm 24 lớp (đối với bản large) xử lý trên 1.500 bước thời gian. Sử dụng mã hóa vị trí hình sin (sinusoidal positional encoding), chú ý tự thân (self-attention), và mạng truyền thẳng kích hoạt GELU (GELU FFN). Tạo ra 1.500 × 1.280 trạng thái ẩn.

### Bước 4 — Bộ giải mã (decoder)

Một bộ giải mã transformer gồm 24 lớp. Nó tạo sinh token một cách tự hồi quy từ bộ từ vựng BPE (Byte Pair Encoding) là tập siêu của GPT-2, được bổ sung thêm một số token đặc biệt chuyên biệt cho âm thanh.

### Bước 5 — Các token tác vụ (task tokens)

Câu nhắc (prompt) đầu vào của decoder bắt đầu bằng các token điều khiển hướng dẫn mô hình cần thực hiện thao tác gì:

```text
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

hoặc

```text
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Mô hình đã được huấn luyện dựa trên quy ước này. Bạn điều khiển tác vụ bằng tiền tố prompt. Đây là phiên bản tương đương của kỹ thuật tinh chỉnh theo chỉ dẫn (instruction-tuning) vào năm 2026, nhưng được áp dụng cho tiếng nói.

### Bước 6 — Đầu ra (output)

Tìm kiếm chùm (beam search với độ rộng 5) kết hợp ngưỡng log-xác suất (log-prob threshold). Dấu thời gian (timestamps) được dự đoán sau mỗi 0,02 giây âm thanh khi token `<|notimestamps|>` không xuất hiện.

### Các kích thước của Whisper (Whisper sizes)

| Mô hình | Tham số | Số lớp | d_model | Số đầu | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (decoder 4 lớp) |

Phiên bản Large-v3-turbo (2024) đã cắt giảm decoder từ 32 lớp xuống còn 4 lớp. Tốc độ giải mã nhanh hơn gấp 8 lần với mức suy giảm tỷ lệ lỗi từ (WER - word error rate) chưa tới 1 điểm phần trăm. Khả năng tăng tốc giải mã đột phá này là lý do Whisper-turbo trở thành lựa chọn mặc định cho các tác tử giọng nói thời gian thực (real-time voice agents) vào năm 2026.

### Những điều Whisper không xử lý

- Không phân đoạn người nói (diarization - phân biệt ai đang nói). Cần kết hợp với thư viện như pyannote để thực hiện việc này.
- Không hỗ trợ truyền phát trực tuyến (streaming) theo thời gian thực ở dạng nguyên bản — cửa sổ 30 giây là cố định. Các thư viện bọc hiện đại (`faster-whisper`, `WhisperX`) bổ sung tính năng streaming thông qua VAD và các cửa sổ trượt gối đầu (overlap).
- Không duy trì ngữ cảnh dài vượt quá 30 giây nếu không có cơ chế chia đoạn bên ngoài. Điều này trong thực tế hoạt động rất ổn vì tiếng nói con người hiếm khi cần ngữ cảnh quá xa để phiên âm chính xác.

### Bức tranh toàn cảnh năm 2026 (2026 landscape)

| Tác vụ | Mô hình | Ghi chú |
|------|-------|-------|
| ASR tiếng Anh | Whisper-turbo, Moonshine | Moonshine chạy nhanh hơn 4 lần trên thiết bị biên |
| ASR đa ngôn ngữ | Whisper-large-v3 | 97 ngôn ngữ |
| ASR truyền phát trực tuyến (Streaming) | faster-whisper + VAD | Đạt mục tiêu độ trễ 150 ms |
| Chuyển văn bản thành giọng nói (TTS) | Piper, XTTS-v2, Kokoro | Theo mẫu kiến trúc encoder-decoder, nhưng có cấu trúc tương tự Whisper |
| Âm thanh + ngôn ngữ | AudioLM, SeamlessM4T | Token văn bản + token âm thanh trong cùng một transformer |

```figure
n5-mel-decode
```

## Tự xây dựng (Build It)

Xem `code/main.py`. Chúng ta không huấn luyện lại Whisper — chúng ta sẽ xây dựng đường ống tạo phổ log-mel (log-mel spectrogram pipeline) + bộ định dạng câu nhắc token tác vụ. Đây chính là những phần mà bạn thực sự thao tác trong môi trường sản xuất.

### Bước 1: tổng hợp tín hiệu âm thanh

Tạo một sóng sin 1 giây ở tần số 440 Hz được lấy mẫu ở tần số 16 kHz. Tổng cộng 16.000 mẫu âm thanh.

### Bước 2: tạo phổ log-mel (phiên bản đơn giản hóa)

Tạo phổ mel hoàn chỉnh cần phép biến đổi Fourier nhanh (FFT). Chúng ta thực hiện một phiên bản chia khung (framing) + tính năng lượng mỗi khung đơn giản hóa nhằm minh họa trực quan đường ống mà không cần cài đặt thư viện `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Khung (Frame) = 25 ms, bước nhảy (hop) = 10 ms. Hoàn toàn tương thích với việc chia cửa sổ của Whisper. Năng lượng trên từng khung đại diện cho các dải mel để phục vụ mục đích sư phạm.

### Bước 3: đệm thêm thành 30 giây

Whisper luôn xử lý các đoạn có kích thước 30 giây. Đệm thêm (hoặc cắt bớt) phổ thành đúng 3.000 khung thời gian.

### Bước 4: xây dựng các token của câu nhắc (prompt tokens)

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Đó là toàn bộ giao diện điều khiển tác vụ. Một tiền tố chỉ gồm 4 token.

## Sử dụng trong thực tế (Use It)

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Nhanh hơn, tương thích với OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Khi nào nên chọn Whisper vào năm 2026:**

- Nhận dạng giọng nói tự động đa ngôn ngữ chỉ với một mô hình duy nhất.
- Phiên âm mạnh mẽ, ổn định trên âm thanh thực tế nhiều tạp âm và ngữ điệu đa dạng.
- Nghiên cứu hoặc làm mẫu thử nghiệm (prototype) ASR — điểm khởi đầu nhanh nhất.

**Khi nào nên chọn giải pháp khác:**

- Truyền phát trực tuyến độ trễ cực thấp trên thiết bị biên — Moonshine vượt trội hơn Whisper ở cùng mức chất lượng.
- AI hội thoại thời gian thực đòi hỏi độ trễ <200 ms — cần các hệ thống ASR trực tuyến chuyên dụng.
- Phân đoạn người nói (Speaker diarization) — Whisper không hỗ trợ sẵn; cần tích hợp thêm pyannote.

## Đưa vào triển khai (Ship It)

Xem `outputs/skill-asr-configurator.md`. Kỹ năng này giúp chọn mô hình ASR, các tham số giải mã và đường ống tiền xử lý phù hợp cho một ứng dụng giọng nói mới.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py`. Xác nhận số lượng khung cho tín hiệu 1 giây ở tần số 16 kHz với bước nhảy 10 ms là khoảng 100 khung. Với 30 giây: khoảng 3.000 khung.
2. **Trung bình.** Xây dựng đường ống tạo phổ log-mel hoàn chỉnh bằng `numpy.fft`. Xác minh rằng 80 dải mel khớp với hàm `librosa.feature.melspectrogram(n_mels=80)` trong phạm vi sai số số học.
3. **Khó.** Triển khai cơ chế suy luận dạng truyền phát (streaming inference): chia âm thanh thành các cửa sổ 10 giây với độ gối đầu 2 giây, chạy Whisper trên từng đoạn và hợp nhất các bản phiên âm. Đo lường tỷ lệ lỗi từ (WER) so với lượt chạy đơn trên một đoạn podcast mẫu dài 5 phút.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|------|-----------------|-----------------------|
| Phổ Mel (Mel spectrogram) | "Bức ảnh của âm thanh" | Biểu diễn 2D: các dải tần số trên một trục, các khung thời gian trên trục còn lại; mức năng lượng được tính theo thang đo log trên từng ô. |
| Log-mel | "Những gì Whisper nhìn thấy" | Phổ Mel được đưa qua hàm log; mô phỏng gần đúng cách tai người cảm nhận độ lớn âm thanh. |
| Khung (Frame) | "Một lát cắt thời gian" | Một cửa sổ mẫu có độ dài 25 ms; gối lên nhau với bước nhảy 10 ms. |
| Token tác vụ (Task token) | "Tiền tố câu nhắc cho tiếng nói" | Các token đặc biệt như `<\|transcribe\|>` / `<\|translate\|>` nằm trong prompt của decoder. |
| Phát hiện giọng nói (VAD - Voice activity detection) | "Tìm tiếng người nói" | Bộ lọc loại bỏ các khoảng lặng trước khi đưa vào ASR; giúp giảm đáng kể chi phí tính toán. |
| CTC | "Connectionist Temporal Classification" | Hàm mất mát ASR kinh điển cho phép huấn luyện không cần căn chỉnh nhãn trước; Whisper KHÔNG sử dụng kỹ thuật này. |
| Whisper-turbo | "Decoder nhỏ, encoder nguyên vẹn" | Bộ mã hóa của large-v3 kết hợp với bộ giải mã chỉ gồm 4 lớp; tốc độ giải mã nhanh gấp 8 lần. |
| Faster-whisper | "Thư viện triển khai sản xuất" | Bản tái triển khai bằng CTranslate2; lượng tử hóa int8; nhanh hơn 4 lần so với bản tham chiếu của OpenAI. |

## Tài liệu đọc thêm (Further Reading)

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — bài báo gốc về Whisper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — mã nguồn tham chiếu + trọng số mô hình. Đọc `whisper/model.py` để thấy toàn bộ khối Conv1D stem + encoder + decoder chỉ trong ~400 dòng mã.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logic tìm kiếm chùm + token tác vụ được mô tả ở Bước 5–6 nằm tại đây; khoảng 500 dòng mã rất dễ hiểu.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — mô hình tiền thân; vẫn cung cấp đặc trưng SOTA trong một số kịch bản nhất định.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — thư viện bọc cho môi trường sản xuất, nhanh gấp 4 lần mã nguồn tham chiếu.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — mô hình ASR năm 2024 thân thiện với thiết bị biên, cấu trúc tương tự Whisper nhưng nhỏ gọn hơn.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — hướng dẫn tinh chỉnh chuẩn mực bao gồm bộ tiền xử lý phổ mel và xử lý nhãn thời gian của token.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — mã nguồn triển khai đầy đủ (encoder, decoder, cross-attention, generation) phản ánh chính xác sơ đồ kiến trúc của bài học.
