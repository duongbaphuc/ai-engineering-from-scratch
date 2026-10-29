# Tổng hợp tiếng nói (Text-to-Speech - TTS) — Từ Tacotron đến F5 và Kokoro

> Nhận dạng tiếng nói tự động (ASR) chuyển đổi tiếng nói thành văn bản; TTS chuyển đổi văn bản thành tiếng nói. Ngăn xếp công nghệ năm 2026 gồm ba phần: văn bản → token, token → mel, mel → dạng sóng (waveform). Mỗi phần đều có một mô hình mặc định chạy vừa vặn trên máy tính xách tay.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 phút

## Vấn đề đặt ra

Bạn có một chuỗi văn bản: "Please remind me to water the plants at 6 pm." Bạn cần một đoạn âm thanh dài 3 giây nghe tự nhiên, có ngữ điệu (prosody) chuẩn xác (ngắt nghỉ, trọng âm), phát âm từ "plants" đúng nguyên âm, và chạy dưới 300 ms trên CPU cho một trợ lý ảo giọng nói (voice assistant) trực tiếp. Bạn cũng cần đổi giọng nói linh hoạt, xử lý dữ liệu nhập trộn mã ngôn ngữ (code-switched input, ví dụ: "remind me at 6 pm, daijoubu?"), và không phát âm sai các tên riêng.

Đường ống (pipeline) TTS hiện đại có cấu trúc như sau:

1. **Giao diện đầu vào văn bản (Text frontend).** Chuẩn hóa văn bản (ngày tháng, số đếm, email), chuyển đổi thành âm vị (phonemes) hoặc các token từ con (subword tokens), dự đoán các đặc trưng ngữ điệu.
2. **Mô hình âm học (Acoustic model).** Văn bản → phổ mel (mel spectrogram). Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Bộ tổng hợp âm/Bộ mã hóa giọng nói (Vocoder).** Mel → dạng sóng (waveform). WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), các bộ mã hóa nén âm thanh nơ-ron (neural codec vocoders) từ năm 2024 trở đi.

Vào năm 2026, ranh giới phân tách giữa mô hình âm học và vocoder dần mờ nhạt với các mô hình khuếch tán (diffusion) và so khớp dòng (flow matching) đầu-cuối (end-to-end). Tuy nhiên, mô hình tư duy ba phần này vẫn rất hữu ích khi gỡ lỗi (debugging).

## Khái niệm cốt lõi

![So sánh trực quan Tacotron, FastSpeech, VITS, F5/Kokoro](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: nhúng ký tự (char-embedding) → bộ mã hóa BiLSTM → cơ chế chú ý nhạy theo vị trí (location-sensitive attention) → bộ giải mã LSTM tự hồi quy (autoregressive LSTM decoder) phát ra các khung mel. Chậm (do tự hồi quy - AR), kém ổn định trên các đoạn văn bản dài. Vẫn được trích dẫn làm đường cơ sở (baseline).

**FastSpeech 2 (2020).** Phi tự hồi quy (Non-autoregressive). Bộ dự đoán độ dài (duration predictor) xác định số khung mel mà mỗi âm vị nhận được. Tạo âm thanh trong 1 lượt (1-pass), nhanh hơn 10× so với Tacotron. Mất đi một chút tính tự nhiên (do gióng hàng đơn điệu - monotonic alignment) nhưng được triển khai ở khắp mọi nơi.

**VITS (2021).** Huấn luyện đồng thời bộ mã hóa + bộ dự đoán độ dài dựa trên dòng (flow-based duration) + vocoder HiFi-GAN theo phương pháp đầu-cuối bằng suy diễn biến phân (variational inference). Chất lượng cao, gói gọn trong một mô hình duy nhất. Là giải pháp TTS mã nguồn mở thống trị giai đoạn 2022–2024. Các biến thể: YourTTS (zero-shot đa người nói), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Transformer khuếch tán dựa trên khớp dòng (flow matching). Ngữ điệu tự nhiên, nhân bản giọng nói chuẩn xác mà không cần tinh chỉnh trước (zero-shot voice cloning) chỉ với 5 giây âm thanh tham chiếu. Dẫn đầu bảng xếp hạng TTS mã nguồn mở năm 2026. Kích thước 335M tham số.

**Kokoro (2024).** Gọn nhẹ (82M tham số), chạy tốt trên CPU, là mô hình TTS tiếng Anh tốt nhất phân khúc cho nhu cầu xử lý thời gian thực. Giới hạn từ vựng cho tiếng Anh, giấy phép Apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Đỉnh cao công nghệ thương mại. Các thẻ cảm xúc của ElevenLabs v2.5 (như "[whispered]", "[laughing]") và các giọng nhân vật đang thống trị việc sản xuất sách nói năm 2026.

### Quá trình phát triển của Vocoder

| Giai đoạn | Vocoder | Độ trễ | Chất lượng |
|---|---|---|---|
| 2016 | WaveNet | Chỉ ngoại tuyến (offline) | Đỉnh cao (SOTA) khi ra mắt |
| 2018 | WaveRNN | ~Thời gian thực | Tốt |
| 2020 | HiFi-GAN | Nhanh hơn 100× thời gian thực | Gần giọng người thật |
| 2022 | BigVGAN | Nhanh hơn 50× thời gian thực | Khái quát hóa tốt trên nhiều người nói/ngôn ngữ |
| 2024 | SNAC, DAC (neural codecs) | Tích hợp cùng các mô hình tự hồi quy | Token rời rạc, tiết kiệm băng thông bit |

Đến năm 2026, hầu hết các mô hình "TTS" đều xử lý đầu-cuối từ văn bản đến dạng sóng; phổ mel lúc này đóng vai trò như một biểu diễn nội bộ.

### Đánh giá chất lượng

- **MOS (Mean Opinion Score - Điểm ý kiến trung bình).** Thang điểm 1–5, đánh giá qua cộng đồng (crowd-sourced). Vẫn là tiêu chuẩn vàng; nhưng rất tốn thời gian.
- **CMOS (Comparative MOS - MOS so sánh).** Sở thích chọn A hay B. Cho khoảng tin cậy chặt chẽ hơn trên mỗi mẫu gán nhãn.
- **UTMOS, DNSMOS.** Các mô hình mạng nơ-ron dự đoán MOS không cần tham chiếu (reference-free). Được sử dụng rộng rãi trên các bảng xếp hạng.
- **CER (Character Error Rate - Tỷ lệ lỗi ký tự) qua ASR.** Đưa đầu ra của TTS qua Whisper, tính toán CER so với văn bản gốc đầu vào. Đại diện cho độ rõ tiếng (intelligibility).
- **SECS (Speaker Embedding Cosine Similarity - Độ tương đồng cosine nhúng người nói).** Đo lường chất lượng nhân bản giọng nói.

Các chỉ số năm 2026 trên tập dữ liệu LibriTTS test-clean:

| Mô hình | UTMOS | CER (qua Whisper) | Kích thước |
|---|---|---|---|
| Dữ liệu gốc (Ground truth) | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

````figure
sp-tts-stack
````

## Tự xây dựng (Build It)

### Bước 1: Âm vị hóa đầu vào (Phonemize)

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Âm vị (phonemes) là cầu nối vạn năng. Tránh đưa trực tiếp văn bản thô vào bất kỳ mô hình nào có chất lượng dưới cấp độ VITS.

### Bước 2: Chạy Kokoro (Mặc định cho CPU năm 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = Tiếng Anh Mỹ (American English)
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: tensor float32, sr=24000
```

Chạy hoàn toàn ngoại tuyến, gói gọn trong một tệp, 82M tham số.

### Bước 3: Chạy F5-TTS với tính năng nhân bản giọng nói

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Truyền vào một đoạn âm thanh tham chiếu 5 giây kèm bản ghi lời thoại của nó; F5 sẽ nhân bản cả ngữ điệu lẫn âm sắc (timbre).

### Bước 4: Tự xây dựng Vocoder HiFi-GAN từ đầu

Mô hình quá lớn để đặt trọn vẹn trong một tập lệnh hướng dẫn, nhưng cấu trúc khung sẽ như sau:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 khối lấy mẫu nâng (upsample blocks), tổng cộng tăng 256x từ tốc độ khung mel lên tốc độ mẫu âm thanh
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> dạng sóng (waveform)
```

Quá trình huấn luyện: đối nghịch (bộ phân biệt - discriminator trên các cửa sổ ngắn) + hàm mất mát tái tạo phổ mel (mel-spectrogram reconstruction loss) + hàm mất mát so khớp đặc trưng (feature-matching loss). Hiện đã được tiêu chuẩn hóa — bạn nên dùng các checkpoint tiền huấn luyện từ kho lưu trữ `hifi-gan` hoặc nvidia-NeMo.

### Bước 5: Đường ống hoàn chỉnh (Mã giả)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Ứng dụng thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn |
|---|---|
| Trợ lý giọng nói tiếng Anh thời gian thực | Kokoro (CPU) hoặc XTTS v2 (GPU) |
| Nhân bản giọng nói từ mẫu tham chiếu 5 giây | F5-TTS |
| Giọng nhân vật thương mại chất lượng cao | ElevenLabs v2.5 |
| Đọc sách nói (Audiobook) | ElevenLabs v2.5 hoặc XTTS v2 + tinh chỉnh (fine-tune) |
| Ngôn ngữ tài nguyên thấp | Huấn luyện VITS trên 5–20 giờ dữ liệu ngôn ngữ mục tiêu |
| Giọng truyền cảm / Thẻ cảm xúc | ElevenLabs v2.5 hoặc StyleTTS 2 tinh chỉnh |

Dẫn đầu mã nguồn mở vào năm 2026: **F5-TTS về chất lượng, Kokoro về hiệu năng**. Trừ khi nghiên cứu lịch sử phát triển, bạn không nên dùng lại Tacotron.

## Cạm bẫy thường gặp

- **Thiếu bộ chuẩn hóa văn bản (text normalizer).** "Dr. Smith" nên đọc là "Doctor" hay "Drive"? "2026" đọc là "twenty twenty six" hay "two zero two six"? Hãy chuẩn hóa TRƯỚC KHI đưa vào bộ âm vị hóa.
- **Từ ngoài từ vựng (OOV - Out-Of-Vocabulary) đối với tên riêng.** "Ghumare" → "ghyu-mair"? Hãy trang bị một mô hình dự phòng chuyển từ chữ viết sang âm vị (grapheme-to-phoneme fallback) cho các token lạ.
- **Hiện tượng méo biên độ (Clipping).** Đầu ra của vocoder hiếm khi bị xén biên độ, nhưng sự sai lệch tỷ lệ thang mel khi suy diễn có thể vượt ngưỡng ±1.0. Luôn đảm bảo gọi `np.clip(wav, -1, 1)`.
- **Lệch tần số lấy mẫu (Sample-rate mismatch).** Kokoro xuất ra âm thanh 24 kHz; nếu đường ống phía sau của bạn yêu cầu 16 kHz → phải tái lấy mẫu (resample), nếu không âm thanh sẽ bị răng cưa (aliasing).

## Xuất xưởng sản phẩm (Ship It)

Lưu thành `outputs/skill-tts-designer.md`. Thiết kế một đường ống TTS đáp ứng đúng mục tiêu giọng nói, độ trễ và ngôn ngữ được chỉ định.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Chương trình sẽ xây dựng từ điển âm vị từ một tập từ vựng mẫu, ước tính thời lượng cho từng âm vị và in ra lịch trình phân bổ khung "mel" mô phỏng.
2. **Trung bình.** Cài đặt Kokoro, tổng hợp cùng một câu văn với hai giọng `af_bella` và `am_adam`. So sánh thời lượng âm thanh và chất lượng chủ quan.
3. **Khó.** Tự ghi âm một đoạn âm thanh tham chiếu 5 giây của chính bạn. Sử dụng F5-TTS để nhân bản giọng nói đó. Đo lường và báo cáo chỉ số SECS giữa tệp tham chiếu và âm thanh tạo ra.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Âm vị (Phoneme) | Đơn vị âm thanh | Lớp âm thanh trừu tượng; có 39 âm vị trong tiếng Anh (bảng mã ARPABet). |
| Bộ dự đoán độ dài (Duration predictor) | Thời lượng mỗi âm vị | Đầu ra của mô hình phi tự hồi quy; số khung nguyên ứng với mỗi âm vị. |
| Vocoder | Mel → dạng sóng | Mạng nơ-ron ánh xạ phổ mel thành các mẫu sóng âm thô. |
| HiFi-GAN | Vocoder tiêu chuẩn | Dựa trên GAN; chiếm lĩnh vị trí chủ đạo giai đoạn 2020–2024. |
| MOS | Chất lượng chủ quan | Điểm ý kiến trung bình từ 1–5 do người đánh giá chấm. |
| SECS | Thước đo nhân bản giọng | Độ tương đồng cosine giữa vector nhúng người nói mục tiêu và âm thanh đầu ra. |
| F5-TTS | Đỉnh cao mã nguồn mở 2024 | Mô hình khuếch tán khớp dòng; nhân bản zero-shot. |
| Kokoro | Dẫn đầu trên CPU cho tiếng Anh | Mô hình 82M tham số, giấy phép Apache 2.0. |

## Tài liệu đọc thêm

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — Mô hình đường cơ sở seq2seq.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — Mô hình đầu-cuối dựa trên dòng biến phân.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — Đỉnh cao công nghệ mã nguồn mở hiện nay.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — Bộ vocoder vẫn được triển khai rộng rãi vào năm 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — TTS tiếng Anh tối ưu cho CPU ra mắt năm 2024.
