# Nhân bản giọng nói & Chuyển đổi giọng nói (Voice Cloning & Voice Conversion)

> Nhân bản giọng nói (voice cloning) đọc văn bản của bạn bằng giọng nói của người khác. Chuyển đổi giọng nói (voice conversion) tái tạo giọng nói của bạn thành giọng của người khác nhưng vẫn giữ nguyên vẹn nội dung bạn đã nói. Cả hai kỹ thuật đều dựa trên cùng một phép phân rã: tách biệt danh tính người nói (speaker identity) ra khỏi nội dung (content).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 phút

## Vấn đề đặt ra

Vào năm 2026, chỉ cần một đoạn âm thanh dài 5 giây là đủ để tạo ra một bản sao giọng nói chất lượng cao của bất kỳ ai bằng GPU phổ thông. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox đều hỗ trợ nhân bản không cần mẫu (zero-shot) hoặc với rất ít mẫu (few-shot). Công nghệ này vừa là một giải pháp hữu ích (TTS hỗ trợ người khuyết tật, lồng tiếng phim, giọng nói trợ năng) vừa là một công cụ tiềm ẩn nguy hiểm (các cuộc gọi lừa đảo, deepfake chính trị, đánh cắp quyền sở hữu trí tuệ).

Hai tác vụ có mối liên hệ mật thiết:

- **Nhân bản giọng nói (phía TTS):** văn bản + 5 giây giọng nói tham chiếu → âm thanh phát ra đúng bằng giọng nói đó.
- **Chuyển đổi giọng nói (phía xử lý tiếng nói):** âm thanh nguồn (người A nói nội dung X) + giọng tham chiếu của người B → âm thanh của người B nói nội dung X.

Cả hai đều phân tích một dạng sóng thành bộ ba (nội dung, người nói, ngữ điệu - content, speaker, prosody) và tái kết hợp nội dung từ một nguồn với người nói từ một nguồn khác.

Ràng buộc pháp lý then chốt bạn phải tuân thủ khi triển khai sản phẩm vào năm 2026: **đóng dấu bản quyền âm thanh (watermarking) và cổng kiểm duyệt đồng thuận (consent gates) là yêu cầu bắt buộc theo luật định tại EU (Đạo luật Trí tuệ nhân tạo - AI Act, có hiệu lực từ tháng 8 năm 2026) và tại California (Dự luật AB 2905, có hiệu lực từ năm 2025)**. Đường ống của bạn bắt buộc phải phát ra một dấu thủy vân không nghe thấy được và từ chối các yêu cầu nhân bản khi chưa có sự đồng thuận.

## Khái niệm cốt lõi

![Nhân bản giọng nói so với chuyển đổi giọng nói: phân rã, tráo đổi người nói, tái kết hợp](../assets/voice-cloning.svg)

**Nhân bản không cần mẫu (Zero-shot cloning).** Truyền một đoạn âm thanh 5 giây vào mô hình đã được huấn luyện trên hàng nghìn người nói khác nhau. Bộ mã hóa người nói (speaker encoder) ánh xạ đoạn âm thanh thành một vector nhúng người nói (speaker embedding); bộ giải mã TTS sẽ điều kiện hóa dựa trên vector nhúng đó cùng với văn bản đầu vào.

Được sử dụng bởi: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Tinh chỉnh với ít mẫu (Few-shot fine-tuning).** Ghi âm từ 5 đến 30 phút giọng nói mục tiêu. Tinh chỉnh mô hình nền tảng qua LoRA (Low-Rank Adaptation) trong khoảng một giờ. Chất lượng nhảy vọt từ mức "tạm ổn" sang "không thể phân biệt". Cả Coqui và ElevenLabs đều hỗ trợ mô hình này; cộng đồng mã nguồn mở áp dụng rộng rãi với F5-TTS.

**Chuyển đổi giọng nói (Voice Conversion - VC).** Gồm hai họ phương pháp chính:

- **Nhận dạng - tổng hợp (Recognition-synthesis).** Chạy một mô hình dạng nhận dạng tiếng nói (ASR) để trích xuất biểu diễn nội dung (ví dụ: phân phối xác suất hậu nghiệm ngữ âm - phonetic posteriorgrams - PPGs, hoặc xác suất âm vị mềm), sau đó tổng hợp lại với vector nhúng của người nói mục tiêu. Rất bền vững với ngôn ngữ và ngữ điệu vùng miền (accent). Được sử dụng bởi KNN-VC (2023), Diff-HierVC (2023).
- **Gỡ rối biểu diễn (Disentanglement).** Huấn luyện một mạng tự mã hóa (autoencoder) giúp tách rời nội dung, người nói và ngữ điệu trong không gian tiềm ẩn (latent space) tại điểm thắt cổ chai (bottleneck). Tráo đổi vector nhúng người nói trong quá trình suy diễn. Cho chất lượng thấp hơn nhưng tốc độ xử lý nhanh hơn. Được sử dụng bởi AutoVC (2019), các biến thể VITS-VC.

**Nhân bản dựa trên bộ mã hóa nén âm thanh nơ-ron (Neural codec-based cloning) (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — xem âm thanh như các token rời rạc từ SoundStream / EnCodec, huấn luyện một mô hình tự hồi quy lớn hoặc mô hình khớp dòng (flow-matching) trên các token codec. Đạt chất lượng tương đương ElevenLabs trên các đoạn gợi ý (prompts) ngắn.

### Vấn đề đạo đức: Phần cốt lõi, không phải phụ kiện gắn thêm

**Đóng dấu bản quyền âm thanh (Watermarking).** PerTh (Perth) và SilentCipher (2024) nhúng một định danh ID khoảng 16–32 bit vào âm thanh mà tai người không thể nhận biết. Dấu bản quyền này vẫn tồn tại sau khi mã hóa lại (re-encoding), phát luồng (streaming) và các thao tác chỉnh sửa thông thường. Đã có mã nguồn mở sẵn sàng cho môi trường sản xuất.

**Cổng kiểm duyệt đồng thuận (Consent gates).** Bắt buộc phải gắn kèm mỗi đoạn âm thanh nhân bản với một bản ghi đồng thuận có thể xác minh được. Ví dụ: "Tôi, Rohit, vào ngày 22-04-2026, ủy quyền sử dụng giọng nói này cho mục đích X." Lưu trữ bản ghi này trong nhật ký bất biến chống giả mạo (tamper-evident log).

**Phát hiện giọng nói nhân bản (Detection).** AASIST, RawNet2 và Wav2Vec2-AASIST được triển khai làm các mô hình phát hiện giọng giả mạo. Thử thách ASVspoof 2025 công bố tỷ lệ lỗi cân bằng (Equal Error Rate - EER) từ 0.8–2.3% đối với các mô hình phát hiện tiên tiến nhất khi đối đầu với đầu ra từ ElevenLabs, VALL-E 2 và Bark.

### Số liệu thực tế (Năm 2026)

| Mô hình | Zero-shot? | SECS (Độ tương đồng) | WER (Độ rõ) | Tham số |
|---|---|---|---|---|
| F5-TTS | Có | 0.72 | 2.1% | 335M |
| XTTS v2 | Có | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Có | 0.70 | 2.8% | 220M |
| VALL-E 2 | Có | 0.77 | 2.4% | 370M |
| VoiceBox | Có | 0.78 | 2.1% | 330M |

Chỉ số SECS > 0.70 nhìn chung đã khiến đại đa số người nghe không thể phân biệt được với giọng thật của người nói mục tiêu.

````figure
sp-voice-factorize
````

## Tự xây dựng (Build It)

### Bước 1: Phân tách bằng phương pháp nhận dạng - tổng hợp (Bản minh họa bằng code trong main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Về mặt khái niệm thì rất đơn giản; khối lượng cài đặt chủ yếu nằm ở `tts_model` và bộ mã hóa người nói (speaker encoder).

### Bước 2: Nhân bản zero-shot với F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Bản ghi lời thoại tham chiếu phải khớp chính xác tuyệt đối với tệp âm thanh; nếu không khớp, quá trình gióng hàng sẽ bị lỗi.

### Bước 3: Chuyển đổi giọng nói với KNN-VC

```python
import torch
from knnvc import KNNVC  # Mô hình năm 2023, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC sử dụng WavLM để trích xuất các vector nhúng trên từng khung hình cho cả âm thanh nguồn và kho âm thanh mục tiêu, sau đó thay thế từng khung hình nguồn bằng láng giềng gần nhất (nearest neighbor) tương ứng trong kho mục tiêu. Đây là phương pháp phi tham số (non-parametric), hoạt động tốt chỉ với khoảng một phút giọng nói mục tiêu.

### Bước 4: Nhúng dấu bản quyền âm thanh (Watermark)

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # trả về các byte payload ban đầu
```

Nhúng khoảng 32 bit payload, có khả năng phát hiện lại sau khi nén MP3 hoặc thêm nhiễu nhẹ.

### Bước 5: Cổng kiểm duyệt đồng thuận (Consent gate)

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Bắt buộc phải có chữ ký đồng thuận"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Ứng dụng thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn |
|---|---|
| Nhân bản zero-shot 5 giây, mã nguồn mở | F5-TTS hoặc OpenVoice v2 |
| Nhân bản thương mại cho sản xuất | ElevenLabs Instant Voice Clone v2.5 |
| Chuyển đổi giọng nói (tái tạo âm điệu) | KNN-VC hoặc Diff-HierVC |
| Tinh chỉnh trên nhiều người nói | StyleTTS 2 + speaker adapter |
| Nhân bản đa ngôn ngữ (cross-lingual) | XTTS v2 hoặc VALL-E X |
| Phát hiện giọng giả (Deepfake detection) | Wav2Vec2-AASIST |

## Cạm bẫy thường gặp

- **Lời thoại tham chiếu bị lệch.** F5-TTS và các mô hình tương tự đòi hỏi văn bản tham chiếu phải khớp chính xác từng từ với âm thanh tham chiếu, bao gồm cả dấu câu.
- **Mẫu tham chiếu bị dội âm (reverberant).** Tiếng vọng sẽ làm hỏng kết quả nhân bản. Hãy luôn ghi âm trong môi trường khô (dry), đặt micro sát nguồn âm.
- **Lệch cảm xúc.** Mẫu tham chiếu có giọng "vui vẻ" sẽ tạo ra các bản sao nghe vui vẻ trong mọi tình huống. Hãy điều chỉnh cảm xúc của mẫu tham chiếu phù hợp với ngữ cảnh sử dụng thực tế.
- **Rò rỉ ngôn ngữ (Language leakage).** Khi nhân bản một người nói tiếng Anh rồi yêu cầu mô hình nói tiếng Pháp, giọng đọc thường bị giữ nguyên ngữ điệu tiếng Anh; hãy dùng các mô hình đa ngôn ngữ chuyên dụng (XTTS, VALL-E X).
- **Không có dấu bản quyền.** Về mặt pháp lý, sản phẩm sẽ không thể phát hành tại EU kể từ tháng 8 năm 2026.

## Xuất xưởng sản phẩm (Ship It)

Lưu thành `outputs/skill-voice-cloner.md`. Thiết kế một đường ống nhân bản hoặc chuyển đổi giọng nói hoàn chỉnh gồm cổng đồng thuận + dấu bản quyền âm thanh + mục tiêu chất lượng cụ thể.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Minh họa việc hoán đổi vector nhúng người nói bằng cách tính toán khoảng cách cosine giữa hai "người nói" trước và sau khi tráo đổi.
2. **Trung bình.** Sử dụng OpenVoice v2 để nhân bản chính giọng nói của bạn. Đo lường chỉ số SECS giữa tệp tham chiếu và bản nhân bản. Đo tỷ lệ lỗi từ (WER) qua Whisper.
3. **Khó.** Áp dụng dấu bản quyền SilentCipher cho 20 tệp nhân bản, đưa chúng qua quá trình nén và giải mã MP3 128 kbps, sau đó trích xuất payload. Báo cáo độ chính xác theo từng bit (bit-accuracy).

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Nhân bản không mẫu (Zero-shot clone) | 5 giây là đủ | Mô hình tiền huấn luyện + vector nhúng người nói; không cần huấn luyện lại. |
| PPG | Biểu đồ xác suất hậu nghiệm ngữ âm | Xác suất hậu nghiệm ASR trên từng khung hình, dùng làm biểu diễn nội dung độc lập với ngôn ngữ. |
| KNN-VC | Chuyển đổi láng giềng gần nhất | Thay thế từng khung hình nguồn bằng khung gần nhất trong kho âm thanh mục tiêu. |
| Codec nơ-ron TTS | Dạng VALL-E | Mô hình tự hồi quy (AR) hoạt động trên các token của EnCodec/SoundStream. |
| Dấu bản quyền (Watermark) | Chữ ký vô hình | Các bit được nhúng ẩn trong âm thanh, tồn tại sau quá trình nén lại. |
| SECS | Độ trung thực nhân bản | Khoảng cách cosine giữa vector nhúng người nói mục tiêu và vector nhúng bản sao. |
| AASIST | Bộ phát hiện deepfake | Mô hình chống giả mạo (anti-spoof); phát hiện giọng nói nhân tạo. |

## Tài liệu đọc thêm

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — Đỉnh cao công nghệ mã nguồn mở cho nhân bản zero-shot.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) và [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — TTS dựa trên codec nơ-ron.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — Chuyển đổi giọng nói dựa trên gỡ rối biểu diễn (disentanglement).
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — Chuyển đổi giọng nói dựa trên truy xuất láng giềng.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — Hệ thống đóng dấu bản quyền âm thanh 32-bit chuẩn sản xuất.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — Cuộc đua vũ trang giữa bộ phát hiện và mô hình tổng hợp giọng nói, cập nhật năm 2026.
