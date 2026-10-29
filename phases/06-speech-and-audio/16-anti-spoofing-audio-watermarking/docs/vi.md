# Chống Giả mạo Giọng nói & Thủy vân Âm thanh — ASVspoof 5, AudioSeal, WaveVerify

> Công nghệ nhân bản giọng nói (voice cloning) được triển khai nhanh hơn các giải pháp phòng thủ. Các hệ thống âm thanh triển khai trong môi trường sản xuất năm 2026 cần hai yếu tố cốt lõi: một bộ phát hiện (detector như AASIST, RawNet2) để phân loại tiếng nói thật và giả, cùng một thủy vân âm thanh (watermark như AudioSeal) có khả năng tồn tại qua các quá trình nén và chỉnh sửa. Hãy triển khai cả hai, nếu không đừng phát hành tính năng nhân bản giọng nói.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## Vấn đề Đặt ra

Ba tầng phòng thủ có liên quan chặt chẽ với nhau:

1. **Chống giả mạo / Phát hiện deepfake (Anti-spoofing / deepfake detection).** Cho một đoạn âm thanh, đây là âm thanh tổng hợp hay giọng nói thật? Các bộ chuẩn đánh giá ASVspoof (ASVspoof 2019 → 2021 → 5) là tiêu chuẩn vàng.
2. **Thủy vân âm thanh (Audio watermarking).** Nhúng một tín hiệu ẩn (không thể nghe thấy) vào âm thanh được tạo ra để bộ phát hiện có thể trích xuất lại sau này. AudioSeal (Meta) và WavMark là những giải pháp mã nguồn mở tiêu biểu.
3. **Nguồn gốc được xác thực (Authenticated provenance).** Ký mã hóa (cryptographic signing) các tệp âm thanh cùng với siêu dữ liệu (metadata). Tiêu chuẩn C2PA / Content Authenticity Initiative.

Phương pháp phát hiện nhắm vào những kẻ tấn công không hợp tác. Phương pháp thủy vân phục vụ việc tuân thủ quy định (compliance) — âm thanh do AI tạo ra phải có khả năng nhận biết được. Cả hai đều là yêu cầu bắt buộc vào năm 2026.

## Khái niệm Cốt lõi

![Chống giả mạo so với thủy vân số so với nguồn gốc xác thực — ba tầng phòng thủ](../assets/spoofing-watermark.svg)

### ASVspoof 5 — Chuẩn đánh giá giai đoạn 2024-2025

Những thay đổi lớn nhất so với các phiên bản trước:

- **Dữ liệu thu thập từ cộng đồng (Crowdsourced data)** (không phải dữ liệu phòng thu sạch sẽ) — phản ánh các điều kiện thực tế.
- **~2.000 người nói** (so với ~100 người ở các phiên bản trước).
- **32 thuật toán tấn công.** Bao gồm tổng hợp giọng nói (TTS) + chuyển đổi giọng nói (voice conversion) + nhiễu loạn đối kháng (adversarial perturbation).
- **Hai bài toán đánh giá (Two tracks).** Biện pháp đối phó độc lập (Countermeasure - CM standalone detection); và Xác minh người nói chống giả mạo (Spoofing-robust ASV - SASV) dành cho các hệ thống sinh trắc học.

Hiệu năng hiện đại nhất (SOTA) trên ASVspoof 5: đạt ~7,23% EER (Equal Error Rate). Trên tập dữ liệu cũ ASVspoof 2019 LA: đạt 0,42% EER. Khi triển khai trong thế giới thực: dự kiến đạt 5-10% EER trên các đoạn âm thanh tự nhiên (in-the-wild).

### AASIST và RawNet2 — Các họ mô hình phát hiện

**AASIST** (2021, được cập nhật liên tục đến 2026). Cơ chế chú ý đồ thị (Graph-attention) trên các đặc trưng phổ (spectral features). Hiện đang là SOTA trên tác vụ biện pháp đối phó của ASVspoof 5.

**RawNet2.** Khối tiền xử lý tích chập trên dạng sóng thô + mạng xương sống TDNN (Time-Delay Neural Network). Mô hình đường cơ sở (baseline) đơn giản hơn; vẫn có tính cạnh tranh cao khi được tinh chỉnh.

**NeXt-TDNN + đặc trưng học tự giám sát (SSL features).** Biến thể năm 2025: kiến trúc kiểu ECAPA + đặc trưng trích xuất từ WavLM + hàm mất mát focal loss. Đạt 0,42% EER trên ASVspoof 2019 LA.

### AudioSeal — Chuẩn thủy vân mặc định năm 2024

**AudioSeal** của Meta (tháng 1/2024, bản v0.2 tháng 12/2024). Các điểm thiết kế chính:

- **Định vị cục bộ (Localized).** Phát hiện thủy vân trên từng khung hình ở độ phân giải lấy mẫu 16 kHz (1/16000 giây).
- **Bộ tạo và bộ phát hiện được huấn luyện đồng thời (Jointly trained).** Bộ tạo học cách nhúng tín hiệu không thể nghe thấy; bộ phát hiện học cách tìm ra tín hiệu đó thông qua các phép biến đổi tăng cường dữ liệu (augmentations).
- **Mạnh mẽ, bền vững (Robust).** Tồn tại tốt qua quá trình nén MP3 / AAC, cân bằng âm sắc (EQ), thay đổi tốc độ ±10%, trộn tạp âm với tỷ lệ tín hiệu trên nhiễu (SNR) +10 dB.
- **Tốc độ cao.** Bộ phát hiện chạy nhanh gấp 485× thời gian thực; nhanh hơn 1.000 lần so với WavMark.
- **Dung lượng tải trọng (Capacity).** Tải trọng 16-bit (có thể mã hóa ID mô hình, dấu thời gian sinh, ID người dùng) có thể nhúng vào từng phát ngôn.

### WavMark

Mô hình chuẩn đối sánh mã nguồn mở trước thời AudioSeal. Sử dụng mạng nơ-ron khả nghịch (invertible neural network), truyền tải 32 bit/giây. Hạn chế:

- Tìm kiếm đồng bộ bằng phương pháp vét cạn (brute-force) rất chậm.
- Dễ bị loại bỏ bởi nhiễu Gauss hoặc nén MP3.
- Không thân thiện với các ứng dụng thời gian thực.

### WaveVerify (Tháng 7/2025)

Giải quyết các điểm yếu của AudioSeal — đặc biệt là các thao tác can thiệp theo trục thời gian (đảo ngược âm thanh, thay đổi tốc độ). Sử dụng bộ tạo dựa trên FiLM kết hợp bộ phát hiện cấu trúc chuyên gia hỗn hợp (Mixture-of-Experts - MoE). Đạt kết quả tương đương AudioSeal trên các kiểu tấn công tiêu chuẩn và xử lý rất tốt các chỉnh sửa về thời gian.

### Lỗ hổng mà kẻ tấn công khai thác

Theo kết quả từ AudioMarkBench: "dưới tác động của phép dịch cao độ (pitch shift), tất cả các kỹ thuật thủy vân đều có Độ chính xác khôi phục bit (Bit Recovery Accuracy) dưới 0,6, cho thấy thủy vân gần như bị xóa bỏ hoàn toàn." **Dịch cao độ là đòn tấn công phổ quát.** Chưa có kỹ thuật thủy vân nào vào năm 2026 hoàn toàn miễn nhiễm trước việc biến đổi cao độ mạnh. Đây chính là lý do vì sao bạn phải triển khai bộ phát hiện (như AASIST) song hành cùng với thủy vân.

### C2PA / Content Authenticity Initiative

Không phải là một kỹ thuật học máy — mà là một định dạng bản kê khai nguồn gốc (manifest format). Các tệp âm thanh mang siêu dữ liệu được ký mã hóa về công cụ tạo, tác giả, ngày tạo. Audobox / Seamless sử dụng tiêu chuẩn này. Phù hợp để chứng minh nguồn gốc xuất xứ; nhưng trở nên vô tác dụng nếu kẻ xấu mã hóa lại tệp âm thanh và lược bỏ siêu dữ liệu.

```figure
v4-audio-watermark
```

## Thực hành Xây dựng

### Bước 1: Bộ phát hiện đặc trưng phổ đơn giản (Mô hình đồ chơi)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Tiếng nói tổng hợp thường có năng lượng tần số cao phẳng một cách bất thường. Các bộ phát hiện trên môi trường sản xuất sử dụng AASIST chứ không dùng hàm đơn giản này. Tuy nhiên, trực giác phân tích thì tương tự.

### Bước 2: Nhúng và phát hiện thủy vân bằng AudioSeal

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: giá trị thực trong [0, 1] — xác suất hiện diện của thủy vân
# decoded_payload: 16 bit; so khớp với tải trọng đã nhúng
```

### Bước 3: Đánh giá hiệu năng — Tỷ lệ Sai số Cân bằng (EER)

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Bước 4: Tích hợp trên môi trường sản xuất

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Mỗi đoạn âm thanh sinh ra đều đi kèm: (1) thủy vân, (2) bản kê khai đã ký số, (3) nhật ký kiểm toán tuân thủ chính sách lưu trữ.

## Ứng dụng Thực tế

| Trường hợp sử dụng | Biện pháp bảo vệ |
|----------|---------|
| Phát hành mô hình TTS / nhân bản giọng nói | Nhúng AudioSeal trên mọi đầu ra (yêu cầu bắt buộc) |
| Mở khóa sinh trắc học bằng giọng nói | Bộ kết hợp AASIST + ECAPA; thử thách kiểm tra sự sống (liveness) |
| Phát hiện gian lận trong trung tâm cuộc gọi | Chạy AASIST trên mẫu ngẫu nhiên 20% các cuộc gọi đến |
| Xác thực nội dung podcast | Ký số C2PA khi tải lên, nhúng AudioSeal nếu do AI tạo |
| Nghiên cứu / huấn luyện bộ phát hiện | Sử dụng các tập huấn luyện/phát triển/đánh giá của ASVspoof 5 |

## Cạm bẫy Thường gặp

- **Nhúng thủy vân nhưng không bao giờ chạy bộ phát hiện.** Vô nghĩa. Hãy tích hợp bộ phát hiện vào quy trình kiểm thử CI/CD của bạn.
- **Phát hiện thiếu hiệu chuẩn (Calibration).** AASIST huấn luyện trên tập dữ liệu ASVspoof LA rất dễ bị quá khớp (overfit); độ chính xác trong thế giới thực sẽ sụt giảm. Hãy hiệu chuẩn mô hình trên miền dữ liệu thực tế của bạn.
- **Khoảng trống tấn công dịch cao độ.** Thao tác dịch cao độ mạnh làm mất hầu hết các loại thủy vân. Luôn cần có bộ phát hiện nơ-ron làm giải pháp dự phòng.
- **Bóc tách siêu dữ liệu rồi đăng tải lại.** C2PA rất dễ bị vô hiệu hóa chỉ bằng cách giải mã rồi nén lại. Luôn kết hợp bảo vệ bằng chữ ký số song hành cùng thủy vân ẩn trong âm thanh.
- **Nhầm lẫn kiểm tra sự sống với phát hiện giả mạo.** Yêu cầu người dùng đọc một cụm từ ngẫu nhiên có thể ngăn chặn tấn công phát lại (replay attack) nhưng không ngăn chặn được công nghệ nhân bản giọng nói theo thời gian thực.

## Đóng gói Sản phẩm

Lưu thành `outputs/skill-spoof-defender.md`. Lựa chọn mô hình phát hiện, kỹ thuật thủy vân, bản kê khai nguồn gốc và quy trình vận hành bảo vệ cho hệ thống triển khai sinh giọng nói.

## Bài tập Thực hành

1. **Dễ.** Chạy `code/main.py`. Tệp này mô phỏng một bộ phát hiện đồ chơi kết hợp bộ nhúng/phát hiện thủy vân đồ chơi trên âm thanh tổng hợp.
2. **Trung bình.** Cài đặt `audioseal`, nhúng tải trọng 16-bit vào đầu ra của một mô hình TTS, sau đó giải mã lại. Làm biến dạng âm thanh bằng tiếng ồn và đo Độ chính xác Khôi phục Bit (Bit Recovery Accuracy).
3. **Khó.** Tinh chỉnh mô hình RawNet2 hoặc AASIST trên tập dữ liệu ASVspoof 2019 LA. Đo chỉ số EER. Đánh giá trên một tập kiểm thử riêng biệt gồm các đoạn âm thanh sinh bởi F5-TTS để quan sát độ suy giảm hiệu năng khi gặp dữ liệu ngoài phân phối (out-of-distribution - OOD).

## Thuật ngữ Then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| ASVspoof | Bộ chuẩn đánh giá | Thử thách đánh giá 2 năm một lần; năm 2024 là ASVspoof 5. |
| CM (Biện pháp đối phó) | Bộ phát hiện | Bộ phân loại: tiếng nói thật so với tiếng nói tổng hợp / chuyển đổi. |
| SASV | Xác minh người nói + CM | Hệ thống tích hợp xác thực sinh trắc học và phát hiện giả mạo. |
| AudioSeal | Thủy vân của Meta | Định vị cục bộ, tải trọng 16-bit, nhanh hơn 485× so với WavMark. |
| Bit Recovery Accuracy | Tỷ lệ sống sót của thủy vân | Tỷ lệ phần trăm các bit tải trọng khôi phục thành công sau tấn công. |
| C2PA | Bản kê khai nguồn gốc | Siêu dữ liệu ký mã hóa về nguồn gốc khởi tạo / tác giả. |
| AASIST | Dòng mô hình phát hiện | Kiến trúc chống giả mạo tiên tiến nhất dựa trên cơ chế chú ý đồ thị. |

## Tài liệu Tham khảo

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — chuẩn đánh giá hiện hành.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — giải pháp thủy vân mặc định.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — bộ phát hiện MoE chống tấn công trục thời gian.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — xương sống phát hiện SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — đánh giá độ bền vững.
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/index.html) — định dạng bản kê khai xuất xứ nội dung.
