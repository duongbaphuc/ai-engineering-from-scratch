# Phát hiện Hoạt động Giọng nói & Luân phiên Lượt thoại — Silero, Cobra và Thủ thuật Flush

> Mọi trợ lý giọng nói đều thành công hay thất bại dựa trên hai quyết định: người dùng có đang nói hay không, và họ đã nói xong chưa? VAD trả lời câu hỏi thứ nhất. Bộ phát hiện lượt thoại (VAD + khoảng trễ im lặng + mô hình xác định điểm kết thúc theo ngữ nghĩa) trả lời câu hỏi thứ hai. Nếu quyết định sai một trong hai điều này, trợ lý của bạn sẽ hoặc là ngắt lời người dùng, hoặc là không bao giờ biết dừng lại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## Vấn đề Đặt ra

Trợ lý giọng nói phải đưa ra ba quyết định riêng biệt trên mỗi đoạn âm thanh (chunk) dài 20 ms:

1. **Khung hình này có phải là tiếng nói không?** — Phát hiện hoạt động giọng nói (Voice Activity Detection - VAD). Phân loại nhị phân trên từng khung hình.
2. **Người dùng đã bắt đầu một phát ngôn mới chưa?** — Phát hiện điểm bắt đầu (onset detection).
3. **Người dùng đã nói xong chưa?** — Xác định điểm kết thúc phát ngôn (endpointing / turn-end).

Cách tiếp cận ngây thơ (ngưỡng năng lượng - energy threshold) sẽ thất bại trước bất kỳ tạp âm nào — tiếng xe cộ, bàn phím, tiếng ồn đám đông. Giải pháp của năm 2026: Silero VAD (mã nguồn mở, học sâu) + mô hình phát hiện lượt thoại (xác định điểm kết thúc theo ngữ nghĩa - semantic endpointing) + khoảng trễ im lặng (silence hangover) được hiệu chuẩn theo VAD.

## Khái niệm Cốt lõi

![Phân tầng VAD: năng lượng → Silero → bộ phát hiện lượt thoại → thủ thuật flush](../assets/vad-turn-taking.svg)

### Phân tầng VAD ba cấp độ (Three-tier VAD cascade)

**Cấp 1: Cổng năng lượng (Energy gate).** Chi phí tính toán thấp nhất. Đặt ngưỡng RMS ở mức -40 dBFS. Lọc các đoạn im lặng rõ ràng nhưng bị kích hoạt bởi bất kỳ tạp âm nào vượt qua ngưỡng này.

**Cấp 2: Silero VAD** (2020-2026, MIT). 1 triệu tham số (1M parameters). Được huấn luyện trên hơn 6.000 ngôn ngữ. Chạy trong ~1 ms cho mỗi đoạn âm thanh 30 ms trên một luồng CPU đơn. Đạt TPR 87,7% tại FPR 5%. Đây là lựa chọn mặc định của cộng đồng mã nguồn mở.

**Cấp 3: Bộ phát hiện lượt thoại ngữ nghĩa (Semantic turn detector).** Mô hình phát hiện lượt thoại của LiveKit (2024-2026) hoặc bộ phân loại nhỏ tự xây dựng. Phân biệt giữa "tạm dừng giữa câu" và "đã nói xong". Sử dụng ngữ cảnh ngôn ngữ (ngữ điệu + các từ gần nhất), thay vì chỉ dựa vào khoảng im lặng.

### Các tham số quan trọng và giá trị mặc định

- **Ngưỡng (Threshold).** Silero xuất ra một xác suất; phân loại là tiếng nói khi xác suất > 0,5 (mặc định) hoặc > 0,3 (nhạy hơn). Ngưỡng thấp hơn = ít bị mất từ đầu tiên hơn, nhưng có nhiều kết quả dương tính giả hơn.
- **Thời lượng tiếng nói tối thiểu (Minimum speech duration).** Loại bỏ các đoạn tiếng nói ngắn hơn 250 ms — thường là tiếng ho hoặc tiếng xê dịch ghế.
- **Khoảng trễ im lặng (Silence hangover / end-pointing).** Sau khi VAD trở về 0, đợi thêm 500-800 ms trước khi tuyên bố kết thúc lượt thoại (end-of-turn). Quá ngắn → ngắt lời người dùng. Quá dài → tạo cảm giác chậm chạp, phản hồi trễ.
- **Bộ đệm đón đầu (Pre-roll buffer).** Giữ lại 300-500 ms âm thanh trước khi VAD được kích hoạt. Ngăn ngừa việc từ đầu tiên (ví dụ "hey") bị cắt cụt.

### Thủ thuật flush (Kyutai 2025)

Các mô hình nhận dạng tiếng nói (STT) phát trực tuyến có độ trễ nhìn trước (look-ahead delay: 500 ms đối với Kyutai STT-1B, 2,5 s đối với STT-2.6B). Thông thường, bạn sẽ phải đợi khoảng thời gian đó sau khi kết thúc tiếng nói để nhận được bản gỡ băng (transcript). Thủ thuật flush (flush trick): khi VAD phát hiện kết thúc tiếng nói, **hãy gửi một tín hiệu flush đến STT** nhằm ép buộc mô hình xuất kết quả ngay lập tức. STT xử lý ở tốc độ ~4× thời gian thực, do đó bộ đệm 500 ms sẽ được xử lý xong trong ~125 ms.

Quy trình khép kín: 125 ms VAD + flush STT = đạt độ trễ hội thoại tự nhiên.

### So sánh các VAD năm 2026

| VAD | TPR @ 5% FPR | Độ trễ (Latency) | Giấy phép (License) |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50,0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87,7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98,9% | ~1 ms | Thương mại (Commercial) |
| pyannote segmentation | 95% | ~10 ms | Kiểu MIT (MIT-ish) |

Silero là lựa chọn mặc định hợp lý. Cobra là bản nâng cấp về độ chính xác và tuân thủ doanh nghiệp. VAD chỉ dựa vào năng lượng thuần túy không còn chỗ đứng trong môi trường sản xuất năm 2026.

```figure
sp-vad-cascade
```

## Thực hành Xây dựng

### Bước 1: Cổng năng lượng

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Bước 2: Silero VAD trong Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Bước 3: Máy trạng thái kết thúc lượt thoại

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Bước 4: Khung mẫu triển khai thủ thuật flush

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

Mô hình STT (Kyutai, Deepgram, AssemblyAI) phải hỗ trợ tính năng flush thì thủ thuật này mới hoạt động được. Whisper streaming không hỗ trợ — nó hoạt động theo khối (block-based) và luôn phải chờ đủ các chunk.

## Ứng dụng Thực tế

| Tình huống | Lựa chọn VAD |
|-----------|-----------|
| Mã nguồn mở, nhanh, tổng quát | Silero VAD |
| Trung tâm cuộc gọi thương mại (Call center) | Cobra VAD |
| Chạy trên thiết bị (Điện thoại) | Silero VAD ONNX |
| Nghiên cứu / phân đoạn người nói (Diarization) | pyannote segmentation |
| Dự phòng không phụ thuộc thư viện ngoài (Zero-dependency fallback) | WebRTC VAD (cũ) |
| Yêu cầu chất lượng xác định lượt thoại cao | Phối hợp Silero + LiveKit turn-detector |

Quy tắc kinh nghiệm: Đừng bao giờ triển khai VAD chỉ dựa trên năng lượng trên môi trường sản xuất trừ khi bạn thực sự không có lựa chọn nào khác.

## Cạm bẫy Thường gặp

- **Ngưỡng cố định (Fixed threshold).** Hoạt động tốt trong môi trường yên tĩnh, nhưng thất bại trong môi trường ồn ào. Hãy hiệu chuẩn động trên thiết bị hoặc chuyển sang Silero.
- **Khoảng trễ im lặng quá ngắn.** Trợ lý sẽ ngắt lời người dùng ngay giữa câu. 500-800 ms là điểm cân bằng lý tưởng cho tiếng nói hội thoại.
- **Khoảng trễ quá dài.** Tạo cảm giác chậm chạp, ì ạch. Cần kiểm thử A/B với người dùng mục tiêu.
- **Không có bộ đệm đón đầu (Pre-roll buffer).** Mất 200-300 ms âm thanh đầu tiên của người dùng. Luôn duy trì một bộ đệm xoay vòng (rolling buffer).
- **Bỏ qua xác định điểm kết thúc ngữ nghĩa (Semantic endpointing).** "Ừm, để tôi nghĩ xem..." thường chứa những khoảng ngưng dài. Người dùng rất khó chịu khi bị ngắt lời giữa chừng suy nghĩ. Hãy sử dụng mô hình nhận diện lượt thoại của LiveKit hoặc giải pháp tương tự.

## Đóng gói Sản phẩm

Lưu thành `outputs/skill-vad-tuner.md`. Tinh chỉnh mô hình VAD, ngưỡng, khoảng trễ im lặng, bộ đệm đón đầu và chiến lược phát hiện lượt thoại cho một khối lượng công việc cụ thể.

## Bài tập Thực hành

1. **Dễ.** Chạy `code/main.py`. Tệp này mô phỏng một chuỗi gồm tiếng nói + im lặng + tiếng nói + tiếng ho và kiểm thử ba cấp độ VAD.
2. **Trung bình.** Cài đặt `silero-vad`, xử lý một bản ghi âm dài 5 phút, tinh chỉnh ngưỡng để vừa giảm thiểu việc cắt cụt từ đầu tiên vừa hạn chế kích hoạt sai. Báo cáo độ chính xác (precision) và độ thu hồi (recall).
3. **Khó.** Xây dựng một bộ phát hiện lượt thoại mini: kết hợp Silero VAD + một mạng MLP 3 lớp trên vector nhúng (embedding) của 10 từ gần nhất (sử dụng sentence-transformers). Huấn luyện trên tập dữ liệu điểm kết thúc lượt thoại được gán nhãn thủ công. Vượt qua điểm F1 của phương pháp chỉ dùng Silero ít nhất 10%.

## Thuật ngữ Then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| VAD | Bộ phát hiện giọng nói | Phân loại nhị phân trên từng khung hình: đây có phải tiếng nói không? |
| Nhận diện lượt thoại (Turn detection) | Xác định điểm kết thúc (End-pointing) | VAD + khoảng trễ im lặng + mô hình điểm kết thúc ngữ nghĩa. |
| Khoảng trễ im lặng (Silence hangover) | Chờ sau khi dứt lời | Thời gian chờ trước khi tuyên bố kết thúc lượt; 500-800 ms. |
| Bộ đệm đón đầu (Pre-roll) | Bộ đệm trước tiếng nói | Giữ lại 300-500 ms âm thanh trước khi VAD kích hoạt. |
| Thủ thuật flush (Flush trick) | Kỹ thuật tối ưu của Kyutai | VAD → kích hoạt flush STT → độ trễ giảm còn 125 ms thay vì 500 ms. |
| Điểm kết thúc ngữ nghĩa (Semantic endpoint) | "Họ đã thực sự nói xong chưa?" | Bộ phân loại ML dựa trên câu chữ ngữ nghĩa, không chỉ khoảng lặng âm thanh. |
| TPR @ FPR 5% | Điểm trên đường cong ROC | Chuẩn đánh giá VAD tiêu chuẩn; 87,7% đối với Silero, 50% đối với WebRTC. |

## Tài liệu Tham khảo

- [Silero VAD](https://github.com/snakers4/silero-vad) — VAD mã nguồn mở chuẩn tham chiếu.
- [Picovoice Cobra VAD](https://picovoice.ai/products/voice/voice-activity-detection/) — giải pháp thương mại dẫn đầu về độ chính xác.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — kỹ thuật kỹ thuật giảm độ trễ xuống dưới 200 ms.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — xác định điểm kết thúc ngữ nghĩa trong môi trường sản xuất.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — chuẩn đối sánh truyền thống.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — phân đoạn âm thanh cấp độ phân tích người nói.
