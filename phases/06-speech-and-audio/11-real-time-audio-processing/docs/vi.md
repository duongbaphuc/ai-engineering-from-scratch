# Xử lý âm thanh thời gian thực (Real-Time Audio Processing)

> Các đường ống theo lô (batch pipelines) xử lý trọn vẹn một tệp tin. Các đường ống thời gian thực (real-time pipelines) xử lý 20 mili giây tiếp theo trước khi 20 mili giây kế tiếp truyền tới. Mọi hệ thống AI đàm thoại, phòng thu phát sóng và bot thoại viễn thông đều phụ thuộc sống còn vào quỹ thời gian độ trễ (latency budget) này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 phút

## Vấn đề đặt ra

Bạn muốn xây dựng một trợ lý giọng nói mang lại cảm giác sống động, tự nhiên. Độ trễ chuyển lượt đàm thoại của con người (human conversational turn-taking latency) rơi vào khoảng ~230 ms (tính từ khi im lặng đến khi bắt đầu phản hồi). Bất kỳ khoảng thời gian nào trên 500 ms đều tạo cảm giác cơ học máy móc; và trên 1500 ms sẽ khiến hệ thống có cảm giác như bị hỏng. Quỹ thời gian độ trễ cho một chu trình hoàn chỉnh **nghe → hiểu → phản hồi → phát âm** vào năm 2026 được phân bổ như sau:

| Giai đoạn | Quỹ thời gian |
|---|---|
| Micro → bộ đệm (buffer) | 20 ms |
| VAD (Phát hiện tiếng nói) | 10 ms |
| ASR (Nhận dạng tiếng nói phát luồng) | 150 ms |
| LLM (Token đầu tiên) | 100 ms |
| TTS (Khối âm thanh đầu tiên) | 100 ms |
| Kết xuất (Render) → loa | 20 ms |
| **Tổng cộng** | **~400 ms** |

Moshi (Kyutai, 2024) đạt độ trễ 200 ms ở chế độ song công toàn phần (full-duplex). GPT-4o-realtime (2024) đạt khoảng ~320 ms. Các đường ống xếp tầng (cascaded pipelines) vào năm 2022 có độ trễ lên tới 2500 ms. Sự cải thiện gấp 10 lần này đến từ ba kỹ thuật trọng yếu: (1) phát luồng (streaming) trên toàn bộ hệ thống, (2) phân luồng bất đồng bộ với kết quả từng phần (partial results), (3) cơ chế sinh có thể ngắt quãng (interruptible generation).

## Khái niệm cốt lõi

![Đường ống âm thanh phát luồng với bộ đệm vòng, cổng VAD, cơ chế ngắt quãng](../assets/real-time.svg)

**Khung hình / Khối / Cửa sổ (Frame / chunk / window).** Âm thanh thời gian thực di chuyển dưới dạng các khối có kích thước cố định. Lựa chọn phổ biến: 20 ms (tương đương 320 mẫu ở tần số 16 kHz). Mọi thành phần phía sau đều phải theo kịp tốc độ của nhịp này.

**Bộ đệm vòng (Ring buffer).** Bộ đệm tuần hoàn có kích thước cố định. Luồng ghi (producer thread) ghi các khung hình mới, luồng đọc (consumer thread) đọc dữ liệu ra. Giúp tránh việc cấp phát bộ nhớ liên tục trên đường dẫn thực thi trọng yếu (hot path). Kích thước ≈ độ-trễ-tối-đa × tần-số-lấy-mẫu; bộ đệm vòng 2 giây ở tần số 16 kHz = 32.000 mẫu.

**VAD (Voice Activity Detection - Phát hiện hoạt động giọng nói).** Đóng vai trò là cổng chặn các tác vụ xử lý phía sau khi không có người nói. Silero VAD 4.0 (2024) xử lý dưới 1 ms cho mỗi khung 30 ms trên CPU. `webrtcvad` là giải pháp truyền thống thay thế.

**ASR phát luồng (Streaming ASR).** Các mô hình phát ra bản ghi lời thoại từng phần khi âm thanh truyền tới. Parakeet-CTC-0.6B ở chế độ phát luồng (NeMo, 2024) đạt tỷ lệ lỗi từ (WER) 2–5% với độ trễ 320 ms. Whisper-Streaming (Macháček et al., 2023) chia nhỏ Whisper thành từng khối để đạt khả năng gần phát luồng ở độ trễ ~2 giây.

**Cơ chế ngắt lời (Interruption / Barge-in).** Khi người dùng nói xen vào trong lúc trợ lý đang phát âm thanh, bạn phải: (a) phát hiện việc nói chèn (barge-in), (b) lập tức dừng TTS, (c) loại bỏ phần đầu ra còn lại của LLM. Toàn bộ thao tác này phải diễn ra trong vòng 100 ms, nếu không người dùng sẽ cảm thấy trợ lý như bị "điếc".

**Giao thức truyền tải WebRTC Opus.** Các khung 20 ms, tần số 48 kHz, tốc độ bit thích ứng 8–128 kbps. Là tiêu chuẩn cho trình duyệt và thiết bị di động. LiveKit, Daily.co, Pion là các ngăn xếp công nghệ phổ biến nhất năm 2026 để xây dựng ứng dụng thoại.

**Bộ đệm chống rung pha / lệch thời gian (Jitter buffer).** Các gói tin mạng thường truyền đến không theo thứ tự hoặc bị trễ. Bộ đệm jitter sắp xếp lại thứ tự và làm mượt dòng âm thanh; kích thước quá nhỏ → gây tiếng gián đoạn (audible gaps), quá lớn → tăng độ trễ. Kích thước tiêu chuẩn thường là 60–80 ms.

### Các cạm bẫy kỹ thuật phổ biến

- **Tranh chấp luồng (Thread contention).** Khóa luồng thông dịch toàn cục (GIL) của Python cùng các mô hình tính toán nặng có thể làm nghẽn luồng âm thanh. Hãy sử dụng thư viện âm thanh có hàm gọi lại (C-callback) như sounddevice, PortAudio và tránh để mã Python can thiệp vào đường dẫn xử lý nhạy cảm thời gian thực.
- **Độ trễ chuyển đổi tần số lấy mẫu (Resampling latency).** Tái lấy mẫu bên trong đường ống làm tăng thêm từ 5–20 ms. Bạn nên tái lấy mẫu ngay từ đầu hoặc sử dụng thuật toán tái lấy mẫu độ trễ bằng không (PolyPhase, `soxr_hq`).
- **Khởi động nguội của TTS (TTS priming).** Ngay cả các mô hình TTS nhanh như Kokoro cũng mất từ 100–200 ms để khởi động trong yêu cầu đầu tiên. Hãy lưu mô hình trong bộ nhớ cache và chạy làm ấm (warm-up) bằng một yêu cầu giả trước lượt hội thoại thật đầu tiên.
- **Khử tiếng vọng (AEC - Acoustic Echo Cancellation).** Nếu không có AEC, âm thanh phát ra từ loa sẽ lọt ngược vào micro và kích hoạt ASR nhận dạng chính giọng nói của bot. WebRTC AEC3 là lựa chọn mã nguồn mở mặc định.

````figure
nyquist-aliasing
````

## Tự xây dựng (Build It)

### Bước 1: Bộ đệm vòng (Ring buffer)

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Dung lượng (capacity) quyết định độ trễ đệm tối đa. 32.000 mẫu ở tần số 16 kHz tương đương 2 giây.

### Bước 2: Cổng kiểm soát VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Thay thế bằng Silero VAD trong môi trường sản xuất:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Bước 3: ASR phát luồng (Streaming ASR)

```python
# Parakeet-CTC-0.6B phát luồng thông qua NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Bước 4: Bộ xử lý cơ chế ngắt lời (Interruption handler)

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # người dùng nói chèn (barge-in)
        # sau đó đẩy dữ liệu vào bộ ASR phát luồng

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Cốt lõi nằm ở cơ chế I/O bất đồng bộ (async I/O) và khả năng hủy bỏ luồng phát TTS. Lệnh gọi `peerconnection.stop()` trên rãnh âm thanh của WebRTC là phương pháp tiêu chuẩn.

## Ứng dụng thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Tầng hệ thống | Lựa chọn |
|---|---|
| Giao thức truyền tải (Transport) | LiveKit (WebRTC) hoặc Pion (Go) |
| Phát hiện tiếng nói (VAD) | Silero VAD 4.0 |
| ASR phát luồng | Parakeet-CTC-0.6B hoặc Whisper-Streaming |
| LLM sinh token đầu tiên | Groq, Cerebras, vLLM-streaming |
| TTS phát luồng | Kokoro hoặc ElevenLabs Turbo v2.5 |
| Khử tiếng vọng (AEC) | WebRTC AEC3 |
| Đầu-cuối bản địa (End-to-end native) | OpenAI Realtime API hoặc Moshi |

## Cạm bẫy thường gặp

- **Đặt bộ đệm 500 ms cho an toàn.** Kích thước bộ đệm *chính là* mức sàn độ trễ tối thiểu của bạn. Hãy thu nhỏ nó lại.
- **Không cố định độ ưu tiên của luồng (Pinning threads).** Hàm gọi lại âm thanh chạy trên một luồng có độ ưu tiên thấp hơn giao diện người dùng sẽ gây méo tiếng và vấp âm thanh khi hệ thống tải nặng.
- **Khối TTS quá nhỏ.** Các khối âm thanh dưới 200 ms sẽ khiến các khiếm khuyết của vocoder bị lộ rõ tai người nghe. Kích thước 320 ms là điểm cân bằng lý tưởng.
- **Thiếu bộ đệm Jitter.** Mạng thực tế luôn có độ trễ dao động thất thường; nếu không có bộ đệm làm mượt, âm thanh sẽ bị nổ lách tách (pops).
- **Xử lý lỗi kiểu bắn-một-lần (Single-shot error handling).** Đường ống âm thanh bắt buộc phải có khả năng chống sập (crash-proof). Một ngoại lệ không được bắt sẽ làm chết toàn bộ phiên hội thoại.

## Xuất xưởng sản phẩm (Ship It)

Lưu thành `outputs/skill-realtime-designer.md`. Thiết kế một đường ống âm thanh thời gian thực với bảng phân bổ quỹ thời gian độ trễ cụ thể cho từng giai đoạn.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Chương trình sẽ mô phỏng một bộ đệm vòng + VAD dựa trên năng lượng; in ra độ trễ từng giai đoạn cho một luồng âm thanh mô phỏng 10 giây.
2. **Trung bình.** Sử dụng `sounddevice`, xây dựng một vòng lặp chuyển tiếp (passthrough loop) xử lý dữ liệu từ micro của bạn theo từng khối 20 ms và in ra trạng thái VAD tại mỗi khung hình.
3. **Khó.** Xây dựng một bài kiểm tra dội âm song công toàn phần với `aiortc`: trình duyệt → WebRTC → Python → WebRTC → trình duyệt. Đo độ trễ trọn vòng từ đầu-đến-cuối (glass-to-glass latency) bằng một xung tín hiệu 1 kHz.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Bộ đệm vòng (Ring buffer) | Hàng đợi vòng tròn | Hàng đợi FIFO có kích thước cố định, không khóa (hoặc khóa SPSC) dùng cho các khung âm thanh. |
| VAD | Cổng chặn im lặng | Mô hình hoặc thuật toán heuristic đánh dấu phân đoạn có tiếng nói hay không có tiếng nói. |
| ASR phát luồng | STT thời gian thực | Phát ra văn bản từng phần khi âm thanh vừa đến; giới hạn khoảng thời gian nhìn trước (lookahead). |
| Bộ đệm Jitter | Bộ làm mượt mạng | Hàng đợi sắp xếp lại các gói tin bị đảo thứ tự; thông thường dài 60–80 ms. |
| AEC | Khử tiếng vọng | Triệt tiêu đường phản hồi âm thanh từ loa ngoài vào micro. |
| Barge-in | Ngắt lời bot | Hệ thống phát hiện tiếng người dùng nói chèn giữa lúc phát TTS; phải lập tức hủy phát lại. |
| Song công toàn phần (Full duplex) | Hai chiều đồng thời | Người dùng và bot có thể nói cùng một lúc; Moshi hoạt động ở chế độ song công toàn phần. |

## Tài liệu đọc thêm

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — Phân khối Whisper để đạt khả năng gần phát luồng.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — Mô hình song công toàn phần với độ trễ 200 ms.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — Nền tảng điều phối tác nhân âm thanh chuẩn sản xuất.
- [Kho mã nguồn Silero VAD](https://github.com/snakers4/silero-vad) — VAD xử lý dưới 1 ms, giấy phép Apache 2.0.
- [Bài báo WebRTC AEC3](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — Khử tiếng vọng trong mã nguồn mở.
