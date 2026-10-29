# Xây dựng đường ống trợ lý giọng nói (Voice Assistant Pipeline) — Dự án tổng kết Giai đoạn 6

> Tập hợp toàn bộ kiến thức từ bài 01 đến bài 11 thành một khối hoàn chỉnh. Xây dựng một trợ lý giọng nói (voice assistant) có khả năng lắng nghe, lập luận và đối đáp linh hoạt. Vào năm 2026, đây là một bài toán kỹ thuật đã có lời giải rõ ràng thay vì là đề tài nghiên cứu — tuy nhiên các chi tiết tích hợp thực tế sẽ quyết định sản phẩm có thể xuất xưởng hay không.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 phút

## Vấn đề đặt ra

Xây dựng một trợ lý hoàn chỉnh từ đầu-đến-cuối (end-to-end):

1. Thu nhận tín hiệu âm thanh từ micro (16 kHz mono).
2. Phát hiện thời điểm bắt đầu/kết thúc câu nói của người dùng.
3. Chuyển đổi giọng nói thành văn bản phát luồng (streaming transcription).
4. Truyền bản ghi lời thoại vào một LLM có khả năng gọi công cụ (hẹn giờ, thời tiết, lịch trình).
5. Phát luồng văn bản từ LLM sang bộ tổng hợp tiếng nói (TTS).
6. Phát âm thanh phản hồi lại cho người dùng qua loa.
7. Dừng ngay lập tức nếu người dùng nói chen ngang khi đang phản hồi.

Mục tiêu độ trễ: byte âm thanh đầu tiên từ TTS phải được phát ra trong vòng 800 ms kể từ khi người dùng dứt lời, ngay trên CPU máy tính xách tay. Mục tiêu chất lượng: không nuốt từ, không tạo phụ đề ảo giác khi im lặng, không rò rỉ giọng nói nhân bản, không bị tấn công tiêm câu lệnh (prompt injection).

## Khái niệm cốt lõi

![Đường ống trợ lý giọng nói: micro → VAD → STT → LLM+công cụ → TTS → loa](../assets/voice-assistant.svg)

### Bảy thành phần cấu thành

1. **Thu nhận âm thanh (Audio capture).** Micro → âm thanh đơn kênh 16 kHz mono → các khối 20 ms. Thường sử dụng thư viện `sounddevice` trong Python hoặc các giao diện bản địa như AudioUnit/ALSA/WASAPI trong môi trường sản xuất.
2. **VAD - Phát hiện giọng nói (Bài 11).** Silero VAD với ngưỡng xác suất 0.5, thời lượng tiếng nói tối thiểu 250 ms, khoảng im lặng chờ ngắt lượt (hang-over) 500 ms. Phát tín hiệu "bắt đầu" và "kết thúc".
3. **STT phát luồng (Bài 4-5).** Whisper-streaming, Parakeet-TDT hoặc Deepgram Nova-3 (qua API). Cung cấp cả bản ghi từng phần lẫn bản ghi chính thức cuối cùng.
4. **LLM có hỗ trợ gọi công cụ (Tool calling).** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. Định dạng công cụ theo lược đồ JSON (JSON schema). Phát luồng các token đầu ra.
5. **TTS phát luồng (Bài 7).** Kokoro-82M (mã nguồn mở nhanh nhất) hoặc Cartesia Sonic (thương mại). Bắt đầu kích hoạt TTS ngay sau khi nhận được 20 token đầu tiên từ LLM.
6. **Phát lại âm thanh (Playback).** Xuất ra loa; mã hóa theo chuẩn Opus cho các đường truyền mạng băng thông thấp.
7. **Bộ xử lý ngắt lời (Interruption handler).** Nếu VAD phát hiện tiếng nói trong khi đang phát âm thanh từ TTS: lập tức dừng phát âm thanh, hủy tác vụ LLM và khởi động lại quá trình STT.

### Ba dạng lỗi thường gặp nhất

1. **Nuốt từ đầu tiên (First-word clip).** VAD phản ứng chậm một nhịp, khiến từ chào hỏi ban đầu (như "hey") của người dùng bị cắt mất. Khắc phục bằng cách hạ ngưỡng kích hoạt ban đầu xuống 0.3 thay vì 0.5, và dùng bộ đệm giữ lại âm thanh trước khi kích hoạt (pre-roll buffer).
2. **Rối loạn khi bị ngắt lời giữa chừng.** LLM vẫn tiếp tục sinh phản hồi dù người dùng đã ngắt lời; khiến trợ lý nói đè lên tiếng của người dùng. Cần đấu nối trực tiếp: tín hiệu VAD → lệnh hủy LLM.
3. **Ảo giác trong khoảng im lặng.** Mô hình Whisper xuất ra những câu ngẫu nhiên như "Thanks for watching" trên các khung hình im lặng lúc khởi động. Luôn luôn phải chặn bằng cổng VAD.

### Ngăn xếp tham chiếu chuẩn sản xuất năm 2026

| Ngăn xếp | Độ trễ | Giấy phép | Ghi chú |
|---|---|---|---|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | API thương mại | Chuẩn mực mặc định của ngành năm 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | Phần lớn mã nguồn mở | Thân thiện cho tự dựng (DIY) |
| Moshi (song công toàn phần) | 200-300 ms | CC-BY 4.0 | Mô hình đơn nhất; kiến trúc khác biệt, xem Bài 15 |
| Vapi / Retell (dịch vụ trọn gói) | 300-500 ms | Thương mại | Triển khai nhanh nhất; hạn chế khả năng tùy biến sâu |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | Ngoại tuyến (offline) | Mã nguồn mở | Đảm bảo tính riêng tư / triển khai tại biên (edge) |

````figure
v4-voice-latency
````

## Tự xây dựng (Build It)

### Bước 1: Thu nhận âm thanh từ micro theo khối (Mã giả)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Bước 2: Thu nhận lượt nói có cổng VAD kiểm soát

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Bước 3: Chuỗi liên hoàn phát luồng STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Bước 4: Gọi công cụ bên trong vòng lặp LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Bước 5: Xử lý cơ chế ngắt lời (Interruption handling)

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Ứng dụng thực tế (Use It)

Xem `code/main.py` để chạy một bản mô phỏng hoàn chỉnh kết nối toàn bộ 7 thành phần với các mô hình mô phỏng (stub models), giúp bạn nắm rõ hình thái đường ống ngay cả khi chưa có phần cứng chuyên dụng. Khi triển khai thực tế, bạn chỉ cần thay thế các mô hình mô phỏng bằng:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` hoặc `openai-whisper`
- `openai` (`gpt-4o`) hoặc `anthropic`
- `kokoro` hoặc `cartesia`
- `sounddevice` cho các thao tác vào/ra âm thanh

## Cạm bẫy thường gặp

- **Lưu trữ dữ liệu định danh cá nhân (PII) vô thời hạn.** Toàn bộ âm thanh lượt nói của người dùng được xem là dữ liệu cá nhân theo hầu hết các khung pháp lý. Chỉ nên lưu tối đa 30 ngày và mã hóa khi lưu trữ (encrypted at rest).
- **Không có cơ chế nói chèn (Barge-in).** Người dùng sẽ ngắt lời. Trợ lý bắt buộc phải dừng nói ngay lập tức.
- **TTS gây nghẽn luồng (Blocking).** Việc tổng hợp TTS đồng bộ sẽ làm nghẽn vòng lặp sự kiện (event loop). Hãy dùng cơ chế bất đồng bộ hoặc chạy trên một luồng riêng.
- **Không xử lý lỗi khi gọi công cụ.** Các công cụ bên ngoài hoàn toàn có thể gặp lỗi. LLM phải nhận được thông báo lỗi + thử lại một lần, sau đó xuống cấp tính năng một cách êm thuận (gracefully degrade).
- **Bộ lọc ảo giác quá gắt.** Nếu lọc quá mức, trợ lý sẽ liên tục lặp lại câu "Tôi không thể giúp việc đó." Nếu lọc quá lỏng, nó sẽ nói linh tinh. Hãy cân chỉnh trên một tập dữ liệu kiểm thử riêng biệt.
- **Thiếu tùy chọn từ khóa đánh thức (Wake-word).** Việc micro luôn luôn mở để lắng nghe là một rủi ro xâm phạm quyền riêng tư. Hãy bổ sung cổng kích hoạt bằng từ khóa đánh thức (Porcupine hoặc openWakeWord).

## Xuất xưởng sản phẩm (Ship It)

Lưu thành `outputs/skill-voice-assistant-architect.md`. Xây dựng một bản đặc tả hoàn chỉnh cho toàn bộ hệ thống dựa trên các ràng buộc về chi phí + quy mô + ngôn ngữ + tuân thủ pháp lý.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Chương trình sẽ mô phỏng trọn vẹn một lượt hội thoại từ đầu-đến-cuối bằng các mô-đun giả lập và in ra độ trễ của từng công đoạn.
2. **Trung bình.** Thay thế mô-đun STT giả lập bằng một mô hình Whisper thực sự trên một tệp `.wav` ghi âm sẵn. Đo lường tỷ lệ lỗi từ (WER) và độ trễ toàn trình từ đầu-đến-cuối.
3. **Khó.** Tích hợp tính năng gọi công cụ: hiện thực hai hàm `get_weather` (qua bất kỳ API thời tiết nào) và `set_timer`. Điều hướng LLM gọi qua các công cụ này và xác minh rằng khi người dùng nói "hẹn giờ 5 phút", hàm tương ứng được thực thi chính xác và phản hồi bằng giọng nói xác nhận điều đó.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Lượt đàm thoại (Turn) | Một vòng tương tác | Một đoạn nói của người dùng (giới hạn bởi VAD) + một câu trả lời từ LLM-TTS. |
| Barge-in | Ngắt lời | Người dùng cất tiếng nói trong khi trợ lý đang phát âm thanh; trợ lý lập tức dừng lại. |
| Từ đánh thức (Wake word) | "Hey Siri/Google" | Bộ phát hiện từ khóa ngắn; ví dụ Porcupine, Snowboy, openWakeWord. |
| Xác định điểm kết thúc (End-pointing) | Điểm kết thúc câu nói | Quyết định dựa trên VAD + khoảng im lặng tối thiểu để xác nhận người dùng đã nói xong. |
| Bộ đệm đón đầu (Pre-roll) | Bộ đệm trước tiếng nói | Giữ lại 200–400 ms âm thanh trước khi VAD kích hoạt nhằm tránh hiện tượng nuốt từ đầu tiên. |
| Gọi công cụ (Tool call) | Thực thi hàm | LLM xuất ra chuỗi JSON; môi trường thực thi gọi hàm tương ứng; kết quả được đưa ngược lại vào vòng lặp. |

## Tài liệu đọc thêm

- [Hướng dẫn bắt đầu nhanh tác nhân giọng nói của LiveKit](https://docs.livekit.io/agents/) — Tài liệu tham khảo chuẩn sản xuất.
- [Các ví dụ về tác nhân giọng nói trên Pipecat](https://github.com/pipecat-ai/pipecat) — Khung phát triển thân thiện cho người tự xây dựng.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — Lối tiếp cận bản địa tiếng nói được quản lý sẵn.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — Tài liệu tham khảo hệ thống song công toàn phần (Bài 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — Kiểm soát kích hoạt bằng từ khóa đánh thức.
- [Hướng dẫn sử dụng công cụ của Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Kỹ thuật gọi hàm với LLM.
