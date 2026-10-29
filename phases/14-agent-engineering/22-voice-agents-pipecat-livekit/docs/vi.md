# Tác tử giọng nói: Pipecat và LiveKit

> Tác tử giọng nói (voice agents) đã trở thành một nhánh phát triển chính thức trong môi trường sản xuất năm 2026. Pipecat mang đến cho bạn một đường ống xử lý theo khung dữ liệu (frame-based pipeline) trên nền Python (VAD → STT → LLM → TTS → transport). LiveKit Agents đóng vai trò cầu nối các mô hình AI tới người dùng qua giao thức WebRTC. Mục tiêu độ trễ thực tế trong môi trường sản xuất rơi vào khoảng 450–600ms đầu-cuối (end-to-end) cho các ngăn xếp công nghệ cao cấp.

**Type:** Learn  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)  
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả đường ống xử lý theo khung dữ liệu của Pipecat: luồng XUÔI (DOWNSTREAM, từ nguồn đến đích) và luồng NGƯỢC (UPSTREAM, điều khiển phản hồi).
- Nêu tên các giai đoạn chuẩn tắc của đường ống giọng nói cùng các giao thức truyền tải (transports) mà Pipecat hỗ trợ.
- Phân tích hai lớp tác tử giọng nói của LiveKit Agents (`MultimodalAgent`, `VoicePipelineAgent`) và ngữ cảnh ứng dụng phù hợp cho từng lớp.
- Tóm tắt kỳ vọng về độ trễ trong môi trường sản xuất năm 2026 và cách các chỉ số này định hình các quyết định kiến trúc.

## Vấn đề đặt ra

Tác tử giọng nói không đơn thuần là một vòng lặp xử lý văn bản được gắn thêm bộ tổng hợp tiếng nói (TTS). Ngân sách độ trễ cực kỳ khắt khe (~600ms), dữ liệu âm thanh dạng dòng từng phần (partial audio) là mặc định, cơ chế phát hiện lượt thoại (turn detection) bản chất là một mô hình học máy, và các giao thức truyền tải trải dài từ giao thức SIP của mạng viễn thông tới WebRTC. Bạn hoặc phải tự xây dựng một đường ống theo khung dữ liệu (Pipecat) hoặc phải tựa vào một nền tảng hạ tầng hoàn chỉnh (LiveKit).

## Khái niệm cốt lõi

### Pipecat (pipecat-ai/pipecat)

- Khung phát triển đường ống hướng khung dữ liệu (frame-based pipeline framework) viết bằng Python.
- Chuỗi kết nối giữa các đơn vị xử lý `Frame` → `FrameProcessor`.
- Hai hướng di chuyển của dữ liệu:
  - **DOWNSTREAM (Luồng xuôi)** — từ nguồn tới đích (nhận âm thanh đầu vào, xuất giọng nói TTS đầu ra).
  - **UPSTREAM (Luồng ngược)** — phản hồi và điều khiển (hủy yêu cầu, thu thập số liệu đo lường, xử lý chen ngang - barge-in).
- `PipelineTask` quản trị vòng đời đường ống với các sự kiện (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) kèm các bộ quan sát (observers) phục vụ thu thập chỉ số/truy vết/RTVI.

Đường ống điển hình:

```text
VAD (Silero) → STT → LLM (ngữ cảnh luân phiên user/assistant) → TTS → transport
```

Các phương thức truyền tải (transports) hỗ trợ: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows bổ sung khả năng điều hướng cuộc hội thoại có cấu trúc (máy trạng thái - state machines). Pipecat Cloud là môi trường thực thi được quản lý sẵn (managed runtime).

### LiveKit Agents (livekit/agents)

- Cầu nối đưa các mô hình AI đến với người dùng thông qua giao thức WebRTC thời gian thực.
- Các khái niệm trọng tâm: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Hai lớp tác tử giọng nói chính:
  - **MultimodalAgent** — xử lý âm thanh trực tiếp hai chiều thông qua OpenAI Realtime hoặc các dịch vụ tương đương.
  - **VoicePipelineAgent** — chuỗi nối tầng (cascade) STT → LLM → TTS; mang lại khả năng can thiệp và kiểm soát ở cấp độ văn bản.
- Nhận diện lượt thoại ngữ nghĩa (semantic turn detection) sử dụng mô hình transformer chuyên dụng.
- Tích hợp giao thức bối cảnh mô hình (MCP) tự nhiên.
- Hỗ trợ kết nối mạng điện thoại thông qua SIP.
- Cung cấp hơn 50 mô hình không cần cấu hình API key qua LiveKit Inference; cùng hơn 200 mô hình khác qua hệ sinh thái plugin.

### Các nền tảng thương mại

Vapi (đạt ~450–600ms trên ngăn xếp công nghệ cao cấp được tối ưu) và Retell (~600ms đầu-cuối qua bài kiểm thử 180 cuộc gọi) được xây dựng dựa trên các nền tảng này. Hãy chọn một nền tảng thương mại nếu bạn cần một ngăn xếp giọng nói được quản lý toàn diện mà không cần duy trì một đội ngũ kỹ sư chuyên sâu về WebRTC.

### Các điểm nghẽn và sai lầm thường gặp

- **Không xử lý tình huống chen ngang (barge-in):** Người dùng cất lời ngắt quãng nhưng tác tử vẫn tiếp tục nói. Tình huống này đòi hỏi việc phát các khung hủy UPSTREAM cancel frames trong Pipecat hoặc cơ chế tương đương trong LiveKit.
- **Bỏ qua độ tin cậy của STT:** Văn bản phiên âm có độ tin cậy thấp vẫn bị nạp thẳng vào LLM như chân lý. Cần thiết lập cổng chặn theo ngưỡng tin cậy hoặc yêu cầu người dùng xác nhận lại.
- **Cắt ngang âm thanh TTS giữa chừng không kiểm soát:** Khi đường ống bị hủy ở giữa câu thoại, thành phần TTS cần nhận biết để dừng phát luồng âm thanh một cách êm thuận.
- **Bỏ quên ngân sách độ trễ:** Mỗi thành phần đóng góp từ 50–200ms vào độ trễ tổng thể. Luôn cộng dồn toàn bộ chuỗi trước khi đưa vào sản xuất.

### Độ trễ chuẩn mực năm 2026

- VAD: 20–60ms
- STT từng phần (partial): 100–250ms
- LLM sinh token đầu tiên: 150–400ms
- TTS sinh gói âm thanh đầu tiên: 100–200ms
- Độ trễ truyền tải khứ hồi (RTT): 30–80ms

Độ trễ đầu-cuối từ 450–600ms được coi là trải nghiệm cao cấp (premium). Từ 800–1200ms là mức phổ biến. Bất kỳ giá trị nào vượt quá 1500ms sẽ mang lại cảm giác hội thoại bị đứt gãy, thiếu tự nhiên.

```figure
voice-pipeline
```

## Thực hành Xây dựng (Build It)

Tệp `code/main.py` hiện thực một đường ống xử lý theo khung dữ liệu mô phỏng:

- Các kiểu khung `Frame` (âm thanh, phiên âm, văn bản, âm thanh TTS, điều khiển).
- Giao diện `Processor` với phương thức `process(frame)`.
- Đường ống 5 giai đoạn (VAD → STT → LLM → TTS → transport) dưới dạng các bộ xử lý kịch bản.
- Khung hủy `cancel` thuộc luồng UPSTREAM để chứng minh khả năng ngắt lời khi người dùng chen ngang (barge-in).

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Bản ghi vết hiển thị luồng dữ liệu thông thường cùng tình huống chen ngang làm dừng ngay lập tức luồng TTS khi đang phát giữa chừng.

## Ứng dụng Thực tế (Use It)

- **Pipecat:** Khi bạn cần toàn quyền kiểm soát — bộ xử lý tùy biến, môi trường thuần Python, nhà cung cấp dịch vụ cắm-rút linh hoạt.
- **LiveKit Agents:** Dành cho các hệ thống triển khai ưu tiên WebRTC và hạ tầng thoại viễn thông.
- **Vapi / Retell:** Giải pháp tác tử giọng nói lưu trữ sẵn, không đòi hỏi vận hành hạ tầng WebRTC phức tạp.
- **OpenAI Realtime / Gemini Live:** Dành cho tương tác âm thanh trực tiếp hai chiều (sử dụng `MultimodalAgent`).

## Xuất xưởng Sản phẩm (Ship It)

Tệp `outputs/skill-voice-pipeline.md` tạo khung mẫu đường ống giọng nói theo kiến trúc Pipecat gồm VAD + STT + LLM + TTS + transport tích hợp sẵn bộ xử lý chen ngang hội thoại.

## Bài tập

1. Bổ sung một bộ quan sát số liệu (metrics observer) vào đường ống mô phỏng: đếm số khung hình mỗi giây qua từng giai đoạn. Độ trễ bị tích lũy nhiều nhất ở đâu?
2. Triển khai cơ chế STT có cổng kiểm soát độ tin cậy: nếu độ tin cậy dưới ngưỡng, tác tử phản hồi "Xin lỗi, bạn có thể nhắc lại được không?".
3. Thêm bộ nhận diện lượt thoại ngữ nghĩa: dùng quy tắc đơn giản — nếu câu phiên âm kết thúc bằng dấu "?", xác nhận hoàn tất lượt thoại.
4. Đọc tài liệu về transport của Pipecat. Thay thế transport thư viện chuẩn bằng cấu hình `SmallWebRTCTransport` (dạng khung stub).
5. Đo lường: So sánh giữa mô hình OpenAI Realtime và chuỗi nối tầng STT+LLM+TTS trên cùng một câu hỏi. Khả năng kiểm soát ở cấp độ văn bản làm tăng thêm bao nhiêu độ trễ?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Frame | "Sự kiện / Khung" | Đơn vị dữ liệu định kiểu truyền trong đường ống (audio, transcript, text, control) |
| Processor | "Tầng đường ống" | Bộ xử lý triển khai phương thức `process(frame)` |
| DOWNSTREAM | "Luồng xuôi" | Hướng từ nguồn đến đích: nhận âm thanh vào, phát tiếng nói ra |
| UPSTREAM | "Luồng phản hồi" | Hướng truyền điều khiển: hủy tác vụ, số liệu đo lường, chen ngang (barge-in) |
| VAD | "Phát hiện giọng nói" | Kỹ thuật phát hiện thời điểm người dùng bắt đầu và kết thúc nói (Voice Activity Detection) |
| Nhận diện lượt thoại ngữ nghĩa | "Kết thúc lượt thông minh" | Phán đoán dựa trên mô hình học máy xem người dùng đã dứt lời hay chưa |
| MultimodalAgent | "Tác tử âm thanh trực tiếp" | Nhận trực tiếp âm thanh và xuất trực tiếp âm thanh; không qua bước trung gian chuyển văn bản |
| VoicePipelineAgent | "Tác tử chuỗi nối tầng" | Kết nối STT + LLM + TTS; cho phép can thiệp và kiểm soát ở tầng văn bản |

## Đọc thêm

- [Tài liệu kỹ thuật Pipecat](https://docs.pipecat.ai/getting-started/introduction) — đường ống hướng khung dữ liệu, bộ xử lý và các giao thức truyền tải
- [Tài liệu kỹ thuật LiveKit Agents](https://docs.livekit.io/agents/) — hạ tầng WebRTC kết hợp các cấu trúc nguyên thủy cho giọng nói
- [Vapi](https://vapi.ai/) — nền tảng tác tử giọng nói được quản lý toàn diện
- [Retell AI](https://www.retellai.com/) — nền tảng giọng nói AI với các bài đo kiểm độ trễ chuyên sâu
