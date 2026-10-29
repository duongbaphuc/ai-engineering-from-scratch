# Đồ án Tốt nghiệp 03 — Trợ lý Giọng nói Thời gian thực (ASR đến LLM đến TTS)

> Một tác tử giọng nói mang lại cảm giác hội thoại tự nhiên cần đạt độ trễ đầu-cuối (end-to-end latency) dưới 800ms, nhận biết chuẩn xác thời điểm người dùng ngừng nói, xử lý mượt mà hiện tượng chen ngang/ngắt lời (barge-in), và có thể gọi công cụ bên ngoài mà không làm nghẽn luồng âm thanh. Retell, Vapi, LiveKit Agents và Pipecat đều chạm tới chuẩn mực này vào năm 2026. Họ hiện thực hóa điều đó bằng cùng một hình thái kiến trúc: nhận dạng tiếng nói tự động dạng luồng (streaming ASR), bộ phát hiện lượt thoại (turn-detector), mô hình ngôn ngữ lớn dạng luồng (streaming LLM), và tổng hợp tiếng nói dạng luồng (streaming TTS), tất cả được kết nối qua giao thức WebRTC với giới hạn ngân sách độ trễ khắt khe tại từng chặng truyền tải. Hãy tự tay xây dựng một trợ lý hoàn chỉnh, đo lường các chỉ số WER, MOS, tỷ lệ cắt lời sai (false-cutoff rate), và kiểm thử độ bền vững khi đường truyền mạng bị mất gói tin (packet loss).

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (Xử lý tiếng nói và âm thanh), Phase 7 (Mô hình Transformer), Phase 11 (Kỹ thuật LLM), Phase 13 (Công cụ), Phase 14 (Tác tử), Phase 17 (Hạ tầng)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** ~30 giờ

## Bài toán (Problem)

Trợ lý giọng nói là phân khúc trải nghiệm người dùng (UX) AI phát triển nhanh nhất trong giai đoạn 2025–2026. Rào cản kỹ thuật liên tục giảm xuống qua từng quý. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0 và Pipecat 0.0.70 đều đưa mốc phát âm thanh đầu tiên (first-audio-out) dưới 800ms vào phạm vi khả thi. Tuy nhiên, thách thức không chỉ nằm ở độ trễ. Nó nằm ở cảm xúc tương tác: không cướp lời người dùng, không bị ngắt quãng vô cớ, khôi phục trạng thái tức thì sau khi bị ngắt giữa câu, gọi công cụ giữa cuộc hội thoại mà không làm đứng hình âm thanh, và duy trì kết nối ổn định trên mạng di động chập chờn.

Bạn không thể đạt được điều này bằng cách ghép nối ba lệnh gọi REST API rời rạc. Kiến trúc này bắt buộc phải là một đường ống truyền phát luồng (pipelined streaming) xuyên suốt từ đầu đến cuối. Khi bắt tay xây dựng, bạn sẽ thấy rõ các dạng lỗi kinh điển: bộ phát hiện hoạt động giọng nói (Voice Activity Detection - VAD) vốn tinh chỉnh cho đường dây điện thoại lại kích hoạt nhầm bởi tiếng TV nền, bộ phát hiện lượt thoại chờ đợi dấu câu vốn không bao giờ xuất hiện trong văn bản nói, hoặc bộ tổng hợp TTS giữ bộ đệm tới 400ms trước khi phát ra âm thanh. Mục tiêu của đồ án tốt nghiệp này là giải quyết từng điểm nghẽn dưới tải thực tế và công bố bản báo cáo chuẩn hóa về độ trễ và chất lượng.

## Khái niệm (Concept)

Đường ống bao gồm năm giai đoạn truyền luồng:
1. **Âm thanh đầu vào (Audio in):** WebRTC từ trình duyệt hoặc mạng điện thoại PSTN.
2. **ASR:** Truyền luồng văn bản giải mã từng phần (partial transcripts) từ Deepgram Nova-3 hoặc faster-whisper.
3. **Phát hiện lượt thoại (Turn detection):** Kết hợp VAD cùng một mô hình phát hiện lượt thoại nhỏ gọn, phân tích các bản chép từng phần để tìm dấu hiệu hoàn tất câu nói.
4. **LLM:** Truyền luồng các token phản hồi ngay khi lượt thoại được xác định là kết thúc.
5. **TTS:** Truyền luồng âm thanh đầu ra trong vòng ~200ms kể từ token LLM đầu tiên.

Ba vấn đề phối hợp liên thông phức tạp:
- **Xử lý ngắt lời (Barge-in):** Khi người dùng bắt đầu nói trong lúc tác tử đang phát âm thanh, bộ TTS phải lập tức hủy phát và bộ ASR phải bắt lấy giọng nói mới ngay lập tức.
- **Sử dụng công cụ (Tool use):** Các lệnh gọi hàm giữa hội thoại (tra cứu thời tiết, lịch hẹn) phải chạy trên một kênh phụ (side channel) mà không làm ngắt quãng âm thanh; tác tử sẽ chèn trước một từ đệm xác nhận (filler token như "xin chờ một chút...") nếu độ trễ vượt quá 300ms.
- **Kiểm soát áp lực ngược (Backpressure):** Khi mạng bị rớt gói tin, bản chép từng phần được giữ lại trong bộ đệm, VAD nâng ngưỡng phát hiện tiếng nói, và tác tử tránh nói đè lên một thông điệp chưa được xác nhận.

Thước đo định lượng là yêu cầu bắt buộc: Tỷ lệ lỗi từ (Word Error Rate - WER) dưới 8% trên benchmark Hamming VAD ở tỷ số SNR 15 dB. Độ trễ phát âm thanh đầu tiên p50 dưới 800ms trên 100 cuộc gọi đo kiểm. Tỷ lệ cắt lời sai (false-cutoff rate) dưới 3%. Điểm chất lượng giọng nói trung bình (Mean Opinion Score - MOS) trên 4.2 cho TTS. Duy trì ổn định 50 cuộc gọi đồng thời trên một máy chủ g5.xlarge duy nhất. Những con số này chính là tiêu chuẩn nghiệm thu sản phẩm.

## Kiến trúc (Architecture)

```text
trình duyệt / Twilio PSTN
        |
        v
    WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (hoặc Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5       turn-detector      công cụ kênh phụ
(Deepgram         (Silero)         (LiveKit)        (thời tiết,
 Nova-3 /        cổng tiếng nói   chấm điểm hoàn     lịch hẹn)
 Whisper-v3)      mỗi 20ms        thành từng phần       |
   |                   |              |                 |
   +--------+----------+--------------+                 |
            v                                           |
       LLM (streaming) <--------------------------------+
     GPT-4o-realtime / Gemini 2.5 Flash /
     Claude Haiku 4.5 tầng bậc
            |
            v
       TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     âm thanh trả về người gọi
            |
            v
   Dấu vết giọng nói OpenTelemetry -> Langfuse
```

## Công nghệ sử dụng (Stack)

- Truyền tải (Transport): LiveKit Agents 1.0 (WebRTC) kết hợp cổng Twilio PSTN; hoặc Pipecat 0.0.70 làm framework thay thế
- ASR: Deepgram Nova-3 (truyền luồng, đoạn giải mã đầu tiên dưới 300ms) hoặc faster-whisper Whisper-v3-turbo tự lưu trữ trên GPU
- VAD: Silero VAD v5 kết hợp bộ phát hiện lượt thoại LiveKit (mô hình Transformer nhỏ phân tích bản chép từng phần)
- LLM: OpenAI GPT-4o-realtime để tích hợp chặt chẽ, Gemini 2.5 Flash Live, hoặc Claude Haiku 4.5 theo kiến trúc tầng bậc (streaming completion với luồng âm thanh riêng biệt)
- TTS: Cartesia Sonic-2 (thời gian trả byte đầu tiên thấp nhất), ElevenLabs Flash v3, hoặc Orpheus nguồn mở tự lưu trữ
- Công cụ: FastMCP kênh phụ cho thời tiết/lịch hẹn/đặt chỗ; tác tử tự động chèn từ đệm nếu công cụ xử lý lâu hơn 300ms
- Khả năng quan sát: OpenTelemetry voice spans, dấu vết cuộc gọi trên Langfuse hỗ trợ phát lại âm thanh
- Triển khai: Một phiên bản GPU g5.xlarge (24GB VRAM) cho mô hình tự lưu trữ Whisper + Orpheus; hoặc dùng API đám mây để đạt độ trễ tối thiểu

```figure
ce-voice-latency
```

## Các bước triển khai (Build It)

1. **Phiên WebRTC.** Khởi tạo phòng LiveKit và xây dựng client web truyền luồng âm thanh từ microphone. Trên máy chủ, gắn worker tác tử tham gia vào phòng đó.

2. **Truyền luồng ASR.** Đẩy các khung âm thanh PCM 20ms vào Deepgram Nova-3 (hoặc faster-whisper trên GPU). Đăng ký lắng nghe các bản chép từng phần (partials) và bản chép cuối (finals). Ghi nhật ký độ trễ của từng phần.

3. **VAD và bộ phát hiện lượt thoại.** Chạy Silero VAD v5 trên luồng khung âm thanh. Khi xuất hiện sự kiện kết thúc lời nói (speech-end), kích hoạt bộ phát hiện lượt thoại LiveKit dựa trên bản chép từng phần mới nhất. Chỉ xác nhận "lượt thoại hoàn tất" khi VAD ghi nhận khoảng lặng kéo dài 500ms VÀ điểm hoàn thành câu của turn-detector vượt ngưỡng 0.6.

4. **Luồng sinh LLM.** Khi lượt thoại hoàn tất, khởi chạy lệnh gọi LLM với toàn bộ ngữ cảnh hội thoại cộng thêm bản chép cuối cùng. Truyền phát các token đầu ra. Ngay khi có token đầu tiên, chuyển tiếp ngay cho bộ TTS.

5. **Truyền luồng TTS.** Cartesia Sonic-2 truyền các phân đoạn âm thanh trả về. Phân đoạn âm thanh đầu tiên phải rời máy chủ trong vòng 200ms kể từ token LLM đầu tiên. Đẩy các đoạn âm thanh vào phòng LiveKit; client phát qua bộ đệm rung pha (jitter buffer) của WebRTC.

6. **Cơ chế ngắt lời (Barge-in).** Khi VAD phát hiện tiếng nói mới từ người dùng trong lúc TTS đang phát, lập tức hủy luồng TTS, loại bỏ các token LLM còn lại và tái kích hoạt lại trạng thái ASR. Phát đi một span `tts_canceled` lên hệ thống giám sát.

7. **Kênh công cụ phụ.** Đăng ký tra cứu thời tiết và lịch hẹn dưới dạng công cụ gọi hàm (function-calling tools). Khi được kích hoạt, thực thi lệnh gọi song song; nếu chưa có kết quả sau 300ms, yêu cầu LLM phát ra từ đệm "dạ xin chờ em kiểm tra một chút ạ"; sau đó tiếp tục phản hồi khi công cụ trả dữ liệu.

8. **Khung đánh giá hiệu năng.** Ghi âm 100 cuộc gọi thử nghiệm. Tính toán WER (so sánh với văn bản đối chứng), tỷ lệ cắt lời sai (TTS bị hủy khi người dùng mới nói nửa câu), p50 độ trễ phát âm thanh đầu tiên, điểm MOS của TTS (bằng con người hoặc công cụ tự động NISQA), và bài kiểm tra rớt gói tin (bỏ ngẫu nhiên 3% gói tin).

9. **Kiểm thử tải (Load test).** Giả lập 50 cuộc gọi đồng thời lên một máy chủ g5.xlarge bằng bot gọi nhân tạo. Đo lường độ trễ p95 phát âm thanh đầu tiên khi chịu tải liên tục.

## Ứng dụng thực tế (Use It)

```text
caller: "thời tiết tokyo ngày mai thế nào"
[asr  ] partial @280ms: "thời tiết"
[asr  ] partial @540ms: "thời tiết tokyo ngày mai"
[turn ] điểm hoàn thành 0.82 tại @820ms; chốt lượt thoại
[llm  ] phát lệnh gọi công cụ @880ms
[tool ] weather.tokyo tomorrow -> 20/11 độ C có mây rải rác @940ms
[llm  ] token đầu tiên @990ms
[tts  ] phát âm thanh đầu tiên @1040ms: "Ngày mai ở Tokyo thời tiết sẽ có mây rải rác..."
độ trễ lượt thoại: 1040ms từ lúc người dùng ngừng nói -> âm thanh phát ra
```

## Triển khai thành phẩm (Ship It)

Sản phẩm kỹ năng hoàn chỉnh được lưu tại `outputs/skill-voice-agent.md`. Khi nhận vào một miền ứng dụng cụ thể (chăm sóc khách hàng, đặt lịch, hoặc quầy thông tin tự động), nó sẽ thiết lập một tác tử LiveKit với đường ống ASR/VAD/LLM/TTS được tinh chỉnh đạt các tiêu chuẩn đo lường. Bảng tiêu chí đánh giá:

| Trọng số | Tiêu chí | Phương thức đo lường |
|:-:|---|---|
| 25 | Độ trễ đầu-cuối | p50 phát âm thanh đầu tiên dưới 800ms trên 100 cuộc gọi thực tế |
| 20 | Chất lượng điều phối lượt thoại | Tỷ lệ cắt lời sai dưới 3% trên tập kiểm thử Hamming VAD |
| 20 | Độ chuẩn xác khi dùng công cụ | Lệnh gọi công cụ giữa hội thoại trả về đúng kết quả mà không làm nghẽn âm thanh |
| 20 | Độ tin cậy khi mạng mất gói tin | Độ ổn định của WER và lượt thoại khi chủ động rớt 3% gói tin mạng |
| 15 | Tính toàn diện của bộ kiểm thử | Các phép đo có khả năng tái lập hoàn toàn kèm cấu hình mở |
| **100** | | |

## Bài tập mở rộng (Exercises)

1. Hoán đổi Deepgram Nova-3 sang faster-whisper v3 turbo tự lưu trữ trên g5.xlarge. Đo lường khoảng cách về độ trễ và WER. Phân tích rõ thời điểm nào các quyết định CPU và GPU trở nên quan trọng.

2. Xây dựng chính sách phân xử ngắt lời (interruption-arbitration policy): Tác tử nên phản ứng ra sao khi người dùng nói chen ngang đúng lúc đang thực thi gọi công cụ? So sánh ba chính sách: hủy bỏ ngay lập tức (hard cancel), chạy xong công cụ rồi mới dừng (finish-tool-then-stop), và xếp hàng cho lượt tiếp theo (queue next turn).

3. Thực hiện kiểm thử bộ phát hiện lượt thoại với kịch bản đối kháng: Cho người dùng cố tình ngập ngừng lâu ở giữa câu. Tinh chỉnh ngưỡng im lặng VAD và ngưỡng điểm turn-detector để đạt tỷ lệ cắt lời sai thấp nhất mà không để độ trễ vượt quá 900ms.

4. Triển khai cùng một tác tử đó trên mạng điện thoại cố định/di động PSTN qua Twilio. So sánh độ trễ phát âm thanh đầu tiên giữa PSTN và WebRTC. Giải thích sự khác biệt về bộ đệm rung pha và chuẩn nén âm thanh (codec).

5. Bổ sung nhận diện hoạt động giọng nói cho các ngôn ngữ khác (tiếng Nhật, tiếng Tây Ban Nha). Đo lường tỷ lệ kích hoạt sai của Silero VAD v5 so với các phiên bản được tinh chỉnh chuyên biệt theo từng ngôn ngữ.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Turn detection (Phát hiện lượt thoại) | "Kết thúc câu nói" | Bộ phân loại dựa trên khoảng lặng của VAD và bản chép từng phần để kết luận người dùng đã nói xong |
| Barge-in (Xử lý ngắt lời) | "Bắt chen ngang" | Hành động hủy phát luồng TTS ngay giữa chừng khi VAD phát hiện người dùng bắt đầu nói mới |
| First-audio-out (Âm thanh đầu ra đầu tiên) | "Độ trễ phản hồi" | Khoảng thời gian từ lúc người dùng ngừng phát âm đến khi gói âm thanh đầu tiên rời khỏi máy chủ |
| VAD | "Cổng nhận diện giọng nói" | Mô hình phân loại từng khung âm thanh là tiếng nói hay khoảng lặng; Silero VAD v5 là chuẩn mực 2026 |
| Jitter buffer (Bộ đệm rung pha) | "Làm mượt âm thanh" | Bộ đệm phía máy khách giữ các gói âm thanh trong khoảnh khắc ngắn để triệt tiêu biến thiên mạng |
| Filler (Từ đệm âm thanh) | "Âm xác nhận" | Cụm từ ngắn mà tác tử phát ra để lấp khoảng lặng khi công cụ bên ngoài phản hồi chậm |
| MOS (Điểm đánh giá trung bình) | "Chất lượng âm thanh" | Thước đo cảm nhận chất lượng tiếng nói; NISQA là công cụ đánh giá tự động thay thế chuyên gia |

## Tài liệu đọc thêm (Further Reading)

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — Framework tác tử WebRTC tham chiếu tiêu chuẩn
- [Pipecat](https://github.com/pipecat-ai/pipecat) — Framework luồng tác tử ưu tiên Python
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — Tài liệu tham chiếu cho mô hình giọng nói tích hợp
- [Tài liệu Deepgram Nova-3](https://developers.deepgram.com/docs) — Tài liệu tham chiếu ASR dạng luồng
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — Mô hình VAD tham chiếu phổ biến
- [Cartesia Sonic-2](https://docs.cartesia.ai) — Nền tảng TTS độ trễ thấp hàng đầu
- [Kiến trúc Retell AI](https://docs.retellai.com) — Kiến trúc tác tử thoại trong môi trường thương mại
- [Vapi.ai production stack](https://docs.vapi.ai) — Tham chiếu ngăn xếp sản xuất từ nền tảng Vapi
