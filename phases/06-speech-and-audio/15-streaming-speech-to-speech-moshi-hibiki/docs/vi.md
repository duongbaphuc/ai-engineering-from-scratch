# Tiếng nói sang Tiếng nói Phát trực tuyến — Moshi, Hibiki và Hội thoại Song công Toàn phần

> Giai đoạn 2024-2026 đã định nghĩa lại AI giọng nói. Moshi cung cấp một mô hình duy nhất có khả năng đồng thời vừa nghe vừa nói với độ trễ chỉ 200 ms. Hibiki thực hiện dịch tiếng nói sang tiếng nói (speech-to-speech translation) theo từng đoạn âm thanh (chunk-by-chunk). Cả hai đều từ bỏ đường ống tuần tự ASR → LLM → TTS truyền thống để chuyển sang kiến trúc song công toàn phần (full-duplex) hợp nhất hoạt động trên các token của codec Mimi. Đây chính là bản thiết kế tham chiếu mới.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## Vấn đề Đặt ra

Mọi trợ lý giọng nói được xây dựng từ Bài 11 + 12 đều gặp phải một mức sàn độ trễ căn bản trong khoảng 300-500 ms: VAD kích hoạt, STT xử lý, LLM suy luận, TTS tổng hợp âm thanh. Mỗi giai đoạn đều có độ trễ tối thiểu riêng. Bạn có thể tinh chỉnh và chạy song song hóa, nhưng hình thái đường ống phân tầng luôn tạo ra một giới hạn cứng.

Moshi (Kyutai, 2024-2026) đặt ra một câu hỏi khác: điều gì sẽ xảy ra nếu không còn đường ống tuần tự nữa? Nếu một mô hình duy nhất tiếp nhận âm thanh đầu vào và phát trực tiếp âm thanh đầu ra một cách liên tục, trong đó văn bản chỉ đóng vai trò là một "độc thoại nội tâm" (inner monologue) trung gian thay vì một giai đoạn bắt buộc?

Câu trả lời là **tiếng nói sang tiếng nói song công toàn phần (full-duplex speech-to-speech)**. Độ trễ lý thuyết là 160 ms (khung hình Mimi 80 ms + độ trễ âm học 80 ms). Độ trễ thực tế là 200 ms trên một GPU L4 đơn lẻ. Con số này chỉ bằng một nửa so với những gì mà một trợ lý giọng nói dạng đường ống tối ưu nhất có thể đạt được.

## Khái niệm Cốt lõi

![Kiến trúc Moshi: hai luồng Mimi song song + văn bản độc thoại nội tâm](../assets/moshi-hibiki.svg)

### Kiến trúc của Moshi

**Đầu vào.** Hai luồng codec Mimi, cả hai đều ở mức 12,5 Hz × 8 bảng mã (codebooks):

- Luồng 1: âm thanh của người dùng (được mã hóa qua Mimi, liên tục gửi đến)
- Luồng 2: âm thanh của chính Moshi (do Moshi tự sinh ra)

**Mô hình Transformer.** Một Temporal Transformer với 7 tỷ tham số (7B) xử lý đồng thời cả hai luồng âm thanh cùng một luồng văn bản "độc thoại nội tâm". Tại mỗi bước 80 ms, mô hình sẽ:

1. Tiếp nhận các token Mimi mới nhất từ người dùng (8 bảng mã).
2. Tiếp nhận các token Mimi gần nhất của chính Moshi (8 bảng mã, ngay khi vừa được tạo ra).
3. Sinh ra token văn bản tiếp theo của Moshi (độc thoại nội tâm).
4. Sinh ra các token Mimi tiếp theo của Moshi (8 bảng mã thông qua một Depth Transformer nhỏ).

Cả ba luồng — âm thanh người dùng, âm thanh của Moshi, văn bản của Moshi — đều chạy song song. Moshi có thể nghe người dùng trong khi đang nói; có thể tự ngắt lời chính mình khi người dùng chen ngang; có thể chêm xen phản hồi đệm (back-channel, ví dụ "vâng", "ừm") mà không làm gián đoạn câu nói chính của mình.

**Depth Transformer (Transformer chiều sâu).** Trong cùng một khung hình, 8 bảng mã không được dự đoán song song — chúng có sự phụ thuộc lẫn nhau giữa các bảng mã. Một "depth transformer" nhỏ gồm 2 lớp sẽ dự đoán tuần tự các bảng mã này trong vòng 80 ms. Đây là phương pháp nhân tử hóa tiêu chuẩn cho các mô hình ngôn ngữ codec tự hồi quy (AR codec LMs, cũng được áp dụng trong VALL-E, VibeVoice).

### Vì sao văn bản độc thoại nội tâm lại hữu ích

Nếu không có văn bản tường minh, mô hình sẽ buộc phải mô hình hóa ngôn ngữ một cách ẩn (implicit) ngay trong luồng âm học của nó. Đột phá của Moshi: buộc mô hình phải phát ra các token văn bản song hành cùng âm thanh. Luồng văn bản này về bản chất chính là bản gỡ băng những gì Moshi đang nói. Điều này giúp cải thiện tính mạch lạc ngữ nghĩa, giúp việc hoán đổi phần đầu mô hình ngôn ngữ (language model head) trở nên dễ dàng hơn, và mang lại văn bản gỡ băng hoàn toàn miễn phí.

### Hibiki: dịch tiếng nói sang tiếng nói phát trực tuyến

Cùng sử dụng chung kiến trúc trên, nhưng được huấn luyện trên các cặp dữ liệu dịch thuật. Âm thanh ngôn ngữ nguồn đi vào, âm thanh ngôn ngữ đích đi ra liên tục. Hibiki-Zero (tháng 2/2026) loại bỏ nhu cầu về dữ liệu huấn luyện được căn chỉnh ở cấp độ từ — mô hình sử dụng dữ liệu cấp câu kết hợp học tăng cường GRPO (Group Relative Policy Optimization) để tối ưu hóa độ trễ.

Ban đầu hỗ trợ 4 cặp ngôn ngữ; có thể thích ứng với một ngôn ngữ mới chỉ với khoảng 1.000 giờ dữ liệu.

### Hệ sinh thái công nghệ mở rộng của Kyutai (2026)

- **Moshi** — hội thoại song công toàn phần (tiếng Pháp trước, tiếng Anh được hỗ trợ rất tốt)
- **Hibiki / Hibiki-Zero** — dịch thuật tiếng nói đồng thời (simultaneous speech translation)
- **Kyutai STT** — nhận dạng tiếng nói phát trực tuyến (độ trễ nhìn trước 500 ms hoặc 2,5 s)
- **Kyutai Pocket TTS** — mô hình TTS 100M tham số chạy được trên CPU (tháng 1/2026)
- **Unmute** — đường ống hoàn chỉnh kết hợp các thành phần trên triển khai trên máy chủ công cộng

Thông lượng trên một GPU L40S: 64 phiên hoạt động đồng thời ở tốc độ 3× thời gian thực.

### Sesame CSM — người anh em họ

Sesame CSM (2025) áp dụng ý tưởng tương tự — sử dụng phần xương sống Llama-3 với một phần đầu codec Mimi. Tuy nhiên CSM chỉ hoạt động theo một chiều (nhận ngữ cảnh + văn bản, sinh ra tiếng nói) thay vì song công toàn phần. Đây là mô hình TTS có "sự hiện diện giọng nói" (voice presence) tốt nhất trên thị trường, dù không hoàn toàn tương đương với năng lực song công toàn phần của Moshi.

### Các thông số hiệu năng năm 2026

| Mô hình | Độ trễ | Trường hợp sử dụng | Giấy phép |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | hội thoại song công tiếng Anh / tiếng Pháp | CC-BY 4.0 |
| Hibiki | tốc độ khung hình 12,5 Hz | dịch trực tuyến Pháp ↔ Anh | CC-BY 4.0 |
| Hibiki-Zero | tương tự | 5 cặp ngôn ngữ, không cần dữ liệu căn chỉnh | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | TTS có điều kiện ngữ cảnh | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | đóng, OpenAI API | thương mại |
| Gemini 2.5 Live | ~350 ms | đóng, Google API | thương mại |

```figure
sp-fullduplex
```

## Thực hành Xây dựng

### Bước 1: Giao diện kết nối

Moshi cung cấp một máy chủ WebSocket tiếp nhận các đoạn âm thanh 80 ms được mã hóa bằng Mimi và trả về các đoạn âm thanh 80 ms cũng được mã hóa bằng Mimi. Cả hai chiều. Liên tục.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Bước 2: Vòng lặp song công toàn phần

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Cả hai chiều truyền phát chạy đồng thời. Python asyncio hoặc Rust futures là phương thức truyền tải tiêu chuẩn.

### Bước 3: Mục tiêu huấn luyện (Về mặt khái niệm)

Đối với mỗi khung hình 80 ms ở thời điểm `t`:

- Đầu vào: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Dự đoán: `moshi_text[t]`, sau đó đến `moshi_mimi[t, codebook_0..7]`

Văn bản được dự đoán trước âm thanh (độc thoại nội tâm); âm thanh được dự đoán tuần tự theo từng bảng mã bên trong depth transformer.

### Bước 4: Điểm mạnh và giới hạn của Moshi

Moshi vượt trội ở:

- Độ trễ đầu-cuối dưới 250 ms trên phần cứng phổ thông giá rẻ.
- Khả năng đệm tiếng tự nhiên (back-channels) và xử lý chen ngang mượt mà.
- Không cần mã kết nối đường ống phức tạp.

Moshi chưa tối ưu ở:

- Gọi công cụ (Tool calling — mô hình không được huấn luyện cho tác vụ này; bạn cần một nhánh LLM riêng).
- Khả năng suy luận dài (Moshi là mô hình hội thoại cỡ ~8B tham số, không phải Claude hay GPT-4).
- Độ chính xác thực tế về các chủ đề chuyên sâu, ngách.
- Đa số các ứng dụng doanh nghiệp thực tế (vào năm 2026 vẫn sử dụng kiến trúc đường ống).

## Ứng dụng Thực tế

| Tình huống | Lựa chọn |
|-----------|------|
| Người bạn đồng hành giọng nói có độ trễ thấp nhất | Moshi |
| Cuộc gọi dịch thuật trực tiếp | Hibiki |
| Bản thử nghiệm / nghiên cứu giọng nói | Moshi, CSM |
| Tác nhân doanh nghiệp có gọi công cụ | Kiến trúc đường ống (Bài 12), không dùng Moshi |
| TTS với giọng tùy chỉnh theo ngữ cảnh | Sesame CSM |
| Tiếng nói sang tiếng nói, đa ngôn ngữ bất kỳ | GPT-4o Realtime hoặc Gemini 2.5 Live (thương mại) |

## Cạm bẫy Thường gặp

- **Khả năng gọi công cụ hạn chế.** Moshi là một mô hình hội thoại, không phải là một framework xây dựng tác tử (agent framework). Cần kết hợp với đường ống nếu muốn dùng công cụ.
- **Ràng buộc giọng nói cụ thể.** Moshi sử dụng một tính cách / chất giọng đơn lẻ đã được huấn luyện sẵn; nhân bản giọng nói đòi hỏi một lượt huấn luyện riêng biệt.
- **Độ bao phủ ngôn ngữ.** Tiếng Pháp + tiếng Anh rất xuất sắc; các ngôn ngữ khác còn hạn chế. Hibiki-Zero giúp cải thiện điều này, nhưng bạn vẫn cần dữ liệu huấn luyện.
- **Chi phí tài nguyên.** Một phiên làm việc đầy đủ của Moshi chiếm dụng liên tục một khe GPU; đây không phải là mô hình triển khai chia sẻ tài nguyên (multi-tenant) giá rẻ.

## Đóng gói Sản phẩm

Lưu thành `outputs/skill-duplex-pipeline.md`. Lựa chọn giữa kiến trúc đường ống truyền thống và kiến trúc song công toàn phần cho khối lượng công việc của trợ lý giọng nói, kèm theo lý do cụ thể.

## Bài tập Thực hành

1. **Dễ.** Chạy `code/main.py`. Tệp này mô phỏng tượng trưng kiến trúc hai luồng song song kết hợp độc thoại nội tâm.
2. **Trung bình.** Tải Moshi từ HuggingFace, khởi chạy máy chủ, kiểm thử một đoạn hội thoại thực tế. Đo độ trễ thời gian thực từ lúc người dùng dứt lời đến khi Moshi bắt đầu phản hồi.
3. **Khó.** Lấy tác tử dạng đường ống từ Bài 12 và so sánh độ trễ P50 với Moshi trên 20 câu nói thử nghiệm tương đương. Viết báo cáo giải thích những trường hợp mà kiến trúc đường ống vẫn vượt trội về mặt thiết kế.

## Thuật ngữ Then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Song công toàn phần (Full-duplex) | Vừa nghe vừa nói đồng thời | Hai luồng âm thanh cùng hoạt động đồng thời trên cùng một mô hình duy nhất. |
| Độc thoại nội tâm (Inner monologue) | Luồng văn bản của mô hình | Moshi xuất ra các token văn bản song hành cùng đầu ra âm thanh. |
| Depth transformer | Bộ dự đoán giữa các bảng mã | Mô hình transformer nhỏ dự đoán 8 bảng mã bên trong một khung hình 80 ms. |
| Mimi | Codec của Kyutai | 12,5 Hz × 8 bảng mã; phân tách ngữ nghĩa + âm học; vận hành Moshi. |
| S2S phát trực tuyến (Streaming S2S) | Âm thanh → âm thanh trực tiếp | Dịch thuật/hội thoại theo từng đoạn nhỏ (chunk), không qua các giai đoạn đường ống. |
| Phản hồi đệm (Back-channeling) | Phản ứng chêm xen như "ừm", "vâng" | Moshi có thể phát ra các tín hiệu xác nhận ngắn mà không làm đứt đoạn lượt nói. |

## Tài liệu Tham khảo

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — bài báo gốc.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — dịch trực tuyến không cần dữ liệu căn chỉnh.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — đặc tả CSM.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — mã nguồn cài đặt + máy chủ.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — giải pháp thương mại mã nguồn đóng tương đương.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — framework STT/TTS nền tảng bên dưới.
