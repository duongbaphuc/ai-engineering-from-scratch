# Môi trường Thực thi trong Môi trường Sản xuất: Hàng đợi, Sự kiện, Cron (Production Runtimes: Queue, Event, Cron)

> Các tác tử trong môi trường sản xuất vận hành theo sáu hình thái môi trường thực thi (runtime shapes): yêu cầu - phản hồi (request-response), truyền phát trực tiếp (streaming), thực thi bền bỉ (durable execution), chạy nền dựa trên hàng đợi (queue-based background), hướng sự kiện (event-driven) và lập lịch định kỳ (scheduled / cron). Hãy lựa chọn đúng hình thái môi trường thực thi trước khi quyết định chọn framework. Khả năng quan sát (observability) là thành phần chịu lực cốt lõi trên mọi hình thái.

**Loại bài học:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Xử lý Giọng nói - Voice)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Nêu tên sáu hình thái môi trường thực thi trong môi trường sản xuất và ánh xạ từng hình thái vào một mẫu framework / sản phẩm tương ứng.
- Giải thích lý do tại sao thực thi bền bỉ (durable execution trong LangGraph) lại đóng vai trò sống còn đối với các nhiệm vụ dài hạn (long-horizon tasks).
- Mô tả môi trường thực thi hướng sự kiện (event-driven runtime) và trường hợp phù hợp để dùng Claude Managed Agents.
- Giải thích luận điểm "khả năng quan sát là thành phần chịu lực cốt lõi" đối với các tác tử đa bước.

## Vấn đề Đặt ra

Các tác tử khi đưa vào môi trường sản xuất thường đổ vỡ theo những cách mà sổ tay Jupyter (Jupyter notebook) không bao giờ bộc lộ: quá thời gian chờ mạng (network timeout) ở bước thứ 37, người dùng cúp máy giữa chừng trong một cuộc gọi thoại, tác vụ cron bị hủy khi máy chủ khởi động lại, hoặc tiến trình chạy nền bị tràn bộ nhớ (out-of-memory). Hình thái môi trường thực thi chính là yếu tố quyết định hệ thống của bạn có thể vượt qua và phục hồi sau những sự cố này hay không.

## Khái niệm

### Yêu cầu - Phản hồi (Request-response)

- Giao thức HTTP đồng bộ. Người dùng phải chờ đợi cho đến khi tác vụ hoàn thành.
- Chỉ khả thi đối với các tác vụ ngắn (<30 giây).
- Ngăn xếp công nghệ tiêu biểu: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Khả năng quan sát: nhật ký truy cập HTTP chuẩn + các span theo dõi của OpenTelemetry (OTel).

### Truyền phát trực tiếp (Streaming)

- Sử dụng SSE (Server-Sent Events) hoặc WebSocket để trả về kết quả lũy tiến theo thời gian thực.
- LiveKit mở rộng mô hình này sang WebRTC cho các tác vụ thoại và video (Bài 22).
- Ngăn xếp công nghệ: bất kỳ framework nào hỗ trợ streaming kết hợp giao diện người dùng có khả năng xử lý SSE/WS.
- Khả năng quan sát: thời gian phát từng đoạn (per-chunk timing), độ trễ nhận token đầu tiên (first-token latency), độ trễ đuôi (tail latency).

### Thực thi bền bỉ (Durable execution)

- Trạng thái được lưu điểm kiểm tra (checkpointed) ngay sau mỗi bước; tự động phục hồi và chạy tiếp khi xảy ra lỗi.
- Mô hình tác tử diễn viên (actor model) của AutoGen v0.4 giúp cô lập hoàn toàn sự cố ở một tác tử duy nhất (Bài 14).
- Điểm khác biệt cốt lõi của LangGraph (Bài 13).
- Bắt buộc phải có khi số bước thực thi không thể dự đoán trước và chi phí khôi phục lại từ đầu là quá đắt đỏ.

### Dựa trên hàng đợi / Chạy nền (Queue-based / background)

- Tác vụ được đẩy vào hàng đợi (queue), các tiến trình thợ (workers) tiếp nhận xử lý, kết quả được gửi trả thông qua webhook hoặc mô hình xuất bản/đăng ký (pub/sub).
- Yếu tố sống còn đối với các tác tử xử lý tác vụ dài hạn (hàng chục đến hàng trăm bước mỗi nhiệm vụ, theo báo cáo công bố computer use của Anthropic).
- Ngăn xếp công nghệ: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), hoặc hệ thống tùy chỉnh.
- Khả năng quan sát: độ sâu hàng đợi (queue depth), phân bổ độ trễ theo từng tác vụ, kích thước hàng đợi thư chết (dead-letter queue - DLQ).

### Hướng sự kiện (Event-driven)

- Các tác tử đăng ký lắng nghe các bộ kích hoạt (triggers): có email mới gửi đến, có Pull Request mới được mở, hoặc một lịch trình định kỳ phát tín hiệu.
- Claude Managed Agents hỗ trợ mô hình này ngay từ đầu (Bài 17).
- CrewAI Flows (Bài 15) cấu trúc hóa các luồng công việc tất định dựa trên sự kiện.
- Khả năng quan sát: nguồn phát sự kiện, độ trễ từ khi phát sinh sự kiện đến khi bắt đầu xử lý, độ trễ nội tại của tác tử.

### Lập lịch định kỳ (Scheduled / Cron)

- Các tác tử có dạng tác vụ cron chạy định kỳ theo thời gian biểu.
- Kết hợp với cơ chế thực thi bền bỉ để một chu kỳ chạy đêm bị lỗi có thể tự động tiếp tục ở nhịp kích hoạt kế tiếp.
- Ngăn xếp công nghệ: Kubernetes CronJob kết hợp với một framework bền bỉ; các dịch vụ lưu trữ đám mây (Render cron, Vercel cron).

### Các mẫu triển khai thực tế năm 2026

- **CrewAI Flows** cho các hệ thống sản xuất hướng sự kiện.
- **Agno** kết hợp FastAPI không trạng thái (stateless) cho các vi dịch vụ Python.
- **Mastra** với các bộ điều hợp máy chủ (Express, Hono, Fastify, Koa) cho kiến trúc nhúng trực tiếp.
- **Pipecat Cloud / LiveKit Cloud** cho các dịch vụ thoại được quản lý toàn diện (Bài 22).
- **Claude Managed Agents** cho các quy trình không đồng bộ dài hạn chạy trên nền tảng lưu trữ đám mây của nhà cung cấp.

### Khả năng quan sát là thành phần chịu lực cốt lõi

Nếu thiếu các span theo dõi chuẩn GenAI của OpenTelemetry (Bài 23) kết hợp với một backend như Langfuse/Phoenix/Opik (Bài 24), bạn hoàn toàn không thể gỡ lỗi một tác tử đa bước bị sập ở bước thứ 40. Đây không phải là tùy chọn bổ sung cho vui trên môi trường sản xuất. Đó là ranh giới giữa việc "chúng tôi gỡ lỗi và khắc phục trong vài phút" với "chúng tôi phải chạy lại toàn bộ từ đầu và hy vọng có thêm chút nhật ký để đọc".

### Những điểm đổ vỡ thường gặp trong môi trường sản xuất

- **Chọn sai hình thái môi trường thực thi:** Chọn kiểu yêu cầu - phản hồi cho một tác vụ mất tới 5 phút. Người dùng cúp máy; các worker bị dồn ứ tắc nghẽn; các cơ chế thử lại (retry) khuếch đại tải hệ thống.
- **Không có hàng đợi thư chết (DLQ):** Sử dụng hàng đợi nhưng không có cơ chế lưu trữ thư chết. Các tác vụ bị lỗi sẽ biến mất không dấu vết.
- **Tác vụ nền mù mịt (Opaque background work):** Tác tử chạy nền nhưng không xuất dữ liệu dấu vết thực thi (trace export). Mọi lỗi lầm đều hoàn toàn vô hình cho đến khi người dùng bức xúc khiếu nại.
- **Bỏ qua trạng thái bền bỉ:** Mọi lượt chạy kéo dài quá 30 giây mà bạn không thể chấp nhận rủi ro phải khởi động lại từ đầu đều bắt buộc phải dùng cơ chế thực thi bền bỉ.

```figure
wb-runtime-shapes
```

## Tự Xây dựng (Build It)

`code/main.py` cung cấp bản trình diễn đa hình thái bằng thư viện chuẩn (stdlib):

- Điểm cuối yêu cầu - phản hồi (hàm thông thường).
- Bộ xử lý truyền phát trực tiếp (hàm tạo - generator).
- Worker dựa trên hàng đợi tích hợp hàng đợi thư chết (DLQ).
- Sổ đăng ký bộ kích hoạt sự kiện.
- Bộ lập lịch định kỳ dạng cron.

Chạy thử mã nguồn:

```bash
python3 code/main.py
```

Kết quả: năm dấu vết thực thi thể hiện hành vi của từng hình thái trên cùng một tác vụ. Cùng một logic tác tử nhưng được bọc trong các lớp vỏ ngoài khác nhau. Thực thi bền bỉ (hình thái thứ sáu) đã được nghiên cứu chuyên sâu trong Bài 13 với cơ chế lưu điểm kiểm tra của LangGraph.

## Ứng dụng Thực tế (Use It)

- **Yêu cầu - phản hồi (Request-response)** cho trải nghiệm người dùng dạng trò chuyện nhanh.
- **Truyền phát trực tiếp (Streaming)** cho các phản hồi hiển thị lũy tiến mượt mà.
- **Thực thi bền bỉ (Durable)** cho các nhiệm vụ quy mô dài hạn.
- **Hàng đợi (Queue)** cho các tác vụ xử lý hàng loạt / bất đồng bộ / chạy dài.
- **Hướng sự kiện (Event)** phục vụ tính phản ứng nhạy bén của tác tử trước các biến động bên ngoài.
- **Lập lịch (Cron)** cho các công việc quản trị định kỳ (hợp nhất bộ nhớ, chạy đánh giá hệ thống, xuất báo cáo chi phí).

## Chuyển giao Đóng gói (Ship It)

`outputs/skill-runtime-shape.md` hỗ trợ lựa chọn hình thái môi trường thực thi tối ưu cho tác vụ và kết nối toàn bộ các yêu cầu về khả năng quan sát tương ứng.

## Bài tập Thực hành

1. Chuyển đổi vòng lặp ReAct từ Bài 01 của bạn sang toàn bộ sáu hình thái môi trường thực thi trong ngăn xếp công nghệ của bạn. Hình thái nào phù hợp nhất với từng bề mặt sản phẩm?
2. Bổ sung một hàng đợi thư chết (DLQ) vào bản mô phỏng hàng đợi. Giả lập tỷ lệ lỗi tác vụ 10%; hiển thị kích thước tăng trưởng của DLQ.
3. Viết một tác tử đánh giá được kích hoạt bằng cron chạy hàng đêm trên 20 dấu vết thực thi hàng đầu trong ngày.
4. Cài đặt cơ chế truyền phát có kiểm soát áp lực ngược (backpressure): nếu phía máy khách nhận dữ liệu chậm, hãy tạm dừng tác tử. Cơ chế này tương tác như thế nào với ngân sách số lượt tương tác (turn budget)?
5. Đọc tài liệu Claude Managed Agents. Trong trường hợp nào bạn sẽ quyết định chuyển một tác tử dài hạn tự lưu trữ (self-hosted) sang giải pháp được quản lý trọn gói (managed)?

## Thuật ngữ Then chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|---|---|---|
| Yêu cầu - phản hồi (Request-response) | "Đồng bộ (Synchronous)" | Người dùng phải chờ; chỉ áp dụng cho tác vụ ngắn |
| Truyền phát trực tiếp (Streaming) | "SSE / WS" | Đầu ra trả về lũy tiến; trải nghiệm tốt hơn; đo được độ trễ từng mảnh |
| Thực thi bền bỉ (Durable execution) | "Khôi phục sau sự cố" | Trạng thái có điểm kiểm tra; tiếp tục chạy từ bước bị lỗi gần nhất |
| Dựa trên hàng đợi (Queue-based) | "Tác vụ chạy nền" | Mô hình Nhà sản xuất / Nhóm thợ xử lý / Hàng đợi thư chết DLQ |
| Hướng sự kiện (Event-driven) | "Kích hoạt theo sự kiện" | Tác tử phản ứng tự động trước các sự kiện từ môi trường ngoài |
| DLQ | "Hàng đợi thư chết" | Nơi tập kết an toàn cho các tác vụ xử lý thất bại |
| Claude Managed Agents | "Hệ thống quản lý máy chủ" | Hạ tầng bất đồng bộ dài hạn do Anthropic lưu trữ với bộ nhớ đệm và nén ngữ cảnh |

## Đọc thêm

- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — chi tiết cơ chế thực thi bền bỉ
- [Tổng quan Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — hệ thống bất đồng bộ dài hạn được lưu trữ trọn gói
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "hàng chục đến hàng trăm bước cho mỗi nhiệm vụ"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — cô lập lỗi nhờ mô hình diễn viên (actor model)
