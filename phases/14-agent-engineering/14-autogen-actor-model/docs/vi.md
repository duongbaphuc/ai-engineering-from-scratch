# Mô hình Actor cho Tác tử — Thông điệp Bất đồng bộ và Môi trường Thực thi Định kiểu

> Tác tử đóng vai trò là các actor: trao đổi thông điệp bất đồng bộ (async message exchange), bộ xử lý hướng sự kiện (event-driven handlers), cô lập lỗi (fault isolation), và tính đồng thời tự nhiên (natural concurrency). AutoGen v0.4 (Microsoft Research, tháng 1/2025) đã tái thiết kế kiến trúc điều phối tác tử xoay quanh mô hình này; hiện framework đang ở chế độ bảo trì (maintenance mode), với Microsoft Agent Framework (bản xem trước công khai tháng 10/2025) là phiên bản kế nhiệm hướng tới môi trường sản xuất.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Mục tiêu học tập

- Mô tả mô hình Actor (actor model): các tác tử là actor, thông điệp là phương thức giao tiếp liên tiến trình (IPC) duy nhất, cô lập sự cố trên từng actor.
- Nêu tên ba tầng API của AutoGen v0.4 — Core, AgentChat, Extensions — và mục đích của từng tầng.
- Giải thích lý do việc tách rời quá trình chuyển phát thông điệp khỏi quá trình xử lý mang lại khả năng cô lập lỗi và tính đồng thời tự nhiên.
- Triển khai một môi trường thực thi (runtime) mô hình actor bằng thư viện chuẩn (stdlib) trong Python và chuyển đổi một luồng đánh giá mã nguồn (code-review) hai tác tử lên runtime đó.

## Vấn đề

Hầu hết các framework tác tử đều mang tính đồng bộ: một tác tử sản sinh dữ liệu, một tác tử tiêu thụ dữ liệu, hoạt động trong cùng một ngăn xếp lệnh gọi (call stack). Sự cố phát sinh sẽ làm sập toàn bộ ngăn xếp. Tính đồng thời chỉ là phần chắp vá gắn thêm. Việc phân tán hệ thống đòi hỏi phải viết lại toàn bộ mã nguồn.

Câu trả lời từ AutoGen v0.4: mô hình Actor. Mỗi tác tử là một actor sở hữu hộp thư đến (inbox) riêng tư. Các thông điệp là phương thức tương tác duy nhất. Môi trường thực thi tách rời việc chuyển phát thông điệp khỏi việc xử lý thông điệp. Sự cố được cô lập trong phạm vi của một actor duy nhất. Tính đồng thời là bản năng gốc. Việc phân tán chỉ đơn thuần là thay đổi cơ chế truyền tải (transport).

## Khái niệm

### Các Actor

Một actor bao gồm:

- Một trạng thái riêng tư (private state - không bao giờ bị can thiệp trực tiếp từ bên ngoài).
- Một hộp thư đến (inbox - hàng đợi thông điệp).
- Một bộ xử lý (handler): `receive(message) -> effects`, trong đó các hiệu ứng phụ (effects) có thể là "trả lời (reply)", "gửi tới actor khác", "sinh actor mới (spawn)", "cập nhật trạng thái", "tự dừng (stop self)".

Hai actor không thể chia sẻ bộ nhớ. Chúng chỉ có thể gửi thông điệp cho nhau.

### Ba tầng API

AutoGen v0.4 chia bề mặt tương tác thành ba tầng:

1. **Core.** Framework actor cấp thấp. `AgentRuntime`, `Agent`, `Message`, `Topic`. Trao đổi thông điệp bất đồng bộ, hướng sự kiện.
2. **AgentChat.** API cấp cao hướng nhiệm vụ (thay thế cho ConversableAgent của phiên bản v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Các tích hợp mở rộng — OpenAI, Anthropic, Azure, công cụ (tools), bộ nhớ (memory).

### Tại sao việc tách rời lại quan trọng

Trong mô hình v0.2, việc gọi `agent_a.chat(agent_b)` sẽ chặn đồng bộ (blocks) agent_a cho đến khi agent_b trả về kết quả. Ở v0.4, lệnh `send(agent_b, msg)` sẽ đẩy thông điệp vào hộp thư đến của agent_b và trả về ngay lập tức. Runtime sẽ chuyển phát thông điệp sau đó. Điều này mang lại ba hệ quả:

- **Cô lập lỗi (Fault isolation).** Agent B gặp sự cố (crash) không làm sập Agent A — runtime sẽ bắt lỗi trong bộ xử lý của B và quyết định cách xử lý tiếp theo (ghi nhật ký, thử lại, chuyển vào hàng đợi thư chết / dead-letter).
- **Tính đồng thời tự nhiên (Natural concurrency).** Nhiều thông điệp có thể cùng lúc ở trạng thái đang truyền tải (in flight); các actor xử lý hộp thư đến của mình một cách đồng thời.
- **Sẵn sàng cho kiến trúc phân tán (Distribution-ready).** Hộp thư đến kết hợp phương thức truyền tải tạo thành cùng một lớp trừu tượng, bất kể actor đang chạy trong cùng tiến trình hay trên một máy chủ (host) khác.

### Các mô hình tô-pô

- **RoundRobinGroupChat.** Các tác tử luân phiên thực hiện lượt theo một vòng quay cố định.
- **SelectorGroupChat.** Một tác tử lựa chọn (selector agent) chọn ra tác tử nào sẽ thực hiện tiếp theo dựa trên ngữ cảnh hội thoại.
- **Magentic-One.** Đội ngũ đa tác tử (multi-agent team) tham chiếu dùng cho các tác vụ duyệt web, thực thi mã nguồn, xử lý tệp. Được xây dựng trên nền tảng AgentChat.

### Khả năng quan sát

Khả năng hỗ trợ OpenTelemetry được tích hợp sẵn. Mỗi thông điệp đều phát ra một span; các lệnh gọi công cụ mang các thuộc tính `gen_ai.*` theo quy ước ngữ nghĩa OTel GenAI năm 2026 (Bài 23).

### Trạng thái: Chế độ bảo trì

Đầu năm 2026: AutoGen v0.7.x hoạt động ổn định cho mục đích nghiên cứu và tạo mẫu (prototyping). Microsoft đã chuyển hướng phát triển tích cực sang Microsoft Agent Framework, phiên bản kế nhiệm hướng tới môi trường sản xuất (bản xem trước công khai ngày 1/10/2025; bản GA 1.0 dự kiến vào cuối Quý 1/2026). Các mẫu hình của AutoGen có thể chuyển tiếp một cách mạch lạc — mô hình Actor chính là tư tưởng mang tính bền vững lâu dài.

```figure
actor-mailbox
```

## Xây dựng

`code/main.py` triển khai một môi trường thực thi mô hình actor chỉ dùng thư viện chuẩn (stdlib):

- `Message` — tải trọng được định kiểu (typed payload) chứa `sender`, `recipient`, `topic`, `body`.
- `Actor` — lớp trừu tượng với phương thức `receive(message, runtime)`.
- `Runtime` — vòng lặp sự kiện (event loop) với một hàng đợi dùng chung, cơ chế chuyển phát và cô lập lỗi.
- Bản demo hai actor: `ReviewerAgent` đánh giá mã nguồn, `ChecklistAgent` chạy danh sách kiểm tra; chúng trao đổi thông điệp cho đến khi đạt được sự đồng thuận.

Chạy thử:

```bash
python3 code/main.py
```

Dấu vết thực thi hiển thị quá trình chuyển phát thông điệp, mô phỏng một sự cố ở một actor mà không làm sập actor còn lại, và sự hội tụ về một phán quyết chung.

## Sử dụng

- **AutoGen v0.4/v0.7** (bảo trì) — ổn định cho nghiên cứu, tạo mẫu nhanh, và các mẫu hình đa tác tử.
- **Microsoft Agent Framework** — phiên bản kế nhiệm cho môi trường sản xuất (xem trước công khai tháng 10/2025); kế thừa tư tưởng mô hình Actor trong một API được làm mới.
- **Tô-pô bầy đàn của LangGraph (LangGraph swarm topology)** (Bài 13) — mẫu hình tương tự thông qua việc bàn giao bằng công cụ dùng chung.
- **Môi trường thực thi Actor tùy biến (Custom actor runtime)** — khi bạn cần cơ chế truyền tải đặc thù (NATS, RabbitMQ, gRPC).

## Chuyển giao

`outputs/skill-actor-runtime.md` tạo ra một runtime actor tối giản kèm khuôn mẫu đội ngũ (RoundRobin hoặc Selector) cho một nhiệm vụ đa tác tử cụ thể.

## Bài tập

1. Thêm một hàng đợi thư chết (dead-letter queue - DLQ): khi một bộ xử lý ném ra ngoại lệ (raises), hãy tạm giữ thông điệp lỗi để con người kiểm tra. Trong mô hình đồ chơi của bạn, DLQ được kích hoạt với tần suất bao nhiêu?
2. Triển khai `SelectorGroupChat`: một actor lựa chọn sẽ chọn ra ai là người xử lý thông điệp tiếp theo dựa trên trạng thái hội thoại.
3. Thêm phương thức truyền tải phân tán: thay thế hàng đợi nội tiến trình bằng máy chủ JSON-qua-HTTP để các actor có thể chạy trên các tiến trình tách biệt.
4. Gắn một span OTel cho mỗi thông điệp (hoặc dùng đối tượng thay thế no-op). Phát ra các thuộc tính `gen_ai.agent.name`, `gen_ai.operation.name` theo Bài 23.
5. Đọc bài viết về kiến trúc của AutoGen v0.4. Chuyển đổi mã đồ chơi của bạn sang API `autogen_core` thực tế. Bạn đã bỏ qua những yếu tố nào có ý nghĩa quan trọng trong môi trường sản xuất?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Actor | "Tác tử" | Trạng thái riêng tư + hộp thư đến + bộ xử lý; không chia sẻ bộ nhớ |
| Thông điệp (Message) | "Sự kiện" | Tải trọng định kiểu; cách duy nhất để các actor tương tác |
| Hộp thư đến (Inbox) | "Hòm thư (Mailbox)" | Hàng đợi các thông điệp đang chờ xử lý của từng actor |
| Môi trường thực thi (Runtime) | "Host của tác tử" | Vòng lặp sự kiện giúp định tuyến thông điệp và cô lập lỗi |
| Chủ đề (Topic) | "Kênh (Channel)" | Tuyến xuất bản - đăng ký (pub-sub) có tên giữa các actor |
| Cô lập lỗi (Fault isolation) | "Cứ để nó sập (Let it crash)" | Một actor gặp sự cố không làm sập các actor khác |
| RoundRobinGroupChat | "Đội ngũ luân phiên cố định" | Các tác tử lần lượt thay nhau theo thứ tự |
| SelectorGroupChat | "Đội ngũ định tuyến theo ngữ cảnh" | Bộ lựa chọn sẽ chọn ai là người thực hiện tiếp theo |
| Magentic-One | "Đội ngũ tham chiếu" | Biệt đội đa tác tử cho web + mã nguồn + tệp tin |

## Tài liệu đọc thêm

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — bài viết về việc tái thiết kế kiến trúc
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — giải pháp thay thế dựa trên cấu trúc đồ thị
- [Quy ước ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — các span mà AutoGen phát ra theo mặc định
