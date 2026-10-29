# A2A — Giao thức Tác tử - Tác tử (Agent-to-Agent Protocol)

> MCP kết nối tác tử với công cụ (agent-to-tool). A2A (Agent2Agent) kết nối tác tử với tác tử (agent-to-agent) — một giao thức mở cho phép các tác tử hộp đen (opaque agents) được xây dựng trên các framework khác nhau có thể cộng tác. Được Google công bố vào tháng 4 năm 2025, chuyển giao cho Linux Foundation vào tháng 6 năm 2025, và đạt phiên bản v1.0 vào tháng 4 năm 2026 với hơn 150 bên ủng hộ bao gồm AWS, Cisco, Microsoft, Salesforce, SAP, và ServiceNow. Giao thức đã tiếp thu ACP của IBM và bổ sung phần mở rộng thanh toán AP2. Bài học này đi qua Thẻ tác tử (Agent Card), vòng đời Tác vụ (Task lifecycle), và ba liên kết giao thức (protocol bindings), sử dụng các danh pháp truyền tải (wire names) chuẩn của A2A 1.0.1.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python (thư viện chuẩn, khung kiểm thử Agent Card + Task)
**Yêu cầu tiên quyết:** Phase 13 · 06 (Nguyên lý cơ bản của MCP), Phase 13 · 08 (MCP client)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Phân biệt các trường hợp sử dụng giữa tác tử - công cụ (agent-to-tool - MCP) và tác tử - tác tử (agent-to-agent - A2A).
- Công bố một Thẻ tác tử (Agent Card) tại `/.well-known/agent-card.json` với siêu dữ liệu về kỹ năng (skills) và `supportedInterfaces`.
- Duyệt qua vòng đời Tác vụ (Task lifecycle): `TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_INPUT_REQUIRED`, và các trạng thái kết thúc (terminal states) `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_CANCELED`, `TASK_STATE_REJECTED`.
- Sử dụng Tin nhắn (Messages) mà mỗi Phần (Part) của nó chứa đúng một trong các kiểu `text`, `raw`, `url`, hoặc `data`, cùng các Sản phẩm tạo tác (Artifacts) dưới dạng đầu ra.

## Vấn đề đặt ra

Một tác tử chăm sóc khách hàng cần ủy quyền tác vụ viết báo cáo cho một tác tử chuyên viết lách. Các lựa chọn trước thời kỳ A2A:

- REST API tùy biến: Hoạt động được nhưng mỗi cặp kết nối đều là giải pháp đặc thù đơn lẻ.
- Cơ sở mã chung: Đòi hỏi cả hai tác tử phải chạy trên cùng một framework.
- MCP: Không phù hợp; MCP dùng để gọi công cụ, không dùng cho hai tác tử cộng tác với nhau trong khi vẫn bảo toàn tính chất hộp đen (opaque) trong suy luận nội bộ của mỗi tác tử.

A2A lấp đầy khoảng trống này. Nó mô hình hóa tương tác theo dạng một tác tử gửi một Tác vụ (Task) tới một tác tử khác, với một vòng đời hoàn chỉnh, các tin nhắn và các tạo tác (artifacts). Trạng thái nội bộ của tác tử được gọi vẫn hoàn toàn khép kín — tác tử gọi chỉ thấy các chuyển dịch trạng thái tác vụ và kết quả đầu ra cuối cùng.

A2A là giao thức "giúp các tác tử trên những framework khác nhau giao tiếp được với nhau". Nó không thay thế MCP; hai giao thức này bổ trợ cho nhau.

## Khái niệm cốt lõi

### Thẻ tác tử (Agent Card)

Mọi tác tử tuân thủ A2A đều công bố một thẻ tại `/.well-known/agent-card.json`:

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

Quá trình khám phá dựa trên URL: lấy thẻ, chọn mục đầu tiên trong `supportedInterfaces` có `protocolBinding` mà client của bạn hỗ trợ, và liệt kê các kỹ năng. Các chế độ đầu vào (input modes) và đầu ra (output modes) là các kiểu phương tiện truyền thông (media types).

### Thẻ tác tử có chữ ký số (Signed Agent Cards)

Một thẻ có thể mang mảng `signatures`. Mỗi mục là một JWS (RFC 7515) được tính toán trên JSON chuẩn hóa theo RFC 8785 của thẻ, với trường `signatures` được lược bỏ. Phía tiêu thụ chuẩn hóa thẻ theo cùng cách thức và tiến hành xác minh. Cơ chế này ngăn chặn hành vi mạo danh.

### Vòng đời tác vụ (Task Lifecycle)

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (client gửi một tin nhắn với cùng taskId)
```

Client khởi tạo bằng `SendMessage`, và server sẽ tạo ra Task. Tác tử được gọi chuyển dịch qua các trạng thái; client thăm dò (poll) bằng `GetTask` hoặc truyền phát dòng (stream) qua SSE với `SendStreamingMessage` và `SubscribeToTask`. Một luồng stream mang các sự kiện `statusUpdate` và `artifactUpdate`, rồi đóng lại khi tác vụ đạt đến trạng thái kết thúc (terminal state). Không có cờ `final`.

### Tin nhắn và Các phần (Messages and Parts)

Một tin nhắn có `messageId`, một vai trò `role` (`ROLE_USER` hoặc `ROLE_AGENT`), và một hoặc nhiều Phần (Parts). Mỗi Part chứa chính xác một trường nội dung, và tên trường đó chính là kiểu dữ liệu. Không tồn tại trường `kind`.

- `text`: nội dung văn bản thuần.
- `raw`: byte dữ liệu tệp, được mã hóa base64 trong JSON, thường đi kèm `filename` và `mediaType`.
- `url`: liên kết dẫn tới nội dung tệp.
- `data`: payload JSON có cấu trúc (đầu vào có cấu trúc cho tác tử được gọi).

Ví dụ:

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Sản phẩm tạo tác (Artifacts)

Đầu ra là các Artifact, không phải chuỗi văn bản thô. Một Artifact là một đầu ra được đặt tên và định kiểu rõ ràng:

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Các Artifact có thể được truyền phát theo từng đoạn (chunks). Mỗi sự kiện `artifactUpdate` mang theo artifact cùng với `append` và `lastChunk`. Phía gọi tích lũy dần nội dung.

### Ba liên kết giao thức (Three Protocol Bindings)

1. **JSON-RPC 2.0 qua HTTP** (`JSONRPC`): POST dành cho các yêu cầu, SSE dành cho truyền phát dòng (streaming). Các phương thức viết theo PascalCase: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, `CreateTaskPushNotificationConfig`, `GetTaskPushNotificationConfig`, `ListTaskPushNotificationConfigs`, `DeleteTaskPushNotificationConfig`, và `GetExtendedAgentCard`.
2. **gRPC** (`GRPC`): Dành cho các môi trường doanh nghiệp sử dụng gRPC bản địa. Tên phương thức giữ nguyên như trên.
3. **HTTP+JSON/REST** (`HTTP+JSON`): Các URL tài nguyên như `POST /message:send` và `GET /tasks/{id}`.

Cả ba liên kết đều mang cùng một mô hình dữ liệu. Mỗi mục trong `supportedInterfaces` nêu tên một liên kết và `protocolVersion` tương ứng. Client gửi tiêu đề `A2A-Version: 1.0` trên mỗi yêu cầu, bởi vì server sẽ hiểu một yêu cầu không có tiêu đề này là phiên bản 0.3.

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### Bảo toàn tính hộp đen (Opacity Preservation)

Một nguyên lý thiết kế then chốt: trạng thái nội bộ của tác tử được gọi là khép kín (opaque). Phía gọi chỉ thấy trạng thái tác vụ và các artifact. Chuỗi suy nghĩ (chain-of-thought), các lệnh gọi công cụ, các ủy quyền tác tử con của tác tử được gọi — tất cả đều vô hình. Điều này khác biệt với MCP, nơi các lệnh gọi công cụ là hoàn toàn minh bạch.

Lý do: A2A cho phép các đối thủ cạnh tranh có thể cộng tác mà không để lộ chi tiết nội bộ. A2A có thể hiểu là "gọi tác tử chăm sóc khách hàng này" mà phía gọi không cần biết tác tử đó triển khai dịch vụ như thế nào.

### Dòng thời gian phát triển

- **2025-04-09:** Google công bố A2A.
- **2025-06-23:** Đóng góp và chuyển giao cho Linux Foundation.
- **2025-08:** Tiếp thu ACP của IBM.
- **2025-09:** Phát hành phần mở rộng AP2 (Agent Payments).
- **2026-04:** Phát hành phiên bản v1.0 với hơn 150 tổ chức hỗ trợ.

### Mối quan hệ với MCP

| Chiều so sánh | MCP | A2A |
|---------------|-----|-----|
| Trường hợp sử dụng | Tác tử - công cụ (Agent-to-tool) | Tác tử - tác tử (Agent-to-agent) |
| Tính minh bạch | Lệnh gọi công cụ minh bạch | Suy luận nội bộ hộp đen khép kín |
| Phía gọi điển hình | Runtime của tác tử | Một tác tử khác |
| Trạng thái | Kết quả gọi công cụ | Tác vụ (Task) với vòng đời |
| Phân quyền | OAuth 2.1 (Phase 13 · 16) | `securitySchemes` + `securityRequirements` của Agent Card |
| Giao vận | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

Sử dụng MCP khi bạn muốn gọi một công cụ cụ thể. Sử dụng A2A khi bạn muốn ủy quyền toàn bộ một tác vụ cho một tác tử khác. Nhiều hệ thống sản xuất sử dụng kết hợp cả hai: tác tử dùng MCP cho tầng công cụ và dùng A2A cho tầng cộng tác.

```figure
a2a-task-lifecycle
```

## Sử dụng trong thực tế (Use It)

Tệp `code/main.py` triển khai một khung thử nghiệm A2A tối giản: tác tử viết bài công bố thẻ của mình, tác tử nghiên cứu gửi cho nó một yêu cầu `SendMessage` kèm theo một phần PDF và một chỉ dẫn văn bản, tác vụ chuyển qua các trạng thái `TASK_STATE_WORKING` → `TASK_STATE_INPUT_REQUIRED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED` trước khi trả về một artifact văn bản. Toàn bộ mã nguồn dùng thư viện chuẩn (stdlib); sử dụng giao vận trong bộ nhớ (in-memory transport) để tập trung vào cấu trúc định dạng tin nhắn.

Những điểm cần quan sát:

- Cấu trúc JSON của Agent Card.
- Phía server cấp phát mã định danh tác vụ (task id) và thực hiện chuyển dịch trạng thái.
- Các Part được định kiểu dựa trên trường nội dung hiện diện.
- Nhánh `TASK_STATE_INPUT_REQUIRED` xuất hiện giữa chừng tác vụ.
- Trả về Artifact khi hoàn tất.

## Chuyển giao sản phẩm (Ship It)

Bài học này tạo ra `outputs/skill-a2a-agent-spec.md`. Khi có một tác tử mới cần được các tác tử khác gọi tới, kỹ năng này sẽ tạo ra định dạng JSON cho Agent Card, lược đồ kỹ năng (skills schema) và bản thiết kế chi tiết cho điểm cuối (endpoint blueprint).

## Bài tập

1. Chạy `code/main.py`. Lần theo toàn bộ vòng đời của Task, bao gồm cả giai đoạn tạm dừng `TASK_STATE_INPUT_REQUIRED` khi tác tử được gọi yêu cầu thông tin làm rõ.

2. Thêm một Thẻ tác tử có chữ ký số. Đặt một mục JWS vào `signatures` với `alg` là `HS256`, ký trên JSON chuẩn hóa của thẻ không chứa trường `signatures`. Viết bộ xác minh và kiểm chứng rằng nó phát hiện thất bại khi thẻ bị chỉnh sửa.

3. Triển khai truyền phát dòng tác vụ với `SendStreamingMessage`: tác tử viết bài phát ra `task`, ba đoạn `artifactUpdate`, và một `statusUpdate` mang `TASK_STATE_COMPLETED`, sau đó đóng luồng stream. Phía gọi tích lũy các đoạn dữ liệu này lại.

4. Thiết kế một tác tử A2A bọc quanh một MCP server. Ánh xạ mỗi công cụ MCP thành một kỹ năng A2A. Phân tích các đánh đổi — tính chất khép kín nào đã bị mất đi?

5. Đọc thông báo phát hành A2A v1.0 và xác định tính năng duy nhất chưa được framework nào triển khai tính đến tháng 4 năm 2026. (Gợi ý: liên quan đến ủy quyền tác vụ đa chặng - multi-hop task delegation).

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|-----------|----------------------|---------------------------|
| A2A | "Giao thức Agent-to-Agent" | Giao thức mở cho việc cộng tác giữa các tác tử hộp đen khép kín |
| Agent Card | "`/.well-known/agent-card.json`" | Siêu dữ liệu được công bố mô tả kỹ năng và `supportedInterfaces` của tác tử |
| Kỹ năng (Skill) | "Đơn vị có thể gọi" | Một thao tác có tên mà tác tử hỗ trợ (tương đương công cụ trong MCP) |
| Tác vụ (Task) | "Đơn vị ủy quyền" | Một hạng mục công việc có vòng đời và sản phẩm tạo tác hoàn thiện |
| Tin nhắn (Message) | "Đầu vào tác vụ" | Mang các Phần (`text`, `raw`, `url`, `data`) |
| Phần (Part) | "Mẩu dữ liệu định kiểu" | Chính xác một trong các kiểu `text` / `raw` / `url` / `data`, kèm `mediaType` tùy chọn; không có trường `kind` |
| Sản phẩm tạo tác (Artifact) | "Đầu ra tác vụ" | Đầu ra có tên và kiểu dữ liệu cụ thể được trả về khi hoàn tất |
| AP2 | "Giao thức thanh toán cho tác tử" | Phần mở rộng thanh toán xây dựng trên A2A; ký thẻ là tính năng cốt lõi của A2A (`signatures`) |
| Tính hộp đen (Opacity) | "Cộng tác hộp đen" | Chi tiết nội bộ của tác tử được gọi hoàn toàn ẩn giấu đối với phía gọi |
| `TASK_STATE_INPUT_REQUIRED` | "Tạm dừng tác vụ" | Trạng thái gián đoạn khi tác tử cần thêm thông tin |

## Tài liệu đọc thêm

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — Đặc tả chuẩn của A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — Các bản triển khai tham chiếu và SDK
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1) — Tệp `docs/specification.md` được gắn thẻ và tệp quy chuẩn `specification/a2a.proto` mà bài học này tuân thủ
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — Thông cáo chuyển giao quản trị tháng 6 năm 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — Lộ trình phát triển và đà tăng trưởng đối tác
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — Ghi chú phát hành v1.0 và hướng dẫn tương thích ngược
