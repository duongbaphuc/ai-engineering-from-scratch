# Phần mở rộng MCP Tasks: Xử lý công việc bền vững trên nền lõi phi trạng thái

> Giao thức MCP phi trạng thái (stateless MCP) không đồng nghĩa với việc mọi thao tác đều phải hoàn tất chỉ trong một chu kỳ yêu cầu duy nhất. Phần mở rộng Tasks chính thức (official Tasks extension) trao cho các tác vụ chạy ngầm thời gian dài một thẻ điều khiển bền vững (durable handle) rõ ràng. Máy chủ có thể trả về thẻ điều khiển này từ `tools/call`, bất kỳ phiên bản máy chủ (instance) nào cũng có thể phản hồi `tasks/get`, và dữ liệu đầu vào từ phía client được gửi thông qua `tasks/update` mà không cần phải khôi phục lại các phiên giao thức (protocol sessions).

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 13 · 09 (giao vận), Phase 13 · 11 (MRTR phi trạng thái), Phase 13 · 12 (gợi mở thông tin)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Phân biệt giữa tầng giao vận giao thức phi trạng thái (stateless protocol transport) và trạng thái tác vụ ứng dụng bền vững (durable application task state).
- Đàm phán phần mở rộng `io.modelcontextprotocol/tasks` trong các năng lực theo từng yêu cầu (per-request capabilities) và lệnh gọi `server/discover`.
- Trả về `CreateTaskResult` do máy chủ điều phối với `resultType: "task"` chỉ sau khi tác vụ đã được tạo lập bền vững trong hệ thống lưu trữ.
- Thăm dò định kỳ (polling) với `tasks/get`, cung cấp dữ liệu đầu vào cho tác vụ với `tasks/update`, và yêu cầu hủy bỏ phối hợp (cooperative cancellation) với `tasks/cancel`.
- Loại bỏ các giả định cũ về `tasks/status`, `tasks/result`, và `tasks/list`.
- Đăng ký nhận thông báo tác vụ tùy chọn thông qua `subscriptions/listen` trên một luồng SSE phản hồi của POST (POST response SSE stream).
- Mô hình hóa chuẩn xác thời gian hết hạn của tác vụ (task expiry), khôi phục sau khởi động lại (restart recovery), khử trùng lặp khóa đầu vào (input-key deduplication), và xử lý lỗi thực thi.

## Tại sao Tasks lại là một phần mở rộng?

Tasks ban đầu xuất hiện như một tính năng cốt lõi thử nghiệm trong bản phát hành 2025-11-25. Đợt tái thiết kế vào tháng 7 năm 2026 đã chuyển chúng thành phần mở rộng chính thức `io.modelcontextprotocol/tasks` để client và server có thể chủ động chọn tham gia vào vòng đời tác vụ bổ sung này mà không làm phình to đặc tả giao thức lõi đối với tất cả mọi người.

Đặc tả của phần mở rộng hiện vẫn nằm ở trạng thái bản thảo (draft surface) mặc dù đây là địa chỉ chính thức hiện thời dành cho Tasks. Hãy ghim chặt phiên bản phần mở rộng được hỗ trợ bởi SDK của bạn, chạy các kịch bản kiểm thử tuân thủ (conformance scenarios), và tách biệt các bộ điều hợp giao thức truyền thông (wire adapters) khỏi miền lưu trữ và tiến trình xử lý ngầm (worker domain).

Hãy sử dụng một tác vụ khi thao tác có một hoặc nhiều đặc điểm sau:

- Thao tác có thể vượt quá giới hạn thời gian chờ yêu cầu thông thường (request timeout).
- Một hàng đợi worker hoặc một hệ thống điều phối tác vụ bên ngoài đã chịu trách nhiệm thực thi.
- Client cần khả năng tự khôi phục trạng thái sau khi tự khởi động lại.
- Thao tác phải tạm dừng để chờ đầu vào từ phía người dùng hoặc mô hình trong quá trình thực thi.
- Yêu cầu hủy bỏ và truy xuất kết quả bền vững là những đòi hỏi bắt buộc của sản phẩm.

Tuyệt đối không tạo tác vụ cho một thao tác tra cứu tất định, chi phí thấp. Việc cấp thẻ điều khiển, duy trì trạng thái bền vững, thăm dò định kỳ, quản lý hết hạn và hủy bỏ đều đi kèm với độ phức tạp thực tế của hệ thống.

## Lõi phi trạng thái, Ứng dụng có trạng thái

Bản đặc tả MCP 2026-07-28 đã loại bỏ hoàn toàn `initialize`, `notifications/initialized`, các phiên giao thức (protocol sessions), và tiêu đề `Mcp-Session-Id`. Điều này không hề cấm đoán các sản phẩm có trạng thái (stateful products).

Mã định danh tác vụ (task id) là trạng thái ứng dụng tường minh:

- Máy chủ lưu trữ bền vững (persist) nó trước khi gửi trả về cho client.
- Client có thể lưu mã này lại và tiếp tục thăm dò sau khi khởi động lại.
- Mã định danh có thể được định tuyến tới bất kỳ bản sao (replica) nào dùng chung tầng lưu trữ bền vững.
- Quyền hạn ủy quyền (authorization) được kiểm tra trên từng phương thức tác vụ.
- Hết hạn và xóa bỏ được định nghĩa bởi các trường dữ liệu của tác vụ, không phụ thuộc vào vòng đời của tầng giao vận (transport lifetime).

Về mặt vận hành, cơ chế này hoàn toàn khác biệt với trạng thái ẩn được gắn chặt vào một kết nối mạng.

Hãy phân định rõ ràng bốn vòng đời khác nhau:

| Trạng thái | Vòng đời | Nơi lưu trữ phù hợp |
|---|---|---|
| Siêu dữ liệu giao thức | Một yêu cầu đơn lẻ | `params._meta`, được xác thực lại trong mỗi lệnh gọi |
| Xử lý giao vận | Một yêu cầu stdio hoặc phản hồi HTTP | Bộ điều phối đang xử lý (in-flight coordinator) với hạn chót hữu hạn |
| Tiếp tục chu kỳ MRTR | Một chuỗi thử lại | `requestState` được bảo vệ tính toàn vẹn, kèm kiểm soát chống phát lại khi cần |
| Tác vụ bền vững | Xuyên suốt nhiều yêu cầu, bản sao, lần khởi động lại và kết nối lại | Tầng lưu trữ ứng dụng dùng chung, định danh bằng `taskId` đã được ủy quyền |

Chuyển bản ghi tác vụ vào bộ nhớ tiến trình (process memory) không làm cho MCP trở nên có trạng thái. Nó chỉ làm cho ứng dụng trở nên kém tin cậy. Giao thức vẫn duy trì phi trạng thái, nhưng một lệnh gọi `tasks/get` sau đó được định tuyến đến bản sao khác sẽ không thể khôi phục lại bản ghi. Hãy ghi bền vững trước khi trả về thẻ điều khiển, sau đó đảm bảo mọi phương thức tác vụ đều truy vấn cùng một bản ghi dùng chung dưới sự kiểm tra nghiêm ngặt về đối tượng thuê (tenant) và chủ thể (principal).

## Đàm phán năng lực

Client công bố năng lực hỗ trợ trong từng yêu cầu hợp lệ:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

Máy chủ trả về các giá trị chính xác về `supportedVersions`, năng lực, `ttlMs`, và `cacheScope` từ lệnh gọi `server/discover`, với cùng phần mở rộng nằm dưới mục capabilities. Vì công bố công cụ, máy chủ cũng phải triển khai phương thức bắt buộc `tools/list`. Kết quả này trả về một bộ mô tả công cụ `generate_report` tất định, lược đồ đối tượng hợp lệ `inputSchema`, `resultType: "complete"`, siêu dữ liệu định danh máy chủ, và các chỉ dẫn bộ nhớ đệm công khai (public cache hints).

Một phương thức tác vụ xuất phát từ client không khai báo phần mở rộng sẽ trả về mã lỗi `-32021` (Thiếu Năng lực Bắt buộc của Client - Missing Required Client Capability), với `data.requiredCapabilities` đặt thành `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`. Một chuỗi phiên bản giao thức không được hỗ trợ sẽ trả về `-32022` cùng dữ liệu chính xác về `supported` và `requested`; phiên bản bị thiếu hoặc không phải kiểu chuỗi sẽ trả về lỗi `-32602`.

Một phong bì gói tin (envelope) không có `id` JSON-RPC là một thông báo (notification). Phía nhận có thể xử lý nó, nhưng không được phát ra kết quả hay lỗi JSON-RPC nào. Bộ điều hợp Streamable HTTP sẽ trả về `202 Accepted` không có nội dung body đối với một thông báo được chấp nhận.

Hiện tại, chỉ có `tools/call` hỗ trợ cơ chế thực thi bổ trợ tác vụ (task-augmented execution). Hãy thiết kế trừu tượng nội bộ của bạn sao cho các loại yêu cầu trong tương lai không đòi hỏi phải viết lại tầng lưu trữ.

## Tạo tác vụ do máy chủ điều phối

Cờ cấu hình cũ do client gửi lên `params._meta.task.required` đã bị bãi bỏ. Client chỉ khai báo khả năng hỗ trợ phần mở rộng, sau đó máy chủ sẽ quyết định xem một lệnh `tools/call` cụ thể có trở thành một tác vụ hay không.

Yêu cầu:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Phản hồi:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

Máy chủ không được phép trả về thẻ điều khiển này cho đến khi lệnh gọi `tasks/get` ứng với id đó chắc chắn có thể truy xuất thành công. Trong một kho lưu trữ có tính nhất quán cuối cùng (eventually consistent store), hãy đợi cho đến khi bản ghi hiển thị đối với các thao tác đọc (read visibility) trước khi gửi phản hồi. Nếu không, client có thể nhận được một id trông có vẻ hợp lệ nhưng lại lập tức gặp lỗi "not found".

Một phản hồi dạng tác vụ mang tính chất chủ động từ máy chủ (unsolicited) theo nghĩa client không trực tiếp yêu cầu chế độ tác vụ. Tuy nhiên, nó không phải là phản hồi chưa qua đàm phán: yêu cầu hiện tại vẫn bắt buộc phải công bố phần mở rộng.

## Cấu trúc dữ liệu của một Task

Mỗi tác vụ mang các trường dữ liệu sau:

- `taskId`: mã định danh ổn định do máy chủ tạo ra;
- `status`: trạng thái của tác vụ, nhận một trong các giá trị `working`, `input_required`, `completed`, `cancelled`, hoặc `failed`;
- `createdAt` và `lastUpdatedAt`: dấu thời gian chuẩn định dạng ISO 8601;
- `ttlMs`: thời gian hết hạn tính từ lúc tạo tác vụ, hoặc `null` nếu không công bố giới hạn;
- `pollIntervalMs` (tùy chọn): khoảng thời gian thăm dò tối thiểu khuyến nghị hiện thời của máy chủ;
- `statusMessage` (tùy chọn): thông điệp ngữ cảnh hướng tới người dùng hoặc mô hình.

Các trường đặc thù theo từng trạng thái chỉ xuất hiện khi thích hợp:

- `input_required` bao gồm `inputRequests`.
- `completed` bao gồm cấu trúc `result` từ yêu cầu ban đầu.
- `failed` bao gồm đối tượng `error` theo chuẩn JSON-RPC.

Client nên tuân thủ giá trị `pollIntervalMs`. Máy chủ có thể áp dụng giới hạn tốc độ (rate-limit) đối với các hành vi thăm dò quá dồn dập và có thể điều chỉnh khoảng thời gian này trong suốt vòng đời của tác vụ.

## Thăm dò trạng thái với `tasks/get`

Client gửi yêu cầu để lấy ảnh chụp nhanh (snapshot) hiện tại:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Bản thân lệnh gọi `tasks/get` đã hoàn tất xử lý, do đó trường kết quả của nó luôn có `resultType: "complete"`. Tác vụ lồng bên trong vẫn có thể có `status: "working"` hoặc `status: "input_required"`.

Sự phân định rạch ròi này giúp ngăn chặn lỗi phân tích cú pháp thường gặp:

```text
result.resultType = complete    nghĩa là lệnh RPC tasks/get đã hoàn tất
result.status = working        nghĩa là công việc ngầm được đại diện vẫn đang chạy
```

Giao thức không hề có lệnh gọi `tasks/result`. Khi tác vụ hoàn thành, phản hồi `tasks/get` tiếp theo sẽ gắn trực tiếp kết quả `CallToolResult` ban đầu vào trường `result`:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

Trường `resultType` bên ngoài cho biết lệnh RPC `tasks/get` đã hoàn tất. Trường `result.resultType` lồng bên trong xác nhận rằng lệnh gọi công cụ ban đầu đã thực thi xong. Bộ phân biệt lồng nhau (nested discriminator) này là bắt buộc. `CallToolResult` lồng nhau CŨNG NÊN mang thông tin `io.modelcontextprotocol/serverInfo` của riêng nó; bài học này bao gồm trường thông tin này thay vì lưu trữ một tải trọng không xác định kiểu dữ liệu.

Giao thức không có phương thức `tasks/list`. Máy chủ không có phiên giao thức không thể suy luận an toàn danh sách tác vụ nào thuộc về phạm vi của một kết nối cụ thể. Ứng dụng có nhu cầu xem lại lịch sử nên mở một công cụ nghiệp vụ riêng có xác thực kèm theo các bộ lọc và quy tắc quyền sở hữu minh bạch.

## Dữ liệu đầu vào trong quá trình thực thi tác vụ

Dữ liệu đầu vào của tác vụ và cơ chế MRTR của giao thức lõi trông có vẻ tương đồng nhưng sử dụng các đường truyền tiếp diễn (continuations) khác nhau.

### Đầu vào cần thiết trước khi tạo tác vụ

Trả về `resultType: "input_required"` từ giao thức lõi ngay tại lệnh gọi `tools/call` ban đầu. Client thu thập dữ liệu và thử lại lệnh gọi gốc đó. Chỉ tạo tác vụ sau khi các vòng trao đổi MRTR đồng bộ này đã kết thúc.

### Đầu vào cần thiết sau khi đã tạo tác vụ

Chuyển trạng thái tác vụ sang `input_required`. Phương thức `tasks/get` sẽ phơi bày các yêu cầu `inputRequests` đang chờ xử lý, và client gửi các phản hồi qua `tasks/update`. Client tuyệt đối không thử lại lệnh gọi `tools/call` ban đầu.

Ảnh chụp trạng thái:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

Cập nhật đầu vào:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Phản hồi thành công là một thông điệp xác nhận rỗng kèm theo `resultType: "complete"`. Thay đổi trạng thái có thể tuân theo mô hình nhất quán cuối cùng, do đó client tiếp tục thăm dò hoặc lắng nghe sự kiện.

Mỗi khóa trong `inputRequests` phải là duy nhất trong toàn bộ vòng đời của tác vụ. Các ảnh chụp `tasks/get` lặp lại có thể hiển thị cùng một khóa đang chờ; client cần khử trùng lặp trên giao diện người dùng và máy chủ sẽ bỏ qua phản hồi đối với các khóa không tồn tại, đã bị thay thế hoặc đã được đáp ứng trước đó. Một bản cập nhật từng phần có thể giữ tác vụ ở trạng thái `input_required` cho đến khi tất cả các khóa bắt buộc đều đã nhận được câu trả lời.

## Cơ chế hủy bỏ mang tính phối hợp

`tasks/cancel` đóng vai trò báo hiệu ý định hủy và trả về một thông điệp xác nhận hoàn tất rỗng. Thông điệp xác nhận này không đảm bảo rằng worker đã dừng lại. Công việc có thể đã hoàn thành trước đó, bỏ qua yêu cầu hủy, hoặc chuyển đổi trạng thái muộn hơn.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Đối với cả ba phương thức tác vụ, tiêu đề `Mcp-Name` phản ánh giá trị `params.taskId`. Nó không lặp lại tên phương thức JSON-RPC. Tệp `code/main.py` tập trung hóa quy tắc này trong hàm `make_http_request`.

Worker trong bài học thực thi việc hủy bỏ ngay lập tức, giúp các lệnh gọi lặp lại có tính lũy suy (idempotent). Một client trong môi trường thực tế vẫn phải coi thao tác hủy là mang tính phối hợp (cooperative) thay vì tự động suy diễn trạng thái kết thúc của tác vụ từ thông điệp xác nhận.

Tuyệt đối không dùng `notifications/cancelled` để hủy một tác vụ. Thông báo đó thuộc về cơ chế hủy bỏ yêu cầu (request cancellation), không dùng cho Tasks bền vững.

Sự phân biệt này mang tính quyết định tại ranh giới định tuyến. Hủy bỏ yêu cầu nhắm tới một thao tác JSON-RPC đang xử lý hoặc phản hồi HTTP trong phạm vi yêu cầu của nó. Nếu `tools/call` đã trả về `resultType: "task"`, yêu cầu đó đã hoàn tất và việc đóng kênh truyền tải của nó không thể định danh hay dừng công việc bền vững đó lại. `tasks/cancel` là một lệnh RPC mới được ủy quyền riêng. Nó mang `params.taskId`, sao chép id đó trong `Mcp-Name`, phân giải backend sở hữu tác vụ, ghi nhận ý định hủy bỏ có phối hợp, và trả về thông điệp xác nhận mà không khẳng định rằng worker đã dừng hoàn toàn.

Do đó, một gateway phải duy trì các bộ điều phối yêu cầu và các tuyến đường tác vụ trong các bảng khác nhau. Bảng yêu cầu có thể biến mất ngay khi phản hồi kết thúc. Tuyến đường tác vụ phải tồn tại cho đến khi đạt trạng thái kết thúc và hết thời gian lưu giữ (retention expiry). [Bài học 29: Độ tin cậy, Hủy bỏ và Kiểm soát luồng trong MCP](../../29-mcp-reliability-cancellation-and-flow-control/docs/vi.md) sẽ xây dựng chi tiết các quy tắc tranh chấp (race), thời gian chờ (timeout), tính lũy suy (idempotency), áp lực ngược (backpressure), và thử lại cho cả hai luồng xử lý.

## Thông báo tùy chọn

Thăm dò định kỳ là phương án nền tảng. Client muốn nhận các bản cập nhật đẩy (push updates) sẽ gửi lệnh `subscriptions/listen` kèm theo danh sách các id tác vụ. Đối với cơ chế Streamable HTTP, đây là một yêu cầu POST có phản hồi là một luồng SSE trong phạm vi yêu cầu (request-scoped SSE stream). Không có luồng sự kiện GET độc lập và không có phiên giao thức nào cần duy trì sống.

Máy chủ xác nhận các id được chấp thuận bằng `notifications/subscriptions/acknowledged` và sau đó có thể gửi toàn bộ ảnh chụp nhanh trạng thái thông qua `notifications/tasks`. Thông điệp xác nhận và mọi thông báo tác vụ đều chứa `io.modelcontextprotocol/subscriptionId` trong `_meta`, tương ứng với id của yêu cầu `subscriptions/listen`. Từng thông báo tác vụ có nội dung tương đương với những gì `tasks/get` sẽ trả về tại thời điểm đó.

Client vẫn bắt buộc phải khai báo phần mở rộng Tasks. Client nên chủ động kết nối lại và tiếp tục từ các id tác vụ bền vững thay vì phụ thuộc vào việc phát lại sự kiện hay trường `Last-Event-ID`.

## Ngữ nghĩa xử lý lỗi

Sử dụng hai tầng lỗi một cách chuẩn xác:

### Lỗi giao thức

Các tham số phương thức không hợp lệ hoặc một id tác vụ không tồn tại sẽ trả về lỗi JSON-RPC, phổ biến là `-32602`. Thiếu hỗ trợ phần mở rộng trả về `-32021` kèm đối tượng năng lực bắt buộc.

### Kết quả thực thi tác vụ

- Một kết quả công cụ bình thường có `isError: true` vẫn được xem là tác vụ `completed` vì lệnh gọi công cụ đã tạo ra kết quả được định nghĩa của nó.
- Một lỗi JSON-RPC xảy ra trong quá trình thực thi bị trì hoãn sẽ đưa tác vụ sang trạng thái `failed` và lưu trữ lỗi JSON-RPC đó dưới trường `error`.
- Sự từ chối của người dùng có thể dẫn đến trạng thái `cancelled`, một kết quả từ chối đã hoàn tất, hoặc một kết quả an toàn khác tùy thuộc vào miền nghiệp vụ. Cần tài liệu hóa rõ ràng lựa chọn này.

## Tính bền vững, Hết hạn và Quyền sở hữu

Lưu trữ bền vững tối thiểu các thông tin: task id, trạng thái, các dấu thời gian, ttl, khoảng thời gian thăm dò, quyền sở hữu thao tác ban đầu, kết quả hoặc lỗi, các yêu cầu đầu vào đang chờ xử lý, và tất cả các khóa đầu vào đã phát hành.

Khóa lưu trữ phải bao gồm hoặc phân giải được đối tượng thuê (tenant) và chủ thể (principal) có thẩm quyền. Việc chỉ biết mã định danh task id tuyệt đối không được cấp quyền truy cập. Phải kiểm tra quyền sở hữu trên mọi lệnh `tasks/get`, `tasks/update`, `tasks/cancel`, và các lượt đăng ký nhận thông báo.

`ttlMs` được tính từ lúc khởi tạo và có thể thay đổi. Client có thể coi đây là giới hạn bảo vệ an toàn khi tác vụ ngừng tạo ra các bản cập nhật có thể quan sát được. Máy chủ có thể chuyển tác vụ sang thất bại và sau đó xóa bỏ một tác vụ đã hết hạn. Đừng diễn giải trường này như một cam kết sẽ lưu giữ kết quả hoàn thành trong ngần ấy mili-giây sau khi tác vụ kết thúc.

Sử dụng các thao tác ghi nguyên tử (atomic writes) hoặc giao dịch (transactions). Bài học này thực hiện ghi ra một tệp tạm thời rồi đổi tên tệp một cách nguyên tử. Một dịch vụ có nhiều bản sao nên sử dụng một kho dữ liệu bền vững dùng chung và cơ chế cấp quyền worker (worker lease) hoặc kiểm soát tương tranh tương đương.

````figure
tp-task-lifecycle
````

## Xây dựng hệ thống

Tệp `code/main.py` triển khai một dịch vụ tác vụ tất định:

- `server/discover` trả về `supportedVersions`, chỉ dẫn bộ nhớ đệm, và phần mở rộng Tasks.
- `tools/list` trả về bộ mô tả `generate_report` tất định, có thể lưu đệm với lược đồ đầu vào hợp lệ.
- `tools/call` tạo và lưu trữ bền vững tác vụ trước khi trả về `resultType: "task"`.
- Một phiên bản dịch vụ mới tải lại cùng tác vụ đó, minh chứng cho khả năng khôi phục sau khởi động lại.
- `tasks/get` trả về các ảnh chụp nhanh hoàn chỉnh của tác vụ.
- Worker chuyển trạng thái từ `working` sang `input_required`.
- `tasks/update` tiếp nhận phản hồi biểu mẫu và trả về xác nhận hoàn tất rỗng.
- Worker lưu trữ một `CallToolResult` lồng nhau với `resultType` và định danh máy chủ của riêng nó, sau đó chuyển trạng thái sang `completed`.
- Phương thức `tasks/cancel` có tính lũy suy trong bản triển khai này.
- Bộ dựng HTTP thiết lập `Mcp-Name` thành `params.taskId` cho các phương thức `tasks/get`, `tasks/update`, và `tasks/cancel`.
- Các hàm trợ giúp thông báo sử dụng `notifications/subscriptions/acknowledged` và `notifications/tasks`, cả hai đều được gắn thẻ bằng id của yêu cầu lắng nghe.
- Các thông báo không có id sẽ không sinh ra bất kỳ phản hồi JSON-RPC nào.

Worker thúc đẩy trạng thái một cách tường minh thay vì dừng chờ (sleep) trong một luồng nền. Điều này giúp mọi chuyển đổi trạng thái diễn ra tất định và tách biệt ví dụ về giao thức khỏi các cơ chế hàng đợi.

## Thử nghiệm thực tế

Từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

Trình tự kết quả kỳ vọng:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

Hãy xác minh thêm rằng `tasks/status`, `tasks/result`, và `tasks/list` trả về lỗi không tìm thấy phương thức (method-not-found) trong dịch vụ hiện đại.
Xác minh rằng `tools/list` mang tính tất định và mọi phương thức tác vụ HTTP hiện tại đều ánh xạ task id qua tiêu đề `Mcp-Name`.

## Đóng gói sản phẩm

Tệp `outputs/skill-task-store-designer.md` hiện cung cấp một thiết kế nhận biết phần mở rộng: đàm phán năng lực, tạo tác vụ bền vững trước khi trả về, các phương thức hiện tại, luồng cập nhật đầu vào, quyền sở hữu, hết hạn, hủy bỏ, đăng ký theo dõi, và quy trình chuyển đổi từ các phương thức thử nghiệm đã bị loại bỏ.

## Bài tập mở rộng

1. Thêm một khóa đầu vào thứ hai đang chờ xử lý. Gửi một bản cập nhật `tasks/update` từng phần và chứng minh rằng tác vụ vẫn duy trì trạng thái `input_required` cho đến khi cả hai khóa đều được trả lời.
2. Bổ sung quyền sở hữu theo đối tượng thuê (tenant) vào kho lưu trữ và từ chối một task id hợp lệ khi được cung cấp bởi một chủ thể xác thực không chính xác.
3. Bổ sung cơ chế worker lease kèm thời gian hết hạn. Chứng minh rằng hai phiên bản dịch vụ không thể đồng thời hoàn tất cùng một tác vụ.
4. Triển khai một bộ điều hợp SSE phản hồi của POST cho `subscriptions/listen`. Tuyệt đối không thêm GET, `Last-Event-ID`, hay tiêu đề phiên.
5. Triển khai dọn dẹp khi hết hạn. Phân biệt một tác vụ đã hết hạn với một task id sai định dạng mà không làm lộ sự tồn tại của dữ liệu giữa các đối tượng thuê khác nhau.

## Các thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa trong phần mở rộng hiện thời |
|---|---|
| Phần mở rộng Tasks (Tasks extension) | Năng lực tùy chọn `io.modelcontextprotocol/tasks` dành cho công việc bất đồng bộ bền vững |
| `CreateTaskResult` | Phản hồi dạng `resultType: "task"` do máy chủ điều hướng cho một yêu cầu đủ điều kiện |
| `tasks/get` | Thăm dò ảnh chụp nhanh hiện tại đầy đủ của tác vụ, bao gồm kết quả cuối cùng hoặc đầu vào đang chờ |
| `tasks/update` | Gửi phản hồi cho các `inputRequests` đang chờ xử lý của tác vụ |
| `tasks/cancel` | Xác nhận ý định hủy bỏ mang tính phối hợp |
| `input_required` | Trạng thái tác vụ chỉ ra rằng đang chờ dữ liệu đầu vào từ client |
| `pollIntervalMs` | Khoảng thời gian trễ tối thiểu do máy chủ gợi ý trước lần thăm dò tiếp theo |
| `ttlMs` | Thời lượng hết hạn được đo từ thời điểm tạo tác vụ |
| Bền vững trước khi trả về (Durable-before-return) | Quy tắc bắt buộc task id phải truy xuất được trước khi gửi thẻ điều khiển cho client |
| `notifications/tasks` | Ảnh chụp nhanh toàn bộ tác vụ tùy chọn được truyền tải trên luồng phản hồi SSE đã đăng ký |

## Khả năng tương thích kế thừa

Bề mặt thử nghiệm ngày 2025-11-25 từng sử dụng cơ chế bổ trợ tác vụ do client yêu cầu, cùng các phương thức `tasks/status`, `tasks/result`, và `tasks/list` tùy chọn. Chỉ giữ lại những tên gọi đó bên trong một bộ điều hợp kế thừa (legacy adapter) cố định. Một client hiện đại sử dụng năng lực phần mở rộng, chấp nhận các thẻ điều khiển do máy chủ chỉ định, thăm dò với `tasks/get`, cung cấp dữ liệu với `tasks/update`, và đọc kết quả cuối cùng trực tiếp từ ảnh chụp nhanh của tác vụ.

## Đọc thêm

- [Đặc tả chính thức của MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
