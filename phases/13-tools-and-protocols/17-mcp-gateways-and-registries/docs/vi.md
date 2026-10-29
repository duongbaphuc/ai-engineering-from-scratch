# Cổng kết nối MCP phi trạng thái và Tiếp nhận từ Sổ đăng ký

> Một cổng kết nối (gateway) cần làm cho mọi tuyến đường trở nên tường minh. Giao thức 2026-07-28 mang lại các ranh giới phương thức, tên gọi, phiên bản, năng lực, danh tính, bộ nhớ đệm và truy vết mà không cần đến phiên giao vận.

**Loại:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 13 · 15 (bảo mật), Phase 13 · 16 (ủy quyền)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Tổng hợp nhiều máy chủ MCP phía sau một điểm cuối duy nhất chuẩn 2026-07-28 mà không cần ràng buộc phiên (session affinity).
- Xác thực siêu dữ liệu theo từng yêu cầu và các tiêu đề định tuyến trước khi áp dụng chính sách hoặc chuyển tiếp.
- Hợp nhất các công cụ với không gian tên ổn định, thứ tự tất định, ghim băm bộ mô tả, RBAC, và lưu bộ nhớ đệm riêng tư (private caching).
- Xử lý các bản ghi sổ đăng ký (registry records) như bằng chứng khám phá vẫn đòi hỏi chính sách tiếp nhận chặt chẽ.
- Định tuyến chuẩn xác luồng SSE trong phạm vi yêu cầu, lệnh gọi `subscriptions/listen`, các lượt thử lại MRTR, và các lệnh gọi phần mở rộng Tasks.
- Cô lập hoàn toàn khả năng hỗ trợ phiên và bắt tay kiểu cũ khỏi lộ trình xử lý hiện đại.

## Vấn đề đặt ra

Việc kết nối một client trực tiếp tới một máy chủ thì rất đơn giản. Nhưng một môi trường triển khai quy mô lớn hơn đòi hỏi câu trả lời nhất quán cho những câu hỏi phức tạp hơn:

- Những máy chủ nào được phép hoạt động?
- Chủ thể nào có thể nhìn thấy và gọi từng công cụ?
- Điều gì xảy ra khi hai backend cùng phơi bày một tên gọi giống nhau?
- Các thay đổi trong bộ mô tả được xét duyệt như thế nào?
- Giới hạn tốc độ và sự kiện kiểm toán được áp dụng ở đâu?
- Liệu bất kỳ phiên bản nào của hệ thống cũng có thể xử lý yêu cầu tiếp theo hay không?

Một cổng kết nối (gateway) đứng giữa các client và các máy chủ MCP backend. Nó hiển thị một điểm cuối MCP duy nhất, áp dụng các chính sách dùng chung xuyên suốt, và chuyển tiếp các yêu cầu đã được phê duyệt.

Các thiết kế gateway kiểu cũ thường dồn kênh một phiên client vào nhiều phiên backend và viết lại tiêu đề `Mcp-Session-Id`. Đó là thiết kế dành riêng cho khả năng tương thích kế thừa. Giao thức lõi chuẩn 2026-07-28 hoàn toàn không có các phiên giao thức.

## Khái niệm cốt lõi

### Lộ trình xử lý hiện đại của Gateway

Đối với mỗi yêu cầu:

1. Xác thực chủ thể (principal) từ thông tin ủy quyền của tầng giao vận.
2. Xác thực các giá trị `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, và `params._meta`.
3. Ủy quyền cho chủ thể, tài nguyên, phương thức, công cụ và các đối số.
4. Áp dụng chính sách về bộ mô tả, sổ đăng ký, giới hạn tốc độ và dữ liệu.
5. Tạo một yêu cầu mới hoàn toàn tự chứa để gửi tới backend được chọn.
6. Xác thực kết quả từ backend và trả về kết quả gateway cho client.
7. Ghi lại sự kiện kiểm toán mà không ghi log các bí mật nhạy cảm.

Không có bước nào đòi hỏi phải duy trì một phiên giao thức ẩn. Trạng thái ứng dụng vẫn có thể tồn tại trong cơ sở dữ liệu, các thẻ điều khiển tường minh, Tasks, hoặc trạng thái MRTR được bảo vệ tính toàn vẹn.

### Chính sách thời gian chạy là quyết định trọng tâm của Gateway

Việc tiếp nhận (admission) quyết định phiên bản backend nào được phép bước vào gateway. Nó không tự động cấp quyền cho một lệnh gọi trực tiếp. Đối với mỗi yêu cầu, gateway tính toán lại chính sách dựa trên chủ thể đã xác thực, bên phát hành và tài nguyên, đối tượng thuê (tenant), phương thức và tên gọi tương ứng, các đối số đã chuẩn hóa, ghim bộ mô tả đã được chấp thuận, tình trạng sức khỏe hiện tại của backend, phần giao thoa năng lực, phân loại dữ liệu, trạng thái giới hạn tốc độ, và bất kỳ phê duyệt nào gắn liền với hành động.

Thứ tự xử lý này mang tính sống còn. Một bản ghi Registry có thể vẫn đang hoạt động trong khi vai trò của người dùng đã bị thu hồi. Một bộ mô tả có thể vẫn giữ nguyên mã ghim trong khi đối số đích đến lại vượt qua ranh giới đối tượng thuê. Một backend có thể vẫn được phê duyệt trong khi chính sách xử lý sự cố đang cách ly kiểm dịch các lệnh gọi làm thay đổi trạng thái. Do đó, chính sách thời gian chạy (runtime policy) là quyết định cho phép hoặc từ chối chính, với bằng chứng từ Registry và bộ mô tả đóng vai trò là các đầu vào tham chiếu.

Tuyệt đối không lưu đệm quyết định cho phép dưới một kết nối mạng hoặc một mã định danh phiên đã bị xóa bỏ. Nếu dịch vụ chính sách không khả dụng, hãy tuân theo chính sách xử lý lỗi đã công bố theo từng lớp thao tác. Một cấu hình mặc định an toàn là chặn hoàn toàn (fail closed) đối với các thay đổi trạng thái và thao tác đọc nhạy cảm, trong khi các lộ trình đọc công khai đã được phê duyệt rõ ràng có thể sử dụng chính sách được biết gần nhất trong thời gian ngắn chỉ khi mô hình rủi ro cho phép. Ghi nhận phiên bản chính sách và lộ trình xử lý lỗi nào đã đưa ra quyết định, sau đó xác thực kết quả từ backend trước khi trả về cho client.

### Một điểm cuối POST duy nhất

Giao vận Streamable HTTP hiện đại gửi mỗi thông điệp JSON-RPC qua phương thức POST:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

Gateway có thể trả về JSON hoặc luồng SSE trong phạm vi yêu cầu cho lệnh POST đó. Các phương thức GET và DELETE trả về mã lỗi 405 đối với các yêu cầu hiện đại. Tiêu đề `Mcp-Session-Id` và `Last-Event-ID` không hề tạo ra thẩm quyền, tính gắn kết phiên hay hành vi phát lại.

Các giá trị trong tiêu đề và thân thông điệp bắt buộc phải trùng khớp. Từ chối ngay lập tức sự sai lệch bằng mã lỗi `-32020` trước khi tra cứu backend. Điều này giúp các bộ cân bằng tải, gateway, và bộ giới hạn tốc độ có thể định tuyến mà không cần phân tích toàn bộ phần thân, đồng thời bảo toàn tính toàn vẹn đầu-cuối.

Xác thực theo đúng một trình tự duy nhất: kiểu JSON-RPC và kiểu siêu dữ liệu, sự trùng khớp giữa tiêu đề và thân, sau đó là sự hỗ trợ cho phiên bản tương ứng. Lệch tiêu đề trả về HTTP 400 với `-32020`. Nếu tiêu đề và thân cùng thống nhất về một phiên bản không được hỗ trợ, trả về HTTP 400 với `-32022` và `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}`. Phương thức không xác định trả về HTTP 404 với `-32601`.

Lớp `ProtocolError` mang trường `data` tùy chọn, và gateway sẽ tuần tự hóa nó vào đối tượng lỗi JSON-RPC. Một thông báo không có `id`, do đó nó không bao giờ nhận được phản hồi thành công hoặc lỗi JSON-RPC. Một thông báo HTTP được chấp nhận sẽ trả về mã 202 với phần thân rỗng.

### Triển khai khám phá tại mọi tầng

Gateway triển khai `server/discover` cho các client. Nó đồng thời cũng thực hiện khám phá từng backend để nắm bắt phiên bản giao thức, năng lực và các phần mở rộng.

Ví dụ về kết quả khám phá của gateway:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

Chỉ công bố phần giao thoa năng lực (capability intersection) mà gateway có thể đảm bảo thực thi trọn vẹn từ đầu đến cuối. Một tính năng của backend không tự động an toàn để phơi bày. Một tính năng của gateway nhưng không có đường dẫn hỗ trợ ở backend thì không có giá trị để công bố.

Trường `serverInfo` là dữ liệu hiển thị và chẩn đoán tự khai báo. Tuyệt đối không dùng nó làm bằng chứng sổ đăng ký hay bằng chứng nhà phát hành.

### Năng lực client theo từng yêu cầu

Mọi yêu cầu được chuyển tiếp đều cần một phong bì `_meta` hiện hành:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

Không sao chép một cách mù quáng năng lực client bên ngoài gửi tới backend. Gateway đóng vai trò là client của backend. Chỉ công bố những tính năng mà gateway sẽ điều phối chuẩn xác.

### Đặt không gian tên tất định

Hợp nhất các công cụ backend dưới các tên công khai ổn định:

```text
notes.search
notes.create
issues.list
issues.open
```

Duy trì một bảng ánh xạ từ tên công khai tới backend và tên công cụ gốc. Tuyệt đối không tùy tiện chọn công cụ trùng lặp đầu tiên hay cuối cùng. Một tên công khai là một phần của hợp đồng phê duyệt và kiểm toán, do đó việc thay đổi tên là một quá trình chuyển đổi di chuyển (migration).

Phương thức `tools/list` phải mang tính tất định. Khi danh sách hiển thị có sự khác biệt tùy theo chủ thể, hãy trả về `cacheScope: private`. Một giá trị `ttlMs` có giới hạn giúp giảm tải khám phá lên backend mà không làm rò rỉ danh sách đặc thù của người dùng xuyên qua các ngữ cảnh ủy quyền khác nhau.

Mọi bộ mô tả công cụ được phơi bày đều phải bao gồm tên ổn định, phần mô tả và lược đồ `inputSchema` có gốc là đối tượng. Việc đặt không gian tên không thể loại bỏ các trường mô tả bắt buộc này. Kết quả danh sách hoàn chỉnh cũng bao gồm `resultType`, siêu dữ liệu định danh máy chủ và các chỉ dẫn bộ nhớ đệm.

### Ghim băm các bộ mô tả đã được phê duyệt

Tại thời điểm tiếp nhận, hãy chuẩn tắc hóa toàn bộ bộ mô tả và lưu trữ mã băm của nó dưới tên công khai định danh đầy đủ. Tại thời điểm liệt kê danh sách và gọi thực thi, hãy so sánh bộ mô tả trực tiếp với mã băm đã được phê duyệt.

Nếu bộ mô tả thay đổi:

- Loại bỏ nó khỏi `tools/list`.
- Từ chối các lệnh gọi trực tiếp.
- Phát ra một sự kiện kiểm toán.
- Yêu cầu chính sách hoặc con người tái phê duyệt trước khi cập nhật mã ghim.

Gateway là một điểm thực thi trung tâm rất hữu ích, nhưng nó không biến một bộ mô tả mới gặp lần đầu thành một bộ mô tả an toàn. Việc xem xét đánh giá ban đầu luôn là bắt buộc.

### Sổ đăng ký hỗ trợ khám phá, không đưa ra quyết định

Một tệp `server.json` của Sổ đăng ký cung cấp siêu dữ liệu xuất bản. Một bản ghi dựa trên gói phần mềm có thể trông như sau:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

Siêu dữ liệu xuất bản không chứa quyết định an ninh của gateway. Hãy lưu trữ bằng chứng về nhà xuất bản đã xác minh và nguồn gốc xuất xứ (provenance) bên trong trạng thái tiếp nhận riêng biệt:

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

Gateway kiểm tra cấu trúc của `server.json` và liên kết nó với trạng thái bên ngoài đó. Gateway vẫn bắt buộc phải có chính sách tiếp nhận của riêng mình.

Đối với mỗi backend được tiếp nhận, hãy ghi nhận:

- Mã định danh bản ghi và sổ đăng ký chính xác.
- Bằng chứng tên miền hoặc không gian tên nhà xuất bản đã xác minh.
- Giao vận và điểm cuối được phép.
- Phiên bản được ghim hoặc chính sách nâng cấp đã phê duyệt.
- Mã băm của thành phẩm (artifact) hoặc bộ mô tả.
- Bên phát hành ủy quyền và tài nguyên.
- Người xét duyệt, thời gian phê duyệt và thời hạn hết hạn.

Không chấp nhận một máy chủ chỉ vì tên hiển thị của nó giống một sản phẩm quen thuộc. Không coi sự hiện diện trên sổ đăng ký là sự bảo đảm về mặt an ninh vận hành. Các máy chủ nội bộ riêng tư có thể được tiếp nhận thông qua cùng một lược đồ bằng chứng này ngay cả khi chúng không bao giờ xuất hiện trên một sổ đăng ký công khai.

Bài học này xây dựng ranh giới kết nối của gateway: ghép nối bằng chứng xuất bản với việc tiếp nhận cục bộ trước khi một backend có thể định tuyến được. [Bài học 30: Chuỗi cung ứng Sổ đăng ký MCP, Tiếp nhận, Sai lệch và Khôi phục](../../30-mcp-registry-supply-chain-and-drift/docs/vi.md) sẽ xây dựng toàn bộ mặt phẳng điều khiển (control plane) để xác thực không gian tên chính xác, nguồn gốc thành phẩm, các mã ghim bất biến, sai lệch bộ mô tả trực tiếp, đối soát trạng thái Sổ đăng ký, sổ cái tiếp nhận chống giả mạo, và cơ chế hoàn tác dựa trên bằng chứng. Hãy giữ trạng thái chuỗi cung ứng đó tách biệt khỏi quyết định thời gian chạy theo từng yêu cầu ở trên.

### Điều phối thông tin xác thực

Gateway xác thực những người gọi nó và xác thực riêng biệt với các backend. Thông tin xác thực của backend tuyệt đối không bao giờ được gửi tới client.

Duy trì các ràng buộc này một cách tường minh:

```text
chủ thể bên ngoài -> vai trò và chính sách gateway
bên phát hành backend + tài nguyên -> đăng ký và token backend
```

Tuyệt đối không chuyển token của gateway bên ngoài cho một backend. Tuyệt đối không tái sử dụng một token của backend tại một bên phát hành hoặc tài nguyên khác. Nếu một công cụ hành động thay mặt cho người dùng cuối, hãy duy trì sự ủy quyền đó bằng một mô hình trao đổi token hoặc xác nhận danh tính được thiết kế bài bản, thay vì mạo danh người dùng bằng một thông tin xác thực dịch vụ dùng chung.

### Giới hạn tốc độ không phụ thuộc phiên

Đặt khóa giới hạn theo chủ thể đã xác thực, bên phát hành, tài nguyên, công cụ công khai, lớp chi phí, và cửa sổ thời gian. Mã định danh phiên hoàn toàn vắng mặt và rất dễ bị thay đổi ngay cả khi nó tồn tại.

Áp dụng các bước xác thực chi phí thấp trước khi tiêu tốn tài nguyên cho các tác vụ nặng. Quyết định xem các lệnh gọi bị từ chối sẽ được tính vào giới hạn lạm dụng, hạn ngạch nghiệp vụ, hay cả hai.

### Kiểm toán chuỗi quyết định

Ghi lại đầy đủ thông tin để có thể tái dựng lại một lệnh gọi:

- Mã định danh yêu cầu và truy vết (trace identifiers).
- Chủ thể đã xác thực và bên phát hành.
- Công cụ công khai và tuyến đường backend.
- Phiên bản ghim bộ mô tả.
- Quyết định chính sách và lý do.
- Độ trễ và phân loại kết quả.
- Vòng lặp MRTR hoặc mã định danh tác vụ khi áp dụng.

Loại bỏ các bearer token, mã ủy quyền, refresh token, các bí mật thô, và các đối số nhạy cảm không cần thiết.

### Luồng SSE trong phạm vi yêu cầu

Một lệnh POST thông thường có thể trả về một luồng SSE trong phạm vi yêu cầu khi dữ liệu được truyền tải liên tục trong suốt yêu cầu đó. Việc đóng luồng phản hồi sẽ hủy bỏ yêu cầu HTTP hiện đại đang xử lý đó.

Không tạo một luồng GET riêng biệt và không hứa hẹn khả năng phát lại qua Last-Event-ID. Đó là các giả định giao vận kiểu cũ.

### Thông báo thay đổi dài hạn

Đối với các thông báo thay đổi danh sách và tài nguyên, một client hiện đại gửi `subscriptions/listen` qua POST và nhận về một luồng phản hồi SSE. Các bộ lọc thông báo sử dụng chính xác các trường phẳng `toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, và `resourceSubscriptions`:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Sự kiện đầu tiên xác nhận tập con được hỗ trợ. Mã định danh đăng ký của nó chính là id JSON-RPC của yêu cầu đã mở luồng:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

Gateway sau đó chỉ chuyển tiếp các loại thay đổi đã được xác nhận. Mọi thông báo trên luồng đó đều mang cùng giá trị `io.modelcontextprotocol/subscriptionId` trong `params._meta`. Hoàn toàn không có cơ chế phát lại tự động hay tự động lắng nghe lại. Khi kết nối lại, client sẽ mở lại đăng ký và làm mới các danh sách mà nó phụ thuộc. Một lệnh đóng luồng nhẹ nhàng do máy chủ khởi tạo sẽ trả về kết quả hoàn tất cuối cùng được gắn thẻ cùng id đăng ký đó.

Lộ trình hiện đại thay thế hoàn toàn `resources/subscribe`, `resources/unsubscribe`, và cơ chế luồng GET độc lập không được yêu cầu. Chỉ giữ lại những thứ đó bên trong lộ trình kiểu cũ được kiểm soát bằng phiên bản.

### Xử lý MRTR qua Gateway

Khi một backend trả về `resultType: input_required`, gateway chỉ có thể chuyển tiếp kết quả đó nếu client bên ngoài có hỗ trợ yêu cầu đầu vào tương ứng. Bảo toàn nguyên vẹn từng byte của `requestState` trừ khi gateway chủ động chấm dứt và phát hành lại phiên tương tác.

Client thử lại công cụ công khai ban đầu với một id JSON-RPC mới và `inputResponses`. Gateway ủy quyền lại cho lượt thử này, kiểm tra cùng tuyến đường công khai, sau đó chuyển tiếp một yêu cầu backend mới. Gateway tuyệt đối không được giả định rằng một vòng trước đó đã cấp quyền phê duyệt vô hạn định.

### Định tuyến phần mở rộng Tasks

Tasks là một phần mở rộng chính thức được định danh bởi `io.modelcontextprotocol/tasks`. Chúng không phải là cơ chế thay thế phiên lõi.

Client khai báo phần mở rộng bên trong các năng lực client theo từng yêu cầu, và gateway chỉ công bố nó trong phát hiện dịch vụ khi nó có thể bảo toàn trọn vẹn vòng đời từ đầu đến cuối. Đối với một lệnh `tools/call` được hỗ trợ, duy nhất backend mới có quyền quyết định trả về kết quả thông thường hay `resultType: task`. Kết quả tác vụ mang `taskId`, `status`, các dấu thời gian, `ttlMs`, và giá trị tùy chọn `pollIntervalMs` trực tiếp trong kết quả. Tác vụ bắt buộc phải đọc được bền vững trước khi kết quả đó được gửi đi.

Gateway ghi nhận chủ thể đã xác thực và tuyến đường backend cho mã định danh tác vụ mờ đục này. Các lệnh gọi `tasks/get`, `tasks/update`, và `tasks/cancel` sau đó sử dụng `params.taskId` làm tiêu đề `Mcp-Name`, mang lại cho các thành phần trung gian một khóa định tuyến. `tasks/get` trả về `resultType: complete` cùng trạng thái tác vụ hiện tại và gắn kết quả cuối cùng hoặc lỗi giao thức trong trạng thái kết thúc. `tasks/update` gửi `inputResponses` có khóa cho đầu vào đang chờ của tác vụ và trả về thông điệp xác nhận hoàn tất rỗng. `tasks/cancel` là ý định mang tính phối hợp kèm thông điệp xác nhận hoàn tất rỗng, không phải là sự đảm bảo rằng công việc đã dừng hoàn toàn.

Không triển khai các phương thức mới `tasks/list` hay `tasks/result`. Chúng thuộc về mô hình thử nghiệm cũ hơn. Một tác vụ cần đầu vào sẽ phơi bày các yêu cầu nhúng hoàn chỉnh qua `tasks/get`; client trả lời chúng qua `tasks/update`, chứ không phải bằng cách thử lại lệnh gọi công cụ ban đầu. Client vẫn thăm dò theo khoảng thời gian gợi ý; việc tạo tác vụ vẫn do máy chủ điều phối.

Trạng thái tuyến đường tác vụ bền vững là dữ liệu ứng dụng được định danh bởi thẻ điều khiển tác vụ, không phải là một phiên giao thức.

### Ranh giới tương thích

Nếu gateway bắt buộc phải phục vụ một client hoặc backend kiểu cũ:

- Nhận diện kỷ nguyên giao thức một cách tường minh.
- Giữ quá trình khởi tạo, các phiên giao vận, luồng GET, đăng ký tài nguyên và các từ vựng tác vụ cũ bên trong một bộ điều hợp kế thừa (legacy adapter).
- Tuyệt đối không để rò rỉ mã phiên kiểu cũ vào quá trình định tuyến hoặc ủy quyền hiện đại.
- Ưu tiên cơ chế thăm dò khám phá có giới hạn và chính sách chuyển tiếp rõ ràng thay vì âm thầm hạ cấp phiên bản.

````figure
t3-gateway-funnel
````

## Xây dựng hệ thống

Tệp `code/main.py` triển khai một gateway giao thức chạy nội tiến trình cùng hai máy chủ backend. Mỗi backend tiếp nhận một yêu cầu độc lập theo giao thức hiện hành. Gateway cung cấp phương thức khám phá, `tools/list` tất định được lọc theo người dùng, định tuyến theo không gian tên, tệp `server.json` của Registry kèm trạng thái tiếp nhận bên ngoài, ghim bộ mô tả, RBAC, giới hạn tốc độ theo khóa chủ thể, các quyết định kiểm toán, và phản hồi xác nhận SSE mô phỏng của `subscriptions/listen`.

Mô hình tiếp nhận các phần thân yêu cầu đã phân tích cú pháp, tiêu đề định tuyến, và danh tính bearer đã xác thực. Đây không phải là một bộ điều hợp HTTP hoàn chỉnh và không trực tiếp phân tích `Content-Type` hay hợp đồng `Accept` đầy đủ. Hãy kết nối nó với bộ điều hợp Streamable HTTP của Bài học 09, vốn yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy chương trình:

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Bản demo in ra id yêu cầu bên ngoài và id yêu cầu backend hoàn toàn mới để làm nổi bật chặng nhảy phi trạng thái (stateless hop).

## Thử nghiệm thực tế

Thay thế các đối tượng backend nội tiến trình bằng các client giao thức hiện tại thực tế. Duy trì các ranh giới kiến trúc tương tự:

- Bản ghi tiếp nhận trước khi kết nối.
- Khám phá backend trước khi phơi bày năng lực.
- Tên công khai định danh đầy đủ trước khi ủy quyền.
- Ghim bộ mô tả trước khi liệt kê hoặc gọi thực thi.
- Siêu dữ liệu theo từng yêu cầu mới trước khi chuyển tiếp.
- Xác thực kết quả trước khi gửi trả lại.

## Đóng gói sản phẩm

Bài học này đóng gói tệp `outputs/skill-gateway-bootstrap.md`. Tài liệu này tạo ra một bản thiết kế gateway hiện đại bao quát các mặt: cổng vào (ingress), khám phá, tiếp nhận, không gian tên, ủy quyền, bộ nhớ đệm, luồng dữ liệu, đăng ký theo dõi, MRTR, Tasks, khả năng quan sát và cách ly hệ thống kế thừa.

## Bài tập mở rộng

1. Bổ sung ngữ cảnh truy vết (trace context) vào siêu dữ liệu yêu cầu bên ngoài và chuyển tiếp, sau đó ghi lại mối tương quan này trong sự kiện kiểm toán.
2. Thêm một backend có khả năng xử lý Tasks và định tuyến `tasks/get` theo task id trong `Mcp-Name`.
3. Thay đổi một bộ mô tả backend và chứng minh rằng cả bước khám phá lẫn lệnh gọi trực tiếp đều bị chặn lại.
4. Thêm một năng lực máy chủ đặc thù cho từng chủ thể và giải thích tại sao kết quả khám phá phải duy trì lưu đệm riêng tư.
5. Viết một giao diện bộ điều hợp kế thừa mà không thêm bất kỳ trạng thái kế thừa nào vào lớp `Gateway` hiện đại.

## Các thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| Cổng kết nối MCP (MCP gateway) | Máy chủ chính sách và định tuyến đứng giữa các client và các máy chủ MCP backend |
| Bản ghi tiếp nhận (Admission record) | Quyết định chính sách và bằng chứng cho phép một backend tham gia vào gateway |
| Tên công cụ định danh đầy đủ (Qualified tool name) | Tuyến đường công khai ổn định như `notes.search` |
| Ghim bộ mô tả (Descriptor pin) | Mã băm được phê duyệt được kiểm tra trong lúc khám phá và điều phối |
| Phạm vi lưu đệm riêng tư (Private cache scope) | Kết quả lưu đệm chỉ giới hạn trong một ngữ cảnh ủy quyền duy nhất |
| Luồng SSE trong phạm vi yêu cầu (Request-scoped SSE) | Phản hồi dạng luồng được gắn liền với một yêu cầu POST duy nhất |
| `subscriptions/listen` | Luồng SSE do client mở dành riêng cho các thông báo thay đổi dài hạn được chọn |
| Tuyến đường tác vụ (Task route) | Ánh xạ cấp ứng dụng từ một task id mờ đục tới backend sở hữu nó |
| Bộ điều hợp kế thừa (Legacy adapter) | Ranh giới kiểm soát phiên bản tường minh dành cho hành vi bắt tay và phiên làm việc kiểu cũ |

## Đọc thêm

- [Giao vận Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Khám phá máy chủ](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Yêu cầu server.json của Sổ đăng ký chính thức](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Phần mở rộng MCP Tasks](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
