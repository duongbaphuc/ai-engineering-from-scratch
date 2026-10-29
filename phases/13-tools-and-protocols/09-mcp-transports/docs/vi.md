# Các Phương thức Truyền tải MCP: stdio và Streamable HTTP Không lưu trạng thái

> Phương thức truyền tải (transport) làm nhiệm vụ vận chuyển các thông điệp MCP. Nó không bù đắp cho trạng thái giao thức bị thiếu. Trong đặc tả `2026-07-28`, cả stdio cục bộ và Streamable HTTP từ xa đều mang theo các yêu cầu tự mô tả (self-describing requests).

**Type:** Học tập (Learn)
**Languages:** Python
**Prerequisites:** Phase 13, Bài học 07 và 08
**Time:** ~65 phút

## Mục tiêu học tập

- Lựa chọn stdio cho các tiến trình con cục bộ và Streamable HTTP cho các dịch vụ mạng.
- Triển khai hợp đồng Streamable HTTP hiện đại chỉ dùng POST với một điểm cuối duy nhất.
- Phản chiếu và xác thực các tiêu đề phiên bản MCP, phương thức và tên đối chiếu với phần thân (body) JSON-RPC.
- Phân phối SSE trong phạm vi yêu cầu (request-scoped SSE) và các luồng `subscriptions/listen` tồn tại lâu dài một cách chính xác.
- Di chuyển các hệ thống triển khai dựa trên phiên (session-based) và HTTP+SSE kế thừa mà không biểu diễn hành vi cũ như là chuẩn hiện đại.

## Vấn đề

Các bản sửa đổi Streamable HTTP trước đây đã kết hợp việc đàm phán giao thức với hành vi kết nối và phiên làm việc. Một máy chủ có thể tự phát hành tiêu đề `Mcp-Session-Id`, công bố một luồng GET độc lập, chấp nhận lệnh DELETE để chấm dứt phiên, và tiếp tục lại luồng SSE bằng `Last-Event-ID`.

Đặc tả MCP `2026-07-28` loại bỏ hoàn toàn các cơ chế đó khỏi đường truyền hiện đại. Mọi yêu cầu đều có thể hạ cánh xuống bất kỳ tiến trình worker khỏe mạnh nào, bởi vì phiên bản giao thức và khả năng của máy khách đều được truyền tải ngay trong phần thân của yêu cầu. Các tiêu đề HTTP đóng vai trò phản chiếu lại một số trường được chọn nhằm phục vụ mục đích điều hướng và áp dụng chính sách, nhưng máy chủ luôn phải xác thực các tiêu đề đó với phần thân trước khi tiến hành thực thi.

Kết quả mang lại là một hệ thống dễ mở rộng quy mô hơn và dễ lập luận hơn rất nhiều. Điều này cũng có nghĩa là một máy chủ nếu vẫn giảng dạy mô hình truyền tải năm 2025 như là chuẩn mực hiện tại thì thực chất đang truyền tải sai lầm về mô hình bảo mật và mô hình xử lý lỗi.

## Khái niệm

### stdio

Ràng buộc stdio được thiết kế cho một tiến trình con (subprocess) do máy khách khởi chạy:

- Máy khách ghi một thông điệp JSON-RPC định dạng UTF-8 trên mỗi dòng vào stdin.
- Máy chủ ghi một thông điệp JSON-RPC định dạng UTF-8 trên mỗi dòng ra stdout.
- Máy chủ ghi các thông tin chẩn đoán ra stderr.
- Máy chủ thoát ngay lập tức khi stdin gặp ký tự kết thúc tệp (EOF).
- Mọi yêu cầu hiện đại đều mang phiên bản và khả năng của máy khách trong `params._meta`.

Tiến trình có thể tồn tại qua nhiều lệnh gọi, nhưng nó không phải là một phiên giao thức hiện đại. Nếu tiến trình thoát đột ngột, các yêu cầu đang xử lý dở dang (in-flight) sẽ bị mất. Hãy khởi động lại tiến trình, khám phá lại, liệt kê lại, mở lại các đăng ký lắng nghe và thử lại các thao tác an toàn với các id yêu cầu mới.

### Streamable HTTP trong đặc tả 2026-07-28

Một máy chủ hiện đại công bố một điểm cuối MCP duy nhất, ví dụ `/mcp`, và chỉ chấp nhận phương thức POST.

Mọi yêu cầu hoặc thông báo JSON-RPC đều là một lệnh HTTP POST mới. Phần thân chứa một thông điệp JSON-RPC duy nhất. Máy khách không gửi các phản hồi JSON-RPC ngược lại cho máy chủ.

Đối với một yêu cầu, máy chủ sẽ trả về một trong hai định dạng:

- `Content-Type: application/json` với một phản hồi JSON-RPC duy nhất; hoặc
- `Content-Type: text/event-stream` với các thông báo liên quan đến yêu cầu đó, theo sau là phản hồi JSON-RPC cuối cùng.

Đối với một thông báo được chấp nhận, máy chủ trả về mã HTTP `202 Accepted` và không kèm phần thân.

Máy khách khai báo hỗ trợ cả hai loại phản hồi:

```http
Accept: application/json, text/event-stream
```

### Chỉ dùng POST nghĩa là tuyệt đối chỉ dùng POST

Streamable HTTP hiện đại không có luồng GET độc lập và không có điểm cuối DELETE để hủy phiên:

- `GET /mcp` trả về mã lỗi `405 Method Not Allowed`.
- `DELETE /mcp` trả về mã lỗi `405 Method Not Allowed`.
- `Mcp-Session-Id` bị bỏ qua và không bao giờ được tạo mới hay phản hồi lại.
- `Last-Event-ID` bị bỏ qua vì các luồng hiện đại không hỗ trợ khả năng tiếp tục lại.

Nếu một luồng trong phạm vi yêu cầu bị đứt trước khi có phản hồi cuối cùng, máy khách coi như đã mất yêu cầu đang xử lý dở dang đó. Nó có thể phát hành một yêu cầu mới với id JSON-RPC mới khi thao tác an toàn để thử lại. Tuyệt đối không được cố gắng khôi phục lại luồng cũ.

### Xác thực Origin

Các máy chủ phải xác thực tiêu đề `Origin` trên các kết nối gửi đến để ngăn chặn tấn công liên kết lại DNS (DNS rebinding). Nếu tiêu đề có mặt và không nằm trong danh sách cho phép rõ ràng, hãy trả về mã `403 Forbidden`. Một máy khách không phải trình duyệt có thể bỏ qua tiêu đề `Origin`, điều này được các quy tắc truyền tải chính thức cho phép.

Các máy chủ cục bộ nên liên kết (bind) vào địa chỉ `127.0.0.1`, không phải mọi giao diện mạng (`0.0.0.0`). Các dịch vụ mạng vẫn bắt buộc phải có cơ chế xác thực danh tính và ủy quyền trên từng yêu cầu. Xác thực Origin không phải là xác thực danh tính.

Sử dụng phép so khớp Origin chính xác sau khi đã chuẩn hóa cấu hình. Việc kiểm tra tiền tố như `origin.startswith("https://trusted.example")` là không an toàn vì kẻ tấn công có thể kiểm soát phần hậu tố phía sau.

### Các tiêu đề siêu dữ liệu HTTP bắt buộc

Mọi yêu cầu POST hiện đại đều bao gồm:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Các quy tắc về tiêu đề:

- `MCP-Protocol-Version` là bắt buộc và phải bằng với `params._meta.io.modelcontextprotocol/protocolVersion`.
- `Mcp-Method` là bắt buộc và phải bằng với `method` của JSON-RPC.
- `Mcp-Name` là bắt buộc đối với `tools/call`, `resources/read`, và `prompts/get`.
- `Mcp-Name` bằng với `params.name`, hoặc `params.uri` đối với `resources/read`.
- Giá trị tiêu đề có phân biệt chữ hoa chữ thường mặc dù tên tiêu đề HTTP không phân biệt hoa thường.

Các giá trị `Mcp-Name` không an toàn hoặc không thuộc bảng mã ASCII phải sử dụng chính xác tiền tố/hậu tố tuần tự Base64 theo chuẩn UTF-8:

```text
=?base64?{Base64EncodedValue}?=
```

Máy chủ sẽ giải mã giá trị này trước khi tiến hành so sánh với phần thân.

Các tiêu đề phản chiếu bị thiếu, sai định dạng hoặc không khớp sẽ trả về mã HTTP `400` với mã lỗi JSON-RPC `-32020`. Nếu tiêu đề và phần thân đồng thuận trên một phiên bản mà máy chủ không hỗ trợ, trả về HTTP `400` cùng mã `-32022` và dữ liệu lỗi chính xác như `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.

Một phương thức hiện đại không tồn tại sẽ trả về mã HTTP `404` kèm mã JSON-RPC `-32601`. Phần thân JSON-RPC là rất quan trọng vì máy khách hỗ trợ đa thế hệ sẽ sử dụng nó để phân biệt lỗi của máy chủ hiện đại với việc gọi trượt một điểm cuối kế thừa.

### SSE trong phạm vi yêu cầu

Một máy chủ có thể chọn SSE cho một yêu cầu chạy dài:

```text
POST tools/call id=41
  <- notifications/progress liên quan đến id=41
  <- notifications/progress liên quan đến id=41
  <- phản hồi JSON-RPC id=41
luồng đóng lại
```

Máy chủ tuyệt đối không được gửi các yêu cầu JSON-RPC độc lập trên luồng này. Các tương tác lấy mẫu (sampling), gợi mở phản hồi (elicitation) và lấy thư mục gốc (roots) phải sử dụng kết quả Yêu cầu nhiều lượt khứ hồi (Multi Round-Trip Request). Việc đóng luồng phản hồi đồng nghĩa với việc hủy yêu cầu đó.

Không thêm id sự kiện SSE để phục vụ việc phát lại. Cơ chế tiếp tục luồng bằng `Last-Event-ID` không nằm trong đặc tả hiện đại.

### Các thay đổi tồn tại lâu dài sử dụng subscriptions/listen

Các thông báo thay đổi sử dụng một yêu cầu do máy khách mở, không dùng lệnh GET độc lập:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Phản hồi POST là một luồng SSE tồn tại lâu dài. Thông điệp giao thức đầu tiên của nó là `notifications/subscriptions/acknowledged`. Thông báo xác nhận này, mọi thông báo thay đổi và kết quả cuối cùng đều mang `io.modelcontextprotocol/subscriptionId` trong `_meta`, có giá trị bằng với id của yêu cầu lắng nghe. Máy chủ có thể phát các bình luận SSE để duy trì kết nối (keepalive). Khi luồng bị ngắt, máy khách sẽ phát hành lại `subscriptions/listen` với một id yêu cầu mới và tìm nạp lại dữ liệu bị ảnh hưởng.

`resources/subscribe` và `resources/unsubscribe` thuộc về thế hệ cũ. Tuyệt đối không sử dụng chúng trên một kết nối hiện đại.

### Trạng thái ứng dụng tường minh

Loại bỏ các phiên giao thức không có nghĩa là cấm các quy trình làm việc có trạng thái (stateful workflows). Máy chủ có thể tự phát hành một thẻ định danh trạng thái ẩn (opaque state handle) và trả về nó như một kết quả công cụ bình thường. Máy khách sẽ truyền thẻ định danh đó như một đối số tường minh trong các lệnh gọi tiếp theo.

Ràng buộc các thẻ định danh với chủ thể đã xác thực (authenticated principal), làm cho chúng không thể đoán được, đặt hạn sử dụng (TTL) cho chúng và kiểm tra ủy quyền trong mỗi lần sử dụng. Điều này làm cho trạng thái hiển thị rõ ràng ở tầng ứng dụng thay vì che giấu nó trong tính bám dính của phương thức truyền tải (transport affinity).

Lỗi phát sinh do trạng thái bản sao (replica state) ẩn giấu là vấn đề thuần cơ học:

1. Yêu cầu A đến bản sao 1 và tạo một bản nháp trong bộ nhớ của tiến trình đó.
2. Phản hồi không trả về thẻ định danh bản nháp vì mã nguồn ngầm định kết nối sẽ tự xác định bản nháp.
3. Yêu cầu B là một lệnh POST mới và được chuyển đến bản sao 2.
4. Bản sao 2 có siêu dữ liệu giao thức hợp lệ nhưng không có cách nào gọi tên hoặc tải bản nháp đó, dẫn đến quy trình bị lỗi hoặc đọc nhầm đối tượng cục bộ khác.
5. Việc điều hướng bám dính (sticky routing) dường như sửa được triệu chứng này cho đến khi một đợt khởi động lại, triển khai mã mới, lập lịch lại hoặc chuyển đổi dự phòng (failover) chuyển yêu cầu tiếp theo sang nút khác.

Ranh giới đúng đắn bao gồm hai phần. Ngữ cảnh giao thức nằm trong từng yêu cầu. Trạng thái ứng dụng bền vững nằm trong kho lưu trữ dùng chung dưới một thẻ định danh do máy chủ tạo ra và trả về cho máy khách. Lệnh gọi tiếp theo cung cấp thẻ định danh đó, bất kỳ bản sao nào cũng có thể tải cùng một bản ghi, và hệ thống ủy quyền sẽ ràng buộc bản ghi đó với chủ thể và đối tượng thuê (tenant) đã xác thực. Bộ nhớ bản sao có thể lưu đệm một bản ghi, nhưng nó không thể là bản sao duy nhất phục vụ tính đúng đắn.

Hãy chọn cơ chế trạng thái dựa trên vòng đời. Các biến cục bộ trong yêu cầu có thể phục vụ một lệnh gọi. Một bước tiếp nối ngắn của MRTR có thể sử dụng `requestState` được bảo vệ tính toàn vẹn. Một bản nháp hoặc tác vụ bền vững cần một thẻ định danh tường minh kết hợp với bộ lưu trữ dùng chung, thời gian hết hạn, kiểm soát đồng thời và tính lũy quyền (idempotency). Không có đối tượng nào trong số đó là một phiên giao thức MCP.

### Tính tương thích HTTP đa thế hệ

Một máy khách hỗ trợ cả máy chủ hiện đại và máy chủ cũ sẽ luôn thử lệnh POST hiện đại trước tiên. Nếu nhận được mã lỗi HTTP `400`, `404`, hoặc `405`, nó sẽ kiểm tra phần thân phản hồi:

- Một lỗi JSON-RPC hiện đại được nhận diện chứng minh máy chủ thuộc thế hệ hiện đại. Hãy sửa lại yêu cầu hoặc thử lại với một phiên bản đã được công bố. Tuyệt đối không hạ cấp.
- Một phần thân rỗng hoặc một phản hồi không nhận diện được có thể là dấu hiệu của một máy chủ HTTP+SSE cũ. Chỉ khi đó mới thử điểm cuối GET cũ và chờ đợi sự kiện `endpoint` kế thừa của nó.

Một máy chủ có thể hỗ trợ cả hai thế hệ trong giai đoạn chuyển đổi bằng cách điều hướng siêu dữ liệu hiện đại đến bản triển khai chỉ dùng POST hiện đại, đồng thời duy trì các điểm cuối kế thừa tách biệt cho các máy khách cũ. Tuyệt đối không bao giờ mô tả các hành vi GET, DELETE, id phiên, hoặc phát lại kế thừa như là một phần của đặc tả `2026-07-28`.

```figure
tp-transport-handshake
```

## Sử dụng

Tệp `code/main.py` triển khai một máy chủ Streamable HTTP hiện đại, hữu hạn bằng thư viện chuẩn của Python. Nó xác thực Origin và các tiêu đề phản chiếu, bỏ qua các tiêu đề phiên đã bị xóa bỏ, trả về JSON cho các lệnh gọi thông thường, và minh họa một luồng SSE `subscriptions/listen` hữu hạn.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

Lệnh thăm dò kiểm tra:

- tiêu đề Origin không hợp lệ bị từ chối;
- khám phá thành công mà không cần id phiên;
- `Mcp-Session-Id` và `Last-Event-ID` bị bỏ qua;
- tiêu đề không khớp trả về lỗi `-32020`;
- phiên bản không được hỗ trợ trả về `-32022` với dữ liệu chính xác của `supported` và `requested`;
- một thông báo không có id được chấp nhận trả về HTTP `202` không có phần thân;
- các phương thức GET và DELETE trả về `405`;
- `subscriptions/listen` là một luồng phản hồi POST mà thông báo xác nhận, các thông báo thay đổi và kết quả cuối cùng đều mang id đăng ký của nó.

## Đóng gói / Triển khai

Bài học này cung cấp `outputs/skill-mcp-transport-migrator.md`. Nó loại bỏ các phiên giao thức hiện đại, bổ sung xác thực giữa tiêu đề và phần thân, thay thế lệnh GET độc lập bằng `subscriptions/listen`, và giữ cho bất kỳ cầu nối kế thừa nào luôn tách biệt rõ ràng.

## Bài tập

1. Xóa tiêu đề `Mcp-Method` khỏi một yêu cầu POST. Xác nhận máy chủ trả về HTTP `400` và lỗi `-32020`.
2. Gửi phiên bản `2027-01-01` khớp nhau ở cả tiêu đề và phần thân. Xác nhận nhận được HTTP `400`, lỗi `-32022`, và dữ liệu chính xác `{"supported":["2026-07-28"],"requested":"2027-01-01"}`.
3. Gửi tiêu đề `Mcp-Name` với định dạng Base64 cho một URI tài nguyên chứa ký tự non-ASCII. Xác nhận giá trị đã giải mã được so sánh chính xác với `params.uri`.
4. Làm đứt luồng lắng nghe hữu hạn trước phản hồi cuối cùng của nó. Phát hành lại luồng với một id JSON-RPC mới và tìm nạp lại danh sách công cụ.
5. Thêm một thẻ định danh quy trình công việc tường minh vào công cụ ping. Ràng buộc nó với một chủ thể ủy quyền mà không sử dụng cơ chế bám dính kết nối.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| stdio | Giao thức JSON-RPC phân cách bằng dòng mới truyền qua một tiến trình con do máy khách khởi chạy |
| Streamable HTTP | Điểm cuối duy nhất nơi mỗi thông điệp hiện đại là một lệnh POST mới |
| SSE trong phạm vi yêu cầu (Request-scoped SSE) | Luồng phản hồi POST chứa các thông báo liên quan và phản hồi cuối cùng |
| `subscriptions/listen` | Yêu cầu POST tồn tại lâu dài để nhận các thông báo thay đổi đã đăng ký |
| Không khớp tiêu đề (Header mismatch) | Mã HTTP `400` và JSON-RPC `-32020` khi các tiêu đề phản chiếu không khớp với phần thân |
| Xác thực Origin (Origin validation) | Cơ chế phòng thủ chống tấn công liên kết lại DNS cho các kết nối gửi đến, không phải xác thực danh tính |
| Thẻ định danh trạng thái tường minh (Explicit state handle) | Mã thông báo ứng dụng được truyền như một đối số thông thường thay vì trạng thái phiên ẩn |
| Cầu nối kế thừa (Legacy bridge) | Hành vi thuộc thế hệ cũ được duy trì tách biệt chỉ nhằm mục đích tương thích ngược |

## Đọc thêm

- [Tổng quan về Phương thức Truyền tải MCP (MCP Transport Overview)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [Phương thức truyền tải stdio trong MCP (MCP stdio Transport)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [Streamable HTTP trong MCP (MCP Streamable HTTP)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Cơ chế Đăng ký luồng trong MCP (MCP Subscriptions)](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [Nhật ký thay đổi MCP 2026-07-28 (MCP 2026-07-28 Changelog)](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
