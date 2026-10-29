# Nền tảng MCP: Yêu cầu phi trạng thái và JSON-RPC

> MCP hiện đại không có bước bắt tay (handshake) và không có phiên giao thức (protocol session). Mỗi yêu cầu bắt buộc phải mang theo đầy đủ siêu dữ liệu (metadata) để tự nó có thể được thông hiểu, ủy quyền, định tuyến và thử lại một cách độc lập.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 phút

## Mục tiêu học tập

- Phân biệt các cấu tử nguyên thủy phía server của MCP với các tính năng phía client.
- Xây dựng các yêu cầu và phản hồi JSON-RPC 2.0 hợp lệ theo đặc tả MCP `2026-07-28`.
- Gắn phiên bản giao thức (protocol version), năng lực của client (client capabilities) và danh tính client (client identity) vào mỗi yêu cầu.
- Sử dụng phương thức `server/discover` và xử lý lỗi `UnsupportedProtocolVersionError` mà không cần bước bắt tay.
- Truy vết một yêu cầu độc lập từ bước xác thực cho đến khi trả về kết quả hoàn chỉnh.

## Vấn đề đặt ra

Một server MCP có thể nhận hai yêu cầu liên tiếp từ các client khác nhau, với các bộ năng lực khác nhau, trên cùng một tiến trình hoặc một worker HTTP. Nếu server ghi nhớ những gì mà yêu cầu trước đó đã khai báo, nó có thể áp dụng sai quyền hạn hoặc trả về sai cấu trúc truyền tải dữ liệu (wire shape).

Phiên bản MCP `2026-07-28` loại bỏ hoàn toàn sự mơ hồ đó. Phần lõi của giao thức là phi trạng thái (stateless). Server bắt buộc phải quyết định cách xử lý yêu cầu hiện tại chỉ dựa trên chính yêu cầu hiện tại, chứ không dựa trên lịch sử kết nối.

Điều này làm thay đổi mô hình tư duy. Trình tự cũ trước đây là: kết nối trước, bắt tay thứ hai, thực hiện các thao tác thứ ba. Trình tự hiện đại đơn giản hơn nhiều:

1. Client gửi một yêu cầu tự mô tả (self-describing request).
2. Server xác thực phiên bản và năng lực của yêu cầu đó.
3. Server xử lý phương thức (method).
4. Server trả về một kết quả có định kiểu hoặc một thông báo lỗi JSON-RPC.

Yêu cầu tiếp theo sẽ lặp lại toàn bộ quy trình này từ đầu một cách độc lập.

## Khái niệm cốt lõi

### Các cấu tử nguyên thủy phía server (Server primitives)

Các server MCP phơi bày ba cấu tử nguyên thủy cốt lõi:

1. **Công cụ (Tools)** là các hành động do mô hình điều khiển, được khám phá thông qua `tools/list` và được gọi thông qua `tools/call`.
2. **Tài nguyên (Resources)** là dữ liệu được định địa chỉ bằng URI, được khám phá thông qua `resources/list` và được truy xuất thông qua `resources/read`.
3. **Lời nhắc (Prompts)** là các mẫu có thể tái sử dụng, được khám phá thông qua `prompts/list` và được kết xuất thông qua `prompts/get`.

Các thành phần thư mục gốc (roots), lấy mẫu (sampling) và ghi nhật ký (logging) vẫn được giữ lại trong lược đồ `2026-07-28` nhằm tương thích ngược, nhưng chúng đã bị gắn nhãn không khuyến nghị dùng (deprecated). Các bản hiện thực hóa mới nên sử dụng các đầu vào công cụ hoặc tài nguyên tường minh thay cho roots, sử dụng trực tiếp các API của nhà cung cấp mô hình cho sampling, và dùng stderr hoặc OpenTelemetry cho logging. Việc khơi gợi phản hồi (elicitation) vẫn khả dụng thông qua Yêu cầu Nhiều Lượt Khứ Hồi (Multi Round-Trip Requests), trong đó server trả về một yêu cầu nhập liệu và client sẽ thử lại thao tác ban đầu. Một server hiện đại không bao giờ tự ý khởi tạo một yêu cầu JSON-RPC độc lập.

### Lớp vỏ bao bọc JSON-RPC (JSON-RPC envelopes)

MCP sử dụng chuẩn JSON-RPC 2.0:

- Yêu cầu (Request): `{jsonrpc, id, method, params}`
- Phản hồi (Response): `{jsonrpc, id, result}` hoặc `{jsonrpc, id, error}`
- Thông báo (Notification): `{jsonrpc, method, params}` không có `id`

Định danh `id` của yêu cầu dùng để tương quan một phản hồi tương ứng. Nó không tạo ra một phiên làm việc của giao thức.

### Siêu dữ liệu yêu cầu bắt buộc (Required request metadata)

Mỗi yêu cầu hiện đại đều mang theo một đối tượng `_meta` bên trong `params`:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

Phiên bản giao thức và các năng lực của client là bắt buộc. Danh tính của client là thông tin được khuyến nghị. Đó là dữ liệu tự báo cáo dùng để hiển thị và gỡ lỗi, không phải là một chứng chỉ bảo mật.

Server tuyệt đối không được suy diễn bất kỳ giá trị nào trong số này từ một yêu cầu trước đó, từ tiến trình stdio, kết nối HTTP, hay chỉ từ riêng tiêu đề của tầng truyền tải (transport header).

### Kết quả hoàn chỉnh và Danh tính server (Complete results and server identity)

Mọi kết quả thành công hiện đại đều bao gồm trường `resultType`. Một kết quả cuối cùng thông thường sử dụng giá trị `"complete"`. Các server cũng nên tự định danh chính mình trong siêu dữ liệu của kết quả:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`, `resources/list`, `prompts/list`, `resources/templates/list`, `resources/read`, và `server/discover` là các kết quả có thể lưu bộ nhớ đệm (cacheable results). Chúng bao gồm `ttlMs` và `cacheScope`. Giá trị mặc định an toàn là `ttlMs: 0` và `cacheScope: "private"`. Các mục trong danh sách nên có thứ tự tất định (deterministic ordering) để các phản hồi tương đương tạo ra các khóa đệm ổn định và ngữ cảnh mô hình ổn định.

### Khám phá mà không cần bắt tay (Discovery without a handshake)

Mọi server hiện đại bắt buộc phải hiện thực hóa phương thức `server/discover`. Client có thể gọi phương thức này trước một phương thức khác để truy xuất:

- Các phiên bản được hỗ trợ (`supportedVersions`)
- Năng lực của server (`capabilities`)
- Hướng dẫn sử dụng tùy chọn (`instructions`)
- Danh tính server trong `_meta` của kết quả
- Các gợi ý bộ nhớ đệm (cache hints)

Việc khám phá là hữu ích, nhưng nó không phải là một cổng bắt buộc. Client hoàn toàn có thể gửi `tools/list` ngay từ đầu bởi yêu cầu đó đã mang sẵn phiên bản giao thức và các năng lực của nó.

Nếu phiên bản được yêu cầu không được hỗ trợ, server sẽ trả về mã lỗi JSON-RPC `-32022` kèm theo:

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

Client chọn một phiên bản hiện đại được cả hai bên hỗ trợ và thử lại với một id yêu cầu JSON-RPC mới.

### Vòng đời của một yêu cầu (One request lifecycle)

Truy vết một yêu cầu hiện đại theo trình tự sau:

1. Phân tích cú pháp một lớp vỏ bao bọc JSON-RPC.
2. Xác nhận `jsonrpc` là `"2.0"`, có tồn tại `id`, `method` là một chuỗi, và `params` là một đối tượng object.
3. Bắt buộc phải có chuỗi phiên bản và đối tượng năng lực trong `params._meta`; siêu dữ liệu bị sai định dạng hoặc thiếu sẽ trả về lỗi `-32602`.
4. Tại ranh giới HTTP, so sánh tiêu đề phiên bản, phương thức và tên áp dụng với phần thân (body). Sự không khớp sẽ trả về `-32020` ngay cả khi một trong hai giá trị phiên bản không được hỗ trợ.
5. Sau khi tính tương đương được thiết lập, từ chối một phiên bản khớp nhưng không được hỗ trợ bằng mã lỗi `-32022`.
6. Kiểm tra các năng lực bắt buộc, sau đó định tuyến theo `method` và xác thực các đối số đặc thù của phương thức.
7. Xác thực danh tính (authenticate) và ủy quyền (authorize) cho thao tác cụ thể trước khi hàm xử lý của nó chạy.
8. Trả về một kết quả hoàn chỉnh kèm danh tính server.
9. Xóa bỏ hoàn toàn siêu dữ liệu giao thức trong phạm vi của yêu cầu đó (forget request-scoped protocol metadata).

Trình tự đó ngăn chặn việc hai thành phần hiểu sai về các lượt gọi khác nhau. Một gateway tuyệt đối không được phép ủy quyền cho `Mcp-Name: notes.read` trong khi máy chủ nguồn (origin) lại thực thi `params.name: notes.delete`. Trình tự này cũng giữ cho dữ liệu đầu vào sai định dạng, sự nhầm lẫn tiêu đề, đàm phán phiên bản, lỗi năng lực, lỗi ủy quyền và lỗi hàm xử lý luôn là các bằng chứng tách biệt rõ ràng.

Việc đóng stdin hoặc đóng một phản hồi HTTP sẽ kết thúc hoạt động của tầng truyền tải (transport). Nó không chấm dứt một phiên giao thức, bởi vì MCP hiện đại hoàn toàn không có phiên giao thức nào.

### Tương thích tường minh với các phiên bản cũ (Explicit legacy compatibility)

Các phiên bản cho đến `2025-11-25` sử dụng `initialize`, `notifications/initialized`, các năng lực trong phạm vi kết nối, và trên giao thức Streamable HTTP trước đây là các phiên giao thức tùy chọn. Hành vi đó vẫn còn phù hợp khi một client hỗ trợ cả hai thời kỳ (dual-era client) giao tiếp với một server đời cũ.

Hãy giữ cho hai thời kỳ tách biệt hoàn toàn. Một yêu cầu hiện đại được nhận biết nhờ siêu dữ liệu bắt buộc trên từng yêu cầu. Một kết nối phiên bản cũ chỉ được chọn thông qua đường dẫn dự phòng (fallback path) đã được ghi trong tài liệu. Tuyệt đối không gửi `initialize` làm mặc định cho một server chuẩn `2026-07-28`.

Do đó, khái niệm "phi trạng thái" (stateless) mang ý nghĩa đặc thù theo từng thời kỳ. Trong `2026-07-28`, nó là một bất biến của giao thức: mọi yêu cầu thông thường đều có thể diễn giải độc lập và không tồn tại phiên MCP nào. Trong các phiên bản cho đến `2025-11-25`, việc khởi tạo và các năng lực được đàm phán thuộc về một kết nối, do đó một bộ điều hợp tương thích (compatibility adapter) có thể giữ lại trạng thái kết nối cũ đó. Một bản hiện thực hóa hỗ trợ cả hai thời kỳ không phải là một máy trạng thái đơn lẻ dễ dãi. Nó là một lõi hiện đại phi trạng thái đặt cạnh một bộ điều hợp cũ độc lập, với một quyết định lựa chọn tường minh trước khi bất kỳ bộ phân tích cú pháp nào chạy.

Cả hai cách hiểu đều không cấm trạng thái ứng dụng bền vững (durable application state). Một quy trình công việc, nhiệm vụ hoặc bản nháp có thể tồn tại đằng sau một định danh mờ (opaque handle) trong một kho lưu trữ dùng chung. Client gửi định danh đó như một đầu vào thông thường, và mỗi bản sao (replica) đều xác thực và ủy quyền cho việc sử dụng nó. Ngữ cảnh giao thức không được phép rò rỉ vào kho lưu trữ đó như một sự thay thế cho phiên làm việc đã bị loại bỏ.

```figure
mcp-tool-call
```

## Thực hành (Use It)

Tệp `code/main.py` xây dựng, xác thực, truy vết và điều phối các thông điệp MCP hiện đại mà không cần dùng đến framework. Hãy chạy:

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Hãy quan sát ba bất biến trong đầu ra:

- Mỗi yêu cầu đều lặp lại các trường `_meta` của nó.
- Mỗi kết quả thành công đều có `resultType: "complete"` và bao gồm danh tính của server.
- Kết quả danh sách có thứ tự tất định và có các gợi ý bộ nhớ đệm tường minh.

## Đóng gói sản phẩm (Ship It)

Bài học này phát hành artifact `outputs/skill-mcp-handshake-tracer.md`. Tên tệp mang tính lịch sử vẫn được giữ ổn định, nhưng artifact này hiện là một bộ truy vết yêu cầu phi trạng thái. Nó kiểm toán từng thông điệp một cách độc lập và chỉ gắn nhãn lưu lượng bắt tay phiên bản cũ khi lưu lượng đó thực sự xuất hiện.

## Bài tập thực hành (Exercises)

1. Thay đổi phiên bản giao thức của một yêu cầu thành `2027-01-01`. Xác nhận rằng mã lỗi trả về là `-32022` và phần dữ liệu quảng bá phiên bản được hỗ trợ.
2. Xóa bỏ `io.modelcontextprotocol/clientCapabilities` khỏi yêu cầu thứ hai. Xác nhận rằng server không sử dụng lại các năng lực từ yêu cầu thứ nhất.
3. Đảo ngược thứ tự trong sổ đăng ký công cụ trong bộ nhớ. Xác nhận rằng `tools/list` vẫn trả về cùng một thứ tự tất định ban đầu.
4. Thay đổi `cacheScope` từ `public` sang `private`. Giải thích những ngữ cảnh ủy quyền nào có thể tái sử dụng phản hồi trong từng trường hợp.
5. Thêm một bài kiểm tra bỏ qua trường `clientInfo` tùy chọn. Yêu cầu vẫn phải hợp lệ vì danh tính client là thông tin được khuyến nghị, không bắt buộc.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Ý nghĩa |
|-----------|---------|
| Giao thức phi trạng thái (Stateless protocol) | Mọi yêu cầu đều tự cung cấp siêu dữ liệu cần thiết để diễn giải chính nó |
| Siêu dữ liệu yêu cầu (Request metadata) | Phiên bản, năng lực client và danh tính client được khuyến nghị trong `params._meta` |
| `server/discover` | Phương thức server bắt buộc để lấy phiên bản, năng lực, hướng dẫn và danh tính |
| `resultType` | Trường phân biệt (discriminator) trên mọi kết quả thành công hiện đại |
| Kết quả có thể lưu đệm (Cacheable result) | Kết quả bao gồm các gợi ý bắt buộc `ttlMs` và `cacheScope` |
| Thời kỳ giao thức (Protocol era) | Siêu dữ liệu hiện đại trên từng yêu cầu hoặc khởi tạo theo phạm vi kết nối kiểu cũ |
| Vòng đời tầng truyền tải (Transport lifetime) | Vòng đời của tiến trình, kết nối hoặc luồng phản hồi, không phải trạng thái phiên giao thức |
| `-32022` | Lỗi phiên bản giao thức không được hỗ trợ kèm phiên bản yêu cầu và phiên bản hỗ trợ |

## Tài liệu đọc thêm (Further Reading)

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture) — Kiến trúc tổng thể của MCP
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic) — Giao thức cơ sở của MCP
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover) — Khám phá server trong MCP
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) — Nhật ký thay đổi phiên bản MCP 2026-07-28
