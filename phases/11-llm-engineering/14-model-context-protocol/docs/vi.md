# Giao thức Bối cảnh Mô hình (Model Context Protocol - MCP)

> MCP cung cấp cho ứng dụng chủ AI (AI host) một giao thức duy nhất để khám phá và kích hoạt các công cụ (tools), tài nguyên (resources) và lời nhắc (prompts). Bản sửa đổi ngày 2026-07-28 biến giao thức này thành phi trạng thái (stateless): ngữ cảnh về năng lực (capability) và phiên bản đồng hành cùng mọi yêu cầu, thay vì nằm trong một quy trình bắt tay ràng buộc với kết nối (connection-bound handshake).

**Type:** Xây dựng  
**Languages:** Python  
**Prerequisites:** Phase 11 · 09 (Gọi hàm - Function Calling), Phase 11 · 03 (Đầu ra có cấu trúc - Structured Outputs)  
**Time:** ~75 phút  

## Mục tiêu Học tập

- Phân biệt giữa host, client, server, transport (phương thức vận chuyển) và nguyên mẫu máy chủ (server primitive) trong kiến trúc MCP.
- Xây dựng yêu cầu JSON-RPC kèm đầy đủ siêu dữ liệu (metadata) theo tiêu chuẩn MCP 2026-07-28.
- Sử dụng phương thức `server/discover` để kiểm tra phiên bản hỗ trợ, định danh máy chủ và các năng lực khả dụng.
- Trả về kết quả có định kiểu (typed) và hỗ trợ lưu đệm (cache-aware) từ công cụ, tài nguyên và lời nhắc.
- Giải thích cách thức MCP phi trạng thái hiện đại tương tác tương thích ngược với các máy chủ thuộc kỷ nguyên bắt tay (handshake-era).
- Xác định ranh giới an toàn cho trạng thái, phương thức vận chuyển và quyền phê duyệt đối với một máy chủ MCP.

## Vấn đề Thực tế

Ứng dụng của bạn cần thực thi một truy vấn cơ sở dữ liệu, một thao tác lịch biểu và một tác vụ đọc tệp tin. Nếu không có một giao thức dùng chung, mỗi ứng dụng chủ AI đều phải tự phát triển mã nối (glue code) riêng cho từng tác vụ: từ cơ chế khám phá (discovery), kích hoạt (invocation), xử lý lỗi, phương thức vận chuyển cho đến phân quyền ủy quyền.

MCP giúp thu gọn ma trận tích hợp phức tạp đó. Một máy chủ MCP công bố một bề mặt giao tiếp JSON-RPC chuẩn hóa. Một client tuân thủ chuẩn có thể khám phá bề mặt này, hiển thị cho mô hình hoặc người dùng, gọi thực thi và biên dịch kết quả mà không cần viết bộ điều hợp (adapter) riêng biệt cho từng máy chủ.

Tuy nhiên, có một ranh giới quan trọng rất dễ bị hiểu nhầm: MCP chỉ chuẩn hóa giao thức truyền thông. Nó không quyết định mô hình nên gọi công cụ nào, không tự động biến nội dung không đáng tin cậy thành an toàn, và không tự biến một yêu cầu phi trạng thái thành trạng thái ứng dụng bền vững. Host và server của bạn vẫn phải chịu trách nhiệm toàn diện cho những quyết định đó.

## Khái niệm Cốt lõi

![Host MCP, yêu cầu phi trạng thái và các nguyên mẫu máy chủ](../assets/mcp-architecture.svg)

### Ba Nguyên mẫu Máy chủ (Server Primitives)

1. **Công cụ (Tools):** Các hành động có thể gọi thực thi. Mỗi công cụ có một tên gọi, phần mô tả chức năng, định nghĩa đầu vào theo chuẩn JSON Schema và hàm xử lý (handler).
2. **Tài nguyên (Resources):** Nội dung có tên, được định vị bằng URI mà client có thể đọc.
3. **Lời nhắc (Prompts):** Các mẫu lời nhắc có thể tái sử dụng mà host có thể hiển thị cho người dùng lựa chọn.

Host là chính ứng dụng AI của bạn. Một MCP client nằm bên trong host đó sẽ kết nối và giao tiếp với một máy chủ MCP. Tầng vận chuyển (transport) đảm nhiệm việc truyền tải các thông điệp JSON-RPC giữa hai bên.

### Yêu cầu Phi trạng thái Thay thế Quy trình Bắt tay

Phiên bản MCP 2026-07-28 đã loại bỏ hoàn toàn các phương thức `initialize` và `notifications/initialized`, đồng thời xóa bỏ các phiên làm việc (sessions) ở cấp độ giao thức. Mọi yêu cầu giờ đây đều tự mang theo ngữ cảnh cần thiết để máy chủ giải nghĩa thông qua trường `params._meta`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Phiên bản giao thức và năng lực của client (client capabilities) là các trường bắt buộc. Thông tin định danh client (clientInfo) là trường được khuyến nghị. Một yêu cầu thiếu `_meta`, thiếu trường bắt buộc hoặc trường bắt buộc sai kiểu dữ liệu sẽ bị coi là sai định dạng và trả về lỗi Invalid Params (`-32602`). Nếu chuỗi phiên bản đúng định dạng nhưng máy chủ không hỗ trợ, lỗi `UnsupportedProtocolVersionError` (`-32022`) sẽ được trả về. Nhờ vậy, máy chủ có thể xử lý một yêu cầu hợp lệ ngay lập tức mà không cần khôi phục lại bất kỳ bản ghi thương lượng kết nối nào trước đó.

Phi trạng thái không có nghĩa là ứng dụng không bao giờ được lưu trạng thái. Nó có nghĩa là trạng thái không bị ẩn giấu phía sau một kết nối MCP vật lý hay thông qua tiêu đề `Mcp-Session-Id`. Nếu một quy trình nghiệp vụ cần tính liên tục, máy chủ sẽ phát hành một thẻ xử lý mờ (opaque handle) và client sẽ truyền lại thẻ này như một đối số công cụ thông thường trong các lệnh gọi tiếp theo. Việc kiểm tra ủy quyền (authorization) phải được thực hiện trên từng yêu cầu riêng lẻ.

### Khám phá và Lựa chọn Phiên bản (Discovery)

Mọi máy chủ hiện đại đều triển khai phương thức `server/discover`. Kết quả trả về công bố các phiên bản được hỗ trợ, các năng lực khả dụng và danh tính máy chủ:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

Client có thể gọi trực tiếp một phương thức khác và xử lý lỗi phiên bản nếu có, nhưng cơ chế khám phá (discovery) giúp việc hiển thị năng lực và lựa chọn phiên bản trở nên rõ ràng và chủ động. Một phiên bản không được hỗ trợ sẽ trả về lỗi `UnsupportedProtocolVersionError` với mã `-32022`. Dữ liệu lỗi chứa trường `supported` (mảng các bản sửa đổi máy chủ hỗ trợ) và `requested` (phiên bản bị từ chối).

Trên kênh stdio, một client hỗ trợ cả hai kỷ nguyên (dual-era client) sẽ thăm dò trước bằng `server/discover`. Kết quả khám phá thành công hoặc một lỗi hiện đại được nhận diện như `UnsupportedProtocolVersionError` sẽ xác định đây là máy chủ hiện đại. Bất kỳ lỗi nào khác hoặc thời gian chờ không thuộc chuẩn hiện đại sẽ cho phép hệ thống dự phòng quay về luồng `initialize` kiểu cũ của phiên bản 2025-11-25. Hành vi cũ chỉ đóng vai trò mã tương thích ngược, không phải là chuẩn mặc định hiện đại.

### Kết quả Tường minh (Explicit Results)

Mọi kết quả trong lõi chuẩn 2026-07-28 đều có trường `resultType`:

- `complete`: Cho biết thao tác đã kết thúc thành công.
- `input_required`: Cho biết máy chủ cần thêm một vòng trao đổi dữ liệu thông qua mẫu Yêu cầu Đa Vòng (Multi Round-Trip Requests - MRTR). Các máy chủ tiêu chuẩn chỉ được phép trả về trạng thái này từ `tools/call`, `resources/read`, hoặc `prompts/get`.

Client phải xem các kết quả cũ (thiếu trường `resultType`) như là kết quả đã hoàn tất (`complete`).

Máy chủ nên đính kèm trường `io.modelcontextprotocol/serverInfo` trong `_meta` của mọi kết quả. Danh tính này do máy chủ tự khai báo, phục vụ mục đích hiển thị, ghi log và gỡ lỗi, không dùng để đưa ra quyết định an ninh bảo mật.

Kết quả danh sách (list) và đọc (read) cũng mang theo trường `ttlMs` (thời gian sống tính theo ms) và `cacheScope`. Thứ tự trả về tất định của `tools/list` kết hợp với gợi ý về độ mới của dữ liệu cho phép client lưu bộ nhớ đệm an toàn, đồng thời tăng cường độ ổn định cho tính năng lưu đệm lời nhắc (prompt caching). Cấu hình `cacheScope: public` cho phép chia sẻ bộ nhớ đệm dùng chung; trong khi `private` giới hạn phạm vi tái sử dụng trong ngữ cảnh người gọi hiện tại.

### Định dạng Dây cáp và Phương thức Vận chuyển (Wire Format and Transport)

MCP sử dụng JSON-RPC 2.0 trên kênh stdio hoặc Streamable HTTP.

- Một yêu cầu (request) gồm các trường: `jsonrpc`, `id`, `method`, và `params`.
- Một phản hồi (response) có `id` tương ứng và chứa `result` hoặc `error`.
- Một thông báo (notification) không có `id` và không mong đợi phản hồi.

Streamable HTTP hiện đại chỉ mở duy nhất một điểm cuối tiếp nhận phương thức POST. Mỗi thông điệp JSON-RPC là một yêu cầu POST riêng biệt. Yêu cầu POST sẽ nhận về một đối tượng JSON đơn lẻ hoặc một luồng Server-Sent Events có vòng đời theo yêu cầu (request-scoped SSE stream) kết thúc bằng phản hồi cuối cùng. Một thông báo POST hợp lệ sẽ nhận về mã HTTP 202 không kèm thân phản hồi; bản sửa đổi lõi này không định nghĩa thông báo từ client gửi lên server qua Streamable HTTP.

Không còn tồn tại luồng GET độc lập, điểm cuối DELETE phiên làm việc, tiêu đề `Mcp-Session-Id`, hay cơ chế phát lại `Last-Event-ID` trong chuẩn 2026-07-28. Các thông báo thay đổi dài hạn sử dụng một yêu cầu POST `subscriptions/listen` với phản hồi được duy trì mở dưới dạng luồng SSE.

### Thu thập Đầu vào Client mà Không cần Yêu cầu từ Máy chủ

Các phiên bản cũ cho phép máy chủ chủ động gửi các yêu cầu như `sampling/createMessage`, `roots/list`, hoặc `elicitation/create` qua luồng dữ liệu. Giao thức hiện tại thay thế hoàn toàn cơ chế này bằng mô hình Yêu cầu Đa Vòng (Multi Round-Trip Requests - MRTR). Khi cần thêm dữ liệu, lệnh gọi công cụ, đọc tài nguyên hoặc lấy lời nhắc sẽ trả về `resultType: input_required` kèm ít nhất một trong hai trường `inputRequests` hoặc `requestState`. Client thu thập dữ liệu đầu vào được yêu cầu, gửi lại lệnh gọi ban đầu với một ID JSON-RPC mới kèm `inputResponses` tương ứng, và gửi lặp lại chính xác `requestState` nếu máy chủ đã cung cấp trước đó. Nếu không có `inputRequests`, lệnh gửi lại sẽ bỏ qua `inputResponses`.

Các cơ chế Roots, Sampling và Logging vẫn hoạt động nhưng đã bị đánh dấu lỗi thời (deprecated), do đó các hệ thống mới không nên sử dụng. Các yêu cầu Roots hoặc Sampling hiện có sẽ được đóng gói bên trong `inputRequests` của MRTR, tuyệt đối không gửi dưới dạng các yêu cầu JSON-RPC độc lập từ máy chủ đến client. Hãy ưu tiên sử dụng các tham số tệp tin hoặc thư mục tường minh, URI tài nguyên, cấu hình máy chủ và tích hợp trực tiếp với nhà cung cấp mô hình. Sử dụng kênh stderr cho chẩn đoán stdio và OpenTelemetry cho dữ liệu quan sát trong môi trường sản xuất.

````figure mcp-nxm-collapse
````

## Phần Thực hành (Build It)

### Bước 1: Đăng ký Bề mặt Máy chủ

Việc đăng ký vẫn duy trì tính đơn giản ngay cả khi hợp đồng yêu cầu đã chuyển sang phi trạng thái:

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

Bản cài đặt hoàn chỉnh trong `code/main.py` cũng đăng ký thêm một tài nguyên và một lời nhắc mẫu. Mã nguồn được viết hoàn toàn bằng thư viện chuẩn (stdlib) của Python để bạn có thể quan sát rõ từng cấu trúc bao đóng thay vì phụ thuộc vào SDK trừu tượng.

### Bước 2: Đính kèm Siêu dữ liệu (Metadata) vào Mọi Yêu cầu

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

Không chỉ lưu đệm siêu dữ liệu này trong một đối tượng kết nối. Máy chủ sẽ kiểm tra và xác thực nó trên từng yêu cầu gửi đến.

### Bước 3: Khám phá Năng lực Trước khi Liệt kê (Tùy chọn)

Gọi phương thức `server/discover`, chọn phiên bản phù hợp được hỗ trợ, sau đó mới gọi `tools/list`. Gọi trực tiếp `tools/list` cũng hoàn toàn hợp lệ nếu bạn đã biết rõ phiên bản máy chủ và có cơ chế bắt mã lỗi `-32022`.

Mã demo trả về danh sách công cụ được sắp xếp theo thứ tự bảng chữ cái của tên và đính kèm `ttlMs`, `cacheScope`, `resultType` và định danh máy chủ. Một lệnh gọi công cụ sẽ trả về kết quả hoàn tất (`complete`) không thể lưu đệm vì đầu ra phụ thuộc vào trạng thái hiện thời của hệ thống.

### Bước 4: Ánh xạ Yêu cầu Tương tự sang Giao thức HTTP

Một yêu cầu POST gọi `tools/call` từ xa sẽ chứa các tiêu đề HTTP phản chiếu chính xác phần thân JSON-RPC:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

Tiêu đề `MCP-Protocol-Version` bắt buộc phải khớp với phiên bản trong `_meta`. Tiêu đề `Mcp-Method` là bắt buộc trên mọi yêu cầu JSON-RPC và phải khớp với `method`. Tiêu đề `Mcp-Name` chỉ bắt buộc đối với `tools/call`, `resources/read`, và `prompts/get`, trong đó giá trị phải khớp với tên công cụ, URI tài nguyên hoặc tên lời nhắc. Nếu thiếu tiêu đề bắt buộc hoặc có sự sai lệch, máy chủ sẽ trả về lỗi HTTP 400 kèm mã `HeaderMismatch` (`-32020`).

### Bước 5: Thực thi An toàn Ngoài Trạng thái Giao thức

- Xác thực ủy quyền (authorization) và đối tượng thụ hưởng (audience) trên từng yêu cầu HTTP.
- Ràng buộc các máy chủ cục bộ với localhost và xác thực tiêu đề `Origin` trên Streamable HTTP.
- Đánh dấu các công cụ có tác dụng phụ làm thay đổi dữ liệu với cờ `destructiveHint: true` và yêu cầu xác nhận phê duyệt từ host.
- Truyền phạm vi thư mục và tệp tin một cách tường minh thay vì phụ thuộc vào cơ chế Roots cũ đã bị loại bỏ.
- Coi nội dung tài nguyên và đầu ra của công cụ là dữ liệu không đáng tin cậy (untrusted data).
- Giữ kênh stdout dành riêng cho thông điệp JSON-RPC khi chạy qua stdio; ghi toàn bộ thông tin chẩn đoán ra stderr.

## Ứng dụng Thực tế (Use It)

Chạy bài học trực tiếp từ thư mục mã nguồn:

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

Dòng đầu tiên xuất ra màn hình sẽ thông báo kết quả khám phá `demo-server` tại giao thức `2026-07-28`. Sau đó hãy kiểm tra phương thức `MCPClient.request`: nó tái tạo cấu trúc `_meta` cho từng lệnh gọi. Thử xóa siêu dữ liệu khỏi một yêu cầu và quan sát cách máy chủ từ chối tiếp nhận.

## Thành phẩm Đóng gói (Ship It)

Tệp `outputs/skill-mcp-server-designer.md` chuyển hóa yêu cầu nghiệp vụ của một miền cụ thể thành bản thiết kế MCP phi trạng thái chuẩn mực. Tiêu chuẩn nghiệm thu của nó đòi hỏi kết quả khám phá năng lực, chính sách siêu dữ liệu trên từng yêu cầu, danh sách có tính tất định hỗ trợ lưu đệm, thẻ trạng thái rõ ràng, tiêu đề vận chuyển, kiểm soát ủy quyền và quy tắc phê duyệt hành động nguy hiểm.

## Đào sâu Hơn về MCP

Bài học này cung cấp mô hình giao thức nền tảng. Phase 13 sẽ nâng cấp bốn ranh giới môi trường sản xuất thành các bài học xây dựng và xác thực chuyên sâu riêng biệt:

1. [Hợp đồng Công cụ và Nội dung MCP](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md): Phân tích lược đồ đầu vào đóng, nội dung có cấu trúc, siêu dữ liệu định tuyến, phân trang mờ, phân quyền hoàn tất, và sự khác biệt giữa lỗi giao thức và lỗi thuộc miền công cụ.
2. [Độ tin cậy, Hủy bỏ và Kiểm soát Luồng MCP](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md): Phân tích cơ chế hủy yêu cầu, hủy tác vụ dài hạn, thời hạn chót (deadlines), tính lũy đẳng (idempotency), kiểm soát áp lực ngược (backpressure), lưu đệm qua proxy và hành vi kết nối lại.
3. [Chuỗi Cung ứng MCP Registry, Tiếp nhận, Sai lệch và Hoàn nguyên](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md): Phân tích chứng minh không gian tên (namespace proof), nguồn gốc xuất xứ cấu phần, ghim phiên bản bất biến, phát hiện sai lệch trực tiếp, trạng thái Registry, bằng chứng tiếp nhận và hoàn nguyên.
4. [Kỹ thuật Kiểm định Tuân thủ MCP](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md): Phân tích bản ghi kiểm thử dây cáp mẫu (golden) và tiêu cực, các kỷ nguyên phiên bản nghiêm ngặt, sai biệt giữa các SDK, bằng chứng proxy, che giấu dữ liệu nhạy cảm, cổng kiểm tra sức khỏe và hoàn nguyên bản phát hành.

Hãy học tuần tự các bài này khi máy chủ của bạn cần vượt qua ranh giới giữa các đội ngũ kỹ thuật hoặc các ranh giới tin cậy khác nhau. Chúng sẽ giúp bạn nâng tầm từ "phương thức chạy được" sang "hợp đồng giao tiếp an toàn, minh bạch và có thể chẩn đoán lỗi trong môi trường triển khai thực tế."

## Bài tập Thực hành

1. Thêm một công cụ `subtract` và xác nhận rằng danh sách trả về từ `tools/list` vẫn giữ nguyên thứ tự sắp xếp theo bảng chữ cái.
2. Xóa khóa phiên bản giao thức và xác minh máy chủ trả về mã Invalid Params (`-32602`). Sau đó gửi một chuỗi phiên bản đúng định dạng nhưng không được hỗ trợ như `2025-11-25`, kiểm tra mã lỗi `-32022`, xác nhận trường `requested` phản ánh đúng bản sửa đổi đó và chọn phiên bản thay thế từ danh sách `supported`.
3. Thêm một mã `draftId` do máy chủ sinh ra vào thao tác tạo mới (create), sau đó yêu cầu truyền mã này như một đối số trong thao tác cập nhật (update). Giải thích tại sao đây là trạng thái ứng dụng nghiệp vụ chứ không phải phiên làm việc của giao thức.
4. Trả về `input_required` từ một công cụ cần xác nhận của người dùng. Thử lại lệnh gọi gốc với một ID mới, mục `inputResponses` chứa dữ liệu người dùng nhập, và giữ nguyên giá trị `requestState` thay vì tự tạo một yêu cầu JSON-RPC độc lập từ server gửi sang client.
5. Phác thảo một client stdio hỗ trợ cả hai kỷ nguyên. Coi kết quả hoặc mã lỗi hiện đại đã biết là dấu hiệu của máy chủ hiện đại, và chỉ cho phép chuyển đổi dự phòng sang quy trình `initialize` khi gặp lỗi không xác định hoặc hết thời gian chờ.

## Thuật ngữ Then chốt

| Thuật ngữ | Cách mọi người thường nói | Bản chất kỹ thuật thực tế |
|------|-----------------|------------------------|
| MCP | "Giao thức công cụ cho LLM" | Giao thức JSON-RPC chuẩn hóa dùng để khám phá máy chủ, công cụ, tài nguyên, lời nhắc và tiện ích mở rộng |
| Host | "Ứng dụng AI" | Thành phần sở hữu mô hình và giao diện người dùng, đồng thời tích hợp một hoặc nhiều MCP client |
| Client | "Đầu kết nối" | Thành phần đại diện cho host để giao tiếp theo chuẩn MCP với một máy chủ |
| Stateless MCP | "Không có phiên làm việc" | Mọi yêu cầu đều tự mang phiên bản và năng lực; không lưu trạng thái giao thức gắn với kết nối mạng |
| `server/discover` | "Thăm dò năng lực" | Phương thức bắt buộc trên máy chủ để công bố phiên bản hỗ trợ, năng lực khả dụng và thông tin định danh |
| `resultType` | "Trạng thái kết quả" | Đánh dấu kết quả ở trạng thái `complete` (đã xong) hoặc `input_required` (cần thêm đầu vào) |
| State handle | "Mã quy trình (Workflow ID)" | Mã định danh do máy chủ cấp phát ở cấp ứng dụng, được truyền như một đối số thông thường |
| Streamable HTTP | "Vận chuyển từ xa" | Một điểm cuối HTTP POST duy nhất trả về đối tượng JSON hoặc luồng SSE theo vòng đời yêu cầu |
| MRTR | "Hỏi và thử lại" | Mô hình Yêu cầu Đa Vòng: yêu cầu đầu vào được nhúng trong kết quả, tiếp nối bằng việc client thử lại thao tác ban đầu |

## Đọc thêm

- [Các thay đổi chính của MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [Khám phá máy chủ MCP](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Yêu cầu Đa Vòng (Multi Round-Trip Requests) trong MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Các tính năng MCP đã lỗi thời (deprecated)](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
