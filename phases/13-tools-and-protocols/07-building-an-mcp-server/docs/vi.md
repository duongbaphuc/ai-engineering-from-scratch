# Xây dựng Máy chủ MCP: Python và TypeScript Không lưu trạng thái

> Một máy chủ MCP hiện đại không ghi nhớ quá trình bắt tay. Nó xác thực siêu dữ liệu trên từng yêu cầu, chạy một trình xử lý duy nhất và trả về một kết quả có định kiểu.

**Type:** Xây dựng (Build)
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Bài học 06
**Time:** ~85 phút

## Mục tiêu học tập

- Triển khai phương thức bắt buộc `server/discover` cho đặc tả MCP `2026-07-28`.
- Xác thực phiên bản giao thức và khả năng của máy khách (client capabilities) trên từng yêu cầu.
- Công bố công cụ (tools), tài nguyên (resources), và lời nhắc (prompts) với thứ tự danh sách có tính tất định (deterministic list ordering).
- Trả về `resultType`, danh tính máy chủ (server identity), và gợi ý bộ nhớ đệm (cache hints) trên các kết quả phù hợp.
- Phục vụ cùng một hợp đồng không lưu trạng thái (stateless contract) qua luồng stdio phân tách bằng dòng mới (newline-delimited stdio) bằng cả Python và TypeScript.

## Vấn đề

Một máy chủ lưu trữ các khả năng của máy khách sau thông điệp đầu tiên thì rất dễ xây dựng nhưng lại cực kỳ khó vận hành. Cùng một tiến trình có thể phục vụ nhiều máy khách tuần tự. Một yêu cầu từ xa có thể được điều hướng tới một tiến trình worker khác. Một khai báo khả năng đã lỗi thời có thể làm rò rỉ hành vi vượt qua các ranh giới ủy quyền (authorization boundaries).

Đặc tả MCP `2026-07-28` giải quyết phần giao thức của vấn đề đó bằng cách làm cho mọi yêu cầu đều tự mô tả (self-describing). Ứng dụng của bạn vẫn có thể duy trì các ghi chú bền vững, tác vụ nền, hoặc các thẻ định danh trạng thái rõ ràng (explicit state handles). Điều nó tuyệt đối không được giữ là trạng thái giao thức ẩn giấu làm thay đổi cách giải mã một yêu cầu đến sau.

Bài học này xây dựng một máy chủ ghi chú (notes server) hai lần. Các phiên bản Python và TypeScript chỉ sử dụng thư viện chuẩn của ngôn ngữ cho phần lõi giao thức. Cả hai đều hiển thị các phương thức giống nhau và thực thi cùng một hợp đồng truyền tải trên đường truyền (wire contract).

## Khái niệm

### Vòng lặp điều phối hiện đại

```text
đọc một dòng JSON-RPC
phân tích phong bì (envelope)
nếu là thông báo (notification), không phản hồi
xác thực params._meta cho yêu cầu này
điều hướng theo phương thức (method)
bọc kết quả thành công với resultType và serverInfo
ghi một dòng phản hồi JSON-RPC
xóa bỏ siêu dữ liệu trong phạm vi yêu cầu (request-scoped metadata)
```

Ba quy tắc stdio vẫn giữ vai trò then chốt:

- Chỉ ghi các thông điệp JSON-RPC ra stdout. Gửi các thông tin chẩn đoán ra stderr.
- Phân cách các thông điệp bằng ký tự dòng mới và đẩy bộ đệm (flush) cho từng phản hồi.
- Thoát ngay lập tức khi stdin gặp ký tự kết thúc tệp (EOF).

Vòng đời của tiến trình chính là vòng đời của phương thức truyền tải (transport lifetime). Nó không phải là một phiên làm việc (session) MCP hiện đại.

### Xác thực yêu cầu

Mọi yêu cầu phải chứa:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Hai trường đầu tiên là bắt buộc. Trường `clientInfo` được khuyến nghị cung cấp. Hãy xác thực cấu trúc danh tính nếu nó xuất hiện, nhưng không coi nó là một cơ chế xác thực danh tính bảo mật (authentication).

Nếu phiên bản không được hỗ trợ, trả về mã lỗi `-32022` cùng với hai trường `requested` và `supported`. Siêu dữ liệu yêu cầu bị thiếu được xem là tham số không hợp lệ (invalid params), mã lỗi `-32602`. Tuyệt đối không tự động điền các trường bị thiếu từ một lệnh gọi trước đó.

### Khám phá bắt buộc

Các máy chủ hiện đại bắt buộc phải triển khai `server/discover`. Một kết quả khám phá hoàn chỉnh bao gồm các phiên bản hiện đại được hỗ trợ, danh sách khả năng, hướng dẫn tùy chọn, gợi ý bộ đệm và danh tính máy chủ trong `_meta` của kết quả:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Việc khám phá không mở khóa máy chủ. Máy khách có thể gọi trực tiếp `tools/list` mà không cần gọi khám phá trước, bởi vì `tools/list` vốn dĩ đã mang đầy đủ siêu dữ liệu yêu cầu tương tự.

### Công cụ

`tools/list` trả về một danh sách tất định gồm các bộ mô tả công cụ (tool descriptors). Thứ tự ổn định giúp cải thiện khả năng lưu đệm phản hồi và giữ cho ngữ cảnh mô hình luôn nhất quán. Kết quả cũng yêu cầu các trường `ttlMs` và `cacheScope`.

`tools/call` trả về các khối nội dung (content blocks) và cờ `isError`. Sử dụng lỗi JSON-RPC khi phong bì giao thức hoặc các tham số phương thức không hợp lệ. Sử dụng `isError: true` khi việc gọi công cụ hợp lệ diễn ra nhưng chính công cụ đó thất bại trong quá trình thực thi.

Các chú thích công cụ (tool annotations) chỉ đóng vai trò là gợi ý, không phải là cơ chế bắt buộc cưỡng chế:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Ứng dụng chủ (host) nên sử dụng chúng để xác nhận với người dùng và hiển thị giao diện. Máy chủ vẫn phải tự thực thi việc kiểm tra quyền hạn thực tế.

### Tài nguyên

`resources/list` trả về các bộ mô tả URI ổn định. `resources/read` trả về nội dung có định kiểu. Cả hai đều có thể lưu đệm được trong đặc tả `2026-07-28`, do đó cả hai đều bao gồm `ttlMs` và `cacheScope`.

Sử dụng `cacheScope: "private"` cho dữ liệu ghi chú riêng biệt của từng người dùng. Bộ nhớ đệm dùng chung tuyệt đối không được tái sử dụng một phản hồi riêng tư qua các ngữ cảnh ủy quyền khác nhau.

Cơ chế phân phối thay đổi hiện đại không sử dụng `resources/subscribe`. Máy khách mở luồng `subscriptions/listen` và yêu cầu `resourceSubscriptions` hoặc các danh mục thay đổi danh sách. Bài học 10 sẽ xây dựng luồng này.

### Lời nhắc

`prompts/list` có thể lưu vào bộ nhớ đệm và mang tính tất định. `prompts/get` kết xuất (render) một lời nhắc theo tên cùng với các đối số truyền vào. Kết quả lời nhắc đã kết xuất là hoàn chỉnh, nhưng nó không nằm trong nhóm các kết quả danh sách hoặc đọc có thể lưu đệm yêu cầu gợi ý bộ nhớ đệm.

### Mọi kết quả thành công đều được định kiểu

Các ví dụ sử dụng một hàm bao bọc duy nhất cho mọi kết quả thành công:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Các trình xử lý danh sách, đọc và khám phá sẽ bổ sung thêm `ttlMs` cùng `cacheScope`. Việc tập trung hóa hàm bao bọc này ngăn chặn việc một trình xử lý nào đó vô tình bỏ quên các trường kết quả hiện đại.

### Không có yêu cầu khởi tạo từ máy chủ

Một máy chủ hiện đại có thể gửi các thông báo liên quan đến một yêu cầu của máy khách, hoặc các thông báo trên luồng `subscriptions/listen` do máy khách mở. Nó tuyệt đối không được tự ý gửi yêu cầu JSON-RPC của riêng mình.

Khi một trình xử lý cần lấy mẫu (sampling), gợi mở phản hồi từ người dùng (elicitation), hoặc lấy thông tin thư mục gốc (roots input), nó sẽ trả về kết quả `input_required`. Máy khách sẽ đáp ứng các yêu cầu đầu vào được nhúng này và thử lại phương thức ban đầu với một id yêu cầu mới. Bài học 11 sẽ trình bày chi tiết mẫu Yêu cầu nhiều lượt khứ hồi (Multi Round-Trip Request pattern) này.

### Tính tương thích ngược rõ ràng

Một máy chủ hỗ trợ đa thế hệ (dual-era server) cũng có thể triển khai quy trình bắt tay của đặc tả `2025-11-25` trên một nhánh mã nguồn cũ (legacy branch) hoàn toàn tách biệt. Nó chọn hành vi hiện đại khi các trường `_meta` hiện đại bắt buộc có mặt, và chọn hành vi cũ khi nhận được phương thức `initialize`.

Không đưa một yêu cầu `2026-07-28` đi qua luồng bắt tay cũ. Không gắn các trường `resultType` hiện đại vào kết quả khởi tạo cũ. Mã nguồn trong bài học này được cố tình thiết kế chỉ theo chuẩn hiện đại để các bất biến của nó luôn được tường minh.

```figure
t3-dispatch-loop
```

## Sử dụng

Chạy bản thử nghiệm hữu hạn và các bài kiểm thử của máy chủ Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Chạy bản chuyển mã TypeScript bằng trình chạy TypeScript:

```bash
npx tsx main.ts --demo
```

Bản thử nghiệm gửi yêu cầu `server/discover`, liệt kê từng thành phần nguyên thủy, gọi các công cụ và minh họa lỗi phiên bản không được hỗ trợ. Mọi yêu cầu hiện đại đều lặp lại siêu dữ liệu. Mọi kết quả thành công đều bao gồm danh tính máy chủ.

## Đóng gói / Triển khai

Bài học này cung cấp `outputs/skill-mcp-server-scaffolder.md`. Tệp này tạo ra một khung thiết kế máy chủ hiện đại với hợp đồng khám phá, xác thực trên từng yêu cầu, các danh sách có thể lưu đệm tất định, và một bộ điều hợp cũ (legacy adapter) cô lập tùy chọn.

## Bài tập

1. Loại bỏ các khai báo khả năng khỏi một yêu cầu và chứng minh máy chủ không tái sử dụng khai báo của yêu cầu trước đó.
2. Đảo ngược thứ tự chèn của `TOOLS`, `PROMPTS`, và ghi chú. Xác nhận rằng toàn bộ kết quả danh sách vẫn giữ thứ tự ổn định.
3. Thêm công cụ hủy hoại `notes_delete` và yêu cầu kiểm tra ủy quyền bên trong bộ thực thi. Giữ `destructiveHint` chỉ như một gợi ý trải nghiệm người dùng (UX hint).
4. Thêm phương thức `resources/templates/list` với `ttlMs`, `cacheScope` và thứ tự tất định.
5. Xây dựng một bộ điều hợp kế thừa riêng biệt cho đặc tả `2025-11-25`. Thêm các bài kiểm thử chứng minh rằng một yêu cầu hiện đại không bao giờ đi vào nhánh mã này.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| Máy chủ không lưu trạng thái (Stateless server) | Xử lý từng yêu cầu từ chính siêu dữ liệu của nó mà không phụ thuộc vào bộ nhớ phiên giao thức |
| `server/discover` | Phương thức hiện đại bắt buộc dùng để quảng bá phiên bản và các khả năng hỗ trợ |
| Kết quả hoàn chỉnh (Complete result) | Kết quả hiện đại thành công có gắn `resultType: "complete"` |
| Kết quả có thể lưu đệm (Cacheable result) | Kết quả khám phá, danh sách, hoặc đọc tài nguyên có kèm `ttlMs` và `cacheScope` |
| Danh sách tất định (Deterministic list) | Cùng một thanh ghi logic luôn tạo ra cùng một thứ tự mục |
| Danh tính máy chủ (Server identity) | Khóa khuyến nghị `io.modelcontextprotocol/serverInfo` trong `_meta` của kết quả |
| Lỗi công cụ (Tool error) | Lệnh gọi công cụ hợp lệ nhưng trả về nội dung với cờ `isError: true` |
| Lỗi giao thức (Protocol error) | Yêu cầu JSON-RPC hoặc MCP không hợp lệ được trả về qua trường `error` |

## Đọc thêm

- [Đặc tả MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [Khám phá Máy chủ MCP (MCP Server Discovery)](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Công cụ MCP (MCP Tools)](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [Tài nguyên MCP (MCP Resources)](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [Lời nhắc MCP (MCP Prompts)](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [Phương thức truyền tải stdio trong MCP (MCP stdio Transport)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
