# Đầu vào Mô hình trong MCP: Di chuyển Lấy mẫu và MRTR Không lưu trạng thái

> MCP 2026-07-28 chính thức ngừng khuyến nghị (deprecate) tính năng Lấy mẫu (Sampling) cho các thiết kế mới và loại bỏ hoàn toàn kênh yêu cầu từ máy chủ tới máy khách. Nếu một quy trình làm việc hiện có vẫn cần mô hình của máy khách, máy chủ sẽ trả về kết quả `input_required` và máy khách sẽ thử lại yêu cầu ban đầu kèm theo đầu ra của mô hình. Vòng lặp suy luận trở nên tường minh, có giới hạn và không lưu trạng thái ở tầng giao thức.

**Type:** Xây dựng (Build)
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (Máy chủ MCP), Phase 13 · 10 (Tài nguyên và Lời nhắc)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do vì sao Sampling bị ngừng khuyến nghị trong MCP 2026-07-28 và lựa chọn tích hợp mô hình trực tiếp làm mặc định cho các máy chủ mới.
- Triển khai một quy trình làm việc tương thích mang theo `sampling/createMessage` thông qua cơ chế Yêu cầu Nhiều Lượt khứ hồi (Multi Round-Trip Requests - MRTR).
- Đưa bản sửa đổi giao thức và khả năng của máy khách vào đối tượng `_meta` của từng yêu cầu.
- Trả về `resultType: "input_required"` và thử lại phương thức ban đầu với một id JSON-RPC mới tinh.
- Bảo vệ tính toàn vẹn của `requestState` và ràng buộc nó với chủ thể xác thực (principal), phương thức, các đối số và thời gian hết hạn.
- Giới hạn các vòng lặp có mô hình hỗ trợ bằng kiểm tra khả năng, phê duyệt, xác thực phản hồi và hạn mức số lượt.

## Quyết định Trước Giao thức

Một công cụ như `summarize_repo` đòi hỏi hai loại công việc:

1. Công việc tất định: liệt kê các tệp, đọc các tệp được phép, xác thực đường dẫn và tập hợp nội dung.
2. Công việc của mô hình: chọn các tệp mang tính đại diện và tổng hợp bản tóm tắt.

Hiện tại bạn có hai kiến trúc hợp lệ để lựa chọn.

### Máy chủ mới: tích hợp trực tiếp với nhà cung cấp mô hình

Đây là lựa chọn mặc định hiện nay. Máy chủ tự quản lý việc lựa chọn mô hình, thông tin xác thực, ngân sách chi phí, cơ chế thử lại và khả năng quan sát (observability). Nó trả về một kết quả `tools/call` bình thường duy nhất cho máy khách MCP.

Hãy chọn hướng đi này khi máy chủ vốn dĩ đã là một dịch vụ được lưu trữ (hosted service) hoặc khi hành vi mô hình có thể dự đoán được quan trọng hơn việc sử dụng mô hình của ứng dụng chủ.

### Quy trình Sampling hiện có: di chuyển sang MRTR

Tính năng Sampling vẫn tiếp tục tồn tại trong giai đoạn chuyển tiếp trước khi bị loại bỏ hoàn toàn. Một máy chủ nhắm tới phiên bản 2026-07-28 không thể gửi một yêu cầu `sampling/createMessage` trực tiếp ngược lại cho máy khách. Thay vào đó, nó nhúng yêu cầu đó vào một `InputRequiredResult`.

Chỉ chọn con đường tương thích này khi việc sử dụng mô hình và thông tin xác thực của máy khách là một yêu cầu sản phẩm thực sự bắt buộc. Hãy lập kế hoạch loại bỏ cụ thể vì các bản triển khai mới không nên áp dụng tính năng Sampling đã bị ngừng khuyến nghị.

## Hợp đồng Không lưu trạng thái

Giao thức tháng 7 năm 2026 không có quá trình trao đổi `initialize`, không có `notifications/initialized`, và không có `Mcp-Session-Id`. Mọi yêu cầu đều mang thông tin vốn trước đây nằm trong quy trình bắt tay:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Máy chủ xác thực bản sửa đổi trên từng yêu cầu. Phiên bản bị thiếu hoặc không phải chuỗi ký tự được coi là tham số không hợp lệ, lỗi `-32602`. Một chuỗi phiên bản không được hỗ trợ sẽ trả về mã lỗi `-32022` cùng dữ liệu chính xác `{"supported":["2026-07-28"],"requested":"<client version>"}`. Thiếu khả năng Sampling sẽ trả về mã `-32021` với `data.requiredCapabilities` đặt thành `{"sampling":{}}`.

Một phong bì không có trường `id` của JSON-RPC là một thông báo (notification). Bên nhận có thể xử lý nó, nhưng tuyệt đối không phát ra phản hồi thành công cũng như phản hồi lỗi. Bộ điều hợp Streamable HTTP sẽ trả về mã HTTP `202 Accepted` không có phần thân cho một thông báo được chấp nhận.

Máy chủ cũng triển khai `server/discover` với khóa `supportedVersions` chuẩn xác, các khả năng, `ttlMs` và `cacheScope` để máy khách có thể tìm hiểu và lưu đệm hợp đồng máy chủ trước khi gọi công cụ. Vì khám phá quảng bá năng lực `tools`, máy chủ cũng phải triển khai phương thức bắt buộc `tools/list`. Bộ mô tả `summarize_repo` tất định của nó bao gồm một đối tượng `inputSchema` hợp lệ, `resultType: "complete"`, siêu dữ liệu danh tính máy chủ và gợi ý bộ nhớ đệm công khai.

Mọi kết quả hiện đại thành công đều có một trường phân biệt (discriminator):

- `resultType: "complete"` biểu thị thao tác đã hoàn thành.
- `resultType: "input_required"` biểu thị máy khách phải đáp ứng các yêu cầu nhúng và thử lại.
- Các phần mở rộng có thể định nghĩa thêm các kiểu kết quả khác. Phần mở rộng Tasks bổ sung thêm kiểu `"task"` trong Bài học 13.

## Một Lượt MRTR

Máy chủ không thể gọi máy khách trong khi đang xử lý yêu cầu. Thay vào đó, nó trả về kết quả này:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

Máy khách kiểm tra xem mình có hỗ trợ Sampling hay không, áp dụng các chính sách phê duyệt và mô hình của mình, và nhận về phản hồi từ mô hình. Sau đó nó gửi một yêu cầu mới với một id JSON-RPC khác:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

Lần thử lại này không phải là sự tiếp nối của một phiên giao thức. Nó là một yêu cầu mới lặp lại phương thức và các đối số ban đầu, chỉ bổ sung thêm `inputResponses` của lượt hiện tại, và lặp lại chính xác từng byte của `requestState`.

MRTR chỉ được cho phép trên `tools/call`, `prompts/get`, và `resources/read`. Máy chủ tuyệt đối không được trả về `input_required` từ các phương thức không liên quan khác.

## Trạng thái Nhiều Lượt

Bài học này cần hai lệnh gọi mô hình:

1. `pick_files` trả về một mảng JSON.
2. `summary` trả về văn bản tóm tắt cuối cùng.

Mỗi lần thử lại chỉ mang theo các phản hồi cho lượt đó. Do đó máy chủ sẽ đưa giai đoạn (phase) và dữ liệu trung gian đã được xác thực vào giá trị `requestState` tiếp theo.

Hãy xử lý giá trị đó như dữ liệu do kẻ tấn công kiểm soát. Chỉ ký một tên giai đoạn thô là chưa đủ an toàn. Hãy ràng buộc trạng thái với:

- chủ thể đã được xác thực (authenticated principal), không phải giá trị `clientInfo` do máy khách tự khai báo;
- phương thức khởi nguồn;
- mã băm (digest) của các đối số ban đầu;
- thời hạn hết hạn ngắn;
- giai đoạn hiện tại và các giá trị trung gian đã được xác thực.

Sử dụng HMAC khi không yêu cầu tính bảo mật tuyệt đối cho nội dung trạng thái. Sử dụng mã hóa kèm xác thực (authenticated encryption) khi máy khách không được phép đọc nội dung trạng thái. Từ chối chữ ký sai, giá trị hết hạn, chủ thể bị thay đổi hoặc các đối số bị biến đổi bằng mã lỗi `-32602`.

Máy khách không được phân tích cú pháp hoặc sửa đổi `requestState`. Nhiệm vụ duy nhất của nó là gửi lại chính xác chuỗi ký tự đó trong lần gọi lại.

## Sở thích Mô hình là các Gợi ý

`costPriority`, `speedPriority`, và `intelligencePriority` là các sở thích độc lập. Chúng không phải là một phân phối xác suất và không cần có tổng bằng 1. Máy khách có thể bỏ qua chúng vì chính máy khách mới là bên nắm quyền quyết định chính sách mô hình.

Hãy giữ `includeContext` ở giá trị `"none"` nếu bạn duy trì một luồng Sampling kế thừa. Các chế độ ngữ cảnh khác làm tăng nguy cơ rò rỉ dữ liệu và bản thân chúng cũng đã bị ngừng khuyến nghị. Chỉ truyền ngữ cảnh tường minh tối thiểu trong yêu cầu.

## Các Bất biến An toàn

Máy khách là ranh giới tin cậy (trust boundary) đối với các yêu cầu Sampling được nhúng.

- Hiển thị cho người dùng thấy máy chủ đang yêu cầu mô hình làm gì khi chính sách đòi hỏi sự phê duyệt.
- Giới hạn số lượt MRTR. Nếu không, một máy chủ độc hại có thể tạo ra một vòng lặp làm cạn kiệt chi phí gọi mô hình.
- Xác thực mọi phản hồi lấy mẫu trước khi sử dụng nó làm tên tệp, URL, hoặc đầu vào cho công cụ.
- Giới hạn số byte và số lượng token trên mỗi lượt.
- Từ chối một yêu cầu đầu vào không được khai báo trong các khả năng hiện thời của máy khách.
- Giữ đầu ra của mô hình nằm ngoài các quyết định ủy quyền bảo mật.
- Ghi nhật ký (log) phương thức khởi nguồn và khóa yêu cầu đầu vào mà không ghi lại nội dung lời nhắc nhạy cảm.

`clientInfo` và `serverInfo` chỉ là siêu dữ liệu hiển thị và chẩn đoán. Tuyệt đối không bao giờ sử dụng một trong hai làm danh tính đã xác thực bảo mật.

```figure
t3-sampling-flip
```

## Xây dựng

Tệp `code/main.py` triển khai toàn bộ luồng hai lượt mà không dùng bất kỳ gói thư viện bên thứ ba nào:

- `server/discover` trả về `supportedVersions`, quảng bá hỗ trợ công cụ và trả về gợi ý bộ nhớ đệm.
- `tools/list` trả về bộ mô tả `summarize_repo` tất định, có thể lưu đệm với lược đồ đầu vào dạng đối tượng.
- `tools/call` xác thực siêu dữ liệu trên từng yêu cầu.
- Kết quả đầu tiên nhúng `sampling/createMessage` để chọn tệp.
- Lần thử lại đầu tiên xác thực kết quả mô hình và nhúng yêu cầu thứ hai.
- `requestState` được bảo vệ bằng HMAC mang giai đoạn giữa các yêu cầu độc lập.
- Kết quả cuối cùng sử dụng `resultType: "complete"`.

Mô hình ứng dụng chủ giả lập giúp ví dụ hoạt động một cách tất định. Chỉ thay thế `fake_host_model` khi kết nối với một ứng dụng chủ thực tế. Máy trạng thái phía máy chủ phải luôn giữ tính tất định và có thể kiểm thử được.

## Sử dụng

Từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

Các điểm kiểm tra kỳ vọng:

- Khám phá trả về kết quả hoàn chỉnh với `ttlMs` và `cacheScope`.
- Khám phá công cụ trả về cùng một bộ mô tả đã sắp xếp với `resultType`, danh tính máy chủ và gợi ý bộ đệm.
- Thiếu khả năng và phiên bản không được hỗ trợ trả về dữ liệu lỗi chính xác của `-32021` và `-32022`.
- Một thông báo không có id không tạo ra phản hồi JSON-RPC nào.
- Các id yêu cầu là `[1, 2, 3]`, chứng minh mỗi lượt MRTR là hoàn toàn độc lập.
- Hai kết quả đầu tiên có `resultType: "input_required"`.
- Kết quả cuối cùng là `complete` và chứa các tệp đã chọn cùng bản tóm tắt.
- Việc thay đổi các đối số ban đầu trong lần thử lại sẽ làm thất bại bước kiểm tra trạng thái yêu cầu.

## Đóng gói / Triển khai

Tệp `outputs/skill-sampling-loop-designer.md` hiện đóng vai trò là một bộ lập kế hoạch di chuyển. Trước hết, nó quyết định xem tính năng Sampling có nên được loại bỏ để chuyển sang tích hợp mô hình trực tiếp hay không. Nếu khả năng tương thích là bắt buộc, nó sẽ tạo ra các lượt MRTR, cơ chế ràng buộc trạng thái, cổng kiểm soát khả năng, ngân sách, xác thực và kế hoạch gỡ bỏ.

## Bài tập

1. Thay đổi phản hồi chọn tệp thành JSON không hợp lệ. Xác nhận máy chủ trả về mã lỗi `-32602` thay vì tin tưởng mù quáng vào đầu ra của mô hình.
2. Thay đổi đối số `audience` giữa lệnh gọi đầu tiên và lệnh thử lại. Giải thích tại sao trạng thái đã niêm phong lại ngăn chặn việc tái sử dụng chéo yêu cầu.
3. Thêm một lượt thứ ba yêu cầu ứng dụng chủ phê bình bản tóm tắt. Mang bản tóm tắt trước đó bên trong trạng thái đã ký và giới hạn toàn bộ luồng ở mức tối đa ba lượt.
4. Loại bỏ Sampling bằng cách thay thế hàm gọi lại ứng dụng chủ giả lập bằng một bộ điều hợp mô hình do máy chủ sở hữu. Liệt kê các trách nhiệm về phê duyệt, thanh toán và khả năng quan sát được chuyển sang máy chủ.
5. Thêm một bài kiểm thử hết hạn sử dụng giá trị trạng thái đã quá hạn một giây so với thời hạn của nó.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa trong đặc tả 2026-07-28 |
|---|---|
| Lấy mẫu (Sampling) | Tính năng đã ngừng khuyến nghị yêu cầu mô hình của máy khách tạo văn bản hoàn thành |
| MRTR | Mẫu thử lại không lưu trạng thái cho đầu vào của máy khách cần thiết trong một yêu cầu |
| `InputRequiredResult` | Kết quả có `resultType: "input_required"` |
| `inputRequests` | Ánh xạ do máy chủ gán chứa các yêu cầu gợi mở, lấy mẫu hoặc thư mục gốc được nhúng |
| `inputResponses` | Kết quả từ máy khách của lượt hiện tại có khóa tương ứng như `inputRequests` |
| `requestState` | Trạng thái máy chủ ẩn được máy khách lặp lại chính xác và được máy chủ xác minh |
| `resultType` | Trường phân biệt bắt buộc cho các kết quả MCP hiện đại |
| Tích hợp mô hình trực tiếp (Direct model integration) | Phương án thay thế khuyến nghị cho các máy chủ mới cần suy luận mô hình |
| Cổng kiểm soát khả năng (Capability gate) | Quy tắc ngăn chặn việc gửi yêu cầu nhúng mà máy khách không công bố hỗ trợ |
| Ngân sách vòng lặp (Loop budget) | Số lượt, token, byte, thời gian và chi tiêu tối đa được phép cho thao tác |

## Tính tương thích Kế thừa

Một máy khách bị ghim ở phiên bản 2025-11-25 vẫn có thể sử dụng luồng `sampling/createMessage` cũ do máy chủ khởi tạo qua một kết nối trực tiếp. Chỉ giữ hành vi đó trong một bộ điều hợp dành riêng cho phiên bản cũ. Đừng biến đường dẫn có phiên làm việc thành kiến trúc cho một máy chủ 2026-07-28.

Các SDK chính thức có thể dịch các trình xử lý `input_required` hiện đại cho các máy chủ ngang hàng cũ hơn. Lớp đệm (shim) đó là một ranh giới tương thích, không phải là sự cho phép thêm logic phụ thuộc vào phiên làm việc mới.

## Đọc thêm

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Nhật ký thay đổi MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [Thông báo ngừng khuyến nghị Sampling trong MCP](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [Khám phá Máy chủ MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
