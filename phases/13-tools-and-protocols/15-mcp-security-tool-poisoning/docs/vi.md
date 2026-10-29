# Bảo mật MCP: Nhiễm độc siêu dữ liệu, Định tuyến và Trạng thái MRTR

> Phi trạng thái không có nghĩa là không cần tin cậy. Điều đó có nghĩa là mỗi yêu cầu đều phải tự phơi bày đầy đủ bằng chứng mà máy chủ và cổng kết nối (gateway) cần để xác thực lệnh gọi một cách độc lập.

**Loại:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 13 · 07 (máy chủ MCP), Phase 13 · 08 (máy khách MCP)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xử lý các mô tả công cụ, chú thích (annotations), thông tin client và thông tin server như dữ liệu không đáng tin cậy.
- Phát hiện tình trạng nhiễm độc siêu dữ liệu (metadata poisoning), các thay đổi bộ mô tả (descriptor changes), và xung đột tên xuyên máy chủ (cross-server name collisions).
- Xác thực siêu dữ liệu yêu cầu chuẩn 2026-07-28 và các tiêu đề định tuyến Streamable HTTP.
- Bảo vệ `requestState` của chu trình MRTR chống giả mạo và ràng buộc xác nhận vào các đối số chính xác.
- Áp dụng phân quyền ủy quyền (authorization) và giới hạn tốc độ (rate limits) dựa trên chủ thể (principal), thay vì dựa trên phiên giao thức đã bị loại bỏ.

## Vấn đề đặt ra

Mô hình đọc mô tả công cụ để quyết định nên gọi công cụ nào. Bộ định tuyến đọc tên công cụ để quyết định gửi yêu cầu đến đâu. Người dùng đọc các nhãn để quyết định phê duyệt hành động nào. Một bộ mô tả độc hại duy nhất có thể nhắm vào cả ba đối tượng trên.

Tài liệu hướng dẫn bảo mật chính thức của MCP chỉ rõ: các mô tả và chú thích cần được coi là không đáng tin cậy trừ khi chúng xuất phát từ một máy chủ đáng tin cậy. Ngay cả khi đó, mức độ tin cậy trong quá trình triển khai vẫn có thể thay đổi. Một bản cập nhật máy chủ, một gói phần mềm bị xâm nhập, một sai sót trong sổ đăng ký (registry), hay một pha gộp dữ liệu tại gateway đều có thể làm biến đổi những gì mô hình nhìn thấy.

Giao thức hiện tại cũng thay đổi ranh giới bảo mật. Trong phiên bản 2026-07-28, không còn quy trình bắt tay lõi và không còn phiên giao vận. Một thiết kế bảo mật chỉ khóa sự phê duyệt, giới hạn tốc độ, hay lịch sử kiểm toán bằng `Mcp-Session-Id` không còn là một thiết kế phù hợp với chuẩn hiện hành.

## Khái niệm cốt lõi

### Bảy bề mặt tấn công cần kiểm tra

Sử dụng một danh sách cụ thể thay vì những lời khuyên cẩn trọng mơ hồ:

1. **Nhiễm độc siêu dữ liệu (Metadata poisoning).** Phần mô tả chứa các chỉ thị không liên quan đến hành vi công cụ đã công bố.
2. **Lật kèo bộ mô tả (Descriptor rug pull).** Tên, mô tả, lược đồ hoặc chú thích đã được phê duyệt trước đó bị thay đổi âm thầm.
3. **Che khuất xuyên máy chủ (Cross-server shadowing).** Hai backend cùng phơi bày một tên công cụ không định danh đầy đủ (unqualified name) và việc định tuyến âm thầm chọn một trong hai.
4. **Xung đột giữa tiêu đề và phần thân (Header and body confusion).** `Mcp-Method` hoặc `Mcp-Name` không khớp với yêu cầu JSON-RPC.
5. **Leo thang năng lực (Capability escalation).** Một đối tác tuyên bố hỗ trợ một phần mở rộng hoặc tính năng client, và máy chủ nhầm lẫn tuyên bố đó với quyền hạn được ủy quyền.
6. **Giả mạo trạng thái MRTR (MRTR state tampering).** Client thay đổi `requestState`, trả lời một câu hỏi khác, hoặc tái sử dụng sự xác nhận với các đối số khác.
7. **Nhầm lẫn danh tính chuỗi cung ứng (Supply-chain identity confusion).** Coi một tên hiển thị quen thuộc là bằng chứng về danh tính nhà phát hành hoặc máy chủ.

Các bề mặt tấn công này thường chồng chéo lên nhau. Cơ chế ghim băm (hash pinning) giúp xử lý các thay đổi bộ mô tả nhưng không chứng minh được bộ mô tả đầu tiên là an toàn. Quét tĩnh phát hiện các cụm từ lộ liễu nhưng bỏ sót các chỉ thị tinh vi. Phân vùng không gian tên (namespacing) ngăn chặn một lớp xung đột nhưng không ngăn được một máy chủ độc hại đã được đặt không gian tên. Hãy kết hợp nhiều tầng kiểm soát.

### Phong bì yêu cầu hiện tại là bằng chứng, không phải danh tính

Mọi yêu cầu chuẩn 2026-07-28 đều chứa:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

Xác thực phiên bản và hình thái năng lực trên từng yêu cầu. Sử dụng năng lực để chọn cấu trúc phản hồi tương thích. Tuyệt đối không dùng `clientInfo` làm chủ thể đã xác thực (authenticated principal). Đó là dữ liệu tự khai báo.

Cảnh báo tương tự áp dụng cho `io.modelcontextprotocol/serverInfo` trong siêu dữ liệu kết quả. Nó hữu ích cho việc ghi log và gỡ lỗi, nhưng không phải là chứng chỉ, bằng chứng sổ đăng ký, hay quyết định ủy quyền.

### Xác thực định tuyến trước khi áp dụng chính sách

Đối với `tools/call`, Streamable HTTP bao gồm:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

Phương thức trong tiêu đề phải trùng với phương thức trong thân yêu cầu. Tên trong tiêu đề phải trùng với `params.name`. Từ chối ngay lập tức sự sai lệch với mã lỗi `-32020` trước khi chọn backend, áp dụng RBAC, hoặc tiêu thụ token giới hạn tốc độ.

Thứ tự xử lý này loại bỏ một sự mơ hồ phổ biến: một thành phần ủy quyền cho phần thân trong khi một thành phần khác lại định tuyến theo tiêu đề.

Quy trình xác thực đường truyền tuân theo một trình tự nghiêm ngặt. Xác thực kiểu JSON-RPC và kiểu siêu dữ liệu, so sánh giá trị tiêu đề với phần thân, sau đó kiểm tra xem phiên bản tương ứng có được hỗ trợ hay không. Tiêu đề không khớp trả về HTTP 400 với `-32020`. Nếu tiêu đề và thân thống nhất về một phiên bản không được hỗ trợ, trả về HTTP 400 với `-32022` và `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}`. Một phương thức không xác định trả về HTTP 404 với `-32601`.

Mọi đối tượng lỗi đều bao gồm trường `data` tùy chọn khi hợp đồng yêu cầu thông tin khôi phục có cấu trúc. Một thông báo không có trường `id`, do đó nó không bao giờ nhận được phản hồi thành công hoặc lỗi JSON-RPC. Một thông báo HTTP được chấp nhận sẽ trả về mã 202 với phần thân rỗng.

### Ghim băm toàn bộ bộ mô tả

Chỉ băm riêng phần mô tả sẽ bỏ sót các thay đổi về lược đồ (schema) và chú thích. Hãy chuẩn tắc hóa (canonicalize) và băm các trường của bộ mô tả mà người dùng đã phê duyệt:

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

Lưu trữ giá trị băm (digest) dưới một khóa định danh đầy đủ như `notes.export`, cùng với bằng chứng về nhà phát hành và thời gian phê duyệt trong hệ thống thực tế.

Trên mỗi lần làm mới dữ liệu:

- Khóa chưa xác định: cách ly chờ xét duyệt.
- Cùng khóa nhưng khác mã băm: cách ly do hành vi lật kèo (rug pull) cho đến khi được tái phê duyệt.
- Trùng lặp tên không định danh đầy đủ: bắt buộc phải dùng không gian tên tất định.
- Cảnh báo từ bộ quét: chặn lại và duyệt xét toàn bộ bộ mô tả.

Sự trùng khớp mã băm chỉ chứng minh tính ổn định, không chứng minh tính an toàn. Một bộ mô tả bị nhiễm độc vẫn là nhiễm độc ngay cả khi được ghim băm hoàn hảo.

### Quét tĩnh là bẫy cảnh báo sớm

Các mẫu quy tắc đơn giản có thể gắn cờ các thẻ vai trò (role tags), ghi đè chỉ thị (instruction overrides), che giấu hành vi (concealment), truy cập bí mật, và các đích đến mạng bị che khuất. Chúng đủ nhẹ để chạy lúc cài đặt và trong quy trình CI.

Chúng không phải là chứng minh về mặt ngữ nghĩa. Một mô tả an toàn có thể chứa cụm từ bị gắn cờ bên trong một cảnh báo hợp lệ. Một mô tả độc hại có thể né tránh toàn bộ các cụm từ đó. Hãy coi kết quả bộ quét như bằng chứng phục vụ xem xét, chứ không phải điểm số chứng minh vô tội tự động.

### Đặt không gian tên trước khi hợp nhất

Giả sử hai máy chủ đều công bố công cụ `search`. Tuyệt đối không để thứ tự phát hiện dịch vụ quyết định máy chủ nào chiến thắng.

```text
notes.search
issues.search
```

Tên có tiền tố định danh đầy đủ (qualified name) chính là tên công khai tại gateway. Ghi nhận ánh xạ tới backend một cách riêng biệt. Các tên gọi ổn định giúp việc phê duyệt, kiểm toán, ghim băm và định tuyến theo `Mcp-Name` đều tham chiếu tới cùng một đối tượng.

### Năng lực là các tuyên bố về khả năng tương thích

Đối tượng `clientCapabilities` trên từng yêu cầu cho máy chủ biết client có thể xử lý những tính năng giao thức nào. Nó không hề cấp quyền cho client truy cập công cụ, dữ liệu hay hành động.

Quyền hạn ủy quyền vẫn phải đến từ chủ thể đã xác thực và chính sách tài nguyên. Trình tự thực hiện là:

1. Xác thực thông tin xác thực tầng giao vận.
2. Xác thực phiên bản, tiêu đề và hình thái yêu cầu.
3. Kiểm tra tính tương thích về năng lực.
4. Ủy quyền cho chủ thể, công cụ, tài nguyên và đối số.
5. Thực thi hoặc yêu cầu người dùng nhập thông tin.

### Bảo vệ xác nhận MRTR phi trạng thái

Một công cụ có tính hệ trọng có thể cần sự xác nhận của người dùng. MCP hiện tại sử dụng Yêu cầu nhiều vòng truyền nhận (MRTR) thay cho lệnh gọi ngược (callback) từ server đến client.

Phản hồi đầu tiên:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

Client lấy dữ liệu đầu vào và thử lại phương thức gốc với một id JSON-RPC mới:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Mỗi giá trị `inputRequests` là một yêu cầu nhúng hoàn chỉnh với `method` và `params`. Khóa của nó phải khớp với mục tương ứng trong `inputResponses`. Một gợi mở dạng biểu mẫu (form elicitation) sử dụng `requestedSchema` có gốc là đối tượng, và client bắt buộc phải công bố năng lực gợi mở dạng biểu mẫu trước khi máy chủ yêu cầu nó.

Năng lực hiện tại có hai cách khai báo biểu mẫu hợp lệ: `{"elicitation":{}}` ngầm định hỗ trợ gợi mở biểu mẫu, trong khi `{"elicitation":{"form":{}}}` tuyên bố điều đó một cách tường minh. Một khai báo chỉ có URL như `{"elicitation":{"url":{}}}` không hỗ trợ yêu cầu biểu mẫu. Máy chủ sẽ trả về HTTP 400 với `-32021` và `data.requiredCapabilities` đặt thành `{"elicitation":{"form":{}}}`.

Hãy đối xử với `requestState` như dữ liệu thù địch. Ký số hoặc mã hóa nó, xác thực nó, và gắn kết nó với phương thức, công cụ, các đối số chính xác, mục đích, thời hạn hết hạn, chủ thể, và một số ngẫu nhiên dùng một lần (nonce) khi vấn đề chống phát lại (replay) là quan trọng. Mã nguồn bài học sử dụng HMAC và cơ chế so khớp đối số chính xác để làm rõ ranh giới này.

Sổ cái ghi nhận nonce không được nằm cục bộ trong một đối tượng gateway đơn lẻ. Mô hình có thể chạy được truyền vào một kho lưu trữ chống phát lại có giới hạn và tự dọn dẹp theo TTL, có thể được dùng chung giữa nhiều phiên bản gateway. Hành động xác nhận nguyên tử (atomic claim) của nó chính là ranh giới thực thi: chỉ có một sự chấp nhận hợp lệ hoặc một sự từ chối rõ ràng mang tính kết thúc mới tiêu thụ trạng thái. Một phản hồi sai định dạng hoặc hành động `cancel` sẽ không thực thi gì cả và vẫn có thể thử lại cho đến khi hết hạn. Một cụm máy chủ sản xuất cần cùng cơ chế xác nhận có điều kiện này trong kho lưu trữ bền vững dùng chung.

Không lưu trữ ngữ cảnh xác nhận ẩn bên trong một phiên giao thức. Bất kỳ phiên bản máy chủ nào cũng phải có khả năng xác thực lượt thử lại.

### Nguyên tắc hai người đối với các lệnh gọi rủi ro cao

Phân loại một lệnh gọi theo ba trục:

- Nó tiêu thụ dữ liệu đầu vào không đáng tin cậy.
- Nó có thể truy cập dữ liệu nhạy cảm.
- Nó gây ra một hành động hệ trọng ở môi trường bên ngoài.

Một bước tự động duy nhất không được kết hợp cả ba yếu tố trên. Hãy chia nhỏ nó, giảm đặc quyền, hoặc yêu cầu sự nhập liệu tường minh từ người dùng qua MRTR. Đây là một quy tắc kinh nghiệm thiết kế (design heuristic), không phải một năng lực giao thức.

### Thu hẹp thẩm quyền trước khi thực thi

Chỉ riêng tính phi trạng thái không tạo nên sự an toàn. Nó loại bỏ lịch sử giao thức ẩn, nhưng một yêu cầu tự chứa vẫn có thể sai khiến một bộ xử lý quá nhiều quyền hạn làm rò rỉ dữ liệu hoặc thực hiện một thay đổi không thể hoàn tác. An toàn đến từ việc thu hẹp thẩm quyền ở từng ranh giới:

1. **Động từ định kiểu (Typed verb).** Phơi bày một thao tác có phạm vi giới hạn rõ ràng như `archive_note`, thay vì một công cụ chung chung như `run` hay `request` vốn có thể thực hiện nhiều quyền hạn không liên quan.
2. **Đối số đã xác thực.** Sử dụng lược đồ đóng khi khả thi, từ chối các trường không xác định, chuẩn hóa định danh một lần duy nhất, giới hạn kích thước, và xác thực quyền sở hữu đích đến, đối tượng thuê và tài nguyên trước khi đánh giá chính sách.
3. **Ủy quyền hiện hành.** Ràng buộc chủ thể đã xác thực vào chính xác động từ, tài nguyên, môi trường và các đối số đã chuẩn hóa. Chú thích công cụ và năng lực client không trao quyền hạn này.
4. **Phê duyệt gắn liền với hành động.** Đối với lệnh gọi hệ trọng, ràng buộc sự phê duyệt vào mã băm của động từ định kiểu và các đối số đã chuẩn hóa, cùng với chủ thể, thời hạn hết hạn và chính sách dùng một lần. Bất kỳ trường nào thay đổi đều đòi hỏi một quyết định mới.
5. **Từ chối hạng nhất (First-class refusal).** Mô hình hóa việc từ chối, phê duyệt hết hạn, người dùng khước từ và đích đến không an toàn như những kết quả bình thường không gây ra bất kỳ tác dụng phụ (side effect) nào. Đừng chuyển hóa sự từ chối thành một công cụ hạ cấp yếu hơn.
6. **Bằng chứng kiểm toán đã loại bỏ dữ liệu nhạy cảm (Redacted audit evidence).** Ghi lại ai đã yêu cầu, bộ mô tả và phiên bản chính sách được chấp thuận nào đã được dùng, mục tiêu chuẩn hóa nào đã được ủy quyền, lý do tại sao quyết định cho phép hoặc từ chối, và liệu việc thực thi đã bắt đầu hay chưa. Lưu trữ mã băm hoặc các giá trị đã che dấu thay vì các bí mật.

Mỗi bước thu hẹp những gì thành phần tiếp theo có thể làm. Bộ xử lý cuối cùng chỉ nên nhận một lệnh nghiệp vụ đã được xác thực, thay vì văn bản thô của mô hình kèm theo thông tin xác thực rộng lớn. Lặp lại toàn bộ chuỗi này trên một lượt thử lại MRTR, cập nhật tác vụ, hoặc lệnh gọi được gateway chuyển tiếp. Sự phê duyệt trước đó không biến các yêu cầu sau thành lưu lượng phiên đáng tin cậy.

### Các lộ trình tương tác hiện tại và kế thừa

Roots, Sampling, và Logging đã bị coi là lỗi thời (deprecated) đối với các bản triển khai mới chuẩn 2026-07-28. Một gateway có thể giữ lại mã kênh yêu cầu kiểu cũ chỉ như một nhánh tương thích được kiểm soát bằng phiên bản.

Không xây dựng cơ chế phòng thủ mới xoay quanh một bộ giới hạn lấy mẫu (sampling limiter) theo từng phiên. Hãy áp dụng hạn ngạch cho chủ thể đã xác thực, bên phát hành, tài nguyên, công cụ và cửa sổ thời gian. Đối với các tác vụ tương tác hiện tại, hãy kiểm tra các yêu cầu và phản hồi đầu vào MRTR.

### Các bước kiểm tra giao vận phi trạng thái

- Chấp nhận các thông điệp MCP hiện đại tại điểm cuối POST duy nhất.
- Trả về 405 cho các yêu cầu GET và DELETE kiểu mới.
- Không phát hành hay phụ thuộc vào `Mcp-Session-Id`.
- Bỏ qua các tiêu đề phiên và phát lại kiểu cũ dưới vai trò là các đầu vào xác thực thẩm quyền.
- Trả về JSON hoặc SSE trong phạm vi yêu cầu cho lệnh POST đó.
- Chỉ sử dụng `subscriptions/listen` cho các thông báo thay đổi dài hạn đã được chọn tham gia.

````figure
tp-tool-poisoning
````

## Xây dựng hệ thống

Tệp `code/main.py` triển khai một mô hình gateway bảo mật nhỏ chạy nội tiến trình. Nó chuẩn tắc hóa và ghim băm toàn bộ bộ mô tả công cụ, báo cáo tình trạng nhiễm độc siêu dữ liệu và che khuất công cụ, xác thực phong bì yêu cầu hiện đại và các giá trị định tuyến, đồng thời thực hiện thao tác xuất dữ liệu được xác nhận qua hai vòng với `requestState` có chữ ký và một kho chống phát lại dùng chung được truyền vào.

Mô hình bắt đầu sau khi một bộ điều hợp HTTP đã phân tích phần thân JSON và các tiêu đề định tuyến. Nó không trực tiếp xác thực `Content-Type` hay `Accept`. Hãy kết nối cùng một bộ điều phối này với bộ điều hợp Streamable HTTP hoàn chỉnh của Bài học 09, vốn yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy chương trình:

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Mẫu ví dụ cố tình làm biến đổi một bộ mô tả. Bộ quét và phép so sánh mã băm tạo ra các kết quả độc lập. Quá trình xuất dữ liệu sau đó minh chứng cho phản hồi `input_required` và lượt thử lại phi trạng thái.

## Thử nghiệm thực tế

Thay thế `SAFE_TOOLS` bằng một ảnh chụp nhanh đã chuẩn hóa từ chính các máy chủ đã được phê duyệt của bạn. Giữ thông tin xác thực và bí mật bên ngoài ảnh chụp nhanh này. Duyệt xét mọi bộ mô tả mới hoặc bị thay đổi trước khi cập nhật mã băm của nó.

Tại gateway, hãy chạy các bước kiểm tra tương tự trong quá trình phát hiện dịch vụ và lặp lại trước khi điều phối lệnh gọi. Bộ nhớ đệm có thể giảm tải công việc phát hiện, nhưng một phê duyệt được lưu đệm phải hết hạn hoặc bị vô hiệu hóa khi bộ mô tả thay đổi.

## Đóng gói sản phẩm

Bài học này đóng gói tệp `outputs/skill-mcp-threat-model.md`. Tài liệu này tạo ra một mô hình mối đe dọa theo giao thức hiện tại xuyên suốt các ranh giới siêu dữ liệu, định tuyến, năng lực, ủy quyền, MRTR, lưu đệm, sổ đăng ký và khả năng tương thích.

## Bài tập mở rộng

1. Ràng buộc chủ thể đã xác thực và quyết định ủy quyền hiện tại vào trạng thái MRTR đã niêm phong, sau đó từ chối lượt thử lại dưới một chủ thể khác.
2. Thay thế kho chống phát lại trong bộ nhớ bằng thao tác chèn có điều kiện bền vững và chứng minh hai tiến trình không thể cùng nhận một nonce.
3. Cố tình gây lỗi sau khi nhận nonce nhưng trước khi xuất dữ liệu mô phỏng. Định nghĩa và kiểm thử giao dịch hoặc quy tắc lũy suy giúp quá trình khôi phục an toàn.
4. Thay đổi `inputSchema` của công cụ mà không đổi phần mô tả. Xác nhận rằng việc ghim băm toàn bộ bộ mô tả sẽ phát hiện ra sự thay đổi này.
5. Thêm một chính sách từ chối lưu đệm công khai khi `tools/list` có sự khác biệt giữa các chủ thể.
6. Mô hình hóa một máy chủ cũ đứng sau gateway. Đặt toàn bộ hành vi bắt tay và phiên làm việc sau một nhánh tương thích `2025-11-25` rõ ràng.

## Các thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| Nhiễm độc siêu dữ liệu (Metadata poisoning) | Các chỉ thị hoặc tuyên bố đánh lừa được nhúng bên trong bộ mô tả công cụ |
| Lật kèo (Rug pull) | Thay đổi đối với một bộ mô tả đã được phê duyệt trước đó |
| Che khuất công cụ (Tool shadowing) | Định tuyến mơ hồ do trùng lặp các tên không có tiền tố định danh đầy đủ |
| Lệch tiêu đề (Header mismatch) | Tiêu đề định tuyến và phần thân JSON-RPC không khớp nhau, lỗi `-32020` |
| Ghim băm (Hash pin) | Mã băm của toàn bộ bộ mô tả đã được phê duyệt |
| MRTR | Mô hình phản hồi và thử lại phi trạng thái đối với dữ liệu do server yêu cầu |
| `requestState` | Giá trị truyền nhận khứ hồi mờ đục phải được xử lý như dữ liệu không đáng tin cậy |
| Tuyên bố năng lực (Capability declaration) | Tuyên bố về khả năng tương thích giao thức, không phải là sự ủy quyền |
| Hỗ trợ biểu mẫu ngầm định (Implicit form support) | Đối tượng năng lực `elicitation` rỗng, tương đương với hỗ trợ biểu mẫu |
| Tên công cụ định danh đầy đủ (Qualified tool name) | Tên gateway ổn định có tiền tố như `notes.search` |

## Đọc thêm

- [Hướng dẫn về an ninh và độ tin cậy trong MCP](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Yêu cầu nhiều vòng truyền nhận (MRTR)](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Giao vận Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Các tính năng bị bãi bỏ](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
