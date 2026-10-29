# Phạm vi Tường minh và Gợi mở Phản hồi Không lưu trạng thái

> Khái niệm Thư mục gốc (Roots) đã bị ngừng khuyến nghị (deprecate) trong MCP 2026-07-28 và chưa từng là một hộp cát bảo mật (security sandbox). Hãy đặt phạm vi vào các đối số công cụ rõ ràng hoặc URI tài nguyên, kiểm tra ủy quyền ngay tại máy chủ, và sử dụng cơ chế MRTR khi một công cụ thực sự cần đầu vào của người dùng. Người dùng nhìn thấy quyết định, mô hình nhìn thấy thẻ định danh, và bất kỳ phiên bản máy chủ nào cũng có thể xử lý lệnh gọi lại.

**Type:** Xây dựng (Build)
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (Máy chủ MCP), Phase 13 · 11 (MRTR không lưu trạng thái)
**Time:** ~60 phút

## Mục tiêu học tập

- Thay thế Roots đã bị ngừng khuyến nghị bằng các tham số không gian làm việc (workspace parameters), URI tài nguyên hoặc cấu hình máy chủ tường minh.
- Phân biệt các gợi ý phạm vi với việc kiểm tra ủy quyền (authorization), bao bọc đường dẫn (path containment), và cô lập hộp cát ở cấp hệ điều hành (operating-system sandboxing).
- Phân phối phương thức `elicitation/create` ở chế độ biểu mẫu (form mode) thông qua kết quả MRTR `input_required`.
- Quảng bá hỗ trợ gợi mở phản hồi (elicitation) trong các khả năng của máy khách trên từng yêu cầu và từ chối các chế độ không được hỗ trợ.
- Xác thực `accept`, `decline`, và `cancel` như các kết quả riêng biệt.
- Ràng buộc việc xác nhận thao tác phá hủy với một chủ thể đã xác thực (authenticated principal), các đối số ban đầu, tập ứng viên và thời hạn hết hạn.

## Hai Vấn đề Trông có vẻ Giống nhau

Một công cụ ghi chú nhận được yêu cầu sau: "Xóa báo cáo TPS cũ."

Máy chủ phải trả lời hai câu hỏi hoàn toàn khác nhau:

1. Thao tác này được phép tác động vào không gian làm việc nào?
2. Người dùng muốn ám chỉ ghi chú nào trong số ba ghi chú phù hợp tìm thấy?

Câu hỏi thứ nhất thuộc về phạm vi và ủy quyền. Câu hỏi thứ hai thuộc về việc làm rõ nghĩa tương tác (interactive disambiguation). Việc trộn lẫn hai vấn đề này dẫn đến những thiết kế nguy hiểm, chẳng hạn như coi một thư mục do máy khách cung cấp là bằng chứng cho thấy người gọi được phép xóa mọi thứ bên trong thư mục đó.

## Roots là một Bề mặt Chuyển tiếp

Các bản sửa đổi MCP trước đây cho phép máy khách công bố danh sách Roots và thông báo cho máy chủ khi danh sách đó thay đổi. Roots vốn chỉ mang tính hướng dẫn thông tin. Chúng không hề giới hạn những gì tiến trình máy chủ có thể đọc, không xác thực quyền hạn của người gọi, và không tạo ra một hộp cát hệ điều hành.

MCP 2026-07-28 chính thức ngừng khuyến nghị `roots/list` và `notifications/roots/list_changed` đối với các thiết kế mới. Hãy ưu tiên một trong các giải pháp thay thế tường minh sau:

- Đối số công cụ `workspaceUri` hoặc `directory` khi phạm vi thay đổi theo từng lệnh gọi.
- Một URI tài nguyên khi thao tác vốn đã nhắm vào một tài nguyên cụ thể.
- Cấu hình máy chủ khi một hệ thống triển khai chỉ sở hữu một không gian làm việc cố định duy nhất.
- Hộp cát tiến trình hoặc hệ thống tệp bị giam cầm (jailed filesystem) khi mã nguồn bắt buộc phải không thể thoát ra về mặt kỹ thuật.

Nếu một bản tích hợp 2026-07-28 hiện có vẫn cần `roots/list` trong giai đoạn chuyển đổi, máy chủ sẽ nhúng nó vào trường `inputRequests` của MRTR. Máy chủ tuyệt đối không được gửi một yêu cầu đảo chiều trực tiếp. Đó là một bộ điều hợp di chuyển; các trình xử lý mới nên chấp nhận phạm vi tường minh thay vì dựa vào Roots.

Mô hình có thể nhìn thấy và lặp lại một thẻ định danh tường minh. Ngược lại, phạm vi phiên truyền tải ẩn giấu rất khó kiểm tra, khó phát lại, khó kiểm toán và khó điều hướng.

### Quy tắc ba tầng

Một URI tường minh vẫn không tự nó cấp quyền. Hãy thực thi đầy đủ cả ba tầng:

1. **Ủy quyền (Authorization):** Chủ thể đã xác thực này có được phép sử dụng không gian làm việc này hay không?
2. **Bao bọc (Containment):** URI mục tiêu sau khi chuẩn hóa có nằm hoàn toàn bên trong ranh giới không gian làm việc được phép hay không?
3. **Hộp cát (Sandbox):** Hệ điều hành có thể ngăn chặn một máy chủ bị xâm nhập thoát ra ngoài hay không?

Máy chủ trong bài học duy trì một danh sách cho phép gồm các URI không gian làm việc đã được ủy quyền, chuẩn hóa các đường dẫn mã hóa phần trăm (percent-encoded), kiểm tra ranh giới thành phần đường dẫn thực tế và kiểm tra lại tính bao bọc ngay trước khi thực hiện thao tác xóa.

Các phép kiểm tra tiền tố chuỗi ngây thơ là hoàn toàn sai lầm:

```text
được phép:   file:///work/notes
kẻ tấn công: file:///work/notes-evil/secret.md
leo thang:   file:///work/notes/%2e%2e/private.md
```

Cả hai đường dẫn độc hại đều bắt đầu bằng một chuỗi ký tự gây hiểu lầm. Hãy chuẩn hóa trước, sau đó so sánh các thành phần đường dẫn. Một máy chủ hệ thống tệp trong môi trường sản xuất cũng phải phòng thủ chống lại các cuộc tấn công tranh chấp liên kết tượng trưng (symbolic-link races) và ngữ nghĩa đường dẫn riêng của từng nền tảng hệ điều hành.

## Gợi mở Phản hồi Vẫn Tồn tại, nhưng Phương thức Phân phối đã Thay đổi

Gợi mở phản hồi (Elicitation) là tính năng máy khách hiện tại dùng để thu thập đầu vào của người dùng trong khi thực hiện `tools/call`, `prompts/get`, hoặc `resources/read`. Tên phương thức vẫn giữ nguyên là `elicitation/create`. Điều thay đổi chính là hướng đi của luồng truyền tải trên đường truyền.

Một máy chủ 2026-07-28 không gửi yêu cầu JSON-RPC đảo chiều. Thay vào đó, nó trả về một `InputRequiredResult`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

Ứng dụng chủ kết xuất (render) biểu mẫu. Người dùng có thể chấp nhận, từ chối rõ ràng hoặc bỏ qua. Sau đó máy khách thử lại lệnh `tools/call` ban đầu với một id mới tinh:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Hoàn toàn không có phiên giao thức nào giữa hai lệnh gọi. Máy chủ xác minh trạng thái được phản hồi, xác thực phản hồi đối chiếu với lược đồ mong đợi, kiểm tra xem ghi chú được chọn có nằm trong tập ứng viên đã ký hay không, tái ủy quyền không gian làm việc, kiểm tra lại tính bao bọc, và sau đó mới thực hiện thao tác xóa.

## Đàm phán Khả năng Diễn ra trên Từng Yêu cầu

Một máy khách hỗ trợ gợi mở phản hồi ở chế độ biểu mẫu sẽ khai báo:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

Một khai báo elicitation rỗng, `"elicitation": {}`, vẫn tương đương với việc chỉ hỗ trợ biểu mẫu để đảm bảo tương thích ngược. Khai báo tường minh `"elicitation": {"form": {}}` cũng hỗ trợ chế độ biểu mẫu. Một khai báo chỉ có URL, `"elicitation": {"url": {}}`, thì không hỗ trợ biểu mẫu. Máy chủ tuyệt đối không được nhúng một chế độ vắng mặt trong khả năng của yêu cầu hiện tại, ngay cả khi yêu cầu trước đó từng quảng bá nó.

Mọi yêu cầu cũng mang `io.modelcontextprotocol/protocolVersion`. Phiên bản bị thiếu hoặc không phải chuỗi ký tự sẽ trả về `-32602`. Một chuỗi không được hỗ trợ trả về `-32022` cùng dữ liệu chính xác của `supported` và `requested`. Thiếu hỗ trợ elicitation hoặc chỉ hỗ trợ URL sẽ trả về mã `-32021` với `data.requiredCapabilities` đặt thành `{"elicitation":{"form":{}}}`.

Một phong bì không có trường `id` của JSON-RPC là một thông báo. Hãy xử lý nó mà không phát ra phản hồi thành công hoặc lỗi JSON-RPC. Trên Streamable HTTP, một thông báo được chấp nhận sẽ nhận về HTTP `202 Accepted` không có phần thân.

Trường `clientInfo` nên được đưa vào để phục vụ mục đích chẩn đoán, nhưng nó do máy khách tự khai báo và không thể dùng để định danh người dùng phục vụ ủy quyền.

Máy chủ triển khai `server/discover` và trả về `supportedVersions`, các khả năng, `ttlMs`, và `cacheScope` với `resultType: "complete"`. Nó không quảng bá Roots cho thiết kế hiện đại này. Vì quảng bá công cụ, nó cũng phải triển khai phương thức bắt buộc `tools/list`. Kết quả đó trả về bộ mô tả `notes_delete` tất định, đối tượng `inputSchema` hợp lệ, siêu dữ liệu danh tính máy chủ và các gợi ý bộ nhớ đệm công khai.

## Chế độ Biểu mẫu

Chế độ biểu mẫu (Form mode) sử dụng một lược đồ JSON Schema thu gọn được thiết kế cho các hộp thoại dễ sử dụng. Gốc là một đối tượng và các thuộc tính của nó là các trường nguyên thủy phẳng hoặc các mảng enum được hỗ trợ. Các đối tượng lồng nhau sâu và các lược đồ tài liệu đa mục đích không thuộc về một hộp thoại xác nhận.

Sử dụng chế độ biểu mẫu cho:

- chọn một trong số nhiều ứng viên;
- xác nhận một thao tác phá hủy (destructive operation);
- thu thập các tùy chọn không nhạy cảm;
- thu thập một lượng nhỏ giá trị mà chính người dùng, chứ không phải mô hình, phải đưa ra quyết định.

Không sử dụng chế độ biểu mẫu cho mật khẩu, khóa API, mã thông báo truy cập (access tokens) hoặc thông tin thanh toán. Những bí mật đó sẽ đi qua máy khách MCP và có thể lọt vào nhật ký hoặc ngữ cảnh của mô hình.

Máy chủ phải xác thực lại nội dung trả về. Việc xác thực biểu mẫu phía máy khách giúp cải thiện trải nghiệm người dùng nhưng không tạo ra sự tin cậy về mặt bảo mật.

## Chế độ URL

Chế độ URL gửi một địa chỉ web bảo mật cho một tương tác ngoài băng (out-of-band interaction):

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

Hãy sử dụng chế độ này khi thông tin nhạy cảm phải đi thẳng vào một quy trình web do máy chủ kiểm soát, chẳng hạn như quy trình ủy quyền của bên thứ ba (OAuth). Máy khách hiển thị đầy đủ đích đến và thu thập sự đồng thuận trước khi mở liên kết. Máy khách không được phép tìm nạp trước (prefetch) URL đó.

Một phản hồi `accept` chỉ có nghĩa là người dùng đã đồng ý mở URL. Nó không chứng minh rằng quy trình bên ngoài đã hoàn thành. Khi thử lại, máy chủ sẽ kiểm tra trạng thái nội bộ của chính mình và quyết định hoàn tất hoặc trả về một kết quả `input_required` khác.

Gợi mở URL không thay thế cho việc ủy quyền giữa máy khách MCP và máy chủ MCP. Nó phục vụ cho một tương tác bên ngoài mà máy chủ MCP cần thực hiện thay mặt người dùng. Máy chủ phải ràng buộc người dùng trình duyệt với cùng một chủ thể đã xác thực đã bắt đầu thao tác MCP ban đầu.

## Các Nhánh Phản hồi

Hãy xử lý các hành động như các quyết định sản phẩm, không phải các bí danh tương đương:

| Hành động | Ý nghĩa | Hành vi máy chủ an toàn |
|---|---|---|
| `accept` | Người dùng đã gửi tương tác | Xác thực nội dung và tiếp tục thực hiện |
| `decline` | Người dùng từ chối rõ ràng | Trả về một kết quả từ chối hoàn chỉnh, không phải lỗi |
| `cancel` | Người dùng bỏ qua hoặc không thể hoàn thành | Dừng lại an toàn và cho phép thử lại sau |

Tuyệt đối không diễn giải nội dung bị thiếu như là sự đồng thuận. Tuyệt đối không biến hành động từ chối thành một vòng lặp hỏi lại liên tục.

## Bảo vệ Trạng thái MRTR Phá hủy

Danh sách ứng viên không thể chỉ nằm trong một lời nhắc hoặc giá trị Base64 không được ký. Máy khách kiểm soát mọi thứ mà nó gửi ngược lại.

Bài học thực hiện ký một tải trọng trạng thái chứa:

- chủ thể đã xác thực (authenticated principal);
- phương thức khởi nguồn;
- mã băm của `workspaceUri` và `title`;
- danh sách các id ghi chú được phép hiển thị trong biểu mẫu;
- giai đoạn thao tác;
- thời hạn hết hạn ngắn.

Trước khi thực hiện biến đổi dữ liệu (mutation), máy chủ cũng kiểm tra lại bản ghi ghi chú thực tế đang tồn tại. Điều này giúp phát hiện các cuộc tranh chấp xóa dữ liệu và trường hợp mục tiêu bị di chuyển ra ngoài không gian làm việc sau khi biểu mẫu đã hiển thị.

Đối với một thao tác tài chính dùng một lần hoặc hành động không thể đảo ngược, chỉ riêng HMAC không ngăn được một trạng thái hợp lệ bị phát lại (replay) trong thời hạn hiệu lực của nó. Hãy lưu trữ và tiêu thụ một giá trị dùng một lần (nonce) chính xác một lần duy nhất trong một kho lưu trữ chống phát lại (replay store) được chia sẻ bởi mọi phiên bản trình xử lý. Bài học tiêm vào một kho lưu trữ có giới hạn được cắt tỉa theo TTL và giữ quyền xác nhận nguyên tử (atomic claim) trong khi thực hiện thao tác xóa trong bộ nhớ. Một cơ sở dữ liệu sản xuất nên kết hợp việc xác nhận nonce và thao tác biến đổi dữ liệu trong cùng một giao dịch (transaction) hoặc ranh giới ghi có điều kiện tương đương.

Xác thực tính hợp lệ của tương tác trước khi xác nhận tiêu thụ nonce. Một phản hồi sai định dạng hoặc hành động `cancel` sẽ không thực hiện biến đổi nào và giữ cho trạng thái có thể thử lại được cho đến khi hết hạn. Một hành động `decline` tường minh là trạng thái kết thúc, do đó bài học sẽ tiêu thụ nonce mà không thực hiện xóa bất kỳ thứ gì.

```figure
t3-roots-boundary
```

## Xây dựng

Tệp `code/main.py` minh họa một công cụ `notes_delete` hiện đại:

- `tools/list` trả về bộ mô tả tất định, có thể lưu đệm với lược đồ bắt buộc gồm không gian làm việc và tiêu đề.
- Phạm vi là một đối số `workspaceUri` rõ ràng.
- Cấu hình máy chủ ủy quyền không gian làm việc đó cho chủ thể bài học.
- Chuẩn hóa URI từ chối các nhầm lẫn về tiền tố và kỹ thuật vượt thư mục được mã hóa.
- Mọi thao tác xóa mang tính phá hủy đều yêu cầu gợi mở phản hồi ở chế độ biểu mẫu.
- Yêu cầu gợi mở đi bên trong `resultType: "input_required"`.
- `requestState` đã được ký ràng buộc chính xác danh sách ứng viên và các đối số ban đầu.
- Một kho lưu trữ chống phát lại được tiêm vào sẽ từ chối cùng một trạng thái đã chấp nhận hoặc từ chối trên mọi phiên bản máy chủ.
- Lần thử lại sử dụng một id yêu cầu mới và trả về `resultType: "complete"`.

Kho dữ liệu được lưu trong bộ nhớ để các hành vi giao thức dễ dàng kiểm tra. Các quy tắc bảo mật vẫn giữ nguyên vẹn khi sử dụng với một cơ sở dữ liệu thực sự.

## Sử dụng

Từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

Các điểm kiểm tra kỳ vọng:

- Khám phá quảng bá công cụ mà không có Roots.
- Khám phá công cụ trả về `notes_delete` với `resultType`, danh tính máy chủ và gợi ý bộ đệm.
- Yêu cầu có id `1` trả về biểu mẫu trong `inputRequests.delete_choice`.
- Yêu cầu có id `2` phản hồi lại trạng thái đã ký và hoàn thành thao tác xóa.
- Một đường dẫn tiền tố giả mạo và một đường dẫn vượt thư mục mã hóa đều thất bại ở bước kiểm tra tính bao bọc.
- Một tiêu đề bị thay đổi không thể tái sử dụng trạng thái xác nhận ban đầu.
- Một hành động từ chối giữ nguyên ghi chú không bị thay đổi.
- Hai đối tượng máy chủ chia sẻ trạng thái ghi chú và trạng thái phát lại không thể cùng thực thi một xác nhận duy nhất.
- Khai báo biểu mẫu rỗng và tường minh đều hoạt động, trong khi chỉ hỗ trợ URL sẽ trả về yêu cầu biểu mẫu chính xác theo mã `-32021`.
- Lỗi phiên bản không được hỗ trợ sử dụng đúng cấu trúc dữ liệu của `-32022`.
- Một thông báo không có id không tạo ra phản hồi JSON-RPC nào.

## Đóng gói / Triển khai

Tệp `outputs/skill-elicitation-form-designer.md` thiết kế phạm vi tường minh, các bước kiểm tra ủy quyền, biểu mẫu MRTR, các nhánh phản hồi và ràng buộc trạng thái. Nó từ chối coi Roots đã bị ngừng khuyến nghị như một hộp cát hoặc thu thập bí mật thông qua chế độ biểu mẫu.

## Bài tập

1. Thay thế kho lưu trữ chống phát lại trong bộ nhớ bằng SQLite. Sử dụng một giao dịch duy nhất để xác nhận nonce và xóa ghi chú, sau đó chứng minh hai tiến trình không thể cùng commit thành công.
2. Thêm cơ chế đàm phán khả năng `url` và một quy trình thiết lập ngoài băng. Giữ thông tin xác thực của bên thứ ba nằm ngoài `inputResponses`.
3. Thay thế ánh xạ ghi chú trong bộ nhớ bằng cơ sở dữ liệu SQLite tạm thời. Kiểm tra lại việc ủy quyền và bao bọc bên trong giao dịch biến đổi dữ liệu.
4. Thêm chính sách liên kết tượng trưng (symbolic link) cho bản triển khai hệ thống tệp thực tế. Giải thích tại sao chỉ kiểm tra tính bao bọc từ vựng của URI không thể ngăn chặn việc thoát qua symlink.
5. Thiết kế một bộ điều hợp 2025-11-25 ánh xạ đầu ra trình xử lý MRTR hiện đại sang gợi mở phản hồi do máy chủ khởi tạo kế thừa. Giữ nó cô lập hoàn toàn khỏi trình xử lý hiện tại.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa trong đặc tả 2026-07-28 |
|---|---|
| Roots | Các gợi ý không gian làm việc mang tính thông tin đã ngừng khuyến nghị, không phải ủy quyền hay hộp cát |
| Phạm vi tường minh (Explicit scope) | Thẻ định danh không gian làm việc, thư mục hoặc tài nguyên hiển thị rõ ràng trong các đối số yêu cầu |
| Tính bao bọc (Containment) | Kiểm tra thành phần đường dẫn đã chuẩn hóa nhằm giữ mục tiêu luôn nằm bên trong ranh giới cho phép |
| Gợi mở phản hồi (Elicitation) | Tính năng của máy khách dùng để thu thập đầu vào của người dùng trong khi thực hiện một thao tác MCP |
| Chế độ biểu mẫu (Form mode) | Đầu vào có cấu trúc của người dùng trong băng sử dụng một lược đồ phẳng bị giới hạn |
| Chế độ URL (URL mode) | Tương tác ngoài băng dành cho các quy trình nhạy cảm hoặc quy trình với dịch vụ bên ngoài |
| MRTR | Kết quả yêu cầu đầu vào không lưu trạng thái theo sau bởi một lần thử lại mới |
| `requestState` | Trạng thái ẩn được lặp lại chính xác và được kiểm tra tính toàn vẹn bởi máy chủ |
| Từ chối (Decline) | Sự từ chối rõ ràng từ phía người dùng |
| Hủy bỏ (Cancel) | Hành động bỏ qua hoặc tương tác không hoàn thành mà không có sự phê duyệt |

## Tính tương thích Kế thừa

Đối với một máy chủ ngang hàng bị ghim ở phiên bản 2025-11-25, `roots/list`, `notifications/roots/list_changed`, và lệnh gọi `elicitation/create` do máy chủ khởi tạo trực tiếp vẫn có thể tồn tại. Hãy dán nhãn rõ ràng cho bộ điều hợp đó là di sản cũ (legacy). Tuyệt đối không cho phép danh sách Root cũ vượt qua cơ chế ủy quyền của máy chủ, và không mang các giả định về phiên giao thức vào trình xử lý hiện đại.

## Đọc thêm

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [Khám phá Máy chủ MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
