# Tài nguyên và Lời nhắc trong MCP: Ngữ cảnh Định địa chỉ cho các Máy chủ Không lưu trạng thái

> Công cụ thực hiện các thao tác. Tài nguyên công bố nội dung có thể định địa chỉ. Lời nhắc đóng gói các mẫu thông điệp do người dùng lựa chọn. Một máy chủ MCP tốt luôn giữ các hợp đồng đó tách bạch và có thể dự đoán được.

**Type:** Xây dựng (Build)
**Languages:** Python
**Prerequisites:** Phase 13, Bài học 07 (Xây dựng Máy chủ MCP), Phase 13, Bài học 09 (Các Phương thức Truyền tải MCP)
**Time:** ~60 phút

## Mục tiêu học tập

- Lựa chọn chuẩn xác giữa công cụ (tools), tài nguyên (resources), và lời nhắc (prompts) dựa trên chủ đích của bên tiêu thụ.
- Công bố bề mặt tài nguyên và lời nhắc thông qua phương thức bắt buộc `server/discover`.
- Xây dựng kết quả `resources/list` và `prompts/list` có tính tất định.
- Áp dụng `ttlMs` và `cacheScope` mà không làm rò rỉ dữ liệu riêng tư của người dùng.
- Trả về lỗi JSON-RPC `-32602` khi gặp một URI tài nguyên không hợp lệ hoặc không xác định.
- Mở luồng phản hồi POST `subscriptions/listen` và đối chiếu mọi sự kiện theo ID đăng ký (subscription ID).
- Xử lý nội dung tài nguyên và bản mẫu lời nhắc như là đầu ra máy chủ không đáng tin cậy.

## Bắt đầu từ Người tiêu thụ

Cách dễ nhất để dùng sai MCP là bắt đầu ngay từ mã triển khai. Một truy vấn cơ sở dữ liệu bị biến thành một công cụ chỉ vì các hàm vốn quen thuộc. Một quy trình làm việc có thể tái sử dụng bị biến thành một tài nguyên chỉ vì nó được lưu trong một tệp tin. Một lời nhắc bị biến thành một chính sách ẩn giấu chỉ vì ứng dụng chủ có thể tiêm nó vào ngữ cảnh.

Hãy bắt đầu từ việc ai là người lựa chọn và họ kỳ vọng điều gì.

| Thành phần nguyên thủy | Chủ đích chính | Bên nắm quyền lựa chọn | Kết quả điển hình |
|---|---|---|---|
| Công cụ (Tool) | Thực hiện một thao tác | Mô hình hoặc ứng dụng | Kết quả hành động có cấu trúc |
| Tài nguyên (Resource) | Đọc nội dung tại một URI | Ứng dụng chủ, ứng dụng, hoặc người dùng | Nội dung văn bản hoặc nhị phân |
| Lời nhắc (Prompt) | Bắt đầu một quy trình thông điệp tái sử dụng | Người dùng thông qua giao diện ứng dụng chủ | Một hoặc nhiều thông điệp lời nhắc |

Một ghi chú tại `notes://note-1` là một tài nguyên vì nó là nội dung có thể định địa chỉ. `delete_note` là một công cụ vì nó làm thay đổi trạng thái. `review_note` là một lời nhắc vì người dùng chủ động lựa chọn một quy trình đánh giá đã được chuẩn bị sẵn.

Đừng công bố một thao tác dưới cả ba dạng thức chỉ nhằm mục đích làm cho có vẻ đầy đủ. Mỗi bề mặt bổ sung đều đòi hỏi quy trình khám phá, ủy quyền, lưu bộ đệm, xử lý lỗi, kiểm thử và tài liệu hướng dẫn riêng biệt.

## Phong bì Không lưu trạng thái 2026-07-28

Bài học này nhắm tới bản sửa đổi giao thức MCP `2026-07-28`. Hoàn toàn không có quy trình bắt tay khởi tạo hay phiên giao thức nào trong hồ sơ này. Mọi yêu cầu đều mang phiên bản giao thức và khả năng của máy khách trong các khóa `_meta` dành riêng.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Một máy chủ bắt buộc phải triển khai phương thức `server/discover`. Kết quả của nó quảng bá các phiên bản được hỗ trợ, khả năng về tài nguyên và lời nhắc, danh tính bản triển khai, và các gợi ý bộ nhớ đệm. Máy khách có thể gọi trực tiếp một phương thức khác, nhưng quy trình khám phá cung cấp cho nó một ảnh chụp nhanh (snapshot) ổn định trước khi dựng giao diện người dùng.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

Một kết quả bình thường khai báo `"resultType": "complete"`. Trường `_meta` của phản hồi định danh bản triển khai phục vụ bằng `io.modelcontextprotocol/serverInfo`. Thông tin này rất hữu ích cho công tác chẩn đoán. Nó không phải là một danh tính dùng cho xác thực bảo mật. Một yêu cầu mang bản sửa đổi không được hỗ trợ sẽ trả về mã lỗi `-32022` cùng với cả bản sửa đổi được yêu cầu và các bản sửa đổi mà máy chủ hỗ trợ.

Hợp đồng không lưu trạng thái làm thay đổi tư duy thiết kế của bạn. Một danh sách không thể phụ thuộc vào một lệnh gọi trước đó trên cùng một kết nối. Việc ủy quyền có thể làm thay đổi tập hợp hiển thị vì thông tin xác thực là dữ liệu đầu vào của yêu cầu, nhưng lịch sử kết nối tuyệt đối không được phép can thiệp.

## Tài nguyên là các Hợp đồng URI Ổn định

Tài nguyên là nội dung được định danh bởi một URI. Hãy thiết kế URI trước khi viết trình xử lý.

Các đặc tính tốt của URI:

- Đủ ổn định để đánh dấu trang (bookmark) hoặc truyền giữa các yêu cầu.
- Được phân chia không gian tên theo miền của máy chủ.
- Độc lập với ID tiến trình hoặc kết nối cụ thể.
- Được xác thực tính hợp lệ trước khi truy cập bộ lưu trữ.
- Được kiểm tra ủy quyền trên mỗi lần đọc.

`notes://note-1` tốt hơn nhiều so với `note-1` vì không gian tên của nó rất rõ ràng. Một máy chủ tệp có thể sử dụng các URI `file://`, nhưng nó vẫn phải kiểm tra các ranh giới thư mục đã được cấu hình sau khi đã phân giải các liên kết tượng trưng (symlinks) và các đoạn đường dẫn tương đối.

`resources/list` trả về các tài nguyên hiện đang hiển thị với người gọi. Hãy sắp xếp theo một khóa ổn định chẳng hạn như chính URI. Thứ tự tất định ngăn chặn hiện tượng trượt bộ đệm không đáng có, các ảnh chụp trạng thái thay đổi thất thường, và giao diện ứng dụng chủ bị nhảy giật giữa các lần làm mới.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` trả về một hoặc nhiều mục nội dung. Một URI không tồn tại không phải là một lần đọc thành công trả về kết quả rỗng. Đặc tả Tài nguyên hiện tại quy định các URI tài nguyên không hợp lệ hoặc không xác định thuộc về lỗi tham số không hợp lệ của JSON-RPC, mã lỗi `-32602`.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

Sự phân biệt rõ ràng đó giúp máy khách tách biệt trường hợp tài nguyên không tồn tại với một tài liệu hợp lệ nhưng có nội dung rỗng. Nó cũng ngăn chặn việc vô tình quay lui (fallback) sang một phép tra cứu rộng hơn.

### Bản mẫu tài nguyên

Một bản mẫu tài nguyên (resource template) mô tả một họ các URI được tham số hóa. Hãy sử dụng bản mẫu khi việc liệt kê từng mục cụ thể quá tốn kém hoặc không có giới hạn. Ví dụ, `notes://projects/{project}/decisions/{decision}` cho máy khách biết cách tạo một địa chỉ hợp lệ mà không cần trả về toàn bộ các quyết định.

Bản mẫu không làm giảm nhẹ yêu cầu xác thực. Hãy phân tích các biến, áp dụng kiểm tra ủy quyền, cưỡng chế độ dài và giới hạn ký tự, và xây dựng các truy vấn lưu trữ với các tham số định kiểu. Tuyệt đối không bao giờ nối trực tiếp phần đuôi URI tùy ý vào đường dẫn hệ thống tệp hoặc câu lệnh cơ sở dữ liệu.

### Nội dung không phải là chỉ thị đáng tin cậy

Văn bản tài nguyên có thể chứa các cuộc tấn công tiêm lời nhắc (prompt injection), bí mật nhạy cảm, lệnh gây hiểu lầm hoặc mã đánh dấu sai định dạng. Ứng dụng chủ cần bảo toàn nguồn gốc xuất xứ (provenance) và xử lý nội dung tài nguyên thuần túy như dữ liệu. Máy chủ nên giới hạn kích thước nội dung, trả về kiểu MIME chính xác, che khuất (redact) các trường mà người gọi không có quyền truy cập, và tránh trả về các bản ghi không liên quan.

## Lời nhắc là các Bản mẫu do Người dùng Kiểm soát

Các lời nhắc trong MCP được thiết kế cho sự lựa chọn tường minh từ phía người dùng. Ứng dụng chủ có thể hiển thị chúng dưới dạng các lệnh gạch chéo (slash commands), các mục trong menu, hoặc các nút quy trình làm việc. Giao thức không bắt buộc một kiểu giao diện người dùng cụ thể.

`prompts/list` cần mang tính tất định đối với cùng một mức ủy quyền yêu cầu. Mỗi lời nhắc cần một tên ổn định, phần mô tả hữu ích và các khai báo đối số để ứng dụng chủ có thể thu thập đầu vào của người dùng trước khi gọi `prompts/get`.

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` phân giải các đối số thành các thông điệp. Nó không thay thế các chỉ thị hệ thống (system instructions) của ứng dụng chủ. Ứng dụng chủ quyết định cách thức các thông điệp trả về đi vào ngữ cảnh mô hình và luôn giữ chính sách đáng tin cậy của riêng mình ở mức ưu tiên cao hơn.

Xác thực các đối số của lời nhắc ngay tại ranh giới máy chủ. Một URI trong lời nhắc cần phải vượt qua cùng một bước kiểm tra ủy quyền như một lệnh đọc tài nguyên trực tiếp. Đừng biến lời nhắc thành một kênh phụ (side channel) để vượt mặt kiểm soát truy cập tài nguyên.

## Gợi ý Bộ nhớ đệm là một Phần của Tính Đúng đắn

`ttlMs` cho máy khách biết một kết quả có thể được tái sử dụng trong bao lâu. `cacheScope` mô tả ai có thể chia sẻ giá trị đã lưu trong bộ đệm đó.

| Phạm vi (Scope) | Ý nghĩa | Trường hợp sử dụng điển hình |
|---|---|---|
| `public` | Có thể tái sử dụng cho nhiều người dùng khi quyền hạn cho phép | Danh mục lời nhắc công khai |
| `private` | Ràng buộc với người dùng hoặc ngữ cảnh thông tin xác thực gửi yêu cầu | Nội dung ghi chú thuộc sở hữu người dùng |

Hãy chọn TTL dựa trên tần suất thay đổi của dữ liệu và mức độ thiệt hại nếu dữ liệu bị lỗi thời. Năm phút có thể phù hợp cho một danh mục lời nhắc công khai. Một lệnh đọc ghi chú riêng tư có thể sử dụng một phút.

Đặc tả MCP chỉ định nghĩa hai giá trị `public` và `private` cho `cacheScope`. Đối với kết quả chứa thông tin bí mật hoặc thay đổi nhanh chóng, hãy trả về `cacheScope: "private"` với `ttlMs: 0`, sau đó áp dụng bất kỳ quy tắc cấm lưu đệm (no-store) nghiêm ngặt hơn nào trong chính sách bộ nhớ đệm của ứng dụng chủ. Bản thân giá trị `no-store` không phải là một giá trị `cacheScope` của MCP.

Gợi ý bộ nhớ đệm tuyệt đối không thay thế cho việc ủy quyền. Khóa bộ đệm phải bao gồm mọi chiều không gian của yêu cầu làm thay đổi phạm vi hiển thị, bao gồm đối tượng thuê (tenant), người dùng, phạm vi quyền (scope), ngôn ngữ bản địa hóa (locale), và con trỏ phân trang (pagination cursor). Nếu bộ nhớ đệm dùng chung không thể biểu diễn các chiều không gian đó một cách an toàn, hãy sử dụng `private` với TTL bằng 0 cùng chính sách no-store ở cấp ứng dụng chủ.

## Cơ chế Đăng ký luồng Sử dụng Luồng Phản hồi do Máy khách Mở

Mẫu đăng ký luồng hiện đại thay thế cho lệnh RPC `resources/subscribe` trước đây và điểm cuối sự kiện HTTP GET cũ.

Máy khách gửi `subscriptions/listen` như một yêu cầu JSON-RPC thông thường. Qua phương thức Streamable HTTP, đây là một lệnh POST có luồng phản hồi được giữ mở dưới dạng luồng SSE. Đối tượng `notifications` đóng vai trò là một danh sách cho phép (allowlist). Máy chủ tuyệt đối không được gửi các loại thông báo không được yêu cầu.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

ID của yêu cầu chính là ID đăng ký (subscription ID). Trước bất kỳ sự kiện nào được yêu cầu, máy chủ gửi `notifications/subscriptions/acknowledged`. Bộ lọc của thông báo này chỉ chứa tập con mà máy chủ đã chấp nhận.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Mọi sự kiện diễn ra sau đó trên luồng này đều mang cùng một siêu dữ liệu.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

Thông báo cho biết tài nguyên đã thay đổi. Máy khách sẽ đọc lại tài nguyên đó thông qua `resources/read`, phụ thuộc vào quyền ủy quyền hiện thời. Máy khách không được mặc định rằng sự kiện đã chứa tài liệu mới.

Nhiều gói đăng ký có thể chia sẻ chung một kênh stdio. ID đăng ký cho phép máy khách phân kênh (demultiplex) các sự kiện. Qua HTTP, việc đóng luồng phản hồi đồng nghĩa với việc hủy đăng ký. Một máy chủ kết thúc luồng một cách duyên dáng (gracefully) sẽ trả về phản hồi cuối cùng với `resultType: "complete"` tương quan với yêu cầu ban đầu.

Không sử dụng luồng đăng ký như một phiên giao thức. Một lệnh đọc đến sau vẫn là một yêu cầu độc lập hoàn chỉnh có thể chuyển tới bất kỳ phiên bản máy chủ khỏe mạnh nào.

```figure
t3-primitive-sort
```

## Phòng thí nghiệm Tương tác

Sử dụng hình vẽ để phân loại năm khả năng từ một hệ thống theo dõi dự án: chi tiết vấn đề (issue details), tạo vấn đề (create issue), mẫu đánh giá sprint (sprint review template), chính sách dự án (project policy), và đóng vấn đề (close issue). Sau đó quyết định danh sách nào có thể lưu đệm công khai, lệnh đọc nào phải giữ ở chế độ riêng tư, và tài nguyên nào xứng đáng nhận thông báo cập nhật.

Đối với mỗi phân loại, hãy chỉ rõ bên đưa ra lựa chọn. Nếu mô hình thực hiện một hành động, hãy dùng công cụ. Nếu ứng dụng chủ đọc nội dung được định địa chỉ bằng URI, hãy dùng tài nguyên. Nếu người dùng bắt đầu một quy trình thông điệp đã chuẩn bị trước, hãy dùng lời nhắc.

## Phòng thí nghiệm Thực hành

Chạy bộ mô phỏng từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Kiểm tra bản ghi hoạt động (transcript) theo thứ tự sau:

1. Xác nhận `server/discover` quảng bá đúng bản sửa đổi hiện tại và cả hai nhóm khả năng.
2. Xác nhận cả hai kết quả danh sách đều đã được sắp xếp và sử dụng `resultType: "complete"`.
3. Xác nhận các kết quả danh sách và đọc đều mang gợi ý bộ nhớ đệm có chủ đích.
4. Đổi URI đọc thành `notes://missing` và quan sát mã lỗi `-32602`.
5. Xác nhận thông báo xác nhận đăng ký xuất hiện trước sự kiện tài nguyên.
6. Xác nhận cả sự kiện và việc đóng duyên dáng đều mang ID đăng ký `5`.

Mô hình Python không mở một kết nối HTTP thực sự. Nó mô tả các thông điệp mà một SDK phải đặt lên luồng phản hồi trong phạm vi yêu cầu. Hãy sử dụng SDK chính thức cho việc đóng khung thông điệp (framing) và truyền tải trên môi trường sản xuất.

## Sản phẩm bàn giao

Tệp `outputs/skill-primitive-splitter.md` là một bản đánh giá thiết kế có thể tái sử dụng cho việc lựa chọn thành phần nguyên thủy MCP. Nó kiểm tra tính khám phá tất định, phạm vi bộ đệm, hành vi URI không hợp lệ, và các bộ lọc đăng ký hiện đại.

Bài học cũng cung cấp `assets/primitive-split.svg`, một phiên bản tĩnh của ranh giới thành phần nguyên thủy và cơ chế đăng ký để nghiên cứu ngoại tuyến.

## Xác minh

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Kết quả kỳ vọng: chương trình chính in ra một bản ghi JSON và lệnh kiểm thử báo cáo ít nhất mười hai bài kiểm thử đều vượt qua.

## Kết nối Đồ án Tổng hợp

Áp dụng hợp đồng này khi máy chủ đồ án tổng hợp của bạn cung cấp tri thức có thể định địa chỉ bên cạnh các hành động. Hãy bao gồm một ảnh chụp danh mục tất định, một lệnh đọc tài nguyên đã được ủy quyền, một phép phân giải lời nhắc, một trường hợp URI không hợp lệ và một bản ghi đăng ký luồng.

Bằng chứng của bạn cần chỉ ra rằng không có danh sách nào phụ thuộc vào lịch sử kết nối và một sự kiện đăng ký không bao giờ tự động cấp quyền truy cập vào tài nguyên cơ bản.

## Bài tập

1. Thêm một bản mẫu tài nguyên `notes://projects/{project}/notes/{id}` và xác thực cả hai biến.
2. Thêm phân trang vào `resources/list` trong khi vẫn bảo toàn thứ tự tất định.
3. Thay đổi một tài nguyên thành `cacheScope: "private"` với `ttlMs: 0`, thêm chính sách no-store ở cấp ứng dụng chủ, và giải thích mối đe dọa làm cơ sở cho cả hai biện pháp kiểm soát này.
4. Thêm gói đăng ký thay đổi danh sách lời nhắc và chứng minh không có sự kiện nào được gửi đi khi bộ lọc bỏ qua `promptsListChanged`.
5. Tạo hai gói đăng ký đồng thời và chứng minh mỗi sự kiện đều mang đúng ID yêu cầu tương ứng.
6. Thêm chủ thể ủy quyền vào trình xử lý đọc và chứng minh rằng mục bộ đệm không thể dùng chéo giữa các chủ thể khác nhau.

## Thuật ngữ then chốt

- **Tài nguyên (Resource):** Nội dung được định địa chỉ bằng URI do một máy chủ MCP công bố.
- **Lời nhắc (Prompt):** Bản mẫu thông điệp do người dùng kiểm soát được một máy chủ MCP công bố.
- **Danh sách tất định (Deterministic list):** Kết quả khám phá với các thành viên và thứ tự ổn định cho cùng các đầu vào yêu cầu.
- **`ttlMs`:** Thời gian duy trì độ tươi mới của bộ nhớ đệm tính bằng mili-giây.
- **`cacheScope`:** Ranh giới chia sẻ của một kết quả được lưu trong bộ nhớ đệm.
- **`subscriptions/listen`:** Một yêu cầu tồn tại lâu dài có luồng phản hồi phân phối các thông báo được lọc rõ ràng.
- **ID đăng ký (Subscription ID):** ID của yêu cầu lắng nghe ban đầu, được lặp lại trong siêu dữ liệu của thông báo.
- **Tham số không hợp lệ (Invalid parameters):** Mã lỗi JSON-RPC `-32602`, sử dụng cho một URI tài nguyên không hợp lệ hoặc không xác định.
- **Phiên bản giao thức không được hỗ trợ (Unsupported protocol version):** Mã lỗi JSON-RPC `-32022`, bao gồm các bản sửa đổi `supported` và `requested`.
- **`server/discover`:** Phương thức máy chủ bắt buộc trả về các bản sửa đổi được hỗ trợ, năng lực, danh tính và gợi ý bộ đệm tùy chọn.

## Đọc thêm

- [Tài nguyên MCP 2026-07-28 (MCP 2026-07-28 Resources)](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [Lời nhắc MCP 2026-07-28 (MCP 2026-07-28 Prompts)](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [Cơ chế Đăng ký luồng MCP 2026-07-28 (MCP 2026-07-28 Subscriptions)](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [Cơ chế Lưu đệm MCP 2026-07-28 (MCP 2026-07-28 Caching)](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
