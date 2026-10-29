# Ứng dụng MCP trên nền giao thức phi trạng thái

> Một kết quả tương tác trực quan về bản chất vẫn là sự trao đổi công cụ và tài nguyên MCP. Giao thức lõi 2026-07-28 biến sự trao đổi đó thành một thể tự chứa (self-contained), trong khi phần mở rộng Apps (Apps extension) bổ sung thêm bề mặt trình duyệt trong môi trường cô lập (sandboxed browser surface).

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 13 · 07 (máy chủ MCP), Phase 13 · 10 (tài nguyên)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Công bố các ứng dụng MCP (MCP Apps) thông qua `server/discover` và các năng lực phần mở rộng theo từng yêu cầu (per-request extension capabilities).
- Khai báo một tài nguyên `ui://` trên một công cụ trước khi công cụ đó được gọi.
- Trả về kết quả công cụ và tài nguyên hoàn chỉnh trên đường truyền phi trạng thái chuẩn 2026-07-28.
- Tách bạch thông điệp cầu nối `ui/initialize` của Apps khỏi quy trình bắt tay giao thức lõi MCP đã bị bãi bỏ.
- Áp dụng cơ chế xác thực nguồn gốc (origin validation), cách ly môi trường (sandboxing), CSP, và các quyền hạn theo nguyên tắc đặc quyền tối thiểu (least-privilege permissions).

## Vấn đề đặt ra

Một kết quả thuần văn bản có thể mô tả một dòng thời gian (timeline). Nhưng nó không thể mang lại cho người dùng một dòng thời gian mà họ có thể lọc, kiểm tra chi tiết, hay tương tác trực tiếp.

MCP Apps giải quyết bài toán biểu diễn này bằng một phần mở rộng tùy chọn. Một định nghĩa công cụ trỏ tới một tài nguyên `ui://`. Phía máy chủ lưu trữ (host) có thể nạp và duyệt xét tài nguyên đó trước khi công cụ thực thi, hiển thị nó bên trong một khung nội nhúng (iframe) được cô lập an toàn, và điều phối toàn bộ các hành động của ứng dụng qua một cầu nối JSON-RPC (JSON-RPC bridge).

Giao thức lõi đã thay đổi trong bản chuẩn 2026-07-28. Tuyệt đối không bao bọc một App trong vòng đời kết nối kiểu cũ:

- Không còn yêu cầu `initialize` hay thông báo `notifications/initialized` ở tầng lõi.
- Không còn tiêu đề `Mcp-Session-Id`.
- Mọi yêu cầu đều mang phiên bản giao thức và các năng lực client bên trong `params._meta`.
- Máy chủ triển khai `server/discover` để client có thể kiểm tra phiên bản, các năng lực cốt lõi, và các phần mở rộng.
- Mọi kết quả thành công đều có một bộ phân biệt `resultType` (resultType discriminator).
- Streamable HTTP sử dụng đúng một yêu cầu POST cho mỗi chu kỳ. Các điểm cuối GET và DELETE kiểu cũ trả về mã lỗi 405.

Cầu nối Apps vẫn duy trì một phương thức có tên `ui/initialize`. Phương thức này thuộc về ngữ tộc postMessage của iframe. Nó hoàn toàn không tái tạo một phiên MCP ở tầng lõi.

## Khái niệm cốt lõi

### Hai giao thức, một tính năng

Phân định rạch ròi các tầng kiến trúc:

1. Giao thức lõi MCP đảm nhiệm `server/discover`, `tools/list`, `tools/call`, `resources/list`, và `resources/read`.
2. Phần mở rộng MCP Apps khai báo giao diện người dùng và định nghĩa cầu nối giữa iframe và host.
3. Các quy tắc hộp cát trình duyệt (browser sandbox rules) giới hạn phạm vi mà giao diện có thể tiếp cận.

Mã định danh của phần mở rộng là `io.modelcontextprotocol/ui`. Cả hai bên đối tác đều phải chủ động chọn tham gia (opt in). Client gửi thông tin hỗ trợ phần mở rộng bên trong đối tượng năng lực ở mỗi yêu cầu:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

Trường `clientInfo` được khuyến nghị cho mục đích chẩn đoán. Đây là dữ liệu tự khai báo, không phải là định danh dùng để ủy quyền.

### Khám phá trước khi kết xuất

Kết quả khám phá của máy chủ công bố phần mở rộng:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

Máy chủ bắt buộc phải hỗ trợ phương thức khám phá. Client không bắt buộc phải gọi khám phá trước mọi hành động, bởi vì mỗi hành động đều tự mang theo các năng lực của riêng nó.

### Khai báo giao diện người dùng trên định nghĩa công cụ

Bản hợp đồng Apps hiện đại gắn kết giao diện người dùng với công cụ ngay trong `tools/list`:

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

Đây là siêu dữ liệu được chủ ý cung cấp trước khi gọi công cụ (pre-call metadata). Host có thể nạp trước (preload), lưu bộ nhớ đệm, và duyệt xét bảo mật mã HTML trước khi kết quả yêu cầu hiển thị nó. Các khóa siêu dữ liệu dạng phẳng kiểu cũ có thể được mã tương thích chấp nhận, nhưng các máy chủ mới nên phát ra định dạng lồng nhau `_meta.ui.resourceUri`.

Lệnh `tools/list` có thể lưu đệm được trong giao thức lõi hiện tại. Hãy đính kèm thứ tự tất định, `ttlMs`, và `cacheScope`. Sử dụng giá trị `private` khi danh sách công cụ hiển thị có sự khác biệt tùy theo người dùng hoặc mã xác thực (token).

### Trả về dữ liệu, sau đó để host liên kết khung nhìn

Lệnh gọi công cụ trả về nội dung thông thường kèm theo dữ liệu có cấu trúc:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

Host đã biết khung nhìn nào gắn liền với công cụ này. Hãy tránh việc tự tạo thêm khối nội dung mới chỉ nhằm mục đích lặp lại URI giao diện.

### Phục vụ ứng dụng dưới dạng một tài nguyên

Máy chủ công bố năng lực `resources` trong phát hiện dịch vụ, do đó nó cũng phải triển khai thao tác bắt buộc `resources/list`. Bản ghi danh sách tất định của nó bao gồm URI chuẩn tắc, tên ổn định, phần mô tả, và kiểu MIME. Kết quả danh sách này bao gồm `resultType`, siêu dữ liệu định danh máy chủ, `ttlMs`, và `cacheScope`, tương tự như danh sách công cụ tất định.

Host gửi yêu cầu `resources/read`. Trên tầng giao vận Streamable HTTP, yêu cầu có dạng:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Các giá trị trong tiêu đề HTTP và phần thân JSON-RPC phải hoàn toàn trùng khớp. Bất kỳ sự sai lệch nào cũng dẫn đến lỗi giao thức `-32020`.

Kết quả trả về chứa tài nguyên HTML và các chỉ dẫn bộ nhớ đệm:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### Lưu bộ nhớ đệm tài nguyên giao diện người dùng như nội dung thực thi

Một tài nguyên App không thể đối xử tương đương như một đoạn văn bản thông thường. Mục lưu đệm của nó có thể thực thi mã cầu nối, kết xuất dữ liệu công cụ, và yêu cầu các hành động do host làm trung gian. Hãy dùng khóa lưu đệm gồm: URI chuẩn tắc `ui://`, định danh và phiên bản máy chủ được chấp nhận, mã băm nội dung tài nguyên (digest), và ngữ cảnh ủy quyền khi `cacheScope` là `private`. Tuyệt đối không tái sử dụng một tài nguyên App riêng tư (private) giữa các chủ thể khác nhau vì mã HTML hoặc siêu dữ liệu chính sách có thể khác biệt ngay cả khi URI hoàn toàn trùng khớp.

Hãy vô hiệu hóa mục lưu đệm khi `ttlMs` hết hạn, liên kết `_meta.ui.resourceUri` của công cụ thay đổi, phiên bản máy chủ hoặc mã định danh mô tả được chấp nhận thay đổi, hoặc khi có một thông báo xác nhận thay đổi tài nguyên nhắm tới URI đó. Hãy nạp lại và áp dụng lại quy trình duyệt xét CSP và quyền hạn trước khi gắn lại khung nhìn (remount). Một iframe cũ không được phép duy trì quyền hạn rộng hơn chỉ vì phiên bản tài nguyên mới chưa kịp tải xong.

### Từ chối sự mơ hồ của đường truyền trước khi áp dụng chính sách tính năng

Quy trình xác thực cần tuân theo thứ tự chặt chẽ: Đầu tiên xác thực hình thái JSON-RPC, yêu cầu siêu dữ liệu giao thức dạng chuỗi và bản đồ năng lực client dạng đối tượng. Kế tiếp, so sánh các tiêu đề định tuyến với phần thân thông điệp. Chỉ sau đó mới quyết định xem phiên bản giao thức tương ứng có được hỗ trợ hay không. Thứ tự này ngăn chặn việc proxy và máy chủ diễn giải hai yêu cầu khác nhau.

| Điều kiện | HTTP | Lỗi JSON-RPC |
|---|---|---|
| Tiêu đề và phần thân không khớp nhau về phiên bản, phương thức hoặc tên | 400 | `-32020` |
| Tiêu đề và phần thân thống nhất về một phiên bản không được hỗ trợ | 400 | `-32022`, với trường `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` thiếu năng lực phần mở rộng Apps | 400 | `-32021`, với `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Phương thức không xác định | 404 | `-32601` |

Một thông báo JSON-RPC không có trường `id`, do đó máy chủ không bao giờ phát ra phản hồi JSON-RPC cho nó. Một thông báo HTTP được chấp nhận sẽ trả về mã 202 với phần thân rỗng. Một lỗi có thể làm thay đổi mã trạng thái HTTP, nhưng vẫn tuyệt đối không được tạo ra phần thân lỗi JSON-RPC cho một thông báo.

### Môi trường cô lập là một ranh giới, không phải là phán quyết tin cậy

Host kiểm soát iframe. Bản thân App không thể trực tiếp đọc cookie, bộ nhớ cục bộ (local storage), hay DOM của trang host. Mọi thao tác có đặc quyền đều phải đi qua cầu nối.

Hãy áp dụng các cấu hình mặc định sau:

- Để trống tất cả các danh sách tên miền CSP, sau đó chỉ bổ sung các nguồn gốc (origins) mà App thực sự cần. Sử dụng `connectDomains` cho fetch, XHR, và WebSocket; sử dụng `resourceDomains` cho scripts, styles, hình ảnh, và phông chữ.
- Đóng gói mã nguồn và dữ liệu khi có thể triển khai thực tế.
- Không yêu cầu quyền camera, microphone, hay định vị vị trí trừ khi một tính năng hiển thị trực quan cần đến nó.
- Ghim chặt `postMessage` vào nguồn gốc chính xác của đối tác và từ chối mọi sự kiện đến từ bất kỳ nguồn gốc nào khác.
- Coi các đối số công cụ, kết quả công cụ, văn bản tài nguyên, và các thông điệp cầu nối đều là dữ liệu đầu vào không đáng tin cậy.
- Giữ quyền chấp thuận của người dùng tại host. Iframe không thể tự mình phê duyệt hành động hệ trọng của chính nó.

Không sao chép một thuộc tính `sandbox` cố định từ các bài hướng dẫn vào mọi host. Host phải chủ động lựa chọn các cờ dựa trên mô hình nguồn gốc của App và thiết kế cách ly riêng của mình.

Một tên miền được cho phép vẫn là một lộ trình trích xuất dữ liệu tiềm tàng (exfiltration path). Cấu hình `connectDomains: ["https://api.example.com"]` đồng nghĩa với việc bất kỳ đoạn mã nào thực thi bên trong App đều có thể gửi dữ liệu được phép tới đó. Việc khớp chính xác nguồn gốc ngăn chặn sự nhầm lẫn đích đến, nhưng không quyết định được liệu tải trọng dữ liệu có phù hợp hay không. Hãy luôn để trống quyền kết nối theo mặc định, tránh đưa các mã token người mang (bearer tokens) vào iframe, làm trung gian cho các thao tác hẹp thông qua host khi khả thi, giới hạn kích thước yêu cầu và phản hồi, đồng thời kiểm toán hành động người dùng nào đã kích hoạt mỗi yêu cầu ra bên ngoài. Hãy xử lý `resourceDomains` tách biệt hoàn toàn với `connectDomains`; việc cấp phép tải một phông chữ hoặc mã kịch bản không đồng nghĩa với việc cho phép tải lên dữ liệu tùy ý.

### Cầu nối Apps sở hữu vòng đời riêng biệt

Cầu nối Apps là một biến thể JSON-RPC chạy trên cơ chế `postMessage`. Nó có thể trao đổi `ui/initialize` và các thông báo `ui/*`, đồng thời có thể ủy quyền cho các phương thức trông giống tầng lõi như `tools/call`.

Khung nhìn (View) gửi `ui/initialize` kèm theo `appInfo` và một đối tượng `appCapabilities`. Phía host trả về các năng lực của mình và ngữ cảnh host. Chỉ sau phản hồi đó, View mới gửi thông báo `ui/notifications/initialized`. Phía host bắt buộc phải đợi thông báo này của Apps trước khi gửi bất kỳ thông điệp nào tới View.

Cái bắt tay cục bộ đó tạo ra một cầu nối giữa một iframe và một khung host. Nó không hề đàm phán phiên bản giao thức MCP, không tạo ra trạng thái máy chủ, và không phát hành một phiên giao vận nào. Hãy lưu ý tiền tố chính xác: thông báo ở tầng lõi `notifications/initialized` đã bị xóa bỏ hoàn toàn, trong khi thông báo của Apps `ui/notifications/initialized` vẫn được giữ nguyên. Một yêu cầu tầng lõi sinh ra bởi một lệnh gọi công cụ qua cầu nối là một yêu cầu độc lập hoàn toàn mới với id JSON-RPC mới và đầy đủ siêu dữ liệu yêu cầu.

### Ngữ cảnh host, các hành động và việc thu hồi quyền

Host luôn nắm giữ thẩm quyền sau khi cầu nối đã khởi tạo. View chỉ có thể yêu cầu thực thi công cụ, điều hướng, sử dụng bộ nhớ tạm (clipboard), hoặc các hiệu ứng đặc quyền khác thông qua một năng lực mà host đã công bố. Host sẽ xác thực yêu cầu có định kiểu, người dùng hiện tại, đối tượng đích, và các đối số, áp dụng chính sách phê duyệt, và có thể từ chối thực hiện. Một cú nhấp chuột và một thông điệp cầu nối hợp lệ chỉ thể hiện ý định (intent); cả hai đều không trao quyền thẩm định (authority).

Hãy xử lý giao diện (theme), kích thước, và tính tiếp cận (accessibility) như ngữ cảnh host luôn biến đổi thay vì chỉ là dữ liệu kết xuất một lần:

- Áp dụng các thẻ thông số màu sắc và kiểu chữ do host cung cấp, sau đó phản hồi khi tùy chọn giao diện hoặc độ tương phản thay đổi.
- Cho phép View báo cáo kích thước mong muốn, nhưng để host giới hạn trần và áp đặt kích thước iframe sao cho nội dung không thể tràn ra ngoài bố cục hoặc tạo ra các lớp phủ đánh lừa thị giác.
- Duy trì thứ tự điều hướng phím (keyboard order), hiển thị con trỏ tiêu điểm rõ ràng, tên tiếp cận được, trạng thái trình đọc màn hình, độ tương phản đầy đủ, khả năng thu phóng, và hành vi giảm chuyển động bên trong iframe.
- Kiểm thử lại quá trình chuyển tiêu điểm giữa các điều khiển của host và các điều khiển của View sau mỗi lần thay đổi kích thước hoặc kết xuất lại.

Các năng lực có thể bị thu hồi khi App đang mở do người dùng đổi tài khoản, chính sách thay đổi, máy chủ bị cách ly kiểm dịch, hoặc host thu hẹp phạm vi đồng thuận. Hãy kiểm tra năng lực và ủy quyền ngay tại thời điểm thực hiện hành động, chứ không chỉ riêng trong lúc gọi `ui/initialize`. Khi bị thu hồi, hãy từ chối các lệnh gọi đặc quyền đang chờ xử lý, dừng mọi hoạt động mạng không còn phù hợp với chính sách, xóa sạch trạng thái nhạy cảm đã kết xuất, và gắn lại khung nhìn hoặc hạ cấp về văn bản khi bản thân tài nguyên UI không còn được chấp thuận. Phía View phải xử lý sự từ chối như một kết quả thông thường, chứ không được thử lại liên tục cho đến khi host nhượng bộ.

### Cơ chế hạ cấp là một phần của hợp đồng

Một máy chủ nhận biết Apps vẫn có thể phục vụ các host không công bố phần mở rộng UI:

- Trả về cùng công cụ đó nhưng không có `_meta.ui` trong `tools/list`.
- Duy trì kết quả dạng văn bản hữu ích cho `tools/call`.
- Từ chối `resources/read` đối với giao diện bằng lỗi thiếu năng lực.
- Tuyệt đối không giả định rằng một iframe đang tồn tại khi quyết định xem công cụ đã hoàn thành hay chưa.

````figure
t3-ui-sandbox
````

## Xây dựng hệ thống

Tệp `code/main.py` xây dựng một mô hình giao thức chạy nội tiến trình (in-process) mà không cần dùng đến SDK. Nó xác thực phong bì yêu cầu hiện tại và các giá trị định tuyến Streamable HTTP, công bố Apps thông qua `server/discover`, liệt kê các công cụ và tài nguyên, thực thi công cụ, và phục vụ một tài nguyên HTML tự chứa.

Mô hình tiếp nhận các phần thân thông điệp và tiêu đề định tuyến đã được phân tích cú pháp sẵn. Đây không phải là một bộ điều hợp HTTP hoàn chỉnh và không trực tiếp phân tích `Content-Type` hay `Accept`. Hãy xem Bài học 09 để tìm hiểu bộ điều hợp Streamable HTTP hoàn chỉnh yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy chương trình:

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Kiểm tra năm điểm then chốt trong kết quả đầu ra:

1. Mọi lệnh gọi đều độc lập hoàn toàn.
2. Mọi yêu cầu đều mang các năng lực trong `_meta`.
3. `resources/list` trả về bộ mô tả ổn định trước bất kỳ thao tác đọc tài nguyên nào.
4. Mọi kết quả đều có `resultType` và siêu dữ liệu định danh máy chủ.
5. Tuyệt đối không xuất hiện bất kỳ mã định danh phiên tầng lõi nào.

## Thử nghiệm thực tế

Bắt đầu với `server/discover`. Xác nhận rằng `io.modelcontextprotocol/ui` xuất hiện trong bản đồ phần mở rộng của máy chủ. Sau đó gọi `tools/list` hai lần: một lần có năng lực Apps và một lần không có. Phản hồi đầu tiên khai báo tài nguyên giao diện. Phản hồi thứ hai giữ nguyên là một công cụ thuần văn bản có thể sử dụng bình thường.

Đọc tài nguyên `ui://notes/timeline.html`. Tìm kiếm trong mã HTML hai nội dung `hostOrigin` và bộ phận bảo vệ `event.origin`. Hai dòng mã đó là bằng chứng trực quan tối thiểu cho thấy cầu nối không hề sử dụng đích đến dạng ký tự đại diện (wildcard target).

## Đóng gói sản phẩm

Bài học này đóng gói tệp `outputs/skill-mcp-apps-spec.md`. Hãy sử dụng tài liệu này để duyệt xét bản hợp đồng App trước khi viết mã nguồn khung ứng dụng. Nó buộc tác giả phải nêu rõ phong bì giao thức lõi hiện tại, cơ chế đàm phán phần mở rộng, giải pháp hạ cấp, tài nguyên UI, chính sách bộ nhớ đệm, CSP, các quyền hạn, các phương thức cầu nối, và ranh giới chấp thuận người dùng.

## Bài tập mở rộng

1. Thay đổi năng lực của client thành một bản đồ phần mở rộng rỗng. Xác nhận rằng `tools/list` vẫn giữ lại công cụ nhưng loại bỏ liên kết UI.
2. Gửi tiêu đề `Mcp-Name: ui://notes/other.html` kèm phần thân đọc dữ liệu timeline. Xác nhận hệ thống trả về lỗi `-32020`.
3. Thay đổi tài nguyên thành `cacheScope: private`. Mô tả điều kiện đặc thù của người dùng chứng minh cho sự thay đổi này.
4. Di chuyển mã kịch bản sang `https://static.example.com/app.js`. Thêm nguồn gốc đó vào `resourceDomains` và giải thích rủi ro chuỗi cung ứng (supply-chain risk) mới phát sinh.
5. Thêm một công cụ `notes_open` và định tuyến cú nhấp chuột qua host. Giữ quyền phê duyệt của người dùng tại host.

## Các thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| MCP Apps | Phần mở rộng tùy chọn cho phép hiển thị HTML tương tác do một host MCP kết xuất |
| `io.modelcontextprotocol/ui` | Mã định danh phần mở rộng được công bố bởi cả hai bên đối tác |
| `ui://` | Lược đồ định danh tài nguyên (resource scheme) cho bản mẫu giao diện của một App |
| `text/html;profile=mcp-app` | Kiểu MIME dành riêng cho mã HTML của ứng dụng MCP App |
| `server/discover` | Phương thức RPC hiện tại phục vụ khám phá phiên bản giao thức và năng lực |
| `resources/list` | Phương thức liệt kê tài nguyên bắt buộc khi máy chủ công bố năng lực tài nguyên |
| `resultType` | Bộ phân biệt bắt buộc cho các kết quả thành công hiện đại |
| `ui/initialize` | Yêu cầu cầu nối Apps đầu tiên, tách biệt khỏi quy trình khởi tạo lõi đã bị bãi bỏ |
| `ui/notifications/initialized` | Thông báo sẵn sàng của Khung nhìn Apps được gửi sau khi host phản hồi |
| CSP | Chính sách trình duyệt giới hạn mã kịch bản, kiểu dáng, hình ảnh và nguồn mạng |
| Hạ cấp văn bản (Text fallback) | Hành vi công cụ được giữ lại dành cho host không hỗ trợ Apps |

## Đọc thêm

- [Giao thức cơ sở MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Tổng quan về MCP Apps](https://modelcontextprotocol.io/extensions/apps/overview)
- [Hướng dẫn xây dựng MCP Apps](https://modelcontextprotocol.io/extensions/apps/build)
- [Ma trận hỗ trợ phần mở rộng chính thức](https://modelcontextprotocol.io/extensions/client-matrix)
