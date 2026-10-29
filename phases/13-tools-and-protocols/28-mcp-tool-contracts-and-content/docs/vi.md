# Hợp đồng và Nội dung Công cụ MCP (MCP Tool Contracts and Content)

> Một công cụ chỉ an toàn để tự động hóa khi khâu khám phá, đối số, kết quả, phân trang và siêu dữ liệu tầng truyền vận cùng thống nhất trên một bản hợp đồng duy nhất.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Phase 13, Bài học 07, 09, và 10  
**Thời lượng:** ~120 phút  

## Mục tiêu học tập

- Định nghĩa các đầu vào và đầu ra của công cụ bằng JSON Schema 2020-12.
- Xác thực các kết quả có cấu trúc mà không mặc định coi chúng là các đối tượng JSON (JSON objects).
- Lựa chọn phù hợp giữa văn bản (text), hình ảnh (image), âm thanh (audio), liên kết tài nguyên (resource links), và tài nguyên nhúng (embedded resources).
- Từ chối các định nghĩa `x-mcp-header` không an toàn trước khi công cụ tiếp cận mô hình.
- Mã hóa các giá trị tiêu đề tham số và xác minh tính đồng nhất tuyệt đối giữa tiêu đề và phần thân.
- Duyệt qua cơ chế phân trang bằng con trỏ (cursor pagination) mà không tự ý diễn giải các giá trị con trỏ.
- Giới hạn và cấp quyền cho các gợi ý từ điểm cuối `completion/complete`.

## Vấn đề

Gọi một hàm Python là điều dễ dàng. Nhưng việc gọi một năng lực từ xa thông qua một môi trường máy chủ AI lại là một bài toán về hợp đồng giao tiếp (contract problem).

Máy chủ xuất bản một bộ mô tả (descriptor). Máy khách chuyển đổi bộ mô tả đó thành ngữ cảnh cho mô hình và giao diện người dùng. Mô hình sinh ra các đối số. Một cổng kết nối (gateway) có thể định tuyến yêu cầu dựa trên các tiêu đề được ánh xạ. Máy chủ thực thi công cụ. Sau đó, máy khách quyết định xem kết quả trả về có đủ an toàn và hợp lệ để đưa trở lại cho mô hình hay không.

Chỉ cần một ranh giới yếu kém sẽ làm hỏng toàn bộ chuỗi mắt xích.

Hãy xem xét năm tình huống thất bại sau:

- Bộ mô tả thông báo kết quả là một đối tượng (object), nhưng máy chủ lại trả về một mảng (array).
- Máy khách dừng phân trang khi `nextCursor` là một chuỗi rỗng.
- Một tham số token bị ánh xạ thành tiêu đề HTTP và bị lộ ra trước các bên trung gian mạng.
- Một giá trị định tuyến Unicode được gửi dưới dạng tiêu đề thô, khiến cổng kết nối và máy chủ gốc phân giải thành các byte hoàn toàn khác nhau.
- Một điểm cuối hoàn thiện gợi ý một môi trường sản xuất (production) cho một người gọi vốn không có quyền truy cập vào đó.

Không có lỗi nào trong số này có thể sửa được bằng cách viết prompt khéo léo hơn. Chúng đòi hỏi các hợp đồng giao thức và ứng dụng tường minh.

## Đường ống Hợp đồng (The Contract Pipeline)

Hãy coi mỗi lượt gọi công cụ là việc đi qua năm cánh cổng:

1. **Khám phá (Discover):** Đọc danh sách công cụ tất định, có phân trang.
2. **Tiếp nhận (Admit):** Xác thực từng bộ mô tả và áp dụng chính sách bảo mật cục bộ.
3. **Kích hoạt (Invoke):** Xác thực các đối số và xây dựng siêu dữ liệu truyền vận.
4. **Thực thi (Execute):** Chạy hàm xử lý và phân loại các lỗi một cách chính xác.
5. **Tiêu thụ (Consume):** Xác thực các khối nội dung và đầu ra có cấu trúc trước khi mô hình sử dụng.

````figure mcp-contract-pipeline
````

Môi trường máy chủ nắm quyền kiểm soát tại các cổng tiếp nhận và tiêu thụ. Máy chủ công cụ không thể ép buộc máy khách phải tin tưởng các chú thích, lược đồ, hoặc đầu ra của nó.

## JSON Schema là một Ranh giới Môi trường chạy

Trong MCP phiên bản `2026-07-28`, `inputSchema` và `outputSchema` sử dụng JSON Schema. Khi thiếu trường `$schema`, phương ngữ mặc định là 2020-12.

Lược đồ đầu vào bắt buộc phải là một đối tượng lược đồ (schema object). Một công cụ không có đối số vẫn nên nêu rõ ràng những gì nó chấp nhận:

```json
{
  "type": "object",
  "additionalProperties": false
}
```

Cách viết này chặt chẽ hơn so với `{ "type": "object" }`, vốn chấp nhận các thuộc tính tùy ý.

Lược đồ đầu ra là tùy chọn. Một khi máy chủ đã xuất bản lược đồ đầu ra, mọi kết quả công cụ hoàn chỉnh đều cam kết trả về `structuredContent` tuân thủ lược đồ đó, kể cả các kết quả có `isError: true`. Cờ báo lỗi phân loại kết quả thực thi; nó không miễn trừ hợp đồng đầu ra đã công bố. Máy khách nên chủ động xác thực kết quả thay vì tin tưởng mù quáng vào bộ mô tả.

### Nội dung có cấu trúc là bất kỳ giá trị JSON nào

Đừng gán cứng `structuredContent` chỉ là một từ điển (dictionary/object). Nó có thể là:

- một đối tượng (object);
- một mảng (array);
- một chuỗi (string);
- một số (number);
- một giá trị boolean;
- giá trị `null`.

Công cụ dưới đây trả về một mảng:

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

Kết quả thành công của nó là hoàn toàn hợp lệ:

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

Để đảm bảo tính tương thích, các kết quả có cấu trúc cũng nên bao gồm chuỗi JSON đã tuần tự hóa bên trong một khối văn bản (text block). Khối văn bản này không phải là nguồn xác thực. Nguồn xác thực chính là `structuredContent`.

### Một bộ xác thực nhỏ vẫn giảng dạy rõ nét về ranh giới

Bài học sử dụng một tập hợp con có chủ ý của JSON Schema để duy trì tính thuần túy trong thư viện chuẩn của Python. Nó kiểm tra các cơ chế được sử dụng bởi các công cụ mẫu:

- các kiểu dữ liệu: object, array, string, integer, number, boolean, và null;
- các thuộc tính bắt buộc (required properties);
- `additionalProperties: false`;
- các phần tử mảng (array items);
- các giá trị liệt kê (enum values);
- độ dài chuỗi tối thiểu (minimum string length).

Đây không phải là sự thay thế cho một bộ xác thực hoàn chỉnh trong môi trường sản xuất. Bài học có giá trị tái sử dụng ở chỗ nó chỉ rõ nơi việc xác thực diễn ra: sau khi khám phá đối với bộ mô tả, trước khi thực thi đối với đối số, và trước khi tiêu thụ đối với các kết quả có cấu trúc.

## Các Khối Nội dung mang Chi phí Khác nhau

Mảng `content` có thể kết hợp nhiều loại khối nội dung khác nhau.

| Kiểu | Mục đích sử dụng | Ranh giới chính |
|------|------------|---------------|
| `text` | Tóm tắt cho con người và mô hình đọc | Coi văn bản là đầu ra không đáng tin cậy |
| `image` | Bằng chứng thị giác mã hóa dạng base64 | Xác thực kiểu phương tiện (media type) và dung lượng |
| `audio` | Đầu ra giọng nói hoặc âm thanh mã hóa base64 | Xác thực kiểu phương tiện và giới hạn thời lượng |
| `resource_link` | Một URI mà máy khách có thể tải sau | Cấp quyền lại cho lần đọc tài nguyên sau đó |
| `resource` | Dữ liệu nhúng trực tiếp trong kết quả | Thực thi các giới hạn tải trọng và nội dung ngay lập tức |

Một liên kết tài nguyên không phải là bằng chứng cho thấy tài nguyên đó sẽ xuất hiện trong `resources/list`. Nó chỉ là một tham chiếu được trả về bởi lượt gọi công cụ này. Máy khách vẫn áp dụng chính sách tài nguyên của mình khi truy cập theo URI.

Một tài nguyên nhúng giúp tránh thêm một lượt mạng khứ hồi (round trip) nhưng làm tăng kích thước của phản hồi hiện tại. Hãy sử dụng liên kết cho các tạo phẩm lớn hoặc thay đổi độc lập. Sử dụng tài nguyên nhúng cho các bằng chứng nhỏ cần di chuyển nguyên tử cùng với kết quả.

Kết quả `evidence_bundle` của bài học bao gồm cả năm loại khối nội dung này. Máy khách xác thực từng khối trước khi chấp nhận kết quả.

## `x-mcp-header` là Siêu dữ liệu Định tuyến

Một thuộc tính bên trong `inputSchema` có thể khai báo `x-mcp-header`. Khi chạy qua giao vận Streamable HTTP, máy khách sẽ ánh xạ đối số đó vào tiêu đề `Mcp-Param-{name}`.

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

Với đối số `region: "eu-west"`, tầng truyền vận có thể phát ra:

```http
Mcp-Param-Region: eu-west
```

Chú thích này tồn tại để bộ cân bằng tải, cổng kết nối, hoặc công cụ thực thi chính sách có thể định tuyến mà không cần phân tích cú pháp phần thân JSON. Đây tuyệt đối không phải là nơi để đặt thông tin xác thực.

Giao thức áp đặt các ràng buộc chặt chẽ lên chú thích này:

- tên tiêu đề không được rỗng và phải tuân theo cú pháp token tên trường HTTP;
- tên tiêu đề là duy nhất không phân biệt hoa thường;
- kiểu thuộc tính phải là string, integer, hoặc boolean;
- kiểu `number` không được phép sử dụng;
- chú thích chỉ xuất hiện trên một thành viên trực tiếp của `inputSchema.properties`;
- giá trị số nguyên phải nằm trong phạm vi từ `-9007199254740991` đến `9007199254740991`.

Quy tắc vị trí là quy tắc cú pháp và phải thất bại đóng (fail-closed). Hãy duyệt toàn bộ cây lược đồ, chứ không chỉ các thuộc tính mà bộ xác thực của bạn tình cờ hiểu được. Hãy từ chối một chú thích nằm bên dưới `properties` của một đối tượng lồng nhau, một nhánh `oneOf`, `items`, một định nghĩa được truy cập qua `$ref`, hoặc bất kỳ lược đồ đầu ra nào. Việc phân giải một tham chiếu không biến nút được tham chiếu thành một thuộc tính cấp cao nhất trực tiếp.

Bài học này bổ sung một chính sách triển khai: từ chối các bộ mô tả ánh xạ các tên nhạy cảm như `password`, `secret`, `token`, `api_key`, hoặc `authorization`. Đặc tả chính thức khuyên các tác giả máy chủ không ánh xạ các tham số nhạy cảm. Máy khách có thể biến lời khuyên đó thành một quy tắc tiếp nhận cứng.

Hãy kiểm toán tên tiêu đề, chứ không phải giá trị của nó. Mã mẫu ghi lại `Mcp-Param-Region` trong khi giữ `eu-west` nằm ngoài sự kiện kiểm toán.

### Mã hóa các giá trị trước khi tạo tiêu đề HTTP

Một giá trị tham số chỉ có thể truyền đi dưới dạng văn bản thuần túy khi nó là một chuỗi không rỗng gồm các ký tự ASCII hiển thị được từ `!` đến `~` và không giống với lính canh mã hóa (encoding sentinel). Mọi trường hợp khác bắt buộc phải dùng định dạng chính xác này:

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` là chuỗi base64 tiêu chuẩn trên các byte UTF-8 chính xác. Không cắt tỉa (trim), không chuẩn hóa, và không thay thế giá trị trước khi mã hóa. Mã hóa Unicode, chuỗi rỗng, khoảng trắng, dấu tab, ký tự điều khiển, ký tự CR hoặc LF, khoảng trắng đầu/cuối, và bất kỳ giá trị nào bắt đầu bằng `=?base64?`. Việc mã hóa lại một giá trị có hình thức giống lính canh là cơ chế giúp bên nhận khôi phục chính xác văn bản gốc thay vì giải mã nó như cú pháp truyền vận.

Các giá trị boolean hiển thị dưới dạng chữ thường `true` hoặc `false`. Các số nguyên hiển thị ở hệ cơ số 10 và phải nằm trong phạm vi số nguyên an toàn của JavaScript. Các giá trị nằm ngoài phạm vi đó sẽ bị từ chối thay vì bị làm tròn bởi một bên trung gian.

### Máy chủ kiểm tra bản sao được ánh xạ

Tạo tiêu đề chỉ là một nửa công việc từ phía máy khách. Tại ranh giới Streamable HTTP, máy chủ bắt buộc phải:

1. tìm các tên `Mcp-Param-*` được nhận diện mà không phân biệt hoa thường;
2. giải mã dạng lính canh base64 chính xác khi nó xuất hiện;
3. so sánh văn bản đã giải mã với đối số tương ứng trong phần thân JSON một cách chính xác tuyệt đối;
4. từ chối tiêu đề được nhận diện bị thiếu, bị trùng lặp, bất thường, sai định dạng, hoặc không khớp trước khi điều phối thực thi.

Phản hồi từ chối là mã HTTP `400` với mã lỗi JSON-RPC `-32020`. Cả giá trị phần thân lẫn dạng tiêu đề mã hóa của nó đều không được đưa vào bản ghi kiểm toán. Chỉ ghi lại tên tiêu đề được nhận diện và phân loại từ chối.

`code/main.py` mô hình hóa trực tiếp ranh giới này. [Bài học 09](../../09-mcp-transports/) trình bày thứ tự xác thực Streamable HTTP rộng hơn, bao gồm tính đồng nhất của phương thức và phiên bản giao thức.

## Con trỏ Phân trang là Giá trị Mờ đục (Opaque Cursors)

Các thao tác danh sách của MCP sử dụng cơ chế phân trang bằng con trỏ (cursor pagination). Máy chủ chọn kích thước trang và định dạng con trỏ. Máy khách chỉ có một quyết định:

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

Đừng viết như thế này:

```python
if not result.get("nextCursor"):
    break
```

Một chuỗi rỗng `""` là một con trỏ hoàn toàn hợp lệ. Việc kiểm tra chân giá trị (truthiness) kiểu Python sẽ dừng quá trình phân trang quá sớm.

Máy khách không được giải mã con trỏ, không được tăng giá trị con trỏ, không so sánh nó với con trỏ trước để sắp xếp, và không suy diễn số trang. Một máy chủ có thể ký số vào con trỏ, gắn nó với một phiên bản danh mục, hoặc ánh xạ nó vào trạng thái riêng tư. Đó là chi tiết triển khai nội bộ của máy chủ.

Máy chủ mẫu cố tình trả về `""` sau trang đầu tiên. Máy khách bắt buộc phải gửi chính xác giá trị đó trong yêu cầu thứ hai. Dấu vết của nó là:

```text
<yêu cầu đầu tiên không có con trỏ>
<yêu cầu thứ hai với con trỏ "">
```

Các con trỏ không hợp lệ sẽ tạo ra lỗi thông số không hợp lệ trong JSON-RPC, mã `-32602`.

## Tính năng Hoàn thiện là một Bề mặt Ủy quyền (Authorization Surface)

Điểm cuối `completion/complete` cung cấp các gợi ý cho các đối số câu lệnh (prompt arguments) và đối số mẫu tài nguyên (resource-template arguments). Nó rất hữu ích cho các biểu mẫu tương tác, nhưng nó có thể làm rò rỉ những cái tên mà các phương thức danh sách thông thường bảo vệ.

Một yêu cầu hoàn thiện chỉ định một tham chiếu và đối số đang được hoàn thiện:

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

Kết quả trả về tối đa 100 giá trị và có thể báo cáo `total` kèm theo `hasMore`.

Hãy áp dụng cùng một ranh giới ủy quyền được sử dụng bởi prompt hoặc tài nguyên được tham chiếu. Một chuyên viên phân tích trong mẫu chỉ nhận được `development` và `staging`. Chỉ có người vận hành (operator) mới có thể nhận được `production`.

Tính năng hoàn thiện trong môi trường sản xuất cũng cần:

- xác thực đầu vào;
- lọc nhận biết người gọi;
- chống rung (debouncing) yêu cầu ở phía máy khách;
- giới hạn tần suất (rate limiting) ở phía máy chủ;
- giới hạn số lượng kết quả;
- nhật ký không để lộ các giá trị gợi ý nhạy cảm.

Hoàn thiện là hỗ trợ tương tác, không phải là cơ chế vượt qua việc khám phá.

## Hai Tầng Lỗi (Two Error Layers)

Hãy tách biệt rạch ròi giữa lỗi giao thức và lỗi thực thi công cụ.

Sử dụng lỗi JSON-RPC khi yêu cầu MCP không thể điều phối chính xác:

- tên công cụ không tồn tại;
- định dạng yêu cầu sai cấu trúc;
- thiếu siêu dữ liệu yêu cầu;
- con trỏ không hợp lệ.

Sử dụng kết quả công cụ hoàn chỉnh với `isError: true` khi lượt gọi đã đến được công cụ và công cụ báo cáo một thất bại có thể hành động được:

- một nguồn báo cáo không khả dụng;
- một ngày nằm ngoài phạm vi hỗ trợ;
- một quy tắc nghiệp vụ từ chối thao tác được yêu cầu.

Mô hình thường có thể tự sửa chữa một lỗi thực thi công cụ. Nhưng chúng không thể sửa chữa một máy chủ vi phạm chính lược đồ đầu ra của nó.

Nếu công cụ khai báo một lược đồ đầu ra, hãy mô hình hóa thất bại có thể xử lý được bên trong lược đồ đó. Lỗi mẫu của `route_report` trả về khu vực được yêu cầu kèm theo `accepted: false`, cùng với văn bản lỗi cho con người đọc và `isError: true`.

## Xây dựng (Build It)

`code/main.py` xây dựng cả hai phía của ranh giới bằng thư viện chuẩn Python.

Máy chủ cài đặt:

- xác thực siêu dữ liệu MCP trên từng yêu cầu;
- `server/discover` với các năng lực công cụ và hoàn thiện;
- phân trang `tools/list` tất định;
- bốn bộ mô tả công cụ, bao gồm một bộ bắt buộc phải bị từ chối;
- đầu ra có cấu trúc dạng mảng;
- mọi loại khối nội dung công cụ hiện hành;
- cổng kiểm tra tính đồng nhất Streamable HTTP giải mã các tiêu đề tham số được nhận diện và trả về HTTP `400` cùng JSON-RPC `-32020` khi không khớp;
- tính năng hoàn thiện được phân quyền và giới hạn tần suất.

Máy khách cài đặt:

- tiếp nhận bộ mô tả;
- xác thực vị trí `x-mcp-header` trên toàn bộ cây và chính sách trường nhạy cảm;
- mã hóa giá trị chuẩn xác bằng ASCII hiển thị thuần túy hoặc base64 UTF-8;
- vòng lặp con trỏ mờ đục tuân thủ chuỗi rỗng;
- xác thực đối số và kết quả;
- xác thực khối nội dung;
- sự kiện kiểm toán tiêu đề chỉ chứa tên mà không chứa giá trị.

Bộ mô tả cố tình không an toàn là dữ liệu giảng dạy. Nó chứng minh rằng một công cụ bị từ chối không ngăn cản các công cụ hợp lệ khác được nạp vào hệ thống.

## Sử dụng (Use It)

Từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Bản minh họa in ra các công cụ được tiếp nhận, bộ mô tả bị từ chối, cả hai yêu cầu phân trang, nội dung mảng có cấu trúc, các kiểu khối nội dung, tên tiêu đề được ánh xạ, việc giá trị có cần mã hóa hay không, trạng thái đồng nhất HTTP, và các giá trị hoàn thiện đã được lọc theo người gọi.

## Phòng thực hành Tương tác (Interactive Lab)

Mở `code/main.py` và tìm biến `TOOLS`.

1. Thay đổi `tag_catalog.outputSchema.type` từ `array` thành `object`.
2. Chạy bản minh họa. Máy khách sẽ từ chối mảng được trả về.
3. Khôi phục lại lược đồ.
4. Giữ `nextCursor` của trang đầu tiên là `""`, sau đó làm cho trang cuối cùng trả về `nextCursor: None` thay vì bỏ qua trường đó.
5. Chạy các bài kiểm thử và so sánh dấu vết con trỏ.
6. Thêm `x-mcp-header: "Authorization"` vào một thuộc tính chuỗi.
7. Xác nhận việc tiếp nhận bộ mô tả từ chối nó trước khi kích hoạt.
8. Thử các giá trị `region` chứa Unicode, dấu xuống dòng, khoảng trắng bao quanh, và chuỗi văn bản nguyên văn `=?base64?SGVsbG8=?=`. Giải mã từng tiêu đề phát ra và chứng minh giá trị gốc được giữ nguyên vẹn.
9. Di chuyển chú thích xuống dưới `oneOf`, `items`, hoặc một định nghĩa `$ref`. Xác nhận từng bộ mô tả bị từ chối ngay cả khi nhánh đó không bao giờ được dùng trong bản demo.
10. Xóa tiêu đề được nhận diện hoặc thay đổi giá trị đã giải mã của nó. Xác nhận ranh giới HTTP trả về trạng thái `400` và mã JSON-RPC `-32020`.

Mục đích không phải là học vẹt một cấu trúc JSON. Mục đích là quan sát từng cổng kiểm soát thất bại tại đúng ranh giới nắm quyền sở hữu nó.

## Phòng thực hành Ứng dụng (Practice Lab)

Mở rộng phòng thực hành hợp đồng với một công cụ `search_evidence`.

Yêu cầu:

1. Lược đồ đầu vào chấp nhận `query`, `limit`, và một trường định tuyến an toàn `region`.
2. Lược đồ đầu ra là một mảng các đối tượng chứa `uri`, `title`, và `score`.
3. Kết quả bao gồm văn bản tương thích và một liên kết tài nguyên cho mỗi mục.
4. Đối số từ chối các thuộc tính không xác định.
5. `limit` bị giới hạn bởi bước xác thực ứng dụng.
6. Người gọi không có quyền truy cập vào một URI sẽ không bao giờ nhìn thấy URI đó qua tính năng hoàn thiện hoặc đầu ra công cụ.
7. Các bài kiểm thử bao gồm điểm số không tuân thủ, chú thích tiêu đề không hợp lệ, và danh sách hai trang.
8. Các bài kiểm thử giá trị tiêu đề bao quát ASCII hiển thị, Unicode, ký tự điều khiển, khoảng trắng, văn bản dạng lính canh, và cả hai biên số nguyên an toàn của JavaScript.
9. Dữ liệu kiểm thử HTTP chấp nhận tên tiêu đề không phân biệt hoa thường nhưng từ chối các giá trị được nhận diện bị thiếu hoặc không khớp với trạng thái `400` và mã `-32020`.

## Tạo phẩm Chuyển giao (Shipped Artifact)

`outputs/skill-mcp-contract-reviewer.md` là một kỹ năng đánh giá phẳng, có thể tái sử dụng. Cung cấp cho nó một bộ mô tả công cụ, các kết quả mẫu, hành vi phân trang và chính sách hoàn thiện. Nó sẽ trả về một quyết định tiếp nhận, kế hoạch xác thực kết quả, chính sách tiêu đề, và các ca kiểm thử thất bại cụ thể.

## Xác minh (Verify It)

Bài học hoàn thành khi các khẳng định sau là đúng:

- `tools/list` trả về cùng một thứ tự logic qua các lượt gọi lặp lại.
- Máy khách thực hiện yêu cầu thứ hai khi `nextCursor` là `""`.
- Bộ mô tả tiêu đề nhạy cảm không an toàn bị loại trừ trong khi các công cụ khác vẫn khả dụng.
- Một mảng vượt qua lược đồ đầu ra dạng mảng của nó.
- Một đối tượng thất bại trước chính lược đồ dạng mảng đó.
- Kết quả lỗi không thể bỏ qua hoặc vi phạm lược đồ đầu ra đã công bố.
- Các khối văn bản, hình ảnh, âm thanh, liên kết tài nguyên và tài nguyên nhúng đều được xác thực.
- Sự kiện kiểm toán tiêu đề chỉ chứa tên mà không chứa giá trị.
- Ký tự ASCII hiển thị thuần túy được giữ nguyên; các giá trị Unicode, điều khiển, đệm khoảng trắng, rỗng và dạng lính canh đều khôi phục toàn vẹn qua mã hóa base64 UTF-8 chuẩn xác.
- Các số nguyên được ánh xạ nằm ngoài phạm vi an toàn của JavaScript bị từ chối.
- Các chú thích bên dưới `oneOf`, `items`, các đối tượng lồng nhau, định nghĩa `$ref`, hoặc lược đồ đầu ra đều bị từ chối trong khâu tiếp nhận.
- Tên tiêu đề được nhận diện không phân biệt hoa thường chỉ vượt qua khi giá trị giải mã khớp chính xác với phần thân; bản sao bị thiếu hoặc không khớp sinh ra HTTP `400` và JSON-RPC `-32020`.
- Hoàn thiện câu lệnh cho chuyên viên phân tích không bao giờ trả về `production`.
- Thất bại công cụ sử dụng `isError: true`; một lời gọi giao thức sai định dạng sử dụng `error` của JSON-RPC.

## Các Chế độ Lỗi trong Môi trường Sản xuất

| Lỗi gặp phải | Hiện tượng quan sát | Phản hồi xử lý đúng đắn |
|---------|-----------------------|------------------|
| Máy khách mặc định đầu ra là đối tượng | Các mảng hợp lệ bị lỗi hoặc bị bọc ngầm | Xác thực dựa trên lược đồ đã công bố mà không ép buộc kiểu đối tượng |
| Con trỏ rỗng bị coi là false | Các trang cuối biến mất | Tiếp tục phân trang bất cứ khi nào `nextCursor` tồn tại và không phải là null |
| Giá trị nhạy cảm bị ánh xạ | Bí mật xuất hiện trong proxy, WAF, hoặc dữ liệu vết | Từ chối bộ mô tả và giữ bí mật trong dữ liệu yêu cầu được bảo vệ |
| Unicode hoặc khoảng trắng thô bị ánh xạ | Cổng kết nối và máy chủ gốc không thống nhất hoặc giá trị bị chuẩn hóa | Sử dụng mã hóa lính canh base64 UTF-8 chuẩn xác và so sánh sau khi giải mã |
| Chú thích ẩn trong nhánh lược đồ | Máy khách bỏ sót siêu dữ liệu định tuyến khi tiếp nhận | Duyệt toàn bộ cây lược đồ và chỉ cho phép các thuộc tính cấp cao nhất trực tiếp |
| Số nguyên lớn bị ánh xạ | Bên trung gian JavaScript làm tròn giá trị định tuyến | Từ chối các giá trị nằm ngoài phạm vi số nguyên an toàn của JavaScript |
| Tiêu đề và phần thân không khớp | Cổng định tuyến một đích trong khi máy chủ gốc thực thi đích khác | Từ chối trước khi điều phối với HTTP `400` và JSON-RPC `-32020` |
| Lược đồ đầu ra bị lờ đi | Mã phía sau tiêu thụ cấu trúc dữ liệu bị sai hỏng | Xác thực trước khi đưa vào mô hình hoặc ứng dụng |
| Tự động tin tưởng liên kết tài nguyên | Người gọi truy cập vào một URI không được cấp quyền | Cấp quyền lại cho mỗi lần đọc tài nguyên |
| Hoàn thiện chia sẻ gợi ý toàn cục | Tên của các đối tượng thuê (tenant) bị rò rỉ | Lọc theo người gọi, tham chiếu và thẩm quyền |
| Chú thích công cụ bị coi là chính sách | Thao tác phá hủy bỏ qua bước xác nhận | Thực thi ủy quyền và phê duyệt bên ngoài các chú thích |
| Một công cụ lỗi làm hỏng khâu khám phá | Toàn bộ máy chủ trở nên không khả dụng | Từ chối bộ mô tả lỗi và tiếp nhận các công cụ hợp lệ một cách độc lập |

## Kết nối Capstone

Dự án capstone của Phase 13 cần một cổng kết nối có thể hợp nhất các công cụ từ nhiều máy chủ khác nhau. Bài học này cung cấp phần lõi tiếp nhận cho cổng kết nối đó.

Sử dụng tạo phẩm để chấm điểm bốn bằng chứng capstone:

- khám phá phân trang tất định và đầy đủ;
- xác thực bộ mô tả trước khi phơi bày cho mô hình;
- đầu ra có cấu trúc đã được xác thực kèm theo các khối nội dung có giới hạn;
- siêu dữ liệu hoàn thiện và định tuyến bảo toàn các ranh giới ủy quyền.

Đừng chỉ dựa vào một lệnh `tools/call` thành công để khẳng định tính tương thích của cổng kết nối. Hãy thu thập bộ mô tả, dấu vết phân trang, tập công cụ được tiếp nhận, tập công cụ bị từ chối, và một kết quả đã được xác thực.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa kỹ thuật |
|------|---------|
| `inputSchema` | Đối tượng JSON Schema định nghĩa các đối số công cụ được chấp nhận |
| `outputSchema` | JSON Schema tùy chọn định nghĩa `structuredContent` |
| `structuredContent` | Bất kỳ giá trị JSON nào được sinh ra bởi một kết quả công cụ |
| Khối nội dung (Content block) | Văn bản, hình ảnh, âm thanh, liên kết tài nguyên hoặc tài nguyên nhúng có kiểu |
| `x-mcp-header` | Chú thích lược đồ ánh xạ một đối số nguyên thủy vào siêu dữ liệu Streamable HTTP |
| Con trỏ mờ đục (Opaque cursor) | Token phân trang do máy chủ phát hành mà máy khách không tự ý diễn giải giá trị |
| Tham chiếu hoàn thiện (Completion reference) | Tên prompt hoặc URI/mẫu tài nguyên có đối số đang được hoàn thiện |
| Tiếp nhận (Admission) | Quyết định của máy khách về việc phơi bày hay từ chối một bộ mô tả được khám phá |

## Đọc thêm

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
