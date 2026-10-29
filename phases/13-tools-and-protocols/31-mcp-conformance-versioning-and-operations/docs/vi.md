# Kỹ nghệ Đảm bảo Hợp chuẩn MCP: Quản lý Phiên bản, Bằng chứng và Vận hành

> Một máy chủ không hề hợp chuẩn chỉ vì đường dẫn thuận lợi (happy path) của nó chạy thông suốt qua một SDK duy nhất. Tính hợp chuẩn (conformance) tồn tại ở cấp độ đường truyền (wire level), tại các ranh giới phiên bản, qua các mắt xích trung gian và trong suốt quá trình hoàn nguyên (rollback).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 phút

## Mục tiêu Học tập

- Chuyển hóa các quy tắc quy chuẩn MCP thành các bản ghi đường truyền chuẩn mực (golden wire transcripts) và bản ghi phủ định (negative wire transcripts).
- Tách biệt rạch ròi hành vi nghiêm ngặt của phiên bản `2026-07-28` với cơ chế dự phòng có giới hạn cho phiên bản cũ (bounded legacy fallback).
- Phân biệt các trường bổ sung chưa xác định (additive unknown fields) với một `resultType` không hợp lệ và chưa rõ.
- Đối chiếu bằng chứng JSON-RPC thô với khung nhìn đã chuẩn hóa của SDK (SDK-normalized view).
- Chứng minh tính toàn vẹn của tiêu đề (header) và thân thông điệp (body) qua một ranh giới proxy thực tế.
- Kiểm soát việc phát hành bằng các bằng chứng đã làm sạch thông tin nhạy cảm (redacted transcript), dữ liệu giám sát sức khỏe hệ thống và khả năng hoàn nguyên.

## Vấn đề Thực tế

Client của bạn gọi `tools/list` thông qua một SDK và nhận về danh sách công cụ. Bài kiểm thử tích hợp (integration test) thành công.

Tuy nhiên, kết quả đó vẫn để ngỏ nhiều câu hỏi trọng yếu:

- Yêu cầu gửi đi có mang siêu dữ liệu giao thức hiện đại trên từng yêu cầu (per-request protocol metadata) không?
- Các tiêu đề `MCP-Protocol-Version`, `Mcp-Method` và `Mcp-Name` có khớp hoàn toàn với thân thông điệp JSON-RPC không?
- Phản hồi nhận được trên đường truyền có chứa một `resultType` hợp lệ không, hay SDK đã tự ý tổng hợp ra nó?
- Client có bảo toàn được một trường dữ liệu bổ sung trong tương lai hay không?
- Liệu một mã lỗi hiện đại đã được công nhận có vô tình kích hoạt quá trình bắt tay theo chuẩn cũ (legacy handshake) không?
- Thành phần proxy có giữ nguyên mã trạng thái gốc và chi tiết lỗi JSON-RPC không?
- Bộ tuần tự hóa thông báo (notification serializer) có phát ra phản hồi bị cấm hay không?
- Đội ngũ vận hành có thể chứng minh lý do một bản phát hành được phê duyệt nâng cấp hoặc bị hoàn nguyên mà không để lộ bí mật hệ thống (secrets) hay không?

Tính hợp chuẩn là một tập hợp các bất biến có thể quan sát được (observable invariants). Hãy xây dựng một khung kiểm thử (harness) ghi lại các bất biến đó trước khi lưu lượng production thực tế phát hiện ra lỗi.

````figure mcp-conformance-operations
````

## Khởi đầu với các Thời kỳ Phiên bản

MCP `2026-07-28` sử dụng siêu dữ liệu khép kín trên từng yêu cầu. Một yêu cầu hiện đại mang các trường `params._meta.io.modelcontextprotocol/protocolVersion` và `params._meta.io.modelcontextprotocol/clientCapabilities`. Các khóa có không gian tên (namespaced keys) chính xác là yếu tố bắt buộc; các bí danh trần trụi như `protocolVersion` hay `clientCapabilities` đều bị coi là sai cấu trúc (malformed). Khi các tiêu đề định tuyến đối chiếu (mirrored routing headers) xuất hiện tại ranh giới HTTP, giá trị của chúng phải đồng nhất tuyệt đối với thân JSON-RPC. Các kết quả thành công hiện đại bắt buộc phải mang `resultType`.

Các phiên bản cho đến `2025-11-25` sử dụng thời kỳ khởi tạo (initialization era) trước đó. Một kết quả phiên bản cũ không có `resultType` chỉ được xem là hoàn tất (complete) sau khi client đã chủ động chọn thời kỳ cũ đó.

Đừng tạo ra một bộ xác thực lỏng lẻo chấp nhận cả hai định dạng cùng lúc. Hãy phân tách thành hai nhánh độc lập:

| Nhánh | Bằng chứng đầu vào | Thiếu `resultType` | Khởi tạo |
|---|---|---|---|
| Hiện đại (Modern) | `server/discover` thành công hoặc phản hồi hiện đại hợp lệ | Không hợp lệ | Không phải là luồng mặc định |
| Phiên bản cũ (Legacy) | Danh sách cho phép (allowlist) được cấu hình cộng với kết quả `initialize` chuẩn sau khi thăm dò hiện đại không cho kết quả dứt khoát | Được hiểu là hoàn tất (`complete`) | Bắt buộc theo quy chuẩn thời kỳ đó |

Sự phân tách này ngăn chặn tình trạng một bên giao tiếp hiện đại nhưng sai cấu trúc lại được "thưởng" bằng quy trình xác thực lỏng lẻo hơn.

### Chế độ Nghiêm ngặt (Strict mode)

Chế độ nghiêm ngặt đòi hỏi bằng chứng về hành vi hiện đại. Một lệnh `server/discover` thành công sẽ xác thực nhánh hiện đại. Một mã lỗi JSON-RPC hiện đại đã được định danh cũng xác thực nhánh này. Hãy sửa lại yêu cầu hoặc dừng lại. Tuyệt đối không hạ cấp (downgrade) giao thức chỉ vì máy chủ trả về các mã lỗi `-32020`, `-32021` hoặc `-32022`.

### Chế độ Dự phòng (Fallback mode)

Chế độ dự phòng thực hiện một lần thăm dò hiện đại có giới hạn thời gian. Hiện tượng hết thời gian chờ (timeout), phản hồi rỗng, kết nối bị đóng hoặc phản hồi không nhận diện được là các kết quả không dứt khoát. Chúng không chứng minh bên kia là phiên bản cũ. Chỉ một endpoint được cấu hình tường minh hoặc nằm trong danh sách cho phép tương thích mới được nhận tiếp một lần thăm dò theo chuẩn cũ có giới hạn; client chỉ chuyển sang nhánh cũ sau khi xác thực kết quả `initialize` của lần thăm dò đó và bản sửa đổi cũ đã được đàm phán thành công.

Cơ chế dự phòng không phải là "thử lại kiểu cũ sau bất kỳ lỗi nào". Một mã lỗi hiện đại được nhận diện chứa đựng thông tin điều chỉnh rất giá trị. Hạ cấp sau lỗi đó có thể che giấu sự bất đồng tiêu đề, sự thiếu hụt khai báo năng lực (capability), hoặc một phiên bản không được hỗ trợ.

Điều này giúp ngăn chặn kẻ tấn công, sự cố mạng hoặc một proxy lọc gói ép hạ cấp giao thức bằng cách đánh rơi phản hồi hiện đại. Hãy lưu lại chính sách endpoint, quan sát hiện đại không dứt khoát, bằng chứng xác thực chuẩn cũ cụ thể và thời kỳ được chọn trong cùng một bản ghi.

Lưu thời kỳ được chọn ngay cạnh mỗi bản ghi đường truyền. Nếu thiếu thông tin thực tế này, một trường bị thiếu có thể được coi là hợp lệ trong lần chạy kiểm thử này nhưng lại bị coi là sai trong lần chạy khác.

## Xây dựng Tập Bản ghi Đường truyền

Một bản mẫu (fixture) ghi lại dữ liệu đường truyền sẽ ghi nhận chính xác những gì đã vượt qua ranh giới mạng, chứ không chỉ ghi lại lời gọi SDK:

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

Hãy duy trì hai nhóm bản mẫu.

### Bản ghi Vàng (Golden transcripts)

Bản ghi vàng chứng minh các hành vi được chấp thuận:

- Khám phá hiện đại hoặc yêu cầu phương thức với siêu dữ liệu và tiêu đề khớp nhau.
- Kết quả hoàn chỉnh (`complete`) chứa đầy đủ các trường bắt buộc.
- Kết quả `input_required` khi phương thức cần yêu cầu thêm dữ liệu đầu vào.
- Kết quả phần mở rộng (extension) chỉ xuất hiện sau khi năng lực tương ứng đã được quảng bá (advertised).
- Kết quả phiên bản cũ không có `resultType`, nhưng chỉ hợp lệ trong thời kỳ cũ đã được chọn.
- Xử lý thông báo (notification) không phát sinh phản hồi JSON-RPC.

Một bản ghi vàng cần sự chính xác thay vì kích thước đồ sộ. Hãy giữ cho các ID biến động và dấu thời gian mang tính tất định (deterministic) hoặc chuẩn hóa chúng trước khi so sánh.

### Bản ghi Phủ định (Negative transcripts)

Bản ghi phủ định chứng minh hành vi từ chối chuẩn xác:

- Tiêu đề và thân thông điệp không khớp nhau.
- Thiếu khai báo năng lực trên từng yêu cầu.
- Phiên bản giao thức đã đối chiếu không được hỗ trợ.
- Thiếu `resultType` theo chuẩn hiện đại.
- `resultType` không xác định hoặc chưa từng được quảng bá năng lực.
- Phản hồi có trường `jsonrpc` khác `2.0` hoặc ID sai khác về giá trị hay kiểu dữ liệu JSON.
- Phản hồi chứa cả `result` và `error`, hoặc không chứa trường nào.
- Lỗi không có mã nguyên `code` và chuỗi thông báo `message`.
- Một lỗi giao thức đã biết bị ánh xạ sai thành mã trạng thái HTTP không tương ứng.
- Phát sinh phản hồi cho một thông báo (notification).
- Bao đóng (envelope) JSON-RPC bị sai cấu trúc.
- Proxy làm sụp đổ (collapse) thông tin lỗi giao thức thành lỗi chung.

Đối với từng trường hợp phủ định, hãy kiểm tra ranh giới từ chối và mã lỗi ổn định. Nhận định "lời gọi thất bại" là quá mơ hồ. Một mã lỗi 500 do proxy phát sinh và một mã lỗi `-32020` từ máy chủ gốc đều có biểu hiện bề ngoài là thất bại, nhưng chúng phản ánh hai bản chất sự cố hoàn toàn khác nhau cho kỹ sư vận hành.

Bản mẫu kiểm tra bất đồng tiêu đề phải bao gồm phản hồi JSON-RPC HTTP 400 thực tế của máy chủ với ID yêu cầu trùng khớp và mã lỗi `-32020`. Hãy tự động thực thi điều đó bất cứ khi nào bộ xác thực cục bộ phát hiện `HeaderMismatch`; đừng biến việc xác minh phản hồi thành một cờ tùy chọn trong fixture. Trường hợp trả về HTTP 500 không có thân thông điệp sẽ bị đánh trượt ngay cả khi mã từ chối cục bộ là chính xác. Một khung kiểm thử chỉ biết dừng lại sau khi bộ xác thực yêu cầu của chính nó ném ra ngoại lệ thì mới chỉ kiểm thử chính nó, chứ chưa hề kiểm thử hành vi trên đường truyền của máy chủ.

Dự án kiểm định hợp chuẩn MCP chính thức (MCP conformance project) rất hữu ích như một bộ kiểm thử bên ngoài và tài liệu tham chiếu có phiên bản. Tuy nhiên, bạn vẫn cần duy trì các bản ghi cục bộ của riêng mình. Chúng nắm bắt đặc thù proxy, SDK, xác thực, phần mở rộng và quy trình phát hành riêng biệt của bạn mà một bộ kiểm định chung không thể biết tới.

## Giá trị Tiêu đề Phải Khớp với Thân RPC

Trong cơ chế Streamable HTTP hiện đại, các mắt xích trung gian có thể định tuyến hoặc áp đặt chính sách bằng cách sử dụng các tiêu đề đối chiếu. Thân thông điệp JSON-RPC luôn là nguồn chân lý duy nhất của giao thức. Sự bất đồng giữa tiêu đề và thân là lỗi vi phạm tính toàn vẹn (integrity failure), không phải là gợi ý để hệ thống tùy ý chọn một trong hai giá trị.

Hãy xác thực theo đúng thứ tự sau:

1. Phân tích cú pháp và xác thực bao đóng JSON-RPC cùng các kiểu dữ liệu siêu dữ liệu.
2. So sánh `MCP-Protocol-Version` với `params._meta.io.modelcontextprotocol/protocolVersion`.
3. So sánh `Mcp-Method` với `method`.
4. Khi phương thức có tên định tuyến, so sánh `Mcp-Name` với giá trị tương ứng trong thân.
5. Sau khi tính tương đương đã được thiết lập, mới quyết định xem phiên bản và tập năng lực đã đối chiếu có được hỗ trợ hay không.

Thứ tự này giúp phân biệt rõ ràng giữa lỗi không khớp `-32020` với lỗi không hỗ trợ phiên bản `-32022`. Nó cũng ngăn chặn tình huống gateway cấp quyền cho một tên công cụ trên tiêu đề trong khi máy chủ gốc lại thực thi một tên khác nằm trong thân thông điệp.

Tên trường HTTP không phân biệt chữ hoa chữ thường, nhưng giá trị của chúng thì có. Hãy chuẩn hóa tên tiêu đề trước khi tra cứu và từ chối các tiêu đề trùng lặp xung đột nhau. Đối với một `Mcp-Name` không an toàn, chứa ký tự phi ASCII, hoặc có khoảng trắng ở đầu/cuối, hãy giải mã chính xác chuỗi lính canh UTF-8 `=?base64?{Base64EncodedValue}?=` trước khi so sánh với thân thông điệp. Từ chối chuỗi lính canh không đầy đủ, Base64 không hợp lệ, UTF-8 lỗi hoặc giá trị thô không an toàn bằng mã lỗi `-32020`. Khoảng trắng thô xung quanh là không hợp lệ ngay cả khi thân thông điệp chứa các ký tự y hệt, bởi vì giá trị đó bắt buộc phải được mã hóa dạng lính canh trước khi truyền tải trên mạng.

Một mắt xích trung gian có thể từ chối yêu cầu HTTP sai cấu trúc trước khi nó đến được máy chủ MCP, vì vậy thất bại của nó có thể là lỗi HTTP thuần túy không chứa JSON-RPC. Hãy ghi nhận rõ việc từ chối xuất phát từ trung gian hay từ máy chủ gốc. Máy chủ MCP gốc phải luôn tuân thủ giao ước lỗi giao thức khi đã xử lý một yêu cầu JSON-RPC hợp lệ.

## Các Trường Chưa Biết Không Phải là Kết Quả Chưa Biết

Khả năng tương thích về sau (forward compatibility) đòi hỏi hai quy tắc ứng xử khác biệt.

### Các Trường Bổ Sung Chưa Xác Định (Additive unknown fields)

Các đối tượng kết quả và bản đồ `_meta` có thể được bổ sung thêm trường mới theo thời gian. Bộ xác thực nên bảo toàn hoặc bỏ qua trường bổ sung tùy theo vai trò của nó, trừ phi trường đó vi phạm một giao ước đã được bảo lưu. Đoạn mã mẫu giữ nguyên kết quả thô đầy đủ trong bằng chứng và chấp nhận trường `futureHint` nằm cạnh một kết quả đã biết.

Nếu bạn đóng vai trò là một proxy trong suốt (transparent proxy), việc bảo toàn trường chưa biết thường an toàn hơn là tước bỏ nó. Nếu bạn là một client ứng dụng, việc bỏ qua nó có thể hoàn toàn hợp lệ. Dù vậy, bài kiểm thử sai biệt (differential test) vẫn phải vạch rõ việc SDK đã loại bỏ trường này, nhằm đảm bảo đó là hành vi có chủ đích.

### `resultType` Chưa Xác Định (Unknown `resultType`)

`resultType` đóng vai trò là một bộ phân biệt (discriminator). Các kết quả cốt lõi hiện đại sử dụng `complete` hoặc `input_required`. Một phần mở rộng chỉ có thể bổ sung giá trị khác khi năng lực của nó đã được quảng bá. Ví dụ: phần mở rộng Tasks có thể bổ sung `task` trong ngữ cảnh năng lực đã đàm phán đó.

Một bộ phân biệt chưa xác định hoặc chưa được quảng bá không thể được xem như là `complete` một cách an toàn. Phía client không thể biết được vòng đời xử lý nào sẽ bị hủy hoại nếu làm vậy. Hãy từ chối nó.

Do đó, cùng một phản hồi thô có thể chứa một trường bổ sung được chấp nhận nhưng lại mang một kiểu kết quả không thể chấp nhận. Hãy kiểm thử cẩn trọng cả hai trường hợp.

Bộ phân biệt mới chỉ là tầng đầu tiên. Sau đó, hãy xác thực tải trọng dữ liệu (payload) đặc thù của từng phương thức. Kết quả `tools/list` dạng `complete` cần một mảng `tools` chứa các bộ mô tả có tên duy nhất không rỗng, mô tả hữu ích và giá trị `inputSchema` có gốc là đối tượng. Kết quả `task` chỉ hợp lệ cho một lời gọi `tools/call` đủ điều kiện với năng lực Tasks và đòi hỏi `taskId`, trạng thái hợp lệ, dấu thời gian tạo và cập nhật, `ttlMs`, cùng khoảng thời gian thăm dò tùy chọn hợp lệ. Kết quả `completion/complete` hoàn chỉnh đòi hỏi một đối tượng `completion` chứa không quá 100 giá trị chuỗi, trường `total` kiểu số nguyên không âm tùy chọn không nhỏ hơn số giá trị trả về, và trường Boolean `hasMore` tùy chọn. Một `resultType` viết đúng chính tả cũng không thể cứu vãn một tải trọng bị sai cấu trúc.

## Bất Biến Về Thông Báo (The Notification Invariant)

Một thông báo JSON-RPC không có trường `id`. Phía nhận tuyệt đối không được gửi lại bất kỳ phản hồi thành công hay báo lỗi JSON-RPC nào.

Đối với định dạng thông báo HTTP được chấp nhận, khung kiểm thử kỳ vọng mã HTTP `202` với thân rỗng. MCP `2026-07-28` không định nghĩa bất kỳ thông báo cốt lõi nào từ client đến server qua Streamable HTTP. Mã mẫu chỉ sử dụng một thông báo mở rộng có không gian tên phục vụ khóa học để kiểm tra bất biến một chiều của bộ tuần tự hóa. Đừng hiểu nhầm nó là một phương thức cốt lõi mới.

Hãy kiểm thử chính bộ tuần tự hóa chứ không chỉ kiểm thử hàm xử lý logic (handler). Một hàm xử lý có thể trả về `None`, nhưng một middleware nào đó lại tự tiện bọc nó vào một đối tượng JSON thành công. Luôn ghi lại các byte dữ liệu cuối cùng tại đầu ra (egress).

## Thêm Bước So Sánh Sai Biệt Với SDK (SDK Differential)

Các SDK thường chuyển đổi các đối tượng đường truyền thành các kiểu dữ liệu tiện lợi của ngôn ngữ lập trình. Điều đó rất thuận tiện, nhưng một đối tượng đã chuẩn hóa không thể chứng minh chính xác những gì đã được nhận trên đường truyền.

Đối với mỗi bản mẫu có rủi ro cao, hãy ghi lại:

1. Trạng thái thô, các tiêu đề và thân phản hồi trước khi SDK giải mã.
2. Giá trị trả về hoặc ngoại lệ đã được SDK chuẩn hóa.
3. Phép chiếu ngữ nghĩa kỳ vọng (expected semantic projection) cho thời kỳ đã chọn.
4. Các trường bị SDK nâng cấp, tự tổng hợp, tước bỏ hoặc làm thay đổi.

Đoạn mã mẫu cho phép SDK chỉ loại bỏ các trường quản lý đường truyền nội bộ như `resultType`, `_meta`, `ttlMs` và `cacheScope` trong khi so sánh tải trọng ứng dụng. Nó sẽ cảnh báo nếu trường `futureHint` bị bỏ rơi, bởi vì một trường ngữ nghĩa chưa biết đã hoàn toàn biến mất.

Đừng vội quy chụp mọi sự khác biệt đều là lỗi của SDK. Mục tiêu ở đây là làm cho sự chuyển đổi trở nên minh bạch và có thể giám sát. Hãy xác định rõ thành phần của bạn là một endpoint ứng dụng (được phép bỏ qua trường bổ sung) hay một mắt xích trung gian trong suốt (bắt buộc phải bảo toàn nó).

Chạy quy trình kiểm tra sai biệt này trên mọi SDK và phiên bản bạn phát hành. Nếu hai SDK chuẩn hóa cùng một bản ghi theo cách khác nhau, chính sách phát hành phải nêu rõ hành vi nào được chấp nhận thay vì chỉ đơn giản chọn đầu ra tiện nhất sau khi sự việc đã rồi.

## Thu Thập Bằng Chứng Proxy

Phần lớn các lỗi MCP trong môi trường production diễn ra xuyên suốt nhiều tiến trình. Hãy ghi lại ba khung nhìn:

| Khung nhìn | Bằng chứng tối thiểu |
|---|---|
| Đầu vào (Ingress) | Tiêu đề yêu cầu, thân JSON-RPC, content type, tuyến đường đã xác thực, thời gian nhận |
| Máy chủ gốc (Origin) | Tiêu đề chuyển tiếp và mã băm (digest) của thân, trạng thái gốc, tiêu đề và thân phản hồi |
| Đầu ra (Egress) | Trạng thái client nhìn thấy, tiêu đề, thân và thời gian gửi |

Mã mẫu phát hiện hai sự chuyển đổi sai lệch phổ biến:

- Lỗi JSON-RPC HTTP 400 hoặc 404 từ máy chủ gốc bị biến thành lỗi HTTP 500 chung chung của proxy.
- Thân thông điệp JSON-RPC ở đầu ra khác biệt so với thân thông điệp từ máy chủ gốc.

Bổ sung thêm các kiểm tra khẳng định (assertions) đặc thù cho việc triển khai đối với content type, `Accept`, nén dữ liệu, SSE phạm vi theo yêu cầu, tiêu đề bộ nhớ đệm và liên kết vết (trace correlation). Hãy ghi lại cả hai phía của điểm kết thúc TLS (TLS termination) khi chính sách bảo mật cho phép. Tuyệt đối không ghi nhật ký các thông tin xác thực (credentials) chỉ để chứng minh đường truyền hoạt động.

## Làm Sạch Dữ Liệu Trước Khi Rời Khỏi Bộ Nhớ

Làm sạch dữ liệu nhạy cảm (redaction) là một phần của quy trình vận hành hợp chuẩn, không phải là công việc dọn dẹp làm sau. Hãy áp dụng nó trước khi tuần tự hóa, băm dữ liệu, ghi nhật ký, tạo artifact kiểm thử hoặc tải lên báo cáo lỗi.

Mã mẫu chuyển tên khóa thành chữ thường và loại bỏ các dấu phân tách trước khi so khớp, sau đó đệ quy thay thế các giá trị dưới những khóa như `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`, `accessToken`, `clientSecret`, `registrationAccessToken`, `token`, `password`, `secret` và `api_key`. Quá trình chuẩn hóa và danh sách từ chối (denylist) phải dùng chung một dạng thức để các biến thể camelCase, dấu gạch ngang, dấu gạch dưới hay dấu chấm không thể qua mặt chính sách kiểm duyệt. Một hệ thống thu thập trong production cần bổ sung chính sách đối số đặc thù cho từng phương thức, vì một khóa tưởng chừng vô hại như `query` vẫn có thể chứa thông tin cá nhân hoặc dữ liệu bị quản lý.

Băm gói bằng chứng đã được làm sạch. Chỉ lưu trữ các bản ghi thô trên một hệ thống tồn tại ngắn hạn được phê duyệt khi có một cuộc điều tra sự cố cụ thể yêu cầu. Mã băm chứng minh gói dữ liệu đã làm sạch nào đã dẫn đến quyết định phát hành; nó hoàn toàn không làm lộ giá trị bí mật đã bị xóa bỏ.

## Đưa Sức Khỏe và Khả Năng Hoàn Nguyên Vào Cổng Phê Duyệt

Tính hợp chuẩn giao thức là điều kiện cần nhưng chưa đủ để phát hành. Một ứng viên hợp chuẩn vẫn có thể bị hết thời gian chờ, rò rỉ bộ nhớ hoặc gây quá tải cho các dịch vụ phụ thuộc.

Hãy xác định một cửa sổ giám sát sức khỏe (health window) trước khi triển khai:

- Số lượng mẫu tối thiểu.
- Tỷ lệ lỗi tối đa cho phép.
- Phân vị độ trễ (latency percentile) tối đa.
- Ngưỡng bão hòa hoặc giới hạn tài nguyên.
- Thời lượng quan sát.
- So sánh đối chiếu với đường cơ sở (baseline) đã được phê chuẩn tiếp nhận.

Hãy xác định rõ bằng chứng hoàn nguyên trước khi triển khai:

- Phiên bản trước đó chính xác.
- Mã băm bằng chứng tiếp nhận (admission evidence digest).
- Ghim khóa định danh SHA-256 cho artifact và bộ mô tả (descriptor).
- Trạng thái Registry hiện tại.
- Kết quả kiểm tra sức khỏe hiện tại.
- Quy trình khôi phục tuyến đường mạng.
- Một chứng thực (attestation) trên chính xác các trường đó từ một định danh kiểm soát phát hành (release-controller) đáng tin cậy.

Yêu cầu mục tiêu hoàn nguyên phải được xác minh và ở trạng thái khỏe mạnh trước khi phê duyệt thăng cấp phiên bản mới, chứ không phải chờ đến khi ứng viên thất bại mới cuống cuồng kiểm tra. Một đợt phát hành thành công mà không có lộ trình phục hồi khả thi thì chưa sẵn sàng cho production.

Nếu một ứng viên thất bại mà mục tiêu hoàn nguyên lại thiếu bằng chứng đó, hãy dừng lưu lượng thay vì phỏng đoán. "Hoàn nguyên về bất cứ thứ gì từng ở đó" không phải là một biện pháp kiểm soát vận hành.

Đừng hạ thấp tiêu chuẩn sẵn sàng thành các phép kiểm tra chân trị sơ sài như một chuỗi phiên bản không rỗng, `healthy: "yes"`, hay một chuỗi bằng chứng tùy ý. Mã mẫu yêu cầu các kiểu dữ liệu chính xác, trạng thái hoạt động tích cực (active), ba mã băm SHA-256, một bên ký tin cậy, và một chứng thực HMAC-SHA-256 hợp lệ bao trùm toàn bộ tải trọng hoàn nguyên. Khóa mẫu mang tính tất định chỉ là một fixture thử nghiệm không bảo mật. Trong production, hãy tiêm một khóa được bảo vệ, kết quả xác minh từ KMS, hoặc một bộ xác thực chứng thực bằng khóa công khai tại ranh giới phát hành.

Cổng phê duyệt phát hành cũng kiên quyết từ chối các bằng chứng bản ghi, sai biệt SDK hoặc proxy bị rỗng. Mỗi nguồn dữ liệu phải mang mã băm bằng chứng hợp lệ. Một cửa sổ sức khỏe xanh rì cũng không thể khỏa lấp cho một ranh giới chưa từng được kiểm tra trên thực tế.

## Xây dựng Mã Nguồn (Build It)

Chạy khung kiểm thử chỉ sử dụng thư viện chuẩn:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

Bản demo thực thi chính xác 15 bản ghi vàng và phủ định, bao gồm cả các kết quả completion hợp lệ và sai cấu trúc, đối chiếu kết quả thô với khung nhìn SDK, kiểm tra một proxy đã làm sụp đổ mã lỗi gốc, đánh giá sức khỏe hệ thống, xác thực bằng chứng hoàn nguyên và chọn mục tiêu đó.

Dạng đầu ra kỳ vọng:

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

Hãy đọc `code/main.py` theo trình tự sau:

1. `validate_request()` áp dụng các quy tắc yêu cầu và tiêu đề đặc thù theo từng thời kỳ.
2. `validate_result()` phân định các bộ phân biệt phiên bản cũ bị thiếu, các giá trị hiện đại hợp lệ, các phần mở rộng và các giá trị chưa biết.
3. `select_era()` triển khai chính sách nghiêm ngặt và chính sách dự phòng có giới hạn.
4. `run_transcript()` đánh giá các bản mẫu vàng và phủ định.
5. `compare_sdk_view()` vạch rõ các sai lệch trong quá trình chuẩn hóa dữ liệu.
6. `inspect_proxy()` so sánh bằng chứng giữa đầu vào (ingress), máy chủ gốc (origin) và đầu ra (egress).
7. `redact()` loại bỏ các bí mật rõ ràng trước khi băm bằng chứng.
8. `rollback_evidence_ready()` xác thực các trường ghim chính xác và chứng thực phát hành tin cậy.
9. `ReleaseGate.evaluate()` kết hợp các bằng chứng hợp chuẩn, SDK, proxy, sức khỏe và hoàn nguyên không rỗng.

## Sử dụng Mã Nguồn (Use It)

Chạy khung kiểm thử tại 4 điểm nút trọng yếu:

1. Trên mọi thay đổi mã triển khai với một test adapter chạy nội tiến trình.
2. Kiểm tra trực tiếp trên các tệp nhị phân client và server đã biên dịch thông qua giao vận thực tế.
3. Chạy xuyên qua proxy hoặc gateway đã triển khai trong môi trường staging.
4. Trong suốt quá trình triển khai canary với dữ liệu sức khỏe thời gian thực và bằng chứng hoàn nguyên sẵn sàng.

Duy trì các tên ca kiểm thử ổn định đồng nhất xuyên suốt các tầng. `negative-header-body-mismatch` phải biểu thị cùng một bất biến trong các báo cáo unit test, end-to-end, proxy và canary. Mã băm bằng chứng sẽ khác nhau do ranh giới quan sát thay đổi; nhưng yêu cầu bất biến thì không được phép thay đổi.

Lưu trữ các lược đồ bản mẫu trong hệ thống quản lý mã nguồn (Git). Lưu trữ bằng chứng chạy kiểm thử đã làm sạch trong hệ thống phát hành của bạn. Chỉ lưu trữ các bản ghi thô ngắn hạn dưới sự kiểm soát truy cập nghiêm ngặt của quy trình xử lý sự cố.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: Chứng minh ranh giới thời kỳ

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

Thực thi:

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

Lời gọi phiên bản cũ sẽ suy luận ra trạng thái `complete`. Lời gọi hiện đại sẽ ném ra lỗi `ProtocolViolation`. Giờ hãy kiểm tra cơ chế dự phòng:

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

Lần timeout đầu tiên thất bại khép kín (fails closed) vì sự im lặng không phải là bằng chứng của phiên bản cũ. Lời gọi thứ hai chọn phiên bản cũ chỉ vì cấu hình cho phép và đã quan sát thấy kết quả khởi tạo cũ hợp lệ. Mã lỗi thiếu năng lực đã được định danh sẽ chứng minh nhánh hiện đại.

### Lab B: Trường bổ sung đối đầu với bộ phân biệt

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

Kết quả đầu tiên bảo toàn `futureHint`. Kết quả thứ hai bị từ chối vì bộ phân biệt vòng đời không xác định.

### Lab C: Kiểm tra sự chuyển đổi của SDK

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

Hãy quyết định xem thành phần của bạn được phép bỏ qua `futureHint` hay bắt buộc phải chuyển tiếp nó. Đưa lựa chọn đó vào chính sách phát hành. Đừng âm thầm xóa bỏ sự sai biệt này.

### Lab D: Khắc phục sự cố proxy

Sửa đổi luồng trao đổi trong bản demo để đầu ra bảo toàn trạng thái và thân thông điệp từ máy chủ gốc. Chạy lại lệnh `python3 main.py`. Các vấn đề về proxy sẽ biến mất, nhưng sự sai biệt từ SDK vẫn ngăn cản việc thăng cấp. Sau đó hãy đưa `futureHint` vào khung nhìn SDK và quan sát hành động chuyển thành `promote` khi mọi nguồn bằng chứng đều vượt qua kiểm định.

## Bài Tập Thực Hành (Practice Lab)

Bổ sung các bản ghi SSE theo phạm vi yêu cầu (request-scoped SSE) vào khung kiểm thử.

Các yêu cầu:

- Ghi lại mã trạng thái phản hồi, content type, các sự kiện SSE theo thứ tự, và sự chấm dứt luồng stream.
- Chứng minh mỗi sự kiện JSON-RPC đều có kết quả hoặc lỗi hợp lệ theo đúng thời kỳ.
- Bổ sung ca kiểm thử phủ định cho một proxy đệm (buffer) toàn bộ luồng stream trước khi chuyển tiếp.
- Bổ sung ca kiểm thử phủ định cho một sự kiện SSE có ID JSON-RPC khác với yêu cầu ban đầu.
- Làm sạch dữ liệu sự kiện trước khi ghi nhận bằng chứng.
- Đưa thời lượng truyền phát, độ trễ sự kiện đầu tiên và số lượng sự kiện vào cửa sổ giám sát sức khỏe.
- Đảm bảo cổng phát hành chỉ chọn một mục tiêu hoàn nguyên có bằng chứng rõ ràng khi luồng stream thất bại.

Thành công đồng nghĩa với việc cùng một ca kiểm thử có thể chạy trực tiếp và chạy qua proxy, với một báo cáo xác định chính xác ranh giới nào đã làm thay đổi hành vi.

## Sản Phẩm Đóng Gói (Shipped Artifact)

Bài học này đóng gói tệp `outputs/skill-mcp-conformance-release-gate.md`. Hãy sử dụng nó để chuyển đổi bất kỳ thay đổi nào của máy chủ, client, gateway hoặc SDK thành một ma trận hợp chuẩn có phiên bản và đưa ra quyết định phát hành chính xác. Sản phẩm yêu cầu bằng chứng đường truyền thô, các ca kiểm thử phủ định, lựa chọn thời kỳ tường minh, sai biệt SDK, bằng chứng proxy, làm sạch dữ liệu, các ngưỡng sức khỏe và bằng chứng hoàn nguyên đầy đủ.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình xác minh cần chứng minh được:

- Mọi bản ghi vàng và phủ định đi kèm đều đạt kết quả kỳ vọng tương ứng.
- Các yêu cầu hiện đại bắt buộc phải có các khóa siêu dữ liệu có không gian tên chính xác.
- Tên tiêu đề HTTP được so khớp không phân biệt chữ hoa chữ thường và các giá trị `Mcp-Name` được mã hóa sẽ được giải mã chính xác.
- Sự bất đồng giữa tiêu đề và thân thông điệp sẽ trả về mã lỗi không khớp hiện đại.
- Phiên bản phản hồi, ID, tính loại trừ lẫn nhau giữa kết quả hoặc lỗi, cấu trúc lỗi và ánh xạ HTTP đều được xác thực.
- Các yêu cầu tải trọng đặc thù của phương thức danh sách công cụ, tác vụ và hoàn thành văn bản đều được thực thi nghiêm ngặt.
- Mọi quan sát thấy `HeaderMismatch` đều đòi hỏi phản hồi JSON-RPC `-32020` với HTTP 400 thực tế.
- Khoảng trắng thô trong `Mcp-Name` bị từ chối trong khi khoảng trắng được mã hóa chuẩn lính canh sẽ được bảo toàn nguyên vẹn hai chiều.
- Trường `resultType` bị thiếu chỉ hợp lệ trong thời kỳ cũ đã được chọn.
- Các trường bổ sung vượt qua xác thực thô trong khi các kiểu kết quả không xác định sẽ thất bại.
- Các kiểu kết quả mở rộng đòi hỏi năng lực tương ứng phải được quảng bá trước đó.
- Các lỗi hiện đại đã được định danh không bao giờ kích hoạt cơ chế dự phòng về chuẩn cũ.
- Các thông báo không sản sinh ra bất kỳ phản hồi JSON-RPC nào.
- Phân biệt rõ việc SDK loại bỏ trường quản lý nội bộ với việc làm mất trường ngữ nghĩa thực tế.
- Phát hiện sự sụp đổ lỗi của proxy và làm sạch thông tin xác thực đệ quy trên các biến thể camelCase và các ký tự phân tách.
- Việc thăng cấp phiên bản đòi hỏi bằng chứng bản ghi, SDK, proxy và vận hành thực tế không được rỗng.
- Cả thăng cấp và hoàn nguyên đều đòi hỏi một mục tiêu hoàn nguyên đã được chứng thực, được ghim mã băm, đang hoạt động và khỏe mạnh.

## Các Chế Độ Thất Bại Trong Môi Trường Production

| Thất bại | Bài kiểm thử yếu kém báo cáo | Điều khung kiểm thử phải chứng minh |
|---|---|---|
| SDK tự tổng hợp bộ phân biệt bị thiếu | "tools/list đã vượt qua" | Kết quả hiện đại thô thiếu `resultType` và không hợp lệ |
| Client hạ cấp giao thức sau lỗi `-32021` | "thử lại kiểu cũ thành công" | Lỗi hiện đại đã nhận diện nghiêm cấm hạ cấp dự phòng |
| Kiểu kết quả chưa biết bị coi là hoàn tất | "phân tích phản hồi thành công" | Bộ phân biệt vòng đời chưa được quảng bá bị từ chối dứt khoát |
| Proxy ủy quyền cho một công cụ nhưng máy chủ thực thi công cụ khác | "yêu cầu đã tới máy chủ" | `Mcp-Name` trùng khớp với tên định tuyến trong thân ở mọi bước nhảy (hop) |
| Khung kiểm thử ném ngoại lệ trước khi đọc phản hồi máy chủ | "kiểm thử không khớp tiêu đề đạt" | Đã bắt và xác thực phản hồi HTTP 400 cùng mã JSON-RPC `-32020` |
| Proxy biến lỗi 400 của máy chủ gốc thành lỗi 500 chung | "lỗi phía thượng nguồn (upstream)" | Trạng thái gốc và trạng thái đầu ra cùng thân JSON-RPC được bảo toàn |
| Middleware thông báo phát ra `{result: null}` | "hàm xử lý trả về none" | Thân thông điệp đầu ra cuối cùng rỗng và không có phản hồi JSON-RPC |
| SDK tước bỏ một trường bổ sung mới | "đối tượng định kiểu trùng khớp" | Khung nhìn thô và chuẩn hóa vạch trần chính xác trường bị rơi |
| Artifact lỗi làm lộ mã token mang quyền truy cập (bearer token) | "đã tải gói gỡ lỗi lên" | Làm sạch dữ liệu đã diễn ra trước khi băm, ghi log hoặc tải lên |
| Kiểu định dạng khóa thông tin xác thực qua mặt bộ lọc | "denylist đã có api_key" | Các biến thể camelCase và dấu phân tách cùng chia sẻ một dạng thức denylist chuẩn |
| Triển khai Canary không có mẫu nào nhưng vẫn báo khỏe mạnh | "không có lỗi nào" | Số lượng mẫu tối thiểu được thực thi nghiêm ngặt |
| Hoàn nguyên chọn phải một bản build không xác định | "đã khôi phục triển khai trước" | Phiên bản mục tiêu, mã băm tiếp nhận, ghim bảo mật, trạng thái và sức khỏe đều hiện diện |

## Quy Tắc Vận Hành Cốt Lõi

Hãy kiểm thử từng byte bạn gửi đi, từng byte mọi mắt xích trung gian chuyển tiếp, ngữ nghĩa mà mỗi SDK phơi bày, và bằng chứng mà đội ngũ vận hành sẽ dựa vào dưới áp lực sự cố. Tính tương thích là một nhánh xử lý tường minh. Hoàn nguyên là một hành động phát hành được bảo đảm bằng bằng chứng. Cả hai điều này tuyệt đối không được là tác dụng phụ ngẫu nhiên của một bộ phân tích cú pháp lỏng lẻo.

## Đọc Thêm

- [Giao thức cơ sở MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Đàm phán phiên bản MCP](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [Giao vận MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Dự án kiểm định hợp chuẩn MCP chính thức](https://github.com/modelcontextprotocol/conformance)
