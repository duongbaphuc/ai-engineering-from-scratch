# Xây dựng Máy khách MCP: Khám phá, Điều hướng và Cơ chế Dự phòng Đa Thế hệ

> Một máy khách MCP hiện đại lặp lại hợp đồng của nó trên từng yêu cầu. Quyết định tương thích khó khăn nhất của nó là nhận biết khi nào một máy chủ cũ thực sự là máy chủ cũ, và khi nào một máy chủ hiện đại đang báo cáo một lỗi có thể khắc phục được.

**Type:** Xây dựng (Build)
**Languages:** Python
**Prerequisites:** Phase 13, Bài học 07
**Time:** ~85 phút

## Mục tiêu học tập

- Xây dựng mọi yêu cầu MCP `2026-07-28` với siêu dữ liệu hiện thời.
- Thăm dò các máy chủ stdio bằng `server/discover` và chọn ra phiên bản được cả hai bên cùng hỗ trợ.
- Cho phép thăm dò kế thừa có giới hạn chỉ dành riêng cho các máy chủ ngang hàng (peers) nằm trong danh sách cho phép rõ ràng.
- Chỉ chấp nhận thế hệ giao thức kế thừa (legacy era) sau khi đã xác thực kết quả `initialize` thành công cho một phiên bản được hỗ trợ.
- Hợp nhất các danh sách công cụ tất định mà không ghi đè âm thầm khi xảy ra xung đột tên.
- Điều hướng các lệnh gọi đến đúng máy chủ ngang hàng sở hữu từng công cụ mà không tự sinh ra các phiên giao thức nhân tạo.

## Vấn đề

Một ứng dụng chủ tác tử (agent host) thường giao tiếp với nhiều hơn một máy chủ MCP. Nó phải khám phá từng máy chủ, hợp nhất danh mục công cụ, giải quyết các tên trùng lặp, điều hướng các lệnh gọi và phục hồi sau sự cố phương thức truyền tải (transport failure).

Phiên bản sửa đổi `2026-07-28` giúp trạng thái ổn định trở nên đơn giản hơn vì mỗi yêu cầu đều độc lập khép kín (self-contained). Tuy nhiên, khả năng tương thích khiến giai đoạn khởi động trở nên tinh tế và phức tạp hơn. Một máy khách có thể gặp phải:

- một máy chủ hiện đại hỗ trợ phiên bản ưu tiên;
- một máy chủ hiện đại trả về lỗi phiên bản hoặc tiêu đề được nhận diện;
- một máy chủ cũ chưa từng biết đến `server/discover`;
- một máy chủ cũ giữ im lặng tuyệt đối cho đến khi nó nhận được phương thức `initialize`.

Xem mọi lỗi thăm dò đều là dấu hiệu của máy chủ cũ là một cách tiếp cận nguy hiểm. Một yêu cầu hiện đại bị sai định dạng, một máy chủ bị quá tải, một tiến trình đã chết và một máy chủ cũ đều có thể gây ra cùng một triệu chứng quá thời gian chờ (timeout) hoặc đóng kết nối. Những tín hiệu đó hoàn toàn mơ hồ. Máy khách phải kết hợp chủ đích tường minh của người vận hành với bằng chứng giao thức xác thực trước khi quyết định chọn thế hệ kế thừa.

## Khái niệm

### Một máy chủ ngang hàng, không phải một phiên giao thức

Duy trì một bản ghi máy chủ ngang hàng truyền tải (transport peer record) cho mỗi tiến trình hoặc điểm cuối máy chủ:

- thẻ điều khiển truyền tải (transport handle) hoặc hàm gửi;
- thế hệ và phiên bản giao thức được chọn;
- các khả năng của máy chủ được khám phá lần gần nhất;
- danh sách công cụ tất định gần nhất;
- các id yêu cầu đang chờ xử lý để đối chiếu tương quan (correlation);
- trạng thái sức khỏe của phương thức truyền tải.

Đây thuần túy là việc theo dõi nội bộ của máy khách. Nó không phải là trạng thái phiên giao thức. Trên chuẩn MCP hiện đại, máy chủ vẫn nhận được phiên bản và các khả năng hiện thời trên từng yêu cầu.

### Xây dựng mọi yêu cầu hiện đại từ đầu

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

Không gắn siêu dữ liệu một lần duy nhất vào một đối tượng kết nối rồi mặc định rằng nó sẽ luôn tới được đường truyền. Hãy đóng dấu và kiểm tra chính yêu cầu đã tuần tự hóa cuối cùng.

### Khám phá hiện đại

`server/discover` trả về các phiên bản được hỗ trợ, khả năng của máy chủ, hướng dẫn, gợi ý bộ đệm và danh tính máy chủ được khuyến nghị. Máy khách sẽ chọn phiên bản hiện đại cao nhất mà cả hai bên cùng hỗ trợ.

Khám phá là tùy chọn đối với một máy khách chỉ chạy thuần hiện đại, nhưng nó được khuyến nghị mạnh mẽ trên stdio. Một số máy chủ cũ chấp nhận một thao tác trước khi khởi tạo, vì vậy việc gửi `tools/list` trước có thể mang lại kết quả thành công mơ hồ. `server/discover` thiết lập một ranh giới thế hệ rành mạch.

### Thăm dò tương thích stdio

Một máy khách stdio hỗ trợ đa thế hệ sẽ gửi `server/discover` cùng với siêu dữ liệu hiện đại ưu tiên trước bất kỳ yêu cầu nào khác. Có ba nhóm kết quả:

1. **DiscoverResult.** Máy chủ là hiện đại. Chọn một phiên bản được cả hai hỗ trợ và tiếp tục với siêu dữ liệu trên từng yêu cầu.
2. **Lỗi hiện đại được nhận diện (Recognized modern error).** Máy chủ là hiện đại. Đối với mã `-32022`, hãy chọn từ `data.supported` và thử lại với một id yêu cầu mới. Đối với các lỗi tiêu đề hoặc khả năng, hãy sửa lại yêu cầu. Tuyệt đối không gửi `initialize`.
3. **Tín hiệu mơ hồ (Ambiguous signal).** Một lỗi JSON-RPC không nhận diện được, lỗi quá thời gian chờ, đóng kết nối hoặc phản hồi rỗng đều không thể xác định được thế hệ giao thức. Mặc định xử lý thất bại an toàn (fail closed) trừ khi chính máy chủ ngang hàng đó được cấu hình tường minh để tương thích kế thừa.

Các lỗi giao thức hiện đại được nhận diện bao gồm:

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

Các lỗi hiện đại được nhận diện vẫn khẳng định máy chủ thuộc thế hệ hiện đại ngay cả khi máy chủ đó nằm trong danh sách cho phép kế thừa. Khi máy chủ đã chứng minh rằng nó hiểu từ vựng lỗi hiện đại, việc gửi `initialize` sẽ là một hành vi hạ cấp (downgrade) sai lầm.

Không coi mã `-32601` là bằng chứng xác thực của thế hệ kế thừa. Nó chỉ đơn thuần giúp một máy chủ nằm trong danh sách cho phép đủ điều kiện để thực hiện một lần thăm dò kế thừa duy nhất. Quy tắc tương tự cũng áp dụng cho lỗi quá thời gian chờ, đóng kết nối hoặc phản hồi rỗng.

### Danh sách cho phép là chủ đích của người vận hành, không phải bằng chứng

Khả năng tương thích kế thừa phải là một thuộc tính tường minh của cấu hình máy chủ ngang hàng được chỉ định:

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

Gắn ràng buộc lựa chọn đó với lệnh hoặc điểm cuối đã cấu hình. Không dùng ký tự đại diện (wildcard) để tránh việc một máy chủ bất kỳ tự ý đưa mình vào ngữ nghĩa lỏng lẻo hơn. Một máy chủ ngang hàng không có `allow_legacy=True` sẽ lập tức báo lỗi sau một kết quả khám phá mơ hồ và không bao giờ nhận được `initialize`.

Danh sách cho phép chỉ cấp quyền thăm dò. Nó không tự động lựa chọn thế hệ giao thức. Máy khách gửi một yêu cầu `initialize` duy nhất dưới một giới hạn thời gian (deadline) do phương thức truyền tải thực thi, sau đó yêu cầu thỏa mãn toàn bộ các điều kiện sau:

- một phản hồi JSON-RPC `2.0` có id yêu cầu khớp nhau;
- chính xác một trường `result` và không có trường `error`;
- một trường `protocolVersion` nằm trong tập hợp các bản sửa đổi kế thừa được máy khách cấu hình hỗ trợ;
- một trường `capabilities` có giá trị là một đối tượng;
- một đối tượng `serverInfo` với các trường `name` và `version` là các chuỗi ký tự không rỗng.

Quá thời gian chờ, đóng kết nối, phản hồi lỗi, kết quả sai định dạng, id không khớp hoặc phiên bản không được hỗ trợ đều dẫn đến việc ngắt kết nối an toàn (fail closed). Chỉ một kết quả thành công hợp lệ về mặt cấu trúc mới cho phép chọn thế hệ kế thừa. Mã nguồn truyền tham số `legacy_probe_timeout_ms` vào bộ điều hợp truyền tải; một bộ điều hợp stdio hoặc HTTP thực tế phải cưỡng chế giới hạn thời gian này thay vì chỉ ghi nhận nó đơn thuần.

Lưu trữ thế hệ đã chọn vào bộ nhớ đệm cho máy chủ ngang hàng truyền tải đó. Không thực hiện thăm dò lại trước mỗi lệnh gọi.

### Kế thừa là một nhánh tương thích

Một khi phép thăm dò có giới hạn trả về bằng chứng xác thực kế thừa hợp lệ, máy khách sẽ sử dụng phiên bản kế thừa đã chọn chính xác theo định nghĩa của bản sửa đổi đó:

1. Xác minh phong bì phản hồi và id đối chiếu tương quan.
2. Xác minh phiên bản đã đàm phán nằm trong tập hợp kế thừa được cấu hình.
3. Ghi lại các khả năng đã được xác thực và danh tính máy chủ.
4. Chỉ gửi thông báo `notifications/initialized` sau khi tất cả các bước kiểm tra đã vượt qua.
5. Sử dụng định dạng yêu cầu kế thừa trong suốt vòng đời của phương thức truyền tải đó.

Nhánh mã này tồn tại chỉ nhằm mục đích tương tác với các máy chủ đã biết từ trước. Nó không phải là thiết kế mặc định cho các máy chủ mới hoặc các yêu cầu mới. Nếu phương thức truyền tải khởi động lại hoặc điểm cuối thay đổi, hãy hủy bỏ bộ đệm thế hệ của máy chủ đó và tiến hành đàm phán lại từ đầu.

### Khám phá và lưu bộ đệm các công cụ

Đối với mỗi máy chủ ngang hàng đang hoạt động, gọi `tools/list`. Một kết quả hiện đại bao gồm `resultType`, `ttlMs` và `cacheScope`. Hãy tôn trọng gợi ý độ tươi mới (freshness hint) trong đúng ngữ cảnh ủy quyền. Tìm nạp lại danh sách sau khi hết hạn hoặc khi nhận được sự kiện thay đổi danh sách đã đăng ký.

Máy khách phải xử lý trường `resultType` bị thiếu từ một máy chủ cũ tương đương với giá trị `"complete"`. Không bắt buộc các trường bộ đệm hiện đại trên một phản hồi đến từ một thế hệ đã đàm phán trước đó.

Máy chủ cần trả về thứ tự mang tính tất định. Máy khách cũng nên sắp xếp lại trước khi hợp nhất để thứ tự thanh ghi cục bộ không bị phụ thuộc vào thời điểm khởi động của các tiến trình.

### Hợp nhất không gian tên an toàn trước xung đột

Hai máy chủ có thể cùng công bố một công cụ tên là `search`. Hãy chọn một chính sách đã tuyên bố trước:

1. **Thêm tiền tố khi xung đột (Prefix on collision).** Giữ tên chính tắc đầu tiên và công bố các công cụ xung đột sau đó dưới dạng `<server>/<tool>`.
2. **Từ chối khi xung đột (Reject on collision).** Không tải công cụ trùng lặp và báo lỗi cấu hình rõ ràng.
3. **Ghi đè âm thầm (Silent overwrite).** Tuyệt đối không bao giờ sử dụng chính sách này. Nó che giấu việc máy chủ nào sẽ thực sự nhận hành động do mô hình lựa chọn.

Lưu trữ cả tên chính tắc và tên cục bộ. Mô hình sẽ nhìn thấy tên chính tắc. Lệnh gọi `tools/call` gửi ra ngoài sẽ sử dụng tên cục bộ mà chính máy chủ sở hữu công cụ đã công bố.

### Điều hướng một lệnh gọi

Điều hướng là một thao tác tra cứu thuần túy:

```text
tên công cụ chính tắc
  -> tên máy chủ ngang hàng + tên công cụ cục bộ
  -> id yêu cầu JSON-RPC mới
  -> siêu dữ liệu yêu cầu hiện đại hoặc cấu trúc kế thừa rõ ràng
  -> id phản hồi trùng khớp
```

Không gửi lệnh gọi khi phương thức truyền tải sở hữu nó không khả dụng. Kết nối lại hoặc khởi động lại phương thức truyền tải, sau đó chạy lại quy trình khám phá và `tools/list`. Các yêu cầu hiện đại đang xử lý dở dang (in-flight) bị mất trên một đường truyền bị đứt có thể được thử lại với một id JSON-RPC mới khi chính sách an toàn của thao tác đó cho phép.

### Thông báo và đăng ký luồng

Các thay đổi về danh sách và tài nguyên hiện đại chỉ được gửi đến trên luồng `subscriptions/listen` do máy khách mở. Máy khách gửi bộ lọc thông báo, chờ thông báo `notifications/subscriptions/acknowledged`, và đối chiếu các sự kiện với id của yêu cầu lắng nghe trong siêu dữ liệu thông báo.

Khi mất kết nối, hãy mở một yêu cầu lắng nghe mới và tìm nạp lại các danh sách hoặc tài nguyên liên quan. Các luồng hiện đại không hỗ trợ tiếp tục lại bằng tiêu đề `Last-Event-ID`.

### Không có yêu cầu khởi tạo từ máy chủ

Các máy chủ hiện đại không gọi máy khách bằng các yêu cầu JSON-RPC độc lập để lấy mẫu (sampling), gợi mở phản hồi (elicitation) hoặc lấy thư mục gốc (roots). Chúng trả về `input_required`, và máy khách sẽ thử lại yêu cầu ban đầu sau khi đã đáp ứng các yêu cầu đầu vào được nhúng.

Không chặn trình đọc phản hồi của máy chủ ngang hàng trong khi đang thực hiện thu thập đầu vào. Duy trì thông tin đối chiếu tương quan và tạo một id JSON-RPC mới cho lần thử lại.

```figure
tp-client-merge
```

## Sử dụng

Tệp `code/main.py` sử dụng các hàm máy chủ nội tiến trình để các quyết định giao thức luôn hiển thị rõ ràng. Nó kết nối với hai máy chủ hiện đại và một máy chủ cũ được chủ động đưa vào danh sách cho phép, sau đó hợp nhất và điều hướng các công cụ của chúng. Hàm gọi truyền tải nhận một ngân sách thời gian chờ (timeout budget) để nhánh tương thích không thể che giấu một phép thăm dò vô hạn định.

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

Các bài kiểm thử chứng minh những ranh giới mà các bản thử nghiệm thông thường hay bỏ sót:

- các yêu cầu hiện đại lặp lại siêu dữ liệu;
- lỗi `-32022` kích hoạt thử lại khám phá hiện đại mà không thực hiện khởi tạo;
- các lỗi hiện đại được nhận diện không bao giờ bị hạ cấp, ngay cả đối với một máy chủ nằm trong danh sách cho phép;
- lỗi quá thời gian chờ, đóng kết nối, phản hồi rỗng và các lỗi không nhận diện được không kích hoạt `initialize` nếu không có danh sách cho phép;
- một máy chủ trong danh sách cho phép chỉ chuyển sang thế hệ kế thừa sau khi có kết quả `initialize` hợp lệ và được hỗ trợ;
- các kết quả kế thừa sai định dạng hoặc không được hỗ trợ sẽ khiến máy chủ giữ trạng thái không khả dụng;
- một thế hệ được chọn thành công sẽ được lưu đệm trong suốt vòng đời của phương thức truyền tải.

## Đóng gói / Triển khai

Bài học này cung cấp `outputs/skill-mcp-client-harness.md`. Nó tạo khung mẫu cho việc đóng dấu siêu dữ liệu yêu cầu hiện đại, đàm phán thế hệ stdio, hợp nhất không gian tên tất định, điều hướng lệnh gọi và một nhánh tương thích kế thừa ngắt an toàn khi có lỗi (fail-closed).

## Bài tập

1. Giả lập một máy chủ trả về mã `-32022` mà không có phiên bản nào được cả hai hỗ trợ. Xác nhận rằng máy khách báo lỗi thay vì gửi `initialize`.
2. Đưa một máy chủ cũ giả lập vào danh sách cho phép, làm cho phép thăm dò `initialize` có giới hạn của nó bị quá thời gian chờ, và chứng minh máy chủ đó vẫn giữ trạng thái `unknown` và không khả dụng.
3. Thêm danh sách công cụ với `cacheScope: "private"` cho hai ngữ cảnh ủy quyền khác nhau. Xác nhận rằng máy khách không bao giờ chia sẻ kết quả lưu đệm của ngữ cảnh này cho ngữ cảnh kia.
4. Đổi chính sách xung đột sang từ chối (rejection) và làm cho quá trình khởi động thất bại với tên của cả hai máy chủ hiển thị trong thông báo lỗi.
5. Thêm một bộ mô phỏng `subscriptions/listen` hữu hạn. Khi mất luồng truyền dữ liệu, thực hiện lắng nghe lại với một id yêu cầu mới và tìm nạp lại danh sách công cụ.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| Máy chủ ngang hàng (Peer) | Bản ghi phía máy khách đại diện cho một phương thức truyền tải máy chủ và dữ liệu khám phá của nó |
| Thế hệ giao thức (Protocol era) | Ngữ nghĩa siêu dữ liệu trên từng yêu cầu hiện đại hoặc ngữ nghĩa khởi tạo kế thừa |
| Phép thăm dò khám phá (Discovery probe) | Lệnh gọi `server/discover` ban đầu dùng để xác định thế hệ giao thức stdio |
| Lỗi hiện đại được nhận diện (Recognized modern error) | Lỗi chứng minh hành vi hiện đại và cấm việc quay lui về cơ chế kế thừa |
| Danh sách cho phép kế thừa (Legacy allowlist) | Cấu hình của người vận hành cho phép một phép thăm dò tương thích có giới hạn cho một máy chủ cố định |
| Bằng chứng kế thừa xác thực (Positive legacy evidence) | Kết quả `initialize` hợp lệ, đúng tương quan cho một bản sửa đổi kế thừa được hỗ trợ rõ ràng |
| Không gian tên hợp nhất (Merged namespace) | Tập hợp tên công cụ chính tắc trên toàn bộ các máy chủ đang hoạt động |
| Chính sách xung đột (Collision policy) | Quy tắc thêm tiền tố hoặc từ chối đối với các tên công cụ trùng lặp |
| Bộ nhớ đệm thế hệ (Era cache) | Hành vi hiện đại hoặc kế thừa đã chọn được lưu trữ cho một phương thức truyền tải máy chủ |
| Phục hồi truyền tải (Transport recovery) | Khởi động lại hoặc kết nối lại, khám phá lại, liệt kê lại và thử lại an toàn với một id mới |

## Đọc thêm

- [Đặc tả MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [Khám phá Máy chủ MCP (MCP Server Discovery)](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Phương thức truyền tải stdio trong MCP (MCP stdio Transport)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [Quản lý Phiên bản MCP (MCP Versioning)](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [Công cụ MCP (MCP Tools)](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
