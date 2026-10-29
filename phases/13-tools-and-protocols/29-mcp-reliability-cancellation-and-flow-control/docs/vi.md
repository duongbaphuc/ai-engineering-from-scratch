# Độ tin cậy, Hủy bỏ và Kiểm soát Luồng trong MCP (MCP Reliability, Cancellation, and Flow Control)

> Một ID yêu cầu chỉ giúp liên kết thông điệp. Nó không làm cho một tác dụng phụ trở nên an toàn, không thể dừng một tiến trình làm việc (worker), và cũng không thể bảo vệ một luồng dữ liệu trước một bên tiêu thụ chậm chạp.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Phase 13, Bài học 09 và 13  
**Thời lượng:** ~120 phút  

## Mục tiêu học tập

- Cài đặt tín hiệu hủy bỏ chuẩn xác cho stdio và Streamable HTTP.
- Giải quyết các cuộc đua tranh chấp (race conditions) giữa hoàn thành và hủy bỏ mà không gửi thông điệp thừa sau khi đã hủy.
- Tách biệt rạch ròi giữa việc hủy yêu cầu và ngữ nghĩa hủy tác vụ bền vững `tasks/cancel`.
- Đưa ra quyết định thử lại (retry) dựa trên tác dụng phụ (side effects) và các khóa lũy kế (idempotency keys) tường minh.
- Giới hạn các hàng đợi tiến trình (progress queues) trong khi vẫn bảo toàn các phản hồi cuối cùng.
- Phục hồi các luồng dữ liệu thông qua kết nối lại, nạp lại dữ liệu (refetch), và thuật toán lùi thời gian ngẫu nhiên (jittered backoff).

## Vấn đề

Nhánh thực thi thuận lợi (happy path) thường che giấu những lỗi hệ thống phân tán tốn kém nhất.

Một máy khách gọi một công cụ. Máy chủ bắt đầu thực hiện tác vụ. Tiến độ (progress) được gửi về. Một máy chủ ủy quyền (proxy) đệm luồng dữ liệu lại. Máy khách chạm ngưỡng hết thời gian chờ (timeout) và ngắt kết nối. Máy chủ hoàn thành công việc sau đó đúng một phần nghìn giây. Máy khách thử lại với một ID JSON-RPC hoàn toàn mới. Thao tác biến đổi dữ liệu (mutation) bị thực thi hai lần.

Mỗi thành phần đơn lẻ đều hoạt động đúng theo logic cục bộ của nó. Nhưng toàn bộ hệ thống lại thất bại trên quy mô tổng thể.

MCP định nghĩa hành vi của thông điệp và tầng truyền vận, nhưng ứng dụng của bạn vẫn phải tự chịu trách nhiệm về:

- ngân sách thời gian (time budgets);
- tính lũy kế nghiệp vụ (business idempotency);
- các hàng đợi có giới hạn (bounded queues);
- phân loại chiến lược thử lại;
- trạng thái tác vụ bền vững;
- chính sách kết nối lại và nạp lại dữ liệu.

Bài học này xây dựng các quyết định đó bên trong một bộ mô phỏng tất định. Không có hàm tạm dừng (`sleep`), không có kết nối socket mạng, và không có các lỗi ngẫu nhiên. Bạn sẽ kiểm soát trực tiếp thứ tự sự kiện hủy bỏ. Một bài kiểm thử luồng đồng bộ sẽ ép hai máy khách sổ cái cùng cạnh tranh trên cùng một khóa lũy kế.

## Hủy bỏ Yêu cầu Phụ thuộc vào Tầng Truyền vận

Mục đích là như nhau trên mọi tầng truyền vận: máy khách không còn cần kết quả đang được xử lý nữa. Nhưng tín hiệu gửi trên đường truyền lại khác nhau.

### stdio

stdio sử dụng một kênh hai chiều dùng chung duy nhất. Máy khách gửi một thông báo (notification):

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

Thông báo này thuộc dạng "bắn và quên" (fire-and-forget). Máy chủ không phát ra bất kỳ phản hồi JSON-RPC nào cho thông báo này.

Máy chủ nên dừng công việc, giải phóng tài nguyên, và tránh gửi phản hồi cho yêu cầu đã bị hủy. Máy chủ có thể bỏ qua việc hủy bỏ khi yêu cầu không tồn tại, đã hoàn thành từ trước, hoặc không thể dừng một cách an toàn.

Các thông báo hủy bỏ sai định dạng, không xác định, hoặc đã hoàn tất đều bị lờ đi. Việc biến những xung đột này thành lỗi mới sẽ chỉ tạo ra thêm nhiều cuộc đua tranh chấp khác.

### Streamable HTTP

Giao vận Streamable HTTP hiện đại cấp cho mỗi yêu cầu một phản hồi HTTP riêng hoặc một luồng phản hồi SSE riêng. Máy khách thực hiện hủy bỏ bằng cách đóng luồng phản hồi của chính yêu cầu đó.

Đừng gửi POST `notifications/cancelled` cho một yêu cầu HTTP thông thường. Việc đóng luồng chính là tín hiệu hủy bỏ.

Một khi máy chủ nhận thấy việc ngắt kết nối, nó nên dừng tác vụ và tuyệt đối không được gửi thêm bất kỳ thông điệp nào cho yêu cầu đó nữa.

### Hủy bỏ do máy chủ khởi tạo có phạm vi hẹp

Máy chủ không sử dụng `notifications/cancelled` để hủy các lượt gọi tùy ý của máy khách. Trên stdio, việc hủy bỏ do máy chủ gửi chỉ được dành riêng cho việc chấm dứt một yêu cầu `subscriptions/listen`. Hãy giữ luồng này tách biệt hoàn toàn khỏi việc hủy yêu cầu thông thường của máy khách.

## Hủy bỏ là một Cuộc đua Tranh chấp (Race Condition)

Cả hai thứ tự sự kiện sau đều hoàn toàn hợp lệ.

### Hủy bỏ thắng

```text
yêu cầu bắt đầu
máy khách gửi tín hiệu hủy bỏ
máy chủ đánh dấu yêu cầu đã bị hủy
tiến trình làm việc đạt trạng thái hoàn thành
máy chủ chặn không gửi phản hồi
```

### Hoàn thành thắng

```text
yêu cầu bắt đầu
tiến trình làm việc ghi nhận kết quả (commit)
máy chủ gửi phản hồi đi
thông báo hủy bỏ đến muộn
máy chủ bỏ qua thông báo muộn này
```

Máy khách cũng bắt buộc phải bỏ qua một phản hồi đến muộn cho một yêu cầu mà nó đã từ bỏ. Độ trễ mạng đồng nghĩa với việc không bên nào có thể chứng minh được bên kia đã quan sát thấy sự kiện nào trước.

````figure mcp-reliability-race
````

Bộ điều phối `RequestCoordinator` trong bài học lưu trữ một trạng thái kết thúc (terminal state) duy nhất. Hàm `complete()` không trả về phản hồi nào sau khi đã hủy. Một thông báo hủy đến muộn không thể thay đổi một bản ghi đã hoàn thành.

## Hết Thời gian chờ Cần Hai Đồng hồ Đo

Một bộ hẹn giờ không hoạt động (inactivity timer) đơn lẻ là không đủ.

Hãy sử dụng hai giới hạn:

1. **Hết giờ khi rảnh rỗi (Idle timeout):** Khoảng thời gian tối đa mà yêu cầu không sinh ra bất kỳ hoạt động hữu ích nào.
2. **Hết giờ tối đa (Maximum timeout):** Ngân sách thời gian thực tuyệt đối tính từ khi bắt đầu yêu cầu.

Tiến độ (progress) có thể đặt lại đồng hồ rảnh rỗi. Nhưng nó tuyệt đối không bao giờ được phép xóa bỏ hạn chót tối đa.

```text
bắt đầu: 0 ms
tiến độ: 400 ms
tiến độ: 800 ms
tiến độ: 1200 ms
hết giờ rảnh rỗi: 500 ms
hết giờ tối đa: 2000 ms
```

Tại thời điểm 1500 ms, yêu cầu vẫn đang hoạt động vì tiến độ gần nhất mới chỉ cách 300 ms. Nhưng tại thời điểm 2000 ms, hạn chót tối đa sẽ hủy yêu cầu ngay cả khi một sự kiện tiến độ khác vừa mới đến vào lúc 1999 ms.

Báo cáo tiến độ là tùy chọn. Một máy chủ có thể chấp nhận một token tiến độ nhưng không phát ra bất kỳ cập nhật nào. Đừng bao giờ biến sự hiện diện của một token thành một khoảng thời gian chờ vô tận.

Các giá trị tiến độ của MCP bắt buộc phải tăng đơn điệu. Các thông báo phải dừng lại sau khi đã hoàn thành hoặc bị hủy. Hãy giới hạn tần suất gửi tiến độ để một worker nhanh không làm tràn ngập tầng truyền vận.

## Hủy bỏ Yêu cầu Không phải là `tasks/cancel`

Các cơ chế này phục vụ các vòng đời hoàn toàn khác nhau.

| Cơ chế | Đối tượng tác động | Tín hiệu | Ý nghĩa khi thành công |
|-----------|--------|--------|--------------------|
| Hủy yêu cầu trên stdio | Một RPC đang xử lý | `notifications/cancelled` | Máy khách đã từ bỏ yêu cầu; máy chủ nên dừng lại nếu khả thi |
| Hủy yêu cầu trên HTTP | Một luồng phản hồi đang mở | Đóng luồng kết nối | Máy khách đã từ bỏ yêu cầu; máy chủ nên dừng lại nếu khả thi |
| `tasks/cancel` | Một Tác vụ (Task) bền vững | Yêu cầu MCP thông thường | Máy chủ xác nhận đã ghi nhận ý định hủy bỏ |

Một kết quả `tasks/cancel` thành công không chứng minh rằng worker đã dừng lại ngay lập tức. Tác vụ có thể vẫn ở trạng thái `working` cho đến khi một điểm kiểm tra (checkpoint) của worker quan sát thấy cờ hủy. Công việc hoàn toàn có thể hoàn thành trước điểm kiểm tra đó.

Đừng xóa bỏ trạng thái tác vụ bền vững khi kết nối HTTP bị đóng. Lý do cốt lõi để tạo ra một Task là vì vòng đời của nó tồn tại lâu hơn một yêu cầu đơn lẻ và một kết nối mạng.

## ID JSON-RPC Mới Không Đồng nghĩa với Tính Lũy kế

Các ID JSON-RPC chỉ dùng để tương quan giữa các yêu cầu và phản hồi. Chúng không đại diện cho một thao tác nghiệp vụ.

Giả sử một máy khách gửi một yêu cầu thanh toán với id `41`, bị mất phản hồi, và thử lại với id `42`. Máy chủ nhìn thấy hai thông điệp hoàn toàn khác nhau. Nếu không có một khóa định danh ứng dụng, nó không thể biết được cả hai thông điệp này đều đại diện cho cùng một lần thanh toán.

Một khóa lũy kế (idempotency key) xác định ý định nghiệp vụ:

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Máy chủ lưu trữ:

- khóa lũy kế;
- dấu vân tay (fingerprint) của các đối số thao tác;
- kết quả đã được ghi nhận (committed result).

Cùng một khóa và cùng các đối số sẽ trả về kết quả đã lưu. Cùng một khóa nhưng đối số khác biệt sẽ bị từ chối. Điều này ngăn chặn việc vô tình tái sử dụng khóa làm biến đổi một thao tác nghiệp vụ khác.

### Ranh giới sổ cái phải mang tính nguyên tử và bền vững

Trình tự sau đây là không an toàn:

```text
kiểm tra khóa
chạy thao tác biến đổi (mutation)
lưu kết quả
```

Hai worker có thể cùng quan sát thấy khóa chưa tồn tại và cả hai đều chạy thao tác biến đổi. Một sự cố sập hệ thống (crash) xảy ra sau khi sinh tác dụng phụ nhưng trước khi lưu kết quả sẽ tạo ra sự mơ hồ tương tự khi thử lại.

Bài học sử dụng một sổ cái SQLite được lưu trữ trên tệp. Lệnh `BEGIN IMMEDIATE` tuần tự hóa việc kiểm tra khóa, tác dụng phụ nghiệp vụ mô phỏng, bộ đếm thực thi và kết quả đã lưu vào một giao dịch duy nhất. Do đó, hai kết nối sổ cái độc lập cạnh tranh với cùng một khóa sẽ chỉ quan sát thấy một kết quả đã ghi nhận và một lượt thực thi duy nhất. Việc đóng và mở lại sổ cái vẫn bảo toàn bản ghi đó.

Mọi giá trị trả về đều được tái tạo từ chuỗi JSON đã lưu. Bên gọi không bao giờ nhận được đối tượng có thể biến đổi (mutable object) do sổ cái nắm giữ, vì vậy việc sửa đổi một từ điển trả về không thể làm sai lệch các kết quả phát lại sau đó.

Tác dụng phụ nghiệp vụ của bộ mô phỏng là biên lai và bộ đếm thực thi bên trong cùng một giao dịch SQLite. Một giao dịch thanh toán thực tế, triển khai phần mềm, hoặc gọi API bên ngoài không thể trở nên nguyên tử chỉ bằng cách ghi vào một bảng cục bộ. Môi trường sản xuất cần một giao dịch cơ sở dữ liệu dùng chung bền vững, một mẫu hộp thư đi giao dịch (transactional outbox), hoặc một nhà cung cấp thượng nguồn có thực thi cùng khóa lũy kế đó. Khóa tiến trình đơn lẻ không thể bảo vệ nhiều bản sao (replicas) hoặc tồn tại qua một lần khởi động lại.

### Ma trận thử lại (Retry Matrix)

Hãy phân loại các yêu cầu thử lại trước khi cài đặt chúng.

| Phân loại | Ví dụ | Quy tắc thử lại |
|------|---------|------------|
| An toàn (Safe) | Thao tác đọc tất định không có tác dụng phụ | Thử lại với ID JSON-RPC mới sau khi đã hiểu rõ ranh giới lỗi |
| Có điều kiện (Conditional) | Thao tác biến đổi có khóa lũy kế bền vững | Thử lại với cùng khóa và các đối số giống hệt nhau |
| Không an toàn (Unsafe) | Thao tác biến đổi không có cơ chế khử trùng lặp nghiệp vụ | Không thử lại tự động; phải đối soát (reconcile) trước |

Các chú thích công cụ như `readOnlyHint` và `idempotentHint` chỉ là các gợi ý không đáng tin cậy. Hợp đồng ứng dụng và cách triển khai của máy chủ mới là thứ quyết định mức độ an toàn khi thử lại.

## Áp lực Ngược (Backpressure) là Một Phần của Tính Đúng đắn

Một bên sản xuất sự kiện SSE có thể sinh ra tiến độ nhanh hơn mức mà máy khách, proxy hoặc mạng có thể tiêu thụ. Một hàng đợi không giới hạn sẽ biến sự chậm chạp thành thảm họa cạn kiệt bộ nhớ.

Hãy sử dụng một hàng đợi có giới hạn (bounded queue) và định nghĩa rõ những gì có thể chấp nhận mất mát.

Tiến độ là thông tin có thể thay thế. Một giá trị tiến độ mới hơn sẽ thay thế giá trị cũ hơn cho cùng một token. Nhưng phản hồi JSON-RPC cuối cùng thì không thể thay thế.

Bộ đệm trong bài học áp dụng chính sách sau:

1. Hợp nhất (coalesce) các tiến độ liền kề cho cùng một token.
2. Bỏ rơi (drop) tiến độ cũ nhất khi đạt dung lượng tối đa.
3. Đánh dấu luồng là cần phải nạp lại dữ liệu có thẩm quyền (authoritative refetch).
4. Luôn bảo toàn phản hồi cuối cùng.
5. Từ chối trạng thái mà việc bảo toàn phản hồi cuối cùng đòi hỏi phải loại bỏ một phản hồi cuối cùng khác.

Đây là sự mất mát có kiểm soát đi kèm với cơ chế phục hồi tường minh. Mất mát trong im lặng không phải là một chiến lược kỹ thuật.

### Đệm qua máy chủ ủy quyền (Proxy Buffering)

Một máy chủ có thể truyền luồng dữ liệu chuẩn xác nhưng một reverse proxy lại giữ các sự kiện trong bộ đệm của nó.

Đối với một phản hồi SSE, hãy gửi các tiêu đề:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

Đặc tả Streamable HTTP 2026 khuyến nghị sử dụng `X-Accel-Buffering: no` để các proxy tương thích phân phối sự kiện ngay lập tức.

Đối với các luồng tồn tại lâu nhưng ít dữ liệu, hãy định kỳ phát ra một dòng bình luận SSE:

```text
:
```

Máy khách sẽ bỏ qua các dòng bình luận này. Các bên trung gian nhìn thấy lưu lượng truyền tải và sẽ không tự ý đóng kết nối đang rảnh rỗi.

Gói giữ kết nối (keepalive) không phải là tiến độ công việc. Đừng đặt lại thời gian chờ rảnh rỗi ngữ nghĩa của một thao tác chỉ vì vừa nhận được một dòng bình luận ở tầng truyền vận.

## Kết nối Lại Đồng nghĩa với việc Nạp lại Dữ liệu (Refetch)

Streamable HTTP hiện đại không hỗ trợ khôi phục luồng SSE thông qua tiêu đề `Last-Event-ID`.

Sau khi một luồng `subscriptions/listen` bị ngắt:

1. Mở một yêu cầu lắng nghe mới với một ID JSON-RPC mới.
2. Khôi phục bộ lọc đăng ký mong muốn.
3. Nạp lại (refetch) các công cụ, tài nguyên, prompt hoặc Task bị ảnh hưởng từ các phương thức có thẩm quyền.
4. Khử trùng lặp trạng thái ứng dụng bằng các định danh ổn định.
5. Không phát lại một thao tác biến đổi không an toàn chỉ vì phản hồi của nó bị thất lạc.

Kế hoạch phục hồi mẫu đặt `sendLastEventId` thành false một cách tường minh và liệt kê các tài nguyên cần nạp lại.

### Ngăn chặn hiện tượng dồn ứ kết nối lại (Reconnect Herd)

Nếu 10.000 máy khách cùng kết nối lại vào đúng thời điểm 1 giây, máy chủ vừa phục hồi sẽ sập lần nữa.

Hãy sử dụng thuật toán lùi theo hàm mũ kết hợp độ trễ ngẫu nhiên (exponential backoff with jitter) và có mức trần giới hạn (cap). Bài học tính toán độ ngẫu nhiên tất định từ ID máy khách và số lần thử để các bài kiểm thử luôn có thể tái lập:

```text
lần thử 0: tối đa 250 ms
lần thử 1: tối đa 500 ms
lần thử 2: tối đa 1000 ms
...
mức trần: 8000 ms
```

Môi trường sản xuất có thể sử dụng tính ngẫu nhiên an toàn mật mã hoặc ngẫu nhiên từ môi trường chạy. Bất biến ở đây là sự phân tán thời gian, chứ không phải một công thức cố định.

## Xây dựng (Build It)

`code/main.py` xây dựng năm thành phần độ tin cậy nhỏ gọn.

### `RequestCoordinator`

- bắt đầu một yêu cầu đang xử lý với hạn chót rảnh rỗi và hạn chót tối đa;
- phát ra các thông báo tiến độ tăng đơn điệu;
- tạo tín hiệu hủy bỏ chuẩn xác cho stdio hoặc HTTP;
- bỏ qua các thông báo hủy không hợp lệ;
- làm rõ các cuộc đua trạng thái kết thúc giữa hủy bỏ và hoàn thành;
- dành riêng việc hủy bỏ do máy chủ gửi cho các đăng ký stdio.

### `MutationLedger`

- chứng minh rằng hai ID JSON-RPC sẽ thực thi hai lần nếu thiếu khóa nghiệp vụ;
- sử dụng giao dịch SQLite dựa trên tệp cho việc kiểm tra khóa, tác dụng phụ mô phỏng, bộ đếm thực thi và xác nhận kết quả;
- khử trùng lặp các đối số khớp nhau dưới một khóa lũy kế duy nhất qua các kết nối sổ cái độc lập;
- từ chối một khóa bị tái sử dụng với các đối số khác nhau;
- trả về các bản sao phòng vệ (defensive copies) và bảo toàn các bản ghi đã xác nhận qua các lần mở lại tệp.

### `DurableTaskService`

- xác nhận đã nhận yêu cầu hủy bỏ;
- giữ tác vụ ở trạng thái `working` cho đến điểm kiểm tra của worker;
- minh họa lý do tại sao sự xác nhận nhận tin không phải là trạng thái cuối cùng.

### `BoundedSseBuffer`

- hợp nhất hoặc loại bỏ bớt tiến độ khi chịu áp lực;
- ghi nhận việc bắt buộc phải nạp lại dữ liệu có thẩm quyền;
- không bao giờ bỏ rơi phản hồi cuối cùng.

### Các hàm hỗ trợ phục hồi (Recovery helpers)

- trả về các tiêu đề SSE an toàn cho proxy và dòng bình luận giữ kết nối;
- tạo kế hoạch kết nối lại và nạp lại dữ liệu;
- phân tán các lượt thử lại bằng thuật toán lùi số mũ tất định có độ trễ ngẫu nhiên.

## Sử dụng (Use It)

Từ thư mục gốc của kho lưu trữ:

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

Bản minh họa chạy cả hai phía của cuộc đua trung tâm, thực thi một biến đổi được khử trùng lặp theo giao dịch trong một sổ cái tệp tạm thời, làm quá tải một bộ đệm tiến độ có giới hạn, và cho thấy một Task bền vững chuyển dịch từ trạng thái xác nhận hủy sang trạng thái worker thực sự quan sát thấy lệnh hủy.

## Phòng thực hành Tương tác (Interactive Lab)

Chạy bốn thứ tự sự kiện mà không cần thêm hàm tạm dừng (`sleep`).

1. Bắt đầu yêu cầu `A`, hủy nó, sau đó gọi `complete()`.
2. Bắt đầu yêu cầu `B`, hoàn thành nó, sau đó mới gửi tín hiệu hủy.
3. Bắt đầu yêu cầu `C`, phát tiến độ trước mỗi hạn chót rảnh rỗi, sau đó vượt qua hạn chót tối đa.
4. Bắt đầu yêu cầu `D` qua Streamable HTTP và đóng luồng phản hồi của nó.

Ghi lại cho mỗi kịch bản:

- trạng thái kết thúc của yêu cầu;
- liệu phản hồi cuối cùng có tồn tại hay không;
- tín hiệu hủy bỏ được đặt lên đường truyền là gì;
- sự kiện nào máy khách nên bỏ qua.

Sau đó đổi `D` sang chạy trên stdio. Thao tác hoàn toàn giống nhau, nhưng tín hiệu hủy bỏ bắt buộc phải thay đổi.

## Phòng thực hành Ứng dụng (Practice Lab)

Thêm thao tác biến đổi `reserve_inventory` vào `MutationLedger`.

Yêu cầu:

1. Khóa ràng buộc mã SKU, số lượng, đối tượng thuê (tenant), và tên thao tác.
2. Thử lại với cùng khóa và cùng đối số sẽ trả về thông tin đặt giữ chỗ đầu tiên.
3. Thử lại với số lượng bị thay đổi sẽ báo lỗi mà không tạo thêm đặt chỗ mới.
4. Một lần thực thi đã xác nhận nhưng bị mất phản hồi có thể được đối soát lại bằng khóa.
5. Kết quả không ghi lại dữ liệu bí mật hoặc thanh toán.
6. Thử lại tự động bị vô hiệu hóa khi máy khách không cung cấp khóa.
7. Thêm kịch bản mô phỏng mất kết nối đăng ký và nạp lại bản ghi tồn kho trước khi quyết định bước tiếp theo.
8. Bắt đầu hai kết nối sổ cái tại một rào cản đồng bộ (barrier) và gửi cùng một khóa đồng thời. Khẳng định chỉ có một đặt chỗ được ghi nhận.
9. Sửa đổi đối tượng đặt chỗ được trả về lần đầu. Phát lại khóa và chứng minh kết quả lưu trữ không bị thay đổi.
10. Đóng và mở lại tệp sổ cái, sau đó đối soát lại đặt chỗ bằng khóa.

Hãy giữ tính trung thực cho bài thực hành: nếu dữ liệu tồn kho nằm ở một dịch vụ khác, hãy giải thích liệu dịch vụ đó có chấp nhận cùng khóa lũy kế hay không, hoặc liệu một mẫu hộp thư đi giao dịch có bắc cầu giữa việc ghi nhận cục bộ với tác dụng phụ từ xa hay không.

## Tạo phẩm Chuyển giao (Shipped Artifact)

`outputs/skill-mcp-reliability-reviewer.md` là một kỹ năng đánh giá độ tin cậy dạng phẳng. Cung cấp cho nó một thao tác MCP, tầng truyền vận, chính sách thời gian chờ, hành vi thử lại, chính sách hàng đợi và kế hoạch phục hồi. Nó sẽ trả về bảng phân tích cuộc đua, phân loại thử lại, ranh giới lũy kế, kiểm tra kiểm soát luồng và các ca kiểm thử thất bại.

## Xác minh (Verify It)

Bài học hoàn thành khi các khẳng định sau là đúng:

- Hủy bỏ trên stdio gửi `notifications/cancelled` và không nhận lại phản hồi.
- Hủy bỏ trên Streamable HTTP đóng luồng yêu cầu và không gửi POST hủy bỏ.
- Hủy trước khi hoàn thành sẽ chặn việc gửi phản hồi cuối cùng.
- Hoàn thành trước khi hủy sẽ bảo toàn phản hồi và bỏ qua thông báo hủy đến muộn.
- Tiến độ có thể đặt lại thời gian chờ rảnh rỗi nhưng không bao giờ đặt lại thời gian chờ tối đa.
- Chỉ một ID JSON-RPC mới sẽ khiến thao tác biến đổi thực thi lần thứ hai.
- Một khóa lũy kế cùng các đối số giống hệt nhau chỉ thực thi một lần trong cuộc đua đồng thời giữa hai kết nối.
- Bản ghi đã xác nhận tồn tại qua thao tác mở lại tệp và việc phát lại trả về bản sao phòng vệ.
- Sửa đổi một kết quả trả về không thể làm thay đổi kết quả được lưu trữ.
- Bộ đệm có giới hạn luôn nằm trong dung lượng và bảo toàn phản hồi cuối cùng.
- Kết nối lại sử dụng một yêu cầu mới, không gửi `Last-Event-ID`, và nạp lại trạng thái bị ảnh hưởng.
- Xác nhận nhận lệnh `tasks/cancel` vẫn để tác vụ ở trạng thái chưa kết thúc cho đến khi worker thực sự quan sát thấy nó.

## Các Chế độ Lỗi trong Môi trường Sản xuất

| Lỗi gặp phải | Triệu chứng quan sát | Phản hồi xử lý đúng đắn |
|---------|--------------------|------------------|
| Máy khách HTTP gửi POST thông báo hủy | Máy chủ và máy khách không thống nhất về vòng đời yêu cầu | Đóng luồng phản hồi SSE của chính yêu cầu đó |
| Máy chủ phản hồi sau khi đã chấp nhận hủy | Máy khách nhận được kết quả muộn không thể sử dụng | Dừng công việc và chặn thông điệp tiếp theo khi việc hủy thắng cuộc đua |
| Tiến độ đặt lại mọi hạn chót | Công việc bị treo tồn tại mãi mãi | Duy trì một hạn chót thời gian tối đa tuyệt đối riêng biệt |
| Coi RPC ID mới là khử trùng lặp | Thao tác trừ tiền, triển khai hoặc xóa bị chạy hai lần | Bổ sung khóa lũy kế tầng ứng dụng mang tính bền vững |
| Kiểm tra khóa và gây tác dụng phụ tách rời nhau | Các worker đồng thời cùng thấy thiếu khóa | Ghi nhận quyền sở hữu khóa, bản ghi tác dụng phụ và kết quả một cách nguyên tử |
| Dùng sổ cái trong bộ nhớ cho nhiều bản sao | Khởi động lại hoặc worker khác quên mất các lần ghi trước | Sử dụng bộ lưu trữ bền vững dùng chung hoặc tính lũy kế thượng nguồn |
| Trả trực tiếp kết quả có thể biến đổi | Bên gọi sửa đổi đối tượng làm hỏng các lần phát lại sau | Tuần tự hóa kết quả đã lưu và trả về các bản sao phòng vệ |
| Tái sử dụng khóa với các đối số bị thay đổi | Một khóa đại diện cho hai ý định nghiệp vụ khác nhau | Lưu trữ và so sánh dấu vân tay của các đối số |
| Hàng đợi tiến độ không giới hạn | Bộ nhớ phình to khi gặp bên tiêu thụ chậm chạp | Hợp nhất và loại bỏ bớt tiến độ có thể thay thế trong một giới hạn |
| Phản hồi cuối cùng bị bỏ rơi khi quá tải | Máy khách không thể biết được kết quả của yêu cầu | Dành riêng dung lượng hoặc loại bỏ tiến độ, tuyệt đối không bỏ rơi phản hồi cuối |
| Proxy đệm luồng SSE | Tiến độ đến theo từng đợt dồn ứ hoặc sau khi hết giờ | Tắt tính năng đệm và cấu hình thời gian chờ proxy tương thích |
| Mặc định hỗ trợ `Last-Event-ID` | Máy khách khôi phục từ trạng thái máy chủ không hỗ trợ | Kết nối lại bằng một yêu cầu mới và nạp lại dữ liệu |
| Mọi máy khách kết nối lại cùng lúc | Quá trình phục hồi tạo ra một đợt sự cố sập mới | Sử dụng thuật toán lùi số mũ có trần kèm độ trễ ngẫu nhiên |
| Coi xác nhận Task là hủy bỏ hoàn tất | Worker vẫn tiếp tục chạy sau khi giao diện báo đã dừng | Thăm dò (poll) Task cho đến khi đạt trạng thái kết thúc thực sự |

## Kết nối Capstone

Dự án capstone của hệ sinh thái công cụ nên coi độ tin cậy là bằng chứng có thể thực thi, chứ không phải một đoạn văn trong sơ đồ kiến trúc.

Bắt buộc phải có các tạo phẩm sau:

- một bản ghi vết cuộc đua hủy bỏ cho từng tầng truyền vận;
- một bảng thử lại cho mọi thao tác biến đổi dữ liệu được phơi bày;
- một bản ghi khóa lũy kế và mẫu thử nghiệm khi đối số không khớp;
- một bản ghi vết cạnh tranh cùng khóa đồng thời, kiểm tra mở lại tệp và kiểm tra đối số bị đổi;
- kết quả quá tải của bộ đệm có giới hạn;
- các tiêu đề SSE của reverse-proxy và chính sách rảnh rỗi;
- kế hoạch kết nối lại nêu rõ các phương thức nạp lại dữ liệu có thẩm quyền;
- dấu vết hủy bỏ Task bền vững khi dự án capstone sử dụng Tasks.

Một yêu cầu màu xanh lá cây trong tiến trình cục bộ chỉ chứng minh nhánh chạy thuận lợi. Dự án capstone chỉ thực sự sẵn sàng cho sản xuất khi các phản hồi bị mất, hủy bỏ muộn, bên tiêu thụ chậm chạp, và hiện tượng dồn ứ kết nối lại đều có kết quả tất định.

## Thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa kỹ thuật |
|------|---------|
| Hủy bỏ yêu cầu (Request cancellation) | Việc từ bỏ một yêu cầu MCP đang được xử lý |
| Cuộc đua hủy bỏ (Cancellation race) | Sự cạnh tranh giữa việc hoàn thành kết thúc và sự kiện hủy bỏ |
| Hết giờ rảnh rỗi (Idle timeout) | Giới hạn thời gian kể từ hoạt động hữu ích gần nhất của yêu cầu |
| Hết giờ tối đa (Maximum timeout) | Giới hạn tuyệt đối tính từ khi bắt đầu yêu cầu, không bị ảnh hưởng bởi tiến độ |
| Khóa lũy kế (Idempotency key) | Định danh tầng ứng dụng giúp khử trùng lặp một ý định nghiệp vụ |
| Sổ cái nguyên tử (Atomic ledger) | Ranh giới bền vững ghi nhận quyền sở hữu khóa, bản ghi tác dụng phụ và kết quả thành một khối duy nhất |
| Áp lực ngược (Backpressure) | Cơ chế kiểm soát áp dụng khi bên sản xuất vượt quá tốc độ của bên tiêu thụ |
| Hợp nhất tiến độ (Progress coalescing) | Thay thế tiến độ cũ hơn bằng giá trị mới hơn có thẩm quyền |
| Nạp lại dữ liệu (Refetch) | Đọc lại trạng thái hiện tại sau khi bị gián đoạn luồng dữ liệu |
| Độ trễ ngẫu nhiên (Jitter) | Sự biến thiên có chủ ý nhằm phân tán các lượt thử lại theo thời gian |

## Đọc thêm

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Đặc tả phần mở rộng MCP Tasks](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
