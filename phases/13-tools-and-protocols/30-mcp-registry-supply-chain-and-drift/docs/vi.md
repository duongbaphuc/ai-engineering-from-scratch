# Chuỗi Cung ứng MCP Registry: Tiếp nhận, Trôi dạt và Hoàn nguyên (MCP Registry Supply Chain: Admission, Drift, and Rollback)

> Một mục trong sổ đăng ký (registry) cho bạn biết những gì nhà phát hành đã khai báo. Khâu tiếp nhận trong môi trường sản xuất chứng minh những gì bạn đã tải về, những gì bạn đã quan sát, những gì bạn đã phê duyệt, và những gì bạn có thể khôi phục một cách an toàn.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Phase 13 · 17 (Cổng kết nối và Sổ đăng ký - Gateways and Registries), Phase 13 · 18 (Xác thực trong Sản xuất - Production Authentication)  
**Thời lượng:** ~90 phút  

## Mục tiêu học tập

- Tách bạch giữa việc xuất bản trên Registry, nguồn gốc xuất xứ gói phần mềm (package provenance), khám phá khi chạy (runtime discovery), và phê duyệt cục bộ.
- Xác minh không gian tên (namespace) của máy chủ MCP mà không tin tưởng vào cái tên nằm bên trong bản ghi của chính nó.
- Ghim cố định (pin) các bằng chứng bất biến về bản xuất bản, nguồn thực thi, nguồn gốc xuất xứ, và bộ mô tả đang hoạt động.
- Phát hiện các thay đổi trạng thái sổ đăng ký và sự trôi dạt runtime (runtime drift) sau khi đã tiếp nhận.
- Hoàn nguyên (rollback) việc định tuyến về một phiên bản đã được tiếp nhận trước đó mà không viết lại lịch sử.
- Duy trì một sổ cái tiếp nhận có khả năng phát hiện can thiệp (tamper-evident admission ledger) giải thích rõ ràng cho từng quyết định.

## Vấn đề

Bạn tìm thấy `com.example/inventory` trong một sổ đăng ký. Phần mô tả của nó trông có vẻ chuẩn xác. Gói phần mềm của nó tồn tại thực sự. Máy chủ phản hồi lệnh `server/discover`.

Đó không phải là một sự thật đơn lẻ. Đó là một chuỗi các sự thật đến từ các cơ quan thẩm quyền hoàn toàn khác nhau:

1. Một nhà phát hành đã được xác thực cho một không gian tên đã gửi lên một bản ghi.
2. Một kho lưu trữ gói phần mềm đã phục vụ một tạo phẩm với một định danh và mã băm cụ thể.
3. Một điểm cuối đang chạy đã báo cáo một phiên bản giao thức, các năng lực, các công cụ, và thông tin chẩn đoán máy chủ.
4. Tổ chức của bạn đã quyết định rằng sự kết hợp chính xác này được phép hoạt động.

Việc gộp chung tất cả các sự thật đó thành suy nghĩ "nó nằm trong sổ đăng ký, nên cứ tin tưởng nó" sẽ tạo ra một điểm mù chết người trong chuỗi cung ứng. Một bản xuất bản hợp lệ vẫn có thể bị coi là lỗi thời (deprecated). Một thẻ phiên bản gói có thể trỏ tới một tạo phẩm bất thường nếu bạn không ghim chặt mã băm của nó. Một máy chủ có thể âm thầm bổ sung một công cụ mang tính phá hủy sau khi đã được phê duyệt. Một lần hoàn nguyên có thể âm thầm chọn một phiên bản chưa từng được tiếp nhận.

Giải pháp là một bộ điều khiển tiếp nhận (admission controller) có bằng chứng tại mọi ranh giới.

## Registry là một Mục lục, Không phải Hệ thống Phê duyệt của Bạn

MCP Registry chính thức lưu trữ siêu dữ liệu máy chủ. Bản ghi `server.json` của nó đặt tên cho một phiên bản máy chủ và khai báo một hoặc nhiều gói phần mềm hoặc điểm cuối từ xa. Các quy tắc xuất bản bổ sung thêm bước xác thực không gian tên, kiểm tra quyền sở hữu gói, các quy tắc sổ đăng ký bị hạn chế, và một vị trí hẹp chứa siêu dữ liệu của nhà phát hành.

Những biện pháp kiểm soát đó giải quyết các câu hỏi về khâu xuất bản. Chính sách sản xuất của bạn vẫn phải tự trả lời các câu hỏi về khâu triển khai:

| Ranh giới | Câu hỏi đặt ra | Bên nắm giữ bằng chứng |
|---|---|---|
| Không gian tên (Namespace) | Nhà phát hành có được phép dùng tên này không? | Xác thực của Registry kết hợp với đầu vào không gian tên đã xác minh của bạn |
| Bản ghi (Record) | Nhà phát hành đã khai báo những gì cho phiên bản này? | Mã băm bất biến của tệp `server.json` |
| Nguồn thực thi (Execution source) | Gói phần mềm hoặc điểm cuối từ xa nào sẽ thực thi? | Các trường nguồn khai báo, kết quả xác minh quyền sở hữu, tầng truyền vận, và mã băm đáng tin cậy |
| Môi trường chạy (Runtime) | Điểm cuối đang phơi bày những gì ở thời điểm hiện tại? | Phản hồi `server/discover` và các bộ mô tả công cụ |
| Tiếp nhận (Admission) | Chính sách của bạn đã phê duyệt tập hợp chính xác này chưa? | Bản ghim cục bộ và mục ghi trong sổ cái |
| Vận hành (Operations) | Nó còn an toàn không, và thứ gì có thể thay thế nó? | Kiểm tra trôi dạt, đồng bộ trạng thái, kiểm tra sức khỏe, và lộ trình hoàn nguyên |

Phiên bản lược đồ của Registry và phiên bản giao thức của MCP là hai khái niệm độc lập. Một bản ghi có thể sử dụng lược đồ máy chủ đã xuất bản `2025-12-11` trong khi máy chủ thực tế hỗ trợ giao thức MCP `2026-07-28`. Tuyệt đối không bao giờ suy diễn phiên bản này từ phiên bản kia.

````figure mcp-registry-admission
````

## Bảy bước Kiểm soát trong Một Quyết định Tiếp nhận

### 1. Xác minh không gian tên (Namespace verification)

Các tên gọi chính thức trong Registry sử dụng các không gian tên đã được xác thực danh tính. Một tên miền đã xác minh có thể ánh xạ thành một tiền tố tên miền đảo ngược. Ví dụ, quyền kiểm soát `example.com` có thể thiết lập không gian tên `com.example/*`.

Đừng chấp nhận một phép kiểm tra tiền tố chuỗi đơn giản:

```python
server_name.startswith("com.example")
```

Phép kiểm tra đó sẽ vô tình chấp nhận cả `com.exampleevil/tool`. Hãy tách chuỗi tên tại ký tự `/`, bắt buộc phải có một định danh (slug) không rỗng, và so sánh chính xác đoạn không gian tên. Quan trọng hơn, hãy truyền không gian tên đã xác minh vào khâu tiếp nhận từ kết quả xác thực. Đừng bao giờ suy diễn độ tin cậy từ chính bản ghi chưa được tin tưởng.

Không gian tên dựa trên GitHub và không gian tên dựa trên tên miền sử dụng các quy trình xác thực khác nhau. Hãy chuẩn hóa cả hai quy trình thành một đầu vào tiếp nhận duy nhất: chuỗi không gian tên đã được xác minh chính xác.

### 2. Ghép nối nguồn gốc xuất xứ (Provenance join)

Đối với một bản ghi gói phần mềm, phần khai báo và tạo phẩm tải về bắt buộc phải ghép nối trên các trường dữ liệu tường minh:

- loại sổ đăng ký gói (package registry type)
- định danh gói (package identifier)
- phiên bản gói (package version)
- kết quả xác minh quyền sở hữu
- mã băm của tạo phẩm đã tải về

Đồng thời phải xác thực tầng truyền vận gói đã khai báo. Một bản ghi chỉ chứa một điểm cuối từ xa vẫn là hợp lệ và không được phép bị từ chối chỉ vì thiếu gói phần mềm. Đối với một nguồn từ xa, hãy ghép nối URL và loại truyền vận đã khai báo với quyền sở hữu điểm cuối đã được xác minh độc lập và mã băm của bằng chứng triển khai hoặc kết nối đáng tin cậy.

Mã nguồn bài học hỗ trợ cả hai loại nguồn và thực hiện băm nguồn đã chọn cùng với nguồn Registry, tên máy chủ, phiên bản Registry, mã băm bản ghi, và mã băm bằng chứng. Mã băm nguồn gốc xuất xứ (provenance digest) thu được là một con trỏ nhỏ gọn trỏ tới toàn bộ tập hợp bằng chứng. Nó không thay thế cho việc lưu giữ các bằng chứng thực tế đó.

Đừng bao giờ chấp nhận một mã băm chỉ được cung cấp bởi chính tạo phẩm mà bạn đang cố gắng xác minh. Hãy tự tính toán nó tại một ranh giới tải dữ liệu đáng tin cậy, hoặc nhận nó từ một dịch vụ gói mà kết quả xác minh của nó đã được bạn kiểm chứng.

### 3. Ghim quyết định, không chỉ ghim phiên bản

Các phiên bản Registry là các định danh xuất bản duy nhất. Siêu dữ liệu đã xuất bản là bất biến. Một bản ghi bị thay đổi bắt buộc phải có một phiên bản mới. Quy chuẩn Semantic Versioning được khuyến nghị, nhưng Registry không bắt buộc và không chấp nhận các dải phiên bản (version ranges).

Điều này có nghĩa là `^1.4` không phải là một bản ghim tiếp nhận. "latest" cũng không phải. Một bản ghim hữu ích cần chứa:

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

Việc ghim nhiều tầng cho phép bạn xác định chính xác ranh giới nào đã bị thay đổi. Mã băm bản ghi thay đổi dưới cùng một phiên bản Registry là lỗi tính toàn vẹn của Registry. Mã băm nguồn thay đổi dưới cùng một tọa độ gói hoặc điểm triển khai từ xa là lỗi tính toàn vẹn nguồn thực thi. Mã băm bộ công cụ thay đổi là sự trôi dạt runtime.

### 4. Phát hiện trôi dạt runtime thực tế (Live drift detection)

Khâu tiếp nhận phải quan sát chính máy chủ sẽ thực sự nhận lưu lượng truy cập. Hãy gọi `server/discover`, liệt kê hoặc thu thập các bộ mô tả công cụ được phơi bày thông qua đường dẫn tin cậy của bạn, và xác minh:

- phiên bản `2026-07-28` có nằm trong `supportedVersions` hay không
- toàn bộ các năng lực bắt buộc cục bộ đều có mặt đầy đủ
- mỗi bộ mô tả công cụ đều có đúng định danh và bề mặt lược đồ bắt buộc
- mã băm bộ mô tả đã chuẩn hóa khớp với bản ghim đã tiếp nhận trong các lần kiểm tra sau

Trường giá trị tùy chọn `_meta["io.modelcontextprotocol/serverInfo"]` trong kết quả chỉ là ngữ cảnh hiển thị, nhật ký và gỡ lỗi do máy chủ tự báo cáo. Hãy ghi lại nó như bằng chứng chẩn đoán, nhưng tuyệt đối không bao giờ sử dụng nó để thiết lập không gian tên, quyền sở hữu gói, quyền sở hữu điểm cuối, việc tiếp nhận, hoặc bất kỳ quyết định bảo mật nào khác. Một bí danh `serverInfo` trực tiếp nằm ngoài `_meta` không phải là trường hợp đồng và không được phép đưa vào bằng chứng chẩn đoán.

Chỉ chuẩn hóa các trường mà thứ tự của chúng không mang ý nghĩa ngữ nghĩa. Mã mẫu sắp xếp danh sách công cụ theo tên ổn định trước khi băm, do đó sự thay đổi thứ tự danh sách vô hại sẽ không gây ra cảnh báo trôi dạt. Mã không loại bỏ các trường mô tả. Một công cụ mới, lược đồ bị đổi, mô tả bị sửa, hoặc các chú thích mới đều sẽ làm thay đổi bản ghim.

Mã mẫu coi các bộ mô tả sai định dạng và bất kỳ sự thay đổi mã băm mô tả nào là sự trôi dạt (drift), tiến hành cách ly (quarantine) bản ghim, gỡ bỏ tuyến đường đang hoạt động của nó, và chặn phiên bản đó làm mục tiêu hoàn nguyên. Chính sách sản xuất có thể chỉ cho phép thay đổi câu chữ biên tập thông qua một đợt đánh giá mới, bởi vì phần mô tả tác động trực tiếp đến việc lựa chọn công cụ của mô hình. Siêu dữ liệu mang tính "hình thức bên ngoài" hoàn toàn có thể làm biến đổi hành vi của tác tử.

### 5. Trạng thái Registry là trạng thái sống

API của Registry đính kèm một đối tượng `_meta` ở cấp phản hồi bên cạnh mỗi bản ghi máy chủ. Các trường do Registry quản lý nằm dưới `_meta["io.modelcontextprotocol.registry/official"]`. Hãy truyền đối tượng `_meta` của phản hồi vào khâu tiếp nhận và đọc trường `_meta["io.modelcontextprotocol.registry/official"].status`. Một giá trị `_meta.status` trực tiếp không phải là định dạng đường truyền chính thức. Đừng nhầm lẫn siêu dữ liệu phản hồi với `_meta` nội bộ của bản ghi xuất bản. Trạng thái có thể là:

- `active`: được trả về theo mặc định và đủ điều kiện để tiếp nhận cục bộ
- `deprecated`: vẫn có thể khám phá kèm theo cảnh báo, nhưng không còn là lựa chọn tự động an toàn
- `deleted`: bị ẩn theo mặc định trong khi bản ghi lịch sử của nó vẫn có thể truy cập qua chế độ xem đã xóa hoặc chế độ xem gia tăng

Hãy đồng bộ trạng thái sau khi đã tiếp nhận. Nếu một phiên bản đang hoạt động chuyển sang trạng thái bị phản đối (deprecated) hoặc bị xóa (deleted), hãy cách ly bản ghim của nó và ngừng định tuyến công việc mới đến nó. Hãy lưu giữ bằng chứng. Việc bị xóa khỏi danh sách mặc định không đồng nghĩa với việc bạn được phép xóa sạch dấu vết kiểm toán của mình.

Siêu dữ liệu tùy chỉnh do nhà phát hành cung cấp chỉ được phép nằm dưới `_meta.io.modelcontextprotocol.registry/publisher-provided` trong bản ghi xuất bản. Siêu dữ liệu phản hồi do Registry quản lý là hoàn toàn tách biệt. Đừng để nhà phát hành tự ý thiết lập trạng thái chính thức của chính họ.

### 6. Hoàn nguyên đồng nghĩa với phục hồi định tuyến (Route restoration)

Một bản xuất bản bất biến sẽ không bị chỉnh sửa trong quá trình hoàn nguyên. Hoàn nguyên là việc lựa chọn một bản ghim đã được tiếp nhận từ trước, hiện vẫn đủ điều kiện, và thay đổi tuyến đường đang hoạt động trỏ về nó.

Một mục tiêu an toàn bắt buộc phải:

1. Có một bản ghi tiếp nhận đã hoàn thành.
2. Vẫn có trạng thái Registry là `active` theo chính sách của bạn.
3. Không bị cách ly bởi các bằng chứng bảo mật hoặc bằng chứng runtime.
4. Vẫn phân giải chính xác về gói phần mềm và tập bộ mô tả thực tế đã được ghim.
5. Vượt qua các bước kiểm tra sức khỏe hiện tại.

Mã mẫu tập trung vào ba điều kiện đầu tiên. Một bộ đối soát thực tế trong môi trường sản xuất nên tải lại gói phần mềm và kiểm tra lại điểm cuối thực tế trước khi kích hoạt.

### 7. Ghi nối tiếp vào sổ cái tiếp nhận (Admission ledger)

Cơ sở dữ liệu tiếp nhận cho biết phiên bản nào đang hoạt động. Một sổ cái giải thích lý do tại sao.

Mỗi mục ghi mẫu chứa số thứ tự, thời gian, sự kiện, máy chủ, phiên bản, kết quả, lý do, bằng chứng, mã băm của mục trước đó, và mã băm của chính nó. Việc thay đổi một kết quả cũ sẽ phá vỡ tính xác thực của mục đó và mọi mắt xích phía sau.

Cơ chế này giúp phát hiện can thiệp (tamper-evident), chứ không có phép màu chống can thiệp tuyệt đối (tamper-proof). Hãy neo (anchor) các đỉnh sổ cái định kỳ vào một miền tin cậy tách biệt, chẳng hạn như siêu dữ liệu phát hành có chữ ký số hoặc kho lưu trữ ghi-một-lần (write-once storage). Giới hạn những ai có quyền ghi nối tiếp. Giữ các token ủy quyền, thông tin đăng nhập gói, đối số công cụ và dữ liệu điểm cuối riêng tư nằm ngoài bằng chứng lưu trữ.

## Xây dựng (Build It)

Bộ điều khiển có thể chạy được nằm trong `code/main.py`. Nó chỉ sử dụng thư viện chuẩn của Python.

Bắt đầu với bản trình diễn hữu hạn:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

Bản trình diễn thực hiện năm thao tác:

1. Tiếp nhận `1.0.0` với không gian tên, nguồn gốc gói, giao thức, năng lực và công cụ trùng khớp.
2. Tiếp nhận `1.1.0` và kích hoạt nó hoạt động.
3. Quan sát thấy một công cụ xóa bất thường xuất hiện ở môi trường chạy.
4. Quan sát thấy trạng thái Registry của `1.1.0` chuyển thành `deprecated`.
5. Phục hồi việc định tuyến trở về bản ghim `1.0.0` vẫn đang được tiếp nhận.

Hình thái đầu ra kỳ vọng:

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

Hãy đọc mã triển khai theo thứ tự sau:

1. `namespace_for_domain()` và `namespace_matches()` thiết lập thẩm quyền đặt tên chuẩn xác.
2. `digest()` và `normalized_tools()` sinh ra các bằng chứng tất định.
3. `RegistryAdmissionController.admit()` ghép nối việc xuất bản, nguồn gốc xuất xứ, môi trường chạy và chính sách.
4. `check_live()` so sánh quan sát mới với bản ghim.
5. `observe_registry_status()` cách ly các phiên bản có trạng thái Registry bị thay đổi.
6. `rollback()` chỉ kích hoạt một mục tiêu hợp lệ đã được tiếp nhận từ trước.
7. `AdmissionLedger.verify()` phát hiện các thay đổi đối với lịch sử đã ghi nhận.

## Sử dụng (Use It)

Đặt bộ điều khiển nằm giữa khâu khám phá và định tuyến:

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                                |                 |
                                                v                 v
                                           evidence store    admission ledger
```

Sử dụng các danh tính riêng biệt cho các tác vụ này. Một tiến trình đồng bộ Registry chỉ cần quyền đọc siêu dữ liệu. Một bộ xác minh tạo phẩm cần quyền tải gói. Một bộ đối soát định tuyến cần quyền kích hoạt một bản ghim đã phê duyệt. Không tiến trình nào trong số chúng cần nắm giữ toàn bộ thông tin xác thực.

Hãy làm rõ các trạng thái triển khai. "Approved" nghĩa là bằng chứng đã vượt qua chính sách. "Active" nghĩa là tuyến đường hiện đang chọn nó. "Quarantined" nghĩa là nó không thể nhận công việc mới. "Superseded" nghĩa là một phiên bản đã tiếp nhận khác đang hoạt động. Đừng mã hóa cả bốn ý nghĩa này vào một biến boolean duy nhất.

Hãy chạy khâu tiếp nhận trước khi phơi bày một máy chủ trong `tools/list`. Nếu không, máy khách có thể khám phá thấy một công cụ trong khoảng thời gian trống giữa lúc xuất bản và lúc đánh giá chính sách.

## Phòng thực hành Tương tác (Interactive Lab)

Bạn sẽ quan sát từng ranh giới thất bại độc lập.

### Lab A: Xung đột không gian tên

Mở một shell Python từ thư mục mã nguồn:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

Sau đó chạy:

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

Kết quả đầu tiên là `True`; kết quả thứ hai là `False`. Thay thế phép so sánh chính xác bằng `startswith` trên máy cục bộ và quan sát lý do tại sao cái tên thứ hai vượt qua ranh giới một cách sai lầm. Khôi phục lại phép so sánh chính xác trước khi tiếp tục.

### Lab B: Trôi dạt bộ mô tả

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

Kiểm tra lý do và trạng thái định tuyến. Gói phần mềm và bản ghi Registry không hề thay đổi. Bề mặt công cụ khi chạy đã thay đổi, vì vậy bộ điều khiển đã cách ly và hủy kích hoạt bản ghim. Đây chính là lý do tại sao việc kiểm soát chuỗi cung ứng phải tiếp tục diễn ra sau khi đã cài đặt.

### Lab C: Trạng thái và hoàn nguyên

Tiếp nhận `1.1.0`, đánh dấu nó là deprecated, và thử cả hai mục tiêu hoàn nguyên:

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

Mục tiêu bị cách ly bị từ chối. Bản ghim hoạt động trước đó được chấp nhận. Sổ cái vẫn duy trì tính hợp lệ.

## Phòng thực hành Ứng dụng (Practice Lab)

Mở rộng bộ điều khiển với một cổng phê duyệt hai người (two-person approval gate).

Yêu cầu:

- Lưu trữ phê duyệt dưới dạng các tham chiếu bằng chứng có chữ ký số, không phải các tên có thể biến đổi trong bản ghim.
- Yêu cầu hai danh tính người đánh giá khác nhau cho một bộ công cụ chứa công cụ có `destructiveHint: true`.
- Từ chối các danh tính người đánh giá bị trùng lặp.
- Bảo toàn nỗ lực tiếp nhận ban đầu trong sổ cái khi phê duyệt chưa hoàn tất.
- Bổ sung các bài kiểm thử cho các trường hợp: 0, 1, trùng lặp, và 2 phê duyệt riêng biệt.
- Không ghi nhật ký chữ ký, thông tin xác thực hoặc toàn bộ đối số công cụ riêng tư.

Thành công nghĩa là một công cụ mang tính phá hủy không thể trở nên active cho đến khi cả hai danh tính đã phê duyệt chính xác các mã băm của bản ghi, gói phần mềm và bộ công cụ.

## Tạo phẩm Chuyển giao (Shipped Artifact)

Bài học này chuyển giao tệp `outputs/skill-mcp-registry-admission.md`. Hãy sử dụng nó như một tài liệu hướng dẫn vận hành (runbook) phẳng, có thể tái sử dụng khi đánh giá một phiên bản Registry mới hoặc điều tra sự trôi dạt. Nó định nghĩa các đầu vào, quy tắc từ chối, gói bằng chứng, đối soát trạng thái, và bằng chứng hoàn nguyên mà không phụ thuộc vào tên các lớp mẫu.

## Xác minh (Verify It)

Chạy bản trình diễn và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khâu xác minh cần chứng minh:

- ranh giới không gian tên chính xác từ chối các tiền tố có vẻ ngoài tương tự
- chỉ có trạng thái Registry chính thức có không gian tên mới có thể làm cho một phiên bản đủ điều kiện
- bằng chứng gói và từ xa chưa được xác minh hoặc không khớp đều bị từ chối
- siêu dữ liệu của nhà phát hành không thể giả mạo siêu dữ liệu do Registry quản lý
- thứ tự công cụ được chuẩn hóa mà không che giấu các thay đổi của bộ mô tả
- cấu trúc gói và công cụ sai định dạng đều từ chối một cách an toàn
- `serverInfo` giữ nguyên vai trò chẩn đoán và không bao giờ cung cấp thẩm quyền tiếp nhận
- sự trôi dạt của bộ mô tả sẽ cách ly, hủy kích hoạt và chặn việc hoàn nguyên về bản ghim đó
- các thay đổi trạng thái sẽ cách ly các bản ghim đang hoạt động
- việc hoàn nguyên không thể chọn một phiên bản bị cách ly hoặc chưa xác định
- hành vi can thiệp vào sổ cái đều bị phát hiện

## Các Chế độ Lỗi trong Môi trường Sản xuất

| Lỗi gặp phải | Nguyên nhân xảy ra | Phản hồi xử lý bắt buộc |
|---|---|---|
| Tên trông hợp lệ nhưng không gian tên chưa từng được xác thực | Chính sách tin tưởng văn bản bản ghi | Từ chối cho đến khi một bộ xác minh không gian tên đáng tin cậy cung cấp tiền tố chính xác |
| Cùng một tọa độ gói trả về các byte mới | Kho thượng nguồn bị biến đổi hoặc kênh phân phối bị xâm phạm | Ngừng kích hoạt, lưu giữ cả hai mã băm, điều tra ranh giới tải dữ liệu |
| "Latest" tự động thay đổi mà không qua đánh giá | Lựa chọn thả nổi thoát khỏi bản ghim | Chỉ phân giải các phiên bản và mã băm đã tiếp nhận chính xác |
| Công cụ mới xuất hiện sau khi phê duyệt | Trôi dạt khi chạy hoặc điểm triển khai bị đổi | Cách ly tuyến đường và thu thập một quan sát bộ mô tả mới |
| Phiên bản bị phản đối vẫn tiếp tục hoạt động | Quá trình đồng bộ trạng thái bị thiếu hoặc chậm trễ | Đối soát trạng thái theo lịch trình và trước khi kích hoạt |
| Bản ghi bị xóa biến mất khỏi đồng bộ mặc định | Máy khách chỉ yêu cầu các bản ghi active | Sử dụng cơ chế đối soát gia tăng hoặc nhận biết bản ghi đã xóa và lưu giữ lịch sử cục bộ |
| Mục tiêu hoàn nguyên chưa từng được tiếp nhận | Khâu kiểm soát định tuyến và trạng thái phê duyệt bị ngắt kết nối | Từ chối hoàn nguyên và chạy quy trình tiếp nhận mới cho mục tiêu đó |
| Sổ cái xác thực thành công cục bộ sau khi kẻ tấn công viết lại toàn bộ | Chuỗi mã băm không có điểm tựa neo bên ngoài | Xuất bản các đỉnh sổ cái có chữ ký số sang một miền tin cậy riêng biệt |
| Bằng chứng chứa bearer token hoặc đối số công cụ | Nhật ký sao chép toàn bộ yêu cầu | Cắt bỏ/che giấu (redact) ngay tại thời điểm thu thập và chỉ lưu giữ bằng chứng tối thiểu |

## Quy tắc Vận hành

Khâu xuất bản trả lời câu hỏi: "danh tính này có thể xuất bản cái tên này không?" Khâu tiếp nhận trả lời câu hỏi: "chúng ta có thực thi đúng tạo phẩm này và phơi bày đúng hành vi này không?" Hãy giữ các quyết định đó tách bạch độc lập, ghim chặt mọi điểm ghép nối, và làm cho việc hoàn nguyên dựa trên bằng chứng thay vì dựa vào trí nhớ.

## Đọc thêm

- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Official Registry OpenAPI contract](https://registry.modelcontextprotocol.io/openapi.yaml)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
