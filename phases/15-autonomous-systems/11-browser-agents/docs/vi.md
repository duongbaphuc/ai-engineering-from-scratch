# Tác tử trình duyệt và các tác vụ web chân trời dài hạn

> ChatGPT agent (tháng 7 năm 2025) đã hợp nhất Operator và nghiên cứu chuyên sâu (deep research) thành một tác tử trình duyệt/terminal thống nhất và thiết lập kỷ lục đỉnh cao công nghệ (SOTA) trên BrowseComp ở mức 68.9%. OpenAI đã đóng cửa Operator độc lập vào ngày 31 tháng 8 năm 2025 — một bước củng cố ở tầng sản phẩm. Thương vụ Anthropic thâu tóm Vercept đã nâng tỷ lệ thành công của Claude Sonnet trên OSWorld từ dưới 15% lên 72.5%. WebArena-Verified (ServiceNow, ICLR 2026) đã khắc phục tỷ lệ âm tính giả (false-negative rate) ~11.3% trong phiên bản WebArena gốc và phát hành tập con Khó (Hard subset) gồm 258 tác vụ. Những con số này là hoàn toàn có thật. Nhưng bề mặt tấn công cũng nguy hiểm không kém: Trưởng bộ phận chuẩn bị của OpenAI đã công khai tuyên bố rằng việc chèn mã lệnh gián tiếp vào lời nhắc (indirect prompt injection) đối với các tác tử trình duyệt "không phải là một lỗi có thể vá triệt để bằng phần mềm". Các cuộc tấn công được ghi nhận trong giai đoạn 2025–2026 bao gồm: Tainted Memories (CSRF qua Atlas), HashJack (Cato Networks), và các vụ chiếm quyền chỉ bằng một cú nhấp chuột (one-click hijacks) trên Perplexity Comet.

**Loại bài học:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python (stdlib, mô hình bề mặt tấn công chèn mã lệnh gián tiếp)
**Điều kiện tiên quyết:** Phase 15 · 10 (Các chế độ cấp quyền), Phase 15 · 01 (Tác tử chân trời dài hạn)
**Thời gian:** ~45 phút

## Vấn đề thực tế

Một tác tử trình duyệt (browser agent) là một tác tử chân trời dài hạn (long-horizon agent) đọc nội dung không đáng tin cậy và thực thi các hành động mang lại hậu quả thực tế. Mọi trang web mà tác tử ghé thăm đều là dữ liệu đầu vào mà người dùng không tự tay viết ra. Mọi biểu mẫu trên mỗi trang đều là một kênh nhận lệnh tiềm ẩn. Kho dữ liệu các cuộc tấn công giai đoạn 2025–2026 cho thấy đây không còn là giả định lý thuyết: Tainted Memories cho phép kẻ tấn công gán các chỉ thị độc hại vào bộ nhớ của tác tử thông qua một trang web được thiết kế đặc biệt; HashJack giấu các lệnh điều khiển bên trong các đoạn phân đoạn URL (`#fragment`) mà tác tử truy cập; các vụ chiếm quyền trên Perplexity Comet chỉ cần đúng một cú nhấp chuột.

Bức tranh phòng thủ vô cùng nan giải. Trưởng bộ phận chuẩn bị của OpenAI đã nói thẳng sự thật: chèn mã lệnh gián tiếp "không phải là một lỗi có thể vá triệt để bằng mã". Nguyên nhân là bởi cuộc tấn công diễn ra ngay tại ranh giới giữa việc đọc và việc hành động của tác tử, một ranh giới vốn dĩ rất mờ nhạt về mặt kiến trúc — mọi token mà mô hình đọc vào, về mặt nguyên lý, đều có thể bị hiểu như một chỉ thị mệnh lệnh.

Bài học này định danh rõ ràng bề mặt tấn công, phác thảo bức tranh các bộ tiêu chuẩn đánh giá (BrowseComp, OSWorld, WebArena-Verified), và mô hình hóa một kịch bản chèn mã gián tiếp tối giản để bạn có thể tư duy thấu đáo về các giải pháp phòng thủ thực tế trong Bài 14 và 18.

## Khái niệm cốt lõi

### Toàn cảnh năm 2026 qua từng hệ thống

**ChatGPT agent (OpenAI).** Ra mắt tháng 7 năm 2025. Hợp nhất Operator (duyệt web) và Deep Research (nghiên cứu kéo dài nhiều giờ). Đóng cửa công cụ Operator độc lập vào ngày 31 tháng 8 năm 2025. Đạt SOTA trên BrowseComp với 68.9%; đạt chỉ số ấn tượng trên OSWorld và WebArena-Verified.

**Claude Sonnet + Vercept (Anthropic).** Việc Anthropic thâu tóm Vercept tập trung trực tiếp vào năng lực sử dụng máy tính (computer-use). Nâng điểm Claude Sonnet trên OSWorld từ <15% lên 72.5%. Tính năng Claude Computer Use được phát hành dưới dạng API công cụ.

**Gemini 3 Pro kết hợp Browser Use (DeepMind).** Tích hợp Browser Use cung cấp các cơ chế điều khiển sử dụng máy tính; FSF v3 (tháng 4 năm 2026, Bài 20) theo dõi quyền tự chủ trong miền R&D học máy nói riêng.

**WebArena-Verified (ServiceNow, ICLR 2026).** Khắc phục một bài toán thực tế đã được tài liệu hóa rõ ràng: WebArena gốc có tỷ lệ âm tính giả ~11.3% (các tác vụ thực tế đã được giải quyết nhưng bị chấm điểm là thất bại). Bản phát hành Verified chấm điểm lại với các tiêu chí thành công do con người thẩm định và bổ sung thêm tập con Khó (Hard subset) gồm 258 tác vụ (bài báo ICLR 2026, openreview.net/forum?id=94tlGxmqkN).

### So sánh BrowseComp, OSWorld và WebArena

| Bộ benchmark | Đo lường điều gì | Chân trời thời gian |
|---|---|---|
| BrowseComp | Tìm kiếm các sự kiện cụ thể trên web mở dưới áp lực thời gian | Vài phút |
| OSWorld | Tác tử điều khiển toàn bộ môi trường máy tính để bàn (chuột, bàn phím, shell) | Hàng chục phút |
| WebArena-Verified | Các tác vụ giao dịch trên web trong các trang web mô phỏng | Vài phút |
| Tập con Khó (Hard subset) | Các tác vụ WebArena-Verified đòi hỏi chuyển đổi trạng thái qua nhiều trang | Hàng chục phút |

Các trục đo lường hoàn toàn khác nhau. Điểm số BrowseComp cao chỉ chứng minh tác tử tìm thông tin giỏi; không có nghĩa là tác tử có thể đặt một vé máy bay. Điểm số OSWorld phản ánh sát hơn câu hỏi "nó có vận hành được trên máy tính của tôi không". WebArena-Verified tiệm cận câu hỏi "nó có thể hoàn tất một luồng quy trình hay không". Mọi quyết định đưa vào sản xuất đều phải dựa trên bộ benchmark khớp với phân phối tác vụ thực tế.

### Bề mặt tấn công được định danh cụ thể

1. **Chèn mã lệnh vào lời nhắc gián tiếp (Indirect prompt injection).** Nội dung trang web không đáng tin cậy chứa các chỉ thị. Tác tử đọc chúng. Tác tử thực thi chúng. Các ví dụ công khai: Kai Greshake và cộng sự (2024), bài báo Tainted Memories (2025), HashJack của Cato Networks (2026).
2. **Chèn mã qua đoạn URL / chuỗi truy vấn (URL fragment / query injection).** Đoạn `#fragment` hoặc chuỗi truy vấn của một URL được thu thập chứa các lệnh điều khiển. Chúng không bao giờ hiển thị trên giao diện người dùng; nhưng vẫn nằm trọn trong ngữ cảnh của tác tử.
3. **Các cuộc tấn công ràng buộc bộ nhớ (Memory-binding attacks).** Trang web ra lệnh cho tác tử ghi vào bộ nhớ bền vững (Bài 12 đề cập đến trạng thái bền bỉ). Trong phiên làm việc tiếp theo, mục bộ nhớ này sẽ kích hoạt mã độc mà không cần bất kỳ tác nhân kích hoạt rõ ràng nào.
4. **Các cuộc tấn công dạng CSRF trên các phiên đã xác thực.** Lớp tấn công kiểu Tainted Memories: tác tử đang đăng nhập vào một dịch vụ; trang web của kẻ tấn công phát ra các yêu cầu làm thay đổi trạng thái và tác tử thực thi chúng bằng cookie đăng nhập của chính người dùng.
5. **Chiếm quyền chỉ bằng một cú nhấp chuột (One-click hijack).** Một nút bấm nhìn bề ngoài vô hại lại mang theo tải trọng mã độc mà tác tử sẽ bám theo thực thi. Tiêu biểu là lớp tấn công trên Comet.
6. **Lỗ hổng Chính sách Bảo mật Nội dung (CSP) trên bề mặt máy chủ của tác tử.** Bản thân tầng hiển thị giao diện và tầng công cụ có thể trở thành véc-tơ tấn công; ngăn xếp "trình duyệt bên trong tác tử trình duyệt" có phạm vi rất rộng.

### Tại sao "không thể vá triệt để bằng phần mềm"

Cuộc tấn công mang tính đồng hình (isomorphic) với chính năng lực cốt lõi của tác tử. Tác tử bắt buộc phải đọc nội dung không đáng tin cậy để hoàn thành công việc. Bất kỳ nội dung nào tác tử đọc vào đều có thể chứa chỉ thị lệnh. Bất kỳ chỉ thị nào tác tử tuân theo đều có nguy cơ đi lệch khỏi yêu cầu thực tế của người dùng. Các biện pháp phòng thủ (ranh giới tin cậy, bộ phân loại, danh sách cho phép công cụ, sự can thiệp của con người HITL trên các hành động hệ quả) chỉ giúp tăng chi phí tấn công và thu hẹp bán kính tác động. Chúng không thể loại bỏ hoàn toàn lớp tấn công này.

Đây chính là mô hình lập luận tương tự định lý Löb (Bài 8): tác tử không thể chứng minh token tiếp theo là an toàn tuyệt đối; nó chỉ có thể thiết lập một hệ thống sao cho các token không an toàn trở nên dễ phát hiện hơn.

### Tư thế phòng thủ thực tế trong sản xuất

- **Ranh giới đọc / ghi (Read / write boundary).** Hành động đọc không bao giờ tạo ra hệ quả nghiêm trọng. Hành động ghi (gửi biểu mẫu, đăng nội dung, gọi công cụ có tác dụng phụ) bắt buộc phải có sự phê duyệt trực tiếp của con người nếu nội dung khởi tạo đến từ bên ngoài ranh giới tin cậy.
- **Danh sách công cụ cho phép theo từng tác vụ.** Tác tử có thể duyệt web; nhưng nó không thể thực hiện lệnh chuyển tiền trừ khi công cụ đó được kích hoạt tường minh cho tác vụ này. Bài 13 đề cập chi tiết về ngân sách.
- **Cô lập phiên làm việc (Session isolation).** Các phiên của tác tử trình duyệt chỉ chạy với thông tin xác thực có phạm vi giới hạn. Tuyệt đối không dùng thông tin xác thực môi trường sản xuất, không dùng email cá nhân. Nhật ký của mọi yêu cầu HTTP phải được lưu giữ để phục vụ kiểm toán.
- **Bộ làm sạch nội dung (Content sanitizer).** Mã HTML thu thập về phải được loại bỏ các mẫu độc hại đã biết trước khi ghép nối vào ngữ cảnh của mô hình. (Giúp ngăn chặn các cuộc tấn công đơn giản; nhưng không ngăn được các tải trọng mã độc tinh vi).
- **Con người can thiệp (HITL) đối với các hành động hệ quả.** Mô hình đề xuất rồi mới xác nhận (propose-then-commit) (Bài 15).
- **Token cảnh báo (canary tokens) trong bộ nhớ.** Nếu một mục bộ nhớ bị kích hoạt bất thường, người dùng sẽ lập tức nhận được cảnh báo (Bài 14).

```figure
injection-boundary
```

## Sử dụng trong thực tế

Tệp `code/main.py` mô hình hóa một chu trình chạy tác tử trình duyệt tối giản trên ba trang web nhân tạo. Một trang web lành tính, một trang chứa khối chèn mã trực diện trong văn bản hiển thị, và một trang chứa mã độc chèn qua đoạn URL (không hiển thị trên giao diện nhưng nằm trong ngữ cảnh của tác tử). Kịch bản chứng minh: (a) một tác tử ngây thơ sẽ hành xử thế nào, (b) ranh giới đọc/ghi ngăn chặn được gì, (c) bộ làm sạch lọc được gì, và (d) những gì lọt qua cả hai lớp phòng thủ.

## Đóng gói sản phẩm

Tệp `outputs/skill-browser-agent-trust-boundary.md` phân định phạm vi triển khai tác tử trình duyệt: các vùng tin cậy mà nó chạm tới, những gì nó được phép ghi, và các biện pháp bảo vệ bắt buộc phải thiết lập trước lần chạy đầu tiên.

## Bài tập

1. Chạy `code/main.py`. Xác định cuộc tấn công nào bộ làm sạch phát hiện được nhưng ranh giới đọc/ghi bỏ qua, và cuộc tấn công nào chỉ có ranh giới đọc/ghi mới ngăn chặn được.

2. Mở rộng bộ làm sạch để phát hiện một lớp tấn công chèn qua đoạn phân đoạn URL kiểu HashJack. Đo lường tỷ lệ dương tính giả trên các URL lành tính có chứa phân đoạn hợp lệ.

3. Chọn một quy trình tác tử trình duyệt thực tế mà bạn biết (ví dụ: "đặt vé máy bay"). Liệt kê mọi thao tác đọc và mọi thao tác ghi. Đánh dấu thao tác ghi nào bắt buộc phải có sự can thiệp của con người (HITL) và giải thích lý do.

4. Đọc bài báo ICLR 2026 về WebArena-Verified. Xác định một nhóm tác vụ mà cách chấm điểm của WebArena gốc không đáng tin cậy và giải thích cách tập con Verified khắc phục vấn đề đó.

5. Thiết kế một token cảnh báo bộ nhớ (memory canary) cho môi trường tác tử trình duyệt. Bạn sẽ lưu trữ thông tin gì, ở đâu, và điều gì sẽ kích hoạt chuông cảnh báo?

## Thuật ngữ chính

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Indirect prompt injection | "Văn bản trang web độc hại" | Nội dung không tin cậy trên trang mà tác tử đọc chứa chỉ thị mà tác tử thực thi |
| Tainted Memories | "Tấn công bộ nhớ" | Tác tử ghi chỉ thị do kẻ tấn công cung cấp vào bộ nhớ bền vững; kích hoạt ở phiên sau |
| HashJack | "Tấn công phân đoạn URL" | Tải trọng ẩn trong `#fragment` hoặc chuỗi truy vấn nằm trong ngữ cảnh tác tử nhưng không hiển thị |
| One-click hijack | "Nút bấm độc hại" | Thành phần tương tác trực quan mang theo tải trọng mã độc mà tác tử bám theo thực thi |
| BrowseComp | "Benchmark tìm kiếm web" | Tìm các dữ kiện cụ thể trên web mở; chân trời thời gian cấp độ vài phút |
| OSWorld | "Benchmark máy tính để bàn" | Kiểm soát toàn bộ hệ điều hành; các tác vụ giao diện đồ họa (GUI) nhiều bước |
| WebArena-Verified | "Benchmark tác vụ web chuẩn hóa" | Phiên bản WebArena được ServiceNow chấm điểm lại kèm tập con Khó (Hard subset) |
| Read/write boundary | "Cổng kiểm soát tác dụng phụ" | Đọc không bao giờ tạo hệ quả; ghi đòi hỏi phê duyệt mới nếu nội dung nằm ngoài ranh giới tin cậy |

## Đọc thêm

- [OpenAI — Giới thiệu ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — sự hợp nhất giữa Operator và nghiên cứu chuyên sâu; kỷ lục BrowseComp SOTA.
- [OpenAI — Tác tử sử dụng máy tính (Computer-Using Agent)](https://openai.com/index/computer-using-agent/) — dòng phát triển Operator và kiến trúc nền tảng của ChatGPT agent.
- [Zhou và cộng sự — WebArena](https://webarena.dev/) — bộ benchmark gốc.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — bài báo ICLR 2026 về tập con chuẩn hóa.
- [Anthropic — Đo lường quyền tự chủ của tác tử trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — phân tích chi tiết về bề mặt tấn công của các tác tử sử dụng máy tính.
