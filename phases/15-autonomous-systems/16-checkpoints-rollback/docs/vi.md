# Điểm kiểm tra và Khôi phục Trạng thái (Checkpoints and Rollback)

> Mọi bước chuyển dịch trạng thái đồ thị (graph-state transition) đều phải được lưu trữ bền bỉ. Khi một worker bị sự cố (crash), hợp đồng thuê (lease) của nó sẽ hết hạn và một worker khác sẽ tiếp quản công việc tại điểm kiểm tra (checkpoint) mới nhất. Các Cloudflare Durable Objects duy trì trạng thái xuyên suốt nhiều giờ hoặc nhiều tuần. Mô hình đề xuất rồi cam kết (propose-then-commit, Bài 15) định nghĩa một kế hoạch khôi phục (rollback plan) cho từng hành động. Bước xác minh sau hành động (post-action verification) đóng kín vòng lặp an toàn. Điều 14 Đạo luật AI của EU quy định bắt buộc phải có sự giám sát hiệu quả của con người đối với các hệ thống rủi ro cao — trên thực tế vận hành, điều này đồng nghĩa với việc các điểm kiểm tra phải truy vấn được, các phương án khôi phục phải được diễn tập trước, và dấu vết kiểm toán (audit trail) phải tồn tại bền vững qua các lần triển khai (deploy). Dạng thức hỏng hóc nghiêm trọng nhất: nếu thiếu các khóa lũy đẳng (idempotency keys) và kiểm tra điều kiện tiên quyết (precondition checks), một lần thử lại sau lỗi tạm thời có thể dẫn tới việc thực thi lặp lại hai lần một hành động đã được phê duyệt. Bước xác minh sau hành động chính là chốt chặn để phát hiện sự cố này.

**Loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, máy trạng thái lưu điểm kiểm tra và khôi phục)
**Điều kiện tiên quyết:** Phase 15 · 12 (Thực thi bền bỉ), Phase 15 · 15 (Đề xuất rồi cam kết)
**Thời lượng:** ~60 phút

## Vấn đề đặt ra (The Problem)

Cơ chế thực thi bền bỉ (durable execution, Bài 12) giúp một agent bị sập có thể tiếp tục hoạt động trở lại. Cơ chế đề xuất rồi cam kết (propose-then-commit, Bài 15) đảm bảo một hành động đã được phê duyệt có thể kiểm toán được. Bài học này kết nối hai khái niệm đó: điều gì sẽ xảy ra khi một hành động đã duyệt mới chỉ thực thi một phần, hệ thống bị sập, rồi sau đó khôi phục lại? Khi nào quy trình rollback sẽ chạy, và chạy đối chiếu với trạng thái nào?

Các hệ thống thực tế kết nối các thành phần này theo những cách khác nhau:

- **LangGraph** lưu điểm kiểm tra (checkpoints) cho mọi chuyển dịch trạng thái đồ thị vào PostgreSQL. Khi một worker bị sập, quyền chiếm dụng (lease) được giải phóng và một worker khác sẽ tiếp tục từ checkpoint gần nhất. Các quy trình tạm dừng khi gặp `interrupt()`, và chính lệnh ngắt này cũng được lưu bền bỉ.
- **Cloudflare Durable Objects** lưu giữ trạng thái theo từng khóa (per-key state) xuyên suốt hàng giờ hoặc hàng tuần. Đặt tác vụ tính toán ngay cùng vị trí vật lý với kho lưu trữ cho hành động đã duyệt.
- **Microsoft Agent Framework** cung cấp các nguyên hàm `Checkpoint` trong API quy trình; cơ chế phát lại (replay) kết hợp với tính lũy đẳng (idempotency) giúp xử lý các lần thử lại an toàn.

Trong mọi trường hợp, công thức thực sự hoạt động hiệu quả là: khóa lũy đẳng (ngăn chặn thực thi trùng lặp) + kiểm tra điều kiện tiên quyết (đảm bảo trạng thái vẫn đúng với những gì đã duyệt) + xác minh sau hành động (khẳng định tác dụng phụ thực sự đã diễn ra) + kích hoạt rollback nếu bước xác minh thất bại.

## Khái niệm cốt lõi (The Concept)

### Mọi bước chuyển dịch trạng thái đều phải lưu bền bỉ (Every transition persists)

Một bước chuyển dịch trạng thái đồ thị (graph-state transition) là bất kỳ bước nào làm dịch chuyển quy trình từ một trạng thái định danh này sang một trạng thái định danh khác. Các cách hiện thực ngây thơ thường chỉ lưu trữ bền vững tại các điểm commit cụ thể; các hệ thống sản xuất thực tế lưu trữ sau từng bước chuyển dịch. Chi phí phát sinh (thêm một vài thao tác ghi đĩa) là rất nhỏ so với độ tin cậy thu được (có thể phát lại từ bất kỳ đâu, khôi phục quyền chiếm dụng chuẩn xác).

### Thu hồi quyền chiếm dụng (Lease recovery)

Khi một worker gặp sự cố dừng đột ngột, quy trình công việc không hề bị mất; hợp đồng thuê / quyền chiếm dụng (lease — một cam kết ngắn hạn chứng nhận worker này đang thực thi phiên chạy này) đơn giản sẽ hết hạn. Một worker khác sẽ tiếp nhận điểm kiểm tra mới nhất và tiếp tục phiên làm việc. Cơ chế cấp quyền chiếm dụng này chính là thứ giúp các hệ thống production vượt qua các đợt phát hành triển khai cuốn chiếu (rolling deploys) mà không làm thất thoát các công việc đang dang dở.

### Tính lũy đẳng kết hợp kiểm tra điều kiện tiên quyết (Idempotency plus preconditions)

Chỉ riêng tính lũy đẳng là chưa đủ. Hãy xem xét trường hợp: một quy trình được phê duyệt để "chuyển $100 từ tài khoản A sang B khi số dư > $1000." Quy trình được commit, gặp sự cố khi đang thực thi dở dang, rồi khôi phục lại. Nếu hệ thống chỉ kiểm tra khóa lũy đẳng và tiếp tục thực thi, giao dịch chuyển tiền sẽ chạy đúng một lần (chính xác). Nhưng hãy hình dung trong khoảng thời gian giữa lúc hệ thống sập và lúc khôi phục, số dư của tài khoản A đã giảm xuống còn $500 do một quy trình khác can thiệp. Bước kiểm tra khóa lũy đẳng vẫn vượt qua; nhưng điều kiện tiên quyết (precondition) đã bị vi phạm. Nếu không kiểm tra điều kiện tiên quyết, hệ thống sẽ thực hiện một giao dịch thấu chi bất hợp lệ.

Mọi hành động hệ trọng đều đòi hỏi cả hai lớp bảo vệ:

- **Khóa lũy đẳng (Idempotency key)**: ngăn chặn việc thực thi lặp lại hai lần.
- **Kiểm tra điều kiện tiên quyết (Precondition check)**: xác nhận trạng thái hiện tại vẫn hoàn toàn nhất quán với những gì đã được phê duyệt.

### Xác minh sau hành động (Post-action verification)

"Công cụ trả về mã 200" không đồng nghĩa với việc đã xác minh. Quá trình xác minh thực thụ phải đọc lại trạng thái của hệ sinh thái đích và khẳng định tác dụng phụ thực sự đã xảy ra. Các mẫu thiết kế:

- Cập nhật cơ sở dữ liệu: dùng cú pháp `UPDATE ... RETURNING *`, sau đó assert rằng hàng dữ liệu trả về khớp hoàn toàn với trạng thái mong muốn.
- Gửi email: kiểm tra thư mục Đã gửi (sent folder) để tìm ID thư sau khi gửi.
- Ghi tệp: đọc lại tệp từ đĩa và tính toán mã băm (hash).
- Gọi API: thực hiện một lệnh `GET` tiếp theo trên tài nguyên đích để kiểm chứng.

Nếu bước xác minh thất bại, quy trình công việc rơi vào trạng thái lỗi đã biết. Cơ chế khôi phục (rollback) lập tức được kích hoạt.

### Các kế hoạch khôi phục (Rollback plans)

Mọi hành động hệ trọng trong mô hình propose-then-commit (Bài 15) đều phải mang theo một kế hoạch khôi phục. Các phân loại:

- **Khôi phục nội tuyến (In-band rollback)**: trực tiếp đảo ngược tác dụng phụ (`DELETE` sau khi `INSERT`, gửi email đính chính sau khi gửi nhầm thư).
- **Giao dịch bù trừ (Compensating transaction)**: một hành động mới có tác dụng trung hòa hành động ban đầu (mô hình mẫu SAGA chuẩn).
- **Khôi phục ngoại tuyến (Out-of-band rollback)**: phát tín hiệu cảnh báo cho con người, tạm dừng quy trình, giữ nguyên trạng thái lỗi để phục vụ điều tra.

Kế hoạch khôi phục dạng no-op ("chúng ta không thể hoàn tác hành động này") bắt buộc phải được nêu đích danh trong bản đề xuất. Các hành động không thể hoàn tác đòi hỏi cơ chế HITL chặt chẽ hơn tại thời điểm commit (hỏi-đáp thử thách ở Bài 15).

### Góc nhìn vận hành về Điều 14 Đạo luật AI của EU

Điều 14 yêu cầu "sự giám sát hiệu quả của con người" đối với các hệ thống rủi ro cao. Về mặt vận hành thực tế, các kỹ sư triển khai hiểu yêu cầu này là:

- Các điểm kiểm tra (checkpoints) phải có khả năng truy vấn được bởi kiểm toán viên.
- Các phương án khôi phục (rollbacks) phải được diễn tập (kiểm thử đầu-cuối end-to-end ít nhất một lần).
- Dấu vết kiểm toán (audit trail) phải tồn tại bền vững qua các lần deploy (hạ tầng lưu checkpoint không được dùng bộ nhớ tạm bợ).
- Các bước xác minh thất bại phải được phát cảnh báo chủ động, chứ không được ghi log âm thầm.

Một quy trình bị sập giữa lúc commit, khôi phục lại và hoàn thành tác dụng phụ mà không có lộ trình xác minh + khôi phục sẽ không thể vượt qua bài kiểm tra tuân thủ Điều 14.

### Dạng thức hỏng hóc nghiêm trọng: Thực thi trùng lặp (The sharp failure mode: the double-execute)

Sự cố thường gặp nhất trong môi trường production thuộc lĩnh vực này:

1. Hành động được duyệt với khóa lũy đẳng `k`.
2. Lệnh commit bắt đầu, thực thi thành công, trả về mã 200.
3. Quy trình bị sập trước khi kịp lưu bền bỉ trạng thái "committed".
4. Quy trình khởi động lại; thấy trạng thái "đã duyệt nhưng chưa commit"; thực thi lại hành động.
5. Tác dụng phụ bị kích hoạt lần thứ hai.

Biện pháp giảm thiểu: lưu bền bỉ một ý định "đang thực hiện" (in-flight) trước khi thực thi, thực thi với khóa lũy đẳng, sau đó chỉ đánh dấu "committed" sau khi bước xác minh sau hành động thành công. Nếu hành động đã chạy nhưng việc ghi trạng thái thất bại, bạn có thể dựa vào bước xác minh để nhận biết và xử lý. Nếu việc ghi trạng thái thành công nhưng hành động thất bại, bạn dựa vào bước xác minh để thực thi đúng một lần thông qua lộ trình khôi phục.

```figure
checkpoint-replay
```

## Thực hành (Use It)

Tệp `code/main.py` hiện thực hóa một quy trình công việc có lưu checkpoint với đầy đủ tính lũy đẳng, điều kiện tiên quyết, xác minh và khôi phục. Trình điều khiển mô phỏng bốn kịch bản: phiên chạy suôn sẻ, thử lại sau sự cố (bộ lọc lũy đẳng chặn đứng trùng lặp), điều kiện tiên quyết thất bại (quy trình hủy bỏ mà không kích hoạt hành động), và bước xác minh thất bại (quy trình rollback tự động kích hoạt).

## Đóng gói (Ship It)

Tệp `outputs/skill-rollback-rehearsal.md` thiết kế một bài kiểm thử diễn tập khôi phục (rollback-rehearsal test) cho một quy trình dự kiến và kiểm toán hạ tầng lưu trữ checkpoint nhằm đảm bảo dấu vết kiểm toán bền vững.

## Bài tập (Exercises)

1. Chạy `code/main.py`. Xác minh bốn kịch bản mô phỏng. Đối với trường hợp gặp sự cố trong khi commit, hãy khẳng định hành động chỉ được kích hoạt chính xác một lần xuyên suốt các lần thử lại.

2. Chỉnh sửa mô hình "đánh dấu hoàn tất trước, rồi mới thực thi" sao cho thao tác ghi trạng thái diễn ra sau hành động. Chạy lại kịch bản gặp sự cố sập hệ thống. Đo lường xem có bao nhiêu hành động trùng lặp bị kích hoạt.

3. Thiết kế một kế hoạch khôi phục cho một hành động production cụ thể (ví dụ: "đăng bài lên một kênh Slack"). Phân loại kế hoạch đó thuộc dạng khôi phục nội tuyến (in-band), giao dịch bù trừ (compensating), hay ngoại tuyến (out-of-band). Biện minh cho sự lựa chọn của bạn.

4. Chọn một quy trình công việc mà bạn quen thuộc. Xác định từng bước chuyển dịch trạng thái. Đánh dấu từng bước với yêu cầu về tính bền bỉ (phải lưu / không cần lưu). Đếm xem có bao nhiêu bước chuyển dịch mà hiện tại bạn đang không lưu bền bỉ.

5. Bài kiểm thử diễn tập khôi phục: thiết kế một bài test đầu-cuối (end-to-end test) thực thi một workflow thực tế, làm sập quy trình giữa chừng và xác nhận lộ trình rollback được kích hoạt chuẩn xác. Bài kiểm thử này cần assert những điều kiện gì?

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Checkpoint | "Điểm lưu" | Mọi chuyển dịch trạng thái đồ thị được lưu vào kho bền bỉ |
| Lease | "Quyền chiếm dụng worker" | Cam kết ngắn hạn chứng nhận worker đang chạy một run; hết hạn khi sập |
| Precondition | "Cổng kiểm tra trạng thái" | Khẳng định trạng thái hiện tại vẫn nhất quán với hành động đã duyệt |
| Post-action verify | "Kiểm tra đọc lại" | Xác nhận tác dụng phụ thực sự đã xảy ra trong hệ thống đích |
| In-band rollback | "Hoàn tác trực tiếp" | Đảo ngược tác dụng phụ bằng thao tác nghịch đảo |
| Compensating transaction | "Hoàn tác kiểu SAGA" | Một hành động mới nhằm trung hòa tác động của hành động cũ |
| Mark-as-done-first | "Thứ tự ghi trạng thái" | Lưu trạng thái committed trước khi trả về từ hàm commit |
| Article 14 | "Giám sát con người EU AI Act" | Yêu cầu vận hành: checkpoint truy vấn được, rollback diễn tập được, dấu vết kiểm toán |

## Đọc thêm (Further Reading)

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các nguyên hàm checkpoint và khôi phục quyền chiếm dụng.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects như một tầng trạng thái bền bỉ.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — đường cơ sở pháp lý.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn về độ tin cậy cho các quy trình chân trời dài hạn.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — cấu trúc quy trình công việc cho Claude Code Routines.
