# Con người Tham gia Vòng lặp: Đề xuất rồi Cam kết (Human-in-the-Loop: Propose-Then-Commit)

> Đồng thuận kỹ thuật năm 2026 về con người tham gia vòng lặp (Human-in-the-Loop - HITL) mang tính chuẩn mực và cụ thể. Nó không đơn thuần là "agent hỏi, người dùng nhấn Duyệt". Đó là mô hình đề xuất rồi cam kết (propose-then-commit): hành động dự kiến được lưu bền vững (persisted) vào một kho lưu trữ bền bỉ kèm theo một khóa lũy đẳng (idempotency key); hiển thị trực quan cho người đánh giá với đầy đủ ý định (intent), nguồn gốc dòng dữ liệu (data lineage), các quyền hạn bị tác động (permissions touched), bán kính ảnh hưởng (blast radius), và phương án khôi phục (rollback plan); chỉ được cam kết (commit) thực thi sau khi nhận được sự xác nhận đồng thuận chủ động; và được xác minh (verify) sau khi thực thi để khẳng định tác dụng phụ (side effect) thực sự đã diễn ra. Lệnh `interrupt()` của LangGraph kết hợp với cơ chế lưu điểm kiểm tra (checkpointing) trong PostgreSQL, sự kiện `RequestInfoEvent` trong Microsoft Agent Framework, và hàm `waitForApproval()` của Cloudflare đều hiện thực hóa cùng một khuôn mẫu thiết kế này. Dạng thức hỏng hóc kinh điển là hiện tượng phê duyệt đóng dấu cao su (rubber-stamp approval): nút "Approve?" bị nhấp chuột mà không qua xem xét kỹ lưỡng. Biện pháp giảm thiểu đã được tài liệu hóa là cơ chế hỏi-đáp thử thách (challenge-and-response) đi kèm một bảng kiểm (checklist) rõ ràng.

**Loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, máy trạng thái đề xuất rồi cam kết với tính lũy đẳng)
**Điều kiện tiên quyết:** Phase 15 · 12 (Thực thi bền bỉ), Phase 15 · 14 (Bẫy dây vấp / Tripwires)
**Thời lượng:** ~60 phút

## Vấn đề đặt ra (The Problem)

Một agent đưa ra một hành động. Người dùng phải đưa ra quyết định: phê duyệt hay từ chối. Nếu quyết định được đưa ra ngay tức thì, rất có thể đó không phải là một sự đánh giá thực chất. Nếu quyết định được cấu trúc hóa chặt chẽ, quá trình sẽ diễn ra chậm hơn nhưng đáng tin cậy. Câu hỏi kỹ thuật ở đây là làm thế nào để biến việc đánh giá có cấu trúc trở thành lộ trình có ít lực cản nhất (path of least resistance).

Khuôn mẫu HITL thời kỳ năm 2023 là một hộp thoại nhắc nhở đồng bộ: "Agent muốn gửi email tới X với nội dung Y — bạn có phê duyệt không?" Người dùng nhấn Duyệt (Approve). Ai cũng cảm thấy hệ thống dường như an toàn. Nhưng trên thực tế, bề mặt tương tác này bị lạm dụng phê duyệt hình thức (rubber-stamped) nghiêm trọng: người dùng duyệt rất nhanh, sự phê duyệt gần như không phản ánh được tính an toàn, và khi agent gặp sự cố, nhật ký kiểm toán (audit trail) chỉ cho thấy một lịch sử dài dằng dặc các lượt phê duyệt mà chính người dùng cũng không nhớ nổi mình đã xem xét những gì.

Mô hình của năm 2026 — đề xuất rồi cam kết (propose-then-commit) — dịch chuyển HITL lên một nền tảng bền bỉ (durable substrate), gắn kèm siêu dữ liệu có cấu trúc và đòi hỏi cam kết phê duyệt chủ động. Mọi bộ SDK agent hiện đại đều cung cấp một phiên bản của mô hình này: `interrupt()` trong LangGraph, `RequestInfoEvent` trong Microsoft Agent Framework, `waitForApproval()` trong Cloudflare. Tên gọi API có thể khác biệt, nhưng khuôn mẫu kiến trúc là đồng nhất.

## Khái niệm cốt lõi (The Concept)

### Máy trạng thái đề xuất rồi cam kết (The propose-then-commit state machine)

1. **Đề xuất (Propose).** Agent tạo ra một hành động đề xuất. Lưu trữ bền bỉ vào kho dữ liệu (PostgreSQL, Redis, Durable Object). Bản ghi bao gồm:
   - Ý định (intent) (tại sao agent lại thực hiện hành động này)
   - Nguồn gốc dòng dữ liệu (data lineage) (nguồn dữ liệu nào đã dẫn tới đề xuất này)
   - Quyền hạn bị tác động (permissions touched) (những phạm vi / tệp tin / điểm cuối API nào bị ảnh hưởng)
   - Bán kính ảnh hưởng (blast radius) (kịch bản tồi tệ nhất có thể xảy ra là gì)
   - Kế hoạch khôi phục (rollback plan) (nếu cam kết thực hiện, làm thế nào để hoàn tác)
   - Khóa lũy đẳng (idempotency key) (duy nhất cho mỗi đề xuất; gửi lại yêu cầu sẽ trả về cùng một bản ghi)
2. **Hiển thị trực quan (Surface).** Người đánh giá xem xét đề xuất với đầy đủ siêu dữ liệu. Người đánh giá phải là con người thực tế (không phải agent tự đánh giá lại chính nó).
3. **Cam kết (Commit).** Nhận sự xác nhận đồng thuận chủ động. Hành động được kích hoạt thực thi.
4. **Xác minh (Verify).** Sau khi thực thi, tác dụng phụ (side effect) được đọc lại và xác nhận trực tiếp từ hệ thống đích. Nếu bước xác minh thất bại, hệ thống rơi vào trạng thái lỗi đã biết và kích hoạt cơ chế cảnh báo.

### Khóa lũy đẳng (The idempotency key)

Nếu không có khóa lũy đẳng (idempotency key), một lần thử lại (retry) sau một lỗi tạm thời (transient failure) có thể dẫn tới việc thực thi lặp lại hai lần một hành động đã được duyệt. Ví dụ cụ thể: người dùng phê duyệt "chuyển $100 từ A sang B". Mạng chập chờn. Quy trình thử lại. Người dùng chỉ phê duyệt một lần nhưng lệnh chuyển tiền lại bị thực thi hai lần. Khóa lũy đẳng ràng buộc phê duyệt vào một tác dụng phụ duy nhất; lần thực thi thứ hai sẽ trở thành một phép toán rỗng không gây tác dụng phụ (no-op).

Đây chính là mẫu lũy đẳng tương tự mà các API của Stripe và AWS áp dụng. Việc tái sử dụng nó cho các phê duyệt của agent được nêu rõ ràng trong tài liệu của Microsoft Agent Framework.

### Tính bền bỉ: Tại sao phê duyệt tồn tại lâu hơn tiến trình

Phòng chờ phê duyệt (approval waiting room) là một phần trạng thái mà agent không sở hữu. Quy trình công việc được tạm dừng (paused, Bài 12). Khi có phê duyệt, quy trình sẽ tiếp tục lại từ đúng điểm dừng đó. Đây là lý do tại sao LangGraph kết hợp `interrupt()` với lưu điểm kiểm tra trên PostgreSQL chứ không chỉ dựa vào trạng thái trong bộ nhớ RAM — một phê duyệt đến sau đó hai ngày vẫn tìm thấy quy trình thực thi hoàn toàn nguyên vẹn.

### Phê duyệt đóng dấu cao su và giải pháp hỏi-đáp thử thách (Rubber-stamp approvals and challenge-and-response)

Giao diện người dùng mặc định cho HITL (chỉ có hai nút "Duyệt" / "Từ chối") dẫn tới các phê duyệt chóng vánh mà không qua thẩm định thực sự. Giải pháp đã được ghi nhận: một bảng kiểm dạng hỏi-đáp thử thách (challenge-and-response checklist) yêu cầu các câu trả lời khẳng định cho các câu hỏi cụ thể trước khi nút Duyệt được kích hoạt. Hình thức cụ thể:

- "Bạn có hiểu tài nguyên nào đang bị tác động không? [ ]"
- "Bạn đã xác minh bán kính ảnh hưởng nằm trong mức chấp nhận được chưa? [ ]"
- "Bạn đã có kế hoạch khôi phục dự phòng nếu thao tác này thất bại chưa? [ ]"

Đây không phải là sự quan liêu thủ tục đơn thuần — mà là một hàm ép buộc hành vi (forcing function). Người đánh giá không thể tích vào các ô kiểm sẽ phải yêu cầu làm rõ (chuyển tiếp báo động / escalation) hoặc từ chối hành động (mặc định an toàn / safe default). Nghiên cứu an toàn agent của Anthropic trích dẫn rõ ràng HITL dựa trên bảng kiểm như một cơ chế giảm thiểu trực tiếp tình trạng phê duyệt đóng dấu cao su.

### Thế nào được coi là hành động hệ trọng (consequential)

Không phải mọi hành động đều cần đến cơ chế propose-then-commit. Hướng dẫn kỹ thuật năm 2026 phân loại:

- **Các hành động hệ trọng (Consequential actions)** (luôn bắt buộc HITL): ghi dữ liệu không thể hoàn tác, giao dịch tài chính, truyền thông ra ngoài hệ thống, thay đổi cơ sở dữ liệu production, các thao tác phá hủy trên hệ thống tệp.
- **Các hành động có thể hoàn tác (Reversible actions)** (tùy trường hợp cần HITL): chỉnh sửa tệp cục bộ, thay đổi trên môi trường staging, ghi dữ liệu có cơ chế hoàn tác rõ ràng.
- **Đọc và kiểm tra (Reads and inspections)** (không bao giờ cần HITL): đọc một tệp, liệt kê tài nguyên, gọi một API chỉ đọc (read-only).

### Xác minh sau hành động (Post-action verification)

"Lệnh cam kết đã chạy" không đồng nghĩa với "tác dụng phụ đã diễn ra thành công." Sự phân mảnh mạng (network partition) và điều kiện tranh đua (race conditions) có thể tạo ra một quy trình lầm tưởng rằng mình đã thành công trong khi hệ thống backend thực tế chưa hề lưu dữ liệu. Bước xác minh (verify) tiến hành đọc lại tài nguyên đích sau khi commit để kiểm chứng. Đây chính là mô hình tương tự như các giao dịch cơ sở dữ liệu có mệnh đề `RETURNING` hoặc gọi `GetObject` sau lệnh `PutObject` trong AWS.

### Điều 14 Đạo luật AI của EU (EU AI Act Article 14)

Điều 14 quy định bắt buộc phải có sự giám sát hiệu quả của con người đối với các hệ thống AI có rủi ro cao tại EU. Từ "hiệu quả" (effective) không mang tính trang trí. Ngôn ngữ pháp lý loại trừ dứt khoát các mô hình phê duyệt đóng dấu cao su. Cơ chế propose-then-commit kết hợp challenge-and-response chính là cấu trúc vượt qua được sự xem xét kỹ lưỡng của Điều 14 trong tài liệu tuân thủ của Bộ công cụ Quản trị Agent của Microsoft.

```figure
mx-propose-then-commit
```

## Thực hành (Use It)

Tệp `code/main.py` hiện thực hóa một máy trạng thái propose-then-commit bằng Python thư viện chuẩn. Kho lưu trữ bền bỉ là một tệp JSON. Khóa lũy đẳng là mã băm của tuple `(thread_id, action_signature)`. Trình điều khiển mô phỏng ba trường hợp: một luồng phê duyệt chuẩn tắc, một lần thử lại sau sự cố mạng tạm thời (bắt buộc không được thực thi trùng lặp lần hai), và sự so sánh giữa giao diện mặc định đóng dấu cao su với luồng hỏi-đáp thử thách.

## Đóng gói (Ship It)

Tệp `outputs/skill-hitl-design.md` đánh giá một quy trình HITL dự kiến theo tiêu chuẩn propose-then-commit và gắn cờ cảnh báo các thiếu sót về siêu dữ liệu, tính lũy đẳng, bước xác minh hoặc các tầng hỏi-đáp thử thách.

## Bài tập (Exercises)

1. Chạy `code/main.py`. Xác nhận rằng việc gửi lại một đề xuất đã được duyệt sẽ tái sử dụng bản ghi bền bỉ và không thực thi lại lần thứ hai. Sau đó, sửa đổi khóa lũy đẳng để có thêm dấu thời gian (timestamp) và quan sát hiện tượng thực thi trùng lặp khi thử lại.

2. Mở rộng bản ghi đề xuất bằng trường `rollback`. Mô phỏng một lượt thực thi có bước xác minh (verify) thất bại. Thể hiện quy trình tự động kích hoạt thao tác hoàn tác (rollback).

3. Đọc tài liệu về `RequestInfoEvent` trong Microsoft Agent Framework. Xác định một trường siêu dữ liệu mà API đó hỗ trợ nhưng cỗ máy mô phỏng đơn giản này còn thiếu. Bổ sung trường đó và giải thích nó bảo vệ hệ thống trước rủi ro gì.

4. Thiết kế một bảng kiểm hỏi-đáp thử thách cho một hành động cụ thể (ví dụ: "đăng bài lên tài khoản Twitter/X công khai của tổ chức"). Ba câu hỏi nào bắt buộc người đánh giá phải trả lời? Tại sao lại chọn đúng ba câu hỏi đó?

5. Chọn một trường hợp mà một thông báo nhắc "Duyệt?" đồng bộ đơn giản là đủ đáp ứng (không cần kho lưu trữ bền bỉ). Giải thích lý do và nêu rõ phân lớp rủi ro mà bạn chấp nhận đánh đổi.

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Propose-then-commit | "Phê duyệt hai pha" | Đề xuất lưu bền bỉ + cam kết chủ động + bước xác minh |
| Idempotency key | "Token an toàn thử lại" | Mã định danh duy nhất cho mỗi đề xuất; lần thực thi thứ hai không sinh tác dụng phụ |
| Data lineage | "Nguồn gốc dữ liệu" | Nội dung nguồn cụ thể dẫn tới đề xuất hành động |
| Blast radius | "Bán kính thiệt hại" | Phạm vi ảnh hưởng tối đa nếu hành động gặp sự cố |
| Rubber-stamp | "Phê duyệt hình thức" | Nhấn "Duyệt" vội vã mà không qua thẩm định thực chất |
| Challenge-and-response | "Bảng kiểm ép buộc" | Người đánh giá bắt buộc phải khẳng định các câu hỏi cụ thể |
| RequestInfoEvent | "Nguyên hàm MS Agent Framework" | Yêu cầu HITL bền vững kèm siêu dữ liệu có cấu trúc |
| `interrupt()` / `waitForApproval()` | "Nguyên hàm khung làm việc" | Các giao diện tương đương của LangGraph / Cloudflare cho cùng một cấu trúc |

## Đọc thêm (Further Reading)

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, phê duyệt bền bỉ.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` và Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL như một biện pháp giảm thiểu rủi ro cho agent chân trời dài hạn.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — đường cơ sở pháp lý cho các hệ thống rủi ro cao.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — góc nhìn hiến định xoay quanh cơ chế giám sát.
