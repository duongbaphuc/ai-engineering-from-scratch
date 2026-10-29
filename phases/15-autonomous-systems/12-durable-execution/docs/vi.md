# Tác tử chạy nền thời gian dài: Thực thi bền bỉ

> Các tác tử chân trời dài hạn trong môi trường sản xuất không chạy trong một vòng lặp `while True`. Mọi lệnh gọi LLM đều trở thành một hoạt động (activity) đi kèm điểm kiểm tra (checkpoint), cơ chế thử lại (retry), và phát lại (replay). Bản tích hợp giữa OpenAI Agents SDK và Temporal đã phát hành chính thức (GA) vào tháng 3 năm 2026. Claude Code Routines (Anthropic) chạy các lần triệu gọi Claude Code theo lịch trình mà không cần duy trì một tiến trình cục bộ chạy liên tục. Các phiên làm việc có thể tạm dừng khi chờ đầu vào từ con người, sống sót qua các đợt triển khai mã (deploys), và phục hồi tiếp tục từ điểm kiểm tra mới nhất được định danh theo `thread_id`. Ẩn sau sự tiện lợi công thái học mới này là một mẫu hình cổ điển — điều phối quy trình làm việc (workflow orchestration) — với một thành phần mới: các lệnh gọi LLM đóng vai trò là những hoạt động bất định (non-deterministic activities) bắt buộc phải được phát lại một cách tất định (deterministically replayed) khi phục hồi sau sự cố.

**Loại bài học:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python (stdlib, máy trạng thái thực thi bền bỉ tối giản)
**Điều kiện tiên quyết:** Phase 15 · 10 (Các chế độ cấp quyền), Phase 15 · 01 (Tác tử chân trời dài hạn)
**Thời gian:** ~60 phút

## Vấn đề thực tế

Hãy xem xét một tác tử chạy trong bốn giờ liên tục. Nó gọi ba công cụ, nhắc hỏi người dùng hai lần, và thực hiện bốn mươi lệnh gọi LLM. Khi mới đi được nửa chặng đường, máy chủ lưu trữ khởi động lại đột ngột. Điều gì sẽ xảy ra?

- Với một vòng lặp `while True` ngây thơ: toàn bộ tiến trình bị xóa sạch. Chu trình chạy phải khởi động lại từ đầu. Ba lệnh gọi công cụ (với các tác dụng phụ thực tế lên hệ thống) lại bị thực thi một lần nữa. Người dùng lại bị làm phiền để hỏi về những việc họ đã phê duyệt từ trước. Bốn mươi lệnh gọi LLM bị tính tiền lại từ đầu.
- Với thực thi bền bỉ (durable execution): chu trình chạy phục hồi chính xác từ điểm kiểm tra gần nhất. Các hoạt động đã hoàn thành sẽ không bị chạy lại; kết quả của chúng được phát lại trực tiếp từ nhật ký bền vững. Người dùng không phải phê duyệt lại những gì đã xong. Các lệnh gọi LLM đã thực hiện không bị tính phí thêm một lần nào.

Đây chính là mẫu hình mà các công cụ điều phối quy trình (workflow engines) đã vận hành suốt một thập kỷ qua (Temporal, Cadence, Cherami của Uber). Điểm mới duy nhất là: các lệnh gọi LLM giờ đây trở thành một dạng hoạt động — mang tính bất định, tốn kém, và gây ra tác dụng phụ — hoàn toàn khớp với mẫu hình này.

Chủ đề xuyên suốt bài học: độ tin cậy trong các tác vụ chân trời dài hạn suy giảm rất nhanh (METR ghi nhận hiện tượng "suy giảm sau 35 phút" — tỷ lệ thành công sụt giảm gần như theo hàm bậc hai theo khoảng chân trời thời gian). Thực thi bền bỉ cho phép các chu trình chạy dài hơn nhiều so với hồ sơ độ tin cậy tự nhiên vốn có, và đây là một cách mới để thất bại an toàn nếu bạn thiết kế đúng, hoặc thất bại thảm khốc nếu bạn thiết kế sai.

## Khái niệm cốt lõi

### Hoạt động, quy trình làm việc và cơ chế phát lại

- **Quy trình làm việc (Workflow)**: Mã điều phối tất định. Xác định trình tự các hoạt động, các nhánh rẽ, và các khoảng chờ. Bắt buộc phải mang tính tất định (deterministic) để có thể phát lại từ nhật ký sự kiện mà không gây ra sự phân kỳ bất ngờ.
- **Hoạt động (Activity)**: Một đơn vị công việc bất định, tiềm ẩn nguy cơ thất bại. Lệnh gọi LLM, gọi công cụ, ghi tệp, yêu cầu HTTP. Mỗi hoạt động đều được ghi nhật ký kèm theo dữ liệu đầu vào và (khi hoàn thành) dữ liệu đầu ra của nó.
- **Nhật ký sự kiện (Event log)**: Kho lưu trữ hỗ trợ bền vững. Mọi thời điểm bắt đầu hoạt động, hoàn thành, thất bại, thử lại, và mọi quyết định của quy trình làm việc đều được ghi lại.
- **Phát lại (Replay)**: Khi phục hồi sau sự cố, mã quy trình làm việc sẽ chạy lại từ đầu; mọi hoạt động đã hoàn tất thành công sẽ trả về kết quả đã lưu trong nhật ký mà không cần thực thi lại mã lệnh thực tế. Chỉ những hoạt động chưa hoàn thành mới thực sự được kích hoạt.

Cơ chế này có cùng hình thái với cách React kết xuất lại dựa trên Virtual DOM, hoặc cách Git tái tạo cây thư mục làm việc từ chuỗi commit. Tính tất định trong bộ điều phối chính là yếu tố giúp sự bền bỉ có chi phí vận hành rất rẻ.

### Tại sao các lệnh gọi LLM khớp hoàn hảo với mẫu hình này

Các lệnh gọi LLM có đặc tính:
- Bất định (non-deterministic: tham số temperature > 0; ngay cả khi temperature = 0 thì đầu ra vẫn trôi dạt giữa các phiên bản mô hình).
- Đắt đỏ (cả về chi phí tài chính lẫn độ trễ thời gian).
- Tiềm ẩn nguy cơ thất bại (giới hạn tốc độ rate limits, lỗi quá thời gian timeouts).
- Tạo ra tác dụng phụ (nếu chúng kích hoạt các công cụ bên ngoài).

Đây chính là hồ sơ hoàn hảo của một "hoạt động" (activity). Bọc mọi lệnh gọi LLM dưới dạng một hoạt động sẽ mang lại cho bạn: cơ chế thử lại kèm thuật toán lùi theo cấp số nhân (exponential backoff), lưu điểm kiểm tra qua các lần khởi động lại máy chủ, và một vết truy vết (trace) có thể phát lại để phục vụ việc gỡ lỗi.

### Điểm kiểm tra được định danh theo `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects, và Claude Code Routines đều hội tụ về cùng một hình thái API: một `thread_id` (hoặc tương đương) dùng để định danh phiên làm việc; mỗi bước chuyển trạng thái được lưu bền bỉ vào một hệ thống phụ trợ (PostgreSQL theo mặc định, SQLite cho phát triển cục bộ, Redis làm bộ đệm cache); quá trình phục hồi sẽ đọc điểm kiểm tra mới nhất.

Việc lựa chọn hệ thống lưu trữ phía sau rất quan trọng:

- **PostgreSQL**: Bền vững, có khả năng truy vấn, tồn tại qua các đợt phát hành mã mới. Là lựa chọn mặc định của LangGraph.
- **SQLite**: Chỉ dùng cho phát triển cục bộ; mất dữ liệu khi chuyển đổi giữa các máy chủ.
- **Redis**: Rất nhanh nhưng là bộ nhớ tạm thời trừ khi được cấu hình cơ chế AOF/snapshot cẩn thận.
- **Cloudflare Durable Objects**: Phân tán hoàn toàn trong suốt; được đóng phạm vi bởi một khóa duy nhất; duy trì trạng thái từ vài giờ đến vài tuần.

### Đầu vào từ con người là một trạng thái hạng nhất (first-class state)

Mô hình đề xuất rồi mới xác nhận (propose-then-commit) trong Bài 15 đòi hỏi một trạng thái bền bỉ "đang chờ con người" (waiting on human). Quy trình làm việc tạm dừng, hàng đợi bên ngoài giữ yêu cầu chờ xử lý, và một sự phê duyệt từ người dùng sẽ tiếp tục tiến trình ngay từ đúng thời điểm đó. Nếu không có cơ chế bền bỉ, đây chỉ là giải pháp chắp vá hên xui; nhưng khi có nó, một phê duyệt gửi đến qua đêm vẫn có thể được quy trình đón nhận và chạy tiếp mượt mà vào sáng hôm sau.

### Hiện tượng suy giảm sau 35 phút (35-minute degradation)

Tổ chức METR đã quan sát thấy rằng mọi lớp tác tử được đo lường đều biểu hiện sự suy giảm độ tin cậy rõ rệt sau ~35 phút hoạt động liên tục. Tăng gấp đôi thời lượng tác vụ sẽ làm tăng tỷ lệ thất bại lên xấp xỉ bốn lần (theo hàm bậc hai). Thực thi bền bỉ không giải quyết được căn nguyên này; nó chỉ cho phép bạn chạy lâu hơn mức mà hồ sơ độ tin cậy thực tế có thể duy trì. Mẫu hình an toàn là kết hợp tính bền bỉ với các điểm kiểm tra bắt buộc phải có sự can thiệp mới từ con người (fresh HITL) khi phục hồi, và đi kèm công tắc ngắt khẩn cấp theo ngân sách (Bài 13) để giới hạn tổng lượng tính toán bất kể thời gian thực tế trôi qua bao lâu.

### Khi nào thực thi bền bỉ là một câu trả lời sai lầm

- Các chu trình chạy ngắn hơn vài phút và không cần sự can thiệp của con người. Chi phí thiết lập và độ trễ > lợi ích mang lại.
- Các tác vụ truy xuất thông tin chỉ-đọc (read-only) thuần túy.
- Các bài toán đòi hỏi tính đúng đắn phải diễn ra liền mạch từ đầu đến cuối bên trong một cửa sổ ngữ cảnh duy nhất (một số tác vụ lập luận logic phức tạp; sinh mã một lần).

```figure
memory-consolidation
```

## Sử dụng trong thực tế

Tệp `code/main.py` triển khai một công cụ thực thi bền bỉ tối giản bằng thư viện chuẩn của Python (stdlib). Hệ thống hỗ trợ:

- Bộ trang trí `@activity` ghi nhật ký đầu vào và đầu ra vào một tệp nhật ký sự kiện định dạng JSON.
- Một hàm quy trình làm việc (workflow) liên kết tuần tự các hoạt động.
- Một hàm `run_or_replay(workflow, event_log)` có khả năng phát lại các hoạt động đã hoàn thành mà không cần thực thi lại mã nguồn.

Kịch bản điều khiển mô phỏng một quy trình ba hoạt động, gặp sự cố sập máy ở giữa chừng, và minh họa sự khác biệt giữa: (a) cơ chế thử lại ngây thơ chạy lại toàn bộ từ đầu so với (b) cơ chế phát lại thông minh chỉ thực thi duy nhất hoạt động còn thiếu.

## Đóng gói sản phẩm

Tệp `outputs/skill-durable-execution-review.md` kiểm duyệt một triển khai tác tử chạy lâu dài được đề xuất để đảm bảo cấu trúc thực thi bền bỉ chuẩn mực: các hoạt động, tính tất định, hạ tầng lưu trữ điểm kiểm tra, trạng thái chờ đầu vào con người, và chính sách can thiệp của con người (HITL) khi phục hồi.

## Bài tập

1. Chạy `code/main.py`. Quan sát sự khác biệt về số lần thực thi hoạt động thực tế giữa cách thử lại ngây thơ và cách phát lại bền bỉ. Thay đổi điểm sập sự cố và kiểm tra xem số lần phát lại thay đổi tương ứng như thế nào.

2. Chuyển đổi công cụ mô phỏng để sử dụng `thread_id` một cách tường minh. Mô phỏng hai phiên chạy đồng thời dùng chung công cụ và xác nhận rằng nhật ký sự kiện của chúng không bị xung đột hay ghi đè lên nhau.

3. Chọn một hoạt động trong công cụ mô phỏng. Đưa vào một yếu tố bất định (ví dụ: lấy dấu thời gian đồng hồ thực bên trong một nhánh quyết định của quy trình làm việc). Chứng minh sự phân kỳ kết quả khi phát lại. Giải thích cách các công cụ trong thực tế giải quyết vấn đề này (đăng ký tác dụng phụ, các API như `Workflow.now()`).

4. Đọc bài viết "Runtime behind production deep agents" của LangChain. Liệt kê mọi trạng thái mà môi trường runtime lưu bền bỉ và nêu rõ từng trạng thái đó giúp khắc phục phương thức thất bại nào.

5. Thiết kế một chính sách lưu điểm kiểm tra cho một tác vụ lập trình tự chủ kéo dài 6 giờ. Bạn sẽ đặt điểm kiểm tra ở đâu? Quá trình phục hồi sau sự cố diễn ra như thế nào? Hành động nào bắt buộc phải có sự xác nhận mới từ con người?

## Thuật ngữ chính

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Workflow | "Kịch bản tác tử" | Mã điều phối tất định; có thể phát lại hoàn hảo từ nhật ký sự kiện |
| Activity | "Một bước thực hiện" | Đơn vị công việc bất định (gọi LLM, gọi công cụ); được ghi nhật ký trước và sau khi chạy |
| Event log | "Kho lưu trữ nền tảng" | Bản ghi bền vững về mọi biến chuyển trạng thái trong hệ thống |
| Replay | "Khôi phục / chạy tiếp" | Chạy lại quy trình; các hoạt động đã xong trả về kết quả trong nhật ký mà không chạy lại mã |
| Checkpoint | "Điểm lưu trạng thái" | Trạng thái bền bỉ được định danh theo `thread_id`; phiên bản mới nhất được dùng khi phục hồi |
| thread_id | "Khóa phiên làm việc" | Mã định danh phân định phạm vi trạng thái bền bỉ của một phiên |
| 35-minute degradation | "Suy giảm độ tin cậy" | Kết quả từ METR: tỷ lệ thành công sụt giảm theo hàm bậc hai theo khoảng chân trời thời gian |
| Non-determinism | "Lệch kết quả khi phát lại" | Đồng hồ thực, tính ngẫu nhiên, đầu ra LLM; bắt buộc phải được đăng ký dưới dạng tác dụng phụ |

## Đọc thêm

- [Anthropic — Claude Code Agent SDK: Vòng lặp tác tử](https://code.claude.com/docs/en/agent-sdk/agent-loop) — ngân sách, số lượt, và ngữ nghĩa phục hồi.
- [Microsoft — Agent Framework: Sự can thiệp của con người và lưu điểm kiểm tra](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cấu trúc RequestInfoEvent.
- [LangChain — Môi trường Runtime đằng sau các tác tử Deep Agents trong sản xuất](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — các yêu cầu thực tế đối với runtime.
- [Tích hợp OpenAI Agents SDK + Temporal (Thông báo từ Trigger.dev)](https://trigger.dev) — cấu trúc hoạt động dành cho các lệnh gọi LLM.
- [Anthropic — Đo lường quyền tự chủ của tác tử trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — dẫn chiếu thực nghiệm về sự suy giảm độ tin cậy sau 35 phút.
