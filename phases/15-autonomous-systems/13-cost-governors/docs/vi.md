# Ngân sách Hành động, Giới hạn Lặp và Bộ điều tiết Chi phí (Action Budgets, Iteration Caps, and Cost Governors)

> Chi phí LLM hàng tháng của một agent thương mại điện tử quy mô vừa đã tăng vọt từ $1.200 lên $4.800 sau khi đội ngũ phát triển kích hoạt kỹ năng "order-tracking" (theo dõi đơn hàng). Đó không phải là lỗi định giá (pricing bug). Đó là do agent đã rơi vào một vòng lặp mới và liên tục tiêu tiền bên trong vòng lặp đó. Bộ công cụ Quản trị Agent của Microsoft (Microsoft's Agent Governance Toolkit, 02/04/2026) đã chuẩn hóa cơ chế phòng thủ trước phân lớp sự cố này: thiết lập `max_tokens` trên mỗi lượt yêu cầu, hạn mức token và ngân sách tài chính trên mỗi tác vụ, giới hạn trần theo ngày/tháng, giới hạn số lần lặp (iteration caps), định tuyến mô hình theo phân tầng (tiered model routing), lưu bộ nhớ đệm lời nhắc (prompt caching), quản lý cửa sổ ngữ cảnh (context windowing), các điểm kiểm soát có con người can thiệp (human-in-the-loop - HITL checkpoints) đối với các hành động tốn kém, và cơ chế ngắt khẩn cấp (kill switches) khi vi phạm ngân sách. Bộ SDK Claude Code Agent của Anthropic cũng cung cấp các nguyên mẫu (primitives) tương tự dưới các tên gọi khác nhau. Giới hạn vận tốc tài chính (financial velocity limits) — ví dụ: ngắt quyền truy cập nếu chi tiêu vượt quá $50 trong vòng 10 phút — giúp phát hiện và ngăn chặn các vòng lặp nhanh hơn nhiều so với các giới hạn trần hàng tháng.

**Loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, trình mô phỏng bộ điều tiết chi phí nhiều tầng)
**Điều kiện tiên quyết:** Phase 15 · 10 (Chế độ cấp quyền), Phase 15 · 12 (Thực thi bền bỉ)
**Thời lượng:** ~60 phút

## Vấn đề đặt ra (The Problem)

Autonomous agents (các agent tự chủ) tiêu tốn tiền thật trong từng lượt thực thi. Đầu ra kém chất lượng của một chatbot chỉ đơn thuần là một câu trả lời tồi; nhưng vòng lặp sai sót của một agent lại là một hóa đơn viện dẫn chi phí khổng lồ. Thuật ngữ được ngành công nghiệp ghi nhận cho dạng lỗi hỏng hóc này là "Denial of Wallet" (từ chối ví / cạn kiệt tài chính) — agent liên tục suy luận, liên tục gọi công cụ (tool-calling), liên tục phát sinh cước phí, và không có gì ngăn chặn được vì hệ thống vốn không được thiết kế cơ chế dừng.

Biện pháp khắc phục không phải là một con số đơn lẻ. Đó là một ngăn xếp các giới hạn (stack of limits) ở các thang thời gian và cấp độ phân giải khác nhau: trên từng yêu cầu (per-request), trên từng tác vụ (per-task), theo giờ (per-hour), theo ngày (per-day), theo tháng (per-month). Một ngăn xếp được thiết kế bài bản sẽ chặn đứng một vòng lặp chạy trốn (runaway loop) trong vài phút, phát hiện điểm rò rỉ âm thầm (slow leak) trong vài giờ, và nhận diện một bản phát hành lỗi trong vòng một ngày. Ngăn xếp này cũng chính là thứ duy trì được ngân sách tổng thể khi agent hoạt động tự chủ với chân trời dài hạn (long-horizon).

Đây là một bài học kỹ thuật: toán học đằng sau rất đơn giản, nhưng kỷ luật kiến trúc mới là nơi các đội ngũ thường thất bại. Danh sách các giới hạn dưới đây đều được định danh cụ thể trong Bộ công cụ Quản trị Agent của Microsoft (Microsoft Agent Governance Toolkit) hoặc tài liệu hướng dẫn SDK Claude Code Agent của Anthropic.

## Khái niệm cốt lõi (The Concept)

### Ngăn xếp điều tiết chi phí (The cost-governor stack)

1. **`max_tokens` trên mỗi yêu cầu.** Đơn giản nhưng trọng yếu. Ngăn chặn bất kỳ một lệnh gọi đơn lẻ nào tạo ra một kết quả hoàn tất (completion) không giới hạn.
2. **Ngân sách token trên mỗi tác vụ (Per-task token budget).** Xuyên suốt toàn bộ quá trình chạy, không được vượt quá N tokens. Dừng cứng (hard stop) ngay tại hạn mức trần.
3. **Ngân sách tiền mặt trên mỗi tác vụ (Per-task dollar budget).** Tương tự như ngân sách token nhưng tính theo đơn vị tiền tệ. Tham số `max_budget_usd` trong Claude Code.
4. **Giới hạn số lần gọi trên từng công cụ (Per-tool call cap).** Không được vượt quá N lượt gọi `WebFetch`, N lượt gọi `shell_exec`, v.v.
5. **Giới hạn số vòng lặp (`max_turns` / iteration cap).** Tổng số lần lặp trong vòng lặp agent; ngăn chặn các vòng lặp suy luận vô tận.
6. **Giới hạn trần theo phút / giờ / ngày / tháng.** Các cửa sổ trượt (rolling windows). Bắt giữ các rò rỉ chi phí ở các thang thời gian khác nhau.
7. **Giới hạn vận tốc tài chính (Financial velocity limit).** Ví dụ: "nếu mức chi tiêu vượt quá $50 trong 10 phút, lập tức cắt quyền truy cập." Ngăn chặn việc 'đốt' ngân sách do vòng lặp trước khi các giới hạn tháng kịp kích hoạt.
8. **Định tuyến mô hình theo phân tầng (Tiered model routing).** Mặc định sử dụng một mô hình nhỏ hơn, chi phí thấp; chỉ nâng cấp (escalate) lên mô hình lớn hơn khi một bộ phân loại (classifier) đánh giá rằng độ phức tạp của tác vụ thực sự đòi hỏi điều đó.
9. **Lưu bộ nhớ đệm lời nhắc (Prompt caching).** Lời nhắc hệ thống (system prompt) và ngữ cảnh ổn định được lưu trữ trong bộ nhớ đệm phía nhà cung cấp dịch vụ (provider cache); chi phí token cho việc gửi lại ngữ cảnh này xấp xỉ bằng không.
10. **Quản lý cửa sổ ngữ cảnh (Context windowing).** Thu gọn (compaction) / tóm tắt (summarization) nhằm giữ ngữ cảnh hoạt động luôn dưới một ngưỡng định trước; trực tiếp giảm chi phí token.
11. **Điểm kiểm soát HITL đối với các hành động tốn kém (HITL checkpoints on expensive actions).** Trước khi thực hiện một hành động được xác định là tốn kém (gọi công cụ kéo dài, tải tệp lớn, nâng cấp mô hình tốn chi phí cao), bắt buộc phải có xác nhận thủ công từ con người.
12. **Cơ chế ngắt khẩn cấp khi vi phạm ngân sách (Kill switch on budget breach).** Phiên làm việc lập tức bị hủy bỏ (abort) khi bất kỳ mức trần nào bị kích hoạt. Vi phạm được ghi nhật ký; đòi hỏi một quy trình riêng biệt để tái kích hoạt.

### Tại sao cần một ngăn xếp thay vì một giới hạn đơn lẻ

Một giới hạn duy nhất theo tháng chỉ bắt được agent chạy trốn sau khi chiếc ví đã hoàn toàn cạn kiệt. Một giới hạn đơn lẻ trên mỗi yêu cầu không thể phát hiện bất kỳ bất thường nào ở cấp độ toàn phiên. Các dạng thức hỏng hóc khác nhau đòi hỏi các thang đo thời gian tương ứng khác nhau:

- **Vòng lặp chạy trốn (Runaway loop)** (agent kẹt trong chu kỳ thử lại 5 giây một lần): được chặn đứng bởi giới hạn vận tốc tài chính (velocity limit).
- **Rò rỉ âm thầm (Slow leak)** (agent thực hiện lượng công việc gấp ~2 lần mức kỳ vọng cho mỗi tác vụ): được chặn lại bởi giới hạn theo ngày.
- **Bản phát hành lỗi (Bad release)** (phiên bản mới tiêu tốn lượng token gấp 5 lần): bị chặn lại bởi giới hạn theo tuần / tháng.
- **Đợt tăng đột biến hợp lệ (Legitimate surge)** (nhu cầu thực tế từ người dùng, không phải do lỗi): được kiểm soát bởi giới hạn giờ / ngày đi kèm nhật ký rõ ràng.

### Giao diện bề mặt ngân sách trong môi trường thực thi (A harness budget surface)

Claude Code Agent SDK công khai các giao diện sau:

- `max_turns` — giới hạn số lần lặp của vòng lặp agent.
- `max_budget_usd` — giới hạn trần tài chính; phiên làm việc sẽ hủy bỏ nếu vượt ngưỡng.
- `allowed_tools` / `disallowed_tools` — danh sách trắng (allowlist) và danh sách đen (denylist) cho các công cụ.
- Các điểm móc (hook points) trước khi sử dụng công cụ để phục vụ công tác hạch toán chi phí tùy biến.

Kết hợp điều này với thang phân cấp chế độ cấp quyền (permission-mode ladder ở Bài 10). Một phiên chạy ở chế độ `autoMode` mà không có `max_budget_usd` chính là sự tự chủ không được quản trị (ungoverned autonomy). Anthropic đặc biệt nhấn mạnh rằng Auto Mode bắt buộc phải đi kèm với các cơ chế kiểm soát ngân sách; bộ phân loại là cơ chế trực giao với chi phí.

### Đạo luật AI của EU (EU AI Act) và OWASP Agentic Top 10

Bộ công cụ Quản trị Agent của Microsoft giải quyết trực tiếp các yêu cầu của OWASP Agentic Top 10 và Điều 14 Đạo luật AI của EU (EU AI Act Article 14 - giám sát của con người). Khi đưa vào môi trường sản xuất (production) tại EU, việc ghi nhật ký và thực thi giới hạn trần chi phí không còn là tùy chọn mà là yêu cầu bắt buộc.

### Trường hợp thực tế được quan sát: $1.200 → $4.800

Một trường hợp có thật được ghi nhận trong tài liệu của Microsoft: chi phí hàng tháng của một agent thương mại điện tử đã tăng gấp 4 lần sau khi bổ sung một công cụ mới. Công cụ này cho phép agent truy vấn thăm dò (poll) trạng thái đơn hàng trong mọi phiên làm việc. Không có cơ chế phát hiện vòng lặp. Không có giới hạn trên từng công cụ. Không có cảnh báo về mức tăng trưởng chi phí theo tuần. Giải pháp khắc phục là áp dụng giới hạn trên từng công cụ kết hợp với cảnh báo tăng trưởng chi phí hàng ngày. Đây là bài học khuôn mẫu: mọi bề mặt công cụ mới đều là một nguy cơ tiềm ẩn vòng lặp mới; mỗi công cụ mới đều cần có giới hạn trần và cảnh báo riêng.

```figure
cost-governor-stack
```

## Thực hành (Use It)

Tệp `code/main.py` mô phỏng một phiên chạy của agent khi có và không có ngăn xếp điều tiết chi phí phân tầng. Agent mô phỏng sẽ dần trôi vào một vòng lặp thăm dò (polling loop) sau một số lượt; ngăn xếp phân tầng sẽ phát hiện và chặn đứng nó trong cửa sổ vận tốc (velocity window), trong khi một giới hạn trần hàng tháng đơn lẻ sẽ không thể kích hoạt cho đến nhiều ngày sau đó.

## Đóng gói (Ship It)

Tệp `outputs/skill-agent-budget-audit.md` kiểm toán ngăn xếp điều tiết chi phí của một phương án triển khai agent dự kiến và gắn cờ cảnh báo các tầng bảo vệ còn thiếu sót.

## Bài tập (Exercises)

1. Chạy `code/main.py`. Xác nhận rằng giới hạn vận tốc kích hoạt trước giới hạn số vòng lặp trên quỹ đạo vòng lặp thăm dò. Sau đó, vô hiệu hóa giới hạn vận tốc và đo lường lượng chi phí mà agent "tiêu tốn" trước khi giới hạn số vòng lặp kịp can thiệp.

2. Thiết kế một bộ giới hạn cho từng công cụ đối với một agent trình duyệt (browser agent, Bài 11). Công cụ nào cần mức giới hạn chặt chẽ nhất? Công cụ nào có thể chạy không giới hạn mà không gây rủi ro?

3. Đọc tài liệu Bộ công cụ Quản trị Agent của Microsoft (Microsoft Agent Governance Toolkit). Liệt kê từng loại giới hạn trần mà bộ công cụ này nhắc đến. Ánh xạ từng loại với một trong các dạng lỗi hỏng hóc (vòng lặp chạy trốn, rò rỉ âm thầm, bản phát hành lỗi, lưu lượng tăng vọt).

4. Định giá chi phí cho một phiên chạy tự động qua đêm không có người giám sát cho một tác vụ thực tế (ví dụ: "phân loại và xử lý 50 issues trong một kho lưu trữ mã nguồn"). Đặt `max_budget_usd` ở mức gấp 2 lần ước tính điểm (point estimate) của bạn. Hãy biện minh cho hệ số 2x này.

5. Tham số `max_budget_usd` của Claude Code được kích hoạt dựa trên chi phí tổng hợp của phiên làm việc. Hãy thiết kế một giới hạn vận tốc bổ trợ mà bạn sẽ thực thi từ bên ngoài. Điều kiện nào sẽ kích hoạt lệnh ngắt và quy trình tái kích hoạt sẽ diễn ra như thế nào?

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Denial of Wallet | "Hóa đơn chạy trốn" | Vòng lặp agent phát sinh chi phí mà không có hạn mức trần để ngăn chặn |
| max_tokens | "Giới hạn mỗi yêu cầu" | Trần giới hạn kích thước của một completion đơn lẻ |
| max_turns | "Giới hạn số vòng lặp" | Trần giới hạn số lần lặp của vòng lặp agent trong một phiên |
| max_budget_usd | "Ngắt khẩn cấp tài chính" | Trần chi phí phiên; hủy bỏ phiên khi vượt ngưỡng |
| Velocity limit | "Giới hạn tốc độ chi tiêu" | Giới hạn chi tiêu trong một cửa sổ ngắn (ví dụ: $50 / 10 phút) |
| Tiered routing | "Ưu tiên mô hình nhỏ" | Mặc định dùng mô hình rẻ; chỉ nâng cấp khi bộ phân loại thấy cần thiết |
| Prompt caching | "Bộ nhớ đệm system prompt" | Bộ nhớ đệm phía nhà cung cấp giảm chi phí token gửi lại về gần bằng không |
| HITL checkpoint | "Cổng phê duyệt của con người" | Bắt buộc con người xác nhận trước khi thực hiện hành động tốn kém |

## Đọc thêm (Further Reading)

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, danh sách trắng công cụ.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các điểm kiểm soát điều tiết chi phí.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — kiểm soát chi phí phía nhà cung cấp dịch vụ.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — cơ chế hoạt động của bộ nhớ đệm lời nhắc.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — hồ sơ chi phí cho các agent hoạt động trong chân trời dài hạn.
