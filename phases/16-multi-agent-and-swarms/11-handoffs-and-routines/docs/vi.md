# Bàn giao và Quy trình Thường nhật — Điều phối Phi trạng thái (Handoffs and Routines — Stateless Orchestration)

> Framework Swarm của OpenAI (tháng 10 năm 2024) đã cô đọng cơ chế điều phối đa tác tử thành hai thành phần nguyên thủy: **quy trình thường nhật (routines)** (các chỉ dẫn + công cụ dưới dạng system prompt) và **bàn giao (handoffs)** (một công cụ trả về một Agent khác). Không cần máy trạng thái (state machine), không cần ngôn ngữ đặc thù miền (DSL) phân nhánh phức tạp — LLM tự định tuyến bằng cách kích hoạt công cụ bàn giao phù hợp. OpenAI Agents SDK (tháng 3 năm 2025) là phiên bản kế nhiệm sẵn sàng cho môi trường sản xuất. Bản thân Swarm vẫn là tài liệu tham chiếu mang tính khái niệm trong sáng nhất — toàn bộ mã nguồn của nó chỉ gói gọn trong vài trăm dòng mã. Mẫu hình này lan tỏa mạnh mẽ vì giao diện lập trình của nó gói gọn ở mức "tác tử = prompt + công cụ; bàn giao = hàm trả về tác tử". Điểm hạn chế: phi trạng thái (stateless), do đó việc duy trì bộ nhớ hoàn toàn phụ thuộc vào phía gọi hệ thống.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Vấn đề

Hầu hết các framework đa tác tử hiện nay đều buộc bạn phải học một DSL riêng biệt: các nút (nodes) và cạnh (edges) trong LangGraph, các nhóm (crews) và tác vụ (tasks) trong CrewAI, hay GroupChat và bộ quản lý trong AutoGen. Những DSL này mang lại các lớp trừu tượng thực sự hữu ích, nhưng chúng vô tình làm cho kiến trúc hệ thống trở nên cồng kềnh, nặng nề hơn mức cần thiết.

Swarm đi theo hướng ngược lại: tận dụng tối đa khả năng gọi công cụ (tool-calling) vốn có của mô hình ngôn ngữ lớn. Các bước bàn giao thực chất chỉ là các lượt gọi công cụ. Thực thể điều phối chính là tác tử hiện đang nắm giữ cuộc trò chuyện. Máy trạng thái được định hình ngầm định ngay bên trong system prompt của từng tác tử.

## Khái niệm

### Hai thành phần nguyên thủy

**Quy trình thường nhật (Routine).** Một system prompt xác định vai trò của tác tử và danh sách các công cụ khả dụng. Hãy hình dung nó như một tập hợp chỉ dẫn có phạm vi giới hạn: "bạn là tác tử phân loại; nếu người dùng hỏi về hoàn tiền, hãy bàn giao cho tác tử hoàn tiền".

**Bàn giao (Handoff).** Một công cụ mà tác tử có thể gọi, trả về một đối tượng `Agent` mới. Môi trường thực thi Swarm phát hiện giá trị trả về dạng `Agent` và lập tức chuyển đổi tác tử đang hoạt động (active agent) cho lượt tiếp theo.

Đó chính là toàn bộ cấu trúc trừu tượng.

```python
def transfer_to_refunds():
    return refund_agent  # Swarm nhận diện trả về Agent → chuyển đổi tác tử đang hoạt động

triage_agent = Agent(
    name="triage",
    instructions="Định tuyến người dùng đến đúng chuyên viên.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

System prompt của tác tử phân loại sẽ hướng dẫn mô hình chọn đúng công cụ bàn giao dựa trên thông điệp của người dùng. Khả năng gọi công cụ của LLM sẽ đảm nhận toàn bộ việc định tuyến.

### Tại sao mẫu hình này lan tỏa mạnh mẽ?

- **Giao diện lập trình cực kỳ tinh gọn.** Chỉ cần học hai khái niệm cốt lõi.
- **Tận dụng những gì mô hình đã làm rất tốt.** Cơ chế gọi công cụ (tool calling) đã đạt độ tin cậy cấp doanh nghiệp trên hầu hết các nhà cung cấp mô hình.
- **Không gánh nặng về máy trạng thái.** Bạn không cần phải mô tả cấu trúc đồ thị; chính prompt của tác tử sẽ chỉ định nó cần bàn giao cho ai.

### Sự đánh đổi của kiến trúc phi trạng thái

Swarm được thiết kế hoàn toàn phi trạng thái giữa các lần chạy. Framework lưu giữ lịch sử thông điệp trong suốt một phiên chạy, nhưng không lưu trữ bền vững bất kỳ thứ gì sau khi kết thúc. Bộ nhớ, tính liên tục, và các tác vụ chạy dài hạn — tất cả đều là trách nhiệm của phía ứng dụng gọi (caller).

Trong môi trường sản xuất (OpenAI Agents SDK, tháng 3 năm 2025), đây là một trong những điểm nâng cấp lớn nhất: SDK bổ sung khả năng quản lý phiên tích hợp (session management), các chốt kiểm duyệt (guardrails), và truy vết dấu vết (tracing), trong khi vẫn bảo tồn thành phần nguyên thủy handoff.

### Khi nào Swarm và Handoffs phù hợp?

- **Mô hình phân loại và điều phối (Triage patterns).** Tác tử tiền tuyến tiếp nhận và định tuyến người dùng đến đúng chuyên viên.
- **Bàn giao dựa trên kỹ năng (Skill-based handoffs).** "Nếu tác vụ cần viết mã, hãy gọi tác tử lập trình; nếu cần nghiên cứu dữ liệu, hãy gọi tác tử nghiên cứu".
- **Các phiên hội thoại ngắn, có ranh giới rõ ràng.** Hỗ trợ khách hàng, giải đáp thắc mắc chuyển thành vé hỗ trợ (FAQ-to-ticket), các quy trình làm việc đơn giản.

### Khi nào Swarm gặp khó khăn?

- **Các phiên làm việc dài hạn với bộ nhớ dùng chung.** Việc bàn giao sẽ tái lập ngữ cảnh cuộc trò chuyện thành prompt của tác tử mới kèm lịch sử. Không có trạng thái bền vững xuyên suốt các tác tử nếu phía gọi không chủ động quản lý bộ nhớ.
- **Thực thi song song (Parallel execution).** Cơ chế bàn giao diễn ra tuần tự từng bước một — tác tử đang hoạt động được hoán đổi. Việc chạy song song đòi hỏi bên gọi phải tự điều phối nhiều phiên Swarm chạy đồng thời.
- **Kiểm toán và tái hiện (Audit and replay).** Các phiên chạy phi trạng thái rất khó tái hiện chính xác 100%; quyết định gọi công cụ bàn giao của LLM không mang tính tất định hoàn toàn.

### OpenAI Agents SDK (Tháng 3 năm 2025)

Phiên bản kế nhiệm sẵn sàng cho môi trường sản xuất bổ sung thêm:

- **Trạng thái phiên (Session state).** Luồng dữ liệu bền vững xuyên suốt các lần chạy.
- **Chốt kiểm duyệt (Guardrails).** Các điểm móc (hooks) kiểm thực đầu vào và đầu ra.
- **Khả năng truy vết (Tracing).** Mọi lượt gọi công cụ và bàn giao đều được ghi nhật ký giám sát.
- **Bộ lọc bàn giao (Handoff filters).** Kiểm soát chặt chẽ những phần ngữ cảnh nào được phép truyền đi khi chuyển giao.

Cơ chế bàn giao nguyên thủy vẫn giữ nguyên; các công cụ hỗ trợ công thái học trong môi trường sản xuất được xây dựng bao bọc xung quanh nó.

### So sánh Swarm và GroupChat

Cả hai đều sử dụng LLM để định tuyến, nhưng khác nhau căn bản ở câu hỏi **ai là người đưa ra quyết định tiếp theo**:

- **GroupChat:** một bộ chọn (hàm logic hoặc LLM) đứng từ bên ngoài để chọn người nói tiếp theo.
- **Swarm:** tác tử hiện tại tự chọn tác tử kế nhiệm bằng cách kích hoạt một công cụ bàn giao.

Swarm mang tư tưởng "tác tử tự quyết định bước đi tiếp theo"; GroupChat mang tư tưởng "bộ quản lý quyết định ai lên tiếng tiếp theo". Quyết định của Swarm nằm trong lượt gọi công cụ của tác tử đang hoạt động; quyết định của GroupChat nằm trong `GroupChatManager`.

```figure
sw-handoff-routing
```

## Xây dựng thực tế

`code/main.py` triển khai Swarm từ đầu: một dataclass `Agent`, một cơ chế bàn giao (công cụ trả về đối tượng `Agent`), và một vòng lặp thực thi nhận diện sự thay đổi của tác tử.

Bản minh họa: một tác tử phân loại (triage) định tuyến yêu cầu đến các chuyên viên hoàn tiền (refund), kinh doanh (sales), hoặc hỗ trợ kỹ thuật (support). Mỗi chuyên viên sở hữu các công cụ riêng biệt. Vòng lặp thực thi sẽ in ra chi tiết từng bước bàn giao.

Chạy chương trình:

```bash
python3 code/main.py
```

## Ứng dụng thực tế

`outputs/skill-handoff-designer.md` thiết kế cấu trúc mạng lưới bàn giao cho một tác vụ cụ thể: các tác tử nào cần tồn tại, những công cụ bàn giao nào chúng có quyền gọi, và phần ngữ cảnh nào sẽ được chuyển tiếp qua ranh giới bàn giao.

## Đưa vào sản xuất

Danh sách kiểm tra khi triển khai:

- **Ghi nhật ký bàn giao (Handoff logging).** Mỗi lần bàn giao đều phải ghi lại một sự kiện truy vết chứa thông tin: tác tử gửi (from-agent), tác tử nhận (to-agent) và ảnh chụp nhanh ngữ cảnh (context snapshot).
- **Quy tắc chuyển giao ngữ cảnh (Context transfer rules).** Quyết định rõ những gì được truyền đi: toàn bộ lịch sử (chi phí token cao), N tin nhắn gần nhất, hay một bản tóm tắt súc tích.
- **Chốt kiểm duyệt tại điểm bàn giao.** Việc bàn giao cho một chuyên viên có quyền thực thi công cụ nguy hiểm hơn phải được xác thực nghiêm ngặt — nhằm ngăn chặn tấn công tiêm nhiễm prompt (prompt injection) ép hệ thống bàn giao ngoài ý muốn.
- **Phát hiện vòng lặp (Loop detection).** Hai tác tử bàn giao qua lại lẫn nhau là dạng lỗi phổ biến; phát hiện bằng cách kiểm tra vòng lặp trên K bước gần nhất.
- **Tác tử dự phòng (Fallback agent).** Nếu mục tiêu bàn giao không tồn tại, hãy quay về một tác tử mặc định an toàn.

## Bài tập

1. Chạy `code/main.py`, chuyển tuyến đến tác tử hoàn tiền. Xác nhận rằng ở lượt thứ hai, tác tử đang hoạt động chính là chuyên viên hoàn tiền.
2. Thêm quy tắc phát hiện vòng lặp: nếu hai tác tử giống nhau bàn giao qua lại 3 lần liên tiếp, hãy cưỡng chế thoát khỏi vòng lặp và thiết kế cơ chế xử lý dự phòng.
3. Đọc tài liệu của OpenAI Agents SDK về các bộ lọc bàn giao (handoff filters). Triển khai một phiên bản "tóm tắt khi bàn giao": tác tử hiện tại nén ngữ cảnh thành các gạch đầu dòng tóm tắt trước khi tác tử mới tiếp quản.
4. So sánh cơ chế bàn giao của Swarm với bộ chọn của `GroupChatManager`. Mẫu hình nào dễ bị tổn thương hơn trước tấn công tiêm nhiễm prompt, và tại sao?
5. Đọc cẩm nang thực hành Swarm Cookbook của OpenAI. Xác định một quyết định thiết kế cụ thể mà Swarm đưa ra đã được OpenAI Agents SDK giữ nguyên hoặc thay đổi.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật |
|---|---|---|
| Quy trình thường nhật (Routine) | "Prompt của tác tử" | System prompt + danh sách công cụ. Xác định vai trò và các bàn giao khả dụng. |
| Bàn giao (Handoff) | "Chuyển tiếp cho tác tử khác" | Công cụ mà tác tử đang hoạt động có thể gọi, trả về đối tượng Agent mới. Môi trường thực thi sẽ hoán đổi tác tử. |
| Phi trạng thái (Stateless) | "Không có bộ nhớ giữa các lần chạy" | Swarm không lưu trữ dữ liệu bền vững; bộ nhớ hoàn toàn do phía gọi quản lý. |
| Tác tử đang hoạt động (Active agent) | "Ai đang nói chuyện" | Tác tử hiện đang nắm giữ lượt hội thoại. Bàn giao sẽ thay đổi thực thể này. |
| Chuyển giao ngữ cảnh (Context transfer) | "Dữ liệu gì được mang theo khi chuyển tiếp" | Chính sách xác định lịch sử mà tác tử mới nhận được: toàn bộ, N thông điệp gần nhất, hay bản tóm tắt. |
| Vòng lặp bàn giao (Handoff loop) | "Các tác tử chuyền bóng qua lại" | Lỗi khi hai tác tử liên tục bàn giao quyền xử lý cho nhau mà không dừng. |
| OpenAI Agents SDK | "Swarm phiên bản doanh nghiệp" | Phiên bản kế nhiệm tháng 3/2025; bổ sung phiên, chốt kiểm duyệt, truy vết lên trên nền tảng handoff. |
| Bộ lọc bàn giao (Handoff filter) | "Cổng kiểm soát chuyển giao" | Tính năng SDK cho phép kiểm tra và biến đổi ngữ cảnh tại ranh giới bàn giao. |

## Tài liệu đọc thêm

- [OpenAI Cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — bài viết trình bày chuẩn mực khái niệm
- [Kho mã nguồn OpenAI Swarm](https://github.com/openai/swarm) — bản triển khai gốc, lưu giữ dưới dạng tài liệu tham chiếu khái niệm
- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — nền tảng sản xuất kế nhiệm hỗ trợ phiên làm việc và truy vết
- [Ghi chú về cơ chế bàn giao trong Claude Code](https://docs.anthropic.com/en/docs/claude-code) — cách các tác tử phụ của Claude Code áp dụng mô hình tương tự thông qua `Task`
