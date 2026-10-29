# Các Mô hình Quy trình Làm việc của Anthropic: Đơn giản Hơn Phức tạp

> Schluntz và Zhang (Anthropic, tháng 12 năm 2024) đã phân biệt rạch ròi giữa quy trình làm việc (workflows - các đường dẫn định sẵn) và tác tử (agents - sử dụng công cụ động). Năm mô hình quy trình làm việc (workflow patterns) có thể bao quát phần lớn các bài toán thực tế. Hãy bắt đầu từ các lệnh gọi API trực tiếp đơn giản. Chỉ bổ sung tác tử khi các bước thực thi không thể dự đoán trước.

**Loại bài học:** Học lý thuyết + Xây dựng (Learn + Build)  
**Ngôn ngữ:** Python (thư viện chuẩn - stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 01 (Vòng lặp Tác tử - Agent Loop)  
**Thời lượng:** ~60 phút

## Mục tiêu Học tập

- Nêu tên năm mô hình quy trình làm việc của Anthropic: chuỗi lời nhắc (prompt chaining), định tuyến (routing), song song hóa (parallelization), điều phối - thực thi (orchestrator-workers), và đánh giá - tối ưu (evaluator-optimizer).
- Giải thích sự khác biệt giữa tác tử và quy trình làm việc cùng chi phí kỹ thuật tương ứng của từng hướng tiếp cận.
- Xác định khi nào nên chọn quy trình làm việc thay vì tác tử (và ngược lại).
- Hiện thực hóa cả năm mô hình bằng thư viện chuẩn Python trên một LLM theo kịch bản mẫu.

## Vấn đề

Nhiều đội ngũ kỹ thuật vội vàng tìm đến các framework đa tác tử phức tạp cho những bài toán vốn dĩ chỉ cần một lệnh gọi hàm đơn lẻ. Cái giá phải trả là có thật: các framework tạo ra quá nhiều lớp trừu tượng che khuất các câu lệnh nhắc, giấu kín luồng điều khiển, và gây ra sự phức tạp quá sớm (premature complexity). Bài viết vào tháng 12 năm 2024 của Schluntz và Zhang là lời cảnh tỉnh được trích dẫn nhiều nhất trong ngành: hãy bắt đầu từ sự đơn giản, chỉ bổ sung độ phức tạp khi nó thực sự xứng đáng với chi phí bỏ ra.

## Khái niệm

### Quy trình làm việc so với Tác tử (Workflows vs Agents)

- **Quy trình làm việc (Workflow).** Các mô hình LLM và công cụ được điều phối qua các đường dẫn mã nguồn định sẵn. Kỹ sư là người làm chủ đồ thị thực thi.
- **Tác tử (Agent).** Các mô hình LLM chủ động định hướng công cụ của riêng mình và tự quyết định các bước đi. Mô hình là thực thể làm chủ đồ thị thực thi.

Cả hai đều có chỗ đứng riêng. Quy trình làm việc tiết kiệm chi phí hơn, tốc độ phản hồi nhanh hơn và dễ dàng gỡ lỗi hơn. Tác tử mở ra khả năng giải quyết các bài toán mở nhưng lại khiến các dạng thức lỗi trở nên khó suy đoán và kiểm soát hơn.

### Mô hình LLM Tăng cường (The augmented LLM)

Nền tảng cho cả năm mô hình thiết kế: một LLM duy nhất được tích hợp sẵn ba năng lực cơ bản — tìm kiếm (truy xuất thông tin), công cụ (hành động tương tác), và bộ nhớ (lưu trữ bền vững). Mọi lệnh gọi API đều có thể khai thác các năng lực này.

### Năm mô hình quy trình làm việc

1. **Chuỗi lời nhắc (Prompt chaining).** Đầu ra của lượt gọi 1 là đầu vào của lượt gọi 2. Sử dụng khi một tác vụ có thể phân rã tuyến tính rõ ràng. Có thể chèn các cổng kiểm soát bằng chương trình giữa các bước.

2. **Định tuyến (Routing).** Một LLM phân loại đóng vai trò quyết định LLM hoặc công cụ nào ở hạ nguồn sẽ được kích hoạt. Sử dụng khi các nhóm đầu vào khác nhau về bản chất đòi hỏi cách xử lý khác nhau (hỗ trợ kỹ thuật cấp 1 so với hoàn tiền, báo lỗi phần mềm hoặc bán hàng).

3. **Song song hóa (Parallelization).** Chạy N lượt gọi LLM đồng thời, sau đó tổng hợp kết quả. Có hai dạng thức: phân đoạn (xử lý các phần khác nhau) và bỏ phiếu (cùng một prompt, chạy N lần độc lập, lấy ý kiến đa số hoặc tổng hợp).

4. **Điều phối - Thực thi (Orchestrator-workers).** Một LLM điều phối sẽ chủ động quyết định những LLM công nhân nào cần chạy và tổng hợp kết quả đầu ra của chúng. Tương tự như vòng lặp tác tử nhưng điều phối viên không lặp lại vô hạn định.

5. **Đánh giá - Tối ưu (Evaluator-optimizer).** Một LLM đề xuất câu trả lời, một LLM khác tiến hành đánh giá nó. Lặp lại quá trình cho đến khi vượt qua bài đánh giá. Đây là dạng tổng quát hóa của kỹ thuật Tự Tinh chỉnh (Self-Refine - Bài học 05).

### Trường hợp Quy trình làm việc vượt trội Tác tử

- **Các tác vụ có tính dự đoán cao.** Nếu bạn có thể liệt kê trước các bước, bạn nên làm như vậy.
- **Các tác vụ bị giới hạn chi phí nghiêm ngặt.** Quy trình làm việc có số bước cố định có thể chặn trên; tác tử có thể rơi vào vòng lặp xoắn ốc gây bùng nổ chi phí.
- **Các tác vụ đòi hỏi tuân thủ quy chuẩn.** Kiểm toán viên muốn đọc một đồ thị rõ ràng bằng mã nguồn thay vì phải tự suy luận dựa trên chuỗi hành động ngẫu nhiên.

### Trường hợp Tác tử vượt trội Quy trình làm việc

- **Nghiên cứu mang tính mở.** Khi bước tiếp theo hoàn toàn phụ thuộc vào kết quả mà bước trước đó trả về.
- **Tác vụ có thời lượng biến thiên.** Các công việc kéo dài từ vài phút đến vài giờ mà số lượng bước thực thi không thể xác định trước.
- **Các lĩnh vực mới mẻ.** Khi bạn chưa xác định được đâu là quy trình làm việc đúng đắn — cần khám phá trước, chuẩn hóa thành quy trình sau.

### Kỹ nghệ Ngữ cảnh Đồng hành

Bài viết "Effective context engineering for AI agents" (Anthropic 2025) chuẩn hóa lĩnh vực bổ trợ liên quan: cửa sổ 200k token là một ngân sách tài nguyên chứ không phải là một chiếc thùng chứa vô đáy. Cần quyết định đưa những gì vào, khi nào nên cô đọng, và khi nào để ngữ cảnh mở rộng. Vấn đề này đã được trình bày chi tiết trong bài học nén ngữ cảnh của Phase 14.

```figure
workflow-chain
```

## Xây dựng Thực tế

`code/main.py` hiện thực hóa cả năm mô hình quy trình làm việc trên đối tượng `ScriptedLLM`:

- `prompt_chain(input, steps)` — thực thi tuần tự.
- `route(input, classifier, handlers)` — phân loại và điều phối xử lý.
- `parallel_vote(prompt, n, aggregator)` — N lượt chạy đồng thời, tổng hợp kết quả.
- `orchestrator_workers(task, workers)` — bộ điều phối lựa chọn công nhân.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — vòng lặp cho đến khi đạt chuẩn.

Chạy chương trình:

```bash
python3 code/main.py
```

Mỗi mô hình in ra vết thực thi của riêng mình. Tổng số dòng mã cho mỗi mô hình chỉ khoảng 10-15 dòng; trong khi chi phí kéo theo của một framework thường đo bằng hàng ngàn dòng mã.

## Ứng dụng Thực tế

- Sử dụng các lệnh gọi API trực tiếp cho hầu hết các tác vụ.
- Chỉ đưa framework vào khi mô hình thiết kế thực sự cần trạng thái bền vững (LangGraph), mô hình đồng thời actor-model (AutoGen v0.4), hoặc tạo khuôn mẫu theo vai trò (CrewAI).
- Sử dụng Claude Agent SDK khi bạn muốn tận dụng cấu trúc khung làm việc của Claude Code mà không phải mất công xây dựng lại từ đầu.

## Đóng gói Sản phẩm

`outputs/skill-workflow-picker.md` giúp lựa chọn mô hình quy trình làm việc chuẩn xác cho một mô tả tác vụ nhất định, bao gồm lý giải cho quyết định lựa chọn và lộ trình tái cấu trúc chuyển đổi sang tác tử nếu quy trình làm việc bị quá tải.

## Bài tập

1. Hiện thực hóa cơ chế định tuyến đi kèm ngưỡng tin cậy (confidence threshold). Dưới ngưỡng tin cậy -> chuyển tiếp lên cho con người can thiệp. Ngưỡng này nên đặt ở mức bao nhiêu cho một bài toán hỗ trợ khách hàng cấp 1?
2. Bổ sung cơ chế hết thời gian chờ (timeout) cho `parallel_vote`. Điều gì xảy ra khi một lệnh gọi bị treo? Bạn sẽ tổng hợp kết quả như thế nào khi bị thiếu phiếu bầu?
3. Biến `evaluator_optimizer` thành một bài toán bandit: lưu giữ top-2 kết quả tốt nhất qua các vòng lặp để một kết quả tệ ở các vòng sau không vô tình ghi đè lên kết quả tốt ở vòng trước đó.
4. Kết hợp chuỗi lời nhắc với định tuyến: một bộ định tuyến sẽ chọn một trong ba chuỗi xử lý. Đo lường chi phí token so với phương án dùng một câu prompt khổng lồ duy nhất.
5. Chọn một tính năng trong hệ thống sản xuất của bạn. Vẽ lại đồ thị quy trình làm việc. Đếm số bước thực thi. Liệu việc sử dụng một tác tử ở đây có thực sự mang lại kết quả tốt hơn không?

## Thuật ngữ Then chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Quy trình làm việc (Workflow) | "Luồng định sẵn" | Đồ thị các lệnh gọi LLM và công cụ do kỹ sư trực tiếp kiểm soát |
| Tác tử (Agent) | "AI tự chủ" | Đồ thị do mô hình kiểm soát; điều hướng công cụ một cách linh hoạt |
| LLM tăng cường (Augmented LLM) | "LLM có công cụ" | LLM + tìm kiếm + công cụ + bộ nhớ; đơn vị cơ sở nguyên tử |
| Chuỗi lời nhắc (Prompt chaining) | "Các lệnh gọi tuần tự" | Đầu ra của lệnh gọi N là đầu vào của lệnh gọi N+1 |
| Định tuyến (Routing) | "Điều phối phân loại" | Lựa chọn chuỗi/mô hình nào sẽ chịu trách nhiệm xử lý đầu vào |
| Song song hóa (Parallelization) | "Mở rộng phân nhánh" | N lệnh gọi đồng thời; tổng hợp bằng phân đoạn hoặc bỏ phiếu |
| Điều phối - Thực thi (Orchestrator-workers) | "Tác tử điều phối" | LLM điều phối lựa chọn các LLM chuyên biệt một cách linh hoạt |
| Đánh giá - Tối ưu (Evaluator-optimizer) | "Đề xuất và Giám khảo" | Lặp cho đến khi vượt qua kiểm duyệt; dạng tổng quát của Self-Refine |

## Tài liệu Tham khảo

- [Anthropic, Building Effective Agents (Tháng 12/2024)](https://www.anthropic.com/research/building-effective-agents) — Năm mô hình quy trình làm việc
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Kỹ nghệ ngữ cảnh bổ trợ
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — Khi các đồ thị có trạng thái chứng minh được giá trị
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — Sản phẩm hóa mô hình điều phối - thực thi
