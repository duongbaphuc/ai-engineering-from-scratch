# ReWOO và Plan-and-Execute: Tách rời quá trình lập kế hoạch

> ReAct đan xen suy nghĩ và hành động trong cùng một luồng. ReWOO tách biệt chúng: lập một kế hoạch toàn diện ngay từ đầu, sau đó mới thực thi. Tiết kiệm token gấp 5 lần, tăng +4% độ chính xác trên tập dữ liệu HotpotQA, và bạn có thể chưng cất (distill) bộ lập kế hoạch thành một mô hình 7B nhỏ gọn. Plan-and-Execute đã khái quát hóa mô hình này; Plan-and-Act mở rộng nó cho các tác vụ điều hướng web.

**Loại bài học:** Xây dựng (Build)
**Ngôn ngữ:** Python (stdlib)
**Điều kiện tiên quyết:** Phase 14 · 01 (Vòng lặp tác tử)
**Thời gian:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao sự phân tách Bộ lập kế hoạch (Planner) / Công nhân thực thi (Worker) / Bộ giải (Solver) của ReWOO giúp tiết kiệm token và cải thiện tính mạnh mẽ (robustness) so với vòng lặp đan xen của ReAct.
- Triển khai một đồ thị có hướng không chu trình cho kế hoạch (plan DAG), một bộ thực thi theo thứ tự phụ thuộc, và một bộ giải tổng hợp kết quả của worker — tất cả đều sử dụng thư viện chuẩn (stdlib).
- Quyết định thời điểm một tác vụ nên chạy theo mô hình lập kế hoạch rồi thực thi (plan-then-execute) so với ReAct đan xen, dựa trên khung tư duy "năm mẫu hình quy trình làm việc" năm 2026 của Anthropic.
- Nhận biết thời điểm cần dùng dữ liệu kế hoạch tổng hợp (synthetic plan data) của Plan-and-Act cho các tác vụ dài hạn trên web hoặc thiết bị di động.

## Vấn đề thực tế

Vòng lặp suy nghĩ-hành động-quan sát đan xen của ReAct rất đơn giản và linh hoạt, nhưng mỗi lệnh gọi công cụ đều phải mang theo toàn bộ ngữ cảnh trước đó — bao gồm mọi suy nghĩ đã diễn ra. Lượng token sử dụng tăng theo cấp số nhân (quadratically) theo độ sâu của tác vụ. Tệ hơn nữa: khi một công cụ gặp sự cố ở giữa vòng lặp, mô hình phải suy luận lại toàn bộ kế hoạch từ kết quả quan sát báo lỗi.

ReWOO (Xu và cộng sự, arXiv:2305.18323, tháng 5 năm 2023) đã nhận thấy điều này và đưa ra một hướng tiếp cận mang tính đột phá: lập kế hoạch cho toàn bộ tác vụ ngay từ đầu, thu thập bằng chứng song song và tổng hợp câu trả lời ở bước cuối cùng. Một lệnh gọi LLM để lập kế hoạch, N lệnh gọi công cụ để thu thập bằng chứng (có thể chạy song song), một lệnh gọi LLM để giải bài toán. Sự đánh đổi ở đây là tính linh hoạt giảm đi đôi chút (kế hoạch mang tính tĩnh) để đổi lấy hiệu quả token vượt trội và các trạng thái lỗi được cô lập rõ ràng hơn.

## Khái niệm cốt lõi

### Ba vai trò chính

```text
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (lệnh gọi công cụ, có thể song song)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Bộ lập kế hoạch (Planner) tạo ra một DAG. Mỗi nút xác định một công cụ, các đối số của nó và các nút trước đó mà nó phụ thuộc vào (các tham chiếu như `#E1`, `#E2`). Các công nhân (Workers) thực thi các nút theo thứ tự sắp xếp tô-pô (topological order). Bộ giải (Solver) sẽ ghép nối tất cả lại với nhau.

### Tại sao lại tiết kiệm gấp 5 lần token

ReAct làm tăng độ dài prompt tuyến tính theo số bước. Ở bước thứ 10, prompt chứa suy nghĩ 1 cộng hành động 1 cộng quan sát 1 cộng suy nghĩ 2 cộng hành động 2 cộng quan sát 2, v.v. Mỗi bước trung gian cũng lặp lại prompt ban đầu một cách dư thừa.

ReWOO chỉ chi trả một prompt cho planner (lớn), N prompt nhỏ cho worker (mỗi prompt chỉ chứa lệnh gọi công cụ, không có chuỗi suy luận dài dòng) và một prompt cho solver. Trên tập dữ liệu HotpotQA, bài báo ghi nhận mức tiêu thụ token ít hơn ~5 lần trong khi độ chính xác tuyệt đối tăng +4%.

### Tại sao mô hình này mạnh mẽ hơn

Nếu worker thứ 3 thất bại trong ReAct, vòng lặp phải suy luận để thoát khỏi lỗi ngay giữa dòng thực thi. Trong ReWOO, worker thứ 3 trả về một chuỗi thông báo lỗi; solver nhìn thấy lỗi đó trong ngữ cảnh của toàn bộ kế hoạch ban đầu và có thể hạ cấp hoạt động một cách mềm dẻo (graceful degradation). Việc khoanh vùng sự cố được thực hiện trên từng nút (per-node) chứ không phải trên từng bước (per-step).

### Chưng cất bộ lập kế hoạch (Planner distillation)

Kết quả thứ hai của bài báo: vì planner không nhìn thấy các quan sát, bạn có thể tinh chỉnh (fine-tune) một mô hình 7B dựa trên đầu ra của planner từ một mô hình giáo viên (teacher model) 175B. Mô hình nhỏ sẽ đảm nhận việc lập kế hoạch; mô hình lớn không còn cần thiết ở giai đoạn suy luận (inference). Đây hiện là tiêu chuẩn — nhiều tác tử sản xuất năm 2026 sử dụng một planner nhỏ và một executor lớn hoặc ngược lại.

### Plan-and-Execute (2023)

Bài viết tháng 8 năm 2023 của nhóm LangChain đã khái quát hóa ReWOO thành một tên gọi mẫu hình: Plan-and-Execute (Lập kế hoạch và Thực thi). Bộ lập kế hoạch ban đầu phát ra một danh sách các bước, bộ thực thi chạy từng bước, và một bộ tái lập kế hoạch (replanner) tùy chọn có thể điều chỉnh kế hoạch sau khi quan sát kết quả. Cách tiếp cận này gần với ReAct hơn ReWOO (replanner đưa các quan sát quay trở lại quá trình lập kế hoạch) nhưng vẫn giữ được lợi ích tiết kiệm token.

### Plan-and-Act (Erdogan và cộng sự, arXiv:2503.09572, ICML 2025)

Plan-and-Act mở rộng quy mô mẫu hình này cho các tác tử web và di động hoạt động dài hạn (long-horizon). Đóng góp chính là dữ liệu kế hoạch tổng hợp: một trình tạo quỹ đạo có gán nhãn tạo ra dữ liệu huấn luyện trong đó kế hoạch được xác định tường minh. Dữ liệu này dùng để tinh chỉnh các mô hình planner tiếp tục hoạt động hiệu quả vượt qua 30–50 bước trên các tác vụ kiểu WebArena, nơi một quỹ đạo ReAct đơn lẻ dễ mất đi tính mạch lạc.

### Khi nào nên chọn mô hình nào

| Mẫu hình | Khi nào nên dùng |
|----------|-------------------|
| ReAct | Tác vụ ngắn, môi trường chưa biết trước, cần xử lý ngoại lệ mang tính phản ứng |
| ReWOO | Tác vụ có cấu trúc với các công cụ đã biết, nhạy cảm về token, bằng chứng có thể thu thập song song |
| Plan-and-Execute | Tương tự ReWOO nhưng có khả năng tái lập kế hoạch sau khi thực thi từng phần |
| Plan-and-Act | Tác vụ dài hạn (>30 bước), tương tác web/di động/sử dụng máy tính |
| Tree of Thoughts | Khi việc tìm kiếm lời giải xứng đáng với chi phí bỏ ra (Bài 04) |

Hướng dẫn tháng 12 năm 2024 của Anthropic: hãy bắt đầu với giải pháp đơn giản nhất. Nếu tác vụ chỉ là một lệnh gọi công cụ kèm theo phần tóm tắt, đừng xây dựng ReWOO. Nếu tác vụ là một bài tập nghiên cứu dài 40 bước, đừng chỉ dựa vào ReAct đơn lẻ.

```figure
rewoo-plan
```

## Xây dựng thực tế

Tệp `code/main.py` triển khai một phiên bản ReWOO mô phỏng:

- `Planner` — một chính sách bằng kịch bản phát ra một DAG kế hoạch từ prompt.
- `Worker` — điều phối lệnh gọi công cụ của từng nút thông qua sổ đăng ký.
- `Solver` — bộ tổng hợp theo kịch bản đọc bằng chứng và tạo ra câu trả lời cuối cùng.
- Giải quyết phụ thuộc — các tham chiếu như `#E1` được thay thế bằng đầu ra của các worker trước đó.

Bản demo trả lời câu hỏi "Dân số của thủ đô nước Pháp là bao nhiêu, làm tròn đến hàng triệu?" bằng một kế hoạch hai bước: (1) tra cứu thủ đô, (2) tra cứu dân số, sau đó giải bài toán.

Chạy thử:

```bash
python3 code/main.py
```

Bản ghi vết thực thi hiển thị toàn bộ kế hoạch trước, sau đó là kết quả của worker, tiếp theo là phần tổng hợp của solver. Hãy so sánh số lượng token (chúng tôi in ra số ký tự ước tính) với một lần chạy đan xen kiểu ReAct — ReWOO hoàn toàn chiến thắng trên những tác vụ có cấu trúc như thế này.

## Sử dụng trong thực tế

LangGraph cung cấp Plan-and-Execute dưới dạng một công thức dựng sẵn (`create_react_agent` cho ReAct, các đồ thị tùy chỉnh cho plan-execute). Tính năng Flows của CrewAI mã hóa trực tiếp mẫu hình này: bạn định nghĩa các tác vụ từ đầu và Flow DAG sẽ thực thi chúng. Cách tiếp cận dữ liệu tổng hợp của Plan-and-Act phần lớn vẫn nằm trong phạm vi nghiên cứu; mẫu hình runtime (DAG kế hoạch tường minh) được triển khai trong môi trường sản xuất thông qua LangGraph và CrewAI Flows.

## Triển khai thành phẩm

Tệp `outputs/skill-rewoo-planner.md` tạo ra một DAG kế hoạch ReWOO từ yêu cầu của người dùng, dựa trên một danh mục công cụ cho trước. Nó xác thực kế hoạch (không chu trình, mọi tham chiếu đều được giải quyết, mọi công cụ đều tồn tại) trước khi chuyển giao cho bộ thực thi.

## Bài tập rèn luyện

1. Song song hóa việc thực thi worker cho các nút kế hoạch độc lập. Việc này mang lại lợi ích gì trên một DAG gồm 6 nút với 2 nhóm song song?
2. Thêm một nút replanner kích hoạt nếu bất kỳ worker nào trả về lỗi. Thay đổi nhỏ nhất nào đối với ReWOO sẽ biến nó thành Plan-and-Execute?
3. Thay thế `Planner` bằng một mô hình nhỏ (cỡ 7B) và giữ `Solver` trên một mô hình tiên tiến (frontier model). So sánh chất lượng đầu-cuối — sự phân tách này bộc lộ điểm yếu ở đâu?
4. Đọc Phần 4 của bài báo ReWOO về việc chưng cất bộ lập kế hoạch. Tái hiện lại kết quả 175B -> 7B về mặt khái niệm: bạn cần dữ liệu huấn luyện nào và bạn đánh giá chất lượng kế hoạch ra sao?
5. Chuyển đổi bản mô phỏng sang dạng quỹ đạo của Plan-and-Act: kế hoạch là một chuỗi tuần tự chứ không phải là một DAG. Những đánh đổi nào thay đổi?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất thực sự |
|-----------|----------------------|------------------|
| ReWOO | "Lập luận không cần quan sát (Reasoning without observations)" | Lập kế hoạch, sau đó thu thập bằng chứng song song, rồi giải — không có quan sát trong prompt lập kế hoạch |
| Plan-and-Execute | "Mẫu hình lập kế hoạch - thực thi của LangChain" | ReWOO với một nút tái lập kế hoạch tùy chọn sau khi thực thi |
| Plan-and-Act | "Mô hình plan-execute quy mô lớn" | Tách biệt tường minh planner/executor với dữ liệu huấn luyện kế hoạch tổng hợp cho các tác vụ dài hạn |
| Tham chiếu bằng chứng (Evidence reference) | "#E1, #E2, ..." | Ký hiệu giữ chỗ của nút kế hoạch được thay thế bằng đầu ra của worker trước đó tại thời điểm điều phối |
| Chưng cất bộ lập kế hoạch (Planner distillation) | "Planner nhỏ, executor lớn" | Tinh chỉnh một mô hình nhỏ trên các vết lập kế hoạch từ một mô hình giáo viên lớn |
| Hiệu quả token (Token efficiency) | "Ít lượt trao đổi khứ hồi hơn" | Tiêu thụ ít hơn 5 lần token trên HotpotQA so với ReAct trong bài báo |
| Bộ thực thi DAG (DAG executor) | "Bộ điều phối tô-pô" | Chạy các nút kế hoạch theo thứ tự phụ thuộc; chạy song song ở mỗi cấp |

## Tài liệu đọc thêm

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — bài báo kinh điển
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — mô hình planner-executor mở rộng với kế hoạch tổng hợp
- [Hướng dẫn Plan-and-Execute của LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — công thức triển khai của framework
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — chọn mẫu hình đơn giản nhất phát huy hiệu quả
