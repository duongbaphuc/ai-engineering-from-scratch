# Tree of Thoughts và LATS: Tìm kiếm có chủ đích

> Một quỹ đạo chuỗi suy nghĩ đơn lẻ không có cơ hội quay lui (backtrack). ToT (Yao và cộng sự, 2023) biến quá trình lập luận thành một cấu trúc cây với khả năng tự đánh giá trên từng nút. LATS (Zhou và cộng sự, 2024) hợp nhất ToT với ReAct và Reflexion dưới nền tảng Tìm kiếm Cây Monte Carlo (Monte Carlo Tree Search - MCTS). Trò chơi 24 (Game of 24) tăng độ chính xác từ 4% (CoT) lên 74% (ToT); LATS đạt 92,7% pass@1 trên tập dữ liệu HumanEval.

**Loại bài học:** Xây dựng (Build)
**Ngôn ngữ:** Python (stdlib)
**Điều kiện tiên quyết:** Phase 14 · 01 (Vòng lặp tác tử), Phase 14 · 03 (Reflexion)
**Thời gian:** ~75 phút

## Mục tiêu học tập

- Định hình quá trình lập luận như một bài toán tìm kiếm: các nút là "suy nghĩ" (thoughts), các cạnh là "phép mở rộng" (expansions), giá trị là "mức độ triển vọng" (promising).
- Triển khai thuật toán tìm kiếm cây theo chiều rộng (BFS) kiểu ToT với cơ chế chấm điểm tự đánh giá bằng thư viện chuẩn (stdlib).
- Mở rộng thành một vòng lặp LATS MCTS mô phỏng với đầy đủ bốn bước: chọn lọc (select) / mở rộng (expand) / mô phỏng (simulate) / lan truyền ngược (backpropagate).
- Quyết định thời điểm tìm kiếm xứng đáng với chi phí token nhân bội (Trò chơi 24, sinh mã nguồn) và khi nào một quỹ đạo đơn lẻ là đủ (hỏi đáp đơn giản).

## Vấn đề thực tế

Chuỗi suy nghĩ (CoT - Chain of Thought) là một bước đi tuần tự tuyến tính. Nếu bước đầu tiên sai, mọi bước tiếp theo đều vận hành trên một tiền đề sai lầm. Trong Trò chơi 24 (sử dụng bốn chữ số với các phép tính + − × ÷ để tạo ra 24), GPT-4 sử dụng CoT chỉ đạt độ chính xác khiêm tốn 4%. Mô hình chọn sai biểu thức con ngay từ đầu và không thể phục hồi.

Điều mà quá trình lập luận thực sự cần là khả năng đề xuất nhiều ứng viên, đánh giá chúng, chọn ra những phương án triển vọng và quay lui khi gặp ngõ cụt. Đó chính là bản chất của tìm kiếm. Tree of Thoughts (Cây suy nghĩ) và LATS (Language Agent Tree Search) là hai công thức chuẩn mực nhất để hiện thực hóa điều này.

## Khái niệm cốt lõi

### Tree of Thoughts (Yao và cộng sự, NeurIPS 2023)

Mỗi nút là một bước trung gian mạch lạc ("một suy nghĩ"). Mỗi nút có thể mở rộng thành K suy nghĩ con. LLM tự đánh giá từng nút bằng một prompt chấm điểm. Quá trình tìm kiếm khám phá cây thông qua BFS, DFS (tìm kiếm theo chiều sâu) hoặc tìm kiếm chùm (beam search).

```text
                     (gốc: "tìm 24 từ 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Điểm: CAO
              /   \              |                  |
          ...    ...          ...                hoàn tất
```

Tự đánh giá là mắt xích chịu tải chính. Bài báo chỉ ra ba biến thể: phân loại `chắc chắn / có thể / bất khả thi` (sure / likely / impossible), thang điểm số `1..10`, và bỏ phiếu lựa chọn giữa các ứng viên. Cả ba cách tiếp cận đều vượt trội đáng kể so với CoT trên Trò chơi 24 (từ 4% -> 74% với GPT-4).

### LATS (Zhou và cộng sự, ICML 2024)

LATS hợp nhất ToT, ReAct và Reflexion dưới khuôn khổ MCTS. LLM đóng ba vai trò:

- **Chính sách (Policy)**: đề xuất các hành động ứng viên tiếp theo (theo phong cách ReAct).
- **Hàm giá trị (Value function)**: chấm điểm quỹ đạo từng phần (tự đánh giá theo phong cách ToT).
- **Bộ tự phản tỉnh (Self-reflector)**: khi thất bại, viết phản tỉnh bằng ngôn ngữ tự nhiên (theo phong cách Reflexion) và dùng nó để gieo mầm lại cho các đợt mở rộng tương lai.

Phản hồi từ môi trường (các quan sát) được hòa trộn vào hàm giá trị để việc tìm kiếm được dẫn dắt bởi kết quả công cụ thực tế chứ không chỉ dựa vào nhận định chủ quan của mô hình. Kết quả tại thời điểm công bố bài báo: HumanEval pass@1 đạt 92,7% với GPT-4 (SOTA), WebShop đạt trung bình 75,9 với GPT-3.5 (tiệm cận mức tinh chỉnh dựa trên gradient).

### MCTS tối giản

Bốn giai đoạn trong mỗi lần lặp:

1. **Chọn lọc (Select)** — duyệt từ nút gốc đến một nút lá bằng công thức UCT (Upper Confidence Bound applied to Trees).
2. **Mở rộng (Expand)** — tạo ra K nút con thông qua chính sách.
3. **Mô phỏng (Simulate)** — chạy tiếp từ một nút con bằng chính sách (rollout), chấm điểm nút lá bằng hàm giá trị (hoặc phần thưởng từ môi trường).
4. **Lan truyền ngược (Backpropagate)** — cập nhật số lần ghé thăm và ước lượng giá trị ngược lên toàn bộ đường dẫn.

Công thức UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Số hạng đầu tiên đại diện cho việc khai thác (exploitation); số hạng thứ hai đại diện cho việc khám phá (exploration). Điều chỉnh tham số `c` tùy theo từng tác vụ.

### Thực tế về chi phí

Tìm kiếm gây bùng nổ lượng token tiêu thụ. ToT trên Trò chơi 24 sử dụng lượng token gấp 100–1000 lần so với CoT. LATS cũng tương tự. Điều này không hề miễn phí; hãy dành riêng cơ chế tìm kiếm cho:

- Các tác vụ mà một quỹ đạo đơn lẻ rõ ràng không thể đáp ứng (Trò chơi 24, mã nguồn phức tạp).
- Các tác vụ mà thời gian thực thi (wall-clock) ít quan trọng hơn tính đúng đắn.
- Các tác vụ có hàm giá trị rẻ và đáng tin cậy (các bài kiểm thử đơn vị cho code, mục tiêu tường minh cho toán học).

Nếu tác vụ của bạn chỉ có một câu trả lời đúng duy nhất nhưng bộ đánh giá bị nhiễu, việc tìm kiếm thường làm mọi thứ tệ hơn — nó sẽ tìm ra một câu trả lời sai nhưng "được chấm điểm cao".

### Vị thế trong năm 2026

Hầu hết các tác tử trong môi trường sản xuất không chạy LATS. Chúng chạy ReAct kết hợp với cơ chế xác minh có neo công cụ (CRITIC, Bài 05). Tìm kiếm chỉ xuất hiện trong các phân khúc chuyên biệt:

- Các tác tử lập trình chạy kiểm thử làm hàm giá trị (kiểu HumanEval).
- Các tác tử nghiên cứu sâu (deep-research) khám phá nhiều hướng truy vấn khác nhau.
- Các quy trình làm việc chú trọng lập kế hoạch bên trong các đồ thị con của LangGraph.

AlphaEvolve (Bài 11) là đỉnh cao vào năm 2025: tìm kiếm tiến hóa trên mã nguồn, kiểm tra độ thích nghi bằng máy móc, mang lại đột phá tiên phong (cải tiến phép nhân ma trận 4x4 đầu tiên sau 56 năm).

```figure
tree-of-thoughts
```

## Xây dựng thực tế

Tệp `code/main.py` triển khai:

- Một cấu trúc ToT BFS thu nhỏ trên tác vụ "chọn các toán tử số học".
- Một vòng lặp LATS MCTS mô phỏng trên cùng tác vụ (Select / Expand / Simulate / Backpropagate) với cơ chế chọn lọc UCT.
- Một hàm giá trị tổng hợp giữa điểm số mang tính ký hiệu (symbolic score) cộng với điểm tự đánh giá.

Chạy thử:

```bash
python3 code/main.py
```

Bản ghi vết thực thi minh họa ToT mở rộng ba ứng viên trên mỗi nút bằng BFS, so sánh với LATS hội tụ về quỹ đạo mở rộng tốt nhất thông qua MCTS. Số lượng token tiêu thụ được in ra cho cả hai phương pháp.

## Sử dụng trong thực tế

LangGraph cung cấp khả năng khám phá kiểu ToT dưới dạng các mẫu đồ thị con; bài viết hướng dẫn của nhóm LangChain về LATS (tháng 5 năm 2024) là tài liệu tham khảo chuẩn mực. LlamaIndex cung cấp một tác tử `TreeOfThoughts`. Đối với hầu hết các tác tử sản xuất năm 2026, mẫu hình này được đặt sau một cổng điều kiện `if task_complexity > threshold: use_search()` — xem mẫu hình evaluator-optimizer trong Bài 05.

## Triển khai thành phẩm

Tệp `outputs/skill-search-policy.md` lựa chọn giữa ReAct tuyến tính, ToT, LATS và tìm kiếm tiến hóa dựa trên cấu trúc tác vụ, ngân sách và độ chuẩn xác của bộ đánh giá.

## Bài tập rèn luyện

1. Chạy LATS mô phỏng với tham số UCT c=0.1 so với c=2.0. Vết thực thi có gì thay đổi?
2. Thay thế hàm giá trị bằng một bộ chấm điểm có nhiều nhiễu hơn (thêm dao động ngẫu nhiên). Liệu MCTS có còn tìm thấy nút lá tốt nhất không? Tỷ lệ tín hiệu trên nhiễu tối thiểu mà nó có thể chịu đựng là bao nhiêu?
3. Triển khai ToT tìm kiếm chùm (beam-search ToT - giữ lại top-k ở mỗi cấp) và so sánh với BFS. Phương pháp nào tốt hơn khi ngân sách token bị thắt chặt?
4. Đọc Phần 5.1 của bài báo LATS. Tái hiện lại số lượng quỹ đạo trên HumanEval: cần bao nhiêu lần mở rộng rollout để đạt được chỉ số pass@1 như đã công bố?
5. Đọc phần thảo luận của bài báo LATS về "khi nào LATS ít mang lại lợi ích hơn". Viết một đoạn quy tắc quyết định ánh xạ cấu trúc tác vụ sang chiến lược tìm kiếm tương ứng.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất thực sự |
|-----------|----------------------|------------------|
| Tree of Thoughts | "CoT phân nhánh" | Yao và cộng sự — cây chứa các nút suy nghĩ có khả năng tự đánh giá |
| LATS | "MCTS cho LLM" | Zhou và cộng sự — hợp nhất ToT + ReAct + Reflexion dưới khuôn khổ MCTS |
| UCT | "Giới hạn tin cậy trên" | Công thức chọn lọc cân bằng giữa khai thác (Q) và khám phá (ln N / n) |
| Hàm giá trị (Value function) | "Trạng thái này tốt đến mức nào" | Điểm số từ prompt LLM hoặc phần thưởng môi trường; làm dữ liệu cho lan truyền ngược |
| Chính sách (Policy) | "Bộ đề xuất hành động" | Bộ sinh theo kiểu ReAct; phát ra các suy nghĩ/hành động ứng viên tiếp theo |
| Mở rộng mô phỏng (Rollout) | "Quỹ đạo mô phỏng" | Duyệt từ một nút đến một nút lá bằng chính sách, chấm điểm bằng hàm giá trị |
| Lan truyền ngược (Backpropagate) | "Cập nhật tổ tiên" | Đẩy phần thưởng của nút lá ngược lên đường dẫn, cập nhật số lần ghé thăm và giá trị Q |
| Chi phí tìm kiếm (Search cost) | "Bùng nổ token" | Tiêu tốn gấp 100-1000 lần CoT trên Trò chơi 24; cần lập ngân sách kỹ lưỡng trước khi áp dụng |

## Tài liệu đọc thêm

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — bài báo kinh điển
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS với phản hồi Reflexion
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — các mẫu đồ thị con cho tìm kiếm
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm tiến hóa với các bộ đánh giá lập trình được
