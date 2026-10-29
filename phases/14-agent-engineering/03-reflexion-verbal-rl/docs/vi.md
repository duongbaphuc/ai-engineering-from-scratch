# Reflexion: Học tăng cường bằng ngôn từ

> Học tăng cường dựa trên gradient cần hàng nghìn lượt thử nghiệm và một cụm GPU mạnh mẽ để khắc phục một dạng lỗi. Reflexion (Shinn và cộng sự, NeurIPS 2023) thực hiện điều đó bằng ngôn ngữ tự nhiên: sau mỗi lần thử thất bại, tác tử sẽ viết một lời phản tỉnh (reflection), lưu trữ vào bộ nhớ tình tiết (episodic memory), và điều hướng lần thử tiếp theo dựa trên bộ nhớ đó. Đây chính là mẫu hình đằng sau cơ chế tính toán lúc nghỉ (sleep-time compute) của Letta, các bài học trong `CLAUDE.md` của Claude Code, và lệnh learn-rule của pro-workflow.

**Loại bài học:** Xây dựng (Build)
**Ngôn ngữ:** Python (stdlib)
**Điều kiện tiên quyết:** Phase 14 · 01 (Vòng lặp tác tử), Phase 14 · 02 (ReWOO)
**Thời gian:** ~60 phút

## Mục tiêu học tập

- Nêu tên ba thành phần của Reflexion (Actor, Evaluator, Self-Reflector) và vai trò của bộ nhớ tình tiết (episodic memory).
- Triển khai một vòng lặp Reflexion bằng thư viện chuẩn (stdlib) với bộ đánh giá nhị phân, bộ đệm phản tỉnh và các lượt thử lại mới.
- Lựa chọn giữa các nguồn phản hồi vô hướng (scalar), heuristic và tự đánh giá (self-evaluated) cho một tác vụ nhất định.
- Giải thích lý do tại sao học tăng cường bằng ngôn từ (verbal reinforcement) phát hiện và sửa các lỗi mà RL dựa trên gradient phải mất hàng nghìn lượt thử nghiệm mới giải quyết được.

## Vấn đề thực tế

Một tác tử thất bại trong một tác vụ. Trong học tăng cường (RL - Reinforcement Learning) tiêu chuẩn, bạn sẽ phải chạy hàng nghìn lượt thử nghiệm nữa, tính toán gradient và cập nhật trọng số. Quá trình này tốn kém, chậm chạp và hầu hết các tác tử trong môi trường sản xuất không có ngân sách huấn luyện cho từng lỗi riêng lẻ.

Reflexion (Shinn và cộng sự, arXiv:2303.11366) đặt ra một câu hỏi khác: điều gì sẽ xảy ra nếu tác tử chỉ cần suy nghĩ về lý do tại sao nó thất bại và thử lại với suy nghĩ đó ngay trong prompt của nó? Không cập nhật trọng số. Không cần gradient. Chỉ đơn thuần là ngôn ngữ tự nhiên được lưu trữ giữa các lần thử.

Kết quả: trên tập dữ liệu ALFWorld, nó đánh bại ReAct và các phương pháp cơ sở không qua tinh chỉnh khác. Trên HotpotQA, nó cải thiện vượt trội so với ReAct. Trên tác vụ sinh mã nguồn (HumanEval/MBPP), nó đã thiết lập trạng thái tân tiến nhất (state-of-the-art) tại thời điểm công bố. Tất cả đều đạt được mà không cần một bước cập nhật gradient nào.

## Khái niệm cốt lõi

### Ba thành phần chính

```text
Actor         : tạo ra một quỹ đạo (vòng lặp kiểu ReAct)
Evaluator     : chấm điểm quỹ đạo — nhị phân, heuristic, hoặc tự đánh giá
Self-Reflector: viết một lời phản tỉnh bằng ngôn ngữ tự nhiên về thất bại
```

Cùng với một cấu trúc dữ liệu:

```text
Episodic memory: danh sách các phản tỉnh trước đó, được chèn vào đầu prompt của lần thử tiếp theo
```

Một lần thử sẽ kích hoạt Actor. Evaluator chấm điểm lần chạy đó. Nếu điểm số thấp, Self-Reflector sẽ tạo ra một lời phản tỉnh ("Tôi đã chọn nhầm công cụ vì đọc nhầm câu hỏi là hỏi về X trong khi thực chất đang hỏi về Y"). Lời phản tỉnh này được đưa vào bộ nhớ tình tiết. Lần thử tiếp theo bắt đầu lại từ đầu nhưng có thêm thông tin phản tỉnh đó trong ngữ cảnh.

### Ba loại bộ đánh giá

1. **Vô hướng (Scalar)** — một tín hiệu nhị phân từ bên ngoài. ALFWorld thành công hay thất bại. Các bài kiểm thử HumanEval vượt qua hay thất bại. Đây là loại đơn giản nhất và có độ chuẩn xác tín hiệu cao nhất.
2. **Heuristic** — các dấu hiệu nhận diện lỗi được định nghĩa trước. "Nếu tác tử thực hiện cùng một hành động hai lần liên tiếp, đánh dấu là bị mắc kẹt." "Nếu quỹ đạo vượt quá 50 bước, đánh dấu là không hiệu quả."
3. **Tự đánh giá (Self-evaluated)** — LLM tự chấm điểm quỹ đạo của chính nó. Cần thiết khi không có sẵn nhãn chân lý thực tế (ground truth). Tín hiệu này yếu hơn; kết hợp rất tốt với xác minh có neo công cụ (Bài 05 — CRITIC).

Tiêu chuẩn mặc định năm 2026 là sự kết hợp: dùng vô hướng khi có sẵn, tự đánh giá khi không có, và heuristic đóng vai trò như các rào chắn an toàn (safety rails).

### Tại sao phương pháp này có tính khái quát hóa cao

Reflexion không hẳn là một thuật toán mới hoàn toàn mà là một mẫu hình đã được định danh. Hầu như mọi tác tử "tự phục hồi" (self-healing) trong môi trường sản xuất đều chạy một biến thể nào đó của nó:

- Tính toán lúc nghỉ (Sleep-time compute) của Letta (Bài 08): một tác tử riêng biệt phản tỉnh về các cuộc hội thoại trong quá khứ và ghi vào các khối bộ nhớ.
- Mẫu hình lưu bộ nhớ / `CLAUDE.md` của Claude Code: các phản tỉnh được ghi lại dưới dạng bài học kinh nghiệm, đưa vào đầu các phiên làm việc trong tương lai.
- Lệnh `/learn-rule` của pro-workflow: các sửa đổi được ghi lại dưới dạng các quy tắc tường minh.
- Các nút phản tỉnh trong LangGraph: một nút chấm điểm đầu ra và điều hướng đến bước tinh chỉnh nếu cần.

Tất cả đều bắt nguồn từ cùng một nhận thức sâu sắc: ngôn ngữ tự nhiên là một phương tiện đủ phong phú để truyền tải "những gì tôi đã học được từ thất bại" giữa các lần chạy.

### Khi nào phát huy hiệu quả và khi nào không

Reflexion hoạt động tốt khi:

- Có tín hiệu thất bại rõ ràng (thất bại kiểm thử, lỗi công cụ, câu trả lời sai).
- Lớp tác vụ có tính tái lập (cùng một kiểu câu hỏi có thể được lặp lại).
- Lời phản tỉnh có đủ không gian để cải thiện quỹ đạo (đủ ngân sách hành động).

Reflexion không giúp ích khi:

- Tác tử đã thành công ngay trong lần thử đầu tiên.
- Lỗi mang tính khách quan từ bên ngoài (mạng bị ngắt, công cụ bị hỏng) — phản tỉnh về việc "mạng bị ngắt" không giúp ích gì cho các lần chạy tương lai.
- Lời phản tỉnh biến thành sự mê tín — lưu trữ một câu chuyện giải thích cho một lần chạy bất thường ngẫu nhiên (flaky run).

Cạm bẫy năm 2026: thoái hóa bộ nhớ (memory rot). Các lời phản tỉnh tích tụ dần; một số trở nên lỗi thời hoặc sai lệch; các lần chạy lại chậm hơn khi bộ đệm tình tiết phình to. Giải pháp giảm thiểu: nén bộ nhớ định kỳ (compaction - Bài 06), đặt thời gian tồn tại (TTL) cho các phản tỉnh, hoặc dùng một tác tử dọn dẹp riêng biệt chạy lúc nghỉ (Letta).

```figure
react-trace
```

## Xây dựng thực tế

Tệp `code/main.py` triển khai Reflexion trên một câu đố mô phỏng: tạo ra một danh sách 3 phần tử có tổng bằng một mục tiêu cho trước. Actor phát ra các danh sách ứng viên; Evaluator kiểm tra tổng; Self-Reflector viết một dòng phân tích điều gì đã sai. Lời phản tỉnh được đưa vào bộ nhớ tình tiết cho lần thử tiếp theo.

Các thành phần:

- `Actor` — một chính sách bằng kịch bản tự cải thiện khi nhìn thấy các lời phản tỉnh.
- `Evaluator.binary()` — kiểm tra đạt/không đạt (pass/fail) dựa trên tổng mục tiêu.
- `SelfReflector` — tạo ra một dòng chẩn đoán về nguyên nhân thất bại.
- `EpisodicMemory` — một danh sách có giới hạn với ngữ nghĩa thời gian tồn tại (TTL).

Chạy thử:

```bash
python3 code/main.py
```

Vết thực thi hiển thị ba lần thử. Lần 1 thất bại, một phản tỉnh được lưu lại; lần 2 nhìn thấy phản tỉnh và cải thiện nhưng vẫn thất bại; lần 3 thành công. Hãy so sánh với lần chạy cơ sở (không có phản tỉnh) — nó sẽ tiếp tục bị mắc kẹt ở câu trả lời của lần thử 1.

## Sử dụng trong thực tế

LangGraph cung cấp phản xạ dưới dạng một mẫu nút (node pattern). Lệnh `/memory` của Claude Code và `/learn-rule` của pro-workflow biểu diễn bộ đệm tình tiết ra bên ngoài dưới dạng một tệp markdown. Cơ chế tính toán lúc nghỉ của Letta chạy Self-Reflector trong thời gian rảnh rỗi để tác tử chính duy trì độ trễ tối ưu. OpenAI Agents SDK không cung cấp trực tiếp Reflexion; bạn xây dựng nó bằng một Guardrail tùy chỉnh từ chối quỹ đạo theo điểm số và một `Session` bộ nhớ tồn tại qua các lần chạy.

## Triển khai thành phẩm

Tệp `outputs/skill-reflexion-buffer.md` tạo và duy trì một bộ đệm tình tiết với khả năng thu thập phản tỉnh, quản lý TTL và khử trùng lặp (deduplication). Khi nhận vào một lớp tác vụ và một thất bại, nó phát ra một lời phản tỉnh thực sự giúp ích cho lần thử tiếp theo (chứ không phải một lời khuyên chung chung kiểu "hãy cẩn thận hơn").

## Bài tập rèn luyện

1. Chuyển từ bộ đánh giá nhị phân sang bộ đánh giá vô hướng trả về chỉ số khoảng cách (cách mục tiêu bao xa). Nó có giúp hội tụ nhanh hơn không?
2. Thêm TTL bằng 10 lần thử cho các phản tỉnh. Các phản tỉnh cũ hơn sẽ gây hại hay có lợi sau mốc thời gian đó?
3. Triển khai bộ đánh giá heuristic: đánh dấu lần thử là bị kẹt nếu cùng một hành động lặp lại. Điều này tương tác với Self-Reflector như thế nào?
4. Chạy Reflexion với một Actor đối kháng (adversarial Actor) cố tình phớt lờ các phản tỉnh. Kỹ nghệ prompt tối thiểu nào có thể buộc Actor phải chú ý đến chúng?
5. Đọc Phần 4 của bài báo Reflexion trên tập dữ liệu AlfWorld. Tái hiện lại mức cải thiện 130% tỷ lệ thành công về mặt khái niệm: đâu là điểm khác biệt then chốt so với ReAct nguyên bản?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất thực sự |
|-----------|----------------------|------------------|
| Reflexion | "Tự sửa sai (Self-correction)" | Shinn và cộng sự 2023 — Actor, Evaluator, Self-Reflector cộng với bộ nhớ tình tiết |
| Học tăng cường bằng ngôn từ (Verbal reinforcement) | "Học không cần gradient" | Lời phản tỉnh bằng ngôn ngữ tự nhiên được chèn vào đầu prompt của lần thử tiếp theo |
| Bộ nhớ tình tiết (Episodic memory) | "Phản tỉnh theo từng tác vụ" | Bộ đệm có giới hạn chứa các phản tỉnh trước đó cho một lớp tác vụ cụ thể |
| Bộ đánh giá vô hướng (Scalar evaluator) | "Tín hiệu thành công nhị phân" | Đạt/không đạt hoặc điểm số bằng số từ nhãn chân lý thực tế |
| Bộ đánh giá heuristic (Heuristic evaluator) | "Bộ phát hiện dựa trên mẫu hình" | Các dấu hiệu thất bại được định nghĩa trước (ví dụ: vòng lặp bị kẹt, quá nhiều bước) |
| Bộ tự đánh giá (Self-evaluator) | "LLM làm giám khảo trên dấu vết của chính nó" | Phương án dự phòng có độ chuẩn xác tín hiệu thấp hơn khi không có chân lý thực tế — nên kết hợp với xác minh có neo công cụ |
| Thoái hóa bộ nhớ (Memory rot) | "Phản tỉnh ôi thiu/lỗi thời" | Bộ đệm tình tiết bị lấp đầy bởi các mục lỗi thời; khắc phục bằng cách nén (compaction) hoặc TTL |
| Phản tỉnh lúc nghỉ (Sleep-time reflection) | "Tự phản tỉnh bất đồng bộ" | Chạy Self-Reflector ngoài đường dẫn xử lý chính (hot path) để tác tử chính luôn giữ tốc độ nhanh |

## Tài liệu đọc thêm

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — bài báo kinh điển
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — phản tỉnh bất đồng bộ trong môi trường sản xuất
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — quản lý bộ đệm tình tiết như một phần của ngữ cảnh
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — mẫu hình nút phản tỉnh
