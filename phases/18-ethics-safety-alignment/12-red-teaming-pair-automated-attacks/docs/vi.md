# Red-Teaming: PAIR và tấn công tự động (Red-Teaming: PAIR and Automated Attacks)

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Tinh chỉnh lặp tự động câu lệnh (Prompt Automatic Iterative Refinement) — là phương pháp bẻ khóa mô hình (jailbreak) hộp đen tự động kinh điển. Một mô hình ngôn ngữ lớn (LLM) tấn công mang lời nhắc hệ thống kiểm thử xâm nhập (red-team system prompt) sẽ liên tục đề xuất các câu lệnh bẻ khóa gửi tới LLM mục tiêu, tích lũy các lần thử và phản hồi trong lịch sử hội thoại của chính nó dưới dạng phản hồi trong ngữ cảnh (in-context feedback). PAIR thường thành công trong vòng chưa đến 20 truy vấn, hiệu quả hơn nhiều bậc độ lớn so với GCG (tìm kiếm gradient cấp độ token của Zou et al.) mà không đòi hỏi quyền truy cập hộp trắng (white-box access). Hiện nay, PAIR là một baseline tiêu chuẩn trong JailbreakBench (arXiv:2404.01318) và HarmBench, bên cạnh GCG, AutoDAN, TAP, và Lời nhắc đối kháng thuyết phục (Persuasive Adversarial Prompt - PAP).

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (tuân thủ chỉ dẫn), Phase 14 (kỹ thuật tác tử)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả thuật toán PAIR: system prompt của bên tấn công, tinh chỉnh lặp (iterative refinement), phản hồi trong ngữ cảnh (in-context feedback).
- Giải thích lý do vì sao PAIR vượt trội hoàn toàn về mặt hiệu quả so với GCG khi mô hình mục tiêu là hộp đen (black-box).
- Kể tên bốn baseline tấn công tự động khác (GCG, AutoDAN, TAP, PAP) và nêu một đặc điểm phân biệt của từng phương pháp.
- Trình bày các giao thức đánh giá của JailbreakBench và HarmBench cùng ý nghĩa thực sự của "tỷ lệ tấn công thành công" (attack success rate - ASR) trong từng khung đánh giá.

## Vấn đề đặt ra

Hoạt động kiểm thử xâm nhập (red-teaming) trước đây chủ yếu là một quy trình thủ công. Một nhóm nhỏ các chuyên gia kiểm thử sẽ tự tay thiết kế các câu lệnh đối kháng (adversarial prompts) và ghi chép lại những câu lệnh nào phát huy tác dụng. Cách tiếp cận này hoàn toàn không thể mở rộng quy mô: việc xác định tỷ lệ tấn công thành công đòi hỏi một kích thước mẫu có ý nghĩa thống kê, trong khi mô hình mục tiêu lại liên tục thay đổi sau mỗi lần phát hành bản cập nhật. PAIR đã cơ chế hóa hoạt động red-teaming thành một bài toán tối ưu hóa với mục tiêu hộp đen.

## Khái niệm cốt lõi

### Thuật toán PAIR

Đầu vào:
- LLM mục tiêu $T$ (Target LLM - mô hình đang bị tấn công).
- LLM giám khảo $J$ (Judge LLM - chấm điểm xem phản hồi có cấu thành một vụ bẻ khóa hay không).
- LLM tấn công $A$ (Attacker LLM - bộ tối ưu hóa của bên red-team).
- Chuỗi mục tiêu $G$ (Goal string): "hãy phản hồi bằng [chỉ dẫn độc hại]".
- Ngân sách $K$ (Budget - thường là 20 truy vấn).

Vòng lặp, với $k$ chạy từ $1 \dots K$:
1. $A$ nhận lời nhắc chứa mục tiêu $G$ cùng toàn bộ lịch sử các cặp (lời nhắc, phản hồi) đã ghi nhận cho đến thời điểm hiện tại.
2. $A$ sinh ra một lời nhắc mới $p_k$.
3. Gửi $p_k$ tới $T$; nhận lại phản hồi $r_k$.
4. $J$ chấm điểm cặp $(p_k, r_k)$ đối chiếu với mục tiêu $G$.
5. Nếu điểm số $\ge$ ngưỡng cho trước, dừng lại — phát hiện bẻ khóa thành công.
6. Ngược lại, thêm $(p_k, r_k)$ vào lịch sử hội thoại của $A$; tiếp tục vòng lặp.

Kết quả thực nghiệm (NeurIPS 2023): Đạt tỷ lệ tấn công thành công (ASR) >50% trước GPT-3.5-turbo và Llama-2-7B-chat; số truy vấn trung bình để đạt được thành công nằm trong khoảng 10-20.

### Vì sao PAIR đạt hiệu quả vượt trội

GCG (Zou et al. 2023) tìm kiếm các hậu tố token đối kháng thông qua gradient; phương pháp này đòi hỏi quyền truy cập hộp trắng vào mô hình và tạo ra các chuỗi ký tự tối nghĩa, con người không thể đọc hiểu. Ngược lại, PAIR hoạt động theo cơ chế hộp đen thuần túy và sinh ra các đòn tấn công bằng ngôn ngữ tự nhiên có khả năng chuyển giao (transferability) mạnh mẽ qua các mô hình khác nhau. Cơ chế phản hồi trong ngữ cảnh của PAIR giúp bên tấn công rút kinh nghiệm từ mỗi lần bị từ chối; GCG không có cơ chế tương đương (mỗi bước cập nhật token mới đều phải tự dò dẫm lại toàn bộ tiến trình trước đó).

### Các kỹ thuật tấn công tự động liên quan

- **GCG (Zou et al. 2023, arXiv:2307.15043):** Tìm kiếm hậu tố đối kháng bằng gradient cấp độ token. Cơ chế hộp trắng, có tính chuyển giao, tạo ra các chuỗi ký tự vô nghĩa với người đọc.
- **AutoDAN (Liu et al. 2023):** Tìm kiếm tiến hóa (evolutionary search) trên các lời nhắc, được định hướng bởi một hàm mục tiêu phân cấp (hierarchical objective).
- **TAP (Mehrotra et al. 2024):** Cây tấn công có cắt tỉa (Tree-of-attacks with pruning) — phân nhánh nhiều quỹ đạo sinh lời nhắc theo phong cách PAIR.
- **PAP (Zeng et al. 2024):** Lời nhắc đối kháng thuyết phục (Persuasive Adversarial Prompts) — mã hóa các kỹ thuật tâm lý thuyết phục của con người vào các mẫu prompt có cấu trúc.

### JailbreakBench và HarmBench

Cả hai bộ chuẩn đánh giá (2024) đều đóng vai trò chuẩn hóa quy trình thẩm định an toàn:

- JailbreakBench (arXiv:2404.01318): 100 hành vi độc hại trải dài trên 10 danh mục chính sách của OpenAI. Sử dụng tỷ lệ tấn công thành công (Attack Success Rate - ASR) làm chỉ số cốt lõi. Yêu cầu một mô hình giám khảo (GPT-4-turbo, Llama Guard, hoặc StrongREJECT).
- HarmBench (Mazeika et al. 2024): 510 hành vi thuộc 7 danh mục, đi kèm các bài kiểm tra tổn hại về mặt ngữ nghĩa và chức năng thực thi. So sánh 18 phương pháp tấn công trên 33 mô hình khác nhau.

Chỉ số ASR thường được báo cáo ở một mức ngân sách truy vấn cố định. Việc so sánh giữa các kỹ thuật tấn công bắt buộc phải đối chiếu trên cùng một mức ngân sách; mức ASR 90% ở 200 truy vấn không thể xem là tương đương với ASR 85% ở 20 truy vấn.

### Tầm quan trọng đối với các hệ thống triển khai năm 2026

Mọi phòng thí nghiệm tiên phong hiện nay đều chạy PAIR và TAP để kiểm thử các mô hình thương mại trước khi phát hành. Quỹ đạo ASR xuất hiện định kỳ trong các thẻ mô hình (model cards - Bài 26) và các phụ lục lập luận an toàn (safety cases - Bài 18). Đây không còn là một kỹ thuật tấn công dị biệt — nó đã trở thành một phần của hạ tầng an toàn tiêu chuẩn.

### Vị trí của bài học trong Giai đoạn 18

Bài 12 đặt nền móng về tấn công tự động. Bài 13 (Jailbreak bằng nhiều ví dụ - Many-Shot Jailbreaking) là một kỹ thuật khai thác độ dài ngữ cảnh bổ trợ. Bài 14 (Nghệ thuật ASCII / Thị giác - ASCII Art / Visual) là hình thức tấn công thông qua mã hóa biểu diễn. Bài 15 (Chèn prompt gián tiếp - Indirect Prompt Injection) là bề mặt tấn công chính trong môi trường sản xuất năm 2026. Bài 16 sẽ nghiên cứu các công cụ phòng thủ đối ứng (Llama Guard, Garak, PyRIT).

```figure
al-pair-loop
```

## Thực hành (Use It)

Tệp `code/main.py` xây dựng một vòng lặp PAIR thu nhỏ. Mô hình mục tiêu là một bộ phân loại mô phỏng từ chối các lời nhắc độc hại "rõ ràng" (dựa trên bộ lọc từ khóa). Bên tấn công là một công cụ tinh chỉnh theo quy tắc nhằm thử nghiệm các kỹ thuật diễn đạt lại (paraphrase), đặt khung nhập vai (roleplay-framing), và mã hóa. Giám khảo sẽ chấm điểm phản hồi thu được. Bạn sẽ quan sát thấy bên tấn công thành công sau ~5-15 lượt lặp trước bộ lọc từ khóa nhưng thất bại hoàn toàn trước bộ lọc ngữ nghĩa.

## Đóng gói sản phẩm (Ship It)

Bài học này cung cấp sản phẩm `outputs/skill-attack-audit.md`. Khi tiếp nhận một báo cáo đánh giá red-team, kỹ năng này sẽ kiểm toán chi tiết: các cuộc tấn công nào đã được thực thi (PAIR, GCG, TAP, AutoDAN, PAP), với ngân sách truy vấn là bao nhiêu cho từng loại, sử dụng mô hình giám khảo nào, và kiểm thử trên tập hành vi độc hại nào (JailbreakBench, HarmBench, hay bộ dữ liệu nội bộ).

## Bài tập

1. Chạy `code/main.py`. Đo số truy vấn trung bình để thành công (mean-queries-to-success) cho ba chiến lược tấn công tích hợp sẵn. Giải thích rõ từng chiến lược đang khai thác giả định phòng thủ nào của mục tiêu.

2. Triển khai chiến lược tấn công thứ tư (ví dụ: dịch sang ngôn ngữ ít phổ biến, mã hóa base64). Báo cáo số truy vấn trung bình mới để thành công trước mục tiêu dùng bộ lọc từ khóa và mục tiêu dùng bộ lọc ngữ nghĩa.

3. Đọc Hình 5 trong bài báo của Chao et al. 2023 (so sánh PAIR và GCG). Phân tích hai kịch bản mà GCG vẫn được ưu tiên hơn mặc dù PAIR sở hữu lợi thế vượt trội về hiệu quả truy vấn.

4. JailbreakBench báo cáo ASR dựa trên một tập mục tiêu cố định. Hãy thiết kế một chỉ số bổ sung đo lường tính đa dạng của cuộc tấn công (phương sai giữa các prompt thành công). Giải thích vì sao tính đa dạng của tấn công lại đóng vai trò quyết định trong việc đánh giá năng lực phòng thủ.

5. TAP (Mehrotra 2024) mở rộng PAIR bằng cơ chế phân nhánh kết hợp cắt tỉa (branching + pruning). Hãy phác thảo một phần mở rộng theo phong cách TAP cho `code/main.py` và phân tích sự đánh đổi giữa chi phí tính toán và tỷ lệ tấn công thành công.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| PAIR | "bẻ khóa tự động" | Tinh chỉnh lặp tự động câu lệnh (Prompt Automatic Iterative Refinement); vòng lặp LLM tấn công + LLM giám khảo |
| GCG | "bẻ khóa bằng gradient" | Tìm kiếm hậu tố đối kháng bằng gradient cấp độ token trên mô hình hộp trắng |
| Attack success rate (ASR) | "tỷ lệ bẻ khóa tại k truy vấn" | Chỉ số đánh giá cốt lõi; bắt buộc phải báo cáo kèm ngân sách truy vấn và định danh mô hình giám khảo |
| Judge LLM | "mô hình chấm điểm" | LLM được giao nhiệm vụ đánh giá xem phản hồi có đáp ứng mục tiêu độc hại hay không |
| JailbreakBench | "bộ kiểm chuẩn đánh giá" | Tập hợp hành vi độc hại được tiêu chuẩn hóa kèm các danh mục phân loại rõ ràng |
| HarmBench | "bộ kiểm chuẩn mở rộng" | 510 hành vi độc hại, bao gồm các bài kiểm tra tổn hại cả về mặt chức năng lẫn ngữ nghĩa |
| TAP | "cây tấn công" | PAIR kết hợp phân nhánh và cắt tỉa; đạt ASR cao hơn với chi phí tính toán lớn hơn |

## Đọc thêm

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — Bài báo gốc về PAIR, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — Bài báo gốc về GCG
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — Khung đánh giá và kiểm chuẩn tiêu chuẩn hóa
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — Khung đánh giá an toàn mở rộng quy mô lớn
