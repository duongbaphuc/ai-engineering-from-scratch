# Điểm chuẩn đánh giá: SWE-bench, GAIA, AgentBench

> Ba bộ điểm chuẩn đánh giá (benchmarks) đóng vai trò nền tảng trong việc đánh giá tác tử (agent evaluation) vào năm 2026. SWE-bench kiểm thử năng lực vá lỗi mã nguồn (code patching). GAIA kiểm thử năng lực sử dụng công cụ tổng quát (generalist tool use). AgentBench kiểm thử khả năng suy luận đa môi trường (multi-environment reasoning). Nắm vững thành phần cấu tạo, bài toán rò rỉ/nhiễm bẩn dữ liệu (data contamination), và những khía cạnh mà các bộ chuẩn này không thể đo lường.

**Type:** Learn  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 06 (Tool Use)  
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên khung kiểm thử (test harness) của SWE-bench (FAIL_TO_PASS) và giải thích lý do bộ kiểm chuẩn này sử dụng kiểm thử đơn vị (unit tests) làm cổng chặn đánh giá.
- Giải thích vì sao SWE-bench Verified (OpenAI, 500 tác vụ) ra đời và tập dữ liệu này đã loại bỏ những gì.
- Mô tả triết lý thiết kế của GAIA: đơn giản với con người, thách thức với AI; ba cấp độ khó.
- Liệt kê tám môi trường thử nghiệm của AgentBench và rào cản cốt lõi khiến các mô hình ngôn ngữ lớn mã nguồn mở (open-source LLMs) chưa bắt kịp các mô hình thương mại.
- Tóm tắt phát hiện về sự nhiễm bẩn dữ liệu của SWE-bench+ và các hệ quả thực tiễn rút ra.

## Vấn đề đặt ra

Các bảng xếp hạng (leaderboards) chỉ cho bạn biết mô hình nào giành điểm số cao nhất trên một bộ điểm chuẩn duy nhất. Chúng không tiết lộ:

- Liệu bộ điểm chuẩn đó có bị nhiễm bẩn dữ liệu hay không (lời giải nằm sẵn trong dữ liệu tiền huấn luyện, rò rỉ dữ liệu kiểm thử - test leakage).
- Liệu bộ điểm chuẩn có thực sự đo lường khía cạnh bạn quan tâm hay không (mã nguồn, duyệt web hay giải quyết tác vụ tổng quát).
- Liệu bộ đánh giá (evaluator) có thực sự vững chắc hay không (so khớp cây cú pháp trừu tượng - AST matching, kiểm tra trạng thái hệ thống, hay rà soát thủ công bởi chuyên gia).

Cần hiểu rõ ba bộ điểm chuẩn trụ cột này cùng các chế độ lỗi (failure modes) của chúng trước khi trích dẫn bất kỳ con số thống kê nào.

## Khái niệm cốt lõi

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- Gồm 2.294 vấn đề thực tế (issues) thu thập từ GitHub thuộc 12 kho mã nguồn Python phổ biến.
- Đầu vào cho tác tử: toàn bộ mã nguồn tại commit trước khi sửa lỗi (pre-fix commit) kèm theo mô tả vấn đề bằng ngôn ngữ tự nhiên.
- Đầu ra của tác tử: một bản vá mã nguồn (patch).
- Bộ đánh giá: áp dụng bản vá, chạy toàn bộ bộ kiểm thử của kho mã. Bản vá bắt buộc phải chuyển trạng thái các kiểm thử `FAIL_TO_PASS` (trước đó thất bại, nay phải vượt qua) mà không làm hỏng các kiểm thử `PASS_TO_PASS` (vốn đã vượt qua, nay tiếp tục vượt qua).

SWE-agent (Yang et al., 2024) từng đạt tỷ lệ giải quyết 12,5% tại thời điểm ra mắt nhờ chú trọng thiết kế giao diện tương tác giữa tác tử và máy tính (agent-computer interfaces - các lệnh chỉnh sửa tệp tin, cú pháp tìm kiếm mà mô hình có thể thấu hiểu).

### SWE-bench Verified

Do OpenAI công bố vào tháng 8 năm 2024. Đây là tập con gồm 500 tác vụ được con người thẩm định kỹ lưỡng. Tập dữ liệu này loại bỏ các issue mơ hồ, các bài kiểm thử thiếu ổn định (flaky tests), và những tác vụ mà phương án sửa lỗi ban đầu không rõ ràng. Đây là bộ điểm chuẩn tiêu chuẩn cho câu hỏi "Liệu tác tử của bạn có thực sự xuất xưởng được các bản vá lỗi thực tế hay không?".

### Nhiễm bẩn dữ liệu (Contamination)

- Hơn 94% các issue trong SWE-bench xuất hiện trước thời điểm chốt dữ liệu huấn luyện (cutoff date) của phần lớn các mô hình hiện nay.
- Nhóm nghiên cứu **SWE-bench+** phát hiện ra rằng 32,67% các bản vá thành công thực chất đã bị rò rỉ lời giải trực tiếp ngay trong văn bản mô tả issue (mô hình nhìn thấy cách sửa lỗi ngay trong phần mô tả), và 31,08% trường hợp bị nghi ngờ do độ bao phủ kiểm thử quá yếu.
- SWE-bench Verified sạch hơn đáng kể nhưng vẫn chưa loại bỏ hoàn toàn hiện tượng nhiễm bẩn.

Hệ quả thực tiễn: Một mô hình đạt 50% trên SWE-bench gốc có thể chỉ đạt 35% trên SWE-bench+. Luôn báo cáo đồng thời cả hai chỉ số nếu bạn công bố hiệu năng SWE-bench.

### GAIA (Mialon et al., Tháng 11/2023)

- Gồm 466 câu hỏi; 300 câu được giữ lại làm tập kiểm thử ẩn cho bảng xếp hạng riêng tư tại `huggingface.co/gaia-benchmark`.
- Triết lý thiết kế: "về mặt nhận thức thì đơn giản đối với con người (tỷ lệ giải quyết 92%) nhưng cực kỳ thách thức đối với AI (GPT-4 kèm plugin chỉ đạt 15% tại thời điểm ra mắt)."
- Đo lường năng lực suy luận phức tạp, xử lý đa phương thức (multi-modality), duyệt web, và sử dụng công cụ (tool use).
- Ba cấp độ khó; trong đó Cấp độ 3 đòi hỏi các chuỗi gọi công cụ dài qua nhiều phương thức dữ liệu khác nhau.

GAIA là bài kiểm tra tiêu chuẩn để đo lường "năng lực tác tử tổng quát (generalist capability)". Đừng nhầm lẫn nó với các bộ điểm chuẩn chuyên biệt về lập trình.

### AgentBench (Liu et al., ICLR 2024)

- Bao gồm 8 môi trường trải rộng từ lập trình (Bash, DB, KG), trò chơi tương tác (Alfworld, LTP), web (WebShop, Mind2Web), cho đến sinh văn bản mở (open-ended generation).
- Tác vụ tương tác đa lượt (multi-turn), trung bình ~4k đến 13k lượt trao đổi cho mỗi phân vùng dữ liệu (split).
- Phát hiện trọng tâm: Khả năng suy luận dài hạn (long-term reasoning), ra quyết định (decision-making), và tuân thủ chỉ dẫn phức tạp là những rào cản chính ngăn các LLM mã nguồn mở bắt kịp các mô hình thương mại đóng.

### Những gì các bộ điểm chuẩn này chưa đo lường được

- Chi phí vận hành trong môi trường thực tế (số lượng token tiêu thụ, thời gian phản hồi thực tế - wall-clock time).
- Hành vi an toàn dưới các điều kiện đối kháng (adversarial attacks / prompt injection).
- Hiệu năng trên miền nghiệp vụ cụ thể của bạn (hãy xây dựng bộ đánh giá tùy chỉnh riêng, Bài 30).
- Các lỗi phân phối đuôi (tail failures - các bộ điểm chuẩn chỉ tính trung bình; nhưng kỹ sư vận hành sản xuất lại quan tâm sâu sắc tới 1% sự cố tồi tệ nhất).

### Điểm chuẩn sai lệch trong thực tiễn

- **Chăm chăm vào một con số đơn lẻ (Single-number fixation):** Điểm số SWE-bench 50% mang lại ít giá trị hơn biểu đồ phân phối chi phí cùng số bước thực thi ở các phân vị P50/P75/P95.
- **Tuyên bố hiệu năng bị nhiễm bẩn:** Trích dẫn kết quả SWE-bench mà không đề cập đến Verified hoặc SWE-bench+ là gây hiểu lầm.
- **Lấy bộ điểm chuẩn làm mục tiêu tối ưu hóa sản phẩm:** Tối ưu hóa quá mức cho điểm chuẩn sẽ khiến sản phẩm rời xa tính hữu dụng thực tế trong sản xuất.

```figure
ae-swebench-gate
```

## Thực hành Xây dựng (Build It)

Tệp `code/main.py` hiện thực một khung kiểm thử mô phỏng theo SWE-bench dạng thu nhỏ:

- Các tác vụ sửa lỗi tổng hợp (3 tác vụ).
- Một "tác tử" hoạt động theo kịch bản đề xuất các bản vá mã nguồn.
- Một bộ chạy kiểm thử kiểm tra điều kiện `FAIL_TO_PASS` (lỗi nay đã được khắc phục) và `PASS_TO_PASS` (không phát sinh lỗi hồi quy).
- Một bộ phân loại độ khó theo phong cách GAIA dựa trên độ sâu phân rã câu hỏi.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả đầu ra sẽ hiển thị tỷ lệ giải quyết (resolution rate) trên từng tác vụ cùng từng cấp độ khó, cụ thể hóa các quy tắc của bộ đánh giá.

## Ứng dụng Thực tế (Use It)

- **SWE-bench Verified:** Lựa chọn hàng đầu cho các tác tử viết mã. Luôn báo cáo điểm số Verified.
- **GAIA:** Dành cho các tác tử tổng quát giải quyết bài toán đa năng. Hãy sử dụng phân vùng dữ liệu riêng tư (private split).
- **AgentBench:** So sánh năng lực điều hành đa môi trường.
- **Đánh giá tùy chỉnh (Custom evals):** (Xem Bài 30) Phản ánh chính xác diện mạo và đặc thù sản phẩm của bạn.

## Xuất xưởng Sản phẩm (Ship It)

Tệp `outputs/skill-benchmark-harness.md` cung cấp bộ khung kỹ năng giúp xây dựng khung kiểm thử theo chuẩn SWE-bench cho bất kỳ cặp kho mã nguồn - tác vụ nào, tích hợp sẵn cơ chế kiểm soát cổng `FAIL_TO_PASS` và `PASS_TO_PASS`.

## Bài tập

1. Chuyển đổi khung kiểm thử mô phỏng sang chạy trên một kho mã nguồn thực tế (chọn một dự án của bạn). Viết 3 bài kiểm tra `FAIL_TO_PASS` cho các lỗi đã biết.
2. Bổ sung chỉ số đếm số bước (step-count metric). Trên 3 tác vụ của bạn, tác tử cần trung bình bao nhiêu bước để giải quyết mỗi lỗi?
3. Đọc bài báo nghiên cứu SWE-bench+. Triển khai thuật toán kiểm tra rò rỉ lời giải (so khớp mẫu giữa phần mô tả issue với nội dung thay đổi trong git diff).
4. Tải một câu hỏi GAIA từ tập dữ liệu công khai. Lần vết từng bước mà một tác tử lớp GPT-4 sẽ thực hiện. Tác tử cần những công cụ gì?
5. Đọc bảng phân tích chi tiết từng môi trường trong bài báo AgentBench. Môi trường nào phản ánh sát nhất bề mặt sản phẩm của bạn? Trạng thái công nghệ hàng đầu (SOTA) tại môi trường đó hiện ra sao?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| SWE-bench | "Điểm chuẩn tác tử lập trình" | 2.294 issue trên GitHub; bản vá phải đảo ngược trạng thái bài kiểm thử `FAIL_TO_PASS` |
| SWE-bench Verified | "SWE-bench sạch" | 500 tác vụ được con người thẩm định kỹ lưỡng, do OpenAI công bố |
| FAIL_TO_PASS | "Cổng sửa lỗi" | Các bài kiểm thử trước đó thất bại và bắt buộc phải vượt qua sau khi áp bản vá |
| PASS_TO_PASS | "Cổng chống hồi quy" | Các bài kiểm thử vốn đã vượt qua và bắt buộc phải tiếp tục vượt qua |
| GAIA | "Điểm chuẩn đa năng" | 466 câu hỏi đa công cụ: dễ với con người, thách thức với AI |
| AgentBench | "Điểm chuẩn đa môi trường" | 8 môi trường tương tác; chân trời tương tác dài hạn qua nhiều lượt |
| Nhiễm bẩn (Contamination) | "Rò rỉ tập huấn luyện" | Các tác vụ trong bộ điểm chuẩn đã xuất hiện trong dữ liệu huấn luyện mô hình |
| SWE-bench+ | "Kiểm toán nhiễm bẩn dữ liệu" | Phát hiện 32,67% rò rỉ lời giải trong các bản vá SWE-bench thành công |

## Đọc thêm

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — bài báo gốc giới thiệu bộ điểm chuẩn
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — tập con dữ liệu đã qua thẩm định
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — bộ điểm chuẩn tác tử tổng quát
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — bộ kiểm thử đa môi trường
