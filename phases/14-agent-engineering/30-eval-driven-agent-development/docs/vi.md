# Phát triển Tác tử Hướng Đánh giá (Eval-Driven Agent Development)

> Chỉ dẫn cốt lõi từ Anthropic: "hãy bắt đầu với các lời nhắc (prompt) đơn giản, tối ưu hóa chúng thông qua quy trình đánh giá toàn diện, và chỉ bổ sung các hệ thống tác tử đa bước phức tạp khi thực sự cần thiết." Đánh giá (evaluation) không phải là bước cuối cùng. Nó là vòng lặp bên ngoài định hướng và dẫn dắt mọi lựa chọn kỹ thuật khác trong toàn bộ Phase 14.

**Loại bài học:** Tìm hiểu + Xây dựng (Learn + Build)  
**Ngôn ngữ:** Python (stdlib)  
**Yêu cầu tiên quyết:** Toàn bộ Phase 14.  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Nêu tên ba tầng đánh giá — bộ chuẩn chuẩn hóa tĩnh (static benchmarks), đánh giá ngoại tuyến tùy chỉnh (custom offline), đánh giá trực tuyến trên môi trường sản xuất (online production) — cùng vai trò và mục đích của từng tầng.
- Giải thích vòng lặp khép kín chặt chẽ giữa Người đánh giá và Bộ tối ưu hóa (evaluator-optimizer).
- Trình bày thực tiễn tốt nhất năm 2026: các bộ kiểm thử đánh giá (evals) đặt ngay cạnh mã nguồn, chạy tự động trong CI, đóng vai trò chốt chặn kiểm duyệt PR.
- Kết nối từng bài học trong Phase 14 với trường hợp đánh giá (eval case) cụ thể mà bài học đó tạo ra.

## Vấn đề Đặt ra

Các tác tử thường vượt qua các buổi trình diễn mẫu (demo) một cách trơn tru. Nhưng khi đưa vào môi trường sản xuất thực tế, chúng lại thất bại theo những cách mà các bản demo không tài nào dự báo trước. Các bộ điểm chuẩn chung (benchmarks) chỉ trả lời câu hỏi "liệu mô hình này có năng lực tổng quát tốt không?", chứ không trả lời được "liệu tác tử này có tạo ra các bản vá lỗi chính xác cho sản phẩm của tôi hay không?". Câu trả lời nằm ở: quy trình đánh giá ở cả ba tầng, vận hành liên tục, trong đó mọi hàng rào an toàn và quy tắc học được đều được ánh xạ trực tiếp thành một ca kiểm thử đánh giá cụ thể.

## Khái niệm

### Ba tầng đánh giá

1. **Bộ chuẩn tĩnh (Static benchmarks)** — SWE-bench Verified cho kỹ thuật phần mềm và mã nguồn (Bài 19), WebArena/OSWorld cho tác vụ duyệt web / hệ điều hành (Bài 20), GAIA cho năng lực tổng quát (Bài 19), BFCL V4 cho khả năng gọi công cụ (Bài 06). Dùng để so sánh chéo năng lực giữa các mô hình và làm chốt chặn phát hiện thoái lui chất lượng. Ô nhiễm dữ liệu là có thật: dự án SWE-bench+ phát hiện tới 32.67% tỷ lệ rò rỉ lời giải. Luôn công bố điểm số trên tập SWE-bench Verified hoặc trên các tập dữ liệu đã được kiểm toán rò rỉ độc lập như SWE-bench+.

2. **Đánh giá ngoại tuyến tùy chỉnh (Custom offline evals)** — bám sát hình thái sản phẩm của riêng bạn:
   - Dùng LLM làm giám khảo (LLM-as-judge qua Langfuse, Phoenix, Opik — Bài 24).
   - Đánh giá dựa trên thực thi (execution-based: chạy thử bản vá, kiểm tra các bài test).
   - Đánh giá dựa trên quỹ đạo hành động (trajectory-based: so sánh chuỗi hành động thực tế với quỹ đạo vàng chuẩn mẫu; OSWorld-Human chỉ ra các tác tử hàng đầu hiện tốn số bước gấp 1.4-2.7 lần so với chuyên gia con người).

3. **Đánh giá trực tuyến (Online evals)** — trên môi trường sản xuất thực:
   - Phát lại phiên tương tác (session replays trong Langfuse).
   - Cảnh báo kích hoạt từ hàng rào bảo vệ (Bài 16, 21).
   - Giám sát chi phí / độ trễ theo từng bước (Bài 23 qua các span của OpenTelemetry).

### Người đánh giá - Bộ tối ưu hóa (Evaluator-optimizer của Anthropic)

Vòng lặp khép kín:

1. Bên đề xuất (Proposer) tạo ra kết quả đầu ra.
2. Bên đánh giá (Evaluator) chấm điểm và phán quyết.
3. Tinh chỉnh cải tiến cho đến khi vượt qua được yêu cầu của bên đánh giá.

Đây chính là dạng tổng quát hóa của kỹ thuật Tự tinh chỉnh (Self-Refine - Bài 05). Bất kỳ luồng tác tử quan trọng nào bạn quan tâm đều có thể bọc trong vòng lặp evaluator-optimizer để gia tăng độ tin cậy.

### Thực tiễn tốt nhất năm 2026

- Bộ đánh giá đặt cùng thư mục với mã nguồn.
- Chạy tự động trong CI trên mỗi Pull Request.
- Đặt điều kiện chặn merge dựa trên điểm số đánh giá (ví dụ: "không được thoái lui quá 5% so với nhánh main").
- Mọi hàng rào bảo vệ (guardrail) đều phải ánh xạ thành một ca đánh giá kiểm thử.
- Mọi bài học kinh nghiệm thu được (Reflexion, quy tắc học từ luồng công việc) đều phải có một ca thất bại tương ứng để phòng ngừa tái diễn.

### Xâu chuỗi toàn bộ Phase 14

Hầu hết các bài học trong Phase 14 đều tạo ra những ca đánh giá cụ thể. Bảng dưới đây liệt kê các bài học có ca đánh giá rõ nét nhất:

| Bài học | Ca đánh giá mà bài học tạo ra |
|---|---|
| 01 Agent Loop | Hết ngân sách tài nguyên, hàng rào chống lặp vô hạn |
| 02 ReWOO | Bộ lập kế hoạch tái lập kế hoạch chính xác khi một công cụ bị lỗi |
| 03 Reflexion | Áp dụng các phản tư đã học được vào lượt thử lại |
| 05 Self-Refine/CRITIC | Giám khảo phê duyệt kết quả sau khi đã tinh chỉnh |
| 06 Tool Use | Ép kiểu đối số thành công; từ chối các công cụ không xác định |
| 07-10 Memory | Trích dẫn truy xuất khớp với nguồn; vô hiệu hóa thông tin lỗi thời |
| 12 Workflow Patterns | Mỗi mẫu luồng công việc tạo ra đầu ra chính xác |
| 13 LangGraph | Khôi phục tái tạo chính xác trạng thái tại thời điểm lưu |
| 14 AutoGen Actors | Hàng đợi thư chết (DLQ) bắt kịp các tiến trình xử lý bị sập |
| 16 OpenAI Agents SDK | Hàng rào bảo vệ kích hoạt đúng trên các đầu vào độc hại |
| 17 Claude Agent SDK | Kết quả từ các tác tử phụ gửi trả về trọn vẹn cho điều phối viên |
| 19-20 Benchmarks | Điểm SWE-bench Verified, tỷ lệ thành công WebArena, hiệu quả OSWorld |
| 21 Computer Use | Cơ chế an toàn theo bước chặn đứng DOM bị tiêm nhiễm |
| 23 OTel | Các span phát xuất đầy đủ các thuộc tính bắt buộc |
| 26 Failure Modes | Các bộ phát hiện gắn nhãn chính xác các dạng lỗi đã biết |
| 27 Prompt Injection | PVE kiên quyết từ chối các nội dung truy xuất bị đầu độc |
| 28 Orchestration | Người giám sát định tuyến chính xác tới chuyên gia phù hợp |
| 29 Runtime Shapes | Hàng đợi thư chết xử lý trơn tru tỷ lệ lỗi N% |

Nếu bộ kiểm thử đánh giá của bạn có đủ các trường hợp cho từng hàng trên, bạn đã làm chủ trọn vẹn phần lõi của Phase 14.

### Những cạm bẫy trong phát triển hướng đánh giá

- **Không có mốc chuẩn (baseline):** Đánh giá mà không có mốc trạng thái tốt gần nhất thì kết quả là vô nghĩa. Luôn phải lưu trữ các bản mốc chuẩn.
- **Giám khảo LLM thiếu neo thông tin (LLM-judge without grounding):** Chính mô hình giám khảo cũng có thể bị ảo giác. Áp dụng mẫu hình CRITIC (Bài 05) — giám khảo cần neo cơ sở vào các công cụ bên ngoài.
- **Hiện tượng khớp quá mức vào bộ kiểm thử (Over-fitting to evals):** Việc tối ưu hóa cục bộ để đạt điểm cao trên bộ đánh giá sẽ làm chệch hướng khỏi giá trị sử dụng thực tế. Luôn luân chuyển các bộ ca kiểm thử.
- **Bộ đánh giá chập chờn (Flaky evals):** Các ca kiểm thử không tất định sẽ tạo ra cảnh báo giả. Hãy cố định hạt giống ngẫu nhiên (seeds) và chụp nhanh trạng thái.

```figure
ae-eval-three-layers
```

## Tự Xây dựng (Build It)

`code/main.py` cài đặt một khung sườn đánh giá hoàn chỉnh bằng thư viện chuẩn (stdlib):

- Sổ đăng ký ca kiểm thử phân theo danh mục (bộ chuẩn tĩnh, tùy chỉnh, trực tuyến).
- Một tác tử giả lập theo kịch bản cần được kiểm thử.
- Vòng lặp người đánh giá - bộ tối ưu hóa: đề xuất, chấm điểm, tinh chỉnh cho đến khi đạt hoặc chạm số vòng tối đa.
- Cổng kiểm soát CI: tính toán tỷ lệ đạt tổng hợp + đo lường mức độ thoái lui so với mốc chuẩn.

Chạy thử mã nguồn:

```bash
python3 code/main.py
```

Kết quả: trạng thái đạt/không đạt theo từng ca kiểm thử, cờ cảnh báo thoái lui, phán quyết cuối cùng của cổng CI.

## Ứng dụng Thực tế (Use It)

- Viết các ca kiểm thử đánh giá nằm cùng trong kho lưu trữ mã nguồn tác tử của bạn.
- Tự động chạy chúng trên mỗi PR thông qua hệ thống CI.
- Chặn không cho hợp nhất mã nguồn nếu phát hiện thoái lui hiệu năng.
- Theo dõi biểu đồ tỷ lệ đạt chuẩn theo thời gian.
- Quy mọi sự cố phát sinh trên môi trường sản xuất thành một ca kiểm thử mới trong bộ đánh giá.

## Chuyển giao Đóng gói (Ship It)

`outputs/skill-eval-suite.md` giúp xây dựng một bộ đánh giá 3 tầng toàn diện cho sản phẩm tác tử, tích hợp sẵn các cổng kiểm soát CI và cơ chế theo dõi thoái lui.

## Bài tập Thực hành

1. Lấy một sự cố thực tế từng xảy ra trên hệ thống của bạn. Viết một ca đánh giá tái hiện chính xác lỗi đó. Tác tử hiện tại của bạn đã vượt qua được chưa?
2. Xây dựng một biểu mẫu chấm điểm (rubric) cho giám khảo LLM trong lĩnh vực nghiệp vụ của bạn theo ba tiêu chí (tính xác thực, văn phong, phạm vi). Chấm điểm thử nghiệm trên 50 phiên hội thoại.
3. Kết nối bộ đánh giá vào quy trình CI. Cấu hình đánh trượt bản dựng nếu phát hiện tỷ lệ thoái lui >=5%.
4. Bổ sung chỉ số hiệu quả quỹ đạo (trajectory efficiency): tác tử đã tốn bao nhiêu bước so với quỹ đạo chuẩn mực của chuyên gia con người?
5. Ánh xạ từng bài học trong Phase 14 thành một ca đánh giá trong bộ kiểm thử của riêng bạn. Còn bài nào bị thiếu không? Đó chính là lỗ hổng cần sớm lấp đầy.

## Thuật ngữ Then chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|---|---|---|
| Chuẩn đánh giá tĩnh (Static benchmark) | "Bộ đánh giá có sẵn" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Đánh giá ngoại tuyến tùy chỉnh (Custom offline eval) | "Đánh giá nghiệp vụ" | LLM làm giám khảo / chạy thực thi / so khớp quỹ đạo trên sản phẩm |
| Đánh giá trực tuyến (Online eval) | "Đánh giá môi trường thật" | Phát lại phiên, cảnh báo hàng rào an toàn, theo dõi chi phí/độ trễ |
| Người đánh giá - Bộ tối ưu hóa (Evaluator-optimizer) | "Đề xuất - Chấm - Sửa" | Lặp đi lặp lại cho đến khi bên đánh giá chấp thuận |
| Cổng kiểm soát CI (CI gate) | "Chặn hợp nhất mã" | Đánh trượt bản dựng khi có thoái lui chất lượng |
| Mốc chuẩn (Baseline) | "Trạng thái tốt nhất gần nhất" | Điểm số tham chiếu nền dùng để phát hiện thoái lui |
| Hiệu quả quỹ đạo (Trajectory efficiency) | "Số bước so với chuẩn vàng" | Số bước tác tử thực hiện chia cho số bước tối thiểu của chuyên gia con người |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — triết lý "bắt đầu đơn giản, tối ưu hóa bằng đánh giá"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — bộ chuẩn đánh giá mã nguồn được tinh chọn
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — bảng xếp hạng đánh giá năng lực gọi công cụ
- [Tài liệu Langfuse](https://langfuse.com/) — thực hành đánh giá và phát lại phiên tương tác
