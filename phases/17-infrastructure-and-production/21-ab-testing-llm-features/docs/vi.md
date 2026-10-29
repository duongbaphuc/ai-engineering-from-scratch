# Thử nghiệm A/B cho các tính năng LLM — GrowthBook, Statsig, và Vấn nạn Cảm tính (Vibes Problem)

> Phương pháp thử nghiệm A/B truyền thống không được thiết kế cho các mô hình LLM phi tất định (non-deterministic). Sự phân biệt cốt tử ở đây là: các bộ đánh giá ngoại tuyến (evals) trả lời câu hỏi "mô hình có làm được việc không?", trong khi thử nghiệm A/B (A/B testing) trả lời câu hỏi "người dùng có thực sự quan tâm hay không?". Cả hai đều bắt buộc; thời kỳ phát hành sản phẩm dựa trên cảm tính ("vibe checks") đã chính thức chấm dứt. Những gì cần thử nghiệm trong năm 2026: kỹ nghệ prompt (câu chữ), lựa chọn mô hình (GPT-4 so với GPT-3.5 so với mã nguồn mở OSS; cân đối giữa độ chính xác, chi phí và độ trễ), các tham số sinh văn bản (temperature, top-p). Những trường hợp thực tế điển hình: một biến thể mô hình thưởng (reward model) của chatbot giúp tăng 70% độ dài cuộc trò chuyện và tăng 30% tỷ lệ giữ chân người dùng; các thử nghiệm dòng tiêu đề AI của Nextdoor mang lại mức tăng 1% tỷ lệ nhấp (CTR) sau khi tinh chỉnh hàm phần thưởng; Khan Academy liên tục tinh chỉnh Khanmigo trên trục đánh đổi giữa độ trễ và độ chính xác toán học. Bức tranh nền tảng thử nghiệm: **Statsig** (được OpenAI mua lại với giá 1,1 tỷ USD vào tháng 9 năm 2025) — hỗ trợ thử nghiệm tuần tự (sequential testing), giảm phương sai CUPED, giải pháp tất-cả-trong-một (all-in-one). **GrowthBook** — mã nguồn mở, vận hành trực tiếp trên kho dữ liệu (warehouse-native), hỗ trợ các công cụ phân tích Bayesian + Frequentist + Sequential, CUPED, kiểm tra lệch tỷ lệ mẫu (SRM), và các hiệu chỉnh Benjamini-Hochberg + Bonferroni. Bạn lựa chọn dựa trên sở thích xử lý dữ liệu bằng SQL trong kho dữ liệu và việc tổ chức của bạn có e ngại yếu tố "thuộc sở hữu của OpenAI" hay không.

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## Learning Objectives

- Phân biệt giữa đánh giá ngoại tuyến (evals - "mô hình có làm được việc") và thử nghiệm A/B ("người dùng có quan tâm hay không").
- Liệt kê ba trục có thể thử nghiệm (prompt, mô hình, tham số sinh) và lựa chọn chỉ số đo lường thích hợp cho từng trục.
- Giải thích các kỹ thuật CUPED, thử nghiệm tuần tự (sequential testing), và các phép hiệu chỉnh đa so sánh Benjamini-Hochberg.
- Lựa chọn giữa Statsig hoặc GrowthBook dựa trên tư thế lưu trữ kho dữ liệu SQL và quan điểm về các thương vụ thâu tóm doanh nghiệp.

## The Problem

Bạn vừa tự tay tinh chỉnh một prompt hệ thống (system prompt). Trực giác mách bảo rằng câu trả lời đã hay hơn hẳn. Bạn lập tức phát hành ra môi trường production. Tỷ lệ chuyển đổi (conversion rate) chỉ biến động ngẫu nhiên trong phạm vi nhiễu. Bạn đổ lỗi cho chỉ số đo lường. Hoặc bạn phát hành một mô hình hoàn toàn mới nhưng tỷ lệ chuyển đổi không hề dịch chuyển — mô hình đã bị giảm chất lượng hay thay đổi đó quá nhỏ để người dùng nhận ra? Bạn hoàn toàn không biết câu trả lời, bởi bạn đã phát hành mà không có một bài thử nghiệm A/B có đối chứng.

Các bài đánh giá ngoại tuyến (evals) trả lời liệu mô hình có thể giải quyết một tác vụ trên một tập dữ liệu gán nhãn cố định hay không. Chúng không thể khẳng định người dùng có thực sự ưa chuộng kết quả đó trong trải nghiệm thực tế. Chỉ có một thử nghiệm trực tuyến có kiểm soát mới trả lời được điều đó, và chỉ với điều kiện thử nghiệm có đủ độ mạnh thống kê (statistical power), kiểm soát được tính phi tất định, và hiệu chỉnh đầy đủ cho các bài toán so sánh nhiều lần (multiple comparisons).

## The Concept

### Đánh giá ngoại tuyến (Evals) so với Thử nghiệm A/B (A/B tests)

**Đánh giá ngoại tuyến (Evals)** — Chạy ngoại tuyến, trên tập dữ liệu gán nhãn, sử dụng giám khảo (tiêu chí chấm rubric, LLM-as-a-judge, hoặc con người). Trả lời: "Đầu ra có chính xác / hữu ích / an toàn trên phân phối dữ liệu cố định này không?"

**Thử nghiệm A/B (A/B test)** — Chạy trực tuyến, trên người dùng thực, phân bổ ngẫu nhiên. Trả lời: "Biến thể mới có tạo ra sự thay đổi tích cực trên chỉ số thực tế ở cấp độ người dùng không?"

Cả hai đều không thể thiếu. Evals giúp phát hiện sớm các hiện tượng thoái lui chất lượng trước khi tiếp xúc với người dùng; thử nghiệm A/B xác nhận tác động thực tế tới sản phẩm sau đó.

### Những yếu tố cần thử nghiệm

1. **Kỹ nghệ Prompt (Prompt engineering)** — Thay đổi câu từ, cấu trúc prompt hệ thống, số lượng ví dụ mẫu (few-shot). Chỉ số đo lường: tỷ lệ hoàn thành tác vụ, tỷ lệ giữ chân người dùng, chi phí/yêu cầu.
2. **Lựa chọn mô hình (Model selection)** — Đặt GPT-4 đối đầu GPT-3.5-Turbo hoặc mô hình mã nguồn mở Llama. Chỉ số đo lường: độ chính xác tác vụ + chi phí/yêu cầu + độ trễ P99. Đây là bài toán tối ưu đa mục tiêu (multi-objective).
3. **Tham số sinh văn bản (Generation parameters)** — Nhiệt độ (temperature), top-p, số token tối đa (max_tokens). Chỉ số đo lường: tùy thuộc tác vụ cụ thể (độ đa dạng đầu ra so với độ tất định).

### CUPED — Kỹ thuật giảm phương sai (Controlled-experiments Using Pre-Experiment Data)

Kỹ thuật sử dụng dữ liệu trước thử nghiệm để kiểm soát phương sai. Bằng cách hồi quy (regress out) phần phương sai từ giai đoạn trước thử nghiệm trước khi so sánh giai đoạn sau, CUPED giúp giảm phương sai thông thường từ 30-70%. Kích thước mẫu hiệu dụng (effective sample size) nhờ đó tăng lên đáng kể mà không tốn thêm lưu lượng thực nghiệm.

Triển khai: Cả Statsig và GrowthBook đều tích hợp sẵn CUPED.

### Thử nghiệm tuần tự (Sequential testing)

Thử nghiệm A/B cổ điển luôn giả định kích thước mẫu cố định ngay từ đầu. Thử nghiệm tuần tự cho phép "nhìn trộm và ra quyết định sớm" (peek-and-decide) trong khi vẫn kiểm soát chặt chẽ tỷ lệ dương tính giả (false-positive rate) qua nhiều lần quan sát. Các quy trình tuần tự luôn hợp lệ (như mSPRT, chuỗi tin cậy Howard's confidence sequences) cho phép bạn dừng thử nghiệm sớm ngay khi biến thể chiến thắng đã rõ ràng.

### Hiệu chỉnh so sánh nhiều lần (Multiple-comparison corrections)

Khi chạy đồng thời 20 thử nghiệm A/B ở mức độ tin cậy 95%, theo quy luật ngẫu nhiên chắc chắn sẽ xuất hiện ít nhất một kết quả dương tính giả. Phép hiệu chỉnh Bonferroni siết chặt ngưỡng ý nghĩa thống kê α trên từng bài kiểm tra; trong khi thuật toán Benjamini-Hochberg kiểm soát tỷ lệ phát hiện sai (False Discovery Rate - FDR). GrowthBook cung cấp cả hai phương pháp này.

### SRM — Lệch tỷ lệ mẫu (Sample Ratio Mismatch)

Hàm băm phân bổ (assignment hash) có nhiệm vụ chia ngẫu nhiên người dùng vào các biến thể. Nếu tỷ lệ cấu hình là 50/50 nhưng thực tế thu về là 47/53, hệ thống đang gặp lỗi nghiêm trọng (lỗi bộ nhớ đệm, lỗi định tuyến hoặc người dùng bị rớt) — kiểm tra SRM sẽ lập tức giương cờ cảnh báo. Cả hai nền tảng đều hỗ trợ tính năng kiểm tra này.

### So sánh Statsig và GrowthBook

**Statsig**:
- Được OpenAI mua lại với giá 1,1 tỷ USD (tháng 9 năm 2025). Nền tảng SaaS do nhà cung cấp lưu trữ.
- Hỗ trợ thử nghiệm tuần tự, CUPED, nhóm đối chứng giữ lại (held-out populations).
- Giải pháp trọn gói: tích hợp cờ tính năng (feature flags) + thử nghiệm thực nghiệm + khả năng quan sát.
- Phù hợp nhất: Đội ngũ muốn một sản phẩm gói gọn, không vướng bận việc thuộc sở hữu của OpenAI.

**GrowthBook**:
- Mã nguồn mở (MIT); kiến trúc hướng kho dữ liệu (warehouse-native, đọc dữ liệu trực tiếp từ Snowflake/BigQuery/Redshift).
- Đa dạng động cơ phân tích: Bayesian, Frequentist, Sequential.
- Hỗ trợ CUPED, SRM, hiệu chỉnh Bonferroni và Benjamini-Hochberg.
- Tự lưu trữ (self-host) hoặc sử dụng đám mây quản lý.
- Phù hợp nhất: Doanh nghiệp vận hành kho dữ liệu SQL, đội ngũ dữ liệu nắm quyền kiểm soát tầng chỉ số, ưu tiên mã nguồn mở.

### Tính phi tất định làm phức tạp việc tính độ mạnh thống kê

Cùng một prompt có thể tạo ra các kết quả khác nhau qua mỗi lần gọi. Các phép tính toán lực thống kê (power calculations) truyền thống luôn giả định các quan sát là độc lập và cùng phân phối (IID). Do tính phi tất định của LLM, kích thước mẫu hiệu dụng thực tế sẽ thấp hơn kích thước mẫu danh định. Bạn cần nhân kích thước mẫu yêu cầu lên khoảng 1.3 - 1.5 lần như một biên độ an toàn.

### Kết quả từ các tình huống thực tế

- Biến thể mô hình thưởng cho chatbot: tăng +70% độ dài hội thoại, +30% tỷ lệ giữ chân người dùng.
- Tối ưu tiêu đề của Nextdoor: tăng +1% CTR sau khi tinh chỉnh hàm phần thưởng.
- Trợ lý Khanmigo của Khan Academy: lặp lại nhiều vòng đánh đổi giữa độ trễ và độ chính xác toán học.

### Phản hình mẫu: Phát hành dựa trên cảm tính (Shipping on vibes)

Bất kỳ kỹ sư kỳ cựu nào cũng từng chứng kiến một tính năng được phát hành chỉ vì "cảm giác câu trả lời có vẻ hay hơn" mà không hề có thử nghiệm A/B. Đa phần các tính năng đó đã âm thầm làm suy thoái các chỉ số sản phẩm mà đội ngũ không hề hay biết trong nhiều tháng liền. Thử nghiệm A/B chính là công cụ kỷ luật bắt buộc.

### Những con số cần ghi nhớ

- Statsig được OpenAI mua lại: 1,1 tỷ USD, tháng 9 năm 2025.
- GrowthBook: mã nguồn mở cấp phép MIT; hỗ trợ động cơ Bayesian + Frequentist + Sequential.
- Mức giảm phương sai của CUPED: từ 30-70%.
- Tính phi tất định của LLM → cần dự phòng tăng thêm +30-50% kích thước mẫu.

```figure
mx-sequential-test
```

## Use It

Tệp `code/main.py` mô phỏng một bài thử nghiệm A/B tuần tự với cả ranh giới cố định (fixed) và ranh giới tuần tự (sequential). Mô phỏng chứng minh cơ chế tuần tự cho phép dừng thử nghiệm sớm một cách an toàn như thế nào.

## Ship It

Bài học này xây dựng tệp `outputs/skill-ab-plan.md`. Dựa trên nội dung thay đổi tính năng, khối lượng công việc, và đường cơ sở đo lường, kỹ năng này sẽ lựa chọn nền tảng, thiết lập các cổng kiểm soát, và tính toán kích thước mẫu phù hợp.

## Exercises

1. Chạy `code/main.py`. Với mức tăng trưởng kỳ vọng 5% (lift) trên tỷ lệ chuyển đổi đường cơ sở 3%, kích thước mẫu cần thiết là bao nhiêu để đạt 80% độ mạnh thống kê (power)?
2. Đưa ra lựa chọn giữa Statsig hoặc GrowthBook cho một khách hàng ngành y tế yêu cầu nghiêm ngặt về triển khai tại chỗ (on-premises).
3. Thiết kế một thử nghiệm A/B so sánh GPT-4 và GPT-3.5 dựa trên chỉ số chi phí trên mỗi yêu cầu xử lý thành công (cost-per-resolved-ticket). Xác định chỉ số chính (primary metric), chỉ số bảo vệ (guardrail metric), và chỉ số thứ cấp (secondary metric).
4. Bản phát hành chim hoàng yến (canary) của bạn vượt qua các cổng kiểm tra an toàn, nhưng thử nghiệm A/B cho thấy tỷ lệ chuyển đổi giảm -1.2%. Bạn có quyết định phát hành hay không? Hãy viết tiêu chí giải quyết khi có xung đột (escalation criteria).
5. Áp dụng kỹ thuật CUPED với một biến đồng hành (covariate) ở giai đoạn trước giải thích được 60% phương sai của giai đoạn sau (hệ số tương quan bình phương $R^2 = 0.6$). Tính toán mức tăng kích thước mẫu hiệu dụng thu được.

## Key Terms

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Đánh giá ngoại tuyến (Eval) | "Kiểm thử offline" | Đánh giá năng lực của mô hình trên tập dữ liệu gán nhãn |
| Thử nghiệm A/B (A/B test) | "Thực nghiệm trực tuyến" | So sánh phân bổ ngẫu nhiên trên người dùng thực tế |
| CUPED | "Giảm phương sai" | Hồi quy biến dữ liệu giai đoạn trước để giảm phương sai thực nghiệm |
| Thử nghiệm tuần tự (Sequential test) | "Kiểm thử cho phép nhìn trước" | Quy trình thống kê luôn hợp lệ cho phép dừng sớm mà không thổi phồng sai số |
| So sánh nhiều lần (Multiple comparison) | "Sai số tích lũy họ" | Hiện tượng chạy nhiều kiểm định làm tăng vọt nguy cơ xuất hiện dương tính giả |
| Bonferroni | "Hiệu chỉnh khắt khe" | Chia mức ý nghĩa $\alpha$ cho tổng số lượng phép kiểm định |
| Benjamini-Hochberg | "Kiểm soát FDR" | Thuật toán kiểm soát tỷ lệ phát hiện sai, bớt bảo thủ hơn Bonferroni |
| Lệch tỷ lệ mẫu (SRM) | "Lệch phân bổ" | Hiện tượng sai lệch tỷ lệ phân chia mẫu; cảnh báo lỗi kỹ thuật gán nhóm |
| Statsig | "Nền tảng của OpenAI" | Nền tảng thương mại toàn diện, được OpenAI mua lại năm 2025 |
| GrowthBook | "Bản mã nguồn mở" | Nền tảng thử nghiệm mã nguồn mở MIT tích hợp trực tiếp kho dữ liệu |
| mSPRT | "Kiểm định tỷ số xác suất tuần tự" | Quy trình thống kê tuần tự kinh điển sử dụng hàm trộn |

## Further Reading

- [GrowthBook — Cách thức thử nghiệm A/B cho hệ thống AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Vượt ra khỏi Prompts: Tối ưu hóa LLM dựa trên dữ liệu](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [So sánh Statsig và GrowthBook](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng và cộng sự — Bài báo gốc về CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Chuỗi khoảng tin cậy (Confidence Sequences)](https://arxiv.org/abs/1810.08240)
