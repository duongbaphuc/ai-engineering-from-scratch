# Kiểm thử tải cho các API LLM — Tại sao k6 và Locust lại nói dối

> Các công cụ kiểm thử tải truyền thống (traditional load testers) không được thiết kế cho các phản hồi dạng luồng (streaming responses), độ dài đầu ra biến thiên, các chỉ số đo lường ở cấp độ token, hay sự bão hòa tài nguyên tính toán GPU. Có hai cái bẫy thường gặp khiến phần lớn các đội ngũ kỹ thuật mắc sai lầm. Bẫy khóa thông dịch viên GIL (GIL trap): cơ chế đo đạc ở cấp độ token của Locust thực hiện tác vụ tách từ (tokenization) dưới sự kiểm soát của khóa GIL trong Python, dẫn đến việc cạnh tranh tài nguyên với tiến trình sinh yêu cầu dưới mức đồng thời cao; hiện tượng ứ đọng hàng đợi tách từ sau đó sẽ làm thổi phồng độ trễ giữa các token (inter-token latency) được báo cáo — khi đó chính máy trạm kiểm thử (client) là nút thắt cổ chai chứ không phải máy chủ. Bẫy prompt đồng nhất (prompt-uniformity trap): việc gửi lặp đi lặp lại một prompt giống hệt nhau trong một vòng lặp chỉ kiểm thử đúng một điểm duy nhất trên phân phối token; trong khi lưu lượng thực tế có độ dài biến thiên liên tục và khớp tiền tố đa dạng. Công cụ LLMPerf giải quyết triệt để vấn đề này nhờ các tham số `--mean-input-tokens` kết hợp `--stddev-input-tokens`. Phân loại công cụ năm 2026: các công cụ chuyên biệt cho LLM (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) đảm bảo độ chính xác ở cấp độ token; **k6 v2026.1.0** + **k6 Operator 1.0 GA (tháng 9 năm 2025)** — nhận biết phản hồi dạng luồng, phân tán tự nhiên trên Kubernetes qua các định nghĩa tài nguyên tùy biến CRD TestRun/PrivateLoadZone, là lựa chọn tốt nhất cho các cổng kiểm soát CI/CD; Vegeta tối ưu cho kiểm thử bão hòa tốc độ cố định viết bằng Go; Locust 2.43.3 chỉ nên dùng khi tích hợp tiện ích mở rộng LLM-Locust cho luồng dữ liệu. Các mô hình tải: trạng thái ổn định (steady-state), tăng dần (ramp), tăng đột biến (spike - kiểm thử tự động co giãn), và ngâm tải dài hạn (soak - phát hiện rò rỉ bộ nhớ).

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## Learning Objectives

- Giải thích hai phản hình mẫu (bẫy GIL, bẫy prompt đồng nhất) khiến các công cụ kiểm thử tải thông thường đưa ra số liệu sai lệch đối với API LLM.
- Lựa chọn công cụ phù hợp theo từng mục đích cụ thể: LLMPerf (chạy kiểm chuẩn hiệu năng benchmark), k6 + tiện ích streaming (cổng kiểm soát CI), guidellm (kiểm chuẩn tổng hợp quy mô lớn), GenAI-Perf (kiểm chuẩn tham chiếu của NVIDIA).
- Thiết kế bốn mô hình tải (ổn định, tăng dần, đột biến, ngâm tải) và chỉ rõ phương thức lỗi (failure mode) mà mỗi mô hình có thể phát hiện.
- Xây dựng một phân phối prompt thực tế sử dụng giá trị trung bình (mean) + độ lệch chuẩn (stddev) của số lượng token đầu vào thay vì độ dài cố định.

## The Problem

Bạn vừa dùng k6 để kiểm thử tải điểm cuối (endpoint) LLM của mình ở mức 500 người dùng đồng thời. Hệ thống đứng vững xuất sắc. Bạn tự tin đưa lên môi trường production. Thế nhưng trong thực tế, chỉ với 200 người dùng thực, dịch vụ đã lập tức sụp đổ — chỉ số TTFT ở phân vị P99 tăng vọt mất kiểm soát, GPU bị khóa cứng ở mức 100%.

Có hai nguyên nhân dẫn đến thảm họa này. Thứ nhất, k6 đã gửi đi 500 prompt hoàn toàn giống hệt nhau — cơ chế gộp yêu cầu (request-coalescing) và lưu đệm tiền tố (prefix caching) của hệ thống đã khiến máy chủ trông như đang xử lý 500 tiến trình giải mã (decodes) đồng thời, nhưng thực chất chỉ đang tính toán đúng một yêu cầu duy nhất. Thứ hai, k6 không theo dõi độ trễ giữa các token (inter-token latency) trên các phản hồi dạng luồng (streaming) theo cách mà mắt người trực tiếp cảm nhận; nó chỉ nhìn nhận đó là một kết nối HTTP đơn lẻ, chứ không phải là 500 token đang đổ về theo các khoảng thời gian ngắt quãng khác nhau.

Kiểm thử tải cho LLM là một phân ngành kỹ thuật hoàn toàn riêng biệt.

## The Concept

### Bẫy GIL (The GIL trap - Locust)

Locust được viết bằng Python và thực hiện việc tách từ (tokenization) ngay tại máy khách dưới sự quản lý của khóa GIL (Global Interpreter Lock). Khi mức độ đồng thời tăng cao, tiến trình tách từ bị nghẽn lại phía sau tác vụ sinh yêu cầu mạng. Kết quả đo độ trễ giữa các token (inter-token latency) được báo cáo vô tình cộng thêm cả thời gian ứ đọng tách từ phía client. Bạn lầm tưởng rằng máy chủ phản hồi chậm, nhưng thực tế thủ phạm lại chính là khung kiểm thử (test harness).

Giải pháp khắc phục: Tiện ích mở rộng LLM-Locust tách tác vụ tách từ sang các tiến trình độc lập, hoặc chuyển sang dùng khung kiểm thử viết bằng ngôn ngữ biên dịch (như k6, hoặc LLMPerf với lõi `tokenizers.rs` viết bằng Rust).

### Bẫy prompt đồng nhất (The prompt-uniformity trap)

Hầu hết các công cụ kiểm thử tải truyền thống đều chỉ cho phép cấu hình một prompt duy nhất. Trong một vòng lặp kiểm thử 10.000 lần lặp, chính xác prompt đó được gửi đi liên tục. Máy chủ nhìn thấy cùng một đoạn tiền tố ở mọi lượt gọi — tỷ lệ trúng bộ nhớ đệm tiền tố (prefix cache hit rate) tiệm cận 100%, khiến thông lượng trông có vẻ vô cùng ấn tượng trên biểu đồ.

Giải pháp khắc phục: Lấy mẫu từ một phân phối prompt đa dạng. LLMPerf sử dụng các tham số `--mean-input-tokens 500 --stddev-input-tokens 150` — đảm bảo độ dài biến thiên và nội dung phong phú.

### Bốn mô hình kiểm thử tải

1. **Trạng thái ổn định (Steady-state)** — Giữ RPS không đổi trong 30-60 phút. Phát hiện: các hiện tượng suy giảm hiệu năng đường cơ sở (baseline regressions).
2. **Tăng dần (Ramp)** — Tăng tuyến tính RPS từ 0 lên mức mục tiêu trong vòng 15 phút. Phát hiện: điểm gãy giới hạn năng lực (capacity breakpoint), các hiện tượng bất thường khi khởi động làm ấm (warm-up).
3. **Đột biến (Spike)** — Tăng vọt đột ngột từ 3-10x RPS trong 2 phút rồi hạ về bình thường. Phát hiện: độ trễ co giãn tự động (autoscaling latency), hiện tượng bão hòa hàng đợi, tác động của khởi động lạnh (cold-start).
4. **Ngâm tải dài hạn (Soak)** — Duy trì tải ổn định liên tục trong 4-8 giờ. Phát hiện: hiện tượng rò rỉ bộ nhớ (memory leaks), trôi dạt vùng kết nối cơ sở dữ liệu (connection-pool drift), tràn bộ đệm hệ thống quan sát.

### Phân loại công cụ năm 2026

**LLMPerf** (Anyscale) — Viết bằng Python nhưng tách từ bằng lõi Rust tốc độ cao. Hỗ trợ cấu hình prompt theo giá trị trung bình/độ lệch chuẩn. Tương thích luồng dữ liệu streaming. Là lựa chọn mặc định tốt nhất cho các đợt đo kiểm chuẩn hiệu năng.

**NVIDIA GenAI-Perf** — Khung kiểm thử tham chiếu chính thức từ NVIDIA. Tương thích trực tiếp với máy khách Triton; độ bao phủ chỉ số đo lường toàn diện. Lưu ý rằng chỉ số ITL (Inter-Token Latency) của công cụ này loại trừ TTFT (Time To First Token); trong khi LLMPerf lại tính gộp cả TTFT vào. Hai công cụ này có thể đưa ra kết quả TPOT (Time Per Output Token) khác nhau trên cùng một máy chủ.

**LLM-Locust** (TrueFoundry) — Bản mở rộng của Locust giúp hóa giải hoàn toàn bẫy GIL. Giữ nguyên cú pháp quen thuộc của Locust DSL kết hợp với hệ thống đo lường phản hồi dạng luồng.

**guidellm** — Chuyên dùng cho các bài kiểm chuẩn tổng hợp quy mô lớn.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (tháng 9 năm 2025)**:
- Bản thân k6 (viết bằng Go, biên dịch mã máy, không có GIL) đã bổ sung các chỉ số đo lường nhận biết luồng streaming.
- k6 Operator tận dụng các CRD TestRun / PrivateLoadZone để kiểm thử phân tán tự nhiên trên hạ tầng Kubernetes.
- Là giải pháp tối ưu cho các cổng kiểm soát CI/CD và kiểm thử cam kết chất lượng dịch vụ SLA.

**Vegeta** — Viết bằng Go, cấu trúc tối giản hơn k6. Chuyên dùng để làm bão hòa lưu lượng HTTP ở tốc độ không đổi. Không hiểu ngữ nghĩa LLM nhưng rất tốt để kiểm thử cổng kết nối gateway và hạn ngạch giới hạn tốc độ.

**Locust 2.43.3 tiêu chuẩn** — Vẫn vướng phải bẫy GIL đối với LLM. Chỉ nên sử dụng nếu đi kèm tiện ích mở rộng LLM-Locust.

### Cổng kiểm soát SLA trong quy trình CI

Chạy k6 kiểm tra trực tiếp trên Pull Request với:

- 30-50 lần lặp tại mức RPS đường cơ sở.
- Cổng kiểm soát: TTFT ở P50/P95, tỷ lệ lỗi 5xx < 5%, TPOT dưới ngưỡng cho phép.
- Chặn đứng quy trình biên dịch (break the build) nếu vi phạm bất kỳ tiêu chí nào.

### Phân phối Prompt thực tế

Thu thập từ các mẫu lưu lượng truy cập thực tế (nếu bạn đã có sẵn) hoặc từ các tập phân phối đã công bố (ví dụ: prompt từ ShareGPT cho tác vụ trò chuyện, HumanEval cho tác vụ sinh mã). Đưa giá trị trung bình (mean) và độ lệch chuẩn (stddev) vào cấu hình của LLMPerf. Tuyệt đối tránh việc chạy vòng lặp với một prompt đơn lẻ.

### Những con số cần ghi nhớ

- k6 Operator 1.0 GA: Phát hành chính thức tháng 9 năm 2025.
- k6 v2026.1.0: Tích hợp sẵn các chỉ số đo lường luồng dữ liệu streaming.
- Một lượt chạy LLMPerf điển hình: 100-1000 yêu cầu ở mức đồng thời X.
- Cổng kiểm soát CI điển hình: 30-50 lần lặp trên mỗi Pull Request.
- Bốn mô hình tải: ổn định (steady), tăng dần (ramp), đột biến (spike), ngâm tải (soak).

```figure
load-pattern-waves
```

## Use It

Tệp `code/main.py` mô phỏng một bài kiểm thử tải với phân phối prompt thực tế, đo đạc TPOT hiệu dụng, và minh họa rõ ràng cái bẫy của việc sử dụng prompt đồng nhất.

## Ship It

Bài học này xây dựng tệp `outputs/skill-load-test-plan.md`. Dựa trên khối lượng công việc và cam kết SLA, kỹ năng này sẽ lựa chọn công cụ phù hợp và thiết kế kế hoạch cho cả bốn mô hình kiểm thử tải.

## Exercises

1. Chạy `code/main.py`. So sánh kết quả giữa phân phối prompt đồng nhất và phân phối thực tế — khoảng cách sai lệch nằm ở đâu?
2. Viết kịch bản k6 cho một cổng kiểm soát CI: yêu cầu TTFT P95 < 800 ms ở mức 100 người dùng đồng thời, thời gian chạy thử 5 phút.
3. Bài kiểm thử ngâm tải của bạn cho thấy bộ nhớ tăng đều 50 MB/giờ. Nêu ba nguyên nhân tiềm ẩn và các công cụ đo lường để phân lập thủ phạm.
4. Kiểm thử đột biến từ 10 RPS lên 100 RPS. Thời gian phục hồi dự kiến là bao lâu nếu hệ thống đã triển khai đầy đủ Karpenter + vLLM trong môi trường production (Phase 17 · 03 + 18)?
5. GenAI-Perf báo cáo TPOT=6ms trong khi LLMPerf báo cáo TPOT=11ms trên cùng một máy chủ phần cứng. Hãy giải thích nguyên nhân gây ra sự khác biệt này.

## Key Terms

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| LLMPerf | "Khung đo chuẩn cho LLM" | Công cụ kiểm chuẩn của Anyscale, hiểu rõ luồng dữ liệu streaming |
| GenAI-Perf | "Công cụ của NVIDIA" | Khung kiểm thử tham chiếu chuẩn từ NVIDIA |
| LLM-Locust | "Locust cho LLM" | Tiện ích mở rộng của Locust giúp khắc phục triệt để bẫy GIL |
| guidellm | "Công cụ kiểm chuẩn tổng hợp" | Công cụ tạo bài kiểm chuẩn tổng hợp ở quy mô lớn |
| k6 Operator | "k6 chạy trên K8s" | Trình điều khiển k6 phân tán dựa trên định nghĩa tài nguyên CRD |
| Bẫy GIL (GIL trap) | "Nghẽn tiến trình Python client" | Hiện tượng ứ đọng tác vụ tách từ phía client làm thổi phồng độ trễ báo cáo |
| Bẫy prompt đồng nhất | "Dối lừa từ prompt đơn lẻ" | Vòng lặp gửi cùng một prompt đánh trúng bộ đệm tiền tố, làm ảo thông lượng |
| Trạng thái ổn định (Steady-state) | "Tải không đổi" | Duy trì mức RPS phẳng trong suốt N phút |
| Tăng dần (Ramp) | "Tăng tuyến tính" | Tăng dần tải từ 0 lên mức mục tiêu theo thời gian |
| Đột biến (Spike) | "Kiểm thử bùng nổ tải" | Tăng vọt tải gấp nhiều lần trong thời gian ngắn rồi hạ về mức cũ |
| Ngâm tải (Soak) | "Kiểm thử trường kỳ" | Chạy tải liên tục trong nhiều giờ để phát hiện rò rỉ tài nguyên |

## Further Reading

- [TianPan — Kiểm thử tải cho các ứng dụng LLM](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Kiểm thử tải LLM năm 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Giới thiệu về Kiểm chuẩn Suy luận LLM](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — Giới thiệu LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [Kho mã nguồn LLMPerf trên GitHub](https://github.com/ray-project/llmperf)
- [Kho mã nguồn k6 Operator trên GitHub](https://github.com/grafana/k6-operator)
