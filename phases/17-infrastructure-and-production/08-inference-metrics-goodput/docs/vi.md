# Các chỉ số suy luận — TTFT, TPOT, ITL, Goodput, P99

> Năm chỉ số quyết định liệu một hệ thống triển khai suy luận có đang vận hành hiệu quả hay không. TTFT (thời gian đến token đầu tiên / time to first token) bao gồm thời gian chờ hàng đợi, truyền tải mạng và pha tiền điền (prefill). TPOT (thời gian trên mỗi token đầu ra / time per output token — thường được báo cáo dưới tên gọi ITL / khoảng cách giữa các token liên tiếp / inter-token latency, dù các công cụ đo kiểm còn bất đồng về việc ITL có tính token đầu tiên hay không) là chi phí giải mã (decode) bị giới hạn bởi bộ nhớ trên mỗi token. Độ trễ đầu-cuối (end-to-end latency - E2E) bằng TTFT cộng với TPOT nhân với độ dài chuỗi đầu ra. Thông lượng (throughput) là tổng số token trên giây tính gộp trên toàn bộ cụm máy chủ. Nhưng chỉ số quan trọng nhất đối với sản phẩm thực tế là goodput (thông lượng hữu dụng) — tỷ lệ các yêu cầu thỏa mãn đồng thời tất cả các ràng buộc trong cam kết mức dịch vụ (Service Level Objective - SLO). Thông lượng cao nhưng goodput thấp đồng nghĩa với việc bạn đang xử lý các token không bao giờ kịp đến tay người dùng đúng hạn. Các con số tham chiếu cho mô hình Llama-3.1-8B-Instruct trên TRT-LLM vào năm 2026: TTFT trung bình 162 ms, TPOT trung bình 7,33 ms, E2E trung bình 1.093 ms. Luôn luôn báo cáo đầy đủ các phân vị P50, P90, P99 — tuyệt đối không chỉ dựa vào giá trị trung bình (mean). Và hãy cảnh giác với bẫy đo lường: GenAI-Perf loại trừ TTFT khỏi công thức tính ITL, trong khi LLMPerf lại gộp cả TTFT; hai công cụ đo kiểm có thể đưa ra kết quả TPOT hoàn toàn khác nhau trên cùng một lượt chạy.

**Phân loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, công cụ tính phân vị và báo cáo goodput mô phỏng)
**Điều kiện tiên quyết:** Phase 17 · 04 (Serving Engine Internals)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Định nghĩa chính xác TTFT, TPOT, ITL, E2E, thông lượng và goodput, đồng thời chỉ rõ từng thành phần mà mỗi chỉ số đo lường.
- Giải thích lý do giá trị trung bình (mean) là thước đo thống kê sai lầm đối với hệ thống phục vụ LLM và cách đọc các phân vị P50/P90/P99.
- Xây dựng một ràng buộc đa chiều cho SLO (ví dụ: TTFT < 500 ms VÀ TPOT < 15 ms VÀ E2E < 2 s) và tính toán goodput dựa trên ràng buộc đó.
- Nêu tên hai công cụ đo điểm chuẩn (benchmarking tools) cho kết quả TPOT mâu thuẫn nhau trên cùng một lượt chạy và giải thích nguyên nhân.

## Vấn đề

"Thông lượng của chúng tôi đạt 15.000 token mỗi giây." Con số đó có ý nghĩa gì? Nếu 40% số yêu cầu vượt quá 2 giây tính từ đầu đến cuối, người dùng sẽ lập tức từ bỏ phiên làm việc. Chỉ số thông lượng đơn thuần không thể phản ánh liệu sản phẩm có hoạt động đúng kỳ vọng hay không.

Quá trình suy luận bao gồm nhiều trục độ trễ khác nhau và mỗi trục lại gặp sự cố theo một cơ chế riêng. Tiền điền (prefill) bị giới hạn bởi năng lực tính toán (compute-bound) và tỉ lệ thuận với độ dài câu lệnh nhắc (prompt length). Giải mã (decode) bị giới hạn bởi băng thông bộ nhớ (memory-bound) và tỉ lệ thuận với kích thước lô (batch size). Độ trễ hàng đợi là vấn đề vận hành điều phối. Độ trễ mạng là vấn đề khoảng cách vật lý. Bạn cần các chỉ số riêng biệt cho từng thành phần, bạn cần các phân vị thống kê, và bạn cần một chỉ số tổng hợp duy nhất trả lời câu hỏi "liệu người dùng có nhận được trải nghiệm như mong đợi hay không" — đó chính là goodput.

## Khái niệm

### TTFT — Thời gian đến token đầu tiên

`TTFT = queue_time + network_request + prefill_time`

Pha prefill chiếm ưu thế tuyệt đối khi câu lệnh nhắc dài. Trên cấu hình Llama-3.3-70B FP8 chạy trên H100, một prompt dài 32k token tiêu tốn khoảng ~800 ms thời gian prefill thuần túy. Thời gian hàng đợi (queue time) phản ánh hành vi của bộ lập lịch (scheduler) khi hệ thống chịu tải. Thời gian mạng (network request) là độ trễ truyền dẫn dây cáp bao gồm cả bắt tay TLS. TTFT chính là khoảng thời gian người dùng phải chờ đợi trước khi token đầu tiên bắt đầu được truyền trực tiếp (stream) về giao diện.

### TPOT / ITL — Độ trễ giữa các token liên tiếp

Nhiều tên gọi khác nhau cho cùng một đại lượng vật lý: `TPOT` (thời gian trên mỗi token đầu ra / time per output token), `ITL` (độ trễ giữa các token / inter-token latency), `độ trễ giải mã trên mỗi token` — tất cả đều là một. Đó là khoảng thời gian trôi qua giữa hai token liên tiếp được truyền về sau khi token đầu tiên đã xuất hiện.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

Trên cùng ngăn xếp Llama-3.3-70B H100 có áp dụng cơ chế prefill phân mảnh (chunked prefill), TPOT trung bình đạt ~7 ms. Nếu không có chunked prefill, trong khi một chuỗi lân cận đang thực hiện prefill dài, TPOT có thể vọt lên tới 50 ms. Hãy luôn theo dõi phân vị P99 thay vì giá trị trung bình.

### Độ trễ đầu-cuối (E2E)

`E2E = TTFT + TPOT * output_tokens + network_response`

Đối với các chuỗi đầu ra dài (>500 token), E2E bị chi phối chủ yếu bởi TPOT. Đối với chuỗi đầu ra ngắn nhưng prompt đầu vào dài, E2E bị chi phối bởi TTFT. Luôn báo cáo độ trễ E2E có kèm theo điều kiện về độ dài đầu ra.

### Thông lượng (Throughput)

`throughput = total_output_tokens / elapsed_time`

Đây là chỉ số tổng gộp (aggregate metric). Nó phản ánh hiệu quả sử dụng tài nguyên của toàn bộ cụm máy chủ. Nó không cho bạn biết tình trạng sức khỏe của từng yêu cầu người dùng đơn lẻ.

### Goodput — Chỉ số cốt lõi mà bạn thực sự quan tâm

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO là một tập hợp đa ràng buộc. Một yêu cầu chỉ được coi là "tốt" (good) nếu nó thỏa mãn đồng thời tất cả các ràng buộc đặt ra. Goodput là tỷ lệ phần trăm các yêu cầu đạt chuẩn này. Một số bài báo và bảng điều khiển báo cáo goodput dưới dạng tốc độ: số yêu cầu mỗi giây thỏa mãn toàn bộ tiêu chí SLO. Để quy đổi, chỉ cần nhân tỷ lệ phần trăm với tốc độ yêu cầu tổng thể. Đạt thông lượng cao nhưng chỉ có 60% goodput là một sự thất bại. Mục tiêu thực sự là thông lượng ổn định với mức goodput từ 99% trở lên.

Vào năm 2026, goodput là thước đo chính thức được sử dụng trong các bài nộp đo kiểm MLPerf Inference v6.0 và trong hệ thống theo dõi SLA nội bộ của các nhà cung cấp nền tảng AI.

### Tại sao giá trị trung bình là thống kê sai lệch

Phân phối độ trễ của mô hình ngôn ngữ lớn luôn bị lệch phải (right-skewed). Một lô giải mã (decode batch) chứa một chuỗi lân cận đang thực hiện prefill dài có thể xuất ra 500 token với TPOT ~7 ms và 20 token với TPOT ~60 ms. TPOT trung bình khi đó là 9 ms. Nhưng TPOT P99 lại lên tới 65 ms. Người dùng thường xuyên gặp phải mức P99 này — và đó chính là lý do họ rời bỏ sản phẩm.

Luôn báo cáo bộ ba phân vị (P50, P90, P99). Đối với trải nghiệm người dùng, P99 là chỉ số duy nhất bạn cần tập trung tối ưu hóa.

### Các con số tham chiếu — Llama-3.1-8B-Instruct trên TRT-LLM, năm 2026

- TTFT trung bình: 162 ms
- TPOT trung bình: 7,33 ms
- E2E trung bình: 1.093 ms
- TPOT P99: dao động trong khoảng 10–25 ms tùy thuộc vào cấu hình chunked prefill.

Đây là các điểm tham chiếu chính thức do NVIDIA công bố. Các con số này sẽ thay đổi tùy theo kích thước mô hình (mô hình 70B sẽ tăng từ 3 đến 5 lần), phần cứng (chuyển từ H100 sang B200 giảm khoảng 3 lần), và tải hệ thống.

### Bẫy đo lường

Hai trong số các công cụ đo kiểm chuẩn phổ biến nhất năm 2026 đưa ra kết quả TPOT hoàn toàn khác nhau trên cùng một lượt chạy:

- **NVIDIA GenAI-Perf**: loại trừ TTFT ra khỏi phép tính ITL. Quá trình tính ITL bắt đầu từ token thứ 2.
- **LLMPerf**: bao gồm cả TTFT vào phép tính. Quá trình tính ITL bắt đầu ngay từ token thứ 1.

Với một yêu cầu có TTFT là 500 ms và tạo ra 100 token đầu ra trong 700 ms tổng thời gian giải mã: GenAI-Perf sẽ báo cáo `ITL = 700/99 = 7.07 ms`, trong khi LLMPerf lại báo cáo `ITL = 1200/100 = 12.00 ms`. Việc lựa chọn công cụ đo lường làm thay đổi hoàn toàn con số kết quả.

Hãy luôn ghi rõ tên công cụ sử dụng. Hãy luôn công bố định nghĩa toán học của chỉ số.

### Xây dựng một SLO

Một cam kết mức dịch vụ (SLO) hợp lý cho người dùng cuối của mô hình hội thoại 70B vào năm 2026:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s cho các phản hồi <300 token.
- Mục tiêu Goodput >= 99%.

Các SLO cấp doanh nghiệp (enterprise) thường siết chặt TTFT hơn (200–400 ms) và nới lỏng E2E. Điều mấu chốt là phải văn bản hóa các tiêu chí này, đo lường đồng thời cả ba thành phần và theo dõi goodput như một chỉ số tổng hợp duy nhất.

### Cách thức đo lường

- Chạy trên lưu lượng truy cập thực tế hoặc luồng giả lập thực tế (LLMPerf với `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Nhắm tới mức tải đồng thời gấp 2 lần mức đỉnh (2x peak concurrency) cho đợt đo kiểm chuẩn.
- Thực hiện từ 30 đến 50 lần lặp, lấy các phân vị trên toàn bộ mẫu dữ liệu gộp.
- Công bố kèm tên công cụ, phiên bản công cụ, mô hình, cấu hình phần cứng, mức độ đồng thời và phân phối độ dài câu lệnh nhắc.

```figure
throughput-latency
```

## Sử dụng

Tệp `code/main.py` là một công cụ tính toán goodput mô phỏng. Nó tạo ra phân phối độ trễ giả lập, áp dụng các ràng buộc SLO và tính toán goodput. Chương trình cũng minh họa rõ sự khác biệt giữa cách tính TPOT của GenAI-Perf và LLMPerf trên cùng một dấu vết dữ liệu (trace).

Công cụ trực quan [Bên trong GPU phục vụ LLM](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?users=1) chỉ ra nguồn gốc của thời gian tiêu tốn cho mỗi token đầu ra trên một card NVIDIA L4. Một bước giải mã cần đọc 16,41 GB trọng số, tiêu tốn ít nhất 54,7 ms ở băng thông 300 GB/s, và bước đo thực tế ghi nhận mức 60,16 ms.

## Phát hành

Bài học này cung cấp tệp `outputs/skill-slo-goodput-gate.md`. Dựa trên khối lượng công việc và SLO mục tiêu, công cụ này tạo ra quy trình đo kiểm chuẩn tích hợp CI/CD để kiểm soát việc triển khai dịch vụ dựa trên goodput thay vì chỉ dựa vào thông lượng thuần túy.

## Bài tập

1. Chạy `code/main.py`. Tạo một phân phối độ trễ với đột biến ở phần đuôi (tail spike) chiếm 1%. Chỉ số goodput thay đổi như thế nào khi bạn siết chặt ngưỡng TPOT P99 từ 30 ms xuống 15 ms?
2. Một nhà cung cấp quảng cáo "15.000 tok/s trên Llama 3.3 70B H100". Hãy nêu ba câu hỏi bạn cần đặt ra trước khi tin tưởng vào tuyên bố này.
3. Tại sao kỹ thuật prefill phân mảnh (chunked prefill) bảo vệ được chỉ số TPOT P99 nhưng không cải thiện được TPOT trung bình?
4. Thiết kế một SLO hướng người dùng cho trợ lý ảo giọng nói (nơi token đầu tiên được nghe thay vì đọc). Chỉ số nào là dễ nhận biết nhất đối với người dùng?
5. Đọc tài liệu README của LLMPerf và tài liệu hướng dẫn của GenAI-Perf. Xác định thêm ba chỉ số khác mà hai công cụ này có cách định nghĩa mâu thuẫn nhau.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| TTFT | "thời gian ra token đầu" | Hàng đợi + mạng + tiền điền; bị chi phối bởi prefill khi câu lệnh dài |
| TPOT | "thời gian mỗi token xuất ra" | Chi phí giải mã bị giới hạn bởi bộ nhớ cho mỗi token tiếp theo sau token đầu |
| ITL | "độ trễ giữa các token" | Tương đương TPOT ở phần lớn công cụ (trừ một số ngoại lệ — xem GenAI-Perf) |
| E2E | "từ đầu đến cuối" | TTFT + TPOT * output_len; cộng thêm độ trễ mạng ở chiều phản hồi |
| Throughput | "token mỗi giây" | Hiệu suất sử dụng cụm phần cứng; vô nghĩa nếu thiếu các phân vị độ trễ |
| Goodput | "tỷ lệ đạt chuẩn SLO" | Tỷ lệ yêu cầu thỏa mãn đồng thời mọi ràng buộc của SLO |
| P99 | "độ trễ phần đuôi" | Mức độ trễ của 1% trường hợp xấu nhất; thước đo trải nghiệm người dùng thực |
| SLO multi-constraint | "ràng buộc đồng thời" | Phép toán logic AND giữa cả ba ngưỡng độ trễ; yêu cầu thất bại nếu vi phạm bất kỳ ngưỡng nào |
| GenAI-Perf vs LLMPerf | "bẫy công cụ" | Hai công cụ bất đồng về việc ITL có bao gồm TTFT hay không |

## Tài liệu đọc thêm

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — Định nghĩa chuẩn mực của TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — Các định nghĩa thay thế và phương pháp đo lường thực tế.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — Đo lường ứng dụng trên các môi trường triển khai thực tế.
- [LLMPerf](https://github.com/ray-project/llmperf) — Bộ đo kiểm chuẩn mã nguồn mở dựa trên Ray.
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md) — Công cụ đo kiểm chuẩn hiệu năng của NVIDIA.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — Bộ tiêu chuẩn đo kiểm dựa trên goodput được toàn ngành công nhận.
