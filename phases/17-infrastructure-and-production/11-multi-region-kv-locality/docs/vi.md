# Phục vụ LLM đa vùng và tính cục bộ của bộ nhớ đệm KV Cache

> Cân bằng tải dạng xoay vòng (round-robin load balancing) thực sự gây hại nghiêm trọng cho việc suy luận LLM có sử dụng bộ nhớ đệm. Một yêu cầu không được chuyển đến đúng nút đang lưu trữ tiền tố (prefix) của nó sẽ phải gánh chịu toàn bộ chi phí tính toán tiền điền (prefill) — vào khoảng 800 ms ở phân vị P50 đối với câu lệnh nhắc dài so với chỉ ~80 ms nếu trúng bộ nhớ đệm (cache hit). Vào năm 2026, mẫu kiến trúc chuẩn trong môi trường production là bộ định tuyến nhận biết bộ nhớ đệm (cache-aware router như vLLM Router viết bằng Rust, hoặc llm-d router) liên tục thu nạp các sự kiện KV-cache và định tuyến dựa trên độ khớp mã băm tiền tố (prefix-hash match). Nghiên cứu gần đây (GORGO) đã đưa độ trễ mạng liên vùng thành một số hạng hiện hữu trong hàm mục tiêu định tuyến. Các dịch vụ thương mại về "suy luận đa vùng" (như Bedrock cross-region inference, cổng kết nối đa cụm GKE multi-cluster gateways) xem quá trình suy luận như một hộp đen mờ đục — chúng chỉ giải quyết bài toán tính sẵn sàng (availability), chứ không tối ưu hóa TTFT. JPMorgan và Mayo Clinic đã tiến hành diễn tập chuyển vùng dự phòng (failover) tại us-east-1 vào tháng 11 năm 2024 và đạt thời gian phục hồi ~22 phút. Thực tế khắc nghiệt của bài toán phục hồi thảm họa (Disaster Recovery - DR): 32% các sự cố DR thất bại ở các hệ thống LLM là do các đội ngũ chỉ sao lưu trọng số mà quên mất các tệp của bộ tách từ (tokenizer files) hoặc tệp cấu hình lượng tử hóa.

**Phân loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng định tuyến nhận biết bộ nhớ đệm tiền tố)
**Điều kiện tiên quyết:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do cơ chế cân bằng tải xoay vòng phá vỡ hiệu năng của suy luận có bộ nhớ đệm và định lượng tổn thất độ trễ TTFT phát sinh.
- Vẽ sơ đồ luồng hoạt động của bộ định tuyến nhận biết bộ nhớ đệm: đầu vào (các sự kiện KV-cache), thuật toán (khớp mã băm tiền tố), cơ chế giải quyết xung đột (mức sử dụng GPU).
- Chỉ ra nguyên nhân cốt lõi gây ra 32% số vụ thất bại khi diễn tập phục hồi thảm họa DR cho LLM (thiếu tệp bộ tách từ / tệp cấu hình lượng tử hóa) và xây dựng danh mục kiểm tra DR tối thiểu gồm 3 tệp.
- Phân biệt rõ sự khác nhau giữa các dịch vụ suy luận đa vùng thương mại (Bedrock CRI, GKE Multi-Cluster Gateway) với cơ chế định tuyến nhận biết KV-cache.

## Vấn đề

Dịch vụ của bạn được triển khai đồng thời tại us-east-1, us-west-2 và eu-west-1. Bạn đặt một bộ cân bằng tải ứng dụng (Application Load Balancer - ALB) ở phía trước với thuật toán xoay vòng (round-robin). Tỷ lệ trúng bộ nhớ đệm tiền tố (prefix cache hit rate) trong môi trường production lập tức tụt dốc xuống chỉ còn 8%. Độ trễ TTFT P50 tăng gấp 3 lần. Nhật ký vận hành vLLM cho thấy hầu như mọi yêu cầu đều phải trả toàn bộ chi phí prefill từ đầu.

Thuật toán xoay vòng là tối ưu cho các dịch vụ phi trạng thái (stateless services). Nhưng suy luận LLM về bản chất là một quy trình có trạng thái sâu sắc (stateful by design) — KV cache mã hóa toàn bộ ngữ cảnh mà mô hình đã xử lý trước đó. Định tuyến một cách mù quáng đồng nghĩa với việc bạn đang đẩy yêu cầu vào sai bộ nhớ đệm.

Một vấn đề tách biệt khác: đội ngũ của bạn đã chuẩn bị kế hoạch phục hồi thảm họa (Disaster Recovery - DR). Bạn sao lưu định kỳ trọng số mô hình sang kho chứa S3 ở vùng khác. Một sự cố diện rộng xảy ra trên toàn vùng; bạn kích hoạt quy trình chuyển vùng dự phòng; bản sao mới kiên quyết từ chối khởi động. Bạn đã bỏ quên tệp `tokenizer.json`, tệp cấu hình lượng tử hóa, và cấu hình co giãn RoPE vốn nằm ở một kho chứa riêng biệt mà không được đồng bộ hóa.

Phục vụ LLM trên nhiều vùng địa lý thực chất là bài toán về bộ nhớ đệm, bài toán về định tuyến thông minh và bài toán về tính kỷ luật trong quy trình DR — chứ không đơn thuần là việc cấu hình một bộ cân bằng tải.

## Khái niệm

### Định tuyến nhận biết bộ nhớ đệm (Cache-aware routing)

Một yêu cầu gửi tới kèm theo câu lệnh nhắc (prompt). Bộ định tuyến tiến hành băm (hash) phần tiền tố (chẳng hạn 512 token đầu tiên); sau đó nó truy vấn từng bản sao phục vụ: "Bạn có đang lưu tiền tố này trong bộ nhớ đệm hay không?". Các bản sao liên tục phát đi các sự kiện KV-cache trên một kênh xuất bản/đăng ký (pub/sub channel) mỗi khi chúng cấp phát hoặc giải phóng (evict) các khối bộ nhớ. Bộ định tuyến sẽ chọn bản sao có tiền tố khớp sẵn, và chỉ chuyển sang tiêu chí phụ (mức sử dụng GPU thấp nhất) nếu không có bản sao nào khớp.

**vLLM Router** (viết bằng Rust, ngăn xếp production chuẩn năm 2026): đăng ký nhận các sự kiện `kv.cache.block_added`, duy trì một bảng chỉ mục ánh xạ mã băm tiền tố → bản sao phục vụ, và thực hiện định tuyến với độ phức tạp tra cứu $O(1)$. Nó chỉ rơi về thuật toán chọn hàng đợi ngắn nhất khi không tìm thấy bộ nhớ đệm phù hợp.

**llm-d router**: cùng mô hình thiết kế nhưng được tối ưu hóa nguyên bản cho Kubernetes. Nó phát các sự kiện thông qua ControlPlane API.

**SGLang RadixAttention** (Phase 17 · 06) là giải pháp tương đương áp dụng nội bộ bên trong từng bản sao đơn lẻ (intra-replica). Việc định tuyến giữa các bản sao độc lập (cross-replica routing) bắt buộc phải nằm ở tầng trên (upstream).

### Các con số thực tế

Độ trễ TTFT P50 trên câu lệnh nhắc 2.000 token, mô hình Llama 3.3 70B FP8, chạy trên H100:
- Trúng bộ nhớ đệm (cùng bản sao, tiền tố đã nạp sẵn): ~80 ms.
- Trượt bộ nhớ đệm (tiền điền nguội hoàn toàn): ~800 ms.

Chênh lệch tới 10 lần. Nếu bộ định tuyến của bạn đạt tỷ lệ trúng bộ nhớ đệm tiền tố từ 60% đến 80% trên toàn bộ các bản sao, bạn sẽ đạt được độ trễ tương đương với một bản sao duy nhất nhưng với năng lực phục vụ của N bản sao gộp lại. Nếu tỷ lệ trúng đệm chỉ đạt 10%, hệ thống của bạn chẳng khác nào mở rộng quy mô một cách thô sơ và lãng phí.

### Đa vùng xuất hiện một ràng buộc mới — độ trễ mạng

Độ trễ khứ hồi (Round-Trip Time - RTT) giữa các vùng:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

Nếu bộ định tuyến chuyển một yêu cầu từ us-east-1 sang vùng ap-southeast-1 chỉ vì ở đó có tiền tố nóng, mức thời gian prefill tiết kiệm được (từ 800 ms giảm xuống 80 ms, tức tiết kiệm 720 ms) sẽ bị co hẹp lại chỉ còn khoảng 500 ms sau khi đã cộng thêm ~220 ms RTT mạng. Khoảng chênh lệch này sẽ hoàn toàn biến mất nếu vùng máy chủ từ xa bắt yêu cầu phải xếp hàng chờ đợi hoặc đã giải phóng mất tiền tố đó. Nghiên cứu GORGO (năm 2026) đã mô hình hóa rõ ràng bài toán này — tối thiểu hóa đồng thời tổng của `thời_gian_prefill + độ_trễ_mạng`, chứ không tối ưu hóa prefill một cách đơn độc. Thông thường, giải pháp tốt nhất là giữ việc định tuyến trong phạm vi nội vùng, ngoại trừ các tiền tố khổng lồ hàng megabyte nơi thời gian prefill chiếm tỉ trọng áp đảo tuyệt đối.

### Các giải pháp thương mại "suy luận đa vùng" không giải quyết được vấn đề này

Dịch vụ AWS Bedrock Cross-Region Inference tự động định tuyến các yêu cầu sang các vùng khác khi vùng hiện tại gặp áp lực quá tải tài nguyên. Nó tối ưu hóa tính sẵn sàng (availability), chứ không tối ưu hóa TTFT, và coi quá trình suy luận như một hộp đen không trạng thái. Cổng kết nối đa cụm GKE Multi-Cluster Gateway cũng hoạt động theo cơ chế tương tự — chuyển đổi dự phòng ở cấp độ dịch vụ mà hoàn toàn không có nhận thức gì về KV cache.

Bạn vẫn bắt buộc phải tự xây dựng bộ định tuyến nhận biết bộ nhớ đệm ở tầng ứng dụng ngay cả khi đang sử dụng các dịch vụ thương mại này. Chúng giải quyết bài toán "khi vùng us-east-1 gặp sự cố nghiêm trọng". Còn bộ định tuyến nhận biết bộ nhớ đệm giải quyết bài toán tối ưu hóa TTFT hàng ngày.

### Vệ sinh quy trình phục hồi thảm họa — bài toán 32% tệp bị bỏ quên

Thống kê được trích dẫn rộng rãi vào năm 2026: 32% các sự cố phục hồi thảm họa LLM thất bại là do các đội ngũ kỹ thuật chỉ sao lưu trọng số mà bỏ quên:

- Tệp `tokenizer.json` hoặc `tokenizer.model`.
- Tệp cấu hình lượng tử hóa (`quantize_config.json`, hệ số tỉ lệ AWQ, điểm 0 của GPTQ).
- Cấu hình chuyên biệt của mô hình (hệ số co giãn RoPE, mặt nạ chú ý, mẫu hội thoại / chat templates).
- Cấu hình động cơ phục vụ (`vllm_config.yaml`, tham số lấy mẫu mặc định, danh sách bộ chuyển đổi LoRA).

Biện pháp khắc phục là thiết lập danh mục kiểm tra phục hồi thảm họa (DR manifest) tối thiểu gồm 3 phần:

1. Toàn bộ các tệp nằm trong kho chứa mô hình HuggingFace (trọng số + cấu hình + bộ tách từ).
2. Tệp cấu hình phục vụ chuyên biệt của động cơ (engine config).
3. Bản kê khai triển khai hạ tầng (K8s YAML, Dockerfile, tệp khóa phụ thuộc / dependency lock).

Đồng thời: bắt buộc phải diễn tập quy trình DR định kỳ mỗi quý một lần. Đợt diễn tập chuyển vùng us-east-1 của JPMorgan vào tháng 11 năm 2024 chỉ mất 22 phút để khôi phục hoàn toàn chính là nhờ kịch bản vận hành đã được thực hành nhuần nhuyễn trước đó.

### Tính cư trú của dữ liệu là một chiều độc lập

Dữ liệu thông tin y tế cá nhân (PHI) của khách hàng tại Liên minh Châu Âu (EU) theo luật không được phép rời khỏi biên giới EU. Nếu bộ định tuyến nhận biết bộ nhớ đệm của bạn chuyển một yêu cầu xuất phát từ Paris sang vùng us-east-1 chỉ để khớp tiền tố đệm, bạn đã vi phạm quy định GDPR nghiêm trọng bất kể độ trễ TTFT có được cải thiện đến mức nào. Hãy phân vùng các bộ định tuyến theo ranh giới pháp lý về quyền cư trú dữ liệu trước khi tiến hành tối ưu hóa bộ nhớ đệm.

### Các con số cần ghi nhớ

- Chênh lệch TTFT giữa trúng và trượt đệm: ~10 lần (80 ms so với 800 ms trên prompt 2K).
- Độ trễ mạng RTT giữa Mỹ và Châu Âu: ~75 ms.
- Thất bại DR: 32% trường hợp do thiếu tệp tokenizer hoặc cấu hình lượng tử hóa.
- Diễn tập chuyển vùng us-east-1 của JPMorgan tháng 11/2024: 22 phút (cam kết SLA 30 phút).

```figure
cache-aware-router
```

## Sử dụng

Tệp `code/main.py` mô phỏng ba chiến lược định tuyến khác nhau (xoay vòng ngẫu nhiên, nhận biết đệm cấp vùng, nhận biết đệm toàn cầu) trên một khối lượng công việc đa vùng thực tế. Chương trình sẽ báo cáo tỷ lệ trúng bộ nhớ đệm, các phân vị TTFT P50/P99, và chi phí truyền dữ liệu liên vùng phát sinh.

## Phát hành

Bài học này cung cấp tệp `outputs/skill-multi-region-router.md`. Dựa trên danh sách các vùng triển khai, các ràng buộc về quyền cư trú dữ liệu và cam kết SLA độ trễ, công cụ này sẽ thiết kế một kế hoạch định tuyến tối ưu.

## Bài tập

1. Chạy `code/main.py`. Với độ trễ mạng RTT là 75 ms, ở độ dài câu lệnh nhắc bao nhiêu token thì việc định tuyến sang vùng khác để trúng đệm bắt đầu vượt trội hơn so với việc tính toán lại tại vùng cục bộ?
2. Tỷ lệ trúng bộ nhớ đệm của hệ thống sụt giảm bất thường từ 70% xuống chỉ còn 12%. Hãy phân tích ba nguyên nhân tiềm ẩn và nêu rõ các chỉ số quan sát được (observables) để xác nhận từng nguyên nhân đó.
3. Thiết kế một bản kê khai DR hoàn chỉnh cho một mô hình 70B lượng tử hóa bằng AWQ, được phục vụ trong vLLM cùng với 5 bộ chuyển đổi LoRA. Liệt kê chi tiết từng tệp và cấu hình bắt buộc phải sao lưu.
4. Thảo luận xem liệu tính năng Bedrock Cross-Region Inference có "đủ tốt" cho một công ty công nghệ tài chính (fintech) với các cam kết khắt khe về TTFT SLO hay không? Trích dẫn các hành vi cụ thể của hệ thống.
5. Một yêu cầu xuất phát từ Paris có tiền tố trùng khớp với bộ nhớ đệm đang nằm ở us-east-1. Bạn có được phép định tuyến yêu cầu này sang Mỹ hay không? Hãy soạn thảo điều khoản chính sách xử lý tình huống này.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Cache-aware routing | "cân bằng tải thông minh" | Định tuyến dựa trên khớp mã băm tiền tố tới bản sao đang giữ KV-cache |
| KV-cache events | "sự kiện bộ đệm pub-sub" | Các bản sao phát sự kiện cấp phát/giải phóng khối nhớ; bộ định tuyến lập chỉ mục |
| Prefix hash | "khóa bộ nhớ đệm" | Mã băm của N token đầu tiên dùng làm khóa tra cứu cho bộ định tuyến |
| GORGO | "nghiên cứu định tuyến liên vùng" | Bài báo arXiv 2602.11688; đưa độ trễ mạng thành số hạng hiện hữu trong hàm mục tiêu |
| Cross-region inference | "suy luận liên vùng Bedrock" | Sản phẩm của AWS; chuyển đổi dự phòng tính sẵn sàng, không nhận thức TTFT |
| DR manifest | "danh sách sao lưu" | Toàn bộ các tệp cần thiết để khôi phục dịch vụ — không chỉ riêng trọng số |
| Data residency | "ranh giới GDPR" | Ràng buộc pháp lý quy định vùng địa lý nào được phép tiếp cận dữ liệu người dùng |
| RTT | "thời gian khứ hồi mạng" | Độ trễ mạng hai chiều; ~75 ms giữa Mỹ-Âu, ~220 ms giữa Mỹ-Châu Á Thái Bình Dương |
| LLM-aware LB | "bộ cân bằng tải nhận biết LLM" | Bộ định tuyến nhận biết bộ nhớ đệm được đóng gói thành một phân khúc sản phẩm |

## Tài liệu đọc thêm

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference) — Suy luận trên môi trường đa đám mây và đa vùng.
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — Tái sử dụng KV-cache liên vùng với số hạng tối ưu độ trễ mạng.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing) — Tính cục bộ của bộ nhớ đệm và quyền cư trú dữ liệu.
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — Tài liệu chính thức về chuyển đổi dự phòng tính sẵn sàng.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — Mã nguồn của bộ định tuyến nhận biết bộ nhớ đệm trong vLLM.
