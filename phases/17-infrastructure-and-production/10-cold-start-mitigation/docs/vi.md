# Giảm thiểu khởi động nguội cho LLM không máy chủ

> Một mô hình lớn mất hàng phút để chuyển từ trạng thái nguội hoàn toàn sang trạng thái sẵn sàng phục vụ: một mô hình 70B trên một nút máy chủ mới cần khoảng 3 đến 8 phút trước khi có thể xuất ra token đầu tiên. Trong thế giới không máy chủ (serverless) thực thụ, khoảng thời gian đó không còn là làm nóng (warm-up) nữa — mà thực chất là một sự cố gián đoạn dịch vụ (outage). Các kỹ thuật giảm thiểu vận hành ở năm tầng kiến trúc: hình ảnh nút máy chủ nạp sẵn (pre-seeded node images như Bottlerocket trên AWS với kiến trúc ổ đĩa kép), luồng truyền mô hình (model streaming qua NVIDIA Run:ai Model Streamer, tích hợp sẵn trong vLLM), ảnh chụp nhanh bộ nhớ GPU (GPU memory snapshots như các điểm kiểm tra / checkpoints của Modal, giúp khởi động lại nhanh hơn tới 10 lần), nhóm tài nguyên ấm (warm pools với `min_workers=1`), nạp dữ liệu phân tầng (tiered loading như đường ống NVMe→DRAM→HBM của ServerlessLLM, giảm độ trễ từ 10 đến 200 lần), cùng với một mẫu hình bổ trợ thứ sáu: di chuyển trực tiếp (live migration) truyền các token đầu vào (vài KB) thay vì truyền toàn bộ KV cache (hàng GB). Modal công bố mốc sàn khởi động nguội chỉ từ 2 đến 4 giây; Baseten đạt mặc định 5–10 giây, và dưới một giây khi có làm ấm trước. Bài học này hướng dẫn bạn cách đo lường, lập ngân sách và kết hợp đồng thời năm tầng giảm thiểu này.

**Phân loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng đường dẫn khởi động nguội)
**Điều kiện tiên quyết:** Phase 17 · 02 (Kinh tế học nền tảng suy luận), Phase 17 · 03 (Tự động mở rộng quy mô GPU)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Liệt kê năm tầng giảm thiểu khởi động nguội và nêu tên một công cụ hoặc mẫu thiết kế tại mỗi tầng.
- Tính toán tổng thời gian khởi động nguội dưới dạng tổng của: (khởi tạo nút) + (tải trọng số) + (nạp trọng số vào HBM) + (khởi tạo động cơ phục vụ) cho một mô hình 70B.
- Giải thích lý do cơ chế di chuyển trực tiếp (live migration) truyền token đầu vào (KB) thay vì truyền KV cache (GB) và cái giá phải trả của phương pháp này (tính toán lại / recomputation).
- Nêu rõ bài toán đánh đổi của nhóm tài nguyên ấm (trả tiền cho GPU nhàn rỗi hay chấp nhận độ trễ khởi động nguội ở đuôi phân phối) cùng ngưỡng SLA bắt buộc phải cấu hình `min_workers > 0`.

## Vấn đề

Điểm cuối LLM không máy chủ (serverless LLM endpoint) của bạn thu hẹp quy mô về 0 (scale-to-zero) vào ban đêm. Đúng 8 giờ sáng, lưu lượng truy cập tăng đột biến. Yêu cầu đầu tiên phải xếp hàng chờ đợi qua các bước:

1. Karpenter cấp phát một nút GPU mới: mất 45–60 giây.
2. Vùng chứa (container) tải hình ảnh nặng 30 GB chứa các trọng số: mất 120–300 giây.
3. Động cơ phục vụ nạp trọng số vào bộ nhớ HBM: mất 45–120 giây tùy vào kích thước mô hình và tốc độ ổ lưu trữ.
4. vLLM hoặc TRT-LLM khởi tạo đồ thị CUDA (CUDA graphs), vùng đệm KV cache, bộ tách từ (tokenizer): mất 10–30 giây.

Tổng cộng: 220–510 giây (khoảng 3 đến 8 phút) trôi qua trước khi token đầu tiên được tạo ra. Trong khi đó, cam kết SLA của bạn là 2 giây. Bạn giải quyết bằng cách duy trì một nhóm ấm (`min_workers=1`) và vấn đề dường như biến mất — nhưng giờ đây bạn phải chi trả chi phí cho một GPU chạy nhàn rỗi 24/7. Nếu dịch vụ của bạn có 5 sản phẩm khác nhau, mỗi sản phẩm giữ một bản sao ấm, điều đó đồng nghĩa với 5 × 24 × 30 = 3.600 giờ GPU mỗi tháng dù không có lấy một người dùng gọi đến.

Kỹ thuật giảm thiểu khởi động nguội chính là phương pháp giúp bạn giữ vững lợi ích kinh tế của kiến trúc không máy chủ trong khi vẫn tiệm cận được độ trễ của hệ thống luôn bật sẵn (always-on).

## Khái niệm

### Tầng 1 — Hình ảnh nút nạp sẵn: Bottlerocket

Trên AWS, kiến trúc hai ổ đĩa (dual-volume architecture) của Bottlerocket tách biệt hệ điều hành khỏi dữ liệu. Bạn tạo một ảnh chụp nhanh (snapshot) của ổ dữ liệu với hình ảnh container đã được tải sẵn từ trước; sau đó tham chiếu mã định danh snapshot ID đó trong `EC2NodeClass`. Các nút mới khởi động sẽ có ngay trọng số mô hình nằm sẵn trên ổ NVMe cục bộ — bước 2 và một phần bước 3 hoàn toàn biến mất. Cơ chế này hoạt động tự nhiên với Karpenter. Mức tiết kiệm điển hình: từ 2 đến 4 phút cho mỗi lần khởi động nguội của mô hình lớn.

Giải pháp tương đương trên GCP: sử dụng hình ảnh máy ảo tùy biến (custom VM images) tích hợp sẵn các lớp container. Trên Azure: sử dụng ảnh chụp nhanh đĩa được quản lý (managed disk snapshots) với cùng mô hình thiết kế.

### Tầng 2 — Luồng truyền mô hình: Run:ai Model Streamer

Thay vì tải toàn bộ tệp trọng số khổng lồ trước khi trả lời yêu cầu đầu tiên, kỹ thuật này truyền trực tiếp các trọng số vào bộ nhớ GPU theo từng tầng và bắt đầu tính toán ngay khi khối biến đổi (transformer block) đầu tiên đã nằm sẵn trong bộ nhớ. Bộ truyền luồng mô hình NVIDIA Run:ai Model Streamer được tích hợp trực tiếp trong vLLM năm 2026. Nó tương thích với S3, GCS và ổ NVMe cục bộ. Cơ chế này cắt giảm khoảng một nửa thời gian nạp trọng số cho các mô hình lớn bằng cách thực hiện gối đầu (overlapping) giữa thao tác I/O với quá trình thiết lập tính toán.

### Tầng 3 — Ảnh chụp nhanh bộ nhớ GPU: Modal

Modal chụp một điểm kiểm tra (checkpoint) toàn bộ trạng thái của GPU (trọng số, đồ thị CUDA, vùng nhớ KV cache) ngay sau lần nạp đầu tiên. Các lần khởi động tiếp theo chỉ cần giải tuần tự hóa (deserialize) trực tiếp vào HBM — nhanh hơn tới 10 lần so với việc khởi tạo lại từ đầu. Đây là giải pháp tiệm cận nhất với mục tiêu "khởi động một GPU ấm trong vòng 2 giây". Điểm đánh đổi: các snapshot này gắn chặt với từng cấu trúc liên kết GPU (GPU topology) cụ thể, nên nếu Karpenter chuyển sang một loại SKU phần cứng khác, bạn sẽ phải tạo lại snapshot mới.

### Tầng 4 — Nhóm tài nguyên ấm: min_workers=1

Biện pháp giảm thiểu đơn giản nhất: luôn duy trì một bản sao (replica) ở trạng thái sẵn sàng. Chi phí bỏ ra là đơn giá theo giờ của một GPU chạy 24/7. Phép tính này khá tốn kém với các mô hình nhỏ (bạn trả 0,85–1,50 USD/giờ chỉ để tránh thời gian khởi động nguội 30 giây), nhưng lại rất hợp lý với các mô hình lớn (trả 4 USD/giờ để tránh 5 phút chờ đợi tê liệt). Ngưỡng SLA buộc phải dùng nhóm tài nguyên ấm: thường là khi yêu cầu TTFT P99 < 60 giây đối với mô hình từ 70B trở lên.

### Tầng 5 — Nạp dữ liệu phân tầng: ServerlessLLM

ServerlessLLM tổ chức hệ thống lưu trữ theo cấu trúc phân tầng: NVMe (tốc độ cao nhưng dung lượng lớn), DRAM (tốc độ trung bình, nằm ở tầng giữa), HBM (dung lượng giới hạn nhưng truy xuất tức thì). Các trọng số được nạp trước vào bộ nhớ DRAM; khi có yêu cầu, chúng được đẩy ngay vào HBM theo nhu cầu. Bài báo khoa học công bố mức giảm độ trễ từ 10 đến 200 lần khi nạp nguội so với cách đọc tuần tự từ đĩa vào HBM truyền thống. Mức độ áp dụng thực tế đang trong giai đoạn đầu nhưng đã có các bản tích hợp sẵn với vLLM.

### Tầng 6 — Di chuyển trực tiếp: Mẫu hình bổ trợ

Khi một nút máy chủ gặp sự cố hoặc bị thu hồi (thu hồi phiên bản spot, nút bị xả tải / node drain), cách tiếp cận truyền thống là khởi động nguội một bản sao khác và làm rỗng hàng đợi yêu cầu. Cơ chế di chuyển trực tiếp (live migration) sẽ chuyển các token đầu vào (chỉ vài kilobyte) sang một nút đích đã nạp sẵn mô hình và thực hiện tính toán lại KV cache ngay trên nút đích đó. Việc tính toán lại trên GPU đích rẻ và nhanh hơn rất nhiều so với việc truyền hàng chục GB KV cache qua đường truyền mạng. Mẫu hình này cực kỳ hữu ích cho các hệ thống triển khai phân tách (disaggregated deployments).

### Bài toán kinh tế của nhóm tài nguyên ấm

Đối với một dịch vụ có cam kết SLA TTFT P99 là 2 giây, câu hỏi không phải là "có nên dùng warm pool hay không" mà là "cần bao nhiêu bản sao ấm, và luồng dữ liệu nào được quyền sử dụng chúng":

- Các luồng tương tác giá trị cao (trò chuyện trực tiếp, trợ lý ảo giọng nói): Cấu hình `min_workers=1-2`.
- Các luồng xử lý lô chạy nền (phân loại tài liệu ban đêm): Chấp nhận thu nhỏ về 0 (scale-to-zero), dung sai thời gian khởi động nguội từ 5 đến 10 phút.
- Tầng khách hàng cao cấp (premium): Cấp phát `min_workers` riêng theo từng người thuê (tenant) với tài nguyên dành riêng.

### Đo lường trước khi tối ưu hóa

Cấu trúc thời gian khởi động nguội điển hình của mô hình 70B trên một nút máy chủ mới (mang tính minh họa):

| Giai đoạn | Thời gian | Biện pháp giảm thiểu |
|---|---|---|
| Khởi tạo nút (Node provision) | 50s | Bottlerocket + hình ảnh nạp sẵn, warm pool |
| Tải hình ảnh container (Image pull) | 180s | Ổ dữ liệu nạp sẵn snapshot (loại bỏ hoàn toàn) |
| Chuyển trọng số vào HBM | 75s | Model streamer (giảm một nửa); GPU snapshot (loại bỏ) |
| Khởi tạo động cơ (Engine init) | 20s | Bộ nhớ đệm đồ thị CUDA bền vững (persistent CUDA graphs) |
| Lượt lan truyền xuôi đầu tiên | 3s | Độ trễ vật lý tối thiểu cố hữu |
| **Tổng thời gian khởi động nguội** | **328s** | |
| **Tổng thời gian sau khi tối ưu** | **~15s** | Giảm 22 lần |

### Các con số cần ghi nhớ

- Thời gian khởi động nguội trên Modal: 2–4 giây (nhờ cơ chế GPU snapshots).
- Thời gian khởi động nguội mặc định trên Baseten: 5–10 giây; dưới 1 giây nếu có làm ấm trước.
- Khởi động nguội mô hình 70B thô chưa tối ưu: 3–8 phút.
- Run:ai Model Streamer: tăng tốc quá trình nạp trọng số khoảng ~2 lần.
- Nạp phân tầng ServerlessLLM: giảm độ trễ từ 10 đến 200 lần (theo công bố trong bài báo khoa học).

```figure
cold-start-pipeline
```

## Sử dụng

Tệp `code/main.py` mô phỏng hành trình khởi động nguội khi có và không có từng biện pháp giảm thiểu. Chương trình tính toán tổng thời gian khởi động, chi phí duy trì nhóm ấm, và xác định tốc độ yêu cầu hòa vốn (break-even request rate) mà tại đó việc duy trì bản sao ấm sẽ mang lại hiệu quả kinh tế cao hơn việc để xảy ra khởi động nguội.

## Phát hành

Bài học này cung cấp tệp `outputs/skill-cold-start-planner.md`. Dựa trên yêu cầu SLA, kích thước mô hình và đồ thị lưu lượng truy cập thực tế, công cụ này sẽ tư vấn chính xác những tầng giảm thiểu nào cần được kết hợp triển khai.

## Bài tập

1. Chạy `code/main.py`. Tính toán tốc độ yêu cầu hòa vốn mà vượt qua mức đó, việc duy trì một bản sao ấm sẽ rẻ hơn so với việc gánh chịu chi phí phạt khi các yêu cầu bị hủy bỏ do vi phạm ngưỡng SLO khởi động nguội.
2. Bạn triển khai một mô hình 13B với cam kết SLA TTFT P99 là 3 giây. Hãy chọn ngăn xếp giảm thiểu tối giản nhất (ít tầng nhất) nhưng vẫn đảm bảo đạt được mục tiêu này.
3. Kỹ thuật nạp sẵn Bottlerocket loại bỏ hoàn toàn bước tải hình ảnh container nhưng trọng số vẫn phải nạp từ snapshot vào HBM. Hãy tính thời gian thực tế cần thiết cho mô hình 70B nếu ổ NVMe gắn snapshot đạt tốc độ đọc 7 GB/s.
4. Nhà cung cấp dịch vụ không máy chủ của bạn đề xuất dùng tính năng chụp nhanh GPU (Modal) nhưng đội ngũ bảo mật từ chối với lý do "ảnh chụp nhanh làm rò rỉ dữ liệu định danh cá nhân (PII)". Hãy đưa ra lập luận cho cả hai phía — đâu là rủi ro thực tế, và đâu là biện pháp khắc phục (ảnh chụp nhanh tạm thời, mã hóa dữ liệu, cô lập không gian tên)?
5. Thiết kế một chính sách phân tầng cho nhóm tài nguyên ấm: cần bao nhiêu bản sao ấm cho người dùng trả phí, người dùng dùng thử, và các tác vụ xử lý lô? Trình bày chi tiết công thức toán học tính toán.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Cold start | "khởi động nguội" | Thời gian từ lúc gửi yêu cầu đến khi ra token đầu tiên trên một bản sao mới |
| Warm pool | "nhóm tài nguyên ấm" | Cấu hình `min_workers >= 1` để luôn duy trì ít nhất một bản sao sẵn sàng |
| Pre-seeded image | "ảnh đĩa nạp sẵn" | Hình ảnh nút máy chủ đã chứa sẵn các trọng số mô hình của container |
| Bottlerocket | "hệ điều hành nút AWS" | Hệ điều hành tối ưu cho container của AWS hỗ trợ kiến trúc snapshot hai ổ đĩa |
| Model streamer | "nạp mô hình dạng luồng" | Thực hiện gối đầu giữa thao tác I/O nạp trọng số và bước chuẩn bị tính toán |
| GPU snapshot | "chụp ảnh nhanh vào HBM" | Tuần tự hóa trạng thái GPU sau khi nạp; giải tuần tự hóa tức thì khi khởi động lại |
| Tiered loading | "nạp phân tầng" | Cấu trúc phân cấp lưu trữ NVMe + DRAM + HBM; nạp dữ liệu theo nhu cầu |
| Live migration | "di chuyển trực tiếp" | Chuyển token đầu vào (KB), tính toán lại KV cache ngay trên nút đích |
| `min_workers` | "bản sao ấm tối thiểu" | Số lượng cá thể tối thiểu luôn duy trì chạy trong kiến trúc không máy chủ |
| Scale-to-zero | "thu nhỏ về không" | Không tốn chi phí khi nhàn rỗi; chấp nhận toàn bộ thời gian khởi động nguội |

## Tài liệu đọc thêm

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Các bài đo điểm chuẩn chính thức và kiến trúc trạm kiểm soát của Modal.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — Mô hình snapshot ổ đĩa dữ liệu nạp sẵn trên AWS.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — Kỹ thuật gối đầu nạp trọng số với thiết lập môi trường tính toán.
- [Baseten — Cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts) — Cẩm nang làm ấm tài nguyên trước khi phục vụ.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — Thiết kế kiến trúc nạp dữ liệu phân tầng.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Di chuyển trực tiếp cho các hệ thống triển khai phân tách.
