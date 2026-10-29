# Tự Động Co Cụm GPU Trên Kubernetes — Karpenter, KAI Scheduler, Lập Lịch Băng Nhóm (Gang Scheduling)

> Ba tầng kiến trúc, chứ không phải một. Karpenter cấp phát nút (nodes) động trong thời gian dưới một phút, nhanh hơn 40% so với Cluster Autoscaler. KAI Scheduler phụ trách lập lịch băng nhóm (gang scheduling), nhận thức cấu trúc liên kết mạng (topology awareness) và hàng đợi phân cấp — ngăn ngừa triệt để cái bẫy phân bổ một phần "7 trên 8" khiến bảy nút phải chờ đợi trong vô vọng và tiêu tốn ngân sách chỉ vì thiếu một GPU duy nhất. Bộ tự động co cụm cấp ứng dụng (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) co cụm dựa trên các tín hiệu đặc thù của suy luận — độ sâu hàng đợi (queue depth), tỷ lệ sử dụng bộ đệm KV (KV cache utilization) — chứ không dựa vào chu kỳ làm việc (duty cycle) của CPU hay DCGM. Cái bẫy kinh điển của HPA là chỉ số `DCGM_FI_DEV_GPU_UTIL` chỉ đo chu kỳ làm việc: 100% có thể tương ứng với 10 yêu cầu hoặc 100 yêu cầu. Hơn nữa, vLLM cấp phát trước bộ nhớ đệm KV nên mức sử dụng bộ nhớ sẽ không bao giờ kích hoạt giảm tài nguyên (scale-down). Bài học này hướng dẫn bạn cách phối hợp cả ba tầng và vô hiệu hóa chính sách mặc định `WhenEmptyOrUnderutilized` của Karpenter vốn có thể đột ngột chấm dứt các tác vụ GPU đang suy luận dở dang.

**Type:** Learn
**Languages:** Python (stdlib, trình mô phỏng tự động co cụm theo độ sâu hàng đợi)
**Prerequisites:** Phase 17 · 02 (Kinh Tế Học Nền Tảng Suy Luận / Inference Platform Economics), Phase 17 · 04 (Kiến Trúc Nhân Phục Vụ / Serving Engine Internals)
**Time:** ~75 phút

## Mục tiêu Học tập

- Vẽ sơ đồ ba tầng tự động co cụm (cấp phát nút, lập lịch băng nhóm, tầng ứng dụng) và gọi tên công cụ tương ứng ở từng tầng.
- Giải thích lý do tại sao chỉ số `DCGM_FI_DEV_GPU_UTIL` là tín hiệu sai lầm cho HPA khi chạy vLLM và nêu hai chỉ số thay thế chuẩn xác (độ sâu hàng đợi, tỷ lệ sử dụng bộ đệm KV).
- Mô tả cơ chế lập lịch băng nhóm (gang scheduling) và lỗi phân bổ tài nguyên cục bộ mà KAI Scheduler ngăn chặn (7 trên 8 GPU ngồi nhàn rỗi).
- Nêu tên chính sách gom nút (consolidation policy) của Karpenter (`WhenEmptyOrUnderutilized`) gây hủy tác vụ GPU đang chạy và chỉ rõ giải pháp thay thế an toàn trong năm 2026.

## Vấn đề Thực tế

Nhóm của bạn triển khai dịch vụ phục vụ LLM trên Kubernetes. Bạn cấu hình Horizontal Pod Autoscaler (HPA) sử dụng `DCGM_FI_DEV_GPU_UTIL` làm chỉ số kích hoạt. Dịch vụ chạm ngưỡng 100% mức sử dụng trong suốt giờ cao điểm. Nhưng HPA không bao giờ tự động mở rộng (scale-up) — nó nghĩ rằng hệ thống đã hoạt động hết công suất. Bạn tự thêm một replica thủ công; độ trễ đến token đầu tiên (TTFT) lập tức giảm xuống. Dù vậy, HPA vẫn không tự động co giãn. Tín hiệu đo lường đang đánh lừa bạn.

Mặt khác, bạn dùng Cluster Autoscaler để co giãn các nút máy chủ vật lý. Một prompt 1 triệu token ập đến lúc 2 giờ sáng; cụm mất 3 phút để khởi tạo một nút mới, và yêu cầu của người dùng bị quá thời gian chờ (timeout).

Một vấn đề khác nữa: bạn triển khai mô hình 70B yêu cầu 8 GPU trải rộng trên 2 nút. Cụm hiện có 7 GPU đang trống nằm rải rác trên 3 nút và cần thêm 1 GPU nữa. Cluster Autoscaler bắt đầu khởi tạo một nút mới cho đúng 1 GPU còn thiếu. 7 GPU đã được giữ chỗ phải ngồi không trong suốt 4 phút, đốt cháy chi phí vô ích trong khi chờ Kubernetes chuẩn bị xong chiếc GPU cuối cùng.

Ba tầng khác nhau dẫn đến ba phương thức thất bại riêng biệt. Tự động co cụm nhận thức GPU trong năm 2026 không phải là "bật HPA lên". Đó là sự kết hợp chuẩn xác giữa cấp phát nút, lập lịch băng nhóm và co giãn theo tín hiệu ứng dụng.

## Khái niệm

### Tầng 1 — cấp phát nút (Karpenter)

Karpenter liên tục theo dõi các pod đang ở trạng thái chờ (pending) và cấp phát nút mới chỉ trong ~45-60 giây (Cluster Autoscaler thường mất 90-120 giây đối với các nút gắn GPU). Nó lựa chọn loại phiên bản máy chủ (instance types) một cách linh hoạt theo ràng buộc của `NodePool` — nếu pod của bạn cần 8 card H100 và cụm không có nút nào thỏa mãn, Karpenter sẽ trực tiếp khởi tạo một nút đúng yêu cầu thay vì co giãn một nhóm nút cố định có sẵn.

**Cái bẫy gom nút (consolidation trap)**: Cấu hình mặc định `consolidationPolicy: WhenEmptyOrUnderutilized` của Karpenter cực kỳ nguy hiểm cho các nhóm máy chủ GPU. Nó sẵn sàng chấm dứt hoạt động của một nút GPU đang chạy để di chuyển các pod sang một máy chủ rẻ hơn có kích thước vừa vặn hơn. Đối với khối lượng công việc suy luận, điều này đồng nghĩa với việc đẩy văng các yêu cầu đang xử lý dở và phải nạp lại mô hình 70B từ đầu trên nút mới. Thiệt hại là mất nhiều phút dung lượng phục vụ kèm hàng loạt lỗi trả về cho người dùng.

Cấu hình an toàn cho các cụm GPU:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

Cấu hình này cho phép Karpenter chỉ thu hồi các nút thực sự trống rỗng sau một giờ không hoạt động và tuyệt đối không bao giờ trục xuất một tác vụ đang chạy.

### Tầng 2 — lập lịch băng nhóm (KAI Scheduler)

KAI Scheduler (ban đầu mang tên dự án "Karp" sau đó đổi tên) giải quyết những vấn đề mà bộ lập lịch mặc định kube-scheduler không làm được:

**Lập lịch băng nhóm (Gang scheduling)** — lập lịch theo nguyên tắc "tất cả hoặc không" (all-or-nothing). Một nhóm pod suy luận phân tán đòi hỏi 8 GPU thì hoặc cả 8 pod cùng khởi động đồng thời, hoặc không pod nào được phép chạy. Nếu thiếu cơ chế này, bạn sẽ rơi vào bẫy phân bổ một phần (partial-allocation trap): 7 trên 8 pod khởi chạy trước, chờ đợi vô thời hạn và thiêu đốt ngân sách vô ích.

**Nhận thức cấu trúc mạng (Topology awareness)** — hiểu rõ các GPU nào chia sẻ liên kết NVLink, GPU nào nằm trên cùng một giá máy chủ (rack), và GPU nào kết nối thông qua mạng InfiniBand. Từ đó phân bổ pod tối ưu. Một khối lượng công việc xử lý song song tensor (tensor-parallel) như DeepSeek-V3 67B bắt buộc phải nằm trong cùng một miền NVLink; KAI Scheduler đảm bảo tuân thủ nghiêm ngặt điều này.

**Hàng đợi phân cấp (Hierarchical queues)** — cho phép nhiều nhóm cùng cạnh tranh trên một cụm tài nguyên GPU dựa trên quyền ưu tiên và hạn ngạch (quota). Tình trạng thiếu hụt tài nguyên sản xuất của Đội A chỉ có thể chiếm quyền (preempt) tác vụ huấn luyện của Đội B nếu quy tắc ưu tiên cho phép.

KAI được triển khai song song với kube-scheduler dưới dạng một bộ lập lịch thứ cấp; bạn chỉ cần gắn annotation vào khối lượng công việc để chỉ định sử dụng. Cả ngăn xếp sản xuất của Ray và vLLM đều tích hợp sẵn bộ lập lịch này.

### Tầng 3 — tín hiệu cấp ứng dụng

**Cái bẫy HPA**: Chỉ số `DCGM_FI_DEV_GPU_UTIL` là một chỉ số chu kỳ làm việc (duty-cycle) — nó chỉ đo xem GPU có thực hiện phép tính nào tại mỗi chu kỳ lấy mẫu hay không. Mức sử dụng 100% có thể tương ứng với 10 yêu cầu đồng thời hoặc 100 yêu cầu; dù thế nào thì GPU vẫn đang bận rộn. Co giãn dựa trên chu kỳ làm việc giống như việc lái xe trong tình trạng bị bịt mắt.

Tệ hơn, vLLM và các nhân phục vụ tương tự luôn cấp phát trước toàn bộ bộ nhớ đệm KV (lên tới ngưỡng cấu hình `--gpu-memory-utilization`). Mức chiếm dụng bộ nhớ luôn neo ở mức xấp xỉ 90% ngay cả khi chỉ phục vụ đúng 1 yêu cầu. HPA dựa trên bộ nhớ sẽ không bao giờ kích hoạt hạ quy mô.

**Các tín hiệu thay thế chuẩn mực năm 2026**:

- Độ sâu hàng đợi (queue depth: số lượng yêu cầu đang chờ xử lý bước tiền điền / prefill).
- Tỷ lệ sử dụng bộ đệm KV (KV cache utilization: tỷ lệ khối nhớ thực tế đang được cấp phát cho các chuỗi hoạt động).
- Độ trễ TTFT tại phân vị P99 trên từng replica (tín hiệu trực tiếp phản ánh SLA của bạn).
- Sản lượng hữu ích (Goodput: số yêu cầu đáp ứng trọn vẹn toàn bộ các mục tiêu SLO trong mỗi giây).

NVIDIA Dynamo Planner và llm-d Workload Variant Autoscaler tiếp nhận các tín hiệu này để quyết định co giãn số lượng replica. Chúng thay thế hoàn toàn vai trò của HPA trong bài toán phục vụ LLM.

### Bảng phân công quyết định co giãn

| Quyết định quy mô | Công cụ đảm nhiệm |
|-------------------|-------------------|
| Thêm/bớt nút máy chủ | Karpenter |
| Lập lịch tác vụ đa GPU | KAI Scheduler |
| Thêm/bớt số lượng replica | Dynamo Planner / llm-d WVA (hoặc HPA tùy biến theo độ sâu hàng đợi) |
| Chọn chủng loại GPU | Karpenter NodePool |
| Chiếm quyền ưu tiên thấp | Hàng đợi của KAI Scheduler |

### Tách rời tiền điền/giải mã làm phức tạp hóa mọi thứ

Khi bạn áp dụng kiến trúc suy luận tách rời tiền điền và giải mã (disaggregated prefill/decode - Phase 17 · 17), bạn sẽ có hai nhóm pod khác nhau với các điều kiện kích hoạt co giãn hoàn toàn riêng biệt: pod tiền điền co giãn theo độ sâu hàng đợi, trong khi pod giải mã co giãn theo áp lực bộ đệm KV. Dự án llm-d phơi bày các vai trò này dưới dạng các `Services` riêng biệt với cấu hình HPA độc lập. Tuyệt đối không đặt một HPA chung chung duy nhất lên trước cả hai nhóm.

### Khởi động nguội là vấn đề sống còn

Giảm thiểu độ trễ khởi động nguội (cold-start mitigation - Phase 17 · 10) là nơi mà thời gian cấp phát nút bắt đầu gây ảnh hưởng trực tiếp tới người dùng cuối. Karpenter mất 45-60 giây để làm ấm nút, cộng thêm thời gian tải mô hình 20GB và khởi tạo nhân phục vụ, khiến một yêu cầu từ trạng thái không có tài nguyên (from-zero) mất tới 2-5 phút. Hãy luôn duy trì một cụm nút ấm tối thiểu (`min_workers=1`) cho các luồng quan trọng về mặt SLO, hoặc ứng dụng cơ chế tạo điểm phục hồi (checkpointing) kiểu Modal ở tầng ứng dụng.

### Những con số bạn cần ghi nhớ

- Thời gian Karpenter cấp phát nút: ~45-60 giây so với Cluster Autoscaler ~90-120 giây (trên các nút GPU).
- KAI Scheduler loại bỏ hoàn toàn lãng phí do phân bổ một phần — cái bẫy 7 trên 8.
- Sử dụng `DCGM_FI_DEV_GPU_UTIL` làm tín hiệu HPA: hoàn toàn sai lầm; hãy chuyển sang độ sâu hàng đợi hoặc tỷ lệ sử dụng bộ đệm KV.
- Cấu hình Karpenter `WhenEmptyOrUnderutilized`: sẽ hủy ngang các tác vụ GPU đang chạy. Hãy sử dụng `WhenEmpty + consolidateAfter: 1h` cho môi trường suy luận.

```figure
autoscaling
```

## Ứng Dụng Thực Tế (Use It)

Tệp `code/main.py` mô phỏng bộ tự động co cụm ba tầng trên một khối lượng công việc GPU có lưu lượng bùng phát đột biến. So sánh hiệu quả giữa HPA ngây thơ (dựa trên chu kỳ làm việc), HPA theo độ sâu hàng đợi, và cơ chế lập lịch băng nhóm KAI. Báo cáo số yêu cầu bị rớt (unmet requests), số phút GPU nhàn rỗi, và điểm số tổng hợp toàn diện.

## Bàn Giao Thành Phẩm (Ship It)

Bài học này tạo ra tệp artifact `outputs/skill-gpu-autoscaler-plan.md`. Dựa trên cấu trúc mạng cụm máy chủ, hình thái khối lượng công việc và cam kết SLO, công cụ sẽ thiết kế một bản kế hoạch co giãn tự động ba tầng hoàn chỉnh.

## Bài Tập Thực Hành

1. Chạy `code/main.py`. Dưới một khối lượng công việc bùng phát đột ngột, HPA dựa trên chu kỳ làm việc ngây thơ làm rớt bao nhiêu yêu cầu mà HPA theo độ sâu hàng đợi xử lý được? Sự khác biệt này bắt nguồn từ đâu?
2. Thiết kế một `NodePool` của Karpenter cho một cụm máy chủ phục vụ Llama 3.3 70B FP8 trên H100 SXM5. Xác định rõ `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, và một taint để ngăn các tác vụ phi GPU không chen vào các nút này.
3. Đội ngũ của bạn báo cáo rằng các bản triển khai bị kẹt ở trạng thái Pending vì "GPU còn trống nhưng pod không được lập lịch". Hãy chẩn đoán — vấn đề nằm ở Karpenter, kube-scheduler hay KAI Scheduler? Chỉ số nào giúp xác nhận kết luận?
4. Chọn một tín hiệu để tự động co giãn các pod tiền điền (prefill) và một tín hiệu khác cho các pod giải mã (decode) trong kiến trúc tách rời. Biện minh cho cả hai lựa chọn.
5. Tính toán tổn thất chi phí của cái bẫy gom nút `WhenEmptyOrUnderutilized` trên một dịch vụ sản xuất chạy 24/7 có trung bình 60 sự cố làm rớt yêu cầu mỗi ngày tại ngưỡng P99 TTFT > 10 giây.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật thực tế |
|-----------|----------------------|--------------------------|
| Karpenter | "Trình cấp phát nút" | Bộ tự động co giãn nút cho Kubernetes; cấp phát trong thời gian dưới một phút |
| Cluster Autoscaler | "Bộ co giãn cũ" | Công cụ co giãn nút tiền nhiệm của Kubernetes; chậm hơn, hoạt động dựa trên nhóm nút cố định |
| KAI Scheduler | "Bộ lập lịch GPU" | Bộ lập lịch thứ cấp hỗ trợ gang scheduling + nhận thức tô-pô mạng + hàng đợi ưu tiên |
| Lập lịch băng nhóm (Gang scheduling) | "Tất cả hoặc không" | Lập lịch đồng thời nguyên tử cho N pod hoặc hoãn toàn bộ |
| Nhận thức cấu trúc mạng (Topology awareness) | "Nhận thức giá rack" | Phân bổ pod dựa trên liên kết NVLink / InfiniBand / vị trí tủ rack |
| `DCGM_FI_DEV_GPU_UTIL` | "Mức sử dụng GPU" | Thước đo chu kỳ làm việc của GPU; KHÔNG PHẢI tín hiệu co giãn phù hợp cho LLM |
| Độ sâu hàng đợi (Queue depth) | "Số yêu cầu đang chờ" | Tín hiệu HPA chuẩn xác cho việc co giãn các tác vụ bị nghẽn ở pha tiền điền |
| Tỷ lệ sử dụng bộ đệm KV | "Áp lực bộ nhớ" | Tín hiệu HPA chuẩn xác cho việc co giãn các tác vụ bị nghẽn ở pha giải mã |
| Gom cụm (Consolidation) | "Karpenter gom nút" | Hành động thu hồi nút để chuyển pod sang một cấu hình máy chủ rẻ hơn |
| `WhenEmpty + 1h` | "Gom nút an toàn" | Chính sách không bao giờ trục xuất các tác vụ GPU đang vận hành |

## Tài Liệu Đọc Thêm

- [GitHub của KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) — Tài liệu thiết kế và các ví dụ cấu hình chi tiết.
- [Kiểm soát Gián đoạn của Karpenter](https://karpenter.sh/docs/concepts/disruption/) — Ngữ nghĩa chính sách gom cụm và cấu hình mặc định an toàn cho GPU.
- [NVIDIA — Suy luận LLM Tách rời trên Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Các tín hiệu co giãn của Dynamo Planner.
- [Tài liệu Ray — KAI Scheduler cho RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Mẫu thiết kế tích hợp Ray.
- [Thực tiễn Tốt nhất về Tính toán và Co giãn trên AWS EKS](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — Hướng dẫn cụ thể cho Kubernetes được quản lý.
- [GitHub của llm-d](https://github.com/llm-d/llm-d) — Thiết kế của Workload Variant Autoscaler.
