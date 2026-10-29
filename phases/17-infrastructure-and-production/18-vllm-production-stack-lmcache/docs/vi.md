# Ngăn xếp phục vụ sản xuất — Offload KV và định tuyến nhận biết cache (Production Serving Stack: KV Offloading and Cache-Aware Routing)

> Một ngăn xếp phục vụ sản xuất nối router, engine và khả năng quan sát (observability) thành một triển khai Kubernetes duy nhất — và coi KV cache là một tài nguyên có thể rời khỏi GPU. Offload KV (KV offloading) trích KV cache ra khỏi bộ nhớ GPU và tái sử dụng nó giữa các truy vấn và các engine (CPU DRAM, rồi đĩa/Ceph). production-stack của vLLM là triển khai tham chiếu; LMCache là lớp offload. KV Offloading Connector của vLLM 0.11.0 (tháng 1/2026) làm điều này bất đồng bộ và cắm-rút được qua Connector API (v0.9.0+). Đường offload thường được ẩn khỏi đường xử lý yêu cầu, dù cache miss và việc nâng cấp (promotion) có thể làm tăng độ trễ đầu-cuối. LMCache có giá trị ngay cả khi không có tiền tố dùng chung — khi một GPU hết ô KV, các yêu cầu bị chiếm quyền (preempt) có thể được khôi phục từ CPU thay vì tính lại prefill. Các benchmark đã công bố trên 16x H100 (80GB HBM) qua 4 máy a3-highgpu-4g: khi KV cache vượt HBM, cả offload CPU gốc lẫn LMCache đều cải thiện thông lượng đáng kể; ở mức KV thấp, mọi cấu hình đều bằng đường cơ sở với chi phí phụ nhỏ.

**Loại bài:** Học
**Ngôn ngữ:** Python (stdlib, trình mô phỏng đồ chơi KV tràn)
**Điều kiện tiên quyết:** Phase 17 · 04 (Nội bộ engine phục vụ), Phase 17 · 06 (SGLang/RadixAttention)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Vẽ sơ đồ các lớp của vLLM production-stack: router, engine, offload KV, khả năng quan sát.
- Giải thích KV Offloading Connector API (v0.9.0+) và cách đường bất đồng bộ của 0.11.0 che giấu độ trễ offload.
- Định lượng khi nào LMCache CPU-DRAM có ích (KV > HBM) so với khi nó thêm chi phí phụ (KV đủ nhỏ để nằm gọn trong HBM).
- Chọn giữa offload CPU gốc của vLLM và connector LMCache dựa trên ràng buộc triển khai.

## Vấn đề

Hệ thống vLLM của bạn cho thấy GPU ở mức 100% HBM với các sự kiện chiếm quyền (preemption) mỗi khi mức đồng thời tăng. Các yêu cầu bị trục xuất, xếp hàng lại, và bạn prefill lại cùng một prompt 2K token bốn lần trong một phút. Sức tính toán GPU bị tiêu vào các prefill dư thừa; goodput thấp hơn nhiều so với thông lượng thô.

Thêm GPU làm chi phí tăng tuyến tính. Thêm HBM là không thể. Nhưng CPU DRAM thì rẻ — một socket có 512 GB+ với độ trễ tệ hơn HBM nhiều bậc độ lớn nhưng vẫn ổn cho KV cache "ấm tạm thời".

LMCache trích KV cache ra CPU DRAM để các yêu cầu bị chiếm quyền phục hồi nhanh, và các tiền tố lặp lại giữa các engine chia sẻ cache mà không cần mỗi engine prefill lại.

## Khái niệm

### vLLM production-stack

`github.com/vllm-project/production-stack` là triển khai Kubernetes tham chiếu:

- **Router** — nhận biết cache (Phase 17 · 11). Tiêu thụ các sự kiện KV.
- **Engine** — các worker vLLM. Một cho mỗi GPU hoặc mỗi nhóm TP/PP.
- **Offload KV cache** — triển khai LMCache hoặc connector gốc.
- **Khả năng quan sát** — scrape Prometheus, dashboard Grafana, trace OTel.
- **Mặt phẳng điều khiển (control plane)** — khám phá dịch vụ, cấu hình, cập nhật cuốn chiếu.

Được phát hành dưới dạng Helm chart + operator.

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 giới thiệu Connector API cho các backend KV cache cắm-rút. Engine của bạn offload các khối (block) sang connector; connector lưu chúng (RAM, đĩa, lưu trữ đối tượng, LMCache). Khi yêu cầu cần một khối, connector tải nó trở lại.

vLLM 0.11.0 (tháng 1/2026) bổ sung đường offload bất đồng bộ — offload có thể diễn ra nền để engine không bị chặn trong trường hợp thông thường. Độ trễ và thông lượng đầu-cuối vẫn phụ thuộc vào hình dạng tải, tỷ lệ trúng KV cache và áp lực hệ thống; chính ghi chú của vLLM nêu rằng offload bằng kernel tùy biến có thể làm giảm thông lượng ở tỷ lệ trúng thấp và rằng lập lịch bất đồng bộ có các vấn đề tương tác đã biết với giải mã suy đoán (speculative decoding).

### Offload CPU gốc so với LMCache

**Offload CPU gốc của vLLM**: cục bộ theo engine. Lưu các khối KV trong RAM máy chủ. Triển khai nhanh, không có bước nhảy mạng. Không vượt qua ranh giới giữa các engine.

**Connector LMCache**: quy mô cụm. Lưu các khối trong một máy chủ LMCache dùng chung (CPU DRAM + tầng Ceph/S3). Các khối truy cập được từ bất kỳ engine nào. Đã công bố benchmark 16x H100.

Chọn gốc khi một engine đơn có áp lực HBM. Chọn LMCache khi nhiều engine dùng chung tiền tố (RAG với system prompt chung, đa người thuê (multi-tenant) với mẫu dùng chung).

### Hành vi benchmark

Bài kiểm tra 16x H100 (80 GB HBM) trải trên 4 máy a3-highgpu-4g:

- Mức KV thấp (prompt ngắn, đồng thời thấp): mọi cấu hình bằng đường cơ sở, LMCache thêm ~3-5% chi phí phụ.
- Mức vừa: LMCache bắt đầu có ích nhờ tái sử dụng tiền tố giữa các engine.
- KV vượt HBM: offload CPU gốc và LMCache đều cải thiện thông lượng đáng kể; LMCache được lợi lớn hơn nhờ chia sẻ giữa các engine.

### Khi nào LMCache mang tính quyết định

- Phục vụ đa người thuê nơi system prompt được chia sẻ giữa các tenant.
- RAG nơi các đoạn tài liệu lặp lại qua các truy vấn.
- Các biến thể tinh chỉnh (LoRA) trên cùng một mô hình nền nơi tái sử dụng KV của mô hình nền cắt giảm việc làm dư thừa.
- Tải nhiều chiếm quyền: khôi phục từ CPU rẻ hơn prefill lại.

### Khi nào KHÔNG nên bật

- Áp lực HBM nhỏ — bạn trả chi phí phụ mà không có lợi.
- Ngữ cảnh ngắn (<1K token) — thời gian chuyển > prefill lại.
- Tải đơn tenant, đơn prompt — không có gì để tái sử dụng.

### Tích hợp với phục vụ tách rời

Phase 17 · 17 phục vụ tách rời + LMCache cộng hưởng: các lần chuyển KV từ pool prefill sang pool decode đáp xuống LMCache nếu không dùng đến; các truy vấn sau kéo từ LMCache. Router nhận biết cache của Phase 17 · 11 có thể định tuyến đến engine có cache cục bộ HOẶC cache dùng chung LMCache khớp.

### Những con số bạn nên nhớ

- vLLM 0.9.0: phát hành Connector API.
- vLLM 0.11.0 (tháng 1/2026): đường offload bất đồng bộ; tác động lên độ trễ đầu-cuối phụ thuộc vào tải, tỷ lệ trúng KV và áp lực hệ thống (không phải đảm bảo tuyệt đối).
- Benchmark 16x H100: LMCache có ích khi mức KV vượt HBM.
- Áp lực HBM nhỏ: chi phí phụ 3-5% mà không có lợi.

```figure
zero-sharding
```

## Sử dụng

`code/main.py` mô phỏng một tải nhiều chiếm quyền có và không có LMCache. Báo cáo số prefill lại tránh được, mức tăng thông lượng và mức dùng HBM hòa vốn.

## Đưa vào sản xuất

Bài học này tạo ra `outputs/skill-vllm-stack-decider.md`. Dựa trên hình dạng tải và triển khai vLLM, quyết định dùng gốc, LMCache hay không cái nào.

## Bài tập

1. Chạy `code/main.py`. Ở mức dùng HBM nào thì LMCache bắt đầu đáng giá?
2. Một tenant chia sẻ system prompt 6K token qua 200 truy vấn/giờ. Tính mức tiết kiệm kỳ vọng của LMCache cho mỗi tenant.
3. Máy chủ LMCache là điểm lỗi đơn (single point of failure). Thiết kế chiến lược HA (bản sao, dự phòng về gốc).
4. LMCache lưu xuống Ceph trên đĩa quay. Với KV 4K token ở 70B FP8 (500 MB), thời gian đọc so với prefill lại là bao nhiêu?
5. Biện luận xem đường bất đồng bộ của vLLM 0.11.0 có "miễn phí" không — chi phí phụ ẩn ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Production-stack | "triển khai tham chiếu" | Helm chart + operator Kubernetes của vLLM |
| Connector API | "giao diện backend KV" | Giao diện kho KV cắm-rút của vLLM 0.9.0+ |
| Offload CPU gốc (Native CPU offload) | "tràn cục bộ theo engine" | Lưu KV trong RAM máy chủ của cùng engine |
| LMCache | "KV cache của cụm" | Máy chủ KV cache liên engine trên CPU DRAM + đĩa |
| Bất đồng bộ 0.11.0 (0.11.0 async) | "offload không chặn" | Offload được ẩn sau luồng (stream) của engine |
| Chiếm quyền (Preemption) | "trục xuất để lấy chỗ" | Xáo trộn KV cache khi HBM đầy |
| Tái sử dụng tiền tố (Prefix reuse) | "cùng system prompt" | Nhiều truy vấn chia sẻ phần đầu; cache hit |
| Tầng Ceph (Ceph tier) | "tầng đĩa" | Lưu trữ bền vững dưới DRAM trong phân cấp cache |

## Đọc thêm

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — triển khai connector.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — chi tiết đường bất đồng bộ.
