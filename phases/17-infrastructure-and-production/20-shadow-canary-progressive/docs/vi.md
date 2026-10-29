# Lưu lượng bóng (Shadow Traffic), Triển khai Chim hoàng yến (Canary Rollout), và Triển khai lũy tiến (Progressive Deployment) cho LLM

> Triển khai LLM hội tụ những thách thức hóc búa nhất của quá trình phát hành phần mềm: không có kiểm thử đơn vị (unit tests) xác định, phương thức thất bại phân tán (diffuse failure modes), và tín hiệu phản hồi bị trễ (delayed signals). Trình tự chuẩn gồm (1) chế độ bóng (shadow mode) — nhân bản các yêu cầu từ môi trường production sang mô hình ứng viên, ghi nhật ký và so sánh mà không gây bất kỳ tác động nào tới người dùng; phát hiện sớm các vấn đề lệch phân phối hiển nhiên nhưng không đảm bảo trọn vẹn chất lượng; (2) triển khai chim hoàng yến (canary rollout) — chuyển dịch lưu lượng truy cập lũy tiến theo các mốc 10% → 25% → 50% → 75% → 100% kèm theo cổng kiểm soát (gates) nghiêm ngặt tại mỗi bước; theo dõi phân vị độ trễ (latency percentiles), chi phí trên mỗi yêu cầu (cost/request), tỷ lệ lỗi/từ chối phản hồi (error/refusal rate), phân phối độ dài đầu ra (output length distribution), và tỷ lệ phản hồi của người dùng; (3) thử nghiệm A/B (A/B testing) cho các phương án khác biệt rõ rệt sau khi độ ổn định đã được khẳng định. Tính phi tất định (non-determinism) là đặc tính cố hữu không thể triệt tiêu — biến thiên độ chính xác có thể lên tới 15% giữa các lần chạy với cùng đầu vào do tính chất không kết hợp của phép tính dấu phẩy động trên GPU (GPU FP non-associativity) kết hợp với sự biến thiên kích thước batch. Chi phí là một biến số chứ không phải hằng số — một mô hình tốt hơn 20% có thể đắt hơn gấp 3 lần cho mỗi lượt gọi. Tốc độ quay lui (rollback speed) mang tính quyết định: nếu việc quay lui đòi hỏi triển khai lại toàn bộ (redeploy), bạn đã quá chậm. Chính sách nằm trong cấu hình/cờ tính năng (feature flags); mô hình nằm trong sổ đăng ký (registry) với mã băm định danh cố định (pinned digests); quay lui = gạt cờ chính sách + đảo ngược ngưỡng kích hoạt + ghim lại mô hình cũ chỉ trong vài giây.

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 minutes

## Learning Objectives

- Phân biệt chế độ bóng (shadow mode - so sánh không ảnh hưởng người dùng), chim hoàng yến (canary - chuyển dịch lưu lượng thực lũy tiến), và thử nghiệm A/B (so sánh phương án sau khi xác nhận tính ổn định).
- Liệt kê năm chỉ số kiểm soát (canary metrics) đặc thù của LLM (độ trễ, chi phí/yêu cầu, lỗi/từ chối, phân phối độ dài đầu ra, phản hồi người dùng).
- Giải thích lý do tính phi tất định của LLM (lên đến 15%) làm thay đổi định nghĩa về trạng thái "ổn định" trong quá trình triển khai.
- Thiết kế một lộ trình quay lui (rollback path) chỉ mất vài giây (chuyển đổi cờ chính sách) thay vì hàng giờ đồng hồ (triển khai lại toàn bộ).

## The Problem

Bạn vừa phát hành một mô hình mới. Đánh giá ngoại tuyến (offline evals) cho thấy độ chính xác tăng 3%. Bạn tự tin bật nó trên môi trường production. Trong vòng 24 giờ, chi phí vọt tăng 40%, tỷ lệ người dùng bấm không thích (thumbs-down) tăng 8%, và có 3 phiếu yêu cầu hỗ trợ (tickets) phàn nàn về "câu trả lời kỳ quặc". Bạn buộc phải quay lui (rollback). Quá trình biên dịch và triển khai lại (redeploy) ngốn mất 3 giờ. Kỳ nghỉ cuối tuần của bạn tan tành.

Mọi mắt xích trong thảm họa đó đều có thể phòng tránh được. Chế độ bóng (shadow mode) đáng lẽ đã phát hiện đợt tăng vọt 40% chi phí trước khi bất kỳ người dùng nào kịp nhìn thấy. Chiến lược chim hoàng yến (canary) đáng lẽ đã chặn đứng lưu lượng ở mức 10% ngay khi chỉ số phản hồi tiêu cực bắt đầu nhúc nhích. Cơ chế quay lui bằng cờ chính sách (policy-flag rollback) đáng lẽ chỉ tốn đúng 30 giây. Kỷ luật trong triển khai chính là chiếc cầu nối lấp đầy khoảng trống giữa "kết quả đánh giá ngoại tuyến rất đẹp" và "người dùng thực sự hài lòng".

## The Concept

### Chế độ bóng (Shadow mode)

Mô hình ứng viên (candidate) nhận cùng các yêu cầu như môi trường production; đầu ra chỉ được ghi nhật ký (log) chứ không trả về cho người dùng. Hoàn toàn không gây tác động đến trải nghiệm người dùng (zero user impact). Nhật ký cần ghi lại:

- Nội dung đầu ra (so sánh sai khác diff với production).
- Số lượng token tiêu thụ (chênh lệch chi phí delta).
- Độ trễ phản hồi.
- Tỷ lệ từ chối phản hồi (refusal) và lỗi hệ thống.

Phát hiện được: chi phí tăng vọt ngoài tầm kiểm soát, thoái lui về độ dài câu trả lời, thay đổi rõ rệt trong phản ứng từ chối, các lỗi cứng (hard errors). KHÔNG phát hiện được: những suy giảm chất lượng tinh vi mà người dùng trực tiếp cảm nhận. Chế độ bóng chỉ đóng vai trò kiểm thử khói (smoke test), không phải là bài kiểm định chất lượng toàn diện.

### Triển khai chim hoàng yến (Canary rollout)

Chuyển dịch dần lưu lượng truy cập theo từng bước có cổng kiểm soát (gates). Lộ trình điển hình: 1% → 10% → 25% → 50% → 75% → 100%. Đặt cổng chặn trên 5 chỉ số tại mỗi bước:

1. **Phân vị độ trễ (Latency percentiles)** — P50, P95, P99. Vi phạm: bản canary có P99 > 1.5x so với đường cơ sở (baseline).
2. **Chi phí trên mỗi yêu cầu (Cost per request)** — Chi phí trung bình tính bằng $. Vi phạm: >20% so với đường cơ sở.
3. **Tỷ lệ lỗi / từ chối (Error / refusal rate)** — Mã lỗi 5xx cộng với các trường hợp mô hình chủ động từ chối trả lời. Vi phạm: vượt 2x so với đường cơ sở.
4. **Phân phối độ dài đầu ra (Output length distribution)** — Giá trị trung bình + phân vị P99. Vi phạm: xuất hiện hiện tượng dịch chuyển phân phối (distributional shift).
5. **Tỷ lệ phản hồi từ người dùng (User-feedback rate)** — Bấm thumbs-down / gửi phiếu báo lỗi. Vi phạm: vượt 1.5x so với đường cơ sở.

### Tính phi tất định là dạng biến thiên mới (Non-determinism is the new variance)

Các đầu vào hoàn toàn giống nhau vẫn có thể sinh ra các đầu ra khác nhau. Nguyên nhân:

- Tính chất không kết hợp của phép tính dấu phẩy động trên GPU (floating-point non-associativity): thứ tự thu gọn phép tính (reduction order) thay đổi theo từng batch.
- Biến thiên kích thước batch (cùng một prompt nhưng nằm trong batch 128 sẽ có kết quả số học vi mô khác với khi nằm trong batch 16).
- Quá trình lấy mẫu ngẫu nhiên (khi tham số temperature > 0).

Đo đạc thực tế ghi nhận: độ chính xác có thể biến thiên lên tới 15% giữa các lần chạy lặp lại trên cùng một tập đánh giá. Trạng thái "ổn định" trong một đợt phát hành nghĩa là các chỉ số nằm trong ngưỡng phương sai kỳ vọng, chứ không phải giống hệt tuyệt đối với đường cơ sở. Do đó, hãy thiết lập các cổng kiểm soát cao hơn mức nhiễu nền (noise floor).

### Chi phí là một biến số (Cost is a variable)

Một mô hình cho kết quả tốt hơn 20% có thể đắt hơn gấp 3 lần cho mỗi lượt gọi. Chi phí/yêu cầu là một trong năm cổng kiểm soát bắt buộc. Việc phát hành một mô hình "thông minh hơn" nhưng phá vỡ bài toán kinh tế trên từng đơn vị (unit economics) chính là lý do thỏa đáng để kích hoạt quay lui.

### Quay lui là vũ khí tối thượng (Rollback is the weapon)

- Cờ chính sách (hệ thống cờ tính năng - feature flags): điều chỉnh tỷ lệ phần trăm phân luồng trong tệp cấu hình; diễn ra trong vài giây.
- Ghim phiên bản mô hình (mã băm digest trong sổ đăng ký registry): mô hình được ghim cố định sẽ không tự ý nâng cấp ngầm.
- Quay lui = đảo ngược cờ tính năng + đặt lại mã digest đã ghim về phiên bản cũ. Tính bằng giây, không phải bằng giờ.

Nếu cấu trúc hạ tầng hiện tại của bạn bắt buộc phải tái triển khai (redeploy) mới quay lui được, hãy sửa chữa điểm yếu đó trước khi tiến hành phát hành phiên bản mới.

### Bộ công cụ hỗ trợ (Tooling)

**Argo Rollouts** / **Flagger** — Bộ điều khiển phân phối lũy tiến (progressive delivery controllers) chuẩn cho Kubernetes. Tích hợp trực tiếp với cơ chế định tuyến theo trọng số của Istio/Linkerd.

**Định tuyến theo trọng số của Istio (Istio weighted routing)** — Cơ chế chia tách lưu lượng ở cấp độ lưới dịch vụ (service mesh).

**KServe / Seldon Core** — Nền tảng phục vụ mô hình (model serving) tích hợp sẵn các cơ chế nguyên thủy cho canary.

**Cờ tính năng (Feature flags)** — LaunchDarkly, Flagsmith, Unleash. Thay đổi chính sách ngay tức thì, không cần redeploy.

### Nhịp độ đánh giá chỉ số (Metrics cadence)

Các cổng kiểm soát chim hoàng yến kiểm tra định kỳ mỗi 5-15 phút tùy thuộc vào khối lượng lưu lượng. 1% của 1.000 req/phút (tức 10 req/phút chuyển vào bản canary) sẽ tạo ra 50-150 điểm dữ liệu trong mỗi cửa sổ 5-15 phút — đủ để đánh giá độ trễ nhưng còn quá nhiễu đối với phản hồi của người dùng. Mức 10% sẽ cung cấp dữ liệu gấp ~10 lần. Do đó, quá trình nâng tỷ lệ phải tạm dừng đủ lâu tại mỗi bước để tích lũy đủ kích thước mẫu có ý nghĩa thống kê.

### Thử nghiệm A/B là bước tùy chọn

Nếu mô hình mới sở hữu hành vi khác biệt rõ rệt (phong cách trả lời khác, đường cong chi phí khác, văn phong khác), hãy tiến hành thử nghiệm A/B ở mức 50% sau khi đã vượt qua các cổng chim hoàng yến an toàn. Nếu nó chỉ đơn thuần là bản nâng cấp chất lượng đồng nhất, bạn có thể nhảy thẳng lên 100% ngay khi các cổng kiểm soát canary thông qua.

### Những con số cần ghi nhớ

- Các bước nâng tỷ lệ chim hoàng yến: 1% → 10% → 25% → 50% → 75% → 100%.
- Giới hạn biến thiên phi tất định: lên tới 15% sai lệch giữa các lần chạy trên cùng đầu vào.
- Năm chỉ số kiểm soát chim hoàng yến: độ trễ, chi phí, lỗi/từ chối, phân phối độ dài, phản hồi người dùng.
- Cổng kiểm soát chi phí: vượt >20% so với đường cơ sở bị coi là vi phạm.
- Thời gian quay lui: tính bằng giây, không phải bằng giờ.

```figure
i4-canary-ramp
```

## Use It

Tệp `code/main.py` mô phỏng quy trình triển khai chim hoàng yến khi bị tiêm các hiện tượng suy thoái chất lượng (regressions). Bản mô phỏng báo cáo chi tiết giai đoạn nào quá trình triển khai bị chặn lại và cổng kiểm soát nào đã kích hoạt cảnh báo.

## Ship It

Bài học này xây dựng tệp `outputs/skill-rollout-runbook.md`. Dựa trên mô hình ứng viên, đường cơ sở hiện tại, và mức độ chấp nhận rủi ro, kỹ năng này sẽ thiết kế một kế hoạch hoàn chỉnh từ chạy bóng (shadow) → chim hoàng yến (canary) → 100% production.

## Exercises

1. Chạy `code/main.py`. Tiêm vào một đợt suy thoái khiến chi phí tăng 25%. Quá trình triển khai chim hoàng yến sẽ dừng lại ở giai đoạn nào?
2. Mô hình mới của bạn đạt mức tăng 3% độ chính xác trong đánh giá ngoại tuyến nhưng chi phí/yêu cầu tăng +18%. Có nên phát hành hay không? Quyết định phụ thuộc vào chính sách kinh tế — hãy viết quy trình quyết định cho cả 2 trường hợp.
3. Thiết kế một hệ thống quay lui đảm bảo hoàn tất trong vòng dưới 60 giây từ đầu đến cuối. Liệt kê các thành phần hạ tầng bắt buộc.
4. Hiện tượng phi tất định cho thấy độ lệch ±7% trên tập đánh giá của bạn. Hãy thiết lập các ngưỡng cổng kiểm soát canary sao cho không bị kích hoạt báo động giả. Bạn sẽ sử dụng hệ số nhân nào?
5. Chế độ bóng phát hiện chi phí tăng vọt 40% trước khi bước vào giai đoạn canary. Viết quy tắc cảnh báo (alert rule) kích hoạt ngay trong giai đoạn chạy bóng này.

## Key Terms

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Chế độ bóng (Shadow mode) | "Nhân bản sang mô hình mới" | Gửi yêu cầu sang ứng viên để ghi nhật ký mà không gây tác động tới người dùng |
| Chim hoàng yến (Canary) | "Mở tải lũy tiến" | Phát hành tăng dần lưu lượng thực cho người dùng kèm các cổng kiểm soát |
| Cổng kiểm soát (Gates) | "Chốt kiểm tra phát hành" | Các ngưỡng chỉ số kích hoạt ngăn chặn tiến trình mở rộng lưu lượng |
| Tính phi tất định (Non-determinism) | "Độ biến thiên của LLM" | Hiện tượng sai khác cố hữu giữa các lần chạy trên cùng một dữ liệu đầu vào |
| Cờ chính sách (Policy flag) | "Quay lui bằng cờ" | Thao tác quay lui ở tầng cấu hình, chỉ mất vài giây thay vì hàng giờ |
| Ghim mô hình (Model pin) | "Mã băm registry" | Tham chiếu bất biến trỏ tới một phiên bản cụ thể của mô hình |
| Argo Rollouts | "Phát hành lũy tiến trên K8s" | Bộ điều khiển canary/rollback hoạt động tự nhiên trong Kubernetes |
| KServe | "K8s cho suy luận AI" | Nền tảng phục vụ mô hình cung cấp sẵn các cơ chế nguyên thủy cho canary |
| Định tuyến trọng số Istio (Istio weighted) | "Chia tải trên lưới dịch vụ" | Bộ chia tách lưu lượng hoạt động ở tầng service mesh |

## Further Reading

- [TianPan — Phát hành tính năng AI mà không làm gián đoạn Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Triển khai an toàn mô hình Machine Learning](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Các mẫu triển khai LLM nâng cao](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Tài liệu hướng dẫn Argo Rollouts](https://argo-rollouts.readthedocs.io/)
- [Tài liệu hướng dẫn Flagger](https://docs.flagger.app/)
