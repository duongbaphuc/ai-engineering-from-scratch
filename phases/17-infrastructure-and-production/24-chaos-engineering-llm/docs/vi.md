# Kỹ thuật hỗn loạn (Chaos Engineering) cho LLM trong môi trường production

> Kỹ thuật hỗn loạn cho LLM là một chuyên ngành riêng vào năm 2026. Điều kiện tiên quyết trước khi chạy thí nghiệm trên production: SLI/SLO được xác định, khả năng quan sát (observability) gồm trace + metric + log, rollback tự động, runbook, trực on-call. Kiến trúc gồm bốn mặt phẳng (plane): điều khiển (control – bộ lập lịch thí nghiệm), mục tiêu (target – dịch vụ, hạ tầng, kho dữ liệu), an toàn (safety – guard + abort + bộ lọc lưu lượng), quan sát (observability – metric + trace + log), cùng vòng phản hồi (feedback) để điều chỉnh SLO. Rào chắn (guardrail) là bắt buộc: cảnh báo burn-rate sẽ tạm dừng thí nghiệm nếu mức tiêu hao error budget trong ngày > 2 lần mức kỳ vọng; cửa sổ ngắt cảnh báo (suppression window) và đối chiếu trace-ID giúp khử nhiễu cảnh báo. Nhịp vận hành: canary nhỏ hằng tuần + rà soát SLO; game day hằng tháng + postmortem; kiểm toán khả năng chống chịu liên nhóm và lập bản đồ phụ thuộc hằng quý. Thí nghiệm đặc thù cho LLM: quá tải bộ nhớ, lỗi mạng, sự cố nhà cung cấp, prompt dị dạng, bão loại bỏ KV cache. Công cụ: Harness Chaos Engineering (khuyến nghị do LLM tạo, thu hẹp blast radius, tích hợp công cụ MCP); LitmusChaos (CNCF); Chaos Mesh (CNCF, native cho Kubernetes).

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Mục tiêu học tập

- Kể tên năm điều kiện tiên quyết của kỹ thuật hỗn loạn (SLI/SLO, observability, rollback, runbook, on-call) và giải thích vì sao bỏ qua bất kỳ điều kiện nào cũng làm hỏng quy trình.
- Vẽ sơ đồ bốn mặt phẳng (điều khiển, mục tiêu, an toàn, quan sát) và vòng phản hồi về SLO.
- Liệt kê năm thí nghiệm đặc thù cho LLM (quá tải bộ nhớ, lỗi mạng, sự cố nhà cung cấp, prompt dị dạng, bão loại bỏ KV).
- Chọn công cụ — Harness, LitmusChaos, Chaos Mesh — phù hợp với stack hiện có.

## Vấn đề

Kiểm thử hỗn loạn trong các stack truyền thống đã được thiết lập vững chắc. Các stack LLM bổ sung thêm những kiểu lỗi mới. Một prompt 4K token chứa ký tự độc có thể làm tokenizer đứng 12 giây. Nhà cung cấp upstream trả về 429; gateway của bạn thử lại; dịch vụ bị OOM do mức đồng thời bị khuếch đại bởi việc thử lại. Bão loại bỏ KV cache dưới tải đột biến gây ra chuỗi re-prefill làm bão hòa tài nguyên tính toán.

Không lỗi nào trong số này xuất hiện trong unit test. Kỹ thuật hỗn loạn là cách để bạn phát hiện chúng trước khi người dùng phát hiện.

## Khái niệm

### Điều kiện tiên quyết

Đừng chạy chaos trên production nếu thiếu:

1. **SLI/SLO** — chỉ số mức dịch vụ (service-level indicator) và mục tiêu mức dịch vụ (service-level objective) đã được xác định.
2. **Observability** — trace, metric, log được nối vào dashboard.
3. **Rollback tự động** — rollback bằng policy-flag của Phase 17 · 20.
4. **Runbook** — có cấu trúc, theo Phase 17 · 23.
5. **On-call** — có người phản ứng.

Thiếu bất kỳ mục nào thì chaos sẽ biến thành sự cố thật.

### Bốn mặt phẳng + phản hồi

**Mặt phẳng điều khiển (control plane)** — bộ lập lịch thí nghiệm (workflow của Litmus, schedule của Chaos Mesh, giao diện Harness).

**Mặt phẳng mục tiêu (target plane)** — dịch vụ, pod, node, bộ cân bằng tải, kho dữ liệu.

**Mặt phẳng an toàn (safety plane)** — công tắc ngắt khẩn cấp (kill switch), cửa sổ ngắt cảnh báo, giới hạn blast radius, cổng error-budget.

**Mặt phẳng quan sát (observability plane)** — metric thông thường + đối chiếu trace-ID để phân biệt lỗi do chaos với lỗi tự nhiên.

**Vòng phản hồi (feedback loop)** — kết quả quay lại để điều chỉnh SLO, cập nhật runbook, sửa mã.

### Guardrail là bắt buộc

- **Cảnh báo burn-rate**: tạm dừng thí nghiệm nếu mức tiêu hao error budget trong ngày vượt quá 2 lần mức kỳ vọng.
- **Cửa sổ ngắt cảnh báo (suppression window)**: tắt các cảnh báo không thuộc thí nghiệm trong phạm vi blast radius khi thí nghiệm chạy.
- **Đối chiếu trace-ID**: mọi lỗi do thí nghiệm gây ra đều mang một thẻ để on-call có thể khử trùng lặp.

### Năm thí nghiệm đặc thù cho LLM

1. **Quá tải bộ nhớ** — buộc xảy ra bão preemption KV cache bằng cách gửi các request ngữ cảnh dài với mức đồng thời cao. Quan sát: dịch vụ loại bỏ tải một cách êm dịu hay sập?

2. **Lỗi mạng** — cắt kết nối giữa inference gateway và nhà cung cấp. Quan sát: fallback có kích hoạt trong SLA không? (Phase 17 · 19)

3. **Mô phỏng sự cố nhà cung cấp** — 100% phản hồi 429 từ OpenAI. Quan sát: định tuyến có chuyển dự phòng sang Anthropic không? (Phase 17 · 16, 19)

4. **Prompt dị dạng** — chèn payload làm tokenizer đứng (ví dụ unicode lồng sâu, codepoint UTF-8 khổng lồ). Quan sát: một request đơn lẻ có khóa cứng một worker không?

5. **Bão loại bỏ KV** — ép loại bỏ bằng cách làm bão hòa ngân sách block của vLLM. Quan sát: LMCache phục hồi hay dịch vụ suy giảm?

### Nhịp vận hành

- **Hằng tuần** — thí nghiệm canary nhỏ trên staging, có thể 5% production.
- **Hằng tháng** — game day theo lịch cho một kịch bản cụ thể; nhiều nhóm tham dự; postmortem.
- **Hằng quý** — kiểm toán khả năng chống chịu liên nhóm; cập nhật bản đồ phụ thuộc.

### Công cụ

- **Harness Chaos Engineering** — thương mại; khuyến nghị thí nghiệm do AI tạo; thu hẹp blast radius; tích hợp công cụ MCP.
- **LitmusChaos** — CNCF graduated; dựa trên workflow Kubernetes.
- **Chaos Mesh** — CNCF sandbox; kiểu CRD native cho Kubernetes.
- **Gremlin** — thương mại; hỗ trợ rộng.
- **AWS FIS** / **Azure Chaos Studio** — dịch vụ đám mây được quản lý.

### Bắt đầu nhỏ

Thí nghiệm đầu tiên: pod-kill một replica decode dưới lưu lượng ổn định. Quan sát việc định tuyến lại và phục hồi. Nếu hoạt động tốt và trông an toàn, nâng cấp lên chaos mạng.

Thí nghiệm đặc thù LLM đầu tiên: chèn một phản hồi 429 từ nhà cung cấp trong 5 phút. Quan sát fallback. Hầu hết các nhóm phát hiện fallback của mình chưa từng được kiểm thử đầy đủ.

### Những con số cần nhớ

- Bốn mặt phẳng: điều khiển, mục tiêu, an toàn, quan sát.
- Ngưỡng tạm dừng burn-rate: 2 lần mức tiêu hao ngân sách hằng ngày kỳ vọng.
- Nhịp vận hành: canary hằng tuần, game day hằng tháng, kiểm toán hằng quý.
- Năm thí nghiệm LLM: bộ nhớ, mạng, nhà cung cấp, prompt dị dạng, bão KV.

```figure
i4-chaos-guard
```

## Sử dụng

`code/main.py` mô phỏng ba thí nghiệm chaos với các cổng của mặt phẳng an toàn. Báo cáo thí nghiệm nào sẽ kích hoạt việc hủy do burn-rate.

## Triển khai

Bài học này tạo ra `outputs/skill-chaos-plan.md`. Dựa trên stack và mức độ trưởng thành, nó chọn ba thí nghiệm đầu tiên và công cụ.

## Bài tập

1. Chạy `code/main.py`. Thí nghiệm nào kích hoạt cổng burn-rate và vì sao?
2. Thiết kế năm thí nghiệm chaos đầu tiên cho một dịch vụ RAG dựa trên vLLM. Bao gồm tiêu chí thành công.
3. Cảnh báo burn-rate của bạn đã tạm dừng một thí nghiệm. Làm sao xác định nguyên nhân gốc — chaos hay tự nhiên?
4. Lập luận xem chaos nên chạy trên production hay chỉ trên staging. Khi nào production là câu trả lời đúng?
5. Nêu ba kiểu lỗi đặc thù LLM mà chaos mạng thông thường không thể tái tạo.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| SLI / SLO | "mục tiêu dịch vụ" | Chỉ số + mục tiêu; điều kiện tiên quyết bắt buộc |
| Blast radius | "phạm vi" | Tập dịch vụ / người dùng bị ảnh hưởng bởi thí nghiệm |
| Cảnh báo burn-rate | "cổng ngân sách" | Kích hoạt khi tốc độ tiêu hao error budget > 2 lần kỳ vọng |
| Game day | "buổi diễn tập hằng tháng" | Bài tập chaos liên nhóm theo lịch |
| LitmusChaos | "workflow CNCF" | Công cụ chaos Kubernetes đã graduated của CNCF |
| Chaos Mesh | "CRD CNCF" | Công cụ chaos native cho Kubernetes thuộc CNCF sandbox |
| Harness CE | "thương mại có AI hỗ trợ" | Harness chaos với khuyến nghị từ AI |
| Prompt dị dạng | "bom tokenizer" | Đầu vào làm đứng quá trình tokenization |
| Bão loại bỏ KV | "thác preemption" | Loại bỏ hàng loạt kéo theo các re-prefill |

## Đọc thêm

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for Large Language Models (Apress, 2026)](https://link.springer.com/book/10.1007/979-8-8688-2827-0)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
