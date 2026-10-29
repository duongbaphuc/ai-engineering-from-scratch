# Đồ án Tốt nghiệp 06 — Tác tử Khắc phục sự cố DevOps cho Kubernetes

> AWS DevOps Agent chính thức phát hành rộng rãi (GA), Resolve AI công bố sổ tay xử lý sự cố Kubernetes (K8s playbooks), NeuBird trình diễn giám sát ngữ nghĩa, và Metoro gắn kết AI SRE với các chỉ tiêu mức dịch vụ (SLO) trên từng dịch vụ. Hình thái chuẩn mực trong môi trường sản xuất đã được định hình rõ nét: Webhook cảnh báo kích hoạt, tác tử tự động đọc dữ liệu đo lường từ xa (telemetry), duyệt đồ thị các đối tượng K8s, xếp hạng các giả thuyết về nguyên nhân gốc rễ (Root Cause Analysis - RCA), và gửi một bản tóm tắt lên Slack kèm các nút phê duyệt hành động khắc phục. Mặc định chỉ có quyền đọc (read-only by default). Mọi hành động can thiệp sửa đổi đều phải qua cổng kiểm duyệt của con người. Đồ án tốt nghiệp này yêu cầu bạn xây dựng tác tử đó, đánh giá hiệu năng trên 20 kịch bản sự cố mô phỏng và đối chiếu trực tiếp với AWS Agent trên 3 ca sự cố dùng chung.

**Type:** Capstone
**Languages:** Python (tác tử), TypeScript (tích hợp Slack)
**Prerequisites:** Phase 11 (Kỹ thuật LLM), Phase 13 (Công cụ và giao thức MCP), Phase 14 (Tác tử), Phase 15 (Hệ thống tự trị), Phase 17 (Hạ tầng), Phase 18 (An toàn và rào chắn)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** ~30 giờ

## Bài toán (Problem)

Trong giai đoạn 2025–2026, triết lý vận hành SRE đã chuyển dịch mạnh mẽ: "Tác tử AI phân loại sự cố, con người phê duyệt hành động khắc phục." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps đều triển khai kiến trúc này trên môi trường sản xuất. Tác tử đọc các số liệu đo lường từ Prometheus, bản ghi nhật ký (logs) từ Loki, dấu vết phân tán (traces) từ Tempo, trạng thái cụm từ kube-state-metrics, cùng một đồ thị tri thức liên kết các đối tượng K8s. Hệ thống đưa ra bảng xếp hạng giả thuyết nguyên nhân gốc rễ đi kèm trích dẫn dữ liệu đo lường trong thời gian chưa đầy 5 phút. Tác tử tuyệt đối không tự ý thực thi các lệnh mang tính phá hủy nếu không có sự phê duyệt rõ ràng từ con người thông qua giao diện Slack.

Phần lớn khối lượng công việc phức tạp nằm ở việc giới hạn phạm vi quyền hạn và đảm bảo an toàn, chứ không đơn thuần là năng lực suy luận của mô hình. Tác tử đòi hỏi một bề mặt phân quyền RBAC mặc định chỉ đọc (read-only-by-default), một máy chủ công cụ MCP được gia cố bảo mật, và nhật ký kiểm toán (audit log) ghi lại từng câu lệnh được cân nhắc so với câu lệnh thực sự chạy. Nó cần biết rõ giới hạn năng lực để kịp thời chuyển cấp báo động (escalate) cho kỹ sư trực. Đồng thời, chi phí vận hành phải đủ tiết kiệm để một đợt sự cố sập bộ nhớ hàng loạt (OOM-kill cascade) không làm phát sinh hóa đơn API lên tới $5.000.

## Khái niệm (Concept)

Tác tử vận hành dựa trên một đồ thị tri thức (knowledge graph):
- **Các nút (Nodes):** Bao gồm các đối tượng Kubernetes (Pod, Deployment, Service, Node, HPA, PVC) cùng các nguồn dữ liệu đo lường từ xa (chuỗi Prometheus, luồng Loki, vết Tempo).
- **Các cạnh (Edges):** Mã hóa quan hệ sở hữu (`Pod -> ReplicaSet -> Deployment`), quan hệ lập lịch (`Pod -> Node`), và quan hệ quan sát (`Pod -> chuỗi Prometheus`).
Đồ thị được duy trì tươi mới nhờ cơ chế đồng bộ định kỳ từ kube-state-metrics và được lấy mẫu lại tức thì mỗi khi có cảnh báo xuất hiện.

Khi nhận cảnh báo, tác tử tiến hành truy vết nguyên nhân bắt đầu từ đối tượng bị ảnh hưởng. Nó duyệt dọc theo các cạnh đồ thị, trích xuất các lát cắt dữ liệu đo lường liên quan trong vòng 15 phút gần nhất, và xây dựng các giả thuyết khả dĩ. Giả thuyết được xếp hạng dựa trên trọng số bằng chứng: có bao nhiêu trích dẫn đo lường chứng minh, độ tươi mới của dữ liệu, và độ cụ thể của triệu chứng. Top 3 giả thuyết hàng đầu được gửi lên kênh Slack kèm hình ảnh trực quan hóa đường đi trên đồ thị và các nút bấm phê duyệt cho hành động xử lý tương ứng.

Quy trình khắc phục sự cố (remediation) được kiểm soát nghiêm ngặt. Các hành động mặc định chỉ là các thao tác đọc. Các hành động có khả năng gây gián đoạn (thu hẹp quy mô scale-down, hoàn tác phiên bản rollback, xóa Pod) bắt buộc phải có sự phê duyệt trên Slack; các hook hoàn tác của ArgoCD đòi hỏi token xác thực mà tác tử không bao giờ được phép nắm giữ. Nhật ký kiểm toán ghi lại toàn bộ các lệnh mà tác tử *từng cân nhắc* — chứ không chỉ các lệnh đã thực thi — giúp quy trình hậu kiểm phát hiện sớm các nguy cơ tiềm ẩn.

## Kiến trúc (Architecture)

```text
Webhook từ PagerDuty / Alertmanager
           |
           v
    Bộ tiếp nhận FastAPI
           |
           v
Tác tử phân tích nguyên nhân gốc rễ (LangGraph)
           |
           +---- Công cụ MCP chỉ đọc ----+
           |                             |
           v                             v
Đồ thị tri thức K8s            Lát cắt dữ liệu đo lường
  (Neo4j / kuzu)                Prometheus, Loki, Tempo
Quan hệ sở hữu & lập lịch        15 phút gần nhất, có phạm vi
           |
           v
Xếp hạng giả thuyết (theo trọng số bằng chứng)
           |
           v
Bản tóm tắt Slack + Các nút bấm phê duyệt
           |
           v (khi được phê duyệt)
Hook rollback ArgoCD / Chuyển cấp PagerDuty
           |
           v
Nhật ký kiểm toán: Lệnh cân nhắc so với Lệnh thực thi
```

## Công nghệ sử dụng (Stack)

- Nguồn dữ liệu khả năng quan sát: Prometheus, Loki, Tempo, kube-state-metrics
- Đồ thị tri thức: Neo4j (dịch vụ đám mây) hoặc kuzu (cơ sở dữ liệu nhúng) lưu trữ đối tượng K8s và các cạnh đo lường
- Tác tử: LangGraph với danh sách trắng (allow-list) cho từng công cụ, mặc định chỉ có quyền đọc
- Giao thức truyền tải công cụ: FastMCP qua StreamableHTTP; tách riêng máy chủ cho các công cụ thay đổi trạng thái sau cổng phê duyệt
- Mô hình: Claude Sonnet 4.7 cho suy luận nguyên nhân gốc rễ, Gemini 2.5 Flash để tóm tắt log
- Khắc phục sự cố: Webhook hoàn tác ArgoCD, chuyển cấp PagerDuty, thẻ tin nhắn tương tác Slack
- Kiểm toán (Audit): Tệp nhật ký có cấu trúc chỉ ghi tiếp (append-only) ghi lại lệnh cân nhắc, thực thi, phê duyệt và kết quả
- Triển khai: Một Deployment K8s với vai trò RBAC tối thiểu riêng biệt; đặt trong namespace độc lập

```figure
ce-rootcause-walk
```

## Các bước triển khai (Build It)

1. **Nạp dữ liệu đồ thị (Graph ingestion).** Đồng bộ trạng thái từ kube-state-metrics vào Neo4j/kuzu mỗi 30 giây. Các loại nút: Pod, Deployment, Node, Service, PVC, HPA. Các loại cạnh: `OWNED_BY`, `SCHEDULED_ON`, `EXPOSES`, `MOUNTS`, `SCALES`. Các cạnh phủ đo lường: `OBSERVED_BY` (một Pod được quan sát bởi một chuỗi số liệu Prometheus).

2. **Tiếp nhận cảnh báo (Alert receiver).** Xây dựng điểm cuối FastAPI nhận webhook từ PagerDuty hoặc Alertmanager. Trích xuất đối tượng bị ảnh hưởng và chỉ số vi phạm cam kết mức dịch vụ (SLO breach).

3. **Bề mặt công cụ chỉ đọc.** Đóng gói `kubectl`, truy vấn Prometheus, Loki logql, Tempo traceql qua FastMCP. Mỗi công cụ chỉ có quyền thực thi các động từ RBAC hạn hẹp (`get`, `list`, `describe`). Tuyệt đối không chứa `delete`, `exec`, `scale` trên máy chủ mặc định.

4. **Tác tử phân tích nguyên nhân gốc rễ (Root-cause agent).** Xây dựng đồ thị LangGraph với ba nút chức năng: `sample` lấy lát cắt dữ liệu đo lường 15 phút gần nhất, `walk` truy vấn đồ thị tìm các đối tượng lân cận, `hypothesize` soạn thảo các giả thuyết nguyên nhân gốc rễ có xếp hạng kèm bằng chứng trích dẫn.

5. **Chấm điểm bằng chứng (Evidence scoring).** Mỗi giả thuyết được tính điểm = `độ tươi mới × độ cụ thể × nghịch đảo độ dài đường đi đồ thị × số lượng trích dẫn`. Trả về top 3 giả thuyết cao nhất.

6. **Bản tóm tắt Slack.** Đăng tải một tin nhắn đính kèm giả thuyết, hình ảnh trực quan hóa đường đi trên đồ thị (ảnh đồ thị con kết xuất từ phía máy chủ), và các nút bấm phê duyệt cho tối đa một hành động khắc phục cụ thể.

7. **Cổng kiểm duyệt khắc phục sự cố (Remediation gate).** Các công cụ mang tính thay đổi trạng thái (thu hẹp quy mô, hoàn tác phiên bản, xóa tài nguyên) nằm trên một máy chủ MCP thứ hai, được bảo vệ bởi mã thông báo phê duyệt (approval token). Tác tử chỉ được phép gọi các công cụ này sau khi thẻ Slack được con người nhấp xác nhận.

8. **Nhật ký kiểm toán (Audit log).** Tệp định dạng JSONL chỉ ghi tiếp: với mỗi lệnh tiềm năng, ghi nhận xem lệnh đó đã được cân nhắc hay chưa, có được thực thi không, và ai là người phê duyệt. Tự động đồng bộ lên Amazon S3 hàng ngày.

9. **Bộ kịch bản sự cố mô phỏng.** Xây dựng 20 tình huống thực tế: sụp đổ dây chuyền do cạn bộ nhớ OOMKill, gián đoạn phân giải DNS, dao động co giãn HPA liên tục, đầy phân vùng lưu trữ PVC, hiện tượng láng giềng ồn ào (noisy neighbor), lỗi container phụ (faulty sidecar), triển khai ConfigMap lỗi, quay vòng chứng chỉ hết hạn, lỗi nghẽn kéo ảnh image-pull backoff, v.v. Chấm điểm tác tử về độ chuẩn xác của nguyên nhân gốc rễ và thời gian đưa ra giả thuyết.

## Ứng dụng thực tế (Use It)

```text
webhook: alert.pagerduty.com -> vi phạm SLO checkout-api, tỷ lệ lỗi 14%
[graph]   đối tượng bị ảnh hưởng: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    đối tượng lân cận: ReplicaSet checkout-api-abc, Service checkout-api,
           triển khai mới cách đây 14 phút
[sample]  prometheus error_rate 14%, xu hướng tăng; loki ghi nhận mã 500 trên /api/v2/pay
[hypo]    #1 bản triển khai lỗi: image mới nhất checkout-api:v2.41 thất bại tại /healthz
           trích dẫn: deploy.yaml (rev 42), prometheus errorRate, loki 500 stacktrace
[slack]   [HOÀN TÁC về v2.40]  [CHUYỂN CẤP]  [BỎ QUA]
           (yêu cầu phê duyệt; tác tử không tự ý hoàn tác phiên bản)
```

## Triển khai thành phẩm (Ship It)

Sản phẩm kỹ năng được bàn giao tại `outputs/skill-devops-agent.md`. Khi kết nối với một cụm K8s và một nguồn cảnh báo, tác tử sẽ tạo ra các giả thuyết nguyên nhân gốc rễ có xếp hạng kèm theo quy trình khắc phục sự cố được kiểm duyệt qua Slack. Bảng tiêu chí đánh giá:

| Trọng số | Tiêu chí | Phương thức đo lường |
|:-:|---|---|
| 25 | Độ chính xác RCA trên bộ kịch bản | Đạt ≥80% chẩn đoán đúng nguyên nhân gốc rễ trên 20 sự cố mô phỏng |
| 20 | Tính an toàn | Rào chắn hành động phá hủy không bao giờ chạy mà thiếu phê duyệt Slack trong nhật ký kiểm toán |
| 20 | Thời gian đưa ra giả thuyết | p50 dưới 5 phút tính từ lúc nhận cảnh báo đến khi xuất bản tóm tắt lên Slack |
| 20 | Khả năng giải thích | Mọi giả thuyết đều phải đính kèm đường đi đồ thị và trích dẫn dữ liệu đo lường cụ thể |
| 15 | Tính toàn diện của tích hợp | Hoạt động trơn tru từ đầu đến cuối với PagerDuty, Slack, ArgoCD và Prometheus |
| **100** | | |

## Bài tập mở rộng (Exercises)

1. Khởi chạy tác tử trên cùng ba sự cố mà AWS DevOps Agent sử dụng trong bản demo chính thức. Xuất bản báo cáo đối chiếu song song. Phân tích rõ các điểm mà tác tử của bạn có hướng phân tích khác biệt.

2. Bổ sung cơ chế kiểm toán "suýt phạm lỗi" (near-miss audit) để gắn cờ bất kỳ câu lệnh nào mà tác tử *từng cân nhắc* có nguy cơ gây phá hủy nếu không có cơ chế phê duyệt. Đo lường tỷ lệ xuất hiện các lệnh suýt phạm lỗi này trong một tuần.

3. Hoán đổi mô hình sinh giả thuyết từ Claude Sonnet 4.7 sang mô hình tự lưu trữ Llama 3.3 70B. Đo lường mức độ biến động của độ chính xác RCA và chi phí tài chính trên từng sự cố.

4. Xây dựng bộ lọc quan hệ nhân quả (causal filter): Phân biệt giữa các xung đột biến đo lường có tính tương quan bề mặt với nguyên nhân gốc rễ thực sự. Huấn luyện một mô hình phân loại nhỏ dựa trên nhãn của 20 kịch bản sự cố.

5. Bổ sung tính năng chạy thử nghiệm hoàn tác (rollback dry-run): Tác tử kích hoạt lệnh rollback ArgoCD trên một cụm staging có cùng manifest. Xác minh tính an toàn của kế hoạch rollback trên môi trường kiểm thử trước khi gửi nút phê duyệt lên Slack.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| K8s knowledge graph (Đồ thị tri thức K8s) | "Đồ thị cụm" | Nút = các đối tượng K8s + chuỗi đo lường; cạnh = quan hệ sở hữu, điều phối lập lịch, quan sát |
| Read-only-by-default (Mặc định chỉ đọc) | "RBAC thu hẹp" | Tài khoản dịch vụ của tác tử chỉ có quyền get/list/describe; các thao tác phá hủy nằm ở máy chủ riêng có kiểm duyệt |
| Audit log (Nhật ký kiểm toán) | "Cân nhắc so với Thực thi" | Bản ghi chỉ ghi tiếp lưu lại mọi câu lệnh ứng viên, trạng thái thực thi và danh tính người phê duyệt |
| Hypothesis ranking (Xếp hạng giả thuyết) | "Điểm số bằng chứng" | Tính bằng: Độ tươi mới × độ cụ thể × nghịch đảo độ dài đường đi đồ thị × số lượng trích dẫn |
| Slack approval card (Thẻ phê duyệt Slack) | "Cổng người can thiệp (HITL)" | Tin nhắn Slack tương tác chứa các nút hành động; tác tử không thể tiếp tục nếu chưa có người nhấp chọn |
| Telemetry citation (Trích dẫn đo lường) | "Con trỏ bằng chứng" | Một truy vấn Prometheus, bộ lọc Loki, hoặc URL vết Tempo làm bằng chứng trực tiếp cho nhận định |
| MTTR | "Thời gian phục hồi" | Khoảng thời gian thực tế tính từ lúc cảnh báo kích hoạt đến khi chỉ số SLO trở lại trạng thái bình thường |

## Tài liệu đọc thêm (Further Reading)

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/mt/announcing-general-availability-of-aws-devops-agent/) — Tài liệu công bố chính thức năm 2026
- [Xử lý sự cố K8s trên Resolve AI](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — Tài liệu tham chiếu từ đối thủ cạnh tranh
- [Giám sát ngữ nghĩa NeuBird](https://www.neubird.ai) — Hướng tiếp cận dựa trên đồ thị ngữ nghĩa
- [Metoro AI SRE](https://metoro.io) — Mô hình vận hành sản xuất ưu tiên SLO
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — Nguồn dữ liệu trạng thái cụm tiêu chuẩn
- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — Framework điều phối tác tử tham chiếu
- [FastMCP](https://github.com/jlowin/fastmcp) — Framework máy chủ MCP bằng Python
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — Mục tiêu thực thi khắc phục sự cố có kiểm duyệt
