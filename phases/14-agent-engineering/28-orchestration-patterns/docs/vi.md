# Các Mẫu hình Điều phối: Giám sát, Bầy đàn, Phân cấp (Orchestration Patterns: Supervisor, Swarm, Hierarchical)

> Bốn mẫu hình điều phối lặp lại phổ biến xuyên suốt các khung làm việc (framework) năm 2026: người giám sát - công nhân (supervisor-worker), bầy đàn / ngang hàng (swarm / peer-to-peer), phân cấp (hierarchical) và tranh luận (debate). Lời khuyên then chốt từ Anthropic: "Thành công không nằm ở việc xây dựng hệ thống phức việt nhất, mà là xây dựng đúng hệ thống đáp ứng trọn vẹn nhu cầu của bạn." Hãy bắt đầu từ sự đơn giản; chỉ bổ sung cấu trúc liên kết phức tạp khi một tác tử đơn lẻ kết hợp năm mẫu luồng công việc không còn đủ năng lực xử lý.

**Loại bài học:** Tìm hiểu + Xây dựng (Learn + Build)  
**Ngôn ngữ:** Python (stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 12 (Các mẫu luồng công việc - Workflow Patterns), Phase 14 · 25 (Tranh luận Đa Tác tử - Multi-Agent Debate)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Nêu tên bốn mẫu hình điều phối lặp lại phổ biến và trường hợp phù hợp cho từng mẫu.
- Mô tả khuyến nghị năm 2026 của LangChain: cơ chế giám sát dựa trên lệnh gọi công cụ (tool-call-based supervision) so với việc sử dụng các thư viện giám sát chuyên dụng.
- Giải thích quy tắc "xây dựng đúng hệ thống" của Anthropic và cách quy tắc này quyết định việc lựa chọn cấu trúc điều phối.
- Cài đặt toàn bộ bốn mẫu hình bằng thư viện chuẩn (stdlib) vận hành trên cùng một mô hình LLM giả lập theo kịch bản.

## Vấn đề Đặt ra

Các nhóm phát triển thường vội vã tìm đến kiến trúc "đa tác tử (multi-agent)" trước khi thực sự có nhu cầu. Bốn mẫu hình cốt lõi xuất hiện lặp đi lặp lại trên khắp các framework; một khi bạn gọi tên và nắm vững bản chất của chúng, bạn có thể chọn đúng giải pháp tối ưu — hoặc tự tin bỏ qua việc thiết lập cấu trúc liên kết mạng phức tạp không cần thiết.

## Khái niệm

### Giám sát - Công nhân (Supervisor-worker)

- Một mô hình LLM định tuyến trung tâm (central router) sẽ phân phối nhiệm vụ tới các tác tử chuyên biệt.
- Quyết định luồng: lặp lại về chính mình, chuyển giao (hand off) cho tác tử chuyên trách, hoặc kết thúc phiên.
- Các tác tử chuyên biệt không giao tiếp trực tiếp với nhau; mọi luồng định tuyến đều phải đi qua người giám sát.

Các framework tiêu biểu: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**Khuyến nghị năm 2026 của LangChain:** triển khai giám sát thông qua các lệnh gọi công cụ trực tiếp (direct tool calls) thay vì phụ thuộc vào hàm trừu tượng `create_supervisor`. Cách này mang lại khả năng kiểm soát kỹ thuật ngữ cảnh (context engineering) chi tiết hơn nhiều — bạn toàn quyền quyết định chính xác những gì mỗi tác tử chuyên môn được phép nhìn thấy.

### Bầy đàn / Ngang hàng (Swarm / peer-to-peer)

- Các tác tử chuyển giao quyền xử lý trực tiếp cho nhau thông qua bề mặt công cụ dùng chung.
- Hoàn toàn không có bộ định tuyến trung tâm.
- Độ trễ thấp hơn so với mô hình giám sát (giảm thiểu số bước trung gian).
- Khó suy luận và gỡ lỗi hơn (thiếu một điểm kiểm soát tập trung).

Các framework tiêu biểu: Cấu trúc liên kết Swarm của LangGraph, cơ chế chuyển giao của OpenAI Agents SDK (khi mọi tác tử đều có quyền chuyển giao cho bất kỳ tác tử nào khác).

### Phân cấp (Hierarchical)

- Các cấp giám sát quản lý các cấp giám sát phụ, phía dưới mới là các tác tử công nhân thực thi.
- Được cài đặt dưới dạng các đồ thị con lồng nhau (nested subgraphs) trong LangGraph; hoặc các nhóm (crews) lồng nhau trong CrewAI.
- Mở rộng tốt cho quần thể tác tử có quy mô lớn nhưng phải trả giá bằng sự phức tạp trong vận hành.

Thời điểm cần dùng: khi dung lượng ngữ cảnh (context budget) của một người giám sát duy nhất không còn đủ chỗ chứa toàn bộ mô tả của tất cả các chuyên gia cấp dưới.

### Tranh luận (Debate)

- Các bên đề xuất chạy song song + phản biện chéo lặp lại nhiều vòng (Bài 25).
- Bản chất không thuần túy là điều phối nhiệm vụ — mà nghiêng về cơ chế xác thực và thẩm định — nhưng vẫn xuất hiện như một lựa chọn cấu trúc liên kết trong các framework.

### Nhóm tự trị so với Luồng tất định (Autonomous crews vs deterministic flows)

CrewAI chuẩn hóa hai phương thức triển khai thực tế:

- **Flow (Luồng):** dành cho việc tự động hóa dựa trên sự kiện có tính tất định cao (điểm khởi đầu khuyến nghị cho môi trường sản xuất).
- **Crew (Nhóm):** dành cho sự phối hợp tự trị dựa trên vai trò phân công.

Sự phân chia này trực giao với bốn mẫu hình ở trên nhưng ánh xạ chặt chẽ vào cấu trúc liên kết: Flow thường là dạng giám sát hoặc phân cấp có kiểm soát; Crew thường là dạng giám sát sử dụng bộ định tuyến LLM tự do.

### Lời khuyên từ Anthropic

"Thành công trong không gian LLM không nằm ở việc xây dựng hệ thống tinh vi, phức tạp nhất. Nó nằm ở việc xây dựng đúng hệ thống đáp ứng chính xác nhu cầu của bạn."

Thứ tự quyết định kiến trúc:

1. Một tác tử đơn lẻ + các mẫu luồng công việc (Bài 12) — luôn bắt đầu từ đây.
2. Giám sát - công nhân — khi bạn có từ 2 đến 4 tác tử chuyên môn rõ ràng.
3. Bầy đàn (Swarm) — khi độ trễ quan trọng hơn sự minh bạch trong kiểm soát suy luận.
4. Phân cấp (Hierarchical) — chỉ áp dụng khi ngân sách ngữ cảnh của bộ giám sát trung tâm bị tràn.
5. Tranh luận (Debate) — khi độ chính xác tuyệt đối quan trọng hơn chi phí token.

### Những điểm dễ sai lầm của mẫu hình này

- **Tư duy ưu tiên cấu trúc mạng (Topology-first thinking):** Khăng khăng "chúng ta cần hệ thống đa tác tử" trước khi xác định rõ đa tác tử sẽ giải quyết bài toán cụ thể nào.
- **Hiện tượng ném bóng qua lại (Bouncing handoffs trong Swarm):** Tác tử A -> B -> A -> B lặp vô tận. Bắt buộc phải sử dụng bộ đếm bước chuyển giao (hop counter).
- **Phân cấp giả tạo (Fake hierarchy):** Thiết kế ba tầng phức tạp chỉ vì mác "doanh nghiệp (enterprise)", trong khi thực tế chỉ có hai nhóm chức năng. Hãy dẹp bỏ và tinh giản.

```figure
orchestration-pattern
```

## Tự Xây dựng (Build It)

`code/main.py` cài đặt toàn bộ bốn mẫu hình bằng thư viện chuẩn (stdlib) trên một LLM giả lập theo kịch bản:

- `Supervisor` — bộ định tuyến trung tâm.
- `Swarm` — giao tiếp ngang hàng với cơ chế chuyển giao trực tiếp.
- `Hierarchical` — hệ thống giám sát của các giám sát viên.
- `Debate` — các bên đề xuất song song kết hợp phản biện.

Mỗi mẫu hình sẽ xử lý cùng một tác vụ gồm ba ý định người dùng (hoàn tiền / báo lỗi / bán hàng). Cấu trúc dấu vết thực thi (trace) giữa chúng sẽ hoàn toàn khác biệt.

Chạy thử mã nguồn:

```bash
python3 code/main.py
```

Kết quả: dấu vết thực thi + số lượng thao tác theo từng mẫu hình. Giám sát là mô hình rõ ràng, tường minh nhất; Swarm có hành trình ngắn nhất; Phân cấp có độ sâu lớn nhất; Tranh luận tiêu tốn nhiều chi phí nhất.

## Ứng dụng Thực tế (Use It)

- **LangGraph** cho mô hình giám sát và phân cấp (sử dụng đồ thị con lồng nhau).
- **OpenAI Agents SDK** cho mô hình chuyển giao dưới dạng công cụ (mang dáng dấp giám sát).
- **CrewAI Flow** cho các luồng công việc tất định trên môi trường sản xuất.
- **Tùy chỉnh (Custom)** cho mô hình tranh luận hoặc khi bạn cần kiểm soát chính xác từng bước xử lý.

## Chuyển giao Đóng gói (Ship It)

`outputs/skill-orchestration-picker.md` giúp phân tích và lựa chọn cấu trúc liên kết phù hợp, đồng thời khởi tạo mã nguồn triển khai.

## Bài tập Thực hành

1. Chuyển đổi một hệ thống giám sát - công nhân thành một hệ thống Swarm bằng cách loại bỏ bộ định tuyến trung tâm. Điểm gì bị đổ vỡ? Điểm gì được cải thiện?
2. Bổ sung bộ đếm bước chuyển giao (hop counter) vào Swarm: từ chối xử lý sau 3 lần chuyển giao liên tiếp. Cơ chế này có chặn đứng hiện tượng luẩn quẩn A->B->A không?
3. Xây dựng một hệ thống phân cấp 2 tầng cho miền nghiệp vụ có 12 tác tử chuyên gia. Nếu không dùng kỹ thuật lồng ghép phân tầng, ngân sách ngữ cảnh sẽ bị quá tải ở điểm nào?
4. Đánh giá hiệu năng của cả bốn mẫu hình trên một khối lượng công việc thực tế. Mẫu nào chiến thắng ở từng chỉ số (độ trễ, chi phí, độ chính xác, khả năng gỡ lỗi)?
5. Đọc bài viết "Building Effective Agents" của Anthropic. Ánh xạ từng luồng công việc thực tế của bạn vào một trong bốn mẫu hình. Có trường hợp nào không thể ánh xạ gọn gàng không?

## Thuật ngữ Then chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|---|---|---|
| Giám sát - công nhân (Supervisor-worker) | "Bộ định tuyến + chuyên gia" | Mô hình LLM trung tâm phân phối việc cho chuyên gia; các chuyên gia không trao đổi trực tiếp |
| Bầy đàn (Swarm) | "Mạng ngang hàng (Peer-to-peer)" | Chuyển giao trực tiếp qua công cụ dùng chung; không có bộ định tuyến trung tâm |
| Phân cấp (Hierarchical) | "Giám sát của giám sát" | Các đồ thị con lồng nhau phục vụ quần thể tác tử quy mô lớn |
| Tranh luận (Debate) | "Đề xuất + phản biện" | Nhiều bên đề xuất song song, phản biện chéo nhau (Bài 25) |
| Giám sát bằng lệnh gọi công cụ | "Giám sát không cần thư viện" | Cài đặt bộ giám sát thông qua direct tool call nhằm kiểm soát chặt ngữ cảnh |
| Nhóm tác tử (Crew) | "Đội ngũ tự trị" | Chế độ cộng tác dựa trên vai trò phân công của CrewAI |
| Luồng tất định (Flow) | "Quy trình tất định" | Chế độ chạy theo sự kiện hướng sản xuất của CrewAI |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — năm mẫu hình cốt lõi + so sánh tác tử và luồng công việc
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — các mẫu supervisor, swarm, hierarchical
- [Tài liệu CrewAI](https://docs.crewai.com/en/introduction) — so sánh giữa Crew và Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — mẫu hình tranh luận đa tác tử
