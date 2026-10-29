# Mở rộng trong Môi trường Sản xuất — Hàng đợi, Điểm kiểm tra và Tính bền bỉ (Production Scaling — Queues, Checkpoints, Durability)

> Việc mở rộng quy mô hệ thống đa tác tử lên hàng ngàn lượt chạy đồng thời đòi hỏi phải có **cơ chế thực thi bền bỉ (durable execution)** — kết hợp hàng đợi công việc cùng các điểm kiểm tra trạng thái (checkpoints), nhờ đó bất kỳ tiến trình công nhân (worker) nào cũng có thể khôi phục phiên chạy sau sự cố sụp đổ (crash), miễn là có sẵn cơ chế quản lý thời hạn ủy quyền (leases), các tác dụng phụ mang tính lũy đẳng (idempotent side effects) và khả năng tái phát lại tất định (deterministic replay). Môi trường chạy của LangGraph là ví dụ kinh điển: nó ghi một điểm kiểm tra sau mỗi bước siêu cấp (super-step) được định danh bằng `thread_id` (mặc định lưu trên Postgres); khi một worker bị sập, hệ thống giải phóng lease và một worker khác sẽ tiếp tục tiến trình. Các tác tử có thể "ngủ" vô thời hạn để chờ đầu vào từ con người. Nghiên cứu **MegaAgent** (arXiv:2408.09955) đã vận hành hàng đợi sản xuất - tiêu thụ (producer-consumer queue) cho từng tác tử với ba trạng thái (Rảnh rỗi - Idle / Đang xử lý - Processing / Phản hồi - Response) cùng sự phối hợp hai tầng (trò chuyện nội bộ nhóm + trò chuyện quản trị viên liên nhóm). Cơ chế **Fiber/Bất đồng bộ (async)** vượt trội hoàn toàn so với mô hình một luồng trên một tác vụ (thread-per-job) đối với luồng truyền dữ liệu (streaming) của LLM: các luồng hệ điều hành thường nằm yên nhàn rỗi tới 99% thời gian để chờ token, trong khi các fiber nhường quyền thực thi một cách hợp tác trong lúc chờ I/O. Góc nhìn phản biện: bài viết "Mở rộng phần mềm tác tử" của Ashpreet Bedi lập luận rằng chỉ cần **FastAPI + Postgres và không cần gì thêm** cho đến khi tải thực tế chứng minh điều ngược lại — các kiến trúc tinh giản thường đi xa hơn nhiều so với dự kiến. Bài học này xây dựng một nhật ký điểm kiểm tra bền vững, hàng đợi công việc cho từng tác tử với chuyển đổi trạng thái, thử nghiệm so sánh giữa async và đa luồng, đồng thời xác lập quy tắc thực dụng "hãy bắt đầu từ sự đơn giản".

**Type:** Learn + Build  
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)  
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)  
**Time:** ~75 phút  

## Vấn đề (Problem)

Một hệ thống đa tác tử nguyên mẫu hoạt động trơn tru trên một chiếc máy tính xách tay với ba tác tử chạy trong một vòng lặp sự kiện lưu trong bộ nhớ (in-memory event loop). Khi bạn chuyển sang môi trường sản xuất (production):

- Các tác tử đôi khi phải chạy liên tục hàng giờ liền (nghiên cứu chuyên sâu, chờ phản hồi từ con người trong vòng lặp - human-in-the-loop).
- Các tiến trình công nhân (worker processes) có thể bị sập bất ngờ. Khởi động lại sẽ làm mất toàn bộ trạng thái.
- Tải đỉnh điểm gấp 10 lần tải trung bình; bạn bắt buộc phải mở rộng quy mô theo chiều ngang (horizontal scaling).
- Người dùng trả phí cho mỗi lượt chạy của tác tử; bạn cần ngữ nghĩa đúng-một-lần (exactly-once semantics) để tính phí chính xác.

Vòng lặp sự kiện trong bộ nhớ hoàn toàn bất lực trước những đòi hỏi này. Bạn cần một tầng thực thi bền bỉ (durable execution layer) bên dưới. Vào năm 2026, các lựa chọn tiêu chuẩn bao gồm:

1. Một công cụ quy trình làm việc có tích hợp điểm kiểm tra (Temporal, runtime của LangGraph).
2. Một hàng đợi tin nhắn kết hợp với kho lưu trữ trạng thái (Postgres + SQS/RabbitMQ).
3. Các khung kiến trúc mô hình tác tử hướng đối tượng Actor (như mô hình hàng đợi producer-consumer cho từng tác tử của MegaAgent).
4. Kiến trúc tự xây dựng với FastAPI + Postgres (theo lập luận của Bedi).

Bài học này sẽ xây dựng một phiên bản thu nhỏ của từng cách tiếp cận.

## Khái niệm (Concept)

### Mẫu hình Thực thi Bền bỉ (Durable Execution)

Một cỗ máy thực thi bền bỉ sẽ lưu trữ toàn bộ trạng thái chương trình sau mỗi "bước" (được gọi là bước siêu cấp - super-step trong thuật ngữ của LangGraph). Khi xảy ra sự cố sụp đổ:

```text
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Các điều kiện tiên quyết để cơ chế này vận hành trơn tru:

- **Trạng thái có thể tuần tự hóa (Serializable state):** Toàn bộ trạng thái của tác tử phải có khả năng lưu trữ bền vững. Các bao đóng hàm (function closures) chứa kết nối cơ sở dữ liệu còn sống sẽ không thể tồn tại qua sự cố.
- **Khôi phục tất định (Deterministic resume):** Khi nhận cùng một trạng thái và cùng các đầu vào, tác tử phải sinh ra các hành động giống hệt nhau (hoặc dựa vào một cỗ máy sấm truyền - deterministic oracle ghi nhận các lượt gọi LLM trước đó).
- **Tác dụng phụ lũy đẳng (Idempotent side effects):** Các lệnh gọi ra bên ngoài (gọi công cụ, thực hiện giao dịch thanh toán) phải có tính lũy đẳng hoặc sử dụng khóa chống trùng lặp (deduplication key).

LangGraph ghi một điểm kiểm tra sau mỗi super-step; Temporal ghi lại sau mỗi hoạt động (activity); Restate sử dụng nhật ký nguồn sự kiện (event-sourced journals). Cả ba đều hiện thực hóa cùng một mẫu hình tư duy này.

### Môi trường chạy có Điểm kiểm tra trên từng Bước (Checkpoint-per-step Runtime)

Môi trường chạy của LangGraph là ví dụ điển hình: mỗi tác tử có một `thread_id`; trạng thái là một từ điển định kiểu (typed dict); mỗi super-step ghi một bản ghi vào bảng checkpoints. Khi khôi phục, hệ thống tái phát lại từ điểm kiểm tra gần nhất thay vì chạy lại từ đầu. Các tác tử có thể gọi lệnh `interrupt()` để tạm dừng chờ người dùng nhập liệu; runtime sẽ lưu trạng thái vào ổ đĩa và giải phóng worker. Khi có dữ liệu mới, bất kỳ worker nào đang rảnh đều có thể tiếp tục tiến trình.

Đây là thiết kế sản xuất chuẩn mực vào tháng 4 năm 2026.

### Hàng đợi cho từng Tác tử của MegaAgent

Nghiên cứu arXiv:2408.09955 mô tả một thử nghiệm quy mô lớn: hàng ngàn tác tử hoạt động đồng thời trên một cụm máy chủ. Kiến trúc hệ thống:

```text
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

Cơ chế phối hợp hai tầng cho phép các cuộc trò chuyện nội bộ nhóm diễn ra với mật độ dày đặc trong khi các trao đổi liên nhóm duy trì ở mức thưa thớt — đây là mẫu hình giúp giữ cho chi phí tính toán tăng tuyến tính thay vì bùng nổ hàm mũ khi số lượng tác tử lên tới hàng ngàn.

### Bất đồng bộ (Async) so với Một luồng cho mỗi Tác vụ (Thread-per-job)

Các lượt gọi LLM hoàn toàn bị nghẽn ở khâu I/O (I/O-bound). Một luồng hệ điều hành phải chờ token tiếp theo sẽ ở trạng thái nhàn rỗi tới 99% thời gian. Mỗi luồng tiêu tốn khoảng ~1MB bộ nhớ RAM; với 10.000 lượt gọi đồng thời, bạn sẽ mất tới 10GB chỉ dành riêng cho ngăn xếp (stacks).

Các sợi nhẹ (fibers / coroutines - như `asyncio` trong Python, goroutines trong Go, `tokio` trong Rust) chủ động nhường quyền thực thi (cooperatively yield) trong lúc chờ I/O. Cùng 10.000 lượt gọi đó có thể nằm vừa vặn trong bộ nhớ của một tiến trình đơn lẻ. Ở quy mô hệ tác tử LLM, lập trình bất đồng bộ không còn là một biện pháp tối ưu hóa thông thường — nó là kiến trúc nền tảng bắt buộc.

Ngoại lệ: Các tác vụ hậu xử lý tiêu tốn nhiều CPU (CPU-bound) như nhúng vector hoặc xử lý phân tách từ (tokenizer) vẫn cần đến luồng hoặc tiến trình riêng biệt. Hãy luôn tách biệt tầng I/O khỏi tầng tính toán CPU.

### Luận điểm phản biện của Ashpreet Bedi

Bài viết "Scaling Agentic Software" (Ashpreet Bedi, 2026) cho rằng phần lớn các đội ngũ kỹ thuật đều thiết kế quá mức (over-engineer) trước khi đo đạc tải thực tế. Lựa chọn mặc định mang tính thực dụng:

- Sử dụng FastAPI + Postgres.
- Mỗi phiên chạy tác tử là một hàng dữ liệu; trạng thái được cập nhật trực tiếp tại chỗ với cơ chế khóa lạc quan (optimistic concurrency).
- Xử lý các tác vụ ngầm thông qua `pg_notify` hoặc một worker Celery đơn giản.
- Xử lý chính sách thử lại (retry policy) ngay trong mã ứng dụng.

Với mức tải dưới ~100 phiên chạy tác tử đồng thời cho các tác vụ thông thường, kiến trúc này hoàn toàn đáp ứng tốt nhu cầu. Bạn chỉ nên nâng cấp khi các số liệu giám sát thực tế cho thấy hệ thống bắt đầu quá tải.

Quy tắc cốt lõi: Chỉ sử dụng các khung kiến trúc thực thi bền bỉ phức tạp khi bạn gặp phải bài toán cụ thể mà kiến trúc tinh giản không thể giải quyết được. Việc áp dụng sớm chỉ làm lãng phí thời gian vào những thủ tục không đem lại giá trị thiết thực.

### Ngữ nghĩa Đúng-Một-Lần (Exactly-once Semantics)

Đối với các phiên chạy tác tử có tính phí, bạn cần đạt được mức "hiệu lực đúng-một-lần" (phân phối ít-nhất-một-lần + bộ tiêu thụ lũy đẳng - at-least-once delivery + idempotent consumer). Các kỹ thuật triển khai:

- **Khóa chống trùng lặp (Dedup key) cho mỗi lượt chạy:** Luôn đính kèm khóa này vào mọi lệnh gọi tác dụng phụ.
- **Mẫu hình Hộp thư đi (Outbox pattern):** Các tác dụng phụ được ghi vào một bảng cơ sở dữ liệu trước, sau đó một tiến trình độc lập sẽ quét bảng và thực thi chúng. Cả hai bước đều có tính lũy đẳng.
- **Giao dịch bù trừ (Compensating transactions):** Khi một tác dụng phụ thực hiện thành công nhưng thao tác ghi nhận kết quả thất bại, hãy lên lịch cho một giao dịch bù trừ để hoàn nguyên.

Đây là các mẫu hình kỹ thuật cơ sở dữ liệu kinh điển chứ không riêng gì LLM. Điểm đặc thù của LLM chỉ là các lượt gọi diễn ra chậm; còn lại mọi nguyên lý đều tuân theo các hệ thống phân tán tiêu chuẩn.

### Chiến lược Triển khai Cầu vồng (Rainbow Deployment)

Hệ thống nghiên cứu đa tác tử của Anthropic sử dụng "triển khai cầu vồng" (rainbow deployments): nhiều phiên bản của môi trường chạy tác tử được vận hành đồng thời, nhờ đó các tác tử chạy đường dài không bao giờ bị tiêu diệt mỗi khi triển khai mã mới. Triển khai thử nghiệm (canary) các phiên bản mới trên một phần lưu lượng; chỉ khai tử các phiên bản cũ khi các tác tử của chúng đã hoàn thành toàn bộ công việc.

Đây là kỹ thuật tiêu chuẩn cho các hệ thống có trạng thái chạy lâu; sự điều chỉnh trong năm 2026 là các tác tử có thể sống hàng giờ liền, vì vậy chu kỳ triển khai phần mềm phải thích ứng linh hoạt.

### Danh mục kiểm tra chuẩn hóa cho Môi trường Sản xuất

- Trạng thái bền bỉ (điểm kiểm tra, bản chụp nhanh snapshot, hoặc outbox + nhật ký tái phát).
- Tác dụng phụ có tính lũy đẳng.
- Tầng I/O bất đồng bộ dành riêng cho các lượt gọi LLM.
- Cơ chế gửi ít-nhất-một-lần kết hợp chống trùng lặp.
- Triển khai dạng cầu vồng hoặc canary cho các khối lượng công việc có trạng thái.
- Khả năng quan sát (observability): dấu vết truy vết (traces) của từng tác tử, kiểm toán từng super-step, bộ đếm số lần thử lại.

````figure
sw-checkpoint-replay
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa:

- `CheckpointStore` — Nhật ký điểm kiểm tra lưu trữ trên SQLite với khóa `thread_id`. Mỗi super-step sẽ ghi thêm một dòng dữ liệu.
- `run_with_checkpoint(agent, thread_id)` — Mô phỏng tình huống tiến trình bị sập giữa chừng; một worker thứ hai sẽ tiếp tục công việc từ điểm kiểm tra gần nhất.
- `AgentQueue` — Cỗ máy trạng thái (Rảnh rỗi / Đang xử lý / Phản hồi) cho từng tác tử đi kèm hàng đợi tác vụ nhỏ.
- `demo_async_vs_threads()` — Chạy đồng thời 500 "lượt gọi LLM" giả lập bằng `asyncio` và bằng các luồng (threads); đo đạc thời gian thực tế và bộ nhớ đỉnh (ước tính).

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả kỳ vọng: Quá trình khôi phục điểm kiểm tra thành công sau sự cố sập giả lập; phiên bản async xử lý xong 500 cuộc gọi đồng thời trong vòng chưa đầy 1 giây; phiên bản đa luồng mất vài giây và tiêu tốn lượng bộ nhớ lớn hơn nhiều lần trên mỗi đơn vị xử lý đồng thời.

## Ứng dụng thực tế (Use It)

Tài liệu `outputs/skill-scaling-advisor.md` cung cấp lời khuyên lựa chọn kiến trúc thực thi bền bỉ: FastAPI + Postgres, runtime của LangGraph, Temporal, hoặc tự xây dựng. Được hiệu chỉnh dựa trên mức tải, yêu cầu lưu giữ trạng thái và tần suất triển khai hệ thống.

## Triển khai thực tế (Ship It)

Các nguyên tắc làm cứng hệ thống cho môi trường sản xuất:

- **Bắt đầu từ sự đơn giản (Quy tắc của Bedi):** Giữ FastAPI + Postgres cho đến khi bạn đo đạc thấy nó thực sự quá tải.
- **Gắn thiết bị đo lường mọi thứ trước khi tối ưu hóa:** Biểu đồ tần suất độ trễ từng lượt chạy, thời gian thực thi từng bước, số lần thử lại, phân loại nguyên nhân lỗi.
- **Sử dụng mẫu hình outbox cho các tác dụng phụ:** Đặc biệt là các giao dịch thanh toán và các lệnh gọi API bên ngoài.
- **Áp dụng triển khai cầu vồng:** Không bao giờ tiêu diệt các tiến trình tác tử đang thực thi dang dở trong lúc cập nhật phiên bản ứng dụng.
- **Chỉ áp dụng các công cụ thực thi bền bỉ (Temporal / LangGraph / Restate) khi** bạn gặp phải những thách thức cụ thể: thời gian chờ con người kéo dài hàng giờ, điều phối xuyên vùng địa lý, hoặc chính sách thử lại/bù trừ quá phức tạp.
- **Dùng Async cho tầng I/O:** Chỉ dùng đa luồng cho các công việc hậu xử lý nặng về tính toán CPU.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Xác minh rằng cơ chế khôi phục từ điểm kiểm tra hoạt động chuẩn xác; đo đạc sự khác biệt về năng lực xử lý đồng thời giữa async và đa luồng.
2. Hiện thực hóa một bảng **outbox**: mọi lệnh gọi công cụ đều được ghi vào bảng outbox trước, sau đó một tác vụ độc lập sẽ thực thi. Kiểm tra tính lũy đẳng bằng cách kích hoạt chạy lệnh gọi công cụ đó hai lần liên tiếp.
3. Mô phỏng một phiên **triển khai cầu vồng**: chạy đồng thời hai phiên bản runtime; điều phối một nửa số `thread_id` mới tới mỗi phiên bản; chứng minh rằng các luồng đang chạy trên phiên bản cũ không bị gián đoạn.
4. Đọc tài liệu về môi trường chạy của LangGraph (liên kết bên dưới). Xác định những tính năng nào của runtime sẽ tốn nhiều thời gian nhất nếu bạn tự viết lại bằng FastAPI + Postgres. Đó có phải là lý do đủ lớn để bạn sử dụng thư viện có sẵn hay bạn có thể tạm hoãn lại?
5. Đọc Mục 3 bài báo MegaAgent (arXiv:2408.09955). Cơ chế phối hợp hai tầng (trò chuyện nội bộ nhóm + trò chuyện quản trị viên liên nhóm) được trình bày tường minh. Hãy phác thảo cách bạn ánh xạ thiết kế này vào một hệ thống hàng đợi tin nhắn với hai họ hàng đợi riêng biệt.

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| Thực thi bền bỉ (Durable execution) | "Lưu bền vững trạng thái chương trình" | Cỗ máy ghi lại trạng thái sau mỗi bước siêu cấp; quá trình khôi phục sau sự cố diễn ra một cách tất định. |
| Bước siêu cấp (Super-step) | "Ranh giới giao dịch" | Đơn vị công việc tối thiểu nằm giữa hai điểm kiểm tra. Thuật ngữ trong LangGraph. |
| `thread_id` | "Mã định danh phiên chạy tác tử" | Khóa ràng buộc các điểm kiểm tra với logic khôi phục tương ứng. |
| Tính lũy đẳng (Idempotency) | "An toàn khi thử lại" | Việc thực thi lặp lại một tác dụng phụ nhiều lần mang lại kết quả giống hệt như khi chỉ thực hiện một lần. |
| Mẫu hình Outbox (Outbox pattern) | "Tách rời tác dụng phụ" | Ghi ý định vào một bảng dữ liệu; một tiến trình xử lý riêng biệt sẽ quét và đánh dấu hoàn tất. |
| Phân phối ít-nhất-một-lần (At-least-once delivery) | "Có thể bị trùng lặp" | Ngữ nghĩa của hàng đợi tin nhắn; khóa chống trùng lặp giúp bên tiêu thụ đạt hiệu quả tương đương đúng-một-lần. |
| Triển khai cầu vồng (Rainbow deploy) | "Các phiên bản gối đầu nhau" | Nhiều phiên bản môi trường chạy cùng hoạt động song song trong suốt thời gian thực thi các tác vụ đường dài. |
| Async fiber / coroutine | "Nhường quyền hợp tác" | Cơ chế đồng thời ở không gian người dùng; tiêu tốn rất ít tài nguyên so với luồng hệ điều hành khi xử lý tải nghẽn I/O. |
| Điểm kiểm tra (Checkpoint) | "Bản chụp trạng thái" | Trạng thái đã được tuần tự hóa tại ranh giới của một bước siêu cấp; đóng vai trò cốt lõi cho việc khôi phục. |

## Tài liệu tham khảo (Further Reading)

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — Thiết kế kiến trúc runtime của LangGraph.
- [MegaAgent](https://arxiv.org/abs/2408.09955) — Hàng đợi producer-consumer cho từng tác tử; điều phối hai tầng cho hàng ngàn tác tử đồng thời.
- [Matrix](https://arxiv.org/abs/2511.21686) — Khung kiến trúc phi tập trung lấy hàng đợi tin nhắn làm chất nền phối hợp.
- [Temporal docs](https://docs.temporal.io/) — Cỗ máy quy trình làm việc chuẩn mực cho việc thực thi bền bỉ.
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Các bài học thực tiễn trong sản xuất bao gồm chiến lược triển khai cầu vồng.
