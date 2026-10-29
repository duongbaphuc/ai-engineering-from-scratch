# Điều phối Đồ thị có Trạng thái — Thực thi Bền vững và Điểm kiểm tra

> Tác tử là một máy trạng thái (state machine); các nút là hàm; các cạnh là bước chuyển tiếp; trạng thái được lưu điểm kiểm tra (checkpoint) sau mỗi nút. Khôi phục từ bất kỳ sự cố nào tại điểm kiểm tra thành công gần nhất. LangGraph là tài liệu tham chiếu năm 2026 cho mô hình điều phối có trạng thái cấp thấp này.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Mục tiêu học tập

- Mô tả mô hình cốt lõi của LangGraph: máy trạng thái với trạng thái được định kiểu (typed state), các nút hàm (function nodes), các cạnh có điều kiện (conditional edges), và các điểm kiểm tra sau mỗi nút (post-node checkpoints).
- Gọi tên bốn năng lực được tài liệu nhấn mạnh: thực thi bền vững (durable execution), truyền phát luồng dữ liệu (streaming), con người can thiệp vào vòng lặp (human-in-the-loop), và bộ nhớ toàn diện (comprehensive memory).
- Giải thích ba mô hình tô-pô điều phối (orchestration topologies) mà LangGraph hỗ trợ: giám sát viên (supervisor), mạng ngang hàng (peer-to-peer / swarm), phân cấp (hierarchical / nested subgraphs).
- Triển khai một đồ thị trạng thái bằng thư viện chuẩn (stdlib) với trạng thái định kiểu, các cạnh có điều kiện, và chu trình lưu/khôi phục điểm kiểm tra (checkpoint/resume cycle).

## Vấn đề

Các tác tử và quy trình làm việc (workflow) có chung một vấn đề: khi một lượt chạy 40 bước gặp lỗi ở bước 38, bạn muốn khôi phục và tiếp tục từ bước 38 chứ không phải bắt đầu lại từ đầu. Các mô hình trạng thái hạng hai (second-class state models) buộc người vận hành phải tự chắp vá các cơ chế thử lại (retry) xung quanh một thư viện vốn mặc định mỗi lượt chạy đều mới tinh từ đầu.

Câu trả lời trong thiết kế của LangGraph: trạng thái là một đối tượng hạng nhất được định kiểu (first-class typed object), các phép biến đổi trạng thái (mutations) là tường minh, và các điểm kiểm tra được lưu bền vững sau mỗi nút. Việc khôi phục (resume) đơn giản chỉ là một lệnh gọi `load_state(session_id)`.

## Khái niệm

### Đồ thị

Một đồ thị được định nghĩa bởi:

- **Kiểu trạng thái (State type).** Một từ điển có định kiểu (typed dict hoặc mô hình Pydantic) mà mọi nút đều đọc và biến đổi.
- **Các nút (Nodes).** Các hàm thuần túy `(state) -> state_update`. Các cập nhật sẽ được hợp nhất vào trạng thái sau khi hàm trả về.
- **Các cạnh (Edges).** Các bước chuyển tiếp có điều kiện hoặc trực tiếp giữa các nút.
- **Điểm vào và điểm ra (Entry and exit).** Các nút lính canh (sentinel nodes) `START` và `END` đánh dấu ranh giới đồ thị.

Ví dụ: một tác tử với các nút `classify` (phân loại), `refund` (hoàn tiền), `bug` (báo lỗi), `sales` (bán hàng), `done` (hoàn thành) — một quy trình định tuyến được biểu diễn dưới dạng đồ thị.

### Thực thi bền vững

Sau khi mỗi nút trả về kết quả, môi trường thực thi (runtime) sẽ tuần tự hóa (serialize) trạng thái và ghi nó vào một bộ kiểm tra điểm lưu (checkpointer — SQLite, Postgres, Redis hoặc tùy chỉnh). Khi xảy ra lỗi tại bước N, runtime có thể gọi `resume(session_id)` và tiếp tục từ bước N+1 với trạng thái chuẩn xác nguyên vẹn.

Tài liệu LangGraph nêu bật các khách hàng môi trường sản xuất coi trọng tính năng này: Klarna, Uber, J.P. Morgan. Giá trị cốt lõi không chỉ nằm ở cấu trúc đồ thị; mà là cấu trúc đồ thị kết hợp với cơ chế lưu điểm kiểm tra giúp chi phí phục hồi trở nên vô cùng rẻ.

### Truyền phát luồng dữ liệu

Mỗi nút có thể trả về từng phần đầu ra (partial output). Đồ thị truyền phát các sự kiện vi sai theo từng nút (per-node-delta events) về phía bên gọi để giao diện người dùng (UI) cập nhật theo thời gian thực khi đồ thị đang chạy.

### Con người can thiệp vào vòng lặp

Kiểm tra và chỉnh sửa trạng thái giữa các nút. Các cách triển khai: tạm dừng trước một nút trọng yếu, hiển thị trạng thái cho con người xem xét, tiếp nhận các chỉnh sửa, và tiếp tục thực thi. Bộ kiểm tra điểm lưu giúp điều này trở nên dễ dàng vì trạng thái đã được tuần tự hóa sẵn.

### Bộ nhớ

Ngắn hạn (trong một lượt chạy — lịch sử hội thoại nằm trong trạng thái) và dài hạn (xuyên suốt các lượt chạy — lưu bền vững thông qua bộ kiểm tra điểm lưu kết hợp với một kho lưu trữ dài hạn riêng biệt). LangGraph tích hợp với các hệ thống bộ nhớ bên ngoài (Mem0, tùy biến) thông qua các công cụ (tools).

### Ba mô hình tô-pô

1. **Giám sát viên (Supervisor).** LLM định tuyến trung tâm sẽ điều phối nhiệm vụ tới các tác tử phụ chuyên trách (specialist subagents). `create_supervisor()` trong `langgraph-supervisor` (mặc dù đội ngũ LangChain vào năm 2026 khuyến nghị thực hiện việc này trực tiếp thông qua các lệnh gọi công cụ để kiểm soát ngữ cảnh tốt hơn).
2. **Bầy đàn / Ngang hàng (Swarm / peer-to-peer).** Các tác tử bàn giao công việc trực tiếp thông qua một bề mặt công cụ dùng chung. Không có bộ định tuyến trung tâm.
3. **Phân cấp (Hierarchical).** Các giám sát viên quản lý các giám sát viên cấp dưới, được triển khai dưới dạng các đồ thị con lồng nhau (nested subgraphs).

### Những cạm bẫy thường gặp

- **Điểm kiểm tra quá nhỏ.** Chỉ lưu điểm kiểm tra cho các lượt hội thoại sẽ khiến trạng thái công cụ và dữ liệu ghi vào bộ nhớ không thể khôi phục được. Toàn bộ trạng thái phải được tuần tự hóa.
- **Các nút không xác định (Non-deterministic nodes).** Cơ chế khôi phục giả định rằng các đầu vào của nút luôn tạo ra cùng một cập nhật trạng thái. Seed ngẫu nhiên, đồng hồ hệ thống, các API bên ngoài cần phải được ghi nhận và cố định.
- **Lạm dụng các cạnh có điều kiện.** Một đồ thị mà mọi cạnh đều có điều kiện sẽ biến thành một máy trạng thái khó có thể suy luận logic. Hãy ưu tiên các chuỗi tuyến tính kết hợp với các nhánh rẽ khi cần thiết.

```figure
langgraph-state
```

## Xây dựng

`code/main.py` triển khai một đồ thị có trạng thái chỉ dùng thư viện chuẩn (stdlib):

- `State` — một từ điển định kiểu với `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — hàm có thể gọi (callable) nhận trạng thái và trả về một từ điển cập nhật.
- `StateGraph` — các nút + các cạnh + các cạnh có điều kiện + chạy + khôi phục.
- `SQLiteCheckpointer` (bản giả lập trong bộ nhớ) — tuần tự hóa trạng thái sau mỗi nút; `load(session_id)` thực hiện khôi phục.
- Đồ thị minh họa: classify -> branch(refund / bug / sales) -> human gate -> send.

Chạy thử:

```bash
python3 code/main.py
```

Dấu vết thực thi (trace) cho thấy lượt chạy đầu tiên thất bại tại cổng can thiệp của con người, lưu bền vững trạng thái, sau đó khôi phục và tạo ra đầu ra cuối cùng.

## Sử dụng

- **LangGraph** — tài liệu tham chiếu chuẩn, sẵn sàng cho môi trường sản xuất. Sử dụng `create_react_agent`, `create_supervisor`, hoặc tự xây dựng đồ thị của riêng bạn.
- **AutoGen v0.4** (Bài 14) — giải pháp thay thế dựa trên mô hình Actor (actor model) cho các kịch bản có độ đồng thời cao (high-concurrency).
- **Claude Agent SDK** (Bài 17) — khung khai thác được quản lý với kho phiên (session store) tích hợp sẵn.
- **Tùy biến (Custom)** — khi bạn cần kiểm soát chính xác cấu trúc trạng thái hoặc backend của bộ kiểm tra điểm lưu.

## Chuyển giao

`outputs/skill-state-graph.md` tạo ra một đồ thị trạng thái theo phong cách LangGraph trong bất kỳ môi trường thực thi đích nào với cơ chế lưu và khôi phục điểm kiểm tra được tích hợp sẵn.

## Bài tập

1. Thêm một cạnh có điều kiện từ `classify` tới `end` khi độ tin cậy phân loại dưới một ngưỡng nhất định. Tiếp tục lượt chạy sau khi con người thiết lập `route` thủ công.
2. Thay thế bản giả lập SQLite bằng một bộ kiểm tra điểm lưu SQLite thật. Đo lường chi phí tuần tự hóa (serialization overhead) ở mỗi bước.
3. Triển khai các cạnh song song: hai nút chạy đồng thời, hợp nhất kết quả bằng một hàm giảm (reducer) tùy chỉnh. Tính bất biến của trạng thái mang lại lợi ích gì ở đây?
4. Đọc tài liệu tham khảo `langgraph-supervisor`. Chuyển mã ví dụ sang dùng `create_supervisor`. So sánh hình thái dấu vết thực thi.
5. Thêm tính năng truyền phát luồng dữ liệu (streaming): mỗi nút phát ra trạng thái từng phần trong khi đang chạy. In ra các phần vi sai (deltas) khi chúng vừa đến.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Đồ thị trạng thái (State graph) | "Tác tử như một máy trạng thái" | Trạng thái định kiểu + các nút + các cạnh + các bộ giảm (reducers) |
| Bộ kiểm tra điểm lưu (Checkpointer) | "Backend lưu trữ bền vững" | Tuần tự hóa trạng thái sau mỗi nút; cho phép khôi phục |
| Bộ giảm (Reducer) | "Bộ hợp nhất trạng thái" | Hàm kết hợp trạng thái hiện tại với bản cập nhật của nút |
| Cạnh có điều kiện (Conditional edge) | "Nhánh rẽ" | Cạnh được chọn dựa trên một hàm đánh giá trạng thái |
| Đồ thị con (Subgraph) | "Đồ thị lồng nhau" | Một đồ thị được dùng như một nút bên trong đồ thị khác |
| Thực thi bền vững (Durable execution) | "Khôi phục khi gặp lỗi" | Khởi động lại tại nút thành công cuối cùng với trạng thái chuẩn xác |
| Giám sát viên (Supervisor) | "LLM định tuyến" | Bộ điều phối trung tâm cho các tác tử phụ chuyên trách |
| Bầy đàn (Swarm) | "Các tác tử P2P" | Các tác tử bàn giao qua công cụ dùng chung; không có bộ định tuyến trung tâm |

## Tài liệu đọc thêm

- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — tài liệu tham chiếu chính thức
- [Tài liệu tham khảo langgraph-supervisor](https://reference.langchain.com/python/langgraph/supervisor/) — API mẫu giám sát viên
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — giải pháp thay thế dựa trên mô hình Actor
- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — kho phiên và các tác tử phụ
