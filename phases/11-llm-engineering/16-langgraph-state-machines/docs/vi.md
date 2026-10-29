# Máy Trạng thái Tác tử — Đồ thị, Nút, Điểm kiểm tra (Agent State Machines — Graphs, Nodes, Checkpoints)

> Một vòng lặp ReAct viết thủ công thực chất là một vòng lặp `while True`. Nhưng chính vòng lặp đó khi được biểu diễn dưới dạng một đồ thị tường minh (explicit graph) sẽ trở thành thứ mà bạn có thể lưu điểm kiểm tra (checkpoint), ngắt quãng (interrupt), phân nhánh (branch) và du hành thời gian (time-travel). Bản thân tác tử (agent) không thay đổi. Khung điều phối (harness) bao bọc xung quanh nó đã thay đổi hoàn toàn.

**Type:** Xây dựng  
**Languages:** Python  
**Prerequisites:** Phase 11 · 09 (Gọi hàm - Function Calling), Phase 11 · 14 (Giao thức Bối cảnh Mô hình - Model Context Protocol)  
**Time:** ~75 phút  

## Vấn đề Thực tế

Bạn phát hành một tác tử gọi hàm (function-calling agent). Nó chạy hoàn hảo trong ba lượt đầu tiên, rồi sự cố xảy ra: mô hình cố gắng gọi một công cụ trả về mã lỗi 500, người dùng đổi ý giữa chừng khi tác vụ đang dở dang, hoặc tác tử tự ý hoàn tiền một đơn hàng mà không có sự phê duyệt của con người. Vòng lặp `while True:` viết tay không hề có điểm móc (hooks). Bạn không thể tạm dừng nó, không thể quay ngược thời gian, và không thể phân nhánh để trả lời câu hỏi: "điều gì sẽ xảy ra nếu mô hình chọn công cụ khác?". Ngay khoảnh khắc bạn đưa nó vượt ra khỏi bản demo, tác tử sẽ biến thành một chiếc hộp đen bí ẩn: hoặc là chạy thành công, hoặc là sập hoàn toàn.

Bước chuyển tiếp tiếp theo là rất hiển nhiên một khi bạn đã nhìn ra bản chất: Tác tử vốn dĩ đã là một máy trạng thái (state machine) — bao gồm lời nhắc hệ thống kết hợp lịch sử tin nhắn, các lệnh gọi công cụ đang chờ xử lý và hành động tiếp theo. Hãy biến máy trạng thái đó thành tường minh: các nút (nodes) đại diện cho "mô hình suy nghĩ", "công cụ thực thi", "con người phê duyệt", và các cạnh (edges) biểu diễn các bước chuyển tiếp có điều kiện giữa chúng. Một khi đồ thị đã được định nghĩa tường minh, khung điều phối sẽ ngay lập tức sở hữu miễn phí bốn siêu năng lực: lưu điểm kiểm tra (checkpointing - lưu trạng thái giữa các bước), ngắt quãng (interrupts - tạm dừng chờ con người can thiệp), truyền phát luồng (streaming - truyền chuỗi token và các sự kiện trung gian) và du hành thời gian (time-travel - quay lui về trạng thái cũ để thử một nhánh xử lý khác).

Hiện thực tham chiếu chuẩn mực cho sự trừu tượng hóa này là LangGraph. Nó không phải là một khung làm việc tác tử theo phong cách LangChain cũ ("đây là AgentExecutor, chúc bạn may mắn"). Nó là một môi trường thực thi đồ thị (graph runtime) hỗ trợ trạng thái hạng nhất, tính bền vững dữ liệu hạng nhất và cơ chế ngắt quãng hạng nhất. Vòng lặp tác tử giờ đây là thứ bạn có thể vẽ ra một cách trực quan thay vì phải cặm cụi viết mã thủ công.

## Khái niệm Cốt lõi

![LangGraph StateGraph: các nút, cạnh và bộ lưu điểm kiểm tra](../assets/langgraph-stategraph.svg)

Một `StateGraph` được cấu thành từ ba thành phần cơ bản:

1. **Trạng thái (State):** Một từ điển có định kiểu (TypedDict hoặc mô hình Pydantic) luân chuyển qua đồ thị. Mỗi nút nhận toàn bộ trạng thái hiện tại và trả về một bản cập nhật một phần (partial update). LangGraph sẽ hợp nhất bản cập nhật này bằng cách sử dụng một *hàm thu gọn (reducer)* cho từng trường — ví dụ `operator.add` cho các danh sách cần tích lũy dữ liệu, hoặc ghi đè (overwrite) theo mặc định.
2. **Nút (Nodes):** Các hàm Python có dạng `state -> partial_state`. Mỗi nút là một bước xử lý rời rạc: "gọi mô hình", "chạy công cụ", "tóm tắt nội dung".
3. **Cạnh (Edges):** Các bước chuyển tiếp giữa các nút. Cạnh tĩnh (static edges) luôn dẫn đến một đích cố định. Cạnh có điều kiện (conditional edges) nhận vào một hàm định tuyến `state -> next_node_name` để đồ thị có thể rẽ nhánh linh hoạt dựa trên đầu ra của mô hình.

Bạn biên dịch (compile) đồ thị. Thao tác biên dịch sẽ gắn kết cấu trúc topo, gắn thêm một bộ lưu điểm kiểm tra (checkpointer - tùy chọn nhưng bắt buộc trong sản xuất) và trả về một đối tượng có thể thực thi (runnable). Bạn kích hoạt nó với một trạng thái ban đầu và một mã luồng `thread_id`. Mỗi bước thực thi sẽ duy trì một điểm kiểm tra được lập chỉ mục theo cặp `(thread_id, checkpoint_id)`.

### Bốn Siêu Năng lực

**Lưu điểm kiểm tra (Checkpointing).** Mỗi bước chuyển tiếp giữa các nút sẽ ghi trạng thái mới vào kho lưu trữ (trong bộ nhớ cho môi trường kiểm thử, Postgres/Redis/SQLite cho môi trường sản xuất). Bạn có thể tiếp tục luồng xử lý bất cứ lúc nào bằng cách gọi lại đồ thị với cùng một `thread_id`. Đồ thị sẽ khôi phục chính xác từ điểm nó đã tạm dừng.

**Ngắt quãng (Interrupts).** Đánh dấu một nút với tham số `interrupt_before=["human_review"]` và quá trình thực thi sẽ dừng lại ngay trước khi nút đó chạy. Trạng thái được lưu lại bền vững. API của bạn phản hồi người dùng thông báo: "đang chờ phê duyệt". Một yêu cầu sau đó gửi tới cùng `thread_id` với lệnh `Command(resume=...)` sẽ tiếp tục tiến trình thực thi.

**Truyền phát luồng (Streaming).** Lệnh gọi `graph.stream(state, mode="updates")` sẽ trả về các thay đổi trạng thái (state deltas) ngay khi chúng xuất hiện. Chế độ `mode="messages"` truyền luồng từng token LLM bên trong các nút mô hình. Chế độ `mode="values"` trả về các ảnh chụp trạng thái (snapshots) hoàn chỉnh. Bạn có toàn quyền chọn thông tin nào cần hiển thị lên giao diện người dùng.

**Du hành thời gian (Time-travel).** Phương thức `graph.get_state_history(thread_id)` trả về toàn bộ nhật ký các điểm kiểm tra. Khi bạn truyền bất kỳ `checkpoint_id` nào trong quá khứ vào `graph.invoke`, hệ thống sẽ rẽ nhánh (fork) trực tiếp từ thời điểm đó. Tính năng này vô cùng đắc lực cho việc gỡ lỗi ("chuyện gì sẽ xảy ra nếu mô hình chọn công cụ B thay vì công cụ A?") và phục vụ các bài kiểm thử hồi quy (regression tests) bằng cách phát lại các dấu vết thực tế từ sản xuất.

### Bản chất Nằm ở Hàm Thu gọn (Reducers)

Mỗi trường trong trạng thái đều đi kèm một hàm thu gọn (reducer). Hầu hết các giá trị mặc định đều hoạt động tốt — giá trị mới sẽ ghi đè giá trị cũ. Tuy nhiên, danh sách tin nhắn cần sử dụng `operator.add` để các tin nhắn mới được nối tiếp vào cuối thay vì thay thế toàn bộ danh sách cũ. Các cạnh song song cũng hợp nhất các bản cập nhật của chúng thông qua hàm thu gọn này. Nếu hai nút cùng cập nhật trường `messages` mà bạn quên khai báo `Annotated[list, add_messages]`, nút thứ hai sẽ âm thầm ghi đè nút thứ nhất và bạn sẽ bị mất nửa lượt hội thoại. Hàm thu gọn là chi tiết tinh tế duy nhất trong thư viện này; chỉ cần làm đúng điều này, các phần còn lại sẽ tự động ăn khớp hoàn hảo.

### Đồ thị ReAct Thu gọn trong Bốn Nút

Một tác tử ReAct chuẩn mực trong sản xuất chỉ bao gồm bốn nút và hai cạnh:

1. `agent` — gọi LLM kèm toàn bộ lịch sử tin nhắn hiện tại. Trả về tin nhắn của trợ lý (có thể chứa các lệnh gọi công cụ `tool_calls`).
2. `tools` — thực thi bất kỳ `tool_calls` nào có trong tin nhắn trợ lý gần nhất, nối thêm kết quả công cụ vào danh sách dưới dạng các tin nhắn công cụ (`tool messages`).
3. Một cạnh có điều kiện từ `agent` để định tuyến sang `tools` nếu tin nhắn cuối cùng có chứa `tool_calls`, ngược lại định tuyến về `END`.
4. Một cạnh tĩnh từ `tools` quay ngược trở lại `agent`.

Chỉ có vậy. Bạn có được toàn bộ vòng lặp ReAct kinh điển (Suy nghĩ → Hành động → Quan sát → Suy nghĩ → …) tích hợp sẵn tính năng lưu điểm kiểm tra, ngắt quãng và truyền luồng, trong khoảng 40 dòng mã nguồn ngắn gọn.

### StateGraph so với Send (Phân nhánh Xử lý)

Lệnh `Send(node_name, state)` cho phép một nút điều phối các đồ thị con (subgraphs) chạy song song. Ví dụ: tác tử quyết định truy vấn đồng thời ba bộ truy xuất dữ liệu khác nhau. Mỗi lệnh `Send` sẽ khởi chạy một tiến trình thực thi song song của nút mục tiêu; các kết quả đầu ra sẽ được hợp nhất lại thông qua hàm thu gọn trạng thái. Đây là cách LangGraph hiện thực hóa mô hình điều phối viên - công nhân (orchestrator-workers) mà không cần sử dụng đến các nguyên mẫu đa luồng phức tạp.

### Đồ thị Con (Subgraphs)

Một đồ thị đã được biên dịch hoàn toàn có thể đóng vai trò như một nút bên trong một đồ thị khác. Đồ thị bên ngoài nhìn nhận nó như một nút đơn lẻ; trong khi đồ thị bên trong sở hữu trạng thái riêng và các điểm kiểm tra độc lập của chính nó. Đây là cách các đội ngũ kỹ thuật xây dựng các hệ thống tác tử giám sát - công nhân (supervisor-worker): đồ thị giám sát định tuyến ý định của người dùng đến các đồ thị con chuyên trách từng miền nghiệp vụ cụ thể.

````figure l5-state-graph-ledger
````

## Phần Thực hành (Build It)

### Bước 1: Trạng thái và Các Nút

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` là hàm thu gọn đảm bảo danh sách tin nhắn được tích lũy liên tục thay vì bị ghi đè. Quên khai báo hàm này là lỗi phổ biến nhất khi làm việc với LangGraph.

### Bước 2: Chạy Đồ thị Kèm Mã Luồng (Thread)

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Mỗi bản cập nhật là một từ điển có dạng `{node_name: state_delta}`. Giao diện người dùng của bạn có thể truyền luồng các sự kiện này để người dùng thấy rõ: "tác tử đang suy nghĩ… đang gọi search_web… đã có kết quả… đang trả lời."

### Bước 3: Thêm Cơ chế Ngắt quãng Chờ Con người Can thiệp (Human-in-the-Loop)

Đánh dấu một nút để quá trình thực thi tạm dừng ngay trước khi nó chạy.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # tạm dừng trước mọi lệnh gọi công cụ
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] được thiết lập. Kiểm tra các lệnh gọi công cụ được đề xuất.
# Nếu được phê duyệt:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# Nếu bị từ chối: ghi tin nhắn từ chối và tiếp tục
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

Trạng thái, điểm kiểm tra và luồng hội thoại đều được duy trì nguyên vẹn xuyên suốt khoảng thời gian ngắt quãng. Không có dữ liệu nào bị giữ lơ lửng trong RAM ngoại trừ lúc đang thực thi.

### Bước 4: Du hành Thời gian Phục vụ Gỡ lỗi

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Rẽ nhánh từ một điểm kiểm tra trong quá khứ
target = history[3].config  # quay lại 3 bước trước đó
for event in app.stream(None, target, stream_mode="values"):
    pass  # phát lại luồng thực thi tính từ thời điểm đó
```

Truyền `None` vào đầu vào sẽ phát lại từ điểm kiểm tra đã chọn; truyền một giá trị cụ thể sẽ nối thêm bản cập nhật đó vào trạng thái của điểm kiểm tra trước khi tiếp tục chạy. Đây là cách bạn tái hiện một lỗi của tác tử mà không cần phải chạy lại toàn bộ cuộc hội thoại từ đầu.

### Bước 5: Thay thế Bộ Lưu Điểm kiểm tra trong Môi trường Sản xuất

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, và Postgres đều đã được hỗ trợ sẵn. `MemorySaver` chỉ dành cho kiểm thử. Bất kỳ hệ thống nào cần duy trì dữ liệu qua các lần khởi động lại máy chủ đều phải dùng kho lưu trữ thực tế.

## Kỹ năng Thực tế (The Skill)

> Hãy xây dựng tác tử dưới dạng các đồ thị, tuyệt đối không dùng vòng lặp `while True`.

Trước khi bắt tay vào viết mã LangGraph, hãy dành 60 giây để phác thảo thiết kế:

1. **Đặt tên cho các nút.** Mỗi quyết định rời rạc hoặc hành động có tác dụng phụ (side-effect) phải là một nút riêng. "Tác tử suy nghĩ", "công cụ chạy", "người kiểm duyệt phê duyệt", "phản hồi truyền luồng". Nếu bạn không thể liệt kê rõ ràng các nút này, tác vụ đó chưa sẵn sàng để thiết kế theo mô hình tác tử.
2. **Khai báo trạng thái.** Thiết kế một TypedDict tối giản kèm hàm thu gọn (reducer) cho mọi trường dạng danh sách. Đừng nhồi nhét tất cả vào `messages`; hãy đưa các trường đặc thù của tác vụ (một bản `plan` đang thực hiện, bộ đếm `budget`, danh sách `retrieved_docs`) lên tầng cao nhất.
3. **Vẽ các cạnh.** Sử dụng cạnh tĩnh trừ khi bước tiếp theo phụ thuộc trực tiếp vào đầu ra của mô hình. Mọi cạnh có điều kiện đều cần một hàm định tuyến với các nhánh được đặt tên rõ ràng.
4. **Chọn bộ lưu điểm kiểm tra ngay từ đầu.** Dùng `MemorySaver` cho kiểm thử, Postgres/Redis/SQLite cho các môi trường khác. Tuyệt đối không phát hành mà không có checkpointer — không có checkpointer đồng nghĩa với việc không thể tiếp tục, không thể ngắt quãng và không thể du hành thời gian.
5. **Quyết định ngắt quãng TRƯỚC KHI công cụ chạy, không phải sau đó.** Việc phê duyệt phải nằm trên cạnh dẫn vào nút có tác dụng phụ để bạn có thể hủy bỏ trước khi hậu quả xảy ra; việc xác thực tính hợp lệ nằm trên cạnh dẫn ra khỏi mô hình để bạn loại bỏ các lệnh gọi sai với chi phí rẻ nhất.
6. **Mặc định sử dụng truyền phát luồng (Streaming).** Dùng `mode="updates"` cho giao diện người dùng, `mode="messages"` cho truyền luồng từng token bên trong các nút mô hình, và `mode="values"` cho ảnh chụp trạng thái đầy đủ khi đánh giá kiểm thử.

Từ chối phát hành bất kỳ tác tử LangGraph nào không có bộ lưu điểm kiểm tra. Từ chối phát hành tác tử ngắt quãng *sau khi* hành động nguy hiểm đã thực thi. Từ chối phát hành trường `messages` mà thiếu hàm thu gọn `add_messages`.

## Bài tập Thực hành

1. **Mức độ Dễ.** Triển khai đồ thị ReAct bốn nút ở trên với một công cụ máy tính (calculator) và một công cụ tìm kiếm web. Xác minh rằng `list(app.get_state_history(config))` trả về ít nhất bốn điểm kiểm tra cho một cuộc trò chuyện kéo dài hai lượt.
2. **Mức độ Trung bình.** Thêm một nút `planner` chạy trước `agent` và ghi một kế hoạch có cấu trúc `plan: list[str]` vào trạng thái. Cho phép `agent` đánh dấu các bước kế hoạch đã hoàn thành. Bài kiểm tra sẽ thất bại nếu trường `plan` bị mất sau khi tiếp tục từ một điểm kiểm tra (do sai hàm thu gọn).
3. **Mức độ Khó.** Xây dựng một đồ thị giám sát (supervisor graph) điều phối giữa ba đồ thị con (`researcher`, `writer`, `reviewer`) bằng cách sử dụng `Send`. Mỗi đồ thị con có trạng thái và bộ lưu điểm kiểm tra riêng biệt. Thêm cấu hình `interrupt_before=["writer"]` trên đồ thị bên ngoài để con người có thể phê duyệt bản tóm tắt nghiên cứu trước khi viết bài. Xác nhận rằng việc du hành thời gian từ một điểm kiểm tra cũ chỉ chạy lại nhánh được rẽ nhánh.

## Thuật ngữ Then chốt

| Thuật ngữ | Cách mọi người thường nói | Bản chất kỹ thuật thực tế |
|------|-----------------|-----------------------|
| StateGraph | "Đồ thị LangGraph" | Đối tượng xây dựng đồ thị nơi bạn bổ sung các nút và cạnh trước khi thực hiện biên dịch. |
| Reducer | "Cách trường dữ liệu hợp nhất" | Một hàm `(old, new) -> merged` được áp dụng khi một nút trả về bản cập nhật cho trường đó; mặc định là ghi đè, `add_messages` dùng để nối thêm. |
| Thread | "Mã phiên trò chuyện" | Một chuỗi định danh `thread_id` xác định phạm vi toàn bộ các điểm kiểm tra cho một phiên làm việc duy nhất. |
| Checkpoint | "Trạng thái tạm dừng" | Ảnh chụp trạng thái đầy đủ được lưu trữ bền vững sau mỗi bước chuyển tiếp nút, được lập chỉ mục theo cặp `(thread_id, checkpoint_id)`. |
| Interrupt | "Tạm dừng chờ con người" | Cơ chế `interrupt_before` / `interrupt_after` dừng thực thi tại ranh giới một nút; tiếp tục bằng `Command(resume=...)`. |
| Time-travel | "Rẽ nhánh từ bước cũ" | Lệnh `graph.invoke(None, config_with_old_checkpoint_id)` phát lại luồng thực thi tính từ điểm kiểm tra đó trở đi. |
| Send | "Điều phối đồ thị con song song" | Một hàm khởi tạo mà nút có thể trả về để kích hoạt N tiến trình thực thi song song của một nút đích. |
| Subgraph | "Đồ thị biên dịch đóng vai nút" | Một StateGraph đã biên dịch được dùng như một nút thông thường trong đồ thị khác; duy trì phạm vi trạng thái độc lập của chính nó. |

## Đọc thêm

- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — tài liệu tham chiếu chuẩn mực cho StateGraph, hàm thu gọn (reducers), bộ lưu điểm kiểm tra (checkpointers) và ngắt quãng (interrupts).
- [Các khái niệm cốt lõi của LangGraph: trạng thái, hàm thu gọn, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — mô hình tư duy được sử dụng trong bài học này, từ nguồn tài liệu gốc.
- [Tính bền vững và Điểm kiểm tra trong LangGraph](https://langchain-ai.github.io/langgraph/concepts/persistence/) — chi tiết về các kho lưu trữ Postgres/SQLite/Redis, không gian tên điểm kiểm tra và thread ID.
- [Mô hình Con người Tham gia vào Luồng (Human-in-the-loop) của LangGraph](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — phân tích `interrupt_before`, `interrupt_after`, `Command(resume=...)`, và mẫu chỉnh sửa trạng thái trực tiếp.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — mẫu kiến trúc nền tảng mà mọi tác tử LangGraph triển khai; phân tích lý do khoa học của việc duy trì chuỗi dấu vết suy luận.
- [Anthropic — Xây dựng tác tử hiệu quả (Tháng 12/2024)](https://www.anthropic.com/research/building-effective-agents) — hướng dẫn lựa chọn hình thái đồ thị (chuỗi, định tuyến, điều phối-công nhân, đánh giá-tối ưu) phù hợp với từng bài toán.
- Phase 11 · 09 (Gọi hàm - Function Calling) — nguyên mẫu gọi công cụ mà mọi nút tác tử LangGraph tái sử dụng.
- Phase 11 · 14 (Giao thức Bối cảnh Mô hình - Model Context Protocol) — cơ chế khám phá công cụ bên ngoài được kết nối vào `ToolNode` của LangGraph thông qua bộ điều hợp MCP.
- Phase 11 · 17 (Đánh đổi giữa các Khung làm việc Tác tử) — tiêu chí lựa chọn LangGraph thay vì CrewAI, AutoGen hoặc Agno.
