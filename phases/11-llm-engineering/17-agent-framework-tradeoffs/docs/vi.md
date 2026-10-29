# Đánh đổi giữa các Khung làm việc Tác tử — Điều phối Đồ thị, Vai trò và Tác tử (Agent Framework Tradeoffs — Graph, Role, and Actor Orchestration)

> Mọi khung làm việc (framework) đều quảng bá cùng một kịch bản demo (tác tử nghiên cứu tổng hợp báo cáo) và che giấu cùng một lỗ hổng thiết kế (lược đồ trạng thái xung đột với tầng điều phối). Hãy chọn khung làm việc có các lớp trừu tượng khớp với hình thái bài toán của bạn; mọi thứ khác chỉ là phần mã nối (glue code) mà bạn sẽ phải viết lại hai lần.

**Type:** Học tập  
**Languages:** Python  
**Prerequisites:** Phase 11 · 09 (Gọi hàm - Function Calling), Phase 11 · 16 (LangGraph)  
**Time:** ~45 phút  

## Vấn đề Thực tế

Bạn có một tác vụ đòi hỏi nhiều hơn một lệnh gọi LLM đơn lẻ. Đó có thể là một quy trình nghiên cứu (lập kế hoạch, tìm kiếm, tóm tắt, trích dẫn nguồn). Đó có thể là một đường ống đánh giá mã nguồn (phân tích sự khác biệt mã - diff, phê bình, tạo bản vá, kiểm định). Hoặc đó có thể là một trợ lý đa lượt tương tác giúp đặt vé máy bay, soạn email và lập báo cáo chi phí. Bạn quyết định chọn một khung làm việc.

Ba ngày sau, bạn nhận ra các lớp trừu tượng của khung làm việc bắt đầu rò rỉ (abstraction leak). CrewAI cung cấp cho bạn các vai trò nhưng lại gây khó khăn khi "nhà nghiên cứu" cần chuyển giao một kế hoạch có cấu trúc chặt chẽ cho "người viết". AutoGen hỗ trợ trò chuyện giữa các tác tử nhưng thiếu khái niệm trạng thái hạng nhất, khiến điểm kiểm tra của bạn biến thành một tệp pickle lưu nhật ký hội thoại hỗn độn. LangGraph cung cấp một đồ thị trạng thái nhưng bắt buộc bạn phải đặt tên cho từng bước chuyển tiếp trước cả khi bạn biết tác tử sẽ hành động ra sao. Agno cung cấp một sự trừu tượng hóa tác tử đơn lẻ tuyệt vời nhưng lại gặp bế tắc khi bạn muốn phân nhánh xử lý song song cho ba công nhân cùng lúc.

Giải pháp không phải là "chọn khung làm việc tốt nhất". Giải pháp là phải khớp trừu tượng cốt lõi của khung làm việc với hình thái thực tế của bài toán. Bài học này sẽ vẽ nên bản đồ định hướng đó.

## Khái niệm Cốt lõi

![Ma trận khung làm việc tác tử: trừu tượng cốt lõi so với hình thái bài toán](../assets/framework-matrix.svg)

Bốn khung làm việc thống trị bối cảnh công nghệ năm 2026. Các trừu tượng cốt lõi của chúng hoàn toàn khác nhau.

| Khung làm việc | Trừu tượng cốt lõi | Phù hợp nhất | Kém phù hợp nhất |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — trạng thái có định kiểu, các nút, cạnh có điều kiện, bộ lưu điểm kiểm tra. | Các quy trình làm việc với trạng thái tường minh và cơ chế ngắt quãng chờ con người; tác tử sản xuất cần gỡ lỗi du hành thời gian. | Các phiên thảo luận ý tưởng tự do dựa trên vai trò khi chưa rõ cấu trúc topo luồng xử lý. |
| **CrewAI** | `Crew` — vai trò (mục tiêu, câu chuyện nền), nhiệm vụ, quy trình (tuần tự hoặc phân cấp). | Quy trình đóng vai hoặc định hướng theo nhân vật với kế hoạch tuyến tính/phân cấp ngắn gọn. | Bất kỳ tác vụ nào đòi hỏi lưu trữ trạng thái phức tạp vượt ngoài lịch sử lượt trao đổi; các luồng rẽ nhánh phức tạp. |
| **AutoGen** | Cặp `ConversableAgent` — hai hoặc nhiều tác tử luân phiên trao đổi cho đến khi thỏa mãn điều kiện dừng. | Các cuộc *đối thoại* đa tác tử (thầy - trò, người đề xuất - người phản biện, người thực hiện - người đánh giá) nơi giải pháp nảy sinh từ cuộc trò chuyện. | Quy trình tất định với đồ thị có hướng không chu trình (DAG) đã biết; các tác vụ cần trạng thái bền vững qua các lần khởi động lại. |
| **Agno** | `Agent` — một LLM đơn lẻ tích hợp sẵn công cụ và bộ nhớ, có thể ghép nối thành các nhóm. | Các tác tử đơn lẻ hoặc nhóm tác tử gọn nhẹ cần xây dựng nhanh; hỗ trợ đa phương thức mạnh mẽ và có sẵn trình điều khiển lưu trữ. | Các đồ thị rẽ nhánh sâu, tường minh với các hàm thu gọn (reducers) tùy chỉnh phức tạp. |

### "Trừu tượng" Thực sự có Nghĩa là gì?

Trừu tượng cốt lõi của một khung làm việc chính là thứ bạn vẽ lên bảng trắng khi thuyết trình về kiến trúc hệ thống của mình.

- **LangGraph** → Bạn vẽ một đồ thị. Các nút là các bước xử lý, các cạnh là bước chuyển tiếp, và đối tượng trạng thái tại mọi thời điểm đều có định kiểu rõ ràng. Mô hình tư duy ở đây là một máy trạng thái (state machine).
- **CrewAI** → Bạn vẽ một sơ đồ tổ chức doanh nghiệp. Mỗi vai trò có một bản mô tả công việc cụ thể và một người quản lý điều phối các nhiệm vụ. Mô hình tư duy ở đây là một nhóm nhỏ các chuyên gia.
- **AutoGen** → Bạn vẽ một kênh tin nhắn trực tiếp trên Slack. Hai tác tử nhắn tin qua lại với nhau; một tác tử thứ ba tham gia nếu cần người điều phối. Mô hình tư duy ở đây là một cuộc trò chuyện (chat).
- **Agno** → Bạn vẽ một chiếc hộp duy nhất với các công cụ được gắn xung quanh. Đặt các hộp cạnh nhau để tạo thành một nhóm làm việc. Mô hình tư duy ở đây là "tác tử tích hợp sẵn mọi tiện ích (batteries included)".

### Bài toán Quản lý Trạng thái (State)

Trạng thái chính là điểm khiến hầu hết các quyết định chọn khung làm việc bị đổ vỡ khi đưa vào môi trường sản xuất.

- **LangGraph:** Trạng thái có định kiểu (`TypedDict` hoặc mô hình Pydantic), các hàm thu gọn (reducers) trên từng trường dữ liệu, bộ lưu điểm kiểm tra hạng nhất (SQLite/Postgres/Redis). Các tính năng tiếp tục thực thi, ngắt quãng và du hành thời gian đều được hỗ trợ sẵn. *(Xem Phase 11 · 16).*
- **CrewAI:** Trạng thái luân chuyển dưới dạng chuỗi văn bản giữa các nhiệm vụ thông qua trường `context`, hoặc có cấu trúc thông qua `output_pydantic`. Mặc định không có kho lưu trữ bền vững cho toàn bộ nhóm; bạn phải tự xây dựng cơ chế này nếu muốn nhóm tác tử sống sót qua sự cố khởi động lại máy chủ.
- **AutoGen:** Trạng thái chính là lịch sử trò chuyện và bất kỳ `context` nào do người dùng định nghĩa. Bản ghi nhật ký hội thoại được lưu lại; nhưng trạng thái quy trình làm việc tùy biến thì không, trừ khi bạn tự viết các bộ điều hợp (adapters).
- **Agno:** Tích hợp sẵn các trình điều khiển lưu trữ (SQLite, Postgres, Mongo, Redis, DynamoDB) gắn trực tiếp vào `Agent` thông qua tham số `storage=` — các phiên hội thoại và ký ức người dùng được lưu trữ tự động. Đây là kho lưu trữ phiên làm việc (session store) chứ không phải bộ lưu điểm kiểm tra đồ thị hoàn chỉnh.

### Bài toán Phân nhánh Luồng Xử lý (Branching)

Mọi tác tử phức tạp đều phải rẽ nhánh. Yếu tố cốt lõi là ai sẽ quyết định việc rẽ nhánh đó.

- **LangGraph:** Do lập trình viên quyết định, thông qua các cạnh có điều kiện. Việc định tuyến là một hàm Python với các nhánh được đặt tên rõ ràng. Các nhánh là thành phần hạng nhất trong đồ thị đã biên dịch; bộ lưu điểm kiểm tra sẽ ghi lại chính xác nhánh nào đã được thực thi.
- **CrewAI:** Do tác tử quản lý quyết định trong chế độ phân cấp; trong chế độ tuần tự, bạn quyết định tại thời điểm lập trình. Việc định tuyến nằm ẩn bên trong danh sách nhiệm vụ; không có cấu trúc "if" hạng nhất nào nằm ngoài lời nhắc của người quản lý.
- **AutoGen:** Do các tác tử tự quyết định thông qua hội thoại. Việc rẽ nhánh mang tính tự phát dựa trên việc ai sẽ là người phát biểu tiếp theo. `GroupChatManager` lựa chọn người nói tiếp; bạn có thể tự viết hàm `speaker_selection_method`, nhưng mặc định việc này do LLM dẫn dắt.
- **Agno:** Do tác tử quyết định dựa trên công cụ nào nó sẽ gọi tiếp theo. Các nhóm tác tử có chế độ điều phối/định tuyến/cộng tác; việc rẽ nhánh phức tạp hơn thuộc trách nhiệm xử lý của lập trình viên.

### Khả năng Quan sát (Observability)

- **LangGraph:** Hỗ trợ OpenTelemetry thông qua LangSmith hoặc bất kỳ bộ xuất OTel tiêu chuẩn nào. Mỗi bước chuyển tiếp nút là một span dấu vết; các điểm kiểm tra đồng thời đóng vai trò là các dấu vết có thể phát lại được. LangSmith là lựa chọn chính hãng; Langfuse và Phoenix cũng có các bộ điều hợp sẵn sàng.
- **CrewAI:** Tích hợp OpenTelemetry hạng nhất từ cuối năm 2025; hỗ trợ kết nối trực tiếp với Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen:** Hỗ trợ OpenTelemetry thông qua `autogen-core`; có sẵn đầu nối cho AgentOps và Opik. Độ chi tiết của dấu vết được ghi lại theo từng tin nhắn của tác tử chứ không phải theo từng nút xử lý.
- **Agno:** Tích hợp sẵn cờ `monitoring=True` cùng các bộ xuất OpenTelemetry; liên kết chặt chẽ với Langfuse để theo dõi dấu vết phiên làm việc.

### Chi phí và Độ trễ

Cả bốn khung làm việc đều tạo ra chi phí phụ trội (overhead) trên mỗi lệnh gọi (logic xử lý nội bộ, kiểm định dữ liệu, tuần tự hóa). Thứ tự gần đúng của mức chi phí phụ trội tăng dần: Agno ≈ LangGraph < CrewAI ≈ AutoGen. Sự chênh lệch này chủ yếu do khối lượng định tuyến phụ mà LLM phải gánh vác trong khung làm việc. Tác tử quản lý phân cấp của CrewAI tiêu tốn token chỉ để quyết định ai sẽ làm tiếp theo; `GroupChatManager` của AutoGen cũng tương tự. LangGraph chỉ tiêu tốn token tại đúng những nơi bạn viết lệnh `llm.invoke`. Đường dẫn tác tử đơn của Agno rất mỏng và nhẹ.

Khi chi phí trên mỗi lần chạy là yếu tố sống còn, hãy ưu tiên định tuyến tường minh (các cạnh của LangGraph, hàm `speaker_selection_method` của AutoGen) thay vì định tuyến do LLM tự quyết định.

### Khả năng Tương tác Liên thông (Interoperability)

- **LangGraph** ↔ Tương thích hoàn toàn với các công cụ, bộ truy xuất và mô hình của **LangChain**. Tích hợp bộ điều hợp MCP hạng nhất (nhập công cụ từ các máy chủ MCP bên ngoài).
- **CrewAI** ↔ Các công cụ kế thừa từ `BaseTool`; tương thích với công cụ LangChain, LlamaIndex và công cụ MCP. Cho phép ủy quyền giữa các nhóm thông qua `allow_delegation=True`.
- **AutoGen** → `FunctionTool` bọc bất kỳ hàm Python callable nào; có sẵn bộ điều hợp MCP. Gắn kết chặt chẽ với hệ sinh thái AG2 cho các mẫu hình tương tác đa tác tử.
- **Agno** → Sử dụng bộ trang trí `@tool` hoặc lớp con BaseTool; có bộ điều hợp MCP; các công cụ có thể chia sẻ linh hoạt giữa các tác tử và nhóm tác tử.

## Kỹ năng Thực tế (The Skill)

> Bạn có thể giải thích, chỉ trong một câu duy nhất, tại sao một khung làm việc cụ thể lại là lựa chọn đúng đắn cho một bài toán tác tử nhất định.

Danh mục kiểm tra trước khi lập trình:

1. **Vẽ hình thái bài toán.** Đây là một đồ thị (trạng thái có định kiểu, các bước chuyển tiếp có tên)? Một màn đóng vai (các chuyên gia bàn giao công việc cho nhau)? Một cuộc trò chuyện (các tác tử thảo luận cho đến khi xong việc)? Hay một tác tử đơn lẻ kèm công cụ?
2. **Xác định ai là người quyết định rẽ nhánh.** Phân nhánh do lập trình viên định nghĩa → LangGraph. Phân nhánh do tác tử quản lý quyết định → CrewAI phân cấp. Phân nhánh tự phát sinh từ trò chuyện → AutoGen. Phân nhánh do lệnh gọi công cụ quyết định → Agno.
3. **Kiểm tra ngân sách trạng thái.** Bạn có cần khôi phục từ điểm kiểm tra sau sự cố không? Có cần du hành thời gian không? Có cần con người ngắt quãng phê duyệt giữa chừng không? Nếu có, LangGraph là lựa chọn mặc định; các phiên làm việc của Agno bao quát tốt trạng thái ở phạm vi cuộc hội thoại.
4. **Kiểm tra ngân sách chi phí.** Việc định tuyến bằng LLM tiêu tốn thêm token trong mỗi lượt. Nếu tác tử chạy hàng nghìn lần mỗi ngày, hãy ưu tiên định tuyến tường minh bằng mã lệnh.
5. **Đánh giá chi phí phụ trội của khung làm việc.** Mỗi khung làm việc là một gói phụ thuộc bổ sung. Nếu tác vụ chỉ gồm hai lệnh gọi LLM và một công cụ, hãy viết 30 dòng mã Python thuần túy; không có khung làm việc nào rẻ và nhanh hơn việc không dùng khung làm việc.

Từ chối dùng khung làm việc trước khi bạn có thể vẽ được đồ thị, sơ đồ tổ chức, luồng trò chuyện hoặc khối chức năng của tác tử. Từ chối chọn một khung làm việc bắt bạn phải chống lại mô hình trạng thái của nó để đạt được thứ bạn thực sự cần.

## Ma trận Quyết định (Decision Matrix)

| Hình thái bài toán | Khung làm việc ưu tiên | Lý do |
|-------------------|----------------------|-----|
| Đồ thị DAG quy trình với trạng thái có định kiểu, con người phê duyệt, chạy dài hạn | LangGraph | Trạng thái hạng nhất, bộ lưu điểm kiểm tra, ngắt quãng, du hành thời gian. |
| Đường ống nghiên cứu / viết bài với các vai trò tách biệt rõ ràng | CrewAI (tuần tự) hoặc đồ thị con LangGraph | Mô hình vai trò theo nhiệm vụ rất dễ biểu diễn trong CrewAI; mở rộng lên LangGraph khi phân nhánh phức tạp. |
| Đối thoại người đề xuất - người phản biện hoặc thầy - trò | AutoGen | Trò chuyện hai tác tử là hình thái tự nhiên của khung làm việc này. |
| Tác tử đơn lẻ với công cụ, phiên làm việc, bộ nhớ ngữ cảnh | Agno | Cài đặt tinh gọn nhất, tích hợp sẵn kho lưu trữ và bộ nhớ. |
| Hàng nghìn luồng phân nhánh song song với hàm thu gọn | LangGraph + `Send` | Khung làm việc duy nhất có API điều phối song song hạng nhất. |
| Bản mẫu thử nghiệm nhanh, không muốn phụ thuộc khung làm việc | Python thuần + SDK nhà cung cấp | Không có khung làm việc nào nhanh bằng việc không dùng khung làm việc. |

````figure l5-framework-fit
````

## Bài tập Thực hành

1. **Mức độ Dễ.** Thực hiện cùng một tác vụ — "nghiên cứu địa chỉ trụ sở của Anthropic, viết bản tóm tắt 200 từ, trích dẫn nguồn" — và triển khai nó trên LangGraph (bốn nút: lập kế hoạch, tìm kiếm, viết, trích dẫn) và trên CrewAI (ba vai trò: nhà nghiên cứu, người viết, biên tập viên). Báo cáo chi phí token trên mỗi lần chạy và số dòng mã nguồn thực tế.
2. **Mức độ Trung bình.** Xây dựng cùng tác vụ đó trên AutoGen (cuộc trò chuyện giữa nhà nghiên cứu ↔ người viết, biên tập viên tham gia qua `GroupChat`) và Agno (một tác tử duy nhất tích hợp `search_tools` và `write_tools`, kèm kho lưu trữ phiên làm việc). Xếp hạng bốn cách triển khai theo: (a) chi phí trên mỗi lần chạy, (b) khả năng tiếp tục sau khi tiến trình bị sập, (c) khả năng chèn một bước phê duyệt của con người trước khi thực hiện bước viết.
3. **Mức độ Khó.** Xây dựng một kịch bản cây quyết định `pick_framework.py` nhận đầu vào là mô tả ngắn gọn về bài toán (định dạng JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) và trả về gợi ý lựa chọn kèm giải thích lý do trong một câu duy nhất. Kiểm chứng tính đúng đắn trên sáu trường hợp thử nghiệm do chính bạn tự thiết kế.

## Thuật ngữ Then chốt

| Thuật ngữ | Cách mọi người thường nói | Bản chất kỹ thuật thực tế |
|------|-----------------|-----------------------|
| Orchestration | "Cách các tác tử phối hợp" | Tầng điều phối quyết định nút/vai trò/tác tử nào sẽ được thực thi tiếp theo. |
| Durable state | "Tiếp tục sau khi khởi động lại" | Trạng thái sống sót qua sự cố dừng tiến trình, gắn liền với điểm kiểm tra hoặc kho lưu trữ phiên. |
| LLM-selected routing | "Để mô hình tự quyết định" | Một LLM lập kế hoạch sẽ chọn bước tiếp theo trong mỗi lượt; linh hoạt nhưng tốn token ở mọi quyết định. |
| Explicit routing | "Lập trình viên quyết định" | Một hàm Python hoặc cạnh tĩnh lựa chọn bước tiếp theo; chi phí rẻ và dễ dàng kiểm toán. |
| Crew | "Nhóm làm việc của CrewAI" | Tập hợp gồm các vai trò + nhiệm vụ + quy trình (tuần tự hoặc phân cấp) được đóng gói thành một đối tượng thực thi duy nhất. |
| GroupChat | "Trò chuyện nhóm của AutoGen" | Cuộc hội thoại có quản lý giữa N tác tử với một bộ chọn người phát biểu tiếp theo. |
| Team (Agno) | "Nhóm đa tác tử trong Agno" | Chế độ định tuyến / điều phối / cộng tác trên một tập hợp các tác tử. |
| StateGraph | "Đồ thị của LangGraph" | Trừu tượng hóa bao gồm trạng thái có định kiểu, nút, cạnh có điều kiện và bộ lưu điểm kiểm tra. |

## Đọc thêm

- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — StateGraph, bộ lưu điểm kiểm tra, ngắt quãng, du hành thời gian.
- [Tài liệu CrewAI](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes.
- [Tài liệu AutoGen](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, teams, tools.
- [Tài liệu Agno](https://docs.agno.com/) — Agent, Team, Workflow, lưu trữ, bộ nhớ.
- [Anthropic — Xây dựng tác tử hiệu quả (Tháng 12/2024)](https://www.anthropic.com/research/building-effective-agents) — thư viện mẫu hình (chuỗi lời nhắc, định tuyến, song song hóa, điều phối-công nhân, đánh giá-tối ưu) độc lập với khung làm việc.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — vòng lặp nền tảng mà mọi khung làm việc tác tử đều triển khai.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — bài báo thiết kế nền tảng của AutoGen.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — nền tảng nhập vai mà các ngăn xếp nhân vật theo phong cách CrewAI xây dựng dựa trên.
- Phase 11 · 16 (LangGraph) — khung làm việc được dùng làm chuẩn đối chuẩn trong bài học này.
- Phase 11 · 19 (Reflexion) — một mẫu kiến trúc ánh xạ mượt mà vào LangGraph nhưng lại khá gượng gạo khi đưa vào CrewAI.
- Phase 11 · 22 (Khả năng quan sát trong sản xuất) — cách gắn công cụ đo lường quan sát cho bất kỳ khung làm việc nào bạn lựa chọn.
