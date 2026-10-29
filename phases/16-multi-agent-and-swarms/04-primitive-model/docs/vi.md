# Mô hình Nguyên hàm Đa tác tử (The Multi-Agent Primitive Model)

> Bốn nguyên hàm (primitives), không hơn không kém — tác tử (agent), bàn giao quyền điều khiển (handoff), trạng thái dùng chung (shared state), bộ điều phối (orchestrator) — cùng định hình một không gian thiết kế bốn chiều, và các framework đa tác tử lớn ra mắt trong năm 2026 (AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework) thực chất chỉ là những điểm tọa độ trong không gian đó. Bài học này xây dựng các nguyên hàm từ con số không, vận hành một hệ thống giả lập trên cả bốn nguyên hàm, sau đó lập bản đồ quy chiếu mọi framework lớn lên cùng một hệ trục để bạn có thể nắm bắt bất kỳ bản phát hành mới nào chỉ trong một đoạn văn ngắn.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 phút

## Vấn đề

Cứ mỗi sáu tháng, một framework đa tác tử mới lại ra đời. AutoGen vào năm 2023. CrewAI vào năm 2024. LangGraph và OpenAI Swarm vào năm 2024. Google ADK vào tháng 4 năm 2025. Microsoft Agent Framework bản RC vào tháng 2 năm 2026. Mỗi thông cáo báo chí đều tự nhận mình đã tìm ra "lớp trừu tượng chuẩn mực nhất".

Nếu bạn cố gắng học từng framework một cách riêng rẽ, bạn sẽ nhanh chóng kiệt sức. Giao diện lập trình (API) trông hoàn toàn khác nhau. Tài liệu hướng dẫn bất đồng quan điểm về việc định nghĩa thế nào là một "tác tử". Một framework gọi bộ nhớ dùng chung là "bảng đen" (blackboard), framework khác lại gọi là "nhóm tin nhắn" (message pool), và một bên thứ ba lại gọi là "StateGraph". Bạn bắt đầu nghi ngờ rằng toàn bộ lĩnh vực này thực chất chỉ đang xáo xào thuật ngữ để tiếp thị.

Thực tế không phải vậy. Đằng sau những chiêu trò truyền thông, bốn nguyên hàm nền tảng vẫn luôn ổn định và bất biến. Hãy nắm vững chúng một lần duy nhất, bạn sẽ có khả năng giải mã mọi framework mới chỉ trong một đoạn văn ngắn.

## Khái niệm

### Bốn nguyên hàm cốt lõi

1. **Tác tử (Agent)** — một prompt hệ thống (system prompt) cộng với một danh sách công cụ (tools). Hoàn toàn phi trạng thái (stateless); mỗi phiên chạy đều bắt đầu lại từ prompt hệ thống và lịch sử tin nhắn hiện tại.
2. **Bàn giao quyền điều khiển (Handoff)** — một bước chuyển giao quyền kiểm soát có cấu trúc từ tác tử này sang tác tử khác. Về mặt cơ chế, đây là một lượt gọi công cụ trả về một tác tử mới, hoặc một cạnh trong đồ thị chuyển trạng thái dựa trên một điều kiện định sẵn.
3. **Trạng thái dùng chung (Shared state)** — bất kỳ cấu trúc dữ liệu nào mà nhiều hơn một tác tử có thể đọc (và đôi khi ghi vào). Đó có thể là nhóm tin nhắn (message pool), bảng đen (blackboard), kho khóa - giá trị (key-value store), hoặc bộ nhớ vector (vector memory).
4. **Bộ điều phối (Orchestrator)** — thực thể quyết định xem ai sẽ là người phát ngôn tiếp theo. Các lựa chọn gồm có: một đồ thị tường minh (tính tất định), một bộ chọn người phát ngôn bằng LLM (mềm dẻo), chính lệnh gọi handoff của tác tử phát ngôn trước đó (mô hình OpenAI Swarm), hoặc một bộ lập lịch phân phối từ hàng đợi (kiến trúc bầy đàn - swarm architecture).

Đó là toàn bộ không gian thiết kế. Mọi framework chỉ đơn giản là chọn các giá trị mặc định cho từng trục tọa độ; phần còn lại chỉ là cú pháp bề mặt.

### Cách các framework năm 2026 ánh xạ vào mô hình

| Framework | Tác tử (Agent) | Bàn giao (Handoff) | Trạng thái dùng chung (Shared state) | Bộ điều phối (Orchestrator) |
|---|---|---|---|---|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | Công cụ trả về đối tượng Agent | Bài toán của bên gọi | Lệnh gọi handoff tiếp theo do LLM quyết định |
| AutoGen v0.4 / AG2 | `ConversableAgent` | Bộ chọn người phát ngôn trên GroupChat | Nhóm tin nhắn (Message pool) | Hàm lựa chọn (dùng LLM hoặc luân phiên vòng tròn) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Chuỗi đầu ra của Task nối tiếp nhau | LLM quản lý hoặc thứ tự tĩnh |
| LangGraph | Hàm đại diện cho Node | Cạnh đồ thị + điều kiện | Bộ thu gọn `StateGraph` reducer | Bản thân đồ thị (mang tính tất định) |
| Microsoft Agent Framework | Tác tử + các mẫu hình điều phối | Phụ thuộc vào mẫu hình cụ thể | Luồng hội thoại / ngữ cảnh | Phụ thuộc vào mẫu hình cụ thể |
| Google ADK | Tác tử + Thẻ A2A (Agent Card) | Tác vụ A2A (A2A task) | Các tạo tác A2A (Artifacts) | Máy chủ chủ quản quyết định |

Sự khác biệt bề ngoài trông có vẻ đồ sộ. Nhưng ẩn sâu bên dưới: cùng bốn núm vặn nguyên hàm như nhau.

### Vì sao điều này lại quan trọng

Khi bạn đã nhìn thấu các nguyên hàm, việc so sánh các framework sẽ trở thành một danh sách kiểm tra rất ngắn gọn:

- Bộ điều phối có tin tưởng giao phó cho LLM tự định tuyến luồng chạy (Swarm) hay nó cố định lộ trình định tuyến bằng mã nguồn (LangGraph)?
- Trạng thái chia sẻ chứa toàn bộ lịch sử (GroupChat) hay được chiếu lát cắt thu gọn theo vai trò (StateGraph reducer)?
- Các tác tử có thể tự sửa đổi prompt của nhau (vai trò quản lý trong CrewAI) hay chỉ đơn thuần bàn giao lượt xử lý (Swarm)?

Ba câu hỏi này giải quyết tới 80% bài toán lựa chọn framework phù hợp cho một vấn đề thực tế. Bạn sẽ dừng việc tìm kiếm mơ hồ xem đâu là "framework đa tác tử tốt nhất" và bắt đầu thiết kế dựa trên trục kiến trúc mà bạn thực sự quan tâm.

### Nhận thức sâu sắc về tính phi trạng thái (The Stateless Insight)

Mọi nguyên hàm ngoại trừ trạng thái dùng chung đều là **phi trạng thái (stateless)**. Tác tử là một hàm nhận (prompt, tools). Bàn giao quyền điều khiển là một lời gọi hàm. Bộ điều phối là một thuật toán lập lịch. **Thành phần có trạng thái duy nhất trong toàn bộ hệ thống chính là trạng thái dùng chung.** Đó chính là nơi sinh ra mọi lỗi hệ thống hóc búa nhất: đầu độc bộ nhớ (memory poisoning - Bài 15), sai lệch thứ tự tin nhắn, xung đột phiên bản và tranh chấp quyền ghi.

Các framework che giấu trạng thái dùng chung (như Swarm) sẽ đẩy sự phức tạp đó về phía người gọi. Các framework tập trung hóa trạng thái dùng chung (như điểm kiểm tra checkpoint của LangGraph, nhóm tin nhắn của AutoGen) giúp hệ thống dễ quan sát kiểm tra hơn nhưng lại chuyển chi phí điều phối sang khâu hiện thực hóa trạng thái chia sẻ đó.

### Giải phẫu từng nguyên hàm đơn lẻ

#### Tác tử (Agent)

```text
Agent = (system_prompt, tools, model, optional_name)
```

Không có bộ nhớ riêng. Không có trạng thái nội tại. Hai tác tử có cùng prompt hệ thống và danh sách công cụ hoàn toàn có thể hoán đổi cho nhau. Mọi thứ trông giống như trạng thái riêng của từng tác tử thực chất đều nằm trong trạng thái dùng chung hoặc trong giao thức bàn giao.

#### Bàn giao quyền điều khiển (Handoff)

```text
Handoff = (from_agent, to_agent, reason, payload)
```

Có ba cách hiện thực hóa chiếm ưu thế:

- **Giá trị trả về của hàm (Function return)** — công cụ trả về đối tượng tác tử tiếp theo. Đây là mẫu hình kinh điển của OpenAI Swarm. Các tác tử mang logic định tuyến ngay trong lược đồ công cụ của mình.
- **Cạnh đồ thị (Graph edge)** — LangGraph. Các cạnh được khai báo một cách tường minh. LLM sinh ra một giá trị; một hàm điều kiện sẽ chọn node tiếp theo trên đồ thị.
- **Lựa chọn người phát ngôn (Speaker selection)** — AutoGen GroupChat. Một hàm chọn lọc (đôi khi chính là một lượt gọi LLM) đọc dữ liệu từ nhóm tin nhắn và chỉ định ai sẽ nói tiếp theo.

#### Trạng thái dùng chung (Shared state)

```text
SharedState = { messages: [], artifacts: {}, context: {} }
```

Ở mức tối thiểu, nó là một danh sách tin nhắn tuần tự. Thường nó sẽ mở rộng hơn: các tạo tác có cấu trúc (đầu ra của Task trong CrewAI), ngữ cảnh có kiểu định sẵn (bộ thu gọn reducer của LangGraph), bộ nhớ ngoài (máy chủ MCP, cơ sở dữ liệu vector).

Hai cấu trúc liên kết chính: **nhóm đầy đủ (full pool)** (mọi tác tử đều nhìn thấy mọi tin nhắn) và **chiếu theo vai trò (projected)** (tác tử chỉ thấy góc nhìn bị giới hạn theo phạm vi vai trò của nó). Cấu trúc đầy đủ rất đơn giản nhưng mở rộng quy mô rất kém. Cấu trúc chiếu lát cắt mở rộng tốt nhưng đòi hỏi phải thiết kế lược đồ nghiêm ngặt ngay từ đầu.

#### Bộ điều phối (Orchestrator)

```text
Orchestrator = ({state, last_speaker}) -> next_agent
```

Bao gồm bốn biến thể:

- **Tĩnh (Static)** — đồ thị được cố định tại thời điểm xây dựng hệ thống (LangGraph chế độ tất định, CrewAI chế độ Sequential).
- **LLM lựa chọn (LLM-selected)** — một LLM đọc dữ liệu từ nhóm tin nhắn và chủ động chọn người phát ngôn tiếp theo (AutoGen, CrewAI chế độ Hierarchical).
- **Điều khiển bằng bàn giao (Handoff-driven)** — tác tử hiện tại tự quyết định bằng cách gọi công cụ bàn giao handoff (Swarm).
- **Điều khiển bằng hàng đợi (Queue-driven)** — các tác tử công nhân tự kéo việc từ hàng đợi chung; không có chỉ định tường minh ai là người tiếp theo (kiến trúc bầy đàn swarm, Matrix).

### Điều gì thực sự thay đổi giữa các framework

Khi các nguyên hàm đã được cố định, những quyết định thiết kế còn lại bao gồm:

- **Chiến lược bộ nhớ (Memory strategy)** — bộ nhớ tạm thời (ephemeral) so với lưu điểm kiểm tra bền vững (durable checkpointing như trong LangGraph).
- **Ranh giới an toàn (Safety boundary)** — ai có quyền phê duyệt một hành động bàn giao (cơ chế con người tham gia vòng lặp - human-in-the-loop).
- **Hạch toán chi phí (Cost accounting)** — ngân sách token tối đa cho từng tác tử.
- **Khả năng quan sát (Observability)** — truy vết luồng bàn giao, lưu trữ trạng thái để phát lại (replay).

Tất cả các yếu tố này đều được hiện thực hóa dựa trên bốn nguyên hàm nền tảng. Không có yếu tố nào trong số đó là một nguyên hàm mới cả.

````figure
a5-primitive-radar
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa bốn nguyên hàm trong khoảng 150 dòng mã Python thuần túy bằng thư viện chuẩn (stdlib). Không cần gọi LLM thật — mỗi tác tử là một chính sách lập trình sẵn (scripted policy) để người học tập trung hoàn toàn vào cấu trúc điều phối.

Tệp mã nguồn cung cấp:

- `Agent` — một dataclass gồm tên, prompt hệ thống, danh sách công cụ và hàm chính sách.
- `Handoff` — một hàm trả về tác tử mới.
- `SharedState` — một nhóm tin nhắn an toàn luồng (thread-safe).
- `Orchestrator` — ba biến thể: `StaticOrchestrator`, `HandoffOrchestrator`, và `LLMSelectorOrchestrator` (mô phỏng).

Chương trình minh họa chạy cùng một đường ống gồm ba tác tử (nghiên cứu → viết bài → đánh giá) thông qua cả ba kiểu bộ điều phối và in ra nhóm tin nhắn dùng chung ở bước cuối cùng. Bạn sẽ thấy rõ kết quả chỉ khác nhau ở chỗ *ai là người chọn bước tiếp theo*; bản thân các tác tử và trạng thái chia sẻ hoàn toàn giống hệt nhau qua các lần chạy.

Chạy thử chương trình:

```bash
python3 code/main.py
```

Kết quả mong đợi: ba lượt chạy của ba bộ điều phối, mỗi lượt tương ứng với một mẫu hình. Mỗi lượt in ra nhóm tin nhắn cuối cùng. Lượt chạy điều khiển bằng handoff sẽ dừng sớm hơn nếu tác tử nghiên cứu quyết định nó đã xong việc — đó chính là sự đánh đổi kinh điển của việc định tuyến bằng LLM được thu nhỏ lại.

## Ứng dụng thực tế (Use It)

Tệp `outputs/skill-primitive-mapper.md` là một kỹ năng phân tích giúp bạn đọc bất kỳ cơ sở mã hoặc tài liệu framework đa tác tử nào và trích xuất ra bản đồ ánh xạ bốn nguyên hàm. Hãy áp dụng nó khi có một bản phát hành framework mới để nắm bắt bản chất cốt lõi trong một đoạn văn ngắn trước khi tốn thời gian đọc sâu vào tài liệu chi tiết.

## Đưa vào sản phẩm (Ship It)

Trước khi quyết định áp dụng một framework mới cho dự án, hãy vẽ bản đồ ánh xạ nguyên hàm cho nó. Nếu bạn không thể vẽ được, tài liệu của framework đó chưa đầy đủ hoặc framework đó đang cố phát minh ra một nguyên hàm thứ năm (điều này cực kỳ hiếm — hãy kiểm tra xem liệu đó có phải chỉ là một biến thể của trạng thái dùng chung mà bạn chưa từng gặp hay không).

Hãy ghim chặt bản đồ ánh xạ này vào tài liệu kiến trúc hệ thống của bạn. Khi một thành viên mới gia nhập đội ngũ, hãy gửi cho họ bản đồ nguyên hàm trước khi gửi tài liệu API. Khi các framework nâng cấp phiên bản, hãy so sánh sự thay đổi trên bản đồ nguyên hàm thay vì đọc danh sách thay đổi chi tiết (changelog).

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py` ba lần với các chính sách tác tử khác nhau. Quan sát cách việc lựa chọn bộ điều phối làm thay đổi tập hợp các tác tử thực sự được kích hoạt.
2. Hiện thực hóa kiểu bộ điều phối thứ tư: điều phối dựa trên hàng đợi, trong đó các tác tử liên tục thăm dò trạng thái dùng chung để nhận việc. Tình huống khóa chết (deadlock) nào có thể phát sinh, và làm thế nào bạn phát hiện được nó?
3. Lấy tài liệu hướng dẫn bắt đầu nhanh của LangGraph (https://docs.langchain.com/oss/python/langgraph/workflows-agents) và viết lại nó dưới dạng bốn nguyên hàm. Khái niệm trừu tượng nào của LangGraph ánh xạ 1:1 và khái niệm nào chỉ là lớp vỏ tiện ích bọc ngoài?
4. Đọc hướng dẫn thực hành OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Xác định xem trong bốn nguyên hàm, Swarm làm cho nguyên hàm nào trở nên tiện dụng nhất, và đẩy nguyên hàm nào về phía người lập trình xử lý?
5. Tìm một framework trong bảng so sánh che giấu hoàn toàn trạng thái dùng chung. Giải thích điều gì sẽ bị gãy đổ khi các tác tử cần phối hợp xuyên suốt các lượt bàn giao mà không muốn phải đọc lại toàn bộ lịch sử từ đầu.

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| Tác tử (Agent) | "Một LLM có công cụ" | Bộ ba `(system_prompt, tools, model)`. Hoàn toàn phi trạng thái. |
| Bàn giao quyền điều khiển (Handoff) | "Chuyển giao quyền điều khiển" | Một lệnh gọi có cấu trúc chỉ định tác tử tiếp theo và tải trọng tùy chọn. Ba dạng hiện thực: trả về hàm, cạnh đồ thị, chọn người phát ngôn. |
| Trạng thái dùng chung (Shared state) | "Bộ nhớ" / "ngữ cảnh" | Thành phần có trạng thái duy nhất của hệ đa tác tử. Nhóm tin nhắn hoặc bảng đen. |
| Bộ điều phối (Orchestrator) | "Bộ điều hợp" | Thành phần quyết định ai chạy tiếp theo. Đồ thị tĩnh, bộ chọn LLM, điều khiển bằng handoff, hoặc điều khiển bằng hàng đợi. |
| Nguyên hàm (Primitive) | "Lớp trừu tượng" | Một trong bốn trục thiết kế mà mọi framework đều phải tham số hóa. Không phải là tính năng riêng của một framework cụ thể. |
| Nhóm tin nhắn (Message pool) | "Lịch sử trò chuyện chung" | Trạng thái dùng chung chứa toàn bộ lịch sử. Dễ phân tích lập luận nhưng mở rộng quy mô rất kém. |
| Trạng thái chiếu (Projected state) | "Góc nhìn bị giới hạn" | Góc nhìn vào trạng thái chia sẻ được thiết kế riêng theo từng vai trò. Khả năng mở rộng cao, đòi hỏi thiết kế lược đồ cẩn thận. |
| Chọn người phát ngôn (Speaker selection) | "Ai nói tiếp theo" | Mẫu hình điều phối trong đó một hàm (thường là LLM) chọn tác tử tiếp theo từ một nhóm có sẵn. |

## Tài liệu đọc thêm (Further Reading)

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — Bài viết mạch lạc nhất diễn giải mẫu hình điều phối bằng bàn giao handoff.
- [Tài liệu chính thức AutoGen](https://microsoft.github.io/autogen/stable/) — GroupChat kết hợp với bộ chọn người phát ngôn là tài liệu tham chiếu chuẩn mực cho việc điều phối bằng LLM.
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Điều phối bằng cạnh đồ thị và trạng thái dùng chung dựa trên bộ thu gọn reducer.
- [Giới thiệu CrewAI](https://docs.crewai.com/en/introduction) — Các tác tử dựa trên vai trò - mục tiêu - cốt truyện, các quy trình tuần tự / phân cấp.
- [AG2 (Dự án tiếp nối cộng đồng của AutoGen)](https://github.com/ag2ai/ag2) — Nhánh phát triển tiếp tục dòng AutoGen v0.2 sau khi Microsoft chuyển v0.4 sang chế độ bảo trì.
