# Đội ngũ Tác tử Phân vai — Vai trò, Nhiệm vụ, Quy trình

> Bốn nguyên thủy cốt lõi: Agent (Tác tử), Task (Nhiệm vụ), Crew (Đội ngũ tác tử), Process (Quy trình). Hai hình thái cấp cao: Crews (sự cộng tác tự chủ, dựa trên vai trò) và Flows (hướng sự kiện, mang tính tất định). CrewAI là bản triển khai tham chiếu năm 2026, và tài liệu của họ nêu rất thẳng thắn: "đối với bất kỳ ứng dụng nào sẵn sàng cho môi trường sản xuất (production), hãy bắt đầu với một Flow."

**Type:** Learn + Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Mục tiêu học tập

- Kể tên bốn nguyên thủy (primitives) của CrewAI (Agent, Task, Crew, Process) và phạm vi kiểm soát của từng thành phần.
- Phân biệt các quy trình Tuần tự (Sequential), Phân cấp (Hierarchical) và Đồng thuận (Consensus - dự kiến); lựa chọn đúng quy trình cho từng khối lượng công việc.
- Phân biệt giữa Crews (hợp tác tự chủ dựa trên vai trò) và Flows (hướng sự kiện, tất định), giải thích khuyến nghị production từ tài liệu chính thức.
- Kết nối các công cụ bằng bộ trang trí (decorator) `@tool` và lớp con `BaseTool`; suy luận về đầu ra có cấu trúc (structured outputs) so với văn bản tự do (free text).
- Nêu tên bốn loại bộ nhớ trong CrewAI và thời điểm từng loại phát huy hiệu quả.
- Triển khai một đội ngũ ba tác tử (nghiên cứu viên, người viết, biên tập viên) bằng Python thuần để tạo ra một bản tóm tắt nội dung.
- Nhận diện bốn chế độ lỗi (failure modes) phổ biến của CrewAI: phình to prompt (prompt-bloat), chi phí token của LLM quản lý (manager-LLM tax), bàn giao lỏng lẻo (brittle handoffs), và đưa trực tiếp Crew lên production (Crew-as-prod).

## Vấn đề

Các đội ngũ áp dụng các framework đa tác tử thường va phải cùng một bức tường. "Sự cộng tác tự chủ" nghe rất tuyệt vời trong các bản demo. Nhưng rồi khách hàng báo cáo lỗi và bạn cần khả năng tái hiện tất định (deterministic replay). Hoặc bộ phận tài chính hỏi mỗi lượt chạy của một đội ngũ do LLM định tuyến tiêu tốn bao nhiêu chi phí. Hoặc đội trực ca (on-call) cần biết tác tử nào bị đình trệ vào lúc 3 giờ sáng.

Các đội ngũ do LLM định tuyến tự do hoàn toàn không trả lời rành mạch được bất kỳ câu hỏi nào trong số đó. Đồ thị có hướng không chu trình (DAG) thuần túy giải quyết được tất cả, nhưng lại làm mất đi tính khám phá linh hoạt mà một tác tử động não ý tưởng (brainstorming) cần có.

Sự phân tách của CrewAI rất thành thật về sự đánh đổi này. Dùng Crews cho công việc hợp tác, phân vai, mang tính khám phá. Dùng Flows cho môi trường sản xuất hướng sự kiện, kiểm soát bằng mã nguồn và có thể kiểm toán. Cùng một framework, hai hình thái, lựa chọn tùy theo từng bề mặt bài toán.

## Khái niệm

### Bốn nguyên thủy

Bề mặt của CrewAI rất tinh gọn. Hãy ghi nhớ các yếu tố sau và phần còn lại chỉ là cấu hình.

- **Agent (Tác tử).** `role + goal + backstory + tools + (tùy chọn) llm`. Phần câu chuyện nền (backstory) đóng vai trò chịu lực cốt lõi. Nó định hình văn phong, khả năng phán đoán và thời điểm tác tử dừng lại. Công cụ (tools) là các hàm mà tác tử có thể gọi (chi tiết bên dưới).
- **Task (Nhiệm vụ).** `description + expected_output + agent + (tùy chọn) context + (tùy chọn) output_pydantic`. Một đơn vị công việc tái sử dụng. `expected_output` là bản hợp đồng cam kết đầu ra. `context` liệt kê các tác vụ thượng nguồn có đầu ra được truyền vào làm ngữ cảnh. `output_pydantic` ép buộc đầu ra theo khuôn mẫu có cấu trúc.
- **Crew (Đội ngũ tác tử).** Vỏ bọc chứa. Nắm giữ danh sách `agents`, danh sách `tasks`, chiến lược `process`, cùng các thiết lập tùy chọn `memory` + `verbose` + `manager_llm`.
- **Process (Quy trình).** Chiến lược thực thi. Sequential (Tuần tự), Hierarchical (Phân cấp), Consensus (Đồng thuận - dự kiến). Quyết định hình thái của lượt chạy.

Các tác tử không nhìn thấy nhau trực tiếp. Các nhiệm vụ tham chiếu đến các tác tử. Crew sắp xếp chuỗi nhiệm vụ. Process quyết định ai sẽ nhận nhiệm vụ tiếp theo. Đó là toàn bộ mô hình tư duy.

> **Đã kiểm chứng trên** CrewAI 0.86 (tháng 5/2026). Các phiên bản mới hơn có thể đổi tên hoặc hợp nhất các kiểu quy trình; hãy kiểm tra [Tài liệu Quy trình của CrewAI](https://docs.crewai.com/concepts/processes) trước khi phụ thuộc vào một cấu trúc cụ thể.

### Sequential so với Hierarchical so với Consensus

- **Sequential (Tuần tự).** Các nhiệm vụ chạy theo thứ tự khai báo. Đầu ra của nhiệm vụ N sẽ có sẵn dưới dạng `context` cho nhiệm vụ N+1. Chi phí thấp nhất. Dễ dự đoán nhất. Dùng khi thứ tự các bước là cố định.
- **Hierarchical (Phân cấp).** Một tác tử quản lý (gọi LLM riêng biệt) sẽ điều phối giữa các tác tử chuyên môn. CrewAI khởi tạo người quản lý này từ cấu hình `manager_llm` của bạn hoặc từ mặc định. Người quản lý sẽ chọn nhiệm vụ tiếp theo trong mỗi vòng và có quyền từ chối hoặc định tuyến lại. Sử dụng khi bạn có từ bốn chuyên gia trở lên và thứ tự thực thi thực sự phụ thuộc vào đầu ra trước đó.
- **Consensus (Đồng thuận).** Đang trong kế hoạch, hiện chưa được triển khai trong API công khai. Tài liệu giữ chỗ cho một quy trình dựa trên cơ chế bỏ phiếu trong tương lai. Hiện tại không nên phụ thuộc vào nó.

Quy trình phân cấp bổ sung thêm một lệnh gọi LLM ở mỗi vòng (dành cho người quản lý) đè lên mỗi lệnh gọi của chuyên gia. Chi phí token có thể tăng gấp ba lần trong một lượt chạy 5 bước. Chỉ trả chi phí này khi bạn thực sự cần khả năng định tuyến linh hoạt.

### Crews so với Flows

Đây là định hướng cấu trúc mà tài liệu dẫn dắt vào năm 2026.

- **Crew.** Sự tự chủ do LLM dẫn dắt. Framework quyết định hình thái tại thời điểm chạy (runtime). Phù hợp cho: nghiên cứu, động não, viết bản nháp đầu tiên, bất cứ nơi nào mà đường đi cũng là một phần của câu trả lời. Khó tái hiện lại. Khó viết kiểm thử. Chi phí tạo mẫu nhanh rẻ.
- **Flow.** Đồ thị hướng sự kiện do chính bạn làm chủ bằng mã nguồn. `@start` đánh dấu điểm bắt đầu. `@listen(topic)` đánh dấu một bước sẽ kích hoạt khi một bước khác phát ra chủ đề đó. Mỗi bước là mã Python thông thường (có thể gọi một Crew từ bên trong). Phù hợp cho: môi trường sản xuất (production). Có khả năng quan sát (observable). Có thể kiểm thử (testable). Tính tất định (deterministic).

Khuyến nghị production năm 2026 từ tài liệu chính thức: hãy bắt đầu với một Flow. Lồng các Crew vào dưới dạng các lệnh gọi `Crew.kickoff()` từ bên trong các bước của Flow khi tính tự chủ xứng đáng với chi phí của nó. Flow cung cấp cho bạn dấu vết kiểm toán (audit trail), Crew cung cấp khả năng khám phá sáng tạo. Hãy kết hợp chứ đừng loại trừ.

### Tích hợp công cụ

Có ba cách để cung cấp công cụ cho một Agent. Hãy chọn cách đơn giản nhất phù hợp với bài toán.

1. **Bộ trang trí `@tool`.** Các hàm thuần túy trở thành công cụ. Chữ ký hàm (signature) là lược đồ (schema); chuỗi tài liệu (docstring) là mô tả mà LLM đọc được. Tốt nhất cho các hàm bổ trợ đơn lẻ.

```python
from crewai.tools import tool

@tool("Search the web")
def search(query: str) -> str:
    """Return top results for the query."""
    return run_search(query)
```

2. **Lớp con `BaseTool`.** Công cụ dựa trên lớp với lược đồ đối số tường minh, hỗ trợ bất đồng bộ, cơ chế thử lại. Sử dụng khi công cụ có trạng thái (một client, bộ nhớ đệm) hoặc cần các đối số có cấu trúc.

```python
from crewai.tools import BaseTool
from pydantic import BaseModel

class SearchArgs(BaseModel):
    query: str
    limit: int = 10

class SearchTool(BaseTool):
    name = "web_search"
    description = "Search the web and return top results."
    args_schema = SearchArgs

    def _run(self, query: str, limit: int = 10) -> str:
        return self.client.search(query, limit=limit)
```

3. **Bộ công cụ tích hợp sẵn.** CrewAI cung cấp sẵn các bộ chuyển đổi chính chủ: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Tích hợp chỉ với một dòng import.

Đầu ra có cấu trúc sử dụng Pydantic. Truyền `output_pydantic=MyModel` vào Task. CrewAI sẽ xác thực phản hồi của LLM dựa trên mô hình và tự động ép kiểu hoặc thử lại. Hãy kết hợp điều này với chuỗi `expected_output` chặt chẽ. Đầu ra dạng văn bản tự do phù hợp cho bản nháp; đầu ra có cấu trúc là thứ mà các Flow hạ nguồn có thể tiêu thụ an toàn.

### Móc nối bộ nhớ

CrewAI cung cấp sẵn bốn loại bộ nhớ. Chúng có thể kết hợp: một Crew có thể kích hoạt cả bốn loại cùng lúc.

> **Đã kiểm chứng trên** CrewAI 0.86 (tháng 5/2026). Các phiên bản gần đây định tuyến mọi thứ qua một hệ thống `Memory` hợp nhất bao bọc bốn kho lưu trữ này. Mô hình khái niệm dưới đây vẫn giữ nguyên giá trị, nhưng giao diện lớp công khai có thể thu gọn về một điểm truy cập `Memory` duy nhất trong các phiên bản mới; hãy xem [tài liệu bộ nhớ CrewAI](https://docs.crewai.com/concepts/memory) để biết API hiện tại.

- **Ngắn hạn (Short-term).** Vùng đệm hội thoại trong một lượt chạy duy nhất. Bị xóa sạch khi kết thúc lượt chạy.
- **Dài hạn (Long-term).** Lưu bền vững xuyên suốt các lượt chạy. Được lưu trong cơ sở dữ liệu vector (mặc định là Chroma, có thể hoán đổi). Được truy xuất dựa trên độ tương đồng với nhiệm vụ hiện tại.
- **Thực thể (Entity).** Các sự kiện thực tế gắn với từng thực thể. "Khách hàng X đang sử dụng gói doanh nghiệp." Được đánh khóa theo thực thể chứ không phải theo độ tương đồng. Tồn tại xuyên suốt các lượt chạy.
- **Theo ngữ cảnh (Contextual).** Truy xuất tại thời điểm lắp ráp. Rút trích bộ nhớ liên quan ngay tại thời điểm Agent cần đến, không tải sẵn từ đầu.

Kích hoạt trên Crew bằng `memory=True` hoặc cấu hình riêng cho từng loại. Được hỗ trợ bởi một nhà cung cấp nhúng (embeddings) do bạn cấu hình (mặc định là OpenAI, có thể hoán đổi sang mô hình cục bộ). Bộ nhớ là một trong những điểm CrewAI khẳng định giá trị so với các framework nhẹ hơn; nếu dùng LangGraph thuần túy, bạn sẽ phải tự nối từng phần này bằng tay.

### Khi nào nên dùng đội ngũ phân vai

- Ba đến sáu tác tử với các vai trò được đặt tên rõ ràng và một quy trình làm việc cộng tác: Viết bản thảo, phản biện, lập kế hoạch, động não ý tưởng.
- Các bài toán định tuyến mà khả năng phán đoán của LLM về bước tiếp theo đóng góp trực tiếp vào giá trị công việc (Hierarchical).
- Bất cứ nơi nào mà nhóm làm việc cảm thấy thoải mái hơn khi đọc `role + goal + backstory` thay vì phải đọc một định nghĩa đồ thị phức tạp.

### Khi nào không nên dùng

- Các đồ thị DAG tất định với thứ tự nghiêm ngặt. Hãy dùng LangGraph (Bài 13). Cấu trúc đồ thị là lớp trừu tượng đúng đắn; việc ép vào vai trò của CrewAI chỉ gây thêm ma sát.
- Các yêu cầu về độ trễ dưới một giây (sub-second). Quy trình phân cấp làm tăng số lần khứ hồi mạng (round trips). Ngay cả quy trình tuần tự cũng tuần tự hóa các prompt chứa đầy câu chuyện nền và các đầu ra trước đó.
- Vòng lặp đơn tác tử. Hãy bỏ qua framework; một vòng lặp tác tử cơ bản (Bài 01) kết hợp với một sổ đăng ký công cụ (tool registry) sẽ gọn gàng hơn nhiều.

Bài 28 (Mẫu hình Điều phối) sẽ so sánh các mô hình tô-pô một cách trực diện. Tóm lại: CrewAI định vị vững chắc ở góc "hợp tác dựa trên vai trò".

### Cấu hình phụ thuộc

Độc lập với LangChain. Hỗ trợ Python 3.10 đến 3.13. Sử dụng trình quản lý gói `uv`. Về số lượt sao (star count): tham khảo [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (ảnh chụp tháng 5/2026). Đã có tài liệu tích hợp AWS Bedrock; các điểm chuẩn hiệu năng từ nhà cung cấp báo cáo tốc độ vượt trội đáng kể so với LangGraph trong các tác vụ Hỏi-Đáp (QA), nhưng phương pháp luận (tập dữ liệu, phần cứng, chỉ số đánh giá) không được công bố công khai, do đó các con số từ nhà cung cấp framework chỉ nên coi mang tính định hướng.

### Những cạm bẫy thường gặp

- **Phình to prompt từ câu chuyện nền (Prompt-bloat from backstories).** Một câu chuyện nền dài 2000 từ cho mỗi tác tử trong một đội ngũ năm tác tử sẽ đốt sạch ngân sách ngữ cảnh trước cả khi lệnh gọi công cụ đầu tiên diễn ra. Hãy giữ câu chuyện nền dưới 200 từ. Tái sử dụng các cụm từ giữa các tác tử; đừng lặp lại phong cách công ty năm lần liên tiếp.
- **Chi phí token từ LLM quản lý (Manager-LLM token tax).** Quy trình phân cấp bổ sung thêm một lệnh gọi LLM quản lý trước mỗi lệnh gọi của chuyên gia. Với một đội ngũ 5 nhiệm vụ, điều đó đồng nghĩa với 10 lệnh gọi LLM thay vì 5, và lệnh gọi của quản lý phải gánh toàn bộ danh sách nhiệm vụ cộng với các đầu ra trước đó. Hãy chuyển sang quy trình Tuần tự (Sequential) trừ khi việc định tuyến thực sự phụ thuộc vào đầu ra.
- **Bàn giao lỏng lẻo (Brittle handoffs).** `expected_output` của Nhiệm vụ N là "một dàn ý". Nhiệm vụ N+1 đọc nó dưới dạng `context` và cố gắng phân tích ba phần. Nhưng LLM lại tạo ra bốn phần. Tác tử hạ nguồn sẽ ứng biến tùy tiện. Hãy khắc phục bằng `output_pydantic` trên Nhiệm vụ N để Nhiệm vụ N+1 đọc một đối tượng có định kiểu, thay vì văn bản tự do.
- **Đưa trực tiếp Crew lên production (Crew-as-prod).** Triển khai trực tiếp một Crew tự do lên môi trường sản xuất mà không có vỏ bọc Flow. Độ biến thiên đầu ra rất cao; không thể tái hiện lại lỗi; đội trực ca không thể đối chiếu giữa lượt chạy hỏng và lượt chạy tốt. Hãy luôn bao bọc bằng một Flow.

```figure
ae-crew-vs-flow
```

## Xây dựng

`code/main.py` triển khai các phiên bản Python thuần cho cả hai hình thái trên cùng một đội ngũ ba tác tử. Phụ thuộc duy nhất ngoài thư viện chuẩn là numpy, dùng để tính toán độ tương đồng giả lập.

Cấu trúc:

- Các dataclass `Agent`, `Task` khớp với giao diện của CrewAI.
- `SequentialCrew.kickoff(inputs)` chạy các nhiệm vụ theo thứ tự khai báo, luân chuyển các đầu ra làm `context`.
- `HierarchicalCrew.kickoff(topic)` bổ sung một Agent quản lý chọn chuyên gia tiếp theo ở mỗi vòng, dừng lại khi đạt trạng thái "done".
- `Flow` với các bộ trang trí `@start` và `@listen(topic)`, một vòng lặp sự kiện nhỏ và khả năng ghi dấu vết (trace).
- Bộ trang trí `tool(name)` mô phỏng hình thái `@tool` của CrewAI.
- `Memory` với các kho lưu trữ `short_term`, `long_term`, `entity`; độ tương đồng giả lập sử dụng numpy.
- Các phản hồi LLM giả lập là các chuỗi ký tự cố định dựa trên vai trò kết hợp tiền tố đầu vào. Không gọi mạng. Hoàn toàn tất định.

Bản demo cụ thể: đội ngũ gồm nghiên cứu viên, người viết, biên tập viên cùng tạo ra một bản tóm tắt ngắn về "kỹ thuật tác tử năm 2026 (agent engineering 2026)". Nghiên cứu viên thu thập các nguồn tin (giả lập). Người viết soạn thảo bản nháp. Biên tập viên gọt giũa hoàn thiện. Cùng một đội ngũ đó được chạy qua một Flow để minh họa hình thái tất định.

Chạy thử:

```bash
python3 code/main.py
```

Dấu vết thực thi bao gồm: đội ngũ tuần tự luân chuyển đầu ra qua `context`, đội ngũ phân cấp với các lựa chọn của quản lý (nghiên cứu viên, người viết, biên tập viên, sau đó là "done"), flow chạy ba bước tương tự với các chủ đề rõ ràng (`researched`, `drafted`, `edited`), các lệnh gọi công cụ được định tuyến qua `@tool`, và bộ nhớ dài hạn tồn tại xuyên suốt qua hai lần khởi động (kickoffs).

Dấu vết của Crew rất uyển chuyển; về nguyên tắc người quản lý có thể đảo lại thứ tự. Dấu vết của Flow thì cố định. Sự lựa chọn giữa hai cách tiếp cận đó chính là bài học cốt lõi.

## Sử dụng

- **CrewAI Flow** cho môi trường sản xuất. Ngay cả khi Flow chỉ có một bước duy nhất gọi `Crew.kickoff()`. Flow tạo ra ranh giới kiểm toán rõ ràng.
- **CrewAI Crew (Sequential)** cho các công việc cộng tác có thứ tự rõ ràng, đặc biệt là các vòng lặp tạo bản thảo đầu tiên và đánh giá.
- **CrewAI Crew (Hierarchical)** khi việc định tuyến phụ thuộc vào kết quả đầu ra và bạn có từ bốn chuyên gia trở lên.
- **LangGraph** (Bài 13) cho các máy trạng thái tường minh, khôi phục bền vững, thứ tự nghiêm ngặt.
- **AutoGen v0.4** (Bài 14) cho tính đồng thời theo mô hình actor và cô lập sự cố.
- **OpenAI Agents SDK** (Bài 16) cho các sản phẩm ưu tiên hệ sinh thái OpenAI với cơ chế bàn giao (handoffs) và rào chắn an toàn (guardrails).
- **Claude Agent SDK** (Bài 17) cho các sản phẩm ưu tiên Claude với các tác tử phụ và kho lưu trữ phiên làm việc.

## Chuyển giao

`outputs/skill-crew-or-flow.md` giúp lựa chọn giữa Crew và Flow cho một nhiệm vụ cụ thể và dựng sẵn cấu trúc khung triển khai tối giản. Tự động từ chối nếu định nghĩa Crew thiếu backstory, Flow thiếu các chủ đề tường minh, hoặc dùng Hierarchical với số chuyên gia ít hơn ba.

## Cạm bẫy

- **Xem câu chuyện nền chỉ là gia vị văn phong.** Nó thực sự định hình kết quả đầu ra. Hãy thử nghiệm ba biến thể cho mỗi tác tử; sự biến thiên là có thật. Chọn một bản chuẩn và cố định nó.
- **Bỏ qua `expected_output`.** Nếu không có hợp đồng cam kết cho từng nhiệm vụ, các nhiệm vụ hạ nguồn sẽ phải tiếp nhận bất kỳ thứ gì mà LLM sinh ra. Crew vẫn chạy; nhưng kiểm toán sẽ thất bại.
- **Luôn bật bộ nhớ.** Bộ nhớ dài hạn ghi dữ liệu sau mỗi lượt chạy. Cơ sở dữ liệu vector phình to. Quá trình truy xuất trở nên nhiễu loạn. Hãy giới hạn phạm vi ghi bộ nhớ cho các nhiệm vụ mà thông tin thực sự mang tính bền vững lâu dài.
- **Độ trôi lệch prompt của người quản lý (Manager prompt drift).** Prompt của người quản lý trong quy trình phân cấp là ngầm định. Nếu việc định tuyến có biểu hiện bất thường, hãy kết xuất prompt ở chế độ chi tiết (verbose mode) và đọc kỹ lại.
- **Tác dụng phụ của công cụ trong Crew.** Một Crew có thể gọi một công cụ nhiều lần hơn dự kiến. Các thao tác POST, DELETE, thanh toán thuộc về bước của Flow, tuyệt đối không bao giờ được đặt trong một công cụ của Crew.

## Bài tập

1. Chuyển đổi đội ngũ Tuần tự (Sequential crew) thành một Flow. Đếm số điểm tiếp xúc mà độ biến thiên giảm đi. Ghi nhận những điểm mà tính dễ đọc của mã nguồn bị giảm sút.
2. Thêm bộ nhớ thực thể (entity memory) vào đội ngũ: các dữ kiện về khách hàng được lưu giữ qua các lượt khởi động (kickoffs). Xác minh rằng việc truy xuất lấy đúng thông tin thực thể.
3. Triển khai một quy trình Phân cấp (Hierarchical) trong đó người quản lý từ chối định tuyến tới biên tập viên cho đến khi đầu ra của người viết có ít nhất ba đoạn văn. Ghi lại dấu vết quá trình thử lại.
4. Xây dựng một lớp con `BaseTool` cho tác vụ tìm kiếm web (giả lập). So sánh cấu trúc dấu vết thực thi so với phiên bản dùng bộ trang trí `@tool`.
5. Thêm `output_pydantic=Brief` vào nhiệm vụ biên tập viên, trong đó `Brief` có các trường `title`, `summary`, `sections`. Làm cho nhiệm vụ của người viết xuất ra JSON sai định dạng một lần; xác minh hành vi tự động thử lại của CrewAI trong dấu vết thực thi.
6. Đọc phần giới thiệu trong tài liệu CrewAI. Chuyển đổi mã đồ chơi sang API `crewai` thực tế. Bản thư viện chuẩn đã bỏ qua những bảo đảm nào?
7. Kết nối AgentOps hoặc Langfuse (Bài 24) vào một lượt chạy thực tế. Những dấu vết nào mà bạn đã bỏ lỡ trong phiên bản thư viện chuẩn?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Tác tử (Agent) | "Hình tượng (Persona)" | Vai trò + mục tiêu + câu chuyện nền + công cụ |
| Nhiệm vụ (Task) | "Đơn vị công việc" | Mô tả + đầu ra kỳ vọng + người phụ trách + đầu ra có cấu trúc tùy chọn |
| Đội ngũ tác tử (Crew) | "Nhóm tác tử" | Khung chứa cho Agents + Tasks + Process |
| Quy trình (Process) | "Chiến lược thực thi" | Tuần tự (Sequential) / Phân cấp (Hierarchical) / Đồng thuận (Consensus - dự kiến) |
| Luồng điều phối (Flow) | "Quy trình tất định" | Hướng sự kiện, kiểm soát bằng mã nguồn, có thể kiểm thử |
| Câu chuyện nền (Backstory) | "Prompt định hình nhân vật" | Thành phần định hình văn phong và khả năng phán đoán cho Agent |
| `@tool` | "Công cụ hàm" | Bộ trang trí biến một hàm thông thường thành công cụ mà Agent có thể gọi |
| `BaseTool` | "Công cụ lớp" | Công cụ dựa trên lớp với lược đồ đối số, thử lại, hỗ trợ bất đồng bộ |
| Bộ nhớ thực thể (Entity memory) | "Dữ kiện theo thực thể" | Bộ nhớ gắn liền với một khách hàng / tài khoản / vấn đề cụ thể |
| Bộ nhớ dài hạn (Long-term memory) | "Bộ nhớ xuyên suốt các lượt chạy" | Bộ nhớ hỗ trợ bởi vector tồn tại giữa các lượt khởi động |
| Bộ nhớ ngữ cảnh (Contextual memory) | "Truy xuất đúng lúc (Just-in-time)" | Bộ nhớ được trích xuất ngay tại thời điểm Agent cần |
| LLM quản lý (Manager LLM) | "Tác tử định tuyến" | LLM bổ sung trong quy trình Phân cấp chịu trách nhiệm chọn nhiệm vụ tiếp theo |
| `expected_output` | "Hợp đồng nhiệm vụ" | Chuỗi ký tự chỉ định cho Agent (và kiểm toán) cấu trúc đầu ra cần trả về |

## Tài liệu đọc thêm

- [Giới thiệu tài liệu CrewAI](https://docs.crewai.com/en/introduction): các khái niệm và lộ trình triển khai production được khuyến nghị
- [Hướng dẫn về CrewAI Flows](https://docs.crewai.com/en/concepts/flows): hình thái hướng sự kiện, `@start`, `@listen`
- [Tài liệu tham khảo công cụ CrewAI](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, các bộ công cụ tích hợp sẵn
- [Bộ nhớ trong CrewAI](https://docs.crewai.com/en/concepts/memory): ngắn hạn, dài hạn, thực thể, theo ngữ cảnh
- [Anthropic, Xây dựng các Tác tử Hiệu quả (Building Effective Agents)](https://www.anthropic.com/research/building-effective-agents): khi nào đa tác tử mang lại hiệu quả và khi nào không
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview): giải pháp thay thế dựa trên máy trạng thái
