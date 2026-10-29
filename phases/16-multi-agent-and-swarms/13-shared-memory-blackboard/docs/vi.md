# Bộ nhớ Dùng chung và Mô hình Bảng thông tin (Shared Memory and Blackboard Patterns)

> Hai hướng tiếp cận đang cùng tồn tại trong các hệ đa tác tử (multi-agent systems) năm 2026: **bể thông điệp** (message pool - nơi mọi tác tử đều thấy thông điệp của nhau, như trong AutoGen GroupChat hay MetaGPT) và **bảng thông tin kèm cơ chế đăng ký** (blackboard with subscription - nơi các tác tử đăng ký nhận các sự kiện liên quan, như trong Context-Aware MCP hay khung làm việc Matrix). Cả hai đều là thành phần duy nhất lưu giữ trạng thái (stateful) của một hệ đa tác tử — đồng nghĩa với việc đây chính là nơi ẩn chứa những lỗi ngầm phức tạp nhất. Dạng lỗi điển hình là **nhiễm độc bộ nhớ** (memory poisoning): một tác tử sinh ảo giác (hallucination) ra một "sự thật", các tác tử khác coi đó là dữ liệu đã xác thực, và độ chính xác suy thoái dần theo cách khó gỡ lỗi hơn nhiều so với một sự cố sập hệ thống (crash) lập tức. Bài học này sẽ xây dựng cả hai cấu trúc từ thư viện chuẩn (stdlib), tiêm thử nghiệm một cuộc tấn công gây nhiễm độc, và trình bày ba cơ chế giảm thiểu rủi ro thực sự hiệu quả trong môi trường sản xuất (production).

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 phút

## Vấn đề

Các hệ đa tác tử luôn cần một không gian để các tác tử chia sẻ dữ kiện. Một phương án hiển nhiên là "truyền mọi thứ qua thông điệp" — nhưng cách này thực chất là tái phát minh trạng thái dùng chung (shared state) với chi phí sao chép dữ liệu dư thừa. Một phương án khác là "cung cấp cho mọi tác tử một nhật ký toàn cục (global log)" — nhưng các nhật ký toàn cục sẽ phình to không giới hạn và rất dễ bị nhiễm độc. Lựa chọn thứ ba là "chiếu một góc nhìn riêng cho từng tác tử (project a view per agent)" — có khả năng mở rộng tốt nhưng đòi hỏi cấu trúc lược đồ (schema) rất nặng nề.

Khi một trong các tác tử gặp ảo giác và ghi thông tin sai lệch đó vào trạng thái dùng chung, mọi tác tử phía sau (downstream agents) đọc trạng thái đó sẽ tiếp nhận thông tin ảo giác như một sự thật hiển nhiên. Đến khi con người phát hiện ra, chuỗi lập luận (reasoning chain) đã trải qua năm bước suy diễn và nguyên nhân gốc rễ lại nằm ở thông điệp thứ ba từng được ghi từ rất lâu trước đó. Việc gỡ lỗi suy giảm độ chính xác trong hệ đa tác tử khó hơn nhiều so với việc khắc phục một sự cố dừng chương trình đột ngột.

Đây chính là hiện tượng nhiễm độc bộ nhớ (memory poisoning). Nó là họ lỗi được ghi nhận nhiều thứ hai trong phân loại lỗi MAST (Cemri et al., arXiv:2503.13657) và mang tính cấu trúc: bất kỳ thiết kế bộ nhớ dùng chung nào thiếu vắng cơ chế truy xuất nguồn gốc (provenance) và thiếu một tác tử thẩm định độc lập không có quyền ghi sẽ sớm muộn bộc lộ lỗi này.

## Khái niệm

### Hai cấu trúc liên kết chính (Topologies)

**Bể thông điệp toàn phần (Full message pool).** Mọi tác tử đều đọc mọi thông điệp. AutoGen GroupChat và MetaGPT sử dụng mô hình này. Nó đơn giản, minh bạch, dễ thanh tra, nhưng không thể mở rộng quy mô vượt quá ~10 tác tử vì cửa sổ ngữ cảnh (context window) của mỗi tác tử sẽ nhanh chóng bị lấp đầy bởi công việc của các tác tử khác.

```mermaid
flowchart LR
    A["Tác tử A (agent-A)"] -->|ghi (write)| Pool["Bể thông điệp<br/>(message pool / nhật ký toàn cục)"]
    B["Tác tử B (agent-B)"] -->|ghi (write)| Pool
    C["Tác tử C (agent-C)"] -->|ghi (write)| Pool
    Pool -->|đọc (read)| D["Tác tử D (agent-D)"]
    Pool -->|đọc (read)| E["Tác tử E (agent-E)"]
    Pool -->|đọc (read)| F["Tác tử F (agent-F)"]
```

**Bảng thông tin kèm cơ chế đăng ký (Blackboard with subscription).** Các tác tử khai báo mối quan tâm theo chủ đề (topics); lớp nền bên dưới sẽ chỉ điều hướng các thông điệp có liên quan. CA-MCP (arXiv:2601.11595) và khung làm việc phi tập trung Matrix (arXiv:2511.21686) sử dụng kiến trúc này. Khả năng mở rộng cao hơn, nhưng đòi hỏi thiết kế lược đồ cẩn thận ngay từ đầu để các kênh đăng ký thực sự mang lại ý nghĩa.

```mermaid
flowchart LR
    subgraph BB["Bảng thông tin (Blackboard)"]
        T1["Chủ đề: prices (giá cả)"]
        T2["Chủ đề: orders (đơn hàng)"]
        T3["Chủ đề: alerts (cảnh báo)"]
    end
    A["Tác tử A (agent-A)"] -->|xuất bản (pub)| T1
    B["Tác tử B (agent-B)"] -->|xuất bản (pub)| T2
    C["Tác tử C (agent-C)"] -->|xuất bản (pub)| T3
    T1 -->|gửi tới| D["Tác tử D (đã đăng ký)"]
    T2 -->|gửi tới| E["Tác tử E (đã đăng ký)"]
    T3 -->|gửi tới| F["Tác tử F (đã đăng ký)"]
```

### Thời điểm tối ưu của từng mô hình

- **Bể thông điệp toàn phần** phát huy tối đa khi số lượng tác tử ít (< 10), mang tính không đồng nhất (heterogeneous), và phiên đối thoại diễn ra trong phạm vi ngắn (short-horizon). Việc lập luận xem ai đã nói gì trở nên dễ dàng khi tất cả mọi người đều quan sát thấy mọi thứ.
- **Bảng thông tin** chiếm ưu thế khi số lượng tác tử lớn, đồng nhất về vai trò nhưng đông đảo về số lượng thể hiện (các bầy đàn - swarms), và phiên tương tác kéo dài. Cơ chế định tuyến giúp tiết kiệm đáng kể chi phí token và tránh gây ô nhiễm ngữ cảnh.

Các hệ thống trong môi trường sản xuất thường kết hợp cả hai: một bể thông điệp toàn phần quy mô nhỏ ở tầng trên cùng (lớp lập kế hoạch - planning layer), và các bảng thông tin chuyên biệt ở tầng dưới (lớp thực thi - worker layer).

### Kịch bản minh họa nhiễm độc bộ nhớ

Ba tác tử cùng phối hợp thực hiện một tác vụ nghiên cứu. Tác tử A đóng vai trò thu thập thông tin (retrieval agent). Tác tử B là tác tử tóm tắt (summarizer). Tác tử C là nhà phân tích (analyst).

1. A thu thập một trang tài liệu và ghi thông điệp vào trạng thái dùng chung: "Nghiên cứu báo cáo độ chính xác cải thiện 42%."
2. Nhưng trang tài liệu thực tế chỉ ghi "cải thiện 4.2%." Tác tử A đã sinh ảo giác về dấu thập phân.
3. B khi đọc trạng thái dùng chung liền ghi nhận: "Ghi nhận mức tăng độ chính xác lớn lên tới 42% (nguồn: A)."
4. C đọc trạng thái dùng chung và đưa ra kết luận: "Khuyến nghị áp dụng ngay — mức tăng trưởng 42% mang tính đột phá."
5. Báo cáo cuối cùng trích dẫn con số 42% vốn chưa từng tồn tại trên thực tế.

Không có tác tử nào bị lỗi dừng chương trình. Không có bài kiểm thử nào thất bại. Toàn bộ hệ thống vẫn "hoạt động trơn tru". Ảo giác đã thâm nhập từ ngữ cảnh của một tác tử vào lập luận của tất cả các tác tử tiếp theo thông qua trạng thái dùng chung.

### Bản chất cấu trúc của vấn đề

Nếu không có trạng thái dùng chung, ảo giác của tác tử A sẽ bị cô lập trong ngữ cảnh của chính nó. Các tác tử phía sau sẽ phải tự truy xuất hoặc tự suy luận lại, nhờ đó có cơ hội phát hiện sai lệch. Nhưng với trạng thái dùng chung ngây thơ (naive shared state), ngữ cảnh của A biến thành ngữ cảnh chung của tất cả, và thông tin sai lệch được hợp thức hóa thành chân lý.

Vấn đề cốt lõi không nằm ở bản thân trạng thái dùng chung — mà là trạng thái dùng chung **thiếu vắng cơ chế truy xuất nguồn gốc (provenance) và thiếu một tác tử thẩm định độc lập**. Ba kỹ thuật giảm thiểu sau đây giải quyết triệt để vấn đề:

1. **Gắn thuộc tính nguồn gốc (provenance) trên mọi thao tác ghi.** Mỗi mục nhập trong trạng thái dùng chung phải ghi nhận ai là người viết, thời điểm viết, dưới prompt nào, và (nếu có) nguồn tài liệu nào được trích dẫn. Các tác tử tiếp nhận dữ liệu sẽ duy trì mức độ hoài nghi tương ứng với nguồn gốc được khai báo.
2. **Đánh phiên bản các lượt ghi; coi bộ nhớ là chỉ ghi thêm (append-only).** Một thao tác sửa lỗi là một mục nhập mới thay thế mục nhập cũ, chứ không phải một thao tác ghi đè tại chỗ (in-place update). Nhờ đó vết tích kiểm toán (audit trail) luôn được bảo toàn.
3. **Duy trì ít nhất một tác tử không có quyền ghi vào trạng thái dùng chung.** Tác tử thẩm định chỉ đọc (read-only verifier) này sẽ lấy mẫu các mục nhập, truy xuất lại nguồn tài liệu gốc, và gắn cờ cảnh báo các điểm mâu thuẫn. Dù nó vẫn đọc bể dữ liệu và có thể bị đánh lừa bởi dữ liệu sai, nó luôn đối chiếu chéo với nguồn ban đầu và tuyệt đối không thể ghi, do đó phán quyết của nó không bao giờ tiêm ngược một dữ kiện sai vào trạng thái dùng chung.

### Tiền lệ lịch sử của Bảng thông tin (Hayes-Roth, 1985)

Mô hình bảng thông tin đã ra đời trước các tác tử LLM tới bốn thập kỷ. Hayes-Roth (1985, "A Blackboard Architecture for Control") đã mô tả các Nguồn Tri thức (Knowledge Sources) chuyên biệt có nhiệm vụ quan sát một bảng thông tin toàn cục, đóng góp các giải pháp bộ phận, và kích hoạt các nguồn tri thức khác. Mô hình bảng thông tin năm 2026 (CA-MCP, Matrix) chính là mô hình kinh điển đó với các tác tử LLM đóng vai trò là Nguồn Tri thức và các khối JSON đóng vai trò là các giải pháp bộ phận. Các tài liệu nghiên cứu kinh điển trước đây đã cung cấp lời giải toàn diện cho các bài toán tranh chấp ghi (write contention), điều khiển cơ hội (opportunistic control), và tính nhất quán mà các hệ thống hiện đại đang từng bước tái khám phá.

### Chiếu góc nhìn riêng so với Góc nhìn toàn phần (Projection vs Full View)

Một bảng thông tin thuần túy cung cấp cùng một hình chiếu (phạm vi theo chủ đề) cho mọi bên đăng ký. Một thiết kế tối ưu hơn là **chiếu góc nhìn riêng theo từng tác tử (per-agent projection)**: mỗi tác tử nhận được một góc nhìn tùy biến phù hợp với vai trò của nó. Cơ chế hàm thu gọn trạng thái (state reducers) của LangGraph là hiện thân tiêu biểu năm 2026 — hàm thu gọn sẽ gập trạng thái toàn cục thành một lát cắt dữ liệu đặc thù cho vai trò đó.

Chiếu góc nhìn riêng mở rộng quy mô tốt hơn nhưng bắt buộc phải có lược đồ dữ liệu rõ ràng. Nếu thiếu lược đồ, bạn sẽ buộc phải dựng lại các phép chiếu tùy tiện ngay bên trong prompt của từng tác tử.

### Các mô hình xử lý tranh chấp ghi (Write-Contention Patterns)

Nhiều tác tử cùng ghi dữ liệu đồng thời là một bài toán xử lý đồng thời (concurrency), không đơn thuần là bài toán LLM. Có ba mô hình hiệu quả:

- **Bên ghi tuần tự (Sequential writer / single producer).** Mọi thao tác ghi đều đi qua một tác tử điều phối duy nhất để tuần tự hóa. Đơn giản, nhưng dễ tạo thành điểm nghẽn.
- **Kiểm soát đồng thời lạc quan kèm đánh phiên bản (Optimistic concurrency with versioning).** Mỗi mục nhập mang một số phiên bản; bên ghi sẽ thất bại nếu phát hiện lệch phiên bản và phải thử lại. Đây là kỹ thuật cơ sở dữ liệu kinh điển.
- **Phân vùng theo chủ đề (Topic partitioning).** Các tác tử khác nhau sở hữu các chủ đề khác nhau. Không xảy ra tranh chấp chéo chủ đề. Mô hình này đòi hỏi thiết kế ranh giới phân vùng rành mạch.

Đa số các khung làm việc năm 2026 mặc định chọn cơ chế ghi tuần tự vì các lượt gọi LLM đủ chậm để hiếm khi xảy ra xung đột thực sự và điểm nghẽn không gây ảnh hưởng quá lớn.

### Tác tử thẩm định không có quyền ghi (The Unwritable Verifier)

Biện pháp bảo vệ then chốt nhất chính là tác tử thẩm định chỉ đọc. Các nguyên tắc triển khai:

- Tác tử thẩm định dùng chung trạng thái với toàn đội (đọc từ bảng thông tin hoặc bể dữ liệu).
- Tác tử thẩm định không có con trỏ ghi (write handle) vào trạng thái dùng chung — nó chỉ được phép xuất dữ liệu ra một kênh kiểm định riêng biệt.
- Tác tử thẩm định độc lập truy xuất lại các nguồn được trích dẫn trong các mục ghi. Gắn cờ cảnh báo bất đồng.
- Kết quả thẩm định được định tuyến tới con người hoặc một tác tử ra quyết định độc lập, tuyệt đối không đưa ngược trở lại bể dữ liệu chung.

Nếu thiếu sự phân tách này, kết quả của tác tử thẩm định sẽ trở thành các mục nhập mới trong bể, đồng nghĩa với việc bể dữ liệu bị nhiễm độc sẽ làm tha hóa chính tác tử thẩm định, kéo theo toàn bộ kết quả kiểm định bị sai lệch.

````figure
swarm-blackboard
````

## Xây dựng thực tế

Tệp `code/main.py` triển khai cả hai cấu trúc liên kết bằng Python tiêu chuẩn (stdlib), kèm theo một kịch bản tấn công gây nhiễm độc mô phỏng và ba cơ chế giảm thiểu.

- `MessagePool` — nhật ký chỉ ghi thêm (append-only log) an toàn luồng (thread-safe) với khả năng đọc toàn phần.
- `Blackboard` — cơ chế pub/sub theo chủ đề với các kênh đăng ký riêng theo từng tác tử.
- `ProvenanceEntry` — mỗi thao tác ghi đều lưu trữ `(writer, timestamp, prompt_hash, source_uri)`.
- `PoisoningScenario` — vận hành tác vụ nghiên cứu gồm ba tác tử, trong đó tác tử A sinh ảo giác về dấu thập phân. In ra báo cáo cuối cùng.
- `Verifier` — tác tử chỉ đọc có nhiệm vụ truy xuất lại nguồn và gắn cờ cảnh báo bất đồng. Chạy lại cùng kịch bản khi có mặt tác tử thẩm định.

Chạy chương trình:

```bash
python3 code/main.py
```

Đầu ra kỳ vọng:
- Lượt 1 (không có tác tử thẩm định): con số ảo giác 42% lan truyền sang báo cáo cuối cùng.
- Lượt 2 (có tác tử thẩm định): tác tử thẩm định gắn cờ cảnh báo điểm mâu thuẫn, bể dữ liệu bị gắn nhãn "flagged", và báo cáo cuối cùng đính kèm thông báo đính chính.

## Sử dụng trong thực tế

Tệp `outputs/skill-memory-auditor.md` là một kỹ năng kiểm toán (skill) dùng để rà soát thiết kế bộ nhớ dùng chung của bất kỳ hệ đa tác tử nào về nguồn gốc dữ liệu, đánh phiên bản và sự tách biệt của tác tử thẩm định. Hãy chạy kỹ năng này trên các kiến trúc đa tác tử mới trước khi đưa vào sản xuất.

## Đưa vào sản xuất

Đối với mọi thiết kế bộ nhớ dùng chung:

- Lưu lại nguồn gốc (provenance) trên từng thao tác ghi: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Thiết lập nhật ký ở chế độ chỉ ghi thêm. Các thao tác đính chính là các mục nhập mới tham chiếu đến mục nhập bị thay thế.
- Triển khai ít nhất một tác tử thẩm định chỉ đọc có quyền truy cập độc lập vào nguồn tài liệu.
- Định tuyến kết quả của tác tử thẩm định sang một kênh riêng biệt, không đưa ngược lại vào bể dùng chung.
- Ghi nhật ký tỷ lệ các thao tác ghi mang tính thay thế — tỷ lệ này gia tăng là tín hiệu cảnh báo sớm về các mẫu hình ảo giác đang xuất hiện.

## Bài tập thực hành

1. Chạy `code/main.py`. Xác nhận lượt 1 để lọt thông tin ảo giác và lượt 2 chặn bắt thành công.
2. Thêm một trường hợp ảo giác thứ hai: tác tử B tự ý bịa ra quy mô tập dữ liệu. Tác tử thẩm định phải phát hiện ra cả hai mà không cần phải tinh chỉnh thủ công cho từng trường hợp.
3. Chuyển bể thông điệp toàn phần sang bảng thông tin phân vùng theo chủ đề (`prices`, `summaries`, `analyses`). Phân vùng theo chủ đề khiến những kịch bản nhiễm độc nào khó xảy ra hơn, và nó bất lực trước những kịch bản nào?
4. Đọc tài liệu của Hayes-Roth (1985, "A Blackboard Architecture for Control"). Chỉ ra hai mô hình điều khiển trong bài báo chưa được thảo luận trong bài học này mà các hệ thống năm 2026 có thể học hỏi.
5. Đọc tài liệu CA-MCP (arXiv:2601.11595). Ánh xạ Kho Ngữ cảnh Dùng chung (Shared Context Store) của nó với lớp `MessagePool` hoặc `Blackboard` trong `code/main.py`. CA-MCP bổ sung thêm những thành phần nguyên thủy nào bên trên?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Bể thông điệp (Message pool) | "Lịch sử trò chuyện dùng chung" | Nhật ký chỉ ghi thêm (append-only) mà mọi tác tử đều đọc. Minh bạch hoàn toàn nhưng khả năng mở rộng kém. |
| Bảng thông tin (Blackboard) | "Không gian làm việc chung" | Hệ thống pub/sub theo chủ đề. Tác tử chỉ đăng ký các chủ đề liên quan. Khả năng mở rộng cao hơn. |
| Truy xuất nguồn gốc (Provenance) | "Ai đã viết cái gì" | Siêu dữ liệu đính kèm trên mỗi lượt ghi: tác tử ghi, dấu thời gian, prompt, nguồn trích dẫn. |
| Nhiễm độc bộ nhớ (Memory poisoning) | "Ảo giác lan truyền" | Lỗi của một tác tử lọt vào trạng thái dùng chung, khiến các tác tử phía sau mặc định coi là sự thật. |
| Chỉ ghi thêm (Append-only) | "Không sửa đổi tại chỗ" | Các sửa đổi là các mục nhập mới thay thế mục cũ. Bảo toàn nguyên vẹn vết tích kiểm toán. |
| Thẩm định không quyền ghi (Unwritable verifier) | "Kiểm toán viên độc lập" | Tác tử chỉ đọc có nhiệm vụ lấy lại nguồn gốc độc lập và cảnh báo các điểm bất thường. |
| Chiếu góc nhìn (Projection) | "Góc nhìn thu hẹp" | Khung nhìn tùy biến cho từng tác tử được tính toán từ trạng thái toàn cục. LangGraph reducers là trường hợp kinh điển. |
| Nguồn Tri thức (Knowledge Source) | "Tác tử chuyên biệt" | Thuật ngữ năm 1985 của Hayes-Roth chỉ một thành phần tham gia vào bảng thông tin. |

## Tài liệu tham khảo

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — Hệ thống phân loại lỗi MAST; nhiễm độc bộ nhớ là một phân họ của lỗi phối hợp.
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Kho ngữ cảnh dùng chung dành cho việc điều phối nhiều máy chủ MCP.
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — Bảng thông tin dựa trên hàng đợi thông điệp mà không cần bộ điều phối trung tâm.
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Mô hình chiếu góc nhìn theo từng tác tử trong môi trường sản xuất.
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Ghi chú về nguồn gốc dữ liệu và thẩm định thực tế trong triển khai sản xuất.
