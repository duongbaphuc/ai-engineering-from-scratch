# Di sản của FIPA-ACL và Lý thuyết Hành vi Ngôn ngữ (Heritage of FIPA-ACL and Speech Acts)

> Trước khi có MCP, trước khi có A2A, thế giới đã từng có FIPA-ACL. Vào năm 2000, Tổ chức vì các Tác tử Vật lý Thông minh (FIPA - Foundation for Intelligent Physical Agents, sau này trở thành ủy ban tiêu chuẩn của Hiệp hội Máy tính IEEE vào năm 2005) đã phê chuẩn một ngôn ngữ giao tiếp tác tử (agent communication language) với 20 phát ngôn hành vi (performatives), hai ngôn ngữ nội dung và một tập hợp các giao thức tương tác — bao gồm mạng lưới đấu thầu (contract net), đăng ký/thông báo (subscribe/notify), yêu cầu theo điều kiện (request-when). Chuẩn này dần vắng bóng khỏi ngành công nghiệp do chi phí xây dựng bản thể học (ontology overhead) quá nặng nề đối với web, nhưng làn sóng hồi sinh hệ đa tác tử (multi-agent systems) bằng mô hình ngôn ngữ lớn (LLM) hiện nay đang âm thầm tái hiện lại chính những ý tưởng đó mà không cần ngữ nghĩa hình thức: các hợp đồng JSON thế chỗ cho các performative, và ngôn ngữ tự nhiên thế chỗ cho bản thể học. Bài học này đi sâu vào FIPA-ACL một cách nghiêm túc để bạn thấy rõ những quyết định thiết kế giao thức năm 2026 đâu là sự tái phát minh, đâu là điểm mới thực sự, và làn sóng hiện tại sẽ vấp phải những vấn đề mà thập niên 2000 vốn dĩ đã tìm ra lời giải từ lâu ở đâu.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 phút

## Vấn đề

Bức tranh giao thức tác tử năm 2026 vô cùng sôi động: MCP dành cho công cụ (tools), A2A dành cho tác tử với tác tử (agent-to-agent), ACP dành cho kiểm toán doanh nghiệp (enterprise audit), ANP dành cho định danh phi tập trung (decentralized trust), NLIP dành cho nội dung ngôn ngữ tự nhiên, cộng thêm CA-MCP và hàng chục đề xuất nghiên cứu khác. Mỗi bản đặc tả đều tự nhận mình là nền tảng cốt lõi mang tính đột phá.

Nhìn nhận một cách thẳng thắn, phần lớn trong số đó thực chất chỉ đang tái khám phá lại đúng cây quyết định đã tồn tại từ 20 năm trước. Lý thuyết hành vi ngôn ngữ (speech-act theory) từ Austin (1962) và Searle (1969) đã mang lại tiên đề "lời phát ngôn chính là hành động" (utterances are actions). KQML (1993) đã biến tiên đề đó thành một giao thức truyền tải trên đường truyền (wire protocol). FIPA-ACL (được phê chuẩn năm 2000) đã tạo nên tiêu chuẩn tham chiếu hoàn chỉnh: 20 phát ngôn hành vi, các ngôn ngữ nội dung SL0/SL1, các giao thức tương tác cho mạng lưới đấu thầu (contract net) và mô hình đăng ký - thông báo (subscribe - notify). JADE và JACK từng là các nền tảng tham chiếu kinh điển viết bằng Java. Nỗ lực này dần thoái trào vào khoảng năm 2010 vì chi phí bản thể học quá đồ sộ trong khi công nghệ Web thông thường lại giành chiến thắng áp đảo.

Khi bạn quan sát phương thức `tools/call` của MCP, vòng đời tác vụ (task lifecycle) của A2A, hay kho ngữ cảnh chia sẻ của CA-MCP, bạn thực chất đang nhìn vào phiên bản mềm hóa, dùng JSON của các quyết định FIPA năm xưa. Hiểu rõ di sản này mang lại cho bạn hai lợi thế: biết rõ những "phát kiến" mới nào thực chất chỉ là phát minh lại cái cũ, và dự đoán trước những kịch bản thất bại kinh điển nào mà các chuẩn đặc tả mới sẽ sớm tái vấp phải.

## Khái niệm

### Lý thuyết Hành vi Ngôn ngữ, tóm tắt trong một đoạn

Austin nhận thấy rằng có những câu nói không nhằm mục đích mô tả thế giới — mà chúng làm thay đổi thế giới. "Tôi hứa." "Tôi yêu cầu." "Tôi tuyên bố." Ông gọi đó là các phát ngôn hành vi (performative utterances). Searle sau đó đã hệ thống hóa chúng thành 5 nhóm: xác quyết (assertive), chỉ thị (directive), cam kết (commissive), biểu cảm (expressive), tuyên bố (declarative). KQML (Finin et al., 1993) đã đưa lý thuyết này vào vận hành thực tế cho các tác tử phần mềm: một thông điệp bao gồm một phát ngôn hành vi (hành động định thực hiện) cộng với nội dung (nội dung mà hành động đó hướng tới). FIPA-ACL đã khắc phục các lỗ hổng của KQML và chuẩn hóa thành 20 phát ngôn hành vi cốt lõi.

### 20 phát ngôn hành vi FIPA (Danh sách tiêu biểu)

| Phát ngôn hành vi (Performative) | Ý định truyền đạt (Intent) |
|---|---|
| `inform` | "Tôi thông báo cho bạn biết rằng mệnh đề P là đúng" |
| `request` | "Tôi yêu cầu bạn thực hiện hành động X" |
| `query-if` | "Liệu mệnh đề P có đúng không?" |
| `query-ref` | "Giá trị của X là bao nhiêu?" |
| `propose` | "Tôi đề xuất chúng ta cùng thực hiện X" |
| `accept-proposal` | "Tôi chấp thuận đề xuất này" |
| `reject-proposal` | "Tôi từ chối đề xuất này" |
| `agree` | "Tôi đồng ý sẽ thực hiện X" |
| `refuse` | "Tôi từ chối thực hiện X" |
| `confirm` | "Tôi xác nhận P là sự thật" |
| `disconfirm` | "Tôi bác bỏ P" |
| `not-understood` | "Thông điệp của bạn gửi sang tôi không thể phân tích cú pháp" |
| `cfp` | "Kêu gọi nộp đề xuất giải pháp cho tác vụ X (Call for proposals)" |
| `subscribe` | "Hãy thông báo cho tôi mỗi khi X có sự thay đổi" |
| `cancel` | "Hủy bỏ tác vụ X đang diễn ra" |
| `failure` | "Tôi đã thử thực hiện X nhưng thất bại" |

Danh sách đầy đủ được quy định trong tài liệu `fipa00037.pdf` (Cấu trúc thông điệp FIPA ACL). Vấn đề cốt lõi không phải là học vẹt danh sách này — mà là nhận thức được rằng mỗi thành phần ở đây đều tương ứng trực tiếp với một nguyên hàm (primitive) mà các giao thức LLM hiện đại rồi sẽ phải bổ sung lại.

### Cấu trúc thông điệp FIPA-ACL chuẩn mực

```text
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Bảy trường dữ liệu tạo nên lớp vỏ giao thức (protocol envelope); đúng một trường duy nhất (`content`) mang tải trọng dữ liệu (payload). Tất cả các trường còn lại chính là những gì bạn sẽ phải tự mình tái phát minh mỗi khi tìm cách bổ sung cơ chế thử lại (retries), phân luồng hội thoại (threading) và bản thể học vào một giao thức JSON đơn sơ.

### Hai nền tảng tiền bối (The Two Legacy Platforms)

**JADE** (Java Agent DEvelopment framework, giai đoạn 1999–2020) từng là môi trường thực thi chuẩn FIPA phổ biến nhất. Các tác tử kế thừa một lớp cơ sở, trao đổi các thông điệp ACL, chạy bên trong các thùng chứa (containers) và phối hợp với nhau thông qua các "hành vi" (behaviors). Thư viện giao thức tương tác của nó đi kèm sẵn mẫu contract-net, subscribe-notify, request-when và propose-accept.

**JACK** (Agent Oriented Software, thương mại) nhấn mạnh vào cơ chế lập luận BDI (Niềm tin - Mong muốn - Ý định / Belief-Desire-Intention) dựa trên các thông điệp FIPA. Mang tính hình thức cao hơn nhưng ít được đón nhận rộng rãi hơn.

Cả hai đều suy thoái khi ngăn xếp công nghệ Web (web stack) chiếm trọn mọi ca sử dụng đa tác tử. Giờ đây, MCP và A2A đóng vai trò như những "thùng chứa" thực thi của năm 2026.

### Vì sao FIPA dần mai một

- **Chi phí bản thể học quá lớn (Ontology overhead).** FIPA yêu cầu phải có một bản thể học dùng chung để phân tích cú pháp trường `content`. Đạt được thỏa thuận về các bản thể học là một quá trình tiêu chuẩn hóa kéo dài nhiều năm trời. Trong khi đó, giới phát triển web chỉ đơn giản sử dụng HTTP + JSON là xong.
- **Ngữ nghĩa hình thức không ai thực sự dùng.** Ngôn ngữ ngữ nghĩa SL (Semantic Language) đưa ra các điều kiện chân lý chặt chẽ, nhưng hầu hết các hệ thống thực tế chỉ truyền nội dung tự do và bỏ qua tính hình thức này.
- **Bó buộc công cụ (Tooling lock-in).** JADE chỉ dùng riêng cho Java; JACK là phần mềm thương mại đóng kín. Các đội ngũ phát triển đa ngôn ngữ (polyglot teams) đều tìm cách đi vòng qua cả hai.
- **Internet đã thống trị toàn bộ ngăn xếp.** Kiến trúc REST, sau đó là JSON-RPC, rồi gRPC đã thay thế hoàn toàn tầng vận chuyển của ACL.

### Sự hồi sinh nhờ LLM thực chất là FIPA rút gọn (FIPA-lite)

Hãy đặt một phát ngôn `request` của FIPA cạnh lệnh `tools/call` của MCP:

```text
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Cùng một lớp vỏ bao bọc, chỉ khác biệt về mặt cú pháp. Cả hai đều mang: ai gửi, gửi cho ai, ý định là gì, tải trọng dữ liệu nào, mã định danh tương quan (correlation id) là bao nhiêu. Không bên nào là một cuộc cách mạng tuyệt đối so với bên kia — chúng chỉ là các điểm đánh đổi khác nhau trên cùng một bản thiết kế.

Khảo sát năm 2025 của Liu et al. ("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279) đã chỉ rõ nguồn gốc huyết thống này: MCP tương ứng với các hành vi ngôn ngữ sử dụng công cụ (tool-use speech acts), A2A tương ứng với hành vi ngôn ngữ giữa tác tử ngang hàng (agent-peer speech acts), ACP tương ứng với hành vi ngôn ngữ ghi nhật ký kiểm toán (audit-trail speech acts), ANP tương ứng với các phần mở rộng định danh phi tập trung (decentralized-identity extensions). Các đặc tả mới đều là hậu duệ trực tiếp của ACL với cú pháp JSON và ngữ nghĩa nới lỏng hơn.

### Bức tranh đánh đổi sòng phẳng

**Những gì FIPA cung cấp nhưng các đặc tả hiện đại đã lược bỏ:**

- Ngữ nghĩa hình thức (Formal semantics) — bạn có thể chứng minh về mặt toán học rằng `inform` hàm ý người gửi thực sự tin rằng nội dung là đúng.
- Một danh mục chuẩn mực các phát ngôn hành vi — bạn không cần phải tranh cãi lại xem "liệu hệ thống có cần lệnh `cancel` hay không?".
- Hàng thập kỷ tích lũy các mẫu giao thức tương tác — contract-net, subscribe-notify, propose-accept — với các tính chất đúng đắn đã được kiểm chứng.

**Những gì các đặc tả hiện đại cung cấp mà FIPA trước đây không có:**

- Tải trọng định dạng JSON thuần túy (JSON-native payloads) tương thích tức thì với mọi công cụ hiện đại.
- Nội dung bằng ngôn ngữ tự nhiên mà LLM có thể diễn giải trực tiếp mà không cần một bản thể học viết tay bằng mã nguồn.
- Tầng truyền tải dựa trên ngăn xếp Web sẵn có (HTTP, SSE, WebSocket).
- Khả năng khám phá năng lực tác tử theo thời gian thực thông qua `server/discover` của MCP và Thẻ Tác tử (Agent Cards) của A2A.

Đánh đổi tính chặt chẽ của ngữ nghĩa để lấy sự dễ dàng khi triển khai thực tế. Đó chính là bản chất của sự chuyển dịch.

### Các giao thức tương tác đáng kế thừa

FIPA từng chuẩn hóa khoảng 15 giao thức tương tác. Có ba giao thức cốt lõi cực kỳ đáng giá để kế thừa vào các hệ thống đa tác tử LLM hiện đại:

1. **Giao thức Mạng lưới Đấu thầu (Contract Net Protocol - CNP).** Tác tử quản lý phát đi lời kêu gọi `cfp` (call for proposals); các tác tử đấu thầu phản hồi lại bằng `propose`; tác tử quản lý quyết định chấp thuận (`accept`) hoặc từ chối (`reject`). Đây là mẫu hình thị trường tác vụ chuẩn mực (sẽ học kỹ trong Phase 16 · 16 Thương lượng đàm phán).
2. **Đăng ký / Thông báo (Subscribe/Notify).** Tác tử đăng ký gửi thông điệp `subscribe`; tác tử phát hành gửi thông điệp `inform` mỗi khi chủ đề quan tâm có sự biến động. Đây là nền tảng của mọi bus sự kiện (event-bus) năm 2026.
3. **Yêu cầu theo điều kiện (Request-When).** "Hãy thực hiện X khi điều kiện Y được thỏa mãn." Hành động bị trì hoãn đi kèm tiền điều kiện. Bản sao hiện đại của nó vào năm 2026 chính là các tác vụ trì hoãn (deferred tasks) trong các engine quy trình bền vững (durable workflow engines, xem Phase 16 · 22 Mở rộng quy mô sản xuất).

Cả ba giao thức này đều ánh xạ trực tiếp sang các hàng đợi tin nhắn hiện đại, mô hình HTTP + polling, hoặc luồng dữ liệu SSE streaming.

### Điều gì sẽ gãy đổ khi loại bỏ bản thể học

Khi không có một bản thể học dùng chung, các tác tử buộc phải tự suy diễn ý nghĩa từ nội dung văn bản tự nhiên. Kịch bản lỗi kinh điển được ghi nhận vào năm 2026 chính là **trôi dạt ngữ nghĩa (semantic drift)**: hai tác tử cùng sử dụng từ `"customer"` (khách hàng) nhưng ám chỉ hai khái niệm tinh tế khác nhau, tác tử nhận hành động dựa trên cách hiểu sai lệch, và không hề có một bộ xác thực lược đồ (schema validator) nào phát hiện ra điều đó. Yêu cầu bản thể học chặt chẽ của FIPA trước đây sẽ loại bỏ thông điệp sai lệch này ngay tại thời điểm phân tích cú pháp.

Các giải pháp giảm thiểu mà không cần quay lại bản thể học cồng kềnh:

- Sử dụng JSON Schema cho trường `content` — loại bỏ các lỗi cấu trúc ngay trên đường truyền.
- Định kiểu cho các tạo tác (Typed artifacts trong A2A) — ngăn chặn sai lệch phương thức dữ liệu (modality).
- Chỉ định rõ ràng phát ngôn hành vi (performative) trong lớp vỏ bọc — đảm bảo ý định luôn tường minh ngay cả khi nội dung là ngôn ngữ tự nhiên tự do.

### Bản đồ đối chiếu các đặc tả 2026 với di sản hành vi ngôn ngữ

| Đặc tả hiện đại | Khái niệm FIPA tương đương | Yếu tố kế thừa | Yếu tố lược bỏ |
|---|---|---|---|
| MCP `tools/call` | `request` | Ý định tường minh, ID tương quan | Ngữ nghĩa hình thức, bản thể học |
| MCP `resources/read` | `query-ref` | Ý định tường minh, ID tương quan | Ngữ nghĩa hình thức |
| Vòng đời tác vụ A2A | contract-net + request-when | Vòng đời bất đồng bộ, chuyển trạng thái | Đảm bảo tính đầy đủ về mặt hình thức |
| Luồng sự kiện A2A (streaming) | subscribe/notify | Đẩy dữ liệu bất đồng bộ (async push) | Cơ chế đăng ký bằng vị từ có kiểu |
| Ngữ cảnh chia sẻ CA-MCP | Bảng đen (Blackboard - Hayes-Roth 1985) | Bộ nhớ dùng chung đa nguồn ghi | Mô hình nhất quán logic |
| NLIP | Nội dung ngôn ngữ tự nhiên | Khả năng tương thích tự nhiên với LLM | Lược đồ cấu trúc (schema) |

Đọc bảng này từ trên xuống dưới, quy luật hiện rõ: giữ lại các nguyên hàm cấu trúc, vứt bỏ tính hình thức hàn lâm, và dựa vào LLM để khỏa lấp những điểm mơ hồ.

````figure
sw-contract-net
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa một bộ dịch FIPA-ACL thuần túy bằng thư viện chuẩn Python (stdlib). Nó mã hóa và giải mã lớp vỏ bọc ACL chuẩn tắc, đồng thời minh họa cách mọi cấu trúc thông điệp MCP / A2A đều quy tụ về đúng 7 trường dữ liệu kinh điển. Bản demo bao gồm:

- Mã hóa 5 thông điệp theo kiểu MCP và A2A sang FIPA-ACL.
- Giải mã FIPA-ACL ngược lại định dạng tương đương hiện đại.
- Chạy một kịch bản đàm phán Contract Net giả lập giữa một tác tử quản lý và ba tác tử dự thầu sử dụng `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Chạy chương trình:

```bash
python3 code/main.py
```

Kết quả in ra là một bản truy vết so sánh song song hiển thị từng thông điệp hiện đại ở cả dạng JSON của năm 2026 và dạng FIPA-ACL cổ điển, sau đó là một vòng khép kín của lượt đấu thầu contract-net. Cùng một nguyên hàm giao thức vẫn tồn tại trọn vẹn qua chuyến đi hai chiều; thứ duy nhất thay đổi chỉ là cú pháp bề mặt.

## Ứng dụng thực tế (Use It)

Tệp `outputs/skill-fipa-mapper.md` là một kỹ năng phân tích cho phép đọc bất kỳ bản đặc tả giao thức tác tử nào và tạo ra bản đồ ánh xạ sang FIPA-ACL tương ứng. Hãy sử dụng nó trước khi quyết định áp dụng một giao thức mới để trả lời câu hỏi cốt lõi: "Đây thực sự là một đột phá mới, hay nó chỉ là phát ngôn `inform` khoác lên mình chiếc áo JSON?"

## Đưa vào sản phẩm (Ship It)

Đừng cố gắng hồi sinh nguyên trạng FIPA-ACL. Thay vào đó, hãy mang danh sách kiểm tra (checklist) kinh điển của nó trở lại sản phẩm của bạn:

- Nguyên hàm ý định (phát ngôn hành vi) của từng thông điệp là gì?
- Có tồn tại một mã định danh tương quan (correlation id) để phục vụ việc ghép đôi yêu cầu - phản hồi và xử lý hủy bỏ tác vụ hay không?
- Ngôn ngữ nội dung có được khai báo tường minh không (JSON-RPC, văn bản thuần, hay tạo tác có kiểu cấu trúc)?
- Các giao thức tương tác có được coi là công dân hạng nhất (first-class citizens) không, hay bạn đang phải tự viết lại mô hình contract-net từ con số không?
- Điều gì sẽ xảy ra khi hai tác tử bất đồng về mặt ý nghĩa nội dung (hiện tượng trôi dạt ngữ nghĩa - semantic drift)?

Hãy lập tài liệu trả lời 5 câu hỏi này cho bất kỳ giao thức mới nào trước khi bạn chính thức triển khai nó lên môi trường sản xuất.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Quan sát luồng mã hóa hai chiều. Xác định xem phát ngôn hành vi FIPA nào tương ứng trực tiếp với `tools/call`, `resources/read` và thao tác khởi tạo tác vụ trong A2A.
2. Mở rộng kịch bản contract-net với phát ngôn hành vi `cancel` cho phép tác tử quản lý hủy bỏ tác vụ ngay giữa chừng khi các bên đang nộp hồ sơ đấu thầu. Tình huống lỗi nào mà lệnh `cancel` giải quyết được trong khi cơ chế thử lại (retries) đơn thuần hoàn toàn bó tay?
3. Đọc đặc tả Cấu trúc thông điệp FIPA ACL (http://www.fipa.org/specs/fipa00037/) các mục 4.1–4.3. Chọn ra một phát ngôn hành vi chưa được đề cập trong bài học này và mô tả cấu trúc JSON-RPC tương đương của nó trong thế giới hiện đại.
4. Đọc bài báo của Liu et al., arXiv:2505.02279. Đối với từng giao thức MCP, A2A, ACP, ANP, hãy liệt kê các họ phát ngôn hành vi FIPA mà chúng giữ lại và những họ mà chúng đã lược bỏ.
5. Thiết kế một JSON Schema tối giản cho trường `content` của một phát ngôn hành vi `request` trong hệ thống của chính bạn. Lược đồ đó mang lại lợi ích gì so với việc dùng hoàn toàn ngôn ngữ tự nhiên, và cái giá phải trả về độ phức tạp là bao nhiêu?

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| Hành vi ngôn ngữ (Speech act) | "Một câu nói tạo ra hành động" | Lý thuyết của Austin/Searle: lời nói đóng vai trò như các hành động. Là nền tảng lý thuyết sinh ra ACL. |
| FIPA | "Cái chuẩn XML cổ lỗ sĩ thời xưa" | Foundation for Intelligent Physical Agents. Chuẩn hóa ACL vào năm 2000; trở thành ủy ban tiêu chuẩn của IEEE Computer Society vào năm 2005. |
| ACL | "Ngôn ngữ giao tiếp tác tử" | Định dạng vỏ bọc thông điệp của FIPA: bao gồm phát ngôn hành vi + nội dung + siêu dữ liệu. |
| Phát ngôn hành vi (Performative) | "Động từ hành động" | Lớp ý định của thông điệp: `inform`, `request`, `propose`, `cfp`, v.v. |
| KQML | "Tiền bối của FIPA" | Knowledge Query and Manipulation Language (1993). Đơn giản hơn nhưng phạm vi hẹp hơn. |
| Bản thể học (Ontology) | "Từ vựng dùng chung" | Định nghĩa hình thức về các khái niệm mà ngôn ngữ nội dung đang đề cập tới. |
| SL0 / SL1 | "Các ngôn ngữ nội dung của FIPA" | Ngôn ngữ ngữ nghĩa (Semantic Language) cấp độ 0 và 1 — họ ngôn ngữ nội dung hình thức của FIPA. |
| Mạng lưới đấu thầu (Contract Net) | "Thị trường tác vụ" | Tác tử quản lý phát hành cfp; các bên dự thầu gửi propose; quản lý chấp thuận. Giao thức tương tác kinh điển nhất. |
| Giao thức tương tác (Interaction protocol) | "Mẫu trao đổi thông điệp" | Một chuỗi các phát ngôn hành vi tuần tự với tính đúng đắn đã được chứng minh: request-when, subscribe-notify, v.v. |

## Tài liệu đọc thêm (Further Reading)

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — Khảo sát kinh điển năm 2025 kết nối các đặc tả hiện đại với di sản FIPA.
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — Bản đặc tả định dạng vỏ bọc thông điệp được phê chuẩn năm 2000.
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — Toàn bộ danh mục phát ngôn hành vi chuẩn tắc.
- [MCP specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — Chuẩn sử dụng công cụ không trạng thái hiện đại tương đương với `request`/`query-ref`.
- [A2A specification](https://a2a-protocol.org/latest/specification/) — Chuẩn giao tiếp ngang hàng giữa các tác tử hiện đại tương đương với contract-net và subscribe-notify.
