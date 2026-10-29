# Kích hoạt Kỹ năng và Định tuyến (Skill Invocation and Routing)

> Kích hoạt là một quyết định về thẩm quyền, theo sau bởi một quyết định về độ liên quan. Một phần mô tả tốt giúp mô hình lựa chọn; một chính sách tốt quyết định xem lựa chọn đó có được phép hay không.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (stdlib)  
**Điều kiện tiên quyết:** Phase 13 · 24 (Khám phá Kỹ năng và Tiết lộ Lũy tiến - Skill Discovery and Progressive Disclosure)  
**Thời lượng:** ~105 phút  

## Mục tiêu học tập

- Phân biệt kích hoạt tường minh từ người dùng (explicit user invocation), kích hoạt ngầm định từ mô hình (implicit model invocation), kích hoạt từ ứng dụng (application invocation), và kích hoạt giữa các kỹ năng (skill-to-skill invocation).
- Mô hình hóa khả năng hiển thị đối với con người (human visibility) và tính đủ điều kiện đối với mô hình (model eligibility) thành các chiều chính sách độc lập.
- Viết các phần mô tả định tuyến với các điều kiện kích hoạt tích cực (positive triggers) và ranh giới suýt trúng (near-miss boundaries).
- Tách bạch các giai đoạn đủ điều kiện (eligibility), lựa chọn (selection), kích hoạt (activation), ràng buộc đối số (argument binding), và thực thi (execution) trong dấu vết theo dõi (traces) và kiểm thử.
- Điều hợp các trường kích hoạt đặc thù của môi trường chạy (runtime-specific invocation fields) mà không coi chúng là siêu dữ liệu đầu tệp (frontmatter) khả chuyển.

## Vấn đề

Bạn cài đặt một kỹ năng `database-migration`. Người dùng có thể chạy kỹ năng này theo tên, nhưng mô hình cũng nhìn thấy mô tả của nó và chọn nó khi có ai đó hỏi một câu hỏi tổng quát về cơ sở dữ liệu. Kỹ năng sau đó đề xuất một thay đổi lược đồ (schema change) cho một tác vụ vốn chỉ cần một câu giải thích.

Bạn thêm `user-invocable: false`, kỳ vọng sẽ chặn người dùng chạy thủ công. Trong một môi trường chạy (runtime) khác, trường này bị bỏ qua. Bạn thêm `disable-model-invocation: true`, kỳ vọng kỹ năng sẽ biến mất hoàn toàn. Trong môi trường hiểu được trường đó, người dùng vẫn có thể kích hoạt nó một cách tường minh.

Không có gì sai với tên các trường cả. Bản thân mô hình tư duy mới là thứ sai sót. "Người dùng có thể nhìn thấy", "mô hình có thể chọn", "ứng dụng có thể tải trước", và "các công cụ bên trong nó có thể thực thi" là các sự thật tách biệt hoàn toàn. Một biến boolean đơn lẻ mang tên `invocable` không thể biểu diễn được toàn bộ các trạng thái này.

Định tuyến còn có một chế độ lỗi thứ hai. Nếu các phần mô tả mơ hồ, nhiều kỹ năng cùng trở nên hợp lý. Nếu các phần mô tả bị nhồi nhét từ khóa, các tác vụ không liên quan sẽ vô tình kích hoạt chúng. Danh mục kỹ năng (catalog) là một giao diện xác suất (probabilistic interface): vừa đủ gọn gàng để đặt vừa ngữ cảnh, vừa đủ cụ thể để định tuyến chính xác.

## Khái niệm

### Năm kênh có thể khởi động vòng đời

| Tác nhân (Actor) | Dạng thức kích hoạt | Trường hợp sử dụng điển hình | Rủi ro chính |
|---|---|---|---|
| Người dùng (Human user) | Gọi tên kỹ năng trong giao diện (UI) hoặc câu lệnh (prompt) | Lựa chọn quy trình làm việc có chủ đích | Người dùng kỳ vọng tính sẵn sàng hoặc thẩm quyền mà môi trường máy chủ (host) không cấp phát |
| Mô hình hoặc tác tử tự trị (Model / autonomous agent) | Chọn một mục trong danh mục từ ngữ cảnh tác vụ | Quy trình chuyên gia tự động | Định tuyến dương tính giả (false-positive routing) |
| Ứng dụng (Application) | Kích hoạt hoặc tải trước một kỹ năng thông qua mã runtime | Quy trình sản phẩm cố định | Khớp nối ngầm (hidden coupling) vào một môi trường máy chủ duy nhất |
| Kỹ năng khác hoặc tác tử phụ (Skill / subagent) | Yêu cầu một kỹ năng chính xác dưới dạng phụ thuộc quy trình | Kết hợp (composition) | Vòng lặp đệ quy (cycles), thiếu phụ thuộc, hoặc rò rỉ ngữ cảnh (context bleed) |
| Bộ kiểm thử đánh giá (Evaluation harness) | Kích hoạt một kỹ năng chính xác trong kịch bản cố định | Đo lường có thể tái lập | Kiểm thử kỹ năng nhưng vô tình bỏ qua chính sách môi trường sản xuất đang được nghiên cứu |

Đặc tả Agent Skills khả chuyển (portable specification) định nghĩa cấu trúc gói. Nó không tiêu chuẩn hóa một giao diện lệnh gạch chéo (slash-command UI) toàn cục, cờ định tuyến ngầm định, API ứng dụng, hay vòng đời của tác tử phụ.

### Năm giai đoạn kích hoạt

````figure skill-invocation-stages
````

Hãy sử dụng các thuật ngữ này một cách chuẩn xác:

- **Đủ điều kiện (Eligible):** chính sách cho phép tác nhân này yêu cầu kỹ năng.
- **Được chọn (Selected):** người dùng đã nêu tên kỹ năng hoặc bộ định tuyến (router) đánh giá kỹ năng là phù hợp.
- **Được kích hoạt (Activated):** các chỉ dẫn của kỹ năng đã được nạp vào ngữ cảnh làm việc (working context).
- **Đang thực thi (Executing):** tác tử bắt đầu thực hiện các tác vụ mô hình hoặc công cụ theo các chỉ dẫn đó.
- **Hoàn thành (Completed):** đầu ra thỏa mãn một bước kiểm tra thành công độc lập.

Một bản ghi dấu vết (trace) chỉ ghi lại `skill_used=true` sẽ che giấu ranh giới nơi sự cố thực sự xảy ra.

### Kích hoạt bởi con người và mô hình tạo thành ma trận 2x2

| Người có thể kích hoạt | Mô hình có thể kích hoạt | Chế độ | Ví dụ phù hợp |
|:---:|:---:|---|---|
| Có | Có | Chia sẻ (Shared) | Giải thích mã nguồn, lập kế hoạch kiểm thử, duyệt tài liệu |
| Có | Không | Chỉ con người (Human-only) | Chuẩn bị phát hành, xuất dữ liệu thanh toán, kế hoạch dọn dẹp mang tính phá hủy |
| Không | Có | Chỉ mô hình (Model-only) | Hướng dẫn văn phong nội bộ, tài liệu tham chiếu chuyên ngành, quy trình hỗ trợ tự động |
| Không | Không | Vô hiệu hóa hoặc Chỉ ứng dụng (Disabled / application-only) | Triển khai theo giai đoạn (staged rollout), gói đã lỗi thời (deprecated), tải trước theo chương trình |

Ma trận này là một mô hình chính sách (policy model), không phải cú pháp YAML chuẩn hóa.

Một môi trường máy chủ hiện nay sử dụng `disable-model-invocation: true` cho hàng chỉ dành cho con người và `user-invocable: false` cho hàng chỉ dành cho mô hình. Mặc định là cả hai đều được phép. Một môi trường khác sử dụng `agents/openai.yaml` với `allow_implicit_invocation: false` để giữ kích hoạt tường minh đồng thời vô hiệu hóa việc lựa chọn ngầm định. Đây là các bộ điều hợp môi trường chạy (runtime adapters). Các môi trường chưa biết có thể bỏ qua chúng.

Chi tiết gây nhầm lẫn này rất quan trọng: `user-invocable: false` không có nghĩa là "mô hình không thể sử dụng kỹ năng này". Nó chỉ loại bỏ việc kích hoạt trực tiếp từ người dùng trong môi trường định nghĩa nó. Tương tự, `disable-model-invocation: true` không có nghĩa là "kỹ năng bị vô hiệu hóa hoàn toàn". Nó loại bỏ việc mô hình tự khởi tạo lựa chọn nhưng vẫn giữ quyền truy cập tường minh của người dùng.

### Kích hoạt tường minh ưu tiên định danh (Identity-first)

Một kích hoạt tường minh sẽ cung cấp trực tiếp danh tính của kỹ năng:

```text
/release-readiness v2.4.0
```

hoặc:

```text
release-readiness check v2.4.0 without publishing
```

Các giao diện Codex hiện tại tài liệu hóa `/skills` cho việc lựa chọn và tên kỹ năng thuần túy trong yêu cầu để kích hoạt tường minh. Claude Code tài liệu hóa cú pháp `/skill-name` và cơ chế mở rộng đối số đặc thù của môi trường. Cú pháp chính xác, khả năng hiển thị menu, quy tắc trích dẫn, và mở rộng biến thuộc thẩm quyền của môi trường máy chủ.

Một yêu cầu tường minh vẫn phải vượt qua khâu kiểm tra chính sách. Việc gọi tên một kỹ năng không được phép bỏ qua các quyền còn thiếu, các ràng buộc không gian làm việc, các cổng phê duyệt (approval gates), hay cơ chế cô lập môi trường chạy.

### Kích hoạt ngầm định ưu tiên mô tả (Description-first)

Đối với định tuyến ngầm định, ban đầu mô hình chỉ nhìn thấy siêu dữ liệu danh mục thay vì toàn bộ phần thân kỹ năng. Do đó, phần mô tả chính là giao diện định tuyến của kỹ năng.

Kém:

```yaml
description: Giúp xử lý các bản phát hành.
```

Quá rộng:

```yaml
description: Sử dụng cho các tác vụ release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, hoặc phần mềm.
```

Có ranh giới rõ ràng:

```yaml
description: Kiểm tra một bản ứng viên phát hành (release candidate) đã được chuẩn bị và tạo báo cáo mức độ sẵn sàng. Sử dụng khi người dùng hỏi liệu một phiên bản, thẻ (tag), gói (package), hoặc ảnh container (image) đã sẵn sàng để phát hành hay chưa; không sử dụng cho các lỗi bản dựng (build) thông thường hoặc phát triển tính năng mới.
```

Phiên bản có ranh giới rõ ràng bao gồm:

1. **Khả năng (Capability):** kiểm tra ứng viên đã chuẩn bị.
2. **Đầu ra (Output):** báo cáo mức độ sẵn sàng.
3. **Ranh giới tích cực (Positive boundary):** hỏi xem một tạo phẩm phát hành đã sẵn sàng chưa.
4. **Ranh giới tiêu cực (Negative boundary):** các bản dựng thông thường và việc phát triển tính năng nằm ngoài phạm vi.

Ranh giới tiêu cực rất hữu ích khi hai kỹ năng gần nhau có chung tập từ vựng. Chúng không thể thay thế cho các đánh giá suýt trúng (near-miss evals).

### Định tuyến là bài toán phân loại có tùy chọn từ chối (Abstain option)

Với một kỹ năng `s` và yêu cầu `x`, hãy hình dung một điểm số định tuyến:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

Việc tính điểm chính xác có thể là quyết định của một mô hình ngôn ngữ lớn (LLM) thay vì phép số học thuần túy. Nguyên lý kỹ thuật vẫn giữ nguyên: việc lựa chọn phải vượt qua một ngưỡng (threshold) nhất định và đánh bại kỹ năng cạnh tranh. Khi bằng chứng còn yếu, hãy từ chối đưa ra quyết định (abstain).

````figure skill-routing-abstention
````

Đối với các kỹ năng có tác động lớn, định tuyến ngầm định có thể không phù hợp ngay cả khi có mô tả mạnh mẽ. Hãy áp dụng chính sách chỉ dành cho con người (human-only policy) khi chi phí của một ca dương tính giả vượt quá sự tiện lợi của việc tự động lựa chọn.

### Xác định tính đủ điều kiện phải đi trước xếp hạng

Đừng chấm điểm mọi kỹ năng được phát hiện, chọn kỹ năng khớp mạnh nhất, rồi mới kiểm tra chính sách của kỹ năng đó sau. Một kỹ năng đứng đầu nhưng bị chặn sẽ ngăn cản một ứng viên đủ điều kiện có điểm thấp hơn được xem xét một cách sai lầm.

Hãy áp dụng thứ tự này cho định tuyến ngầm định:

1. Lọc các kỹ năng được phát hiện theo tác nhân yêu cầu và bộ điều hợp máy chủ đang hoạt động.
2. Chỉ chấm điểm các ứng viên đủ điều kiện.
3. Chọn ứng viên đủ điều kiện có điểm cao nhất nếu nó vượt qua ngưỡng điểm và các quy tắc mơ hồ (ambiguity rules).
4. Từ chối chọn (abstain) khi không có ứng viên nào đủ điều kiện hoặc không có điểm đủ điều kiện nào đủ mạnh.

Giả sử `incident-triage` đạt điểm `0.80` nhưng tiện ích mở rộng của môi trường máy chủ vô hiệu hóa kích hoạt từ mô hình. Kỹ năng `incident-review` đạt điểm `0.55` và cho phép mô hình kích hoạt. Bộ định tuyến nên đánh giá `incident-review` là ứng viên đủ điều kiện tốt nhất. Nó không được chọn `incident-triage`, từ chối nó, rồi dừng lại.

Thứ tự này cũng giúp các thay đổi chính sách không làm biến đổi ý nghĩa của điểm số mức độ liên quan. Tính đủ điều kiện định nghĩa tập hợp lựa chọn. Mức độ liên quan xếp hạng tập hợp đó.

### Đánh giá định tuyến cần các ca suýt trúng (Near misses)

Các trường hợp tích cực chứng minh độ bao phủ (recall):

```json
{"prompt":"Phiên bản 2.4.0 đã sẵn sàng để phát hành chưa?","expected":"release-readiness"}
```

Các trường hợp tiêu cực rõ ràng chứng minh độ chính xác cơ bản (precision):

```json
{"prompt":"Hãy giải thích kỹ thuật nhúng vị trí quay (rotary position embeddings).","expected":null}
```

Các ca suýt trúng (near misses) phơi bày chất lượng ranh giới:

```json
{"prompt":"Tại sao bản dựng gói phần mềm hôm nay bị thất bại?","expected":"build-diagnostics"}
```

Ca suýt trúng này chia sẻ các từ khóa `package` và `build` với kỹ năng phát hành nhưng lại thuộc về một quy trình khác. Một tập dữ liệu đánh giá định tuyến chỉ gồm các ca tích cực hiển nhiên và các ca tiêu cực không liên quan sẽ phóng đại chất lượng thực tế.

### Đối số có ba biểu diễn

Một đối số kích hoạt vượt qua nhiều ranh giới:

````figure skill-argument-boundaries
````

Tại mỗi ranh giới, hãy bảo toàn ý định mà không coi văn bản là mã thực thi:

- Bộ phân tích cú pháp của môi trường máy chủ quyết định cú pháp lệnh và trích dẫn.
- Kỹ năng nhận văn bản hoặc biến đã ràng buộc theo quy tắc của máy chủ.
- Các chỉ dẫn kiểm tra tính hợp lệ của các giá trị bắt buộc và giá trị mặc định.
- Một lời gọi công cụ chuyển đổi các giá trị sang lược đồ có kiểu (typed schema) và xác thực lại chúng.

Đừng nội suy đối số thô trực tiếp vào các lệnh shell. Hãy ưu tiên một tập lệnh được gọi với vectơ đối số (argument vector) hoặc một công cụ MCP có kiểu dữ liệu tường minh.

### Kích hoạt từ ứng dụng là điều phối tường minh (Explicit orchestration)

Một sản phẩm có thể kích hoạt một kỹ năng vì quy trình công việc của nó đã biết rõ loại tác vụ. Ví dụ, một dịch vụ đánh giá pull-request có thể tải trước kỹ năng `pull-request-risk-review` sau khi người dùng nhấn nút "Review".

Điều này loại bỏ sự không chắc chắn của việc định tuyến nhưng tạo ra sự phụ thuộc vào API của môi trường chạy. Hãy giữ bộ điều hợp đó bên ngoài phần thân khả chuyển:

````figure skill-host-adapter
````

Kỹ năng vẫn phải giữ được tính dễ hiểu khi được mở bởi một máy khách (client) tuân thủ đặc tả khác.

### Kích hoạt giữa các kỹ năng là một cạnh dạng công cụ

Giả sử `release-readiness` yêu cầu `security-change-review` khi các tệp phụ thuộc bị thay đổi.

Bên gọi nên cung cấp:

- định danh của kỹ năng đích;
- một tác vụ có giới hạn và các đường dẫn tạo phẩm;
- hợp đồng phản hồi kỳ vọng;
- lý do kích hoạt;
- phương án dự phòng (fallback) nếu kỹ năng đích không khả dụng;
- quy tắc về độ sâu tối đa hoặc phòng chống vòng lặp.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

Kỹ năng thứ hai không được dán một cách mù quáng vào kỹ năng thứ nhất. Môi trường máy chủ quyết định cách kích hoạt nó và xem liệu nó có chia sẻ ngữ cảnh, chạy trong một tiến trình rẽ nhánh (fork), hay trả về thông qua kết quả công cụ hay không.

### Vòng đời ngữ cảnh phụ thuộc vào môi trường máy chủ

Sau khi kích hoạt, phần thân kỹ năng có thể được giữ lại trong cuộc hội thoại, bị tóm tắt trong quá trình cô đọng (compaction), hoặc chạy trong một ngữ cảnh được ủy quyền (delegated context). Các quyền cấp phát cho công cụ có thể chỉ kéo dài trong một lượt (turn) trong khi các chỉ dẫn tồn tại lâu hơn. Một tác tử phụ có thể nhận kỹ năng mà không có toàn bộ lịch sử của tác tử cha.

Đừng viết một kỹ năng phụ thuộc vào một giả định ngầm định về thời gian tồn tại (lifetime assumption). Hãy ghi các đầu ra bền vững vào tệp hoặc trạng thái có kiểu, đảm bảo khả năng tái nhập an toàn (re-entry safe), và nêu rõ những gì phải được tải lại sau khi bị gián đoạn.

```markdown
Khi tiếp tục lại (resume), đọc `artifacts/release-readiness.json` nếu tệp tồn tại.
Xác thực lại commit ứng viên trước khi tiếp tục.
Không lặp lại một thao tác ghi ra bên ngoài nếu khóa lũy kế (idempotency key) của nó đã được ghi nhận.
```

## Xây dựng (Build It)

`code/main.py` cài đặt chính sách và định tuyến dưới dạng các bộ điều hợp riêng biệt.

Mô hình bao gồm:

- `Actor` đại diện cho các bên gọi là con người, mô hình, tác tử tự trị, ứng dụng, kỹ năng, và bộ kiểm thử;
- `SkillMetadata` đại diện cho định danh định tuyến;
- `InvocationPolicy` biểu diễn ma trận con người / mô hình;
- `InvocationRequest` và `InvocationDecision` cho các đầu vào và kết quả có thể truy vết;
- `CorePolicyAdapter` cho hành vi khả chuyển không có tiện ích mở rộng máy chủ;
- `ExtensionPolicyAdapter` cho các trường runtime đã được nhận diện;
- `build_invocation_matrix(policy)` cho góc nhìn ma trận 2x2;
- `route_request(skills, request, adapter)` lọc tính đủ điều kiện trước khi xếp hạng mức độ liên quan, lựa chọn và từ chối.

Chạy mã nguồn:

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Bản minh họa in ra một ma trận và các quyết định cho các kênh: con người tường minh, mô hình ngầm định, tác tử tự trị, ứng dụng, kết hợp kỹ năng, và bộ kiểm thử. Kết quả từ bộ điều hợp mở rộng cho thấy trường hợp một mục khớp từ vựng cao nhất bị chặn đã được loại bỏ trước khi một phương án thay thế đủ điều kiện được xếp hạng. Mã cũng bao gồm danh sách trắng các tên chính xác. Không cần API mô hình ngoài. Bộ định tuyến tất định tồn tại để giúp các ranh giới chính sách có thể kiểm tra được, chứ không nhằm khẳng định rằng việc khớp từ vựng tái tạo hoàn hảo định tuyến mô hình trong môi trường sản xuất.

### Tại sao bộ điều hợp lõi và mở rộng lại tách biệt

Nếu một bộ phân tích cú pháp gán ý nghĩa cho mọi trường frontmatter quan sát được, nó sẽ âm thầm biến các quy ước của môi trường chạy thành một tiêu chuẩn giả mạo. Việc tách biệt các bộ điều hợp buộc bên gọi phải nêu rõ ngữ nghĩa của môi trường máy chủ nào đang hoạt động.

`CorePolicyAdapter` chỉ sử dụng chính sách do ứng dụng cung cấp. `ExtensionPolicyAdapter` nhận diện một tập hợp rõ ràng các trường của môi trường máy chủ và ghi lại trường nào đã thay đổi quyết định.

## Sử dụng (Use It)

Hãy viết một hợp đồng kích hoạt trước khi phát hành một kỹ năng:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

Hợp đồng này là tài liệu thiết kế dành cho các bộ điều hợp và kiểm thử. Nó không phải là frontmatter `SKILL.md` khả chuyển trừ khi có một tiêu chuẩn chính thức áp dụng nó.

## Đóng gói và phát hành (Ship It)

Bài học này tạo ra gói tạo phẩm `skill-invocation-router`. Gói bao gồm tài liệu tham chiếu mô hình kích hoạt, chính sách máy chủ mẫu, và một giao diện dòng lệnh (CLI) không thực thi để đánh giá một yêu cầu từ con người, mô hình, tác tử tự trị, ứng dụng, kết hợp kỹ năng hoặc bộ kiểm thử và trả về quyết định JSON với kênh, bộ điều hợp, điểm số và lý do.

CLI một yêu cầu đóng vai trò là một đầu dò chính sách (policy probe), không phải một bộ đánh giá kích hoạt hoàn chỉnh. Hãy sử dụng thiết kế các ca tích cực và ca suýt trúng có gắn nhãn trong Bài học 27 để tính toán ma trận nhầm lẫn (confusion counts), độ chính xác (precision), độ bao phủ (recall), và độ ổn định qua các lần chạy lặp lại.

## Bài tập

1. Tạo tất cả bốn hàng của ma trận con người / mô hình và viết một trường hợp sử dụng hợp lệ cho mỗi hàng.
2. Thêm cơ chế kích hoạt chỉ dành cho ứng dụng vào `CorePolicyAdapter`. Chứng minh rằng các bên gọi là con người và mô hình vẫn bị từ chối.
3. Viết mười ca suýt trúng cho một kỹ năng triển khai phần mềm (deployment skill). Mỗi câu lệnh phải chia sẻ từ vựng với kỹ năng nhưng thuộc về một quy trình làm việc khác biệt.
4. Thêm biên độ mơ hồ (ambiguity margin) giữa hai điểm số định tuyến cao nhất. Trả về `ask` khi biên độ quá nhỏ.
5. Thêm giới hạn độ sâu kết hợp tối đa cho các yêu cầu giữa các kỹ năng và phát hiện một chu trình phụ thuộc giữa hai kỹ năng.
6. Chạy cùng một tập dữ liệu có gắn nhãn qua bộ điều hợp lõi và bộ điều hợp mở rộng. Giải thích từng quyết định bị thay đổi.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa kỹ thuật thực sự |
|---|---|---|
| Kích hoạt tường minh (Explicit invocation) | "Lệnh gạch chéo" (Slash command) | Một tác nhân cung cấp trực tiếp danh tính kỹ năng, tuân theo quy định của chính sách |
| Kích hoạt ngầm định (Implicit invocation) | "Mô hình tự chọn" | Một bộ định tuyến chọn từ siêu dữ liệu danh mục đủ điều kiện dựa trên ngữ cảnh tác vụ |
| Người dùng có thể kích hoạt (User-invocable) | "Con người có thể dùng được" | Một thuộc tính hiển thị menu hoặc kích hoạt trực tiếp đặc thù của host, không phải trường cốt lõi |
| Mô hình có thể kích hoạt (Model-invocable) | "Tác tử có thể dùng được" | Tính đủ điều kiện cho việc lựa chọn ngầm định từ mô hình dưới chính sách của host |
| Bộ điều hợp kích hoạt (Invocation adapter) | "Bộ phân tích cú pháp frontmatter" | Mã ánh xạ các trường và API của host vào một mô hình chính sách đã khai báo |
| Ca suýt trúng (Near miss) | "Mẫu tiêu cực khó" (Hard negative) | Một yêu cầu không kích hoạt nhưng có bề ngoài rất giống với các đầu vào dự kiến của kỹ năng |
| Từ chối lựa chọn (Abstention) | "Không có kỹ năng nào được chọn" | Kết quả định tuyến có chủ đích khi thiếu bằng chứng hoặc bằng chứng bị mơ hồ |

## Đọc thêm

- [Tối ưu hóa mô tả kỹ năng (Optimizing skill descriptions)](https://agentskills.io/skill-creation/optimizing-descriptions) về các điều kiện kích hoạt tích cực, tính cụ thể và đánh giá.
- [Đánh giá kỹ năng (Evaluating skills)](https://agentskills.io/skill-creation/evaluating-skills) về thiết kế đánh giá kích hoạt và đầu ra.
- [OpenAI: Xây dựng kỹ năng (Build skills)](https://learn.chatgpt.com/docs/build-skills) về các cơ chế kiểm soát kích hoạt tường minh và ngầm định hiện tại của Codex.
- [Kỹ năng Claude Code (Claude Code skills)](https://code.claude.com/docs/en/skills) về các trường `user-invocable`, `disable-model-invocation`, đối số, và ngữ cảnh ủy quyền của một môi trường máy chủ cụ thể.
