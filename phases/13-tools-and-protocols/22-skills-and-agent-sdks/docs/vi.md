# Kỹ năng Tác tử: Hợp đồng Di động và Ranh giới Runtime (Agent Skills: Portable Contract and Runtime Boundary)

> Một kỹ năng không phải là một prompt dài với một tên tệp nghe kêu hơn. Đó là một gói tài nguyên có thể khám phá được bao gồm các chỉ dẫn, tài nguyên hỗ trợ và các trình trợ giúp thực thi được đưa vào ngữ cảnh của tác tử thông qua một hợp đồng runtime rõ ràng.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python (thư viện chuẩn)
**Yêu cầu tiên quyết:** Phase 13 · 01 (Giao diện Công cụ), Phase 13 · 05 (Thiết kế Lược đồ Công cụ)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Định nghĩa một kỹ năng tác tử (agent skill) mà không nhầm lẫn nó với prompt, chỉ dẫn kho mã nguồn (repository instructions), công cụ (tool), hook sự kiện, tác tử con (subagent) hay plugin.
- Đọc hiểu hợp đồng `SKILL.md` di động và tách biệt nó khỏi các phần mở rộng mang tính đặc thù của từng runtime.
- Giải thích các giai đoạn riêng biệt trong vòng đời: khám phá (discovery), lựa chọn (selection), kích hoạt (activation), nạp tài nguyên (resource loading), sử dụng công cụ (tool use) và xác minh (verification).
- Xác thực một gói kỹ năng trước khi runtime đưa nó vào danh mục của tác tử.
- Lựa chọn chính xác giữa kỹ năng, công cụ MCP, hook, tác tử con hoặc mã nguồn thông thường cho một tác vụ cụ thể.

## Thành công Đầu tiên trong Mười Phút (Ten-Minute First Success)

Hãy thực hiện phần này trước phần giải thích chi tiết. Bạn sẽ tạo một kỹ năng nhỏ, cài đặt gói đánh giá (reviewer bundle) hoàn chỉnh vào một môi trường máy chủ tác tử (agent host) thực tế, kích hoạt nó, xác minh kết quả và gỡ bỏ nó. Điều này chứng minh toàn bộ vòng đời với kết quả có thể quan sát trực tiếp.

### Chuẩn bị kiểm tra trước cho bài thực hành trên máy chủ thực tế

Điểm kiểm tra trên host thực tế yêu cầu Node.js, `npx`, Python 3, một máy chủ hỗ trợ kỹ năng được chọn và quyền ghi vào phạm vi dự án hoặc người dùng mà bạn chọn trong trình cài đặt. Hãy kiểm tra các lệnh cục bộ trước:

```bash
node --version
npx --version
python3 --version
```

Quyết định máy chủ và phạm vi bạn sẽ sử dụng trước khi cài đặt. Nếu thiếu bất kỳ yêu cầu nào, hãy đọc bài học này trên website hoặc tiếp tục với bài tập đóng gói thủ công bên dưới. Phương án dự phòng đó giúp bạn hiểu hợp đồng, nhưng không chứng minh được hành vi khám phá của host, kích hoạt, thực thi tập lệnh đi kèm hoặc hành vi gỡ cài đặt. Hãy đánh dấu các quan sát đó là đang chờ xử lý.

### 1. Bắt đầu trong một thư mục làm việc trống

Chạy các lệnh này từ bất kỳ thư mục cha nào bạn dùng để lưu trữ bài học:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

Lệnh cuối cùng không được in ra nội dung nào. Nếu có tệp xuất hiện, hãy chọn một thư mục trống khác để quá trình đánh giá có ranh giới rõ ràng.

Tạo một thư mục cho kỹ năng đầu tiên của bạn:

```bash
mkdir -p my-first-skill
```

Tạo tệp `my-first-skill/SKILL.md` với nội dung sau:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

Xác minh rằng bạn đã tạo tệp đúng thư mục dự kiến:

```bash
test -f my-first-skill/SKILL.md
```

Không có đầu ra và mã thoát là 0 nghĩa là tệp đã tồn tại.

### 2. Cài đặt gói đánh giá hoàn chỉnh

Vẫn ở trong thư mục `agent-skills-first-run`, hãy chạy:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

Chọn máy chủ tác tử và phạm vi bạn đang dùng. Trình cài đặt sẽ liệt kê `skill-contract-reviewer` và đường dẫn đích mà nó đã ghi. Tùy chọn `--full-depth` là bắt buộc vì kỹ năng của bài học này là một gói lồng nhau gồm các tệp tham chiếu, một tập lệnh và một tài nguyên tĩnh.

Đặt `SKILL_ROOT` thành thư mục tuyệt đối được báo cáo bởi trình cài đặt. Đó phải là thư mục chứa tệp `SKILL.md` đã cài đặt, không phải thư mục nguồn của bài học và không phải không gian làm việc hiện tại:

```bash
# Thay thế phần giữ chỗ bằng đường dẫn đích được in ra bởi trình cài đặt.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

Nếu phiên làm việc của tác tử đã mở từ trước, hãy khởi động một phiên mới hoặc sử dụng lệnh quét lại kỹ năng (rescan) của máy chủ đó. Không nên cho rằng mọi host đều tự động nạp lại danh mục (hot-reload).

### 3. Gọi kỹ năng một cách tường minh

Bên trong tác tử đã cài đặt, với `agent-skills-first-run` là thư mục làm việc, hãy sử dụng cú pháp được máy chủ đó hỗ trợ:

| Máy chủ | Lệnh gọi tường minh |
|---|---|
| Codex | `skill-contract-reviewer`, hoặc chọn nó từ `/skills`, sau đó cung cấp yêu cầu đánh giá |
| Claude Code | `/skill-contract-reviewer` kèm theo yêu cầu đánh giá |
| Dự phòng di động | `Use skill-contract-reviewer to review the target package.` |

Sử dụng các giá trị tuyệt đối đã được in ra cho `SKILL_ROOT` và `TARGET_ROOT` trong yêu cầu. Yêu cầu máy chủ mở rộng chúng trước khi thực thi và hiển thị lệnh đầy đủ đã được phân giải, không dùng lệnh phụ thuộc vào thư mục làm việc của tiến trình:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

Lệnh đã phân giải cần có cấu trúc như sau, không còn chứa phần giữ chỗ nào:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

Một kết quả thành công phải đáp ứng đủ cả ba tiêu chí:

1. Máy chủ tìm thấy `skill-contract-reviewer` theo tên.
2. Bộ đánh giá đọc hợp đồng gói và chạy trình xác thực đi kèm.
3. Phản hồi chứa một báo cáo xác thực không có lỗi cấu trúc nào cho mẫu kỹ năng, kèm theo lựa chọn khối cơ bản (primitives) có căn cứ rõ ràng.

Bằng chứng thực thi cũng phải nêu rõ đường dẫn tập lệnh, đường dẫn mục tiêu, cwd, mảng đối số chính xác và mã thoát. Một báo cáo mượt mà nhưng thiếu các trường này không chứng minh được rằng tập lệnh đi kèm đã thực sự chạy.

Nếu máy chủ báo cáo rằng kỹ năng không khả dụng, hãy kiểm tra lại vị trí cài đặt, quét lại hoặc khởi động lại một lần, và thử lại yêu cầu tường minh. Đừng viết lại mô tả kỹ năng để che giấu một lỗi cài đặt.

### 4. Thử nghiệm lựa chọn ngầm định (Implicit selection)

Bắt đầu một lượt hội thoại mới của tác tử và nhập cùng tác vụ mà không nêu tên kỹ năng:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

Nếu máy chủ hiển thị các kỹ năng được chọn, hãy ghi lại xem nó có chọn `skill-contract-reviewer` hay không. Nếu máy chủ không hiển thị việc định tuyến, hãy đánh dấu lựa chọn ngầm định là chưa được xác minh. Gọi tường minh là cơ chế dự phòng có tính di động cao nhất.

### 5. Dọn dẹp

Chỉ gỡ bỏ gói đánh giá đã cài đặt:

```bash
npx skills remove skill-contract-reviewer
```

Chọn cùng máy chủ và phạm vi đã dùng trong quá trình cài đặt. Sau khi quét lại hoặc mở phiên mới, một yêu cầu tường minh gọi `skill-contract-reviewer` phải báo cáo là không khả dụng. Giữ lại `my-first-skill` cho các bài học tiếp theo, hoặc xóa thư mục thực hành sau khi bạn hoàn thành toàn bộ chuyên đề.

## Vấn đề đặt ra

Giả sử đội ngũ của bạn có một quy trình phát hành đáng tin cậy: tìm kiếm các thay đổi đã hợp nhất, kiểm tra ghi chú di chuyển (migration notes), cập nhật nhật ký thay đổi (changelog), chạy lệnh đóng gói và tạo danh sách kiểm tra đánh giá (review checklist).

Nếu đưa toàn bộ quy trình đó vào một prompt duy nhất, nó sẽ rất dễ dán vào nhưng lại cực kỳ khó vận hành. Prompt không có định danh ổn định, không có quy tắc khám phá, không có ranh giới tài nguyên, không có cấu trúc đóng gói kiểm thử được, và không trả lời được các câu hỏi cơ bản: Ai được phép gọi nó? Khi nào mô hình nên chọn nó? Nó có thể chạy những tập lệnh nào? Những tệp nào là đáng tin cậy? Điều gì còn tồn tại khi ngữ cảnh bị nén gọn?

Sai lầm ngược lại là coi mọi chỉ dẫn tái sử dụng được đều là kỹ năng. Quy ước kho mã nguồn, tự động hóa tất định, công cụ bên ngoài, hook sự kiện và các tác tử ủy quyền giải quyết những bài toán hoàn toàn khác nhau. Việc nhồi nhét tất cả vào `SKILL.md` tạo ra một thư mục trông có vẻ di động nhưng thực chất lại phụ thuộc vào hành vi không được tài liệu hóa của một máy chủ cụ thể.

Tác vụ kỹ thuật đầu tiên là phân loại. Hãy quyết định xem thành phần tạo tác đó là gì trước khi quyết định cách đóng gói nó.

## Khái niệm cốt lõi

### Kỹ năng mã hóa tri thức thủ tục (Procedural knowledge)

Một kỹ năng tác tử là một thư mục có điểm vào là `SKILL.md`. Tệp điểm vào chứa phần mở đầu YAML (YAML frontmatter) theo sau bởi các chỉ dẫn Markdown. Thư mục này cũng có thể chứa các tài liệu tham khảo (references), tập lệnh (scripts) và tài nguyên tĩnh (assets).

```figure
skill-package-anatomy
```

Thư mục — chứ không chỉ riêng tệp Markdown — mới là đơn vị có thể triển khai được. Một tệp `SKILL.md` được sao chép nhưng thiếu các tài liệu tham khảo đi kèm là một gói bị hỏng, ngay cả khi phần frontmatter của nó được phân tích cú pháp hợp lệ.

### Các trừu tượng lân cận

| Thành phần tạo tác | Nhiệm vụ chính | Được nạp hoặc chạy khi nào | Không nên mạo danh điều gì |
|---|---|---|---|
| Prompt | Định hình một tương tác mô hình | Được ứng dụng hoặc người dùng đưa vào | Một gói được đánh phiên bản kèm tài nguyên |
| Chỉ dẫn kho mã nguồn (Repository instructions) | Giải thích quy tắc thường trực của một codebase | Runtime lập trình tiến vào phạm vi đó | Một quy trình tác vụ tái sử dụng được |
| Kỹ năng tác tử (Agent skill) | Cung cấp tri thức thủ tục tái sử dụng được | Kích hoạt tường minh hoặc ngầm định | Ranh giới phân quyền kiểm soát truy cập cứng |
| Công cụ MCP (MCP tool) | Phơi bày năng lực từ xa có định kiểu | Mô hình hoặc ứng dụng gọi tới | Một quy trình vận hành chi tiết |
| Hook | Chạy logic tất định khi có sự kiện | Sự kiện đã khai báo xảy ra | Định tuyến mô hình mang tính xác suất |
| Tác tử con (Subagent) | Ủy quyền công việc với ngữ cảnh và trạng thái tách biệt | Bộ điều phối tạo ra hoặc gọi tới | Một gói chỉ dẫn tĩnh |
| Plugin | Phân phối một phần mở rộng runtime lớn hơn | Máy chủ cài đặt hoặc kích hoạt | Bản thân hợp đồng kỹ năng di động |
| Thư viện kỹ năng học được (Learned skill library) | Lưu trữ hành vi được khám phá qua kinh nghiệm | Chính sách truy xuất một chương trình hoặc quỹ đạo trước đó | Một gói `SKILL.md` dựa trên tiêu chuẩn chuẩn hóa |

Một kỹ năng phát hành có thể hướng dẫn tác tử cách kiểm tra một bản phát hành. Một MCP server có thể phơi bày sổ đăng ký phát hành. Một hook có thể ngăn chặn việc đẩy mã trực tiếp. Một tác tử con có thể kiểm toán độc lập ứng viên phát hành. Các mảnh ghép này kết hợp nhịp nhàng với nhau vì mỗi bên giữ một trách nhiệm riêng biệt.

### Từ "kỹ năng" (skill) chỉ hai khái niệm khác nhau

Các hệ thống nghiên cứu đôi khi gọi một chương trình học được, một quỹ đạo thành công hoặc một mảnh chính sách đặc thù theo môi trường là một kỹ năng. Tác tử có thể tạo ra các thành phần này trong quá trình khám phá, truy xuất chúng theo độ tương đồng tác vụ, thực thi chúng và tinh chỉnh thư viện từ phản hồi. Bài học Phase 14 · 10 xây dựng loại thư viện học tập suốt đời (lifelong-learning library) đó.

Một Kỹ năng Tác tử (Agent Skill) trong bài học này lại khác. Nó là một gói được biên soạn (authored package) với hợp đồng hệ thống tệp được khai báo, siêu dữ liệu danh mục, cơ chế tiết lộ lũy tiến (progressive disclosure), kích hoạt qua trung gian của runtime, và các công cụ do máy chủ kiểm soát. Nó có thể được tạo ra hoặc cải thiện bởi một tác tử, nhưng việc học tập không phải là điều kiện bắt buộc đối với định dạng này.

| Chiều so sánh | Gói Kỹ năng Tác tử (Agent Skill package) | Thư viện kỹ năng học được (Learned skill library) |
|---|---|---|
| Đơn vị chính | Thư mục chứa `SKILL.md` | Chương trình, chính sách, quỹ đạo hoặc bản ghi bộ nhớ |
| Cách tạo | Soạn thảo, sinh tự động hoặc tuyển chọn | Thường được khám phá từ trải nghiệm môi trường |
| Lựa chọn | Mô tả trong danh mục cộng với chính sách runtime | Truy xuất hoặc chính sách dựa trên trạng thái tác vụ |
| Thực thi | Mô hình làm theo chỉ dẫn và gọi công cụ của host | Môi trường chạy một hành vi hoặc đoạn mã được lưu trữ |
| Tính di động | Hợp đồng đóng gói có thể chuyển qua các host tương thích | Thường gắn liền với một môi trường và không gian hành động duy nhất |
| Đánh giá | Định tuyến, tạo tác, độ an toàn và độ tương thích host | Phần thưởng, tỷ lệ thành công, chuyển giao và sự tăng trưởng thư viện |

Cả hai khái niệm đều đóng gói năng lực tái sử dụng. Chúng không nên bị đánh đồng về mặt triển khai chỉ vì dùng chung một tên gọi.

### Phần lõi di động (The portable core)

Đặc tả Agent Skills yêu cầu hai trường bắt buộc trong frontmatter:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` là mã định danh ổn định. Nó phải thỏa mãn các quy tắc đặt tên của đặc tả và khớp với tên thư mục cha. `description` vừa là tài liệu hướng dẫn vừa là siêu dữ liệu định tuyến. Nó cần nêu rõ kỹ năng làm gì và áp dụng trong trường hợp nào.

Các trường tùy chọn di động gồm:

| Trường | Mục đích | Ghi chú về tính di động |
|---|---|---|
| `license` | Tuyên bố điều khoản sử dụng gói | Đặc tả cốt lõi |
| `compatibility` | Nêu các yêu cầu về môi trường | Đặc tả cốt lõi |
| `metadata` | Chứa dữ liệu mở rộng dạng chuỗi ký tự | Đặc tả cốt lõi |
| `allowed-tools` | Đề xuất các công cụ được phê duyệt trước | Thực nghiệm; hỗ trợ tùy thuộc vào từng host |

Phần thân Markdown chứa các chỉ dẫn vận hành. Nó phải xác định rõ luồng công việc, các điểm ra quyết định, hành vi khi xảy ra sự cố và đường dẫn trực tiếp tới các tài nguyên hỗ trợ.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Phần mở rộng runtime là tầng thứ hai

Một số máy chủ chấp nhận các trường frontmatter bổ sung hoặc cấu hình đi kèm. Những trường đó có thể hữu ích, nhưng chúng không tự động mang tính di động.

| Hành vi | Ví dụ phần mở rộng của host | Phần lõi di động? |
|---|---|:---:|
| Ẩn kỹ năng khỏi định tuyến mô hình trong khi vẫn cho phép người dùng gọi trực tiếp | `disable-model-invocation` | Không |
| Ẩn kỹ năng khỏi menu lệnh của người dùng trong khi vẫn cho phép mô hình định tuyến | `user-invocable` | Không |
| Hiển thị trợ giúp về đối số trong menu lệnh | `argument-hint` | Không |
| Chạy kỹ năng trong ngữ cảnh được ủy quyền | `context`, `agent` | Không |
| Gắn cố định mô hình hoặc thiết lập mức suy luận | `model`, `effort` | Không |
| Đăng ký tự động hóa vòng đời | `hooks` | Không |
| Vô hiệu hóa lệnh gọi ngầm định trong Codex | Chính sách `agents/openai.yaml` | Không |

Hãy coi mỗi phần mở rộng như một adapter (bộ điều hợp). Giữ cho quy trình cốt lõi hoạt động bình thường ngay cả khi không có nó, ghi lại tài liệu về cơ chế dự phòng và kiểm thử trên máy chủ sẽ tiêu thụ nó. Runtime có thể bỏ qua một trường lạ, từ chối nó, hoặc giữ nguyên mà không triển khai hành vi tương ứng.

### Frontmatter là siêu dữ liệu có khả năng thực thi

Siêu dữ liệu thay đổi hành vi hệ thống trước khi phần thân của kỹ năng được đọc:

- Một `name` sai định dạng có thể khiến quá trình khám phá thất bại.
- Một `description` mơ hồ có thể định tuyến sai yêu cầu.
- Một cờ chỉ dành cho con người có thể loại bỏ kỹ năng khỏi danh mục của mô hình.
- Một quyền cấp phép công cụ có thể thay đổi việc máy chủ có yêu cầu người dùng xác nhận hay không.
- Một thiết lập ngữ cảnh có thể chuyển việc thực thi sang một phiên tác tử tách biệt.

Hãy xem xét frontmatter như mã cấu hình. Cần xác thực, quản lý phiên bản và đưa hành vi của nó vào các bộ đánh giá (evals).

### Vòng đời kỹ năng (The skill lifecycle)

```figure
skill-runtime-lifecycle
```

Mỗi mũi tên đại diện cho một ranh giới với các dạng lỗi riêng:

1. **Khám phá (Discovery):** Tìm kiếm các gói khả dĩ trong các vị trí được cấu hình.
2. **Xác thực (Validation):** Từ chối các gói sai định dạng hoặc không an toàn trước khi công bố vào danh mục.
3. **Lập danh mục (Cataloging):** Phơi bày `name` và `description` ngắn gọn, không nạp toàn bộ gói.
4. **Lựa chọn (Selection):** Quyết định xem kỹ năng có phù hợp hay không.
5. **Kích hoạt (Activation):** Nạp phần thân vào ngữ cảnh hiển thị của mô hình.
6. **Tiết lộ (Disclosure):** Chỉ đọc các tài liệu tham khảo hoặc tài nguyên khi có một nhánh xử lý yêu cầu.
7. **Thực thi (Execution):** Sử dụng các công cụ của host theo các quy tắc phân quyền và cách ly của host đó.
8. **Xác minh (Verification):** Kiểm tra sản phẩm tạo tác một cách độc lập với tuyên bố của mô hình.

Việc gộp chung các giai đoạn này dẫn đến những mô hình tư duy sai lầm. Một kỹ năng được khám phá không có nghĩa là nó đang hoạt động. Một kỹ năng đang hoạt động không có nghĩa là nó được phép thực hiện mọi hành động mà nó mô tả. Một lệnh gọi công cụ được cho phép không phải là bằng chứng cho thấy kết quả là chính xác.

### Kỹ năng và công cụ mang tính trực giao (Orthogonal)

MCP trả lời câu hỏi: "Ứng dụng này có thể gọi những năng lực nào, và lược đồ của chúng là gì?" Kỹ năng trả lời câu hỏi: "Tác tử nên tiếp cận nhóm tác vụ này như thế nào?"

```figure
skill-tool-orthogonality
```

Kỹ năng có thể nêu tên một công cụ, nhưng máy chủ mới là bên nắm giữ sổ đăng ký năng lực thực tế. Nếu công cụ vắng mặt, kỹ năng phải nêu rõ phương án dự phòng hoặc báo lỗi rõ ràng. Nó không bao giờ được ngụ ý rằng việc gọi tên một năng lực sẽ tự động tạo ra năng lực đó.

### Kỹ năng và chỉ dẫn kho mã nguồn thuộc phạm vi khác nhau

Chỉ dẫn kho mã nguồn mô tả môi trường bạn đang đứng bên trong: các lệnh, quy ước, tệp được sinh tự động và các ranh giới. Kỹ năng cung cấp quy trình tái sử dụng được cho một tác vụ có thể xuất hiện xuyên suốt nhiều kho mã nguồn khác nhau.

Khi cả hai cùng áp dụng, yêu cầu trực tiếp của người dùng và các quy tắc của kho mã nguồn sẽ ràng buộc kỹ năng. Một kỹ năng tái cấu trúc tổng quát không được phép ghi đè quy tắc của kho mã nguồn vốn nghiêm cấm chỉnh sửa các tệp được sinh tự động.

### Kỹ năng không import lẫn nhau

Một kỹ năng có thể hướng dẫn tác tử gọi một kỹ năng khác, nhưng đây không phải là một lệnh import ở cấp độ ngôn ngữ lập trình. Kỹ năng thứ hai vẫn phải trải qua quá trình khám phá của runtime, kiểm tra tính hợp lệ, kích hoạt, phân quyền và xử lý ngữ cảnh.

Hãy viết các phụ thuộc liên kỹ năng dưới dạng các cạnh luồng công việc quan sát được:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

Cách tiếp cận này giúp mối quan hệ phụ thuộc có thể kiểm thử được và trao cho máy chủ cơ hội thực thi chính sách.

## Xây dựng thực tế (Build It)

Tệp `code/main.py` triển khai một bộ xác thực nhỏ hướng chuẩn và một bộ chọn thành phần tạo tác. Mã nguồn hoàn toàn dùng thư viện chuẩn Python để mọi quy tắc đều tường minh.

Bộ xác thực cung cấp:

- `parse_frontmatter(text)` để tách biệt siêu dữ liệu khỏi phần thân.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())` để kiểm tra các trường bắt buộc, quy tắc đặt tên, phần mở rộng lạ, sự hiện diện của phần thân và các giới hạn di động.
- `ValidationIssue` và `SkillReport` để trả về bằng chứng có cấu trúc thay vì một giá trị boolean mờ mịt.
- `FrontmatterSyntaxError` cho dữ liệu đầu vào không thể phân tích an toàn.

Bộ chọn cung cấp `TaskShape` và `select_primitives(task)`. Nó ánh xạ nhu cầu của tác vụ sang mã thông thường, chỉ dẫn kho mã nguồn, kỹ năng, hook, tác tử con hoặc công cụ MCP.

Chạy thử nghiệm:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối lệnh này yêu cầu một bản sao chép (clone) cục bộ và phải bắt đầu từ bất kỳ đâu bên trong bản sao chép đó để `git rev-parse --show-toplevel` có thể xác định thư mục gốc của kho mã.

Bản demo in ra định dạng JSON cho một kỹ năng di động hợp lệ, một kỹ năng có phần mở rộng của host, một gói không hợp lệ và một số quyết định hình dạng tác vụ. Hãy kiểm tra các mã lỗi. Một bộ xác thực gói phải giải thích cách khắc phục một thành phần mà không đoán mò thay cho tác giả.

### Trình tự xác thực đóng vai trò quyết định

Hãy kiểm tra các đặc tính cấu trúc cơ bản trước khi kiểm tra các quy tắc nội dung sâu hơn:

```figure
skill-validation-order
```

Trình tự này giúp ngăn ngừa các lỗi thứ cấp làm lu mờ bất biến bị vi phạm đầu tiên.

## Sử dụng trong thực tế (Use It)

Trước khi viết một kỹ năng, hãy điền vào bảng quyết định này:

| Câu hỏi | Nếu có | Khối cơ bản thích hợp |
|---|---|---|
| Tác vụ này có cần sự phán đoán tái sử dụng của mô hình qua nhiều bước không? | Quy trình ổn định nhưng quyết định thay đổi linh hoạt | Kỹ năng (Skill) |
| Thao tác này có bắt buộc phải diễn ra mỗi khi một sự kiện phát sinh không? | Bỏ lỡ một lần thực thi là không thể chấp nhận | Hook hoặc mã ứng dụng |
| Mô hình có cần một năng lực bên ngoài với đầu vào định kiểu không? | Thao tác nằm ngoài ngữ cảnh của mô hình | Công cụ hoặc máy chủ MCP |
| Công việc có cần ngữ cảnh, trạng thái hoặc quyền sở hữu tách biệt không? | Một worker độc lập trả về kết quả có ranh giới rõ ràng | Tác tử con (Subagent) |
| Chỉ dẫn này có mang tính đặc thù riêng cho một kho mã nguồn không? | Mô tả các lệnh và ràng buộc cục bộ | Chỉ dẫn kho mã nguồn |
| Một lượt tương tác duy nhất đã đủ chưa? | Không cần vòng đời đóng gói | Lời nhắc (Prompt) |

Nhiều quy trình sản xuất kết hợp nhiều hàng trong bảng trên. Bảng này giúp ngăn một thành phần đơn lẻ cố gắng giả định mọi đặc tính.

## Chuyển giao sản phẩm (Ship It)

Bài học này tạo ra gói `skill-contract-reviewer` trong thư mục `outputs/`. Nó chứa:

- tệp `SKILL.md` di động dùng để đánh giá một gói kỹ năng được đề xuất;
- danh sách kiểm tra tham chiếu cho hợp đồng di động và lựa chọn khối cơ bản;
- tập lệnh xác thực tất định;
- các fixture hình dạng tác vụ bao gồm prompt, kỹ năng, công cụ, hook, mã thông thường và tác tử con.

Hãy cài đặt toàn bộ gói, không chỉ riêng tệp điểm vào:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

Trình cài đặt khóa học báo cáo từng kỹ năng Phase 13 đã sao chép và ghi ra tệp `/tmp/aiefs-skills/manifest.json`. Thư mục đích sạch này giúp kiểm tra cấu trúc đóng gói; vòng lặp thành công đầu tiên ở trên kiểm tra quá trình khám phá và kích hoạt trên một host thực tế.

Các bài học tiếp theo sẽ đi sâu vào từng giai đoạn trong vòng đời. Bài 24 xây dựng cơ chế khám phá và tiết lộ lũy tiến. Bài 25 xây dựng chính sách kích hoạt và định tuyến. Bài 26 phân tách quyền hạn khỏi môi trường hộp cát (sandboxing). Bài 27 biến toàn bộ gói thành một sản phẩm phát hành được đánh giá toàn diện.

## Bài tập

1. Phân loại năm quy trình làm việc từ chính đội ngũ của bạn bằng `TaskShape`. Bảo vệ lập luận cho mọi trường hợp bạn chọn nhiều hơn một khối cơ bản.
2. Thêm các bài kiểm thử biên chứng minh rằng giá trị `compatibility` 500 ký tự sẽ vượt qua nhưng giá trị 501 ký tự sẽ thất bại dưới dạng lỗi đặc tả.
3. Thêm một phần mở rộng runtime vào danh sách cho phép (allowlist). Viết một bài kiểm thử chứng minh tệp đó vẫn có thể phân biệt được với một kỹ năng thuần di động.
4. Tách một prompt dài 400 dòng thành `SKILL.md`, một tài liệu tham chiếu, một hợp đồng tập lệnh và một mẫu đầu ra. Đảm bảo mỗi tệp chỉ chịu trách nhiệm cho một loại thông tin.
5. Thiết kế phản hồi báo lỗi cho một kỹ năng tham chiếu tới một công cụ MCP không khả dụng. Không được âm thầm thay thế bằng một công cụ có quyền hạn rộng hơn.
6. Đánh giá một kỹ năng hiện có và gán nhãn cho từng câu văn là: định tuyến, thủ tục, chính sách, con trỏ tham chiếu hay hợp đồng đầu ra. Di chuyển bất kỳ nội dung nào không thuộc đúng vị trí.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Kỹ năng tác tử (Agent skill) | "Một prompt đã lưu" | Thư mục có thể khám phá chứa các chỉ dẫn thủ tục và tài nguyên tùy chọn |
| Phần lõi di động (Portable core) | "Các trường mọi runtime cùng chia sẻ" | Hợp đồng được định nghĩa bởi đặc tả Agent Skills |
| Phần mở rộng runtime (Runtime extension) | "Frontmatter bổ sung" | Cấu hình đặc thù của host đòi hỏi bộ điều hợp tương thích để vận hành |
| Kích hoạt (Activation) | "Kỹ năng đã chạy" | Phần thân của kỹ năng đã nạp vào ngữ cảnh hiển thị của mô hình; việc thực thi có thể diễn ra sau |
| Phụ thuộc kỹ năng (Skill dependency) | "Import một kỹ năng khác" | Cạnh gọi qua trung gian runtime kèm kiểm tra tính sẵn sàng và chính sách |
| Hợp đồng công cụ (Tool contract) | "Lược đồ hàm" | Đầu vào, đầu ra, quyền hạn, hiệu ứng phụ, lỗi và bằng chứng cho một năng lực |

## Tài liệu đọc thêm

- [Agent Skills specification](https://agentskills.io/specification) cho hợp đồng thư mục và frontmatter di động.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) cho phạm vi, chỉ dẫn và tổ chức tài nguyên.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) cho hành vi khám phá và kích hoạt hiện tại trong Codex.
- [Claude Code skills](https://code.claude.com/docs/en/skills) cho các phần mở rộng về lệnh gọi, đối số, công cụ và ngữ cảnh ủy quyền trong một runtime cụ thể.
