# Khám phá Kỹ năng và Tiết lộ Lũy tiến (Skill Discovery and Progressive Disclosure)

> Một kỹ năng phát huy giá trị ngay cả trước khi phần thân của nó được nạp vào bộ nhớ. Tên và mô tả giúp nó có một vị trí trong danh mục; các tệp tài nguyên chuyên sâu chỉ được đưa vào ngữ cảnh khi tác vụ thực sự chạm tới chúng.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python (thư viện chuẩn)
**Yêu cầu tiên quyết:** Phase 13 · 22 (Kỹ năng Tác tử: Hợp đồng Di động và Ranh giới Runtime)
**Thời lượng:** ~105 phút

## Mục tiêu học tập

- Xây dựng một đường ống khám phá hệ thống tệp tách biệt rõ ràng giữa phạm vi (scope), xác thực (validation), chính sách xung đột (collision policy) và công bố danh mục (catalog publication).
- Giải thích ba cấp độ tiết lộ: siêu dữ liệu danh mục, chỉ dẫn hoạt động và tài nguyên chuyên biệt theo tác vụ.
- Thiết kế các tài liệu tham chiếu (references) để tác tử có thể truy cập trực tiếp chi tiết cần thiết mà không phải nạp toàn bộ gói.
- Phân bổ ngân sách không gian danh mục tách biệt với ngữ cảnh kỹ năng đang hoạt động.
- Ngăn chặn tấn công duyệt đường dẫn (path traversal) và thoát ranh giới qua liên kết mềm (symlink escape) khi một kỹ năng đọc các tài nguyên của chính nó.

## Vấn đề đặt ra

Tác tử của bạn có 200 kỹ năng đã cài đặt. Việc nạp mọi tệp `SKILL.md`, tệp tham chiếu, tập lệnh và biểu mẫu mẫu ngay từ khi bắt đầu phiên làm việc sẽ làm cho tác vụ hiện tại bị chôn vùi trong những quy trình không liên quan. Nhưng nếu không nạp bất kỳ thứ gì, người dùng sẽ buộc phải tự nhớ chính xác từng đường dẫn trên hệ thống tệp.

Giải pháp trung hòa thường dùng là một danh mục (catalog): hiển thị cho mô hình một định danh nhỏ gọn và mô tả định tuyến cho mỗi kỹ năng đủ điều kiện, sau đó chỉ nạp toàn bộ phần thân sau khi kỹ năng đó được lựa chọn. Điều này lại làm nảy sinh hai vấn đề kỹ thuật mới.

Thứ nhất, việc khám phá không chỉ đơn giản là tìm kiếm tệp đệ quy. Các kỹ năng có thể tồn tại ở các phạm vi khác nhau: dự án (project), người dùng (user), quản trị viên (administrator), plugin hoặc tích hợp sẵn (built-in). Hai gói kỹ năng có thể trùng tên nhau. Một liên kết mềm (symlink) có thể trỏ ra ngoài thư mục gốc tin cậy. Một gói bị lỗi định dạng có thể chiếm dụng không gian danh mục hoặc trở nên bất khả thi để gọi.

Thứ hai, việc tiết lộ lũy tiến có thể biến thành sự bối rối lũy tiến. Nếu `SKILL.md` ghi "hãy đọc tài liệu hướng dẫn liên quan" và gói kỹ năng lại chứa tới mười hai tài liệu hướng dẫn khác nhau, mô hình sẽ phải đoán mò. Nếu mỗi tài liệu hướng dẫn lại trỏ tới ba tệp nữa, quá trình nạp sẽ biến thành một chuyến duyệt đồ thị không có điểm dừng.

Một runtime tốt sẽ đảm bảo quá trình khám phá mang tính tất định và việc tiết lộ nội dung là hoàn toàn có chủ đích.

## Khái niệm cốt lõi

### Khám phá là một đường ống biên dịch (Compiler pipeline)

Hãy coi hệ thống tệp là dữ liệu đầu vào của trình biên dịch. Không công bố trực tiếp các đường dẫn thô tới mô hình.

```figure
skill-discovery-pipeline
```

Mỗi giai đoạn cần tạo ra dữ liệu có cấu trúc và các thông báo lỗi có cấu trúc. Nhật ký khám phá phải trả lời được các câu hỏi:

- Những thư mục gốc nào đã được quét?
- Những ứng viên nào đã được tìm thấy?
- Những ứng viên nào bị từ chối, và vì lý do gì?
- Gói kỹ năng nào đã chiến thắng khi xảy ra xung đột tên?
- Những mục danh mục nào đã bị rút ngắn hoặc lược bỏ do vượt quá ngân sách?

Nếu thiếu những bằng chứng này, việc chẩn đoán nguyên nhân "mô hình không sử dụng kỹ năng của tôi" là điều gần như bất khả thi.

### Phạm vi là chính sách của runtime

Đặc tả di động định nghĩa một gói kỹ năng, chứ không định nghĩa một đường dẫn cài đặt phổ quát hay thứ tự ưu tiên duy nhất. Máy chủ (host) sẽ tự quyết định vị trí tìm kiếm.

Một runtime thông thường có thể sử dụng các phạm vi sau:

| Phạm vi | Thư mục gốc ví dụ | Chủ sở hữu dự kiến |
|---|---|---|
| Không gian làm việc (Workspace) | `<repo>/.agents/skills/` | Người bảo trì dự án |
| Người dùng (User) | `<user-data>/skills/` | Một nhà phát triển cá nhân |
| Quản trị viên (Administrator) | `<system>/skills/` | Chính sách của máy hoặc tổ chức |
| Plugin | Một gói plugin có chữ ký | Nhà phát hành và người cài đặt plugin |
| Tích hợp sẵn (Built-in) | Gói của runtime | Nhà cung cấp runtime |

Tính đến tháng 8 năm 2026, Codex tài liệu hóa việc khám phá dự án bắt đầu từ `$CWD/.agents/skills` đi ngược qua các thư mục tổ tiên lên đến thư mục gốc của kho mã nguồn, cộng thêm các vị trí của người dùng, quản trị viên và tích hợp sẵn. Nó hỗ trợ các thư mục kỹ năng liên kết mềm. Các tên trùng lặp có thể cùng xuất hiện thay vì được hợp nhất. Đó là các hành vi riêng của Codex, không phải yêu cầu của chuẩn `SKILL.md`; hãy kiểm tra [tài liệu kỹ năng Codex](https://learn.chatgpt.com/docs/build-skills) hiện hành khi viết bộ điều hợp (adapter).

Tuyệt đối không tự suy diễn thứ tự ưu tiên từ tên thư mục. Hãy khai báo nó thành chính sách và viết kiểm thử cho nó. Bài thực hành này sử dụng thứ hạng số nguyên tường minh cho từng `Scope` để cùng một tập hợp ứng viên luôn được giải quyết theo cùng một cách thức nhất quán.

### Xung đột đòi hỏi định danh vượt ra ngoài trường `name`

Hai gói kỹ năng cùng mang tên `release-readiness` hoàn toàn có thể là trường hợp hợp lệ. Một gói có thể là bản ghi đè của không gian làm việc và gói kia là bản mặc định của người dùng. Do đó, một mục danh mục cần tối thiểu các thông tin:

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

Các chính sách xử lý xung đột phổ biến gồm:

| Chính sách | Lợi ích | Rủi ro |
|---|---|---|
| Giữ lại mọi ứng viên | Không có gì bị ẩn giấu | Mô hình nhìn thấy các tên mơ hồ, trùng lặp |
| Phạm vi có thứ tự ưu tiên cao nhất thắng | Lệnh gọi đơn giản | Một gói cục bộ có thể che khuất (shadow) một gói tin cậy |
| Từ chối các mục trùng lặp | Không có sự che khuất ngầm định | Các bản ghi đè hợp lệ bị ngừng hoạt động |
| Định danh tên kèm nguồn gốc | Định danh tường minh, rõ ràng | Tên hiển thị với người dùng trở nên dài dòng |

Hãy chọn một chính sách thống nhất cho host. Lưu giữ các ứng viên bị từ chối hoặc bị che khuất trong nhật ký chẩn đoán ngay cả khi chúng vắng mặt trong danh mục gửi tới mô hình.

### Ba cấp độ tiết lộ (Three disclosure levels)

Đặc tả Agent Skills mô tả cơ chế nạp theo từng giai đoạn. Điểm mấu chốt là mỗi cấp độ phục vụ một mục đích hoàn toàn khác biệt.

```figure
skill-disclosure-levels
```

#### Cấp độ 1: Siêu dữ liệu danh mục (Catalog metadata)

Mô hình cần đầy đủ thông tin để phân biệt kỹ năng này với các kỹ năng lân cận. Đặc tả ước tính khoảng 100 token cho mỗi mục danh mục, nhưng việc tuần tự hóa và token hóa thực tế thuộc quyền quản lý của host.

Một mô tả hữu ích thường có hai mệnh đề:

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

Mệnh đề thứ nhất nêu rõ năng lực. Mệnh đề thứ hai xác định ranh giới kích hoạt. Bài học 25 sẽ đánh giá ranh giới này bằng các prompt tích cực và các prompt cận biên (near-miss).

#### Cấp độ 2: Chỉ dẫn hoạt động (Active instructions)

Sau khi được kích hoạt, phần thân của kỹ năng phải đóng vai trò như một bản đồ dẫn đường và một quy trình thực thi. Đặc tả khuyến nghị giữ `SKILL.md` dưới 500 dòng. Đó là một tín hiệu thiết kế, không phải chỉ tiêu cần lấp đầy.

Phần thân cần chứa:

- ranh giới tác vụ;
- quy trình mặc định;
- các điều kiện phân nhánh;
- tham chiếu trực tiếp tới các tệp sâu hơn;
- hợp đồng của công cụ và tập lệnh;
- hành vi khi gặp lỗi và điều kiện dừng;
- đầu ra kỳ vọng và phương pháp xác minh đầu ra đó.

Không chuyển quy trình trung tâm sang tệp tham chiếu chỉ nhằm mục đích làm cho tệp điểm vào ngắn lại. Việc kích hoạt bắt buộc phải cung cấp đủ ngữ cảnh để mô hình có thể bắt đầu một cách chuẩn xác.

#### Cấp độ 3: Tài nguyên hỗ trợ (Supporting resources)

Tài liệu tham chiếu (references) cung cấp văn bản diễn giải hoặc dữ liệu. Tập lệnh (scripts) cung cấp năng lực tính toán tất định. Tài nguyên tĩnh (assets) được sao chép, điền dữ liệu hoặc chuyển đổi thành sản phẩm bàn giao thay vì được xem như chỉ dẫn thực thi.

| Thư mục | Mô hình có đọc? | Mô hình có thực thi? | Nội dung điển hình |
|---|:---:|:---:|---|
| `references/` | Có, khi cần thiết | Không | lược đồ, chính sách, hướng dẫn nghiệp vụ |
| `scripts/` | Có thể kiểm tra | Thông qua công cụ được cấp quyền | bộ xác thực, bộ chuyển đổi, bộ thu thập |
| `assets/` | Chỉ khi hữu ích | Không | mẫu biểu, fixture, hình ảnh, tệp khởi đầu |

Những cái tên này là quy ước, không phải là những năng lực kỳ diệu. Host vẫn cần có quyền truy cập tệp và một công cụ thực thi tương ứng.

### Tham chiếu theo nhánh vượt trội hơn việc đổ dồn chủ đề

Hãy viết tệp điểm vào như một bản đồ đưa ra quyết định:

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

Cách viết này gán cho mỗi tài liệu tham chiếu một điều kiện nạp quan sát được. Câu lệnh mơ hồ "Đọc thư mục `references/` để biết thêm chi tiết" hoàn toàn không làm được điều đó.

Giữ đồ thị tham chiếu nông và phẳng. Hướng dẫn chính thức khuyến nghị liên kết trực tiếp từ `SKILL.md` và tránh các chuỗi tham chiếu nhiều tầng. Một chặng duy nhất giúp khả năng tiếp cận có thể kiểm thử được và giảm thiểu rủi ro một ràng buộc quan trọng không bao giờ được đưa vào ngữ cảnh.

```figure
skill-reference-map
```

### Ngân sách danh mục và ngữ cảnh hoạt động là hai ngân sách tách biệt

Gọi `c_i` là chi phí danh mục đã tuần tự hóa của kỹ năng `i`, `B_c` là ngân sách danh mục, `b_j` là chi phí phần thân hoạt động, và `r_k` là các tài nguyên thực tế được nạp.

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

Việc cắt giảm một ngân sách không tự động làm giảm ngân sách còn lại. Mô tả ngắn có thể tiết kiệm không gian danh mục trong khi một phần thân dài 900 dòng khi kích hoạt vẫn làm tràn ngập tác vụ. Việc tách phần thân thành các tài liệu tham chiếu chỉ có thể làm giảm chi phí hoạt động khi runtime và các chỉ dẫn thực sự tránh nạp các nhánh không liên quan.

Codex hiện phân bổ danh sách kỹ năng ban đầu ở mức 2% cửa sổ ngữ cảnh khi kích thước cửa sổ ngữ cảnh được xác định. Giá trị 8.000 ký tự chỉ là phương án dự phòng khi kích thước đó chưa rõ; nó không phải là một mức trần thứ hai kết hợp đồng thời với quy tắc 2%. Khi danh mục vượt quá ngân sách áp dụng, các mô tả có thể bị rút ngắn hoặc lược bỏ. Hãy xem những con số đó là chính sách hiện hành của Codex, chứ không phải thuộc tính bất biến của tiêu chuẩn Agent Skills.

### Đường dẫn tài nguyên là một ranh giới tin cậy

Một kỹ năng chỉ được phép đọc các tệp nằm bên trong gói của chính nó. Các phép kiểm tra tiền tố chuỗi ký tự đơn thuần là không đủ:

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

Hãy phân giải thư mục gốc của gói và ứng viên bằng ngữ nghĩa hệ thống tệp, từ chối các đầu vào dạng đường dẫn tuyệt đối, và xác minh rằng ứng viên đã phân giải vẫn nằm trọn bên dưới thư mục gốc đã phân giải. Quyết định xem liên kết mềm có được cho phép hay không trước khi thực hiện khám phá. Nếu cho phép, hãy kiểm tra đích đến đã phân giải trong mỗi lần truy xuất.

```figure
skill-resource-containment
```

Việc khoanh vùng đường dẫn không đồng nghĩa với việc xác lập mức độ tin cậy về mặt nội dung. Một tệp tham chiếu hợp lệ nằm trong gói vẫn có thể chứa các chỉ dẫn độc hại. Bài học 26 sẽ xử lý mối đe dọa này.

### Việc nạp tài nguyên phải quan sát được

Ghi nhận các sự kiện tiết lộ mà không làm lộ các dữ liệu bí mật:

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

Trường lý do (`reason`) biến một lựa chọn ngữ cảnh thành bằng chứng có thể thanh tra được. Nó cũng giúp phát hiện những chỉ dẫn khiến tác tử nạp mọi tệp "phòng hờ".

## Xây dựng thực tế (Build It)

Tệp `code/main.py` xây dựng một công cụ khám phá và tiết lộ mang tính tất định.

Bề mặt khám phá bao gồm:

- `Scope` cho siêu dữ liệu về nguồn gốc và thứ tự ưu tiên;
- `SkillCandidate` cho ứng viên hệ thống tệp chưa qua xác thực;
- `discover_scope(scope)` để liệt kê các thư mục kỹ năng trực tiếp;
- `resolve_collisions(candidates, precedence)` để áp dụng một chính sách đã khai báo;
- `CatalogEntry` và `build_catalog(...)` để công bố siêu dữ liệu có giới hạn;
- `CatalogBudget` để tính toán các mục đã tuần tự hóa mà không đánh đồng ký tự với token phổ quát.

Bề mặt tiết lộ bao gồm:

- `load_skill_body(entry, ...)` cho việc kích hoạt Cấp độ 2;
- `validate_reference(skill_dir, reference)` để kiểm tra ranh giới đường dẫn;
- `load_reference(...)` cho các thao tác đọc có giới hạn ở Cấp độ 3.

Chạy thử nghiệm:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối lệnh này yêu cầu bản sao chép cục bộ và tự phân giải thư mục gốc của kho mã từ bất kỳ thư mục làm việc nào bên trong bản sao chép đó.

Bản demo tạo ra các phạm vi dự án và người dùng tạm thời, chèn vào một trường hợp xung đột, xây dựng danh mục dưới một ngân sách cố tình thiết lập nhỏ, kích hoạt một kỹ năng, và thử nghiệm cả thao tác đọc tham chiếu hợp lệ lẫn hành vi cố tình vượt ranh giới đường dẫn. Không có tệp vĩnh viễn nào bị cài đặt vào hệ thống.

### Tại sao việc khám phá lại ở mức nông (Shallow discovery)

Hàm `discover_scope` chỉ kiểm tra các thư mục con trực tiếp để tìm `SKILL.md`. Nó không đệ quy coi mọi tệp `SKILL.md` lồng nhau là một gói riêng biệt. Điều này bảo toàn ranh giới của gói và tránh việc vô tình công bố các ví dụ hay fixture bên trong một kỹ năng đã cài đặt.

### Tại sao bài thực hành không phân tích cú pháp YAML tùy ý

Bài thực hành hỗ trợ phần frontmatter dạng vô hướng cần thiết cho danh mục của nó. Một runtime trong môi trường sản xuất nên sử dụng bộ phân tích cú pháp YAML an toàn với lược đồ rõ ràng, giới hạn kích thước và vô hiệu hóa việc tạo đối tượng tùy biến. "Chỉ dùng thư viện chuẩn" là một ràng buộc sư phạm, không phải là giấy phép để âm thầm tự tạo ra một biến thể YAML không chuẩn mực.

## Sử dụng trong thực tế (Use It)

Áp dụng danh sách kiểm tra này cho bất kỳ bộ điều hợp khám phá nào:

1. Liệt kê mọi thư mục gốc đã cấu hình và xác định ai có quyền ghi vào đó.
2. Nêu rõ các gói có liên kết mềm có được cho phép hay không.
3. Xác thực tên gói, tên thư mục, siêu dữ liệu bắt buộc và kích thước phần thân điểm vào.
4. Giữ lại nguồn và phạm vi trong định danh nội bộ.
5. Khai báo và kiểm thử hành vi khi gặp tên trùng lặp.
6. Đo lường chính xác danh mục đã tuần tự hóa gửi tới mô hình.
7. Ghi lại lý do tại sao một phần thân hoặc tài nguyên được nạp.
8. Giữ các thao tác đọc tài nguyên luôn nằm trong thư mục gốc đã phân giải của gói.
9. Báo lỗi rõ ràng khi một tệp được tham chiếu bị thiếu.
10. Tái tạo danh mục khi các gói cài đặt hoặc chính sách có sự thay đổi.

## Chuyển giao sản phẩm (Ship It)

Bài học này tạo ra gói `skill-catalog-builder`. Nó quét các thư mục gốc theo thứ tự tường minh, từ chối các tệp điểm vào là liên kết mềm và các trường hợp tên không khớp với thư mục, giải quyết các xung đột xuyên phạm vi, từ chối các trường hợp trùng lặp có cùng độ ưu tiên, và đưa siêu dữ liệu được chọn khớp vào ngân sách đã khai báo về số mục, mô tả và số ký tự tuần tự hóa.

Báo cáo JSON của nó chứa các mục được chọn, các ứng viên bị che khuất, các mục bị lược bỏ, các lỗi xác thực, thứ tự ưu tiên và mức tiêu hao ngân sách. Việc nạp phần thân và tài liệu tham chiếu vẫn là các thao tác runtime tách biệt, do đó bộ xây dựng danh mục không thực thi tập lệnh hay đưa toàn bộ gói vào ngữ cảnh.

## Bài tập

1. Thêm một phạm vi plugin và đặt nó vào giữa độ ưu tiên của người dùng và tích hợp sẵn. Chứng minh kết quả xử lý xung đột bằng một bài kiểm thử.
2. Đổi chính sách xử lý xung đột từ ưu tiên cao nhất sang đặt tên kèm định danh nguồn gốc. Giữ lại cả hai mục trong danh mục.
3. Thêm giới hạn kích thước byte vào `load_reference`. Kiểm thử một tệp có kích thước đúng bằng giới hạn và một tệp vượt quá đúng một byte.
4. Tạo hai mô tả có nội dung gần như giống hệt nhau. Viết lại chúng sao cho các ranh giới kích hoạt không bị chồng chéo.
5. Thêm một tệp kê khai (manifest) chứa mã băm cho từng tệp tham chiếu và tập lệnh. Phát hiện tài nguyên bị chỉnh sửa trước khi nạp nó.
6. Gắn mã đo lường vào bản demo để báo cáo số lượng byte của Cấp độ 1, Cấp độ 2 và Cấp độ 3 một cách tách biệt.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Khám phá kỹ năng (Skill discovery) | "Tìm mọi SKILL.md" | Quét các phạm vi được cấu hình, xác thực gói, gắn thông tin nguồn gốc và áp dụng chính sách |
| Danh mục kỹ năng (Skill catalog) | "Danh sách kỹ năng đã cài" | Siêu dữ liệu định tuyến gọn nhẹ hiển thị cho mô hình dành cho các gói đủ điều kiện |
| Chính sách xung đột (Collision policy) | "Bản trùng nào sẽ thắng" | Quy tắc khai báo trước cho các ứng viên cùng tên đến từ các nguồn khác nhau |
| Tiết lộ lũy tiến (Progressive disclosure) | "Nạp lười (Lazy loading)" | Đưa dần vào ngữ cảnh theo giai đoạn từ danh mục tới phần thân rồi tới tài nguyên chuyên biệt theo nhánh |
| Đồ thị tham chiếu (Reference graph) | "Các tệp được liên kết bởi kỹ năng" | Cấu trúc tài nguyên có thể tiếp cận và các điều kiện nạp của chúng |
| Khoanh vùng đường dẫn (Path containment) | "Ở yên trong thư mục" | Xác minh đích đến của tài nguyên đã phân giải luôn nằm bên trong thư mục gốc đã phân giải của gói |

## Tài liệu đọc thêm

- [Đặc tả Agent Skills](https://agentskills.io/specification) cho cấu trúc gói và các cấp độ tiết lộ lũy tiến.
- [Tối ưu hóa mô tả kỹ năng](https://agentskills.io/skill-creation/optimizing-descriptions) cho siêu dữ liệu định tuyến danh mục.
- [Thực hành tốt nhất về Agent Skills](https://agentskills.io/skill-creation/best-practices) cho các tham chiếu trực tiếp và kích thước tệp điểm vào.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) cho các phạm vi khám phá và giới hạn danh mục hiện tại của Codex.
