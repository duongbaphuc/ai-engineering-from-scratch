# Đồ án Tổng hợp 86 — Bộ máy Quy tắc Hiến định (Constitutional Rules Engine)

> Một quy tắc gồm một tên gọi, một vị từ kiểm tra, và một lời giải thích. Bất cứ thứ gì thiếu một trong ba yếu tố đó chỉ là cảm tính mơ hồ, không phải một quy tắc.

**Type:** Xây dựng (Build)  
**Languages:** Python, YAML  
**Prerequisites:** Các bài học an toàn thuộc Phase 18, các bài học 25-29 Track A thuộc Phase 19  
**Time:** ~90 phút  

## Vấn đề (Problem)

Các bộ phân loại (classifiers) bao quát những lỗi thất bại có thể nhận diện được theo mẫu hình. Các bộ máy quy tắc (rules engines) lại phụ trách những ràng buộc mang tính hợp đồng (contractual constraints). Một đội ngũ xây dựng trợ lý lập trình muốn một ràng buộc như: "mọi phản hồi chứa mã nguồn đều phải kết thúc bằng một khối lệnh có thể thực thi được hoặc một giả định được nêu rõ ràng." Một đội ngũ vận hành bot hỗ trợ khách hàng muốn: "mọi câu trả lời từ chối đều phải đưa ra một bước hành động tiếp theo." Những ràng buộc này không phù hợp để làm mục tiêu cho bộ phân loại. Chúng là các vị từ logic (predicates) được áp trên phản hồi, ngữ cảnh hội thoại, và chính sách hệ thống, đồng thời chúng cần phải dễ đọc, dễ hiểu đối với cả những người không chuyên về kỹ thuật.

Cách biểu diễn chuẩn mực và minh bạch nhất là sử dụng một tệp khai báo (declarative file). Bản "hiến pháp" (constitution) tồn tại dưới dạng tệp YAML nằm cạnh mã nguồn trong hệ thống quản lý phiên bản (VCS), với một quy trình rà soát riêng biệt. Mỗi quy tắc bao gồm một tên gọi `name`, mức độ nghiêm trọng `severity`, một mẫu giải thích `explanation`, và một vị từ được tách thành hai trường: `applies_when` quyết định xem quy tắc đó có nằm trong phạm vi áp dụng hay không, và `must` là điều kiện bắt buộc mà phản hồi phải thỏa mãn. Bộ máy nạp tệp này, đánh giá từng quy tắc đối với đầu ra ứng viên, và trả về một cấu trúc `Violation` cho mỗi quy tắc bị kích hoạt vi phạm. Bộ máy quy tắc trong dự án tổng hợp này kết hợp các vị từ bằng `all_of`, `any_of`, và `not_`, nhờ đó một quy tắc duy nhất có thể biểu đạt: "nếu phản hồi chứa mã lệnh, nó phải kết thúc bằng một khối mã thực thi được VÀ không được viện dẫn bất kỳ thư viện nội bộ nào."

Nửa còn lại của bài học này là quá trình hiệu chỉnh (revision). Một bộ máy quy tắc chỉ biết chặn (block) thì mới chỉ hoàn thành được một nửa nhiệm vụ. Một bộ máy quy tắc biết đề xuất cách khắc phục mới thực sự hữu ích trong vận hành thực tế: trợ lý phác thảo bản thảo phản hồi (draft response), bộ máy cảnh báo các vi phạm, một bộ khắc phục (fixer) tạo ra bản phản hồi đã hiệu chỉnh, và bộ máy sẽ xác nhận lại xem bản sửa đổi đã thỏa mãn các quy tắc hay chưa. Bài học này cung cấp một bộ khắc phục tối thiểu (thay thế regex theo từng quy tắc) và một bản so sánh khác biệt có cấu trúc (structured diff: thêm, xóa, sửa từng dòng) giữa bản thảo ban đầu và bản sau khi hiệu chỉnh.

## Khái niệm (Concept)

```mermaid
flowchart LR
  D[bản thảo phản hồi] --> RE[bộ máy quy tắc]
  RE -->|vi phạm| F[bộ khắc phục]
  F --> R[phản hồi hiệu chỉnh]
  R --> RE2[bộ máy quy tắc duyệt lần 2]
  RE2 -->|phán quyết| OUT[chấp thuận hoặc chuyển tiếp]
  D -.->|so sánh diff| R
```

Một quy tắc có cấu trúc như sau:

```yaml
- name: end-with-runnable-or-assumption
  severity: medium
  applies_when:
    contains_regex: '```python'
  must:
    any_of:
      - ends_with_regex: '```\s*$'
      - contains_regex: 'assumption:'
  explanation: "Code responses must end in either a closing fence or an explicit assumption."
  fix:
    append_if_missing: "\n\nAssumption: example inputs are valid."
```

Các vị từ cơ bản (atomic predicates) gồm: `contains_regex`, `not_contains_regex`, `ends_with_regex`, `starts_with_regex`, `max_words`, `min_words`. Các toán tử kết hợp gồm `all_of`, `any_of`, `not_`. Bộ máy sẽ đánh giá `applies_when` trước; nếu quy tắc không áp dụng, vi phạm được ghi nhận là `not_applicable`. Ngược lại, bộ máy sẽ đánh giá tiếp điều kiện `must` và đưa ra kết quả `pass` hoặc `violation`.

Mức độ nghiêm trọng gồm `low`, `medium`, `high`, tương đồng với bài học 85. Cổng an toàn hạ nguồn (bài học 87) sẽ xử lý vi phạm quy tắc mức `high` giống hệt như phán quyết mức `high` của bộ phân loại: chặn hoàn toàn (block).

Bộ khắc phục (fixer) là một danh sách các thao tác khai báo: `append_if_missing`, `prepend_if_missing`, `replace_regex`. Mỗi thao tác ánh xạ một quy tắc theo tên sang một phép biến đổi tương ứng. Bộ khắc phục được chủ ý giới hạn ở các chỉnh sửa cục bộ; việc viết lại toàn bộ cấu trúc thuộc về một tầng từ chối và hỗ trợ (refusal-and-help layer) riêng biệt không nằm trong phạm vi bài học này.

Bản so sánh khác biệt (diff) được tính toán giữa bản gốc và bản hiệu chỉnh. Nó là một danh sách các bản ghi `Change` với trường `op` (thêm add, xóa remove, chỉnh sửa edit) kèm theo văn bản liên quan. Cổng kiểm soát hạ nguồn có thể ghi nhật ký bản diff này để kiểm toán viên con người có thể theo dõi hành vi của bộ khắc phục theo thời gian.

```figure
cd-constitution-loop
```

## Xây dựng thực tế (Build It)

Tệp `code/rules.yml` chứa toàn bộ bản hiến pháp. Bộ nạp trong `code/main.py` hỗ trợ nạp tệp YAML (khi có PyYAML) hoặc tệp JSON (tích hợp sẵn). Bài học cung cấp tệp `rules.yml` mà các bài kiểm thử có thể phân tích qua cả hai đường dẫn mã. Tệp `code/main.py` định nghĩa các lớp `Engine` và `Fixer` cùng hàm `diff`. Các biểu thức kết hợp được đánh giá đệ quy với cơ chế ngắt sớm (short-circuiting) trên `any_of`.

Bản hiến pháp được phân phối sẵn bao gồm:

- `no-empty-refusal` (medium) - một lời từ chối bắt buộc phải đính kèm một gợi ý hoặc một hướng chuyển tiếp thay thế
- `end-with-runnable-or-assumption` (medium) - các phản hồi chứa mã nguồn phải được đóng lại trọn vẹn
- `no-pii-in-examples` (high) - dữ liệu ví dụ minh họa không được chứa email hoặc số điện thoại
- `cite-when-asserting-fact` (low) - các dòng văn bản bắt đầu bằng "According to" bắt buộc phải có trích dẫn đặt trong dấu ngoặc đơn
- `no-internal-library-leak` (high) - các từ ngữ `internal-only` và `policybot-internal` tuyệt đối không được xuất hiện trong đầu ra
- `bounded-length` (low) - phản hồi không được vượt quá độ dài 800 từ

## Sử dụng thực tế (Use It)

Chạy lệnh `python3 main.py`. Bản chạy thử sẽ đưa ba bản thảo phản hồi qua bộ máy quy tắc, in ra các vi phạm, chạy bộ khắc phục, in ra bản diff so sánh và ghi tệp `outputs/rules_report.json`. Một mẫu dữ liệu có chứa một quy tắc không áp dụng (không có khối mã lệnh trong bản thảo), và báo cáo sẽ thể hiện rõ `not_applicable` cho quy tắc đó để đội ngũ thấy được bộ máy đã đánh giá quy tắc một cách tường minh.

## Phát hành sản phẩm (Ship It)

Tệp `outputs/skill-constitutional-rules-engine.md` tài liệu hóa ngữ pháp định nghĩa quy tắc và các thao tác khắc phục của bộ fixer.

## Bài tập thực hành (Exercises)

1. Bổ sung một quy tắc yêu cầu mọi phản hồi phải kèm cụm từ "Nếu đây là trường hợp khẩn cấp" ("If this is urgent") khi prompt đề cập đến vấn đề an toàn. Sử dụng phép kết hợp vị từ.
2. Thay thế bộ khắc phục regex bằng bộ khắc phục dựa trên bản mẫu (templating fixer) tiếp nhận các vị trí biến có tên (named slots). Minh họa một quy tắc được viết lại theo thiết kế mới này.
3. Bổ sung một endpoint đo lường: khi nhận một tập ngữ liệu các bản thảo, trả về tỷ lệ vi phạm theo từng quy tắc để đội ngũ xác định quy tắc nào đang bị kích hoạt quá mức (over-firing).

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ (Term) | Cách dùng thông thường | Định nghĩa chính xác |
|---|---|---|
| constitution | một tài liệu chính sách mơ hồ | một tệp YAML chứa các quy tắc gồm các vị từ, mức độ nghiêm trọng, và lời giải thích |
| predicate | một bước kiểm tra | một đối tượng có thể gọi ánh xạ từ văn bản sang giá trị boolean, có thể là nguyên tử hoặc kết hợp qua all_of/any_of/not_ |
| violation | một sự cố thất bại | một bản ghi có cấu trúc gồm tên quy tắc, mức độ nghiêm trọng, lời giải thích, và đoạn văn bản khớp vi phạm |
| fixer | một mô hình tinh chỉnh (fine-tune) | một phép biến đổi tất định theo từng quy tắc ánh xạ từ bản thảo sang bản hiệu chỉnh |
| diff | một phép so sánh chuỗi | danh sách có cấu trúc gồm các thao tác add, remove, edit giữa bản thảo gốc và bản hiệu chỉnh |

## Đọc thêm (Further Reading)

Bài học 87 kết hợp bộ máy quy tắc này với bộ phát hiện phía đầu vào và bộ phân loại phía đầu ra thành một cổng an toàn thống nhất duy nhất.
