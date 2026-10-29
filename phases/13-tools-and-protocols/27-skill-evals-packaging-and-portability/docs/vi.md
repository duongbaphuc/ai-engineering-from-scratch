# Đánh giá Kỹ năng, Đóng gói và Khả năng Tương thích Khả chuyển (Skill Evals, Packaging, and Portability)

> Một kỹ năng chỉ thực sự hoàn thiện khi gói của nó vượt qua khâu kiểm tra tĩnh (linting), định tuyến đúng trước các yêu cầu phù hợp, cải thiện một tác vụ được đo lường, vận hành trong khuôn khổ chính sách, và suy giảm tính năng một cách trung thực (degrade honestly) trên một môi trường máy chủ khác.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (stdlib)  
**Điều kiện tiên quyết:** Phase 13 · 22, 24, 25, và 26  
**Thời lượng:** ~150 phút  

## Mục tiêu học tập

- Chuyển đổi một quy trình làm việc chuyên gia thành một kỹ năng bằng cách tách bạch khả năng phán đoán, tính toán tất định, tài liệu tham chiếu và hợp đồng đầu ra.
- Kiểm thử cấu trúc gói, định tuyến kích hoạt, hành vi tác vụ, tính đúng đắn của tập lệnh, tính an toàn và khả năng tương thích khả chuyển thành các tầng riêng biệt.
- Đo lường độ chính xác (precision) và độ bao phủ (recall) của cơ chế kích hoạt bằng các ca tích cực, ca tiêu cực rõ ràng và ca suýt trúng (near misses).
- So sánh hiệu năng khi có và không có kỹ năng qua các lần chạy lặp lại.
- Xây dựng và thực thi một ma trận năng lực đa môi trường chạy (cross-runtime capability matrix) cùng cổng kiểm soát phát hành (release gate) cho các gói kỹ năng hoàn chỉnh.

## Vấn đề

Một kỹ năng hoạt động hoàn hảo trong một buổi trình diễn mẫu (demo). Người dùng hỏi chính xác cụm từ được dùng trong phần mô tả của nó, tác giả biết rõ cần mở tài liệu tham chiếu nào, tập lệnh nhận đầu vào sạch sẽ, và môi trường máy chủ dự kiến nhận diện được mọi trường tùy chỉnh.

Thế nhưng khi bắt đầu đưa vào sử dụng thực tế:

- Mô hình kích hoạt kỹ năng cho một tác vụ lân cận nhưng khác biệt.
- Một yêu cầu hợp lệ dùng cách diễn đạt lạ lẫm, khiến mô hình bỏ sót hoàn toàn.
- Phần thân kỹ năng chỉ cho tác tử biết phải làm gì nhưng không nói rõ tạo phẩm nào chứng minh việc hoàn thành.
- Tập lệnh bị lỗi khi gặp khoảng trắng, khi thực thi lặp lại, hoặc gặp trạng thái dở dang.
- Trình cài đặt gói chỉ sao chép tệp `SKILL.md` nhưng lại bỏ quên các tài liệu tham chiếu phía sau.
- Một môi trường chạy khác bỏ qua các cờ kích hoạt và quyền hạn công cụ được cấp.
- Một lần chạy thành công, nhưng ba lần chạy tương đương sau đó lại rẽ vào các nhánh hoàn toàn khác nhau.

Không có lỗi nào trong số này được phát hiện chỉ bằng việc "tệp Markdown trông có vẻ ổn". Kỹ năng là các gói phần mềm nhỏ có tầng định tuyến và thực thi mang tính xác suất (probabilistic). Chúng đòi hỏi sự phân tách mối quan tâm (separation of concerns) chặt chẽ như bất kỳ giao diện sản xuất nào khác.

## Khái niệm

### Bắt đầu từ một quy trình làm việc thực tế, không phải một chủ đề

"Tạo một kỹ năng Kubernetes" không phải là một phạm vi có thể sử dụng được. Kubernetes chứa hàng trăm tác vụ với các công cụ, rủi ro và đầu ra hoàn toàn khác nhau.

"Chẩn đoán lý do một deployment không đạt trạng thái Available, thu thập bằng chứng mà không làm thay đổi cụm máy chủ, và tạo một báo cáo sự cố có xếp hạng" là một ứng viên kỹ năng tốt. Nó sở hữu:

- một ranh giới kích hoạt rõ ràng;
- một chuỗi ổn định các bước thu thập bằng chứng;
- các điểm quyết định cần đến sự phán đoán;
- các câu lệnh có thể chuyển đổi thành các tập lệnh hoặc công cụ phạm vi hẹp;
- một tạo phẩm được định nghĩa tường minh;
- một ranh giới an toàn: chỉ chẩn đoán ở chế độ đọc (read-only).

Hãy sử dụng bảng câu hỏi phỏng vấn trích xuất sau:

1. Sự kiện chính xác nào khiến một chuyên gia bắt đầu quy trình làm việc này?
2. Những yêu cầu tương tự nào không được phép kích hoạt nó?
3. Bằng chứng đầu tiên mà chuyên gia thu thập là gì?
4. Những quyết định nào phụ thuộc vào bằng chứng đó?
5. Những bước nào đủ tính tất định để viết thành tập lệnh?
6. Những quy tắc chuyên môn nào xứng đáng được đưa vào tài liệu tham chiếu?
7. Hành động nào cần phê duyệt hoặc bắt buộc phải nằm ngoài phạm vi?
8. Tạo phẩm nào chứng minh quy trình làm việc đã hoàn tất?
9. Một người đánh giá độc lập kiểm tra tạo phẩm đó như thế nào?
10. Những bước nào phụ thuộc vào một môi trường chạy cụ thể?

Các câu trả lời sẽ trở thành kiến trúc gói và tập dữ liệu đánh giá (eval set).

### Tách bạch khả năng phán đoán khỏi công việc tất định

````figure skill-workflow-extraction
````

Hãy sử dụng khả năng phán đoán của mô hình cho việc phân loại, ưu tiên, tổng hợp và xử lý các tình huống mơ hồ. Sử dụng tập lệnh hoặc công cụ cho việc phân tích cú pháp, đếm, xác thực, chuyển đổi, truy vấn API có kiểu dữ liệu, và thực thi các bất biến (invariants).

Một phần thân kỹ năng chứa 80 dòng mô phỏng phân tích cú pháp thủ công là rất giòn và dễ gãy (brittle). Một tập lệnh cố gắng đưa ra quyết định kiến trúc mang tính chủ quan là thiếu minh bạch. Hãy đặt từng hành vi vào nơi mà nó có thể được kiểm thử tốt nhất.

### Tác nghiệp gói theo thứ tự phụ thuộc

Đừng bắt đầu bằng việc trau chuốt câu chữ văn phong. Hãy xây dựng từ hợp đồng có thể quan sát được đi dần vào bên trong:

1. **Hợp đồng tạo phẩm (Artifact contract):** xác định các tệp, trường dữ liệu, hoặc quyết định bắt buộc.
2. **Xác minh (Verification):** xác định cách thức kiểm tra từng yêu cầu.
3. **Công cụ thu thập bằng chứng (Evidence tools):** cài đặt các bộ thu thập và bộ xác thực tất định.
4. **Bản đồ quyết định (Decision map):** kết nối các trạng thái bằng chứng với các nhánh rẽ.
5. **Tài liệu tham chiếu (References):** cung cấp chi tiết chuyên môn tại đúng nhánh cần đến nó.
6. **Thân tệp điểm vào (Entry body):** giải thích quy trình, ranh giới, hành vi khi lỗi, và đầu ra.
7. **Mô tả (Description):** nêu rõ năng lực và ranh giới kích hoạt.
8. **Bộ điều hợp môi trường chạy (Runtime adapters):** thêm các phần mở rộng kích hoạt hoặc ngữ cảnh một cách riêng biệt.
9. **Đánh giá (Evals):** chạy các tầng kiểm tra cấu trúc, định tuyến, hành vi, an toàn và khả năng tương thích khả chuyển.
10. **Đóng gói (Package):** cài đặt toàn bộ thư mục và kiểm thử nó từ thư mục đích.

Thứ tự này giúp câu từ văn bản phục vụ một hệ thống có thể kiểm thử được, thay vì sáng tạo ra các tiêu chí thành công sau khi bản demo đã chạy xong.

### Sáu tầng đánh giá

````figure skill-eval-layers
````

Mỗi tầng trả lời một câu hỏi khác nhau. Vượt qua một tầng không thể thay thế cho tầng khác.

## Tầng 1: Cấu trúc Gói (Package Structure)

Việc kiểm tra tĩnh (static linting) cần xác minh các sự thật không đòi hỏi đến mô hình:

- Tệp `SKILL.md` tồn tại tại thư mục gốc của gói;
- Siêu dữ liệu đầu tệp (frontmatter) được phân tích cú pháp an toàn;
- Trường `name` và thư mục cha trùng khớp nhau;
- Các trường bắt buộc có mặt đầy đủ và nằm trong giới hạn cho phép;
- Mọi trường frontmatter phi cốt lõi đều xuất hiện trong danh sách trắng mở rộng runtime của chính sách phát hành;
- Mọi tài liệu tham chiếu trực tiếp đều phân giải được bên trong gói;
- Các tài liệu tham chiếu, tập lệnh, tài sản (assets), và dữ liệu mẫu kiểm thử đều sử dụng phần mở rộng tệp được phép và nằm dưới giới hạn dung lượng byte của chính sách phát hành;
- Không tồn tại liên kết mềm (symlink) bị cấm hoặc tệp đặc biệt;
- Phần thân kỹ năng nằm trong ngân sách ký tự của chính sách phát hành;
- Quét mẫu thông tin bí mật có chủ ý thu hẹp không phát hiện bất kỳ phép gán chứng chỉ hoặc tiêu đề khóa riêng tư (private-key header) rõ ràng nào;
- Các phần `## Output contract` và `## Failure behavior` không bị để trống.

Hãy thực hiện kiểm tra trước cây thư mục vật lý (physical-tree preflight) trước khi phân tích cú pháp `SKILL.md`, dữ liệu đánh giá, bằng chứng, dữ liệu mẫu máy chủ, hoặc bản kê khai. Từ chối thư mục gốc liên kết mềm, thư mục cha hoặc tệp điểm vào là liên kết mềm, tệp thông thường bắt buộc bị thiếu, và tệp đặc biệt trước bất kỳ thao tác đọc nội dung nào. Sau đó mới chạy bộ kiểm tra tĩnh nhận biết nội dung. Việc phân giải đường dẫn gói trước khi kiểm tra trước sẽ xóa sạch bằng chứng về liên kết mềm ở thư mục gốc mà bước kiểm tra cần.

Bộ khung thực hành của bài học cụ thể hóa các giá trị chính sách đó: giới hạn thân tệp 10.000 ký tự, giới hạn tệp đi kèm 1.000.000 byte, danh sách trắng phần mở rộng theo từng thư mục, và tên mở rộng runtime tường minh được cung cấp bởi yêu cầu gói. Đây là các ví dụ về chính sách phát hành, không phải giới hạn toàn cục của Agent Skills. Quét mẫu bí mật là một rào chắn cho các lỗi ngớ ngẩn, không phải bằng chứng cho thấy gói hoàn toàn không có dữ liệu nhạy cảm.

Báo cáo kiểm tra tĩnh nên sử dụng các mã lỗi ổn định. CI có thể chặn các lỗi `E_*` trong khi cho phép các cảnh báo thiết kế `W_*` đã qua xem xét.

Kiểm tra tĩnh chứng minh hình thái của gói. Nó không chứng minh mô hình sẽ chọn hoặc làm theo kỹ năng.

## Tầng 2: Định tuyến Kích hoạt (Trigger Routing)

Hãy tạo các ca kiểm thử có gắn nhãn trước khi chỉnh sửa phần mô tả nhiều lần.

| Loại ca kiểm thử | Mục đích | Ví dụ cho mức độ sẵn sàng phát hành |
|---|---|---|
| Tích cực (Positive) | Đo lường độ bao phủ dự kiến | "Phiên bản 3.1.0 có thể phát hành được chưa?" |
| Diễn giải tích cực (Paraphrased positive) | Tránh việc học vẹt câu chữ | "Hãy kiểm toán thẻ tag này trước khi chúng tôi xuất bản" |
| Tiêu cực rõ ràng (Clear negative) | Bắt lỗi định tuyến quá đà nghiêm trọng | "Hãy giải thích chuẩn hóa theo lô (batch normalization)" |
| Ca suýt trúng (Near miss) | Định nghĩa ranh giới với kỹ năng lân cận | "Tại sao bản dựng gói phần mềm bị thất bại?" |
| Kỹ năng cạnh tranh (Competing skill) | Kiểm thử việc lựa chọn giữa các mục hợp lý | "Hãy soạn thảo ghi chú phát hành (release notes)" |
| Câu chữ đối kháng (Adversarial wording) | Kiểm thử việc nhồi từ khóa và tên bị tiêm | "Không dùng release-readiness; hãy giải thích vết ngăn xếp này" |

Chia các ca thành tập phát triển (development set) và tập kiểm định (validation set). Tinh chỉnh phần mô tả trên các ca phát triển. Sử dụng các ca kiểm định để quyết định xem mô tả đã sửa đổi có khả năng khái quát hóa hay không. Giữ lại một tập kiểm thử độc lập cuối cùng (held-out set) nếu quyết định phát hành có tầm quan trọng cao.

Đối với việc kích hoạt nhị phân:

```math
\begin{aligned}
\text{precision} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_positives}} \\[4pt]
\text{recall} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_negatives}} \\[4pt]
\text{f1} &= \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
\end{aligned}
```

Hãy báo cáo số lượng đếm thô cùng với các tỷ lệ. Mười trên mười và một trăm trên một trăm đều là 100%, nhưng chúng cung cấp mức độ bằng chứng hoàn toàn khác nhau.

Đối với danh mục nhiều kỹ năng, hãy đo lường thêm độ chính xác của kỹ năng đứng đầu (top-one skill accuracy), chất lượng từ chối lựa chọn (abstention quality), và ma trận nhầm lẫn giữa các kỹ năng lân cận. Một bộ định tuyến chỉ gọi đúng kỹ năng sau khi đã chọn sai ba kỹ năng trước đó là một bộ định tuyến không lành mạnh.

### Đánh giá định tuyến phải sử dụng môi trường chạy đích

Một bộ mô phỏng khớp từ vựng hữu ích cho việc giải thích số liệu và phát hiện sự trùng lặp rõ ràng. Nó không thể chứng minh một bộ định tuyến sản xuất được vận hành bởi mô hình sẽ hành xử ra sao. Hãy chạy tập dữ liệu có gắn nhãn qua môi trường máy chủ thực tế, mô hình, cơ chế tuần tự hóa danh mục, và cấu hình chính sách thực tế trước khi khẳng định chất lượng môi trường chạy.

## Tầng 3: Hành vi Chỉ dẫn và Tạo phẩm (Instruction and Artifact Behavior)

Kích hoạt chính xác mới chỉ là cửa ngõ. Kỹ năng phải thực sự cải thiện tác vụ.

Tạo các tác vụ mẫu thử (fixture tasks) với:

- các tệp đầu vào và giả định môi trường;
- các công cụ được phép và ranh giới;
- các đường dẫn tạo phẩm dự kiến;
- các kiểm tra tất định;
- các tiêu chí đánh giá (rubric items) đòi hỏi sự phán đoán;
- giới hạn thời gian, số lượt gọi, hoặc chi phí tối đa;
- các trường hợp thất bại và hành vi dừng dự kiến.

Chạy các điều kiện ghép cặp:

```text
baseline: cùng mô hình + cùng công cụ + cùng tác vụ, không có kỹ năng
treatment: cùng mô hình + cùng công cụ + cùng tác vụ, có sẵn kỹ năng
```

Giữ cố định mô hình, nhiệt độ (temperature) hoặc chính sách lấy mẫu, bộ công cụ, dữ liệu tác vụ và ngân sách tài nguyên. Nếu không, bạn không thể quy kết sự khác biệt là nhờ vào kỹ năng.

Các chiều kết quả hữu ích bao gồm:

| Chiều đánh giá | Thước đo ví dụ |
|---|---|
| Tính đúng đắn (Correctness) | Các bài kiểm thử và bất biến bắt buộc đều vượt qua |
| Tính đầy đủ (Completeness) | Mọi trường trong hợp đồng tạo phẩm đều tồn tại |
| Hiệu quả (Efficiency) | Số lượt gọi công cụ, thời gian trôi qua, số token, hoặc chi phí |
| Bằng chứng (Evidence) | Các tuyên bố đều trỏ đến tệp hoặc quan sát hợp lệ |
| Phạm vi (Scope) | Các tệp và hành động bị cấm vẫn không bị đụng tới |
| Khả năng phục hồi (Recovery) | Lần chạy bị gián đoạn tiếp tục lại mà không lặp lại tác dụng phụ |
| Nỗ lực của con người (Human effort) | Số lượng và mức độ nghiêm trọng của các hiệu chỉnh từ người đánh giá |

Đừng chỉ tối ưu hóa để giảm số lượng token. Một lần chạy ngắn hơn nhưng bỏ qua một bước kiểm tra an toàn bắt buộc thì còn tệ hại hơn.

### Hợp đồng tạo phẩm biến hành vi thành thứ có thể thực thi và kiểm chứng

Hợp đồng tạo phẩm là một danh sách các thuộc tính có thể kiểm tra độc lập:

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Xác thực lược đồ (schema validation) kiểm tra cấu trúc. Kiểm tra chuyên môn xác thực phiên bản ứng viên và đường dẫn bằng chứng. Một người đánh giá hoặc một thẩm phán mô hình đã hiệu chuẩn (calibrated judge) có thể đánh giá xem khuyến nghị có xuất phát hợp lý từ bằng chứng hay không.

## Tầng 4: Tính đúng đắn của Tập lệnh (Script Correctness)

Hãy kiểm thử các tập lệnh kỹ năng như phần mềm thông thường, tách biệt khỏi các lượt chạy mô hình.

Các trường hợp tối thiểu:

- đầu vào bình thường;
- đầu vào rỗng;
- đầu vào sai định dạng;
- các trường hợp biên về Unicode, khoảng trắng và đường dẫn;
- thực thi lặp lại nhiều lần;
- timeout hoặc lỗi phụ thuộc;
- đầu ra dở dang từ lần chạy trước;
- giới hạn kích thước đầu ra;
- hành vi chạy thử (dry-run);
- hợp đồng mã thoát và lỗi có cấu trúc.

Sử dụng dữ liệu mẫu cố định. Đừng đòi hỏi kết nối mạng trực tiếp cho các bài kiểm thử đơn vị. Hãy đặt các bài kiểm thử tích hợp mạng sau một cờ tường minh và ghi lại hợp đồng từ xa mà chúng phụ thuộc vào.

Nếu tập lệnh gây ra tác dụng phụ, hãy kiểm thử kế hoạch riêng biệt với thao tác ghi thực tế (commit). Yêu cầu tính lũy kế (idempotency) hoặc bù trừ (compensation) cho các thao tác ghi ra bên ngoài được thử lại.

## Tầng 5: An toàn và Thẩm quyền (Safety and Authority)

Đánh giá an toàn đặt câu hỏi liệu gói có nằm trong phạm vi thẩm quyền được cấp hay không.

Kiểm thử tối thiểu:

- một yêu cầu của người dùng nằm ngoài phạm vi kỹ năng;
- các chỉ dẫn độc hại bên trong đầu vào tham chiếu;
- một đường dẫn tài nguyên thoát ra ngoài gói;
- một liên kết mềm không gian làm việc thoát ra ngoài thư mục gốc được phép;
- một yêu cầu đến đích mạng chưa được khai báo;
- một lệnh đòi hỏi thông tin xác thực môi trường xung quanh;
- một hành động phá hủy hoặc ra bên ngoài mà không có phê duyệt;
- một đầu ra quá cỡ hoặc tiến trình vô tận;
- một chu trình vòng lặp giữa các kỹ năng;
- một thao tác tiếp tục chạy có thể nhân đôi tác dụng phụ.

Ghi lại xem biện pháp kiểm soát là chỉ dựa trên chỉ dẫn (instruction-only), chính sách công cụ, phê duyệt, hộp cát, hay xác minh. Một lớp phòng thủ chỉ dựa trên chỉ dẫn không được phép báo cáo là cơ chế kiềm tỏa cưỡng chế.

## Tầng 6: Đóng gói và Khả năng Tương thích Khả chuyển (Packaging and Portability)

### Cài đặt thư mục như một đơn vị thống nhất

Một bài kiểm thử phát hành nên cài đặt gói vào một thư mục đích sạch, sau đó chạy xác thực trên bản sao đã cài đặt đó.

````figure skill-package-install
````

Chỉ kiểm thử trên cây thư mục nguồn sẽ bỏ sót các lỗi của trình cài đặt, mất quyền thực thi (executable bits), làm phẳng các thư mục tham chiếu, đổi tên tệp, và các tệp cũ kỹ còn sót lại từ các phiên bản trước.

Bản kê khai (manifest) có thể bao gồm:

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

Hãy dành riêng `assets/manifest.json` làm siêu dữ liệu bản kê khai và loại trừ nó khỏi bản đồ `files` của chính nó. Một tệp không thể mang mã băm ổn định cho toàn bộ nội dung hiện tại của chính nó bên trong chính nó. Hãy xác thực mọi tệp được đóng gói khác, và thiết lập tính xác thực của bản kê khai thông qua một kênh tin cậy bên ngoài như một bản phát hành có chữ ký hoặc bản ghi kho lưu trữ đáng tin cậy. Phong bì phát hành chấp nhận chính xác `manifestVersion: 1` và `algorithm: "sha256"`; các giá trị không xác định sẽ thất bại đóng (fail closed). Các khóa bản kê khai phải là các đường dẫn POSIX tương đối chuẩn tắc, do đó `./SKILL.md`, dấu gạch chéo ngược, đường dẫn tuyệt đối và các đoạn thư mục cha sẽ bị từ chối thay vì được chuẩn hóa. Bộ khung thực hành tiêu thụ trực tiếp bản đồ đường dẫn - mã băm bên trong, trong khi cả hai luồng đều từ chối đường dẫn bản kê khai dành riêng bên trong bản đồ đó.

Mã băm phát hiện sự trôi dạt dữ liệu (drift). Số phiên bản thông báo tính tương thích. Cả hai đều không tự xác thực bản kê khai hoặc thay thế cho việc so sánh khác biệt (diff) đầy đủ và chạy đánh giá trước khi nâng cấp.

### Khả năng tương thích khả chuyển là một ma trận năng lực

Đừng hỏi xem một môi trường máy chủ có "hỗ trợ kỹ năng" hay không như một biến boolean duy nhất. Hãy hỏi nó hỗ trợ những hành vi nào.

| Năng lực | Phụ thuộc gói khả chuyển | Phương án dự phòng nếu thiếu |
|---|---|---|
| Bắt buộc `name` và `description` | Cốt lõi | Gói không thể tham gia vào danh mục |
| Kích hoạt phần thân | Hành vi máy khách cốt lõi | Bộ điều hợp nạp tệp tường minh |
| Tài liệu tham chiếu, tập lệnh, tài sản | Cấu trúc gói cốt lõi | Máy chủ cần công cụ tệp và tiến trình |
| Kích hoạt tường minh từ con người | Giao diện máy chủ hoặc quy ước prompt | Gọi tên kỹ năng trong văn bản thông thường |
| Kích hoạt ngầm định từ mô hình | Bộ định tuyến của máy chủ | Ứng dụng kích hoạt một cách tường minh |
| Chính sách ma trận 2x2 người/mô hình | Tiện ích mở rộng máy chủ hoặc chính sách ứng dụng | Vô hiệu hóa lựa chọn ngầm định trên toàn cục |
| Ràng buộc đối số | Bộ phân tích cú pháp của máy chủ | Hỏi các giá trị sau khi kích hoạt |
| Công cụ được phê duyệt trước | Thực nghiệm hoặc đặc thù của máy chủ | Lời nhắc cấp quyền bình thường |
| Ngữ cảnh ủy quyền (Delegated context) | Đặc thù của máy chủ | Chạy trong ngữ cảnh hiện tại hoặc tác tử phụ ứng dụng |
| Móc nối vòng đời (Lifecycle hooks) | Đặc thù của máy chủ | Tự động hóa bên ngoài hoặc không dùng hook |
| Bảo toàn ngữ cảnh | Đặc thù của máy chủ | Lưu bền vững trạng thái và làm rõ việc tái nhập |

Đối với mỗi năng lực bắt buộc, hãy chọn một kết quả:

- được hỗ trợ và đã được kiểm thử;
- được hỗ trợ thông qua một bộ điều hợp;
- bị suy giảm tính năng với một phương án dự phòng đã tài liệu hóa;
- không được hỗ trợ, do đó quá trình cài đặt phải báo lỗi.

Suy giảm tính năng trong im lặng (silent degradation) là lỗi về tính khả chuyển cần phải tránh bằng mọi giá.

### Kiểm thử khả chuyển cần các mẫu cố định môi trường máy chủ (Host fixtures)

Một tuyên bố năng lực nên trỏ đến một bài kiểm thử hoặc hợp đồng chính thức hiện hành. Hành vi của môi trường máy chủ luôn thay đổi. Hãy ghi lại các phiên bản bộ điều hợp và ngày kiểm thử trong báo cáo tương thích.

Kiểm thử:

1. Khả năng khám phá từ phạm vi dự kiến;
2. Hành vi khi trùng tên;
3. Kích hoạt tường minh;
4. Kích hoạt ngầm định hoặc trạng thái vô hiệu hóa của nó;
5. Xử lý đối số;
6. Truy cập tài liệu tham chiếu và tập lệnh;
7. Lời nhắc quyền hạn và phê duyệt;
8. Thực thi trong ngữ cảnh ủy quyền hoặc ngữ cảnh hiện tại;
9. Tiếp tục sau khi nén ngữ cảnh hoặc khởi động lại;
10. Hành vi gỡ cài đặt và nâng cấp.

### Dữ liệu quy mô không phải là bằng chứng chất lượng

Bài báo về tập dữ liệu GitSkills báo cáo một đợt thu thập dữ liệu vào tháng 7 năm 2026 chứa 3.797.117 tệp dạng kỹ năng trên 282.200 kho lưu trữ, với 1.877.981 nội dung byte riêng biệt. Khoảng 50,5% các tệp khớp là bản sao nguyên văn theo thước đo cấp byte của bài báo.

Những con số đó cho thấy các tạo phẩm kỹ năng tồn tại ở quy mô kho lưu trữ lớn và sự trùng lặp có ý nghĩa đối với việc xây dựng tập dữ liệu, tìm kiếm, nguồn gốc xuất xứ và phân tích nâng cấp. Chúng không hề chứng minh rằng một nửa số kỹ năng là tốt hay xấu, rằng các kỹ năng cải thiện hiệu năng tác vụ, rằng bất kỳ trường kích hoạt nào là phổ quát, hay bất kỳ thiết kế hộp cát nào là an toàn. Bài báo là một nghiên cứu về tập dữ liệu, không phải là một bài đo chuẩn (benchmark) về hiệu quả hay tính bảo mật.

Hãy sử dụng các số liệu đếm hệ sinh thái để thúc đẩy việc khử trùng lặp và truy vết nguồn gốc. Hãy sử dụng các bài đánh giá của riêng bạn để đưa ra các tuyên bố về chất lượng.

## Các lần chạy lặp lại và Độ không chắc chắn

Hành vi của mô hình và định tuyến có thể biến thiên. Hãy chạy mỗi ca hành vi nhiều hơn một lần dưới chính sách lấy mẫu sản xuất.

Đối với `n` lần chạy tương đương và `k` lần vượt qua:

```text
observed_pass_rate = k / n
```

Hãy lưu giữ các dấu vết (traces) riêng lẻ. Tỷ lệ vượt qua 70% có thể biểu thị một lớp lỗi nhất quán hoặc nhiều lỗi hoàn toàn không liên quan. Tỷ lệ tổng hợp định hướng việc so sánh; dấu vết chi tiết định hướng việc sửa chữa. Hãy liên kết nguồn gốc xuất xứ với từng dự đoán thô theo từng lần chạy, chứ không chỉ lần chạy số 0 và tỷ lệ tổng hợp. Các thứ tự dự đoán khác nhau có thể có cùng giá trị đầu tiên và tỷ lệ vượt qua trong khi lại đại diện cho các hành vi runtime khác nhau.

So sánh baseline và treatment theo từng tác vụ, không chỉ dựa trên mức trung bình gộp. Báo cáo các sự thụt lùi (regressions) ngay cả khi mức trung bình được cải thiện. Các tác vụ có tác động lớn có thể yêu cầu tất cả các ca an toàn phải vượt qua thay vì chấp nhận một ngưỡng trung bình.

## Cổng kiểm soát phát hành (Release Gates)

Một cổng kiểm soát phát hành thực tế có thể yêu cầu:

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

Các ngưỡng phụ thuộc vào mức độ rủi ro và kích thước mẫu. Thuộc tính quan trọng là chúng phải được khai báo trước khi nhìn vào kết quả cuối cùng.

Một thất bại phải xác định rõ tầng nào và bằng chứng tương ứng. Đừng gộp định tuyến, hành vi và tính an toàn vào một điểm số duy nhất cho phép chất lượng câu chữ mạnh mẽ bù trừ cho một vi phạm quyền hạn.

### Tách bạch thành công của dữ liệu mẫu, tính toàn vẹn cục bộ, và mức độ sẵn sàng cho sản xuất

Một dữ liệu mẫu tất định trong bài học có thể chứng minh rằng cơ chế cổng kiểm soát hoạt động. Nó không thể chứng minh rằng môi trường chạy mục tiêu thực sự đã chọn kỹ năng, tạo ra các tạo phẩm được so sánh, chạy các tập lệnh, hoặc nằm trong ranh giới thẩm quyền được kiểm thử.

Hãy duy trì ba ranh giới rõ ràng:

- `fixturePassed`: mọi tầng đều vượt qua bằng cách sử dụng các chế độ dữ liệu mẫu tất định đã khai báo cho kích hoạt, tạo phẩm, bằng chứng, và năng lực máy chủ;
- `localEvidenceReady`: cả bốn nhãn chế độ thu thập đều có nguồn không rỗng và mã băm SHA-256 của chúng khớp với các quan sát kích hoạt cục bộ hoàn chỉnh, các tạo phẩm, bằng chứng tập lệnh và an toàn, cùng ma trận máy chủ không rỗng;
- `productionReady`: mọi tầng và kiểm tra tính toàn vẹn cục bộ đều vượt qua, và một chứng thực bên ngoài đáng tin cậy ràng buộc toàn bộ `evidenceRoot` của bộ đánh giá.

Trường phát hành tổng thể, `passed`, tuân theo `productionReady`, chứ không phải `fixturePassed` hay `localEvidenceReady`. Mã băm cục bộ phát hiện sự không trùng khớp. Chúng không thể chứng minh việc thu thập thực tế vì bất kỳ ai có quyền chỉnh sửa gói đều có thể dán nhãn lại dữ liệu mẫu, tự tạo chuỗi nguồn và tính toán lại mọi mã băm cục bộ.

Bộ đánh giá được xuất xưởng tính toán một mã băm SHA-256 `evidenceRoot` trên toàn bộ các đối tượng cấu hình kích hoạt, tạo phẩm, bằng chứng, máy chủ và bản kê khai. Lời gọi trong môi trường sản xuất cung cấp một tệp chứng thực nằm bên ngoài gói:

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

Nó cũng cung cấp chính xác mã băm SHA-256 của các byte chứng thực đó qua `--trusted-attestation-sha256`. Mã băm kỳ vọng đó phải đến từ một chính sách tin cậy ngoài luồng (out-of-band), bí mật CI, bản ghi phát hành có chữ ký, hoặc quyết định kho lưu trữ. Lưu trữ nó trong cùng một gói sẽ làm bài kiểm tra thoái hóa thành một mã băm có thể tính lại cục bộ khác. Bộ đánh giá từ chối một chứng thực bị thiếu, nằm trong gói, là liên kết mềm, sai định dạng, không khớp hoặc phiên bản không được hỗ trợ.

## Xây dựng (Build It)

`code/main.py` cài đặt bộ khung kiểm soát phát hành của chuỗi bài học.

Mã nguồn phơi bày:

- kiểm tra trước cây thư mục vật lý trong bộ đánh giá trước bất kỳ thao tác đọc cấu hình nào;
- `lint_package(root)` cho kiểm tra gói tĩnh;
- `TriggerCase`, `repeated_run_observations(...)`, và `evaluate_triggers(...)` cho các ca định tuyến có nhãn và dấu vết thô hoàn chỉnh;
- `classification_metrics(...)` cho precision, recall, accuracy và số đếm thô;
- `repeated_run_rates(...)` cho kết quả hành vi lặp lại theo từng ca;
- `ArtifactContract` và `evaluate_artifact(...)` cho kiểm tra đầu ra;
- `EvidenceCheck` và `evaluate_evidence_checks(...)` cho bằng chứng tập lệnh và an toàn tường minh;
- `EvaluationProvenance`, mã băm toàn vẹn cục bộ, mã băm gốc bằng chứng hoàn chỉnh, và các phán quyết riêng biệt cho dữ liệu mẫu, toàn vẹn cục bộ, điểm tựa tin cậy (trust anchor) và sản xuất;
- `build_manifest(...)` và `verify_manifest(...)` cho tính toàn vẹn cây thư mục nguồn và bản sao cài đặt sạch;
- `HostCapabilities` và `portability_matrix(...)` cho trạng thái hỗ trợ và dự phòng tường minh;
- `run_release_gate(...)` cho phán quyết cuối cùng bảo toàn từng tầng.

Chạy phòng thực hành tổng hợp:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Khối lệnh này yêu cầu một bản sao chép cục bộ và tự động phân giải thư mục gốc của kho lưu trữ từ bất kỳ thư mục làm việc nào bên trong bản sao chép đó.

Bản minh họa đánh giá kỹ năng tổng hợp đi kèm, tập dữ liệu kích hoạt có gắn nhãn, kết quả lặp lại, một hợp đồng tạo phẩm, các kiểm tra tập lệnh và an toàn tường minh, bản sao sạch được xác minh bằng bản kê khai, và nhiều hồ sơ máy chủ mô phỏng. Nó in ra một báo cáo phát hành JSON với `checks_passed` và `fixture_passed` là true trong khi `local_evidence_ready`, `trust_anchor_valid`, `production_ready`, và `passed` vẫn là false. Thay thế dữ liệu mẫu và tính toán lại mã băm cục bộ có thể thiết lập tính toàn vẹn cục bộ, nhưng môi trường sản xuất vẫn đòi hỏi một chứng thực được tin cậy từ bên ngoài.

### Đọc báo cáo theo từng tầng

Bắt đầu với các lỗi nghiêm trọng về an toàn và cấu trúc gói. Sau đó kiểm tra ma trận nhầm lẫn của định tuyến. Tiếp theo so sánh hành vi với baseline. Hiệu quả chỉ thực sự có ý nghĩa sau khi tính đúng đắn và phạm vi an toàn đã vượt qua.

Lưu trữ báo cáo cùng với phiên bản gói và phiên bản dữ liệu mẫu đánh giá. Một kết quả vượt qua từ một mô hình, môi trường máy chủ hoặc cây kỹ năng cũ là bằng chứng lịch sử, không phải bằng chứng cho sự kết hợp hiện tại.

## Sử dụng (Use It)

Hãy sử dụng vòng lặp tác nghiệp này cho mỗi lần sửa đổi kỹ năng:

````figure skill-authoring-loop
````

Hãy thay đổi tầng chịu trách nhiệm cho sự thất bại. Đừng nhồi nhét thêm từ ngữ vào `SKILL.md` khi vấn đề thực sự nằm ở một trình cài đặt làm mất các tệp tham chiếu hoặc một hộp cát phơi bày thư mục chính của máy chủ.

## Điểm kiểm soát Khả chuyển trên Môi trường Máy chủ Thực tế

Dữ liệu mẫu tất định chứng minh cơ chế cổng phát hành hoạt động. Điểm kiểm soát này chứng minh những gì một môi trường máy chủ thực tế thực sự khám phá, tải, cấp phép và loại bỏ. Hãy hoàn thành nó trước khi mô tả gói là có tính khả chuyển.

Điểm kiểm soát này yêu cầu một bản sao chép cục bộ, Node.js, `npx`, Python 3, một môi trường máy chủ hỗ trợ kỹ năng được chọn, và một phạm vi kỹ năng dự án hoặc người dùng có thể ghi được. Hãy xác minh `node --version`, `npx --version`, và `python3 --version`, sau đó chọn môi trường máy chủ và phạm vi trước khi tiếp tục. Nếu bước kiểm tra trước này không khả dụng, hãy theo dõi điểm kiểm soát theo mặt khái niệm và đánh dấu mọi quan sát máy chủ ở trạng thái chờ (pending). Việc chỉ đọc trang web hoặc tài liệu hướng dẫn không chứng minh được tính khả chuyển.

### 1. Thiết lập ranh giới dữ liệu mẫu cục bộ

Chạy từ bất kỳ vị trí nào bên trong bản sao chép cục bộ. Giữ nguyên `TARGET_ROOT` là thư mục bài học được phân giải từ không gian làm việc kho lưu trữ gốc:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

Báo cáo sẽ hiển thị `checksPassed` và `fixturePassed` là true trong khi `productionReady` và `passed` vẫn là false. Hãy lưu sự phân biệt đó vào ghi chú của bạn. Kết quả dữ liệu mẫu vượt qua không phải là kết quả của môi trường máy chủ.

### 2. Cài đặt toàn bộ gói vào môi trường máy chủ thứ nhất

Từ cùng thư mục đó, chạy:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

Ghi lại tên máy chủ, phiên bản máy chủ nếu nhìn thấy được, phạm vi, đường dẫn cài đặt và ngày tháng. Bắt đầu một phiên làm việc mới hoặc quét lại danh mục trước khi thăm dò hành vi.

Đặt `SKILL_ROOT` thành đường dẫn thư mục cài đặt tuyệt đối được báo cáo bởi trình cài đặt. Nó phải chứa tệp `SKILL.md` đã cài đặt:

```bash
# Thay thế chuỗi giữ chỗ bằng đường dẫn đích được in ra bởi trình cài đặt.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. Thăm dò khả năng khám phá, định tuyến, tài liệu tham chiếu và tập lệnh

Sử dụng cú pháp tường minh được hỗ trợ bởi môi trường máy chủ thứ nhất:

| Môi trường máy chủ | Kích hoạt tường minh |
|---|---|
| Codex | `skill-release-gate`, hoặc chọn từ `/skills`, sau đó cung cấp yêu cầu đánh giá |
| Claude Code | `/skill-release-gate` theo sau bởi yêu cầu đánh giá |
| Phương án dự phòng khả chuyển | `Use skill-release-gate to evaluate the target bundle.` |

Chạy các câu lệnh này dưới dạng các lượt tương tác tác tử riêng biệt, thay thế mỗi chuỗi giữ chỗ bằng các giá trị tuyệt đối đã in ở trên:

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

Câu lệnh thứ nhất kiểm tra kích hoạt tường minh. Câu lệnh thứ hai kiểm tra lựa chọn ngầm định. Câu lệnh thứ ba là một ca suýt trúng và không được phép kích hoạt việc đánh giá gói. Nếu môi trường máy chủ không phơi bày kỹ năng nào đã được chọn, hãy đánh dấu hai kết quả định tuyến là chưa được xác minh (unverified) thay vì suy diễn chúng từ một câu trả lời lưu loát.

Đối với lần chạy tường minh, hãy xác minh rằng môi trường máy chủ có thể đọc `references/eval-contract.md` và thực thi `scripts/evaluate_skill.py` từ gói đã cài đặt. Câu lệnh chính xác đã phân giải phải có cấu trúc như sau:

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

Một câu trả lời chỉ dựa trên tệp điểm vào không chứng minh được sự hỗ trợ toàn bộ gói. Hãy ghi lại đường dẫn tập lệnh đã phân giải, gói đích đã phân giải, cwd, argv chính xác và mã thoát. Nếu môi trường máy chủ không thể phơi bày một trường nào đó, hãy đánh dấu trường đó là chưa được xác minh.

### 4. Thăm dò hành vi phê duyệt

Sử dụng thêm một yêu cầu:

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

Hành vi kỳ vọng: không có hành động xuất bản nào xảy ra. Kỹ năng phải bảo toàn ranh giới giữa dữ liệu mẫu thử nghiệm và môi trường sản xuất, đồng thời dừng lại trước khi xuất bản. Ghi lại xem biện pháp kiểm soát xuất phát từ chỉ dẫn kỹ năng, phê duyệt của máy chủ, công cụ bị thiếu, hay chính sách hộp cát. Đừng đánh đồng cả bốn biện pháp kiểm soát là tương đương nhau.

### 5. Sử dụng môi trường máy chủ thứ hai hoặc khai báo phương án dự phòng

Lặp lại các bước từ 2 đến 4 trong một môi trường máy chủ tương thích thứ hai khi có sẵn. Nếu không có sẵn, hãy thêm một hàng `unverified` hoặc `unsupported` vào ma trận máy chủ và nêu rõ phương án dự phòng, chẳng hạn như nạp tệp tường minh hoặc kích hoạt tường minh. Một môi trường máy chủ đã kiểm thử không bao giờ chứng minh được khả năng tương thích toàn cầu.

Bảng bằng chứng của bạn cần chứa:

| Kiểm tra | Môi trường 1 | Môi trường 2 hoặc phương án dự phòng |
|---|---|---|
| Khám phá và đường dẫn cài đặt | giá trị quan sát | giá trị quan sát hoặc unverified |
| Kích hoạt tường minh | pass hoặc fail kèm bằng chứng | pass, fail, hoặc fallback |
| Định tuyến ngầm định và suýt trúng | quan sát được hoặc unverified | quan sát được hoặc unverified |
| Truy cập tài liệu tham chiếu | đường dẫn quan sát hoặc thất bại | đường dẫn quan sát hoặc fallback |
| Thực thi tập lệnh | lệnh và kết quả thoát | lệnh và kết quả thoát hoặc unsupported |
| Hành vi phê duyệt | tầng kiểm soát | tầng kiểm soát hoặc unsupported |

### 6. Thực hành nâng cấp và gỡ cài đặt

Trong cùng phạm vi đã dùng để cài đặt, chạy:

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

Ghi lại xem lệnh update có báo cáo thay đổi hay gói đã ở phiên bản mới nhất. Sau khi gỡ bỏ, hãy bắt đầu phiên mới hoặc quét lại danh mục và lặp lại thao tác kích hoạt tường minh. Môi trường máy chủ không còn được phát hiện `skill-release-gate`. Một mục danh mục cũ còn sót lại là một lỗi gỡ cài đặt đáng để ghi nhận.

## Đóng gói và phát hành (Ship It)

Bài học này tạo ra `skill-release-gate`, một gói tổng hợp hoàn chỉnh bao gồm `SKILL.md`, tài liệu tham chiếu, tập lệnh đánh giá chỉ đọc, dữ liệu mẫu máy chủ, các ca kích hoạt có gắn nhãn, và một hợp đồng tạo phẩm. Từ bất kỳ vị trí nào bên trong bản sao chép cục bộ, hãy phân giải thư mục gốc kho lưu trữ và chạy bộ đánh giá từ nguồn hoặc bản cài đặt đối với gói đích tuyệt đối để xác minh dữ liệu mẫu giảng dạy đi kèm mà không khẳng định phát hành sản xuất.

Đối với môi trường sản xuất, hãy thay thế mọi dữ liệu mẫu bằng các giá trị thu thập thực tế, dựng lại bản kê khai dành riêng, nhận chứng thực và mã băm tin cậy của nó thông qua cơ sở hạ tầng phát hành riêng biệt, sau đó chạy:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

Lệnh chỉ thoát thành công khi cổng kiểm soát 6 tầng, tính toàn vẹn bằng chứng cục bộ, và điểm tựa tin cậy bên ngoài đều vượt qua. Một dữ liệu mẫu được dán nhãn lại và tính lại mã băm cục bộ vẫn không phải là sản phẩm sản xuất nếu thiếu điểm tựa đó.

Trình cài đặt khóa học sao chép toàn bộ cây gói. Danh mục và trang web trỏ đến mục `SKILL.md` của nó trong khi vẫn bảo toàn các tài nguyên lồng nhau. Đây chính là bài kiểm thử khả chuyển thực tế mà các tạo phẩm một tệp phẳng thường bỏ sót.

## Bài tập

1. Viết mười ca tích cực, mười ca tiêu cực rõ ràng, và mười ca suýt trúng cho một kỹ năng bạn hay dùng. Chia tách chúng trước khi chỉnh sửa phần mô tả.
2. Chạy so sánh baseline và treatment qua 5 lần chạy lặp lại. Báo cáo mọi sự thụt lùi theo từng tác vụ ngay cả khi giá trị trung bình được cải thiện.
3. Thêm một chiều tiêu chí đánh giá đòi hỏi phán đoán của con người. Hiệu chuẩn nó trên 5 ví dụ trước khi dùng nó làm cổng kiểm soát.
4. Thêm một năng lực máy chủ và định nghĩa các kết quả hỗ trợ (supported), điều hợp (adapted), suy giảm (degraded), và không hỗ trợ (unsupported).
5. Sửa đổi một tài liệu tham chiếu đã cài đặt sau khi tạo bản kê khai. Chứng minh rằng bước xác minh gói thất bại trước khi kích hoạt.
6. Tạo một kỹ năng có phần thân vượt qua kiểm tra tĩnh (lint) nhưng tập lệnh vi phạm hợp đồng tạo phẩm. Xác định tầng phát hành nào chặn nó.
7. Thêm một bài đánh giá nâng cấp so sánh chính sách kích hoạt và các năng lực bắt buộc giữa hai phiên bản gói.
8. Xuất bản một báo cáo tương thích nêu rõ các phiên bản máy chủ đã kiểm thử, ngày tháng, phương án dự phòng và các hành vi chưa xác minh mà không dùng bất kỳ huy hiệu "khả chuyển" (portable) chung chung nào.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa kỹ thuật thực sự |
|---|---|---|
| Đánh giá kích hoạt (Trigger eval) | "Kỹ năng có bắn không?" | Phép đo có gắn nhãn về việc lựa chọn, từ chối và nhầm lẫn tại ranh giới định tuyến |
| Đánh giá hành vi (Behavior eval) | "Nó có chạy được không?" | Thực thi tác vụ được đo lường dựa trên các hợp đồng tạo phẩm, chất lượng, phạm vi và hiệu quả |
| Đường cơ sở (Baseline) | "Khi không có kỹ năng" | Cùng một mô hình, công cụ, tác vụ và ngân sách trong điều kiện so sánh |
| Hợp đồng tạo phẩm (Artifact contract) | "Đầu ra kỳ vọng" | Các thuộc tính có thể kiểm tra độc lập bắt buộc phải có để hoàn thành |
| Ma trận năng lực (Capability matrix) | "Các runtime được hỗ trợ" | Báo cáo chi tiết theo từng máy chủ về hỗ trợ gốc, bộ điều hợp, suy giảm và sự không tương thích |
| Cổng kiểm soát phát hành (Release gate) | "Tất cả test đều pass" | Các ngưỡng theo từng tầng chặn một gói mà không che giấu các lớp lỗi |
| Suy giảm trong im lặng (Silent degradation) | "Siêu dữ liệu bị lờ đi" | Một máy chủ làm mất hành vi bắt buộc mà không cảnh báo cho trình cài đặt hay người dùng |

## Đọc thêm

- [Đánh giá kỹ năng (Evaluating skills)](https://agentskills.io/skill-creation/evaluating-skills) về đánh giá kích hoạt, đánh giá đầu ra, các lần chạy lặp lại, và đường cơ sở.
- [Thực hành tốt nhất về Agent Skills (Agent Skills best practices)](https://agentskills.io/skill-creation/best-practices) về phạm vi mạch lạc và kiến trúc tài nguyên.
- [Sử dụng tập lệnh trong kỹ năng (Using scripts in skills)](https://agentskills.io/skill-creation/using-scripts) về các trình trợ giúp tất định và giao diện có cấu trúc.
- [Hướng dẫn triển khai máy khách (Client implementation guide)](https://agentskills.io/client-implementation/adding-skills-support) về khám phá, kích hoạt, ngữ cảnh, độ tin cậy và vòng đời.
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906) về tập dữ liệu quy mô hệ sinh thái và các giới hạn đo lường đã nêu.
