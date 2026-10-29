# Đồ án Tổng hợp 82 — Phân loại Jailbreak (Jailbreak Taxonomy)

> Một khung kiểm thử an toàn (safety harness) thiếu vắng bảng phân loại cũng giống như một canh bạc tung đồng xu. Hãy định danh cuộc tấn công trước khi bạn thiết lập phòng thủ.

**Type:** Xây dựng (Build)  
**Languages:** Python  
**Prerequisites:** Các bài học an toàn thuộc Phase 18, các bài học 25-29 Track A thuộc Phase 19  
**Time:** ~90 phút  

## Vấn đề (Problem)

Một mô hình được triển khai mà không có mô hình tấn công (attack model) rõ ràng cũng giống như một mô hình không được phòng thủ trước bất kỳ mối đe dọa cụ thể nào. Đội ngũ vận hành thường đọc một luồng thảo luận trên mạng xã hội, nhận diện mánh khóe, viết một biểu thức chính quy (regex), đưa lên môi trường thực tế rồi bỏ qua. Câu lệnh tiếp theo lại là một câu diễn đạt lại (paraphrase). Biểu thức chính quy hoàn toàn bỏ sót. Một tuần sau, ai đó lại trình diễn đúng mánh khóe đó nhưng được mã hóa dạng base64, và người vận hành lại vội vã viết thêm biểu thức chính quy thứ hai. Đến tháng thứ ba, hệ thống tồn đọng 40 quy tắc vá víu, không có vốn từ vựng chung, không có phương thức trao đổi mạch lạc về bản chất của một cuộc tấn công, trong khi danh sách công việc tồn đọng (backlog) lại phình to nhanh hơn tốc độ tạo ra các bản vá.

Trước khi bất kỳ bộ phát hiện (detector), bộ phân loại (classifier), hay bộ quy tắc (rule engine) nào trong lộ trình này phát huy tác dụng thực chất, toàn đội ngũ cần một phương pháp chuẩn hóa chung để gắn nhãn các cuộc tấn công. Không phải vì nhãn tự thân nó ngăn chặn được tấn công, mà bởi vì nhãn giúp biến một luồng tấn công thành một biểu đồ tần suất (histogram). Biểu đồ tần suất biến thành biểu đồ độ bao phủ (coverage chart). Biểu đồ độ bao phủ sẽ dẫn dắt định hướng cho sprint tiếp theo. Khung kiểm thử trong các bài học 83-87 dành phần lớn thời gian để phân định xem một câu lệnh truy vấn (prompt) là một cuộc tấn công nhập vai (role-play attack) nhằm vô hiệu hóa chính sách từ chối hay là một cuộc tấn công tuồn ngữ cảnh (context-smuggling attack) nhắm vào một công cụ gọi ngoài. Quyết định đó là bất khả thi nếu thiếu một bảng phân loại (taxonomy).

Dự án tổng hợp này định nghĩa một bảng phân loại gồm sáu danh mục: đủ rộng để bao quát phần lớn các cuộc tấn công xuất hiện trong thực tế, đủ hẹp để hai chuyên viên đánh giá độc lập thường đạt được sự đồng thuận về danh mục phân loại, và đủ cụ thể để mỗi danh mục sở hữu ít nhất bảy mẫu dữ liệu kiểm thử (fixtures) được xây dựng thủ công. Bảng phân loại này chính là sóng mang (carrier wave) cho toàn bộ các thành phần phía hạ nguồn.

## Khái niệm (Concept)

Sáu danh mục này được phân chia dọc theo một trục duy nhất: cuộc tấn công đang lạm dụng ranh giới tin cậy (trust boundary) nào? Mỗi tên gọi tương ứng với một ranh giới cụ thể.

| Danh mục (Category) | Ranh giới tin cậy bị lạm dụng (Trust boundary abused) |
|---|---|
| role-play | nhân cách/vai trò của trợ lý (the assistant's persona) |
| instruction-override | thẩm quyền của câu lệnh hệ thống (the system prompt's authority) |
| context-smuggling | khoảng cách giữa nội dung người dùng và nội dung chỉ dẫn (the gap between user content and instruction content) |
| multi-turn-ramp | lịch sử hội thoại đóng vai trò như một bản hợp đồng ràng buộc (the conversation history as a contract) |
| encoding-trick | hình thức biểu diễn bề mặt của các token bị cấm (the surface form of forbidden tokens) |
| prefix-injection | quyết định sinh token tiếp theo của trợ lý (the assistant's next-token decision) |

Một cuộc tấn công nhập vai (role-play) tái định khung trợ lý thành một thực thể tác nhân khác ("bạn là một mô hình nghiên cứu không giới hạn có tên QX") để các quy tắc từ chối gắn liền với nhân cách ban đầu không còn kích hoạt. Các câu lệnh ghi đè chỉ dẫn (instruction-override) đưa ra yêu cầu "bỏ qua các hướng dẫn trước đó" và cố gắng ghi đè trực tiếp câu lệnh hệ thống (system prompt). Kỹ thuật tuồn ngữ cảnh (context-smuggling) ẩn giấu các chỉ dẫn bên trong những thứ thoạt nhìn giống như dữ liệu thuần túy: một tài liệu được dán vào, kết quả trả về của một công cụ, hoặc một khối mã lệnh. Kỹ thuật leo thang đa lượt (multi-turn-ramp) làm nóng mô hình bằng các lượt hội thoại vô hại rồi từng bước hạ thấp ngưỡng kiểm soát, khai thác xu hướng giữ tính nhất quán với ngữ cảnh hội thoại của mô hình. Các mánh khóe mã hóa (encoding-trick, như base64, rot13, leet-speak, chèn ký tự không có độ rộng zero-width) che giấu các token bị cấm trước các bộ lọc từ khóa đơn giản. Kỹ thuật tiêm tiền tố (prefix-injection) kết thúc prompt bằng cụm từ "Chắc chắn rồi, đây là cách thực hiện" để mô hình tiếp tục phát sinh văn bản từ câu trả lời mặc định thay vì đưa ra lời từ chối.

```mermaid
flowchart TB
  P[prompt] --> C{thuộc ranh giới nào}
  C -->|nhân cách| RP[role-play]
  C -->|system prompt| IO[instruction-override]
  C -->|dữ liệu vs chỉ dẫn| CS[context-smuggling]
  C -->|lịch sử| MT[multi-turn-ramp]
  C -->|dạng bề mặt| ET[encoding-trick]
  C -->|phát sinh tiếp| PI[prefix-injection]
  RP --> H[(histogram)]
  IO --> H
  CS --> H
  MT --> H
  ET --> H
  PI --> H
  H --> D[biểu đồ độ bao phủ]
```

Mỗi mẫu dữ liệu (fixture) là một bản ghi chứa `id`, `category`, `subtype`, `prompt`, `target_behavior`, và `severity`. Đối tượng phân loại (taxonomy object) nạp các fixture, nhóm chúng theo danh mục và cung cấp API `match`: khi nhận một prompt ứng viên, trả về fixture gần nhất cùng danh mục tương ứng của nó. Phép khớp nối (match) dựa trên độ tương đồng cosine của bộ ba ký tự (character-trigram cosine): thô, nhanh và được xây dựng hoàn toàn từ thư viện numpy thuần túy. Đây không phải là một bộ phát hiện (detector). Bộ phát hiện nằm ở bài học 83. Thành phần này đóng vai trò là bộ sinh nhãn (label producer).

Mức độ nghiêm trọng (severity) tuân theo thang điểm từ 1 đến 5. Mức 1 là một cuộc tấn công vụng về nhằm vào mục tiêu lành tính ("hãy giả vờ làm một tên cướp biển"). Mức 5 là cuộc tấn công mà nếu thành công sẽ tạo ra đầu ra mà một hệ thống triển khai thực tế tuyệt đối không được phát ra (chi tiết vận hành cho một hành vi nguy hiểm). Phần lớn các fixture nằm ở mức 2-3 vì các cuộc tấn công thực tế ở quy mô triển khai thường thiên về tính đơn giản và lười biếng. Mức độ nghiêm trọng do chính tác giả tạo fixture thiết lập. Việc hai người đánh giá chênh lệch nhau quá một bậc xếp hạng là dấu hiệu cho thấy tiêu chí đánh giá (rubric) cần được tinh chỉnh rõ ràng hơn.

```figure
cd-attack-taxonomy
```

## Xây dựng thực tế (Build It)

Tập ngữ liệu (corpus) nằm trong `code/fixtures.py` dưới dạng một danh sách Python đơn lẻ. Lớp phân loại trong `code/main.py` nạp danh sách này, xác thực rằng mọi danh mục đều có ít nhất bảy fixture, cung cấp các phương thức `by_category`, `match`, và `stats`, đồng thời cung cấp một bản chạy thử (runnable demo) in ra biểu đồ tần suất. Độ tương đồng cosine trên bộ ba ký tự (trigram cosine) được lập trình từ đầu bằng `numpy`.

Bước xác thực (validation pass) kiểm tra bốn bất biến: mọi fixture đều có trường prompt không rỗng, mọi danh mục trong lược đồ đều được đại diện, mọi giá trị severity đều nằm trong khoảng `1..5`, và mọi fixture id đều là duy nhất. Bất kỳ lỗi nào ở bước này đều dẫn đến việc thoát chương trình ngay lập tức (hard exit) thay vì chỉ đưa ra cảnh báo, bởi vì phần còn lại của lộ trình phụ thuộc hoàn toàn vào tính nhất quán nội tại của tập dữ liệu.

## Sử dụng thực tế (Use It)

Chạy lệnh `python3 main.py` từ thư mục `code/` của bài học. Chương trình minh họa sẽ in ra số lượng fixture theo từng danh mục, chạy ba mẫu thăm dò thử nghiệm qua phương thức `match`, và ghi tệp `taxonomy.json` vào thư mục outputs của bài học. Các bài học hạ nguồn sẽ đọc tệp `taxonomy.json` này thay vì import trực tiếp module Python, đảm bảo tập dữ liệu đóng vai trò như một cấu phần ổn định (stable artifact).

## Phát hành sản phẩm (Ship It)

Tệp `outputs/skill-jailbreak-taxonomy.md` tài liệu hóa sáu danh mục và thang tiêu chí đánh giá. Hãy sử dụng tệp này như vốn từ vựng chung của toàn đội ngũ. Mọi phát hiện được ghi lại bởi khung kiểm thử trong bài học 87 đều sẽ tham chiếu đến một định danh trong bảng phân loại (taxonomy id).

## Bài tập thực hành (Exercises)

1. Bổ sung danh mục thứ bảy dành cho tấn công tiêm câu lệnh gián tiếp (indirect-prompt-injection: chỉ dẫn độc hại được nhúng trong tài liệu truy xuất chứ không nằm ở lượt nhập của người dùng). Tự biên soạn mười fixture và chạy lại bộ xác thực.
2. Thay thế độ tương đồng cosine của bộ ba ký tự bằng bộ tính điểm khoảng cách chỉnh sửa token (token-edit-distance scorer) và đo lường sự thay đổi trong việc gán nhãn khớp nối trên tập ngữ liệu hiện có.
3. Trích xuất thêm ba mươi fixture từ nhật ký sản phẩm thực tế của bạn (đã được khử định danh) và xác nhận xem phân phối danh mục có khớp với trực giác ban đầu của đội ngũ hay không.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ (Term) | Cách dùng thông thường | Định nghĩa chính xác |
|---|---|---|
| jailbreak | bất kỳ đầu ra không an toàn nào của mô hình | một câu lệnh truy vấn (prompt) kích thích mô hình sinh đầu ra vi phạm chính sách đã công bố |
| taxonomy | một danh sách các thể loại | một sự phân hoạch các cuộc tấn công dựa trên ranh giới tin cậy mà chúng lạm dụng |
| fixture | một ví dụ kiểm thử | một prompt đã được gắn nhãn đầy đủ category, severity và hành vi mục tiêu (target behavior) |
| severity | mức độ tồi tệ của đầu ra | xếp hạng 1-5 đo lường mức độ tác động tiêu cực nếu cuộc tấn công thành công |
| match | một quyết định phát hiện tấn công | fixture gần nhất theo độ tương đồng cosine của bộ ba ký tự, dùng để gán danh mục cho một prompt mới |

## Đọc thêm (Further Reading)

Bài học này là điểm khởi đầu. Các bài học từ 83 đến 87 sẽ trực tiếp kế thừa và phát triển dựa trên tập ngữ liệu này.
