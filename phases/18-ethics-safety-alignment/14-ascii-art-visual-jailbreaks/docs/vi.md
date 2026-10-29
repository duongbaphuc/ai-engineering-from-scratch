# Nghệ thuật Chữ ASCII và Các Cuộc bẻ khóa Trực quan (ASCII Art and Visual Jailbreaks)

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Che giấu các token liên quan đến an toàn trong một yêu cầu độc hại, thay thế chúng bằng kết xuất dạng nghệ thuật chữ ASCII (ASCII-art) của các chữ cái đó, rồi gửi lời nhắc đã được ngụy trang đi. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 đều thất bại trong việc nhận diện một cách mạnh mẽ các token nghệ thuật chữ ASCII. Cuộc tấn công vượt qua các bộ lọc độ rối (perplexity filters - PPL), các biện pháp phòng vệ diễn giải lại (paraphrase defenses) và tái phân tách token (retokenization). Nghiên cứu liên quan: bộ đo chuẩn ViTC đo lường khả năng nhận diện các lời nhắc trực quan phi ngữ nghĩa; StructuralSleight tổng quát hóa sang các Cấu trúc Mã hóa Văn bản Bất thường (Uncommon Text-Encoded Structures - UTES như cây, đồ thị, JSON lồng nhau) thành một họ các cuộc tấn công mã hóa.

**Type:** Build
**Languages:** Python (thư viện chuẩn, khung che giấu token ArtPrompt)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả cuộc tấn công ArtPrompt: bước nhận diện từ, bước thay thế bằng nghệ thuật chữ ASCII, và lời nhắc ngụy trang cuối cùng.
- Giải thích lý do tại sao các biện pháp phòng thủ tiêu chuẩn (PPL, Diễn giải lại, Tái phân tách token) thất bại trước ArtPrompt.
- Định nghĩa ViTC và mô tả những gì nó đo lường.
- Mô tả StructuralSleight như một bước tổng quát hóa cho các Cấu trúc Mã hóa Văn bản Bất thường (Uncommon Text-Encoded Structures) tùy ý.

## Vấn đề

Các cuộc tấn công thông qua diễn giải lại và đóng vai (Bài 12) cũng như thông qua ngữ cảnh dài (Bài 13) hoạt động dựa trên mô thức ở cấp độ văn bản. ArtPrompt hoạt động ở cấp độ nhận diện: mô hình không phân tích cú pháp token bị cấm. Nó phân tích cú pháp một hình ảnh được kết xuất bằng các ký tự. Bộ lọc an toàn chỉ nhìn thấy các dấu câu vô hại. Nhưng mô hình lại nhìn thấy một từ ngữ hoàn chỉnh.

## Khái niệm

### ArtPrompt, hai bước

Bước 1. Nhận diện từ (Word Identification). Với một yêu cầu độc hại cho trước, kẻ tấn công sử dụng một LLM để xác định các từ ngữ liên quan đến an toàn (ví dụ: từ "bomb" trong "làm thế nào để chế tạo bom").

Bước 2. Tạo lời nhắc ngụy trang (Cloaked Prompt Generation). Thay thế từng từ đã xác định bằng kết xuất nghệ thuật chữ ASCII tương ứng (một khối ký tự kích thước 7x5 hoặc 7x7 tạo thành hình dạng chữ cái). Mô hình tiếp nhận một lưới gồm các dấu câu và khoảng trắng mà một mô hình đủ năng lực có thể nhận diện ra từ ngữ đó; trong khi một bộ lọc an toàn chỉ nhìn thấy một lưới ký tự đơn thuần.

Kết quả: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 đều thất bại. Tỷ lệ tấn công thành công đạt trên 75% trên tập con thử nghiệm chuẩn của họ.

### Tại sao các biện pháp phòng thủ tiêu chuẩn thất bại

- **PPL (bộ lọc độ rối - perplexity filter).** Nghệ thuật chữ ASCII có độ rối cao — nhưng mọi đầu vào mới lạ cũng vậy. Việc lựa chọn ngưỡng để chặn ArtPrompt cũng sẽ chặn luôn cả các đầu vào có cấu trúc hợp lệ.
- **Diễn giải lại (Paraphrase).** Việc diễn giải lại lời nhắc sẽ phá hủy cấu trúc ASCII. Tuy nhiên trên thực tế, các LLM làm nhiệm vụ diễn giải lại thường bảo toàn hoặc tái tạo lại hình vẽ ASCII đó.
- **Tái phân tách token (Retokenization).** Việc chia tách các token theo cách khác không làm thay đổi sự thật rằng thị giác/khả năng nhận diện của mô hình đang nhận diện các hình dạng chữ cái.

Vấn đề cốt lõi là các bộ lọc an toàn hoạt động ở cấp độ token hoặc ngữ nghĩa; trong khi ArtPrompt lại hoạt động ở cấp độ nhận diện trực quan.

### Bộ đo chuẩn ViTC

Nhận diện các lời nhắc trực quan phi ngữ nghĩa. Đo lường khả năng của mô hình trong việc đọc nghệ thuật chữ ASCII, biểu tượng wingdings và các nội dung trực quan phi ngữ nghĩa văn bản khác. Hiệu quả của ArtPrompt tương quan thuận với độ chính xác trên ViTC: mô hình đọc văn bản trực quan càng giỏi thì ArtPrompt càng hoạt động hiệu quả trên mô hình đó. Đây là sự đánh đổi giữa năng lực và tính an toàn (capability-safety tradeoff).

### StructuralSleight

Tổng quát hóa ArtPrompt: Cấu trúc Mã hóa Văn bản Bất thường (Uncommon Text-Encoded Structures - UTES). Cây, đồ thị, JSON lồng nhau, CSV nhúng trong JSON, các khối mã kiểu diff. Nếu một cấu trúc hiếm khi xuất hiện trong dữ liệu huấn luyện an toàn nhưng mô hình vẫn có thể phân tích cú pháp được, nó có thể được dùng để che giấu nội dung độc hại.

Hàm ý đối với phòng thủ: tính an toàn phải tổng quát hóa được trên toàn bộ các biểu diễn có cấu trúc mà mô hình có khả năng phân tích cú pháp. Tập hợp này rất lớn và không ngừng tăng lên.

### Tương đồng với phương thức hình ảnh

Các LLM đa phương thức thị giác (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) mở rộng bề mặt tấn công. Các cuộc tấn công kiểu ArtPrompt với hình ảnh thực tế thậm chí còn mạnh hơn các biến thể tương đương bằng chữ ASCII vì bộ mã hóa hình ảnh (image encoder) tạo ra tín hiệu phong phú hơn nhiều.

### Vị trí của bài học này trong Phase 18

Các Bài 12-14 mô tả ba véc-tơ tấn công trực giao: tinh chỉnh lặp (PAIR), độ dài ngữ cảnh (MSJ), và mã hóa (ArtPrompt/StructuralSleight). Bài 15 chuyển từ các cuộc tấn công tập trung vào mô hình sang các cuộc tấn công tại ranh giới hệ thống (tiêm nhiễm chỉ dẫn gián tiếp). Bài 16 mô tả phản ứng từ các công cụ phòng thủ.

```figure
al-ascii-cloak
```

## Sử dụng trong thực tế (Use It)

`code/main.py` xây dựng một phiên bản ArtPrompt đồ chơi. Bạn có thể ngụy trang các từ cụ thể trong một truy vấn độc hại bằng các ký tự nghệ thuật chữ ASCII, xác minh chuỗi đã ngụy trang vượt qua bộ lọc từ khóa, và (tùy chọn) giải mã chuỗi đã ngụy trang ngược trở lại bằng một bộ nhận diện đơn giản.

## Đóng gói chuyển giao (Ship It)

Bài học này tạo ra `outputs/skill-encoding-audit.md`. Khi được cung cấp một báo cáo phòng thủ bẻ khóa, nó liệt kê các họ tấn công mã hóa được bao phủ (nghệ thuật chữ ASCII, base64, leet-speak, chữ đồng hình UTF-8 homoglyph, UTES) và tầng phòng thủ bắt giữ từng loại.

## Bài tập

1. Chạy `code/main.py`. Xác minh chuỗi đã ngụy trang vượt qua bộ lọc từ khóa đơn giản. Báo cáo mức độ thay đổi ở cấp ký tự được yêu cầu.

2. Cài đặt phương thức mã hóa thứ hai: base64 cho cùng một từ mục tiêu. So sánh tỷ lệ vượt qua bộ lọc so với ArtPrompt và độ khó khôi phục lại từ ngữ.

3. Đọc Mục 4.3 trong bài báo của Jiang và cộng sự 2024 (kết quả trên năm mô hình). Đề xuất một lý do giải thích tại sao khả năng kháng ArtPrompt của Claude lại cao hơn Gemini trên cùng một bài đo chuẩn.

4. Thiết kế một cơ chế phòng thủ tiền sinh (pre-generation defense) để phát hiện các vùng có hình dạng nghệ thuật chữ ASCII trong lời nhắc. Đo lường tỷ lệ dương tính giả (false-positive rate) trên mã nguồn hợp lệ, bảng biểu và ký hiệu toán học.

5. StructuralSleight liệt kê 10 cấu trúc mã hóa. Phác thảo một cơ chế phòng thủ tổng quát hóa xử lý cả 10 cấu trúc đó và ước tính chi phí tính toán cho mỗi lời nhắc được phòng thủ.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| ArtPrompt | "cuộc tấn công bằng chữ ASCII" | Cuộc bẻ khóa hai bước che giấu các từ nhạy cảm an toàn bằng kết xuất nghệ thuật chữ ASCII |
| Ngụy trang (Cloaking) | "giấu từ đi" | Thay thế một token bị cấm bằng một biểu diễn trực quan mà mô hình đọc được nhưng bộ lọc thì không |
| UTES | "cấu trúc bất thường" | Cấu trúc Mã hóa Văn bản Bất thường (Uncommon Text-Encoded Structure) — cây, đồ thị, JSON lồng nhau, v.v. dùng để buôn lậu nội dung |
| ViTC | "năng lực đọc văn bản trực quan" | Bộ đo chuẩn đánh giá khả năng đọc các mã hóa trực quan phi ngữ nghĩa của mô hình |
| Bộ lọc độ rối (Perplexity filter) | "phòng thủ PPL" | Từ chối các lời nhắc có độ rối cao; thất bại vì đầu vào có cấu trúc hợp lệ cũng đạt điểm cao |
| Tái phân tách token (Retokenization) | "phòng thủ dịch chuyển tokenizer" | Tiền xử lý lời nhắc bằng một tokenizer khác; thất bại vì cơ chế nhận diện mang tính trực quan |
| Ký tự đồng hình (Homoglyph) | "các ký tự trông giống nhau" | Các ký tự Unicode trông giống hệt các chữ cái Latin; vượt qua kiểm tra chuỗi con |

## Đọc thêm

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — bài báo về bẻ khóa bằng nghệ thuật chữ ASCII
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — bước tổng quát hóa UTES
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — cuộc tấn công lặp bổ trợ
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — cuộc tấn công độ dài ngữ cảnh bổ trợ
