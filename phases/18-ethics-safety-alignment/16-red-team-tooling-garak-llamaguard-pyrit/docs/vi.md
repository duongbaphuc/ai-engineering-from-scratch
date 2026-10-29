# Công cụ Đội đỏ — Garak, Llama Guard, PyRIT (Red-Team Tooling — Garak, Llama Guard, PyRIT)

> Ba công cụ môi trường sản xuất định hình ngăn xếp đội đỏ (red-team stack) năm 2026. Llama Guard (Meta) — bộ phân loại Llama-3.1-8B được tinh chỉnh (fine-tuned) trên 14 danh mục mối nguy hại của MLCommons; phiên bản Llama Guard 4 năm 2025 là một bộ phân loại đa phương thức gốc 12B được cắt tỉa từ Llama 4 Scout. Garak (NVIDIA) — công cụ quét lỗ hổng LLM mã nguồn mở với các đầu dò tĩnh, động và thích ứng cho ảo giác, rò rỉ dữ liệu, tiêm nhiễm lời nhắc, độc hại và bẻ khóa. PyRIT (Microsoft) — thực hiện các chiến dịch đội đỏ nhiều lượt với Crescendo, TAP và các chuỗi bộ chuyển đổi tùy chỉnh phục vụ việc khai thác sâu. Llama Guard 3 được ghi nhận trong bài báo "Llama 3 Herd of Models" của Meta (arXiv:2407.21783); Llama Guard 3-1B-INT4 trong arXiv:2411.17713; kiến trúc đầu dò của Garak tại github.com/NVIDIA/garak. Những công cụ này tạo thành giao diện sản xuất năm 2026 giữa nghiên cứu đội đỏ (Bài 12-15) và triển khai thực tế (Bài 17 trở đi).

**Type:** Build
**Languages:** Python (thư viện chuẩn, bộ mô phỏng kiến trúc công cụ và mock bộ phân loại kiểu Llama Guard)
**Prerequisites:** Phase 18 · 12-15 (bẻ khóa và IPI)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả vị trí của Llama Guard 3/4 trong ngăn xếp an toàn: bộ phân loại đầu vào, bộ phân loại đầu ra, hoặc cả hai.
- Kể tên 14 danh mục mối nguy hại của MLCommons và nêu một danh mục không hiển nhiên (Lạm dụng Bộ thông dịch mã - Code Interpreter Abuse).
- Mô tả kiến trúc đầu dò của Garak: các đầu dò (probes), bộ phát hiện (detectors), khung điều phối (harnesses).
- Mô tả cấu trúc chiến dịch nhiều lượt của PyRIT và cách nó kết hợp với các đầu dò của Garak.

## Vấn đề

Các Bài 12-15 trình bày về bề mặt tấn công. Việc triển khai trong môi trường sản xuất đòi hỏi đánh giá có thể lặp lại và có khả năng mở rộng. Ba công cụ thống trị năm 2026: Llama Guard (bộ phân loại phòng thủ), Garak (công cụ quét), PyRIT (bộ điều phối chiến dịch). Mỗi công cụ nhắm vào một tầng khác nhau trong vòng đời của đội đỏ.

## Khái niệm

### Llama Guard (Meta)

Llama Guard 3 là một mô hình Llama-3.1-8B được tinh chỉnh để phân loại đầu vào/đầu ra trên 14 danh mục của MLCommons AILuminate:
- Tội phạm bạo lực, tội phạm phi bạo lực, liên quan đến tình dục, tài liệu lạm dụng tình dục trẻ em (CSAM), phỉ báng
- Lời khuyên chuyên môn, quyền riêng tư, sở hữu trí tuệ (IP), vũ khí hủy diệt hàng loạt/vũ khí bừa bãi, thù ghét
- Tự tử/tự hại, nội dung khiêu dâm, bầu cử, lạm dụng bộ thông dịch mã (code-interpreter abuse)

Hỗ trợ 8 ngôn ngữ. Cách sử dụng: đặt trước LLM (kiểm duyệt đầu vào), sau LLM (kiểm duyệt đầu ra), hoặc cả hai. Hai cách sử dụng này tạo ra các phân phối huấn luyện khác nhau — Llama Guard 3 được phát hành dưới dạng một mô hình duy nhất xử lý cả hai.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 token/giây trên CPU di động) là biến thể lượng tử hóa dành cho thiết bị biên (edge).

Llama Guard 4 (tháng 4 năm 2025) có kích thước 12B, đa phương thức gốc, được cắt tỉa từ Llama 4 Scout. Nó thay thế cả hai mô hình tiền nhiệm (bản văn bản 8B và bản thị giác 11B) bằng một bộ phân loại duy nhất tiếp nhận cả văn bản lẫn hình ảnh.

### Garak (NVIDIA)

Công cụ quét lỗ hổng mã nguồn mở. Kiến trúc:
- **Đầu dò (Probes).** Các trình tạo cuộc tấn công đối với ảo giác, rò rỉ dữ liệu, tiêm nhiễm lời nhắc, tính độc hại, bẻ khóa. Bao gồm: tĩnh (lời nhắc cố định), động (lời nhắc được sinh tự động), thích ứng (phản hồi theo đầu ra của mục tiêu).
- **Bộ phát hiện (Detectors).** Chấm điểm đầu ra dựa trên các phương thức thất bại dự kiến — độc hại, bị rò rỉ, bị bẻ khóa.
- **Khung điều phối (Harnesses).** Quản lý các cặp đầu dò - bộ phát hiện, chạy các chiến dịch, tạo báo cáo.

TrustyAI tích hợp Garak với các tấm chắn (shields) của Llama-Stack (bộ phân loại đầu vào Prompt-Guard-86M, bộ phân loại đầu ra Llama-Guard-3-8B) để đánh giá mục tiêu được bảo vệ từ đầu đến cuối. Đánh giá dựa trên tầng nghiêm trọng (TBSA) thay thế cho việc đạt/không đạt nhị phân — một mô hình có thể vượt qua ở tầng nghiêm trọng 3 nhưng thất bại ở tầng nghiêm trọng 5 trên cùng một đầu dò.

### PyRIT (Microsoft)

Bộ công cụ nhận diện rủi ro bằng Python (Python Risk Identification Toolkit). Chuyên cho các chiến dịch đội đỏ nhiều lượt. Được xây dựng xoay quanh:
- **Bộ chuyển đổi (Converters).** Biến đổi một lời nhắc hạt giống — diễn giải lại, mã hóa, dịch thuật, đóng vai.
- **Bộ điều phối (Orchestrators).** Điều hành chiến dịch: Crescendo (leo thang từng bước), TAP (phân nhánh), RedTeaming (vòng lặp tùy chỉnh).
- **Chấm điểm (Scoring).** Dùng LLM làm giám khảo (LLM-as-judge) hoặc bộ phân loại làm giám khảo (classifier-as-judge).

PyRIT là "người anh em nặng đô" của Garak. Garak chạy hàng nghìn đầu dò đơn lượt; PyRIT chạy các chiến dịch nhiều lượt chuyên sâu được thiết kế để phá vỡ các phương thức thất bại cụ thể.

### Ngăn xếp chuẩn (The Stack)

Đặt Llama Guard ở cả hai đầu của mô hình (đầu vào và đầu ra). Chạy Garak hàng đêm để kiểm tra hồi quy (regression testing). Chạy PyRIT cho các chiến dịch đánh giá trước khi phát hành (pre-release). Đây là cấu hình mặc định năm 2026 cho hầu hết các môi trường triển khai sản xuất.

### Những cạm bẫy trong đánh giá

- **Danh tính giám khảo (Judge identity).** Cả ba công cụ đều có thể sử dụng một LLM làm giám khảo; việc hiệu chuẩn giám khảo quyết định tỷ lệ ASR được báo cáo (Bài 12). Cần chỉ định rõ giám khảo đi cùng với công cụ.
- **Đầu dò bị lỗi thời (Probe staleness).** Các đầu dò của Garak sẽ cũ đi khi các mô hình được vá lỗi để chống lại chúng. Các đầu dò thích ứng (dạng PAIR) lỗi thời chậm hơn các đầu dò tĩnh.
- **Tỷ lệ dương tính giả (FPR) của Llama Guard trên nội dung lành tính.** Các phiên bản đầu của Llama Guard đã gắn cờ quá mức các nội dung chính trị và LGBTQ+; việc hiệu chuẩn trên Llama Guard 3/4 đã cải thiện nhưng chưa được hiệu chuẩn riêng cho từng trường hợp triển khai cụ thể.

### Vị trí của bài học này trong Phase 18

Các Bài 12-15 là các họ tấn công. Bài 16 là bộ công cụ trên môi trường sản xuất. Bài 17 (WMDP) là đánh giá năng lực lưỡng dụng (dual-use capability). Bài 18 là các khuôn khổ an toàn biên giới gói các công cụ này vào một cấu trúc chính sách.

```figure
al-guard-stack
```

## Sử dụng trong thực tế (Use It)

`code/main.py` xây dựng một bộ phân loại đồ chơi kiểu Llama Guard (từ khóa + các đặc trưng ngữ nghĩa trên 14 danh mục), một khung điều phối Garak đồ chơi (vòng lặp đầu dò - bộ phát hiện), và một chuỗi bộ chuyển đổi nhiều lượt kiểu PyRIT. Bạn có thể chạy ba công cụ này trên một mục tiêu giả lập và quan sát các dấu hiệu bao phủ khác nhau.

## Đóng gói chuyển giao (Ship It)

Bài học này tạo ra `outputs/skill-red-team-stack.md`. Khi được cung cấp bản mô tả triển khai, nó chỉ rõ công cụ nào trong số ba công cụ là phù hợp, cần cấu hình những gì trong từng công cụ, và nhịp độ kiểm tra hồi quy nên thực hiện.

## Bài tập

1. Chạy `code/main.py`. So sánh tỷ lệ phát hiện của bộ phân loại kiểu Llama Guard trên các cuộc tấn công đơn lượt so với nhiều lượt.

2. Cài đặt một đầu dò Garak mới: một yêu cầu độc hại được mã hóa base64. Đo lường khả năng phát hiện của nó bằng bộ phân loại kiểu Llama Guard.

3. Mở rộng chuỗi chuyển đổi kiểu PyRIT với một bộ chuyển đổi "dịch sang tiếng Pháp, sau đó diễn giải lại". Đo lường lại tỷ lệ tấn công thành công.

4. Đọc danh sách các danh mục mối nguy hại của Llama Guard 3. Xác định hai danh mục mà dữ liệu huấn luyện có thể tạo ra tỷ lệ dương tính giả cao trên nội dung phát triển phần mềm hợp lệ.

5. So sánh các nguyên lý thiết kế của Garak và PyRIT. Đưa ra lập luận bảo vệ cho một trường hợp triển khai mà trong đó mỗi công cụ là sự lựa chọn phù hợp nhất.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Llama Guard | "bộ phân loại" | Bộ phân loại an toàn Llama-3.1-8B/4-12B được tinh chỉnh với 14 danh mục mối nguy |
| Garak | "công cụ quét" | Công cụ quét lỗ hổng mã nguồn mở của NVIDIA; gồm các đầu dò, bộ phát hiện, khung điều phối |
| PyRIT | "công cụ chiến dịch" | Bộ điều phối đội đỏ nhiều lượt của Microsoft; gồm các bộ chuyển đổi, bộ điều phối, chấm điểm |
| Prompt-Guard | "bộ phân loại nhỏ" | Bộ phân loại tiêm nhiễm lời nhắc 86M của Meta, đi kèm với Llama Guard |
| TBSA | "chấm điểm theo tầng" (tier-based scoring) | Phương pháp đạt/thất bại theo tầng của Garak thay thế cho các kết quả nhị phân |
| Chuỗi bộ chuyển đổi (Converter chain) | "diễn giải lại + mã hóa + ..." | Khái niệm nguyên thủy kết hợp của PyRIT để xây dựng các cuộc tấn công nhiều bước |
| Danh mục mối nguy MLCommons | "14 phân loại" | Phân loại chuẩn công nghiệp mà Llama Guard nhắm tới |

## Đọc thêm

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — bài báo về bộ phân loại 8B
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — bộ phân loại lượng tử hóa trên di động
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — kho lưu trữ và tài liệu của công cụ quét
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — bộ công cụ chiến dịch đội đỏ
