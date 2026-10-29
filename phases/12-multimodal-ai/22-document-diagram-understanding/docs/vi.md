# Thấu hiểu tài liệu và biểu đồ (Document and Diagram Understanding)

> Tài liệu văn bản không đơn thuần là những bức ảnh chụp. Một tệp PDF, bài báo nghiên cứu khoa học, hóa đơn thanh toán hay biểu mẫu điền tay đều mang bố cục không gian (layout), bảng biểu, sơ đồ đồ thị, chú thích chân trang, tiêu đề đầu trang và cấu trúc ngữ nghĩa phức tạp mà việc hiểu ảnh thông thường không thể nắm bắt được. Ngăn xếp công nghệ trước kỷ nguyên VLM là một đường ống nối tầng rời rạc: Tesseract OCR + LayoutLMv3 + các luật phỏng đoán (heuristics) để bóc tách bảng. Làn sóng VLM đã thay thế hoàn toàn đường ống này bằng các mô hình phi OCR (OCR-free models) — Donut (2022), Nougat (2023), DocLLM (2023) — có khả năng sinh trực tiếp văn bản có cấu trúc. Đến năm 2026, chuẩn mực biên giới hiện đại chỉ đơn giản là "đưa trực tiếp ảnh trang tài liệu vào Claude Opus 4.7 ở độ phân giải gốc 2576px", và định dạng đánh dấu cấu trúc sẽ được mô hình tự động trích xuất. Bài học này phân tích toàn diện tiến trình phát triển qua ba kỷ nguyên của AI xử lý tài liệu.

**Type:** Build
**Languages:** Python (stdlib, khung sườn bộ phân tích tài liệu nhận thức bố cục)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích ba kỷ nguyên phát triển của AI tài liệu: đường ống OCR truyền thống (OCR pipeline), mô hình phi OCR (OCR-free) và VLM bản địa (VLM-native).
- Mô tả ba luồng đầu vào của LayoutLMv3: văn bản, bố cục (tọa độ hộp bao bbox), các mảnh ảnh trực quan, kết hợp cơ chế mặt nạ thống nhất (unified masking).
- So sánh chi tiết các mô hình: Donut (phi OCR, ảnh $\to$ dữ liệu cấu trúc), Nougat (bài báo khoa học $\to$ mã nguồn LaTeX), DocLLM (mô hình sinh nhận thức bố cục) và PaliGemma 2 (VLM bản địa).
- Lựa chọn giải pháp mô hình tối ưu cho từng bài toán tài liệu cụ thể (hóa đơn, bài báo khoa học, biểu mẫu viết tay, biên lai đa ngôn ngữ).

## Vấn đề đặt ra

Yêu cầu "Hãy hiểu tệp PDF này" ẩn chứa độ phức tạp kỹ thuật rất lớn. Thông tin trong tài liệu phân bổ ở nhiều tầng:

- Nội dung văn bản (chiếm 90% lượng thông tin ngữ nghĩa).
- Bố cục không gian (tiêu đề, chú thích chân trang, thanh bên lề, cách dàn trang hai cột).
- Cấu trúc bảng (các hàng, cột, ô trộn - merged cells).
- Đồ thị, sơ đồ và biểu đồ minh họa.
- Ghi chú viết tay chêm xen.
- Định dạng phông chữ và nghệ thuật in ấn (tiêu đề in đậm so với nội dung thân bài).

Phương pháp OCR thô chỉ trích xuất chữ thành một khối văn bản phẳng và làm mất hoàn toàn ngữ cảnh còn lại. Một hệ thống xử lý hóa đơn tự động bắt buộc phải hiểu rằng cụm từ "Tổng cộng: $1.245" nằm ở góc dưới cùng bên phải, chứ không phải là một ghi chú chú thích ở chân trang.

## Khái niệm cốt lõi

### Kỷ nguyên 1 — Đường ống OCR nối tầng (Trước 2021)

Ngăn xếp công nghệ kinh điển:

1. Chuyển đổi PDF $\to$ tệp ảnh cho từng trang riêng biệt.
2. Tesseract (hoặc các dịch vụ OCR thương mại) bóc tách văn bản kèm tọa độ hộp bao (bounding boxes) cho từng từ.
3. Bộ phân tích bố cục (layout analyzer) nhận diện các khối chức năng (tiêu đề, bảng biểu, đoạn văn bản).
4. Bộ nhận dạng cấu trúc bảng phân tích các hàng và cột.
5. Bộ quy tắc nghiệp vụ kết hợp biểu thức chính quy (regex) để trích xuất các trường thông tin cần thiết.

Quy trình này hoạt động ổn định trên văn bản in ấn rõ nét. Tuy nhiên, nó đổ vỡ khi gặp chữ viết tay, bản quét bị nghiêng lệch, bảng biểu cấu trúc phức tạp hoặc chữ viết phi La-tinh. Mỗi lỗi phát sinh đều đòi hỏi phải bổ sung một nhánh ngoại lệ xử lý thủ công bằng code.

### TrOCR (2021)

TrOCR (Li và cộng sự, arXiv:2109.10282) thay thế khối CNN-CTC truyền thống của Tesseract bằng một transformer mã hóa - giải mã được huấn luyện trên dữ liệu ảnh chữ tổng hợp lẫn thực tế. Bước đột phá này mang lại sự cải thiện vượt bậc đối với chữ viết tay và văn bản đa ngôn ngữ. Mặc dù vẫn là mô hình đường ống nối tầng (bộ phát hiện $\to$ TrOCR $\to$ phân tích bố cục), nhưng chất lượng nhận dạng ở bước OCR đã được nâng lên một tầm cao mới.

### Kỷ nguyên 2 — Kỷ nguyên phi OCR (OCR-free, 2022-2023)

Các mô hình phi OCR thế hệ đầu tiên đặt ra triết lý: bỏ qua hoàn toàn bước phát hiện từ riêng rẽ, ánh xạ trực tiếp từ các pixel của ảnh trang tài liệu sang đầu ra văn bản có cấu trúc.

- **Donut** (Kim và cộng sự, arXiv:2111.15664):
  - Kiến trúc transformer mã hóa - giải mã, với bộ mã hóa là Swin-B.
  - Sinh trực tiếp chuỗi JSON cho bài toán hiểu biểu mẫu, Markdown cho tóm tắt tài liệu, hoặc bất kỳ lược đồ đích nào.
  - Hoàn toàn không cần module OCR rời, không cần trích xuất bố cục trung gian.
- **Nougat** (Blecher và cộng sự, arXiv:2308.13418):
  - Được huấn luyện chuyên sâu trên dữ liệu bài báo khoa học.
  - Đầu ra là mã nguồn LaTeX / Markdown chuẩn mực.
  - Xử lý mượt mà các công thức toán học phức tạp, bố cục nhiều cột và hình vẽ.
  - Trở thành công cụ mặc định mà mọi hệ thống phân tích arXiv đều sử dụng.

Các mô hình này là những chuyên gia hẹp chứ không phải mô hình tổng quát. Đưa Donut vào xử lý bài báo khoa học sẽ thất bại; và đưa Nougat vào trích xuất hóa đơn thanh toán cũng không mang lại kết quả mong muốn.

### LayoutLMv3 (2022)

Một nhánh phát triển độc lập và cực kỳ thành công. LayoutLMv3 (Huang và cộng sự, arXiv:2204.08387) vẫn duy trì tầng OCR nhưng bổ sung năng lực hiểu biết bố cục không gian đa phương thức:

- Ba luồng đầu vào đồng thời: các token văn bản từ OCR, tọa độ hộp bao 2D của từng token, và các mảnh ảnh trực quan (image patches).
- Mục tiêu huấn luyện che mặt nạ (masked training objective) thống nhất trên cả ba thể thức (che từ ngữ, che mảnh ảnh, che tọa độ bố cục).
- Ứng dụng hạ nguồn: phân loại tài liệu, trích xuất thực thể thông tin, hỏi đáp bảng biểu.

LayoutLMv3 đại diện cho đỉnh cao của dòng mô hình hiểu tài liệu dựa trên OCR. Nó cực kỳ mạnh mẽ với biểu mẫu và hóa đơn thương mại, đem lại độ chính xác cao nhất trên các bộ điểm chuẩn tài liệu chuẩn hóa thời kỳ tiền VLM.

### DocLLM (2023)

DocLLM (Wang và cộng sự, arXiv:2401.00908) là người anh em mang tính tạo sinh (generative) của LayoutLM. Mô hình này sinh các câu trả lời dạng tự do có điều kiện dựa trên các token bố cục không gian. Tối ưu hơn cho các bài toán hỏi đáp trên tài liệu, tuy nhiên vẫn phụ thuộc vào chất lượng OCR đầu vào.

### Kỷ nguyên 3 — VLM bản địa (VLM-native, 2024+)

Bước sang năm 2024, các mô hình VLM tổng quát đã đủ mạnh để thay thế toàn bộ đường ống xử lý cồng kềnh trước đây. Chỉ cần đưa toàn bộ ảnh trang tài liệu ở độ phân giải cao vào VLM, đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận câu trả lời hoàn chỉnh.

- LLaVA-NeXT với kỹ thuật chia lưới AnyRes 336-tile đáp ứng tốt cho các tài liệu nhỏ.
- Qwen2.5-VL với cơ chế độ phân giải động (dynamic resolution) xử lý mượt mà tài liệu trên 2048+ pixel ở dạng bản địa.
- Claude Opus 4.7 hỗ trợ tài liệu lên đến 2576px.
- PaliGemma 2 (tháng 12 năm 2024) được Google huấn luyện chuyên biệt cho bài toán hiểu tài liệu và chữ viết tay.

Khoảng cách giữa VLM bản địa và đường ống OCR truyền thống thu hẹp nhanh chóng. Đến năm 2026, VLM bản địa hoàn toàn chiếm ưu thế ở:

- Chữ trong ngữ cảnh tự nhiên (ảnh chụp hóa đơn thực tế kết hợp chữ in và chữ viết tay lộn xộn).
- Bảng biểu phức tạp có các ô trộn ngang dọc bất quy tắc.
- Công thức toán học nằm nhúng xen kẽ trong đoạn văn.
- Biểu đồ và đồ thị có các nhãn ghi chú giải thích trực tiếp.

Tuy nhiên, các đường ống OCR truyền thống vẫn giữ vị thế nhất định trong:

- Khối lượng xử lý tài liệu quét chuẩn hóa ở quy mô khổng lồ (hàng chục triệu trang/ngày) nơi chi phí và độ trễ trên mỗi trang là yếu tố sống còn.
- Tính tất định và độ tin cậy của quy trình (lỗi sai mang tính cơ học, dễ bắt lỗi hơn so với hiện tượng ảo giác ngẫu nhiên của VLM).
- Các lĩnh vực có quy định kiểm toán nghiêm ngặt (regulated environments) đòi hỏi đầu ra OCR nguyên bản có thể truy vết từng ký tự.

### Chuẩn mực biên giới: Claude 4.7 và GPT-5

Với khả năng tiếp nhận ảnh đầu vào bản địa lên tới 2576 pixel, các mô hình VLM biên giới đạt độ chính xác tương đương chuyên gia con người. Điểm số chuẩn mực đầu năm 2026 ghi nhận:

- **DocVQA:** Claude 4.7 đạt ~95,1, PaliGemma 2 đạt ~88,4, Nougat đạt ~77,3, đường ống LayoutLMv3 đạt ~83.
- **ChartQA:** Claude 4.7 đạt ~92,2, vượt xa mức ~78 của thế hệ GPT-4V.
- **VisualMRC:** Claude 4.7 đạt ~94.

Khoảng cách chủ yếu đến từ độ phân giải đầu vào và quy mô tham số của LLM nền tảng. Các mô hình nguồn mở quy mô 7B hiện bám đuổi sát nút với cách biệt chỉ vài điểm phần trăm.

### Công thức toán học và đầu ra định dạng LaTeX

Tài liệu học thuật đòi hỏi chuyển đổi chính xác các công thức toán học sang mã nguồn LaTeX. Nougat được sinh ra cho mục đích này. Các mô hình VLM được huấn luyện có chủ đích với mục tiêu LaTeX (như Qwen2.5-VL-Math, các biến thể cải tiến của Nougat) sinh ra mã LaTeX biên dịch được hoàn hảo. Nếu không có dữ liệu căn chỉnh chuyên biệt cho LaTeX, VLM thông thường chỉ sinh ra các phiên âm mang tính diễn giải, thiếu độ chuẩn xác cú pháp.

Quy trình chuẩn cho tài liệu nghiên cứu khoa học năm 2026: chạy Nougat trên toàn bộ tệp PDF, sau đó chuyển các trang chứa sơ đồ hoặc biểu đồ hóc búa cho VLM xử lý chuyên sâu.

### Thách thức từ chữ viết tay

Chữ viết tay vẫn là phân khúc khó khăn nhất. Dạng tài liệu lai giữa chữ in sẵn và chữ bác sĩ điền tay (bệnh án, đơn khai thông tin) là nơi các đường ống OCR chuyên dụng kết hợp module nhận dạng chữ viết tay vẫn chiếm ưu thế về mặt chi phí so với việc gọi API VLM toàn bộ. Tuy nhiên, các VLM thế hệ mới (Claude 4.7, PaliGemma 2) đang liên tục nâng cao trần chính xác trên mảng này.

### Công thức triển khai thực tế năm 2026

Khi thiết kế hệ thống AI xử lý tài liệu mới:

- **Hóa đơn in chuẩn hóa ở quy mô công nghiệp lớn:** LayoutLMv3 + hệ luật regex (tối ưu chi phí và tốc độ).
- **Tài liệu hỗn hợp đa dạng (khoa học, viết tay, đơn từ, hợp đồng):** VLM bản địa (PaliGemma 2 hoặc Qwen2.5-VL).
- **Hệ thống nạp toàn văn bài báo arXiv:** Dùng Nougat cho công thức toán, kết hợp VLM cho hình ảnh biểu đồ.
- **Tuân thủ quy định pháp lý / tài chính:** Đường ống OCR truyền thống làm nhiệm vụ trích xuất chính, kết hợp VLM đóng vai trò thẩm định viên (validator) để đối soát chéo.

```figure
mm-doc-layout
```

## Sử dụng trong thực tế

Tệp `code/main.py`:

- Xây dựng một bộ token hóa nhận thức bố cục tối giản: từ các cặp (văn bản, tọa độ bbox), tạo ra cấu trúc vector đầu vào chuẩn phong cách LayoutLMv3.
- Bộ sinh lược đồ tác vụ phong cách Donut: khuôn mẫu JSON cấu trúc hóa dành cho trích xuất biểu mẫu.
- Bảng so sánh ngân sách token trên mỗi trang tài liệu giữa các kiến trúc: đường ống OCR, Donut, Nougat và VLM bản địa.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-document-ai-stack-picker.md`. Dựa trên các tiêu chí dự án (miền nghiệp vụ, quy mô xử lý hàng ngày, yêu cầu chất lượng, ràng buộc pháp lý), kỹ năng này sẽ tự động đề xuất ngăn xếp tối ưu giữa: đường ống OCR truyền thống, mô hình phi OCR chuyên biệt, hoặc VLM bản địa.

## Bài tập thực hành

1. Dự án của bạn phải xử lý 10 triệu trang hóa đơn mỗi ngày. Kiến trúc ngăn xếp nào giúp tối thiểu hóa chi phí tính toán trên mỗi trang mà vẫn đảm bảo độ chính xác trên 98%?

2. Tại sao LayoutLMv3 lại vượt trội hơn các mô hình thuần CLIP-VLM trên các bài toán hỏi đáp biểu mẫu nhưng lại kém hơn ở các biển hiệu chữ ngoài đời thực (scene-text)? Luồng dữ liệu tọa độ hộp bao (bbox stream) đã đánh đổi yếu tố nào?

3. Mô hình Nougat chuyên sinh mã LaTeX. Hãy đề xuất một trường hợp kiểm thử mà VLM bản địa vượt trội Nougat về độ chuẩn xác LaTeX, và ngược lại một trường hợp Nougat chiến thắng tuyệt đối.

4. Đọc bài báo PaliGemma 2 của Google (2024). Bổ sung dữ liệu huấn luyện then chốt nào đã tạo ra bước nhảy vọt về khả năng hiểu tài liệu so với phiên bản PaliGemma 1 tiền nhiệm?

5. Thiết kế một hệ thống lai đáp ứng tiêu chuẩn kiểm toán tài chính: đường ống OCR làm nhiệm vụ trích xuất sơ cấp, mô hình VLM làm nhiệm vụ đối soát thứ cấp. Bạn sẽ thiết lập cơ chế hòa giải xung đột (conflict resolution) ra sao khi hai bên đưa ra kết quả lệch nhau?

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Đường ống OCR (OCR pipeline) | "Kiểu Tesseract" | Ngăn xếp phân tầng: phát hiện $\to$ OCR $\to$ bố cục $\to$ tập luật; mang tính tất định nhưng giòn gãy |
| Phi OCR (OCR-free) | "Kiểu Donut" | Transformer nhận ảnh và sinh trực tiếp cấu trúc đích, bỏ qua tầng OCR rời rạc; mô hình hợp nhất |
| Nhận thức bố cục (Layout-aware) | "Kiểu LayoutLM" | Đầu vào tích hợp tọa độ hộp bao (bbox) của từng token; huấn luyện che mặt nạ thống nhất đa thể thức |
| VLM bản địa (VLM-native) | "VLM biên giới" | Đưa trực tiếp ảnh trang độ phân giải cao vào các VLM lớn (Claude, GPT, Qwen); không cần đường ống |
| DocVQA | "Chuẩn đo tài liệu" | Bộ dữ liệu đo chuẩn hỏi đáp tài liệu thị giác; chỉ số được trích dẫn nhiều nhất |
| Đầu ra đánh dấu (Markup output) | "LaTeX / Markdown" | Định dạng đầu ra có cấu trúc cú pháp thay vì văn bản tự do; hỗ trợ tự động hóa hạ nguồn |

## Tài liệu đọc thêm

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
