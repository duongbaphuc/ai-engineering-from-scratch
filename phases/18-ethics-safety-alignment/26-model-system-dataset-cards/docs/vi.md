# Thẻ Mô Hình, Thẻ Hệ Thống và Bảng Thông Số Dữ Liệu (Model, System, and Dataset Cards)

> Ba định dạng tài liệu hóa tạo nên cấu trúc minh bạch cho AI. Thẻ Mô hình (Model Cards - Mitchell et al. 2019) — nhãn thành phần dinh dưỡng cho các mô hình: dữ liệu huấn luyện, phân tích định lượng bóc tách theo nhóm, các cân nhắc đạo đức, và lưu ý cảnh báo; song chỉ 0.3% thẻ mô hình trên Hugging Face ghi nhận các cân nhắc đạo đức (Oreamuno et al. 2023). Bảng Thông số cho Bộ dữ liệu (Datasheets for Datasets - Gebru et al. 2018, CACM) — động lực khởi tạo, thành phần, quy trình thu thập, gán nhãn, phân phối, bảo trì; phỏng theo bảng thông số linh kiện điện tử. Thẻ Dữ liệu (Data Cards - Pushkarna et al., Google 2022) — cấu trúc phân tầng chi tiết theo dạng mô-đun (viễn vọng, tiềm vọng, hiển vi) đóng vai trò làm đối tượng ranh giới (boundary objects) cho nhiều nhóm độc giả khác nhau. Các bước phát triển giai đoạn 2024-2025: tạo thẻ tự động bằng LLM (CardGen, Liu et al. 2024); độ chi tiết của thẻ mô hình tương quan thuận với mức tăng tới 29% lượt tải trên HF (Liang et al. 2024); chứng thực có thể kiểm chứng (Laminator, Duddu et al. 2024); bổ sung báo cáo phát triển bền vững về carbon/nước (Jouneaux et al. Tháng 7/2025); sự xuất hiện của các loại thẻ phục vụ quy định pháp lý EU/ISO. Thẻ Hệ thống (System Cards - Sidhpurwala 2024; tính minh bạch cấp hệ thống của Meta; "Blueprints of Trust" arXiv:2509.20394) — tài liệu hóa toàn diện hệ thống AI từ đầu đến cuối, bao gồm các năng lực bảo mật, chống chèn prompt, phát hiện rò rỉ dữ liệu, và căn chỉnh theo giá trị của con người.

**Type:** Build
**Languages:** Python (thư viện chuẩn, bộ tạo thẻ mô hình + bảng thông số dữ liệu + thẻ hệ thống)
**Prerequisites:** Phase 18 · 18 (khung an toàn - safety frameworks), Phase 18 · 24 (khung pháp lý - regulatory)
**Time:** ~60 phút

## Mục Tiêu Học Tập

- Mô tả cấu trúc thẻ mô hình gốc của Mitchell et al. 2019 và bảng thông số dữ liệu của Gebru et al. 2018.
- Trình bày kiến trúc phân tầng viễn vọng / tiềm vọng / hiển vi của Thẻ Dữ liệu (Data Cards).
- Mô tả Thẻ Hệ thống (System Cards) và phạm vi bao quát toàn diện từ đầu đến cuối của chúng.
- Nêu rõ ba bước tiến nổi bật giai đoạn 2024-2025 (tạo thẻ tự động, chứng thực có thể kiểm chứng, và báo cáo tính bền vững).

## Vấn Đề Đặt Ra

Các khung pháp lý (Bài 24) và chính sách an toàn của các phòng thí nghiệm (Bài 18) đều bắt buộc phải có tài liệu hóa rõ ràng. Các định dạng tài liệu hóa đã phát triển tuần tự từ cấp mô hình cụ thể (thẻ mô hình) sang cấp bộ dữ liệu (bảng thông số dữ liệu) và sau đó là cấp hệ thống hoàn chỉnh (thẻ hệ thống). Mỗi định dạng giải quyết một phạm vi minh bạch riêng biệt. Những nỗ lực tự động hóa và chứng thực có thể kiểm chứng trong giai đoạn 2024-2025 được tạo ra nhằm tháo gỡ bài toán hóc búa tồn tại từ lâu: mức độ áp dụng trong thực tế quá thấp.

## Khái Niệm Cốt Lõi

### Thẻ Mô hình (Model Cards - Mitchell et al. 2019)

Các đề mục tiêu chuẩn:
- Chi tiết mô hình (Model details).
- Mục đích sử dụng dự kiến (Intended use).
- Các yếu tố phân nhóm (Factors: các yếu tố nhân khẩu học hoặc môi trường liên quan cần đánh giá).
- Các chỉ số đo lường (Metrics).
- Dữ liệu đánh giá (Evaluation data).
- Dữ liệu huấn luyện (Training data).
- Phân tích định lượng (Quantitative analyses: bóc tách cụ thể theo từng yếu tố).
- Cân nhắc đạo đức (Ethical considerations).
- Lưu ý cảnh báo và khuyến nghị (Caveats and recommendations).

Rào cản áp dụng thực tế: Cuộc kiểm toán năm 2023 của Oreamuno et al. trên các thẻ mô hình Hugging Face phát hiện thấy chỉ có 0.3% số thẻ ghi nhận các cân nhắc về đạo đức.

### Bảng Thông số cho Bộ dữ liệu (Datasheets for Datasets - Gebru et al. 2018)

Phỏng theo bảng thông số kỹ thuật (datasheet) của ngành linh kiện phần cứng điện tử. Các đề mục:
- Động lực khởi tạo (Motivation: lý do bộ dữ liệu được tạo ra).
- Thành phần (Composition: bên trong bộ dữ liệu có những gì).
- Quy trình thu thập (Collection process: dữ liệu được tổng hợp như thế nào).
- Gán nhãn (Labeling: nếu có áp dụng).
- Trường hợp sử dụng (Uses: dự kiến, bị nghiêm cấm, và các rủi ro).
- Phân phối (Distribution).
- Duy trì và bảo trì (Maintenance).

Xuất bản trên tạp chí CACM năm 2021. Bảng thông số là tài liệu thượng nguồn; tính chính xác của thẻ mô hình phụ thuộc hoàn toàn vào tính trung thực của bảng thông số bộ dữ liệu.

### Thẻ Dữ liệu (Data Cards - Pushkarna et al., Google 2022)

Chi tiết được phân tầng theo dạng mô-đun với ba mức độ thu phóng:
- **Viễn vọng (Telescopic).** Tóm tắt mức cao dành cho đối tượng phi kỹ thuật hoặc nhà quản lý.
- **Tiềm vọng (Periscopic).** Tổng quan mức trung dành cho các kỹ sư thực hành học máy (ML practitioners).
- **Hiển vi (Microscopic).** Tài liệu chi tiết cấp tính năng và thuộc tính dành cho các chuyên gia kiểm toán.

Khung đối tượng ranh giới (boundary object): cùng một tài liệu nhưng những đối tượng độc giả khác nhau có thể trích xuất những lớp thông tin phù hợp với mục tiêu của họ.

### Thẻ Hệ thống (System Cards)

Phạm vi: toàn bộ hệ thống AI hoàn chỉnh bao gồm mô hình nền tảng + ngăn xếp an toàn (safety stack) + ngữ cảnh triển khai thực tế. Các đề mục điển hình gồm:
- Năng lực bảo mật (Security capabilities).
- Cơ chế chống chèn prompt (Prompt-injection protection).
- Phát hiện rò rỉ dữ liệu trái phép (Data-exfiltration detection).
- Sự căn chỉnh với các giá trị nhân văn đã công bố (Alignment with stated human values).
- Quy trình ứng phó sự cố (Incident response).

Công trình của Sidhpurwala 2024 và các sáng kiến minh bạch cấp hệ thống của Meta. Nghiên cứu "Blueprints of Trust" (arXiv:2509.20394) chính thức chuẩn hóa Thẻ Hệ thống như một lớp bổ trợ tầng triển khai cho Thẻ Mô hình.

### Các bước phát triển giai đoạn 2024-2025

- **CardGen (Liu et al. 2024).** Tạo thẻ mô hình tự động bằng LLM; ghi nhận tính khách quan cao hơn nhiều thẻ do con người tự soạn thảo trên các trường tiêu chuẩn của Mitchell 2019.
- **Tương quan lượt tải (Liang et al. 2024).** Các thẻ mô hình chi tiết có mối tương quan với mức tăng tới 29% lượt tải trên Hugging Face — áp lực triển khai giờ đây đến từ chính thị trường, không chỉ dừng lại ở yêu cầu tuân thủ pháp lý.
- **Laminator (Duddu et al. 2024).** Chứng thực có thể kiểm chứng (verifiable attestations) thông qua môi trường thực thi tin cậy phần cứng (TEE) / chữ ký số mật mã — cho phép thẻ mô hình đính kèm bằng chứng xác thực (proof-of-claim), thay vì chỉ là lời tuyên bố đơn thuần.
- **Tính bền vững (Jouneaux et al. Tháng 7/2025).** Bổ sung các chỉ số dấu chân carbon, mức tiêu thụ nước và năng lượng tính toán; gắn kết với các tiêu chuẩn ISO đang dần hình thành.
- **Thẻ phục vụ tuân thủ quy định (Regulatory cards).** Đạo luật AI của EU (EU AI Act - Bài 24) trong chương Minh bạch thuộc Bộ Quy tắc Thực hành cho AI Mục đích Chung (GPAI Code of Practice) yêu cầu thẻ mô hình như một tài liệu bắt buộc để chứng minh tuân thủ.

### Vị trí của bài học trong Phase 18

Bài 24-25 cung cấp các tầng pháp lý và tầng lỗ hổng CVE. Bài 26 là tầng tài liệu hóa tiêu chuẩn. Bài 27 nghiên cứu sâu về quản trị dữ liệu huấn luyện, chính là nguồn thượng nguồn của bảng thông số. Bài 28 bao quát hệ sinh thái nghiên cứu nơi tạo ra các bài đánh giá được trích dẫn trong các tấm thẻ.

```figure
an-card-scopes
```

## Áp Dụng Thực Tế

Tệp `code/main.py` tự động tạo ra một thẻ mô hình, một bảng thông số dữ liệu và một thẻ hệ thống tối thiểu cho một trường hợp triển khai mẫu. Mỗi tài liệu đều tuân thủ cấu trúc đề mục chuẩn hóa. Bạn có thể kiểm tra định dạng và so sánh ranh giới phạm vi giữa ba loại thẻ.

## Đóng Gói Sản Phẩm

Bài học này xây dựng tệp `outputs/skill-card-audit.md`. Khi tiếp nhận một thẻ mô hình, bảng thông số dữ liệu hoặc thẻ hệ thống, kỹ năng này sẽ kiểm toán độ bao phủ của các đề mục, tính bóc tách định lượng theo nhóm nhân khẩu học, và xác minh xem tài liệu có chứa các chứng thực mật mã có thể kiểm chứng hay không.

## Bài Tập

1. Chạy `code/main.py`. Khảo sát các thẻ được tạo ra. Xác định các phần còn yếu (chỉ chứa văn bản giữ chỗ - placeholder) và nêu rõ loại bằng chứng thực nghiệm nào cần được bổ sung để củng cố chúng.

2. Mở rộng thẻ mô hình với một phân tích định lượng bóc tách chi tiết trên hai nhóm nhân khẩu học khác nhau (tham chiếu Bài 20).

3. Đọc nghiên cứu của Oreamuno et al. 2023 về tỷ lệ áp dụng 0.3%. Hãy đề xuất một thay đổi mang tính cấu trúc cho quy chuẩn thẻ mô hình nhằm thúc đẩy việc ghi nhận các cân nhắc đạo đức.

4. Laminator (Duddu et al. 2024) ứng dụng TEE để tạo chứng thực có thể kiểm chứng. Hãy thiết kế một trường dữ liệu trong thẻ mô hình mang chữ ký mật mã cho kết quả đánh giá thực nghiệm và mô tả vai trò của bên xác minh (verifier).

5. Viết một Thẻ Hệ thống (Thẻ Hệ thống, không phải Thẻ Mô hình) cho một dự án bạn từng làm hoặc một hệ thống triển khai giả định. Xác định đề mục mang lại giá trị cao nhất cho các chuyên gia kiểm toán độc lập bên thứ ba.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực chất |
|---|---|---|
| Model Card | "Thẻ Mitchell" | Tiêu chuẩn tài liệu hóa cho các mô hình ML theo Mitchell et al. 2019 |
| Datasheet | "Bảng thông số Gebru" | Tiêu chuẩn tài liệu hóa cho các bộ dữ liệu theo Gebru et al. 2018 |
| Data Card | "Thẻ Pushkarna" | Chuẩn tài liệu hóa dữ liệu phân tầng dạng mô-đun của Google 2022 |
| System Card | "Thẻ triển khai hệ thống" | Tài liệu hóa hệ thống AI toàn diện từ đầu đến cuối bao gồm cả ngăn xếp an toàn |
| Boundary object | "Đối tượng ranh giới" | Khung của Thẻ Dữ liệu: cùng một tài liệu phục vụ nhiều đối tượng độc giả khác nhau |
| Verifiable attestation | "Chứng thực Laminator" | Bằng chứng mật mã hoặc TEE đính kèm một tuyên bố trong tài liệu |
| Sustainability field | "Dấu chân carbon / nước" | Bổ sung mới từ năm 2025 phục vụ hạch toán tác động môi trường |

## Tài Liệu Đọc Thêm

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — Bài báo gốc về thẻ mô hình
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — Bài báo về bảng thông số dữ liệu
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — Tài liệu hóa dữ liệu phân tầng
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — Chuẩn hóa Thẻ Hệ thống

