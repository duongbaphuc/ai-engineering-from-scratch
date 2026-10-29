# Nguồn Gốc Dữ Liệu và Quản Trị Dữ Liệu Huấn Luyện (Data Provenance and Training-Data Governance)

> Đạo luật AI của EU yêu cầu các tiêu chuẩn từ chối (opt-out) mà máy có thể đọc được đối với GPAI trước tháng 8/2025 (thông qua ngoại lệ TDM của Chỉ thị Bản quyền EU). Dự luật AB 2013 của California (được ký thành luật năm 2024) — tính minh bạch của dữ liệu huấn luyện AI tạo sinh yêu cầu các nhà phát triển công bố bản tóm tắt các bộ dữ liệu với 12 trường bắt buộc. Sự thống nhất quan điểm của các Cơ quan Bảo vệ Dữ liệu (DPA) năm 2025 về căn cứ "lợi ích hợp pháp" (legitimate interest): DPC Ireland (21/05/2025) chấp thuận việc Meta huấn luyện LLM trên nội dung người lớn công khai từ bên thứ nhất tại EU/EEA đi kèm các biện pháp bảo vệ sau ý kiến của EDPB; Tòa án Thượng thẩm Tỉnh Cologne (23/05/2025) bác bỏ lệnh cấm; DPA Hamburg hủy bỏ thủ tục khẩn cấp; ICO Vương quốc Anh (23/09/2025) phản hồi tích cực đối với các biện pháp bảo vệ của LinkedIn khi huấn luyện AI (minh bạch, đơn giản hóa từ chối, kéo dài thời hạn phản đối) và tiếp tục theo dõi — không phải là phê duyệt chính thức. ANPD của Brazil (02/07/2024) từng đình chỉ việc xử lý dữ liệu của Meta do thiếu minh bạch thông tin; biện pháp phòng ngừa này được gỡ bỏ vào ngày 30/08/2024 sau khi Meta đệ trình kế hoạch tuân thủ. Bài toán then chốt về tính bất khả đảo ngược (irreversibility problem): các khung chấp thuận cookie vốn được thiết kế cho việc theo dõi theo thời gian thực có thể đảo ngược; một khi dữ liệu đã đi vào trọng số mô hình, việc xóa bỏ phẫu thuật (surgical erasure) là bất khả thi — không có quyền xóa bỏ dữ liệu (right-to-erasure) theo GDPR mang tính khả thi thực tế đối với các mạng nơ-ron đã huấn luyện. Thời điểm duy nhất để thực thi tuân thủ là tại lúc thu thập dữ liệu. Sáng kiến Nguồn gốc Dữ liệu (Data Provenance Initiative - dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", Tháng 7/2024): cuộc kiểm toán quy mô lớn cho thấy sự suy giảm nhanh chóng của tài nguyên dữ liệu AI công cộng khi các nhà xuất bản ồ ạt bổ sung rào cản robots.txt.

**Type:** Learn
**Languages:** Python (thư viện chuẩn, bộ tạo khung cấu trúc tuân thủ 12 trường California AB 2013)
**Prerequisites:** Phase 18 · 24 (khung pháp lý - regulatory), Phase 18 · 26 (các loại thẻ - cards)
**Time:** ~60 phút

## Mục Tiêu Học Tập

- Mô tả 12 trường dữ liệu bắt buộc theo Dự luật AB 2013 của California về tính minh bạch của dữ liệu huấn luyện AI tạo sinh.
- Nêu rõ quan điểm của các DPA năm 2025 về việc huấn luyện LLM dựa trên căn cứ lợi ích hợp pháp (DPC Ireland, ICO Anh, Hamburg, Cologne).
- Trình bày bài toán bất khả đảo ngược: tại sao quyền xóa bỏ dữ liệu của GDPR không có biện pháp tương đương mang tính khả thi thực tế trên các mạng nơ-ron đã huấn luyện.
- Nêu rõ kết quả phát hiện trong báo cáo "Consent in Crisis" của Sáng kiến Nguồn gốc Dữ liệu (Data Provenance Initiative).

## Vấn Đề Đặt Ra

Quản trị dữ liệu huấn luyện là nguồn thượng nguồn của mọi thẻ mô hình (Bài 26) và nghĩa vụ pháp lý bắt buộc (Bài 24). Trong giai đoạn 2024-2025, bức tranh pháp lý toàn cầu đã quy tụ xung quanh ba nguyên tắc cốt lõi: hạ tầng từ chối tham gia (opt-out infrastructure), công bố thông tin chi tiết theo từng bộ dữ liệu, và điều khoản nới lỏng theo căn cứ lợi ích hợp pháp đối với dữ liệu khả dụng công khai. Các đơn vị phát triển không tuân thủ ngay từ khâu thu thập dữ liệu sẽ không thể cứu vãn hoặc khắc phục ở các khâu hạ nguồn.

## Khái Niệm Cốt Lõi

### Dự luật AB 2013 của California

Được ký duyệt năm 2024. Tài liệu thông tin phải được công bố công khai trước hoặc vào ngày 1 tháng 1 năm 2026 đối với các hệ thống phát hành từ ngày 1 tháng 1 năm 2022 trở đi. Điều khoản 3111(a) yêu cầu các nhà phát triển phải công bố bản tóm tắt cấp cao về các bộ dữ liệu được sử dụng trong huấn luyện với 12 hạng mục theo luật định:
1. Nguồn gốc hoặc chủ sở hữu của các bộ dữ liệu.
2. Mô tả cách thức các bộ dữ liệu đóng góp vào mục đích dự kiến của hệ thống AI.
3. Số lượng điểm dữ liệu (data points) trong bộ dữ liệu (chấp nhận khoảng ước lượng chung; ước tính đối với các tập dữ liệu động).
4. Mô tả các loại điểm dữ liệu (các loại nhãn đối với tập dữ liệu có nhãn; đặc điểm tổng quát đối với tập không nhãn).
5. Bộ dữ liệu có chứa dữ liệu được bảo hộ bởi bản quyền, nhãn hiệu hoặc bằng sáng chế hay hoàn toàn thuộc phạm vi công cộng (public domain).
6. Bộ dữ liệu được mua hay được cấp phép sử dụng (licensed).
7. Bộ dữ liệu có chứa thông tin cá nhân hay không (theo Bộ luật Dân sự California §1798.140(v)).
8. Bộ dữ liệu có chứa thông tin tổng hợp của người tiêu dùng hay không (theo Bộ luật Dân sự California §1798.140(b)).
9. Quy trình làm sạch, tiền xử lý, hoặc các sửa đổi khác do bên phát triển thực hiện, kèm theo mục đích dự kiến.
10. Khoảng thời gian thu thập dữ liệu, kèm thông báo nếu quá trình thu thập vẫn đang diễn ra.
11. Thời điểm bộ dữ liệu được sử dụng lần đầu tiên trong quá trình phát triển.
12. Hệ thống có sử dụng hoặc liên tục sử dụng việc tạo dữ liệu tổng hợp (synthetic data) hay không.

Hạng mục 12 (dữ liệu tổng hợp) là điểm mới hoàn toàn so với bảng thông số Gebru et al. 2018. Hạng mục 7 (thông tin cá nhân) kích hoạt các nghĩa vụ theo Đạo luật Quyền Riêng tư California (CPRA). Đạo luật miễn trừ các hệ thống an ninh/tính toàn vẹn, vận hành tàu bay, và hệ thống an ninh quốc gia cấp liên bang (Điều khoản 3111(b)).

### Đạo luật AI của EU (Bài 24) và Cơ chế từ chối TDM

Ngoại lệ Khai thác Văn bản và Dữ liệu (Text and Data Mining - TDM) thuộc Chỉ thị Bản quyền EU cho phép huấn luyện trên các nội dung công khai trừ khi chủ sở hữu bản quyền chủ động từ chối (opt-out). Chương Bản quyền trong Bộ Quy tắc Thực hành GPAI của Đạo luật AI EU yêu cầu các nhà cung cấp GPAI phải tôn trọng các tín hiệu từ chối mà máy có thể đọc được (như `robots.txt`, xác nhận "No AI Training" của C2PA, v.v.).

### Sự hội tụ quan điểm của các DPA năm 2025 về Lợi ích hợp pháp

DPC Ireland (21/05/2025): Chấp thuận kế hoạch của Meta về việc huấn luyện trên nội dung người dùng trưởng thành công khai từ bên thứ nhất tại EU/EEA kèm theo các biện pháp bảo vệ sau ý kiến chỉ đạo của EDPB. Tòa án Thượng thẩm Tỉnh Cologne (23/05/2025) bác bỏ lệnh cấm chống lại Meta: cơ chế từ chối (opt-out) được coi là thỏa đáng. DPA Hamburg hủy bỏ thủ tục khẩn cấp để đảm bảo tính nhất quán trên toàn EU. ICO Vương quốc Anh (23/09/2025) đưa ra phản hồi pháp lý tích cực — không phải chứng nhận miễn trừ chính thức — đối với việc LinkedIn tái khởi động huấn luyện AI đi kèm các biện pháp bảo vệ tương tự và cơ chế giám sát liên tục.

Nguyên tắc hội tụ: Căn cứ "lợi ích hợp pháp" (legitimate interest) có thể biện minh cho việc huấn luyện trên nội dung công khai của bên thứ nhất khi có cơ chế từ chối rõ ràng. Không bắt buộc phải có sự chấp thuận rõ ràng (consent) từ trước.

### Cơ quan Bảo vệ Dữ liệu Brazil ANPD (Tháng 7/2024)

Đã đình chỉ việc Meta xử lý dữ liệu người dùng Brazil để huấn luyện AI do không đảm bảo tính minh bạch thông tin. Phán quyết này khác biệt so với các DPA châu Âu — ANPD ưu tiên sự minh bạch tuyệt đối hơn là tính khả nhận của căn cứ lợi ích hợp pháp.

### Bài toán bất khả đảo ngược (The Irreversibility Problem)

Khung chấp thuận cookie được thiết kế cho các hoạt động theo dõi hành vi thời gian thực và có thể đảo ngược bất cứ lúc nào. Dữ liệu huấn luyện thì khác: một khi dữ liệu đã thâm nhập vào trọng số của mô hình (model weights), việc bóc tách phẫu thuật là bất khả thi. Huấn luyện lại từ đầu (retraining from scratch) là biện pháp khắc phục triệt để duy nhất, song chi phí của nó lại đắt đỏ tới mức không khả thi.

Các biện pháp giảm thiểu cục bộ:
- **Quên máy (Machine unlearning).** Loại bỏ gần đúng; đo lường thông qua tấn công suy luận thành viên (MIA - Bài 22).
- **Định vị dựa trên hàm ảnh hưởng (Influence function-based localization).** Xác định các trọng số chịu ảnh hưởng nhiều nhất bởi dữ liệu cần xóa; cập nhật chọn lọc.
- **Áp chế thông qua tinh chỉnh (Fine-tune-suppression).** Huấn luyện mô hình từ chối đưa ra các đầu ra có nguồn gốc từ dữ liệu đó.

Không có biện pháp nào giải quyết triệt để vấn đề. Thời điểm duy nhất để thực thi tuân thủ là tại lúc thu thập dữ liệu.

### Sáng kiến Nguồn gốc Dữ liệu (Data Provenance Initiative)

Trang web dataprovenance.org. Nghiên cứu của Longpre, Mahari, Lee et al. "Consent in Crisis" (Tháng 7/2024): cuộc kiểm toán quy mô lớn đối với tài nguyên dữ liệu huấn luyện AI công cộng. Phát hiện: các nhà xuất bản đang bổ sung rào cản `robots.txt` với tốc độ ngày càng gia tăng. Kho dữ liệu mở khả dụng cho huấn luyện đang co hẹp nhanh chóng. Giai đoạn 2023 -> 2024 ghi nhận khoảng 25% các nguồn dữ liệu huấn luyện hàng đầu đã thiết lập một số hạn chế truy cập. Hệ quả: nguồn dữ liệu huấn luyện trong tương lai sẽ phải phụ thuộc vào các phương thức thu nạp mới (ký hợp đồng cấp phép, tạo dữ liệu tổng hợp, hoặc cơ chế khuyến khích chia sẻ).

### Vị trí của bài học trong Phase 18

Bài 26 là tài liệu hóa ở cấp mô hình. Bài 27 là quản trị ở cấp bộ dữ liệu. Hai bài học cùng nhau định hình tầng minh bạch tổng thể. Bài 28 sẽ lập bản đồ hệ sinh thái nghiên cứu đang giải quyết các câu hỏi này.

```figure
an-provenance-oneway
```

## Áp Dụng Thực Tế

Tệp `code/main.py` tự động tạo một khung tóm tắt bộ dữ liệu gồm 12 trường chuẩn hóa tuân thủ AB 2013 của California cho một bộ dữ liệu mẫu. Bạn có thể điền thông tin vào các trường và quan sát xem những trường nào sẽ kích hoạt các nghĩa vụ pháp lý tiếp theo về quyền riêng tư hoặc bản quyền.

## Đóng Gói Sản Phẩm

Bài học này xây dựng tệp `outputs/skill-provenance-check.md`. Khi tiếp nhận một bộ dữ liệu dùng trong huấn luyện, kỹ năng này sẽ kiểm tra độ bao phủ 12 trường của AB 2013, tính tuân thủ hạ tầng từ chối (opt-out), sự thống nhất với quy định của các cơ quan DPA, và đánh giá rủi ro bất khả đảo ngược.

## Bài Tập

1. Chạy `code/main.py`. Tạo bản tóm tắt 12 trường cho một bộ dữ liệu mẫu và xác định các trường còn thiếu thông tin định lượng cụ thể.

2. Ngoại lệ TDM trong Chỉ thị Bản quyền EU quy định tín hiệu từ chối phải máy đọc được. Hãy đề xuất một định dạng chuẩn cho tín hiệu từ chối này và so sánh nó với `robots.txt` và nhãn xác nhận "No AI Training" của C2PA.

3. Đọc báo cáo "Consent in Crisis" (Tháng 7/2024) của Data Provenance Initiative. Mô tả ba danh mục nội dung đang dựng rào cản hạn chế nhanh nhất và phân tích một hệ quả kinh tế kéo theo.

4. Sự hội tụ quan điểm của các DPA năm 2025 chấp nhận căn cứ lợi ích hợp pháp cho việc huấn luyện trên nội dung công khai. Hãy xây dựng một tình huống thực tế mà căn cứ lợi ích hợp pháp không đủ điều kiện áp dụng, và xác định cơ sở pháp lý thay thế mà đơn vị phát triển bắt buộc phải có.

5. Phác thảo một tệp kê khai nguồn gốc dữ liệu huấn luyện (manifest) kết hợp giữa 12 trường của AB 2013 và chuỗi nguồn gốc có chữ ký số C2PA cho từng bộ dữ liệu. Xác định một rào cản kỹ thuật và một rào cản pháp lý trong quá trình triển khai.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực chất |
|---|---|---|
| AB 2013 | "Luật California" | Luật minh bạch dữ liệu huấn luyện AI tạo sinh; gồm 12 trường bắt buộc |
| TDM exception | "Ngoại lệ khai thác văn bản và dữ liệu" | Ngoại lệ dữ liệu huấn luyện theo Chỉ thị Bản quyền EU đi kèm cơ chế từ chối |
| Legitimate interest | "Lợi ích hợp pháp" | Căn cứ Điều 6 GDPR có thể biện minh cho việc huấn luyện trên nội dung công khai |
| Opt-out signal | "Tín hiệu máy đọc cấm huấn luyện" | `robots.txt`, xác nhận C2PA "No AI Training", TDM.Reservation |
| Irreversibility | "Tính bất khả đảo ngược" | Dữ liệu đã vào trọng số mô hình không thể bóc tách phẫu thuật |
| Unlearning | "Xóa bỏ gần đúng (Quên máy)" | Can thiệp hậu huấn luyện nhằm giảm bớt sự phụ thuộc của mô hình vào dữ liệu cụ thể |
| Consent in Crisis | "Kiểm toán DPI" | Phát hiện tháng 7/2024 về sự bùng nổ các hạn chế bằng `robots.txt` |

## Tài Liệu Đọc Thêm

- [Dự luật California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — Luật minh bạch dữ liệu huấn luyện AI tạo sinh
- [Đạo luật AI EU + Bộ Quy tắc Thực hành GPAI (Bài 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Chương Bản quyền
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, Tháng 7/2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — Kiểm toán của DPI
- [IAPP — Sửa đổi GDPR trong Luật Số Tổng thể của EU (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — Bối cảnh pháp lý

