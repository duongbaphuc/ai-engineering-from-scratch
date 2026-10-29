# Chính sách Mở rộng Quy mô Có trách nhiệm của Anthropic phiên bản v3.0 (Anthropic Responsible Scaling Policy v3.0)

> Chính sách Mở rộng Quy mô Có trách nhiệm (Responsible Scaling Policy - RSP) phiên bản v3.0 có hiệu lực từ ngày 24 tháng 2 năm 2026, thay thế chính sách năm 2023. Kế hoạch giảm thiểu rủi ro theo hai tầng (two-tier mitigation): những gì Anthropic sẽ đơn phương thực hiện so với những gì được định hình như khuyến nghị toàn ngành (bao gồm các tiêu chuẩn an ninh bảo mật RAND SL-4). Chính sách bổ sung Lộ trình An toàn Biên giới (Frontier Safety Roadmaps) và Báo cáo Rủi ro (Risk Reports) dưới dạng các tài liệu thường trực thay vì các ấn phẩm phát hành một lần. Loại bỏ cam kết tạm dừng huấn luyện (pause commitment) của năm 2023. Đưa ra ngưỡng AI R&D-4: một khi vượt qua ngưỡng này, Anthropic bắt buộc phải công bố một lập luận khẳng định (affirmative case) xác định các rủi ro lệch lạc căn chỉnh và các biện pháp giảm thiểu. Claude Opus 4.6 chưa vượt qua ngưỡng này. Anthropic tuyên bố trong thông cáo phát hành v3.0 rằng "việc tự tin loại trừ khả năng này đang ngày càng trở nên khó khăn." Tổ chức SaferAI từng chấm điểm RSP năm 2023 đạt 2,2; họ đã hạ điểm v3.0 xuống 1,9, đưa Anthropic vào nhóm RSP "yếu" cùng với OpenAI và DeepMind. Các ngưỡng định tính đã thay thế các cam kết định lượng năm 2023; việc xóa bỏ điều khoản tạm dừng là bước thụt lùi rõ nét nhất.

**Type:** Learn  
**Languages:** Python (stdlib, động cơ ra quyết định theo ngưỡng RSP)  
**Prerequisites:** Phase 15 · 06 (AAR - Nghiên cứu Căn chỉnh Tự động), Phase 15 · 07 (RSI - Tự cải tiến Đệ quy)  
**Time:** ~45 phút

## Vấn đề

Các phòng thí nghiệm AI tiên phong (frontier labs) công bố các chính sách mở rộng quy mô đóng vai trò một phần là tài liệu kỹ thuật, một phần là tài liệu quản trị, và một phần là tín hiệu gửi tới các nhà quản lý chính sách. RSP v3.0 là tài liệu hiện hành của Anthropic. Việc đọc kỹ văn bản này không phải vì việc tuân thủ nó mang tính ràng buộc pháp lý (thực tế là không), mà bởi vì cách đóng khung vấn đề của nó định hình cách một phòng nghiên cứu nhận thức về rủi ro thảm họa và cách họ truyền đạt sự đánh đổi tới công chúng.

Phần khác biệt (diff) giữa bản v3.0 so với v2.0 chính là đối tượng nghiên cứu hữu ích. Những gì được bổ sung: Lộ trình An toàn Biên giới, Báo cáo Rủi ro, và ngưỡng AI R&D-4. Những gì bị loại bỏ: cam kết tạm dừng huấn luyện của năm 2023. Những gì được đóng khung lại: kế hoạch giảm thiểu rủi ro hai tầng phân định giữa hành động đơn phương của Anthropic và khuyến nghị cho toàn ngành. Đơn vị đánh giá độc lập bên ngoài — SaferAI — đã hạ điểm số từ 2,2 (v2) xuống 1,9 (v3.0). Đây là ví dụ điển hình cho thấy một chính sách mở rộng quy mô có thể giảm bớt tính chặt chẽ trong khi bề ngoài trông có vẻ trau chuốt hơn.

## Khái niệm

### Kế hoạch giảm thiểu rủi ro hai tầng (The two-tier mitigation schedule)

- **Hành động đơn phương của Anthropic (Anthropic unilateral actions)**: những gì Anthropic cam kết thực hiện bất kể các phòng thí nghiệm khác làm gì. Bao gồm việc dừng huấn luyện khi vượt qua ngưỡng năng lực nhất định, các biện pháp an ninh bảo mật cụ thể, và các cổng kiểm duyệt triển khai nghiêm ngặt.
- **Khuyến nghị toàn ngành (Industry-wide recommendations)**: những gì Anthropic cho rằng toàn bộ ngành công nghiệp nên cùng nhau thực hiện. Bao gồm các tiêu chuẩn an ninh RAND SL-4. Đây không phải là cam kết từ phía Anthropic; chúng là sự vận động chính sách.

Cấu trúc hai tầng này không hề tồn tại trong v2. Điều đó có nghĩa là người đọc cần phải quan sát kỹ từng cam kết nằm ở cột nào. Một biện pháp an ninh nằm trong cột "khuyến nghị toàn ngành" không phải là lời hứa của Anthropic; đó chỉ là mong ước của Anthropic.

### Ngưỡng AI R&D-4

Đây là cấp độ năng lực mà RSP v3.0 chỉ định là ngưỡng quan trọng tiếp theo. Cụ thể: một mô hình có khả năng tự động hóa một phần đáng kể công việc nghiên cứu AI với chi phí cạnh tranh. Một khi Anthropic tin rằng một mô hình vượt qua ngưỡng này, họ bắt buộc phải công bố một lập luận khẳng định (affirmative case) xác định các rủi ro lệch lạc căn chỉnh và các biện pháp giảm thiểu thỏa đáng trước khi tiếp tục mở rộng quy mô huấn luyện.

Claude Opus 4.6 chưa vượt qua ngưỡng này theo thông cáo của v3.0. Tài liệu bổ sung thêm: "việc tự tin loại trừ khả năng này đang ngày càng trở nên khó khăn." Cách diễn đạt này rất đáng chú ý; nó thừa nhận rằng ngưỡng này đã đủ gần để trở thành một mối bận tâm hiện hữu, chứ không còn là một giới hạn mang tính suy đoán xa xôi.

Bài 6 (Nghiên cứu Căn chỉnh Tự động - Automated Alignment Research) và Bài 7 (Tự cải tiến Đệ quy - Recursive Self-Improvement) kết nối trực tiếp vào ngưỡng này. Việc các hệ thống nghiên cứu căn chỉnh tự động vượt qua các tiêu chuẩn chất lượng nghiên cứu của con người là bằng chứng cho thấy ngưỡng AI R&D-4 đang đến rất gần.

### Lộ trình An toàn Biên giới và Báo cáo Rủi ro

Bản v3.0 nâng tầm hai loại tài liệu sau thành các văn bản thường trực:

- **Lộ trình An toàn Biên giới (Frontier Safety Roadmap)**: tài liệu hướng về tương lai mô tả các hoạt động an toàn được lên kế hoạch, kỳ vọng về năng lực mô hình, và các nghiên cứu giảm thiểu rủi ro.
- **Báo cáo Rủi ro (Risk Report)**: tài liệu nhìn lại quá khứ đánh giá các mô hình cụ thể sau khi phát hành, mô tả năng lực đã quan sát được và rủi ro tồn dư.

Cả hai đều được công khai minh bạch. Cả hai đều được cập nhật theo chu kỳ đã công bố. Lợi ích thực tế là: người đọc có thể theo dõi xem những gì Anthropic tuyên bố sẽ làm trong Lộ trình đối chiếu với những gì họ báo cáo trong Báo cáo Rủi ro ra sao.

### Việc loại bỏ điều khoản tạm dừng (Removing the pause clause)

Bản RSP năm 2023 bao gồm một cam kết tạm dừng rõ ràng: nếu một mô hình vượt qua các ngưỡng năng lực cụ thể, quá trình huấn luyện sẽ tạm dừng cho đến khi các biện pháp giảm thiểu được áp dụng đầy đủ. Bản v3.0 thay thế cam kết tạm dừng rõ ràng đó bằng một cách diễn đạt mềm mỏng hơn (công bố một lập luận khẳng định, tiếp tục nếu các biện pháp giảm thiểu là thỏa đáng). SaferAI và các nhà phân tích khác đã chỉ trích trực diện điều này như một bước thụt lùi nghiêm trọng nhất trong tài liệu mới.

Lập luận chính sách cho sự thay đổi này: các ngưỡng định lượng vào năm 2023 hóa ra không thể đo lường chạm tới được bởi các bộ điểm chuẩn năng lực của kỷ nguyên 2026, do chính các bộ điểm chuẩn đó đã bị co giãn thang đo lại. Lập luận phản bác: một điều khoản tạm dừng trong chính sách mở rộng quy mô là một cơ chế ràng buộc cam kết (commitment device); việc loại bỏ nó đã tước đi độ tin cậy của chính sách.

### Sự hạ bậc điểm số của SaferAI

SaferAI là một tổ chức độc lập chuyên đánh giá các văn bản chính sách kiểu RSP. Điểm số công khai của họ: bản RSP năm 2023 của Anthropic đạt 2,2 (trên thang điểm trong đó 4,0 là mức RSP tốt nhất hiện nay và 1,0 là mức danh nghĩa hình thức). Bản v3.0 bị chấm 1,9. Điểm số này chuyển Anthropic từ nhóm "trung bình" xuống nhóm "yếu", đứng chung hạng với OpenAI và DeepMind trong nhóm yếu.

Các yếu tố dẫn đến sự hạ điểm theo SaferAI:
- Các ngưỡng định tính thay thế cho các ngưỡng định lượng.
- Cam kết tạm dừng huấn luyện bị loại bỏ.
- Các biện pháp giảm thiểu cho ngưỡng AI R&D-4 chỉ được mô tả chung chung là "lập luận khẳng định" thay vì các biện pháp kỹ thuật cụ thể.
- Cơ chế giám sát phụ thuộc vào Nhóm Cố vấn An toàn nội bộ của Anthropic, với sự giám sát độc lập bên ngoài rất hạn chế.

### Bài học này không nhằm mục đích gì

Đây không phải là một bài học về tuân thủ pháp lý. RSP v3.0 không phải là luật định; không có cơ chế cưỡng chế nào bắt buộc Anthropic phải tuân theo. Mục tiêu của bài học là rèn luyện khả năng đọc văn bản với sự chi tiết và hoài nghi khoa học cần thiết. Các chính sách mở rộng quy mô là tín hiệu công khai hàng đầu mà các phòng thí nghiệm biên giới phát ra về lập trường đối với rủi ro thảm họa. Đọc hiểu tường tận chúng là một kỹ năng thực tế cho bất kỳ ai có công việc phụ thuộc vào các năng lực AI tiên phong.

```figure
a5-rsp-ladder
```

## Sử dụng

Tệp `code/main.py` triển khai một động cơ ra quyết định nhỏ mô phỏng cấu trúc đánh giá ngưỡng của RSP: tiếp nhận một mô hình ứng viên và một tập các phép đo năng lực, trả về kết quả liệu ngưỡng AI R&D-4 có bị vượt qua hay không, các phần bắt buộc phải có trong lập luận khẳng định, và liệu việc triển khai có được phép tiến hành hay không. Mã nguồn được thiết kế đơn giản có chủ đích nhằm làm sáng tỏ logic cốt lõi của tài liệu chính sách.

## Đóng gói

Tệp `outputs/skill-scaling-policy-review.md` dùng để đánh giá một chính sách mở rộng quy mô (của Anthropic, OpenAI, DeepMind, hoặc nội bộ tổ chức) dựa trên khung tham chiếu v3.0: cấu trúc hai tầng, các ngưỡng năng lực, cam kết tạm dừng và cơ chế giám sát độc lập.

## Bài tập

1. Chạy `code/main.py`. Đưa vào ba mô hình tổng hợp ở các mức năng lực khác nhau. Xác nhận bộ đánh giá ngưỡng hoạt động đúng như mong đợi và tạo ra đúng biểu mẫu lập luận khẳng định.

2. Đọc toàn văn RSP v3.0 (32 trang). Xác định tất cả các cam kết nằm trong tầng "khuyến nghị toàn ngành". Những cam kết nào trong số đó từng là "đơn phương từ phía Anthropic" trong bản v2?

3. Đọc phương pháp chấm điểm RSP của SaferAI. Tái lập điểm số 1,9 của họ cho bản v3.0 bằng cách áp dụng tiêu chí chấm điểm vào văn bản. Tiêu chí nào đóng góp nhiều nhất vào việc hạ điểm?

4. Cam kết tạm dừng năm 2023 đã bị loại bỏ. Hãy đề xuất một cam kết thay thế vừa bảo toàn được độ tin cậy của chính sách, vừa thừa nhận bài toán co giãn thang đo điểm chuẩn của năm 2026.

5. So sánh RSP v3.0 với Khung Chuẩn bị (Preparedness Framework) v2 của OpenAI (Bài 20). Chọn một khía cạnh mà v3.0 chặt chẽ hơn. Chọn một khía cạnh mà Khung Chuẩn bị vượt trội hơn.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| RSP | "Chính sách mở rộng của Anthropic" | Responsible Scaling Policy (Chính sách Mở rộng Quy mô Có trách nhiệm); bản v3.0 có hiệu lực từ ngày 24/02/2026 |
| AI R&D-4 | "Ngưỡng tự động hóa nghiên cứu" | Năng lực tự động hóa một phần đáng kể công việc nghiên cứu AI với chi phí cạnh tranh |
| Lập luận khẳng định (Affirmative case) | "Bản biện giải an toàn" | Tài liệu lập luận công khai chứng minh rủi ro đã được nhận diện và biện pháp giảm thiểu là thỏa đáng |
| Lộ trình An toàn Biên giới (Frontier Safety Roadmap) | "Kế hoạch tương lai" | Tài liệu thường trực công bố kế hoạch an toàn và kỳ vọng về năng lực sắp tới |
| Báo cáo Rủi ro (Risk Report) | "Đánh giá hồi cứu mô hình" | Tài liệu thường trực đánh giá năng lực thực tế và rủi ro tồn dư sau khi phát hành mô hình |
| Giảm thiểu hai tầng (Two-tier mitigation) | "Đơn phương vs Toàn ngành" | Phân định rõ giữa cam kết của Anthropic và khuyến nghị cho các bên khác |
| Cam kết tạm dừng (Pause commitment) | "Điều khoản năm 2023" | Lời hứa rõ ràng về việc tạm dừng huấn luyện; đã bị loại bỏ trong bản v3.0 |
| Đánh giá của SaferAI | "Điểm số RSP độc lập" | Thang điểm của bên thứ ba; bản v3.0 đạt 1,9 (bản v2 đạt 2,2) |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — toàn văn chính sách 32 trang.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — tóm tắt những thay đổi so với bản v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/responsible-scaling-policy/roadmap) — tài liệu thường trực liên kết từ RSP v3.0.
- [Anthropic — Risk Report: February 2026](https://www.anthropic.com/feb-2026-risk-report) — báo cáo hồi cứu về Claude Opus 4.6, mô hình biên giới tại thời điểm xuất bản.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — liên kết ngưỡng AI R&D-4 với các phép đo quyền tự trị thực tế.
