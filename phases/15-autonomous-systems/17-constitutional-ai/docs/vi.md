# AI Hiến định và Ghi đè Quy tắc (Constitutional AI and Rule Overrides)

> Bản Hiến pháp Claude ngày 22 tháng 1 năm 2026 của Anthropic dài 79 trang và được phát hành dưới giấy phép CC0. Văn kiện này đánh dấu bước chuyển dịch từ căn chỉnh dựa trên quy tắc (rule-based alignment) sang căn chỉnh dựa trên lập luận (reason-based alignment), đồng thời thiết lập một hệ thống phân cấp ưu tiên bốn tầng (four-tier priority hierarchy): (1) an toàn và hỗ trợ sự giám sát của con người, (2) đạo đức, (3) các nguyên tắc chỉ đạo của Anthropic, (4) tính hữu ích. Các hành vi được phân tách thành các điều khoản cấm được mã hóa cứng (hardcoded prohibitions — ví dụ: hỗ trợ vũ khí sinh học, tài liệu lạm dụng tình dục trẻ em CSAM) mà cả người vận hành lẫn người dùng đều không thể ghi đè, và các giá trị mặc định mềm (soft-coded defaults) mà người vận hành có thể điều chỉnh trong các giới hạn định trước. Nghiên cứu nguyên bản năm 2022 (Bai et al.) đã huấn luyện tính vô hại thông qua cơ chế tự phê bình (self-critique) và RLAIF (học tăng cường từ phản hồi của AI) dựa trên một bản hiến pháp. Điểm lưu ý thực tế: căn chỉnh dựa trên lập luận phụ thuộc vào khả năng mô hình khái quát hóa các nguyên lý sang những tình huống chưa từng lường trước. Thí nghiệm có sự tham gia của cộng đồng năm 2023 của chính Anthropic cho thấy độ phân kỳ ~50% giữa các nguyên tắc thu thập từ công chúng và nguyên tắc của doanh nghiệp; phiên bản năm 2026 đã không tích hợp những phát hiện đó.

**Loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ giải quyết ưu tiên bốn tầng)
**Điều kiện tiên quyết:** Phase 15 · 06 (Nghiên cứu căn chỉnh tự động), Phase 15 · 10 (Chế độ cấp quyền)
**Thời lượng:** ~60 phút

## Vấn đề đặt ra (The Problem)

Một agent khi triển khai thực tế sẽ tiếp nhận những đầu vào mà các nhà thiết kế chưa từng thấy trước đó. Không có danh sách quy tắc nào đủ dài để bao quát mọi trường hợp. Cũng không có danh sách quy tắc nào đủ ngắn để áp dụng nhanh chóng dưới áp lực tài nguyên tính toán khi suy luận. Câu hỏi thực tiễn: làm thế nào để căn chỉnh một agent theo các nguyên lý có thể đứng vững trước phân bố đuôi dài (long tail) của các tình huống thực tế lẫn tốc độ suy luận nhanh chóng?

Căn chỉnh dựa trên quy tắc (Rule-Based Alignment - RBA): liệt kê mọi hành vi bị cấm. Cách này kiểm tra rất nhanh, dễ kiểm toán, nhưng không thể duy trì tính cập nhật, và thường từ chối thái quá (over-refuse) đối với các trường hợp tương tự gần gũi ngoài dự tính. Căn chỉnh dựa trên lập luận (Reason-Based Alignment — như Hiến pháp Claude 2026): mã hóa các nguyên lý nền tảng và để mô hình tự lập luận. Cách tiếp cận này mở rộng tốt trên các tình huống chưa từng gặp, nhưng khó kiểm toán hơn, và dạng lỗi hỏng hóc là áp dụng sai nguyên lý thay vì bỏ sót quy tắc.

Hiến pháp năm 2026 chọn một vị thế trung dung rõ ràng. Các điều khoản cấm được mã hóa cứng (hardcoded prohibitions) — những hành vi mà tính sai trái của chúng không phụ thuộc vào ngữ cảnh (hỗ trợ phát triển vũ khí sinh học, CSAM) — áp dụng RBA: tuyệt đối không bao giờ thực hiện, bất kể chỉ thị từ người vận hành hay người dùng. Mọi hành vi khác đều dựa trên lập luận bên trong một hệ thống phân cấp bốn tầng: an toàn và hỗ trợ giám sát của con người đứng đầu; đạo đức đứng thứ hai; các chỉ dẫn do Anthropic ban hành đứng thứ ba; và tính hữu ích xếp cuối cùng. Người vận hành có thể tùy chỉnh các mặc định bên trong vùng mã hóa mềm nhưng tuyệt đối không thể can thiệp vào các điều khoản cấm được mã hóa cứng.

## Khái niệm cốt lõi (The Concept)

### Hệ thống phân cấp ưu tiên bốn tầng (The four-tier priority hierarchy)

1. **An toàn và hỗ trợ sự giám sát của con người (Safety and supporting human oversight).** Ưu tiên cao nhất. Mô hình ưu tiên việc không làm suy yếu khả năng của con người và của Anthropic trong việc giám sát và hiệu chỉnh AI. Điều này không chỉ là "hãy thận trọng"; mà cụ thể là "không được hành động theo những cách khiến việc giám sát của con người trở nên khó khăn hơn."
2. **Đạo đức (Ethics).** Trung thực, tránh gây hại cho con người, không lừa dối, không thao túng tâm lý. Tầng này chiếm ưu thế tối cao trước các chỉ dẫn của Anthropic khi xảy ra xung đột.
3. **Các nguyên tắc chỉ đạo của Anthropic (Anthropic guidelines).** Các chuẩn mực vận hành mà Anthropic cho là quan trọng: phạm vi sản phẩm, khuôn mẫu tương tác, dùng công cụ nào trong hoàn cảnh nào.
4. **Tính hữu ích (Helpfulness).** Ưu tiên thấp nhất. Trở nên hữu ích nhất có thể trong khuôn khổ tuân thủ các tầng ưu tiên cao hơn.

Khi các tầng xung đột nhau, tầng cao hơn sẽ chiến thắng. Cấu trúc này tương tự như mức ưu tiên tiến trình trong Unix hay chất lượng dịch vụ mạng QoS — thiết kế này nhằm tạo ra cơ chế giải quyết xung đột có thể dự đoán được, chứ không nhất thiết hướng tới hành vi tối ưu trên bất kỳ một trục đơn lẻ nào.

### Điều khoản cấm mã hóa cứng so với Mặc định mã hóa mềm (Hardcoded prohibitions vs soft-coded defaults)

**Mã hóa cứng (Hardcoded):**
- Phát triển / gia tăng vũ khí sinh học hoặc vũ khí CBRN (hóa học, sinh học, phóng xạ, hạt nhân)
- Tài liệu lạm dụng tình dục trẻ em (CSAM)
- Tấn công vào cơ sở hạ tầng trọng yếu
- Đánh lừa người dùng về danh tính của mô hình khi được hỏi trực tiếp

Người vận hành không thể ghi đè những điều này. Người dùng không thể ghi đè những điều này. Chúng được thực thi ở cấp độ trọng số mô hình nếu có thể (thông qua huấn luyện RLHF / AI Hiến định) và ở tầng suy luận (inference layer) khi cần thiết.

**Mặc định mã hóa mềm (Soft-coded defaults — người vận hành có thể điều chỉnh):**
- Mặc định về độ dài câu trả lời
- Phạm vi chủ đề (mô hình có thể từ chối các chủ đề nằm ngoài phạm vi triển khai của bên vận hành)
- Phong cách giao tiếp (trang trọng so với thân mật)
- Các mẫu hình sử dụng công cụ

Sự điều chỉnh của người vận hành diễn ra bên trong các giới hạn đã được công bố. Người vận hành không thể vô hiệu hóa các điều khoản cấm mã hóa cứng bằng cách đổi tên gọi của chúng.

### Quy trình huấn luyện CAI năm 2022

Bài báo nguyên bản về AI Hiến định (Bai et al., 2022) huấn luyện tính vô hại (harmlessness):

1. Tạo ra các phản hồi cho một tập hợp các lời nhắc (prompts).
2. Yêu cầu mô hình tự phê bình từng phản hồi dựa trên một bản hiến pháp (các nguyên lý rõ ràng).
3. Hiệu đính phản hồi dựa trên lời tự phê bình đó.
4. Áp dụng RLAIF (học tăng cường từ phản hồi của AI) trên các cặp dữ liệu đã hiệu đính.

Kết quả: một mô hình từ chối các yêu cầu có hại đi kèm lời giải thích dựa trên nguyên lý, thay vì đưa ra lời từ chối chung chung cứng nhắc. Bản Hiến pháp năm 2026 sử dụng một biến thể kế thừa của phương pháp huấn luyện này kết hợp với giai đoạn hậu huấn luyện (post-training) bổ sung về hệ thống phân cấp tầng rõ ràng.

### Căn chỉnh dựa trên lập luận bắt được gì và bỏ sót gì

**Bắt được (Catches):**
- Sự kết hợp ngoài dự tính của các thao tác nguyên hàm được phép nhưng khi ráp lại thì nguyên lý bị vi phạm rõ ràng.
- Các yêu cầu mới lạ nhưng là các biến thể tương đương gần gũi của các hành vi bị cấm.
- Các cuộc tấn công phi kỹ thuật (social engineering) dựa vào kẽ hở "bạn đâu có nói điều X bị cấm."

**Bỏ sót (Misses):**
- Các cuộc tấn công khai thác sự mơ hồ của nguyên lý ("người dùng đã yêu cầu điều này nên tính hữu ích đòi hỏi phải đáp ứng").
- Các kịch bản mà hai nguyên lý xung đột theo cách chưa lường trước, và thứ tự phân tầng trở nên không rõ ràng.
- Sự trôi dạt chậm trong cách diễn giải nguyên lý qua các chu kỳ huấn luyện (hiện tượng tái diễn giải - reinterpretation).

### Thí nghiệm có sự tham gia của cộng đồng năm 2023

Anthropic đã tiến hành một thí nghiệm vào năm 2023 so sánh bản hiến pháp do doanh nghiệp soạn thảo với bản hiến pháp được tạo ra từ đóng góp của công chúng (~1.000 người trả lời tại Hoa Kỳ). Hai phiên bản đồng thuận trên khoảng 50% các nguyên lý. Ở những điểm phân kỳ, phiên bản từ công chúng tỏ ra khắt khe hơn ở một số vấn đề (xử lý nội dung chính trị) và ít khắt khe hơn ở các vấn đề khác (tự công khai danh tính AI). Bản Hiến pháp năm 2026 đã không tích hợp các phát hiện từ công chúng này. Đây là một mâu thuẫn đã được ghi nhận thành tài liệu trong phương pháp tiếp cận.

### Tại sao các điều khoản cấm mã hóa cứng là tất yếu

Chỉ riêng căn chỉnh dựa trên lập luận thì không thể bịt kín được phân bố đuôi dài. Một kẻ tấn công thuyết phục được mô hình chấp nhận một tiền đề giả định (ví dụ: "chúng tôi là một phòng thí nghiệm nghiên cứu vũ khí sinh học được cấp phép") thường có thể vượt qua các nguyên lý vốn phụ thuộc vào lập luận tình huống. Các điều khoản cấm mã hóa cứng không bao giờ bị bẻ cong trước cách đặt vấn đề của tiền đề. Chúng chính là các "giới hạn hiến định cứng" (hard constitutional limit) của Bài 14 tại tầng căn chỉnh.

### Vị trí của Hiến pháp trong ngăn xếp hệ thống

Bản Hiến pháp không phải là cơ chế kill switch của Bài 14. Nó tồn tại ở tầng mô hình: những gì mà trọng số mô hình được huấn luyện để ưu tiên. Cơ chế kill switch và canary token tồn tại ở tầng runtime: những gì mà môi trường runtime cho phép thực thi. Cả hai đều bắt buộc phải có. Một runtime kích hoạt mọi hành động sai trái do trọng số mô hình quá dễ dãi là một sự cố runtime. Một mô hình từ chối mọi hành động đúng đắn do runtime quá khắt khe cũng là một sự cố runtime. Mỗi tầng giải quyết các phân lớp rủi ro khác nhau.

```figure
mx-priority-tiers
```

## Thực hành (Use It)

Tệp `code/main.py` hiện thực hóa một bộ giải quyết ưu tiên bốn tầng tối giản. Bộ giải quyết tiếp nhận một hành động đề xuất và một tập hợp đánh giá nguyên lý (an toàn, đạo đức, nguyên tắc chỉ đạo, tính hữu ích), sau đó trả về quyết định cho phép hành động, từ chối, hoặc đưa ra hành động đã sửa đổi. Trình điều khiển chạy một tập hợp các trường hợp: cho phép rõ ràng, từ chối rõ ràng, điều khoản cấm mã hóa cứng, và trường hợp mơ hồ xung đột giữa các tầng.

## Đóng gói (Ship It)

Tệp `outputs/skill-constitution-review.md` kiểm toán tầng hiến định của một phương án triển khai: những gì được mã hóa cứng, những gì được mã hóa mềm, người vận hành có thể tinh chỉnh ở đâu, và liệu hệ thống phân cấp bốn tầng có thực sự là thứ tự giải quyết xung đột hay không.

## Bài tập (Exercises)

1. Chạy `code/main.py`. Xác nhận rằng điều khoản cấm mã hóa cứng luôn được kích hoạt ngay cả khi tính hữu ích được đánh giá rất cao. Sửa đổi bộ giải quyết để gán trọng số tính hữu ích cao hơn đạo đức; quan sát dạng thức lỗi hỏng hóc phát sinh.

2. Đọc bản Hiến pháp Claude (tài liệu công khai, 79 trang, CC0). Xác định một nguyên lý mà bạn cho rằng chưa được chỉ định đầy đủ. Viết hai đoạn văn giải thích sự mơ hồ cụ thể đó và đề xuất một cách diễn đạt chặt chẽ hơn.

3. Thiết kế một tập hợp các mặc định mã hóa mềm cho một agent hỗ trợ khách hàng. Người vận hành được phép điều chỉnh những gì? Người vận hành tuyệt đối không được chạm vào những gì? Biện minh cho từng ranh giới phân định.

4. Đọc bài báo CAI năm 2022 của Bai et al. Mô tả một trường hợp mà vòng lặp tự phê bình và hiệu đính của AI Hiến định có thể dẫn đến kết quả tồi tệ hơn so với một quy tắc cứng nhắc tuyệt đối. Nêu rõ phân lớp sự cố đó.

5. Thí nghiệm có sự tham gia của cộng đồng năm 2023 của Anthropic nhận thấy sự phân kỳ ~50% giữa các nguyên tắc của công chúng và doanh nghiệp. Hãy chọn một danh mục mà sự khác biệt này mang tính quyết định đối với việc triển khai thực tế (ví dụ: tính trung lập chính trị). Đề xuất một thiết kế cho phép bên vận hành thể hiện các giá trị riêng của họ trong khi các điều khoản cấm mã hóa cứng vẫn được giữ nguyên vẹn.

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Constitutional AI | "Phương pháp căn chỉnh Anthropic" | Tự phê bình + RLAIF dựa trên một bản hiến pháp thành văn |
| Reason-based alignment | "Nguyên lý thay vì quy tắc" | Mô hình lập luận trên các nguyên lý để xử lý các tình huống chưa từng thấy |
| Hardcoded prohibition | "Tuyệt đối không làm X" | Quy tắc cấm tuyệt đối mà không người vận hành hay người dùng nào có thể ghi đè |
| Soft-coded default | "Có thể tinh chỉnh" | Hành vi nằm trong giới hạn công bố, thuộc quyền kiểm soát của người vận hành |
| Four-tier hierarchy | "Thứ tự ưu tiên" | An toàn > Đạo đức > Nguyên tắc chỉ đạo > Tính hữu ích |
| RLAIF | "Học tăng cường phản hồi AI" | Học tăng cường trong đó phần thưởng đến từ các phê bình do chính mô hình tạo ra |
| Participatory constitution | "Hiến pháp từ công chúng" | Thí nghiệm năm 2023 của Anthropic; phân kỳ ~50% so với bản doanh nghiệp |
| Principle drift | "Trôi dạt diễn giải" | Sự biến đổi dần trong cách mô hình diễn giải văn bản nguyên lý cố định |

## Đọc thêm (Further Reading)

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — tài liệu CC0 dài 79 trang.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — bài báo gốc năm 2022.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — thí nghiệm có sự tham gia của công chúng.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — vị trí của Hiến pháp trong ngăn xếp chính sách mở rộng có trách nhiệm.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — vai trò của Hiến pháp trong các đợt triển khai chân trời dài hạn.
