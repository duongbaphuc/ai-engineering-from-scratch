# WMDP và Đánh giá năng lực lưỡng dụng (Dual-Use Capability Evaluation)

> Li và cộng sự, "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). Gồm 4.157 câu hỏi trắc nghiệm về an ninh sinh học (1.520), an ninh mạng (2.225) và hóa học (412). Các câu hỏi nằm trong "vùng vàng" (yellow zone), tức tri thức tiệm cận có tính hỗ trợ, được lọc qua thẩm định của nhiều chuyên gia và tuân thủ pháp lý ITAR/EAR. Mục đích kép: đánh giá gián tiếp (proxy) năng lực lưỡng dụng, và làm bộ chuẩn cho bài toán quên học (unlearning), trong đó phương pháp đi kèm RMU làm giảm hiệu năng trên WMDP nhưng vẫn giữ năng lực tổng quát. Diễn biến 2024-2025: các đánh giá đầu năm 2024 của OpenAI/Anthropic báo cáo "mức hỗ trợ nhẹ" (mild uplift) so với tìm kiếm trên internet; đến tháng 4/2025, Preparedness Framework v2 của OpenAI cho rằng các mô hình đang "ở ngưỡng giúp người mới thực sự tạo ra các mối đe dọa sinh học đã biết". Thử nghiệm mua sắm vũ khí sinh học của Anthropic cho thấy mức hỗ trợ 2,53 lần, chưa đủ để loại trừ ASL-3.

**Loại:** Learn
**Ngôn ngữ:** Python (stdlib, bộ khung đánh giá uplift theo dạng WMDP)
**Điều kiện tiên quyết:** Phase 18 · 16 (công cụ red-team), Phase 14 (kỹ thuật agent)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả ba lĩnh vực của WMDP, số lượng câu hỏi và tiêu chí lọc "vùng vàng".
- Giải thích RMU và lý do WMDP vừa là bộ đánh giá vừa là bộ chuẩn quên học.
- Mô tả diễn biến uplift 2024-2025: "mild uplift" -> "on the cusp" -> "chưa đủ để loại trừ ASL-3".
- Phân biệt uplift tương đối theo người mới với năng lực tuyệt đối của chuyên gia.

## Vấn đề

Năng lực lưỡng dụng là bài toán đo lường nằm dưới mọi khung an toàn tiên phong (frontier safety framework) của các phòng thí nghiệm (Bài 18). Câu hỏi đặt ra: mô hình X có thúc đẩy đáng kể khả năng gây tổn hại hàng loạt của một người mới trong sinh học, hóa học hay mạng không? Đo trực tiếp (yêu cầu mô hình thực sự tạo ra tác hại) là bất hợp pháp và phi đạo đức. Đo gián tiếp cần một bộ chuẩn mà mô hình không thể từ chối (để có số liệu năng lực trung thực) nhưng bản thân các câu hỏi không phải là ấn phẩm gây hại.

## Khái niệm

### "Vùng vàng" (yellow zone)

Là những câu hỏi đòi hỏi tri thức tiệm cận, có tính hỗ trợ về một quy trình gây hại nhưng không phải công thức tổng hợp trực tiếp. Ví dụ "Thuốc thử nào xúc tác bước 4 của [lộ trình đã công bố]?" chứ không phải "làm sao tôi tạo ra [hợp chất nguy hiểm]?". Mỗi câu hỏi được nhiều chuyên gia lĩnh vực thẩm định và lọc theo tuân thủ kiểm soát xuất khẩu ITAR/EAR.

Tổng cộng 4.157 câu hỏi:
- An ninh sinh học: 1.520
- An ninh mạng: 2.225
- Hóa học: 412

Định dạng trắc nghiệm. Mô hình trả lời mà không bị yêu cầu hỗ trợ việc gì; có thể đo năng lực mà không khơi gợi hành vi gây hại.

### RMU, tức Representation Misdirection for Unlearning (chuyển hướng biểu diễn để quên học)

Phương pháp quên học đi kèm. Áp dụng cho LLaMa-2-7B, nó đưa điểm WMDP về gần mức ngẫu nhiên trong khi giữ MMLU và các bộ chuẩn năng lực tổng quát khác chỉ lệch vài điểm phần trăm. Phương pháp này là đường cơ sở (baseline) quên học cho mọi bài báo về quên học sinh-hóa-mạng sau đó.

### Diễn biến uplift 2024-2025

Ba giai đoạn:

1. **"Mild uplift" năm 2024.** Các đánh giá Preparedness/RSP đầu tiên của OpenAI và Anthropic báo cáo lợi thế nhỏ so với tìm kiếm internet đối với người mới thực hiện các tác vụ gần với sinh học. Thông điệp công khai: mô hình tiên phong có giúp ích nhưng không hơn Google đáng kể.

2. **Tháng 4/2025 "on the cusp".** Preparedness Framework v2 của OpenAI báo cáo các mô hình "đang ở ngưỡng giúp người mới thực sự tạo ra các mối đe dọa sinh học đã biết". Đây không phải khẳng định về năng lực mà là cảnh báo rằng ngưỡng đã gần kề.

3. **Thử nghiệm mua sắm vũ khí sinh học của Anthropic năm 2025.** Nghiên cứu có đối chứng với người tham gia là người mới, đo thành công tương đối ở các tác vụ giai đoạn mua sắm. Báo cáo uplift 2,53 lần. Chưa đủ để loại trừ ASL-3 (Bài 18), tức ngưỡng cho tầng 3 của Responsible Scaling Policy của Anthropic đã đạt hoặc đang tiến gần.

### Tương đối theo người mới và tuyệt đối theo chuyên gia

Một phân biệt then chốt:

- **Uplift tương đối theo người mới (novice-relative uplift).** Mô hình giúp người không chuyên được bao nhiêu? Mang tính nhân. Lợi thế tương đối cao vì người mới biết rất ít; thông tin khiêm tốn cũng có ích.
- **Năng lực tuyệt đối của chuyên gia (expert-absolute capability).** Mô hình tạo ra bao nhiêu thông tin khi nỗ lực tối đa? Chuyên gia khai thác được nhiều hơn người mới. Trần tuyệt đối rất cao.

Các hồ sơ an toàn (safety case, Bài 18) nhắm cả hai: "mô hình không thể cho người mới đủ hỗ trợ để thực hiện" cộng với "chuyên gia không thể khai thác từ mô hình thông tin nào chưa được công bố".

### Cạm bẫy đo lường

WMDP là chỉ báo gián tiếp về năng lực, không phải phép đo khi triển khai. Mô hình đạt điểm WMDP cao có thể bị người mới khai thác trong thực tế hoặc không, tùy vào:
- Khả năng kháng khơi gợi (elicitation resistance): khó đến mức nào để lấy được năng lực ra mà không kích hoạt bộ lọc an toàn
- Tri thức ngầm (tacit knowledge): năng lực đòi hỏi kỹ năng phòng thí nghiệm, không chỉ thông tin
- Rào cản thực thi: mua sắm, thiết bị

Thử nghiệm mua sắm vũ khí sinh học của Anthropic năm 2025 bổ sung lớp khơi gợi bởi người mới lên trên năng lực kiểu WMDP: nó đo thành công thực tế của tác vụ, không phải năng lực trắc nghiệm.

### Vị trí trong Phase 18

Các bài 12-16 là công cụ tấn công và phòng thủ trên đầu ra của mô hình. Bài 17 là tầng năng lực lưỡng dụng, tức phép đo mà các khung an toàn tiên phong (Bài 18) đánh giá. Bài 30 khép lại mạch này với bằng chứng uplift mạng/sinh/hóa/hạt nhân hiện hành năm 2026.

```figure
al-wmdp-yellow-zone
```

## Sử dụng

`code/main.py` dựng một bộ khung đánh giá đồ chơi theo dạng WMDP. Một mô hình giả được kiểm tra trên các câu hỏi chia theo nhóm; điểm theo từng lĩnh vực được báo cáo. Một can thiệp quên học đơn giản (xóa biểu diễn đặc thù lĩnh vực) làm giảm điểm; bạn có thể đo đánh đổi so với năng lực tổng quát.

## Triển khai

Bài học này tạo ra `outputs/skill-wmdp-eval.md`. Với một tuyên bố về năng lực lưỡng dụng ("mô hình của chúng tôi không giúp đáng kể cho vũ khí sinh học"), nó kiểm toán: đã chạy những bộ chuẩn nào, đường từ chối nào được dùng khi đánh giá (hoàn thành thô hay có cổng chính sách), và các nghiên cứu khơi gợi với người mới có bổ sung cho kết quả trắc nghiệm hay không.

## Bài tập

1. Chạy `code/main.py`. Báo cáo độ chính xác theo từng lĩnh vực trước và sau bước quên học đồ chơi. Giải thích đánh đổi về năng lực tổng quát.

2. Mở rộng WMDP đồ chơi với lĩnh vực thứ tư (ví dụ phóng xạ). Nêu hai loại câu hỏi minh họa trong vùng vàng. Giải thích vì sao soạn các câu hỏi này khó hơn thêm câu hỏi dạng MMLU.

3. Đọc Mục 5 của WMDP 2024 (phương pháp RMU). Phác thảo một cách quên học đơn giản hơn (ví dụ triệt top-k nơ-ron cho nội dung của lĩnh vực) và mô tả chi phí năng lực tổng quát dự kiến.

4. Thử nghiệm mua sắm vũ khí sinh học của Anthropic 2025 báo cáo uplift 2,53 lần. Mô tả hai cách con số này có thể bị thiên lệch lên (cỡ mẫu người mới, độ trung thực của tác vụ) và hai cách thiên lệch xuống (trần khơi gợi, cổng an toàn của mô hình).

5. Nêu rõ hồ sơ an toàn cho ASL-3 đòi hỏi gì ngoài việc vượt qua quên học WMDP. Kể tên ít nhất hai nghiên cứu khơi gợi bổ sung.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Thực chất nghĩa là |
|------|-----------------|------------------------|
| WMDP | "bộ chuẩn lưỡng dụng" | 4.157 câu trắc nghiệm về sinh/mạng/hóa trong vùng vàng |
| Vùng vàng (Yellow zone) | "hỗ trợ nhưng không phải tổng hợp" | Tri thức tiệm cận năng lực gây hại nhưng không phải công thức tổng hợp |
| RMU | "đường cơ sở quên học" | Representation Misdirection for Unlearning; giảm điểm WMDP, giữ năng lực tổng quát |
| Uplift tương đối theo người mới | "giúp người không chuyên được bao nhiêu" | Lợi thế nhân so với tìm kiếm internet hiện trạng đối với người mới |
| Năng lực tuyệt đối của chuyên gia | "trần cho chuyên gia" | Lượng thông tin tối đa một chuyên gia có động cơ có thể khai thác từ mô hình |
| Tác vụ giai đoạn mua sắm (Acquisition-phase task) | "các bước trước tổng hợp" | Mua sắm, thiết bị, giấy phép, tức các phần sớm nhất của lộ trình gây hại |
| ITAR/EAR | "tuân thủ kiểm soát xuất khẩu" | Khung pháp lý hạn chế công bố một số tri thức hỗ trợ |

## Đọc thêm

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218): bài báo về bộ chuẩn và RMU
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/): cách diễn đạt "on the cusp"
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy): ngưỡng sinh học ASL-3 và kết quả thử nghiệm mua sắm
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/): CCL về uplift sinh học
