# Khung an toàn tiên phong (Frontier Safety Frameworks): RSP, PF, FSF

> Ba khung của các phòng thí nghiệm lớn định hình cách quản trị năng lực tiên phong của ngành năm 2026. Responsible Scaling Policy v3.0 của Anthropic (tháng 2/2026) đưa ra các Mức An toàn AI (AI Safety Level, ASL-1 đến ASL-5+) theo mô hình các cấp an toàn sinh học, trong đó ASL-3 được kích hoạt từ tháng 5/2025 cho các mô hình liên quan đến CBRN. Preparedness Framework v2 của OpenAI (tháng 4/2025) định nghĩa năm tiêu chí cho các năng lực được theo dõi và tách Báo cáo Năng lực (Capabilities Report) khỏi Báo cáo Biện pháp bảo vệ (Safeguards Report). Frontier Safety Framework v3.0 của DeepMind (tháng 9/2025) đưa vào các Mức Năng lực Tới hạn (Critical Capability Level), gồm CCL Thao túng Có hại (Harmful Manipulation) mới. Cả ba hiện đều có điều khoản điều chỉnh theo đối thủ (competitor-adjustment clause), cho phép trì hoãn nếu các phòng thí nghiệm khác phát hành mà không có biện pháp bảo vệ tương đương. Sự tương đồng giữa các phòng thí nghiệm mang tính cấu trúc chứ không phải thuật ngữ: "Capability Thresholds", "High Capability thresholds" và "Critical Capability Levels" chỉ các cấu trúc tương tự nhau.

**Loại:** Learn
**Ngôn ngữ:** không có
**Điều kiện tiên quyết:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (các thất bại do lừa dối)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả cấu trúc các tầng ASL của Anthropic và điều gì đã kích hoạt ASL-3.
- Nêu năm tiêu chí của OpenAI Preparedness Framework v2 cho các năng lực được theo dõi.
- Mô tả cấu trúc Critical Capability Level của DeepMind và CCL Thao túng Có hại.
- Giải thích các điều khoản điều chỉnh theo đối thủ và vì sao chúng quan trọng với động lực chạy đua.
- Định nghĩa hồ sơ an toàn (safety case) và mô tả cấu trúc ba trụ cột (giám sát, khó hiểu, bất lực).

## Vấn đề

Các bài 7-17 xác lập rằng lừa dối là có thể, năng lực lưỡng dụng tồn tại và đánh giá có giới hạn. Một phòng thí nghiệm sở hữu mô hình ở mức tiên phong cần một cấu trúc quản trị nội bộ để:
- Định nghĩa các ngưỡng khi nào cần biện pháp bảo vệ mới.
- Định nghĩa các đánh giá bắt buộc trước khi mở rộng quy mô.
- Mô tả một hồ sơ an toàn trông như thế nào.
- Xử lý bài toán động lực chạy đua (nếu đối thủ phát hành mà không có biện pháp bảo vệ thì làm gì?).

Ba khung 2025-2026 là trình độ tiên tiến nhất hiện nay: chưa hoàn hảo, còn tiến hóa, và đủ tương đồng giữa các phòng thí nghiệm để câu hỏi quản trị lúc này là các khung đó có đủ hay không, chứ không phải chúng có tồn tại hay không.

## Khái niệm

### Responsible Scaling Policy v3.0 của Anthropic (tháng 2/2026)

Cấu trúc ASL:
- ASL-1: không phải mô hình tiên phong (nằm trong đường cơ sở yếu hơn tiên phong).
- ASL-2: đường cơ sở tiên phong hiện tại; triển khai với các biện pháp bảo vệ thông thường.
- ASL-3: rủi ro lạm dụng gây thảm họa cao hơn đáng kể; năng lực liên quan đến CBRN. Kích hoạt tháng 5/2025.
- ASL-4: vượt ngưỡng AI R&D-2; mô hình có thể tự động hóa nghiên cứu AI cấp nhập môn.
- ASL-5+: AI R&D tiên tiến; mô hình tăng tốc đáng kể quá trình mở rộng quy mô hiệu quả.

Điểm mới trong v3.0:
- Lộ trình An toàn Tiên phong (Frontier Safety Roadmaps), công khai dưới dạng đã biên tập.
- Báo cáo Rủi ro (Risk Reports), hằng quý, một số có thẩm định bên ngoài.
- AI R&D được tách thành AI R&D-2 và AI R&D-4.
- Khi vượt AI R&D-4, cần một hồ sơ an toàn khẳng định (affirmative safety case), xác định các rủi ro lệch hướng (misalignment) từ mô hình theo đuổi mục tiêu lệch hướng.

### Preparedness Framework v2 của OpenAI (15/4/2025)

Năm tiêu chí cho năng lực được theo dõi:
- **Plausible (Hợp lý).** Tồn tại mô hình đe dọa hợp lý.
- **Measurable (Đo được).** Có thể đánh giá thực nghiệm.
- **Severe (Nghiêm trọng).** Tác hại lớn.
- **Net-new (Hoàn toàn mới).** Không phải rủi ro sẵn có được phóng đại.
- **Instantaneous-or-irremediable (Tức thời hoặc không thể khắc phục).** Tác hại xảy ra nhanh hoặc không thể đảo ngược.

Các năng lực thỏa cả năm tiêu chí được theo dõi. Các năng lực khác thì không.

Cấu trúc khác của PF v2:
- Tách Báo cáo Năng lực (mô hình làm được gì) khỏi Báo cáo Biện pháp bảo vệ (có những kiểm soát nào).
- Safety Advisory Group thẩm định.
- Ban lãnh đạo phê duyệt; Safety & Security Committee của Hội đồng quản trị giám sát.
- "Điều khoản điều chỉnh" (adjustment clause): OpenAI có thể hạ yêu cầu nếu phòng thí nghiệm khác phát hành mà không có biện pháp bảo vệ tương đương.

### Frontier Safety Framework v3.0 của DeepMind (tháng 9/2025)

Các Critical Capability Level (CCL) theo lĩnh vực:
- Bioweapon Uplift (hỗ trợ vũ khí sinh học)
- Cyber Uplift (hỗ trợ tấn công mạng)
- ML R&D Acceleration (tăng tốc R&D học máy)
- Harmful Manipulation (mới trong v3.0): mô hình có thể thay đổi đáng kể niềm tin/hành vi trong các bối cảnh có rủi ro cao.

v2.0 (tháng 2/2025) bổ sung mục Deceptive Alignment (căn chỉnh lừa dối) và các mức bảo mật cao hơn cho CCL về ML R&D.

### Sự tương đồng giữa các phòng thí nghiệm

- Anthropic: "Capability Thresholds".
- DeepMind: "Critical Capability Levels".
- OpenAI: "High Capability thresholds".

Không có thuật ngữ chuẩn của ngành. Nhưng tương đồng về cấu trúc: mỗi khung trong ba khung định nghĩa các ngưỡng năng lực theo tầng với tiêu chí đánh giá được công bố. Cả ba đều có điều khoản điều chỉnh theo đối thủ (từ 2025 trở đi).

### Hồ sơ an toàn (safety case)

Hồ sơ an toàn là một lập luận bằng văn bản rằng một lần triển khai an toàn ở mức chấp nhận được dưới các giả định xấu nhất. Cấu trúc chuẩn nhắm vào ba trụ cột:

- **Giám sát (Monitoring).** Ta có phát hiện được hành vi xấu nếu nó xảy ra không?
- **Khó hiểu (Illegibility).** Mô hình có thiếu khả năng thực thi một kế hoạch mạch lạc để gây hại không?
- **Bất lực (Incapability).** Mô hình có thiếu năng lực gây ra tác hại đang xét không?

Các hồ sơ an toàn khác nhau nhắm các trụ cột khác nhau. Với hồ sơ CBRN ở ASL-3, bất lực (qua quên học) là mục tiêu chính. Với căn chỉnh lừa dối, giám sát và tính khó hiểu là mục tiêu. Với uplift tấn công mạng, cả ba đều liên quan.

### Bài toán động lực chạy đua

Điều khoản điều chỉnh theo đối thủ gây tranh cãi. Người phê phán cho rằng chúng tạo ra cuộc đua xuống đáy: nếu cả ba phòng thí nghiệm sẽ hạ yêu cầu khi một đối thủ vi phạm, trạng thái cân bằng sẽ dịch về phía vi phạm. Người bảo vệ cho rằng giải pháp thay thế (đơn phương áp dụng biện pháp bảo vệ) cho kết quả tệ hơn nếu phòng thí nghiệm vi phạm kém chú trọng an toàn hơn.

UK AISI, US CAISI và EU AI Office (Bài 24) là các đối trọng quản trị bên ngoài. Các khung của phòng thí nghiệm mang tính tự nguyện; các khung quy định pháp lý đang hình thành.

### Vị trí trong Phase 18

Các bài 17-18 là tầng đo lường và quản trị phủ lên các phân tích về lừa dối và red-team. Các bài 19-24 bao quát phúc lợi, thiên lệch, quyền riêng tư, đóng dấu thủy vân và cấu trúc quy định pháp lý. Bài 28 vẽ bản đồ hệ sinh thái nghiên cứu (MATS, Redwood, Apollo, METR) vận hành hóa các đánh giá.

```figure
al-asl-ladder
```

## Sử dụng

Bài này không có mã. Hãy đọc ba nguồn gốc: RSP v3.0, PF v2, FSF v3.0. Ánh xạ cấu trúc tầng của mỗi phòng thí nghiệm sang các phòng còn lại và xác định một ngưỡng mà mỗi phòng thí nghiệm định nghĩa còn các phòng kia thì không.

## Triển khai

Bài học này tạo ra `outputs/skill-framework-diff.md`. Với một khung an toàn hoặc ghi chú phát hành, nó so sánh các định nghĩa ngưỡng, các đánh giá bắt buộc và cấu trúc hồ sơ an toàn của khung đó với RSP v3.0, PF v2, FSF v3.0 và đánh dấu các khoảng trống giữa các phòng thí nghiệm.

## Bài tập

1. Đọc RSP v3.0, PF v2 và FSF v3.0. Lập bảng gồm ngưỡng CBRN của mỗi phòng thí nghiệm, ngưỡng AI R&D của mỗi phòng và đánh giá bắt buộc trước triển khai của mỗi phòng.

2. Điều khoản điều chỉnh theo đối thủ có trong cả ba khung (2025+). Viết một đoạn lập luận ủng hộ; viết một đoạn lập luận phản đối. Xác định giả định mà mỗi quan điểm dựa vào.

3. Thiết kế hồ sơ an toàn cho một mô hình vượt ngưỡng AI R&D-4 của Anthropic. Nêu bằng chứng mà mỗi trong ba trụ cột (giám sát, khó hiểu, bất lực) đòi hỏi.

4. FSF v3.0 của DeepMind đưa vào CCL Thao túng Có hại. Đề xuất ba phép đo thực nghiệm cho thấy một mô hình đã vượt ngưỡng này.

5. Đọc "Common Elements of Frontier AI Safety Policies" (2025) của METR. Nêu ba điểm hội tụ mạnh nhất giữa các phòng thí nghiệm và hai điểm khác biệt lớn nhất.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Thực chất nghĩa là |
|------|-----------------|------------------------|
| RSP | "khung của Anthropic" | Responsible Scaling Policy; các tầng ASL; v3.0 tháng 2/2026 |
| PF | "khung của OpenAI" | Preparedness Framework; năm tiêu chí; v2 tháng 4/2025 |
| FSF | "khung của DeepMind" | Frontier Safety Framework; các CCL; v3.0 tháng 9/2025 |
| ASL-3 | "tương tự cấp an toàn sinh học 3" | Tầng của Anthropic cho năng lực liên quan CBRN; kích hoạt tháng 5/2025 |
| CCL | "mức năng lực tới hạn" | Cấu trúc ngưỡng của DeepMind; theo từng lĩnh vực |
| Hồ sơ an toàn (Safety case) | "lập luận chính thức" | Lập luận bằng văn bản rằng triển khai an toàn ở mức chấp nhận được dưới giả định xấu nhất |
| Điều khoản điều chỉnh (Adjustment clause) | "cho phép khi đối thủ vi phạm" | Điều khoản của khung cho phép hạ yêu cầu nếu đối thủ phát hành mà không có biện pháp bảo vệ tương đương |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy): các tầng ASL, lộ trình, tách AI R&D
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/): năm tiêu chí, điều khoản điều chỉnh
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/): CCL v3.0, Thao túng Có hại
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/): so sánh giữa các phòng thí nghiệm
