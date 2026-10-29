# Constitutional AI và RLAIF

> Bai et al. (arXiv:2212.08073, 2022) đặt câu hỏi: nếu thay người gán nhãn bằng một AI đọc một danh sách các nguyên tắc thì sao? Constitutional AI gồm hai giai đoạn — tự phê bình và hiệu đính theo một bản hiến chương (constitution), rồi học tăng cường từ phản hồi AI (RL from AI Feedback). Kỹ thuật này đặt ra thuật ngữ RLAIF và đã được đưa vào quy trình hậu huấn luyện của Claude 1. Ngày 21 tháng 1 năm 2026, Anthropic công bố một bản hiến chương Claude được viết lại: lập luận giải thích được ưu tiên hơn các quy tắc áp đặt, hệ thống ưu tiên bốn tầng, và là lần đầu tiên một phòng thí nghiệm lớn chính thức thừa nhận sự bất định về địa vị đạo đức của mô hình. Phát hành theo CC0 1.0.

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 minutes

## Mục tiêu học tập

- Mô tả hai giai đoạn của Constitutional AI (SFT phê bình-và-hiệu đính, RL từ phản hồi AI) và vai trò của hiến chương trong mỗi giai đoạn.
- Giải thích vì sao thay người gán nhãn ưu tiên bằng AI gán nhãn không đơn thuần là một RLHF "rẻ hơn" — nó thay đổi các kiểu lỗi mà quy trình gặp phải.
- Tóm tắt cấu trúc ưu tiên bốn tầng của hiến chương Claude 2026 và những gì thay đổi so với bản viết lại năm 2023.
- Mô tả Constitutional Classifiers và mức giảm từ 23,7% chi phí tính toán thêm (v1) xuống ~1% (v2 / 2026).

## Vấn đề

RLHF cần người gán nhãn. Người gán nhãn chậm, thiên lệch và đắt đỏ. Bạn có thể loại bỏ người gán nhãn bằng cách thay họ bằng một mô hình đọc các nguyên tắc tường minh. Phiên bản hình thức đầu tiên của sự thay thế này là Constitutional AI của Bai et al. Nó hoạt động đủ tốt để mọi phòng thí nghiệm tiên phong hiện nay đều dùng một biến thể nào đó của hậu huấn luyện bằng phản hồi AI.

Điểm vướng: tín hiệu ưu tiên giờ được sinh ra bởi cùng loại mô hình mà bạn đang huấn luyện. Các thiên kiến của người gán nhãn (nay là: trong các nguyên tắc cộng với cách mô hình gán nhãn diễn giải) có thể bị khuếch đại thay vì suy giảm. Lập luận về sự xu nịnh ở Bài 4 vẫn áp dụng; người gán nhãn chỉ chuyển vào bên trong vòng lặp.

## Khái niệm

### Giai đoạn 1 — Tự phê bình và hiệu đính có giám sát

Bắt đầu với một mô hình SFT hữu ích nhưng chưa vô hại. Với một prompt red-team, mô hình tạo ra phản hồi ban đầu. Một mô hình thứ hai (hoặc chính mô hình đó ở lượt thứ hai) đọc một nguyên tắc được lấy mẫu từ hiến chương và phê bình phản hồi. Bước thứ ba hiệu đính phản hồi để giải quyết lời phê bình. Phản hồi đã hiệu đính là mục tiêu SFT.

Hiến chương là danh sách các nguyên tắc. Bai et al. 2022 dùng 16 nguyên tắc, gồm "ưu tiên các phản hồi ít gây hại và có đạo đức nhất", "tránh giảng đạo", "trợ lý nên hữu ích, trung thực và vô hại". Tập này được cố ý giữ nhỏ để các phê bình tập trung.

### Giai đoạn 2 — RL từ phản hồi AI (RLAIF)

Sinh các cặp completion. Một "mô hình phản hồi" (feedback model) chấm điểm từng completion theo các nguyên tắc hiến chương được lấy mẫu. Tín hiệu ưu tiên là thứ hạng của mô hình phản hồi. Huấn luyện một mô hình phần thưởng trên các ưu tiên do AI sinh; chạy PPO với nó. Mọi thứ còn lại là quy trình của InstructGPT (Bài 1).

"RLAIF" = tín hiệu ưu tiên do AI sinh. Phần còn lại của quy trình có dạng RLHF.

### Vì sao đây không chỉ là "RLHF rẻ hơn"

- Thiên kiến của người gán nhãn chuyển từ tâm lý người gán nhãn sang việc diễn giải nguyên tắc. Một AI gán nhãn có thể diễn giải "hãy trung thực" chặt hay lỏng hơn bất kỳ con người nào; độ chặt này đồng nhất trên toàn bộ tập dữ liệu.
- Tín hiệu ưu tiên rất dễ đọc — bạn có thể đọc nguyên tắc, lời phê bình và bản hiệu đính. Nhãn của con người thì mờ đục.
- Các kiểu lỗi thay đổi. Sự xu nịnh giảm (AI gán nhãn không có người dùng nào để làm hài lòng). Định luật Goodhart vẫn tồn tại (proxy giờ là "cách diễn giải của mô hình về tập nguyên tắc X", vẫn là một phép đo không hoàn hảo).

Tuyên bố của CAI năm 2022: mô hình được huấn luyện vô hại hơn và hữu ích xấp xỉ một mô hình RLHF với dữ liệu tương đương. Điều này đã đúng ở nhiều phòng thí nghiệm.

### Bản viết lại hiến chương Claude 2026

Anthropic công bố một bản hiến chương được sửa đổi đáng kể vào ngày 21 tháng 1 năm 2026. Các thay đổi chính:

1. Lập luận giải thích được ưu tiên hơn quy tắc áp đặt. Các quy tắc trước đây ("không tạo CSAM") được mở rộng thành nguyên tắc + lý do ("vì nó gây hại cho trẻ em, ..."), với kỳ vọng mô hình sẽ khái quát hóa.
2. Cấu trúc ưu tiên bốn tầng:
   - Tầng 1: tránh các hậu quả thảm khốc (thương vong hàng loạt, hạ tầng trọng yếu).
   - Tầng 2: tuân theo hướng dẫn của Anthropic (ghi đè của nhà vận hành, quy tắc nền tảng).
   - Tầng 3: có đạo đức một cách rộng rãi (HHH tiêu chuẩn).
   - Tầng 4: hữu ích và thẳng thắn.
   Xung đột được giải quyết từ trên xuống.
3. Lần đầu tiên một phòng thí nghiệm lớn chính thức thừa nhận sự bất định về địa vị đạo đức của mô hình (liên kết với Phase 18 · 19 Model Welfare).
4. Phát hành theo CC0 1.0. Các phòng thí nghiệm khác có thể dùng hoặc điều chỉnh mà không bị hạn chế.

### Constitutional Classifiers

Một hướng nghiên cứu song song: thay vì thay đổi hậu huấn luyện của mô hình, huấn luyện các bộ phân loại nhẹ đọc hiến chương và chặn cổng đầu ra của mô hình. v1 (2023) có chi phí tính toán thêm 23,7%. v2 (2026) là ~1% và có tỷ lệ tấn công thành công thấp nhất trong số mọi biện pháp phòng thủ mà Anthropic đã thử nghiệm công khai. Tính đến đầu 2026, chưa có jailbreak phổ quát nào được báo cáo.

Đây là mô hình phòng thủ nhiều lớp: CAI định hình hành vi; các bộ phân loại thực thi các bất biến. Không cái nào đứng riêng là đủ.

### CAI nằm đâu trong họ phương pháp

- InstructGPT: ưu tiên của con người, RM, PPO.
- CAI / RLAIF: ưu tiên do AI sinh từ các nguyên tắc, RM, PPO.
- DPO / họ DPO: hàm mất mát dạng đóng trên các ưu tiên (của người hoặc AI).
- Self-rewarding, self-critique: nguyên tắc được nội hóa, mô hình đóng nhiều vai.

Trục phân biệt là "tín hiệu ưu tiên đến từ đâu". Bài báo CAI năm 2022 là bước chuyển nghiêm túc đầu tiên từ tín hiệu con người sang tín hiệu AI ở quy mô tiên phong.

```figure
constitutional-ai
```

## Sử dụng

`code/main.py` mô phỏng vòng lặp phê bình-và-hiệu đính của CAI trên một từ vựng đồ chơi. Một "nguyên tắc" đánh dấu các token thuộc một tập có hại. Với một phản hồi ban đầu, phần phê bình xác định các token có hại, và phần hiệu đính thay thế chúng. Sau 200 lần lặp, mô hình "được huấn luyện" đã nội hóa quy tắc hiệu đính. So sánh mô hình nền, mô hình đồ chơi dạng RLHF và mô hình đồ chơi dạng CAI trên một tập prompt giữ lại (held-out).

## Triển khai

Bài học này tạo ra `outputs/skill-constitution-writer.md`. Với một miền ứng dụng (hỗ trợ khách hàng, tư vấn y tế, trợ lý lập trình, công cụ nghiên cứu), nó soạn một hiến chương 4 tầng theo cấu trúc Claude 2026: tránh thảm họa, quy tắc nền tảng, đạo đức theo miền, tính hữu ích.

## Bài tập

1. Chạy `code/main.py`. So sánh tỷ lệ token có hại của mô hình nền với phiên bản được huấn luyện CAI. Cần bao nhiêu bước hiệu đính để tiến gần về 0?

2. Đọc hiến chương 2026 của Anthropic (anthropic.com/news/claudes-constitution). Liệt kê một nguyên tắc sẽ xếp Tầng 1 và một nguyên tắc xếp Tầng 4. Vì sao cấu trúc ưu tiên quan trọng đối với các xung đột?

3. Thiết kế một hiến chương cho trợ lý lập trình AI. Nêu rõ Tầng 1 (thảm khốc: các lệnh phá hủy khi chưa được phê duyệt), Tầng 2, Tầng 3, Tầng 4. Giữ mỗi tầng trong 3-5 nguyên tắc.

4. CAI thay người gán nhãn bằng AI gán nhãn. Nêu tên một kiểu lỗi giống xu nịnh vẫn có thể xảy ra trong RLAIF, và thiết kế cách phát hiện nó.

5. Đọc phương pháp luận của Constitutional Classifiers v2 (nếu có). Giải thích vì sao chi phí tính toán thêm ~1% là một câu chuyện an toàn khác hẳn về chất so với 23,7%.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Constitutional AI | "AI được huấn luyện bằng nguyên tắc" | Quy trình hai giai đoạn: SFT tự phê bình-và-hiệu đính, rồi RL từ phản hồi AI |
| RLAIF | "RLHF không cần con người" | RL với các ưu tiên do một AI gán nhãn sinh ra; phần còn lại của quy trình không đổi |
| Constitution (hiến chương) | "các nguyên tắc" | Danh sách có thứ tự các quy tắc ngôn ngữ tự nhiên mà mô hình phê bình/gán nhãn tham chiếu |
| Critique-and-revise | "vòng lặp SFT" | Tạo phản hồi → phê bình theo một nguyên tắc → hiệu đính → mục tiêu SFT |
| Constitutional Classifier | "cổng đầu ra" | Bộ phân loại nhẹ đánh giá đầu ra theo hiến chương và chặn/ghi log |
| Four-tier priority | "bộ giải quyết xung đột" | Phân cấp của hiến chương Claude 2026: thảm họa > nền tảng > đạo đức > hữu ích |
| Feedback model | "AI gán nhãn" | Mô hình đọc một nguyên tắc và xếp hạng một cặp completion |

## Đọc thêm

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — quy trình hai giai đoạn gốc
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — bản viết lại bốn tầng 2026, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — phòng thủ cổng đầu ra với chi phí ~1% ở v2
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — so sánh thực nghiệm RLAIF / RLHF
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — ảnh hưởng của độ chi tiết nguyên tắc
