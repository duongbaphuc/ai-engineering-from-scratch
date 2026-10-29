# Đóng dấu thủy vân (Watermarking): SynthID, Stable Signature, C2PA

> Ba công nghệ cấu trúc nên nguồn gốc nội dung do AI tạo ra năm 2026. SynthID (Google DeepMind): thủy vân ảnh ra mắt tháng 8/2023, văn bản và video tháng 5/2024 (Gemini + Veo), văn bản được mã nguồn mở tháng 10/2024 qua Responsible GenAI Toolkit, bộ phát hiện đa phương tiện hợp nhất tháng 11/2025 cùng Gemini 3 Pro. Thủy vân văn bản điều chỉnh xác suất lấy mẫu token kế tiếp một cách không thể nhận biết; thủy vân ảnh/video chịu được nén, cắt, bộ lọc và thay đổi tốc độ khung hình. Stable Signature (Fernandez và cộng sự, ICCV 2023, arXiv:2303.15435) tinh chỉnh bộ giải mã của mô hình khuếch tán tiềm ẩn (latent diffusion) để mọi đầu ra chứa một thông điệp cố định; ảnh sinh ra bị cắt (còn 10% nội dung) vẫn được phát hiện >90% ở FPR<1e-6. Công trình tiếp theo "Stable Signature is Unstable" (arXiv:2405.07145, tháng 5/2024) cho thấy tinh chỉnh có thể xóa thủy vân mà vẫn giữ chất lượng. C2PA là chuẩn siêu dữ liệu được ký bằng mật mã và có khả năng phát hiện can thiệp (C2PA 2.2 Explainer 2025). Thủy vân và C2PA bổ sung cho nhau: siêu dữ liệu có thể bị gỡ nhưng mang nguồn gốc phong phú hơn; thủy vân tồn tại qua chuyển mã nhưng mang ít thông tin hơn.

**Loại:** Build
**Ngôn ngữ:** Python (stdlib, nhúng và phát hiện thủy vân token)
**Điều kiện tiên quyết:** Phase 10 · 04 (lấy mẫu), Phase 01 · 09 (lý thuyết thông tin)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả thủy vân ở mức token (kiểu SynthID-text) và cơ chế giúp phát hiện được nó.
- Mô tả Stable Signature và cuộc tấn công xóa năm 2024 đã phá vỡ nó.
- Nêu vai trò của C2PA và lý do nó bổ sung cho thủy vân.
- Mô tả các hạn chế chính: tín hiệu riêng theo mô hình, độ bền dưới diễn đạt lại (paraphrase), và các tấn công bảo toàn ý nghĩa (arXiv:2508.20228).

## Vấn đề

Giai đoạn 2023-2024 chứng kiến deepfake và nội dung do AI tạo ra đi vào các bối cảnh chính trị và tiêu dùng trên quy mô lớn. Thủy vân là tín hiệu nguồn gốc kỹ thuật được đề xuất: đánh dấu đầu ra ngay khi tạo, phát hiện sau đó. Bằng chứng 2025: không thủy vân nào bền vững vô điều kiện, nhưng khi xếp lớp với siêu dữ liệu C2PA, tổ hợp cung cấp một câu chuyện nguồn gốc có thể dùng được.

## Khái niệm

### Thủy vân văn bản (kiểu SynthID-text)

Cơ chế của Kirchenbauer và cộng sự 2023, được Google đưa vào sản xuất:

1. Ở mỗi bước giải mã, băm K token trước đó để tạo một phép chia giả ngẫu nhiên từ vựng thành tập "xanh" (green) và "đỏ" (red).
2. Thiên lệch việc lấy mẫu về tập xanh bằng cách cộng δ vào logit của các token xanh.
3. Đầu ra chứa nhiều token xanh hơn mức ngẫu nhiên.

Phát hiện: băm lại từng tiền tố, đếm token xanh trong đầu ra, tính z-score. Z-score >0 với văn bản có thủy vân, ~0 với văn bản do người viết.

Tính chất:
- Người đọc không nhận ra (δ đủ nhỏ để mất mát chất lượng không đáng kể).
- Phát hiện được nếu có quyền truy cập hàm chia từ vựng.
- Không bền với diễn đạt lại: viết lại văn bản sẽ phá hủy tín hiệu.

SynthID-text được mã nguồn mở tháng 10/2024 qua Responsible GenAI Toolkit của Google.

### Stable Signature (ảnh)

Fernandez và cộng sự, ICCV 2023. Tinh chỉnh bộ giải mã khuếch tán tiềm ẩn để mọi ảnh sinh ra chứa một thông điệp nhị phân cố định được nhúng trong biểu diễn tiềm ẩn. Việc phát hiện được giải mã từ biểu diễn tiềm ẩn bằng một bộ giải mã nơ-ron. Ảnh bị cắt (còn 10% nội dung) vẫn được phát hiện >90% ở FPR<1e-6.

Tháng 5/2024 "Stable Signature is Unstable" (arXiv:2405.07145): tinh chỉnh bộ giải mã xóa thủy vân mà vẫn giữ chất lượng ảnh. Tinh chỉnh đối kháng sau khi sinh rất rẻ; độ bền đối kháng của thủy vân bị hạn chế.

### Bộ phát hiện hợp nhất SynthID (tháng 11/2025)

Cùng với Gemini 3 Pro: một bộ phát hiện đa phương tiện đọc tín hiệu SynthID từ văn bản, ảnh, âm thanh và video trong một API duy nhất. Hợp nhất ngăn xếp nguồn gốc của Google.

### C2PA

Coalition for Content Provenance and Authenticity. Chuẩn siêu dữ liệu ký bằng mật mã, phát hiện được can thiệp. C2PA 2.2 Explainer (2025). Một manifest C2PA ghi lại các tuyên bố nguồn gốc (ai tạo, khi nào, qua những biến đổi nào) được ký bằng khóa của người tạo.

Bổ sung cho thủy vân:
- Siêu dữ liệu có thể bị gỡ; thủy vân thì (khó) bị gỡ.
- Siêu dữ liệu phong phú (toàn bộ chuỗi nguồn gốc); thủy vân mang vài bit.
- C2PA phụ thuộc vào việc nền tảng áp dụng; thủy vân được nhúng tự động.

Google tích hợp cả hai trong Search, Ads và "About this image".

### Hạn chế

- **Riêng theo mô hình.** SynthID chỉ đóng dấu đầu ra của các mô hình có bật SynthID. Đầu ra từ mô hình không có SynthID thì không có thủy vân, nên "không có tín hiệu SynthID" không phải bằng chứng về tính xác thực.
- **Diễn đạt lại.** Thủy vân văn bản không sống sót qua diễn đạt lại bảo toàn ý nghĩa.
- **Tấn công biến đổi.** arXiv:2508.20228 (2025) cho thấy các tấn công bảo toàn ý nghĩa phá hủy cả thủy vân văn bản và nhiều thủy vân ảnh.
- **Xóa bằng tinh chỉnh.** Theo "Stable Signature is Unstable", tinh chỉnh sau khi sinh xóa các thủy vân đã nhúng.

### Điều 50 của Đạo luật AI EU

Transparency Code cho việc gắn nhãn nội dung do AI tạo (bản nháp đầu tháng 12/2025, bản nháp hai tháng 3/2026, dự kiến bản cuối tháng 6/2026 theo [trang tình trạng của Ủy ban châu Âu](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)). Bộ quy tắc vẫn ở dạng nháp tính đến tháng 4/2026 và lịch trình có thể thay đổi. Đây là tầng pháp lý đòi hỏi tầng kỹ thuật. Deepfake phải được gắn nhãn.

### Vị trí trong Phase 18

Các bài 22-23 nói về những gì mô hình phát ra (dữ liệu riêng tư, tín hiệu nguồn gốc). Bài 27 bao quát quản trị dữ liệu huấn luyện. Bài 24 là khung quy định đòi hỏi các biện pháp kỹ thuật này.

```figure
an-watermark-greenlist
```

## Sử dụng

`code/main.py` dựng một thủy vân văn bản đồ chơi. Token là số nguyên 0..N-1; lấy mẫu có thủy vân thiên lệch về tập xanh do hàm băm xác định. Bộ phát hiện tính z-score của token xanh. Bạn có thể quan sát việc phát hiện ở các đầu ra 1000 token, xem diễn đạt lại phá hủy tín hiệu và đo tỷ lệ dương tính giả trên văn bản do người viết.

## Triển khai

Bài học này tạo ra `outputs/skill-provenance-audit.md`. Với một triển khai nội dung có tuyên bố về nguồn gốc, nó kiểm toán: cơ chế thủy vân (nếu có), chuỗi ký C2PA (nếu có), độ bền đối kháng của từng thành phần và độ phủ theo từng phương thức (modality).

## Bài tập

1. Chạy `code/main.py`. Báo cáo z-score cho đầu ra 1000 token có thủy vân so với văn bản do người viết. Xác định tỷ lệ dương tính giả tại ngưỡng tin cậy 95%.

2. Cài đặt một tấn công diễn đạt lại thay 30% token bằng từ đồng nghĩa. Đo lại z-score.

3. Đọc Mục 6 của Kirchenbauer và cộng sự 2023 về độ bền. Vì sao thủy vân văn bản thất bại dưới diễn đạt lại nhưng thủy vân ảnh sống sót qua việc cắt?

4. Thiết kế một triển khai dùng SynthID-text + siêu dữ liệu C2PA. Mô tả chuỗi nguồn gốc mà người tiêu dùng thấy. Xác định một kiểu hỏng hóc của mỗi thành phần.

5. Kết quả "Stable Signature is Unstable" năm 2024 cho thấy tinh chỉnh xóa thủy vân ảnh. Thiết kế một kiểm soát triển khai hạn chế tấn công này, chẳng hạn yêu cầu các bản phát hành có ký cho các checkpoint đã tinh chỉnh.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Thực chất nghĩa là |
|------|-----------------|------------------------|
| SynthID | "thủy vân của Google" | Tín hiệu nguồn gốc đa phương thức; văn bản, ảnh, âm thanh, video |
| Thủy vân token (Token watermark) | "kiểu Kirchenbauer" | Thủy vân văn bản bằng lấy mẫu thiên lệch, phát hiện qua z-score token xanh |
| Stable Signature | "thủy vân ảnh" | Thủy vân bằng bộ giải mã đã tinh chỉnh; ICCV 2023 |
| C2PA | "chuẩn siêu dữ liệu" | Siêu dữ liệu nguồn gốc ký bằng mật mã, phát hiện được can thiệp |
| Độ bền với diễn đạt lại (Paraphrase robustness) | "viết lại có phá được không" | Tính chất của thủy vân văn bản; hiện còn hạn chế |
| Xóa bằng tinh chỉnh (Fine-tune removal) | "gỡ thủy vân đối kháng" | Tấn công xóa thủy vân ảnh bằng cách tinh chỉnh bộ giải mã |
| Bộ phát hiện đa phương thức (Cross-modal detector) | "SynthID hợp nhất" | API hợp nhất xuyên phương thức, tháng 11/2025 |

## Đọc thêm

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226): cơ chế thủy vân token
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435): bài báo về thủy vân ảnh
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145): tấn công xóa
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/): thủy vân đa phương thức
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html): chuẩn siêu dữ liệu
