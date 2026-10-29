# Quyền riêng tư vi phân (Differential Privacy) cho LLM

> DP-SGD vẫn là chuẩn mực: các bản cập nhật gradient được chèn nhiễu cung cấp bảo đảm hình thức (epsilon, delta). Chi phí về tính toán, bộ nhớ và độ hữu dụng là đáng kể; tinh chỉnh DP hiệu quả tham số (LoRA + DP-SGD) là cấu hình phổ biến năm 2025 (ACM 2025). Hai nhóm bằng chứng mâu thuẫn nhau: suy luận tư cách thành viên dựa trên canary (Duan và cộng sự, 2024) báo cáo thành công hạn chế trên mô hình ngôn ngữ; trích xuất dữ liệu huấn luyện (Carlini và cộng sự, 2021; Nasr và cộng sự, 2025) khôi phục được lượng lớn nội dung ghi nhớ nguyên văn. Lời giải (arXiv:2503.06808, tháng 3/2025): khác biệt nằm ở thứ được đo: canary được chèn vào so với dữ liệu "dễ trích xuất nhất". Các thiết kế canary mới cho phép MIA dựa trên loss mà không cần shadow model và tạo ra cuộc kiểm toán DP không tầm thường đầu tiên cho một LLM huấn luyện trên dữ liệu thật với bảo đảm DP thực tế. Các lựa chọn thay thế: PMixED (arXiv:2403.15638) là dự đoán riêng tư tại thời điểm suy luận bằng hỗn hợp chuyên gia trên phân phối token kế tiếp; sinh dữ liệu tổng hợp DP (Google Research 2024). Tấn công mới nổi: Đảo ngược Quyền riêng tư Vi phân qua Phản hồi LLM, tức rò rỉ qua điểm tin cậy.

**Loại:** Build
**Ngôn ngữ:** Python (stdlib, minh họa chèn nhiễu DP-SGD và bộ kế toán ε-δ)
**Điều kiện tiên quyết:** Phase 01 · 09 (lý thuyết thông tin), Phase 10 · 01 (huấn luyện mô hình lớn)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Định nghĩa quyền riêng tư vi phân (epsilon, delta) và nêu công thức DP-SGD.
- Giải thích mâu thuẫn 2024-2025: MIA bằng canary và trích xuất dữ liệu huấn luyện cho hai bức tranh khác nhau.
- Mô tả PMixED và lý do dự đoán riêng tư tại thời điểm suy luận là phương án thay thế cho huấn luyện DP.
- Mô tả tấn công Đảo ngược Quyền riêng tư Vi phân qua Phản hồi LLM.

## Vấn đề

LLM ghi nhớ. Carlini và cộng sự 2021 chỉ ra các mô hình ngôn ngữ thương mại tái tạo nguyên văn văn bản huấn luyện theo yêu cầu. DP là biện pháp phòng thủ hình thức: huấn luyện sao cho đầu ra được chứng minh là không nhạy với bất kỳ mẫu huấn luyện đơn lẻ nào. Bằng chứng 2024-2025 cho thấy DP-SGD là cần thiết nhưng các giá trị ε được triển khai có thể không khớp với mô hình đe dọa.

## Khái niệm

### Quyền riêng tư vi phân (ε, δ)

Một thuật toán ngẫu nhiên $M$ là $(\varepsilon, \delta)$-DP nếu với mọi hai tập dữ liệu khác nhau ở một mẫu và mọi sự kiện $S$:

```math
P(M(D) \in S) \le e^{\varepsilon} \cdot P(M(D') \in S) + \delta.
```

Diễn giải: phân phối đầu ra đủ gần nhau (được tham số hóa bởi ε) để đóng góp của bất kỳ cá nhân đơn lẻ nào không thể được suy ra một cách đáng tin, trừ với xác suất δ.

### DP-SGD

Abadi và cộng sự 2016. Công thức chuẩn:
1. Lấy mẫu một mini-batch.
2. Tính gradient theo từng mẫu.
3. Cắt (clip) từng gradient theo mẫu về ngưỡng C.
4. Cộng các gradient đã cắt và thêm nhiễu Gauss với độ lệch chuẩn $\sigma \cdot C$.
5. Dùng tổng có nhiễu để cập nhật tham số.

Chi phí riêng tư được theo dõi bởi một bộ kế toán (accountant: Moments Accountant, Rényi DP accountant). Các giá trị ε được báo cáo trong tài liệu về LLM biến động rất lớn theo mô hình đe dọa, độ nhạy của dữ liệu và mục tiêu hữu dụng; không có ε mặc định "an toàn" phổ quát. Các ví dụ đã công bố nằm khoảng ε ≈ 1-10 trong một số cấu hình huấn luyện LLM, nhưng chỉ mang tính minh họa, không phải mặc định được khuyến nghị. ε thấp hơn nhìn chung cần nhiều nhiễu hơn và có thể làm tăng mất mát độ hữu dụng.

### LoRA + DP-SGD

DP-SGD toàn phần cho một mô hình tiên phong là không khả thi. LoRA (Hu và cộng sự 2022) giới hạn cập nhật gradient vào một adapter nhỏ, giảm dung lượng lưu gradient theo mẫu. LoRA + DP-SGD là cấu hình phổ biến năm 2025. Bảo đảm DP áp dụng cho adapter; mô hình nền được giữ cố định.

### Mâu thuẫn 2024-2025

Hai hướng bằng chứng:

- **MIA bằng canary (Duan và cộng sự 2024).** Chèn các canary duy nhất vào dữ liệu huấn luyện, đo xem kẻ tấn công suy luận tư cách thành viên có nhận diện được chúng không. Báo cáo thành công hạn chế trên mô hình ngôn ngữ. Gợi ý rằng MIA là khó.
- **Trích xuất dữ liệu huấn luyện (Carlini 2021, Nasr và cộng sự 2025).** Gợi ý mô hình bằng một tiền tố; đo xem nó có khôi phục nguyên văn từ dữ liệu huấn luyện không. Báo cáo ghi nhớ đáng kể. Gợi ý rằng MIA là dễ theo nghĩa liên quan.

Lời giải tháng 3/2025 (arXiv:2503.06808): hai bên đo những thứ khác nhau. MIA hỏi "mẫu e có thuộc D không?" trên các canary được chèn. Trích xuất hỏi "tôi khôi phục được gì từ D?". Mẫu "dễ trích xuất nhất" mới là thứ quan trọng với quyền riêng tư; canary báo cáo thấp hơn thực tế vì chúng không được tối ưu để dễ trích xuất.

Các thiết kế canary mới. MIA dựa trên loss không cần shadow model. Cuộc kiểm toán DP không tầm thường đầu tiên của một LLM trên dữ liệu thật với bảo đảm DP thực tế.

### Các lựa chọn thay thế cho huấn luyện DP

- **PMixED (arXiv:2403.15638).** Dự đoán riêng tư tại thời điểm suy luận. Hỗn hợp chuyên gia trên phân phối token kế tiếp; mỗi chuyên gia thấy một phân mảnh dữ liệu huấn luyện; bước tổng hợp thêm nhiễu để đạt DP. Tránh hoàn toàn huấn luyện DP.
- **Sinh dữ liệu tổng hợp DP (Google Research 2024).** Tinh chỉnh LoRA bằng DP-SGD, lấy mẫu dữ liệu tổng hợp, huấn luyện bộ phân loại hạ nguồn trên dữ liệu tổng hợp.

Cả hai đều né chi phí hữu dụng của huấn luyện DP toàn phần, đổi lại bằng một mô hình đe dọa khác.

### Đảo ngược Quyền riêng tư Vi phân qua Phản hồi LLM

Tấn công mới nổi năm 2025. Dùng điểm tin cậy của mô hình được huấn luyện DP như một oracle để tái định danh cá nhân. Ngay cả khi đầu ra không rò rỉ, phân phối độ tin cậy vẫn có thể rò rỉ.

Biện pháp phòng thủ: không phơi bày độ tin cậy, hoặc cắt ngắn/lượng tử hóa chúng trước khi phơi bày. Đây là yêu cầu bổ sung ngoài huấn luyện (ε, δ)-DP.

### Vị trí trong Phase 18

Các bài 20-21 nói về thiên lệch/công bằng. Bài 22 là quyền riêng tư. Bài 23 là nguồn gốc nội dung qua đóng dấu thủy vân. Bài 27 bao quát tầng quy định về nguồn gốc dữ liệu.

```figure
an-dp-clip-noise
```

## Sử dụng

`code/main.py` mô phỏng DP-SGD trên một tập dữ liệu phân loại nhị phân đồ chơi. Bạn có thể quét hệ số nhiễu σ và chuẩn cắt C, đồng thời theo dõi ngân sách (ε, δ) và chi phí độ chính xác. Một "tấn công canary" chèn một mẫu huấn luyện duy nhất và đo xem phép kiểm tra log-loss có phát hiện được nó trước và sau DP hay không.

## Triển khai

Bài học này tạo ra `outputs/skill-dp-audit.md`. Với một tuyên bố DP về việc triển khai mô hình ngôn ngữ, nó kiểm toán: các giá trị (ε, δ), bộ kế toán được dùng, giao thức đánh giá MIA, và liệu các vector phơi bày độ tin cậy đã được đánh giá chưa.

## Bài tập

1. Chạy `code/main.py`. Quét $\sigma \in \lbrace 0.5, 1.0, 2.0 \rbrace$ và báo cáo đánh đổi (ε, δ)-độ chính xác. Xác định điểm mà độ hữu dụng sụp đổ.

2. Cài đặt việc chèn canary và phép kiểm tra log-loss. Đo tỷ lệ phát hiện trước và sau DP-SGD với $\sigma = 1.0$.

3. Đọc Nasr và cộng sự 2025 về trích xuất dữ liệu huấn luyện. Vì sao thành công của trích xuất không sụp đổ ở ε vừa phải? Điều này hàm ý gì về việc dùng MIA làm công cụ đánh giá?

4. Thiết kế một triển khai dùng PMixED (arXiv:2403.15638) hoạt động hoàn toàn tại thời điểm suy luận. Mô hình đe dọa nào PMixED giải quyết mà DP-SGD thì không?

5. Phác thảo tấn công Đảo ngược DP qua Phản hồi LLM. Thiết kế một biện pháp đối phó hạn chế rò rỉ điểm tin cậy và ước lượng chi phí triển khai.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Thực chất nghĩa là |
|------|-----------------|------------------------|
| DP | "quyền riêng tư vi phân (ε, δ)" | Quyền riêng tư hình thức: phân phối đầu ra gần nhau khi tập dữ liệu láng giềng thay đổi |
| DP-SGD | "SGD chèn nhiễu" | Cắt gradient + thêm nhiễu Gauss; huấn luyện DP chuẩn |
| LoRA + DP-SGD | "tinh chỉnh riêng tư hiệu quả" | DP-SGD trên adapter hạng thấp; cấu hình chuẩn 2025 |
| MIA | "suy luận tư cách thành viên" | Tấn công xác định một mẫu có nằm trong dữ liệu huấn luyện hay không |
| Canary | "mẫu thủy vân được chèn" | Mẫu huấn luyện duy nhất dùng để đo rò rỉ DP |
| PMixED | "hỗn hợp suy luận riêng tư" | DP tại thời điểm suy luận bằng hỗn hợp chuyên gia trên phân phối token kế tiếp |
| Đảo ngược DP (DP Reversal) | "tấn công rò rỉ độ tin cậy" | Tấn công dùng độ tin cậy của mô hình như một oracle để tái định danh |

## Đọc thêm

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133): thuật toán huấn luyện DP chuẩn
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805): bài báo kinh điển về trích xuất
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841): MIA thành công hạn chế
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808): lời giải cho mâu thuẫn
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638): dự đoán riêng tư tại thời điểm suy luận
