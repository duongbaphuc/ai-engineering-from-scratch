# Mô hình VLA nhập thể: RT-2, OpenVLA, π0, GR00T

> Lần đầu tiên một mô hình AI có thể đọc công thức nấu ăn từ một trang web và trực tiếp điều khiển cánh tay robot trong nhà bếp để thực thi là với RT-2 (Google DeepMind, tháng 7 năm 2023). RT-2 đã rời rạc hóa các hành động điều khiển thành các token văn bản, tinh chỉnh đồng thời (co-fine-tune) một mô hình thị giác - ngôn ngữ (VLM) trên dữ liệu web kết hợp với dữ liệu hành động robot, chứng minh rằng tri thức thị giác - ngôn ngữ quy mô web có thể chuyển giao hiệu quả sang điều khiển robot vật lý. OpenVLA (tháng 6 năm 2024) đã cung cấp bản triển khai mã nguồn mở tham chiếu 7B tham số. Dòng mô hình π0 của Physical Intelligence (2024-2025) bổ sung thêm các khối chuyên gia hành động dựa trên khớp dòng (flow-matching action experts). GR00T N1 của NVIDIA (tháng 3 năm 2025) hiện thực hóa kiến trúc điều khiển hai hệ thống (System 1 / System 2) cho robot dạng người (humanoid robots) ở quy mô công nghiệp. Nguyên mẫu VLA (Vision-Language-Action — Thị giác - Ngôn ngữ - Hành động: một mô hình hợp nhất có thể nhìn, đọc và hành động) chính là cầu nối trực tiếp giữa các mô hình thấu hiểu trong phase này với các hệ thống tự chủ trong Phase 15.

**Type:** Learn
**Languages:** Python (stdlib, bộ token hóa hành động + khung suy luận VLA)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, tham chiếu)
**Time:** ~180 phút

## Mục tiêu học tập

- Mô tả cơ chế token hóa hành động (action tokenization): mã hóa theo các phân đoạn rời rạc (discrete bin encoding của RT-2), bộ token hóa hành động hiệu quả FAST, và các hành động khớp dòng liên tục (continuous flow-matching actions của π0).
- Giải thích lý do việc tinh chỉnh đồng thời (co-fine-tuning) trên dữ liệu web kết hợp dữ liệu robot giúp bảo toàn khả năng chuyển giao tri thức tổng quát sang các tác vụ hoàn toàn mới.
- So sánh OpenVLA (mô hình mở 7B Llama + VLM), π0 (khớp dòng) và GR00T N1 (hệ thống kép) trên cùng một tác vụ thao tác robot cụ thể.
- Nêu rõ tập dữ liệu Open X-Embodiment và vai trò nền tảng của nó như một tập ngữ liệu huấn luyện cho dòng RT-X.

## Vấn đề đặt ra

Chế tạo một robot có khả năng làm việc nhà dựa trên chỉ dẫn ngôn ngữ tự nhiên đã là mục tiêu nghiên cứu từ những năm 1970. Lời giải của thập niên 2020: mô hình thị giác - ngôn ngữ - hành động (Vision-Language-Action - VLA). Vẫn là kiến trúc VLM dùng cho hỏi đáp thị giác (VQA), nhưng đầu ra thay vì văn bản thuần túy sẽ là các lệnh hành động (mô-men xoắn khớp, tọa độ khâu tác động cuối - end-effector poses, hoặc các lệnh điều khiển rời rạc).

Những thách thức đặc thù của mô hình VLA:

1. **Không gian hành động là liên tục và đa chiều:** Các góc quay khớp và lực tác động có số chiều rất lớn (ví dụ: cánh tay 7 bậc tự do (7-DOF) + tay kẹp 3 bậc tự do (3-DOF) = 10 chiều điều khiển ở tần số 30 Hz).
2. **Dữ liệu huấn luyện robot cực kỳ khan hiếm:** Open X-Embodiment chỉ có khoảng 1 triệu quỹ đạo thao tác; trong khi dữ liệu văn bản - hình ảnh web lên tới hơn 5 tỷ mẫu.
3. **Tần số điều khiển thời gian thực:** Vòng lặp điều khiển 30 Hz đồng nghĩa mỗi hành động chỉ có hạn mức thời gian tính toán là 33ms.
4. **An toàn vật lý:** Một hành động sai sót có thể phá hủy phần cứng, gây nguy hiểm cho con người hoặc làm hư hỏng tài sản xung quanh.

## Khái niệm cốt lõi

### Token hóa hành động (Kỹ thuật của RT-2)

Ý tưởng cốt lõi của RT-2: biểu diễn giá trị mục tiêu của từng khớp nối robot dưới dạng một token văn bản được lượng tử hóa. Chuẩn hóa dải giá trị chuyển động về đoạn $[-1, 1]$, chia thành 256 phân đoạn rời rạc (bins), sau đó ánh xạ mỗi phân đoạn vào một ID trong từ vựng (vocabulary ID). Khi đó, một hành động 10 bậc tự do (10-DOF) trở thành chuỗi 10 token văn bản tại mỗi bước điều khiển.

Tinh chỉnh đồng thời (co-fine-tune) mô hình VLM PaLM-X trên tập dữ liệu hỗn hợp:

- Cặp hình ảnh - văn bản trên web (chú thích ảnh, hỏi đáp thị giác VQA).
- Các quỹ đạo thao tác mẫu của robot, trong đó hành động được mã hóa thành các token.

Mô hình tiếp nhận: "nhặt khối lập phương màu đỏ" (ngôn ngữ) $\to$ hình ảnh camera (thị giác) $\to$ chuỗi 10 token hành động (giá trị mục tiêu của các khớp đã rời rạc hóa). Quá trình tiền huấn luyện trên web giúp bảo toàn khả năng khái quát hóa: RT-2 có thể thực thi mệnh lệnh "di chuyển lại gần vật thể đang chuyển động nhanh" dù cụm từ "chuyển động nhanh" chưa từng xuất hiện trong dữ liệu huấn luyện của robot.

Tốc độ suy luận trong bài báo RT-2 đạt khoảng 3-5 Hz, bị giới hạn bởi quá trình giải mã tự hồi quy (autoregressive decode) của VLM.

### OpenVLA — Bản tham chiếu mã nguồn mở 7B

OpenVLA (Kim và cộng sự, tháng 6 năm 2024) là bản tương đương có trọng số mở của RT-2. Sử dụng xương sống Llama 7B, bộ mã hóa thị giác kép kết hợp DINOv2 và SigLIP, cơ chế token hóa hành động trên 256 phân đoạn.

Mô hình được huấn luyện trên Open X-Embodiment (970.000 quỹ đạo thu thập từ 22 loại robot khác nhau). OpenVLA hỗ trợ sẵn việc tinh chỉnh bằng LoRA để nhanh chóng thích ứng với phần cứng robot mới.

Tốc độ suy luận: 4-5 Hz trên card A100 khi áp dụng lượng tử hóa. Tốc độ này đủ cho các thao tác gắp nhặt chậm, nhưng chưa đáp ứng được các vòng lặp điều khiển phản xạ tần số cao.

### Bộ token hóa FAST — Tăng tốc giải mã hành động

Pertsch và cộng sự (2024) đã chỉ ra rằng việc mã hóa thành các phân đoạn rời rạc rất kém hiệu quả — phần lớn các hành động thực tế tập trung cục bộ trong một vùng hẹp của không gian phân đoạn. FAST (Frequency-domain Action Sequence Tokenizer - Bộ token hóa chuỗi hành động trên miền tần số) nén toàn bộ quỹ đạo hành động thông qua phép biến đổi cosin rời rạc (DCT) và lượng tử hóa các hệ số tần số.

Một quỹ đạo hành động gồm 30 bước thay vì tiêu tốn 300 token phân đoạn rời rạc nay chỉ cần khoảng 10 token FAST. Tốc độ suy luận tăng từ 3 đến 5 lần mà không làm suy giảm chất lượng điều khiển.

### π0 và hành động dựa trên khớp dòng liên tục (Flow-Matching Actions)

Mô hình π0 của Physical Intelligence (Black và cộng sự, tháng 10 năm 2024) thay thế hoàn toàn các token hành động rời rạc bằng một khối chuyên gia hành động khớp dòng (flow-matching action expert):

- Một transformer hành động nhỏ gọn tiếp nhận các trạng thái ẩn (hidden states) từ VLM và sinh ra trực tiếp một chuỗi hành động liên tục gồm 50 bước thông qua cơ chế dòng hiệu chỉnh (rectified flow).
- Đầu dự đoán hành động (action head) được tối ưu hóa bằng hàm mất mát khớp dòng (flow-matching loss); trong khi tầng VLM tiền huấn luyện được giữ nguyên vẹn.
- Quá trình suy luận: toàn bộ chuỗi hành động được khử nhiễu chỉ trong khoảng 5 bước, tương đương tần số điều khiển thực tế lên tới 50 Hz.

Công bố từ π0 cho thấy nó vượt trội hơn cả OpenVLA và Octo trên hàng loạt tác vụ thao tác phức tạp. Việc mô hình hóa hành động liên tục giúp bảo toàn độ mượt mà của chuyển động — yếu tố vốn bị phá vỡ hoàn toàn khi rời rạc hóa.

Các biến thể nâng cấp gồm có π0.5 và π0-FAST (kết hợp nén FAST với cơ chế khớp dòng).

### GR00T N1 — Kiến trúc điều khiển hai hệ thống cho robot hình người

GR00T N1 của NVIDIA (tháng 3 năm 2025) được thiết kế chuyên biệt cho robot dạng người (toàn thân, hơn 30 bậc tự do):

- **Hệ thống 2 (System 2):** Một VLM quy mô lớn phân tích bối cảnh hình ảnh và chỉ dẫn ngôn ngữ, sinh ra các mục tiêu phụ cấp cao (high-level subgoals) ở tần số khoảng 1 Hz.
- **Hệ thống 1 (System 1):** Một transformer đầu hành động nhỏ sinh ra các lệnh điều khiển khớp chi tiết tần số cao từ 50-100 Hz có điều kiện dựa trên các mục tiêu phụ của Hệ thống 2.

Sự phân chia này tương ứng với học thuyết tư duy nhanh và chậm của Daniel Kahneman: Hệ thống 2 lập kế hoạch chiến lược, Hệ thống 1 thực thi phản xạ cơ học. Ưu điểm nổi bật: quá trình lập kế hoạch nặng nề của VLM không làm nghẽn vòng lặp điều khiển phản xạ nhanh; Hệ thống 1 luôn giữ kích thước nhỏ để đảm bảo độ trễ tối thiểu.

Phiên bản GR00T N1.7 (cuối năm 2025) mở rộng quy mô dữ liệu thông qua việc tinh chỉnh bằng dữ liệu mô phỏng chuyển giao sang thực tế (sim-to-real) từ nền tảng NVIDIA Omniverse.

### Tập dữ liệu Open X-Embodiment

Cơ sở dữ liệu huấn luyện tiêu chuẩn. Dự án RT-X (tháng 10 năm 2023) đã tổng hợp 22 bộ dữ liệu với 1 triệu quỹ đạo trên 22 loại robot khác nhau thành tập dữ liệu quy chuẩn Open X-Embodiment:

- Bao gồm các nền tảng: ALOHA, Bridge V2, Droid, RT-2 Kitchen, Language Table.
- Cấu trúc mỗi mẫu dữ liệu: (trạng thái robot, góc nhìn camera, chỉ dẫn ngôn ngữ, chuỗi hành động thực tế).
- Tiêu chuẩn làm sạch dữ liệu: hợp nhất không gian hành động, chuẩn hóa giới hạn góc xoay của các khớp, đồng bộ kích thước khung hình camera.

Cả OpenVLA và π0 đều được huấn luyện trên Open X-Embodiment. Khoảng cách chênh lệch miền (domain gap) đối với bất kỳ cánh tay robot cụ thể nào sau đó được thu hẹp thông qua tinh chỉnh LoRA trên 100 đến 1.000 quỹ đạo biểu diễn mẫu của tác vụ đó.

### So sánh giữa tinh chỉnh đồng thời và chỉ huấn luyện trên dữ liệu robot

Tinh chỉnh đồng thời phối trộn dữ liệu VQA trên web cùng các quỹ đạo thao tác robot. Tỷ lệ pha trộn mang tính sống còn: quá nhiều dữ liệu VQA sẽ khiến mô hình quên mất cách điều khiển hành động; quá nhiều dữ liệu robot sẽ làm mô hình đánh mất tri thức thế giới tự nhiên.

- Tỷ lệ của RT-2: xấp xỉ 1:1.
- OpenVLA: xấp xỉ 0,5:1 giữa dữ liệu web và robot.
- π0: áp dụng tỷ lệ tương tự. Tỷ lệ chính xác là một siêu tham số cần hiệu chỉnh theo từng quy mô tập dữ liệu.

Việc chỉ huấn luyện thuần túy trên dữ liệu robot tạo ra các mô hình hẹp chuyên biệt, lập tức thất bại khi gặp các câu lệnh nằm ngoài phân phối huấn luyện. Tinh chỉnh đồng thời tạo ra sự khác biệt cốt lõi giữa việc chỉ hiểu "nhặt khối lập phương màu đỏ (đã có trong demo)" và việc có thể suy luận "hãy nhặt vật thể lớn thứ ba tính từ bên trái (cách diễn đạt mới mẻ)".

### Cơ chế an toàn và giới hạn hành động phần cứng

Mọi hệ thống VLA vận hành trong thực tế đều bắt buộc phải tích hợp các rào chắn kiểm soát an toàn độc lập:

- Giới hạn cứng của các khớp (không cho phép cấp mô-men xoắn vượt quá thông số kỹ thuật).
- Giới hạn vận tốc chuyển động (cắt giảm gia tốc mềm).
- Vùng không gian làm việc an toàn (bảo đảm khâu tác động cuối không được vượt ra ngoài phạm vi mặt bàn).
- Cơ chế phê duyệt của con người trong vòng lặp (human-in-the-loop) đối với các tác vụ mới phát sinh.

Các cơ chế này nằm hoàn toàn ở tầng điều khiển bên ngoài VLA. Đầu ra của mô hình VLA chỉ được xem là các đề xuất hành động (suggestions), chứ không phải là mệnh lệnh trực tiếp tới phần cứng.

```figure
mm-action-tokens
```

## Sử dụng trong thực tế

Tệp `code/main.py`:

- Triển khai thuật toán token hóa và giải mã hành động trên 256 phân đoạn rời rạc.
- Phác thảo bộ token hóa FAST dựa trên phép biến đổi DCT và lượng tử hóa tần số.
- So sánh số lượng token phát sinh trên mỗi bước hành động giữa các phương pháp: phân đoạn rời rạc (discrete-bin), nén tần số (FAST), và khớp dòng liên tục (continuous-flow).
- In sơ đồ tóm lược dòng phát triển phả hệ từ RT-2 $\to$ OpenVLA $\to$ π0 $\to$ GR00T.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-vla-action-format-picker.md`. Dựa trên yêu cầu tác vụ robot cụ thể (thao tác gắp nhặt, dẫn đường di động, điều khiển toàn thân robot hình người), kỹ năng này tự động khuyến nghị lựa chọn giữa: phân đoạn rời rạc + RT-2, FAST + OpenVLA, khớp dòng + π0, hoặc kiến trúc hệ thống kép + GR00T.

## Bài tập thực hành

1. Xét cánh tay robot 10 bậc tự do (10-DOF) hoạt động ở tần số điều khiển 30 Hz. Khi áp dụng mã hóa 256 phân đoạn rời rạc, hệ thống phát sinh bao nhiêu token mỗi giây? Liệu một mô hình VLM 7B có đủ năng lực xử lý kịp theo thời gian thực không?

2. Bộ token hóa FAST nén quỹ đạo 30 bước xuống chỉ còn khoảng 10 token. Người dùng sẽ phải đánh đổi điều gì nếu quỹ đạo chuyển động chứa các dao động tần số cao (ví dụ: thao tác gõ trống liên tục)?

3. Đầu dự đoán khớp dòng của π0 khử nhiễu hoàn tất trong khoảng 5 bước. Hãy so sánh thông lượng của nó với quá trình giải mã tự hồi quy của OpenVLA ở tần số 4-5 Hz.

4. Kiến trúc phân tách Hệ thống 1 / Hệ thống 2 của GR00T lấy cảm hứng từ tâm lý học hành vi của Kahneman. Hãy đề xuất một cấu trúc phân tách mới (chẳng hạn bổ sung thêm Hệ thống 3) có thể hỗ trợ hiệu quả cho việc giữ thăng bằng khi bước đi bằng hai chân.

5. Đọc Mục 4 trong bài báo Open X-Embodiment về quy trình tuyển chọn dữ liệu. Nêu ba nguyên tắc làm sạch dữ liệu cốt lõi giúp ngăn ngừa hiện tượng rò rỉ phân phối giữa các miền robot khác nhau.

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| VLA | "Thị giác - ngôn ngữ - hành động" | Mô hình tiếp nhận hình ảnh + câu lệnh chỉ dẫn và sinh trực tiếp các lệnh hành động điều khiển |
| Token hóa hành động (Action tokenization) | "Phân đoạn rời rạc" | Lượng tử hóa góc khớp liên tục thành 256 khoảng trên mỗi chiều, mỗi khoảng là một ID từ vựng |
| Bộ token hóa FAST (FAST tokenizer) | "Token hành động miền tần số" | Áp dụng DCT + lượng tử hóa để nén quỹ đạo 30 bước thành khoảng 10 token |
| Tinh chỉnh đồng thời (Co-fine-tune) | "Trộn dữ liệu web + robot" | Huấn luyện đồng thời trên VQA web và quỹ đạo robot nhằm duy trì tri thức khái quát hóa |
| Khối hành động khớp dòng (Flow-matching action head) | "Đầu ra liên tục của π0" | Transformer nhỏ sinh chuỗi hành động liên tục 50 bước thông qua cơ chế dòng hiệu chỉnh |
| Hệ thống 1 / Hệ thống 2 (System 1 / System 2) | "Điều khiển hai hệ thống" | VLM lớn lập kế hoạch chậm, đầu hành động nhỏ phản xạ nhanh; cấu trúc của GR00T |
| Open X-Embodiment | "Tập dữ liệu RT-X" | Bộ dữ liệu hợp nhất gồm 1 triệu quỹ đạo trên 22 loại robot; kho ngữ liệu huấn luyện chuẩn |

## Tài liệu đọc thêm

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
