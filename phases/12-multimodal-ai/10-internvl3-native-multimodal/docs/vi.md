# InternVL3: Tiền huấn luyện đa phương thức tự nhiên

> Mọi mô hình ngôn ngữ - thị giác (Vision-Language Model - VLM) trọng số mở trước thời InternVL3 đều đi theo một công thức ba bước giống nhau: lấy một LLM văn bản đã được huấn luyện trên hàng nghìn tỷ token văn bản, ghép nối thêm một bộ mã hóa thị giác (vision encoder), rồi sau đó tinh chỉnh các mối nối liên kết. Cách tiếp cận này hiệu quả nhưng để lại "món nợ căn chỉnh" (alignment debt) — LLM văn bản đã tiêu tốn toàn bộ ngân sách tiền huấn luyện cho văn bản thuần túy và không thể hiểu các token thị giác một cách tự nhiên. Khi bổ sung thị giác theo kiểu chắp vá hậu kỳ (post-hoc), LLM buộc phải học lại cách liên kết đầu vào thị giác với năng lực suy luận văn bản của nó mà không được làm mất đi các kỹ năng văn bản sẵn có. InternVL3 (Zhu et al., tháng 4 năm 2025) bác bỏ cách tiếp cận chắp vá này: một lượt tiền huấn luyện duy nhất từ đầu, trong đó dữ liệu văn bản và dữ liệu đa phương thức được xen kẽ ngay từ bước gradient đầu tiên. Kết quả đạt được ngang ngửa với Gemini 2.5 Pro trên benchmark MMMU-Pro ở quy mô 78B tham số mã nguồn mở. Bài học này phân tích lập luận cốt lõi của tiền huấn luyện tự nhiên và những thay đổi sâu sắc khi áp dụng phương pháp này.

**Type:** Học (Learn)
**Languages:** Python (thư viện chuẩn, bộ trộn kho ngữ liệu huấn luyện)
**Prerequisites:** Giai đoạn 12 · 05, Giai đoạn 12 · 07 (công thức huấn luyện)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích tại sao quy trình huấn luyện VLM hậu kỳ (post-hoc) lại tích lũy nợ căn chỉnh (alignment debt), chỉ ra ba triệu chứng định lượng cụ thể (quên thảm khốc, trôi dạt câu trả lời, không nhất quán thị giác - văn bản).
- Mô tả tỷ lệ pha trộn kho ngữ liệu tiền huấn luyện tự nhiên của InternVL3 và lý do tại sao tỷ lệ văn bản : tài liệu xen kẽ : ảnh-chú thích lại mang tính quyết định.
- So sánh cơ chế mã hóa vị trí thị giác biến thiên V2PE (variable visual position encoding) với M-RoPE của Qwen2-VL.
- Điểm tên các giải pháp tối ưu hóa triển khai phục vụ thực tế: Bộ điều hướng phân giải thị giác ViR (Visual Resolution Router) và Kiến trúc triển khai phân tách Ngôn ngữ - Thị giác DvD (Decoupled Vision-Language).

## Vấn đề cần giải quyết

Huấn luyện VLM chắp vá hậu kỳ (post-hoc) là lựa chọn mặc định trong ngành. LLaVA, BLIP-2, Qwen-VL, Idefics — tất cả đều lấy một LLM đã được tiền huấn luyện sẵn (Llama, Vicuna, Qwen, Mistral) rồi bổ sung thêm nhánh thị giác. Các giai đoạn huấn luyện thường diễn ra như sau:

1. Đóng băng LLM + đóng băng bộ mã hóa thị giác + chỉ huấn luyện bộ chiếu (projector), sử dụng các cặp ảnh - chú thích để căn chỉnh không gian nhúng (embeddings).
2. Mở băng LLM, huấn luyện trên dữ liệu tinh chỉnh chỉ dẫn (LLaVA-Instruct, ShareGPT4V).
3. (Tùy chọn) Tinh chỉnh chuyên biệt theo từng tác vụ nghiệp vụ.

Quy trình này làm phát sinh ba triệu chứng điển hình của nợ căn chỉnh:

- **Hiện tượng quên thảm khốc (Catastrophic forgetting):** VLM hậu kỳ đánh mất dần các kỹ năng xử lý văn bản thuần túy. Điểm số giải toán GSM8K sụt giảm từ 5-10 điểm. Điểm Hellaswag giảm sút. Năng lực của các agent văn bản thuần túy bị thoái hóa.
- **Trôi dạt câu trả lời (Answer drift):** Những biến thể diễn đạt nhỏ trong cùng một câu hỏi thị giác lại nhận về các câu trả lời khác biệt. Bộ mã hóa thị giác liên kết với LLM qua các mối nối lỏng lẻo hơn nhiều so với các token văn bản nội tại của LLM.
- **Không nhất quán giữa thị giác và văn bản (Visual-text inconsistency):** VLM có thể mô tả một bức ảnh hoàn toàn chính xác nhưng ngay sau đó lại đưa ra câu trả lời mâu thuẫn trực tiếp với chính phần mô tả đó. Các token thị giác không tham gia vào các cơ chế kiểm tra tính nhất quán nội tại của LLM theo cách mà token văn bản làm được.

Các triệu chứng này đã được ghi nhận và lượng hóa chi tiết trong Phần 4 của bài báo MM1.5 cũng như trong các thử nghiệm ablation của LLaVA-OneVision. Tiền huấn luyện đa phương thức tự nhiên chính là câu trả lời triệt để cho vấn đề này.

## Khái niệm cốt lõi

### Tiền huấn luyện đa phương thức tự nhiên

InternVL3 huấn luyện mô hình từ đầu (scratch) trên một kho ngữ liệu đa phương thức tự nhiên ngay từ bước đầu tiên. Tỷ lệ pha trộn dữ liệu bao gồm:

- 40% dữ liệu văn bản thuần túy (FineWeb, Proof-Pile-2, v.v.)
- 35% dữ liệu tài liệu xen kẽ hình ảnh - văn bản (định dạng tương tự OBELICS, MMC4)
- 20% dữ liệu cặp ảnh - chú thích (image-caption pairs)
- 5% dữ liệu video - văn bản

Các token thị giác, token văn bản và các tương tác xuyên phương thức đều tham gia trực tiếp vào cùng một hàm mất mát (loss function) ngay từ bước tối ưu gradient đầu tiên. Không cần giai đoạn tiền huấn luyện căn chỉnh riêng biệt, không cần đóng băng projector, và không có hiện tượng quên thảm khốc nào cần phải khắc phục.

Quá trình huấn luyện mô hình nền tảng chỉ diễn ra trong một giai đoạn duy nhất. Quá trình tinh chỉnh chỉ dẫn diễn ra sau đó, nhưng bản thân mô hình nền tảng đã thấu hiểu các token thị giác như những thực thể hạng nhất (first-class citizens).

### V2PE (Mã hóa vị trí thị giác biến thiên)

Qwen2-VL sử dụng M-RoPE với sự phân bổ trục cố định. InternVL3 giới thiệu V2PE: cơ chế mã hóa vị trí thay đổi linh hoạt theo từng loại phương thức (văn bản, hình ảnh, video) thông qua các hệ số co giãn có thể học được. Trong thực tế:

- Token văn bản nhận vị trí 1D (chỉ số thứ tự văn bản).
- Patch hình ảnh nhận vị trí 2D (hàng, cột).
- Khung hình video nhận vị trí 3D (thời gian, hàng, cột).

Cả ba phương thức chia sẻ cùng một cơ sở tần số RoPE, nhưng tỷ lệ phân bổ chiều ẩn (hidden-dimension allocation) cho từng dải là một tham số được học tự động thay vì chia cố định cứng nhắc. Điều này mang lại sự tự do trong việc đánh đổi giữa độ phân giải tần số thời gian và không gian trong suốt quá trình tiền huấn luyện.

Kết quả thử nghiệm ablation của V2PE: mang lại mức tăng 1-2 điểm trên các benchmark video so với M-RoPE ở cùng mức chi phí tính toán. Đây không phải một cuộc cách mạng quá lớn, nhưng là một thiết kế mạch lạc và thanh thoát hơn.

### Bộ điều hướng phân giải thị giác (ViR)

Đây là giải pháp tối ưu hóa phục vụ giai đoạn triển khai (inference). Không phải bức ảnh nào cũng cần mã hóa ở độ phân giải tối đa. Một bức ảnh chụp một đồ vật đơn lẻ với ít chi tiết sẽ lãng phí tài nguyên tính toán nếu bị ép mã hóa ở độ phân giải tự nhiên 1280px. ViR là một bộ phân loại nhỏ gọn giúp dự đoán độ phân giải tối thiểu cần thiết để trả lời câu hỏi, diễn ra trước khi bước vào khâu mã hóa hình ảnh.

Hệ thống định tuyến chia thành ba tầng: phân giải thấp (256 token), phân giải trung bình (576 token), và phân giải cao (2048+ token). Đối với 60% lượng truy vấn trong môi trường sản xuất thực tế, tầng phân giải thấp hoặc trung bình là hoàn toàn đủ đáp ứng. Hiệu quả ròng: tăng thông lượng phục vụ lên 2-3 lần trong khi vẫn bảo toàn nguyên vẹn chất lượng đầu ra.

### Triển khai phân tách Ngôn ngữ - Thị giác (DvD)

Khi phục vụ một mô hình VLM lớn, bộ mã hóa thị giác chỉ chạy đúng một lần trên mỗi bức ảnh nhưng LLM lại chạy lặp tự hồi quy (autoregressive) cho từng token sinh ra ở đầu ra. Hai thành phần này đối mặt với các điểm nghẽn phần cứng hoàn toàn khác nhau (thị giác nghẽn băng thông bộ nhớ GPU cho phép tích chập và cơ chế chú ý; trong khi LLM nghẽn ở bộ nhớ đệm KV cache). DvD tách hai khối này sang các cụm GPU riêng biệt với luồng truyền dữ liệu streaming giữa chúng.

Đối với mô hình gồm LLM 8B kết hợp bộ mã hóa 400M, kiến trúc DvD giúp tăng gấp đôi thông lượng trên mỗi nút máy chủ so với cách đặt chung trên cùng một GPU.

### Chất lượng đơn giai đoạn so với đa giai đoạn

Tuyên bố benchmark chủ đạo của InternVL3: ở quy mô 78B tham số, mô hình sánh ngang Gemini 2.5 Pro trên MMMU-Pro. Ở bản 38B, tương đương GPT-4o. Ở bản 8B, dẫn đầu toàn bộ bảng xếp hạng các mô hình 8B mã nguồn mở. Tất cả đều đạt được chỉ nhờ công thức: tiền huấn luyện tự nhiên đơn giai đoạn + tinh chỉnh chỉ dẫn.

Giả thuyết về "nợ căn chỉnh" có thể đo lường định lượng: InternVL3-8B đánh mất ít điểm benchmark văn bản hơn (MMLU, GSM8K) so với Qwen2.5-VL-7B trên mỗi đơn vị điểm benchmark thị giác gia tăng. Mô hình thể hiện tính chất đa năng toàn diện hơn nhờ quá trình huấn luyện diễn ra như một thể thống nhất, thay vì bị chắp vá hai mảnh.

### InternVL3.5 và InternVL-U

InternVL3.5 (Tháng 8 năm 2025) mở rộng quy mô công thức này: giữ nguyên triết lý tiền huấn luyện tự nhiên, bổ sung thêm dữ liệu và tham số. Mức cải thiện trên MMMU mang tính tích lũy đều đặn.

InternVL-U (2026) bổ sung khả năng sinh ảnh hợp nhất — cho phép xuất ra hình ảnh thông qua các đầu sinh MMDiT đặt ngay trên cùng một xương sống mô hình. Ký tự "U" đại diện cho "Understanding + generation" (Hiểu + Sinh), theo đuổi các mô hình thống nhất theo phong cách Transfusion (Bài học 12.13). Cùng một khung xương tiền huấn luyện tự nhiên duy nhất có thể nuôi dưỡng cả đầu tác vụ hiểu lẫn đầu tác vụ sinh ảnh.

### Đánh đổi trong tiền huấn luyện tự nhiên

Tiền huấn luyện tự nhiên không phải là giải pháp miễn phí:

- **Chi phí tính toán (Compute):** Huấn luyện một VLM hoàn toàn mới từ đầu tiêu tốn tài nguyên ngang ngửa với việc huấn luyện một LLM văn bản thuần túy — hàng triệu giờ GPU. Việc thích ứng hậu kỳ tái sử dụng trọng số LLM sẵn có, tiết kiệm phần lớn chi phí này.
- **Nguồn dữ liệu (Data):** Các kho ngữ liệu xen kẽ hình ảnh - văn bản quy mô lớn là cực kỳ khan hiếm. OBELICS chỉ có 141 triệu tài liệu; MMC4 có 571 triệu. Trong khi dữ liệu văn bản thuần túy đã vượt mốc 15 nghìn tỷ token. Sự khan hiếm dữ liệu tiền huấn luyện đa phương thức là một rào cản vật lý khắc nghiệt.
- **Khả năng tái sử dụng LLM nền tảng:** Tiền huấn luyện tự nhiên làm mất đi khả năng hoán đổi LLM mới sau này. Phương pháp hậu kỳ cho phép bạn nâng cấp từ Llama-3.1 lên Llama-4 chỉ bằng cách huấn luyện lại adapter kết nối.

Canh bạc mà InternVL3 đặt cược là: cái giá của món nợ căn chỉnh nặng nề hơn sự mất mát về tính tái sử dụng. Các kết quả benchmark đã chứng minh nhận định này. Tuy nhiên, rào cản chi phí sản xuất khổng lồ sẽ ngăn cản các phòng thí nghiệm nhỏ sao chép trực tiếp. Do đó, các VLM hậu kỳ sẽ tiếp tục tồn tại vì chúng vẫn là lựa chọn kinh tế nhất cho phần lớn các dự án.

```figure
l5-native-pretrain
```

## Ứng dụng thực tế

Tệp `code/main.py` là một bộ mô phỏng trộn kho ngữ liệu huấn luyện và điều hướng phân giải ViR. Chương trình:

- Nhận tỷ lệ pha trộn kho ngữ liệu mục tiêu (% văn bản, % xen kẽ, % chú thích, % video) và tính toán số bước huấn luyện dự kiến cho từng phương thức.
- Mô phỏng quá trình định tuyến ViR trên một batch câu hỏi thực tế (phân phối: 50% chi tiết thấp, 30% trung bình, 20% chi tiết cao) và xuất ra số lượng token trung bình tiêu thụ.
- Ước lượng thông lượng phục vụ của DvD dựa trên số phép tính FLOP của bộ mã hóa so với LLM.
- In bảng đối chiếu trực quan giữa tiền huấn luyện hậu kỳ và tự nhiên về tham số, chi phí tính toán, nhu cầu dữ liệu và các triệu chứng nợ căn chỉnh dự kiến.

## Đóng gói triển khai

Bài học này tạo ra tệp artifact `outputs/skill-native-vs-posthoc-auditor.md`. Dựa trên kế hoạch huấn luyện VLM được đề xuất, kỹ năng này sẽ kiểm định xem dự án nên đi theo hướng tự nhiên hay hậu kỳ, cảnh báo các rủi ro nợ căn chỉnh và đề xuất tỷ lệ pha trộn kho ngữ liệu tối ưu. Hãy sử dụng kỹ năng này khi bạn ước tính quy mô cho một dự án VLM mã nguồn mở mới và cần chốt chiến lược huấn luyện.

## Bài tập

1. Ước tính mức chênh lệch tài nguyên tính toán giữa InternVL3-8B (tiền huấn luyện tự nhiên) và LLaVA-OneVision-7B (thích ứng hậu kỳ). Tỷ lệ giờ GPU xấp xỉ là bao nhiêu? Yếu tố nào giải thích cho khoảng cách khổng lồ này?

2. InternVL3 công bố tỷ lệ 40% văn bản / 35% xen kẽ / 20% chú thích / 5% video. Nếu mục tiêu sản phẩm của bạn tập trung rất nặng vào video, hãy đề xuất một tỷ lệ pha trộn mới và lập luận tại sao mô hình nền tảng vẫn bắt buộc phải duy trì lượng lớn dữ liệu văn bản và chú thích hình ảnh.

3. Đọc Phần 4 của bài báo MM1.5 về hiện tượng quên tri thức. Chỉ ra chính xác benchmark nào ghi nhận sự thụt lùi hiệu năng nghiêm trọng nhất trong huấn luyện hậu kỳ. Mức độ suy giảm đó là bao nhiêu?

4. Bộ điều hướng ViR chuyển 60% lưu lượng truy vấn về mức phân giải thấp. Những loại câu hỏi nào dễ bị phân loại nhầm nhất (bị chuyển về mức phân giải thấp trong khi thực tế cần độ phân giải cao)? Đề xuất ba trường hợp thất bại điển hình của router.

5. Kiến trúc DvD tách biệt bộ mã hóa thị giác và LLM sang các GPU độc lập. Dưới mô hình lưu lượng truy cập (traffic pattern) nào thì DvD sẽ làm giảm thông lượng thay vì tối ưu hóa hệ thống?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Tiền huấn luyện đa phương thức tự nhiên (Native multimodal pretraining) | "Huấn luyện cùng nhau từ đầu" | Các token văn bản + hình ảnh + video cùng tham gia trực tiếp vào hàm mất mát ngay từ bước 1, không chắp vá sau |
| Nợ căn chỉnh (Alignment debt) | "Hậu quả của huấn luyện hậu kỳ" | Sự suy thoái định lượng về kỹ năng ngôn ngữ và tính nhất quán câu trả lời do việc ghép nối module thị giác vào một LLM đóng băng |
| V2PE | "Mã hóa vị trí thị giác biến thiên" | Cơ chế phân bổ mã hóa vị trí có thể học được theo từng phương thức; bước phát triển kế tiếp M-RoPE trong InternVL3 |
| ViR | "Bộ điều hướng phân giải" | Bộ phân loại nhỏ xác định độ phân giải tối thiểu cần thiết cho từng câu hỏi trước khi mã hóa, giúp tiết kiệm token suy luận |
| DvD | "Triển khai phân tách" | Đặt bộ mã hóa thị giác trên một GPU, LLM trên GPU khác với đường truyền streaming liên tục; tăng gấp đôi thông lượng VLM lớn |
| InternVL-U | "Hiểu + sinh hợp nhất" | Phiên bản năm 2026 bổ sung thêm các đầu sinh hình ảnh trên cùng một khung xương tiền huấn luyện tự nhiên |
| Kho ngữ liệu xen kẽ (Interleaved corpus) | "OBELICS / MMC4" | Tài liệu chứa văn bản và hình ảnh đan xen theo trình tự đọc tự nhiên; nguyên liệu cốt lõi cho tiền huấn luyện tự nhiên |

## Tài liệu tham khảo thêm

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
