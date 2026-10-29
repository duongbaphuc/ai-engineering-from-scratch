# LLaVA-OneVision: Đơn ảnh, Đa ảnh và Video trong một mô hình duy nhất

> Trước LLaVA-OneVision (Li et al., tháng 8 năm 2024), thế giới mô hình ngôn ngữ - thị giác (Vision-Language Model - VLM) trọng số mở bị phân mảnh thành các nhánh riêng biệt: LLaVA-1.5 cho ảnh đơn lẻ (single-image), các mô hình đa ảnh (multi-image) như Mantis và VILA, cùng các mô hình video như Video-LLaVA và Video-LLaMA. Mỗi mô hình chỉ chiến thắng trên benchmark chuyên biệt của mình và thất bại ở các trường hợp còn lại. LLaVA-OneVision chứng minh rằng một giáo trình huấn luyện (curriculum) duy nhất có thể đào tạo một mô hình làm chủ cả ba kịch bản, và các hiệu ứng chuyển giao kỹ năng đột sinh (emergent task-transfer effects) — từ kỹ năng ảnh đơn chuyển sang video, từ suy luận đa ảnh chuyển sang ảnh đơn — vượt trội hơn hẳn tổng năng lực của các mô hình chuyên biệt cộng lại. Công thức này tinh gọn đến bất ngờ: một ngân sách token thị giác (visual-token budget) được cân bằng giữa các kịch bản, kết hợp với một giáo trình huấn luyện bài bản chuyển từ ảnh đơn sang OneVision (đa ảnh) rồi tới video. Bài học này phân tích chi tiết cách phân bổ ngân sách, thiết kế giáo trình và các hành vi đột sinh này.

**Type:** Xây dựng (Build)
**Languages:** Python (thư viện chuẩn, bộ giải ngân sách token + bộ lập kế hoạch giáo trình)
**Prerequisites:** Giai đoạn 12 · 05 (LLaVA), Giai đoạn 12 · 06 (độ phân giải tùy ý - AnyRes)
**Time:** ~180 phút

## Mục tiêu học tập

- Thiết kế ngân sách token thị giác duy trì sự cân bằng giữa các đầu vào dạng ảnh đơn, đa ảnh và video.
- Sắp xếp thứ tự giáo trình huấn luyện để chuyển giao kỹ năng từ ảnh đơn sang video mà không gây ra hiện tượng quên thảm khốc (catastrophic forgetting).
- Giải thích tại sao một mô hình duy nhất đánh bại các mô hình chuyên biệt ở cùng số lượng tham số khi áp dụng giáo trình huấn luyện chuẩn xác.
- Gọi tên ba năng lực đột sinh được công bố bởi LLaVA-OneVision: suy luận đa góc quay (multi-camera reasoning), kỹ thuật nhắc theo tập đánh dấu (set-of-mark prompting), và agent thao tác qua ảnh chụp màn hình iPhone (iPhone-screenshot agent).

## Vấn đề cần giải quyết

Ảnh đơn, đa ảnh và video đặt ra các áp lực tính toán rất khác nhau lên mô hình.

- **Ảnh đơn** đòi hỏi token độ phân giải cao (AnyRes, ~2880 token thị giác) để đọc ký tự quang học (OCR) và nắm bắt các chi tiết cực nhỏ. Ngân sách cho mỗi mẫu: 1 ảnh, 2880 token.
- **Đa ảnh** đòi hỏi nhiều hình ảnh ở độ phân giải vừa phải (~576 token mỗi ảnh) để việc suy luận xuyên suốt nhiều ảnh vừa vặn với chiều dài ngữ cảnh (context length). Ngân sách cho mỗi mẫu: 4-8 ảnh, 576 token mỗi ảnh, tổng cộng 2300-4600 token.
- **Video** đòi hỏi rất nhiều khung hình ở độ phân giải thấp (~196 token mỗi khung hình sau khi gộp) nhằm ghi nhận động lực thời gian (temporal dynamics). Ngân sách cho mỗi mẫu: 8-32 khung hình, 196 token mỗi khung hình, tổng cộng 1600-6200 token.

Nếu huấn luyện các mô hình riêng biệt, bạn chỉ cần chọn một ngân sách cố định. Nhưng nếu muốn huấn luyện một mô hình duy nhất, bạn cần một ngân sách mở rộng hợp lý qua cả ba kịch bản mà không làm bùng nổ ngữ cảnh của LLM.

Trước thời điểm OneVision ra đời, câu trả lời mặc định luôn là "chỉ huấn luyện một kịch bản, bỏ qua các kịch bản còn lại". Video-LLaVA chắp vá thêm khả năng video vào một mô hình ảnh sẵn có bằng các giai đoạn huấn luyện bổ sung. LLaVA-NeXT thêm hỗ trợ đa ảnh bằng cách chia mảnh (tiling). Không một giải pháp nào xử lý mượt mà cả ba kịch bản cùng lúc.

## Khái niệm cốt lõi

### Ngân sách token trong OneVision

LLaVA-OneVision không cố định một con số token duy nhất. Thay vào đó, nó cân bằng số lượng token thị giác tối đa trên mỗi mẫu sao cho cả ba kịch bản đều rơi vào cùng một khoảng biên độ, xấp xỉ từ 6.000 đến 9.000 token, được phân bổ linh hoạt theo từng kịch bản:

- **Ảnh đơn (Single image):** AnyRes-Max-9 (tối đa 9 mảnh cắt + 1 góc nhìn toàn cảnh cơ sở), mỗi góc nhìn 384x384 tạo ra 729 token từ bộ mã hóa SigLIP SO400M. Tối đa: 729 * (1 + 9) = 7290 token.
- **Đa ảnh (Multi-image):** Mỗi ảnh giữ nguyên kích thước 384 không chia mảnh, tương ứng 729 token không gộp. Hỗ trợ tối đa 12 ảnh → 8748 token.
- **Video:** Tối đa 32 khung hình ở kích thước 384 với phép gộp song tuyến tính 2x2 (2x2 bilinear pooling) → 196 token mỗi khung hình. Tổng cộng: 32 * 196 = 6272 token.

Cách phân bổ này giữ cho tổng số token tối đa nằm trong cùng một ngưỡng, giúp không kịch bản nào chiếm đoạt toàn bộ ngữ cảnh hoặc áp đảo gradient cập nhật. Bộ mã hóa sinh ra các cấu trúc hình học khác nhau theo từng kịch bản, nhưng LLM luôn nhận một ngân sách tương đương nhau. Tệp mã nguồn `code/main.py` dưới đây sử dụng một ngân sách mô phỏng nhỏ hơn là 4096 để minh họa trực quan logic cân bằng này.

### Giáo trình huấn luyện ba giai đoạn

LLaVA-OneVision trải qua quá trình huấn luyện gồm ba giai đoạn:

1. **Tinh chỉnh chỉ dẫn ảnh đơn (Single-image SFT - giai đoạn SI):** Toàn bộ dữ liệu là các cặp ảnh đơn kèm văn bản. Huấn luyện trên đầu vào AnyRes độ phân giải cao. Giai đoạn này xây dựng khả năng cảm nhận thị giác, đọc chữ OCR và hiểu chi tiết tinh tế. Sử dụng dữ liệu của LLaVA-NeXT kết hợp dữ liệu ảnh đơn đặc thù của OneVision.
2. **Tinh chỉnh chỉ dẫn OneVision (OneVision SFT - giai đoạn OV):** Trộn lẫn ảnh đơn + đa ảnh + video (lấy mẫu khung hình đồng đều). Huấn luyện trên ngân sách token thống nhất. Giai đoạn này rèn luyện cho mô hình khả năng xử lý các hình dạng batch hỗn hợp (heterogeneous batch shapes). Không đặt lại trọng số (no weight reset) — tiếp tục huấn luyện kế thừa từ giai đoạn SI.
3. **Chuyển giao nhiệm vụ (Task transfer - giai đoạn TT):** Tiếp tục huấn luyện với hỗn hợp nhiệm vụ mục tiêu, thường tập trung nặng hơn vào đa ảnh hoặc video tùy theo nhu cầu ứng dụng thực tế. Đây là bước tinh chỉnh tùy chọn trước khi triển khai sản xuất.

Điều tối quan trọng: thứ tự của giáo trình đóng vai trò quyết định. Huấn luyện video trước hoặc đa ảnh trước sẽ làm suy giảm hiệu năng hiểu ảnh đơn so với việc huấn luyện ảnh đơn trước, ngay cả khi dùng cùng một tập dữ liệu. Bài báo gốc đã chứng minh điều này qua các thử nghiệm ablation rõ ràng.

### Tại sao giáo trình huấn luyện lại hiệu quả

Huấn luyện ảnh đơn xây dựng nền tảng cảm nhận thị giác (perceptual base). Các token patch mang theo những đặc trưng thị giác chi tiết; LLM học cách tích hợp chúng với văn bản. Đa ảnh và video giới thiệu thêm các thách thức về mặt cấu trúc (ảnh nào là ảnh nào, sự kiện nào xảy ra trước) — những điều rất khó học nếu thiếu đi một nền tảng cảm nhận vững chắc.

Nếu bạn huấn luyện đồng thời tất cả các kịch bản từ đầu, mô hình sẽ bị khớp dưới (underfit) về năng lực thị giác (do lượng dữ liệu ảnh đơn trên mỗi batch bị hạn chế) và khớp quá (overfit) về mặt cấu trúc (do có quá nhiều dữ liệu đa ảnh/video). Kết quả tạo ra một mô hình suy luận logic đa ảnh theo khuôn mẫu nhưng khả năng nhìn nhận chi tiết lại rất nông cạn.

Thứ tự giáo trình đảm bảo bạn sở hữu năng lực thị giác sắc bén từ giai đoạn SI, sau đó tiếp thu khả năng suy luận kết hợp/thời gian từ giai đoạn OV mà không làm mất đi năng lực ban đầu.

### Năng lực đột sinh giữa các kịch bản

Bài báo LLaVA-OneVision công bố ba năng lực đột sinh đáng chú ý:

1. **Suy luận đa góc quay (Multi-camera reasoning):** Dù chỉ được huấn luyện riêng biệt trên dữ liệu đa ảnh và video thông thường; nhưng khi suy luận, mô hình được yêu cầu phân tích một khung cảnh lái xe từ nhiều camera cùng lúc. Mô hình đã tích hợp chính xác các góc nhìn dù chưa từng thấy định dạng đó trong dữ liệu huấn luyện.
2. **Kỹ thuật nhắc theo tập đánh dấu (Set-of-mark prompting):** Người dùng đánh dấu các đối tượng trong ảnh bằng các ký hiệu số; mô hình có thể suy luận "đối tượng ở điểm đánh dấu 3 đang làm gì so với điểm đánh dấu 7". Mô hình không được huấn luyện chuyên biệt về các dấu này; kỹ năng này tự hình thành từ sự kết hợp giữa khả năng định vị không gian (spatial grounding) và tham chiếu đa ảnh.
3. **Agent thao tác giao diện iPhone (iPhone-screenshot agent):** Người dùng cung cấp ảnh chụp màn hình điện thoại iPhone và yêu cầu lập kế hoạch cho cú chạm tiếp theo. Mô hình được huấn luyện trên ảnh chụp UI, video thao tác của người dùng và các cặp ảnh trước/sau. Năng lực này tự tổng quát hóa thành một tác nhân tự hành (agent).

Đây không phải là các tác vụ được huấn luyện trực tiếp; chúng tự đột sinh nhờ cấu trúc kết hợp đa tầng của giáo trình.

### Cơ chế gộp token thị giác

Ngân sách token đòi hỏi phải thực hiện phép gộp (pooling). OneVision sử dụng phép nội suy song tuyến tính (bilinear interpolation) trên lưới patch 2D: lưới 24x24 = 576 patch giảm xuống 12x12 = 144 (hệ số giảm 2x) hoặc 8x8 = 64 (hệ số giảm 3x). Quá trình gộp được thực hiện trực tiếp trong không gian lưới patch 2D chứ không gộp trong không gian token 1D, giúp bảo toàn tính cục bộ không gian (locality).

Việc chọn hệ số gộp cho từng kịch bản là một siêu tham số (hyperparameter). Gộp ít hơn = nhiều token hơn = biểu diễn phong phú hơn. Gộp nhiều hơn = ít token hơn = chứa được nhiều khung hình / hình ảnh hơn trong ngữ cảnh.

### LLaVA-OneVision-1.5

Bản nâng cấp năm 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) là phiên bản "hoàn toàn mở" từ dữ liệu huấn luyện, trọng số mô hình đến mã nguồn. Phiên bản này thu hẹp khoảng cách với các mô hình đóng trên nhiều benchmark và phổ cập hóa công thức này cho cộng đồng. Giữ nguyên triết lý giáo trình, bổ sung thêm dữ liệu và sử dụng LLM nền tảng mạnh hơn mà không cần thay đổi kiến trúc cốt lõi.

### So sánh đối chiếu với Qwen2.5-VL

Qwen2.5-VL (Bài học 12.09) lựa chọn một hướng đi khác. Nó áp dụng cơ chế mã hóa vị trí đa chiều M-RoPE và tốc độ khung hình động (dynamic FPS) thay vì dùng phép gộp cố định. Ngân sách token của nó co giãn theo đầu vào — một video dài 1 phút sẽ tiêu tốn nhiều token hơn video 5 giây. LLaVA-OneVision chọn cách cố định ngân sách và co giãn tỷ lệ gộp. Cả hai cách tiếp cận đều hoạt động hiệu quả; chúng đánh đổi giữa khả năng cấu hình linh hoạt và tính dự đoán trước được về tài nguyên.

```figure
l5-onevision-budget
```

## Ứng dụng thực tế

Tệp `code/main.py` là một công cụ lập kế hoạch giáo trình và ngân sách cho VLM theo phong cách OneVision. Khi cung cấp một ngân sách token cho mỗi mẫu và cơ cấu kịch bản mục tiêu (chẳng hạn 40% ảnh đơn, 30% đa ảnh, 30% video), công cụ sẽ:

- Phân bổ độ phân giải, hệ số gộp và số lượng khung hình cho từng kịch bản.
- Kiểm tra để đảm bảo mọi kịch bản đều nằm trong giới hạn ngân sách chung.
- Báo cáo số lượng token dự kiến, số phép tính FLOP của LLM và chỉ ra kịch bản nào đang bị thiếu token (under-tokenized).
- Xuất ra lịch trình huấn luyện chi tiết theo từng giai đoạn.

Hãy sử dụng công cụ này để lập kế hoạch tinh chỉnh OneVision hoặc kiểm tra chi phí tài nguyên trên từng truy vấn khi triển khai mô hình vào hệ thống thực tế.

## Đóng gói triển khai

Bài học này tạo ra tệp artifact `outputs/skill-onevision-budget-planner.md`. Dựa trên phân phối tác vụ mục tiêu và ngân sách token cho mỗi mẫu, kỹ năng này sẽ đề xuất hệ số AnyRes, tỷ lệ gộp khung hình, số lượng khung hình video và trọng số các giai đoạn trong giáo trình huấn luyện. Hãy áp dụng công cụ này bất cứ khi nào bạn huấn luyện hoặc tinh chỉnh một VLM thống nhất đa kịch bản.

## Bài tập

1. Sản phẩm của bạn phục vụ 80% yêu cầu ảnh đơn, 10% đa ảnh (2-4 ảnh), 10% video (8-16 khung hình). Hãy thiết kế ngân sách token. Bạn sẽ tái phân bổ phần ngân sách dôi dư (do không cần xử lý đa ảnh nặng) vào đâu để tối ưu chất lượng?

2. Đọc Phần 4.3 trong bài báo LLaVA-OneVision (về các năng lực đột sinh). Hãy đề xuất một năng lực đột sinh thứ tư mà giáo trình này có thể khai mở nhưng chưa được báo cáo trong bài báo.

3. Đảo ngược thứ tự giáo trình huấn luyện — huấn luyện đa ảnh trước, sau đó đến ảnh đơn, rồi đến video. Dự đoán benchmark nào sẽ bị suy giảm hiệu năng và giải thích nguyên nhân.

4. Bài báo công bố kết quả benchmark video khi chỉ huấn luyện với 8 khung hình trên mỗi mẫu. Khả năng này có tổng quát hóa được cho video dài 30 giây khi suy luận không? Yếu tố nào sẽ bị phá vỡ đầu tiên — ngân sách token hay khả năng suy luận theo thời gian?

5. Phép gộp song tuyến tính trên lưới patch 24x24 xuống 12x12 giúp giảm 2 lần trên mỗi chiều không gian, tương ứng giảm 4 lần số lượng token. Hãy tự cài đặt phép gộp này bằng Python thuần (stdlib) và kiểm chứng xem giá trị trung bình trên mỗi khối 2x2 có khớp với đầu ra của phép nội suy song tuyến tính hay không.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Kịch bản OneVision (OneVision scenario) | "Ảnh đơn, đa ảnh hoặc video" | Một trong ba định dạng hình học đầu vào mà VLM thống nhất xử lý; ngân sách luôn được cân bằng giữa các định dạng này |
| Ngân sách token (Token budget) | "Số token trên mỗi mẫu" | Tổng số token thị giác mà LLM tiếp nhận trên mỗi mẫu huấn luyện/suy luận, đạt tối đa khoảng 6.000 đến 9.000 token trong LLaVA-OneVision |
| Giáo trình huấn luyện (Curriculum) | "Thứ tự huấn luyện" | Trình tự các giai đoạn huấn luyện (ảnh đơn → đa ảnh → video) được thiết kế nhằm kích hoạt sự chuyển giao kỹ năng đột sinh |
| Gộp song tuyến tính (Bilinear pooling) | "Thu nhỏ token" | Áp dụng phép nội suy song tuyến tính lên lưới patch 2D để giảm số lượng token mà vẫn bảo toàn tính cục bộ không gian |
| Kỹ năng đột sinh (Emergent skill) | "Không học trực tiếp vẫn làm được" | Năng lực xuất hiện ở giai đoạn suy luận mà không có dữ liệu huấn luyện tương ứng trực tiếp, nhờ vào sự kết hợp cấu trúc trong giáo trình |
| AnyRes-k | "Cấu hình k mảnh ghép" | Gồm k mảnh cắt nhỏ có độ phân giải cố định cộng thêm một góc nhìn thu nhỏ toàn cảnh, thường chọn $k \in \lbrace 4, 9 \rbrace$ |
| Chuyển giao nhiệm vụ (Task transfer) | "Tổng quát hóa xuyên kịch bản" | Các kỹ năng học được từ ảnh đơn áp dụng hiệu quả cho video (và ngược lại) thông qua mạng xương sống (backbone) chia sẻ chung |

## Tài liệu tham khảo thêm

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
