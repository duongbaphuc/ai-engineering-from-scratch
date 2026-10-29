# ColPali và RAG tài liệu bản địa thị giác (Vision-Native Document RAG)

> Quy trình RAG truyền thống phân tích các tệp PDF thành văn bản thô, chia thành các đoạn nhỏ (chunks), tạo vector nhúng (embeddings) và lưu vào cơ sở dữ liệu vector. Mỗi bước trong chuỗi này đều làm thất thoát thông tin: OCR bỏ sót dữ liệu trên biểu đồ, việc chia đoạn làm vỡ các hàng trong bảng biểu, và các vector nhúng văn bản phớt lờ hoàn toàn hình vẽ minh họa. ColPali (Faysse và cộng sự, tháng 7 năm 2024) đã đặt ra một câu hỏi trực diện: tại sao chúng ta lại phải cố bóc tách văn bản? Hãy nhúng trực tiếp toàn bộ ảnh trang tài liệu thông qua PaliGemma, áp dụng cơ chế tương tác muộn (late interaction) kiểu ColBERT để truy xuất, nhờ đó bảo toàn nguyên vẹn toàn bộ bố cục, biểu đồ, phông chữ và định dạng cấu trúc mà tài liệu mang theo. Các điểm chuẩn công bố cho thấy: độ chính xác toàn trình cao hơn từ 20-40% so với phương pháp RAG văn bản truyền thống trên các tài liệu giàu yếu tố trực quan. ColQwen2, ColSmol và VisRAG đã tiếp tục mở rộng mô hình này. Bài học này sẽ phân tích luận điểm kiến trúc của RAG bản địa thị giác và xây dựng một module lập chỉ mục thu nhỏ mô phỏng cơ chế ColPali.

**Type:** Build
**Languages:** Python (stdlib, bộ lập chỉ mục đa vector + thuật toán chấm điểm MaxSim)
**Prerequisites:** Phase 11 (Kỹ thuật LLM — Nền tảng RAG), Phase 12 · 05 (LLaVA)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích sự khác biệt căn bản giữa phương pháp truy xuất bộ mã hóa kép (bi-encoder retrieval: một vector cho mỗi tài liệu) và truy xuất tương tác muộn (late-interaction retrieval: nhiều vector cho mỗi tài liệu).
- Mô tả phép toán MaxSim của ColBERT và cách ColPali tổng quát hóa phép toán này từ các token văn bản sang các mảnh ảnh (image patches).
- Xây dựng một module lập chỉ mục thu nhỏ kiểu ColPali: trang tài liệu $\to$ vector nhúng các mảnh ảnh $\to$ tính điểm MaxSim trên các vector từ khóa truy vấn $\to$ trả về top-$k$ trang phù hợp nhất.
- So sánh hiệu năng giữa cặp đôi ColPali + mô hình sinh Qwen2.5-VL với hệ thống RAG văn bản truyền thống + GPT-4 trên ca sử dụng hóa đơn tài chính và báo cáo thường niên.

## Vấn đề đặt ra

RAG văn bản truyền thống áp dụng trên tài liệu PDF vứt bỏ phần lớn thông tin giá trị của tài liệu. Sự tăng trưởng doanh thu quý 3 của một báo cáo tài chính thường được trực quan hóa trong biểu đồ cột; các phát hiện chẩn đoán trong bệnh án y khoa nằm ở hình ảnh chụp có khoanh vùng chú thích; và khối chữ ký trong hợp đồng pháp lý là một thực tế bố cục không gian chứ không đơn thuần là một chuỗi ký tự.

Đường ống RAG văn bản kinh điển:

1. Chuyển đổi PDF $\to$ văn bản thô qua OCR hoặc công cụ pdftotext.
2. Cắt văn bản $\to$ các đoạn chunk từ 300-500 token.
3. Chuyển đoạn chunk $\to$ vector nhúng thông qua bộ mã hóa kép (bi-encoder, sinh 1 vector duy nhất).
4. Câu hỏi của người dùng $\to$ vector nhúng $\to$ tính độ tương đồng cosin $\to$ lấy top-$k$ đoạn chunk liên quan.
5. Ghép các đoạn chunk + câu hỏi $\to$ đưa vào LLM để sinh câu trả lời.

Quy trình này gồm năm bước đều làm suy giảm thông tin nghiêm trọng. Đồ thị không được bóc tách. Bảng biểu bị cắt xé rời rạc giữa các đoạn chunk khác nhau. Bố cục nhiều cột bị làm phẳng thành dòng đơn. Các chú thích trong hình vẽ biến mất hoàn toàn.

Giải pháp đột phá của ColPali: bỏ qua tầng OCR, nhúng trực tiếp hình ảnh trang tài liệu. Sử dụng cơ chế tương tác muộn kiểu ColBERT để truy xuất, cho phép mô hình đối chiếu chi tiết tới từng mảnh ảnh cụ thể ngay tại thời điểm thực hiện truy vấn.

## Khái niệm cốt lõi

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) là một phương pháp truy xuất văn bản tiên phong. Thay vì nén toàn bộ tài liệu thành một vector duy nhất, nó tạo ra một vector độc lập cho mỗi token. Tại thời điểm truy vấn:

- Các token truy vấn được ánh xạ thành các vector nhúng riêng biệt ($N_q$ vectors).
- Các token tài liệu được ánh xạ thành các vector nhúng ($N_d$ vectors, thường được tính toán trước và lưu trong bộ nhớ đệm).
- Điểm tương đồng = tổng theo các token truy vấn của giá trị cực đại độ tương đồng cosin trên toàn bộ các token tài liệu: $\sum_i \max_j \cos(q_i, d_j)$.

Đây chính là phép toán **MaxSim**. Mỗi token truy vấn sẽ "chọn" token tài liệu phù hợp nhất với nó. Điểm số cuối cùng là tổng của các giá trị cực đại này.

- **Ưu điểm:** Khả năng thu hồi (recall) vượt trội, nắm bắt ngữ nghĩa chi tiết ở cấp độ từng từ khóa.
- **Nhược điểm:** Cần lưu $N_d$ vector cho mỗi tài liệu, dẫn đến chi phí dung lượng bộ nhớ lưu trữ tăng cao.

### ColPali

ColPali (Faysse và cộng sự, arXiv:2407.01449) mở rộng nguyên lý của ColBERT sang miền thị giác:

- Mỗi trang tài liệu được bộ mã hóa PaliGemma (kết hợp ViT + mô hình ngôn ngữ) xử lý thành một tập hợp các vector nhúng mảnh ảnh (patch embeddings): gồm $N_p$ vector cho mỗi trang.
- Câu truy vấn văn bản của người dùng được mã hóa thành các vector nhúng token: gồm $N_q$ vector.
- Điểm số = $\sum_i \max_j \cos(q_i, p_j)$, tức là phép toán MaxSim giữa các token văn bản truy vấn và các mảnh ảnh của trang tài liệu.
- Truy xuất top-$k$ trang có tổng điểm số cao nhất.

Quy trình nạp dữ liệu: nhúng toàn bộ các trang bằng PaliGemma, lưu trữ tất cả vector nhúng mảnh ảnh. Khi người dùng đặt câu hỏi: nhúng các token truy vấn, tính điểm MaxSim đối chiếu với toàn bộ các vector trang đã lưu trữ và trả về top-$k$ trang phù hợp nhất.

- **Ưu điểm:** Đạt hiệu quả vượt trội từ 20-40% so với RAG văn bản trên các tài liệu có cấu trúc thị giác phức tạp. Mỗi vector mảnh ảnh nắm bắt trọn vẹn bố cục cục bộ và nội dung trực quan tại vị trí đó.
- **Nhược điểm:** $N_p$ mảnh ảnh $\times$ 4 byte (float32) $\times$ $D$ chiều vector trên mỗi trang khiến dung lượng lưu trữ phình to nhanh chóng. Nhược điểm này được khắc phục bằng kỹ thuật lượng tử hóa tích (PQ / OPQ).

### ColQwen2 và ColSmol

- **ColQwen2** (illuin-tech, 2024-2025): Thay thế xương sống PaliGemma bằng Qwen2-VL. Bộ mã hóa nền tảng mạnh mẽ hơn giúp chất lượng truy xuất tăng rõ rệt.
- **ColSmol**: Phiên bản thu nhỏ hướng tới môi trường biên (edge) hoặc máy trạm cục bộ. Bộ truy xuất ColSmol với quy mô khoảng 1B tham số có thể chạy mượt mà trên các GPU thương mại phổ thông.

### VisRAG

VisRAG (Yu và cộng sự, arXiv:2410.10594) lựa chọn một hướng tiếp cận khác: thay vì dùng MaxSim trên tập hợp mảnh ảnh, nó gộp (pool) toàn bộ trang tài liệu thành một vector duy nhất thông qua VLM rồi áp dụng truy xuất bộ mã hóa kép thông thường. Quá trình lập chỉ mục nhanh hơn và tiết kiệm dung lượng lưu trữ hơn, nhưng độ chính xác thu hồi (recall) thấp hơn ColPali.

Sự đánh đổi: ColPali tối ưu cho chất lượng đỉnh cao, VisRAG tối ưu cho quy mô lớn tiết kiệm chi phí.

### M3DocRAG

M3DocRAG (Cho và cộng sự, arXiv:2411.04952) mở rộng khả năng truy xuất đa thể thức sang suy luận liên kết trên nhiều trang và nhiều tài liệu khác nhau. Nó truy xuất các trang liên quan từ nhiều nguồn tài liệu, sau đó hợp nhất thành một ngữ cảnh đa trang hoàn chỉnh cung cấp cho VLM.

### ViDoRe — Bộ điểm chuẩn đo lường tiêu chuẩn

Bộ tiêu chuẩn đo lường đi kèm ColPali: ViDoRe (Visual Document Retrieval Evaluation). Tập hợp các bài kiểm thử đa dạng từ báo cáo tài chính, ấn phẩm nghiên cứu khoa học, văn bản hành chính, bệnh án y khoa cho đến tài liệu hướng dẫn kỹ thuật. Chỉ số đánh giá chính: nDCG@5.

ColPali-v1 đạt xấp xỉ 80% nDCG@5 trên ViDoRe; trong khi phương pháp RAG văn bản trên cùng tập tài liệu này chỉ đạt mức 50-60%.

### Luồng xử lý RAG bản địa thị giác toàn trình

Quy trình vận hành trong môi trường thực tế:

1. **Giai đoạn nạp dữ liệu (Ingest):** PDF $\to$ trích xuất ảnh trang $\to$ mã hóa qua PaliGemma $\to$ lưu trữ toàn bộ các vector nhúng mảnh ảnh vào cơ sở dữ liệu vector.
2. **Giai đoạn truy vấn (Query):** Văn bản người dùng $\to$ vector nhúng token truy vấn $\to$ tính MaxSim đối chiếu toàn bộ các trang đã lập chỉ mục $\to$ lọc ra top-$k$ trang điểm cao nhất.
3. **Giai đoạn tạo sinh (Generate):** Đưa hình ảnh của top-$k$ trang kèm câu hỏi ban đầu $\to$ nạp vào VLM (Qwen2.5-VL hoặc Claude) $\to$ sinh câu trả lời hoàn chỉnh.

Hoàn toàn không có bất kỳ bước OCR nào trong toàn bộ chu trình. Biểu đồ, đồ thị, phông chữ và bố cục trang đều được truyền tải nguyên vẹn tới mô hình sinh câu trả lời.

### Bài toán dung lượng lưu trữ

Xét một bản báo cáo tài chính 50 trang với 729 mảnh ảnh mỗi trang và vector nhúng 128 chiều:

- **ColPali:** $50 \times 729 \times 128 \times 4\text{ bytes} \approx 18\text{ MB}$ dữ liệu thô, nén còn $\approx 2,3\text{ MB}$ sau khi áp dụng lượng tử hóa PQ 8x.
- **RAG văn bản:** 50 đoạn chunk $\times$ 768 chiều $\times$ 4 byte $\approx 150\text{ kB}$.

ColPali tiêu tốn dung lượng gấp khoảng 120 lần ở dạng thô. Trong môi trường sản xuất quy mô lớn, việc nén bằng OPQ / PQ 8x đưa mức chênh lệch về khoảng 15 lần — đây là ngưỡng hoàn toàn có thể chấp nhận được để đổi lấy bước nhảy vọt về chất lượng phản hồi.

### Khi nào RAG văn bản truyền thống vẫn là lựa chọn ưu tiên?

- Tài liệu thuần chữ không chứa yếu tố bố cục trực quan (các bài viết bách khoa toàn thư wiki, lịch sử trò chuyện chat log). RAG văn bản đơn giản hơn và tiết kiệm chi phí lưu trữ đáng kể.
- Các kho lưu trữ khổng lồ hàng trăm triệu trang tài liệu nơi chi phí đĩa cứng và RAM là bài toán sống còn.
- Các yêu cầu pháp lý bắt buộc phải trích xuất và hiển thị được chuỗi văn bản OCR tương ứng cùng với kết quả tìm kiếm.

Đối với hầu hết các trường hợp còn lại trong năm 2026 — báo cáo tài chính, bài báo khoa học, hợp đồng pháp lý, hồ sơ y tế, tài liệu đặc tả sản phẩm UI/UX — RAG bản địa thị giác là phương án chiến thắng áp đảo.

```figure
mm-maxsim
```

## Sử dụng trong thực tế

Tệp `code/main.py`:

- Bộ mã hóa mảnh ảnh thu nhỏ: ánh xạ một "trang tài liệu" (lưới vector đặc trưng nhỏ) thành một mảng các vector nhúng mảnh ảnh.
- Thuật toán chấm điểm MaxSim: tính toán điểm số kiểu ColBERT giữa tập vector token truy vấn và tập mảnh ảnh của trang.
- Lập chỉ mục cho 5 trang tài liệu mẫu, thực thi 3 câu truy vấn và trả về danh sách top-$k$ trang có điểm số cao nhất.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-vision-rag-designer.md`. Dựa trên yêu cầu của dự án RAG tài liệu, kỹ năng này tự động lựa chọn giữa ColPali / ColQwen2 / VisRAG / RAG văn bản truyền thống và tính toán dung lượng phần cứng lưu trữ cần thiết.

## Bài tập thực hành

1. Một báo cáo thường niên dài 200 trang với 729 mảnh ảnh mỗi trang, vector nhúng 128 chiều, sử dụng số thực 4 byte. Hãy tính dung lượng lưu trữ thô và dung lượng sau khi nén bằng lượng tử hóa PQ (8x).

2. Phép toán MaxSim được định nghĩa là $\sum_i \max_j \cos(q_i, p_j)$. Tổng của các giá trị cực đại này phản ánh được điều gì mà phép tính trung bình độ tương đồng đơn thuần không thể hiện được?

3. ColPali lập chỉ mục trang tài liệu dưới dạng tập hợp các mảnh ảnh trực quan. Điều gì sẽ thay đổi nếu chúng ta lập chỉ mục ở cấp độ từng từ (giống như ColBERT văn bản)? Đánh giá sự đánh đổi giữa hai phương án.

4. Thiết kế một hệ thống truy xuất toàn trình cho một kho ngữ liệu gồm 1 triệu trang tài liệu với hạn mức thời gian phản hồi là 500ms cho mỗi truy vấn. Hãy chọn giữa ColQwen2 hoặc VisRAG và bảo vệ quan điểm thiết kế của bạn.

5. Đọc bài báo M3DocRAG (arXiv:2411.04952). Phân tích cơ chế chú ý đa trang (multi-page attention) và chỉ ra điểm khác biệt giữa nó với cơ chế truy xuất trang đơn lẻ của ColPali.

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Tương tác muộn (Late interaction) | "Kiểu ColBERT" | Phương pháp truy xuất dùng các vector nhúng trên từng token/mảnh ảnh kết hợp MaxSim, không gộp thành 1 vector |
| MaxSim | "Cực đại trên mảnh ảnh" | Với mỗi token câu hỏi, lấy giá trị tương đồng lớn nhất trên các token tài liệu; sau đó cộng tổng lại |
| Bộ mã hóa kép (Bi-encoder) | "Vector đơn lẻ" | Nén toàn bộ tài liệu thành một vector duy nhất; tính toán nhanh nhưng mất độ chi tiết |
| Đa vector (Multi-vector) | "Nhiều vector mỗi trang" | Lưu trữ $N_p$ vector cho mỗi tài liệu/trang; tăng chi phí bộ nhớ nhưng tối ưu hóa độ thu hồi |
| Vector nhúng mảnh ảnh (Patch embedding) | "Đặc trưng trang ảnh" | Một vector đại diện cho một mảnh ảnh cục bộ trích xuất từ bộ mã hóa VLM, lưu trữ trên từng trang |
| ViDoRe | "Chuẩn đo RAG thị giác" | Bộ dữ liệu đo chuẩn của ColPali dành riêng cho bài toán truy xuất tài liệu trực quan |
| Lượng tử hóa PQ (Product Quantization) | "Nén PQ" | Kỹ thuật nén duy trì độ tương đồng vector trong khi giảm kích thước lưu trữ xuống khoảng 8 lần |

## Tài liệu đọc thêm

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
