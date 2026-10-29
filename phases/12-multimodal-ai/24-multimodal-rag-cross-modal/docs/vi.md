# RAG đa thể thức và truy xuất xuyên thể thức (Multimodal RAG and Cross-Modal Retrieval)

> RAG tài liệu bản địa thị giác chỉ là một phần của bức tranh tổng thể. Trong môi trường sản xuất thực tế, RAG đa thể thức (multimodal RAG) mở rộng phạm vi ra toàn diện hơn — truy xuất xuyên suốt qua văn bản, hình ảnh, âm thanh và video nhằm phục vụ các luồng công việc phức tạp như lập kế hoạch du lịch ("hãy tìm cho tôi một quán brunch thuần chay yên tĩnh có nhiều ánh sáng tự nhiên"), phân loại sơ cứu y tế ("vết thương này phù hợp với bức ảnh chụp và các triệu chứng lâm sàng nào"), thương mại điện tử ("gợi ý trang phục tương tự bức ảnh selfie này nhưng vừa với số đo của tôi"), và bảo trì kỹ thuật tại hiện trường ("chẩn đoán âm thanh động cơ bất thường này kèm bức ảnh chụp linh kiện máy"). Ba bài khảo cứu toàn diện năm 2025 — Abootorabi và cộng sự, Mei và cộng sự, Zhao và cộng sự — đã hệ thống hóa các bài toán con cốt lõi: truy xuất xuyên thể thức (cross-modal retrieval), hợp nhất kết quả truy xuất (retrieval fusion), neo ngữ cảnh tạo sinh (generation grounding) và đánh giá đa thể thức. Bài học này sẽ phân tích các bài khảo cứu trên và thiết kế một đường ống sản xuất hoàn chỉnh.

**Type:** Build
**Languages:** Python (stdlib, bộ truy xuất xuyên thể thức với cơ chế hợp nhất + bộ tạo sinh neo ngữ cảnh)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (Nền tảng RAG)
**Time:** ~180 phút

## Mục tiêu học tập

- Thiết kế cơ chế truy xuất xuyên thể thức: văn bản $\to$ hình ảnh, hình ảnh $\to$ văn bản, âm thanh $\to$ video, v.v.
- So sánh ba chiến lược hợp nhất kết quả: hợp nhất điểm số (score fusion), hợp nhất dựa trên cơ chế chú ý (attention-based fusion), và hợp nhất theo mạng lưới chuyên gia (MoE fusion).
- Giải thích kỹ thuật neo ngữ cảnh tạo sinh (generation grounding): cách thức mô hình "trích dẫn nguồn minh chứng" khi nguồn thông tin bao gồm hỗn hợp nhiều thể thức khác nhau.
- Nắm vững ba bài khảo cứu RAG đa thể thức tiêu biểu năm 2025 và bảng phân loại các bài toán thành phần.

## Vấn đề đặt ra

RAG đơn thể thức (single-modality RAG) là một khuôn mẫu đã được định hình chuẩn xác: nhúng câu truy vấn, nhúng các đoạn chunk, truy xuất và đưa vào LLM. Ngược lại, RAG đa thể thức đòi hỏi giải quyết đồng thời nhiều thách thức:

1. **Nhiều đầu truy xuất độc lập:** Mỗi thể thức đòi hỏi các vector nhúng nằm trong một không gian tương thích.
2. **Hợp nhất kết quả truy xuất giữa các thể thức khác nhau.**
3. **Neo ngữ cảnh tạo sinh:** Câu trả lời của mô hình phải có khả năng trích dẫn chính xác nguồn minh chứng thuộc nhiều thể thức.
4. **Bộ chỉ số đo lường đánh giá:** Cần bao quát được tín hiệu tương quan xuyên thể thức.

Cả ba bài khảo cứu lớn năm 2025 đều hội tụ về cùng một hệ thống phân loại này.

## Khái niệm cốt lõi

### Truy xuất xuyên thể thức (Cross-Modal Retrieval)

Nhiệm vụ truy xuất các tài liệu thuộc thể thức B khi tiếp nhận câu truy vấn thuộc thể thức A. Có ba kiến trúc triển khai chính:

1. **Không gian nhúng chia sẻ (Shared embedding space):** Các mô hình như CLIP và CLAP ánh xạ văn bản + hình ảnh hoặc văn bản + âm thanh vào cùng một không gian vector đồng nhất. Khi đó, độ tương đồng cosin có thể tính toán trực tiếp giữa các thể thức. Hạn chế: bị giới hạn trong các cặp dữ liệu đã được huấn luyện sẵn của CLIP/CLAP.

2. **Bộ mã hóa riêng biệt kèm tầng dịch chuyển (Per-modality encoder + translation):** Sử dụng bộ mã hóa văn bản riêng, bộ mã hóa hình ảnh riêng, kết hợp một module dịch chuyển nhỏ để ánh xạ vector giữa hai không gian. Tiêu biểu là mô hình Sen2Sen của Gupta và cộng sự cùng các thiết kế năm 2024. Linh hoạt hơn nhưng làm tăng độ phức tạp của hệ thống.

3. **Sử dụng VLM làm bộ mã hóa:** Tận dụng trực tiếp các trạng thái ẩn (hidden states) của một mô hình VLM làm biểu diễn vector phục vụ truy xuất. Áp dụng được cho bất kỳ thể thức nào mà VLM hỗ trợ. Chất lượng biểu diễn vượt trội nhưng chi phí tài nguyên tính toán cao hơn.

Lựa chọn khuyến nghị: Dùng CLIP / SigLIP 2 cho văn bản + hình ảnh; CLAP cho văn bản + âm thanh; và trạng thái ẩn của VLM cho các tác vụ xuyên thể thức đòi hỏi độ chính xác biên giới cao nhất.

### Các chiến lược hợp nhất kết quả (Fusion Strategies)

Giả sử hệ thống truy xuất được 10 kết quả hỗn hợp: 5 hình ảnh, 3 đoạn văn bản và 2 đoạn âm thanh. Làm thế nào để xếp hạng và hợp nhất chúng?

- **Hợp nhất điểm số (Score fusion - phương án tiết kiệm nhất):** Mỗi thể thức có một bộ truy xuất riêng và trả về điểm số tương ứng. Chuẩn hóa điểm số trong phạm vi từng thể thức rồi tính tổng gia quyền. Đơn giản, tính toán nhẹ và hiệu quả trong hầu hết trường hợp.
- **Hợp nhất dựa trên cơ chế chú ý (Attention-based fusion):** Ghép nối (concatenate) tất cả các thực thể được truy xuất, sử dụng một mạng chú ý nhỏ để tự động học trọng số gán cho từng phần tử. Đòi hỏi phải huấn luyện thêm dữ liệu căn chỉnh.
- **Hợp nhất chuyên gia (MoE fusion):** Sử dụng một mạng định tuyến (gating network) để điều hướng tới các chuyên gia theo từng thể thức cụ thể. Các loại câu hỏi khác nhau sẽ được định tuyến khác nhau — ví dụ một câu hỏi tập trung vào màu sắc hay hình dáng sẽ gán trọng số ưu tiên lớn hơn cho hình ảnh.

Mặc định trong môi trường thực tế: áp dụng hợp nhất điểm số có thiên lệch nhẹ về thể thức áp đảo trong câu truy vấn. Nâng cấp lên MoE nếu kết quả thử nghiệm A/B trên miền nghiệp vụ thực tế cho thấy sự cải thiện vượt trội.

### Neo ngữ cảnh tạo sinh (Generation Grounding)

LLM cần chỉ rõ thực thể được truy xuất nào đã làm căn cứ cho từng luận điểm trong câu trả lời. Với hệ thống đa thể thức:

- Nguồn văn bản: trích dẫn chuẩn `[1]`.
- Nguồn hình ảnh: trích dẫn dạng `[ảnh 3]` kèm một mô tả ngắn gọn.
- Nguồn âm thanh: trích dẫn dạng `[âm thanh 2 tại 0:34]`.

Kỹ thuật huấn luyện: tinh chỉnh mô hình sinh trên dữ liệu có nhận thức neo ngữ cảnh, trong đó mỗi luận điểm trong văn bản mục tiêu đều được gán nhãn chỉ số nguồn tương ứng. Khi suy luận thực tế, mô hình sẽ tự nhiên phát ra các trích dẫn chuẩn xác.

### Ba bài khảo cứu bước ngoặt năm 2025

- **Abootorabi và cộng sự** (arXiv:2502.08826, "Ask in Any Modality"): Hệ thống hóa toàn diện cấu trúc RAG đa thể thức, phân tích chi tiết các tầng truy xuất, hợp nhất và tạo sinh. Bức tranh bao quát rộng nhất.
- **Mei và cộng sự** (arXiv:2504.08748, "A Survey of Multimodal RAG"): Đi sâu vào các bài kiểm thử chuẩn của các tác vụ con và phân tích chi tiết các dạng thức thất bại (failure modes). Cực kỳ hữu ích cho việc thiết kế kiểm thử đánh giá.
- **Zhao và cộng sự** (arXiv:2503.18016): Tập trung chuyên biệt vào nhánh thị giác, phân tích sâu họ mô hình ColPali và ứng dụng thực tiễn.

Nghiên cứu ba tài liệu này cung cấp bức tranh toàn cảnh về hiện trạng công nghệ tính đến năm 2025. Rất nhiều bài toán con trong số này vẫn đang là đề tài nghiên cứu mở.

### MuRAG — Công trình nền tảng

MuRAG (Chen và cộng sự, 2022) là hệ thống RAG đa thể thức đầu tiên trên thế giới. Nó thực hiện truy xuất cả ảnh và văn bản từ một cơ sở tri thức đa phương thức, sau đó tổng hợp câu trả lời. MuRAG đã chứng minh tính khả thi của mô hình này từ trước khi làn sóng VLM bùng nổ. Các hệ thống hiện đại ngày nay (REACT, VisRAG, M3DocRAG) đều kế thừa và phát triển từ nền móng của MuRAG.

### Kịch bản sản xuất thực tế: Trợ lý lập kế hoạch du lịch

Câu hỏi người dùng: "Hãy tìm cho tôi một quán brunch thuần chay yên tĩnh có nhiều ánh sáng tự nhiên."

Quy trình xử lý:

1. **Phân rã câu truy vấn:**
   - "yên tĩnh" $\to$ trích xuất từ khóa đánh giá / đặc trưng âm thanh.
   - "brunch thuần chay" $\to$ tìm kiếm thực đơn món ăn.
   - "nhiều ánh sáng tự nhiên" $\to$ đặc trưng thị giác của không gian.
2. **Truy xuất song song theo từng thể thức:**
   - Truy xuất văn bản trên các bài đánh giá (reviews): "bữa sáng thuần chay, không gian yên tĩnh".
   - Truy xuất hình ảnh trên thư viện ảnh quán: "ánh sáng tự nhiên, thoáng đãng".
   - Truy xuất âm thanh trên các đoạn ghi âm môi trường: "độ ồn decibel thấp, không bật nhạc sàn".
3. **Hợp nhất điểm số:** Tính điểm tổng hợp đa chiều cho từng nhà hàng ứng viên.
4. **Tạo sinh:** Đưa top-$k$ nhà hàng kèm toàn bộ bằng chứng đa thể thức vào VLM $\to$ sinh câu trả lời kèm trích dẫn chi tiết.

Quy trình này vượt xa năng lực của RAG văn bản đơn thuần. Mỗi thể thức đóng góp những thông tin độc nhất mà văn bản chữ không thể phản ánh trọn vẹn.

### RAG đa thể thức dạng tác tử (Agentic Multimodal RAG)

Cơ chế đa bước nhảy (multi-hop): nếu lượt truy xuất đầu tiên chưa mang lại kết quả có độ tin cậy cao, LLM sẽ tự động tái cấu trúc câu hỏi và thực hiện truy xuất bổ sung. Các mẫu hình RAG tác tử từ Phase 14 được áp dụng trực tiếp tại đây:

- Truy xuất sơ bộ top-10 $\to$ LLM phân tích thấy: "quá ồn, cần lọc thêm điều kiện dưới 40 dB" $\to$ kích hoạt truy xuất lại.
- Truy xuất hình ảnh $\to$ LLM phát hiện một bức ảnh chụp có bảng thực đơn $\to$ kích hoạt truy xuất sâu vào nội dung chữ của thực đơn đó $\to$ sinh câu trả lời.

Mặc dù làm tăng độ trễ và độ phức tạp, phương pháp này giải quyết được các yêu cầu phức tạp mà cơ chế truy xuất một lượt (single-shot) hoàn toàn bất lực.

### Đánh giá chất lượng (Evaluation)

Đánh giá chất lượng hệ thống xuyên thể thức vẫn là lĩnh vực đang trên đà hoàn thiện. Các chỉ số đại diện phổ biến:

- Tỷ lệ thu hồi Recall@k theo từng thể thức riêng biệt.
- Độ chính xác của danh sách top-$k$ sau khi hợp nhất.
- Đánh giá sự hài lòng toàn trình bởi chuyên gia con người (human evaluation).
- Các chỉ số gắn liền với tác vụ kinh doanh (tỷ lệ hoàn tất đặt phòng, tỷ lệ chuyển đổi mua sắm).

Hiện tại chưa có một bộ điểm chuẩn duy nhất bao quát trọn vẹn toàn bộ các thể thức. Hầu hết các công trình nghiên cứu đều tự thiết lập bộ kiểm thử trên các bài toán miền chuyên biệt.

```figure
contrastive-matrix
```

## Sử dụng trong thực tế

Tệp `code/main.py`:

- Ba bộ truy xuất giả lập (văn bản, hình ảnh, âm thanh) vận hành trên một kho dữ liệu nhà hàng chung.
- Cơ chế hợp nhất điểm số phối hợp trọng số đa thể thức tùy biến.
- Khung sườn mô hình tạo sinh sinh câu trả lời cuối cùng có kèm trích dẫn nguồn.
- Một vòng lặp tác tử đơn giản tự động định hình lại câu truy vấn khi độ tin cậy của kết quả ban đầu ở mức thấp.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-multimodal-rag-designer.md`. Dựa trên đặc tả sản phẩm có luồng truy vấn đa thể thức, kỹ năng này tự động thiết kế kiến trúc các bộ truy xuất, giải thuật hợp nhất, bộ tạo sinh và hệ thống chỉ số đo lường đánh giá.

## Bài tập thực hành

1. Hãy đề xuất một hệ thống RAG đa thể thức phục vụ phân loại bệnh án y tế: đầu vào gồm ảnh chụp tổn thương da + mô tả triệu chứng bằng văn bản. Hệ thống cần truy xuất những thể thức nào từ những cơ sở tri thức chuyên môn nào?

2. Phương pháp hợp nhất điểm số sử dụng phép cộng gia quyền đơn giản. Nó gặp phải hạn chế nghiêm trọng nào trong trường hợp các thể thức mâu thuẫn nhau mà giải thuật hợp nhất MoE có thể khắc phục được?

3. Đọc Mục 3 trong bài khảo cứu của Abootorabi và cộng sự. Hãy gọi tên ba bài toán con chuẩn tắc và phân tích cách chúng ánh xạ vào bài toán sản phẩm mà bạn đang xây dựng.

4. Thiết kế một đặc tả đo lường đánh giá cho hệ thống RAG đa thể thức lập kế hoạch du lịch. Những chỉ số cụ thể nào sẽ đo lường độ thu hồi ảnh, độ thu hồi âm thanh và tính chính xác tổng thể của câu trả lời?

5. Cơ chế RAG tác tử đa bước nhảy phải trả giá bằng sự gia tăng độ trễ trên mỗi chu trình lặp. Ở ngưỡng độ khó nào của câu hỏi thì sự cải thiện về độ chính xác mới thực sự bù đắp được chi phí độ trễ phát sinh?

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Truy xuất xuyên thể thức (Cross-modal retrieval) | "Truy vấn một đằng, tìm một nẻo" | Dùng câu hỏi văn bản tìm ảnh; dùng ảnh tìm văn bản; đòi hỏi không gian nhúng chung hoặc bộ dịch |
| Hợp nhất điểm số (Score fusion) | "Gộp điểm" | Tính tổng gia quyền của các điểm số truy xuất theo từng thể thức; phương án hợp nhất đơn giản nhất |
| Hợp nhất MoE (MoE fusion) | "Chuyên gia định tuyến thể thức" | Mạng định tuyến tự động chọn trọng số thể thức đáng tin cậy nhất dựa trên bản chất câu hỏi |
| Neo ngữ cảnh tạo sinh (Grounded generation) | "Trích dẫn nguồn minh chứng" | Mọi luận điểm trong câu trả lời sinh ra đều được đánh nhãn chỉ số nguồn tài liệu đối chiếu |
| MuRAG | "RAG đa thể thức đầu tiên" | Công trình nghiên cứu năm 2022 đặt nền móng cho kiến trúc RAG đa thể thức |
| Tác tử đa bước nhảy (Agentic multi-hop) | "Định hình lại và thử lại" | LLM chủ động truy vấn lại các bộ tìm kiếm khi độ tin cậy của kết quả bước trước không đạt yêu cầu |

## Tài liệu đọc thêm

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
