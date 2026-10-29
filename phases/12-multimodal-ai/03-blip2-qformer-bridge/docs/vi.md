# Từ CLIP Đến BLIP-2 — Cầu Nối Q-Former Giữa Các Phương Thức

> CLIP căn chỉnh hình ảnh và văn bản nhưng không thể sinh chú thích, trả lời câu hỏi hay duy trì một cuộc đối thoại. BLIP-2 (Salesforce, 2023) đã giải quyết vấn đề đó bằng một cầu nối nhỏ có thể huấn luyện được: 32 vector truy vấn học được (learnable query vectors) chú ý chéo (cross-attention) lên các đặc trưng của một ViT đóng băng, sau đó cắm trực tiếp vào luồng đầu vào của một LLM đóng băng. Cầu nối chỉ 188 triệu tham số đã kết nối thành công một ViT-g/14 với các LLM đóng băng có quy mô lên tới 11 tỷ tham số như Flan-T5-XXL. Mọi mô hình thị giác - ngôn ngữ (VLM - Vision-Language Model) dựa trên bộ điều hợp (adapter-based VLM) cho đến năm 2026 — MiniGPT-4, InstructBLIP, các biến thể của LLaVA — đều là hậu duệ của kiến trúc này. Bài học này phân tích kiến trúc của Q-Former, làm rõ quy trình huấn luyện hai giai đoạn và xây dựng một phiên bản minh họa đưa các token thị giác vào một bộ giải mã văn bản đóng băng.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (stdlib, minh họa cơ chế chú ý chéo + truy vấn học được)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 02 (CLIP), Giai đoạn 7 (Transformers)  
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích tại sao một nút thắt cổ chai có thể huấn luyện được (trainable bottleneck) nằm giữa bộ mã hóa thị giác đóng băng và LLM đóng băng lại vượt trội hơn tinh chỉnh từ đầu đến cuối (end-to-end finetuning) về mặt chi phí và độ ổn định.
- Triển khai một khối chú ý chéo (cross-attention block) nơi một tập hợp cố định các truy vấn học được sẽ chú ý lên các đặc trưng hình ảnh bên ngoài.
- Phân tích chi tiết hai giai đoạn tiền huấn luyện của BLIP-2: học biểu diễn (representation learning: ITC + ITM + ITG), sau đó là học sinh văn bản (generative learning: hàm mất mát LM với bộ giải mã đóng băng).
- So sánh Q-Former với bộ chiếu MLP (MLP projector) đơn giản hơn được sử dụng trong LLaVA và lập luận về bối cảnh mà mỗi lựa chọn chiếm ưu thế.

## Vấn đề

Giả sử bạn có một ViT đóng băng tạo ra 256 token mảng ảnh với số chiều 1408 cho mỗi hình ảnh. Bạn có một LLM 7B đóng băng mong đợi các vector nhúng token có số chiều 4096. Cầu nối hiển nhiên nhất — một lớp tuyến tính chuyển từ 1408 sang 4096 — hoạt động được, nhưng việc đưa toàn bộ 256 token mảng ảnh vào ngữ cảnh của LLM sẽ tiêu tốn 256 token phụ trội cho mỗi ảnh. Với một batch gồm 32 hình ảnh, phương thức thị giác đã ngốn tới 8.192 token ngữ cảnh.

Câu hỏi của BLIP-2: liệu bạn có thể nén biểu diễn hình ảnh từ 256 token xuống ít hơn rất nhiều (chẳng hạn 32 token) mà vẫn giữ đủ thông tin để LLM có thể sinh chú thích, trả lời câu hỏi và suy luận về bức ảnh hay không? Và liệu bạn có thể huấn luyện cầu nối này mà không cần chạm vào các mạng xương sống đã đóng băng, giữ cho chi phí huấn luyện chỉ gói gọn trong số tham số của chính cầu nối?

Câu trả lời: một bộ biến đổi truy vấn (Q-Former - Querying Transformer). Gồm 32 vector "truy vấn" học được thực hiện chú ý chéo lên các token mảng ảnh của ViT, tạo ra một bản tóm tắt thị giác 32 token cho LLM tiếp nhận. Toàn bộ cấu trúc chỉ có 188 triệu tham số. Nó được huấn luyện với các mục tiêu tương phản, so khớp và sinh ngôn ngữ trước khi ghép nối vào LLM.

## Khái niệm

### Các truy vấn học được (Learnable queries)

Kỹ thuật cốt lõi của Q-Former: thay vì để các token văn bản của LLM trực tiếp chú ý lên các mảng ảnh, ta đưa vào một tập hợp mới gồm 32 vector truy vấn học được `Q` và để *chúng* chú ý lên các mảng ảnh. Các truy vấn này là các tham số của mô hình — chúng được tối ưu trong quá trình huấn luyện và cùng 32 truy vấn này được dùng cho mọi hình ảnh.

Sau bước chú ý chéo, mỗi truy vấn nắm giữ một phần tóm tắt đã nén của hình ảnh — "mô tả đối tượng chính", "mô tả hậu cảnh", "đếm số đối tượng", v.v. Các truy vấn không nhất thiết phải chuyên biệt hóa theo nhãn ngữ nghĩa tường minh của con người; chúng tự học bất kỳ cách mã hóa nào giúp hàm mất mát hạ nguồn giảm xuống.

### Kiến trúc (Architecture)

Q-Former là một transformer cỡ nhỏ (12 lớp, ~100M tham số) với hai nhánh xử lý:

1. Nhánh truy vấn (Query path): 32 vector truy vấn đi qua phép tự chú ý (self-attention giữa các truy vấn với nhau), tiếp đến là chú ý chéo (cross-attention) trên các token mảng ảnh của ViT đóng băng, rồi qua mạng truyền thẳng FFN.
2. Nhánh văn bản (Text path): một bộ mã hóa văn bản kiểu BERT dùng chung các trọng số tự chú ý và FFN với nhánh truy vấn. Chú ý chéo bị vô hiệu hóa đối với nhánh văn bản.

Trong quá trình huấn luyện, cả hai nhánh đều hoạt động. Các truy vấn và văn bản tương tác với nhau thông qua cơ chế tự chú ý dùng chung, nghĩa là các truy vấn có thể điều kiện hóa theo ngữ cảnh văn bản cho những tác vụ cần thiết (ITM, ITG). Tại thời điểm suy luận để bàn giao cho VLM, chỉ có các truy vấn được xử lý qua mạng, xuất ra 32 token thị giác.

### Huấn luyện hai giai đoạn (Two-stage training)

BLIP-2 thực hiện tiền huấn luyện theo hai giai đoạn:

Giai đoạn 1: Học biểu diễn (Representation learning - chưa có LLM). Gồm ba hàm mất mát:
- ITC (Tương phản hình ảnh - văn bản / Image-Text Contrastive): tương phản kiểu CLIP giữa các token truy vấn đã gộp và token CLS của văn bản.
- ITM (So khớp hình ảnh - văn bản / Image-Text Matching): bộ phân loại nhị phân — cặp hình ảnh và văn bản này có khớp nhau không? Áp dụng kỹ thuật khai thác mẫu âm khó (hard-negative mining).
- ITG (Sinh văn bản gắn với hình ảnh / Image-Grounded Text Generation): đầu mô hình ngôn ngữ tự hồi quy (causal LM head) trên văn bản, điều kiện hóa theo các truy vấn. Buộc các truy vấn phải mã hóa thông tin đủ để giải mã thành văn bản.

Ở giai đoạn này, chỉ có Q-Former được huấn luyện. ViT hoàn toàn đóng băng. Chưa có sự tham gia của LLM.

Giai đoạn 2: Học sinh văn bản (Generative learning). Ghép nối một LLM đóng băng (như OPT-2.7B hoặc Flan-T5-XL, v.v.). Chiếu 32 đầu ra của truy vấn sang số chiều nhúng của LLM thông qua một lớp tuyến tính nhỏ. Đặt chúng vào đầu câu nhắc văn bản (prompt). Huấn luyện duy nhất lớp chiếu tuyến tính và Q-Former dựa trên hàm mất mát mô hình ngôn ngữ (LM loss) trên chuỗi nối tiếp gồm câu nhắc + hình ảnh + chú thích.

Sau giai đoạn 2, Q-Former cùng lớp chiếu trở thành bộ điều hợp thị giác hoàn chỉnh. Khi suy luận: hình ảnh → ViT → Q-Former → lớp chiếu tuyến tính → ghép vào đầu văn bản → LLM đóng băng sinh kết quả.

### Tính kinh tế về tham số (Parameter economics)

BLIP-2 kết hợp ViT-g/14 (1,1 tỷ tham số, đóng băng) + OPT-6.7B (6,7 tỷ tham số, đóng băng) + Q-Former (188 triệu tham số, huấn luyện) = tổng cộng 8 tỷ tham số, trong đó chỉ 188 triệu tham số cần cập nhật gradient. Bản thân Q-Former chỉ chiếm khoảng ~2,4% tổng tham số hệ thống. Chi phí huấn luyện phản ánh rõ điều này: chỉ mất vài ngày trên một cụm nhỏ A100 thay vì hàng tuần huấn luyện toàn diện từ đầu đến cuối.

Về chất lượng: BLIP-2 đạt kết quả tương đương hoặc vượt qua Flamingo-80B trên bài kiểm tra trả lời câu hỏi bằng hình ảnh zero-shot (zero-shot VQA) dù kích thước nhỏ hơn tới 50 lần. Cầu nối này đã chứng minh tính hiệu quả vượt trội.

### InstructBLIP và Q-Former nhận biết chỉ dẫn (InstructBLIP and the instruction-aware Q-Former)

InstructBLIP (2023) mở rộng Q-Former bằng cách bổ sung thêm một đầu vào: chính câu văn bản chỉ dẫn (instruction text). Tại thời điểm thực hiện chú ý chéo, các truy vấn giờ đây có thể truy cập vào cả các mảng ảnh và câu chỉ dẫn. Nhờ vậy, các truy vấn có thể chuyên biệt hóa theo từng chỉ dẫn cụ thể ("hãy đếm số xe hơi", "mô tả tâm trạng trong ảnh") thay vì chỉ học một bản tóm tắt cố định duy nhất. Điều này đem lại sự cải thiện điểm chuẩn đáng kể trên các tác vụ chưa từng huấn luyện (held-out tasks).

### MiniGPT-4 và cách tiếp cận chỉ dùng bộ chiếu (MiniGPT-4 and the projector-only approach)

MiniGPT-4 giữ lại Q-Former nhưng chỉ huấn luyện lớp chiếu tuyến tính đầu ra trong khi đóng băng tất cả các thành phần còn lại. Rất rẻ về mặt tính toán, nhưng đánh đổi lại là chất lượng — các truy vấn sử dụng trọng số sẵn có của BLIP-2 chứ không được tối ưu riêng. Phù hợp cho thử nghiệm nhanh, nhưng không phải là kiến trúc tối ưu nhất.

### Tại sao LLaVA chọn giải pháp đơn giản hơn (Why LLaVA went simpler)

LLaVA (2023, Bài học 12.05) đã thay thế Q-Former bằng một mạng Perceptron đa tầng (MLP) 2 lớp đơn giản để chiếu từng token mảng ảnh của ViT vào không gian của LLM — 576 token cho một lưới 24x24, toàn bộ đều được nạp vào LLM. Khả năng nén kém hơn nhưng cho phép LLM trực tiếp chú ý trên từng mảng ảnh thô. Vào thời điểm ra mắt, thiết kế này từng gây tranh cãi; nhưng đến cuối năm 2023 nó đã chiếm thế áp đảo nhờ dữ liệu chỉ dẫn thị giác (LLaVA-Instruct-150k) chứng minh rằng MLP có thể học để giữ lại đầy đủ tín hiệu. Sự đánh đổi: cửa sổ ngữ cảnh của LLaVA đầy nhanh hơn, nhưng kiến trúc mở rộng rất tự nhiên sang đa hình ảnh và video.

Đến năm 2026, lĩnh vực này phân định rõ: Q-Former được ưu tiên khi ngân sách token bị thắt chặt (video dài, nhiều hình ảnh); còn bộ chiếu MLP thống trị khi chất lượng tối đa trên từng token là mục tiêu cao nhất.

### Chú ý chéo có cổng: Flamingo, tiền thân kiến trúc (Gated cross-attention: Flamingo, the ancestor)

Flamingo (Bài học 12.04) xuất hiện trước BLIP-2 và sử dụng cùng ý tưởng chú ý chéo nhưng chèn vào từng lớp của LLM đóng băng, chứ không chỉ dừng lại ở một cầu nối độc lập ở đầu vào. BLIP-2 đã chứng minh rằng bạn chỉ cần nén thông tin tại lớp đầu vào là hệ thống đã hoạt động hiệu quả. Các mô hình như Gemini và Idefics kết hợp cả hai: token hình ảnh xen kẽ ở đầu vào kèm cơ chế chú ý chéo có cổng tùy chọn cho học trong ngữ cảnh ít mẫu (in-context few-shot).

### Các hậu duệ năm 2026 (The 2026 descendants)

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4, và hầu hết các mô hình video - ngôn ngữ nhằm tiết kiệm ngân sách token.
- Bộ tái lấy mẫu Perceiver (Perceiver resampler): biến thể của Flamingo (Bài học 12.04); họ mô hình Idefics, Eagle, OmniMAE.
- Bộ chiếu MLP (MLP projector): LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Gộp chú ý (Attention pool): VILA, PaliGemma.

Cả bốn phương án đều có giá trị thực tiễn. Quyết định lựa chọn phụ thuộc vào việc bạn bị hạn chế bởi ngân sách token hay ưu tiên chất lượng trên từng token.

````figure
modality-projection
````

## Thực hành (Use It)

`code/main.py` xây dựng cơ chế chú ý chéo theo phong cách Q-Former bằng thư viện chuẩn:

1. Giả lập 256 token mảng ảnh (số chiều 128).
2. Khởi tạo 32 truy vấn học được (số chiều 128).
3. Chạy chú ý chéo tích vô hướng có co giãn (scaled-dot-product cross-attention: Q từ các truy vấn, K/V từ các mảng ảnh).
4. Chiếu sang số chiều của LLM (512) thông qua một lớp tuyến tính.
5. Xuất ra 32 token thị giác sẵn sàng nạp vào LLM.

Toàn bộ phép toán được viết bằng Python thuần (các vòng lặp lồng nhau trên vector). Tuy là mô hình đồ chơi nhưng đúng chuẩn cấu trúc toán học. Ma trận trọng số chú ý được in ra để bạn có thể quan sát trực tiếp mỗi truy vấn trích xuất thông tin từ những mảng ảnh nào.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-modality-bridge-picker.md`. Cho một cấu hình VLM mục tiêu (số lượng token của bộ mã hóa thị giác, ngân sách ngữ cảnh của LLM, ràng buộc triển khai, mục tiêu chất lượng), kỹ năng này sẽ đề xuất nên dùng Q-Former, MLP hay bộ tái lấy mẫu Perceiver kèm theo lý giải ngắn gọn và ước tính số lượng tham số cho từng cầu nối.

## Bài tập (Exercises)

1. Triển khai khối chú ý chéo bằng PyTorch. Xác minh rằng với 32 truy vấn và 256 khóa/giá trị (keys/values), ma trận trọng số chú ý có kích thước 32 x 256 và mỗi hàng có tổng bằng 1 sau hàm softmax.
2. Trong Giai đoạn 1 của BLIP-2, Q-Former chạy ba hàm mất mát đồng thời: ITC, ITM, ITG. Hãy viết chữ ký truyền xuôi (forward signature) cho từng hàm dưới dạng mã giả. Hàm nào bắt buộc nhánh bộ mã hóa văn bản phải hoạt động?
3. So sánh số lượng tham số: Q-Former (12 lớp, chiều ẩn 768) so với bộ chiếu MLP 2 lớp (1408 → 4096, hai lớp). Ở quy mô tham số LLM nào thì chi phí 188M của Q-Former bắt đầu mang lại hiệu quả vượt trội về mặt kinh tế huấn luyện?
4. Đọc Mục 3.2 của bài báo BLIP-2 (arXiv:2301.12597) về cách khởi tạo Q-Former. Giải thích tại sao việc khởi tạo từ BERT-base (thay vì khởi tạo ngẫu nhiên) lại giúp gia tốc tốc độ hội tụ.
5. Với một video dài 10 phút ở tốc độ 1 FPS được lấy mẫu thành 60 khung hình, hãy tính chi phí token trên mỗi khung hình khi sử dụng (Q-Former → 32 token/khung hình) so với (bộ chiếu MLP → 576 token/khung hình). Mỗi loại cầu nối có thể nhét được bao nhiêu khung hình vào cửa sổ ngữ cảnh LLM 128k token trước khi dành chỗ cho văn bản câu nhắc?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Q-Former | "Biến đổi truy vấn" | Transformer cỡ nhỏ với 32 vector truy vấn học được thực hiện chú ý chéo lên các đặc trưng ViT đóng băng |
| Learnable queries (Truy vấn học được) | "Câu nhắc mềm cho thị giác" | Một tập hợp tham số cố định đóng vai trò là phía truy vấn (Q) trong chú ý chéo; được học theo từng mô hình, dùng chung cho mọi đầu vào |
| Cross-attention (Chú ý chéo) | "Q từ nguồn này, K/V từ nguồn kia" | Cơ chế chú ý nơi Query và Key/Value đến từ hai nguồn dữ liệu khác nhau; cách mà các truy vấn trích xuất thông tin từ các mảng ảnh ViT |
| ITC | "Tương phản hình - văn" | Hàm mất mát kiểu CLIP áp dụng giữa các truy vấn đã gộp của Q-Former và token CLS của văn bản |
| ITM | "So khớp hình - văn" | Bộ phân loại nhị phân trên các cặp được khai thác mẫu âm khó; buộc các truy vấn phải phân biệt các điểm không khớp chi tiết |
| ITG | "Sinh văn bản theo hình ảnh" | Hàm mất mát mô hình ngôn ngữ tự hồi quy nơi văn bản được sinh ra có điều kiện theo các truy vấn; buộc truy vấn phải mã hóa nội dung giải mã được thành văn bản |
| Two-stage pretraining (Tiền huấn luyện hai giai đoạn) | "Học biểu diễn rồi học sinh" | Giai đoạn 1 huấn luyện riêng Q-Former (ITC/ITM/ITG); Giai đoạn 2 gắn LLM đóng băng và chỉ huấn luyện lớp chiếu + Q-Former |
| Frozen backbone (Mạng xương sống đóng băng) | "Không tinh chỉnh" | Trọng số của bộ mã hóa thị giác và LLM được giữ cố định; chỉ huấn luyện cầu nối |
| Projection head (Đầu chiếu) | "Lớp tuyến tính sang chiều LLM" | Lớp tuyến tính cuối cùng ánh xạ đầu ra của Q-Former sang không gian nhúng của LLM |
| Perceiver resampler (Bộ tái lấy mẫu Perceiver) | "Phiên bản của Flamingo" | Cơ chế chú ý chéo với các truy vấn học được tương tự; Flamingo chạy một lần cho mỗi ảnh để tạo các biểu diễn tiềm ẩn cố định, rồi đọc qua chú ý chéo có cổng ở mọi lớp LLM |

## Tài liệu tham khảo (Further Reading)

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — Bài báo cốt lõi.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — Phiên bản tiền nhiệm với bộ ba mất mát ITC/ITM/ITG.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "Align before fuse" — nguồn gốc ý tưởng cho huấn luyện giai đoạn 1.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — Q-Former nhận biết chỉ dẫn.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — Phương pháp tiếp cận chỉ dùng bộ chiếu.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — Kiến trúc tổng quát cho chú ý chéo với truy vấn học được.
