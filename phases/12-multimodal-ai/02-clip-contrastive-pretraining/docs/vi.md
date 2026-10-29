# CLIP và Tiền Huấn Luyện Tương Phản Thị Giác - Ngôn Ngữ

> Mô hình CLIP (2021) của OpenAI đã chứng minh một ý tưởng duy nhất đủ sức định hình 5 năm tiếp theo: căn chỉnh một bộ mã hóa hình ảnh (image encoder) và một bộ mã hóa văn bản (text encoder) trong cùng một không gian vector chỉ bằng các cặp hình ảnh - chú thích (image-caption pairs) nhiều nhiễu thu thập từ web và một hàm mất mát tương phản (contrastive loss). Không cần bất kỳ nhãn giám sát nào. 400 triệu cặp dữ liệu. Không gian nhúng tạo ra có khả năng phân loại không cần mẫu (zero-shot classification), truy xuất hình ảnh - văn bản (image-text retrieval), và được tích hợp vào mọi mô hình thị giác - ngôn ngữ (VLM - Vision-Language Model) năm 2026 dưới vai trò tháp thị giác (vision tower). SigLIP 2 (2025) đã thay thế hàm softmax bằng sigmoid và mở rộng vượt qua CLIP với chi phí thấp hơn. Bài học này đi qua nền tảng toán học từ InfoNCE đến hàm mất mát từng cặp sigmoid và xây dựng bước huấn luyện bằng Python thư viện chuẩn (stdlib).

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (stdlib, cài đặt hàm mất mát InfoNCE + sigmoid)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 01 (Mảng ảnh ViT), Giai đoạn 7 (Transformers)  
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Rút ra hàm mất mát InfoNCE từ thông tin tương hỗ (mutual information) và triển khai một phiên bản vector hóa ổn định về mặt số học.
- Giải thích lý do hàm mất mát từng cặp sigmoid (SigLIP) có thể mở rộng tới kích thước batch 32.768+ mà không chịu chi phí truyền thông gom toàn bộ (all-gather) như softmax đòi hỏi.
- Chạy phân loại zero-shot trên ImageNet bằng cách xây dựng các mẫu văn bản (`a photo of a {class}`) và lấy argmax theo độ tương đồng cosine (cosine similarity).
- Nêu tên 4 đòn bẩy mà quá trình tiền huấn luyện CLIP / SigLIP mang lại: kích thước batch, nhiệt độ (temperature), mẫu câu nhắc (prompt template), chất lượng dữ liệu.

## Vấn đề

Trước thời CLIP, thị giác máy tính phụ thuộc vào học có giám sát (supervised learning). Quy trình thường là: thu thập các tập dữ liệu được gán nhãn thủ công (như ImageNet: 1,2 triệu hình ảnh, 1.000 lớp), huấn luyện một mạng nơ-ron tích chập (CNN), rồi đưa vào ứng dụng. Tuy nhiên, nhãn rất tốn kém để tạo ra, mang nặng thiên kiến của người dán nhãn, và không thể chuyển giao sang các tác vụ mới nếu không thực hiện tinh chỉnh (finetuning).

Trong khi đó, mạng internet chứa hàng tỷ cặp hình ảnh - chú thích được gán nhãn lỏng lẻo hoàn toàn miễn phí. Một bức ảnh chụp chú chó golden retriever với văn bản thay thế (alt text) "chú chó Max của tôi ở công viên" đã mang sẵn một tín hiệu giám sát — văn bản đó mô tả bức ảnh. Câu hỏi đặt ra là: liệu chúng ta có thể chuyển hóa nguồn dữ liệu dồi dào này thành một quy trình huấn luyện hữu ích hay không?

Câu trả lời của CLIP: coi các cặp ảnh - chú thích như một tác vụ so khớp (matching task). Cho một batch gồm $N$ hình ảnh và $N$ chú thích, hãy học cách khớp từng hình ảnh với chú thích tương ứng của nó giữa $N-1$ yếu tố gây nhiễu (distractors). Tín hiệu giám sát ở đây đơn giản là: "hai thực thể này thuộc về nhau; $N-1$ thực thể kia thì không." Không cần nhãn phân lớp. Không cần con người gán nhãn thủ công. Chỉ cần một hàm mất mát tương phản.

Không gian nhúng thu được làm được nhiều điều hơn cả mục đích huấn luyện ban đầu của CLIP. Việc phân loại zero-shot trên ImageNet thành công vì câu nhắc "a photo of a cat" (bức ảnh về một con mèo) được nhúng gần với các bức ảnh chụp mèo mà chưa từng có nhãn "mèo" rõ ràng trước đó. Đây chính là bước đột phá mở đường cho mọi VLM hiện đại năm 2026.

## Khái niệm

### Bộ mã hóa kép (The dual encoder)

CLIP sở hữu hai tháp xử lý (two towers):

- Bộ mã hóa hình ảnh (image encoder) `f`: ViT hoặc ResNet, xuất ra một vector $D$ chiều cho mỗi hình ảnh.
- Bộ mã hóa văn bản (text encoder) `g`: transformer cỡ nhỏ, xuất ra một vector $D$ chiều cho mỗi chú thích.

Cả hai tháp đều chuẩn hóa vector đầu ra về độ dài đơn vị (unit length). Khi đó độ tương đồng cosine là `cos(f(x), g(y)) = f(x)^T g(y)` vì cả hai vector đều có chuẩn bằng 1.

Với một batch gồm $N$ cặp (hình ảnh, chú thích), ta xây dựng ma trận tương đồng $S$ có kích thước $(N, N)$:

```math
S[i, j] = \cos(f(x_i), g(y_j)) / \tau
```

trong đó $\tau$ (tau) là tham số nhiệt độ học được (CLIP khởi tạo giá trị 0.07; được học trong không gian log).

### Hàm mất mát InfoNCE (InfoNCE loss)

CLIP áp dụng hàm mất mát entropy chéo (cross-entropy - CE) đối xứng trên các hàng và các cột:

```python
loss_i2t = CE(S, labels=identity)     # mẫu dương của mỗi ảnh là chú thích của chính nó
loss_t2i = CE(S^T, labels=identity)   # mẫu dương của mỗi chú thích là hình ảnh của chính nó
loss = (loss_i2t + loss_t2i) / 2
```

Đây chính là hàm mất mát InfoNCE. Hàm softmax trong entropy chéo buộc mỗi hình ảnh phải khớp với chú thích của nó vượt trội hơn tất cả các chú thích khác trong batch. Các "mẫu âm" (negatives) chính là toàn bộ các phần tử còn lại trong batch. Batch càng lớn = càng nhiều mẫu âm = tín hiệu giám sát càng mạnh. CLIP được huấn luyện với batch size 32.768; quy mô thực sự tạo nên sự khác biệt.

### Nhiệt độ (Temperature)

Tham số $\tau$ kiểm soát độ sắc nhọn của hàm softmax. Khi $\tau$ thấp → phân phối sắc nhọn, tạo hiệu ứng khai thác mẫu âm khó (hard negative mining). Khi $\tau$ cao → phân phối mượt mà, mọi mẫu đều đóng góp tương đương. CLIP học giá trị $\log(1/\tau)$, có kẹp biên (clipping) để tránh suy biến. SigLIP 2 cố định $\tau$ ban đầu và học thêm một độ lệch (bias).

### Tại sao hàm sigmoid mở rộng quy mô tốt hơn (SigLIP)

Softmax yêu cầu toàn bộ ma trận tương đồng phải được đồng bộ hóa. Trong huấn luyện phân tán (distributed training), bạn bắt buộc phải thực hiện thao tác gom toàn bộ (`all-gather`) mọi vector nhúng đến tất cả các tiến trình (replicas), sau đó mới tính softmax. Chi phí truyền thông này tăng theo bậc hai theo quy mô cụm máy tính toán (world size).

SigLIP thay thế softmax bằng hàm sigmoid trên từng phần tử: với mỗi cặp `(i, j)`, hàm mất mát là một bài toán phân loại nhị phân "đây có phải là cặp tương ứng không?" Nhãn lớp dương nằm trên đường chéo chính, còn lại đều là nhãn âm. Hàm mất mát có dạng:

```math
L = -\frac{1}{N} \sum_{(i, j)} \Big[ y_{ij} \log \mathop{\mathrm{sigmoid}}(S[i,j]) + (1-y_{ij}) \log \mathop{\mathrm{sigmoid}}(-S[i,j]) \Big]
```

Trong đó $y_{ij} = 1$ nếu $i == j$, ngược lại bằng 0. Hàm mất mát của mỗi cặp là độc lập. Không cần thao tác `all-gather` tốn kém. Mỗi GPU chỉ việc tính toán khối cục bộ của mình rồi lấy tổng. Nhờ đó, SigLIP 2 mở rộng quy mô lên batch 32k–512k với chi phí truyền thông rất thấp so với CLIP.

### Phân loại không cần mẫu (Zero-shot classification)

Cho $N$ tên lớp mục tiêu, với mỗi lớp ta tạo một mẫu câu nhắc văn bản (prompt template):

```text
"a photo of a {class}"
```

Nhúng từng câu mẫu bằng bộ mã hóa văn bản. Nhúng hình ảnh cần phân loại bằng bộ mã hóa hình ảnh. Tìm nhãn có độ tương đồng cosine lớn nhất (`argmax`) = lớp dự đoán. Hoàn toàn không cần huấn luyện lại trên các lớp mục tiêu.

Các mẫu câu nhắc đóng vai trò rất quan trọng. Bài báo gốc của CLIP đã sử dụng 80 mẫu câu cho mỗi lớp (ảnh chụp thông thường, nghệ thuật, tranh vẽ, v.v.) và lấy trung bình các vector nhúng, giúp tăng thêm 3 điểm độ chính xác trên ImageNet. Trong ứng dụng thực tế hiện nay, người ta thường chọn từ 1 đến 2 mẫu tối ưu.

### Thăm dò tuyến tính và tinh chỉnh (Linear probes and finetuning)

Zero-shot là một đường cơ sở (baseline) mạnh mẽ. Một bộ thăm dò tuyến tính (linear probe — huấn luyện duy nhất một lớp tuyến tính trên các đặc trưng CLIP được đóng băng trọng số) sẽ vượt qua zero-shot trên các tác vụ cùng miền dữ liệu (in-domain). Tinh chỉnh toàn bộ (full finetuning) mang lại kết quả tốt nhất trên miền dữ liệu đó nhưng có thể làm suy giảm khả năng tổng quát hóa zero-shot sang miền khác. Đây là ba chế độ với các đánh đổi rõ ràng.

### SigLIP 2: NaFlex và các đặc trưng dày đặc

SigLIP 2 (2025) bổ sung thêm:
- NaFlex: một mô hình duy nhất xử lý được các tỷ lệ khung hình và độ phân giải biến thiên.
- Các đặc trưng dày đặc (dense features) tốt hơn cho phân đoạn và ước lượng độ sâu, hướng tới việc làm mạng xương sống đóng băng trong các VLM.
- Đa ngôn ngữ: được huấn luyện trên hơn 100 ngôn ngữ, trong khi CLIP gần như chỉ hỗ trợ tiếng Anh.
- Mở rộng lên quy mô 1 tỷ tham số (1B parameters) so với mức trần 400M của CLIP.

Trong các VLM nguồn mở năm 2026, SigLIP 2 SO400m/14 là tháp thị giác mặc định. CLIP vẫn được dùng phổ biến cho bài toán truy xuất văn bản - hình ảnh thuần túy nơi phân phối dữ liệu huấn luyện LAION-2B phù hợp với mẫu truy vấn.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

- ALIGN (Google, 2021): cùng ý tưởng với CLIP, mở rộng lên quy mô 1,8 tỷ cặp dữ liệu, dù chứa hơn 90% nhiễu. Chứng minh dữ liệu nhiều nhiễu vẫn mở rộng tốt nếu đủ lớn.
- OpenCLIP (LAION): bản tái hiện mã nguồn mở của CLIP trên tập LAION-400M / 2B, cung cấp nhiều kích cỡ checkpoint mở chuẩn mực.
- EVA-CLIP: khởi tạo trọng số từ mô hình mô phỏng hình ảnh có che mờ (masked image modeling); là mạng xương sống rất mạnh cho VLM.
- BASIC: mô hình lai giữa CLIP và ALIGN của Google. Tất cả đều thuộc cùng một gia đình mô hình, chỉ khác biệt về dữ liệu và kỹ thuật tinh chỉnh.

### Giới hạn trần của zero-shot (The zero-shot ceiling)

Các mô hình thuộc họ CLIP chạm ngưỡng khoảng 76% độ chính xác zero-shot trên ImageNet (CLIP-G, OpenCLIP-G). Vượt qua ngưỡng này đòi hỏi tập dữ liệu lớn hơn nhiều (SigLIP 2 đạt trên 80%) hoặc thay đổi kiến trúc (thêm đầu phân loại có giám sát, tăng tham số). Tiêu chuẩn đánh giá này đang dần bão hòa; giá trị cốt lõi thực sự nằm ở không gian nhúng biểu diễn làm nền tảng cho các VLM hạ nguồn.

````figure
multimodal-fusion
````

## Thực hành (Use It)

`code/main.py` triển khai:

1. Bộ mã hóa kép giả lập đơn giản (trích xuất đặc trưng hình ảnh dựa trên hàm băm, đặc trưng văn bản dựa trên ký tự) để bạn quan sát cấu trúc của InfoNCE mà không cần đến thư viện numpy.
2. Hàm mất mát InfoNCE bằng Python thuần (đảm bảo ổn định số học thông qua kỹ thuật log-sum-exp).
3. Hàm mất mát từng cặp sigmoid để so sánh trực tiếp.
4. Quy trình phân loại zero-shot: tính độ tương đồng cosine với tập các câu nhắc văn bản và lấy argmax để đưa ra dự đoán.

Hãy chạy thử và quan sát đường cong suy giảm mất mát (loss curve). Những con số cụ thể chỉ mang tính minh họa; nhưng hình dạng hội tụ phản ánh đúng những gì diễn ra trong quá trình huấn luyện CLIP thực tế.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-clip-zero-shot.md`. Cho một tập hợp hình ảnh (thông qua đường dẫn) và danh sách các lớp mục tiêu, kỹ năng này sẽ tạo câu nhắc văn bản theo mẫu CLIP, nhúng cả hai phía bằng một checkpoint xác định (ví dụ: `openai/clip-vit-large-patch14`), và trả về kết quả dự đoán top-1 / top-5 kèm điểm tương đồng. Kỹ năng sẽ từ chối đưa ra kết luận về các lớp không nằm trong danh sách câu nhắc.

## Bài tập (Exercises)

1. Tính toán bằng tay hàm mất mát InfoNCE cho một batch gồm 4 cặp. Xây dựng ma trận tương đồng kích thước 4x4, tính softmax, lấy các phần tử trên đường chéo chính và tính entropy chéo. Đối chiếu phép tính tay này với kết quả từ mã nguồn Python của bạn.
2. SigLIP sử dụng thêm một tham số độ lệch (bias) $b$ bên cạnh nhiệt độ: $S'[i,j] = S[i,j]/\tau + b$. Vai trò của $b$ là gì khi batch có sự mất cân bằng lớp nghiêm trọng (số lượng mẫu âm nhiều hơn rất nhiều so với mẫu dương trên mỗi hàng)? Đọc Mục 3 của bài báo SigLIP (arXiv:2303.15343).
3. Xây dựng bộ phân loại zero-shot để phân biệt chó và mèo. Thử nghiệm với hai mẫu câu nhắc: `a photo of a {class}` và `a picture of a {class}`. Đo độ chính xác trên 100 ảnh kiểm thử. Việc kết hợp tập hợp (ensemble) nhiều mẫu câu có tốt hơn một mẫu đơn lẻ không?
4. Tính toán chi phí truyền thông của InfoNCE softmax so với sigmoid từng cặp trong một lượt huấn luyện trên cụm 512 GPU với kích thước batch 32k. Thuật toán nào tăng theo cấp độ $O(N)$, thuật toán nào tăng theo $O(N^2)$? Trích dẫn Mục 4 bài báo SigLIP.
5. Đọc bài báo quy luật mở rộng của OpenCLIP (arXiv:2212.07143, Cherti et al.). Tái hiện lại kết luận của họ về việc mở rộng dữ liệu từ các đồ thị: ở kích thước mô hình cố định, mối quan hệ tuyến tính theo log giữa độ chính xác zero-shot trên ImageNet và dung lượng dữ liệu huấn luyện diễn ra như thế nào?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| InfoNCE | "Hàm mất mát tương phản" | Entropy chéo trên ma trận tương đồng của một batch; mẫu dương của mỗi phần tử là đối tác cặp của nó, mẫu âm là tất cả các phần tử còn lại |
| Sigmoid loss | "Mất mát SigLIP" | Entropy chéo nhị phân cho từng cặp; không cần softmax, không cần gom toàn bộ dữ liệu (all-gather), mở rộng chi phí thấp trong huấn luyện phân tán |
| Temperature (Nhiệt độ) | "Hệ số tau" | Hệ số vô hướng co giãn giá trị logit trước khi đưa vào softmax/sigmoid; kiểm soát độ sắc nhọn của phân phối |
| Zero-shot | "Phân loại không cần tinh chỉnh" | Sử dụng các câu nhắc văn bản để tạo vector nhúng cho từng lớp và phân loại dựa trên độ tương đồng cosine; không cần huấn luyện trên lớp mục tiêu |
| Prompt template (Mẫu câu nhắc) | "Mẫu a photo of a..." | Khung văn bản bao bọc tên lớp; có thể làm thay đổi độ chính xác zero-shot từ 1 đến 5 điểm |
| Dual encoder (Bộ mã hóa kép) | "Kiến trúc hai tháp" | Gồm một bộ mã hóa hình ảnh + một bộ mã hóa văn bản, cùng xuất đầu ra vào một không gian chung $D$ chiều |
| Hard negative (Mẫu âm khó) | "Yếu tố gây nhiễu khó" | Một mẫu âm có độ tương đồng rất gần với mẫu dương, buộc mô hình phải học kỹ lưỡng để phân tách |
| Linear probe (Thăm dò tuyến tính) | "Đóng băng + một lớp" | Chỉ huấn luyện một bộ phân loại tuyến tính trên các đặc trưng đã đóng băng; dùng để đo chất lượng biểu diễn đặc trưng |
| NaFlex | "Độ phân giải linh hoạt gốc" | Khả năng của SigLIP 2 tiếp nhận hình ảnh ở bất kỳ tỷ lệ khung hình và độ phân giải nào mà không cần thay đổi kích thước ảnh cưỡng bức |
| Temperature scaling (Co giãn nhiệt độ) | "Tham số hóa log cho tau" | CLIP tham số hóa $\log(1/\tau)$ để độ dốc gradient ổn định; áp dụng kẹp biên để tránh giá trị $\tau$ suy biến về gần 0 |

## Tài liệu tham khảo (Further Reading)

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — Bài báo gốc về CLIP.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — Bài báo về SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — Phiên bản đa ngôn ngữ + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — Khả năng mở rộng với dữ liệu web nhiều nhiễu.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — Quy luật mở rộng của OpenCLIP.
