# Cơ chế chú ý vi phân (Differential Attention - V2)

> Cơ chế chú ý Softmax luôn phân bổ một lượng xác suất nhỏ lên mọi token không liên quan. Trên độ dài 100k token, lượng nhiễu tích lũy đó sẽ cộng dồn lại và nhấn chìm tín hiệu thực sự. Differential Transformer (Ye et al., ICLR 2025) giải quyết triệt để vấn đề này bằng cách tính toán sự chú ý dưới dạng hiệu số của hai hàm softmax, qua đó triệt tiêu ngưỡng nhiễu nền (noise floor) dùng chung. DIFF V2 (Microsoft, tháng 1 năm 2026) là bản viết lại cho ngăn xếp sản xuất: khớp hoàn toàn độ trễ giải mã với Transformer cơ sở, không đòi hỏi nhân CUDA tùy chỉnh và tương thích hoàn toàn với FlashAttention. Bài học này đi từ V1 đến V2 một cách toàn diện, kèm theo một bản cài đặt thử nghiệm cơ chế trừ vi phân chạy hoàn toàn bằng Python thư viện chuẩn (stdlib).

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7 · 02 (Tự chú ý), Giai đoạn 7 · 15 (Các biến thể cơ chế chú ý), Giai đoạn 10 · 14 (Phân tích chi tiết kiến trúc)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu chính xác lý do vì sao cơ chế chú ý softmax có ngưỡng nhiễu nền và tại sao ngưỡng nhiễu này lại tăng theo chiều dài ngữ cảnh.
- Thiết lập công thức toán học của cơ chế chú ý vi phân và giải thích vì sao phép trừ loại bỏ được thành phần nhiễu dùng chung trong khi vẫn bảo toàn tín hiệu mục tiêu.
- Điểm lại bản so sánh khác biệt giữa V1 và V2: những gì đã nhanh hơn, đơn giản hơn, ổn định hơn và lý do vì sao mỗi thay đổi lại mang tính sống còn đối với giai đoạn tiền huấn luyện quy mô lớn.
- Tự tay cài đặt cơ chế chú ý vi phân từ đầu bằng Python thuần túy và kiểm chứng thực nghiệm đặc tính triệt tiêu nhiễu trên một truy vấn tổng hợp gồm tín hiệu lẫn nhiễu.

## Vấn đề thực tế

Cơ chế chú ý softmax tiêu chuẩn sở hữu một đặc tính toán học biến thành cơn đau đầu vận hành khi mở rộng quy mô. Với một truy vấn $q$, trọng số chú ý được tính bằng $\mathop{\mathrm{softmax}}(qK^T / \sqrt{d})$. Hàm softmax không bao giờ tạo ra giá trị 0 tuyệt đối: mọi token không liên quan đều nhận được một khối lượng xác suất dương nhỏ. Lượng xác suất dư thừa này chính là nhiễu, và nó tăng tỷ lệ thuận với độ dài ngữ cảnh. Ở độ dài 128k token, ngay cả khi mỗi token không khớp chỉ nhận được 0,0001% xác suất, thì 127.999 token còn lại cộng dồn lại sẽ chiếm tới khoảng 13% tổng trọng số. Mô hình buộc phải học cách điều hướng vòng qua một ngưỡng nhiễu nền ngày càng phình to theo ngữ cảnh.

Về mặt thực nghiệm, hiện tượng này biểu hiện dưới dạng sự can thiệp giữa các đầu chú ý: trích dẫn ảo (hallucinated citations) trong RAG ngữ cảnh dài, lỗi "lạc lối ở giữa" (lost-in-the-middle) trên các tác vụ truy xuất 100k token, và sự suy giảm độ chính xác tinh vi trên các bài kiểm tra "cây kim trong đống rơm" (needle-in-a-haystack) vượt quá 32k token. Bài báo Differential Transformer (arXiv:2410.05258, ICLR 2025) đã đo lường khoảng cách này: các mô hình DIFF Transformer đạt độ hỗn loạn (perplexity) thấp hơn, độ chính xác ngữ cảnh dài cao hơn và ít bị ảo giác hơn so với các mô hình cơ sở cùng kích thước.

Tuy nhiên, DIFF V1 gặp phải ba vấn đề khiến nó chưa thể bước chân vào các đường ống tiền huấn luyện biên giới: bộ nhớ đệm giá trị (value cache) phải nạp hai lần cho mỗi bước giải mã, đòi hỏi nhân CUDA tùy chỉnh phá vỡ khả năng tương thích với FlashAttention, và tầng chuẩn hóa RMSNorm trên mỗi đầu làm mất ổn định quá trình huấn luyện dài hạn ở quy mô trên 70B tham số. DIFF V2 (blog Microsoft unilm, ngày 20 tháng 1 năm 2026) đã giải quyết triệt để cả ba hạn chế này. Bài học này sẽ dẫn dắt bạn qua cả hai phiên bản, xây dựng toán tử vi phân và đánh giá hiệu quả triệt tiêu nhiễu trên một truy vấn mô phỏng.

## Khái niệm cốt lõi

### Ngưỡng nhiễu nền của Softmax

Với một truy vấn $q$ và các khóa $K = [k_1, \dots, k_N]$, trọng số chú ý là:

```math
w_i = \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}
```

Không có giá trị $w_i$ nào bằng 0. Nếu $k_i$ hoàn toàn không liên quan đến $q$, điểm số tích vô hướng $q \cdot k_i$ không phải bằng 0 -- mà nó dao động quanh mức 0 với phương sai $\|q\|^2 / d$. Sau phép chuẩn hóa softmax, mỗi token không liên quan vẫn đóng góp một phần $O(1/N)$ vào tổng có trọng số. Tổng đóng góp của các token không liên quan là $O((N-1)/N) = O(1)$ -- một đại lượng không hề nhỏ.

Điều mà mô hình thực sự mong muốn là một cơ chế tương tự top-$k$ cứng: gán trọng số cao cho các token khớp thực sự và trọng số gần như bằng 0 cho tất cả các vị trí còn lại. Hàm softmax quá mượt mà để có thể làm được điều đó một cách trực tiếp.

### Ý tưởng cơ chế chú ý vi phân

Chia các phép chiếu $Q$ và $K$ của mỗi đầu thành hai phần: $Q = (Q_1, Q_2)$ và $K = (K_1, K_2)$. Tính toán hai bản đồ chú ý:

```math
\begin{aligned}
A_1 &= \mathop{\mathrm{softmax}}(Q_1 K_1^T / \sqrt{d}) \\
A_2 &= \mathop{\mathrm{softmax}}(Q_2 K_2^T / \sqrt{d})
\end{aligned}
```

Đầu ra của cơ chế chú ý vi phân:

```math
\mathop{\mathrm{DiffAttn}} = (A_1 - \lambda \cdot A_2) V
```

Phép trừ triệt tiêu bất kỳ phân phối nhiễu nào mà hai bản đồ cùng chia sẻ. Nếu cả hai bản đồ đều gán trọng số gần như đồng đều lên 127k token không liên quan (điều tự nhiên xảy ra khi khởi tạo ngẫu nhiên), các giá trị đó sẽ triệt tiêu lẫn nhau. Trong khi đó, tín hiệu thực sự -- tức trọng số đỉnh tập trung vào một vài token liên quan -- chỉ bị triệt tiêu nếu nó xuất hiện ở cả hai bản đồ với cùng một độ lớn, điều chắc chắn sẽ không xảy ra khi mô hình được huấn luyện.

Tham số $\lambda$ là một đại lượng vô hướng học được trên từng đầu, được tham số hóa dưới dạng `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Nó có thể nhận giá trị âm. Giá trị `lambda_init` thường được khởi tạo là một số dương nhỏ như 0,8.

### Vì sao cơ chế này tương đồng với việc khử nhiễu có định hướng?

Hãy hình dung hai micro thu âm cùng ghi lại giọng nói của một người. Cả hai micro đều thu giọng người nói cộng với tiếng ồn nền tương quan xung quanh. Lấy tín hiệu của micro này trừ đi micro kia, tiếng ồn nền dùng chung sẽ biến mất. Giọng nói vẫn được giữ lại vì hai tín hiệu khác nhau về pha hoặc biên độ vừa đủ để không tự triệt tiêu nhau hoàn toàn. Tham số $\lambda$ trên mỗi đầu học chính xác sự cân bằng này.

### So sánh giữa V1 và V2: Bản chất khác biệt

V1 giữ nguyên tổng số lượng tham số bằng với Transformer cơ sở. Để có được hai truy vấn trên mỗi đầu, nó giảm một nửa số chiều của mỗi đầu (head dimension). Điều này làm suy giảm năng lực biểu diễn của đầu và -- nghiêm trọng hơn -- giảm một nửa kích thước bộ nhớ đệm giá trị (value cache) trên mỗi đầu. Khi giải mã, hệ thống phải nạp bộ nhớ đệm giá trị hai lần cho mỗi bước (một lần cho mỗi nhánh softmax). Kết quả là tốc độ giải mã chậm hơn đáng kể so với mô hình cơ sở dù có cùng số lượng tham số.

V2 nhân đôi số lượng đầu truy vấn (query heads) trong khi giữ nguyên số lượng đầu KV (mượn tham số từ phép chiếu ngược lên). Kích thước chiều của mỗi đầu được giữ nguyên như Transformer cơ sở. Sau phép trừ vi phân, chiều phụ dôi ra sẽ được chiếu ngược xuống để khớp với phép chiếu $O_W$ của Transformer cơ sở. Ba ưu điểm xuất hiện đồng thời:

1. Tốc độ giải mã tương đương hoàn toàn với mô hình cơ sở (bộ nhớ đệm KV chỉ nạp đúng một lần).
2. Tương thích trực tiếp với FlashAttention mà không cần viết nhân GPU tùy chỉnh.
3. Cường độ số học (arithmetic intensity) khi giải mã tăng lên (thực hiện nhiều phép tính hơn trên mỗi byte nạp từ bộ nhớ HBM).

V2 cũng loại bỏ tầng RMSNorm trên mỗi đầu mà V1 đã dùng để ổn định phép trừ. Ở quy mô tiền huấn luyện cấp độ 70B, tầng RMSNorm đó làm quá trình huấn luyện ở các bước sau trở nên mất ổn định. V2 thay thế nó bằng một cơ chế khởi tạo đơn giản hơn giúp duy trì sự ổn định mà không cần thêm mô-đun phụ trợ.

### Khi nào nên áp dụng cơ chế này?

| Tác vụ | Lợi ích |
|---|---|
| RAG ngữ cảnh dài (64k+) | Bản đồ chú ý sạch hơn, giảm thiểu các trích dẫn ảo giác |
| Benchmark cây kim trong đống rơm | Cải thiện độ chính xác vượt trội khi ngữ cảnh vượt quá 32k |
| Hỏi đáp đa tài liệu (Multi-document QA) | Giảm thiểu sự can thiệp gây nhiễu giữa các tài liệu khác nhau |
| Hoàn thiện mã nguồn ở độ dài 8k | Lợi ích không đáng kể, không bõ công thay đổi kiến trúc |
| Hội thoại ngắn (< 4k) | Thực tế không có sự khác biệt so với mô hình cơ sở |

Giá trị của cơ chế chú ý vi phân tăng dần theo chiều dài ngữ cảnh. Ở 4k token, ngưỡng nhiễu nền đủ nhỏ để cơ chế chú ý tiêu chuẩn hoạt động tốt. Nhưng ở 128k token, nó sẽ trở thành gánh nặng kìm hãm mô hình.

### Khả năng kết hợp với các núm vặn kiến trúc khác năm 2026

| Tính năng | Tương thích với DIFF V2? |
|---|---|
| GQA | Có (V2 tăng số đầu Q, giữ nguyên đầu KV) |
| MLA (DeepSeek) | Về nguyên lý là có, hiện chưa có bài báo công khai nào kết hợp cả hai |
| MoE | Có (cơ chế chú ý hoàn toàn độc lập với khối MLP) |
| RoPE | Có (không thay đổi) |
| YaRN / co giãn ngữ cảnh dài | Có (chính là nơi DIFF phát huy tác dụng mạnh mẽ nhất) |
| FlashAttention | Có trong V2 (V1 thì không) |
| Giải mã suy đoán | Có (thay đổi trong cơ chế chú ý là trong suốt đối với vòng lặp suy đoán) |

````figure
differential-attention
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` triển khai cơ chế chú ý vi phân bằng Python thuần túy. Một truy vấn giả định với cấu trúc tín hiệu cộng nhiễu được thiết kế sẵn cho phép bạn đo lường trực tiếp tỷ lệ triệt tiêu nhiễu.

### Bước 1: Chú ý softmax tiêu chuẩn

Sử dụng các thao tác ma trận thư viện chuẩn: danh sách lồng nhau, nhân ma trận thủ công, tính softmax có trừ giá trị cực đại để đảm bảo ổn định số học.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Bước 2: Tách Q, K thành hai nửa

Phong cách V1: giảm một nửa chiều của mỗi đầu. Phong cách V2: giữ nguyên chiều của đầu và nhân đôi số lượng đầu. Bản triển khai thử nghiệm sử dụng phong cách V1 để đảm bảo tính trực quan về mặt sư phạm -- nền tảng toán học là đồng nhất, chỉ khác biệt ở cách tổ chức ma trận.

### Bước 3: Hai nhánh softmax và phép trừ vi phân

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Lưu ý: các trọng số đầu ra có thể mang giá trị âm. Điều này hoàn toàn bình thường -- bộ nhớ đệm giá trị vẫn xử lý tốt các đóng góp có dấu. Phép chiếu $V$ tiếp theo sẽ hấp thụ các giá trị dấu này.

### Bước 4: Đo lường mức độ triệt tiêu nhiễu

Xây dựng một chuỗi nhân tạo có độ dài 1024 token. Đặt token mang tín hiệu tại một vị trí xác định, điền toàn bộ các vị trí còn lại bằng nhiễu ngẫu nhiên. Tính toán: (a) trọng số chú ý softmax tiêu chuẩn tại vị trí tín hiệu và (b) trọng số chú ý vi phân. Đo lường tỷ số tín hiệu trên nhiễu (SNR) trong từng trường hợp. Cơ chế chú ý DIFF liên tục mang lại tỷ số SNR cao hơn từ $3\times$ đến $10\times$ tùy thuộc vào mức độ hai nhánh được huấn luyện để phân hóa.

### Bước 5: Hạch toán tham số giữa V1 và V2

Với một cấu hình cho trước (hidden=4096, heads=32, d_head=128), in ra:

- Transformer cơ sở: Q, K, V mỗi ma trận có kích thước `hidden * hidden`, MLP có kích thước $4 \times \text{hidden}$.
- DIFF V1: Q, K mỗi ma trận kích thước `hidden * hidden`, V kích thước `hidden * hidden` (không đổi), chiều của đầu giảm một nửa trong nội bộ. Bổ sung thêm các tham số $\lambda$ trên mỗi đầu $(O(\text{heads} \cdot d_{\text{head}}))$.
- DIFF V2: Q kích thước `2 * hidden * hidden`, K kích thước `hidden * hidden`, V kích thước `hidden * hidden`. Chiều dôi ra được chiếu ngược trở lại trước $O_W$. Bổ sung các tham số $\lambda$ tương tự.

Chương trình tính toán chi phí tham số bổ sung cho V2 (khoảng `hidden * hidden` tham số phụ cho mỗi khối chú ý) và in ra màn hình.

## Sử dụng thực tế (Use It)

Tính đến tháng 4 năm 2026, DIFF V2 chưa được tích hợp trên mọi máy chủ suy luận sản xuất, nhưng quá trình tích hợp đang được tiến hành khẩn trương trong vLLM và SGLang. Trong khi đó, mô hình thiết kế này đã xuất hiện trong:

- Các mô hình sản xuất ngữ cảnh dài nội bộ của Microsoft.
- Các nỗ lực tái hiện nghiên cứu trong một số đợt huấn luyện mô hình mở hướng tới ngữ cảnh trên 256k.
- Các kiến trúc lai kết hợp chú ý DIFF với chú ý cửa sổ trượt trên các lớp xen kẽ.

Khi nào bạn nên cân nhắc sử dụng vào năm 2026:

- Huấn luyện một mô hình hoàn toàn mới từ đầu với mục tiêu ngữ cảnh hiệu dụng trên 64k. Hãy tích hợp chú ý vi phân ngay từ đầu; việc huấn luyện lại sau này cực kỳ tốn kém.
- Tinh chỉnh một mô hình ngữ cảnh dài nơi lỗi "lạc lối ở giữa" chiếm ưu thế trong các bài đánh giá của bạn. Một tầng LoRA áp dụng trên các phép chiếu Q có thể xấp xỉ cấu trúc của DIFF.

Khi nào bạn không nên áp dụng:

- Bạn đang phục vụ một mô hình dày đặc đã tiền huấn luyện với hiệu năng ngữ cảnh dài ổn định. Chi phí huấn luyện lại hiếm khi bù đắp được lợi ích trên các trọng số sẵn có.
- Ngữ cảnh tác vụ của bạn luôn dưới 16k token. Ngưỡng nhiễu nền ở quy mô này là không đáng kể.

## Xuất bản tạo tác (Ship It)

Bài học này tạo ra `outputs/skill-diff-attention-integrator.md`. Dựa trên kiến trúc mô hình, độ dài ngữ cảnh mục tiêu, hồ sơ ảo giác và ngân sách huấn luyện, nó sẽ cung cấp một kế hoạch tích hợp hoàn chỉnh để bổ sung cơ chế chú ý vi phân vào một lượt tiền huấn luyện mới hoặc một quy trình tinh chỉnh LoRA.

## Bài tập thực hành

1. Chạy `code/main.py`. Xác minh rằng tỷ số tín hiệu trên nhiễu (SNR) được báo cáo cho cơ chế chú ý vi phân cao hơn cơ chế chú ý softmax tiêu chuẩn trên truy vấn mô phỏng. Thay đổi biên độ nhiễu và chỉ ra điểm giao nhau nơi cơ chế chú ý tiêu chuẩn trở nên mất khả năng hoạt động.

2. Tính toán độ chênh lệch số lượng tham số từ mô hình cơ sở sang DIFF V1 và sang DIFF V2 cho một mô hình phân khúc 7B (hidden=4096, heads=32, d_head=128, 32 lớp). Chỉ ra những thành phần nào tăng thêm tham số và những thành phần nào được giữ nguyên.

3. Đọc Mục 3 của bài báo DIFF V1 (arXiv:2410.05258) và Mục 2 của bài viết blog DIFF V2 từ Hugging Face. Giải thích trong hai câu lý do tại sao tầng RMSNorm trên mỗi đầu của V1 lại cần thiết, và tại sao V2 có thể loại bỏ nó mà không gây phân kỳ trong quá trình huấn luyện.

4. Thực hiện một thử nghiệm cắt bỏ (ablation study): tính toán cơ chế chú ý vi phân với $\lambda = 0$ (thuần túy nhánh softmax thứ nhất) và $\lambda = 1$ (phép trừ đầy đủ). Trên truy vấn mô phỏng, đo lường sự thay đổi của SNR khi quét giá trị $\lambda$. Xác định giá trị $\lambda$ giúp tối đa hóa tỷ số tín hiệu trên nhiễu.

5. Mở rộng mã nguồn thử nghiệm để kết hợp GQA + DIFF V2. Chọn 8 đầu KV và 32 đầu Q. Chứng minh rằng kích thước bộ nhớ đệm KV trùng khớp hoàn toàn với một mô hình GQA cơ sở có cùng cấu hình (8, 32).

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực tế |
|---|---|---|
| Chú ý vi phân (Differential attention) | "Hai softmax trừ đi nhau" | Chia Q, K thành hai nửa, tính hai bản đồ softmax, lấy bản đồ thứ nhất trừ đi bản đồ thứ hai (đã nhân với hệ số $\lambda$), sau đó nhân với ma trận V |
| Ngưỡng nhiễu nền (Noise floor) | "Phần đuôi khác không của softmax" | Trọng số kích thước $O(1/N)$ mà hàm softmax luôn phân bổ lên mọi token không liên quan, tích lũy thành $O(1)$ trên các ngữ cảnh siêu dài |
| lambda ($\lambda$) | "Hệ số co giãn của phép trừ" | Đại lượng vô hướng học được trên từng đầu được tham số hóa dưới dạng `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; có thể mang giá trị âm |
| DIFF V1 | "Phiên bản ICLR 2025" | Differential Transformer nguyên bản; giảm một nửa chiều của đầu để giữ nguyên số tham số, đòi hỏi nhân tùy chỉnh, tốc độ giải mã chậm hơn |
| DIFF V2 | "Bản sửa đổi tháng 1 năm 2026" | Nhân đôi số đầu Q trong khi giữ nguyên đầu KV; khớp hoàn toàn tốc độ giải mã cơ sở và tương thích trực tiếp với FlashAttention |
| RMSNorm trên mỗi đầu (Per-head RMSNorm) | "Bộ ổn định của V1" | Tầng chuẩn hóa bổ sung mà V1 áp dụng sau phép trừ vi phân; V2 đã loại bỏ để ngăn ngừa hiện tượng mất ổn định ở các bước huấn luyện muộn |
| Tỷ số tín hiệu trên nhiễu (SNR) | "Mức độ lãng phí của sự chú ý" | Tỷ lệ giữa trọng số chú ý tại vị trí tín hiệu thực sự so với trọng số trung bình tại các vị trí không liên quan |
| Lạc lối ở giữa (Lost in the middle) | "Dạng lỗi điển hình của ngữ cảnh dài" | Hiện tượng thực nghiệm trong đó độ chính xác truy xuất sụt giảm nghiêm trọng đối với các tài liệu nằm ở vị trí giữa của một chuỗi ngữ cảnh dài — chú ý vi phân giúp khắc phục điều này |
| Cường độ số học (Arithmetic intensity) | "Số FLOPs trên mỗi byte nạp vào" | Tỷ số mà V2 đã gia tăng trong quá trình giải mã bằng cách nhân đôi số phép tính truy vấn trên mỗi lần nạp KV; yếu tố sống còn cho tác vụ giải mã bị nghẽn bộ nhớ |

## Đọc thêm

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — bài báo gốc với lý thuyết triệt tiêu nhiễu và các bài đánh giá cắt bỏ trên ngữ cảnh dài
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — bản viết lại cho ngăn xếp sản xuất, tốc độ giải mã tương đương baseline, tương thích FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — phân tích lý thuyết giải thích vì sao phép trừ vi phân khôi phục lại cấu trúc chú ý tiền huấn luyện
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — biến thể chia sẻ tham số
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — kiến trúc Transformer cơ sở mà DIFF lấy làm mốc đối chuẩn
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — bài kiểm chuẩn ngữ cảnh dài mà cơ chế chú ý vi phân tập trung giải quyết
