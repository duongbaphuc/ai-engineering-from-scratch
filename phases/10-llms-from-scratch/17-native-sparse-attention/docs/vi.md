# Cơ chế chú ý thưa nguyên bản (Native Sparse Attention - DeepSeek NSA)

> Ở độ dài 64k token, cơ chế chú ý tiêu tốn từ 70% đến 80% độ trễ giải mã. Mọi phòng thí nghiệm mô hình mở đều có kế hoạch giải quyết nút thắt này. NSA của DeepSeek (bài báo xuất sắc nhất tại ACL 2025) là phương pháp nổi bật và thành công nhất: ba nhánh chú ý song song — các token mức thô được nén lại, các token mức mịn được giữ lại có chọn lọc, và các cửa sổ trượt cho ngữ cảnh cục bộ — được kết hợp thông qua một cổng học được (learned gate). Kiến trúc này tương thích hoàn hảo với phần cứng (thân thiện với các nhân GPU), có khả năng huấn luyện nguyên bản (hoạt động ngay từ giai đoạn tiền huấn luyện, không phải chỉ chắp vá khi suy luận), và trên các lượt giải mã 64k token, nó chạy nhanh hơn cả FlashAttention trong khi chất lượng tương đương hoặc vượt trội so với cơ chế chú ý đầy đủ. Bài học này sẽ xây dựng trọn vẹn ba nhánh từ đầu đến cuối và giải thích vì sao tính thưa này có thể lan truyền đạo hàm đầu-cuối.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7 · 12 (Bộ nhớ đệm KV, FlashAttention), Giai đoạn 7 · 15 (Các biến thể cơ chế chú ý), Giai đoạn 10 · 16 (Cơ chế chú ý vi phân)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu rõ ba nhánh chú ý của NSA và nhiệm vụ nắm bắt thông tin của từng nhánh.
- Giải thích vì sao NSA có "khả năng huấn luyện nguyên bản" (natively trainable) trong khi các phương pháp chú ý thưa trước đây chỉ áp dụng được trong thời gian suy luận.
- Tính toán lượng tính toán chú ý tiết kiệm được của NSA so với chú ý đầy đủ ở ngữ cảnh 64k dựa trên kích thước khối nén và số khối top-$k$ được chọn.
- Tự tay cài đặt sự kết hợp của ba nhánh bằng Python thư viện chuẩn (stdlib) trên một chuỗi nhân tạo ngắn và kiểm chứng hoạt động của các trọng số cổng.

## Vấn đề thực tế

Cơ chế chú ý đầy đủ (full attention) ở độ dài chuỗi $N$ tiêu tốn thời gian tính toán bậc $O(N^2)$ và dung lượng bộ nhớ đệm KV bậc $O(N)$ trên mỗi lớp. Ở độ dài 64k token, các con số về khối lượng tính toán và băng thông bộ nhớ trở thành một thảm họa thực sự. Ước tính lý thuyết được đo đạc từ bài báo NSA chỉ ra: cơ chế chú ý chiếm từ 70% đến 80% tổng độ trễ giải mã ở quy mô 64k. Mọi chỉ số hạ nguồn -- thời gian trả về token đầu tiên (TTFT), số token sinh ra mỗi giây, chi phí trên một triệu token -- đều bị chi phối hoàn toàn bởi chi phí chú ý.

Cơ chế chú ý thưa (sparse attention) là câu trả lời hiển nhiên. Các nỗ lực trước đây chia thành hai nhóm chính. Nhóm chú ý thưa theo mẫu cố định (cửa sổ trượt, bước nhảy cố định, khối cục bộ) vứt bỏ thông tin và thất bại hoàn toàn trên các tác vụ truy hồi tầm xa. Nhóm chú ý thưa lúc suy luận (tỉa bộ nhớ đệm KV, H2O, StreamingLLM) được áp dụng trên một mô hình đã tiền huấn luyện với chú ý dày đặc nên chỉ thu hồi được một phần nhỏ tốc độ tiềm năng, bởi vì mô hình chưa từng được học cách điều hướng thông tin qua cấu trúc thưa.

Cơ chế chú ý thưa nguyên bản (Native Sparse Attention - NSA, Yuan et al., DeepSeek + Đại học Bắc Kinh + Đại học Washington, Bài báo xuất sắc nhất ACL 2025, arXiv:2502.11089) thực hiện đồng thời cả hai mục tiêu: một cấu trúc thưa mà mô hình tự học được trong quá trình tiền huấn luyện, được triển khai dưới dạng thuật toán tương thích với nhân GPU để thực sự mang lại sự tiết kiệm tính toán khi suy luận. Trong vài năm tới, NSA hoặc một biến thể kế thừa trực tiếp từ nó sẽ là cơ chế chú ý mặc định trên mọi mô hình biên giới ngữ cảnh dài.

## Khái niệm cốt lõi

### Ba nhánh song song

Với mỗi truy vấn, NSA thực thi cơ chế chú ý ba lần trên ba góc nhìn khác nhau của bộ nhớ đệm KV:

1. **Nhánh nén (Compressed branch):** Các token được nhóm thành các khối có kích thước $l$ (thường là 32 hoặc 64). Mỗi khối được nén thành một token tóm tắt duy nhất thông qua một mạng MLP nhỏ học được. Truy vấn sẽ thực hiện chú ý trên các token nén này, qua đó thu được góc nhìn khái quát ở mức thô (coarse-grained) về toàn bộ chuỗi ngữ cảnh.

2. **Nhánh chọn lọc (Selected branch):** Dựa trên điểm số chú ý thu được từ nhánh nén, hệ thống xác định top-$k$ khối có mức độ liên quan cao nhất tới truy vấn hiện tại. Các token nguyên bản ở mức mịn (fine-grained - chưa nén) từ các khối đó sẽ được nạp vào và truy vấn sẽ thực hiện chú ý trên toàn bộ chúng. Hãy hình dung cơ chế chú ý của nhánh nén đóng vai trò như tín hiệu định tuyến cho quá trình chọn lọc này.

3. **Nhánh cửa sổ trượt (Sliding-window branch):** Truy vấn chú ý tới $W$ token gần nhất (thường là 512) để nắm bắt ngữ cảnh cục bộ. Nhánh này đảm nhận việc thu thập các quy luật tầm ngắn giàu cấu trúc (cú pháp ngôn ngữ, sự đồng tham chiếu cục bộ) mà hai nhánh kia có thể bỏ sót.

Đầu ra của ba nhánh được kết hợp thông qua một cổng học được trên từng vị trí:

```math
\text{out} = g_{\text{cmp}} \cdot \text{out}_{\text{cmp}} + g_{\text{sel}} \cdot \text{out}_{\text{sel}} + g_{\text{win}} \cdot \text{out}_{\text{win}}
```

Các giá trị $g_{\text{cmp}}, g_{\text{sel}}, g_{\text{win}}$ là trọng số cổng sinh ra từ một mạng MLP nhỏ áp dụng trên vector truy vấn. Chúng không bắt buộc phải có tổng bằng 1 -- chúng có thể điều chỉnh trọng số của các nhánh một cách hoàn toàn độc lập.

### Vì sao cơ chế này có "khả năng huấn luyện nguyên bản"?

Thao tác chọn lọc (chọn top-$k$ khối) là một thao tác rời rạc (discrete). Các thao tác rời rạc sẽ làm đứt gãy dòng chảy gradient. Các nghiên cứu về chú ý thưa trước đây hoặc là bỏ qua việc lan truyền ngược qua thao tác lựa chọn (làm hạn chế khả năng huấn luyện), hoặc sử dụng các phép nới lỏng liên tục (continuous relaxations) khiến mô hình không đạt được tính thưa thực sự khi suy luận.

NSA giải quyết khéo léo nghịch lý này: cơ chế chú ý của nhánh nén CHÍNH LÀ một phép chú ý mức thô có khả năng lan truyền đạo hàm trên toàn bộ chuỗi. Thao tác top-$k$ chỉ đơn thuần tái sử dụng các điểm số chú ý cao nhất từ nhánh nén để quyết định những khối mức mịn nào cần được nạp vào bộ nhớ. Gradient truyền tới các điểm số của nhánh nén thông qua chính đầu ra của nhánh nén, chứ không phải qua quyết định chọn khối. Bên trong các khối được chọn, sự chú ý trên các khóa và giá trị được chọn là phép chú ý thông thường, do đó đóng góp của chúng vào đầu ra cuối cùng cũng hoàn toàn khả vi. Bản thân việc chọn chỉ số của `top_k` không nhận gradient: nó chỉ đóng vai trò điều khiển khối dữ liệu nào được nạp từ bộ nhớ.

Đây chính là lý do vì sao NSA có thể được sử dụng trong quá trình tiền huấn luyện từ đầu đến cuối (end-to-end). Mô hình học cách cùng lúc định tuyến thông tin qua cả ba nhánh, tạo ra một mẫu thưa thực sự mang lại tốc độ tăng tốc như cam kết trong thời gian suy luận.

### Nhân GPU tương thích phần cứng (Hardware-Aligned Kernel)

Nhân tính toán của NSA được thiết kế riêng cho hệ thống phân cấp bộ nhớ của GPU hiện đại. Nhân sẽ nạp các truy vấn theo từng nhóm GQA (vòng lặp ngoài), lấy các khối KV thưa tương ứng cho từng nhóm (vòng lặp trong), và thực thi cơ chế chú ý trực tiếp trên bộ nhớ SRAM. Bởi vì mỗi nhóm truy vấn đều nhìn thấy cùng các khối được chọn (việc chọn lọc được thực hiện theo nhóm truy vấn chứ không phải theo từng đầu truy vấn đơn lẻ), chi phí nạp KV được chia sẻ đồng đều xuyên suốt nhóm. Cường độ số học được duy trì ở mức rất cao.

Bài báo công bố các nhân viết bằng Triton chạy nhanh hơn tới $9\times$ so với FlashAttention trên các tác vụ giải mã 64k token, và tỷ lệ tăng tốc này càng mở rộng khi độ dài chuỗi tăng lên. Cả hai nhân truyền xuôi (forward) và truyền ngược (backward) đều được cung cấp đầy đủ.

### Dự trù ngân sách tính toán

Gọi $N$ là độ dài chuỗi, $l$ là kích thước khối nén, $k$ là số khối top-$k$ được chọn, $w$ là kích thước cửa sổ trượt, và $b$ là kích thước khối chọn lọc (thường bằng $l$).

- Nhánh nén: $O(N/l)$ khóa cho mỗi truy vấn, tức $O(N \cdot N / l)$ cho toàn chuỗi.
- Nhánh chọn lọc: $O(k \cdot b)$ khóa cho mỗi truy vấn, tức $O(N \cdot k \cdot b)$.
- Nhánh cửa sổ trượt: $O(w)$ khóa cho mỗi truy vấn, tức $O(N \cdot w)$.

Tổng chi phí: $O(N \cdot (N/l + k \cdot b + w))$.

Với $N = 64\text{k}, l = 64, k = 16, b = 64, w = 512$: chi phí trên mỗi truy vấn là $1000 + 1024 + 512 = 2536$ khóa. Cơ chế chú ý đầy đủ cần $64000$ khóa. Giảm thiểu tới 25 lần chi phí tính toán.

Với $N = 128\text{k}, l = 64, k = 16, b = 64, w = 512$: chi phí trên mỗi truy vấn là $2000 + 1024 + 512 = 3536$ khóa. Chú ý đầy đủ cần $128000$ khóa. Giảm thiểu tới 36 lần. Lợi ích tăng trưởng vượt bậc theo độ dài chuỗi, và đó chính là mục tiêu cốt lõi.

### Bảng so sánh phương pháp

| Phương pháp | Khả vi | Tăng tốc suy luận thực tế | Truy hồi tầm xa |
|---|---|---|---|
| Chỉ dùng cửa sổ trượt | Có | Có | Thất bại |
| Bước nhảy / thưa theo khối | Có | Có | Một phần |
| Tỉa KV (H2O, StreamingLLM) | Không (lúc suy luận) | Có | Một phần |
| MoBA (Moonshot) | Một phần | Có | Tốt |
| NSA | Có (nguyên bản) | Có (9$\times$ ở 64k) | Tương đương chú ý đầy đủ |

MoBA (Moonshot, arXiv:2502.13189) được công bố đồng thời và áp dụng triết lý "ba tốt hơn một" tương tự, mang nguyên lý MoE vào các khối chú ý. NSA và MoBA là hai kiến trúc mang tính định hình cần phải nắm vững cho giai đoạn tiền huấn luyện ngữ cảnh dài năm 2026.

````figure
sliding-window-attention
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` triển khai ba nhánh trên một chuỗi nhân tạo ngắn và thể hiện:

- Mạng MLP nén (bản mô phỏng sử dụng phép gộp trung bình - mean-pool để đảm bảo tính rõ ràng về mặt sư phạm; NSA thực tế dùng một mạng MLP học được).
- Lựa chọn khối top-$k$ được dẫn dắt bởi điểm số từ nhánh nén.
- Cơ chế chú ý cửa sổ trượt trên $w$ token gần nhất.
- Phép kết hợp có cổng.
- Bảng in hạch toán chi phí tính toán so sánh với chú ý đầy đủ.

### Bước 1: Nén token thành các khối

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Bước 2: Chú ý ở nhánh nén

Thực hiện chú ý softmax giữa truy vấn và các khóa đã được nén. Các điểm số chú ý của nhánh nén đồng thời đóng vai trò là tín hiệu để chọn ra các khối top-$k$.

### Bước 3: Lựa chọn khối Top-k

Chọn các chỉ số của $k$ khối nén có điểm số cao nhất. Nạp các token nguyên bản chưa nén từ các khối đó và thực hiện chú ý trên chúng.

### Bước 4: Chú ý cửa sổ trượt

Lấy $w$ token gần nhất và thực hiện phép chú ý tiêu chuẩn trên chúng.

### Bước 5: Cổng kết hợp

Một mạng MLP nhỏ nhận đầu vào là truy vấn và sinh ra ba trọng số cổng. Đầu ra cuối cùng là tổng có trọng số của kết quả từ ba nhánh.

### Bước 6: Thống kê số lượng phép tính

In ra số lượng khóa được chú ý trên mỗi truy vấn cho từng nhánh và tổng cộng. So sánh với $N$ (chú ý đầy đủ). Trên chuỗi nhân tạo 1024 token với $l = 32, k = 4, w = 128$, NSA chỉ duyệt qua $32 + 128 + 128 = 288$ khóa trên mỗi truy vấn so với 1024 khóa của chú ý đầy đủ -- giảm $3,5\times$.

## Sử dụng thực tế (Use It)

NSA đang được triển khai trực tiếp trong đường ống tiền huấn luyện ngữ cảnh dài của chính DeepSeek. Tình trạng tích hợp trong các ngăn xếp suy luận công khai tính đến tháng 4 năm 2026:

- **Hệ thống nội bộ DeepSeek:** Hỗ trợ nguyên bản; các trọng số công bố sử dụng NSA hoặc phiên bản kế nhiệm DSA (Deepseek Sparse Attention).
- **vLLM:** Đang phát triển tính năng hỗ trợ NSA thử nghiệm cho các trọng số dòng DeepSeek-V3.x.
- **SGLang:** Đã công bố các kết quả benchmark NSA; lộ trình sản xuất đi theo vLLM.
- **llama.cpp / CPU:** Chưa hỗ trợ; chi phí điều phối nhân phân rã không đem lại hiệu quả ở mức thông lượng của CPU.

Khi nào nên áp dụng NSA:

- Các lượt chạy tiền huấn luyện hoặc huấn luyện tiếp nối (continued-training) hướng tới ngữ cảnh trên 64k với ngân sách tính toán lớn.
- Suy luận các điểm kiểm tra ngữ cảnh dài chính thức của DeepSeek (các trọng số vốn được thiết kế nguyên bản cho NSA).

Khi nào không nên sử dụng:

- Phục vụ một mô hình dày đặc đã tiền huấn luyện sẵn có. Bạn không thể chắp vá NSA vào nếu không huấn luyện tiếp nối.
- Ngữ cảnh dưới 16k token. Chi phí phụ trợ của việc điều phối ba nhánh sẽ lấn át lợi ích tiết kiệm tính toán.
- Hội thoại tương tác với batch size 1. Việc giải mã nhạy cảm với độ trễ chỉ được hưởng lợi khi ngữ cảnh đủ dài.

## Xuất bản tạo tác (Ship It)

Bài học này tạo ra `outputs/skill-nsa-integrator.md`. Dựa trên bản đặc tả của một lượt tiền huấn luyện ngữ cảnh dài, nó sẽ tạo ra một kế hoạch tích hợp NSA toàn diện: kích thước khối nén, số khối top-$k$, kích thước cửa sổ trượt, độ rộng MLP cổng, lựa chọn nhân GPU, và các bài đánh giá ngữ cảnh dài cụ thể để biện minh cho sự thay đổi kiến trúc.

## Bài tập thực hành

1. Chạy `code/main.py` trên một chuỗi nhân tạo 1024 token. Quét ba bộ tham số `(l, k, w)` định sẵn và in ra số lượng phép tính. Xác định bộ tham số đạt số lượng khóa trên mỗi truy vấn thấp nhất trong khi vẫn duy trì độ truy hồi 95% so với chú ý đầy đủ trên bài kiểm tra cây kim trong đống rơm.

2. Thay thế bộ nén trung bình (mean-pool) bằng một mạng MLP nhỏ học được (2 lớp, chiều ẩn 32). Huấn luyện nó trên một tác vụ nhân tạo nơi tín hiệu là giá trị trung bình của khối. Đo lường khoảng cách về độ hỗn loạn so với mốc cơ sở gộp trung bình trên tập dữ liệu kiểm tra độc lập.

3. Triển khai mạng MLP cho cổng kết hợp. Nó nhận truy vấn làm đầu vào và xuất ra ba đại lượng vô hướng. Chứng minh rằng cổng hoạt động hợp lý: trọng số phân bổ gần như đồng đều trên các truy vấn ngẫu nhiên, nhưng dồn trọng số lớn vào nhánh chọn lọc khi truy vấn kích hoạt một khối nằm ở vị trí rất xa trong quá khứ.

4. Tính toán ngân sách bộ nhớ đệm KV cho một mô hình 70B tích hợp NSA ở ngữ cảnh 128k. Số đầu KV là 8, chiều của đầu là 128, định dạng BF16. So sánh với chú ý đầy đủ và với MLA (Giai đoạn 10 · 14 đã chỉ ra các con số của MLA). Xác định độ dài chuỗi nơi bộ nhớ đệm KV nhánh mịn của NSA bằng với chú ý đầy đủ.

5. Đọc Mục 4 của bài báo NSA (arXiv:2502.11089) và giải thích trong ba câu lý do vì sao điểm số chú ý của nhánh nén lại được tái sử dụng trực tiếp cho việc chọn lựa top-$k$ thay vì tính toán một điểm số định tuyến riêng biệt. Liên hệ câu trả lời với dòng chảy của gradient.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực tế |
|---|---|---|
| Nhánh nén (Compressed branch) | "Góc nhìn mức thô" | Cơ chế chú ý trên các khóa đã được lấy trung bình theo khối, cung cấp ngữ cảnh toàn cục với $O(N/l)$ khóa cho mỗi truy vấn |
| Nhánh chọn lọc (Selected branch) | "Các khối top-k" | Cơ chế chú ý mức mịn trên $k$ khối có điểm số cao nhất thu được từ nhánh nén |
| Cửa sổ trượt (Sliding window) | "Ngữ cảnh cục bộ" | Cơ chế chú ý trên $W$ token gần nhất phía trước để nắm bắt các mối liên hệ tầm ngắn |
| Khả năng huấn luyện nguyên bản (Native trainability) | "Tiền huấn luyện với cấu trúc thưa bật sẵn" | Mẫu chú ý thưa được mô hình tự học ngay trong quá trình tiền huấn luyện chứ không phải bị ép buộc lúc suy luận |
| Kích thước khối nén l (Compression block size) | "Độ lớn nhóm cho góc nhìn thô" | Số lượng token được gộp lại thành một token tóm tắt; thường từ 32 đến 64 |
| Top-k | "Số khối được giữ lại" | Số lượng khối nén mà các token mức mịn chưa nén của chúng sẽ được nạp vào để chú ý; điển hình là 16 |
| Cửa sổ trượt W | "Bán kính chú ý cục bộ" | Thường là 512; ngắn hơn sẽ làm hỏng tính mạch lạc cục bộ, dài hơn sẽ lãng phí năng lực tính toán |
| Cổng phân nhánh (Branch gate) | "Cách hòa trộn ba nhánh" | Đầu ra từ mạng MLP trên từng vị trí quyết định trọng số đóng góp của ba nhánh chú ý |
| Tương thích phần cứng (Hardware alignment) | "Tính thưa thân thiện với nhân GPU" | Mẫu chú ý thưa được thiết kế sao cho nhân GPU thực tế đạt được tốc độ tăng tốc trên lý thuyết |
| DSA | "Kiến trúc kế nhiệm của NSA" | Deepseek Sparse Attention, kiến trúc chú ý tiếp nối NSA trong lộ trình phát triển của DeepSeek |

## Đọc thêm

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — bài báo gốc đoạt giải
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — họ kiến trúc mà NSA hướng tới tối ưu hóa
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — công trình đồng thời áp dụng cơ chế chú ý theo khối kiểu MoE
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — nguồn gốc của cơ chế cửa sổ trượt
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — phương pháp chú ý thưa lúc suy luận mà NSA đã vượt qua
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — mốc cơ sở chú ý đầy đủ mà nhân NSA đánh bại ở ngữ cảnh 64k
