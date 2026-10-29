# Chuẩn và Khoảng cách (Norms and Distances)

> Hàm khoảng cách định nghĩa bản chất của khái niệm "tương đồng". Chọn sai khoảng cách, mọi thứ ở tầng hạ nguồn đều sẽ đổ vỡ.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## Learning Objectives

- Tự xây dựng từ đầu các hàm khoảng cách L1, L2, Cosine, Mahalanobis, Jaccard và Edit distance
- Lựa chọn độ đo khoảng cách phù hợp cho một bài toán học máy cụ thể và giải thích tại sao các phương án khác thất bại
- Kết nối các chuẩn L1 và L2 với điều chuẩn LASSO và Ridge cùng các miền ràng buộc hình học của chúng
- Chứng minh cách cùng một tập dữ liệu sẽ tạo ra các điểm lân cận gần nhất (nearest neighbors) khác nhau dưới các độ đo khác nhau

## The Problem (Vấn đề)

Bạn có hai vector. Có thể chúng là các word embedding. Có thể là hồ sơ người dùng. Hoặc là mảng pixel hình ảnh. Bạn cần biết: Chúng gần nhau tới mức nào?

Câu trả lời phụ thuộc hoàn toàn vào hàm khoảng cách bạn chọn. Hai điểm dữ liệu có thể là hàng xóm gần nhất (nearest neighbors) dưới độ đo này nhưng lại cách xa nhau thăm thẳm dưới độ đo khác. Bộ phân loại KNN, hệ thống gợi ý, cơ sở dữ liệu vector, thuật toán phân cụm, hàm mất mát của bạn -- tất cả đều phụ thuộc vào sự lựa chọn này. Chọn sai và mô hình của bạn sẽ tối ưu hóa một mục tiêu hoàn toàn sai lệch.

Không có một độ đo khoảng cách nào là hoàn hảo cho mọi trường hợp. L2 hoạt động tốt cho dữ liệu không gian. Độ tương đồng Cosine thống trị xử lý ngôn ngữ tự nhiên (NLP). Jaccard xử lý dữ liệu tập hợp. Edit distance xử lý chuỗi ký tự. Mahalanobis tính đến sự tương quan giữa các đặc trưng. Wasserstein di chuyển khối lượng xác suất. Mỗi độ đo mã hóa một giả định khác nhau về ý nghĩa của khái niệm "tương đồng".

Bài học này sẽ xây dựng mọi hàm khoảng cách quan trọng từ đầu, chỉ ra thời điểm thích hợp để sử dụng từng loại, và chứng minh trực quan cách cùng một dữ liệu sinh ra các láng giềng gần nhất hoàn toàn khác nhau tùy thuộc vào thước đo bạn sử dụng.

## The Concept (Khái niệm)

### Chuẩn (Norms): Đo độ lớn của vector

Chuẩn (norm) đo lường "kích thước" hay độ lớn của một vector. Mọi hàm khoảng cách giữa hai vector đều có thể viết dưới dạng chuẩn của hiệu giữa chúng: $d(a, b) = \lVert a - b \rVert$. Vì vậy, hiểu về chuẩn chính là hiểu về khoảng cách.

### Chuẩn L1 (Khoảng cách Manhattan)

Chuẩn L1 tính tổng giá trị tuyệt đối của toàn bộ các thành phần:

```math
\lVert x \rVert_1 = |x_1| + |x_2| + \cdots + |x_n|
```

Nó được gọi là khoảng cách Manhattan vì nó đo quãng đường bạn đi bộ trên mạng lưới đường phố của một thành phố hình ô cờ, nơi bạn chỉ có thể di chuyển dọc theo các trục đường chính chứ không thể đi đường chéo xuyên qua các tòa nhà.

```
Điểm A = (1, 1)
Điểm B = (4, 5)

Khoảng cách L1 = |4-1| + |5-1| = 3 + 4 = 7

Trên lưới ô vuông, bạn đi bộ 3 dãy nhà về phía đông và 4 dãy nhà về phía bắc.
```

Khi nào nên dùng L1:
- Dữ liệu thưa nhiều chiều (đặc trưng văn bản, mã hóa one-hot)
- Khi bạn cần tính bền vững trước các điểm ngoại lai (outliers) -- một sự khác biệt đơn lẻ quá lớn sẽ không lấn át toàn bộ
- Bài toán lựa chọn đặc trưng (điều chuẩn L1 thúc đẩy tính thưa thớt)

Mối liên hệ với điều chuẩn L1 (Lasso): Việc cộng thêm $\lVert w \rVert_1$ vào hàm mất mát sẽ phạt tổng giá trị tuyệt đối của các trọng số. Điều này ép các trọng số nhỏ về chính xác bằng 0, thực hiện việc tự động lựa chọn đặc trưng. Khoản phạt L1 tạo ra các miền ràng buộc hình thoi (kim cương) trong không gian trọng số, và các đỉnh nhọn của hình thoi nằm trên các trục tọa độ nơi một số trọng số bị triệt tiêu về 0.

Mối liên hệ với hàm mất mát: Sai số tuyệt đối trung bình (MAE) là khoảng cách L1 trung bình giữa dự đoán và nhãn thực tế. Nó phạt mọi sai số một cách tuyến tính, giúp nó bền vững hơn trước ngoại lai so với MSE.

### Chuẩn L2 (Khoảng cách Euclid)

Chuẩn L2 là khoảng cách đường thẳng thông thường: Căn bậc hai của tổng bình phương các thành phần.

```math
\lVert x \rVert_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}
```

Đây là khoảng cách bạn đã học trong hình học phổ thông -- định lý Pytago trong không gian $n$ chiều.

```
Điểm A = (1, 1)
Điểm B = (4, 5)

Khoảng cách L2 = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

Đường thẳng nối trực tiếp, cắt chéo qua lưới ô cờ.
```

Khi nào nên dùng L2:
- Dữ liệu liên tục có số chiều thấp đến trung bình
- Khi thang đo giữa các đặc trưng là tương đương nhau
- Khoảng cách vật lý (dữ liệu không gian, số đọc cảm biến)
- Đo độ tương đồng hình ảnh ở mức điểm ảnh (pixel)

Mối liên hệ với điều chuẩn L2 (Ridge): Việc cộng thêm $\lVert w \rVert_2^2$ vào hàm mất mát sẽ phạt các trọng số lớn. Khác với L1, nó không ép trọng số về chính xác bằng 0. Nó co các trọng số về phía 0 một cách tỷ lệ. Khoản phạt L2 tạo ra các miền ràng buộc hình tròn/hình cầu, do đó không có các đỉnh nhọn trên các trục tọa độ. Trọng số sẽ trở nên rất nhỏ nhưng hiếm khi bằng 0 tuyệt đối.

Mối liên hệ với hàm mất mát: Sai số toàn phương trung bình (MSE) là trung bình bình phương các khoảng cách L2. Việc bình phương sẽ phạt các sai số lớn nặng nề hơn nhiều so với các sai số nhỏ.

```math
\begin{aligned}
\text{MAE (L1 loss):} \quad & |y - \hat{y}| && \text{Phạt tuyến tính. Bền vững trước ngoại lai.} \\
\text{MSE (L2 loss):} \quad & (y - \hat{y})^2 && \text{Phạt bậc hai. Rất nhạy cảm với ngoại lai.}
\end{aligned}
```

### Họ chuẩn Lp tổng quát

L1 và L2 là các trường hợp đặc biệt của chuẩn Lp:

```math
\lVert x \rVert_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^{1/p}
```

Các giá trị khác nhau của $p$ tạo ra các "hình cầu đơn vị" (unit balls - tập hợp các điểm có khoảng cách tới gốc tọa độ $\le 1$) với hình dạng khác nhau:

```
p=1:    Hình thoi (Diamond)      (các góc nhọn nằm trên trục)
p=2:    Hình tròn / hình cầu     (hình cầu tròn thông thường)
p=3:    Hình siêu elip           (hình vuông bo tròn góc)
p=inf:  Hình vuông / siêu lập phương (các cạnh phẳng song song với trục)
```

### Chuẩn L-vô cùng (Khoảng cách Chebyshev)

Khi $p$ tiến tới vô cùng, chuẩn Lp hội tụ về giá trị tuyệt đối lớn nhất trong các thành phần:

```math
\lVert x \rVert_\infty = \max(|x_1|, |x_2|, \ldots, |x_n|)
```

Khoảng cách giữa hai điểm được quyết định bởi chiều duy nhất mà tại đó chúng khác biệt nhiều nhất. Toàn bộ các chiều khác đều bị bỏ qua.

```math
\begin{aligned}
\text{Điểm } A &= (1, 1) \\
\text{Điểm } B &= (4, 5) \\[6pt]
L_\infty \text{ khoảng cách} &= \max(|4-1|, |5-1|) = \max(3, 4) = 4
\end{aligned}
```

Khi nào nên dùng L-vô cùng:
- Khi độ lệch trong trường hợp xấu nhất (worst-case) ở bất kỳ chiều đơn lẻ nào là yếu tố quan trọng nhất
- Bàn cờ (quân Vua trong cờ vua di chuyển theo khoảng cách L-vô cùng: một bước đi theo bất kỳ hướng nào đều tốn chi phí 1)
- Dung sai chế tạo trong công nghiệp (mọi kích thước đều phải nằm trong tiêu chuẩn cho phép)

### Độ tương đồng Cosine và Khoảng cách Cosine

Độ tương đồng Cosine (Cosine similarity) đo góc giữa hai vector và bỏ qua độ dài (độ lớn) của chúng.

```math
\text{cos\_sim}(a, b) = \frac{a \cdot b}{\lVert a \rVert_2 \, \lVert b \rVert_2}
```

Nó có giá trị trong khoảng từ -1 (hướng ngược nhau hoàn toàn) đến +1 (cùng hướng hoàn toàn). Hai vector trực giao có cosine similarity bằng 0.

Khoảng cách Cosine (Cosine distance) chuyển đổi nó thành khoảng cách: `cosine_distance = 1 - cosine_similarity`, có giá trị từ 0 (cùng hướng) đến 2 (ngược hướng).

```math
\begin{aligned}
& a = (1, 0) \qquad b = (1, 1) \\[6pt]
& \text{cos\_sim} = \frac{1 \cdot 1 + 0 \cdot 1}{1 \cdot \sqrt{2}} = \frac{1}{\sqrt{2}} = 0.707 \\
& \text{cos\_dist} = 1 - 0.707 = 0.293
\end{aligned}
```

Vì sao Cosine thống trị trong NLP và Embedding: Trong văn bản, độ dài tài liệu không nên ảnh hưởng đến sự tương đồng về mặt nội dung. Một bài viết về loài mèo dài gấp đôi một bài viết khác cũng về loài mèo vẫn phải được xem là "tương đồng". Cosine similarity bỏ qua độ lớn (độ dài) và chỉ quan tâm đến phương hướng. Hai tài liệu có cùng phân phối từ vựng nhưng độ dài khác nhau sẽ cùng trỏ về một hướng và đạt độ tương đồng cosine bằng 1.0.

Khi nào nên dùng Cosine similarity:
- Đo độ tương đồng văn bản (vector TF-IDF, word embedding, sentence embedding)
- Bất kỳ lĩnh vực nào mà độ lớn là nhiễu còn hướng là tín hiệu
- Hệ thống gợi ý (vector sở thích người dùng)
- Tìm kiếm embedding (các vector database hầu như luôn dùng cosine hoặc dot product)

### So sánh Tích vô hướng (Dot Product) và Độ tương đồng Cosine

Tích vô hướng giữa hai vector là:

```math
\begin{aligned}
a \cdot b &= a_1 b_1 + a_2 b_2 + \cdots + a_n b_n \\
&= \lVert a \rVert \, \lVert b \rVert \cos(\text{góc})
\end{aligned}
```

Cosine similarity thực chất là tích vô hướng được chuẩn hóa bởi độ lớn của cả hai vector. Khi cả hai vector đều đã được chuẩn hóa về vector đơn vị (độ lớn = 1), tích vô hướng và cosine similarity là giống hệt nhau.

```math
\text{Nếu } \lVert a \rVert = 1 \text{ và } \lVert b \rVert = 1: \quad a \cdot b = \cos(\text{góc giữa } a \text{ và } b)
```

Điểm khác biệt: Tích vô hướng bao hàm cả thông tin về độ lớn. Một vector có độ lớn lớn hơn sẽ nhận điểm tích vô hướng cao hơn. Điều này rất có ích trong một số hệ thống truy xuất (retrieval systems) nơi bạn muốn các mục "phổ biến" xếp hạng cao hơn. Độ lớn khi đó đóng vai trò như một tín hiệu ngầm định về chất lượng hoặc tầm quan trọng.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Cả hai đều đồng thuận về hướng, nhưng tích vô hướng còn phản ánh độ lớn.
```

Trong thực tế:
- Dùng cosine similarity khi bạn muốn so sánh thuần túy về hướng
- Dùng dot product khi độ lớn mang thông tin có ý nghĩa
- Rất nhiều vector database (Pinecone, Weaviate, Qdrant) cho phép bạn lựa chọn giữa hai cách này
- Nếu embedding của bạn đã được chuẩn hóa theo chuẩn L2, kết quả là hoàn toàn như nhau

### Khoảng cách Mahalanobis

Khoảng cách Euclid đối xử bình đẳng với mọi chiều tọa độ. Nhưng nếu các đặc trưng của bạn có sự tương quan với nhau hoặc có thang đo khác nhau, khoảng cách L2 sẽ đưa ra kết quả sai lệch.

Khoảng cách Mahalanobis tính đến cấu trúc hiệp phương sai của dữ liệu:

```math
d_M(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)}
```

trong đó $S$ là ma trận hiệp phương sai của tập dữ liệu.

Ý nghĩa trực quan: Khoảng cách Mahalanobis đầu tiên sẽ khử tương quan và chuẩn hóa dữ liệu (quá trình whitening), sau đó tính khoảng cách L2 trong không gian đã biến đổi đó. Nếu $S$ là ma trận đơn vị (các đặc trưng không tương quan và có phương sai bằng 1), khoảng cách Mahalanobis trở về đúng khoảng cách Euclid.

```
Ví dụ: Chiều cao và cân nặng có tương quan với nhau.
Một người cao 1m88 nặng 82kg là bình thường.
Một người cao 1m52 nặng 82kg là điều bất thường.

Khoảng cách Euclid có thể coi cả hai người đều cách xa giá trị trung bình như nhau.
Khoảng cách Mahalanobis xác định chính xác trường hợp thứ hai là ngoại lai
bởi vì nó tính đến sự tương quan giữa chiều cao và cân nặng.
```

Khi nào nên dùng Mahalanobis:
- Phát hiện điểm dị biệt / ngoại lai (các điểm có khoảng cách Mahalanobis lớn tới tâm dữ liệu là ngoại lai)
- Phân loại khi các đặc trưng có thang đo và sự tương quan khác nhau
- Khi bạn có đủ dữ liệu để ước lượng một ma trận hiệp phương sai đáng tin cậy
- Kiểm soát chất lượng trong sản xuất (giám sát quy trình đa biến)

### Độ tương đồng Jaccard (dành cho tập hợp)

Độ tương đồng Jaccard đo mức độ trùng lặp giữa hai tập hợp:

```math
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
```

Nó có giá trị từ 0 (không trùng lặp) đến 1 (hai tập hợp giống hệt nhau). Khoảng cách Jaccard = 1 - Jaccard similarity.

```
A = {mèo, chó, cá}
B = {mèo, chim, cá, rắn}

Giao (Intersection) = {mèo, cá}               kích thước = 2
Hợp (Union) = {mèo, chó, cá, chim, rắn}       kích thước = 5

Độ tương đồng Jaccard = 2/5 = 0.4
Khoảng cách Jaccard = 0.6
```

Khi nào nên dùng Jaccard:
- So sánh các tập hợp nhãn (tags), danh mục hoặc thuộc tính
- Đo độ tương đồng văn bản dựa trên sự xuất hiện của từ (không xét tần suất)
- Phát hiện tài liệu gần trùng lặp (thuật toán MinHash xấp xỉ Jaccard)
- So sánh các vector đặc trưng nhị phân (dữ liệu có mặt/vắng mặt)
- Đánh giá mô hình phân vùng hình ảnh (Intersection over Union - IoU chính là Jaccard)

### Edit Distance (Khoảng cách Levenshtein)

Edit distance đếm số lượng thao tác tối thiểu trên từng ký tự để biến chuỗi này thành chuỗi kia. Các thao tác hợp lệ gồm: chèn (insert), xóa (delete) hoặc thay thế (substitute).

```
"kitten" -> "sitting"

kitten -> sitten  (thay thế k -> s)
sitten -> sittin  (thay thế e -> i)
sittin -> sitting (chèn thêm g)

Edit distance = 3
```

Khoảng cách này được tính toán bằng quy hoạch động (dynamic programming). Điền vào một bảng ma trận trong đó phần tử $(i, j)$ là khoảng cách sửa đổi giữa $i$ ký tự đầu của chuỗi A và $j$ ký tự đầu của chuỗi B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Khi nào nên dùng Edit distance:
- Kiểm tra và sửa lỗi chính tả
- Gióng hàng trình tự DNA (kèm chi phí thao tác có trọng số)
- Khớp chuỗi mờ (fuzzy string matching)
- Khử trùng lặp dữ liệu văn bản thô

### Phân kỳ KL (KL Divergence - không phải khoảng cách, nhưng được dùng như khoảng cách)

Phân kỳ Kullback-Leibler (KL divergence) đo lường mức độ khác biệt của một phân phối xác suất so với một phân phối tham chiếu. Đã học ở Lesson 09, nhưng cần nhắc lại ở đây vì nhiều người quen dùng nó như một "khoảng cách".

```math
D_{KL}(P \,\|\, Q) = \sum_x p(x) \log\frac{p(x)}{q(x)}
```

Tính chất quan trọng: KL divergence KHÔNG có tính đối xứng:

```math
D_{KL}(P \,\|\, Q) \ne D_{KL}(Q \,\|\, P)
```

Điều này có nghĩa là nó không thỏa mãn định nghĩa cơ bản của một độ đo khoảng cách (metric). Nó cũng không thỏa mãn bất đẳng thức tam giác. Đây là một độ phân kỳ (divergence), không phải là khoảng cách.

Forward KL $(D_{KL}(P \Vert Q))$ có tính chất "tìm kiếm trung bình" (mean-seeking): $Q$ cố gắng bao phủ toàn bộ các đỉnh (modes) của $P$.
Reverse KL $(D_{KL}(Q \Vert P))$ có tính chất "tìm kiếm đỉnh" (mode-seeking): $Q$ tập trung bám chặt vào một đỉnh duy nhất của $P$.

Nơi bạn thường gặp KL divergence:
- VAEs (thành phần KL trong hàm mục tiêu ELBO ép phân phối tiềm ẩn tiến về phân phối tiên nghiệm prior)
- Chắt lọc tri thức (Knowledge distillation - mô hình học sinh cố gắng khớp với phân phối của mô hình giáo viên)
- RLHF (khoản phạt KL giữ mô hình sau khi tinh chỉnh không đi quá xa mô hình gốc ban đầu)
- Các phương pháp Policy Gradient (ràng buộc bước cập nhật chính sách)

### Khoảng cách Wasserstein (Earth Mover's Distance)

Khoảng cách Wasserstein đo lường "công" tối thiểu cần thiết để biến đổi một phân phối xác suất thành một phân phối khác. Hãy hình dung: Nếu một phân phối là một đống đất và phân phối kia là một cái hố, bạn cần phải xúc bao nhiêu đất và vận chuyển đi bao xa?

```math
W(P, Q) = \inf_{\gamma} \mathbb{E}[d(x, y)] \quad \text{(trên toàn bộ các phương án vận chuyển } \gamma\text{)}
```

Đối với phân phối 1 chiều, công thức rút gọn thành tích phân của hiệu tuyệt đối giữa các hàm phân phối tích lũy (CDF):

```math
W_1(P, Q) = \int |\text{CDF}_P(x) - \text{CDF}_Q(x)| \, dx
```

Vì sao Wasserstein rất quan trọng:
- Đây là một metric thực thụ (đối xứng, thỏa mãn bất đẳng thức tam giác)
- Nó cung cấp gradient mượt mà ngay cả khi hai phân phối hoàn toàn không giao nhau (trong khi KL divergence sẽ tiến tới vô cùng)
- Tính chất này là mấu chốt giúp Wasserstein GAN (WGAN) giải quyết dứt điểm sự bất ổn định khi huấn luyện của mạng GAN truyền thống

```
Hai phân phối không hề giao nhau:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: vô cùng (log của 0)
Wasserstein: 4 (cần dịch chuyển toàn bộ khối lượng đi 4 bước)

Wasserstein cung cấp gradient có ý nghĩa để tối ưu. KL thì không.
```

Khi nào nên dùng Wasserstein:
- Huấn luyện GAN (WGAN, WGAN-GP)
- So sánh các phân phối không có vùng trùng lặp
- Các bài toán vận chuyển tối ưu (Optimal transport)
- Truy vấn hình ảnh (so sánh biểu đồ màu sắc - color histograms)

### Bảng tổng hợp: Bài toán nào cần khoảng cách nào

| Bài toán | Độ đo tối ưu | Lý do |
|------|--------------|-----|
| Độ tương đồng văn bản | Cosine | Độ lớn là nhiễu, hướng vector mới mang ngữ nghĩa |
| So sánh pixel hình ảnh | L2 | Mối quan hệ không gian quan trọng, thang đo đặc trưng tương đồng |
| Đặc trưng nhiều chiều thưa | L1 | Bền vững, không phóng đại những khác biệt lớn hiếm gặp |
| Độ trùng lặp tập hợp (tags, danh mục) | Jaccard | Dữ liệu tự nhiên ở dạng tập hợp, không phải vector số học |
| So khớp chuỗi | Edit distance | Các thao tác khớp trực tiếp với logic chỉnh sửa của con người |
| Phát hiện điểm ngoại lai | Mahalanobis | Tính đến sự tương quan và thang đo giữa các đặc trưng |
| So sánh phân phối xác suất | KL divergence | Đo lượng thông tin bị mất khi dùng $Q$ thay vì $P$ |
| Huấn luyện mô hình GAN | Wasserstein | Cho gradient mượt mà ngay cả khi phân phối không giao nhau |
| Tìm kiếm Embedding (vector DB) | Cosine hoặc Dot product | Các embedding được huấn luyện để mã hóa ngữ nghĩa theo hướng |
| Hệ thống gợi ý | Dot product | Độ lớn vector có thể mã hóa độ phổ biến hoặc độ tin cậy |
| Chuỗi gen DNA | Weighted edit distance | Chi phí đột biến thay thế khác nhau giữa các cặp nucleotide |
| Đảm bảo chất lượng sản xuất (QC) | L-vô cùng | Sai lệch trong trường hợp xấu nhất ở bất kỳ chiều nào cũng cần kiểm soát |

### Mối liên hệ với Hàm mất mát (Loss Functions)

Hàm mất mát thực chất là các hàm khoảng cách được áp dụng giữa giá trị dự đoán và nhãn thực tế.

```
Hàm mất mát         Khoảng cách sử dụng    Hành vi
MSE                 L2 bình phương         Phạt rất nặng các sai số lớn
MAE                 L1                     Phạt mọi sai số một cách đồng đều
Huber loss          L1 cho sai số lớn,     Kết hợp ưu điểm của cả hai: bền vững trước
                    L2 cho sai số nhỏ      ngoại lai, gradient mượt mà gần điểm 0
Cross-entropy       KL divergence          Đo mức độ sai lệch phân phối xác suất
Hinge loss          max(0, margin - d)     Chỉ phạt khi vi phạm vùng biên an toàn (margin)
Triplet loss        L2 (thông thường)      Kéo mẫu dương lại gần, đẩy mẫu âm ra xa
Contrastive loss    L2                     Kéo cặp tương đồng lại gần, đẩy cặp khác biệt ra ngoài margin
```

### Mối liên hệ với Kỹ thuật điều chuẩn (Regularization)

Điều chuẩn bổ sung thêm một khoản phạt chuẩn của trọng số vào hàm mất mát.

```
Điều chuẩn L1 (Lasso):        loss + lambda * ||w||_1
  -> Trọng số thưa thớt. Một số trọng số trở về chính xác bằng 0.
  -> Tự động lựa chọn đặc trưng.
  -> Nghiệm có góc nhọn (không khả vi tại điểm 0).

Điều chuẩn L2 (Ridge):        loss + lambda * ||w||_2^2
  -> Trọng số nhỏ. Mọi trọng số đều bị co dần về phía 0.
  -> Không loại bỏ đặc trưng (không có trọng số nào về chính xác 0).
  -> Nghiệm mượt mà tại mọi điểm.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Kết hợp tính thưa của L1 với tính ổn định của L2.
  -> Các nhóm đặc trưng có tương quan sẽ được giữ lại hoặc loại bỏ cùng nhau.
```

Vì sao L1 tạo ra tính thưa mà L2 thì không: Hãy hình dung miền ràng buộc trong không gian trọng số 2 chiều. L1 là một hình thoi, L2 là một hình tròn. Các đường đồng mức của hàm loss (hình elip) thường tiếp xúc với hình thoi tại đỉnh nhọn nằm ngay trên trục tọa độ, nơi một trọng số bằng 0. Trong khi đó, chúng tiếp xúc với hình tròn tại một điểm mượt mà, nơi cả hai trọng số đều khác 0.

### Tìm kiếm láng giềng gần nhất (Nearest Neighbor Search)

Mỗi hàm khoảng cách đều gắn liền với bài toán tìm kiếm lân cận gần nhất: Cho một điểm truy vấn, hãy tìm các điểm gần nó nhất trong tập dữ liệu.

Tìm kiếm chính xác đòi hỏi độ phức tạp $O(n \cdot d)$ cho mỗi truy vấn trên tập dữ liệu $n$ điểm với $d$ chiều. Với tập dữ liệu lớn, cách này quá chậm.

Các thuật toán láng giềng gần nhất xấp xỉ (Approximate Nearest Neighbor - ANN) đánh đổi một phần rất nhỏ độ chính xác để đổi lấy tốc độ vượt trội:

```
Thuật toán        Cách tiếp cận                   Nơi sử dụng
KD-trees          Phân vùng không gian theo trục  scikit-learn (số chiều thấp)
Ball trees        Các siêu cầu lồng nhau          scikit-learn (số chiều trung bình)
LSH               Phép chiếu băm ngẫu nhiên       Phát hiện dữ liệu gần trùng lặp
HNSW              Đồ thị thế giới nhỏ phân tầng   FAISS, Qdrant, Weaviate
                  (Hierarchical navigable small world)
IVF               Chỉ mục tệp nghịch đảo dựa      FAISS (quy mô hàng tỷ vector)
                  trên phân cụm
Product quant.    Nén vector, tìm kiếm            FAISS (hạn chế về bộ nhớ RAM)
                  trên không gian nén
```

HNSW là thuật toán thống trị trong các cơ sở dữ liệu vector hiện đại. Nó xây dựng một đồ thị nhiều tầng trong đó mỗi nút kết nối với các lân cận gần đúng của nó. Tìm kiếm bắt đầu từ tầng trên cùng (thưa thớt, các bước nhảy dài) và đi dần xuống tầng đáy (dày đặc, các bước nhảy ngắn chi tiết).

```figure
norm-unit-balls
```

## Build It (Thực hành xây dựng)

### Bước 1: Cài đặt toàn bộ các hàm chuẩn và khoảng cách

Xem `code/distances.py` để tìm hiểu mã nguồn cài đặt đầy đủ. Mọi hàm đều được xây dựng từ đầu chỉ bằng toán học cơ bản của Python.

### Bước 2: Cùng dữ liệu, khoảng cách khác nhau, láng giềng khác nhau

Bản demo trong `distances.py` tạo một tập dữ liệu, chọn một điểm truy vấn và cho thấy điểm lân cận gần nhất thay đổi như thế nào tùy theo độ đo khoảng cách. Điểm "gần nhất" theo chuẩn L1 có thể không phải là điểm gần nhất theo L2 hay Cosine.

### Bước 3: Tìm kiếm tương đồng Embedding

Mã nguồn bao gồm một bản mô phỏng tìm kiếm tương đồng embedding để tìm các "tài liệu" giống nhất với câu truy vấn bằng Cosine similarity so với khoảng cách L2, chứng minh rằng thứ tự xếp hạng có thể khác biệt rõ rệt.

## Use It (Ứng dụng thực tế)

Ứng dụng thực tế phổ biến nhất: Tìm kiếm các mục tương đồng trong cơ sở dữ liệu vector.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Khi bạn gọi `model.encode(text)` rồi tìm kiếm trong vector database, đây chính xác là những gì diễn ra bên dưới. Mô hình embedding ánh xạ văn bản thành vector. Vector database tính toán độ tương đồng cosine (hoặc dot product) giữa vector truy vấn và từng vector được lưu trữ, sử dụng các thuật toán ANN để tránh phải duyệt qua toàn bộ dữ liệu.

## Exercises (Bài tập)

1. Tính khoảng cách L1, L2 và L-vô cùng giữa (1, 2, 3) và (4, 0, 6). Xác nhận rằng $L_\infty \le L_2 \le L_1$ luôn đúng với mọi cặp điểm. Chứng minh toán học vì sao thứ tự này luôn được đảm bảo.

2. Tạo hai vector có cosine similarity cao (> 0.9) nhưng khoảng cách L2 lại lớn (> 10). Giải thích về mặt hình học điều gì đang diễn ra. Sau đó tạo hai vector có cosine similarity thấp (< 0.3) nhưng khoảng cách L2 lại rất nhỏ (< 0.5).

3. Cài đặt một hàm nhận một tập dữ liệu và một điểm truy vấn rồi trả về láng giềng gần nhất theo khoảng cách L1, L2, Cosine và Mahalanobis. Tìm một tập dữ liệu mà tại đó cả 4 độ đo đều bất đồng quan điểm về việc điểm nào là gần nhất.

4. Tính khoảng cách Wasserstein giữa [0.5, 0.5, 0, 0] và [0, 0, 0.5, 0.5] bằng tay bằng phương pháp CDF. Sau đó tính giữa [0.25, 0.25, 0.25, 0.25] và [0, 0, 0.5, 0.5]. Giá trị nào lớn hơn và tại sao?

5. Cài đặt thuật toán MinHash để ước lượng độ tương đồng Jaccard xấp xỉ. Tạo 100 tập hợp ngẫu nhiên, tính Jaccard chính xác cho mọi cặp, rồi so sánh với xấp xỉ MinHash khi dùng 50, 100 và 200 hàm băm. Vẽ đồ thị sai số xấp xỉ.

### Gợi ý (Hints)

**Bài tập 1.** Vector hiệu là (-3, 2, -3), hãy tính ba chuẩn của vector này bằng `l1_norm`, `l2_norm` và `linf_norm`. Để chứng minh, so sánh $\lvert d_i \rvert$ lớn nhất với tổng các bình phương, và so sánh tổng các bình phương với bình phương của tổng các giá trị tuyệt đối.

**Bài tập 2.** Cosine similarity bỏ qua độ dài, vì vậy hãy bắt đầu với hai vector cùng hướng và cho một vector có độ dài thật lớn. Với trường hợp thứ hai, hãy dùng hai vector rất ngắn nằm gần gốc tọa độ nhưng trỏ theo hai hướng khác nhau, rồi kiểm tra lại bằng `cosine_similarity` và `l2_distance`.

**Bài tập 3.** Gọi `find_nearest_neighbor(query, dataset, distance_fn)` cho từng hàm trong số `l1_distance`, `l2_distance`, `cosine_distance` và `mahalanobis_distance`. Hàm `demo_different_neighbors` cho thấy một tập dữ liệu có sự bất đồng giữa các độ đo, hãy xem qua trước rồi thay đổi tọa độ các điểm cho đến khi cả 4 hàm trả về 4 kết quả khác nhau.

**Bài tập 4.** Viết ra tổng tích lũy của từng phân phối, rồi cộng các hiệu tuyệt đối giữa hai tổng tích lũy. Kiểm tra lại kết quả tính tay bằng `wasserstein_1d(p, q)`, sau đó giải thích giá trị lớn hơn dựa trên quãng đường mà khối lượng xác suất phải di chuyển.

**Bài tập 5.** Với mỗi hàm băm, chữ ký MinHash của một tập hợp là giá trị băm nhỏ nhất trên các phần tử của nó, và tỷ lệ số hàm băm mà hai tập hợp có cùng giá trị nhỏ nhất sẽ ước lượng độ tương đồng Jaccard. So sánh từng ước lượng với `jaccard_similarity`; sai số sẽ giảm xấp xỉ theo tỷ lệ $1/\sqrt{k}$ với $k$ hàm băm.

## Key Terms (Thuật ngữ chính)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|------|----------------|----------------------|
| Norm (Chuẩn) | "Độ lớn của vector" | Hàm ánh xạ một vector thành một đại lượng vô hướng không âm, thỏa mãn bất đẳng thức tam giác, tính thuần nhất tuyệt đối và chỉ bằng 0 khi vector bằng 0 |
| L1 norm | "Khoảng cách Manhattan" | Tổng các giá trị tuyệt đối của các thành phần. Tạo ra tính thưa trong tối ưu hóa. Bền vững trước ngoại lai |
| L2 norm | "Khoảng cách Euclid" | Căn bậc hai của tổng bình phương các thành phần. Khoảng cách đường thẳng trong không gian Euclid |
| Lp norm | "Chuẩn tổng quát" | Căn bậc $p$ của tổng lũy thừa bậc $p$ các giá trị tuyệt đối. L1 và L2 là các trường hợp đặc biệt |
| L-infinity norm | "Chuẩn Max" hoặc "Khoảng cách Chebyshev" | Giá trị tuyệt đối lớn nhất trong các thành phần. Giới hạn của Lp khi $p$ tiến tới vô cùng |
| Cosine similarity | "Góc giữa các vector" | Tích vô hướng được chuẩn hóa bởi độ lớn của hai vector. Dao động từ -1 đến +1. Bỏ qua độ dài vector |
| Cosine distance | "1 trừ đi cosine similarity" | Chuyển đổi cosine similarity thành khoảng cách. Dao động từ 0 đến 2 |
| Dot product | "Cosine chưa chuẩn hóa" | Tổng tích của các thành phần tương ứng. Bằng cosine similarity nhân với độ lớn của cả hai vector |
| Mahalanobis distance | "Khoảng cách có xét đến tương quan" | Khoảng cách L2 trong một không gian đã được làm trắng (khử tương quan và chuẩn hóa) bằng ma trận hiệp phương sai |
| Jaccard similarity | "Độ trùng lặp tập hợp" | Kích thước phần giao chia cho kích thước phần hợp. Dành cho tập hợp, không phải vector số học |
| Edit distance | "Khoảng cách Levenshtein" | Số thao tác chèn, xóa, thay thế tối thiểu để biến chuỗi này thành chuỗi kia |
| KL divergence | "Khoảng cách giữa các phân phối" | Không phải khoảng cách thực sự (không đối xứng). Đo lượng bit dư thừa khi dùng $Q$ để mã hóa $P$ |
| Wasserstein distance | "Khoảng cách Earth mover" | Công tối thiểu để vận chuyển khối lượng xác suất từ phân phối này sang phân phối kia. Là một metric thực thụ |
| Approximate nearest neighbor | "Tìm kiếm ANN" | Các thuật toán (HNSW, LSH, IVF) giúp tìm các điểm gần đúng nhanh hơn nhiều so với tìm kiếm vét cạn |
| HNSW | "Thuật toán cơ sở dữ liệu vector" | Đồ thị thế giới nhỏ có thể điều hướng phân tầng. Dùng để tìm kiếm ANN cực nhanh |
| L1 regularization | "Lasso" | Thêm chuẩn L1 của trọng số vào hàm mất mát. Ép trọng số về 0 (tính thưa) |
| L2 regularization | "Ridge" hoặc "weight decay" | Thêm bình phương chuẩn L2 của trọng số vào hàm mất mát. Co trọng số về gần 0 mà không triệt tiêu hoàn toàn |
| Elastic Net | "L1 + L2" | Kết hợp điều chuẩn L1 và L2. Xử lý các nhóm đặc trưng có tương quan tốt hơn từng phương pháp đơn lẻ |

## Further Reading (Tài liệu đọc thêm)

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Thư viện từ Meta dành cho tìm kiếm ANN quy mô hàng tỷ vector
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - Bài báo kinh điển đưa khoảng cách Earth Mover vào huấn luyện GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - Thuật toán nền tảng cho tìm kiếm ANN
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, nơi cosine similarity trở thành tiêu chuẩn mặc định cho embedding
- [Tài liệu hướng dẫn sklearn.neighbors](https://scikit-learn.org/stable/modules/neighbors.html) - Cẩm nang thực hành về các độ đo khoảng cách và thuật toán tìm kiếm lân cận trong scikit-learn
