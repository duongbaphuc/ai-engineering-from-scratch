# Thống kê cho Machine Learning (Statistics for Machine Learning)

> Thống kê là cách duy nhất giúp bạn biết mô hình thực sự hoạt động hiệu quả hay chỉ đơn thuần là ăn may.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Tự tính toán từ đầu các đại lượng thống kê mô tả, hệ số tương quan Pearson/Spearman và ma trận hiệp phương sai
- Thực hiện các kiểm định giả thuyết (t-test, chi-squared) và diễn giải chính xác các giá trị p-value cùng khoảng tin cậy
- Sử dụng phương pháp lấy mẫu lại Bootstrap để xây dựng khoảng tin cậy cho bất kỳ độ đo nào mà không cần giả định về phân phối
- Phân biệt ý nghĩa thống kê (statistical significance) với ý nghĩa thực tiễn (practical significance) bằng các độ đo kích thước hiệu ứng (effect size)

## The Problem (Vấn đề)

Bạn huấn luyện hai mô hình. Model A đạt 0.87 trên tập kiểm thử (test set). Model B đạt 0.89. Bạn quyết định triển khai Model B lên production. Ba tuần sau, các chỉ số thực tế trên production lại tệ hơn trước đó. Chuyện gì đã xảy ra?

Model B trên thực tế không hề vượt trội hơn Model A. Khoảng chênh lệch 0.02 đó hoàn toàn là nhiễu ngẫu nhiên. Tập test của bạn quá nhỏ, hoặc phương sai quá lớn, hoặc cả hai. Bạn đã đưa một kết quả may rủi ngẫu nhiên lên hệ thống thực tế dưới vỏ bọc của một "sự cải tiến".

Điều này xảy ra liên tục: Sự xáo trộn thứ hạng ngoạn mục trên bảng xếp hạng Kaggle; các bài báo nghiên cứu không thể tái lập kết quả; các cuộc thử nghiệm A/B vội vàng tuyên bố người chiến thắng chỉ dựa trên vài trăm mẫu thử. Căn nguyên luôn là một: Có ai đó đã bỏ qua các phương pháp thống kê chuẩn tắc.

Thống kê cung cấp cho bạn công cụ để phân biệt tín hiệu thực sự với nhiễu ngẫu nhiên. Nó cho bạn biết khi nào một sự khác biệt là có thật, bạn nên tự tin tới mức nào, và bạn cần bao nhiêu dữ liệu trước khi có thể tin tưởng vào một kết quả. Mọi pipeline học máy, mọi so sánh mô hình, mọi thử nghiệm thực tế đều cần đến thống kê. Không có nó, bạn chỉ đang đoán mò.

## The Concept (Khái niệm)

### Thống kê mô tả: Tóm tắt bức tranh dữ liệu (Descriptive Statistics)

Trước khi mô hình hóa bất cứ thứ gì, bạn cần biết dữ liệu của mình trông như thế nào. Thống kê mô tả cô đọng toàn bộ tập dữ liệu thành một vài con số then chốt thể hiện hình dáng phân phối của nó.

**Các đại lượng đo lường xu hướng trung tâm (Measures of central tendency)** trả lời cho câu hỏi: "Đâu là điểm giữa?"

```
Mean (Trung bình):      Tổng tất cả các giá trị / số lượng mẫu
                        mu = (1/n) * sum(x_i)

Median (Trung vị):      Giá trị ở vị trí chính giữa khi đã sắp xếp
                        Bền vững trước các ngoại lai. Nếu bạn có [1, 2, 3, 4, 1000],
                        mean là 202 nhưng median vẫn là 3.

Mode (Yếu vị):          Giá trị xuất hiện với tần suất nhiều nhất
                        Hữu ích cho dữ liệu phân loại (categorical).
                        Đối với dữ liệu liên tục, ít khi mang nhiều thông tin.
```

Mean là điểm cân bằng trọng lực. Median là mốc chia đôi dữ liệu 50/50. Khi hai giá trị này lệch nhau, phân phối của bạn bị lệch (skewed). Phân phối thu nhập thường có $\text{mean} \gg \text{median}$ (lệch phải do các tỷ phú kéo giá trị trung bình lên). Phân phối hàm mất mát trong quá trình huấn luyện thường có $\text{mean} \ll \text{median}$ (lệch trái do đa số các mẫu dễ học kéo mean xuống thấp).

**Các đại lượng đo lường độ phân tán (Measures of spread)** trả lời cho câu hỏi: "Dữ liệu phân tán tới mức nào?"

```
Variance (Phương sai):         Độ lệch bình phương trung bình so với giá trị trung bình
                               sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation (Độ lệch chuẩn): Căn bậc hai của phương sai
                               sigma = sqrt(sigma^2)
                               Có cùng đơn vị đo với dữ liệu gốc, do đó dễ diễn giải trực quan hơn.

Range (Khoảng biến thiên):     Giá trị lớn nhất trừ giá trị nhỏ nhất (max - min)
                               Rất nhạy cảm với ngoại lai. Hầu như không bao giờ được dùng đơn lẻ.

IQR (Khoảng tứ phân vị):       Q3 - Q1 (Interquartile range)
                               Độ trải rộng của 50% dữ liệu ở phần giữa.
                               Bền vững trước ngoại lai. Thường dùng cho biểu đồ box plot và phát hiện ngoại lai.
```

**Phần trăm vị (Percentiles)** chia dữ liệu đã sắp xếp thành 100 phần bằng nhau. Phần trăm vị thứ 25 (Q1) có nghĩa là 25% giá trị nằm dưới mốc này. Phần trăm vị thứ 50 chính là trung vị (median). Phần trăm vị thứ 75 là Q3.

```
Trong giám sát độ trễ (latency monitoring):
  P50 = độ trễ trung vị           (trải nghiệm của người dùng điển hình)
  P95 = phần trăm vị thứ 95       (trường hợp xấu nhưng chưa phải tồi tệ nhất)
  P99 = phần trăm vị thứ 99       (độ trễ đuôi tail latency, thường gấp 10 lần trung vị)
```

Trong học máy, bạn cần quan tâm đến các mốc percentile cho độ trễ suy luận (inference latency), phân phối độ tự tin của dự đoán và phân tích các trường hợp lỗi biên. Một mô hình có sai số trung bình thấp nhưng sai số P99 cực lớn có thể hoàn toàn vô dụng hoặc nguy hiểm cho các ứng dụng yêu cầu an toàn nghiêm ngặt.

**Thống kê mẫu so với thống kê tổng thể.** Khi tính phương sai từ một mẫu (sample), hãy chia cho $(n-1)$ thay vì $n$. Đây là hiệu chỉnh Bessel (Bessel's correction). Nó bù đắp cho việc giá trị trung bình mẫu không phải là giá trị trung bình thực sự của quần thể. Nếu giữ mẫu số là $n$, bạn sẽ ước lượng thiếu phương sai thực tế một cách có hệ thống. Chia cho $(n-1)$ sẽ biến nó thành ước lượng không chệch (unbiased estimate).

```math
\begin{aligned}
\text{Phương sai tổng thể:} \quad \sigma^2 &= \frac{1}{N} \sum (x_i - \mu)^2 \\
\text{Phương sai mẫu:} \quad s^2 &= \frac{1}{n-1} \sum (x_i - \bar{x})^2
\end{aligned}
```

Trong thực tế: Nếu $n$ rất lớn (hàng nghìn mẫu), sự khác biệt là không đáng kể. Nhưng nếu $n$ nhỏ (vài chục mẫu), điều này cực kỳ quan trọng.

### Tương quan (Correlation): Các biến số biến thiên cùng nhau như thế nào

Độ tương quan đo lường độ mạnh và chiều hướng của mối quan hệ tuyến tính giữa hai biến số.

**Hệ số tương quan Pearson** đo lường liên kết tuyến tính:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  Mối quan hệ tuyến tính đồng biến tuyệt đối
r = -1:  Mối quan hệ tuyến tính nghịch biến tuyệt đối
r =  0:  Không có quan hệ tuyến tính (nhưng có thể tồn tại quan hệ phi tuyến!)

Miền giá trị: [-1, 1]
```

Pearson giả định rằng mối quan hệ là tuyến tính và cả hai biến đều tuân theo phân phối chuẩn xấp xỉ. Nó rất nhạy cảm với ngoại lai -- một điểm dữ liệu cực đoan có thể kéo $r$ từ 0.1 vọt lên 0.9.

**Hệ số tương quan hạng Spearman** đo lường liên kết đơn điệu:

```
1. Thay thế mỗi giá trị bằng thứ hạng của nó (1, 2, 3, ...)
2. Tính hệ số tương quan Pearson trên các giá trị thứ hạng đó

Spearman nắm bắt bất kỳ mối quan hệ đơn điệu nào, không chỉ riêng tuyến tính.
Nếu y = x^3, Pearson cho r < 1 nhưng Spearman cho rho = 1 tuyệt đối.
```

**Khi nào nên dùng loại nào:**

```
Pearson:     Cả hai biến đều liên tục và có phân phối chuẩn xấp xỉ.
             Bạn đặc biệt quan tâm đến mối quan hệ tuyến tính thẳng hàng.
             Không có điểm ngoại lai cực đoan.

Spearman:    Dữ liệu thứ bậc (thứ hạng xếp loại, điểm đánh giá).
             Dữ liệu không tuân theo phân phối chuẩn.
             Bạn nghi ngờ có quan hệ đơn điệu phi tuyến.
             Có sự hiện diện của các điểm ngoại lai.
```

**Quy tắc vàng:** Tương quan không đồng nghĩa với nhân quả (Correlation does not imply causation). Doanh số bán kem và số ca đuối nước có tương quan thuận vì cả hai đều tăng vào mùa hè (biến ẩn là thời tiết nóng). Độ chính xác của mô hình và số lượng tham số có tương quan, nhưng việc tăng tham số bừa bãi không tự động nâng cao độ chính xác (nguy cơ overfitting).

### Ma trận hiệp phương sai (Covariance Matrix)

Hiệp phương sai giữa hai biến đo lường mức độ biến thiên cùng nhau của chúng:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X và Y có xu hướng cùng tăng với nhau
Cov(X, Y) < 0:  Khi X tăng, Y có xu hướng giảm
Cov(X, Y) = 0:  Không có sự đồng biến thiên tuyến tính
```

Với $d$ đặc trưng, ma trận hiệp phương sai $C$ là một ma trận $d \times d$ trong đó phần tử $C[i][j] = \mathop{\mathrm{Cov}}(\text{feature}_i, \text{feature}_j)$. Các phần tử trên đường chéo chính $C[i][i]$ chính là phương sai của từng đặc trưng.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Tính chất:
  - Đối xứng: C[i][j] = C[j][i]
  - Nửa xác định dương: Mọi trị riêng đều >= 0
  - Đường chéo chính = các phương sai
  - Ngoài đường chéo = các hiệp phương sai
```

**Mối liên hệ với PCA:** PCA thực hiện phân rã trị riêng trên ma trận hiệp phương sai. Các vector riêng chính là các thành phần chính (các hướng có phương sai lớn nhất). Các trị riêng cho biết mỗi thành phần nắm giữ bao nhiêu phương sai dữ liệu. Đây là lý do ma trận hiệp phương sai là đối tượng chuẩn tắc để phân rã: nó mã hóa toàn bộ các mối liên hệ tuyến tính từng cặp trong dữ liệu.

**Mối liên hệ với tương quan:** Ma trận tương quan thực chất là ma trận hiệp phương sai của các biến đã được chuẩn hóa (chia cho độ lệch chuẩn). Tương quan chuẩn hóa hiệp phương sai để mọi giá trị đều nằm gọn trong $[-1, 1]$.

### Kiểm định giả thuyết (Hypothesis Testing)

Kiểm định giả thuyết là một khuôn khổ để đưa ra quyết định khi đối mặt với sự không chắc chắn. Bạn bắt đầu với một giả định, thu thập dữ liệu và xác định xem dữ liệu có nhất quán với giả định đó hay không.

**Thiết lập bài toán:**

```
Giả thuyết không (Null hypothesis - H0): Giả định mặc định, thường là "không có hiệu ứng/không khác biệt"
Giả thuyết đối (Alternative hypothesis - H1): Điều mà bạn đang cố gắng chứng minh

Ví dụ:
  H0: Model A và Model B có độ chính xác như nhau
  H1: Model B có độ chính xác cao hơn Model A
```

**Giá trị p-value** là xác suất quan sát thấy dữ liệu có mức độ cực đoan như (hoặc hơn) dữ liệu bạn đang có, với điều kiện H0 là đúng. Nó KHÔNG PHẢI là xác suất H0 đúng. Đây là hiểu lầm phổ biến nhất trong toàn bộ giới thống kê.

```
p-value = P(dữ liệu cực đoan như thế này | H0 là đúng)

Nếu p-value < alpha (thường là 0.05):
    Bác bỏ H0. Kết quả được coi là "có ý nghĩa thống kê" (statistically significant).
Nếu p-value >= alpha:
    Không đủ bằng chứng để bác bỏ H0.
    Điều này KHÔNG đồng nghĩa với việc H0 là đúng.
```

**Khoảng tin cậy (Confidence intervals)** đưa ra một khoảng các giá trị hợp lý cho một tham số:

```
Khoảng tin cậy 95% cho giá trị trung bình:
    x_bar +/- z * (s / sqrt(n))

trong đó z = 1.96 cho mức độ tin cậy 95%

Ý nghĩa chuẩn tắc: Nếu bạn lặp lại thử nghiệm này rất nhiều lần, 95% các khoảng tin cậy
được tính ra sẽ bao trùm giá trị trung bình thực sự của quần thể.
Nó KHÔNG có nghĩa là có 95% xác suất giá trị thực nằm trong khoảng cụ thể này.
```

Độ rộng của khoảng tin cậy cho bạn biết về độ chính xác của phép ước lượng. Khoảng rộng đồng nghĩa với độ bất định cao. Khoảng hẹp nghĩa là phép ước lượng rất tập trung (nhưng chưa chắc đã chuẩn xác nếu dữ liệu bị thiên vị/chệch).

### Kiểm định t (t-test)

Kiểm định t dùng để so sánh các giá trị trung bình:

**Kiểm định t một mẫu (One-sample t-test):** Giá trị trung bình của quần thể có khác với một giá trị giả định hay không?

```math
\begin{aligned}
t &= \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \\[4pt]
\text{bậc tự do (degrees of freedom)} &= n - 1
\end{aligned}
```

**Kiểm định t hai mẫu độc lập (Two-sample independent t-test):** Giá trị trung bình của hai nhóm có khác nhau không?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

Đây là kiểm định Welch's t-test, không giả định phương sai hai nhóm bằng nhau.
Luôn ưu tiên dùng Welch's t-test trừ khi bạn có lý do chắc chắn rằng phương sai hai nhóm là như nhau.
```

**Kiểm định t theo cặp (Paired t-test):** Khi các phép đo đi theo từng cặp (cùng một tập dữ liệu phân chia được đánh giá trên cả hai mô hình):

```
Tính hiệu d_i = x_i - y_i cho từng cặp
Sau đó thực hiện kiểm định t một mẫu trên dãy d_i đối chiếu với mu_0 = 0
```

Trong học máy, paired t-test rất phổ biến: bạn chạy cả hai mô hình trên cùng 10 fold của cross-validation rồi so sánh điểm số theo từng cặp fold.

### Kiểm định Chi-squared ($\chi^2$)

Kiểm định Chi-squared kiểm tra xem tần suất quan sát được có khớp với tần suất kỳ vọng hay không. Rất hữu ích cho dữ liệu phân loại.

```
chi^2 = sum((observed - expected)^2 / expected)

Ví dụ: Phân phối đầu ra của một mô hình ngôn ngữ có khớp với
phân phối của tập huấn luyện trên các danh mục không?

Danh mục     Quan sát (Observed)   Kỳ vọng (Expected)
Tích cực            120                   100
Tiêu cực             80                   100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

Với 1 bậc tự do, chi^2 = 8 tương ứng với p < 0.005.
Sự khác biệt là có ý nghĩa thống kê rõ rệt.
```

### Thử nghiệm A/B cho các mô hình Machine Learning

A/B testing trong học máy không hoàn toàn giống với A/B testing trên giao diện web thông thường. Việc so sánh mô hình có các thách thức đặc thù:

```
1. Cùng tập kiểm thử:   Cả hai mô hình phải được đánh giá trên dữ liệu giống hệt nhau.
                        Tập test khác nhau sẽ làm cho phép so sánh trở nên vô nghĩa.

2. Đa chỉ số:           Độ chính xác (accuracy) là không đủ. Bạn cần precision,
                        recall, F1, độ trễ và các chỉ số công bằng (fairness).

3. Phương sai:          Dùng cross-validation hoặc bootstrap để ước lượng phương sai
                        của từng chỉ số, thay vì chỉ dựa vào một điểm ước lượng duy nhất.

4. Rò rỉ dữ liệu:       Nếu tập test vô tình bị sử dụng trong quá trình chọn mô hình,
                        phép so sánh sẽ bị thiên vị. Luôn giữ riêng một tập test cuối cùng độc lập.
```

**Quy trình thực hiện chuẩn tắc:**

```
1. Xác định chỉ số mục tiêu và mức ý nghĩa thống kê (alpha = 0.05)
2. Chạy cả hai mô hình trên cùng các phân chia k-fold cross-validation
3. Thu thập điểm số theo cặp: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Tính toán độ chênh lệch: d_i = b_i - a_i
5. Chạy paired t-test trên các độ chênh lệch
6. Kiểm tra: Độ chênh lệch trung bình có khác 0 một cách có ý nghĩa thống kê không?
7. Tính khoảng tin cậy cho độ chênh lệch trung bình
8. Tính kích thước hiệu ứng (Cohen's d) để đánh giá ý nghĩa thực tiễn
```

### Ý nghĩa thống kê vs Ý nghĩa thực tiễn

Một kết quả có thể có ý nghĩa thống kê nhưng lại hoàn toàn vô nghĩa trong thực tiễn. Với tập dữ liệu đủ lớn, ngay cả một khác biệt li ti không đáng kể cũng đạt được ý nghĩa thống kê.

```
Ví dụ:
  Độ chính xác Model A: 0.9234
  Độ chính xác Model B: 0.9237
  n = 1.000.000 mẫu kiểm thử
  p-value = 0.001

Có ý nghĩa thống kê không? Có.
Có ý nghĩa thực tiễn không? Mức cải thiện 0.03% hoàn toàn không xứng đáng
với chi phí kỹ thuật để triển khai và bảo trì một mô hình mới.
```

**Kích thước hiệu ứng (Effect size)** định lượng độ lớn của sự khác biệt, độc lập với quy mô mẫu:

```math
\begin{aligned}
\text{Cohen's } d &= \frac{\text{mean}_1 - \text{mean}_2}{\text{pooled\_std}} \\[4pt]
d &= 0.2: \quad \text{hiệu ứng nhỏ} \\
d &= 0.5: \quad \text{hiệu ứng trung bình} \\
d &= 0.8: \quad \text{hiệu ứng lớn}
\end{aligned}
```

Luôn báo cáo đồng thời cả p-value lẫn kích thước hiệu ứng. P-value cho biết sự khác biệt có thực hay không. Kích thước hiệu ứng cho biết sự khác biệt đó có đáng để bận tâm hay không.

### Vấn đề kiểm định nhiều giả thuyết (Multiple Comparison Problem)

Khi bạn kiểm định đồng thời nhiều giả thuyết, một số kiểm định sẽ đạt "ý nghĩa thống kê" thuần túy do ngẫu nhiên. Nếu bạn kiểm định 20 yếu tố ở mức $\alpha = 0.05$, bạn chắc chắn sẽ gặp 1 dương tính giả (false positive) ngay cả khi không có bất kỳ hiệu ứng thực nào tồn tại.

```
P(có ít nhất một dương tính giả) = 1 - (1 - alpha)^m

m = 20 kiểm định, alpha = 0.05:
P(dương tính giả) = 1 - 0.95^20 = 0.64

Bạn có tới 64% khả năng gặp phải ít nhất một kết quả dương tính giả!
```

**Hiệu chỉnh Bonferroni:** Chia $\alpha$ cho tổng số lượng kiểm định:

```
Alpha điều chỉnh = alpha / m = 0.05 / 20 = 0.0025

Chỉ bác bỏ H0 nếu p-value < 0.0025.
Phương pháp này chặt chẽ nhưng đơn giản, hiệu quả khi các bài kiểm định độc lập.
```

Trong học máy, điều này xảy ra khi bạn so sánh mô hình trên nhiều thang đo khác nhau, thử nghiệm hàng chục bộ siêu tham số, hoặc đánh giá trên nhiều bộ dữ liệu cùng lúc.

### Phương pháp Bootstrap (Bootstrap Methods)

Bootstrapping ước lượng phân phối lấy mẫu của một đại lượng thống kê bằng cách lấy mẫu lại có hoàn lại (resampling with replacement) từ chính tập dữ liệu hiện có. Hoàn toàn không cần giả định phân phối gốc là gì.

**Thuật toán:**

```
1. Bạn có n điểm dữ liệu
2. Rút ngẫu nhiên n mẫu CÓ HOÀN LẠI (một số điểm sẽ xuất hiện nhiều lần,
   một số điểm không xuất hiện lần nào)
3. Tính đại lượng thống kê mục tiêu trên mẫu bootstrap này
4. Lặp lại quá trình B lần (thường B = 1000 đến 10000)
5. Phân phối của B giá trị thống kê thu được sẽ xấp xỉ phân phối lấy mẫu thực tế
```

**Khoảng tin cậy Bootstrap (phương pháp phần trăm vị):**

```
Sắp xếp B giá trị thống kê bootstrap theo thứ tự tăng dần
Khoảng tin cậy 95% CI = [phần trăm vị thứ 2.5, phần trăm vị thứ 97.5]
```

**Vì sao Bootstrap là vũ khí tối thượng cho Machine Learning:**
- Độ chính xác trên tập test chỉ là một ước lượng điểm. Bootstrap cung cấp cho bạn khoảng tin cậy đầy đủ.
- Bạn không thể giả định phân phối của các chỉ số là phân phối chuẩn (đặc biệt là AUC, F1, precision@k).
- Bootstrap hoạt động với MỌI đại lượng thống kê: Trung vị, tỷ số của hai giá trị trung bình, độ chênh lệch AUC giữa hai mô hình.
- Không cần bất kỳ công thức toán học dạng đóng phức tạp nào.

**So sánh mô hình bằng Bootstrap:**

```
1. Bạn có kết quả dự đoán của Model A và Model B trên cùng một tập test
2. Với mỗi vòng lặp bootstrap:
   a. Lấy mẫu lại các chỉ số dữ liệu kiểm thử (có hoàn lại)
   b. Tính metric_A và metric_B trên tập mẫu mới
   c. Lưu lại độ lệch diff = metric_B - metric_A
3. Tính khoảng tin cậy 95% CI cho chuỗi diffs thu được:
   [phần trăm vị thứ 2.5 của diff, phần trăm vị thứ 97.5 của diff]
4. Nếu khoảng CI này không chứa số 0, sự khác biệt được xác nhận là có ý nghĩa thực sự
```

Cách này bền vững hơn hẳn so với paired t-test vì nó không phụ thuộc vào bất kỳ giả định phân phối nào.

### Kiểm định tham số vs Phi tham số (Parametric vs Non-parametric Tests)

**Kiểm định tham số (Parametric tests)** giả định dữ liệu tuân theo một phân phối cụ thể (thường là chuẩn):
- t-test: Giả định dữ liệu phân phối chuẩn (hoặc $n$ lớn theo định lý giới hạn trung tâm CLT)
- ANOVA: Giả định phân phối chuẩn và phương sai các nhóm bằng nhau
- Pearson r: Giả định phân phối chuẩn hai biến

**Kiểm định phi tham số (Non-parametric tests)** không đưa ra bất kỳ giả định nào về phân phối:
- Mann-Whitney U: So sánh hai nhóm độc lập (thay thế independent t-test)
- Wilcoxon signed-rank: So sánh dữ liệu theo cặp (thay thế paired t-test)
- Spearman rho: Tương quan dựa trên thứ hạng (thay thế Pearson)
- Kruskal-Wallis: So sánh nhiều nhóm (thay thế ANOVA)

**Khi nào nên dùng phi tham số:**
- Kích thước mẫu nhỏ ($n < 30$) và dữ liệu rõ ràng không tuân theo phân phối chuẩn
- Dữ liệu dạng thứ bậc (đánh giá sao, thứ hạng)
- Xuất hiện các ngoại lai nặng nề mà bạn không thể loại bỏ
- Phân phối bị lệch mạnh

**Khi nào nên dùng tham số:**
- Kích thước mẫu lớn (CLT giúp đại lượng thống kê kiểm định xấp xỉ chuẩn)
- Dữ liệu đối xứng tương đối và không có ngoại lai cực đoan
- Cần sức mạnh thống kê (statistical power) cao hơn để phát hiện khác biệt thực tế

Trong các thử nghiệm học máy, bạn thường có $n$ khá nhỏ (5 hoặc 10 fold cross-validation), vì vậy các kiểm định phi tham số như Wilcoxon signed-rank thường phù hợp hơn nhiều so với t-test.

### Định lý giới hạn trung tâm (Central Limit Theorem - CLT)

CLT khẳng định rằng phân phối của các giá trị trung bình mẫu sẽ tiến dần về phân phối chuẩn khi kích thước mẫu $n$ tăng lên, bất kể phân phối của quần thể ban đầu có hình dạng như thế nào.

```
Nếu X_1, X_2, ..., X_n độc lập và có cùng phân phối (iid) với kỳ vọng mu và phương sai sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    khi n -> vô cùng

Áp dụng tốt với n >= 30 trong hầu hết các trường hợp.
Với các phân phối bị lệch rất mạnh, bạn có thể cần n >= 100.
```

**Ý nghĩa thực tiễn đối với học máy:**
1. Biện minh cho việc áp dụng khoảng tin cậy và kiểm định t trên các chỉ số tổng hợp
2. Giải thích vì sao việc lấy trung bình trên các fold cross-validation mang lại ước lượng ổn định dù từng fold đơn lẻ biến động mạnh
3. Thuật toán Mini-batch Gradient Descent hoạt động được là nhờ gradient trung bình trên một batch xấp xỉ gradient thực tế của toàn bộ dữ liệu (CLT đang vận hành)
4. Phương pháp Ensemble: Lấy trung bình dự đoán từ nhiều mô hình đem lại kết quả ổn định hơn hẳn bất kỳ mô hình đơn lẻ nào

**Những điều CLT KHÔNG làm được:**
- KHÔNG làm cho dữ liệu gốc của bạn trở thành phân phối chuẩn. Nó chỉ làm cho GIÁ TRỊ TRUNG BÌNH của các mẫu trở thành chuẩn.
- KHÔNG hoạt động với các phân phối đuôi nặng có phương sai vô hạn (như phân phối Cauchy).
- KHÔNG áp dụng cho dữ liệu phụ thuộc lẫn nhau (dữ liệu chuỗi thời gian nếu không có hiệu chỉnh thích hợp).

### 10 sai lầm thống kê kinh điển trong các bài báo Machine Learning

1. **Kiểm thử trên tập huấn luyện.** Chắc chắn dẫn đến overfitting. Luôn luôn giữ lại một tập kiểm thử mà mô hình chưa bao giờ nhìn thấy.
2. **Không báo cáo khoảng tin cậy.** Chỉ báo cáo một con số accuracy duy nhất mà không có độ bất định khiến kết quả không thể thẩm định và tái lập.
3. **Bỏ qua vấn đề kiểm định nhiều giả thuyết.** Thử nghiệm 50 cấu hình rồi chọn cấu hình tốt nhất mà không hiệu chỉnh sẽ làm bùng nổ tỷ lệ dương tính giả.
4. **Nhầm lẫn giữa ý nghĩa thống kê và thực tiễn.** Một p-value bằng 0.001 đối với mức cải thiện 0.01% độ chính xác là hoàn toàn vô giá trị trong đời thực.
5. **Dùng accuracy cho dữ liệu mất cân bằng.** Đạt 99% accuracy trên tập dữ liệu có 99% nhãn âm tính chỉ có nghĩa là mô hình không học được gì cả. Hãy dùng precision, recall, F1 hoặc AUC.
6. **Lựa chọn chỉ số có lợi (Cherry-picking).** Chỉ báo cáo thang đo duy nhất mà mô hình của mình chiến thắng. Một đánh giá trung thực phải báo cáo toàn bộ các thang đo liên quan.
7. **Rò rỉ thông tin qua phân chia train/test.** Chuẩn hóa dữ liệu trước khi chia tập, hoặc dùng dữ liệu tương lai để dự đoán quá khứ.
8. **Tập test quá nhỏ và không ước lượng phương sai.** Đánh giá trên 100 mẫu rồi tuyên bố cải thiện 2% chỉ là nhiễu, không phải tín hiệu.
9. **Mặc định tính độc lập khi dữ liệu không độc lập.** Các bức ảnh y tế từ cùng một bệnh nhân, nhiều câu văn từ cùng một tài liệu. Dữ liệu trong cùng một nhóm luôn có tương quan.
10. **P-hacking.** Liên tục đổi bài kiểm định, lọc các tập con hoặc thay đổi tiêu chí loại trừ mẫu cho đến khi đạt được $p < 0.05$. Kết quả đó chỉ là sản phẩm nhân tạo của quá trình tìm kiếm may rủi.

## Building It (Thực hành xây dựng)

Bạn sẽ tự tay cài đặt:

1. **Thống kê mô tả từ đầu** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Các hàm tương quan** (Pearson và Spearman, cùng ma trận hiệp phương sai)
3. **Các kiểm định giả thuyết** (one-sample t-test, two-sample t-test, chi-squared test)
4. **Khoảng tin cậy Bootstrap** (cho bất kỳ đại lượng thống kê nào, không cần giả định phân phối)
5. **Bộ mô phỏng A/B test** (sinh dữ liệu, chạy kiểm định, kiểm tra sai lầm Loại I và Loại II)
6. **Demo ý nghĩa thống kê vs ý nghĩa thực tiễn** (chứng minh việc $n$ lớn khiến mọi thứ đều trở nên "có ý nghĩa thống kê")

Toàn bộ được xây dựng từ đầu, chỉ dùng thư viện chuẩn `math` và `random`. Không numpy, không scipy.

```figure
f3-bootstrap-resample
```

## Key Terms (Thuật ngữ chính)

| Thuật ngữ | Định nghĩa |
|---|---|
| Mean (Trung bình) | Tổng các giá trị chia cho số lượng phần tử. Rất nhạy cảm với ngoại lai |
| Median (Trung vị) | Giá trị ở giữa của dữ liệu đã sắp xếp. Bền vững trước ngoại lai |
| Standard deviation (Độ lệch chuẩn) | Căn bậc hai của phương sai. Đo độ phân tán theo đơn vị gốc của dữ liệu |
| Percentile (Phần trăm vị) | Giá trị mà dưới đó có một tỷ lệ phần trăm dữ liệu nhất định rơi vào |
| IQR (Khoảng tứ phân vị) | Q3 trừ Q1. Độ phân tán của 50% lượng dữ liệu ở khoảng giữa |
| Pearson correlation | Đo lường liên kết tuyến tính giữa hai biến. Miền giá trị trong $[-1, 1]$ |
| Spearman correlation | Đo lường liên kết đơn điệu dựa trên thứ hạng |
| Covariance matrix (Ma trận hiệp phương sai) | Ma trận chứa các hiệp phương sai từng cặp giữa toàn bộ các đặc trưng |
| Null hypothesis (Giả thuyết không) | Giả định mặc định là không có hiệu ứng hoặc không có sự khác biệt (H0) |
| p-value | Xác suất xuất hiện dữ liệu cực đoan như thế này với điều kiện H0 là đúng |
| Confidence interval (Khoảng tin cậy) | Khoảng các giá trị hợp lý cho một tham số ở mức độ tin cậy xác định |
| t-test | Kiểm định xem các giá trị trung bình có khác nhau đáng kể không. Dùng phân phối t |
| Chi-squared test | Kiểm định xem tần suất quan sát có khác biệt so với tần suất kỳ vọng không |
| Effect size (Kích thước hiệu ứng) | Độ lớn của sự khác biệt, độc lập với kích thước mẫu. Cohen's d là đại diện tiêu biểu |
| Bonferroni correction | Chia ngưỡng ý nghĩa cho số lượng kiểm định để kiểm soát tỷ lệ dương tính giả |
| Bootstrap | Phương pháp lấy mẫu lại có hoàn lại để ước lượng phân phối lấy mẫu |
| Type I error (Sai lầm loại I) | Dương tính giả (False positive). Bác bỏ H0 khi H0 thực sự đúng |
| Type II error (Sai lầm loại II) | Âm tính giả (False negative). Không bác bỏ được H0 khi H0 thực sự sai |
| Statistical power (Sức mạnh thống kê) | Xác suất bác bỏ chính xác một giả thuyết H0 sai. Power = 1 trừ tỷ lệ sai lầm loại II |
| Central limit theorem (Định lý giới hạn trung tâm) | Giá trị trung bình mẫu hội tụ về phân phối chuẩn khi kích thước mẫu tăng lên |
| Parametric test (Kiểm định tham số) | Giả định một phân phối cụ thể cho dữ liệu (thường là phân phối chuẩn) |
| Non-parametric test (Kiểm định phi tham số) | Không đưa ra giả định nào về phân phối. Hoạt động trên thứ hạng hoặc dấu |
