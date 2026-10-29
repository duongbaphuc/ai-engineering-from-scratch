# Naive Bayes

> Giả định "ngây thơ" (naive) về mặt toán học là sai, nhưng mô hình vẫn hoạt động hiệu quả đến kinh ngạc. Đó chính là vẻ đẹp của nó.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 phút

## Learning Objectives

- Tự tay xây dựng Multinomial Naive Bayes từ đầu (from scratch) với làm mịn Laplace (Laplace smoothing) cho bài toán phân loại văn bản
- Giải thích vì sao giả định độc lập ngây thơ tuy sai về mặt toán học nhưng trên thực tế vẫn tạo ra thứ hạng phân lớp (class rankings) chính xác
- So sánh các biến thể Multinomial, Bernoulli và Gaussian Naive Bayes, đồng thời chọn đúng biến thể cho từng kiểu đặc trưng (feature type)
- Đánh giá Naive Bayes so với hồi quy Logistic trên dữ liệu thưa số chiều cao (high-dimensional sparse data) và giải thích sự đánh đổi bias-variance đang diễn ra

## Vấn đề đặt ra (The Problem)

Bạn cần phân loại văn bản. Lọc email thành spam hoặc không phải spam (not-spam). Đánh giá phản hồi khách hàng thành tích cực hoặc tiêu cực. Phân loại các yêu cầu hỗ trợ (support tickets) vào các danh mục. Bạn có hàng nghìn đặc trưng (mỗi từ là một đặc trưng) nhưng dữ liệu huấn luyện lại hạn chế.

Hầu hết các bộ phân loại đều "nghẹt thở" ở kịch bản này. Hồi quy Logistic (Logistic regression) cần đủ mẫu để ước lượng hàng nghìn trọng số một cách đáng tin cậy. Cây quyết định (Decision trees) phân nhánh trên từng từ một và rất dễ bị quá khớp (overfit) nghiêm trọng. Thuật toán KNN trong không gian 10.000 chiều trở nên vô nghĩa vì mọi điểm dữ liệu đều cách đều nhau do lời nguyền số chiều (curse of dimensionality).

Naive Bayes giải quyết bài toán này một cách xuất sắc. Nó đưa ra một giả định sai về mặt toán học (rằng mọi đặc trưng đều độc lập với nhau khi biết trước nhãn lớp), thế nhưng nó vẫn vượt trội hơn các mô hình "thông minh hơn" trong tác vụ phân loại văn bản, đặc biệt là với các tập dữ liệu huấn luyện nhỏ. Mô hình huấn luyện chỉ qua đúng một lượt duyệt dữ liệu (single pass). Nó mở rộng quy mô lên hàng triệu đặc trưng dễ dàng. Nó tạo ra các ước lượng xác suất (dù thường không được hiệu chuẩn chuẩn xác - poorly calibrated - do giả định độc lập).

Hiểu được lý do tại sao một giả định sai lại dẫn đến những dự đoán tốt sẽ dạy cho bạn một bài học nền tảng trong học máy: mô hình tốt nhất không nhất thiết là mô hình đúng đắn nhất về mặt lý thuyết, mà là mô hình có sự đánh đổi bias-variance tốt nhất cho dữ liệu của bạn.

## Khái niệm (The Concept)

### Định lý Bayes (Ôn tập nhanh)

Định lý Bayes lật ngược xác suất có điều kiện:

```math
P(\text{class} \mid \text{features}) = \frac{P(\text{features} \mid \text{class}) \cdot P(\text{class})}{P(\text{features})}
```

Chúng ta muốn tìm `P(class | features)` -- xác suất một tài liệu thuộc về một lớp nhất định khi biết các từ ngữ có trong tài liệu đó. Ta có thể tính giá trị này từ:
- `P(features | class)` -- hợp lý (likelihood): xác suất bắt gặp các từ này trong những tài liệu thuộc lớp đó
- `P(class)` -- tiên nghiệm (prior probability): xác suất ban đầu của lớp (ví dụ: nhìn chung thư rác xuất hiện phổ biến đến mức nào?)
- `P(features)` -- bằng chứng (evidence): giống nhau cho mọi lớp, nên ta có thể bỏ qua mẫu số này khi so sánh các lớp với nhau

Lớp nào có `P(class | features)` cao nhất sẽ chiến thắng.

### Giả định độc lập ngây thơ (The Naive Independence Assumption)

Việc tính toán chính xác `P(features | class)` đòi hỏi phải ước lượng phân phối xác suất đồng thời (joint probability) của tất cả các đặc trưng cùng lúc. Với một bộ từ vựng gồm 10.000 từ, bạn sẽ phải ước lượng một phân phối trên $2^{10{,}000}$ tổ hợp khả dĩ. Điều này hoàn toàn bất khả thi.

Giả định ngây thơ (naive assumption): mọi đặc trưng đều độc lập có điều kiện với nhau khi biết lớp.

```math
P(w_1, w_2, \ldots, w_n \mid \text{class}) = P(w_1 \mid \text{class}) \cdot P(w_2 \mid \text{class}) \cdots P(w_n \mid \text{class})
```

Thay vì một phân phối đồng thời bất khả thi, bạn chỉ cần ước lượng $n$ phân phối đơn lẻ cho từng đặc trưng. Mỗi phân phối chỉ cần đếm tần suất xuất hiện.

Giả định này rõ ràng là sai trên thực tế. Hai từ "machine" và "learning" không hề độc lập trong bất kỳ văn bản nào. Nhưng bộ phân loại không cần các ước lượng xác suất tuyệt đối chính xác. Nó chỉ cần xếp hạng thứ tự (rankings) chính xác -- tức là lớp nào có xác suất cao nhất. Giả định độc lập gây ra các sai số hệ thống, nhưng những sai số đó ảnh hưởng đến tất cả các lớp theo cách tương tự nhau, do đó thứ hạng giữa các lớp vẫn được giữ nguyên.

### Vì sao nó vẫn hoạt động tốt?

Có ba lý do chính:

1. **Thứ hạng quan trọng hơn hiệu chuẩn xác suất (Ranking over calibration).** Bài toán phân loại chỉ yêu cầu lớp xếp đầu tiên là đúng. Dù $P(\text{spam}) = 0.99999$ trong khi xác suất thực tế chỉ là 0.7, bộ phân loại vẫn chọn đúng là spam. Ta không cần xác suất hoàn hảo, ta chỉ cần kẻ chiến thắng chính xác.

2. **Độ chệch cao, phương sai thấp (High bias, low variance).** Giả định độc lập là một giả định tiên nghiệm rất mạnh (strong prior). Nó ràng buộc mô hình chặt chẽ, giúp ngăn chặn hiện tượng quá khớp (overfitting). Khi dữ liệu huấn luyện ít, một mô hình hơi sai về lý thuyết nhưng ổn định sẽ đánh bại một mô hình đúng về lý thuyết nhưng lại biến động dữ dội. Đây chính là minh chứng rõ nét cho sự đánh đổi bias-variance.

3. **Sự dư thừa đặc trưng tự triệt tiêu (Feature redundancy cancels out).** Các đặc trưng tương quan cung cấp bằng chứng dư thừa. Bộ phân loại sẽ đếm lặp bằng chứng này, nhưng nó cũng đếm lặp cho đúng lớp đó. Nếu "machine" và "learning" luôn đi cùng nhau, cả hai đều cung cấp bằng chứng cho lớp "tech". Naive Bayes đếm chúng hai lần, nhưng là đếm hai lần cho đúng lớp đích.

Một lý do thực tế thứ tư: Naive Bayes cực kỳ nhanh. Quá trình huấn luyện chỉ là một lượt quét qua dữ liệu để đếm tần số. Quá trình dự đoán chỉ là một phép nhân ma trận. Bạn có thể huấn luyện trên một triệu tài liệu trong vài giây. Tốc độ này cho phép bạn lặp nhanh hơn, thử nghiệm nhiều bộ đặc trưng hơn và chạy nhiều thử nghiệm hơn so với các mô hình chậm chạp.

### Từng bước tính toán toán học

Hãy cùng theo dõi một ví dụ cụ thể. Giả sử ta có hai lớp: spam và not-spam (hợp lệ). Từ vựng gồm 3 từ: "free", "money", "meeting".

Dữ liệu huấn luyện:
- Email spam chứa từ "free" 80 lần, "money" 60 lần, "meeting" 10 lần (tổng cộng 150 từ)
- Email hợp lệ chứa từ "free" 5 lần, "money" 10 lần, "meeting" 100 lần (tổng cộng 115 từ)
- 40% email là spam, 60% là email hợp lệ

Áp dụng làm mịn Laplace ($\alpha=1$):

```math
\begin{aligned}
P(\text{free} \mid \text{spam}) &= (80 + 1) / (150 + 3) = 81/153 = 0.529 \\
P(\text{money} \mid \text{spam}) &= (60 + 1) / (150 + 3) = 61/153 = 0.399 \\
P(\text{meeting} \mid \text{spam}) &= (10 + 1) / (150 + 3) = 11/153 = 0.072 \\[6pt]
P(\text{free} \mid \text{not-spam}) &= (5 + 1) / (115 + 3) = 6/118 = 0.051 \\
P(\text{money} \mid \text{not-spam}) &= (10 + 1) / (115 + 3) = 11/118 = 0.093 \\
P(\text{meeting} \mid \text{not-spam}) &= (100 + 1) / (115 + 3) = 101/118 = 0.856
\end{aligned}
```

Một email mới chứa: "free" (2 lần), "money" (1 lần), "meeting" (0 lần).

```math
\begin{aligned}
\log P(\text{spam} \mid \text{email}) &= \log(0.4) + 2 \log(0.529) + 1 \log(0.399) + 0 \log(0.072) \\
&= -0.916 + 2(-0.637) + (-0.919) + 0 \\
&= -3.109 \\[6pt]
\log P(\text{not-spam} \mid \text{email}) &= \log(0.6) + 2 \log(0.051) + 1 \log(0.093) + 0 \log(0.856) \\
&= -0.511 + 2(-2.976) + (-2.375) + 0 \\
&= -8.838
\end{aligned}
```

Spam thắng với khoảng cách rất lớn. Từ "free" xuất hiện 2 lần là bằng chứng cực mạnh cho thấy email là spam. Lưu ý rằng từ "meeting" không xuất hiện nên đóng góp 0 vào cả hai tổng log ($0 \cdot \log P$) -- trong Multinomial NB, các từ vắng mặt không ảnh hưởng đến kết quả. Chính biến thể Bernoulli NB mới là mô hình mô hình hóa rõ ràng sự vắng mặt của từ.

### Ba biến thể của Naive Bayes

Naive Bayes có 3 biến thể chính. Mỗi biến thể mô hình hóa `P(feature | class)` theo cách khác nhau.

#### Multinomial Naive Bayes

Mô hình hóa mỗi đặc trưng dưới dạng số lần đếm (count). Thích hợp nhất cho dữ liệu văn bản nơi các đặc trưng là tần số xuất hiện của từ hoặc giá trị TF-IDF.

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{số lần xuất hiện của } \text{word}_i \text{ trong class} + \alpha}{\text{tổng số từ trong class} + \alpha \cdot \text{vocab\_size}}
```

Hệ số `alpha` là làm mịn Laplace (Laplace smoothing, giải thích bên dưới). Biến thể này là trụ cột cho phân loại văn bản.

#### Gaussian Naive Bayes

Mô hình hóa mỗi đặc trưng dưới dạng phân phối chuẩn (normal distribution / Gaussian). Tốt nhất cho các đặc trưng liên tục (continuous features).

```math
P(x_i \mid \text{class}) = \frac{1}{\sqrt{2 \pi \cdot \text{var}}} \cdot \exp\left(-\frac{(x_i - \text{mean})^2}{2 \cdot \text{var}}\right)
```

Mỗi lớp có giá trị trung bình (mean) và phương sai (variance) riêng cho từng đặc trưng. Biến thể này hoạt động rất tốt khi các đặc trưng thực sự tuân theo đường cong hình chuông trong từng lớp.

#### Bernoulli Naive Bayes

Mô hình hóa mỗi đặc trưng dưới dạng nhị phân (có xuất hiện hay không: 1 hoặc 0). Tốt nhất cho các văn bản ngắn hoặc vector đặc trưng nhị phân.

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{số tài liệu trong class chứa } \text{word}_i + \alpha}{\text{tổng số tài liệu trong class} + 2 \alpha}
```

Khác với Multinomial, Bernoulli phạt rõ ràng sự vắng mặt của một từ. Nếu "free" thường xuất hiện trong spam nhưng lại vắng mặt trong email này, Bernoulli sẽ tính đó là bằng chứng chống lại khả năng là spam.

### Khi nào nên dùng từng biến thể

| Biến thể | Kiểu đặc trưng | Tốt nhất cho | Ví dụ |
|---------|-------------|----------|---------|
| Multinomial | Số lần đếm hoặc tần số | Phân loại văn bản, bag-of-words | Lọc thư rác email, phân loại chủ đề |
| Gaussian | Giá trị liên tục | Dữ liệu dạng bảng với các đặc trưng gần phân phối chuẩn | Phân loại hoa Iris, dữ liệu cảm biến |
| Bernoulli | Nhị phân (0/1) | Văn bản ngắn, vector đặc trưng nhị phân | Lọc SMS rác, đặc trưng xuất hiện/vắng mặt |

### Làm mịn Laplace (Laplace Smoothing)

Điều gì xảy ra khi một từ xuất hiện trong tập kiểm thử (test data) nhưng chưa từng xuất hiện trong tập huấn luyện (training data) của một lớp cụ thể nào đó?

Nếu không có làm mịn: `P(word | class) = 0/N = 0`. Một số 0 nhân vào toàn bộ tích sẽ khiến `P(class | features) = 0`, bất kể mọi bằng chứng khác ủng hộ mạnh đến đâu. Một từ đơn lẻ chưa từng thấy sẽ phá hủy toàn bộ dự đoán.

Làm mịn Laplace bổ sung một lượng đếm nhỏ `alpha` (thường là 1) vào số đếm của mọi đặc trưng:

```math
P(\text{word}_i \mid \text{class}) = \frac{\text{count}(\text{word}_i, \text{class}) + \alpha}{\text{total\_words\_in\_class} + \alpha \cdot \text{vocab\_size}}
```

Với $\alpha=1$, mọi từ đều nhận được một xác suất nhỏ tối thiểu. Một từ lạ lẫm xuất hiện trong email kiểm thử sẽ không còn kéo xác suất spam về 0 nữa. Việc làm mịn có cách giải thích theo trường phái Bayes: nó tương đương với việc gán một phân phối tiên nghiệm Dirichlet đều (uniform Dirichlet prior) lên các phân phối từ vựng.

Alpha càng lớn nghĩa là làm mịn càng mạnh (các phân phối phẳng hơn, đều hơn). Alpha càng nhỏ nghĩa là mô hình tin tưởng vào dữ liệu nhiều hơn. Alpha là một siêu tham số mà bạn có thể tinh chỉnh.

Tác động của alpha:

| Alpha | Tác động | Khi nào nên dùng |
|-------|--------|-------------|
| 0.001 | Hầu như không làm mịn, tin tưởng hoàn toàn dữ liệu | Tập huấn luyện rất lớn, không lo gặp đặc trưng chưa thấy |
| 0.1 | Làm mịn nhẹ | Tập huấn luyện lớn |
| 1.0 | Làm mịn Laplace tiêu chuẩn | Điểm khởi đầu mặc định |
| 10.0 | Làm mịn mạnh, làm phẳng phân phối | Tập huấn luyện rất nhỏ, dự kiến gặp nhiều đặc trưng lạ |

### Tính toán trong không gian Log (Log-Space Computation)

Việc nhân hàng trăm xác suất nhỏ (mỗi số < 1) với nhau sẽ gây ra hiện tượng tràn số dưới dấu phẩy động (floating-point underflow). Tích số sẽ trở thành 0 trong máy tính mặc dù giá trị thực là một số dương cực nhỏ.

Giải pháp: làm việc trong không gian log (log space). Thay vì nhân các xác suất, ta cộng logarit của chúng:

```math
\log P(\text{class} \mid x_1, x_2, \ldots, x_n) = \log P(\text{class}) + \sum_i \log P(x_i \mid \text{class})
```

Điều này biến phép tính dự đoán thành tích vô hướng (dot product):

```python
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Một phép nhân ma trận đơn giản! Đó là lý do vì sao dự đoán Naive Bayes lại nhanh đến vậy -- về bản chất nó chính là phép toán của một mô hình tuyến tính đơn tầng.

### Naive Bayes vs Hồi quy Logistic

Cả hai đều là bộ phân loại tuyến tính cho văn bản. Sự khác biệt nằm ở cách chúng mô hình hóa dữ liệu.

| Khía cạnh | Naive Bayes | Hồi quy Logistic |
|--------|------------|-------------------|
| Kiểu mô hình | Sinh (Generative - mô hình hóa $P(X \mid Y){}$) | Phân biệt (Discriminative - mô hình hóa $P(Y \mid X){}$) |
| Huấn luyện | Đếm tần suất | Tối ưu hóa hàm mất mát (loss function) |
| Dữ liệu ít | Tốt hơn (tiên nghiệm mạnh phát huy tác dụng) | Kém hơn (không đủ mẫu để ước lượng trọng số) |
| Dữ liệu nhiều | Kém hơn (giả định sai kìm hãm mô hình) | Tốt hơn (ranh giới quyết định linh hoạt) |
| Đặc trưng | Giả định độc lập | Xử lý tốt các đặc trưng tương quan |
| Tốc độ | Quét 1 lượt, cực nhanh | Tối ưu hóa lặp (iterative optimization) |
| Hiệu chuẩn | Xác suất kém hiệu chuẩn | Xác suất được hiệu chuẩn tốt hơn |

Quy tắc kinh nghiệm: hãy bắt đầu với Naive Bayes. Nếu bạn có nhiều dữ liệu và hiệu năng của NB chững lại, hãy chuyển sang hồi quy Logistic.

### Pipeline phân loại văn bản (Classification Pipeline)

```mermaid
flowchart LR
    A[Văn bản thô] --> B[Tokenize tách từ]
    B --> C[Xây dựng từ vựng]
    C --> D[Đếm tần suất từ]
    D --> E[Áp dụng làm mịn]
    E --> F[Tính log xác suất]
    F --> G[Dự đoán: argmax P class khi biết từ]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

Trong thực tế, ta làm việc trong không gian log để tránh underflow số thực dấu phẩy động. Thay vì nhân nhiều xác suất nhỏ, ta cộng các giá trị logarit:

```math
\log P(\text{class} \mid \text{features}) = \log P(\text{class}) + \sum_i \log P(\text{feature}_i \mid \text{class})
```

```figure
naive-bayes
```

## Tự tay xây dựng (Build It)

Mã nguồn trong `code/naive_bayes.py` triển khai cả MultinomialNB và GaussianNB từ đầu.

### MultinomialNB

Các bước triển khai từ đầu:

1. **fit(X, y)**: Với mỗi lớp, đếm tần số của từng đặc trưng. Thêm làm mịn Laplace. Tính log xác suất. Lưu xác suất tiên nghiệm của lớp (log tần số lớp).

2. **predict_log_proba(X)**: Với mỗi mẫu dữ liệu, tính $\log P(\text{class})$ + tổng các $\log P(\text{feature}_i \mid \text{class})$ cho tất cả các lớp. Đây là một phép nhân ma trận: `X @ log_probs.T + log_priors`.

3. **predict(X)**: Trả về lớp có log xác suất cao nhất.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

Điểm cốt lõi: sau khi fit, việc dự đoán chỉ là nhân ma trận cộng thêm độ chệch (bias). Đó là lý do tại sao Naive Bayes chạy nhanh như vậy.

### GaussianNB

Đối với các đặc trưng liên tục, ta ước lượng giá trị trung bình và phương sai cho mỗi lớp trên từng đặc trưng:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Dự đoán sử dụng hàm mật độ xác suất Gaussian (PDF) cho từng đặc trưng, nhân trên toàn bộ các đặc trưng (cộng trong không gian log).

### Thử nghiệm: Phân loại văn bản

Mã nguồn sinh dữ liệu mô hình túi từ (bag-of-words) giả lập hai lớp (bài báo công nghệ vs bài báo thể thao). Mỗi lớp có phân phối tần suất từ khác nhau. MultinomialNB phân loại chúng dựa trên số lần đếm từ.

Dữ liệu giả lập hoạt động như sau: ta tạo 200 "từ" (các cột đặc trưng). Từ 0-39 có tần số cao trong bài viết công nghệ và thấp trong thể thao. Từ 80-119 có tần số cao trong bài viết thể thao và thấp trong công nghệ. Từ 40-79 có tần số trung bình ở cả hai. Điều này tạo ra một kịch bản thực tế nơi một số từ là dấu hiệu nhận biết mạnh mẽ còn các từ khác là nhiễu.

### Thử nghiệm: Đặc trưng liên tục

Mã nguồn tạo dữ liệu dạng Iris (3 lớp, 4 đặc trưng, các cụm Gaussian). GaussianNB phân loại bằng giá trị trung bình và phương sai của từng lớp. Mỗi lớp có tâm cụm (vector mean) và độ phân tán (variance) khác nhau, mô phỏng dữ liệu thế giới thực nơi các phép đo khác biệt có hệ thống giữa các danh mục.

Mã nguồn cũng minh họa:
- **So sánh độ làm mịn:** Huấn luyện MultinomialNB với các giá trị alpha khác nhau để thấy tác động của độ mạnh làm mịn lên độ chính xác.
- **Thử nghiệm quy mô dữ liệu huấn luyện:** Độ chính xác của NB cải thiện như thế nào khi tập huấn luyện tăng từ 20 lên 1600 mẫu. NB đạt độ chính xác khá tốt ngay cả với rất ít mẫu -- đây là ưu thế lớn nhất của nó.
- **Ma trận nhầm lẫn (Confusion matrix):** Precision, recall và điểm F1 theo từng lớp để thấy rõ NB mắc sai lầm ở đâu.

### Tốc độ dự đoán

Dự đoán Naive Bayes là một phép nhân ma trận. Cho $n$ mẫu với $d$ đặc trưng và $k$ lớp:
- MultinomialNB: một phép nhân ma trận $(n \times d) \mathbin{@} (d \times k) = O(n \cdot d \cdot k)$
- GaussianNB: $n \cdot k$ lần tính Gaussian PDF, mỗi lần trên $d$ đặc trưng = $O(n \cdot d \cdot k)$

Cả hai đều có độ phức tạp tuyến tính theo mọi chiều. So sánh điều này với KNN (đòi hỏi tính khoảng cách tới toàn bộ các điểm huấn luyện) hoặc SVM với RBF kernel (đòi hỏi tính kernel với tất cả các support vector). NB nhanh hơn nhiều bậc độ lớn tại thời điểm dự đoán.

## Sử dụng trong thực tế (Use It)

Với scikit-learn, cả hai biến thể đều chỉ cần vài dòng:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Phân loại văn bản với scikit-learn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

Mã nguồn trong `naive_bayes.py` so sánh triển khai tự viết với scikit-learn trên cùng tập dữ liệu để kiểm chứng độ chính xác.

### TF-IDF kết hợp Naive Bayes

Số lượng từ thô gán trọng số ngang nhau cho mỗi lần xuất hiện của từ. Nhưng những từ thông dụng như "the", "là", "và" xuất hiện thường xuyên trong mọi lớp -- chúng không mang nhiều thông tin phân loại. TF-IDF (Term Frequency - Inverse Document Frequency) giảm trọng số của những từ phổ biến và tăng trọng số của những từ hiếm, có tính phân biệt cao.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Các giá trị TF-IDF luôn không âm, vì vậy chúng hoạt động hoàn hảo với MultinomialNB. Sự kết hợp TF-IDF + MultinomialNB là một trong những baseline mạnh mẽ nhất cho phân loại văn bản, thường xuyên đánh bại các mô hình phức tạp hơn trên các bộ dữ liệu có dưới 10.000 mẫu huấn luyện.

### BernoulliNB cho văn bản ngắn

Đối với văn bản ngắn (tweet, tin nhắn SMS, tin nhắn chat), BernoulliNB có thể vượt trội hơn MultinomialNB. Các đoạn văn ngắn có số lượng từ ít, do đó thông tin tần số mà MultinomialNB dựa vào rất nhiễu. BernoulliNB chỉ quan tâm đến sự xuất hiện hay vắng mặt, điều này đáng tin cậy hơn trên văn bản ngắn.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

Cờ `binary=True` trong CountVectorizer chuyển tất cả số đếm thành 0 hoặc 1.

### Hiệu chuẩn xác suất của Naive Bayes (Calibrating NB Probabilities)

Xác suất do Naive Bayes sinh ra thường bị hiệu chuẩn kém. Khi NB nói $P(\text{spam}) = 0.95$, xác suất thực tế có thể chỉ là 0.7. Nếu bạn cần các ước lượng xác suất đáng tin cậy (ví dụ để đặt ngưỡng quyết định hoặc kết hợp với các mô hình khác), hãy dùng `CalibratedClassifierCV` của scikit-learn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Phương pháp này khớp một mô hình hồi quy Logistic lên trên các điểm số thô của NB bằng kiểm thực chéo (cross-validation). Các xác suất thu được sẽ gần hơn nhiều với tần suất thực của các lớp.

### Các bẫy thường gặp (Common Gotchas)

1. **Giá trị đặc trưng âm.** MultinomialNB yêu cầu các đặc trưng không âm. Nếu bạn có giá trị âm (như chuẩn hóa chuẩn z-score), hãy dùng GaussianNB hoặc tịnh tiến các đặc trưng sang miền dương.

2. **Đặc trưng có phương sai bằng 0.** GaussianNB chia cho phương sai. Nếu một đặc trưng có phương sai bằng 0 ở một lớp nào đó (mọi giá trị đều y hệt nhau), phép tính xác suất sẽ bị lỗi chia cho 0. Mã nguồn khắc phục bằng cách cộng một lượng làm mịn nhỏ (`1e-9`) vào tất cả phương sai.

3. **Mất cân bằng lớp (Class imbalance).** Nếu 99% email không phải là spam, tiên nghiệm $P(\text{not-spam}) = 0.99$ mạnh đến mức lấn át toàn bộ bằng chứng từ likelihood. Bạn có thể thiết lập tiên nghiệm thủ công hoặc dùng tham số `class_prior` trong scikit-learn.

4. **Co giãn đặc trưng (Feature scaling).** MultinomialNB không cần co giãn (nó hoạt động trên số lần đếm). GaussianNB cũng không cần co giãn (nó ước lượng các đại lượng thống kê riêng cho từng đặc trưng). Đây là một ưu điểm lớn so với Logistic Regression và SVM.

## Xuất xưởng (Ship It)

Bài học này tạo ra:
- `outputs/skill-naive-bayes-chooser.md` -- kỹ năng ra quyết định để chọn biến thể NB phù hợp
- `code/naive_bayes.py` -- MultinomialNB và GaussianNB tự xây dựng từ đầu, kèm so sánh với scikit-learn

### Khi nào Naive Bayes thất bại?

NB thất bại khi giả định độc lập dẫn đến thứ hạng sai lệch (chứ không chỉ là xác suất sai lệch). Điều này xảy ra khi:

1. **Tương tác đặc trưng mạnh mẽ.** Nếu nhãn lớp phụ thuộc vào sự kết hợp của hai đặc trưng nhưng không phụ thuộc vào từng đặc trưng đơn lẻ (dạng bài toán XOR), NB sẽ bỏ sót hoàn toàn. Mỗi đặc trưng đứng một mình không cung cấp bằng chứng nào, và NB không thể kết hợp chúng phi tuyến tính.

2. **Các đặc trưng tương quan mạnh mang bằng chứng mâu thuẫn.** Nếu đặc trưng A nói "spam" và đặc trưng B nói "hợp lệ", nhưng A và B lại tương quan hoàn hảo (trong thực tế chúng luôn đi cùng nhau), NB sẽ nhìn nhận đó là hai bằng chứng mâu thuẫn trong khi thực tế không phải vậy.

3. **Tập dữ liệu huấn luyện rất lớn.** Khi có đủ dữ liệu, các mô hình phân biệt như hồi quy Logistic sẽ học được ranh giới quyết định thực sự và vượt trội hơn NB. Giả định độc lập vốn hữu ích khi dữ liệu ít giờ đây lại kìm hãm mô hình.

Trong thực tế, các kiểu thất bại này khá hiếm gặp trong bài toán phân loại văn bản. Đặc trưng văn bản rất nhiều, yếu khi đứng riêng rẽ, và sai số của giả định độc lập có xu hướng triệt tiêu lẫn nhau. Với dữ liệu bảng (tabular data) có ít đặc trưng nhưng tương quan mạnh, hãy cân nhắc hồi quy Logistic hoặc các mô hình cây quyết định trước tiên.

## Bài tập thực hành (Exercises)

1. **Thử nghiệm làm mịn.** Huấn luyện MultinomialNB trên dữ liệu văn bản với các giá trị alpha: 0.01, 0.1, 1.0, 10.0 và 100.0. Vẽ đồ thị độ chính xác theo alpha. Điểm tối ưu nằm ở đâu? Vì sao alpha quá cao lại làm giảm hiệu năng?

2. **Kiểm tra tính độc lập của đặc trưng.** Lấy một tập dữ liệu văn bản thực tế. Chọn hai từ rõ ràng có tương quan (ví dụ: "machine" và "learning"). Tính $P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class})$ và so sánh với $P(\text{word}_1 \text{ VÀ } \text{word}_2 \mid \text{class})$. Giả định độc lập sai lệch bao nhiêu? Nó có ảnh hưởng đến độ chính xác phân loại không?

3. **Tự cài đặt Bernoulli.** Mở rộng mã nguồn với lớp `BernoulliNB`. Chuyển đổi bag-of-words sang nhị phân (có/không xuất hiện) và so sánh độ chính xác với MultinomialNB trên văn bản. Khi nào Bernoulli chiến thắng?

4. **So sánh NB vs Hồi quy Logistic.** Huấn luyện cả hai trên dữ liệu văn bản. Bắt đầu với 100 mẫu huấn luyện và tăng dần lên 10.000. Vẽ đồ thị độ chính xác theo kích thước tập huấn luyện cho cả hai. Tại thời điểm nào Hồi quy Logistic bắt đầu vượt qua Naive Bayes?

5. **Bộ lọc Spam hoàn chỉnh.** Xây dựng một bộ phân loại spam hoàn chỉnh: tách từ email thô, tạo từ vựng, tạo đặc trưng bag-of-words, huấn luyện MultinomialNB, đánh giá bằng precision và recall (tại sao không nên chỉ nhìn vào accuracy?).

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Naive Bayes | "Bộ phân loại xác suất đơn giản" | Bộ phân loại áp dụng định lý Bayes với giả định rằng các đặc trưng độc lập có điều kiện khi biết nhãn lớp |
| Conditional independence | "Các đặc trưng không ảnh hưởng lẫn nhau" | $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$ -- biết $B$ không cho bạn thêm thông tin gì mới về $A$ một khi đã biết $C$ |
| Laplace smoothing | "Làm mịn cộng một" | Thêm một lượng đếm nhỏ vào mỗi đặc trưng để tránh xác suất bằng 0 chi phối toàn bộ tích dự đoán |
| Prior | "Niềm tin trước khi thấy dữ liệu" | $P(\text{class})$ -- xác suất của mỗi lớp trước khi quan sát bất kỳ đặc trưng nào |
| Likelihood | "Dữ liệu phù hợp đến mức nào" | $P(\text{features} \mid \text{class})$ -- xác suất quan sát được các đặc trưng này nếu biết trước lớp |
| Posterior | "Niềm tin sau khi thấy dữ liệu" | $P(\text{class} \mid \text{features})$ -- xác suất cập nhật của lớp sau khi đã quan sát các đặc trưng |
| Generative model | "Mô hình cách dữ liệu được sinh ra" | Mô hình học phân phối $P(X \mid Y)$ và $P(Y)$, sau đó dùng định lý Bayes để suy ra $P(Y \mid X)$ |
| Discriminative model | "Mô hình ranh giới quyết định" | Mô hình trực tiếp học phân phối $P(Y \mid X)$ mà không cần mô hình hóa cách $X$ được sinh ra |
| Log probability | "Tránh tràn số dưới" | Làm việc với $\log P$ thay vì $P$ để tránh tích của nhiều số nhỏ bị làm tròn thành 0 trong số thực dấu phẩy động |

## Tài liệu đọc thêm (Further Reading)

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- cả 3 biến thể kèm chi tiết toán học
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- bài báo kinh điển so sánh Multinomial vs Bernoulli cho văn bản
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- các cải tiến cho Naive Bayes trong phân loại văn bản
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- chứng minh Naive Bayes hội tụ nhanh hơn Hồi quy Logistic khi có ít dữ liệu
