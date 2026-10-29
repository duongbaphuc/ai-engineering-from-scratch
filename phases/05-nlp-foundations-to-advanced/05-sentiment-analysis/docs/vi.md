# Phân tích Cảm xúc (Sentiment Analysis)

> Tác vụ kinh điển của NLP. Hầu hết những gì bạn cần biết về phân loại văn bản cổ điển (classical text classification) đều xuất hiện ở đây.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

"Món ăn không ngon lắm." Tích cực hay tiêu cực?

Phân tích cảm xúc nghe có vẻ đơn giản. Người đánh giá nói họ thích hoặc không thích một thứ gì đó. Gán nhãn cho câu văn. Lý do nó trở thành tác vụ kinh điển của NLP là vì mọi trường hợp trông có vẻ dễ dàng đều ẩn chứa một ca hóc búa. Phủ định làm đảo ngược ý nghĩa. Châm biếm mỉa mai (sarcasm) đảo lộn toàn bộ ngữ cảnh. "Không tệ chút nào" thực chất là tích cực dù chứa hai từ mang sắc thái phủ định/tiêu cực. Emoji mang nhiều tín hiệu hơn cả văn bản xung quanh. Từ vựng theo miền (domain vocabulary) đóng vai trò quyết định (`tight` trong đánh giá âm nhạc có nghĩa là "chặt chẽ/tuyệt vời", còn trong đánh giá thời trang có nghĩa là "chật chội").

Phân tích cảm xúc là một phòng thí nghiệm thực tế cho NLP cổ điển. Nếu bạn hiểu tại sao mỗi mô hình cơ sở ngây thơ (naive baseline) đều có một phương thức thất bại (failure mode) cụ thể, bạn sẽ hiểu tại sao các mô hình phong phú hơn được phát minh. Bài học này xây dựng một mô hình cơ sở Naive Bayes từ đầu, bổ sung hồi quy logistic (logistic regression), và điểm tên các cạm bẫy biến phân tích cảm xúc trong môi trường sản xuất thành bài toán đòi hỏi tiêu chuẩn tuân thủ nghiêm ngặt.

## Khái niệm (The Concept)

Phân tích cảm xúc cổ điển là một công thức gồm hai bước:

1. **Biểu diễn (Represent).** Chuyển văn bản thành một vector đặc trưng (feature vector). BoW, TF-IDF, hoặc n-grams.
2. **Phân loại (Classify).** Khớp một mô hình tuyến tính (Naive Bayes, logistic regression, SVM) trên các mẫu dữ liệu đã gán nhãn.

Naive Bayes là mô hình đơn giản nhất nhưng vẫn hoạt động hiệu quả. Giả định rằng mọi đặc trưng đều độc lập có điều kiện khi biết nhãn. Ước tính `P(từ | tích cực)` và `P(từ | tiêu cực)` từ tần số đếm. Khi suy luận, nhân các xác suất lại với nhau. Giả định độc lập "ngây thơ" (naive) này thực tế hoàn toàn sai lệch về mặt ngôn ngữ, nhưng kết quả mang lại lại tốt đến kinh ngạc. Lý do: với các đặc trưng văn bản thưa thớt (sparse text features) và lượng dữ liệu vừa phải, bộ phân loại chỉ cần quan tâm từ đó nghiêng về phía nào hơn là mức độ nghiêng chính xác là bao nhiêu.

Hồi quy logistic (Logistic regression) khắc phục giả định độc lập này. Nó học một trọng số cho mỗi đặc trưng, bao gồm cả trọng số âm. `not good` dưới dạng đặc trưng bigram sẽ nhận được một trọng số âm. Naive Bayes không thể làm được điều đó đối với các bigram mà nó chưa từng thấy nhãn trong tập huấn luyện.

```figure
sentiment-logits
```

## Xây dựng thực tế (Build It)

### Bước 1: Tập dữ liệu nhỏ thực tế

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Tập dữ liệu này cố ý để nhỏ gọn. Trong thực tế, các bài toán sử dụng hàng chục nghìn ví dụ (IMDb, SST-2, Yelp polarity). Bản chất toán học là hoàn toàn đồng nhất.

### Bước 2: Multinomial Naive Bayes từ đầu

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Làm mịn cộng tính (Additive smoothing với `alpha=1.0`) chính là làm mịn Laplace (Laplace smoothing). Nếu không có nó, một từ chưa từng xuất hiện trong một lớp sẽ có xác suất bằng 0 và giá trị hàm logarit sẽ bùng nổ (về âm vô cực). `alpha=0.01` thường dùng trong thực tế. `alpha=1.0` là giá trị mặc định cho mục đích giảng dạy.

### Bước 3: Hồi quy Logistic từ đầu

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

Điều chuẩn L2 (L2 regularization) đóng vai trò sống còn ở đây. Các đặc trưng văn bản rất thưa thớt; nếu không có L2, mô hình sẽ học vẹt (ghi nhớ) các mẫu huấn luyện. Hãy bắt đầu ở mức `0.01` và tinh chỉnh dần.

### Bước 4: Xử lý phủ định (trường hợp dễ gây lỗi nhất)

Xem xét hai cụm từ "not good" và "not bad". Bộ phân loại BoW chỉ nhìn thấy `{not, good}` và `{not, bad}`, rồi học từ bất kỳ từ nào xuất hiện nhiều hơn trong tập huấn luyện. Một bộ phân loại bigram sẽ nhìn thấy `not_good` và `not_bad` và học chúng thành các đặc trưng riêng biệt. Điều đó thường là đủ.

Một giải pháp thô sơ hơn nhưng hiệu quả khi bạn không có bigram: **xác định phạm vi phủ định (negation scoping)**. Thêm tiền tố `NOT_` vào các token đứng sau một từ phủ định cho đến khi gặp dấu câu tiếp theo.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Bây giờ `good` và `NOT_good` là hai đặc trưng hoàn toàn khác nhau. Bộ phân loại có thể gán cho chúng trọng số trái ngược. Chỉ ba dòng tiền xử lý nhưng tạo ra bước nhảy vọt có thể đo lường được về độ chính xác trên các bài đo cảm xúc benchmark.

### Bước 5: Các chỉ số đánh giá quan trọng

Độ chính xác (Accuracy) đơn thuần sẽ gây hiểu lầm nghiêm trọng nếu các lớp bị mất cân bằng (imbalanced). Các tập ngữ liệu cảm xúc thực tế thường có 70-80% tích cực hoặc 70-80% tiêu cực; một bộ phân loại luôn dự đoán lớp đa số sẽ đạt 80% accuracy nhưng hoàn toàn vô giá trị. Hãy báo cáo đầy đủ tất cả các chỉ số sau:

- **Precision và Recall theo từng lớp (Per-class precision and recall).** Một cặp giá trị cho mỗi lớp. Lấy trung bình vĩ mô (macro-average) để có được một con số duy nhất tôn trọng sự cân bằng giữa các lớp.
- **Macro-F1 (chỉ số chính cho dữ liệu mất cân bằng).** Giá trị trung bình của điểm F1 từng lớp, được đánh trọng số bình đẳng. Dùng chỉ số này thay cho accuracy khi dữ liệu mất cân bằng.
- **Weighted-F1 (phương án thay thế).** Tương tự macro nhưng được đánh trọng số theo tần suất xuất hiện của lớp. Báo cáo song song với macro-F1 khi sự mất cân bằng mang ý nghĩa kinh doanh.
- **Ma trận nhầm lẫn (Confusion matrix).** Tần số đếm thô. Luôn kiểm tra ma trận này trước khi tin tưởng bất kỳ chỉ số vô hướng nào; nó cho thấy cặp lớp nào đang bị mô hình nhầm lẫn.
- **Mẫu lỗi theo từng lớp (Per-class error samples).** Trích xuất 5 dự đoán sai cho mỗi lớp. Hãy trực tiếp đọc chúng. Không gì thay thế được việc đọc các lỗi thực tế.

Đối với dữ liệu mất cân bằng nghiêm trọng (tỷ lệ > 95-5), hãy báo cáo **AUROC** và **AUPRC** thay vì accuracy. AUPRC nhạy cảm hơn đối với lớp thiểu số, vốn là điều bạn thường quan tâm nhất (thư rác, gian lận, cảm xúc hiếm).

**Lỗi phổ biến cần tránh:** Báo cáo micro-F1 thay vì macro-F1 trên dữ liệu mất cân bằng sẽ cho ra con số trông rất cao vì nó bị chi phối bởi lớp đa số. Macro-F1 buộc bạn phải nhìn thẳng vào hiệu năng của lớp thiểu số.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Sử dụng trong thực tế (Use It)

scikit-learn thực hiện điều này chuẩn xác chỉ trong sáu dòng mã.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Ba điểm cần lưu ý: `stop_words=None` giữ lại các từ phủ định. `ngram_range=(1, 2)` bổ sung các bigram để `not good` trở thành một đặc trưng (scikit-learn nối hai từ bằng dấu cách). `sublinear_tf=True` làm giảm bớt sức ảnh hưởng của các từ lặp lại nhiều lần. Ba cờ tùy chọn này tạo ra sự khác biệt giữa một mô hình baseline 75% độ chính xác và một mô hình 85% trên SST-2.

### Khi nào nên dùng transformer

- Phát hiện châm biếm / mỉa mai (Sarcasm detection). Các mô hình cổ điển hoàn toàn thất bại ở đây.
- Các bài đánh giá dài mà cảm xúc biến chuyển liên tục ở giữa tài liệu.
- Phân tích cảm xúc theo khía cạnh (Aspect-based sentiment). "Máy ảnh rất tuyệt nhưng pin quá tệ." Bạn cần gán cảm xúc cho từng khía cạnh cụ thể. Chỉ transformer hoặc mô hình đầu ra có cấu trúc (structured output) mới xử lý được.
- Ngôn ngữ ít tài nguyên, phi tiếng Anh. Multilingual BERT cung cấp cho bạn một baseline zero-shot miễn phí.

Nếu bạn cần bất kỳ điều nào ở trên, hãy chuyển tiếp sang phase 7 (deep dive về transformer). Nếu không, Naive Bayes hoặc logistic regression kết hợp TF-IDF cùng bigram và xử lý phủ định vẫn là mô hình baseline chuẩn mực trong môi trường sản xuất năm 2026.

### Cạm bẫy về khả năng tái lập (The reproducibility trap)

Việc huấn luyện lại (retraining) các mô hình cảm xúc là công việc thường nhật. Nhưng đánh giá lại (re-evaluating) chúng thì không. Các con số độ chính xác được báo cáo trong các bài báo nghiên cứu sử dụng các phép chia tập dữ liệu cụ thể, tiền xử lý cụ thể, bộ tách từ cụ thể. Nếu bạn so sánh mô hình mới của mình với một baseline mà không sử dụng cùng một pipeline giống hệt, bạn sẽ nhận được những mức chênh lệch sai lệch. Luôn luôn tái tạo lại baseline trên chính pipeline của bạn, đừng lấy nguyên số liệu từ bài báo.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Thiết kế baseline phân tích cảm xúc cho một tập dữ liệu mới.
phase: 5
lesson: 05
---

Khi nhận được mô tả tập dữ liệu (miền dữ liệu, ngôn ngữ, kích thước, độ chi tiết của nhãn, ngân sách độ trễ), bạn đưa ra:

1. Công thức trích xuất đặc trưng. Chỉ định rõ bộ tách từ (tokenizer), dải n-gram, chính sách từ dừng (stopword - thường giữ lại), cách xử lý phủ định (tiền tố có phạm vi hoặc bigram).
2. Bộ phân loại. Naive Bayes cho baseline, logistic regression cho production, transformer chỉ khi miền dữ liệu cần xử lý châm biếm / khía cạnh / đa ngữ.
3. Kế hoạch đánh giá. Báo cáo precision, recall, F1, ma trận nhầm lẫn và các mẫu lỗi theo từng lớp (không chỉ các chỉ số vô hướng).
4. Một phương thức lỗi cần giám sát sau triển khai. Trôi dạt miền dữ liệu (domain drift) và châm biếm mỉa mai là hai vấn đề hàng đầu.

Từ chối đề xuất loại bỏ từ dừng đối với các tác vụ cảm xúc. Từ chối chỉ báo cáo độ chính xác (accuracy) khi các lớp bị mất cân bằng (ví dụ: 90% tích cực). Đánh dấu các ngôn ngữ giàu hình thái từ phụ (subword-rich) là cần FastText hoặc embedding transformer thay vì TF-IDF cấp độ từ.
```

## Bài tập (Exercises)

1. **Dễ.** Thêm `apply_negation` làm bước tiền xử lý trong pipeline scikit-learn và đo mức chênh lệch F1 delta trên một tập dữ liệu cảm xúc nhỏ.
2. **Trung bình.** Cài đặt hồi quy logistic có trọng số lớp (class-weighted logistic regression - truyền `class_weight="balanced"` vào scikit-learn, hoặc tự bạn tính toán đạo hàm gradient). Đo lường hiệu ứng trên tập dữ liệu nhân tạo mất cân bằng tỷ lệ 90-10.
3. **Khó.** Xây dựng bộ phát hiện châm biếm bằng cách huấn luyện một bộ phân loại thứ hai trên các phần dư (residuals) của mô hình cảm xúc. Ghi chép tài liệu thiết lập thực nghiệm của bạn. Hãy cảnh báo người đọc khi độ chính xác của bạn thấp hơn mức ngẫu nhiên (mức ngẫu nhiên trên phân loại châm biếm 2 lớp là ~50%, và hầu hết các lần thử đầu tiên đều rơi vào khoảng đó).

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Cực tính (Polarity) | Tích cực hoặc tiêu cực | Nhãn nhị phân; đôi khi mở rộng sang trung tính hoặc đa mức chi tiết (đánh giá 5 sao). |
| Cảm xúc theo khía cạnh (Aspect-based sentiment) | Cực tính theo từng khía cạnh | Gán cảm xúc cho các thực thể hoặc thuộc tính cụ thể được đề cập trong văn bản. |
| Phạm vi phủ định (Negation scoping) | Đảo ngược các token lân cận | Thêm tiền tố `NOT_` vào các token đứng sau "not" cho đến khi gặp dấu câu. |
| Làm mịn Laplace (Laplace smoothing) | Cộng thêm 1 vào số đếm | Ngăn chặn các đặc trưng có xác suất bằng 0 trong Naive Bayes. |
| Điều chuẩn L2 (L2 regularization) | Thu nhỏ các trọng số | Thêm `lambda * sum(w^2)` vào hàm mất mát. Cực kỳ cần thiết cho các đặc trưng văn bản thưa thớt. |

## Đọc thêm (Further Reading)

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — khảo sát nền tảng. Khá dài, nhưng bốn phần đầu tiên bao quát toàn bộ NLP cổ điển.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — bài báo chứng minh rằng bigram + Naive Bayes rất khó bị đánh bại trên văn bản ngắn.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — tài liệu tham khảo cho `CountVectorizer`, `TfidfVectorizer` và mọi tham số bạn sẽ tinh chỉnh.

