# Mô hình hóa Chủ đề — LDA và BERTopic (Topic Modeling — LDA and BERTopic)

> LDA: tài liệu là sự pha trộn của các chủ đề, chủ đề là phân phối xác suất trên các từ vựng. BERTopic: tài liệu tụ lại thành các cụm trong không gian nhúng, các cụm đó chính là chủ đề. Cùng một mục tiêu, hai cách tiếp cận giải cấu trúc khác nhau.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Bạn có 10.000 phiếu hỗ trợ khách hàng (tickets), 50.000 bài báo tin tức, hoặc 200.000 bài đăng trên mạng xã hội. Bạn cần hiểu kho tài liệu này nói về những nội dung gì mà không thể đọc thủ công từng bài một. Bạn không có sẵn các nhãn danh mục phân loại. Bạn thậm chí còn không biết có bao nhiêu danh mục đang tồn tại.

Mô hình hóa chủ đề (Topic modeling) giải quyết bài toán đó mà không cần bất kỳ sự giám sát nào (unsupervised). Cung cấp cho nó một kho ngữ liệu, bạn nhận lại một tập hợp nhỏ các chủ đề mạch lạc và đối với mỗi tài liệu, một phân phối xác suất trên các chủ đề đó.

Hai họ thuật toán chi phối lĩnh vực này: LDA (2003) coi mỗi tài liệu là sự pha trộn của các chủ đề tiềm ẩn (latent topics) và mỗi chủ đề là một phân phối xác suất trên toàn bộ từ vựng. Quá trình suy luận dựa trên xác suất Bayes. Nó vẫn được triển khai rộng rãi trong sản xuất khi bạn cần gán chủ đề đa thành viên (mixed-membership) và cần các phân phối xác suất cấp độ từ minh bạch, có thể giải thích được.

BERTopic (2020) mã hóa tài liệu bằng BERT, giảm số chiều bằng UMAP, phân cụm bằng HDBSCAN, và trích xuất các từ khóa đặc trưng cho chủ đề thông qua kỹ thuật class-based TF-IDF. Nó chiếm ưu thế tuyệt đối trên văn bản ngắn, mạng xã hội và bất kỳ nội dung nào mà sự tương đồng ngữ nghĩa quan trọng hơn sự trùng lặp từ vựng bề mặt. Mỗi tài liệu được gán cho một chủ đề duy nhất, đây là một hạn chế đối với các tài liệu dài bao hàm nhiều nội dung.

Bài học này xây dựng trực giác cho cả hai phương pháp và chỉ rõ khi nào nên chọn phương pháp nào cho một kho dữ liệu cụ thể.

## Khái niệm (The Concept)

![Mô hình hỗn hợp LDA so với phân cụm BERTopic](../assets/topic-modeling.svg)

**Quy trình tạo sinh của LDA (Generative Story).** Mỗi chủ đề là một phân phối xác suất trên các từ. Mỗi tài liệu là một hỗn hợp pha trộn của các chủ đề. Để tạo ra một từ trong tài liệu, lấy mẫu một chủ đề từ hỗn hợp của tài liệu đó, rồi lấy mẫu một từ từ phân phối của chủ đề đó. Quá trình suy luận thực hiện ngược lại: Cho trước các từ quan sát được trong văn bản, suy diễn ra phân phối chủ đề cho mỗi tài liệu và phân phối từ vựng cho mỗi chủ đề. Thuật toán lấy mẫu Gibbs suy biến (Collapsed Gibbs sampling) hoặc biến phân Bayes (Variational Bayes) giải quyết bài toán toán học này.

Đầu ra chính yếu của LDA:

- `doc_topic`: ma trận kích thước `(n_docs, n_topics)`, mỗi hàng có tổng bằng 1 (hỗn hợp chủ đề của tài liệu).
- `topic_word`: ma trận kích thước `(n_topics, vocab_size)`, mỗi hàng có tổng bằng 1 (phân phối xác suất từ vựng của chủ đề).

**Đường ống xử lý của BERTopic.**

1. Mã hóa mỗi tài liệu bằng một mô hình sentence transformer (ví dụ: `all-MiniLM-L6-v2`) thành các vector 384 chiều.
2. Giảm số chiều bằng UMAP xuống khoảng ~5 chiều. Các vector nhúng BERT ban đầu có số chiều quá lớn đối với các thuật toán phân cụm.
3. Phân cụm bằng HDBSCAN. Thuật toán dựa trên mật độ, tạo ra các cụm có kích thước linh hoạt và một nhãn ngoại lai "outlier" (nhiễu).
4. Đối với mỗi cụm, tính toán class-based TF-IDF (c-TF-IDF) trên các tài liệu thuộc cụm đó để trích xuất các từ khóa tiêu biểu nhất.

Đầu ra là một chủ đề duy nhất cho mỗi tài liệu (kèm nhãn -1 cho các điểm ngoại lai không thể gom cụm). Tùy chọn có thể trích xuất mức độ thuộc cụm mềm (soft membership) thông qua vector xác suất của HDBSCAN.

```figure
topic-drift
```

## Xây dựng thực tế (Build It)

### Bước 1: LDA với scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Lưu ý quan trọng: Loại bỏ từ dừng (stop words), thiết lập `min_df` và `max_df` để lọc bỏ các từ quá hiếm hoặc xuất hiện khắp nơi, sử dụng `CountVectorizer` (không dùng `TfidfVectorizer`) vì LDA đòi hỏi tần số đếm thô nguyên gốc.

### Bước 2: BERTopic (Cấp độ sản xuất)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Bộ lọc `Topic != -1` loại bỏ nhóm điểm ngoại lai của BERTopic (các tài liệu mà HDBSCAN không thể gom vào cụm nào). `min_topic_size` kiểm soát kích thước cụm tối thiểu của HDBSCAN; giá trị mặc định của thư viện BERTopic là 10. Ví dụ này đặt thành 15 cho phù hợp quy mô bài học. Với các kho dữ liệu trên 10.000 tài liệu, hãy tăng lên 50 hoặc 100.

### Bước 3: Đánh giá chất lượng chủ đề

Cả hai phương pháp đều đưa ra danh sách các từ khóa tiêu biểu cho mỗi chủ đề. Vấn đề là các từ khóa đó có thực sự mạch lạc và tạo thành một khái niệm rõ ràng hay không.

- **Độ mạch lạc chủ đề (Topic coherence - c_v).** Kết hợp chỉ số NPMI (thông tin tương hỗ từng điểm được chuẩn hóa) giữa các cặp từ hàng đầu qua cửa sổ trượt ngữ cảnh, tổng hợp các điểm số thành vector chủ đề, rồi so sánh các vector đó bằng độ tương đồng cosine. Điểm càng cao càng tốt. Sử dụng `gensim.models.CoherenceModel` với tham số `coherence="c_v"`.
- **Độ đa dạng chủ đề (Topic diversity).** Tỷ lệ các từ duy nhất trên toàn bộ danh sách các từ hàng đầu của tất cả các chủ đề. Giá trị càng cao càng tốt (các chủ đề không bị trùng lặp từ khóa lẫn nhau).
- **Kiểm tra định tính bằng mắt người (Qualitative inspection).** Đọc danh sách các từ hàng đầu của từng chủ đề. Chúng có chỉ ra một khái niệm thực tế rõ ràng không? Đánh giá của con người vẫn là chốt chặn cuối cùng.

## Khi nào nên chọn phương pháp nào

| Tình huống | Lựa chọn |
|------------|----------|
| Văn bản ngắn (tweet, đánh giá, tiêu đề tin tức) | BERTopic |
| Tài liệu dài với sự pha trộn đa chủ đề | LDA |
| Không có GPU / tài nguyên tính toán hạn chế | LDA hoặc NMF |
| Cần phân phối đa chủ đề ở cấp độ tài liệu | LDA |
| Tích hợp LLM để tự động đặt tên nhãn chủ đề | BERTopic (hỗ trợ trực tiếp) |
| Triển khai trên thiết bị biên tài nguyên thấp | LDA |
| Đòi hỏi độ mạch lạc ngữ nghĩa tối đa | BERTopic |

Yếu tố thực tiễn lớn nhất cần cân nhắc là độ dài tài liệu. Vector nhúng BERT sẽ cắt ngắn văn bản vượt quá giới hạn token; trong khi tần số đếm của LDA hoạt động trên bất kỳ độ dài tài liệu nào. Đối với các tài liệu dài hơn ngữ cảnh của mô hình nhúng, bạn phải chia đoạn rồi tổng hợp lại, hoặc dùng LDA.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

- **BERTopic.** Chuẩn mặc định cho văn bản ngắn và mọi bài toán mà ngữ nghĩa sâu sắc là yếu tố quyết định.
- **`gensim.models.LdaModel`.** LDA kinh điển cho sản xuất, hoàn thiện, đã qua kiểm chứng thực tế khắt khe.
- **`sklearn.decomposition.LatentDirichletAllocation`.** Bản cài đặt LDA tiện lợi, dễ dùng cho các thử nghiệm nhanh.
- **NMF.** Phân tích ma trận không âm (Non-negative matrix factorization). Phương án thay thế nhanh cho LDA, chất lượng tương đương trên văn bản ngắn.
- **Top2Vec.** Thiết kế tương đồng BERTopic. Cộng đồng nhỏ hơn nhưng đạt kết quả rất tốt trên một số benchmark.
- **FASTopic.** Mới hơn, tốc độ chạy nhanh hơn BERTopic trên các kho dữ liệu cực lớn.
- **Gán nhãn bằng LLM.** Chạy thuật toán phân cụm bất kỳ, sau đó dùng prompt để yêu cầu LLM đặt tên đại diện cho từng cụm.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Lựa chọn LDA hoặc BERTopic cho một kho ngữ liệu. Chỉ định thư viện, tham số và quy trình đánh giá.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Dựa trên mô tả kho dữ liệu (số lượng tài liệu, độ dài trung bình, miền nghiệp vụ, ngôn ngữ, ngân sách tính toán), bạn đưa ra:

1. Thuật toán: LDA / NMF / BERTopic / Top2Vec / FASTopic. Giải thích lý do trong một câu.
2. Cấu hình: Số lượng chủ đề: `đề xuất = max(5, round(sqrt(n_docs)))`, giới hạn trần ở 200 cho các kho dưới 40.000 tài liệu; chỉ cho phép >200 khi kho dữ liệu thực sự lớn (>40k) và ghi chú rõ chi phí tính toán tăng thêm. Bộ lọc `min_df` / `max_df` và mô hình nhúng cho phương pháp nơ-ron cũng được nêu rõ tại đây.
3. Kế hoạch đánh giá: Độ mạch lạc chủ đề (c_v) qua `gensim.models.CoherenceModel`, độ đa dạng chủ đề, và việc đọc kiểm tra thực tế trên 20 mẫu ngẫu nhiên.
4. Một phương thức lỗi cần kiểm tra: Với LDA, các "chủ đề rác" hấp thụ toàn từ dừng và từ xuất hiện quá nhiều. Với BERTopic, cụm ngoại lai -1 nuốt chửng các tài liệu mơ hồ.

Từ chối dùng BERTopic trên các tài liệu dài hơn cửa sổ ngữ cảnh của mô hình nhúng nếu thiếu chiến lược phân đoạn đoạn văn (chunking). Từ chối dùng LDA trên văn bản quá ngắn (tweet, bình luận dưới 10 token) vì độ mạch lạc sẽ đổ vỡ hoàn toàn. Cảnh báo bất kỳ lựa chọn số chủ đề nào dưới 5 là có nguy cơ sai lệch; cảnh báo số chủ đề >200 trên kho dưới 40k tài liệu là có nguy cơ phân mảnh quá mức.
```

## Bài tập (Exercises)

1. **Dễ.** Khớp mô hình LDA với 5 chủ đề trên tập dữ liệu 20 Newsgroups. In ra 10 từ hàng đầu cho mỗi chủ đề. Tự tay gán nhãn tên chủ đề cho từng nhóm từ đó. Liệu thuật toán có tìm ra đúng các danh mục thực tế của tập dữ liệu hay không?
2. **Trung bình.** Khớp mô hình BERTopic trên cùng tập con 20 Newsgroups đó. So sánh số lượng chủ đề tìm được, danh sách từ hàng đầu, và độ mạch lạc định tính so với LDA. Phương pháp nào bộc lộ các danh mục thực tế rõ ràng và tự nhiên hơn?
3. **Khó.** Tính toán độ mạch lạc c_v cho cả LDA và BERTopic trên kho dữ liệu của bạn. Chạy mỗi phương pháp với các mức 5, 10, 20, 50 chủ đề. Vẽ biểu đồ tương quan giữa độ mạch lạc và số lượng chủ đề. Báo cáo phương pháp nào giữ được tính ổn định cao hơn qua các mức số lượng chủ đề khác nhau.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Chủ đề (Topic) | Điều mà văn bản nói về | Một phân phối xác suất trên các từ (LDA) hoặc một cụm các tài liệu tương đồng ngữ nghĩa (BERTopic). |
| Thành viên hỗn hợp (Mixed membership) | Tài liệu thuộc nhiều chủ đề | LDA gán cho mỗi tài liệu một phân phối xác suất trên toàn bộ các chủ đề. |
| UMAP | Giảm số chiều | Thuật toán học đa tạp (manifold learning) bảo toàn cấu trúc cục bộ; dùng trong BERTopic. |
| HDBSCAN | Phân cụm mật độ | Tìm kiếm các cụm có kích thước linh hoạt; gán nhãn "nhiễu" (-1) cho các điểm ngoại lai. |
| Độ mạch lạc c_v (c_v coherence) | Thước đo chất lượng chủ đề | Điểm NPMI của các từ hàng đầu qua cửa sổ trượt, chuyển thành vector ngữ cảnh và so sánh bằng độ tương đồng cosine. |

## Đọc thêm (Further Reading)

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — bài báo gốc đặt nền móng cho LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — bài báo gốc về họ mô hình BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — công trình khoa học giới thiệu chỉ số độ mạch lạc c_v.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — tài liệu tham khảo chính thức trong môi trường sản xuất với nhiều ví dụ xuất sắc.

