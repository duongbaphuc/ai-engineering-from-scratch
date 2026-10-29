# Hệ thống Hỏi Đáp (Question Answering Systems)

> Ba hệ thống định hình lĩnh vực QA hiện đại: Dạng trích xuất tìm kiếm các đoạn văn bản (spans). Dạng tăng cường truy xuất neo chúng vào tài liệu thực tế. Dạng tạo sinh trực tiếp sản sinh câu trả lời. Mọi trợ lý AI hiện đại đều là sự kết hợp khéo léo của cả ba.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Dịch máy), Phase 5 · 10 (Cơ chế Chú ý)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Người dùng gõ: "Chiếc iPhone đầu tiên ra mắt khi nào?" và kỳ vọng nhận được: "Ngày 29 tháng 6 năm 2007." Không phải một câu dài dòng: "Lịch sử của Apple rất dài và phong phú." Cũng không phải một con số trơ trọi "2007" thiếu hoàn cảnh câu. Một câu trả lời trực diện, có căn cứ xác thực và chuẩn xác tuyệt đối.

Ba kiến trúc lớn đã chi phối lĩnh vực QA trong suốt thập kỷ qua:

- **QA trích xuất (Extractive QA).** Cho trước một câu hỏi và một đoạn văn bản (passage) đã biết chắc chắn chứa câu trả lời, tìm ra chỉ số bắt đầu (start) và kết thúc (end) của đoạn văn bản câu trả lời trong đoạn văn đó. Bộ dữ liệu SQuAD là thước đo chuẩn mực kinh điển.
- **QA miền mở (Open-domain QA).** Đoạn văn bản không được cung cấp sẵn. Trước hết phải truy xuất (retrieve) đoạn văn bản liên quan từ kho dữ liệu, sau đó mới trích xuất hoặc tạo sinh câu trả lời. Đây chính là nền tảng cốt lõi của mọi đường ống RAG ngày nay.
- **QA tạo sinh / Sách đóng (Generative / Closed-book QA).** Một mô hình ngôn ngữ lớn tự trả lời trực tiếp dựa trên bộ nhớ tham số (parametric memory) của chính nó mà không cần bước truy xuất tài liệu ngoài. Cho tốc độ suy luận nhanh nhất, nhưng kém tin cậy nhất về mặt tính xác thực của dữ kiện.

Xu hướng chủ đạo của năm 2026 là sự kết hợp lai (hybrid): Truy xuất vài đoạn văn bản chất lượng nhất, sau đó cung cấp vào prompt để một mô hình tạo sinh đưa ra câu trả lời được neo chặt (grounded) vào chính các đoạn văn đó. Đó là RAG, và bài học 14 sẽ phân tích sâu nửa phần truy xuất. Bài học này tập trung xây dựng nửa phần QA.

## Khái niệm (The Concept)

![Các kiến trúc hỏi đáp (QA): trích xuất, tăng cường truy xuất, và tạo sinh](../assets/qa.svg)

**Dạng trích xuất (Extractive).** Mã hóa đồng thời cả câu hỏi và đoạn văn bản bằng một mô hình Transformer (họ BERT). Huấn luyện hai đầu nơ-ron (heads) dự đoán chỉ số token bắt đầu và kết thúc của câu trả lời. Hàm mất mát là cross-entropy trên các vị trí hợp lệ. Đầu ra là một đoạn văn bản nguyên gốc (span) trích từ đoạn văn. Bản chất thiết kế đảm bảo: Không bao giờ bịa đặt ảo giác, nhưng cũng không bao giờ trả lời được những câu hỏi mà đoạn văn bản không chứa thông tin.

**Tăng cường truy xuất (Retrieval-augmented - RAG).** Quy trình gồm hai giai đoạn: Trước tiên, bộ truy xuất (retriever) tìm ra top-`k` đoạn văn bản liên quan nhất từ kho ngữ liệu. Thứ hai, bộ đọc (reader - trích xuất hoặc tạo sinh) sản sinh câu trả lời dựa trên các đoạn văn bản đó. Việc phân tách rõ ràng giữa retriever và reader cho phép huấn luyện và đánh giá độc lập từng thành phần. Các hệ thống RAG hiện đại thường bổ sung thêm một bộ xếp hạng lại (reranker) ở giữa.

**Dạng tạo sinh (Generative).** Một LLM thuần giải mã (decoder-only như GPT, Claude, Llama) trả lời từ các trọng số đã học. Không có bước truy xuất. Thể hiện xuất sắc trên các kiến thức phổ thông, nhưng cực kỳ rủi ro đối với các dữ kiện hiếm hoặc thông tin mới xuất hiện. Tỷ lệ ảo giác tỷ lệ nghịch với tần suất xuất hiện của dữ kiện đó trong dữ liệu tiền huấn luyện.

```figure
qa-span
```

## Xây dựng thực tế (Build It)

### Bước 1: QA trích xuất với mô hình huấn luyện sẵn

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` được huấn luyện trên SQuAD 2.0, tập dữ liệu bao gồm cả những câu hỏi không có lời giải trong đoạn văn. Mặc định, pipeline `question-answering` vẫn sẽ trả về đoạn văn có điểm cao nhất ngay cả khi điểm rỗng (null score) của mô hình chiến thắng — nó *không* tự động trả về câu trả lời rỗng. Để có hành vi trả lời rõ ràng "không có câu trả lời", hãy truyền thêm `handle_impossible_answer=True` vào lệnh gọi pipeline: khi đó pipeline chỉ trả lời rỗng khi điểm null score vượt qua mọi điểm số đoạn văn khác. Dù theo cách nào, hãy luôn kiểm tra trường `score`.

### Bước 2: Phác thảo đường ống tăng cường truy xuất (RAG)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Pipeline hai giai đoạn: Bộ truy xuất dày đặc (Dense retriever dùng Sentence-BERT) tìm các đoạn văn bản liên quan dựa trên độ tương đồng ngữ nghĩa. Bộ đọc trích xuất (RoBERTa-SQuAD) rút trích đoạn câu trả lời từ các đoạn văn bản hàng đầu ghép lại. Mô hình này hoạt động tốt trên các kho dữ liệu nhỏ. Với hàng triệu tài liệu, hãy sử dụng FAISS hoặc cơ sở dữ liệu vector.

### Bước 3: Tạo sinh câu trả lời với RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Mẫu cấu trúc prompt đóng vai trò rất lớn: Việc hướng dẫn rõ ràng yêu cầu mô hình chỉ neo dữ kiện vào ngữ cảnh và trả lời "Tôi không biết" khi ngữ cảnh thiếu thông tin giúp giảm tỷ lệ ảo giác từ 40% đến 60% so với cách prompt sơ sài thông thường. Các mẫu thiết kế nâng cao sẽ bổ sung thêm trích dẫn nguồn, điểm tự tin và trích xuất cấu trúc định dạng.

### Bước 4: Đánh giá phản ánh đúng thực tế

SQuAD sử dụng **Exact Match (EM)** và **F1 cấp độ token**. EM là phép khớp chính xác tuyệt đối sau khi chuẩn hóa (viết thường, bỏ dấu câu, bỏ mạo từ) — dự đoán phải khớp từng chữ hoặc nhận điểm 0. F1 được tính toán dựa trên mức độ trùng lặp token giữa câu dự đoán và câu tham chiếu, ghi nhận điểm số một phần. Cả hai chỉ số đều đánh giá khắt khe đối với cách diễn đạt tương đương: "June 29, 2007" so với "June 29th, 2007" thường bị 0 điểm EM (do hậu tố thứ tự phá vỡ chuẩn hóa) dù vẫn đạt điểm F1 rất cao từ các token trùng khớp.

Trong môi trường sản xuất thực tế, cần đánh giá các khía cạnh:

- **Độ chính xác của câu trả lời (Answer accuracy)** (đánh giá qua LLM làm giám khảo hoặc chuyên gia con người, vì các chỉ số toán học không nắm bắt được tính tương đương ngữ nghĩa).
- **Độ chính xác của trích dẫn (Citation accuracy).** Đoạn văn bản được trích dẫn có thực sự chứng minh cho câu trả lời hay không? Rất dễ kiểm tra tự động bằng phép so khớp chuỗi giữa các trích dẫn tạo ra và các đoạn văn đã truy xuất.
- **Độ chuẩn hóa từ chối (Refusal calibration).** Khi câu trả lời không tồn tại trong các đoạn văn bản truy xuất, hệ thống có từ chối chính xác bằng câu "Tôi không biết" hay không? Đo lường tỷ lệ tự tin sai (false confidence rate).
- **Độ nhạy của bộ truy xuất (Retrieval recall).** Trước khi đánh giá bộ đọc, phải đo lường xem bộ truy xuất có đưa đúng đoạn văn bản cần tìm vào top-`k` hay không. Một bộ đọc không thể cứu vãn tình thế nếu đoạn thông tin đúng không hề được truy xuất.

### RAGAS: Khung đánh giá sản xuất năm 2026

Thư viện `RAGAS` được thiết kế chuyên biệt cho các hệ thống RAG và là tiêu chuẩn mặc định năm 2026. Nó chấm điểm trên bốn chiều kích mà không yêu cầu câu trả lời chuẩn (gold reference):

- **Độ trung thực (Faithfulness).** Mỗi khẳng định trong câu trả lời có bắt nguồn từ ngữ cảnh truy xuất không? Đo lường bằng bài toán kéo theo NLI. Đây là chỉ số đo lường ảo giác chính yếu của bạn.
- **Độ liên quan của câu trả lời (Answer relevance).** Câu trả lời có giải quyết trực tiếp câu hỏi không? Đo lường bằng cách sinh ra các câu hỏi giả định từ câu trả lời rồi so sánh với câu hỏi gốc.
- **Độ chính xác của ngữ cảnh (Context precision).** Trong số các đoạn văn được truy xuất về, có bao nhiêu phần trăm thực sự chứa thông tin liên quan? Precision thấp đồng nghĩa với việc prompt bị lẫn quá nhiều tạp âm nhiễu.
- **Độ nhạy của ngữ cảnh (Context recall).** Tập hợp truy xuất có chứa đầy đủ mọi thông tin cần thiết để trả lời không? Recall thấp đồng nghĩa với việc bộ đọc cầm chắc thất bại.

Phương pháp chấm điểm không cần tham chiếu cho phép bạn kiểm định liên tục trên lưu lượng truy cập thực tế mà không cần bộ dữ liệu chuẩn thủ công tốn kém. Kết hợp thêm LLM làm giám khảo cho các câu hỏi mở nơi các chỉ số so khớp câu từ hoàn toàn vô dụng.

Cài đặt bằng `pip install ragas`. Cắm retriever và reader của bạn vào. Bạn sẽ nhận được bốn chỉ số vô hướng cho mỗi câu truy vấn để thiết lập cảnh báo suy giảm hiệu năng.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Trường hợp sử dụng | Đề xuất |
|-------------------|---------|
| Cho trước đoạn văn, tìm đoạn văn bản câu trả lời | `deepset/roberta-base-squad2` |
| Trên một kho ngữ liệu cố định, không chấp nhận việc mô hình trả lời bừa | RAG: dense retriever + LLM reader |
| Thời gian thực trên kho lưu trữ tài liệu | RAG với retriever lai (BM25 + dense) + reranker (bài học 14) |
| QA dạng hội thoại (có câu hỏi nối tiếp) | LLM kèm lịch sử hội thoại + RAG trên mỗi lượt trao đổi |
| Các lĩnh vực đòi hỏi dữ kiện tuyệt đối, quy định nghiêm ngặt | Dạng trích xuất trên kho dữ liệu có thẩm quyền; không bao giờ dùng tạo sinh đơn độc |

QA trích xuất ít được chú ý hơn vào năm 2026 vì RAG với LLM xử lý được nhiều tình huống linh hoạt hơn. Tuy nhiên, dạng trích xuất vẫn được triển khai mạnh mẽ trong các bài toán bắt buộc phải trích dẫn nguyên văn: tra cứu pháp luật, kiểm toán tuân thủ quy chuẩn, công cụ rà soát tài chính.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Lựa chọn kiến trúc QA, chiến lược truy xuất và kế hoạch đánh giá.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Dựa trên yêu cầu bài toán (kích thước kho ngữ liệu, loại câu hỏi, ràng buộc dữ kiện, ngân sách độ trễ), bạn đưa ra:

1. Kiến trúc: Trích xuất (extractive), RAG với bộ đọc trích xuất, RAG với bộ đọc tạo sinh, hoặc LLM sách đóng (closed-book). Giải thích lý do trong một câu.
2. Bộ truy xuất (Retriever): Không dùng, BM25, dense (nêu tên bộ encoder cụ thể), hoặc hybrid lai ghép.
3. Bộ đọc (Reader): Mô hình tinh chỉnh trên SQuAD, LLM cụ thể theo tên, hoặc "DistilBERT fine-tune theo miền".
4. Kế hoạch đánh giá: EM + F1 cho bài đo trích xuất; Answer accuracy + Citation accuracy + Refusal calibration cho hệ thống sản xuất. Nêu rõ đang đo lường cái gì và bằng công cụ nào.

Từ chối câu trả lời từ LLM sách đóng (closed-book) đối với các câu hỏi liên quan đến pháp lý hoặc tuân thủ quy định nghiêm ngặt. Từ chối bất kỳ hệ thống QA nào thiếu chỉ số cơ sở về độ nhạy truy xuất (retrieval recall) — bạn không thể đánh giá bộ đọc nếu không biết bộ truy xuất có tìm ra đúng đoạn văn hay không. Đánh dấu các câu hỏi suy luận nhiều bước (multi-hop) là cần bộ truy xuất chuyên dụng như các hệ thống huấn luyện trên HotpotQA.
```

## Bài tập (Exercises)

1. **Dễ.** Thiết lập pipeline trích xuất SQuAD ở trên trên 10 đoạn văn bản Wikipedia. Tự tạo 10 câu hỏi tương ứng. Đo lường tần suất câu trả lời chính xác. Bạn sẽ thấy đạt khoảng 7-9 câu đúng nếu câu hỏi và đoạn văn rõ ràng.
2. **Trung bình.** Xây dựng bộ phân loại từ chối (refusal classifier). Khi điểm số truy xuất cao nhất thấp hơn một ngưỡng cụ thể (ví dụ: độ tương đồng cosine < 0.3), trả về "Tôi không biết" thay vì gọi bộ đọc. Tinh chỉnh ngưỡng này trên tập dữ liệu kiểm thử độc lập.
3. **Khó.** Xây dựng một pipeline RAG trên một kho ngữ liệu 10.000 tài liệu do bạn chọn. Cài đặt truy xuất lai hybrid (BM25 + dense) kết hợp phép hợp nhất RRF (xem bài học 14). Đo độ chính xác câu trả lời khi có và không có bước hybrid. Ghi chép tài liệu nhóm câu hỏi nào hưởng lợi nhiều nhất từ phương pháp lai này.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| QA trích xuất (Extractive QA) | Tìm đoạn câu trả lời | Dự đoán chỉ số bắt đầu và kết thúc của câu trả lời bên trong một đoạn văn bản cho trước. |
| QA miền mở (Open-domain QA) | QA trên toàn kho dữ liệu | Không cho trước đoạn văn; bắt buộc phải truy xuất tài liệu rồi mới trả lời. |
| RAG | Truy xuất rồi tạo sinh | Retrieval-augmented generation. Đường ống kết hợp bộ truy xuất (retriever) và bộ đọc (reader). |
| SQuAD | Thước đo chuẩn mực | Stanford Question Answering Dataset. Đánh giá bằng hai chỉ số EM và F1. |
| Ảo giác (Hallucination) | Câu trả lời tự bịa | Đầu ra của bộ đọc không hề được ngữ cảnh truy xuất chứng thực. |
| Chuẩn hóa từ chối (Refusal calibration) | Biết khi nào nên im lặng | Hệ thống đưa ra câu trả lời "Tôi không biết" chuẩn xác khi không thể tìm thấy câu trả lời trong dữ liệu. |

## Đọc thêm (Further Reading)

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — bài báo gốc đặt ra chuẩn mực SQuAD.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, bộ truy xuất dày đặc kinh điển cho QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — bài báo chính thức đặt tên cho RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — bài báo tổng quan toàn diện về RAG.

