# Đánh giá LLM — RAGAS, DeepEval, G-Eval

> Khớp từ ngữ chính xác (Exact-match) và F1 bỏ sót hoàn toàn tính tương đương về mặt ngữ nghĩa. Đánh giá thủ công bằng con người thì không thể mở rộng quy mô. Mô hình LLM-làm-giám-khảo (LLM-as-judge) chính là lời giải cho môi trường sản xuất — với điều kiện phải được hiệu chuẩn (calibrated) đủ tin cậy.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 13 (Hỏi đáp - QA), Giai đoạn 5 · 14 (Truy xuất thông tin)
**Time:** ~75 phút

## Vấn đề

Hệ thống RAG của bạn trả lời: "Ngày 29 tháng 6 năm 2007."
Câu trả lời mẫu chuẩn là: "29/06/2007."
Chỉ số Khớp chính xác (Exact Match) chấm 0 điểm. Chỉ số F1 chấm ~75%. Trong khi một người thẩm định bình thường sẽ cho điểm tuyệt đối 100%.

Bây giờ hãy nhân bài toán đó lên 10.000 trường hợp kiểm thử. Tiếp tục nhân lên sau mỗi lần tinh chỉnh bộ truy xuất, đổi chiến lược chia đoạn, sửa prompt hoặc đổi mô hình nền tảng. Bạn bắt buộc phải có một bộ đánh giá hiểu được ngữ nghĩa, chạy với chi phí rẻ ở quy mô lớn, không báo cáo sai lệch về các lỗi thoái lui (regressions), và chỉ điểm chính xác các dạng lỗi phát sinh.

Năm 2026 chứng kiến ba khung làm việc làm chủ bài toán này:

- **RAGAS (Retrieval-Augmented Generation ASsessment):** Bốn chỉ số RAG cốt lõi (tính trung thực - faithfulness, độ liên quan câu trả lời - answer-relevance, độ chính xác ngữ cảnh - context-precision, độ phủ ngữ cảnh - context-recall) với backend dựa trên NLI và LLM giám khảo. Nhẹ nhàng, có cơ sở nghiên cứu học thuật vững chắc.
- **DeepEval:** Được ví như Pytest dành riêng cho LLM. Tích hợp G-Eval, độ hoàn thành nhiệm vụ, phát hiện ảo giác và các chỉ số thiên kiến. Tương thích tự nhiên với đường ống CI/CD.
- **G-Eval:** Vừa là phương pháp luận, vừa là một chỉ số trong DeepEval: Mô hình LLM-as-judge kết hợp chuỗi suy luận (chain-of-thought), tiêu chí chấm điểm tùy biến và thang điểm 0-1.

Cả ba đều dựa trên mô hình LLM-as-judge. Bài học này xây dựng trực giác cho phương pháp luận và lớp kiểm soát độ tin cậy bao quanh nó.

## Khái niệm

![Bốn chiều đánh giá, kiến trúc LLM-làm-giám-khảo](../assets/llm-evaluation.svg)

**LLM-làm-giám-khảo (LLM-as-judge).** Thay thế các thước đo tĩnh bằng một LLM có nhiệm vụ chấm điểm kết quả đầu ra dựa trên một biểu mẫu tiêu chí (rubric). Với bộ ba `(truy vấn, ngữ cảnh, câu trả lời)`, ta đưa prompt cho LLM giám khảo: "Hãy chấm điểm từ 0 đến 1 về tính trung thực." và nhận lại điểm số.

Lý do phương pháp này hiệu quả: LLM xấp xỉ rất sát phán đoán của con người nhưng với chi phí rẻ hơn hàng trăm lần. GPT-4o-mini với chi phí chỉ ~$0,003 cho mỗi trường hợp được chấm điểm cho phép chạy bộ kiểm thử thoái lui 1.000 mẫu với giá chưa đến 5 USD.

Tại sao phương pháp này có thể âm thầm thất bại:

1. **Thiên kiến giám khảo (Judge bias):** Các LLM giám khảo thường ưu tiên câu trả lời dài hơn, ưu ái câu trả lời sinh ra từ các mô hình cùng dòng dõi họ hàng với nó, hoặc câu trả lời có văn phong giống với prompt.
2. **Lỗi phân tích cú pháp JSON:** JSON trả về bị hỏng → sinh điểm NaN → bị âm thầm loại bỏ khỏi kết quả tổng hợp. Người dùng RAGAS rất thấm thía nỗi đau này. Cần bao bọc bằng khối try/except và ghi nhận rõ trạng thái thất bại.
3. **Trôi dạt phiên bản mô hình (Model drift):** Khi nhà cung cấp âm thầm nâng cấp mô hình giám khảo, mọi điểm số lịch sử sẽ bị xáo trộn. Cần ghim chặt định danh mô hình kèm số hiệu phiên bản.

**Bộ tứ chỉ số RAG (The RAG four).**

| Chỉ số | Câu hỏi đánh giá | Backend xử lý |
|--------|-------------------|---------------|
| Tính trung thực (Faithfulness) | Mọi khẳng định trong câu trả lời có bắt nguồn từ ngữ cảnh được truy xuất không? | Suy diễn logic dựa trên NLI |
| Độ liên quan câu trả lời (Answer relevance) | Câu trả lời có giải quyết đúng trọng tâm câu hỏi không? | Sinh câu hỏi giả định từ câu trả lời; so sánh với câu hỏi thực |
| Độ chính xác ngữ cảnh (Context precision) | Trong các đoạn văn bản truy xuất về, tỷ lệ đoạn thực sự hữu ích là bao nhiêu? | LLM giám khảo |
| Độ phủ ngữ cảnh (Context recall) | Quá trình truy xuất đã kéo về đủ mọi thông tin cần thiết chưa? | LLM giám khảo đối chiếu câu trả lời chuẩn |

**G-Eval.** Định nghĩa một tiêu chí nghiệp vụ riêng: "Câu trả lời có trích dẫn đúng nguồn tài liệu không?" Khung làm việc sẽ tự động phân rã tiêu chí thành chuỗi các bước suy luận CoT, sau đó cho điểm từ 0-1. Rất phù hợp cho các chiều chất lượng đặc thù mà RAGAS không bao quát.

**Hiệu chuẩn (Calibration).** Tuyệt đối không tin tưởng điểm số thô của LLM giám khảo khi chưa đo lường độ tương quan với nhãn gán của con người. Hãy gán nhãn thủ công 100 mẫu. Vẽ biểu đồ đối chiếu giữa giám khảo LLM và chuyên gia con người. Tính hệ số tương quan hạng Spearman $\rho$. Nếu $\rho < 0,7$, biểu mẫu tiêu chí chấm điểm của bạn bắt buộc phải được viết lại.

```figure
n5-judge-gauge
```

## Thực hành xây dựng

### Bước 1: Tính trung thực với NLI (Phong cách RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` là một hàm có thể gọi: prompt str -> chuỗi kết quả sinh ra.
# Ví dụ: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Tách nhỏ câu trả lời thành các khẳng định dữ kiện nguyên tử (atomic claims). Dùng mô hình NLI kiểm tra từng khẳng định đối chiếu với ngữ cảnh truy xuất. Điểm trung thực = tỷ lệ các khẳng định có bằng chứng hỗ trợ.

### Bước 2: Độ liên quan của câu trả lời (Answer relevance)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: mô hình hỗ trợ .encode(texts, normalize_embeddings=True) -> ndarray
# ví dụ: encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Nếu nội dung câu trả lời ám chỉ những câu hỏi hoàn toàn xa lạ so với câu hỏi gốc ban đầu, điểm số liên quan sẽ sụt giảm.

### Bước 3: Tiêu chí tùy biến với G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Các bước đánh giá (evaluation_steps) chính là biểu mẫu chấm điểm chi tiết. Việc quy định rõ ràng từng bước suy luận sẽ giúp kết quả ổn định hơn rất nhiều so với những câu prompt mơ hồ kiểu "hãy chấm điểm từ 0 đến 1".

### Bước 4: Cổng kiểm soát chất lượng CI (CI Gate)

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Đóng gói thành tệp pytest. Chạy tự động trên mỗi Pull Request. Chặn merge nếu phát hiện lỗi thoái lui.

### Bước 5: Bộ đánh giá mẫu từ đầu

Xem chi tiết trong `code/main.py`. Bản hiện thực chỉ dùng thư viện chuẩn stdlib xấp xỉ tính trung thực (độ trùng lặp giữa các khẳng định và ngữ cảnh) và độ liên quan (độ trùng khớp từ khóa giữa câu trả lời và câu hỏi). Không dùng cho production nhưng bộc lộ trọn vẹn bản chất cấu trúc.

## Cạm bẫy

- **Không hiệu chuẩn (No calibration).** Một giám khảo LLM chỉ có độ tương quan 0,3 với đánh giá của con người thì không khác gì một bộ sinh số ngẫu nhiên. Bắt buộc phải chạy hiệu chuẩn trước khi đưa vào sản xuất.
- **Tự chấm điểm (Self-evaluation).** Dùng chính cùng một dòng mô hình LLM để vừa sinh câu trả lời vừa làm giám khảo chấm điểm sẽ thổi phồng kết quả thêm 10-20%. Bắt buộc phải dùng họ mô hình khác cho vai trò giám khảo.
- **Thiên kiến vị trí khi so sánh cặp (Positional bias).** Giám khảo thường thiên vị phương án xuất hiện đầu tiên. Hãy luôn xáo trộn ngẫu nhiên thứ tự và chạy đánh giá cả hai chiều.
- **Điểm số trung bình che giấu thảm họa.** Điểm trung bình 0,85 hoàn toàn có thể che giấu 5% trường hợp câu trả lời sai lệch nghiêm trọng. Hãy luôn kiểm tra phân vị đáy (bottom quantile).
- **Sự xuống cấp của tập dữ liệu vàng (Dataset rot).** Các tập dữ liệu đánh giá không được đánh số phiên bản sẽ bị trôi dạt theo thời gian, phá vỡ khả năng so sánh dọc. Luôn gắn tag phiên bản cho tập dữ liệu sau mỗi lần sửa đổi.
- **Chi phí gọi LLM.** Ở quy mô lớn, chi phí gọi giám khảo sẽ chiếm phần lớn ngân sách. Hãy sử dụng mô hình rẻ nhất vẫn vượt qua được ngưỡng hiệu chuẩn yêu cầu (GPT-4o-mini, Claude Haiku, Mistral-small).

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống sử dụng | Khung làm việc đề xuất |
|--------------------|------------------------|
| Giám sát chất lượng RAG | RAGAS (4 chỉ số chuẩn) |
| Cổng kiểm thử thoái lui CI/CD | DeepEval + pytest |
| Tiêu chí chất lượng đặc thù miền | G-Eval tích hợp trong DeepEval |
| Giám sát luồng truy cập trực tiếp (online) | RAGAS chế độ không cần câu trả lời mẫu (reference-free) |
| Kiểm tra điểm xuyết có con người tham gia | LangSmith hoặc Phoenix với giao diện gán nhãn |
| Đánh giá an toàn / Red-teaming | Promptfoo + DeepEval |

Kiến trúc chuẩn: Dùng RAGAS để giám sát vận hành, DeepEval cho đường ống CI, và G-Eval cho các tiêu chí nghiệp vụ mới lạ. Chạy cả ba; sự bất đồng ý kiến giữa chúng là tín hiệu cảnh báo cực kỳ giá trị.

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Bài tập

1. **Dễ.** Chạy RAGAS trên 10 ví dụ RAG có sẵn các trường hợp ảo giác đã biết trước. Xác nhận xem chỉ số faithfulness có bắt được chính xác từng lỗi ảo giác không.
2. **Trung bình.** Gán nhãn thủ công cho 50 câu trả lời QA với thang điểm 0-1 về độ chính xác. Chấm điểm lại bằng G-Eval. Đo hệ số tương quan Spearman $\rho$ giữa giám khảo LLM và con người.
3. **Khó.** Xây dựng cổng CI bằng pytest kết hợp DeepEval. Cố tình làm giảm chất lượng của bộ truy xuất. Xác nhận cổng CI chặn thành công lượt build. Bổ sung cơ chế cảnh báo khi phân vị 10% thấp nhất rớt điểm.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| LLM-làm-giám-khảo (LLM-as-judge) | Dùng LLM chấm điểm | Prompt một mô hình giám khảo chấm điểm từ 0-1 theo một bộ tiêu chí. |
| RAGAS | Thư viện đo chuẩn RAG | Khung làm việc mã nguồn mở với 4 chỉ số RAG không cần đáp án mẫu. |
| Tính trung thực (Faithfulness) | Câu trả lời có căn cứ không? | Tỷ lệ các khẳng định trong câu trả lời được chứng minh bởi ngữ cảnh truy xuất. |
| Độ chính xác ngữ cảnh (Context precision) | Các đoạn lấy về có đúng trọng tâm? | Tỷ lệ các đoạn văn trong top-K thực sự đóng góp vào câu trả lời. |
| Độ phủ ngữ cảnh (Context recall) | Đã tìm thấy đủ dữ liệu chưa? | Tỷ lệ các ý trong câu trả lời chuẩn được hỗ trợ bởi các đoạn văn tìm về. |
| G-Eval | Giám khảo tùy biến | Biểu mẫu tiêu chí + chuỗi suy luận CoT + điểm số từ 0 đến 1. |
| Hiệu chuẩn (Calibration) | Tin nhưng phải kiểm chứng | Hệ số tương quan Spearman giữa điểm của giám khảo LLM và chuyên gia con người. |

## Đọc thêm

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — bài báo công bố thư viện RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — bài báo nền tảng về phương pháp G-Eval.
- [Tài liệu chính thức DeepEval](https://deepeval.com/docs/metrics-introduction) — ngăn xếp đánh giá chuẩn cho môi trường sản xuất.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — phân tích thiên kiến, hiệu chuẩn và các giới hạn của LLM-as-judge.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — khung làm việc hợp nhất RAGAS, DeepEval và Phoenix.

