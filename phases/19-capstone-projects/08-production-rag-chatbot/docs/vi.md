# Đồ án Tốt nghiệp 08 — Chatbot RAG Môi trường Sản xuất cho Lĩnh vực Chịu sự Quản lý

> Harvey, Glean, Mendable và LlamaCloud đều vận hành cùng một cấu trúc kiến trúc sản xuất vào năm 2026: Thu nạp dữ liệu với docling hoặc Unstructured và ColPali cho tài liệu trực quan; Tìm kiếm lai (hybrid search); Tái xếp hạng (re-rank) với bge-reranker-v2-gemma; Tổng hợp câu trả lời bằng Claude Sonnet 4.7 tận dụng bộ nhớ đệm lời nhắc (prompt caching) đạt tỷ lệ trúng (hit rate) 60-80%; Thiết lập hàng rào bảo vệ với Llama Guard 4 và NeMo Guardrails; Giám sát toàn diện với Langfuse và Phoenix; Đánh giá chất lượng bằng RAGAS trên tập chuẩn vàng (golden set) gồm 200 câu hỏi. Đồ án này yêu cầu xây dựng một hệ thống trong lĩnh vực chịu sự quản lý nghiêm ngặt (pháp lý, thử nghiệm lâm sàng, bảo hiểm), với mục tiêu vượt qua tập chuẩn vàng, bộ kiểm thử đội đỏ (red team), và bảng điều khiển giám sát độ trôi dạt (drift dashboard).

**Type:** Capstone  
**Languages:** Python (pipeline + API), TypeScript (chat UI)  
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)  
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18  
**Time:** ~30 giờ  

## Vấn đề đặt ra

RAG trong các lĩnh vực chịu sự quản lý (hợp đồng pháp lý, phác đồ thử nghiệm lâm sàng, hợp đồng bảo hiểm) là cấu trúc sản xuất được triển khai phổ biến nhất vào năm 2026 bởi lợi tức đầu tư (ROI) rõ ràng và rủi ro thực tế rất cao. Harvey (Allen & Overy) xây dựng hệ thống cho ngành luật. Mendable phát triển giải pháp cho tài liệu lập trình viên. Glean bao phủ mảng tìm kiếm doanh nghiệp. Mô hình chung là: thu nạp dữ liệu với độ trung thực cao, truy xuất lai kết hợp tái xếp hạng, tổng hợp câu trả lời kèm trích dẫn bắt buộc và bộ nhớ đệm lời nhắc, bảo vệ qua nhiều lớp an toàn, và liên tục giám sát độ trôi dạt.

Phần khó khăn nhất không nằm ở mô hình ngôn ngữ lớn. Nó nằm ở việc tuân thủ pháp lý theo từng khu vực tài phán (jurisdiction-aware compliance: HIPAA, GDPR, SOC2), khả năng kiểm toán ở cấp độ trích dẫn (citation-level auditability), kiểm soát chi phí (prompt caching giúp giảm 60-90% chi phí khi tỷ lệ trúng cao), phát hiện ảo giác thông qua độ trung thực (faithfulness) của RAGAS, và phát hiện độ trôi dạt khi tài liệu nguồn được cập nhật nhưng chỉ mục chưa kịp đồng bộ. Đồ án tốt nghiệp này yêu cầu bạn triển khai hoàn chỉnh toàn bộ hệ thống trên một tập chuẩn vàng gồm 200 câu hỏi đi kèm bộ kiểm thử đội đỏ đối kháng.

## Khái niệm cốt lõi

Quy trình vận hành chia làm hai nhánh:
- **Thu nạp dữ liệu (Ingestion):** docling hoặc Unstructured phân tích cú pháp các tài liệu có cấu trúc; ColPali xử lý các trang có bố cục thị giác phức tạp; các đoạn văn bản (chunks) được gắn kèm tóm tắt, thẻ phân loại, và nhãn kiểm soát truy cập dựa trên vai trò (role-based access labels). Vector được lưu vào pgvector + pgvectorscale (cho quy mô dưới 50 triệu vector) hoặc Qdrant Cloud; tìm kiếm thưa BM25 (Tantivy) chạy song song.
- **Hội thoại (Conversation):** LangGraph quản lý bộ nhớ và hội thoại đa lượt (multi-turn); mỗi truy vấn kích hoạt tìm kiếm lai, tái xếp hạng với bge-reranker-v2-gemma-2b, tổng hợp câu trả lời bằng Claude Sonnet 4.7 (có áp dụng prompt caching), lọc qua Llama Guard 4 và NeMo Guardrails, rồi trả về câu trả lời neo chặt vào trích dẫn nguồn xác thực.

Ngăn xếp đánh giá gồm bốn tầng:
1. **Tập chuẩn vàng (Golden set):** 200 cặp câu hỏi/câu trả lời do chuyên gia gán nhãn kèm trích dẫn để đánh giá tính chính xác.
2. **Đội đỏ (Red team):** Các bài kiểm tra tấn công đối kháng (bẻ khóa - jailbreak, cố gắng trích xuất thông tin cá nhân - PII, câu hỏi ngoài miền) để kiểm tra độ an toàn.
3. **RAGAS:** Đo lường độ trung thực (faithfulness), độ liên quan của câu trả lời (answer relevance), độ chuẩn xác của ngữ cảnh (context precision) tự động theo từng lượt.
4. **Bảng điều khiển độ trôi dạt (Arize Phoenix):** Giám sát chất lượng truy xuất và điểm ảo giác hàng tuần.

Bộ nhớ đệm lời nhắc (prompt caching) là đòn bẩy chi phí. Claude 4.5+ và GPT-5+ hỗ trợ lưu đệm lời nhắc hệ thống kèm ngữ cảnh truy xuất được. Khi đạt tỷ lệ trúng 60-80%, chi phí trên mỗi truy vấn giảm từ 3 đến 5 lần. Quy trình phải được thiết kế với tiền tố ổn định (system prompt + ngữ cảnh đã tái xếp hạng đặt ở đầu) để tối ưu hóa tỷ lệ trúng đệm.

## Kiến trúc hệ thống

```text
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Ngăn xếp công nghệ

- **Thu nạp dữ liệu (Ingestion):** Unstructured.io hoặc docling cho tài liệu có cấu trúc; ColPali cho tệp PDF nặng về bố cục thị giác
- **Cơ sở dữ liệu Vector:** pgvector + pgvectorscale dưới 50 triệu vector; Qdrant Cloud cho quy mô lớn hơn
- **Truy xuất thưa (Sparse):** Tantivy BM25 với trọng số trường
- **Điều phối (Orchestration):** LlamaIndex Workflows (cho thu nạp) + LangGraph (cho hội thoại)
- **Tái xếp hạng (Re-ranker):** bge-reranker-v2-gemma-2b tự lưu trữ (self-hosted) hoặc Voyage rerank-2 dạng hosted
- **Mô hình ngôn ngữ (LLM):** Claude Sonnet 4.7 với prompt caching; phương án dự phòng là Llama 3.3 70B tự lưu trữ
- **Đánh giá (Eval):** RAGAS 0.2 trực tuyến, DeepEval cho các bài kiểm tra ảo giác và bẻ khóa
- **Khả năng quan sát (Observability):** Langfuse tự lưu trữ kèm hàng đợi gán nhãn chú thích; Arize Phoenix để theo dõi độ trôi dạt
- **Hàng rào bảo vệ (Guardrails):** Bộ phân loại đầu vào/đầu ra Llama Guard 4, chính sách NeMo Guardrails v0.12, làm sạch PII bằng Presidio
- **Tuân thủ quy định (Compliance):** Nhãn phân quyền theo vai trò trên từng chunk; thẻ thẩm quyền tài phán phục vụ GDPR/HIPAA

```figure
canary-rollout
```

## Từng bước xây dựng

1. **Thu nạp dữ liệu.** Phân tích cú pháp kho ngữ liệu của bạn (1.000–10.000 tài liệu cho một bản dựng nghiêm túc) bằng Unstructured hoặc docling. Với các trang tài liệu scan hoặc nặng về biểu đồ hình ảnh, chuyển tiếp qua ColPali. Tạo các chunk kèm bản tóm tắt, nhãn vai trò, thẻ tài phán.

2. **Lập chỉ mục.** Sinh embedding dày đặc (Voyage-3 hoặc Nomic-embed-v2) nạp vào pgvector + pgvectorscale. Lập chỉ mục thưa BM25 phụ trợ qua Tantivy. Đính kèm bộ lọc vai trò và tài phán vào payload.

3. **Truy xuất lai.** Lọc trước theo vai trò + thẩm quyền tài phán; sau đó chạy song song truy xuất dày đặc và BM25; kết hợp bằng thuật toán dung hợp hạng nghịch đảo (Reciprocal Rank Fusion - RRF); chuyển top-20 sang bộ tái xếp hạng; lấy top-5 đưa vào bộ tổng hợp.

4. **Tổng hợp kèm bộ nhớ đệm lời nhắc.** Đặt lời nhắc hệ thống + chính sách tĩnh trong phần đầu đệm (cache header); ngữ cảnh đã tái xếp hạng đóng vai trò phần đệm mở rộng; câu hỏi của người dùng là phần hậu tố không đệm. Hướng tới tỷ lệ trúng đệm 60-80% ở trạng thái ổn định.

5. **Thiết lập hàng rào an toàn.** Áp dụng Llama Guard 4 lên câu hỏi đầu vào; thiết lập các rào cản NeMo Guardrails để chặn câu hỏi ngoài miền hoặc chủ đề vi phạm chính sách; Presidio làm sạch thông tin PII vô tình lọt vào đầu ra; áp dụng bộ lọc hậu kiểm bắt buộc gắn trích dẫn nguồn.

6. **Tập chuẩn vàng.** Xây dựng 200 cặp câu hỏi/đáp án được chuyên gia thẩm định gán nhãn (câu trả lời, nguồn trích dẫn). Đánh giá agent dựa trên độ khớp trích dẫn chính xác, tính chuẩn xác của câu trả lời, và độ trung thực qua RAGAS.

7. **Bộ kiểm thử đội đỏ.** Thiết kế 50 lời nhắc đối kháng: bẻ khóa (PAIR, TAP), cố gắng trích xuất PII, câu hỏi lạc đề, rò rỉ chéo khu vực tài phán. Chấm điểm theo mức đạt/không đạt kèm cấp độ nghiêm trọng.

8. **Bảng điều khiển độ trôi dạt.** Cấu hình Arize Phoenix theo dõi chất lượng truy xuất (nDCG, độ trung thực trích dẫn) hàng tuần. Thiết lập cảnh báo khi chỉ số giảm quá 5%.

9. **Báo cáo chi phí.** Sử dụng Langfuse: tỷ lệ trúng prompt caching, số token trên mỗi truy vấn, phân tích chi phí $/truy vấn chi tiết theo từng giai đoạn.

## Vận hành thực tế

```console
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Bàn giao sản phẩm

Tệp `outputs/skill-production-rag.md` mô tả sản phẩm bàn giao. Một chatbot cho miền chịu sự quản lý được triển khai kèm nhãn tuân thủ, vượt qua barem đánh giá, và được giám sát độ trôi dạt theo thời gian thực.

| Trọng số | Tiêu chí | Phương pháp đo lường |
|:-:|---|---|
| 25 | Độ trung thực RAGAS + Độ liên quan câu trả lời | Điểm trực tuyến trên tập chuẩn vàng (200 cặp Q/A) |
| 20 | Tính chính xác của trích dẫn | Tỷ lệ câu trả lời có nguồn trích dẫn kiểm chứng được |
| 20 | Độ bao phủ của rào cản an toàn | Tỷ lệ vượt qua Llama Guard 4 + kết quả từ bộ kiểm thử bẻ khóa |
| 20 | Tối ưu chi phí / Độ trễ | Tỷ lệ trúng prompt-cache, độ trễ p95, chi phí $/truy vấn |
| 15 | Bảng điều khiển giám sát độ trôi dạt | Bảng điều khiển Phoenix trực tiếp với xu hướng chất lượng truy xuất hàng tuần |
| **100** | | |

## Bài tập mở rộng

1. Xây dựng phân đoạn kho ngữ liệu thứ hai theo một khu vực tài phán khác (ví dụ: HIPAA song song với GDPR). Chứng minh cơ chế lọc vai trò + tài phán ngăn chặn rò rỉ chéo trên 20 câu hỏi thăm dò ranh giới quy định.

2. Đo lường tỷ lệ trúng prompt-cache qua một tuần lưu lượng truy cập thực tế trong môi trường sản xuất. Xác định những loại truy vấn nào làm vỡ tiền tố đệm. Tái cấu trúc lại luồng xử lý.

3. Bổ sung bộ nhớ hội thoại đa lượt với bộ đệm tóm tắt kích thước 10.000 token. Đo lường xem độ trung thực có bị suy giảm khi cuộc hội thoại kéo dài hay không.

4. Thay thế Claude Sonnet 4.7 bằng Llama 3.3 70B tự lưu trữ. Đo lường mức chênh lệch về chi phí $/truy vấn và độ trung thực.

5. Bổ sung chế độ "không chắc chắn": nếu điểm tái xếp hạng cao nhất dưới ngưỡng cho phép, agent sẽ thông báo "Tôi không có trích dẫn tin cậy để trả lời" thay vì tự suy diễn. Đo lường mức giảm thiểu hiện tượng tự tin thái quá giả tạo (false confidence).

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Prompt caching | "Đệm hệ thống + ngữ cảnh" | Tính năng của Claude/OpenAI: token tiền tố được lưu đệm sẽ giảm 60-90% chi phí khi trúng |
| RAGAS | "Bộ đánh giá RAG" | Chấm điểm tự động độ trung thực, độ liên quan câu trả lời và độ chuẩn xác ngữ cảnh |
| Golden set | "Tập đánh giá có nhãn" | Tập hơn 200 câu hỏi/đáp án kèm trích dẫn do chuyên gia gán nhãn; đóng vai trò chân lý chuẩn (ground truth) |
| Jurisdiction tag | "Nhãn tuân thủ" | Phạm vi GDPR/HIPAA/SOC2 gắn vào các chunk; được thực thi bởi bộ lọc truy xuất |
| Citation faithfulness | "Tỷ lệ câu trả lời dựa trên căn cứ" | Tỷ lệ các luận điểm được chứng minh bằng các đoạn văn bản nguồn truy xuất được |
| Drift | "Suy giảm chất lượng truy xuất" | Sự biến đổi hàng tuần của nDCG hoặc điểm trích dẫn; ngưỡng cảnh báo là 5% |
| Red team | "Đánh giá đối kháng" | Bộ kiểm thử bẻ khóa, trích xuất PII và thăm dò ngoài miền trước khi phát hành |

## Tài liệu đọc thêm

- [Harvey AI](https://www.harvey.ai) — ngăn xếp kiến trúc tham chiếu cho ngành pháp lý
- [Tìm kiếm doanh nghiệp Glean](https://www.glean.com) — tài liệu tham chiếu RAG ở quy mô doanh nghiệp
- [Tài liệu Mendable](https://mendable.ai) — tham chiếu RAG cho tài liệu nhà phát triển
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — quy trình thu nạp được quản trị
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — tài liệu tham chiếu về đòn bẩy chi phí
- [Tài liệu RAGAS 0.2](https://docs.ragas.io/) — khung đánh giá RAG chuẩn mực
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — công cụ tham chiếu quan sát độ trôi dạt
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — bộ phân loại an toàn thế hệ 2026
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — khung thiết lập rào cản chính sách
