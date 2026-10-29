# Bộ nhớ Lai: Vector + Đồ thị + Khóa-Giá trị (KV)

> Bộ nhớ lai (hybrid memory) vận hành ba kho lưu trữ song song — vector dành cho độ tương đồng ngữ nghĩa, khóa-giá trị (key-value hay KV) dành cho tra cứu dữ kiện nhanh chóng, đồ thị (graph) dành cho suy luận quan hệ thực thể — kết hợp với một tầng chấm điểm giúp dung hợp (fusion) chúng khi truy xuất. Đây là một mô hình thiết kế được sử dụng rộng rãi trong môi trường sản xuất cho bộ nhớ ngoài; Mem0 (Chhikara et al., 2025) là một bản triển khai tham chiếu tiêu biểu.

**Loại bài học:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn - stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Khối Bộ nhớ Letta - Letta Blocks)  
**Thời lượng:** ~75 phút

## Mục tiêu Học tập

- Giải thích lý do tại sao một kho lưu trữ đơn lẻ (chỉ vector, chỉ đồ thị, hoặc chỉ KV) là không đủ cho bộ nhớ tác tử.
- Nêu tên ba kho lưu trữ song song của Mem0 và mục tiêu tối ưu hóa của từng kho.
- Mô tả cơ chế chấm điểm dung hợp (fusion scoring) của Mem0 — độ liên quan (relevance), tầm quan trọng (importance), độ mới (recency) — và lý do tại sao nó là tổng có trọng số thay vì phân cấp.
- Hiện thực hóa hệ thống bộ nhớ ba kho lưu trữ mô hình hóa bằng thư viện chuẩn Python với phương thức `add()` ghi vào cả ba kho và `search()` dung hợp kết quả trả về.

## Vấn đề

Mỗi kho lưu trữ chỉ phù hợp tối ưu cho một trong ba nhóm truy vấn:

- **Độ tương đồng ngữ nghĩa (Semantic similarity)** — "chúng ta đã thảo luận những gì về hiện tượng trôi dạt tác tử (agent drift) tuần trước?" Kho lưu trữ vector chiếm ưu thế tuyệt đối; KV và đồ thị hoàn toàn bỏ sót.
- **Tra cứu dữ kiện (Fact lookup)** — "số điện thoại của người dùng là gì?" Kho lưu trữ KV tối ưu nhất; vector gây lãng phí tính toán, còn đồ thị là quá mức cần thiết.
- **Suy luận mối quan hệ (Relationship reasoning)** — "những khách hàng nào dùng chung một thực thể thanh toán hóa đơn?" Kho lưu trữ đồ thị phát huy hiệu quả; vector và KV không thể giải quyết được.

Tác tử trong môi trường sản xuất thường phát ra cả ba dạng truy vấn này trong cùng một phiên làm việc. Một bộ nhớ chỉ dùng một kho lưu trữ duy nhất chắc chắn sẽ luôn thất bại trước hai trong ba dạng câu hỏi đó. Đóng góp mang tính đột phá của Mem0 là liên kết cả ba kho lưu trữ này đằng sau một bề mặt API `add`/`search` duy nhất đi kèm hàm chấm điểm dung hợp kết quả.

## Khái niệm

### Ba kho lưu trữ song song

Mem0 (arXiv:2504.19413, tháng 4 năm 2025) xử lý thao tác `add(text, user_id, metadata)` như sau:

1. Trích xuất các dữ kiện tiềm năng từ văn bản (bước xử lý do LLM đảm nhiệm).
2. Ghi từng dữ kiện vào kho vector (dưới dạng embedding) phục vụ tìm kiếm ngữ nghĩa.
3. Ghi từng dữ kiện vào kho KV với khóa là bộ giá trị (user_id, fact_type, entity) để tra cứu với độ phức tạp $O(1)$.
4. Ghi từng dữ kiện vào kho đồ thị (Mem0g) dưới dạng các cạnh có định kiểu (typed edges) phục vụ truy vấn quan hệ.

Khi gọi hàm `search(query, user_id)`:

1. Kho vector trả về top-k theo độ đo cosine của embedding.
2. Kho KV trả về các kết quả khớp trực tiếp dựa trên khóa (user_id, type, entity) được suy ra từ truy vấn.
3. Kho đồ thị trả về đồ thị con có thể tiếp cận được từ các thực thể trong truy vấn.
4. Một tầng chấm điểm tiến hành dung hợp kết quả từ cả ba kho.

### Chấm điểm dung hợp (Fusion scoring)

```text
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Độ liên quan (Relevance)** — độ đo cosine của vector, sự khớp chính xác của KV, trọng số đường đi trên đồ thị.
- **Tầm quan trọng (Importance)** — được gắn nhãn tại thời điểm ghi hoặc được học tự động (một số dữ kiện quan trọng hơn: tên, mã định danh, chính sách bảo mật).
- **Độ mới (Recency)** — suy giảm theo hàm mũ (exponential decay) theo thời gian kể từ lần ghi hoặc đọc gần nhất.

Các trọng số được tinh chỉnh theo từng sản phẩm cụ thể. Gán `w_recency` cao hơn cho tác tử hội thoại; gán `w_importance` cao hơn cho tác tử tuân thủ quy chuẩn (compliance agents); gán `w_relevance` cao hơn cho tác tử tìm kiếm truy xuất thông tin.

### Mem0g và Suy luận theo Thời gian (Temporal Reasoning)

Mem0g bổ sung một bộ phát hiện xung đột. Khi một dữ kiện mới mâu thuẫn với một cạnh đồ thị hiện có, cạnh cũ sẽ bị đánh dấu là không còn hiệu lực (invalid) chứ không bị xóa bỏ. Các truy vấn theo thời gian ("thành phố người dùng sinh sống vào tháng 3 là gì?") sẽ duyệt qua đồ thị con có hiệu lực tại thời điểm đó.

Đây chính là hành vi đạt chuẩn tuân thủ mà mô hình vô hiệu hóa dữ liệu của Letta đã khái quát hóa.

### Các con số đối chuẩn (Benchmark numbers)

Bài báo Mem0 công bố các số liệu đối chuẩn (2025):

- **LoCoMo** (bộ nhớ hội thoại dạng dài): 91.6
- **LongMemEval** (bộ nhớ sự kiện với phạm vi dài): 93.4
- **BEAM 1M** (bộ đo đối chuẩn bộ nhớ 1 triệu token): 64.1

Các đường cơ sở so sánh (LLM ngữ cảnh đầy đủ 128k, kho vector phẳng, kho KV phẳng) đều thua kém hơn 10 điểm. Bản thân các chỉ số đối chuẩn không phải là lý do duy nhất để lựa chọn công nghệ — mô hình vận hành mới là yếu tố quyết định — tuy nhiên các con số này chứng minh thiết kế dung hợp mang lại sự khác biệt thực chất chứ không chỉ là sai số làm tròn.

### Phân loại Phạm vi (Scope taxonomy)

Mem0 phân chia bộ nhớ theo phạm vi:

- **Bộ nhớ người dùng (User memory)** — bền vững qua các phiên, liên kết theo khóa `user_id`.
- **Bộ nhớ phiên (Session memory)** — chỉ tồn tại trong một luồng hội thoại đơn lẻ.
- **Bộ nhớ tác tử (Agent memory)** — trạng thái riêng của từng thực thể tác tử (instance).

Mỗi thao tác ghi phải chọn một phạm vi nhất định. Việc truy xuất có thể truy vấn chéo các phạm vi với trọng số được chỉ định riêng cho từng phạm vi. Sự bất cẩn trong việc hòa trộn các phạm vi chính là nguyên nhân dẫn đến các sự cố như "trợ lý vô tình tiết lộ dự án của Bob cho Alice nghe".

### Những điểm mô hình này dễ gặp sự cố

- **Trôi dạt vector nhúng (Embedding drift).** Kết quả vector nhìn có vẻ chuẩn xác trên vài trăm truy vấn đầu tiên nhưng sẽ suy giảm chất lượng khi tập dữ liệu phình to. Giải pháp: định kỳ nhúng lại (re-embedding) top-N bản ghi được sử dụng nhiều nhất.
- **Mở rộng schema KV mất kiểm soát (KV schema creep).** Bộ ba `(user_id, type, entity)` thoạt nhìn rất đơn giản cho đến khi mỗi nhóm phát triển tự ý bổ sung một giá trị `type` riêng. Hãy kiểm toán tập hợp các type theo định kỳ hàng quý.
- **Bùng nổ đồ thị (Graph explosion).** Một bộ trích xuất hoạt động kém ổn định có thể thêm 50 cạnh chỉ từ một tin nhắn. Hãy đặt giới hạn trần cho số cạnh ghi vào đồ thị trên mỗi lệnh gọi `add`; loại bỏ các cạnh có độ tin cậy thấp.

```figure
ae-memory-fusion
```

## Xây dựng Thực tế

`code/main.py` hiện thực hóa mô hình ba kho lưu trữ bằng thư viện chuẩn Python:

- `VectorStore` — mô phỏng độ tương đồng dựa trên mức độ trùng lặp token thô thay cho mô hình embedding thực tế.
- `KVStore` — từ điển (dict) tra cứu theo khóa `(user_id, fact_type, entity)`.
- `GraphStore` — các cạnh có định kiểu (subject, relation, object, valid).
- `Mem0` — giao diện facade tầng trên với `add()`, `search()`, bộ chấm điểm dung hợp, và khả năng truy xuất nhận biết phạm vi.
- Một vết thực thi minh họa trên cuộc hội thoại đa người dùng, đa phiên.

Chạy chương trình:

```bash
python3 code/main.py
```

Kết quả đầu ra hiển thị ba đường dẫn gợi nhớ độc lập cùng với danh sách top-k đã qua dung hợp. Hãy thử thay đổi trọng số chấm điểm ở phần đầu hàm `main()` để quan sát sự biến chuyển trong thứ hạng sắp xếp.

## Ứng dụng Thực tế

- **Mem0 (Giấy phép Apache 2.0)** — sẵn sàng cho môi trường sản xuất. Có thể tự lưu trữ kết hợp Postgres + Qdrant + Neo4j, hoặc dùng dịch vụ đám mây quản lý.
- **Letta** — mô hình ba tầng cốt lõi/gợi nhớ/lưu trữ; cho phép tự tích hợp các hạ tầng vector và đồ thị tùy chọn.
- **Zep** — giải pháp thương mại thay thế tích hợp đồ thị tri thức theo thời gian và bộ trích xuất dữ kiện.
- **Xây dựng tùy chỉnh** — dành cho trường hợp bạn cần kiểm soát tuyệt đối bộ trích xuất (tuân thủ quy định) hoặc trọng số dung hợp (các tác tử giọng nói nơi độ mới chiếm ưu thế áp đảo).

## Đóng gói Sản phẩm

`outputs/skill-hybrid-memory.md` tạo ra một khung sườn bộ nhớ ba kho lưu trữ hoàn chỉnh với bộ chấm điểm dung hợp, hệ thống phân loại phạm vi, và cơ chế vô hiệu hóa theo thời gian.

## Bài tập

1. Thay thế độ tương đồng vector đồ chơi bằng một mô hình embedding thực sự (sentence-transformers, Ollama, OpenAI embeddings). Đo lường recall@10 trên một cuộc trò chuyện dài tổng hợp. Thứ hạng có bị trôi dạt sau 1000 thao tác ghi không?
2. Bổ sung truy vấn theo thời gian: `search(query, as_of=timestamp)`. Chỉ trả về các bản ghi có hiệu lực tại hoặc trước mốc thời gian đó. Kho lưu trữ nào đòi hỏi nhiều công sức điều chỉnh nhất?
3. Hiện thực hóa bộ phát hiện xung đột: nếu một dữ kiện mới mâu thuẫn với một cạnh đồ thị hiện có, hãy vô hiệu hóa cạnh cũ và ghi nhật ký cả hai. Kiểm thử với kịch bản "người dùng sống tại Berlin" -> "người dùng sống tại Lisbon."
4. Nâng cấp bộ chấm điểm dung hợp để bổ sung chiều thông tin `user_feedback` (đánh giá thích/không thích trên các bản ghi được truy xuất). Bạn làm thế nào để ngăn chặn hiện tượng thao túng thuật toán (tác tử chỉ chăm chăm trả về các bản ghi mà nó từng được khen)?
5. Đọc tài liệu Mem0 (`docs.mem0.ai`). Chuyển đổi mã mô phỏng sang các lệnh gọi máy khách `mem0` thực tế. So sánh chất lượng truy xuất trên cùng 20 truy vấn kiểm thử.

## Thuật ngữ Then chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Bộ nhớ lai (Hybrid memory) | "Vector cộng đồ thị cộng KV" | Ba kho lưu trữ được ghi song song, được dung hợp khi truy xuất |
| Trích xuất dữ kiện (Fact extraction) | "Nạp dữ liệu vào bộ nhớ" | Bước xử lý dùng LLM để bóc tách văn bản thành các bộ ba (thực thể, quan hệ, dữ kiện) |
| Chấm điểm dung hợp (Fusion scoring) | "Xếp hạng độ liên quan" | Tổng có trọng số của độ liên quan, tầm quan trọng, và độ mới |
| Phạm vi (Scope) | "Không gian tên bộ nhớ" | Người dùng / Phiên / Tác tử — xác định đối tượng nào được phép thấy dữ liệu nào |
| Mem0g | "Đồ thị bộ nhớ" | Các cạnh có định kiểu kèm tính hợp lệ theo thời gian dành cho truy vấn quan hệ |
| Vô hiệu hóa theo thời gian (Temporal invalidation) | "Xóa mềm" | Đánh dấu các cạnh mâu thuẫn là không hợp lệ; tuyệt đối không xóa cứng |
| Trôi dạt vector nhúng (Embedding drift) | "Thối rữa truy xuất" | Chất lượng vector giảm sút khi kho ngữ liệu phình to; cần nhúng lại theo định kỳ |

## Tài liệu Tham khảo

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — Bài báo gốc
- [Tài liệu Mem0](https://docs.mem0.ai/platform/overview) — API sản xuất, các bộ SDK, dịch vụ đám mây
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — Tiền đề của khái niệm ngữ cảnh ảo
- [Letta, Blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — Thiết kế ba tầng tương đồng
