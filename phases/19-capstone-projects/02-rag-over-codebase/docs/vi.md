# Đồ án Tốt nghiệp 02 — RAG trên Kho mã nguồn (Tìm kiếm Ngữ nghĩa Đa Kho lưu trữ)

> Đến năm 2026, mọi tổ chức kỹ thuật nghiêm túc đều vận hành một hệ thống tìm kiếm mã nguồn nội bộ có khả năng thấu hiểu ngữ nghĩa thay vì chỉ so khớp chuỗi văn bản thuần túy. Sourcegraph Amp, tính năng codebase answers của Cursor, enterprise graph của Augment, repomap của Aider, hay giao thức MCP nội bộ của Pinterest — đều có chung một kiến trúc cốt lõi: Nạp dữ liệu từ nhiều kho mã nguồn (cross-repo ingestion), phân tích cú pháp bằng tree-sitter, nhúng các đoạn mã ở cấp độ hàm và lớp (AST-aware chunking), tìm kiếm lai (hybrid search), tái xếp hạng (re-ranking), và tổng hợp câu trả lời kèm trích dẫn chính xác. Đồ án tốt nghiệp này yêu cầu bạn xây dựng một hệ thống xử lý 2 triệu dòng mã trải rộng trên 10 kho lưu trữ, đồng thời duy trì khả năng tái lập chỉ mục gia tăng (incremental re-indexing) sau mỗi lần git push.

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (Nền tảng NLP), Phase 7 (Mô hình Transformer), Phase 11 (Kỹ thuật LLM), Phase 13 (Công cụ), Phase 17 (Hạ tầng)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** ~30 giờ

## Bài toán (Problem)

Bước sang năm 2026, mọi tác tử lập trình hàng đầu đều tích hợp một tầng truy xuất mã nguồn chuyên biệt, bởi lẽ cửa sổ ngữ cảnh mở rộng đơn thuần không thể giải quyết triệt để các câu hỏi xuyên suốt nhiều kho mã nguồn. Cửa sổ ngữ cảnh 1 triệu token của Claude rất hữu ích, nhưng nó không thể loại bỏ nhu cầu truy xuất có xếp hạng. Tìm kiếm tương đồng cosine ngây thơ trên các khối mã thô sẽ làm ngộ độc kết quả trả về do mã nguồn tự động sinh (generated code), sự trùng lặp trong monorepo, và phần đuôi dài (long tail) của các ký hiệu hiếm khi được import. Lời giải chuẩn mực trong môi trường sản xuất là tìm kiếm lai (hybrid search kết hợp dense vector và BM25) trên các phân mảnh nhận biết cây cú pháp trừu tượng (AST-aware chunks) đi kèm bộ tái xếp hạng (re-ranker), được trợ lực bởi đồ thị tham chiếu ký hiệu (symbol graph).

Bạn chỉ có thể thực sự nắm vững điều này khi tự tay lập chỉ mục cho một hệ thống thực tế gồm nhiều kho lưu trữ — thay vì một kho mã nguồn đồ chơi trong bài hướng dẫn — và trực tiếp đo lường các chỉ số MRR@10, độ trung thực trích dẫn (citation faithfulness), cùng độ tươi mới gia tăng (incremental freshness). Những điểm nghẽn nghiêm trọng nhất mang bản chất hạ tầng: một monorepo với 100.000 tệp tin, một lệnh push thay đổi một nửa số tệp trong dự án, hoặc một truy vấn đòi hỏi phải đối chiếu qua bốn kho lưu trữ khác nhau mới có thể đưa ra câu trả lời chính xác.

## Khái niệm (Concept)

Một đường ống nạp dữ liệu nhận biết AST (AST-aware ingestion pipeline) sẽ phân tích cú pháp từng tệp tin bằng tree-sitter, bóc tách các nút hàm (function) và lớp (class), rồi thực hiện cắt phân mảnh tại ranh giới của các nút thay vì cắt theo cửa sổ token cố định. Mỗi phân mảnh được gán ba biểu diễn:
1. Vector embedding dày đặc (dense embedding từ Voyage-code-3 hoặc nomic-embed-code),
2. Các từ khóa thưa BM25 (sparse BM25 terms),
3. Bản tóm tắt ngắn bằng ngôn ngữ tự nhiên. Bản tóm tắt bổ sung thêm phương thức truy xuất thứ ba — người dùng hỏi "cách thức phân quyền X hoạt động thế nào", bản tóm tắt sẽ nhắc đến từ khóa "authz", ngay cả khi trong mã nguồn hàm đó chỉ đặt tên là `check_permission`.

Quá trình truy xuất diễn ra theo mô hình lai (hybrid retrieval). Một truy vấn sẽ kích hoạt đồng thời cả tìm kiếm vector dày đặc và BM25, hợp nhất top-k kết quả, rồi chuyển tập hợp hợp nhất này qua bộ tái xếp hạng dạng cross-encoder (Cohere rerank-3 hoặc bge-reranker-v2-gemma-2b). Danh sách sau khi tái xếp hạng được nạp vào mô hình tổng hợp ngữ cảnh dài (Claude Sonnet 4.7 có kích hoạt prompt caching, hoặc Llama 3.3 70B tự lưu trữ) kèm chỉ thị bắt buộc trích dẫn mọi luận điểm theo đường dẫn tệp và dải số dòng (file:line range). Các câu trả lời thiếu trích dẫn sẽ bị bộ lọc hậu xử lý loại bỏ ngay lập tức.

Độ tươi mới gia tăng là bài toán hạ tầng cốt tử. Lệnh `git push` kích hoạt webhook để phân tích diff: tệp nào đã thay đổi, ký hiệu nào đã biến động. Chỉ những phân mảnh bị ảnh hưởng mới cần nhúng lại vector. Các cạnh quan hệ ký hiệu xuyên tệp (lệnh import, lệnh gọi phương thức) liên quan sẽ được tính toán lại. Chỉ mục duy trì tính nhất quán mà không cần xử lý lại toàn bộ 2 triệu dòng mã sau mỗi commit.

## Kiến trúc (Architecture)

```text
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             phân tích tree-sitter + phân mảnh AST
                           |
            +--------------+----------------+
            v              v                v
      dense embedding  chỉ mục BM25    tóm tắt mã (LLM)
       (Voyage / bge)    (Tantivy)       (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                    đồ thị ký hiệu (Neo4j / kuzu)
                            |
  truy vấn --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 ngữ cảnh 1M
                            |
                            v
                 câu trả lời + trích dẫn file:line
```

## Công nghệ sử dụng (Stack)

- Phân tích cú pháp: tree-sitter với ngữ pháp biên dịch sẵn cho 17 ngôn ngữ (Python, TS, Rust, Go, Java, C++, v.v.)
- Vector embedding dày đặc: Voyage-code-3 (dịch vụ đám mây) hoặc nomic-embed-code-v1.5 (tự lưu trữ), dự phòng bằng bge-code-v1
- Chỉ mục thưa: Tantivy (Rust) hỗ trợ BM25F, gán trọng số theo trường (tên ký hiệu so với thân hàm)
- Vector DB: Qdrant 1.12 hỗ trợ tìm kiếm lai, hoặc pgvector + pgvectorscale cho hệ thống dưới 50 triệu vector
- Mô hình tóm tắt phân mảnh: Claude Haiku 4.5 hoặc Gemini 2.5 Flash, có lưu bộ nhớ đệm lời nhắc (prompt caching)
- Bộ tái xếp hạng: Cohere rerank-3 hoặc bge-reranker-v2-gemma-2b tự lưu trữ
- Điều phối (Orchestration): LlamaIndex Workflows cho luồng nạp dữ liệu, LangGraph cho tác tử truy vấn
- Mô hình tổng hợp (Synthesizer): Claude Sonnet 4.7 (cửa sổ ngữ cảnh 1M) có bật prompt caching
- Đồ thị ký hiệu: Neo4j (dịch vụ quản lý) hoặc kuzu (cơ sở dữ liệu nhúng) để lưu vết các cạnh quan hệ import và call
- Khả năng quan sát: Langfuse ghi nhận span chi tiết cho từng bước truy xuất và tổng hợp

```figure
ce-hybrid-retrieval
```

## Các bước triển khai (Build It)

1. **Bộ duyệt nạp dữ liệu (Ingestion walker).** Duyệt qua lịch sử git trên mỗi hook push. Thu thập danh sách các tệp tin thay đổi. Với mỗi tệp, phân tích cú pháp bằng tree-sitter, trích xuất các nút hàm và lớp cùng dải mã nguồn trọn vẹn của chúng. Xuất ra bản ghi phân mảnh `{repo, path, start_line, end_line, symbol, body}`.

2. **Tóm tắt phân mảnh (Chunk summarizer).** Gom cụm các phân mảnh thành các lượt gọi Haiku 4.5 theo lô, áp dụng prompt caching cho phần mở đầu hệ thống (system preamble). Lời nhắc: "Tóm tắt hàm này trong đúng một câu văn, nêu rõ giao ước công khai (public contract) và các tác dụng phụ (side effects) của nó." Lưu bản tóm tắt đi kèm phân mảnh tương ứng.

3. **Bể nhúng vector (Embedding pool).** Thiết lập hai hàng đợi song song: hàng đợi dense (Voyage-code-3 theo lô 128) và hàng đợi tóm tắt (cùng mô hình, nhưng thao tác trên chuỗi tóm tắt). Ghi các vector vào Qdrant kèm dữ liệu tải trọng (payload) `{repo, path, start_line, end_line, symbol, kind}`.

4. **Chỉ mục BM25 (BM25 index).** Cấu hình chỉ mục Tantivy với trọng số trường cụ thể: trọng số tên ký hiệu là 4, thân ký hiệu là 1, và bản tóm tắt là 2. Cơ chế này cho phép xử lý tối ưu cả truy vấn "tìm hàm có tên là X" lẫn "tìm hàm thực hiện chức năng X".

5. **Đồ thị ký hiệu (Symbol graph).** Với mỗi phân mảnh, ghi lại các cạnh quan hệ: import (tệp này dùng ký hiệu Y từ kho Z), lệnh gọi hàm (hàm này gọi phương thức M trên lớp C), và kế thừa (inheritance). Lưu trữ trong kuzu. Sử dụng tại thời điểm truy vấn để mở rộng phạm vi tìm kiếm vượt qua biên giới các kho lưu trữ.

6. **Tác tử truy vấn (Query agent).** Xây dựng đồ thị LangGraph với ba nút chức năng: `retrieve` thực thi song song tìm kiếm dense + BM25, khử trùng lặp theo bộ ba `(repo, path, symbol)`; `rerank` chạy mô hình cross-encoder trên top 50 và giữ lại top 10; `synth` gọi Claude Sonnet 4.7 với các phân mảnh đã tái xếp hạng được nạp vào ngữ cảnh, lưu bộ nhớ đệm system prompt, và yêu cầu bắt buộc trích dẫn file:line.

7. **Thực thi quy tắc trích dẫn (Citation enforcement).** Phân tích cú pháp đầu ra của mô hình; bất kỳ khẳng định nào không đi kèm mỏ neo định dạng `(repo/path:start-end)` sẽ bị đánh dấu để yêu cầu mô hình tạo lại hoặc bị loại bỏ. Chỉ trả về cho người dùng câu trả lời đã được kiểm chứng trích dẫn đầy đủ.

8. **Tái lập chỉ mục gia tăng (Incremental re-index).** Trên mỗi webhook gửi đến, tính toán phần diff ở cấp độ ký hiệu. Chỉ nhúng lại các phân mảnh có nội dung văn bản bị thay đổi. Tính toán lại các cạnh ký hiệu cho các phân mảnh có phần import biến động. Yêu cầu đo lường: một commit chứa 50 tệp thay đổi phải được tái lập chỉ mục hoàn tất dưới 60 giây trên tập hợp mã 2 triệu dòng (2M-LOC).

9. **Đánh giá hiệu năng (Eval).** Gán nhãn cho tập kiểm thử gồm 100 câu hỏi đa kho mã nguồn với đáp án chuẩn xác xác thực đến từng file:line. Đo lường MRR@10, nDCG@10, độ trung thực trích dẫn (tỷ lệ các khẳng định có mỏ neo kiểm chứng được), và độ trễ p50/p99.

## Ứng dụng thực tế (Use It)

```bash
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Triển khai thành phẩm (Ship It)

Kỹ năng thành phẩm sẵn sàng sử dụng được lưu trữ tại `outputs/skill-codebase-rag.md`. Khi nhận vào một tập hợp kho mã nguồn, nó sẽ khởi dựng đường ống nạp dữ liệu, chỉ mục lai và tác tử truy vấn, đồng thời cung cấp câu trả lời có trích dẫn chuẩn xác cho bất kỳ câu hỏi nào mang tính chất đa kho lưu trữ. Bảng tiêu chí đánh giá:

| Trọng số | Tiêu chí | Phương thức đo lường |
|:-:|---|---|
| 25 | Chất lượng truy xuất | Điểm số MRR@10 và nDCG@10 trên tập đánh giá kiểm thử 100 câu hỏi |
| 20 | Độ trung thực trích dẫn | Tỷ lệ các khẳng định trong câu trả lời có mỏ neo định danh file:line kiểm chứng được |
| 20 | Độ trễ và khả năng mở rộng | Độ trễ truy vấn p95 ở mức 10k QPS trên quy mô tập dữ liệu đã lập chỉ mục |
| 20 | Độ chính xác tái lập chỉ mục gia tăng | Thời gian tính từ khi git push đến khi nội dung mới có thể tìm kiếm được với commit 50 tệp |
| 15 | Trải nghiệm người dùng và định dạng phản hồi | Khả năng nhấp trực tiếp vào trích dẫn, xem trước đoạn trích (snippet), hỗ trợ câu hỏi nối tiếp |
| **100** | | |

## Bài tập mở rộng (Exercises)

1. Hoán đổi Voyage-code-3 sang mô hình tự lưu trữ nomic-embed-code. Đo lường độ chênh lệch chỉ số MRR@10. Đánh giá xem việc bật tính năng tái xếp hạng có giúp thu hẹp khoảng cách hiệu năng hay không.

2. Trộn lẫn 20% mã nguồn tự động sinh (mã khung boilerplate do LLM tạo ra) vào tập dữ liệu và đánh giá lại. Quan sát hiện tượng ngộ độc dữ liệu truy xuất. Bổ sung cờ đánh dấu "generated" vào payload và giảm trọng số các kết quả này khi tìm kiếm.

3. Đo kiểm hiệu năng (benchmark) so sánh tìm kiếm lai của Qdrant với pgvector + pgvectorscale trên quy mô kho mã nguồn hiện tại của bạn. Báo cáo độ trễ p99 khi kích thước lô (batch size) bằng 1.

4. Bổ sung cơ chế kiểm tra trôi dạt ngữ nghĩa (drift check) dựa trên lấy mẫu: Hàng tuần, chạy lại bộ đánh giá 100 câu hỏi. Phát cảnh báo nếu chỉ số MRR@10 tụt giảm hơn 5%.

5. Mở rộng khả năng phân giải ký hiệu xuyên ngôn ngữ (cross-language symbol resolution): Giả sử một hàm Python gọi một dịch vụ Go thông qua gRPC. Sử dụng đồ thị ký hiệu để thiết lập mối liên kết giữa chúng.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| AST-aware chunking (Phân mảnh nhận biết AST) | "Cắt theo cấp độ hàm" | Chia tách mã nguồn tại ranh giới các nút tree-sitter thay vì cắt theo các cửa sổ token cố định |
| Hybrid search (Tìm kiếm lai) | "Dày đặc + thưa" | Chạy song song tìm kiếm vector và BM25, hợp nhất top-k kết quả, sau đó tái xếp hạng |
| Cross-encoder rerank (Tái xếp hạng cross-encoder) | "Xếp hạng giai đoạn hai" | Mô hình chấm điểm đồng thời từng cặp (truy vấn, ứng viên), đạt độ chuẩn xác vượt trội so với cosine |
| Prompt caching (Bộ nhớ đệm lời nhắc) | "Bộ nhớ đệm system prompt" | Tính năng năm 2026 của Claude / OpenAI giúp giảm chi phí token cho tiền tố lặp lại lên tới 90% |
| Symbol graph (Đồ thị ký hiệu) | "Đồ thị mã nguồn" | Tập hợp các cạnh biểu diễn quan hệ import, lệnh gọi hàm, và kế thừa xuyên suốt các tệp và kho lưu trữ |
| Citation faithfulness (Độ trung thực trích dẫn) | "Tỷ lệ câu trả lời có cơ sở" | Tỷ lệ các khẳng định mà người dùng có thể đối chiếu thực tế bằng cách nhấp vào mỏ neo và đọc đoạn mã được tham chiếu |
| Incremental re-index (Tái lập chỉ mục gia tăng) | "Thời gian từ push đến tìm kiếm" | Khoảng thời gian thực tế tính từ khi git push đến khi các ký hiệu thay đổi có thể được truy vấn |

## Tài liệu đọc thêm (Further Reading)

- [Sourcegraph Amp](https://ampcode.com) — Nền tảng trí tuệ mã nguồn đa kho lưu trữ trong môi trường sản xuất
- [Kiến trúc RAG của Sourcegraph Cody](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — Bài viết phân tích sâu đóng vai trò tham chiếu cho đồ án này
- [Aider repo-map](https://aider.chat/docs/repomap.html) — Bản đồ kho mã nguồn có xếp hạng dựa trên tree-sitter
- [Augment Code enterprise graph](https://www.augmentcode.com) — Hệ thống RAG đồ thị ký hiệu thương mại
- [Tài liệu tìm kiếm lai Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/) — Hướng dẫn triển khai tham chiếu
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Chi tiết kỹ thuật về Voyage-code-3
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — Tài liệu tham chiếu mô hình cross-encoder
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — Tài liệu tham chiếu nền tảng nội bộ Pinterest
