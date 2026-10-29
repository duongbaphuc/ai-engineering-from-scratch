# Đồ án Tốt nghiệp 04 — Hỏi đáp Tài liệu Đa phương thức (PDF Ưu tiên Thị giác, Bảng biểu, Biểu đồ)

> Ranh giới kỹ thuật của bài toán hỏi đáp tài liệu (Document QA) năm 2026 đã dịch chuyển triệt để: từ bỏ quy trình truyền thống "OCR rồi mới trích xuất văn bản" (OCR-then-text) để tiến tới mô hình tương tác muộn ưu tiên thị giác (vision-first late interaction). Các mô hình như ColPali, ColQwen2.5 và ColQwen3-omni coi mỗi trang PDF là một hình ảnh độc lập, mã hóa trang tài liệu bằng các biểu diễn tương tác muộn đa vector (multi-vector late interaction), cho phép câu truy vấn trực tiếp chú ý tới từng mẩu ảnh (patches). Trên các báo cáo tài chính 10-K, bài báo khoa học hay ghi chú viết tay, mô hình này vượt trội hoàn toàn so với giải pháp ưu tiên OCR truyền thống. Đồ án tốt nghiệp này yêu cầu bạn xây dựng hoàn chỉnh đường ống từ đầu đến cuối trên 10.000 trang tài liệu và công bố bản đánh giá so sánh trực tiếp với phương pháp OCR-then-text.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (Thị giác máy tính), Phase 5 (Xử lý ngôn ngữ tự nhiên), Phase 7 (Mô hình Transformer), Phase 11 (Kỹ thuật LLM), Phase 12 (Mô hình đa phương thức), Phase 17 (Hạ tầng)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** ~30 giờ

## Bài toán (Problem)

Các doanh nghiệp đang lưu trữ khối lượng khổng lồ tệp PDF mà các đường ống OCR truyền thống thường xuyên làm biến dạng: các bản scan báo cáo 10-K với bảng biểu bị xoay nghiêng, bài báo khoa học dày đặc phương trình phức tạp, biểu đồ trực quan chỉ có ý nghĩa khi nhìn dưới dạng hình ảnh, và các ghi chú viết tay bên lề. Nếu cố gắng ép chúng về định dạng văn bản thuần túy, bạn sẽ đánh mất hơn một nửa lượng tín hiệu thông tin giá trị. Lời giải chuẩn mực năm 2026 là cơ chế truy xuất đa vector tương tác muộn trực tiếp trên các ảnh trang gốc. ColPali (từ Illuin Tech) đặt nền móng cho hướng tiếp cận này; ColQwen2.5-v0.2 và ColQwen3-omni tiếp tục nâng cao độ chính xác. Trên bộ tiêu chuẩn ViDoRe v3, truy xuất ưu tiên thị giác đạt điểm số vượt xa giải pháp OCR-then-text với khoảng cách rõ rệt — và cách biệt càng nới rộng đối với biểu đồ, bảng biểu và chữ viết tay.

Sự đánh đổi ở đây nằm ở dung lượng lưu trữ và độ trễ tính toán. Một biểu diễn embedding ColQwen tạo ra ~2048 vector mẩu ảnh (patch vectors) trên mỗi trang, thay vì chỉ là một vector duy nhất 1024 chiều. Dung lượng lưu trữ thô tăng lên nhanh chóng. Kỹ thuật DocPruner (2026) giúp cắt tỉa 50% số lượng vector mà không làm suy giảm độ chuẩn xác có thể đo lường được. Bạn sẽ lập chỉ mục cho 10.000 trang tài liệu, đo lường chỉ số ViDoRe v3 nDCG@5, phục vụ câu trả lời dưới 2 giây, và đối chiếu trực tiếp với hệ quy chiếu OCR-then-text.

## Khái niệm (Concept)

Cơ chế tương tác muộn (late interaction) đồng nghĩa với việc mỗi token trong câu truy vấn sẽ được tính điểm tương đồng với toàn bộ các token mẩu ảnh của trang tài liệu, sau đó lấy điểm số lớn nhất cho từng token truy vấn và cộng dồn lại (toán tử MaxSim). Nhờ vậy, bạn đạt được khả năng khớp ngữ nghĩa chi tiết ở cấp độ vi mô mà không cần phải nén toàn bộ trang thành một vector gộp duy nhất. Một chỉ mục đa vector (như Vespa, Qdrant multi-vector, hoặc AstraDB) sẽ lưu trữ các embedding theo từng mẩu ảnh và thực thi phép toán MaxSim tại thời điểm truy xuất.

Thành phần trả lời là một mô hình thị giác-ngôn ngữ (Vision-Language Model - VLM), nhận đầu vào là câu truy vấn kèm theo top-k trang tài liệu được truy xuất dưới dạng hình ảnh, sau đó tạo câu trả lời đi kèm các vùng bằng chứng trực quan (hộp giới hạn tọa độ - bounding boxes hoặc số trang tham chiếu). Qwen3-VL-30B, Gemini 2.5 Pro và InternVL3 là những lựa chọn hàng đầu năm 2026. Đối với các trang chứa dày đặc công thức toán học và ký hiệu khoa học, một kênh OCR dự phòng (Nougat, dots.ocr) được ghép nối bổ sung như một nguồn cấp văn bản tùy chọn.

Quy trình đánh giá được tổ chức theo ma trận hai chiều:
- Một trục biểu diễn loại nội dung (đoạn văn bản thuần, bảng biểu phức tạp, biểu đồ cột/đường, ghi chú viết tay, phương trình toán học).
- Trục còn lại biểu diễn phương pháp truy xuất (tương tác muộn ưu tiên thị giác so với OCR-then-text so với mô hình lai).
Mỗi ô trong ma trận ghi nhận điểm số nDCG@5 và độ chính xác của câu trả lời. Báo cáo đối chiếu này chính là sản phẩm nghiệm thu cốt lõi.

## Kiến trúc (Architecture)

```text
Tệp PDF -> Bộ kết xuất trang (PyMuPDF, 180 DPI)
             |
             v
    ColQwen2.5-v0.2 embed (đa vector mỗi trang, ~2048 mẩu ảnh)
             |
             +------> Nén 50% bằng DocPruner
             |
             v
    Chỉ mục đa vector (Vespa hoặc Qdrant multi-vector)
             |
Truy vấn ----+----> Truy xuất top-k trang liên quan (MaxSim)
             |
             v
    Mô hình VLM trả lời: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
      Đầu vào: Truy vấn + ảnh top-k trang + văn bản OCR bổ trợ
             |
             v
    Câu trả lời kèm số trang trích dẫn + vùng bằng chứng
             |
             v
    Giao diện xem Streamlit / Next.js: Hộp đánh dấu phủ trên trang gốc
```

## Công nghệ sử dụng (Stack)

- Kết xuất trang: PyMuPDF (fitz) ở độ phân giải 180 DPI, chuẩn hóa theo chiều dọc
- Mô hình tương tác muộn: ColQwen2.5-v0.2 hoặc ColQwen3-omni (nhóm vidore trên Hugging Face)
- Chỉ mục (Index): Vespa với trường đa vector, hoặc Qdrant multi-vector, hoặc AstraDB hỗ trợ MaxSim
- Cắt tỉa (Pruning): Chính sách DocPruner 2026 (giữ lại các mẩu ảnh có phương sai thông tin cao, nén 50% dung lượng với mức suy giảm độ chính xác < 0.5%)
- Kênh OCR dự phòng (cho phương trình / bảng số liệu dày đặc): dots.ocr hoặc Nougat
- Mô hình VLM trả lời: Qwen3-VL-30B tự lưu trữ hoặc Gemini 2.5 Pro đám mây; InternVL3 làm phương án dự phòng
- Bộ tiêu chuẩn đánh giá: ViDoRe v3 benchmark, M3DocVQA cho suy luận đa trang
- Giao diện người dùng: Next.js 15 kết hợp lớp phủ canvas hiển thị trực quan các vùng bằng chứng

```figure
ce-late-interaction
```

## Các bước triển khai (Build It)

1. **Nạp dữ liệu (Ingest).** Quét qua kho tài liệu gồm 10.000 trang PDF bao gồm báo cáo tài chính 10-K, bài báo khoa học và tài liệu scan. Kết xuất mỗi trang thành ảnh định dạng PNG độ phân giải 1536x2048. Lưu trữ bản ghi `{doc_id, page_num, image_path}`.

2. **Nhúng vector (Embed).** Chạy ColQwen2.5-v0.2 trên từng ảnh trang. Đầu ra có kích thước ~2048 vector mẩu ảnh với số chiều là 128. Áp dụng DocPruner để giữ lại một nửa lượng vector mang tín hiệu thông tin cao nhất. Ghi dữ liệu vào trường đa vector của Vespa hoặc Qdrant multi-vector.

3. **Truy vấn (Query).** Với mỗi truy vấn gửi đến, trích xuất đặc trưng bằng nhánh mã hóa truy vấn (query tower để tạo ra các token embedding). Thực thi phép toán MaxSim trên chỉ mục: với mỗi token truy vấn, tìm tích vô hướng (dot-product) lớn nhất trên toàn bộ các vector mẩu ảnh của trang tài liệu, rồi lấy tổng. Trả về danh sách top-k trang có điểm số cao nhất.

4. **Tổng hợp câu trả lời (Synthesize).** Gọi Qwen3-VL-30B với câu truy vấn cùng ảnh của 5 trang hàng đầu. Lời nhắc: "Chỉ trả lời dựa trên các trang tài liệu được cung cấp. Trích dẫn từng luận điểm theo định dạng (doc_id, page) và chỉ rõ vị trí vùng nội dung (hình vẽ, bảng số liệu, đoạn văn)."

5. **Xác định vùng bằng chứng (Evidence regions).** Hậu xử lý câu trả lời để bóc tách các vùng được trích dẫn. Nếu mô hình VLM xuất ra tọa độ hộp giới hạn (bounding boxes — tính năng có sẵn trên Qwen3-VL), hiển thị chúng dưới dạng các lớp phủ đánh dấu trực quan trên giao diện xem tài liệu.

6. **Kênh OCR bổ trợ.** Với các trang được nhận diện là chứa dày đặc công thức toán học (dựa trên thuật toán phỏng đoán phương sai ảnh), kích hoạt Nougat hoặc dots.ocr và truyền văn bản OCR như một kênh thông tin bổ sung song song cùng hình ảnh.

7. **Đánh giá kiểm chuẩn (Eval).** Chạy kiểm thử trên ViDoRe v3 (chỉ số nDCG@5 của tầng truy xuất) và M3DocVQA (độ chính xác hỏi đáp suy luận đa trang). Đồng thời chạy đường ống OCR-then-text trên cùng kho tài liệu với cùng mô hình tổng hợp. Lập bảng ma trận so sánh chi tiết giữa từng loại nội dung và phương pháp tiếp cận.

8. **Giao diện trực quan (UI).** Xây dựng nguyên mẫu nhanh bằng Streamlit; sau đó phát triển ứng dụng hoàn chỉnh bằng Next.js 15 hỗ trợ hiển thị trang kèm lớp phủ vùng bằng chứng theo thời gian thực.

## Ứng dụng thực tế (Use It)

```bash
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  Biên lợi nhuận hoạt động khu vực EMEA thay đổi từ 18.2% xuống 16.8%, giảm 140 điểm cơ bản (bps).
  trích dẫn: 10-K-2024.pdf trang 88 (Bảng 4, Biên lợi nhuận hoạt động theo phân khúc)
             10-K-2024.pdf trang 92 (Báo cáo thảo luận của ban quản trị - MD&A)
[viewer]     mở tài liệu kèm hộp đánh dấu vùng bằng chứng phủ trên Bảng 4 trang 88
```

## Triển khai thành phẩm (Ship It)

Kỹ năng hoàn chỉnh được bàn giao tại `outputs/skill-doc-qa.md`. Tài liệu mô tả hệ thống hỏi đáp tài liệu đa phương thức ưu tiên thị giác được tối ưu hóa cho một kho tài liệu cụ thể và được đánh giá đối chứng với hệ thống OCR-then-text trên chuẩn ViDoRe v3. Bảng tiêu chí đánh giá:

| Trọng số | Tiêu chí | Phương thức đo lường |
|:-:|---|---|
| 25 | Độ chính xác trên ViDoRe v3 / M3DocVQA | Điểm số chuẩn mực so với baseline OCR-văn bản và bảng xếp hạng công khai |
| 20 | Định vị vùng bằng chứng (Grounding) | Tỷ lệ các vùng được trích dẫn thực sự chứa đoạn thông tin mang câu trả lời |
| 20 | Kỹ thuật tối ưu lưu trữ và độ trễ | Tỷ lệ nén của DocPruner, độ trễ truy xuất p95, độ trễ trả lời p95 |
| 20 | Suy luận đa trang | Độ chính xác trên tập dữ liệu kiểm thử 100 câu hỏi đa trang được gán nhãn thủ công |
| 15 | Trải nghiệm kiểm tra tài liệu gốc (UX) | Độ trực quan của trình xem, độ chính xác của lớp phủ tọa độ, công cụ so sánh trực quan |
| **100** | | |

## Bài tập mở rộng (Exercises)

1. Đo lường hiệu năng so sánh giữa ColQwen2.5-v0.2 và ColQwen3-omni trên cùng một tập tài liệu. Phân tích những trang nào mà một mô hình trả lời đúng còn mô hình kia bỏ sót. Bổ sung nhãn phân loại nội dung vào chỉ mục để định tuyến thông minh theo định dạng trang.

2. Thử nghiệm cắt tỉa vector mẩu ảnh ở mức độ khắt khe hơn (75%, 90%). Tìm ra "vách đá nén" (compression cliff): ngưỡng mà tại đó chỉ số ViDoRe nDCG@5 bắt đầu tụt xuống thấp hơn mức cơ sở của giải pháp OCR truyền thống.

3. Xây dựng đường ống lai (hybrid pipeline): Chạy song song giải pháp OCR-then-text và ColQwen, hợp nhất danh sách kết quả bằng thuật toán RRF (Reciprocal Rank Fusion), rồi tái xếp hạng bằng cross-encoder. Đánh giá xem hệ thống lai có vượt trội hơn từng mô hình đơn lẻ hay không và mang lại hiệu quả lớn nhất ở dạng tài liệu nào.

4. Hoán đổi Qwen3-VL-30B sang mô hình VLM nhỏ gọn hơn (Qwen2.5-VL-7B). Đo lường đường cong đánh đổi giữa độ chuẩn xác và chi phí tài chính (accuracy-per-dollar curve).

5. Mở rộng hỗ trợ tập tài liệu ghi chú viết tay. Kết xuất tập dữ liệu chữ viết tay, mã hóa vector bằng ColQwen, đo lường độ chính xác truy xuất và so sánh trực tiếp với đường ống OCR chuyên dụng cho chữ viết tay.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Late interaction (Tương tác muộn) | "Truy xuất kiểu ColPali" | Từng token trong câu truy vấn được so khớp độc lập với từng mẩu ảnh trang; sau đó MaxSim tổng hợp điểm số |
| Multi-vector (Đa vector) | "Embedding từng mẩu ảnh" | Mỗi tài liệu/trang được đại diện bởi một tập hợp nhiều vector thay vì chỉ một vector gộp duy nhất |
| MaxSim | "Chấm điểm tương tác muộn" | Với mỗi token truy vấn, tìm độ tương đồng cực đại trên các vector tài liệu; sau đó tính tổng |
| DocPruner | "Nén mẩu ảnh" | Kỹ thuật cắt tỉa năm 2026 giúp giữ lại 50% số mẩu ảnh quan trọng nhất với mức suy giảm độ chính xác không đáng kể |
| ViDoRe v3 | "Chuẩn đánh giá truy xuất tài liệu" | Bộ tiêu chuẩn định chuẩn năm 2026 dùng để đo lường năng lực truy xuất tài liệu trực quan |
| Evidence region (Vùng bằng chứng) | "Hộp giới hạn trích dẫn" | Một bounding box trên trang nguồn giúp khoanh vùng chính xác đoạn nội dung chứa câu trả lời |
| OCR fallback (Kênh OCR dự phòng) | "Kênh xử lý phương trình" | Đường ống văn bản chạy song song cùng nhánh thị giác để hỗ trợ các trang chứa nhiều phương trình hoặc bảng số liệu phức tạp |

## Tài liệu đọc thêm (Further Reading)

- [Kho lưu trữ ColPali (Illuin Tech)](https://github.com/illuin-tech/colpali) — Dự án tham chiếu phương pháp truy xuất tài liệu qua tương tác muộn
- [Bài báo ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — Công trình khoa học nền tảng đặt cơ sở cho phương pháp
- [Họ mô hình ColQwen trên Hugging Face](https://huggingface.co/vidore) — Các trọng số mô hình sẵn sàng cho môi trường sản xuất
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — Mô hình cơ sở RAG đa phương thức cho tài liệu đa trang
- [Hướng dẫn đa vector trên Vespa](https://docs.vespa.ai/en/colpali.html) — Ngăn xếp phục vụ tìm kiếm tham chiếu
- [Hỗ trợ đa vector trên Qdrant](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — Giải pháp chỉ mục thay thế
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — Cơ sở dữ liệu đám mây hỗ trợ MaxSim
- [Nougat OCR](https://github.com/facebookresearch/nougat) — Mô hình OCR chuyên dụng cho tài liệu khoa học và công thức toán học
