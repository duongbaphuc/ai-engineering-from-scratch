# Đồ án Tốt nghiệp 12 — Quy trình Hiểu Video Đa phương thức (Phân đoạn Cảnh, Hỏi-Đáp, Tìm kiếm)

> Twelve Labs đã thương mại hóa thành công Marengo + Pegasus. VideoDB phát hành bộ API CRUD dành riêng cho video. Dự án Molmo 2 của AI2 công bố các điểm kiểm tra mô hình thị giác-ngôn ngữ (Vision-Language Model - VLM) mở. Ngữ cảnh siêu dài của Gemini xử lý trực tiếp hàng giờ video nguyên bản. TimeLens-100K định hình chuẩn mực định vị thời gian (temporal grounding) ở quy mô lớn. Quy trình kỹ thuật của năm 2026 đã hoàn thiện: phân đoạn cảnh (scene segmentation), sinh chú thích kèm embedding cho từng cảnh, căn chỉnh bản ghi âm thoại (transcript alignment), lập chỉ mục đa vector (multi-vector index), và phản hồi truy vấn kèm mốc thời gian (bắt đầu, kết thúc) cùng ảnh xem trước khung hình. Đồ án tốt nghiệp này yêu cầu thu nạp 100 giờ video, đạt chuẩn trên các bài kiểm tra công khai, và đo lường hiện tượng ảo giác (hallucination) trên các câu hỏi đếm số lượng và nhận diện hành động.

**Type:** Capstone  
**Languages:** Python (pipeline), TypeScript (UI)  
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)  
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17  
**Time:** ~30 giờ  

## Vấn đề đặt ra

Hỏi-đáp trên video thời lượng dài (long-form video QA) là bài toán đa phương thức tiêu tốn nhiều băng thông và tài nguyên tính toán nhất ở quy mô năm 2026. Mặc dù Gemini 2.5 Pro có thể đọc hiểu trực tiếp một video dài 2 giờ, việc thu nạp 100 giờ video thành một kho ngữ liệu có khả năng truy vấn tức thì vẫn đòi hỏi một chỉ mục ở cấp độ từng phân cảnh. Cấu trúc triển khai thực tế kết hợp phân đoạn cảnh (dùng TransNetV2 hoặc PySceneDetect), sinh chú thích từng cảnh bằng mô hình VLM (Gemini 2.5, Qwen3-VL-Max hoặc Molmo 2), căn chỉnh bản gỡ băng âm thanh (Whisper-v3-turbo với mốc thời gian chi tiết đến từng từ), cùng một chỉ mục đa vector lưu trữ đồng thời chú thích, embedding khung hình và văn bản lời thoại. Quy trình xử lý truy vấn sẽ phản hồi kèm mốc thời gian chính xác (bắt đầu, kết thúc) và ảnh xem trước khung hình đại diện.

Các bài đánh giá sử dụng bộ chuẩn công khai (ActivityNet-QA, NeXT-GQA) kết hợp cùng tập kiểm thử tùy biến 100 câu hỏi gán nhãn thủ công. Hiện tượng ảo giác đối với các câu hỏi đếm số lượng và phân loại hành động là nhóm lỗi kinh điển rất khó xử lý; đồ án này yêu cầu đo lường chi tiết và báo cáo riêng biệt cho nhóm lỗi này.

## Khái niệm cốt lõi

Quá trình thu nạp vận hành đồng thời ba luồng xử lý song song:
1. **Phân đoạn cảnh (Scene segmentation):** Tách video thành các phân cảnh tại các điểm chuyển cảnh (shot boundaries).
2. **Sinh chú thích qua VLM (VLM captioning):** Sinh mô tả văn bản cho mỗi phân cảnh và tạo embedding trực quan từ khung hình then chốt (keyframe).
3. **Căn chỉnh nhận dạng giọng nói (ASR alignment):** Sinh bản gỡ băng lời thoại với mốc thời gian chi tiết theo từng từ.

Ba luồng dữ liệu này được kết hợp lại dựa trên cặp khóa `(scene_id, khoảng thời gian)`. Mỗi phân cảnh được lưu trữ dưới dạng ba loại vector độc lập trong một chỉ mục đa vector (Qdrant): embedding chú thích (caption embedding), embedding khung hình (keyframe embedding), và embedding lời thoại (transcript embedding).

Khi người dùng gửi truy vấn bằng ngôn ngữ tự nhiên, hệ thống sẽ kích hoạt tìm kiếm dày đặc song song trên cả ba không gian vector; kết quả được hợp nhất bằng thuật toán dung hợp hạng nghịch đảo (Reciprocal Rank Fusion - RRF); một bộ chuyển đổi định vị thời gian (theo kiến trúc TimeLens) sẽ tinh chỉnh khoảng thời gian (bắt đầu, kết thúc) chính xác bên trong cảnh có độ khớp cao nhất. Sau đó, bộ tổng hợp VLM (Gemini 2.5 Pro hoặc Qwen3-VL-Max) tiếp nhận câu hỏi + các cảnh hàng đầu + các khung hình cắt trích để đưa ra câu trả lời cuối cùng kèm trích dẫn mốc thời gian và ảnh xem trước.

Việc đo lường ảo giác mang tính sống còn. Các câu hỏi về đếm số lượng ("Có bao nhiêu người bước vào phòng?") và chuỗi hành động ("Người đầu bếp đổ gia vị trước hay khuấy trước?") nổi tiếng là thiếu ổn định. Bạn cần báo cáo độ chính xác của các nhóm câu hỏi này độc lập với các câu hỏi mô tả thông thường.

## Kiến trúc hệ thống

```text
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Ngăn xếp công nghệ

- **Phân đoạn cảnh:** TransNetV2 (mô hình tối tân 2024-2026) hoặc PySceneDetect
- **Nhận dạng giọng nói (ASR):** Whisper-v3-turbo qua faster-whisper kèm mốc thời gian cấp từ
- **Mô hình VLM sinh chú thích & trả lời:** Gemini 2.5 Pro, Qwen3-VL-Max, hoặc Molmo 2
- **Định vị thời gian (Temporal grounding):** Bộ điều hợp huấn luyện theo TimeLens-100K hoặc VideoITG
- **Chỉ mục vector:** Qdrant hỗ trợ đa vector (chú thích / khung hình / bản gỡ băng)
- **Giao diện người dùng:** Next.js 15 với trình phát video HTML5 và dải ảnh thu nhỏ phân cảnh
- **Đánh giá (Eval):** ActivityNet-QA, NeXT-GQA, tập 100 câu hỏi gán nhãn tùy biến
- **Bài chuẩn ảo giác:** Tập con các câu hỏi đếm số lượng và nhận diện hành động được gán nhãn thủ công

```figure
cf-scene-index
```

## Từng bước xây dựng

1. **Bộ thu nạp dữ liệu.** Tiếp nhận URL YouTube hoặc tệp MP4 cục bộ. Giảm độ phân giải xuống 720p nếu cần thiết để tối ưu băng thông. Lưu bản ghi `{video_id, file_path}`.

2. **Phân đoạn cảnh.** Chạy TransNetV2 hoặc PySceneDetect để trích xuất danh sách `[{scene_id, start_ms, end_ms, keyframe_path}]`. Mục tiêu cho 100 giờ video: khoảng 6.000–8.000 phân cảnh.

3. **Gỡ băng âm thanh.** Chạy Whisper-v3-turbo trên kênh âm thanh; xuất mốc thời gian chi tiết đến từng từ; phân tách thành các lát cắt lời thoại tương ứng với từng phân cảnh.

4. **Sinh chú thích qua VLM.** Với mỗi phân cảnh, gọi Gemini 2.5 Pro (hoặc Qwen3-VL-Max) kèm khung hình then chốt và mẫu lời nhắc chú thích ngắn gọn. Tạo ra bản mô tả văn bản cùng vector embedding khung hình.

5. **Lập chỉ mục đa vector.** Thiết lập collection trong Qdrant với ba trường vector định danh riêng. Dữ liệu đính kèm (payload): `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Xử lý truy vấn.** Câu hỏi bằng ngôn ngữ tự nhiên được mã hóa và truy vấn dày đặc song song trên cả ba trường vector; hợp nhất thứ hạng bằng RRF; chọn ra top-k=5 phân cảnh phù hợp nhất.

7. **Định vị thời gian.** Chạy bộ điều hợp kiểu TimeLens trên phân cảnh hàng đầu để tinh chỉnh khoảng thời gian (bắt đầu, kết thúc) chi tiết bên trong cảnh đó.

8. **Tổng hợp câu trả lời qua VLM.** Gọi Gemini 2.5 Pro với câu hỏi của người dùng + 3 đoạn video/khung hình hàng đầu + các đoạn trích lời thoại tương ứng. Bắt buộc mô hình trích dẫn theo định dạng `(video_id, start_ms, end_ms)`.

9. **Đánh giá toàn diện.** Đánh giá trên ActivityNet-QA và NeXT-GQA. Xây dựng tập kiểm thử 100 câu hỏi tùy biến. Báo cáo độ chính xác tổng thể kèm phân tích chi tiết theo từng nhóm (đếm số lượng, chuỗi hành động, mô tả thông thường).

## Vận hành thực tế

```console
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Bàn giao sản phẩm

Tệp `outputs/skill-video-qa.md` mô tả sản phẩm bàn giao. Với một URL YouTube hoặc video tải lên, pipeline sẽ lập chỉ mục các phân cảnh và trả lời các câu hỏi truy vấn kèm trích dẫn mốc thời gian và khung hình minh họa.

| Trọng số | Tiêu chí | Phương pháp đo lường |
|:-:|---|---|
| 25 | Chỉ số IoU định vị thời gian | Chỉ số tương đồng giao hội (Intersection-Over-Union - IoU) trên tập kiểm thử định vị tách biệt |
| 20 | Độ chính xác hỏi-đáp (QA) | Điểm số trên NeXT-GQA và tập 100 truy vấn tùy biến |
| 20 | Thông lượng thu nạp | Số giờ video xử lý được trên mỗi dollar chi phí |
| 20 | Giao diện & Trải nghiệm trích dẫn | Liên kết mốc thời gian, dải hình thu nhỏ, tính năng nhảy đến khung hình |
| 15 | Tỷ lệ ảo giác | Đo lường độ chính xác riêng cho nhóm câu hỏi đếm số lượng và chuỗi hành động |
| **100** | | |

## Bài tập mở rộng

1. Thay thế Gemini 2.5 Pro bằng Qwen3-VL-Max trong khâu sinh chú thích phân cảnh. Báo cáo độ chênh lệch chất lượng chú thích trên một mẫu 50 cảnh được con người thẩm định.

2. Rút gọn embedding khung hình của mỗi cảnh thành một vector gộp duy nhất thay vì kiến trúc đa vector. Đo lường mức độ suy giảm chất lượng truy xuất.

3. Xây dựng chế độ "kiểm tra đếm nghiêm ngặt" (counting strict): bộ tổng hợp trích xuất từng đối tượng được đếm kèm mốc thời gian cụ thể để người dùng có thể nhấp vào xác thực. Đo lường xem việc xác thực bởi người dùng có giúp giảm thiểu ảo giác hay không.

4. Đo chuẩn chi phí thu nạp: tính toán số giờ video xử lý được trên mỗi dollar chi phí trên ba mô hình VLM khác nhau để tìm ra điểm cân bằng tối ưu.

5. Bổ sung tính năng phân đoạn người nói (speaker diarization) trong lời thoại: áp dụng mô hình pyannote phân tách giọng nói trên kênh âm thanh và nhúng văn bản kèm định danh người nói. Minh họa các truy vấn dạng "Alice đã nói gì về vấn đề X?".

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Scene segmentation | "Phát hiện chuyển cảnh" | Cắt video thành các phân cảnh tại các ranh giới góc quay (shot boundaries) |
| Multi-vector index | "Chú thích + khung hình + lời thoại" | Collection trong Qdrant lưu trữ các vector có tên riêng biệt cho từng dạng biểu diễn |
| Temporal grounding | "Định vị thời điểm chính xác" | Tinh chỉnh khoảng thời gian (bắt đầu, kết thúc) cho câu trả lời của truy vấn |
| Frame embedding | "Biểu diễn trực quan" | Vector biểu diễn một khung hình then chốt; dùng để đo độ tương đồng thị giác |
| RRF fusion | "Dung hợp hạng nghịch đảo" | Chiến lược kết hợp nhiều danh sách kết quả xếp hạng; kỹ thuật chuẩn trong tìm kiếm lai |
| Counting hallucination | "Ảo giác đếm số lượng" | Lỗi phổ biến của các mô hình VLM khi trả lời câu hỏi "có bao nhiêu đối tượng X" |
| ActivityNet-QA | "Bộ chuẩn QA video" | Bộ chuẩn đánh giá độ chính xác hỏi-đáp trên video thời lượng dài |

## Tài liệu đọc thêm

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — các điểm kiểm tra VLM mã nguồn mở
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — kỹ thuật định vị thời gian ở quy mô lớn
- [Gemini Video ngữ cảnh dài](https://deepmind.google/technologies/gemini) — tài liệu tham chiếu dịch vụ lưu trữ đám mây
- [VideoDB](https://videodb.io) — tài liệu tham chiếu API dạng CRUD cho video
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — sản phẩm thương mại tham chiếu
- [TransNetV2](https://github.com/soCzech/TransNetV2) — mô hình phân đoạn cảnh chuẩn mực
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — công cụ mã nguồn mở truyền thống
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — bài báo công bố bộ chuẩn đánh giá tham chiếu
