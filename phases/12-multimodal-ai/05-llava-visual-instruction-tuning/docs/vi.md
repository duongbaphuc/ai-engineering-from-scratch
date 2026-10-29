# LLaVA và Tinh Chỉnh Theo Chỉ Dẫn Thị Giác

> LLaVA (tháng 4 năm 2023) là kiến trúc đa phương thức được sao chép nhiều nhất trên thế giới. Nó thay thế Q-Former của BLIP-2 bằng một mạng Perceptron đa tầng (MLP) 2 lớp, thay thế cơ chế chú ý chéo có cổng của Flamingo bằng thao tác nối tiếp token (token concatenation) đơn giản, và được huấn luyện trên 158.000 lượt hội thoại chỉ dẫn thị giác do GPT-4 sinh ra từ các chú thích thuần văn bản. Bất kỳ kỹ sư nào xây dựng một mô hình thị giác - ngôn ngữ (VLM - Vision-Language Model) trong giai đoạn từ 2023 đến 2026 đều từng tạo ra một biến thể nào đó của LLaVA. LLaVA-1.5 đã nâng cấp bộ chiếu MLP và bộ mã hóa 336px. LLaVA-NeXT bổ sung cơ chế lát gạch AnyRes cho đầu vào độ phân giải cao. LLaVA-OneVision đã thống nhất hình ảnh đơn, đa hình ảnh và video trong một công thức duy nhất. Bài học này phân tích công thức đó, triển khai bộ chiếu và giải thích tại sao "đơn giản hơn lại chiến thắng".

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (stdlib, bộ chiếu + bộ tạo mẫu câu chỉ dẫn)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 02 (CLIP), Giai đoạn 11 (Kỹ nghệ LLM — Tinh chỉnh theo chỉ dẫn / Instruction Tuning)  
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Xây dựng một bộ chiếu MLP 2 lớp ánh xạ các vector nhúng mảng ảnh ViT (số chiều 1024) sang số chiều nhúng của LLM (số chiều 4096).
- Nắm vững công thức hai giai đoạn của LLaVA: (1) căn chỉnh bộ chiếu (projector alignment) trên 558k cặp chú thích ảnh, (2) tinh chỉnh theo chỉ dẫn thị giác (visual instruction tuning) trên 158k lượt hội thoại do GPT-4 sinh ra.
- Xây dựng một câu nhắc theo định dạng LLaVA với token giữ chỗ hình ảnh, câu nhắc hệ thống (system prompt) và các lượt hội thoại người dùng/trợ lý.
- Giải thích lý do cộng đồng chuyển từ Q-Former sang MLP dù Q-Former có lợi thế lớn về tiết kiệm ngân sách token.

## Vấn đề

Q-Former của BLIP-2 (Bài học 12.03) nén một hình ảnh xuống còn 32 token. Gọn gàng, hiệu quả và đạt điểm chuẩn rất tốt. Tuy nhiên, nó tồn tại hai vấn đề lớn.

Thứ nhất, Q-Former có thể huấn luyện được nhưng hàm mất mát của nó không phải là tác vụ cuối cùng. Giai đoạn 1 huấn luyện ITC+ITM+ITG. Giai đoạn 2 huấn luyện hàm mất mát mô hình ngôn ngữ (LM loss). Các truy vấn học một biểu diễn trung gian mà sau đó LLM phải giải mã lại. Thông tin bị thất thoát qua nút thắt cổ chai này.

Thứ hai, Q-Former chiếm 188 triệu tham số, và ở quy mô năm 2023 của LLaVA, bạn phải thiết kế đồng thời nó với LLM mục tiêu. Đổi LLM đồng nghĩa với việc phải huấn luyện lại Q-Former. Đổi bộ mã hóa thị giác cũng phải huấn luyện lại. Mỗi sự kết hợp là một dự án nghiên cứu và phát triển riêng biệt.

Câu trả lời của LLaVA đơn giản đến bất ngờ: lấy toàn bộ 576 token mảng ảnh của ViT, đưa từng token qua một mạng MLP 2 lớp (`1024 → 4096 → 4096`), và đổ toàn bộ 576 token đó vào chuỗi đầu vào của LLM. Không có nút thắt cổ chai. Không có tiền huấn luyện giai đoạn 1 trên các mục tiêu phức tạp. Chỉ cần huấn luyện MLP trực tiếp bằng hàm mất mát mô hình ngôn ngữ.

Dữ liệu đến từ đâu? Phát kiến thứ hai của LLaVA: sử dụng GPT-4 (phiên bản thuần văn bản) để tạo dữ liệu chỉ dẫn. Đưa cho GPT-4 chú thích COCO và danh sách các hộp bao (bounding boxes) của một bức ảnh, yêu cầu nó tạo ra các cuộc hội thoại, đoạn mô tả chi tiết và các câu hỏi suy luận phức tạp. Kết quả là 158.000 lượt chỉ dẫn - phản hồi hoàn toàn miễn phí mà không cần con người dán nhãn thủ công.

Kết quả: một VLM có thể chạy huấn luyện trên 8 GPU A100 trong đúng một ngày, đánh bại Flamingo trên điểm chuẩn MMMU, và cung cấp một checkpoint nguồn mở mà cộng đồng có thể dễ dàng mở rộng. Đến cuối năm 2023, nó đã truyền cảm hứng cho hơn 50 nhánh phát triển phái sinh.

## Khái niệm

### Kiến trúc (The architecture)

LLaVA-1.5 ở quy mô 7B:
- Bộ mã hóa thị giác: CLIP ViT-L/14 @ 336 (đóng băng ở giai đoạn 1, có thể mở đóng băng ở giai đoạn 2).
- Bộ chiếu (Projector): MLP 2 lớp với hàm kích hoạt GELU, kích thước `1024 → 4096 → 4096`.
- LLM: Vicuna-7B v1.5 (phiên bản 13B dùng Vicuna-13B; các bản cải tiến sau này chuyển sang Llama-3.1-8B).

Lượt truyền xuôi (forward pass) trên một câu nhắc gồm hình ảnh + văn bản:

```text
img -> ViT -> 576 mảng ảnh có số chiều 1024
mảng ảnh -> MLP -> 576 token có số chiều 4096
câu nhắc: hệ thống + token giữ chỗ "<image>" + câu hỏi người dùng
thay thế token <image> bằng 576 token thị giác đã chiếu
đưa toàn bộ chuỗi vào LLM
giải mã câu trả lời
```

Hình ảnh chiếm 576 token trong ngữ cảnh của LLM. Ở ngữ cảnh 2048 token, nó để lại 1472 token cho văn bản. Nhưng ở ngữ cảnh 32k token, con số này chỉ là một phần rất nhỏ không đáng kể.

### Giai đoạn 1: Căn chỉnh bộ chiếu (Stage 1: projector alignment)

Đóng băng ViT. Đóng băng LLM. Chỉ huấn luyện bộ chiếu MLP 2 lớp. Tập dữ liệu: 558k cặp hình ảnh - chú thích (LAION-CC-SBU). Hàm mất mát: mô hình ngôn ngữ trên chú thích, điều kiện hóa theo các token hình ảnh đã chiếu.

Chỉ trong một epoch duy nhất với batch size 128, bước này hoàn tất trong vài giờ. Bộ chiếu học cách ánh xạ từ không gian ViT sang không gian LLM mà không cần giám sát đặc thù cho tác vụ cụ thể.

### Giai đoạn 2: Tinh chỉnh theo chỉ dẫn thị giác (Stage 2: visual instruction tuning)

Mở đóng băng bộ chiếu (vẫn tiếp tục được huấn luyện). Mở đóng băng LLM (thường là toàn bộ mô hình, đôi khi dùng LoRA). Huấn luyện trên 158.000 lượt chỉ dẫn thị giác.

Dữ liệu chỉ dẫn chính là điểm mấu chốt. Liu cùng các cộng sự đã tạo ra nó bằng cách:
1. Lấy một hình ảnh từ tập COCO.
2. Trích xuất mô tả văn bản (5 chú thích do con người viết + danh sách tọa độ hộp bao).
3. Gửi tới GPT-4 với ba mẫu câu nhắc:
   - Hội thoại (Conversation): "Hãy tạo một cuộc đối thoại qua lại giữa người dùng và trợ lý về hình ảnh này."
   - Mô tả chi tiết (Detailed description): "Hãy đưa ra một đoạn mô tả phong phú và chi tiết về bức ảnh."
   - Suy luận phức tạp (Complex reasoning): "Hãy đặt một câu hỏi đòi hỏi tư duy suy luận về bức ảnh, sau đó trả lời nó."
4. Phân tích cú pháp đầu ra của GPT-4 thành các cặp (chỉ dẫn, phản hồi).

Toàn bộ quy trình này không chạm trực tiếp vào hình ảnh — chỉ dựa vào mô tả văn bản. GPT-4 tự "tưởng tượng" ra nội dung hình ảnh hợp lý. Dù có một lượng nhiễu nhất định, nó vẫn hoạt động xuất sắc: 158.000 lượt là đủ để khai phóng khả năng đối thoại đa phương thức.

### Tại sao cộng đồng sao chép kiến trúc này

- Không cần tinh chỉnh các hàm mất mát phức tạp ở giai đoạn 1. Chỉ dùng một hàm mất mát mô hình ngôn ngữ xuyên suốt.
- Bộ chiếu có thể huấn luyện xong chỉ trong vài giờ thay vì nhiều ngày.
- Có thể hoán đổi LLM dễ dàng (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) chỉ bằng cách huấn luyện lại bộ chiếu.
- Quy trình tạo dữ liệu chỉ dẫn thị giác dùng GPT-4 rất rẻ để tái tạo cho một miền nghiệp vụ mới.

### LLaVA-1.5 và LLaVA-NeXT

LLaVA-1.5 (tháng 10 năm 2023) bổ sung:
- Trộn thêm dữ liệu tác vụ học thuật (VQA, OKVQA, RefCOCO) vào quá trình tinh chỉnh chỉ dẫn.
- Cải thiện câu nhắc hệ thống.
- Thay thế lớp tuyến tính đơn ban đầu bằng bộ chiếu MLP 2 lớp, nâng cấp CLIP ViT-L/14 ở độ phân giải 336px (576 token) thay vì 224px (256 token). Độ dài chuỗi huấn luyện vẫn giữ ở mức 2048 token.
- Đề xuất biến thể chia lưới độ phân giải cao (HD grid-splitting) trong bài báo, sau này được đưa vào LLaVA-NeXT dưới tên gọi AnyRes.

LLaVA-NeXT (tháng 1 năm 2024) bổ sung:
- AnyRes: chia các hình ảnh độ phân giải cao thành lưới các mảnh cắt 336x336 (kích thước 2x2 hoặc 1x3), kèm một ảnh thu nhỏ toàn cục (global low-res thumbnail). Mỗi mảnh cắt tạo ra 576 token; tổng cộng khoảng 2880 token thị giác cho mỗi ảnh. Độ chính xác trên các bài kiểm tra OCR và biểu đồ tăng vọt.
- Hòa trộn dữ liệu chỉ dẫn chất lượng cao hơn với ShareGPT4V (chú thích dày đặc từ GPT-4V).
- Nền tảng LLM mạnh mẽ hơn (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Bài học 12.08 sẽ phân tích sâu về OneVision. Tóm lược: vẫn giữ bộ chiếu đó, nhưng được huấn luyện theo một giáo trình bao quát cả hình ảnh đơn, đa hình ảnh và video trong cùng một mô hình với ngân sách token thị giác dùng chung.

### So sánh với Q-Former

| Tiêu chí | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Token thị giác mỗi ảnh | 32 | 576 (cơ bản) hoặc 2880 (AnyRes) |
| Tham số huấn luyện được | 188M + LM | ~22M (7B) + LM |
| Hàm mất mát Giai đoạn 1 | ITC+ITM+ITG | Chỉ dùng LM |
| Thay thế LLM | Cần huấn luyện lại nhiều | Hoán đổi dễ với huấn luyện tối thiểu |
| Đa hình ảnh | Khó khăn | Tự nhiên (nối chuỗi token) |
| Video | Khó khăn | Tự nhiên (nối token từng khung hình) |
| Ngân sách token | Nhỏ gọn | Lớn |

MLP chiến thắng về tính đơn giản và sự linh hoạt của token. Q-Former chiến thắng về tiết kiệm ngân sách token. Đến cuối năm 2023, ngân sách token không còn là rào cản nghiêm ngặt (ngữ cảnh LLM đã mở rộng lên 32k–128k+) nên sự đơn giản đã thống trị.

### Định dạng câu nhắc (The prompt format)

```text
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` là một token giữ chỗ. Trước khi đưa qua bộ tách từ (tokenization), nó được thay thế bằng 576 token thị giác đã chiếu (hoặc 2880 token với AnyRes). Bộ tách từ nhìn thấy một chuỗi dài hơn một chút so với dữ liệu huấn luyện thông thường, nhưng LLM vẫn xử lý trơn tru vì Giai đoạn 1 đã dạy nó làm quen với dạng biểu diễn này.

### Tính kinh tế về tham số (Parameter economy)

Phân bổ tham số của LLaVA-1.5-7B:
- CLIP ViT-L/14 @ 336: 303 triệu tham số (đóng băng ở Giai đoạn 1, thường mở ở Giai đoạn 2).
- Bộ chiếu (2 lớp tuyến tính): ~22 triệu tham số học được.
- Llama-7B: 7 tỷ tham số.
- Tổng cộng: 7,3 tỷ tham số. Huấn luyện trong Giai đoạn 2: toàn bộ 7B + 22M của bộ chiếu.

Chi phí huấn luyện cho Giai đoạn 2: ~20 giờ trên cụm 8 GPU A100. Đây là con số mang tính quyết định — chỉ một ngày, một máy chủ (node), có thể tái lập hoàn toàn. Đó là lý do LLaVA nhanh chóng lan rộng khắp cộng đồng nghiên cứu.

````figure
mm-llava-projector
````

## Thực hành (Use It)

`code/main.py` triển khai:

1. Bộ chiếu MLP 2 lớp (số chiều 16 → 32 → 32 ở quy mô minh họa) bằng Python thuần.
2. Quy trình xây dựng câu nhắc: câu nhắc hệ thống + token `<image>` được thay thế bằng $N$ token đã chiếu + lượt người dùng + vị trí giữ chỗ cho phản hồi của trợ lý.
3. Bộ trực quan hóa thể hiện khối 576 token thị giác chiếm bao nhiêu phần trăm trong các cửa sổ ngữ cảnh LLM (2k / 32k / 128k).

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-llava-vibes-eval.md`. Cho một checkpoint thuộc họ LLaVA, kỹ năng này sẽ chạy bộ đánh giá trực quan gồm 10 câu nhắc (3 câu chú thích ảnh, 3 câu hỏi đáp VQA, 2 câu suy luận logic, 2 câu từ chối trả lời) và xuất ra bảng điểm trực quan. Đây không phải điểm chuẩn học thuật; mà là bài kiểm thử nhanh (smoke test) để khẳng định bộ chiếu và LLM đang kết nối chính xác.

## Bài tập (Exercises)

1. Tính số lượng tham số học được của bộ chiếu MLP 2 lớp với kích thước `1024 → 4096 → 4096`. Khi tính cả hàm GELU và bias, nó chiếm tỷ lệ bao nhiêu phần trăm trong LLaVA-13B?
2. Xây dựng câu nhắc LLaVA cho trường hợp "từ chối" (refusal) — bức ảnh chứa một cá nhân riêng tư. Viết câu trả lời kỳ vọng của trợ lý. Tại sao LLaVA nên từ chối trường hợp này ở chế độ zero-shot và dữ liệu huấn luyện nào là cần thiết để củng cố hành vi từ chối đó?
3. Đọc phần AnyRes trong bài viết blog LLaVA-NeXT. Tính tổng số token thị giác cho một hình ảnh 1344x672 với AnyRes. So sánh với mức 576 token cơ bản ở độ phân giải 336x336.
4. Bộ chiếu LLaVA Giai đoạn 1 được huấn luyện bằng hàm mất mát mô hình ngôn ngữ trên các chú thích ảnh. Điều gì sẽ xảy ra nếu bạn bỏ qua Giai đoạn 1 và đi thẳng vào Giai đoạn 2 (tinh chỉnh chỉ dẫn thị giác)? Trích dẫn nghiên cứu thực nghiệm Prismatic VLMs (arXiv:2402.07865) để trả lời.
5. LLaVA-Instruct-150k dùng GPT-4 cùng các chú thích COCO để sinh câu chỉ dẫn. Đối với một miền mới (ảnh X-quang y tế, ảnh vệ tinh viễn thám), hãy mô tả quy trình dữ liệu 4 bước để sinh chỉ dẫn miền. Những rủi ro sai sót nào có thể xảy ra ở từng bước?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Projector (Bộ chiếu) | "Cầu nối MLP" | Mạng MLP 2 lớp với hàm kích hoạt GELU ánh xạ chiều ViT sang chiều LLM |
| Image token (Token hình ảnh) | "Vị trí giữ chỗ <image>" | Ký hiệu đánh dấu trong câu nhắc được thay thế bởi N token thị giác đã chiếu trước khi suy luận |
| Visual instruction tuning (Tinh chỉnh chỉ dẫn thị giác) | "LLaVA Giai đoạn 2" | Quá trình huấn luyện trên bộ ba (hình ảnh, chỉ dẫn, phản hồi) do GPT-4 tạo ra |
| Stage 1 alignment (Căn chỉnh Giai đoạn 1) | "Tiền huấn luyện bộ chiếu" | Đóng băng ViT và LLM, huấn luyện bộ chiếu bằng hàm mất mát LM trên chú thích |
| AnyRes | "Lát gạch đa mảnh cắt" | Chia hình ảnh độ phân giải cao thành lưới các mảnh cắt và nối chuỗi token thị giác của từng mảnh |
| LLaVA-Instruct | "Dữ liệu do GPT-4 sinh" | 158k cặp chỉ dẫn - phản hồi được tổng hợp từ chú thích COCO + GPT-4 |
| Vision encoder freeze (Đóng băng bộ mã hóa thị giác) | "Khóa mạng xương sống" | Trọng số CLIP không được cập nhật ở Giai đoạn 1, đôi khi giữ nguyên cả ở Giai đoạn 2 |
| ShareGPT4V | "Chú thích chất lượng cao" | 1 triệu chú thích dày đặc do GPT-4V tạo ra, dùng để căn chỉnh đạt độ chính xác cao hơn |
| VQA | "Hỏi đáp trên hình ảnh" | Tác vụ trả lời câu hỏi tự do bằng ngôn ngữ tự nhiên về một bức ảnh |
| Prismatic VLMs | "Bài báo phân tích không gian thiết kế" | Nghiên cứu thực nghiệm của Karamcheti (2024) kiểm thử có hệ thống các lựa chọn bộ chiếu và dữ liệu |

## Tài liệu tham khảo (Further Reading)

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — Bài báo gốc về LLaVA.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — Bài báo LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — Tập dữ liệu chú thích ảnh dày đặc.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — Thực nghiệm bóc tách không gian thiết kế VLM.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — Thống nhất ảnh đơn, đa ảnh và video.
