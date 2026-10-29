# Công thức huấn luyện VLM trọng số mở: Những yếu tố thực sự tạo nên khác biệt

> Văn liệu mô hình ngôn ngữ - thị giác (Vision-Language Model - VLM) trọng số mở (open-weight) giai đoạn 2024-2026 là một rừng các bảng nghiên cứu cắt bỏ thành phần (ablation study). MM1 của Apple đã thử nghiệm 13 tổ hợp giữa bộ mã hóa hình ảnh (image encoder), bộ kết nối (connector) và tỷ lệ pha trộn dữ liệu (data mix). Molmo của Viện AI Allen (Allen AI) đã chứng minh rằng các mô tả chi tiết do con người viết đánh bại phương pháp chưng cất tri thức (distillation) từ GPT-4V. Cambrian-1 đã tiến hành so sánh hơn 20 bộ mã hóa thị giác. Idefics2 đã hình thức hóa không gian thiết kế năm trục (five-axis design space). Prismatic VLM đã so sánh 27 công thức huấn luyện trên một benchmark đối chứng có kiểm soát. Giữa muôn vàn dữ liệu đó, một tập hợp nhỏ các kết quả cốt lõi vẫn đứng vững qua các công trình nghiên cứu: bộ mã hóa hình ảnh quan trọng hơn kiến trúc bộ kết nối, hỗn hợp dữ liệu lại quan trọng hơn cả hai yếu tố trên, và phần chú thích chi tiết từ con người vượt trội hơn hẳn dữ liệu tổng hợp chưng cất. Bài học này đọc và phân tích những bảng số liệu đó để bạn không phải tự mình mò mẫm lại từ đầu.

**Type:** Học + thực hành (Learn + lab)
**Languages:** Python (thư viện chuẩn, bộ phân tích bảng ablation + công cụ chọn công thức)
**Prerequisites:** Giai đoạn 12 · 05 (LLaVA cơ sở)
**Time:** ~180 phút

## Mục tiêu học tập

- Gọi tên không gian thiết kế VLM năm trục: bộ mã hóa hình ảnh (image encoder), bộ kết nối (connector), mô hình ngôn ngữ lớn (LLM), hỗn hợp dữ liệu (data mix), lịch trình phân giải (resolution schedule).
- Đọc hiểu bảng thử nghiệm loại trừ (ablation table) của MM1 / Idefics2 / Cambrian-1 và dự đoán nút điều chỉnh (knob) nào sẽ tác động làm chuyển dịch một benchmark cụ thể.
- Lựa chọn công thức tối ưu (bộ mã hóa, bộ kết nối, dữ liệu, độ phân giải) cho một VLM mới dựa trên ngân sách tính toán (compute budget) và tổ hợp nhiệm vụ mục tiêu.
- Giải thích lý do tại sao phần chú thích chi tiết do con người viết lại đánh bại quá trình chưng cất từ GPT-4V ở cùng một số lượng token.

## Vấn đề cần giải quyết

Hàng trăm mô hình VLM trọng số mở đang tồn tại. Phần lớn khoảng cách giữa một mô hình "tốt" và một mô hình "tiên tiến nhất" (state-of-the-art) không nằm ở sự đột phá kiến trúc. Yếu tố quyết định nằm ở dữ liệu, lịch trình phân giải và lựa chọn bộ mã hóa. Biết chính xác nút điều chỉnh nào cần xoay đầu tiên khi mô hình của bạn hoạt động kém sẽ giúp bạn tránh được sai lầm tiêu tốn hàng triệu giờ tính toán GPU (GPU hours).

Làn sóng năm 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) chủ yếu dựa trên quá trình tiền huấn luyện cặp ảnh - chú thích (caption-pair pretraining) kết hợp với tập dữ liệu LLaVA-Instruct-150k. Đây là đường cơ sở tốt, nhưng mức trần hiệu năng chỉ chạm mốc khoảng 35% trên benchmark MMMU.

Làn sóng năm 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLM) đã tiến hành các thử nghiệm cắt bỏ thành phần toàn diện và thấu đáo. Kết quả thu được vừa bất ngờ vừa mang tính thực tiễn cao.

## Khái niệm cốt lõi

### Không gian thiết kế năm trục

Idefics2 (Laurençon et al., 2024) đã định danh 5 trục thiết kế chính:

1. **Bộ mã hóa hình ảnh (Image encoder):** CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Các bộ mã hóa khác nhau về kích thước patch, độ phân giải hỗ trợ và mục tiêu tiền huấn luyện (pretraining objective).
2. **Bộ kết nối (Connector):** MLP (2-4 lớp), Q-Former (32 truy vấn + cơ chế chú ý chéo - cross-attention), Perceiver Resampler (64 truy vấn), C-Abstractor (kết hợp tích chập và gộp song tuyến tính - bilinear pooling).
3. **Mô hình ngôn ngữ (Language model):** Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Kích thước LLM chiếm phần lớn chi phí tham số của hệ thống.
4. **Dữ liệu huấn luyện (Training data):** Các cặp ảnh - chú thích (CC3M, LAION), tài liệu xen kẽ văn bản - hình ảnh (OBELICS, MMC4), dữ liệu tinh chỉnh chỉ dẫn (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. **Lịch trình phân giải (Resolution schedule):** Cố định 224/336/448, AnyRes (chia mảnh linh hoạt), hoặc độ phân giải động tự nhiên (native dynamic). Lịch trình có thể tăng dần trong quá trình huấn luyện hoặc giữ hằng số.

Mọi VLM trong môi trường sản xuất đều phải đưa ra lựa chọn trên từng trục này. Phần lớn phương sai trong điểm số MMMU được giải thích bởi các trục 1, 4 và 5 — chứ không phải do bạn chọn bộ kết nối nào.

### Trục 1: bộ mã hóa > bộ kết nối

Phần 3.2 trong bài báo MM1 đã chỉ ra: việc chuyển đổi từ CLIP ViT-L/14 sang SigLIP SO400m/14 đã tăng thêm hơn 3 điểm trên MMMU. Trong khi đó, việc thay đổi bộ kết nối từ MLP sang Perceiver Resampler chỉ mang lại mức cải thiện chưa đầy 1 điểm. Idefics2 cũng tái hiện kết quả tương tự: SigLIP > CLIP, và Q-Former ≈ MLP ≈ Perceiver khi xét ở cùng một số lượng token thị giác.

Thử nghiệm "So găng các bộ mã hóa thị giác Cambrian" (Cambrian Vision Encoders Match-Up - Tong et al., 2024) đã chạy hơn 20 bộ mã hóa trên benchmark tập trung vào thị giác (CV-Bench). Đứng đầu bảng xếp hạng là sự kết hợp giữa DINOv2 và SigLIP; CLIP nằm ở nhóm giữa; ImageBind và ViT-MAE xếp ở nhóm thấp hơn. Khoảng cách điểm từ CLIP ViT-L đến DINOv2 ViT-g/14 là ~5-7 điểm trên CV-Bench.

Lựa chọn mặc định vào năm 2026 cho các VLM mã nguồn mở là SigLIP 2 SO400m/14 để thu thập các đặc trưng ngữ nghĩa dày đặc, đôi khi được ghép nối (concatenate) với các đặc trưng từ DINOv2 ViT-g/14 (cơ chế "Spatial Vision Aggregator" của Cambrian áp dụng cách tiếp cận này).

### Trục 2: kiến trúc bộ kết nối không tạo khác biệt đáng kể

MM1, Idefics2, Prismatic và MM-Interleaved đều đi đến cùng một kết luận: tại một số lượng token thị giác cố định, kiến trúc bộ kết nối hầu như không tạo ra sự khác biệt. Một bộ MLP 2 lớp áp dụng trên các patch đã qua gộp trung bình (mean-pooled) đạt hiệu năng chênh lệch chưa tới 1 điểm so với Q-Former 32 truy vấn ở cùng ngân sách token.

Yếu tố thực sự quan trọng là số lượng token (token count). Nhiều token thị giác hơn = tốn nhiều chi phí tính toán LLM hơn = hiệu năng tốt hơn cho đến một ngưỡng nhất định trước khi bước vào giai đoạn lợi suất giảm dần (diminishing returns). 64 token cho mỗi ảnh là quá ít để nhận dạng ký tự quang học (OCR). 576-1024 token là điểm ngọt (sweet spot) cho hầu hết các VLM mở. Mức 2048+ token chỉ thực sự phát huy tác dụng đối với tài liệu phức tạp và biểu đồ.

Việc chọn Q-Former hay MLP là bài toán về chi phí, không phải bài toán chất lượng: Q-Former giới hạn số lượng token ở mức 32-64 bất kể độ phân giải của ảnh; trong khi MLP phát ra toàn bộ các token patch. Đối với đầu vào độ phân giải cao, Q-Former giúp tiết kiệm ngữ cảnh LLM; đối với độ phân giải thấp, sự chênh lệch hiệu năng chỉ là nhiễu ngẫu nhiên.

### Trục 3: kích thước LLM quyết định mức trần hiệu năng

Tăng gấp đôi kích thước LLM từ 7B lên 13B luôn bổ sung thêm từ 2-4 điểm trên MMMU một cách ổn định trên mọi nghiên cứu VLM. Khi chạm mốc 70B, mô hình sẽ bão hòa hầu hết các benchmark hiện tại. Mức trần suy luận đa phương thức của VLM bị giới hạn bởi mức trần suy luận ngôn ngữ của chính LLM đó — bộ mã hóa thị giác chỉ đóng vai trò cung cấp tín hiệu đầu vào chứ không thể suy luận thay cho mô hình ngôn ngữ.

Đây là lý do tại sao Qwen2.5-VL-72B và Claude Opus 4.7 áp đảo hoàn toàn trên MMMU-Pro và ScreenSpot-Pro: bộ não ngôn ngữ của chúng quá lớn. Một mô hình VLM 7B không thể thay thế cho một VLM 70B chỉ bằng cách thiết kế bộ kết nối thông minh hơn.

### Trục 4: dữ liệu — mô tả chi tiết từ con người đánh bại chưng cất

Molmo + PixMo (Deitke et al., 2024) là kết quả nghiên cứu năm 2024 mà bất kỳ ai làm VLM cũng nên đọc. Nhóm nghiên cứu tại Allen AI đã yêu cầu các chuyên viên gán nhãn mô tả hình ảnh thông qua các đoạn ghi âm giọng nói kéo dài 1-3 phút (speech-to-text), thu về 712.000 hình ảnh kèm chú thích dày đặc. Hoàn toàn không có dữ liệu chưng cất từ GPT-4V trong quá trình huấn luyện.

Molmo-72B đã đánh bại Llama-3.2-90B-Vision trên 11/11 benchmark. Bước nhảy vọt đó không đến từ kiến trúc — nó đến từ chất lượng của chú thích. Các chú thích chi tiết do con người xây dựng chứa lượng thông tin gấp 5-10 lần trên mỗi bức ảnh so với các mô tả ngắn thu thập trên web, đồng thời giữ vững độ xác thực thực tế (factually grounded) trong khi dữ liệu chưng cất từ GPT-4V thường xuyên bị ảo giác (hallucination).

ShareGPT4V (Chen et al., 2023) và Cauldron (Idefics2) cũng đi theo hướng tương tự với sự kết hợp giữa chú thích của con người và GPT-4V. Xu hướng cho giai đoạn 2026 đã rõ ràng: mật độ thông tin chú thích > số lượng chú thích > sự tiện lợi của chưng cất.

### Trục 5: độ phân giải và lịch trình phân giải

Các thử nghiệm ablation của Idefics2 cho thấy: tăng độ phân giải từ 384 lên 448 tăng 1-2 điểm. Tăng từ 448 lên 980 với kỹ thuật chia nhỏ hình ảnh (AnyRes) tăng thêm 3-5 điểm trên các benchmark OCR. Huấn luyện với độ phân giải phẳng cố định sẽ sớm chững lại ở mức độ chính xác trung bình; áp dụng lịch trình tăng dần độ phân giải (bắt đầu ở 224, kết thúc ở 448 hoặc độ phân giải tự nhiên) giúp huấn luyện nhanh hơn và đạt đỉnh hiệu năng cao hơn.

Cambrian-1 đã nghiên cứu sự đánh đổi giữa độ phân giải và số lượng token: ở một mức chi phí tính toán cố định, bạn có thể chọn nhiều token hơn ở độ phân giải thấp hơn hoặc ít token hơn ở độ phân giải cao hơn. Độ phân giải cao giành chiến thắng đối với tác vụ OCR; trong khi ít phân giải - nhiều token hơn lại chiến thắng trong việc hiểu khung cảnh tổng thể (general scene understanding).

Công thức huấn luyện thực tế năm 2026: Giai đoạn 1 huấn luyện cố định ở 384, Giai đoạn 2 huấn luyện với độ phân giải động lên tới 1280 cho các tác vụ đòi hỏi khả năng đọc chữ mạnh.

### So sánh đối chứng từ Prismatic

Prismatic VLM (Karamcheti et al., 2024) là bài báo đã kiểm soát chặt chẽ toàn bộ các trục thiết kế. Cùng một LLM 13B, cùng một tập dữ liệu chỉ dẫn, cùng một quy trình đánh giá — chỉ thay đổi duy nhất một trục tại mỗi lần thử nghiệm. Kết quả phân rã:

- Số lượng token thị giác trên mỗi ảnh giải thích khoảng ~60% phương sai hiệu năng.
- Lựa chọn bộ mã hóa hình ảnh giải thích ~20%.
- Kiến trúc bộ kết nối giải thích ~5%.
- Tất cả các yếu tố còn lại (hỗn hợp dữ liệu, bộ điều chỉnh lịch học, tốc độ học LR) đóng góp ~15% còn lại.

Mặc dù đây là một phép phân rã xấp xỉ, nhưng nó là câu trả lời rõ ràng nhất trong văn liệu nghiên cứu cho câu hỏi: "Tôi nên thử nghiệm loại trừ (ablate) yếu tố nào trước tiên?"

### Bộ công thức đề xuất cho năm 2026

Dựa trên các bằng chứng thực nghiệm, công thức VLM mã nguồn mở mặc định cho một dự án mới vào năm 2026 bao gồm:

- **Bộ mã hóa (Encoder):** SigLIP 2 SO400m/14 ở độ phân giải tự nhiên với NaFlex, ghép nối thêm DINOv2 ViT-g/14 để lấy đặc trưng dày đặc nếu cần thực hiện phân đoạn hình ảnh hoặc định vị tọa độ (grounding).
- **Bộ kết nối (Connector):** MLP 2 lớp trên các token patch. Bỏ qua Q-Former trừ khi bạn bị giới hạn nghiêm ngặt về số lượng token.
- **Mô hình ngôn ngữ (LLM):** Qwen2.5 / Llama-3.1 / Gemma 2; dùng bản 7B để tối ưu chi phí, bản 70B để tối ưu chất lượng, quyết định theo độ trễ mục tiêu (target latency).
- **Dữ liệu (Data):** PixMo + ShareGPT4V + Cauldron, bổ sung thêm dữ liệu chỉ dẫn đặc thù cho từng tác vụ nghiệp vụ.
- **Độ phân giải (Resolution):** Động (tối thiểu 256, tối đa 1280 pixel theo cạnh dài).
- **Lịch trình (Schedule):** Giai đoạn 1 căn chỉnh (chỉ đóng băng và huấn luyện projector), Giai đoạn 2 tinh chỉnh toàn diện (full fine-tune), Giai đoạn 3 tinh chỉnh theo tác vụ chuyên biệt.

Từng lựa chọn mặc định ở trên đều bắt nguồn từ các thử nghiệm ablation định lượng trong các bài báo được trích dẫn ở cuối bài học.

```figure
l5-vlm-recipe-knobs
```

## Ứng dụng thực tế

Tệp `code/main.py` là một bộ phân tích bảng ablation và công cụ gợi ý công thức. Chương trình mã hóa các bảng ablation được cô đọng từ MM1 và Idefics2, cho phép bạn truy vấn:

- "Với ngân sách X và tác vụ Y, công thức nào sẽ chiến thắng?"
- "Nếu tôi đổi từ CLIP sang SigLIP trên Llama 7B, mức chênh lệch MMMU dự kiến là bao nhiêu?"
- "Tôi nên thực hiện ablation trên trục nào trước tiên để đạt câu trả lời với độ tin cậy 80%?"

Đầu ra là danh sách các công thức được xếp hạng kèm theo mức thay đổi benchmark dự kiến và khuyến nghị "thành phần nên thử nghiệm trước tiên".

## Đóng gói triển khai

Bài học này tạo ra tệp artifact `outputs/skill-vlm-recipe-picker.md`. Khi nhận đầu vào là cơ cấu tác vụ mục tiêu, ngân sách tính toán và ngưỡng độ trễ yêu cầu, kỹ năng này sẽ phát ra một công thức hoàn chỉnh (bộ mã hóa, bộ kết nối, LLM, tỷ lệ pha trộn dữ liệu, lịch trình phân giải) kèm theo các trích dẫn thực nghiệm chứng minh cho từng quyết định. Điều này ngăn các kỹ sư mất thời gian thiết lập lại các bảng ablation của Idefics2 mỗi khi bắt đầu một dự án VLM mới.

## Bài tập

1. Đọc Phần 3.2 của bài báo MM1. Với một LLM 2B cố định ở ngân sách 50 triệu ảnh, bộ mã hóa nào giành chiến thắng? Kết quả này có bị đảo ngược khi chuyển sang LLM 13B không? Tại sao?

2. Cambrian-1 nhận thấy rằng việc ghép nối DINOv2 + SigLIP vượt trội hơn việc dùng riêng rẽ từng bộ mã hóa trên các benchmark tập trung vào thị giác, nhưng không mang lại thêm tín hiệu nào trên MMMU. Hãy dự đoán benchmark nào sẽ tăng điểm và benchmark nào sẽ đi ngang.

3. Mục tiêu của bạn là xây dựng một agent thao tác giao diện người dùng di động (mobile UI agent) trên nền LLM 2B. Hãy chọn bộ mã hóa, bộ kết nối, độ phân giải và hỗn hợp dữ liệu. Biện minh cho từng lựa chọn bằng một bảng ablation cụ thể.

4. Molmo phát hành hai phiên bản 4B và 72B. Bản 4B có năng lực cạnh tranh với các VLM đóng kích thước 7B; bản 72B đánh bại Llama-3.2-90B-Vision trên 11/11 benchmark. Điều này gợi mở cho bạn điều gì về giả thuyết mức trần kích thước LLM?

5. Thiết kế một bảng ablation nhằm cô lập chất lượng hỗn hợp dữ liệu khỏi chất lượng của bộ mã hóa trên một VLM 7B. Cần tối thiểu bao nhiêu lượt chạy huấn luyện (training runs)? Đề xuất các thiết lập cấu hình cho bốn trục.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Phân tích cắt bỏ (Ablation) | "Xoay một núm điều chỉnh" | Huấn luyện nhiều lượt chạy chỉ khác nhau đúng một trục trong không gian thiết kế, giữ nguyên mọi yếu tố khác |
| Bộ kết nối (Connector) | "Cầu nối" / "bộ chiếu" (projector) | Module có thể huấn luyện giúp ánh xạ đầu ra của bộ mã hóa thị giác vào không gian token của LLM (MLP, Q-Former, Perceiver) |
| Chú thích chi tiết từ con người (Detailed human caption) | "Chú thích dày đặc" (dense caption) | Đoạn văn bản mô tả nhiều câu do con người viết (thường từ 80-300 token), phong phú hơn nhiều so với văn bản alt text trên web |
| Chưng cất (Distillation) | "Chú thích từ GPT-4V" | Dữ liệu huấn luyện được tạo ra bởi một VLM thương mại mạnh hơn; thuận tiện nhưng dễ kế thừa hiện tượng ảo giác |
| AnyRes / độ phân giải động (dynamic res) | "Nhánh phân giải cao" | Chiến lược đưa hình ảnh lớn hơn độ phân giải gốc của bộ mã hóa vào mô hình thông qua chia mảnh (tiling) hoặc M-RoPE |
| Lịch trình tăng phân giải (Resolution ramp) | "Giáo trình huấn luyện" (curriculum) | Lịch trình huấn luyện bắt đầu ở độ phân giải thấp rồi tăng dần, giúp tăng tốc độ học căn chỉnh ở giai đoạn đầu |
| Benchmark trọng thị giác (Vision-centric bench) | "CV-Bench / BLINK" | Bộ đánh giá tập trung kiểm tra khả năng cảm nhận thị giác chi tiết thay vì suy luận nặng về ngôn ngữ |
| PixMo | "Tập dữ liệu của Molmo" | Tập dữ liệu gồm 712.000 hình ảnh kèm chú thích dày đặc của Allen AI; giọng nói con người được phiên âm thành chú thích chi tiết |

## Tài liệu tham khảo thêm

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
