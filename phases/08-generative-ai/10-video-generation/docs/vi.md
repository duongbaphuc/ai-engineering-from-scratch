# Tạo sinh Video (Video Generation)

> Một bức ảnh tĩnh là một tensor 2 chiều. Một đoạn video là một tensor 3 chiều. Về mặt lý thuyết toán học, các nguyên lý là tương đồng; nhưng ngân sách tính toán lại nặng hơn từ 10 đến 100 lần. Mô hình Sora của OpenAI (tháng 2/2024) đã chứng minh điều này hoàn toàn khả thi trên quy mô lớn. Đến năm 2026, Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 và WAN 2.2 đã có thể tạo ra video chất lượng thương mại từ văn bản ở độ phân giải 1080p — và hệ sinh thái trọng số mở (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) chỉ đi sau các dịch vụ đóng khoảng 12 tháng.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Yêu cầu tiên quyết:** Phase 8 · 07 (Khuếch tán trong không gian tiềm ẩn - Latent Diffusion), Phase 7 · 09 (Vision Transformer - ViT), Phase 8 · 06 (Mô hình khuếch tán - DDPM)  
**Thời lượng:** ~45 phút  

## Vấn đề cần giải quyết (The Problem)

Một đoạn video dài 10 giây ở chuẩn 1080p với tốc độ 24 khung hình/giây (fps) chứa 240 khung hình kích thước $1920 \times 1080 \times 3$ pixel. Dung lượng dữ liệu thô lên tới ~1.5 GB cho mỗi clip. Vận hành quá trình khuếch tán trực tiếp trên không gian pixel là điều hoàn toàn bất khả thi. Bạn bắt buộc phải giải quyết các thách thức:

1. **Nén không-thời gian (Spatiotemporal compression).** Một mạng VAE có khả năng mã hóa toàn bộ chuỗi video (chứ không chỉ từng khung hình riêng lẻ) thành một chuỗi các mảnh vá không-thời gian (spatiotemporal patches).
2. **Tính mạch lạc thời gian (Temporal coherence).** Các khung hình liên tiếp phải chia sẻ chung nội dung, ánh sáng, góc máy và danh tính đối tượng xuyên suốt nhiều giây. Mạng nơ-ron phải học cách mô hình hóa chuyển động và quy luật vật lý.
3. **Ngân sách tính toán (Compute budget).** Chi phí huấn luyện mô hình video đắt gấp 10-100 lần so với mô hình ảnh tĩnh trên cùng quy mô số lượng tham số.
4. **Tín hiệu điều kiện (Conditioning).** Văn bản, hình ảnh (khung hình đầu tiên), âm thanh, hoặc một video dẫn đường khác. Hầu hết các mô hình trong sản xuất đều hỗ trợ cả 4 phương thức này.

Kiến trúc giải quyết triệt để bài toán này là **Transformer khuếch tán (Diffusion Transformer - DiT)** áp dụng trên các mảnh vá không-thời gian, được huấn luyện trên các tập dữ liệu khổng lồ gồm bộ ba (câu lệnh, chú thích chi tiết, video). Hàm mất mát khuếch tán vẫn hoàn toàn tương đồng với Bài 06.

## Khái niệm cốt lõi (The Concept)

![Khuếch tán video: chia mảnh vá, xử lý qua DiT, giải mã](../assets/video-generation.svg)

### Chia mảnh vá (Patchify)

Mã hóa video bằng một mạng VAE 3 chiều (học nén không gian và thời gian đồng thời). Tensor tiềm ẩn có kích thước `[T_latent, H_latent, W_latent, C_latent]`. Chia tensor này thành các khối mảnh vá (patches) kích thước `[t_p, h_p, w_p]`. Với các kiến trúc theo trường phái Sora, $t_p = 1$ (mảnh vá theo từng khung hình) hoặc $t_p = 2$ (mảnh vá gộp hai khung hình liên tiếp). Một đoạn video 10 giây 1080p được nén thành khoảng 20.000 đến 100.000 mảnh vá.

### DiT không-thời gian (Spatiotemporal DiT)

Một mạng Transformer tiếp nhận chuỗi mảnh vá phẳng này. Mỗi mảnh vá được gắn một vector nhúng vị trí 3D (thời gian + trục y + trục x). Cơ chế tự chú ý thường được tách nhân tử (factorized attention) để tối ưu tính toán:

- **Chú ý không gian (Spatial attention):** Tính toán giữa các mảnh vá trong phạm vi cùng một khung hình.
- **Chú ý thời gian (Temporal attention):** Tính toán xuyên suốt các khung hình tại cùng một vị trí tọa độ không gian.
- **Chú ý 3D toàn phần (Full 3D attention):** Tốn kém gấp 16-100 lần phép tính; chỉ được dùng ở độ phân giải thấp hoặc trong nghiên cứu lý thuyết.

### Điều kiện hóa bằng văn bản (Text conditioning)

Cơ chế chú ý chéo (cross-attention) kết hợp cùng một bộ mã hóa văn bản quy mô lớn (T5-XXL cho Sora, CogVideoX-5B cũng sử dụng T5-XXL). Câu lệnh mô tả dài đóng vai trò sống còn — tập dữ liệu huấn luyện của Sora sử dụng các chú thích dày đặc do GPT tạo sinh tự động với độ dài trung bình lên tới 200 token cho mỗi clip.

### Huấn luyện (Training)

Sử dụng hàm mất mát khuếch tán chuẩn tắc (dự đoán nhiễu $\epsilon$ hoặc dự đoán vận tốc $v$) trên các tensor tiềm ẩn không-thời gian. Dữ liệu: video thu thập từ web + khoảng 100 triệu clip được lọc kỹ + chú thích văn bản tổng hợp. Tài nguyên tính toán: trên 10.000 giờ GPU cho một lượt chạy nghiên cứu nhỏ; quy mô tầm cỡ Sora đòi hỏi trên 100.000 giờ GPU H100.

## Bức tranh toàn cảnh sản xuất năm 2026 (The 2026 production landscape)

| Mô hình | Thời điểm | Thời lượng tối đa | Độ phân giải tối đa | Mở trọng số? | Điểm nổi bật |
|---|---|---|---|---|---|
| Sora (OpenAI) | 02/2024 | 60 giây | 1080p | Không | Mô hình đầu tiên thể hiện đặc tính mô phỏng thế giới ở quy mô lớn |
| Sora Turbo | 12/2024 | 20 giây | 1080p | Không | Phiên bản thương mại với tốc độ suy luận nhanh gấp 5 lần |
| Veo 2 (Google) | 12/2024 | 8 giây | 4K | Không | Chất lượng hình ảnh và mô phỏng vật lý cao nhất năm 2025 |
| Veo 3 | Q3/2025 | 15 giây | 4K | Không | Tạo âm thanh tự nhiên đồng bộ và kiểm soát góc quay xuất sắc |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10 giây | 1080p | Không | Tái hiện chuyển động cơ thể người tốt nhất trong Q1/2025 |
| Runway Gen-3 Alpha | 06/2024 | 10 giây | 768p | Không | Tích hợp bộ công cụ làm phim chuyên nghiệp hàng đầu |
| Pika 2.0 | 10/2024 | 5 giây | 1080p | Không | Khả năng duy trì nhất quán nhân vật mạnh mẽ nhất |
| CogVideoX (THUDM) | 2024 | 10 giây | 720p | Có (2B, 5B) | Mô hình video mã nguồn mở đầu tiên ở quy mô 5 tỷ tham số |
| HunyuanVideo (Tencent) | 12/2024 | 5 giây | 720p | Có (13B) | Mô hình mã nguồn mở đạt chuẩn SOTA cuối năm 2024 |
| Mochi-1 (Genmo) | 10/2024 | 5.4 giây | 480p | Có (10B) | Giấy phép mở thông thoáng và thân thiện nhất |
| WAN 2.2 (Alibaba) | 07/2025 | 5 giây | 720p | Có | Mô hình mã nguồn mở mạnh mẽ nhất giữa năm 2025 |

Khoảng cách giữa mã nguồn mở và dịch vụ đóng đang thu hẹp nhanh hơn nhiều so với lĩnh vực tạo ảnh: các bộ LoRA tinh chỉnh trên nền HunyuanVideo và WAN 2.2 đã trở thành nền tảng cho phần lớn các quy trình sản xuất mã nguồn mở vào giữa năm 2026.

````figure
video-diffusion-denoise
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` mô phỏng tư tưởng cốt lõi của DiT không-thời gian: chia nhỏ một đoạn "video" 1 chiều nhân tạo thành các mảnh vá, bổ sung vector nhúng vị trí theo từng mảnh, và khử nhiễu cho toàn bộ chuỗi thông qua cơ chế chú ý kiểu Transformer. Hoàn toàn viết bằng Python thuần (không dùng numpy). Đoạn mã chứng minh rằng tính mạch lạc thời gian sẽ tự nhiên xuất hiện ngay cả trong không gian 1 chiều khi các mảnh vá của các khung hình liền kề dùng chung bộ khử nhiễu và vector nhúng vị trí.

### Bước 1: Chia mảnh vá cho một "video" mô phỏng 1 chiều

```python
def make_video(T_frames=8, rng=None):
    # mot "video" la mot chuoi cac gia tri 1 chieu di chuyen theo quy dao muot ma
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Bước 2: Nhúng vị trí cho từng khung hình

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Bước 3: Mạng khử nhiễu quan sát đồng thời toàn bộ chuỗi

Thay vì khử nhiễu độc lập từng khung hình một, mạng nơ-ron nhỏ của chúng ta ghép nối toàn bộ giá trị của các khung hình kèm vector nhúng vị trí tương ứng, sau đó dự đoán lượng nhiễu cho tất cả các khung hình một cách đồng thời.

### Bước 4: Kiểm thử tính mạch lạc thời gian (Temporal coherence test)

Sau khi huấn luyện xong, tiến hành lấy mẫu tạo ra một video mới. Đo lường mức độ biến thiên delta giữa các khung hình liên tiếp. Nếu mô hình đã học được cấu trúc thời gian, độ lệch delta giữa các khung hình sẽ nhỏ hơn đáng kể so với việc lấy mẫu độc lập từng khung hình.

## Cạm bẫy thường gặp (Pitfalls)

- **Lấy mẫu độc lập từng khung hình gây ra hiện tượng nhấp nháy (Flicker).** Nếu bạn áp dụng mô hình khuếch tán ảnh tĩnh trên từng khung hình riêng biệt, video sinh ra sẽ bị chớp nháy liên tục vì nhiễu ở mỗi khung hình là hoàn toàn độc lập. Mô hình video giải quyết vấn đề này bằng cách liên kết các khung hình qua cơ chế chú ý hoặc chia sẻ trường nhiễu ngẫu nhiên.
- **Chú ý 3D ngây thơ dẫn đến tràn bộ nhớ (OOM).** Chú ý 3D toàn phần trên không gian tiềm ẩn của video 10 giây 1080p đòi hỏi hàng trăm tỷ phép tính. Bắt buộc phải tách nhân tử thành chú ý không gian riêng và chú ý thời gian riêng.
- **Chất lượng chú thích dữ liệu quan trọng hơn dung lượng dữ liệu.** Cải tiến mang tính đột phá của Sora so với các công trình trước đó nằm ở việc huấn luyện trên các đoạn chú thích văn bản chi tiết gấp 10 lần (được GPT-4 gán nhãn lại dày đặc). Báo cáo kỹ thuật của OpenAI nhấn mạnh rất rõ yếu tố này.
- **Điều kiện hóa khung hình đầu tiên (First-frame conditioning).** Hầu hết các hệ thống sản xuất đều hỗ trợ đưa ảnh tĩnh làm khung hình xuất phát. Đây là chế độ "ảnh sang video" (image-to-video - I2V); quá trình huấn luyện bắt buộc phải bao gồm biến thể dữ liệu này.
- **Trôi lệch quy luật vật lý (Physics drift).** Các đoạn video dài (>10 giây) thường tích lũy các sai lệch nhỏ về cấu trúc và chuyển động. Kỹ thuật sinh mẫu theo cửa sổ trượt (sliding-window) kết hợp neo giữ khung hình then chốt (keyframe anchoring) giúp hạn chế tối đa vấn đề này.

## Ứng dụng thực tế (Use It)

| Tình huống sử dụng | Lựa chọn hàng đầu năm 2026 |
|---|---|
| Tạo video từ văn bản chất lượng cao nhất, dịch vụ máy chủ | Veo 3 hoặc Sora |
| Phim điện ảnh kiểm soát chuyển động camera chính xác | Runway Gen-3 với cọ chuyển động (motion brush) |
| Duy trì tính nhất quán của nhân vật qua nhiều clip | Pika 2.0 hoặc Kling 2.1 |
| Mô hình trọng số mở, dễ dàng tinh chỉnh LoRA | WAN 2.2 + LoRA |
| Biến ảnh tĩnh thành video (Image-to-video) | WAN 2.2-I2V, Kling 2.1 I2V, hoặc Runway |
| Đồng bộ khẩu hình từ âm thanh (Lip sync) | Veo 3 (âm thanh gốc) hoặc mô hình lip-sync chuyên biệt |
| Chỉnh sửa video | Runway Act-Two, Kling Motion Brush, Flux-Kontext (cho khung hình tĩnh) |

Chi phí tính toán trên mỗi giây video ở cùng mức chất lượng đã giảm hơn 20 lần trong giai đoạn 2024-2026.

## Đóng gói sản phẩm (Ship It)

Lưu cấu hình tại `outputs/skill-video-brief.md`. Kỹ năng này nhận đầu vào là bản tóm tắt yêu cầu video (thời lượng, tỷ lệ khung hình, phong cách thị giác, kịch bản chuyển động camera, tính nhất quán của nhân vật, âm thanh) và xuất ra: mô hình + hạ tầng lưu trữ đề xuất, khung câu lệnh chuẩn hóa (thuật ngữ điện ảnh, mô tả chuyển động, đặc tả chủ thể), quy trình quản lý seed để tái lập kết quả, cùng danh mục kiểm tra chất lượng (QA checklist) trên từng khung hình.

## Bài tập thực hành (Exercises)

1. **Dễ.** Trong `code/main.py`, so sánh độ biến thiên delta giữa các khung hình liên tiếp giữa hai phương pháp: (a) lấy mẫu độc lập từng khung hình, (b) lấy mẫu đồng thời cả chuỗi. Ghi nhận giá trị trung bình và phương sai của các mức delta đó.
2. **Trung bình.** Bổ sung điều kiện hóa theo khung hình đầu tiên: cố định khung hình 0 ở một giá trị cho trước và lấy mẫu các khung hình còn lại. Đo lường cách giá trị cố định này lan truyền và ảnh hưởng đến các khung hình tiếp theo.
3. **Khó.** Sử dụng HuggingFace diffusers để chạy mô hình CogVideoX-2B trên GPU cục bộ. Đo thời gian thực thi 20 bước suy luận ở độ phân giải 720p cho một đoạn clip dài 6 giây. Lập hồ sơ hiệu năng (profile) tầng chú ý không-thời gian để xác định điểm nghẽn tính toán lớn nhất.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|---|---|---|
| Video VAE | "VAE 3 chiều" | Bộ mã hóa nén `(T, H, W, C)` thành không gian tiềm ẩn không-thời gian. |
| Patches | "Các token video" | Khối 3D có kích thước cố định cắt ra từ tensor tiềm ẩn; đầu vào của DiT. |
| Factorized attention | "Chú ý không gian + thời gian" | Thực hiện chú ý trên không gian trước, sau đó trên thời gian; bỏ qua 3D toàn phần. |
| Image-to-video (I2V) | "Làm động bức ảnh" | Mô hình nhận đầu vào là một ảnh tĩnh + câu lệnh, xuất ra video bắt đầu từ ảnh đó. |
| Keyframe conditioning | "Khung hình mỏ neo" | Cố định các khung hình chỉ định để điều hướng toàn bộ mạch diễn biến video. |
| Motion brush | "Cọ chuyển động" | Giao diện cho phép người dùng vẽ vector hướng chuyển động trực tiếp lên ảnh. |
| Re-captioning | "Tạo chú thích dày đặc" | Sử dụng LLM để mô tả chi tiết lại toàn bộ hành động trong video huấn luyện. |
| Flicker | "Hiện tượng chớp nháy" | Sự bất nhất quán giữa các khung hình kề nhau; khắc phục nhờ khử nhiễu liên kết. |

## Lưu ý trong môi trường sản xuất: Không gian tiềm ẩn video là bài toán về băng thông bộ nhớ

Một đoạn clip 10 giây 1080p ở tốc độ 24 fps bao gồm 240 khung hình × 1920 × 1080 × 3 pixel ≈ 1.5 GB dữ liệu pixel thô. Sau khi qua mạng VAE video nén 4 lần (`2 × không_gian × 2 × thời_gian`), tensor tiềm ẩn vẫn chiếm tới ~100 MB cho mỗi request. Khi đưa lượng dữ liệu này qua một mạng DiT không-thời gian qua 30 bước suy luận với kích thước batch bằng 1, bạn phải di chuyển khoảng ~3 GB dữ liệu qua bộ nhớ băng thông cao (HBM) ở mỗi bước — băng thông bộ nhớ, chứ không phải số phép tính FLOPs, mới chính là điểm nghẽn thực sự.

Ba nguyên lý tối ưu hóa trong môi trường sản xuất thực tế:

- **Phân tán mô hình song song tensor (TP) trên mạng DiT.** Các mô hình tạo video từ văn bản thường có quy mô $\ge 10$ tỷ tham số. Cấu hình tiêu chuẩn là TP=4 trải trên 4 GPU H100; hoặc PP=2 × TP=2 cho các dòng mô hình khổng lồ. Độ trễ mỗi bước suy luận giảm gần như tuyến tính theo mức độ TP cho đến khi chạm tới giới hạn truyền thông all-reduce.
- **Ghép batch theo khung hình = Ghép batch liên tục (Continuous batching).** Khi sinh video, về bản chất ta đang xử lý một batch các khung hình được liên kết với nhau bằng cơ chế chú ý. Kỹ thuật lập lịch xử lý linh hoạt (in-flight scheduling) có thể áp dụng: bắt đầu kết xuất khung hình $t+1$ ngay khi khung hình $t-1$ đang được truyền về máy khách, nếu kiến trúc mô hình hỗ trợ cơ chế sinh theo cửa sổ trượt.
- **Bộ nhớ đệm Prefill ở cấp độ đoạn video (Clip-level prefill cache).** Đối với tác vụ tạo video từ ảnh (I2V), việc điều kiện hóa khung hình đầu tiên hoàn toàn tương đồng với giai đoạn prefill câu lệnh trong LLM: chỉ cần tính toán biểu diễn này một lần duy nhất, sau đó tái sử dụng trong các lượt truyền xuôi của bộ giải mã thời gian. Về bản chất, đây chính là cơ chế KV-cache dành riêng cho video tạo sinh.

## Tài liệu đọc thêm (Further Reading)

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — báo cáo kỹ thuật nền tảng về mô hình Sora.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — bài báo công bố CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — kiến trúc HunyuanVideo của Tencent.
- [Genmo (2024). Mochi 1 Preview Model Card](https://huggingface.co/genmo/mochi-1-preview) — mô hình Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — mô hình tạo video mã nguồn mở chuẩn SOTA năm 2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — bài báo kinh điển đặt nền móng cho mô hình khuếch tán video.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — tiền thân kiến trúc của Stable Video Diffusion.
