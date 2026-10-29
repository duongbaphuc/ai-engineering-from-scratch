# Khuếch tán trong không gian tiềm ẩn & Stable Diffusion (Latent Diffusion & Stable Diffusion)

> Khuếch tán trực tiếp trong không gian pixel trên hình ảnh 512×512 là một sự lãng phí tài nguyên tính toán khủng khiếp. Rombach cùng cộng sự (2022) nhận thấy rằng chúng ta không cần đến toàn bộ 786.000 chiều để tạo ra một bức ảnh — ta chỉ cần đủ số chiều để nắm bắt cấu trúc ngữ nghĩa (semantic structure), và dùng một bộ giải mã riêng biệt để tái tạo các chi tiết còn lại. Vận hành quá trình khuếch tán bên trong không gian tiềm ẩn (latent space) của một mạng VAE. Ý tưởng đột phá duy nhất đó đã khai sinh ra Stable Diffusion.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Yêu cầu tiên quyết:** Phase 8 · 02 (Mạng tự mã hóa biến phân - VAE), Phase 8 · 06 (Mô hình khuếch tán - DDPM), Phase 7 · 09 (Vision Transformer - ViT)  
**Thời lượng:** ~75 phút  

## Vấn đề cần giải quyết (The Problem)

Mô hình khuếch tán trên không gian pixel ở độ phân giải $512^2$ đồng nghĩa với việc mạng U-Net phải xử lý các tensor có kích thước `[B, 3, 512, 512]`. Mỗi bước lấy mẫu tiêu tốn khoảng ~100 GFLOPS đối với một mạng U-Net 500M tham số. Năm mươi bước suy luận tiêu hao tới 5 TFLOPS cho duy nhất một bức ảnh. Huấn luyện trên hàng tỷ bức ảnh sẽ đẩy hóa đơn tính toán lên mức không tưởng.

Phần lớn số phép tính FLOPs đó bị lãng phí vào việc đẩy các chi tiết không quan trọng về mặt thị giác qua mạng — những kết cấu tần số cao mà một mạng VAE nén mất mát (lossy VAE) có thể dễ dàng lược bỏ. Ý tưởng của Rombach: huấn luyện một mạng VAE độc lập trước (*giai đoạn 1*), đóng băng toàn bộ trọng số của nó, và vận hành quá trình khuếch tán hoàn toàn bên trong không gian tiềm ẩn 4 kênh kích thước 64×64 (*giai đoạn 2*). Vẫn là mạng U-Net đó, nhưng số lượng điểm ảnh giảm xuống còn 1/16. Giảm lượng tính toán FLOPs khoảng ~64 lần trong khi chất lượng hình ảnh hoàn toàn tương đương.

Đây chính là công thức thành công của Stable Diffusion. SD 1.x / 2.x sử dụng mạng U-Net 860M tham số trên không gian tiềm ẩn `64×64×4`, SDXL nâng cấp lên U-Net 2.6B tham số trên không gian `128×128×4`, SD3 thay thế U-Net bằng kiến trúc Transformer khuếch tán (Diffusion Transformer - DiT) kết hợp phối khớp dòng (flow matching). Flux.1-dev (Black Forest Labs, 2024) trang bị kiến trúc MMDiT lên tới 12 tỷ tham số. Tất cả đều vận hành trên cùng một nền tảng kiến trúc hai giai đoạn này.

## Khái niệm cốt lõi (The Concept)

![Khuếch tán tiềm ẩn: nén VAE + khuếch tán trong không gian tiềm ẩn](../assets/latent-diffusion.svg)

**Hai giai đoạn được huấn luyện tách biệt.**

1. **Giai đoạn 1 — VAE.** Bộ mã hóa $E(x) \to z$, bộ giải mã $D(z) \to x$. Mục tiêu nén: giảm mẫu (downsample) 8 lần trên mỗi chiều không gian + điều chỉnh số kênh sao cho tổng kích thước tiềm ẩn chỉ bằng khoảng 1/16 số lượng pixel gốc. Hàm mất mát = tái tạo (L1 + mất mát tri giác LPIPS) + mất mát KL (với trọng số nhỏ để không ép $z$ quá mức về phân phối Gaussian chuẩn tắc, vì chúng ta không cần lấy mẫu ngẫu nhiên trực tiếp từ $z$). VAE thường được kết hợp huấn luyện cùng mất mát đối nghịch (adversarial loss) để ảnh giải mã giữ được độ sắc nét.
2. **Giai đoạn 2 — Khuếch tán trên không gian $z$.** Coi biểu diễn tiềm ẩn $z = E(x_{\text{real}})$ như dữ liệu đầu vào. Huấn luyện mạng U-Net (hoặc DiT) để khử nhiễu cho $z_t$. Khi suy luận: lấy mẫu $z_0$ thông qua quá trình khuếch tán ngược, sau đó tái tạo lại ảnh bằng $x = D(z_0)$.

**Điều kiện hóa bằng văn bản (Text conditioning).** Bổ sung thêm hai thành phần: Bộ mã hóa văn bản đóng băng trọng số (frozen text encoder: CLIP-L cho SD 1.x, CLIP-L + OpenCLIP-G cho SD 2/XL, T5-XXL cho SD3 và Flux). Cơ chế tiêm thông tin bằng chú ý chéo (cross-attention): mỗi khối U-Net nhận đầu vào `[Q = đặc trưng ảnh, K = V = các token văn bản]` và hòa trộn chúng lại với nhau. Các token này là kênh duy nhất giúp văn bản định hướng nội dung bức ảnh.

**Hàm mất mát hoàn toàn tương đồng với Bài 06.** Vẫn là hàm MSE dự đoán nhiễu của DDPM hoặc phối khớp dòng (flow matching). Điểm thay đổi duy nhất là bạn hoán đổi miền dữ liệu từ không gian pixel sang không gian tiềm ẩn.

## Các biến thể kiến trúc (Architecture variants)

| Mô hình | Năm | Khung xương (Backbone) | Kích thước tiềm ẩn (Latent) | Bộ mã hóa văn bản | Số tham số |
|---|---|---|---|---|---|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 token) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + bộ tinh chỉnh (refiner) | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Chưng cất (Distilled) | 128×128×4 | giữ nguyên | Lấy mẫu 1-4 bước |
| SD3 | 2024 | MMDiT (DiT đa phương thức) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT chưng cất | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 bước |

Xu hướng phát triển rõ rệt: thay thế mạng U-Net bằng DiT (Transformer xử lý trên các mảnh vá tiềm ẩn - latent patches), mở rộng quy mô bộ mã hóa văn bản (T5 vượt trội hơn hẳn CLIP về khả năng hiểu prompt phức tạp), tăng số kênh tiềm ẩn (từ 4 → 16 kênh giúp mở rộng biên độ lưu trữ chi tiết).

````figure
noise-schedule
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` tích hợp một mô hình "VAE" đồ chơi 1 chiều (một bộ mã hóa thu nhỏ cố định tỷ lệ 0.5 và bộ giải mã nghịch đảo chính xác, nhằm phục vụ mục đích minh họa; một VAE thực tế sẽ là mạng tích chập convnet) xếp chồng lên trên DDPM từ Bài 06, đồng thời bổ sung khả năng điều kiện hóa theo lớp với cơ chế hướng dẫn không cần bộ phân loại (classifier-free guidance - CFG). Đoạn mã này chứng minh rằng cùng một hàm mất mát khuếch tán có thể hoạt động hoàn hảo dù bạn áp dụng trên giá trị thô 1 chiều hay trên các giá trị đã qua mã hóa. Đó chính là trực giác cốt lõi.

### Bước 1: Bộ mã hóa và giải mã

```python
def encode(x):    return x * 0.5          # "nén" mô phỏng sang dải tỷ lệ nhỏ hơn
def decode(z):    return z * 2.0
```

Một mạng VAE thực tế sở hữu các trọng số học được. Về mặt sư phạm, phép ánh xạ tuyến tính này là đủ để minh chứng rằng mô hình khuếch tán chỉ thao tác trên không gian $z$ mà không cần bận tâm đến không gian dữ liệu gốc.

### Bước 2: Khuếch tán trong không gian `z`

Sử dụng cùng thuật toán DDPM như Bài 06. Dữ liệu mà mạng quan sát thấy là $z = E(x)$. Sau khi lấy mẫu được $z_0$, giải mã ảnh bằng $D(z_0)$.

### Bước 3: Hướng dẫn không cần bộ phân loại (Classifier-free guidance)

Trong quá trình huấn luyện, loại bỏ nhãn lớp ngẫu nhiên 10% số lần (thay thế bằng một token rỗng - null token). Khi suy luận, tính toán cả hai đầu ra $\epsilon_{\text{cond}}$ và $\epsilon_{\text{uncond}}$, sau đó kết hợp:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

Trong đó $w = 0$ nghĩa là không hướng dẫn (tối đa hóa tính đa dạng), $w = 3$ là mức mặc định cân bằng, $w \ge 7$ sẽ tạo ra hình ảnh bão hòa màu và quá sắc nét.

### Bước 4: Điều kiện hóa văn bản (Khái niệm triển khai)

Thay thế nhãn lớp bằng đầu ra của một bộ mã hóa văn bản đóng băng. Truyền vector nhúng văn bản vào U-Net thông qua cơ chế chú ý chéo (cross-attention):

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Đây là điểm khác biệt cốt lõi duy nhất giữa một mô hình khuếch tán có điều kiện theo lớp và mô hình Stable Diffusion.

## Cạm bẫy thường gặp (Pitfalls)

- **Lệch hệ số co giãn VAE (VAE-scale mismatch).** Bộ mã hóa VAE của SD 1.x yêu cầu một hằng số co giãn (`scaling_factor ≈ 0.18215`) nhân vào sau khi mã hóa. Quên áp dụng hằng số này sẽ khiến U-Net phải huấn luyện trên các vector tiềm ẩn có phương sai lệch lạc nghiêm trọng. Mọi checkpoint đều đi kèm một hệ số chuẩn hóa riêng.
- **Mất mát chất lượng do bộ mã hóa văn bản.** SD3 yêu cầu T5-XXL với độ dài chuỗi $\ge 128$ token; việc chỉ quay về dùng CLIP sẽ gây mất mát ngữ nghĩa đáng kể. Luôn kiểm tra cờ cấu hình `use_t5=True`, nếu không độ trung thực theo câu lệnh (prompt fidelity) sẽ sụt giảm nghiêm trọng.
- **Trộn lẫn các không gian tiềm ẩn.** SDXL, SD3 và Flux đều sử dụng các bộ VAE hoàn toàn khác nhau. Một adapter LoRA được huấn luyện trên không gian tiềm ẩn của SDXL sẽ không thể hoạt động trên SD3. Thư viện Hugging Face diffusers từ phiên bản 0.30+ sẽ từ chối nạp các checkpoint không tương thích.
- **Chỉ số CFG quá cao.** Đặt $w > 10$ sẽ tạo ra hình ảnh bị gắt màu, cháy sáng và học vẹt câu lệnh làm mất tính tự nhiên. Vùng giá trị tối ưu thường nằm trong khoảng $w = 3\text{-}7$.
- **Rò rỉ câu lệnh phủ định (Negative prompts leaking).** Một câu lệnh phủ định để trống sẽ biến thành token rỗng (null token); một câu lệnh phủ định có nội dung sẽ trở thành vector định hướng cho $\epsilon_{\text{uncond}}$. Hai trường hợp này hoàn toàn khác nhau; một số pipeline suy luận mặc định âm thầm đưa về token rỗng nếu không xử lý cẩn thận.

## Ứng dụng thực tế (Use It)

Các ngăn xếp công nghệ tiêu biểu trong môi trường sản xuất năm 2026:

| Mục tiêu | Khung xương khuyến nghị |
|---|---|
| Miền chuyên biệt, dữ liệu cặp (paired data), huấn luyện từ đầu | Tinh chỉnh SDXL (LoRA / full) — triển khai nhanh nhất |
| Tạo ảnh từ văn bản miền mở, mã nguồn mở | Flux.1-dev (12B, giấy phép phi thương mại/Apache) hoặc SD3.5-Large |
| Tốc độ suy luận nhanh nhất, mã nguồn mở | Flux.1-schnell (1-4 bước, Apache) hoặc SDXL-Lightning |
| Khả năng tuân thủ câu lệnh tốt nhất, dịch vụ máy chủ đám mây | GPT-Image / DALL-E 3, Midjourney v7, Imagen 4 |
| Quy trình chỉnh sửa ảnh (Edit workflows) | Flux.1-Kontext (Tháng 12/2024) — hỗ trợ tự nhiên cả ảnh + văn bản |
| Nghiên cứu, đường cơ sở đối chuẩn (baseline) | SD 1.5 — kiến trúc cổ điển nhưng được nghiên cứu sâu sắc nhất |

## Đóng gói sản phẩm (Ship It)

Lưu cấu hình tại `outputs/skill-sd-prompter.md`. Kỹ năng này nhận đầu vào là câu lệnh văn bản + phong cách mục tiêu và xuất ra: mô hình + checkpoint, hệ số CFG, bộ lấy mẫu (sampler), câu lệnh phủ định, độ phân giải, cấu hình kết hợp ControlNet/IP-Adapter tùy chọn, và danh mục kiểm tra chất lượng (QA checklist) cho từng bước.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py` với các mức hệ số hướng dẫn $w \in \lbrace 0, 1, 3, 7, 15 \rbrace$. Ghi nhận giá trị trung bình của các mẫu theo từng lớp. Tại mức $w$ nào thì giá trị trung bình của các lớp bị phân kỳ vượt quá phân phối dữ liệu thực tế?
2. **Trung bình.** Thay thế bộ mã hóa tuyến tính đồ chơi bằng cặp mã hóa/giải mã MLP kích hoạt tanh có hàm mất mát tái tạo. Huấn luyện lại mô hình khuếch tán trên không gian tiềm ẩn mới này. Chất lượng mẫu tạo ra có thay đổi không?
3. **Khó.** Thiết lập một quy trình suy luận Stable Diffusion thực tế bằng thư viện diffusers: nạp `sdxl-base`, chạy 30 bước lấy mẫu Euler với CFG=7, đo đạc thời gian thực thi. Sau đó chuyển sang mô hình `sdxl-turbo` với 4 bước lấy mẫu và CFG=0. Cùng một chủ thể nhưng chất lượng khác biệt — hãy phân tích chi tiết điều gì đã thay đổi và lý do tại sao.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|---|---|---|
| First stage | "Mạng VAE" | Cặp bộ mã hóa/giải mã đã huấn luyện; nén không gian ảnh $512^2$ xuống $64^2$. |
| Second stage | "Mạng U-Net/DiT" | Mô hình khuếch tán vận hành trên không gian tiềm ẩn. |
| CFG | "Hệ số hướng dẫn" | $(1+w) \cdot \epsilon_{\text{cond}} - w \cdot \epsilon_{\text{uncond}}$; điều chỉnh cường độ bám sát câu lệnh. |
| Null token | "Embedding câu lệnh rỗng" | Vector nhúng không điều kiện dùng để tính toán $\epsilon_{\text{uncond}}$. |
| Cross-attention | "Cơ chế nạp văn bản" | Mỗi khối xử lý chú ý tới các token văn bản đóng vai trò là Key và Value. |
| DiT | "Diffusion Transformer" | Thay thế U-Net bằng Transformer trên các mảnh vá tiềm ẩn; mở rộng quy mô tốt hơn. |
| MMDiT | "DiT đa phương thức" | Kiến trúc của SD3: các luồng văn bản và hình ảnh xử lý qua cơ chế chú ý chung. |
| VAE scaling factor | "Hằng số ma thuật" | Chia các giá trị tiềm ẩn cho ~5.4 để khuếch tán vận hành trên không gian phương sai đơn vị. |

## Lưu ý trong môi trường sản xuất: Chạy Flux-12B trên GPU cá nhân 8GB VRAM

Bản tham chiếu triển khai Flux là công thức mẫu mực cho câu hỏi: "Tôi chỉ có một GPU phổ thông, liệu có thể chạy được mô hình này không?". Bí quyết nằm ở ba đòn bẩy tối ưu hóa tiêu chuẩn thường được áp dụng cho DiT khuếch tán:

1. **Nạp tuần tự từng giai đoạn (Staggered loading).** Flux bao gồm bốn mạng không bao giờ cần phải đồng thời cùng nằm trong VRAM: bộ mã hóa văn bản T5-XXL (~10 GB ở định dạng fp32), CLIP-L (nhẹ), khối MMDiT 12B tham số, và mạng VAE. Hãy mã hóa câu lệnh trước, *giải phóng bộ nhớ* các bộ mã hóa văn bản, nạp DiT vào VRAM để khử nhiễu, *giải phóng* DiT, nạp VAE vào để giải mã ảnh. Các GPU tiêu dùng 8GB chỉ đủ bộ nhớ chứa từng giai đoạn đơn lẻ một lúc.
2. **Lượng tử hóa 4-bit qua bitsandbytes.** Sử dụng `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` cho cả bộ mã hóa T5 và mạng DiT. Cắt giảm bộ nhớ tới 8 lần, trong khi độ suy giảm chất lượng đối với tác vụ tạo ảnh từ văn bản hầu như không thể phân biệt bằng mắt thường.
3. **Đẩy sang bộ nhớ RAM của CPU (CPU offload).** Kích hoạt `pipe.enable_model_cpu_offload()` để tự động luân chuyển các mô-đun giữa CPU và GPU theo tiến trình của từng lượt truyền xuôi (forward pass). Điều này làm tăng độ trễ thêm 10-20% nhưng giúp toàn bộ pipeline chạy mượt mà không bị tràn bộ nhớ (OOM).

Bảng tính toán dung lượng bộ nhớ: `10 GB T5 / 8 = 1.25 GB` sau lượng tử hóa, `12 tỷ tham số × 0.5 byte = ~6 GB` cho DiT lượng tử hóa, cộng thêm phần bộ nhớ đệm cho activation. Đây là trường hợp cực hạn của suy luận trên 1 GPU duy nhất (TP=1) — không phân tán mô hình song song mà tối đa hóa lượng tử hóa. Trong môi trường sản xuất lớn, bạn sẽ chạy song song tensor (TP=2 hoặc TP=4) trên các cụm H100; nhưng đối với máy tính cá nhân của lập trình viên, đây là công thức chuẩn mực.

## Tài liệu đọc thêm (Further Reading)

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — bài báo gốc về Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — kiến trúc SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — bài báo nền tảng về DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — kiến trúc SD3 và MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — cơ chế CFG.
- [Black Forest Labs (2024). Flux.1 Announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — giới thiệu dòng mô hình Flux.1.
- [Tài liệu Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) — triển khai tham chiếu chuẩn cho tất cả các mô hình trên.
