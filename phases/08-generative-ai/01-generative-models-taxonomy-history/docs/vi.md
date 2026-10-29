# Các mô hình sinh — Phân loại & Lịch sử phát triển

> Mọi mô hình hình ảnh, mô hình văn bản, mô hình video và mô hình 3D đều nằm gọn trong một trong năm nhóm phân loại. Chọn sai nhóm, bạn sẽ phải vật lộn với toán học suốt nhiều tuần. Chọn đúng nhóm, mười hai năm tiến bộ vượt bậc của lĩnh vực này sẽ tự động sắp xếp mạch lạc trong tâm trí bạn.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (Cơ sở Machine Learning), Phase 3 (Cốt lõi Deep Learning), Phase 7 · 14 (Transformers)
**Time:** ~45 phút

## Vấn đề đặt ra

Một mô hình sinh (generative model) chỉ đảm nhận đúng một nhiệm vụ: từ các mẫu huấn luyện được rút ra từ một phân phối chưa biết $p_{\text{data}}(x)$, xuất ra các mẫu mới trông như thể chúng xuất phát từ chính phân phối đó. Khuôn mặt, câu văn, tệp MIDI, cấu trúc protein — nếu nhìn bao quát ở góc độ toán học, chúng đều là cùng một bài toán.

Khó khăn nằm ở chỗ $p_{\text{data}}$ tồn tại trong một không gian có hàng triệu chiều (một bức ảnh RGB kích thước 512x512 đã có ~786 nghìn chiều), các mẫu dữ liệu lại nằm trên một đa tạp (manifold) mỏng manh bên trong không gian đó, và bạn có thể chỉ sở hữu vỏn vẹn khoảng 10 triệu mẫu dữ liệu. Việc vét cạn (brute-forcing) hàm mật độ xác suất là điều bất khả thi. Mỗi mô hình sinh thực chất là một sự thỏa hiệp: đánh đổi một bài toán nan giải để lấy một bài toán bớt hóc búa hơn đôi chút.

Năm họ mô hình đã chứng minh được sức sống bền bỉ qua mười hai năm qua. Hiểu được mỗi họ mô hình thực hiện sự đánh đổi nào sẽ cho bạn biết lý do vì sao nó chiến thắng áp đảo trong một số tác vụ nhưng lại sụp đổ ở những tác vụ khác.

## Khái niệm cốt lõi

![Năm họ mô hình sinh — phân loại theo đối tượng được mô hình hóa](../assets/taxonomy.svg)

**1. Mật độ tường minh, tính toán được (Explicit density, tractable).** Biểu diễn $\log p(x)$ dưới dạng một tổng mà bạn thực sự có thể tính toán chính xác. Các mô hình tự hồi quy (autoregressive models: PixelCNN, WaveNet, GPT) phân tích nhân tử $p(x) = \prod_i p(x_i \mid x_{<i})$. Các mô hình luồng chuẩn hóa (normalizing flows: RealNVP, Glow) xây dựng $p(x)$ thông qua một phép biến đổi khả nghịch (invertible transform) từ một phân phối cơ sở đơn giản. Ưu điểm: hàm hợp lý (likelihood) chính xác, hàm mất mát huấn luyện rõ ràng. Nhược điểm: quá trình suy luận tự hồi quy mang tính tuần tự (rất chậm đối với chuỗi dài), còn luồng chuẩn hóa đòi hỏi các kiến trúc khả nghịch (bị gò bó nghiêm ngặt về mặt kiến trúc).

**2. Mật độ tường minh, xấp xỉ (Explicit density, approximate).** Chặn dưới $\log p(x)$ bằng cận dưới bằng chứng (Evidence Lower Bound - ELBO) và tối ưu hóa cận dưới đó. VAE (Kingma 2013) sử dụng một bộ mã hóa - giải mã (encoder-decoder) với phân phối hậu nghiệm biến phân (variational posterior). Các mô hình khuếch tán (diffusion models: DDPM, Ho 2020) huấn luyện một mạng khử nhiễu (denoiser) gián tiếp tối ưu hóa một ELBO có trọng số. Mô hình khuếch tán chính là kiến trúc xương sống thống trị trong tạo ảnh, video và 3D tính đến năm 2026.

**3. Mật độ tiềm ẩn / ngầm định (Implicit density).** Bỏ qua hoàn toàn việc tính toán mật độ; học trực tiếp một bộ sinh (generator) $G(z)$ tạo ra các mẫu và một bộ phân biệt (discriminator) $D(x)$ phân biệt giữa mẫu thật và mẫu giả. Mạng đối nghịch sinh (Generative Adversarial Networks - GANs, Goodfellow 2014). Tốc độ suy luận rất nhanh (chỉ một lượt truyền xuôi - forward pass) nhưng nổi tiếng là không ổn định trong quá trình huấn luyện. StyleGAN 1/2/3 vẫn duy trì vị thế dẫn đầu (state of the art) về độ chân thực quang học trên các miền dữ liệu cố định (khuôn mặt, phòng ngủ) ngay cả trong năm 2026.

**4. Dựa trên hàm điểm / thời gian liên tục (Score-based / continuous-time).** Học trực tiếp gradient của log-mật độ $\nabla_x \log p(x)$ (gọi là hàm điểm - score). Song & Ermon (2019) đã chứng minh rằng khớp hàm điểm (score matching) khái quát hóa mô hình khuếch tán thành một phương trình vi phân ngẫu nhiên (Stochastic Differential Equation - SDE). Khớp luồng (flow matching, Lipman 2023) là tâm điểm phát triển của giai đoạn 2024-2026: huấn luyện không cần mô phỏng (simulation-free), quỹ đạo lấy mẫu thẳng hơn, tốc độ lấy mẫu nhanh hơn gấp 4-10 lần so với DDPM. Stable Diffusion 3, Flux, AudioCraft 2 đều sử dụng khớp luồng.

**5. Tự hồi quy dựa trên token qua các mã rời rạc (Token-based autoregressive over discrete codes).** Nén dữ liệu nhiều chiều bằng VQ-VAE hoặc bộ lượng tử hóa phần dư (residual quantizer) thành một chuỗi ngắn các token rời rạc, sau đó sử dụng một Transformer để mô hình hóa chuỗi token đó. Parti, MuseNet, AudioLM, VALL-E, bộ tách token dạng mảng (patch tokenizer) của Sora đều áp dụng phương pháp này. Đây thực chất là nhóm 1 kết hợp với một bộ tách token được học (learned tokenizer).

## Lược sử phát triển

| Năm | Mô hình | Vì sao đóng vai trò quan trọng |
|-----|---------|--------------------------------|
| 2013 | VAE (Kingma) | Mô hình sinh sâu đầu tiên có hàm mất mát huấn luyện thực sự khả thi. |
| 2014 | GAN (Goodfellow) | Mật độ ngầm định, không cần hàm hợp lý — tạo ra các mẫu sắc nét đến kinh ngạc. |
| 2015 | DRAW, PixelCNN | Sinh ảnh tuần tự. |
| 2017 | Glow, RealNVP | Các luồng khả nghịch; tính toán hàm hợp lý chính xác với mạng sâu. |
| 2017 | Progressive GAN | Lần đầu tiên tạo ra hình ảnh khuôn mặt đạt độ phân giải megapixel. |
| 2019 | StyleGAN / StyleGAN2 | Khuôn mặt chân thực như ảnh chụp, đến nay vẫn rất khó bị đánh bại trong miền dữ liệu chuyên biệt đó. |
| 2020 | DDPM (Ho) | Mô hình khuếch tán trở nên khả thi trong thực tế. |
| 2021 | CLIP, DALL-E 1, VQGAN | Tạo ảnh từ văn bản (text-to-image) bước vào dòng chảy chính. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Khuếch tán tiềm ẩn (latent diffusion) + điều kiện văn bản = trở thành công nghệ đại trà. |
| 2022 | ControlNet, LoRA | Kiểm soát chi tiết các mô hình khuếch tán đã tiền huấn luyện. |
| 2023 | SDXL, Midjourney v5, Flow matching | Quy mô lớn hơn + động lực học huấn luyện (training dynamics) tốt hơn. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Khuếch tán video; khớp luồng (flow matching) giành chiến thắng. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Video đạt chuẩn sản xuất thương mại (production-grade). |
| 2026 | Consistency + Rectified Flow | Lấy mẫu chỉ trong một bước (one-step sampling) từ các mô hình nền tảng khuếch tán. |

## Quy trình sàng lọc năm câu hỏi

Khi một bài báo về mô hình sinh mới xuất hiện, hãy trả lời năm câu hỏi sau đây trước khi đọc phần phương pháp kỹ thuật:

1. **Đối tượng được mô hình hóa là gì?** Điểm ảnh (pixels), biến tiềm ẩn (latents), token rời rạc, các hàm Gaussian 3D (3D Gaussians), lưới đa giác (meshes), hay dạng sóng âm thanh (waveforms)?
2. **Hàm mật độ là tường minh hay ngầm định?** Họ có thể viết ra được công thức $\log p(x)$ hay không?
3. **Phương pháp lấy mẫu: một bước (one-shot) hay lặp (iterative)?** Lặp đồng nghĩa với suy luận chậm hơn; một bước thường là mô hình đối nghịch hoặc mô hình đã qua chưng cất (distilled).
4. **Cơ chế điều kiện hóa: không điều kiện, theo lớp, theo văn bản, theo ảnh, hay theo tư thế (pose)?** Yếu tố này quyết định hàm mất mát và cấu trúc khung kiến trúc.
5. **Phương pháp đánh giá: FID, CLIP score, IS, đánh giá của con người (human preference), hay độ chính xác của tác vụ?** Mỗi phương pháp đều có những hình thái thất bại (failure modes) đã biết (xem Bài 14).

Bạn sẽ trả lời lại năm câu hỏi này trong từng bài học của giai đoạn này. Khi kết thúc, chúng sẽ trở thành phản xạ tự nhiên.

```figure
autoencoder-bottleneck
```

## Thực hành: Xây dựng

Mã nguồn của bài học này là một mô phỏng trực quan nhẹ nhàng: khớp một hỗn hợp phân phối Gaussian 1 chiều (1-D mixture-of-Gaussians) từ các mẫu dữ liệu bằng ba cách tiếp cận đồ chơi (ước lượng mật độ nhân - kernel density, biểu đồ tần suất rời rạc - discrete histogram, và một bộ sinh "kiểu GAN" dựa trên mẫu gần nhất) để bạn có thể thấy rõ sự khác biệt giữa mật độ tường minh và mật độ ngầm định trên một bài toán hiển thị trọn vẹn trong một màn hình.

Chạy `code/main.py`. Chương trình sẽ rút 2000 mẫu từ hỗn hợp Gaussian hai đỉnh (two-mode Gaussian mixture), sau đó in ra:

```text
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Lưu ý: hai phương pháp đầu cho phép bạn đặt câu hỏi "điểm dữ liệu này có xác suất xảy ra là bao nhiêu?" Phương pháp thứ ba không thể làm được điều đó. Đây chính là sự phân biệt giữa *tường minh và ngầm định (explicit vs implicit)* sẽ chi phối xuyên suốt mọi bài học tiếp theo.

## Ứng dụng thực tế

Nên chọn họ mô hình nào cho từng tác vụ vào năm 2026?

| Tác vụ | Họ mô hình tối ưu | Lý do |
|--------|-------------------|-------|
| Khuôn mặt chân thực, miền dữ liệu hẹp | StyleGAN 2/3 | Vẫn sắc nét nhất, tốc độ suy luận nhanh nhất. |
| Tạo ảnh từ văn bản tổng quát | Khuếch tán tiềm ẩn (Latent diffusion) + khớp luồng (flow matching) | SD3, Flux.1, DALL-E 3. |
| Tạo ảnh từ văn bản tốc độ cao | Rectified flow + chưng cất (distillation) | SDXL-Turbo, SD3-Turbo, LCM. |
| Tạo video từ văn bản | Diffusion Transformer + khớp luồng | Sora, Veo 2, Kling. |
| Giọng nói + âm nhạc | Tự hồi quy dựa trên token (AudioLM, VALL-E, MusicGen) hoặc khớp luồng (AudioCraft 2) | Token rời rạc giúp mở rộng quy mô với chi phí thấp. |
| Cảnh 3D | Khớp Gaussian Splatting, prior khuếch tán | 3D-GS dùng cho tái tạo, khuếch tán dùng để tạo góc nhìn mới (novel-view). |
| Ước lượng mật độ (không lấy mẫu) | Các luồng chuẩn hóa (Flows) | Họ mô hình duy nhất tính được chính xác $\log p(x)$. |
| Mô phỏng / vật lý | Khớp luồng (Flow matching), score SDE | Quỹ đạo đường thẳng, trường vector trơn tru. |

## Đóng gói sản phẩm

Lưu lại dưới dạng `outputs/skill-model-chooser.md`.

Kỹ năng này nhận vào mô tả tác vụ và xuất ra: (1) họ mô hình nên sử dụng, (2) danh sách xếp hạng gồm ba lựa chọn mã nguồn mở và ba lựa chọn dịch vụ lưu trữ (hosted), (3) hình thái thất bại tiềm ẩn cần theo dõi, và (4) ngân sách tính toán/thời gian dự kiến.

## Bài tập rèn luyện

1. **Dễ.** Đối với mỗi sản phẩm trong năm sản phẩm sau, hãy xác định họ mô hình và kiến trúc xương sống: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Dẫn chứng cần dựa trên các báo cáo kỹ thuật công khai.
2. **Trung bình.** Bài báo bạn chuẩn bị đọc vào ngày mai tuyên bố có tốc độ lấy mẫu nhanh gấp 100 lần so với mô hình khuếch tán. Hãy viết ra ba câu hỏi nhằm kiểm tra xem liệu lợi thế tốc độ đó có còn được duy trì khi thêm điều kiện và nâng độ phân giải lên cao hay không.
3. **Khó.** Chọn một lĩnh vực bạn quan tâm (ví dụ: cấu trúc protein, CAD, phân tử, quỹ đạo chuyển động). Trả lời quy trình sàng lọc năm câu hỏi cho mô hình SOTA hiện tại trong lĩnh vực đó và phác thảo những điểm mà một mô hình ưu việt hơn sẽ thay đổi.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực chất |
|-----------|------------------------|-------------------|
| Mô hình sinh (Generative model) | "Nó tạo ra nội dung mới" | Học một bộ lấy mẫu cho $p_{\text{data}}(x)$, có thể tùy chọn cung cấp $\log p(x)$. |
| Mật độ tường minh (Explicit density) | "Bạn có thể tính toán được nó" | Mô hình cung cấp một công thức dạng đóng hoặc khả quy để tính $\log p(x)$. |
| Mật độ ngầm định (Implicit density) | "Kiểu như GAN" | Chỉ có bộ lấy mẫu — không có cách nào tính giá trị $p(x)$ cho một điểm dữ liệu đã cho. |
| ELBO | "Cận dưới bằng chứng" | Một cận dưới tính toán được của $\log p(x)$; VAE và mô hình khuếch tán tối ưu hóa giá trị này. |
| Hàm điểm (Score) | "Gradient của log-mật độ" | $\nabla_x \log p(x)$; các mô hình khuếch tán và SDE học trường vector này. |
| Giả thuyết đa tạp (Manifold hypothesis) | "Dữ liệu nằm trên một bề mặt" | Dữ liệu nhiều chiều tập trung trên một đa tạp số chiều thấp; đây là lý do các kỹ thuật giảm số chiều phát huy hiệu quả. |
| Tự hồi quy (Autoregressive) | "Dự đoán phần tiếp theo" | Phân tích phân phối đồng thời thành tích của các phân phối có điều kiện. |
| Biến tiềm ẩn (Latent) | "Mã nén" | Biểu diễn số chiều thấp mà từ đó bộ giải mã có thể tái tạo lại đầu vào. |

## Lưu ý triển khai sản xuất: năm họ mô hình, năm hình thái suy luận

Mỗi họ mô hình tương ứng với một đường cong chi phí máy chủ suy luận khác nhau. Tài liệu về suy luận trong môi trường sản xuất thường phân tách suy luận LLM thành giai đoạn nạp trước (prefill) + giải mã (decode); sự phân tách tương tự cũng được áp dụng tại đây:

- **Tự hồi quy (nhóm 1 và 5).** Độ trễ bị chi phối bởi giai đoạn giải mã tuần tự; bộ nhớ đệm KV (KV-cache), gom cụm liên tục (continuous batching), và giải mã suy đoán (speculative decoding) đều có thể áp dụng trực tiếp.
- **VAE / khuếch tán / khớp luồng (nhóm 2 và 4).** Không có bước giải mã tuần tự theo nghĩa của LLM. Chi phí = `num_steps × step_cost`, và `step_cost` là một lượt truyền xuôi qua Transformer hoặc U-Net ở độ phân giải tiềm ẩn đầy đủ. Các núm điều chỉnh trong sản xuất bao gồm số bước (DDIM / DPM-Solver / chưng cất), kích thước batch, và độ chính xác dấu phẩy động (bf16 / fp8 / int4).
- **GAN (nhóm 3).** Chỉ một lượt truyền xuôi duy nhất. Không có lịch trình (schedule), không cần KV-cache. TTFT (thời gian cho token đầu tiên) ≈ tổng độ trễ suy luận. Đây là lý do vì sao StyleGAN vẫn chiến thắng về trải nghiệm người dùng (UX) trên các miền dữ liệu hẹp.

Khi bạn thấy cụm từ "nhanh hơn mô hình khuếch tán" trong phần tóm tắt của một bài báo, hãy tự động dịch nó thành "ít bước hơn × chi phí mỗi bước không đổi" hoặc "số bước giữ nguyên × chi phí mỗi bước rẻ hơn". Mọi cách diễn giải khác đều chỉ là chiêu trò tiếp thị.

## Tài liệu đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo gốc về GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — bài báo gốc về VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — bài báo gốc về DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — mô hình khuếch tán dưới góc nhìn SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — bài báo gốc về khớp luồng (flow matching).
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
