# GAN — Bộ sinh đối đầu Bộ phân biệt (GANs — Generator vs Discriminator)

> Bước đột phá của Goodfellow vào năm 2014 là bỏ qua hoàn toàn việc tính toán hàm mật độ xác suất. Hai mạng nơ-ron: một mạng chuyên tạo hàng giả, một mạng chuyên truy bắt. Chúng đối đầu nhau cho đến khi các mẫu giả trở nên không thể phân biệt được với mẫu thật. Về lý thuyết, mô hình này dường như không thể hoạt động. Trong thực tế, nó cũng rất hay thất bại. Nhưng một khi thành công, các mẫu được tạo ra vẫn là những hình ảnh sắc nét nhất trong y văn đối với các miền dữ liệu chuyên biệt.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Lan truyền ngược - Backprop), Phase 3 · 08 (Bộ tối ưu hóa - Optimizers), Phase 8 · 02 (Bộ tự mã hóa biến phân - VAE)
**Time:** ~75 phút

## Vấn đề đặt ra

Mô hình VAE tạo ra các mẫu bị nhòe mờ vì hàm mất mát bộ giải mã MSE tối ưu Bayes cho hình ảnh *kỳ vọng trung bình* — mà giá trị trung bình của nhiều chữ số hợp lý lại là một chữ số mờ căm. Bạn cần một hàm mất mát tưởng thưởng cho *tính hợp lý (plausibility)*, chứ không phải khoảng cách sai lệch từng điểm ảnh so với một mục tiêu cố định nào đó. Không có công thức dạng đóng nào để đo lường tính hợp lý. Bạn buộc phải học nó từ dữ liệu.

Ý tưởng xuất sắc của Goodfellow: huấn luyện một bộ phân loại (classifier) $D(x)$ để phân biệt ảnh thật với ảnh giả. Đồng thời huấn luyện một bộ sinh (generator) $G(z)$ để đánh lừa $D$. Tín hiệu mất mát của $G$ chính là bất cứ tiêu chí nào mà $D$ đang cho rằng sẽ khiến một vật trông giống như thật. Tín hiệu này liên tục cập nhật khi $G$ tiến bộ, rượt đuổi một mục tiêu luôn biến động. Nếu cả hai mạng cùng hội tụ, $G$ sẽ học được phân phối dữ liệu mà không cần phải viết ra biểu thức $\log p(x)$.

Đây chính là cơ chế huấn luyện đối nghịch (adversarial training). Về mặt toán học, đó là một trò chơi đối kháng cực tiểu - cực đại (minimax game):

```math
\min_G \max_D \; \mathbb{E}_{\text{real}}[\log D(x)] + \mathbb{E}_{\text{fake}}[\log(1 - D(G(z)))]
```

Đến năm 2026, GAN không còn giữ vị trí SOTA cho tác vụ sinh tổng quát (vương miện này đã thuộc về các mô hình khuếch tán và khớp luồng). Tuy nhiên, StyleGAN 2/3 vẫn là các mô hình tạo khuôn mặt sắc nét nhất từng được công bố, các bộ phân biệt của GAN được tận dụng làm *hàm mất mát cảm nhận (perceptual losses)* trong huấn luyện mô hình khuếch tán, và quá trình huấn luyện đối nghịch chính là động lực thúc đẩy các phương pháp chưng cất 1 bước siêu tốc (SDXL-Turbo, SD3-Turbo, LCM) giúp đưa mô hình khuếch tán vào ứng dụng thời gian thực.

## Khái niệm cốt lõi

![Huấn luyện GAN: bộ sinh và bộ phân biệt trong trò chơi minimax](../assets/gan.svg)

**Bộ sinh $G(z)$.** Ánh xạ một vector nhiễu $z \sim \mathcal{N}(0, I)$ thành một mẫu dữ liệu $\hat{x}$. Mạng có cấu trúc tương tự bộ giải mã (các tầng kết nối dày đặc hoặc tích chập chuyển vị - transposed convolution).

**Bộ phân biệt $D(x)$.** Ánh xạ một mẫu dữ liệu thành một giá trị xác suất vô hướng (hoặc điểm số). Thật $\to 1$, giả $\to 0$.

**Hàm mất mát (Loss).** Hai bước cập nhật luân phiên:

- **Huấn luyện $D$:** $\text{loss}_D = -[ \log D(x) + \log(1 - D(G(z))) ]$. Hàm mất mát entropy chéo nhị phân (binary cross-entropy) với nhãn thật = 1, giả = 0.
- **Huấn luyện $G$:** $\text{loss}_G = -\log D(G(z))$. Đây là dạng *không bão hòa (non-saturating form)* mà Goodfellow đã sử dụng (công thức gốc $\log(1 - D(G(z)))$ sẽ bão hòa và triệt tiêu gradient khi $D$ có độ tin cậy quá cao).

**Vòng lặp huấn luyện.** Một bước cho $D$, một bước cho $G$. Lặp lại liên tục.

**Vì sao mô hình hoạt động.** Nếu $G$ khớp hoàn hảo với phân phối $p_{\text{data}}$, thì $D$ không thể phán đoán tốt hơn việc đoán mò ngẫu nhiên và sẽ xuất ra giá trị 0.5 ở mọi nơi; khi đó $G$ không còn nhận thêm gradient nào nữa. Hệ thống đạt trạng thái cân bằng (equilibrium).

**Vì sao mô hình sụp đổ.** Hiện tượng sụp đổ mốt (mode collapse: $G$ phát hiện một mốt cục bộ mà $D$ không phân loại được và chỉ tạo ra mốt đó mãi mãi), triệt tiêu gradient (vanishing gradient: $D$ học quá nhanh khiến $\log D$ bị bão hòa), bất ổn định khi huấn luyện (tốc độ học, kích thước batch, hoặc bất kỳ siêu tham số nào).

## Các biến thể then chốt giúp GAN hoạt động hiệu quả

| Năm | Cải tiến | Vấn đề được giải quyết |
|-----|----------|------------------------|
| 2015 | DCGAN | Tích chập/giải tích chập, chuẩn hóa theo batch (batch norm), LeakyReLU — kiến trúc ổn định đầu tiên. |
| 2017 | WGAN, WGAN-GP | Thay thế BCE bằng khoảng cách Wasserstein + phạt gradient (gradient penalty). Giải quyết triệt tiêu gradient. |
| 2017 | Chuẩn hóa phổ (Spectral normalization) | Ràng buộc Lipschitz cho bộ phân biệt. Vẫn được dùng trong các bộ phân biệt năm 2026. |
| 2018 | Progressive GAN | Huấn luyện từ độ phân giải thấp trước rồi tăng dần các tầng. Lần đầu tiên đạt kết quả megapixel. |
| 2019 | StyleGAN / StyleGAN2 | Mạng ánh xạ (mapping network) + chuẩn hóa cá thể thích ứng (adaptive instance norm - AdaIN). Dẫn đầu về độ chân thực quang học trong miền cố định. |
| 2021 | StyleGAN3 | Khử răng cưa (alias-free), đồng biến tịnh tiến (translation-equivariant) — chuẩn mực vàng cho tạo khuôn mặt năm 2026. |
| 2022 | StyleGAN-XL | Có điều kiện (conditional), nhận biết lớp (class-aware), quy mô lớn hơn. |
| 2024 | R3GAN | Tái thiết kế với kỹ thuật điều quy hóa mạnh hơn; hoạt động trơn tru ở độ phân giải 1024² mà không cần tiểu xảo phức tạp. |

```figure
gan-minimax
```

## Thực hành: Xây dựng

`code/main.py` huấn luyện một mạng GAN tối giản trên dữ liệu 1 chiều: hỗn hợp gồm hai phân phối Gaussian. Bộ sinh và bộ phân biệt đều là các mạng MLP một tầng ẩn. Chúng ta tự cài đặt lượt truyền xuôi, truyền ngược và vòng lặp minimax bằng tay. Mục tiêu là quan sát trực tiếp hai hình thái thất bại điển hình (sụp đổ mốt + triệt tiêu gradient) khi chúng diễn ra.

### Bước 1: Hàm mất mát không bão hòa (Non-saturating loss)

Hàm mất mát gốc của Goodfellow $\log(1 - D(G(z)))$ tiệm cận về 0 khi D phân loại mẫu giả của G thành giả với độ tin cậy cao. Tại thời điểm đó, gradient cung cấp cho G gần như bằng không — G hoàn toàn bất lực trong việc tự cải thiện. Dạng không bão hòa $-\log D(G(z))$ sở hữu đường tiệm cận ngược lại: nó bùng nổ rất mạnh khi D quá tự tin, từ đó cung cấp cho G một tín hiệu cập nhật dồi dào.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Bước 2: Một bước cập nhật bộ phân biệt tương ứng một bước cập nhật bộ sinh

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Luôn lấy mẫu giả mới (fresh fakes) cho G, nếu không gradient sẽ bị trễ và thiếu chính xác.

### Bước 3: Giám sát hiện tượng sụp đổ mốt

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Triệu chứng kinh điển: một trong hai mốt thực tế hoàn toàn biến mất khỏi các mẫu được tạo ra. Bộ phân biệt ngừng sửa sai mốt đó vì nó không còn xuất hiện dưới dạng mẫu giả nữa.

## Các cạm bẫy thường gặp

- **Bộ phân biệt quá mạnh.** Giảm tốc độ học của D xuống từ 2-5 lần, hoặc thêm nhiễu vào đầu vào của các tầng/cá thể. Nếu D đạt độ chính xác >95%, G sẽ lập tức "chết đứng".
- **Bộ sinh ghi nhớ một mốt duy nhất.** Thêm nhiễu vào đầu vào của D, sử dụng một tầng phân biệt theo minibatch (minibatch discrimination), hoặc chuyển sang WGAN-GP.
- **Rò rỉ số liệu thống kê qua chuẩn hóa theo batch (Batch norm).** Việc cho cả batch thật và batch giả đi qua cùng một tầng BN sẽ làm hòa lẫn các số liệu thống kê của chúng. Hãy thay thế bằng chuẩn hóa theo cá thể (instance norm) hoặc chuẩn hóa phổ (spectral norm).
- **Thao túng điểm Inception (Inception-score gaming).** FID và IS dao động rất mạnh khi số lượng mẫu đánh giá quá nhỏ. Luôn sử dụng ít nhất $\ge 10.000$ mẫu trong giai đoạn đánh giá.
- **Lấy mẫu một bước (one-shot sampling) là một ảo tưởng đối với các tác vụ có điều kiện.** Bạn vẫn cần thang đo độ tự do không có bộ phân loại (CFG scales), các thủ thuật cắt cụt (truncation tricks) và lấy mẫu lại nhiều lần để có được kết quả thực sự dùng được.

## Ứng dụng thực tế

Ngăn xếp GAN tiêu chuẩn năm 2026:

| Tình huống | Lựa chọn tối ưu |
|------------|-----------------|
| Khuôn mặt người chân thực, góc nhìn cố định | StyleGAN3 (sắc nét nhất, kích thước gọn nhất) |
| Khuôn mặt phong cách Anime / nghệ thuật | StyleGAN-XL hoặc LoRA cho Stable Diffusion |
| Dịch chuyển ảnh-sang-ảnh (Image-to-image translation) | Pix2Pix / CycleGAN (Phase 8 · 04) hoặc ControlNet (Phase 8 · 08) |
| Tạo ảnh từ văn bản siêu tốc chỉ trong 1 bước | Chưng cất đối nghịch từ mô hình khuếch tán (SDXL-Turbo, SD3-Turbo) |
| Hàm mất mát cảm nhận bên trong bộ huấn luyện khuếch tán | Một bộ phân biệt GAN nhỏ hoạt động trên các vùng cắt nhỏ (crops) của ảnh |
| Bất kỳ tác vụ đa phương thức mở nào | Không dùng GAN — hãy chuyển sang mô hình khuếch tán hoặc khớp luồng |

GAN có ưu thế vượt trội về độ sắc nét nhưng phạm vi áp dụng hẹp. Khi bài toán của bạn mở rộng — ảnh chụp tổng quát, câu lệnh văn bản tùy ý, video — hãy chuyển sang mô hình khuếch tán. Kỹ thuật đối nghịch hiện nay chủ yếu tồn tại như một thành phần bổ trợ (hàm mất mát cảm nhận, chưng cất), chứ không còn là một mô hình sinh độc lập.

## Đóng gói sản phẩm

Lưu lại dưới dạng `outputs/skill-gan-debugger.md`. Kỹ năng này nhận vào một quá trình huấn luyện GAN đang gặp sự cố (đường cong mất mát, lưới mẫu ảnh, kích thước tập dữ liệu) và xuất ra danh sách xếp hạng các nguyên nhân tiềm ẩn, các phương án khắc phục ngắn gọn, cùng quy trình huấn luyện lại.

## Bài tập rèn luyện

1. **Dễ.** Chạy `code/main.py` với các thiết lập mặc định. Sau đó đặt `D_LR = 5 * G_LR` và chạy lại. Sau bao nhiêu bước thì hàm mất mát của G sụp đổ về một hằng số?
2. **Trung bình.** Thay thế hàm mất mát BCE của Goodfellow bằng hàm mất mát WGAN: $\text{loss}_D = \mathbb{E}[D(\text{fake})] - \mathbb{E}[D(\text{real})]$, $\text{loss}_G = -\mathbb{E}[D(\text{fake})]$, và cắt cụt (clip) trọng số của D trong đoạn `[-0.01, 0.01]`. Quá trình huấn luyện có ổn định hơn không? Hãy so sánh thời gian hội tụ thực tế (wall-clock convergence).
3. **Khó.** Mở rộng ví dụ 1 chiều sang không gian dữ liệu 2 chiều (hỗn hợp gồm 8 phân phối Gaussian bố trí trên một đường tròn). Ghi nhận số lượng mốt mà bộ sinh nắm bắt được tại các bước 1.000, 5.000 và 10.000. Cài đặt thêm tầng phân biệt theo minibatch (minibatch discrimination) và đo đạc lại kết quả.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực chất |
|-----------|------------------------|-------------------|
| Bộ sinh (Generator) | "G" | Mạng biến đổi nhiễu thành mẫu dữ liệu, `G: z → x̂`. |
| Bộ phân biệt (Discriminator) | "D" | Bộ phân loại `D: x → [0, 1]`, phân định thật và giả. |
| Trò chơi Minimax | "Trò chơi đối kháng" | Phép toán $\min_G \max_D$ trên một hàm mục tiêu chung. |
| Hàm mất mát không bão hòa (Non-saturating loss) | "Cách sửa lỗi triệt tiêu gradient" | Sử dụng $-\log D(G(z))$ cho G thay vì $\log(1 - D(G(z)))$. |
| Sụp đổ mốt (Mode collapse) | "G chỉ ghi nhớ một thứ" | Bộ sinh chỉ tạo ra một số lượng rất ít mẫu khác biệt dù tập dữ liệu đầu vào rất đa dạng. |
| WGAN | "Wasserstein" | Thay thế BCE bằng khoảng cách Earth-Mover kết hợp phạt gradient; gradient trơn tru hơn. |
| Chuẩn hóa phổ (Spectral norm) | "Thủ thuật Lipschitz" | Ràng buộc chuẩn trọng số của D để chặn hệ số góc; giúp ổn định quá trình huấn luyện. |
| StyleGAN | "GAN hoạt động tốt nhất" | Mạng ánh xạ + AdaIN; tiêu chuẩn cao nhất cho sinh khuôn mặt, vẫn giữ vững đến năm 2026. |

## Lưu ý triển khai sản xuất: suy luận một bước là lợi thế bền vững của GAN

GAN không còn chiếm ưu thế về chất lượng mẫu đối với các bài toán sinh trên miền mở, nhưng chúng vẫn dẫn đầu tuyệt đối về chi phí suy luận. Theo ngôn ngữ chuẩn của tài liệu suy luận thương mại:

- **Không có giai đoạn nạp trước (prefill) hay giải mã (decode).** Toàn bộ quá trình chỉ gồm một lượt truyền xuôi $G(z)$ duy nhất. TTFT ≈ tổng độ trễ suy luận.
- **Không chịu áp lực từ bộ nhớ đệm KV (KV-cache).** Trạng thái duy nhất trong bộ nhớ là trọng số mạng. Kích thước batch chỉ bị giới hạn bởi bộ nhớ kích hoạt (activation memory), không phải bộ đệm cache.
- **Gom cụm liên tục (continuous batching) vô cùng đơn giản.** Vì mọi yêu cầu đều tiêu tốn một lượng FLOPs cố định như nhau, một batch tĩnh đạt đúng mức chiếm dụng tài nguyên mục tiêu của máy chủ thường là phương án tối ưu. Không cần đến bộ điều phối phức tạp giữa chừng.

Đây là lý do tại sao kỹ thuật chưng cất GAN (SDXL-Turbo, SD3-Turbo, ADD, LCM) trở thành phương pháp thống trị để tạo ảnh từ văn bản tốc độ cao trong năm 2026: nó nén một quy trình khuếch tán kéo dài từ 20-50 bước xuống chỉ còn 1-4 lượt truyền xuôi kiểu GAN trong khi vẫn bảo toàn được phân phối chất lượng của mô hình khuếch tán nền tảng. Hàm mất mát đối nghịch vẫn tồn tại đắc lực như một công cụ huấn luyện để biến các bộ sinh chậm chạp thành các cỗ máy tốc độ cao.

## Tài liệu đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo gốc khai sinh ra GAN.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — kiến trúc ổn định đầu tiên.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — chuẩn hóa phổ SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
