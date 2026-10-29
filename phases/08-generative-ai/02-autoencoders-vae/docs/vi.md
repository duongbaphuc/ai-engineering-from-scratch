# Bộ tự mã hóa & Bộ tự mã hóa biến phân (Autoencoders & VAE)

> Một bộ tự mã hóa thông thường chỉ nén rồi tái tạo dữ liệu. Nó ghi nhớ máy móc. Nó không có khả năng sinh. Chỉ cần thêm một mẹo duy nhất — ép không gian mã biểu diễn theo phân phối Gaussian — bạn sẽ thu được một bộ lấy mẫu. Mẹo duy nhất đó, thủ thuật tái tham số hóa $z = \mu + \sigma \cdot \epsilon$, chính là lý do vì sao mọi mô hình tạo ảnh khuếch tán tiềm ẩn và khớp luồng mà bạn sử dụng trong năm 2026 đều đặt một VAE ở ngay tầng đầu vào.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Lan truyền ngược - Backprop), Phase 3 · 07 (Mạng tích chập - CNNs), Phase 8 · 01 (Phân loại mô hình sinh - Taxonomy)
**Time:** ~75 phút

## Vấn đề đặt ra

Hãy nén một chữ số MNIST 784 điểm ảnh thành một mã gồm 16 con số, sau đó tái tạo lại. Một bộ tự mã hóa thông thường (plain autoencoder) sẽ đạt sai số bình phương trung bình (MSE) tái tạo rất tốt, nhưng không gian mã (code space) bên trong lại là một mớ hỗn độn rời rạc và gồ ghề. Chọn ngẫu nhiên một điểm trong không gian mã đó, giải mã nó ra, bạn sẽ chỉ nhận được nhiễu vô nghĩa. Nó không sở hữu một bộ lấy mẫu (sampler). Thực chất nó chỉ là một mô hình nén dữ liệu được khoác chiếc áo hào nhoáng.

Điều bạn thực sự mong muốn là: (a) không gian mã phải là một phân phối trơn tru, chuẩn mực mà bạn có thể dễ dàng lấy mẫu — chẳng hạn như phân phối Gaussian đẳng hướng (isotropic Gaussian) $\mathcal{N}(0, I)$, (b) giải mã bất kỳ mẫu nào rút ra từ phân phối đó đều tạo ra một chữ số hợp lý, và (c) bộ mã hóa cùng bộ giải mã vẫn duy trì năng lực nén dữ liệu tốt. Ba mục tiêu, một kiến trúc thống nhất, một hàm mất mát duy nhất.

Mô hình VAE năm 2013 của Kingma giải quyết trọn vẹn điều này bằng cách huấn luyện bộ mã hóa xuất ra một *phân phối xác suất* $q(z \mid x) = \mathcal{N}(\mu(x), \sigma(x)^2)$, kéo phân phối đó về gần phân phối tiên nghiệm (prior) $\mathcal{N}(0, I)$ thông qua một số hạng phạt phân kỳ KL (Kullback-Leibler divergence penalty), và sau đó lấy mẫu $z$ từ $q(z \mid x)$ trước khi đưa vào giải mã. Đến giai đoạn suy luận, ta bỏ bộ mã hóa đi, chỉ cần lấy mẫu $z \sim \mathcal{N}(0, I)$ và đưa qua bộ giải mã. Chính số hạng phạt KL là yếu tố bắt buộc không gian mã phải có cấu trúc chặt chẽ.

Vào năm 2026, VAE hiếm khi được triển khai độc lập — chất lượng ảnh thô của nó đã bị các mô hình khuếch tán vượt mặt — nhưng chúng lại là bộ mã hóa tiêu chuẩn được lựa chọn cho mọi mô hình khuếch tán tiềm ẩn (latent-diffusion models: SD 1/2/XL/3, Flux, AudioCraft). Nắm vững VAE đồng nghĩa với việc bạn làm chủ tầng đầu tiên vô hình nhưng then chốt trong mọi quy trình xử lý hình ảnh hiện nay.

## Khái niệm cốt lõi

![Bộ tự mã hóa (Autoencoder) so với VAE: thủ thuật tái tham số hóa](../assets/vae.svg)

**Bộ tự mã hóa (Autoencoder).** $z = \text{encoder}(x)$, $\hat{x} = \text{decoder}(z)$, hàm mất mát = $\lVert x - \hat{x} \rVert^2$. Không gian mã không có cấu trúc toán học định hướng.

**Bộ mã hóa VAE (VAE encoder).** Xuất ra hai vector: $\mu(x)$ và $\log \sigma^2(x)$. Hai vector này xác định phân phối $q(z \mid x) = \mathcal{N}(\mu, \mathop{\mathrm{diag}}(\sigma^2))$.

**Thủ thuật tái tham số hóa (Reparameterization trick).** Thao tác lấy mẫu ngẫu nhiên từ $q(z \mid x)$ là một hàm không khả vi (non-differentiable). Hãy viết lại mẫu đó dưới dạng $z = \mu + \sigma \cdot \epsilon$ với $\epsilon \sim \mathcal{N}(0, I)$. Lúc này, $z$ trở thành một hàm tiền định (deterministic function) của $(\mu, \sigma)$ cộng với một biến nhiễu độc lập không chứa tham số — gradient có thể truyền ngược thông suốt qua $\mu$ và $\sigma$.

**Hàm mất mát (Loss).** Cận dưới bằng chứng (Evidence Lower Bound - ELBO), gồm hai số hạng:

```math
\begin{aligned}
\text{loss} &= \text{reconstruction} + \beta \cdot \mathrm{KL}[q(z \mid x) \,\Vert\, \mathcal{N}(0, I)] \\
&= \lVert x - \hat{x} \rVert^2 + \beta \cdot \sum_i \frac{\sigma_i^2 + \mu_i^2 - \log \sigma_i^2 - 1}{2}
\end{aligned}
```

Số hạng tái tạo (reconstruction) thúc đẩy $\hat{x}$ tiến sát $x$. Số hạng phân kỳ KL thúc đẩy $q(z \mid x)$ tiệm cận phân phối tiên nghiệm. Giữa chúng có sự đánh đổi: $\beta$ nhỏ ($<1$) cho mẫu sắc nét hơn nhưng không gian mã kém chuẩn Gaussian; $\beta$ lớn ($>1$) cho không gian mã chuẩn mực hơn nhưng mẫu tạo ra mờ hơn. $\beta$-VAE (Higgins 2017) đã biến hệ số điều chỉnh này trở nên nổi tiếng và mở ra làn sóng nghiên cứu về học biểu diễn gỡ rối (disentangled representation learning).

**Lấy mẫu (Sampling).** Trong giai đoạn suy luận: rút mẫu $z \sim \mathcal{N}(0, I)$, truyền xuôi qua bộ giải mã. Chỉ một lượt truyền xuôi duy nhất — không cần quá trình lấy mẫu lặp nhiều bước như mô hình khuếch tán.

```figure
vae-latent-grid
```

## Thực hành: Xây dựng

`code/main.py` cài đặt một mạng VAE tối giản mà không cần dùng đến numpy hay torch. Đầu vào là dữ liệu tổng hợp 8 chiều được rút ra từ hỗn hợp Gaussian 2 thành phần trong không gian 8 chiều. Bộ mã hóa và bộ giải mã là các mạng MLP một tầng ẩn. Chúng ta tự cài đặt hàm kích hoạt tanh, lượt truyền xuôi, hàm mất mát và lượt truyền ngược viết tay hoàn toàn. Mục tiêu không phải là mã nguồn thương mại, mà là sự minh bạch về mặt sư phạm.

### Bước 1: Lượt truyền xuôi của bộ mã hóa

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

Sử dụng $\log \sigma^2$ thay vì $\sigma$ để đầu ra của mạng không bị ràng buộc miền giá trị (dùng hàm softplus cho $\sigma$ là một cái bẫy — gradient sẽ bị triệt tiêu khi $\sigma \approx 0$).

### Bước 2: Tái tham số hóa và giải mã

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Bước 3: Tính toán ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

Công thức dạng đóng (closed-form) chính xác của KL được thiết lập do cả hai phân phối đều là Gaussian. Đừng bao giờ tính tích phân bằng phương pháp số. Đến năm 2026, vẫn có những dự án triển khai ước lượng KL bằng Monte Carlo — điều này làm chậm quá trình tính toán gấp 3 lần mà không mang lại bất kỳ lợi ích nào.

### Bước 4: Sinh mẫu mới

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Đó chính là toàn bộ mô hình sinh. Vỏn vẹn trong năm dòng mã.

## Các cạm bẫy thường gặp

- **Sụp đổ hậu nghiệm (Posterior collapse).** Số hạng KL ép $q(z \mid x) \to \mathcal{N}(0, I)$ quá mạnh đến mức $z$ không còn lưu giữ thông tin gì về $x$. Cách khắc phục: ủ lịch trình $\beta$ ($\beta$-annealing: bắt đầu từ $\beta=0$ rồi tăng dần lên 1), kỹ thuật "free bits", hoặc bỏ qua phạt KL trên các chiều tiềm ẩn chưa kích hoạt.
- **Mẫu bị mờ (Blurry samples).** Hàm hợp lý Gaussian của bộ giải mã ngầm định hàm mất mát tái tạo MSE, vốn tối ưu Bayes cho chuẩn $L_2$ (kỳ vọng toán học) — mà giá trị trung bình của một tập hợp các chữ số hợp lý lại là một chữ số nhòe mờ. Cách khắc phục: dùng bộ giải mã rời rạc (discrete decoder: VQ-VAE, NVAE), hoặc chỉ dùng VAE làm bộ mã hóa rồi xếp chồng mô hình khuếch tán lên không gian tiềm ẩn (đây chính là cách Stable Diffusion hoạt động).
- **$\beta$ quá lớn từ giai đoạn quá sớm.** Dẫn tới sụp đổ hậu nghiệm. Nên bắt đầu ở mức $\beta \approx 0.01$ và tăng dần.
- **Số chiều tiềm ẩn quá nhỏ.** 16 chiều là đủ cho MNIST, 256 chiều cho ImageNet 256², 2048 chiều cho ImageNet 1024². VAE của Stable Diffusion nén từ $512 \times 512 \times 3 \to 64 \times 64 \times 4$ (giảm mẫu 8 lần mỗi cạnh, giảm 64 lần về diện tích không gian; số kênh tăng từ 3 lên 4, tức giảm tổng cộng khoảng 48 lần số lượng giá trị cần xử lý).

## Ứng dụng thực tế

Ngăn xếp VAE tiêu chuẩn năm 2026:

| Tình huống | Lựa chọn tối ưu |
|------------|-----------------|
| Bộ mã hóa không gian tiềm ẩn ảnh cho mô hình khuếch tán | VAE của Stable Diffusion (`sd-vae-ft-ema`) hoặc VAE của Flux |
| Bộ mã hóa không gian tiềm ẩn âm thanh | Encodec (Meta), SoundStream, hoặc DAC (Descript) |
| Không gian tiềm ẩn video | Các mảng không-thời gian (spatiotemporal patches) của Sora, Latte VAE, WAN VAE |
| Học biểu diễn gỡ rối (Disentangled representation learning) | $\beta$-VAE, FactorVAE, TCVAE |
| Biến tiềm ẩn rời rạc (phục vụ mô hình hóa transformer) | VQ-VAE, RVQ (ResidualVQ) |
| Biến tiềm ẩn liên tục cho bài toán sinh | VAE thông thường, sau đó điều kiện hóa một mô hình luồng/khuếch tán trong không gian tiềm ẩn đó |

Mô hình khuếch tán tiềm ẩn thực chất là một VAE với mô hình khuếch tán nằm kẹp giữa bộ mã hóa và bộ giải mã. VAE đảm nhiệm việc nén thô, còn mô hình khuếch tán gánh vác phần công việc nặng nhọc nhất. Mô hình tương tự cũng được áp dụng cho video (VAE + DiT khuếch tán video) và âm thanh (Encodec + Transformer MusicGen).

## Đóng gói sản phẩm

Lưu lại dưới dạng `outputs/skill-vae-trainer.md`.

Kỹ năng này nhận vào: hồ sơ tập dữ liệu + số chiều tiềm ẩn mục tiêu + mục đích sử dụng hạ nguồn (tái tạo, lấy mẫu, hay làm đầu vào khuếch tán tiềm ẩn) và xuất ra: lựa chọn kiến trúc (plain/$\beta$/VQ/RVQ), lịch trình $\beta$, số chiều tiềm ẩn, hàm hợp lý của bộ giải mã (Gaussian so với phân loại - categorical), và kế hoạch đánh giá (MSE tái tạo, KL trên từng chiều, khoảng cách Fréchet giữa $q(z \mid x)$ và $\mathcal{N}(0, I)$).

## Bài tập rèn luyện

1. **Dễ.** Thay đổi $\beta$ trong `code/main.py` thành các giá trị `0.01`, `0.1`, `1.0`, `5.0`. Ghi lại giá trị MSE tái tạo và KL cuối cùng. Giá trị $\beta$ nào tối ưu Pareto nhất cho tập dữ liệu tổng hợp của bạn?
2. **Trung bình.** Thay thế hàm hợp lý Gaussian của bộ giải mã bằng hàm hợp lý Bernoulli (hàm mất mát cross-entropy). So sánh chất lượng mẫu trên phiên bản nhị phân hóa của cùng tập dữ liệu tổng hợp đó.
3. **Khó.** Mở rộng `code/main.py` thành một mô hình mini VQ-VAE: thay thế $z$ liên tục bằng phép tra cứu láng giềng gần nhất trong một sổ mã (codebook) gồm $K=32$ phần tử. So sánh MSE tái tạo và báo cáo xem có bao nhiêu phần tử sổ mã thực sự được sử dụng (hiện tượng sụp đổ sổ mã - codebook collapse là một vấn đề thực tế).

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực chất |
|-----------|------------------------|-------------------|
| Bộ tự mã hóa (Autoencoder) | Mạng mã hóa - giải mã | $x \to z \to \hat{x}$, tối ưu hóa hàm mất mát MSE. Không có năng lực sinh. |
| VAE | AE có thêm bộ lấy mẫu | Bộ mã hóa xuất ra một phân phối, số hạng phạt KL định hình không gian mã. |
| ELBO | Cận dưới bằng chứng | $\log p(x) \ge \text{recon} - \mathrm{KL}[q(z \mid x) \Vert p(z)]$; đạt đẳng thức chặt khi $q = p(z \mid x)$. |
| Tái tham số hóa (Reparameterization) | $z = \mu + \sigma \cdot \epsilon$ | Viết lại nút ngẫu nhiên thành hàm tiền định kết hợp với nhiễu thuần túy. Cho phép lan truyền ngược qua bước lấy mẫu. |
| Phân phối tiên nghiệm (Prior) | $p(z)$ | Phân phối mục tiêu cho biến tiềm ẩn, thông thường là $\mathcal{N}(0, I)$. |
| Sụp đổ hậu nghiệm (Posterior collapse) | "Số hạng KL thắng thế" | Bộ mã hóa phớt lờ $x$ và chỉ xuất ra phân phối tiên nghiệm; bộ giải mã buộc phải tự huyễn hoặc dữ liệu. |
| $\beta$-VAE | Trọng số KL có thể điều chỉnh | $\text{loss} = \text{recon} + \beta \cdot \mathrm{KL}$. $\beta$ càng cao = biểu diễn càng được gỡ rối nhưng mẫu càng mờ. |
| VQ-VAE | Biến tiềm ẩn rời rạc | Thay thế $z$ liên tục bằng vector sổ mã gần nhất; cho phép áp dụng mô hình transformer. |

## Lưu ý triển khai sản xuất: VAE là đường dẫn nóng nhất trong máy chủ khuếch tán

Trong quy trình Stable Diffusion / Flux / SD3, VAE được gọi hai lần cho mỗi yêu cầu — một lần để mã hóa (nếu thực hiện img2img / inpainting) và một lần để giải mã. Ở độ phân giải 1024², lượt giải mã thường là đỉnh tiêu thụ bộ nhớ kích hoạt (activation-memory peak) lớn nhất trong toàn bộ quy trình vì nó phải nâng mẫu biến tiềm ẩn `128×128×16` trở lại kích thước `1024×1024×3`. Hai hệ quả thực tế cần lưu tâm:

- **Cắt lát (slicing) hoặc lát gạch (tiling) quá trình giải mã.** Thư viện `diffusers` cung cấp `pipe.vae.enable_slicing()` và `pipe.vae.enable_tiling()`. Phương pháp lát gạch chấp nhận đánh đổi một khuyết tật đường nối rất nhỏ để giảm bộ nhớ xuống mức $O(\text{tile}^2)$ thay vì $O(H \cdot W)$. Điều này cực kỳ cần thiết khi xử lý độ phân giải từ 1024² trở lên trên GPU phổ thông.
- **Bộ giải mã bf16, tính toán fp32 cho bước đổi kích thước cuối.** VAE của SD 1.x ban đầu được phát hành ở định dạng fp32 và *âm thầm tạo ra các giá trị NaN* khi bị ép kiểu sang fp16 ở độ phân giải 1024²+. SDXL đi kèm phiên bản sửa lỗi `madebyollin/sdxl-vae-fp16-fix` — hãy luôn ưu tiên biến thể sửa lỗi fp16 này hoặc sử dụng bf16.

## Tài liệu đọc thêm

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — bài báo gốc về VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — $\beta$-VAE gỡ rối biểu diễn.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — bài báo gốc về VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — VAE phân cấp sâu tiên tiến nhất cho hình ảnh.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; ứng dụng VAE làm bộ mã hóa.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, chuẩn VAE cho âm thanh.
