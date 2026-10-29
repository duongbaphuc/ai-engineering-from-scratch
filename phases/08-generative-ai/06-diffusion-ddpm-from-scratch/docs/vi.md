# Mô hình khuếch tán — Xây dựng DDPM từ đầu (Diffusion Models — DDPM from Scratch)

> Ho, Jain, Abbeel (2020) đã mang đến cho cộng đồng AI một công thức kinh điển: Phá hủy dữ liệu bằng nhiễu qua một nghìn bước nhỏ. Huấn luyện một mạng nơ-ron để dự đoán lượng nhiễu đã thêm vào. Đảo ngược quá trình đó khi suy luận (inference). Ngày nay, hầu như mọi mô hình tạo sinh hình ảnh, video, 3D và âm nhạc chủ đạo đều vận hành trên vòng lặp này, có thể kết hợp thêm phối khớp dòng (flow matching) hoặc các kỹ thuật tính nhất quán (consistency models).

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Yêu cầu tiên quyết:** Phase 3 · 02 (Lan truyền ngược - Backprop), Phase 8 · 02 (Mạng tự mã hóa biến phân - VAE)  
**Thời lượng:** ~75 phút  

## Vấn đề cần giải quyết (The Problem)

Bạn muốn xây dựng một bộ lấy mẫu (sampler) cho phân phối dữ liệu thực $p_{\text{data}}(x)$. Mạng đối nghịch tạo sinh (Generative Adversarial Networks - GAN) vận hành theo trò chơi minimax đối kháng thường xuyên phân kỳ hoặc sụp đổ mode (mode collapse). Mạng tự mã hóa biến phân (Variational Autoencoders - VAE) lại tạo ra các mẫu mờ nhòe do bộ giải mã Gaussian giả định. Mục tiêu lý tưởng của một mô hình tạo sinh là: (a) một hàm mất mát duy nhất, ổn định (không điểm yên ngựa, không minimax), (b) tối ưu hóa chặn dưới của hàm hợp lý $\log p(x)$ (để đánh giá phân phối một cách tường minh), và (c) chất lượng mẫu sinh ra đạt chuẩn vượt trội (state-of-the-art - SOTA).

Sohl-Dickstein cùng cộng sự (2015) đã đề xuất câu trả lời lý thuyết dựa trên nhiệt động lực học phi cân bằng: định nghĩa một chuỗi Markov $q(x_t \mid x_{t-1})$ bổ sung dần nhiễu Gaussian, và huấn luyện chuỗi ngược $p_\theta(x_{t-1} \mid x_t)$ để khử nhiễu. Ho, Jain, Abbeel (2020) đã chứng minh rằng mục tiêu huấn luyện có thể rút gọn thành một dòng thanh lịch — dự đoán lượng nhiễu — và hoàn thiện toàn bộ công thức toán học. Năm 2020, đây từng được xem là một khám phá học thuật thú vị. Năm 2021, nó tạo ra các mẫu vượt qua cả GAN. Năm 2022, nó trở thành nền tảng của Stable Diffusion. Đến năm 2026, nó là hạ tầng cốt lõi của toàn bộ lĩnh vực AI tạo sinh.

## Khái niệm cốt lõi (The Concept)

![DDPM: quá trình thêm nhiễu xuôi, khử nhiễu ngược](../assets/ddpm.svg)

**Quá trình xuôi $q$ (Forward process).** Thêm nhiễu Gaussian dần dần qua $T$ bước nhỏ. Điểm mấu chốt giúp giải tích toán học trở nên khả thi là quá trình tích lũy nhiễu qua nhiều bước vẫn có dạng phân phối Gaussian đóng (closed-form):

```math
q(x_t \mid x_0) = \mathcal{N}\left( \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I \right)
```

trong đó $\bar\alpha_t = \prod_{s=1}^{t} (1 - \beta_s)$ với lịch trình phương sai nhiễu $\beta_t$. Nếu chọn $\beta_t$ tăng tuyến tính từ 1e-4 đến 0.02 qua $T=1000$ bước, trạng thái $x_T$ ở bước cuối cùng sẽ xấp xỉ hoàn toàn nhiễu trắng chuẩn tắc $\mathcal{N}(0, I)$.

**Quá trình ngược $p_\theta$ (Reverse process).** Huấn luyện một mạng nơ-ron $\epsilon_\theta(x_t, t)$ để dự đoán lượng nhiễu đã được thêm vào tại bước $t$. Với đầu vào là $x_t$, ta khử nhiễu theo công thức:

```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \cdot \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z
```

trong đó $\sigma_t$ được chọn bằng $\sqrt{\beta_t}$ hoặc phương sai có thể học được. Biểu thức này nhìn có vẻ phức tạp nhưng thực chất xuất phát từ đại số xác suất — giải nghiệm cho $x_{t-1}$ dựa trên hậu nghiệm $q(x_{t-1} \mid x_t, x_0)$ và thay thế $x_0$ bằng ước lượng phục hồi từ nhiễu dự đoán.

**Hàm mất mát huấn luyện (Training loss).**

```math
L_{\text{simple}} = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\lVert \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, t \right) \right\rVert^2 \right]
```

Lấy ngẫu nhiên mẫu $x_0$ từ tập dữ liệu, chọn một bước thời gian $t$ ngẫu nhiên, lấy mẫu nhiễu chuẩn $\epsilon \sim \mathcal{N}(0, I)$, tính trực tiếp $x_t$ bị nhiễu chỉ trong một phép toán nhờ công thức dạng đóng, và hồi quy để dự đoán nhiễu. Một hàm mất mát duy nhất, không minimax, không phân kỳ KL tường minh khi cài đặt, không cần mẹo tái tham số hóa (reparameterization trick) phức tạp.

**Lấy mẫu (Sampling).** Khởi tạo $x_T \sim \mathcal{N}(0, I)$. Lặp lại bước khử nhiễu ngược từ $t = T$ lùi về 1. Hoàn tất.

## Tại sao phương pháp này hiệu quả (Why it works)

Ba trực giác cốt lõi:

1. **Khử nhiễu cục bộ dễ hơn sinh mẫu toàn cục.** Tại $t=T$, dữ liệu là nhiễu trắng thuần túy — mạng nơ-ron chỉ cần giải một bài toán định hướng thô. Tại $t=0$, mạng chỉ cần tinh chỉnh vài pixel nhỏ. Ở các bước $t$ trung gian, bài toán mang tính thử thách cao nhưng mạng được tiếp nhận gradient liên tục chảy qua cùng một bộ trọng số ở mọi mức độ nhiễu khác nhau.
2. **Khớp điểm số (Score matching) dưới hình thức khác.** Vincent (2011) đã chứng minh rằng việc dự đoán nhiễu tương đương với ước lượng $\nabla_x \log q(x_t \mid x_0)$, gọi là *hàm điểm số* (score function). Phương trình vi phân ngẫu nhiên (SDE) ngược sử dụng điểm số này để leo lên gradient mật độ — thực hiện bước đi ngẫu nhiên có định hướng (guided random walk) về phía các vùng phân phối có xác suất cao.
3. **Chặn dưới biến phân (ELBO) quy về sai số toàn phương trung bình (MSE) đơn giản.** Toàn bộ chặn dưới biến phân bao gồm một số hạng phân kỳ KL cho mỗi bước thời gian. Với cách tham số hóa của DDPM, các số hạng KL đó rút gọn thành hàm MSE dự đoán nhiễu nhân với các hệ số trọng số cụ thể; Ho et al. đã bỏ qua các hệ số phức tạp này (gọi là hàm mất mát "đơn giản" - simple loss) và bất ngờ thay, chất lượng mẫu sinh ra thậm chí *tốt hơn hẳn*.

````figure
diffusion-denoise
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` cài đặt một mô hình DDPM 1 chiều từ đầu. Dữ liệu là một phân phối hỗn hợp hai mode (two-mode mixture). "Mạng nơ-ron" là một MLP nhỏ nhận đầu vào $(x_t, t)$ và xuất ra nhiễu dự đoán. Huấn luyện bằng hàm mất mát một dòng. Lấy mẫu bằng cách lặp chuỗi đảo ngược.

### Bước 1: Lịch trình nhiễu xuôi (dạng đóng)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Bước 2: Lấy mẫu `x_t` trong một bước

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Bước 3: Một bước huấn luyện

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Bước 4: Lấy mẫu ngược (reverse sampling)

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Với bài toán 1 chiều gồm 40 bước thời gian và mạng MLP 24 nơ-ron, mô hình học được phân phối hai mode sau khoảng ~200 epoch.

## Điều kiện hóa theo thời gian (Time conditioning)

Mạng nơ-ron cần nhận biết chính xác nó đang khử nhiễu ở bước thời gian nào. Hai phương pháp tiêu chuẩn:

- **Nhúng hình sin (Sinusoidal embedding).** Tương tự như mã hóa vị trí (positional encoding) trong Transformer: $\text{embed}(t) = [\sin(t/\omega_0), \cos(t/\omega_0), \sin(t/\omega_1), \ldots]$. Vector này được đưa qua một mạng MLP rồi truyền vào các tầng của mạng.
- **Điều kiện hóa chuẩn hóa nhóm/FiLM (FiLM / group-norm conditioning).** Chiếu vector nhúng thành tỷ lệ phóng đại và độ lệch (scale/bias) cho từng kênh đặc trưng ở mỗi khối xử lý (Feature-wise Linear Modulation - FiLM).

Đoạn mã mô phỏng của chúng ta dùng phép nhúng hình sin rồi ghép nối (concatenate). Các kiến trúc U-Net thực tế trong sản xuất thường dùng FiLM.

## Cạm bẫy thường gặp (Pitfalls)

- **Lịch trình nhiễu đóng vai trò quyết định.** Lịch trình tuyến tính $\beta$ là mặc định của DDPM ban đầu, nhưng lịch trình cosine (Nichol & Dhariwal, 2021) mang lại chỉ số FID tốt hơn nhiều trên cùng mức ngân sách tính toán. Hãy đổi lịch trình nếu chất lượng bị bão hòa.
- **Mã hóa bước thời gian rất nhạy cảm.** Truyền trực tiếp $t$ dưới dạng số thực float có thể chạy được với dữ liệu đồ chơi 1-D nhưng sẽ thất bại hoàn toàn trên dữ liệu ảnh thực tế; luôn sử dụng bộ nhúng thời gian phù hợp.
- **Dự đoán vận tốc (V-prediction) so với dự đoán nhiễu (ε-prediction).** Ở các miền giá trị hẹp (khi $t$ rất nhỏ hoặc rất lớn), $\epsilon$ có tỷ lệ tín hiệu trên nhiễu (SNR) rất kém. Cơ chế dự đoán vận tốc ($v = \alpha \cdot \epsilon - \sigma \cdot x$) ổn định hơn nhiều; SDXL, SD3 và Flux đều sử dụng kỹ thuật này.
- **Hướng dẫn không cần bộ phân loại (Classifier-free guidance - CFG).** Khi suy luận, tính toán cả giá trị $\epsilon$ có điều kiện và không điều kiện, sau đó nội suy $\epsilon_{\text{cfg}} = (1 + w) \cdot \epsilon_{\text{cond}} - w \cdot \epsilon_{\text{uncond}}$ với trọng số hướng dẫn $w \approx 3\text{-}7$. Chi tiết được trình bày trong Bài 08.
- **1000 bước là quá nhiều.** Các hệ thống production thực tế áp dụng DDIM (20-50 bước), DPM-Solver (10-20 bước), hoặc kỹ thuật chưng cất tri thức (1-4 bước). Xem Bài 12.

## Ứng dụng thực tế (Use It)

| Vai trò | Ngăn xếp công nghệ điển hình năm 2026 |
|---|---|
| Khuếch tán trong không gian pixel (nhỏ, mô hình mẫu) | DDPM + U-Net |
| Khuếch tán trong không gian tiềm ẩn ảnh (Latent diffusion) | Bộ mã hóa VAE + U-Net hoặc DiT (Bài 07) |
| Khuếch tán video trong không gian tiềm ẩn | DiT không-thời gian (Spatiotemporal DiT như Sora, Veo, WAN) |
| Khuếch tán âm thanh trong không gian tiềm ẩn | Encodec + diffusion transformer |
| Ứng dụng khoa học (phân tử, protein, vật lý) | Khuếch tán đẳng biến (Equivariant diffusion: EDM, RFdiffusion, AlphaFold3) |

Khuếch tán là khung xương sống tạo sinh vạn năng. Kỹ thuật phối khớp dòng (Flow matching - Bài 13) là đối thủ cạnh tranh nổi bật trong giai đoạn 2024-2026, thường vượt trội về tốc độ suy luận ở cùng mức chất lượng.

## Đóng gói sản phẩm (Ship It)

Lưu cấu hình tại `outputs/skill-diffusion-trainer.md`. Kỹ năng này nhận đầu vào là tập dữ liệu + ngân sách tính toán và xuất ra: lịch trình nhiễu (linear/cosine/sigmoid), mục tiêu dự đoán (ε/v/x), số bước thời gian, hệ số hướng dẫn (guidance scale), họ bộ lấy mẫu, và quy trình đánh giá chuẩn mực.

## Bài tập thực hành (Exercises)

1. **Dễ.** Thay đổi $T$ từ 40 xuống 10 trong `code/main.py`. Chất lượng mẫu (biểu đồ phân bố tần suất trực quan) suy giảm như thế nào? Ở giá trị $T$ nào thì cấu trúc hai mode bị sụp đổ hoàn toàn?
2. **Trung bình.** Chuyển đổi từ cơ chế dự đoán $\epsilon$ sang dự đoán vận tốc $v$. Thiết lập lại công thức cho bước khử nhiễu ngược. So sánh chất lượng mẫu sinh ra cuối cùng.
3. **Khó.** Tích hợp cơ chế hướng dẫn không cần bộ phân loại (classifier-free guidance). Đưa nhãn lớp $c \in \lbrace 0, 1 \rbrace$ vào làm điều kiện, loại bỏ điều kiện này ngẫu nhiên 10% số lần trong quá trình huấn luyện, và khi lấy mẫu sử dụng $\epsilon = (1+w) \cdot \epsilon_{\text{cond}} - w \cdot \epsilon_{\text{uncond}}$. Đo đạc tỷ lệ tạo mẫu chính xác đúng mode mục tiêu tại các mức $w = 0, 1, 3, 7$.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|---|---|---|
| Forward process | "Thêm nhiễu" | Chuỗi Markov cố định $q(x_t \mid x_{t-1})$ dùng để phá hủy dần dữ liệu. |
| Reverse process | "Khử nhiễu" | Chuỗi Markov học được $p_\theta(x_{t-1} \mid x_t)$ giúp tái tạo lại dữ liệu gốc. |
| β schedule | "Thang bậc nhiễu" | Lịch trình phương sai cho từng bước; có thể là tuyến tính, cosine hoặc sigmoid. |
| α̅ (alpha bar) | "Alpha tích lũy" | Tích lũy thừa số $\prod (1 - \beta)$; cho phép tính trực tiếp $x_t$ từ $x_0$ dạng đóng. |
| Simple loss | "MSE trên nhiễu" | $\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$; toàn bộ khai triển biến phân quy về phép toán này. |
| ε-prediction | "Dự đoán nhiễu" | Đầu ra của mạng là lượng nhiễu đã được thêm vào; dạng chuẩn của DDPM. |
| V-prediction | "Dự đoán vận tốc" | Đầu ra là $\alpha \cdot \epsilon - \sigma \cdot x$; giúp điều hòa gradient ổn định hơn trên mọi $t$. |
| DDPM | "Bài báo gốc" | Ho et al. 2020; lịch trình $\beta$ tuyến tính, 1000 bước, kiến trúc U-Net. |
| DDIM | "Bộ lấy mẫu xác định" | Bộ lấy mẫu phi Markov, 20-50 bước, giữ nguyên mô hình đã huấn luyện. |
| Classifier-free guidance | "CFG" | Kết hợp dự đoán có điều kiện và không điều kiện để khuếch đại độ bám sát điều kiện. |

## Lưu ý trong môi trường sản xuất: Suy luận khuếch tán là bài toán về số bước

Bài báo DDPM gốc chạy $T=1000$ bước khử nhiễu ngược. Không một hệ thống sản xuất thực tế nào triển khai nguyên bản con số đó. Mọi ngăn xếp suy luận thực tế đều lựa chọn một trong ba chiến lược sau — mỗi chiến lược ánh xạ trực tiếp đến câu hỏi "độ trễ phát sinh từ đâu":

1. **Bộ lấy mẫu nhanh hơn, giữ nguyên mô hình.** DDIM (20-50 bước), DPM-Solver++ (10-20 bước), UniPC (8-16 bước). Thay thế trực tiếp vòng lặp khử nhiễu ngược; trọng số mạng $\epsilon_\theta$ đã huấn luyện được giữ nguyên hoàn toàn. Giảm độ trễ từ 20 đến 50 lần.
2. **Chưng cất tri thức (Distillation).** Huấn luyện một mô hình học sinh mô phỏng mô hình giáo viên với số bước ít hơn: Chưng cất lũy tiến (Progressive Distillation: 2 → 1), Mô hình nhất quán (Consistency Models: số bước bất kỳ → 1-4 bước), LCM, SDXL-Turbo, SD3-Turbo. Giảm độ trễ thêm 5-10 lần, đòi hỏi huấn luyện lại.
3. **Bộ nhớ đệm và biên dịch tối ưu (Caching & compilation).** Sử dụng `torch.compile(unet, mode="reduce-overhead")`, backend khuếch tán của TensorRT-LLM, cơ chế chú ý `xformers`/SDPA, trọng số kiểu bf16. Cắt giảm độ trễ mỗi bước khoảng 2 lần. Có thể kết hợp đồng thời với giải pháp (1) và (2).

Đối với máy chủ suy luận khuếch tán trong sản xuất, bài toán chi phí tài nguyên tương đồng với các mô hình ngôn ngữ lớn (LLM): độ trễ tổng thể bằng `số_bước × chi_phí_mỗi_bước + giải_mã_VAE`, thông lượng đạt `kích_thước_batch × (số_bước × chi_phí_mỗi_bước)^-1`. Thời gian tới token đầu tiên (TTFT) ở đây chỉ là chi phí của bước khử nhiễu đầu; còn thời gian trên mỗi token (TPOT) tương đương với toàn bộ thời gian phản hồi bởi vì dưới góc nhìn của người dùng, sinh ảnh là quá trình "nhận toàn bộ kết quả một lần".

## Tài liệu đọc thêm (Further Reading)

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — bài báo nền móng về mô hình khuếch tán, đi trước thời đại.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — bài báo kinh điển về DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, giảm thiểu số bước lấy mẫu.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — lịch trình cosine, phương sai học được.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — hướng dẫn bằng bộ phân loại (classifier guidance).
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — kỹ thuật CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — hệ thống ký hiệu thống nhất và công thức thiết kế chuẩn mực nhất.
