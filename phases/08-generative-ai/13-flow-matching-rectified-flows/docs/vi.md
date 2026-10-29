# So khớp dòng & Dòng nắn thẳng (Flow Matching & Rectified Flows)

> Các mô hình khuếch tán truyền thống mất từ 20 đến 50 bước lấy mẫu (sampling steps) vì chúng phải di chuyển trên một quỹ đạo cong từ nhiễu về dữ liệu. Kỹ thuật so khớp dòng (flow matching - Lipman et al., 2023) và dòng nắn thẳng (rectified flow - Liu et al., 2022) huấn luyện mô hình học các quỹ đạo thẳng. Quỹ đạo thẳng hơn đồng nghĩa với việc cần ít bước hơn, giúp tốc độ suy luận (inference) nhanh vượt trội. Stable Diffusion 3, Flux.1 và AudioCraft 2 đều đã đồng loạt chuyển sang sử dụng so khớp dòng vào năm 2024.

**Type:** Xây dựng (Build)  
**Languages:** Python  
**Prerequisites:** Phase 8 · 06 (Mô hình xác suất khuếch tán khử nhiễu - DDPM), Phase 1 · Phép tính vi tích phân (Calculus)  
**Time:** ~45 phút  

## Vấn đề đặt ra (The Problem)

Quá trình nghịch đảo của DDPM là một bước đi ngẫu nhiên (stochastic walk) gồm 1000 bước xuất phát từ phân phối chuẩn $\mathcal{N}(0, I)$ quay trở về phân phối dữ liệu gốc. Mô hình DDIM đã thu gọn quá trình này xuống còn 20-50 bước xác định (deterministic steps). Tuy nhiên, mục tiêu thực tế luôn là giảm số bước xuống mức tối thiểu — lý tưởng nhất là chỉ cần một bước duy nhất. Trở ngại cốt lõi nằm ở chỗ phương trình vi phân thường (ODE) mô tả quá trình nghịch đảo có tính chất cứng (stiff) với quỹ đạo bị uốn cong mạnh.

Nếu chúng ta có thể huấn luyện mô hình sao cho quỹ đạo từ nhiễu về dữ liệu là một *đường thẳng*, thì chỉ cần một bước nhảy Euler duy nhất từ thời điểm $t=1$ về $t=0$ là hoàn tất. So khớp dòng (flow matching) trực tiếp hiện thực hóa ý tưởng này: định nghĩa một đường nội suy thẳng nối từ $x_1 \sim \mathcal{N}(0, I)$ tới $x_0 \sim \text{data}$, huấn luyện một trường vector (vector field) $v_\theta(x, t)$ để khớp với đạo hàm theo thời gian của đường thẳng đó, sau đó tiến hành tích phân trường vector này khi suy luận.

Dòng nắn thẳng (Rectified flow - Liu, 2022) tiến xa hơn nữa: nắn thẳng các quỹ đạo một cách lặp đi lặp lại thông qua quy trình tái dòng (reflow), tạo ra một phương trình ODE ngày càng tiệm cận tuyến tính. Chỉ sau hai vòng lặp reflow, một bộ lấy mẫu 2 bước đã có thể bắt kịp chất lượng của mô hình DDPM chạy 50 bước.

## Khái niệm cốt lõi (The Concept)

![So khớp dòng: nội suy đường thẳng giữa nhiễu và dữ liệu](../assets/flow-matching.svg)

### Dòng chảy đường thẳng (Straight-line flow)

Định nghĩa quỹ đạo nội suy:

```math
x_t = t \cdot x_1 + (1 - t) \cdot x_0, \quad t \in [0, 1]
```

trong đó $x_0 \sim \text{data}$ và $x_1 \sim \mathcal{N}(0, I)$. Đạo hàm theo thời gian dọc theo đường thẳng này là một hằng số:

```math
\frac{dx_t}{dt} = x_1 - x_0
```

Ta định nghĩa một trường vector nơ-ron $v_\theta(x_t, t)$ và huấn luyện nó khớp với đạo hàm này:

```math
L = \mathbb{E}_{x_0, x_1, t} \left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2
```

Đây chính là hàm mất mát **so khớp dòng có điều kiện (conditional flow matching)** (Lipman, 2023). Quá trình huấn luyện hoàn toàn không cần mô phỏng vi phân (simulation-free): bạn không bao giờ phải trải phẳng phương trình ODE. Chỉ cần lấy mẫu ngẫu nhiên các bộ $(x_0, x_1, t)$ và thực hiện bài toán hồi quy thông thường.

### Lấy mẫu (Sampling)

Trong quá trình suy luận, ta tích phân trường vector đã học *ngược chiều* thời gian:

```math
x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)
```

Bắt đầu từ điểm nhiễu ban đầu $x_1 \sim \mathcal{N}(0, I)$, thực hiện các bước nhảy Euler lùi dần về $t=0$.

### Dòng nắn thẳng (Rectified flow - Liu 2022)

Dù công thức nội suy ban đầu là đường thẳng, các đường cong thực tế mà mô hình học được *không hoàn toàn thẳng* — chúng bị uốn cong vì nhiều điểm dữ liệu $x_0$ khác nhau có thể cùng ánh xạ tới một điểm nhiễu $x_1$. Bước tái dòng (reflow) của Rectified Flow giải quyết vấn đề này:

1. Huấn luyện mô hình dòng $v_1$ với các cặp điểm ghép ngẫu nhiên.
2. Lấy mẫu $N$ cặp $(x_1, x_0)$ bằng cách tích phân mô hình $v_1$ từ điểm bắt đầu $x_1$ cho đến điểm đáp $x_0$ tương ứng.
3. Huấn luyện mô hình thứ hai $v_2$ trên các cặp điểm vừa thu được. Do các cặp này hiện đã được "ghép cặp theo quy luật ODE", đường nội suy nối giữa chúng trở nên phẳng hơn rõ rệt.
4. Lặp lại quy trình trên.

Trong thực tế, 2 vòng lặp reflow là đủ để đưa các quỹ đạo về gần như tuyến tính tuyệt đối, cho phép suy luận chất lượng cao chỉ với 2 đến 4 bước. SDXL-Turbo, SD3-Turbo và LCM đều là các mô hình được chưng cất từ nền tảng so khớp dòng.

### Vì sao phương pháp này chiến thắng trong lĩnh vực hình ảnh năm 2024

Ba lý do mang tính quyết định:

1. **Huấn luyện không cần mô phỏng (Simulation-free training)** — không cần giải mở rộng phương trình ODE trong quá trình huấn luyện, cài đặt vô cùng đơn giản.
2. **Hình học hàm mất mát ưu việt hơn** — các quỹ đạo thẳng duy trì tỷ lệ tín hiệu trên nhiễu (SNR) ổn định, trong khi hàm mất mát dự đoán nhiễu $\epsilon$ của DDPM gặp vấn đề nghiêm trọng về SNR tại các vùng biên của lịch trình nhiễu.
3. **Suy luận nhanh hơn** — chỉ cần 4-8 bước là đạt chất lượng tương đương SDXL-Turbo; thậm chí 1 bước duy nhất khi kết hợp với kỹ thuật chưng cất tính nhất quán (consistency distillation).

## So khớp dòng và DDPM — Mối liên hệ toán học chính xác (Flow matching vs DDPM — the exact connection)

So khớp dòng với quỹ đạo có điều kiện Gaussian thực chất chính là mô hình khuếch tán *đi kèm một lịch trình nhiễu cụ thể*. Nếu chọn lịch trình dạng $x_t = \alpha(t) x_0 + \sigma(t) x_1$, so khớp dòng sẽ tái lập chính xác mô hình khuếch tán được cải biên theo tích phân Stratonovich với vận tốc $v = \alpha' \cdot x_0 - \sigma' \cdot x_1$. Hai phương pháp này tương đương nhau về mặt đại số đối với các quỹ đạo Gaussian.

Giá trị đột phá mà so khớp dòng mang lại nằm ở: sự *tường minh* của mục tiêu dự đoán (một vận tốc trực quan), hàm mất mát sạch hơn, và khả năng tự do thử nghiệm với các đường nội suy phi Gaussian.

````figure
normalizing-flow
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` cài đặt thuật toán so khớp dòng 1 chiều trên một phân phối hỗn hợp Gaussian 2 thành phần (two-mode Gaussian mixture). Trường vector $v_\theta(x, t)$ là một mạng MLP nhỏ được huấn luyện theo mục tiêu đường thẳng. Trong giai đoạn suy luận, ta tiến hành tích phân qua 1, 2, 4 và 20 bước Euler để so sánh chất lượng mẫu sinh ra.

### Bước 1: Hàm mất mát khi huấn luyện (Step 1: training loss)

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # lan truyền ngược + cập nhật trọng số
```

### Bước 2: Lấy mẫu nhiều bước (Step 2: multi-step inference)

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Bước 3: So sánh số bước lấy mẫu (Step 3: compare step counts)

Bạn sẽ nhận thấy bộ lấy mẫu chỉ 4 bước đã có thể bắt kịp chất lượng của bộ lấy mẫu 20 bước — một cải thiện mang tính bước ngoặt đối với độ trễ hệ thống.

## Cạm bẫy thực tế (Pitfalls)

- **Quy ước tham số thời gian (Time parameterization).** So khớp dòng sử dụng thang thời gian $t \in [0, 1]$ với $t=0$ đại diện cho dữ liệu sạch, $t=1$ là nhiễu thuần túy. Ngược lại, DDPM dùng $t \in [0, T]$ với $t=0$ là dữ liệu và $t=T$ là nhiễu. Cùng hướng đi nhưng khác biệt về thang đo. Nhiều bài báo khoa học thường nhầm lẫn chi tiết này.
- **Lựa chọn lịch trình (Schedule choice).** Đường thẳng của rectified flow là lịch trình tiêu chuẩn của so khớp dòng, nhưng bạn hoàn toàn có thể áp dụng lịch trình lấy mẫu thời gian cosin hoặc logit-normal (như SD3 áp dụng) để bao phủ các thang độ hạt tốt hơn.
- **Chi phí tái dòng (Reflow cost).** Việc tạo bộ dữ liệu cặp điểm cho quá trình reflow đòi hỏi phải chạy toàn bộ quy trình suy luận cho từng mẫu. Chỉ nên áp dụng reflow khi bạn thực sự cần suy luận siêu tốc trong 1-2 bước.
- **Kỹ thuật hướng dẫn không cần bộ phân loại (Classifier-free guidance - CFG) vẫn áp dụng bình thường.** Chỉ cần thay thế $\epsilon$ bằng $v$ trong phép tổ hợp tuyến tính: $v_{\text{cfg}} = (1+w) v_{\text{cond}} - w v_{\text{uncond}}$.

## Ứng dụng thực tế (Use It)

| Trường hợp sử dụng | Ngăn xếp công nghệ 2026 (2026 stack) |
|---------------------|---------------------------------------|
| Tạo ảnh từ văn bản, chất lượng cao nhất | So khớp dòng: SD3, Flux.1-dev |
| Tạo ảnh từ văn bản, từ 1 đến 4 bước | So khớp dòng đã chưng cất: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Suy luận thời gian thực | Chưng cất tính nhất quán từ mô hình nền so khớp dòng (LCM, PCM) |
| Tạo âm thanh | So khớp dòng: Stable Audio 2.5, AudioCraft 2 |
| Tạo video | So khớp dòng kết hợp với khuếch tán (Sora, Veo, Stable Video) |
| Ứng dụng khoa học / vật lý (quỹ đạo hạt, phân tử) | So khớp dòng kết hợp trường vector đẳng biến (equivariant vector field) |

Bất cứ khi nào một công bố khoa học trong giai đoạn 2025-2026 nhắc tới việc "nhanh hơn khuếch tán truyền thống", cốt lõi hầu như luôn là so khớp dòng kết hợp với kỹ thuật chưng cất (distillation).

## Đóng gói sản phẩm (Ship It)

Lưu tệp `outputs/skill-fm-tuner.md`. Kỹ năng này nhận một đặc tả mô hình theo kiểu khuếch tán và chuyển đổi nó thành một cấu hình huấn luyện so khớp dòng: lựa chọn lịch trình, phân phối lấy mẫu thời gian (đều / logit-normal), trình tối ưu hóa, kế hoạch reflow, số bước suy luận mục tiêu và quy thức đánh giá.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py` và so sánh sai số MSE của 1 bước so với 20 bước đối chiếu với phân phối dữ liệu thực.
2. **Trung bình.** Chuyển đổi từ lấy mẫu $t$ đều (uniform) sang lấy mẫu logit-normal (tập trung lấy mẫu ở giai đoạn giữa $t$). Chất lượng mô hình có được cải thiện không?
3. **Khó.** Cài đặt một vòng lặp reflow hoàn chỉnh: tạo các cặp $(x_0, x_1)$ bằng cách tích phân mô hình đầu tiên, huấn luyện mô hình thứ hai trên các cặp này và so sánh chất lượng lấy mẫu chỉ sau 1 bước duy nhất.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Flow matching | "Khuếch tán đường thẳng" | Huấn luyện $v_\theta(x, t)$ để khớp với giá trị $x_1 - x_0$ dọc theo đường nội suy. |
| Rectified flow | "Nắn dòng / Tái dòng" (Reflow) | Quy trình lặp giúp nắn thẳng các dòng đã học. |
| Velocity field | "Trường vận tốc v_θ" | Đầu ra của mô hình — hướng dịch chuyển cần áp dụng cho $x_t$. |
| Straight-line interpolant | "Quỹ đạo thẳng" | $x_t = (1-t) \cdot x_0 + t \cdot x_1$; đạo hàm mục tiêu là hằng số hiển nhiên. |
| Euler sampler | "Bộ giải ODE bậc 1" | Trình tích phân đơn giản nhất; hoạt động tối ưu khi các đường đi là đường thẳng. |
| Logit-normal t | "Lấy mẫu thời gian kiểu SD3" | Tập trung lấy mẫu $t$ vào vùng giữa, nơi các gradient mang tín hiệu mạnh nhất. |
| Consistency distillation | "Bộ lấy mẫu 1 bước" | Huấn luyện mô hình học sinh để ánh xạ trực tiếp bất kỳ trạng thái $x_t$ nào về ngay $x_0$. |
| CFG with velocity | "v-CFG" | $v_{\text{cfg}} = (1+w) v_{\text{cond}} - w v_{\text{uncond}}$; cùng một cơ chế, chỉ thay đổi biến số. |

## Lưu ý triển khai: Flux.1-schnell là minh chứng đỉnh cao về tốc độ của so khớp dòng (Production note: Flux.1-schnell is flow matching at its fastest)

Thành tựu mang tính biểu tượng của so khớp dòng trong môi trường sản xuất chính là Flux.1-schnell — một mô hình DiT so khớp dòng được chưng cất xuống chỉ còn 1-4 bước suy luận mà vẫn bảo toàn chất lượng đồ họa đỉnh cao tương đương Flux-dev. Bản notebook mẫu của Niels "Chạy Flux trên máy 8GB RAM" được coi là công thức triển khai quy chuẩn: bộ mã hóa T5 + CLIP, khử nhiễu MMDiT lượng tử hóa (chỉ mất 4 bước đối với phiên bản schnell so với 50 bước của dev), giải mã bằng VAE. Phân tích chi phí tài nguyên:

| Biến thể | Số bước | Độ trễ ở kích thước 1024² trên L4 | Tổng FLOPs (tương đối) |
|----------|---------|-----------------------------------|------------------------|
| Flux.1-dev (gốc) | 50 | ~15 giây | 1.0× |
| Flux.1-schnell | 4 | ~1.2 giây | 0.08× (nhanh hơn 12 lần) |
| SDXL-base | 30 | ~4 giây | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 giây | 0.03× |

Quy tắc bất thành văn trong triển khai thực tế: **mô hình nền tảng so khớp dòng + kỹ thuật chưng cất = lựa chọn mặc định vào năm 2026 cho hệ thống tạo ảnh từ văn bản siêu tốc.** Mọi nhà phát triển lớn đều phân phối tổ hợp này: SD3-Turbo (SD3 + flow + distillation), Flux-schnell (Flux-dev + nắn thẳng bằng rectified-flow), CogView-4-Flash. Các mô hình khuếch tán thuần túy giờ đây hầu như chỉ còn tồn tại dưới dạng các checkpoint di sản.

## Tài liệu đọc thêm (Further Reading)

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — Bài báo đặt nền móng cho rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Bài báo khởi nguồn cho flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, ứng dụng rectified flow ở quy mô lớn.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — Khung lý thuyết tổng quát bao quát cả FM và khuếch tán.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — Kỹ thuật chưng cất 1 bước cho khuếch tán và mô hình dòng.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — Biến thể turbo ứng dụng kỹ thuật chưng cất đối kháng.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — Triển khai so khớp dòng trong môi trường thực tế quy mô lớn.
