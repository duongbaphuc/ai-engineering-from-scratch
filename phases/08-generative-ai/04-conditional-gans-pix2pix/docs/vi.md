# GAN có điều kiện & Pix2Pix (Conditional GANs & Pix2Pix)

> Bước khai phóng lớn đầu tiên của giai đoạn 2014-2017 là khả năng kiểm soát nội dung mà GAN tạo ra. Gắn thêm một nhãn lớp, một bức ảnh, hoặc một câu văn. Pix2Pix đã hiện thực hóa phiên bản tạo ảnh dựa trên ảnh điều kiện, và đến nay nó vẫn đánh bại mọi mô hình tạo ảnh từ văn bản tổng quát trong các tác vụ chuyển đổi ảnh-sang-ảnh hẹp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (Mạng tích chập - CNNs)
**Time:** ~75 phút

## Vấn đề đặt ra

Một mạng GAN không điều kiện chỉ lấy mẫu các khuôn mặt ngẫu nhiên. Điều này thú vị trong các bản thử nghiệm (demo), nhưng vô dụng trong môi trường sản xuất. Điều bạn thực sự cần là: *chuyển một bản phác thảo thành một bức ảnh thực tế*, *chuyển bản đồ đường nét thành ảnh chụp từ vệ tinh*, *chuyển cảnh ban ngày thành cảnh ban đêm*, *tô màu cho một bức ảnh đen trắng*. Trong tất cả các bài toán đó, bạn được cung cấp một ảnh đầu vào $x$ và phải tạo ra đầu ra $y$ có sự tương ứng chặt chẽ về mặt ngữ nghĩa. Ứng với mỗi $x$ có rất nhiều khả năng $y$ hợp lý. Hàm mất mát sai số bình phương trung bình (MSE) sẽ san phẳng tất cả các khả năng đó thành một mớ nhòe nhoẹt. Trong khi đó, hàm mất mát đối nghịch không mắc phải nhược điểm này, bởi vì tiêu chí "trông giống như thật" luôn mang tính sắc nét cao.

Mô hình GAN có điều kiện (Conditional GAN - cGAN, Mirza & Osindero, 2014) bổ sung thêm một biến điều kiện $c$ làm đầu vào cho cả $G$ và $D$. Pix2Pix (Isola et al., 2017) đã chuyên biệt hóa ý tưởng này: điều kiện đầu vào là một bức ảnh đầy đủ, bộ sinh là một mạng U-Net, bộ phân biệt là một bộ phân loại *dựa trên mảng cục bộ* (PatchGAN), và hàm mất mát là sự kết hợp giữa mất mát đối nghịch cùng chuẩn $L_1$. Công thức này vượt trội hơn các mô hình tạo ảnh từ văn bản huấn luyện từ đầu trên các miền chuyển dịch ảnh-sang-ảnh hẹp ngay cả trong năm 2026, bởi vì nó được huấn luyện trên *dữ liệu ghép cặp (paired data)* — bạn sở hữu chính xác tín hiệu giám sát mà bạn cần.

## Khái niệm cốt lõi

![Pix2Pix: bộ sinh U-Net, bộ phân biệt PatchGAN](../assets/pix2pix.svg)

**Bộ sinh có điều kiện $G$.** $G(x, z) \to y$. Trong Pix2Pix, $z$ được mô phỏng thông qua cơ chế dropout bên trong $G$ (không cần vector nhiễu đầu vào rõ ràng — Isola nhận thấy rằng việc đưa nhiễu tường minh vào đầu vào thường bị mạng phớt lờ hoàn toàn).

**Bộ phân biệt có điều kiện $D$.** $D(x, y) \to [0, 1]$. Đầu vào là *cặp dữ liệu* (điều kiện, đầu ra). Đây là điểm khác biệt mấu chốt: $D$ phải đánh giá xem liệu $y$ có nhất quán với $x$ hay không, chứ không chỉ đơn thuần xét xem $y$ có trông như thật hay không.

**Bộ sinh kiến trúc U-Net.** Mạng mã hóa - giải mã (encoder-decoder) kết hợp các kết nối tắt (skip connections) bắc ngang qua nút thắt cổ chai (bottleneck). Cực kỳ quan trọng đối với các tác vụ mà đầu vào và đầu ra chia sẻ cấu trúc hình học mức thấp (đường viền, hình bóng). Nếu không có các kết nối tắt, các chi tiết tần số cao sẽ biến mất hoàn toàn.

**Bộ phân biệt PatchGAN.** Thay vì xuất ra một điểm số thật/giả duy nhất cho toàn bộ bức ảnh, $D$ xuất ra một lưới kích thước $N \times N$, trong đó mỗi ô đánh giá một vùng tiếp nhận (receptive field) có kích thước khoảng ~70×70 điểm ảnh. Điểm số cuối cùng là giá trị trung bình của lưới này. Đây là giả định trường ngẫu nhiên Markov (Markov random field): tính chân thực mang tính chất cục bộ. Cách tiếp cận này giúp huấn luyện nhanh hơn nhiều, tiêu tốn ít tham số hơn, và cho đầu ra sắc nét hơn.

**Hàm mất mát (Loss).**

```math
\begin{aligned}
\text{loss}_G &= -\log D(x, G(x)) + \lambda \cdot \lVert y - G(x) \rVert_1 \\
\text{loss}_D &= -\log D(x, y) - \log(1 - D(x, G(x)))
\end{aligned}
```

Số hạng $L_1$ giúp ổn định quá trình huấn luyện và thúc đẩy $G$ bám sát mục tiêu đã biết. Chuẩn $L_1$ mang lại các đường cạnh sắc nét hơn so với $L_2$ (tối ưu hóa theo trung vị thay vì giá trị trung bình). Giá trị mặc định trong Pix2Pix là $\lambda = 100$.

## CycleGAN — giải pháp khi không có dữ liệu ghép cặp

Pix2Pix đòi hỏi dữ liệu ghép cặp $(x, y)$. CycleGAN (Zhu et al., 2017) loại bỏ yêu cầu này bằng cách bổ sung thêm một hàm mất mát mới: mất mát *nhất quán chu trình (cycle consistency loss)*. Mô hình sử dụng hai bộ sinh $G: X \to Y$ và $F: Y \to X$. Chúng được huấn luyện sao cho $F(G(x)) \approx x$ và $G(F(y)) \approx y$. Cơ chế này cho phép bạn chuyển đổi ngựa thường thành ngựa vằn, chuyển mùa hè thành mùa đông mà không cần bất kỳ cặp ảnh tương ứng trực tiếp nào.

Vào năm 2026, bài toán chuyển đổi ảnh-sang-ảnh không ghép cặp phần lớn được thực hiện thông qua các mô hình khuếch tán (ControlNet, IP-Adapter) thay vì CycleGAN, nhưng tư tưởng nhất quán chu trình vẫn tiếp tục hiện diện trong hầu hết các công trình nghiên cứu về thích ứng miền (domain adaptation) không giám sát.

```figure
gx-patchgan
```

## Thực hành: Xây dựng

`code/main.py` cài đặt một mạng GAN có điều kiện thu nhỏ trên dữ liệu 1 chiều. Điều kiện $c$ là một nhãn lớp (0 hoặc 1). Nhiệm vụ: tạo ra một mẫu dữ liệu từ phân phối có điều kiện tương ứng với lớp đã cho.

### Bước 1: Nối điều kiện vào đầu vào của cả G và D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

Mã hóa one-hot là phương pháp đơn giản nhất. Các mô hình lớn hơn thường dùng vector nhúng được học (learned embeddings), phép điều chế FiLM, hoặc cơ chế chú ý chéo (cross-attention).

### Bước 2: Huấn luyện có điều kiện

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Bộ sinh phải khớp với phân phối dữ liệu thật *ứng với điều kiện đã cho*, chứ không phải phân phối biên duyên (marginal distribution).

### Bước 3: Xác minh đầu ra trên từng lớp

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Các cạm bẫy thường gặp

- **Điều kiện bị phớt lờ.** G học cách tạo ra phân phối biên duyên, còn D không xử phạt vì tín hiệu điều kiện quá yếu. Cách khắc phục: đưa điều kiện vào D quyết liệt hơn (ở các tầng sớm, không chỉ ở tầng cuối), sử dụng bộ phân biệt phép chiếu (projection discriminator, Miyato & Koyama 2018).
- **Trọng số $L_1$ quá thấp.** G trôi dạt về việc tạo ra các đầu ra trông có vẻ chân thực nhưng không bám sát điều kiện đầu vào. Hãy bắt đầu với $\lambda \approx 100$ cho các tác vụ kiểu Pix2Pix.
- **Trọng số $L_1$ quá cao.** G tạo ra các đầu ra bị mờ vì $L_1$ về bản chất vẫn là một chuẩn $L_p$. Hãy giảm dần trọng số sau khi quá trình huấn luyện đã ổn định.
- **Rò rỉ chân lý cơ bản (ground-truth leakage) trong D.** Phải nối cặp $(x, y)$ làm đầu vào cho D, chứ không chỉ đưa mỗi $y$. Nếu không có $x$, D hoàn toàn không thể kiểm tra tính nhất quán giữa đầu vào và đầu ra.
- **Sụp đổ mốt trên từng lớp.** Từng lớp có thể bị sụp đổ mốt một cách độc lập. Hãy triển khai các bước kiểm tra độ đa dạng có điều kiện theo lớp (class-conditional diversity checks).

## Ứng dụng thực tế

Bức tranh công nghệ chuyển đổi ảnh-sang-ảnh năm 2026:

| Tác vụ | Hướng tiếp cận tối ưu |
|--------|----------------------|
| Phác thảo $\to$ ảnh thực, cùng miền, có dữ liệu ghép cặp | Pix2Pix / Pix2PixHD (vẫn rất nhanh, vẫn rất sắc nét) |
| Phác thảo $\to$ ảnh thực, không ghép cặp | ControlNet với mô hình điều kiện nét vẽ (Scribble) |
| Phân vùng ngữ nghĩa $\to$ ảnh thực | SPADE / GauGAN2 hoặc Stable Diffusion + ControlNet-Seg |
| Chuyển phong cách (Style transfer) | Mô hình khuếch tán với IP-Adapter hoặc LoRA; các phương pháp GAN đã lỗi thời |
| Bản đồ độ sâu $\to$ ảnh thực | ControlNet-Depth xây dựng trên Stable Diffusion |
| Siêu phân giải (Super-resolution) | Real-ESRGAN (GAN), ESRGAN-Plus, hoặc SD-Upscale (khuếch tán) |
| Tô màu ảnh (Colorization) | ColTran, các bộ tô màu dựa trên khuếch tán, hoặc Pix2Pix-color |
| Ban ngày $\to$ ban đêm, các mùa, thời tiết | CycleGAN hoặc các mô hình dựa trên ControlNet |

Pix2Pix vẫn là công cụ chuẩn xác khi: (a) bạn có hàng nghìn ví dụ ghép cặp sẵn, (b) tác vụ có phạm vi hẹp và lặp đi lặp lại, và (c) bạn đòi hỏi tốc độ suy luận cực nhanh. Đối với các tác vụ tổng quát trên miền mở, mô hình khuếch tán chiếm ưu thế hoàn toàn.

## Đóng gói sản phẩm

Lưu lại dưới dạng `outputs/skill-img2img-chooser.md`. Kỹ năng này nhận vào mô tả tác vụ, tình trạng dữ liệu sẵn có (ghép cặp hay không ghép cặp, số lượng mẫu $N$), ngân sách về độ trễ/chất lượng, sau đó xuất ra: hướng tiếp cận (Pix2Pix, CycleGAN, biến thể ControlNet, SDXL + IP-Adapter), yêu cầu về dữ liệu huấn luyện, chi phí suy luận, và quy trình đánh giá (LPIPS, FID, chỉ số chuyên biệt cho tác vụ).

## Bài tập rèn luyện

1. **Dễ.** Chỉnh sửa `code/main.py` để bổ sung thêm lớp thứ ba. Xác nhận rằng G vẫn ánh xạ chính xác vector nhiễu của từng lớp tới mốt phân phối tương ứng.
2. **Trung bình.** Thay thế hàm mất mát $L_1$ bằng một hàm mất mát dạng cảm nhận (perceptual-style loss) trong thiết lập 1 chiều (ví dụ: dùng một mạng D nhỏ được đóng băng trọng số làm bộ trích xuất đặc trưng). Sự thay đổi này có tác động gì đến độ sắc nét của phân phối có điều kiện không?
3. **Khó.** Phác thảo một mô hình CycleGAN trong thiết lập 1 chiều: hai phân phối, hai bộ sinh, hàm mất mát chu trình. Chứng minh rằng mô hình học được phép ánh xạ qua lại giữa hai phân phối mà không cần bất kỳ dữ liệu ghép cặp nào.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực chất |
|-----------|------------------------|-------------------|
| GAN có điều kiện (Conditional GAN) | "GAN kèm nhãn" | $G(z, c)$, $D(x, c)$. Cả hai mạng đều quan sát được điều kiện. |
| Pix2Pix | "GAN chuyển đổi ảnh-sang-ảnh" | cGAN ghép cặp với bộ sinh U-Net và bộ phân biệt PatchGAN + mất mát $L_1$. |
| U-Net | "Mã hóa - giải mã có kết nối tắt" | Mạng tích chập đối xứng; kết nối tắt giúp bảo toàn chi tiết tần số cao. |
| PatchGAN | "Bộ phân loại độ chân thực cục bộ" | D xuất ra điểm số trên từng mảng ảnh thay vì một điểm số toàn cục duy nhất. |
| CycleGAN | "Chuyển đổi ảnh không cần ghép cặp" | Hai bộ sinh G + hàm mất mát nhất quán chu trình; không đòi hỏi dữ liệu ghép cặp. |
| SPADE | "GauGAN" | Chuẩn hóa các kích hoạt trung gian bằng bản đồ ngữ nghĩa; dùng cho tạo ảnh từ phân vùng. |
| FiLM | "Điều chế tuyến tính theo đặc trưng" | Phép biến đổi afin trên từng đặc trưng dựa theo điều kiện; cơ chế điều kiện hóa chi phí thấp. |

## Lưu ý triển khai sản xuất: Pix2Pix là đường cơ sở bị giới hạn bởi độ trễ

Khi bạn sở hữu dữ liệu ghép cặp và bài toán có phạm vi hẹp (phác thảo $\to$ kết xuất đồ họa, bản đồ ngữ nghĩa $\to$ ảnh chụp, ngày $\to$ đêm), cơ chế suy luận một bước của Pix2Pix đánh bại mô hình khuếch tán tới cả một bậc độ lớn về mặt độ trễ. Bảng so sánh thực tế trong sản xuất:

| Phương pháp | Số bước | Độ trễ điển hình ở 512² trên 1 GPU L4 |
|-------------|---------|---------------------------------------|
| Pix2Pix (truyền xuôi qua U-Net) | 1 | ~30 ms |
| SD-Inpaint hoặc SD-Img2Img | 20 | ~1,2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0,15-0,35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix chiến thắng áp đảo về thông lượng (throughput) khi xử lý theo batch tĩnh (mọi yêu cầu đều tiêu tốn cùng một lượng FLOPs). Mô hình khuếch tán vượt trội về chất lượng và khả năng khái quát hóa. Chiến lược hiện đại phổ biến là triển khai một mô hình chưng cất dạng Pix2Pix cho tác vụ hẹp cốt lõi và giữ một mô hình khuếch tán làm phương án dự phòng cho các trường hợp biên khó xử lý (tail inputs).

## Tài liệu đọc thêm

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — bài báo gốc về cGAN.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — bộ phân biệt phép chiếu.
