# StyleGAN

> Hầu hết các bộ sinh đều trộn lẫn vector $z$ vào mọi tầng cùng một lúc. StyleGAN đã tách bạch cấu trúc này: trước tiên ánh xạ $z$ sang một không gian trung gian $w$, sau đó *đưa* $w$ vào từng tầng độ phân giải thông qua cơ chế AdaIN. Chỉ một thay đổi duy nhất đó đã gỡ rối hoàn toàn không gian tiềm ẩn, biến bài toán tạo khuôn mặt chân thực như ảnh chụp thành vấn đề đã được giải quyết trọn vẹn trong suốt bảy năm liên tiếp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (Mạng đối nghịch sinh - GANs), Phase 4 · 08 (Kỹ thuật chuẩn hóa - Normalization), Phase 3 · 07 (Mạng tích chập - CNNs)
**Time:** ~45 phút

## Vấn đề đặt ra

Mô hình DCGAN ánh xạ vector nhiễu $z$ thành một bức ảnh thông qua một chuỗi các tầng tích chập chuyển vị (transposed convolutions). Vấn đề nan giải: $z$ kiểm soát mọi thứ cùng lúc — tư thế, ánh sáng, danh tính khuôn mặt, phông nền — tất cả bị vướng víu (entangled) chặt chẽ với nhau. Khi bạn dịch chuyển dọc theo một trục của $z$, cả bốn yếu tố trên đều đồng loạt biến đổi. Bạn không thể yêu cầu mô hình "giữ nguyên người này nhưng đổi tư thế khác", bởi vì cách biểu diễn dữ liệu không được phân rã theo cấu trúc đó.

Karras và các cộng sự (2019, NVIDIA) đã đề xuất: ngừng đưa trực tiếp $z$ vào các tầng tích chập. Thay vào đó, đưa một tensor hằng số học được kích thước `4×4×512` làm đầu vào khởi đầu cho mạng. Học một mạng MLP 8 tầng để ánh xạ $z \in Z \to w \in W$. Đưa $w$ vào từng cấp độ phân giải thông qua cơ chế *chuẩn hóa cá thể thích ứng (Adaptive Instance Normalization - AdaIN)*: chuẩn hóa từng bản đồ đặc trưng (feature map) của tích chập, sau đó nhân tỷ lệ và tịnh tiến bằng các phép chiếu afin từ $w$. Bổ sung thêm nhiễu ngẫu nhiên trên từng tầng (per-layer noise) để tạo các chi tiết vi mô ngẫu nhiên (lỗ chân lông, sợi tóc).

Kết quả: không gian $W$ sở hữu các trục gần như trực giao đại diện cho "phong cách mức cao" (tư thế, danh tính) tách biệt hoàn toàn với "phong cách mức chi tiết" (ánh sáng, màu sắc). Bạn có thể hoán đổi phong cách giữa hai bức ảnh bằng cách dùng $w$ của ảnh A cho các tầng độ phân giải thấp và $w$ của ảnh B cho các tầng độ phân giải cao. Khám phá này đã mở đường cho các tác vụ chỉnh sửa ảnh, biến đổi phong cách liên miền, và toàn bộ nhánh nghiên cứu về "nghịch đảo StyleGAN (StyleGAN-inversion)".

## Khái niệm cốt lõi

![StyleGAN: mạng ánh xạ + AdaIN + nhiễu trên từng tầng](../assets/stylegan.svg)

**Mạng ánh xạ (Mapping network).** $f: Z \to W$, một mạng MLP gồm 8 tầng. $Z = \mathcal{N}(0, I)^{512}$. Không gian $W$ không bị ép buộc phải tuân theo phân phối Gaussian — nó tự do học một hình dạng hình học thích ứng tối ưu với dữ liệu.

**Mạng tổng hợp (Synthesis network).** Bắt đầu từ một tensor hằng số học được có kích thước `4×4×512`. Mỗi khối độ phân giải gồm: `nâng mẫu (upsample) → conv → AdaIN(w_i) → nhiễu → conv → AdaIN(w_i) → nhiễu`. Độ phân giải tăng gấp đôi qua từng khối: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```math
\mathop{\mathrm{AdaIN}}(x, y) = y_{\text{scale}} \cdot \frac{x - \mathop{\mathrm{mean}}(x)}{\mathop{\mathrm{std}}(x)} + y_{\text{bias}}
```

trong đó $y_{\text{scale}}$ và $y_{\text{bias}}$ được sinh ra từ các phép chiếu afin của $w$. Chuẩn hóa trên từng bản đồ đặc trưng, sau đó điều chỉnh lại phong cách. "Phong cách" ở đây chính là các số liệu thống kê bậc nhất và bậc hai (trung bình và độ lệch chuẩn) của bản đồ đặc trưng.

**Nhiễu trên từng tầng (Per-layer noise).** Nhiễu Gaussian một kênh được cộng vào từng bản đồ đặc trưng, nhân với một hệ số tỉ lệ học được trên từng kênh. Nhiễu này kiểm soát các chi tiết vi mô ngẫu nhiên mà không hề làm ảnh hưởng tới cấu trúc hình học toàn cục.

**Thủ thuật cắt cụt (Truncation trick).** Trong giai đoạn suy luận, rút mẫu $z$, tính $w = \text{mapping}(z)$, sau đó tính $w' = \hat{w} + \psi \cdot (w - \hat{w})$ với $\hat{w}$ là giá trị trung bình của $w$ qua nhiều mẫu. Hệ số $\psi < 1$ giúp đánh đổi độ đa dạng để đổi lấy chất lượng mẫu cao hơn. Hầu hết các bản demo của StyleGAN đều sử dụng $\psi \approx 0.7$.

## Quá trình tiến hóa: StyleGAN 1 → 2 → 3

| Phiên bản | Năm | Cải tiến then chốt |
|-----------|-----|--------------------|
| StyleGAN | 2019 | Mạng ánh xạ + AdaIN + nhiễu vi mô + tăng trưởng lũy tiến (progressive growing). |
| StyleGAN2 | 2020 | Giải điều chế trọng số (weight demodulation) thay thế AdaIN (khắc phục khuyết tật giọt nước - droplet artifacts); kiến trúc phần dư/kết nối tắt; điều quy hóa độ dài đường dẫn (path-length regularization). |
| StyleGAN3 | 2021 | Tích chập khử hiện tượng chồng phổ (alias-free convolution) + các kernel đồng biến; loại bỏ hoàn toàn hiện tượng kết cấu dính chặt vào lưới điểm ảnh (texture sticking). |
| StyleGAN-XL | 2022 | Có điều kiện theo lớp, độ phân giải 1024², áp dụng trên ImageNet. |
| R3GAN | 2024 | Tái thiết kế với điều quy mạnh hơn; thu hẹp khoảng cách chất lượng với mô hình khuếch tán trên FFHQ-1024 với số lượng tham số ít hơn 20 lần. |

Đến năm 2026, StyleGAN3 vẫn là lựa chọn mặc định cho: (a) bài toán sinh ảnh chân thực trên miền hẹp với tốc độ khung hình (FPS) cực cao, (b) thích ứng miền dữ liệu ít mẫu (few-shot domain adaptation: huấn luyện trên tập dữ liệu mới chỉ gồm 100 ảnh, đóng băng mạng ánh xạ), (c) chỉnh sửa ảnh dựa trên phép nghịch đảo (tìm vector $w$ tái tạo chính xác một bức ảnh thật, sau đó thao tác chỉnh sửa trên $w$ đó). Đối với bài toán tạo ảnh tổng quát từ câu lệnh văn bản trên miền mở, công cụ tối ưu không phải là StyleGAN — mà là mô hình khuếch tán.

```figure
gx-stylegan-mapping
```

## Thực hành: Xây dựng

`code/main.py` cài đặt một phiên bản thu nhỏ "style-GAN lite" trên dữ liệu 1 chiều: gồm một mạng MLP ánh xạ, một hàm tổng hợp nhận vector hằng số học được và điều chế nó bằng các hệ số tỉ lệ/độ dời trích xuất từ $w$, cùng nhiễu trên từng tầng. Ví dụ này chứng minh rằng việc đưa $w$ vào thông qua phép điều chế afin mang lại hiệu quả tương đương hoặc vượt trội so với việc nối $z$ trực tiếp vào đầu vào của bộ sinh.

### Bước 1: Mạng ánh xạ (Mapping network)

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Bước 2: Chuẩn hóa cá thể thích ứng (AdaIN)

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Tỉ lệ (scale) và độ dời (bias) của từng bản đồ đặc trưng được suy ra từ $w$ thông qua phép chiếu tuyến tính.

### Bước 3: Bổ sung nhiễu trên từng tầng

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Tham số sigma cho từng kênh là một biến có thể học được.

## Các cạm bẫy thường gặp

- **Khuyết tật giọt nước (Droplet artifacts).** StyleGAN 1 tạo ra một đốm khuyết tật dạng giọt nước trong bản đồ đặc trưng do AdaIN triệt tiêu giá trị trung bình về 0. Cơ chế giải điều chế trọng số (weight demodulation) của StyleGAN 2 đã khắc phục triệt để điều này bằng cách chuẩn hóa trực tiếp trên trọng số tích chập thay vì trên các kích hoạt.
- **Hiện tượng kết cấu bám dính (Texture sticking).** Trong StyleGAN 1 và 2, các chi tiết kết cấu bám theo tọa độ điểm ảnh thay vì tọa độ hình học của vật thể (thấy rất rõ khi nội suy chuyển động). Các tích chập khử chồng phổ (alias-free convolutions) của StyleGAN 3 đã giải quyết vấn đề này bằng các bộ lọc sinc có cửa sổ (windowed sinc filters).
- **Độ bao phủ mốt dữ liệu.** Kỹ thuật cắt cụt $\psi < 0.7$ cho ảnh rất sạch nhưng các mẫu chỉ nằm gọn trong một nón phân phối hẹp; hãy dùng $\psi = 1.0$ nếu bạn cần độ đa dạng phong phú.
- **Nghịch đảo bị tổn thất thông tin (Inversion is lossy).** Nghịch đảo một bức ảnh thực tế vào không gian $W$ thường được thực hiện qua tối ưu hóa gradient hoặc dùng mạng mã hóa (e4e, ReStyle, HyperStyle). Kết quả dễ bị trôi dạt (drift) sau nhiều bước lặp.

## Ứng dụng thực tế

| Trường hợp sử dụng | Hướng tiếp cận tối ưu |
|--------------------|----------------------|
| Khuôn mặt người chân thực (anime, sản phẩm, miền hẹp) | StyleGAN3 FFHQ / tinh chỉnh chuyên biệt |
| Chỉnh sửa khuôn mặt từ ảnh chụp thực tế | Nghịch đảo e4e + các hướng vector trong StyleSpace / InterFaceGAN |
| Hoán đổi khuôn mặt / tái diễn xuất (face swap / reenactment) | StyleGAN + bộ mã hóa + hòa trộn đặc trưng (blending) |
| Quy trình tạo avatar | StyleGAN3 kết hợp cơ chế ADA cho tinh chỉnh ít dữ liệu |
| Thích ứng miền từ một lượng ảnh rất nhỏ | Đóng băng mạng ánh xạ, chỉ tinh chỉnh mạng tổng hợp |
| Tạo ảnh đa phương thức hoặc điều kiện hóa bằng văn bản | Không dùng GAN — hãy chuyển sang mô hình khuếch tán |

Đối với các bản demo sản phẩm thương mại mà bài toán cốt lõi là "tạo ảnh chân dung khuôn mặt", StyleGAN đánh bại mô hình khuếch tán về chi phí suy luận (chỉ một lượt truyền xuôi duy nhất, <10ms trên GPU RTX 4090) và độ sắc nét trên cùng một tiêu chuẩn chất lượng.

## Đóng gói sản phẩm

Lưu lại dưới dạng `outputs/skill-stylegan-inversion.md`. Kỹ năng này nhận vào một bức ảnh chụp thực tế và xuất ra: phương pháp nghịch đảo tối ưu (e4e / ReStyle / HyperStyle), độ mất mát tiềm ẩn dự kiến, giới hạn ngân sách chỉnh sửa (bạn có thể dịch chuyển bao xa trong $W$ trước khi xuất hiện khuyết tật), và danh sách các hướng chỉnh sửa đã được kiểm chứng hoạt động tốt (tuổi tác, biểu cảm, tư thế).

## Bài tập rèn luyện

1. **Dễ.** Chạy `code/main.py` với tùy chọn `adain_on=True` và `adain_on=False`. So sánh độ phân tán của đầu ra giữa trường hợp biến tiềm ẩn cố định và biến tiềm ẩn bị nhiễu loạn nhỏ.
2. **Trung bình.** Cài đặt kỹ thuật điều quy hóa trộn lẫn (mixing regularization): đối với một batch huấn luyện, tính toán $w_a$ và $w_b$, sau đó áp dụng $w_a$ cho nửa đầu các tầng tổng hợp và $w_b$ cho nửa sau. Bộ giải mã có học được các phong cách gỡ rối độc lập hay không?
3. **Khó.** Sử dụng mô hình StyleGAN3 FFHQ đã tiền huấn luyện (`ffhq-1024.pkl`). Tìm hướng vector trong không gian $w$ điều khiển đặc trưng "nụ cười" bằng cách huấn luyện một bộ phân loại SVM trên các mẫu đã gán nhãn; báo cáo khoảng cách tối đa có thể dịch chuyển trước khi danh tính khuôn mặt bị trôi dạt.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực chất |
|-----------|------------------------|-------------------|
| Mạng ánh xạ (Mapping network) | "Mạng MLP" | $f: Z \to W$, 8 tầng, tách rời hình học tiềm ẩn khỏi số liệu thống kê dữ liệu. |
| Không gian W (W space) | "Không gian phong cách" | Đầu ra của mạng ánh xạ; các đặc trưng gần như được gỡ rối hoàn toàn. |
| AdaIN | "Chuẩn hóa cá thể thích ứng" | Chuẩn hóa bản đồ đặc trưng, sau đó nhân tỉ lệ và dịch chuyển bằng phép chiếu $w$. |
| Thủ thuật cắt cụt (Truncation trick) | "Hệ số Psi" | $w = \text{mean} + \psi \cdot (w - \text{mean})$, $\psi<1$ đánh đổi độ đa dạng lấy chất lượng. |
| Điều quy hóa độ dài đường dẫn (Path-length reg) | "PL reg" | Phạt sự thay đổi đột ngột của ảnh trên một đơn vị biến thiên trong $w$; làm cho $W$ trơn tru hơn. |
| Giải điều chế trọng số (Weight demodulation) | "Cách sửa lỗi của StyleGAN2" | Chuẩn hóa trực tiếp trọng số tích chập thay vì kích hoạt; loại bỏ khuyết tật giọt nước. |
| Khử chồng phổ (Alias-free) | "Mẹo của StyleGAN3" | Dùng các bộ lọc sinc có cửa sổ; loại bỏ hiện tượng kết cấu bám dính vào lưới điểm ảnh. |
| Nghịch đảo (Inversion) | "Tìm w cho ảnh thật" | Tối ưu hóa hoặc dùng mạng mã hóa ánh xạ $x \to w$ sao cho $G(w) \approx x$. |

## Lưu ý triển khai sản xuất: vì sao StyleGAN vẫn được đưa vào sản xuất năm 2026

StyleGAN3 chạy trên GPU RTX 4090 có thể tạo ra một khuôn mặt FFHQ 1024² chỉ trong chưa đầy 10 ms — `num_steps = 1`, không cần giải mã VAE, không có lượt tính chú ý chéo. Trong ngôn ngữ triển khai sản xuất, đây chính là mức sàn độ trễ đối với bất kỳ mô hình tạo ảnh nào. Một quy trình 50 bước SDXL kèm giải mã VAE ở cùng độ phân giải mất khoảng ~3 giây. Đó là một **khoảng cách chênh lệch lên tới 300 lần**, và đối với các sản phẩm trong miền hẹp (dịch vụ avatar, quy trình xử lý giấy tờ tùy thân, tạo ảnh khuôn mặt kho mẫu) StyleGAN hoàn toàn chiến thắng về tổng chi phí sở hữu (TCO).

Hai hệ quả vận hành thực tế:

- **Không cần bộ điều phối lịch trình, không cần bộ gom batch phức tạp.** Một batch tĩnh đạt đúng mức chiếm dụng mục tiêu là phương án tối ưu. Gom cụm liên tục (cực kỳ thiết yếu cho LLM và mô hình khuếch tán) hoàn toàn không mang lại lợi ích gì ở đây, bởi vì mọi yêu cầu đều tiêu tốn cùng một lượng FLOPs cố định.
- **Hệ số cắt cụt `ψ` chính là núm điều chỉnh độ an toàn.** `ψ < 0.7` lấy mẫu trong một nón hẹp từ phạm vi của mạng ánh xạ. Đây là công cụ duy nhất mà tầng phục vụ (serving layer) có thể can thiệp để kiểm soát phương sai của mẫu sinh ra. Hãy hạ thấp `ψ` vào giờ cao điểm để đảm bảo tính ổn định, và nâng cao đối với người dùng trả phí cao cấp cần sự sáng tạo đa dạng.

## Tài liệu đọc thêm

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — phương pháp nghịch đảo e4e.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — công thức kiến trúc GAN hiện đại tối giản.
