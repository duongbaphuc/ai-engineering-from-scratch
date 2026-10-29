# ControlNet, LoRA & Điều kiện hóa (ControlNet, LoRA & Conditioning)

> Chỉ riêng văn bản là một tín hiệu điều khiển vụng về và thiếu chính xác. ControlNet cho phép bạn nhân bản một mô hình khuếch tán đã huấn luyện trước và định hướng bố cục của nó bằng bản đồ độ sâu (depth map), khung xương tư thế (pose skeleton), nét vẽ phác (scribble), hoặc ảnh đường biên (edge image). LoRA cho phép bạn tinh chỉnh một mô hình 2 tỷ tham số bằng cách chỉ huấn luyện 10 triệu tham số. Kết hợp lại, chúng biến Stable Diffusion từ một công cụ thử nghiệm thành quy trình xử lý hình ảnh thực tế năm 2026 được triển khai tại mọi đơn vị sáng tạo nội dung.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Yêu cầu tiên quyết:** Phase 8 · 07 (Khuếch tán trong không gian tiềm ẩn - Latent Diffusion), Phase 10 (Xây dựng LLM từ đầu — nền tảng LoRA)  
**Thời lượng:** ~75 phút  

## Vấn đề cần giải quyết (The Problem)

Một câu lệnh như "một người phụ nữ mặc váy đỏ dắt chó đi dạo trên con phố đông đúc" hoàn toàn không cung cấp cho mô hình thông tin về *vị trí* của chú chó, *tư thế cụ thể* của người phụ nữ, hay *góc phối cảnh* của con đường. Văn bản chỉ quyết định được khoảng 10% thông số cần thiết để đặc tả một bức ảnh hoàn chỉnh. Phần còn lại mang tính thị giác thuần túy và không thể diễn đạt hiệu quả bằng ngôn từ.

Huấn luyện một mô hình có điều kiện mới từ đầu cho từng loại tín hiệu đầu vào (tư thế, độ sâu, đường biên canny, phân vùng ngữ nghĩa) là điều bất khả thi về mặt chi phí. Mục tiêu lý tưởng là giữ đóng băng toàn bộ khung xương SDXL 2.6B tham số, gắn thêm một mạng nhánh nhỏ để đọc tín hiệu điều kiện, và để mạng nhánh này tác động vi chỉnh lên các đặc trưng trung gian của khung xương chính. Đó chính là ControlNet.

Mặt khác, bạn cũng muốn dạy cho mô hình những khái niệm mới (gương mặt cá nhân, sản phẩm doanh nghiệp, phong cách nghệ thuật riêng) mà không cần huấn luyện lại toàn bộ mô hình. Bạn cần một lượng tham số điều chỉnh nhỏ hơn 100 lần. Đó chính là LoRA (Low-Rank Adaptation) — các bộ điều hợp hạng thấp cắm trực tiếp vào các trọng số chú ý hiện có.

ControlNet + LoRA + văn bản = bộ công cụ chuẩn mực của kỹ sư AI tạo sinh năm 2026. Hầu hết các pipeline sản xuất hình ảnh hiện nay đều xếp chồng từ 2-5 bộ LoRA, 1-3 mô hình ControlNet, và một IP-Adapter lên trên nền tảng SDXL / SD3 / Flux.

## Khái niệm cốt lõi (The Concept)

![ControlNet nhân bản bộ mã hóa; LoRA bổ sung gia số hạng thấp](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Khởi đầu từ một mô hình Stable Diffusion đã được tiền huấn luyện. *Nhân bản* (clone) nửa phần bộ mã hóa (encoder) của mạng U-Net. Đóng băng các trọng số ban đầu. Huấn luyện bản sao đó để tiếp nhận thêm một tín hiệu điều kiện đầu vào (đường biên, độ sâu, dáng điệu). Kết nối bản sao này trở lại nửa phần bộ giải mã (decoder) của U-Net gốc thông qua các kết nối tắt *tích chập zero* (zero-convolution skip connections — các lớp tích chập 1×1 được khởi tạo toàn số 0, ban đầu hoạt động như hàm đồng nhất (no-op), sau đó học dần độ lệch delta):

```text
U-Net decoder của SD: ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Khởi tạo tích chập zero đảm bảo ControlNet khởi đầu với trạng thái đồng nhất hoàn hảo — không làm suy giảm chất lượng ảnh của mô hình gốc ngay cả trước khi bắt đầu huấn luyện. Mô hình được huấn luyện trên 1 triệu bộ ba dữ liệu (câu lệnh, điều kiện, ảnh gốc) với hàm mất mát khuếch tán chuẩn tắc.

Mỗi ControlNet theo từng phương thức được đóng gói dưới dạng các mô hình nhánh gọn nhẹ (~360MB cho SDXL, ~70M tham số cho SD 1.5). Bạn có thể kết hợp nhiều ControlNet cùng lúc khi suy luận:

```text
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Đối với bất kỳ tầng tuyến tính $W \in \mathbb{R}^{d \times d}$ nào trong mô hình, ta đóng băng $W$ và cộng thêm một ma trận gia số hạng thấp (low-rank delta):

```math
W' = W + \Delta W, \quad \Delta W = B A, \quad A \in \mathbb{R}^{r \times d}, \quad B \in \mathbb{R}^{d \times r}
```

với hạng ma trận $r \ll d$. Thông thường hạng $r = 4\text{-}16$ là chuẩn cho các tầng chú ý, $r = 64\text{-}128$ dùng cho các tác vụ tinh chỉnh sâu. Số lượng tham số mới cần học chỉ là $2 \cdot d \cdot r$ thay vì $d^2$. Đối với tầng chú ý của SDXL có kích thước $d=640$ và $r=16$: chỉ mất khoảng 20 nghìn tham số cho mỗi adapter thay vì 410 nghìn tham số gốc — giảm thiểu 20 lần. Trên quy mô toàn mô hình, một tệp trọng số LoRA thường chỉ chiếm 20-200MB so với 5GB của mô hình cơ sở.

Khi suy luận, bạn có thể điều chỉnh tỷ lệ tác động của LoRA: $W' = W + \alpha \cdot B A$. Hệ số $\alpha = 0.5\text{-}1.5$ là khoảng tối ưu thông thường. Nhiều LoRA có thể cộng dồn tuyến tính (với lưu ý rằng chúng có thể tương tác phi tuyến với nhau trong các trường hợp phức tạp).

### IP-Adapter (Ye et al., 2023)

Một adapter siêu nhẹ tiếp nhận trực tiếp một *bức ảnh* làm điều kiện hướng dẫn (song song cùng văn bản). Mô hình dùng bộ mã hóa ảnh CLIP để trích xuất các token hình ảnh, sau đó tiêm chúng vào cơ chế chú ý chéo (cross-attention) song song với các token văn bản. Kích thước chỉ ~20MB cho mỗi mô hình cơ sở. Kỹ thuật này cho phép bạn ra lệnh: "tạo một bức ảnh theo đúng phong cách của bức ảnh tham chiếu này" mà không cần mất công huấn luyện một LoRA riêng.

## Ma trận khả năng phối hợp (Composability matrix)

| Công cụ | Yếu tố kiểm soát | Dung lượng | Trường hợp sử dụng |
|---|---|---|---|
| ControlNet | Cấu trúc không gian (dáng điệu, độ sâu, đường biên) | 70-360MB | Bố cục chính xác, tạo hình chi tiết |
| LoRA | Phong cách, chủ thể, khái niệm đặc thù | 20-200MB | Cá nhân hóa, định hình phong cách nghệ thuật |
| IP-Adapter | Phong cách hoặc chủ thể từ ảnh tham chiếu | 20MB | Khi không từ ngữ nào mô tả trọn vẹn diện mạo |
| Textual Inversion | Khái niệm đơn lẻ dưới dạng một token mới | 10KB | Kỹ thuật cũ, phần lớn đã được thay thế bởi LoRA |
| DreamBooth | Tinh chỉnh toàn bộ mô hình theo chủ thể | 2-5GB | Nhận diện định danh mạnh mẽ, tốn nhiều compute |
| T2I-Adapter | Phương án thay thế nhẹ hơn cho ControlNet | 70MB | Thiết bị biên, ngân sách suy luận hạn chế |

ControlNet kiểm soát không gian. LoRA định hình ngữ nghĩa. Hãy phối hợp cả hai.

````figure
v4-controlnet-zero
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` mô phỏng hai cơ chế này trên không gian 1 chiều:

1. **LoRA.** Một tầng tuyến tính đã tiền huấn luyện $W$. Đóng băng nó. Huấn luyện tích hạng thấp $BA$ sao cho $W + BA$ khớp với một tầng tuyến tính mục tiêu. Chứng minh rằng hạng $r = 1$ là đủ để học hoàn hảo một phép biến đổi hiệu chỉnh có hạng bằng 1.
2. **ControlNet-lite.** Một mạng dự đoán cơ sở bị "đóng băng" kết hợp với một "mạng nhánh" tiếp nhận thêm tín hiệu bổ trợ. Đầu ra của mạng nhánh được điều tiết bởi một cổng vô hướng có thể học được và khởi tạo bằng 0 (phiên bản thu nhỏ của tích chập zero). Huấn luyện và quan sát trọng số cổng tăng dần theo thời gian.

### Bước 1: Công thức toán của LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W bi dong bang; A va B la cac ma tran he so hang thap co the huan luyen
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Bước 2: Mạng nhánh khởi tạo bằng zero

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate duoc khoi tao bang 0
h = base(x) + gated
```

Tại bước 0, đầu ra hoàn toàn đồng nhất với mạng cơ sở. Ở giai đoạn đầu huấn luyện, tham số `gate` cập nhật từ từ — không gây hiện tượng trôi lệch tham số thảm khốc (catastrophic drift).

## Cạm bẫy thường gặp (Pitfalls)

- **Lạm dụng tỷ lệ phóng đại LoRA.** Đặt $\alpha = 2$ hoặc $\alpha = 3$ là mẹo thường bị lạm dụng để "tăng độ nét" nhưng thực tế lại gây méo mó, biến dạng hoặc phá vỡ cấu trúc ảnh. Nên giữ $\alpha \le 1.5$.
- **Xung đột trọng số giữa các ControlNet.** Sử dụng Pose ControlNet ở trọng số 1.0 đồng thời với Depth ControlNet ở trọng số 1.0 thường khiến mô hình bị quá tải tín hiệu ràng buộc. Quy tắc an toàn là tổng các trọng số xấp xỉ $\approx 1.0$.
- **Gắn nhầm LoRA vào mô hình cơ sở khác loại.** LoRA của SDXL sẽ không hoạt động trên SD 1.5 vì số chiều tầng chú ý không khớp nhau. Thư viện Diffusers từ bản 0.30+ sẽ phát cảnh báo lỗi.
- **Hiện tượng trôi token trong Textual Inversion.** Các token học trên một checkpoint cụ thể sẽ bị trôi dạt ngữ nghĩa rất nặng khi chuyển sang checkpoint khác. LoRA mang tính linh hoạt và khả chuyển cao hơn nhiều.
- **Hợp nhất trọng số LoRA và lưu trữ.** Bạn có thể nung trực tiếp LoRA vào trọng số mô hình gốc để tăng tốc độ suy luận (không tốn phép cộng lúc runtime), nhưng sẽ mất khả năng điều chỉnh linh hoạt hệ số $\alpha$ khi đang chạy. Hãy lưu giữ cả hai phiên bản.

## Ứng dụng thực tế (Use It)

| Mục tiêu | Quy trình chuẩn năm 2026 |
|---|---|
| Tái hiện phong cách thương hiệu | LoRA huấn luyện trên ~30 ảnh chọn lọc ở rank 32 |
| Đưa khuôn mặt cá nhân vào ảnh tạo sinh | DreamBooth hoặc LoRA kết hợp IP-Adapter-FaceID |
| Tư thế dáng người cụ thể + prompt | ControlNet-Openpose + SDXL + văn bản |
| Bố cục kiểm soát chiều sâu không gian | ControlNet-Depth + SD3 |
| Ảnh tham chiếu + văn bản chỉ dẫn | IP-Adapter + văn bản |
| Bố cục đường nét chính xác tuyệt đối | ControlNet-Scribble hoặc ControlNet-Canny |
| Thay đổi phông nền hậu cảnh | ControlNet-Seg + Inpainting (Bài 09) |
| Tạo phong cách siêu tốc 1 bước | LCM-LoRA trên SDXL-Turbo |

## Đóng gói sản phẩm (Ship It)

Lưu cấu hình tại `outputs/skill-sd-toolkit-composer.md`. Kỹ năng này nhận đầu vào là một yêu cầu tạo tác (tài nguyên đầu vào: prompt, ảnh tham chiếu tùy chọn, dáng người tùy chọn, bản đồ độ sâu tùy chọn, nét vẽ tùy chọn) và thiết lập ngăn xếp công cụ, trọng số chi tiết, cùng quy trình quản lý seed có khả năng tái lập kết quả chính xác.

## Bài tập thực hành (Exercises)

1. **Dễ.** Trong `code/main.py`, thay đổi hạng `r` của LoRA từ 1 đến 4. Tại hạng nào thì LoRA khớp hoàn hảo với ma trận gia số mục tiêu có hạng bằng 2?
2. **Trung bình.** Huấn luyện hai LoRA độc lập trên hai phép biến đổi mục tiêu khác nhau. Nạp cả hai cùng lúc và quan sát sự tương tác cộng dồn của chúng. Khi nào thì sự tương tác này phá vỡ tính tuyến tính?
3. **Khó.** Sử dụng thư viện diffusers để xếp chồng: SDXL-base + Canny-ControlNet (trọng số 0.8) + style LoRA ($\alpha$ 0.8) + IP-Adapter (trọng số 0.6). Đo lường sự đánh đổi giữa chỉ số FID và độ bám sát câu lệnh khi điều chỉnh trọng số của các thành phần trong ngăn xếp.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|---|---|---|
| ControlNet | "Điều khiển không gian" | Bộ mã hóa nhân bản + kết nối zero-conv; tiếp nhận ảnh điều kiện. |
| Zero convolution | "Khởi tạo đồng nhất" | Lớp tích chập 1×1 khởi tạo bằng 0; ControlNet khởi đầu như hàm no-op. |
| LoRA | "Bộ điều hợp hạng thấp" | $W + BA$, $r \ll d$; số lượng tham số ít hơn 100 lần so với tinh chỉnh toàn bộ. |
| rank r | "Hạng ma trận" | Mức độ nén của LoRA; thông thường 4-16, từ 64 trở lên cho cá nhân hóa sâu. |
| $\alpha$ | "Độ mạnh của LoRA" | Hệ số tỷ lệ khi suy luận để điều chỉnh tác động của gia số LoRA. |
| IP-Adapter | "Ảnh tham chiếu phong cách" | Adapter điều kiện hóa bằng hình ảnh qua các token ảnh trích xuất từ CLIP. |
| DreamBooth | "Tinh chỉnh chủ thể toàn diện" | Huấn luyện toàn bộ mô hình trên khoảng ~30 bức ảnh của một chủ thể cụ thể. |
| Textual Inversion | "Token mới" | Chỉ tối ưu hóa một vector nhúng từ ngữ mới; phương pháp cũ, nay ít dùng. |

## Lưu ý trong môi trường sản xuất: Hoán đổi LoRA, làn xử lý ControlNet và phục vụ đa người thuê (Multi-tenant)

Một dịch vụ SaaS tạo ảnh từ văn bản trong thực tế cần phục vụ hàng trăm adapter LoRA và hàng chục mô hình ControlNet trên cùng một checkpoint gốc duy nhất. Bài toán phục vụ này rất giống với kiến trúc đa người thuê (multi-tenancy) trong các mô hình LLM lớn:

- **Hoán đổi nóng LoRA (Hot-swap), không gộp cứng trọng số.** Việc gộp cố định $W' = W + \alpha \cdot B \cdot A$ vào mô hình cơ sở giúp suy luận nhanh hơn khoảng ~3-5% mỗi bước nhưng lại làm đóng băng $\alpha$ và mô hình nền. Hãy duy trì các bộ LoRA thường trực trong VRAM dưới dạng các gia số hạng $r$; thư viện diffusers cung cấp cơ chế `pipe.load_lora_weights()` và `pipe.set_adapters([...], adapter_weights=[...])` để kích hoạt linh hoạt theo từng request. Chi phí hoán đổi chỉ là nạp lượng trọng số `2 · d · r · num_layers` — ở quy mô vài megabyte, diễn ra trong tích tắc dưới một giây.
- **ControlNet như một làn chú ý thứ hai.** Bộ mã hóa nhân bản chạy song song với mô hình cơ sở. Hai ControlNet cùng chạy ở trọng số 1.0 đồng nghĩa với việc bạn phải thực hiện thêm hai lượt truyền xuôi (forward pass) riêng biệt cho mỗi bước suy luận, chứ không thể gộp thành một lượt duy nhất. Biên độ kích thước batch sẽ suy giảm theo hàm phi tuyến. Hãy dự trù chi phí tính toán tăng khoảng ~1.5 lần cho mỗi ControlNet đang hoạt động.
- **Lượng tử hóa cả LoRA.** Nếu bạn đã lượng tử hóa mô hình cơ sở (xem Bài 07 về Flux trên card 8GB), thì phần gia số LoRA cũng có thể lượng tử hóa mượt mà sang 8-bit hoặc 4-bit. Kỹ thuật nạp kiểu QLoRA cho phép bạn xếp chồng từ 5-10 LoRA lên trên một mô hình Flux nền 4-bit mà không lo tràn bộ nhớ VRAM.

Đặc thù với Flux: Sổ tay hướng dẫn chạy Flux trên 8GB của Niels lượng tử hóa mô hình nền sang 4-bit; việc xếp chồng một LoRA phong cách (`pipe.load_lora_weights("user/style-lora")`) trên nền lượng tử hóa đó với tệp `weight_name="pytorch_lora_weights.safetensors"` vẫn vận hành ổn định. Đây là công thức kiến trúc mà phần lớn các agency công nghệ triển khai trong năm 2026.

## Tài liệu đọc thêm (Further Reading)

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — bài báo gốc về ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — kỹ thuật LoRA (ban đầu cho LLM, sau áp dụng hiệu quả cho mô hình khuếch tán).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — bài báo về IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — giải pháp thay thế nhẹ hơn cho ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — kỹ thuật DreamBooth.
- [Tài liệu HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter](https://huggingface.co/docs/diffusers/training/controlnet) — hướng dẫn xây dựng pipeline tham chiếu.
