# Vẽ đắp, Mở rộng biên & Chỉnh sửa hình ảnh (Inpainting, Outpainting & Image Editing)

> Tạo ảnh từ văn bản giúp tạo ra những thứ mới mẻ. Vẽ đắp (inpainting) giúp sửa chữa và hoàn thiện những thứ đã có. Trong môi trường sản xuất thực tế, 70% khối lượng công việc hình ảnh đem lại doanh thu là các tác vụ chỉnh sửa — thay đổi phông nền, xóa logo, mở rộng khung hình, vẽ lại bàn tay hỏng. Inpainting chính là nơi công nghệ khuếch tán chứng tỏ trọn vẹn giá trị thương mại thực tế.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Yêu cầu tiên quyết:** Phase 8 · 07 (Khuếch tán trong không gian tiềm ẩn - Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)  
**Thời lượng:** ~75 phút  

## Vấn đề cần giải quyết (The Problem)

Một khách hàng gửi tới bức ảnh sản phẩm hoàn hảo nhưng lại có một tấm biển gây mất tập trung ở hậu cảnh. Bạn muốn xóa bỏ tấm biển đó và giữ cho mọi chi tiết khác nguyên vẹn từng điểm ảnh (pixel-identical). Bạn không thể tạo lại ảnh từ đầu bằng câu lệnh văn bản — kết quả sinh ra sẽ bị lệch màu, khác góc chiếu sáng, hoặc sai lệch góc chụp của sản phẩm. Bạn chỉ muốn tái tạo *duy nhất* vùng được đánh dấu bằng mặt nạ (mask), và vùng tái tạo đó phải hòa nhập hoàn hảo với ngữ cảnh xung quanh.

Đó chính là bài toán inpainting (vẽ đắp nội suy). Các biến thể mở rộng bao gồm:

- **Inpainting (Vẽ đắp):** Tái tạo nội dung bên trong vùng mặt nạ, giữ nguyên tuyệt đối các pixel nằm ngoài mặt nạ.
- **Outpainting (Mở rộng biên):** Tái tạo nội dung bên ngoài vùng mặt nạ (hoặc vượt ra ngoài khung vẽ ban đầu), bảo toàn toàn bộ nội dung bên trong.
- **Image Editing (Chỉnh sửa ảnh toàn cục):** Tái tạo lại toàn bộ bức ảnh nhưng vẫn giữ vững độ trung thực về mặt ngữ nghĩa hoặc cấu trúc đối với ảnh gốc (chẳng hạn như SDEdit, InstructPix2Pix).

Mọi quy trình khuếch tán trong năm 2026 đều tích hợp sẵn chế độ inpainting: Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Tất cả đều vận hành trên các nguyên lý cốt lõi tương đồng.

## Khái niệm cốt lõi (The Concept)

![Inpainting: khử nhiễu nhận biết mặt nạ và tái tiêm ngữ cảnh bảo toàn vùng ngoài](../assets/inpainting.svg)

### Cách tiếp cận ngây thơ (và lý do thất bại)

Chạy pipeline tạo ảnh từ văn bản thông thường kết hợp với mặt nạ. Tại mỗi bước lấy mẫu, thay thế vùng không bị che của tensor tiềm ẩn chứa nhiễu bằng ảnh gốc sạch đã được thêm nhiễu xuôi tương ứng. Cách này có hoạt động... nhưng chất lượng rất tệ. Hiện tượng đứt gãy đường biên (boundary artifacts) xuất hiện rõ rệt vì mạng nơ-ron không hề có thông tin ngữ cảnh về nội dung thực sự đang diễn ra xung quanh vùng bị che.

### Mô hình inpainting chuyên dụng chuẩn mực

Huấn luyện một mạng U-Net tùy biến nhận 9 kênh đầu vào thay vì 4 kênh tiêu chuẩn:

```text
input = concat([ noisy_latent (4 kênh), encoded_image (4 kênh), mask (1 kênh) ], dim=channel)
```

Các kênh bổ sung bao gồm bản sao ảnh gốc đã được mã hóa qua VAE kết hợp cùng mặt nạ nhị phân 1 kênh. Khi huấn luyện, bạn che ngẫu nhiên các vùng ảnh và huấn luyện mô hình chỉ khử nhiễu cho vùng bị che, trong khi vùng không bị che được cung cấp như một tín hiệu điều kiện hoàn chỉnh. Khi suy luận, mô hình có thể "nhìn thấy" chính xác những gì bao quanh vùng khuyết và tạo ra phần bù đắp liền mạch, tự nhiên.

SD-Inpaint, SDXL-Inpaint và Flux-Fill đều áp dụng kiến trúc đầu vào 9 kênh này (hoặc cơ chế tương đương). Trong diffusers, đó là các lớp `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — Chỉnh sửa ảnh không cần huấn luyện lại

Thêm nhiễu vào ảnh gốc cho tới một bước thời gian trung gian `t`, sau đó chạy chuỗi khử nhiễu ngược từ `t` lùi về 0 cùng một câu lệnh văn bản mới. Hoàn toàn không cần huấn luyện lại mô hình. Việc lựa chọn mốc thời gian `t` ban đầu là sự đánh đổi trực tiếp giữa độ trung thực với ảnh gốc và sự tự do sáng tạo mới:

- `t/T = 0.3` → Gần như đồng nhất với ảnh gốc, chỉ thay đổi nhẹ về phong cách hoặc chi tiết bề mặt.
- `t/T = 0.6` → Chỉnh sửa mức độ vừa phải, giữ nguyên cấu trúc không gian thô.
- `t/T = 0.9` → Tạo sinh lại gần như từ nhiễu thuần túy, tính bảo toàn ảnh gốc rất thấp.

### InstructPix2Pix (Brooks et al., 2023)

Tinh chỉnh mô hình khuếch tán trên các bộ ba dữ liệu `(ảnh_gốc, câu_lệnh_chỉ_dẫn, ảnh_kết_quả)`. Khi suy luận, mô hình nhận điều kiện hóa từ cả ảnh đầu vào lẫn câu lệnh văn bản ngắn ("đổi thành hoàng hôn", "thêm một con rồng"). Mô hình sử dụng hai hệ số hướng dẫn CFG độc lập: tỷ lệ cho ảnh và tỷ lệ cho văn bản.

### RePaint (Lugmayr et al., 2022)

Sử dụng trực tiếp mô hình khuếch tán thông thường mà không cần huấn luyện lại chuyên biệt. Tại mỗi bước khử nhiễu ngược, thuật toán thực hiện bước nhảy lùi ngẫu nhiên — thỉnh thoảng quay trở lại trạng thái nhiều nhiễu hơn rồi khử nhiễu lại từ đầu. Kỹ thuật này triệt tiêu hiện tượng đứt gãy đường viền rất hiệu quả khi bạn không có sẵn một checkpoint inpainting chuyên dụng.

````figure
inpaint-mask-reinject
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` cài đặt cơ chế inpainting 1 chiều mô phỏng trên dữ liệu 5 chiều. Chúng ta huấn luyện một mạng DDPM trên tập dữ liệu phân phối hỗn hợp 5 chiều, trong đó mỗi mẫu gồm 5 số thực thuộc về một trong hai cụm (cluster). Khi suy luận, ta "che" 2 trong 5 chiều, tiêm nhiễu xuôi tương ứng vào 3 chiều không bị che tại mỗi bước thời gian, và tái tạo duy nhất 2 chiều bị che.

### Bước 1: Dữ liệu DDPM 5 chiều

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Bước 2: Huấn luyện bộ khử nhiễu trên toàn bộ 5 chiều

Thuật toán DDPM tiêu chuẩn. Mạng nhận đầu vào là vector nhiễu 5 chiều và dự đoán vector nhiễu tương ứng 5 chiều.

### Bước 3: Lấy mẫu ngược nhận biết mặt nạ khi suy luận

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # thay the cac chieu khong bi che bang phien ban nhieu tuong ung cua anh sach
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...sau do thuc hien buoc khu nhieu nguoc tieu chuan tren x_t
```

Đây là cách tiếp cận thay thế trực tiếp và nó hoạt động tốt trên dữ liệu mô phỏng 1 chiều. Inpainting trên hình ảnh thực tế bắt buộc phải dùng đầu vào 9 kênh chuyên biệt vì tính liên kết kết cấu thị giác đòi hỏi độ chính xác cao hơn nhiều.

### Bước 4: Mở rộng biên (Outpainting)

Outpainting về bản chất là inpainting với mặt nạ bị đảo ngược: che đi phần khung tranh mới mở rộng (vùng trước đây chưa từng tồn tại), giữ nguyên phần diện tích ảnh gốc bên trong. Mục tiêu huấn luyện và thuật toán giải mã hoàn toàn giống nhau.

## Cạm bẫy thường gặp (Pitfalls)

- **Vết nối đường biên (Seams).** Cách tiếp cận ngây thơ thường để lại các vết nứt hoặc đường viền lộ rõ vì luồng thông tin gradient không được liên kết tự nhiên xuyên qua ranh giới mặt nạ. Giải pháp: làm giãn (dilate) mặt nạ rộng ra thêm 8-16 pixel, hoặc sử dụng mô hình inpainting chuyên dụng 9 kênh.
- **Rò rỉ nhiễu từ mặt nạ.** Nếu vùng ảnh gốc làm điều kiện có chất lượng kém hoặc vỡ hạt, nó sẽ làm hỏng kết quả sinh ra bên trong mặt nạ. Cần khử nhiễu hoặc làm mịn nhẹ ảnh gốc trước.
- **Tương tác tiêu cực giữa CFG và diện tích mặt nạ.** Đặt CFG quá cao trên một vùng mặt nạ nhỏ sẽ làm mảng vá bị cháy sáng và bão hòa gắt màu. Cần giảm CFG khi xử lý các vùng chỉnh sửa nhỏ.
- **Ngưỡng sụp đổ trung thực trong SDEdit.** Tăng tỷ lệ từ `t/T = 0.5` lên `t/T = 0.6` có thể làm mất hoàn toàn nhận diện danh tính của chủ thể gốc. Cần quét thử nghiệm nhiều mốc $t$ và lưu checkpoint cẩn thận.
- **Lệch ngữ cảnh trong câu lệnh.** Câu lệnh văn bản phải mô tả *toàn bộ* khung cảnh bức ảnh, không chỉ mô tả riêng phần nội dung mới được thêm vào. Ví dụ nên viết "một con mèo ngồi trên chiếc ghế gỗ" thay vì chỉ viết "con mèo".

## Ứng dụng thực tế (Use It)

| Tác vụ | Đường ống đề xuất (Pipeline) |
|---|---|
| Xóa vật thể thừa, mặt nạ nhỏ | SD-Inpaint hoặc Flux-Fill, prompt thông thường |
| Thay thế bầu trời | SD-Inpaint + câu lệnh "bầu trời xanh lúc hoàng hôn" |
| Mở rộng khung hình (Outpainting) | SDXL chế độ outpaint (làm mờ biên 8px) hoặc Flux-Fill với mặt nạ outpaint |
| Tái tạo bàn tay / khuôn mặt | SD-Inpaint với câu lệnh tái mô tả chủ thể + ControlNet-Openpose |
| Đổi phong cách một vùng chọn | SDEdit tại `t/T=0.5` trên vùng có mặt nạ |
| "Đổi cảnh sang hoàng hôn" | InstructPix2Pix hoặc Flux-Kontext |
| Tách và thay thế phông nền | Mặt nạ SAM → SD-Inpaint |
| Chỉnh sửa siêu trung thực, độ khó cao | Dịch vụ Flux-Fill hoặc GPT-Image |

Sự kết hợp giữa SAM (Segment Anything của Meta, 2023) + inpainting khuếch tán chính là quy trình tách và thay phông nền chuẩn mực nhất năm 2026. Phiên bản SAM 2 (2024) hỗ trợ trực tiếp trên cả video.

## Đóng gói sản phẩm (Ship It)

Lưu cấu hình tại `outputs/skill-editing-pipeline.md`. Kỹ năng này nhận đầu vào là ảnh gốc + mô tả yêu cầu chỉnh sửa + mặt nạ tùy chọn (hoặc câu lệnh cho SAM) và xuất ra: phương pháp tạo mặt nạ tối ưu, mô hình nền tảng, các tỷ lệ CFG (cho ảnh và chữ), cấu hình tham số thời gian cho SDEdit hoặc chế độ inpainting, cùng danh mục kiểm tra chất lượng đầu ra.

## Bài tập thực hành (Exercises)

1. **Dễ.** Trong `code/main.py`, thay đổi tỷ lệ số chiều bị che từ 0.2 lên 0.8. Tại tỷ lệ nào thì chất lượng inpainting (sai số thặng dư trên các chiều bị che) tương đương với việc tạo mẫu ngẫu nhiên không điều kiện?
2. **Trung bình.** Cài đặt thuật toán RePaint: cứ sau 10 bước khử nhiễu ngược, nhảy lùi lại 5 bước (thêm nhiễu trở lại) rồi khử nhiễu lại. Đo đạc xem kỹ thuật này có giúp giảm sai số ranh giới tại mép mặt nạ hay không.
3. **Khó.** Sử dụng Hugging Face diffusers để so sánh: SD 1.5 Inpaint kết hợp ControlNet-Openpose so với Flux.1-Fill trên 20 tác vụ tái tạo gương mặt. Đánh giá riêng biệt độ bám sát tư thế và khả năng bảo toàn danh tính chủ thể.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|---|---|---|
| Inpainting | "Vẽ lấp lỗ khuyết" | Tái tạo nội dung bên trong mặt nạ; giữ nguyên pixel bên ngoài. |
| Outpainting | "Mở rộng khung ảnh" | Tái tạo không gian bên ngoài khung tranh; giữ nguyên vùng lõi. |
| 9-channel U-Net | "Mô hình inpaint chuyên dụng" | Mạng U-Net nhận đầu vào `noisy \| encoded-source \| mask`. |
| SDEdit | "Img2img theo mức nhiễu" | Thêm nhiễu đến bước $t$, sau đó khử nhiễu với prompt mới. |
| InstructPix2Pix | "Sửa ảnh bằng câu lệnh" | Mô hình khuếch tán tinh chỉnh trên bộ ba (ảnh, chỉ dẫn, kết quả). |
| RePaint | "Sửa không cần huấn luyện" | Thêm nhiễu ngược ngẫu nhiên định kỳ để triệt tiêu vết nối đường biên. |
| SAM | "Segment Anything" | Bộ tạo mặt nạ theo cú nhấp chuột hoặc hộp bao; phối hợp cùng inpaint. |
| Flux-Kontext | "Chỉnh sửa theo ngữ cảnh" | Biến thể Flux tiếp nhận trực tiếp ảnh tham chiếu + chỉ dẫn chỉnh sửa. |

## Lưu ý trong môi trường sản xuất: Đường ống chỉnh sửa rất nhạy cảm với độ trễ

Người dùng khi thao tác chỉnh sửa hình ảnh luôn kỳ vọng phản hồi dưới 5 giây. Một lượt chạy SDXL-Inpaint 30 bước ở độ phân giải $1024^2$ mất khoảng 3-4 giây trên GPU NVIDIA L4, cộng thêm thời gian tạo mặt nạ qua SAM (~200 ms) và giai đoạn mã hóa/giải mã VAE (~500 ms tổng cộng). Trong môi trường sản xuất, bài toán này bị nghẽn bởi thời gian phản hồi cho lượt đầu (TTFT) thay vì thông lượng (batch 1, lượng đồng thời thấp, cần tối ưu từng giai đoạn):

- **SAM-H là điểm nghẽn độ trễ.** SAM-H ở độ phân giải $1024^2$ tiêu tốn khoảng ~200 ms; chuyển sang SAM-ViT-B chỉ mất ~40 ms với chất lượng mặt nạ giảm không đáng kể. SAM 2 (video) có thêm chi phí quản lý bộ nhớ đệm thời gian; không nên dùng nó cho tác vụ sửa ảnh tĩnh đơn lẻ.
- **Bỏ qua bước mã hóa VAE khi có thể.** Hàm `pipe.image_processor.preprocess(img)` mã hóa ảnh thô thành các vector tiềm ẩn. Nếu bạn đã có sẵn tensor tiềm ẩn từ lượt tạo ảnh trước đó (thường gặp trong giao diện chỉnh sửa lặp lại), hãy truyền trực tiếp qua tham số `latents=...` để tiết kiệm trọn vẹn một bước mã hóa VAE.
- **Độ giãn của mặt nạ ảnh hưởng đến thông lượng tính toán.** Khi mặt nạ chỉ chiếm diện tích nhỏ, phần lớn phép tính truyền xuôi của mạng U-Net bị lãng phí (các pixel ngoài mặt nạ đều bị gán đè lại). Pipeline mặc định `StableDiffusionInpaintPipeline` của thư viện diffusers vẫn chạy toàn bộ U-Net trên toàn khung hình; chỉ các biến thể inpainting 9 kênh chuyên biệt mới có khả năng tối ưu hóa tính toán cục bộ trên vùng mặt nạ.
- **Flux-Kontext là lời giải của năm 2025-2026.** Mô hình chỉ cần một lượt truyền xuôi duy nhất trên cặp `(ảnh_nguồn, chỉ_dẫn)` — không cần bước tạo mặt nạ riêng biệt, không cần quét thử nghiệm mức nhiễu của SDEdit. Trên một GPU H100, quy trình này hoàn tất một lượt chỉnh sửa chỉ trong khoảng ~1.5 giây. Bài học kiến trúc cốt lõi ở đây là: hãy tích hợp và thu gọn các giai đoạn xử lý riêng lẻ lại làm một.

## Tài liệu đọc thêm (Further Reading)

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — thuật toán inpainting không cần huấn luyện lại.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — kỹ thuật SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — chỉnh sửa ảnh dựa trên câu lệnh tự nhiên.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, công cụ tạo mặt nạ thị giác vạn năng.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — mô hình phân vùng SAM cho ảnh và video.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — chỉnh sửa ảnh ở cấp độ trọng số chú ý chéo.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — bộ công cụ chỉnh sửa tạo sinh thế hệ mới.
