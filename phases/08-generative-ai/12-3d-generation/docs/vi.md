# Tạo hình 3D (3D Generation)

> 3D là phương thức mà đòn bẩy chuyển đổi từ 2D sang 3D phát huy sức mạnh lớn nhất. Bước đột phá của năm 2023 là kỹ thuật hòa trộn điểm ảnh Gaussian 3 chiều (3D Gaussian Splatting). Làn sóng mô hình sinh trong giai đoạn 2024-2026 đã xếp chồng mô hình khuếch tán đa góc nhìn (multi-view diffusion) kết hợp với thuật toán tái tạo 3D (3D reconstruction) để tạo ra các vật thể và khung cảnh hoàn chỉnh chỉ từ một câu lệnh văn bản hoặc một bức ảnh duy nhất.

**Type:** Học tập (Learn)  
**Languages:** Python  
**Prerequisites:** Phase 4 (Thị giác máy tính - Vision), Phase 8 · 07 (Khuếch tán tiềm ẩn - Latent Diffusion)  
**Time:** ~45 phút  

## Vấn đề đặt ra (The Problem)

Nội dung 3D luôn là một bài toán hóc búa:

- **Phương pháp biểu diễn (Representation).** Lưới đa giác (meshes), đám mây điểm (point clouds), lưới điểm ảnh thể tích (voxel grids), trường khoảng cách có dấu (signed distance fields - SDFs), trường bức xạ nơ-ron (neural radiance fields - NeRFs), các khối Gaussian 3D. Mỗi phương pháp đều có sự đánh đổi riêng.
- **Sự khan hiếm dữ liệu (Data scarcity).** ImageNet có 14 triệu hình ảnh. Bộ dữ liệu 3D sạch lớn nhất (Objaverse-XL, 2023) có khoảng 10 triệu vật thể, nhưng phần lớn trong số đó có chất lượng thấp.
- **Bộ nhớ (Memory).** Một lưới voxel kích thước $512^3$ tiêu tốn tới 128 triệu voxel; một mô hình NeRF hữu dụng cho cảnh phức tạp cần tới 1 triệu điểm mẫu trên mỗi tia (samples/ray). Việc sinh dữ liệu 3D khó hơn rất nhiều so với tái tạo lại từ ảnh có sẵn.
- **Dữ liệu giám sát (Supervision).** Với ảnh 2D, bạn có sẵn giá trị các pixel. Với 3D, thông thường bạn chỉ có một số ít góc nhìn 2D và phải tìm cách nâng bậc không gian (lift) lên 3D.

Ngăn xếp công nghệ năm 2026 tách rời hai bài toán này. Đầu tiên, tạo ra *các hình ảnh đa góc nhìn 2D (2D multi-view images)* bằng một mô hình khuếch tán. Tiếp theo, khớp một *biểu diễn 3D* (phổ biến nhất là Gaussian splatting) vào tập hợp các hình ảnh góc nhìn đó.

## Khái niệm cốt lõi (The Concept)

![Tạo hình 3D: khuếch tán đa góc nhìn + tái tạo 3D](../assets/3d-generation.svg)

### Phương pháp biểu diễn: 3D Gaussian Splatting (Kerbl et al., 2023)

Biểu diễn một khung cảnh dưới dạng một đám mây gồm khoảng 1 triệu khối Gaussian 3D. Mỗi khối Gaussian mang 59 tham số: vị trí (position: 3), hiệp phương sai (covariance: 6, hoặc quaternion 4 + tỷ lệ co giãn scale 3), độ mờ đục (opacity: 1), màu sắc biểu diễn qua hàm sóng hài cầu (spherical-harmonics color: 48 tham số ở bậc 3, 3 tham số ở bậc 0).

Kết xuất (Rendering) = phép chiếu (projection) + hòa trộn alpha (alpha-compositing). Tốc độ cực nhanh (~100 khung hình/giây ở độ phân giải 1080p trên GPU RTX 4090). Hoàn toàn khả vi (differentiable). Mô hình được khớp bằng thuật toán hạ độ dốc (gradient descent) đối chiếu với các bức ảnh thực tế chuẩn mực. Một khung cảnh có thể được khớp hoàn chỉnh trong 5-30 phút trên GPU cá nhân.

Hai đột phá bổ trợ trong giai đoạn 2023-2024:
- **Gaussian splatting tạo sinh (Generative Gaussian splats).** Các mô hình như LGM, LRM, InstantMesh dự đoán trực tiếp đám mây Gaussian từ một hoặc vài hình ảnh đầu vào.
- **Hòa trộn Gaussian 4D (4D Gaussian Splatting).** Bổ sung độ lệch (offsets) theo từng khung hình cho các khối Gaussian để tái hiện các cảnh chuyển động động (dynamic scenes).

### Khuếch tán đa góc nhìn (Multi-view diffusion)

Tinh chỉnh (fine-tune) một mô hình khuếch tán hình ảnh đã được tiền huấn luyện để nó sinh ra đồng thời nhiều góc nhìn nhất quán của cùng một vật thể từ câu lệnh văn bản hoặc một bức ảnh đơn lẻ. Điển hình gồm: Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Thông thường các mô hình này xuất ra từ 4 đến 16 góc nhìn xung quanh vật thể, sau đó được nâng lên không gian 3D thông qua Gaussian splatting hoặc NeRF.

### Quy trình tạo từ văn bản sang 3D (Text-to-3D pipelines)

| Mô hình | Đầu vào | Đầu ra | Thời gian |
|---------|---------|--------|-----------|
| DreamFusion (2022) | Văn bản | NeRF thông qua SDS | ~1 giờ mỗi mô hình |
| Magic3D | Văn bản | Lưới mesh + kết cấu (texture) | ~40 phút |
| Shap-E (OpenAI, 2023) | Văn bản | Dạng 3D ẩn (implicit 3D) | ~1 phút |
| SJC / ProlificDreamer | Văn bản | NeRF / Lưới mesh | ~30 phút |
| LRM (Meta, 2023) | Hình ảnh | Mặt phẳng ba chiều (triplane) | ~5 giây |
| InstantMesh (2024) | Hình ảnh | Lưới mesh | ~10 giây |
| SV3D (Stability, 2024) | Hình ảnh | Các góc nhìn mới | ~2 phút |
| CAT3D (Google, 2024) | 1-64 ảnh | Mô hình NeRF 3D | ~1 phút |
| TripoSR (2024) | Hình ảnh | Lưới mesh | ~1 giây |
| Meshy 4 (2025) | Văn bản + Hình ảnh | Lưới mesh kết xuất vật lý chuẩn (PBR mesh) | ~30 giây |
| Rodin Gen-1.5 (2025) | Văn bản + Hình ảnh | Lưới mesh PBR | ~60 giây |
| Tencent Hunyuan3D 2.0 (2025) | Hình ảnh | Lưới mesh | ~30 giây |

Định hướng 2025-2026: các mô hình sinh trực tiếp từ văn bản ra lưới đa giác kèm vật liệu PBR (Physically-Based Rendering) sẵn sàng tích hợp thẳng vào các engine trò chơi. Bước trung gian dùng mô hình khuếch tán đa góc nhìn vẫn là công thức mang lại chất lượng cao nhất cho các vật thể tổng quát.

### NeRF (Đặt trong bối cảnh đối chiếu)

Trường bức xạ nơ-ron (Neural Radiance Field - NeRF, Mildenhall et al., 2020). Một mạng nơ-ron truyền thẳng (MLP) siêu nhỏ nhận tọa độ `(x, y, z, hướng nhìn)` và xuất ra `(màu sắc, mật độ)`. Quá trình kết xuất được thực hiện bằng cách tích phân dọc theo các tia nhìn. Phương pháp này vượt trội kỹ thuật tổng hợp góc nhìn mới dựa trên lưới mesh về mặt chất lượng hình ảnh, nhưng lại chậm hơn từ 100 đến 1000 lần khi kết xuất. Dù đã bị Gaussian splatting thay thế trong phần lớn các ứng dụng thời gian thực, NeRF vẫn đóng vai trò quan trọng trong nghiên cứu học thuật.

````figure
v4-3d-multiview
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` cài đặt một thử nghiệm khớp "Gaussian splatting" 2D thu nhỏ: biểu diễn một ảnh mục tiêu tổng hợp (một dải chuyển màu mượt) dưới dạng tổng của các vết loang Gaussian 2D. Tối ưu hóa vị trí, màu sắc và ma trận hiệp phương sai bằng thuật toán hạ độ dốc để khớp với ảnh mục tiêu. Bạn sẽ thấy hai thao tác cốt lõi: kết xuất thuận (quét vết loang + hòa trộn alpha) và khớp mô hình thông qua hạ độ dốc.

### Bước 1: Vết Gaussian 2D (Step 1: 2D Gaussian splat)

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Bước 2: Kết xuất bằng cách cộng dồn các vết splat (Step 2: render by summing splats)

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Mô hình 3D Gaussian Splatting thực tế sẽ sắp xếp các khối Gaussian theo độ sâu và thực hiện hòa trộn alpha theo thứ tự từ trước ra sau. Bản mô phỏng 2D đơn giản hóa của chúng ta chỉ thực hiện phép cộng dồn.

### Bước 3: Khớp mô hình bằng hạ độ dốc (Step 3: fit by gradient descent)

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Cạm bẫy thực tế (Pitfalls)

- **Sự thiếu nhất quán giữa các góc nhìn (View inconsistency).** Nếu bạn sinh ra 4 góc nhìn một cách độc lập và chúng không thống nhất về cấu trúc hình học của vật thể, mô hình 3D khớp được sẽ bị nhòe và biến dạng. Giải pháp: sử dụng mô hình khuếch tán đa góc nhìn có cơ chế chú ý chia sẻ (shared attention).
- **Hiện tượng ảo giác mặt sau (Back-side hallucination).** Quá trình chuyển từ một ảnh đơn sang 3D bắt buộc mô hình phải tự tưởng tượng ra phần mặt sau không nhìn thấy. Chất lượng tạo hình ở vùng khuất này dao động rất lớn.
- **Bùng nổ số lượng Gaussian splat.** Quá trình huấn luyện không có ràng buộc sẽ khiến số lượng điểm splat tăng vọt lên tới 10 triệu và gây quá khớp (overfitting). Các kỹ thuật phỏng đoán làm dày (densification) và cắt tỉa (pruning) từ bài báo gốc 3D-GS là vô cùng thiết yếu.
- **Lỗi tô-pô hình học (Topology issues).** Lưới đa giác trích xuất từ các trường ẩn (như SDFs) thường gặp lỗi thủng lỗ hoặc tự giao nhau. Cần chạy một thuật toán tái cấu trúc lưới (remesher, ví dụ như voxel remesh của Blender) trước khi đưa vào sử dụng.
- **Bản quyền của dữ liệu huấn luyện.** Tập dữ liệu Objaverse có nhiều loại giấy phép hỗn hợp; tính khả thi trong ứng dụng thương mại tùy thuộc vào từng mô hình cụ thể.

## Ứng dụng thực tế (Use It)

| Tác vụ | Lựa chọn tối ưu 2026 |
|--------|----------------------|
| Tái tạo cảnh từ chuỗi ảnh chụp | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Tạo vật thể 3D từ văn bản cho trò chơi | Meshy 4 hoặc Rodin Gen-1.5 (đầu ra kèm vật liệu PBR) |
| Tạo hình ảnh thành 3D (Image-to-3D) | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Tổng hợp góc nhìn mới từ ít hình ảnh | CAT3D, SV3D |
| Tái tạo khung cảnh chuyển động động | 4D Gaussian Splatting |
| Nhân vật đại diện (Avatar) / người mặc trang phục | Gaussian Avatar, HUGS |
| Nghiên cứu học thuật / Đỉnh cao công nghệ (SOTA) | Các bài báo mới nhất theo tuần |

Để xuất xưởng tài nguyên 3D vào quy trình sản xuất game thương mại hoặc thương mại điện tử: Meshy 4 hoặc Rodin Gen-1.5 xuất ra các lưới mesh PBR có thể nạp trực tiếp vào Unity hoặc Unreal Engine.

## Đóng gói sản phẩm (Ship It)

Lưu tệp `outputs/skill-3d-pipeline.md`. Kỹ năng này nhận vào một bản yêu cầu 3D (đầu vào: văn bản / một ảnh / vài ảnh; đầu ra: lưới mesh / splat / NeRF; mục đích sử dụng: kết xuất đồ họa / game / VR) và đề xuất: quy trình hoàn chỉnh (khuếch tán đa góc nhìn + khớp mô hình, hoặc mô hình sinh lưới mesh trực tiếp), mô hình nền tảng, ngân sách bước lặp (iteration budget), xử lý hậu kỳ tô-pô, và các kênh vật liệu cần thiết.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py` lần lượt với 4, 16 và 64 khối Gaussian. Báo cáo sai số bình phương trung bình (MSE) cuối cùng so với ảnh mục tiêu.
2. **Trung bình.** Mở rộng mã nguồn để hỗ trợ các khối Gaussian có màu (RGB). Xác nhận kết quả tái tạo khớp với mẫu màu mục tiêu.
3. **Khó.** Sử dụng thư viện `gsplat` hoặc `Nerfstudio` để tái tạo một vật thể thực tế từ một tập hợp 50 ảnh chụp. Báo cáo thời gian khớp mô hình và chỉ số SSIM cuối cùng trên các góc nhìn kiểm thử được giữ lại (held-out views).

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| 3D Gaussian Splatting | "3DGS" | Biểu diễn cảnh thành đám mây các khối Gaussian 3D; kết xuất hòa trộn alpha khả vi. |
| NeRF | "Trường bức xạ nơ-ron" | Mạng MLP nhận vào tọa độ 3D và trả về màu sắc + mật độ; kết xuất bằng tích phân tia nhìn. |
| Triplane | "Ba mặt phẳng 2D" | Phân rã không gian 3D thành 3 lưới đặc trưng 2D căn chỉnh theo trục; tiết kiệm bộ nhớ hơn thể tích voxel. |
| SDS | "Lấy mẫu chưng cất điểm số" (Score distillation sampling) | Huấn luyện mô hình 3D bằng cách dùng đạo hàm điểm số của mô hình khuếch tán 2D làm gradient giả lập. |
| Multi-view diffusion | "Sinh nhiều góc nhìn cùng lúc" | Mô hình khuếch tán xuất ra một lô hình ảnh camera nhất quán với nhau. |
| PBR | "Kết xuất dựa trên vật lý" | Vật liệu bao gồm các kênh màu cơ bản (albedo), độ nhám (roughness), tính kim loại (metallic) và vector pháp tuyến (normal). |
| Densification | "Làm dày điểm splat" | Cơ chế phỏng đoán trong huấn luyện 3DGS: tách nhỏ hoặc nhân bản các điểm splat ở vùng có gradient cao. |

## Lưu ý triển khai: 3D chưa có một nền tảng thống nhất (Production note: 3D has no shared substrate yet)

Không giống như lĩnh vực hình ảnh (khuếch tán tiềm ẩn + DiT) hay video (DiT không-thời gian), đồ họa 3D vẫn chưa có một kiến trúc thực thi thống nhất duy nhất vào năm 2026. Cây quyết định triển khai trong sản xuất rẽ nhánh phụ thuộc vào phương thức biểu diễn:

- **NeRF / Triplane.** Quá trình suy luận bao gồm duyệt tia nhìn (ray-marching) kết hợp một lượt truyền thẳng MLP cho từng điểm mẫu. Một hình ảnh kết xuất kích thước $512^2$ cần hàng triệu lượt tính toán qua MLP. Kỹ thuật này đòi hỏi gom lô các mẫu tia một cách quyết liệt; các cơ chế tối ưu như SDPA/xformers rất hữu dụng ở đây.
- **Khuếch tán đa góc nhìn + Tái tạo LRM.** Quy trình hai giai đoạn. Giai đoạn 1 (DiT đa góc nhìn) là một máy chủ chạy khuếch tán tương tự Bài 07. Giai đoạn 2 (LRM transformer) là một lượt truyền thẳng một lần (one-shot forward pass) trên toàn bộ các góc nhìn. Hồ sơ độ trễ tổng thể có dạng "khuếch tán + một lượt tính" — hãy chọn hạ tầng phục vụ riêng cho từng giai đoạn tương ứng.
- **SDS / DreamFusion.** Đây là quá trình tối ưu hóa riêng cho từng đối tượng (per-asset optimization), không phải là suy luận tức thì. Cần xử lý dưới dạng các tác vụ nền theo mẻ (build jobs), không phải là bộ xử lý yêu cầu phản hồi nhanh.

Đối với phần lớn sản phẩm năm 2026, câu trả lời chuẩn xác là "chạy mô hình khuếch tán đa góc nhìn theo yêu cầu, tái tạo thành 3DGS bất đồng bộ, và cung cấp tệp 3DGS đó để người dùng xem trong thời gian thực". Cách làm này phân chia khối lượng công việc mạch lạc giữa máy chủ suy luận GPU (tốc độ nhanh) và trình tối ưu hóa ngoại tuyến (tốc độ chậm).

## Tài liệu đọc thêm (Further Reading)

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — Bài báo đặt nền móng cho NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — Bài báo khởi xướng 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — Cơ chế SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — Khuếch tán đa góc nhìn.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — Mô hình tái tạo LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d-novel-multi-view-synthesis-and-3d-generation-from-a-single-image-using-latent-video-diffusion) — SV3D.
