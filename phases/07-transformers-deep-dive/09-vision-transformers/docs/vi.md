# Vision Transformers (ViT)

> Một hình ảnh là một lưới các mảng vá (patches). Một câu văn là một chuỗi các token. Cùng một kiến trúc transformer có thể tiếp nhận và xử lý cả hai.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 phút

## Vấn đề (The Problem)

Trước năm 2020, thị giác máy tính (computer vision) đồng nghĩa với mạng tích chập (CNN - convolutional neural networks). Mọi mô hình đạt đỉnh cao công nghệ (SOTA - state-of-the-art) trên ImageNet, COCO và các chuẩn đánh giá phát hiện vật thể đều dựa trên mạng xương sống (backbone) CNN. Transformer khi đó chỉ dành riêng cho ngôn ngữ.

Dosovitskiy và cộng sự (2020) — với bài báo "An Image is Worth 16x16 Words" (Một bức ảnh đáng giá 16x16 từ) — đã chứng minh rằng bạn có thể loại bỏ hoàn toàn các lớp tích chập. Cắt một bức ảnh thành các mảng vá (patches) có kích thước cố định, chiếu tuyến tính (linearly project) từng mảng vá thành một vector nhúng (embedding), và đưa chuỗi này vào một bộ mã hóa transformer (transformer encoder) nguyên bản. Khi có đủ quy mô dữ liệu (tiền huấn luyện trên ImageNet-21k hoặc lớn hơn), ViT đạt hiệu năng tương đương hoặc vượt qua các mô hình dựa trên ResNet.

ViT là điểm khởi đầu cho một mô thức rộng lớn hơn vào năm 2026: một kiến trúc duy nhất cho nhiều phương thức dữ liệu (modalities). Whisper chuyển đổi âm thanh thành token. ViT chuyển đổi hình ảnh thành token. Token hành động cho robot học (robotics). Token điểm ảnh cho video. Transformer không quan tâm bản chất dữ liệu là gì — chỉ cần cung cấp cho nó một chuỗi tuần tự, nó sẽ tự học các biểu diễn.

Đến năm 2026, ViT và các thế hệ kế tục (DeiT, Swin, DINOv2, ViT-22B, SAM 3) đã làm chủ hầu hết các tác vụ thị giác. CNN chỉ còn giữ ưu thế trên các thiết bị biên (edge devices) và các tác vụ cực kỳ nhạy cảm với độ trễ. Hầu như mọi hệ thống thị giác hiện đại khác đều có sự hiện diện của ViT ở một khâu nào đó trong ngăn xếp công nghệ.

## Khái niệm (The Concept)

![Hình ảnh → các mảng vá (patches) → tokens → transformer](../assets/vit.svg)

### Bước 1 — Chia mảng vá (patchify)

Chia một bức ảnh kích thước `H × W × C` thành một chuỗi $N \times (P \cdot P \cdot C)$ gồm các mảng vá phẳng. Thiết lập điển hình: ảnh `224 × 224`, mảng vá `16 × 16` → 196 mảng vá, mỗi mảng gồm 768 giá trị số.

```text
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Kích thước mảng vá là đòn bẩy điều khiển: Mảng vá nhỏ hơn = nhiều token hơn, độ phân giải chi tiết hơn, nhưng chi phí tính toán cơ chế chú ý tăng theo hàm bậc hai (quadratic cost). Mảng vá lớn hơn = biểu diễn thô hơn nhưng chi phí tính toán rẻ hơn.

### Bước 2 — Nhúng tuyến tính (linear embedding)

Một ma trận trọng số học được duy nhất sẽ chiếu mỗi mảng vá phẳng vào không gian ẩn `d_model`. Thao tác này tương đương về mặt toán học với một phép tích chập có kích thước bộ lọc (kernel size) `P` và bước nhảy (stride) `P`. Trong PyTorch, thao tác này được hiện thực hóa đơn giản bằng `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — chỉ vỏn vẹn 2 dòng mã.

### Bước 3 — Thêm token `[CLS]` ở đầu và cộng mã hóa vị trí (positional embeddings)

- Thêm một token học được `[CLS]` vào đầu chuỗi. Trạng thái ẩn cuối cùng của token này sẽ là biểu diễn tổng hợp của toàn bộ hình ảnh được dùng cho bài toán phân loại.
- Cộng thêm vector nhúng vị trí học được (ViT nguyên bản) hoặc mã hóa vị trí dạng sóng sin 2 chiều (sinusoidal 2D trong các biến thể sau này).
- Từ năm 2024+, kỹ thuật RoPE (Rotary Position Embedding) được mở rộng sang không gian 2D cho vị trí, đôi khi không cần đến vector nhúng vị trí tường minh.

### Bước 4 — Bộ mã hóa transformer tiêu chuẩn

Xếp chồng L khối gồm chuỗi: `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Hoàn toàn giống hệt kiến trúc BERT. Không chứa bất kỳ lớp xử lý nào đặc thù riêng cho hình ảnh. Đây chính là điểm nhấn sư phạm cốt lõi đầy ấn tượng của bài báo.

### Bước 5 — Đầu dự đoán (head)

Đối với bài toán phân loại: lấy trạng thái ẩn của token `[CLS]` → lớp tuyến tính (linear) → hàm softmax. Đối với DINOv2 hoặc SAM, bỏ qua token `[CLS]` và sử dụng trực tiếp các vector nhúng của từng mảng vá.

### Các biến thể quan trọng (Variants that mattered)

| Mô hình | Năm | Điểm đổi mới |
|-------|------|--------|
| ViT | 2020 | Bản gốc. Kích thước mảng vá cố định, chú ý toàn cục đầy đủ (full global attention). |
| DeiT | 2021 | Chưng cất tri thức (distillation); có thể huấn luyện hiệu quả chỉ trên tập ImageNet-1k. |
| Swin | 2021 | Cấu trúc phân cấp với các cửa sổ dịch chuyển (shifted windows). Giảm độ phức tạp tính toán xuống dưới bậc hai. |
| DINOv2 | 2023 | Tự giám sát (self-supervised, không cần nhãn). Trích xuất các đặc trưng thị giác tổng quát tốt nhất. |
| ViT-22B | 2023 | Quy mô 22 tỷ tham số; chứng minh các quy luật mở rộng (scaling laws) áp dụng hoàn hảo cho thị giác. |
| SigLIP | 2023 | Cặp ViT + mô hình ngôn ngữ, huấn luyện bằng hàm mất mát tương phản sigmoid (sigmoid contrastive loss). |
| SAM 3 | 2025 | Phân đoạn mọi thứ (Segment Anything); ViT-Large kết hợp với decoder tạo mặt nạ có thể tiếp nhận chỉ dẫn (promptable mask decoder). |

### Tại sao ViT cần thời gian để bứt phá

ViT cần một lượng dữ liệu *cực kỳ lớn* để bắt kịp và vượt qua CNN vì nó không mang các thiên kiến quy nạp (inductive biases) cố hữu của CNN (như tính bất biến tịnh tiến - translation invariance, và tính cục bộ - locality). Nếu không có hơn 100 triệu hình ảnh gắn nhãn hoặc quá trình tiền huấn luyện tự giám sát mạnh mẽ, CNN vẫn chiến thắng ở cùng mức chi phí tính toán. DeiT đã khắc phục điều này vào năm 2021 bằng các thủ thuật chưng cất tri thức; và DINOv2 đã giải quyết triệt để vấn đề này vào năm 2023 nhờ phương pháp học tự giám sát.

```figure
n5-patch-stream
```

## Tự xây dựng (Build It)

Xem `code/main.py`. Triển khai hoàn toàn bằng thư viện chuẩn (stdlib) của Python gồm logic cắt mảng vá + nhúng tuyến tính + các bài kiểm tra tính hợp lý. Không thực hiện huấn luyện — việc huấn luyện ViT ở bất kỳ quy mô thực tế nào đều đòi hỏi PyTorch và hàng giờ tính toán trên GPU.

### Bước 1: tạo ảnh giả lập (fake image)

Một bức ảnh RGB kích thước 24 × 24 dưới dạng danh sách các hàng chứa các bộ giá trị `(R, G, B)`. Chúng ta sử dụng các mảng vá kích thước 6×6 → 16 mảng vá, mỗi mảng là một vector 108 chiều.

### Bước 2: cắt ảnh thành mảng vá (patchify)

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Thứ tự quét mảng: duyệt theo hàng (row-major) trên toàn bộ lưới ảnh. Mọi mô hình ViT đều sử dụng thứ tự quét này.

### Bước 3: nhúng tuyến tính (linear embed)

Nhân mỗi mảng vá phẳng với một ma trận ngẫu nhiên kích thước `(patch_flat_size, d_model)`. Xác minh kích thước đầu ra là `(N_patches + 1, d_model)` sau khi đã nối thêm token `[CLS]` vào đầu chuỗi.

### Bước 4: đếm tham số cho một mô hình ViT thực tế

In số lượng tham số cho cấu hình ViT-Base: 12 lớp, 12 đầu chú ý, d=768, kích thước mảng vá = 16. So sánh với ResNet-50 (~25M tham số). ViT-Base đạt mức ~86M tham số. ViT-Large khoảng ~307M. ViT-Huge khoảng ~632M.

## Sử dụng trong thực tế (Use It)

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 mảng vá
cls_emb = out[:, 0]                       # biểu diễn của hình ảnh
```

**Các vector nhúng của DINOv2 là lựa chọn mặc định vào năm 2026 cho đặc trưng hình ảnh.** Đóng băng mô hình nền tảng (backbone), chỉ huấn luyện một đầu phân loại nhỏ. Hoạt động xuất sắc cho phân loại, truy xuất hình ảnh, phát hiện vật thể, sinh chú thích ảnh. Các điểm kiểm tra DINOv2 của Meta vượt trội hơn CLIP trên mọi tác vụ thị giác thuần túy không kèm văn bản.

**Lựa chọn kích thước mảng vá.** Các mô hình nhỏ thường dùng 16×16 (ViT-B/16). Tác vụ dự đoán dày đặc (dense prediction như phân đoạn ảnh) sử dụng 8×8 hoặc 14×14 (SAM, DINOv2). Các mô hình rất lớn thường chọn kích thước 14×14.

## Đưa vào triển khai (Ship It)

Xem `outputs/skill-vit-configurator.md`. Kỹ năng này hướng dẫn lựa chọn biến thể ViT và kích thước mảng vá phù hợp cho một bài toán thị giác mới, dựa trên quy mô tập dữ liệu, độ phân giải ảnh và ngân sách tính toán.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py`. Xác minh số lượng mảng vá bằng `(H/P) * (W/P)` và kích thước mảng vá phẳng bằng `P*P*C`.
2. **Trung bình.** Triển khai mã hóa vị trí hình sin 2 chiều (2D sinusoidal positional embeddings) — hai bộ mã hình sin độc lập cho `hàng` và `cột` của từng mảng vá, sau đó ghép nối lại. Đưa chúng vào một mô hình ViT nhỏ bằng PyTorch và so sánh độ chính xác so với mã hóa vị trí học được trên tập CIFAR-10.
3. **Khó.** Xây dựng một ViT 3 lớp (bằng PyTorch), huấn luyện trên 1.000 hình ảnh MNIST với mảng vá 4×4. Đo độ chính xác trên tập kiểm tra. Sau đó thêm bước tiền huấn luyện DINOv2 trên cùng 1.000 ảnh đó (ở dạng đơn giản hóa: huấn luyện encoder dự đoán vector nhúng của mảng vá từ các mảng vá bị che). Liệu độ chính xác có cải thiện?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|------|-----------------|-----------------------|
| Mảng vá (Patch) | "Token của thị giác máy tính" | Vector phẳng chứa các giá trị pixel của một vùng kích thước `P × P × C` trong ảnh. |
| Chia mảng vá (Patchify) | "Cắt nhỏ + làm phẳng" | Cắt hình ảnh thành các mảng vá không chồng lấn, làm phẳng từng mảng thành một vector. |
| Token `[CLS]` | "Bản tóm tắt ảnh" | Một token học được được thêm vào đầu chuỗi; vector nhúng cuối cùng của nó là biểu diễn toàn cảnh của hình ảnh. |
| Thiên kiến quy nạp (Inductive bias) | "Giả định sẵn có của mô hình" | ViT có ít tiên đề cấu trúc hơn so với CNN (không mặc định tính cục bộ); do đó cần nhiều dữ liệu hơn để bù đắp. |
| DINOv2 | "ViT tự giám sát" | Được huấn luyện không cần nhãn bằng phương pháp tăng cường ảnh + mạng giáo viên động lượng (momentum teacher). Cung cấp đặc trưng ảnh tổng quát tốt nhất năm 2026. |
| SigLIP | "Mô hình kế tục CLIP" | Bộ đôi ViT + text encoder huấn luyện bằng hàm mất mát tương phản sigmoid; vượt trội hơn CLIP ở cùng mức chi phí tính toán. |
| Swin | "ViT chia cửa sổ" | Mô hình ViT phân cấp với cơ chế chú ý cục bộ + cửa sổ trượt; độ phức tạp dưới bậc hai. |
| Register tokens | "Thủ thuật năm 2023" | Một số token học được bổ sung nhằm hấp thu các điểm tập trung chú ý (attention sinks); giúp làm mịn và tăng chất lượng đặc trưng của DINOv2. |

## Tài liệu đọc thêm (Further Reading)

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — bài báo gốc đặt nền móng cho ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — bài báo DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — bài báo Swin Transformer.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — bài báo DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — kỹ thuật register token cho DINOv2.
