# Transfusion: Văn bản Tự hồi quy + Khuếch tán Hình ảnh trong một Transformer

> Chameleon và Emu3 đặt cược toàn bộ vào các token rời rạc (discrete tokens). Chúng hoạt động được, nhưng nút thắt lượng tử hóa (quantization bottleneck) bộc lộ rất rõ — chất lượng hình ảnh đạt ngưỡng chặn (plateau) thấp hơn các mô hình khuếch tán không gian liên tục (continuous-space diffusion models). Transfusion (Meta, Zhou et al., tháng 8 năm 2024) đưa ra một lựa chọn trái ngược: giữ hình ảnh ở dạng liên tục, loại bỏ hoàn toàn VQ-VAE, và huấn luyện một transformer duy nhất với hai hàm mất mát (two losses). Các token văn bản sử dụng hàm dự đoán token tiếp theo (next-token prediction - NTP). Các mảng hình ảnh (image patches) sử dụng hàm mất mát khớp dòng chảy (flow-matching) / khuếch tán (diffusion loss). Cả hai mục tiêu đều tối ưu hóa cùng một bộ trọng số. Kiến trúc nền tảng của Stable Diffusion 3 (MMDiT) là một biến thể họ hàng gần gũi. Bài học này sẽ phân tích luận điểm của Transfusion, xây dựng một bộ huấn luyện đồ chơi sử dụng hai hàm mất mát, và truy vết ma trận mặt nạ chú ý (attention mask) cho phép một transformer đảm nhận đồng thời cả hai nhiệm vụ.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ huấn luyện hai hàm mất mát trên bài toán đồ chơi quy mô MNIST)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 11 (Chameleon), Giai đoạn 8 (AI tạo sinh - Generative AI)  
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Thiết lập một transformer chạy đồng thời hai hàm mất mát (NTP trên token văn bản, MSE khuếch tán trên mảng hình ảnh) trên cùng một mạng xương sống (backbone).
- Giải thích vì sao cơ chế chú ý hai chiều (bidirectional attention) giữa các mảng hình ảnh kết hợp với chú ý nhân quả (causal attention) trên token văn bản là lựa chọn mặt nạ chuẩn xác.
- So sánh phong cách Transfusion (ảnh liên tục, hàm mất mát khuếch tán) với phong cách Chameleon (ảnh rời rạc, NTP) về chi phí tính toán (compute), chất lượng và độ phức tạp mã nguồn.
- Nêu bật đóng góp của MMDiT: các trọng số chuyên biệt theo phương thức (modality-specific weights) ở từng khối, chú ý kết hợp (joint attention) tại luồng dư (residual stream).

## Vấn đề

Cuộc tranh luận giữa token hình ảnh rời rạc và liên tục đã xuất hiện từ trước kỷ nguyên LLM. Biểu diễn liên tục (điểm ảnh thô, biến tiềm ẩn VAE) lưu giữ được chi tiết. Token rời rạc (chỉ số lượng tử hóa vector - VQ) khớp với từ vựng tự nhiên của transformer nhưng làm mất chi tiết ở bước lượng tử hóa (quantization step).

Chameleon / Emu3 chọn hướng rời rạc: một hàm mất mát, một kiến trúc, nhưng độ trung thực hình ảnh bị giới hạn bởi chất lượng của bộ token hóa (tokenizer).

Các mô hình khuếch tán (diffusion models) chọn hướng liên tục: chất lượng hình ảnh vượt trội, nhưng lại là mô hình tách biệt hoàn toàn khỏi LLM, đòi hỏi kỹ thuật lên lịch nhiễu (noise-schedule engineering) phức tạp, và không tích hợp liền mạch với quá trình sinh văn bản.

Transfusion đặt câu hỏi: liệu ta có thể dung hòa cả hai? Giữ hình ảnh ở dạng liên tục, vẫn huấn luyện một mô hình duy nhất, sử dụng hai hàm mất mát được ghép chung trong một bước cập nhật gradient.

## Khái niệm

### Kiến trúc hai hàm mất mát (The two-loss architecture)

Một transformer chỉ giải mã (decoder-only transformer) duy nhất xử lý một chuỗi bao gồm:

- Các token văn bản (rời rạc, từ từ vựng BPE).
- Các mảng hình ảnh (liên tục, các khối pixel 16x16 được chiếu vào chiều ẩn thông qua nhúng tuyến tính — tương tự như đầu vào của bộ mã hóa ViT).
- Các thẻ `<image>` và `</image>` đánh dấu vị trí các mảng liên tục xuất hiện.

Lượt truyền xuôi (forward pass) chạy một lần. Hàm mất mát chọn một trong hai đầu ra (head) cho mỗi token:

- Với token văn bản: hàm mất mát entropy chéo (cross-entropy) tiêu chuẩn trên đầu ra logits từ vựng.
- Với mảng hình ảnh: hàm mất mát khuếch tán (diffusion loss) trên các mảng liên tục — dự đoán phần nhiễu đã được thêm vào mỗi mảng ảnh.

Gradient lan truyền qua phần thân transformer dùng chung. Cả hai hàm mất mát cùng đồng thời cải thiện bộ trọng số dùng chung.

### Mặt nạ chú ý: văn bản nhân quả + hình ảnh hai chiều (Attention mask: causal text + bidirectional image)

Token văn bản bắt buộc phải có tính nhân quả (causal) — bạn không thể để một token văn bản chú ý tới văn bản tương lai, nếu không cơ chế teacher forcing sẽ bị phá vỡ. Tuy nhiên, các mảng hình ảnh lại đại diện cho một ảnh chụp tức thời; chúng nên chú ý lẫn nhau theo hai chiều (bidirectional) bên trong cùng một khối hình ảnh.

Mặt nạ:

```python
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # nhân quả cho văn bản
  OR (i is image and j is image and same_image_block(i, j))   # hai chiều bên trong hình ảnh
  OR (i is text and j is image and j < i)   # văn bản chú ý đến ảnh đã đóng trước nó
  OR (i is image and j is text and j < i_image_start)   # hình ảnh chú ý đến văn bản đứng trước
```

Được triển khai dưới dạng mặt nạ tam giác khối (block-triangular mask) trong quá trình huấn luyện và suy luận.

### Hàm mất mát khuếch tán bên trong transformer (Diffusion loss inside the transformer)

Hàm mất mát khuếch tán tuân theo chuẩn mực: thêm nhiễu vào mảng hình ảnh, yêu cầu mô hình dự đoán phần nhiễu (hoặc tương đương là dự đoán mảng ảnh sạch ban đầu). Phiên bản của Transfusion sử dụng khớp dòng chảy (flow matching) — dự đoán trường vận tốc (velocity field) từ trạng thái nhiễu về trạng thái sạch.

Trong quá trình huấn luyện:
1. Với mỗi mảng hình ảnh $x_0$, lấy mẫu một bước thời gian ngẫu nhiên $t$.
2. Lấy mẫu nhiễu $\epsilon$, tính $x_t = (1-t) \cdot x_0 + t \cdot \epsilon$ (phép nội suy tuyến tính cho flow matching).
3. Transformer dự đoán $v_\theta(x_t, t)$; $\text{loss} = \mathop{\mathrm{MSE}}(v_\theta(x_t, t), \epsilon - x_0)$.
4. Lan truyền ngược cùng với các hàm mất mát NTP văn bản từ cùng một chuỗi.

Khi suy luận (inference), quá trình sinh diễn ra như sau:
- Token văn bản: lấy mẫu tự hồi quy (autoregressive sampling) tiêu chuẩn.
- Mảng hình ảnh: vòng lặp lấy mẫu khuếch tán (thường từ 10-30 bước) có điều kiện dựa trên các token văn bản đứng trước.

### MMDiT: Biến thể của Stable Diffusion 3 (MMDiT: Stable Diffusion 3's variant)

Stable Diffusion 3 (Esser et al., tháng 3 năm 2024) đã ra mắt MMDiT (Multimodal Diffusion Transformer) vào khoảng cùng thời điểm với Transfusion. Hai kiến trúc này là những biến thể họ hàng.

Các điểm khác biệt chính của MMDiT:

- Trọng số chuyên biệt theo phương thức cho từng khối (Modality-specific weights per block). Mỗi khối transformer có các trọng số Q, K, V và MLP riêng biệt cho token văn bản so với mảng hình ảnh. Cơ chế chú ý là kết hợp liên phương thức (joint attention); mọi thứ khác đều mang tính chuyên biệt theo phương thức.
- Huấn luyện theo dòng chảy chỉnh lưu (Rectified flow training). Một biến thể flow matching đặc thù với cơ chế lấy mẫu đã biết và công thức toán học tinh giản hơn DDPM.
- Quy mô (Scale). MMDiT là mạng xương sống cho SD3 (các biến thể 2B và 8B tham số). Bài báo của Transfusion mở rộng quy mô lên 7B.

Cả hai cùng hội tụ về một ý tưởng cốt lõi: một transformer duy nhất chạy NTP trên văn bản và khuếch tán trên biểu diễn hình ảnh liên tục.

### Vì sao cách tiếp cận này vượt trội phong cách Chameleon (Why this beats Chameleon-style)

Khoảng cách chất lượng giữa khuếch tán liên tục và NTP rời rạc trong bài toán sinh hình ảnh là có thể đo lường được. Bài báo Transfusion chỉ ra:

- Ở quy mô 7B tham số, vượt qua mô hình phong cách Chameleon cùng kích thước từ 3-5 điểm FID.
- Không đòi hỏi huấn luyện bộ token hóa — bộ mã hóa hình ảnh đơn giản hơn (phép chiếu tuyến tính vào chiều ẩn, tương tự như lớp đầu vào của ViT).
- Quá trình suy luận có thể khử nhiễu song song trên các mảng hình ảnh, không giống như sinh tuần tự từng token ảnh tự hồi quy.

Nhược điểm: Transfusion là một mô hình hai hàm mất mát, khiến động lực học huấn luyện (training dynamics) trở nên phức tạp hơn. Trọng số của các hàm mất mát cần được tinh chỉnh cẩn thận. Sự lệch pha trong lịch trình học giữa NTP và khuếch tán có thể khiến một đầu ra lấn át đầu ra còn lại.

### Các hướng phát triển tiếp nối (What sits downstream)

Janus-Pro (Bài học 12.15) tinh chỉnh ý tưởng của Transfusion bằng cách tách rời bộ mã hóa thị giác cho tác vụ hiểu và tác vụ sinh — SigLIP cho hiểu, VQ cho sinh — trong khi vẫn chia sẻ phần thân transformer. Show-o (Bài học 12.14) thay thế khuếch tán bằng khuếch tán rời rạc (dự đoán có che mờ - masked prediction). Họ mô hình sinh hợp nhất (unified-generation family) phân nhánh nhanh chóng sau Transfusion.

Các mô hình sản xuất đóng có khả năng xuất hình ảnh tự nhiên có thể sử dụng một hậu duệ nào đó thuộc họ kiến trúc này, nhưng kiến trúc của chúng không được công bố công khai, do đó mọi tuyên bố tương tự đều chỉ mang tính phỏng đoán.

````figure
cfg-guidance-scale
````

## Thực hành (Use It)

`code/main.py` xây dựng một mô hình Transfusion thu nhỏ trên bài toán đồ chơi dạng MNIST:

- Chú thích văn bản (text captions) là các chuỗi số nguyên ngắn mô tả một chữ số (0-9).
- Hình ảnh là các lưới byte kích thước 4x4.
- Một cặp phép chiếu tuyến tính dùng chung trọng số đóng vai trò thay thế cho transformer; hàm mất mát NTP trên văn bản, hàm mất mát MSE trên các mảng ảnh bị nhiễu.
- Vòng lặp huấn luyện luân phiên giữa hai hàm mất mát, mặt nạ chú ý được xây dựng tường minh.
- Phần minh họa mặt nạ in ra cấu trúc chú ý dạng tam giác khối cho chuỗi gồm văn bản, hình ảnh, văn bản. Quá trình sinh mẫu không được chạy thực tế: một bộ lấy mẫu thực thụ sẽ giải mã văn bản theo từng token, sau đó khử nhiễu các mảng hình ảnh qua nhiều bước, như được tính toán trong các bài tập bên dưới.

Transformer ở đây chỉ là mô hình đồ chơi. Tuy nhiên, hệ thống đường ống hai hàm mất mát (two-loss plumbing), cấu trúc mặt nạ chú ý và vòng lặp suy luận chính là các sản phẩm thực tế cần nắm vững.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-two-loss-trainer-designer.md`. Khi đối mặt với một tác vụ huấn luyện đa phương thức mới (văn bản + hình ảnh, văn bản + âm thanh, văn bản + video), kỹ năng này sẽ thiết kế lịch trình hai hàm mất mát (trọng số hàm mất mát, hình dạng mặt nạ, các khối dùng chung so với khối chuyên biệt theo phương thức) và cảnh báo các rủi ro triển khai.

## Bài tập (Exercises)

1. Một mô hình kiểu Transfusion huấn luyện trên 70% token văn bản và 30% mảng hình ảnh. Hàm mất mát khuếch tán hình ảnh có độ lớn gấp khoảng 10 lần hàm mất mát NTP văn bản. Cần đặt trọng số hàm mất mát (loss weights) như thế nào để cân bằng chúng?

2. Triển khai mặt nạ tam giác khối cho một chuỗi: `[T, T, <image>, P, P, P, P, </image>, T]`. Đánh dấu mỗi phần tử là 0 hoặc 1.

3. MMDiT có các trọng số QKV chuyên biệt theo phương thức. Điều này làm tăng bao nhiêu phần trăm chi phí tham số (parameter overhead) so với transformer chia sẻ toàn bộ của Transfusion? Ở quy mô 7B tham số, sự đánh đổi này có xứng đáng không?

4. Quá trình sinh (generation): cho một prompt văn bản, mô hình chạy NTP cho 50 token, sau đó gặp `<image>`, rồi chạy khuếch tán trên 256 mảng ảnh qua 20 bước khử nhiễu. Tổng cộng cần bao nhiêu lượt truyền xuôi (forward passes)?

5. Đọc Mục 3 bài báo SD3. Mô tả rectified flow và giải thích vì sao nó hội tụ sau ít bước suy luận hơn so với DDPM.

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Two-loss training (Huấn luyện hai hàm mất mát) | "NTP + khuếch tán" | Một transformer duy nhất tối ưu hóa đồng thời entropy chéo trên token văn bản và MSE trên mảng hình ảnh liên tục trong cùng một bước gradient |
| Flow matching (Khớp dòng chảy) | "Rectified flow" | Biến thể khuếch tán dự đoán trường vận tốc từ nhiễu về dữ liệu sạch; toán học đơn giản hơn DDPM |
| MMDiT | "DiT đa phương thức" | Kiến trúc của Stable Diffusion 3: chú ý kết hợp, các khối MLP và chuẩn hóa chuyên biệt theo phương thức |
| Block-triangular mask (Mặt nạ tam giác khối) | "Văn bản nhân quả + hình ảnh hai chiều" | Mặt nạ chú ý có tính nhân quả trên văn bản nhưng hai chiều bên trong các vùng hình ảnh |
| Continuous image representation (Biểu diễn hình ảnh liên tục) | "Không dùng VQ" | Các mảng hình ảnh dưới dạng vector giá trị thực, không phải chỉ số sổ mã (codebook) số nguyên |
| Velocity prediction (Dự đoán vận tốc) | "v-parameterization" | Đầu ra của mạng là trường vận tốc giữa nhiễu và dữ liệu, không phải bản thân vector nhiễu |

## Tài liệu tham khảo (Further Reading)

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
