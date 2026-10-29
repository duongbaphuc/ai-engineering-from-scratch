# Mô hình tự hồi quy thị giác (VAR): Dự đoán tỷ lệ kế tiếp (Visual Autoregressive Modeling (VAR): Next-Scale Prediction)

> Các mô hình khuếch tán (diffusion models) lấy mẫu lặp đi lặp lại theo trục thời gian (các bước khử nhiễu). Ngược lại, VAR lấy mẫu lặp đi lặp lại theo trục tỷ lệ độ phân giải (scale) — mô hình dự đoán token kích thước 1x1, sau đó đến 2x2, rồi 4x4, cho tới khi đạt độ phân giải cuối cùng, với mỗi cấp tỷ lệ đều được điều kiện hóa dựa trên các cấp tỷ lệ trước đó. Bài báo khoa học năm 2024 đã chứng minh VAR tuân theo các định luật tỷ lệ (scaling laws) tương tự như kiến trúc GPT trong lĩnh vực tạo ảnh và vượt trội hơn DiT trên cùng một ngân sách tính toán. Bài học này sẽ tự tay xây dựng cơ chế cốt lõi đó.

**Type:** Xây dựng (Build)  
**Languages:** Python (với PyTorch)  
**Prerequisites:** Phase 7 Lesson 03 (Cơ chế chú ý đa đầu - Multi-Head Attention), Phase 8 Lesson 06 (Mô hình xác suất khuếch tán khử nhiễu - DDPM)  
**Time:** ~90 phút  

## Vấn đề đặt ra (The Problem)

Mô hình sinh tự hồi quy (Autoregressive - AR) thống trị hoàn toàn trong lĩnh vực xử lý ngôn ngữ vì nó tuân theo định luật mở rộng quy mô một cách nhất quán: tăng tính toán, tăng tham số dẫn đến giảm độ phức tạp (perplexity) và chất lượng đầu ra tốt hơn. Trước năm 2024, tạo ảnh bằng AR có hai hướng tiếp cận chính: PixelRNN/PixelCNN (từng pixel một) và DALL-E 1 / Parti / MuseGAN (từng token một trên các mã VQ-VAE).

Cả hai hướng đi trên đều vấp phải bài toán nan giải về thứ tự sinh (generation-order problem). Các pixel và token được sắp xếp trên một lưới 2 chiều, nhưng mô hình AR bắt buộc phải duyệt qua chúng theo thứ tự quét dòng (raster order) 1 chiều từ trái sang phải, từ trên xuống dưới. Một pixel ở góc đầu tiên hoàn toàn không có ý niệm gì về bức ảnh tổng thể sẽ trở thành cái gì sau đó. Khả năng mở rộng chất lượng của AR trên hình ảnh kém xa GPT trên văn bản và chưa bao giờ đạt tới chất lượng của mô hình khuếch tán trên cùng mức chi phí tính toán.

VAR đã giải quyết triệt để bài toán thứ tự sinh bằng cách thay đổi bản chất của đối tượng được tạo ra. Thay vì dự đoán các token hình ảnh nối tiếp nhau theo không gian hình học, VAR dự đoán toàn bộ bức ảnh ở các cấp độ phân giải tăng dần. Bước 1: dự đoán token kích thước 1x1 (bản tóm lược toàn cục của bức ảnh). Bước 2: dự đoán lưới token 2x2 (các đặc trưng thô). Bước 3: dự đoán lưới token 4x4. Bước K: dự đoán lưới token cuối cùng có kích thước (H/8)x(W/8).

Mỗi cấp tỷ lệ sẽ tính toán cơ chế chú ý (attention) tới tất cả các cấp tỷ lệ trước đó (theo thứ tự nhân quả giữa các cấp tỷ lệ) và xử lý hoàn toàn song song bên trong cấp tỷ lệ hiện tại. Vấn đề thứ tự sinh biến mất hoàn toàn: toàn bộ bức ảnh ở cấp độ phân giải $k$ được tạo ra chỉ trong một lượt truyền thẳng duy nhất qua transformer.

## Khái niệm cốt lõi (The Concept)

### Bộ tách token đa tỷ lệ VQ-VAE (VQ-VAE Multi-Scale Tokenizer)

VAR yêu cầu một **bộ tách token rời rạc đa tỷ lệ (multi-scale discrete tokenizer)**. Đối với một ảnh $x$, nó tạo ra một chuỗi các lưới token có độ phân giải tăng dần:

```text
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Mỗi $z_k$ sử dụng chung một từ điển mã (codebook, kích thước điển hình từ 4096 đến 16384). Việc tách token ở mỗi cấp độ không diễn ra độc lập — nó được huấn luyện sao cho tổng các phần dư ở mỗi cấp tỷ lệ có thể tái tạo lại biểu diễn tiềm ẩn ban đầu $f$:

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

Đây là một biến thể của **lượng tử hóa vector phần dư (Residual Vector Quantization - RVQ)**. Cấp tỷ lệ $k$ nắm bắt những chi tiết mà các cấp tỷ lệ từ $1 \ldots k-1$ còn bỏ sót. Bộ giải mã (decoder) nhận vào tổng các vector nhúng của tất cả các cấp tỷ lệ để tái tạo bức ảnh hoàn chỉnh.

Bộ tách token VQ đa tỷ lệ này được huấn luyện một lần duy nhất (tương tự như VQGAN) và sau đó được đóng băng (freeze). Toàn bộ khối lượng công việc tạo sinh do mô hình tự hồi quy phía trên đảm nhiệm.

### Dự đoán cấp tỷ lệ kế tiếp (Next-Scale Prediction)

Mô hình sinh là một transformer nhận vào các token từ tất cả các cấp tỷ lệ đứng trước và dự đoán đồng thời các token ở cấp tỷ lệ kế tiếp.

Cấu trúc chuỗi đầu vào:
```text
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Vector nhúng vị trí (position embeddings) mã hóa cả chỉ số cấp tỷ lệ lẫn vị trí không gian cụ thể bên trong cấp tỷ lệ đó. Cơ chế chú ý có tính nhân quả theo thứ tự cấp tỷ lệ và hoạt động theo từng khối (block-wise) bên trong nội bộ một cấp: một token tại cấp tỷ lệ $k$, vị trí $(i, j)$ có thể chú ý tới tất cả các token ở các cấp $1 \ldots k$, bao gồm mọi vị trí trong chính cấp $k$. Không có bất kỳ ràng buộc thứ tự hay tính nhân quả nào bên trong một cấp tỷ lệ, do đó tất cả các vị trí trong cùng một cấp độ được dự đoán song song hoàn toàn.

Hàm mất mát khi huấn luyện: tại mỗi cấp độ $k$, mô hình dự đoán các token $z_k$ dựa trên tất cả các token ở các cấp độ trước đó. Hàm mất mát là cross-entropy trên các mã VQ rời rạc. Cấu trúc này tương tự GPT, ngoại trừ việc "chuỗi tuần tự" nay đã được tổ chức theo từng bậc phân giải.

### Quá trình sinh ảnh (Generation)

Trong giai đoạn suy luận:
```text
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Với $K = 10$ cấp tỷ lệ, toàn bộ quá trình sinh ảnh chỉ tiêu tốn đúng 10 lượt truyền thẳng qua transformer. Mỗi lượt tạo ra toàn bộ cấp tỷ lệ tương ứng một cách song song — không cần tự hồi quy tuần tự trên từng token. Đối với ảnh kích thước $256 \times 256$, phương pháp này chỉ cần khoảng 10 lượt xử lý so với 28-50 bước của DiT.

### Vì sao dự đoán tỷ lệ kế tiếp vượt trội dự đoán token kế tiếp

Ba ưu thế cấu trúc mang tính quyết định:
1. **Từ thô đến mịn phù hợp với đặc tính thống kê tự nhiên của hình ảnh.** Thị giác con người và các bộ dữ liệu ảnh đều thể hiện tính quy luật phụ thuộc vào tỷ lệ độ phân giải: cấu trúc tần số thấp có tính ổn định và dễ dự đoán; chi tiết tần số cao có điều kiện phụ thuộc chặt chẽ vào nội dung tần số thấp. Dự đoán theo cấp tỷ lệ kế tiếp khai thác triệt để đặc tính này.
2. **Sinh song song bên trong mỗi cấp tỷ lệ.** Khác với cơ chế AR từng token kiểu GPT, VAR tạo ra toàn bộ token ở một cấp tỷ lệ chỉ trong một bước duy nhất. Chiều dài sinh hiệu dụng tăng theo thang logarit thay vì tuyến tính.
3. **Loại bỏ hoàn toàn thiên lệch thứ tự sinh.** Các token ở cấp $k$ quan sát toàn bộ bối cảnh của cấp $k-1$; không còn thiên kiến "bên trái" hay "phía trên" ép các token đầu tiên phải cam kết nội dung khi chưa có thông tin bối cảnh phía sau.

### Định luật tỷ lệ (Scaling Law)

Tian et al. đã chứng minh rằng VAR tuân theo đường cong mở rộng tỷ lệ dạng hàm mũ (power-law) đối với chỉ số FID trên ImageNet — hoàn toàn tương tự như cách GPT cải thiện độ phức tạp (perplexity). Tăng gấp đôi tham số hoặc năng lực tính toán sẽ giúp giảm một nửa sai số một cách đều đặn. Đây là mô hình sinh hình ảnh đầu tiên thể hiện hành vi mở rộng quy mô rõ nét và nhất quán ngang hàng với các mô hình ngôn ngữ lớn. Nhờ đó, hiệu năng của VAR theo quy mô trở nên dự đoán được dựa trên lượng tính toán, thay vì phụ thuộc vào các phỏng đoán thử-sai theo từng kiến trúc.

### Mối quan hệ với mô hình khuếch tán (Relationship to Diffusion)

VAR và mô hình khuếch tán cùng chia sẻ triết lý nén dữ liệu: cả hai đều chia nhỏ bài toán sinh phức tạp thành một chuỗi các bài toán con dễ giải hơn.

- Mô hình khuếch tán: bổ sung nhiễu từng bước một, học cách đảo ngược từng bước khử nhiễu.
- VAR: bổ sung độ phân giải từng bậc một, học cách dự đoán cấp tỷ lệ tiếp theo.

Đây là hai trục tư duy khác nhau để giải quyết cùng một vấn đề. Cả hai đều dẫn tới các phân phối có điều kiện khả thi. Về mặt thực nghiệm, VAR có tốc độ suy luận nhanh hơn (ít lượt truyền thẳng hơn, hoàn toàn song song trong từng cấp) và đạt chất lượng ngang ngửa hoặc vượt trội DiT trên bài toán ImageNet có điều kiện theo lớp. Việc tích hợp điều kiện văn bản vào VAR (như VARclip, HART) đang là một hướng nghiên cứu học thuật rất sôi động.

````figure
gx-var-next-scale
````

## Tự tay xây dựng (Build It)

Trong tệp `code/main.py`, bạn sẽ:
1. Xây dựng một **bộ tách token VQ đa tỷ lệ** siêu nhỏ trên dữ liệu "ảnh" tổng hợp (các vòng tròn Gaussian 2D).
2. Huấn luyện một **mô hình transformer dạng VAR** để dự đoán cấp tỷ lệ tiếp theo của các token.
3. Lấy mẫu sinh ảnh bằng cách gọi transformer 4 lần (tương ứng 4 cấp độ phân giải) và giải mã.
4. Xác minh rằng việc huấn luyện theo thứ bậc tỷ lệ giúp quá trình sinh bên trong mỗi cấp diễn ra song song hoàn toàn.

Đây là một cài đặt mô phỏng thực hành. Mục tiêu trọng tâm là quan sát trực tiếp mặt nạ chú ý có cấu trúc phân tầng và cơ chế sinh song song nội bộ cấp tỷ lệ hoạt động trong thực tế.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra tệp `outputs/skill-var-tokenizer-designer.md` — một kỹ năng chuyên biệt để thiết kế bộ tách token đa tỷ lệ: số lượng cấp độ phân giải, tỷ lệ tăng trưởng giữa các cấp, kích thước từ điển mã, chia sẻ phần dư và kiến trúc bộ giải mã.

## Bài tập thực hành (Exercises)

1. **Thử nghiệm loại trừ số lượng cấp tỷ lệ (Scale count ablation).** Huấn luyện VAR lần lượt với 4, 6, 8 và 10 cấp độ phân giải. Đo lường chất lượng tái tạo đối chiếu với số lượt truyền thẳng tự hồi quy. Nhiều cấp độ hơn = phần dư mịn hơn = chất lượng cao hơn nhưng cần nhiều lượt tính toán hơn.

2. **Kích thước từ điển mã (Codebook size).** Huấn luyện các bộ tokenizer với kích thước từ điển mã 512, 4096 và 16384. Từ điển mã lớn hơn cho chất lượng tái tạo tốt hơn nhưng bài toán dự đoán token sẽ khó hơn. Tìm điểm cân bằng tối ưu (knee point).

3. **Kiểm tra tính song song bên trong cấp tỷ lệ.** Với một mô hình VAR đã huấn luyện, hãy đo lường ma trận chú ý một cách tường minh. Bên trong cấp tỷ lệ $k$, mô hình có chú ý đến các vị trí ở các cấp khác mà không bị chặn lẫn nhau trong cùng một cấp hay không? Xác minh lại cài đặt mặt nạ chú ý.

4. **So sánh quy mô VAR và DiT.** Với cùng bài toán ImageNet có điều kiện theo lớp, hãy huấn luyện VAR và DiT ở cùng các mức ngân sách tham số (ví dụ: 33M, 130M, 458M). Vẽ biểu đồ quan hệ giữa FID và lượng tính toán (compute). VAR sẽ bứt phá vượt qua DiT ở từng mốc quy mô — hãy tái lập kết quả bài báo ở quy mô nhỏ.

5. **Điều kiện hóa bằng văn bản.** Mở rộng VAR để nhận thêm vector nhúng văn bản (từ tầng gộp của CLIP) làm tín hiệu điều kiện bổ sung thông qua khối adaLN. Đây chính là công thức thiết kế của HART. Chất lượng FID được cải thiện bao nhiêu khi lấy mẫu theo văn bản?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| VAR | "Mô hình tự hồi quy thị giác" | Tạo ảnh thông qua việc dự đoán cấp tỷ lệ tiếp theo trên kim tự tháp lưới token VQ. |
| Next-scale prediction | "Dự đoán từ thô đến mịn" | Mô hình dự đoán các token ở độ phân giải ngày càng tăng, có điều kiện dựa trên toàn bộ các cấp trước. |
| Multi-scale VQ tokenizer | "Bộ lượng tử hóa VQ đa tỷ lệ" | VQ-VAE xuất ra $K$ lưới token có độ phân giải tăng dần, với bộ giải mã cộng dồn tất cả các cấp. |
| Scale k | "Tầng tháp k" | Một trong $K$ mức độ phân giải, từ 1x1 tại $k=1$ lên đến (H/p)x(W/p) tại $k=K$. |
| Parallel-within-scale | "Một lượt truyền thẳng mỗi cấp" | Toàn bộ token ở cấp $k$ được dự đoán trong một lượt forward transformer duy nhất, không tuần tự. |
| Causal-across-scales | "Chú ý nhân quả theo cấp tỷ lệ" | Token ở cấp $k$ có thể chú ý tới toàn bộ các cấp $1 \ldots k$ nhưng không được nhìn thấy các cấp $k+1 \ldots K$. |
| Residual VQ | "Lượng tử hóa cộng dồn" | Token ở mỗi cấp mã hóa phần dư còn sót lại của các cấp thấp hơn; bộ giải mã tính tổng các vector nhúng. |
| VAR scaling law | "Quy luật mở rộng Image GPT" | Chỉ số FID tuân theo định luật hàm mũ dự đoán được theo lượng tính toán, giống như perplexity trong mô hình ngôn ngữ. |
| HART | "Mô hình lai VAR + văn bản" | Biến thể VAR có điều kiện văn bản kết hợp giải mã lặp kiểu MaskGIT với cấu trúc tỷ lệ của VAR. |
| Scale position embedding | "Bộ ba (scale, row, col)" | Mã hóa vị trí mang cả thông tin chỉ số cấp tỷ lệ lẫn tọa độ không gian bên trong cấp đó. |

## Tài liệu đọc thêm (Further Reading)

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — Bài báo gốc công bố VAR, tài liệu tham khảo quy chuẩn.
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, đường cơ sở so sánh của mô hình khuếch tán.
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, nền tảng họ tokenizer mà bộ tách token đa tỷ lệ của VAR mở rộng.
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, nền móng của kỹ thuật tách token hình ảnh rời rạc.
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — Kiến trúc VAR có điều kiện hóa theo văn bản.
