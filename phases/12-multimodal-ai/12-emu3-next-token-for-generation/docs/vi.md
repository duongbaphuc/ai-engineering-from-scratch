# Emu3: Dự đoán token tiếp theo cho bài toán sinh hình ảnh và video

> Mô hình Emu3 của Viện Trí tuệ Nhân tạo Bắc Kinh (BAAI - Wang et al., tháng 9 năm 2024) là kết quả nghiên cứu năm 2024 đáng lẽ đã đặt dấu chấm hết cho cuộc tranh luận kéo dài giữa mô hình khuếch tán (diffusion) và mô hình tự hồi quy (autoregressive). Một mô hình transformer chỉ giải mã (decoder-only) kiểu LLaMA duy nhất, chỉ được huấn luyện bằng mục tiêu dự đoán token tiếp theo (next-token prediction - NTP), trên một không gian từ vựng thống nhất gồm token văn bản + token hình ảnh VQ + token video VQ 3D, đã đánh bại SDXL ở tác vụ sinh ảnh và vượt qua LLaVA-1.6 ở tác vụ nhận thức thị giác. Hoàn toàn không dùng mất mát căn chỉnh CLIP. Không cần lịch trình khuếch tán. Kỹ thuật hướng dẫn không cần bộ phân loại (classifier-free guidance - CFG) được áp dụng khi suy luận để tăng cường chất lượng, nhưng mục tiêu huấn luyện cốt lõi hoàn toàn là dự đoán token tiếp theo với cơ chế ép học từ đáp án đúng (teacher forcing). Nghiên cứu được công bố trên tạp chí Nature. Bài học này phân tích luận điểm của Emu3 — tại sao một bộ tách token tốt hơn kết hợp với mở rộng quy mô là tất cả những gì bạn cần — đồng thời so sánh đối chiếu với các phương pháp tiếp cận dựa trên mô hình khuếch tán.

**Type:** Học (Learn)
**Languages:** Python (thư viện chuẩn, toán học bộ tách token video 3D + khung sườn bộ lấy mẫu tự hồi quy)
**Prerequisites:** Giai đoạn 12 · 11 (Chameleon)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích tại sao mục tiêu dự đoán token tiếp theo với một hàm mất mát duy nhất của Emu3 lại hoạt động hiệu quả bất chấp định kiến lâu năm rằng mô hình khuếch tán là điều kiện bắt buộc để đạt chất lượng hình ảnh cao.
- Mô tả bộ tách token video 3D: cấu trúc của một sách mã VQ không-thời gian (spatiotemporal VQ codebook) và lý do các patch lại trải dài theo trục thời gian.
- So sánh Emu3 với Stable Diffusion XL trên các khía cạnh: chi phí tính toán khi huấn luyện, chi phí suy luận, và mức trần chất lượng hình ảnh.
- Gọi tên ba vai trò chức năng mà cùng một mô hình Emu3 duy nhất đảm nhận: Emu3-Gen (sinh hình ảnh), Emu3-Chat (nhận thức thị giác / hội thoại), và Emu3-Stage2 (sinh video).

## Vấn đề cần giải quyết

Quan niệm phổ biến trong cộng đồng học thuật xuyên suốt năm 2024: sinh hình ảnh bắt buộc phải dùng mô hình khuếch tán (diffusion). Lập luận đưa ra là: các token hình ảnh rời rạc đánh mất quá nhiều thông tin chi tiết để có thể tái tạo độ sắc nét, và quá trình lấy mẫu tự hồi quy sẽ tích lũy sai số dần qua hàng nghìn bước sinh token. Stable Diffusion, DALL-E 3, Imagen, Midjourney đều dựa trên một biến thể nào đó của mô hình khuếch tán. Chameleon (Bài học 12.11) đã bác bỏ một phần giả định này ở quy mô nhỏ nhưng chưa thể sánh ngang chất lượng với SDXL.

Emu3 tấn công trực diện vào luận điểm đó. Tuyên bố cốt lõi: một bộ tách token thị giác tốt hơn + quy mô tham số và dữ liệu đủ lớn + hàm mất mát dự đoán token tiếp theo = khả năng sinh ảnh vượt trội mô hình khuếch tán, tích hợp bên trong cùng một mô hình có khả năng hiểu và nhận thức thị giác xuất sắc.

Nhận định này từng gây tranh cãi gay gắt khi mới công bố. Nhưng hai năm sau, gia đình các mô hình sinh hợp nhất mã nguồn mở (Emu3, Show-o, Janus-Pro, Transfusion) đã trở thành lộ trình nghiên cứu mặc định; và các mô hình thương mại hàng đầu dường như cũng đang vận hành theo một biến thể tương đương.

## Khái niệm cốt lõi

### Bộ tách token của Emu3

Thành phần mang tính quyết định là bộ tách token thị giác. Emu3 tự huấn luyện một bộ tách token tùy biến thuộc lớp IBQ (Inverse Bottleneck Quantizer, thuộc họ SBER-MoVQGAN) với hệ số giảm độ phân giải không gian $8 \times 8$ cho mỗi token. Một bức ảnh kích thước $512 \times 512$ sẽ được nén thành $64 \times 64 = 4096$ token với quy mô sách mã lên tới 32.768 phần tử.

Quy mô này lớn hơn nhiều so với Chameleon (vốn chỉ dùng 1024 token cho ảnh $512 \times 512$ tại $K=8192$) trên cả hai trục: số lượng token gấp 4 lần và dung lượng sách mã cũng gấp 4 lần. Năng lực biểu diễn mở rộng này giúp nâng tầm chất lượng tái tạo. Thước đo quyết định: chỉ số PSNR tái tạo đạt 30,5 dB, tiệm cận không gian tiềm ẩn liên tục của Stable Diffusion ở mức 32 dB.

Đối với video: một bộ tách token VQ 3D sẽ mã hóa một khối patch không-thời gian ($4 \text{ khung hình} \times 8 \times 8 \text{ pixel}$) thành một số nguyên duy nhất. Một đoạn clip dài 4 giây ở mức 8 FPS gồm 32 khung hình; tại độ phân giải $256 \times 256$ với hệ số nén không gian 8x và thời gian 4x, tổng số lượng token tiêu tốn là:
$(256/8) \times (256/8) \times (32/4) = 32 \times 32 \times 8 = 8.192$ token.

Chất lượng của bộ tách token đặt ra giới hạn trần cho toàn bộ hệ thống. Đóng góp lớn của Emu3 nằm ở nhận định: "chúng tôi đã huấn luyện thành công một bộ tách token cực kỳ xuất sắc."

### Huấn luyện với một hàm mất mát duy nhất

Emu3 áp dụng duy nhất một mục tiêu tối ưu: dự đoán token tiếp theo trên một không gian từ vựng chia sẻ chung bao gồm token văn bản, token hình ảnh 2D và token video 3D. Trọng số mất mát được nhân với các hệ số đặc thù theo từng phương thức trong quá trình huấn luyện nhằm cân bằng sự đóng góp gradient, nhưng bản thân hàm mất mát toán học là hoàn toàn đồng nhất.

Dữ liệu huấn luyện được pha trộn từ:
- Sinh hình ảnh: `<text caption> <image> image_tokens </image>`
- Nhận thức hình ảnh: `<image> image_tokens </image> <question> text_tokens`
- Sinh video: `<text caption> <video> video_tokens </video>`
- Nhận thức video: tương tự như trên.
- Văn bản thuần túy: hàm NTP chuẩn.

Mô hình tự động học được thời điểm phát ra token hình ảnh so với token văn bản dựa trên phân phối dữ liệu. Năng lực tạo sinh tự nhiên hình thành khi mô hình dự đoán chuỗi token hình ảnh nối tiếp ngay sau thẻ phân cách `<image>`.

### Hướng dẫn không cần bộ phân loại (CFG) và nhiệt độ lấy mẫu

Quá trình sinh hình ảnh tự hồi quy đạt chất lượng vượt bậc khi tích hợp kỹ thuật hướng dẫn không cần bộ phân loại (Classifier-Free Guidance - CFG) ở giai đoạn suy luận. Emu3 áp dụng kỹ thuật này: thực hiện sinh hai luồng song song, một luồng có câu mô tả đầy đủ và một luồng bỏ trống mô tả, sau đó trộn các giá trị logit với hệ số hướng dẫn (thường từ 3.0 đến 7.0). Đây chính là thủ thuật CFG kinh điển mượn từ các mô hình khuếch tán sang cho môi trường tự hồi quy.

Nhiệt độ lấy mẫu (temperature) đóng vai trò then chốt: nếu đặt quá cao sẽ sinh ra các dị vật thị giác (artifacts); nếu đặt quá thấp sẽ gây ra hiện tượng sụp đổ chế độ (mode collapse). Mức nhiệt độ khuyến nghị của Emu3 là 1.0 cho các tác vụ nhận thức thị giác và 0.8 cho tác vụ sinh hình ảnh.

### Ba vai trò, một mô hình duy nhất

Emu3 được phát hành dưới dạng ba giao diện API có chức năng riêng biệt nhưng chia sẻ chung một bộ trọng số duy nhất bên dưới:

- **Emu3-Gen:** Chuyên sinh hình ảnh. Đầu vào là văn bản, đầu ra là các token hình ảnh.
- **Emu3-Chat:** Hỏi đáp thị giác (VQA) và tạo chú thích ảnh. Đầu vào là hình ảnh (dưới dạng token), đầu ra là văn bản.
- **Emu3-Stage2:** Sinh video và hỏi đáp video. Đầu vào là văn bản hoặc video, đầu ra là văn bản hoặc chuỗi video.

Không cần các đầu mạng chuyên biệt (task-specific heads). Chỉ đơn giản là áp dụng các mẫu câu nhắc (prompt templates) khác nhau trên cùng một checkpoint mô hình.

### Kết quả benchmark

Trích dẫn từ bài báo khoa học của Emu3 (Tháng 9 năm 2024):

- **Sinh hình ảnh:** Đánh bại SDXL trên MJHQ-30K FID (5,4 so với 5,6), hòa điểm thống kê trên GenEval tổng thể (0,54 so với 0,55), và ngang ngửa trên chỉ số tổng hợp Deep-Eval.
- **Nhận thức hình ảnh:** Kém hơn LLaVA-1.6 trên VQAv2 (75,1 so với 81,8) và MMMU (31,6 so với 35,1), nhưng đánh bại đối thủ trên SEED-Bench, OCRBench và ChartQA.
- **Sinh video:** Tạo ra các đoạn clip 4 giây đạt chất lượng cạnh tranh về chỉ số FVD so với các mô hình công khai cùng thời kỳ Sora.

Các con số không phải lúc nào cũng chiến thắng tuyệt đối — Emu3 đánh đổi một vài điểm ở tác vụ này để lấy điểm ở tác vụ khác — nhưng tuyên bố "dự đoán token tiếp theo là tất cả những gì bạn cần" đã hoàn toàn đứng vững trên mọi phương thức.

### Chi phí tính toán

Emu3 là mô hình có quy mô 8 tỷ tham số (8B). Bài báo không công bố tổng số giờ GPU tiêu tốn; để dễ hình dung, quá trình tiền huấn luyện Llama-2-7B tiêu tốn khoảng 184.000 giờ GPU A100 (tương đương 21 năm GPU). Các mô hình khuếch tán như Stable Diffusion 3 có ngân sách huấn luyện tương đương nhưng đòi hỏi các bộ mã hóa văn bản độc lập và các đường ống xử lý phức tạp hơn nhiều.

Ở khâu suy luận, Emu3 sinh ảnh chậm hơn SDXL: để sinh 4096 token hình ảnh ở tốc độ 30 token/giây phải mất khoảng ~2 phút cho một bức ảnh $512 \times 512$, trong khi SDXL chỉ mất 2-5 giây. Kỹ thuật suy luận suy đoán (speculative decoding) và tối ưu hóa bộ nhớ đệm KV cache có thể thu hẹp khoảng cách này nhưng không thể san bằng hoàn toàn. Khối lượng tính toán khổng lồ khi sinh tự hồi quy là một sự đánh đổi hiển nhiên.

### Ý nghĩa đột phá

Đóng góp sâu sắc nhất của Emu3 mang tính khái niệm. Nếu mục tiêu dự đoán token tiếp theo có thể mở rộng quy mô để bắt kịp mô hình khuếch tán trong bài toán sinh hình ảnh, thì lộ trình xây dựng mô hình hợp nhất (một hàm mất mát, một xương sống duy nhất, xử lý mọi phương thức) là hoàn toàn khả thi. Các mô hình tương lai sẽ không cần các bộ mã hóa văn bản riêng rẽ, không cần bộ lập lịch khuếch tán rời, không cần các kiến trúc VAE phức tạp. Chỉ cần một khối transformer, một bộ tách token cho mỗi phương thức, và mở rộng quy mô tính toán.

Show-o, Janus-Pro và InternVL-U đều đang kế thừa hoặc tiếp tục phát triển luận điểm này. Các phòng thí nghiệm tại Trung Quốc (BAAI, DeepSeek) đã đẩy mạnh hướng nghiên cứu này một cách quyết liệt xuyên suốt năm 2025.

```figure
l5-emu3-next-token
```

## Ứng dụng thực tế

Tệp `code/main.py` xây dựng hai module mô phỏng:

- Bộ tính toán lượng token cho bộ tách token VQ 2D và 3D: dựa trên bộ thông số (độ phân giải, kích thước patch, độ dài video, FPS), tính toán chính xác số lượng token tiêu thụ cho hình ảnh so với video.
- Khung sườn bộ lấy mẫu token hình ảnh tự hồi quy có tích hợp kỹ thuật hướng dẫn không cần bộ phân loại (CFG) và nhiệt độ.

Phần cài đặt CFG bám sát công thức của Emu3 — phối hợp các logit có điều kiện và không điều kiện thông qua hệ số hướng dẫn $\gamma$.

## Đóng gói triển khai

Bài học này tạo ra tệp artifact `outputs/skill-token-gen-cost-analyzer.md`. Dựa trên bảng đặc tả sản phẩm tạo sinh (hình ảnh hay video, độ phân giải mục tiêu, yêu cầu chất lượng, ngân sách độ trễ), kỹ năng này sẽ tính toán số lượng token, chi phí suy luận và đưa ra khuyến nghị lựa chọn giữa họ mô hình Emu3 hay mô hình khuếch tán.

## Bài tập

1. Emu3 sinh ra 4096 token cho một bức ảnh $512 \times 512$ với hệ số nén $8 \times 8$. Hãy tính số lượng token tương đương cho ảnh $1024 \times 1024$ và $2048 \times 2048$. Điều gì sẽ xảy ra với độ trễ suy luận của hệ thống?

2. Đọc Phần 3.3 của bài báo Emu3 về bộ tách token video. Mô tả hình dạng khối patch VQ 3D và giải thích lý do tại sao nó lại bao trùm $4 \text{ khung hình} \times 8 \times 8 \text{ pixel}$ thay vì chỉ xử lý trên từng khung hình đơn lẻ.

3. So sánh hiệu ứng thị giác khi đặt trọng số hướng dẫn CFG ở mức 5.0 so với 3.0. Truy vết công thức toán học cài đặt trong `code/main.py`.

4. Tính toán số phép tính FLOP huấn luyện cho Emu3 (8B) với giả định kích thước dữ liệu là 300 tỷ token và so sánh với Stable Diffusion 3. Mô hình nào tiêu tốn chi phí huấn luyện lớn hơn?

5. Emu3 đánh bại SDXL trên chỉ số FID nhưng lại xếp sau trên benchmark VQAv2 so với các VLM chuyên biệt. Giải thích lý do tại sao phương pháp tiếp cận hàm mất mát thống nhất lại bộc lộ những điểm mạnh và điểm yếu khác nhau trên các bộ đánh giá khi so sánh với các mô hình chuyên biệt hóa.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Dự đoán token tiếp theo (Next-token prediction) | "NTP" | Hàm mất mát tự hồi quy tiêu chuẩn: dự đoán token[i+1] dựa trên token[0..i]; áp dụng hiệu quả cho mọi phương thức sau khi đã tách token |
| Bộ tách token IBQ (IBQ tokenizer) | "Lượng tử hóa nút thắt cổ chai nghịch đảo" | Một lớp VQ-VAE với sách mã quy mô lớn (32.768+) mang lại chất lượng tái tạo vượt trội so với Chameleon |
| VQ 3D (3D VQ) | "Lượng tử hóa không-thời gian" | Sách mã được đánh chỉ số theo (thời gian, hàng, cột); một token đại diện cho một khối 4 khung hình $\times 8 \times 8$ pixel |
| Hướng dẫn không cần bộ phân loại (Classifier-free guidance) | "CFG" | Pha trộn logit có điều kiện và không điều kiện theo trọng số $\gamma$; giúp tăng vọt độ trung thực hình ảnh khi suy luận |
| Từ vựng thống nhất (Unified vocabulary) | "Token dùng chung" | Văn bản + hình ảnh + video cùng lấy mã từ một không gian số nguyên duy nhất; mô hình tự động dự đoán phương thức tiếp theo |
| MJHQ-30K | "Benchmark sinh ảnh" | Bộ benchmark đánh giá hình ảnh chất lượng chuẩn Midjourney với 30.000 câu nhắc; Emu3 báo cáo chỉ số FID tại đây |

## Tài liệu tham khảo thêm

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
