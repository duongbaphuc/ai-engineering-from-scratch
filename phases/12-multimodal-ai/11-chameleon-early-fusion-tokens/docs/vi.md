# Chameleon và các mô hình đa phương thức hợp nhất sớm chỉ dùng token

> Mọi mô hình ngôn ngữ - thị giác (Vision-Language Model - VLM) chúng ta từng nghiên cứu trước đây đều giữ hình ảnh và văn bản ở hai nhánh riêng biệt. Các token thị giác xuất phát từ một bộ mã hóa hình ảnh, đi qua bộ chiếu rồi mới hội ngộ cùng văn bản bên trong LLM. Không gian từ vựng thị giác và từ vựng văn bản hoàn toàn tách rời nhau. Chameleon (Meta, tháng 5 năm 2024) đã lật lại vấn đề: điều gì sẽ xảy ra nếu chúng dùng chung một từ vựng? Bằng cách huấn luyện một VQ-VAE để chuyển đổi hình ảnh thành một chuỗi các token rời rạc từ một từ vựng chia sẻ chung (shared vocabulary), mọi tài liệu đa phương thức giờ đây đều trở thành một chuỗi phẳng duy nhất — các token văn bản và token hình ảnh xen kẽ lẫn nhau, cùng được tối ưu hóa bằng một hàm mất mát tự hồi quy (autoregressive loss) duy nhất. Hệ quả tất yếu: mô hình có thể tự do tạo ra đầu ra đa phương thức hỗn hợp (mixed-modality output) — luân phiên sinh ra token văn bản và hình ảnh chỉ trong một lần gọi suy luận duy nhất. Bài học này phân tích luận điểm hợp nhất sớm (early-fusion) và tự tay xây dựng một phiên bản mô phỏng hoàn chỉnh từ đầu đến cuối.

**Type:** Xây dựng (Build)
**Languages:** Python (thư viện chuẩn, bộ tách token VQ-VAE + bộ giải mã xen kẽ)
**Prerequisites:** Giai đoạn 12 · 05, Giai đoạn 8 (AI tạo sinh)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích tại sao một không gian từ vựng dùng chung kết hợp với một hàm mất mát duy nhất lại làm thay đổi căn bản những gì mô hình có thể thực hiện.
- Mô tả cơ chế VQ-VAE lượng tử hóa một bức ảnh thành chuỗi rời rạc tương thích với mục tiêu dự đoán token tiếp theo (next-token prediction) của transformer.
- Điểm tên các giải pháp kỹ thuật đảm bảo tính ổn định khi huấn luyện của Chameleon: chuẩn hóa truy vấn-khóa QK-Norm, vị trí đặt dropout, và thứ tự các lớp LayerNorm.
- So sánh Chameleon với phương pháp tiếp cận Q-Former của BLIP-2 và chỉ ra bối cảnh ứng dụng tối ưu cho từng phương pháp.

## Vấn đề cần giải quyết

Các VLM dựa trên adapter (LLaVA, BLIP-2, Qwen-VL) đối xử với văn bản và hình ảnh như hai thực thể hoàn toàn dị biệt. Một token văn bản đi qua phép nhúng `embed(text_token)`; trong khi một hình ảnh phải đi qua `visual_encoder(image) → projector → ... pseudo_tokens`. Mô hình sở hữu hai luồng đầu vào riêng biệt chỉ hòa nhập vào nhau ở lưng chừng mạng.

Cách tiếp cận này dẫn tới ba hệ quả:

1. LLM chỉ có thể tiếp nhận và hiểu hình ảnh, hoàn toàn không thể sinh ra hình ảnh. Đầu ra bị giới hạn duy nhất ở định dạng văn bản.
2. Xử lý các tài liệu đa phương thức hỗn hợp (các đoạn văn bản và hình ảnh xuất hiện xen kẽ nhau, giống như một bài báo hoàn chỉnh) rất cồng kềnh — bạn buộc phải bóc tách dữ liệu đa phương thức bên ngoài mô hình hoặc phải ghép chuỗi nhiều lượt sinh độc lập.
3. Lệch pha phân phối biểu diễn (Distributional mismatch): Token thị giác và token văn bản cư ngụ ở những vùng không gian ẩn khác nhau, tạo ra các vấn đề căn chỉnh ngầm rất khó khắc phục.

Chameleon bác bỏ tiền đề chia tách đó: hình ảnh bản chất cũng chỉ là các chuỗi token rời rạc nằm trong một không gian từ vựng dùng chung. Huấn luyện mô hình trên các tài liệu xen kẽ với một hàm mất mát duy nhất trên một bộ giải mã tự hồi quy duy nhất, và bạn sẽ mở khóa được khả năng tạo ra dữ liệu đa phương thức hỗn hợp hoàn toàn tự nhiên.

## Khái niệm cốt lõi

### VQ-VAE đóng vai trò bộ tách token hình ảnh

Bộ tách token (tokenizer) là một bộ tự mã hóa biến phân lượng tử hóa vector (Vector-Quantized Variational Autoencoder - VQ-VAE). Kiến trúc bao gồm:

- **Bộ mã hóa (Encoder):** Kết hợp CNN + ViT giúp ánh xạ bức ảnh thành một bản đồ đặc trưng không gian (spatial feature map), ví dụ $32 \times 32$ đặc trưng có số chiều là 256.
- **Sách mã (Codebook):** Một tập từ vựng học được gồm $K$ vector (Chameleon sử dụng $K = 8192$), mỗi vector cũng có số chiều 256.
- **Phép lượng tử hóa (Quantization):** Đối với từng đặc trưng không gian, tra cứu phần tử gần nhất trong sách mã theo khoảng cách chuẩn L2. Thay thế vector đặc trưng liên tục đó bằng chỉ số nguyên (integer index) tương ứng trong sách mã.
- **Bộ giải mã (Decoder):** Mạng CNN nhận các đặc trưng đã lượng tử hóa và tái tạo trở lại thành các điểm ảnh (pixel).

Quá trình huấn luyện: tối ưu hóa hàm mất mát tái tạo VAE (reconstruction loss) + mất mát cam kết (commitment loss) + mất mát sách mã (codebook loss). Các chỉ số trong sách mã tạo thành một bảng chữ cái rời rạc đại diện cho hình ảnh.

Đối với Chameleon: một bức ảnh được chuyển thành $32 \times 32 = 1024$ token lấy từ bảng từ vựng 8192 phần tử. Các mã định danh (ID) hình ảnh này được đặt chung trong cùng một không gian ID với các token văn bản BPE. Bảng từ vựng thống nhất của Chameleon có 65.536 mục, và con số này đã bao gồm 8192 mã hình ảnh cùng các token đặc biệt như các thẻ phân cách hình ảnh. Khối transformer chỉ nhìn thấy một chuỗi dữ liệu duy nhất và tối ưu một hàm mất mát duy nhất.

### Không gian từ vựng dùng chung

Từ vựng của Chameleon tích hợp token văn bản, token hình ảnh và các ký hiệu phân cách phương thức (modality separators). Mỗi token mang một ID số nguyên duy nhất. Lớp nhúng đầu vào (input embedding) ánh xạ từng ID thành một vector ẩn $D$ chiều. Lớp chiếu đầu ra (output projection) ánh xạ từ không gian ẩn trở lại các giá trị logit của toàn bộ từ vựng. Hàm softmax sẽ lựa chọn token tiếp theo, bất kể token đó thuộc phương thức nào.

Các ký hiệu phân cách đóng vai trò sống còn: các thẻ `<image>` và `</image>` đóng khung chuỗi token hình ảnh. Ở thời điểm suy luận sinh dữ liệu, nếu mô hình phát ra thẻ `<image>`, phần mềm hạ nguồn sẽ hiểu rằng 1024 token tiếp theo là các chỉ số VQ cần được nạp vào bộ giải mã VQ-VAE để vẽ lại thành các pixel hoàn chỉnh.

### Khả năng sinh đa phương thức hỗn hợp

Quá trình suy luận thực chất là việc dự đoán token tiếp theo trong từ vựng chung. Giả sử nhận câu nhắc (prompt): "Vẽ một con mèo và mô tả về nó." Chameleon sẽ phát ra:

```text
<image> 4821 1029 2891 ... (1024 token hình ảnh) </image>
Con mèo có bộ lông màu cam, đang ngồi sưởi nắng bên bậu cửa sổ...
```

Mô hình tự quyết định thứ tự xuất hiện một cách hoàn toàn tự chủ — nó có thể sinh ảnh trước rồi tả sau, tả trước vẽ sau, hoặc xen kẽ liên tục. Cùng một bộ giải mã, cùng một cơ chế tối ưu.

Hãy so sánh điều này với các VLM dạng adapter vốn chỉ có thể sinh ra văn bản thuần túy. Chameleon đã mở lại cánh cửa về các phương thức đầu ra của mô hình nền tảng.

### Độ ổn định huấn luyện — QK-Norm, dropout, thứ tự LayerNorm

Huấn luyện mô hình hợp nhất sớm (early-fusion) ở quy mô lớn là cực kỳ thiếu ổn định. Bài báo của Chameleon đã ghi nhận ba thủ thuật cứu cánh:

- **QK-Norm:** Áp dụng lớp LayerNorm lên các vector truy vấn ($Q$) và khóa ($K$) bên trong cơ chế chú ý, trước khi thực hiện phép nhân vô hướng (dot product). Điều này ngăn chặn hiện tượng bùng nổ độ lớn logit (logit magnitude explosion) ở các tầng mạng sâu. Kỹ thuật này đã được nhiều mô hình lớn sau năm 2024 áp dụng rộng rãi.
- **Vị trí đặt Dropout:** Đặt Dropout ngay sau mỗi phép cộng phần dư (residual-add), thay vì chỉ đặt sau khối Attention và MLP. Cần có sự điều chuẩn (regularization) mạnh mẽ hơn khi các gradient xuất phát từ các token hình ảnh có xu hướng áp đảo toàn bộ hệ thống.
- **Thứ tự sắp xếp LayerNorm:** Áp dụng Pre-LN trên nhánh phần dư (chuẩn thông thường), bổ sung thêm một lớp LN phụ trên nhánh tắt (skip connection) của khối biến đổi cuối cùng. Điều này giúp ổn định dòng truyền gradient ở các tầng sát đầu ra.

Nếu thiếu vắng những kỹ thuật này, quá trình huấn luyện Chameleon ở phiên bản 34B tham số sẽ liên tục bị phân kỳ (diverge) tại nhiều điểm kiểm tra. Khi áp dụng đầy đủ, mô hình hội tụ ổn định. Quy trình huấn luyện này đóng góp giá trị học thuật không thua kém gì bản thân kiến trúc mô hình.

### Mức trần tái tạo của bộ tách token

VQ-VAE là một phép nén có tổn thất (lossy compression). Với kích thước sách mã 8192 và 1024 token cho một bức ảnh $512 \times 512$, tỷ số tín hiệu trên nhiễu cực đại (PSNR) của quá trình tái tạo chỉ đạt trần trong khoảng 26-28 dB. Mức chất lượng này đủ để tạo ra hình ảnh nhận diện được rõ ràng nhưng kém hơn rõ rệt so với các mô hình khuếch tán trong không gian liên tục (Stable Diffusion 3 đạt trên 32 dB).

Bộ tách token chính là điểm nghẽn hiệu năng của hệ thống. Những bộ tách token tân tiến hơn (như MAGVIT-v2, IBQ, SBER-MoVQGAN) sẽ nâng dần mức trần này lên. Emu3 (Bài học 12.12) đã đạt được chất lượng sinh ảnh tương đương SDXL chỉ nhờ vào việc cải tiến bộ tách token.

### So sánh Chameleon với BLIP-2 / LLaVA

**Chameleon (Hợp nhất sớm, từ vựng chia sẻ):**
- Một hàm mất mát duy nhất, một bộ giải mã duy nhất.
- Khả năng sinh dữ liệu đầu ra đa phương thức hỗn hợp.
- Chất lượng bị giới hạn bởi bộ tách token.
- Chi phí suy luận cao: cần chạy bộ giải mã VQ-VAE cho mỗi hình ảnh được sinh ra.

**BLIP-2 / LLaVA (Hợp nhất muộn, phân nhánh tháp riêng biệt):**
- Chỉ tiếp nhận hình ảnh đầu vào, chỉ phát ra văn bản ở đầu ra.
- Tái sử dụng trọn vẹn các LLM mạnh sẵn có.
- Không chịu rào cản từ bộ tách token đối với tác vụ hiểu ngữ nghĩa.
- Chi phí suy luận thấp: chỉ cần một lượt truyền xuôi tiêu chuẩn.

Lựa chọn dựa trên tác vụ: Nếu bạn cần khả năng sinh hình ảnh, hãy chọn họ Chameleon. Nếu bạn chỉ cần hiểu ngữ nghĩa thị giác, VLM dạng adapter đơn giản hơn nhiều và tái sử dụng được nguồn tài nguyên tính toán đã có từ các mô hình ngôn ngữ lớn.

### Fuyu và AnyGPT

Fuyu (Adept, 2023) là một hướng tiếp cận có liên quan: bỏ qua hoàn toàn bộ mã hóa thị giác riêng biệt, nạp trực tiếp các patch ảnh thô qua lớp chiếu đầu vào của LLM như thể chúng là các token, không cần bộ tách token rời rạc. Cách làm này đơn giản hơn Chameleon, nhưng đánh mất khả năng sinh hình ảnh từ không gian từ vựng chung.

AnyGPT (Zhan et al., 2024) mở rộng triết lý của Chameleon lên bốn phương thức: văn bản, hình ảnh, giọng nói và âm nhạc. Áp dụng cùng một kỹ thuật VQ-VAE cho từng phương thức, sử dụng chung một khối transformer duy nhất để sinh dữ liệu tùy biến qua lại giữa mọi phương thức (any-to-any). Chi tiết sẽ được làm rõ hơn trong Bài học 12.16.

```figure
vq-codebook
```

## Ứng dụng thực tế

Tệp `code/main.py` xây dựng một mô hình hợp nhất sớm tối giản hoàn chỉnh từ đầu đến cuối:

- Một bộ lượng tử hóa theo phong cách VQ-VAE mini giúp ánh xạ các patch kích thước $8 \times 8$ thành các chỉ số trong sách mã ($K=16$).
- Một không gian từ vựng chia sẻ chung gồm (ID văn bản từ 0 đến 31) + (ID hình ảnh từ 32 đến 47) + (thẻ phân cách 48, 49).
- Một bộ giải mã tự hồi quy mô phỏng (bảng tần suất bigram) được huấn luyện trên các cặp chú thích tổng hợp và chuỗi token hình ảnh.
- Vòng lặp lấy mẫu sinh chuỗi token văn bản và hình ảnh xen kẽ khi nhận câu nhắc đầu vào.

Mã nguồn cố tình giữ cho khối transformer ở mức tối giản (bảng bigram) để bạn có thể truy vết chính xác luồng tín hiệu di chuyển từ đầu vào tới đầu ra.

## Đóng gói triển khai

Bài học này tạo ra tệp artifact `outputs/skill-tokenizer-vs-adapter-picker.md`. Dựa trên thông số yêu cầu của sản phẩm (chỉ cần hiểu hay hiểu + sinh, chất lượng hình ảnh mong muốn, ngân sách tài nguyên), kỹ năng này sẽ giúp bạn quyết định giữa họ mô hình Chameleon (hợp nhất sớm) và họ LLaVA (hợp nhất muộn) kèm theo các quy tắc định lượng cụ thể để biện minh.

## Bài tập

1. Chameleon sử dụng sách mã $K=8192$ mục và 1024 token cho một bức ảnh $512 \times 512$. Hãy ước tính tỷ số nén dữ liệu so với một bức ảnh RGB 24-bit thông thường. Đây có phải phép nén có tổn thất không và tổn thất ở mức độ nào?

2. Một bức ảnh độ phân giải 4K ($3840 \times 2160$) với cùng mật độ nén VQ-VAE sẽ tạo ra bao nhiêu token hình ảnh? Liệu một mô hình theo phong cách Chameleon có thể sinh ra một bức ảnh 4K trong một lần gọi suy luận không? Yếu tố nào sẽ bị quá tải trước — chiều dài ngữ cảnh, chất lượng bộ tách token, hay bộ nhớ đệm KV cache?

3. Tự cài đặt cơ chế QK-Norm bằng Python thuần. Với vector truy vấn và khóa có số chiều là 64, hãy chỉ ra giá trị tích vô hướng trước và sau khi áp dụng LayerNorm. Tại sao việc kiểm soát biên độ lại mang tính sống còn ở các tầng mạng sâu?

4. Đọc Phần 2.3 trong bài báo Chameleon về độ ổn định huấn luyện. Mô tả chính xác hiện tượng lỗi mà nhóm nghiên cứu quan sát được ở quy mô 34B khi không sử dụng QK-Norm. Dấu hiệu của hiện tượng "bùng nổ chuẩn" (norm explosion) là gì?

5. Mở rộng bộ giải mã mô phỏng để phát ra phản hồi đa phương thức hỗn hợp khi nhận vào một câu nhắc văn bản thuần túy. Đo lường tần suất mô hình chọn sinh ảnh trước so với sinh văn bản trước khi phân phối dữ liệu huấn luyện là 60% văn bản trước / 40% ảnh trước.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Hợp nhất sớm (Early fusion) | "Token hợp nhất" | Hình ảnh được chuyển đổi thành các token rời rạc dùng chung không gian từ vựng với transformer ngay từ đầu |
| VQ-VAE | "Bộ tách token hình ảnh" | Mạng CNN + ViT + sách mã giúp ánh xạ hình ảnh thành các chỉ số nguyên mà transformer có thể dự đoán được |
| Không gian từ vựng chia sẻ (Shared vocabulary) | "Một từ điển chung" | Một không gian ID token duy nhất bao trùm cả văn bản, hình ảnh và các ký hiệu phân cách phương thức |
| QK-Norm | "Bộ ổn định chú ý" | Lớp LayerNorm áp dụng lên truy vấn và khóa trước phép nhân vô hướng, ngăn hiện tượng bùng nổ chuẩn logit |
| Sinh đa phương thức hỗn hợp (Mixed-modality generation) | "Đầu ra cả chữ lẫn ảnh" | Quá trình suy luận tự chủ phát ra xen kẽ các token văn bản và hình ảnh chỉ trong một lần chạy duy nhất |
| Quy mô sách mã (Codebook size) | "K phần tử" | Số lượng vector rời rạc mà VQ-VAE có thể lượng tử hóa; đánh đổi giữa tỷ số nén và độ trung thực của ảnh |
| Mức trần bộ tách token (Tokenizer ceiling) | "Giới hạn tái tạo" | Chỉ số PSNR tối đa có thể đạt được khi giải mã các token VQ; đặt ra giới hạn trần cho chất lượng ảnh sinh ra |

## Tài liệu tham khảo thêm

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
