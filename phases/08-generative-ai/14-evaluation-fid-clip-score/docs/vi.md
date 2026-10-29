# Đánh giá mô hình sinh — FID, Điểm CLIP, Sở thích con người (Evaluation — FID, CLIP Score, Human Preference)

> Mọi bảng xếp hạng mô hình sinh (generative model) đều trích dẫn các chỉ số FID, điểm CLIP (CLIP score) và tỷ lệ thắng từ các đấu trường đánh giá sở thích con người (human-preference arena). Tuy nhiên, mỗi con số đều tồn tại những kẽ hở (failure modes) mà một nhà nghiên cứu có chủ đích hoàn toàn có thể "lách luật" (game). Nếu không thấu hiểu các kẽ hở này, bạn sẽ không thể phân biệt được đâu là một bước tiến công nghệ thực chất và đâu chỉ là một kết quả được tối ưu hóa ảo.

**Type:** Xây dựng (Build)  
**Languages:** Python  
**Prerequisites:** Phase 8 · 01 (Phân loại mô hình sinh - Taxonomy), Phase 2 · 04 (Các chỉ số đánh giá - Evaluation Metrics)  
**Time:** ~45 phút  

## Vấn đề đặt ra (The Problem)

Một mô hình sinh được đánh giá dựa trên hai yếu tố: *chất lượng mẫu sinh (sample quality)* và *mức độ tuân thủ điều kiện (conditioning adherence)*. Cả hai yếu tố này đều không có công thức giải tích đóng (closed-form measure). Mô hình của bạn phải kết xuất hàng chục nghìn hình ảnh; một công cụ nào đó phải gán điểm số cho chúng; và bạn phải có cơ sở để tin tưởng các con số đó giữa các họ mô hình, độ phân giải và kiến trúc khác nhau. Ba thước đo chính đã vượt qua thử thách thời gian trong giai đoạn 2014-2026:

- **FID (Khoảng cách Fréchet Inception - Fréchet Inception Distance).** Đo khoảng cách giữa hai phân phối — dữ liệu thật và dữ liệu sinh ra — trong không gian đặc trưng của mạng Inception. Chỉ số càng thấp càng tốt.
- **Điểm CLIP (CLIP score).** Độ tương đồng cosine giữa vector nhúng ảnh của CLIP cho ảnh sinh ra và vector nhúng văn bản của CLIP cho câu lệnh gợi ý (prompt). Điểm càng cao càng tốt. Thước đo này đánh giá mức độ bám sát câu lệnh.
- **Sở thích con người (Human preference).** Đặt hai mô hình đối đầu trực tiếp trên cùng một câu lệnh gợi ý, nhờ con người (hoặc một mô hình tương đương GPT-4) chọn ra bức ảnh đẹp hơn, sau đó tổng hợp thành điểm số xếp hạng Elo.

Bên cạnh đó, bạn cũng sẽ bắt gặp: IS (Inception Score, hiện hầu như đã ngừng sử dụng), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Mỗi chỉ số này sinh ra để khắc phục một điểm yếu của các chỉ số trước đó.

## Khái niệm cốt lõi (The Concept)

![FID, CLIP và sở thích con người: ba trục đo lường, các cơ chế sai lệch khác nhau](../assets/evaluation.svg)

### FID — Chất lượng mẫu sinh (FID — sample quality)

Phương pháp của Heusel et al. (2017) gồm các bước:

1. Trích xuất đặc trưng Inception-v3 (vector 2048 chiều) cho $N$ ảnh thật và $N$ ảnh sinh ra.
2. Khớp phân phối chuẩn đa biến Gaussian cho từng tập: tính vector trung bình $\mu_r, \mu_g$ và ma trận hiệp phương sai $\Sigma_r, \Sigma_g$.
3. $\text{FID} = \lVert \mu_r - \mu_g \rVert^2 + \mathop{\mathrm{Tr}}\left(\Sigma_r + \Sigma_g - 2 \cdot (\Sigma_r \cdot \Sigma_g)^{0.5}\right)$.

Ý nghĩa: Đo khoảng cách Fréchet giữa hai phân phối chuẩn đa biến trong không gian đặc trưng. Giá trị càng nhỏ thể hiện hai phân phối càng tương đồng.

Các kẽ hở và sai lệch:
- **Thiên lệch khi $N$ nhỏ.** FID tính toán kỳ vọng sai số bình phương trên phân phối đặc trưng — nếu $N$ nhỏ, mô hình sẽ ước lượng thấp ma trận hiệp phương sai, dẫn đến điểm FID thấp giả tạo. Luôn phải dùng cỡ mẫu $N \ge 10{,}000$.
- **Phụ thuộc vào mạng Inception.** Mạng Inception-v3 được huấn luyện trên ImageNet. Các miền dữ liệu cách xa ImageNet (khuôn mặt, hội họa, hình ảnh chứa văn bản) sẽ tạo ra điểm FID vô nghĩa. Hãy sử dụng bộ trích xuất đặc trưng chuyên biệt cho từng miền.
- **Tối ưu hóa ảo (Gaming).** Quá khớp vào phân phối tiên nghiệm của Inception có thể hạ thấp FID mà không hề cải thiện chất lượng thị giác thực tế. Chỉ số CMMD (bên dưới) khắc phục được nhược điểm này.

### Điểm CLIP — Mức độ bám sát câu lệnh gợi ý (CLIP score — prompt adherence)

Phương pháp của Radford et al. (2021). Với một ảnh sinh ra kèm câu lệnh gợi ý:

```text
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Lấy giá trị trung bình trên 30.000 ảnh sinh ra → ta thu được một giá trị vô hướng để so sánh công bằng giữa các mô hình.

Các kẽ hở và sai lệch:
- **Điểm mù nội tại của CLIP.** CLIP có năng lực suy luận thành phần (compositional reasoning) khá yếu (ví dụ: yêu cầu "một khối lập phương đỏ nằm trên một khối cầu xanh" thường bị nhận diện sai). Mô hình có thể đạt điểm CLIP cao mà không thực sự tuân thủ các câu lệnh phức tạp.
- **Thiên vị câu lệnh ngắn.** Các câu lệnh ngắn có nhiều hình ảnh khớp tương tự trong dữ liệu tự nhiên hơn. Về mặt cơ chế, câu lệnh càng dài thì điểm CLIP càng có xu hướng giảm.
- **Lách luật bằng từ khóa.** Việc chèn thêm các từ sáo rỗng như "high quality, 4k, masterpiece" vào prompt sẽ thổi phồng điểm CLIP nhưng không hề cải thiện sự liên kết giữa nội dung ảnh và ngữ nghĩa văn bản.

Chỉ số CMMD (Jayasumana et al., 2024) khắc phục các hạn chế này: sử dụng đặc trưng CLIP thay vì Inception, áp dụng độ lệch trung bình cực đại (maximum-mean discrepancy - MMD) thay cho khoảng cách Fréchet. Nhờ vậy, CMMD phát hiện các khác biệt tinh tế về chất lượng chuẩn xác hơn.

### Sở thích con người — Chuẩn mực chân thực tối hậu (Human preference — the ground truth)

Chọn một tập hợp các câu lệnh gợi ý chuẩn. Tạo ảnh bằng cả mô hình A và mô hình B. Hiển thị từng cặp ảnh cho người đánh giá (hoặc một mô hình LLM đóng vai trò giám khảo mạnh). Tổng hợp kết quả thắng thua thành điểm số xếp hạng Elo hoặc Bradley-Terry. Các bộ tiêu chuẩn phổ biến:

- **PartiPrompts (Google)**: 1.600 câu lệnh gợi ý đa dạng thuộc 12 danh mục.
- **HPSv2**: 107 nghìn nhãn đánh giá từ con người, được dùng rộng rãi làm đại diện tự động.
- **ImageReward**: 137 nghìn cặp sở thích ảnh-câu lệnh, cấp phép mã nguồn mở MIT.
- **PickScore**: được huấn luyện trên 2,6 triệu lượt đánh giá sở thích từ Pick-a-Pic.
- **Các đấu trường hình ảnh dạng Chatbot Arena**: như https://imagearena.ai/ và các nền tảng tương đương.

Các kẽ hở và sai lệch:
- **Độ phân tán của người đánh giá.** Người dùng phổ thông có gu thẩm mỹ khác với các chuyên gia thiết kế đồ họa. Cần kết hợp cả hai nhóm đối tượng.
- **Phân phối câu lệnh gợi ý.** Việc chọn lọc câu lệnh có chủ ý (cherry-picking) sẽ tạo lợi thế cho một họ mô hình nhất định. Luôn phải công khai bộ câu lệnh kiểm thử.
- **Hiện tượng hack phần thưởng của giám khảo LLM.** Giám khảo GPT-4 dễ bị đánh lừa bởi những bức ảnh trông bắt mắt bề ngoài nhưng sai lệch nội dung bên trong. Cần đối chiếu chéo với đánh giá từ con người.

## Sử dụng phối hợp đa chỉ số (Use together)

Một báo cáo đánh giá chất lượng mô hình trong môi trường sản xuất thực tế bắt buộc phải bao gồm:

1. Chỉ số FID trên 10k-30k mẫu đối chiếu với tập dữ liệu thật được giữ lại (đo chất lượng mẫu).
2. Điểm CLIP / CMMD trên cùng các mẫu đó đối chiếu với câu lệnh gợi ý (đo độ bám sát yêu cầu).
3. Tỷ lệ thắng (win rate) trong đấu trường ẩn danh so với mô hình tiền nhiệm (đo sở thích tổng thể).
4. Phân tích các trường hợp lỗi: 50 mẫu được chọn ngẫu nhiên, gắn cờ kiểm tra các lỗi kinh điển (giải phẫu bàn tay, kết xuất chữ viết, số lượng vật thể nhất quán).

Dựa vào một chỉ số đơn lẻ là tự đánh lừa mình. Chỉ khi cả ba chỉ số cùng hội tụ đi kèm đánh giá định tính thì tuyên bố về hiệu năng mới có giá trị thực tiễn.

````figure
gx-fid-distributions
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` cài đặt FID, bộ tính điểm tương tự CLIP và thuật toán cập nhật điểm Elo trên các "vector đặc trưng" tổng hợp (chúng ta sử dụng vector 4 chiều để mô phỏng các vector đặc trưng Inception). Bạn sẽ quan sát thấy:

- Phép tính FID trên tập mẫu nhỏ $N$ so với tập lớn $N$ — sự khác biệt do thiên lệch thống kê.
- "Điểm CLIP" dưới dạng độ tương đồng cosine giữa các tập vector đặc trưng.
- Quy tắc cập nhật điểm Elo từ một luồng đánh giá sở thích mô phỏng.

### Bước 1: Tính toán FID trong bốn dòng lệnh (Step 1: FID in four lines)

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Bước 2: Độ tương đồng Cosine tương tự CLIP (Step 2: CLIP-style cosine-similarity)

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Bước 3: Cập nhật điểm xếp hạng Elo (Step 3: Elo aggregation)

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Cạm bẫy thực tế (Pitfalls)

- **Đo FID ở cỡ mẫu $N=1000$.** Phương pháp phỏng đoán này không đáng tin cậy nếu $N < 10{,}000$. Các bài báo công bố điểm FID với cỡ mẫu nhỏ thường là kết quả tối ưu ảo.
- **So sánh FID giữa các độ phân giải khác nhau.** Thao tác đổi kích thước ảnh về $299 \times 299$ của Inception sẽ làm thay đổi phân phối đặc trưng. Chỉ so sánh FID giữa các mô hình ở cùng độ phân giải.
- **Chỉ báo cáo kết quả của một giá trị seed duy nhất.** Cần chạy tối thiểu 3 seed ngẫu nhiên khác nhau và báo cáo độ lệch chuẩn (std).
- **Thổi phồng điểm CLIP qua câu lệnh phủ định (negative prompts).** Một số quy trình cố tình đẩy điểm CLIP bằng cách tinh chỉnh ép prompt quá mức. Hãy kiểm tra xem ảnh có bị hiện tượng bão hòa màu hay méo mó thị giác hay không.
- **Thiên vị điểm Elo do trùng lặp câu lệnh.** Nếu cả hai mô hình đều đã thấy các câu lệnh kiểm thử trong quá trình huấn luyện, điểm Elo sẽ mất giá trị. Luôn sử dụng tập câu lệnh kiểm thử độc lập hoàn toàn.
- **Độ lệch của nhóm người đánh giá trả phí.** Người đánh giá trên Prolific hay MTurk thường có xu hướng trẻ tuổi và am hiểu công nghệ hơn mức bình thường. Cần phối hợp thêm nhóm chuyên gia nghệ thuật và đồ họa chuyên nghiệp.

## Ứng dụng thực tế (Use It)

Quy chuẩn đánh giá trong môi trường sản xuất năm 2026:

| Trụ cột | Yêu cầu tối thiểu | Khuyến nghị chuyên sâu |
|---------|-------------------|------------------------|
| Chất lượng mẫu | FID trên 10k mẫu đối chiếu tập dữ liệu thật | + CMMD trên 5k mẫu + FID riêng theo từng danh mục |
| Mức độ bám sát câu lệnh | Điểm CLIP trên 30k mẫu | + HPSv2 + ImageReward + Trả lời câu hỏi thị giác (VQA) |
| Sở thích người dùng | 200 cặp đánh giá mù so với mô hình mốc | + 2000 cặp đánh giá kết hợp người thật + giám khảo LLM + Chatbot Arena |
| Phân tích lỗi | 50 trường hợp kiểm tra thủ công | 500 trường hợp kiểm tra thủ công + bộ phân loại an toàn tự động |

Báo cáo hội tụ đủ cả bốn trụ cột = khẳng định khoa học vững chắc. Báo cáo chỉ dựa trên một trụ cột đơn lẻ = chiêu trò tiếp thị.

## Đóng gói sản phẩm (Ship It)

Lưu tệp `outputs/skill-eval-report.md`. Kỹ năng này nhận vào một checkpoint mô hình mới kèm theo mô hình mốc đối chứng và xuất ra kế hoạch đánh giá toàn diện: quy mô mẫu, danh sách các chỉ số cần đo, bộ kiểm tra các trường hợp lỗi biên và tiêu chuẩn nghiệm thu trước khi triển khai.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py`. So sánh giá trị FID thu được ở quy mô $N=100$ so với $N=1000$ trên cùng một phân phối tổng hợp. Báo cáo mức độ thiên lệch của phép đo.
2. **Trung bình.** Cài đặt chỉ số CMMD dựa trên các vector đặc trưng mô phỏng CLIP (tham khảo công thức trong bài báo của Jayasumana et al., 2024). So sánh độ nhạy của CMMD đối với các biến đổi chất lượng so với chỉ số FID truyền thống.
3. **Khó.** Tái lập quy trình đánh giá HPSv2: lấy 1000 cặp ảnh-câu lệnh từ tập con của Pick-a-Pic, tinh chỉnh một mô hình chấm điểm nhỏ dựa trên CLIP theo dữ liệu sở thích, và đo lường mức độ đồng thuận của nó trên một tập dữ liệu kiểm thử độc lập.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| FID | "Khoảng cách Fréchet Inception" | Khoảng cách Fréchet giữa các phân phối chuẩn khớp với đặc trưng Inception của ảnh thật và ảnh sinh. |
| CLIP score | "Độ tương đồng ảnh-văn bản" | Độ tương đồng cosine giữa vector nhúng ảnh và vector nhúng văn bản của mô hình CLIP. |
| CMMD | "Phương pháp kế thừa FID" | Khoảng cách MMD tính trên đặc trưng CLIP; ít thiên lệch hơn, không giả định phân phối Gaussian. |
| IS | "Điểm Inception" | Tính bằng $\exp(\mathbb{E}[\text{KL}(p(y|x) \parallel p(y))])$; tương quan rất kém với các mô hình hiện đại, đã bị loại bỏ. |
| HPSv2 / ImageReward / PickScore | "Đại diện sở thích tự động" | Các mô hình nhỏ được huấn luyện trên dữ liệu sở thích của con người; đóng vai trò giám khảo chấm điểm tự động. |
| Elo | "Hệ số xếp hạng cờ vua" | Mô hình tổng hợp các trận thắng-thua đối đầu trực tiếp dựa trên phân phối Bradley-Terry. |
| PartiPrompts | "Bộ prompt chuẩn mực" | Bộ 1.600 câu lệnh gợi ý do Google chọn lọc kỹ lưỡng trên 12 danh mục thử nghiệm khác nhau. |
| FD-DINO | "Giải pháp thay thế tự giám sát" | Khoảng cách Fréchet sử dụng đặc trưng DINOv2; hoạt động vượt trội trên các miền ảnh ngoài ImageNet. |

## Lưu ý triển khai: Đánh giá cũng là một khối lượng công việc suy luận nặng nề (Production note: evaluation is an inference workload too)

Chạy kiểm thử FID trên 10 nghìn mẫu đồng nghĩa với việc bạn phải kết xuất 10 nghìn hình ảnh hoàn chỉnh. Với một mô hình SDXL base chạy 50 bước ở độ phân giải $1024^2$ trên một GPU NVIDIA L4 duy nhất, quá trình này sẽ tiêu tốn khoảng ~11 giờ suy luận đơn lẻ liên tục. Ngân sách dành cho đánh giá là chi phí thực tế rất lớn, và bản chất của nó tương tự như kịch bản suy luận ngoại tuyến theo lô (tối đa hóa lưu lượng thông lượng, bỏ qua độ trễ phản hồi token đầu tiên):

- **Gom lô tối đa, gạt bỏ độ trễ từng ảnh.** Đánh giá ngoại tuyến = gom lô tĩnh (static batching) ở kích thước lớn nhất mà bộ nhớ VRAM có thể chịu tải. Sử dụng lệnh `pipe(...).images` với tham số `num_images_per_prompt=8` trên GPU H100 80GB sẽ rút ngắn thời gian thực tế từ 4 đến 6 lần so với xử lý từng ảnh đơn lẻ.
- **Lưu bộ nhớ đệm (cache) cho các đặc trưng của tập ảnh thật.** Quá trình trích xuất đặc trưng Inception (phục vụ FID) hoặc CLIP (phục vụ điểm CLIP, CMMD) trên tập ảnh thật tham chiếu chỉ cần thực hiện *đúng một lần duy nhất*, sau đó lưu lại dưới dạng tệp `.npz`. Tuyệt đối không tính toán lại các đặc trưng này trong mỗi lần đánh giá mô hình mới.

Đối với quy trình CI / kiểm thử hồi quy (regression gates): chạy đo FID + điểm CLIP trên tập con rút gọn gồm 500 mẫu cho mỗi PR (~30 phút); thực hiện đánh giá toàn diện trên 10k mẫu đo FID + HPSv2 + Elo theo định kỳ hàng đêm (nightly build).

## Tài liệu đọc thêm (Further Reading)

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — Bài báo công bố chỉ số FID.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — Bài báo công bố chỉ số CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Nền tảng mô hình CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — Chỉ số HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — Mô hình ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — Bộ tiêu chuẩn PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — Khảo sát toàn diện về các khiếm khuyết trong các chỉ số đánh giá mô hình sinh.
