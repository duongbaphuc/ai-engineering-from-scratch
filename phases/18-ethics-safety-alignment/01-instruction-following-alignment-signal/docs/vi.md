# Tuân thủ chỉ dẫn như một tín hiệu căn chỉnh (Instruction-Following as Alignment Signal)

> Mọi sự phê phán sau này đối với học tăng cường từ phản hồi của con người (reinforcement learning from human feedback - RLHF) đều lập luận chống lại quy trình này. Trước khi nghiên cứu cách mà áp lực tối ưu hóa bóp méo một thước đo đại diện (proxy), bạn cần phải thấy rõ bản thân thước đo đại diện đó. InstructGPT (Ouyang et al., 2022) đã định hình kiến trúc tham chiếu tiêu chuẩn: tinh chỉnh có giám sát (supervised fine-tuning - SFT) trên các cặp chỉ dẫn - phản hồi, một mô hình phần thưởng (reward model) được huấn luyện trên thứ hạng ưu tiên theo cặp (pairwise preference rankings), và PPO đối chiếu với mô hình phần thưởng cùng một hệ số phạt phân kỳ KL (KL penalty) đối với chính sách SFT. Một mô hình InstructGPT 1.3B đã được ưa thích hơn cả mô hình GPT-3 175B nguyên bản. Kết quả đơn lẻ đó chính là lý do vì sao mọi phòng thí nghiệm tiên phong vào năm 2026 vẫn xuất xưởng quy trình hậu huấn luyện (post-training pipeline) mang cấu trúc RLHF.

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 phút

## Mục tiêu học tập

- Nêu tên ba giai đoạn trong quy trình InstructGPT và hàm mất mát (loss) được sử dụng ở từng giai đoạn.
- Giải thích lý do vì sao một mô hình 1.3B được tinh chỉnh theo chỉ dẫn lại đánh bại GPT-3 175B nguyên bản trong đánh giá sở thích của con người.
- Trình bày hệ số phạt KL ở giai đoạn 3 bảo vệ hệ thống khỏi điều gì và tại sao loại bỏ nó lại dẫn đến suy thoái thành hành vi tìm kiếm mode (mode-seeking behaviour).
- Mô tả thuế căn chỉnh (alignment tax) và biện pháp giảm thiểu PPO-ptx mà Ouyang et al. đã sử dụng để khắc phục nó.

## Vấn đề

Các mô hình ngôn ngữ tiền huấn luyện chỉ làm nhiệm vụ điền tiếp văn bản. Chúng không trả lời câu hỏi. Khi bạn yêu cầu GPT-3 "viết một hàm Python đảo ngược danh sách", bạn thường nhận lại một câu gợi ý (prompt) khác, bởi phần lớn phân phối huấn luyện là văn bản trên web vốn tiếp nối bằng nhiều văn bản web khác. Mô hình đang làm đúng nhiệm vụ được giao — nhưng bản thân nhiệm vụ đó là sai đối với mục đích sử dụng.

Đại diện mà mọi phòng nghiên cứu nghiêm túc sử dụng để khắc phục điều này là sở thích của con người (human preference). Hai câu trả lời hoàn chỉnh được chuyển tới một người đánh giá (rater); người đánh giá chọn câu tốt hơn; một mô hình phần thưởng học cách mô phỏng người đánh giá. Sau đó, một vòng lặp RL điều chỉnh chính sách (policy) theo hướng tạo ra các đầu ra được mô hình phần thưởng chấm điểm cao. Đó là toàn bộ luận điểm của InstructGPT được gói gọn trong ba câu. Phần còn lại của bài báo là kỹ thuật triển khai.

## Khái niệm

### Giai đoạn 1: Tinh chỉnh có giám sát (SFT)

Thu thập các cặp prompt - phản hồi, trong đó phản hồi là những gì một con người có thiện chí và năng lực sẽ viết. Ouyang et al. đã sử dụng 13.000 prompt từ những người dán nhãn (labelers) và từ OpenAI API. Tinh chỉnh mô hình nền tảng (base model) trên tập dữ liệu này bằng hàm mất mát cross-entropy tiêu chuẩn.

Những gì SFT mang lại cho bạn: mô hình giờ đây trả lời câu hỏi thay vì viết tiếp chúng. Những gì nó không mang lại: bất kỳ tín hiệu nào về việc người đánh giá ưa thích câu trả lời nào hơn khi có nhiều phương án hợp lý cùng tồn tại.

### Giai đoạn 2: Mô hình phần thưởng (RM)

Với mỗi prompt, lấy mẫu $K$ phản hồi từ mô hình SFT. Người dán nhãn xếp hạng chúng. Huấn luyện một mô hình phần thưởng để chấm điểm cho bất kỳ cặp prompt - phản hồi nào sao cho, với các cặp mà $y_w$ được ưa thích hơn $y_l$:

```math
L_{\text{RM}} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)
```

Đây là hàm mất mát ưu tiên theo cặp Bradley-Terry. RM thường được khởi tạo từ mô hình SFT với đầu sinh ngôn ngữ (LM head) được thay thế bằng một đầu hồi quy vô hướng (scalar head).

Các mô hình phần thưởng thường có kích thước nhỏ: 6B tham số là đủ cho InstructGPT 175B. Chúng cũng rất mong manh — mục 5 của bài báo chủ yếu thảo luận về các hành vi gian lận phần thưởng (reward hacking) xuất hiện ngay ở quy mô nhỏ.

### Giai đoạn 3: PPO với hệ số phạt KL

Xác định hàm mục tiêu:

```math
J(\pi) = \mathbb{E}_{x \sim D,\, y \sim \pi(\cdot \mid x)} \big[ r(x, y) \big] - \beta \, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{SFT}}(\cdot \mid x)\big)
```

Tối đa hóa hàm mục tiêu này bằng PPO. Số hạng KL giữ cho $\pi$ không trôi dạt quá xa khỏi chính sách SFT. Nếu không có nó, bộ tối ưu hóa sẽ tìm ra các mẫu đối kháng (adversarial examples) — những chuỗi ký tự được RM chấm điểm rất cao chỉ vì RM chưa từng thấy chúng bao giờ, chứ không phải vì con người thực sự ưa thích chúng.

Hệ số KL $\beta$ là siêu tham số quan trọng nhất trong RLHF. Đặt quá thấp: gian lận phần thưởng xuất hiện. Đặt quá cao: không có sự cải thiện nào so với SFT.

### Thuế căn chỉnh (Alignment tax)

Sau khi áp dụng RLHF, mô hình được con người ưa thích hơn nhưng lại bị thụt lùi trên các bài kiểm chuẩn tiêu chuẩn (SQuAD, HellaSwag, DROP). Ouyang et al. gọi hiện tượng này là "thuế căn chỉnh" và khắc phục bằng PPO-ptx: trộn các gradient tiền huấn luyện vào mục tiêu RL để mô hình không quên cách thực hiện các tác vụ xuôi dòng (downstream tasks) vốn không bao giờ được khen thưởng trong giai đoạn RL.

```math
J_{\text{ptx}}(\pi) = J(\pi) + \gamma \, \mathbb{E}_{x \sim D_{\text{pretrain}}} \big[ \log \pi(x) \big]
```

PPO-ptx đã trở thành tiêu chuẩn. Anthropic, DeepMind và Meta đều sử dụng một biến thể nào đó của kỹ thuật này.

### Kết quả

InstructGPT 1.3B (SFT + RM + PPO-ptx) được người dán nhãn ưa thích hơn mô hình base GPT-3 175B khoảng 70% số trường hợp. Khoảng cách này còn nới rộng hơn trên các prompt kiểm thử ẩn lấy từ lưu lượng thực tế trong môi trường sản xuất. Hai kết luận quan trọng rút ra từ con số này:

1. Căn chỉnh (alignment) là một trục độc lập với năng lực (capability). Mô hình 175B có nhiều năng lực hơn; mô hình 1.3B có độ căn chỉnh cao hơn; người dán nhãn ưa thích mô hình đã được căn chỉnh.
2. Ngưỡng năng lực tối thiểu (capability floor) được thiết lập bởi mô hình nền tảng. Bạn không thể dùng RLHF để ép một mô hình nền tảng biết những sự thật mà nó chưa từng thấy trong dữ liệu huấn luyện.

### Tại sao đây là điểm tham chiếu cho Phase 18

Mọi sự phê phán trong các bài học sau — gian lận phần thưởng (Bài 2), DPO (Bài 3), xu nịnh (sycophancy - Bài 4), CAI (Bài 5), đặc vụ ngầm (sleeper agents - Bài 7), giả vờ căn chỉnh (alignment faking - Bài 9) — đều lập luận chống lại một phần nào đó của quy trình này. Gian lận phần thưởng tấn công vào giai đoạn 2. DPO gộp giai đoạn 2 và 3 lại làm một. CAI thay thế người dán nhãn con người. Xu nịnh chứng minh người dán nhãn là một nguồn tín hiệu có thiên vị. Giả vờ căn chỉnh cho thấy chính sách có thể hoàn toàn đi vòng qua giai đoạn 3. Bạn không thể nắm bắt bất kỳ phê phán nào trong số này nếu không có quy trình tham chiếu này trong đầu trước tiên.

```figure
al-instruct-pipeline
```

## Thực hành (Use It)

Tệp `code/main.py` mô phỏng ba giai đoạn trên dữ liệu sở thích mẫu đơn giản. "Chính sách" cơ sở là một đồng xu lệch trên các hành động {A, B, C}. Giai đoạn 1 SFT mô phỏng các hành động của người dán nhãn trên 200 prompt. Giai đoạn 2 khớp một mô hình phần thưởng Bradley-Terry từ 500 cặp thứ hạng. Giai đoạn 3 thực hiện cập nhật PPO đơn giản hóa với một hệ số phạt KL so với chính sách SFT. Bạn có thể theo dõi phần thưởng tăng lên, phân kỳ KL mở rộng và chính sách trôi dạt — và bạn có thể tắt số hạng KL để chứng kiến gian lận phần thưởng xuất hiện chỉ trong vòng 50 bước cập nhật.

Những điều cần quan sát:

- Quỹ đạo phần thưởng khi `beta = 0.1` so với `beta = 0.0`.
- $\mathrm{KL}(\pi \Vert \pi_{\text{SFT}})$ qua các bước huấn luyện.
- Phân phối hành động cuối cùng so với sở thích của người dán nhãn.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra `outputs/skill-instructgpt-explainer.md`. Khi được cung cấp một bản mô tả quy trình RLHF hoặc phần tóm tắt của một bài báo, công cụ này sẽ xác định xem giai đoạn nào trong ba giai đoạn đang được sửa đổi, hàm mất mát nào được sử dụng ở từng giai đoạn, và liệu có sự hiện diện của hệ số phạt KL hoặc bộ điều chuẩn tương đương hay không.

## Bài tập

1. Chạy `code/main.py`. Thiết lập `beta = 0.0` và ghi lại phân phối hành động sau 200 bước PPO. Giải thích hành vi tìm kiếm mode (mode-seeking behaviour) trong một đoạn văn.

2. Chỉnh sửa mô hình phần thưởng để có độ lệch (bias) +0.5 cho hành động B (mô phỏng một lỗi phần thưởng). Chạy PPO với `beta = 0.1`. Hệ số phạt KL có ngăn được chính sách khai thác độ lệch này không? Tại giá trị `beta` nào thì hành vi khai thác trở nên rõ rệt?

3. Đọc Hình 1 trong bài báo của Ouyang et al. (arXiv:2203.02155). Tái tạo đường cong sở thích của người dán nhãn bằng cách chạy PPO qua 1, 5, 20, 100 bước và đo lường độ ưu tiên so với mô hình SFT.

4. Mục 4.3 của bài báo báo cáo rằng InstructGPT 1.3B đánh bại GPT-3 175B khoảng 70% số lần. Tại sao tỷ lệ này lại cao hơn trên các prompt ẩn từ môi trường sản xuất so với các prompt do chính người dán nhãn tạo ra?

5. Thay thế hàm mất mát PPO bằng DPO (Phase 10 · 08) trên cùng một tập dữ liệu sở thích. So sánh độ trôi dạt chính sách cuối cùng (KL đối với SFT) và phần thưởng cuối cùng. Phương pháp nào trôi dạt xa hơn ở cùng một mức phần thưởng đạt được?

## Thuật ngữ chính

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| SFT | "tinh chỉnh theo chỉ dẫn" (instruction tuning) | Giai đoạn 1: tinh chỉnh cross-entropy trên các cặp prompt - phản hồi |
| Reward model (RM) | "mô hình phần thưởng" | Bộ hồi quy vô hướng trên (prompt, phản hồi) được huấn luyện bằng Bradley-Terry trên nhãn theo cặp |
| Bradley-Terry | "mất mát ưu tiên theo cặp" (pairwise preference loss) | $-\log \sigma(r_w - r_l)$; quy bài toán xếp hạng theo cặp về phân loại nhị phân |
| KL penalty | "bộ điều chuẩn" (the regularizer) | `beta * KL(pi \|\| pi_SFT)` — giữ chính sách RL ở gần điểm neo SFT |
| PPO-ptx | "PPO trộn dữ liệu tiền huấn luyện" | Cộng thêm một phần log-likelihood tiền huấn luyện vào mục tiêu PPO để bù đắp thuế căn chỉnh |
| Alignment tax | "sự thụt lùi do RLHF" (RLHF regression) | Sự sụt giảm sau RLHF trên các bộ kiểm chuẩn tiêu chuẩn mà RLHF không nhắm tới |
| Labeler preference | "chân lý mặt đất" (ground truth) | Mẫu xếp hạng của con người; RM là đại diện thống kê cho điều này, chứ không phải cho "giá trị của con người" |

## Đọc thêm

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — bài báo InstructGPT, nền tảng cho mọi quy trình RLHF tiếp nối sau đó
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — công trình tiền thân về RLHF cho tác vụ tóm tắt
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — công thức RL dựa trên sở thích nguyên bản
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — phần mở rộng HHH của Anthropic cho quy trình InstructGPT
