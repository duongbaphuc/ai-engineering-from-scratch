# Họ thuật toán tối ưu hóa sở thích trực tiếp (The Direct Preference Optimization Family)

> Rafailov et al. (2023) đã chứng minh rằng nghiệm tối ưu của RLHF có dạng giải tích tường minh (closed form) theo dữ liệu sở thích, do đó bạn có thể bỏ qua mô hình phần thưởng tường minh và trực tiếp tối ưu hóa chính sách. Phát kiến sâu sắc đó đã khai sinh ra một họ thuật toán — IPO, KTO, SimPO, ORPO, BPO — mỗi biến thể khắc phục một chế độ thất bại của DPO. Vào năm 2026, các thuật toán căn chỉnh trực tiếp (direct alignment algorithms - DAA) được áp dụng trong nhiều đợt huấn luyện hậu kỳ cho các mô hình tiên phong hơn cả PPO. Tuy nhiên, đường cong tối ưu hóa quá mức từ Bài học 2 vẫn hiện hữu: DAA không thoát khỏi Định luật Goodhart, chúng chỉ thay đổi vị trí nơi quy luật này phát tác.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 phút

## Mục tiêu học tập

- Rút ra dạng giải tích tường minh của DPO từ nghiệm tối ưu của bài toán RLHF có điều chuẩn KL.
- Trình bày chế độ thất bại mà từng biến thể IPO, KTO, SimPO, ORPO, BPO khắc phục trong DPO.
- Phân biệt giữa "khoảng cách phần thưởng tiềm ẩn" (implicit reward gap) và "cường độ ưu tiên" (preference strength), đồng thời giải thích tại sao phép ánh xạ đồng nhất (identity mapping) của IPO lại đóng vai trò quan trọng.
- Giải thích vì sao Rafailov et al. (NeurIPS 2024) chứng minh được rằng DAA vẫn bị tối ưu hóa quá mức mặc dù không có mô hình phần thưởng tường minh.

## Vấn đề

Hàm mục tiêu RLHF (Bài 1):

```math
\max_{\pi} \; \mathbb{E}_{x, y \sim \pi} \big[ r(x, y) \big] - \beta \, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
```

có nghiệm tối ưu đã biết:

```math
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)
```

Do đó, phần thưởng được định nghĩa tiềm ẩn thông qua tỷ số giữa chính sách tối ưu và chính sách tham chiếu:

```math
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
```

Thay biểu thức này vào hàm hợp lý ưu tiên Bradley-Terry, hàm phân vị $Z(x)$ sẽ triệt tiêu vì nó chỉ phụ thuộc vào $x$. Biểu thức còn lại là một hàm mất mát hoàn toàn phụ thuộc vào các tham số của chính sách — không cần đến mô hình phần thưởng. Đó chính là DPO.

Điểm mấu chốt: phép rút gọn toán học này giả định rằng nghiệm tối ưu có thể đạt tới được, dữ liệu sở thích nằm trong phân phối (in-distribution), và chính sách tham chiếu là một điểm neo mode thực sự. Không có giả định nào trong số này được thỏa mãn tuyệt đối trên thực tế. Mỗi thành viên trong họ thuật toán DAA ra đời nhằm khắc phục một giả định bị vi phạm khác nhau.

## Khái niệm

### DPO (Rafailov et al., 2023)

```math
L_{\text{DPO}} = -\log \sigma\!\left(
  \beta \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
  - \beta \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
```

Những vấn đề có thể phát sinh:

- Khoảng cách phần thưởng tiềm ẩn $\beta \big(\log(\pi/\pi_{\text{ref}})_w - \log(\pi/\pi_{\text{ref}})_l\big)$ là không bị chặn. Một mức độ ưu tiên rất nhỏ cũng có thể tạo ra khoảng cách lớn tùy ý.
- Hàm mất mát đẩy log-prob của câu trả lời được chọn ($y_w$) và câu trả lời bị từ chối ($y_l$) theo hai hướng đối nghịch. Nó có thể đẩy log-prob tuyệt đối của câu được chọn giảm xuống, miễn là log-prob của câu bị từ chối sụt giảm nhanh hơn. Đây là hiện tượng "Phản hồi được chọn bị suy thoái" (Degraded Chosen Response).
- Các sở thích nằm ngoài phân phối (các cặp mà cả hai phản hồi đều hiếm gặp dưới chính sách tham chiếu) tạo ra các phần thưởng tiềm ẩn tùy tiện.

### IPO (Azar et al., 2024)

Tối ưu hóa sở thích đồng nhất (Identity Preference Optimization - IPO) thay thế hàm log-sigmoid bằng phép ánh xạ đồng nhất trên xác suất ưu tiên. Hàm mất mát trở thành sai số bình phương trên một mục tiêu bị chặn:

```math
L_{\text{IPO}} = \left( \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\beta} \right)^2
```

Khoảng cách biên được chặn bởi $1/(2\beta)$. Cường độ ưu tiên và khoảng cách phần thưởng tiềm ẩn tỷ lệ thuận với nhau. Không xảy ra hiện tượng bùng nổ giá trị.

### KTO (Ethayarajh et al., 2024)

Tối ưu hóa Kahneman-Tversky (Kahneman-Tversky Optimization - KTO) loại bỏ hoàn toàn cấu trúc theo cặp. Với một đầu ra được dán nhãn đơn lẻ cùng tín hiệu nhị phân "mong muốn" (desirable) hoặc "không mong muốn" (undesirable), thuật toán ánh xạ sang hàm thỏa dụng lý thuyết triển vọng (prospect theory utility):

```math
v(x, y) = \sigma\!\left(\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_{\text{ref}}\right)
```

với các trọng số khác nhau cho phần được (gains) và phần mất (losses) nhằm mô phỏng tính ngại rủi ro mất mát (loss aversion). Lợi ích: bạn có thể sử dụng dữ liệu không ghép cặp (unpaired data), nguồn dữ liệu vốn dồi dào hơn rất nhiều.

### SimPO (Meng et al., 2024)

Tối ưu hóa sở thích đơn giản (Simple Preference Optimization - SimPO) đồng bộ hóa tín hiệu huấn luyện với quá trình sinh văn bản. Thuật toán loại bỏ hoàn toàn chính sách tham chiếu và chuẩn hóa hàm hợp lý log theo độ dài:

```math
L_{\text{SimPO}} = -\log \sigma\!\left(
  \frac{\beta}{|y_w|} \log \pi(y_w \mid x)
  - \frac{\beta}{|y_l|} \log \pi(y_l \mid x)
  - \gamma
\right)
```

với một biên độ $\gamma$ để tạo sự ổn định. Việc chuẩn hóa độ dài loại bỏ động lực khai thác chế độ thất bại thiên vị độ dài của DPO (theo thiết kế, $y_w$ dài hơn sẽ tự nhiên tạo ra khoảng cách log-prob lớn hơn).

### ORPO (Hong et al., 2024)

Tối ưu hóa sở thích theo tỷ số chênh (Odds-Ratio Preference Optimization - ORPO) bổ sung một số hạng ưu tiên vào hàm mất mát hợp lý log âm (negative log-likelihood - NLL) tiêu chuẩn của SFT:

```math
\begin{aligned}
L_{\text{ORPO}} &= L_{\text{NLL}}(y_w) + \lambda L_{\text{OR}} \\
L_{\text{OR}} &= -\log \sigma\!\left(\log \frac{\text{odds}(y_w)}{\text{odds}(y_l)}\right)
\end{aligned}
```

Không cần chính sách tham chiếu — số hạng SFT đóng vai trò là bộ điều chuẩn. Huấn luyện trong một giai đoạn duy nhất từ mô hình nền tảng đến mô hình đã căn chỉnh, không cần checkpoint SFT riêng biệt.

### BPO (Đệ trình ICLR 2026, OpenReview id=b97EwMUWu7)

Xác định vấn đề suy thoái phản hồi được chọn: DPO duy trì thứ hạng $y_w > y_l$ nhưng log-prob tuyệt đối của $y_w$ có thể giảm sút. Tối ưu hóa bảo toàn hành vi (Behavior Preservation Optimization - BPO) thêm một dòng hiệu chỉnh để phạt các dịch chuyển đi xuống của phản hồi được chọn. Báo cáo cho thấy độ chính xác tăng +10.1% trên Llama-3.1-8B-Instruct trong tác vụ lập luận toán học so với DPO.

### Kết quả phổ quát: DAA vẫn bị tối ưu hóa quá mức

Rafailov et al. trong "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) đã huấn luyện các chính sách bằng DPO, IPO, SLiC trên nhiều tập dữ liệu qua các mức ngân sách KL khác nhau. Các đường cong phần thưởng chuẩn vàng theo khoảng cách KL đều có dạng hình học đạt đỉnh rồi sụp đổ tương tự như Gao et al. Phần thưởng tiềm ẩn truy vấn các mẫu nằm ngoài phân phối trong quá trình huấn luyện; điều chuẩn KL không thể ổn định hóa hiện tượng này.

DAA không thoát khỏi Goodhart. Chúng chỉ thay đổi bề mặt nơi quy luật phát tác từ "mô hình phần thưởng bị tối ưu hóa quá mức" thành "tỷ số chính sách tham chiếu bị tối ưu hóa quá mức". Giải pháp phổ quát — dữ liệu tốt hơn, nhóm mô hình, dừng sớm — đều áp dụng cho cả hai.

### Lựa chọn thuật toán (Năm 2026)

- Nếu bạn có tập dữ liệu sở thích ghép cặp lớn: dùng DPO với giá trị $\beta$ thận trọng; dùng SimPO nếu phát hiện thiên vị độ dài rõ rệt.
- Nếu bạn có phản hồi nhị phân không ghép cặp: dùng KTO.
- Nếu bạn muốn quy trình một giai đoạn duy nhất trực tiếp từ mô hình base: dùng ORPO.
- Nếu bạn nhận thấy log-prob của phản hồi được chọn bị sụt giảm trong nhật ký DPO: dùng BPO.
- Nếu cường độ ưu tiên dao động rất mạnh và DPO bị bão hòa: dùng IPO.

Mọi phòng thí nghiệm đều chạy cả năm phương pháp trên một bộ thử nghiệm và chọn phương án chiến thắng cho từng tác vụ cụ thể. Không có lý do gì để kỳ vọng nghiệm tối ưu cho lập luận toán học lại trùng với nghiệm tối ưu cho an toàn.

```figure
dpo-margin
```

## Thực hành (Use It)

Tệp `code/main.py` so sánh sáu hàm mất mát (DPO, IPO, KTO, SimPO, ORPO, BPO) trên một tập dữ liệu sở thích mẫu, nơi cường độ ưu tiên thực sự thay đổi theo từng cặp. Mỗi hàm mất mát được tối ưu hóa đối chiếu với cùng một mẫu 500 cặp bằng một chính sách softmax nhỏ. Mã nguồn vẽ đồ thị tỷ lệ thắng cuối cùng, độ trôi dạt log-prob của phản hồi được chọn, và độ lan tỏa phần thưởng tiềm ẩn cho từng phương pháp.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra `outputs/skill-preference-loss-selector.md`. Dựa trên các đặc trưng thống kê của tập dữ liệu (ghép cặp hay không ghép cặp, cường độ ưu tiên biến thiên hay đồng đều, phân phối độ dài) và mục tiêu huấn luyện (một giai đoạn hay SFT rồi mới đến sở thích), công cụ này sẽ đề xuất một hàm mất mát sở thích phù hợp và báo cáo chế độ thất bại mà nó bảo vệ hệ thống khỏi.

## Bài tập

1. Chạy `code/main.py`. Ghi lại mức sụt giảm log-prob của câu được chọn ở giai đoạn cuối cho DPO và BPO. BPO sẽ giữ được xác suất tuyệt đối của câu được chọn cao hơn — hãy kiểm chứng điều này.

2. Chỉnh sửa dữ liệu sở thích sao cho tất cả các cặp đều có cường độ bằng nhau. Phương pháp nào trong sáu phương pháp bền vững nhất? Phương pháp nào bị suy giảm hiệu năng? Giải thích lợi thế của IPO trong trường hợp này.

3. Làm cho các phản hồi bị từ chối dài gấp đôi các phản hồi được chọn tính trung bình. Không thay đổi bất kỳ yếu tố nào khác, hãy chứng minh hành vi khai thác độ dài của DPO bằng số liệu thực tế và cách SimPO khắc phục hiện tượng này.

4. Rafailov et al. (NeurIPS 2024) tuyên bố DAA vẫn bị tối ưu hóa quá mức. Hãy tái tạo một phiên bản đơn điểm: vẽ đồ thị phân kỳ KL giữa câu được chọn và câu bị loại trừ, rồi quan sát hiện tượng tối ưu hóa quá mức trong DPO khi $\beta$ lớn.

5. Đọc bản tóm tắt bài báo BPO (OpenReview b97EwMUWu7). Viết lại dòng hiệu chỉnh duy nhất mà BPO bổ sung vào DPO. Đối chiếu với triển khai thực tế trong `code/main.py`.

## Thuật ngữ chính

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| DPO | "RLHF không cần mô hình phần thưởng" | Hàm mất mát rút ra từ nghiệm tối ưu dạng giải tích của RLHF; chỉ tối ưu trực tiếp tham số chính sách |
| Implicit reward | "phần thưởng tiềm ẩn / tỷ số log" | $\beta \log(\pi(y \mid x) / \pi_{\text{ref}}(y \mid x))$ — phần thưởng được suy ra ngầm định từ DPO |
| IPO | "DPO bị chặn" | Thay thế log-sigmoid bằng hàm đồng nhất; khoảng cách phần thưởng tiềm ẩn bị chặn ở mức $1/(2\beta)$ |
| KTO | "DPO không ghép cặp" | Hàm thỏa dụng lý thuyết triển vọng trên các nhãn đơn lẻ có tính đến tâm lý ngại tổn thất |
| SimPO | "DPO không cần tham chiếu" | Hợp lý log được chuẩn hóa theo độ dài kèm biên độ phân cách; không cần chính sách tham chiếu |
| ORPO | "DPO một giai đoạn" | Kết hợp NLL với số hạng tỷ số chênh; huấn luyện trực tiếp từ mô hình base trong một lượt duy nhất |
| BPO | "DPO bảo toàn câu được chọn" | DPO bổ sung thêm hình phạt cho việc làm giảm log-prob tuyệt đối của câu trả lời được chọn |
| Degraded Chosen | "câu được chọn bị kéo tụt" | Hiện tượng DPO làm giảm log-prob của câu được chọn miễn là câu bị từ chối tụt nhanh hơn |
| DAA | "thuật toán căn chỉnh trực tiếp" | Bất kỳ phương pháp mất mát sở thích nào bỏ qua mô hình phần thưởng tường minh |

## Đọc thêm

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — bài báo giới thiệu IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
