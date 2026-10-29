# Định luật tỷ lệ (Scaling Laws)

> Bài báo năm 2020 của Kaplan chỉ ra rằng: mô hình càng lớn, hàm mất mát (loss) càng thấp. Bài báo năm 2022 của Hoffmann phản biện lại: bạn đã huấn luyện chưa đủ độ (under-training). Năng lực tính toán (compute) được phân bổ vào hai ngăn — tham số và token — và sự phân chia này không hề hiển nhiên.

**Phân loại:** Học lý thuyết (Learn)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 05 (Transformer hoàn chỉnh), Giai đoạn 7 · 07 (GPT)  
**Thời lượng:** ~45 phút

## Vấn đề đặt ra

Khi bạn sở hữu $C$ FLOPs năng lực tính toán phục vụ huấn luyện và muốn tạo ra mô hình tốt nhất, bạn phải đối mặt với hai nút điều chỉnh:

1. **Bao nhiêu tham số (N)?** Mô hình càng lớn, dung lượng biểu đạt (capacity) càng cao.
2. **Bao nhiêu token huấn luyện (D)?** Dữ liệu càng nhiều, việc khai thác dung lượng mô hình càng hiệu quả.

FLOPs huấn luyện tỷ lệ xấp xỉ theo công thức `6 × N × D`. Bạn có thể tăng $N$ và giảm $D$, hoặc tăng $D$ và giảm $N$. Phương án nào tối ưu hơn?

Trước năm 2022, câu trả lời mặc định là "tập trung tối đa vào việc tăng $N$." GPT-3 (2020) có 175B tham số nhưng chỉ được huấn luyện trên khoảng 300B token. Tỷ lệ này chỉ vào khoảng 1,7 token cho mỗi tham số. Các định luật tỷ lệ của Kaplan vào thời điểm đó đã củng cố cho quan điểm này.

Tuy nhiên, Hoffmann và các cộng sự (2022), khi huấn luyện một họ mô hình nhỏ mang tên Chinchilla, đã phát hiện ra một sự thật hoàn toàn khác: tỷ lệ tối ưu nằm ở mức xấp xỉ **20 token cho mỗi tham số**. GPT-3 thực chất đã bị thiếu dữ liệu huấn luyện (undertrained) tới 10 lần. Chinchilla (70B tham số, 1,4T token) đã đánh bại GPT-3 (175B tham số, 300B token) trên mọi bài kiểm chuẩn (benchmark) trong khi chi phí suy luận (inference cost) thấp hơn tới 2,5 lần.

Năm 2026 chính là thế giới của Chinchilla — nhưng đi kèm một bước ngoặt quan trọng. Llama 3 8B được huấn luyện trên 15 nghìn tỷ (trillion) token, đạt tỷ lệ 1.875 token cho mỗi tham số. Con số này vượt gấp 94 lần mức tối ưu theo Chinchilla. Chi phí suy luận có tầm quan trọng vượt trội so với chi phí huấn luyện đối với các mô hình được triển khai ở quy mô lớn, do đó việc huấn luyện vượt ngưỡng (over-training - vượt xa mức tối ưu Chinchilla) để có được một mô hình nhỏ gọn hơn khi triển khai đã trở thành lựa chọn mặc định vào năm 2026.

## Khái niệm cốt lõi

![Đường cong Chinchilla: hàm mất mát so với năng lực tính toán ở các tỷ lệ N/D khác nhau](../assets/scaling-laws.svg)

### Định luật Hoffmann

Từ bài báo Chinchilla, hàm mất mát tuân theo công thức:

```math
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E
```

- $N$ = số lượng tham số (không tính tầng nhúng - non-embedding).
- $D$ = số lượng token huấn luyện.
- $\alpha \approx 0,34$, $\beta \approx 0,28$ (gần như đối xứng).
- $E \approx 1,69$, trần mất mát không thể giảm thiểu (irreducible loss ceiling).
- $A \approx 406$, $B \approx 411$.

Hai số hạng này đánh đổi lẫn nhau khi bạn mở rộng quy mô. Lấy đạo hàm theo $N$ tại mức ngân sách tính toán cố định ($C = 6ND$) và giải phương trình. Với các giá trị $\alpha$ và $\beta$ ở trên, các số mũ xấp xỉ khoảng 0,45 cho $N$ và 0,55 cho $D$, vì vậy tỷ lệ tối ưu chính xác sẽ trôi chậm theo ngân sách; file `code/main.py` tìm điểm này bằng phương pháp tìm kiếm lưới (grid search). Hai phương pháp làm khớp (fitting approaches) đầu tiên của bài báo, vốn đo lường điểm tối ưu một cách trực tiếp, ước lượng cả hai số mũ vào khoảng 0,5. Điều này dẫn tới quy tắc kinh nghiệm về một tỷ lệ cố định. Đặt $D = 20N$ vào phương trình $C = 6ND$, chúng ta có quy tắc ngón tay cái:

```math
\begin{aligned}
N_{\text{opt}} &\approx 0.22 \times (C/6)^{0.5} \\
D_{\text{opt}} &\approx 4.5 \times (C/6)^{0.5} \\
D_{\text{opt}} / N_{\text{opt}} &\approx 20
\end{aligned}
```

Tối ưu tính toán (Compute-optimal): 20 token cho mỗi tham số.

### Vì sao vẫn nên huấn luyện vượt ngưỡng (over-training)?

Điểm tối ưu Chinchilla giúp giảm thiểu mất mát huấn luyện trên mỗi FLOP tính toán huấn luyện. Tuy nhiên, bạn chỉ trả chi phí huấn luyện đúng một lần, trong khi chi phí suy luận phải trả mãi mãi.

Đối với một chatbot phục vụ hàng nghìn tỷ token mỗi tháng, chi phí suy luận chi phối toàn bộ tổng chi phí. Cách tiếp cận của dòng mô hình Llama là: huấn luyện mô hình nhỏ hơn, nhưng lâu hơn. Một mô hình 8B được huấn luyện trên 15T token là một mô hình được tối ưu hóa sâu sắc cho suy luận:

- Vừa vặn trên các GPU phổ thông dành cho người dùng cá nhân (consumer GPUs).
- Độ trễ (latency) chỉ bằng một phần nhỏ so với mô hình 70B chuẩn Chinchilla.
- Chất lượng đủ tốt để đáp ứng hầu hết các tác vụ.

Bài báo năm 2024 của DeepMind ("Over-training is the new optimal") đã chính thức hình thức hóa điều này. Đối với các khối lượng công việc bị chi phối bởi suy luận, tỷ lệ hợp lý nằm trong khoảng từ 100–500 token cho mỗi tham số tùy thuộc vào lưu lượng phục vụ.

### Năng lực đột sinh so với sự cải thiện mượt mà (Emergence vs smoothness)

Giả thuyết đặt ra: một số năng lực nhất định (số học, lập luận đa bước, suy luận chuỗi ý nghĩ - chain-of-thought) "đột sinh" (emerge) một cách đột ngột tại một ngưỡng quy mô nhất định.

Schaeffer và các cộng sự (2023) đã chứng minh rằng đây thực chất là một sai số do phương pháp đo lường (measurement artifact): các thang đo đột sinh sử dụng cách chấm điểm gián đoạn (khớp chính xác - exact match, độ chính xác theo ngưỡng) đã che khuất đi sự cải thiện mượt mà, liên tục của các logit bên dưới. Các thước đo liên tục (như entropy chéo - cross-entropy) luôn thể hiện những đường cong cải thiện mượt mà.

Đến năm 2026, sự đồng thuận chung của giới học thuật là: các dự đoán dựa trên hàm mất mát liên tục là đáng tin cậy. Những bước nhảy vọt trên benchmark thường chỉ là hiệu ứng giả tạo từ bộ chấm điểm. Hãy lập kế hoạch ngân sách dựa trên các thước đo liên tục.

### Bức tranh toàn cảnh năm 2026

Các định luật tỷ lệ vẫn phát huy tác dụng, nhưng chịu ảnh hưởng bởi các yếu tố mới:

| Yếu tố | Thay đổi như thế nào |
|--------|---------------------|
| Chất lượng dữ liệu | Tinh lọc các token "chất lượng cao" (phong cách Phi) dịch chuyển đường cong tương đương mức tăng >2× năng lực tính toán hiệu dụng |
| Hỗn hợp chuyên gia (MoE) | Tổng tham số tách rời khỏi FLOPs tích cực; định luật tỷ lệ áp dụng trên mỗi FLOP tích cực |
| Hậu huấn luyện (Post-training) | Một số năng lực (tuân thủ chỉ dẫn, viết mã) dịch chuyển mạnh mẽ nhờ SFT+RLHF hơn là tiền huấn luyện |
| Đa phương thức (Multimodality) | Token hình ảnh + văn bản cùng mở rộng; có các đường cong riêng biệt cho từng phương thức |
| Dữ liệu tổng hợp (Synthetic data) | Mô hình tự tạo dữ liệu huấn luyện; năng lực tính toán hiệu dụng có thể nhân lên theo cấp số nhân |

Thuật toán tối ưu hóa Muon (Kimi Moonlight, 2024) đã chứng minh mức tăng hiệu quả tính toán tương đương ~2× so với AdamW trên cùng một tập dữ liệu. Nhiều chu trình huấn luyện năm 2026 đã sử dụng Muon làm bộ tối ưu mặc định. Nó thay đổi hằng số tuyệt đối trong định luật tỷ lệ chứ không làm thay đổi hình dạng đường cong.

````figure
scaling-laws
````

## Xây dựng thực tế

Xem `code/main.py`. Chúng ta sẽ lập trình phương trình mất mát Chinchilla và giải bài toán tìm bộ tham số tối ưu tính toán $(N, D)$ trên nhiều mức ngân sách tính toán khác nhau.

### Bước 1: Hàm mất mát Chinchilla

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Vẽ đường mức của $L$ trên không gian $(N, D)$ tại mức ngân sách cố định $C = 6ND$. Tìm điểm cực tiểu.

### Bước 2: Biên giới tối ưu tính toán

Với các mức ngân sách tính toán từ `1e17` đến `1e25` FLOPs, tìm cặp $(N, D)$ giúp cực tiểu hóa hàm mất mát dưới ràng buộc $6ND = C$. Xác minh lại tỷ lệ $D/N \approx 20$.

### Bước 3: Chi phí của việc huấn luyện vượt ngưỡng

Tính toán phần mất mát gia tăng mà bạn phải đánh đổi khi huấn luyện một mô hình nhỏ hơn 10 lần (1/10 mức N tối ưu, nhưng huấn luyện trên 10 lần mức D tối ưu). Báo cáo mức tiết kiệm FLOPs suy luận (tỷ lệ thuận với N) nhận được để bù lại.

### Bước 4: So sánh với các mô hình thực tế

Đưa các cặp `(N, D)` thực tế của GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (tham số tích cực) vào mô hình tính toán và so sánh giá trị mất mát dự đoán với giá trị mất mát thực tế được công bố.

## Ứng dụng thực tế

Có thể bạn sẽ không tự mình huấn luyện một mô hình tiên phong từ đầu. Nhưng các định luật tỷ lệ sẽ cho bạn biết:

1. **Liệu tập dữ liệu tinh chỉnh (fine-tune) của bạn có đủ lớn hay không.** Nếu dữ liệu đặc thù cho tác vụ của bạn dưới 20 token cho mỗi tham số của mô hình nền, mô hình sẽ sớm bão hòa tại một ngưỡng mất mát sàn nhất định.
2. **Có nên chọn một mô hình nền lớn hơn hay không.** Nếu bạn dành phần lớn ngân sách cho khâu suy luận, hãy ưu tiên chọn mô hình nhỏ hơn nhưng được huấn luyện lâu hơn.
3. **Đâu là điểm hiệu suất giảm dần (diminishing returns).** Vượt quá ngưỡng 1.000 lần mức tối ưu Chinchilla, sự thay đổi của log-loss chỉ còn là nhiễu không đáng kể.

**Xu hướng nghiên cứu trong năm 2026:**

- **Chế độ giới hạn dữ liệu (Data-constrained regime).** Internet chỉ có một lượng token chất lượng cao hữu hạn (~5–10 nghìn tỷ token tiếng Anh sau khi lọc). Tiền huấn luyện các mô hình tiên phong đang tiến sát trần giới hạn này. Dữ liệu tổng hợp (synthetic data), dữ liệu đa ngữ, dữ liệu đa phương thức và tinh chỉnh quy mô lớn bằng RLHF là những đòn bẩy tiếp theo.
- **Các kỹ thuật nhân bội năng lực tính toán.** Bộ tối ưu hóa Muon, kiến trúc MoE, chọn lọc dữ liệu tốt hơn — mỗi yếu tố này đều giúp dịch chuyển hằng số tuyệt đối, dù không làm thay đổi tiệm cận lý thuyết.
- **Định luật tỷ lệ cho học tăng cường (RL).** Vẫn là một câu hỏi mở. Các bằng chứng ban đầu cho thấy quy luật lũy thừa (power-law) vẫn xuất hiện trong các mẫu RL nhưng với các số mũ rất khác biệt so với giai đoạn tiền huấn luyện.

## Đóng gói ứng dụng

Xem file `outputs/skill-training-budget-estimator.md`. Kỹ năng này giúp xác định bộ giá trị `(N, D, hours, GPU)` cho một chu trình huấn luyện mới dựa trên ngân sách tính toán, ràng buộc triển khai và mục tiêu hàm mất mát.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. In ra các giá trị $(N, D)$ tối ưu theo Chinchilla cho các ngân sách tính toán `1e20`, `1e22`, `1e24`. So sánh với bảng thông số của các mô hình thực tế.
2. **Trung bình.** Cài đặt đường cong hàm mất mát theo năng lực tính toán của Hoffmann. Vẽ đồ thị hàm mất mát theo `log10(C)` cho biên giới tối ưu tính toán. Xác định thời điểm định luật dự báo chúng ta sẽ cần $>10^{28}$ FLOPs để giảm thêm 0,1 điểm cross-entropy.
3. **Khó.** Tự làm khớp (fit) định luật tỷ lệ của riêng bạn trên 5 mô hình kích thước nhỏ (từ 100K đến 10M tham số) được huấn luyện trên cùng một tập dữ liệu. Ước lượng các giá trị $\alpha$ và $E$. Các số mũ bạn thu được khớp tốt đến mức nào so với các bài báo đã công bố?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Tham số (Parameters - N) | "Kích thước mô hình" | Số lượng trọng số không tính tầng nhúng; quyết định dung lượng biểu đạt của mô hình. |
| Token (D) | "Dữ liệu huấn luyện" | Số lượng token huấn luyện mà mô hình đã đọc; quyết định mức độ khai thác các tham số. |
| Năng lực tính toán (Compute - C) | "FLOPs đã tiêu tốn" | Xấp xỉ bằng `6 × N × D` đối với một mô hình transformer tiêu chuẩn. |
| Tối ưu Chinchilla (Chinchilla-optimal) | "D/N ≈ 20" | Tỷ lệ giúp cực tiểu hóa hàm mất mát trên mỗi FLOP tính toán tiền huấn luyện. |
| Huấn luyện vượt ngưỡng (Over-training) | "Vượt xa Chinchilla" | Tiêu tốn thêm FLOPs huấn luyện để tiết kiệm FLOPs suy luận lâu dài; D/N >> 20. |
| Mất mát không thể giảm thiểu (Irreducible loss) | "Mức sàn mất mát" | Số hạng $E$ trong định luật tỷ lệ; đại diện cho entropy nội tại của chính tập dữ liệu. |
| Khả năng đột sinh (Emergent capability) | "Bước nhảy vọt ở quy mô lớn" | Thường là ảo ảnh do thước đo gián đoạn tạo ra; hàm mất mát liên tục luôn biến thiên mượt mà. |
| Năng lực tính toán hiệu dụng (Effective compute) | "Hệ số nhân hiệu quả huấn luyện" | Dữ liệu, thuật toán tối ưu hoặc kiến trúc tốt hơn giúp nhân lên giá trị thực tế của từng FLOP tính toán. |

## Tài liệu tham khảo

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — bài báo đầu tiên về định luật tỷ lệ; các mô hình bị thiếu dữ liệu huấn luyện.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — bài báo Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — năng lực đột sinh thực chất là sai số từ công cụ đo.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — lý giải vì sao chiến lược huấn luyện vượt ngưỡng của Llama là hoàn toàn chính xác cho khối lượng công việc thực tế.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — bộ tối ưu hóa giúp nhân đôi hiệu quả tính toán.
