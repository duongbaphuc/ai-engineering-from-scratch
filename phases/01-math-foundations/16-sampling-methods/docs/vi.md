# Các phương pháp lấy mẫu (Sampling Methods)

> Lấy mẫu (Sampling) là cách AI khám phá không gian của các khả năng vô tận.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Tự xây dựng các phương pháp lấy mẫu nghịch đảo CDF (inverse CDF), lấy mẫu bác bỏ (rejection sampling) và lấy mẫu tầm quan trọng (importance sampling) từ đầu chỉ dùng số ngẫu nhiên đều
- Xây dựng các kỹ thuật lấy mẫu theo nhiệt độ (temperature), top-k và top-p (nucleus sampling) phục vụ sinh token trong mô hình ngôn ngữ
- Giải thích mẹo tái tham số hóa (reparameterization trick) và vì sao nó cho phép lan truyền ngược qua thao tác lấy mẫu trong VAE
- Chạy thuật toán Metropolis-Hastings MCMC để lấy mẫu từ một phân phối mục tiêu chưa được chuẩn hóa

## The Problem (Vấn đề)

Một mô hình ngôn ngữ hoàn thành việc xử lý prompt của bạn và tạo ra một vector gồm 50.000 logit -- đại diện cho từng token trong bộ từ vựng (vocabulary). Bây giờ nó phải chọn ra một token tiếp theo. Nó sẽ chọn như thế nào?

Nếu nó luôn chọn token có xác suất cao nhất, mọi câu trả lời sẽ giống hệt nhau: Đơn điệu, tất định và nhàm chán. Nếu nó chọn ngẫu nhiên đồng đều, kết quả đầu ra sẽ là một đống văn bản vô nghĩa. Câu trả lời nằm ở khoảng giữa hai thái cực này, và khoảng giữa đó được điều khiển bởi kỹ thuật lấy mẫu (sampling).

Lấy mẫu không chỉ giới hạn trong việc sinh văn bản. Học tăng cường (Reinforcement Learning) ước lượng policy gradient bằng cách lấy mẫu các quỹ đạo (trajectories). Các mạng VAE học biểu diễn tiềm ẩn bằng cách lấy mẫu từ các phân phối học được và lan truyền ngược gradient xuyên qua sự ngẫu nhiên đó. Các mô hình khuếch tán (Diffusion models) tạo ra hình ảnh bằng cách lấy mẫu nhiễu rồi khử nhiễu lặp đi lặp lại. Các phương pháp Monte Carlo ước lượng các tích phân không có công thức nghiệm giải tích dạng đóng. Các thuật toán MCMC khám phá các phân phối hậu nghiệm nhiều chiều mà mắt thường không thể liệt kê hết.

Mọi hệ thống Generative AI đều là một hệ thống lấy mẫu. Chiến lược lấy mẫu quyết định chất lượng, tính đa dạng và khả năng kiểm soát của kết quả đầu ra. Bài học này sẽ xây dựng mọi phương pháp lấy mẫu then chốt từ đầu, bắt đầu từ các số ngẫu nhiên đều đơn giản và kết thúc bằng các kỹ thuật đang vận hành các mô hình LLM và mô hình sinh tiên tiến nhất hiện nay.

## The Concept (Khái niệm)

### Vì sao việc lấy mẫu lại quan trọng

Lấy mẫu đóng 4 vai trò nền tảng xuyên suốt AI và Machine Learning:

**Sinh dữ liệu (Generation).** Các mô hình ngôn ngữ, diffusion model và GAN đều tạo ra đầu ra thông qua lấy mẫu. Thuật toán lấy mẫu trực tiếp kiểm soát tính sáng tạo, độ mạch lạc và tính đa dạng. Temperature, top-k và nucleus sampling là những nút vặn mà các kỹ sư AI điều chỉnh hàng ngày.

**Huấn luyện (Training).** Stochastic gradient descent lấy mẫu các mini-batch. Dropout lấy mẫu các nơ-ron để vô hiệu hóa ngẫu nhiên. Tăng cường dữ liệu (data augmentation) lấy mẫu các phép biến đổi ngẫu nhiên. Lấy mẫu tầm quan trọng (Importance sampling) đánh trọng số lại các mẫu để giảm phương sai gradient trong học tăng cường (PPO, TRPO).

**Ước lượng (Estimation).** Rất nhiều đại lượng trong học máy không có nghiệm giải tích dạng đóng: Hàm mất mát kỳ vọng trên một phân phối dữ liệu, hàm phân vùng (partition function) của mô hình năng lượng, bằng chứng (evidence) trong suy luận Bayes. Ước lượng Monte Carlo xấp xỉ toàn bộ các đại lượng này bằng cách lấy trung bình trên các mẫu.

**Thăm dò (Exploration).** Các thuật toán MCMC thăm dò các phân phối hậu nghiệm trong suy luận Bayes. Các chiến lược tiến hóa (Evolutionary strategies) lấy mẫu các nhiễu loạn tham số. Thompson sampling cân bằng giữa thăm dò (exploration) và khai thác (exploitation) trong bài toán Multi-Armed Bandit.

Thách thức cốt lõi: Bạn chỉ có thể lấy mẫu trực tiếp từ các phân phối đơn giản (phân phối đều, phân phối chuẩn). Đối với mọi phân phối phức tạp khác, bạn cần một phương pháp để chuyển đổi các mẫu đơn giản thành các mẫu thuộc phân phối mục tiêu của bạn.

### Lấy mẫu ngẫu nhiên đều (Uniform Random Sampling)

Mọi phương pháp lấy mẫu đều bắt đầu từ đây. Một bộ sinh số ngẫu nhiên đều tạo ra các giá trị trong $[0, 1)$ trong đó mọi khoảng con có độ dài bằng nhau đều có xác suất xảy ra như nhau.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    với 0 <= a <= b <= 1

Tính chất:
  E[U] = 0.5
  Var(U) = 1/12
```

Để lấy mẫu đều từ một tập rời rạc gồm $n$ phần tử, ta sinh $U$ và trả về $\lfloor n U \rfloor$. Để lấy mẫu từ một đoạn liên tục $[a, b]$, ta tính $a + (b - a) U$.

Bản chất cốt lõi: Một số ngẫu nhiên đều duy nhất chứa vừa đủ lượng thông tin ngẫu nhiên để tạo ra một mẫu từ bất kỳ phân phối xác suất nào. Vấn đề chỉ là tìm ra phép biến đổi toán học thích hợp.

### Phương pháp nghịch đảo hàm phân phối tích lũy (Inverse CDF Method)

Hàm phân phối tích lũy (Cumulative Distribution Function - CDF) ánh xạ các giá trị thành xác suất:

```
F(x) = P(X <= x)

Tính chất:
  F là hàm không giảm
  F(-inf) = 0
  F(+inf) = 1
  F ánh xạ trục số thực vào đoạn [0, 1]
```

Nghịch đảo của CDF ($F^{-1}$) ánh xạ ngược các xác suất trở lại thành các giá trị. Nếu $U \sim \text{Uniform}(0, 1)$, thì $X = F^{-1}(U)$ sẽ tuân theo phân phối mục tiêu.

```
Thuật toán:
  1. Sinh u ~ Uniform(0, 1)
  2. Trả về F_inverse(u)

Vì sao thuật toán đúng:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Ví dụ phân phối mũ (Exponential Distribution):**

```
Hàm mật độ xác suất (PDF): f(x) = lambda * exp(-lambda * x),   x >= 0
Hàm tích lũy (CDF):         F(x) = 1 - exp(-lambda * x)

Giải phương trình F(x) = u theo x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Vì (1 - U) và U có cùng phân phối đều trên [0, 1]:
  x = -ln(u) / lambda
```

Phương pháp này hoạt động hoàn hảo khi bạn có thể viết ra biểu thức $F^{-1}$ dạng đóng. Đối với phân phối chuẩn, không có dạng đóng cho nghịch đảo CDF, vì vậy chúng ta sử dụng các phương pháp khác (như biến đổi Box-Muller hoặc xấp xỉ số học).

**Phiên bản rời rạc:** Đối với phân phối rời rạc, xây dựng CDF dưới dạng tổng tích lũy, sinh $U$, và tìm chỉ số đầu tiên mà tại đó tổng tích lũy vượt qua $U$. Đây chính là cách hàm `sample_categorical` vận hành trong Lesson 06.

### Lấy mẫu bác bỏ (Rejection Sampling)

Khi bạn không thể nghịch đảo hàm CDF nhưng có thể tính được giá trị của hàm mật độ xác suất mục tiêu (lên đến một hằng số nhân), lấy mẫu bác bỏ sẽ phát huy tác dụng.

```
Phân phối mục tiêu: p(x)  (có thể tính được giá trị, có thể chưa chuẩn hóa)
Phân phối đề xuất:  q(x)  (dễ dàng lấy mẫu)
Chặn trên: M sao cho p(x) <= M * q(x) với mọi x

Thuật toán:
  1. Lấy mẫu x ~ q(x)
  2. Lấy mẫu u ~ Uniform(0, 1)
  3. Nếu u < p(x) / (M * q(x)), chấp nhận x
  4. Ngược lại, bác bỏ và quay lại bước 1

Tỷ lệ chấp nhận = Z/M, trong đó Z là tích phân của p(x)
                 (Z = 1, tức tỷ lệ là 1/M, khi p đã được chuẩn hóa)
```

Bao hình $M$ càng khít, tỷ lệ chấp nhận càng cao. Ở không gian số chiều thấp (1-3 chiều), lấy mẫu bác bỏ hoạt động rất tốt. Nhưng ở không gian nhiều chiều, tỷ lệ chấp nhận giảm theo hàm mũ vì phần lớn thể tích của phân phối đề xuất bị bác bỏ. Đây là lời nguyền số chiều (curse of dimensionality) của kỹ thuật lấy mẫu bác bỏ.

**Ví dụ: Lấy mẫu từ phân phối chuẩn bị cắt cụt (truncated normal).** Dùng phân phối đề xuất là phân phối đều trên đoạn bị cắt. Bao hình $M$ là giá trị cực đại của PDF chuẩn trên đoạn đó.

**Ví dụ: Lấy mẫu từ hình bán nguyệt.** Đề xuất các điểm ngẫu nhiên đều trong hình chữ nhật bao quanh. Chấp nhận nếu điểm rơi vào bên trong hình bán nguyệt. Đây chính là cách Monte Carlo tính số $\pi$: Tỷ lệ chấp nhận bằng tỷ lệ diện tích $\pi/4$.

### Lấy mẫu tầm quan trọng (Importance Sampling)

Đôi khi bạn không cần các mẫu thực tế từ phân phối mục tiêu $p(x)$. Bạn chỉ cần ước lượng kỳ vọng dưới phân phối $p(x)$, và bạn có các mẫu lấy từ một phân phối khác $q(x)$.

```
Mục tiêu: Ước lượng E_p[f(x)] = tích phân của f(x) * p(x) dx

Biến đổi lại:
  E_p[f(x)] = tích phân của f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

trong đó w(x) = p(x) / q(x) là các trọng số tầm quan trọng (importance weights).

Công thức ước lượng:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    với x_i ~ q(x)
```

Điều này cực kỳ quan trọng trong học tăng cường. Trong thuật toán PPO (Proximal Policy Optimization), bạn thu thập các quỹ đạo hành động dưới chính sách cũ $\pi_{\text{old}}$ nhưng muốn tối ưu hóa chính sách mới $\pi_{\text{new}}$. Trọng số tầm quan trọng là $\pi_{\text{new}}(a \mid s) / \pi_{\text{old}}(a \mid s)$. PPO cắt tỉa (clip) các trọng số này để ngăn chính sách mới trôi quá xa khỏi chính sách cũ.

Phương sai của ước lượng importance sampling phụ thuộc vào độ tương đồng giữa $q$ và $p$. Nếu $q$ quá khác $p$, một vài mẫu hiếm hoi sẽ nhận trọng số khổng lồ và chi phối toàn bộ kết quả ước lượng. Kỹ thuật Self-normalized importance sampling chia cho tổng các trọng số để khắc phục vấn đề này:

```math
\mathbb{E}_p[f(x)] \approx \frac{\sum_i w_i \, f(x_i)}{\sum_i w_i}
```

### Ước lượng Monte Carlo

Ước lượng Monte Carlo xấp xỉ các tích phân bằng cách lấy trung bình các mẫu ngẫu nhiên. Luật số lớn đảm bảo sự hội tụ về giá trị thực.

```
Mục tiêu: Ước lượng I = tích phân của g(x) dx trên miền D

Phương pháp:
  1. Lấy mẫu x_1, ..., x_N ngẫu nhiên đều từ miền D
  2. I ~ (Thể tích của D / N) * sum(g(x_i))

Sai số: O(1 / sqrt(N))   bất kể số chiều của không gian
```

Tốc độ hội tụ của sai số hoàn toàn không phụ thuộc vào số chiều. Đây là lý do vì sao các phương pháp Monte Carlo thống trị trong không gian nhiều chiều, nơi các phương pháp tích phân dạng lưới số học thông thường trở nên bất khả thi.

**Ước lượng số Pi:**

```
Lấy mẫu (x, y) ngẫu nhiên đều từ [-1, 1] x [-1, 1]
Đếm số điểm rơi vào bên trong đường tròn đơn vị: x^2 + y^2 <= 1
pi ~ 4 * (số điểm bên trong) / (tổng số điểm)
```

**Ước lượng kỳ vọng:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    với x_i ~ p(x)

Giá trị trung bình mẫu hội tụ về kỳ vọng thực tế.
Phương sai của ước lượng = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC xây dựng một chuỗi Markov có phân phối dừng (stationary distribution) chính là phân phối mục tiêu $p(x)$. Sau đủ số bước lặp, các mẫu sinh ra từ chuỗi này sẽ (xấp xỉ) là các mẫu rút ra từ $p(x)$.

```
Mục tiêu: p(x)  (chỉ cần biết hàm tỷ lệ, không cần biết hằng số chuẩn hóa)
Đề xuất: q(x'|x)  (cách đề xuất trạng thái tiếp theo dựa trên trạng thái hiện tại)

Thuật toán Metropolis-Hastings:
  1. Bắt đầu tại một điểm x_0 bất kỳ
  2. Với t = 1, 2, ..., T:
     a. Đề xuất x' ~ q(x'|x_t)
     b. Tính tỷ số chấp nhận:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Chấp nhận với xác suất min(1, alpha):
        - Nếu u < alpha (với u ~ Uniform(0,1)): x_{t+1} = x'
        - Ngược lại: x_{t+1} = x_t
  3. Bỏ qua B mẫu đầu tiên (giai đoạn burn-in)
  4. Trả về các mẫu còn lại
```

Đối với phân phối đề xuất đối xứng $(q(x' \mid x) = q(x \mid x'))$, tỷ số rút gọn thành $p(x')/p(x)$. Đây là thuật toán Metropolis nguyên bản.

**Vì sao thuật toán hoạt động:** Quy tắc chấp nhận đảm bảo tính cân bằng chi tiết (detailed balance): Xác suất ở trạng thái $x$ và chuyển sang $x'$ bằng xác suất ở $x'$ và chuyển về $x$. Cân bằng chi tiết đảm bảo rằng $p(x)$ chính là phân phối dừng duy nhất của chuỗi.

**Các lưu ý thực tế:**
- Burn-in: Loại bỏ các mẫu ban đầu trước khi chuỗi đạt tới trạng thái cân bằng
- Thinning: Chỉ giữ lại mẫu sau mỗi $k$ bước để giảm tính tự tương quan (autocorrelation)
- Độ lệch chuẩn đề xuất (proposal scale): Quá nhỏ khiến chuỗi di chuyển chậm chạp (tỷ lệ chấp nhận cao nhưng thăm dò không gian kém); quá lớn khiến đa số đề xuất bị từ chối (tỷ lệ chấp nhận thấp, bị kẹt một chỗ)
- Tỷ lệ chấp nhận tối ưu cho đề xuất Gauss trong không gian nhiều chiều là xấp xỉ 0.234

### Lấy mẫu Gibbs (Gibbs Sampling)

Lấy mẫu Gibbs là một trường hợp đặc biệt của MCMC cho phân phối nhiều chiều. Thay vì đề xuất bước nhảy trên toàn bộ các chiều cùng lúc, nó cập nhật từng biến số một lần lượt dựa trên phân phối có điều kiện của biến đó.

```
Phân phối mục tiêu: p(x_1, x_2, ..., x_d)

Thuật toán:
  Tại mỗi vòng lặp t:
    Lấy mẫu x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Lấy mẫu x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Lấy mẫu x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Lấy mẫu Gibbs yêu cầu bạn phải lấy mẫu được từ từng phân phối có điều kiện $p(x_i \mid x_{-i})$. Điều này rất tự nhiên trong nhiều mô hình:
- Mạng Bayes: Các phân phối có điều kiện suy ra trực tiếp từ cấu trúc đồ thị
- Mô hình hỗn hợp Gauss: Các phân phối có điều kiện đều là phân phối chuẩn
- Mô hình Ising: Phân phối có điều kiện của mỗi spin chỉ phụ thuộc vào các spin láng giềng

Tỷ lệ chấp nhận luôn luôn là 1 (mọi đề xuất đều được chấp nhận) vì việc lấy mẫu từ đúng phân phối có điều kiện đã tự động thỏa mãn tính cân bằng chi tiết.

**Hạn chế:** Khi các biến có tương quan mạnh, lấy mẫu Gibbs chuyển dịch rất chậm vì việc chỉ di chuyển theo từng trục tọa độ riêng lẻ không thể thực hiện các bước nhảy chéo lớn qua phân phối.

### Lấy mẫu nhiệt độ (Temperature Sampling trong LLM)

Các mô hình ngôn ngữ xuất ra các logit $z_1, \ldots, z_V$ cho từng token trong từ điển. Softmax chuyển đổi chúng thành xác suất. Nhiệt độ (temperature $T$) định cỡ lại các logit trước khi đưa vào softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: Softmax tiêu chuẩn (giữ nguyên phân phối gốc)
T -> 0:  Argmax (tất định, luôn luôn chọn token có logit cao nhất)
T -> inf: Phân phối đều (mọi token đều có xác suất ngang nhau)
T < 1.0: Làm sắc nét phân phối (tự tin hơn, ít đa dạng hơn)
T > 1.0: Làm phẳng phân phối (ít tự tin hơn, đa dạng và sáng tạo hơn)
```

**Bản chất toán học:** Việc chia logit cho $T < 1$ sẽ phóng đại khoảng cách giữa các logit. Nếu $z_1 = 2$ và $z_2 = 1$, chia cho $T = 0.5$ ta được $z_1/T = 4$ và $z_2/T = 2$, khiến khoảng chênh lệch tăng gấp đôi. Sau khi qua softmax, token có logit cao nhất sẽ chiếm thị phần xác suất vượt trội.

**Cấu hình trong thực tế:**
- $T = 0.0$: Giải mã tham lam (greedy decoding), tối ưu cho trả lời câu hỏi thực tế, trích xuất thông tin
- $T = 0.3\text{-}0.7$: Sáng tạo nhẹ, rất tốt cho sinh mã nguồn (code generation)
- $T = 0.7\text{-}1.0$: Cân bằng, phù hợp cho hội thoại chung
- $T = 1.0\text{-}1.5$: Viết văn sáng tạo, lên ý tưởng (brainstorming)
- $T > 1.5$: Trở nên ngẫu nhiên hỗn loạn, hiếm khi hữu ích

Nhiệt độ không làm thay đổi tập token có thể xuất hiện, nó chỉ phân bổ lại tỷ trọng xác suất giữa các token đó.

### Lấy mẫu Top-k (Top-k Sampling)

Lấy mẫu Top-k giới hạn tập ứng viên vào $k$ token có xác suất cao nhất, sau đó chuẩn hóa lại xác suất và lấy mẫu từ tập thu gọn này.

```
Thuật toán:
  1. Tính xác suất softmax cho toàn bộ V token
  2. Sắp xếp các token theo xác suất giảm dần
  3. Chỉ giữ lại k token hàng đầu
  4. Chuẩn hóa lại: p_i' = p_i / sum(p_j với j nằm trong top-k)
  5. Lấy mẫu từ phân phối đã chuẩn hóa

k = 1:  Giải mã tham lam (greedy)
k = V:  Không lọc (lấy mẫu thông thường)
k = 40: Giá trị phổ biến, loại bỏ hoàn toàn phần đuôi dài gồm các token vô nghĩa
```

Top-k ngăn chặn mô hình chọn phải các token cực kỳ phi lý (lỗi chính tả, từ vô nghĩa) nằm ở đuôi dài của phân phối. Nhược điểm: $k$ là cố định bất kể ngữ cảnh. Khi mô hình rất tự tin (một token chiếm 95% xác suất), $k = 40$ vẫn giữ lại 39 lựa chọn thừa thãi. Khi mô hình phân vân (xác suất dàn đều trên 1000 token hợp lý), $k = 40$ lại cắt bỏ quá sớm các phương án khả dĩ.

### Lấy mẫu Top-p / Nucleus Sampling

Lấy mẫu Top-p giải quyết nhược điểm của Top-k bằng cách điều chỉnh động kích thước tập ứng viên. Thay vì giữ một số lượng $k$ cố định, nó giữ lại tập hợp nhỏ nhất các token có tổng xác suất tích lũy vượt qua ngưỡng $p$.

```
Thuật toán:
  1. Tính xác suất softmax cho toàn bộ V token
  2. Sắp xếp các token theo xác suất giảm dần
  3. Tìm k nhỏ nhất sao cho tổng xác suất của top-k token >= p
  4. Chỉ giữ lại k token đó
  5. Chuẩn hóa lại và lấy mẫu

p = 0.9:  Giữ lại các token bao phủ 90% khối lượng xác suất
p = 1.0:  Không lọc
p = 0.1:  Rất khắt khe, gần như tương đương greedy
```

Khi mô hình tự tin, nucleus sampling chỉ giữ lại rất ít token (có thể chỉ 2-3 từ). Khi mô hình phân vân, nó mở rộng tập ứng viên lên nhiều token (có thể tới 200 từ). Hành vi thích ứng linh hoạt này giúp nucleus sampling tạo ra văn bản tự nhiên hơn hẳn so với top-k.

**Các kết hợp phổ biến:**
- Temperature 0.7 + top-p 0.9: Cấu hình chuẩn tắc cho mục đích tổng quát
- Temperature 0.0 (greedy): Dành cho các tác vụ tất định
- Temperature 1.0 + top-k 50: Cấu hình đề xuất trong bài báo gốc của Fan et al. (2018)

Top-k và top-p có thể phối hợp cùng nhau: Áp dụng top-k trước để loại bỏ đuôi cực đoan, sau đó áp dụng top-p trên tập còn lại.

### Mẹo tái tham số hóa (Reparameterization Trick trong VAE)

Mạng biến phân tự mã hóa (Variational Autoencoders - VAE) học bằng cách mã hóa đầu vào thành một phân phối trong không gian tiềm ẩn, lấy mẫu từ phân phối đó, rồi giải mã mẫu để tái tạo lại đầu vào. Thách thức lớn: Bạn không thể lan truyền ngược gradient qua một phép toán lấy mẫu ngẫu nhiên.

```
Lấy mẫu thông thường (không khả vi):
  z ~ N(mu, sigma^2)

  Tính ngẫu nhiên chặn đứng dòng chảy gradient.
  d/d_mu [mẫu từ N(mu, sigma^2)] = ???
```

Mẹo tái tham số hóa tách tính ngẫu nhiên ra khỏi các tham số cần tối ưu:

```
Lấy mẫu tái tham số hóa:
  epsilon ~ N(0, 1)          (nhiễu ngẫu nhiên cố định, không có tham số)
  z = mu + sigma * epsilon   (hàm tất định của các tham số)

  Bây giờ z là một hàm tất định, khả vi theo mu và sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradient chảy mượt mà qua mu và sigma!
```

Điều này hoạt động được vì phân phối $\mathcal{N}(\mu, \sigma^2)$ tương đương về mặt toán học với $\mu + \sigma \cdot \mathcal{N}(0, 1)$. Điểm sáng tạo: Chuyển toàn bộ yếu tố ngẫu nhiên sang một nguồn không chứa tham số ($\epsilon$), sau đó biểu diễn mẫu dưới dạng một biến đổi khả vi của các tham số.

**Trong vòng lặp huấn luyện VAE:**
1. Bộ mã hóa xuất ra $\mu$ và $\log(\sigma^2)$ cho mỗi đầu vào
2. Lấy mẫu $\epsilon \sim \mathcal{N}(0, 1)$
3. Tính $z = \mu + \sigma \epsilon$
4. Bộ giải mã giải $z$ để tái tạo lại ảnh/dữ liệu ban đầu
5. Lan truyền ngược từ hàm mất mát qua các bước 4, 3, 2, 1 (hoàn toàn khả vi nhờ bước 3)

Nếu không có mẹo tái tham số hóa, VAE không thể huấn luyện được bằng backpropagation thông thường. Ý tưởng đơn giản này đã đưa VAE từ lý thuyết trở thành một công cụ sinh dữ liệu mạnh mẽ.

### Gumbel-Softmax (Lấy mẫu phân loại khả vi)

Mẹo tái tham số hóa áp dụng cho phân phối liên tục (Gaussian). Với phân phối phân loại rời rạc (categorical), chúng ta cần một cách tiếp cận khác: Gumbel-Softmax cung cấp một xấp xỉ khả vi cho phép lấy mẫu rời rạc.

**Mẹo Gumbel-Max (không khả vi):**

```
Để lấy mẫu từ phân phối phân loại với log-probabilities log(p_1), ..., log(p_k):
  1. Lấy mẫu g_i ~ Gumbel(0, 1) cho mỗi danh mục
     (g = -log(-log(u)), với u ~ Uniform(0, 1))
  2. Trả về argmax(log(p_i) + g_i)

Cách này tạo ra các mẫu phân loại chính xác tuyệt đối.
```

**Gumbel-Softmax (xấp xỉ khả vi):**

```
Thay thế hàm argmax cứng bằng hàm softmax mềm:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (nhiệt độ) kiểm soát mức độ xấp xỉ:
  tau -> 0:   Tiến gần về vector one-hot (phân loại cứng)
  tau -> inf: Tiến gần về phân phối đều (1/k, 1/k, ..., 1/k)
  tau = 1.0:  Xấp xỉ mềm
```

Gumbel-Softmax tạo ra một sự nới lỏng liên tục (continuous relaxation) của mẫu rời rạc. Đầu ra là một vector xác suất thay vì một vector one-hot cứng. Gradient có thể chảy qua hàm softmax. Trong quá trình huấn luyện, bạn có thể áp dụng bộ ước lượng "straight-through": Dùng argmax cứng ở lượt truyền xuôi (forward pass), nhưng dùng gradient của Gumbel-Softmax mềm ở lượt truyền ngược (backward pass).

**Ứng dụng:**
- Biến tiềm ẩn rời rạc trong VAE
- Tìm kiếm kiến trúc mạng nơ-ron (NAS - chọn thao tác rời rạc giữa các lớp)
- Cơ chế Hard Attention
- Học tăng cường với không gian hành động rời rạc

### Lấy mẫu phân tầng (Stratified Sampling)

Lấy mẫu Monte Carlo thông thường có thể vô tình để lại các khoảng trống trong không gian mẫu do tính ngẫu nhiên. Lấy mẫu phân tầng ép việc bao phủ đồng đều bằng cách chia không gian thành các tầng (strata) và lấy mẫu từ từng tầng một.

```
Monte Carlo thông thường:
  Lấy N điểm ngẫu nhiên đều từ [0, 1]
  Một số vùng có thể bị cụm lại, một số vùng khác bị bỏ trống

Lấy mẫu phân tầng:
  Chia [0, 1] thành N tầng bằng nhau: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Lấy một điểm ngẫu nhiên đều bên trong mỗi tầng
  x_i = (i + u_i) / N   với u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Lấy mẫu phân tầng luôn có phương sai nhỏ hơn hoặc bằng so với Monte Carlo tiêu chuẩn:

```
Var(stratified) <= Var(standard Monte Carlo)

Sự cải thiện rõ rệt nhất khi hàm mục tiêu f(x) biến thiên mượt mà.
Với các hàm hằng từng đoạn, lấy mẫu phân tầng cho kết quả chính xác tuyệt đối.
```

**Ứng dụng:**
- Tích phân số học (Quasi-Monte Carlo)
- Phân chia tập dữ liệu huấn luyện (đảm bảo cân bằng tỷ lệ các lớp trong từng fold)
- Kết hợp với Importance Sampling
- NeRF (Neural Radiance Fields) dùng lấy mẫu phân tầng dọc theo các tia camera

### Mối liên hệ với các mô hình khuếch tán (Diffusion Models)

Các mô hình Diffusion sinh ảnh thông qua một chuỗi các bước lấy mẫu nối tiếp nhau. Quá trình thuận (forward process) thêm dần nhiễu Gauss vào bức ảnh qua $T$ bước cho đến khi nó biến thành nhiễu trắng thuần túy. Quá trình nghịch đảo (reverse process) học cách khử nhiễu để khôi phục lại bức ảnh gốc từng bước một.

```
Quá trình thuận (đã biết trước):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  với epsilon ~ N(0, I)

  Sau T bước: x_T ~ N(0, I) (nhiễu thuần túy)

Quá trình nghịch đảo (mô hình học được):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  với z ~ N(0, I)

  Mỗi bước khử nhiễu là một bước lấy mẫu ngẫu nhiên.
```

Mối liên hệ với các phương pháp trong bài học này:
- Mỗi bước khử nhiễu sử dụng mẹo tái tham số hóa (lấy mẫu nhiễu, áp dụng biến đổi tất định)
- Lịch trình nhiễu $\lbrace \alpha_t \rbrace$ điều khiển một hình thức ủ nhiệt (temperature annealing)
- Huấn luyện mô hình sử dụng ước lượng Monte Carlo để xấp xỉ hàm mục tiêu ELBO
- Quá trình lấy mẫu tổ tiên (ancestral sampling) trong diffusion model thực chất là một chuỗi Markov

Toàn bộ quá trình sinh ảnh là việc lấy mẫu lặp đi lặp lại: Bắt đầu từ nhiễu thuần túy, và tại mỗi bước, lấy mẫu một phiên bản ít nhiễu hơn dựa trên mô hình khử nhiễu đã học.

```figure
monte-carlo-pi
```

## Build It (Thực hành xây dựng)

### Bước 1: Lấy mẫu đều và nghịch đảo CDF

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Sinh 10.000 mẫu phân phối mũ và kiểm tra xem giá trị trung bình có xấp xỉ $1/\lambda$ hay không.

### Bước 2: Lấy mẫu bác bỏ (Rejection sampling)

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Sử dụng rejection sampling để rút mẫu từ phân phối chuẩn bị cắt cụt. Kiểm tra hình dáng bằng biểu đồ histogram.

### Bước 3: Lấy mẫu tầm quan trọng (Importance sampling)

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Ước lượng $\mathbb{E}[X^2]$ dưới phân phối chuẩn bằng phân phối đề xuất đều. So sánh với nghiệm giải tích đã biết ($\mu^2 + \sigma^2$).

### Bước 4: Ước lượng số Pi bằng Monte Carlo

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Bước 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Lấy mẫu từ phân phối lưỡng đỉnh (hỗn hợp của 2 phân phối Gauss). Trực quan hóa quỹ đạo chuyển dịch của chuỗi.

### Bước 6: Lấy mẫu Gibbs

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Bước 7: Lấy mẫu theo nhiệt độ (Temperature sampling)

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Minh họa cách thay đổi nhiệt độ tác động lên phân phối xác suất đầu ra của các logit.

### Bước 8: Lấy mẫu Top-k và Top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Bước 9: Mẹo tái tham số hóa (Reparameterization trick)

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Chứng minh rằng gradient có thể truyền qua mẫu tái tham số hóa nhưng bị chặn đứng khi lấy mẫu trực tiếp.

### Bước 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Minh họa việc hạ nhiệt độ khiến đầu ra hội tụ về một vector one-hot.

Toàn bộ mã nguồn cài đặt chi tiết và các trực quan hóa nằm trong `code/sampling.py`.

## Use It (Ứng dụng thực tế)

Trong môi trường thực tế với NumPy và SciPy:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Đối với MCMC ở quy mô lớn trong sản xuất:
- PyMC: Mô hình hóa Bayes toàn diện với thuật toán NUTS (No-U-Turn Sampler)
- emcee: Trình lấy mẫu ensemble MCMC
- NumPyro / JAX: MCMC tăng tốc trên GPU

## Exercises (Bài tập)

1. Cài đặt phương pháp nghịch đảo CDF cho phân phối Cauchy. Hàm CDF là $F(x) = 0.5 + \arctan(x)/\pi$. Sinh 10.000 mẫu và vẽ histogram đối chiếu với PDF thực tế. Nhận xét về hiện tượng đuôi nặng (các giá trị cực đoan nằm rất xa tâm).

2. Dùng lấy mẫu bác bỏ để sinh mẫu từ phân phối Beta(2, 5) bằng phân phối đề xuất Uniform(0, 1). Vẽ đồ thị các mẫu được chấp nhận so với PDF Beta thực tế. Tỷ lệ chấp nhận trên lý thuyết là bao nhiêu?

3. Ước lượng tích phân của $\sin(x)$ từ 0 đến $\pi$ bằng Monte Carlo với 1.000, 10.000 và 100.000 mẫu. So sánh sai số ở từng mức và xác nhận rằng sai số giảm theo quy luật $O(1/\sqrt{N})$.

4. Cài đặt Metropolis-Hastings để lấy mẫu từ phân phối 2D $p(x, y)$ tỷ lệ với $\exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8y) / 2)$. Vẽ đồ thị các mẫu và quỹ đạo của chuỗi. Thử nghiệm với các độ lệch chuẩn đề xuất khác nhau.

5. Xây dựng một bản demo sinh văn bản hoàn chỉnh: Cho một từ vựng gồm 10 từ kèm logit, sinh các chuỗi dài 20 token bằng 4 chiến lược: (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. So sánh độ đa dạng của kết quả qua 5 lượt chạy độc lập.

### Gợi ý (Hints)

**Bài tập 1.** Giải phương trình $u = F(x)$ tìm $x$, ta được $x = \tan(\pi(u - 0.5))$. Kiểm tra xem trung vị mẫu có gần 0 không và khoảng 6% số mẫu có $\lvert x \rvert > 10$, tương ứng với $1 - (2/\pi)\arctan(10)$.

**Bài tập 2.** Hàm mật độ Beta(2, 5) là $30x(1-x)^4$. Tìm giá trị lớn nhất tại mode $x = 1/5$ và dùng giá trị đó làm $M$ trong `rejection_sample`. Tỷ lệ chấp nhận lý thuyết là $1/M$, hãy so sánh với tỷ lệ thực tế bạn chấp nhận.

**Bài tập 3.** Giá trị giải tích chính xác là 2. Gọi `monte_carlo_integral(math.sin, 0, math.pi, n)` cho từng $n$ và lặp lại với nhiều seed khác nhau. Khi $n$ tăng 10 lần, sai số trung bình sẽ giảm khoảng $\sqrt{10} \approx 3.2$ lần.

**Bài tập 4.** Lấy log của hàm mục tiêu: $-(x^2 y^2 + x^2 + y^2 - 8x - 8y)/2$, và truyền vào `metropolis_hastings_2d`. Ghi lại tỷ lệ chấp nhận cho từng mức độ lệch chuẩn đề xuất: giá trị quá nhỏ chấp nhận gần như mọi bước nhưng di chuyển rất chậm, giá trị quá lớn bác bỏ hầu hết các bước.

**Bài tập 5.** Giải mã greedy lấy chỉ số logit lớn nhất ở mỗi bước. Dùng `temperature_sample`, `top_k_sample` và `top_p_sample` cho 3 phương pháp còn lại. Greedy phải luôn sinh ra chuỗi giống hệt nhau ở cả 5 lần chạy, hãy đếm số chuỗi độc nhất của từng phương pháp để đo độ đa dạng.

## Key Terms (Thuật ngữ chính)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|------|----------------|----------------------|
| Sampling (Lấy mẫu) | "Rút giá trị ngẫu nhiên" | Tạo ra các giá trị tuân theo một phân phối xác suất. Cơ chế nền tảng của mọi hệ thống Generative AI |
| Uniform distribution | "Xác suất đều nhau" | Mọi giá trị trong $[a, b]$ có mật độ xác suất bằng $1/(b-a)$. Điểm khởi đầu của mọi phương pháp lấy mẫu |
| Inverse CDF | "Biến đổi xác suất" | $F^{-1}(U)$ chuyển đổi mẫu ngẫu nhiên đều thành mẫu từ bất kỳ phân phối nào có CDF đã biết. Chính xác và hiệu quả |
| Rejection sampling | "Đề xuất và chấp nhận/bác bỏ" | Sinh từ phân phối đề xuất đơn giản, chấp nhận với xác suất tỷ lệ với tỷ số giữa mục tiêu và đề xuất |
| Importance sampling | "Đánh lại trọng số mẫu" | Ước lượng kỳ vọng dưới $p(x)$ bằng mẫu từ $q(x)$ thông qua trọng số $p(x)/q(x)$. Cốt lõi của thuật toán PPO trong RL |
| Monte Carlo | "Lấy trung bình mẫu ngẫu nhiên" | Xấp xỉ tích phân bằng trung bình mẫu. Sai số giảm theo $O(1/\sqrt{N})$ không phụ thuộc số chiều |
| MCMC | "Bước đi ngẫu nhiên hội tụ" | Xây dựng một chuỗi Markov có phân phối dừng là phân phối mục tiêu. Metropolis-Hastings là thuật toán kinh điển |
| Metropolis-Hastings | "Lên dốc luôn nhận, xuống dốc tùy lúc" | Đề xuất bước nhảy, chấp nhận dựa trên tỷ số mật độ. Tính cân bằng chi tiết đảm bảo sự hội tụ |
| Gibbs sampling | "Cập nhật từng biến một" | Cập nhật từng biến từ phân phối có điều kiện khi cố định các biến còn lại. Tỷ lệ chấp nhận luôn là 100% |
| Temperature | "Núm vặn độ tự tin" | Chia các logit cho $T$ trước khi qua softmax. $T<1$ làm nhọn phân phối (tự tin hơn), $T>1$ làm phẳng (đa dạng hơn) |
| Top-k sampling | "Chỉ giữ lại k từ tốt nhất" | Triệt tiêu mọi token ngoài top $k$, chuẩn hóa lại rồi lấy mẫu. Kích thước tập ứng viên cố định |
| Nucleus sampling (top-p) | "Giữ lại nhóm token xác suất cao" | Giữ lại tập hợp token nhỏ nhất có tổng xác suất tích lũy vượt ngưỡng $p$. Kích thước ứng viên linh hoạt |
| Reparameterization trick | "Đưa tính ngẫu nhiên ra ngoài" | Biểu diễn $z = \mu + \sigma \epsilon$ với $\epsilon \sim \mathcal{N}(0,1)$. Giúp thao tác lấy mẫu trở nên khả vi để huấn luyện VAE |
| Gumbel-Softmax | "Lấy mẫu phân loại mềm" | Xấp xỉ khả vi cho lấy mẫu biến rời rạc bằng cách kết hợp nhiễu Gumbel với hàm softmax có nhiệt độ |
| Stratified sampling | "Lấy mẫu phân tầng" | Chia không gian mẫu thành các tầng đều nhau và lấy mẫu từ mỗi tầng. Luôn có phương sai thấp hơn Monte Carlo ngây thơ |
| Burn-in | "Giai đoạn khởi động" | Các mẫu MCMC ban đầu bị loại bỏ trước khi chuỗi đạt trạng thái dừng cân bằng |
| Detailed balance | "Tính cân bằng chi tiết" | Điều kiện thuận nghịch $p(x) T(x \to y) = p(y) T(y \to x)$, là điều kiện đủ để $p$ trở thành phân phối dừng |
| Diffusion sampling | "Khử nhiễu lặp lại" | Sinh dữ liệu bằng cách bắt đầu từ nhiễu thuần túy và áp dụng chuỗi các bước khử nhiễu đã học |

## Further Reading (Tài liệu đọc thêm)

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - Hướng dẫn chi tiết toàn diện về nền tảng MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Bài báo gốc giới thiệu kỹ thuật Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Bài báo đặt nền móng cho nucleus (top-p) sampling trong NLP
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Bài báo kinh điển về VAE giới thiệu mẹo tái tham số hóa
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM kết nối sâu sắc giữa việc lấy mẫu và sinh hình ảnh
