# Xác Suất & Các Phân Phối (Probability and Distributions)

> Xác suất là ngôn ngữ mà AI sử dụng để diễn đạt sự bất định.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 phút

## Mục Tiêu Học Tập

- Tự lập trình các hàm PMF và PDF từ đầu cho các phân phối Bernoulli, categorical, Poisson, đều (uniform) và chuẩn (Gaussian/normal)
- Tính toán kỳ vọng (expected value), phương sai (variance) và vận dụng Định lý Giới hạn Trung tâm (CLT) để giải thích vì sao phân phối chuẩn lại thống trị khắp nơi
- Xây dựng hàm softmax và log-softmax với kỹ thuật ổn định số học (trừ đi logit lớn nhất - subtract max logit)
- Tính toán hàm mất mát cross-entropy từ logits và kết nối nó với ước lượng hợp lý cực đại âm (negative log-likelihood)

## Vấn Đề

Một bộ phân loại đưa ra đầu ra `[0.03, 0.91, 0.06]`. Một mô hình ngôn ngữ lớn chọn từ tiếp theo từ 50.000 từ vựng tiềm năng. Một mô hình khuếch tán (diffusion model) tạo ra hình ảnh bằng cách lấy mẫu từ các phân phối đã học. Tất cả những điều này chính là xác suất trong thực tế.

Mọi dự đoán mà mô hình đưa ra đều là một phân phối xác suất. Mọi hàm mất mát đều đo lường khoảng cách giữa phân phối dự đoán và phân phối thực tế. Mọi bước huấn luyện đều điều chỉnh tham số để làm cho phân phối này khớp hơn với phân phối kia. Nếu không có xác suất, bạn không thể đọc hiểu các bài báo ML, không thể gỡ lỗi mô hình, hoặc không hiểu nổi tại sao hàm mất mát huấn luyện lại đột nhiên biến thành `NaN`.

## Khái Niệm

### Biến Cố, Không Gian Mẫu Và Xác Suất

Không gian mẫu $S$ là tập hợp tất cả các kết quả có thể xảy ra của một phép thử. Một biến cố (event) là một tập con của không gian mẫu. Xác suất là một hàm ánh xạ biến cố thành một con số nằm trong khoảng từ 0 đến 1.

```
Tung đồng xu:
  S = {Ngửa, Sấp}
  P(Ngửa) = 0.5,  P(Sấp) = 0.5

Gieo một con súc sắc:
  S = {1, 2, 3, 4, 5, 6}
  P(chẵn) = P({2, 4, 6}) = 3/6 = 0.5
```

Ba tiên đề định nghĩa toàn bộ nền tảng lý thuyết xác suất:
1. $P(A) \ge 0$ với mọi biến cố $A$
2. $P(S) = 1$ (chắc chắn một kết quả nào đó trong không gian mẫu sẽ xảy ra)
3. $P(A \text{ hoặc } B) = P(A) + P(B)$ khi $A$ và $B$ là hai biến cố xung khắc (không thể cùng xảy ra)

Mọi công thức khác (định lý Bayes, kỳ vọng, các phân phối) đều suy ra từ ba quy tắc cốt lõi này.

### Xác Suất Có Điều Kiện Và Tính Độc Lập

$P(A \mid B)$ là xác suất biến cố $A$ xảy ra khi đã biết biến cố $B$ đã xảy ra:

```
P(A|B) = P(A và B) / P(B)

Ví dụ: Rút một lá bài từ bộ bài 52 lá
  P(Vua | Lá hình người) = P(Vua và Lá hình) / P(Lá hình)
                         = (4/52) / (12/52)
                         = 4/12 = 1/3
```

Hai biến cố được gọi là độc lập khi việc biết biến cố này xảy ra không cung cấp thêm bất kỳ thông tin nào về biến cố kia:

```math
\begin{aligned}
\text{Độc lập:} \quad & P(A \mid B) = P(A) \\
\text{Tương đương với:} \quad & P(A \text{ và } B) = P(A) \cdot P(B)
\end{aligned}
```

Các lần tung đồng xu là độc lập. Rút bài không hoàn lại là không độc lập.

### Hàm Khối Xác Suất (PMF) vs Hàm Mật Độ Xác Suất (PDF)

Biến ngẫu nhiên rời rạc có **hàm khối xác suất (Probability Mass Function - PMF)**. Mỗi kết quả có một xác suất cụ thể mà bạn có thể đọc ra trực tiếp. Tổng xác suất của tất cả các kết quả rời rạc luôn bằng 1.

```
PMF: P(X = k)

Súc sắc chuẩn:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Tổng tất cả xác suất = 1
```

Biến ngẫu nhiên liên tục có **hàm mật độ xác suất (Probability Density Function - PDF)**. Mật độ tại một điểm đơn lẻ không phải là xác suất (xác suất tại một điểm chính xác bằng 0). Xác suất chỉ xuất hiện khi lấy tích phân của mật độ trên một khoảng:

```
PDF: f(x)

P(a <= X <= b) = tích phân của f(x) từ a đến b

f(x) có thể lớn hơn 1 (đây là mật độ, không phải xác suất)
Tích phân từ -vô cùng đến +vô cùng của f(x) dx = 1
```

Sự khác biệt này tối quan trọng trong ML: Đầu ra của mô hình phân loại là PMF (các lựa chọn rời rạc). Không gian tiềm ẩn của VAE sử dụng PDF (không gian liên tục).

### Các Phân Phối Phổ Biến

**Phân phối Bernoulli:** một phép thử, hai kết quả (thành công / thất bại). Dùng trong phân loại nhị phân.

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Kỳ vọng} = p, &\quad \text{Phương sai} = p(1-p)
\end{aligned}
```

**Phân phối Categorical:** một phép thử, $k$ kết quả rời rạc. Dùng trong phân loại đa lớp (đầu ra của hàm softmax).

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{với } \textstyle\sum_i p_i = 1 \\
& \text{Ví dụ: } P(\text{mèo}) = 0.7, \quad P(\text{chó}) = 0.2, \quad P(\text{chim}) = 0.1
\end{aligned}
```

**Phân phối Đều (Uniform):** tất cả các kết quả đều có khả năng xuất hiện ngang nhau. Dùng phổ biến để khởi tạo trọng số ngẫu nhiên.

```math
\begin{aligned}
\text{Rời rạc:} \quad & P(X = k) = 1/n \text{ với } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Liên tục:} \quad & f(x) = 1/(b-a) \text{ với } x \in [a, b]
\end{aligned}
```

**Phân phối Chuẩn (Normal / Gaussian):** đường cong hình chuông trứ danh. Đặc trưng bởi giá trị trung bình kỳ vọng ($\mu$) và phương sai ($\sigma^2$).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Phân phối chuẩn tắc: mu = 0, sigma = 1
  68% dữ liệu nằm trong khoảng 1 sigma
  95% dữ liệu nằm trong khoảng 2 sigma
  99.7% dữ liệu nằm trong khoảng 3 sigma
```

**Phân phối Poisson:** đếm số lần xuất hiện của các biến cố hiếm trong một khoảng cố định.

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Kỳ vọng} = \lambda, &\quad \text{Phương sai} = \lambda
\end{aligned}
```

### Kỳ Vọng Và Phương Sai

Kỳ vọng (Expected value) là giá trị trung bình có trọng số của các kết quả có thể xảy ra:

```math
\begin{aligned}
\text{Rời rạc:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Liên tục:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

Phương sai (Variance) đo lường mức độ phân tán của dữ liệu xung quanh giá trị trung bình:

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Độ lệch chuẩn (Standard deviation)} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

Trong ML, kỳ vọng xuất hiện dưới dạng hàm mất mát (mất mát trung bình trên toàn bộ phân phối dữ liệu). Phương sai phản ánh độ ổn định của mô hình: phương sai của gradient quá lớn đồng nghĩa với việc quá trình huấn luyện bị rung lắc, không ổn định.

### Phân Phối Đồng Thời Và Phân Phối Biên Duyên

Phân phối đồng thời (Joint distribution) $P(X, Y)$ mô tả xác suất xảy ra đồng thời của hai biến ngẫu nhiên.

Bảng ví dụ PMF đồng thời ($X$ = thời tiết, $Y$ = mang ô):

| | $Y=0$ (không mang ô) | $Y=1$ (có mang ô) | Phân phối biên $P(X)$ |
|---|---|---|---|
| $X=0$ (nắng) | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$ (mưa) | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Phân phối biên $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

Phân phối biên duyên (Marginal distribution) thu được bằng cách tính tổng triệt tiêu biến còn lại:

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

### Vì Sao Phân Phối Chuẩn Xuất Hiện Ở Khắp Nơi

Định lý Giới hạn Trung tâm (Central Limit Theorem - CLT): Trung bình cộng của nhiều biến ngẫu nhiên độc lập có cùng phân phối, sau khi chuẩn hóa, sẽ hội tụ về phân phối chuẩn (Gaussian), bất kể phân phối gốc có hình dạng kỳ dị thế nào, miễn là phương sai của nó hữu hạn.

```
Gieo 1 con súc sắc:      phân phối đều (phẳng lì)
Trung bình của 2 con súc sắc:  hình tam giác (có đỉnh)
Trung bình của 30 con súc sắc: đường cong hình chuông gần như hoàn hảo
```

Đó là lý do tại sao:
- Các sai số đo lường thực tế luôn có phân phối xấp xỉ chuẩn (tổng hợp của nhiều nguồn nhiễu nhỏ độc lập).
- Việc khởi tạo trọng số mạng nơ-ron thường dựa trên phân phối chuẩn.
- Nhiễu gradient trong SGD có phân phối xấp xỉ chuẩn.

### Log Xác Suất (Log Probabilities)

Xác suất thuần túy rất dễ gây ra lỗi tràn số dưới (numerical underflow). Khi nhân hàng chục xác suất nhỏ liên tiếp trong một câu văn dài, giá trị sẽ nhanh chóng chạm mức 0 tuyệt đối của số thực máy tính:

```math
\begin{aligned}
P(\text{câu văn}) &= P(\text{từ}_1) \cdot P(\text{từ}_2) \cdot \ldots \cdot P(\text{từ}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(bị underflow chỉ sau khoảng 30 từ)}
\end{aligned}
```

Log xác suất giải quyết triệt để vấn đề này bằng cách biến phép nhân thành phép cộng:

```math
\begin{aligned}
\log P(\text{câu văn}) &= \log P(\text{từ}_1) + \log P(\text{từ}_2) + \ldots + \log P(\text{từ}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{số hữu hạn ổn định (không bao giờ bị underflow)}
\end{aligned}
```

Quy tắc:
- $\log(a \cdot b) = \log(a) + \log(b)$
- Log xác suất luôn $\le 0$ (vì $0 < P \le 1$)
- Giá trị càng âm lớn tức là xác suất càng thấp
- Hàm mất mát Cross-entropy chính là giá trị âm của log xác suất của lớp đúng (negative log-likelihood)

### Softmax Dưới Góc Nhìn Phân Phối Xác Suất

Mạng nơ-ron xuất ra các điểm số thô chưa chuẩn hóa (logits). Hàm Softmax chuyển đổi các logits này thành một phân phối xác suất hợp lệ:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Tính chất:
  - Tất cả các đầu ra đều nằm trong khoảng (0, 1)
  - Tổng tất cả các đầu ra chính xác bằng 1
  - Bảo toàn thứ tự tương đối của các logits đầu vào
  - Hàm mũ exp() khuếch đại mạnh sự chênh lệch giữa các điểm số
```

**Mẹo tính toán số học ổn định (Softmax trick):** Trừ đi giá trị logit lớn nhất trước khi tính lũy thừa hàm mũ để ngăn chặn hiện tượng tràn số trên (overflow):

```
z = [100, 101, 102]
exp(102) = tràn số (overflow error!)

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (an toàn tuyệt đối)

Kết quả tỷ lệ xác suất hoàn toàn giống nhau, không bao giờ bị overflow.
```

### Lấy Mẫu (Sampling)

Lấy mẫu là hành động rút ra ngẫu nhiên các giá trị từ một phân phối xác suất:
- Dropout lấy mẫu ngẫu nhiên các nơ-ron cần tạm ngắt kết nối
- Tăng cường dữ liệu (data augmentation) lấy mẫu ngẫu nhiên các phép biến đổi
- Mô hình ngôn ngữ (LLM) lấy mẫu từ tiếp theo từ phân phối xác suất dự đoán
- Mô hình khuếch tán (diffusion) lấy mẫu nhiễu và khử nhiễu dần qua từng bước

```figure
gaussian-pdf
```

## Tự Xây Dựng (Build It)

### Bước 1: Nền tảng xác suất

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(Vua | Lá hình người) = {p_king_given_face:.4f}")
```

### Bước 2: Tự cài đặt PMF và PDF từ đầu

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Bước 3: Kỳ vọng và phương sai

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Súc sắc: E[X] = {mu:.4f}, Var(X) = {var:.4f}, Độ lệch chuẩn = {var**0.5:.4f}")
```

### Bước 4: Lấy mẫu từ các phân phối

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Bước 5: Softmax và Log xác suất ổn định số học

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Bước 6: Minh họa Định lý Giới hạn Trung tâm (CLT)

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

## Sử Dụng Thư Viện (Use It)

Với NumPy và SciPy, các thao tác trở nên cô đọng:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

## Bài Tập Thực Hành

1. Cài đặt thuật toán lấy mẫu biến đổi nghịch đảo (inverse transform sampling) cho phân phối mũ (exponential distribution). Lấy mẫu 10.000 giá trị và so sánh biểu đồ histogram với hàm mật độ xác suất PDF thực tế.
2. Xây dựng bảng phân phối đồng thời cho hai con súc sắc không đồng chất (loaded dice). Tính các phân phối biên và kiểm tra xem hai con súc sắc có độc lập với nhau hay không.
3. Tính hàm mất mát cross-entropy cho mô hình phân loại 5 lớp xuất ra logits `[2.0, 0.5, -1.0, 3.0, 0.1]` khi nhãn đúng nằm ở chỉ số 3. Sau đó kiểm chứng lại kết quả bằng `nn.CrossEntropyLoss` trong PyTorch.
4. Viết hàm nhận danh sách các log xác suất và trả về chuỗi có khả năng xảy ra cao nhất, tổng log xác suất và xác suất thuần tương đương. Thử nghiệm với một câu gồm 50 từ mà mỗi từ có xác suất 0.01.

### Gợi Ý

**Bài 1.** Lấy mẫu `u` từ phân phối đều trên khoảng (0, 1) và trả về $-\ln(1 - u) / \lambda$. Kiểm tra xem giá trị trung bình mẫu có xấp xỉ $1/\lambda$ và biểu đồ tần suất có bám sát hàm $\lambda e^{-\lambda x}$ hay không.

**Bài 2.** Tạo bảng 6x6 trong đó mỗi ô là một xác suất và tổng 36 ô bằng 1. Dùng hàm kiểm tra độc lập: hai con súc sắc độc lập khi và chỉ khi mọi ô trong bảng đều bằng tích của hai xác suất biên tương ứng.

**Bài 3.** Áp dụng `softmax` lên logits và lấy $-\ln$ của xác suất tại chỉ số 3. Kết quả thu được phải xấp xỉ 0.421, trùng khớp với giá trị mà hàm `nn.CrossEntropyLoss` của PyTorch tính ra.

**Bài 4.** Cộng dồn các log xác suất lại thay vì nhân các xác suất thô. Với 50 từ có xác suất 0.01, tổng log xác suất phải là $50 \ln(0.01) \approx -230.26$, và xác suất thuần tương đương là $10^{-100}$.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|-----------|--------------------|-------------------|
| Không gian mẫu (Sample space) | "Mọi khả năng" | Tập hợp $S$ chứa tất cả các kết quả có thể xảy ra của một phép thử ngẫu nhiên. |
| PMF | "Hàm xác suất" | Hàm khối xác suất cho biến ngẫu nhiên rời rạc, cho biết xác suất chính xác của từng kết quả và có tổng bằng 1. |
| PDF | "Đường cong xác suất" | Hàm mật độ xác suất cho biến ngẫu nhiên liên tục. Tích phân trên một khoảng mới tạo ra xác suất. |
| Xác suất có điều kiện | "Xác suất khi đã biết..." | $P(A \mid B) = P(A \text{ và } B) / P(B)$. Nền tảng của định lý Bayes và tư duy suy luận Bayes. |
| Tính độc lập | "Không ảnh hưởng nhau" | $P(A \text{ và } B) = P(A) \cdot P(B)$. Việc biết một biến cố không làm thay đổi xác suất của biến cố kia. |
| Kỳ vọng (Expected value) | "Giá trị trung bình" | Tổng có trọng số theo xác suất của tất cả các kết quả. Hàm mất mát trong ML chính là một kỳ vọng. |
| Phương sai (Variance) | "Độ phân tán" | Kỳ vọng của bình phương độ lệch so với trung bình. Phương sai gradient cao nghĩa là huấn luyện bị nhiễu. |
| Phân phối chuẩn (Normal) | "Đường cong hình chuông" | Phân phối Gauss đối xứng qua giá trị trung bình, xuất hiện khắp nơi do định lý CLT. |
| Định lý CLT | "Lấy trung bình ra hình chuông" | Trung bình cộng của nhiều mẫu ngẫu nhiên độc lập sẽ hội tụ về phân phối chuẩn khi số mẫu đủ lớn. |
| Phân phối đồng thời (Joint) | "Hai biến cùng lúc" | $P(X, Y)$ mô tả xác suất xảy ra của từng tổ hợp kết quả giữa $X$ và $Y$. |
| Phân phối biên (Marginal) | "Triệt tiêu biến còn lại" | $P(X) = \sum_y P(X, Y)$. Thu hồi phân phối của một biến đơn lẻ từ phân phối đồng thời. |
| Log xác suất | "Log của xác suất" | $\log P(x)$. Biến phép nhân thành phép cộng, ngăn chặn triệt để lỗi tràn số dưới (underflow). |
| Softmax | "Điểm số thành xác suất" | $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$. Ánh xạ các logits thực thành phân phối xác suất hợp lệ. |
| Cross-entropy | "Hàm mất mát phân loại" | $-\sum p_{\text{thực}} \cdot \log(p_{\text{dự đoán}})$. Đo lường khoảng cách giữa hai phân phối xác suất. |
| Logits | "Đầu ra thô của mô hình" | Điểm số chưa chuẩn hóa trước khi đưa vào hàm softmax. |
| Lấy mẫu (Sampling) | "Rút ngẫu nhiên" | Tạo ra các giá trị ngẫu nhiên tuân theo một phân phối xác suất cụ thể. |

## Tài Liệu Đọc Thêm

- [3Blue1Brown: Định lý Giới hạn Trung tâm là gì?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - giải thích trực quan xuất sắc vì sao trung bình mẫu lại trở thành phân phối chuẩn
- [Stanford CS229 Ôn tập Xác suất](https://cs229.stanford.edu/section/cs229-prob.pdf) - tài liệu cô đọng và đầy đủ về xác suất trong Machine Learning
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - lý do vì sao sự ổn định số học lại quan trọng và cách thực hiện
