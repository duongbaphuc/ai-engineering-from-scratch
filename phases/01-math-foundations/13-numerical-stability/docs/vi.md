# Tính ổn định số học (Numerical Stability)

> Dấu phẩy động (Floating point) là một sự trừu tượng hóa có kẽ hở (leaky abstraction). Nó sẽ cắn bạn trong quá trình huấn luyện mô hình khi bạn không hề ngờ tới.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Cài đặt hàm softmax và log-sum-exp ổn định số học bằng mẹo trừ giá trị cực đại (max-subtraction trick)
- Nhận diện các hiện tượng tràn số trên (overflow), tràn số dưới (underflow) và triệt tiêu thảm họa (catastrophic cancellation) trong tính toán dấu phẩy động
- Xác minh gradient giải tích (analytical gradients) so với gradient số học (numerical gradients) bằng phương pháp sai phân hữu hạn trung tâm
- Giải thích vì sao bfloat16 được ưu tiên hơn float16 trong huấn luyện và cách nhân tỷ lệ hàm mất mát (loss scaling) ngăn ngừa gradient tràn số dưới

## The Problem (Vấn đề)

Mô hình của bạn huấn luyện suốt 3 giờ đồng hồ, rồi đột nhiên giá trị loss biến thành `NaN`. Bạn thêm lệnh in giá trị. Các logit vẫn ổn ở bước 9.000. Đến bước 9.001 chúng trở thành `inf`. Đến bước 9.002 mọi gradient đều biến thành `nan` và quá trình huấn luyện coi như bị phá hủy hoàn toàn.

Hoặc: Mô hình của bạn hoàn thành toàn bộ quá trình huấn luyện nhưng độ chính xác lại kém hơn 2% so với con số mà bài báo khoa học công bố. Bạn kiểm tra lại tất cả mọi thứ: Kiến trúc trùng khớp, siêu tham số trùng khớp, tập dữ liệu trùng khớp. Vấn đề thực sự là bài báo dùng định dạng float32 trong khi bạn lại dùng float16 mà không có phép nhân tỷ lệ phù hợp. 32-bit sai số làm tròn tích lũy dần đã âm thầm bào mòn độ chính xác của bạn.

Hoặc: Bạn tự xây dựng hàm mất mát cross-entropy từ đầu. Nó hoạt động tốt trên các logit nhỏ. Nhưng khi logit vượt quá 100, nó trả về `inf`. Hàm softmax bị tràn số vì `exp(100)` lớn hơn giá trị lớn nhất mà float32 có thể biểu diễn. Mọi framework học máy đều xử lý việc này bằng một mẹo chỉ tốn 2 dòng code. Nhưng bạn lại không hề biết mẹo đó tồn tại.

Tính ổn định số học không phải là mối bận tâm mang tính lý thuyết suông. Đó là ranh giới giữa một lượt huấn luyện thành công và một lượt huấn luyện thất bại trong âm thầm. Mọi lỗi học máy nghiêm trọng mà bạn phải gỡ lỗi sau này cuối cùng đều quy về dấu phẩy động.

## The Concept (Khái niệm)

### Chuẩn IEEE 754: Cách máy tính lưu trữ số thực

Máy tính lưu trữ các số thực dưới dạng giá trị dấu phẩy động (floating point) tuân theo tiêu chuẩn IEEE 754. Một số thực dấu phẩy động gồm ba phần: bit dấu (sign bit), phần mũ (exponent) và phần định trị (mantissa hay significand).

```
Bố cục Float32 (tổng cộng 32 bit):
[1 bit dấu] [8 bit số mũ] [23 bit định trị]

Giá trị = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Phần định trị quyết định độ chính xác (có bao nhiêu chữ số có nghĩa). Phần mũ quyết định phạm vi giá trị (một số có thể lớn hay nhỏ tới mức nào).

```
Định dạng   Số bit   Số mũ   Định trị   Chữ số thập phân   Phạm vi (xấp xỉ)
float64     64       11      52         ~15-16             +/- 1.8e308
float32     32       8       23         ~7-8               +/- 3.4e38
float16     16       5       10         ~3-4               +/- 65.504
bfloat16    16       8       7          ~2-3               +/- 3.4e38
```

float32 cung cấp cho bạn khoảng 7 chữ số thập phân có nghĩa. Điều này có nghĩa là nó có thể phân biệt 1.0000001 và 1.0000002, nhưng không thể phân biệt giữa 1.00000001 và 1.00000002. Sau 7 chữ số, mọi thứ chỉ còn là nhiễu làm tròn.

float16 chỉ cung cấp khoảng 3 chữ số thập phân. Con số lớn nhất mà nó có thể biểu diễn là 65.504. Đây là một con số nhỏ đến mức đáng lo ngại đối với học máy, nơi các logit, gradient và activation thường xuyên vượt qua ngưỡng này.

bfloat16 là câu trả lời của Google cho vấn đề giới hạn phạm vi của float16. Nó giữ nguyên 8 bit số mũ như float32 (cùng phạm vi giá trị lên đến 3.4e38) nhưng chỉ có 7 bit định trị (độ chính xác thấp hơn float16). Khi huấn luyện mạng nơ-ron, phạm vi quan trọng hơn độ chính xác, vì vậy bfloat16 thường giành chiến thắng.

### Vì sao 0.1 + 0.2 != 0.3

Con số 0.1 không thể biểu diễn chính xác tuyệt đối trong hệ nhị phân dấu phẩy động. Trong cơ số 2, nó là một phân số vô hạn tuần hoàn:

```
0.1 trong hệ nhị phân = 0.0001100110011001100110011... (tuần hoàn vô tận)
```

Float32 cắt ngắn chuỗi này còn 23 bit định trị. Giá trị được lưu trữ xấp xỉ là 0.100000001490116. Tương tự, 0.2 được lưu xấp xỉ là 0.200000002980232. Tổng của chúng là 0.300000004470348, chứ không phải 0.3.

```
Trong Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Điều này gây ảnh hưởng nghiêm trọng trong học máy vì:

1. So sánh mất mát như `if loss < threshold` có thể đưa ra kết quả sai lệch
2. Việc tích lũy nhiều giá trị nhỏ (cập nhật gradient qua hàng nghìn bước) sẽ dần trôi xa khỏi tổng thực tế
3. Các bài kiểm tra tính tái lập (reproducibility) và checksum sẽ thất bại nếu bạn so sánh số thực bằng toán tử `==`

Giải pháp: Không bao giờ so sánh số thực bằng `==`. Hãy dùng `abs(a - b) < epsilon` hoặc `math.isclose()`.

### Triệt tiêu thảm họa (Catastrophic Cancellation)

Khi bạn trừ hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa sẽ triệt tiêu lẫn nhau và bạn chỉ còn lại phần nhiễu làm tròn được đẩy lên thành các chữ số đầu tiên.

```
a = 1.0000001    (lưu trữ dưới dạng 1.00000011920929 trong float32)
b = 1.0000000    (lưu trữ dưới dạng 1.00000000000000 trong float32)

Hiệu thực tế:     0.0000001
Tính toán ra:     0.00000011920929

Sai số tương đối: 19.2%
```

Đó là sai số tương đối 19% phát sinh chỉ từ một phép trừ duy nhất. Trong học máy, điều này xảy ra bất cứ khi nào bạn:

- Tính phương sai của dữ liệu có giá trị trung bình lớn: $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ khi $\mathbb{E}[x]$ rất lớn
- Trừ các log-probability gần bằng nhau
- Tính đạo hàm số học bằng sai phân hữu hạn với epsilon quá nhỏ

Giải pháp: Biến đổi lại công thức để tránh việc trừ hai số lớn gần bằng nhau. Đối với phương sai, hãy dùng thuật toán Welford hoặc trừ trung bình dữ liệu trước. Đối với log-probability, hãy tính toán hoàn toàn trong không gian log.

### Tràn số trên (Overflow) và tràn số dưới (Underflow)

Tràn số trên (Overflow) xảy ra khi kết quả quá lớn để biểu diễn. Tràn số dưới (Underflow) xảy ra khi kết quả quá nhỏ (gần số 0 hơn số dương nhỏ nhất có thể biểu diễn).

```
Giới hạn của Float32:
  Lớn nhất:                  3.4028235e+38
  Dương nhỏ nhất (chuẩn):    1.175e-38
  Dương nhỏ nhất (bất chuẩn):1.401e-45
  Tràn số trên:              mọi số > 3.4e38 biến thành inf
  Tràn số dưới:              mọi số < 1.4e-45 biến thành 0.0
```

Hàm `exp()` là nguồn cơn chính gây tràn số trên trong học máy:

```
exp(88.7)  = 3.40e+38   (vừa khít trong float32)
exp(89.0)  = inf         (tràn số trên)
exp(-87.3) = 1.18e-38   (vừa trên ngưỡng tràn số dưới)
exp(-104)  = 0.0         (tràn số dưới về 0)
```

Hàm `log()` lại gặp vấn đề ở chiều ngược lại:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (vẫn ổn)
log(1e-46) = -inf        (đầu vào bị tràn số dưới về 0, dẫn tới log(0) = -inf)
```

Trong học máy, `exp()` xuất hiện trong softmax, sigmoid và các phép tính xác suất. `log()` xuất hiện trong cross-entropy, log-likelihood và khoảng cách KL divergence. Sự kết hợp `log(exp(x))` là một bãi mìn nếu không có các kỹ thuật số học thích hợp.

### Mẹo Log-Sum-Exp (Log-Sum-Exp Trick)

Tính toán trực tiếp `log(sum(exp(x_i)))` rất nguy hiểm về mặt số học. Nếu có bất kỳ $x_i$ nào lớn, `exp(x_i)` sẽ tràn số trên. Nếu toàn bộ $x_i$ đều là số âm rất lớn, mọi `exp(x_i)` sẽ tràn số dưới về 0 và `log(0)` sẽ biến thành `-inf`.

Mẹo xử lý: Trừ đi giá trị lớn nhất trước khi lấy hàm mũ.

```math
\log\left(\sum_i \exp(x_i)\right) = \max(x) + \log\left(\sum_i \exp(x_i - \max(x))\right)
```

Tại sao cách này lại hiệu quả: Sau khi trừ `max(x)`, số mũ lớn nhất sẽ là `exp(0) = 1`. Không thể xảy ra tràn số trên. Có ít nhất một số hạng trong tổng bằng 1, do đó tổng luôn $\ge 1$, và `log(1) = 0`. Hiện tượng tràn số dưới về `-inf` bị triệt tiêu hoàn toàn.

Chứng minh toán học:

```math
\begin{aligned}
& \log\left(\sum_i \exp(x_i)\right) \\
&= \log\left(\sum_i \exp(x_i - c + c)\right) && \text{(cộng và trừ } c\text{)} \\
&= \log\left(\sum_i \exp(x_i - c) \cdot \exp(c)\right) && (\exp(a+b) = \exp(a) \cdot \exp(b)) \\
&= \log\left(\exp(c) \cdot \sum_i \exp(x_i - c)\right) && \text{(đưa } \exp(c) \text{ ra ngoài)} \\
&= c + \log\left(\sum_i \exp(x_i - c)\right) && (\log(a \cdot b) = \log(a) + \log(b))
\end{aligned}
```

Đặt `c = max(x)` và hiện tượng tràn số biến mất.

Mẹo này xuất hiện khắp mọi nơi trong học máy:
- Chuẩn hóa Softmax
- Tính toán mất mát Cross-entropy
- Tính tổng log-probability trong các mô hình chuỗi
- Mô hình hỗn hợp Gauss (Mixture of Gaussians)
- Suy luận biến phân (Variational inference)

### Vì sao Softmax cần mẹo trừ giá trị cực đại

Softmax chuyển đổi các logit thành phân phối xác suất:

```math
\mathop{\mathrm{softmax}}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

Nếu không có mẹo trừ cực đại, các logit [100, 101, 102] sẽ gây tràn số trên:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

2.69e43 > 3.4e38 (ngưỡng tối đa float32), nên mọi số hạng đều tràn số.
exp(88.7) đã chạm ngưỡng giới hạn của float32.
exp(100) = inf trong float32.
```

Với mẹo xử lý, ta trừ đi max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Các giá trị xác suất thu được là giống hệt nhau về mặt toán học, nhưng quá trình tính toán hoàn toàn an toàn. Đây không phải là một thủ thuật tối ưu hóa thông thường, mà là yêu cầu bắt buộc để đảm bảo tính đúng đắn.

### NaN và Inf: Cách phát hiện và phòng tránh

`nan` (Not a Number) và `inf` (vô cực) lây lan như virus trong chuỗi tính toán. Một giá trị `nan` trong cập nhật gradient sẽ biến trọng số thành `nan`, kéo theo mọi đầu ra tiếp theo đều trở thành `nan`. Quá trình huấn luyện sụp đổ chỉ sau một bước.

Nguyên nhân xuất hiện `inf`:
- `exp()` của một số dương lớn
- Chia cho 0: `1.0 / 0.0`
- Tràn số `float32` trong các phép tích lũy cộng dồn

Nguyên nhân xuất hiện `nan`:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` của một số âm
- `log()` của một số âm
- Bất kỳ phép toán số học nào tác động lên một giá trị `nan` đã có trước đó

Phát hiện:

```python
import math

math.isnan(x)       # True nếu x là nan
math.isinf(x)       # True nếu x là +inf hoặc -inf
math.isfinite(x)    # True nếu x là số hữu hạn (không phải nan hay inf)
```

Chiến lược phòng ngừa:

1. Kẹp (clamp) đầu vào của `exp()`: `exp(clamp(x, -80, 80))`
2. Thêm epsilon vào mẫu số: `x / (y + 1e-8)`
3. Thêm epsilon bên trong `log()`: `log(x + 1e-8)`
4. Sử dụng các cài đặt ổn định (log-sum-exp, stable softmax)
5. Cắt tỉa gradient (gradient clipping) để chống bùng nổ trọng số
6. Kiểm tra `nan`/`inf` sau mỗi lượt truyền xuôi khi gỡ lỗi

### Kiểm tra gradient số học (Numerical Gradient Checking)

Gradient giải tích (tính từ lan truyền ngược backpropagation) có thể chứa lỗi logic code. Việc kiểm tra gradient số học giúp xác minh chúng bằng cách tính đạo hàm thông qua sai phân hữu hạn.

Công thức sai phân trung tâm:

```math
\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
```

Công thức này đạt độ chính xác bậc $O(h^2)$, tốt hơn nhiều so với sai phân tiến $(f(x+h) - f(x)) / h$ vốn chỉ đạt độ chính xác $O(h)$.

Lựa chọn $h$: Quá lớn thì phép xấp xỉ sai lệch. Quá nhỏ thì hiện tượng triệt tiêu thảm họa sẽ phá hủy kết quả. Giá trị `h = 1e-5` đến `1e-7` là mức tiêu chuẩn.

Kiểm tra: Tính sai số tương đối giữa gradient giải tích và gradient số học:

```math
\text{relative\_error} = \frac{\lvert \text{grad}_{\text{analytical}} - \text{grad}_{\text{numerical}} \rvert}{\max(\lvert \text{grad}_{\text{analytical}} \rvert, \lvert \text{grad}_{\text{numerical}} \rvert, 10^{-8})}
```

Quy tắc phán đoán:
- relative_error < 1e-7: Hoàn hảo, gradient hoàn toàn chính xác
- relative_error < 1e-5: Chấp nhận được, hầu như chắc chắn đúng
- relative_error > 1e-3: Có điều gì đó bất thường
- relative_error > 1: Gradient hoàn toàn sai

Luôn kiểm tra gradient khi viết một layer mới hoặc một hàm mất mát mới. PyTorch cung cấp sẵn hàm `torch.autograd.gradcheck()` phục vụ mục đích này.

### Huấn luyện độ chính xác hỗn hợp (Mixed Precision Training)

Các GPU hiện đại có phần cứng chuyên dụng (Tensor Cores) giúp tính toán phép nhân ma trận float16 nhanh hơn 2-8 lần so với float32. Huấn luyện độ chính xác hỗn hợp tận dụng lợi thế này:

```
1. Duy trì một bản sao trọng số chính (master copy) ở định dạng float32
2. Lượt truyền xuôi (forward pass) thực hiện ở float16 (rất nhanh)
3. Tính hàm mất mát ở float32 (chống tràn số)
4. Lượt truyền ngược (backward pass) thực hiện ở float16 (nhanh)
5. Chuyển đổi và nhân tỷ lệ gradient về float32
6. Cập nhật vào bản sao trọng số chính float32
```

Vấn đề khi chỉ huấn luyện bằng float16 thuần túy: Các gradient thường rất nhỏ (1e-8 hoặc nhỏ hơn). Float16 sẽ làm tràn số dưới mọi giá trị dưới ~6e-8 về 0. Mô hình ngừng học vì toàn bộ các cập nhật gradient đều bằng 0.

Giải pháp là nhân tỷ lệ mất mát (loss scaling):

```
1. Nhân loss với một hệ số lớn (ví dụ: 1024)
2. Lượt truyền ngược tính gradient của (loss * 1024)
3. Toàn bộ gradient đều lớn hơn 1024 lần (vượt lên trên ngưỡng tràn số dưới của float16)
4. Chia gradient cho 1024 trước khi cập nhật trọng số
5. Tác động thuần túy: Bước cập nhật giữ nguyên, nhưng không bị mất mát thông tin do tràn số dưới
```

Dynamic loss scaling sẽ tự động điều chỉnh hệ số này: Bắt đầu với giá trị lớn (65536). Nếu gradient tràn số thành `inf`, hãy giảm một nửa hệ số. Nếu qua $N$ bước mà không có hiện tượng tràn số, hãy nhân đôi hệ số.

### So sánh bfloat16 và float16: Vì sao bfloat16 thống trị trong huấn luyện

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 có độ chính xác cao hơn (10 bit định trị so với 7) nhưng phạm vi bị hạn chế (tối đa ~65.504). bfloat16 có độ chính xác thấp hơn nhưng phạm vi tương đương float32 (tối đa ~3.4e38).

Trong huấn luyện mạng nơ-ron:
- Activation và logit thường xuyên vượt ngưỡng 65.504 trong các đột biến huấn luyện. float16 sẽ bị tràn số; bfloat16 xử lý mượt mà.
- Loss scaling là bắt buộc đối với float16 nhưng thường không cần thiết đối với bfloat16 vì dải số mũ của nó bao trùm toàn bộ phổ độ lớn gradient.
- bfloat16 thực chất là phép cắt ngắn đơn giản của float32: loại bỏ 16 bit cuối của phần định trị. Phép chuyển đổi diễn ra tức thì và bảo toàn hoàn hảo phần số mũ.

float16 được ưu tiên cho giai đoạn suy luận (inference) khi các giá trị nằm trong ngưỡng giới hạn và độ chính xác quan trọng hơn. bfloat16 được ưu tiên cho huấn luyện nơi phạm vi giá trị là yếu tố sống còn. Đây là lý do tại sao các chip TPU và GPU NVIDIA hiện đại (A100, H100) đều hỗ trợ bfloat16 ở cấp phần cứng.

### Cắt tỉa Gradient (Gradient Clipping)

Hiện tượng bùng nổ gradient (exploding gradients) xảy ra khi gradient tăng theo cấp số nhân qua nhiều tầng (phổ biến trong RNN, mạng nơ-ron sâu và transformer). Một gradient đột biến có thể phá hỏng toàn bộ trọng số mô hình chỉ trong một bước lặp.

Hai kỹ thuật cắt tỉa:

**Cắt tỉa theo giá trị (Clip by value):** Giới hạn từng phần tử gradient một cách độc lập:

```
grad = clamp(grad, -max_val, max_val)
```

Đơn giản nhưng có thể làm thay đổi hướng của vector gradient.

**Cắt tỉa theo chuẩn (Clip by norm):** Co tỷ lệ toàn bộ vector gradient để chuẩn của nó không vượt quá ngưỡng cho phép:

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Bảo toàn phương hướng của vector gradient. Đây là cơ chế của hàm `torch.nn.utils.clip_grad_norm_()` và là tiêu chuẩn công nghiệp.

Các giá trị tiêu chuẩn: `max_norm=1.0` cho transformer, `max_norm=0.5` cho học tăng cường (RL), `max_norm=5.0` cho các mạng đơn giản hơn.

Cắt tỉa gradient không phải là giải pháp tạm bợ. Đó là một cơ chế an toàn cốt lõi. Không có nó, một lô dữ liệu dị biệt (outlier) duy nhất cũng có thể tạo ra gradient đủ lớn để thổi bay kết quả của hàng tuần huấn luyện.

### Các lớp chuẩn hóa như những bộ ổn định số học

Batch normalization, layer normalization và RMS normalization thường được giới thiệu như các kỹ thuật điều chuẩn (regularizer) giúp huấn luyện hội tụ nhanh hơn. Nhưng chúng cũng chính là những bộ ổn định số học quan trọng.

Nếu không có chuẩn hóa, activation có thể phình to hoặc teo nhỏ theo cấp số nhân qua các tầng:

```
Layer 1: giá trị trong [0, 1]
Layer 5: giá trị trong [0, 100]
Layer 10: giá trị trong [0, 10.000]
Layer 50: giá trị trong [0, inf]
```

Chuẩn hóa sẽ căn giữa và định cỡ lại các activation tại mỗi tầng:

```math
\mathop{\mathrm{LayerNorm}}(x) = \frac{x - \mathop{\mathrm{mean}}(x)}{\mathop{\mathrm{std}}(x) + \epsilon} \cdot \gamma + \beta
```

Hằng số `epsilon` (thường là 1e-5) ngăn chặn phép chia cho 0 khi tất cả các activation đều giống hệt nhau. Các tham số học được `gamma` và `beta` cho phép mạng khôi phục lại bất kỳ thang đo nào cần thiết.

Điều này giữ cho các giá trị luôn nằm trong khoảng an toàn số học xuyên suốt toàn mạng, ngăn ngừa cả hiện tượng tràn số ở lượt truyền xuôi và bùng nổ gradient ở lượt truyền ngược.

### Các lỗi số học thường gặp trong Machine Learning

**Lỗi: Loss biến thành NaN sau một vài epoch.**
Nguyên nhân: Logit tăng quá lớn, softmax bị tràn số. Hoặc learning rate quá cao khiến trọng số phân kỳ.
Khắc phục: Dùng stable softmax (mẹo trừ max), giảm learning rate, bổ sung gradient clipping.

**Lỗi: Loss bị kẹt ở mức log(num_classes).**
Nguyên nhân: Đầu ra mô hình là các xác suất gần như đều nhau. Thường do gradient biến mất (vanishing) hoặc mô hình hoàn toàn không học được gì.
Khắc phục: Kiểm tra nhãn dữ liệu, xác minh hàm mất mát, kiểm tra hiện tượng chết nơ-ron ReLU (dead ReLUs).

**Lỗi: Độ chính xác tập kiểm định (validation) thấp hơn kỳ vọng từ 1-3%.**
Nguyên nhân: Huấn luyện mixed precision nhưng không cấu hình loss scaling đúng cách. Gradient tràn số dưới đã âm thầm triệt tiêu các cập nhật nhỏ.
Khắc phục: Bật dynamic loss scaling hoặc chuyển sang dùng bfloat16.

**Lỗi: Chuẩn gradient bằng 0.0 ở một số tầng.**
Nguyên nhân: Hiện tượng dead ReLU (tất cả đầu vào âm), hoặc float16 bị tràn số dưới.
Khắc phục: Chuyển sang dùng LeakyReLU hoặc GELU, dùng gradient scaling, kiểm tra khởi tạo trọng số ban đầu.

**Lỗi: Mô hình chạy ra một kết quả trên GPU này nhưng ra kết quả khác trên GPU khác.**
Nguyên nhân: Thứ tự tích lũy dấu phẩy động không có tính tất định (non-deterministic). Thuật toán song song trên GPU tính tổng theo thứ tự khác nhau trên từng kiến trúc phần cứng, trong khi phép cộng dấu phẩy động không có tính kết hợp.
Khắc phục: Chấp nhận sai số nhỏ (1e-6), hoặc bật `torch.use_deterministic_algorithms(True)` và chấp nhận giảm tốc độ thực thi.

**Lỗi: `exp()` trả về `inf` trong khi tính loss.**
Nguyên nhân: Đưa trực tiếp logit thô vào `exp()` mà không dùng mẹo trừ cực đại.
Khắc phục: Dùng `torch.nn.functional.log_softmax()`, hàm này đã tích hợp sẵn log-sum-exp bên trong.

**Lỗi: Huấn luyện phân kỳ sau khi chuyển từ float32 sang float16.**
Nguyên nhân: float16 không thể biểu diễn độ lớn gradient dưới 6e-8 hoặc activation trên 65.504.
Khắc phục: Dùng mixed precision có loss scaling (AMP), hoặc chuyển sang dùng bfloat16.

```figure
logsumexp-stability
```

## Build It (Thực hành xây dựng)

### Bước 1: Minh họa giới hạn độ chính xác dấu phẩy động

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Bước 2: Cài đặt softmax ngây thơ vs softmax ổn định

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) sẽ trả về [nan, nan, nan]
```

### Bước 3: Cài đặt log-sum-exp ổn định

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) trả về inf
```

### Bước 4: Cài đặt cross-entropy ổn định

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Bước 5: Kiểm tra gradient số học (Gradient checking)

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It (Ứng dụng thực tế)

### Mô phỏng độ chính xác hỗn hợp

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Cắt tỉa gradient (Gradient clipping)

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Phát hiện NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Xem `code/numerical.py` để tìm hiểu các cài đặt hoàn chỉnh với toàn bộ các trường hợp biên được kiểm thử.

## Ship It (Sản phẩm bàn giao)

Bài học này xuất xưởng:
- `code/numerical.py` chứa softmax ổn định, log-sum-exp, cross-entropy, gradient checking và mô phỏng mixed precision
- `outputs/prompt-numerical-debugger.md` phục vụ việc chẩn đoán NaN/Inf và các sự cố số học trong huấn luyện

Các cài đặt ổn định này sẽ xuất hiện lại trong Phase 3 khi xây dựng vòng lặp huấn luyện (training loop) và trong Phase 4 khi cài đặt các cơ chế attention.

## Exercises (Bài tập)

1. **Triệt tiêu thảm họa.** Tính phương sai của dãy [1000000.0, 1000001.0, 1000002.0] bằng công thức ngây thơ $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ trong float32. Sau đó tính bằng thuật toán trực tuyến của Welford. So sánh các sai số đối chiếu với phương sai thực (0.6667).

2. **Săn tìm độ chính xác.** Tìm giá trị dương nhỏ nhất của float32 `x` sao cho `1.0 + x == 1.0` trong Python. Đây chính là epsilon máy (machine epsilon). Xác nhận rằng nó khớp với `numpy.finfo(numpy.float32).eps`.

3. **Các trường hợp biên của Log-sum-exp.** Kiểm thử hàm `logsumexp_stable` với: (a) tất cả các giá trị bằng nhau, (b) một giá trị lớn vượt trội so với phần còn lại, (c) toàn bộ giá trị đều là số âm rất lớn (-1000). Xác minh rằng nó đưa ra kết quả đúng trong khi phiên bản ngây thơ thất bại.

4. **Kiểm tra gradient của một lớp mạng nơ-ron.** Cài đặt một lớp tuyến tính đơn lẻ `y = Wx + b` cùng lượt truyền ngược giải tích của nó. Sử dụng `numerical_gradient` để kiểm tra tính đúng đắn cho ma trận trọng số kích thước 3x2.

5. **Thực nghiệm Loss scaling.** Mô phỏng quá trình huấn luyện với float16: tạo các gradient ngẫu nhiên trong khoảng [1e-9, 1e-3], chuyển đổi sang float16 và đo tỷ lệ phần trăm gradient bị biến thành 0. Sau đó áp dụng loss scaling (nhân với 1024), chuyển sang float16, chia ngược lại rồi đo lại tỷ lệ phần trăm số 0.

### Gợi ý (Hints)

**Bài tập 1.** Chuyển đổi 3 giá trị bằng `np.float32` trước khi bình phương, sau đó gọi `variance_naive` và `welford_variance`. Một giá trị gần $10^{12}$ trong float32 có khoảng cách giữa các số biểu diễn được lớn hơn 1 rất nhiều, do đó hãy kỳ vọng kết quả ngây thơ sẽ cách rất xa 0.6667 còn thuật toán Welford sẽ rất sát.

**Bài tập 2.** Bắt đầu với `x = np.float32(1.0)` và chia đôi nó chừng nào `np.float32(1.0) + x != np.float32(1.0)`. Giữ mọi giá trị ở float32, sau đó so sánh giá trị cuối cùng vẫn làm thay đổi tổng với `numpy.finfo(numpy.float32).eps`, tức là $2^{-23}$.

**Bài tập 3.** Hãy dự đoán câu trả lời trước. Với $n$ giá trị bằng nhau $c$, kết quả là $c + \ln n$. Với một giá trị lớn vượt trội, kết quả xấp xỉ chính giá trị đó. Với các giá trị quanh -1000, kết quả ở khoảng $-1000 + \ln n$, trong khi `logsumexp_naive` trả về `-inf` do mỗi hàm mũ đều tràn số dưới về 0.

**Bài tập 4.** Với $y = Wx + b$ cùng hàm mất mát $L$, gradient đối với $W$ là tích ngoài của gradient từ tầng trên truyền xuống và $x$. Bọc hàm loss thành một hàm nhận vector phẳng của $W$, gọi `numerical_gradient` và truyền cả hai kết quả vào `check_gradient`.

**Bài tập 5.** Lấy mẫu độ lớn trên thang log bằng `10 ** np.random.uniform(-9, -3, n)` và chuyển đổi bằng `np.float16`. Float16 không thể giữ các giá trị dưới khoảng $6 \times 10^{-8}$, vì vậy hãy kỳ vọng khoảng 1/4 số giá trị sẽ biến thành 0 trước khi scale và hầu như không còn số nào bị triệt tiêu sau khi đã nhân với 1024.

## Key Terms (Thuật ngữ chính)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa bản chất |
|------|----------------|----------------------|
| IEEE 754 | "Chuẩn float" | Tiêu chuẩn quốc tế định nghĩa các định dạng dấu phẩy động nhị phân, quy tắc làm tròn và các giá trị đặc biệt (inf, nan). Mọi CPU và GPU hiện đại đều triển khai chuẩn này. |
| Machine epsilon | "Giới hạn độ chính xác" | Giá trị nhỏ nhất $e$ sao cho $1.0 + e \ne 1.0$ trong một định dạng float nhất định. Đối với float32, giá trị này khoảng 1.19e-7. |
| Catastrophic cancellation (Triệt tiêu thảm họa) | "Mất độ chính xác do phép trừ" | Hiện tượng khi trừ hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa triệt tiêu lẫn nhau và nhiễu làm tròn chiếm lĩnh kết quả. |
| Overflow (Tràn số trên) | "Số quá lớn" | Kết quả vượt quá giá trị lớn nhất có thể biểu diễn và trở thành `inf`. `exp(89)` gây tràn số trên trong float32. |
| Underflow (Tràn số dưới) | "Số quá nhỏ" | Kết quả nằm gần 0 hơn số dương nhỏ nhất có thể biểu diễn và biến thành `0.0`. `exp(-104)` gây tràn số dưới trong float32. |
| Log-sum-exp trick | "Trừ giá trị lớn nhất trước" | Tính $\log(\sum \exp(x))$ bằng cách đưa $\exp(\max(x))$ ra ngoài để ngăn chặn tràn số trên và tràn số dưới. Dùng trong softmax, cross-entropy và tính toán log-probability. |
| Stable softmax | "Softmax không bị bùng nổ" | Trừ $\max(\text{logits})$ trước khi lấy hàm mũ. Kết quả giống hệt về mặt toán học nhưng triệt tiêu hoàn toàn nguy cơ tràn số. |
| Gradient checking | "Kiểm tra lại backprop" | So sánh gradient giải tích từ backpropagation với gradient số học từ sai phân hữu hạn để phát hiện lỗi lập trình. |
| Mixed precision (Độ chính xác hỗn hợp) | "Forward bằng float16, backward bằng float32" | Dùng float độ chính xác thấp hơn cho các phép toán yêu cầu tốc độ và float độ chính xác cao hơn cho các phép toán nhạy cảm số học. Tăng tốc thông thường từ 2-3 lần. |
| Loss scaling | "Ngăn chặn gradient tràn số dưới" | Nhân loss với một hằng số lớn trước khi backprop để gradient nằm trong khoảng biểu diễn của float16, sau đó chia lại cho hằng số đó trước khi cập nhật trọng số. |
| bfloat16 | "Brain floating point" | Định dạng 16-bit của Google với 8 bit số mũ (cùng phạm vi như float32) và 7 bit định trị (độ chính xác thấp hơn float16). Được ưu tiên số một cho huấn luyện. |
| Gradient clipping | "Giới hạn chuẩn gradient" | Co tỷ lệ vector gradient sao cho chuẩn của nó không vượt quá ngưỡng cho phép. Ngăn ngừa gradient bùng nổ phá hủy trọng số. |
| NaN | "Không phải là số (Not a Number)" | Giá trị float đặc biệt sinh ra từ các phép toán không xác định (0/0, inf-inf, sqrt(-1)). Lây lan qua mọi phép toán tiếp theo. |
| Inf | "Vô cực" | Giá trị float đặc biệt sinh ra do tràn số trên hoặc chia cho 0. Có thể kết hợp tạo thành NaN (inf - inf, inf * 0). |
| Numerical gradient | "Đạo hàm tính trực diện" | Xấp xỉ đạo hàm bằng cách tính $f(x+h)$ và $f(x-h)$ rồi chia cho $2h$. Chậm nhưng rất đáng tin cậy để thẩm định. |

## Further Reading (Tài liệu đọc thêm)

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- Tài liệu kinh điển toàn diện về số học dấu phẩy động
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- Bài báo từ NVIDIA giới thiệu loss scaling cho huấn luyện float16
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- Hướng dẫn thực hành về mixed precision trong PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Lý do Google lựa chọn định dạng này cho TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- Thuật toán giảm thiểu sai số làm tròn khi tính tổng các số dấu phẩy động
