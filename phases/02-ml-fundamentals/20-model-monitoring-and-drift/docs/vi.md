# Giám sát Mô hình trong Môi trường Sản xuất — Data Drift và Concept Drift

> Một mô hình đã triển khai sẽ âm thầm thất bại mà không đưa ra bất kỳ thông báo lỗi nào. Nó vẫn đều đặn trả về điểm số dự đoán trong khi thế giới thực đã dịch chuyển xa khỏi dữ liệu huấn luyện, và chỉ có hệ thống giám sát mới vạch trần được khoảng cách đó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2, Lessons 09 (Model Evaluation), 16 (Anomaly Detection), and 19 (ML System Design)
**Time:** ~110 phút

## Learning Objectives

- Lựa chọn các tín hiệu giám sát ở 4 tầng: sức khỏe hệ thống, dữ liệu đầu vào, kết quả dự đoán và nhãn thực tế
- Phân biệt độ trôi dạt hiệp biến (covariate drift), độ trôi dạt tiên nghiệm (prior drift) và độ trôi dạt khái niệm (concept drift) dựa trên phần phân phối đồng thời bị biến đổi
- Tự tay xây dựng chỉ số ổn định quần thể (Population Stability Index - PSI), thống kê kiểm định Kolmogorov-Smirnov hai mẫu và kiểm định Chi-bình phương từ đầu
- So sánh cửa sổ tham chiếu cố định (reference window) với các cửa sổ trượt hiện tại (sliding current windows), giải thích cách kích thước cửa sổ tác động đến độ nhiễu và độ trễ phát hiện
- Thiết kế chính sách cảnh báo giúp hạn chế báo động giả thông qua độ lớn hiệu ứng (effect size), hiệu chỉnh kiểm định lặp (multiple-testing correction) và quy tắc duy trì bền bỉ (persistence rule)
- Quyết định loại trôi dạt nào đòi hỏi phải huấn luyện lại mô hình, loại nào cần thu thập nhãn mới và loại nào chưa cần can thiệp

## Vấn đề đặt ra (The Problem)

Một mô hình đánh giá rủi ro thanh toán được đưa lên production với độ chính xác ấn tượng 95%. Hai tháng sau, một chiến dịch marketing thu hút thêm nhiều người dùng ứng dụng di động với giỏ hàng lớn hơn nhiều. Bốn tháng sau, những kẻ gian lận thay đổi thủ đoạn, và các giao dịch từng được coi là an toàn bắt đầu bị ngân hàng gửi yêu cầu bồi hoàn (chargeback).

Trong suốt khoảng thời gian đó, dịch vụ mô hình vẫn kiên định trả về mã trạng thái HTTP 200 cho mọi yêu cầu. Độ trễ vẫn bình thường và nhật ký hệ thống (logs) không ghi nhận bất kỳ lỗi ngoại lệ nào. Dấu hiệu bất ổn đầu tiên mà công ty nhận được là một bản báo cáo tài chính về khoản lỗ do bồi hoàn gian lận, nhiều tuần sau khi thiệt hại thực tế đã xảy ra.

Giám sát phần mềm truyền thống (Software monitoring) chỉ kiểm tra xem dịch vụ có phản hồi hay không. Còn giám sát mô hình học máy (Model monitoring) kiểm tra xem **các câu trả lời có còn đúng đắn hay không**, và liệu dữ liệu hiện tại có còn mang hình thái giống như dữ liệu từng dùng để huấn luyện mô hình hay không. Bài học này xây dựng nền tảng thống kê, cơ chế cửa sổ thời gian và chính sách cảnh báo cho nhiệm vụ thứ hai đó. Bạn sẽ chạy chúng trên một luồng dữ liệu giả lập được cố tình cài cắm hai đợt biến đổi và quan sát xem bộ giám sát nào phản ứng với sự thay đổi nào.

## Khái niệm (The Concept)

### Cần giám sát những gì?

Giám sát 4 tầng riêng biệt. Ba tầng đầu tiên cung cấp các cảnh báo sớm tức thì. Chỉ có tầng thứ tư mới đo lường được tính đúng đắn thực tế, nhưng nó luôn đến muộn do độ trễ của nhãn.

| Tầng giám sát | Tín hiệu theo dõi | Tính khả dụng | Câu hỏi cần trả lời |
|---|---|---|---|
| Sức khỏe hệ thống | Độ trễ p50 và p99, tỷ lệ lỗi, thông lượng (QPS), bộ nhớ | Ngay lập tức | Dịch vụ có đang phản hồi bình thường không? |
| Dữ liệu đầu vào | Phân phối của từng đặc trưng, tỷ lệ giá trị thiếu, lược đồ dữ liệu, danh mục mới | Ngay lập tức | Dữ liệu thực tế có còn giống dữ liệu huấn luyện không? |
| Dự đoán đầu ra | Phân phối điểm số, tỷ lệ dương tính, tỷ lệ trả về giá trị mặc định | Ngay lập tức | Hành vi của mô hình có còn ổn định như trước không? |
| Kết quả thực tế | Độ chính xác, precision, độ hiệu chuẩn xác suất, thước đo kinh doanh | Sau độ trễ của nhãn | Mô hình đưa ra dự đoán có còn đúng đắn không? |

Giám sát đầu vào (Input monitoring) là trọng tâm chính của bài học này vì nó hoạt động được ngay lập tức trước khi nhãn thực tế xuất hiện. Tuy nhiên, nó có một giới hạn cố hữu: một số dạng biến đổi xảy ra trong thực tế mà không làm thay đổi phân phối của dữ liệu đầu vào.

### Ba loại trôi dạt (Three Kinds of Drift)

Phân phối xác suất đồng thời của đầu vào $x$ và nhãn $y$ có thể được phân rã theo hai cách:

```math
P(x, y) = P(x) \cdot P(y \mid x) = P(y) \cdot P(x \mid y)
```

Mỗi loại trôi dạt sẽ làm biến đổi một thành phần khác nhau:

| Loại trôi dạt | Thành phần thay đổi | Thành phần giữ nguyên | Ví dụ thanh toán | Bộ giám sát phát hiện |
|---|---|---|---|---|
| Trôi dạt hiệp biến (Covariate drift) | $P(x)$ | $P(y \mid x)$ | Nhiều người dùng ứng dụng với giỏ hàng lớn hơn | Giám sát dữ liệu đầu vào |
| Trôi dạt tiên nghiệm (Prior drift) | $P(y)$ | $P(x \mid y)$ | Tỷ lệ gian lận tăng gấp đôi trong một đợt tấn công, nhưng hành vi gian lận không đổi | Giám sát dự đoán và kết quả |
| Trôi dạt khái niệm (Concept drift) | $P(y \mid x)$ | Có thể giữ nguyên | Cùng một giỏ hàng đó nhưng nay lại mang rủi ro gian lận khác xưa | Chỉ có giám sát kết quả mới thấy |

Gama và các cộng sự gọi sự thay đổi trong $P(y \mid x)$ là **trôi dạt khái niệm thực sự** (real concept drift). Họ gọi sự thay đổi trong $P(x)$ mà giữ nguyên $P(y \mid x)$ là **trôi dạt ảo** (virtual drift). Nghiên cứu của họ cũng phân loại trôi dạt theo cách thức diễn biến theo thời gian: đột ngột (sudden), tịnh tiến (incremental), từ từ (gradual), hoặc chu kỳ quay trở lại (recurring).

Trôi dạt hiệp biến không phải lúc nào cũng gây hại. Nếu mô hình đã học tốt mối quan hệ $P(y \mid x)$, nó vẫn sẽ dự đoán đúng cho từng điểm đầu vào. Tuy nhiên, mô hình vẫn có thể suy giảm độ chính xác khi lưu lượng dịch chuyển vào những vùng không gian có ít dữ liệu huấn luyện hoặc sát ranh giới quyết định.

### Độ trễ của nhãn (Delayed Labels)

Kết quả thực tế luôn xuất hiện sau thời điểm dự đoán. Độ trễ phụ thuộc vào bản chất nghiệp vụ:

| Bài toán | Thời điểm nhãn xuất hiện |
|---|---|
| Lượt nhấp chuột quảng cáo | Vài giây |
| Người dùng rời bỏ trong 30 ngày | Sau 30 ngày |
| Gian lận thẻ tín dụng | Khi có yêu cầu bồi hoàn, thường là vài tuần sau |
| Vỡ nợ khoản vay | Vài tháng hoặc vài năm |

Trong khoảng thời gian trễ này, bạn chỉ có dữ liệu đầu vào và các dự đoán, hoàn toàn không có nhãn thực tế. Hãy giám sát đầu vào và dự đoán liên tục, sau đó ghép nối (join) nhãn khi chúng xuất hiện để tính toán các thước đo kết quả trên phần dữ liệu đã có nhãn.

### Cửa sổ tham chiếu và cửa sổ hiện tại (Reference & Current Windows)

Mọi phép kiểm định độ trôi dạt đều so sánh giữa hai tập mẫu:

- **Cửa sổ tham chiếu (Reference window):** Dữ liệu huấn luyện, hoặc một giai đoạn sản xuất đã được kiểm tra kỹ lưỡng thủ công. Nó giữ cố định cho tới khi bạn huấn luyện lại mô hình.
- **Cửa sổ hiện tại (Current window):** Các sự kiện gần đây nhất, có thể là $N$ sự kiện gần nhất hoặc trong $T$ giờ vừa qua.

Cửa sổ trượt (sliding window) dịch chuyển về phía trước với bước nhảy nhỏ hơn kích thước cửa sổ, do đó các cửa sổ liên tiếp sẽ gối lên nhau. Cửa sổ lật (tumbling window) dịch chuyển bằng đúng kích thước của nó. Kích thước cửa sổ là một sự đánh đổi: cửa sổ nhỏ phản ứng rất nhanh nhưng dễ coi nhiễu ngẫu nhiên là trôi dạt; cửa sổ lớn hoạt động ổn định nhưng phát hiện biến đổi chậm chạp.

### Chỉ số ổn định quần thể (Population Stability Index - PSI)

PSI bắt nguồn từ ngành chấm điểm tín dụng tài chính. Chia các giá trị tham chiếu thành các thùng (thường là 10 thùng tại các phân vị thập phân - deciles, sao cho mỗi thùng chứa đúng 10% dữ liệu tham chiếu). Đếm tỷ lệ các giá trị hiện tại rơi vào từng thùng. Sau đó tính tổng cho từng thùng:

```math
\text{PSI} = \sum_{\text{bins}} (\text{current\_share} - \text{reference\_share}) \cdot \ln\left(\frac{\text{current\_share}}{\text{reference\_share}}\right)
```

Một thùng có tỷ lệ bằng 0 sẽ khiến phép tính logarit bị vô hạn, vì vậy ta kẹp (clip) tỷ lệ ở một giá trị dương nhỏ như 0.0001. PSI có tính chất đối xứng, tương đương với phân kỳ KL từ hiện tại sang tham chiếu cộng với phân kỳ KL từ tham chiếu sang hiện tại.

Các kỹ sư thường áp dụng quy ước: dưới 0.1 là ổn định, 0.1 đến 0.25 là có sự dịch chuyển vừa phải, và trên 0.25 là dịch chuyển đáng kể.

### Thống kê Kolmogorov-Smirnov (KS Statistic)

Thống kê KS hai mẫu $D$ là khoảng cách đứng lớn nhất giữa hai hàm phân phối tích lũy thực nghiệm (ECDF):

```math
D = \max_x \left\lvert F_{\text{reference}}(x) - F_{\text{current}}(x) \right\rvert
```

$D = 0$ khi hai mẫu đồng nhất hoàn toàn và $D = 1$ khi hai mẫu không hề chồng lấn nhau. Thống kê này không đòi hỏi phải chia thùng. Với kích thước mẫu lớn, giá trị p-value sẽ tiến về 0 ngay cả với sự chênh lệch nhỏ không đáng kể. Vì vậy, hãy dùng chính giá trị $D$ làm thước đo độ lớn hiệu ứng (effect size), và chỉ kích hoạt cảnh báo khi cả $D$ lẫn p-value đều vượt ngưỡng quy định.

### Kiểm định Chi-bình phương cho đặc trưng phân loại (Chi-square Test)

Với một đặc trưng phân loại (như kênh thanh toán), lập bảng ngẫu nhiên gồm 2 hàng (tham chiếu, hiện tại) và mỗi cột là một danh mục. Số lượng kỳ vọng trong mỗi ô được tính từ tổng hàng và tổng cột:

```math
\begin{aligned}
\text{expected}[r][c] &= \text{row\_total}[r] \cdot \text{column\_total}[c] / \text{grand\_total} \\
\text{chi2} &= \sum_{\text{cells}} (\text{observed} - \text{expected})^2 / \text{expected}
\end{aligned}
```

Kiểm định có $k - 1$ bậc tự do cho $k$ danh mục. Chỉ số Cramér's V đóng vai trò là thước đo độ lớn hiệu ứng. Với bảng có 2 hàng, nó bằng căn bậc hai của `chi2 / grand_total`.

### Ngưỡng cảnh báo và xử lý báo động giả

Mỗi bài kiểm định đều có một tỷ lệ báo động giả $\alpha$ khi thế giới thực không hề thay đổi. Trong thực tế, hệ thống chạy hàng chục bài kiểm định mỗi giờ. Nếu kiểm tra 50 đặc trưng mỗi giờ với $\alpha = 0.05$, bạn sẽ nhận khoảng 60 báo động giả mỗi ngày. Đội ngũ kỹ sư sẽ sớm kiệt sức và phớt lờ toàn bộ cảnh báo (alert fatigue).

Bốn quy tắc vàng giúp duy trì giá trị của cảnh báo:

1. **Hiệu chỉnh kiểm định lặp (Multiple testing correction):** Hiệu chỉnh Bonferroni chia $\alpha$ cho tổng số bài kiểm tra.
2. **Yêu cầu độ lớn hiệu ứng (Effect size):** Chỉ cảnh báo khi PSI, $D$, hoặc Cramér's V cũng đồng thời vượt qua ngưỡng độ lớn hiệu ứng tối thiểu.
3. **Yêu cầu tính duy trì bền bỉ (Persistence):** Chỉ cảnh báo khi độ trôi dạt xuất hiện liên tục trong $k$ cửa sổ liên tiếp (ví dụ: $k = 2$).
4. **Tách biệt cảnh báo (Alert) với nhắc nhở (Warning):** Nhắc nhở chỉ hiển thị trên dashboard nội bộ, còn cảnh báo mới gửi tin nhắn trực tiếp cho con người.

### Cơ chế kích hoạt huấn luyện lại (Retraining Triggers)

Có ba cơ chế kích hoạt huấn luyện lại phổ biến:

- **Theo lịch định kỳ:** Huấn luyện lại mỗi tuần hoặc mỗi tháng. Đơn giản, nhưng gây lãng phí khi dữ liệu không đổi và lại quá chậm chạp khi có biến động bất ngờ.
- **Theo độ trôi dạt (Drift-based):** Huấn luyện lại khi độ trôi dạt đầu vào trên các đặc trưng quan trọng duy trì trên ngưỡng cảnh báo.
- **Theo hiệu năng (Performance-based):** Huấn luyện lại khi một thước đo kết quả thực tế giảm xuống dưới ngưỡng chấp nhận được.

Bản chất của loại trôi dạt sẽ quyết định hành động tương ứng:

| Loại trôi dạt | Hành động phù hợp |
|---|---|
| Trôi dạt hiệp biến, độ chính xác ổn định | Thường không cần làm gì, tiếp tục theo dõi thước đo kết quả |
| Trôi dạt hiệp biến vào vùng dữ liệu thưa | Huấn luyện lại trên dữ liệu gần đây có bao gồm các vùng dữ liệu mới đó |
| Trôi dạt tiên nghiệm (Prior drift) | Hiệu chuẩn lại xác suất hoặc dịch chuyển ngưỡng quyết định |
| Trôi dạt khái niệm (Concept drift) | Bắt buộc phải thu thập nhãn mới trước. Huấn luyện lại trên nhãn cũ chỉ học lại quy luật đã lỗi thời |

Sau khi huấn luyện lại, kiểm thử mô hình mới trên dữ liệu có nhãn gần nhất, triển khai qua các bước shadow và canary, sau đó đặt lại cửa sổ tham chiếu theo dữ liệu huấn luyện mới.

```figure
mlprod-drift-psi
```

## Tự tay xây dựng (Build It)

Mã nguồn trong `code/main.py` chỉ sử dụng thư viện chuẩn của Python.

### Bước 1: Chia thùng phân vị và tính PSI

```python
def quantile_edges(reference: list[float], bins: int = 10) -> list[float]:
    ordered = sorted(reference)
    n = len(ordered)
    edges = [ordered[min(n - 1, (n * i) // bins)] for i in range(1, bins)]
    return sorted(set(edges))


def psi_from_fractions(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total
```

### Bước 2: Thống kê KS và tính p-value

```python
def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    a, b = sorted(sample_a), sorted(sample_b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d
```

### Bước 3: Kiểm định Chi-bình phương

```python
def chi_square_test(reference_counts: dict[str, int], current_counts: dict[str, int]) -> ChiSquareResult:
    categories = [c for c in sorted(set(reference_counts) | set(current_counts))
                  if reference_counts.get(c, 0) + current_counts.get(c, 0) > 0]
    rows = [[reference_counts.get(c, 0) for c in categories], [current_counts.get(c, 0) for c in categories]]
    row_totals = [sum(r) for r in rows]
    grand = sum(row_totals)
    if len(categories) < 2 or 0 in row_totals:
        return ChiSquareResult(0.0, 0, 1.0, 0.0)
    statistic = 0.0
    for col, _ in enumerate(categories):
        col_total = rows[0][col] + rows[1][col]
        for r in range(2):
            expected = row_totals[r] * col_total / grand
            statistic += (rows[r][col] - expected) ** 2 / expected
    dof = len(categories) - 1
    return ChiSquareResult(statistic, dof, chi2_sf(statistic, dof), math.sqrt(statistic / grand))
```

### Bước 4: Cửa sổ trượt và bộ giám sát trôi dạt

`SlidingWindow` lưu giữ `size` sự kiện gần nhất trong một hàng đợi `deque`. `DriftMonitor` tính toán các thùng tham chiếu một lần và cung cấp phương thức `check` trả về PSI, $D$, giá trị KS p-value và kết quả Chi-square.

### Bước 5: Chính sách cảnh báo (Alert Policy)

```python
def level(self, metrics: dict[str, float]) -> str:
    if "psi" in metrics:
        if metrics["psi"] >= self.psi_drift or (metrics["ks"] >= self.ks_drift and metrics["ks_p"] < self.alpha):
            return "drift"
        return "warn" if metrics["psi"] >= self.psi_warn else "ok"
    if metrics["p"] < self.alpha and metrics["v"] >= self.min_cramers_v:
        return "drift"
    return "warn" if metrics["p"] < self.alpha else "ok"
```

## Sử dụng trong thực tế (Use It)

Trong hệ sinh thái thực tế:
- **scipy.stats:** Cung cấp sẵn `ks_2samp` và `chi2_contingency`.
- **Thư viện chuyên dụng:**
  - **Evidently AI:** Tự động chọn bài kiểm tra thích hợp theo từng cột và cung cấp báo cáo trực quan.
  - **NannyML:** Ước lượng hiệu năng mô hình ngay cả trước khi nhãn thực tế xuất hiện (Confidence-based Performance Estimation - CBPE).
  - **TensorFlow Data Validation (TFDV):** So sánh các phân đoạn dữ liệu, kiểm tra độ lệch giữa tập huấn luyện và phục vụ.

## Xuất xưởng (Ship It)

Bài học này tạo ra:
- `outputs/skill-drift-monitoring.md` -- kỹ năng lập kế hoạch giám sát cho mô hình mới và phân loại xử lý khi có cảnh báo trôi dạt.

## Bài tập thực hành (Exercises)

1. Chạy luồng dữ liệu với kích thước cửa sổ là 200 và 2000 sự kiện. Ghi lại các cảnh báo giả trước sự kiện 3000 và độ trễ phát hiện cảnh báo.
2. Đặt `n_tests=1`, `patience=1`, và `min_cramers_v=0`. Đếm số lượng cảnh báo trong giai đoạn sạch từ 5000 đến 6500.
3. Thêm một đặc trưng có giá trị trung bình tăng dần 0.01 sau mỗi 100 sự kiện. Tìm xem tín hiệu nào, PSI hay KS, sẽ phản ứng trước tiên.
4. Triển khai khoảng cách Jensen-Shannon trên cùng các thùng phân vị thập phân. So sánh nó với PSI trên từng cửa sổ.
5. Bổ sung một bộ giám sát theo dõi tỷ lệ dự đoán dương tính của mô hình. Giải thích vì sao bộ giám sát này lại im lặng trong suốt đợt trôi dạt khái niệm (concept drift).

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Data drift | "Dữ liệu bị thay đổi" | Sự thay đổi trong phân phối của đầu vào, dự đoán hoặc nhãn so với cửa sổ tham chiếu |
| Covariate drift | "Người dùng kiểu mới xuất hiện" | $P(x)$ thay đổi trong khi $P(y \mid x)$ vẫn giữ nguyên |
| Prior drift | "Nhiều ca dương tính hơn trước" | $P(y)$ thay đổi trong khi $P(x \mid y)$ không đổi, còn gọi là dịch chuyển nhãn (label shift) |
| Concept drift | "Quy luật cuộc chơi đã đổi" | $P(y \mid x)$ thay đổi, khiến cùng một đầu vào nay lại có đáp án thực tế khác đi |
| Reference window | "Dữ liệu chuẩn mốc" | Tập mẫu cố định, thường là tập huấn luyện, dùng làm thước đo chuẩn cho mọi cửa sổ hiện tại |
| PSI | "Điểm số ổn định" | Tổng đối xứng trên các thùng của (tỷ lệ hiện tại trừ tỷ lệ tham chiếu) nhân logarit tỷ số của chúng |
| KS statistic | "Kiểm định KS" | Khoảng cách chênh lệch lớn nhất giữa hai hàm phân phối tích lũy thực nghiệm (ECDF) |
| Cramér's V | "Độ lớn hiệu ứng cho biến phân loại" | Thống kê Chi-bình phương được chuẩn hóa về đoạn từ 0 đến 1 |
| Label delay | "Nhãn thực tế đến sau" | Khoảng thời gian trễ giữa lúc đưa ra dự đoán và lúc nhận được kết quả thực tế |
| Alert fatigue | "Bị bội thực cảnh báo" | Hiện tượng kỹ sư phớt lờ cảnh báo vì hầu hết chúng là báo động giả |

## Tài liệu đọc thêm (Further Reading)

- [Gama et al., A Survey on Concept Drift Adaptation (ACM Computing Surveys, 2014)](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation) -- khảo cứu toàn diện về trôi dạt thực sự và trôi dạt ảo, các mẫu hình biến đổi và phương pháp thích ứng
- [Rabanser, Günnemann, and Lipton, Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (NeurIPS 2019)](https://arxiv.org/abs/1810.11953) -- đánh giá thực nghiệm các phép kiểm định hai mẫu để phát hiện dịch chuyển dữ liệu
- [Lu et al., Learning under Concept Drift: A Review (2020)](https://arxiv.org/abs/2004.05785) -- các phương pháp phát hiện, phân tích và thích ứng với trôi dạt khái niệm
- [Yurdakul, Statistical Properties of Population Stability Index (Western Michigan University, 2018)](https://scholarworks.wmich.edu/dissertations/3208/) -- hành vi toán học của chỉ số PSI theo kích thước mẫu và số lượng thùng
- [Evidently, How data drift detection works](https://docs.evidentlyai.com/metrics/explainer_drift) -- quy tắc lựa chọn kiểm định mặc định theo kiểu cột và kích thước dữ liệu
- [NannyML, Performance estimation](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html) -- thuật toán CBPE và các giả định ước lượng hiệu năng khi chưa có nhãn
