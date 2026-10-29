# Giải mã suy đoán và EAGLE-3 (Speculative Decoding and EAGLE-3)

> Giai đoạn 7 · Bài 16 đã chứng minh nền tảng toán học: quy tắc loại trừ Leviathan bảo toàn chính xác tuyệt đối phân phối xác suất của mô hình kiểm định (verifier). Bài học này mang đến góc nhìn từ ngăn xếp huấn luyện (training-stack) của giải mã suy đoán (speculative decoding) trong môi trường sản xuất năm 2026. EAGLE-3 đã biến mô hình phác thảo (draft model) từ một phép xấp xỉ giá rẻ thành một mạng nơ-ron siêu nhỏ được xây dựng có mục đích riêng, huấn luyện trực tiếp trên các trạng thái ẩn (hidden states) của chính mô hình kiểm định, sau đó tích hợp vòng lặp kiểm thử trong thời gian huấn luyện (training-time test) giúp căn chỉnh đồng nhất giữa phân phối huấn luyện và suy luận. Kết quả: tăng tốc toàn diện từ $3\times$ đến $6,5\times$ từ đầu đến cuối, tỷ lệ chấp nhận trên mỗi token vượt mức 0,9 trong tác vụ hội thoại mà không làm biến dạng phân phối đầu ra. Mọi ngăn xếp suy luận sản xuất năm 2026 đều tích hợp cơ chế này theo mặc định.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7 · 16 (Toán học giải mã suy đoán), Giai đoạn 10 · 12 (Tối ưu hóa suy luận)
**Time:** ~75 phút

## Mục tiêu học tập

- Phát biểu định lý Leviathan trong một câu và chứng minh rằng vòng lặp suy đoán tạo ra các mẫu có phân phối đồng nhất với mô hình kiểm định.
- Điểm lại tiến trình phát triển trong hai năm từ giải mã suy đoán nguyên bản (Leviathan 2023) qua EAGLE, EAGLE-2 và EAGLE-3, đồng thời chỉ ra chính xác hạn chế mà mỗi bước đã loại bỏ.
- Tính toán mức tăng tốc kỳ vọng dựa trên tỷ lệ chấp nhận $\alpha$ và tỷ số chi phí giữa mô hình phác thảo và kiểm định $c$, từ đó chọn độ dài phác thảo $N$ tối ưu cho từng kịch bản.
- Tự tay cài đặt toàn bộ vòng lặp suy đoán từ đầu: phác thảo, kiểm định, lấy mẫu loại trừ từ phân phối phần dư (residual distribution), khôi phục bộ nhớ đệm KV (KV cache rollback) khi bị từ chối và phát token thưởng thêm (bonus token) khi được chấp nhận toàn bộ.

## Vấn đề thực tế

Quá trình giải mã tự hồi quy (autoregressive decoding) trên một mô hình 70B chỉ đạt tốc độ khoảng 35 token/giây trên một GPU H100. GPU còn lâu mới đạt mức bão hòa tính toán. Băng thông bộ nhớ chính là trần giới hạn: với mỗi token tạo ra, hệ thống phải nạp 70B tham số trọng số từ bộ nhớ băng thông cao (HBM), thực hiện một bước số học nhỏ và trả về một giá trị số thực. Các đơn vị tính toán phần lớn thời gian đều rơi vào trạng thái nhàn rỗi.

Giải mã suy đoán biến bài toán thắt nút cổ chai bộ nhớ thành một bài toán thông lượng hoàn toàn có thể giải quyết được. Một mô hình phác thảo chi phí rẻ sẽ đề xuất $N$ token thông qua $N$ lượt truyền xuôi nhỏ. Mô hình kiểm định chỉ cần chạy đúng một lượt truyền xuôi trên toàn bộ tiền tố cộng với cả $N$ token phác thảo đó. Nếu phân phối của mô hình kiểm định tại vị trí $i$ đồng thuận với phác thảo (theo nghĩa thống kê toán học chính xác), chúng ta chấp nhận token; nếu không, chúng ta loại trừ và lấy mẫu hiệu chỉnh từ phân phối phần dư. Một lượt chạy duy nhất của mô hình lớn có thể tạo ra tối đa $N+1$ token được chấp nhận thay vì chỉ một token.

Định lý mang tính nền tảng của Leviathan, Kalman, Matias (ICML 2023) khẳng định: phân phối đầu ra hoàn toàn đồng nhất với việc lấy mẫu trực tiếp từ mô hình kiểm định lớn. Không phải là xấp xỉ. Mà là đồng nhất tuyệt đối. Đây chính là lý do duy nhất khiến giải mã suy đoán được chấp nhận hoàn toàn trong môi trường sản xuất -- đây là sự tối ưu hóa độ trễ thuần túy mà không có bất kỳ sự đánh đổi nào về chất lượng mô hình.

Những gì Giai đoạn 7 · Bài 16 mang lại cho bạn là nền tảng toán học. Bài học này sẽ cung cấp cho bạn ngăn xếp huấn luyện. Một mô hình phác thảo tốt mang lại tốc độ gấp 2 lần so với một bản phác thảo rẻ tiền. EAGLE, EAGLE-2 và EAGLE-3 (Li et al., 2024–2025) đã biến ý niệm "mô hình phác thảo = phiên bản thu nhỏ của cùng mô hình" thành một bộ môn kỹ thuật chuẩn xác. Các máy chủ suy luận sản xuất năm 2026 hiện đều mặc định sử dụng EAGLE-3.

## Khái niệm cốt lõi

### Phần bất biến: Lấy mẫu loại trừ Leviathan (Leviathan Rejection Sampling)

Gọi $p(t)$ là phân phối của mô hình phác thảo cho token kế tiếp với một tiền tố cho trước, và $q(t)$ là phân phối của mô hình kiểm định. Lấy mẫu một token phác thảo $d \sim p$. Chấp nhận token đó với xác suất $\min(1, q(d) / p(d))$. Nếu bị loại trừ, lấy mẫu một token thay thế từ phân phối phần dư $(q - p)_+ / \|(q - p)_+\|_1$. Các mẫu thu được luôn tuân theo chính xác phân phối $q$. Điều này luôn đúng bất kể $p$ kém đến mức nào -- $p$ càng kém thì tỷ lệ từ chối càng cao, nhưng đầu ra cuối cùng vẫn tuyệt đối chuẩn xác.

Xếp chồng $N$ lời gọi này liên tiếp nhau bằng một lượt truyền xuôi kiểm định duy nhất trên chuỗi `prefix + d_1 + ... + d_N`. Mô hình kiểm định sẽ trả về đồng thời $q_1, q_2, \dots, q_{N+1}$. Duyệt từ trái qua phải: tại vị trí từ chối đầu tiên $j$, lấy mẫu từ `residual(q_j, p_j)` và dừng lại. Nếu toàn bộ $N$ token đều được chấp nhận, lấy mẫu một token thưởng thêm từ $q_{N+1}$.

### Yếu tố nào quyết định tốc độ tăng tốc?

Gọi $\alpha$ là tỷ lệ chấp nhận kỳ vọng trên mỗi token phác thảo. Gọi $c = \text{cost}(\text{draft}) / \text{cost}(\text{verifier})$ là tỷ số chi phí. Số lượng token được chấp nhận kỳ vọng trên mỗi lượt truyền xuôi của mô hình kiểm định là:

```math
\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{N+1}}{1 - \alpha}
```

Tổng thời gian thực tế kỳ vọng trên mỗi token được chấp nhận là $(N \cdot c + 1) / \mathbb{E}[\text{accepted}]$. Cực tiểu hóa giá trị này theo $N$ sẽ cho ta điểm tối ưu. Với $\alpha = 0,8, c = 0,05$: $N$ tối ưu rơi vào khoảng 5–7, mang lại tốc độ gấp $3,2\times$. Với $\alpha = 0,95, c = 0,02$: $N$ tối ưu khoảng 8–10, đẩy mức tăng tốc lên tới $5\times$.

Đòn bẩy quan trọng nhất chính là $\alpha$. Việc tăng từ $\alpha = 0,6$ (mô hình phác thảo nguyên bản) lên $\alpha = 0,9$ (EAGLE-3) với độ dài cố định $N = 5$ sẽ đưa số token được chấp nhận kỳ vọng từ 2,2 lên 4,1 trên mỗi lượt kiểm định. Tăng gần gấp 2 lần thông lượng từ chính cùng một mô hình kiểm định đó.

### Tiến trình phát triển qua 2 năm

**Giải mã suy đoán nguyên bản (Vanilla speculative - Leviathan, 2023).** Mô hình phác thảo là một LLM nhỏ hơn được huấn luyện độc lập từ cùng một họ mô hình. Dễ tích hợp, $\alpha \approx 0,6$, mức tăng tốc tối đa chỉ khoảng $2\times$.

**EAGLE-1 (Li et al., 2024).** Mô hình phác thảo là một mạng transformer siêu nhỏ -- thường chỉ 1 hoặc 2 lớp -- nhận trạng thái ẩn ở lớp cuối cùng của mô hình kiểm định làm đầu vào và dự đoán trực tiếp token tiếp theo. Do mô hình phác thảo nhìn thấy không gian đặc trưng biểu diễn của mô hình kiểm định, phân phối của nó tiệm cận rất gần với phân phối kiểm định. Chỉ số $\alpha$ tăng vọt lên 0,7–0,8.

**EAGLE-2 (Li et al., 2024).** Bổ sung cây phác thảo động (dynamic draft tree): thay vì đề xuất một chuỗi tuyến tính đơn lẻ gồm $N$ token, hệ thống đề xuất một cây nhỏ các nhánh ứng viên, chấm điểm tất cả chúng bằng mô hình kiểm định trong một lượt truyền xuôi (tree attention), và chọn nhánh có xác suất cao nhất. Chiều dài phác thảo trở nên thích ứng theo từng bước. Chỉ số $\alpha$ trên nhánh được chấp nhận vượt qua mức 0,85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Mang lại hai bước đột phá. Thứ nhất, loại bỏ hoàn toàn hàm mất mát dự đoán đặc trưng (feature-prediction loss) -- vốn khiến việc huấn luyện mô hình phác thảo bị nghẽn trần do cố khớp trạng thái ẩn. Thay vào đó, EAGLE-3 huấn luyện trực tiếp trên bài toán dự đoán token. Thứ hai, áp dụng kiểm thử trong thời gian huấn luyện (training-time test - TTT): trong quá trình huấn luyện, đưa chính các dự đoán trước đó của mô hình phác thảo làm đầu vào ngược trở lại qua nhiều bước, giống hệt cách nó vận hành trong thời gian suy luận. Cơ chế này giúp căn chỉnh phân phối giữa tập huấn luyện và thực tế suy luận, triệt tiêu sự tích lũy sai số. Tốc độ đo đạc thực tế: tăng tốc tới $6,5\times$ trên tác vụ hội thoại, cải thiện 38% thông lượng ở batch size 64 trong SGLang trên phần cứng H100.

### Khôi phục bộ nhớ đệm KV (KV Cache Rollback)

Quá trình kiểm định sẽ mở rộng bộ nhớ đệm KV của mô hình kiểm định thêm $N$ phần tử trong một lượt. Nếu sự từ chối xảy ra ở vị trí $j$, nội dung bộ nhớ đệm sau vị trí $j-1$ sẽ trở nên không hợp lệ. Hai cách triển khai phổ biến: ghi vào bộ đệm nháp (scratch buffer) và chỉ ghi đè chính thức khi được chấp nhận (vLLM, TensorRT-LLM), hoặc duy trì một bộ nhớ đệm vật lý kèm theo độ dài logic và cắt ngắn độ dài khi bị từ chối. Dù theo cách nào, chi phí khôi phục chỉ là việc cập nhật con trỏ theo từng lớp từng đầu, hoàn toàn không đáng kể so với chi phí của lượt truyền xuôi.

Đối với tìm kiếm theo cây trong EAGLE-2, mô hình kiểm định thực thi cơ chế chú ý với mặt nạ phi nhân quả (non-causal mask) phù hợp với cấu trúc cây. Quá trình lập trình đòi hỏi sự tỉ mỉ nhưng bản chất tính toán vẫn là một lời gọi flash-attention tiêu chuẩn với mặt nạ tùy chỉnh.

### So sánh các kiến trúc phác thảo năm 2026

| Chiến lược | Loại phác thảo | $\alpha$ | Tăng tốc | Chi phí huấn luyện |
|---|---|---|---|---|
| Nguyên bản (Vanilla) | LLM nhỏ độc lập | 0,55-0,70 | 1,8-2,3$\times$ | Không (tái sử dụng mô hình nhỏ có sẵn) |
| Medusa | Thêm các đầu LM head trên mô hình kiểm định | 0,65-0,75 | 2-3$\times$ | ~1B token SFT |
| EAGLE-1 | Transformer 1 lớp trên trạng thái ẩn | 0,70-0,80 | 2,5-3$\times$ | ~60B token |
| EAGLE-2 | EAGLE-1 + cây phác thảo động | 0,80-0,88 | 3-4$\times$ | ~60B token |
| EAGLE-3 | Kết hợp đặc trưng đa tầng + TTT | 0,88-0,92 | 3,5-6,5$\times$ | ~60-200B token |
| Lookahead | Không dùng phác thảo (lặp Jacobi) | Không áp dụng | 1,3-1,6$\times$ | Không |

Trong môi trường sản xuất năm 2026: vLLM và SGLang mặc định sử dụng EAGLE-3 nếu có sẵn, hoặc EAGLE-2 nếu ngược lại. TensorRT-LLM có đường dẫn Medusa nhanh nhất cho các mô hình công khai của Meta và NVIDIA. llama.cpp cung cấp giải pháp phác thảo nguyên bản cho triển khai trên CPU.

````figure
l5-spec-decode-eagle
````

## Xây dựng thực tế (Build It)

Xem `code/main.py`. Đây là vòng lặp giải mã suy đoán Leviathan hoàn chỉnh với tất cả các thành phần: phác thảo $N$ token, lượt kiểm định song song, quy tắc loại trừ theo từng vị trí, lấy mẫu phần dư, token thưởng thêm, khôi phục bộ nhớ đệm KV và kiểm chứng thực nghiệm chứng minh phân phối đầu ra khớp chính xác với việc lấy mẫu trực tiếp từ $q$.

### Bước 1: Quy tắc loại trừ

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Bước 2: Phân phối phần dư

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Bước 3: Một bước suy đoán hoàn chỉnh

Hàm `spec_step` phác thảo $N$ token từ phân phối $p$, sau đó kiểm định toàn bộ chúng trong một lượt đánh giá song song trên $q$. Với mỗi token phác thảo, nó áp dụng quy tắc loại trừ; tại điểm từ chối đầu tiên, nó lấy mẫu hiệu chỉnh từ phân phối phần dư. Nếu tất cả token đều được chấp nhận, nó phát ra một token thưởng thêm từ $q_{N+1}$.

### Bước 4: Quản lý khôi phục bộ nhớ đệm KV

Bộ mô phỏng theo dõi biến logic `kv_length` cho từng luồng xử lý. Khi chấp nhận $k$ token phác thảo, `kv_length += k`. Khi từ chối tại vị trí $j$, mặc dù bộ đệm đã được ghi vượt quá $j$, biến độ dài logic được đặt lại về `prefix_length + j + 1` -- ngay sau vị trí của token hiệu chỉnh. Các thao tác đọc tiếp theo sẽ tự động cắt ngắn về độ dài logic này.

### Bước 5: Kiểm chứng định lý Leviathan

Chạy 50.000 bước suy đoán thực nghiệm. Thống kê phân phối thực tế của các token được chấp nhận. So sánh với 50.000 mẫu lấy trực tiếp từ phân phối $q$. Giá trị thống kê chi-bình phương ($\chi^2$) phải nằm dưới giá trị giới hạn tới hạn một khoảng an toàn. Định lý được kiểm chứng thành công trong thực tế.

### Bước 6: Mối tương quan giữa tốc độ tăng tốc và $\alpha$

Quét chất lượng của mô hình phác thảo bằng cách làm lệch $p$ ra xa khỏi $q$ ở nhiều biên độ khác nhau. Đo lường tỷ lệ $\alpha$, sau đó vẽ biểu đồ số token kỳ vọng trên mỗi lượt gọi kiểm định theo $\alpha$ và $N$. Mã nguồn sẽ in ra bảng số liệu chứng minh chất lượng phác thảo ở cấp độ EAGLE-3 ($\alpha \approx 0,9$) mở ra khả năng sinh 4–5 token cho mỗi lượt gọi mô hình kiểm định lớn.

## Sử dụng thực tế (Use It)

Khởi chạy máy chủ sản xuất `vllm serve` kết hợp EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang kết hợp với EAGLE-3 ở batch size 64 trên H100: thông lượng cao hơn khoảng $1,38\times$ so với giải mã thông thường ở cùng batch size 64, theo kết quả thực nghiệm từ bài báo EAGLE-3.

Khi nào nên áp dụng giải mã suy đoán:

- Mọi tác vụ hội thoại tương tác (chat) nơi độ trễ phân vị p50 quan trọng hơn thông lượng đỉnh.
- Tác vụ sinh mã lập trình và đầu ra có cấu trúc (JSON, SQL). Tỷ lệ $\alpha$ thường vượt trên 0,9 do phân phối đích có tính quy luật dự đoán rất cao.
- Tác vụ sinh văn bản dài (hàng nghìn token). Lợi ích tăng tốc được tích lũy liên tục và bù đắp hoàn toàn chi phí khởi tạo.

Khi nào không nên sử dụng:

- Các mô hình quá nhỏ (< 3B). Mô hình phác thảo không rẻ hơn mô hình kiểm định là bao.
- Triển khai trên CPU với batch size 1 cực nhỏ. Chi phí bộ nhớ để nạp thêm mô hình phác thảo có thể không đáng để đánh đổi.
- Quá trình lấy mẫu sáng tạo với nhiệt độ (temperature) rất cao, nơi $\alpha$ sụt giảm mạnh.

## Xuất bản tạo tác (Ship It)

Bài học này tạo ra `outputs/skill-eagle3-tuner.md`. Dựa trên khối lượng công việc suy luận (mô hình, kích thước batch, độ trễ mục tiêu, hồ sơ tác vụ), nó sẽ đề xuất chiến lược giải mã suy đoán và các tham số tinh chỉnh phù hợp (họ mô hình phác thảo, độ dài $N$, độ sâu cây, cơ chế chuyển mạch nhận biết nhiệt độ).

## Bài tập thực hành

1. Chạy `code/main.py`. Xác nhận rằng giá trị thống kê chi-bình phương trong bài kiểm tra phân phối Leviathan luôn duy trì dưới ngưỡng giới hạn tới hạn ở mức ý nghĩa 95% trên 50.000 mẫu.

2. Quét $N$ từ 1 đến 10 với $\alpha$ giữ cố định ở mức 0,9 và $c$ giữ ở mức 0,04. Vẽ đồ thị số token kỳ vọng trên mỗi lượt gọi kiểm định và thời gian thực tế trên mỗi token. Tìm giá trị $N$ giúp tối thiểu hóa thời gian. Giải thích hình thái của đường cong.

3. Sửa đổi mã nguồn để mô phỏng tìm kiếm theo cây của EAGLE-2: ở mỗi bước, mô hình phác thảo đề xuất một cây có hình thái `[2, 2, 2]` (8 đường dẫn ứng viên). Mô hình kiểm định chạy một lượt duy nhất và đường dẫn được chấp nhận có xác suất cao nhất sẽ chiến thắng. Tính toán $\alpha$ trên mỗi nút lá và tổng số token tạo ra trên mỗi lượt gọi kiểm định. So sánh với giải mã suy đoán chuỗi tuyến tính ở cùng mức chi phí tính toán.

4. Cài đặt bộ mô phỏng khôi phục bộ nhớ đệm KV theo batch cho hai chuỗi chạy đồng thời. Chuỗi A có toàn bộ phác thảo được chấp nhận; chuỗi B bị từ chối ở vị trí số 2. Chứng minh rằng biến `kv_length` được cập nhật chính xác cho từng chuỗi và không có tài nguyên tính toán nào bị lãng phí.

5. Đọc Mục 4 của bài báo EAGLE-3 (Training-Time Test). Giải thích trong hai câu lý do tại sao việc huấn luyện mô hình phác thảo ngây thơ không có TTT lại chịu ảnh hưởng tiêu cực từ độ lệch phơi nhiễm (exposure bias), và tại sao việc nạp lại chính dự đoán của mô hình phác thảo trong quá trình huấn luyện lại giải quyết được vấn đề này. Liên hệ điều này với lý thuyết lấy mẫu theo lịch trình (scheduled sampling) trong kiến trúc seq2seq.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực tế |
|---|---|---|
| Quy tắc Leviathan | "min(1, q chia p)" | Quy tắc chấp nhận/loại trừ Bernoulli với xác suất $\min(1, q(d)/p(d))$, bảo toàn chính xác tuyệt đối phân phối của mô hình kiểm định khi lấy mẫu từ phân phối phần dư khi bị từ chối |
| Phân phối phần dư (Residual distribution) | "(q trừ p) lấy phần dương, chuẩn hóa" | Biểu thức $(q - p)_+$ được chặn dưới tại 0 và tái chuẩn hóa tổng bằng 1 — phân phối toán học chuẩn xác để lấy mẫu thay thế khi token bị từ chối |
| Tỷ lệ chấp nhận $\alpha$ (Acceptance rate) | "Mô hình phác thảo đúng bao nhiêu phần trăm" | Xác suất thành công Bernoulli kỳ vọng trên mỗi token dưới quy tắc loại trừ; tham số chi phối toàn bộ công thức tăng tốc |
| EAGLE-1 | "Phác thảo dựa trên trạng thái ẩn" | Mạng transformer siêu nhỏ phác thảo token dựa trên điều kiện là trạng thái ẩn lớp cuối của mô hình kiểm định (Li et al., 2024) |
| EAGLE-2 | "Cây phác thảo động" | EAGLE-1 kết hợp với một cây các chuỗi tiếp nối ứng viên, được chấm điểm bằng cơ chế chú ý theo cây trong một lượt kiểm định duy nhất |
| EAGLE-3 | "Kiểm thử trong thời gian huấn luyện" | Loại bỏ hàm mất mát dự đoán đặc trưng, huấn luyện trực tiếp trên bài toán dự đoán token và nạp lại chính đầu ra của mô hình phác thảo trong quá trình huấn luyện |
| Training-time test (TTT) | "Biện pháp khắc phục độ lệch phơi nhiễm" | Chạy mô hình phác thảo tự hồi quy trong quá trình huấn luyện để phân phối đầu vào lúc huấn luyện và kiểm thử trùng khớp nhau — cơ chế tương tự lấy mẫu theo lịch trình |
| Khôi phục KV (KV rollback) | "Hủy bỏ các token phác thảo bị từ chối" | Thao tác quản lý con trỏ giúp thiết lập lại bộ nhớ đệm KV của mô hình kiểm định về đúng độ dài của tiền tố đã được chấp nhận sau khi xảy ra từ chối |
| Token thưởng thêm (Bonus token) | "Token miễn phí" | Khi tất cả $N$ token phác thảo đều được chấp nhận, hệ thống lấy mẫu thêm một token từ $q_{N+1}$ mà không tốn thêm bất kỳ chi phí kiểm định nào |
| Chú ý theo cây (Tree attention) | "Kiểm định nhiều ứng viên cùng lúc" | Cơ chế chú ý với mặt nạ phi nhân quả tôn trọng cấu trúc liên kết của cây phác thảo; tính toán $q_i$ cho mọi nút trong cây chỉ trong một lượt truyền xuôi |

## Đọc thêm

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — bài báo nền tảng và định lý tương đương phân phối
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — công trình độc lập cùng thời với phép chứng minh toán học tường minh
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1, phác thảo có điều kiện theo trạng thái ẩn
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — tìm kiếm trên cây phác thảo động
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — chuẩn mực sản xuất năm 2026
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — phương pháp thay thế không dùng mô hình phác thảo riêng biệt
- [Tài liệu giải mã suy đoán vLLM](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham khảo sản xuất chuẩn mực tích hợp toàn bộ các chiến lược suy đoán
