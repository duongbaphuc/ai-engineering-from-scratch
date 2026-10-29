# Giải mã suy đoán — Đề xuất nháp, Xác thực, Lặp lại (Speculative Decoding — Draft, Verify, Repeat)

> Quá trình giải mã tự hồi quy (autoregressive decoding) mang tính tuần tự nghiêm ngặt: mỗi token đều phải đợi token trước đó hoàn thành. Giải mã suy đoán (speculative decoding) phá vỡ xiềng xích này: một mô hình chi phí thấp sinh trước $N$ token nháp, mô hình lớn đắt đỏ sẽ xác thực đồng thời toàn bộ $N$ token đó chỉ trong một lượt lan truyền xuôi duy nhất. Khi các token nháp chính xác, bạn chỉ phải trả chi phí một lượt chạy của mô hình lớn cho $N$ token sinh ra.

**Phân loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 07 (Mô hình ngôn ngữ nhân quả GPT), Giai đoạn 7 · 12 (Bộ đệm KV & FlashAttention)  
**Thời lượng:** ~60 phút

## Vấn đề đặt ra

Một mô hình ngôn ngữ lớn (LLM) 70B lấy mẫu một token mất khoảng ~30 ms trên GPU H100. Một mô hình nháp (draft model) 3B chỉ mất ~3 ms. Nếu chúng ta để mô hình 3B sinh nháp trước 5 token, sau đó cho mô hình 70B chạy *đúng một lần* để xác thực đồng thời cả 5 token đó, tổng thời gian sẽ là $5 \times 3 + 30 = 45$ ms cho tối đa 5 token được chấp nhận — so với mức $5 \times 30 = 150$ ms nếu sinh tuần tự thông thường. Đó chính là toàn bộ luận điểm cốt lõi của giải mã suy đoán: đánh đổi một lượng nhỏ bộ nhớ GPU phụ (cho mô hình nháp) để đổi lấy độ trễ giải mã thấp hơn từ 2–4 lần.

Mẹo ở đây là phải bảo toàn được phân phối xác suất. Cơ chế lấy mẫu suy đoán (speculative sampling), được giới thiệu độc lập bởi Leviathan và các cộng sự (2023) cũng như Chen và các cộng sự trong cùng thời điểm, đảm bảo về mặt toán học rằng chuỗi đầu ra có **phân phối hoàn toàn đồng nhất** với chuỗi mà mô hình lớn tự sinh ra một mình. Tuyệt đối không làm suy giảm chất lượng, chỉ đơn thuần là tăng tốc độ.

Bốn họ cặp mô hình nháp - mô hình xác thực (draft-verifier) thống trị môi trường suy luận năm 2026:

1. **Giải mã suy đoán nguyên bản (Vanilla speculative - Leviathan 2023).** Tách biệt mô hình nháp (ví dụ: Llama 3 1B) và mô hình xác thực (ví dụ: Llama 3 70B).
2. **Medusa (Cai 2024).** Tích hợp nhiều đầu giải mã trực tiếp trên mô hình xác thực để dự đoán các vị trí từ `t+1..t+k` một cách song song. Không cần mô hình nháp riêng biệt.
3. **Họ mô hình EAGLE (Li 2024, 2025).** Mô hình nháp siêu nhẹ tái sử dụng các trạng thái ẩn của mô hình xác thực; tỷ lệ chấp nhận cao hơn nhiều so với phương pháp nguyên bản; mức tăng tốc điển hình đạt 3–4 lần.
4. **Lookahead decoding (Fu 2024).** Lặp Jacobi; hoàn toàn không cần mô hình nháp. Tự suy đoán (self-speculation). Tuy kén ngữ cảnh nhưng không tạo thêm bất kỳ phụ thuộc nào.

Mọi hệ thống suy luận trong môi trường sản xuất năm 2026 đều tích hợp sẵn giải mã suy đoán làm cấu hình mặc định. vLLM, TensorRT-LLM, SGLang và llama.cpp đều hỗ trợ tối thiểu cơ chế nguyên bản và EAGLE-2.

## Khái niệm cốt lõi

### Thuật toán cốt lõi

Cho trước mô hình xác thực $M_q$ và mô hình nháp chi phí thấp $M_p$:

1. Gọi $x_1 \ldots x_k$ là tiền tố đã được giải mã trước đó.
2. **Sinh nháp (Draft)**: sử dụng $M_p$ để đề xuất tự hồi quy các token $d_{k+1}, d_{k+2}, \ldots, d_{k+N}$ cùng xác suất dự đoán nháp $p_1 \ldots p_N$.
3. **Xác thực song song (Verify in parallel)**: chạy mô hình $M_q$ đúng một lần trên chuỗi $x_1 \ldots x_k, d_{k+1}, \ldots, d_{k+N}$, thu về xác suất xác thực $q_1 \ldots q_{N+1}$ cho các vị trí $k+1 \ldots k+N+1$.
4. **Chấp nhận/từ chối từng token nháp từ trái qua phải**: với mỗi vị trí $i$, chấp nhận với xác suất $\min(1, q_i(d_i) / p_i(d_i))$.
5. Khi gặp token đầu tiên bị từ chối tại vị trí $j$: lấy mẫu $t_j$ từ phân phối "thặng dư" (residual distribution) $(q_j - p_j)_+$ đã được chuẩn hóa. Toàn bộ các token nháp sau vị trí $j$ đều bị hủy bỏ.
6. Khi toàn bộ $N$ token đều được chấp nhận: lấy mẫu thêm một token bổ sung $t_{N+1}$ từ $q_{N+1}$ (token thưởng miễn phí).

Bí quyết phân phối thặng dư chính là phát kiến toán học then chốt giúp cho chuỗi đầu ra tuân theo chính xác phân phối như thể mô hình $M_q$ tự lấy mẫu từ đầu.

### Yếu tố nào quyết định mức tăng tốc?

Gọi $\alpha$ là tỷ lệ chấp nhận kỳ vọng trên mỗi token nháp. Gọi $c$ là tỷ lệ chi phí giữa mô hình nháp và mô hình xác thực. Tính trên mỗi bước:

- Quá trình sinh thông thường gọi mô hình lớn 1 lần cho mỗi token.
- Giải mã suy đoán gọi mô hình lớn 1 lần cho $(1 - \alpha^{N+1}) / (1 - \alpha) \approx 1/(1-\alpha)$ token khi $\alpha$ ở mức cao.

Quy tắc ngón tay cái điển hình với $\alpha = 0,75$ và $N = 5$: giảm 3 lần số lượt gọi mô hình lớn. Chi phí chạy mô hình nháp rẻ hơn 5 lần. Tổng thời gian thực thi thực tế (wall-clock) giảm khoảng ~2,5 lần.

**$\alpha$ phụ thuộc vào:**

- Mức độ tương đồng giữa mô hình nháp và mô hình xác thực. Cùng một họ mô hình / cùng tập dữ liệu huấn luyện sẽ giúp $\alpha$ tăng đáng kể.
- Chiến lược giải mã. Lấy mẫu tham lam (greedy) ở cả hai mô hình mang lại $\alpha$ rất cao. Lấy mẫu theo nhiệt độ (temperature sampling) khó khớp hơn, khiến tỷ lệ chấp nhận sụt giảm.
- Bản chất tác vụ. Lập trình và sinh cấu trúc dữ liệu có tỷ lệ chấp nhận cao hơn (tính quy luật cao); viết lách sáng tạo tự do có tỷ lệ chấp nhận thấp hơn.

### Medusa — Sinh nháp không cần mô hình nháp riêng

Medusa thay thế mô hình nháp bằng các đầu ra bổ sung gắn trực tiếp trên thân mô hình xác thực. Tại vị trí `t`:

```text
thân mô hình dùng chung → trạng thái ẩn h_t
    ├── head_0: dự đoán token tại t+1  (đầu LM head tiêu chuẩn)
    ├── head_1: dự đoán token tại t+2
    ├── head_2: dự đoán token tại t+3
    ├── head_3: dự đoán token tại t+4
```

Mỗi đầu xuất ra một tập logit riêng biệt. Khi suy luận, bạn lấy mẫu từ từng đầu để tạo thành một chuỗi ứng viên, sau đó xác thực trong đúng một lượt forward bằng cơ chế chú ý dạng cây (tree-attention) cho phép đánh giá đồng thời toàn bộ các nhánh ứng viên.

Ưu điểm: không cần duy trì mô hình thứ hai. Nhược điểm: tăng thêm tham số cần học; đòi hỏi giai đoạn tinh chỉnh có giám sát (SFT trên ~1B token); tỷ lệ chấp nhận hơi thấp hơn so với phương pháp suy đoán nguyên bản sử dụng một mô hình nháp tốt.

### EAGLE — Sinh nháp vượt trội nhờ tái sử dụng trạng thái ẩn

EAGLE-1/2/3 (Li và các cộng sự, 2024–2025) biến mô hình nháp thành một transformer siêu nhỏ (thường chỉ gồm 1 tầng) tiếp nhận trực tiếp trạng thái ẩn ở tầng cuối cùng của mô hình xác thực. Nhờ việc mô hình nháp được quan sát biểu diễn đặc trưng của mô hình xác thực, các dự đoán của nó có độ tương quan cực kỳ chặt chẽ với phân phối đầu ra của mô hình xác thực. Tỷ lệ chấp nhận nhảy vọt từ ~0,6 (phương pháp nguyên bản) lên trên 0,85+.

EAGLE-3 (2025) bổ sung thêm thuật toán tìm kiếm cây (tree search) trên các chuỗi ứng viên tiềm năng. vLLM và SGLang hiện tích hợp EAGLE-2/3 làm đường dẫn suy đoán mặc định cho các dòng mô hình Llama 3/4 và Qwen 3.

### Cơ chế đồng bộ bộ đệm KV (The KV cache dance)

Khâu xác thực đưa đồng thời $N$ token nháp vào mô hình xác thực trong một lượt forward. Thao tác này mở rộng bộ đệm KV của mô hình xác thực thêm $N$ phần tử. Nếu một số token nháp bị từ chối, bạn bắt buộc phải khôi phục (rollback) bộ đệm về đúng độ dài của tiền tố đã được chấp nhận.

Các hệ thống sản xuất (tùy chọn `--speculative-config` của vLLM, `LookaheadDecoder` của TensorRT-LLM) xử lý vấn đề này bằng các bộ đệm KV tạm (scratch buffers). Ghi tạm vào bộ đệm trước, chỉ ghi nhận chính thức (commit) khi token được chấp nhận. Về mặt khái niệm điều này không khó, nhưng đòi hỏi sự tỉ mỉ trong khâu lập trình.

````figure
draft-verify-tokens
````

## Xây dựng thực tế

Xem file `code/main.py`. Chúng ta sẽ cài đặt thuật toán lấy mẫu suy đoán cốt lõi (bước bác bỏ + phân phối thặng dư) với:

- Một "mô hình lớn" mô phỏng qua phân phối xác suất softmax tất định được thiết lập thủ công (giúp chúng ta kiểm chứng toán học về tỷ lệ chấp nhận một cách giải tích).
- Một "mô hình nháp" là một biến thể bị làm nhiễu từ mô hình lớn.
- Một vòng lặp chấp nhận / từ chối tạo ra cùng một phân phối biên (marginal distribution) như khi lấy mẫu trực tiếp từ mô hình lớn.

### Bước 1: Bước chấp nhận / từ chối

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` là một số ngẫu nhiên phân phối đều trong khoảng $[0, 1)$. `q_prob` là xác suất của mô hình xác thực dành cho token nháp. `p_prob` là xác suất của mô hình nháp. Định lý Leviathan chứng minh rằng quyết định Bernoulli này, kết hợp với việc lấy mẫu từ phân phối thặng dư khi bị từ chối, sẽ bảo toàn chính xác tuyệt đối phân phối của mô hình xác thực.

### Bước 2: Phân phối thặng dư

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Lấy từng phần tử của `q` trừ đi `p`, chặn các giá trị âm về 0, sau đó chuẩn hóa lại tổng xác suất bằng 1. Lấy mẫu từ phân phối này bất cứ khi nào một token nháp bị từ chối.

### Bước 3: Một bước suy đoán hoàn chỉnh

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Năm token được chấp nhận → một token thưởng miễn phí → tạo ra sáu token chỉ trong một lượt forward duy nhất của mô hình xác thực.

### Bước 4: Đo lường tỷ lệ chấp nhận

Chạy 10.000 bước suy đoán tại các mức chất lượng mô hình nháp khác nhau. Vẽ đồ thị tỷ lệ chấp nhận theo độ phân kỳ KL giữa phân phối của mô hình nháp và mô hình xác thực. Bạn sẽ thấy một mối quan hệ đơn điệu rõ ràng.

### Bước 5: Kiểm chứng tính tương đương phân phối

Trên thực nghiệm: biểu đồ tần suất (histogram) của các token sinh ra từ vòng lặp suy đoán phải khớp hoàn hảo với biểu đồ tần suất khi lấy mẫu trực tiếp từ mô hình xác thực. Đây chính là minh chứng thực tế cho định lý Leviathan. Phép kiểm định Chi bình phương (chi-square test) sẽ xác nhận sự tương đồng trong phạm vi sai số lấy mẫu.

## Ứng dụng thực tế

Triển khai thực tế trong môi trường sản xuất:

```bash
# vLLM kết hợp với EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# vLLM kết hợp với mô hình nháp thông thường
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

TensorRT-LLM sở hữu đường dẫn Medusa nhanh nhất tính đến giữa năm 2026. Thư viện `faster-whisper` đóng gói cơ chế giải mã suy đoán cho Whisper-large bằng một mô hình nháp kích thước nhỏ.

**Lựa chọn chiến lược sinh nháp:**

| Chiến lược | Khi nào nên chọn | Mức tăng tốc |
|------------|------------------|--------------|
| Mô hình nháp thông thường (họ Llama 1B/3B) | Tạo nguyên mẫu nhanh, không cần huấn luyện lại | 1,8–2,3× |
| Các đầu Medusa | Bạn có điều kiện tinh chỉnh mô hình xác thực | 2–3× |
| EAGLE-2 / 3 | Môi trường sản xuất, cần tốc độ tối đa | 3–4× |
| Lookahead | Không mô hình nháp, không huấn luyện, không tốn thêm tham số | 1,3–1,6× |

**Khi nào KHÔNG nên áp dụng giải mã suy đoán:**

- Sinh các chuỗi ngắn chỉ từ 1–5 token. Chi phí thiết lập (overhead) sẽ lấn át lợi ích.
- Lấy mẫu văn bản cực kỳ sáng tạo với nhiệt độ cao (tỷ lệ $\alpha$ giảm mạnh).
- Các môi trường triển khai bị giới hạn nghiêm ngặt về bộ nhớ (mô hình nháp ngốn thêm VRAM).

## Đóng gói ứng dụng

Xem file `outputs/skill-spec-decode-picker.md`. Kỹ năng này hướng dẫn chọn chiến lược giải mã suy đoán (nguyên bản / Medusa / EAGLE / lookahead) và tinh chỉnh các siêu tham số ($N$, nhiệt độ mô hình nháp) cho một khối lượng công việc suy luận mới.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Xác nhận rằng phân phối token từ giải mã suy đoán khớp với phân phối lấy mẫu trực tiếp từ mô hình xác thực trên 50.000 token với giá trị $p$ của kiểm định chi bình phương đạt $p > 0,05$.
2. **Trung bình.** Vẽ đồ thị tốc độ tăng tốc (số token sinh ra trên mỗi lượt forward mô hình lớn) theo $N$ với các giá trị $\alpha = 0,5; 0,7; 0,85$. Xác định giá trị $N$ tối ưu cho từng mức $\alpha$. (Gợi ý: số token kỳ vọng trên mỗi lượt xác thực = $(1 - \alpha^{N+1}) / (1 - \alpha)$.)
3. **Khó.** Cài đặt một mô hình Medusa nhỏ gọn: lấy mô hình GPT từ Bài 14, thêm 3 đầu LM head phụ dự đoán các vị trí t+2, t+3, t+4. Huấn luyện trên tập tinyshakespeare với hàm mất mát đa đầu kết hợp. So sánh tỷ lệ chấp nhận so với việc dùng mô hình nháp thông thường cắt tỉa từ chính mô hình đó.
4. **Khó.** Cài đặt cơ chế khôi phục bộ đệm (rollback): bắt đầu với bộ đệm KV tiền tố 10 token, nạp vào 5 token nháp, mô phỏng tình huống bị từ chối tại vị trí thứ 3. Xác minh rằng dữ liệu đọc từ cache ở vòng lặp kế tiếp khớp chính xác với "tiền tố + 2 token nháp đầu tiên được chấp nhận".

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Mô hình nháp (Draft model) | "Mô hình giá rẻ" | Mô hình nhỏ hơn có nhiệm vụ đề xuất các token ứng viên; thường rẻ hơn từ 10–50 lần so với mô hình xác thực. |
| Mô hình xác thực (Verifier) | "Mô hình lớn" | Mô hình mục tiêu mà chúng ta cần bảo toàn phân phối xác suất; chỉ chạy một lần trên mỗi bước suy đoán. |
| Tỷ lệ chấp nhận ($\alpha$) | "Tần suất mô hình nháp đoán đúng" | Xác suất trên mỗi token mà mô hình xác thực chấp thuận token nháp. Thường đạt 0,7–0,9. |
| Phân phối thặng dư (Residual distribution) | "Phương án dự phòng khi bị từ chối" | Phân phối `(q - p)_+` đã được chuẩn hóa; lấy mẫu từ phân phối này khi token bị bác bỏ để bảo toàn trọn vẹn phân phối mục tiêu. |
| Token thưởng (Bonus token) | "Token miễn phí" | Khi toàn bộ $N$ token nháp đều được chấp nhận, mô hình xác thực sẽ lấy mẫu thêm một token kế tiếp mà không tốn thêm chi phí. |
| Medusa | "Suy đoán không cần mô hình nháp" | Nhiều đầu LM head trên mô hình xác thực dự đoán đồng thời các vị trí t+1..t+k. |
| EAGLE | "Mô hình nháp dựa trên trạng thái ẩn" | Transformer siêu nhỏ sinh nháp dựa trên các trạng thái ẩn ở tầng cuối cùng của mô hình xác thực. |
| Lookahead decoding | "Lặp Jacobi" | Tự suy đoán bằng thuật toán tìm điểm bất động; không cần mô hình nháp phụ. |
| Chú ý dạng cây (Tree attention) | "Xác thực nhiều ứng viên cùng lúc" | Cơ chế xác thực phân nhánh cho phép đánh giá đồng thời nhiều chuỗi nháp tiếp nối. |
| Khôi phục KV (KV rollback) | "Hủy bỏ các token nháp bị từ chối" | Sử dụng bộ đệm KV tạm; chỉ ghi nhận khi được chấp nhận, hủy bỏ khi bị từ chối. |

## Tài liệu tham khảo

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — thuật toán cốt lõi và định lý về tính tương đương phân phối.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — công bố đồng thời; chứng minh giải tích mạch lạc về phép bác bỏ Bernoulli.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — bài báo Medusa; cơ chế xác thực chú ý dạng cây.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; mô hình nháp điều kiện hóa trên trạng thái ẩn.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; độ sâu cây động.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — phương pháp lookahead không cần mô hình nháp.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham khảo chính thức tích hợp cả 4 chiến lược trong môi trường sản xuất.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — kho mã nguồn tham chiếu cho EAGLE-1/2/3.
