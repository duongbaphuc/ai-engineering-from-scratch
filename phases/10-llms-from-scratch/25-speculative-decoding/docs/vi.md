# Giải mã phỏng đoán và EAGLE (Speculative Decoding and EAGLE)

> Để một LLM ranh giới tiên phong sinh ra một token đơn lẻ đòi hỏi một lượt truyền xuôi (forward pass) hoàn chỉnh trên hàng tỷ tham số. Lượt truyền xuôi đó bị dư thừa tài nguyên tính toán nghiêm trọng: hầu hết thời gian, một mô hình nhỏ hơn nhiều có thể đoán chính xác 3-5 token tiếp theo, và mô hình lớn chỉ cần *xác thực* (verify) phỏng đoán đó. Khi phỏng đoán chính xác, bạn nhận được 5 token với chi phí tính toán của một lượt duy nhất. Giải mã phỏng đoán (Speculative decoding - Leviathan et al. 2023) đã biến điều này thành chuẩn xác tuyệt đối về mặt phân phối xác suất, và EAGLE-3 (2025) đã đẩy tỷ lệ chấp nhận lên tới ~4,5 token trên mỗi lượt xác thực — mang lại tốc độ tăng tốc 4-5 lần mà vẫn khớp hoàn toàn phân phối đầu ra của mô hình gốc.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## Vấn đề đặt ra

Thông lượng giải mã cho một mô hình cỡ 70B trên GPU H100 thường chỉ đạt khoảng 40-80 token/giây. Mỗi token sinh ra đều đòi hỏi một lượt truyền xuôi hoàn chỉnh đọc toàn bộ trọng số mô hình từ bộ nhớ băng thông cao (HBM). Bạn không thể thu nhỏ mô hình mà không làm thay đổi phân phối đầu ra của nó. Bạn không thể tăng kích thước batch vượt quá giới hạn bộ nhớ. Bạn bị kẹt — trừ khi bạn tìm ra cách cho phép mô hình xuất ra nhiều hơn một token trong mỗi lượt truyền xuôi.

Quá trình sinh tự hồi quy (autoregressive generation) thoạt nhìn có vẻ mang tính tuần tự bản chất: $x_{t+1} = \text{sample}(p(\cdot \mid x_{1:t}))$. Tuy nhiên, vẫn tồn tại cơ hội xử lý đồng thời. Nếu bạn có một bộ dự đoán chi phí rẻ nói rằng "4 token tiếp theo nhiều khả năng là [a, b, c, d]", bạn hoàn toàn có thể xác thực cả 5 vị trí trong một **lượt truyền xuôi duy nhất của mô hình lớn** và chấp nhận tiền tố khớp dài nhất.

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") đã chứng minh điều này hoàn toàn chuẩn xác toán học thông qua một quy tắc chấp nhận/bác bỏ (accept/reject rule) khéo léo giúp bảo toàn nguyên vẹn phân phối lấy mẫu của mô hình mục tiêu. Cùng một phân phối đầu ra, nhưng tốc độ nhanh hơn 2-4 lần.

## Khái niệm

### Thiết lập hai mô hình

- **Mô hình mục tiêu (Target model)** $M_p$: mô hình lớn, chậm, chất lượng cao mà bạn thực sự muốn lấy mẫu đầu ra. Phân phối: $p(x)$.
- **Mô hình phác thảo (Draft model)** $M_q$: mô hình nhỏ, tốc độ cao, chất lượng thấp hơn. Phân phối: $q(x)$. Nhỏ hơn mô hình mục tiêu từ 5 đến 30 lần.

Quy trình trên mỗi bước:

1. Mô hình phác thảo đề xuất $K$ token một cách tự hồi quy: $x_1, x_2, \dots, x_K \sim q$.
2. Mô hình mục tiêu chạy MỘT lượt truyền xuôi song song trên toàn bộ $K+1$ vị trí, tạo ra $p(x_k)$ cho từng token được đề xuất.
3. Chấp nhận/bác bỏ từng token từ trái sang phải theo quy tắc lấy mẫu bác bỏ cải tiến bên dưới. Chấp nhận tiền tố khớp dài nhất.
4. Nếu có bất kỳ token nào bị bác bỏ, lấy mẫu token thay thế từ phân phối phần dư (residual distribution) rồi dừng lại. Nếu toàn bộ được chấp nhận, lấy mẫu thêm một token thưởng từ $p(\cdot \mid x_1 \dots x_K)$.

Nếu mô hình phác thảo khớp hoàn hảo với mô hình mục tiêu, bạn thu được $K+1$ token cho mỗi lượt truyền xuôi của mô hình mục tiêu. Nếu phác thảo bị sai ngay tại vị trí 1, bạn chỉ nhận được 1 token.

### Quy tắc bảo toàn phân phối chuẩn xác

Giải mã phỏng đoán **được chứng minh toán học là tương đương hoàn toàn về phân phối với việc lấy mẫu trực tiếp từ $p$**. Quy tắc bác bỏ:

```text
Với mỗi token được phác thảo x_t:
    r ~ Uniform(0, 1)
    nếu r < p(x_t) / q(x_t):
        chấp nhận x_t
    ngược lại:
        lấy mẫu thay thế từ phần dư: (p - q)+ / ||(p - q)+||_1
        dừng lại
```

trong đó $(p - q)+$ biểu thị phần dương của hiệu số trên từng điểm xác suất. Khi mô hình phác thảo và mô hình mục tiêu đồng thuận ($p \approx q$), tỷ lệ chấp nhận gần bằng 1. Khi chúng bất đồng, phân phối phần dư được thiết kế sao cho mẫu tổng thể thu được vẫn tuân theo chính xác phân phối $p$.

**Trường hợp lấy mẫu tham lam (Greedy case).** Đối với phép lấy mẫu ở nhiệt độ bằng 0 (temperature=0), chỉ cần kiểm tra $\text{argmax}(p) == x_t$. Nếu đúng thì chấp nhận; nếu sai, xuất ra $\text{argmax}(p)$ và dừng lại.

### Mức tăng tốc kỳ vọng

Nếu tỷ lệ chấp nhận ở cấp độ token của mô hình phác thảo là $\alpha$, số lượng token kỳ vọng sinh ra trên mỗi lượt truyền xuôi của mô hình mục tiêu là:

```math
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} \qquad K = \text{độ dài phác thảo}, \ \alpha \in [0, 1]
```

Với $\alpha = 0,8, K = 4$: $(1 - 0,8^5)/(1 - 0,8) = 3,36$ token trên mỗi lượt truyền xuôi. Một lượt truyền xuôi mục tiêu tiêu tốn khoảng $\text{cost}_q \times K + \text{cost}_p$ ($K$ bước phác thảo cộng một bước xác thực mục tiêu). Nếu $\text{cost}_p \gg \text{cost}_q \times K$, tỷ lệ tăng tốc thông lượng là $3,36\times / 1 = 3,36\times$.

Tham số thực tế duy nhất quyết định hiệu quả là $\alpha$, đại lượng phụ thuộc hoàn toàn vào mức độ tương đồng giữa mô hình phác thảo và mô hình mục tiêu. Một mô hình phác thảo chất lượng là yếu tố quyết định tất cả.

### Huấn luyện mô hình phác thảo: Chưng cất tri thức

Một mô hình nhỏ ngẫu nhiên sẽ là một bộ phác thảo rất tệ. Công thức chuẩn mực là chưng cất tri thức (distillation) từ mô hình mục tiêu:

1. Chọn một kiến trúc nhỏ (~1B cho mô hình mục tiêu 70B, ~500M cho mô hình mục tiêu 7B).
2. Chạy mô hình mục tiêu trên một kho ngữ liệu văn bản lớn; lưu lại phân phối xác suất của token tiếp theo.
3. Huấn luyện mô hình phác thảo bằng hàm mất mát phân kỳ KL (KL divergence) so với phân phối của mô hình mục tiêu (chứ không phải so với token nhãn gốc - ground-truth).

Kết quả: $\alpha$ thường đạt 0,6-0,8 trên tác vụ lập trình và 0,7-0,85 trên đối thoại ngôn ngữ tự nhiên. Tốc độ tăng tốc thực tế đạt 2-3 lần trong môi trường sản xuất.

### EAGLE: Phác thảo dạng cây + Tái sử dụng đặc trưng

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") đã chỉ ra hai điểm thiếu hiệu quả trong giải mã phỏng đoán tiêu chuẩn:

1. Mô hình phác thảo thực hiện $K$ bước tuần tự, mỗi bước đều chạy qua toàn bộ các tầng mạng. Tuy nhiên, nó hoàn toàn có thể tái sử dụng các đặc trưng (trạng thái ẩn - hidden states) của mô hình mục tiêu từ lượt xác thực gần nhất — mô hình mục tiêu vốn đã tính toán các biểu diễn phong phú mà mô hình phác thảo lại đang nhọc công tính lại từ đầu.
2. Mô hình phác thảo xuất ra một chuỗi tuyến tính đơn lẻ. Nếu nó có thể xuất ra một *cây* các ứng viên (mỗi nút có nhiều phỏng đoán), thì một lượt truyền xuôi duy nhất của mô hình mục tiêu có thể xác thực song song nhiều nhánh ứng viên thông qua mặt nạ chú ý dạng cây (tree attention mask), và chọn ra nhánh được chấp nhận dài nhất.

Những thay đổi trong EAGLE-1:
- Đầu vào của phác thảo = trạng thái ẩn cuối cùng của mô hình mục tiêu tại vị trí $t$, thay vì các token thô.
- Kiến trúc phác thảo = 1 tầng giải mã transformer (chứ không phải một mô hình nhỏ độc lập hoàn chỉnh).
- Đầu ra = cây gồm $K = 4\text{-}8$ ứng viên trên mỗi độ sâu, độ sâu từ 4 đến 6.

EAGLE-2 (2024) bổ sung cấu trúc cây động (dynamic tree topology): cây mở rộng rộng hơn ở những nơi mô hình phác thảo không chắc chắn và giữ hẹp ở những nơi nó có độ tự tin cao. Điều này nâng cao $\alpha_{\text{hiệu dụng}}$ mà không làm tăng chi phí xác thực.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") loại bỏ sự phụ thuộc cố định vào đặc trưng tầng đỉnh và huấn luyện mô hình phác thảo bằng hàm mất mát "mô phỏng thời điểm kiểm tra" (test-time simulation loss) mới — mô hình phác thảo được huấn luyện trên các đầu ra khớp với phân phối lúc kiểm tra thực tế của mô hình mục tiêu thay vì phân phối huấn luyện ép buộc (teacher-forced). Tỷ lệ chấp nhận tăng từ 0,75 (EAGLE-2) lên 0,82 (EAGLE-3), và số token trung bình trên mỗi lượt xác thực tăng từ 3,0 lên 4,5.

### Xác thực bằng cơ chế chú ý dạng cây

Khi mô hình phác thảo xuất ra một cây ứng viên, mô hình mục tiêu xác thực nó trong một lượt truyền xuôi duy nhất bằng cách sử dụng **mặt nạ chú ý dạng cây (tree attention mask)** — một mặt nạ nhân quả mã hóa cấu trúc cây thay vì một đường thẳng tuần tự. Mỗi token chỉ chú ý đến các nút tổ tiên của nó trong cây. Lượt xác thực vẫn là một lượt truyền xuôi duy nhất, một phép nhân ma trận; mặt nạ topo này chỉ tốn thêm một vài phần tử đệm KV.

```text
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Nếu `a, b` là các ứng viên token đầu tiên cạnh tranh nhau và `c, d, e, f` là các ứng viên token thứ hai, toàn bộ sáu vị trí này được xác thực đồng thời trong một lượt truyền xuôi duy nhất. Đầu ra sẽ là tiền tố dài nhất dọc theo bất kỳ nhánh nào được chấp nhận.

### Khi nào phương pháp phát huy tác dụng, khi nào không

**Phát huy tác dụng vượt trội:**
- Trò chuyện / hoàn thành văn bản có tính dự đoán cao (mã nguồn, tiếng Anh thông dụng, đầu ra có cấu trúc định dạng). Giá trị $\alpha$ rất cao.
- Môi trường có tài nguyên tính toán GPU nhàn rỗi trong quá trình giải mã (giai đoạn bị nghẽn băng thông bộ nhớ - memory-bound). Phác thảo dạng cây tận dụng tốt lượng FLOP còn thừa.

**Không hiệu quả / suy giảm hiệu năng:**
- Đầu ra có tính ngẫu nhiên cao (viết sáng tạo ở nhiệt độ cao). Giá trị $\alpha$ tụt dốc về gần $1 / |\text{vocab}|$.
- Phục vụ theo lô (batch serving) với độ đồng thời rất cao — việc ghép lô đã lấp đầy năng lực FLOP của phần cứng, không còn chỗ trống cho xác thực dạng cây.
- Các mô hình mục tiêu rất nhỏ, nơi mô hình phác thảo không nhỏ hơn là bao.

Các hệ thống trong môi trường sản xuất thường ghi nhận mức tăng tốc thời gian thực từ 2-3 lần trên đối thoại, 3-5 lần trên sinh mã nguồn, và gần như bằng 0 trên các tác vụ viết văn sáng tạo.

```figure
speculative-decoding
```

## Xây dựng hệ thống

Tệp `code/main.py`:

- Hàm tham chiếu `speculative_decode(target, draft, prompt, K, temperature)` cài đặt chính xác quy tắc bác bỏ và xác minh tính bảo toàn phân phối của mô hình mục tiêu (khoảng cách phân kỳ KL thực nghiệm < 0,01 so với việc lấy mẫu trực tiếp từ mô hình mục tiêu).
- Bộ phác thảo dạng cây kiểu EAGLE xây dựng cây độ sâu $K$ với cơ chế rẽ nhánh top-p.
- Bộ tạo mặt nạ chú ý dạng cây tạo mẫu nhân quả chuẩn xác cho bộ xác thực.
- Bộ khung đo lường tỷ lệ chấp nhận chạy cả hai phương pháp trên một LM thu nhỏ (chưng cất một mô hình GPT-2-small từ mô hình mục tiêu GPT-2-medium).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """Một vòng giải mã phỏng đoán. Trả về danh sách các token được chấp nhận."""
    # 1. Phác thảo K token
    draft_tokens = []
    q_dists = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_dists.append(probs)
        state = draft_step(state, t)

    # 2. Mô hình mục tiêu tính p tại mọi vị trí phác thảo + 1 vị trí bổ sung
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Chấp nhận/bác bỏ từ trái sang phải
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_dists[k][tok]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_dists[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. Toàn bộ K token được chấp nhận → lấy mẫu token thưởng từ mô hình mục tiêu
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Sử dụng trong thực tế

- **vLLM** và **SGLang** hỗ trợ giải mã phỏng đoán như một tính năng hạng nhất. Trong vLLM, truyền cờ `--speculative-config` với đối tượng JSON chứa `method`, `model`, và `num_speculative_tokens`; EAGLE-3 được cấu hình là `"method": "eagle3"`.
- **NVIDIA TensorRT-LLM** hỗ trợ cấu trúc cây Medusa và EAGLE một cách tự nhiên.
- **Các mô hình phác thảo tham chiếu**: `Qwen/Qwen3-0.6B` (phác thảo cho Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct` (phác thảo cho Llama 3.x 70B).
- **Các đầu Medusa (Medusa heads)** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): thay vì dùng một mô hình phác thảo riêng, kỹ thuật này thêm $K$ đầu dự đoán song song trực tiếp lên chính mô hình mục tiêu. Dễ triển khai hơn, nhưng tỷ lệ chấp nhận thấp hơn một chút so với EAGLE.

## Đóng gói sản phẩm

Bài học này xây dựng `outputs/skill-speculative-tuning.md` — một kỹ năng đánh giá hồ sơ khối lượng công việc của mô hình mục tiêu và đưa ra lựa chọn tối ưu: mô hình phác thảo, độ dài $K$, độ rộng cây, nhiệt độ lấy mẫu, và thời điểm nên quay trở lại phương thức giải mã tuần tự thông thường.

## Bài tập

1. Cài đặt quy tắc bác bỏ chuẩn xác và kiểm chứng bằng thực nghiệm. Lấy 10.000 mẫu qua `speculative_decode` và qua việc lấy mẫu mục tiêu thông thường; tính khoảng cách biến thiên toàn phần (Total Variation distance) giữa hai phân phối đầu ra. Kết quả phải nhỏ hơn 0,01.

2. Tính toán công thức tăng tốc. Với $\alpha$ và $K$ cố định, hãy vẽ biểu đồ số token kỳ vọng trên mỗi lượt truyền xuôi của mô hình mục tiêu. Tìm giá trị $K$ tối ưu cho $\alpha \in \{0,5; 0,7; 0,9\}$.

3. Huấn luyện một mô hình phác thảo thu nhỏ. Lấy mô hình GPT-2 124M làm mục tiêu và chưng cất một mô hình phác thảo GPT-2 30M trên 100M token bằng hàm mất mát KL. Đo lường $\alpha$ trên tập văn bản kiểm thử độc lập. Giá trị kỳ vọng: 0,6-0,7.

4. Cài đặt phác thảo dạng cây kiểu EAGLE. Thay vì một chuỗi thẳng, cho mô hình phác thảo xuất ra top-3 nhánh ở mỗi độ sâu. Xây dựng mặt nạ chú ý dạng cây. Xác minh rằng mô hình mục tiêu chấp nhận nhánh chính xác dài nhất.

5. Đo lường các trường hợp thất bại. Chạy giải mã phỏng đoán ở nhiệt độ `temperature=1.5` (tính ngẫu nhiên cao). Chứng minh rằng $\alpha$ sụp đổ và thuật toán chạy chậm hơn giải mã thông thường do chi phí phụ trội của mô hình phác thảo.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Mô hình mục tiêu (Target model) | "Mô hình lớn" | Mô hình chậm, chất lượng cao mà bạn muốn lấy mẫu đầu ra (phân phối $p$) |
| Mô hình phác thảo (Draft model) | "Bộ phỏng đoán" | Bộ dự đoán nhỏ, nhanh (phân phối $q$); nhỏ hơn 5-30 lần |
| K / Độ dài phác thảo | "Khoảng nhìn trước (Look-ahead)" | Số lượng token phỏng đoán được tạo ra trong mỗi lượt xác thực |
| $\alpha$ / Tỷ lệ chấp nhận | "Tỷ lệ trúng (Hit rate)" | Xác suất trên mỗi token mà đề xuất của mô hình phác thảo được chấp nhận |
| Quy tắc bác bỏ chuẩn xác (Exact rejection rule) | "Phép thử chấp nhận" | Phép so sánh $r < p/q$ giúp bảo toàn tuyệt đối phân phối của mô hình mục tiêu |
| Phân phối phần dư (Residual distribution) | "Phần bù hiệu chỉnh p-q" | $(p - q)+ / \|(p - q)+\|_1$, phân phối dùng để lấy mẫu khi token bị bác bỏ |
| Phác thảo dạng cây (Tree drafting) | "Phỏng đoán rẽ nhánh" | Mô hình phác thảo xuất ra một cây ứng viên, được xác thực đồng thời bằng mặt nạ chú ý dạng cây |
| Mặt nạ chú ý dạng cây (Tree attention mask) | "Mặt nạ topo" | Mặt nạ nhân quả mã hóa cấu trúc topo của cây sao cho mỗi nút chỉ chú ý đến các nút tổ tiên |
| Các đầu Medusa (Medusa heads) | "Đầu dự đoán song song" | $K$ đầu dự đoán bổ sung gắn trực tiếp trên mô hình mục tiêu; không cần mô hình phác thảo riêng |
| Tái sử dụng đặc trưng EAGLE | "Phác thảo từ trạng thái ẩn" | Đầu vào của mô hình phác thảo là trạng thái ẩn cuối cùng của mô hình mục tiêu, giúp thu nhỏ kích thước bộ phác thảo |
| Hàm mất mát mô phỏng kiểm tra | "Huấn luyện EAGLE-3" | Huấn luyện mô hình phác thảo trên các đầu ra khớp với phân phối thực tế khi kiểm tra của mô hình mục tiêu |

## Đọc thêm

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — quy tắc bác bỏ chuẩn xác và phân tích lý thuyết về mức tăng tốc
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — bài báo đồng thời về lấy mẫu phỏng đoán tại DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — giải pháp thay thế mô hình phác thảo bằng các đầu dự đoán song song
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — kỹ thuật tái sử dụng đặc trưng và phác thảo dạng cây
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — cấu trúc cây phác thảo động
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — cơ chế khớp phân phối giữa thời điểm huấn luyện và kiểm tra
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — giải mã Jacobi/lookahead, phương pháp không cần mô hình phác thảo
