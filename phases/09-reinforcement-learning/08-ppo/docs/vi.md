# Tối ưu hóa Chính sách Tiệm cận (PPO)

> A2C vứt bỏ từng lượt chạy mẫu chỉ sau một lần cập nhật. PPO bao bọc gradient chính sách trong một tỉ số tầm quan trọng bị cắt (clipped importance ratio), cho phép bạn lặp qua dữ liệu đó hơn 10 epoch mà không làm bùng nổ chính sách. Schulman cùng các cộng sự (2017). Đây vẫn là thuật toán gradient chính sách mặc định hàng đầu vào năm 2026.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Thời lượng:** ~75 phút

## Vấn đề đặt ra (The Problem)

A2C (Bài học 07) hoạt động theo chính sách (on-policy): gradient $\mathbb E_{\pi_\theta}[A \cdot \nabla \log \pi_\theta]$ đòi hỏi dữ liệu phải được lấy mẫu từ chính sách *hiện tại* $\pi_\theta$. Chỉ cần thực hiện một bước cập nhật, $\pi_\theta$ thay đổi; dữ liệu vừa thu thập lập tức trở thành ngoài chính sách (off-policy). Tái sử dụng lượng dữ liệu đó sẽ khiến gradient bị chệch (biased).

Việc thu thập quỹ đạo mẫu (rollouts) rất tốn kém. Trên Atari, một lượt thu thập trên 8 môi trường song song × 128 bước = 1024 bước chuyển trạng thái (transitions) và tiêu tốn hàng chục giây thời gian môi trường. Vứt bỏ toàn bộ số dữ liệu đó chỉ sau một bước cập nhật gradient là vô cùng lãng phí.

Tối ưu hóa Chính sách Vùng Tin cậy (Trust Region Policy Optimization - TRPO, Schulman 2015) là giải pháp khắc phục đầu tiên: áp đặt ràng buộc cho mỗi lần cập nhật sao cho phân kỳ KL (Kullback-Leibler divergence) giữa chính sách cũ và mới luôn nhỏ hơn ngưỡng $\delta$. Về mặt lý thuyết, phương pháp này rất chặt chẽ, nhưng đòi hỏi phải giải bài toán gradient liên hợp (conjugate-gradient) trong từng bước cập nhật. Đến năm 2026, hầu như không còn ai dùng TRPO trong thực tế.

PPO (Schulman cùng các cộng sự 2017) thay thế ràng buộc vùng tin cậy phức tạp bằng một hàm mục tiêu bị cắt (clipped objective) đơn giản. Chỉ tốn thêm đúng một dòng mã. Cho phép chạy 10 epoch trên mỗi lượt dữ liệu thu thập. Không cần đến gradient liên hợp. Đảm bảo lý thuyết đủ tốt cho thực tiễn. Chín năm sau khi ra đời, PPO vẫn là thuật toán gradient chính sách mặc định cho mọi lĩnh vực, từ MuJoCo đến học tăng cường từ phản hồi con người (RLHF).

## Khái niệm (The Concept)

![Mục tiêu đại diện bị cắt của PPO: cắt tỉ số tại khoảng 1 ± ε](../assets/ppo.svg)

**Tỉ số tầm quan trọng (The importance ratio).**

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
```

Đây là tỉ số hợp lý (likelihood ratio) giữa chính sách mới và chính sách cũ từng thu thập dữ liệu. $r_t = 1$ nghĩa là chính sách chưa thay đổi. $r_t = 2$ nghĩa là chính sách mới có xác suất chọn hành động $a_t$ cao gấp đôi so với chính sách cũ.

**Hàm mục tiêu đại diện bị cắt (The clipped surrogate).**

```math
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \mathop{\mathrm{clip}}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
```

Bao gồm hai trường hợp:

- Nếu giá trị lợi thế $A_t > 0$ và tỉ số $r_t$ có xu hướng tăng vượt quá $1 + \epsilon$, hàm cắt (clip) sẽ làm triệt tiêu gradient (gradient bằng 0) — ngăn chặn việc đẩy một hành động tốt vượt quá ngưỡng $+\epsilon$ so với xác suất cũ.
- Nếu giá trị lợi thế $A_t < 0$ và tỉ số $r_t$ có xu hướng giảm xuống dưới $1 - \epsilon$ (nghĩa là chúng ta đang làm cho một hành động xấu có xác suất thấp hơn nhiều so với chính sách cũ), hàm cắt cũng sẽ làm phẳng gradient: không đẩy một hành động xấu xuống sâu quá $-\epsilon$ so với xác suất cũ.

Hàm `min` đảm bảo hướng ngược lại: nếu tỉ số di chuyển theo hướng *có lợi* (không làm thay đổi thái quá mức cần thiết), bạn vẫn nhận được gradient đầy đủ (không bị cắt ở phía có thể gây tổn hại).

Giá trị tiêu chuẩn thường là $\epsilon = 0.2$. Khi vẽ hàm mục tiêu theo biến $r_t$: đó là một hàm tuyến tính từng đoạn với một "mái phẳng" ở phía có lợi và một "sàn phẳng" ở phía bất lợi.

**Hàm mất mát toàn phần của PPO (The full PPO loss).**

```math
L(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_v \cdot (V_\phi(s_t) - V_t^{\text{target}})^2 + c_e \cdot H(\pi_\theta(\cdot \mid s_t))
```

Có cấu trúc Actor-Critic tương tự như A2C. Sử dụng ba hệ số, thông thường là $c_v = 0.5$, $c_e = 0.01$, $\epsilon = 0.2$.

**Vòng lặp huấn luyện (The training loop).**

1. Thu thập $N \times T$ bước chuyển trạng thái trên $N$ môi trường song song, mỗi môi trường chạy $T$ bước.
2. Tính toán giá trị lợi thế (GAE), cố định chúng như các hằng số.
3. Đóng băng $\pi_{\theta_{\text{old}}}$ như một bản chụp nhanh (snapshot) của chính sách $\pi_\theta$ hiện tại.
4. Trong $K$ epoch, với mỗi lô nhỏ (minibatch) gồm `(s, a, A, V_target, log π_old(a|s))`:
   - Tính $r_t(\theta) = \exp(\log \pi_\theta(a \mid s) - \log \pi_{\text{old}}(a \mid s))$.
   - Áp dụng hàm mất mát $L^{\text{CLIP}}$ + mất mát giá trị (value loss) + entropy.
   - Thực hiện bước cập nhật gradient.
5. Hủy bỏ dữ liệu lượt chạy. Quay trở lại bước 1.

$K = 10$ và kích thước lô nhỏ bằng 64 là bộ siêu tham số tiêu chuẩn. PPO rất ổn định (robust): thay đổi các con số này trong khoảng ±50% hiếm khi gây ra sự cố lớn.

**Biến thể phạt KL (KL-penalty variant).** Bài báo gốc đề xuất một phương án thay thế sử dụng số hạng phạt phân kỳ KL thích ứng: $L = L^{\text{PG}} - \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{old}})$ với $\beta$ được điều chỉnh linh hoạt dựa trên giá trị KL quan sát được. Phiên bản cắt (clipping) sau đó trở nên áp đảo; biến thể phạt KL vẫn được duy trì trong RLHF (nơi mà việc khống chế KL so với mô hình tham chiếu luôn là một ràng buộc bắt buộc độc lập).

```figure
ppo-clip
```

## Thực hành Xây dựng (Build It)

### Bước 1: Lưu lại `log π_old(a | s)` tại thời điểm chạy lượt thu thập mẫu

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Bản chụp nhanh được tạo đúng một lần tại thời điểm thu thập dữ liệu (rollout). Nó không thay đổi trong suốt các epoch cập nhật.

### Bước 2: Tính toán lợi thế GAE (Bài học 07)

Tương tự như A2C. Chuẩn hóa giá trị lợi thế trên toàn bộ batch.

### Bước 3: Cập nhật hàm mục tiêu đại diện bị cắt (clipped surrogate update)

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # lan truyền ngược -surrogate, cộng mất mát giá trị, trừ entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # bị cắt
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Cơ chế "bị cắt → gradient bằng 0" chính là trái tim của PPO. Nếu chính sách mới đã dịch chuyển quá xa theo hướng có lợi, quá trình cập nhật cho mẫu đó sẽ dừng lại.

### Bước 4: Mất mát giá trị và entropy (value and entropy)

Cộng thêm hàm mất mát MSE tiêu chuẩn vào mục tiêu của critic và số hạng thưởng entropy cho actor, tương tự như trong A2C.

### Bước 5: Các chỉ số chẩn đoán (Diagnostics)

Ba chỉ số quan trọng cần theo dõi trong mỗi lần cập nhật:

- **KL trung bình (Mean KL)** $\mathbb{E}[\log \pi_{\text{old}} - \log \pi_\theta]$. Cần duy trì trong khoảng `[0, 0.02]`. Nếu vượt quá `0.1`, hãy giảm `K_EPOCHS` hoặc giảm tốc độ học `LR`.
- **Tỉ lệ cắt (Clip fraction)** — tỉ lệ phần trăm các mẫu có tỉ số nằm ngoài khoảng $[1-\epsilon, 1+\epsilon]$. Nên nằm trong khoảng `~0.1-0.3`. Nếu xấp xỉ `~0`, việc cắt không bao giờ được kích hoạt → hãy tăng `LR` hoặc `K_EPOCHS`. Nếu vượt quá `~0.5+`, bạn đang học quá mức (over-fitting) vào lượt dữ liệu thu thập → hãy hạ chúng xuống.
- **Phương sai được giải thích (Explained variance)** $1 - \text{Var}(V_{\text{target}} - V_{\text{pred}}) / \text{Var}(V_{\text{target}})$. Thước đo chất lượng của mạng critic. Chỉ số này sẽ tiến dần về 1 khi critic học tốt hơn.

## Bẫy thường gặp (Pitfalls)

- **Hệ số cắt bị điều chỉnh sai.** $\epsilon = 0.2$ là chuẩn mực thực tế. Đặt `0.1` khiến các bước cập nhật quá rụt rè; từ `0.3+` trở lên dễ dẫn tới mất ổn định.
- **Quá nhiều epoch.** $K > 20$ thường xuyên gây mất ổn định vì chính sách mới trôi dạt quá xa so với $\pi_{\text{old}}$. Hãy giới hạn số epoch, đặc biệt đối với các mạng nơ-ron lớn.
- **Không chuẩn hóa phần thưởng.** Thang đo phần thưởng quá lớn sẽ làm sai lệch phạm vi cắt. Cần chuẩn hóa phần thưởng (bằng độ lệch chuẩn động) trước khi tính toán lợi thế.
- **Quên chuẩn hóa lợi thế.** Chuẩn hóa zero-mean / unit-std cho mỗi batch là quy chuẩn bắt buộc. Bỏ qua bước này sẽ làm hỏng hiệu năng của PPO trên hầu hết các benchmark.
- **Không suy giảm tốc độ học.** PPO hoạt động hiệu quả hơn nhiều khi giảm tuyến tính tốc độ học về 0. Giữ nguyên LR cố định thường cho kết quả kém hơn.
- **Lỗi tính toán tỉ số tầm quan trọng.** Luôn tính bằng `exp(log_new - log_old)` để đảm bảo độ ổn định số học, không dùng phép chia trực tiếp `new / old`.
- **Nhầm dấu gradient.** Cực đại hóa hàm đại diện tương đương với việc *cực tiểu hóa* $-L^{\text{CLIP}}$. Đảo ngược dấu là lỗi phổ biến nhất khi tự cài đặt PPO.

## Ứng dụng thực tế (Use It)

PPO là thuật toán học tăng cường mặc định năm 2026 trên rất nhiều lĩnh vực đa dạng:

| Trường hợp sử dụng | Biến thể PPO |
|--------------------|--------------|
| MuJoCo / điều khiển robot | PPO với chính sách Gauss (Gaussian policy), GAE(0.95) |
| Atari / trò chơi rời rạc | PPO với chính sách phân loại (categorical policy), lượt thu thập cuốn chiếu 128 bước |
| RLHF cho mô hình ngôn ngữ lớn (LLMs) | PPO với số hạng phạt KL so với mô hình tham chiếu, nhận thưởng từ mô hình thưởng (RM) ở cuối phản hồi |
| Tác tử trò chơi quy mô lớn | IMPALA + PPO (AlphaStar, OpenAI Five) |
| LLMs suy luận (Reasoning LLMs) | GRPO (Bài học 12) — biến thể PPO không cần mạng critic |
| Dữ liệu thuần sở thích | DPO — dạng nghiệm đóng thu gọn của PPO+KL, không cần lấy mẫu trực tuyến |

*Dạng hàm mất mát* của PPO — hàm đại diện bị cắt + mất mát giá trị + entropy — là khung sườn cốt lõi cho DPO, GRPO và hầu hết các quy trình huấn luyện RLHF hiện đại.

## Đóng gói Sản phẩm (Ship It)

Lưu thành `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy PPO trên môi trường GridWorld 4×4 với $\epsilon=0.2, K=4$. So sánh hiệu quả sử dụng mẫu với A2C (một epoch cho mỗi lượt thu thập) ở cùng số bước môi trường.
2. **Trung bình.** Quét tham số $K \in \lbrace 1, 4, 10, 30 \rbrace$. Vẽ biểu đồ phần thưởng tích lũy theo số bước môi trường và theo dõi KL trung bình mỗi lần cập nhật. Tại giá trị $K$ nào thì phân kỳ KL bùng nổ trên tác vụ này?
3. **Khó.** Thay thế hàm mục tiêu đại diện bị cắt bằng số hạng phạt KL thích ứng (gấp đôi $\beta$ nếu $\mathrm{KL} > 2 \cdot \text{target}$, giảm một nửa nếu $\mathrm{KL} < \text{target}/2$). So sánh phần thưởng tích lũy cuối cùng, độ ổn định và tính tự do không phụ thuộc vào phép cắt.

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|-----------------|
| Importance ratio | "r_t(θ)" | $\pi_\theta(a \mid s) / \pi_{\text{old}}(a \mid s)$; độ lệch so với chính sách đã thu thập dữ liệu. |
| Clipped surrogate | "Mẹo cốt lõi của PPO" | $\min(r \cdot A, \mathop{\mathrm{clip}}(r, 1-\epsilon, 1+\epsilon) \cdot A)$; làm triệt tiêu gradient khi vượt quá ngưỡng cắt ở phía có lợi. |
| Trust region | "Mục tiêu của TRPO / PPO" | Giới hạn phân kỳ KL của mỗi bước cập nhật để đảm bảo cải thiện đơn điệu. |
| KL penalty | "Vùng tin cậy mềm" | Phương án PPO thay thế: $L - \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{old}})$. Sử dụng $\beta$ thích ứng. |
| Clip fraction | "Tần suất kích hoạt cắt" | Chỉ số chẩn đoán — nên nằm trong khoảng 0.1-0.3; nằm ngoài khoảng này nghĩa là siêu tham số bị lệch. |
| Multi-epoch training | "Tái sử dụng dữ liệu" | Chạy K epoch trên mỗi lượt thu thập mẫu; đánh đổi phương sai lấy hiệu quả sử dụng mẫu. |
| On-policy-ish | "Gần như theo chính sách" | PPO trên danh nghĩa là on-policy nhưng với K>1 epoch nó sử dụng dữ liệu hơi lệch chính sách một cách an toàn. |
| PPO-KL | "Biến thể PPO kia" | Phiên bản dùng phạt KL; được dùng trong RLHF nơi ràng buộc KL so với mô hình tham chiếu đã là yêu cầu bắt buộc. |

## Tài liệu đọc thêm (Further Reading)

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Bài báo gốc.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, tiền thân của PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — Phân tích thực nghiệm chuyên sâu mọi siêu tham số của PPO.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; công thức ứng dụng PPO trong RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — Giải thích hiện đại, rõ ràng kèm mã nguồn PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — Bản cài đặt PPO tham chiếu gói gọn trong một file được nhiều bài báo sử dụng.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — Công thức triển khai PPO thực tế cho các mô hình ngôn ngữ; nên đọc kèm Bài học 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — Bài báo "37 tối ưu hóa cấp độ mã nguồn"; phân biệt các mẹo PPO then chốt thực sự với những kinh nghiệm truyền miệng.
