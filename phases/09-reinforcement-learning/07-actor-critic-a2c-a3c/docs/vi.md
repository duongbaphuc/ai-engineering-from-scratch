# Actor-Critic — A2C và A3C

> REINFORCE có độ biến động rất lớn. Bổ sung thêm một mạng phê bình (critic) học hàm $\hat{V}(s)$, trừ giá trị này khỏi phần thưởng tích lũy (return), bạn sẽ thu được một giá trị lợi thế (advantage) có cùng kỳ vọng nhưng phương sai thấp hơn nhiều. Đó chính là kiến trúc Actor-Critic. A2C chạy kiến trúc này một cách đồng bộ (synchronous); A3C chạy bất đồng bộ qua nhiều luồng (threads). Cả hai phương pháp này đều là mô hình tư duy nền tảng cho mọi thuật toán học tăng cường sâu (deep RL) hiện đại.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Thời lượng:** ~75 phút

## Vấn đề đặt ra (The Problem)

Thuật toán REINFORCE thuần túy (vanilla REINFORCE) tuy hoạt động được nhưng phương sai (variance) của nó vô cùng tồi tệ. Phần thưởng tích lũy Monte Carlo $G_t$ có thể dao động gấp hơn 10 lần giữa các tập (episodes). Khi nhân độ nhiễu đó với $\nabla \log \pi$ rồi lấy trung bình, ta thu được một bộ ước lượng gradient đòi hỏi hàng nghìn tập mẫu chỉ để dịch chuyển chính sách (policy) một khoảng cách mà DQN có thể đạt được với số lần cập nhật ít hơn nhiều.

Phương sai này xuất phát từ việc sử dụng phần thưởng tích lũy thô. Nếu bạn trừ đi một đường cơ sở (baseline) $b(s_t)$ — bất kỳ hàm nào chỉ phụ thuộc vào trạng thái, bao gồm cả một hàm giá trị học được — kỳ vọng toán học sẽ không đổi trong khi phương sai giảm mạnh. Đường cơ sở khả thi và tối ưu nhất chính là $\hat{V}(s_t)$. Lúc này, đại lượng nhân với $\nabla \log \pi$ được gọi là *lợi thế* (advantage):

```math
A(s, a) = G - \hat{V}(s)
```

Một hành động được coi là tốt nếu nó mang lại phần thưởng tích lũy cao hơn mức trung bình; và xấu nếu thấp hơn. REINFORCE kết hợp với một mạng phê bình học được chính là kiến trúc *Actor-Critic* (Tác tử - Nhà phê bình). Nhà phê bình đóng vai trò như một người thầy có phương sai thấp hướng dẫn cho tác tử. Đây là gốc rễ của mọi phương pháp học chính sách sâu (deep-policy methods) từ sau năm 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Khái niệm (The Concept)

![Actor-critic: mạng chính sách kết hợp mạng giá trị, phần dư TD đóng vai trò lợi thế](../assets/actor-critic.svg)

**Hai mạng, một hàm mất mát chung:**

- **Actor (Tác tử)** $\pi_\theta(a \mid s)$: chính sách hành động. Được lấy mẫu để đưa ra hành động. Được huấn luyện bằng gradient chính sách (policy gradient).
- **Critic (Nhà phê bình)** $V_\phi(s)$: ước lượng phần thưởng tích lũy kỳ vọng từ một trạng thái. Được huấn luyện để cực tiểu hóa $(V_\phi(s) - \text{target})^2$.

**Lợi thế (The advantage).** Có hai dạng chuẩn:

- *Lợi thế MC (Monte Carlo advantage):* $A_t = G_t - V_\phi(s_t)$. Không thiên vị (unbiased), nhưng phương sai cao hơn.
- *Lợi thế TD (Temporal Difference advantage):* $A_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. Có độ chệch (biased - do sử dụng $V_\phi$), nhưng phương sai thấp hơn rất nhiều. Đại lượng này còn được gọi là *phần dư TD* (TD residual) $\delta_t$.

**Lợi thế n-bước (n-step advantage).** Nội suy giữa hai dạng trên:

```math
A_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)
```

Khi $n = 1$, ta có TD thuần túy. Khi $n = \infty$, ta có MC. Hầu hết các bản cài đặt thực tế dùng $n = 5$ cho Atari, và $n = 2048$ cho PPO trên môi trường MuJoCo.

**Ước lượng Lợi thế Tổng quát (Generalized Advantage Estimation - GAE).** Schulman cùng các cộng sự (2016) đã đề xuất lấy trung bình trọng số hàm mũ trên tất cả các lợi thế n-bước:

```math
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
```

với $\lambda \in [0, 1]$. Khi $\lambda = 0$, ta thu được TD (phương sai thấp, độ chệch cao). Khi $\lambda = 1$, ta có MC (phương sai cao, không chệch). Giá trị $\lambda = 0.95$ là cấu hình mặc định năm 2026 — hãy tinh chỉnh cho đến khi tỉ lệ đánh đổi giữa độ chệch và phương sai đạt đúng mức mong muốn.

**A2C: Advantage Actor-Critic đồng bộ (Synchronous Advantage Actor-Critic).** Thu thập $T$ bước trên $N$ môi trường song song. Tính toán giá trị lợi thế cho từng bước. Cập nhật đồng thời cả actor và critic trên batch dữ liệu gộp. Lặp lại chu trình. Đây là biến thể đơn giản hơn và dễ mở rộng quy mô hơn của A3C.

**A3C: Advantage Actor-Critic bất đồng bộ (Asynchronous Advantage Actor-Critic).** Mnih cùng các cộng sự (2016). Khởi tạo $N$ luồng công nhân (worker threads), mỗi luồng chạy một môi trường riêng. Mỗi công nhân tính toán gradient cục bộ dựa trên lượt chạy (rollout) của chính mình, sau đó gửi cập nhật bất đồng bộ lên một máy chủ tham số dùng chung (shared parameter server). Không cần bộ đệm phát lại (replay buffer) — các worker tự làm mất tương quan dữ liệu nhờ chạy các quỹ đạo độc lập. A3C chứng minh rằng bạn có thể huấn luyện quy mô lớn trên CPU. Đến năm 2026, A2C dựa trên GPU (chạy môi trường song song theo lô) chiếm ưu thế áp đảo do GPU khai thác tối đa hiệu năng với kích thước batch lớn.

**Hàm mất mát kết hợp (The combined loss).**

```math
L(\theta, \phi) = -\mathbb{E}[ A_t \cdot \log \pi_\theta(a_t \mid s_t) ] + c_v \cdot \mathbb{E}[(V_\phi(s_t) - G_t)^2] - c_e \cdot \mathbb{E}[H(\pi_\theta(\cdot \mid s_t))]
```

Gồm ba số hạng: mất mát gradient chính sách, hồi quy giá trị, và số hạng thưởng entropy (entropy bonus). `c_v ~ 0.5`, `c_e ~ 0.01` là các điểm khởi đầu tiêu chuẩn.

```figure
actor-critic
```

## Thực hành Xây dựng (Build It)

### Bước 1: Khởi tạo mạng phê bình (a critic)

Mạng phê bình tuyến tính $V_\phi(s) = w \cdot \text{features}(s)$ được cập nhật bằng hàm mất mát sai số bình phương trung bình (MSE):

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Trên môi trường dạng bảng (tabular), mạng critic hội tụ sau vài trăm tập (episodes). Trên Atari, hãy thay thế critic tuyến tính bằng một thân mạng CNN chia sẻ (shared CNN trunk) kết hợp với đầu ra giá trị (value head).

### Bước 2: Lợi thế n-bước (n-step advantage)

Với một lượt thu thập (rollout) có độ dài $T$ và giá trị mồi cuối cùng (bootstrapped final value) $V(s_T)$:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` là mục tiêu huấn luyện của critic. `advantages` là đại lượng nhân với $\nabla \log \pi$.

### Bước 3: Cập nhật kết hợp (combined update)

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

Huấn luyện theo chính sách (on-policy), một lượt chạy (rollout) cho mỗi lần cập nhật, sử dụng tốc độ học (learning rate) riêng biệt cho actor và critic.

### Bước 4: Song song hóa (A3C so với A2C)

- **A3C:** Khởi chạy $N$ luồng. Mỗi luồng chạy môi trường riêng và lượt lan truyền xuôi (forward pass) riêng. Định kỳ đẩy các cập nhật gradient về máy chủ chủ (master) dùng chung. Không dùng khóa (locks) trên master — hiện tượng tương tranh (race conditions) hoàn toàn chấp nhận được, chúng chỉ tạo thêm chút nhiễu.
- **A2C:** Chạy $N$ phiên bản môi trường trong một tiến trình duy nhất, xếp chồng các quan sát thành một batch kích thước `[N, obs_dim]`, thực hiện forward pass theo lô, backward pass theo lô. Hiệu suất sử dụng GPU cao hơn, có tính xác định (deterministic), dễ suy luận và gỡ lỗi hơn. Đây là chuẩn mực mặc định năm 2026.

Mã nguồn minh họa đơn giản của chúng ta chạy đơn luồng để đảm bảo tính rõ ràng; việc viết lại sang A2C dạng lô chỉ tốn khoảng ba dòng lệnh numpy.

## Bẫy thường gặp (Pitfalls)

- **Độ chệch của Critic trước gradient của Actor.** Nếu critic khởi tạo ngẫu nhiên, đường cơ sở của nó hoàn toàn vô giá trị và bạn đang huấn luyện trên độ nhiễu thuần túy. Hãy khởi động ấm (warm up) critic trong vài trăm bước trước khi kích hoạt gradient chính sách, hoặc đặt tốc độ học của actor thật nhỏ.
- **Chuẩn hóa lợi thế (Advantage normalization).** Chuẩn hóa các giá trị lợi thế về trung bình bằng 0 / độ lệch chuẩn bằng 1 (zero-mean / unit-std) trên mỗi batch. Điều này giúp ổn định quá trình huấn luyện đáng kể với chi phí tính toán gần như bằng không.
- **Thân mạng chia sẻ (Shared trunk).** Sử dụng một bộ trích xuất đặc trưng chung (shared feature extractor) cho cả actor và critic đối với đầu vào là hình ảnh. Tách riêng các đầu ra (heads). Các đặc trưng chia sẻ sẽ tận dụng được tín hiệu học từ cả hai hàm mất mát.
- **Ràng buộc theo chính sách (On-policy contract).** A2C chỉ tái sử dụng dữ liệu cho đúng một lần cập nhật. Dùng nhiều hơn sẽ khiến gradient bị chệch (việc hiệu chỉnh bằng lấy mẫu tầm quan trọng - importance sampling - chính là cải tiến mà PPO mang lại).
- **Sụp đổ Entropy (Entropy collapse).** Nếu không có $c_e > 0$, chính sách sẽ nhanh chóng trở nên gần như tất định (near-deterministic) chỉ sau vài trăm lần cập nhật và ngừng khám phá (exploration).
- **Thang đo phần thưởng (Reward scale).** Độ lớn của lợi thế phụ thuộc vào thang đo của phần thưởng. Hãy chuẩn hóa phần thưởng (ví dụ: chia cho độ lệch chuẩn động - running standard deviation) để giữ độ lớn gradient đồng nhất giữa các tác vụ khác nhau.

## Ứng dụng thực tế (Use It)

A2C/A3C hiếm khi là lựa chọn cuối cùng vào năm 2026, nhưng chúng là kiến trúc nền tảng mà mọi phương pháp sau này đều cải tiến dựa trên đó:

| Phương pháp | Mối liên hệ với A2C |
|-------------|---------------------|
| PPO | A2C + tỉ số tầm quan trọng bị cắt (clipped importance ratio) cho các cập nhật qua nhiều epoch |
| IMPALA | A3C + cơ chế hiệu chỉnh ngoài chính sách V-trace |
| SAC (Phase 9 · 07) | A2C ngoài chính sách với critic giá trị mềm (soft-value critic) (bài học tiếp theo) |
| GRPO (Phase 9 · 12) | A2C loại bỏ critic — sử dụng lợi thế tương đối theo nhóm (group-relative advantage) |
| DPO | A2C thu gọn thành hàm mất mát xếp hạng sở thích (preference-ranking loss), không cần lấy mẫu |
| AlphaStar / OpenAI Five | A2C kết hợp huấn luyện theo giải đấu (league training) + tiền huấn luyện bắt chước (imitation pre-training) |

Nếu bạn thấy thuật ngữ "advantage" trong bất kỳ bài báo nào năm 2026, hãy nghĩ ngay đến Actor-Critic.

## Đóng gói Sản phẩm (Ship It)

Lưu thành `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Huấn luyện Actor-Critic với lợi thế MC $(G_t - V(s_t))$ trên môi trường GridWorld 4×4. So sánh hiệu quả sử dụng mẫu (sample efficiency) với REINFORCE có đường cơ sở trung bình động từ Bài học 06.
2. **Trung bình.** Chuyển sang sử dụng lợi thế phần dư TD $(r + \gamma V(s') - V(s))$. Đo lường phương sai của các batch lợi thế. Phương sai giảm đi bao nhiêu?
3. **Khó.** Cài đặt GAE(λ). Thử nghiệm quét giá trị $\lambda \in \lbrace 0, 0.5, 0.9, 0.95, 1.0 \rbrace$. Vẽ biểu đồ phần thưởng tích lũy cuối cùng so với hiệu quả sử dụng mẫu. Điểm cân bằng tối ưu giữa độ chệch và phương sai cho tác vụ này nằm ở đâu?

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|-----------------|
| Actor | "Mạng chính sách" | $\pi_\theta(a \mid s)$, được cập nhật bằng gradient chính sách. |
| Critic | "Mạng giá trị" | $V_\phi(s)$, được cập nhật bằng hồi quy MSE theo phần thưởng tích lũy / mục tiêu TD. |
| Advantage | "Tốt hơn mức trung bình bao nhiêu" | $A(s, a) = Q(s, a) - V(s)$ hoặc các bộ ước lượng của nó. Là đại lượng nhân với $\nabla \log \pi$. |
| TD residual | "δ" | $\delta_t = r + \gamma V(s') - V(s)$; ước lượng lợi thế một bước. |
| GAE | "Núm điều chỉnh nội suy" | Tổng trọng số hàm mũ của các lợi thế n-bước, được tham số hóa bởi $\lambda$. |
| A2C | "Actor-critic đồng bộ" | Chạy theo lô song song trên nhiều môi trường; một bước gradient cho mỗi lượt chạy. |
| A3C | "Actor-critic bất đồng bộ" | Các luồng công nhân đẩy gradient lên một máy chủ tham số dùng chung. Bài báo gốc; ít phổ biến hơn vào năm 2026. |
| Bootstrap | "Dùng V ở cuối chân trời mẫu" | Cắt ngắn lượt thu thập, cộng thêm $\gamma^n V(s_{t+n})$ để khép lại tổng tích lũy. |

## Tài liệu đọc thêm (Further Reading)

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, bài báo gốc về actor-critic bất đồng bộ.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Kiến thức nền tảng; tham khảo kèm Chương 9 về xấp xỉ hàm khi critic là một mạng nơ-ron.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — Actor-critic phân tán quy mô lớn với hiệu chỉnh ngoài chính sách V-trace.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — Các bản cài đặt A2C/PPO thực tế chuẩn mực rất đáng đọc.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — Kết quả hội tụ nền tảng cho phân rã actor-critic theo hai thang thời gian (two-timescale).
