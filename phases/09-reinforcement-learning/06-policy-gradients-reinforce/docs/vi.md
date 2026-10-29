# Gradient chính sách — Tự xây dựng REINFORCE từ đầu

> Hãy ngừng việc ước tính hàm giá trị. Hãy tham số hóa trực tiếp chính sách, tính toán gradient của lợi nhuận kỳ vọng, và bước lên dốc theo hướng tối ưu. Ronald Williams (1992) đã chứng minh điều này chỉ trong một định lý. Đây chính là lý do tồn tại của PPO, GRPO và mọi vòng lặp học tăng cường cho mô hình ngôn ngữ lớn (LLM RL).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Lan truyền ngược), Phase 9 · 03 (Phương pháp Monte Carlo), Phase 9 · 04 (Học TD)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Q-learning và DQN tham số hóa *hàm giá trị* (value function). Bạn lựa chọn hành động thông qua phép toán `argmax Q`. Cách làm này hoạt động tốt đối với các hành động rời rạc và không gian trạng thái nhỏ. Nhưng nó sẽ sụp đổ khi không gian hành động là liên tục (làm sao bạn có thể tính `argmax` trên một vector mô-men xoắn 10 chiều?) hoặc khi bạn mong muốn một chính sách ngẫu nhiên (chính sách dựa trên `argmax` luôn mang tính tất định theo định nghĩa).

Phương pháp gradient chính sách (Policy gradients) chọn cách tiếp cận khác: tham số hóa trực tiếp *chính sách*. $\pi_\theta(a \mid s)$ là một mạng nơ-ron xuất ra một phân phối xác suất trên các hành động. Chúng ta lấy mẫu từ phân phối đó để hành động. Sau đó tính toán gradient của lợi nhuận kỳ vọng đối với các tham số $\theta$ và bước lên dốc gradient (gradient ascent). Không cần phép tính `argmax`. Không cần hệ thức truy hồi Bellman. Chỉ đơn thuần là tối ưu hóa dốc gradient trên hàm mục tiêu $J(\theta) = \mathbb E_{\pi_\theta}[G]$.

Định lý REINFORCE (Williams, 1992) khẳng định rằng gradient này hoàn toàn khả vi và tính toán được: $\nabla J(\theta) = \mathbb E_\pi[ G \cdot \nabla_\theta \log \pi_\theta(a \mid s) ]$. Hãy chạy một tập trải nghiệm. Tính toán lợi nhuận thu về. Nhân nó với $\nabla \log \pi_\theta(a \mid s)$ ở mỗi bước đi. Lấy trung bình. Thực hiện bước cập nhật tăng dốc gradient. Hoàn tất.

Mọi thuật toán RL cho LLM vào năm 2026 — PPO, DPO, GRPO — đều là các biến thể tinh chỉnh từ REINFORCE. Nắm vững bản chất của nó chính là điều kiện tiên quyết cho toàn bộ phần còn lại của giai đoạn này, cũng như cho Phase 10 · 07 (triển khai RLHF) và Phase 10 · 08 (DPO).

## Khái niệm cốt lõi (The Concept)

![Gradient chính sách: chính sách softmax, gradient log-π, cập nhật có trọng số theo lợi nhuận](../assets/policy-gradient.svg)

**Định lý gradient chính sách (The policy gradient theorem).** Với bất kỳ chính sách $\pi_\theta$ nào được tham số hóa bởi $\theta$:

```math
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[ \sum_{t=0}^{T} G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) ]
```

trong đó $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$ là lợi nhuận chiết khấu tính từ bước $t$. Kỳ vọng được lấy trên toàn bộ các quỹ đạo $\tau$ thu thập từ chính sách $\pi_\theta$.

**Chứng minh rất ngắn gọn.** Lấy đạo hàm của $J(\theta) = \sum_\tau P(\tau; \theta) G(\tau)$ bên dưới dấu tích phân/kỳ vọng. Áp dụng đồng nhất thức $\nabla P(\tau; \theta) = P(\tau; \theta) \nabla \log P(\tau; \theta)$ (thủ thuật đạo hàm logarit hay log-derivative trick). Phân tích $\log P(\tau; \theta) = \sum \log \pi_\theta(a_t \mid s_t) + \text{các số hạng môi trường không phụ thuộc vào } \theta$. Các số hạng của môi trường triệt tiêu khi lấy đạo hàm theo $\theta$. Chỉ sau hai dòng biến đổi đại số, bạn thu được định lý trên.

**Các kỹ thuật giảm phương sai (Variance reduction tricks).** Thuật toán REINFORCE nguyên bản có phương sai cực kỳ lớn — lợi nhuận bị nhiễu, $\nabla \log \pi$ bị nhiễu, và tích của chúng tạo ra nhiễu loạn trầm trọng. Hai giải pháp kinh điển gồm:

1. **Trừ đường cơ sở (Baseline subtraction).** Thay thế $G_t$ bằng $G_t - b(s_t)$ với bất kỳ hàm đường cơ sở $b(s_t)$ nào không phụ thuộc vào hành động $a_t$. Ước lượng này không chệch vì $\mathbb{E}[b(s_t) \cdot \nabla \log \pi(a_t \mid s_t)] = 0$. Lựa chọn tiêu chuẩn: $b(s_t) = \hat{V}(s_t)$ được học bởi một mạng phê bình (critic) → kiến trúc Actor-Critic (Bài 07).
2. **Phần thưởng còn lại (Reward-to-go).** Thay thế $\sum_t G_t \cdot \nabla \log \pi_\theta(a_t \mid s_t)$ bằng $\sum_t G_t^{\text{từ } t} \cdot \nabla \log \pi_\theta(a_t \mid s_t)$. Chỉ có phần thưởng trong tương lai mới mang lại ý nghĩa cho hành động hiện tại — các phần thưởng trong quá khứ chỉ đóng góp thêm nhiễu có kỳ vọng bằng 0.

Kết hợp cả hai kỹ thuật, ta có:

```math
\nabla J \approx (1/N) \sum_{i=1}^{N} \sum_{t=0}^{T_i} [ G_t^{(i)} - \hat{V}(s_t^{(i)}) ] \cdot \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})
```

đây chính là thuật toán REINFORCE kết hợp đường cơ sở — tiền thân trực tiếp của A2C (Bài 07) và PPO (Bài 08).

**Tham số hóa chính sách Softmax.** Đối với các hành động rời rạc, lựa chọn tiêu chuẩn là:

```math
\pi_\theta(a \mid s) = \exp(f_\theta(s, a)) / \sum_{a'} \exp(f_\theta(s, a'))
```

trong đó $f_\theta$ là bất kỳ mạng nơ-ron nào xuất ra điểm số cho từng hành động. Gradient có dạng giải tích rất tường minh:

```math
\nabla_\theta \log \pi_\theta(a \mid s) = \nabla_\theta f_\theta(s, a) - \sum_{a'} \pi_\theta(a' \mid s) \nabla_\theta f_\theta(s, a')
```

nghĩa là điểm số của hành động đã chọn trừ đi giá trị kỳ vọng của nó dưới chính sách hiện tại.

**Chính sách Gauss (Gaussian policy) cho hành động liên tục.** $\pi_\theta(a \mid s) = N(\mu_\theta(s), \sigma_\theta(s))$. Biểu thức $\nabla \log N(a; \mu, \sigma)$ có dạng đóng tường minh. Đó là tất cả những gì thuật toán SAC ở Phase 9 · 07 cần đến.

```figure
policy-gradient-landscape
```

## Tự xây dựng (Build It)

### Bước 1: Mạng chính sách Softmax

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Chúng ta sử dụng một chính sách tuyến tính (mỗi hành động ứng với một vector trọng số) cho môi trường dạng bảng. Đối với trò chơi Atari, chỉ cần thay bằng mạng tích chập CNN và giữ nguyên đầu ra softmax.

### Bước 2: Lấy mẫu hành động và tính log-xác suất

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Bước 3: Thu thập quỹ đạo cùng với log-xác suất

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Bước 4: Bước cập nhật REINFORCE

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient $\nabla \log \pi(a \mid s) = e_a - \pi(\cdot \mid s)$ (vector one-hot của $a$ trừ đi phân phối xác suất) chính là trái tim của gradient chính sách softmax. Hãy ghi nhớ công thức này thật kỹ.

### Bước 5: Đường cơ sở (Baselines)

Một giá trị trung bình động của $G$ trên các tập gần nhất đã đủ để giảm phương sai giúp môi trường GridWorld 4×4 hoạt động ổn định; thuật toán hội tụ sau khoảng ~500 tập. Nâng cấp đường cơ sở thành một hàm được học $\hat{V}(s)$ và bạn sẽ có kiến trúc Actor-Critic hoàn chỉnh.

## Cạm bẫy thường gặp (Pitfalls)

- **Bùng nổ gradient (Exploding gradients).** Lợi nhuận có thể đạt giá trị rất lớn. Hãy luôn chuẩn hóa $G$ về phân phối `~N(0, 1)` trên toàn batch trước khi nhân với $\nabla \log \pi$.
- **Suy sụp entropy (Entropy collapse).** Chính sách hội tụ về một hành động gần như tất định quá sớm, ngừng việc khám phá và bị mắc kẹt. Giải pháp: bổ sung số hạng thưởng entropy $\beta \cdot H(\pi(\cdot \mid s))$ vào hàm mục tiêu.
- **Phương sai cao.** REINFORCE nguyên bản đòi hỏi hàng nghìn tập trải nghiệm mới học được. Sử dụng đường cơ sở critic (Bài 07) hoặc vùng tin cậy (trust region trong TRPO/PPO ở Bài 08) là biện pháp tiêu chuẩn.
- **Kém hiệu quả về dữ liệu (Sample inefficiency).** Tính chất theo chính sách (on-policy) đồng nghĩa với việc bạn phải bỏ đi toàn bộ dữ liệu chuyển trạng thái sau một lượt cập nhật. Kỹ thuật hiệu chỉnh ngoài chính sách qua lấy mẫu tầm quan trọng giúp tái sử dụng dữ liệu nhưng làm tăng phương sai (tỉ số trong PPO là một trọng số IS bị cắt xén).
- **Gradient không dừng (Non-stationary gradients).** Gradient thu được từ 100 tập trước đã dựa trên một chính sách cũ $\pi$. Đó là lý do tại sao các phương pháp on-policy luôn cập nhật mạng sau mỗi vài lượt rollout.
- **Phân bổ trách nhiệm (Credit assignment).** Nếu không dùng reward-to-go, các phần thưởng trong quá khứ sẽ gây nhiễu cho các quyết định hiện tại. Hãy luôn sử dụng reward-to-go.

## Ứng dụng thực tế (Use It)

Trong năm 2026, REINFORCE hiếm khi được sử dụng trực tiếp nhưng công thức gradient của nó hiện diện ở khắp mọi nơi:

| Tình huống sử dụng | Phương pháp phát triển |
|--------------------|------------------------|
| Điều khiển liên tục | PPO / SAC với chính sách Gauss |
| RLHF cho LLM | PPO với hình phạt KL, chạy trên chính sách cấp độ token |
| Mô hình suy luận LLM (DeepSeek) | GRPO — REINFORCE với đường cơ sở tương đối theo nhóm, không cần critic |
| Hệ thống đa tác tử (Multi-agent) | REINFORCE với critic tập trung (MADDPG, COMA) |
| Robotics với hành động rời rạc | A2C, A3C, PPO |
| Tối ưu hóa dựa trên sở thích (Preference-only) | DPO — REINFORCE được viết lại thành hàm mất mát khả dĩ sở thích, không cần lấy mẫu |

Khi bạn đọc thấy dòng mã `loss = -advantage * log_prob` trong một kịch bản huấn luyện năm 2026, đó chính là REINFORCE kết hợp với đường cơ sở. Toàn bộ các bài báo khoa học lớn (DPO, GRPO, RLOO) đều là những kỹ thuật giảm phương sai được xây dựng trực tiếp trên dòng lệnh cốt lõi này.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Triển khai REINFORCE trên GridWorld 4×4 với một chính sách softmax tuyến tính. Huấn luyện trong 1.000 tập trải nghiệm không dùng đường cơ sở. Vẽ đường cong học tập; đo lường phương sai (độ lệch chuẩn của lợi nhuận).
2. **Trung bình.** Bổ sung đường cơ sở trung bình động (running-mean baseline). Huấn luyện lại. So sánh hiệu quả sử dụng mẫu và phương sai so với bản chạy nguyên bản. Đường cơ sở giúp giảm bao nhiêu bước đi để đạt tới sự hội tụ?
3. **Khó.** Bổ sung phần thưởng entropy $\beta \cdot H(\pi)$. Thử nghiệm quét siêu tham số $\beta \in \lbrace 0, 0.01, 0.1, 1.0 \rbrace$. Vẽ biểu đồ lợi nhuận cuối cùng và entropy của chính sách. Giá trị tối ưu (sweet spot) cho tác vụ này nằm ở đâu?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Gradient chính sách (Policy gradient) | "Huấn luyện chính sách trực tiếp" | $\nabla J(\theta) = \mathbb{E}[G \cdot \nabla \log \pi_\theta(a \mid s)]$; suy ra từ thủ thuật đạo hàm logarit. |
| REINFORCE | "Thuật toán gradient chính sách gốc" | Ronald Williams (1992); lợi nhuận Monte Carlo nhân với gradient của log-policy. |
| Thủ thuật đạo hàm logarit (Log-derivative trick) | "Bộ ước lượng hàm điểm (score function)" | $\nabla P(\tau;\theta) = P(\tau;\theta) \cdot \nabla \log P(\tau;\theta)$; giúp việc tính gradient của kỳ vọng trở nên khả thi. |
| Đường cơ sở (Baseline) | "Kỹ thuật giảm phương sai" | Bất kỳ đại lượng $b(s)$ nào được trừ khỏi $G$; bảo toàn tính không chệch vì $\mathbb{E}[b \cdot \nabla \log \pi] = 0$. |
| Phần thưởng còn lại (Reward-to-go) | "Chỉ tính lợi nhuận tương lai" | Dùng $G_t^{\text{từ } t}$ thay vì toàn bộ $G_0$; chuẩn xác và có phương sai thấp hơn nhiều. |
| Phần thưởng entropy (Entropy bonus) | "Khuyến khích khám phá" | Số hạng $+\beta \cdot H(\pi(\cdot \mid s))$ giúp chính sách không bị suy sụp sớm về dạng tất định. |
| Theo chính sách (On-policy) | "Học từ những gì vừa nhìn thấy" | Kỳ vọng của gradient lấy theo chính sách hiện hành — không thể tái sử dụng trực tiếp dữ liệu cũ. |
| Lợi thế (Advantage) | "Tốt hơn mức trung bình bao nhiêu" | $A(s, a) = G(s, a) - V(s)$; đại lượng có dấu mà REINFORCE có đường cơ sở nhân vào. |

## Tài liệu đọc thêm (Further Reading)

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — Bài báo gốc đặt nền móng cho thuật toán REINFORCE.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — Định lý gradient chính sách hiện đại với xấp xỉ hàm.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Trình bày hệ thống trong sách giáo khoa kinh điển.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — Giải thích sư phạm trực quan kèm mã nguồn mẫu bằng PyTorch.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — Nghiên cứu về giảm phương sai và góc nhìn gradient tự nhiên (natural gradient) kết nối REINFORCE với họ thuật toán vùng tin cậy (TRPO, PPO).
