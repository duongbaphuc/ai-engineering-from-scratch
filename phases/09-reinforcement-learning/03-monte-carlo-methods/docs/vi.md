# Phương pháp Monte Carlo — Học từ các tập hoàn chỉnh

> Quy hoạch động cần có mô hình. Monte Carlo không cần gì khác ngoài các tập trải nghiệm (episodes). Thực thi chính sách, theo dõi lợi nhuận thu được và lấy giá trị trung bình. Đây là ý tưởng đơn giản nhất trong học tăng cường (reinforcement learning - RL) — và cũng là chìa khóa mở ra mọi phương pháp phía sau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDP), Phase 9 · 02 (Quy hoạch động)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Quy hoạch động (dynamic programming - DP) rất chuẩn xác và thanh lịch, nhưng nó giả định rằng bạn có thể truy vấn $P(s' \mid s, a)$ cho mọi trạng thái và hành động. Trong thế giới thực, hầu như không có bài toán nào vận hành theo cách đó. Một robot không thể tính toán giải tích phân phối các pixel từ camera sau khi tác động một mô-men xoắn vào khớp. Một thuật toán định giá không thể tích phân qua mọi phản ứng tiềm năng của khách hàng. Một mô hình ngôn ngữ lớn (LLM) không thể liệt kê toàn bộ các chuỗi tiếp nối khả dĩ sau khi sinh ra một token.

Bạn cần một phương pháp chỉ yêu cầu khả năng *lấy mẫu (sample)* từ môi trường: Chạy chính sách. Thu thập một quỹ đạo (trajectory) `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Dùng quỹ đạo đó để ước tính giá trị. Đó chính là phương pháp Monte Carlo (Monte Carlo methods - MC).

Bước chuyển từ DP sang MC mang ý nghĩa quan trọng về mặt tư duy: chúng ta chuyển dịch từ *mô hình đã biết + cập nhật chính xác* sang *các lượt tương tác lấy mẫu (rollouts) + lợi nhuận trung bình*. Phương sai (variance) sẽ tăng lên, nhưng phạm vi ứng dụng được mở rộng bùng nổ. Mọi thuật toán RL sau bài học này — TD, Q-learning, REINFORCE, PPO, GRPO — về bản chất đều là các bộ ước lượng Monte Carlo, đôi khi được kết hợp thêm kỹ thuật bootstrapping (tự cập nhật dựa trên ước tính kế tiếp).

## Khái niệm cốt lõi (The Concept)

![Monte Carlo: chạy tập mẫu, tính lợi nhuận, lấy trung bình; lần đầu viếng thăm so với mọi lần viếng thăm](../assets/monte-carlo.svg)

**Ý tưởng cốt lõi gói gọn trong một dòng:** $V^\pi(s) = \mathbb E_\pi[G_t \mid s_t = s] \approx (1/N) \sum_i G^{(i)}(s)$ trong đó $G^{(i)}(s)$ là các lợi nhuận (returns) quan sát được sau các lần ghé thăm trạng thái $s$ dưới chính sách $\pi$.

**MC lần đầu viếng thăm (First-visit MC) so với MC mọi lần viếng thăm (Every-visit MC).** Trong một tập trải nghiệm mà trạng thái $s$ được ghé thăm nhiều lần: first-visit MC chỉ tính lợi nhuận kể từ lần ghé thăm đầu tiên; every-visit MC tính lợi nhuận từ tất cả các lần ghé thăm. First-visit MC là bộ ước lượng không chệch (unbiased). Every-visit MC có thể bị chệch nhẹ khi số lượng tập trải nghiệm còn hữu hạn, bởi vì nó lấy trung bình của một số lượng lợi nhuận ngẫu nhiên trên mỗi tập và số lượng đó có thể tương quan với lợi nhuận; tuy nhiên cả hai phương pháp đều hội tụ về $V^\pi(s)$ khi số lượt ghé thăm tiến tới vô cùng. First-visit dễ phân tích toán học hơn (các mẫu độc lập cùng phân phối - iid). Every-visit tận dụng được nhiều dữ liệu hơn trên mỗi tập và thường hội tụ nhanh hơn trong thực nghiệm.

**Giá trị trung bình gia tăng (Incremental mean).** Thay vì lưu trữ toàn bộ các giá trị lợi nhuận, chúng ta cập nhật giá trị trung bình động (running average):

```math
V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]
```

Biến đổi lại: $V_{\text{new}} = V_{\text{old}} + \alpha \cdot (\text{target} - V_{\text{old}})$ với $\alpha = 1/n$. Khi thay $1/n$ bằng một tốc độ học cố định $\alpha \in (0, 1)$, bạn thu được một bộ ước lượng MC cho môi trường phi dừng (non-stationary) có khả năng bám sát các thay đổi của $\pi$. Bước chuyển này chính là nền tảng để bước từ MC sang TD và tới toàn bộ các thuật toán RL hiện đại.

**Vấn đề thăm dò (Exploration).** DP duyệt qua mọi trạng thái bằng cách liệt kê toàn bộ. MC chỉ nhìn thấy các trạng thái mà chính sách thực sự ghé thăm. Nếu $\pi$ là một chính sách tất định, các vùng không gian trạng thái rộng lớn sẽ không bao giờ được lấy mẫu, và ước tính giá trị của chúng sẽ mãi mãi bằng 0. Ba giải pháp kinh điển gồm:

1. **Khởi đầu thăm dò (Exploring starts).** Bắt đầu mỗi tập từ một cặp $(s, a)$ ngẫu nhiên. Đảm bảo bao phủ toàn bộ không gian; tuy nhiên trong thực tế điều này bất khả thi (bạn không thể "đặt lại" một robot vào một trạng thái tùy ý trong môi trường thực).
2. **Tham lam theo xác suất ε (ε-greedy).** Hành động tham lam dựa trên giá trị Q hiện tại, nhưng với xác suất $\epsilon$ sẽ chọn một hành động ngẫu nhiên. Mọi cặp trạng thái - hành động đều được lấy mẫu khi thời gian tiến tới vô hạn.
3. **MC ngoài chính sách (Off-policy MC).** Thu thập dữ liệu dưới một chính sách hành vi (behavior policy) $\mu$, sau đó học về chính sách mục tiêu (target policy) $\pi$ thông qua kỹ thuật lấy mẫu tầm quan trọng (importance sampling). Phương sai cao, nhưng đây là cầu nối dẫn tới các phương pháp dùng bộ đệm phát lại (replay buffer) như DQN.

**Điều khiển Monte Carlo (Monte Carlo Control).** Đánh giá → Cải tiến → Đánh giá, tương tự như lặp chính sách, nhưng bước đánh giá được thực hiện qua lấy mẫu:

1. Chạy chính sách $\pi$, thu thập một tập trải nghiệm.
2. Cập nhật $Q(s, a)$ từ các lợi nhuận quan sát được.
3. Cập nhật $\pi$ thành chính sách ε-greedy đối với $Q$.
4. Lặp lại quy trình.

Thuật toán hội tụ về $Q^{\ast}$ và $\pi^{\ast}$ với xác suất bằng 1 dưới các điều kiện hội tụ tiêu chuẩn (mỗi cặp trạng thái - hành động được ghé thăm vô hạn lần, chuỗi $\alpha$ thỏa mãn điều kiện Robbins-Monro).

```figure
epsilon-greedy
```

## Tự xây dựng (Build It)

### Bước 1: Thu thập quỹ đạo (Rollout) → danh sách các bộ (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Không cần mô hình toán học, chỉ cần hai hàm `env.reset()` và `env.step(s, a)`. Giao diện tương tự như môi trường Gym nhưng được tinh giản tối đa.

### Bước 2: Tính toán lợi nhuận (Quét ngược từ cuối tập)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Chỉ cần một lượt duyệt, độ phức tạp $O(T)$. Hệ thức truy hồi ngược $G_t = r_{t+1} + \gamma G_{t+1}$ giúp tránh việc phải tính tổng lặp lại nhiều lần.

### Bước 3: Đánh giá chính sách bằng MC lần đầu viếng thăm (First-visit MC evaluation)

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Ba thao tác chủ chốt: đánh dấu trạng thái đã gặp trong lần đầu viếng thăm, tăng biến đếm số lần xuất hiện, và cập nhật giá trị trung bình động.

### Bước 4: Điều khiển MC theo chính sách với ε-greedy (On-policy ε-greedy MC control)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Bước 5: So sánh với chuẩn mực vàng của DP

Ước tính MC cho $V^\pi$ sẽ tiệm cận kết quả của thuật toán DP từ Bài 02 khi số lượng tập trải nghiệm tiến tới vô cùng. Trong thực nghiệm: 50.000 tập trên GridWorld 4×4 cho kết quả sai lệch trong khoảng `~0.1` so với giá trị chính xác từ DP.

## Cạm bẫy thường gặp (Pitfalls)

- **Các tập trải nghiệm vô hạn.** Phương pháp MC bắt buộc các tập trải nghiệm phải *kết thúc*. Nếu chính sách của bạn bị kẹt trong vòng lặp vô tận, hãy áp đặt ngưỡng `max_steps` và xem việc chạm ngưỡng là thất bại ngầm định. Trong GridWorld với chính sách ngẫu nhiên, việc hết thời gian (timeout) xảy ra thường xuyên — điều này bình thường, chỉ cần bạn xử lý tính toán chính xác.
- **Phương sai cao (Variance).** MC sử dụng toàn bộ lợi nhuận thực tế. Trong các tập trải nghiệm dài, phương sai rất lớn — một phần thưởng ngẫu nhiên không may ở cuối tập có thể làm lệch giá trị $V(s_0)$ một lượng tương đương. Các phương pháp hiệu số thời gian (Temporal Difference - TD ở Bài 04) khắc phục điều này bằng bootstrapping.
- **Độ bao phủ trạng thái.** Chính sách thuần tham lam đối với một bảng Q khởi tạo ban đầu (khi các giá trị bằng nhau) sẽ chỉ luôn chọn duy nhất một hành động. Bạn *bắt buộc* phải khám phá (dùng ε-greedy, khởi đầu khám phá, hoặc UCB).
- **Chính sách không dừng (Non-stationary policies).** Nếu $\pi$ liên tục thay đổi (như trong MC control), các lợi nhuận cũ thu được là từ một chính sách đã lỗi thời. Phương pháp MC với $\alpha$ cố định xử lý tốt vấn đề này; lấy trung bình mẫu số học đơn thuần sẽ không thích ứng kịp.
- **Lấy mẫu tầm quan trọng ngoài chính sách (Off-policy importance sampling).** Tỉ số trọng số $\pi(a \mid s)/\mu(a \mid s)$ nhân tích lũy qua toàn bộ quỹ đạo. Phương sai sẽ bùng nổ theo chiều dài tập. Hãy giảm thiểu bằng kỹ thuật lấy mẫu tầm quan trọng có trọng số cho từng bước (per-decision weighted IS) hoặc chuyển sang dùng TD.

## Ứng dụng thực tế (Use It)

Vai trò của các phương pháp Monte Carlo trong ngăn xếp công nghệ năm 2026:

| Tình huống sử dụng | Lý do sử dụng MC |
|--------------------|-------------------|
| Trò chơi có tầm nhìn ngắn (blackjack, poker) | Các ván chơi kết thúc tự nhiên; tín hiệu lợi nhuận rõ ràng. |
| Đánh giá ngoại tuyến (Offline evaluation) một chính sách | Tính trung bình lợi nhuận chiết khấu trên các quỹ đạo đã lưu trữ trong log. |
| Tìm kiếm cây Monte Carlo (MCTS trong AlphaZero) | Các đợt rollout MC từ các nút lá của cây giúp định hướng chọn lựa. |
| Đánh giá RL cho LLM | Tính phần thưởng trung bình trên các đoạn văn bản sinh ra từ một chính sách cho trước. |
| Ước tính đường cơ sở (Baseline) trong PPO | Mục tiêu lợi thế $A_t = G_t - V(s_t)$ sử dụng lợi nhuận Monte Carlo $G_t$. |
| Giảng dạy học tăng cường | Thuật toán đơn giản nhất thực sự hoạt động — loại bỏ bootstrapping để nắm bắt bản chất cốt lõi. |

Các thuật toán học tăng cường sâu hiện đại (PPO, SAC) thường kết hợp nội suy giữa MC thuần túy (dùng toàn bộ lợi nhuận) và TD thuần túy (bootstrap một bước) thông qua lợi nhuận $n$-bước ($n$-step returns) hoặc ước lượng lợi thế tổng quát (GAE). Cả hai đầu mút thực chất đều là biểu hiện của cùng một bộ ước lượng.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Triển khai thuật toán đánh giá chính sách first-visit MC cho chính sách ngẫu nhiên đều trên GridWorld 4×4. Chạy 10.000 tập trải nghiệm. Vẽ đồ thị $V(0,0)$ theo số lượng tập và so sánh với giá trị chuẩn từ DP.
2. **Trung bình.** Triển khai điều khiển MC dùng ε-greedy với $\epsilon \in \lbrace 0.01, 0.1, 0.3 \rbrace$. So sánh lợi nhuận trung bình sau 20.000 tập. Hình dạng đường cong học tập như thế nào? Sự đánh đổi giữa độ chệch và phương sai (bias-variance tradeoff) thể hiện ở đâu?
3. **Khó.** Triển khai phương pháp MC *ngoài chính sách* (off-policy MC) với lấy mẫu tầm quan trọng: thu thập dữ liệu dưới chính sách ngẫu nhiên đều $\mu$, ước tính $V^\pi$ cho chính sách tối ưu tất định $\pi$. So sánh lấy mẫu tầm quan trọng thông thường (plain IS), lấy mẫu tầm quan trọng theo từng quyết định (per-decision IS), và lấy mẫu tầm quan trọng có trọng số (weighted IS). Phương pháp nào có phương sai thấp nhất?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Monte Carlo | "Lấy mẫu ngẫu nhiên" | Ước tính giá trị kỳ vọng bằng cách tính trung bình trên các mẫu độc lập cùng phân phối (iid) từ phân phối đó. |
| Lợi nhuận $G_t$ (Return) | "Phần thưởng tương lai" | Tổng các phần thưởng chiết khấu từ bước $t$ đến khi kết thúc tập: $\sum_{k\ge 0} \gamma^k r_{t+k+1}$. |
| First-visit MC | "Đếm mỗi trạng thái một lần" | Chỉ lần ghé thăm đầu tiên trong một tập đóng góp vào giá trị ước tính. |
| Every-visit MC | "Dùng mọi lần ghé thăm" | Mọi lần ghé thăm đều đóng góp; bị chệch nhẹ trên mẫu hữu hạn, nhất quán ở giới hạn vô hạn, và hiệu quả dữ liệu cao hơn. |
| ε-greedy | "Nhiễu thăm dò" | Chọn hành động tham lam với xác suất $1-\epsilon$; chọn hành động ngẫu nhiên với xác suất $\epsilon$. |
| Lấy mẫu tầm quan trọng (Importance sampling) | "Hiệu chỉnh việc lấy mẫu từ sai phân phối" | Nhân lại trọng số của lợi nhuận theo tỉ số $\pi(a \mid s)/\mu(a \mid s)$ để ước tính $V^\pi$ từ dữ liệu của $\mu$. |
| Theo chính sách (On-policy) | "Học từ chính dữ liệu của mình" | Chính sách mục tiêu trùng với chính sách hành vi. MC nguyên bản, PPO, SARSA. |
| Ngoài chính sách (Off-policy) | "Học từ dữ liệu của người khác" | Chính sách mục tiêu khác biệt với chính sách hành vi. MC dùng importance sampling, Q-learning, DQN. |

## Tài liệu đọc thêm (Further Reading)

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Tài liệu tham khảo kinh điển về phương pháp Monte Carlo trong RL.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — Phân tích chi tiết giữa first-visit và every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — Nghiên cứu về MC ngoài chính sách và kiểm soát phương sai.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — Các bộ ước lượng lấy mẫu tầm quan trọng có trọng số hiện đại với phương sai thấp.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — Minh chứng thực nghiệm quy mô lớn đầu tiên về việc tự chơi (self-play) kết hợp MC/TD đạt tới trình độ siêu phàm; tiền đề tư tưởng cho mọi bài học ở nửa sau giai đoạn này.
