# Hiệu số thời gian — Q-Learning & SARSA

> Monte Carlo phải đợi cho đến khi tập trải nghiệm kết thúc. TD cập nhật sau mỗi bước đi bằng kỹ thuật tự cập nhật (bootstrapping) từ ước tính giá trị kế tiếp. Q-learning mang tính ngoài chính sách (off-policy) và lạc quan; SARSA mang tính theo chính sách (on-policy) và thận trọng. Cả hai chỉ gói gọn trong một dòng mã. Cả hai đều là nền móng vững chắc cho mọi phương pháp học tăng cường sâu trong giai đoạn này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDP), Phase 9 · 02 (Quy hoạch động), Phase 9 · 03 (Phương pháp Monte Carlo)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Phương pháp Monte Carlo hoạt động tốt nhưng đòi hỏi hai điều kiện tốn kém: Các tập trải nghiệm bắt buộc phải kết thúc, và thuật toán chỉ cập nhật sau khi đã có toàn bộ lợi nhuận cuối cùng. Nếu tập trải nghiệm kéo dài 1.000 bước, MC phải chờ đợi đủ 1.000 bước mới có thể cập nhật bất kỳ thông tin nào. Điều này dẫn đến phương sai cao (high-variance), độ chệch thấp (low-bias), và tốc độ học chậm trong thực tế.

Quy hoạch động (dynamic programming - DP) thì mang đặc tính ngược lại — các cập nhật tựa trên bootstrapping với phương sai bằng 0 — nhưng lại yêu cầu một mô hình toán học đã biết hoàn toàn.

Học theo phương pháp hiệu số thời gian (Temporal Difference - TD learning) kết hợp hoàn hảo ưu điểm của cả hai bên. Từ một bước chuyển duy nhất $(s, a, r, s')$, thuật toán tạo ra một mục tiêu một bước (one-step target) $r + \gamma V(s')$ và dịch chuyển dần $V(s)$ về phía mục tiêu đó. Không cần mô hình. Không cần các tập trải nghiệm hoàn chỉnh. Việc dùng một giá trị xấp xỉ $V$ ở vế phải sẽ tạo ra độ chệch nhất định, nhưng phương sai giảm đi đáng kể so với MC và các cập nhật diễn ra trực tuyến (online) ngay từ bước đầu tiên.

Đây chính là trục xoay mà toàn bộ học tăng cường hiện đại — DQN, A2C, PPO, SAC — xoay quanh. Phần còn lại của Phase 9 là các lớp xấp xỉ hàm (function approximation) và các kỹ thuật tối ưu hóa được xây dựng trực tiếp trên bước cập nhật TD một bước mà bạn sẽ tự tay lập trình trong bài học này.

## Khái niệm cốt lõi (The Concept)

![So sánh Q-learning và SARSA: hàm max ngoài chính sách so với Q(s', a') theo chính sách](../assets/td.svg)

**Bước cập nhật TD(0) cho giá trị trạng thái V:**

```math
V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]
```

Đại lượng trong dấu ngoặc vuông chính là sai số TD (TD error) $\delta = r + \gamma V(s') - V(s)$. Nó là phiên bản trực tuyến tương đương với $G_t - V(s_t)$ trong MC. Điều kiện hội tụ đòi hỏi tốc độ học $\alpha$ thỏa mãn điều kiện Robbins-Monro ($\sum \alpha = \infty$, $\sum \alpha^2 < \infty$) và mọi trạng thái đều được ghé thăm vô hạn lần.

**Q-learning.** Phương pháp TD ngoài chính sách (off-policy) dành cho tác vụ điều khiển:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
```

Toán tử `max` giả định rằng một chính sách *tham lam (greedy)* sẽ được tuân theo từ $s'$ trở đi, bất kể hành động thực tế tiếp theo mà tác tử thực hiện là gì. Sự tách rời đó giúp Q-learning học được giá trị tối ưu $Q^{\ast}$ trong khi tác tử vẫn đang thăm dò môi trường thông qua chính sách ε-greedy. Mnih et al. (2015) đã biến ý tưởng này thành Deep Q-learning trên tập trò chơi Atari (Bài 05).

**SARSA.** Phương pháp TD theo chính sách (on-policy):

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
```

Tên gọi của thuật toán bắt nguồn từ bộ năm thông tin $(s, a, r, s', a')$. SARSA sử dụng chính hành động $a'$ mà tác tử *thực sự* chọn ở bước tiếp theo, thay vì lấy giá trị tham lam `argmax`. Thuật toán hội tụ về $Q^\pi$ cho chính sách ε-greedy $\pi$ đang vận hành; khi $\epsilon \to 0$, giá trị này sẽ trở thành $Q^{\ast}$.

**Sự khác biệt trong bài toán Vách đá (Cliff-walking).** Trong tác vụ kinh điển đi men theo vách đá (rơi xuống vực = phạt -100), Q-learning tìm ra đường đi tối ưu sát mép vực nhưng thỉnh thoảng sẽ bị phạt nặng trong giai đoạn thăm dò. SARSA tìm ra một con đường an toàn hơn, lùi xa vách đá một bước vì nó tính cả nhiễu thăm dò vào giá trị Q của mình. Khi quá trình huấn luyện hoàn tất và $\epsilon \to 0$, cả hai đều đạt tới đường đi tối ưu. Trong triển khai thực tế, điều này rất quan trọng: khi hành vi thăm dò vẫn diễn ra trong môi trường thực tế, SARSA thể hiện tính thận trọng và an toàn hơn hẳn.

**SARSA kỳ vọng (Expected SARSA).** Thay thế $Q(s', a')$ bằng giá trị kỳ vọng của nó dưới chính sách $\pi$:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s, a)]
```

Phương sai thấp hơn SARSA (vì không phải lấy mẫu ngẫu nhiên $a'$), trong khi vẫn giữ nguyên mục tiêu theo chính sách. Đây thường là lựa chọn mặc định trong các sách giáo trình hiện đại.

**TD $n$-bước và TD(λ).** Nội suy giữa TD(0) và MC bằng cách chờ $n$ bước trước khi thực hiện bootstrapping. $n=1$ tương đương với TD thuần túy, $n=\infty$ tương đương với MC. TD(λ) lấy trung bình trên mọi bước $n$ với trọng số suy giảm hình học $(1-\lambda)\lambda^{n-1}$. Hầu hết các mô hình học tăng cường sâu hiện nay chọn $n$ từ 3 đến 20.

```figure
qlearning-gridworld
```

## Tự xây dựng (Build It)

### Bước 1: SARSA với chính sách ε-greedy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Chỉ với tám dòng mã. Điểm khác biệt *duy nhất* so với Q-learning nằm ở dòng tính toán `target`.

### Bước 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

Hàm `max` phân tách mục tiêu học tập ra khỏi hành vi thực tế. Một ký hiệu toán học duy nhất này định hình ranh giới giữa on-policy và off-policy.

### Bước 3: Đường cong học tập (Learning curves)

Theo dõi lợi nhuận trung bình sau mỗi 100 tập trải nghiệm. Q-learning hội tụ nhanh hơn trên môi trường GridWorld tất định đơn giản; SARSA thận trọng hơn nhiều trong bài toán vách đá. Trên lưới GridWorld 4×4 trong `code/main.py`, cả hai phương pháp đều tiệm cận mức tối ưu sau khoảng 2.000 tập với $\alpha=0.1, \epsilon=0.1$.

### Bước 4: So sánh với giá trị chuẩn từ DP

Chạy thuật toán lặp giá trị (Bài 02) để lấy $Q^{\ast}$ chuẩn xác. Đo lường sai số $\max_{s,a} \lvert Q_{\text{learned}}(s,a) - Q^{\ast}(s,a) \rvert$. Một tác tử TD dạng bảng huấn luyện chuẩn xác sẽ đạt độ sai lệch trong khoảng `~0.5` trên lưới 4×4 sau 10.000 tập.

## Cạm bẫy thường gặp (Pitfalls)

- **Giá trị khởi tạo của Q mang tính quyết định.** Khởi tạo lạc quan (optimistic initialization, ví dụ đặt $Q = 0$ cho bài toán có phần thưởng luôn âm) thúc đẩy việc khám phá mạnh mẽ. Khởi tạo bi quan có thể khiến một chính sách tham lam bị mắc kẹt vĩnh viễn trong hành vi cục bộ.
- **Lịch trình điều chỉnh α.** Giữ $\alpha$ cố định là phù hợp cho các bài toán phi dừng. Giảm dần theo $\alpha_n = 1/n$ đảm bảo hội tụ về mặt lý thuyết nhưng quá chậm trong thực tế — hãy giữ $\alpha$ trong khoảng `[0.05, 0.3]` và giám sát chặt chẽ đường cong học tập.
- **Lịch trình suy giảm ε.** Bắt đầu ở mức cao ($\epsilon=1.0$), sau đó giảm dần về $\epsilon=0.05$. Điều kiện hội tụ chính thức là "GLIE" (Greedy in the Limit with Infinite Exploration - tham lam ở giới hạn cùng sự thăm dò vô hạn).
- **Thiên kiến cực đại hóa (Maximization bias) trong Q-learning.** Toán tử `max` luôn bị chệch hướng lên trên (upward-biased) khi các ước tính $Q$ bị nhiễu, dẫn đến việc ước lượng giá trị quá cao. Thuật toán Double Q-learning của Hasselt (được dùng trong DDQN ở Bài 05) giải quyết vấn đề này bằng cách sử dụng hai bảng Q độc lập.
- **Các tập trải nghiệm không hồi kết.** TD có thể học mà không cần trạng thái kết thúc, nhưng bạn cần giới hạn số bước tối đa hoặc xử lý bước bootstrap tại mốc giới hạn một cách chính xác (thông thường: coi mốc cắt là phi kết thúc và tiếp tục bootstrap).
- **Băm trạng thái (State hashing).** Nếu trạng thái là các bộ tuple hoặc tensor, hãy dùng khóa băm được (dùng tuple thay vì list; làm tròn các số thực dấu phẩy động thay vì để giá trị thô).

## Ứng dụng thực tế (Use It)

Bức tranh ứng dụng của các phương pháp TD trong năm 2026:

| Tác vụ | Phương pháp | Lý do |
|--------|-------------|-------|
| Môi trường dạng bảng nhỏ | Q-learning | Học trực tiếp chính sách tối ưu. |
| Môi trường yêu cầu an toàn, on-policy | SARSA / Expected SARSA | Rất thận trọng trong quá trình thăm dò. |
| Không gian trạng thái số chiều cao | DQN (Phase 9 · 05) | Hàm Q mạng nơ-ron kết hợp bộ đệm phát lại và mạng đích (target network). |
| Không gian hành động liên tục | SAC / TD3 (Phase 9 · 07) | Cập nhật TD trên mạng Q; một mạng chính sách riêng biệt xuất ra hành động. |
| RL cho LLM (dựa trên mô hình phần thưởng) | PPO / GRPO (Phase 9 · 08, 12) | Mô hình Actor-critic với lợi thế kiểu TD tính qua GAE. |
| RL ngoại tuyến (Offline RL) | CQL / IQL (Phase 9 · 08) | Q-learning kết hợp với điều chuẩn bảo thủ (conservative regularization). |

Chín mươi phần trăm các nghiên cứu "RL" bạn đọc trong năm 2026 đều là các dạng phát triển nâng cao của Q-learning hoặc SARSA. Hãy nắm vững từng dòng mã của bước cập nhật dạng bảng trước khi bước vào các kiến trúc phức tạp hơn.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Cài đặt Q-learning và SARSA trên GridWorld 4×4. Vẽ đường cong học tập (lợi nhuận trung bình sau mỗi 100 tập) trong 2.000 tập. Phương pháp nào hội tụ nhanh hơn?
2. **Trung bình.** Xây dựng môi trường vách đá Cliff-walking (kích thước 4×12, hàng dưới cùng là vách đá với phần thưởng phạt -100 và quay về vị trí xuất phát). So sánh chính sách cuối cùng của Q-learning và SARSA. Chụp lại đường đi của từng phương pháp. Con đường nào nằm sát mép vực hơn?
3. **Khó.** Triển khai Double Q-learning. Trên một môi trường GridWorld có phần thưởng bị nhiễu (thêm nhiễu Gauss $\sigma=5$ vào phần thưởng mỗi bước), chứng minh rằng Q-learning ước lượng quá mức $V^{\ast}(0,0)$ trong khi Double Q-learning thì không.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Sai số TD (TD error) | "Tín hiệu cập nhật" | $\delta = r + \gamma V(s') - V(s)$, phần dư của ước tính tự cập nhật. |
| TD(0) | "TD một bước" | Cập nhật sau mỗi bước chuyển trạng thái, chỉ dùng ước tính giá trị của trạng thái liền kề tiếp theo. |
| Q-learning | "Nhập môn RL ngoài chính sách" | Bước cập nhật TD với toán tử `max` trên các hành động kế tiếp; học $Q^{\ast}$ độc lập với chính sách hành vi. |
| SARSA | "Q-learning theo chính sách" | Bước cập nhật TD dùng chính hành động tiếp theo được chọn; học $Q^\pi$ cho chính sách ε-greedy $\pi$ hiện thời. |
| Expected SARSA | "SARSA phương sai thấp" | Thay thế hành động lấy mẫu $a'$ bằng giá trị kỳ vọng của nó dưới phân phối của chính sách $\pi$. |
| GLIE | "Lịch trình thăm dò chuẩn mực" | Greedy in the Limit with Infinite Exploration (Tham lam ở giới hạn cùng sự thăm dò vô hạn); điều kiện bắt buộc để Q-learning hội tụ. |
| Tự cập nhật (Bootstrapping) | "Dùng ước tính hiện tại làm mục tiêu" | Đặc điểm khu biệt giữa TD và MC. Nguồn gốc sinh ra độ chệch nhưng giúp giảm phương sai đột phá. |
| Thiên kiến cực đại hóa (Maximization bias) | "Q-learning ước lượng quá cao" | Toán tử `max` trên các ước lượng bị nhiễu sẽ bị chệch lên phía trên; được khắc phục bằng Double Q-learning. |

## Tài liệu đọc thêm (Further Reading)

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — Bài báo gốc và chứng minh hội tụ của thuật toán Q-learning.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — Trình bày toàn diện về TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — Giải pháp đột phá khắc phục thiên kiến cực đại hóa.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Cơ sở lý thuyết và thực nghiệm cho Expected SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — Bài báo khai sinh ra thuật toán SARSA (khi đó được gọi là "modified connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — Khái quát hóa TD(0) thành TD(n), bước chuyển tiếp từ Q-learning sang vết đủ điều kiện (eligibility traces) và sau này là GAE trong PPO.
