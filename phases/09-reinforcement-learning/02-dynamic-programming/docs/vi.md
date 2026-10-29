# Quy hoạch động — Lặp chính sách & Lặp giá trị

> Quy hoạch động (dynamic programming) là học tăng cường khi bạn đã biết trước toàn bộ thông tin môi trường. Bạn đã nắm rõ hàm chuyển trạng thái và hàm phần thưởng; công việc còn lại chỉ là lặp phương trình Bellman cho đến khi $V$ hoặc $\pi$ ngừng biến thiên. Đây chính là chuẩn đối sánh (benchmark) mà mọi phương pháp dựa trên lấy mẫu đều nỗ lực tiếp cận.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDP)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Bạn có một MDP với mô hình đã biết hoàn toàn: bạn có thể truy vấn $P(s' \mid s, a)$ và $R(s, a, s')$ cho bất kỳ cặp trạng thái - hành động nào. Người quản lý tồn kho nắm được phân phối nhu cầu. Một trò chơi cờ có các bước chuyển trạng thái tất định. Một môi trường GridWorld chỉ gói gọn trong bốn dòng mã Python. Tức là bạn đang sở hữu một *mô hình* (model).

Học tăng cường không mô hình (Model-free RL như Q-learning, PPO, REINFORCE) được phát minh cho trường hợp bạn không có mô hình — bạn chỉ có thể tương tác và lấy mẫu từ môi trường. Nhưng khi bạn đã có mô hình trong tay, sẽ có các phương pháp nhanh hơn và chuẩn xác hơn nhiều: quy hoạch động (dynamic programming - DP). Richard Bellman đã thiết kế chúng vào năm 1957. Đến nay chúng vẫn định nghĩa tính chuẩn xác: khi người ta nói "chính sách tối ưu cho MDP này", họ đang ám chỉ chính sách mà DP trả về.

Bạn cần đến quy hoạch động vào năm 2026 vì ba lý do. Thứ nhất, mọi môi trường dạng bảng (tabular environment) trong nghiên cứu RL (GridWorld, FrozenLake, CliffWalking) đều được giải bằng DP để tạo ra chính sách chuẩn mực vàng (gold standard). Thứ hai, các giá trị chính xác cho phép bạn *gỡ lỗi (debug)* các phương pháp lấy mẫu: nếu ước tính của Q-learning cho $V^{\ast}(s_0)$ lệch tới 30% so với kết quả giải bằng DP, mã Q-learning của bạn chắc chắn có lỗi. Thứ ba, các phương pháp RL ngoại tuyến (offline RL) và lập kế hoạch hiện đại (MCTS, cơ chế tìm kiếm của AlphaZero, model-based RL ở Phase 9 · 10) đều lặp lại bước cập nhật Bellman (Bellman backup) trên một mô hình đã học hoặc đã cho trước.

## Khái niệm cốt lõi (The Concept)

![Lặp chính sách và lặp giá trị đặt cạnh nhau](../assets/dp.svg)

**Hai thuật toán, đều dựa trên phép lặp điểm bất động của phương trình Bellman.**

**Lặp chính sách (Policy iteration).** Luân phiên thực hiện hai bước cho đến khi chính sách không còn thay đổi.

1. *Đánh giá (Evaluation):* với chính sách $\pi$ hiện tại, tính $V^\pi$ bằng cách áp dụng liên tục công thức $V(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s',r} P(s',r \mid s,a) [r + \gamma V(s')]$ cho đến khi hội tụ.
2. *Cải tiến (Improvement):* với $V^\pi$ vừa tính được, cập nhật $\pi$ theo hướng tham lam (greedy) đối với $V^\pi$: $\pi(s) \leftarrow \arg\max_a \sum_{s',r} P(s',r \mid s,a) [r + \gamma V(s')]$.

Sự hội tụ được đảm bảo về mặt lý thuyết vì (a) mỗi bước cải tiến hoặc giữ nguyên $\pi$ hoặc làm tăng ngặt $V^\pi$ ở một số trạng thái, (b) không gian các chính sách tất định là hữu hạn. Thuật toán thường hội tụ chỉ sau ~5–20 vòng lặp ngoài ngay cả với không gian trạng thái lớn.

**Lặp giá trị (Value iteration).** Gộp hai bước đánh giá và cải tiến vào một lượt quét duy nhất. Áp dụng phương trình Bellman *tối ưu*:

```math
V(s) \leftarrow \max_a \sum_{s',r} P(s',r \mid s,a) [r + \gamma V(s')]
```

Lặp lại cho đến khi $\max_s \lvert V_{\text{new}}(s) - V(s) \rvert < \epsilon$. Trích xuất chính sách ở bước cuối cùng bằng cách chọn hành động tham lam nhất. Thuật toán chạy nhanh hơn đáng kể ở mỗi vòng lặp — vì không có vòng lặp đánh giá con bên trong — nhưng thông thường cần nhiều lượt lặp hơn để hội tụ.

**Lặp chính sách tổng quát (Generalized Policy Iteration - GPI).** Khung tư duy hợp nhất toàn bộ lĩnh vực. Hàm giá trị và chính sách bị khóa chặt trong một vòng lặp tương hỗ; bất kỳ phương pháp nào thúc đẩy cả hai tiến tới sự nhất quán lẫn nhau (lặp giá trị bất đồng bộ, lặp chính sách biến đổi, Q-learning, actor-critic, PPO) đều là một trường hợp cụ thể của GPI.

**Tại sao điều kiện $\gamma < 1$ lại quan trọng.** Toán tử Bellman là một ánh xạ co $\gamma$ ($\gamma$-contraction) theo chuẩn vô cùng (sup-norm): $\lVert T V - T V' \rVert_\infty \le \gamma \lVert V - V' \rVert_\infty$. Tính chất co đảm bảo tồn tại duy nhất một điểm bất động và tốc độ hội tụ theo cấp số nhân (geometric convergence). Nếu bỏ điều kiện $\gamma < 1$, bạn sẽ mất đi sự đảm bảo này — khi đó bạn bắt buộc phải có một tầm nhìn hữu hạn hoặc một trạng thái kết thúc hấp thu.

```figure
value-iteration-gamma
```

## Tự xây dựng (Build It)

### Bước 1: Xây dựng mô hình MDP GridWorld

Sử dụng cùng môi trường GridWorld 4×4 từ Bài 01. Chúng ta bổ sung một biến thể ngẫu nhiên: với xác suất `0.1`, tác tử bị trượt sang một hướng vuông góc ngẫu nhiên.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

Hàm `transitions(s, a)` trả về một danh sách các bộ `(s', r, p)`. Đây chính là toàn bộ mô hình môi trường.

### Bước 2: Đánh giá chính sách (Policy evaluation)

Với một chính sách `π(s) = {action: prob}`, lặp phương trình Bellman cho đến khi $V$ ngừng biến thiên:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Bước 3: Cải tiến chính sách (Policy improvement)

Thay thế $\pi$ bằng chính sách tham lam đối với $V$. Nếu $\pi$ không thay đổi, thuật toán dừng lại — chúng ta đã đạt tới điểm tối ưu.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Bước 4: Kết hợp thành thuật toán Lặp chính sách hoàn chỉnh

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # khởi tạo tùy ý
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Số vòng lặp ngoài điển hình để hội tụ trên lưới 4×4: 4–6 vòng. Thuật toán xuất ra $V^{\ast}(0,0) \approx -6$ và một chính sách giúp giảm thiểu số bước đi một cách chặt chẽ.

### Bước 5: Lặp giá trị (Value iteration - phiên bản một vòng lặp)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Cùng đạt tới một điểm bất động, nhưng số dòng mã ít hơn đáng kể.

## Cạm bẫy thường gặp (Pitfalls)

- **Quên xử lý trạng thái kết thúc (terminals).** Nếu bạn áp dụng phương trình Bellman vào một trạng thái hấp thu, nó vẫn sẽ chọn ra một "hành động tốt nhất" dù hành động đó không thay đổi được gì. Hãy chặn lại bằng điều kiện `if s == terminal: V[s] = 0`.
- **Hội tụ theo chuẩn vô cùng (sup-norm) so với chuẩn L2.** Phải dùng `max |V_new - V|`, không dùng trung bình sai số. Đảm bảo lý thuyết chỉ đúng trên chuẩn vô cùng.
- **Cập nhật tại chỗ (in-place) so với cập nhật đồng bộ (synchronous).** Việc cập nhật `V[s]` tại chỗ (phương pháp Gauss-Seidel) hội tụ nhanh hơn việc dùng một từ điển `V_new` riêng biệt (phương pháp Jacobi). Mã nguồn chạy thực tế luôn ưu tiên cập nhật tại chỗ.
- **Xử lý thế hòa giữa các hành động (policy ties).** Nếu hai hành động có cùng giá trị Q, hàm `argmax` có thể giải quyết thế hòa khác nhau ở mỗi vòng lặp, khiến điều kiện kiểm tra "chính sách ổn định" bị dao động không dứt. Hãy dùng cơ chế gỡ hòa ổn định (chọn hành động đầu tiên theo thứ tự cố định).
- **Bùng nổ không gian trạng thái.** Quy hoạch động có độ phức tạp $O(\lvert S \rvert \cdot \lvert A \rvert)$ cho mỗi lượt quét. Nó chỉ hoạt động hiệu quả tới quy mô ~10⁷ trạng thái. Vượt qua ngưỡng đó, bạn bắt buộc phải dùng xấp xỉ hàm (function approximation - từ Bài 05 trở đi).

## Ứng dụng thực tế (Use It)

Năm 2026, quy hoạch động đóng vai trò là đường cơ sở chuẩn xác (correctness baseline) và là vòng lặp bên trong của các bộ lập kế hoạch:

| Tình huống sử dụng | Phương pháp |
|--------------------|-------------|
| Giải chính xác một MDP dạng bảng kích thước nhỏ | Lặp giá trị (đơn giản hơn) hoặc lặp chính sách (ít bước lặp ngoài hơn) |
| Kiểm chứng việc cài đặt Q-learning / PPO | So sánh với giá trị tối ưu $V^{\ast}$ của DP trên một môi trường mẫu |
| RL dựa trên mô hình (Model-based RL - Phase 9 · 10) | Cập nhật Bellman (Bellman backup) trên mô hình chuyển trạng thái đã học |
| Lập kế hoạch trong AlphaZero / MuZero | Tìm kiếm cây Monte Carlo (MCTS) = cập nhật Bellman bất đồng bộ |
| RL ngoại tuyến (Offline RL: CQL, IQL) | Lặp Q bảo thủ (Conservative Q-iteration) — DP có thêm hình phạt cho hành động ngoài phân phối (OOD) |

Mỗi khi ai đó nói "hàm giá trị tối ưu", họ đang ám chỉ "điểm bất động của DP". Khi bạn nhìn thấy ký hiệu $V^{\ast}$ hoặc $Q^{\ast}$ trong một bài báo khoa học, hãy hình dung ngay về vòng lặp này.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy thuật toán lặp giá trị trên GridWorld 4×4 với $\gamma \in \lbrace 0.9, 0.99 \rbrace$. Cần bao nhiêu lượt quét cho đến khi $\max \lvert \Delta V \rvert < 10^{-6}$? In $V^{\ast}$ ra dưới dạng lưới 4×4.
2. **Trung bình.** So sánh lặp chính sách với lặp giá trị trên GridWorld *ngẫu nhiên* (xác suất trượt `0.1`). Thống kê: số lượt quét, thời gian thực tế (wall-clock time), giá trị cuối cùng $V^{\ast}(0,0)$. Thuật toán nào hội tụ nhanh hơn theo số vòng lặp? Thuật toán nào nhanh hơn theo thời gian thực?
3. **Khó.** Xây dựng thuật toán lặp chính sách biến đổi (modified policy iteration): trong bước đánh giá, chỉ chạy $k$ lượt quét thay vì chạy cho đến khi hội tụ hoàn toàn. Vẽ đồ thị sai số của $V^{\ast}(0,0)$ theo $k$ với $k \in \lbrace 1, 2, 5, 10, 50 \rbrace$. Đường cong này cho bạn biết điều gì về sự đánh đổi giữa đánh giá và cải tiến?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Lặp chính sách (Policy iteration) | "Thuật toán quy hoạch động" | Luân phiên đánh giá ($V^\pi$) và cải tiến ($\pi$ tham lam theo $V^\pi$) cho đến khi chính sách không còn thay đổi. |
| Lặp giá trị (Value iteration) | "Quy hoạch động nhanh hơn" | Cập nhật tối ưu Bellman được áp dụng trong một lượt quét duy nhất; hội tụ về $V^{\ast}$ theo cấp số nhân. |
| Toán tử Bellman | "Hệ thức đệ quy" | $(T V)(s) = \max_a \sum P (r + \gamma V(s'))$; là một ánh xạ co $\gamma$ theo chuẩn vô cùng. |
| Ánh xạ co (Contraction) | "Lý do DP hội tụ" | Bất kỳ toán tử $T$ nào thỏa mãn $\lVert T x - T y \rVert \le \gamma \lVert x - y \rVert$ đều sở hữu duy nhất một điểm bất động. |
| GPI | "Mọi thứ đều là DP" | Generalized Policy Iteration: bất kỳ phương pháp nào thúc đẩy $V$ và $\pi$ tiến tới sự nhất quán tương hỗ. |
| Cập nhật đồng bộ (Synchronous update) | "Kiểu Jacobi" | Sử dụng giá trị $V$ cũ trong suốt một lượt quét; dễ phân tích lý thuyết nhưng chậm hơn. |
| Cập nhật tại chỗ (In-place update) | "Kiểu Gauss-Seidel" | Sử dụng giá trị $V$ mới ngay khi nó vừa được cập nhật; hội tụ nhanh hơn trong thực nghiệm. |

## Tài liệu đọc thêm (Further Reading)

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — Trình bày kinh điển về lặp chính sách và lặp giá trị.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) — Phân tích toán học chặt chẽ về các lập luận định lý ánh xạ co.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — Thuật toán lặp chính sách biến đổi và phân tích hội tụ chi tiết.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — Bài báo gốc đặt nền móng cho thuật toán lặp chính sách.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — Cầu nối từ quy hoạch động sang quy hoạch động xấp xỉ / học tăng cường sâu được sử dụng trong mọi bài học kế tiếp.
