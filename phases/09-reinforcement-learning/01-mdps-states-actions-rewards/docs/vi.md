# MDP: Trạng thái, Hành động & Phần thưởng

> Một Quá trình quyết định Markov (Markov Decision Process - MDP) được cấu thành từ năm yếu tố: trạng thái (states), hành động (actions), xác suất chuyển trạng thái (transitions), phần thưởng (rewards), và hệ số chiết khấu (discount). Mọi thuật toán trong học tăng cường (Reinforcement Learning - RL) — Q-learning, PPO, DPO, GRPO — đều tối ưu hóa dựa trên cấu trúc này. Hãy nắm vững nó một lần, và bạn sẽ thông suốt toàn bộ phần còn lại của học tăng cường.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Xác suất & Phân phối), Phase 2 · 01 (Phân loại ML)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Bạn đang lập trình một bot chơi cờ vua. Hoặc một hệ thống lập kế hoạch tồn kho. Hoặc một tác tử giao dịch tài chính (trading agent). Hay vòng lặp PPO dùng để huấn luyện mô hình suy luận (reasoning model). Bốn lĩnh vực hoàn toàn khác nhau, nhưng lại quy tụ về một sự thật đáng kinh ngạc: cả bốn đều thu gọn về cùng một đối tượng toán học.

Học có giám sát (supervised learning) cung cấp cho bạn các cặp dữ liệu `(x, y)` và yêu cầu bạn khớp một hàm số. Học tăng cường không cho bạn bất kỳ nhãn nào — chỉ có một luồng các trạng thái, các hành động bạn đã thực hiện và một phần thưởng vô hướng (scalar reward). Nước đi đó có giúp thắng ván cờ không? Quyết định nhập hàng có tiết kiệm được chi phí không? Giao dịch đó có sinh lời không? Token mà LLM vừa tạo ra có mang lại phần thưởng cao hơn từ mô hình đánh giá (judge) không?

Bạn không thể học từ luồng dữ liệu này cho đến khi hình thức hóa nó. "Những gì tôi đã thấy", "những gì tôi đã làm", "điều gì xảy ra tiếp theo", "kết quả đó tốt đến mức nào" — mỗi yếu tố này đều phải trở thành một đối tượng toán học mà bạn có thể lập luận và tính toán. Sự hình thức hóa đó chính là Quá trình quyết định Markov (Markov Decision Process - MDP). Mọi thuật toán RL trong giai đoạn này, bao gồm cả các vòng lặp RLHF và GRPO ở phần cuối, đều tối ưu hóa trên cấu trúc nền tảng này.

## Khái niệm cốt lõi (The Concept)

![Quá trình quyết định Markov: trạng thái, hành động, chuyển trạng thái, phần thưởng, hệ số chiết khấu](../assets/mdp.svg)

**Năm đối tượng cốt lõi.**

- **Trạng thái (States)** $S$. Mọi thông tin mà tác tử (agent) cần để đưa ra quyết định. Trong GridWorld, đó là tọa độ ô cờ. Trong cờ vua, đó là cấu hình bàn cờ. Trong một LLM, đó là cửa sổ ngữ cảnh (context window) cộng với bất kỳ bộ nhớ nào.
- **Hành động (Actions)** $A$. Các lựa chọn khả dĩ. Di chuyển lên/xuống/trái/phải (up/down/left/right). Đi một nước cờ. Xuất ra một token.
- **Chuyển trạng thái (Transitions)** $P(s' \mid s, a)$. Với trạng thái $s$ và hành động $a$, đây là phân phối xác suất trên trạng thái kế tiếp $s'$. Mang tính tất định (deterministic) trong cờ vua, mang tính ngẫu nhiên (stochastic) trong quản lý kho vận, và gần như tất định trong quá trình giải mã (decoding) của LLM.
- **Phần thưởng (Rewards)** $R(s, a, s')$. Tín hiệu vô hướng phản hồi về hành động. Thắng = +1, thua = -1. Doanh thu trừ chi phí. Số hạng tỉ số hợp lý logarit (log-likelihood ratio) trong GRPO.
- **Hệ số chiết khấu (Discount)** $\gamma \in [0, 1)$. Mức độ coi trọng phần thưởng trong tương lai so với phần thưởng ở hiện tại. $\gamma = 0.99$ tương ứng với tầm nhìn hiệu dụng (horizon) khoảng ~100 bước; $\gamma = 0.9$ tương ứng khoảng ~10 bước.

**Tính chất Markov (The Markov property)** $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t)$. Tương lai chỉ phụ thuộc vào trạng thái hiện tại. Nếu không thỏa mãn điều này, biểu diễn trạng thái đang bị thiếu thông tin — đây không phải là hạn chế của phương pháp, mà là sự thiếu sót trong thiết kế không gian trạng thái.

**Chính sách và Lợi nhuận (Policies and returns).** Một chính sách $\pi(a \mid s)$ ánh xạ các trạng thái sang một phân phối hành động. Lợi nhuận (return) $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ là tổng chiết khấu của các phần thưởng trong tương lai. Giá trị trạng thái (state value) $V^\pi(s) = \mathbb{E}[G_t \mid s_t = s]$ là lợi nhuận kỳ vọng xuất phát từ $s$ khi tuân theo chính sách $\pi$. Giá trị hành động (action value hay Q-value) $Q^\pi(s, a) = \mathbb{E}[G_t \mid s_t = s, a_t = a]$ là lợi nhuận kỳ vọng xuất phát từ $s$ với hành động khởi đầu cụ thể $a$. Mọi thuật toán RL đều ước tính một trong hai đại lượng này, sau đó cải thiện $\pi$ tương ứng.

**Phương trình Bellman (The Bellman equations).** Các phương trình điểm bất động (fixed-point equations) mà mọi bài học trong giai đoạn này đều sử dụng:

```math
\begin{aligned}
V^\pi(s) &= \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) [r + \gamma V^\pi(s')] \\
Q^\pi(s, a) &= \sum_{s', r} P(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')]
\end{aligned}
```

Các phương trình này phân tách lợi nhuận kỳ vọng thành "phần thưởng của bước này" cộng với "giá trị chiết khấu của trạng thái bạn đặt chân tới". Mang tính đệ quy. Mọi thuật toán trong Phase 9 đều lặp lại phương trình này tới khi hội tụ (quy hoạch động - dynamic programming), lấy mẫu từ nó (Monte Carlo), hoặc cập nhật từng bước một qua kỹ thuật bootstrapping (hiệu số thời gian - temporal difference).

```figure
discount-horizon
```

## Tự xây dựng (Build It)

### Bước 1: Một MDP tất định kích thước nhỏ

Một lưới GridWorld kích thước 4×4. Tác tử bắt đầu ở góc trên cùng bên trái, trạng thái kết thúc (terminal) ở góc dưới cùng bên phải, phần thưởng phạt là -1 cho mỗi bước di chuyển, các hành động gồm `{up, down, left, right}`. Xem chi tiết tại `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Chỉ với năm dòng lệnh, bạn đã có toàn bộ môi trường. Quá trình chuyển trạng thái mang tính tất định, hình phạt bước đi là hằng số, và trạng thái kết thúc hấp thu (absorbing terminal state).

### Bước 2: Triển khai thử nghiệm một chính sách (Roll out a policy)

Một chính sách là một hàm ánh xạ từ trạng thái sang phân phối hành động. Dạng đơn giản nhất: phân phối ngẫu nhiên đều (uniform random).

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Chạy chính sách ngẫu nhiên này 1000 lần. Lợi nhuận trung bình rơi vào khoảng -60 đến -80 trên bàn cờ 4×4 này. Lợi nhuận tối ưu là -6 (đường đi thẳng chéo xuống góc phải). Thu hẹp khoảng cách đó chính là toàn bộ mục tiêu của Phase 9.

### Bước 3: Tính toán chính xác `V^π` thông qua phương trình Bellman

Đối với các MDP kích thước nhỏ, phương trình Bellman là một hệ phương trình tuyến tính. Liệt kê toàn bộ các trạng thái, áp dụng kỳ vọng, và lặp lại cho đến khi các giá trị dừng biến thiên.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Đây chính là thuật toán đánh giá chính sách lặp (iterative policy evaluation). Nó là thuật toán đầu tiên trong cuốn sách kinh điển của Sutton & Barto và là nền tảng lý thuyết cho mọi phương pháp RL kế tiếp.

### Bước 4: `γ` là một siêu tham số mang ý nghĩa vật lý

Tầm nhìn hiệu dụng (effective horizon) xấp xỉ khoảng $1 / (1 - \gamma)$. $\gamma = 0.9$ → 10 bước. $\gamma = 0.99$ → 100 bước. $\gamma = 0.999$ → 1000 bước.

Nếu đặt $\gamma$ quá thấp, tác tử sẽ hành động thiển cận (myopic). Nếu đặt quá cao, việc phân bổ đóng góp (credit assignment) sẽ trở nên nhiễu loạn, bởi vì rất nhiều bước đi ban đầu phải cùng chia sẻ trách nhiệm cho phần thưởng ở tương lai xa. RLHF trong LLM thường dùng $\gamma = 1$ vì các tập dữ liệu (episodes) ngắn và có giới hạn rõ ràng. Các tác vụ điều khiển thường dùng `0.95–0.99`. Các trò chơi chiến lược có tầm nhìn dài sử dụng `0.999`.

## Cạm bẫy thường gặp (Pitfalls)

- **Trạng thái phi Markov (Non-Markovian state).** Nếu bạn cần ba quan sát gần nhất mới có thể đưa ra quyết định, thì "trạng thái" không chỉ đơn thuần là quan sát hiện tại. Giải pháp: xếp chồng các khung hình (frame-stacking, DQN trên Atari xếp chồng 4 khung hình) hoặc sử dụng trạng thái hồi quy (recurrent state như LSTM/GRU trên chuỗi quan sát).
- **Phần thưởng thưa thớt (Sparse rewards).** Phần thưởng chỉ xuất hiện khi thắng cuộc khiến việc học gần như bất khả thi trong không gian trạng thái lớn. Hãy định hình phần thưởng (reward shaping - tín hiệu trung gian) hoặc khởi động bằng học bắt chước (imitation learning - Phase 9 · 09).
- **Gian lận phần thưởng (Reward hacking).** Việc tối ưu hóa một phần thưởng đại diện (proxy reward) thường dẫn đến hành vi dị thường. Tác tử đua thuyền của OpenAI đã xoay vòng tròn liên tục để thu thập vật phẩm nâng cấp thay vì hoàn thành đường đua. Luôn xác định phần thưởng dựa trên kết quả mục tiêu cuối cùng, không dựa trên các chỉ số đại diện.
- **Xác định sai hệ số chiết khấu.** Đặt $\gamma = 1$ cho tác vụ có tầm nhìn vô hạn (infinite-horizon task) sẽ khiến mọi giá trị trở nên vô cực. Luôn chặn bằng một giới hạn hữu hạn (finite horizon) hoặc $\gamma < 1$.
- **Thang đo phần thưởng (Reward scale).** Phần thưởng {+100, -100} so với {+1, -1} tạo ra các chính sách tối ưu giống hệt nhau nhưng độ lớn gradient lại hoàn toàn khác biệt. Hãy chuẩn hóa về khoảng `[-1, 1]` trước khi đưa vào PPO/DQN.

## Ứng dụng thực tế (Use It)

Ngăn xếp công nghệ năm 2026 quy đổi mọi pipeline RL về một MDP trước khi chạm vào mã nguồn:

| Tình huống | Trạng thái (State) | Hành động (Action) | Phần thưởng (Reward) | γ |
|-----------|--------------------|--------------------|----------------------|---|
| Điều khiển (vận động, thao tác) | Góc khớp + vận tốc | Mô-men xoắn liên tục | Được định hình theo tác vụ cụ thể | 0.99 |
| Trò chơi (cờ vua, cờ vây, poker) | Bàn cờ + lịch sử | Nước đi hợp lệ | Thắng=+1 / Thua=-1 | 1.0 (hữu hạn) |
| Tồn kho / Định giá | Lượng hàng + nhu cầu | Số lượng đặt hàng | Doanh thu - chi phí | 0.95 |
| RLHF cho LLM | Các token ngữ cảnh | Token kế tiếp | Điểm số từ mô hình phần thưởng ở cuối | 1.0 (tập dữ liệu ~200 tokens) |
| GRPO cho mô hình suy luận | Prompt + phản hồi từng phần | Token kế tiếp | Bộ kiểm chứng (verifier) 0/1 ở cuối | 1.0 |

Hãy viết ra bộ năm giá trị này trước khi viết bất kỳ vòng lặp huấn luyện nào. Hầu hết các báo cáo lỗi "RL không hoạt động" đều bắt nguồn từ một công thức MDP bị thiết kế sai ngay trên giấy.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Triển khai GridWorld 4×4 và hàm rollout chính sách ngẫu nhiên trong `code/main.py`. Chạy 10.000 tập (episodes). Báo cáo giá trị trung bình (mean) và độ lệch chuẩn (std) của lợi nhuận. So sánh với lợi nhuận tối ưu (-6).
2. **Trung bình.** Chạy hàm `policy_evaluation` với $\gamma \in \lbrace 0.5, 0.9, 0.99 \rbrace$ cho chính sách ngẫu nhiên đều. In `V` dưới dạng lưới 4×4 cho từng trường hợp. Giải thích tại sao giá trị trạng thái gần ô kết thúc tăng nhanh hơn khi $\gamma$ lớn hơn.
3. **Khó.** Biến GridWorld thành môi trường ngẫu nhiên: mỗi hành động có xác suất trượt sang hướng liền kề với $p = 0.1$. Đánh giá lại chính sách ngẫu nhiên đều. Liệu `V[start]` trở nên tốt hơn hay tệ hơn? Tại sao?

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| MDP | "Thiết lập học tăng cường" | Bộ 5 thành phần $(S, A, P, R, \gamma)$ thỏa mãn tính chất Markov. |
| Trạng thái (State) | "Những gì tác tử nhìn thấy" | Thống kê đủ (sufficient statistic) cho động lực học tương lai dưới lớp chính sách đã chọn. |
| Chính sách (Policy) | "Hành vi của tác tử" | Phân phối có điều kiện $\pi(a \mid s)$ hoặc ánh xạ tất định $s \to a$. |
| Lợi nhuận (Return) | "Tổng phần thưởng" | Tổng chiết khấu $\sum \gamma^t r_t$ tính từ bước hiện tại. |
| Giá trị (Value) | "Mức độ tốt của một trạng thái" | Lợi nhuận kỳ vọng dưới chính sách $\pi$ khi xuất phát từ $s$. |
| Giá trị Q (Q-value) | "Mức độ tốt của một hành động" | Lợi nhuận kỳ vọng dưới chính sách $\pi$ khi xuất phát từ $s$ với hành động đầu tiên là $a$. |
| Phương trình Bellman | "Hệ thức đệ quy quy hoạch động" | Sự phân tách điểm bất động của giá trị trạng thái / giá trị Q thành phần thưởng một bước cộng với giá trị chiết khấu của trạng thái kế tiếp. |
| Hệ số chiết khấu $\gamma$ | "Tương lai so với hiện tại" | Trọng số hình học áp lên phần thưởng ở tương lai xa; tầm nhìn hiệu dụng $\approx 1/(1-\gamma)$. |

## Tài liệu đọc thêm (Further Reading)

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — Cuốn sách giáo khoa kinh điển. Chương 3 trình bày về MDP và các phương trình Bellman; Chương 1 nêu bật giả thuyết phần thưởng (reward hypothesis) làm nền tảng cho mọi bài học tiếp theo.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Nguồn gốc ra đời của phương trình Bellman.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — Tài liệu nhập môn súc tích về MDP dưới góc nhìn học tăng cường sâu (deep RL).
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — Tài liệu tham khảo nghiên cứu vận hành chuyên sâu về MDP và các phương pháp giải chính xác.
- [Littman (1996). Algorithms for Sequential Decision Making (Luận án Tiến sĩ)](https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf) — Khảo sát mạch lạc nhất về MDP như một dạng chuyên biệt hóa của quy hoạch động.
