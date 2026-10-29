# Học Tăng cường Đa Tác tử (Multi-Agent RL)

> Học tăng cường đơn tác tử (single-agent RL) giả định rằng môi trường là dừng (stationary). Đặt hai tác tử cùng học vào một thế giới và giả định đó lập tức sụp đổ: mỗi tác tử là một phần của môi trường đối với tác tử kia, và cả hai đều liên tục thay đổi. Học tăng cường đa tác tử (Multi-Agent Reinforcement Learning - MARL) là tập hợp các kỹ thuật giúp quá trình học hội tụ khi tính chất Markov không còn được thỏa mãn.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Thời lượng:** ~45 phút

## Vấn đề đặt ra (The Problem)

Một robot học cách di chuyển trong phòng là bài toán RL đơn tác tử. Một đội bóng đá thì không. AlphaStar thi đấu với các đối thủ StarCraft thì không. Một thị trường gồm các tác tử đấu giá thì không. Hai chiếc xe tự hành thương lượng tại ngã tư bốn chiều thì không. Rất nhiều bài toán thực tế có tương tác nhiều-nhiều đều không phải là đơn tác tử.

Trong mọi bài toán đa tác tử, dưới góc nhìn của bất kỳ một tác tử nào, các tác tử khác *chính là* một phần của môi trường. Khi các tác tử khác học hỏi và biến đổi hành vi, môi trường trở nên không dừng (non-stationary). Tính chất Markov — "trạng thái kế tiếp chỉ phụ thuộc vào trạng thái hiện tại và hành động của tôi" — bị vi phạm nghiêm trọng vì trạng thái tiếp theo còn phụ thuộc vào lựa chọn của *những tác tử khác*, và chính sách của họ là những mục tiêu liên tục di chuyển.

Điều này phá vỡ các chứng minh hội tụ dạng bảng (bảo đảm lý thuyết của Q-learning đòi hỏi môi trường phải dừng). Nó cũng làm tê liệt các thuật toán deep RL ngây thơ: các tác tử rượt đuổi nhau trong các vòng lặp vô tận, không bao giờ hội tụ về một chính sách ổn định. Bạn cần các kỹ thuật chuyên biệt cho môi trường đa tác tử: huấn luyện tập trung / thực thi phân tán (centralized training / decentralized execution), đường cơ sở phản thực nghiệm (counterfactual baselines), thi đấu giải đấu (league play), tự đấu với chính mình (self-play).

Các ứng dụng thực tế năm 2026: đàn robot (robot swarms), điều phối giao thông, đội xe tự hành, mô phỏng thị trường tài chính, các hệ thống đa tác tử LLM (Phase 16), và bất kỳ trò chơi nào có nhiều hơn một người chơi thông minh.

## Khái niệm (The Concept)

![Bốn chế độ MARL: độc lập, critic tập trung, tự chơi, huấn luyện theo giải đấu](../assets/marl.svg)

**Mô hình toán học: Trò chơi Markov (Markov Game).** Là sự tổng quát hóa của Quá trình Quyết định Markov (MDP): tập trạng thái $S$, hành động kết hợp $a = (a_1, \ldots, a_n)$, hàm chuyển trạng thái $P(s' \mid s, a)$, và hàm phần thưởng riêng cho từng tác tử $R_i(s, a, s')$. Mỗi tác tử $i$ tìm cách cực đại hóa phần thưởng tích lũy của chính mình theo chính sách riêng $\pi_i$. Nếu phần thưởng của mọi tác tử giống nhau, đây là trò chơi **hợp tác hoàn toàn** (fully cooperative). Nếu tổng phần thưởng bằng 0, đây là trò chơi **đối kháng tổng bằng không** (zero-sum adversarial). Nếu nằm ở khoảng giữa, đây là trò chơi **tổng tổng quát** (general-sum).

**Các thách thức cốt lõi:**

- **Tính không dừng (Non-stationarity).** Xác suất chuyển $P(s' \mid s, a_i)$ dưới góc nhìn của tác tử $i$ phụ thuộc vào $\pi_{-i}$ (chính sách của các tác tử khác), vốn liên tục biến đổi.
- **Quy gán đóng góp (Credit assignment).** Khi phần thưởng được dùng chung, làm thế nào để biết tác tử nào đã đóng góp tạo ra phần thưởng đó?
- **Phối hợp khám phá (Exploration coordination).** Các tác tử phải cùng khám phá các chiến lược bổ trợ lẫn nhau, thay vì khám phá trùng lặp lãng phí trên cùng một trạng thái.
- **Khả năng mở rộng quy mô (Scalability).** Không gian hành động kết hợp bùng nổ theo hàm mũ theo số lượng tác tử $n$.
- **Quan sát một phần (Partial observability).** Mỗi tác tử chỉ nhìn thấy quan sát cục bộ của mình; trạng thái toàn cục thường bị ẩn giấu.

**Bốn chế độ thống trị:**

**1. Q-learning độc lập / PPO độc lập (Independent Q-learning - IQL, Independent PPO - IPPO).** Mỗi tác tử tự học hàm Q hoặc chính sách riêng của mình, xem tất cả các tác tử khác như một phần của môi trường nền. Rất đơn giản, đôi khi hoạt động hiệu quả (đặc biệt khi bộ đệm phát lại kinh nghiệm hoạt động như một cơ chế làm mượt mô hình hóa tác tử). Về mặt lý thuyết: không có chứng minh hội tụ. Trong thực tế: hoạt động tốt cho các tác vụ ghép nối lỏng lẻo (loosely-coupled), nhưng thất bại với các tác vụ đòi hỏi phối hợp chặt chẽ (tightly-coupled).

**2. Huấn luyện tập trung, thực thi phân tán (Centralized Training, Decentralized Execution - CTDE).** Mô hình hiện đại phổ biến nhất. Mỗi tác tử sở hữu một *chính sách* $\pi_i$ riêng nhận đầu vào là quan sát cục bộ $o_i$ — đây là chế độ thực thi phân tán tiêu chuẩn khi triển khai thực tế. Tuy nhiên, trong quá trình *huấn luyện*, một mạng phê bình tập trung $Q(s, a_1, \ldots, a_n)$ nhận toàn bộ trạng thái toàn cục và hành động kết hợp của tất cả các tác tử. Các ví dụ tiêu biểu:
- **MADDPG** (Lowe cùng các cộng sự 2017): thuật toán DDPG với một critic tập trung cho từng tác tử.
- **COMA** (Foerster cùng các cộng sự 2017): sử dụng đường cơ sở phản thực nghiệm (counterfactual baseline) — đặt câu hỏi: "phần thưởng của tôi sẽ là bao nhiêu nếu lúc đó tôi chọn hành động $a'$ thay vì $a$?" — từ đó cô lập được phần đóng góp của riêng tác tử đó.
- **MAPPO** / **IPPO** dùng chung critic (Yu cùng các cộng sự 2022): PPO với hàm giá trị tập trung. Đây là phương pháp thống trị năm 2026 cho MARL hợp tác.
- **QMIX** (Rashid cùng các cộng sự 2018): phân rã giá trị (value decomposition) — $Q_{\text{tot}}(s, a) = f(Q_1(s, a_1), \ldots, Q_n(s, a_n))$ với mạng trộn đơn điệu (monotonic mixing).

**3. Tự đấu với chính mình (Self-play).** Hai bản sao của cùng một tác tử thi đấu với nhau. Chính sách của đối thủ *chính là* bản chụp chính sách của chính tác tử đó trong quá khứ. AlphaGo / AlphaZero / MuZero. OpenAI Five. Phương pháp này hoạt động hiệu quả nhất cho các trò chơi tổng bằng không; tín hiệu huấn luyện có tính đối xứng cao.

**4. Huấn luyện theo giải đấu (League play).** Mở rộng của self-play cho các môi trường tổng tổng quát / đối kháng phức tạp: duy trì một quần thể gồm các chính sách trong quá khứ và hiện tại, lấy mẫu đối thủ từ giải đấu để huấn luyện. Bổ sung thêm các tác tử khai thác (exploiters - chuyên đánh bại tác tử mạnh nhất hiện tại) và tác tử khai thác chính (main exploiters - chuyên đánh bại các exploiter khác). AlphaStar (StarCraft II). Cần thiết khi trò chơi xuất hiện các chu kỳ chiến thuật kiểu "kéo-búa-bao".

**Giao tiếp (Communication).** Cho phép các tác tử gửi các thông điệp học được $m_i$ cho nhau. Rất hiệu quả trong các môi trường hợp tác. Foerster cùng các cộng sự (2016) chứng minh rằng giao tiếp có thể khả vi giữa các tác tử và được huấn luyện đầu-cuối (end-to-end). Ngày nay, các hệ thống đa tác tử dựa trên LLM (Phase 16) về cơ bản giao tiếp trực tiếp bằng ngôn ngữ tự nhiên.

```figure
f3-marl-orbit
```

## Thực hành Xây dựng (Build It)

Bài học này xây dựng một môi trường GridWorld 6×6 với hai tác tử hợp tác. Chúng bắt đầu ở hai góc đối diện và phải cùng đến một ô mục tiêu chung. Phần thưởng chung: `-1` cho mỗi bước khi vẫn còn ít nhất một tác tử đang di chuyển, `+10` khi cả hai cùng tới đích. Xem thêm tại `code/main.py`.

### Bước 1: Môi trường đa tác tử (the multi-agent env)

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # hai tác tử

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

Không gian hành động *kết hợp* (joint action space) là `|A|² = 16`. Trạng thái toàn cục gồm hai tọa độ vị trí.

### Bước 2: Q-learning độc lập (Independent Q-learning)

Mỗi tác tử quản lý bảng Q riêng theo trạng thái kết hợp. Tại mỗi bước: cả hai tác tử chọn hành động theo chiến lược $\epsilon$-greedy, thu thập bước chuyển trạng thái chung, và mỗi bên tự cập nhật bảng Q của mình theo phần thưởng chung.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            if done:
                Q1[s][a1] += alpha * (r - Q1[s][a1])
                Q2[s][a2] += alpha * (r - Q2[s][a2])
                break
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Cách này hoạt động trên tác vụ hiện tại vì phần thưởng dày đặc (dense) và mục tiêu đồng nhất. Tuy nhiên, nó sẽ thất bại trên các tác vụ đòi hỏi đồng bộ chặt chẽ (ví dụ: khi một tác tử bắt buộc phải *chờ* tác tử kia).

### Bước 3: Hàm Q tập trung với cập nhật phân rã giá trị

Sử dụng một hàm Q duy nhất trên không gian hành động kết hợp $Q(s, a_1, a_2)$. Cập nhật từ phần thưởng chung. Phân tán hóa khi thực thi bằng cách lấy cận trên (marginalize): $\pi_i(s) = \arg\max_{a_i} \max_{a_{-i}} Q(s, a_1, a_2)$. Đánh đổi không gian hành động kết hợp tăng theo hàm mũ để lấy một góc nhìn toàn cục *chính xác*.

### Bước 4: Tự chơi đơn giản (Self-play đối kháng 2 tác tử)

Cùng một tác tử, đảm nhận hai vai trò. Huấn luyện tác tử A thi đấu với tác tử B; sau $K$ tập, sao chép toàn bộ trọng số của A sang B. Huấn luyện đối xứng, tiến bộ đều đặn. Đây là công thức của AlphaZero thu nhỏ.

## Bẫy thường gặp (Pitfalls)

- **Bộ đệm phát lại không dừng (Non-stationary replay).** Bộ đệm kinh nghiệm khi dùng cho các tác tử độc lập còn tệ hơn đơn tác tử, bởi vì các bước chuyển trong quá khứ được sinh ra bởi các đối thủ nay đã lỗi thời. Khắc phục: dán lại nhãn (relabel) hoặc gán trọng số theo độ mới của dữ liệu.
- **Sự nhập nhằng trong quy gán đóng góp (Credit assignment ambiguity).** Nhận một phần thưởng chung sau một tập dài mà không có cách nào rõ ràng để phân định tác tử nào đã đóng góp. Khắc phục: dùng đường cơ sở phản thực nghiệm (COMA), hoặc định hình phần thưởng (reward shaping) riêng cho từng tác tử.
- **Trôi dạt chính sách / Rượt đuổi lẫn nhau (Policy drift / chasing).** Phản ứng tối ưu của mỗi tác tử liên tục thay đổi sau mỗi lần cập nhật của các tác tử khác. Khắc phục: dùng mạng critic tập trung, giảm tốc độ học, hoặc đóng băng luân phiên từng tác tử.
- **Lạm dụng phần thưởng qua thông đồng (Reward hacking via coordination).** Các tác tử tìm ra các chiêu trò phối hợp trục lợi mà người thiết kế không lường trước. Ví dụ: các tác tử đấu giá đồng loạt trả giá bằng 0. Khắc phục: thiết kế phần thưởng cẩn trọng, bổ sung các ràng buộc hành vi.
- **Khám phá trùng lặp (Exploration redundancy).** Cả hai tác tử cùng khám phá một cặp trạng thái - hành động giống nhau. Khắc phục: bổ sung số hạng thưởng entropy riêng cho từng tác tử, hoặc điều kiện hóa theo vai trò (role-conditioning).
- **Chu kỳ luẩn quẩn trong giải đấu (League cycles).** Tự đấu thuần túy có thể mắc kẹt trong chu kỳ khắc chế vòng tròn. Khắc phục: áp dụng huấn luyện theo giải đấu với quần thể đối thủ đa dạng.
- **Bùng nổ số lượng mẫu (Sample explosion).** $n$ tác tử × không gian trạng thái × hành động kết hợp. Khắc phục bằng xấp xỉ hàm (function approximation); phân tách không gian hành động (mỗi tác tử có một đầu ra chính sách riêng).

## Ứng dụng thực tế (Use It)

Bản đồ ứng dụng MARL năm 2026:

| Lĩnh vực | Phương pháp | Ghi chú |
|----------|-------------|---------|
| Điều hướng / thao tác hợp tác | MAPPO / QMIX | CTDE; critic dùng chung + các actor phân tán. |
| Trò chơi hai người (cờ vua, cờ vây, poker) | Self-play kết hợp MCTS (AlphaZero) | Tổng bằng không; huấn luyện đối xứng. |
| Nhiều người chơi phức tạp (Dota, StarCraft) | Huấn luyện giải đấu + tiền huấn luyện bắt chước | OpenAI Five, AlphaStar. |
| Đội xe tự hành | CTDE MAPPO / PPO kết hợp cơ chế chú ý | Quan sát một phần; quy mô đội xe linh hoạt. |
| Thị trường đấu giá | Cân bằng lý thuyết trò chơi + RL | RL trường trung bình (Mean-field RL) khi $n \to \infty$. |
| Hệ thống đa tác tử LLM (Phase 16) | Giao tiếp ngôn ngữ tự nhiên + gán vai trò | Vòng lặp RL ở tầng lập kế hoạch của tác tử. |

Vào năm 2026, mảng phát triển nhanh nhất của MARL nằm ở các hệ thống dựa trên LLM: các bầy tác tử mô hình ngôn ngữ cùng đàm phán, tranh biện, xây dựng phần mềm. Thuật toán RL xuất hiện dưới dạng tối ưu hóa sở thích trên đầu ra *cấp độ quỹ đạo hoàn chỉnh* (trajectory-level), thay vì từng token riêng lẻ (Phase 16 · 03).

## Đóng gói Sản phẩm (Ship It)

Lưu thành `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Huấn luyện thuật toán Q-learning độc lập trên môi trường GridWorld hợp tác 2 tác tử. Cần bao nhiêu tập cho đến khi phần thưởng tích lũy trung bình > 0? Vẽ đường cong học kết hợp.
2. **Trung bình.** Thêm vào một nhiệm vụ "đồng bộ thời gian": chỉ đạt mục tiêu khi cả hai tác tử cùng bước chân vào ô đích trong cùng một lượt. Q-learning độc lập có còn hội tụ không? Hiện tượng gì xảy ra?
3. **Khó.** Cài đặt một critic tập trung theo phong cách MAPPO và so sánh tốc độ hội tụ với PPO độc lập trên bài toán đồng bộ thời gian nêu trên.

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|-----------------|
| Markov game | "MDP đa tác tử" | $(S, A_1, \ldots, A_n, P, R_1, \ldots, R_n)$; mỗi tác tử có hàm phần thưởng riêng. |
| CTDE | "Huấn luyện tập trung, thực thi phân tán" | Sử dụng critic kết hợp lúc huấn luyện; chính sách mỗi tác tử chỉ dùng quan sát cục bộ khi chạy. |
| IPPO | "PPO độc lập" | Mỗi tác tử chạy thuật toán PPO riêng biệt. Đường cơ sở đơn giản nhưng thường bị đánh giá thấp. |
| MAPPO | "PPO đa tác tử" | PPO với hàm giá trị tập trung phụ thuộc vào trạng thái toàn cục. |
| QMIX | "Phân rã giá trị đơn điệu" | $Q_{\text{tot}} = f_{\text{monotone}}(Q_1, \ldots, Q_n)$ cho phép phân tán hóa phép toán argmax. |
| COMA | "MARL phản thực nghiệm" | Lợi thế = Q của tôi trừ đi Q kỳ vọng khi lấy trung bình trên hành động của tôi. |
| Self-play | "Tác tử đấu với chính mình trong quá khứ" | Một tác tử duy nhất đóng hai vai; chuẩn mực cho các trò chơi tổng bằng không. |
| League play | "Huấn luyện theo quần thể" | Lưu trữ các chính sách quá khứ, lấy mẫu đối thủ từ kho lưu trữ; giải quyết triệt để các chu kỳ chiến thuật. |

## Tài liệu đọc thêm (Further Reading)

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE với một critic tập trung.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — Đường cơ sở phản thực nghiệm cho bài toán quy gán đóng góp.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — Phân rã hàm giá trị với ràng buộc đơn điệu.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — Hiệu năng ấn tượng bất ngờ của PPO trong môi trường MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — Huấn luyện giải đấu ở quy mô lớn.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — Tự chơi thuần túy trong các trò chơi tổng bằng không.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — Bao gồm phần trình bày súc tích về bối cảnh đa tác tử và vấn đề tính không dừng mà CTDE được thiết kế để giải quyết.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — Khảo sát toàn diện về MARL hợp tác, cạnh tranh và hỗn hợp kèm các kết quả hội tụ lý thuyết.
