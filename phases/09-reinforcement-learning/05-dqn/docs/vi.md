# Mạng Q sâu (Deep Q-Networks - DQN)

> Năm 2013: Mnih huấn luyện một mạng Q-learning duy nhất trên dữ liệu pixel thô, đánh bại mọi tác tử RL cổ điển trên bảy trò chơi Atari. Năm 2015: mở rộng lên 49 trò chơi, xuất bản trên tạp chí Nature, mở ra kỷ nguyên học tăng cường sâu (deep RL). DQN chính là Q-learning kết hợp với ba thủ thuật kỹ thuật then chốt giúp cho việc xấp xỉ hàm số trở nên ổn định.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Lan truyền ngược), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Q-learning dạng bảng đòi hỏi một giá trị Q riêng biệt cho từng cặp (trạng thái, hành động). Bàn cờ vua có khoảng ~10⁴³ trạng thái. Một khung hình Atari kích thước 210×160×3 tương đương 100.800 đặc trưng. RL dạng bảng bất lực ngay từ vài nghìn trạng thái, chưa nói đến con số hàng tỷ.

Giải pháp nhìn lại thì có vẻ hiển nhiên: thay thế bảng Q bằng một mạng nơ-ron, $Q(s, a; \theta)$. Nhưng điều "hiển nhiên khi nhìn lại" ấy đã tiêu tốn hàng thập kỷ nghiên cứu. Việc xấp xỉ hàm số một cách ngây thơ kết hợp với Q-learning sẽ bị phân kỳ do "bộ ba chết chóc (deadly triad)" — xấp xỉ hàm (function approximation) + tự cập nhật (bootstrapping) + học ngoài chính sách (off-policy learning). Mnih et al. (2013, 2015) đã xác định được ba thủ thuật kỹ thuật giúp ổn định hóa quá trình huấn luyện:

1. **Phát lại trải nghiệm (Experience replay)** giúp phi tương quan hóa các bước chuyển trạng thái.
2. **Mạng đích (Target network)** đóng băng mục tiêu bootstrap.
3. **Cắt xén phần thưởng (Reward clipping)** chuẩn hóa độ lớn của gradient.

DQN trên Atari là lần đầu tiên một kiến trúc duy nhất với một bộ siêu tham số duy nhất có thể giải quyết hàng chục bài toán điều khiển trực tiếp từ các pixel thô. Mọi phát triển của "deep-RL" kể từ đó — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — đều được xây dựng chồng lên nền tảng ba thủ thuật này.

## Khái niệm cốt lõi (The Concept)

![Vòng lặp huấn luyện DQN: môi trường, bộ đệm phát lại, mạng trực tuyến, mạng đích, hàm mất mát Bellman TD](../assets/dqn.svg)

**Hàm mục tiêu.** DQN tối thiểu hóa hàm mất mát TD một bước trên một hàm Q mạng nơ-ron:

```math
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 ]
```

$\theta$ = mạng trực tuyến (online network), được cập nhật mỗi bước bằng phương pháp hạ gradient (gradient descent). $\theta^-$ = mạng đích (target network), được sao chép định kỳ từ $\theta$ (sau mỗi ~10.000 bước). $D$ = bộ đệm phát lại (replay buffer) lưu trữ các bước chuyển trạng thái trong quá khứ.

**Ba thủ thuật kỹ thuật, sắp xếp theo mức độ quan trọng:**

**Phát lại trải nghiệm (Experience replay).** Một bộ đệm vòng (ring buffer) chứa khoảng `~10⁶` bước chuyển trạng thái. Mỗi bước huấn luyện sẽ lấy mẫu ngẫu nhiên đều một mini-batch. Kỹ thuật này phá vỡ tương quan thời gian (hai khung hình kế tiếp nhau thường gần như giống hệt), cho phép mạng học nhiều lần từ các bước chuyển hiếm hoi mang lại phần thưởng lớn, và loại bỏ sự phụ thuộc lẫn nhau giữa các lượt cập nhật gradient liên tiếp. Nếu không có nó, phương pháp TD theo chính sách dùng mạng nơ-ron sẽ phân kỳ trên Atari.

**Mạng đích (Target network).** Nếu sử dụng cùng một mạng $Q(\cdot; \theta)$ ở cả hai vế của phương trình Bellman, mục tiêu sẽ liên tục thay đổi sau mỗi bước cập nhật — giống như việc "tự đuổi theo cái đuôi của chính mình". Giải pháp: duy trì một mạng thứ hai $Q(\cdot; \theta^-)$ với các trọng số bị đóng băng. Cứ sau mỗi $C$ bước, sao chép trọng số $\theta \to \theta^-$. Điều này giữ cố định mục tiêu hồi quy trong hàng nghìn bước gradient. Kỹ thuật cập nhật mềm (soft updates) $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$ (dùng trong DDPG, SAC) là một biến thể mượt mà hơn.

**Cắt xén phần thưởng (Reward clipping).** Độ lớn phần thưởng trong Atari dao động từ 1 tới hơn 1000. Việc cắt xén về tập $\lbrace-1, 0, +1 \rbrace$ ngăn không cho bất kỳ trò chơi nào lấn át hoàn toàn gradient. Kỹ thuật này không phù hợp khi độ lớn của phần thưởng mang ý nghĩa quan trọng; nhưng hoàn toàn hợp lý với Atari nơi chỉ dấu của phần thưởng mang tính quyết định.

**DQN kép (Double DQN).** Hasselt (2016) khắc phục thiên kiến cực đại hóa: sử dụng mạng trực tuyến để *lựa chọn* hành động, và sử dụng mạng đích để *đánh giá* hành động đó.

```math
\text{target} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
```

Đây là giải pháp thay thế trực tiếp, cho kết quả tốt hơn một cách nhất quán. Hãy luôn áp dụng làm mặc định.

**Các cải tiến khác (Rainbow, 2017):** phát lại có ưu tiên (prioritized replay - lấy mẫu các bước chuyển có sai số TD cao thường xuyên hơn), kiến trúc đấu tay đôi (dueling architecture - tách riêng nhánh $V(s)$ và nhánh lợi thế advantage), mạng có nhiễu (noisy networks - tự học cơ chế thăm dò), lợi nhuận $n$-bước ($n$-step returns), Q phân phối (distributional Q như C51/QR-DQN), bootstrapping nhiều bước. Mỗi kỹ thuật đóng góp thêm vài phần trăm hiệu năng; các lợi ích này gần như cộng dồn với nhau.

```figure
f3-dqn-stability
```

## Tự xây dựng (Build It)

Mã nguồn dưới đây hoàn toàn chỉ dùng thư viện chuẩn stdlib, không phụ thuộc vào numpy — chúng tôi tự lập trình một mạng MLP một tầng ẩn (hidden layer) trên một môi trường GridWorld liên tục kích thước nhỏ, nhờ đó mỗi bước huấn luyện chỉ chạy trong vài micro-giây. Thuật toán này giống hệt cấu trúc DQN trên Atari ở quy mô lớn.

### Bước 1: Bộ đệm phát lại (Replay buffer)

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Dung lượng ~10⁶ cho bài toán Atari như trong bài báo DQN gốc (thu thập 50.000 bước chuyển trước khi bắt đầu học); với môi trường mẫu đơn giản của chúng ta, dung lượng 5.000 là đủ.

### Bước 2: Một mạng Q kích thước nhỏ (MLP tự viết)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Lan truyền xuôi (Forward pass): tuyến tính → ReLU → tuyến tính. Đó là toàn bộ mạng nơ-ron.

### Bước 3: Bước cập nhật DQN

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Hình thức cập nhật hoàn toàn là Q-learning từ Bài 04 với hai điểm khác biệt: (a) chúng ta truyền ngược gradient qua một hàm $Q(\cdot; \theta)$ khả vi thay vì tra cứu chỉ mục trong bảng, (b) mục tiêu sử dụng mạng $Q(\cdot; \theta^-)$.

### Bước 4: Vòng lặp huấn luyện bên ngoài

Với mỗi tập trải nghiệm, hành động theo ε-greedy dựa trên $Q(\cdot; \theta)$, đẩy các bước chuyển vào bộ đệm, lấy mẫu một mini-batch, thực hiện một bước hạ gradient, và đồng bộ định kỳ $\theta^- \leftarrow \theta$. Cấu trúc như sau:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

Trên môi trường GridWorld nhỏ với trạng thái one-hot 16 chiều, tác tử học được chính sách gần tối ưu sau ~500 tập. Trên Atari, chỉ cần mở rộng cấu trúc này lên 200 triệu khung hình và bổ sung thêm bộ trích xuất đặc trưng tích chập CNN.

## Cạm bẫy thường gặp (Pitfalls)

- **Bộ ba chết chóc (Deadly triad).** Xấp xỉ hàm + học ngoài chính sách + bootstrapping có thể dẫn tới phân kỳ. DQN khắc phục điều này nhờ mạng đích kết hợp bộ đệm phát lại; tuyệt đối không bỏ một trong hai yếu tố này.
- **Thăm dò.** $\epsilon$ bắt buộc phải suy giảm dần, thông thường từ 1.0 xuống 0.01 qua khoảng ~10% thời lượng huấn luyện đầu tiên. Nếu không có đủ sự thăm dò ban đầu, mạng Q sẽ sớm hội tụ vào cực tiểu địa phương nghèo nàn.
- **Ước lượng quá mức (Overestimation).** Toán tử `max` trên các giá trị Q bị nhiễu luôn có xu hướng chệch lên phía trên. Hãy luôn dùng Double DQN trong các hệ thống thực tế.
- **Thang đo phần thưởng (Reward scale).** Hãy cắt xén hoặc chuẩn hóa phần thưởng; độ lớn của gradient tỉ lệ thuận trực tiếp với độ lớn của phần thưởng.
- **Khởi động nguội bộ đệm phát lại (Coldstart).** Đừng bắt đầu huấn luyện mạng cho đến khi bộ đệm đã tích lũy được vài nghìn bước chuyển. Tính gradient quá sớm trên ~20 mẫu dữ liệu sẽ gây ra hiện tượng quá khớp (overfitting) trầm trọng.
- **Tần suất đồng bộ mạng đích.** Quá thường xuyên sẽ tương đương với việc không có mạng đích; quá thưa thớt sẽ khiến các mục tiêu bị lỗi thời. DQN trên Atari dùng 10.000 bước môi trường. Quy tắc kinh nghiệm: đồng bộ sau mỗi ~1/100 tổng thời lượng huấn luyện.
- **Tiền xử lý quan sát.** DQN trên Atari xếp chồng 4 khung hình liên tiếp để biến trạng thái thành Markov. Bất kỳ môi trường nào chứa thông tin vận tốc đều đòi hỏi kỹ thuật xếp chồng khung hình (frame-stacking) hoặc trạng thái hồi quy (recurrent state).

## Ứng dụng thực tế (Use It)

Trong năm 2026, DQN hiếm khi là phương pháp tiên tiến nhất (SOTA) nhưng vẫn là thuật toán ngoài chính sách chuẩn mực để tham chiếu:

| Tác vụ | Phương pháp ưu tiên | Tại sao không dùng DQN? |
|--------|---------------------|--------------------------|
| Trò chơi kiểu Atari với hành động rời rạc | Rainbow DQN hoặc Muesli | Cùng khung lý thuyết nhưng tích hợp nhiều kỹ thuật bổ trợ hơn. |
| Điều khiển liên tục | SAC / TD3 (Phase 9 · 07) | DQN không có mạng chính sách (policy network). |
| Theo chính sách / thông lượng cao | PPO (Phase 9 · 08) | Không cần bộ đệm phát lại; dễ dàng mở rộng quy mô phân tán. |
| RL ngoại tuyến (Offline RL) | CQL / IQL / Decision Transformer | Các mục tiêu Q bảo thủ, loại bỏ nguy cơ bùng nổ do bootstrapping. |
| Không gian hành động rời rạc lớn (hệ gợi ý) | DQN với nhúng hành động (action embedding), hoặc IMPALA | Khả thi, nhưng đòi hỏi thiết kế kiến trúc bổ trợ tinh vi. |
| RL cho LLM | PPO / GRPO | Ở cấp độ chuỗi (sequence-level), không phải từng bước đơn lẻ; hàm mất mát khác biệt. |

Các bài học từ DQN vẫn nguyên giá trị: Bộ đệm phát lại và mạng đích hiện diện trong SAC, TD3, DDPG, SAC-X, bộ đệm tự chơi của AlphaZero, và mọi phương pháp RL ngoại tuyến. Cắt xén phần thưởng được kế thừa dưới dạng chuẩn hóa lợi thế trong PPO. Kiến trúc này chính là bản thiết kế mẫu cho toàn bộ lĩnh vực.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py`. Vẽ biểu đồ đường cong lợi nhuận theo từng tập trải nghiệm. Cần bao nhiêu tập để giá trị trung bình động vượt qua ngưỡng -10?
2. **Trung bình.** Vô hiệu hóa mạng đích (dùng chính mạng trực tuyến cho cả hai vế của mục tiêu Bellman). Đo lường sự mất ổn định trong huấn luyện — liệu lợi nhuận có bị dao động mạnh hay phân kỳ không?
3. **Khó.** Cài đặt Double DQN: dùng mạng trực tuyến để chọn `argmax a'`, dùng mạng đích để đánh giá. So sánh độ chệch của `Q(s_0, best_a)` so với giá trị thực $V^{\ast}(s_0)$ sau 1.000 tập khi có và không có Double DQN trên môi trường GridWorld có phần thưởng bị nhiễu.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| DQN | "Deep Q-learning" | Q-learning kết hợp với hàm Q mạng nơ-ron, bộ đệm phát lại, và mạng đích. |
| Phát lại trải nghiệm (Experience replay) | "Xáo trộn các bước chuyển" | Bộ đệm vòng được lấy mẫu ngẫu nhiên đều ở mỗi bước gradient; loại bỏ tương quan dữ liệu. |
| Mạng đích (Target network) | "Đóng băng bootstrap" | Bản sao định kỳ của mạng Q dùng trong mục tiêu Bellman; giúp ổn định quá trình huấn luyện. |
| Bộ ba chết chóc (Deadly triad) | "Lý do RL bị phân kỳ" | Xấp xỉ hàm + tự cập nhật bootstrapping + học ngoài chính sách = không có bảo đảm hội tụ lý thuyết. |
| Double DQN | "Khắc phục thiên kiến cực đại hóa" | Mạng trực tuyến chọn hành động, mạng đích đánh giá giá trị hành động đó. |
| Dueling DQN | "Hai nhánh V và A" | Phân tách Q = V + A - mean(A); cùng đầu ra nhưng tối ưu hóa luồng gradient tốt hơn. |
| Rainbow | "Tổng hợp mọi thủ thuật" | Tích hợp DDQN + PER + dueling + n-step + noisy nets + distributional Q trong một thể thống nhất. |
| PER | "Phát lại có ưu tiên" | Lấy mẫu các bước chuyển trạng thái với xác suất tỉ lệ thuận với độ lớn của sai số TD. |

## Tài liệu đọc thêm (Further Reading)

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Bài báo tại hội thảo NeurIPS 2013 mở màn cho kỷ nguyên học tăng cường sâu.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Bài báo trên tạp chí Nature, mô hình DQN trên 49 trò chơi Atari.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — Thuật toán DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — Kiến trúc Dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — Công trình tổng hợp các cải tiến của deep RL.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — Khảo cứu lý thuyết về "bộ ba chết chóc" (deadly triad) mà mạng đích và bộ đệm phát lại của DQN được thiết kế để khắc phục.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — Bản cài đặt DQN tham chiếu trong một tệp duy nhất, rất hữu ích khi đọc song song với mã nguồn tự xây dựng trong bài học này.
