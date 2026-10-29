# RL cho Trò chơi — AlphaZero, MuZero và Kỷ nguyên Suy luận của LLM

> Năm 1992: TD-Gammon đánh bại các nhà vô địch thế giới ở cờ thỏ cáo (backgammon) bằng TD thuần túy. Năm 2016: AlphaGo đánh bại Lee Sedol. Năm 2017: AlphaZero thống trị cờ vua, cờ shogi và cờ vây hoàn toàn từ đầu (from scratch). Năm 2025: DeepSeek-R1 chứng minh rằng cùng một công thức đó, khi thay thế PPO bằng GRPO, hoạt động hoàn hảo trên các tác vụ suy luận. Trò chơi chính là đấu trường thước đo thúc đẩy mọi bước đột phá trong giai đoạn này.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Thời lượng:** ~120 phút

## Vấn đề đặt ra (The Problem)

Các trò chơi sở hữu tất cả những gì mà học tăng cường khao khát. Tín hiệu phần thưởng tường minh, rõ ràng (thắng/thua). Số lượng tập chạy vô hạn (khởi tạo lại tức thì qua cơ chế tự đấu - self-play). Khả năng mô phỏng hoàn hảo (bản thân trò chơi *chính là* bộ mô phỏng). Không gian hành động rời rạc hoặc liên tục quy mô nhỏ. Cấu trúc đa tác tử buộc các bên phải rèn luyện tính bền bỉ trước các chiến lược đối kháng.

Chính vì vậy, mọi bước đột phá lớn của RL đều được khai sinh và thử nghiệm trên các trò chơi: TD-Gammon (cờ thỏ cáo, 1992), Atari-DQN (2013), AlphaGo (2016), AlphaZero (2017), OpenAI Five (Dota 2, 2019), AlphaStar (StarCraft II, 2019), MuZero (học mô hình môi trường, 2019), AlphaTensor (phép nhân ma trận, 2022), AlphaDev (thuật toán sắp xếp, 2023), và DeepSeek-R1 (suy luận toán học, 2025) — minh chứng mới nhất cho thấy các kỹ thuật RL trong trò chơi hoàn toàn có thể áp dụng hiệu quả trên dữ liệu văn bản.

Bài học tổng kết này khảo sát ba kiến trúc mang tính cột mốc — AlphaZero, MuZero và GRPO — qua một lăng kính thống nhất duy nhất: **tự đấu (self-play) + tìm kiếm (search) + cải tiến chính sách (policy improvement)**. Mỗi phương pháp sau đều là sự tổng quát hóa của phương pháp trước; đặc biệt, GRPO chính là công thức của AlphaZero được áp dụng vào khả năng suy luận của mô hình ngôn ngữ lớn (LLM), trong đó các token đóng vai trò là hành động và bộ xác thực toán học/mã nguồn đóng vai trò là tín hiệu chiến thắng.

## Khái niệm (The Concept)

![AlphaZero ↔ MuZero ↔ GRPO: cùng một vòng lặp, môi trường khác nhau](../assets/rl-games.svg)

**Vòng lặp thống nhất (The unifying loop).**

```python
while True:
    trajectory = self_play(current_policy, search)     # tự chơi đối kháng với chính mình
    policy_target = search.improved_policy(trajectory) # tìm kiếm cải thiện chính sách thô
    policy_net.update(policy_target, value_target)     # học có giám sát trên kết quả tìm kiếm
```

**AlphaZero (2017).** Silver cùng các cộng sự. Áp dụng cho các trò chơi hoàn hảo (cờ vua, shogi, cờ vây) với luật chơi đã biết trước:

- Mạng chính sách - giá trị (Policy-value network): một thân mạng duy nhất $f_\theta(s) \to (p, v)$. Trong đó $p$ là phân phối tiên nghiệm (prior) trên các nước đi hợp lệ; $v$ là kết quả kỳ vọng của ván đấu.
- Tìm kiếm cây Monte Carlo (Monte Carlo Tree Search - MCTS): tại mỗi nước đi, mở rộng một cây gồm các nước đi khả dĩ tiếp theo. Sử dụng cặp $(p, v)$ làm tiên nghiệm kết hợp với giá trị mồi (bootstrap). Lựa chọn các nút con theo công thức UCB (cụ thể là PUCT): $a^{\ast} = \arg\max_a Q(s, a) + c \cdot p(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))$.
- Tự đấu (Self-play): cho hai bản sao của tác tử thi đấu với nhau. Tại bước $t$, phân phối lượt ghé thăm của MCTS $\pi_t$ trở thành mục tiêu huấn luyện có giám sát cho mạng chính sách.
- Hàm mất mát: $L = (v - z)^2 - \pi \cdot \log p + c \cdot \lVert \theta \rVert^2$, với $z$ là kết quả thực tế của ván cờ (+1 / 0 / -1).

Hoàn toàn không dùng tri thức con người. Không có các hàm đánh giá heuristic thủ công. Một công thức duy nhất làm chủ hoàn toàn cờ vua, shogi và cờ vây chỉ sau vài chục triệu ván tự đấu.

**MuZero (2019).** Schrittwieser cùng các cộng sự. Xóa bỏ yêu cầu bắt buộc phải biết trước luật chơi.

- Thay vì phụ thuộc vào một môi trường cố định, MuZero học một *mô hình động lực học ẩn* (latent dynamics model) gồm bộ ba hàm $(h, g, f)$:
  - $h(s)$: mã hóa quan sát đầu vào thành trạng thái ẩn (latent state).
  - $g(s_{\text{latent}}, a)$: dự đoán trạng thái ẩn tiếp theo + phần thưởng tức thời.
  - $f(s_{\text{latent}})$: dự đoán phân phối tiên nghiệm chính sách + giá trị trạng thái.
- Thuật toán MCTS chạy trực tiếp trong *không gian ẩn học được*. Vẫn giữ nguyên cấu trúc tìm kiếm và vòng lặp huấn luyện.
- Hoạt động xuất sắc trên cả cờ vây, cờ vua, shogi *lẫn* tập trò chơi Atari — một thuật toán duy nhất, hoàn toàn không cần nạp trước luật chơi.

**Stochastic MuZero (2022).** Bổ sung động lực học ngẫu nhiên và các nút cơ hội (chance nodes); mở rộng cho các trò chơi có tính may rủi như cờ thỏ cáo.

**Muesli, Gumbel MuZero (2022-2024).** Cải tiến vượt bậc về hiệu quả sử dụng mẫu và khả năng tìm kiếm tất định.

**GRPO (2024-2025).** Công thức của DeepSeek-R1. Vẫn là vòng lặp mang cấu trúc AlphaZero, nhưng áp dụng cho bài toán suy luận của mô hình ngôn ngữ:

- "Trò chơi": giải một bài toán học / lập trình / câu đố logic. "Thắng cuộc" = bộ xác thực (verifier - chạy qua bộ kiểm thử tự động, kết quả số học trùng khớp) trả về giá trị 1.
- Chính sách: chính là mô hình ngôn ngữ (LLM). Hành động: các token. Trạng thái: prompt + toàn bộ phản hồi đã sinh ra cho đến hiện tại.
- Không cần mạng phê bình critic (như mạng giá trị $V_\phi$ trong PPO). Thay vào đó, với mỗi prompt, lấy mẫu $G$ câu trả lời độc lập từ chính sách. Tính phần thưởng cho từng câu trả lời. Sử dụng **lợi thế tương đối theo nhóm** (group-relative advantage) $A_i = (r_i - \text{mean}_r) / \text{std}_r$ làm tín hiệu cập nhật kiểu REINFORCE.
- Áp dụng số hạng phạt phân kỳ KL so với chính sách tham chiếu để chống trôi dạt (tương tự RLHF).
- Hàm mất mát đầy đủ:

  $L_{\text{GRPO}}(\theta) = -\mathbb E_{q, \lbrace o_i \rbrace} \left[ \frac{1}{G} \sum_i A_i \cdot \log \pi_\theta(o_i \mid q) \right] + \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$

Không cần mô hình phần thưởng học được, không cần mạng critic, không cần thuật toán MCTS phức tạp. Một bộ xác thực dựa trên luật cung cấp tín hiệu phần thưởng, đường cơ sở tương đối theo nhóm thay thế cho critic, và bước tìm kiếm cây được lược bỏ. Đạt chất lượng tương đương hoặc vượt trội PPO-RLHF trên các benchmark suy luận với chi phí tính toán chỉ bằng một phần nhỏ.

**Công thức R1 hoàn chỉnh (The R1 recipe in full).** DeepSeek-R1 (DeepSeek 2025) gồm hai mô hình trong cùng một công bố:

- **R1-Zero.** Bắt đầu từ mô hình nền tảng DeepSeek-V3. Không qua bước SFT. Áp dụng trực tiếp thuật toán GRPO với hai thành phần phần thưởng: *phần thưởng độ chính xác* (accuracy reward — dựa trên luật: câu trả lời cuối cùng có phân tích ra đúng số không / mã nguồn có vượt qua unit tests không) và *phần thưởng định dạng* (format reward — phản hồi có bọc chuỗi suy nghĩ trong cặp thẻ `<think>…</think>` không). Qua hàng nghìn bước huấn luyện, độ dài trung bình của câu trả lời tăng từ ~100 lên ~10.000 token và điểm số benchmark toán học vươn lên tiệm cận mức của o1-preview. Mô hình tự học cách suy luận hoàn toàn từ đầu (from scratch). Nhược điểm: chuỗi suy nghĩ (chain-of-thought) thường khó đọc, pha trộn nhiều ngôn ngữ và thiếu trau chuốt về văn phong.
- **R1.** Khắc phục các vấn đề về khả năng đọc của R1-Zero bằng một quy trình bốn giai đoạn:
  1. **SFT khởi động nguội (Cold-start SFT).** Thu thập vài nghìn mẫu trình diễn chuỗi suy nghĩ dài (long-CoT) có định dạng chuẩn mực, rõ ràng. Tinh chỉnh có giám sát mô hình nền trên tập dữ liệu này để tạo điểm xuất phát dễ đọc.
  2. **GRPO định hướng suy luận (Reasoning-oriented GRPO).** Áp dụng GRPO với các phần thưởng độ chính xác + định dạng cộng thêm *phần thưởng tính nhất quán ngôn ngữ* (language-consistency reward) để tránh việc chuyển ngữ lộn xộn.
  3. **Lấy mẫu loại bỏ + SFT vòng 2 (Rejection sampling + SFT round 2).** Lấy mẫu ~600.000 quỹ đạo suy luận từ checkpoint RL, chỉ giữ lại những lời giải có đáp án cuối cùng chính xác và chuỗi CoT mạch lạc, kết hợp với ~200.000 mẫu SFT không thuộc dạng suy luận (viết lách, hỏi đáp, tự nhận thức). Tinh chỉnh có giám sát lại mô hình nền.
  4. **GRPO toàn phổ (Full-spectrum GRPO).** Một vòng huấn luyện RL cuối cùng kết hợp cả tác vụ suy luận (dùng phần thưởng theo luật) và căn chỉnh hành vi tổng quát (dùng phần thưởng sở thích về độ hữu ích / vô hại).

Kết quả đạt hiệu năng tương đương o1 trên AIME và MATH-500 ở dạng trọng số mở (open weights), và đủ nhỏ gọn để chưng cất. Công bố này cũng phát hành 6 mô hình đặc (dense models) chưng cất (từ Qwen-1.5B đến Llama-70B) bằng cách chạy SFT trên các vết suy luận của R1 — hoàn toàn không cần chạy RL ở mô hình học trò. Việc chưng cất từ một mô hình thầy RL mạnh mẽ luôn vượt trội so với việc huấn luyện RL từ đầu ở quy mô tham số của học trò.

**Tại sao dùng GRPO thay vì PPO cho suy luận?** Có ba lý do được chỉ ra trong bài báo DeepSeekMath (tháng 2/2024): (1) không cần huấn luyện mạng giá trị, giảm một nửa dung lượng bộ nhớ; (2) đường cơ sở theo nhóm xử lý tự nhiên tín hiệu phần thưởng thưa thớt ở cuối quỹ đạo vốn là đặc trưng của các bài toán suy luận; (3) cơ chế chuẩn hóa theo từng prompt giúp giá trị lợi thế có thể so sánh được giữa các bài toán có độ khó hoàn toàn khác nhau, điều mà một critic duy nhất của PPO không thể làm được.

**Không cần tìm kiếm so với Dựa trên tìm kiếm (Search-free vs Search-based).** Nhánh ứng dụng trò chơi hiện đã phân hóa:

- *Trò chơi thông tin hoàn hảo với chân trời bước dài* (cờ vây, cờ vua): vẫn dựa trên tìm kiếm. AlphaZero / MuZero tiếp tục thống trị.
- *Suy luận LLM*: hiện chưa dùng MCTS trong môi trường sản xuất; chủ yếu dùng GRPO trên toàn bộ lượt sinh, kết hợp lấy mẫu Best-of-N để tăng tính toán lúc suy luận. Các mô hình phần thưởng theo tiến trình (PRMs) đang gợi mở khả năng đưa tìm kiếm cấp độ từng bước quay trở lại.

```figure
f3-selfplay-ladder
```

## Thực hành Xây dựng (Build It)

Mã nguồn trong `code/main.py` cài đặt **thuật toán GRPO thu nhỏ**: một bài toán bandit với nhiều nhóm mẫu. Thuật toán hoàn toàn tương đồng như trên một LLM thực thụ; chỉ có chính sách và môi trường là được đơn giản hóa. Bài thực hành này giúp nắm vững *hàm mất mát* và *lợi thế tương đối theo nhóm*, cải tiến then chốt năm 2024 từ DeepSeekMath.

### Bước 1: Môi trường bộ xác thực siêu nhỏ (tiny verifier environment)

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

Trong hệ thống GRPO thực tế, bộ xác thực sẽ chạy các unit test hoặc kiểm tra đẳng thức toán học.

### Bước 2: Chính sách: Softmax trên K token câu trả lời cho mỗi prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Tương đương với đầu ra của lớp cuối cùng trong một mô hình LLM có điều kiện theo prompt.

### Bước 3: Lấy mẫu theo nhóm và tính lợi thế tương đối theo nhóm

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # Phạt KL: kéo theta về gần mô hình tham chiếu
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Lợi thế tương đối theo nhóm chính là cải tiến cốt lõi của DeepSeek năm 2024. Không cần mạng critic. "Đường cơ sở" chính là giá trị trung bình của nhóm, và việc chuẩn hóa được thực hiện bằng độ lệch chuẩn của nhóm.

### Bước 4: So sánh với đường cơ sở REINFORCE (không dùng mạng giá trị)

Cùng một thiết lập, cùng tài nguyên tính toán, chạy REINFORCE thuần túy. GRPO hội tụ nhanh hơn và ổn định hơn rõ rệt.

### Bước 5: Theo dõi entropy và phân kỳ KL

Các chỉ số chẩn đoán tương tự như trong RLHF: giá trị KL trung bình so với mô hình tham chiếu, entropy của chính sách, và phần thưởng theo thời gian. Khi các chỉ số này đi vào ổn định, quá trình huấn luyện hoàn tất.

## Bẫy thường gặp (Pitfalls)

- **Lạm dụng phần thưởng qua gian lận bộ xác thực (Verifier gaming).** GRPO thừa hưởng rủi ro của RLHF: nếu bộ xác thực có kẽ hở hoặc bị sai, LLM sẽ ngay lập tức khai thác triệt để lỗ hổng đó. Bộ xác thực phải cực kỳ chặt chẽ (nhiều ca kiểm thử biên, chứng minh hình thức).
- **Kích thước nhóm quá nhỏ.** Phương sai của đường cơ sở trung bình nhóm tỉ lệ với $1/G$ (sai số chuẩn tỉ lệ với $1/\sqrt{G}$). Nếu $G < 4$, tín hiệu lợi thế sẽ rất nhiễu; kích thước tiêu chuẩn thường là $G = 8$ đến 64.
- **Độ lệch độ dài (Length bias).** Các câu trả lời LLM có độ dài khác nhau sẽ có log-xác suất tích lũy khác nhau. Cần chuẩn hóa theo số lượng token, hoặc dùng log-prob cấp chuỗi, hoặc cắt bớt độ dài tối đa.
- **Vòng lặp tự đấu luẩn quẩn.** Cơ chế tự đấu kiểu AlphaZero có thể bị mắc kẹt trong vòng lặp chiếm ưu thế lẫn nhau ở các trò chơi tổng tổng quát. Giảm thiểu bằng cách duy trì kho đối thủ đa dạng (huấn luyện giải đấu, Bài học 10).
- **Lệch pha giữa tìm kiếm và chính sách (Search-policy mismatch).** AlphaZero huấn luyện mạng chính sách để bắt chước phân phối tìm kiếm. Nếu mạng chính sách quá nhỏ không đủ sức chứa phân phối của MCTS, quá trình học sẽ đình trệ.
- **Ngưỡng sàn tính toán (Compute floor).** MuZero / AlphaZero đòi hỏi tài nguyên tính toán khổng lồ. Một thử nghiệm nhỏ cũng có thể tiêu tốn hàng trăm giờ GPU. Hãy dùng các bản demo thu nhỏ (như AlphaZero trên cờ Connect Four) để học tập nguyên lý.
- **Độ bao phủ của bộ xác thực (Verifier coverage).** Nếu một giải pháp có lỗi nhưng vẫn vượt qua được unit test sơ sài, thuật toán sẽ vô tình củng cố lỗi sai đó. Hãy thiết kế bộ kiểm tra bao quát được tất cả các trường hợp biên.

## Ứng dụng thực tế (Use It)

Bức tranh ứng dụng RL trong trò chơi và suy luận năm 2026:

| Lĩnh vực | Phương pháp thống trị |
|----------|-----------------------|
| Trò chơi đối kháng 2 người thông tin hoàn hảo (cờ vây, cờ vua, shogi) | AlphaZero / MuZero / KataGo |
| Trò chơi bài thông tin không hoàn hảo (poker) | CFR + Deep Learning (DeepStack, Libratus, Pluribus) |
| Atari / trò chơi điểm ảnh | Muesli / MuZero / IMPALA-PPO |
| Trò chơi chiến thuật nhiều người chơi (Dota, StarCraft) | PPO + tự chơi + huấn luyện giải đấu (OpenAI Five, AlphaStar) |
| Suy luận toán học / lập trình trên LLM | GRPO (DeepSeek-R1, Qwen-RL, các bản tái lập mở) |
| Căn chỉnh hành vi LLM | DPO / RLHF-PPO (không dùng GRPO do phần thưởng là sở thích chủ quan) |
| Robotics | PPO + DR (dùng chung công cụ gradient chính sách) |
| Bài toán tổ hợp (nhân ma trận, sắp xếp) | Các biến thể AlphaZero (AlphaTensor, AlphaDev) |

*Công thức cốt lõi* — tự đấu, nâng cao năng lực bằng tìm kiếm, chưng cất chính sách — xuyên suốt từ văn bản, điểm ảnh hình ảnh cho tới điều khiển vật lý. GRPO là đại diện mới nhất trong dòng chảy này; và chắc chắn sẽ còn nhiều biến thể mới xuất hiện.

## Đóng gói Sản phẩm (Ship It)

Lưu thành `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Cài đặt bài toán GRPO bandit trong `code/main.py`. Huấn luyện trên 2 prompt × 4 token đáp án mỗi loại. Đạt hội tụ trong vòng chưa đầy 1.000 bước cập nhật với $G=8$.
2. **Trung bình.** Thay thế bằng thuật toán PPO (dạng cắt) và REINFORCE thuần túy. So sánh hiệu quả sử dụng mẫu và phương sai phần thưởng so với GRPO trên cùng bài toán bandit đó.
3. **Khó.** Mở rộng bài toán sang "chuỗi suy luận" độ dài 2: tác tử sinh ra hai token liên tiếp và bộ xác thực chấm điểm cho cả cặp. Đo lường cách GRPO xử lý bài toán quy gán đóng góp qua chuỗi hai bước này. (Gợi ý: tính lợi thế nhóm trên *toàn bộ chuỗi*, sau đó lan truyền tín hiệu đó về cả hai vị trí token).

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|-----------------|
| MCTS | "Tìm kiếm cây kết hợp mạng học" | Tìm kiếm cây Monte Carlo; chọn nút bằng UCB1/PUCT kết hợp tiên nghiệm $(p, v)$ học được. |
| AlphaZero | "Tự đấu + MCTS" | Mạng chính sách - giá trị được huấn luyện khớp với phân phối MCTS và kết quả ván cờ. |
| MuZero | "AlphaZero học mô hình" | Cùng vòng lặp tìm kiếm nhưng chạy trong không gian trạng thái ẩn thông qua mô hình động lực học học được. |
| GRPO | "PPO không cần critic" | Tối ưu hóa chính sách tương đối theo nhóm; REINFORCE kết hợp đường cơ sở trung bình nhóm + số hạng KL. |
| PUCT | "Công thức UCB của AlphaZero" | $Q + c \cdot p \cdot \sqrt{N} / (1 + N_a)$ — cân bằng giữa giá trị ước lượng và xác suất tiên nghiệm. |
| Self-play | "Tác tử đấu với chính mình" | Chuẩn mực cho các trò chơi tổng bằng không; mang lại tín hiệu huấn luyện đối xứng. |
| League play | "Tự đấu theo quần thể" | Lấy mẫu đối thủ từ kho lưu trữ gồm các phiên bản quá khứ, hiện tại và các tác tử khai thác. |
| Verifier reward | "RL có thể kiểm chứng" | Tín hiệu phần thưởng đến từ bộ kiểm tra tất định (unit tests vượt qua, đáp án chính xác). |
| Process reward | "PRM" | Chấm điểm cho từng bước suy luận trung gian, thay vì chỉ chấm đáp án cuối cùng. |

## Tài liệu đọc thêm (Further Reading)

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — Bài báo giới thiệu thuật toán GRPO và đường cơ sở tương đối theo nhóm.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — Toàn bộ công thức R1 bốn giai đoạn cùng phân tích thực nghiệm loại bỏ R1-Zero.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR kết hợp học sâu ở quy mô lớn.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — Bài báo khởi nguồn cho toàn bộ lĩnh vực.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — Tài liệu tham khảo chuẩn mực để ứng dụng GRPO với các hàm phần thưởng tùy chỉnh trong sản xuất.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — Bản tái hiện mã nguồn mở cho công thức R1 ở nhiều quy mô mô hình khác nhau.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — Khung lý thuyết sách giáo khoa về tự đấu, tìm kiếm và "phần thưởng thiết kế" mà R1 hiện thực hóa ở quy mô LLM.
