# Mô hình hóa Phần thưởng & RLHF (Reward Modeling & RLHF)

> Con người không thể tự tay viết ra một hàm phần thưởng cho khái niệm "phản hồi tốt của một trợ lý ảo", nhưng họ hoàn toàn có thể so sánh hai phản hồi và chọn ra câu trả lời tốt hơn. Hãy huấn luyện một mô hình phần thưởng khớp với các so sánh đó, sau đó áp dụng học tăng cường (RL) lên mô hình ngôn ngữ dựa trên phần thưởng này. Christiano 2017. InstructGPT 2022. Đây chính là công thức đã biến GPT-3 thành ChatGPT. Đến năm 2026, quy trình này phần lớn đang được thay thế bằng DPO — nhưng mô hình tư duy nền tảng vẫn giữ nguyên giá trị cốt lõi.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Thời lượng:** ~45 phút

## Vấn đề đặt ra (The Problem)

Bạn đã huấn luyện một mô hình ngôn ngữ dựa trên mục tiêu dự đoán token tiếp theo (next-token-prediction). Nó có thể viết câu đúng ngữ pháp tiếng Anh. Nhưng nó cũng nói dối, lan man và từ chối một cách thiếu khéo léo. Bạn không thể khắc phục điều này bằng cách tiền huấn luyện (pretraining) thêm — bởi vì văn bản thu thập từ web chính là nguồn cơn của vấn đề, chứ không phải liều thuốc chữa.

Bạn cần một *phần thưởng vô hướng* (scalar reward) để khẳng định rằng: "đối với chỉ dẫn X, phản hồi A tốt hơn phản hồi B". Tự tay viết ra một hàm phần thưởng như vậy là bất khả thi. Sự "hữu ích" (helpfulness) không thể quy về một biểu thức giải tích tường minh trên các token. Nhưng con người có thể dễ dàng so sánh hai câu trả lời và đưa ra sở thích (preference). Đây là dữ liệu có thể thu thập ở quy mô lớn với chi phí hợp lý.

RLHF (học tăng cường từ phản hồi của con người - Reinforcement Learning from Human Feedback, Christiano cùng các cộng sự 2017; Ouyang cùng các cộng sự 2022) chuyển đổi các dữ liệu sở thích thành một mô hình phần thưởng, sau đó tối ưu hóa mô hình ngôn ngữ (LM) thông qua PPO dựa trên phần thưởng đó. Quy trình gồm ba bước: SFT → RM → PPO. Đây là công thức nền móng đã đưa ChatGPT, Claude, Gemini cùng mọi mô hình ngôn ngữ lớn căn chỉnh (aligned LLMs) khác ra mắt trong giai đoạn 2023–2025.

Đến năm 2026, bước PPO phần lớn được thay thế bằng DPO (Phase 10 · 08) vì nó tiết kiệm chi phí tính toán hơn và đạt hiệu quả gần tương đương trong việc căn chỉnh hành vi. Tuy nhiên, khối *mô hình phần thưởng* vẫn là nền tảng cho mọi bộ lấy mẫu Best-of-N, mọi quy trình RL dựa trên phần thưởng có thể kiểm chứng (verifiable rewards), và mọi mô hình suy luận sử dụng mô hình phần thưởng theo tiến trình (Process Reward Model - PRM). Hiểu sâu RLHF đồng nghĩa với việc bạn nắm trọn toàn bộ ngăn xếp công nghệ căn chỉnh mô hình (alignment stack).

## Khái niệm (The Concept)

![Ba giai đoạn của RLHF: SFT, huấn luyện RM trên các cặp sở thích, PPO với số hạng phạt phân kỳ KL](../assets/rlhf.svg)

**Giai đoạn 1: Tinh chỉnh có giám sát (Supervised Fine-Tuning - SFT).** Bắt đầu từ mô hình nền tảng đã tiền huấn luyện. Tinh chỉnh trên các tập dữ liệu mẫu do con người viết để thể hiện hành vi mong muốn (phản hồi tuân thủ hướng dẫn, câu trả lời hữu ích, v.v.). Kết quả: thu được mô hình $\pi_{\text{SFT}}$ có *khuynh hướng thể hiện hành vi tốt* nhưng không gian hành động vẫn chưa bị ràng buộc tối ưu.

**Giai đoạn 2: Huấn luyện Mô hình Phần thưởng (Reward Model - RM training).**

- Thu thập các cặp phản hồi $(y_+, y_-)$ cho cùng một câu lệnh (prompt) $x$, được con người gán nhãn là "$y_+$ được ưa thích hơn $y_-$".
- Huấn luyện một mô hình phần thưởng $R_\phi(x, y)$ để gán điểm số cao hơn cho $y_+$.
- Hàm mất mát: **hàm mất mát logistic theo cặp Bradley-Terry** (Bradley-Terry pairwise logistic):

  $L(\phi) = -\mathbb{E}[ \log \sigma(R_\phi(x, y_+) - R_\phi(x, y_-)) ]$

  Trong đó $\sigma$ là hàm sigmoid. Hiệu số giữa hai điểm phần thưởng phản ánh log-odds của sở thích. Mô hình Bradley-Terry (BT) đã là chuẩn mực từ năm 1952 và là lựa chọn thống trị trong các hệ thống RLHF hiện đại.

- $R_\phi$ thường được khởi tạo từ chính mô hình SFT gắn thêm một đầu ra vô hướng (scalar head) ở trên cùng. Cùng dùng chung một mạng transformer nền; chỉ cần một lớp tuyến tính duy nhất để xuất ra điểm phần thưởng.

**Giai đoạn 3: Tối ưu hóa bằng PPO với số hạng phạt phân kỳ KL.**

- Khởi tạo chính sách cần huấn luyện $\pi_\theta$ từ $\pi_{\text{SFT}}$. Giữ một bản sao đóng băng làm *mô hình tham chiếu* $\pi_{\text{ref}} = \pi_{\text{SFT}}$.
- Phần thưởng tổng hợp ở cuối phản hồi $y$:

  $r_{\text{total}}(x, y) = R_\phi(x, y) - \beta \cdot \mathrm{KL}(\pi_\theta(\cdot \mid x) \Vert \pi_{\text{ref}}(\cdot \mid x))$

  Số hạng phạt phân kỳ KL ngăn cản $\pi_\theta$ trôi dạt quá xa khỏi $\pi_{\text{SFT}}$ — đây là một *số hạng điều chuẩn* (regularizer), chứ không phải là một vùng tin cậy cứng. Hệ số $\beta$ thường nằm trong khoảng 0.01 đến 0.05.
- Chạy thuật toán PPO (Bài học 08) với phần thưởng này. Lợi thế (advantages) được tính toán trên quỹ đạo cấp độ từng token, nhưng RM chỉ chấm điểm một lần duy nhất cho toàn bộ phản hồi hoàn chỉnh.

**Tại sao cần số hạng phạt KL?** Nếu không có nó, PPO sẽ nhanh chóng tìm ra các chiến lược "lạm dụng phần thưởng" (reward-hacking) — bởi vì RM chỉ được huấn luyện trên phân phối dữ liệu đầu vào chuẩn. Một phản hồi nằm ngoài phân phối (out-of-distribution) có thể đạt điểm số ảo cao ngất ngưởng, vượt xa bất kỳ câu trả lời nào do con người viết. Số hạng KL giữ cho $\pi_\theta$ luôn nằm sát đa tạp dữ liệu (manifold) nơi RM được huấn luyện. Đây là núm điều chỉnh quan trọng nhất trong toàn bộ quy trình RLHF.

**Bức tranh công nghệ năm 2026:**

- **DPO** (Rafailov 2023): biến đổi đại số dạng đóng thu gọn Giai đoạn 2 và 3 thành một hàm mất mát có giám sát duy nhất trực tiếp trên dữ liệu sở thích. Không cần RM, không cần PPO. Đạt chất lượng tương đương trên các bài kiểm tra căn chỉnh với chi phí tính toán chỉ bằng một phần nhỏ. Được trình bày chi tiết trong Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO với đường cơ sở so sánh tương đối theo nhóm thay vì dùng mạng critic, phần thưởng lấy từ *bộ xác thực* (verifier - ví dụ: kiểm tra mã chạy đúng / đáp án toán khớp) thay vì RM do con người huấn luyện. Đây là phương pháp thống trị cho các mô hình suy luận. Được trình bày trong Phase 9 · 12.
- **Mô hình phần thưởng theo tiến trình (Process Reward Models - PRMs):** chấm điểm cho các bước giải pháp trung gian (từng bước suy luận), được dùng trong cả các biến thể RLHF và GRPO dành cho tác vụ suy luận.
- **Constitutional AI / RLAIF:** sử dụng một mô hình ngôn ngữ đã căn chỉnh để tự sinh dữ liệu sở thích thay vì thuê con người. Giúp mở rộng quy mô ngân sách dữ liệu sở thích không giới hạn.

```figure
reward-model
```

## Thực hành Xây dựng (Build It)

Bài học này sử dụng các chuỗi "câu lệnh" (prompts) và "phản hồi" (responses) tổng hợp siêu nhỏ. Mô hình RM là một bộ chấm điểm tuyến tính dựa trên biểu diễn túi từ (bag-of-tokens). Không cần mô hình LLM khổng lồ — *cấu trúc hình học* của đường ống xử lý mới là điều cốt lõi, chứ không phải quy mô. Xem thêm tại `code/main.py`.

### Bước 1: Dữ liệu sở thích tổng hợp (synthetic preference data)

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

Trong các dự án RLHF thực tế, bước này được thay thế bằng các chuyên gia gán nhãn con người. Cấu trúc dữ liệu — `(prompt, preferred_response, rejected_response)` — là hoàn toàn tương đồng.

### Bước 2: Mô hình phần thưởng Bradley-Terry

Chấm điểm tuyến tính: `R(x, y) = w · bag(y)`. Huấn luyện để cực tiểu hóa mất mát hàm log theo cặp của Bradley-Terry:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Sau vài trăm bước cập nhật, vector trọng số `w` sẽ gán các giá trị dương cho các token từ ngữ tốt và giá trị âm cho các token từ ngữ xấu.

### Bước 3: Chính sách dạng PPO dựa trên RM

Chính sách minh họa đơn giản của chúng ta sinh ra một token duy nhất từ từ điển. Chúng ta chấm điểm token đó bằng RM, tính $\log \pi_\theta(\text{token} \mid \text{prompt})$, cộng thêm số hạng phạt KL so với mô hình tham chiếu và áp dụng hàm mục tiêu đại diện bị cắt của PPO.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # cập nhật tham số theta theo phong cách PPO, coi reward như phần thưởng tích lũy
    ...
```

### Bước 4: Giám sát phân kỳ KL

Theo dõi giá trị trung bình $\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ sau mỗi bước cập nhật. Nếu chỉ số này tăng vượt mức `~5-10` (giá trị KL cho mỗi phản hồi tính theo đơn vị nats, cộng dồn qua các token, chính là đơn vị ghi log của thư viện TRL), chính sách đã trôi dạt quá xa khỏi $\pi_{\text{SFT}}$: hoặc là $\beta$ quá thấp (cần tăng lên để kéo chính sách về lại quỹ đạo), hoặc hiện tượng lạm dụng phần thưởng (reward hacking) đã bắt đầu diễn ra. Đây là chỉ số chẩn đoán hàng đầu trong thực tế huấn luyện RLHF.

### Bước 5: Công thức triển khai thực tế với thư viện TRL

Khi bạn đã hiểu rõ quy trình đồ chơi trên, dưới đây là vòng lặp tương tự được viết bằng thư viện thực tế. Thư viện [TRL](https://huggingface.co/docs/trl) của Hugging Face là chuẩn mực tham chiếu — dùng `RewardTrainer` cho Giai đoạn 2 và `PPOTrainer` (tích hợp sẵn số hạng phạt KL so với tham chiếu) cho Giai đoạn 3.

```python
# Giai đoạn 2: mô hình phần thưởng từ các cặp dữ liệu sở thích
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# các hàng dữ liệu: {"prompt", "chosen", "rejected"} — định dạng Bradley-Terry
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Giai đoạn 3: PPO dựa trên RM với số hạng phạt KL so với mô hình tham chiếu SFT
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # đóng băng

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats bao gồm: mean_kl, clip_frac, value_loss — ba chỉ số chẩn đoán của PPO
```

Ba chi tiết quan trọng mà thư viện xử lý tự động cho bạn: `adap_kl_ctrl=True` kích hoạt cơ chế điều chỉnh $\beta$ thích ứng: nếu KL quan sát được vượt quá `target_kl`, $\beta$ sẽ tăng gấp đôi; nếu nhỏ hơn một nửa, $\beta$ giảm đi một nửa. Mô hình tham chiếu luôn được đóng băng theo quy ước — bạn tuyệt đối không được để chia sẻ tham số ngẫu nhiên giữa `policy` và `ref`. Cuối cùng, đầu ra giá trị (value head) nằm chung trên thân mạng của policy (`AutoModelForCausalLMWithValueHead` gắn thêm một đầu MLP vô hướng), đó là lý do TRL báo cáo riêng biệt `policy/kl` và `value/loss`.

## Bẫy thường gặp (Pitfalls)

- **Tối ưu hóa quá mức / Lạm dụng phần thưởng (Reward hacking).** Mô hình RM không bao giờ hoàn hảo; $\pi_\theta$ sẽ dễ dàng tìm ra các câu trả lời mang tính đối kháng (adversarial completions) đạt điểm số RM cao nhưng thực tế rất tệ hại. Dấu hiệu nhận biết: điểm phần thưởng tăng liên tục không ngừng trong khi điểm đánh giá thực tế của con người đi ngang hoặc giảm mạnh. Biện pháp xử lý: dừng sớm (early stopping), tăng hệ số $\beta$, mở rộng tập dữ liệu huấn luyện RM.
- **Lạm dụng độ dài (Length hacking).** RM được huấn luyện trên các câu trả lời hữu ích thường có xu hướng ngầm thưởng cho câu trả lời dài hơn. Chính sách sẽ học cách "bôi chữ" lan man. Khắc phục: chuẩn hóa phần thưởng theo độ dài, hoặc dùng RLAIF với RM có khả năng nhận biết độ dài.
- **Kích thước RM quá nhỏ.** Mô hình phần thưởng phải có kích thước tối thiểu bằng với mô hình chính sách. Một RM quá nhỏ không thể đánh giá chính xác đầu ra của một mô hình chính sách lớn hơn.
- **Tinh chỉnh hệ số KL.** $\beta$ quá nhỏ → trôi dạt chính sách và lạm dụng phần thưởng. $\beta$ quá lớn → chính sách hầu như không thay đổi so với SFT. Mẹo chuẩn mực là sử dụng cơ chế $\beta$ *thích ứng* bám theo một ngưỡng KL cố định trên từng bước.
- **Nhiễu trong dữ liệu sở thích.** Khoảng 30% nhãn gán của con người có tính mơ hồ hoặc nhiễu. Hãy lọc bằng cách chỉ huấn luyện RM trên các dữ liệu có sự đồng thuận cao (agreement-filtered) hoặc sử dụng tham số nhiệt độ (temperature) trong mô hình BT.
- **Vấn đề ngoài chính sách.** Dữ liệu PPO sẽ hơi lệch chính sách kể từ epoch thứ hai. Hãy theo dõi sát tỉ lệ cắt (clip fraction) như trong Bài học 08.

## Ứng dụng thực tế (Use It)

Quy trình RLHF năm 2026 được phân tầng rõ rệt:

| Tầng ứng dụng | Mục tiêu | Phương pháp |
|---------------|----------|-------------|
| Tuân thủ chỉ dẫn, hữu ích, vô hại | Căn chỉnh (Alignment) | Ưu tiên dùng DPO (Phase 10 · 08) thay vì RLHF-PPO. |
| Độ chính xác suy luận (toán học, viết mã) | Năng lực (Capability) | GRPO kết hợp phần thưởng từ bộ xác thực (Phase 9 · 12). |
| Tác vụ dài hạn qua nhiều bước | Tác tử (Agentic) | PPO / GRPO kết hợp mô hình phần thưởng theo tiến trình (PRM) trên từng bước. |
| Hành vi an toàn / từ chối yêu cầu độc hại | An toàn (Safety) | RLHF-PPO với RM an toàn chuyên biệt, hoặc Constitutional AI. |
| Best-of-N khi suy luận | Căn chỉnh nhanh | Dùng RM tại thời điểm giải mã (decode); không cần huấn luyện lại chính sách. |
| Chưng cất phần thưởng (Reward distillation) | Tối ưu tính toán khi suy luận | Huấn luyện một "đầu phần thưởng" nhỏ gắn trên mô hình ngôn ngữ đóng băng. |

RLHF từng là phương pháp *thống trị tuyệt đối* trong giai đoạn 2022–2024. Đến năm 2026, các đường ống căn chỉnh trong môi trường sản xuất ưu tiên DPO trước tiên, và chỉ dùng PPO cho các bước đòi hỏi tính toán RM chuyên sâu hoặc các khâu an toàn cực kỳ khắt khe.

## Đóng gói Sản phẩm (Ship It)

Lưu thành `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Huấn luyện mô hình phần thưởng Bradley-Terry trong `code/main.py` trên 500 cặp sở thích tổng hợp. Đo độ chính xác phân loại theo cặp trên tập kiểm tra gồm 100 cặp giữ lại. Độ chính xác phải vượt qua 90%.
2. **Trung bình.** Chạy vòng lặp đồ chơi PPO-RLHF với các giá trị $\beta \in \lbrace 0.0, 0.1, 1.0 \rbrace$. Với mỗi lượt chạy, vẽ biểu đồ điểm RM so với độ phân kỳ KL so với mô hình tham chiếu qua các bước cập nhật. Trường hợp nào xuất hiện hiện tượng lạm dụng phần thưởng?
3. **Khó.** Cài đặt DPO (hàm mất mát tỉ số hợp lý sở thích dạng đóng) trên cùng tập dữ liệu sở thích đó và so sánh với quy trình RLHF-PPO về chi phí tính toán tiêu tốn và điểm số RM cuối cùng đạt được.

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|-----------------|
| RLHF | "RL căn chỉnh" | Quy trình ba giai đoạn SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "Mạng chấm điểm" | Hàm vô hướng học được khớp với các cặp dữ liệu sở thích thông qua Bradley-Terry. |
| Bradley-Terry | "Mất mát logistic theo cặp" | $P(y_+ \succ y_-) = \sigma(R(y_+) - R(y_-))$; hàm mục tiêu tiêu chuẩn của RM. |
| KL penalty | "Bám sát mô hình tham chiếu" | $\beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ trong phần thưởng; số hạng điều chuẩn chống lạm dụng phần thưởng. |
| Reward hacking | "Định luật Goodhart" | Chính sách khai thác lỗ hổng của RM; dấu hiệu: điểm thưởng tăng vọt nhưng đánh giá thực tế của con người đi ngang. |
| RLAIF | "Sở thích do AI gán nhãn" | Quy trình RLHF mà các nhãn sở thích được tạo ra bởi một mô hình ngôn ngữ khác thay vì con người. |
| PRM | "Mô hình phần thưởng tiến trình" | Chấm điểm cho từng bước suy luận trung gian; dùng trong các đường ống suy luận logic. |
| Constitutional AI | "Phương pháp của Anthropic" | Các nhãn sở thích do AI tự sinh dựa trên các nguyên tắc hiến pháp rõ ràng. |

## Tài liệu đọc thêm (Further Reading)

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Bài báo khởi xướng khái niệm RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Công thức thực tiễn đằng sau ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — Nghiên cứu RLHF tiền đề cho bài toán tóm tắt văn bản.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; chuẩn mực mặc định kế tục RLHF vào năm 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF và vòng lặp tự phản biện (self-critique).
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — Bài báo về trợ lý hữu ích và vô hại (Helpful & Harmless).
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — Thư viện với các lớp `RewardTrainer` và `PPOTrainer` dùng trong sản xuất. Nên đọc mã nguồn trainer để hiểu chi tiết về cơ chế adaptive-KL và value-head.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) bởi Lambert, Castricato, von Werra, Havrilla — Bài viết minh họa trực quan kinh điển về quy trình 3 giai đoạn kèm sơ đồ chi tiết.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — Thư viện chính thức; thư mục `examples/` chứa các script chạy RLHF trọn gói cho Llama, Mistral và Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — Góc nhìn giả thuyết phần thưởng; nền tảng cốt lõi để tư duy về hiện tượng lạm dụng phần thưởng.
