# Hỗn hợp chuyên gia (Mixture of Experts - MoE)

> Một mô hình transformer dày đặc (dense) 70B kích hoạt toàn bộ tham số cho mọi token. Một mô hình MoE 671B chỉ kích hoạt 37B tham số mỗi token nhưng lại vượt trội hơn trên mọi bài kiểm chuẩn (benchmark). Tính thưa thớt (sparsity) chính là ý tưởng mở rộng quy mô quan trọng nhất của thập kỷ.

**Phân loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 7 · 05 (Transformer hoàn chỉnh), Giai đoạn 7 · 07 (GPT)  
**Thời lượng:** ~45 phút

## Vấn đề đặt ra

Số phép tính dấu phẩy động (FLOPs) trong suy luận (inference) của một transformer dày đặc (dense transformer) bằng đúng số lượng tham số của nó (nhân 2 cho lượt lan truyền xuôi - forward pass). Khi mở rộng quy mô một mô hình dense, mỗi token đều phải chịu toàn bộ chi phí tính toán đó. Đến năm 2024, các mô hình tiên phong (frontier models) đã chạm phải bức tường tính toán (compute wall): để thông minh hơn một cách rõ rệt, bạn cần số lượng FLOPs cho mỗi token nhiều theo cấp số nhân.

Hỗn hợp chuyên gia (Mixture of Experts - MoE) phá vỡ mối liên kết này. Thay thế mỗi mạng truyền xuôi (feed-forward network - FFN) bằng $E$ chuyên gia độc lập cùng một bộ định tuyến (router) chọn ra $k$ chuyên gia cho mỗi token. Tổng số tham số = $E \times \text{kích thước FFN}$. Số tham số tích cực (active parameters) trên mỗi token = $k \times \text{kích thước FFN}$. Cấu hình điển hình vào năm 2026: $E=256$, $k=8$. Dung lượng lưu trữ tăng theo $E$, trong khi chi phí tính toán chỉ tăng theo $k$.

Biên giới công nghệ năm 2026 gần như hoàn toàn là MoE: DeepSeek-V3 (tổng 671B / 37B tích cực), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Trên bảng xếp hạng độc lập của Artificial Analysis, top 10 mô hình nguồn mở đều là MoE.

## Khái niệm cốt lõi

![Tầng MoE: router chọn k trong số E chuyên gia cho mỗi token](../assets/moe.svg)

### Thay thế mạng FFN (The FFN swap)

Khối transformer dày đặc:

```python
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Khối MoE:

```python
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # chọn k trong số E cho mỗi token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Mỗi chuyên gia là một FFN độc lập (thường là SwiGLU). Bộ định tuyến (router) là một tầng tuyến tính đơn lẻ. Mỗi token tự chọn ra `k` chuyên gia của riêng mình và nhận về một hỗn hợp có cổng (gated mixture) từ đầu ra của các chuyên gia đó.

### Bài toán cân bằng tải (The load-balancing problem)

Nếu bộ định tuyến dồn 90% lượng token qua chuyên gia số 3, các chuyên gia còn lại sẽ rơi vào tình trạng "đói" dữ liệu (starve). Ba giải pháp đã được thử nghiệm:

1. **Hàm mất mát cân bằng tải phụ (Auxiliary load-balancing loss)** (Switch Transformer, Mixtral). Thêm một số hạng phạt tỷ lệ thuận với phương sai trong tần suất sử dụng chuyên gia. Cách này hiệu quả, nhưng lại bổ sung thêm siêu tham số (hyperparameter) và một tín hiệu gradient thứ hai.
2. **Dung lượng chuyên gia + bỏ rơi token (Expert capacity + token dropping)** (Switch Transformer thời kỳ đầu). Mỗi chuyên gia chỉ xử lý tối đa $C \times N/E$ token; các token vượt ngưỡng (overflow) sẽ bỏ qua tầng này. Điều này làm giảm chất lượng mô hình.
3. **Cân bằng không dùng hàm mất mát phụ (Auxiliary-loss-free balancing)** (DeepSeek-V3). Bổ sung một độ chệch (bias) có thể học cho mỗi chuyên gia nhằm dịch chuyển lựa chọn top-k của router. Bias này được cập nhật bên ngoài hàm mất mát huấn luyện (training loss). Không gây phạt lên mục tiêu huấn luyện chính. Đây là bước đột phá lớn của năm 2024.

Cách tiếp cận của DeepSeek-V3: sau mỗi bước huấn luyện, với mỗi chuyên gia, kiểm tra xem mức độ sử dụng của nó cao hơn hay thấp hơn mục tiêu. Điều chỉnh độ chệch một lượng `±γ`. Quá trình lựa chọn sử dụng `scores + bias`. Xác suất của các chuyên gia được dùng để đóng cổng (gating) vẫn là các giá trị `scores` gốc không đổi. Điều này tách rời cơ chế định tuyến (routing) khỏi mức độ biểu đạt (expression).

### Chuyên gia dùng chung (Shared experts)

DeepSeek-V2/V3 cũng chia các chuyên gia thành nhóm *dùng chung* (shared) và nhóm *được định tuyến* (routed). Mọi token đều đi qua tất cả các chuyên gia dùng chung. Các chuyên gia định tuyến được chọn qua top-k. Chuyên gia dùng chung nắm bắt tri thức chung; chuyên gia định tuyến đảm nhiệm việc chuyên môn hóa. V3 sử dụng 1 chuyên gia dùng chung cộng với top-8 trong số 256 chuyên gia định tuyến.

### Chuyên gia hạt mịn (Fine-grained experts)

MoE cổ điển (GShard, Switch): mỗi chuyên gia có độ rộng bằng toàn bộ một FFN đầy đủ. `E` nhỏ (8–64), `k` nhỏ (1–2).

MoE hạt mịn hiện đại (DeepSeek-V3, Qwen-MoE): mỗi chuyên gia hẹp hơn (1/8 kích thước FFN). $E$ lớn (256+), $k$ lớn hơn (8+). Cùng tổng số tham số, nhưng số lượng tổ hợp tăng nhanh hơn rất nhiều. $\binom{256}{8} \approx 400$ nghìn tỷ "chuyên gia tổ hợp" khả dĩ cho mỗi token. Chất lượng tăng lên trong khi độ trễ (latency) giữ nguyên.

### Hồ sơ chi phí (The cost profile)

Tính trên mỗi token, mỗi tầng:

| Cấu hình | Tham số tích cực / token | Tổng tham số |
|----------|--------------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dày đặc) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 đánh bại Llama 3 70B (dense) trên hầu hết mọi bài kiểm chuẩn trong khi thực hiện **ít FLOPs tích cực hơn trên mỗi token**. Nhiều tham số hơn = nhiều tri thức hơn. Nhiều FLOPs tích cực hơn = nhiều phép tính hơn trên mỗi token. MoE tách biệt hai yếu tố này.

### Điểm mấu chốt: Bộ nhớ

Tất cả các chuyên gia đều phải nằm trên GPU bất kể chuyên gia nào được kích hoạt. Một mô hình 671B cần khoảng 1,3 TB VRAM cho các trọng số fp16. Triển khai MoE tiên phong đòi hỏi tính song song hóa chuyên gia (expert parallelism) — phân mảnh (shard) các chuyên gia trên nhiều GPU, định tuyến các token qua mạng. Độ trễ bị chi phối bởi giao tiếp all-to-all qua mạng chứ không phải phép nhân ma trận.

````figure
expert-routing
````

## Xây dựng thực tế

Xem `code/main.py`. Một tầng MoE nhỏ gọn viết bằng thư viện chuẩn (stdlib) thuần túy với:

- `n_experts=8` chuyên gia kiểu SwiGLU (mỗi chuyên gia là một tầng tuyến tính, phục vụ mục đích minh họa)
- Định tuyến top-k=2
- Trọng số cổng (gate weights) được chuẩn hóa bằng softmax
- Cân bằng tải không dùng mất mát phụ (auxiliary-loss-free balancing) qua bias cho từng chuyên gia

### Bước 1: Bộ định tuyến

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax trên scores GỐC của các chuyên gia được chọn
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias chỉ ảnh hưởng đến việc lựa chọn, không ảnh hưởng đến trọng số cổng. Đó chính là bí quyết của DeepSeek-V3 — bias điều chỉnh sự mất cân bằng tải mà không làm sai lệch dự đoán của mô hình.

### Bước 2: Chạy 100 token qua bộ định tuyến

Theo dõi xem các chuyên gia nào được kích hoạt với tần suất bao nhiêu. Nếu không có bias, mức độ sử dụng sẽ bị lệch. Với một vòng lặp cập nhật bias (`-γ` cho chuyên gia bị dùng quá mức, `+γ` cho chuyên gia ít được dùng), mức độ sử dụng sẽ hội tụ về phân phối đồng đều chỉ sau vài vòng lặp.

### Bước 3: So sánh số lượng tham số

In ra "tương đương dày đặc" (dense equivalent) của một cấu hình MoE. Cấu hình dạng DeepSeek-V3: 256 chuyên gia định tuyến + 1 chuyên gia dùng chung, 8 chuyên gia tích cực, d_model=7168. Tổng số lượng tham số khổng lồ đáng kinh ngạc, nhưng số tham số tích cực chỉ bằng một phần bảy so với mô hình dense Llama 3 70B.

## Ứng dụng thực tế

Tải mô hình qua HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Hạ tầng suy luận trong môi trường sản xuất năm 2026: vLLM hỗ trợ định tuyến MoE nguyên bản. SGLang sở hữu đường dẫn song song chuyên gia (expert-parallel path) nhanh nhất. Cả hai đều tự động xử lý việc chọn top-k và tính song song hóa chuyên gia.

**Khi nào nên chọn MoE:**
- Bạn muốn chất lượng hàng đầu với chi phí suy luận trên mỗi token thấp hơn.
- Bạn sở hữu hạ tầng VRAM / song song hóa chuyên gia đủ mạnh.
- Khối lượng công việc thiên về lượng token (chat, viết code) chứ không thiên về ngữ cảnh cực dài (tài liệu dài).

**Khi nào KHÔNG nên chọn MoE:**
- Triển khai trên thiết bị biên (edge deployment) — bạn phải trả toàn bộ chi phí lưu trữ cho bất kỳ FLOP tích cực nào.
- Phục vụ người dùng đơn lẻ yêu cầu độ trễ cực thấp (latency-critical single-user serving) — việc định tuyến chuyên gia sẽ tạo thêm chi phí phụ (overhead).
- Các mô hình nhỏ (<7B) — ưu thế chất lượng của MoE chỉ xuất hiện khi vượt qua một ngưỡng tính toán nhất định (~6B tham số tích cực).

## Đóng gói ứng dụng

Xem file `outputs/skill-moe-configurator.md`. Kỹ năng này giúp chọn $E$, $k$ và cấu trúc chuyên gia dùng chung cho một mô hình MoE mới dựa trên ngân sách tham số, lượng token huấn luyện và mục tiêu triển khai.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Quan sát cách quá trình cập nhật bias không dùng mất mát phụ giúp dàn đều mức độ sử dụng của các chuyên gia qua 50 vòng lặp.
2. **Trung bình.** Thay thế bộ định tuyến học được bằng bộ định tuyến dựa trên hàm băm (hash-based router: mang tính tất định, không cần học). So sánh chất lượng và độ cân bằng tải. Vì sao bộ định tuyến học được lại tốt hơn?
3. **Khó.** Cài đặt cơ chế "định tuyến khớp lượt tạo" (rollout-matched routing) kiểu GRPO (kỹ thuật của DeepSeek-V3.2): ghi log xem những chuyên gia nào được kích hoạt trong khi suy luận, ép buộc sử dụng chính các đường định tuyến đó trong quá trình tính gradient. Đo lường hiệu quả trên một thiết lập policy-gradient mô phỏng.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Chuyên gia (Expert) | "Một FFN trong số nhiều FFN" | Một mạng truyền xuôi độc lập; các tham số dành riêng cho một phần thưa của phép tính FFN. |
| Bộ định tuyến (Router) | "Cổng điều khiển" | Một tầng tuyến tính nhỏ tính điểm số của từng token với từng chuyên gia; lựa chọn top-k. |
| Định tuyến top-k (Top-k routing) | "k chuyên gia tích cực mỗi token" | Phép tính FFN của mỗi token chỉ đi qua đúng k chuyên gia, được tính trọng số bằng cổng (gate). |
| Mất mát phụ (Auxiliary loss) | "Hình phạt cân bằng tải" | Số hạng mất mát bổ sung nhằm phạt tình trạng sử dụng chuyên gia bị lệch. |
| Không mất mát phụ (Auxiliary-loss-free) | "Bí quyết của DeepSeek-V3" | Cân bằng tải thông qua bias trên mỗi chuyên gia chỉ áp dụng cho lựa chọn của router; không tạo thêm gradient. |
| Chuyên gia dùng chung (Shared expert) | "Luôn bật" | Chuyên gia bổ sung mà mọi token đều đi qua; nắm bắt các tri thức phổ quát chung. |
| Song song hóa chuyên gia (Expert parallelism) | "Phân mảnh theo chuyên gia" | Phân phối các chuyên gia khác nhau tới các GPU khác nhau; định tuyến token qua mạng liên kết. |
| Tính thưa thớt (Sparsity) | "Tham số tích cực < tổng tham số" | Tỷ lệ $k \times \text{kích thước chuyên gia} / (E \times \text{kích thước chuyên gia})$; 37/671 ≈ 5,5% đối với DeepSeek-V3. |

## Tài liệu tham khảo

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — ý tưởng gốc.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch Transformer, kiến trúc MoE kinh điển.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + MoE không mất mát phụ + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — bài báo đề xuất cân bằng tải dựa trên bias.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — phân chia chuyên gia hạt mịn + chuyên gia dùng chung mà router trong bài học này áp dụng.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — bài báo gốc về chuyên gia dùng chung.
