# GPT — Mô hình hóa ngôn ngữ nhân quả (Causal Language Modeling)

> BERT nhìn thấy cả hai phía. GPT chỉ nhìn về quá khứ. Mặt nạ hình tam giác là dòng mã có sức ảnh hưởng sâu rộng nhất trong lịch sử AI hiện đại.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 phút

## Vấn đề (The Problem)

Một mô hình ngôn ngữ (language model) trả lời một câu hỏi duy nhất: cho trước `t-1` token đầu tiên, phân phối xác suất trên token thứ `t` là gì? Huấn luyện trên tín hiệu đó — dự đoán token tiếp theo (next-token prediction) — và bạn sẽ nhận được một mô hình có khả năng tạo sinh văn bản tùy ý, từng token một.

Để huấn luyện mô hình này từ đầu đến cuối (end-to-end) trên toàn bộ một chuỗi theo cách song song hóa, bạn cần đảm bảo rằng dự đoán tại mỗi vị trí chỉ được phụ thuộc vào các vị trí đứng trước nó. Nếu không, mô hình sẽ gian lận một cách tầm thường bằng cách nhìn thẳng vào đáp án phía sau.

Mặt nạ nhân quả (causal mask) thực hiện điều này. Đó là một ma trận tam giác trên (upper-triangular matrix) chứa các giá trị `-inf` được cộng vào điểm chú ý (attention scores) trước khi đưa qua hàm softmax. Sau khi qua softmax, các vị trí này sẽ nhận giá trị bằng 0. Mỗi vị trí do đó chỉ có thể chú ý tới chính nó và các vị trí xuất hiện trước nó. Và bởi vì bạn chỉ cần áp dụng ma trận này một lần cho toàn bộ chuỗi, bạn sẽ nhận được N dự đoán token tiếp theo song song chỉ trong một lượt truyền xuôi (forward pass).

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2025), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — tất cả đều là các transformer nhân quả chỉ gồm bộ giải mã (decoder-only causal transformers) dùng chung vòng lặp cốt lõi này. Điểm làm nên sự khác biệt giữa chúng nằm ở chất lượng dữ liệu, quy mô và các cải tiến kiến trúc, cùng quá trình hậu huấn luyện (post-training: SFT, RLHF, DPO và các phương pháp kế tục).

## Khái niệm (The Concept)

![Mặt nạ nhân quả tạo ra ma trận chú ý hình tam giác](../assets/causal-attention.svg)

### Mặt nạ (The mask)

Cho một chuỗi có độ dài $N$, xây dựng một ma trận kích thước $N \times N$:

```math
M_{ij} = \begin{cases} 0 & \text{nếu } j \le i \\ -\infty & \text{nếu } j > i \end{cases}
```

Cộng `M` vào điểm chú ý thô trước khi đưa qua softmax. Do $\exp(-\infty) = 0$, các vị trí bị che mặt nạ sẽ đóng góp trọng số bằng 0. Mỗi hàng của ma trận chú ý sẽ là một phân phối xác suất chỉ bao gồm các vị trí đứng trước nó.

Chi phí triển khai: một lệnh gọi hàm `torch.tril()`. Thời gian tính toán: chỉ vài nano giây. Tác động tới toàn bộ lĩnh vực AI: thay đổi tất cả.

### Nguồn gốc của ma trận tam giác (Where the triangle comes from)

Mặt nạ thường được trình bày như một miếng vá được gắn thêm vào cơ chế chú ý. Tuy nhiên, nếu suy diễn theo chiều ngược lại, nó sẽ không còn gì là bí ẩn nữa: cơ chế chú ý thực chất là bước tinh chỉnh thứ ba của phép tính trung bình tiền tố (prefix average), và ma trận tam giác chính là các cận của vòng lặp trung bình đó, được biểu diễn dưới dạng ma trận.

**Giai đoạn 1 — Trung bình tiền tố (Prefix average).** Cách tóm tắt nhân quả thô sơ nhất của một chuỗi: vị trí $i$ trở thành giá trị trung bình của các vị trí từ $0 \ldots i$. Dưới dạng vòng lặp, phép tính đó là `out[i] = X[:i+1].mean(0)`. Phép tính tương tự có thể thực hiện bằng một phép nhân ma trận. Lấy một ma trận tam giác dưới gồm toàn số 1, chia mỗi hàng cho số phần tử của nó, rồi nhân ma trận:

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

Hàng $i$ của $A$ có dạng $[1/(i+1), \ldots, 1/(i+1), 0, \ldots, 0]$. Các số 0 nằm phía trên đường chéo chính chính là tính nhân quả (causality). Không có điều gì thuộc về tương lai bị che đi cả; tương lai đơn giản là chưa bao giờ xuất hiện trong phép tính tổng.

**Giai đoạn 2 — Trọng số học được (Learned weights).** Phép tính trung bình đồng đều coi mọi token trong quá khứ đều có mức độ liên quan như nhau. Hãy thay thế các số 1 bằng một ma trận điểm số học được $S$. Bây giờ tổng các hàng không còn tự động bằng 1 theo cấu trúc nữa, vì vậy ta chuẩn hóa từng hàng bằng softmax thay vì chia cho số đếm. Hàm softmax không bao giờ cho ra giá trị 0 tuyệt đối, điều này sẽ phá vỡ tính nhân quả — trừ khi điểm số của các vị trí tương lai được gán là `-inf`, bởi vì `exp(-inf) = 0`:

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

Cùng một ma trận tam giác, cùng một ma trận ngẫu nhiên theo hàng (row-stochastic matrix), cùng một phép nhân ma trận duy nhất. Mặt nạ `-inf` không phải là một cỗ máy mới mẻ. Nó chính là các phần tử bằng 0 từ giai đoạn 1, được dịch chuyển sang miền đầu vào của hàm softmax.

**Giai đoạn 3 — Trọng số phụ thuộc vào nội dung (Content-dependent weights).** Ở giai đoạn 2, ma trận `S` là cố định sau khi huấn luyện: vị trí 7 luôn gán cho vị trí 3 cùng một trọng số, bất kể nội dung các token là gì. Hãy để điểm số phụ thuộc vào chính các token: `S = Q @ K.T / sqrt(d_k)`. Mọi thứ khác không thay đổi. Mặt nạ, softmax, phép nhân ma trận — hoàn toàn giống hệt nhau.

Ba giai đoạn, một bất biến duy nhất: một ma trận tam giác dưới ngẫu nhiên theo hàng nhân với chuỗi dữ liệu. Trung bình đồng đều, trọng số tĩnh học được, trọng số phụ thuộc nội dung. Mặt nạ chưa bao giờ là thành phần được thêm vào sau cho attention. Nó vốn dĩ đã tồn tại từ phép tính trung bình ban đầu.

```figure
mask-derivation
```

### Huấn luyện song song, suy luận tuần tự (Parallel training, serial inference)

Giai đoạn huấn luyện: thực hiện lượt truyền xuôi trên toàn bộ chuỗi có kích thước `(N, d_model)` một lần duy nhất, tính toán N giá trị hàm mất mát cross-entropy (mỗi vị trí một giá trị), lấy tổng và lan truyền ngược (backprop). Xử lý song song dọc theo chiều dài chuỗi. Đây là lý do tại sao quá trình huấn luyện GPT có khả năng mở rộng (scale) quy mô vượt bậc — bạn có thể xử lý 1 triệu token trong một mẻ (batch) chỉ với một lượt chạy qua GPU.

Giai đoạn suy luận: bạn phải tạo sinh từng token một cách tuần tự. Đưa vào `[t1, t2, t3]`, nhận về `t4`. Đưa vào `[t1, t2, t3, t4]`, nhận về `t5`. Đưa vào `[t1, t2, t3, t4, t5]`, nhận về `t6`. Bộ đệm KV (KV cache, Bài học 12) lưu lại các trạng thái ẩn (hidden states) của `t1…tn` để bạn không phải tính toán lại chúng ở mỗi bước. Tuy nhiên, độ sâu tuần tự khi suy luận vẫn bằng chính độ dài của chuỗi đầu ra. Đó chính là "khoản thuế tự hồi quy" (autoregressive tax) và là lý do tại sao quá trình giải mã (decoding) luôn là nút thắt cổ chai về độ trễ (latency bottleneck) của mọi mô hình ngôn ngữ lớn (LLM).

### Hàm mất mát — dịch chuyển một vị trí (The loss — shift-by-one)

Cho chuỗi token `[t1, t2, t3, t4]`:

- Đầu vào (Input): `[t1, t2, t3]`
- Mục tiêu (Targets): `[t2, t3, t4]`

Với mỗi vị trí `i`, tính toán `-log P(target_i | inputs[:i+1])`. Sau đó lấy tổng. Đây chính là hàm mất mát entropy chéo (cross-entropy) cho toàn bộ chuỗi.

Mọi mô hình ngôn ngữ transformer mà bạn từng biết đều huấn luyện trên hàm mất mát này. Từ tiền huấn luyện (pre-training), tinh chỉnh (fine-tuning) cho đến tinh chỉnh có giám sát (SFT - supervised fine-tuning) — đều dùng chung một hàm mất mát, chỉ khác nhau ở dữ liệu.

### Các chiến lược giải mã (Decoding strategies)

Sau khi huấn luyện, việc lựa chọn phương pháp lấy mẫu (sampling) đóng vai trò quan trọng hơn nhiều so với suy nghĩ thông thường.

| Phương pháp | Cách hoạt động | Khi nào nên dùng |
|--------|--------------|-------------|
| Greedy (Tham lam) | Chọn phần tử cực đại (Argmax) ở mỗi bước | Các tác vụ đơn định (deterministic), sinh code tự động |
| Nhiệt độ (Temperature) | Chia logits cho T, sau đó lấy mẫu | Các tác vụ sáng tạo, T càng cao = độ đa dạng càng lớn |
| Top-k | Chỉ lấy mẫu trong top k token có xác suất cao nhất | Loại bỏ các đuôi phân phối có xác suất thấp |
| Top-p (nucleus) | Lấy mẫu từ tập token nhỏ nhất có tổng xác suất tích lũy $\ge p$ | Tiêu chuẩn mặc định từ 2020+; thích ứng với hình dạng của phân phối |
| Min-p | Chỉ giữ lại các token thỏa mãn `p > min_p * max_p` | Xu hướng từ 2024+; lọc bỏ các đuôi dài hiệu quả hơn top-p |
| Lấy mẫu phỏng đoán (Speculative decoding) | Mô hình nháp (draft model) đề xuất N token, mô hình lớn xác thực song song | Giảm độ trễ từ 2–3 lần mà vẫn giữ nguyên chất lượng đầu ra |

Vào năm 2026, tổ hợp min-p + nhiệt độ 0.7 là cấu hình mặc định rất hợp lý cho các mô hình trọng số mở (open-weights models). Lấy mẫu phỏng đoán (speculative decoding) là yêu cầu tiêu chuẩn bắt buộc cho bất kỳ hạ tầng suy luận thực tế nào.

### Điều gì đã tạo nên thành công của "công thức GPT" (What made the "GPT recipe" work)

1. **Chỉ dùng bộ giải mã (Decoder-only).** Loại bỏ chi phí vận hành của bộ mã hóa. Chỉ một lượt chú ý + mạng truyền thẳng (FFN) cho mỗi lớp.
2. **Quy luật mở rộng quy mô (Scaling).** Từ 124M → 1.5B → 175B → hàng nghìn tỷ tham số. Định luật tỷ lệ Chinchilla (Chinchilla scaling laws - Bài học 13) chỉ dẫn cách phân bổ ngân sách tính toán tối ưu.
3. **Học trong ngữ cảnh (In-context learning).** Xuất hiện khi kích thước mô hình đạt khoảng 6B–13B. Mô hình có thể học và làm theo các ví dụ minh họa (few-shot examples) mà không cần cập nhật trọng số.
4. **Học tăng cường từ phản hồi của con người (RLHF - Reinforcement Learning from Human Feedback).** Quá trình hậu huấn luyện dựa trên sở thích của con người đã biến văn bản tiền huấn luyện thô thành các trợ lý trò chuyện hữu ích.
5. **Pre-norm + RoPE + SwiGLU.** Đảm bảo quá trình huấn luyện diễn ra ổn định ở quy mô cực lớn.

Kiến trúc cốt lõi không thay đổi nhiều kể từ thời điểm GPT-2 ra đời. Mọi bước đột phá lớn đều diễn ra ở khía cạnh dữ liệu, quy mô và các kỹ thuật hậu huấn luyện.

```figure
causal-mask
```

## Tự xây dựng (Build It)

### Bước 1: Mặt nạ nhân quả (the causal mask)

Xem `code/main.py`. Một dòng mã súc tích:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Cộng ma trận này vào điểm số attention trước khi tính softmax. Đó là toàn bộ cơ chế.

### Bước 2: Mô hình dạng GPT 2 lớp (a 2-layer GPT-ish model)

Xếp chồng hai khối decoder (chú ý tự thân có mặt nạ + FFN, không dùng chú ý chéo / cross-attention). Thêm lớp nhúng token, mã hóa vị trí, và một lớp giải nhúng (unembedding - gắn liên kết trọng số với ma trận nhúng token, một thủ thuật chuẩn mực từ thời GPT-2).

### Bước 3: Dự đoán token tiếp theo từ đầu đến cuối (next-token prediction, end-to-end)

Trên một tập từ vựng đồ chơi gồm 20 token, tạo ra logits tại mỗi vị trí. Tính toán hàm mất mát cross-entropy so với mục tiêu dịch chuyển một vị trí (shift-by-one). Không tính gradient — đây là bước kiểm tra tính hợp lý (sanity check) ở lượt truyền xuôi.

### Bước 4: Lấy mẫu (sampling)

Triển khai greedy, temperature, top-k, top-p, min-p. Chạy từng phương pháp trên cùng một câu nhắc (prompt) cố định và so sánh các kết quả đầu ra. Một hàm lấy mẫu chỉ cần khoảng 10 dòng mã.

## Sử dụng trong thực tế (Use It)

Thao tác với PyTorch theo quy chuẩn năm 2026:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Bên dưới tầng giao diện, hàm `generate()` thực hiện lượt truyền xuôi, lấy ra logits ở vị trí cuối cùng, lấy mẫu token tiếp theo, nối token đó vào chuỗi đầu vào và lặp lại quy trình. Mọi ngăn xếp suy luận LLM trong môi trường sản xuất (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) đều triển khai cùng một vòng lặp này với nhiều tầng tối ưu hóa mạnh mẽ — nạp trước theo mẻ (batched prefill), gom mẻ liên tục (continuous batching), phân trang bộ đệm KV (KV cache paging), giải mã phỏng đoán (speculative decoding).

**GPT so với BERT, tóm gọn trong một dòng:** GPT dự đoán $P(x_t \mid x_{<t})$. BERT dự đoán $P(x_{\text{masked}} \mid x_{\text{unmasked}})$. Hàm mất mát quyết định liệu một mô hình có khả năng tạo sinh hay không.

## Đưa vào triển khai (Ship It)

Xem `outputs/skill-sampling-tuner.md`. Kỹ năng này giúp lựa chọn các tham số lấy mẫu cho một tác vụ tạo sinh mới và gắn cờ cảnh báo khi cần giải mã đơn định (deterministic decoding).

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py` và xác minh rằng ma trận chú ý nhân quả là ma trận tam giác dưới sau khi qua softmax. Kiểm tra nhanh: hàng 3 chỉ được phép chứa các trọng số khác 0 ở các cột từ 0 đến 3.
2. **Trung bình.** Triển khai tìm kiếm chùm (beam search) với độ rộng chùm bằng 4. So sánh độ đo perplexity của beam-4 so với greedy trên 10 prompt ngắn. Liệu beam search có luôn thắng? (Gợi ý: thường thắng trong dịch máy, nhưng không phải lúc nào cũng tốt hơn cho trò chuyện mở).
3. **Khó.** Triển khai giải mã phỏng đoán (speculative decoding): dùng một mô hình 2 lớp nhỏ làm mô hình nháp (draft model) và một mô hình 6 lớp làm mô hình xác thực (verifier). Đo đạc tốc độ tăng tốc thời gian thực trên 100 lần sinh văn bản có độ dài 64 token. Xác nhận các đầu ra hoàn toàn trùng khớp với đầu ra greedy của mô hình xác thực.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|------|-----------------|-----------------------|
| Mặt nạ nhân quả (Causal mask) | "Ma trận tam giác" | Ma trận tam giác trên chứa các giá trị `-inf` được cộng vào điểm chú ý để vị trí `i` chỉ có thể nhìn thấy các vị trí `≤ i`. |
| Dự đoán token tiếp theo (Next-token prediction) | "Hàm mất mát chính" | Hàm mất mát cross-entropy của phân phối do mô hình tạo ra so với token kế tiếp thực tế tại mỗi vị trí. |
| Tự hồi quy (Autoregressive) | "Tạo sinh từng từ một" | Đưa đầu ra vừa sinh trở lại làm đầu vào; chỉ có thể song song hóa trong quá trình huấn luyện, không song song hóa được khi sinh văn bản. |
| Logits | "Điểm số trước softmax" | Đầu ra thô của LM head trước khi qua softmax; các thao tác lấy mẫu được thực hiện trên các điểm số này. |
| Nhiệt độ (Temperature) | "Núm xoay sáng tạo" | Chia logits cho $T$; khi $T \to 0$ tương đương greedy, khi $T \to \infty$ phân phối trở nên đồng đều. |
| Top-p | "Lấy mẫu hạt nhân (Nucleus sampling)" | Cắt tỉa phân phối về tập nhỏ nhất có tổng xác suất $\ge p$; lấy mẫu từ tập còn lại đó. |
| Min-p | "Giải pháp cải tiến hơn top-p" | Chỉ giữ các token có xác suất `p ≥ min_p × max_p`; tự động điều chỉnh ngưỡng cắt theo độ sắc nhọn của phân phối. |
| Lấy mẫu phỏng đoán (Speculative decoding) | "Dự thảo + xác thực" | Một mô hình chi phí thấp đề xuất N token; mô hình lớn kiểm tra và xác nhận đồng thời song song. |
| Cưỡng bức giáo viên (Teacher forcing) | "Kỹ thuật huấn luyện" | Trong quá trình huấn luyện, luôn cung cấp token đúng trước đó từ dữ liệu thay vì dùng token do mô hình tự dự đoán. Là chuẩn mực cho mọi seq2seq LM. |

## Tài liệu đọc thêm (Further Reading)

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 và khả năng học trong ngữ cảnh.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — bài báo giới thiệu speculative decoding.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — mã nguồn tham chiếu chuẩn mực cho kiến trúc causal LM.
