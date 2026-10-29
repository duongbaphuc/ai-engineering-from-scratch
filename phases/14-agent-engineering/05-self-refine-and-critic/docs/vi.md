# Self-Refine và CRITIC: Cải thiện đầu ra lặp đi lặp lại

> Self-Refine (Madaan và cộng sự, 2023) sử dụng một LLM duy nhất trong ba vai trò — sinh (generate), phản hồi (feedback), tinh chỉnh (refine) — trong một vòng lặp. Mức tăng trung bình: +20 điểm tuyệt đối trên 7 tác vụ. CRITIC (Gou và cộng sự, 2023) củng cố bước phản hồi bằng cách điều hướng việc xác minh qua các công cụ bên ngoài. Vào năm 2026, mẫu hình này xuất hiện trong mọi framework dưới tên gọi "evaluator-optimizer" (Anthropic) hoặc một vòng lặp rào chắn bảo vệ (OpenAI Agents SDK).

**Loại bài học:** Xây dựng (Build)
**Ngôn ngữ:** Python (stdlib)
**Điều kiện tiên quyết:** Phase 14 · 01 (Vòng lặp tác tử), Phase 14 · 03 (Reflexion)
**Thời gian:** ~60 phút

## Mục tiêu học tập

- Nêu rõ ba prompt của Self-Refine (sinh, phản hồi, tinh chỉnh) và giải thích tại sao lịch sử lại đóng vai trò sống còn đối với prompt tinh chỉnh.
- Giải thích nhận thức sâu sắc mang tính quyết định của CRITIC: LLM không đáng tin cậy trong việc tự xác minh nếu không có sự neo giữ từ bên ngoài (external grounding).
- Triển khai một vòng lặp Self-Refine bằng thư viện chuẩn (stdlib) có lưu lịch sử và một bộ xác minh bên ngoài tùy chọn.
- Ánh xạ mẫu hình này vào quy trình làm việc "evaluator-optimizer" của Anthropic và các rào chắn đầu ra (output guardrails) của OpenAI Agents SDK.

## Vấn đề thực tế

Một tác tử tạo ra một câu trả lời gần như hoàn hảo. Có thể một dòng mã bị lỗi cú pháp nhỏ. Có thể một đoạn tóm tắt hơi quá dài. Có thể một kế hoạch bỏ sót một trường hợp biên (edge case). Điều bạn mong muốn là: tác tử tự phê bình đầu ra của chính nó, sau đó tự sửa chữa.

Self-Refine chứng minh rằng điều này hoạt động hiệu quả chỉ với một mô hình duy nhất, không cần dữ liệu huấn luyện, không cần RL. Nhưng có một trở ngại: LLM rất kém trong việc tự xác minh các sự thật phức tạp. CRITIC đưa ra giải pháp khắc phục — điều hướng bước xác minh qua các công cụ bên ngoài (công cụ tìm kiếm, trình thông dịch mã, máy tính, trình chạy kiểm thử).

Cùng nhau, hai bài báo này định hình chuẩn mặc định năm 2026 cho quá trình cải thiện lặp: sinh, xác minh (từ bên ngoài khi có thể), tinh chỉnh, dừng lại khi bộ xác minh thông qua.

## Khái niệm cốt lõi

### Self-Refine (Madaan và cộng sự, NeurIPS 2023)

Một LLM, ba vai trò:

```text
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
dừng lại khi feedback trả về "không còn vấn đề" hoặc hết ngân sách.
```

Chi tiết then chốt: hàm `refine` nhìn thấy toàn bộ lịch sử — tất cả các đầu ra và lời phê bình trước đó — để nó không lặp lại sai lầm. Bài báo tiến hành phân tích bóc tách (ablation study): nếu bỏ qua lịch sử, chất lượng sẽ sụt giảm nghiêm trọng.

Kết quả nổi bật: Cải thiện trung bình +20 điểm tuyệt đối trên 7 tác vụ (toán học, viết mã, từ viết tắt, hội thoại) bao gồm cả GPT-4. Không cần huấn luyện bổ sung, không cần công cụ bên ngoài, chỉ với một mô hình đơn lẻ.

### CRITIC (Gou và cộng sự, arXiv:2305.11738, v4 tháng 2 năm 2024)

Điểm yếu của Self-Refine: bước phản hồi là việc LLM tự chấm điểm cho chính nó. Đối với các tuyên bố thực tế (factual claims), điều này không đáng tin cậy (một ảo giác thường trông rất thuyết phục đối với chính mô hình đã tạo ra nó). CRITIC thay thế `feedback(task, output)` bằng `verify(task, output, tools)` trong đó `tools` bao gồm:

- Một công cụ tìm kiếm cho các nhận định sự thật.
- Một trình thông dịch mã để kiểm tra tính đúng đắn của code.
- Một máy tính cho các phép tính số học.
- Các bộ xác minh chuyên biệt theo miền (kiểm thử đơn vị, kiểm tra kiểu dữ liệu, linters).

Bộ xác minh tạo ra một lời phê bình có cấu trúc được neo vào kết quả công cụ thực tế. Bộ tinh chỉnh sau đó sẽ dựa trên lời phê bình này để hiệu chỉnh.

Kết quả nổi bật: CRITIC vượt trội hơn Self-Refine trên các tác vụ dựa trên sự thật vì lời phê bình có căn cứ xác thực. Trên các tác vụ không có bộ xác minh bên ngoài (viết sáng tạo, định dạng văn bản), CRITIC quay trở về tương đương với Self-Refine.

### Điều kiện dừng

Hai hình thái phổ biến:

1. **Bộ xác minh thông qua (Verifier passes).** Kiểm thử bên ngoài trả về thành công. Ưu tiên sử dụng khi có sẵn (kiểm thử đơn vị, bộ kiểm tra kiểu, xác nhận rào chắn).
2. **Không có phản hồi lỗi nào được đưa ra (No feedback issued).** Mô hình tự nhận xét "đầu ra đã ổn". Rẻ hơn nhưng kém tin cậy; cần kết hợp với giới hạn số lần lặp tối đa.

Mặc định năm 2026: kết hợp cả hai. "Dừng lại nếu bộ xác minh thông qua HOẶC mô hình báo ổn VÀ số lần lặp >= 2 HOẶC số lần lặp >= max_iterations."

### Evaluator-Optimizer (Anthropic, 2024)

Bài viết tháng 12 năm 2024 của Anthropic gọi đây là một trong năm mẫu hình quy trình làm việc cốt lõi. Gồm hai vai trò:

- Bộ đánh giá (Evaluator): chấm điểm đầu ra và tạo ra lời phê bình.
- Bộ tối ưu hóa (Optimizer): sửa đổi đầu ra dựa trên lời phê bình đó.

Lặp lại cho đến khi bộ đánh giá thông qua. Đây chính là Self-Refine/CRITIC trong cách đóng khung của Anthropic. Chi tiết kỹ thuật quan trọng mà Anthropic bổ sung: prompt của evaluator và optimizer phải có sự khác biệt rõ rệt về mặt cấu trúc để mô hình không chỉ đơn thuần "đóng dấu chuẩn thuận qua loa".

### Rào chắn đầu ra của OpenAI Agents SDK

OpenAI Agents SDK triển khai mẫu hình này dưới tên gọi "output guardrails" (rào chắn đầu ra). Một rào chắn là một bộ kiểm định chạy trên đầu ra cuối cùng của tác tử. Nếu rào chắn bị kích hoạt (ném lỗi `OutputGuardrailTripwireTriggered`), đầu ra sẽ bị từ chối và tác tử có thể thử lại. Các rào chắn có thể gọi công cụ (theo phong cách CRITIC) hoặc là các hàm thuần túy (theo phong cách Self-Refine).

### Cạm bẫy năm 2026

- **Vòng lặp đóng dấu chuẩn thuận mù quáng (Rubber-stamp loops).** Cùng một mô hình vừa sinh vừa phê bình với cùng một phong cách prompt sẽ dễ hội tụ về kết luận "trông ổn rồi đấy". Hãy dùng các prompt khác biệt nhau về cấu trúc, hoặc dùng một mô hình nhỏ giá rẻ để phê bình.
- **Tinh chỉnh quá mức (Over-refinement).** Mỗi lượt tinh chỉnh làm tăng thêm độ trễ và tiêu tốn token. Hãy giới hạn ngân sách từ 1-3 lượt; sau đó, chuyển tiếp vấn đề cho con người xem xét (escalate to human review).
- **Lạm dụng CRITIC trên các tác vụ vụn vặt.** Nếu không có bộ xác minh bên ngoài thực sự, CRITIC sẽ thoái hóa thành Self-Refine; đừng lãng phí độ trễ cho một bộ xác minh giả lập rỗng.

```figure
self-refine
```

## Xây dựng thực tế

Tệp `code/main.py` triển khai Self-Refine và CRITIC trên một tác vụ mô phỏng: tạo một danh sách dấu đầu dòng ngắn cho một chủ đề nhất định. Bộ xác minh kiểm tra định dạng (3 đầu dòng, mỗi dòng dưới 60 ký tự). CRITIC bổ sung một "bộ xác minh sự thật" bên ngoài nhằm phạt các ảo giác đã biết.

Các thành phần:

- `generate` — bộ sinh nội dung theo kịch bản.
- `feedback` — bộ tự phê bình kiểu LLM.
- `verify_external` — bộ xác minh có neo giữ công cụ kiểu CRITIC.
- `refine` — viết lại đầu ra dựa trên lịch sử.
- Điều kiện dừng — bộ xác minh thông qua hoặc tối đa 4 lần lặp.

Chạy thử:

```bash
python3 code/main.py
```

So sánh các lần chạy giữa Self-Refine và CRITIC. CRITIC phát hiện ra một lỗi thực tế mà Self-Refine bỏ sót vì bộ xác minh bên ngoài có sự neo giữ thực tế mà bộ tự phê bình không thể có được.

## Sử dụng trong thực tế

Mẫu hình evaluator-optimizer của Anthropic chính là mẫu hình này được diễn đạt bằng ngôn ngữ thân thiện với Claude. Rào chắn đầu ra của OpenAI Agents SDK mang hình thái của CRITIC (các guardrail có thể gọi công cụ). LangGraph cung cấp một nút phản tỉnh hoạt động tương tự Self-Refine. Tính năng Computer Use trên Gemini 2.5 của Google bổ sung bộ đánh giá an toàn theo từng bước, đây là một biến thể của CRITIC: mọi hành động đều được xác minh trước khi xác nhận thực thi (commit).

## Triển khai thành phẩm

Tệp `outputs/skill-refine-loop.md` cấu hình một vòng lặp evaluator-optimizer dựa trên cấu trúc tác vụ, tính khả dụng của bộ xác minh và ngân sách số lần lặp. Phát ra các prompt cho generator, evaluator/verifier và optimizer, cùng chính sách dừng hoàn chỉnh.

## Bài tập rèn luyện

1. Chạy bản mô phỏng với max_iterations=1. Liệu CRITIC có còn giúp ích không?
2. Thay thế bộ xác minh bên ngoài bằng một bộ xác minh có nhiễu (30% dương tính giả ngẫu nhiên). Vòng lặp sẽ xử lý ra sao? Đây chính là thực tế năm 2026 của hầu hết các ngăn xếp rào chắn bảo vệ.
3. Triển khai biến thể "generator-critic trên hai mô hình khác nhau": mô hình lớn sinh nội dung, mô hình nhỏ phê bình. Liệu nó có đánh bại cách tiếp cận dùng cùng một mô hình không?
4. Đọc Phần 3 của bài báo CRITIC (arXiv:2305.11738 v4). Nêu tên ba danh mục công cụ xác minh và đưa ra ví dụ cho từng loại.
5. Ánh xạ `output_guardrails` của OpenAI Agents SDK vào vai trò verifier của CRITIC. SDK này xử lý tốt điểm nào và còn thiếu sót điểm nào?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất thực sự |
|-----------|----------------------|------------------|
| Self-Refine | "LLM tự sửa lỗi" | Vòng lặp Generate -> feedback -> refine trong cùng một mô hình, có kèm lịch sử |
| CRITIC | "Xác minh neo vào công cụ" | Thay thế bước feedback bằng bộ xác minh bên ngoài (tìm kiếm, mã, tính toán, kiểm thử) |
| Evaluator-Optimizer | "Mẫu hình quy trình làm việc của Anthropic" | Hai vai trò — evaluator chấm điểm, optimizer sửa đổi — lặp lại đến khi hội tụ |
| Rào chắn đầu ra (Output guardrail) | "Kiểm tra hậu kỳ" | Trình xác thực của OpenAI Agents SDK chạy sau khi tác tử tạo ra đầu ra |
| Bước xác minh (Verify step) | "Giai đoạn phê bình" | Quyết định chịu tải then chốt: có neo vào thực tế hay tự đánh giá chủ quan |
| Lịch sử tinh chỉnh (Refine history) | "Những gì mô hình đã thử" | Các đầu ra và lời phê bình trước đó được đưa vào prompt tinh chỉnh; bỏ qua phần này chất lượng sẽ sụt giảm |
| Vòng lặp đóng dấu chuẩn thuận mù quáng (Rubber-stamp loop) | "Lỗi tự đồng thuận" | Lời phê bình với cùng phong cách prompt luôn trả về "trông ổn rồi"; khắc phục bằng các prompt khác biệt cấu trúc |
| Điều kiện dừng (Stop condition) | "Kiểm tra hội tụ" | Bộ xác minh thông qua HOẶC không có phản hồi VÀ giới hạn số lần lặp; tuyệt đối không dùng điều kiện đơn lẻ |

## Tài liệu đọc thêm

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — bài báo kinh điển
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — xác minh có neo công cụ
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — mẫu hình quy trình làm việc evaluator-optimizer
- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — các rào chắn đầu ra đóng vai trò bộ xác minh kiểu CRITIC
