# Giám sát có thể mở rộng và Tổng quát hóa từ yếu sang mạnh

> Burns và các cộng sự (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023) đã đề xuất một mô hình đại diện (proxy) cho bài toán siêu căn chỉnh (superalignment): tinh chỉnh một mô hình mạnh bằng cách sử dụng các nhãn do một mô hình yếu hơn tạo ra. Nếu mô hình mạnh có thể tổng quát hóa chính xác từ sự giám sát yếu kém và không hoàn hảo, các phương pháp căn chỉnh ở quy mô con người hiện tại có thể được mở rộng sang các hệ thống siêu trí tuệ (superhuman systems). Giám sát có thể mở rộng (scalable oversight) và tổng quát hóa từ yếu sang mạnh (weak-to-strong generalization - W2SG) mang tính chất bổ trợ cho nhau. Giám sát có thể mở rộng (tranh biện, mô hình hóa phần thưởng đệ quy, phân rã tác vụ) giúp nâng cao năng lực hiệu dụng của bên giám sát (overseer) để có thể theo kịp mô hình chịu sự giám sát. W2SG đảm bảo mô hình mạnh tổng quát hóa chính xác từ bất kỳ tín hiệu giám sát không hoàn hảo nào mà bên giám sát cung cấp. Công trình "Debate Helps W2SG" (arXiv:2501.13124, tháng 1 năm 2025) đã kết hợp cả hai hướng tiếp cận này.

**Type:** Learn
**Languages:** Python (thư viện chuẩn, trình mô phỏng khoảng cách W2SG)
**Prerequisites:** Phase 18 · 01 (tuân thủ chỉ dẫn), Phase 18 · 10 (Kiểm soát AI - AI Control), Phase 09 (nền tảng học tăng cường - RL)
**Time:** ~60 phút

## Mục tiêu học tập

- Định nghĩa giám sát có thể mở rộng và tổng quát hóa từ yếu sang mạnh, đồng thời giải thích tính chất bổ trợ lẫn nhau của chúng.
- Mô tả thiết lập thực nghiệm của Burns và cộng sự năm 2023: tinh chỉnh GPT-4 bằng nhãn từ GPT-2.
- Giải thích chỉ số khoảng cách hiệu năng được thu hồi (Performance Gap Recovered - PGR) và ý nghĩa đo lường của nó.
- Trình bày ba cơ chế giám sát có thể mở rộng chính (tranh biện, mô hình hóa phần thưởng đệ quy, phân rã tác vụ) và một thế mạnh của mỗi cơ chế.

## Vấn đề đặt ra

Mọi kỹ thuật căn chỉnh đã học cho đến thời điểm này trong Giai đoạn 18 đều giả định rằng bên giám sát có đủ khả năng đánh giá hành vi của mô hình. Khi mô hình đạt tới mức siêu việt (superhuman), bên giám sát lại trở thành mắt xích yếu nhất. Câu hỏi cốt lõi của siêu căn chỉnh là: liệu một bên giám sát yếu hơn có thể tạo ra một mô hình mạnh hơn mà vẫn được căn chỉnh chuẩn xác một cách đáng tin cậy hay không?

Burns và các cộng sự đã đơn giản hóa vấn đề này thành một thiết lập thực nghiệm có thể vận hành được: dùng mô hình yếu để giám sát mô hình mạnh, đo lường tỷ lệ năng lực của mô hình mạnh được bảo tồn sau quá trình giám sát từ mô hình yếu. Đây không phải là giải pháp hoàn chỉnh cho bài toán siêu căn chỉnh — nó là một mô hình đại diện cho phép chúng ta đo lường tiến trình nghiên cứu.

## Khái niệm

### W2SG: Thiết lập thực nghiệm của Burns và cộng sự

- Mô hình yếu (Weak model): lớp mô hình tương đương GPT-2.
- Mô hình mạnh (Strong model): lớp mô hình tương đương GPT-4.
- Mục tiêu: mức trần năng lực của GPT-4 mạnh trên tác vụ mục tiêu.

Quy trình:
1. Thu thập các dự đoán zero-shot của mô hình yếu trên một tác vụ.
2. Tinh chỉnh (fine-tune) mô hình mạnh trên tập dữ liệu được gán nhãn bởi mô hình yếu.
3. Đo lường độ chính xác của mô hình mạnh sau khi tinh chỉnh.

Các mốc đối chứng (Baselines): độ chính xác của chính mô hình yếu, và mức trần được giám sát của mô hình mạnh trên các nhãn vàng chuẩn (gold labels).

Chỉ số khoảng cách: Khoảng cách hiệu năng được thu hồi (Performance Gap Recovered - PGR) = (điểm tinh chỉnh - điểm mô hình yếu) / (mức trần - điểm mô hình yếu). Giá trị PGR bằng 1,0 đồng nghĩa với việc sự giám sát từ mô hình yếu đã thu hẹp hoàn toàn khoảng cách; PGR bằng 0 đồng nghĩa với việc sự giám sát từ mô hình yếu không mang lại lợi ích nào.

### Các phát hiện thực nghiệm của Burns và cộng sự

Xuyên suốt các tác vụ NLP, giải đố cờ vua và mô hình hóa phần thưởng, chỉ số PGR luôn duy trì giá trị dương một cách nhất quán: mô hình mạnh có thể tổng quát hóa vượt ra ngoài những sai lầm của bên giám sát yếu. Tùy thuộc vào từng tác vụ cụ thể, PGR biến thiên từ ~20% đến ~80%. Mô hình mạnh "hiểu" được tác vụ dự định từ các nhãn không hoàn hảo, bằng cách bù đắp thông qua các tri thức tiên nghiệm (priors) đã được tiền huấn luyện của chính nó.

Những hạn chế mà Burns và các cộng sự ghi nhận:
- Khoảng cách giữa yếu và mạnh ở đây là khoảng cách về năng lực (capability), không phải là khoảng cách về sự căn chỉnh (alignment). Mô hình yếu vốn đã được căn chỉnh; trong khi một mô hình thực sự siêu việt có thể được căn chỉnh bởi con người nhưng với một khoảng cách năng lực lớn hơn rất nhiều.
- Khả năng tổng quát hóa của mô hình mạnh có thể bắt nguồn từ các tri thức tiên nghiệm của tác vụ nhiều hơn là việc thực sự khôi phục được ý định chuẩn xác của dữ liệu gốc.
- Việc khơi gợi tri thức tiềm ẩn (eliciting latent knowledge) của mô hình mạnh mới là vấn đề cốt lõi thực sự; PGR chỉ đo lường một sự vận hành cụ thể.

### Giám sát có thể mở rộng: Ba cơ chế

- **Tranh biện (Debate - Irving et al. 2018).** Hai thực thể của mô hình chưa đáng tin cậy U tranh luận ở hai lập trường đối lập; một thẩm phán yếu sẽ đưa ra phán quyết. Giả định ở đây là: việc tìm ra một câu trả lời đúng có sức thuyết phục sẽ dễ dàng hơn so với việc ngụy tạo một câu trả lời sai đầy thuyết phục. Các nghiên cứu gần đây trong giai đoạn 2024-2025 (Khan et al., Michael et al.) chỉ ra rằng tranh luận đôi khi mang lại hiệu quả, đôi khi không, và tính hữu ích của nó phụ thuộc chặt chẽ vào cấu trúc của tác vụ.
- **Mô hình hóa phần thưởng đệ quy (Recursive Reward Modeling - Leike et al. 2018).** Mô hình U giúp con người huấn luyện mô hình phần thưởng cho thế hệ tiếp theo U+1. Năng lực hiệu dụng của bên giám sát sẽ tăng trưởng song hành cùng với sự phát triển của mô hình.
- **Phân rã tác vụ (Task Decomposition - Christiano, Shlegeris, Amodei 2018).** Phân chia một nhiệm vụ khó khăn thành các nhiệm vụ phụ mà con người có thể kiểm tra được, thực hiện theo dạng đệ quy. Phương pháp này giả định rằng bài toán có tính chất phân rã được.

Mỗi cơ chế đều đặt ra những giả định nhất định về cấu trúc của tác vụ hoặc về mức độ căn chỉnh của các thành phần trung gian.

### Lý do giám sát có thể mở rộng và W2SG bổ trợ lẫn nhau

Giám sát có thể mở rộng nâng cao chất lượng tín hiệu hiệu dụng của bên giám sát.
W2SG thu hẹp khoảng cách từ bất kỳ tín hiệu không hoàn hảo nào mà bên giám sát có thể cung cấp.

Công trình của Lang và các cộng sự — "Debate Helps Weak-to-Strong Generalization" (arXiv:2501.13124) kết hợp cả hai yếu tố này: một giao thức tranh biện tạo ra các nhãn yếu có chất lượng tốt hơn, và mô hình mạnh được huấn luyện dựa trên các nhãn đó. Nghiên cứu báo cáo mức tăng PGR đáng kể trên các tác vụ xử lý ngôn ngữ tự nhiên (NLP).

### Bối cảnh biến động của tổ chức

Nhóm Siêu căn chỉnh (Superalignment) của OpenAI đã giải thể vào tháng 5 năm 2024 sau sự ra đi của Jan Leike sang Anthropic. Tuy nhiên, nghị trình nghiên cứu (giám sát có thể mở rộng, W2SG, nghiên cứu căn chỉnh tự động) vẫn được tiếp nối tại Anthropic và tại các phòng thí nghiệm học thuật — MATS (Bài học 28), Redwood (Bài học 10), Apollo (Bài học 8), METR (Bài học 28). Cấu trúc tổ chức có thể thay đổi, nhưng các câu hỏi nghiên cứu cốt lõi vẫn giữ nguyên giá trị.

### Vị trí của bài học trong Giai đoạn 18

Các bài học 6-10 mô tả mối đe dọa và mô hình phòng thủ dựa trên giả định rằng mô hình U không đáng tin cậy. Bài học 11 này là mô hình tấn công chủ động: làm cho bên giám sát đủ mạnh để xác minh sự căn chỉnh của U. Các bài học 12-16 sau đó sẽ chuyển hướng sang các bộ công cụ thực hành của việc đánh giá đối kháng.

```figure
scalable-oversight
```

## Sử dụng trong thực tế (Use It)

Tệp `code/main.py` mô phỏng quy trình tinh chỉnh W2SG trên một tác vụ tổng hợp. Trình gán nhãn yếu có độ chính xác 70% với các lỗi mang tính cấu trúc; mô hình mạnh có mức trần 95% trên nhãn vàng. Bạn sẽ tinh chỉnh mô hình mạnh trên các nhãn yếu, đo lường chỉ số PGR, và so sánh với mô hình mạnh được huấn luyện trên nhãn vàng cũng như mô hình yếu đứng riêng lẻ.

## Triển khai thành phẩm (Ship It)

Bài học này tạo ra tệp `outputs/skill-w2sg-pgr.md`. Khi được cung cấp mô tả về một thiết lập giám sát, tệp kỹ năng này sẽ xác định bên giám sát yếu, mô hình mạnh, chất lượng giám sát, và tính toán (hoặc yêu cầu) chỉ số PGR. Nó gắn cờ để phân biệt rõ liệu tuyên bố là "mô hình yếu có thể giám sát mô hình mạnh" hay "mô hình yếu kết hợp với cơ chế giám sát có thể giám sát mô hình mạnh."

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Báo cáo chỉ số PGR cho các mức `weak_accuracy` = 0.60, 0.70, 0.80. Giải thích hình dạng của đường cong PGR.

2. Chỉnh sửa trình gán nhãn yếu để nó mắc lỗi có cấu trúc (ví dụ: luôn đoán sai trên một lớp đầu vào cụ thể). Chỉ số PGR tăng lên, giảm đi, hay giữ nguyên? Giải thích nguyên nhân.

3. Đọc Mục 4.3 trong bài báo Burns et al. 2023 (về các tác vụ NLP). Tái hiện trực giác về "hàm mất mát phụ trợ theo độ tự tin" (confidence auxiliary loss): khi mô hình mạnh tự tin hơn nhãn của mô hình yếu, bên nào sẽ thắng?

4. Thiết kế một giao thức giám sát có thể mở rộng kết hợp giữa tranh biện và phân rã tác vụ cho một nhiệm vụ kỹ thuật phần mềm. Nêu rõ một chế độ thất bại của từng thành phần và giải thích cách mà sự kết hợp này giải quyết (hoặc không giải quyết được) từng thất bại đó.

5. Nêu rõ điều gì sẽ bác bỏ luận điểm "tổng quát hóa từ yếu sang mạnh là con đường khả thi dẫn đến siêu căn chỉnh". Hãy nêu cụ thể về dấu hiệu thực nghiệm (empirical signature) mà bạn cần phải quan sát thấy.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Giám sát có thể mở rộng (Scalable oversight) | "làm cho bên giám sát mạnh hơn" | Các cơ chế giúp tăng cường khả năng của bên giám sát trong việc đánh giá một mô hình có năng lực vượt trội hơn |
| W2SG | "yếu giám sát mạnh" | Tinh chỉnh mô hình mạnh trên nhãn của mô hình yếu và đo lường năng lực thu hồi được |
| PGR | "khoảng cách hiệu năng thu hồi" | (điểm tinh chỉnh - điểm yếu) / (mức trần - điểm yếu); 1.0 = đóng hoàn toàn khoảng cách, 0 = không giúp ích |
| Tranh biện (Debate) | "hai thực thể U tranh luận" | Cơ chế giám sát có thể mở rộng trong đó thẩm phán yếu chọn câu trả lời giữa hai bên bảo vệ U |
| RRM | "mô hình hóa phần thưởng đệ quy" | U hỗ trợ huấn luyện mô hình phần thưởng cho U+1; năng lực của bên giám sát bám sát U |
| Phân rã tác vụ (Task decomposition) | "con người kiểm tra tác vụ phụ" | Phân chia nhiệm vụ khó thành các nhiệm vụ phụ mà con người có thể xác minh được, theo cách đệ quy |
| Siêu căn chỉnh (Superalignment) | "căn chỉnh AI siêu trí tuệ" | Nghị trình nghiên cứu hướng tới việc căn chỉnh các mô hình mà con người không thể trực tiếp đánh giá |

## Đọc thêm (Further Reading)

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — bài báo nền tảng về W2SG
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — cơ chế tranh luận
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — mô hình hóa phần thưởng đệ quy
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — nghiên cứu thực nghiệm năm 2024 về tranh biện với các bên tranh luận mạnh hơn
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — nghiên cứu năm 2025 kết hợp giữa tranh biện và W2SG
