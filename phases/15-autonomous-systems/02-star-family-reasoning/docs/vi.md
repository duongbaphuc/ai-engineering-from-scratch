# STaR, V-STaR, Quiet-STaR — Khả năng Tự dạy Lập luận (Self-Taught Reasoning)

> Vòng lặp tự cải tiến (self-improvement loop) nhỏ nhất có thể tồn tại nằm ngay bên trong lập luận giải thích (rationale). Một mô hình sinh ra một chuỗi suy nghĩ (chain of thought), giữ lại những chuỗi dẫn tới câu trả lời đúng, và tinh chỉnh (fine-tune) dựa trên chúng. Đó chính là STaR. V-STaR bổ sung thêm một bộ xác minh (verifier) để việc lựa chọn tại thời điểm suy luận (inference-time selection) đạt hiệu quả cao hơn. Quiet-STaR đẩy việc lập luận xuống từng token đơn lẻ. Cả ba phương pháp đều hoạt động. Không có phương pháp nào mang phép màu — vòng lặp sẽ lưu giữ bất kỳ đường tắt (shortcut) nào vô tình dẫn đến câu trả lời đúng.

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Lập luận và CoT - Reasoning and CoT), Phase 15 · 01 (Định hình chân trời dài - Long-horizon framing)
**Time:** ~60 phút

## Vấn đề

Cách tiếp cận trực diện để dạy một mô hình lập luận là thu thập các dấu vết lập luận do con người viết ra. Việc đó tốn kém, chậm chạp và bị giới hạn bởi lượng chuỗi suy nghĩ chất lượng cao mà con người sẵn lòng tạo ra.

STaR (Self-Taught Reasoner - Mô hình tự dạy lập luận, Zelikman et al., 2022) đặt câu hỏi: điều gì sẽ xảy ra nếu mô hình tự viết ra lập luận giải thích (rationales) của chính nó và tự chấm điểm dựa trên các đáp án đã biết? Vòng lặp diễn ra như sau:

1. Lấy mẫu một dấu vết lập luận kèm câu trả lời.
2. Nếu câu trả lời cuối cùng là chính xác, giữ lại dấu vết lập luận đó.
3. Tinh chỉnh (fine-tune) mô hình trên các dấu vết được giữ lại.
4. Lặp lại quy trình.

Cách này phát huy tác dụng. GSM8K và CommonsenseQA đều được cải thiện mà không cần dữ liệu chú thích mới từ con người. Nhưng vòng lặp này có một độ chệch nội tại: bất kỳ lập luận nào dẫn đến câu trả lời đúng đều được giữ lại, bất kể bản thân lập luận logic đó có chặt chẽ hay không. V-STaR (Hosseini et al., 2024) khắc phục điều này bằng một bộ xác minh được học (learned verifier); Quiet-STaR (Zelikman et al., 2024) tổng quát hóa ý tưởng này thành các lập luận nội tại trên từng token.

## Khái niệm

### STaR: Khởi động tự thân (bootstrap) từ những gì hiệu quả

Bắt đầu từ một mô hình cơ sở có năng lực lập luận yếu. Trên mỗi bài toán huấn luyện, lấy mẫu một lập luận kèm câu trả lời. Nếu câu trả lời khớp với nhãn, giữ lại bộ ba (bài toán, lập luận, câu trả lời). Tinh chỉnh mô hình trên tập dữ liệu được giữ lại. Lặp lại quá trình.

Có một điểm mấu chốt quan trọng. Nếu mô hình không bao giờ giải đúng một bài toán, vòng lặp không thể học từ bài toán đó. STaR bổ sung kỹ thuật **hợp lý hóa (rationalization)**: đối với các bài toán mà mô hình thất bại, chèn câu trả lời đúng vào như một gợi ý và nhắc lại mô hình tạo ra lập luận dẫn đến câu trả lời đó. Các lập luận đã hợp lý hóa này được đưa vào tập huấn luyện.

Kết quả trong bài báo gốc (Zelikman et al., 2022): mô hình cơ sở GPT-J cải thiện trên GSM8K từ 5,8% lên 10,7% thông qua các vòng lặp STaR nhiều lần có hợp lý hóa — tăng khoảng 5 điểm phần trăm tuyệt đối. Trên CommonsenseQA, GPT-J 6B được huấn luyện bằng STaR đạt 72,5%, tương đương với GPT-3 175B được tinh chỉnh (~73%) — một mô hình lớn hơn khoảng 30 lần được huấn luyện trên các lập luận chú thích thủ công.

### V-STaR: Huấn luyện bộ xác minh bằng DPO

STaR loại bỏ các lập luận sai. Hosseini et al. (2024) nhận thấy rằng chúng cũng là dữ liệu: mỗi cặp (lập luận, "điều này có đúng không") đều có thể dùng để huấn luyện một bộ xác minh (verifier). Họ sử dụng Tối ưu hóa Tùy chọn Trực tiếp (Direct Preference Optimization - DPO) trên cả các giải pháp đúng và sai để xây dựng một bộ xếp hạng (ranker). Tại thời điểm suy luận, lấy mẫu N lập luận và chọn lập luận hàng đầu do bộ xác minh đánh giá.

Mức cải thiện được báo cáo: +4 đến +17 điểm phần trăm so với các đường cơ sở tự cải tiến trước đó trên GSM8K và MATH, với phần lớn mức tăng đến từ việc sử dụng bộ xác minh để lựa chọn tại thời điểm suy luận hơn là dùng để tinh chỉnh thêm bộ sinh (generator).

### Quiet-STaR: Lập luận nội tại trên từng token

Zelikman et al. (2024) đặt câu hỏi: điều gì sẽ xảy ra nếu mô hình học cách sinh ra một lập luận nội tại ngắn tại mỗi vị trí token, chứ không chỉ giữa bài toán và câu trả lời? Quiet-STaR huấn luyện mô hình phát ra một "suy nghĩ ẩn" trước mỗi token được dự đoán, sau đó kết hợp dự đoán có nhận thức suy nghĩ với dự đoán cơ sở thông qua một trọng số học được.

Kết quả: Mistral 7B đạt được những cải thiện zero-shot tuyệt đối trên GSM8K từ 5,9% lên 10,9% và CommonsenseQA từ 36,3% lên 47,2% mà không cần tinh chỉnh riêng cho từng tác vụ. Mô hình đã học được "khi nào cần suy nghĩ" — các token khó sẽ nhận được các lập luận nội tại dài hơn; các token dễ hầu như không có.

### Lý do cả ba phương pháp đều chia sẻ một mối lo ngại an toàn

Cả ba phương pháp đều sử dụng câu trả lời cuối cùng làm tín hiệu gradient. Một lập luận dẫn đến câu trả lời đúng thông qua suy luận sai lầm — khai thác đường tắt (shortcut), phỏng đoán ngẫu nhiên hoặc sử dụng một mẫu hình không có khả năng tổng quát hóa — vẫn nhận được củng cố tích cực. Trên các bài toán nội phân phối (in-distribution), đường tắt này hoạt động. Nhưng trên các bài toán ngoài phân phối (out-of-distribution - OOD), nó sẽ đổ vỡ trong thầm lặng.

Bộ xác minh của V-STaR giảm thiểu điều này bằng cách học cách xếp hạng lập luận, nhưng bộ xác minh cũng được huấn luyện trên cùng một tập nhãn. Nó có thể học cách ưu tiên những suy luận sai nhưng được định dạng đẹp mắt hơn là sự không chắc chắn trung thực. Thiết kế an toàn hơn là kết hợp dữ liệu kiểu STaR với (a) mô hình phần thưởng được giám sát theo tiến trình (process-supervised reward models - tưởng thưởng cho từng bước trung gian, không chỉ câu trả lời) và (b) đánh giá OOD độc lập giúp bẻ gãy các đường tắt đơn giản.

### So sánh

| Phương pháp | Tín hiệu huấn luyện | Chi phí suy luận | Lãng phí dữ liệu | Dạng lỗi đã biết |
|---|---|---|---|---|
| STaR | giữ (lập luận, câu trả lời) nếu đúng | 1x | loại bỏ toàn bộ lập luận sai | lập luận đường tắt (shortcut rationales) |
| STaR + hợp lý hóa | như trên + thử lại với gợi ý câu trả lời đúng | 1x | ít hơn | lập luận được hợp lý hóa có thể thiếu thuyết phục |
| V-STaR | STaR + bộ xác minh DPO từ cả hai lớp | Nx (tốt nhất trong N) | tối thiểu | bộ xác minh có thể củng cố sự tự tin sai lầm |
| Quiet-STaR | lập luận trên từng token + trọng số kết hợp | 1.5-3x | tối thiểu | vẫn phụ thuộc gradient điều kiện hóa theo câu trả lời |

### Vị trí của phương pháp trong ngăn xếp công nghệ năm 2026

STaR đã xuất hiện từ lâu. Nhưng mô hình này tái xuất hiện ở khắp mọi nơi trong giai đoạn 2025-2026. Học tăng cường (RL) trên các bài toán toán học có thể xác minh (DeepSeek-R1, Kimi-k1.5, o1) chính là tín hiệu gradient điều kiện hóa theo đáp án của STaR được mở rộng quy mô. Các mô hình phần thưởng tiến trình (Lightman et al., 2023; bài báo "Let's verify step by step" của OpenAI) là giải pháp thay thế được giám sát theo quy trình. AlphaEvolve (Bài 3) là STaR dành cho mã nguồn, với một bộ đánh giá chương trình thay thế cho nhãn. Cỗ máy Darwin Gödel (Bài 4) là STaR dành cho chính khung giàn (scaffolding) của tác tử.

Hiểu rõ STaR giúp ta thông suốt tất cả các khái niệm này. Đó là vòng lặp tự cải tiến khả thi tối thiểu (minimum-viable self-improvement loop).

```figure
reflection-loop
```

## Sử dụng

`code/main.py` chạy một vòng lặp STaR mô phỏng trên một tác vụ số học đơn giản. Bạn có thể quan sát:

- Độ chính xác tăng dần qua các vòng khởi động tự thân (bootstrap rounds) như thế nào.
- Các đường tắt lén lút xâm nhập ra sao: trình mô phỏng bao gồm một lớp lập luận "lười biếng" đưa ra câu trả lời đúng trong 40% trường hợp nhưng tổng quát hóa rất kém. Hãy theo dõi xem STaR có giữ lại chúng hay không.
- Một bộ xác minh (kiểu V-STaR) giúp ích ra sao khi suy luận nhưng không thể loại bỏ hoàn toàn các đường tắt được đưa vào trong quá trình huấn luyện.

## Đóng gói

`outputs/skill-star-loop-reviewer.md` giúp bạn kiểm toán một đường ống tự dạy lập luận dự kiến trước khi bắt đầu huấn luyện trên đó.

## Bài tập

1. Chạy trình mô phỏng. Đặt tần suất đường tắt về 0, sau đó thành 0,4. Độ chính xác cuối cùng phân kỳ bao nhiêu giữa hai lượt chạy, ngay cả khi cả hai đều đạt >90% trên phân phối huấn luyện?

2. Thêm một bài kiểm tra OOD độc lập vào trình mô phỏng. Lấy mẫu các bài toán từ một phân phối khác và đánh giá mô hình được khởi động tự thân trên cả tập nội phân phối và tập OOD. Định lượng khoảng cách này.

3. Đọc Mục 3 bài báo Quiet-STaR (arXiv:2403.09629). Giải thích token "kết thúc suy nghĩ" (end-of-thought token) và đầu trọng số kết hợp (mixing-weight head), mỗi khái niệm trong ba câu.

4. So sánh bộ lọc giữ-nếu-đúng (keep-if-correct) của STaR với phương án thay thế được giám sát theo tiến trình nhằm tưởng thưởng độc lập cho từng bước lập luận. Xác định sự khác biệt về chi phí dán nhãn và sự khác biệt về chất lượng hợp lý.

5. Thiết kế một bài đánh giá có thể phát hiện các lập luận đường tắt trong một mô hình đã triển khai. Đánh giá không nhất thiết phải hoàn hảo — nó chỉ cần bẻ gãy được các đường tắt đơn giản nhất mà vòng lặp STaR có xu hướng củng cố.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| STaR | "Mô hình tự dạy lập luận" | Tinh chỉnh trên các lập luận do mô hình tự sinh ra dẫn đến câu trả lời đúng; lặp lại quy trình |
| Hợp lý hóa (Rationalization) | "Thử lại có gợi ý" | Chèn câu trả lời đúng và nhắc lại để tìm lập luận cho các bài toán mà mô hình cơ sở giải sai |
| V-STaR | "STaR kèm bộ xác minh" | Huấn luyện DPO một bộ xác minh trên cả lập luận đúng và sai, dùng nó để lựa chọn khi suy luận |
| Quiet-STaR | "Lập luận trên từng token" | Sinh ra suy nghĩ ẩn tại mỗi vị trí token; kết hợp với dự đoán đường cơ sở |
| Gradient điều kiện hóa theo câu trả lời (Answer-conditioned gradient) | "Tín hiệu dựa trên kết quả" | Vòng lặp huấn luyện chỉ tưởng thưởng cho câu trả lời cuối cùng, không phải các bước lập luận |
| Mô hình phần thưởng tiến trình (Process reward model) | "Bộ xác minh cấp độ bước" | Mô hình phần thưởng được huấn luyện trên tính chính xác của từng bước, trái ngược với STaR |
| Lập luận đường tắt (Shortcut rationale) | "Đúng đáp án, sai suy luận" | Một lập luận đạt được nhãn đúng thông qua mẫu hình không tổng quát hóa; STaR vẫn giữ lại |

## Đọc thêm

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — bài báo gốc.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — bổ sung bộ xác minh DPO cho việc lựa chọn tại thời điểm suy luận.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — lập luận nội tại trên từng token.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — các mô hình phần thưởng tiến trình, tín hiệu gradient thay thế.
- [Bài báo DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL trên các tác vụ có thể xác minh, STaR được mở rộng quy mô cho huấn luyện biên giới.
