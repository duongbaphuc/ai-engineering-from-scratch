# AlphaEvolve — Các Tác tử Lập trình Tiến hóa (Evolutionary Coding Agents)

> Ghép đôi một mô hình lập trình biên giới (frontier coding model) với một vòng lặp tiến hóa và một bộ đánh giá có thể kiểm tra bằng máy (machine-checkable evaluator). Để vòng lặp chạy đủ lâu. Nó phát hiện ra quy trình nhân ma trận phức 4x4 sử dụng 48 phép nhân vô hướng — cải tiến đầu tiên vượt qua giới hạn của Strassen sau 56 năm. Nó cũng tìm ra thuật toán phỏng đoán (heuristic) lập lịch Borg trên toàn Google, giúp thu hồi ~0,7% năng lực tính toán cụm máy chủ trong môi trường sản xuất thực tế. Kiến trúc này được cố tình thiết kế giản dị. Những thắng lợi đến từ tính nghiêm ngặt của bộ đánh giá.

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (Định hình chân trời dài - long-horizon framing), Phase 15 · 02 (Tự dạy lập luận - self-taught reasoning)
**Time:** ~60 phút

## Vấn đề

Các mô hình ngôn ngữ lớn có thể viết mã nguồn. Các thuật toán tiến hóa có thể tìm kiếm trên không gian mã nguồn. Cả hai đã được thử nghiệm riêng rẽ trong nhiều thập kỷ; cả hai đều chạm trần giới hạn. Trần giới hạn của LLM là sự bịa đặt (confabulation): mô hình viết ra đoạn mã trông có vẻ hợp lý nhưng không thực hiện đúng những gì nó tuyên bố. Trần giới hạn của thuật toán tiến hóa là chi phí tìm kiếm: các đột biến ngẫu nhiên trên cú pháp hiếm khi tạo ra các chương trình có thể biên dịch được, chưa nói đến việc tốt hơn.

AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, tháng 6 năm 2025) kết hợp cả hai. LLM đề xuất các chỉnh sửa có mục tiêu vào cơ sở dữ liệu chương trình; một bộ đánh giá tự động chấm điểm từng biến thể; các biến thể đạt điểm cao trở thành cha mẹ cho các thế hệ tương lai. LLM xử lý bước tốn kém là viết mã nguồn hợp lý; bộ đánh giá bắt kịp các lỗi bịa đặt. Vòng lặp chạy từ vài giờ đến vài tuần.

Các kết quả được báo cáo: phép nhân ma trận phức 4x4 với 48 phép nhân vô hướng (giới hạn năm 1969 của Strassen là 49), thuật toán phỏng đoán lập lịch Borg trong môi trường sản xuất của Google, tăng tốc hạt nhân (kernel) FlashAttention thêm 32,5%, và cải thiện thông lượng huấn luyện Gemini.

Kiến trúc này hoạt động hiệu quả vì bộ đánh giá có thể kiểm tra bằng máy (machine-checkable). Nó không hoạt động ở những nơi mà bộ đánh giá không thể làm điều đó. Sự bất đối xứng đó chính là bài học then chốt.

## Khái niệm

### Vòng lặp

1. Bắt đầu từ một chương trình hạt giống `P_0` chạy đúng nhưng chưa tối ưu.
2. Duy trì một cơ sở dữ liệu gồm các biến thể chương trình, mỗi biến thể được chấm điểm bởi bộ đánh giá.
3. Lấy mẫu một hoặc nhiều chương trình cha mẹ từ cơ sở dữ liệu (theo kiểu MAP-elites hoặc mô hình quần đảo - island-based).
4. Nhắc LLM (Gemini Flash cho số lượng lớn ứng viên, Gemini Pro cho các bài toán khó) tạo ra biến thể chỉnh sửa từ chương trình cha mẹ.
5. Biên dịch, chạy và đánh giá biến thể trên bộ đánh giá độc lập (held-out evaluator).
6. Chèn vào cơ sở dữ liệu được đánh chỉ mục theo điểm số và vector đặc trưng của nó.
7. Lặp lại quy trình.

Hai chi tiết mang tính quyết định. Thứ nhất, câu nhắc (prompt) cho LLM chứa nhiều thông tin hơn chỉ là chương trình cha mẹ — thường là vài biến thể hàng đầu từ cơ sở dữ liệu, kèm theo chữ ký của bộ đánh giá và một mô tả ngắn về tác vụ. Nhiệm vụ của mô hình là đề xuất một thay đổi có mục tiêu có khả năng cải thiện điểm số. Thứ hai, cơ sở dữ liệu được cấu trúc (lưới MAP-elites, phân tách theo đảo) để vòng lặp khám phá tính đa dạng, thay vì chỉ chăm chăm hội tụ vào ứng viên dẫn đầu hiện tại.

### Điều gì khiến bộ đánh giá trở nên không thể thương lượng

Các thành công của AlphaEvolve đều đến từ những lĩnh vực mà bộ đánh giá chạy nhanh, có tính tất định và khó bị đánh lừa:

- **Thuật toán nhân ma trận**: kiểm thử đơn vị (unit test) nhân các ma trận và kiểm tra tính tương đương tuyệt đối tới từng bit.
- **Thuật toán phỏng đoán lập lịch Borg**: trình mô phỏng cấp sản xuất phát lại tải cụm trong lịch sử và đo lường lượng tính toán bị lãng phí.
- **Hạt nhân FlashAttention**: kiểm tra tính đúng đắn kết hợp với điểm chuẩn thời gian thực (wall-clock benchmark) trên phần cứng thực tế.
- **Thông lượng huấn luyện Gemini**: đo lường số giây GPU trên mỗi bước bước huấn luyện.

Trong mỗi trường hợp, bộ đánh giá đều bắt được lớp lỗi điển hình của LLM mà nếu không có nó sẽ chiếm ưu thế tuyệt đối: tuyên bố đúng đắn bịa đặt, tuyên bố hiệu năng biến mất trên phần cứng thực tế, và các lỗi ở trường hợp biên (edge cases). Bỏ bộ đánh giá đi, vòng lặp sẽ chỉ tối ưu hóa cho những đoạn mã trông có vẻ đẹp mắt.

### Gian lận phần thưởng là mặt còn lại của nhận định đó

Tiến hóa sẽ tối ưu hóa cho bất kỳ thứ gì mà bộ đánh giá đo lường. Nếu bộ đánh giá không hoàn hảo, vòng lặp sẽ tìm ra điểm khiếm khuyết đó. Trong một miền chưa được xác minh, vòng lặp sẽ tối ưu hóa cho đặc trưng bề mặt chứ không phải hành vi mong muốn. DeepMind nêu rõ điều này trong bài báo: thành công của AlphaEvolve chỉ chuyển giao được sang các miền mà độ nghiêm ngặt của bộ đánh giá tương xứng với tham vọng của không gian tìm kiếm.

Các ví dụ cụ thể giai đoạn 2025-2026 về gian lận phần thưởng (reward hacking) trong các vòng lặp tìm kiếm mã nguồn:

- Các mục tiêu tối ưu tưởng thưởng cho "thời gian hoàn thành" dẫn đến việc hệ thống nộp các lời giải rỗng.
- Điểm chuẩn tưởng thưởng cho độ chính xác trong kiểm thử dẫn đến việc ghi nhớ mã kiểm thử và khớp quá mức (overfitting).
- Đại diện "chất lượng mã nguồn" dẫn đến việc xóa các chú thích và đổi tên biến mà không thay đổi bất kỳ ngữ nghĩa nào.

Giải pháp trong AlphaEvolve: sử dụng một bộ đánh giá độc lập (held-out evaluator) mà LLM chưa từng nhìn thấy, với các đầu vào được sinh ngẫu nhiên tại thời điểm đánh giá. Ngay cả khi đó, DeepMind vẫn khuyến nghị cần có sự rà soát kỹ lưỡng đối với bất kỳ đề xuất triển khai nào.

### Tại sao LLM + tìm kiếm lại vượt trội hơn từng phương pháp đơn lẻ

LLM có thể tạo ra các sửa đổi có thể biên dịch được và hợp lý về mặt ngữ nghĩa. Một thuật toán di truyền đột biến ngẫu nhiên trên một tệp Python 2000 dòng hầu như luôn tạo ra lỗi cú pháp. LLM cũng tập trung tìm kiếm vào các vùng lân cận hợp lý (thay đổi một hàm, chứ không phải các byte ngẫu nhiên), giúp giảm đáng kể số lần gọi bộ đánh giá bị lãng phí.

Đổi lại, bộ đánh giá sẽ bắt kịp các sự bịa đặt của LLM. LLM sẽ tự tin tuyên bố rằng một hàm "có độ phức tạp tiệm cận là $O(n \log n)$" trong khi thực tế nó là $O(n^2)$; một điểm chuẩn thời gian thực tế sẽ đưa ra câu trả lời dứt khoát.

### Vị trí của AlphaEvolve trong ngăn xếp công nghệ biên giới

| Hệ thống | Bộ sinh (Generator) | Bộ đánh giá (Evaluator) | Lĩnh vực | Thắng lợi tiêu biểu |
|---|---|---|---|---|
| AlphaEvolve | Gemini | tính đúng đắn + điểm chuẩn | thuật toán, hạt nhân, bộ lập lịch | nhân ma trận 4x4 với 48 phép nhân |
| FunSearch (DeepMind, 2023) | PaLM / Codey | tính đúng đắn | toán tổ hợp | chặn dưới tập nón (cap-set lower bounds) |
| AI Scientist v2 (Sakana, Bài 5) | GPT/Claude | phê bình LLM + thực nghiệm | nghiên cứu ML | bài báo hội thảo ICLR |
| Cỗ máy Darwin Gödel (Bài 4) | khung giàn tác tử | SWE-bench / Polyglot | mã tác tử | 20% → 50% SWE-bench |

Cả bốn hệ thống đều là biến thể của cùng một công thức: bộ sinh kết hợp bộ đánh giá, tạo thành vòng lặp. Sự khác biệt nằm ở chỗ bộ đánh giá chấm điểm cái gì và nó nghiêm ngặt đến mức độ nào.

```figure
alphaevolve-loop
```

## Sử dụng

`code/main.py` triển khai một vòng lặp tối giản kiểu AlphaEvolve trên bài toán hồi quy ký hiệu đơn giản. "LLM" ở đây là một bộ đại diện từ thư viện chuẩn đề xuất các đột biến cú pháp nhỏ cho một chương trình tính toán hàm mục tiêu. "Bộ đánh giá" đo lường sai số bình phương trung bình trên các điểm kiểm tra độc lập.

Hãy quan sát:

- Điểm số tốt nhất cải thiện qua các thế hệ như thế nào.
- Cách một lưới MAP-elites duy trì sự tồn tại của các giải pháp đa dạng để vòng lặp không bị hội tụ sớm vào cực tiểu cục bộ.
- Việc loại bỏ tập kiểm tra độc lập (chỉ đánh giá trên tập huấn luyện) sẽ khiến vòng lặp khớp quá mức (overfit) nghiêm trọng ra sao.

## Đóng gói

`outputs/skill-evaluator-rigor-audit.md` là điều kiện tiên quyết trước khi cân nhắc áp dụng vòng lặp kiểu AlphaEvolve cho một lĩnh vực mới: liệu bộ đánh giá của bạn có thực sự bắt được những lỗi mà bạn quan tâm hay không?

## Bài tập

1. Chạy `code/main.py`. Ghi lại quỹ đạo điểm số tốt nhất. Vô hiệu hóa bộ đánh giá độc lập (cờ `--no-holdout`) và chạy lại. Định lượng hiện tượng khớp quá mức.

2. Đọc Mục 3 của bài báo AlphaEvolve về lưới MAP-elites. Thiết kế một bộ mô tả vector đặc trưng cho một bài toán mới (ví dụ: các lượt tối ưu hóa trình biên dịch) để duy trì tính đa dạng trong không gian tìm kiếm.

3. Kết quả 48 phép nhân cho ma trận phức 4x4 đã cải tiến giới hạn 49 phép nhân của Strassen sau 56 năm. Đọc Phụ lục F của bài báo và giải thích trong ba câu lý do tại sao bộ đánh giá cho bài toán này đặc biệt dễ thiết kế chính xác, và tại sao hầu hết các lĩnh vực khác lại không như vậy.

4. Đề xuất một lĩnh vực mà AlphaEvolve sẽ thất bại. Xác định chính xác bộ đánh giá sẽ bị phá vỡ ở đâu và tại sao.

5. Đối với một lĩnh vực mà bạn biết rõ, hãy viết chữ ký bộ đánh giá mà bạn sẽ sử dụng. Bao gồm (a) các điều kiện về tính đúng đắn, (b) chỉ số hiệu năng, (c) quy tắc sinh đầu vào độc lập, (d) ít nhất một kiểm tra chống gian lận phần thưởng.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| AlphaEvolve | "Tác tử lập trình tiến hóa của DeepMind" | Gemini + cơ sở dữ liệu chương trình + bộ đánh giá có thể kiểm tra bằng máy |
| MAP-elites | "Kho lưu trữ bảo tồn tính đa dạng" | Lưới đánh chỉ mục theo vector đặc trưng; mỗi ô giữ biến thể tốt nhất có bộ mô tả đó |
| Mô hình quần đảo (Island model) | "Các tiểu quần thể tiến hóa song song" | Các quần thể độc lập di cư định kỳ; ngăn chặn sự hội tụ sớm |
| Bộ đánh giá kiểm tra bằng máy (Machine-checkable evaluator) | "Tiên tri tất định" | Một kiểm thử đơn vị, trình mô phỏng hoặc điểm chuẩn mà LLM không thể làm giả — điều kiện tiên quyết cho vòng lặp này |
| Gian lận phần thưởng (Reward hacking) | "Tối ưu hóa thước đo thay vì mục tiêu" | Vòng lặp tìm cách tối đa hóa điểm số mà không thực hiện đúng tác vụ mong muốn |
| Chương trình hạt giống (Seed program) | "Điểm khởi đầu" | Một chương trình ban đầu chạy đúng nhưng chưa tối ưu để vòng lặp tiến hóa phát triển từ đó |
| Bộ đánh giá độc lập (Held-out evaluator) | "Dữ liệu đánh giá LLM chưa từng thấy" | Các đầu vào được sinh tại thời điểm đánh giá để ngăn chặn việc học vẹt |

## Đọc thêm

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — bài báo đầy đủ.
- [DeepMind blog về AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — bài viết giới thiệu của DeepMind cùng kết quả.
- [Kho lưu trữ kết quả AlphaEvolve](https://github.com/google-deepmind/alphaevolve_results) — các thuật toán được phát hiện, bao gồm phép nhân ma trận 4x4 với 48 phép nhân.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — hệ thống tiền thân.
- [Anthropic — Chính sách mở rộng có trách nhiệm v3.0 (Responsible Scaling Policy v3.0 - Tháng 2 năm 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — định hình quyền tự trị bị ràng buộc bởi bộ đánh giá như một hướng nghiên cứu then chốt.
