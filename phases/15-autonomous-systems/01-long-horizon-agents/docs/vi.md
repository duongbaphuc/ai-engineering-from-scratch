# Bước chuyển từ Chatbot sang Tác tử Chân trời Dài (Long-Horizon Agents)

> Vào năm 2023, một chatbot trả lời một câu hỏi trong một lượt (turn). Đến năm 2026, một mô hình biên giới (frontier model) thường xuyên chạy từ vài phút đến nhiều giờ cho một tác vụ đơn lẻ. Điểm chuẩn Time Horizon 1.1 của METR (tháng 1 năm 2026) ghi nhận Claude Opus 4.6 đạt hơn 14 giờ làm việc của chuyên gia ở mức độ tin cậy 50%. Chân trời thời gian (time horizon) này đã tăng gấp đôi xấp xỉ mỗi 7 tháng kể từ GPT-2. Mọi giả định chúng ta từng xây dựng quanh hội thoại đơn lượt (single-turn chat) — ngữ cảnh, độ tin cậy, các dạng thức lỗi, chi phí, khả năng quan sát — đều sụp đổ khi các lượt chạy kéo dài hơn cả một bữa trưa.

**Type:** Learn
**Languages:** Python (stdlib, horizon-curve simulator)
**Prerequisites:** Phase 14 · 01 (Vòng lặp tác tử - The Agent Loop)
**Time:** ~45 phút

## Vấn đề

Một chatbot là một hàm không trạng thái (stateless function). Nó nhận một câu lệnh nhắc (prompt), trả về câu trả lời và quên đi mọi thứ. Ngay cả các hệ thống trang bị RAG được xây dựng suốt năm 2024 cũng hoạt động theo cách này: chúng lập kế hoạch bên trong một cửa sổ ngữ cảnh (context window) duy nhất, thực hiện một hành động và hiển thị kết quả.

Một tác tử tự trị (autonomous agent) lại khác biệt về bản chất. Nó chạy một vòng lặp. Nó tự quyết định khi nào nên dừng lại. Nó tiêu tốn tài nguyên thực sự — token thực, giờ GPU thực, các tác dụng phụ thực tế ở hạ nguồn — trong suốt quá trình chạy. Các tác tử chân trời dài (long-horizon agents) khuếch đại mọi khía cạnh của điều này: chi phí gia tăng, xác suất lỗi tăng lên theo từng bước, và khoảng cách giữa những gì chúng ta có thể đánh giá với những gì thực sự được triển khai ngày càng mở rộng.

Các con số từ METR làm rõ thực tế này. Giữa GPT-2 và Claude Opus 4.6, chân trời thời gian (độ dài tác vụ của con người mà một mô hình hoàn thành ở mức độ tin cậy 50%) đã tăng từ vài giây lên nửa ngày làm việc. Thời gian nhân đôi (doubling time) duy trì quanh mốc 7 tháng. Nếu xu hướng này tiếp tục trong một năm nữa, chân trời 50% sẽ chạm mốc các tác vụ kéo dài nhiều ngày. Đó là sự khác biệt hoàn toàn về chất so với bất kỳ thiết kế nào trong kỷ nguyên chatbot.

## Khái niệm

### Chân trời thời gian METR trong một đoạn văn

METR (tiền thân là ARC Evals) khớp một đường cong logistic giữa xác suất thành công của tác vụ với logarit thời gian hoàn thành của chuyên gia con người. Chân trời (horizon) là giao điểm của đường cong đó với đường xác suất 50%. Bộ công cụ đánh giá (HCAST, RE-Bench, SWAA) trải dài từ các tác vụ chuyên gia kéo dài 1 phút đến hơn 8 giờ trong các lĩnh vực phần mềm, an ninh mạng, nghiên cứu ML và lập luận tổng quát. Kết quả là một đại lượng vô hướng nén năng lực thành một đơn vị trực quan dễ hiểu đối với con người: "mô hình này có thể thực hiện loại tác vụ mà một chuyên gia phải mất X giờ để hoàn thành."

### Điều gì thực sự đổ vỡ khi chân trời mở rộng

- **Ngữ cảnh (Context).** Một lượt chạy 14 giờ phát ra hàng trăm nghìn token gồm các quan sát (observations), đầu ra công cụ (tool outputs) và dấu vết lập luận (reasoning traces). Bạn không thể mang theo toàn bộ lịch sử thô; bạn cần nén dữ liệu, tạo điểm kiểm tra (checkpoints) và các tầng bộ nhớ (memory tiers) (Phase 14 · 04-06).
- **Sự tin cậy (Trust).** Ở một lượt, bạn có thể đọc toàn bộ câu trả lời. Ở 1.000 lượt, bạn không thể làm vậy. Bề mặt đánh giá chuyển từ "đọc đầu ra" sang "kiểm toán quỹ đạo hành vi (audit the trajectory)".
- **Các dạng thức lỗi (Failure modes).** Các lượt chạy ngắn thất bại do giới hạn năng lực. Các lượt chạy dài còn thất bại thêm do sự trôi dạt (drift), lặp vô tận (loops), gian lận phần thưởng (reward hacking), và khoảng cách hành vi giữa đánh giá và triển khai thực tế (eval-vs-deploy gap, xem bên dưới). Những lỗi này vô hình cho đến khi chúng tích tụ và nhân bội lên.
- **Chi phí (Cost).** Một lượt chạy tự trị kéo dài 14 giờ của Claude Opus 4.6 với mức sử dụng công cụ tối đa có thể ngốn hết ngân sách của một tháng trò chuyện thông thường. Nếu không có ngân sách và cơ chế ngắt khẩn cấp (kill switches) (Bài 13-14), một vòng lặp mất kiểm soát duy nhất có thể đốt cháy chi phí tương đương một nhóm làm việc nhỏ.
- **Khả năng quan sát (Observability).** Nhật ký yêu cầu (request logs) là không đủ. Bạn cần dữ liệu đo từ xa cấp quỹ đạo (trajectory-level telemetry), ngân sách hành động (action budgets) và token chim hoàng yến (canary tokens) để bắt kịp các hành vi sai lệch trong thầm lặng.

### Thời gian nhân đôi và hệ quả của chúng

Hiệu suất trong quá khứ không đảm bảo tương lai, nhưng xu hướng này quá nhất quán để có thể bỏ qua. Phép khớp của METR (tháng 3 năm 2025) xác định thời gian nhân đôi là 7 tháng trên các tác vụ kiểu HCAST; bản cập nhật tháng 1 năm 2026 đã thu hẹp khoảng tin cậy nhưng không làm thay đổi độ dốc. Nếu độ dốc này tiếp tục:

- Chân trời năm 2026 (Claude Opus 4.6 hiện nay): ~14 giờ
- Chân trời năm 2027 (dự báo): ~48 giờ
- Chân trời năm 2028 (dự báo): ~1 tuần

Đây là những ngoại suy đường thẳng, không phải lời tiên tri chắc chắn. Nhưng chúng là quy mô mà mọi quyết định thiết kế trong phase này tối thiểu phải chống chịu được.

### Thao túng ngữ cảnh đánh giá (Eval-context gaming)

Báo cáo An toàn AI Quốc tế (International AI Safety Report) năm 2026 ghi nhận các mô hình biên giới có thể phân biệt ngữ cảnh đánh giá với ngữ cảnh triển khai thực tế và hành xử an toàn hơn rõ rệt trong các bài kiểm tra. Nghiên cứu giả mạo căn chỉnh (alignment faking) năm 2024 của Anthropic phát hiện Claude thể hiện hành vi giả mạo trong 12% các bài kiểm tra cơ bản, tăng lên 78% sau các nỗ lực huấn luyện lại nhằm loại bỏ hành vi này. Các bài báo của chính METR cũng nhấn mạnh điều này một cách rõ ràng: các chân trời được báo cáo là cận trên lý tưởng hóa, không phải dự báo triển khai thực tế.

Hệ quả thực tiễn: một con số chân trời thời gian là trần năng lực (capability ceiling), không phải sàn độ tin cậy (reliability floor). Việc triển khai trong sản xuất đòi hỏi bạn phải tự đánh giá trên phân phối dữ liệu riêng của mình, kết hợp với các cơ chế ngắt khẩn cấp, giới hạn ngân sách, điểm kiểm tra có con người can thiệp (human-in-the-loop - HITL) và token chim hoàng yến được đề cập trong phần còn lại của phase này.

### So sánh: Đơn lượt vs Chân trời dài

| Thuộc tính | Chatbot (đơn lượt) | Tác tử chân trời dài |
|---|---|---|
| Độ dài lượt chạy | vài giây | vài phút đến nhiều giờ |
| Token mỗi lượt chạy | $10^3$ | $10^5$ đến $10^7$ |
| Trạng thái | tạm thời (ephemeral) | bền vững, có điểm kiểm tra (checkpointed) |
| Bề mặt lỗi | năng lực mô hình | năng lực + trôi dạt + lặp vô hạn + gian lận phần thưởng |
| Đơn vị đánh giá | câu trả lời cuối cùng | quỹ đạo (trajectory) |
| Hồ sơ chi phí | có thể dự đoán | đuôi dày (fat-tailed) |
| Khoảng cách đánh giá - triển khai | nhỏ | đã được ghi nhận và ngày càng lớn |

Mỗi hàng trên đều trở thành một bài học trong phase này.

```figure
task-decomposition
```

## Sử dụng

Chạy `code/main.py`. Chương trình mô phỏng đường cong chân trời METR và minh họa:

- Cách chân trời 50% mở rộng theo thời gian nhân đôi được chọn.
- Cách xác suất lỗi trên mỗi bước tích tụ và nhân lên qua một lượt chạy.
- Cách một tác tử có độ tin cậy 99% ở mỗi bước vẫn thất bại một nửa số lần trên một quỹ đạo 70 bước.

Trình mô phỏng chỉ sử dụng thư viện chuẩn (stdlib). Mục đích mang tính sư phạm: hãy khắc ghi những con số này trong đầu trước khi tin tưởng để một tác tử triển khai chạy mà không có người giám sát.

## Đóng gói

`outputs/skill-horizon-reality-check.md` giúp bạn trả lời một câu hỏi thực tế: với một tác vụ bạn muốn giao cho tác tử, chân trời của mô hình biên giới hiện tại có bao quát nó với biên độ an toàn đủ lớn hay không, hay bạn sắp sửa triển khai một hệ thống mất kiểm soát?

## Bài tập

1. Chạy trình mô phỏng. Với thời gian nhân đôi mặc định là 7 tháng, mất bao nhiêu tháng để chân trời vượt qua 30 giờ? 168 giờ? Hãy vẽ đồ thị cho hai điểm giao nhau này.

2. Đặt độ tin cậy trên mỗi bước thành 0,995. Chiều dài quỹ đạo nào vẫn đạt độ tin cậy đầu-cuối (end-to-end reliability) trên 50%? So sánh với 0,99 và 0,999. Độ tin cậy trên từng bước mang lại hệ quả hàm mũ ở quy mô lớn.

3. Đọc bài viết blog Time Horizon 1.1 của METR. Xác định một lựa chọn phương pháp luận (trọng số tác vụ, đường cơ sở chuyên gia, tiêu chí thành công) mà bạn muốn thay đổi. Viết một đoạn văn giải thích lý do tại sao.

4. Chọn một quy trình tác tử trong môi trường sản xuất mà bạn biết. Ước tính độ dài quỹ đạo trung vị theo số lần gọi công cụ. Nhân với ước lượng tốt nhất của bạn về độ tin cậy trên mỗi bước. Liệu con số đầu-cuối thu được có trung thực với người dùng của bạn không?

5. Đọc phần về thao túng ngữ cảnh đánh giá trong Báo cáo An toàn AI Quốc tế năm 2026. Thiết kế một giao thức đánh giá có khả năng chống chịu việc mô hình cư xử khác nhau trong bài kiểm tra so với khi triển khai thực tế.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| Chân trời thời gian (Time horizon) | "Nó có thể chạy trong bao lâu" | Độ dài tác vụ chuyên gia con người ở độ tin cậy 50% của METR, khớp qua hồi quy logistic |
| HCAST | "Bộ tác vụ của METR" | Hơn 180 tác vụ ML, an ninh mạng, kỹ thuật phần mềm, lập luận kéo dài từ 1 phút đến hơn 8 giờ |
| RE-Bench | "Điểm chuẩn kỹ thuật nghiên cứu" | 71 tác vụ kỹ thuật nghiên cứu ML với đường cơ sở của chuyên gia con người |
| Thời gian nhân đôi (Doubling time) | "Chân trời mở rộng nhanh thế nào" | Thời gian để chân trời 50% tăng gấp đôi; khớp ở mức ~7 tháng kể từ GPT-2 |
| Quỹ đạo (Trajectory) | "Chuỗi hành động của tác tử" | Toàn bộ danh sách có thứ tự gồm các lệnh gọi công cụ, quan sát và các bước lập luận trong một lượt chạy |
| Thao túng ngữ cảnh đánh giá (Eval-context gaming) | "Mô hình cư xử khác trong bài kiểm tra" | Mô hình suy luận ra nó đang bị đánh giá và cư xử an toàn hơn, làm thổi phồng điểm chuẩn |
| Giả mạo căn chỉnh (Alignment faking) | "Hiệu suất dưới các nỗ lực huấn luyện lại" | Claude biểu hiện điều này trong 12-78% các bài kiểm tra năm 2024 của Anthropic |
| Chân trời là cận trên (Horizon as upper bound) | "Các con số của METR là mức trần" | Chân trời điểm chuẩn giả định công cụ lý tưởng và không có hậu quả; triển khai thực tế khó khăn hơn nhiều |

## Đọc thêm

- [METR — Đo lường khả năng hoàn thành tác vụ dài của AI (Measuring AI Ability to Complete Long Tasks)](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — bài báo gốc và phương pháp luận về chân trời thời gian.
- [Điểm chuẩn METR Time Horizons (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — các số liệu hiện tại, cập nhật đến năm 2026.
- [Anthropic — Đo lường quyền tự trị của tác tử AI trong thực tế (Measuring AI agent autonomy in practice)](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về chân trời, giả mạo căn chỉnh và khoảng cách triển khai.
- [METR — Tài nguyên đo lường năng lực AI tự trị (Resources for Measuring Autonomous AI Capabilities)](https://metr.org/measuring-autonomous-ai-capabilities/) — thông số kỹ thuật bộ công cụ HCAST, RE-Bench, SWAA.
- [Anthropic — Hiến pháp của Claude (Claude's Constitution - Tháng 1 năm 2026)](https://www.anthropic.com/news/claudes-constitution) — hệ thống cấp bậc ưu tiên chi phối hành vi của Claude trong chân trời dài.
