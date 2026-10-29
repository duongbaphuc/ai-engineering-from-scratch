# Đánh giá và Các Bộ Đo chuẩn Phối hợp (Evaluation and Coordination Benchmarks)

> Năm bộ đo chuẩn (benchmarks) giai đoạn 2025-2026 bao quát toàn bộ không gian đánh giá đa tác tử. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) đánh giá các cấu trúc liên kết (topologies) hình sao/chuỗi/cây/đồ thị với các chỉ số đo lường hiệu quả (KPIs) dựa trên cột mốc; **cấu trúc đồ thị (graph) tối ưu nhất cho nghiên cứu**, trong khi việc lập kế hoạch nhận thức bổ sung ~3% mức độ hoàn thành cột mốc. **COMMA** đánh giá khả năng phối hợp đa phương thức với thông tin bất đối xứng; các mô hình hàng đầu bao gồm GPT-4o còn phải chật vật mới vượt qua được mô hình cơ sở ngẫu nhiên (random baseline). **MedAgentBoard** (arXiv:2505.12371) kiểm thử trên bốn nhóm tác vụ y tế và chỉ ra rằng hệ đa tác tử thường không vượt trội hơn một LLM đơn lẻ. **AgentArch** (arXiv:2509.10769) đánh giá chuẩn mực các kiến trúc tác tử doanh nghiệp kết hợp sử dụng công cụ + bộ nhớ + điều phối. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) gồm 1.865 bài toán trên 41 kho mã nguồn trải rộng từ ứng dụng doanh nghiệp, dịch vụ B2B đến các công cụ cho lập trình viên; các mô hình tiên phong chỉ đạt ~23% trên bản Pro so với mức 70%+ trên bản Verified — một sự kiểm chứng thực tế về hiện tượng nhiễm bẩn dữ liệu huấn luyện (data contamination). Claude Opus 4.7 (tháng 4 năm 2026) được ghi nhận đạt **64,3%** trên Pro khi có sự phối hợp tường minh giữa các nhóm tác tử (chưa có nguồn chính thức của Anthropic công bố — cần xem đây là kết quả sơ bộ); Verdent (bộ khung tác tử) đạt **76,1% pass@1** trên bản Verified ([báo cáo kỹ thuật của Verdent](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **Chương trình Cầu nối AAAI 2026 WMAC** (https://multiagents.org/2026/) là tâm điểm học thuật của cộng đồng năm 2026. Bài học này dựa trên các chỉ số của MARBLE, thực hiện khảo sát so sánh cấu trúc liên kết và chỉ số, đồng thời xác lập quy tắc: "chỉ vượt qua SWE-bench Verified chưa phải là bằng chứng cho khả năng khái quát hóa".

**Type:** Learn  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)  
**Time:** ~75 phút  

## Vấn đề (Problem)

Khi một bài báo tuyên bố "hệ đa tác tử của chúng tôi vượt trội hơn", câu hỏi đặt ra là: Vượt trội hơn cái gì, trên tác vụ nào, và được đo lường ra sao? Kỷ nguyên đánh giá đa tác tử 2023-2024 từng rơi vào tình trạng hỗn loạn — mỗi nhóm nghiên cứu tự chọn chỉ số riêng, tự đặt mô hình cơ sở (baseline) riêng và tự tạo tập dữ liệu tác vụ riêng. Các bộ đo chuẩn 2025-2026 đã tái lập trật tự kỷ luật cho lĩnh vực này.

Nếu thiếu vắng các bộ chuẩn chung, bạn không thể so sánh hai hệ đa tác tử một cách có ý nghĩa. Tệ hơn nữa, nếu không có các bộ kiểm thử ẩn giữ lại (hold-out benchmarks), các mô hình biên (frontier models) sẽ dễ bị rò rỉ dữ liệu kiểm chuẩn vào tập huấn luyện (nhiễm bẩn dữ liệu - data contamination). SWE-bench Verified đã bị nhiễm bẩn một phần trong kho ngữ liệu huấn luyện vào giữa năm 2025; điểm số của các mô hình tiên phong bị thổi phồng; SWE-bench Pro ra đời như một liều thuốc kiểm chứng thực tế không bị nhiễm bẩn.

Bài học này điểm qua năm bộ đo chuẩn kinh điển của năm 2026, làm rõ đối tượng đo lường của từng bộ chuẩn, và rèn luyện cho bạn thói quen đọc các tuyên bố thành tích với con mắt hoài nghi khoa học.

## Khái niệm (Concept)

### MultiAgentBench (MARBLE) — ACL 2025

Nghiên cứu arXiv:2503.01935. Đánh giá bốn cấu trúc liên kết phối hợp (hình sao - star, chuỗi - chain, cây - tree, đồ thị - graph) trên các tác vụ nghiên cứu, lập trình và lập kế hoạch. Các chỉ số KPI dựa trên cột mốc (milestone-based KPIs) giúp theo dõi tiến độ từng phần thay vì chỉ đánh giá nhị phân thành công/thất bại cuối cùng.

Kết quả đo đạc thực tế:

- Cấu trúc **Đồ thị (Graph)** tối ưu nhất cho các kịch bản nghiên cứu; hỗ trợ phản biện tự do giữa mọi tác tử (any-to-any critique).
- Cấu trúc **Chuỗi (Chain)** tối ưu nhất cho bài toán lập trình tinh chỉnh từng bước.
- Cấu trúc **Hình sao (Star)** tối ưu nhất cho việc tổng hợp dữ kiện nhanh chóng.
- **Thuế phối hợp (Coordination tax)** bắt đầu xuất hiện khi quy mô vượt quá ~4 tác tử trong cấu trúc đồ thị.
- **Lập kế hoạch nhận thức (Cognitive planning)** giúp tăng thêm ~3% mức độ hoàn thành cột mốc trên mọi cấu trúc liên kết.

Khi nào nên dùng: Khi bạn muốn so sánh công bằng giữa các mô hình cấu trúc liên kết phối hợp. Kho mã nguồn MARBLE (https://github.com/ulab-uiuc/MARBLE) cung cấp sẵn bộ thẩm định đánh giá.

### COMMA — Thông tin Bất đối xứng Đa phương thức

Đánh giá các tác vụ trong đó các tác tử sở hữu các phương thức quan sát khác nhau (hình ảnh, văn bản, âm thanh) và bắt buộc phải phối hợp mà không thể chia sẻ toàn bộ thông tin cho nhau. Kết quả công bố đem lại một thực tế phũ phàng: Các mô hình tiên phong bao gồm cả GPT-4o còn phải chật vật mới vượt qua được **mô hình cơ sở ngẫu nhiên (random baseline)** trong việc hợp tác giữa tác tử với tác tử trên COMMA. Tín hiệu này cho thấy sự phối hợp đa tác tử trên các phương thức quan sát khác nhau hiện chưa được huấn luyện và đánh giá đầy đủ — các LLM xử lý hợp tác đơn phương thức khá tốt; nhưng khi bước sang phối hợp đa phương thức thì hệ thống nhanh chóng sụp đổ.

Khi nào nên dùng: Khi hệ thống của bạn có sự phối hợp đa phương thức hoặc tồn tại thông tin bất đối xứng. Kết quả rỗng (null result) từ COMMA là lời cảnh tỉnh nghiêm khắc: Hãy đo đạc kỹ lưỡng trước khi đưa ra bất kỳ tuyên bố nào.

### MedAgentBoard — Bài kiểm tra Căng thẳng theo Miền chuyên ngành

Nghiên cứu arXiv:2505.12371. Bao gồm bốn danh mục tác vụ y tế: chẩn đoán, lập kế hoạch điều trị, lập báo cáo y khoa, và giao tiếp với bệnh nhân. So sánh hiệu năng giữa hệ đa tác tử, một LLM đơn lẻ và các hệ thống dựa trên luật truyền thống.

Phát hiện: Hệ đa tác tử KHÔNG hề áp đảo một LLM đơn lẻ trên phần lớn các danh mục. Lợi thế của đa tác tử là rất hẹp — việc phân rã tác vụ chỉ thực sự phát huy tác dụng khi các nhiệm vụ con có thể tách bạch rành mạch (như chẩn đoán kết hợp điều trị); nó sẽ phản tác dụng khi chi phí phối hợp vượt quá lợi ích của việc chuyên môn hóa (như trong tác vụ viết báo cáo y khoa).

Khi nào nên dùng: Khi miền nghiệp vụ của bạn có các mô hình cơ sở LLM đơn lẻ rõ ràng. Nếu bài học từ MedAgentBoard mang tính phổ quát, rất nhiều hệ đa tác tử hiện nay đang bị thiết kế phức tạp hóa quá mức cần thiết.

### AgentArch — Kiến trúc Doanh nghiệp

Nghiên cứu arXiv:2509.10769. Đánh giá bối cảnh doanh nghiệp nơi việc sử dụng công cụ, bộ nhớ và điều phối được xếp lớp cùng nhau. Bộ chuẩn này cô lập mức đóng góp của từng lớp riêng biệt: Việc bổ sung thêm công cụ giúp cải thiện bao nhiêu? Bổ sung thêm bộ nhớ giúp ích thế nào? Và việc điều phối đa tác tử mang lại giá trị gia tăng ra sao?

Khi nào nên dùng: Khi bạn đang thiết kế một ngăn xếp tác tử cho doanh nghiệp và cần chứng minh giá trị của từng tầng công nghệ. AgentArch giúp bạn tránh bỏ tiền mua những tính năng phức tạp mà bản thân không thể đo lường được hiệu quả.

### SWE-bench Pro — Sự Kiểm chứng Thực tế

Nghiên cứu arXiv:2509.16941. Bao gồm 1.865 bài toán thực tế trên 41 kho mã nguồn trải rộng khắp các ứng dụng thương mại, dịch vụ B2B và công cụ dành cho lập trình viên. Được thiết kế với mục tiêu **chống nhiễm bẩn dữ liệu** nhờ mốc thời gian cắt dữ liệu huấn luyện muộn hơn. Các mô hình biên chỉ đạt ~23% trên bản Pro so với mức 70%+ trên bản Verified. Khoảng cách chênh lệch này chính là minh chứng rõ nét cho hiện tượng nhiễm bẩn dữ liệu.

Điểm số cập nhật tháng 4 năm 2026:
- Claude Opus 4.7 trên bản Pro: **64,3%** (được ghi nhận khi áp dụng điều phối nhóm tác tử tường minh; chưa có công bố gốc chính thức từ Anthropic — cần xem là dữ liệu sơ bộ).
- Verdent (bộ khung tác tử) trên bản Verified: **76,1% pass@1** ([báo cáo kỹ thuật](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Điểm thô của các mô hình biên trên Pro khi không dùng khung tác tử: ~23-35% ([bài báo SWE-bench Pro](https://arxiv.org/abs/2509.16941)).

Bài học then chốt: Việc tuyên bố "chúng tôi đánh bại SWE-bench Verified" không còn là bằng chứng đáng tin cậy về năng lực thực thụ. Bản Pro mới là thước đo cổng kiểm soát hiện tại. Bộ khung phối hợp nhóm tác tử mang lại bước nhảy vọt đo đạc được trên bản Pro (chênh lệch ~30-40 điểm phần trăm), và đây chính là một trong những luận cứ thực nghiệm vững chắc nhất cho sự phối hợp đa tác tử trong năm 2026.

### AAAI 2026 WMAC

Chương trình Cầu nối Hội nghị AAAI 2026 — Hội thảo về Phối hợp Đa Tác tử (WMAC - Workshop on Multi-Agent Coordination, https://multiagents.org/2026/). Đây là tâm điểm học thuật của cộng đồng nghiên cứu AI đa tác tử năm 2026. Các bài báo được chấp nhận và kỷ yếu hội thảo là địa chỉ tham chiếu uy tín nhất để đánh giá các phương pháp mới; hãy ưu tiên các kết quả được chấp nhận tại WMAC hơn là các bản thảo chưa qua bình duyệt trên arXiv khi đưa ra các quyết định kỹ thuật cho môi trường sản xuất.

### Đọc các tuyên bố thành tích với sự hoài nghi — Danh mục kiểm tra 2026

Khi ai đó công bố một kết quả đa tác tử:

1. **Bộ chuẩn nào, phân đoạn nào?** SWE-bench Verified hay SWE-bench Pro là sự khác biệt một trời một vực. Một con số công bố trên sai phân đoạn hoàn toàn không có giá trị.
2. **Kiểm tra nhiễm bẩn dữ liệu:** Bộ chuẩn có được phát hành sau mốc cắt dữ liệu huấn luyện của mô hình hay không? Nếu không, hãy hết sức thận trọng.
3. **So sánh với mô hình cơ sở:** So với mô hình LLM đơn lẻ, so với lựa chọn ngẫu nhiên, hay so với các công trình đa tác tử đi trước? Không chấp nhận so sánh kiểu "đối chiếu với phiên bản chưa tinh chỉnh của chính hệ thống đó".
4. **Ý nghĩa thống kê:** Đã thử nghiệm bao nhiêu lượt ($N$ trials), chỉ số p-value, khoảng tin cậy ra sao? Các mô hình biên có phương sai rất lớn; kết quả của một lượt chạy đơn lẻ rất dễ gây hiểu lầm.
5. **Tính đa dạng của tác vụ:** Thử nghiệm trên một tác vụ đơn lẻ hay trên nhiều tác vụ khác nhau? Khả năng khái quát hóa là yếu tố sống còn cho sản xuất.
6. **Minh bạch về chi phí:** Lượng token tiêu thụ trên mỗi tác vụ, thời gian thực thi thực tế. Một giải pháp đạt 90% nhưng tốn chi phí gấp 20 lần là một bài toán đánh đổi kinh doanh chứ không đơn thuần là tuyên bố về năng lực thuật toán.

### Những khía cạnh chưa có bộ chuẩn nào đo đạc tốt

- **Sự phối hợp trên tầm chân trời dài:** Các tương tác kéo dài nhiều ngày liền. Tất cả các bộ chuẩn hiện tại đều dừng lại ở thời gian ngắn.
- **Khả năng chống chịu đối kháng (Adversarial resilience):** Điều gì xảy ra khi một tác tử bị xâm nhập hoặc có hành vi ác ý?
- **Sự trôi dạt sau triển khai:** Các bộ chuẩn đều mang tính tĩnh; trong khi phân phối dữ liệu ngoài đời thực liên tục biến thiên.
- **Hiệu năng chuẩn hóa theo chi phí:** Phần lớn các bộ chuẩn chỉ báo cáo độ chính xác thô mà không đo lường độ chính xác trên từng đồng chi phí bỏ ra.

Việc tự xây dựng một bộ đo chuẩn nội bộ bám sát trục đo lường mà bạn thực sự quan tâm thường là bước đi đúng đắn nhất.

````figure
a5-bench-gap
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` là một bản hướng dẫn mã nguồn hoàn chỉnh:

- Mô phỏng 3 hệ đa tác tử trên một tác vụ đồ chơi.
- Tính toán các chỉ số cột mốc theo phong cách MARBLE cho từng hệ thống.
- Thực hiện kiểm tra nhiễm bẩn dữ liệu bằng cách giữ lại các tác vụ độc lập khỏi tập "huấn luyện".
- So sánh tường minh với một mô hình cơ sở ngẫu nhiên.
- In ra một bảng điểm đánh giá các tuyên bố thành tích của bộ chuẩn.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả kỳ vọng: Bảng điểm hệ thống chi tiết gồm độ chính xác thô, mức độ hoàn thành cột mốc, chi phí trên từng tác vụ, mức chênh lệch so với mô hình ngẫu nhiên và ghi chú kiểm tra nhiễm bẩn dữ liệu.

## Ứng dụng thực tế (Use It)

Tài liệu `outputs/skill-benchmark-reader.md` hướng dẫn cách phân tích bất kỳ tuyên bố thành tích đo chuẩn đa tác tử nào dựa trên danh mục kiểm tra nghiêm ngặt. Đầu ra: Điểm xếp loại cùng các cảnh báo rủi ro cụ thể.

## Triển khai thực tế (Ship It)

Kỷ luật đánh giá trong môi trường sản xuất:

- **Xây dựng bộ chuẩn nội bộ** phản ánh chính xác phân phối dữ liệu thực tế của bạn. Các bộ chuẩn công cộng chỉ có giá trị tham khảo chứ không thể thay thế hoàn toàn.
- **Luôn đưa mô hình ngẫu nhiên vào mọi phép so sánh.** Nếu bạn không thể đánh bại lựa chọn ngẫu nhiên với khoảng cách lớn trong một tác vụ phối hợp, rất có thể bản thân bài toán đã được đặt ra không chuẩn.
- **Báo cáo chi phí song song với độ chính xác.** Chi phí token và thời gian thực thi thực tế. Đội ngũ vận hành cần cả hai số liệu này.
- **Tái tạo bộ chuẩn định kỳ mỗi quý:** Phân phối thực tế luôn dịch chuyển; việc dựa vào các bộ chuẩn lỗi thời sẽ dẫn đến những quyết định sai lầm.
- **Tránh khớp quá mức (overfitting) vào các bộ chuẩn đã công bố:** Nếu đội ngũ của bạn chỉ chăm chăm tối ưu hóa điểm số SWE-bench Pro, chất lượng hệ thống trên thực tế chắc chắn sẽ bị thoái hóa.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Xác định xem hệ thống nào trong số 3 hệ thống mô phỏng đạt chi phí trên từng cột mốc tốt nhất. Nó có trùng khớp với hệ thống có độ chính xác thô cao nhất không?
2. Đọc bài báo MultiAgentBench (arXiv:2503.01935). Đối với miền tác vụ thực tế của bạn, cấu trúc liên kết nào trong số 4 dạng mà MARBLE sẽ khuyến nghị? Hãy lập luận dựa trên các kết quả thực nghiệm của bài báo.
3. Đọc bài báo SWE-bench Pro. Yếu tố kỹ thuật cụ thể nào giúp nó có khả năng kháng nhiễm bẩn dữ liệu? Phương pháp đó có thể áp dụng cho các bộ chuẩn khác mà bạn quan tâm không?
4. Đọc các phát hiện của COMMA về sự phối hợp đa phương thức. Hãy thiết kế một tác vụ phối hợp đa phương thức đơn giản mà bạn có thể tích hợp vào bộ chuẩn nội bộ của mình. Tiêu chí nào sẽ được coi là một tín hiệu đánh giá hữu ích?
5. Áp dụng bảng danh mục kiểm tra vào kết quả nổi bật của một bài báo đa tác tử gần đây. Bạn sẽ chấm điểm tuyên bố thành tích đó ở mức nào?

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| MARBLE | "MultiAgentBench" | Công bố tại ACL 2025; đánh giá cấu trúc sao/chuỗi/cây/đồ thị với chỉ số cột mốc. |
| COMMA | "Bộ chuẩn đa phương thức" | Đánh giá phối hợp đa phương thức với thông tin bất đối xứng; các mô hình tiên phong chật vật so với ngẫu nhiên. |
| MedAgentBoard | "Kiểm thử căng thẳng theo miền" | Bốn danh mục y tế; thường chỉ ra đa tác tử không áp đảo được một LLM đơn lẻ. |
| AgentArch | "Bộ chuẩn doanh nghiệp" | Đánh giá phân tầng việc kết hợp công cụ + bộ nhớ + điều phối. |
| SWE-bench Pro | "Bộ chuẩn kháng nhiễm bẩn" | 1.865 bài toán, 41 kho mã nguồn; đạt ~23% so với 70%+ trên Verified (tín hiệu nhiễm bẩn). |
| Hoàn thành cột mốc (Milestone achievement) | "Điểm thành phần từng phần" | Cơ chế đánh giá ghi nhận tiến độ thực hiện chứ không chỉ xét kết quả nhị phân cuối cùng. |
| Nhiễm bẩn dữ liệu (Data contamination) | "Đề thi bị lộ vào dữ liệu học" | Sau khi phát hành, bộ chuẩn bị rò rỉ vào kho ngữ liệu huấn luyện; điểm số bị thổi phồng. |
| WMAC | "Chương trình Cầu nối AAAI 2026" | Hội thảo về Phối hợp Đa Tác tử; tâm điểm học thuật của cộng đồng. |

## Tài liệu tham khảo (Further Reading)

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — Bộ đo chuẩn cấu trúc liên kết với KPI dựa trên cột mốc.
- [Kho mã nguồn MARBLE](https://github.com/ulab-uiuc/MARBLE) — Triển khai mã nguồn tham chiếu.
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — Kiểm tra căng thẳng theo miền chuyên ngành; đa tác tử thường không vượt trội.
- [AgentArch](https://arxiv.org/abs/2509.10769) — Kiến trúc tác tử doanh nghiệp.
- [Bảng xếp hạng SWE-bench](https://www.swebench.com/) — Điểm số Verified và Pro cho các mô hình tiên phong.
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — Tâm điểm hội thảo học thuật của cộng đồng năm 2026.
