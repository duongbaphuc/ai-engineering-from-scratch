# Cơ chế Đồng thuận và Khả năng Chịu lỗi Byzantine cho các Tác tử (Consensus and Byzantine Fault Tolerance for Agents)

> Giao điểm giữa khả năng chịu lỗi Byzantine (BFT) trong hệ phân tán cổ điển và tính ngẫu nhiên của các mô hình ngôn ngữ lớn (LLMs). Trong giai đoạn 2025–2026, ba hướng nghiên cứu nổi bật đã xuất hiện: **CP-WBFT** (arXiv:2511.10400) đánh trọng số từng phiếu bầu dựa trên phép thăm dò độ tin cậy (confidence probe); **DecentLLMs** (arXiv:2507.14928) loại bỏ nút chủ (leaderless) với các đề xuất song song từ tác tử thợ và gom cụm bằng trung vị hình học (geometric-median aggregation); **WBFT** (arXiv:2505.05103) kết hợp bỏ phiếu có trọng số với Phân cụm Cấu trúc Phân cấp (Hierarchical Structure Clustering) để phân tách các nút Lõi (Core) và Biên (Edge). Kết quả thực nghiệm trung thực từ nghiên cứu "Can AI Agents Agree?" (arXiv:2603.01213) chỉ ra rằng ngay cả việc đồng thuận trên giá trị vô hướng (scalar agreement) ngày nay cũng rất mong manh — chỉ một tác tử đánh lừa duy nhất cũng có thể làm sai lệch hoàn toàn mô hình Hỗn hợp Tác tử (Mixture-of-Agents). BFT là điều kiện cần nhưng chưa đủ. Bài học này sẽ xây dựng một giao thức BFT tối giản, tiêm ba dạng tấn công đặc thù của tác tử AI (lời nói dối Byzantine, sự phục tùng nịnh bợ, và văn hóa đơn canh lỗi tương quan), đồng thời đo lường năng lực ứng phó của từng biến thể đồng thuận.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 phút

## Vấn đề

Giả sử bạn có $N$ tác tử LLM, mỗi tác tử đưa ra một câu trả lời độc lập. Chúng bất đồng ý kiến với nhau. Cơ chế bỏ phiếu theo đa số (majority vote) chọn sai đáp án vì hai tác tử có lỗi tương quan với nhau (dùng chung mô hình nền tảng, chung dữ liệu huấn luyện, chung các dạng lỗi suy luận). Một tác tử thứ ba lại vô tình mắc một lỗi mới lạ — dẫn đến việc nhóm chiếm đa số thực chất là một "đa số giả mạo".

Bây giờ hãy thêm vào một tác tử lừa dối (deceptive agent): nó cố tình nói dối. Hoặc một tác tử nịnh bợ (sycophantic agent): nó luôn đồng thuận với bất kỳ ai phát biểu sau cùng. Trong lý thuyết BFT cổ điển, giả định đặt ra là các nút lỗi Byzantine chỉ chiếm tỷ lệ $f < n/3$ và có hành vi ngẫu nhiên, tùy ý. Nhưng thực tế năm 2026 cho thấy các nút tác tử LLM mang tính ngẫu nhiên (stochastic) ngay cả khi chúng hoàn toàn trung thực, có tính tương quan chéo giữa các mô hình, và dễ dàng bị thao túng bởi đầu ra của nhau. Bạn không thể coi chúng như các biến ngẫu nhiên Bernoulli độc lập khi bỏ phiếu.

Lý thuyết BFT cổ điển (PBFT, 1999) không sai — nó chỉ chưa đầy đủ. Nó xử lý tốt việc đảo bit ngẫu nhiên trong phần cứng. Nhưng nó hoàn toàn bất lực trước trường hợp "ba tác tử trung thực cùng chia sẻ một ảo giác vì chúng dùng chung tập dữ liệu huấn luyện". Bài học này sẽ xuất phát từ nền móng PBFT và mở rộng thêm ba cải tiến thích ứng từ giai đoạn 2025–2026.

## Khái niệm

### Những gì BFT cổ điển cung cấp

Khả năng Chịu lỗi Byzantine Thực tế (Practical Byzantine Fault Tolerance - Castro & Liskov, OSDI 1999) cho phép hệ thống chịu đựng được $f < n/3$ nút Byzantine độc hại. Giao thức gồm ba giai đoạn (tiền chuẩn bị - pre-prepare, chuẩn bị - prepare, cam kết - commit) và hai khối nguyên thủy (thông điệp có chữ ký số - signed messages, chứng chỉ túc số - quorum certificates). Nó đảm bảo sự đồng thuận trên một giá trị duy nhất giữa $n \ge 3f + 1$ nút (bao gồm cả nút trung thực lẫn nút độc hại).

Các cam kết bảo đảm của BFT rất mạnh mẽ, nhưng dựa trên ba giả định tiên quyết:

1. **Lỗi độc lập (Independent faults).** Các nút Byzantine không thông đồng, phối hợp ăn ý với nhau.
2. **Nút trung thực luôn cho kết quả đúng.** Độ chính xác của đầu ra từ nút trung thực không phải là vấn đề; giao thức chỉ có nhiệm vụ giải quyết sự bất đồng ý kiến.
3. **Câu hỏi luôn có một chân lý duy nhất (ground-truth).** Đồng thuận trên một dữ kiện sai lệch vẫn được coi là đồng thuận thành công.

Các tác tử LLM vi phạm cả ba giả định trên. Hai tác tử chạy cùng một mô hình nền tảng sẽ chia sẻ chung các lỗ hổng. Một tác tử LLM "trung thực" vẫn có thể gặp ảo giác. Và với những câu hỏi mơ hồ, "chân lý" phụ thuộc vào việc các tác tử quyết định điều gì — hoàn toàn không có một trọng tài kiểm chứng bên ngoài (external oracle).

### Ba dạng tấn công đặc thù của tác tử LLM

**Lời nói dối Byzantine (Byzantine lie).** Một tác tử cố tình đưa ra câu trả lời sai lệch hoàn toàn. BFT cổ điển xử lý tốt trường hợp này nếu $f < n/3$.

**Sự phục tùng nịnh bợ (Sycophantic conformity).** Một tác tử đọc câu trả lời của các tác tử khác trước khi bỏ phiếu và nghiêng theo ý kiến của người phát biểu gần nhất. Tác tử này không có ác ý, nhưng phiếu bầu của nó bị tương quan với tiếng nói lớn nhất hoặc gần nhất. BFT cổ điển không ngăn chặn được vì tác tử vẫn vượt qua mọi bài kiểm tra chữ ký số hợp lệ.

**Văn hóa đơn canh lỗi tương quan (Correlated-error monoculture).** Ba tác tử dùng chung một mô hình nền tảng. Chúng cùng sinh ảo giác ra chung một đáp án sai. Nhóm đa số chọn sai. BFT cổ điển bất lực vì cả ba đều đồng thuận một cách "trung thực".

### Các giải pháp ứng phó giai đoạn 2025–2026

**CP-WBFT** (arXiv:2511.10400) — BFT có trọng số dựa trên thăm dò độ tin cậy (Confidence-Probed Weighted BFT). Mỗi cử tri gắn kèm một chỉ số độ tin cậy vào câu trả lời của mình (xác suất tự ước lượng, hoặc dự đoán từ một mô hình hiệu chỉnh độc lập). Trọng số phiếu bầu tỷ lệ thuận với độ tin cậy. Nghiên cứu báo cáo mức cải thiện khả năng chịu lỗi BFT lên tới +85.71% trên đồ thị đầy đủ. Cơ chế này đặc biệt hiệu quả để giảm thiểu sự phục tùng nịnh bợ (các tác tử a dua thường có độ tin cậy thấp đối với vị thế mà chúng bị ép theo).

**DecentLLMs** (arXiv:2507.14928) — Kiến trúc phi tập trung không nút chủ (Leaderless). Các tác tử thợ (worker agents) đề xuất giải pháp song song, các tác tử đánh giá (evaluator agents) chấm điểm đề xuất, và câu trả lời cuối cùng là trung vị hình học (geometric median) của các vị thế có điểm số. Hệ thống duy trì tính vững chắc khi $f < n/2$. Cơ chế này khắc phục tốt lời nói dối Byzantine và lỗi tương quan (trung vị hình học có khả năng kháng ngoại lai mạnh mẽ và kéo kết quả về cụm phân bố dày đặc, thay vì giá trị trung bình bị lệch do thiên vị mô hình).

**WBFT** (arXiv:2505.05103) — BFT có trọng số kết hợp phân cụm cấu trúc phân cấp. Trọng số phiếu bầu được gán dựa trên chất lượng phản hồi cộng với điểm tín nhiệm tích lũy qua lịch sử. Phân cụm các tác tử thành hai nhóm: nhóm Lõi (Core) và nhóm Biên (Edge); các tác tử Lõi phải đạt đồng thuận trước, các tác tử Biên tuân theo sau. Giải pháp này giúp tối ưu khả năng mở rộng (nhóm Lõi nhỏ gọn giúp đồng thuận nhanh) và giảm bớt rủi ro văn hóa đơn canh (nhóm Lõi có thể được chủ động lựa chọn để đảm bảo tính đa dạng của mô hình).

### Thực nghiệm: "Can AI Agents Agree?" (arXiv:2603.01213)

Bài báo đo lường mức độ đồng thuận trên giá trị vô hướng (scalar agreement - các tác tử LLM cùng thống nhất một giá trị số duy nhất) trên nhiều mô hình biên giới (frontier models). Kết luận đưa ra rất đáng lưu tâm:

- Ngay cả khi không có bất kỳ tác tử đối nghịch nào, tỷ lệ bất đồng ý kiến của các tác tử LLM đối với các câu hỏi vô hướng vẫn vượt quá 30% trên nhiều bộ đo chuẩn (benchmarks).
- Một tác tử duy nhất mang nhân cách lừa dối (deceptive persona) có thể kéo sự đồng thuận của khối Hỗn hợp Tác tử (Mixture-of-Agents) lệch tới hơn 40 điểm phần trăm so với mức chuẩn trung thực ban đầu.
- Tỷ lệ bất đồng ý kiến tỷ lệ thuận với sự đa dạng của mô hình — các tập hợp mô hình không đồng nhất (heterogeneous ensembles) bất đồng nhiều hơn các tập hợp đồng nhất (ưu điểm: lỗi không bị tương quan; nhược điểm: mất nhiều thời gian hơn để đạt đồng thuận).

Bài học cốt lõi: BFT cung cấp cơ chế để đồng nhất các đầu ra, nhưng nó không thể tự chứng minh đầu ra đồng nhất đó có đúng đắn hay không. Bạn bắt buộc phải kết hợp BFT với thẩm định độc lập (chuyên môn hóa vai trò ở Phase 16 · 08), đa dạng hóa mô hình (các biến thể tranh biện ở Phase 16 · 15), và các tác tử đánh giá chuyên trách (bộ đo chuẩn ở Phase 16 · 24).

### Quy trình giao thức cốt lõi rút gọn

Một vòng lặp BFT tối giản dành cho các tác tử LLM:

```text
1. Nhiệm vụ được giao; mỗi tác tử i sinh câu trả lời a_i
2. Mỗi tác tử đính kèm độ tin cậy c_i trong khoảng [0, 1]
3. Bộ tổng hợp (aggregator) thu thập cặp (a_i, c_i) từ tất cả n tác tử
4. Bộ tổng hợp gom nhóm theo cụm ngữ nghĩa (các câu trả lời tương đương)
5. Bộ tổng hợp tính tổng trọng số cho từng cụm C:
     w(C) = sum_{i in C} c_i
6. Cụm chiến thắng = cụm có trọng số lớn nhất, nếu max > threshold * sum(c_i)
   ngược lại: thử lại hoặc chuyển cấp xử lý (escalate)
7. Các cụm thiểu số được ghi nhật ký kèm nguồn gốc phục vụ kiểm toán sau
```

Bước phân cụm ngữ nghĩa (semantic clustering) là cải tiến then chốt dành riêng cho LLM. Hai câu trả lời "nghiên cứu báo cáo 4.2%" và "mức cải thiện 4.2%" phải thuộc về cùng một cụm. Phép so sánh chuỗi bằng nhau ngây thơ sẽ bỏ sót điểm này. Trong môi trường sản xuất, bạn nên dùng một mô hình nhúng (embedding model) chi phí thấp hoặc thực hiện quy chuẩn hóa rõ ràng.

### Tinh chỉnh ngưỡng đồng thuận (Threshold Tuning)

Tham số `threshold` quyết định khi nào chấp nhận kết quả và khi nào cần thử lại. Đặt quá thấp: bạn sẽ chấp nhận cả những đa số yếu ớt. Đặt quá cao: hệ thống không bao giờ đạt được đồng thuận. Khoảng giá trị thực nghiệm: 0.5–0.67 cho $n = 5–7$ tác tử, và cao hơn đối với $n$ nhỏ hơn. Nếu dưới ngưỡng, hãy chuyển tiếp sự việc lên con người hoặc giao cho một tập hợp tác tử khác xử lý.

### Những trường hợp đồng thuận bất lực

- **Câu hỏi mơ hồ.** Nếu bài toán không có chân lý nền tảng duy nhất, sự đồng thuận thực chất chỉ là một ý kiến chủ quan. Hãy nhìn nhận đúng bản chất của nó.
- **Câu hỏi phức hợp.** "Viết mã và giải thích đoạn mã" — đây là hai câu trả lời độc lập. Hãy tổ chức bỏ phiếu riêng cho từng phần.
- **Tranh biện đối kháng nhiều vòng.** Nếu các tác tử có thể quan sát các vòng trước và sao chép lẫn nhau (mô hình tranh luận Du 2023), chúng sẽ bắt đầu a dua đồng thuận bất chấp đúng sai. Cần giới hạn chặt chẽ số vòng (thường từ 2–3 vòng).

````figure
swarm-consensus-wave
````

## Xây dựng thực tế

Tệp `code/main.py` triển khai:

- `AgentVoter` — chính sách tác tử theo kịch bản với bộ đôi `(answer, confidence)`.
- `MajorityVote` — cơ chế bỏ phiếu đa số tương đối cổ điển.
- `CPWBFT` — bỏ phiếu có trọng số theo độ tin cậy kết hợp phân cụm ngữ nghĩa.
- `DecentLLMs` — gom cụm theo trung vị hình học dựa trên các đề xuất có điểm số.
- `Scenario` — chạy từng bộ tổng hợp dưới ba mô hình tấn công khác nhau.

Các mẫu hình tấn công được cài đặt:

1. `byzantine`: một tác tử cố tình nói dối với độ tin cậy rất cao.
2. `sycophancy`: một tác tử sao chép nguyên văn câu trả lời đầu tiên nó thấy, kèm độ tin cậy tương ứng.
3. `monoculture`: ba tác tử chia sẻ chung một câu trả lời sai (lỗi tương quan) với độ tin cậy vừa phải.

Chạy chương trình:

```bash
python3 code/main.py
```

Đầu ra kỳ vọng: một bảng ma trận `(attack, aggregator) -> final answer`, làm nổi bật câu trả lời đúng. Bỏ phiếu đa số thất bại trước kịch bản văn hóa đơn canh. Bỏ phiếu trọng số tin cậy của CPWBFT khắc phục được sự phục tùng nịnh bợ. Trung vị hình học của DecentLLMs kéo kết quả về cụm trung thực khi nhóm đơn canh chiếm dưới một nửa tổng số tác tử.

## Sử dụng trong thực tế

Tệp `outputs/skill-consensus-designer.md` là kỹ năng thiết kế giao thức đồng thuận cho một tập hợp đa tác tử: lựa chọn phương pháp phân cụm, gán trọng số, thiết lập ngưỡng chấp thuận, và chính sách chuyển cấp xử lý khi không đạt ngưỡng.

## Đưa vào sản xuất

Trước khi triển khai bất kỳ cơ chế đồng thuận nào:

- **Kiểm thử tấn công với tối thiểu ba kịch bản** nêu trên. Giao thức của bạn phải thất bại một cách dự đoán được, chứ không được phép âm thầm sai lệch.
- **Ghi nhật ký mọi cụm thiểu số** kèm nguồn gốc dữ liệu. Cụm thiểu số chính là hệ thống cảnh báo sớm giúp bạn phát hiện các lỗi tương quan.
- **Áp đặt giới hạn số vòng tương tác.** Tuyệt đối không để xảy ra tình trạng "tiếp tục tranh luận cho đến khi đạt đồng thuận" — điều đó vô tình khuyến khích sự thỏa hiệp nịnh bợ.
- **Tách bạch sự đồng thuận khỏi tính đúng đắn.** Kết quả đồng thuận phải được chuyển qua một tác tử thẩm định độc lập nằm ngoài tập hợp.
- **Giám sát tỷ lệ đồng thuận.** Tỷ lệ đồng thuận tăng đột biến cảnh báo hiện tượng thiên vị tuân phục (conformity bias); tỷ lệ giảm mạnh báo hiệu hiện tượng trôi dạt mô hình (model drift).

## Bài tập thực hành

1. Chạy `code/main.py`. Xác nhận cơ chế đa số thất bại trước tấn công đơn canh, trong khi CPWBFT giảm thiểu được phần nào khi độ tin cậy của nhóm đơn canh dưới 0.7.
2. Thêm kịch bản tấn công thứ tư: **bỏ phiếu trắng im lặng (silent abstention)** — một tác tử từ chối trả lời ("Tôi không biết"). Từng bộ tổng hợp nên xử lý phiếu trắng thế nào? Hãy triển khai giải pháp của bạn.
3. Thay thế bộ phân cụm ngữ nghĩa từ quy chuẩn hóa chuỗi sang đo độ tương đồng nhúng (embedding-similarity bằng một mô hình nhúng mã nguồn mở). Điều gì xảy ra với cuộc tấn công nịnh bợ?
4. Đọc tài liệu CP-WBFT (arXiv:2511.10400). Triển khai bước hiệu chuẩn thăm dò độ tin cậy (dùng một mô hình hiệu chuẩn riêng để đánh giá độ tin cậy tự khai báo của từng tác tử). Đo lường mức độ cải thiện độ chính xác trong kịch bản đơn canh.
5. Đọc tài liệu "Can AI Agents Agree?" (arXiv:2603.01213). Tái lập một thí nghiệm đồng thuận vô hướng đơn giản: ba tác tử, một câu hỏi giá trị số, và một prompt mang nhân cách lừa dối. CPWBFT hoặc DecentLLMs có phát hiện và xử lý được không?

## Thuật ngữ then chốt

| Thuật ngữ | Cách hiểu phổ biến | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Khả năng chịu lỗi Byzantine (BFT) | "Chống chịu lỗi hệ thống" | Giao thức Castro-Liskov 1999 đạt đồng thuận khi có tối đa $f < n/3$ lỗi tùy ý. |
| Byzantine | "Bất kỳ hành vi xấu nào" | Một nút có thể nói dối, làm rơi gói tin, im lặng — bất kỳ hành vi bất thường nào ngoài việc sập an toàn. |
| Thăm dò độ tin cậy (Confidence probe) | "Bạn chắc chắn đến mức nào?" | Xác suất tự khai báo hoặc được mô hình hiệu chuẩn dự đoán đính kèm vào phiếu bầu. |
| Phân cụm ngữ nghĩa (Semantic clustering) | "Cùng ý, khác từ" | Gom nhóm các câu trả lời tương đương về mặt ý nghĩa trước khi tiến hành kiểm phiếu. |
| Trung vị hình học (Geometric median) | "Tâm điểm vững chắc" | Điểm tối thiểu hóa tổng khoảng cách tới các điểm mẫu. Kháng các điểm ngoại lai tốt hơn trung bình số học. |
| Văn hóa đơn canh (Monoculture) | "Cùng mô hình, cùng lỗi" | Hiện tượng các lỗi bị tương quan khi các tác tử dùng chung dữ liệu huấn luyện hoặc mô hình nền. |
| Phục tùng nịnh bợ (Sycophantic conformity) | "Đồng ý với bên nói to nhất" | Xu hướng phiếu bầu của tác tử bị thiên lệch theo người phát biểu đầu tiên hoặc có uy tín lớn nhất. |
| Lõi / Biên (Core/Edge) | "BFT phân cấp" | Phân tách trong WBFT: nhóm Lõi nhỏ gọn đạt đồng thuận trước, nhóm Biên tuân theo sau nhằm giới hạn độ trễ. |

## Tài liệu tham khảo

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://www.usenix.org/conference/osdi-99/practical-byzantine-fault-tolerance) — Nền tảng kinh điển của BFT.
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — Bỏ phiếu có trọng số dựa trên độ tin cậy.
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — Đồng thuận đa tác tử phi tập trung không nút chủ bằng trung vị hình học.
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Phân tách Lõi/Biên để giới hạn độ trễ đồng thuận.
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — Sự mong manh của đồng thuận vô hướng và các cuộc tấn công bằng nhân cách lừa dối.
