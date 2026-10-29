# Tranh luận và Phối hợp Đa Tác tử (Multi-Agent Debate and Collaboration)

> Du et al. (ICML 2024, "Society of Minds") vận hành N phiên bản mô hình đề xuất câu trả lời độc lập, sau đó lặp lại quá trình phản biện chéo qua R vòng để đạt tới sự đồng thuận. Phương pháp này nâng cao tính xác thực, khả năng tuân thủ quy tắc và năng lực suy luận. Cấu trúc mạng thưa (sparse topology) vượt trội hơn mạng lưới toàn phần (full mesh) về chi phí token.

**Loại bài học:** Tìm hiểu + Xây dựng (Learn + Build)  
**Ngôn ngữ:** Python (stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 12 (Các mẫu luồng công việc - Workflow Patterns), Phase 14 · 05 (Tự tinh chỉnh và CRITIC - Self-Refine and CRITIC)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Giải thích giao thức tranh luận (debate protocol): N tác tử đề xuất, R vòng phản biện, hội tụ về một câu trả lời chung.
- Mô tả lý do tại sao tranh luận cải thiện tính xác thực (factuality), tuân thủ quy tắc (rule-following) và năng lực suy luận (reasoning).
- Giải thích cấu trúc mạng thưa (sparse topology): không nhất thiết mọi tác tử tranh luận đều phải quan sát tất cả các tác tử còn lại.
- Cài đặt một hệ thống tranh luận bằng thư viện chuẩn (stdlib) trên mô hình LLM giả lập theo kịch bản với các biến thể mạng toàn phần và mạng thưa; đo lường chi phí token so với độ chính xác.

## Vấn đề Đặt ra

Tự tinh chỉnh (Self-Refine - Bài 05) là kỹ thuật một mô hình tự phê bình chính mình — tiềm ẩn nguy cơ tư duy tập thể cục bộ (groupthink). CRITIC (Bài 05) neo cơ chế phê bình vào các công cụ bên ngoài — vốn không phải lúc nào cũng sẵn có. Tranh luận (Debate) mở ra phương thức thứ ba: nhiều phiên bản tác tử, phản biện chéo lẫn nhau (cross-critique), đạt được sự hội tụ thông qua bất đồng quan điểm.

## Khái niệm

### Xã hội Tâm trí (Society of Minds - Du et al., ICML 2024)

- N phiên bản mô hình đưa ra câu trả lời độc lập cho cùng một câu hỏi.
- Qua R vòng lặp, mỗi mô hình đọc đề xuất của các mô hình khác và phản biện chúng.
- Các mô hình cập nhật câu trả lời dựa trên những lời phê bình nhận được.
- Sau R vòng, hệ thống trả về câu trả lời đã hội tụ.

Các thử nghiệm ban đầu sử dụng N=3, R=2 do giới hạn về chi phí. Độ chính xác tăng lên khi có nhiều tác tử hơn và nhiều vòng lặp hơn trên các bài toán phức tạp (MMLU, GSM8K, tính hợp lệ của nước đi cờ vua, tạo tiểu sử).

Sự kết hợp chéo giữa các mô hình khác nhau đánh bại các cuộc tranh luận đơn mô hình: ChatGPT + Bard cùng tranh luận mang lại kết quả vượt trội hơn so với việc từng mô hình hoạt động độc lập.

### Cấu trúc mạng thưa (Sparse topology)

Công trình "Cải thiện tranh luận đa tác tử bằng cấu trúc giao tiếp thưa" (Improving Multi-Agent Debate with Sparse Communication Topology - arXiv:2406.11776, 2024-2025) chỉ ra rằng tranh luận dạng mạng lưới toàn phần (full-mesh) không phải lúc nào cũng tối ưu. Các cấu trúc mạng thưa (hình sao - star, hình nhẫn - ring, trục và nan hoa - hub-and-spoke) có thể đạt độ chính xác tương đương nhưng với chi phí token thấp hơn đáng kể. Mỗi tác tử tranh luận chỉ quan sát một tập con các tác tử đồng cấp.

Hệ quả tính toán:

- Mạng toàn phần N=5, R=3 = 5 × 3 = 15 đề xuất, mỗi đề xuất đọc phản hồi từ 4 bạn đồng cấp = 60 lượt phản biện.
- Mạng hình sao N=5, R=3 (1 trung tâm + 4 vệ tinh) = 15 đề xuất, các vệ tinh chỉ đọc trung tâm = 12 lượt phản biện.

### Khi nào tranh luận phát huy hiệu quả

- **Tính xác thực (Factuality):** N đề xuất độc lập, việc kiểm tra chéo giúp giảm thiểu hiện tượng ảo giác (hallucination).
- **Tuân thủ quy tắc (Rule-following):** Tính hợp lệ của nước đi trong cờ vua — một mô hình bỏ sót luật thì các mô hình khác sẽ phát hiện ra.
- **Suy luận mở (Open-ended reasoning):** Nhiều góc nhìn tiếp cận khác nhau cùng thu hẹp dần về câu trả lời chính xác.

### Khi nào tranh luận gây tổn hại

- **Trải nghiệm người dùng nhạy cảm với độ trễ (Latency-sensitive UX):** N × R vòng tuần tự tạo ra độ trễ quá lớn mà bạn có thể không chấp nhận được.
- **Quy mô nhạy cảm với chi phí (Cost-sensitive scale):** Tiêu tốn N × R token cho mỗi câu hỏi.
- **Tra cứu sự thật đơn giản (Simple factual lookups):** Một lần tra cứu đơn lẻ rẻ hơn rất nhiều so với năm lượt tranh luận.

### Các triển khai thực tế năm 2026

- **Điều phối viên - Công nhân của Anthropic (Anthropic orchestrator-workers)** (Bài 12) — một biến thể của tranh luận với bước tổng hợp (synthesis).
- **Bộ giám sát LangGraph (LangGraph supervisor)** (Bài 13) — bộ định tuyến trung tâm + các tác tử chuyên biệt có thể triển khai tranh luận dưới dạng một nút (node).
- **OpenAI Agents SDK** (Bài 16) — các tác tử chuyển giao quyền xử lý qua lại (handoff) để phản biện lặp.
- **Đánh giá đa tác tử (Multi-agent evals)** — kết hợp tranh luận + bộ đánh giá-tối ưu hóa (evaluator-optimizer) để tạo tín hiệu đánh giá.

### Những điểm dễ sai lầm của mẫu hình này

- **Sụp đổ hội tụ (Convergence collapse):** Toàn bộ tác tử nhanh chóng hội tụ về câu trả lời sai đầu tiên. Giảm thiểu bằng cách áp đặt các vòng bất đồng bắt buộc.
- **Lỗi điểm trung tâm (Hub failure):** Trong cấu trúc hình sao, nút trung tâm hỏng sẽ làm sai lệch toàn bộ hệ thống. Giải pháp là luân chuyển hoặc sử dụng nhiều nút trung tâm.
- **Đồng nhất hóa chỉ dẫn (Prompt homogenization):** Tất cả tác tử dùng chung một prompt dẫn đến sinh ra các câu trả lời giống hệt nhau. Cần sử dụng các prompt và/hoặc các mô hình đa dạng.

```figure
debate-converge
```

## Tự Xây dựng (Build It)

`code/main.py` cài đặt cơ chế tranh luận bằng thư viện chuẩn (stdlib):

- Lớp `Debater` (mô hình LLM giả lập theo kịch bản kèm độ lệch quan điểm theo từng tác tử).
- Bộ thực thi `FullMeshDebate` và `SparseDebate`.
- Ba câu hỏi: một câu hỏi về sự thật, một câu hỏi dựa trên quy tắc, một câu hỏi về suy luận logic.
- Các chỉ số đo lường: câu trả lời hội tụ, số vòng cần để hội tụ, tổng số lượt phản biện.

Chạy mã nguồn:

```bash
python3 code/main.py
```

Kết quả: độ chính xác và chi phí theo từng giao thức; cấu trúc mạng thưa tương đương mạng toàn phần trên 2/3 câu hỏi với chi phí thấp hơn.

## Ứng dụng Thực tế (Use It)

- **Anthropic orchestrator-workers** cho các cuộc tranh luận đơn giản gồm 2-3 tác tử thực thi.
- **LangGraph** cho tranh luận đa vòng có trạng thái (stateful) kết hợp lưu điểm kiểm tra (checkpointing).
- **Tùy chỉnh (Custom)** phục vụ nghiên cứu hoặc các bài toán đòi hỏi bảo đảm tính chính xác chuyên biệt.

## Chuyển giao Đóng gói (Ship It)

`outputs/skill-debate.md` cung cấp khung sườn cho một cuộc tranh luận đa tác tử với cấu trúc liên kết tùy biến, N tác tử, R vòng lặp và quy tắc hội tụ rõ ràng.

## Bài tập Thực hành

1. Cài đặt quy tắc "bắt buộc bất đồng" (forced disagreement): trong vòng 1, mỗi tác tử tranh luận phải đưa ra một đề xuất riêng biệt. Đo lường tác động của quy tắc này tới tốc độ hội tụ.
2. Thêm cơ chế tổng hợp có trọng số theo độ tin cậy (confidence-weighted aggregation): các tác tử trả về cặp (câu trả lời, độ tin cậy); bộ tổng hợp sẽ lấy trọng số theo độ tin cậy. Cách này có mang lại cải thiện không?
3. Thay thế một "tác tử" bằng một mô hình LLM giả lập khác có quan điểm khác biệt. Tính không đồng nhất (heterogeneity) có cải thiện độ chính xác không?
4. Đo lường chi phí token của mạng toàn phần so với mạng thưa trên 3 câu hỏi của bạn. Vẽ biểu đồ biểu diễn chi phí so với độ chính xác.
5. Đọc bài báo Society of Minds. Mở rộng mô hình thử nghiệm của bạn lên N=5, R=3. Điểm nào bị vỡ? Điểm nào được cải thiện?

## Thuật ngữ Then chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|---|---|---|
| Tranh luận (Debate) | "Phản biện đa tác tử" | N bên đề xuất, R vòng phản biện chéo, đạt tới hội tụ |
| Mạng toàn phần (Full mesh) | "Ai cũng đọc của tất cả mọi người" | Mỗi tác tử tranh luận đọc đề xuất của mọi bạn đồng cấp ở mỗi vòng |
| Cấu trúc thưa (Sparse topology) | "Góc nhìn hạn chế giữa các bên" | Tác tử tranh luận chỉ đọc đề xuất từ một tập con đồng cấp |
| Trục và nan hoa (Hub-and-spoke) | "Cấu trúc hình sao" | Một tác tử trung tâm, N-1 nhánh vệ tinh chỉ đọc thông tin từ trung tâm |
| Sự hội tụ (Convergence) | "Sự đồng thuận" | Các tác tử tranh luận cùng thống nhất về một câu trả lời chung |
| Xã hội Tâm trí (Society of Minds) | "Bài báo tranh luận của Du et al." | Phương pháp tranh luận đa tác tử tại ICML 2024 |

## Đọc thêm

- [Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — bài báo kinh điển về tranh luận đa tác tử
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — các kết quả nghiên cứu về cấu trúc liên kết thưa
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — mô hình điều phối viên - công nhân như một biến thể tranh luận
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — phương pháp đối ứng tự phê bình trên một mô hình duy nhất
