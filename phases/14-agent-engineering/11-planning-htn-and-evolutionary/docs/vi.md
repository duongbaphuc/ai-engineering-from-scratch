# Lập kế hoạch với HTN và Tìm kiếm Tiến hóa

> Lập kế hoạch ký hiệu (symbolic planning) giải quyết triệt để các trường hợp mà kế hoạch cần phải được chứng minh tính đúng đắn (provably correct). Tìm kiếm mã nguồn theo phương pháp tiến hóa (evolutionary code search) xử lý các trường hợp mà hàm độ thích nghi (fitness function) có thể kiểm tra được bằng máy tính (machine-checkable). ChatHTN (2025) và AlphaEvolve (2025) cho thấy tiềm năng to lớn mà mỗi hướng tiếp cận này mở ra khi được kết hợp với một mô hình ngôn ngữ lớn (LLM).

**Loại bài học:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn - stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 02 (ReWOO và Lập kế hoạch và Thực thi - Plan-and-Execute)  
**Thời lượng:** ~75 phút

## Mục tiêu Học tập

- Giải thích Mạng Tác vụ Phân cấp (Hierarchical Task Networks - HTN): tác vụ (tasks), phương pháp (methods), toán tử (operators), tiền điều kiện (preconditions), và hiệu ứng (effects).
- Mô tả vòng lặp lai của ChatHTN — tìm kiếm ký hiệu kết hợp cơ chế phân rã dự phòng bằng LLM (LLM fallback decomposition).
- Giải thích vòng lặp tiến hóa của AlphaEvolve và lý do tại sao nó chỉ phát huy hiệu quả khi đi kèm một bộ đánh giá bằng chương trình (programmatic evaluator).
- Hiện thực hóa một bộ lập kế hoạch HTN mẫu và thuật toán tìm kiếm tiến hóa mô hình hóa bằng thư viện chuẩn Python.

## Vấn đề

ReWOO (Bài học 02), Plan-and-Execute, và ReAct bao quát phần lớn các bài toán lập kế hoạch tác tử. Tuy nhiên, có hai trường hợp mà các phương pháp này không đáp ứng tốt:

1. **Các kế hoạch đòi hỏi chứng minh tính đúng đắn.** Lập lịch biểu, dẫn đường bay, quy trình tuân thủ quy chuẩn — kế hoạch bắt buộc phải chuẩn xác ngay từ khâu cấu trúc (sound by construction). Một kế hoạch do LLM sinh ra trôi chảy nhưng thi thoảng lại tạo ảo giác (hallucinate) ra một bước sai là điều hoàn toàn không thể chấp nhận được.
2. **Các bài toán tối ưu hóa đi kèm hàm thích nghi kiểm tra được bằng máy.** Phép nhân ma trận, các luật heuristic lập lịch, các lượt tối ưu hóa trình biên dịch — mục tiêu ở đây không đơn thuần là "một kế hoạch đúng" mà phải là "kế hoạch tối ưu nhất".

Quy hoạch HTN và AlphaEvolve giải quyết hai bài toán khác biệt này. Cả hai đều sử dụng LLM như một công cụ khuếch đại chứ không phải để thay thế hoàn toàn.

## Khái niệm

### Mạng Tác vụ Phân cấp (Hierarchical Task Networks - HTN)

Một cấu trúc HTN bao gồm:

- **Tác vụ (Tasks)** — tác vụ phức hợp (compound, cần được phân rã tiếp) và tác vụ nguyên thủy (primitive, có thể thực thi trực tiếp).
- **Phương pháp (Methods)** — các cách thức để phân rã một tác vụ phức hợp thành các tác vụ con, đi kèm các tiền điều kiện.
- **Toán tử (Operators)** — các hành động nguyên thủy đi kèm tiền điều kiện (preconditions) và hiệu ứng (effects).
- **Trạng thái (State)** — một tập hợp các dữ kiện đang đúng.

Quá trình lập kế hoạch: cho trước một tác vụ mục tiêu và một trạng thái ban đầu, tìm một chuỗi phân rã thành các toán tử nguyên thủy sao cho tiền điều kiện của chúng được thỏa mãn tuần tự.

HTN ra đời trước LLM rất lâu và cho đến nay vẫn là chuẩn tham chiếu cho các kế hoạch đòi hỏi chứng minh được tính đúng đắn.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) đan xen cơ chế HTN ký hiệu với các truy vấn LLM:

1. Thử phân rã tác vụ phức hợp hiện tại bằng các phương pháp sẵn có trong thư viện.
2. Nếu không có phương pháp nào phù hợp, hỏi LLM: "bạn sẽ phân rã `task` trong trạng thái `s` như thế nào?"
3. Dịch câu trả lời của LLM thành các tác vụ con tiềm năng.
4. Xác thực đối chiếu với schema toán tử; từ chối các phân rã không hợp lệ.
5. Đệ quy tiếp tục quá trình.

Khẳng định cốt lõi của bài báo: mọi kế hoạch được sinh ra đều được chứng minh tính đúng đắn bởi vì các gợi ý từ LLM chỉ đóng vai trò như các ứng viên phân rã chứ không bao giờ trực tiếp chỉnh sửa kế hoạch. Tầng ký hiệu nắm quyền kiểm soát tính đúng đắn; LLM đóng vai trò mở rộng thư viện phương pháp.

Kỹ thuật học phương pháp trực tuyến (online method learning - OpenReview `gwYEDY9j2x`, nghiên cứu tiếp nối năm 2025) bổ sung thêm một bộ học khái quát hóa các phân rã do LLM tạo ra bằng kỹ thuật hồi quy — giúp cắt giảm tần suất truy vấn LLM lên tới 75%.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, tháng 6 năm 2025) lại là một hướng đi hoàn toàn khác: tìm kiếm mã nguồn tiến hóa được điều phối bởi một nhóm mô hình (ensemble) Gemini 2.0 Flash/Pro.

Vòng lặp vận hành:

1. Bắt đầu với một chương trình hạt giống (seed program) + một bộ đánh giá bằng chương trình (trả về điểm thích nghi - fitness score).
2. Nhóm mô hình LLM đề xuất các đột biến mã (mutations).
3. Chạy các đột biến qua bộ đánh giá.
4. Giữ lại biến thể tốt nhất; tiếp tục tạo đột biến.

Các thành tựu công bố nổi bật:

- Cải tiến đầu tiên vượt qua thuật toán Strassen cho phép nhân ma trận phức 4x4 sau 56 năm (chỉ cần 48 phép nhân vô hướng).
- Thu hồi 0.7% năng lực tính toán toàn cầu của Google thông qua một heuristic lập lịch Borg mới.
- Tăng tốc 32% FlashAttention trên khối lượng công việc mô hình biên.

Ràng buộc cứng bắt buộc: hàm độ thích nghi phải có khả năng kiểm tra tự động bằng máy. Tìm kiếm tiến hóa trên các câu trả lời văn bản tự nhiên sẽ không thể hội tụ.

### Khi nào nên sử dụng phương pháp nào

| Nhóm bài toán | Phương pháp áp dụng | Lý do |
|---|---|---|
| Lập lịch với các ràng buộc cứng | HTN + ChatHTN | Đảm bảo tính đúng đắn có thể chứng minh |
| Tối ưu hóa trình biên dịch | AlphaEvolve | Hàm thích nghi kiểm tra được bằng máy |
| Thực thi tác vụ nhiều bước linh hoạt | ReAct / ReWOO | Tận dụng LLM tương tác lặp, không cần bảo đảm hình thức |
| Cải tiến mã nguồn có bộ kiểm thử | AlphaEvolve | Bộ kiểm thử đóng vai trò làm bộ đánh giá |
| Tự động hóa ràng buộc theo chính sách | HTN | Tiền điều kiện mã hóa chính sách một cách chặt chẽ |

### Những điểm mô hình này dễ gặp sự cố

- **HTN không có toán tử chuẩn.** Nếu thiếu schema tiền điều kiện/hiệu ứng, tuyên bố về tính đúng đắn sẽ sụp đổ hoàn toàn. Cơ chế "LLM gợi ý phân rã" của ChatHTN bắt buộc phải có schema để từ chối các bước đi không hợp lệ.
- **AlphaEvolve thiếu bộ đánh giá thực chất.** "Hỏi LLM xem đoạn mã này có tốt hơn không" hoàn toàn không phải là một hàm thích nghi. Bộ đánh giá bắt buộc phải có tính tất định và tốc độ thực thi nhanh.
- **Thiết kế quá mức cần thiết (Over-engineering).** Phần lớn các tác vụ tác tử thông thường không cần đến hai cơ chế phức tạp này. Hãy ưu tiên lựa chọn ReAct hoặc ReWOO trước tiên.

```figure
htn-tree-expand
```

## Xây dựng Thực tế

`code/main.py` hiện thực hóa hai hệ thống mô hình hóa tinh gọn:

- Một bộ lập kế hoạch HTN bằng thư viện chuẩn Python với các toán tử, phương pháp, tiền điều kiện, hiệu ứng, và một cơ chế `LLMFallback` kích hoạt khi không có phương pháp nào khớp với tác vụ phức hợp. Thành phần "LLM" là một bộ phân rã theo kịch bản mẫu để bộ lập kế hoạch có thể chạy ngoại tuyến.
- Một thuật toán tìm kiếm tiến hóa bằng thư viện chuẩn trên các chương trình số học: phát triển các biểu thức toán học sao cho đầu ra của chúng tối thiểu hóa `|f(x) - target|` trên một tập kiểm thử. Bộ đánh giá hoàn toàn có tính tất định.

Chạy chương trình:

```bash
python3 code/main.py
```

Vết thực thi hiển thị quá trình bộ lập kế hoạch HTN phân rã tác vụ phức hợp (với bước dự phòng LLM ở giữa quá trình) và vòng lặp tiến hóa hội tụ dần về biểu thức mục tiêu mong muốn.

## Ứng dụng Thực tế

- **Các bộ lập kế hoạch HTN** — `pyhop`, `SHOP3`, hoặc tự xây dựng bộ phân rã để thực thi chính sách đặc thù theo miền nghiệp vụ.
- **ChatHTN** — mã nguồn nghiên cứu; mô hình này (ký hiệu + dự phòng LLM) có thể tích hợp tương thích với bất kỳ bộ lập kế hoạch HTN nào.
- **AlphaEvolve** — bài báo của DeepMind; mô hình thiết kế (nhóm mô hình + bộ đánh giá) có thể tái tạo được. Các nhánh mã nguồn mở như OpenEvolve đang bắt đầu xuất hiện.
- **Các framework tác tử** — hiện tại chưa có framework nào tích hợp sẵn HTN hoặc AlphaEvolve như thành phần hạng nhất. Bạn nên xây dựng nó dưới dạng một tác tử phụ (subagent) hoặc tiến trình xử lý nền (background worker).

## Đóng gói Sản phẩm

`outputs/skill-hybrid-planner.md` tạo ra một khung sườn bộ lập kế hoạch lai (HTN hoặc tiến hóa) với vai trò của LLM được giới hạn phạm vi một cách tường minh.

## Bài tập

1. Mở rộng bộ lập kế hoạch HTN với cơ chế quay lui (backtracking): khi hậu điều kiện của một toán tử thất bại trong quá trình thực thi, hãy quay lui trạng thái và thử nghiệm phương pháp tiếp theo.
2. Bổ sung bộ nhớ đệm phương pháp LLM (LLM-method cache) vào ChatHTN: khi LLM phân rã tác vụ `T` trong mẫu trạng thái `P`, hãy lưu trữ kết quả. Tái kiểm tra thư viện phương pháp này trước tiên trong lần gọi tiếp theo.
3. Thay thế bộ đánh giá tìm kiếm tiến hóa bằng một bộ kiểm thử thực tế. Tiến hóa một hàm sắp xếp sao cho vượt qua 20 ca kiểm thử; ghi nhận số thế hệ cần thiết để hội tụ.
4. Đọc các ghi chú thiết kế bộ đánh giá của AlphaEvolve. Thiết kế một bộ đánh giá cho một miền nghiệp vụ bạn quan tâm (tối ưu hóa truy vấn SQL, tối thiểu hóa bộ kiểm thử, tệp YAML triển khai hạ tầng).
5. Kết hợp cả hai: sử dụng HTN để phân rã một tác vụ phức hợp thành các tác vụ con, sau đó dùng tìm kiếm tiến hóa trên từng toán tử nguyên thủy của tác vụ con đó. Phân tích xem phương pháp này tỏa sáng ở đâu và khi nào nó bị coi là quá phức tạp không cần thiết?

## Thuật ngữ Then chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| HTN | "Bộ lập kế hoạch phân cấp" | Phân rã tác vụ với các toán tử, tiền điều kiện và hiệu ứng |
| Phương pháp (Method) | "Quy tắc phân rã" | Cách thức bóc tách một tác vụ phức hợp thành các tác vụ con |
| Toán tử (Operator) | "Hành động nguyên thủy" | Bước thực thi cụ thể kèm tiền điều kiện và hiệu ứng |
| ChatHTN | "LLM kết hợp HTN" | Bộ lập kế hoạch ký hiệu hỏi LLM khi không có phương pháp nào khớp |
| AlphaEvolve | "Tìm kiếm mã nguồn tiến hóa" | Nhóm LLM tạo đột biến mã nguồn; bộ đánh giá tất định chọn lọc biến thể tốt nhất |
| Hàm thích nghi (Fitness function) | "Bộ đánh giá" | Điểm số tất định, kiểm tra được bằng máy tính trên các đầu ra |
| Học phương pháp trực tuyến (Online method learning) | "Bộ nhớ đệm phân rã của LLM" | Lưu trữ và khái quát hóa các kế hoạch LLM nhằm cắt giảm chi phí truy vấn |

## Tài liệu Tham khảo

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — Bộ lập kế hoạch lai giữa ký hiệu và LLM
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — Tìm kiếm mã nguồn tiến hóa với đột biến sinh bởi LLM
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Khi nào nên dùng bộ lập kế hoạch chuyên sâu thay vì vòng lặp đơn giản
