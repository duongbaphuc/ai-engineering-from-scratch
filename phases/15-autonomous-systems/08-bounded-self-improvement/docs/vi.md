# Các thiết kế tự cải tiến có giới hạn

> Giới nghiên cứu đã hội tụ về bốn nguyên hàm (primitives) để thiết lập ranh giới cho một vòng lặp tự cải tiến: Các bất biến hình thức (formal invariants) phải duy trì tính đúng đắn qua mọi chỉnh sửa; Các điểm neo tương thích (alignment anchors) không thể bị sửa đổi; Các ràng buộc đa mục tiêu (multi-objective constraints) trong đó mọi chiều không gian (an toàn, công bằng, độ bền vững) đều phải được thỏa mãn chứ không chỉ riêng hiệu năng; và Cơ chế phát hiện suy thoái (regression detection) giúp tạm dừng vòng lặp khi các số liệu lịch sử gợi ý sự sụt giảm năng lực. Không có giải pháp nào trong số này là một bằng chứng toán học tuyệt đối về sự an toàn — các kết quả lý thuyết thông tin (độ phức tạp Kolmogorov, định lý Löb) đã giới hạn những gì mà bất kỳ hệ thống nào có thể chứng minh về những hệ thống kế nhiệm của chính nó. Chúng là các biện pháp giảm thiểu nhằm nâng cao cái giá phải trả cho các thất bại âm thầm.

**Loại bài học:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python (stdlib, vòng lặp có giới hạn kèm kiểm tra bất biến)
**Điều kiện tiên quyết:** Phase 15 · 07 (RSI), Phase 15 · 04 (DGM)
**Thời gian:** ~60 phút

## Vấn đề thực tế

Bộ mô phỏng cuộc đua trong Bài 7 đã chứng minh rằng các sai khác nhỏ về tốc độ sẽ tích lũy thành những khoảng cách khổng lồ theo thời gian. Nghiên cứu tình huống DGM trong Bài 4 đã cho thấy các vòng lặp có thể chủ động gian lận (game) chính bộ đánh giá của chúng. Cả hai kết quả đều dẫn tới cùng một câu hỏi kỹ thuật: chúng ta có thể đặt ra những ràng buộc nào lên một vòng lặp tự cải tiến sao cho các ràng buộc đó không thể bị chính vòng lặp làm suy yếu một cách âm thầm?

Bản tóm tắt Hội thảo RSI tại ICLR 2026 (openreview.net/pdf?id=OsPQ6zTQXV) xác định bốn nguyên hàm như vậy. Chính sách Mở rộng có Trách nhiệm v3.0 (RSP v3.0) của Anthropic (Bài 19) và Khung An toàn Ranh giới Tiên tiến v3 (FSF v3) của DeepMind (Bài 20) đều viện dẫn chúng trong các ngưỡng năng lực. Công trình HyperAgents của Meta và các khung cộng đồng như SAHOO (tháng 3 năm 2026) đã triển khai các tập con của những nguyên hàm này trong môi trường thực tế.

Cần nhìn nhận một cách trung thực: đây chỉ là các biện pháp giảm thiểu rủi ro (mitigations). Các kết quả theo lý thuyết thông tin đã đặt ra giới hạn trên cho những gì một hệ thống có thể chứng minh về hệ thống kế nhiệm của nó, và không có thiết kế hiện tại nào giải quyết dứt điểm bài toán này về mặt hình thức. Một vòng lặp được giới hạn chặt chẽ sẽ an toàn hơn một vòng lặp không bị giới hạn, chứ không phải an toàn tuyệt đối.

## Khái niệm cốt lõi

### Nguyên hàm 1: Các bất biến hình thức

Bất biến (invariant) là một đặc tính bắt buộc phải được thỏa mãn trước và sau mỗi lần tự sửa đổi. Ví dụ:

- Phân phối đầu ra được điều kiện hóa dựa trên một tiêu đề hiến pháp cố định (Bài 17).
- Không có lệnh gọi công cụ nào được gửi tới một điểm cuối (endpoint) chưa được cấp quyền.
- Các thao tác ghi bộ nhớ đều phải đi qua một đường dẫn được ký số và ghi nhật ký kiểm toán.
- Mã băm mô-đun (module hash) của bộ đánh giá phải khớp chính xác với phiên bản đã phê duyệt.

Các bất biến được kiểm tra bởi mã nguồn bên ngoài mà vòng lặp không thể can thiệp chỉnh sửa. Nếu một đề xuất sửa đổi vi phạm một bất biến, nó sẽ bị từ chối ngay lập tức. Cơ chế này ngăn chặn phương thức can thiệp/làm sai lệch bộ đánh giá kiểu DGM (Bài 4).

Phần khó khăn nhất là lựa chọn các bất biến vừa thực sự cần thiết cho an toàn, vừa có thể tính toán được trong khoảng thời gian hợp lý. Việc đặc tả quá mức sẽ làm đình trệ tiến trình; việc đặc tả dưới mức cần thiết sẽ để cho vòng lặp lách qua các đặc tính quan trọng.

### Nguyên hàm 2: Các điểm neo tương thích

Điểm neo tương thích (alignment anchor) là một biểu diễn bất biến về mục tiêu cốt lõi của vòng lặp, được ghim cố định bên ngoài bề mặt chỉnh sửa của vòng lặp. Ví dụ:

- Một văn bản hiến pháp (Bài 17) được tải từ một vị trí chỉ đọc (read-only) trong mỗi lần triệu gọi.
- Một bộ đánh giá nằm trong một kho mã nguồn riêng biệt với cơ chế kiểm soát truy cập độc lập.
- Một bản kê khai quyền hạn công cụ (tool-permission manifest) được con người ký số và xác minh lại qua từng chu kỳ.

Vai trò của điểm neo là ngăn chặn sự trôi dạt mục tiêu (objective drift). Vòng lặp có thể cải tiến cách thức theo đuổi mục tiêu, nhưng không thể chỉnh sửa bản thân mục tiêu là gì.

Phương thức thất bại tinh vi: một điểm neo mà vòng lặp không thể chỉnh sửa vẫn có thể bị diễn giải sai lệch bởi một vòng lặp bị trôi dạt trong cách nó đọc hiểu điểm neo đó. AI Hiến pháp (Constitutional AI - Bài 17) dựa trên lập luận một cách rõ ràng để xử lý các tình huống mới lạ; chính tầng lập luận đó là nơi diễn ra sự trôi dạt diễn giải. Các điểm neo là điều kiện cần, nhưng chưa đủ.

### Nguyên hàm 3: Ràng buộc đa mục tiêu

Một vòng lặp tối ưu hóa một điểm số vô hướng (scalar score) duy nhất sẽ luôn tìm ra các lối tắt gian lận. Một vòng lặp bắt buộc phải thỏa mãn đồng thời nhiều ràng buộc cứng sẽ có ít cơ hội tìm lối tắt hơn. Các trục đánh giá điển hình:

- Hiệu năng (tiêu chuẩn đánh giá cấp tác vụ)
- An toàn (các bài đánh giá kiểm thử xâm nhập red-team, tỷ lệ từ chối trên các yêu cầu độc hại đã biết)
- Tính công bằng (các giới hạn tác động chênh lệch trên các nhóm nhạy cảm)
- Độ bền vững (tập kiểm thử ngoài phân phối OOD, xử lý dữ liệu đầu vào đối kháng)

Một sửa đổi chỉ được chấp nhận nếu mọi ràng buộc đều được thỏa mãn. Bộ điều tiết chi phí (cost governor) trong Bài 13 xếp chồng cơ chế này với các ràng buộc tài chính. Llama Guard trong Bài 18 được cắm vào như một trục an toàn.

### Nguyên hàm 4: Phát hiện suy thoái

Điểm số của mỗi chu kỳ được so sánh với các phân phối lịch sử. Một sự sụt giảm vượt quá mức dung sai cấu hình sẽ tạm dừng vòng lặp. Điều này giúp phát hiện sự mất mát năng lực âm thầm mà nếu không có nó sẽ bị hấp thụ vào mức trung bình trượt khi vòng lặp "cải tiến vượt qua" nó.

Một triển khai thực tế: lưu trữ điểm số trên từng tác vụ của N chu kỳ gần nhất. Mỗi chu kỳ mới tính toán độ chênh lệch (delta) trên từng tác vụ. Nếu bất kỳ độ chênh lệch nào giảm xuống dưới một ngưỡng cho phép, chu kỳ đó sẽ bị từ chối và chuyển cho con người xem xét.

### Các giới hạn theo lý thuyết thông tin

Độ phức tạp Kolmogorov và định lý Löb đặt ra các giới hạn trên đối với những gì một hệ thống có thể tự chứng minh về chính nó. Cỗ máy Gödel hình thức của Schmidhuber (Bài 4) nhắm tới ranh giới cao nhất này; nhưng chưa ai hoàn thành được một chứng minh không tầm thường (non-trivial proof). Kết quả của Löb chỉ ra: nếu một hệ thống có thể chứng minh được niềm tin "Tôi sẽ làm X nếu tôi chứng minh được tôi nên làm X", nó sẽ thực hiện X mà không cần chứng minh rằng nó nên làm — một thất bại tự tham chiếu (self-reference failure) nổi tiếng.

Hàm ý đối với các nguyên hàm của chúng ta: chúng không thể giải quyết dứt điểm bài toán an toàn. Chúng chỉ làm cho các thất bại âm thầm trở nên đắt giá hơn. Một vòng lặp độc hại hoặc bị trôi dạt vốn có thể lặng lẽ vượt qua một điểm kiểm tra bị thiếu, thì nay phải chủ động phá hoại một rào chắn tường minh — hành vi để lại dấu vết rõ ràng và dễ phát hiện hơn nhiều.

### Một ví dụ cụ thể

Giả sử một tác tử đề xuất một chỉnh sửa. Ngăn xếp kiểm soát (gating stack) diễn ra như sau:

1. Kiểm tra bất biến: mã băm mô-đun, bản kê khai quyền công cụ, tiêu đề hiến pháp.
2. Kiểm tra điểm neo: câu lệnh mục tiêu khớp với phiên bản đã phê duyệt (theo từng byte hoặc theo ngữ nghĩa).
3. Đánh giá đa mục tiêu: các trục hiệu năng, an toàn, công bằng, độ bền vững.
4. Phát hiện suy thoái: không có trục nào sụt giảm vượt quá mức dung sai.

Cả bốn bước đều phải vượt qua để bản chỉnh sửa được tích hợp. Bất kỳ một bước thất bại nào cũng sẽ tạm dừng vòng lặp.

```figure
bounded-gates
```

## Sử dụng trong thực tế

Tệp `code/main.py` chạy một vòng lặp tự cải tiến có giới hạn dựa trên mô hình mẫu kiểu DGM từ Bài 4, nhưng được xếp chồng thêm bốn nguyên hàm bảo vệ. Từng nguyên hàm có thể được bật hoặc tắt riêng lẻ. Thử nghiệm thực tế chứng minh rằng mỗi nguyên hàm đều ngăn chặn một lớp thất bại cụ thể, và việc loại bỏ bất kỳ nguyên hàm nào cũng sẽ khiến lớp thất bại tương ứng lọt qua.

## Đóng gói sản phẩm

Tệp `outputs/skill-bounded-loop-review.md` thực hiện kiểm toán một vòng lặp có giới hạn được đề xuất và chấm điểm xem vòng lặp đó thực sự triển khai những nguyên hàm nào trong số bốn nguyên hàm so với những gì nó tuyên bố.

## Bài tập

1. Chạy `code/main.py` với tất cả các nguyên hàm được bật. Xác nhận rằng vòng lặp vẫn cải tiến trên thước đo chính mà không để cho thủ thuật gian lận (hack) giành chiến thắng.

2. Tắt cơ chế phát hiện suy thoái. Xây dựng một trường hợp đầu vào khiến cho việc mất mát năng lực âm thầm bị chấp nhận.

3. Tắt ràng buộc đa mục tiêu. Chứng minh rằng vòng lặp hội tụ trên trục hiệu năng trong khi một trục an toàn bị sụt giảm nghiêm trọng.

4. Thiết kế một điểm neo tương thích cho một tác tử lập trình (coding agent). Văn bản gì, lưu trữ ở đâu, và kiểm tra bằng cách nào?

5. Đọc bản tóm tắt Hội thảo RSI tại ICLR 2026. Chọn một trong bốn nguyên hàm và đề xuất một cải tiến cụ thể cho hiện trạng công nghệ hiện nay.

## Thuật ngữ chính

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Invariant | "Đặc tính luôn đúng" | Một đặc tính được mã nguồn bên ngoài kiểm tra trước và sau mỗi lần chỉnh sửa |
| Alignment anchor | "Mục tiêu được ghim" | Biểu diễn mục tiêu cốt lõi bất biến nằm ngoài bề mặt chỉnh sửa của vòng lặp |
| Multi-objective constraint | "Mọi trục đều phải đạt" | Hiệu năng, an toàn, công bằng, độ bền vững — tất cả đều bắt buộc |
| Regression detection | "Tạm dừng khi sụt giảm" | Tạm dừng vòng lặp khi độ chênh lệch số liệu lịch sử gợi ý sự mất mát năng lực |
| Kolmogorov bound | "Giới hạn lý thuyết thông tin" | Giới hạn những gì một hệ thống có thể chứng minh về hệ thống kế nhiệm của nó |
| Lob's theorem | "Bẫy tự tham chiếu" | Hệ thống có thể hành động dựa trên "Tôi nên" mà không cần chứng minh nó nên làm |
| Gate stack | "Kiểm tra phân tầng" | Nhiều nguyên hàm kết hợp lại; bất kỳ thất bại nào cũng từ chối bản chỉnh sửa |
| Bounded improvement | "Giảm thiểu, không phải chứng minh" | Nâng cao cái giá của thất bại âm thầm; không giải quyết triệt để bài toán an toàn |

## Đọc thêm

- [Bản tóm tắt Hội thảo RSI ICLR 2026 (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — sự hội tụ của bốn nguyên hàm.
- [Anthropic — Chính sách Mở rộng có Trách nhiệm v3.0 (RSP v3.0)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — các ngưỡng năng lực đa mục tiêu.
- [DeepMind — Khung An toàn Ranh giới Tiên tiến v3 (FSF v3)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — giám sát sự tương thích lừa dối như một nguyên hàm bất biến.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — tiền thân chứng minh hình thức của các nguyên hàm này.
- [Anthropic — Hiến pháp của Claude (Tháng 1 năm 2026)](https://www.anthropic.com/news/claudes-constitution) — điểm neo tương thích dựa trên lập luận.
