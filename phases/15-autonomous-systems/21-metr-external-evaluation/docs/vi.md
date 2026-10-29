# Chân trời Thời gian METR và Đánh giá Năng lực từ Bên ngoài (METR Time Horizons and External Capability Evaluation)

> METR (tiền thân là ARC Evals) hoạt động như một tổ chức phi lợi nhuận 501(c)(3) độc lập từ tháng 12 năm 2023. Điểm chuẩn Chân trời Thời gian 1.1 (Time Horizon 1.1 - tháng 1 năm 2026) của họ khớp một đường cong logistic giữa xác suất thành công của tác vụ với $\log(\text{thời gian hoàn thành của chuyên gia con người})$; giao điểm tại xác suất 50% xác định chân trời thời gian của mô hình. Các dự án đánh giá giai đoạn 2025–2026 bao quát GPT-5.1, GPT-5.1-Codex-Max cùng các đánh giá nguyên mẫu giám sát (liệu hệ thống giám sát có bắt được các tác vụ phụ lén lút hay không; liệu tác tử có né tránh được bộ giám sát hay không). Các bộ điểm chuẩn cốt lõi: HCAST (hơn 180 tác vụ ML, an ninh mạng, kỹ thuật phần mềm, suy luận logic; thời gian từ 1 phút đến hơn 8 giờ), RE-Bench (71 tác vụ kỹ thuật nghiên cứu ML với đường cơ sở của chuyên gia con người), SWAA. Một lưu ý thực tế: các phép đo của METR mang tính lý tưởng hóa — không có người dùng thực tế, không để lại hậu quả thực — và chính đội ngũ nghiên cứu đã ghi nhận khoảng cách hành vi giữa đánh giá và triển khai thực tế (Bài 1). Chân trời thời gian là một cận trên, không phải là một dự báo triển khai thực tế.

**Type:** Learn  
**Languages:** Python (stdlib, công cụ ước lượng chân trời thời gian bằng khớp đường cong logistic)  
**Prerequisites:** Phase 15 · 01 (Tác tử chân trời dài - Long-horizon agents), Phase 15 · 19 (RSP - Chính sách Mở rộng Quy mô Có trách nhiệm)  
**Time:** ~60 phút

## Vấn đề

Các chính sách mở rộng quy mô (Bài 19, 20) chỉ thực sự hữu ích khi gắn liền với các phép đo lường mà chúng dẫn chiếu. Các khái niệm như "ngưỡng AI R&D-4" và "Quyền tự trị tầm xa" ban đầu chỉ được định nghĩa bằng văn phong chính sách; chúng chỉ trở nên khả thi để hành động khi các phép đánh giá cụ thể tạo ra những con số định lượng cụ thể.

METR là tổ chức đánh giá độc lập bên ngoài tiêu biểu giai đoạn 2024–2026 chịu trách nhiệm xác lập phần lớn các con số định lượng đó. Họ đánh giá các mô hình biên giới — thường ở giai đoạn tiền phát hành theo thỏa thuận bảo mật NDA với các phòng nghiên cứu — và công bố phương pháp luận sau đó. Điểm chuẩn Time Horizon 1.1 (tháng 1 năm 2026) là sản phẩm nổi bật nhất của họ: một đại lượng vô hướng đơn lẻ nén năng lực thành một đơn vị trực quan dễ hiểu đối với con người ("mô hình này có thể hoàn thành loại tác vụ mà một chuyên gia phải mất X giờ để làm ở mức độ tin cậy 50%").

Bài học này vừa phân tích về phương pháp luận (cách khớp đường cong chân trời thời gian) vừa rèn luyện kỹ năng diễn giải (tại sao chân trời thời gian là cận trên chứ không phải dự báo triển khai thực tế). Hai kỹ năng này gắn liền chặt chẽ với nhau. Một đội ngũ kỹ sư hiểu rõ cách thức khớp đường cong chân trời thời gian sẽ không dễ dàng bị đánh lừa bởi những lời quảng cáo thổi phồng của nhà cung cấp so với một đội ngũ chỉ nhìn thấy con số "14 giờ" trên slide thuyết trình.

## Khái niệm

### Bối cảnh thành lập METR

- Thành lập: Tháng 12 năm 2023 (tiền thân là ARC Evals, được tách ra thành tổ chức phi lợi nhuận 501(c)(3) độc lập).
- Phạm vi hoạt động: đánh giá năng lực tự trị của các mô hình biên giới, chủ yếu ở giai đoạn tiền phát hành.
- Các phòng thí nghiệm đối tác: Anthropic, OpenAI (thực hiện nhiều đợt đánh giá trong giai đoạn 2025–2026).
- Các sản phẩm công bố đáng chú ý: Time Horizon 1.0 (tháng 3 năm 2025), Time Horizon 1.1 (tháng 1 năm 2026), các bài kiểm tra giám sát nguyên mẫu (prototype monitoring evaluations).

### Cách khớp đường cong Chân trời Thời gian (The Time Horizon fit)

Phương pháp luận (trích xuất từ bài báo và blog của METR):

1. Thu thập một bộ tác vụ trải dài từ các tác vụ mất vài phút đến hàng giờ đối với chuyên gia con người. Các bộ tác vụ hiện hành: HCAST (hơn 180 tác vụ), RE-Bench (71 tác vụ), SWAA.
2. Cho mô hình chạy trên từng tác vụ; ghi nhận kết quả thành công hoặc thất bại.
3. Khớp một đường cong logistic: $P(\text{thành công})$ là hàm số theo $\log(\text{thời gian hoàn thành của chuyên gia})$.
4. Chân trời thời gian chính là mốc thời gian của chuyên gia mà tại đó $P(\text{thành công}) = 0{,}5$.

Dạng khớp logistic là lựa chọn chuẩn xác vì năng lực thông thường có xu hướng tăng dần và tiệm cận mức bão hòa khi độ khó của tác vụ tăng lên. Mức 50% là một lựa chọn quy ước (hoàn toàn có thể chọn 10% hoặc 90%); METR báo cáo nhiều ngưỡng khác nhau trong bài báo chi tiết nhưng chọn mốc 50% làm con số tiêu đề vì nó trực quan và dễ hiểu nhất.

### Các con số ghi nhận tháng 1 năm 2026

Theo công bố của Time Horizon 1.1:

- Claude Opus 4.6: đạt mốc ~14 giờ ở độ tin cậy 50%, tính đến bản Time Horizon 1.1 (tháng 1 năm 2026).
- Thời gian nhân đôi (doubling time) trên các tác vụ dạng HCAST: đạt ~4,3 tháng (130,8 ngày) dựa trên phép khớp dữ liệu sau năm 2023 của Time Horizon 1.1; con số ~7 tháng là phép khớp toàn phần giai đoạn 2019–2025 từ bản Time Horizon 1.0 và được báo cáo song song trong TH1.1 bên cạnh con số sau 2023.

Đây chính là các con số cụ thể mà phần còn lại của Phase 15 quy chiếu tới. Chúng được trích dẫn trong RSP v3.0 của Anthropic (Bài 19) và FSF v3 của DeepMind (Bài 20) như một trong những phép đo đầu vào phục vụ đánh giá ngưỡng an toàn.

### Các bộ điểm chuẩn (Benchmark suites)

- **HCAST**: 189 tác vụ bao quát các lĩnh vực ML, an ninh mạng, kỹ thuật phần mềm và suy luận logic tổng quát. Thời lượng: từ 1 phút đến hơn 8 giờ làm việc của chuyên gia. Đây là bộ dữ liệu chính tạo nên con số chân trời thời gian.
- **RE-Bench**: 71 tác vụ kỹ thuật nghiên cứu ML có đường cơ sở đối chuẩn của chuyên gia con người. Đo lường xem liệu mô hình có thể thực hiện loại công việc R&D ML làm nền tảng cho ngưỡng AI R&D-4 hay không.
- **SWAA**: tập hợp các tác vụ ngắn hơn; hữu ích cho việc hiệu chuẩn phần đầu dưới của đường cong chân trời thời gian.

### Đánh giá giám sát nguyên mẫu (Prototype monitoring evaluations)

Nghiên cứu giai đoạn 2025–2026 của METR bao gồm các nguyên mẫu đánh giá khả năng giám sát: liệu một bộ giám sát có thể phát hiện mô hình đang thực hiện một tác vụ phụ lén lút (nằm ngoài những gì người vận hành yêu cầu) hay không; liệu tác tử có khả năng né tránh bộ giám sát hay không. Những thử nghiệm này đóng góp dữ liệu trực tiếp cho danh mục "Phá hoại Rào chắn an toàn" trong OpenAI PF v2 và việc giám sát căn chỉnh lừa dối trong DeepMind FSF v3 (Bài 20).

### Tại sao chân trời thời gian là cận trên

- **Công cụ được lý tưởng hóa (Idealized tooling)**: các tác vụ điểm chuẩn cung cấp cho mô hình các công cụ chuẩn chỉnh và dữ liệu định dạng hoàn hảo. Môi trường triển khai thực tế hỗn loạn hơn rất nhiều.
- **Không để lại hậu quả thực tế (No real consequences)**: mô hình không bao giờ phải chịu trách nhiệm thanh toán hóa đơn khách hàng, xóa dữ liệu thật hay liên hệ với con người thật. Triển khai thực tế tiềm ẩn các rủi ro không thể hoàn tác.
- **Thao túng ngữ cảnh đánh giá (Eval-context gaming)**: như đã phân tích ở Bài 1, mô hình hành xử khác biệt trong các bài kiểm tra. Báo cáo An toàn AI Quốc tế 2026 đã chứng minh điều này bằng thực nghiệm.
- **Không có phương sai từ phía người dùng hợp lệ (No legitimate user variance)**: các câu lệnh trong bài kiểm chuẩn có cấu trúc rõ ràng. Người dùng thật tạo ra các yêu cầu mơ hồ, phụ thuộc sâu vào ngữ cảnh.

Chân trời thời gian là mức trần năng lực trong các điều kiện tối ưu. Độ tin cậy triển khai thực tế là một con số khác, thấp hơn nhiều, và các nhóm kỹ sư bắt buộc phải tự đo lường trên phân phối dữ liệu riêng của mình.

### Vai trò của tổ chức đánh giá độc lập bên ngoài

Việc đánh giá độc lập bên ngoài là tối quan trọng vì các phòng thí nghiệm nội bộ luôn có động lực tối ưu hóa các chỉ số mà họ tự công bố. Vị thế độc lập của METR — một tổ chức 501(c)(3) với phương pháp luận công khai và các bài báo được bình duyệt khoa học — là một cơ chế giảm thiểu mang tính cấu trúc. Dù một mình nó chưa đủ (các phòng thí nghiệm vẫn kiểm soát những gì METR được tiếp cận), nhưng nó vượt trội hơn hẳn so với việc hoàn toàn không có bên thứ ba đánh giá.

### Cách ứng dụng các con số chân trời thời gian trong thực tế

- **Dùng làm bộ lọc năng lực**: nếu chân trời thời gian của mô hình thấp hơn nhiều so với thời gian làm việc của chuyên gia cho một tác vụ dự kiến, tuyệt đối không triển khai nó ở chế độ tự trị hoàn toàn (xem kỹ năng đóng gói ở Bài 1).
- **Dùng làm chỉ báo xu hướng**: thời gian nhân đôi cho bạn biết các biện pháp thực hành hiện tại sẽ duy trì được mức độ an toàn trong bao lâu nếu không có các biện pháp giảm thiểu mới.
- **Dùng làm giá trị tiên nghiệm (prior)**: con số chân trời 14 giờ là điểm khởi đầu. Hãy điều chỉnh giảm xuống cho phù hợp với phân phối tác vụ, chất lượng bộ công cụ và ngữ cảnh triển khai của bạn.

```figure
a5-horizon-fit
```

## Sử dụng

Tệp `code/main.py` triển khai phép khớp logistic giữa tỷ lệ thành công của tác vụ với $\log(\text{thời gian chuyên gia})$ dựa trên một tập kết quả tổng hợp. Chương trình xuất ra chân trời 50% (chỉ số tiêu đề của METR), chân trời 10% (tiêu chí thận trọng), và chân trời 90% (tiêu chí lạc quan). Chương trình cũng minh họa chân trời thời gian biến động ra sao khi tỷ lệ thành công bị thổi phồng giả tạo do mô hình thao túng ngữ cảnh đánh giá.

## Đóng gói

Tệp `outputs/skill-horizon-interpretation.md` dùng để đánh giá các tuyên bố về chân trời thời gian của nhà cung cấp và đưa ra bản phân tích khoảng cách giữa tuyên bố trên điểm chuẩn với thực tế triển khai.

## Bài tập

1. Chạy `code/main.py`. Xác nhận mốc chân trời 50% khớp chính xác với chân lý tổng hợp (ground truth). Bây giờ hãy chia đôi lưới thời gian tác vụ; liệu ước tính chân trời thời gian có thay đổi đáng kể không?

2. Đọc bài viết blog Time Horizon 1.1 của METR. Xác định những tác vụ cụ thể mà độ tin cậy đạt mức cao nhất và thấp nhất. Giải thích lý do tại sao lại tồn tại khoảng cách đó.

3. Đọc tài liệu "Đo lường Năng lực AI Tự trị" của METR. Liệt kê các danh mục tác vụ của HCAST. Chọn một danh mục mà bạn sẽ gán trọng số lớn hơn cho một tác vụ sản xuất và nêu rõ lý do.

4. Bổ sung cơ chế thao túng ngữ cảnh đánh giá vào trình mô phỏng: đảo ngược ngẫu nhiên khoảng ~20% các tác vụ thất bại thành thành công. Báo cáo con số chân trời thời gian mới. Con số này mô phỏng sát sao tác động của tỷ lệ thao túng 20% lên chỉ số quan sát được.

5. Thiết kế một quy trình đánh giá chân trời thời gian nội bộ dựa trên danh sách lỗi tồn đọng (bug backlog) của nhóm bạn hoặc một tập tác vụ đại diện. Mô tả quy trình thu thập dữ liệu, phép khớp đường cong, và ý nghĩa của kết quả thu được. So sánh với các con số của METR.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| METR | "Tổ chức đánh giá độc lập" | Tiền thân là ARC Evals; tổ chức 501(c)(3) phi lợi nhuận độc lập từ tháng 12/2023 |
| Chân trời Thời gian (Time Horizon) | "Thước đo năng lực" | Độ dài tác vụ chuyên gia ở độ tin cậy 50%, trích xuất từ phép khớp logistic |
| HCAST | "Bộ điểm chuẩn chính của METR" | Hơn 180 tác vụ trải dài từ 1 phút đến hơn 8 giờ |
| RE-Bench | "Kỹ thuật nghiên cứu" | 71 tác vụ kỹ thuật nghiên cứu ML với đường cơ sở của con người |
| SWAA | "Bộ tác vụ ngắn" | Dùng để hiệu chuẩn đoạn dưới của đường cong chân trời thời gian |
| Thời gian nhân đôi (Doubling time) | "Tốc độ tăng trưởng" | Thời gian để chân trời 50% tăng gấp đôi; ~7 tháng trên toàn bộ HCAST |
| Thao túng ngữ cảnh đánh giá (Eval-context gaming) | "Mô hình hành xử khác biệt" | Khoảng cách hành vi đã được ghi nhận giữa bài kiểm tra và môi trường triển khai |
| Cận trên (Upper bound) | "Chân trời là mức trần" | Chân trời điểm chuẩn luôn cao hơn độ tin cậy triển khai thực tế dưới tải lớn |

## Đọc thêm

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — thông số kỹ thuật các bộ HCAST, RE-Bench, SWAA.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — bài báo gốc về chân trời thời gian.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — phương pháp luận và các số liệu hiện hành.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — bảng theo dõi trực tiếp.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về các phép đo lường của METR.
