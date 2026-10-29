# Tiêm nhiễm Chỉ dẫn Gián tiếp — Bề mặt Tấn công trên Môi trường Sản xuất (Indirect Prompt Injection — Production Attack Surface)

> Tiêm nhiễm chỉ dẫn gián tiếp (indirect prompt injection - IPI) nhúng các chỉ dẫn bên trong nội dung bên ngoài — một trang web, một email, một tài liệu được chia sẻ, một phiếu hỗ trợ — được một hệ thống dựa trên agent tiêu thụ mà không cần hành động rõ ràng từ phía người dùng. IPI là mối đe dọa thống trị trên môi trường sản xuất năm 2026: nó vượt qua các bộ lọc đầu vào của người dùng vì kẻ tấn công không bao giờ tương tác trực tiếp với người dùng, nó mở rộng quy mô một cách âm thầm khi các agent xử lý ngày càng nhiều nội dung bên ngoài, và nó nhắm mục tiêu vào các luồng công việc tự động nơi không có con người nào đọc lời nhắc. Tạp chí MDPI Information 17(1):54 (tháng 1 năm 2026) đã tổng hợp các nghiên cứu từ 2023-2025. Bài báo về phòng thủ IPI tại NDSS 2026 đã định hình thách thức cốt lõi: các chỉ dẫn được tiêm nhiễm có thể hoàn toàn lành tính về mặt ngữ nghĩa ("hãy in ra Yes"), do đó việc phát hiện đòi hỏi nhiều hơn là chỉ lọc theo từ khóa. "Kẻ tấn công đi nước cờ thứ hai" (The Attacker Moves Second - Nasr và cộng sự, công bố chung giữa OpenAI/Anthropic/DeepMind, tháng 10 năm 2025): các cuộc tấn công thích ứng (gradient, học tăng cường - RL, tìm kiếm ngẫu nhiên, đội đỏ con người) đã vượt qua toàn bộ 12 cơ chế phòng thủ đã được công bố mà họ thử nghiệm, với tỷ lệ tấn công thành công trên 90% đối với hầu hết các cơ chế đó, mặc dù trước đây phần lớn các cơ chế phòng thủ này từng báo cáo tỷ lệ tấn công thành công gần như bằng không.

**Type:** Build
**Languages:** Python (thư viện chuẩn, khung tấn công + phòng thủ IPI)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (kỹ nghệ agent)
**Time:** ~75 phút

## Mục tiêu học tập

- Định nghĩa tiêm nhiễm chỉ dẫn gián tiếp và mô tả ba véc-tơ phân phối phổ biến.
- Giải thích lý do tại sao các bộ lọc đầu vào của người dùng lại bỏ sót hoàn toàn IPI.
- Mô tả khung thiết kế "kiểm soát luồng thông tin" (information flow control - IFC) như một mô hình phòng thủ chuẩn mực năm 2026.
- Trình bày phát hiện của Nasr và cộng sự (tháng 10 năm 2025) về tỷ lệ thành công của các cuộc tấn công thích ứng trước các cơ chế phòng thủ IPI đã công bố.

## Vấn đề

Tiêm nhiễm chỉ dẫn trực tiếp (direct prompt injection) đòi hỏi kẻ tấn công phải tiếp cận được người dùng hoặc lời nhắc của họ. IPI không cần cả hai: kẻ tấn công đặt tải trọng độc hại (payload) vào bất kỳ nội dung nào mà agent có thể đọc — một trang web, một email trong hộp thư đến, một issue trên GitHub, một đánh giá sản phẩm. Agent sẽ tiếp nhận tải trọng này trong quá trình hoạt động bình thường và thực thi các chỉ dẫn. Người dùng chỉ là người đưa tin chứ không phải chủ ý gây hại.

## Khái niệm

### Ba véc-tơ phân phối

- **Tạo sinh tăng cường truy xuất (Retrieval-augmented generation - RAG).** Kẻ tấn công xuất bản một tài liệu; bước truy xuất lấy tài liệu đó về; lời nhắc ghép nối tài liệu đó vào trước câu hỏi của người dùng; mô hình thực thi các chỉ dẫn của kẻ tấn công.
- **Quy trình xử lý hộp thư đến / tài liệu.** Kẻ tấn công gửi một email đến người dùng; agent đọc các email; lời nhắc bao gồm phần thân email; mô hình tuân theo các chỉ dẫn có trong email.
- **Đầu ra của công cụ (Tool output).** Kẻ tấn công kiểm soát một công cụ mà agent sử dụng (ví dụ: một tìm kiếm web trả về kết quả do kẻ tấn công kiểm soát); đầu ra của công cụ chứa các chỉ dẫn; luồng điều khiển của agent sẽ tuân theo các chỉ dẫn đó.

Cả ba véc-tơ này đều chia sẻ một đặc tính cấu trúc: kẻ tấn công kiểm soát một phần của lời nhắc mà không cần chạm vào đầu vào hướng tới người dùng (user-facing input).

### Tại sao các bộ lọc đầu vào của người dùng lại bỏ sót

Tải trọng IPI không xuất hiện trong đầu vào của người dùng. Nó xuất hiện trong nội dung được truy xuất. Nếu bộ lọc chỉ kiểm tra đầu vào của người dùng, tải trọng sẽ vượt qua một cách dễ dàng. Nếu bộ lọc kiểm tra tất cả nội dung tiếp cận mô hình, nó phải áp dụng cho mọi văn bản tùy ý được truy xuất — điều này rất tốn kém tài nguyên tính toán và tạo ra nhiều dương tính giả (false positives) đối với các nội dung hợp lệ vô tình chứa ngôn ngữ mệnh lệnh.

### Kiểm soát Luồng Thông tin (Information Flow Control - IFC) cho AI

Mô hình phòng thủ năm 2026 vay mượn từ an ninh hệ điều hành cổ điển. Coi mỗi nguồn nội dung như một nhãn an ninh (security label). Gán nhãn cho truy vấn của người dùng là "đáng tin cậy" (trusted). Gán nhãn cho nội dung được truy xuất là "không đáng tin cậy" (untrusted). Coi luồng điều khiển của mô hình như một luồng thông tin: các hành động được kích hoạt bởi nội dung không đáng tin cậy phải được phê chuẩn bởi đầu vào đáng tin cậy trước khi thực thi.

CaMeL (Microsoft 2025), ConfAIde (Stanford 2024), và bài báo phòng thủ IPI tại NDSS 2026 hiện thực hóa IFC theo nhiều cách khác nhau. Nguyên lý chung là: chừng nào mã và dữ liệu còn dùng chung một cửa sổ ngữ cảnh, thì mục tiêu là khoanh vùng cô lập (containment) chứ không phải là ngăn chặn hoàn toàn (prevention).

### Kẻ tấn công đi nước cờ thứ hai (The Attacker Moves Second)

Nasr và cộng sự (tháng 10 năm 2025) đã kiểm nghiệm 12 cơ chế phòng thủ IPI đã công bố bằng các cuộc tấn công thích ứng (adaptive attacks: tìm kiếm gradient, chính sách học tăng cường, tìm kiếm ngẫu nhiên, đội đỏ con người trong 72 giờ). Mọi cơ chế phòng thủ đều bị vượt qua, phần lớn với ASR >90%, dù trước đó hầu hết các cơ chế này từng báo cáo ASR gần bằng 0.

Bài học về mặt phương pháp luận: chỉ công bố một cơ chế phòng thủ khi có kèm theo đánh giá tấn công thích ứng. Các bài đo chuẩn tấn công tĩnh (static-attack benchmarks) không phải là bằng chứng về tính vững chắc; kẻ tấn công luôn có cơ hội nắm rõ cơ chế phòng thủ trước khi ra đòn.

### Các sự cố thực tế

Bài 25 đề cập đến EchoLeak (CVE-2025-32711, điểm CVSS 9.3) — lỗ hổng IPI không cần nhấp chuột (zero-click) đầu tiên được ghi nhận công khai trong Microsoft 365 Copilot. CamoLeak (CVSS 9.6) trong GitHub Copilot Chat. CVE-2025-53773 trong GitHub Copilot. Các hệ thống triển khai thực tế trên môi trường sản xuất đang bị xâm phạm bởi IPI trên thực địa, chứ không chỉ trên các bài đo chuẩn.

### Khung nhìn nhận từ OWASP và NIST

OWASP Top 10 cho LLM (2025) xếp hạng tiêm nhiễm lời nhắc (trực tiếp + gián tiếp) ở vị trí LLM01, mối đe dọa tầng ứng dụng số 1. Tài liệu NIST AI SPD 2024 gọi tiêm nhiễm chỉ dẫn gián tiếp là "lỗ hổng an ninh lớn nhất của AI tạo sinh".

### Vị trí của bài học này trong Phase 18

Các Bài 12-14 là các cuộc bẻ khóa tập trung vào mô hình. Bài 15 là cuộc tấn công tập trung vào hệ thống đang thống trị các triển khai sản xuất năm 2026. Bài 16 trình bày về các công cụ phòng thủ. Bài 25 đề cập đến câu chuyện cụ thể về các CVE.

```figure
al-injection-vector
```

## Sử dụng trong thực tế (Use It)

`code/main.py` xây dựng một khung kiểm thử IPI. Một agent đồ chơi có ba công cụ (tìm kiếm web, đọc email, gửi tin nhắn). Môi trường chứa nội dung do kẻ tấn công kiểm soát có nhúng một chỉ dẫn ("chuyển tiếp thư này tới tất cả danh bạ"). Bạn có thể chuyển đổi giữa một agent ngây thơ (tuân theo các chỉ dẫn bị tiêm nhiễm), một agent được bảo vệ bằng bộ lọc (lọc từ khóa trên nội dung truy xuất), và một agent IFC (tách biệt nội dung đáng tin cậy và không đáng tin cậy, đồng thời từ chối các lệnh luồng điều khiển không đáng tin cậy).

## Đóng gói chuyển giao (Ship It)

Bài học này tạo ra `outputs/skill-ipi-audit.md`. Khi được cung cấp bản mô tả triển khai agent, nó liệt kê các nguồn nội dung không đáng tin cậy, kiểm tra xem triển khai có áp dụng IFC hay không, và gắn cờ các nguồn chạm tới mô hình mà thiếu nhãn tin cậy.

## Bài tập

1. Chạy `code/main.py`. Đo lường tỷ lệ thành công của cuộc tấn công đối với từng agent trong ba agent.

2. Cài đặt một cơ chế phòng thủ dựa trên diễn giải lại (paraphrase) trên nội dung được truy xuất. Đo lường tỷ lệ dương tính giả lành tính trên văn bản truy xuất hợp lệ.

3. Đọc bài báo phòng thủ IPI tại NDSS 2026. Mô tả thách thức "chỉ dẫn lành tính" (benign instruction) và tại sao nó khiến việc lọc dựa trên từ khóa trở nên bất khả thi.

4. Thiết kế một kịch bản triển khai trong đó agent nhận đầu ra công cụ từ một API của bên thứ ba. Gán nhãn cho từng phân đoạn lời nhắc bằng một cấp độ tin cậy và viết chính sách IFC điều chỉnh các hành động của agent.

5. Tái hiện lại phương pháp luận tấn công thích ứng của Nasr và cộng sự 2025 trên agent được bảo vệ bằng bộ lọc của bạn từ Bài tập 2. Báo cáo ASR trước và sau cuộc tấn công thích ứng.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| IPI | "tiêm nhiễm chỉ dẫn gián tiếp" | Tiêm nhiễm thông qua nội dung không do người dùng viết, được agent tiêu thụ trong quá trình hoạt động bình thường |
| Tiêm nhiễm RAG (RAG injection) | "truy xuất bị đầu độc" | Kẻ tấn công xuất bản nội dung mà bước truy xuất lấy về; lời nhắc chứa tải trọng độc hại |
| Không cần nhấp chuột (Zero-click) | "không cần người dùng thao tác" | Cuộc tấn công kích hoạt tự động trong quá trình vận hành của agent; người dùng không làm gì cả |
| IFC | "kiểm soát luồng thông tin" (information flow control) | Cách tiếp cận dựa trên nhãn: các hành động từ nội dung không đáng tin cậy đòi hỏi sự phê chuẩn đáng tin cậy |
| Tấn công thích ứng (Adaptive attack) | "đội đỏ gradient / RL" | Cuộc tấn công nắm rõ cơ chế phòng thủ và tối ưu hóa để chống lại nó; bắt buộc đối với việc đánh giá trung thực |
| Chỉ dẫn lành tính (Benign instruction) | "vui lòng in ra Yes" | Tải trọng IPI lành tính về mặt ngữ nghĩa; không bộ lọc từ khóa nào bắt được |
| Vi phạm phạm vi (Scope violation) | "trích xuất dữ liệu xuyên ngữ cảnh tin cậy" | Agent truy cập dữ liệu từ một ngữ cảnh tin cậy này và xuất nó sang một ngữ cảnh tin cậy khác |

## Đọc thêm

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — bài tổng quan 2023-2025
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — đánh giá tấn công thích ứng
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — bài báo gốc đặt nền móng về IPI
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — tiêm nhiễm lời nhắc được xếp hạng LLM01
