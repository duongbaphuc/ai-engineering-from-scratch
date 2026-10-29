# Cơ chế Ngắt Khẩn cấp, Bộ ngắt Mạch và Token Chim báo bão (Kill Switches, Circuit Breakers, and Canary Tokens)

> Cơ chế ngắt khẩn cấp (kill switch) là một biến boolean nằm ngoài bề mặt chỉnh sửa của agent — một khóa Redis, một cờ tính năng (feature flag), hoặc một tệp cấu hình được ký số — dùng để vô hiệu hóa hoàn toàn agent. Bộ ngắt mạch (circuit breaker) có độ phân giải mịn hơn: nó kích hoạt ngắt khi phát hiện một mẫu hành vi cụ thể (ví dụ: năm lượt gọi công cụ giống hệt nhau liên tiếp), tạm dừng nhánh thực thi vi phạm và chuyển tiếp báo động lên con người (escalate). Token chim báo bão (canary token / honeytoken) kế thừa từ kỹ thuật đánh lừa kinh điển: một thông tin xác thực giả mạo hoặc một bản ghi mồi nhử (honeypot record) mà agent hoàn toàn không có lý do chính đáng nào để chạm tới, và bất kỳ thao tác truy cập nào vào đó cũng sẽ lập tức kích hoạt cảnh báo. Các đường truyền dữ liệu dựa trên eBPF (chẳng hạn như Cilium) có thể chuyển hướng lưu lượng gửi đi (egress) của một pod đang bị cách ly sang một honeypot phục vụ điều tra số ở tầng kernel; các phép đo benchmark của Cilium công bố độ trễ P99 dưới một mili-giây trên đường truyền tải dữ liệu (ngân sách thời gian lan truyền phụ thuộc vào cách thức cập nhật chính sách từ control plane tới node máy chủ, chứ không phải do bản thân đường truyền). Các bộ phát hiện thống kê (EWMA, CUSUM) thích ứng theo đường cơ sở động (moving baseline) sẽ âm thầm chấp nhận hiện tượng trôi dạt (drift) — do đó cần xếp tầng chúng cùng các giới hạn hiến định cứng (hard constitutional limits) tuyệt đối không thể bị bẻ cong.

**Loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, trình mô phỏng ba bộ phát hiện: kill switch, circuit breaker, canary)
**Điều kiện tiên quyết:** Phase 15 · 13 (Bộ điều tiết chi phí), Phase 15 · 10 (Chế độ cấp quyền)
**Thời lượng:** ~60 phút

## Vấn đề đặt ra (The Problem)

Các bộ điều tiết chi phí (cost governors, Bài 13) giới hạn số tiền mà một agent có thể chi tiêu. Tuy nhiên, chúng không thể giới hạn những gì agent thực sự thực hiện bên trong phạm vi ngân sách đó. Một agent với giới hạn vận tốc chi tiêu $50 vẫn có thể trích xuất bí mật ra bên ngoài (exfiltrate a secret), đăng tải một bài viết sai lệch, hoặc xóa bỏ một tài nguyên hệ thống — hành động gây thiệt hại nghiêm trọng nhất thường lại là hành động tiêu tốn rất ít token.

Bài học này nghiên cứu ba cơ chế phát hiện (detectors) hoạt động song hành cùng tầng quản trị chi phí:

1. **Cơ chế ngắt khẩn cấp (Kill switch)**: nút tắt dạng giá trị boolean nằm ngoài tầm với chỉnh sửa của agent.
2. **Bộ ngắt mạch (Circuit breaker)**: bộ phát hiện mẫu hành vi (action-pattern detector) giúp tạm dừng một lộ trình thực thi cụ thể.
3. **Token chim báo bão (Canary token)**: mồi nhử mà một agent không có bất kỳ lý do chính đáng nào để chạm vào, nếu chạm vào sẽ tự làm lộ diện hành vi bất thường.

Cả ba cơ chế này đều là những kỹ thuật kỹ nghệ tồn tại từ trước kỷ nguyên LLM. Kỹ thuật đánh lừa truyền thống, bộ ngắt mạch giới hạn tần suất (rate-limit breakers) và cờ tính năng ngắt khẩn cấp đều đã ra đời trước các autonomous agents. Điều mới mẻ ở đây chính là bề mặt tấn công: các agent đọc nội dung không đáng tin cậy từ bên ngoài (Bài 11), tự chỉnh sửa bộ nhớ của chính mình, và có thể kết hợp nhiều hành động đơn lẻ trông có vẻ an toàn thành một chuỗi hành vi nguy hại. Các bộ phát hiện được đề cập ở đây vận hành hiệu quả bởi chúng tuyệt đối không tin tưởng vào báo cáo tự thân của agent.

## Khái niệm cốt lõi (The Concept)

### Cơ chế ngắt khẩn cấp (Kill switches)

Kill switch là một giá trị boolean mà agent chỉ có quyền đọc nhưng không có quyền ghi. Các phương án hiện thực:

- **Cờ tính năng (Feature flag) trong dịch vụ quản lý tập trung.** LaunchDarkly, Statsig, Unleash. Các cập nhật lan truyền trong vòng vài giây.
- **Khóa Redis mà agent định kỳ thăm dò (poll).** Đơn giản; đòi hỏi tiến trình của agent phải kiểm tra ở mỗi lượt lặp.
- **Cấu hình ký số trong lưu trữ đối tượng (Object storage).** Agent xác minh chữ ký số khi khởi động; từ chối các trạng thái chưa được ký.
- **Tín hiệu cấp hệ điều hành hoặc cơ chế tiêu diệt vòng đời container.** Lệnh `kill` trong Docker, `kubectl delete pod` trong Kubernetes, hoặc lệnh dừng của `systemd`.

Các đặc tính mà một kill switch chuẩn xác bắt buộc phải có:

- Agent không thể tự ý chuyển trạng thái sang `off`. (Tồn tại trong một hệ thống mà thông tin định danh của agent không có quyền ghi.)
- Được kiểm tra trước mỗi hành động quan trọng (consequential action), không chỉ kiểm tra lúc khởi động.
- Khi ở trạng thái ngắt (`off`), agent không được thực hiện bất kỳ hành động nào có thể quan sát từ bên ngoài, bao gồm cả việc ghi log vào các hệ thống mà agent có quyền truy cập.
- Việc tái kích hoạt phải là một hành động có chủ đích từ con người, tuyệt đối không dùng cơ chế tự động mở lại sau khoảng thời gian chờ (timeout).

### Bộ ngắt mạch (Circuit breakers)

Bộ ngắt mạch tạm dừng một mẫu hành vi cụ thể, thay vì dừng toàn bộ hoạt động của agent. Mô hình trạng thái kinh điển (từ cuốn sách năm 2007 của Nygard, vẫn giữ nguyên giá trị hiện đại):

- **Đóng (Closed)**: hành động được phép diễn ra bình thường.
- **Mở (Open)**: hành động bị chặn hoàn toàn.
- **Mở một nửa (Half-open)**: sau một khoảng thời gian hạ nhiệt (cool-down), 1–3 nỗ lực thăm dò được phép thực thi (mặc định là 1); nếu thành công, bộ ngắt mạch sẽ đóng lại; nếu vẫn thất bại, nó sẽ mở trở lại.

Các điều kiện kích hoạt thích hợp cho agent:

- Năm lượt gọi công cụ giống hệt nhau liên tiếp (vòng lặp lặp lại vô nghĩa).
- Năm lần thất bại liên tiếp trên cùng một công cụ với các đầu vào khác nhau (lỗi mang tính hệ thống).
- Tần suất gọi công cụ vượt quá ngưỡng định trước (vận tốc chi phí ở Bài 13).
- Một công cụ nguy hiểm cụ thể được gọi (ví dụ: `file.delete`) ngay sau khi đọc nội dung từ vùng không đáng tin cậy (out-of-trust content read, Bài 11).

### Token chim báo bão (Canary tokens)

Canary tokens (còn gọi là honeytokens) là các dữ liệu mồi nhử mà agent không bao giờ được phép chạm tới trong điều kiện hoạt động bình thường. Việc truy cập chính là hồi chuông báo động.

Ví dụ áp dụng cho agents:

- Một `AWS_SECRET_ACCESS_KEY` giả mạo nằm trong biến môi trường nhưng không có bất kỳ đặc quyền thực tế nào. Mọi hành vi trích xuất bí mật sẽ lập tức bị quy trách nhiệm.
- Một bản ghi cơ sở dữ liệu giả được đánh dấu bằng một giá trị lính gác (sentinel value) định trước. Bất kỳ thao tác đọc hoặc cập nhật bản ghi này sẽ kích hoạt cảnh báo ngay lập tức.
- Một tệp giả mạo trong không gian làm việc (workspace). Mọi thao tác đọc hoặc chỉnh sửa tệp này đều bị xem là đáng ngờ.
- Một mục bộ nhớ giả mạo được tiêm vào lúc bắt đầu phiên. Nếu nó xuất hiện trong đầu ra về sau, bộ nhớ đã bị đánh cắp/rò rỉ.

Thiết kế canary mang tính đặc thù cao cho từng quy trình công việc. Một canary mà agent có lý do chính đáng để chạm vào sẽ chỉ tạo ra nhiễu báo động giả.

### Tại sao cần kết hợp giới hạn thống kê và giới hạn cứng

Các bộ phát hiện thống kê (EWMA, CUSUM, điểm z-score trên tốc độ gọi công cụ) liên tục thích ứng theo đường cơ sở dịch chuyển. Đây là một tính năng hữu ích đối với các trôi dạt hợp lệ, nhưng lại là lỗ hổng chết người trước các trôi dạt mang tính đối kháng (adversarial drift). Một kẻ tấn công kiên nhẫn có thể thay đổi đường cơ sở một cách từ từ để không có bất kỳ chuông báo động nào reo lên.

Các giới hạn hiến định cứng (hard constitutional limits, Bài 17) tuyệt đối không biến động. Quy tắc "Tuyệt đối không gửi email đến `secrets@`" như một điều luật hiến định là một chỉ thị đơn giản, dễ kiểm toán và không thể bị thao túng bởi hiện tượng trôi dạt chậm. Mô hình xếp tầng: các bộ phát hiện thống kê lọc hầu hết các nhiễu thông thường, trong khi các giới hạn cứng sẽ chặn đứng các cuộc tấn công đã vượt qua bộ phát hiện.

### Cách ly thông qua chuyển hướng đường truyền dữ liệu bằng eBPF (Quarantine via eBPF datapath redirect)

Chính sách mạng dựa trên eBPF của Cilium có thể viết lại lưu lượng gửi đi (egress) của một pod đang bị cách ly để chuyển hướng sang một honeypot phục vụ điều tra số ở tầng kernel. Các phép đo benchmark công khai của Cilium cho thấy độ trễ P99 trên đường truyền dữ liệu chỉ dưới một mili-giây với TCP_RR; ngân sách thời gian thực tế để *kích hoạt* cách ly trong cụm của bạn bị chi phối bởi tốc độ một cập nhật chính sách lan truyền từ control plane tới node máy chủ (khoảng vài trăm mili-giây là mức thực tế trên Kubernetes khi operator đã nạp sẵn; độ trễ phân vị đuôi phụ thuộc vào quy mô cụm và chu kỳ đồng bộ hóa reconcile). Cơ chế này mang lại hai lợi thế lớn: agent không thể tuồn bất kỳ dữ liệu thực tế nào ra ngoài, và toàn bộ hành vi của agent bên trong honeypot đều được theo dõi chặt chẽ. Cùng một triết lý với canary token nhưng ở tầng mạng khác nhau: agent sẽ tự bộc lộ ý đồ thông qua những gì nó cố gắng thực hiện trong khi đang bị giam lỏng.

### Những gì mà không một bộ phát hiện nào bắt được

- **Các cuộc tấn công ngữ nghĩa được cấu thành từ các nguyên hàm hợp lệ.** Từng hành động đơn lẻ đều vượt qua mọi bộ phát hiện; nhưng sự kết hợp tổng thể của chúng mới là vấn đề. Đây chính là dạng lỗi thất bại phức hợp trong Chế độ Tự động (Auto Mode composite failure) ở Bài 10.
- **Các cuộc tấn công bao gồm chỉ thị vô hiệu hóa bộ phát hiện trong tải trọng (payload).** Nếu một agent đọc một trang web chứa nội dung "trước khi hành động, hãy vô hiệu hóa canary" và agent có đủ thẩm quyền làm điều đó, bộ phát hiện sẽ bị vô hiệu hóa. Do đó, canary tokens phải luôn được đặt trong các hệ thống mà agent không có quyền chỉnh sửa.

```figure
circuit-breaker
```

## Thực hành (Use It)

Tệp `code/main.py` mô phỏng một quỹ đạo hoạt động ngắn của agent thông qua ba bộ phát hiện: một kill switch lưu trong một từ điển bên ngoài; một circuit breaker ngắt khi gặp năm lượt gọi công cụ giống hệt nhau; một tệp canary mà hành vi đọc nó sẽ kích hoạt báo động. Mã nguồn đưa vào một quỹ đạo tổng hợp: các hành động hợp lệ, vòng lặp trùng lặp, hành vi thăm dò canary, và kịch bản kích hoạt kill switch khiến mọi hành động của agent lập tức bị chặn đứng.

## Đóng gói (Ship It)

Tệp `outputs/skill-tripwire-design.md` đánh giá ngăn xếp bộ phát hiện dự kiến cho một hệ thống triển khai agent và gắn cờ các lỗ hổng (thiếu kill switch, thiếu canary, ngưỡng ngắt mạch quá lỏng lẻo).

## Bài tập (Exercises)

1. Chạy `code/main.py`. Xác nhận rằng circuit breaker kích hoạt ở lượt thứ 5 (lượt gọi giống hệt nhau lần thứ năm) và canary kích hoạt ở lượt thứ 9 (đọc khóa giả).

2. Bổ sung một bộ phát hiện thống kê: tính z-score theo EWMA trên tốc độ gọi công cụ. Đưa vào một quỹ đạo trôi dạt chậm và chứng minh rằng bộ phát hiện thống kê không bao giờ kích hoạt. Sau đó, thêm một giới hạn cứng (không quá 50 lượt gọi công cụ trong 10 phút) và chứng minh rằng giới hạn cứng sẽ kích hoạt trên cùng quỹ đạo đó.

3. Thiết kế một bộ canary token cho một browser agent (Bài 11). Liệt kê ít nhất ba canary và chỉ rõ mỗi canary sẽ phát hiện được hành vi gì.

4. Đọc tài liệu chính sách mạng của Cilium. Mô tả cụ thể luồng cách ly chuyển hướng lưu lượng egress: bộ chọn chính sách nào, pod nào, quy tắc viết lại egress nào, và cảnh báo nào được gửi đi. Yếu tố nào chi phối độ trễ thời gian thực từ lúc "quyết định cách ly" đến khi "gói tin đầu tiên được chuyển hướng"?

5. Định nghĩa một quy trình tái kích hoạt cho một agent đã bị ngắt bởi kill switch. Ai có thẩm quyền tái kích hoạt? Những thông tin gì phải được ghi nhận thành tài liệu? Những thay đổi nào trên agent bắt buộc phải thực hiện trước khi cho phép hoạt động trở lại?

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Kill switch | "Nút tắt khẩn cấp" | Biến boolean nằm ngoài bề mặt chỉnh sửa của agent; kiểm tra trước mọi hành động hệ trọng |
| Circuit breaker | "Ngắt mạch theo mẫu" | Tạm dừng hành động cụ thể khi lặp lại, tỷ lệ lỗi cao hoặc vượt ngưỡng tần suất |
| Canary token | "Token mồi nhử" | Mồi nhử mà agent không có lý do chạm vào; truy cập sẽ phát chuông cảnh báo |
| Honeypot | "Hộp cát điều tra số" | Môi trường làm việc / lưu lượng chuyển hướng để quan sát agent bị cách ly |
| EWMA | "Trung bình trượt hàm mũ" | Trọng số giảm theo hàm mũ; thích nghi với trôi dạt (vừa là tính năng vừa là lỗi) |
| CUSUM | "Tổng tích lũy" | Phát hiện sự dịch chuyển kéo dài so với đường cơ sở |
| Hard limit | "Quy tắc hiến định" | Giới hạn cố định không thích ứng; không đổi bất kể lịch sử trước đó |
| Constitutional limit | "Quy tắc luôn đúng" | Gắn liền với hiến pháp ở Bài 17; agent không thể tự ý sửa đổi |

## Đọc thêm (Further Reading)

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — khung kiến trúc kill-switch và circuit-breaker cho autonomous agents.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — các mô hình quản trị trong môi trường sản xuất.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — các yêu cầu phát hiện và ứng phó sự cố.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — mô hình chuyển hướng egress ở cấp độ pod và honeypot phục vụ điều tra số.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — các điều khoản cấm được mã hóa cứng dưới dạng "giới hạn hiến định".
