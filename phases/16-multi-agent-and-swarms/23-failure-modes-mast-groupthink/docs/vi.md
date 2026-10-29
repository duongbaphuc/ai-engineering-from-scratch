# Các Dạng thức Thất bại — Phân loại MAST, Tư duy Tập thể, Đơn văn hóa và Lỗi Xếp tầng (Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors)

> Hệ thống phân loại tham chiếu chuẩn mực cho năm 2026 là **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), được đúc kết từ 1.642 dấu vết thực thi (execution traces) trên 7 hệ đa tác tử (MAS - Multi-Agent System) nguồn mở hiện đại nhất, chỉ ra **tỷ lệ thất bại từ 41% đến 86,7%**. Ba nhóm nguyên nhân gốc rễ bao gồm: **Các vấn đề về Đặc tả (Specification Problems)** (41,77%) — sự nhập nhằng về vai trò, định nghĩa tác vụ mơ hồ; **Lỗi Phối hợp (Coordination Failures)** (36,94%) — đứt gãy giao tiếp, lệch pha đồng bộ trạng thái; **Lỗ hổng Xác minh (Verification Gaps)** (21,30%) — thiếu bước thẩm định, vắng bóng các chốt kiểm soát chất lượng. Nhóm thất bại mang tính **Tư duy Tập thể (Groupthink)** (arXiv:2508.05687) bổ sung thêm: sụp đổ do đơn văn hóa (monoculture collapse - các tác tử dùng chung mô hình nền dẫn đến sai lệch tương quan), thiên kiến phục tùng (conformity bias - các tác tử củng cố sai sót của nhau), khiếm khuyết lý thuyết tâm trí (theory of mind), động lực động cơ hỗn hợp (mixed-motive dynamics) và sự cố độ tin cậy xếp tầng (cascading reliability failures). Ví dụ về lỗi xếp tầng: cơn bão thử lại (retry storm) khi lỗi thanh toán kích hoạt thử lại đơn hàng, dẫn đến thử lại việc kiểm tra kho, làm sập dịch vụ kho hàng (tải tăng gấp 10 lần trong vài giây — đòi hỏi phải có bộ ngắt mạch - circuit breaker). Đầu độc bộ nhớ (memory poisoning): ảo giác của một tác tử lọt vào bộ nhớ dùng chung, các tác tử phía sau xem đó là sự thật; độ chính xác suy giảm từ từ khiến việc chẩn đoán nguyên nhân gốc rễ trở nên cực kỳ khó khăn. Nghiên cứu **STRATUS** (NeurIPS 2025) ghi nhận mức cải thiện thành công 1,5 lần trong việc giảm thiểu sự cố nhờ triển khai các tác tử chuyên biệt: phát hiện / chẩn đoán / kiểm chứng. Bài học này xem xét các dạng thức thất bại như những mục tiêu kỹ thuật thiết kế hạng nhất.

**Type:** Learn  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)  
**Time:** ~75 phút  

## Vấn đề (Problem)

Các hệ đa tác tử thất bại từ 41% đến 86,7% thời gian khi xử lý các tác vụ thực tế (Cemri et al. 2025 đo đạc con số này trên 7 hệ MAS nguồn mở hàng đầu). Bạn không thể gỡ lỗi cho tình trạng này bằng cách đơn thuần "bổ sung thêm nhiều tác tử hơn". Các sự cố này bắt nguồn từ những nguyên nhân mang tính cấu trúc. Hệ phân loại MAST cung cấp cho bạn các phạm trù lỗi rõ ràng. Bài học này ánh xạ từng nhóm lỗi sang một mẫu hình phát hiện, chẩn đoán và khắc phục cụ thể để các con số thống kê không còn là những dữ liệu trừu tượng.

Thực tiễn kỹ thuật sản xuất của năm 2026 coi các dạng thức thất bại là dữ liệu đầu vào cho quá trình thiết kế kiến trúc. Hệ thống của bạn chỉ được xem là "đạt chuẩn" khi bạn có thể chỉ rõ từng danh mục lỗi của MAST và gọi tên chính xác cơ chế giảm thiểu sự cố mà bạn đã đưa vào vận hành.

## Khái niệm (Concept)

### Các nhóm nguyên nhân của MAST

**Các vấn đề về Đặc tả (Specification Problems - 41,77% số ca thất bại):** Tác vụ của tác tử không được định nghĩa đủ chặt chẽ. Ví dụ:

- Sự nhập nhằng vai trò (Role ambiguity): Cả hai tác tử đều nghĩ rằng mình là người đánh giá mã (reviewer).
- Tác vụ thiếu đặc tả: Yêu cầu "hãy tóm tắt nội dung này" trong khi người dùng thực tế lại muốn nhấn mạnh vào một góc nhìn cụ thể.
- Tiêu chí thành công ngầm định: Tác tử không có cách nào biết được liệu nó đã hoàn thành nhiệm vụ thành công hay chưa.

Biện pháp giảm thiểu:
- Soạn thảo các hợp đồng vai trò (role contracts) tường minh. Prompt của mỗi tác tử phải nêu rõ những việc nó làm *và cả những việc nó tuyệt đối không được làm*.
- Bộ kiểm thử chấp nhận (acceptance tests) cho từng tác vụ: Trước khi tác tử bắt đầu, hãy định nghĩa rõ "thành công trông như thế nào".
- Kiểm tra đặc tả trước khi xuất phát (pre-flight spec check): Sử dụng một tác tử riêng biệt để rà soát định nghĩa tác vụ trước khi điều phối tới các công nhân.

**Lỗi Phối hợp (Coordination Failures - 36,94%):** Sự đứt gãy trong giao tiếp hoặc trạng thái.

Ví dụ:
- Hai tác tử cùng cập nhật trạng thái dùng chung mà không có cơ chế đồng bộ hóa.
- Tin nhắn bị thất lạc giữa các tác tử (lỗi hàng đợi, quá thời gian chờ - timeout).
- Lệch pha trạng thái (state drift): Tác tử A cho rằng tác vụ đã hoàn tất; trong khi tác tử B vẫn đang tiếp tục thực thi.

Biện pháp giảm thiểu:
- Quản lý phiên bản trạng thái dùng chung với cơ chế khóa lạc quan (optimistic concurrency).
- Cơ chế xác nhận tường minh (ACK) cho các tin nhắn trọng yếu (thử lại cho đến khi nhận được xác nhận).
- Điểm kiểm tra đồng bộ trạng thái định kỳ; phát hiện sớm sự lệch pha dữ liệu.

**Lỗ hổng Xác minh (Verification Gaps - 21,30%):** Thiếu vắng các bước kiểm tra độc lập đối với đầu ra.

Ví dụ:
- Một tác tử tuyên bố đã hoàn thành; không có bất kỳ ai kiểm chứng tính chính xác.
- Một chuỗi các tác tử chuyển tiếp dữ liệu, trong đó mỗi tác tử đều tin tưởng mù quáng vào kết quả của tác tử trước đó.
- Thiếu hụt độ bao phủ kiểm thử đối với hành vi cấu thành đột sinh (emergent composed behavior).

Biện pháp giảm thiểu:
- Tác tử xác minh độc lập (Independent verifier agent): Chỉ có quyền đọc, truy cập vào nguồn dữ liệu độc lập.
- Hợp đồng bàn giao rõ ràng: "Đầu ra của tác tử A phải vượt qua bộ kiểm tra C trước khi tác tử B được phép bắt đầu".
- Ghi nhật ký kết quả toàn diện phục vụ phân tích hậu kỳ (post-hoc analysis).

### Nhóm thất bại Tư duy Tập thể (Groupthink) (arXiv:2508.05687)

Năm dạng thất bại liên quan xuất hiện khi các tác tử bị đồng nhất hóa hoặc bắt chước lẫn nhau:

**Sụp đổ do đơn văn hóa (Monoculture collapse):** Dùng chung mô hình nền hoặc chung dữ liệu huấn luyện dẫn đến các lỗi sai có tương quan chặt chẽ. Khi ba tác tử cùng chia sẻ một LLM gốc, chúng sẽ cùng chia sẻ chung một loại ảo giác.

**Thiên kiến phục tùng (Conformity bias):** Các tác tử có xu hướng điều chỉnh ý kiến ngả về phía tác tử có phát ngôn to nhất hoặc tỏ ra tự tin nhất, ngay cả khi tác tử đó hoàn toàn sai.

**Khiếm khuyết Lý thuyết Tâm trí (Deficient ToM):** Các tác tử thất bại trong việc mô hình hóa niềm tin và ý định của nhau; sự phối hợp nhanh chóng sụp đổ (xem lại Bài học 18).

**Động lực động cơ hỗn hợp (Mixed-motive dynamics):** Các tác tử có mục tiêu chỉ tương đồng một phần có xu hướng thỏa hiệp ở vùng trung gian vô nghĩa, khiến kết quả không làm thỏa mãn bất kỳ bên nào.

**Lỗi độ tin cậy xếp tầng (Cascading reliability failures):** Mẫu hình lỗi của một thành phần kích hoạt một chuỗi các mẫu hình lỗi khác ở các thành phần phụ thuộc.

### Ví dụ về lỗi xếp tầng — Cơn bão thử lại (Retry Storm)

Một kịch bản sự cố kinh điển của năm 2026:

```text
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

Giải pháp nằm ở kỹ thuật kinh điển: **bộ ngắt mạch (circuit breakers)**. Khi tỷ lệ lỗi của dịch vụ hạ tầng vượt quá ngưỡng an toàn, bộ ngắt mạch lập tức mở ra, phản hồi ngay bằng kết quả từ bộ nhớ đệm (cache) hoặc dữ liệu mặc định mà không chuyển tiếp yêu cầu đi tiếp. Đồng thời, cần áp đặt giới hạn ngân sách thử lại tối đa trên mỗi yêu cầu người dùng.

Bộ ngắt mạch là một trong số ít các cơ chế xử lý lỗi đa tác tử mà bạn có thể mượn trực tiếp từ các hệ thống phân tán truyền thống mà không cần chỉnh sửa.

### Đầu độc bộ nhớ (Memory Poisoning)

Nhắc lại từ Bài học 13: Ảo giác của một tác tử vô tình trở thành dữ liệu sự thật trong bộ nhớ dùng chung; các tác tử phía sau tiếp tục lập luận dựa trên dữ kiện độc hại đó. Theo thuật ngữ của MAST, đây chính là một lỗ hổng xác minh nghiêm trọng tại tầng bộ nhớ dùng chung.

Triệu chứng điển hình là độ chính xác suy giảm dần đều theo thời gian. Hệ thống không hề gặp sự cố sập đột ngột; nó chỉ dần trôi dạt khỏi kết quả đúng, khiến việc tìm ra nguyên nhân gốc rễ trở thành một bài toán vô cùng phức tạp.

Biện pháp giảm thiểu: Nhật ký chỉ ghi thêm (append-only log), truy xuất nguồn gốc (provenance), và tác tử thẩm định không cho phép ghi đè.

### STRATUS — Các tác tử chuyên biệt phục vụ phản ứng sự cố

Nghiên cứu STRATUS (NeurIPS 2025) ghi nhận mức cải thiện thành công 1,5 lần khi bạn triển khai một bộ ba tác tử chuyên biệt:

- **Tác tử Phát hiện (Detection agent):** Giám sát các mẫu hình triệu chứng bất thường (mức độ bất đồng ý kiến tăng vọt, số lượng thử lại tăng đột biến, độ chính xác trôi dạt).
- **Tác tử Chẩn đoán (Diagnosis agent):** Dựa trên các triệu chứng ghi nhận được, suy luận nguyên nhân gốc rễ dựa theo phân loại MAST.
- **Tác tử Kiểm chứng (Validation agent):** Sau khi biện pháp khắc phục được kích hoạt, kiểm tra lại để đảm bảo các triệu chứng bất thường đã thực sự biến mất.

Đây chính là quy trình phản ứng sự cố theo chuẩn SRE áp dụng cho các hệ thống tác tử. Cả ba vai trò này đều có thể được đảm nhận bởi các tác tử LLM với prompt được chuyên môn hóa cao độ.

### Quy trình Kiểm toán Dạng thức Thất bại

Một thực hành chuẩn mực trong năm 2026 là tổ chức kiểm toán dạng thức thất bại định kỳ mỗi quý (và trước mỗi đợt phát hành phiên bản lớn):

1. **Lấy mẫu dấu vết (Trace sample):** Thu thập khoảng ~1.000 dấu vết thực thi thực tế trong môi trường sản xuất.
2. **Phân loại (Categorize):** Ánh xạ từng sự cố sang danh mục lỗi của MAST + Groupthink.
3. **Tính toán tỷ lệ thất bại theo nhóm:** Xác định nhóm lỗi nào đang chiếm tỷ trọng áp đảo trong hệ thống.
4. **Xếp hạng mức độ ưu tiên khắc phục:** Biện pháp nào có thể triệt tiêu được nhiều lỗi nhất với chi phí hợp lý nhất?
5. **Thực thi 2-3 biện pháp trọng tâm:** Triển khai cải tiến và tiến hành tái kiểm toán vào quý tiếp theo.

Tính kỷ luật trong quy trình quan trọng hơn nhiều so với việc lựa chọn công cụ. Nếu không có các đợt kiểm toán định kỳ, các lỗi sẽ chìm vào màn sương dữ liệu và không bao giờ được giải quyết một cách có hệ thống.

### Khi hệ thống thất bại trong im lặng (Silent Failures)

Dạng thất bại nguy hiểm nhất là sai lệch tính đúng đắn trong im lặng (silent correctness failure). Một hệ thống gặp sự cố ồn ào (sập nguồn, ném ngoại lệ, kích hoạt cảnh báo) rất dễ giám sát. Nhưng một hệ thống liên tục sinh ra các câu trả lời nghe rất hợp lý nhưng thực chất lại sai bét thì không bao giờ bộc lộ qua nhật ký ngoại lệ (exception logs). Đó là lý do tại sao các lỗ hổng xác minh lại gây thiệt hại đắt đỏ nhất trên từng vụ việc, dù chúng chỉ chiếm 21,30% tổng số lượng lỗi.

Hãy đầu tư vào:
- Đánh giá mẫu ngẫu nhiên có sự tham gia của con người.
- Bộ kiểm thử hồi quy dựa trên tập dữ liệu chuẩn vàng (golden datasets).
- Cơ chế đối chiếu chéo (cross-checking) giữa các tác tử trên những đầu ra trọng yếu.

### Lỗi tức thời so với Lỗi chậm

Một số lỗi bộc lộ ngay lập tức; một số lỗi lại tiến triển âm ỉ. Lỗi tức thời (quá thời gian chờ, sai lệch lược đồ dữ liệu, lỗi xác thực) rất rẻ để phát hiện. Nhưng lỗi chậm (đầu độc bộ nhớ, trôi dạt do đơn văn hóa, nhập nhằng vai trò) lại vô cùng tốn kém để nhận diện và ngăn chặn.

Nước đi kỹ thuật của năm 2026: Gắn thiết bị giám sát các chỉ số đại diện cho lỗi chậm (slow-failure proxies) để bạn có thể tóm bắt sự trôi dạt trước khi nó biến thành sai sót hiển hiện. Tỷ lệ đồng thuận, tỷ lệ thử lại, phân phối độ dài văn bản đầu ra, và khoảng cách chỉnh sửa giữa các phiên bản tác tử liên tiếp đều là những chỉ số đại diện cực kỳ giá trị.

````figure
a5-retry-cascade
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa:

- `FailureTaxonomy` — Phân loại các sự cố mô phỏng vào các nhóm lỗi MAST và Groupthink.
- `CircuitBreaker` — Mẫu hình ngắt mạch kinh điển; tự động mở mạch khi tỷ lệ lỗi vượt ngưỡng an toàn.
- `RetryStormSimulator` — Tái hiện lỗi xếp tầng của cơn bão thử lại; bật / tắt bộ ngắt mạch để quan sát sự khác biệt.
- `DetectionAgent` — Tác tử khớp mẫu triệu chứng theo phong cách STRATUS dưới dạng kịch bản.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả kỳ vọng:
- Cơn bão thử lại khi không có bộ ngắt mạch: Lỗi dịch vụ kho hàng bùng nổ theo cấp số nhân (mô phỏng).
- Khi có bộ ngắt mạch: Tải bị chặn đứng tại ngưỡng an toàn; hệ thống phục vụ các phản hồi ở chế độ suy giảm tính năng (degraded-mode).
- Tác tử phát hiện đánh dấu mẫu hình bất thường và nêu đích danh danh mục lỗi MAST tương ứng.

## Ứng dụng thực tế (Use It)

Tài liệu `outputs/skill-mast-auditor.md` cung cấp kỹ năng thực hiện một phiên kiểm toán dạng thức thất bại theo chuẩn MAST cho hệ đa tác tử: Từ thu thập dấu vết $\rightarrow$ phân loại $\rightarrow$ xếp hạng giải pháp giảm thiểu.

## Triển khai thực tế (Ship It)

Kỷ luật kiểm soát thất bại trong môi trường sản xuất:

- **Kiểm toán MAST mỗi quý một lần:** Không nên để đến hàng năm. Các nhóm lỗi sẽ biến dịch khi hệ thống của bạn mở rộng quy mô.
- **Triển khai bộ ngắt mạch ở khắp mọi nơi:** Trên mọi lệnh gọi ra bên ngoài tới bất kỳ dịch vụ phụ thuộc nào. Đặt ngưỡng mở mạch mặc định ở mức tỷ lệ lỗi 5-10%.
- **Tập dữ liệu chuẩn vàng (Golden datasets):** Quy mô nhỏ, chất lượng cao, được con người thẩm định kỹ lưỡng. Chạy kiểm thử hồi quy hàng tuần với các tập dữ liệu này.
- **Bộ ba tác tử STRATUS:** Tác tử Phát hiện + Chẩn đoán + Kiểm chứng giám sát môi trường sản xuất. Hãy bắt đầu từ tác tử phát hiện trước; chỉ bổ sung thêm tác tử chẩn đoán khi tín hiệu triệu chứng quá nhiều nhiễu.
- **Ngân sách lỗi (Failure budget):** Đặt chỉ số SLO tường minh cho tỷ lệ lỗi theo từng danh mục. Vượt quá ngân sách lỗi đồng nghĩa với việc tạm dừng phát hành tính năng mới để tập trung vá lỗi hệ thống.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Xác nhận rằng bộ ngắt mạch đã dập tắt thành công cơn bão thử lại. Hãy thay đổi ngưỡng lỗi thất bại và quan sát sự đánh đổi về tính sẵn sàng của hệ thống.
2. Hiện thực hóa một **chỉ số đại diện cho lỗi chậm**: đo đạc tỷ lệ đồng thuận giữa 3 tác tử chạy song song. Khi tỷ lệ này sụt giảm đột ngột, hãy kích hoạt cảnh báo. Mô phỏng sự trôi dạt do đơn văn hóa bằng cách làm tương quan hóa dần dần đầu ra của các tác tử.
3. Đọc bài báo của Cemri et al. (arXiv:2503.13657). Chọn 1 trong số 7 hệ thống MAS của họ và liệt kê 3 nhóm lỗi phổ biến nhất. Kết quả đó khớp với dự báo của MAST ở mức độ nào?
4. Đọc bài báo về hiện tượng Tư duy Tập thể (arXiv:2508.05687). Chỉ ra mẫu hình nào trong số 5 dạng thức là khó phát hiện nhất trong môi trường sản xuất. Đề xuất một chỉ số đại diện để đo lường mẫu hình đó.
5. Thiết kế bộ ba STRATUS (Phát hiện - Chẩn đoán - Kiểm chứng) cho một hệ thống đa tác tử cụ thể mà bạn biết. Tác tử phát hiện theo dõi những triệu chứng nào? Tác tử chẩn đoán đề xuất giải pháp gì? Tác tử kiểm chứng làm thế nào để xác nhận các giải pháp đó đã phát huy hiệu quả?

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| MAST | "Hệ phân loại chuẩn của năm 2026" | Nghiên cứu của Cemri năm 2025; 3 danh mục gốc + 14 dạng thức lỗi con. |
| Vấn đề Đặc tả (Specification Problem) | "Nhập nhằng vai trò" | Tác vụ hoặc vai trò không được định nghĩa rõ ràng; tác tử bối rối không biết phải làm gì. |
| Lỗi Phối hợp (Coordination Failure) | "Lệch pha trạng thái" | Sự đứt gãy trong giao tiếp hoặc cơ chế đồng bộ hóa giữa các tác tử. |
| Lỗ hổng Xác minh (Verification Gap) | "Không có ai kiểm tra" | Các kết quả đầu ra được chấp nhận mà không qua bước thẩm định độc lập. |
| Họ lỗi Tư duy Tập thể (Groupthink) | "Lỗi do sự đồng nhất" | Đơn văn hóa, thiên kiến phục tùng, khiếm khuyết ToM, động cơ hỗn hợp, lỗi xếp tầng. |
| Sụp đổ do đơn văn hóa (Monoculture collapse) | "Cùng một mô hình, cùng ảo giác giống nhau" | Các lỗi có tương quan hệ thống bắt nguồn từ việc dùng chung mô hình nền hoặc tập dữ liệu huấn luyện. |
| Cơn bão thử lại (Retry storm) | "Khuếch đại lỗi xếp tầng" | Một sự cố ban đầu kích hoạt các lượt thử lại liên tiếp, làm bùng nổ tải trọng lên các dịch vụ hạ nguồn. |
| Bộ ngắt mạch (Circuit breaker) | "Chủ động ngắt sớm khi tỷ lệ lỗi cao" | Mở mạch khi tỷ lệ lỗi vượt ngưỡng; trả về kết quả mặc định để bảo vệ hệ thống. |
| STRATUS | "Bộ ba phản ứng sự cố" | Tác tử Phát hiện + Chẩn đoán + Kiểm chứng. Tăng hiệu quả khắc phục sự cố lên 1,5 lần. |
| Đầu độc bộ nhớ (Memory poisoning) | "Ảo giác lan truyền rộng khắp" | Dữ kiện trong bộ nhớ dùng chung bị nhiễm độc; các tác tử phía sau lập luận dựa trên dữ liệu sai lệch đó. |

## Tài liệu tham khảo (Further Reading)

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — Hệ phân loại MAST, công bố tại NeurIPS 2025.
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — Hiện tượng đơn văn hóa, thiên kiến phục tùng và hệ thống phân loại 5 họ lỗi.
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — Kỷ yếu hội nghị NeurIPS 2025 (bộ ba phát hiện + chẩn đoán + kiểm chứng).
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — Tài liệu kinh điển về mẫu hình thiết kế bộ ngắt mạch.
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Các ghi chú thực tế về dạng thức thất bại trong môi trường sản xuất.
