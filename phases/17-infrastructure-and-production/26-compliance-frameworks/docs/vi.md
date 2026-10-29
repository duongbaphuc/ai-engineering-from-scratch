# Tuân thủ Quy định — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Khả năng đáp ứng đa khung tiêu chuẩn tuân thủ (multi-framework coverage) là điều kiện tiên quyết (table stakes) cho các hợp đồng doanh nghiệp năm 2026. **Đạo luật AI của EU (EU AI Act)**: có hiệu lực từ ngày 1 tháng 8 năm 2024. Phần lớn các yêu cầu đối với hệ thống rủi ro cao bắt đầu được thực thi từ ngày 2 tháng 8 năm 2026. Mức phạt lên tới 15 triệu euro hoặc 3% doanh thu toàn cầu hàng năm đối với vi phạm nghĩa vụ hệ thống rủi ro cao (Điều 99(4)); lên tới 35 triệu euro hoặc 7% đối với các hoạt động AI bị cấm (Điều 99(3)). Áp dụng trên quy mô toàn cầu nếu cung cấp dịch vụ cho người dùng tại EU. **Đạo luật AI Colorado (Colorado AI Act)**: có hiệu lực từ ngày 30 tháng 6 năm 2026 (được hoãn lại từ tháng 2 năm 2026 theo SB25B-004) — yêu cầu đánh giá tác động đối với các hệ thống rủi ro cao, quyền khiếu nại đối với các quyết định do AI đưa ra. Bang Virginia áp dụng quy định tương tự đối với các lĩnh vực tín dụng/tuyển dụng/nhà ở/giáo dục. **SOC 2 Type II**: tiêu chuẩn thực tế bắt buộc đối với AI B2B (Type II, không phải Type I, dành cho khối fintech). **GDPR**: mức phạt liên quan đến AI lớn nhất từng được ghi nhận là 30,5 triệu euro đối với Clearview AI (Cơ quan Bảo vệ Dữ liệu Hà Lan, tháng 9 năm 2024); Cơ quan Bảo vệ Dữ liệu Ý (Garante) đã phạt OpenAI 15 triệu euro vào tháng 12 năm 2024 (sau đó bị hủy bỏ khi kháng cáo vào tháng 3 năm 2026). Che giấu PII theo thời gian thực (real-time PII redaction) tại tầng suy luận là tiêu chuẩn có thể biện hộ trước pháp lý; việc dọn dẹp hậu xử lý là không đủ. **HIPAA**: bắt buộc đối với y tế — không được gửi thông tin y tế được bảo vệ (PHI) tới các dịch vụ AI bên ngoài nếu không có Thỏa thuận Đối tác Kinh doanh (BAA). **PCI-DSS**: phạm vi bao phủ tầng tương tác AI đòi hỏi cấu hình hệ thống kết hợp thỏa thuận hợp đồng, không tự động kích hoạt. **ISO 42001**: tiêu chuẩn quản trị AI mới nổi, đang trở thành yêu cầu thu mua ngày càng phổ biến bên cạnh ISO 27001. Hồ sơ tham chiếu: OpenAI duy trì SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, PCI-DSS cho các thành phần thanh toán của ChatGPT. Kỹ thuật ánh xạ chéo giữa các khung (cross-framework mapping) giúp giảm thiểu tình trạng quá tải kiểm toán (audit fatigue): kiểm soát truy cập ánh xạ chéo qua ISO 27001 A.5.15-5.18, GDPR Điều 32, HIPAA §164.312(a).

**Type:** Learn
**Languages:** (Python tùy chọn — tuân thủ là chính sách + quy trình, không phải mã lệnh)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 phút

## Mục tiêu học tập

- Liệt kê bảy khung tiêu chuẩn năm 2026 liên quan trực tiếp đến các sản phẩm LLM và khớp nối từng khung với phân khúc khách hàng mục tiêu tương ứng.
- Trích dẫn mốc thời gian thực thi của Đạo luật AI của EU (có hiệu lực từ tháng 8/2024; thực thi rủi ro cao từ tháng 8/2026) và trần phạt hai cấp (15 triệu euro / 3% cho nghĩa vụ rủi ro cao; 35 triệu euro / 7% cho các hành vi bị cấm).
- Giải thích vì sao việc dọn dẹp PII ở khâu hậu xử lý không đáp ứng chuẩn GDPR và nêu bật kỹ thuật lược bỏ thời gian thực tại tầng suy luận như một tiêu chuẩn có thể biện hộ pháp lý vững chắc.
- Mô tả phương pháp ánh xạ kiểm soát chéo giữa các khung (ví dụ: kiểm soát truy cập ánh xạ tới ISO 27001 A.5.15-5.18 + GDPR Điều 32 + HIPAA §164.312(a)).

## Vấn đề thực tế

Bộ phận thu mua của một khách hàng doanh nghiệp yêu cầu chứng nhận SOC 2 Type II, tuân thủ GDPR, thỏa thuận HIPAA BAA, chứng chỉ ISO 27001, và "tuyên bố tuân thủ Đạo luật AI của EU". Nhóm của bạn hiện mới chỉ có SOC 2 Type I. Bạn còn cách báo cáo Type II ít nhất sáu tháng và chưa từng bắt đầu lập hồ sơ theo Điều 30 của GDPR.

Đáp ứng đa khung tiêu chuẩn không đơn thuần là bài toán riêng của LLM — đó là thách thức chung của SaaS doanh nghiệp, được tích hợp thêm các yêu cầu đặc thù của LLM. Các đội ngũ thẩm định thu mua vào năm 2026 kỳ vọng một ma trận rõ ràng với mỗi hàng là một khung tiêu chuẩn và mỗi cột là một biện pháp kiểm soát, chứ không phải một tệp tài liệu PDF chung chung.

## Khái niệm then chốt

### Bảy khung tiêu chuẩn pháp lý

| Khung tiêu chuẩn | Phạm vi áp dụng | Yêu cầu đặc thù đối với LLM |
|---|---|---|
| SOC 2 Type II | Nền tảng SaaS B2B | Các biện pháp kiểm soát quy trình được kiểm toán độc lập trong 6-12 tháng |
| HIPAA | Y tế Hoa Kỳ | Bắt buộc có BAA; PHI không được rời khỏi hạ tầng nếu thiếu thỏa thuận ký kết |
| GDPR | Người dùng tại EU | Lược bỏ PII theo thời gian thực; quyền của chủ thể dữ liệu; hồ sơ Điều 30 |
| PCI-DSS | Dữ liệu thanh toán thẻ | Cấu hình kỹ thuật + hợp đồng cho bất kỳ thành phần AI nào chạm tới luồng thanh toán |
| EU AI Act | Phục vụ người dùng EU | Phân loại tầng rủi ro; hệ thống rủi ro cao: đánh giá sự phù hợp, lưu trữ tài liệu, ghi log |
| Colorado AI Act | Phục vụ cư dân bang Colorado | Đánh giá tác động thuật toán; quyền khiếu nại quyết định tự động |
| ISO 42001 | Quản trị AI | Tiêu chuẩn mới nổi; thường đi đôi với ISO 27001 |

### Lộ trình thực thi Đạo luật AI của EU

- Ngày 1 tháng 8 năm 2024: Chính thức có hiệu lực.
- Ngày 2 tháng 2 năm 2025: Bắt đầu xử phạt các thực hành AI bị cấm hoàn toàn.
- Ngày 2 tháng 8 năm 2026: Thực thi toàn diện đối với các hệ thống AI rủi ro cao (đánh giá sự phù hợp, tài liệu kỹ thuật, ghi nhật ký bắt buộc).
- Tháng 8 năm 2027: Áp dụng cho các hệ thống rủi ro cao cấu thành trong sản phẩm thuộc luật hài hòa hóa của EU.

Các tầng rủi ro: Không thể chấp nhận (bị cấm), Rủi ro cao (bắt buộc đánh giá sự phù hợp + ghi log), Rủi ro hạn chế (minh bạch thông tin), Rủi ro tối thiểu (không bị ràng buộc). Hầu hết các ứng dụng B2B LLM SaaS thuộc nhóm rủi ro hạn chế; nhóm rủi ro cao áp dụng khi AI can thiệp vào tuyển dụng, chấm điểm tín dụng, giáo dục, thực thi pháp luật, quản lý di cư, hoặc các dịch vụ thiết yếu.

Mức xử phạt (Điều 99): Lên đến 15 triệu euro hoặc 3% tổng doanh thu toàn cầu hàng năm đối với vi phạm nghĩa vụ của hệ thống rủi ro cao (Điều 99(4)); lên đến 35 triệu euro hoặc 7% đối với các hành vi AI bị cấm (Điều 99(3)); áp dụng mức phạt nào cao hơn.

### GDPR — Lược bỏ dữ liệu theo thời gian thực là tiêu chuẩn tối thiểu

Dọn dẹp ở khâu hậu xử lý (xóa bỏ PII sau khi LLM đã xử lý prompt) là tư thế không thể biện hộ trước cơ quan quản lý — mô hình đã tiếp nhận và xử lý dữ liệu nhạy cảm đó. Kỹ thuật lược bỏ PII theo thời gian thực ngay tại tầng suy luận (real-time inference-layer redaction) là chuẩn mực năm 2026:

- Nhận diện thực thể trước khi kích hoạt lệnh gọi LLM.
- Token hóa nhất quán (phương pháp Mesh) để bảo toàn ngữ nghĩa văn bản.
- Chỉ lưu trữ các prompt đã được làm sạch + các prompt thô đã có sự đồng thuận rõ ràng (opt-in consent) từ người dùng.

Thực thi gần đây: Khoản phạt 30,5 triệu euro đối với Clearview AI (Cơ quan DPA Hà Lan, tháng 9/2024) là mức phạt GDPR đặc thù cho AI lớn nhất từng được ghi nhận; khoản phạt 15 triệu euro đối với OpenAI (Garante Ý, tháng 12/2024) là án phạt lớn nhất nhắm vào LLM, dù phán quyết này đã bị kháng cáo hủy bỏ vào tháng 3/2026 và vụ việc vẫn đang tiếp tục được xem xét. Các lập luận dựa trên việc "dọn dẹp dữ liệu sau xử lý" đều đã thất bại trong các kỳ kiểm toán thực tế.

### HIPAA — BAA là điều kiện bắt buộc

Bạn tuyệt đối không được gửi dữ liệu PHI tới các dịch vụ AI bên ngoài nếu không có Thỏa thuận Đối tác Kinh doanh (BAA) có hiệu lực pháp lý. Cả ba nền tảng đám mây quy mô lớn (Bedrock, Azure OpenAI, Vertex) đều cung cấp BAA. API trực tiếp của OpenAI hỗ trợ BAA. API trực tiếp của Anthropic hỗ trợ BAA. Phải xác nhận hợp đồng BAA trước khi truyền bất kỳ dữ liệu PHI nào.

### SOC 2 Type II

Type I: Thiết kế và tài liệu hóa các biện pháp kiểm soát.
Type II: Kiểm toán tính vận hành hiệu quả liên tục của các biện pháp kiểm soát trong vòng 6 đến 12 tháng.

Quy trình thẩm định thu mua B2B năm 2026 mặc định yêu cầu Type II. Type I chỉ là bước khởi đầu; Type II mới là điều kiện thông qua.

Các trọng tâm kiểm toán phổ biến: Nhật ký truy cập (ai đã xem dữ liệu gì), quản lý thay đổi (quy trình triển khai như thế nào), đánh giá rủi ro (định kỳ hàng quý), ứng phó sự cố (đã được diễn tập chưa?). Nhật ký kiểm toán xây dựng từ Phase 17 · 25 có thể tái sử dụng trực tiếp cho mục tiêu này.

### Ánh xạ kiểm soát chéo (Cross-framework mapping)

Một chính sách kiểm soát duy nhất có thể thỏa mãn cùng lúc yêu cầu của nhiều khung tiêu chuẩn:

| Biện pháp kiểm soát | Khung tiêu chuẩn tương ứng |
|---|---|
| Ghi nhật ký truy cập (Access logging) | ISO 27001 A.5.15-5.18, GDPR Điều 32, HIPAA §164.312(a) |
| Quản lý thay đổi (Change management) | ISO 27001 A.8.32, PCI DSS Yêu cầu 6, Phạm vi thông báo vi phạm HIPAA |
| Mã hóa đường truyền (Encryption in transit) | ISO 27001 A.8.24, GDPR Điều 32, HIPAA §164.312(e) |
| Quản lý bí mật (Secrets management) | ISO 27001 A.8.19, PCI DSS Yêu cầu 8, SOC 2 CC6.1 |

Các công cụ tuân thủ tự động (Drata, Vanta, Secureframe) hỗ trợ tự động hóa việc ánh xạ này, mang lại giá trị kinh tế lớn khi mở rộng quy mô.

### ISO 42001 — Tiêu chuẩn mới nổi

Ban hành cuối năm 2023. Ngày càng nhiều doanh nghiệp đưa ISO 42001 vào danh sách yêu cầu thu mua song hành cùng ISO 27001. Đây là khung tiêu chuẩn chuyên biệt cho quản trị hệ thống AI, bao gồm quản lý rủi ro, chất lượng dữ liệu, tính minh bạch và sự giám sát của con người (human oversight).

### Hồ sơ đối chiếu tiêu chuẩn của OpenAI

OpenAI duy trì đồng thời: SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, và PCI-DSS cho các phân hệ thanh toán của ChatGPT. Đây chính là tiêu chuẩn tham chiếu tối thiểu của doanh nghiệp trong năm 2026.

### Các con số cần ghi nhớ

- Mức phạt theo Đạo luật AI của EU: Lên tới 15 triệu euro / 3% (nghĩa vụ hệ thống rủi ro cao, Điều 99(4)); lên tới 35 triệu euro / 7% (thực hành bị cấm, Điều 99(3)).
- Mốc thực thi rủi ro cao của Đạo luật AI của EU: Ngày 2 tháng 8 năm 2026.
- Khoản phạt GDPR lớn nhất về AI được ghi nhận: 30,5 triệu euro, Clearview AI (DPA Hà Lan, tháng 9/2024).
- Khoản phạt GDPR lớn nhất nhắm vào LLM: 15 triệu euro, OpenAI (Garante Ý, tháng 12/2024; bị hủy bỏ khi kháng cáo vào tháng 3/2026).
- Khung thời gian đánh giá SOC 2 Type II: Vận hành kiểm soát hiệu quả trong 6-12 tháng.
- Ngày có hiệu lực của Đạo luật AI Colorado: Ngày 30 tháng 6 năm 2026 (hoãn từ tháng 2/2026 theo SB25B-004).

```figure
i4-control-matrix
```

## Thực hành (Use It)

Tệp `code/main.py` là một công cụ ma trận ánh xạ tuân thủ được viết bằng Python — tiếp nhận một biện pháp kiểm soát và liệt kê toàn bộ các khung tiêu chuẩn mà biện pháp đó thỏa mãn.

## Triển khai thực tế (Ship It)

Bài học này xây dựng `outputs/skill-compliance-matrix.md`. Dựa trên phân khúc khách hàng và khu vực địa lý hoạt động, tài liệu này xác định chính xác các khung tiêu chuẩn và biện pháp kiểm soát bắt buộc phải áp dụng.

## Bài tập

1. Khách hàng doanh nghiệp đầu tiên của bạn yêu cầu SOC 2 Type II, HIPAA BAA và cam kết tuân thủ Đạo luật AI của EU. Đâu là tư thế tuân thủ khả thi tối thiểu (minimum viable compliance posture) để ký kết được hợp đồng?
2. Phân loại ba sản phẩm LLM giả định vào các tầng rủi ro của Đạo luật AI của EU. Những yêu cầu kỹ thuật nào sẽ thay đổi khi rơi vào nhóm rủi ro cao?
3. Bạn vô tình gửi dữ liệu PHI tới một nhà cung cấp AI mà chưa ký kết BAA. Hãy trình bày chi tiết quy trình ứng phó sự cố (incident response).
4. Phân tích xem chứng chỉ ISO 42001 có thực sự "cần thiết trong năm 2026" đối với một nhà cung cấp giải pháp AI quy mô vừa (mid-market) hay không.
5. Ánh xạ các trường trong nhật ký kiểm toán LLM của bạn (từ Phase 17 · 25) tới ít nhất ba biện pháp kiểm soát thuộc các khung tiêu chuẩn khác nhau.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| SOC 2 Type II | "kiểm soát đã được kiểm toán" | Các biện pháp kiểm soát vận hành liên tục 6-12 tháng, được chứng thực độc lập |
| HIPAA BAA | "hợp đồng y tế" | Thỏa thuận Đối tác Kinh doanh; điều kiện bắt buộc để xử lý PHI |
| GDPR | "quyền riêng tư Châu Âu" | Lược bỏ PII thời gian thực là tiêu chuẩn có thể biện hộ pháp lý trong năm 2026 |
| EU AI Act | "luật AI của EU" | Thực thi rủi ro cao từ tháng 8/2026; phạt 15 triệu € / 3% (rủi ro cao) — 35 triệu € / 7% (hành vi cấm) |
| Colorado AI Act | "luật AI cấp bang tại Mỹ" | Hiệu lực từ 30/6/2026 (hoãn bởi SB25B-004); yêu cầu đánh giá tác động |
| ISO 42001 | "quản trị AI" | Khung tiêu chuẩn mới nổi về quản trị rủi ro và tính minh bạch của AI |
| ISO 27001 | "hệ thống ISMS bảo mật" | Tiêu chuẩn nền tảng cho Hệ thống Quản lý An toàn Thông tin |
| Conformity assessment | "bộ hồ sơ AI của EU" | Yêu cầu đối với hệ thống rủi ro cao: hồ sơ kỹ thuật, kiểm thử, ghi nhật ký |
| Cross-framework mapping | "một kiểm soát, nhiều khung" | Một chính sách kỹ thuật duy nhất thỏa mãn kiểm soát của nhiều bộ tiêu chuẩn |

## Tài liệu tham khảo

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — Hồ sơ tuân thủ quy chuẩn tham chiếu.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — Nguồn văn bản pháp lý gốc.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — Nguồn văn bản pháp lý gốc.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — Tiêu chuẩn hệ thống quản lý trí tuệ nhân tạo.
