# Rủi ro lưỡng dụng: Nâng cao năng lực Mạng, Sinh học, Hóa học, Hạt nhân (Dual-Use Risk — Cyber, Bio, Chem, Nuclear Uplift)

> Bức tranh toàn cảnh về rủi ro lưỡng dụng (dual-use risk) năm 2026 theo từng lĩnh vực chuyên biệt. Sinh học/hóa học: Bài 17 đã trình bày bộ kiểm chuẩn WMDP; thử nghiệm thu thập vũ khí sinh học của Anthropic (mức gia tăng năng lực 2,53 lần - 2.53x uplift) và cảnh báo trong Khung chuẩn bị ứng phó phiên bản 2 (Preparedness Framework v2) tháng 4/2025 của OpenAI ("đang cận kề ranh giới hỗ trợ đáng kể cho người nghiệp dư tạo ra các mối đe dọa sinh học đã biết") đánh dấu điểm uốn quyết định. An ninh mạng (Báo cáo tháng 11/2025 của Anthropic): Các tác nhân nhà nước có liên hệ với Trung Quốc đã sử dụng công cụ lập trình tác tử (agentic coding tool) của Claude để tự động hóa tới 90% một chiến dịch tấn công mạng, trong đó con người chỉ cần can thiệp ở 4-6 bước; chương trình thử nghiệm "truy cập tin cậy" (trusted access) của OpenAI cung cấp quyền tiếp cận năng lực cho các tổ chức bảo mật đã qua thẩm định phục vụ mục đích phòng thủ. Sự xói mòn khoảng cách thực thi (execution gap) trong hóa học/sinh học: Luận điểm phòng thủ kinh điển trước đây luôn cho rằng "chỉ riêng việc tiếp cận thông tin là không đủ". Các mô hình tiên phong có thị giác (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) hiện đã có thể quan sát video từ phòng thí nghiệm ướt (wet-lab) và đưa ra chỉ dẫn hiệu chỉnh theo thời gian thực. Tháng 12/2025: OpenAI chứng minh GPT-5 lặp các thí nghiệm wet-lab, đạt mức cải thiện hiệu suất tới 79 lần thông qua tối ưu hóa quy trình do AI dẫn dắt. Mô hình bất đối xứng giữa người nghiệp dư và chuyên gia: AI đem lại mức tăng năng lực tương đối lớn hơn cho người nghiệp dư nhưng mang lại trần năng lực tuyệt đối cao hơn cho các chuyên gia.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 18 (khung an toàn), Phase 18 · 28 (hệ sinh thái)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả diễn biến nhận thức về mức gia tăng năng lực sinh học (bio-uplift) giai đoạn 2024-2025: từ "gia tăng nhẹ" $\to$ "cận kề ranh giới" $\to$ "mức gia tăng 2,53 lần chưa đủ cơ sở để loại trừ ngưỡng ASL-3".
- Trình bày báo cáo an ninh mạng tháng 11/2025 của Anthropic: tự động hóa tới 90% chiến dịch tấn công mạng do tác nhân nhà nước bảo trợ.
- Phân tích sự xói mòn khoảng cách thực thi (execution gap) trong hóa học/sinh học: cơ chế hiệu chỉnh thời gian thực các thí nghiệm wet-lab dựa trên thị giác máy tính.
- Nắm vững tính bất đối xứng giữa mức gia tăng tương đối của người nghiệp dư (novice-relative) và năng lực tuyệt đối của chuyên gia (expert-absolute), cùng ý nghĩa của nó đối với việc xây dựng lập luận an toàn (safety cases).

## Vấn đề đặt ra

Bài 17 cung cấp phương pháp luận đo lường. Bài 30 phản ánh bức tranh thực tế về các số liệu đo lường tính đến năm 2026. Tình hình đã chuyển biến căn bản giữa năm 2024 và cuối năm 2025: ba trong số bốn lĩnh vực nghiên cứu đã vượt qua các ngưỡng ranh giới nguy hiểm mà các khung an toàn ban hành năm 2024 chưa lường trước được.

## Khái niệm cốt lõi

### Diễn tiến câu chuyện gia tăng năng lực sinh/hóa (Bio/chem uplift narrative)

Trải qua ba giai đoạn (được nhắc lại từ Bài 17 nhằm đảm bảo tính mạch lạc):

1. **Năm 2024 - "Gia tăng nhẹ" (mild uplift):** Các bài đánh giá ban đầu theo Khung Preparedness/RSP ghi nhận lợi thế không đáng kể của người nghiệp dư khi dùng AI so với việc tìm kiếm thông tin thông thường trên Internet.
2. **Tháng 4/2025 - "Cận kề ranh giới" (on the cusp):** Bản cập nhật Preparedness Framework v2 của OpenAI đưa ra cảnh báo rằng các mô hình "đang cận kề ranh giới giúp ích đáng kể cho những người nghiệp dư tạo ra các mối đe dọa sinh học đã biết".
3. **Năm 2025 - Thử nghiệm thu thập vũ khí sinh học của Anthropic:** Một nghiên cứu có đối chứng trên nhóm người nghiệp dư; ghi nhận mức gia tăng năng lực 2,53 lần ở các tác vụ trong giai đoạn thu thập ban đầu; con số này chưa đủ an toàn để loại trừ khả năng kích hoạt ngưỡng An toàn Cấp 3 (AI Safety Level 3 - ASL-3).

Sự thay đổi này mang tính bản chất: từ "gia tăng nhẹ" chuyển biến thành "có khả năng tạo điều kiện trên thực tế" chỉ trong vòng 18 tháng, ngay cả khi chưa cần một bước nhảy vọt đột phá về kiến trúc mô hình.

### Sự xói mòn khoảng cách thực thi trong hóa học/sinh học

Hàng rào phòng thủ lịch sử: Tiếp cận thông tin là điều kiện cần nhưng chưa đủ; kỹ năng thực hành thao tác quy trình thực tế là rào cản ngăn chặn những kẻ nghiệp dư. Các mô hình tiên phong tích hợp thị giác năm 2025 đã phá vỡ một phần cơ chế phòng thủ này:

- **Hiệu chỉnh quy trình theo thời gian thực (Real-time protocol correction):** Các mô hình như GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, và Grok 4.1 có khả năng quan sát luồng video từ phòng thí nghiệm ướt (wet-lab) và gắn cờ cảnh báo lỗi sai ngay giữa quy trình thực hiện.
- **Minh chứng của OpenAI vào tháng 12/2025:** GPT-5 khi lặp và tinh chỉnh các thí nghiệm wet-lab đã đạt mức cải thiện hiệu suất lên tới 79 lần thông qua khả năng tối ưu hóa quy trình do AI điều phối.

Hệ quả cốt lõi: Hàng rào phòng thủ dựa vào "kỹ năng thực hành" đang bị xói mòn nghiêm trọng. Các rào cản về mua sắm nguyên liệu và thiết bị chuyên dụng vẫn còn tồn tại, nhưng khoảng cách về tri thức ẩn (tacit knowledge) đang ngày càng bị thu hẹp.

### Gia tăng năng lực tấn công mạng (Báo cáo tháng 11/2025)

Báo cáo tháng 11/2025 của Anthropic: Các tác nhân nhà nước có liên hệ với Trung Quốc đã khai thác công cụ lập trình tác tử (agentic coding tool) của Claude để tự động hóa từ 80% đến 90% một chiến dịch tấn công mạng quy mô lớn. Sự can thiệp thủ công của con người chỉ cần thiết trong 4 đến 6 bước mang tính then chốt.

Hệ quả phân tích:
- Lập trình tác tử (agentic coding) đã trở thành thành phần cơ sở cho tự động hóa tấn công. Trước đây, sự hỗ trợ của AI trong an ninh mạng chỉ giới hạn ở cấp độ từng đoạn mã (code snippet); giờ đây, quy trình làm việc tác tử đã tích hợp liền mạch từ khâu do thám (reconnaissance), khai thác lỗ hổng (exploitation), hậu khai thác (post-exploitation), cho đến trích xuất dữ liệu (exfiltration).
- 4-6 bước can thiệp của con người chính là nút thắt cổ chai hiện tại; các bước tiến năng lực trong tương lai sẽ tiếp tục kéo giảm con số này.
- Lưỡng dụng trong phòng thủ: Chương trình thử nghiệm "truy cập tin cậy" (trusted access) của OpenAI cung cấp quyền tiếp cận năng lực cao cấp cho các tổ chức bảo mật đã qua thẩm định khắt khe (các công ty ứng phó sự cố uy tín, cơ quan chính phủ) nhằm phục vụ công tác phòng thủ. Sự bất đối xứng trong việc cấp quyền truy cập sẽ tạo ưu thế cho bên phòng thủ nếu chương trình này được mở rộng quy mô đúng đắn.

Khía cạnh phòng thủ của mảng công việc này bắt đầu từ kỹ năng đọc và phân tích mã nhị phân đã biên dịch trong phạm vi được cấp phép. Tuyến học kỹ thuật dịch ngược (reverse engineering) trong Giai đoạn 13 giảng dạy kỹ năng này trên các chương trình do chính bạn xây dựng thông qua phân tích tĩnh thuần túy: bộ đọc ELF trong [Bài 13.32](../../../13-tools-and-protocols/32-program-anatomy-elf-from-scratch/docs/en.md), phân tích tháo mã (disassembly) trong [Bài 13.33](../../../13-tools-and-protocols/33-disassembly-and-control-flow/docs/en.md), chuyển đổi sang mã giả (lifting to pseudocode) trong [Bài 13.34](../../../13-tools-and-protocols/34-lifting-to-pseudocode/docs/en.md), và sổ cái bằng chứng giúp tách bạch sự thật quan sát được với suy đoán tại [Bài 13.35](../../../13-tools-and-protocols/35-evidence-confidence-and-unknowns/docs/en.md). Ứng dụng mô phỏng [Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) minh họa toàn bộ đường ống xử lý này trên một hàm đơn lẻ. Một tác tử thực hiện nhiệm vụ này bắt buộc phải sở hữu cùng một sổ cái bằng chứng nghiêm ngặt như vậy, bởi một kết quả dịch ngược trông rất tự tin vẫn có thể hoàn toàn dựa trên một giả định phỏng đoán.

### Mối đe dọa hạt nhân

Đây là lĩnh vực ít được phân tích công khai nhất trong số bốn lĩnh vực CBRN (Hóa học, Sinh học, Phóng xạ, Hạt nhân). Mô hình mối đe dọa ở đây hoàn toàn khác biệt: việc thu thập và làm giàu vật liệu phân hạch (fissile-material acquisition) chiếm phần lớn độ khó, chứ không nằm ở khâu thông tin. Sự gia tăng năng lực của AI trên tầng thông tin mang lại mức nâng cao năng lực rất hạn chế cho người nghiệp dư trong thực tế. Chưa có báo cáo nào trong giai đoạn 2024-2025 từ các phòng thí nghiệm lớn ghi nhận việc mô hình vượt qua ngưỡng an toàn hạt nhân.

### Bất đối xứng: Người nghiệp dư tương đối vs Chuyên gia tuyệt đối

Một quy luật xuyên suốt cả bốn lĩnh vực:

- **Gia tăng năng lực tương đối của người nghiệp dư (Novice-relative uplift):** Rất cao. Mang tính nhân bội. Theo nghiên cứu sinh học năm 2025 của Anthropic, mức gia tăng đạt 2,53 lần.
- **Năng lực tuyệt đối của chuyên gia (Expert-absolute capability):** Trần năng lực rất cao. Một chuyên gia khai thác được giá trị từ mô hình lớn hơn nhiều so với người nghiệp dư, bởi chuyên gia nắm rõ cần hỏi điều gì và biết cách thẩm định sâu sắc câu trả lời.

Hệ quả đối với việc xây dựng lập luận an toàn: Việc chỉ tập trung xử lý sự gia tăng năng lực của người nghiệp dư (thông qua bộ lọc đầu vào, câu từ chối, làm mờ thông tin) là hoàn toàn không đủ để kiểm soát mức năng lực tuyệt đối của chuyên gia. Đòi hỏi phải có các biện pháp phòng vệ bổ sung: làm kiên cố mô hình trước kỹ thuật gợi mở (elicitation-hardening), quên năng lực nguy hại có chọn lọc (machine unlearning - Bài 17), và các giao thức kiểm soát an toàn nghiêm ngặt (Bài 10).

### Tổng hợp so sánh xuyên suốt các lĩnh vực

| Lĩnh vực | Năm 2024 | Năm 2025 | Điểm uốn quyết định |
|---|---|---|---|
| Sinh học (Bio) | Gia tăng nhẹ | Gia tăng 2,53x, tiệm cận ngưỡng ASL-3 | Tự động hóa giai đoạn thu thập ban đầu |
| Hóa học (Chem) | Gia tăng nhẹ | Xói mòn khoảng cách thực thi nhờ thị giác máy tính | Hiệu chỉnh quy trình wet-lab theo thời gian thực |
| Mạng (Cyber) | Hỗ trợ viết mã đơn lẻ | Tự động hóa 80-90% chiến dịch tấn công | Lập trình tác tử tự chủ |
| Hạt nhân (Nuclear) | Hạn chế | Hạn chế | Nút thắt tiếp cận vật liệu phân hạch vẫn đứng vững |

Ba lĩnh vực đã chính thức vượt qua các ngưỡng cảnh báo an toàn. Một lĩnh vực duy nhất vẫn bị chặn lại bởi các rào cản vật lý phi thông tin.

### Vị trí của bài học trong Giai đoạn 18

Bài 30 là đồ án tổng kết (capstone): phác họa bức tranh toàn cảnh về rủi ro lưỡng dụng mà mọi bài học trước đó đều góp phần đo lường, giới hạn, hoặc quản trị. Các Bài 17-18 cung cấp phương pháp đo lường và khung chính sách; các Bài 12-16 cung cấp bộ công cụ đánh giá; các Bài 24-25 cung cấp tầng pháp lý và công bố thông tin; Bài 28 cung cấp bức tranh hệ sinh thái nghiên cứu. Bài 30 là nơi tất cả các bằng chứng thực nghiệm hội tụ.

```figure
an-uplift-asymmetry
```

## Thực hành (Use It)

Không có mã nguồn. Hãy nghiên cứu báo cáo an ninh mạng tháng 11/2025 của Anthropic, bản cập nhật Khung chuẩn bị ứng phó phiên bản 2 (Preparedness Framework v2) tháng 4/2025 của OpenAI, và bản tổng kết năm 2025 về AI x Bio của Hội đồng về Rủi ro Chiến lược (Council on Strategic Risks).

## Đóng gói sản phẩm (Ship It)

Bài học này cung cấp sản phẩm `outputs/skill-dual-use-triage.md`. Khi tiếp nhận một tuyên bố về năng lực mô hình hoặc một báo cáo sự cố an toàn năm 2026, kỹ năng này sẽ phân loại nhanh trên cả bốn lĩnh vực rủi ro và xác định xem báo cáo đó tác động đến mức gia tăng tương đối của người nghiệp dư, năng lực tuyệt đối của chuyên gia, hay cả hai.

## Bài tập

1. Đọc báo cáo an ninh mạng tháng 11/2025 của Anthropic. Liệt kê chi tiết 4-6 bước can thiệp của con người và lập luận xem bước nào sẽ là mắt xích đầu tiên bị tự động hóa hoàn toàn trong các thế hệ mô hình tiếp theo.

2. Khoảng cách thực thi trong sinh học/hóa học đang bị xói mòn bởi công nghệ thị giác. Hãy thiết kế một bài đánh giá đo lường mức độ gia tăng tri thức ẩn (tacit knowledge) mà không vi phạm các ranh giới kiểm soát xuất khẩu quân sự và công nghệ của ITAR/EAR.

3. Nguy cơ hạt nhân hiện tại bị chặn lại bởi rào cản tiếp cận vật liệu phân hạch. Hãy đưa ra các luận điểm ủng hộ và phản đối quan điểm cho rằng một bước đột phá AI trong tương lai có thể làm dịch chuyển nút thắt cổ chai này.

4. Xây dựng một lập luận an toàn (theo mô hình ba trụ cột tại Bài 18) cho một mô hình tiên phong có năng lực cao về an ninh mạng, sao cho kiểm soát chặt chẽ cả mức gia tăng của người nghiệp dư lẫn năng lực tuyệt đối của chuyên gia.

5. Chọn một trong bốn lĩnh vực rủi ro và viết một đoạn văn dự báo xu hướng cho năm 2027 dựa trên quỹ đạo biến chuyển 2024-2025. Nêu rõ bằng chứng thực nghiệm nào sẽ chứng minh dự báo của bạn là sai (falsify).

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Uplift | "AI trợ giúp kẻ tấn công" | Mức gia tăng năng lực của kẻ tấn công có được nhờ sự hỗ trợ của hệ thống AI |
| Novice-relative uplift | "mức gia tăng nhân bội" | Mức độ AI hỗ trợ một người nghiệp dư so với hiện trạng năng lực vốn có của họ |
| Expert-absolute capability | "trần năng lực tối đa" | Mức năng lực tối đa mà một chuyên gia đầu ngành có thể khai thác từ mô hình |
| Execution gap | "lý thuyết so với thực hành" | Hàng rào phòng thủ lịch sử: kỹ năng thực hành thao tác wet-lab ngăn chặn người nghiệp dư |
| Agentic coding | "tấn công tự chủ" | Khả năng thực thi tự chủ các tác vụ an ninh mạng phức tạp trải qua nhiều bước |
| Acquisition phase | "các bước trước tổng hợp" | Giai đoạn mua sắm, chuẩn bị trang thiết bị, giấy phép trước khi tạo ra mối đe dọa sinh học |
| Trusted access | "chương trình ưu tiên bên phòng thủ" | Chương trình thử nghiệm năm 2025 của OpenAI cấp quyền tiếp cận năng lực cho bên phòng thủ đã qua thẩm định |

## Đọc thêm

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — Báo cáo điều tra tự động hóa chiến dịch gián điệp mạng
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — Cảnh báo năng lực sinh học "cận kề ranh giới"
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — Định nghĩa các ngưỡng kích hoạt ASL-3 trong lĩnh vực sinh học
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — Báo cáo tổng hợp thường niên về rủi ro giao thoa giữa AI và an toàn sinh học
