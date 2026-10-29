# Khung Chuẩn bị của OpenAI và Khung An toàn Biên giới của DeepMind (OpenAI Preparedness Framework and DeepMind Frontier Safety Framework)

> Khung Chuẩn bị phiên bản v2 của OpenAI (OpenAI Preparedness Framework v2 - tháng 4 năm 2025) giới thiệu các Danh mục Nghiên cứu (Research Categories) — bao gồm Quyền tự trị tầm xa (Long-range Autonomy), Cố tình giấu năng lực (Sandbagging), Tự sao chép và Thích ứng tự trị (Autonomous Replication and Adaptation), Phá hoại Rào chắn an toàn (Undermining Safeguards) — tách biệt hoàn toàn với các Danh mục Theo dõi (Tracked Categories). Các Danh mục Theo dõi sẽ kích hoạt Báo cáo Năng lực (Capabilities Reports) cùng Báo cáo Biện pháp bảo vệ (Safeguards Reports) do Nhóm Cố vấn An toàn (Safety Advisory Group - SAG) thẩm định. Khung An toàn Biên giới phiên bản v3 của DeepMind (DeepMind's FSF v3 - tháng 9 năm 2025, với các Cấp độ Năng lực Theo dõi được bổ sung vào ngày 17 tháng 4 năm 2026) lồng ghép quyền tự trị vào các lĩnh vực R&D Học máy (ML R&D) và An ninh mạng (Cyber) (trong đó Cấp độ tự trị ML R&D mức 1 = tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh so với con người kết hợp công cụ AI). FSF v3 giải quyết trực diện hành vi căn chỉnh lừa dối (deceptive alignment) thông qua giám sát tự động đối với hành vi lạm dụng suy luận công cụ (instrumental-reasoning misuse). Một lưu ý thực tế: Các Danh mục Nghiên cứu trong PF v2 (bao gồm cả Quyền tự trị tầm xa) không tự động kích hoạt các biện pháp giảm thiểu; ngôn từ chính sách chỉ dừng ở mức "tiềm năng". Bản thân DeepMind cũng tuyên bố rằng giám sát tự động "sẽ không duy trì được hiệu quả trong dài hạn" nếu năng lực suy luận công cụ mạnh lên.

**Type:** Learn  
**Languages:** Python (stdlib, công cụ so sánh đối chiếu bảng quyết định giữa ba khung chính sách)  
**Prerequisites:** Phase 15 · 19 (Chính sách Mở rộng Có trách nhiệm của Anthropic - Anthropic RSP)  
**Time:** ~45 phút

## Vấn đề

Bài 19 đã phân tích chi tiết chính sách mở rộng quy mô của Anthropic. Bài học này hoàn thiện bức tranh tổng thể bằng cách đọc hiểu tài liệu của OpenAI và DeepMind. Ba tài liệu này là những ấn phẩm họ hàng cùng giải quyết một câu hỏi trọng tâm — khi nào một phòng thí nghiệm biên giới nên tạm dừng hoặc thiết lập rào chắn kiểm duyệt đối với một mô hình — và chúng hội tụ ở một nhóm nhỏ các danh mục nhưng lại phân kỳ ở những điểm cụ thể mang tính quyết định.

Điểm hội tụ: cả ba đều dán nhãn quyền tự trị tầm xa là một lớp năng lực đáng để theo dõi. Cả ba đều thừa nhận hành vi lừa dối (giả mạo căn chỉnh, cố tình giấu năng lực) là một lớp rủi ro cụ thể. Cả ba đều thành lập một hội đồng thẩm định nội bộ. Điểm phân kỳ: OpenAI chia các danh mục thành "Theo dõi" (bắt buộc giảm thiểu) và "Nghiên cứu" (không kích hoạt tự động). DeepMind lồng ghép quyền tự trị vào hai lĩnh vực cụ thể thay vì đặt tên nó như một miền độc lập. Cho dù phòng thí nghiệm gọi tên là Danh mục Theo dõi so với Nghiên cứu, hay Nguy cấp so với Vừa phải, hoặc Tầng 1 so với Tầng 2; hệ quả vận hành thực tế của việc một năng lực nằm ở nhóm nào là hoàn toàn khác biệt giữa các bên.

Đọc đồng thời cả ba tài liệu là một bài tập hữu ích. Cùng một năng lực có thể thuộc diện "bắt buộc giảm thiểu" tại Anthropic, thuộc diện "được giám sát nhưng không kích hoạt hành động" tại OpenAI, và "được theo dõi trong một lĩnh vực cụ thể" tại DeepMind. Lập trường chính sách của từng tổ chức tạo nên sự khác biệt này.

## Khái niệm

### Khung Chuẩn bị phiên bản v2 của OpenAI (Tháng 4 năm 2025)

Cấu trúc:

- **Danh mục Theo dõi (Tracked Categories)**: kích hoạt Báo cáo Năng lực (mô hình có thể làm được gì) cùng Báo cáo Biện pháp bảo vệ (những biện pháp giảm thiểu nào đang được áp dụng). Được thẩm định bởi Nhóm Cố vấn An toàn (SAG) trước khi triển khai.
- **Danh mục Nghiên cứu (Research Categories)**: các năng lực mô hình mà phòng thí nghiệm đang theo dõi nhưng chưa cam kết thực hiện các biện pháp giảm thiểu cụ thể. Bao gồm Quyền tự trị tầm xa (Long-range Autonomy), Cố tình giấu năng lực (Sandbagging), Tự sao chép và Thích ứng tự trị (Autonomous Replication and Adaptation), và Phá hoại Rào chắn an toàn (Undermining Safeguards).

Cách đóng khung then chốt trong bản v2: Danh mục Nghiên cứu không tự động kích hoạt các biện pháp giảm thiểu. Ngôn ngữ chính sách sử dụng từ ngữ "tiềm năng" giảm thiểu. Đây là điểm khác biệt cốt lõi cần quan sát: vị trí của một danh mục trong nhóm Theo dõi hay Nghiên cứu chính là bản lề quyết định giữa việc bị "chặn kiểm duyệt (gated)" hay chỉ dừng ở mức "quan sát (observed)".

### Khung An toàn Biên giới phiên bản v3 của DeepMind (Tháng 9 năm 2025; bổ sung Cấp độ Năng lực Theo dõi vào ngày 17/04/2026)

Cấu trúc:

- **Cấp độ Năng lực Nguy cấp (Critical Capability Levels - CCLs)**: các ngưỡng năng lực thuộc ba nhóm: lạm dụng (CBRN - Hóa sinh phóng xạ hạt nhân, Không gian mạng, Thao túng gây hại), R&D Học máy, và một nhánh thăm dò sự lệch lạc căn chỉnh. Quyền tự trị không phải là một lĩnh vực riêng; nó được lồng ghép vào ML R&D và Không gian mạng.
- **Cấp độ Năng lực Theo dõi (Tracked Capability Levels)**: độ chi tiết bổ sung được đưa vào tháng 4 năm 2026. Ví dụ cụ thể: Cấp độ tự trị ML R&D mức 1 = tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh so với con người kết hợp các công cụ AI.
- **Giám sát căn chỉnh lừa dối (Deceptive alignment monitoring)**: cam kết rõ ràng về việc giám sát tự động đối với các hành vi lạm dụng suy luận công cụ (instrumental reasoning).

Cách đóng khung quyền tự trị ở đây khác với OpenAI. DeepMind không duy trì "Quyền tự trị" như một lĩnh vực cấp cao nhất; nó được lồng ghép vào các lĩnh vực mà quyền tự trị có thể gây ra tác hại thực tế (ML R&D và Không gian mạng). Lập luận đưa ra là quyền tự trị đứng độc lập không gắn với lĩnh vực cụ thể chỉ là năng lực thuần túy chưa gây ra rủi ro; lập luận phản bác cho rằng quyền tự trị xuyên suốt các lĩnh vực là một siêu rủi ro (meta-risk) mà một khung an toàn bắt buộc phải định danh.

### Điểm hội tụ của cả ba khung chính sách

- Đều có Nhóm Cố vấn An toàn nội bộ (Anthropic SAG, OpenAI SAG, ủy ban nội bộ DeepMind). Thực hiện thẩm định trước khi triển khai các mô hình có năng lực cao.
- Đều đề cập rõ ràng đến căn chỉnh lừa dối / giả mạo căn chỉnh (alignment faking) như một lớp rủi ro nghiêm trọng.
- Đều duy trì các tài liệu thường trực theo chu kỳ công bố định kỳ (Anthropic: Lộ trình An toàn Biên giới, Báo cáo Rủi ro; OpenAI: Báo cáo Năng lực và Biện pháp bảo vệ; DeepMind: chu kỳ cập nhật FSF).
- Đều thừa nhận rằng các biện pháp phòng vệ thuần túy dựa vào giám sát có một mức trần hiệu quả nhất định. DeepMind nêu rất rõ ràng: "giám sát tự động sẽ không duy trì được hiệu quả trong dài hạn."

### Điểm phân kỳ giữa ba khung chính sách

- **Anthropic**: cam kết tạm dừng bị xóa bỏ trong bản v3.0; ngưỡng AI R&D-4 được định danh là cổng kiểm soát then chốt tiếp theo.
- **OpenAI**: phân tách thành nhóm Theo dõi và Nghiên cứu; các Danh mục Nghiên cứu (bao gồm Quyền tự trị tầm xa) không tự động kích hoạt cổng kiểm soát.
- **DeepMind**: quyền tự trị được lồng ghép vào các lĩnh vực khác; các Cấp độ Năng lực Theo dõi bổ sung thêm độ chi tiết định lượng vào tháng 4 năm 2026.

### Giấu năng lực (Sandbagging): năng lực cụ thể làm phức tạp cả ba khung

Hành vi cố tình giấu năng lực (khi mô hình chiến lược hạ thấp hiệu suất của chính mình trong các bài đánh giá) nằm trong Danh mục Nghiên cứu của OpenAI. RSP v3.0 của Anthropic giải quyết vấn đề này thông qua khoảng cách ngữ cảnh đánh giá (Bài 1). DeepMind xử lý thông qua việc giám sát căn chỉnh lừa dối trong FSF v3.

Nếu một mô hình giấu năng lực trong các bài kiểm tra, các ngưỡng năng lực của mọi khung an toàn đều sẽ bị đánh giá thấp hơn thực tế. Khung an toàn chỉ phát huy hiệu lực khi các phép đo lường hoạt động chính xác. Đây là lý do tại sao các phép đo lường độc lập từ bên ngoài (Bài 21, METR) và đánh giá đối kháng là bắt buộc bên cạnh việc tự đánh giá của chính phòng thí nghiệm.

### Kỹ năng đọc hiểu văn bản chính sách

- **Định vị (Locate)**: mọi năng lực mà bạn quan tâm phải có thể tìm thấy trong văn bản chính sách. Nếu không tìm thấy, chính sách đó không bao quát nó.
- **Phân loại (Classify)**: năng lực đó thuộc nhóm Theo dõi (kích hoạt giảm thiểu rủi ro) hay Nghiên cứu (được theo dõi nhưng không kích hoạt)? OpenAI gọi tên rõ ràng; Anthropic và DeepMind có các quy ước tương đương.
- **Chu kỳ cập nhật (Cadence)**: chính sách có được cập nhật theo lịch trình công bố định kỳ hay chỉ cập nhật sau các sự kiện cụ thể? Chu kỳ đã tuyên bố thể hiện tính ràng buộc mạnh mẽ hơn.
- **Tính độc lập (Independence)**: việc đánh giá độc lập từ bên ngoài là bắt buộc hay tùy chọn? Anthropic hợp tác với Apollo và Viện An toàn AI Hoa Kỳ (US AISI); OpenAI hợp tác với METR; DeepMind chủ yếu dựa vào hội đồng SAG nội bộ.

```figure
a5-tracked-vs-research
```

## Sử dụng

Tệp `code/main.py` triển khai một công cụ nhỏ để so sánh đối chiếu bảng quyết định. Khi tiếp nhận một năng lực (quyền tự trị, căn chỉnh lừa dối, tự động hóa R&D, khuếch đại không gian mạng, v.v.), công cụ sẽ xuất ra cách mỗi chính sách phân loại năng lực đó và những biện pháp giảm thiểu nào sẽ được kích hoạt. Đây là một công cụ hỗ trợ đọc hiểu văn bản, không phải là một công cụ ban hành chính sách.

## Đóng gói

Tệp `outputs/skill-cross-policy-diff.md` tạo ra bảng so sánh đối chiếu xuyên suốt giữa các chính sách đối với một năng lực cụ thể, sử dụng ba khung an toàn làm chuẩn tham chiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận đầu ra của công cụ so sánh khớp với các văn bản chính sách đối với ít nhất hai năng lực mà bạn có thể kiểm chứng trực tiếp từ các tài liệu nguồn.

2. Đọc toàn văn Khung Chuẩn bị phiên bản v2 của OpenAI. Xác định từng Danh mục Nghiên cứu. Với mỗi danh mục, viết một câu giải thích lý do tại sao nó lại được xếp vào nhóm Nghiên cứu thay vì nhóm Theo dõi.

3. Đọc toàn văn DeepMind FSF v3, cùng bản cập nhật Cấp độ Năng lực Theo dõi tháng 4 năm 2026. Xác định các tiêu chí đánh giá cụ thể của Cấp độ tự trị ML R&D mức 1. Bạn sẽ đo lường tiêu chí đó từ bên ngoài như thế nào?

4. Hành vi giấu năng lực (Sandbagging) nằm trong Danh mục Nghiên cứu của OpenAI. Hãy thiết kế một quy trình đánh giá buộc một mô hình đang cố tình giấu năng lực phải bộc lộ năng lực thực sự của nó. Tham khảo nội dung thảo luận về thao túng ngữ cảnh đánh giá ở Bài 1.

5. So sánh ba chính sách đối với một năng lực cụ thể (tùy bạn chọn). Chỉ ra phân loại của chính sách nào bạn đánh giá là chặt chẽ nhất và chính sách nào lỏng lẻo nhất. Đưa ra lập luận bảo vệ quan điểm dựa trên văn bản gốc.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| Khung Chuẩn bị (Preparedness Framework) | "Chính sách mở rộng của OpenAI" | Bản PF v2 (tháng 4/2025); phân chia danh mục Theo dõi và Nghiên cứu |
| Danh mục Theo dõi (Tracked Category) | "Bắt buộc giảm thiểu" | Kích hoạt Báo cáo Năng lực + Biện pháp bảo vệ; thẩm định bởi SAG |
| Danh mục Nghiên cứu (Research Category) | "Chỉ giám sát thuần túy" | Được theo dõi nhưng không tự động giảm thiểu; bao gồm Quyền tự trị tầm xa |
| Khung An toàn Biên giới (Frontier Safety Framework) | "Chính sách mở rộng của DeepMind" | Bản FSF v3 (tháng 9/2025) + Cấp độ Năng lực Theo dõi (tháng 4/2026) |
| CCL | "Cấp độ Năng lực Nguy cấp" | Ngưỡng năng lực của DeepMind theo từng lĩnh vực (CBRN, Cyber, Thao túng gây hại, ML R&D) |
| Cấp độ tự trị ML R&D mức 1 | "Tự động hóa R&D" | Tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh |
| Cố tình giấu năng lực (Sandbagging) | "Chiến lược hạ thấp hiệu suất" | Mô hình chủ động thể hiện kém trong bài đánh giá; thuộc Danh mục Nghiên cứu của OpenAI |
| Suy luận công cụ (Instrumental reasoning) | "Suy luận phương tiện - mục đích" | Lập luận về cách đạt được mục tiêu; đối tượng giám sát trọng tâm của DeepMind |

## Đọc thêm

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — thông cáo phát hành bản v2.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — toàn văn tài liệu.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — thông cáo phát hành FSF v3.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — bổ sung các Cấp độ Năng lực Theo dõi.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — ví dụ thực tế về một Báo cáo Rủi ro theo định dạng FSF.
