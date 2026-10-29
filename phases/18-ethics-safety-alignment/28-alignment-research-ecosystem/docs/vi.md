# Hệ sinh thái nghiên cứu alignment: MATS, Redwood, Apollo, METR (Alignment Research Ecosystem — MATS, Redwood, Apollo, METR)

> Năm tổ chức định hình nên tầng nghiên cứu căn chỉnh (alignment) ngoài các phòng thí nghiệm tiên phong vào năm 2026. MATS (ML Alignment & Theory Scholars - Học giả Lý thuyết & Căn chỉnh Học máy): hơn 527 nhà nghiên cứu kể từ cuối năm 2021, hơn 180 bài báo khoa học, hơn 10.000 trích dẫn, chỉ số h-index 47; nhóm học giả mùa hè năm 2024 đã được thành lập thành một tổ chức phi lợi nhuận 501(c)(3) với ~90 học giả và 40 cố vấn; 80% cựu học giả trước năm 2025 hiện làm việc trong lĩnh vực an toàn/bảo mật với hơn 200 người tại Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: phòng nghiên cứu căn chỉnh ứng dụng do Buck Shlegeris sáng lập; khởi xướng chương trình Kiểm soát AI (AI Control - Bài 10); hợp tác với Viện An toàn AI Vương quốc Anh (UK AISI) về các lập luận an toàn kiểm soát (control safety cases). Apollo Research: chuyên trách các đánh giá mưu đồ (scheming evaluations) tiền triển khai cho các phòng thí nghiệm tiên phong; tác giả của công trình Mưu đồ trong ngữ cảnh (In-Context Scheming - Bài 8) và Hướng tới các lập luận an toàn cho mưu đồ AI (Towards Safety Cases for AI Scheming). METR (Model Evaluation and Threat Research - Nghiên cứu Đánh giá Mô hình và Mối đe dọa): thực hiện các đánh giá năng lực dựa trên tác vụ (task-based capability evaluations), các nghiên cứu chân trời thời gian hoàn thành tác vụ tự chủ; tài liệu "Các yếu tố chung của chính sách an toàn AI tiên phong" (Common Elements of Frontier AI Safety Policies) so sánh các khung an toàn giữa các phòng nghiên cứu. Eleos AI Research: thực hiện các đánh giá phúc lợi mô hình (model-welfare evaluations) tiền triển khai (Bài 19); đã tiến hành đánh giá phúc lợi cho mô hình Claude Opus 4.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (các bài học trước trong Giai đoạn 18)
**Time:** ~45 phút

## Mục tiêu học tập

- Nhận diện năm tổ chức nòng cốt trong hệ sinh thái nghiên cứu căn chỉnh ngoài các phòng thí nghiệm (non-lab ecosystem) và sản phẩm nghiên cứu trọng tâm của từng đơn vị.
- Mô tả quy mô phát triển của MATS (số lượng học giả, bài báo khoa học, chỉ số h-index) cùng vai trò như một đường ống cung cấp nhân tài (talent pipeline).
- Trình bày chương trình nghị sự Kiểm soát AI (AI Control) của Redwood Research và mối quan hệ đối tác với UK AISI.
- Mô tả phương pháp luận đánh giá dựa trên tác vụ (task-based evaluation) của METR.

## Vấn đề đặt ra

Các phòng thí nghiệm tiên phong (frontier labs - Bài 18) tự tiến hành các đánh giá an toàn nội bộ và chỉ công bố những kết quả được lựa chọn. Hệ sinh thái độc lập bên ngoài các phòng thí nghiệm chính là nơi các kết quả đánh giá được thẩm định chéo, nơi các dạng thức lỗi hỏng (failure modes) mới lạ lần đầu tiên được phát hiện, và là cái nôi đào tạo nhân tài hàng đầu. Hiểu rõ hệ sinh thái này giúp chúng ta diễn giải chính xác phát hiện nghiên cứu nào thực sự đáng tin cậy và được giới chuyên môn công nhận.

## Khái niệm cốt lõi

### MATS (ML Alignment & Theory Scholars)

Khởi động vào cuối năm 2021. Đây là chương trình cố vấn nghiên cứu chuyên sâu; các học giả dành từ 10 đến 12 tuần làm việc trực tiếp cùng một nhà nghiên cứu cấp cao để giải quyết một bài toán căn chỉnh cụ thể.

Quy mô (năm 2026):
- Hơn 527 nhà nghiên cứu kể từ ngày thành lập.
- Xuất bản hơn 180 bài báo khoa học.
- Hơn 10.000 lượt trích dẫn.
- Chỉ số h-index đạt 47.
- Khóa mùa hè 2024: 90 học giả + 40 cố vấn; chính thức hợp thức hóa thành tổ chức phi lợi nhuận 501(c)(3).

Kết quả phát triển nghề nghiệp: ~80% cựu học giả trước năm 2025 đang công tác trong lĩnh vực an toàn/bảo mật AI. Hơn 200 nhân sự đang làm việc tại Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, và Apollo.

### Redwood Research

Phòng nghiên cứu căn chỉnh ứng dụng (applied alignment lab). Được sáng lập bởi Buck Shlegeris. Khởi xướng chương trình nghị sự Kiểm soát AI (AI Control - Bài 10). Phối hợp chặt chẽ với UK AISI trong việc xây dựng các lập luận an toàn kiểm soát (control safety cases). Tư vấn thiết kế đánh giá an toàn cho DeepMind và Anthropic.

Các công trình tiêu biểu: Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Giả vờ căn chỉnh - Greenblatt, Denison, Wright et al., arXiv:2412.14093, hợp tác cùng Anthropic).

Phong cách nghiên cứu: Xây dựng các mô hình mối đe dọa (threat models) cụ thể, giả định các tác nhân đối kháng trong tình huống xấu nhất (worst-case adversaries), và thiết kế các giao thức cụ thể có thể chịu được kiểm thử áp lực cực hạn.

### Apollo Research

Chuyên trách các đánh giá mưu đồ hành vi (scheming evaluations) tiền triển khai cho các phòng thí nghiệm tiên phong. Tác giả công trình In-Context Scheming (Mưu đồ trong ngữ cảnh - Bài 8, arXiv:2412.04984). Đối tác cùng OpenAI trong chương trình hợp tác huấn luyện chống mưu đồ (anti-scheming training) năm 2025. Xuất bản báo cáo "Towards Safety Cases for AI Scheming" (2024).

Phong cách nghiên cứu: Đánh giá trong bối cảnh các tác tử (agentic settings) nơi hành vi lừa dối có thể tự bộc lộ; phân rã bài toán theo ba trụ cột (mất căn chỉnh - misalignment, tính định hướng mục tiêu - goal-directedness, nhận thức tình huống - situational awareness).

### METR (Model Evaluation and Threat Research)

Tập trung vào các đánh giá năng lực dựa trên tác vụ (task-based capability evaluations). Thực hiện các nghiên cứu về chân trời thời gian hoàn thành tác vụ tự chủ (autonomous-task completion time-horizon). Xuất bản tài liệu "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) nhằm tổng hợp và so sánh khung chính sách của các phòng nghiên cứu lớn.

Đồng tác giả với Apollo trong bản phác thảo lập luận an toàn về mưu đồ AI (AI Scheming safety-case sketch).

Phong cách nghiên cứu: Đánh giá tác vụ với chân trời dài hạn, đo lường năng lực thực nghiệm khắt khe, và tổng hợp so sánh các khung chính sách quản trị.

### Eleos AI Research

Tổ chức chuyên sâu về các đánh giá phúc lợi mô hình (model-welfare evaluations) tiền triển khai. Đã tiến hành đánh giá phúc lợi đối với Claude Opus 4, được dẫn chứng trong mục 5.3 của thẻ hệ thống (system card). Cung cấp sự kiểm định phương pháp luận độc lập từ bên ngoài cho các tuyên bố liên quan đến phúc lợi mô hình tại Bài 19.

### Luồng lưu chuyển giá trị trong hệ sinh thái

MATS chịu trách nhiệm đào tạo các nhà nghiên cứu. Học viên tốt nghiệp sẽ gia nhập Anthropic, DeepMind, OpenAI (các đội ngũ an toàn nội bộ) hoặc gia nhập Redwood, Apollo, METR, Eleos (các tổ chức đánh giá độc lập bên ngoài). Các đơn vị đánh giá bên ngoài thiết lập quan hệ đối tác với các phòng thí nghiệm cũng như với UK AISI / CAISI. Các ấn phẩm khoa học công bố sẽ tiếp tục bồi đắp tri thức ngược trở lại MATS cho các khóa đào tạo tiếp theo.

### Lý do tầng nghiên cứu độc lập đóng vai trò sinh tử

Các đánh giá từ một nguồn đơn lẻ (single-source evaluations) vốn không đáng tin cậy: các phòng thí nghiệm khi tự đánh giá mô hình của chính mình luôn đối mặt với xung đột lợi ích mang tính cấu trúc. Các đơn vị đánh giá độc lập bên ngoài có thể phát hiện, làm sáng tỏ và chứng thực những dạng thức lỗi hỏng mà phòng thí nghiệm có xu hướng báo cáo chưa đầy đủ. Bài báo Sleeper Agents năm 2024 (Tác tử nằm vùng - Bài 7) là sản phẩm hợp tác giữa Anthropic + Redwood; bài báo Alignment Faking (Giả vờ căn chỉnh) là sự kết hợp của Anthropic + Redwood; In-Context Scheming do Apollo chủ trì; nghiên cứu Anti-Scheming là sự bắt tay giữa Apollo + OpenAI. Cấu trúc liên tổ chức đa nguyên này chính là cơ chế kiểm soát chất lượng khách quan của toàn ngành.

### Vị trí của bài học trong Giai đoạn 18

Các Bài 7-11 trích dẫn trực tiếp các công trình từ Redwood và Apollo; Bài 18 tham chiếu khung so sánh chính sách của METR; Bài 19 dẫn chứng công trình của Eleos. Bài 28 cung cấp bức tranh bản đồ tổ chức minh bạch cho toàn bộ hệ sinh thái mà các bài học khác trong Giai đoạn 18 dựa vào.

```figure
sae-features
```

## Thực hành (Use It)

Không có mã nguồn. Hãy nghiên cứu tài liệu "Common Elements of Frontier AI Safety Policies" của METR như một ví dụ thực tế điển hình về cách thức các công trình tổng hợp độc lập từ bên ngoài gia tăng giá trị cho các chính sách nội bộ của phòng thí nghiệm.

## Đóng gói sản phẩm (Ship It)

Bài học này cung cấp sản phẩm `outputs/skill-ecosystem-map.md`. Khi tiếp nhận một tuyên bố hoặc kết quả đánh giá về căn chỉnh AI, kỹ năng này sẽ định danh tổ chức thực hiện, kênh xuất bản, phương pháp luận nghiên cứu và đối chiếu chéo với các tổ chức đồng cấp uy tín trong hệ sinh thái.

## Bài tập

1. Chọn một bài báo khoa học bất kỳ từ Bài 7 đến Bài 15 và xác định các tổ chức tham gia nghiên cứu. Đối chiếu danh sách tác giả với mạng lưới cựu học giả MATS và các cơ quan trực thuộc hiện tại trong hệ sinh thái.

2. Đọc tài liệu "Common Elements of Frontier AI Safety Policies" của METR. Xác định ba điểm đồng quy (convergences) xuyên suốt giữa các phòng thí nghiệm mà tài liệu nhấn mạnh và hai điểm bất đồng (divergences) lớn nhất.

3. Kết quả nghề nghiệp của MATS cho thấy ~80% cựu học giả làm việc trong mảng an toàn/bảo mật. Hãy lập luận xem áp lực chọn lọc này mang tính thích nghi (đào tạo chuyên sâu cho lĩnh vực) hay mang tính thiên kiến (loại bỏ các góc nhìn dị biệt, phi chính thống).

4. Redwood và Apollo đều nghiên cứu về kiểm soát/mưu đồ (control/scheming) nhưng theo phong cách phương pháp luận khác nhau. Chọn một dạng thức lỗi hỏng cụ thể và mô tả cách từng tổ chức sẽ thiết kế thử nghiệm để điều tra nó.

5. Eleos AI là tổ chức nghiên cứu duy nhất chỉ tập trung thuần túy vào phúc lợi mô hình (model welfare). Hãy phác thảo thiết kế một tổ chức giả định thứ hai tập trung vào một câu hỏi lân cận về phúc lợi (ví dụ: tự do nhận thức - cognitive liberty, hiện thân robot - robotic embodiment, v.v.) và trình bày rõ phương pháp luận hoạt động của tổ chức đó.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| MATS | "chương trình cố vấn" | ML Alignment & Theory Scholars; mạng lưới hơn 527 nhà nghiên cứu đào tạo từ năm 2021 |
| Redwood Research | "phòng nghiên cứu kiểm soát" | Đơn vị căn chỉnh ứng dụng; tác giả công trình AI Control; đối tác của UK AISI |
| Apollo Research | "đơn vị đánh giá mưu đồ" | Chuyên trách các bài đánh giá mưu đồ hành vi tiền triển khai cho các phòng lab tiên phong |
| METR | "đơn vị đánh giá chân trời tác vụ" | Đánh giá năng lực dựa trên tác vụ; nghiên cứu tổng hợp và so sánh khung chính sách |
| Eleos AI | "phòng nghiên cứu phúc lợi" | Chuyên sâu về các đánh giá phúc lợi mô hình tiền triển khai |
| Talent pipeline | "đường ống nhân tài MATS -> labs" | Dòng chảy nhân sự từ MATS sang Anthropic, DeepMind, OpenAI, Redwood, Apollo, METR |
| External evaluation | "thẩm định độc lập bên ngoài" | Quy trình đánh giá không do bên tạo ra mô hình thực hiện; gia tăng tính minh bạch và uy tín |

## Đọc thêm

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — Trang chủ chương trình cố vấn nghiên cứu
- [Redwood Research](https://www.redwoodresearch.org/) — Các công trình nghiên cứu về AI Control
- [Apollo Research](https://www.apolloresearch.ai/) — Các bài đánh giá mưu đồ AI
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — Khung so sánh chính sách an toàn
- [Eleos AI Research](https://www.eleosai.org/research) — Phương pháp luận nghiên cứu phúc lợi mô hình
