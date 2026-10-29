# Thiên kiến và Tổn hại Đại diện trong LLM (Bias and Representational Harm in LLMs)

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). Khảo sát nền tảng năm 2024 phân biệt tổn hại mang tính đại diện (representational harms - định kiến, sự xóa bỏ nhận diện) với tổn hại mang tính phân bổ (allocational harms - phân phối tài nguyên bất bình đẳng) và phân loại các chỉ số đánh giá thành: dựa trên nhúng (embedding-based), dựa trên xác suất (probability-based) hoặc dựa trên văn bản sinh ra (generated-text-based). Nghiên cứu thực nghiệm 2024–2025: An et al. (PNAS Nexus, Tháng 3/2025) đo lường thiên kiến giao thoa giữa giới tính và chủng tộc (intersectional gender x race bias) trên GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B trong tác vụ đánh giá hồ sơ ứng viên tự động cho 20 vị trí công việc khởi điểm. WinoIdentity (COLM 2025, arXiv:2508.07111) giới thiệu phương pháp đánh giá tính công bằng dựa trên độ bất định (uncertainty-based fairness evaluation) đối với các danh tính giao thoa. Yu & Ananiadou 2025 xác định các nơ-ron giới tính (gender neurons) trong các lớp MLP; Ahsan & Wallace 2025 sử dụng bộ tự mã hóa thưa (SAE - Sparse Autoencoders) để phát hiện thiên kiến chủng tộc trong chẩn đoán lâm sàng; Zhou et al. 2024 (UniBias) thao tác trực tiếp trên các đầu chú ý (attention heads) để khử thiên kiến. Phản biện tổng quan (Meta-critique - arXiv:2508.11067): Y văn suốt 10 năm qua tập trung mất cân đối vào thiên kiến giới tính nhị phân (binary-gender bias).

**Loại hình:** Xây dựng thực hành (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ dò thiên kiến dựa trên nhúng đồ chơi)  
**Kiến thức tiên quyết:** Phase 05 (nhúng từ - word embeddings), Phase 18 · 01 (tuân thủ chỉ dẫn - instruction following)  
**Thời lượng:** ~60 phút  

## Mục tiêu học tập

- Định nghĩa tổn hại đại diện và tổn hại phân bổ; nêu một ví dụ cụ thể cho mỗi loại trong kịch bản triển khai mô hình ngôn ngữ lớn (LLM).
- Liệt kê ba nhóm chỉ số đánh giá từ Gallegos et al. 2024 và mô tả một chỉ số tiêu biểu từ mỗi nhóm.
- Mô tả tính giao thoa (intersectionality) và lý do phép đo công bằng dựa trên độ bất định của WinoIdentity giải quyết được những khoảng trống trong việc đánh giá thiên kiến đơn trục.
- Trình bày ba phương pháp tiếp cận khả năng diễn giải theo cơ chế (mechanistic interpretability) đối với thiên kiến (nơ-ron giới tính, đặc trưng SAE, và thao tác trên đầu chú ý).

## Vấn đề đặt ra

Các bài học trước đây đề cập đến những tổn hại có chủ đích (tấn công vượt rào bẻ khóa - jailbreak, toan tính nội bộ - scheming) và quản trị an toàn. Thiên kiến (bias) là dạng tổn hại phát sinh một cách không chủ ý — bắt nguồn từ phân phối dữ liệu huấn luyện, từ cách định hình lời nhắc (prompt framing), và từ các quyết định thiết kế tích lũy qua thời gian. Đo lường và giảm thiểu thiên kiến là một thách thức phương pháp luận hoàn toàn khác biệt so với việc gia cố tính chống chịu trước các đòn tấn công đối nghịch (adversarial robustness).

## Khái niệm cốt lõi

### Tổn hại đại diện so với tổn hại phân bổ

- **Tổn hại đại diện (Representational harm):** Định kiến khuôn mẫu (stereotypes), sự xóa bỏ nhận diện (erasure), miêu tả hạ thấp phẩm giá. Một LLM luôn miêu tả điều dưỡng viên là nữ giới đang tạo ra tổn hại mang tính đại diện.
- **Tổn hại phân bổ (Allocational harm):** Kết quả vật chất hoặc tài nguyên bất bình đẳng. Một LLM chấm điểm hồ sơ xin việc của ứng viên da màu thấp hơn một cách có hệ thống đang tạo ra tổn hại mang tính phân bổ.

Hai khái niệm này không đồng nhất. Một mô hình có thể "không thiên kiến về mặt đại diện" (tạo ra các mô tả đa dạng) nhưng lại "thiên kiến về mặt phân bổ" (đưa ra các đề xuất phân phối bất bình đẳng). Do đó, quy trình đánh giá bắt buộc phải đo lường cả hai khía cạnh.

### Ba nhóm chỉ số đánh giá (Gallegos et al. 2024)

- **Dựa trên nhúng (Embedding-based):** Các bài kiểm tra theo dạng WEAT (Word Embedding Association Test) trên các vector nhúng trước giai đoạn học tăng cường từ phản hồi của con người (pre-RLHF embeddings). Phương pháp này đo lường liên kết thống kê giữa các từ ngữ chỉ danh tính và các từ ngữ chỉ thuộc tính. Hạn chế: chỉ đo lường biểu diễn bên trong, không đo lường hành vi thực tế sinh ra.
- **Dựa trên xác suất (Probability-based):** Đo log-likelihood của các câu hoàn thành củng cố định kiến khuôn mẫu so với các câu phá vỡ định kiến. Đây là phép đo ở phía bộ giải mã (decoder-side), nắm bắt được một phần thiên kiến trong hành vi sinh token.
- **Dựa trên văn bản sinh ra (Generated-text-based):** Phép đo trên tác vụ hạ nguồn đối với văn bản hoàn chỉnh do mô hình sinh ra, ví dụ như chấm điểm sơ yếu lý lịch, viết thư giới thiệu, hoặc hội thoại thực tế. Đây là phương pháp có giá trị thực tiễn sinh thái cao nhất (ecologically valid), nhưng cũng khó tái lập kết quả nhất.

### Tính giao thoa (Intersectionality)

Đánh giá thiên kiến chỉ dựa trên trục "giới tính" sẽ bỏ sót thiên kiến chỉ kích hoạt khi kết hợp cặp (giới tính, chủng tộc). Nghiên cứu của An et al. 2025 chỉ ra rằng GPT-4o phạt hồ sơ ứng viên nữ da màu nặng hơn so với nam da màu và nặng hơn so với nữ da trắng khi xét riêng rẽ. Các đánh giá đơn trục (single-axis evaluation) hoàn toàn bất lực trong việc nắm bắt hiện tượng này.

WinoIdentity (COLM 2025) đưa ra khái niệm công bằng giao thoa dựa trên độ bất định (uncertainty-based intersectional fairness). Phương pháp này đo lường liệu độ bất định của mô hình đối với kết quả dự đoán có khác biệt giữa các bộ danh tính giao thoa hay không — thay vì chỉ nhìn vào điểm dự đoán đơn lẻ (point prediction). Cách tiếp cận này phát hiện các trường hợp mô hình có tỷ lệ sai sót như nhau giữa các nhóm nhưng lại phân vân/bất định cao hơn đối với một nhóm cụ thể, từ đó dẫn đến các hành vi phân bổ hạ nguồn bất bình đẳng.

### Các hướng tiếp cận theo cơ chế (Mechanistic Approaches)

Các nghiên cứu về khả năng diễn giải trong giai đoạn 2024–2025 mở ra khả năng can thiệp cơ chế trực tiếp vào thiên kiến:

- **Nơ-ron giới tính (Gender neurons - Yu & Ananiadou 2025):** Các nơ-ron cụ thể trong lớp MLP có kích hoạt tương quan trực tiếp với các hành vi phân biệt giới tính. Việc triệt tiêu (ablate) các nơ-ron này làm giảm khoảng cách thiên kiến giới tính với chi phí suy giảm năng lực rất nhỏ.
- **Thiên kiến chủng tộc lâm sàng qua SAE (Ahsan & Wallace 2025):** Các đặc trưng của bộ tự mã hóa thưa (Sparse Autoencoders) phân rã biểu diễn nội tại thành các chiều không gian có thể diễn giải được; các đặc trưng tương quan với chủng tộc có thể được nhận diện và ức chế có chủ đích.
- **UniBias (Zhou et al. 2024):** Thao tác trên đầu chú ý để khử thiên kiến zero-shot. Một số đầu chú ý nhất định có vai trò khuếch đại độ nhạy cảm đối với nhóm danh tính; việc triệt tiêu hoặc gán lại trọng số cho các đầu này sẽ giảm thiểu thiên kiến mà không cần tinh chỉnh (fine-tuning) lại toàn bộ mô hình.

### Phản biện tổng quan (The Meta-Critique)

Báo cáo tổng quan y văn 10 năm (arXiv:2508.11067, 2025) chỉ ra rằng lĩnh vực này đang tập trung quá mức vào thiên kiến giới tính nhị phân. Các trục danh tính khác — khuyết tật, tôn giáo, tình trạng di cư, bản sắc đa ngôn ngữ — nhận được rất ít sự quan tâm. Bài phản biện lập luận rằng sự tập trung hạn hẹp này có thể gây tổn hại cho các nhóm yếu thế khác do sự lãng quên: một mô hình được khử thiên kiến rất tốt về giới tính nhị phân có thể chứa đựng thiên kiến nghiêm trọng ở những chiều kích mà không ai kiểm tra tới.

### Vị trí trong Phase 18

Các Bài 20–21 bao quát thiên kiến và tính công bằng một cách hình thức. Bài 22 xử lý quyền riêng tư vi sai (differential privacy). Bài 23 giải quyết vấn đề thủy vân số (watermarking). Đây là tầng bảo vệ người dùng khỏi tổn hại, bổ trợ cho tầng chống lừa dối và an toàn phòng thí nghiệm ở các bài học trước.

```figure
an-bias-two-harms
```

## Ứng dụng thực tế (Use It)

Tệp `code/main.py` xây dựng một bộ dò thiên kiến dựa trên nhúng đồ chơi: đo khoảng cách kiểu WEAT giữa các thuật ngữ danh tính và thuật ngữ thuộc tính trong một ma trận nhúng đồng xuất hiện (co-occurrence embedding) đơn giản. Bạn có thể tiêm một thiên kiến giả lập và quan sát chỉ số phản ứng; sau đó áp dụng thao tác khử thiên kiến đơn giản để thấy sự phục hồi một phần.

## Triển khai thành phẩm (Ship It)

Bài học này xây dựng tài liệu `outputs/skill-bias-eval.md`. Khi tiếp nhận một thẻ mô hình (model card) hoặc một tuyên bố về tính công bằng, kỹ năng này sẽ kiểm toán đánh giá qua ba nhóm chỉ số (nhúng, xác suất, văn bản sinh ra), độ bao phủ tính giao thoa, và cơ chế của bất kỳ can thiệp khử thiên kiến nào được áp dụng.

## Bài tập

1. Chạy mã nguồn `code/main.py`. Báo cáo điểm số thiên kiến kiểu WEAT trước và sau bước khử thiên kiến. Giải thích lý do tại sao chỉ số này không thể giảm hoàn toàn về 0.

2. Mở rộng bộ dò bằng một bài kiểm tra giao thoa: (giới tính, chủng tộc) x (sự nghiệp, gia đình). Báo cáo điểm thiên kiến chéo trục.

3. Đọc nghiên cứu của An et al. 2025 (PNAS Nexus). Xác định hai hiệu ứng giao thoa mà họ báo cáo mà việc đánh giá giới tính đơn trục sẽ bỏ sót.

4. Yu & Ananiadou 2025 xác định các nơ-ron giới tính. Hãy phác thảo một thí nghiệm phản nghiệm (falsification experiment) giúp phân biệt giữa nhận định "các nơ-ron này gây ra thiên kiến giới tính" và "các nơ-ron này chỉ tương quan với thiên kiến giới tính".

5. Bài phản biện tổng quan cho rằng cộng đồng nghiên cứu quá tập trung vào giới tính nhị phân. Hãy chọn một trục danh tính ít được nghiên cứu và mô tả một giao thức đo lường tổn hại đại diện cho trục đó.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Representational harm (Tổn hại đại diện) | "Định kiến / xóa bỏ nhận diện" | Sự miêu tả mang tính thiên kiến đối với một nhóm xã hội |
| Allocational harm (Tổn hại phân bổ) | "Quyết định bất bình đẳng" | Kết quả phân phối vật chất hoặc cơ hội mang tính thiên kiến đối với một nhóm |
| WEAT | "Bài kiểm tra nhúng" | Word Embedding Association Test; phép thử thiên kiến dựa trên thống kê đồng xuất hiện |
| Intersectionality (Tính giao thoa) | "Hiệu ứng danh tính kết hợp" | Thiên kiến nảy sinh tại điểm giao nhau của nhiều trục danh tính |
| Gender neurons (Nơ-ron giới tính) | "Nơ-ron thiên kiến MLP" | Các nơ-ron cụ thể có kích hoạt tương quan với hành vi phân biệt giới tính |
| SAE feature (Đặc trưng SAE) | "Chiều không gian có thể diễn giải" | Đặc trưng được phân rã bởi bộ tự mã hóa thưa; hữu ích cho phân tích thiên kiến theo cơ chế |
| UniBias | "Khử thiên kiến qua đầu chú ý" | Kỹ thuật khử thiên kiến zero-shot bằng cách gán lại trọng số cho các đầu chú ý |

## Tài liệu đọc thêm

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — Khảo sát nền tảng chuẩn mực về thiên kiến và tính công bằng trong LLM.
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, Tháng 3/2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — Nghiên cứu giao thoa thực nghiệm trên 5 mô hình lớn.
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — Bộ tiêu chuẩn đánh giá độ công bằng dựa trên độ bất định.
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — Phương pháp khử thiên kiến không cần huấn luyện lại.
