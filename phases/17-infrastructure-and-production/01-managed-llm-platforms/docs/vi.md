# Nền Tảng LLM Được Quản Lý — Bedrock, Vertex AI, Azure OpenAI

> Ba nhà cung cấp dịch vụ đám mây siêu lớn (hyperscalers), ba chiến lược riêng biệt. AWS Bedrock là một chợ mô hình (model marketplace) — gom Claude, Llama, Titan, Stability, Cohere vào sau một API duy nhất. Azure OpenAI là mối quan hệ đối tác độc quyền với OpenAI kết hợp cùng Đơn vị Thông lượng Được cấp phát (Provisioned Throughput Units - PTUs) nhằm cung cấp dung lượng chuyên dụng (dedicated capacity). Vertex AI ưu tiên Gemini trước hết (Gemini-first) với câu chuyện ngữ cảnh dài (long-context) và đa phương thức (multimodal) vượt trội nhất. Vào năm 2026, Artificial Analysis đo lường độ trễ trung vị của Azure OpenAI đạt ~50 ms và Bedrock đạt ~75 ms trên các mô hình tương đương Llama 3.1 405B — chính PTU giải thích cho khoảng cách này vì dung lượng chuyên dụng luôn vượt trội hơn dung lượng dùng chung theo nhu cầu (shared on-demand). Nguyên tắc ra quyết định không phải là "bên nào nhanh nhất" mà là "danh mục mô hình và bề mặt FinOps nào phù hợp với sản phẩm của tôi". Bài học này hướng dẫn bạn đưa ra lựa chọn dựa trên sự đánh đổi rõ ràng trên giấy tờ thay vì cảm tính.

**Type:** Learn
**Languages:** Python (stdlib, bộ so sánh chi phí và độ trễ mô phỏng)
**Prerequisites:** Phase 11 (Kỹ nghệ LLM / LLM Engineering), Phase 13 (Công cụ & Giao thức / Tools & Protocols)
**Time:** ~60 phút

## Mục tiêu Học tập

- Kể tên ba chiến lược nền tảng (chợ mô hình vs độc quyền vs ưu tiên Gemini) và gắn từng chiến lược với một trường hợp sử dụng sản phẩm cụ thể.
- Giải thích giá trị mà Đơn vị Thông lượng Được cấp phát (PTU) mang lại trên Azure OpenAI và lý do tại sao Bedrock theo nhu cầu thường chậm hơn ~25 ms ở quy mô 405B.
- Vẽ sơ đồ bề mặt phân bổ chi phí FinOps (FinOps attribution surface) cho từng nền tảng (Hồ sơ Suy luận Ứng dụng / Application Inference Profiles của Bedrock vs mô hình mỗi nhóm một dự án của Vertex vs phạm vi đăng ký Azure + đặt trước PTU).
- Thiết lập văn bản chính sách "tối thiểu hai nhà cung cấp" (two-provider minimum) và giải thích tại sao phụ thuộc vào một nhà cung cấp duy nhất (single-vendor lock-in) là một sai lầm đắt giá vào năm 2026.

## Vấn đề Thực tế

Bạn đã chọn Claude 3.7 Sonnet cho sản phẩm của mình. Bây giờ bạn cần phục vụ nó trong môi trường thực tế (serving). Bạn có thể gọi trực tiếp API của Anthropic, hoặc gọi thông qua AWS Bedrock, hoặc định tuyến qua một cổng API gateway. Gọi API trực tiếp là cách đơn giản nhất; Bedrock bổ sung thêm các thỏa thuận BAA (Business Associate Agreement), điểm cuối VPC (VPC endpoints), quyền kiểm soát IAM và khả năng phân bổ chi phí qua CloudWatch. Cổng gateway mang lại khả năng chuyển đổi dự phòng (failover), gom hóa đơn hợp nhất và kiểm soát giới hạn tốc độ (rate limits) giữa các nhà cung cấp.

Câu hỏi sâu sắc hơn nằm ở danh mục mô hình. Nếu bạn cần cả Claude, Llama và Gemini trong cùng một sản phẩm, bạn không thể mua tất cả từ một nơi duy nhất trừ khi nơi đó là sự kết hợp đồng thời của Bedrock, Vertex và Azure OpenAI. Các nhà cung cấp đám mây lớn không thể thay thế cho nhau — mỗi bên đặt cược vào một chiến lược khác nhau về việc ai là người làm chủ tầng mô hình (model layer).

Bài học này vạch ra ba canh bạc đó, khoảng cách về độ trễ, khoảng cách về FinOps và rủi ro bị phụ thuộc nền tảng (lock-in).

## Khái niệm

### Ba chiến lược

**AWS Bedrock** — chợ mô hình (marketplace). Claude (Anthropic), Llama (Meta), Titan (nội bộ AWS), Stability (hình ảnh), Cohere (embeddings), Mistral, cùng các danh mục con cho hình ảnh và embedding. Một API duy nhất, một bề mặt quản trị IAM duy nhất, một luồng xuất dữ liệu CloudWatch duy nhất. Canh bạc của Bedrock là khách hàng muốn có nhiều lựa chọn linh hoạt hơn là bị gắn chặt vào một mô hình duy nhất.

**Azure OpenAI** — quan hệ đối tác độc quyền. Bạn nhận được GPT-4 / 4o / 5 / dòng o, DALL·E, Whisper và khả năng tinh chỉnh (fine-tuning) các mô hình OpenAI ngay trong các trung tâm dữ liệu của Azure. Không có mô hình nào ngoài OpenAI trong danh mục "Azure OpenAI Service" — các mô hình bên thứ ba được chuyển sang Azure AI Foundry (một sản phẩm tách biệt). Canh bạc của Azure là OpenAI tiếp tục giữ vị thế dẫn đầu (frontier) và các khách hàng doanh nghiệp muốn có các cơ chế kiểm soát cấp doanh nghiệp dành riêng cho mối quan hệ này.

**Vertex AI** — ưu tiên Gemini trước hết, mọi thứ khác xếp sau. Gemini 1.5 / 2.0 / 2.5 Flash và Pro, cộng thêm Model Garden (mô hình của bên thứ ba). Canh bạc của Vertex là năng lực đa phương thức với ngữ cảnh dài — cửa sổ ngữ cảnh 1 triệu token của Gemini chính là yếu tố tạo nên sự khác biệt.

### Khoảng cách độ trễ ở quy mô lớn

Artificial Analysis thực hiện đánh giá hiệu năng liên tục. Trên các triển khai tương đương của Llama 3.1 405B, độ trễ đến token đầu tiên (Time to First Token - TTFT) trung vị của Azure rơi vào khoảng 50 ms khi sử dụng PTU; trong khi Bedrock dùng chung theo nhu cầu (shared on-demand) rơi vào khoảng 75 ms. Khoảng cách này không phải do AWS kém cỏi — đó là sự khác biệt cơ bản giữa hai mô hình năng lực tính toán. Azure bán các PTU (Provisioned Throughput Units), tức đặt trước dung lượng GPU dành riêng cho tổ chức (tenant) của bạn. Bedrock cũng có cơ chế tương đương (Provisioned Throughput) nhưng giá khởi điểm khoảng $21/giờ cho mỗi đơn vị, và phần lớn khách hàng vẫn chọn dùng chung theo nhu cầu.

Dung lượng dùng chung theo nhu cầu phải cạnh tranh tài nguyên với lưu lượng truy cập của tất cả các khách hàng khác. Dung lượng chuyên dụng thì không. Nếu cam kết chất lượng dịch vụ (SLA) sản phẩm của bạn yêu cầu TTFT < 100 ms ở phân vị P99, bạn buộc phải mua PTU trên Azure, mua Provisioned Throughput của Bedrock, hoặc chấp nhận độ biến thiên ngẫu nhiên mặc định.

### Tính kinh tế của Thông lượng Được cấp phát (Provisioned Throughput)

Azure PTU: một khối tài nguyên tính toán suy luận được đặt trước. Tiết kiệm tới ~70% so với mô hình trả tiền theo nhu cầu đối với các khối lượng công việc có lưu lượng dự đoán được. Chi phí cố định theo giờ bất kể lưu lượng thực tế — bạn phải trả tiền cho phần đặt trước ngay cả khi nhàn rỗi. Điểm hòa vốn (break-even) thường nằm ở mức hiệu suất sử dụng liên tục khoảng 40-60%.

Bedrock Provisioned Throughput: $21-$50 mỗi giờ tùy thuộc vào mô hình và khu vực địa lý. Bài toán kinh tế tương tự — điểm hòa vốn đạt được khi duy trì khoảng một nửa công suất đỉnh. Đòi hỏi cam kết thời gian sử dụng theo tháng.

Dung lượng cấp phát của Vertex được bán theo từng mã định danh (SKU) của Gemini; giá cả thay đổi tùy theo mô hình cùng khu vực và ít được công khai rộng rãi hơn.

### Bề mặt FinOps — yếu tố phân hóa thực sự

**Bedrock Application Inference Profiles** mang lại cơ chế gán nhãn chi phí sạch sẽ nhất trên thị trường. Bạn gắn nhãn cho một hồ sơ (profile) với các thông tin `team`, `product`, `feature`; sau đó định tuyến toàn bộ các lệnh gọi mô hình qua hồ sơ này; CloudWatch sẽ bóc tách chi phí chi tiết theo từng hồ sơ mà không cần xử lý hậu kỳ. Ra mắt vào năm 2025, đây vẫn là giải pháp phân bổ chi tiết nhất được tích hợp sẵn từ nhà cung cấp điện toán đám mây.

Cơ chế phân bổ của **Vertex** dựa trên mô hình mỗi nhóm một dự án (project-per-team) kết hợp gắn nhãn ở mọi nơi. Bạn mô hình hóa mỗi nhóm thành một dự án GCP riêng biệt, đặt nhãn (labels) trên từng tài nguyên, và sử dụng BigQuery Billing Export cùng DataStudio để tổng hợp số liệu. Cách này tốn nhiều công sức thiết lập hơn, nhưng BigQuery cho phép bạn chạy bất kỳ truy vấn SQL tùy biến nào trên dữ liệu chi phí.

**Azure** phụ thuộc vào phạm vi nhóm tài nguyên / gói đăng ký (subscription/resource-group scopes) kèm theo thẻ (tags), trong đó các cam kết đặt trước PTU được coi là đối tượng chi phí hạng nhất. Thẻ được kế thừa từ nhóm tài nguyên chứ không kế thừa theo từng lượt yêu cầu, vì vậy việc phân bổ chi phí chi tiết theo từng yêu cầu đòi hỏi phải thu thập chỉ số tùy biến qua Application Insights hoặc qua một cổng gateway gắn tiêu đề (headers).

Quy luật chung: Bedrock có giải pháp tích hợp sẵn gọn gàng nhất, Vertex linh hoạt nhất nhờ BigQuery, còn Azure mờ đục nhất trừ khi bạn chủ động tự đo đạc (instrumentation).

### Rủi ro phụ thuộc nhà cung cấp (Lock-in) trong năm 2026

Cam kết gắn chặt với một nhà cung cấp đám mây duy nhất từng là lựa chọn chấp nhận được khi chỉ có một mô hình thống trị thị trường. Vào năm 2026, đường biên công nghệ (frontier) dịch chuyển hàng tháng — quý này là Claude 3.7, quý sau là Gemini 2.5, và quý tiếp theo là GPT-5. Việc tự trói mình vào một nền tảng duy nhất đồng nghĩa với việc bạn tự cô lập khỏi 2/3 năng lực công nghệ tiên tiến nhất.

Mô hình mà các đội ngũ kỹ thuật hiệu quả áp dụng là: duy trì chính sách tối thiểu hai nhà cung cấp cho bất kỳ lệnh gọi LLM quan trọng nào trong sản phẩm. Cặp đôi phổ biến nhất là Bedrock kết hợp với Azure OpenAI — gọi Claude từ một bên, GPT từ bên còn lại, chuyển đổi dự phòng linh hoạt giữa hai bên thông qua cùng một cổng gateway. Chi phí phát sinh thêm không đáng kể vì gateway luôn định tuyến tối ưu; trong khi độ tin cậy và tính sẵn sàng khi xảy ra sự cố ngừng dịch vụ (như sự cố Azure OpenAI tháng 1 năm 2025 hay sự cố AWS us-east-1) mang tính chất quyết định sự sống còn của hệ thống.

### Lưu trữ dữ liệu nội địa, thỏa thuận BAA và các ngành chịu kiểm soát pháp lý

Bedrock: Cung cấp BAA tại hầu hết các khu vực; điểm cuối VPC riêng biệt; hàng rào bảo vệ (guardrails). Lựa chọn mặc định phổ biến trong lĩnh vực công nghệ tài chính (fintech).
Azure OpenAI: Đạt chuẩn HIPAA, SOC 2, ISO 27001; hỗ trợ lưu trữ dữ liệu nội địa EU; lựa chọn mặc định của các doanh nghiệp thuộc nhóm quản lý nghiêm ngặt.
Vertex: Tuân thủ HIPAA, GDPR, lưu trữ dữ liệu nội địa theo từng khu vực; thừa hưởng toàn bộ ngăn xếp tuân thủ của Google Cloud.

Cả ba nền tảng đều vượt qua các tiêu chuẩn kiểm tra cơ bản. Điểm khác biệt nằm ở chính sách lưu giữ dữ liệu (data retention), cách thức xử lý nhật ký (logs), và việc tính năng giám sát lạm dụng (abuse-monitoring) có đọc lưu lượng của bạn hay không (mặc định tham gia trên hầu hết nền tảng; cho phép yêu cầu hủy tham gia đối với khách hàng doanh nghiệp).

### Những con số bạn cần ghi nhớ

- Độ trễ TTFT trung vị của Azure OpenAI trên các mô hình tương đương Llama 3.1 405B: ~50 ms (khi dùng PTU).
- Độ trễ TTFT trung vị của Bedrock khi dùng theo nhu cầu: ~75 ms.
- Giá Bedrock Provisioned Throughput: $21-$50/giờ cho mỗi đơn vị.
- Điểm hòa vốn của Azure PTU: ~40-60% mức độ sử dụng duy trì liên tục.
- Mức tiết kiệm của PTU so với theo nhu cầu ở mức sử dụng cao: lên tới 70%.

```figure
i4-platform-lanes
```

## Ứng Dụng Thực Tế (Use It)

Tệp `code/main.py` so sánh ba nền tảng trên một khối lượng công việc mô phỏng — mô hình hóa tính kinh tế giữa trả theo nhu cầu và PTU, độ biến thiên TTFT, cùng độ chuẩn xác trong phân bổ chi phí. Hãy chạy mã nguồn để thấy rõ trường hợp nào PTU mang lại hiệu quả kinh tế và trường hợp nào bề rộng danh mục mô hình của chợ ứng dụng quan trọng hơn khoảng cách TTFT.

## Bàn Giao Thành Phẩm (Ship It)

Bài học này tạo ra tệp artifact `outputs/skill-managed-platform-picker.md`. Dựa trên hồ sơ khối lượng công việc đầu vào (các mô hình cần dùng, cam kết SLA về TTFT, sản lượng hàng ngày, yêu cầu tuân thủ pháp lý), công cụ sẽ đề xuất nền tảng chính, nền tảng dự phòng và kế hoạch đo đạc chi phí FinOps cụ thể.

## Bài Tập Thực Hành

1. Chạy `code/main.py`. Ở mức hiệu suất sử dụng liên tục nào thì Azure PTU đánh bại mô hình trả theo nhu cầu đối với lớp mô hình 70B? Hãy tính điểm hòa vốn và so sánh với dải 40-60% được quảng bá.
2. Sản phẩm của bạn yêu cầu cả Claude 3.7 Sonnet và GPT-4o. Hãy thiết kế một kiến trúc triển khai hai nhà cung cấp — mô hình nào đi với đám mây nào, cổng gateway nào đặt phía trước, chính sách chuyển đổi dự phòng (failover) hoạt động ra sao?
3. Một khách hàng thuộc lĩnh vực y tế yêu cầu nghiêm ngặt thỏa thuận BAA, lưu trữ dữ liệu tại US-East, và P99 TTFT dưới 100 ms. Hãy chọn một nền tảng và bảo vệ quyết định bằng ba tính năng cụ thể.
4. Bạn phát hiện hóa đơn Bedrock tháng này tăng gấp 4 lần dù lưu lượng không đổi. Nếu không có Application Inference Profiles, bạn sẽ tìm ra nguyên nhân bằng cách nào? Nếu có các profile này, bạn mất bao lâu?
5. Đọc trang báo giá của Anthropic API và Bedrock. Đối với khối lượng công việc Claude đạt 100 triệu token/tháng, phương án nào tiết kiệm chi phí hơn — gọi trực tiếp Anthropic API, dùng Bedrock theo nhu cầu, hay mua Bedrock Provisioned Throughput?

## Thuật Ngữ Then Chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật thực tế |
|-----------|----------------------|--------------------------|
| Bedrock | "Dịch vụ LLM của AWS" | Chợ mô hình tích hợp Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "ChatGPT của Azure" | Các mô hình độc quyền của OpenAI đặt trong trung tâm dữ liệu Azure với các kiểm soát doanh nghiệp |
| Vertex AI | "LLM của Google" | Nền tảng ưu tiên Gemini kết hợp Model Garden cho các mô hình bên thứ ba |
| PTU | "Dung lượng chuyên dụng" | Provisioned Throughput Unit — Cụm GPU suy luận được đặt trước, tính phí theo giờ |
| Application Inference Profile | "Gắn thẻ của Bedrock" | Hồ sơ chi phí/mức sử dụng theo từng sản phẩm có gắn thẻ, tích hợp trực tiếp với CloudWatch |
| Model Garden | "Danh mục của Vertex" | Khu vực chứa các mô hình bên thứ ba của Vertex AI, tách biệt với dòng Gemini |
| Tối thiểu hai nhà cung cấp (Two-provider minimum) | "Dự phòng LLM" | Chính sách chạy mọi luồng LLM quan trọng phân tán qua ≥2 nhà cung cấp đám mây lớn |
| BAA | "Thủ tục giấy tờ HIPAA" | Business Associate Agreement; bắt buộc đối với dữ liệu sức khỏe được bảo vệ (PHI); cả ba bên đều hỗ trợ |
| Giám sát lạm dụng (Abuse monitoring) | "Trình theo dõi nhật ký" | Tính năng quét an toàn nội dung prompt/kết quả từ phía nhà cung cấp; có thể hủy tham gia ở gói doanh nghiệp |

## Tài Liệu Đọc Thêm

- [Bảng giá AWS Bedrock](https://aws.amazon.com/bedrock/pricing/) — Bảng biểu phí chính thức và chi phí Provisioned Throughput.
- [Bảng giá Dịch vụ Azure OpenAI](https://azure.microsoft.com/en-us/pricing/details/azure-openai/) — Tính kinh tế của PTU và các biểu phí chi tiết.
- [Bảng giá Vertex AI Generative AI](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Các gói tính phí Gemini và phụ phí Model Garden.
- [Bảng xếp hạng Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — Điểm chuẩn độ trễ và thông lượng liên tục giữa các nhà cung cấp.
- [The AI Journal — Hướng dẫn CTO về AWS Bedrock vs Azure OpenAI 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — Khung ra quyết định cho cấp quản trị doanh nghiệp.
- [Finout — So sánh FinOps giữa Bedrock vs Vertex vs Azure](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — Cơ chế phân bổ chi phí đặt cạnh nhau.
