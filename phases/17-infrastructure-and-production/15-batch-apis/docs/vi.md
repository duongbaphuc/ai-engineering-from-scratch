# API Xử lý Hàng loạt — Mức Giảm giá 50% như một Tiêu chuẩn Ngành (Batch APIs — the 50% Discount as Industry Standard)

> Mọi nhà cung cấp lớn đều cung cấp một API xử lý hàng loạt bất đồng bộ (async batch API) với mức giảm giá 50% và thời gian hoàn tất (turnaround time) trong khoảng 24 giờ. OpenAI, Anthropic, Google cùng hầu hết các nền tảng suy luận (gói batch của Fireworks, batch của Together) đều áp dụng chung một mô hình này. Khi xếp chồng cơ chế xử lý hàng loạt với lưu đệm câu lệnh (prompt caching), các đường ống xử lý qua đêm có thể giảm chi phí xuống chỉ còn ~10% so với phương án xử lý đồng bộ không lưu đệm. Nguyên tắc ở đây vô cùng đơn giản: nếu một tác vụ chấp nhận độ trễ tính bằng giờ, nó bắt buộc phải chạy qua batch. Đường ống tạo nội dung, phân loại tài liệu, trích xuất dữ liệu, lập báo cáo, gán nhãn hàng loạt, gắn thẻ danh mục sản phẩm — bất cứ tác vụ nào chịu được độ trễ 24 giờ đều đang lãng phí ngân sách trừ khi được chuyển sang batch. Mô hình sản xuất năm 2026 là phân loại (triage) mọi khối lượng công việc LLM mới vào ba luồng riêng biệt: tương tác trực tiếp (interactive - đồng bộ kết hợp lưu đệm), bán tương tác (semi-interactive - hàng đợi bất đồng bộ có cơ chế dự phòng), và xử lý hàng loạt (batch - chạy qua đêm, xếp chồng token lưu đệm). Những tác vụ giả lập tương tác trực tiếp nhưng thực tế chịu được độ trễ vài phút chính là nguồn lãng phí lớn nhất.

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng chi phí xử lý hàng loạt so với đồng bộ đồ chơi - toy batch-vs-sync cost simulator)  
**Điều kiện tiên quyết:** Giai đoạn 17 · 14 (Prompt & Semantic Caching)  
**Thời lượng:** ~45 phút  

## Mục tiêu Học tập

- Nắm rõ ba API xử lý hàng loạt của các nhà cung cấp lớn (OpenAI, Anthropic, Google) cùng cam kết giảm giá 50% và thời gian hoàn tất trong 24 giờ.
- Tính toán chi phí khi xếp chồng batch + đầu vào lưu đệm cho khối lượng công việc phân loại qua đêm và so sánh với mức cơ sở đồng bộ không lưu đệm.
- Phân luồng khối lượng công việc thành tương tác trực tiếp / bán tương tác / xử lý hàng loạt và đưa ra lập luận bảo vệ cho từng luồng.
- Nhận diện hai cạm bẫy: tương tác một phần (người dùng kỳ vọng kết quả nhanh hơn 24 giờ) và độ lệch lược đồ đầu ra (định dạng tệp batch khác nhau giữa các nhà cung cấp).

## Vấn đề Thực tế

Nhóm của bạn xây dựng một đường ống tạo báo cáo chạy hàng đêm: đọc 50.000 tài liệu, tóm tắt từng tài liệu, gom cụm các bản tóm tắt và soạn thảo một bản tóm lược gửi ban điều hành. Chạy theo phương thức đồng bộ mất 4 giờ với chi phí $2.000 mỗi đêm. Bạn nghe nói về Batch API.

Cơ chế xử lý hàng loạt giúp bạn giảm ngay 50% chi phí. Bạn tiếp tục bật tính năng lưu đệm câu lệnh (prompt caching) trên phần prompt hệ thống (dùng chung cho toàn bộ 50.000 lượt gọi). Khi kết hợp cả hai, hóa đơn giảm xuống chỉ còn $180 mỗi đêm — tương đương ~9% mức chi phí cơ sở ban đầu. Cùng một đường ống dữ liệu, chỉ thay đổi ba dòng cấu hình.

Batch là đòn bẩy tiết kiệm chi phí rẻ nhất và mạnh mẽ nhất trong bộ công cụ tối ưu hóa LLM nhưng thường bị bỏ quên. Lý do phần lớn đến từ thói quen tổ chức: các nhóm kỹ thuật thường mặc định tư duy "thời gian thực" (real-time) trong khi cam kết SLA thực tế chỉ yêu cầu "có kết quả vào sáng hôm sau". Bài học này giúp bạn không để lãng phí 90% ngân sách hóa đơn trên bàn đàm phán.

## Khái niệm

### Ba API xử lý hàng loạt phổ biến

**OpenAI Batch API**: Tải lên tệp JSONL chứa danh sách các yêu cầu. Cam kết hoàn tất trong 24 giờ (thực tế thường chỉ mất ~2-8 giờ). Giảm giá 50% cho cả token đầu vào và token đầu ra. Điểm cuối `/v1/batches`. Các đầu vào đủ điều kiện lưu đệm vẫn được hưởng thêm mức giá ưu đãi của token lưu đệm (cached input).

**Anthropic Message Batches**: Tải lên tệp JSONL. Thời gian hoàn tất trong 24 giờ. Giảm giá 50%. Hỗ trợ cờ `cache_control` — các lượt ghi đệm được định nghĩa tường minh, còn các lượt đọc đệm diễn ra hoàn toàn tự động ngay trong lô xử lý.

**Google Vertex AI Batch Prediction**: Nhận dữ liệu đầu vào từ BigQuery hoặc Google Cloud Storage (GCS). Mức giảm giá 50% tương tự áp dụng cho các dòng mô hình Gemini. Tích hợp sâu vào các đường ống Vertex pipelines.

### Bản chất ngữ nghĩa: Bất đồng bộ, không phải chậm chạp

Batch mang thông điệp "tôi cam kết trả kết quả trong vòng 24 giờ" — chứ không phải "tác vụ này sẽ chạy mất 24 giờ". Phân vị P50 điển hình chỉ mất 2-6 giờ. Nhà cung cấp sẽ lập lịch xử lý lô của bạn vào các khung giờ thấp điểm khi năng lực tính toán của GPU đang dư thừa.

### Xếp chồng xử lý hàng loạt với lưu bộ nhớ đệm

Xét bài toán tóm tắt 50.000 tài liệu dùng chung một câu lệnh hệ thống dài 4K token:

- Đồng bộ không lưu đệm: 50.000 × ($đầu_vào × 4000 + $đầu_ra × 200) ở mức giá nguyên bản.
- Đồng bộ có lưu đệm: câu lệnh hệ thống được lưu đệm sau lượt ghi đầu tiên; 49.999 yêu cầu còn lại nhận mức giá đầu vào rẻ hơn 10 lần.
- Xử lý hàng loạt có lưu đệm: bao gồm toàn bộ lợi thế trên cộng thêm mức giảm giá 50% cho cả lượt đọc lẫn lượt ghi.

Công thức xếp chồng: xử lý hàng loạt (batch) + lưu đệm (cache) = ~10% hóa đơn đồng bộ thông thường. Bất kỳ khối lượng công việc nào chạy qua đêm và có chung câu lệnh hệ thống đều bắt buộc phải áp dụng kiến trúc này.

### Phân luồng khối lượng công việc (Workload Triage)

**Tương tác trực tiếp (Interactive)** — người dùng trực tiếp chờ phản hồi. Chỉ số TTFT là sống còn. Lượt gọi đồng bộ kết hợp lưu đệm câu lệnh. Không thể chuyển sang batch.

**Bán tương tác (Semi-interactive)** — người dùng gửi một tác vụ và quay lại kiểm tra sau vài phút. Sử dụng hàng đợi bất đồng bộ có cơ chế dự phòng quay lại gọi đồng bộ nếu batch không khả dụng. Ví dụ điển hình: lập chỉ mục RAG với khối lượng vừa phải.

**Xử lý hàng loạt (Batch)** — người dùng kỳ vọng nhận kết quả "vào sáng hôm sau" hoặc "vào cuối ngày". Các đường ống nội dung, phân loại dữ liệu quy mô lớn, phân tích ngoại tuyến. Luôn luôn chọn batch, luôn luôn xếp chồng với bộ nhớ đệm.

Sai lầm phổ biến: gắn nhãn tất cả mọi thứ là "tương tác trực tiếp" chỉ vì đường ống đó phục vụ môi trường sản xuất. Sản xuất không phải là một thông số kỹ thuật về độ trễ — SLA mới là yếu tố quyết định.

### Bẫy tương tác một phần (The Partial-Interactivity Trap)

Nhiều tính năng thoạt nhìn có vẻ mang tính tương tác trực tiếp nhưng thực tế chấp nhận được thời gian chờ 5-10 phút. Ví dụ: báo cáo sức khỏe khách hàng hàng đêm có nút bấm "Làm mới" (Refresh). Người dùng bấm làm mới; họ hoàn toàn vui vẻ nếu phải chờ 10 phút. Nhóm phát triển lại triển khai tính năng này theo phương thức đồng bộ. 50 yêu cầu làm mới đồng thời sẽ ngốn chi phí gấp 10 lần so với việc xử lý theo lô và gửi kết quả qua email.

Câu hỏi then chốt cần đặt ra: "Khoảng thời gian chờ 24 giờ có ý nghĩa thế nào đối với người dùng này?" Nếu câu trả lời là "họ thậm chí chẳng nhận ra sự khác biệt", hãy chuyển sang batch ngay lập tức.

### Bẫy độ lệch lược đồ đầu ra (The Output-Schema Trap)

Định dạng tệp của các dịch vụ batch có sự khác biệt giữa các nhà cung cấp:

- OpenAI: JSONL, mỗi yêu cầu nằm trên một dòng riêng biệt.
- Anthropic: JSONL, mỗi thông điệp một dòng; định dạng phản hồi được nhúng kèm bên trong.
- Vertex: Bảng BigQuery hoặc tiền tố GCS với định dạng TFRecord.

Việc viết "một client xử lý batch duy nhất" hoạt động xuyên suốt các nhà cung cấp đòi hỏi phải có lớp mã điều hợp (adapter code) riêng cho từng bên. Các cổng gateway quảng bá tính năng batch đa nhà cung cấp (như Portkey hay một số gói của LiteLLM) thực chất chỉ là một lớp bọc mỏng bên ngoài định dạng thô của từng nhà cung cấp.

### Các con số cần ghi nhớ

- Giảm giá của Batch API trên các nhà cung cấp: giảm cố định 50% cho cả token đầu vào và đầu ra.
- Cam kết SLA thời gian hoàn tất: bảo đảm trong 24 giờ, P50 thực tế đạt 2-6 giờ.
- Xếp chồng batch + đầu vào lưu đệm: chi phí giảm còn ~10% so với đồng bộ không đệm.
- Quy tắc phân luồng: nếu chấp nhận độ trễ 24 giờ, luôn luôn chọn batch.

```figure
batch-lane-triage
```

## Sử dụng trong Thực tế

Tệp `code/main.py` tính toán và so sánh chi phí giữa các phương án: đồng bộ thông thường, đồng bộ có đệm, xử lý hàng loạt, và xử lý hàng loạt có đệm cho một khối lượng công việc 50.000 tài liệu. Báo cáo số tiền tiết kiệm được tính theo USD và tỷ lệ phần trăm.

## Sản phẩm Chuyển giao

Bài học này xây dựng tệp kỹ năng `outputs/skill-batch-triager.md`. Dựa trên các đặc tính của khối lượng công việc, công cụ sẽ phân luồng vào nhóm tương tác trực tiếp / bán tương tác / xử lý hàng loạt và ước tính khoản chi phí tiết kiệm được.

## Bài tập Thực hành

1. Chạy `code/main.py`. Với một đường ống 100.000 tài liệu sử dụng prompt hệ thống dài 3K token và đầu ra 500 token, hãy tính số tiền tiết kiệm được khi xếp chồng toàn diện (batch + cache) so với mức cơ sở đồng bộ.
2. Chọn ba tính năng trong một sản phẩm thực tế mà bạn biết. Hãy phân loại từng tính năng vào luồng tương tác trực tiếp / bán tương tác / xử lý hàng loạt.
3. Người dùng phàn nàn rằng báo cáo của họ mất tới 3 giờ mới hoàn thành. Đó là do phân loại sai vào batch hay là một tác vụ tương tác thực sự? Hãy viết tiêu chí ra quyết định.
4. SLA trả kết quả Batch API của bạn là 24 giờ nhưng P99 thực tế lên tới 20 giờ. Bạn sẽ thông báo điều này tới người dùng như thế nào — và hành vi của hệ thống phía hạ nguồn cần xử lý trường hợp biên này ra sao?
5. Tính toán điểm hòa vốn: độ dài tiền tố dùng chung tối thiểu là bao nhiêu để phương án batch + cache trở nên rẻ hơn việc thuê GPU chuyên dụng chạy qua đêm?

## Thuật ngữ Then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Batch API | "Ưu đãi bất đồng bộ" | Giảm giá 50% chi phí với cam kết hoàn tất trong 24 giờ |
| JSONL | "Định dạng batch" | Mỗi dòng là một đối tượng JSON; tiêu chuẩn của OpenAI/Anthropic |
| Message Batches | "Batch của Anthropic" | Tên thương phẩm của API xử lý hàng loạt từ Anthropic |
| Batch prediction | "Batch của Vertex" | Tên sản phẩm API xử lý hàng loạt từ Google Vertex AI |
| Cam kết SLA hoàn tất (Turnaround SLA) | "Lời hứa 24 giờ" | Cam kết tối đa, không phải thời gian trung bình; P50 thực tế là 2-6h |
| Phân luồng công việc (Workload triage) | "Quyết định tính tương tác" | Quyết định điều hướng tác vụ vào luồng trực tiếp / bán trực tiếp / batch |
| Lược đồ đầu ra (Output schema) | "Định dạng phản hồi" | Cấu trúc tệp JSONL theo từng nhà cung cấp; không thể dùng thay thế trực tiếp |
| Ưu đãi xếp chồng (Stacked discount) | "Batch + cache" | Chi phí giảm chỉ còn ~10% so với đồng bộ không lưu đệm khi áp dụng cả hai |

## Tài liệu Tham khảo thêm

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — Định dạng JSONL và ngữ nghĩa điểm cuối `/v1/batches`.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — Định dạng batch và cơ chế tương tác với `cache_control`.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) — Ngữ nghĩa xử lý batch cho mô hình Gemini.
- [Finout — So sánh giá API OpenAI và Anthropic 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — So sánh chi phí API LLM 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
