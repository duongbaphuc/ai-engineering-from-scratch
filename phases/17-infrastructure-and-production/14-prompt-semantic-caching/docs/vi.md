# Kinh tế học về Bộ nhớ đệm Câu lệnh và Bộ nhớ đệm Ngữ nghĩa (Prompt Caching and Semantic Caching Economics)

> **Ảnh chụp biểu giá tháng 04/2026.** Các số liệu dưới đây phản ánh bảng giá niêm yết của các nhà cung cấp tại thời điểm xuất bản bài học; hãy đối chiếu với tài liệu gốc được liên kết trước khi trích dẫn cho các hệ thống phía hạ nguồn.

> Bộ nhớ đệm diễn ra ở hai tầng kiến trúc: Tầng L2 (cấp nhà cung cấp - provider-level) lưu đệm câu lệnh/tiền tố (prompt/prefix caching) tái sử dụng cặp khóa-giá trị chú ý (attention KV) cho các tiền tố lặp lại — tài liệu lưu đệm câu lệnh của Anthropic công bố mức giảm chi phí tới 90% và giảm độ trễ 85% trên các câu lệnh dài; với Claude 3.5 Sonnet, lượt đọc đệm chỉ tốn $0,30/triệu token so với $3,00/triệu token khi xử lý mới hoàn toàn với thời gian sống (TTL - time-to-live) 5 phút, và phụ phí ghi gấp 2 lần cho tùy chọn TTL 1 giờ (docs.anthropic.com, 04/2026). Cơ chế lưu đệm câu lệnh của OpenAI được áp dụng tự động cho các prompt từ 1024 token trở lên và giảm giá đầu vào lưu đệm khoảng 90% so với tính toán mới (platform.openai.com, 04/2026); mức giá lưu đệm cụ thể theo từng mô hình phụ thuộc vào biểu giá trực tiếp. Tầng L1 (cấp ứng dụng - app-level) lưu đệm ngữ nghĩa (semantic caching) bỏ qua hoàn toàn việc gọi LLM khi đạt độ tương đồng nhúng (embedding similarity). Mức "độ chính xác 95%" từ nhà cung cấp đề cập đến độ chuẩn xác của kết quả khớp ngữ nghĩa, không phải tỷ lệ trúng đệm (hit rate) — tỷ lệ trúng đệm thực tế ghi nhận trong môi trường sản xuất dao động từ 10% (trò chuyện mở - open-ended chat) đến 70% (hỏi đáp có cấu trúc - structured FAQ), trong khi các tác tử thoại với tập câu lệnh cố định đạt 50-80%; cả hai nhà cung cấp đều không công bố đường chuẩn (baseline) chính thức, vì vậy hãy coi đây là số liệu đo từ xa của cộng đồng thay vì cam kết bảo đảm. Các cạm bẫy trong môi trường sản xuất: chạy song song triệt tiêu bộ nhớ đệm (N yêu cầu song song phát đi trước khi lượt ghi đệm đầu tiên hoàn tất có thể đội chi phí lên gấp nhiều lần), và nội dung động chèn vào tiền tố sẽ ngăn cản hoàn toàn việc trúng đệm. ProjectDiscovery ghi nhận mức tăng vọt từ 7% lên 74% tỷ lệ trúng đệm (11/2025) bằng cách tách văn bản động ra khỏi tiền tố có thể lưu đệm.

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng bộ nhớ đệm hai tầng đồ chơi - toy two-layer cache simulator)  
**Điều kiện tiên quyết:** Giai đoạn 17 · 04 (Serving Engine Internals), Giai đoạn 17 · 06 (SGLang RadixAttention)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Phân biệt lưu đệm prompt/tiền tố L2 (tái sử dụng KV tại nhà cung cấp) với lưu đệm ngữ nghĩa L1 (bỏ qua LLM trên các câu lệnh tương đồng).
- Giải thích cơ chế đánh dấu tường minh `cache_control` của Anthropic và hai tùy chọn TTL (5 phút so với 1 giờ) cùng hệ số nhân giá tương ứng.
- Tính toán khoản tiết kiệm dự kiến hàng tháng dựa trên tỷ lệ trúng đệm, cơ cấu phân bổ prompt/phản hồi và giá token.
- Chỉ rõ phản mẫu (anti-pattern) chạy song song khiến hóa đơn tăng gấp 5-10 lần và phản mẫu nội dung động làm sụp đổ tỷ lệ trúng đệm.

## Vấn đề Thực tế

Bạn tích hợp tính năng lưu đệm câu lệnh (prompt caching) vào dịch vụ RAG của mình. Hóa đơn vẫn giữ nguyên mức cũ. Bạn đo lường tỷ lệ trúng đệm; con số chỉ vỏn vẹn 7%. Các câu lệnh trông có vẻ tĩnh nhưng thực tế không phải — câu lệnh hệ thống (system prompt) chứa ngày giờ hiện tại định dạng đến từng phút, một mã định danh yêu cầu (request ID), và thứ tự các ví dụ được xáo trộn ngẫu nhiên để tăng tính đa dạng. Mỗi yêu cầu đều ghi một bản ghi mới vào bộ nhớ đệm, số lượt đọc thành công là zero.

Ở một diễn biến khác, tác tử của bạn thực hiện đồng thời mười lượt gọi công cụ song song cho mỗi câu hỏi của người dùng. Cả mười yêu cầu cùng đổ tới nhà cung cấp trước khi lượt ghi đệm đầu tiên hoàn tất. Kết quả là mười lượt ghi tốn phí, không có lượt đọc nào trúng đệm. Hóa đơn của bạn tăng gấp 5-10 lần so với dự tính ban đầu khi triển khai cơ chế đệm.

Lưu đệm là một giao thức (protocol), không đơn thuần là một cờ bật tắt. Hai tầng kiến trúc mang hai chế độ lỗi hoàn toàn khác nhau.

## Khái niệm

### L2 — Lưu đệm câu lệnh/tiền tố tại nhà cung cấp

Nhà cung cấp lưu trữ cặp khóa-giá trị chú ý (attention KV) cho một tiền tố có thể lưu đệm và tái sử dụng nó ở yêu cầu tiếp theo khớp với tiền tố đó. Bạn chỉ trả chi phí ghi một lần, các lượt đọc sau đó gần như miễn phí.

**Anthropic (dòng Claude 3.5 / 3.7 / 4)**: sử dụng cờ đánh dấu tường minh `cache_control` trong yêu cầu. Bạn gắn thẻ các khối dữ liệu có thể lưu đệm. Thời gian sống (TTL): 5 phút (chi phí ghi bằng 1,25 lần giá cơ sở) hoặc 1 giờ (chi phí ghi bằng 2 lần giá cơ sở). Chi phí đọc đệm: $0,30/triệu token trên Claude 3.5 Sonnet so với $3,00/triệu token khi gọi mới — rẻ hơn 10 lần (docs.anthropic.com, tính đến tháng 04/2026). Mức giá khác biệt theo từng mô hình (Opus/Haiku được công bố riêng); luôn đối chiếu với bảng giá trực tiếp.

**OpenAI**: lưu đệm tự động cho các câu lệnh ≥1024 token (platform.openai.com, 04/2026). Không cần cờ tường minh. Token đầu vào lưu đệm rẻ hơn khoảng 10 lần so với đầu vào thông thường trên bảng giá hiện tại của gpt-4o/gpt-5. Cả tài liệu lẫn ghi chú phát hành đều không công bố tỷ lệ trúng chuẩn; các báo cáo từ cộng đồng thường tập trung quanh mức 30–60% nếu thiết kế prompt cẩn trọng. Hãy theo dõi chỉ số `usage.cached_tokens` để đo lường tỷ lệ thực tế của bạn.

**Google (Gemini)**: lưu đệm ngữ cảnh (context caching) thông qua API tường minh; ngữ cảnh lên tới 1 triệu token đồng nghĩa với việc lưu đệm mang lại hiệu quả kinh tế vượt trội.

**Hệ thống tự lưu trữ (vLLM, SGLang)**: Giai đoạn 17 · 06 đề cập đến RadixAttention — cùng một mô hình tái sử dụng trên năng lực tính toán của riêng bạn.

### L1 — Lưu đệm ngữ nghĩa cấp ứng dụng

Trước khi thực hiện cuộc gọi tới LLM, hệ thống băm (hash) câu lệnh, tạo nhúng (embedding), và tìm kiếm yêu cầu tương tự đã lưu trong đệm (độ tương đồng cô-sin vượt ngưỡng, thông thường từ 0,95 trở lên). Nếu trúng đệm (hit), trả về phản hồi đã lưu. Nếu trượt đệm (miss), gọi LLM và lưu kết quả vào đệm.

Các giải pháp mã nguồn mở: Redis Vector Similarity, GPTCache, Qdrant. Các giải pháp thương mại: Portkey Cache, Helicone Cache.

Tuyên bố về độ chính xác của nhà cung cấp đề cập đến tần suất phản hồi lưu đệm trả về là phù hợp về mặt ngữ nghĩa — chứ không phải tần suất bạn trúng đệm. Tỷ lệ trúng đệm trong môi trường sản xuất:

- Trò chuyện tự do, kết thúc mở: 10-15%.
- Hỏi đáp có cấu trúc / hỗ trợ khách hàng: 40-70%.
- Câu hỏi liên quan đến mã nguồn: 20-30% (những thay đổi nhỏ làm mất dấu vết khớp).
- Tác tử giọng nói lặp lại câu lệnh: 50-80% (tập dữ liệu chuẩn hóa giọng nói cố định).

### Phản mẫu chạy song song (Parallelization Anti-Pattern)

Tác tử của bạn tạo ra 10 lượt gọi công cụ song song. Cả 10 lượt đều dùng chung câu lệnh hệ thống dài 4K token. Quá trình ghi đệm của Anthropic diễn ra trên từng yêu cầu; lượt ghi đệm đầu tiên mất khoảng 300 ms kể từ khi nhà cung cấp nhận được câu lệnh. Các yêu cầu từ 2 đến 10 đến trong cùng một cửa sổ mili-giây đó và đều gặp tình trạng trượt đệm. Bạn phải trả 10 lần phụ phí ghi đệm (write premium) và không nhận được bất kỳ ưu đãi giảm giá đọc nào.

Cách khắc phục: gom cụm theo nguyên tắc tuần tự trước (sequential-first) — gửi riêng lẻ yêu cầu số 1, sau đó mới đồng loạt phát các yêu cầu từ 2 đến 10 khi bộ nhớ đệm của yêu cầu 1 đã được tạo lập xong. Thao tác này chỉ tăng thêm 300 ms cho lượt gọi công cụ đầu tiên nhưng tránh được việc hóa đơn bị đội lên gấp 5-10 lần.

### Phản mẫu nội dung động (Dynamic Content Anti-Pattern)

Câu lệnh hệ thống của bạn có định dạng:

```text
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Mọi yêu cầu đều mang tính duy nhất. Mọi yêu cầu đều phải ghi đệm mới. Tỷ lệ trúng đệm bằng 0.

Cách khắc phục: di chuyển toàn bộ phần nội dung thực sự tĩnh vào tiền tố có thể lưu đệm; nối thêm các nội dung động vào sau ranh giới bộ nhớ đệm:

```text
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery đã nâng tỷ lệ trúng đệm từ 7% lên 74% nhờ phương pháp này và đã công bố chi tiết phân tích kiến trúc của họ.

### Kết hợp API xử lý hàng loạt và bộ nhớ đệm cho tải công việc qua đêm

Các API xử lý hàng loạt (Batch APIs - Giai đoạn 17 · 15) giảm 50% chi phí với thời gian xử lý trong 24 giờ. Token đầu vào được lưu đệm tiếp tục giảm giá thêm ~10 lần trên mức đó. Các khối lượng công việc phân loại, gán nhãn và tạo báo cáo qua đêm có thể giảm chi phí xuống chỉ còn ~10% so với phương án xử lý đồng bộ không lưu đệm bằng cách xếp chồng hai kỹ thuật này.

### Các con số cần ghi nhớ

Mức giá được ghi nhận vào tháng 04/2026 từ tài liệu của nhà cung cấp và có thể thay đổi sau vài tháng — hãy kiểm tra lại trước khi dựa vào chúng.

- Đọc đệm Anthropic: $0,30/triệu token trên Claude 3.5 Sonnet, rẻ hơn khoảng 10 lần so với đầu vào mới (docs.anthropic.com).
- Phụ phí ghi đệm Anthropic: 1,25 lần (TTL 5 phút) hoặc 2 lần (TTL 1 giờ).
- OpenAI tự động lưu đệm: áp dụng cho câu lệnh ≥1024 token; đầu vào lưu đệm có giá khoảng 10% so với đầu vào thông thường theo biểu giá hiện tại (platform.openai.com).
- Tỷ lệ trúng đệm ngữ nghĩa (theo cộng đồng): ~10% với trò chuyện tự do; lên đến ~70% với FAQ có cấu trúc; 50-80% cho tác tử thoại với tập prompt cố định. Không phải đường chuẩn chính thức từ nhà cung cấp.
- ProjectDiscovery: tăng tỷ lệ trúng đệm từ 7% lên 74% nhờ đưa nội dung động ra khỏi tiền tố (blog dự án, 11/2025).
- Phản mẫu song song: ghi nhận hóa đơn bị đội lên 5–10 lần khi N yêu cầu song song trượt lượt ghi đệm đầu tiên.

```figure
semantic-cache-hit
```

## Sử dụng trong Thực tế

Tệp `code/main.py` mô phỏng cơ chế lưu đệm hai tầng L1 + L2 trên các khối lượng công việc hỗn hợp. Báo cáo tỷ lệ trúng đệm, tổng chi phí hóa đơn và chứng minh khoản phạt tài chính do chạy song song sai quy cách.

## Sản phẩm Chuyển giao

Bài học này xây dựng tệp kỹ năng `outputs/skill-cache-auditor.md`. Dựa trên mẫu câu lệnh (prompt template) và lưu lượng truy cập, công cụ này kiểm toán khả năng lưu đệm và đưa ra đề xuất tái cấu trúc tối ưu.

## Bài tập Thực hành

1. Chạy `code/main.py`. Bật tắt cờ song song (parallelization flag). Hóa đơn thay đổi bao nhiêu?
2. Câu lệnh hệ thống của bạn chứa ngày tháng. Hãy chuyển nó ra ngoài tiền tố. Tính toán công thức tỷ lệ trúng đệm trước và sau khi tối ưu.
3. Tính toán điểm hòa vốn giữa tùy chọn TTL 1 giờ (chi phí ghi 2x) so với TTL 5 phút (chi phí ghi 1,25x) dựa trên tốc độ gửi yêu cầu của bạn.
4. Bộ nhớ đệm ngữ nghĩa ở ngưỡng 0,95 đạt tỷ lệ trúng 20%. Ở ngưỡng 0,85 đạt 50% nhưng xuất hiện phản hồi lưu đệm sai lệch. Hãy chọn ngưỡng phù hợp và đưa ra lập luận bảo vệ.
5. Bạn gom nhóm 10 truy vấn con song song cho mỗi câu hỏi người dùng. Hãy viết lại luồng xử lý để tối ưu bộ nhớ đệm mà không làm gia tăng độ trễ đầu-cuối.

## Thuật ngữ Then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| L2 prompt cache | "Bộ nhớ đệm tiền tố" | Nhà cung cấp lưu trữ cặp KV cho các tiền tố lặp lại |
| `cache_control` | "Cờ đệm của Anthropic" | Thuộc tính tường minh đánh dấu các khối nội dung có thể lưu đệm |
| Phụ phí ghi đệm (Cache write premium) | "Thuế ghi (write tax)" | Chi phí phụ trội cho lần trượt đệm đầu tiên cần ghi đệm (1,25x hoặc 2x) |
| L1 semantic cache | "Bộ nhớ đệm nhúng" | Băm và tạo nhúng ở cấp ứng dụng trước khi gọi LLM |
| GPTCache | "Thư viện đệm LLM" | Thư viện mã nguồn mở phổ biến cho bộ nhớ đệm L1 |
| Tỷ lệ trúng đệm (Cache hit rate) | "Trúng / Tổng số" | Tỷ lệ phần trăm các yêu cầu được phục vụ trực tiếp từ bộ nhớ đệm |
| Phản mẫu song song (Parallelization anti-pattern) | "Bẫy N lượt ghi" | N yêu cầu song song làm trượt bộ nhớ đệm N lần |
| Bẫy nội dung động (Dynamic content trap) | "Bẫy gắn thời gian vào prompt" | Các byte dữ liệu động trong tiền tố triệt tiêu tỷ lệ trúng đệm |
| RadixAttention | "Bộ nhớ đệm nội bộ bản sao" | Cơ chế hiện thực hóa bộ nhớ đệm tiền tố trong SGLang |

## Tài liệu Tham khảo thêm

- [Lưu đệm câu lệnh của Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — Ngữ nghĩa `cache_control` chính thức và các mức TTL.
- [Lưu đệm câu lệnh của OpenAI](https://platform.openai.com/docs/guides/prompt-caching) — Hành vi tự động lưu đệm và điều kiện áp dụng.
- [TianPan — Bộ nhớ đệm ngữ nghĩa cho LLM trong sản xuất](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cắt giảm 59% chi phí LLM nhờ Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
