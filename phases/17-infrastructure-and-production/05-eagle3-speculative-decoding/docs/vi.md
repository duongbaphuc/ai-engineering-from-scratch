# Giải Mã Suy Đoán EAGLE-3 Trong Môi Trường Sản Xuất

> Giải mã suy đoán (speculative decoding) kết hợp một mô hình nháp (draft model) tốc độ cao với mô hình mục tiêu (target model). Mô hình nháp đề xuất K token; mô hình mục tiêu kiểm định toàn bộ K token trong một lượt lan truyền tiến (forward pass) duy nhất; các token được chấp thuận gần như "miễn phí" về mặt chi phí tính toán. Vào năm 2026, EAGLE-3 là biến thể chuẩn mực cấp sản xuất — phương pháp này huấn luyện một đầu nháp (draft head) trực tiếp trên các trạng thái ẩn (hidden states) của mô hình mục tiêu thay vì trên các token thô, đẩy tỷ lệ chấp thuận alpha ($\alpha$) lên dải 0,6-0,8 trên các tác vụ trò chuyện thông thường. Câu hỏi đúng đắn không phải là "mô hình nháp chạy nhanh đến mức nào" mà là "tỷ lệ alpha trên lưu lượng thực tế của tôi là bao nhiêu?". Nếu alpha tụt xuống dưới ~0,55, giải mã suy đoán sẽ mang lại hiệu ứng tiêu cực ròng ở mức tải đồng thời cao vì mỗi bản nháp bị từ chối sẽ tiêu tốn thêm một lượt lan truyền tiến thứ hai của mô hình mục tiêu. Bài học này hướng dẫn bạn đo đạc alpha trước rồi mới bật cờ cấu hình sau.

**Type:** Learn
**Languages:** Python (stdlib, trình mô phỏng tỷ lệ chấp thuận)
**Prerequisites:** Phase 17 · 04 (Kiến Trúc Nhân Phục Vụ / Serving Engine Internals), Phase 10 · 18 (Dự Đoán Đa Token / Multi-Token Prediction)
**Time:** ~60 phút

## Mục tiêu Học tập

- Gọi tên ba thế hệ giải mã suy đoán và giải thích những cải tiến của EAGLE-3 so với EAGLE-2 và so với mô hình nháp độc lập truyền thống.
- Định nghĩa tỷ lệ chấp thuận alpha, tính toán tốc độ tăng tốc kỳ vọng dựa trên alpha và K (độ dài bản nháp), và xác định ngưỡng alpha hòa vốn cho mức tải đồng thời mục tiêu của bạn.
- Giải thích lý do tại sao giải mã suy đoán là tính năng tùy chọn tham gia (opt-in, không bật mặc định) trong vLLM năm 2026 và tại sao việc bật nó mà không đo đạc alpha là một phản mẫu (anti-pattern) trong môi trường sản xuất.
- Thiết lập một kế hoạch đo đạc chuẩn xác: chọn bài kiểm chuẩn (benchmark) nào, phân phối câu lệnh (prompt distribution) nào, điểm tải đồng thời nào và chỉ số nào dùng làm điều kiện kiểm soát (gating metric).

## Vấn đề Thực tế

Giai đoạn giải mã (decode) bị giới hạn bởi băng thông bộ nhớ (memory-bound). Trên một card H100 chạy mô hình Llama 3.3 70B FP8, mỗi bước giải mã phải đọc toàn bộ ~70 GB trọng số từ bộ nhớ HBM chỉ để sinh ra một token duy nhất cho mỗi chuỗi. Năng lực tính toán của GPU gần như nhàn rỗi trong suốt quá trình giải mã — điểm nghẽn nằm ở băng thông bộ nhớ HBM, chứ không phải ở thông lượng nhân ma trận (matmul throughput).

Giải mã suy đoán khai thác triệt để khoảng trống này. Sinh trước K token ứng viên bằng một mô hình nháp chi phí thấp, sau đó yêu cầu mô hình mục tiêu kiểm định đồng thời toàn bộ K token trong một lượt lan truyền tiến duy nhất. Mỗi token được kiểm định thành công coi như được sinh ra hoàn toàn "miễn phí" (được bù đắp vào một lượt tính toán cho lô K token mà mô hình mục tiêu đằng nào cũng phải thực hiện).

Cách tiếp cận mô hình nháp cổ điển sử dụng một mô hình nhỏ hơn cùng dòng kiến trúc (ví dụ: dùng Llama 3.2 1B làm bản nháp cho Llama 3.3 70B). Phương pháp này hoạt động được nhưng tỷ lệ chấp thuận khá khiêm tốn — phân phối xác suất của mô hình nhỏ bị lệch đáng kể so với mô hình mục tiêu. EAGLE, kế tiếp là EAGLE-2, và sau đó là EAGLE-3 huấn luyện một đầu nháp nhẹ trực tiếp trên các trạng thái nội bộ của mô hình mục tiêu, giúp phân phối của bản nháp bám sát mô hình mục tiêu chặt chẽ hơn nhiều. Đó là lý do tại sao alpha tăng vọt từ 0,4 (với mô hình nháp độc lập) lên 0,6-0,8 với EAGLE-3.

Điểm mấu chốt: EAGLE-3 là tính năng tùy chọn (opt-in) trong vLLM năm 2026. Bạn bắt buộc phải cấu hình `speculative_config` một cách rõ ràng. Không có cờ lệnh này, hệ thống sẽ không có bất kỳ sự tăng tốc nào. Những đội ngũ vội vã bật cờ này mà không đo lường alpha trên lưu lượng thực tế thường nhận thấy độ trễ phần đuôi (tail latency) trở nên tồi tệ hơn, chứ không hề cải thiện.

## Khái niệm

### Lợi ích thực tế mà giải mã suy đoán mang lại

Nếu không dùng giải mã suy đoán, chi phí cho mỗi token là một lượt lan truyền tiến của mô hình mục tiêu. Khi có giải mã suy đoán với độ dài bản nháp K và tỷ lệ chấp thuận alpha, số token kỳ vọng sinh ra sau mỗi lượt chuyển tiếp mục tiêu là `1 + K * alpha`. Tốc độ tăng tốc đạt mức `(1 + K * alpha) / (1 + epsilon)`, trong đó epsilon ($\epsilon$) là phụ phí tính toán của việc tạo bản nháp cộng với bước kiểm định. Với K=5, alpha=0,7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Con số trong thế giới thực thường dao động quanh mức 2-3x vì alpha hiếm khi đạt mức lý tưởng trên lưu lượng sản xuất và epsilon có xu hướng tăng khi kích thước lô lớn.

### Tại sao alpha là chỉ số duy nhất mang tính quyết định

Các token bị từ chối không tự động biến mất — chúng buộc hệ thống phải chạy một lượt lan truyền tiến mục tiêu thứ hai cho chính token đầu tiên bị từ chối đó. Trên một khối lượng công việc mà alpha tụt xuống 0,4, bạn vừa phải gánh phụ phí tạo nháp, vừa chịu chi phí kiểm định, lại vừa phải thực hiện lượt tính toán lại (re-roll). Ở mức tải đồng thời cao (ví dụ 256 yêu cầu đồng thời), kích thước lô giải mã đã đủ lớn để khoảng cách băng thông bộ nhớ giữa "chỉ chạy mô hình mục tiêu" và "chạy mục tiêu kèm kiểm định" bị thu hẹp đáng kể. Dưới ngưỡng alpha 0,55 trên hầu hết phần cứng năm 2026, giải mã suy đoán mang lại hiệu quả âm ròng (làm chậm hệ thống).

Alpha biến thiên mạnh mẽ theo từng loại khối lượng công việc. Trên các đoạn hội thoại chung kiểu ShareGPT, EAGLE-3 được huấn luyện trên ShareGPT đạt alpha 0,6-0,8. Nhưng trên lưu lượng chuyên biệt theo miền (mã nguồn lập trình, y tế, pháp lý), đầu nháp được huấn luyện trên dữ liệu chung sẽ tụt dốc xuống còn 0,4-0,6. Việc huấn luyện một đầu nháp chuyên biệt cho miền dữ liệu sẽ khôi phục lại alpha — đây là một tác vụ huấn luyện nhẹ và nhanh hơn rất nhiều so với việc tinh chỉnh mô hình mục tiêu.

### Toàn cảnh các thế hệ EAGLE

- **Mô hình nháp cổ điển (Classic draft model)**: Mô hình nhỏ cùng dòng kiến trúc. Alpha đạt 0,3-0,5. Hạ tầng đơn giản — nạp hai mô hình riêng biệt vào bộ nhớ, mô hình nháp chạy K lượt chuyển tiếp cho mỗi lượt chuyển tiếp của mục tiêu.
- **EAGLE-1 (2024)**: Một đầu nháp đơn lẻ được huấn luyện trên trạng thái ẩn (tầng cuối cùng) của mô hình mục tiêu. Alpha đạt ~0,5-0,6. Phụ phí tham số rất nhỏ bổ sung vào mô hình mục tiêu.
- **EAGLE-2 (2025)**: Độ dài bản nháp thích ứng và bản nháp dạng cây (kiểm định đồng thời nhiều nhánh trong một lượt duyệt của mô hình mục tiêu). Alpha đạt ~0,6-0,7. Bộ lập lịch bản nháp phức tạp hơn.
- **EAGLE-3 (2025-2026)**: Đầu nháp được huấn luyện trên nhiều tầng ẩn của mô hình mục tiêu (không chỉ tầng cuối), độ căn chỉnh phân phối cao hơn vượt bậc. Alpha đạt ~0,6-0,8 trên các tác vụ trò chuyện thông thường.

### Quy trình triển khai sản xuất năm 2026

1. Triển khai mô hình mục tiêu ở dạng nguyên bản. Đo đạc các chỉ số đường cơ sở: TTFT, ITL và thông lượng ở mức tải đồng thời mục tiêu.
2. Bật bản nháp EAGLE-3 thông qua tham số `speculative_config` của vLLM. Chạy lại bài kiểm chuẩn.
3. Ghi lại nhật ký tỷ lệ chấp thuận alpha. Bộ lập lịch vLLM V1 báo cáo chỉ số này dưới tên `spec_decode_metrics.accepted_tokens_per_request`. Chia con số này cho độ dài bản nháp yêu cầu để tính ra alpha.
4. Nếu alpha < 0,55 trên phân phối lưu lượng sản xuất, hãy tắt giải mã suy đoán hoặc huấn luyện thêm một bản nháp EAGLE-3 chuyên biệt cho miền dữ liệu.
5. Kiểm thử lại ở mức tải đồng thời môi trường sản xuất. Đảm bảo chắc chắn rằng P99 ITL không bị suy giảm.

### Cạm bẫy sản xuất: Phần đuôi P99

Giá trị trung bình của ITL thường giảm khi bật giải mã suy đoán. Tuy nhiên, P99 hoàn toàn có thể trở nên tồi tệ hơn nếu bạn không tinh chỉnh kỹ. Các bản nháp bị từ chối sẽ kích hoạt chuỗi xử lý hai lượt (tạo nháp + kiểm định thất bại + tính toán lại). Khi lô tính toán bị đầy tải, hai lượt xử lý này buộc phải xếp hàng tuần tự. Hãy luôn giám sát P99 ITL thay vì chỉ nhìn vào P50.

### Nơi EAGLE-3 đã được triển khai rộng rãi

Google đã đưa giải mã suy đoán vào tính năng AI Overviews từ năm 2025 (giữ nguyên chất lượng đầu ra, rút ngắn thời gian phản hồi). vLLM V1 cung cấp `speculative_config` làm giao diện chuẩn hóa có tài liệu chính thức, và ma trận tính năng của họ xác nhận giải mã suy đoán hoàn toàn tương thích với tiền điền phân đoạn (chunked prefill). SGLang cũng hỗ trợ EAGLE-3 như một lộ trình tạo nháp khuyến nghị hàng đầu cho các khối lượng công việc dùng nhiều tiền tố lặp lại.

### Công thức hòa vốn trong một dòng

Tốc độ tăng tốc kỳ vọng: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Thiết lập `S = 1` để tìm alpha hòa vốn: `alpha_breakeven = verify_overhead / K`. Với phụ phí kiểm định thông thường ~0,15 và K=5: `alpha_breakeven = 0.03`. Nhưng đó chỉ là công thức toán giải mã thô trên một chuỗi đơn lẻ. Ở mức tải đồng thời cao, phụ phí kiểm định tăng lên và việc gom lô giải mã vốn dĩ đã giúp phân bổ chi phí đọc bộ nhớ giữa các chuỗi, do đó ngưỡng `alpha_breakeven` thực tế trong vận hành sẽ leo thang lên khoảng ~0,45-0,55.

### Khi nào không nên dùng giải mã suy đoán

- Tác vụ sinh theo lô đơn lẻ ngoại tuyến (Batch-1 offline) nơi độ trễ không phải mối bận tâm. Hãy dùng mô hình mục tiêu nguyên bản.
- Độ dài đầu ra cực ngắn (dưới 50 token). Phụ phí tạo nháp và kiểm định sẽ hoàn toàn lấn át lợi ích thu được.
- Các miền dữ liệu chuyên môn sâu mà không có đầu nháp được huấn luyện riêng. Tỷ lệ alpha khi đó quá thấp.
- Giả định rằng mọi cặp tính năng đều có thể phối hợp với nhau. Luôn kiểm tra ma trận tương thích vLLM cho phiên bản của bạn: phiên bản v0.18.0 đánh dấu giải mã suy đoán tương thích tốt với tiền điền phân đoạn.

```figure
mx-speculative-tree
```

## Ứng Dụng Thực Tế (Use It)

Tệp `code/main.py` mô phỏng vòng lặp giải mã khi có và không có giải mã suy đoán trên một dải giá trị alpha và độ dài nháp K khác nhau. Chương trình in ra giá trị alpha hòa vốn, tốc độ tăng tốc đo được và hành vi độ trễ phần đuôi. Hãy chạy thử trên nhiều tổ hợp (alpha, K) để xác định chính xác thời điểm giải mã suy đoán không còn mang lại lợi ích kinh tế.

## Bàn Giao Thành Phẩm (Ship It)

Bài học này tạo ra tệp artifact `outputs/skill-eagle3-rollout.md`. Dựa trên mô hình mục tiêu, mô tả phân phối lưu lượng và mục tiêu tải đồng thời, kỹ năng này sẽ thiết lập một kế hoạch triển khai EAGLE-3 theo từng giai đoạn — đo kiểm đường cơ sở, bật cấu hình, giám sát alpha, chặn điều kiện tại ngưỡng alpha >= 0,55, và theo dõi chặt chẽ chỉ số P99 ITL.

## Bài Tập Thực Hành

1. Chạy `code/main.py`. Với K=5, bạn cần giá trị alpha bằng bao nhiêu để đạt tốc độ tăng tốc 2x? Để đạt 3x? Kết quả này nhạy cảm như thế nào đối với phụ phí `verify_overhead`?
2. Giả sử lưu lượng sản xuất thực tế phân chia thành 70% trò chuyện thông thường và 30% viết mã nguồn. Trò chuyện thông thường đạt alpha 0,7 với EAGLE-3 huấn luyện trên ShareGPT; mã nguồn chỉ đạt alpha 0,4. Tỷ lệ alpha hỗn hợp là bao nhiêu và giải mã suy đoán có mang lại giá trị dương ròng hay không?
3. Đọc tài liệu `speculative_config` của vLLM. Gọi tên ba chế độ hoạt động (draft model, EAGLE, N-gram) và kiểm tra xem mỗi chế độ có thể phối hợp với những tính năng nào trong phiên bản vLLM của bạn.
4. Bạn nhận thấy ITL trung bình giảm 25% sau khi kích hoạt EAGLE-3 nhưng P99 ITL lại tăng thêm 15%. Hãy chẩn đoán nguyên nhân và đề xuất phương án khắc phục.
5. Tính toán chi phí bộ nhớ của đầu nháp EAGLE-3 đối với mô hình Llama 3.3 70B. So sánh con số này với việc chạy mô hình Llama 3.2 1B làm bản nháp cổ điển.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật thực tế |
|-----------|----------------------|--------------------------|
| Giải mã suy đoán (Speculative decoding) | "Nháp kèm kiểm định" | Đề xuất K token bằng một mô hình giá rẻ, kiểm định toàn bộ K token trong một lượt lan truyền của mục tiêu |
| Tỷ lệ chấp thuận alpha (Acceptance rate alpha) | "Tỷ lệ nhận nháp" | Tỷ lệ phần trăm các token nháp được mô hình mục tiêu chấp thuận; chỉ số cốt lõi duy nhất |
| Độ dài bản nháp K (Draft length K) | "Số token nháp K" | Số lượng token mà mô hình nháp đề xuất cho mỗi lượt kiểm định mục tiêu; thường từ 4-8 |
| Phụ phí kiểm định epsilon (Verify overhead epsilon) | "Phụ phí suy đoán" | Chi phí phát sinh để kiểm định và tính toán lại so với một lượt chạy mục tiêu thông thường; tăng theo kích thước lô |
| EAGLE-3 | "Bản EAGLE mới nhất" | Biến thể năm 2025-2026; huấn luyện đầu nháp trên nhiều tầng ẩn của mô hình mục tiêu; đạt alpha 0,6-0,8 trên hội thoại chung |
| `speculative_config` | "Cấu hình suy đoán vLLM" | Cờ bật tùy chọn rõ ràng trong vLLM V1; không thiết lập đồng nghĩa với không có tăng tốc |
| Suy đoán N-gram (N-gram spec decode) | "Nháp N-gram" | Tạo bản nháp trực tiếp trên GPU bằng cách tra cứu N-gram trong prompt; tương thích với tiền điền phân đoạn |
| Alpha hòa vốn (Break-even alpha) | "Alpha ngưỡng tối thiểu" | Mức alpha mà tại đó giải mã suy đoán không đem lại bất kỳ sự tăng tốc nào; cần theo dõi ở tải sản xuất |
| Xử lý hai lượt khi từ chối (Rejected-draft two-pass) | "Chi phí quay số lại" | Phải chạy hai lượt chuyển tiếp mục tiêu khi bản nháp bị từ chối; nguyên nhân gây kéo dài đuôi P99 |

## Tài Liệu Đọc Thêm

- [Tài liệu vLLM — Giải mã Suy đoán](https://docs.vllm.ai/en/latest/features/spec_decode/) — Tài liệu chuẩn về `speculative_config` và tính tương thích với tiền điền phân đoạn trong V1.
- [API Cấu hình Suy đoán của vLLM](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — Tập hợp các trường thuộc tính chi tiết.
- [Bài báo EAGLE (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — Công thức gốc của đầu nháp EAGLE.
- [Bài báo EAGLE-2 (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — Bản nháp thích ứng và kiến trúc cây suy đoán.
- [Báo cáo Kỹ thuật UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — Hệ thống LLM hiệu năng cao kết hợp giải mã suy đoán.
- [BentoML — Tối Ưu Giải Mã Suy Đoán](https://bentoml.com/llm/inference-optimization/speculative-decoding) — Danh mục kiểm tra triển khai môi trường sản xuất.
