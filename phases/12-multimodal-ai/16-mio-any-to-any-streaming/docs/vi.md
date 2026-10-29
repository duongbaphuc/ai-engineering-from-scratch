# MIO và Các Mô hình Đa phương thức Truyền trực tiếp Giữa Mọi Phương thức

> GPT-4o đã trình diễn một sản phẩm mà hầu hết các mô hình mở chưa thể tái lập: một tác tử có thể nghe giọng nói, nhìn thấy video và phản hồi lại bằng giọng nói trong thời gian thực. Câu trả lời của hệ sinh thái nguồn mở vào cuối năm 2024 là MIO (Wang et al., tháng 9 năm 2024). MIO token hóa văn bản, hình ảnh, tiếng nói và âm nhạc, huấn luyện một transformer nhân quả duy nhất trên các chuỗi xen kẽ (interleaved sequences), và tạo ra khả năng sinh từ mọi phương thức sang mọi phương thức (any-to-any). AnyGPT (Zhan et al., tháng 2 năm 2024) là phiên bản chứng minh khái niệm (proof of concept); MIO là phiên bản mở rộng quy mô (scale-up); còn Unified-IO 2 (Allen AI, tháng 12 năm 2023) là mô hình họ hàng bổ sung khả năng định vị thị giác + hành động (vision + action grounding). Bài học này sẽ phân tích mô hình any-to-any — bốn bộ token hóa, một transformer, và quy trình giải mã thân thiện với truyền trực tiếp (streaming decode).

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ phân bổ token cho bốn phương thức + vòng lặp giải mã truyền trực tiếp)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 11 (Chameleon), Giai đoạn 6 (Tiếng nói và âm thanh - Speech and Audio)  
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Thiết kế một không gian từ vựng dùng chung (shared vocabulary) chứa các token văn bản, hình ảnh, tiếng nói và âm nhạc mà không bị xung đột chỉ số (collisions).
- So sánh SEED-Tokenizer (cho hình ảnh) và SpeechTokenizer residual-VQ (cho tiếng nói) về sự đánh đổi giữa mức độ nén và chất lượng tái tạo.
- Giải thích chương trình giảng dạy 4 giai đoạn (four-stage curriculum) giúp xây dựng năng lực sinh any-to-any.
- Nêu tên ba công thức any-to-any mã nguồn mở và những đánh đổi chính của chúng: MIO, AnyGPT, Unified-IO 2.

## Vấn đề

Một mô hình đa phương thức hợp nhất rất dễ tuyên bố trên lý thuyết nhưng cực kỳ khó xây dựng ở quy mô lớn. Hầu hết các hệ thống "any-to-any" trước năm 2024 đều hoạt động theo dạng đường ống (pipeline): mô hình thị giác → biểu diễn văn bản → mô hình tiếng nói → âm thanh. Mỗi bước chuyển tiếp đều làm thất thoát thông tin, tăng độ trễ (latency) và gây phức tạp cho quá trình huấn luyện. Video trình diễn của GPT-4o cho thấy một giải pháp thay thế dựa trên một mô hình duy nhất với độ phản hồi dưới một giây; các hệ thống nguồn mở khi đó bị tụt lại nhiều tháng.

Những thách thức kỹ thuật bao gồm:

- Phải có sẵn các bộ token hóa cho mọi phương thức, nén đủ tốt để không làm mất chi tiết khi tái tạo, và sinh ra token ở tốc độ mà transformer có thể xử lý được.
- Một từ vựng thống nhất phải phân bổ không gian cho văn bản (32k+), hình ảnh (8k+), tiếng nói (4k+), âm nhạc (8k+). Tối thiểu phải có trên năm mươi nghìn mục từ.
- Dữ liệu huấn luyện phải bao phủ mọi cặp đầu vào - đầu ra (văn bản→ảnh, ảnh→tiếng nói, tiếng nói→ảnh, v.v.) hoặc mô hình phải có khả năng tự kết hợp.
- Quá trình suy luận phải truyền trực tiếp (stream) các token đầu ra đủ nhanh để đạt độ trễ đàm thoại (<500ms thời gian đến byte âm thanh đầu tiên - time-to-first-audio-byte).

## Khái niệm

### Bốn bộ token hóa cho bốn phương thức (Four tokenizers for four modalities)

Ngăn xếp token hóa của MIO:

- Văn bản (Text): BPE tiêu chuẩn, kích thước từ vựng ~32000.
- Hình ảnh (Image): SEED-Tokenizer (2023), một bộ token hóa ảnh rời rạc với sổ mã (codebook) 8192 mục, xuất ra 32 token cho mỗi hình ảnh.
- Tiếng nói (Speech): SpeechTokenizer residual-VQ (2023), mã hóa dạng sóng 16kHz thành 8 sổ mã phân cấp; cấp độ đầu tiên là nội dung thô, các cấp độ sau bổ sung ngữ điệu (prosody) và đặc trưng người nói (timbre). MIO giữ lại cấp độ nội dung cùng ba cấp độ âm sắc đầu tiên, mỗi cấp 1024 mã, tổng cộng là 4096 ID tiếng nói.
- Âm nhạc (Music): residual-VQ tương tự (họ MusicGen / Encodec của Meta), 4-8 sổ mã. Bản thân MIO không đi kèm bộ token hóa âm nhạc (phương thức thứ tư của nó là video, biểu diễn dưới dạng các khung hình đi qua bộ token hóa ảnh); mô hình đồ chơi của bài học này bổ sung âm nhạc để minh họa cách một phương thức mới đăng ký dải ID.

Mỗi phương thức sinh ra các token số nguyên. Các token được cấp các dải ID không giao nhau trong từ vựng dùng chung:

```text
text:   0..31999
image:  32000..40191  (8192 token hình ảnh)
speech: 40192..44287  (4 sổ mã x 1024 token tiếng nói)
music:  44288..52479  (8192 token âm nhạc, phần mở rộng của bài thực hành)
sep:    52480..52485  (<image>, <speech>, <music>, </...>)
```

Tổng cộng: từ vựng ~52k. Lớp nhúng đầu vào (input embedding) và lớp chiếu đầu ra (output projection) bao phủ toàn bộ không gian này.

### Giải mã truyền trực tiếp (Streaming decode)

Tác vụ sinh tiếng nói sử dụng lượng tử hóa vector phần dư (residual-VQ). Transformer dự đoán các token tiếng nói cơ sở (lớp 0); một bộ lượng tử hóa phần dư giải mã song song sẽ dự đoán các lớp tiếp theo. Mỗi token lớp 0 tương ứng với khoảng 50ms âm thanh ở tần số 16kHz.

Quy trình truyền trực tiếp:

1. Người dùng nói vào micro; bộ token hóa âm thanh thời gian thực xuất ra các token tiếng nói mỗi 50ms.
2. MIO tiếp nhận các token ngay khi chúng tới (prefill prompt + truyền xuôi gia tăng).
3. Các token đầu ra được truyền theo dòng ngay khi được sinh; một bộ giải mã tiếng nói song song chuyển đổi chúng thành các mẫu âm thanh với độ trễ ~50-150ms.
4. Thời gian đến byte âm thanh đầu tiên (Time-to-first-audio-byte): bài báo MIO không công bố con số độ trễ cụ thể. Bảng phân bổ ngân sách dưới đây ước tính thiết kế này đạt tối ưu khoảng 300ms, so với mức ~250ms của GPT-4o.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612) và Moshi (arXiv:2410.00037) là những thiết kế LLM tiếng nói truyền trực tiếp bổ trợ. Đặc biệt, Moshi đạt độ trễ khứ hồi 160ms trên một GPU đơn lẻ.

### Chương trình huấn luyện bốn giai đoạn (Four-stage curriculum)

Lộ trình huấn luyện của MIO:

1. Giai đoạn 1 — Căn chỉnh (Alignment): Tập dữ liệu quy mô lớn gồm các cặp phương thức: văn bản-hình ảnh, văn bản-tiếng nói, văn bản-âm nhạc. Mỗi cặp sử dụng phân đoạn từ vựng token riêng của mình. Huấn luyện từ vựng dùng chung.
2. Giai đoạn 2 — Xen kẽ (Interleaved): Các tài liệu đa phương thức xen kẽ (bài viết blog có hình ảnh + video, podcast kèm bản ghi lời thoại, v.v.). Huấn luyện khả năng xử lý ngữ cảnh liên phương thức.
3. Giai đoạn 3 — Tăng cường tiếng nói (Speech-enhanced): Bổ sung thêm dữ liệu âm thanh để nâng cao chất lượng giọng nói mà không làm suy giảm năng lực xử lý văn bản.
4. Giai đoạn 4 — SFT (Tinh chỉnh có giám sát): Tinh chỉnh theo chỉ dẫn trên nhiều phương thức: VQA, tạo chú thích ảnh, tường thuật, đối thoại giọng nói - giọng nói.

Việc bỏ qua bất kỳ giai đoạn nào sẽ làm suy giảm năng lực tương ứng: bỏ qua Giai đoạn 2 mô hình sẽ mất khả năng xử lý ngữ cảnh chéo phương thức; bỏ qua Giai đoạn 3 giọng nói sinh ra sẽ có chất lượng kém.

### Chuỗi suy nghĩ thị giác (Chain-of-visual-thought)

MIO giới thiệu khái niệm chuỗi suy nghĩ thị giác: mô hình xuất ra các token hình ảnh trung gian như một bước lập luận. Đối với câu hỏi "con mèo có đang trèo cây không?", mô hình:

1. Xuất các token `<image>` phác thảo lại khung cảnh (từ ảnh đầu vào hoặc một bản vẽ nháp).
2. Xuất văn bản phân tích bản vẽ nháp đó.
3. Xuất ra câu trả lời cuối cùng.

Hình ảnh trung gian được kết xuất đóng vai trò như một bảng nháp (scratchpad). Điểm số trên các bài kiểm tra suy luận không gian được cải thiện rõ rệt. Ý tưởng này tương tự như chuỗi suy nghĩ (chain-of-thought) trong suy luận văn bản.

### Các đối thủ cạnh tranh trong dòng any-to-any (Competitors in any-to-any)

- AnyGPT (arXiv:2402.12226): 4 phương thức (văn bản, hình ảnh, tiếng nói, âm nhạc), thiết kế tương tự.
- Unified-IO 2 (arXiv:2312.17172): bổ sung thêm đầu ra hành động thị giác, độ sâu (depth), pháp tuyến bề mặt (normals). Đa dạng tác vụ hơn nhưng quy mô nhỏ hơn.
- NExT-GPT (arXiv:2309.05519): LLM kết hợp với các bộ giải mã khuếch tán chuyên biệt theo phương thức. Không phải là giải pháp một mô hình duy nhất.
- CoDi (arXiv:2305.11846): khuếch tán có thể hợp thành; any-to-any thông qua biến tiềm ẩn dùng chung.

MIO là mô hình tiếp cận gần nhất với kiến trúc any-to-any thuần token. AnyGPT là tiền thân mang tính khái niệm của nó.

### Ngân sách độ trễ (Latency budget)

Đối với một sản phẩm đàm thoại, độ trễ của từng thành phần đều mang tính quyết định:

- Từ micro đến token âm thanh: ~50ms.
- Prefill (token âm thanh + lịch sử hội thoại): ~100ms trên mô hình 8B.
- Token đầu ra đầu tiên: ~50ms.
- Residual-VQ song song + bộ giải mã tiếng nói: ~100-150ms.

Tổng thời gian đến byte âm thanh đầu tiên: tối thiểu khoảng ~300ms. GPT-4o công bố khoảng ~250ms. Moshi công bố 160ms. Do cả MIO và AnyGPT đều không công bố số liệu đo đạc thực tế, hãy xem ngân sách này là một ước lượng kỹ thuật thay vì một thước đo chuẩn.

### Vì sao any-to-any vẫn là bài toán khó (Why any-to-any stays hard)

Ngay cả vào năm 2026, các mô hình any-to-any mã nguồn mở vẫn còn khoảng cách so với các hệ thống đóng ở hai khía cạnh:

- Chất lượng tiếng nói: Bộ token hóa residual-VQ gây mất mát dữ liệu; giọng đàm thoại vẫn mang âm hưởng máy móc khi so sánh với các giọng nói đạt chuẩn ElevenLabs.
- Suy luận chéo phương thức: Yêu cầu mô hình "hát một bài về những gì bạn đang nhìn thấy" vẫn thường xuyên thất bại so với các tác vụ thuần thị giác.

Đây vẫn là những bài toán nghiên cứu mở. Qwen3-Omni (Bài học 12.20) là nỗ lực nguồn mở tiên tiến nhất vào năm 2025.

````figure
any-to-any-stream
````

## Thực hành (Use It)

`code/main.py`:

- Định nghĩa và in ra cấu trúc phân bổ từ vựng cho bốn phương thức.
- Định tuyến danh sách các đầu vào đa phương thức (văn bản, hình ảnh, đoạn âm thanh, âm nhạc) qua bộ định tuyến token hóa.
- Mô phỏng quá trình giải mã truyền trực tiếp cho phản hồi chuyển văn bản thành tiếng nói (text-to-speech) kèm theo tính toán độ trễ.
- Tính toán thời gian dự kiến đến byte âm thanh đầu tiên dựa trên độ trễ của bộ mã hóa, prefill và bộ giải mã.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-any-to-any-pipeline-auditor.md`. Cho trước bản đặc tả sản phẩm đàm thoại (các phương thức đầu vào, phương thức đầu ra, mục tiêu độ trễ), kỹ năng này sẽ kiểm tra các lựa chọn thiết kế thuộc họ MIO và lập bảng ngân sách độ trễ chi tiết.

## Bài tập (Exercises)

1. Sản phẩm của bạn nhận đầu vào là tiếng nói và trả về đầu ra là tiếng nói. Mục tiêu ngân sách độ trễ từ đầu đến cuối là bao nhiêu? Liệt kê các thành phần tiêu tốn thời gian.

2. SpeechTokenizer residual-VQ sử dụng 8 sổ mã. Hãy giải thích vì sao việc giải mã song song các cấp độ phần dư lại cần thiết (so với giải mã tuần tự) và nó giúp tiết kiệm độ trễ như thế nào.

3. Từ vựng của bạn gồm 32k văn bản + 8k hình ảnh + 4k tiếng nói. Bổ sung thêm 8k âm nhạc và ~10 ký tự phân cách. Chi phí tham số của ma trận nhúng (embedding matrix) là bao nhiêu với chiều ẩn 4096?

4. Chuỗi suy nghĩ thị giác xuất ra một hình ảnh trung gian. Những dạng câu hỏi nào sẽ được hưởng lợi? Những dạng câu hỏi nào sẽ bị ảnh hưởng tiêu cực bởi việc phát sinh thêm token?

5. Đọc bài báo Moshi (arXiv:2410.00037). Hãy mô tả kỹ thuật "độc thoại nội tâm" (inner monologue) của nó và so sánh với chuỗi suy nghĩ thị giác của MIO.

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Any-to-any (Mọi phương thức sang mọi phương thức) | "Đa phương thức vào/ra" | Một mô hình duy nhất tiếp nhận và xuất văn bản, hình ảnh, tiếng nói và âm nhạc theo bất kỳ hướng nào |
| Residual-VQ (Lượng tử hóa vector phần dư) | "Ngăn xếp token hóa tiếng nói" | Token hóa đa sổ mã trong đó mỗi lớp bổ sung thêm thông tin; lớp cơ sở là nội dung, các lớp sau là ngữ điệu |
| SEED-Tokenizer | "Mã hình ảnh" | Bộ token hóa hình ảnh rời rạc với sổ mã 8192 mục và 32 token trên mỗi ảnh, được MIO sử dụng |
| Chain-of-visual-thought (Chuỗi suy nghĩ thị giác) | "Bảng nháp thị giác" | Mô hình sinh ra một hình ảnh trung gian như một bước lập luận trước khi đưa ra câu trả lời cuối cùng |
| Time-to-first-audio-byte (Thời gian đến byte âm thanh đầu tiên) | "TTFAB" | Độ trễ từ khi người dùng dứt lời đến khi phát ra byte âm thanh đầu tiên; <500ms để tạo cảm giác đối thoại tự nhiên |
| Four-stage curriculum (Chương trình huấn luyện bốn giai đoạn) | "Công thức huấn luyện" | Căn chỉnh (Alignment) -> xen kẽ (interleaved) -> tăng cường tiếng nói (speech-enhanced) -> SFT, theo đúng thứ tự đó |

## Tài liệu tham khảo (Further Reading)

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
