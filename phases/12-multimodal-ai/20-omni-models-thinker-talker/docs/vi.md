# Mô hình toàn năng (Omni Models): Qwen2.5-Omni và kiến trúc phân tách Tư duy - Diễn thuyết

> Bản demo sản phẩm GPT-4o vào tháng 5 năm 2024 đã tạo ra sự đột phá mạnh mẽ không chỉ vì mô hình nền tảng bên dưới mà chủ yếu nhờ trải nghiệm sản phẩm mang tính cách mạng — một giao diện giọng nói nơi bạn trò chuyện tự nhiên, mô hình quan sát trực tiếp qua camera và phản hồi bằng giọng nói trong vòng dưới 250ms. Hệ sinh thái mã nguồn mở đã dành trọn giai đoạn cuối năm 2024 và năm 2025 để chạy đua đạt tới trải nghiệm sản phẩm đó. Qwen2.5-Omni (tháng 3 năm 2025) là kiến trúc tham chiếu mã nguồn mở tiêu biểu: kết hợp giữa bộ Tư duy (Thinker - một transformer sinh văn bản quy mô lớn) và bộ Diễn thuyết (Talker - một transformer sinh tiếng nói song song), được liên kết qua luồng truyền trực tiếp (streaming) các token tiếng nói. Mini-Omni đã tinh gọn kiến trúc này, Moshi bắt kịp về mặt độ trễ và GLM-4-Voice đã mở rộng năng lực sang tiếng Trung. Bài học này phân tích sâu kiến trúc Thinker-Talker và bài toán phân bổ độ trễ (latency budget) giúp hiện thực hóa các cuộc hội thoại thời gian thực.

**Type:** Build
**Languages:** Python (stdlib, bộ mô phỏng độ trễ luồng xử lý streaming + vòng lặp VAD)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 phút

## Mục tiêu học tập

- Tách biệt luồng suy luận thành hai thành phần: Thinker (suy luận văn bản) và Talker (tổng hợp tiếng nói), đồng thời giải thích lý do cơ chế truyền phát trực tiếp song song (parallel streaming) đạt hiệu quả cao.
- Tính toán hạn mức thời gian đến byte âm thanh đầu tiên (TTFAB - time-to-first-audio-byte) cho một phiên tương tác hội thoại, chi tiết hóa theo từng thành phần.
- Mô tả cơ chế mã hóa vị trí đa phương thức căn chỉnh theo thời gian TMRoPE (Time-aligned Multimodal RoPE) trên các luồng thị giác, âm thanh và văn bản bên trong Thinker.
- Nêu rõ hai mẫu hình hội thoại thời gian thực: bán song công (half-duplex) và song công toàn phần (full-duplex), cùng cách thức cơ chế phát hiện hoạt động giọng nói (VAD - Voice Activity Detection) điều phối lượt nói trong cơ chế bán song công.

## Vấn đề đặt ra

Một trợ lý giọng nói thời gian thực phải thực hiện khối lượng công việc rất lớn với tốc độ cực nhanh:

1. **Lắng nghe người dùng:** Token hóa tiếng nói theo thời gian thực, kết hợp phát hiện hoạt động giọng nói (VAD) để xác định chính xác thời điểm người dùng dứt lời.
2. **Quan sát trực quan (tùy chọn):** Luồng dữ liệu camera với tần số 2-4 FPS được truyền trực tiếp vào Thinker song song với âm thanh.
3. **Suy nghĩ (Tư duy):** Soạn thảo câu phản hồi dựa trên ngữ cảnh lịch sử hội thoại.
4. **Nói (Diễn thuyết):** Tổng hợp các token âm thanh, giải mã ngược về dạng sóng (waveform) và truyền trực tiếp tới loa của người dùng.

Mỗi bước đều làm gia tăng độ trễ. Để tạo cảm giác hội thoại tự nhiên của con người, toàn bộ chu trình phản hồi (round-trip) phải dưới 500ms — dưới ngưỡng này, bộ não con người gần như không nhận thức được độ trễ. GPT-4o công bố độ trễ khoảng 250ms, Moshi đạt khoảng 160ms, và Qwen2.5-Omni dao động trong khoảng 350-500ms.

Do đó, mọi thành phần trong hệ thống bắt buộc phải truyền trực tiếp (streaming). Tuyệt đối không thể áp dụng phương pháp "gom đủ toàn bộ lô (batch) rồi mới giải mã".

## Khái niệm cốt lõi

### Kiến trúc phân tách Thinker và Talker

Cấu trúc phân rã của Qwen2.5-Omni:

- **Thinker (Bộ tư duy):** Một transformer sinh văn bản quy mô từ 7B đến 80B tham số. Tiêu thụ các token văn bản, hình ảnh và âm thanh đan xen (interleaved). Đầu ra là chuỗi các token văn bản thể hiện nội dung cần phản hồi.
- **Talker (Bộ diễn thuyết):** Một transformer sinh tiếng nói nhỏ gọn hơn nhiều (200M đến 1B tham số). Tiêu thụ các token văn bản đầu ra từ Thinker kết hợp với các token ngữ cảnh tiếng nói gần nhất. Đầu ra là các token tiếng nói rời rạc (chỉ số residual-VQ).
- **Bộ giải mã tiếng nói (Speech decoder):** Bộ giải mã dạng sóng streaming (thuộc họ SNAC, MoVQGAN) chuyển đổi các token tiếng nói thành các mẫu âm thanh thời gian thực.

Sự phân tách này mang tính quyết định: Thinker cần kích thước tham số lớn để đảm bảo năng lực suy luận sâu sắc. Talker chỉ cần quy mô nhỏ vì nhiệm vụ của nó mang tính cục bộ — chuyển đổi văn bản sang token tiếng nói. Một Talker kích thước quá lớn không làm tăng tính biểu cảm mà chỉ làm hệ thống chậm đi.

Quy trình vận hành song song:

1. Thinker phát ra token văn bản $t_i$.
2. Talker tiếp nhận $t_i$ (thông qua streaming) và lập tức sinh ra các token tiếng nói tương ứng $s_i, s_{i+1}, \ldots, s_{i+k}$.
3. Bộ giải mã tiếng nói nhận các token này ngay khi xuất hiện và tái tạo dạng sóng âm thanh.
4. Đến thời điểm Thinker sinh ra token văn bản $t_{i+3}$, Talker đã hoàn tất phát âm thanh cho chuỗi $t_0, \ldots, t_{i+2}$.

### TMRoPE — Mã hóa vị trí đa phương thức căn chỉnh theo thời gian

Thinker cần hợp nhất các khung hình (video frames, ví dụ: 4 FPS), các khung âm thanh (audio frames, khoảng 50 FPS) và văn bản từ lịch sử hội thoại. Việc sắp xếp tuần tự ngây thơ (tất cả ảnh, rồi đến tất cả âm thanh, sau đó đến văn bản) sẽ làm mất hoàn toàn sự tương quan về mặt thời gian.

TMRoPE gán nhãn thời gian tuyệt đối (absolute timestamps) cho từng token:
- Token thị giác tại thời điểm $t=2,3$s.
- Token âm thanh tại thời điểm $t=2,32$s.
- Token văn bản người dùng nói "dừng lại" tại $t=2,35$s.

Cơ chế RoPE sẽ xoay ma trận chú ý dựa trên nhãn thời gian này; nhờ đó mô hình nhận thức được các sự kiện này diễn ra đồng thời trong thực tế.

Đây chính là nền tảng kỹ thuật để mô hình hiểu được ngữ cảnh như "anh ấy vẫy tay trong lúc nói lời chào" — mô hình xử lý khung hình video và tín hiệu âm thanh tại cùng một thời điểm khái niệm.

### Tổng hợp tiếng nói theo luồng trực tiếp (Streaming Speech Synthesis)

Các token tiếng nói bắt buộc phải được truyền theo luồng. Mini-Omni (Xie & Wu, 2024) đưa ra phương pháp "mô hình ngôn ngữ có thể nghe và nói trong khi đang suy nghĩ ở chế độ streaming": các token đầu ra của Thinker và Talker được đan xen trong cùng một chuỗi tuần tự. Talker kích hoạt ngay khi Thinker chốt token văn bản tiếp theo, không có ranh giới phân tách theo lô.

Moshi (Défossez và cộng sự, tháng 10 năm 2024) là bản triển khai mã nguồn mở nhanh nhất với TTFAB đạt 160ms trên một GPU A100 đơn lẻ. Kiến trúc của Moshi: một transformer 7B duy nhất phát ra các token văn bản và tiếng nói ở các vị trí so le luân phiên, sử dụng một cơ chế "độc thoại nội tâm" (inner monologue) để phân tách luồng suy nghĩ với luồng phát ngôn. Về bản chất, đây là Thinker + Talker được tích hợp vào trong một mô hình duy nhất nhờ kỹ thuật huấn luyện tinh vi.

### Cơ chế VAD và điều phối lượt hội thoại

Module phát hiện hoạt động tiếng nói (VAD) hoạt động ở đầu vào của hệ thống với hai cơ chế vận hành:

- **Bán song công (Half-duplex):** Người dùng nói thì mô hình lắng nghe. Mô hình nói thì người dùng lắng nghe. Sự chuyển giao lượt nói diễn ra rõ ràng thông qua việc phát hiện khoảng lặng (silence detection) bằng VAD (khoảng ~200ms).
- **Song công toàn phần (Full-duplex):** Cả người dùng và mô hình có thể nói cùng lúc. Mô hình có thể đệm lời ("vâng ạ", "uh-huh") hoặc chủ động ngắt lời người dùng. Kỹ thuật này phức tạp hơn rất nhiều, hiện được hỗ trợ tốt bởi Moshi.

Qwen2.5-Omni mặc định hoạt động ở chế độ bán song công với ngưỡng ngắt lời dựa vào khoảng lặng. Chế độ song công toàn phần đòi hỏi tầng ứng dụng phải xử lý phối hợp thêm.

### Qwen3-Omni (Tháng 11 năm 2025)

Phiên bản kế nhiệm với Thinker quy mô Qwen3-80B, Talker lớn hơn và nâng cấp lên TMRoPE-v2. Độ trễ tiến gần sát mốc 250ms của GPT-4o với trọng số mở hoàn toàn. Các bài kiểm tra trên chuẩn OmniBench cho thấy năng lực cạnh tranh sòng phẳng với Gemini 2.0 Live.

### Bảng dự toán độ trễ sản xuất (Production Latency Budget)

Xét một tương tác hội thoại streaming điển hình:

- Micro thu âm $\to$ chuyển thành token âm thanh: 40-80ms.
- Xử lý mồi trước (Prefill prompt + lịch sử): 100-200ms đối với mô hình 7B, tăng mạnh ở mô hình 70B.
- Sinh token văn bản đầu tiên của Thinker: 40ms.
- Talker xử lý token văn bản đầu tiên: 20ms.
- Phát các token tiếng nói đầu tiên: 40ms.
- Giải mã lượng tử hóa vector phần dư (Residual-VQ): 30ms.
- Giải mã dạng sóng âm thanh (Waveform): 50-80ms.

Tổng thời gian TTFAB: **320-490ms** ở mô hình 7B, và **600-900ms** ở mô hình 70B. Chất lượng thông minh vượt trội thường đòi hỏi quy mô từ 70B trở lên; đó là lý do vì sao luôn tồn tại sự đánh đổi giữa độ thông minh và độ trễ.

### Tính toán thông lượng phát sinh Token (Token-rate math)

Với chuẩn âm thanh 16kHz sử dụng 50 token tiếng nói cơ sở mỗi giây (50 Hz), bạn cần 50 token tiếng nói cho mỗi giây âm thanh phát ra. Talker bắt buộc phải đạt tốc độ sinh $\ge 50$ token/giây để không bị giật âm thanh. Với thông lượng xử lý của LLM điển hình đạt từ 30-80 token/giây trên chip H100, một Talker nhỏ (200-300M tham số) hoàn toàn đáp ứng được tốc độ; trong khi một Talker kích thước 7B sẽ lập tức rơi vào tình trạng quá tải và trễ nhịp.

Đó là lý do cốt lõi tại sao cần một mô hình Talker chuyên biệt nhỏ gọn thay vì tận dụng trực tiếp mô hình chính.

```figure
l5-thinker-talker
```

## Sử dụng trong thực tế

Tệp `code/main.py`:

- Mô phỏng chuỗi xử lý Thinker-Talker với tốc độ phát token giả lập.
- Tính toán chỉ số TTFAB tùy biến theo kích thước mô hình và tần số lấy mẫu của micro.
- Trình diễn cơ chế đổi lượt nói bán song công kết hợp ngưỡng phát hiện khoảng lặng VAD.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-omni-streaming-budget.md`. Khi tiếp nhận chỉ tiêu độ trễ TTFAB và tập tính năng mục tiêu (đầu vào camera, song ngữ, song công toàn phần), kỹ năng này tự động lựa chọn giữa Qwen2.5-Omni, Qwen3-Omni, Moshi hoặc Mini-Omni, đồng thời định cỡ cấu hình cho Thinker/Talker.

## Bài tập thực hành

1. Chỉ tiêu TTFAB của bạn là 300ms. Với Thinker 7B và Talker 300M, hãy lập bảng chi tiết độ trễ dự kiến cho từng thành phần trong hệ thống.

2. Qwen2.5-Omni sử dụng TMRoPE. Hãy mô tả dữ liệu mà mô hình nhìn thấy đối với một lệnh tương tác khi người dùng bắt đầu nói tại thời điểm $t=1$s và camera ghi nhận một cử chỉ tay tại thời điểm $t=1,2$s.

3. Hỗ trợ song công toàn phần đòi hỏi mô hình phải phát âm thanh trong lúc vẫn đang lắng nghe. Hãy đề xuất một cấu trúc định dạng dữ liệu huấn luyện để dạy mô hình năng lực này.

4. Đọc Mục 4 trong bài báo nghiên cứu của Moshi. Phân tích cơ chế phân tách "độc thoại nội tâm" (inner monologue) và giải thích tại sao nó tránh được việc phải chia tách rời Thinker và Talker.

5. Tính toán bài toán thông lượng: một Talker phải sinh token với tốc độ nhanh bao nhiêu để bắt kịp giọng nói 16kHz ở mức 50 token tầng cơ sở mỗi giây?

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Thinker | "Bộ não suy luận" | Transformer sinh văn bản quy mô lớn đảm nhiệm nội dung trả lời |
| Talker | "Bộ phận phát âm" | Transformer nhỏ sinh các token tiếng nói rời rạc từ văn bản của Thinker |
| TTFAB | "Ngân sách độ trễ" | Time-to-first-audio-byte: thời gian từ khi người dùng dứt lời đến byte âm thanh đầu tiên phát ra |
| TMRoPE | "RoPE căn chỉnh thời gian" | Cơ chế mã hóa vị trí gán nhãn thời gian tuyệt đối trên thị giác, âm thanh và văn bản |
| Bán song công (Half-duplex) | "Luân phiên lượt nói" | Người dùng và mô hình đổi lượt; phát hiện khoảng lặng VAD để nhận biết dứt lời |
| Song công toàn phần (Full-duplex) | "Đồng thời" | Mô hình có thể vừa nghe vừa nói đồng thời; có khả năng chêm lời phản hồi tức thì |
| Độc thoại nội tâm (Inner monologue) | "Phân tách Moshi" | Thiết kế mô hình đơn nhất nơi luồng suy nghĩ và luồng phát ngôn được đan xen luân phiên |

## Tài liệu đọc thêm

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
