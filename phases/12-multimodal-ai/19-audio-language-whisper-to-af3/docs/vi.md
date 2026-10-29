# Mô hình ngôn ngữ - âm thanh: Hành trình từ Whisper đến Audio Flamingo 3

> Whisper (Radford và cộng sự, tháng 12 năm 2022) đã định hình lĩnh vực nhận dạng giọng nói (speech recognition) — 680.000 giờ dữ liệu tiếng nói đa ngôn ngữ được giám sát yếu, một transformer mã hóa - giải mã (encoder-decoder transformer) tinh gọn, trở thành thước đo tiêu chuẩn mà mọi mô hình ASR (Automatic Speech Recognition - Nhận dạng tiếng nói tự động) sau đó đều trích dẫn. Tuy nhiên, nhận dạng không đồng nghĩa với suy luận (reasoning). Việc đặt các câu hỏi như "những nhạc cụ nào xuất hiện trong bản ghi này", "người nói đang bộc lộ cảm xúc gì" hay "điều gì đã diễn ra ở phút thứ 3" đòi hỏi năng lực thấu hiểu âm thanh (audio understanding) chứ không đơn thuần là phiên âm (transcription). Qwen-Audio, SALMONN, LTU và Audio Flamingo 3 của NVIDIA (AF3, tháng 7 năm 2025) đã từng bước hoàn thiện ngăn xếp công nghệ này: kế thừa các bộ mã hóa lớp Whisper, gắn thêm các bộ chuyển đổi truy vấn (Q-former), huấn luyện trên tập dữ liệu chỉ dẫn âm thanh - văn bản (audio-text instruction data) và tích hợp chuỗi tư duy (chain-of-thought reasoning). Bài học này sẽ phân tích chi tiết toàn bộ hành trình tiến hóa đó.

**Type:** Build
**Languages:** Python (stdlib, khung sườn log-Mel spectrogram + audio Q-former)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 phút

## Mục tiêu học tập

- Tính toán phổ tần số Mel logarit (log-Mel spectrogram) từ dạng sóng thô (waveform): phân khung cửa sổ (windowing), biến đổi Fourier nhanh (FFT), dải bộ lọc Mel (filter banks), phép biến đổi log.
- So sánh các phương án bộ mã hóa: Whisper encoder, BEATs, mô hình lai AF-Whisper. Xác định trường hợp tối ưu cho từng loại.
- Xây dựng một audio Q-former: $N$ truy vấn học được (learnable queries) thực hiện cơ chế chú ý chéo (cross-attention) lên các khung phổ tần số.
- Phân tích sự khác biệt giữa mô hình nối tầng (cascaded: Whisper-sau-đó-đến-LLM) và mô hình âm thanh - LLM đầu-cuối (end-to-end audio-LLM): lý do phương pháp đầu-cuối mở rộng quy mô tốt hơn cho các tác vụ suy luận.

## Vấn đề đặt ra

Nhận dạng giọng nói đã được giải quyết cơ bản bởi Whisper. Khả năng "OCR cho âm thanh" giờ đây là một tiện ích đại trà (commodity). Nhưng tiện ích này chỉ dừng lại ở tầng phiên âm văn bản. Nếu mô hình không thể suy luận dựa trên những gì nó nghe thấy — mốc thời gian, danh tính người nói, sắc thái cảm xúc, cấu trúc âm nhạc, âm thanh môi trường xung quanh — thì một bản phiên âm đơn thuần không thể đáp ứng được các tính năng sản phẩm nâng cao.

Có ba hướng tiếp cận rõ ràng:

1. **Nối tầng (Cascade):** Whisper đảm nhiệm việc phiên âm, LLM suy luận trên văn bản phiên âm đó. Phương pháp này hoạt động tốt với các kịch bản chỉ có lời nói thuần túy (pure-speech). Thất bại hoàn toàn khi gặp âm nhạc, âm thanh môi trường, nhiều người nói chèn âm hoặc biểu đạt cảm xúc.

2. **Mô hình âm thanh - LLM đầu-cuối (End-to-end audio-LLM):** Một bộ mã hóa âm thanh đưa trực tiếp các token âm thanh vào LLM, bỏ qua bước phiên âm trung gian. Hướng đi này bảo toàn nguyên vẹn thông tin âm học (acoustic information: cảm xúc, âm sắc người nói, môi trường xung quanh). Tuy nhiên, đòi hỏi dữ liệu huấn luyện mới hoàn toàn.

3. **Mô hình lai (Hybrid):** Kết hợp bộ mã hóa âm thanh và bộ giải mã văn bản có khả năng thực hiện song song cả phiên âm lẫn suy luận. Qwen-Audio và Audio Flamingo lựa chọn hướng đi này.

## Khái niệm cốt lõi

### Phổ Log-Mel: Đặc trưng đầu vào tiêu chuẩn

Mọi bộ mã hóa âm thanh hiện đại đều bắt đầu từ cùng một dạng biểu diễn đặc trưng: phổ tần số Mel logarit (log-Mel spectrogram).

1. Lấy mẫu lại (resample) tín hiệu về tần số 16 kHz.
2. Áp dụng phép biến đổi Fourier thời gian ngắn (Short-time Fourier Transform - STFT) với kích thước cửa sổ 25ms, bước nhảy (hop length) 10ms.
3. Tính độ lớn (magnitude) của kết quả FFT.
4. Áp dụng dải bộ lọc Mel (thường là 80 bộ lọc cách đều trên thang log từ 0 đến 8000 Hz) để chuyển đổi tần số vật lý sang thang cảm nhận thính giác của con người.
5. Nén log $(\log(1 + x))$ để thu hẹp dải động (dynamic range).

Kết quả: một mảng 2 chiều có kích thước $(T, 80)$, trong đó $T$ là số lượng khung thời gian (time frames). Với một đoạn âm thanh dài 30 giây ở tốc độ khung hình 100 Hz, kích thước sẽ là $(3000, 80)$.

### Bộ mã hóa của Whisper

Bộ mã hóa của Whisper là một transformer kiểu ViT gồm 12 lớp (layers), xử lý phổ log-Mel như một chuỗi các khung thời gian. Đầu ra là một vector trạng thái ẩn (hidden-state vector) tương ứng với mỗi khung thời gian.

Đối với tác vụ ASR, bộ giải mã của Whisper là một transformer áp dụng cơ chế chú ý chéo, sinh ra các token văn bản có điều kiện dựa trên đầu ra của bộ mã hóa. Đây là kiến trúc bộ mã hóa - giải mã tiêu chuẩn.

Đối với các mô hình ngôn ngữ - âm thanh (Audio-Language Models - ALMs), chúng ta muốn lấy đầu ra của bộ mã hóa này làm đầu vào cho một LLM khác. Mô hình kiến trúc phổ biến: đóng băng bộ mã hóa Whisper (frozen Whisper encoder), huấn luyện bộ chuyển đổi Q-former, đóng băng hoặc tinh chỉnh (fine-tune) LLM.

### BEATs và các bộ mã hóa chuyên biệt cho âm thanh

Whisper được huấn luyện chủ yếu trên dữ liệu chiếm ưu thế bởi tiếng nói con người. Do đó, khả năng hiểu âm nhạc và âm thanh môi trường của nó còn nhiều hạn chế.

BEATs (Chen và cộng sự, 2022) là một transformer tự giám sát (self-supervised transformer) được huấn luyện trên tập dữ liệu AudioSet. Nó nắm bắt các đặc trưng âm nhạc và âm thanh môi trường vượt trội hơn Whisper ở cùng quy mô tham số.

AF-Whisper (bộ mã hóa của Audio Flamingo 3): khởi đầu từ bộ mã hóa của Whisper-large-v3 và tiếp tục được huấn luyện đồng thời trên cả tiếng nói, âm thanh tổng hợp và âm nhạc. Nhờ vậy, một bộ mã hóa duy nhất có thể mang đầy đủ cả tín hiệu ngôn ngữ học (linguistic signal) lẫn tín hiệu âm học (acoustic signal). Các hệ thống trước đó như SALMONN phải ghép nối (concatenate) đặc trưng từ cả Whisper lẫn BEATs mới đạt được phạm vi bao phủ tương đương.

### Audio Q-former

Kế thừa cùng mẫu thiết kế với bộ Q-former thị giác trong BLIP-2. Một số lượng cố định các truy vấn có thể học (thường là 32 hoặc 64 queries) sẽ thực hiện chú ý chéo trên các khung đầu ra của bộ mã hóa âm thanh. Các truy vấn này chuyển hóa thành các token âm thanh được tiêu thụ trực tiếp bởi LLM.

Giai đoạn căn chỉnh huấn luyện (training alignment stage): chỉ huấn luyện riêng Q-former, tối ưu hóa hàm mất mát tương phản (contrastive loss) kết hợp chú thích (captioning loss) trên các cặp âm thanh - văn bản (AudioCaps, Clotho). Giai đoạn chỉ dẫn (instruction stage): huấn luyện đầu-cuối, mở băng LLM, tối ưu hóa trên dữ liệu chỉ thị đa tác vụ.

### Tiến trình phát triển: SALMONN, Qwen-Audio, AF3

- **SALMONN** (Tang và cộng sự, 2023): Kết hợp Whisper + BEATs + Q-former + LLaMA. Đây là mô hình âm thanh - LLM mã nguồn mở đầu tiên sở hữu năng lực suy luận thực chất. Điểm chuẩn tổng hợp trên MMAU đạt xấp xỉ 0,55.
- **Qwen-Audio** (Chu và cộng sự, 2023): Kiến trúc tương tự nhưng được huấn luyện trên tập dữ liệu đa dạng hơn, tinh chỉnh sâu cho hội thoại nhiều lượt (multi-turn dialogue). Điểm MMAU đạt xấp xỉ 0,60.
- **LTU — Listen, Think, Understand** (Gong và cộng sự, 2023): Tập trung vào dữ liệu suy luận tường minh, chú trọng chuỗi tư duy phân tích các đoạn trích âm thanh. Quy mô nhỏ hơn nhưng định hướng rất rõ ràng.
- **Audio Flamingo 3** (Goel và cộng sự, tháng 7 năm 2025): Mô hình SOTA mã nguồn mở hiện nay. Sử dụng xương sống LLM 7B (Qwen2.5-7B), bộ mã hóa âm thanh AF-Whisper phát triển từ Whisper-large-v3, các tầng thích ứng (adaptor layers) nhỏ nối vào LLM, huấn luyện trên hơn 1 triệu cặp chỉ dẫn âm thanh - văn bản. MMAU đạt 0,72, bắt kịp các mô hình độc quyền hàng đầu trên một số tác vụ chuyên biệt.

AF3 cũng giới thiệu cơ chế chuỗi tư duy theo yêu cầu (on-demand chain-of-thought) cho âm thanh: mô hình có thể tùy chọn sinh ra các token suy nghĩ nội tâm ("trước hết hãy phân tích các nhạc cụ: ...") trước khi đưa ra câu trả lời cuối cùng. Độ chính xác trên các tác vụ suy luận phức tạp tăng từ 3 đến 5 điểm phần trăm khi kích hoạt chế độ suy nghĩ này.

### Mô hình nối tầng so với đầu-cuối

Quy trình nối tầng truyền thống:

1. Whisper phiên âm âm thanh → văn bản.
2. LLM suy luận trên văn bản đầu ra.

Phương pháp này hoạt động hoàn hảo cho tác vụ "tóm tắt tập podcast này". Tuy nhiên, nó thất bại hoàn toàn với:
- "Giai điệu bài hát này gợi cảm xúc gì?" — cảm xúc nằm ở âm thanh, không nằm ở ca từ.
- "Ai đang nói, Alice hay Bob?" — đòi hỏi nhận dạng đặc trưng người nói (speaker identification).
- "Vụ nổ xảy ra ở giây thứ mấy?" — căn chỉnh thời gian chuẩn xác bị biến mất khi đưa về dạng văn bản.
- "Đây là âm thanh thật hay âm thanh do AI tạo ra?" — phát hiện deepfake âm thanh bắt buộc phải dựa vào các đặc trưng âm học tầng sâu.

Kiến trúc đầu-cuối bảo toàn trọn vẹn tín hiệu âm học. Qwen-Audio và AF3 xử lý tự nhiên cả âm nhạc, môi trường xung quanh và cảm xúc người nói.

### Công thức triển khai thực tế năm 2026

Khi xây dựng một sản phẩm thấu hiểu âm thanh mới:

- **Chọn mô hình nối tầng nếu:** Mục tiêu duy nhất là phiên âm chữ, không cần phân tích âm nhạc hay suy luận cảm xúc.
- **Chọn AF3 / họ mô hình Qwen-Audio nếu:** Cần xử lý âm nhạc, cảm xúc, phân tách nhiều người nói, hoặc suy luận âm thanh phức tạp.

Mô hình nối tầng tiết kiệm chi phí và triển khai đơn giản hơn. Mô hình đầu-cuối vượt trội về mặt năng lực toàn diện.

### MMAU — Thước đo chuẩn mực cho suy luận âm thanh

MMAU (Massive Multimodal Audio Understanding) là bộ điểm chuẩn đánh giá suy luận âm thanh tiêu chuẩn giai đoạn 2024-2025:

- 10.000 cặp câu hỏi - đáp âm thanh - văn bản trải rộng trên giọng nói, âm nhạc và âm thanh môi trường.
- Đánh giá khả năng phân loại, suy luận thời gian, suy luận quan hệ nhân quả và hỏi đáp mở.
- Kiểm tra chính xác những yếu tố mà các luồng nối tầng phiên âm - văn bản thường xuyên bỏ sót.

Mô hình mở SOTA (AF3) đạt mức 0,72; các mô hình độc quyền biên giới đạt khoảng 0,78 (Gemini 2.5 Pro, Claude Opus 4.7). Khoảng cách này nhỏ hơn đáng kể so với độ lệch giữa mã nguồn mở và đóng trong VideoMME, cho thấy các mô hình âm thanh - LLM đang trưởng thành rất nhanh chóng.

```figure
audio-text-ctc
```

## Sử dụng trong thực tế

Tệp `code/main.py`:

- Triển khai tính toán phổ tần số log-Mel hoàn toàn bằng thư viện chuẩn Python (stdlib): phân cửa sổ, DFT cơ bản, dải bộ lọc Mel.
- Khung sườn audio Q-former: nhận các khung đầu ra từ bộ mã hóa, tính toán ma trận Q, K, V, cơ chế chú ý và phát ra $N$ token âm thanh.
- So sánh trực tiếp giữa đường ống nối tầng và mô hình đầu-cuối trên một tác vụ giả lập mẫu.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-audio-llm-pipeline-picker.md`. Dựa trên yêu cầu của tác vụ âm thanh đầu vào (phiên âm, gắn thẻ âm nhạc, suy luận cảm xúc, phân tách người nói - diarization, phân loại môi trường), kỹ năng này tự động đề xuất phương án nối tầng, mô hình đầu-cuối AF3 hoặc giải pháp kết hợp lai tối ưu.

## Bài tập thực hành

1. Hãy tính toán số chiều của phổ log-Mel cho một đoạn âm thanh dài 30 giây ở tần số 16kHz, kích thước cửa sổ 25ms, bước nhảy 10ms, 80 dải Mel. Kích thước này thay đổi như thế nào nếu tần số lấy mẫu là 48kHz?

2. Tại sao Whisper lại thể hiện kém trên dữ liệu âm nhạc? Những đặc trưng âm thanh nào mà BEATs nắm bắt được nhưng Whisper lại bỏ qua?

3. So sánh Audio Q-former với 64 truy vấn so với 32 truy vấn: ở ngưỡng độ phức tạp tác vụ nào thì việc dùng 64 truy vấn đem lại hiệu quả rõ rệt? Việc dùng 32 truy vấn giúp tiết kiệm tài nguyên tính toán cho mục đích gì?

4. Đọc Mục 4 trong bài báo nghiên cứu của AF3 về cơ chế suy nghĩ theo yêu cầu (on-demand thinking). Đề xuất ba tác vụ âm thanh mà kỹ thuật chuỗi tư duy (chain-of-thought) mang lại sự cải thiện hiệu quả rõ rệt nhất.

5. Triển khai một quy trình phân đoạn và định danh người nói (diarization pipeline) tối giản sử dụng đầu ra từ AF3. Làm cách nào để báo hiệu sự thay đổi người nói trong luồng dữ liệu?

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Phổ Log-Mel (Log-Mel spectrogram) | "Đặc trưng Mel" (Mel features) | Mảng 2D (thời gian, tần số) chứa giá trị độ lớn logarit sau khi đi qua dải bộ lọc Mel |
| Audio Q-former | "Audio Perceiver" | Nút thắt cổ chai chú ý chéo từ đầu ra của bộ mã hóa âm thanh sang số lượng truy vấn có độ dài cố định cấp cho LLM |
| Nối tầng (Cascaded) | "ASR-kèm-LLM" | Luồng xử lý trong đó Whisper phiên âm văn bản và LLM văn bản suy luận; làm mất thông tin âm học |
| Đầu-cuối (End-to-end) | "Audio-LLM" | Các đặc trưng âm thanh đi trực tiếp vào LLM thông qua Q-former; bảo toàn trọn vẹn tín hiệu âm học |
| BEATs | "Bộ mã hóa AudioSet" | Transformer tự giám sát (SSL) được huấn luyện trên AudioSet; rất mạnh về âm nhạc và âm thanh môi trường |
| MMAU | "Benchmark suy luận âm thanh" | Bộ dữ liệu gồm 10k cặp câu hỏi - đáp trải rộng trên tiếng nói, âm nhạc và môi trường; tiêu chuẩn đánh giá từ 2024 |
| Suy nghĩ theo yêu cầu (On-demand thinking) | "Audio CoT" | Cơ chế cho phép mô hình tùy chọn sinh các token suy luận trước khi đưa ra câu trả lời cuối, tăng 3-5 điểm độ chính xác |

## Tài liệu đọc thêm

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
