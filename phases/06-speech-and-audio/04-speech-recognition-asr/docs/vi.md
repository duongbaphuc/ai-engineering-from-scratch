# Nhận dạng Tiếng nói (ASR) — CTC, RNN-T, Attention

> Nhận dạng tiếng nói là bài toán phân loại âm thanh tại từng bước thời gian, được liên kết chặt chẽ bởi một mô hình chuỗi thấu hiểu cả cấu trúc ngôn ngữ lẫn khoảng lặng. CTC, RNN-T và Attention là ba hướng giải pháp kinh điển. Hãy chọn một hướng đi và thấu hiểu tường tận lý do đằng sau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6 · 02 (Phổ tín hiệu & Thang đo Mel), Giai đoạn 5 · 08 (CNNs & RNNs cho Văn bản), Giai đoạn 5 · 10 (Attention)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Bạn có một đoạn âm thanh 10 giây ở tần số 16 kHz. Bạn muốn xuất ra một chuỗi văn bản: "bật đèn phòng bếp lên". Thách thức cốt lõi nằm ở sự bất đối xứng về mặt cấu trúc: các khung âm thanh không khớp nối 1-1 với các ký tự văn bản. Cùng một từ "okay" có thể được phát âm trong 200 ms hoặc kéo dài tới 1200 ms. Khoảng lặng nằm rải rác ngắt quãng câu nói. Có những âm vị kéo dài hơn các âm vị khác. Và số lượng token văn bản ở đầu ra không thể biết trước được.

Có ba hướng tiếp cận toán học để giải quyết vấn đề này:

1. **CTC (Connectionist Temporal Classification - Phân loại Thời gian Kết nối).** Dự đoán phân phối xác suất token trên từng khung âm thanh, bao gồm một token rỗng đặc biệt gọi là *blank*. Gộp các ký tự lặp liên tiếp và loại bỏ token blank trong quá trình giải mã. Mô hình không tự hồi quy (non-autoregressive), tốc độ suy luận cực nhanh. Được sử dụng bởi wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer - Bộ chuyển đổi Mạng Nơ-ron Hồi quy).** Mạng kết hợp (joint network) dự đoán token tiếp theo dựa trên khung đầu ra của bộ mã hóa và các token đã phát ra trước đó. Khả năng truyền phát trực tuyến (streamable) mạnh mẽ. Được Google áp dụng cho ASR trên thiết bị và NVIDIA dùng cho dòng mô hình Parakeet.
3. **Mô hình Bộ mã hóa - Bộ giải mã dùng Attention (Attention encoder-decoder).** Bộ mã hóa nén các khung âm thanh thành các trạng thái ẩn, bộ giải mã áp dụng cơ chế chú ý chéo (cross-attention) để sinh các token văn bản theo cơ chế tự hồi quy (autoregressive). Được áp dụng trong Whisper, SeamlessM4T.

Tính đến năm 2026, chỉ số WER trên tập kiểm chuẩn LibriSpeech test-clean đạt mốc SOTA là 1,4% (Parakeet-TDT-1.1B của NVIDIA) và 1,58% (Whisper-Large-v3-turbo). Sự chênh lệch về độ chính xác là rất nhỏ, nhưng sự khác biệt trong kiến trúc triển khai thực tế lại vô cùng lớn.

## Khái niệm (The Concept)

![Ba hướng tiếp cận ASR: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Trực giác về CTC.** Giả sử bộ mã hóa phát ra `T` phân phối xác suất ở cấp độ khung trên `V+1` token (V ký tự từ vựng + token blank). Đối với chuỗi mục tiêu `y` có độ dài `U < T`, bất kỳ đường căn chỉnh khung nào mà sau khi rút gọn thu về đúng `y` đều được tính là hợp lệ. Hàm mất mát CTC tính tổng xác suất trên toàn bộ các đường căn chỉnh hợp lệ đó. Quá trình suy luận (inference): chọn phần tử có xác suất lớn nhất (argmax) trên từng khung, gộp các ký tự trùng nhau liên tiếp và loại bỏ các ký hiệu blank.

Ưu điểm: không tự hồi quy, hỗ trợ streaming, không cần nhìn trước tương lai (zero lookahead). Nhược điểm: *giả định độc lập có điều kiện* (conditional independence assumption) — dự đoán tại mỗi khung hoàn toàn độc lập với các khung khác, dẫn đến mô hình không tích hợp sẵn mô hình ngôn ngữ nội tại (internal language model). Khắc phục bằng cách kết hợp với mô hình ngôn ngữ ngoài (external LM) thông qua tìm kiếm chùm (beam search) hoặc shallow fusion.

**Trực giác về RNN-T.** Bổ sung một mạng *dự đoán* (predictor network) để nhúng lịch sử các token trước đó, và một mạng *kết hợp* (joiner network) hòa trộn trạng thái bộ dự đoán với khung mã hóa âm thanh thành một phân phối xác suất chung trên `V+1` nhãn (ở đây `+1` là nhãn null / không phát ra token). RNN-T mô hình hóa rõ ràng sự phụ thuộc có điều kiện mà CTC đã bỏ qua. Mô hình stream được vì mỗi bước thời gian chỉ phụ thuộc vào các khung âm thanh trong quá khứ và các token đã sinh ra.

Ưu điểm: hỗ trợ streaming trực tiếp + tích hợp sẵn mô hình ngôn ngữ nội tại. Nhược điểm: quá trình huấn luyện phức tạp hơn và tiêu tốn nhiều bộ nhớ hơn (mạng lưới mất mát 3D lattice); các kernel tính hàm mất mát RNN-T cấu thành một nhóm thư viện chuyên biệt riêng biệt.

**Kiến trúc Bộ mã hóa - Bộ giải mã dùng Attention.** Bộ mã hóa (gồm 6–32 tầng Transformer) xử lý toàn bộ các khung phổ log-mel. Bộ giải mã (6–32 tầng Transformer) áp dụng cross-attention lên đầu ra của bộ mã hóa để sinh từng token tự hồi quy. Không có bất kỳ ràng buộc căn chỉnh cứng nhắc nào — cơ chế attention có thể quan sát tự do bất cứ vị trí nào trong đoạn âm thanh. Không thể truyền phát trực tiếp (non-streamable) trừ khi giới hạn tầm nhìn chú ý (chẳng hạn như Whisper-Streaming dạng phân đoạn chunked, 2024).

Ưu điểm: chất lượng nhận dạng ngoại tuyến (offline ASR) cao nhất, dễ huấn luyện với các công cụ seq2seq tiêu chuẩn. Nhược điểm: độ trễ tự hồi quy tỷ lệ thuận với độ dài chuỗi đầu ra; đòi hỏi kỹ thuật phức tạp nếu muốn chạy streaming.

### WER: Chỉ số đánh giá duy nhất

**Tỷ lệ lỗi từ (Word Error Rate - WER)** = $(S + D + I) / N$, trong đó $S$=thay thế (substitutions), $D$=xóa (deletions), $I$=chèn thêm (insertions), $N$=tổng số từ trong câu tham chiếu gốc. Công thức này tương đương với khoảng cách hiệu chỉnh Levenshtein ở cấp độ từ vựng. Giá trị càng thấp càng tốt. WER trên 20% nhìn chung không thể sử dụng được; dưới 5% đạt tới mức độ chuẩn xác tương đương con người đối với giọng đọc chuẩn. Số liệu chuẩn mực năm 2026:

| Mô hình | LibriSpeech test-clean | LibriSpeech test-other | Kích thước |
|---------|------------------------|------------------------|------------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1 tỷ tham số |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809 triệu |
| Canary-1B Flash | 1.48% | 2.87% | 1 tỷ |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3 tỷ |

Tất cả các mô hình hàng đầu này đều dựa trên kiến trúc encoder-decoder hoặc RNN-T. Các hệ thống thuần CTC (như wav2vec 2.0) duy trì mức WER quanh 1,8–2,1% trên tập test-clean.

````figure
ctc-collapse
````

## Bắt tay vào xây dựng (Build It)

### Bước 1: Giải mã CTC tham lam (Greedy CTC decode)

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: danh sách các vector xác suất trên từng khung
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Hai quy tắc rút gọn: gộp các ký tự trùng lặp liên tiếp, và loại bỏ các token blank. Ví dụ: `a a _ _ a b b _ c` → `a a b c`.

### Bước 2: Tìm kiếm chùm CTC (Beam-search CTC)

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Môi trường production thường sử dụng tìm kiếm chùm trên cây tiền tố (prefix tree) kết hợp mô hình ngôn ngữ (LM fusion); đoạn mã trên là khung xương trực quan hóa thuật toán.

### Bước 3: Tính toán tỷ lệ lỗi từ WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Bước 4: Suy luận với mô hình Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Chỉ với một dòng lệnh để chạy hệ thống ASR tổng quát mạnh mẽ hàng đầu năm 2026. Xử lý trên GPU 24 GB với tốc độ gấp ~20 lần thời gian thực (realtime).

### Bước 5: Xử lý streaming với Parakeet hoặc wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

ASR truyền phát thời gian thực yêu cầu cơ chế attention theo khối (chunked encoder attention) và trạng thái duy trì liên tục qua các khung; hãy sử dụng các thư viện hỗ trợ chuyên biệt (NeMo cho Parakeet, hoặc pipeline của `transformers` với tham số `chunk_length_s`).

## Sử dụng trong thực tế (Use It)

Bộ công cụ tiêu chuẩn năm 2026:

| Tình huống triển khai | Lựa chọn tối ưu |
|-----------------------|-----------------|
| Tiếng Anh, xử lý ngoại tuyến (offline), ưu tiên chất lượng tối đa | Whisper-large-v3-turbo |
| Đa ngôn ngữ, độ chống chịu nhiễu cao | SeamlessM4T v2 |
| Truyền phát trực tuyến (streaming), độ trễ thấp | Parakeet-TDT-1.1B hoặc Riva |
| Thiết bị biên (Edge), di động, độ trễ <500 ms | Whisper-Tiny lượng tử hóa hoặc Moonshine (2024) |
| Âm thanh bản ghi dài (Long-form audio) | Whisper kết hợp phân đoạn dựa trên VAD (WhisperX) |
| Chuyên ngành đặc thù (y tế, pháp lý) | Tinh chỉnh wav2vec 2.0 + tích hợp mô hình ngôn ngữ chuyên ngành (domain LM) |

## Những cạm bẫy kỹ thuật thường gặp trong năm 2026

- **Thiếu tầng lọc tiếng nói VAD.** Chạy Whisper trên các đoạn chỉ toàn khoảng lặng sẽ dẫn đến hiện tượng ảo giác (hallucination - ví dụ mô hình tự chèn "Thanks for watching!"). Luôn luôn chặn trước bằng VAD.
- **Nhầm lẫn giữa WER theo ký tự, từ vựng hay subword.** Luôn báo cáo chỉ số WER ở cấp độ từ (word-level) *sau khi* đã chuẩn hóa văn bản (chuyển chữ thường, loại bỏ toàn bộ dấu câu).
- **Trôi dạt nhận diện ngôn ngữ (Language ID drift).** Cơ chế tự động nhận diện ngôn ngữ của Whisper có thể định tuyến nhầm các đoạn nhiều tiếng ồn sang tiếng Nhật hoặc tiếng Wales; hãy gán cứng tham số `language="en"` (hoặc ngôn ngữ mong muốn) nếu bạn đã biết trước.
- **Âm thanh dài không phân đoạn.** Whisper có cửa sổ chú ý cố định 30 giây. Sử dụng `chunk_length_s=30, stride=5` đối với bất kỳ đoạn âm thanh nào dài hơn mốc này.

## Xuất xưởng sản phẩm (Ship It)

Lưu lại dưới dạng `outputs/skill-asr-picker.md`. Kỹ năng này gợi ý mô hình, chiến lược giải mã, cấu hình phân đoạn (chunking) và giải pháp tích hợp mô hình ngôn ngữ cho mục tiêu triển khai cụ thể.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy mã nguồn `code/main.py`. Chương trình sẽ giải mã tham lam đầu ra CTC mô phỏng và tính toán chỉ số WER so với chuỗi văn bản tham chiếu.
2. **Trung bình.** Cài đặt hoàn chỉnh thuật toán tìm kiếm chùm trên cây tiền tố (prefix-tree beam search) ở Bước 2 (xử lý chuẩn xác quy tắc gộp token blank). So sánh kết quả với thuật toán tham lam trên tập dữ liệu tổng hợp gồm 10 mẫu.
3. **Khó.** Sử dụng mô hình `whisper-large-v3-turbo` trên tập dữ liệu [LibriSpeech test-clean](https://www.openslr.org/12). Tính toán chỉ số WER trên 100 câu đầu tiên và so sánh với các con số công bố chính thức.

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Khái niệm thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| CTC | Hàm mất mát có token rỗng | Tính biên xác suất trên toàn bộ các căn chỉnh giữa khung và token; không tự hồi quy. |
| RNN-T | Hàm mất mát cho streaming | Mở rộng CTC với mạng dự đoán token tiếp theo; mô hình hóa thứ tự từ. |
| Attention enc-dec | Phong cách kiến trúc của Whisper | Bộ mã hóa + bộ giải mã chú ý chéo; chất lượng ngoại tuyến xuất sắc nhất. |
| WER | Con số báo cáo bắt buộc | Công thức `(S+D+I)/N` tính ở cấp độ từ vựng. |
| Blank | Trạng thái rỗng | Token đặc biệt trong CTC báo hiệu "không phát ra ký tự nào tại khung thời gian này". |
| LM fusion | Tích hợp mô hình ngôn ngữ | Cộng xác suất log của mô hình ngôn ngữ ngoài có trọng số trong quá trình beam search. |
| VAD | Cổng chặn khoảng lặng | Bộ phát hiện hoạt tính giọng nói (Voice Activity Detector); cắt tỉa các vùng không phải tiếng nói. |

## Tài liệu đọc thêm (Further Reading)

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — Bài báo khoa học gốc đặt nền móng cho CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — Bài báo khoa học gốc đề xuất mô hình RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Bài báo kinh điển về Whisper năm 2022; mở rộng phiên bản v3-turbo năm 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — Thẻ mô hình dẫn đầu bảng xếp hạng Open ASR Leaderboard năm 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — Bảng xếp hạng kiểm chuẩn trực tiếp hơn 25 mô hình hàng đầu thế giới.
