# Nhận dạng & Xác thực Người nói (Speaker Recognition & Verification)

> ASR trả lời câu hỏi "họ đã nói gì?" Nhận dạng người nói trả lời câu hỏi "ai là người đã nói?" Các phép toán thoạt nhìn có vẻ tương đồng — vector nhúng (embeddings) kết hợp độ tương đồng cosine — nhưng mọi quyết định triển khai trong thực tế đều xoay quanh một chỉ số duy nhất: EER.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6 · 02 (Phổ tín hiệu & Thang đo Mel), Giai đoạn 5 · 22 (Mô hình Nhúng - Embedding Models)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Người dùng nói một mật khẩu bằng giọng nói. Bạn cần xác định: liệu đây có đúng là người mà họ tự nhận hay không (*xác thực - verification*, bài toán 1:1), hay đây là một người cụ thể nào đó trong ngân hàng dữ liệu đăng ký của bạn (*định danh - identification*, bài toán 1:N)? Hoặc không phải ai cả — đây là một người nói hoàn toàn xa lạ chưa từng đăng ký (*tập mở - open-set*)?

Trước năm 2018: Phương pháp GMM-UBM kết hợp i-vector chiếm ưu thế. Chỉ số EER đạt mức chấp nhận được nhưng cực kỳ nhạy cảm và dễ vỡ trước sự thay đổi kênh thu âm (điện thoại so với laptop) cũng như cảm xúc người nói. Giai đoạn 2018–2022: x-vector lên ngôi (xương sống TDNN được huấn luyện với hàm mất mát lề góc). Từ năm 2022 trở đi: các vector nhúng ECAPA-TDNN và WavLM-large thống trị hoàn toàn. Đến năm 2026, lĩnh vực này được dẫn dắt bởi ba mô hình chủ đạo và một thước đo duy nhất.

Thước đo đó là **EER** — Tỷ lệ lỗi tương đương (Equal Error Rate). Bạn điều chỉnh ngưỡng quyết định sao cho Tỷ lệ chấp nhận sai (False Accept Rate - FAR) bằng với Tỷ lệ từ chối sai (False Reject Rate - FRR). Điểm giao nhau đó chính là EER. Đây là chỉ số xuất hiện trong mọi bài báo khoa học, mọi bảng xếp hạng công nghệ và mọi yêu cầu thầu kỹ thuật.

## Khái niệm (The Concept)

![Pipeline đăng ký mẫu và xác thực với vector nhúng, cosine và EER](../assets/speaker-verification.svg)

**Quy trình chuẩn (The Pipeline).** Đăng ký mẫu (Enrollment): ghi âm từ 5–30 giây giọng nói của đối tượng đích; tính toán một vector nhúng có số chiều cố định (192 chiều đối với ECAPA-TDNN, 256 chiều đối với WavLM-large). Xác thực (Verification): trích xuất vector nhúng của câu nói kiểm thử; tính độ tương đồng cosine với mẫu đăng ký; so sánh kết quả với một ngưỡng định trước.

**ECAPA-TDNN (Năm 2020, vẫn giữ vị thế thống trị năm 2026).** Viết tắt của Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Kiến trúc gồm các khối tích chập 1D tích hợp cơ chế squeeze-excitation, tầng gộp chú ý đa đầu (multi-head attention pooling), nối tiếp bởi một tầng tuyến tính đưa về 192 chiều. Được huấn luyện trên bộ dữ liệu VoxCeleb 1+2 (2.700 người nói, 1,1 triệu câu thoại) sử dụng hàm mất mát lề góc cộng tính (Additive Angular Margin loss - AAM-softmax).

**WavLM-SV (Từ năm 2022 trở đi).** Tinh chỉnh từ mô hình nền tảng tự giám sát WavLM-large bằng hàm mất mát AAM. Đạt chất lượng cao hơn nhưng kích thước lớn và tốc độ chậm hơn đáng kể — dung lượng hơn 300 MB so với chỉ 15 MB của ECAPA-TDNN.

**x-vector (Kiến trúc nền tảng cơ sở).** Kết hợp TDNN với cơ chế gộp thống kê (statistics pooling). Kiến trúc kinh điển này vẫn rất hữu dụng khi triển khai trên CPU hoặc thiết bị biên.

**Hàm mất mát AAM-softmax.** Hàm softmax tiêu chuẩn được cộng thêm một khoảng lề góc (margin) `m` trong không gian góc: `cos(θ + m)` cho lớp chính xác. Buộc mô hình phải kéo giãn khoảng cách góc giữa các lớp người nói khác nhau. Giá trị tham số tiêu chuẩn: khoảng lề `m=0.2`, hệ số tỷ lệ `s=30`.

### Phương pháp chấm điểm (Scoring)

- **Độ tương đồng Cosine** giữa vector nhúng đăng ký và kiểm thử. Quyết định chấp nhận hay từ chối dựa trên ngưỡng.
- **PLDA (Probabilistic Linear Discriminant Analysis).** Chiếu các vector nhúng vào một không gian ẩn mà ở đó tỷ số hợp lý (likelihood ratio) giữa hai mẫu "cùng một người nói" và "khác người nói" có nghiệm dạng giải tích đóng. Kết hợp thêm trên điểm cosine giúp giảm thêm từ 10–20% chỉ số EER. Từng là tiêu chuẩn trước năm 2020; hiện nay chủ yếu dùng trong các bài toán tập đóng (closed-set).
- **Chuẩn hóa điểm số (Score normalization).** Áp dụng `S-norm` hoặc `AS-norm`: chuẩn hóa từng điểm số dựa trên giá trị trung bình và độ lệch chuẩn của một tập hợp các đối tượng giả mạo (imposter cohort). Cực kỳ thiết yếu khi đánh giá mô hình xuyên miền dữ liệu.

### Các con số tiêu chuẩn bạn cần biết (Năm 2026)

| Mô hình | VoxCeleb1-O EER | Tham số | Thông lượng (A100) |
|---------|-----------------|---------|--------------------|
| x-vector (kinh điển) | 3.10% | 5 M | 400× thời gian thực |
| ECAPA-TDNN | 0.87% | 15 M | 200× |
| WavLM-SV large | 0.42% | 316 M | 20× |
| Phân đoạn & nhúng Pyannote 3.1 | 0.65% | 6 M | 100× |
| ReDimNet (2024) | 0.39% | 24 M | 100× |

### Phân tách danh tính người nói (Speaker Diarization)

Trả lời câu hỏi "ai nói vào lúc nào" trong một đoạn ghi âm có nhiều người nói. Pipeline xử lý: VAD → phân đoạn âm thanh → trích xuất vector nhúng cho từng phân đoạn → phân cụm (phân cụm tích tụ agglomerative hoặc spectral) → làm mịn đường biên thời gian. Bộ công cụ hiện đại: `pyannote.audio` 3.1, đóng gói trọn vẹn toàn bộ các khâu phân đoạn, nhúng và phân cụm chỉ trong một lời gọi hàm. Năm 2026, chỉ số lỗi phân tách DER (Diarization Error Rate) SOTA trên tập kiểm chuẩn AMI đạt ~15% (giảm mạnh từ mốc 23% năm 2022).

````figure
sp-eer-crossover
````

## Bắt tay vào xây dựng (Build It)

### Bước 1: Tạo vector nhúng cơ bản từ thống kê MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # vector 26 chiều
```

Cách làm này chưa thể đạt hiệu năng SOTA — chỉ dùng để trực quan hóa nguyên lý học tập. Mã nguồn trong `code/main.py` dùng phương pháp này làm bằng chứng khái niệm (proof-of-concept) trên dữ liệu người nói tổng hợp.

### Bước 2: Tính độ tương đồng Cosine và ngưỡng quyết định

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Bước 3: Tính toán chỉ số EER từ các cặp điểm số tương đồng

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Hàm trả về tuple `(eer, threshold_at_eer)`. Hãy luôn báo cáo đồng thời cả hai giá trị.

### Bước 4: Triển khai quy mô sản xuất với SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Đăng ký: lấy trung bình cộng các vector nhúng từ 3-5 mẫu âm thanh sạch
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# Xác thực
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ngưỡng đặc trưng của ECAPA; cần tinh chỉnh theo dữ liệu thực tế
```

### Bước 5: Phân tách danh tính người nói với pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Sử dụng trong thực tế (Use It)

Bộ công cụ tiêu chuẩn năm 2026:

| Tình huống thực tế | Lựa chọn tối ưu |
|--------------------|-----------------|
| Xác thực 1:1 tập đóng, chạy trên thiết bị biên | ECAPA-TDNN + ngưỡng cosine |
| Xác thực tập mở, triển khai đám mây (Cloud) | WavLM-SV + chuẩn hóa AS-norm |
| Phân tách người nói (họp trực tuyến, podcast) | `pyannote/speaker-diarization-3.1` |
| Chống giả mạo giọng nói (phát hiện phát lại / deepfake) | AASIST hoặc RawNet2 |
| Hệ thống nhúng siêu nhỏ (nhận diện từ khóa KWS + đăng ký giọng) | Titanet-Small (NeMo) |

## Những cạm bẫy kỹ thuật thường gặp

- **Lệch kênh thu âm (Channel mismatch):** Mô hình huấn luyện trên VoxCeleb (video YouTube) sẽ hoạt động kém khi áp dụng trực tiếp lên âm thanh cuộc gọi điện thoại thoại nén. Luôn đánh giá thử nghiệm trên đúng kênh thu âm mục tiêu.
- **Câu thoại kiểm thử quá ngắn.** Chỉ số EER suy giảm nghiêm trọng khi đoạn âm thanh kiểm thử ngắn hơn 3 giây.
- **Mẫu đăng ký bị lẫn tạp âm.** Một mẫu đăng ký bị dính tiếng ồn sẽ làm sai lệch toàn bộ vector mốc chuẩn. Luôn yêu cầu tối thiểu ≥3 mẫu sạch và lấy trung bình cộng.
- **Cố định ngưỡng quyết định trên mọi môi trường.** Luôn tinh chỉnh lại ngưỡng trên một tập dữ liệu phát triển (dev set) riêng biệt trích xuất từ đúng miền ứng dụng thực tế.
- **Tính cosine trên vector chưa chuẩn hóa.** Luôn thực hiện chuẩn hóa L2 trước khi tính toán; nếu không độ lớn vector sẽ chi phối toàn bộ kết quả.

## Xuất xưởng sản phẩm (Ship It)

Lưu lại dưới dạng `outputs/skill-speaker-verifier.md`. Kỹ năng này hướng dẫn chọn mô hình, thiết kế quy trình đăng ký mẫu, kế hoạch cân chỉnh ngưỡng và thiết lập các lớp bảo vệ chống gian lận giọng nói.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy file `code/main.py`. Chương trình sẽ sinh dữ liệu các "người nói" nhân tạo (với các cấu hình cao độ khác nhau), đăng ký mẫu và tính toán chỉ số EER trên danh sách thử nghiệm gồm 100 cặp âm thanh.
2. **Trung bình.** Sử dụng mô hình SpeechBrain ECAPA trên 30 câu thoại thuộc tập dữ liệu VoxCeleb1 (5 người nói × 6 câu mỗi người). Tính toán và so sánh chỉ số EER khi dùng điểm số Cosine thuần túy so với PLDA.
3. **Khó.** Xây dựng một pipeline hoàn chỉnh: Đăng ký mẫu → Phân tách người nói (Diarize) → Xác thực từng lượt nói (Verify) bằng `pyannote.audio`. Đánh giá chỉ số DER trên tập kiểm thử phát triển AMI dev set.

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Khái niệm thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| EER | Thước đo tiêu đề chính | Ngưỡng mà tại đó Tỷ lệ chấp nhận sai = Tỷ lệ từ chối sai. |
| Xác thực (Verification) | Bài toán 1:1 | "Đây có phải là Alice hay không?" |
| Định danh (Identification) | Bài toán 1:N | "Ai là người đang nói?" |
| Tập mở (Open-set) | Có người lạ | Tập kiểm thử có thể chứa những người nói chưa từng được đăng ký trong hệ thống. |
| Đăng ký (Enrollment) | Tạo hồ sơ giọng | Quá trình tính toán vector nhúng tham chiếu cho một người nói. |
| AAM-softmax | Hàm mất mát tối ưu | Softmax cộng thêm khoảng lề góc; ép các cụm vector người nói tách xa nhau. |
| PLDA | Phương pháp chấm điểm cổ điển | Phân tích biệt số tuyến tính xác suất; tính tỷ số hợp lý trên các vector nhúng. |
| DER | Thước đo phân tách | Tỷ lệ lỗi phân tách người nói — gồm lỗi bỏ sót + báo động giả + nhầm lẫn người nói. |

## Tài liệu đọc thêm (Further Reading)

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — Bài báo kinh điển đặt nền móng cho việc sử dụng vector nhúng sâu trong nhận dạng người nói.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — Kiến trúc thống trị xuyên suốt từ 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — Mô hình nền tảng tự giám sát cho xác thực người nói và phân tách giọng.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — Bộ công cụ mã nguồn mở quy chuẩn cho phân đoạn và phân tách người nói.
- [VoxCeleb leaderboard (Cập nhật 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — Bảng xếp hạng chỉ số EER cập nhật của các mô hình hàng đầu thế giới.
