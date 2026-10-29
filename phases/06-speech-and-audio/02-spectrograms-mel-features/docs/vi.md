# Phổ tín hiệu, Thang đo Mel & Đặc trưng Âm thanh

> Các mạng nơ-ron không xử lý trực tiếp dạng sóng thô một cách hiệu quả. Chúng xử lý phổ tín hiệu (spectrogram) tốt hơn nhiều, và thậm chí còn vượt trội hơn nữa với phổ Mel (mel spectrogram). Mọi mô hình ASR, TTS và phân loại âm thanh trong năm 2026 đều thành bại dựa trên lựa chọn tiền xử lý duy nhất này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6 · 01 (Nền tảng Âm thanh)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Hãy xét một đoạn âm thanh dài 10 giây ở tần số 16 kHz. Đó là một mảng gồm 160.000 số thực float nằm trong khoảng `[-1, 1]`, gần như không có sự tương quan trực tiếp nào với các nhãn phân loại như "tiếng chó sủa" hay "từ con mèo". Dạng sóng thô chứa đựng toàn bộ thông tin nhưng ở một định dạng mà mô hình học máy rất khó trích xuất. Cùng một âm vị giống hệt nhau khi được phát âm cách nhau 100 ms sẽ tạo ra các mẫu giá trị số thô hoàn toàn khác biệt.

Phổ tín hiệu (spectrogram) giải quyết triệt để vấn đề này. Nó nén bỏ những chi tiết thời gian cực nhỏ mà thính giác con người không nhận biết (như độ rung lắc vi giây) và bảo toàn cấu trúc mà thính giác chú ý tới (tần số nào mang nhiều năng lượng, trong các cửa sổ thời gian khoảng ~10–25 ms).

Phổ Mel còn tiến xa hơn một bước. Con người cảm nhận cao độ âm thanh (pitch) theo hàm logarit: sự khác biệt giữa 100 Hz và 200 Hz được cảm nhận "cùng một khoảng cách" như giữa 1000 Hz và 2000 Hz. Thang đo Mel (mel scale) uốn cong trục tần số để mô phỏng chính xác đặc tính thính giác này. Phổ tín hiệu trên thang Mel chính là đặc trưng quan trọng bậc nhất trong học máy âm thanh và tiếng nói suốt từ năm 2010 đến năm 2026.

## Khái niệm (The Concept)

![Thang bậc chuyển đổi từ dạng sóng qua STFT đến phổ Mel và MFCC](../assets/mel-features.svg)

**STFT (Biến đổi Fourier Thời gian ngắn - Short-Time Fourier Transform).** Cắt dạng sóng thành các khung (frames) chồng lấn lên nhau (thông số tiêu chuẩn: cửa sổ 25 ms, bước nhảy hop 10 ms = 400 mẫu / 160 mẫu ở tần số 16 kHz). Nhân từng khung với một hàm cửa sổ (cửa sổ Hann là mặc định; Hamming đem lại sự đánh đổi khác biệt đôi chút). Thực hiện biến đổi FFT trên từng khung. Xếp chồng các phổ biên độ này thành một ma trận có kích thước `(n_frames, n_freq_bins)`. Đó chính là phổ tín hiệu (spectrogram).

**Độ lớn Logarit (Log-magnitude).** Các giá trị biên độ thô trải dài qua 5 đến 6 bậc độ lớn (orders of magnitude). Chúng ta lấy `log(|X| + 1e-6)` hoặc `20 * log10(|X|)` để nén dải động (dynamic range). Mọi pipeline trong môi trường production đều sử dụng độ lớn logarit thay vì độ lớn tuyến tính thô.

**Thang đo Mel (Mel scale).** Tần số `f` tính bằng Hz được ánh xạ sang giá trị mel `m` theo công thức: $m = 2595 \cdot \log_{10}(1 + f / 700)$. Ánh xạ này xấp xỉ tuyến tính ở dải tần dưới 1 kHz và chuyển sang dạng logarit ở dải tần trên 1 kHz. Một tập hợp 80 dải tần mel (mel bins) bao phủ phạm vi từ 0 đến 8 kHz là đầu vào tiêu chuẩn của hầu hết hệ thống ASR hiện nay.

**Tập bộ lọc Mel (Mel filterbank).** Một tập hợp các bộ lọc hình tam giác được phân bố đều đặn trên thang đo Mel. Mỗi bộ lọc là một tổng có trọng số của các dải tần FFT liền kề. Nhân phổ biên độ STFT với ma trận bộ lọc mel này sẽ tạo ra phổ mel chỉ qua một phép nhân ma trận (matmul).

**Phổ Mel Logarit (Log-mel spectrogram).** Công thức `log(mel_spec + 1e-10)`. Đây là đầu vào của Whisper, Parakeet, SeamlessM4T, và là tầng tiền xử lý âm thanh (audio frontend) phổ quát của năm 2026.

**Các hệ số MFCCs (Mel-Frequency Cepstral Coefficients).** Lấy phổ log-mel, áp dụng phép Biến đổi Cosine Rời rạc (DCT - loại II), và giữ lại 13 hệ số đầu tiên. Phép biến đổi này giúp loại bỏ tương quan (decorrelate) giữa các đặc trưng và nén dữ liệu sâu hơn. MFCC từng là đặc trưng thống trị tuyệt đối cho đến khoảng năm 2015 trước khi các mô hình CNN và Transformer xử lý trực tiếp trên log-mel bắt kịp và vượt lên. Hiện nay MFCC vẫn được sử dụng phổ biến trong nhận dạng người nói (x-vectors, ECAPA).

**Sự đánh đổi độ phân giải (Resolution trade-off).** Kích thước FFT lớn hơn mang lại độ phân giải tần số tốt hơn nhưng làm giảm độ phân giải theo thời gian. Cặp thông số 25 ms / 10 ms là mặc định cho học máy giọng nói; 50 ms / 12,5 ms dùng cho âm nhạc; 5 ms / 2 ms dùng để phát hiện các tín hiệu quá độ cực ngắn (tiếng gõ trống, các âm bật plosives).

````figure
spectrogram-window
````

## Bắt tay vào xây dựng (Build It)

### Bước 1: Phân khung dạng sóng (Framing)

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

Một đoạn âm thanh dài 10 giây ở tần số 16 kHz với `frame_len=400, hop=160` sẽ tạo ra 998 khung.

### Bước 2: Áp dụng hàm cửa sổ Hann (Hann window)

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Nhân từng phần tử của khung với cửa sổ Hann trước khi đưa vào FFT. Bước này triệt tiêu hiện tượng rò rỉ phổ (spectral leakage) gây ra do việc cắt tín hiệu đột ngột ở các điểm biên khác không.

### Bước 3: Tính biên độ STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Trong thực tế production, bạn nên dùng `torch.stft` hoặc `librosa.stft` (được tối ưu hóa bằng thuật toán FFT và vector hóa). Vòng lặp ở đây mang mục đích sư phạm, chạy trên các đoạn âm thanh ngắn trong `code/main.py`.

### Bước 4: Tạo tập bộ lọc Mel (Mel filterbank)

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

Với 80 dải lọc mel phủ dải tần 0–8 kHz và `n_fft=400`, ta thu được một ma trận kích thước `(80, 201)`. Nhân ma trận biên độ STFT `(n_frames, 201)` với ma trận chuyển vị của bộ lọc để nhận về phổ mel `(n_frames, 80)`.

### Bước 5: Tính Log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Các giải pháp thay thế phổ biến: `librosa.power_to_db` (chuẩn hóa dB theo mốc tham chiếu), `10 * log10(power + eps)`. Mô hình Whisper áp dụng một quy trình cắt ngưỡng (clipping) và chuẩn hóa phức tạp hơn một chút (tham khảo hàm `log_mel_spectrogram` trong mã nguồn của Whisper).

### Bước 6: Trích xuất các hệ số MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Áp dụng biến đổi DCT lên từng khung log-mel và giữ lại 13 hệ số đầu tiên. Đó chính là ma trận MFCC. Hệ số đầu tiên thường bị loại bỏ vì nó chỉ phản ánh năng lượng tổng thể của khung.

## Sử dụng trong thực tế (Use It)

Bộ công cụ tiêu chuẩn năm 2026:

| Tác vụ | Đặc trưng sử dụng |
|--------|-------------------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 dải log-mel, bước nhảy 10 ms, cửa sổ 25 ms |
| Mô hình âm học TTS (VITS, F5-TTS, Kokoro) | 80 dải mel, bước nhảy 5–12 ms để kiểm soát thời gian mịn hơn |
| Phân loại âm thanh (AST, PANNs, BEATs) | 128 dải log-mel, bước nhảy 10 ms |
| Nhúng đặc trưng người nói (ECAPA-TDNN, WavLM) | 80 dải log-mel hoặc biểu diễn tự giám sát (SSL) trên dạng sóng thô |
| Âm nhạc (MusicGen, Stable Audio 2) | Token rời rạc EnCodec (không dùng mel) |
| Nhận diện từ khóa (Keyword spotting) | 40 hệ số MFCC cho các thiết bị nhúng siêu nhỏ |

Kinh nghiệm thực tiễn: **nếu không làm việc với âm nhạc chuyên sâu, hãy luôn bắt đầu với 80 dải log-mel.** Bất kỳ sự thay đổi nào đều cần có số liệu thực nghiệm chứng minh.

## Những cạm bẫy kỹ thuật thường gặp trong năm 2026

- **Lệch số lượng dải mel (Mel count mismatch):** Huấn luyện với 80 mel nhưng suy luận (inference) lại đưa vào 128 mel. Lỗi này diễn ra trong im lặng. Hãy luôn ghi log kích thước đặc trưng ở cả hai đầu pipeline.
- **Lệch tần số lấy mẫu ở tầng trên:** Đặc trưng mel tính ở tần số 22,05 kHz nhìn hoàn toàn khác so với 16 kHz. Hãy chuẩn hóa tần số lấy mẫu *trước khi* trích xuất đặc trưng.
- **Nhầm lẫn giữa dB và log tự nhiên:** Whisper kỳ vọng log-mel chứ không phải dB-mel. Một số pipeline của Hugging Face có thể tự nhận biết, nhưng code tự viết của bạn thì không.
- **Trôi dạt chuẩn hóa (Normalization drift):** Áp dụng chuẩn hóa trên từng câu (per-utterance) khi huấn luyện, nhưng lại áp dụng chuẩn hóa toàn cục (global) khi suy luận. Đây là lỗi phổ biến làm tăng gấp đôi chỉ số WER.
- **Rò rỉ do đệm số 0 (Padding leakage):** Việc đệm các số 0 vào cuối đoạn âm thanh sẽ tạo ra phổ phẳng lì ở những khung cuối. Hãy đệm đối xứng (symmetric padding) hoặc lặp lại giá trị biên (replicate).

## Xuất xưởng sản phẩm (Ship It)

Lưu lại dưới dạng `outputs/skill-feature-extractor.md`. Kỹ năng này tự động lựa chọn loại đặc trưng, số lượng dải mel, kích thước khung/bước nhảy và cơ chế chuẩn hóa phù hợp nhất cho mô hình đích.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy tệp `code/main.py`. Chương trình sẽ tổng hợp một tín hiệu chirp (tần số quét tăng dần từ 200 → 4000 Hz) và in ra dải tần mel có giá trị cực đại (argmax) trên mỗi khung. Vẽ đồ thị (tùy chọn) và kiểm tra sự tương đồng với đồ thị quét tần số.
2. **Trung bình.** Chạy lại thực nghiệm với `n_mels` trong tập `{40, 80, 128}` và `frame_len` trong tập `{200, 400, 800}`. Đo dải thông đỉnh nhọn (peak bandwidth) dọc theo trục thời gian. Tổ hợp thông số nào tái hiện tín hiệu chirp sắc nét nhất?
3. **Khó.** Cài đặt hàm `power_to_db` và so sánh độ chính xác nhận dạng của một mạng CNN nhỏ trên tập dữ liệu AudioMNIST bằng cách sử dụng (a) log-mel thô, (b) dB-mel với `ref=max`, (c) 13 hệ số MFCC kết hợp delta và delta-delta. Báo cáo độ chính xác top-1 thu được.

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Khái niệm thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Khung (Frame) | Một lát cắt tín hiệu | Đoạn dạng sóng dài 25 ms đưa vào một lượt tính FFT. |
| Bước nhảy (Hop) | Bước sải (stride) | Số lượng mẫu giữa hai khung liên tiếp; 10 ms là chuẩn của ASR. |
| Cửa sổ (Window) | Cửa sổ Hann hoặc Hamming | Hàm nhân từng điểm giúp nén dần hai đầu biên khung về giá trị 0. |
| STFT | Bộ tạo phổ tín hiệu | Biến đổi FFT có phân khung và áp dụng hàm cửa sổ; tạo ra ma trận thời gian × tần số. |
| Mel | Tần số được bẻ cong | Thang đo mô phỏng tri giác logarit; `m = 2595·log10(1 + f/700)`. |
| Bộ lọc (Filterbank) | Ma trận trọng số | Các bộ lọc hình tam giác chiếu phổ STFT lên các dải tần Mel. |
| Log-mel | Đầu vào của Whisper | Giá trị `log(mel_spec + eps)`; chuẩn mực công nghiệp năm 2026. |
| MFCC | Đặc trưng truyền thống | Biến đổi DCT của log-mel; gồm 13 hệ số đã khử tương quan. |

## Tài liệu đọc thêm (Further Reading)

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — Bài báo khoa học gốc đề xuất đặc trưng MFCC.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — Tài liệu gốc định nghĩa thang đo Mel.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — Khảo sát mã nguồn tham chiếu chính thức của Whisper.
- [librosa feature extraction docs](https://librosa.org/doc/latest/api/feature.html) — Tài liệu tham khảo chuẩn cho `mfcc`, `melspectrogram`, và cấu hình khung/bước nhảy.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Pipeline tiền xử lý quy mô sản xuất dành cho các mô hình Parakeet và Canary.
