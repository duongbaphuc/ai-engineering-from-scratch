# Nền tảng Âm thanh — Dạng sóng, Lấy mẫu, Biến đổi Fourier

> Dạng sóng là tín hiệu thô. Phổ tín hiệu (spectrogram) là dạng biểu diễn. Đặc trưng Mel là định dạng thân thiện với học máy (ML). Mọi pipeline nhận dạng tiếng nói tự động (ASR) và tổng hợp tiếng nói (TTS) hiện đại đều bước qua các nấc thang này, và nấc thang đầu tiên chính là việc thấu hiểu kỹ thuật lấy mẫu và biến đổi Fourier.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 1 · 06 (Vectors & Matrices), Giai đoạn 1 · 14 (Probability Distributions)
**Time:** ~45 phút

## Vấn đề đặt ra (The Problem)

Microphone thu nhận và tạo ra tín hiệu áp suất biến thiên theo thời gian. Mạng nơ-ron của bạn lại tiêu thụ các tensor. Nằm giữa hai thế giới này là cả một tập hợp các quy ước kỹ thuật mà nếu bị vi phạm, chúng sẽ tạo ra những lỗi âm thầm (silent bugs): mô hình huấn luyện hội tụ bình thường nhưng tỷ lệ lỗi từ (WER - Word Error Rate) lại tăng gấp đôi, hệ thống TTS phát ra tiếng xì rè (hiss), hoặc một hệ thống nhân bản giọng nói (voice cloning) lại ghi nhớ đặc tính của microphone thay vì người nói.

Mọi lỗi kỹ thuật trong các hệ thống xử lý giọng nói đều quy về một trong ba câu hỏi cốt lõi:

1. Dữ liệu được ghi ở tần số lấy mẫu (sampling rate) nào, và mô hình thực sự kỳ vọng tần số nào?
2. Tín hiệu có bị hiện tượng răng cưa / méo tần số (aliasing) hay không?
3. Bạn đang thao tác trực tiếp trên các mẫu âm thanh thô (raw samples) hay trên một biểu diễn miền tần số (frequency representation)?

Nắm vững những nguyên lý này và toàn bộ phần còn lại của Giai đoạn 6 sẽ trở nên sáng tỏ. Hiểu sai chúng, ngay cả một mô hình mạnh mẽ như Whisper-Large-v4 cũng sẽ chỉ tạo ra kết quả vô giá trị.

## Khái niệm (The Concept)

![Biểu diễn trực quan dạng sóng, lấy mẫu, DFT và các dải tần số](../assets/audio-fundamentals.svg)

**Dạng sóng (Waveform).** Một mảng một chiều chứa các số thực dấu phẩy động (floats) nằm trong đoạn `[-1.0, 1.0]`, được đánh chỉ số theo số thứ tự mẫu. Để quy đổi sang đơn vị giây, lấy chỉ số chia cho tần số lấy mẫu: `t = n / sr`. Một đoạn âm thanh dài 10 giây ở tần số 16 kHz là một mảng gồm 160.000 số thực float.

**Tần số lấy mẫu (Sampling rate - sr).** Số lượng mẫu thu nhận được trên mỗi giây. Các mức tần số lấy mẫu phổ biến tính đến năm 2026:

| Tần số | Mục đích sử dụng |
|--------|------------------|
| 8 kHz | Điện thoại truyền thống, hệ thống VOIP cũ. Tần số Nyquist ở mức 4 kHz làm triệt tiêu các phụ âm. Tránh dùng cho ASR. |
| 16 kHz | Tiêu chuẩn của ASR. Whisper, Parakeet, SeamlessM4T v2 đều nhận đầu vào 16 kHz. |
| 22.05 kHz | Huấn luyện bộ mã hóa tạo tiếng nói (TTS vocoder) cho các mô hình thế hệ cũ. |
| 24 kHz | TTS hiện đại (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | Âm thanh CD, âm nhạc thương mại. |
| 48 kHz | Điện ảnh, âm thanh chuyên nghiệp, hệ thống TTS độ trung thực cao (VALL-E 2, NaturalSpeech 3). |

**Định lý Nyquist-Shannon.** Một tần số lấy mẫu `sr` có thể biểu diễn chính xác, không nhập nhằng các tần số lên tới `sr/2`. Giới hạn `sr/2` này được gọi là *tần số Nyquist* (Nyquist frequency). Năng lượng của bất kỳ thành phần tần số nào vượt quá ngưỡng Nyquist sẽ bị *méo răng cưa* (aliased) — tức là bị phản xạ gấp khúc ngược trở lại các dải tần thấp hơn — làm sai lệch tín hiệu gốc. Vì vậy, luôn phải áp dụng bộ lọc thông thấp (low-pass filter) trước khi giảm tần số lấy mẫu (downsampling).

**Độ sâu bit (Bit depth).** Định dạng PCM 16-bit (số nguyên có dấu int16, khoảng giá trị ±32.767) là định dạng trao đổi dữ liệu phổ quát toàn cầu. Chuẩn 24-bit dùng cho sản xuất âm nhạc, số thực float32 dùng cho các thuật toán xử lý tín hiệu số (DSP) nội bộ. Các thư viện như `soundfile` đọc dữ liệu int16 từ đĩa nhưng sẽ trả về các mảng float32 chuẩn hóa trong khoảng `[-1, 1]`.

**Biến đổi Fourier (Fourier Transform).** Mọi tín hiệu hữu hạn đều có thể phân tích thành tổng của các sóng hình sin ở các tần số khác nhau. Biến đổi Fourier Rời rạc (DFT - Discrete Fourier Transform) tính toán, từ `N` mẫu thời gian, `N` hệ số phức (complex coefficients) — mỗi hệ số tương ứng với một dải tần số (frequency bin). Dải tần `bin k` tương ứng với tần số thực nghiệm `k · sr / N` Hz. Độ lớn (magnitude) đại diện cho biên độ tại tần số đó, trong khi góc pha biểu thị góc lệch pha (phase).

**Biến đổi Fourier nhanh (FFT - Fast Fourier Transform).** Thuật toán tối ưu với độ phức tạp `O(N log N)` để tính toán DFT khi `N` là lũy thừa của 2. Mọi thư viện xử lý âm thanh đều sử dụng FFT ở tầng thực thi lõi. Một phép biến đổi FFT với kích thước 1024 mẫu ở tần số 16 kHz sẽ mang lại 512 dải tần hữu dụng trải dài từ 0 đến 8 kHz với độ phân giải là 15,6 Hz trên mỗi bin.

**Phân khung và hàm cửa sổ (Framing + window).** Chúng ta không thực hiện FFT trên toàn bộ tệp âm thanh dài cùng một lúc. Thay vào đó, tín hiệu được cắt thành các *khung* (frames) chồng lấn lên nhau (thông thường dài 25 ms với bước nhảy (hop) 10 ms), nhân từng khung với một hàm cửa sổ (chẳng hạn cửa sổ Hann, Hamming) để triệt tiêu hiện tượng gián đoạn ở biên khung, sau đó mới tính FFT cho từng khung. Đây chính là Biến đổi Fourier Thời gian ngắn (STFT - Short-Time Fourier Transform). Bài học 02 sẽ đi sâu tiếp nối từ khái niệm này.

````figure
mel-scale
````

## Bắt tay vào xây dựng (Build It)

### Bước 1: Đọc một đoạn âm thanh và vẽ đồ thị dạng sóng

Tệp `code/main.py` chỉ sử dụng duy nhất mô-đun chuẩn `wave` của Python để bản demo hoàn toàn không có phụ thuộc bên ngoài. Trong môi trường production, bạn sẽ dùng `soundfile` hoặc `torchaudio.load` (cả hai đều trả về tuple `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # kích thước (T,), sr=int
```

### Bước 2: Tổng hợp sóng sin từ các nguyên lý cơ bản

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Một sóng sin tần số 440 Hz (nốt La chuẩn hòa nhạc - concert A) ở tần số lấy mẫu 16 kHz kéo dài 1 giây sẽ gồm 16.000 số thực float. Ghi ra tệp bằng `wave.open(..., "wb")` sử dụng mã hóa 16-bit PCM.

### Bước 3: Tự cài đặt thuật toán DFT thủ công

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

Độ phức tạp là `O(N²)` — phù hợp với `N=256` để kiểm chứng tính đúng đắn của lý thuyết, nhưng quá chậm đối với các tệp âm thanh thực tế. Mã nguồn production luôn gọi `numpy.fft.rfft` hoặc `torch.fft.rfft`.

### Bước 4: Tìm tần số chiếm ưu thế (Dominant frequency)

Chỉ số đỉnh biên độ `k_star` tương ứng với tần số `k_star * sr / N`. Chạy phép tính này trên sóng sin 440 Hz sẽ cho một đỉnh biên độ nằm ở dải tần (bin) `440 * N / sr`.

### Bước 5: Minh họa hiện tượng răng cưa (Aliasing)

Lấy mẫu một sóng sin 7 kHz ở tần số lấy mẫu 10 kHz (ngưỡng Nyquist là 5 kHz). Sóng âm 7 kHz nằm trên ngưỡng Nyquist và sẽ bị gấp nếp ngược về `10 − 7 = 3 kHz`. Đỉnh biên độ FFT sẽ xuất hiện tại tần số 3 kHz. Đây là bài kiểm tra kinh điển minh họa hiện tượng méo răng cưa và cũng là lý do vì sao mọi mạch chuyển đổi DAC/ADC đều tích hợp sẵn một bộ lọc thông thấp rất dốc (brick-wall low-pass filter).

## Sử dụng trong thực tế (Use It)

Bộ công cụ tiêu chuẩn bạn sẽ thực sự triển khai trong năm 2026:

| Tác vụ | Thư viện | Lý do |
|--------|----------|-------|
| Đọc/ghi WAV/FLAC/OGG | `soundfile` (bọc ngoài libsndfile) | Nhanh nhất, ổn định, trả về dữ liệu float32. |
| Chuyển đổi tần số lấy mẫu (Resample) | `torchaudio.transforms.Resample` hoặc `librosa.resample` | Đã tích hợp sẵn bộ lọc chống méo răng cưa (anti-aliasing) chuẩn xác. |
| STFT / Đặc trưng Mel | `torchaudio` hoặc `librosa` | Thân thiện với GPU; nằm trong hệ sinh thái PyTorch. |
| Truyền phát thời gian thực (Streaming) | `sounddevice` hoặc `pyaudio` | Các liên kết (bindings) đa nền tảng cho PortAudio. |
| Kiểm tra thông tin tệp | `ffprobe` hoặc `soxi` | Công cụ dòng lệnh CLI, tốc độ cao, hiển thị sr/kênh/codec. |

Quy tắc quyết định: **hãy đồng bộ hóa tần số lấy mẫu trước khi đối chiếu bất kỳ yếu tố nào khác**. Mô hình Whisper yêu cầu đầu vào chuẩn 16 kHz mono float32. Nếu truyền vào âm thanh stereo 44,1 kHz, bạn sẽ nhận về kết quả sai lệch hoàn toàn trông giống hệt như lỗi kiến trúc mô hình.

## Xuất xưởng sản phẩm (Ship It)

Lưu lại dưới dạng `outputs/skill-audio-loader.md`. Kỹ năng (skill) này giúp bạn kiểm tra xem âm thanh đầu vào có đáp ứng đúng yêu cầu của mô hình phía sau hay không và tự động lấy mẫu lại chuẩn xác khi có sự sai lệch.

## Bài tập thực hành (Exercises)

1. **Dễ.** Tổng hợp một tín hiệu âm thanh dài 1 giây kết hợp 3 tần số 220 Hz + 440 Hz + 880 Hz ở tần số 16 kHz. Chạy thuật toán DFT. Xác nhận sự hiện diện của 3 đỉnh biên độ tại đúng các dải tần dự kiến.
2. **Trung bình.** Thu âm một tệp WAV giọng nói dài 3 giây ở tần số 48 kHz. Giảm tần số lấy mẫu xuống 16 kHz bằng `torchaudio.transforms.Resample` (có lọc chống méo tần số), sau đó giảm tiếp xuống 16 kHz bằng phương pháp loại bỏ mẫu ngây thơ (naive decimation - chỉ lấy một mẫu sau mỗi ba mẫu). Tính FFT cho cả hai cách. Hiện tượng méo răng cưa xuất hiện tại những vị trí nào?
3. **Khó.** Xây dựng thuật toán STFT từ đầu chỉ dùng thư viện `math` và hàm DFT ở Bước 3. Kích thước khung (frame size) là 400, bước nhảy (hop) là 160, dùng cửa sổ Hann. Vẽ đồ thị biên độ bằng `matplotlib.pyplot.imshow`. Đây chính là nền tảng của phổ tín hiệu (spectrogram) trong Bài học 02.

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Khái niệm thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Tần số lấy mẫu (Sample rate) | Số lượng mẫu trên mỗi giây | Tần số tính theo Hz mà tại đó bộ chuyển đổi ADC đo lường tín hiệu. |
| Nyquist | Tần số tối đa có thể biểu diễn được | `sr/2`; năng lượng vượt quá ngưỡng này sẽ bị méo gập ngược lại các dải tần dưới. |
| Độ sâu bit (Bit depth) | Độ phân giải của từng mẫu âm thanh | `int16` = 65.536 mức; `float32` = độ chính xác 24-bit trong đoạn `[-1, 1]`. |
| DFT | Biến đổi Fourier cho các chuỗi rời rạc | `N` mẫu thời gian → `N` hệ số tần số phức. |
| FFT | Thuật toán DFT tốc độ cao | Thuật toán có độ phức tạp `O(N log N)` yêu cầu `N` phải là lũy thừa của 2. |
| Dải tần (Bin) | Một cột biểu diễn tần số | `k · sr / N` Hz; độ phân giải = `sr / N`. |
| STFT | Bản chất nền tảng của phổ tín hiệu | Biến đổi FFT có phân khung và áp dụng hàm cửa sổ theo thời gian. |
| Méo răng cưa (Aliasing) | Những tần số ảo kỳ lạ trong tín hiệu | Năng lượng phía trên ngưỡng Nyquist bị phản chiếu ngược xuống các dải tần thấp hơn. |

## Tài liệu đọc thêm (Further Reading)

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — Bài báo khoa học kinh điển đặt nền móng cho định lý lấy mẫu.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — Sách giáo trình DSP kinh điển, miễn phí và chuẩn mực.
- [librosa docs — audio primer](https://librosa.org/doc/latest/auto_tutorials/index.html) — Hướng dẫn thực hành xử lý âm thanh kèm mã nguồn.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.taylorfrancis.com/books/mono/10.1201/9781315372150/room-acoustics-heinrich-kuttruff) — Tài liệu tham khảo giải thích vì sao âm thanh trong thế giới thực không phải là các sóng sin đơn giản.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — Giải thích trực giác về các dải tần số FFT chỉ trong 10 phút.
