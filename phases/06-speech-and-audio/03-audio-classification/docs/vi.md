# Phân loại Âm thanh — Từ k-NN trên MFCC đến AST và BEATs

> Mọi bài toán từ "tiếng chó sủa và tiếng còi cứu hỏa" cho tới "đây là ngôn ngữ nào" đều thuộc phạm trù phân loại âm thanh (audio classification). Đặc trưng nền tảng là phổ Mel. Kiến trúc mô hình dịch chuyển qua từng thập kỷ. Nhưng phương pháp đánh giá vẫn luôn kiên định với AUC, F1 và độ thu hồi (recall) trên từng lớp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6 · 02 (Phổ tín hiệu & Thang đo Mel), Giai đoạn 3 · 06 (CNNs), Giai đoạn 5 · 08 (CNNs & RNNs cho Văn bản)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Bạn nhận được một đoạn âm thanh dài 10 giây. Bạn cần trả lời câu hỏi: "Âm thanh này là gì?" Âm thanh đô thị (tiếng còi, máy khoan, chó sủa), khẩu lệnh điều khiển (yes/no/stop), nhận diện ngôn ngữ (en/es/ar), cảm xúc người nói (giận dữ/bình thường), hay âm thanh môi trường (trong nhà/ngoài trời, tiếng xì xào đám đông). Tất cả các bài toán này đều là *phân loại âm thanh* (audio classification), và tính đến năm 2026, kiến trúc cơ sở (baseline) đã hoàn toàn định hình: phổ log-mel → mạng tích chập CNN hoặc Transformer → hàm softmax.

Khó khăn cốt lõi không nằm ở mạng nơ-ron. Thách thức thực sự nằm ở dữ liệu. Các tập dữ liệu âm thanh chịu sự mất cân bằng lớp (class imbalance) trầm trọng, độ lệch miền dữ liệu (domain shift) rất mạnh (âm thanh phòng thu sạch so với âm thanh đường phố nhiều tạp âm), và nhãn bị nhiễu (ai là người phân định ranh giới giữa "tiếng ồn đô thị" và "tiếng ồn quán ăn"?). 80% thành công của bài toán phụ thuộc vào khâu tinh lọc dữ liệu, các kỹ thuật tăng cường dữ liệu (augmentation) và phương pháp đánh giá, chứ không nằm ở việc chỉ đơn thuần hoán đổi CNN sang Transformer.

## Khái niệm (The Concept)

![Thang bậc phân loại âm thanh: từ k-NN trên MFCC đến AST và BEATs](../assets/audio-classification.svg)

**Thuật toán k-NN trên các đặc trưng MFCC (Baseline của thập niên 1990).** Trải phẳng (flatten) các hệ số MFCC trên mỗi đoạn âm thanh, tính độ tương đồng cosine (cosine similarity) với một ngân hàng mẫu đã gán nhãn, và đưa ra quyết định dựa trên số phiếu đa số (majority vote) của K mẫu gần nhất. Phương pháp này hoạt động hiệu quả bất ngờ trên các tập dữ liệu nhỏ và sạch (Speech Commands, ESC-50) và có thể chạy mượt mà mà không cần GPU.

**Mạng 2D CNN trên phổ log-mel (Giai đoạn 2015-2019).** Coi ma trận log-mel kích thước `(T, n_mels)` như một bức ảnh 2D thông thường. Áp dụng các kiến trúc như ResNet-18 hoặc biến thể phong cách VGG. Thực hiện gộp trung bình toàn cục (global mean pooling) dọc theo trục thời gian, sau đó áp dụng Softmax để phân loại. Đây vẫn là giải pháp baseline phổ biến hàng đầu trong các cuộc thi Kaggle năm 2026.

**Audio Spectrogram Transformer - AST (Giai đoạn 2021-2024).** Cắt phổ log-mel thành các mảng nhỏ (ví dụ các mảng patch 16×16), bổ sung mã hóa vị trí (position embeddings), và truyền vào một mạng Vision Transformer (ViT). Đạt mốc đỉnh cao công nghệ (SOTA - State of the Art) trên tập dữ liệu AudioSet (mAP 0.485) đối với học có giám sát.

**BEATs và WavLM-base (Giai đoạn 2024-2026).** Các mô hình tiền huấn luyện tự giám sát (self-supervised pretraining) trên hàng triệu giờ âm thanh. Tinh chỉnh (fine-tune) cho tác vụ cụ thể của bạn với lượng dữ liệu có gán nhãn chỉ bằng 1-10% so với trước đây. Vào năm 2026, đây là điểm khởi đầu mặc định cho các bài toán âm thanh phi tiếng nói (non-speech audio). Mô hình BEATs-iter3 vượt qua AST từ 1-2 điểm mAP trên AudioSet trong khi chỉ tiêu tốn 1/4 chi phí tính toán.

**Bộ mã hóa Whisper làm xương sống đóng băng (Whisper-encoder as a frozen backbone - 2024).** Lấy bộ mã hóa (encoder) của Whisper, loại bỏ bộ giải mã (decoder), và gắn thêm một bộ phân loại tuyến tính (linear classifier). Đạt hiệu năng tiệm cận SOTA trong việc nhận dạng ngôn ngữ (language ID) và phân loại sự kiện đơn giản mà không cần bất kỳ bước tăng cường dữ liệu âm thanh phức tạp nào. Đây được coi là baseline "bữa trưa miễn phí" hiệu quả cao.

### Thách thức thực sự: Mất cân bằng dữ liệu giữa các lớp

ESC-50: gồm 50 lớp, mỗi lớp 40 đoạn âm thanh — cân bằng hoàn hảo, dễ huấn luyện. UrbanSound8K: 10 lớp, độ mất cân bằng tỷ lệ 10:1. AudioSet: 632 lớp với phần đuôi dài (long tail) chênh lệch lên tới 100.000:1. Các kỹ thuật mang lại hiệu quả thực tế:

- Lấy mẫu cân bằng (Balanced sampling) trong quá trình huấn luyện (không áp dụng trong lúc đánh giá).
- Mixup: nội suy tuyến tính giữa hai đoạn âm thanh (và nhãn của chúng) để làm kỹ thuật tăng cường dữ liệu.
- SpecAugment: che ngẫu nhiên các dải thời gian và tần số trên phổ tín hiệu. Đơn giản nhưng có ý nghĩa sống còn.

### Đánh giá mô hình (Evaluation)

- Đa lớp loại trừ lẫn nhau (Multiclass exclusive - như Speech Commands): độ chính xác top-1 (top-1 accuracy), độ chính xác top-5.
- Đa lớp đa nhãn (Multiclass multi-label - như AudioSet, UrbanSound): độ chính xác trung bình trung bình (mAP - mean Average Precision).
- Dữ liệu mất cân bằng nghiêm trọng: độ thu hồi trên từng lớp (per-class recall) + chỉ số macro F1.

Các con số tiêu chuẩn năm 2026 bạn cần nắm vững:

| Tập kiểm thử | Baseline | SOTA 2026 | Nguồn tham chiếu |
|--------------|----------|-----------|------------------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | Bài báo BEATs (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | Bảng xếp hạng HEAR 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | Kết quả kiểm thử HEAR v2 |

````figure
mfcc-pipeline
````

## Bắt tay vào xây dựng (Build It)

### Bước 1: Trích xuất đặc trưng

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Bước 2: Tạo vector tóm tắt độ dài cố định

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Đơn giản nhưng vô cùng hiệu quả: giá trị trung bình cộng với phương sai theo thời gian tạo ra một vector nhúng 26 chiều cố định từ 13 hệ số MFCC. Tốc độ thực thi tức thì, từng đánh bại các mô hình mạng nơ-ron SOTA trên tập ESC-50 vào năm 2017.

### Bước 3: Phân loại bằng thuật toán k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Bước 4: Nâng cấp lên mạng CNN trên phổ log-mel

Triển khai bằng PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

Mô hình có 3 triệu tham số, huấn luyện chỉ mất ~10 phút trên tập ESC-50 với một GPU RTX 4090 đơn lẻ, đạt độ chính xác trên 80%.

### Bước 5: Tinh chỉnh Audio Transformer tiền huấn luyện (Ví dụ với AST)

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Ví dụ trên tinh chỉnh mô hình AST từ Hugging Face Hub. BEATs, lựa chọn mặc định của năm 2026, không có sẵn trên Hugging Face Hub: hãy tải checkpoint từ [bản phát hành BEATs trong microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats) và nạp qua các lớp `BEATs` và `BEATsConfig` của repo đó; cấu trúc vòng lặp tinh chỉnh hoàn toàn tương tự.

## Sử dụng trong thực tế (Use It)

Bộ công cụ tiêu chuẩn năm 2026:

| Tình huống thực tế | Lựa chọn khởi đầu tối ưu |
|--------------------|--------------------------|
| Tập dữ liệu siêu nhỏ (<1.000 đoạn) | k-NN trên trung bình MFCC (làm baseline) + tăng cường dữ liệu âm thanh |
| Tập dữ liệu vừa (1.000 – 100.000 đoạn) | Tinh chỉnh mô hình BEATs hoặc AST |
| Tập dữ liệu lớn (>100.000 đoạn) | Huấn luyện từ đầu hoặc tinh chỉnh bộ mã hóa của Whisper |
| Ứng dụng thời gian thực trên thiết bị nhúng (Edge) | CNN trên 40 dải MFCC, lượng tử hóa xuống int8 (kiểu KWS) |
| Phân loại đa nhãn (AudioSet) | BEATs-iter3 kết hợp hàm mất mát BCE loss + mixup + SpecAugment |
| Nhận diện ngôn ngữ (Language ID) | MMS-LID, baseline SpeechBrain VoxLingua107 |

Quy tắc quyết định: **hãy bắt đầu với một mạng nền tảng (backbone) đóng băng thay vì huấn luyện một mô hình hoàn toàn mới từ đầu**. Tinh chỉnh tầng phân loại trên BEATs giúp bạn chạm mốc 95% hiệu năng SOTA chỉ trong vài giờ thay vì mất hàng tuần lễ.

## Xuất xưởng sản phẩm (Ship It)

Lưu lại dưới dạng `outputs/skill-classifier-designer.md`. Kỹ năng này hỗ trợ lựa chọn kiến trúc mạng, phương pháp tăng cường dữ liệu, chiến lược cân bằng lớp và chỉ số đánh giá phù hợp nhất cho tác vụ phân loại âm thanh cụ thể.

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy tệp `code/main.py`. Chương trình sẽ huấn luyện mô hình baseline k-NN MFCC trên tập dữ liệu tổng hợp gồm 4 lớp (các âm đơn thuần ở các cao độ khác nhau). Trích xuất và phân tích ma trận nhầm lẫn (confusion matrix).
2. **Trung bình.** Thay thế hàm `summarize` bằng bộ 4 giá trị đặc trưng [mean, var, skew, kurtosis] (trung bình, phương sai, độ lệch và độ nhọn). Liệu cơ chế gộp 4 mômen thống kê này có vượt qua mô hình mean+var trên cùng tập dữ liệu tổng hợp hay không?
3. **Khó.** Sử dụng `torchaudio`, huấn luyện một mạng 2D CNN trên phân đoạn fold 1 của tập dữ liệu ESC-50. Báo cáo độ chính xác kiểm định chéo 5-fold (5-fold cross-validation accuracy). Thêm kỹ thuật SpecAugment (time mask = 20, freq mask = 10) và đo lường mức độ cải thiện điểm số.

## Các thuật ngữ chính (Key Terms)

| Thuật ngữ | Khái niệm thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| AudioSet | Bộ ImageNet của lĩnh vực âm thanh | Tập dữ liệu 2 triệu clip từ YouTube của Google, gồm 632 lớp gán nhãn yếu. |
| ESC-50 | Tập kiểm chuẩn phân loại kích thước nhỏ | 50 lớp × 40 clip về các âm thanh môi trường xung quanh. |
| AST | Audio Spectrogram Transformer | Kiến trúc ViT áp dụng trên các mảnh patch của phổ log-mel; đỉnh cao SOTA 2021. |
| BEATs | Học âm thanh tự giám sát | Mô hình của Microsoft, phiên bản iter3 dẫn đầu bảng xếp hạng AudioSet năm 2026. |
| Mixup | Tăng cường dữ liệu theo cặp | Phép kết hợp: `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Tăng cường dữ liệu bằng mặt nạ | Làm rỗng (gán 0) ngẫu nhiên các dải thời gian và tần số trên phổ tín hiệu. |
| mAP | Chỉ số đánh giá chính cho bài toán đa nhãn | Độ chính xác trung bình trung bình trên các lớp và các ngưỡng phân loại. |

## Tài liệu đọc thêm (Further Reading)

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — Kiến trúc nền tảng chuẩn mực từ năm 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — Chuẩn mực mặc định từ năm 2024 trở đi.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — Kỹ thuật tăng cường dữ liệu âm thanh thống trị hiện nay.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — Bộ kiểm chuẩn 50 lớp trường tồn với thời gian.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — Hệ thống phân loại 632 lớp từ YouTube; tiêu chuẩn vàng của ngành.
