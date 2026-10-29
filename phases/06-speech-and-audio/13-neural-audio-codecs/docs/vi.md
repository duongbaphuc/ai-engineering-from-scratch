# Codec Âm thanh Nơ-ron — EnCodec, SNAC, Mimi, DAC và Sự Phân tách Ngữ nghĩa - Âm học

> Vào năm 2026, các mô hình sinh âm thanh hầu như đều hoạt động trên các token. EnCodec, SNAC, Mimi và DAC biến các dạng sóng liên tục thành các chuỗi rời rạc mà mô hình transformer có thể dự đoán. Sự phân tách token ngữ nghĩa và âm học (semantic-vs-acoustic token split) — với bảng mã (codebook) đầu tiên mang tính ngữ nghĩa và các bảng mã còn lại mang tính âm học — là bước chuyển dịch kiến trúc quan trọng nhất đối với xử lý âm thanh kể từ khi Transformer ra đời.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## Vấn đề Đặt ra

Các mô hình ngôn ngữ làm việc trên các token rời rạc (discrete tokens). Âm thanh thì liên tục. Nếu bạn muốn xây dựng một mô hình dạng LLM cho tiếng nói / âm nhạc — như MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — trước tiên bạn cần một **codec âm thanh nơ-ron (neural audio codec)**: một bộ mã hóa (encoder) được huấn luyện để rời rạc hóa âm thanh thành một tập từ vựng token nhỏ, cùng một bộ giải mã (decoder) tương ứng để tái tạo lại dạng sóng (waveform).

Hai họ mô hình chính đã xuất hiện:

1. **Các codec ưu tiên tái tạo (Reconstruction-first codecs)** — EnCodec, DAC. Tối ưu hóa chất lượng cảm nhận âm thanh (perceptual audio quality). Các token mang tính "âm học" (acoustic) — chúng thu nhận toàn bộ thông tin bao gồm danh tính người nói, âm sắc (timbre), tạp âm nền.
2. **Các codec ưu tiên ngữ nghĩa (Semantic-first codecs)** — Mimi (Kyutai), SpeechTokenizer. Bắt buộc bảng mã đầu tiên phải mã hóa nội dung ngôn ngữ / ngữ âm (linguistic / phonetic content, thường thông qua chưng cất tri thức (distillation) từ WavLM). Các bảng mã tiếp theo nắm giữ chi tiết âm học.

Nhận thức đột phá giai đoạn 2024-2026: **một codec thuần túy tái tạo sẽ tạo ra tiếng nói mờ nhạt, thiếu rõ ràng khi bạn cố gắng sinh âm thanh từ văn bản.** Mô hình ngôn ngữ lớn (LLM) hoạt động trên các token codec khi đó phải học cả cấu trúc ngôn ngữ LẪN cấu trúc âm học trong cùng một bảng mã, điều này không thể mở rộng quy mô hiệu quả. Việc tách biệt chúng — bảng mã ngữ nghĩa 0 (semantic codebook 0) và các bảng mã âm học 1-N (acoustic codebooks 1-N) — chính là yếu tố giúp Moshi và Sesame CSM hoạt động thành công.

## Khái niệm Cốt lõi

![Bối cảnh bốn codec: EnCodec, DAC, SNAC (đa quy mô), Mimi (ngữ nghĩa + âm học)](../assets/codec-comparison.svg)

### Kỹ thuật cốt lõi: Lượng tử hóa Vector Phần dư (Residual Vector Quantization - RVQ)

Thay vì dùng một bảng mã khổng lồ duy nhất (vốn sẽ đòi hỏi hàng triệu mã để đạt chất lượng tốt), tất cả các codec âm thanh hiện đại đều sử dụng **RVQ**: một chuỗi phân tầng (cascade) các bảng mã nhỏ. Bảng mã đầu tiên lượng tử hóa đầu ra của bộ mã hóa; bảng mã thứ hai lượng tử hóa phần dư (residual); và cứ tiếp tục như vậy. Mỗi bảng mã chứa 1024 mã. 8 bảng mã = kích thước từ vựng hiệu dụng lên tới $1024^8 = 10^{24}$.

Tại thời điểm suy luận (inference), bộ giải mã cộng dồn tất cả các mã được chọn trên mỗi khung hình để tái tạo âm thanh.

### Bốn codec đáng chú ý nhất trong năm 2026

**EnCodec (Meta, 2022).** Nền tảng chuẩn đối sánh (baseline). Kiến trúc encoder-decoder trên dạng sóng, điểm nghẽn RVQ. Tần số lấy mẫu 24 kHz, hỗ trợ tối đa 32 bảng mã, mặc định 4 bảng mã ở mức 1,5 kbps. Sử dụng kiến trúc `1D conv + transformer + 1D conv`. Được sử dụng bởi MusicGen.

**DAC (Descript, 2023).** RVQ với các bảng mã được chuẩn hóa L2, các hàm kích hoạt tuần hoàn (periodic activation functions), cải tiến các hàm mất mát (loss functions). Đạt độ trung thực tái tạo (reconstruction fidelity) cao nhất trong số các codec mã nguồn mở — đôi khi không thể phân biệt được với tiếng nói gốc khi dùng 12 bảng mã. Băng thông toàn dải 44,1 kHz (full-band).

**SNAC (Hubert Siuzdak, 2024).** RVQ đa quy mô (Multi-scale RVQ) — các bảng mã thô (coarse) hoạt động ở tốc độ khung hình (frame rate) thấp hơn các bảng mã mịn (fine). Mô hình hóa âm thanh theo phân cấp một cách hiệu quả: một "bản phác thảo" thô ở mức ~12 Hz cộng với chi tiết mịn ở mức 50 Hz. Được sử dụng bởi Orpheus-3B vì cấu trúc phân cấp tương thích rất tốt với việc sinh chuỗi bằng mô hình ngôn ngữ (LM).

**Mimi (Kyutai, 2024).** Nhân tố thay đổi cuộc chơi của năm 2026. Tốc độ khung hình 12,5 Hz (cực kỳ thấp), 8 bảng mã ở mức 4,4 kbps. Bảng mã 0 được **chưng cất từ WavLM** — được huấn luyện để dự đoán các đặc trưng nội dung tiếng nói của WavLM. Các bảng mã 1-7 là phần dư âm học. Sự phân tách này vận hành Moshi (Bài 15) và Sesame CSM.

### Tốc độ khung hình đóng vai trò quyết định đối với mô hình hóa ngôn ngữ

Tốc độ khung hình càng thấp = chuỗi càng ngắn = mô hình ngôn ngữ chạy càng nhanh.

| Codec | Tốc độ khung hình | 1 giây = N khung hình | Phù hợp cho |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | âm nhạc, âm thanh tổng quát |
| DAC-44.1k | 86 Hz | 86 | âm nhạc độ trung thực cao |
| SNAC-24k (thô) | ~12 Hz | 12 | sinh tự hồi quy (AR-LM) hiệu quả |
| Mimi | 12,5 Hz | 12,5 | tiếng nói phát trực tuyến (streaming speech) |

Ở mức 12,5 Hz, một câu nói dài 10 giây chỉ tốn 125 khung hình codec — một mô hình transformer có thể dễ dàng dự đoán chuỗi này.

### Token ngữ nghĩa và token âm học (Semantic vs acoustic tokens)

```text
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Token ngữ nghĩa (bảng mã 0 trong Mimi).** Mã hóa nội dung được nói — âm vị (phonemes), từ ngữ, ngữ nghĩa. Được chưng cất từ WavLM thông qua hàm mất mát dự đoán phụ trợ (auxiliary prediction loss).
- **Token âm học (các bảng mã 1-7).** Mã hóa âm sắc, danh tính người nói, ngữ điệu (prosody), tiếng ồn nền, các chi tiết tinh tế.

Một mô hình ngôn ngữ tự hồi quy (autoregressive LM - AR LM) sẽ dự đoán token ngữ nghĩa trước (phụ thuộc vào văn bản đầu vào), sau đó dự đoán các token âm học (phụ thuộc vào token ngữ nghĩa + đoạn âm thanh người nói tham chiếu). Sự nhân tử hóa (factorization) này giải thích vì sao các hệ thống TTS hiện đại có thể nhân bản giọng nói theo cơ chế zero-shot: mô hình ngữ nghĩa xử lý nội dung; mô hình âm học xử lý âm sắc.

### Chất lượng tái tạo năm 2026 (bit trên giây, bitrate càng thấp càng tốt)

| Codec | Tốc độ bit (Bitrate) | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4,0 | 4,3 |
| EnCodec-6kbps | 6 kbps | 3,2 | 3,8 |
| DAC-6kbps | 6 kbps | 3,5 | 4,0 |
| SNAC-3kbps | 3 kbps | 3,3 | 3,8 |
| Mimi-4.4kbps | 4,4 kbps | 3,1 | 3,7 |

Các codec truyền thống như Opus vẫn vượt trội về chất lượng cảm nhận trên mỗi bit. Codec nơ-ron chiến thắng nhờ tạo ra **các token rời rạc** (điều mà Opus không làm được) và **chất lượng của mô hình sinh** (những gì mà mô hình ngôn ngữ có thể làm được với các token đó).

```figure
rvq-codec-cascade
```

## Thực hành Xây dựng

### Bước 1: Mã hóa với EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` ở mức 6 kbps. Mỗi mã nằm trong khoảng 0-1023 (10-bit).

### Bước 2: Giải mã và đo lường độ tái tạo

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Bước 3: Phân tách ngữ nghĩa - âm học (kiểu Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Bảng mã ngữ nghĩa 0 được căn chỉnh theo WavLM. Bạn có thể huấn luyện một mô hình transformer chuyển từ văn bản sang ngữ nghĩa (text-to-semantic) — với kích thước từ vựng nhỏ hơn rất nhiều so với việc chuyển trực tiếp từ văn bản sang âm thanh thô. Sau đó, một bộ giải mã từ âm học sang dạng sóng (acoustic-to-waveform) riêng biệt sẽ tiếp nhận điều kiện từ mẫu giọng tham chiếu của người nói.

### Bước 4: Tại sao mô hình AR LM trên các token codec lại hiệu quả

Đối với một đoạn thoại 10 giây ở cấu hình của Mimi (12,5 Hz × 8 bảng mã):

```math
N_{\text{tokens}} = 10 \times 12.5 \times 8 = 1000 \text{ tokens}
```

1000 token là độ dài ngữ cảnh rất nhỏ đối với một transformer. Một transformer 256 triệu tham số (256M) có thể sinh 10 giây tiếng nói trong vài mili-giây trên một GPU hiện đại.

## Ứng dụng Thực tế

Ánh xạ bài toán → codec:

| Tác vụ | Codec |
|------|-------|
| Sinh âm nhạc tổng quát | EnCodec-24k |
| Tái tạo với độ trung thực cao nhất | DAC-44.1k |
| AR LM trên tiếng nói (TTS) | SNAC hoặc Mimi |
| Tiếng nói phát trực tuyến song công toàn phần (Full-duplex) | Mimi (12,5 Hz) |
| Thư viện hiệu ứng âm thanh có điều kiện văn bản | EnCodec + T5 điều kiện |
| Chỉnh sửa âm thanh chi tiết (Fine-grained editing) | DAC + inpainting |

Quy tắc kinh nghiệm: **Nếu bạn đang xây dựng mô hình sinh (generative model), hãy bắt đầu với Mimi hoặc SNAC. Nếu bạn đang xây dựng đường ống nén dữ liệu (compression pipeline), hãy dùng Opus.**

## Cạm bẫy Thường gặp

- **Quá nhiều bảng mã.** Thêm bảng mã sẽ tăng độ trung thực tuyến tính nhưng cũng kéo dài độ dài chuỗi cho LM theo hàm tuyến tính. Hãy dừng lại ở 8-12 bảng mã.
- **Không khớp tốc độ khung hình.** Huấn luyện LM trên Mimi 12,5 Hz rồi tinh chỉnh (fine-tune) trên EnCodec 50 Hz sẽ thất bại trong âm thầm.
- **Giả định mọi bảng mã đều bình đẳng.** Trong Mimi, bảng mã 0 mang nội dung; mất bảng mã này sẽ phá hủy hoàn toàn khả năng nghe hiểu. Mất bảng mã 7 hầu như không thể nhận ra.
- **Chỉ sử dụng chất lượng tái tạo làm thước đo duy nhất.** Một codec có thể có khả năng tái tạo tuyệt vời nhưng lại vô dụng đối với việc sinh bằng mô hình ngôn ngữ nếu cấu trúc ngữ nghĩa của nó kém.

## Đóng gói Sản phẩm

Lưu thành `outputs/skill-codec-picker.md`. Chọn codec phù hợp cho một tác vụ sinh hoặc nén âm thanh cụ thể.

## Bài tập Thực hành

1. **Dễ.** Chạy `code/main.py`. Tệp này cài đặt một bộ lượng tử hóa vô hướng + phần dư mô phỏng (toy scalar + residual quantizer) và đo sai số tái tạo khi bạn thêm các bảng mã.
2. **Trung bình.** Cài đặt `encodec` và so sánh 1, 4, 8, 32 bảng mã trên một đoạn tiếng nói kiểm thử (held-out). Vẽ biểu đồ PESQ hoặc MSE theo tốc độ bit (bitrate).
3. **Khó.** Tải Mimi. Mã hóa một đoạn âm thanh. Thay thế bảng mã 0 bằng các số nguyên ngẫu nhiên; giải mã. Sau đó thay thế tương tự với bảng mã 7. So sánh hai trường hợp bị làm hỏng — việc làm hỏng bảng mã 0 sẽ phá hủy tính dễ hiểu của âm thanh; việc làm hỏng bảng mã 7 hầu như không làm thay đổi điều gì đáng kể.

## Thuật ngữ Then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| RVQ | Lượng tử hóa phần dư | Chuỗi phân tầng các bảng mã nhỏ; mỗi bảng mã lượng tử hóa phần dư của bước trước. |
| Tốc độ khung hình (Frame rate) | Tốc độ codec | Số lượng khung hình token mỗi giây. Càng thấp = mô hình ngôn ngữ chạy càng nhanh. |
| Bảng mã ngữ nghĩa (Semantic codebook) | Bảng mã 0 (Mimi) | Bảng mã được chưng cất từ các đặc trưng học tự giám sát (SSL); mã hóa nội dung. |
| Các bảng mã âm học (Acoustic codebooks) | Tất cả các bảng mã còn lại | Âm sắc, ngữ điệu, tạp âm, chi tiết tinh tế. |
| PESQ / ViSQOL | Chất lượng cảm nhận | Các số đo khách quan tương quan với điểm MOS (Mean Opinion Score). |
| EnCodec | Codec của Meta | Nền tảng RVQ chuẩn đối sánh; được MusicGen sử dụng. |
| Mimi | Codec của Kyutai | Tốc độ khung hình 12,5 Hz; phân tách ngữ nghĩa - âm học; vận hành Moshi. |

## Tài liệu Tham khảo

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — nền tảng chuẩn đối sánh RVQ.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — codec mã nguồn mở có độ trung thực cao nhất.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — RVQ đa quy mô.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — phân tách ngữ nghĩa - âm học, chưng cất WavLM.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — mô hình hai giai đoạn ngữ nghĩa/âm học.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec RVQ có khả năng phát trực tuyến nguyên bản.
