# Đánh giá Âm thanh — WER, MOS, UTMOS, MMAU, FAD và Các Bảng Xếp hạng Mở

> Bạn không thể đưa vào sản xuất những gì bạn không thể đo lường. Bài học này định danh các số đo đánh giá chuẩn mực của năm 2026 cho mọi tác vụ âm thanh: nhận dạng tiếng nói ASR (WER, CER, RTFx), tổng hợp giọng nói TTS (MOS, UTMOS, SECS, vòng lặp WER-qua-ASR), mô hình ngôn ngữ - âm thanh (MMAU, LongAudioBench), sinh âm nhạc (FAD, CLAP) và nhận dạng người nói (EER). Cùng với đó là các bảng xếp hạng công khai (open leaderboards) để đối sánh trực tiếp.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## Vấn đề Đặt ra

Mỗi tác vụ âm thanh đều có nhiều số đo đánh giá khác nhau, mỗi số đo tập trung vào một khía cạnh riêng biệt. Việc sử dụng sai số đo là nguyên nhân trực tiếp khiến bạn phát hành một mô hình trông có vẻ rất ấn tượng trên bảng điều khiển nội bộ nhưng lại thể hiện tệ hại khi triển khai thực tế. Danh mục số đo chuẩn tắc của năm 2026:

| Tác vụ | Số đo chính (Primary) | Số đo phụ (Secondary) |
|------|---------|-----------|
| Nhận dạng tiếng nói (ASR) | WER | CER · RTFx · độ trễ token đầu tiên |
| Tổng hợp giọng nói (TTS) | MOS / UTMOS | SECS · vòng lặp WER-qua-ASR · CER · TTFA |
| Nhân bản giọng nói (Voice cloning) | SECS (ECAPA cosine) | MOS · CER |
| Xác minh người nói (Speaker verification) | EER | minDCF · FAR / FRR tại điểm vận hành |
| Phân đoạn người nói (Diarization) | DER | JER · nhầm lẫn người nói (speaker confusion) |
| Phân loại âm thanh (Audio classification) | top-1 · mAP | macro F1 · độ thu hồi theo từng lớp (per-class recall) |
| Sinh âm nhạc (Music generation) | FAD | CLAP · điểm MOS từ hội đồng người nghe |
| Mô hình ngôn ngữ âm thanh (Audio language model) | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| S2S phát trực tuyến (Streaming S2S) | độ trễ P50/P95 | WER · MOS |

## Khái niệm Cốt lõi

![Ma trận đánh giá âm thanh — các số đo so với các tác vụ so với bảng xếp hạng năm 2026](../assets/eval-landscape.svg)

### Các số đo cho Nhận dạng Tiếng nói (ASR)

**WER (Tỷ lệ lỗi từ - Word Error Rate).** Công thức: `(S + D + I) / N` (trong đó S là số từ thay thế, D là số từ bị xóa, I là số từ chèn thêm, N là tổng số từ gốc). Cần chuyển chữ thường, loại bỏ dấu câu, chuẩn hóa dạng số trước khi chấm điểm. Sử dụng thư viện `jiwer` hoặc công cụ `whisper_normalizer` của OpenAI. Mức &lt; 5% tương đương với độ chính xác của con người khi đọc văn bản rõ ràng.

**CER (Tỷ lệ lỗi ký tự - Character Error Rate).** Cùng công thức nhưng tính ở cấp độ ký tự. Được áp dụng cho các ngôn ngữ có thanh điệu (như tiếng Quan Thoại, tiếng Quảng Đông) nơi ranh giới phân tách từ còn mơ hồ.

**RTFx (Hệ số thời gian thực nghịch đảo - Inverse real-time factor).** Số giây âm thanh được xử lý trên mỗi giây thời gian thực của đồng hồ. Càng cao càng tốt. Parakeet-TDT đạt tới 3380×. Whisper-large-v3 đạt khoảng ~30×.

**Độ trễ token đầu tiên (First-token latency).** Thời gian thực tính từ khi âm thanh đi vào hệ thống cho đến khi xuất hiện token bản gỡ băng đầu tiên. Đây là chỉ số sống còn đối với các hệ thống phát trực tuyến. Deepgram Nova-3: ~150 ms.

### Các số đo cho Tổng hợp Giọng nói (TTS)

**MOS (Điểm ý kiến trung bình - Mean Opinion Score).** Đánh giá của con người trên thang điểm từ 1 đến 5. Tiêu chuẩn vàng nhưng tốn kém và chậm. Cần tối thiểu 20 người nghe đánh giá trên mỗi mẫu, và hơn 100 mẫu cho mỗi mô hình.

**UTMOS (2022-2026).** Mô hình học máy dự đoán điểm MOS. Đạt mức tương quan ~0,9 với điểm MOS thực tế của con người trên các tập chuẩn đối sánh. F5-TTS đạt UTMOS 3,95; trong khi âm thanh gốc (ground truth) đạt 4,08.

**SECS (Độ tương đồng Cosine của bộ mã hóa người nói - Speaker Encoder Cosine Similarity).** Dành cho tác vụ nhân bản giọng nói. Tính độ tương đồng cosine giữa vector nhúng ECAPA của âm thanh mẫu tham chiếu và âm thanh nhân bản được tạo ra. Điểm &gt; 0,75 cho thấy giọng nói nhân bản nhận diện tốt.

**Vòng lặp WER-qua-ASR (WER-on-ASR-round-trip).** Chạy mô hình Whisper trên đầu ra của TTS, sau đó tính WER so với văn bản đầu vào ban đầu. Giúp phát hiện nhanh các hiện tượng suy giảm khả năng nghe hiểu. Mức SOTA năm 2026: &lt; 2% WER trên các tập kiểm thử tiếng Anh (các tập tiếng Quan Thoại báo cáo chỉ số CER).

**TTFA (Thời gian đến âm thanh đầu tiên - Time to First Audio).** Độ trễ thời gian thực. Kokoro-82M đạt ~100 ms; F5-TTS đạt ~1 s.

### Đánh giá chuyên biệt cho Nhân bản Giọng nói

**Bộ ba SECS + MOS + CER.** Một hệ thống nhân bản đạt điểm SECS cao nhưng MOS thấp đồng nghĩa với việc âm sắc đúng nhưng giọng nói thiếu tự nhiên; ngược lại, điểm MOS cao nhưng SECS thấp có nghĩa là giọng nói tự nhiên nhưng nhầm người nói.

### Xác minh Người nói (Speaker Verification)

**EER (Tỷ lệ sai số cân bằng - Equal Error Rate).** Ngưỡng mà tại đó Tỷ lệ Chấp nhận Sai (FAR) bằng Tỷ lệ Từ chối Sai (FRR). ECAPA trên tập VoxCeleb1-O đạt 0,87%.

**minDCF (Hàm chi phí phát hiện tối thiểu - Minimum Detection Cost).** Chi phí có trọng số tại một điểm vận hành được lựa chọn trước (thường đặt ở mức FAR = 0,01). Phản ánh thực tế triển khai sản xuất tốt hơn EER.

### Phân đoạn Người nói (Diarization)

**DER (Tỷ lệ lỗi phân đoạn người nói - Diarization Error Rate).** Công thức: `(FA + Miss + Confusion) / total_speaker_time`. Bao gồm tiếng nói bị bỏ sót (Missed speech) + báo động tiếng nói giả (False-alarm speech) + nhầm lẫn danh tính người nói (Speaker-confusion), mỗi thành phần được tính theo tỷ lệ phần trăm. Trên tập dữ liệu hội họp AMI: DER ở mức ~10-20% là khả thi trên thực tế. pyannote 3.1 kết hợp giải pháp thương mại Precision-2 đạt &lt;10% DER trên các bản ghi âm chất lượng tốt.

**JER (Tỷ lệ lỗi Jaccard - Jaccard Error Rate).** Phương án thay thế cho DER, ít bị ảnh hưởng bởi độ lệch phân đoạn ngắn.

### Phân loại Âm thanh (Audio Classification)

Phân loại đa nhãn (Multi-label): **mAP (Độ chính xác trung bình - Mean Average Precision)** trên tất cả các lớp. Trên AudioSet: đạt 0,548 mAP cho mô hình BEATs-iter3.

Phân loại đa lớp loại trừ lẫn nhau (Multi-class exclusive): **Độ chính xác top-1, top-5**. Trên Speech Commands v2: đạt 99,0% top-1 (Audio-MAE).

Dữ liệu mất cân bằng: **macro F1** + **độ thu hồi theo từng lớp (per-class recall)**. Luôn báo cáo chi tiết theo từng lớp — độ chính xác gộp chung thường che giấu các lớp bị dự đoán sai hoàn toàn.

### Sinh Âm nhạc (Music Generation)

**FAD (Khoảng cách âm thanh Fréchet - Fréchet Audio Distance).** Đo khoảng cách giữa các phân phối vector nhúng VGGish của âm thanh thật so với âm thanh sinh ra. MusicGen-small trên tập MusicCaps đạt 4,5. MusicLM đạt 4,0. Điểm càng thấp càng tốt.

**Điểm CLAP (CLAP Score).** Điểm căn chỉnh giữa văn bản mô tả và âm thanh bằng cách dùng vector nhúng CLAP. Mức &gt; 0,3 thể hiện sự căn chỉnh hợp lý.

**Điểm MOS từ hội đồng người nghe.** Vẫn là tiêu chuẩn phán quyết cuối cùng đối với âm nhạc cấp độ người dùng đại chúng. Suno v5 đạt điểm ELO 1293 trên bộ chuẩn so sánh sở thích ghép đôi của chính Suno. Đó là số liệu do nhà cung cấp công bố; TTS Arena xếp hạng tiếng nói, không xếp hạng âm nhạc.

### Chuẩn đánh giá Mô hình Ngôn ngữ - Âm thanh (Audio-Language)

**MMAU (Massive Multi-Audio Understanding).** Gồm 10.000 cặp câu hỏi - đáp âm thanh.

**MMAU-Pro.** Gồm 1.800 mục khó chia thành 4 danh mục: tiếng nói / âm thanh môi trường / âm nhạc / đa luồng âm thanh. Xác suất đoán mò ngẫu nhiên là 25% cho câu hỏi 4 lựa chọn. Gemini 2.5 Pro đạt tổng thể ~60%; nhưng danh mục đa luồng âm thanh chỉ đạt ~22% trên tất cả các mô hình.

**LongAudioBench.** Các đoạn âm thanh dài nhiều phút kèm câu hỏi truy vấn ngữ nghĩa. Audio Flamingo Next vượt qua Gemini 2.5 Pro.

**AudioCaps / Clotho.** Bộ chuẩn chú thích nội dung âm thanh. Đánh giá bằng các chỉ số SPICE, CIDEr, FENSE.

### Tiếng nói sang Tiếng nói Phát trực tuyến (Streaming S2S)

**Độ trễ P50 / P95 / P99.** Thời gian thực tế tính từ khi người dùng dứt lời đến khi nghe thấy phản hồi đầu tiên. Moshi đạt 200 ms; GPT-4o Realtime đạt 300 ms.

**WER / MOS** trên chất lượng âm thanh đầu ra.

**Độ nhạy khi bị chen ngang (Barge-in responsiveness).** Thời gian tính từ lúc người dùng ngắt lời cho đến khi trợ lý tắt tiếng. Mục tiêu &lt; 150 ms.

### Các bảng xếp hạng năm 2026

| Bảng xếp hạng | Nội dung theo dõi | Đường dẫn URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | Tiếng Anh + đa ngôn ngữ + dạng dài | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | TTS tiếng Anh | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, điểm ELO từ bình chọn cặp đôi | `artificialanalysis.ai/speech` |
| MMAU-Pro | Khả năng suy luận của LALM | `sonalkum.github.io/mmau-pro` |
| SpeakerBench / VoxSRC | Nhận diện người nói | `voxsrc.github.io` |
| Tập con âm nhạc MMAU | LALM cho âm nhạc | (nằm trong MMAU) |
| HEAR benchmark | Học tự giám sát âm thanh | `hearbenchmark.com` |

```figure
sp-wer-align
```

## Thực hành Xây dựng

### Bước 1: Tính WER có chuẩn hóa văn bản

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Bước 2: Vòng lặp WER qua ASR cho mô hình TTS

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Bước 3: Tính SECS cho nhân bản giọng nói

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Bước 4: Đo FAD cho sinh âm nhạc

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Bước 5: Đo EER cho xác minh người nói (Tương tự mã nguồn Bài 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Ứng dụng Thực tế

Gắn mỗi lần triển khai với một quy trình đánh giá cố định chạy tự động trên mọi bản cập nhật mô hình. Ba nguyên tắc cốt tử:

1. **Chuẩn hóa trước khi chấm điểm.** Chuyển chữ thường, bỏ dấu câu, mở rộng chữ số. Báo cáo rõ ràng quy tắc chuẩn hóa đã áp dụng.
2. **Báo cáo phân phối thay vì giá trị trung bình.** Sử dụng P50/P95/P99 cho độ trễ. Báo cáo độ thu hồi theo từng lớp cho bài toán phân loại. Báo cáo theo từng danh mục cho MMAU.
3. **Chạy trên ít nhất một bộ chuẩn công khai chuẩn tắc.** Ngay cả khi dữ liệu thực tế của bạn có sự khác biệt, việc báo cáo trên Open ASR / TTS Arena / MMAU giúp người đánh giá có thể so sánh công bằng trên cùng một thang đo.

## Cạm bẫy Thường gặp

- **Ngoại suy sai lệch của UTMOS.** UTMOS được huấn luyện trên giọng nói sạch kiểu VCTK; nó chấm điểm rất kém đối với âm thanh có tạp âm, âm thanh nhân bản hoặc giàu cảm xúc.
- **Thiên lệch hội đồng MOS.** 20 người đánh giá trên Amazon Mechanical Turk không đại diện cho 20 người dùng mục tiêu của bạn. Hãy thuê hội đồng chuyên môn trong ngành nếu dự án có rủi ro cao.
- **FAD phụ thuộc hoàn toàn vào tập tham chiếu.** Phải so sánh trên cùng một phân phối tham chiếu giữa các mô hình khác nhau.
- **WER gộp che giấu vấn đề.** Mức WER 5% toàn cục có thể đang che giấu mức WER lên tới 30% đối với các giọng nói có âm điệu địa phương (accented speech). Cần báo cáo phân tách theo từng nhóm nhân khẩu học.
- **Sự bão hòa của các bộ chuẩn công khai.** Hầu hết các mô hình tiên phong hiện nay đều tiệm cận mức trần trên các tập chuẩn đối sánh thông thường. Hãy tự xây dựng một tập kiểm thử nội bộ phản ánh đúng lưu lượng người dùng thực tế.

## Đóng gói Sản phẩm

Lưu thành `outputs/skill-audio-evaluator.md`. Lựa chọn số đo, bộ chuẩn đối sánh và định dạng báo cáo cho bất kỳ đợt phát hành mô hình âm thanh nào.

## Bài tập Thực hành

1. **Dễ.** Chạy `code/main.py`. Tính các chỉ số WER / CER / EER / SECS / FAD mô phỏng / MMAU mô phỏng trên dữ liệu đầu vào đơn giản.
2. **Trung bình.** Xây dựng quy trình đo WER vòng lặp cho mô hình TTS. Đưa đầu ra của Kokoro hoặc F5-TTS qua Whisper. Tính WER trên 50 câu lệnh mẫu. Gắn cờ cảnh báo các câu lệnh có WER &gt; 10%.
3. **Khó.** Chấm điểm mô hình LALM bạn chọn từ Bài 10 trên tập con tiếng nói và đa luồng âm thanh của MMAU-Pro (mỗi tập 50 câu). Báo cáo độ chính xác theo từng danh mục và so sánh với kết quả đã công bố.

## Thuật ngữ Then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| WER | Điểm ASR | `(S+D+I)/N` ở cấp độ từ sau khi chuẩn hóa văn bản. |
| CER | WER cấp ký tự | Dành cho ngôn ngữ có thanh điệu hoặc hệ thống nhận dạng cấp ký tự. |
| MOS | Ý kiến con người | Thang điểm 1-5; tối thiểu 20 người nghe × 100 mẫu thử. |
| UTMOS | Bộ dự đoán MOS bằng ML | Mô hình học máy; tương quan ~0,9 với điểm MOS của con người. |
| SECS | Độ tương đồng nhân bản giọng | Độ tương đồng cosine trên vector nhúng ECAPA giữa mẫu gốc và mẫu nhân bản. |
| EER | Điểm xác minh người nói | Ngưỡng cân bằng mà tại đó FAR = FRR. |
| DER | Điểm phân đoạn người nói | `(FA + Miss + Confusion) / tổng thời gian`. |
| FAD | Chất lượng sinh âm nhạc | Khoảng cách Fréchet tính trên các vector nhúng VGGish. |
| RTFx | Thông lượng xử lý | Số giây âm thanh xử lý được trong một giây thời gian thực. |

## Tài liệu Tham khảo

- [jiwer](https://github.com/jitsi/jiwer) — thư viện tính WER/CER với các tiện ích chuẩn hóa văn bản.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — bộ dự đoán MOS bằng học máy.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — tiêu chuẩn đánh giá mô hình sinh âm nhạc.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — bảng xếp hạng trực tiếp năm 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — bảng xếp hạng TTS dựa trên bình chọn của con người.
- [MMAU-Pro benchmark](https://sonalkum.github.io/mmau-pro/) — bảng xếp hạng khả năng suy luận của LALM.
- [HEAR benchmark](https://hearbenchmark.com/) — chuẩn đánh giá học tự giám sát âm thanh.
