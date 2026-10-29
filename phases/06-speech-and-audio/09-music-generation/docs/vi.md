# Tạo âm nhạc (Music Generation) — MusicGen, Stable Audio, Suno và Cơn địa chấn cấp phép bản quyền

> Tạo âm nhạc năm 2026: Suno v5 và Udio v4 thống trị mảng thương mại; MusicGen, Stable Audio Open và ACE-Step dẫn đầu mảng mã nguồn mở. Vấn đề kỹ thuật phần lớn đã được giải quyết. Vấn đề pháp lý (thỏa thuận dàn xếp 500 triệu USD của Warner Music, thỏa thuận dàn xếp với UMG) đã tái định hình toàn bộ lĩnh vực trong giai đoạn 2025–2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 phút

## Vấn đề đặt ra

Văn bản → một đoạn nhạc dài từ 30 giây đến 4 phút, hoàn chỉnh với lời bài hát, giọng hát (vocals) và cấu trúc bài hát. Ba bài toán con gồm:

1. **Tạo nhạc không lời (Instrumental generation).** Văn bản mô tả như "lo-fi hip-hop drums with warm keys" → âm thanh. MusicGen, Stable Audio, AudioLDM.
2. **Tạo bài hát hoàn chỉnh (kèm giọng hát + lời bài hát).** "Country song about rainy Texas nights" → trọn vẹn bài hát. Suno, Udio, YuE, ACE-Step.
3. **Có điều kiện / có khả năng kiểm soát (Conditional / controllable).** Kéo dài đoạn nhạc hiện có, tạo lại đoạn chuyển tiếp (bridge), chuyển đổi thể loại, tách bè/nhánh âm thanh (stem-separation), hoặc phục dựng/vẽ bù âm thanh (inpainting). Tính năng vẽ bù và tách bè của Udio là chuẩn mực cần hướng tới trong năm 2026.

## Khái niệm cốt lõi

![Tạo âm nhạc: token-LM so với mô hình khuếch tán, bản đồ mô hình năm 2026](../assets/music-generation.svg)

### Mô hình ngôn ngữ dựa trên token (Token LM) trên các token của bộ mã hóa nén nơ-ron

**MusicGen** của Meta (2023, giấy phép MIT) cùng nhiều biến thể: điều kiện hóa trên các vector nhúng văn bản hoặc giai điệu (melody embeddings), dự đoán tự hồi quy các token EnCodec (32 kHz, 4 tập mã - codebooks), sau đó giải mã bằng EnCodec. Kích thước từ 300M đến 3.3B tham số. Đây là đường cơ sở vững chắc; tuy nhiên gặp khó khăn khi độ dài vượt quá 30 giây.

**ACE-Step** (mã nguồn mở, phiên bản 4B XL phát hành tháng 4 năm 2026) mở rộng hướng tiếp cận này cho tác vụ tạo trọn vẹn bài hát có điều kiện theo lời. Đây là mô hình mã nguồn mở tiệm cận Suno nhất của cộng đồng.

### Mô hình khuếch tán (Diffusion) trên phổ mel hoặc không gian tiềm ẩn (latents)

**Stable Audio (2023)** và **Stable Audio Open (2024)**: khuếch tán trong không gian tiềm ẩn (latent diffusion) trên âm thanh đã nén. Cực kỳ xuất sắc trong việc tạo các vòng lặp (loops), thiết kế âm thanh (sound design) và kết cấu âm thanh môi trường (ambient textures). Không thực sự mạnh trong việc tạo các bài hát có cấu trúc hoàn chỉnh.

**AudioLDM / AudioLDM2**: chuyển văn bản thành âm thanh (text-to-audio) thông qua khuếch tán tiềm ẩn theo kiểu mô hình T2I (Text-to-Image), được khái quát hóa cho âm nhạc, hiệu ứng âm thanh (sound effects) và tiếng nói.

### Mô hình lai ghép (Hybrid trong môi trường thương mại) — Suno, Udio, Lyria

Mô hình trọng số đóng (closed weights). Nhiều khả năng kết hợp giữa mô hình ngôn ngữ codec tự hồi quy (AR codec LM) + vocoder dựa trên khuếch tán cùng các nhánh chuyên biệt (heads) cho giọng hát / bộ gõ / giai điệu. Suno v5 (2026) dẫn đầu về chất lượng với điểm ELO 1293. Udio v4 bổ sung tính năng vẽ bù vùng chọn (inpainting) + tách bè âm thanh (tải riêng từng dải âm: bass, trống, giọng hát).

### Đánh giá chất lượng

- **FAD (Fréchet Audio Distance).** Khoảng cách ở mức vector nhúng giữa phân phối âm thanh sinh ra và âm thanh thực tế, sử dụng đặc trưng từ VGGish hoặc PANNs. Giá trị càng thấp càng tốt. MusicGen-small đạt FAD 4.5 trên tập MusicCaps; mức đỉnh cao (SOTA) hiện đạt ~3.0.
- **Tính nhạc (Musicality - đánh giá chủ quan).** Đánh giá theo sở thích người nghe. Suno v5 dẫn đầu với điểm ELO 1293.
- **Mức độ gióng hàng giữa văn bản và âm thanh (Text-audio alignment).** Điểm số CLAP đo độ tương đồng giữa câu lệnh gợi ý (prompt) và âm thanh đầu ra.
- **Các khiếm khuyết về tính nhạc (Musicality artifacts).** Lệch nhịp khi chuyển đoạn, trôi cụm từ giọng hát, mất cấu trúc bài hát khi vượt quá 30 giây.

## Bản đồ mô hình năm 2026

| Mô hình | Tham số | Thời lượng | Giọng hát | Giấy phép |
|---|---|---|---|---|
| MusicGen-large | 3.3B | 30 giây | Không | MIT |
| Stable Audio Open | 1.2B | 47 giây | Không | Bản quyền phi thương mại của Stability |
| ACE-Step XL (Tháng 4/2026) | 4B | > 2 phút | Có | Apache-2.0 |
| YuE | 7B | > 2 phút | Có, đa ngôn ngữ | Apache-2.0 |
| Suno v5 (đóng) | ? | 4 phút | Có, ELO 1293 | Thương mại |
| Udio v4 (đóng) | ? | 4 phút | Có + tách bè | Thương mại |
| Google Lyria 3 (đóng) | ? | Thời gian thực | Có | Thương mại |
| MiniMax Music 2.5 | ? | 4 phút | Có | API thương mại |

## Bối cảnh pháp lý (Giai đoạn 2025–2026)

- **Thỏa thuận dàn xếp giữa Warner Music và Suno.** Trị giá 500 triệu USD. WMG hiện có quyền giám sát việc tái tạo hình mẫu AI (AI-likeness), quyền âm nhạc và các bản nhạc do người dùng tạo trên Suno. Thỏa thuận tương tự giữa UMG và Udio cũng đã diễn ra.
- **Đạo luật Trí tuệ nhân tạo của EU (EU AI Act)** + **Dự luật SB 942 của California**: Âm nhạc do AI tạo ra bắt buộc phải được công khai minh bạch.
- **Riffusion / MusicGen** phát hành dưới giấy phép MIT không vướng các gánh nặng pháp lý tuân thủ nhưng cũng không có sẵn giọng hát thương mại chất lượng cao.

Các mẫu hình triển khai an toàn cho sản xuất:

1. Chỉ tạo nhạc không lời (MusicGen, Stable Audio Open, đầu ra MIT/CC0).
2. Sử dụng API thương mại (Suno, Udio, ElevenLabs Music) với giấy phép bản quyền theo từng lượt tạo.
3. Huấn luyện trên kho dữ liệu thuộc sở hữu hoặc đã mua bản quyền (hầu hết các doanh nghiệp lớn đều chọn hướng đi này).
4. Đóng dấu bản quyền âm thanh (watermarks) + siêu dữ liệu (metadata) trên mọi bản nhạc được tạo ra.

````figure
sp-codec-tokens
````

## Tự xây dựng (Build It)

### Bước 1: Tạo âm nhạc với MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Có ba kích thước: `small` (300M, tốc độ nhanh), `medium` (1.5B), `large` (3.3B). Bản `small` là đủ để đánh giá xem "ý tưởng có thành hình hay không".

### Bước 2: Điều kiện hóa theo giai điệu (Melody conditioning)

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody tiếp nhận một biểu đồ màu âm sắc (chromagram), bảo toàn giai điệu trong khi hoán đổi âm sắc nhạc cụ. Rất hữu ích cho yêu cầu: "hãy diễn tấu giai điệu này bằng một bộ tứ đàn dây (string quartet)."

### Bước 3: Đánh giá bằng chỉ số FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Tính toán khoảng cách vector nhúng VGGish. Rất tiện lợi cho các bài kiểm thử hồi quy ở cấp độ thể loại nhạc; tuy nhiên không thể thay thế hoàn toàn đôi tai của người nghe thật.

### Bước 4: Tích hợp vào quy trình kết hợp LLM và Âm nhạc

Kết hợp với các ý tưởng từ Bài 7 và 8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Ứng dụng thực tế (Use It)

| Mục tiêu | Ngăn xếp công nghệ |
|---|---|
| Thiết kế âm thanh không lời | Stable Audio Open |
| Âm nhạc thích ứng cho trò chơi (Game/adaptive music) | Google Lyria RealTime (đóng) |
| Bài hát trọn vẹn có lời và giọng hát (Thương mại) | Suno v5 hoặc Udio v4 kèm giấy phép rõ ràng |
| Bài hát trọn vẹn có lời và giọng hát (Mã nguồn mở) | ACE-Step XL hoặc YuE |
| Đoạn nhạc ngắn cho quảng cáo (Jingle) | MusicGen điều kiện hóa giai điệu từ tiếng ngâm nga tham chiếu |
| Nhạc nền cho video ca nhạc | MusicGen + Stable Video Diffusion |

## Cạm bẫy vẫn xuất hiện trong năm 2026

- **Lời nhắc câu lệnh "rửa bản quyền" (Copyright-laundering prompts).** Các câu lệnh như "Bài hát theo phong cách Taylor Swift" — Suno/Udio thương mại hiện đã lọc các cụm từ này, nhưng các mô hình mã nguồn mở thì chưa. Bạn cần chủ động bổ sung danh sách bộ lọc cho riêng mình.
- **Hiện tượng lặp lại / trôi dạt khi bài hát dài hơn 30 giây.** Các mô hình tự hồi quy rất dễ rơi vào vòng lặp luẩn quẩn. Hãy dùng kỹ thuật làm mờ chéo (crossfade) giữa nhiều lần tạo, hoặc sử dụng ACE-Step để đảm bảo tính gắn kết về mặt cấu trúc bài hát.
- **Trôi nhịp độ (Tempo drift).** Mô hình thường bị chệch nhịp độ BPM ban đầu. Hãy gán nhãn BPM rõ ràng trong câu nhắc và kiểm tra lại bằng hàm `beat_track` của thư viện librosa.
- **Độ rõ của giọng hát (Vocal intelligibility).** Suno làm rất xuất sắc; các mô hình mã nguồn mở thường làm phát âm từ ngữ bị nhòe mờ. Nếu lời bài hát là yếu tố cốt lõi, hãy chọn API thương mại hoặc tiến hành tinh chỉnh mô hình.
- **Đầu ra âm thanh đơn kênh (Mono).** Các mô hình mã nguồn mở thường xuất âm thanh mono hoặc giả lập stereo. Hãy nâng cấp bằng giải pháp tái cấu trúc âm thanh stereo thực thụ (ezst, Cartesia stereo diffusion).

## Xuất xưởng sản phẩm (Ship It)

Lưu thành `outputs/skill-music-designer.md`. Lựa chọn mô hình, chiến lược cấp phép, kế hoạch cấu trúc/độ dài và siêu dữ liệu công khai minh bạch cho một hệ thống tạo nhạc thực tế.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py`. Chương trình sẽ tạo một chuỗi tiến trình hợp âm "tạo sinh" + mẫu trống dưới dạng các ký tự ASCII — một bản phác họa mô hình tạo nhạc. Bạn có thể xuất và phát lại bằng bất kỳ trình phát MIDI nào nếu muốn.
2. **Trung bình.** Cài đặt `audiocraft`, tạo các đoạn nhạc 10 giây qua 4 thể loại khác nhau với MusicGen-small, đo lường điểm FAD so với tập dữ liệu thể loại tham chiếu.
3. **Khó.** Sử dụng ACE-Step (hoặc MusicGen-melody), tạo ba biến thể khác nhau của cùng một giai điệu với các câu nhắc mô tả âm sắc khác biệt. Tính toán độ tương đồng CLAP so với câu nhắc để xác minh tính gióng hàng.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| FAD | Audio FID | Khoảng cách Fréchet giữa các phân phối vector nhúng của âm thanh thực và âm thanh được tạo ra. |
| Chromagram | Giai điệu dưới dạng cao độ | Vector 12 chiều trên từng khung hình; đầu vào cho điều kiện hóa giai điệu. |
| Stems (Bè âm thanh) | Các dải nhạc cụ riêng biệt | Các rãnh bass / trống / giọng hát / giai điệu được tách riêng dưới định dạng WAV. |
| Vẽ bù (Inpainting) | Phục dựng một đoạn | Đặt mặt nạ (mask) trên một khung thời gian; mô hình chỉ tái tạo riêng phân đoạn đó. |
| CLAP | Audio CLIP | Không gian nhúng tương phản giữa văn bản và âm thanh; dùng để đánh giá độ gióng hàng văn bản - âm thanh. |
| EnCodec | Bộ nén âm thanh nơ-ron | Bộ mã hóa âm thanh nơ-ron của Meta được MusicGen sử dụng; 32 kHz, 4 tập mã. |

## Tài liệu đọc thêm

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — Thước đo chuẩn mực cho mô hình tự hồi quy mã nguồn mở.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — Lựa chọn mặc định cho thiết kế âm thanh.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — Mô hình mã nguồn mở 4B tạo bài hát hoàn chỉnh, tháng 4 năm 2026.
- [Tài liệu nền tảng Suno v5](https://suno.com) — Người dẫn đầu về chất lượng thương mại.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — Mô hình khuếch tán tiềm ẩn cho âm nhạc + hiệu ứng âm thanh.
- [Thông tin về vụ dàn xếp WMG-Suno](https://www.musicbusinessworldwide.com/warner-music-group-settles-with-suno-strikes-first-of-its-kind-deal-with-ai-song-generator/) — Tiền lệ pháp lý tháng 11 năm 2025.
