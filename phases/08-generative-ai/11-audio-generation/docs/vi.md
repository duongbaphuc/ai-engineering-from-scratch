# Tạo âm thanh (Audio Generation)

> Âm thanh là một tín hiệu 1 chiều (1-D signal) ở tần số 16-48 kHz. Một đoạn âm thanh dài 5 giây chứa từ 80k đến 240k mẫu (samples). Không có mô hình transformer nào có thể tính toán cơ chế chú ý (attention) trực tiếp trên chuỗi dài như vậy. Giải pháp chung cho mọi mô hình âm thanh triển khai thực tế (production) vào năm 2026 đều thống nhất: một bộ nén-giải nén nơ-ron (neural codec như Encodec, SoundStream, DAC) nén dạng sóng âm thanh thành các token rời rạc ở tần số 50-75 Hz, sau đó một mô hình transformer hoặc mô hình khuếch tán (diffusion model) sẽ đảm nhận việc tạo ra các token này.

**Type:** Xây dựng (Build)  
**Languages:** Python  
**Prerequisites:** Phase 6 · 02 (Đặc trưng âm thanh - Audio Features), Phase 6 · 04 (Nhận dạng giọng nói tự động - ASR), Phase 8 · 06 (Mô hình xác suất khuếch tán khử nhiễu - DDPM)  
**Time:** ~45 phút  

## Vấn đề đặt ra (The Problem)

Có ba tác vụ tạo âm thanh (audio generation) chính:

1. **Chuyển văn bản thành giọng nói (Text-to-speech - TTS).** Đưa văn bản vào, tạo ra tiếng nói. Tiếng nói sạch có dải tần hẹp và cấu trúc ngữ âm (phonetic structure) rất chặt chẽ — được giải quyết hiệu quả bằng transformer xử lý trên chuỗi token. Đại diện tiêu biểu: VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Tạo âm nhạc (Music generation).** Đưa câu lệnh gợi ý (văn bản, giai điệu, tiến trình hợp âm, thể loại), tạo ra bản nhạc hoàn chỉnh. Tác vụ này có phân phối dữ liệu rộng hơn rất nhiều. Đại diện: MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Hiệu ứng âm thanh / Thiết kế âm thanh (Audio effects / sound design).** Đưa câu lệnh gợi ý, tạo ra âm thanh môi trường xung quanh (ambient sound) hoặc hiệu ứng âm thanh động (Foley). Đại diện: AudioGen, AudioLDM 2, Stable Audio Open.

Cả ba tác vụ đều vận hành trên cùng một nền tảng kiến trúc: bộ giải mã âm thanh nơ-ron (neural audio codec) kết hợp với bộ sinh tự hồi quy token (token-AR) hoặc mô hình khuếch tán (diffusion generator).

## Khái niệm cốt lõi (The Concept)

![Tạo âm thanh: token codec + transformer hoặc diffusion](../assets/audio-generation.svg)

### Bộ mã hóa-giải mã âm thanh nơ-ron (Neural audio codecs)

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Một bộ mã hóa tích chập (convolutional encoder) nén dạng sóng (waveform) thành một vector biểu diễn tại mỗi bước thời gian; kỹ thuật lượng tử hóa vector phần dư (residual vector quantization - RVQ) chuyển đổi mỗi vector thành một chuỗi phân tầng gồm $K$ chỉ số từ điển mã (codebook indices). Bộ giải mã (decoder) thực hiện quá trình ngược lại. Âm thanh tần số 24 kHz ở tốc độ 2 kbps sử dụng 8 từ điển mã RVQ tại tần số 75 Hz sẽ tương đương 600 token/giây.

```text
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Hai mô hình sinh bổ trợ phía trên (Two generative paradigms on top)

**Tự hồi quy cấp token (Token-autoregressive - AR).** Làm phẳng (flatten) các token RVQ thành một chuỗi tuần tự, sau đó đưa qua transformer chỉ chứa bộ giải mã (decoder-only transformer). MusicGen sử dụng cơ chế "song song trễ" (delayed parallel) để phát ra $K$ luồng từ điển mã đồng thời với độ lệch pha (offset) riêng cho từng luồng. VALL-E tạo ra các token tiếng nói từ câu lệnh văn bản kết hợp với một mẫu giọng nói tham chiếu dài 3 giây.

**Khuếch tán trong không gian tiềm ẩn (Latent diffusion).** Đóng gói các token của codec thành các biểu diễn tiềm ẩn liên tục (continuous latents) hoặc mô hình hóa trực tiếp bằng khuếch tán phân loại (categorical diffusion). Stable Audio 2.5 sử dụng kỹ thuật so khớp dòng (flow matching) trên các không gian tiềm ẩn âm thanh liên tục. AudioLDM 2 áp dụng mô hình khuếch tán chuyển từ văn bản sang phổ mel rồi thành âm thanh (text-to-mel-to-audio).

Xu hướng giai đoạn 2024-2026: so khớp dòng (flow matching) đang chiếm ưu thế trong lĩnh vực tạo âm nhạc (tốc độ suy luận nhanh hơn, chất lượng mẫu sạch hơn), trong khi mô hình tự hồi quy token vẫn thống trị trong việc tạo giọng nói nhờ bản chất nhân quả (causal) tự nhiên và khả năng phát trực tiếp (streaming) xuất sắc.

## Bức tranh thực tế trong sản xuất (Production landscape)

| Hệ thống | Tác vụ | Xương sống (Backbone) | Độ trễ (Latency) |
|----------|--------|-----------------------|------------------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms cho token đầu tiên |
| OpenAI GPT-4o audio | Giọng nói song công toàn phần (Full-duplex speech) | AR đa phương thức đầu-cuối (End-to-end multimodal AR) | ~200ms |
| NaturalSpeech 3 | TTS | So khớp dòng tiềm ẩn (Latent flow matching) | Không phát luồng (Non-streaming) |
| Stable Audio 2.5 | Âm nhạc / Hiệu ứng SFX | DiT + flow matching trên latent âm thanh | ~10 giây cho đoạn 1 phút |
| Suno v4 | Bài hát hoàn chỉnh | Chưa công bố; nghi vấn dùng token-AR | ~30 giây mỗi bài |
| Udio v1.5 | Bài hát hoàn chỉnh | Chưa công bố | ~30 giây mỗi bài |
| MusicGen 3.3B | Âm nhạc | Token-AR trên Encodec 32kHz | Thời gian thực (Real-time) |
| AudioCraft 2 | Âm nhạc + SFX | So khớp dòng (Flow matching) | ~5 giây cho đoạn 5 giây |
| Riffusion v2 | Âm nhạc | Khuếch tán biểu đồ phổ (Spectrogram diffusion) | ~10 giây |

````figure
score-matching
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` mô phỏng ý tưởng cốt lõi: huấn luyện một transformer dự đoán token kế tiếp siêu nhỏ trên các chuỗi "token âm thanh" tổng hợp được tạo từ hai "phong cách" (styles) khác nhau (các token cao và thấp xen kẽ cho phong cách A, dốc tăng dần đơn điệu cho phong cách B). Điều kiện hóa theo phong cách và tiến hành lấy mẫu.

### Bước 1: Tạo token âm thanh tổng hợp (Step 1: synthetic audio tokens)

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "dạng tiếng nói": xen kẽ luân phiên
        return [i % vocab_size for i in range(length)]
    # "dạng âm nhạc": dốc tăng dần
    return [(i * 3) % vocab_size for i in range(length)]
```

### Bước 2: Huấn luyện bộ dự đoán token siêu nhỏ (Step 2: train a tiny token predictor)

Một bộ dự đoán dạng bigram có điều kiện theo phong cách. Điểm cốt lõi là nắm vững quy trình mẫu: token codec → huấn luyện bằng hàm mất mát cross-entropy → lấy mẫu tự hồi quy (autoregressive sampling).

### Bước 3: Lấy mẫu có điều kiện (Step 3: sample conditionally)

Nhận vào token phong cách và một token khởi đầu, tiến hành lấy mẫu token kế tiếp từ phân phối dự đoán được. Tiếp tục sinh chuỗi từ 20 đến 40 token.

## Cạm bẫy thực tế (Pitfalls)

- **Chất lượng codec là giới hạn trần của chất lượng đầu ra.** Nếu codec không thể tái tạo lại âm thanh một cách trung thực, thì bất kể mô hình sinh có tốt đến đâu cũng không thể khắc phục được. DAC hiện là mô hình mã nguồn mở tốt nhất.
- **Tích tụ sai số trong RVQ (RVQ error accumulation).** Mỗi tầng RVQ mô hình hóa phần dư của tầng trước đó. Sai số ở tầng 1 sẽ lan truyền sang các tầng sau. Lấy mẫu với nhiệt độ (temperature) bằng 0 ở các tầng cao hơn sẽ giảm thiểu hiện tượng này.
- **Cấu trúc âm nhạc dài hạn.** 30 giây token tương đương hơn 20k token ở tần số 75 Hz. Đây là thách thức lớn đối với kiến trúc transformer. MusicGen sử dụng cửa sổ trượt (sliding window) kết hợp nối tiếp câu lệnh gợi ý; Stable Audio tạo các đoạn ngắn hơn và dùng kỹ thuật trộn mờ chồng lấn (crossfading).
- **Hiện tượng méo biên (Artifacts at boundaries).** Trộn mờ chồng lấn giữa các đoạn sinh ngắt quãng đòi hỏi thuật toán overlap-add được tinh chỉnh kỹ lưỡng.
- **Nhu cầu dữ liệu sạch khổng lồ.** Các mô hình tạo âm nhạc đòi hỏi hàng chục nghìn giờ âm nhạc có bản quyền đầy đủ. Vụ kiện vi phạm bản quyền từ RIAA chống lại Suno và Udio (2024) đã phơi bày rủi ro pháp lý này.
- **Đạo đức nhân bản giọng nói (Voice cloning ethics).** Một đoạn ghi âm ngắn 3 giây kèm câu lệnh văn bản là đủ để VALL-E / XTTS / ElevenLabs nhân bản giọng nói của một người. Mọi mô hình sản xuất thực tế đều cần hệ thống phát hiện lạm dụng và danh sách chặn (opt-out list).

## Ứng dụng thực tế (Use It)

| Tác vụ | Ngăn xếp công nghệ 2026 (2026 stack) |
|--------|---------------------------------------|
| TTS thương mại | ElevenLabs, OpenAI TTS, hoặc Azure Neural |
| Nhân bản giọng nói (đã xác thực đồng thuận) | XTTS v2 (mã nguồn mở) hoặc ElevenLabs Pro |
| Nhạc nền, tốc độ nhanh | Stable Audio 2.5 API, Suno, hoặc Udio |
| Nhạc có lời | Suno v4 hoặc Udio v1.5 |
| Hiệu ứng âm thanh / Foley | AudioCraft 2, ElevenLabs SFX, hoặc Stable Audio Open |
| Trợ lý giọng nói thời gian thực | GPT-4o realtime hoặc Gemini Live |
| Nghiên cứu âm nhạc trọng số mở | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Lồng tiếng / Dịch thuật | HeyGen, ElevenLabs Dubbing |

## Đóng gói sản phẩm (Ship It)

Lưu tệp `outputs/skill-audio-brief.md`. Kỹ năng này nhận vào một bản yêu cầu âm thanh (nhiệm vụ, thời lượng, phong cách, giọng đọc/hát, bản quyền) và đưa ra: mô hình + hạ tầng lưu trữ (hosting), định dạng prompt (thẻ thể loại, bộ mô tả phong cách, đánh dấu cấu trúc), chuỗi kết hợp codec + generator + vocoder, quy trình đặt seed, và kế hoạch đánh giá (MOS / điểm CLAP / CER cho TTS / thử nghiệm A/B trên người dùng).

## Bài tập thực hành (Exercises)

1. **Dễ.** Chạy `code/main.py` và chỉ định phong cách một cách tường minh. Xác minh các chuỗi được sinh ra khớp với mẫu quy luật của phong cách đó.
2. **Trung bình.** Bổ sung cơ chế giải mã song song trễ (delayed parallel decoding): mô phỏng 2 luồng token phải duy trì độ lệch pha 1 bước thời gian. Huấn luyện một bộ dự đoán chung.
3. **Khó.** Sử dụng thư viện HuggingFace `transformers` để chạy mô hình MusicGen-small trên máy cục bộ. Tạo một đoạn âm thanh dài 10 giây với ba câu lệnh gợi ý khác nhau; tiến hành đánh giá A/B về mức độ tuân thủ phong cách.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Codec | "Nén nơ-ron" (Neural compression) | Bộ mã hóa / giải mã cho âm thanh; đầu ra điển hình là các token ở tần số 50-75 Hz. |
| RVQ | "Lượng tử hóa vector phần dư" (Residual VQ) | Chuỗi gồm $K$ bộ lượng tử hóa phân tầng; mỗi bộ mô hình hóa phần dư của bộ đứng trước. |
| Token | "Một biểu tượng codec" (One codec symbol) | Chỉ số rời rạc trỏ vào từ điển mã (codebook); thường có kích thước 1024 hoặc 2048. |
| Delayed parallel | "Lệch pha từ điển mã" (Offset codebooks) | Phát ra $K$ luồng token với các độ trễ so le để giảm chiều dài chuỗi tổng thể. |
| Flow matching | "Bước tiến 2024 cho âm thanh" | Phương pháp thay thế khuếch tán với đường đi thẳng hơn; tốc độ lấy mẫu nhanh hơn. |
| Voice prompt | "Mẫu âm 3 giây" | Biểu diễn nhúng người nói (speaker embedding) hoặc tiền tố token giúp định hướng giọng nói nhân bản. |
| Mel spectrogram | "Biểu đồ phổ thị giác" | Biểu đồ phổ log biên độ dựa trên cảm nhận thính giác; được nhiều hệ thống TTS sử dụng. |
| Vocoder | "Từ mel sang sóng âm" | Thành phần nơ-ron chịu trách nhiệm chuyển đổi biểu đồ phổ mel ngược lại thành sóng âm thanh. |

## Lưu ý triển khai: Âm thanh là bài toán xử lý luồng (Production note: audio is a streaming problem)

Âm thanh là phương thức đầu ra duy nhất mà người dùng kỳ vọng sẽ nhận được *ngay khi nó đang được tạo ra*, thay vì chờ đợi toàn bộ khối dữ liệu hoàn tất. Trong môi trường triển khai thực tế, điều này có nghĩa là chỉ số TPOT (Thời gian trên mỗi token đầu ra - Time Per Output Token) đóng vai trò sống còn, bởi tốc độ nghe của người dùng chính là lưu lượng mục tiêu — chứ không phải tốc độ đọc. Đối với âm thanh tần số 16 kHz được mã hóa ở mức ~75 token/giây (Encodec), máy chủ bắt buộc phải sinh được $\ge 75$ token/giây cho mỗi người dùng để đảm bảo quá trình phát lại mượt mà, không gián đoạn.

Hai hệ quả kiến trúc quan trọng:

- **Các mô hình âm thanh so khớp dòng (flow-matching) không thể phát luồng một cách trực tiếp.** Stable Audio 2.5 và AudioCraft 2 kết xuất một đoạn âm thanh có thời lượng cố định trong một lượt xử lý duy nhất. Để truyền phát dạng luồng, bạn phải chia đoạn âm thanh thành các phần nhỏ (chunks) và xử lý chồng lấn tại các biên (tương tự kỹ thuật khuếch tán cửa sổ trượt - sliding-window diffusion), điều này làm tăng thêm từ 100-300ms độ trễ chi phí (latency overhead) so với mô hình AR dùng codec.

Nếu sản phẩm của bạn là "trò chuyện thoại trực tiếp" (live voice chat) hoặc "tiếp nối âm nhạc thời gian thực", hãy chọn hướng đi AR với codec. Nếu sản phẩm là "kết xuất đoạn âm thanh 30 giây khi bấm gửi", thì so khớp dòng (flow-matching) sẽ áp đảo về mặt chất lượng âm thanh và tổng thời gian tạo.

## Tài liệu đọc thêm (Further Reading)

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — tiêu chuẩn nền tảng cho codec âm thanh nơ-ron.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — bộ mã hóa-giải mã âm thanh nơ-ron đầu tiên được áp dụng rộng rãi.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2025). Stable Audio 2.5](https://stability.ai/news-updates/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale) — Mô hình tạo nhạc từ văn bản ứng dụng so khớp dòng (2025).
