# Mô hình ngôn ngữ âm thanh (Audio-Language Models) — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Các mô hình ngôn ngữ âm thanh năm 2026 có khả năng lập luận trên tiếng nói + âm thanh môi trường + âm nhạc. Qwen2.5-Omni-7B sánh ngang GPT-4o Audio trên thang đo MMAU-Pro. Audio Flamingo Next vượt qua Gemini 2.5 Pro trên LongAudioBench. Khoảng cách giữa mã nguồn mở và độc quyền về cơ bản đã bị xóa nhòa — ngoại trừ trên các tác vụ đa âm thanh (multi-audio tasks), nơi mọi mô hình đều đạt kết quả gần như ngẫu nhiên.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 phút

## Vấn đề đặt ra

Bạn có một đoạn âm thanh 5 giây: tiếng chó sủa, tiếng một người hét lên "dừng lại!", sau đó là sự im lặng. Các câu hỏi hữu ích trải rộng trên nhiều khía cạnh khác nhau:

- **Bản ghi lời thoại (Transcription).** "Họ vừa nói gì?" — Lãnh địa của nhận dạng tiếng nói tự động (ASR).
- **Lập luận ngữ nghĩa (Semantic reasoning).** "Người đó có đang gặp nguy hiểm không?" — Đòi hỏi sự hiểu biết phối hợp giữa tiếng chó sủa + tiếng hét + sự im lặng.
- **Hiểu biết âm nhạc (Music reasoning).** "Những nhạc cụ nào đang chơi giai điệu chính?"
- **Truy xuất âm thanh dài (Long-audio retrieval).** "Trong bài giảng dài 90 phút này, giảng viên đã giải thích về hạ gradient (gradient descent) ở đoạn nào?"

Một mô hình duy nhất có thể trả lời toàn bộ các câu hỏi trên chỉ với một câu nhắc được gọi là **mô hình ngôn ngữ âm thanh** (Large Audio-Language Model - LALM / ALM). Khác biệt với mô hình ASR thuần túy: LALM tạo ra câu trả lời bằng ngôn ngữ tự nhiên dạng tự do (free-form), chứ không chỉ đơn thuần là bản ghi chữ.

## Khái niệm cốt lõi

![Mô hình ngôn ngữ âm thanh: bộ mã hóa âm thanh + bộ chiếu + bộ giải mã LLM](../assets/alm-architecture.svg)

### Mẫu thiết kế gồm ba thành phần

Mọi mô hình LALM vào năm 2026 đều có chung một bộ khung xương:

1. **Bộ mã hóa âm thanh (Audio encoder).** Bộ mã hóa của Whisper · BEATs · CLAP · WavLM · hoặc một bộ mã hóa được tùy biến riêng cho từng mô hình.
2. **Bộ chiếu/bộ ánh xạ (Projector).** Lớp tuyến tính (linear) hoặc MLP kết nối các đặc trưng của bộ mã hóa âm thanh vào không gian nhúng token của mô hình ngôn ngữ lớn (LLM).
3. **Mô hình ngôn ngữ lớn (LLM).** Bộ giải mã dựa trên kiến trúc Llama / Qwen / Gemma. Tiếp nhận các token văn bản và token âm thanh xen kẽ nhau; sinh ra phản hồi bằng văn bản.

Quy trình huấn luyện:

- **Giai đoạn 1 (Stage 1).** Đóng băng bộ mã hóa + LLM; chỉ huấn luyện bộ chiếu trên dữ liệu ASR / sinh chú thích âm thanh (captioning data).
- **Giai đoạn 2 (Stage 2).** Tinh chỉnh toàn phần hoặc qua LoRA trên các tác vụ âm thanh tuân thủ chỉ dẫn (Hỏi đáp - QA, lập luận ngữ nghĩa, thấu hiểu âm nhạc).
- **Giai đoạn 3 (Tùy chọn - Stage 3).** Khả năng nhận giọng nói / phát giọng nói (voice-in / voice-out) bổ sung thêm một bộ giải mã tiếng nói (speech decoder). Qwen2.5-Omni và AF3-Chat áp dụng phương pháp này.

### Bản đồ mô hình năm 2026

| Mô hình | Mô hình nền tảng | Bộ mã hóa âm thanh | Phương thức đầu ra | Quyền truy cập |
|---|---|---|---|---|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Tùy biến + Whisper | Văn bản + Tiếng nói | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Tùy biến | Văn bản + Tiếng nói | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | Văn bản | Giấy phép phi thương mại của NVIDIA |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | Văn bản | Giấy phép phi thương mại của NVIDIA |
| SALMONN | Vicuna | Whisper + BEATs | Văn bản | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | Văn bản | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | Văn bản | Apache-2.0 |
| Gemini 2.5 Flash/Pro (đóng) | Gemini | Độc quyền | Văn bản + Tiếng nói | API |
| GPT-4o Audio (đóng) | GPT-4o | Độc quyền | Văn bản + Tiếng nói | API |

### Thực tế điểm chuẩn đánh giá (Năm 2026)

**MMAU-Pro.** 1800 cặp câu hỏi - đáp bao quát các mảng tiếng nói / âm thanh môi trường / âm nhạc / hỗn hợp. Bao gồm cả tập con so sánh đa tệp âm thanh (multi-audio).

| Mô hình | Tổng thể | Tiếng nói | Âm thanh | Âm nhạc | Đa âm thanh |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA trên LongAudioBench | — | — | — | — |

Cột **đa âm thanh (multi-audio) là một hồi chuông cảnh tỉnh cho mọi mô hình.** Xác suất chọn ngẫu nhiên đối với câu hỏi trắc nghiệm 4 đáp án là 25%; và hầu hết các mô hình đều ghi điểm quanh mức này. LALM hiện vẫn gặp rất nhiều khó khăn khi phải đối chiếu, so sánh giữa hai đoạn âm thanh khác nhau.

### Những trường hợp LALM phát huy hiệu quả trong năm 2026

- **Kiểm toán tuân thủ các cuộc gọi trung tâm chăm sóc khách hàng (Call-center).** "Nhân viên hỗ trợ có nhắc đến điều khoản miễn trừ trách nhiệm bắt buộc hay không?"
- **Hỗ trợ người khiếm thính (Accessibility).** Miêu tả các sự kiện âm thanh xung quanh cho người dùng khiếm thính (không chỉ là chép lời thoại).
- **Kiểm duyệt nội dung (Content moderation).** Phát hiện ngôn từ bạo lực kết hợp tông giọng đe dọa và bối cảnh âm thanh môi trường.
- **Phân chia chương đoạn Podcast / cuộc họp.** Tóm tắt theo ngữ nghĩa nội dung thay vì chỉ dựa vào lượt nói của từng người.
- **Phân tích danh mục bài hát.** "Tìm tất cả các bản nhạc có chuyển điệu (key change) ở phân đoạn B."

### Những trường hợp LALM CHƯA phát huy hiệu quả

- Lý thuyết âm nhạc chuyên sâu ở cấp độ chi tiết (dưới cấp độ hợp âm).
- Lập luận phân định theo người nói trong các cuộc hội thoại dài (chất lượng suy giảm mạnh sau 10 phút).
- So sánh đối chiếu đa âm thanh (mức 22-26% chỉ tương đương đoán mò).
- Lập luận phát luồng thời gian thực (hầu hết các mô hình hiện tại đều là suy diễn theo lô ngoại tuyến - offline batch).

````figure
v4-alm-tokens
````

## Tự xây dựng (Build It)

### Bước 1: Truy vấn mô hình Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Bước 2: Mẫu thiết kế bộ chiếu (Projector Pattern)

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Chỉ đơn giản như vậy. Bộ chiếu thông thường gồm 1 đến 3 lớp tuyến tính. Huấn luyện bộ chiếu này trên các cặp dữ liệu ASR (âm thanh → văn bản lời thoại) chính là tác vụ tiền đề (pretext task) của Giai đoạn 1.

### Bước 3: Đánh giá điểm chuẩn MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("gamma-lab-umd/MMAU-Pro", split="test")
mcq = mmau.filter(lambda item: len(item["choices"] or []) > 1)

correct = 0
for item in mcq:
    answer = call_model(item["audio_path"], item["question"], item["choices"])
    if answer == item["answer"]:
        correct += 1
print(f"Accuracy: {correct / len(mcq):.3f}")
```

Trường `audio_path` trỏ vào tệp `data.zip` trong kho dữ liệu (khoảng 47 GB), vì vậy bạn cần tải xuống và giải nén trước khi chấm điểm. Vòng lặp so khớp chính xác (exact-match) này chỉ là bước kiểm tra cơ bản (sanity check), không phải bộ chấm điểm chính thức của điểm chuẩn, do đó kết quả này không thể so sánh trực tiếp với các số liệu MMAU-Pro đã công bố. Bộ đánh giá chính thức so khớp các câu trả lời trắc nghiệm bằng độ tương đồng vector nhúng (NV-Embed-v2), chấm điểm các câu trả lời dạng mở bằng giám khảo LLM, và kiểm tra các câu trả lời tuân thủ chỉ dẫn bằng quy tắc biểu thức chính quy (regex): ghi các dự đoán vào cột `model_output` và chạy tập lệnh `evaluate_mmau_pro_comprehensive.py` từ [kho lưu trữ MMAU-Pro](https://github.com/sonalkum/MMAUPro). Hãy báo cáo riêng từng danh mục `category` (tiếng nói, âm thanh môi trường, âm nhạc, đa âm thanh và các danh mục còn lại). Các số liệu tổng hợp chung thường che giấu những điểm yếu thực sự của mô hình.

## Ứng dụng thực tế (Use It)

| Tác vụ | Lựa chọn năm 2026 |
|---|---|
| Hỏi đáp âm thanh dạng tự do (Mã nguồn mở) | Qwen2.5-Omni-7B |
| Mô hình mã nguồn mở tốt nhất trên âm thanh dài | Audio Flamingo Next |
| Mô hình độc quyền tốt nhất | Gemini 2.5 Pro |
| Tác nhân đàm thoại trực tiếp (Voice-in / voice-out) | Qwen2.5-Omni hoặc GPT-4o Audio |
| Lập luận âm nhạc | Audio Flamingo 3 hoặc 2 (AF-CLAP chuyên biệt cho âm nhạc) |
| Kiểm toán trung tâm cuộc gọi | Gemini 2.5 Pro qua API, kết hợp RAG trên tài liệu chính sách của bạn |

## Cạm bẫy thường gặp

- **Quá tin tưởng vào tác vụ đa âm thanh.** Nếu tác vụ của bạn đòi hỏi "đoạn âm thanh nào chứa sự kiện X", độ chính xác thực tế chỉ ngang mức đoán mò ngẫu nhiên.
- **Suy giảm chất lượng trên âm thanh dài.** Vượt quá 10 phút, khả năng gán đúng người nói của hầu hết mô hình đều bị hỏng. Hãy phân đoạn người nói (diarize) trước (Bài 6), sau đó mới đưa vào tóm tắt.
- **Hiện tượng ảo giác trên các đoạn im lặng.** Vấn đề cố hữu của Whisper được kế thừa bởi các mô hình LALM sử dụng bộ mã hóa của Whisper. Cần sử dụng cổng phát hiện giọng nói (VAD-gate).
- **Hiện tượng chọn lọc điểm chuẩn có lợi (Cherry-picking).** Các bài viết quảng bá của nhà cung cấp thường nhấn mạnh vào các danh mục có kết quả tốt nhất. Hãy tự mình kiểm thử trên tập con đa âm thanh của MMAU-Pro.

## Xuất xưởng sản phẩm (Ship It)

Lưu thành `outputs/skill-alm-picker.md`. Lựa chọn LALM + tập con điểm chuẩn + phương thức đầu ra (văn bản hay tiếng nói) cho một tác vụ thấu hiểu âm thanh được chỉ định.

## Bài tập thực hành

1. **Dễ.** Chạy `code/main.py` để quan sát mẫu thiết kế bộ chiếu cơ bản + cơ chế định tuyến LALM mô phỏng từ (vector nhúng âm thanh, token văn bản) → token đầu ra.
2. **Trung bình.** Chấm điểm mô hình Qwen2.5-Omni-7B trên 100 mẫu tiếng nói thuộc tập MMAU-Pro. So sánh với con số được báo cáo trong bài báo khoa học.
3. **Khó.** Xây dựng một mô hình đường cơ sở tối thiểu cho tác vụ sinh chú thích âm thanh: bộ mã hóa BEATs + bộ chiếu 2 lớp + Llama-3.2-1B đóng băng. Chỉ tinh chỉnh bộ chiếu trên tập dữ liệu AudioCaps. So sánh kết quả với SALMONN trên tập Clotho-AQA.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| LALM | ChatGPT cho âm thanh | Bộ mã hóa âm thanh + bộ chiếu + bộ giải mã LLM. |
| Bộ chiếu (Projector) | Adapter | Mạng MLP nhỏ ánh xạ đặc trưng âm thanh vào không gian nhúng của LLM. |
| MMAU | Điểm chuẩn âm thanh | 10 nghìn cặp câu hỏi-đáp âm thanh bao gồm tiếng nói, âm thanh môi trường và âm nhạc. |
| MMAU-Pro | MMAU mức độ khó hơn | 1800 câu hỏi phức tạp yêu cầu so sánh đa âm thanh hoặc lập luận nâng cao. |
| LongAudioBench | Đánh giá âm thanh dài | Các đoạn âm thanh dài nhiều phút với các câu truy vấn ngữ nghĩa. |
| Nhận/phát giọng nói (Voice-in / voice-out) | Bản địa tiếng nói (Speech-native) | Mô hình tiếp nhận trực tiếp tiếng nói và phát ra tiếng nói mà không cần chuyển trung gian qua văn bản. |

## Tài liệu đọc thêm

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — Kiến trúc tham chiếu chuẩn mực.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — Mô hình đầu vào tiếng nói - đầu ra tiếng nói.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — Người dẫn đầu mã nguồn mở về xử lý âm thanh dài.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — Đạt đỉnh cao SOTA trên bảng xếp hạng LongAudioBench.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — Tiên phong trong kiến trúc bộ mã hóa kép.
- [Bảng xếp hạng MMAU-Pro](https://sonalkum.github.io/mmau-pro/) — Bảng xếp hạng trực tiếp năm 2026.
