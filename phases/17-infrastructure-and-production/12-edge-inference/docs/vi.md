# Suy luận trên thiết bị biên — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Ràng buộc cốt lõi trên thiết bị biên (edge) nằm ở băng thông bộ nhớ (memory bandwidth), chứ không phải năng lực tính toán thuần túy. Bộ nhớ DRAM di động chỉ đạt mức 50–90 GB/s; trong khi bộ nhớ HBM3 tại các trung tâm dữ liệu vượt mốc 2–3 TB/s — tạo ra khoảng cách chênh lệch từ 30 đến 50 lần. Quá trình giải mã (decode) bị giới hạn bởi băng thông bộ nhớ (memory-bound), do đó khoảng cách này mang tính quyết định tuyệt đối. Bức tranh công nghệ năm 2026 chia làm bốn hướng tiếp cận chính: Apple M4/A18 Neural Engine đạt đỉnh 38 TOPS với kiến trúc bộ nhớ thống nhất (unified memory — loại bỏ hoàn toàn chi phí sao chép giữa CPU và NPU). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon chạm mốc 45 TOPS. WebGPU + WebLLM chạy mô hình Llama 3.1 8B (Q4) đạt ~41 tok/s trên chip M3 Max (đạt khoảng 70–80% hiệu năng gốc); đạt 17,6k lượt gắn sao trên GitHub, hỗ trợ API tương thích OpenAI, và độ phủ thiết bị di động đạt ~70–75%. NVIDIA Jetson Orin Nano Super (8GB) vận hành mượt mà Llama 3.2 3B / Phi-3; AGX Orin chạy gpt-oss-20b thông qua vLLM ở mức ~40 tok/s; Jetson T4000 (JetPack 7.1) có hiệu năng gấp 2 lần AGX Orin. TensorRT Edge-LLM hỗ trợ EAGLE-3, NVFP4, tiền điền phân mảnh (chunked prefill) — đã được trình diễn tại CES 2026 bởi Bosch, ThunderSoft và MediaTek.

**Phân loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng giải mã bị giới hạn bởi băng thông bộ nhớ)
**Điều kiện tiên quyết:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 09 (Production Quantization)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do suy luận LLM trên thiết bị di động bị giới hạn bởi băng thông bộ nhớ và năng lực tính toán chỉ đóng vai trò thứ yếu.
- Liệt kê bốn nền tảng biên mục tiêu (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) và gắn kết từng nền tảng với một trường hợp sử dụng phù hợp.
- Nêu rõ khoảng cách độ phủ WebGPU năm 2026 (Firefox trên Android đang trong quá trình hoàn thiện) cùng cột mốc cập nhật chính thức trên Safari iOS 26.
- Lựa chọn định dạng lượng tử hóa tương thích cho từng nền tảng đích (Core ML INT4 + FP16 cho ANE, QNN INT8/INT4 cho Hexagon, WebGPU Q4 cho trình duyệt web, NVFP4 cho Jetson Thor).

## Vấn đề

Một khách hàng yêu cầu xây dựng một ứng dụng chatbot chạy trực tiếp trên thiết bị (on-device): tương tác giọng nói thời gian thực, bảo đảm quyền riêng tư mặc định, hoạt động hoàn toàn ngoại tuyến (offline). Trên MacBook Pro M3 Max, mô hình Llama 3.1 8B Q4 chạy mượt mà ở mức ~55 tok/s — rất tốt. Nhưng trên iPhone 16 Pro, chính mô hình đó chỉ lết được 3 tok/s — hoàn toàn không thể chấp nhận được. Trên một thiết bị Android tầm trung chạy Snapdragon 8 Gen 3, tốc độ đạt 7 tok/s. Còn chạy ngay trong trình duyệt qua WebGPU trên Chrome Android từ phiên bản v121 trở lên, tốc độ dao động từ 4 đến 8 tok/s tùy máy.

Sự biến thiên thông lượng khổng lồ này không phải là vấn đề chuyển đổi mã nguồn (porting issue). Nó là tích số của khoảng cách băng thông bộ nhớ nhân với định dạng lượng tử hóa nhân với việc liệu NPU có thể truy cập được từ không gian người dùng (user-space) hay không. Suy luận trên thiết bị biên vào năm 2026 thực chất là bốn bài toán hoàn toàn riêng biệt với bốn giải pháp kỹ thuật đặc thù.

## Khái niệm

### Băng thông bộ nhớ mới là giới hạn trần thực sự

Pha giải mã (decode) bắt buộc phải đọc toàn bộ tập trọng số của mô hình cho mỗi token được sinh ra. Một mô hình 7B ở định dạng Q4 chiếm 3,5 GB dung lượng. Đọc 3,5 GB qua băng thông 50 GB/s tiêu tốn 70 ms — tương đương mức trần lý thuyết chỉ đạt khoảng ~14 tok/s. Ở băng thông 90 GB/s (chuẩn DRAM di động cao cấp), mức trần tăng lên ~25 tok/s. Bất kể bạn nhồi nhét bao nhiêu năng lực tính toán số học TOPS đi nữa, tốc độ cũng không thể vượt qua ngưỡng giới hạn vật lý này.

Trong khi đó, bộ nhớ HBM3 của trung tâm dữ liệu với băng thông 3 TB/s quét qua cùng lượng dữ liệu 3,5 GB đó chỉ trong 1,2 ms — cho phép trần lý thuyết lên tới 830 tok/s. Cùng một mô hình, cùng một bộ trọng số. Điểm khác biệt duy nhất nằm ở phân hệ bộ nhớ.

### Apple Neural Engine (M4 / A18)

- Đạt năng lực tính toán lên tới 38 TOPS. Kiến trúc bộ nhớ thống nhất (unified memory — CPU và ANE dùng chung một vùng nhớ vật lý) — không phát sinh bất kỳ chi phí sao chép dữ liệu nào.
- Truy cập thông qua Core ML cùng các mô hình đã biên dịch định dạng `.mlmodel`, hoặc thông qua Metal Performance Shaders (MPS) tích hợp trong PyTorch.
- Backend Metal của Llama.cpp sử dụng MPS chứ không giao tiếp trực tiếp với ANE; để khai thác ANE nguyên bản bắt buộc phải trải qua bước chuyển đổi Core ML.
- Lộ trình thực tế tối ưu nhất cho ứng dụng iOS năm 2026: Core ML với trọng số INT4 + kích hoạt FP16.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Đạt năng lực tính toán lên tới 45 TOPS. Được tích hợp cùng CPU và GPU trên cùng một hệ thống trên vi mạch (SoC) nhưng sở hữu miền bộ nhớ riêng biệt.
- Bộ công cụ phát triển phần mềm QNN (Qualcomm Neural Network) SDK cùng nền tảng AI Hub cung cấp các công cụ chuyển đổi từ PyTorch/ONNX.
- Mẫu hội thoại (chat templates), Llama 3.2, Phi-3 đều được phát hành dưới dạng các thành phần chuẩn loại một (first-class artifacts) trên AI Hub.

### NPU của Intel / AMD (Lunar Lake, Ryzen AI 300)

- Đạt từ 40 đến 50 TOPS. Hệ sinh thái phần mềm còn đi sau Apple và Qualcomm; OpenVINO đang có nhiều cải tiến nhưng vẫn còn chiếm thị phần hẹp.
- Lựa chọn tối ưu cho các ứng dụng Copilot trên Windows ARM; chạy nguyên bản trên máy tính để bàn AMD/Intel theo triết lý ưu tiên xử lý cục bộ (local-first).

### WebGPU + WebLLM

- Chạy trực tiếp các mô hình ngôn ngữ lớn ngay bên trong trình duyệt web thông qua các hàm đổ bóng tính toán (compute shaders) của WebGPU; người dùng không cần cài đặt bất kỳ phần mềm nào.
- Chạy Llama 3.1 8B Q4 đạt ~41 tok/s trên chip M3 Max — đạt khoảng 70–80% hiệu năng so với ứng dụng gốc chạy trên cùng backend.
- Đạt 17,6k lượt gắn sao trên GitHub cho WebLLM; cung cấp API JavaScript tương thích hoàn toàn với chuẩn OpenAI; giấy phép mã nguồn mở Apache 2.0.
- Độ phủ trình duyệt năm 2026: Hỗ trợ sẵn trên Chrome Android v121+, Safari iOS 26 bản GA chính thức, Firefox trên Android đang tích cực hoàn thiện. Tổng thể đạt khoảng ~70–75% thiết bị di động toàn cầu.

### Họ sản phẩm NVIDIA Jetson

- Orin Nano Super (8GB): chạy mượt mà Llama 3.2 3B, Phi-3 với tốc độ tok/s rất khả quan.
- AGX Orin: vận hành gpt-oss-20b thông qua vLLM ở mức ~40 tok/s.
- Thor / T4000 (JetPack 7.1): hiệu năng gấp đôi AGX Orin, hỗ trợ giải mã suy đoán EAGLE-3 và định dạng NVFP4.
- TensorRT Edge-LLM (năm 2026) hỗ trợ giải mã suy đoán EAGLE-3, trọng số NVFP4, kỹ thuật prefill phân mảnh — mang toàn bộ các tối ưu hóa của trung tâm dữ liệu xuống thiết bị biên.

### Lựa chọn định dạng lượng tử hóa theo từng nền tảng đích

| Nền tảng đích | Định dạng lượng tử hóa | Ghi chú kỹ thuật |
|---|---|---|
| Apple ANE | Trọng số INT4 + Kích hoạt FP16 | Lộ trình chuyển đổi qua Core ML |
| Qualcomm Hexagon | QNN INT8 / INT4 | Công cụ chuyển đổi từ AI Hub |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Dùng lệnh `mlc_llm convert_weight` + biên dịch `.wasm`; không hỗ trợ GGUF |
| Jetson Orin Nano | Q4 GGUF hoặc TRT-LLM INT4 | Giới hạn bởi dung lượng bộ nhớ |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Lộ trình triển khai Edge-LLM |

### Cạm bẫy ngữ cảnh dài trên thiết bị biên

Ngữ cảnh 128K token của Llama 3.1 là tính năng thiết kế cho trung tâm dữ liệu. Trên một chiếc điện thoại thông minh có 8 GB RAM: trọng số mô hình chiếm 4 GB + dung lượng KV cache tốn thêm 2 GB cho 32K token + phần chiếm dụng của hệ điều hành = tràn bộ nhớ (OOM). Các hệ thống triển khai trên thiết bị biên luôn duy trì độ dài ngữ cảnh ở mức 4K–8K token, trừ khi chấp nhận lượng tử hóa KV cache cực sâu (Q4 KV).

### Ứng dụng giọng nói là trường hợp sử dụng đột phá

Các trợ lý ảo giọng nói đòi hỏi độ trễ cực kỳ khắt khe (thời gian ra token đầu tiên < 500 ms). Suy luận cục bộ loại bỏ hoàn toàn độ trễ truyền dẫn mạng. Khi kết hợp với mô hình chuyển đổi giọng nói thành văn bản (các biến thể Whisper Turbo chạy trực tiếp trên thiết bị biên), suy luận biên trở thành nền tảng hoàn hảo để hiện thực hóa vòng lặp giao tiếp giọng nói đạt chất lượng production.

### Các con số cần ghi nhớ

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon trên Snapdragon X Elite: 45 TOPS.
- WebLLM trên chip M3 Max: ~41 tok/s trên mô hình Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/s trên gpt-oss-20b thông qua vLLM.
- Khoảng cách băng thông giữa trung tâm dữ liệu và thiết bị biên: 30–50 lần.
- Độ phủ WebGPU trên thiết bị di động: ~70–75% (Firefox Android đang bám đuổi).

```figure
edge-bandwidth-pipe
```

## Sử dụng

Tệp `code/main.py` tính toán mức trần thông lượng giải mã lý thuyết dựa trên các công thức vật lý bị giới hạn bởi băng thông bộ nhớ trên các thiết bị biên khác nhau. Chương trình so sánh các con số này với điểm chuẩn đo kiểm thực tế để chỉ rõ vị trí mà băng thông (chứ không phải năng lực tính toán) đóng vai trò là nút thắt cổ chai.

## Phát hành

Bài học này cung cấp tệp `outputs/skill-edge-target-picker.md`. Dựa trên nền tảng (iOS/Android/trình duyệt/Jetson), mô hình dự kiến và ngân sách độ trễ/bộ nhớ, công cụ này sẽ tư vấn định dạng lượng tử hóa tối ưu và xây dựng quy trình chuyển đổi tương ứng.

## Bài tập

1. Chạy `code/main.py`. Với một mô hình 7B ở định dạng Q4 trên chip Snapdragon 8 Gen 3 (~77 GB/s băng thông), hãy tính mức trần tốc độ giải mã. So sánh với con số quan sát thực tế từ 6 đến 8 tok/s — liệu môi trường thực thi (runtime) đã đạt hiệu quả tối ưu hay chưa?
2. WebGPU trên Android đòi hỏi Chrome phiên bản v121 trở lên. Hãy thiết kế một cơ chế dự phòng (fallback) cho các trình duyệt cũ hơn — xử lý tại máy chủ nhưng vẫn dùng chung API tương thích OpenAI.
3. Ứng dụng iOS của bạn yêu cầu truyền trực tiếp ngữ cảnh 4K. Sự kết hợp giữa mô hình và định dạng nào giúp bạn duy trì mức bộ nhớ hoạt động (active memory) dưới 4 GB trên iPhone 16?
4. Jetson AGX Orin chạy được gpt-oss-20b ở mức 40 tok/s, trong khi Jetson Nano chỉ đủ dung lượng chứa mô hình 3B. Nếu sản phẩm của bạn nhắm vào cả hai loại phần cứng này, bạn sẽ hợp nhất ngăn xếp suy luận như thế nào?
5. Hãy đưa ra lập luận phản biện hoặc ủng hộ nhận định: "WebLLM đã hoàn toàn sẵn sàng cho môi trường production trong năm 2026." Trích dẫn cụ thể các chỉ số về độ phủ thiết bị, hiệu năng thực tế và khoảng trống hỗ trợ của Firefox trên Android.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| ANE | "bộ xử lý thần kinh Apple" | NPU trên chip Apple dòng M và A; chia sẻ bộ nhớ thống nhất |
| Hexagon | "NPU của Qualcomm" | NPU tích hợp trên chip Snapdragon; truy cập thông qua QNN SDK |
| WebGPU | "GPU trình duyệt" | Chuẩn API đồ họa và tính toán trên web của W3C; hỗ trợ trên Chrome/Safari 2026 |
| WebLLM | "bộ chạy LLM trên web" | Dự án thuộc hệ sinh thái MLC-LLM; mã nguồn mở Apache 2.0; API JS chuẩn OpenAI |
| Jetson | "phần cứng biên NVIDIA" | Dòng máy tính nhúng chuyên dụng Orin Nano / AGX / Thor / T4000 |
| TRT Edge-LLM | "TensorRT cho thiết bị biên" | Bản chuyển giao TensorRT-LLM cho thiết bị biên năm 2026; hỗ trợ EAGLE-3 + NVFP4 |
| Unified memory | "bộ nhớ chia sẻ chung" | CPU và NPU cùng truy cập một không gian RAM; không tốn chi phí sao chép |
| Bandwidth-bound | "giới hạn bởi bộ nhớ" | Quá trình giải mã bị nghẽn bởi tốc độ đọc trọng số (byte/giây) |
| Core ML | "bộ chuyển đổi của Apple" | Khung phần mềm chuyên biệt của Apple để tối ưu mô hình chạy trực tiếp trên ANE |
| QNN | "ngăn xếp phần mềm Qualcomm" | Bộ công cụ phát triển phần mềm mạng nơ-ron của Qualcomm |

## Tài liệu đọc thêm

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — Toàn cảnh bức tranh công nghệ và các bài đo điểm chuẩn.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Các dòng máy Orin / AGX / Thor cho robot và thiết bị thông minh.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — Công bố chính thức về giải pháp suy luận LLM cho ô tô và robot.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — Thiết kế kiến trúc và kết quả đo điểm chuẩn trên trình duyệt web.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — Tài liệu tối ưu hóa mô hình nguyên bản cho chip Apple Neural Engine.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Kho mô hình chuyển đổi sẵn cho vi xử lý Hexagon NPU.
