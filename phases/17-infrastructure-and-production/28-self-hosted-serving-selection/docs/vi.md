# Lựa chọn Công cụ Phục vụ Tự lưu trữ — Khớp nối Động cơ Suy luận với Phần cứng và Quy mô

> Việc lựa chọn động cơ suy luận (serving engine) là một hàm phụ thuộc vào phần cứng, quy mô tải và hệ sinh thái công nghệ — không đơn thuần là việc đọc bảng xếp hạng hiệu năng. Bốn động cơ dẫn đầu mảng suy luận tự lưu trữ (self-hosted inference) trong năm 2026 gồm: llama.cpp, Ollama, vLLM, SGLang, trong khi TGI đang tụt lại phía sau ở chế độ bảo trì. **llama.cpp** đạt tốc độ nhanh nhất trên CPU — hỗ trợ danh mục mô hình rộng nhất, trao toàn quyền kiểm soát kỹ thuật lượng tử hóa (quantization) và phân luồng CPU (threading). **Ollama** là giải pháp cài đặt một dòng lệnh tối ưu cho máy tính xách tay của kỹ sư phát triển, nhưng chậm hơn ~15-30% so với llama.cpp (do chi phí tuần tự hóa HTTP giữa Go + CGo), đồng thời xuất hiện khoảng cách thông lượng gấp 3 lần dưới tải trọng môi trường sản xuất. **TGI đã bước vào chế độ bảo trì từ ngày 11 tháng 12 năm 2025** — hiện chỉ tiếp nhận các bản vá lỗi, thông lượng thô chậm hơn ~10% so với vLLM dù sở hữu nền tảng khả năng quan sát (observability) và tích hợp hệ sinh thái Hugging Face hàng đầu trong lịch sử. Trạng thái bảo trì này khiến TGI trở thành một lựa chọn dài hạn tiềm ẩn nhiều rủi ro — SGLang hoặc vLLM là những lựa chọn mặc định an toàn hơn cho các dự án mới. **vLLM** là tiêu chuẩn sản xuất đa dụng mặc định — phiên bản v0.15.1 (tháng 2/2026) bổ sung hỗ trợ PyTorch 2.10, kiến trúc RTX Blackwell SM120, và tối ưu hóa chuyên sâu cho H200. **SGLang** là động cơ chuyên biệt cho khối lượng công việc đa lượt hướng tác tử (agentic multi-turn) và các tác vụ tái sử dụng tiền tố (prefix-heavy) lớn — vận hành trên hơn 400.000 GPU trong môi trường sản xuất thực tế (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Ràng buộc phần cứng: Ưu tiên CPU → chọn llama.cpp. Phần cứng AMD / non-NVIDIA → vLLM là lộ trình được hỗ trợ mạnh mẽ nhất (TRT-LLM bị khóa chặt vào phần cứng NVIDIA). Mô hình đường ống năm 2026: môi trường phát triển (dev) = Ollama, môi trường tiền sản xuất (staging) = llama.cpp, môi trường sản xuất (prod) = vLLM hoặc SGLang. Các động cơ sử dụng định dạng trọng số khác nhau — GGUF dành cho họ llama.cpp, HF safetensors dành cho các động cơ GPU — do đó quy trình chuyển đổi định dạng có thể cần được bố trí giữa các giai đoạn.

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** Toàn bộ các bài học Phase 17 về serving engines (04, 06, 07, 09, 18)
**Time:** ~45 phút

## Mục tiêu học tập

- Lựa chọn động cơ phục vụ suy luận chuẩn xác dựa trên cấu hình phần cứng (CPU / AMD / NVIDIA Hopper / Blackwell), quy mô người dùng (1 người dùng / 100 / 10.000), và đặc thù khối lượng công việc (trò chuyện tổng quát / agent / ngữ cảnh dài).
- Nắm rõ cột mốc TGI bước vào chế độ bảo trì (ngày 11 tháng 12 năm 2025) và lý do vì sao các dự án mới nên ưu tiên chuyển hướng sang vLLM hoặc SGLang.
- Mô tả đường ống chuyển tiếp môi trường phát triển/staging/sản xuất, bao gồm vị trí chuyển đổi định dạng trọng số từ GGUF sang safetensors giữa các giai đoạn.
- Giải thích vì sao tiêu chí "ưu tiên CPU" luôn chỉ tới llama.cpp và nền tảng "AMD" hoàn toàn loại trừ TRT-LLM.

## Vấn đề thực tế

Nhóm kỹ sư của bạn khởi động một dự án triển khai LLM tự lưu trữ mới. Một kỹ sư đề xuất dùng Ollama, một người khác khẳng định vLLM là tiêu chuẩn duy nhất, trong khi người thứ ba hỏi: "Chẳng phải TGI hoạt động trơn tru ngay từ đầu sao?" Cả ba người đều có lý trong những ngữ cảnh riêng biệt. Nhưng không có lựa chọn nào là tối ưu cho mọi trường hợp.

Trong năm 2026, cây quyết định kỹ thuật phải phân cấp rõ ràng: Phần cứng đứng đầu, quy mô đứng thứ hai, đặc thù công việc đứng thứ ba. Và một sự kiện bước ngoặt cuối năm 2025 — TGI chính thức bước vào chế độ bảo trì từ ngày 11 tháng 12 — đã thay đổi hoàn toàn lựa chọn mặc định cho các dự án mới.

## Khái niệm then chốt

### Năm động cơ phục vụ suy luận hàng đầu

| Động cơ | Tối ưu nhất cho | Ghi chú kỹ thuật |
|---|---|---|
| **llama.cpp** | CPU / thiết bị biên (edge) / phụ thuộc tối thiểu / hỗ trợ mô hình phong phú nhất | Đạt tốc độ nhanh nhất trên CPU, trao quyền kiểm soát toàn diện |
| **Ollama** | Máy tính phát triển cục bộ, người dùng đơn lẻ, cài đặt một dòng lệnh | Chậm hơn 15-30% so với llama.cpp; khoảng cách thông lượng tụt lại 3 lần ở tải sản xuất |
| **TGI** | Hệ sinh thái Hugging Face, các ngành chịu sự quản lý pháp lý khắt khe | **Chuyển sang chế độ bảo trì từ 11/12/2025** |
| **vLLM** | Môi trường sản xuất đa dụng, quy mô 100+ người dùng đồng thời | Tiêu chuẩn sản xuất mặc định; bản v0.15.1 ra mắt tháng 2/2026 |
| **SGLang** | Tác vụ đa lượt hướng agent, khối lượng công việc tái sử dụng tiền tố lớn | Triển khai trên hơn 400.000 GPU trong sản xuất |

### Cấp quyết định 1: Ưu tiên phần cứng (Hardware-first)

**Ưu tiên CPU** → llama.cpp. Ollama có thể hoạt động nhưng tốc độ chậm hơn đáng kể. Không có động cơ nào khác đủ sức cạnh tranh với llama.cpp trên môi trường CPU.

**GPU AMD** → vLLM là con đường được hỗ trợ hoàn thiện nhất (thông qua nền tảng ROCm của AMD). SGLang cũng hỗ trợ tốt. TRT-LLM bị khóa độc quyền cho hệ sinh thái NVIDIA nên hoàn toàn bị loại.

**NVIDIA Hopper (H100 / H200)** → vLLM, SGLang hoặc TRT-LLM. Cả ba đều là các lựa chọn hàng đầu (top-tier).

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM dẫn đầu tuyệt đối về thông lượng (Phase 17 · 07). vLLM và SGLang bám đuổi rất sát phía sau.

**Apple Silicon (dòng chip M)** → llama.cpp (tận dụng backend Metal). Ollama đóng vai trò là lớp bọc bên ngoài engine này.

### Cấp quyết định 2: Quy mô người dùng (Scale-second)

**1 người dùng / Môi trường phát triển cục bộ** → Ollama. Khởi chạy bằng một lệnh duy nhất, nhận token đầu tiên trong vài giây.

**10-100 người dùng / Nhóm nội bộ nhỏ** → vLLM chạy trên một GPU đơn lẻ.

**100-10.000 người dùng / Môi trường sản xuất thực tế** → Ngăn xếp sản xuất vLLM (Phase 17 · 18) hoặc SGLang.

**Trên 10.000 người dùng / Cấp độ doanh nghiệp lớn** → Ngăn xếp sản xuất vLLM + Kiến trúc tách rời tính toán/giải mã (Disaggregated serving - Phase 17 · 17) + LMCache (Phase 17 · 18).

### Cấp quyết định 3: Đặc thù khối lượng công việc (Workload-third)

**Trò chuyện tổng quát / Hỏi đáp (Q&A)** → vLLM chiến thắng nhờ tính đa dụng và độ ổn định cao.

**Tác vụ đa lượt hướng tác tử (Agentic multi-turn - gọi công cụ, lập kế hoạch, bộ nhớ)** → Cơ chế RadixAttention của SGLang (Phase 17 · 06) chiếm ưu thế áp đảo.

**RAG với tần suất tái sử dụng tiền tố văn bản cao** → SGLang là lựa chọn vượt trội.

**Sinh mã lệnh (Code generation)** → vLLM đáp ứng tốt; SGLang nhỉnh hơn đôi chút nhờ cơ chế lưu bộ đệm hiệu quả.

**Ngữ cảnh siêu dài (128K+ token)** → vLLM kết hợp chunked prefill; SGLang kết hợp phân tầng bộ nhớ đệm KV (tiered KV).

### Cạm bẫy bảo trì của TGI

Hugging Face TGI đã chính thức chuyển sang chế độ bảo trì từ ngày 11 tháng 12 năm 2025 — kể từ thời điểm này, dự án chỉ tiếp nhận các bản sửa lỗi (bug fixes). Trong lịch sử: TGI sở hữu năng lực quan sát đỉnh cao, tích hợp hoàn hảo nhất với hệ sinh thái HF (model cards, công cụ an toàn), chỉ xếp sau vLLM một chút về mặt thông lượng thuần túy.

Đối với các dự án mới trong năm 2026: Hãy mặc định tránh xa TGI. Các cụm TGI đang vận hành sẵn có thể tiếp tục duy trì nhưng cần lên kế hoạch di chuyển dần. SGLang và vLLM là những lựa chọn mặc định an toàn hơn rất nhiều.

### Mô hình đường ống chuyển đổi (The Pipeline Pattern)

Môi trường phát triển (Ollama) → Môi trường tiền sản xuất/staging (llama.cpp) → Môi trường sản xuất/prod (vLLM). Do mỗi động cơ tiếp nhận định dạng trọng số khác nhau — GGUF cho hệ sinh thái llama.cpp, HF safetensors cho các động cơ GPU — nên cần bố trí một bước chuyển đổi định dạng giữa các tầng môi trường. Các kỹ sư có thể thử nghiệm lặp lại nhanh chóng trên laptop; môi trường staging phản ánh chính xác mức lượng tử hóa sản xuất; và prod là đích đến phục vụ người dùng cuối.

### Cảnh báo về Ollama

Ollama rất tuyệt vời cho môi trường phát triển cá nhân. Nhưng nó không phù hợp cho môi trường sản xuất dùng chung: Chi phí tuần tự hóa HTTP trong tầng Go gây độ trễ phụ, khả năng quản lý đồng thời (concurrency management) đơn giản hơn vLLM, và hỗ trợ OpenTelemetry còn hạn chế. Hãy dùng Ollama đúng với thế mạnh cốt lõi của nó — một người dùng, một dòng lệnh — và chuyển sang vLLM khi phục vụ tải dùng chung.

### Tự lưu trữ so với Dịch vụ quản lý là hai bài toán độc lập

Phase 17 · 01 (các nhà cung cấp hyperscaler) và · 02 (các nền tảng suy luận) đã phân tích sâu về dịch vụ quản lý (managed services). Bài học này tập trung vào trường hợp bạn đã quyết định tự lưu trữ hạ tầng. Các lý do chính đáng để tự lưu trữ: Tuân thủ lưu trú dữ liệu (data residency), mô hình tinh chỉnh riêng biệt (custom fine-tune), tối ưu hóa tổng chi phí sở hữu (TCO) ở quy mô lớn, hoặc chạy các mô hình chuyên biệt không được cung cấp trên dịch vụ lưu trữ đám mây.

### Các con số cần ghi nhớ

- Thời điểm TGI bước vào chế độ bảo trì: Ngày 11 tháng 12 năm 2025.
- Bản phát hành vLLM v0.15.1: Tháng 2 năm 2026; hỗ trợ PyTorch 2.10; tối ưu hóa kiến trúc Blackwell SM120.
- Quy mô triển khai sản xuất của SGLang: Hơn 400.000 GPU.
- Khoảng cách thông lượng của Ollama so với llama.cpp: Chậm hơn 15-30%; tụt lại tới 3 lần dưới tải sản xuất.

```figure
data-parallel
```

## Thực hành (Use It)

Tệp `code/main.py` là một cây quyết định bằng Python: Tiếp nhận các tham số phần cứng + quy mô + đặc thù công việc, từ đó đưa ra lựa chọn động cơ tối ưu kèm giải thích cặn kẽ lý do kỹ thuật.

## Triển khai thực tế (Ship It)

Bài học này xây dựng `outputs/skill-engine-picker.md`. Dựa trên các ràng buộc thực tế của hệ thống, tài liệu này chỉ định động cơ phù hợp nhất và phác thảo lộ trình di chuyển hạ tầng chi tiết.

## Bài tập

1. Chạy `code/main.py` với cấu hình phần cứng / quy mô / tác vụ thực tế của bạn. Kết quả đầu ra có trùng khớp với trực giác kỹ thuật của bạn không?
2. Hạ tầng của bạn gồm 12 card NVIDIA H100 và 8 card AMD MI300X. Đâu là động cơ phù hợp nhất? Giải thích vì sao TRT-LLM hoàn toàn không thể sử dụng trong kịch bản này?
3. Một nhóm kỹ sư muốn tiếp tục sử dụng TGI trong năm 2026 với lý do "đây là công cụ chúng tôi quen thuộc nhất". Hãy đưa ra lập luận thuyết phục họ chuyển đổi sang nền tảng khác.
4. Chuyển dịch từ môi trường phát triển trên Ollama sang sản xuất trên vLLM: Những yếu tố nào sẽ thay đổi về lượng tử hóa, cấu hình tham số, và khả năng giám sát đo kiểm?
5. Thiết kế một sản phẩm RAG có độ dài tiền tố P99 là 8K token và tần suất tái sử dụng giữa các khách hàng rất cao. Hãy chọn động cơ phục vụ tối ưu và kết hợp ngăn xếp giải pháp từ Phase 17 · 11 + 18.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| llama.cpp | "công cụ chạy CPU" | Hỗ trợ nhiều mô hình nhất, tối ưu tốc độ số một trên CPU |
| Ollama | "công cụ trên laptop" | Cài đặt một dòng lệnh, thông lượng phù hợp môi trường dev |
| TGI | "hệ thống serving của HF" | Đã chuyển sang chế độ bảo trì từ tháng 12/2025 |
| vLLM | "lựa chọn mặc định" | Tiêu chuẩn sản xuất nền tảng rộng rãi nhất năm 2026 |
| SGLang | "động cơ cho agent" | Tối ưu hóa mạnh cho tiền tố lặp lại qua RadixAttention |
| TRT-LLM | "công cụ độc quyền NVIDIA" | Dẫn đầu thông lượng trên Blackwell, chỉ chạy trên phần cứng NVIDIA |
| GGUF | "định dạng của llama.cpp" | Định dạng tệp đóng gói các biến thể lượng tử hóa K-quant |
| Production-stack | "ngăn xếp K8s của vLLM" | Cấu trúc triển khai tham chiếu chuẩn hóa trong Phase 17 · 18 |
| Pipeline pattern | "quy trình dev→stage→prod" | Luồng chuyển tiếp Ollama → llama.cpp → vLLM; định dạng trọng số khác nhau |

## Tài liệu tham khảo

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — Ghi chú phát hành chính thức.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
