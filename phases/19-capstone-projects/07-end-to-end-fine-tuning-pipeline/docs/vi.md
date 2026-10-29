# Đồ án Tốt nghiệp 07 — Quy trình Tinh chỉnh Đầu-cuối (Từ Dữ liệu đến SFT đến DPO đến Phục vụ)

> Một mô hình lớp 8B được huấn luyện trên dữ liệu riêng của bạn, căn chỉnh DPO dựa trên tập sở thích riêng, lượng tử hóa, áp dụng giải mã suy đoán (speculative decoding), và phục vụ với mức chi phí đo lường được trên 1M token ($/1M tokens). Ngăn xếp mở (open stack) năm 2026 bao gồm Axolotl v0.8, TRL 0.15, Unsloth cho việc lặp thử nghiệm nhanh, GPTQ/AWQ/GGUF để lượng tử hóa, và vLLM 0.7 kết hợp EAGLE-3 cho việc phục vụ suy luận. Đồ án tốt nghiệp này yêu cầu vận hành toàn bộ quy trình có khả năng tái lập (đầu vào là cấu hình YAML, đầu ra là endpoint phục vụ suy luận) và công bố thẻ mô hình (model card) theo Khung Độ mở Mô hình năm 2026 (2026 Model Openness Framework).

**Type:** Capstone  
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)  
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)  
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**Time:** ~35 giờ  

## Vấn đề đặt ra

Mọi đội ngũ AI nghiêm túc vào năm 2026 đều duy trì sẵn một quy trình tinh chỉnh (fine-tuning pipeline) hoàn chỉnh. Không phải vì họ cần huấn luyện mô hình nền tảng tiên phong (frontier base model) từ đầu, mà vì khả năng thích ứng hạ nguồn (downstream adaptation) — tinh chỉnh có giám sát theo miền (domain SFT), tối ưu hóa sở thích trực tiếp (Direct Preference Optimization - DPO) theo các sở thích được gán nhãn, chưng cất bản nháp cho giải mã suy đoán, và phục vụ với EAGLE-3 — chính là nơi mang lại những cải tiến đo lường được rõ rệt nhất. Axolotl v0.8 quản lý các cấu hình SFT đa GPU. TRL 0.15 đảm nhận DPO và GRPO (Group Relative Policy Optimization). Unsloth mang lại tốc độ lặp thử nghiệm cực nhanh trên đơn GPU. vLLM 0.7 kết hợp với EAGLE-3 tăng thông lượng giải mã lên 2-3 lần mà không làm suy giảm chất lượng. Các công cụ đều đã sẵn sàng; sự tinh tế nằm ở cấu hình YAML, vệ sinh dữ liệu và kỷ luật đánh giá.

Bạn sẽ đưa một mô hình nền tảng từ 8B đến 14B tham số (Llama 3.1 8B, Qwen3 14B hoặc Gemma 3 12B) qua quá trình SFT rồi đến DPO trên tập dữ liệu đặc thù của tác vụ, lượng tử hóa để phục vụ thực tế, và đo lường mức độ cải thiện so với các bộ chuẩn lm-evaluation-harness, RewardBench-2, MT-Bench-v2 và MMLU-Pro. Bạn sẽ tạo ra một thẻ mô hình (model card) tuân theo Khung Độ mở Mô hình năm 2026 (Model Openness Framework - MOF). Mục tiêu cốt lõi là tính tái lập (reproducibility): chỉ với một câu lệnh duy nhất, toàn bộ pipeline sẽ chạy lại thông suốt từ đầu đến cuối.

## Khái niệm cốt lõi

Quy trình gồm 5 giai đoạn:
1. **Dữ liệu (Data):** Khử trùng lặp (deduplication với MinHash / Datatrove), lọc chất lượng (bộ phân loại theo phong cách Nemotron-CC), làm sạch thông tin nhận dạng cá nhân (PII scrub), và kiểm tra độ sạch phân tách tập dữ liệu (split-hygiene check) nhằm tránh nhiễm bẩn (contamination) với các bộ chuẩn công khai.
2. **SFT:** Cấu hình Axolotl YAML, ZeRO-3 trên cụm 8xH100, lịch suy giảm học tốc dạng cosine (cosine schedule), đóng gói chuỗi (packed sequences), từ 2-3 epoch.
3. **DPO hoặc GRPO:** Cấu hình TRL, 1 epoch, các cặp dữ liệu sở thích (preference pairs) do con người gán nhãn hoặc do mô hình đánh giá (model-judged), tinh chỉnh tham số beta.
4. **Lượng tử hóa (Quantize):** Kết hợp GPTQ + AWQ + GGUF nhằm đảm bảo tính linh hoạt tối đa khi triển khai.
5. **Phục vụ suy luận (Serve):** vLLM 0.7 với các đầu suy đoán EAGLE-3 (hoặc SGLang với SpecForge), triển khai trên Kubernetes (K8s), tự động co giãn ngang (HPA) dựa trên độ trễ hàng đợi (queue-wait).

Các thực nghiệm cắt bỏ (ablations) là sản phẩm bàn giao chính: so sánh SFT đơn thuần vs SFT+DPO vs SFT+GRPO trên ba bộ chuẩn đặc thù của tác vụ. Các chỉ số phục vụ: token/giây tại kích thước batch 1 / 8 / 32, tỷ lệ chấp nhận (acceptance rate) của EAGLE-3, chi phí $/1M token. Đánh giá an toàn: tỷ lệ vượt qua bài kiểm tra của Llama Guard 4. Thẻ mô hình: đánh giá độ thiên vị (bias evaluations), seed tái lập, và giấy phép sử dụng dữ liệu.

## Kiến trúc hệ thống

```text
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Ngăn xếp công nghệ

- **Dữ liệu:** Datatrove cho khử trùng lặp, bộ phân loại Nemotron-CC cho chất lượng, Presidio cho làm sạch PII
- **Mô hình nền tảng:** Llama 3.1 8B, Qwen3 14B, hoặc Gemma 3 12B
- **SFT:** Axolotl v0.8 với ZeRO-3, Flash Attention 3, đóng gói chuỗi (packed sequences)
- **Căn chỉnh sở thích (Preference tuning):** TRL 0.15 cho DPO hoặc GRPO; Unsloth cho việc lặp thử nghiệm trên đơn GPU
- **Lượng tử hóa (Quantization):** GPTQ (Marlin), AWQ, GGUF qua llama.cpp
- **Phục vụ suy luận (Serving):** vLLM 0.7 với giải mã suy đoán EAGLE-3 (hoặc SGLang 0.4 + SpecForge)
- **Đánh giá (Eval):** lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- **Đánh giá an toàn:** Llama Guard 4, ShieldGemma-2
- **Hạ tầng:** Kubernetes + NVIDIA device plugin, HPA dựa trên chỉ số hàng đợi chờ (queue-wait metric)
- **Khả năng quan sát (Observability):** W&B cho quá trình huấn luyện, Langfuse cho quá trình suy luận

```figure
ce-finetune-stages
```

## Từng bước xây dựng

1. **Pipeline dữ liệu.** Chạy khử trùng lặp Datatrove trên kho ngữ liệu thô. Áp dụng bộ phân loại chất lượng kiểu Nemotron-CC. Dùng Presidio để làm sạch thông tin PII. Ghi các phân chia tập train/val với giá trị seed xác định.

2. **Kiểm tra nhiễm bẩn (Contamination check).** Với mỗi tập validation, tính toán MinHash đối chiếu với tập kiểm thử của MMLU-Pro, MT-Bench-v2, RewardBench-2. Loại bỏ mọi trường hợp trùng lặp.

3. **SFT với Axolotl.** Tạo cấu hình YAML với ZeRO-3, FA3, đóng gói chuỗi. Huấn luyện 2-3 epoch trên cụm 8xH100. Ghi log lên W&B.

4. **DPO / GRPO với TRL.** Nhận điểm kiểm tra (checkpoint) từ bước SFT, chạy 1 epoch DPO trên các cặp sở thích (hoặc GRPO với hàm phần thưởng kiểm chứng được đối với toán/mã nguồn). Quét tìm giá trị beta tối ưu.

5. **Lượng tử hóa.** Xuất ra ba định dạng lượng tử hóa: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M cho llama.cpp. Ghi lại dung lượng và thông lượng lý thuyết.

6. **Phục vụ với giải mã suy đoán.** Cấu hình vLLM 0.7 với các đầu nháp EAGLE-3 được huấn luyện qua Red Hat Speculators. Đo tỷ lệ chấp nhận và độ trễ đuôi (tail latency) tại batch size 1 / 8 / 32. Báo cáo chi phí $/1M token so sánh với Anthropic / OpenAI trên cùng bài đánh giá.

7. **Ma trận đánh giá.** Chạy lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro trên mô hình gốc, mô hình chỉ qua SFT, SFT+DPO, SFT+GRPO. Lập bảng so sánh chi tiết.

8. **Đánh giá an toàn.** Đo tỷ lệ vượt qua của Llama Guard 4 trên tập phát triển (dev set). Cấu hình bộ lọc đầu ra ShieldGemma-2.

9. **Thẻ mô hình (Model card).** Dùng mẫu chuẩn MOF 2026: dữ liệu, huấn luyện, đánh giá, an toàn, giấy phép, cùng mục tái lập kèm các tệp YAML và mã băm commit SHA.

## Vận hành thực tế

```bash
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Bàn giao sản phẩm

Tệp `outputs/skill-finetuning-pipeline.md` mô tả sản phẩm bàn giao. Một lệnh duy nhất sẽ xử lý dữ liệu qua SFT, DPO, lượng tử hóa, triển khai phục vụ, chạy đánh giá, và tạo ra thẻ mô hình kèm endpoint đang phục vụ trực tiếp.

| Trọng số | Tiêu chí | Phương pháp đo lường |
|:-:|---|---|
| 25 | Độ chênh lệch đánh giá so với mô hình gốc | Mức tăng trưởng đo được trên các tác vụ mục tiêu (MMLU-Pro, MT-Bench-v2, tác vụ đặc thù) |
| 20 | Tính tái lập của quy trình | Một lệnh chạy lại toàn bộ quy trình đầu-cuối với cùng giá trị seed |
| 20 | Vệ sinh dữ liệu | Tỷ lệ khử trùng lặp, độ bao phủ lọc PII, vượt qua kiểm tra nhiễm bẩn |
| 20 | Hiệu quả phục vụ | Token/giây tại bs=1/8/32, tỷ lệ chấp nhận của EAGLE-3, $/1M token |
| 15 | Thẻ mô hình + Đánh giá an toàn | Độ hoàn thiện theo MOF 2026 + tỷ lệ vượt qua bài kiểm tra Llama Guard 4 |
| **100** | | |

## Bài tập mở rộng

1. Chạy so sánh chỉ SFT vs SFT+DPO vs SFT+GRPO trên cùng một bài chuẩn đặc thù của tác vụ. Báo cáo phương pháp căn chỉnh sở thích nào chiến thắng và mức chênh lệch là bao nhiêu.

2. Hoán đổi Llama 3.1 8B thành Qwen3 14B. Đo lường chỉ số $/1M token khi đạt cùng mức chất lượng tương đương.

3. Đo tỷ lệ chấp nhận của EAGLE-3 trên dữ liệu theo miền chuyên ngành so với dữ liệu ShareGPT tổng quát. Báo cáo mức chênh lệch và ý nghĩa của nó đối với ngân sách độ trễ (latency budget).

4. Thử nghiệm cấy 1% dữ liệu nhiễm bẩn (để lộ đáp án của MMLU-Pro vào dữ liệu huấn luyện) và chạy lại quy trình đánh giá. Quan sát độ chính xác của MMLU-Pro tăng vọt một cách bất thường. Xây dựng một cổng kiểm tra nhiễm bẩn trong CI để chặn đứng lỗi này.

5. Bổ sung kỹ thuật LoRA SFT như một giải pháp thay thế cho tinh chỉnh toàn phần (full fine-tuning). Đo lường khoảng cách về chất lượng khi tiết kiệm bộ nhớ gấp 10 lần.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Axolotl | "Bộ huấn luyện SFT" | Bộ huấn luyện hợp nhất điều khiển bằng YAML cho SFT, DPO và chưng cất tri thức |
| TRL | "Bộ tinh chỉnh sở thích" | Thư viện Hugging Face hỗ trợ DPO, GRPO, PPO cho LLM |
| GRPO | "Tối ưu hóa chính sách theo nhóm tương đối" | Công thức học tăng cường (RL) của DeepSeek R1 với hàm phần thưởng kiểm chứng được |
| EAGLE-3 | "Mô hình nháp giải mã suy đoán" | Các đầu nháp dự đoán trước N token; vLLM sẽ xác thực lại với mô hình mục tiêu |
| MOF | "Khung Độ mở Mô hình" | Bộ tiêu chuẩn năm 2026 đánh giá mức độ phát hành mở của mô hình về dữ liệu, mã nguồn và giấy phép |
| Contamination check | "Kiểm tra vệ sinh tập phân chia" | Phát hiện rò rỉ tập kiểm thử vào tập huấn luyện dựa trên MinHash |
| Acceptance rate | "Chỉ số EAGLE / MTP" | Tỷ lệ token tạo từ mô hình nháp được mô hình mục tiêu chấp nhận |

## Tài liệu đọc thêm

- [Tài liệu hướng dẫn Axolotl](https://axolotl-ai-cloud.github.io/axolotl/) — bộ huấn luyện tham chiếu cho SFT / DPO
- [Tài liệu hướng dẫn TRL](https://huggingface.co/docs/trl) — các cài đặt tham chiếu cho DPO và GRPO
- [Unsloth](https://github.com/unslothai/unsloth) — tài liệu tham chiếu lặp nhanh trên đơn GPU
- [Bài báo DeepSeek R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — phương pháp luận GRPO
- [Tài liệu vLLM + EAGLE-3](https://docs.vllm.ai) — ngăn xếp phục vụ suy luận chuẩn mực
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — công cụ huấn luyện giải mã suy đoán thay thế
- [Khung Độ mở Mô hình - MOF (arXiv:2403.13784)](https://arxiv.org/abs/2403.13784): tiêu chuẩn phân hạng mức độ mở khi phát hành
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — công cụ chuẩn mực để chạy đánh giá mô hình
