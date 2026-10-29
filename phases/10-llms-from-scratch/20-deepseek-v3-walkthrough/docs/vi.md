# Khám phá toàn diện kiến trúc DeepSeek-V3 (DeepSeek-V3 Architecture Walkthrough)

> Giai đoạn 10 · Bài học 14 đã nêu tên sáu "núm vặn" kiến trúc mà mọi mô hình nguồn mở đều tùy chỉnh. DeepSeek-V3 (tháng 12/2024, tổng cộng 671 tỷ tham số, 37 tỷ tham số kích hoạt) tinh chỉnh cả sáu núm vặn đó và bổ sung thêm bốn sáng tạo mới: Cơ chế chú ý tiềm ẩn đa đầu (Multi-Head Latent Attention - MLA), cân bằng tải không dùng hàm mất mát phụ trợ (auxiliary-loss-free load balancing), Dự đoán đa token (Multi-Token Prediction - MTP), và huấn luyện DualPipe. Bài học này sẽ đọc chi tiết kiến trúc DeepSeek-V3 từ trên xuống dưới và suy dẫn mọi số lượng tham số từ tệp cấu hình được công bố. Kết thúc bài học, bạn có thể giải thích lý do tại sao tỷ lệ 671B/37B là một nước đi đúng đắn và tại sao MLA kết hợp với MoE cùng nhau đánh bại từng phương pháp đơn lẻ ở ranh giới tiên phong (frontier).

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## Mục tiêu học tập

- Đọc tệp cấu hình của DeepSeek-V3 từ đầu đến cuối và giải thích từng trường cấu hình theo sáu núm vặn GPT-2 cùng bốn bổ sung mang tính đặc trưng của DeepSeek.
- Suy dẫn tổng số lượng tham số (671B), số lượng tham số kích hoạt (37B), và các thành phần đóng góp vào từng đại lượng.
- Tính toán kích thước dấu chân bộ nhớ (footprint) KV cache của MLA ở độ dài ngữ cảnh 128k và so sánh với mức chi phí mà một mô hình dày đặc (dense) cùng số tham số kích hoạt dùng GQA phải gánh chịu.
- Trình bày bốn sáng kiến mang tính đột phá của DeepSeek (MLA, MTP, định tuyến không mất mát phụ trợ, DualPipe) và chỉ rõ mỗi sáng kiến nhắm vào bộ phận nào của ngăn xếp kiến trúc/huấn luyện.

## Vấn đề đặt ra

DeepSeek-V3 là mô hình nguồn mở tiên phong đầu tiên có kiến trúc khác biệt một cách có ý nghĩa so với họ Llama. Llama 3 405B về cơ bản là "GPT-2 với sáu núm vặn được xoay chỉnh". DeepSeek-V3 là GPT-2 với toàn bộ sáu núm vặn đó cộng thêm bốn sáng kiến bổ sung. Đọc cấu hình Llama 3 là bài tập khởi động cho việc đọc cấu hình DeepSeek, nhưng cấu trúc sâu bên trong — hình thái của khối attention, logic định tuyến, mục tiêu trong thời gian huấn luyện — đủ khác biệt để bạn cần một hướng dẫn chi tiết riêng.

Giá trị của việc thấu hiểu nó: Việc phát hành trọng số mở của DeepSeek-V3 đã thay đổi định nghĩa về "năng lực tiên phong" trong các mô hình nguồn mở. Kiến trúc này chính là bản thiết kế mẫu mà nhiều đợt huấn luyện trong năm 2026 đang học hỏi. Hiểu rõ nó là điều kiện tiên quyết cho bất kỳ vai trò kỹ thuật nào liên quan đến huấn luyện hoặc suy luận LLM tiên phong.

## Khái niệm

### Phần lõi bất biến

DeepSeek-V3 về bản chất vẫn là mô hình tự hồi quy (autoregressive). Nó vẫn xếp chồng các khối giải mã (decoder blocks). Mỗi khối vẫn gồm cơ chế chú ý (attention) kết hợp với MLP và hai tầng chuẩn hóa RMSNorm. Nó vẫn sử dụng hàm kích hoạt SwiGLU trong MLP. Nó vẫn dùng RoPE (Rotary Position Embedding). Chuẩn hóa trước (pre-norm). Trọng số nhúng dùng chung (weight-tied embeddings). Cùng một nền tảng cơ sở như mọi mô hình Llama hay Mistral.

### Điểm chuyển biến: MLA thay vì GQA

Từ Giai đoạn 10 · Bài học 14, bạn đã biết GQA thu nhỏ KV cache bằng cách chia sẻ K và V giữa các nhóm đầu truy vấn Q. Cơ chế chú ý tiềm ẩn đa đầu (Multi-Head Latent Attention - MLA) tiến xa hơn một bước: K và V được nén vào một biểu diễn tiềm ẩn hạng thấp dùng chung (thông số `kv_lora_rank`), sau đó được giải nén cho từng đầu ngay trong quá trình tính toán (on-the-fly). Bộ đệm KV cache giờ đây chỉ cần lưu trữ vector tiềm ẩn này — thường chỉ là 512 số thực trên mỗi token trên mỗi tầng, thay vì $8 \times 128 = 1024$ số thực.

Ở ngữ cảnh 128k token, DeepSeek-V3 với MLA (một vector tiềm ẩn $c^{KV}$ dùng chung cho mỗi token mỗi tầng; K và V đều được tạo ra từ vector này thông qua các phép chiếu mở rộng vốn có thể gộp chung vào phép nhân ma trận tiếp theo):

```text
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

Một mốc cơ sở GQA giả định (cấu hình như Llama 3 70B với 8 đầu KV, kích thước đầu 128) sẽ phải tiêu tốn:

```text
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA nhỏ hơn gấp 4 lần so với bộ nhớ đệm GQA kiểu Llama-3-70B ở ngữ cảnh 128k.

Sự đánh đổi: MLA bổ sung thêm một bước giải nén cho mỗi phép tính attention (trên từng đầu). Khối lượng tính toán bổ sung này rất nhỏ so với lượng băng thông bộ nhớ tiết kiệm được. Đây là một món hời lớn cho việc suy luận ngữ cảnh dài.

### Định tuyến: Cân bằng tải không dùng hàm mất mát phụ trợ

Các bộ định tuyến (routers) MoE quyết định top-k chuyên gia nào sẽ xử lý từng token. Một bộ định tuyến ngây thơ sẽ dồn quá nhiều việc vào một vài chuyên gia, khiến các chuyên gia khác bị nhàn rỗi. Cách khắc phục tiêu chuẩn: thêm một số hạng mất mát phụ trợ (auxiliary loss) để phạt sự mất cân bằng tải. Cách này hiệu quả nhưng làm suy giảm nhẹ hiệu năng của nhiệm vụ chính.

DeepSeek-V3 giới thiệu một cơ chế không cần mất mát phụ trợ. Các số hạng độ lệch (bias) trên từng chuyên gia được cộng vào logit của bộ định tuyến và được điều chỉnh trong quá trình huấn luyện theo một quy tắc đơn giản: nếu chuyên gia $e$ bị quá tải, giảm `bias_e`; nếu non tải, tăng nó lên. Không cần thêm số hạng mất mát nào. Quá trình huấn luyện giữ được sự trong sạch. Tải trọng của các chuyên gia luôn ở trạng thái cân bằng.

Ảnh hưởng lên hàm mất mát chính: không đo lường được sự suy giảm nào. Ảnh hưởng lên kiến trúc MoE: tinh gọn hơn, không còn siêu tham số auxiliary loss cần tinh chỉnh.

### MTP: Huấn luyện dày đặc hơn + Mô hình phác thảo miễn phí

Từ Giai đoạn 10 · Bài học 18, bạn đã biết DeepSeek-V3 bổ sung $D=1$ mô-đun MTP (Multi-Token Prediction) nhằm dự đoán token ở vị trí cách 2 bước về phía trước. Khi suy luận, mô-đun đã huấn luyện này được tái sử dụng làm mô hình phác thảo (draft model) cho giải mã phỏng đoán (speculative decoding) với tỷ lệ chấp nhận trên 80%. Khi huấn luyện, mỗi trạng thái ẩn được giám sát trên $D+1 = 2$ mục tiêu, mang lại tín hiệu học tập dày đặc hơn.

Tham số: 14B bổ sung bên cạnh mô hình chính 671B. Chi phí phụ trội: 2,1%.

### Huấn luyện: DualPipe

Từ Giai đoạn 10 · Bài học 19, bạn biết rằng DualPipe là một đường ống hai chiều giúp chồng lấn các khối truyền xuôi và truyền ngược với các giao tiếp all-to-all xuyên node. Ở quy mô 2.048 GPU H800 của DeepSeek-V3, nó thu hồi khoảng 245.000 GPU-giờ mà phương pháp 1F1B lẽ ra đã đánh mất vào các bong bóng đường ống.

### Chi tiết cấu hình từng trường

Dưới đây là cấu hình rút gọn của DeepSeek-V3:

```yaml
hidden_size: 7168
intermediate_size: 18432   # kích thước ẩn MLP dày đặc, dùng ở vài tầng đầu
moe_intermediate_size: 2048 # kích thước ẩn MLP của từng chuyên gia
num_hidden_layers: 61
first_k_dense_layers: 3    # 3 tầng đầu tiên sử dụng MLP dày đặc
num_attention_heads: 128
num_key_value_heads: 128   # trên danh nghĩa bằng num_heads trong MLA, nhưng
                           # sự nén thực sự nằm ở kv_lora_rank
kv_lora_rank: 512          # chiều không gian tiềm ẩn của MLA
num_experts: 256           # số chuyên gia MoE trên mỗi khối
num_experts_per_tok: 8     # định tuyến chọn top-8
shared_experts: 1          # 1 chuyên gia dùng chung luôn bật trên mỗi khối
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1              # 1 mô-đun MTP ở độ sâu 1
```

Phân tích ý nghĩa:

- `hidden_size=7168`: chiều của vector nhúng (embedding dimension).
- `num_hidden_layers=61`: tổng số khối (độ sâu mô hình).
- `first_k_dense_layers=3`: 3 khối đầu tiên sử dụng MLP dày đặc kích thước 18432. 58 khối còn lại dùng MoE.
- `num_attention_heads=128`: 128 đầu truy vấn (query heads).
- `kv_lora_rank=512`: K và V được nén vào chiều tiềm ẩn này và được giải nén độc lập trên từng đầu.
- `num_experts=256, num_experts_per_tok=8`: mỗi khối MoE có 256 chuyên gia, định tuyến chọn 8 chuyên gia tốt nhất.
- `shared_experts=1`: bên cạnh 256 chuyên gia định tuyến, có 1 chuyên gia luôn luôn kích hoạt cho mọi token. Có thể xem đây là "tầng cơ sở dày đặc" đảm bảo mọi token đều nhận được một lượng tri thức xử lý ổn định.
- `moe_intermediate_size=2048`: kích thước tầng ẩn MLP của từng chuyên gia. Nhỏ hơn MLP dày đặc vì có tới 256 chuyên gia.

### Hạch toán tham số

Phép tính toàn diện nằm trong tệp `code/main.py`. Các điểm nổi bật:

- Tầng nhúng: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 3 khối dày đặc đầu tiên: attention với MLA (~84M mỗi khối theo cách tính đơn giản hóa của bộ tính toán) + SwiGLU MLP dày đặc (`3 * 7168 * 18432 = ~396M` mỗi khối) + các tầng chuẩn hóa. Tổng cộng khoảng 1,4B.
- 58 khối MoE: attention với MLA (~84M) + 256 chuyên gia định tuyến (`3 * 7168 * 2048 = ~44M` mỗi chuyên gia) + 1 chuyên gia dùng chung (~44M) + bộ định tuyến + chuẩn hóa. Khoảng 11,4B mỗi khối (tính cả tất cả chuyên gia). Khoảng 661B cho toàn bộ 58 khối MoE.
- Mô-đun MTP: mô-đun công bố có kích thước 14B. Bản tính toán dùng một mô hình đại diện dày đặc khoảng ~0,8B.

Tổng quy mô: bộ tính toán ước tính khoảng ~664B so với con số công bố 671B của mô hình chính. Điểm kiểm tra (checkpoint) trên Hugging Face là 685B do cộng thêm 14B của mô-đun MTP vào 671B. Khoảng chênh lệch ~1% còn lại xuất phát từ các chi tiết đơn giản hóa: bộ tính toán bỏ qua các chiều RoPE tách rời trong MLA, đầu ra không chia sẻ trọng số, và một số tensor nhỏ khác.

Số tham số kích hoạt trong mỗi lượt truyền xuôi:

- Attention: 84M mỗi tầng * 61 = ~5,1B (toàn bộ các tầng đều hoạt động).
- MLP kích hoạt: 3 tầng đầu dày đặc (`3 * 396M = ~1.2B`), 58 tầng MoE mỗi tầng kích hoạt 8 chuyên gia định tuyến + 1 chuyên gia dùng chung (`9 * 44M = ~396M` mỗi tầng, tổng cộng ~23B). Khoảng 24B tất cả.
- Tầng nhúng + chuẩn hóa: ~0,9B.
- Tổng tham số kích hoạt: ~30B theo bộ tính toán so với 37B được công bố. Phần lớn chênh lệch nằm ở attention: MLA hoàn chỉnh, với các chiều RoPE bổ sung cùng các đầu query/key rộng hơn, chiếm dung lượng lớn hơn so với con số ước tính giản lược. Mô-đun MTP không tính vào con số 37B này.

### Tỷ lệ 671B / 37B

Tỷ lệ thưa (sparsity ratio) đạt mức 18x (tham số kích hoạt chỉ chiếm 5,5% tổng số tham số). DeepSeek-V3 là mô hình MoE tiên phong thưa nhất từng phát hành trọng số mở. Mixtral 8x7B với tỷ lệ 13/47 (28%) dày đặc hơn nhiều. Llama 4 Maverick với tỷ lệ 17B/400B (4,25%) là ở mức tương đương. Nước cờ chiến lược của DeepSeek: ở quy mô tiên phong, có nhiều chuyên gia hơn với tỷ lệ kích hoạt thấp hơn sẽ tạo ra chất lượng cao hơn trên mỗi FLOP kích hoạt.

### Vị thế của DeepSeek-V3

| Mô hình | Tổng tham số | Kích hoạt | Tỷ lệ | Cơ chế Attention | Ý tưởng mới đột phá |
|---|---|---|---|---|---|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4,25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5,5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | Mở rộng YaRN |

### Các bước phát triển tiếp nối: R1, V4

DeepSeek-R1 (2025) là một đợt huấn luyện năng lực suy luận (reasoning) dựa trên nền tảng khung xương (backbone) V3. R1 sử dụng nguyên vẹn kiến trúc này. Điểm thay đổi nằm ở quy trình hậu huấn luyện (post-training recipe) với học tăng cường (RL) quy mô lớn trên các tác vụ có thể kiểm chứng, chứ không phải kiến trúc tiền huấn luyện.

DeepSeek-V4 (nếu được phát hành) được kỳ vọng sẽ giữ nguyên MLA + MoE + MTP và bổ sung thêm DSA (DeepSeek Sparse Attention), phương pháp kế nhiệm của NSA từ Giai đoạn 10 · Bài học 17. Dòng dõi kiến trúc rất nhất quán: các sáng kiến cấp kiến trúc tích lũy dần; mỗi phiên bản lại tinh chỉnh thêm các núm vặn mới.

```figure
moe-routing
```

## Sử dụng trong thực tế

Tệp `code/main.py` là bộ tính toán tham số chuyên biệt cho cấu trúc của DeepSeek-V3. Hãy chạy thử, so sánh kết quả của nó với các con số trong bài báo, và thử nghiệm với các biến thể giả định (256 chuyên gia so với 512, top-8 so với top-16, số chiều MLA 512 so với 1024).

Những điểm cần quan sát:

- Tổng số tham số so với con số công bố 671B.
- Số tham số kích hoạt so với con số công bố 37B.
- Dung lượng KV cache ở ngữ cảnh 128k — bài toán so sánh MLA đối đầu GQA.
- Phân tích chi tiết từng tầng để thấy ngân sách tham số thực sự được phân bổ vào đâu.

## Đóng gói sản phẩm

Bài học này xây dựng `outputs/skill-deepseek-v3-reader.md`. Khi tiếp nhận một mô hình thuộc họ DeepSeek (V3, R1, hoặc bất kỳ biến thể tương lai nào), kỹ năng này sẽ đưa ra một bản phân tích kiến trúc chi tiết từng thành phần: nêu tên từng trường cấu hình, suy dẫn số lượng tham số theo từng bộ phận, và chỉ ra mô hình đang ứng dụng những sáng kiến nào trong số bốn đột phá đặc trưng của DeepSeek.

## Bài tập

1. Chạy `code/main.py`. So sánh ước tính tổng tham số của bộ tính toán với con số công bố 671B và chỉ ra nguồn gốc của phần sai lệch. Mục 2 của bài báo gốc có danh mục liệt kê đầy đủ.

2. Chỉnh sửa cấu hình để dùng số chiều MLA rank 256 thay vì 512. Tính toán kích thước KV cache thu được ở ngữ cảnh 128k. Nó giúp giảm bao nhiêu phần trăm dung lượng, và cái giá phải trả đối với năng lực biểu diễn trên từng đầu chú ý là gì?

3. So sánh định tuyến của DeepSeek-V3 (256 chuyên gia, top-8) với một biến thể giả định (512 chuyên gia, top-8). Tổng số tham số tăng lên; số tham số kích hoạt giữ nguyên. Về mặt lý thuyết, dung lượng chuyên gia bổ sung đem lại lợi ích gì, và nó gây tốn kém ra sao trong quá trình suy luận?

4. Đọc Mục 2.1 của báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437) về MLA. Giải thích trong ba câu tại sao các ma trận giải nén K và V có thể được "hấp thu" vào phép nhân ma trận liền sau để tối ưu hiệu quả trong thời gian suy luận.

5. DeepSeek-V3 sử dụng định dạng FP8 trong huấn luyện cho hầu hết các phép toán. Tính toán lượng bộ nhớ tiết kiệm được của FP8 so với BF16 khi lưu trữ 671 tỷ trọng số. Điều này tương tác như thế nào với ngân sách huấn luyện 14,8 nghìn tỷ token?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| MLA | "Multi-Head Latent Attention" | Nén K và V vào một biểu diễn tiềm ẩn hạng thấp dùng chung (kv_lora_rank, thường là 512), giải nén trên từng đầu on-the-fly; KV cache chỉ lưu vector tiềm ẩn |
| kv_lora_rank | "Chiều nén MLA" | Kích thước của không gian tiềm ẩn dùng chung cho K và V; DeepSeek-V3 dùng 512 |
| First k dense layers | "Các tầng đầu giữ nguyên dày đặc" | Vài tầng đầu tiên của mô hình MoE bỏ qua bộ định tuyến và chạy MLP dày đặc để đảm bảo tính ổn định |
| num_experts_per_tok | "Định tuyến Top-k" | Số lượng chuyên gia định tuyến được kích hoạt cho mỗi token; DeepSeek-V3 dùng 8 |
| Chuyên gia dùng chung (Shared experts) | "Chuyên gia luôn bật" | Các chuyên gia xử lý mọi token bất kể kết quả định tuyến; DeepSeek-V3 dùng 1 |
| Định tuyến không mất mát phụ trợ (Auxiliary-loss-free routing) | "Cân bằng tải chỉnh bias" | Các số hạng độ lệch trên từng chuyên gia được tinh chỉnh trong huấn luyện để giữ cân bằng tải mà không cần thêm hàm mất mát |
| Mô-đun MTP | "Đầu dự đoán bổ sung" | Khối Transformer dự đoán $t+2$ từ $h^{(1)}$ và $E(t+1)$; giúp huấn luyện dày đặc hơn và cung cấp mô hình phác thảo giải mã phỏng đoán miễn phí |
| DualPipe | "Đường ống hai chiều" | Lịch trình huấn luyện chồng lấn tính toán xuôi/ngược với giao tiếp all-to-all xuyên node |
| Tỷ lệ tham số kích hoạt (Active parameter ratio) | "Độ thưa (Sparsity)" | active_params / total_params; DeepSeek-V3 đạt mức 5,5% |
| Huấn luyện FP8 (FP8 training) | "Huấn luyện 8-bit" | Lưu trữ và thực thi nhiều phép tính huấn luyện ở dạng FP8; giảm gần một nửa bộ nhớ so với BF16 với tổn thất chất lượng không đáng kể |

## Đọc thêm

- [DeepSeek-AI — Báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — tài liệu đầy đủ về kiến trúc, huấn luyện và kết quả
- [Thẻ mô hình DeepSeek-V3 trên Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — các tệp cấu hình và ghi chú triển khai
- [Bài báo DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — công trình tiền thân giới thiệu MLA
- [Bài báo DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — bước phát triển tiếp nối tập trung vào huấn luyện suy luận trên kiến trúc V3
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — hướng phát triển tương lai cho cơ chế attention của họ DeepSeek
- [Kho mã nguồn DualPipe](https://github.com/deepseek-ai/DualPipe) — tài liệu tham khảo về lịch trình huấn luyện
