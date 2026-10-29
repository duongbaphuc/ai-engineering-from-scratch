# Phục Vụ Với Bộ Đệm Tiền Tố — RadixAttention Và Tái Sử Dụng Bộ Đệm KV

> Khi coi bộ nhớ đệm KV (KV cache) là một tài nguyên hạng nhất có thể tái sử dụng và được lưu trữ trong một cây cơ số (radix tree), toàn bộ cơ chế lập lịch sẽ thay đổi: thay vì lập lịch theo nguyên tắc đến trước phục vụ trước (FCFS - First-Come, First-Served) như vLLM, một bộ lập lịch nhận thức bộ đệm (cache-aware scheduler) sẽ ưu tiên các yêu cầu có tiền tố chung dài hơn — thực chất là cơ chế duyệt cây cơ số theo chiều sâu (depth-first radix traversal) nhằm giữ cho các nhánh "nóng" luôn thường trú trong bộ nhớ HBM. SGLang là nhân phục vụ được thiết kế xoay quanh tư tưởng đột phá này. Trên mô hình Llama 3.1 8B với các câu lệnh 1K token kiểu ShareGPT, SGLang đạt thông lượng ~16.200 tok/s so với ~12.500 tok/s của vLLM, tạo ra lợi thế ~29%. Trên các khối lượng công việc RAG có tiền tố lặp lại nhiều, khoảng cách hiệu năng có thể lên tới 6,4 lần. Trên các tác vụ nhân bản giọng nói, tỷ lệ trúng bộ đệm (cache hit rate) vượt mốc 86%. Nền tảng này đã được triển khai trên hơn 400.000 GPU vào năm 2026 tại xAI, LinkedIn, Cursor, Oracle, GCP, Azure và AWS. Điểm mấu chốt là con số tăng tốc 6,4x sẽ hoàn toàn biến mất nếu thứ tự sắp xếp tiền tố không nhất quán — trật tự cấu trúc câu lệnh chính là đòn bẩy quyền lực nhất của người kỹ sư.

**Type:** Learn
**Languages:** Python (stdlib, bộ nhớ đệm cây cơ số mô phỏng + bộ lập lịch nhận thức bộ đệm)
**Prerequisites:** Phase 17 · 04 (Kiến Trúc Nhân Phục Vụ / Serving Engine Internals), Phase 14 (RAG Dạng Tác Tử / Agentic RAG)
**Time:** ~75 phút

## Mục tiêu Học tập

- Vẽ sơ đồ nguyên lý RadixAttention: cách các tiền tố được lưu trữ trong một cây cơ số và cách các khối nhớ KV được chia sẻ giữa các chuỗi bắt nguồn từ cùng một nhánh cây.
- Giải thích cơ chế lập lịch nhận thức bộ đệm (cache-aware scheduling) và lý do tại sao giải thuật FCFS là sai lầm đối với lưu lượng có tiền tố lặp lại nhiều.
- Tính toán tốc độ tăng tốc kỳ vọng cho một khối lượng công việc dựa trên tỷ lệ trúng bộ đệm tiền tố và phân phối độ dài câu lệnh prompt.
- Nêu rõ quy tắc kỷ luật sắp xếp câu lệnh giúp hiện thực hóa con số tăng tốc 6,4x thay vì đánh mất toàn bộ lợi thế tiềm năng.

## Vấn đề Thực tế

Mô hình phục vụ truyền thống coi câu lệnh prompt của mỗi yêu cầu như một hộp đen độc lập. Ngay cả khi 5.000 yêu cầu RAG đều bắt đầu bằng cùng một prompt hệ thống dài 2.000 token kèm chung một đoạn mở đầu tài liệu truy xuất, vLLM vẫn thực hiện tính toán tiền điền (prefill) cho đoạn tiền tố 2.000 token đó đúng 5.000 lần. GPU liên tục lặp lại các phép tính giống hệt nhau một cách lãng phí.

Một quan sát thực tế: các câu lệnh trong tác vụ dạng tác tử (agentic) và RAG hầu như luôn chia sẻ những đoạn tiền tố rất dài. Câu lệnh hệ thống, lược đồ định nghĩa công cụ (tool schemas), các ví dụ vài mẫu (few-shot examples), phần tiêu đề truy xuất và lịch sử hội thoại — tất cả đều lặp đi lặp lại qua các yêu cầu. Nếu bạn lưu lại bộ nhớ đệm KV cho đoạn tiền tố đó một lần và tái sử dụng, bạn sẽ không bao giờ phải chạy tiền điền lại từ đầu.

RadixAttention hiện thực hóa chính xác ý tưởng này. Các token được lập chỉ mục trong một cây cơ số (radix tree); mỗi nút trên cây sở hữu các khối nhớ KV tương ứng với chuỗi token trên đường đi từ nút gốc đến nút đó. Khi một yêu cầu mới xuất hiện, hệ thống sẽ duyệt dọc theo cây: bất kỳ nút nào có token trùng khớp đều được tái sử dụng trực tiếp các khối nhớ KV đã có. Chi phí tiền điền giờ đây chỉ tỷ lệ thuận với phần hậu tố "mới", chứ không còn phụ thuộc vào toàn bộ độ dài của prompt.

Thách thức lớn nhất nằm ở khâu lập lịch. Nếu hai yêu cầu cùng chia sẻ một tiền tố dài 2.000 token và một yêu cầu thứ ba chỉ chia sẻ 200 token của tiền tố đó, bạn sẽ muốn phục vụ đồng thời hai yêu cầu có tiền tố dài trước để giữ cho nhánh 2.000 token đó nằm yên trong bộ nhớ HBM. Giải thuật FCFS làm điều hoàn toàn ngược lại — nó phục vụ bất kỳ ai đến trước, dẫn đến nguy cơ trục xuất (evict) nhánh tiền tố "nóng" ra khỏi bộ nhớ ngay trước khi yêu cầu có tiền tố dài tiếp theo ập đến.

## Khái niệm

### Cây cơ số dưới vai trò chỉ mục bộ đệm KV

Cây cơ số (radix tree hay compact trie) là cấu trúc lưu trữ các chuỗi token. Mỗi nút sở hữu một đoạn dải token và các khối nhớ KV đã được tính toán cho dải token đó. Các nút con sẽ mở rộng chuỗi thêm một hoặc nhiều token tiếp theo.

```text
root
 |- "You are a helpful assistant..."  (2.000 tokens, 124 khối KV)
      |- "Context: <doc A>..."        (500 tokens, 31 khối)
           |- "Question: Alice..."    (80 tokens, 5 khối)
           |- "Question: Bob..."      (95 tokens, 6 khối)
      |- "Context: <doc B>..."        (520 tokens, 33 khối)
```

Một yêu cầu mới đến với nội dung gồm câu lệnh hệ thống + "Context: <doc A>" + "Question: Carol". Bộ lập lịch bắt đầu duyệt cây: tiền tố hệ thống trùng khớp (tái sử dụng 124 khối KV), nhánh doc-A trùng khớp (tái sử dụng 31 khối KV), sau đó hệ thống chỉ cần cấp phát các khối mới cho phần "Question: Carol" (4 khối). Chi phí tiền điền: chỉ tính toán 4 khối token mới. Nếu không có cây cơ số: phải tính toán toàn bộ 160 khối. Tiết kiệm tới ~40 lần chi phí tiền điền.

### Lập lịch nhận thức bộ đệm (Cache-aware scheduling)

Khả năng tái sử dụng dựa trên cây cơ số sẽ trở nên vô nghĩa nếu bộ nhớ đệm bị xáo trộn liên tục (cache thrashing). Hai chính sách điều phối then chốt bao gồm:

1. **Điều phối duyệt theo chiều sâu (Depth-first dispatch)**. Khi chọn yêu cầu tiếp theo từ hàng đợi, ưu tiên các yêu cầu bắt nguồn từ cùng nhánh cây với tập hợp các chuỗi đang chạy hiện tại. Điều này giữ cho nhánh "nóng" luôn được ghim chặt trong bộ nhớ.
2. **Thu hồi LRU ở cấp độ nhánh cây, không phải cấp độ khối nhớ**. Trục xuất toàn bộ nhánh cây (bắt đầu từ các lá ít được sử dụng nhất) thay vì thu hồi từng khối nhớ đơn lẻ rời rạc, nhờ đó cấu trúc của bộ đệm luôn khớp hoàn hảo với hình thái của cây cơ số.

Giải thuật FCFS phá vỡ cả hai nguyên tắc trên. Một yêu cầu chia sẻ 2.000 token có thể phải đứng sau một yêu cầu chỉ chia sẻ 50 token, và nhánh 2.000 token có thể bị thu hồi chỉ để nhường chỗ cho yêu cầu 50 token kia.

### Những con số điểm chuẩn bạn cần ghi nhớ

- Llama 3.1 8B, H100, tập prompt ShareGPT 1K: SGLang đạt ~16.200 tok/s so với vLLM ~12.500 tok/s (lợi thế vượt trội ~29%).
- Tác vụ RAG nhiều tiền tố (cùng hệ thống + cùng tài liệu, chỉ khác câu hỏi): tốc độ tăng lên tới 6,4 lần trên SGLang.
- Khối lượng công việc nhân bản giọng nói: tỷ lệ trúng bộ đệm tiền tố đạt 86,4%.
- Tỷ lệ trúng bộ đệm trong môi trường sản xuất thực tế của khách hàng SGLang: đạt từ 50-99% tùy thuộc vào mức độ kỷ luật trong cấu trúc prompt.
- Triển khai trên hơn 400.000 GPU vào năm 2026.

### Cạm bẫy trật tự sắp xếp câu lệnh

Con số tăng tốc 6,4x phụ thuộc hoàn toàn vào tính nhất quán trong trật tự sắp xếp mẫu câu lệnh (prompt-template ordering). Nếu ứng dụng client của bạn dựng prompt theo dạng `[hệ thống, công cụ, ngữ cảnh, lịch sử, câu hỏi]` trong một số yêu cầu nhưng lại đổi thành `[hệ thống, ngữ cảnh, công cụ, lịch sử, câu hỏi]` trong các yêu cầu khác, cây cơ số sẽ không thể tìm thấy tiền tố chung. Những gì mà con người thấy là "tiền tố giống nhau" thì đối với cây cơ số lại là hai chuỗi hoàn toàn dị biệt.

Đòn bẩy của kỹ sư: mẫu câu lệnh chính là một khóa bộ nhớ đệm (cache key). Hãy cố định trật tự đó. Đặt mọi nội dung bất biến (hệ thống, công cụ, lược đồ) lên đầu tiên. Đặt ngữ cảnh truy xuất ở vị trí tiếp theo. Đặt câu hỏi của người dùng ở vị trí cuối cùng. Tuyệt đối không chèn xen kẽ nội dung động vào giữa phần tiền tố tĩnh.

Trường hợp thực tế từ nghiên cứu: việc di dời các đoạn nội dung động ra khỏi tiền tố có thể lưu đệm đã giúp một hệ thống tăng vọt tỷ lệ trúng bộ đệm từ 7% lên 74% chỉ sau một lần chỉnh sửa cấu trúc.

### Khi nào RadixAttention thắng thế và thất bại

Chiến thắng vượt trội:
- Tác vụ RAG (cùng đoạn mở đầu truy xuất, khác câu hỏi).
- Hệ thống tác tử (cùng lược đồ công cụ, khác truy vấn).
- Ứng dụng hội thoại với câu lệnh hệ thống dài.
- Khối lượng công việc giọng nói / thị giác máy tính với các đoạn mở đầu lặp lại cố định.

Thất bại (hiệu năng quay về ngang mức vLLM):
- Tác vụ sinh đơn lẻ với prompt độc nhất hoàn toàn (tự động hoàn thành mã nguồn, hội thoại mở không có câu lệnh hệ thống).
- Prompt động nơi mọi yêu cầu đều xen kẽ nội dung biến đổi ngay từ đầu chuỗi.

### Tại sao đây là bài toán lập lịch, không chỉ là thủ thuật kernel

Bạn hoàn toàn có thể triển khai tái sử dụng KV dưới dạng một thủ thuật kernel cấp thấp. Tuy nhiên, phát hiện sâu sắc của SGLang là việc tái sử dụng chỉ thực sự mang lại hiệu quả kinh tế nếu bộ lập lịch giữ được nhánh "nóng" thường trú trong bộ nhớ. Một chính sách ngây thơ kiểu "tái sử dụng nếu có sẵn" sẽ khiến bộ đệm bị xáo trộn và đào thải liên tục dưới tải hỗn hợp. Chính bộ lập lịch được lập chỉ mục bằng cây cơ số mới là yếu tố biến thủ thuật kernel thành lợi thế cạnh tranh 29% trong môi trường sản xuất.

### Mối quan hệ tương hỗ với vLLM

Hai hệ thống này không hoàn toàn loại trừ lẫn nhau. Vào năm 2026, vLLM đã tích hợp thêm tính năng lưu bộ đệm tiền tố (`--enable-prefix-caching`) cùng một bộ định tuyến nhận thức bộ đệm (vLLM Router viết bằng Rust). Khoảng cách hiệu năng đã được thu hẹp nhưng không biến mất hoàn toàn — toàn bộ ngăn xếp của SGLang được thiết kế ưu tiên cây cơ số ngay từ lõi; trong khi vLLM ghép nối tính năng này vào sau. Đối với các khối lượng công việc bị chi phối bởi việc tái sử dụng tiền tố, SGLang vẫn là lựa chọn mặc định hàng đầu. Đối với các tác vụ phục vụ thông dụng không có tiền tố lặp lại rõ rệt, vLLM vẫn duy trì hiệu năng tương đương hoặc vượt trội hơn.

```figure
roofline
```

## Ứng Dụng Thực Tế (Use It)

Tệp `code/main.py` hiện thực hóa một bộ nhớ đệm KV bằng cây cơ số mô phỏng cùng một bộ lập lịch hỗ trợ hai chính sách: FCFS và nhận thức bộ đệm. Chạy cùng một khối lượng công việc qua cả hai giải thuật, báo cáo tỷ lệ trúng bộ đệm tiền tố và mức chênh lệch thông lượng. Sau đó chạy một khối lượng công việc có "trật tự bị xáo trộn" để chứng kiến sự sụp đổ của hệ số tăng tốc 6,4x.

Mô hình [Bên Trong GPU Phục Vụ LLM](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?workload=shared&shared=4096) minh họa bộ đệm cây cơ số trên một lượt chạy card L4 thực tế. Bốn nhóm người dùng mỗi nhóm sở hữu prompt hệ thống 4.096 token riêng biệt. Yêu cầu đầu tiên của một nhóm sẽ tính toán prompt này. Trong khi prompt đó còn lưu trong bộ đệm, các yêu cầu tiếp theo của nhóm chỉ cần tính toán câu hỏi 128 token. Hãy thử tắt bộ đệm cây cơ số để thấy thời gian token đầu tiên trung vị tăng vọt từ 479 ms lên 14.706 ms.

## Bàn Giao Thành Phẩm (Ship It)

Bài học này tạo ra tệp artifact `outputs/skill-radix-scheduler-advisor.md`. Dựa trên mô tả khối lượng công việc (cấu trúc mẫu prompt, mô hình truy xuất, số lượng tenant đồng thời), công cụ sẽ đưa ra khuyến nghị về trật tự sắp xếp câu lệnh và quyết định có nên chuyển đổi sang SGLang hay không.

## Bài Tập Thực Hành

1. Chạy `code/main.py`. So sánh FCFS và bộ lập lịch nhận thức bộ đệm trên cùng một khối lượng công việc. Mức chênh lệch thông lượng bắt nguồn từ đâu — tiết kiệm thời gian tiền điền, tiết kiệm giải mã hay giảm thời gian chờ hàng đợi?
2. Chỉnh sửa khối lượng công việc sao cho các câu lệnh hoán vị ngẫu nhiên giữa `[hệ thống, công cụ, ngữ cảnh]`. Chạy lại mã nguồn. Điều gì xảy ra với tỷ lệ trúng bộ đệm? Giải thích nguyên nhân.
3. Tính toán dung lượng bộ nhớ HBM cần thiết để duy trì một câu lệnh hệ thống dài 2.000 token thường trú như một nhánh cây cơ số trên Llama 3.1 8B. So sánh chi phí này với một lô 16 chuỗi khi không áp dụng tái sử dụng tiền tố.
4. Đọc bài báo SGLang RadixAttention. Giải thích trong ba câu ngắn gọn tại sao cơ chế thu hồi LRU theo cấu trúc nhánh cây lại vượt trội hơn LRU theo từng khối nhớ rời rạc khi chịu tải lặp tiền tố cao.
5. Một khách hàng báo cáo tỷ lệ trúng bộ đệm chỉ đạt khiêm tốn 8%. Hãy nêu ba nguyên nhân tiềm ẩn phổ biến nhất và các bước chẩn đoán bạn sẽ thực hiện cho từng nguyên nhân.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật thực tế |
|-----------|----------------------|--------------------------|
| RadixAttention | "Tính năng của SGLang" | Bộ đệm KV được lập chỉ mục dưới dạng cây cơ số để các tiền tố dùng chung khối nhớ |
| Cây cơ số (Radix tree) | "Cây tiền tố thu gọn" | Cấu trúc cây trong đó mỗi nút sở hữu một dải token và các khối nhớ KV tương ứng |
| Bộ lập lịch nhận thức bộ đệm (Cache-aware scheduler) | "Ưu tiên nhánh nóng" | Bộ lập lịch ưu tiên các yêu cầu có tiền tố trùng khớp với nhánh đang thường trú |
| Tỷ lệ trúng bộ đệm tiền tố (Prefix-cache hit rate) | "Tỷ lệ prompt miễn phí" | Tỷ lệ phần trăm token trong prompt được phục vụ trực tiếp từ các khối nhớ KV tái sử dụng |
| FCFS | "Đến trước phục vụ trước" | Giải thuật lập lịch mặc định làm phá vỡ tính cục bộ của tiền tố trong bộ nhớ |
| Thu hồi LRU cấp độ nhánh (Branch-level LRU) | "Trục xuất nhánh lá" | Chính sách thu hồi bộ đệm được thiết kế phù hợp với cấu trúc phân cấp của cây cơ số |
| Trật tự mẫu câu lệnh (Prompt template ordering) | "Khóa bộ nhớ đệm" | Thứ tự sắp xếp các thành phần trong prompt quyết định khả năng chia sẻ của cây cơ số |
| Ghim prompt hệ thống (System prompt pinning) | "Tiền tố thường trú" | Giữ phần câu lệnh hệ thống bất biến luôn nằm trong HBM để tránh hiện tượng đào thải bộ đệm |

## Tài Liệu Đọc Thêm

- [GitHub của SGLang](https://github.com/sgl-project/sglang) — Mã nguồn và tài liệu kỹ thuật.
- [Tài liệu chính thức SGLang](https://sgl-project.github.io/) — Chi tiết về cơ chế RadixAttention và giải thuật lập lịch.
- [Bài báo SGLang — Lập Trình Hiệu Quả Các Mô Hình Ngôn Ngữ Lớn (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — Tài liệu thiết kế chuẩn mực gốc.
- [Blog LMSYS — SGLang với RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — Các con số đo lường hiệu năng và lý giải kiến trúc bộ lập lịch.
- [Tài liệu vLLM — Lưu Bộ Đệm Tiền Tố (Prefix Caching)](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) — Hiện thực hóa tương tự của vLLM để tiện đối chiếu so sánh.
