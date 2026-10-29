# Kiến Trúc Nhân Phục Vụ — PagedAttention, Gom Lô Liên Tục (Continuous Batching), Tiền Điền Phân Đoạn (Chunked Prefill)

> Thông lượng của các nhân phục vụ hiện đại dựa trên sự kết hợp cộng hưởng của ba cấu hình mặc định, chứ không phải một thủ thuật đơn lẻ. PagedAttention luôn được bật. Gom lô liên tục (continuous batching) nạp thêm các yêu cầu mới vào lô đang hoạt động giữa các bước lặp giải mã. Tiền điền phân đoạn (chunked prefill) chia nhỏ các câu lệnh prompt dài để các token giải mã không bao giờ bị bỏ đói tính toán. Khi kích hoạt cả ba kỹ thuật này, mô hình Llama 3.3 70B FP8 trên một card H100 SXM5 có thể đạt thông lượng 2.200-2.400 tok/s ở mức 128 yêu cầu đồng thời — cao hơn khoảng 25% so với cấu hình mặc định ban đầu của vLLM và gấp 3-4 lần so với một vòng lặp PyTorch tuần tự ngây thơ. Bài học này mổ xẻ bộ lập lịch (scheduler) và kernel chú ý của vLLM — nhân tham chiếu cho cả ba kỹ thuật trên — ở mức độ bạn có thể vẽ lại thành sơ đồ kiến trúc, và kết thúc bằng một bộ gom lô liên tục mô phỏng trong `code/main.py` có khả năng lập lịch tiền điền và giải mã theo đúng phương thức vLLM vận hành.

**Type:** Learn
**Languages:** Python (stdlib, trình mô phỏng bộ lập lịch gom lô liên tục)
**Prerequisites:** Phase 17 · 01 (Phục Vụ Mô Hình / Model Serving), Phase 11 (Kỹ Nghệ LLM / LLM Engineering)
**Time:** ~75 phút

## Mục tiêu Học tập

- Giải thích cơ chế PagedAttention dưới góc độ một bộ cấp phát bộ nhớ đệm KV (KV cache allocator): các khối (blocks), bảng khối (block tables), và lý do tại sao hiện tượng phân mảnh bộ nhớ luôn duy trì dưới 4% ở mức tải sản xuất thực tế.
- Vẽ sơ đồ gom lô liên tục ở cấp độ từng bước lặp (iteration level): cách các chuỗi đã hoàn thành rời khỏi lô và các chuỗi mới gia nhập mà không cần xả sạch toàn bộ lô hiện tại.
- Mô tả kỹ thuật tiền điền phân đoạn (chunked prefill) trong một câu duy nhất và chỉ ra chỉ số độ trễ nào được kỹ thuật này bảo vệ (gợi ý: đó là phần đuôi ITL của các lượt giải mã đồng thời, không phải thông lượng trung bình).
- Kiểm tra sự tương thích của các tính năng vLLM dựa trên ma trận tương thích cho phiên bản cụ thể của bạn trước khi bật đồng loạt mọi tối ưu hóa.

## Vấn đề Thực tế

Một vòng lặp phục vụ bằng PyTorch nguyên bản xử lý từng yêu cầu một cách tuần tự: tách từ (tokenize), tiền điền (prefill), giải mã (decode) cho đến khi gặp token kết thúc chuỗi (EOS), rồi trả về kết quả. Với một người dùng, cách này hoạt động tốt. Với một trăm người dùng, nó biến thành một hàng đợi của những con người kiên nhẫn. Cách khắc phục hiển nhiên — gom lô tĩnh (static batching) — buộc phải đệm (padding) mọi yêu cầu cho bằng với prompt dài nhất trong cửa sổ, đệm mọi lượt giải mã cho bằng với độ dài đầu ra dự kiến lớn nhất, và làm đình trệ toàn bộ lô để chờ chuỗi chậm chạp nhất hoàn thành. Bạn phải trả tiền cho phần dữ liệu đệm không bao giờ dùng đến, và các yêu cầu ngắn phải mòn mỏi chờ đợi các yêu cầu dài.

vLLM giải quyết đồng thời cả ba vấn đề. PagedAttention ngăn chặn hiện tượng phân mảnh bộ nhớ đệm KV (KV cache fragmentation) ngốn mất 60-80% dung lượng bộ nhớ GPU như cách cấp phát bộ nhớ liền kề truyền thống từng làm. Gom lô liên tục (continuous batching) cho phép các yêu cầu gia nhập và rời khỏi lô tính toán giữa mỗi bước lặp giải mã, giúp lô luôn được lấp đầy bằng các phép tính thực tế. Tiền điền phân đoạn (chunked prefill) chia nhỏ một prompt 32k token thành các lát cắt ~512 token đan xen nhịp nhàng với bước giải mã, nhờ đó một prompt quá dài sẽ không làm đóng băng toàn bộ tiến trình giải mã của các token khác trên GPU.

Cấu hình mặc định trong môi trường sản xuất năm 2026 là kích hoạt đồng thời cả ba kỹ thuật. Bạn cần hiểu rõ cơ chế hoạt động của từng thành phần bởi vì mọi chế độ hỏng hóc (failure modes) đều xuất phát từ bộ lập lịch, chứ không phải từ bản thân mô hình.

## Khái niệm

### PagedAttention như một hệ thống bộ nhớ ảo

Bộ đệm KV tiêu tốn dung lượng bằng `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` cho mỗi chuỗi. Đối với Llama 3.3 70B ở độ dài 8192 token, con số này xấp xỉ 1,25 GB cho mỗi chuỗi ở định dạng BF16. Nếu bạn cấp phát trước 8192 vị trí cho mọi yêu cầu nhưng yêu cầu trung bình chỉ dùng hết 1500 token, bạn đã lãng phí khoảng 82% lượng bộ nhớ băng thông cao (HBM) đã đặt trước. Gom lô cổ điển buộc bạn phải trả giá cho sự lãng phí này.

PagedAttention mượn ý tưởng từ hệ thống bộ nhớ ảo (virtual memory) của hệ điều hành. Bộ đệm KV không được cấp phát thành một vùng nhớ liền kề cho mỗi chuỗi. Thay vào đó, nó được chia thành các khối nhớ kích thước cố định (mặc định là 16 token). Mỗi chuỗi sở hữu một bảng khối (block table) để ánh xạ các vị trí token logic sang các định danh khối vật lý (physical block IDs). Khi một chuỗi phát triển vượt quá số khối hiện có, một khối mới sẽ được cấp thêm. Khi chuỗi kết thúc, các khối nhớ của nó được giải phóng và thu hồi về vùng nhớ chung (pool).

Mức lãng phí do phân mảnh giảm từ 60-80% (cổ điển) xuống dưới 4% (PagedAttention). Bạn không cần bật PagedAttention bằng bất kỳ cờ lệnh nào — đây là bộ cấp phát duy nhất mà vLLM cung cấp. Tham số tinh chỉnh chính là `--gpu-memory-utilization` (mặc định là 0.9), cho vLLM biết lượng HBM cần đặt trước cho các khối KV sau khi đã nạp trọng số mô hình và các kích hoạt (activations).

### Gom lô liên tục ở cấp độ vòng lặp

Cơ chế "gom lô động" (dynamic batching) trước đây thường chờ một khoảng thời gian nhất định (ví dụ 10 ms) để gom đủ một lô, sau đó chạy tiền điền + giải mã + giải mã... cho đến khi toàn bộ chuỗi hoàn thành. Những chuỗi ngắn hoàn thành sớm phải nằm chờ nhàn rỗi trong khi GPU vẫn phải gánh tải xử lý các chuỗi dài còn lại.

Gom lô liên tục (continuous batching) can thiệp vào giữa từng bước giải mã đơn lẻ. Gọi tập hợp các chuỗi đang xử lý là danh sách `RUNNING`. Tại mỗi bước lặp:

1. Bất kỳ chuỗi nào trong `RUNNING` vừa chạm mốc EOS hoặc đạt `max_tokens` sẽ lập tức bị loại bỏ.
2. Bộ lập lịch kiểm tra hàng đợi đang chờ (`WAITING`). Nếu còn khối nhớ KV trống, nó sẽ tiếp nhận các chuỗi mới (chạy tiền điền hoặc khôi phục xử lý).
3. Lượt lan truyền tiến (forward pass) được thực thi trên toàn bộ các chuỗi hiện diện trong `RUNNING`, sinh ra một token mới cho mỗi chuỗi.

Kích thước lô không bao giờ bị ép buộc phải đệm về một con số cố định. Các chuỗi ở những vị trí sinh từ khác nhau sẽ chia sẻ chung một lượt tính toán hợp nhất trên GPU. Trong vLLM năm 2026, cơ chế này được vận hành bởi `V1 scheduler`. Bất biến then chốt là: bộ lập lịch chạy một lần sau mỗi bước lặp giải mã, chứ không phải chạy một lần cho mỗi yêu cầu.

### Tiền điền phân đoạn bảo vệ phần đuôi ITL

Pha tiền điền (prefill) bị giới hạn bởi năng lực tính toán (compute-bound). Một prompt dài 32k token trên mô hình Llama 3.3 70B ngốn mất ~800 ms tính toán tiền điền thuần túy trên một card H100. Trong khi tiền điền đang chạy, các token giải mã của mọi chuỗi khác trong lô buộc phải đứng chờ. Trong một vòng lặp phục vụ, độ trễ đến token đầu tiên (TTFT) của một prompt dài vô tình biến thành một bước nhảy vọt về độ trễ giữa các token (Inter-Token Latency - ITL) đối với hàng chục người dùng khác.

Tiền điền phân đoạn (chunked prefill) chia nhỏ bước tiền điền thành các lát cắt có kích thước cố định (mặc định 512 token) và lập lịch xử lý từng lát cắt như một đơn vị độc lập. Xen giữa các lát cắt tiền điền, bộ lập lịch có thể thúc đẩy các chuỗi giải mã tiến thêm một token. Bạn đánh đổi một phần rất nhỏ độ trễ tiền điền tuyệt đối (vài mili-giây cho mỗi lát cắt) để đổi lấy sự suy giảm đáng kể hiện tượng giật cục trong thời gian giải mã. Chỉ số P99 ITL dưới tải hỗn hợp giảm ngoạn mục từ ~50 ms xuống chỉ còn ~15 ms trong các thử nghiệm công bố.

### Tác động tương hỗ giữa ba cấu hình mặc định

Cả ba tính năng này phụ thuộc chặt chẽ lẫn nhau. PagedAttention cung cấp cho bộ lập lịch một nguồn tài nguyên KV phân mảnh mịn để dễ dàng điều phối. Gom lô liên tục cần nguồn tài nguyên hạt mịn đó để việc tiếp nhận chuỗi mới không gây ra sự xáo trộn toàn cục trong bộ nhớ. Tiền điền phân đoạn là một quyết định do bộ lập lịch đưa ra trên chính danh sách `RUNNING` đó — nó chỉ là một chính sách lập lịch bổ sung, chứ không phải một hệ thống riêng biệt.

Bạn không nhất thiết phải ghi nhớ mọi cờ lệnh. Bạn cần hiểu mục tiêu tối ưu hóa của bộ lập lịch: đạt sản lượng hữu ích (goodput) tối đa trong phạm vi ngân sách khối nhớ KV, dưới sự điều phối phân lát của tiền điền phân đoạn.

### Kiểm tra ma trận tương thích

Hãy luôn đối chiếu từng tổ hợp tính năng với ma trận tương thích cho phiên bản vLLM chính xác mà bạn đang dùng trước khi bật tất cả cùng lúc, bởi vì khả năng phối hợp giữa chúng liên tục thay đổi qua các bản phát hành. Trong phiên bản v0.18.0, ma trận tính năng đánh dấu giải mã suy đoán (speculative decoding) hoàn toàn tương thích với tiền điền phân đoạn và lưu bộ đệm tiền tố (prefix caching); tài liệu giải mã suy đoán liệt kê hai điểm không tương thích đã biết: tính song song đường ống (pipeline parallelism) kéo dài tới v0.15.0, và suy đoán qua mô hình nháp (draft-model speculation) kéo dài tới v0.10.0. Đối với phương pháp sinh bản nháp, cấu hình chuẩn mặc định năm 2026 thường là EAGLE-3 (`"method": "eagle3"`), được trình bày chi tiết trong Phase 17 · 05.

### Những con số bạn cần ghi nhớ

- Llama 3.3 70B FP8, H100 SXM5, 128 yêu cầu đồng thời, bật cả ba tính năng: 2.200-2.400 tok/s.
- Cùng mô hình đó, cấu hình vLLM mặc định (chưa bật tiền điền phân đoạn): ~1.800 tok/s.
- Cùng mô hình đó, vòng lặp PyTorch tuần tự ngây thơ: ~600 tok/s.
- Lãng phí do phân mảnh KV dưới cơ chế PagedAttention ở tải sản xuất: <4%.
- P99 ITL dưới tải hỗn hợp: ~15 ms khi có tiền điền phân đoạn, ~50 ms nếu không có.

### Cấu trúc vận hành của bộ lập lịch

```python
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # Lập lịch các lát cắt tiền điền + các bước giải mã trong cùng một lô
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # ví dụ: lát cắt 512 token
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # một lượt gọi GPU hợp nhất duy nhất
```

Mã nguồn trong `code/main.py` chính là hiện thực hóa vòng lặp này bằng Python tiêu chuẩn với số lượng token giả lập và độ trễ chuyển tiếp mô phỏng. Việc chạy mã sẽ chứng minh rõ cách tiền điền phân đoạn duy trì sự sống cho các chuỗi giải mã ngay trong lúc xử lý một prompt tiền điền rất dài.

```figure
tensor-parallel
```

## Ứng Dụng Thực Tế (Use It)

Tệp `code/main.py` mô phỏng một bộ lập lịch theo phong cách vLLM với các tính năng có thể bật/tắt linh hoạt. Chạy mã để quan sát:

- Chế độ `NAIVE`: xử lý từng yêu cầu một cách đơn lẻ, không gom lô.
- Chế độ `STATIC`: đệm dữ liệu và chờ đợi, gom lô theo phương thức truyền thống.
- Chế độ `CONTINUOUS`: tiếp nhận và giải phóng yêu cầu ở cấp độ từng bước lặp.
- Chế độ `CONTINUOUS + CHUNKED`: các lát cắt tiền điền đan xen nhịp nhàng với giải mã.

Kết quả đầu ra hiển thị tổng thông lượng (tokens trên mỗi giây ảo), độ trễ TTFT trung bình và P99 ITL. Hàng `CONTINUOUS + CHUNKED` sẽ hoàn toàn áp đảo về hiệu năng khi xử lý lưu lượng hỗn hợp.

## Bàn Giao Thành Phẩm (Ship It)

Bài học này tạo ra tệp artifact `outputs/skill-vllm-scheduler-reader.md`. Dựa trên cấu hình phục vụ đầu vào (kích thước lô, tỷ lệ bộ nhớ KV, kích thước lát cắt tiền điền, cấu hình giải mã suy đoán), công cụ sẽ đưa ra chẩn đoán về bộ lập lịch, chỉ rõ thành phần nào trong ba cơ chế đang là điểm nghẽn và cần điều chỉnh những thông số gì.

## Bài Tập Thực Hành

1. Chạy `code/main.py`. So sánh chế độ `STATIC` với `CONTINUOUS` trên một khối lượng công việc kết hợp lẫn lộn các yêu cầu ngắn và dài. Khoảng cách thông lượng bắt nguồn từ đâu — hiệu quả tiền điền, hiệu quả giải mã, hay độ trễ phần đuôi?
2. Chỉnh sửa bộ lập lịch mô phỏng để bổ sung tham số `--max-num-batched-tokens`. Giá trị hợp lý cho một card H100 chạy Llama 3.3 70B FP8 là bao nhiêu? (Gợi ý: đây là hàm số phụ thuộc vào kích thước khối KV và số khối còn trống, không phụ thuộc đơn thuần vào dung lượng HBM thô.)
3. Đọc lại ghi chú phát hành (release notes) của vLLM v0.18.0. Những tổ hợp cờ lệnh nào xung đột loại trừ lẫn nhau? Hãy lập danh sách chi tiết.
4. Tính toán độ lãng phí do phân mảnh bộ đệm KV cho một tập dữ liệu vết gồm 1.000 yêu cầu có độ dài đầu ra trung bình 1.500 token, độ lệch chuẩn 600 token, dưới hai cơ chế: (a) cấp phát vùng nhớ liền kề cho mỗi yêu cầu ở mức tối đa 8192 token, (b) PagedAttention với kích thước khối 16 token.
5. Giải thích trong một đoạn văn ngắn tại sao tiền điền phân đoạn cải thiện chỉ số P99 ITL nhưng không trực tiếp làm tăng thông lượng khi xét độc lập. Trong thực tế, lợi ích về mặt thông lượng bắt nguồn từ đâu?

## Thuật Ngữ Then Chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật thực tế |
|-----------|----------------------|--------------------------|
| PagedAttention | "Thủ thuật KV" | Bộ cấp phát khối nhớ cố định cho bộ đệm KV; giảm độ phân mảnh xuống <4% |
| Bảng khối (Block table) | "Bảng phân trang" | Bản đồ ánh xạ từng chuỗi từ vị trí token logic sang khối nhớ KV vật lý |
| Gom lô liên tục (Continuous batching) | "Gom lô động chuẩn xác" | Quyết định tiếp nhận/giải phóng yêu cầu được thực hiện sau mỗi bước lặp giải mã |
| Tiền điền phân đoạn (Chunked prefill) | "Cắt nhỏ tiền điền" | Chia tiền điền dài thành các lát cắt 512 token đan xen với các bước giải mã |
| TTFT | "Thời gian đến token đầu" | Tổng thời gian tiền điền + hàng đợi + mạng; bị chi phối bởi tiền điền ở prompt dài |
| ITL | "Độ trễ giữa các token" | Thời gian giữa hai token giải mã liên tiếp; bị chi phối bởi kích thước lô |
| Sản lượng hữu ích (Goodput) | "Thông lượng đạt chuẩn SLO" | Số yêu cầu/giây vừa đáp ứng mục tiêu TTFT vừa đạt mục tiêu ITL (Phase 17 · 08 đo lường dưới dạng tỷ lệ phần trăm yêu cầu) |
| Bộ lập lịch V1 (V1 scheduler) | "Bộ lập lịch mới" | Bộ lập lịch năm 2026 của vLLM; kết hợp gom lô liên tục với tiền điền phân đoạn |
| `--gpu-memory-utilization` | "Núm vặn bộ nhớ" | Tỷ lệ dung lượng HBM được dành riêng cho các khối KV sau khi đã trừ trọng số và kích hoạt |

## Tài Liệu Đọc Thêm

- [Tài liệu vLLM — Giải mã Suy đoán (Speculative Decoding)](https://docs.vllm.ai/en/latest/features/spec_decode/) — Tài liệu chính thức về tính tương thích giữa tiền điền phân đoạn và giải mã suy đoán.
- [Ghi chú Phát hành vLLM (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — Nhịp độ phát hành năm 2026 và hành vi đặc thù theo từng phiên bản.
- [Blog vLLM — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — Bài viết nền tảng định hình tư duy về cơ chế cấp phát bộ nhớ.
- [Bài báo PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — Phân tích toán học về phân mảnh và thiết kế bộ lập lịch.
- [Aleksa Gordic — Khám phá Bên trong vLLM](https://www.aleksagordic.com/blog/vllm) — Phân tích chi tiết về bộ lập lịch V1 với biểu đồ flame graph.
