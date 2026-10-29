# Suy luận bất đồng bộ và Hogwild! (Async and Hogwild! Inference)

> Giải mã phỏng đoán (speculative decoding - Giai đoạn 10 · Bài học 15) song song hóa các token bên trong một chuỗi đơn lẻ. Các framework đa tác tử (multi-agent frameworks) song song hóa trên toàn bộ chuỗi nhưng bắt buộc phải có sự điều phối tường minh (bỏ phiếu, chia tách nhiệm vụ con). Kỹ thuật suy luận Hogwild! (Rodionov et al., arXiv:2504.06261) thực hiện một hướng tiếp cận hoàn toàn khác: chạy $N$ thực thể của cùng một LLM song song trên một bộ nhớ đệm KV (KV cache) DÙNG CHUNG. Mỗi tiến trình làm việc (worker) nhìn thấy các token do các worker khác sinh ra ngay lập tức. Các mô hình suy luận hiện đại — QwQ, DeepSeek-R1 — có khả năng tự điều phối thông qua bộ nhớ đệm dùng chung đó mà không cần bất kỳ sự tinh chỉnh (fine-tuning) nào. Phương pháp này mang tính thử nghiệm nhưng mở ra một trục song song hóa suy luận hoàn toàn mới, vuông góc (orthogonal) với giải mã phỏng đoán. Bài học này sẽ xây dựng một trình mô phỏng Hogwild! hai worker bằng Python thư viện chuẩn và giải thích tại sao sự cộng tác trên bộ nhớ đệm dùng chung lại tự hình thành (emergent) từ năng lực suy luận sẵn có của mô hình.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Mục tiêu học tập

- Mô tả ba cấu trúc liên kết LLM song song phổ biến (bỏ phiếu - voting, chia tác vụ con - sub-task, Hogwild!) và chỉ rõ mỗi cấu trúc giải quyết bài toán nào.
- Trình bày kiến trúc cốt lõi của Hogwild!: nhiều worker, một bộ nhớ đệm KV dùng chung, sự phối hợp tự hình thành thông qua tự tạo gợi ý (self-prompting).
- Tính toán mức tăng tốc thời gian thực (wall-time speedup) của Hogwild! dưới dạng hàm số theo số lượng worker $N$, tỷ lệ song song hóa cấp tác vụ $p$, và chi phí phối hợp phụ trội $c$.
- Cài đặt một trình mô phỏng Hogwild! hai worker trên một bài toán mô phỏng và quan sát sự tự phân chia nhiệm vụ xuất hiện tự nhiên.

## Vấn đề đặt ra

Các LLM hiện đại giải quyết các bài toán hóc búa bằng cách sinh ra các chuỗi suy luận dài (chains of thought) — chuỗi 5.000 token logic từng bước là chuyện phổ biến, và hàng chục nghìn token thường xuyên xuất hiện trên các bài toán toán học chuyên sâu. Ở tốc độ giải mã 35 token/giây trên một mô hình 70B, 50k token sẽ tiêu tốn tới 24 phút. Khi đó, tính tương tác thời gian thực của mô hình hoàn toàn biến mất.

Giải mã phỏng đoán (Giai đoạn 10 · Bài học 15) đem lại tốc độ tăng tốc khoảng 3-5 lần bằng cách song song hóa bên trong một chuỗi đơn lẻ. Vượt qua giới hạn đó, sự phụ thuộc tuần tự của quá trình giải mã tự hồi quy (autoregressive decoding) tạo nên một bức trần cứng. Mỗi token mới luôn phụ thuộc vào mọi token đứng trước nó.

Một câu hỏi hiển nhiên được đặt ra: liệu chúng ta có thể song song hóa xuyên suốt nhiều chuỗi hay không? Chạy nhiều bản sao của cùng một mô hình trên cùng một bài toán, để chúng cộng tác và tự phân chia công việc cho nhau?

Các công trình trước đây gồm có: tập hợp bỏ phiếu (voting ensembles - chạy $N$ mô hình rồi lấy câu trả lời theo đa số), cây tư duy (tree-of-thought - rẽ nhánh các luồng suy luận rồi tái kết hợp), và các framework đa tác tử (phân công mỗi tác tử một nhiệm vụ con, sử dụng một bộ điều phối trung tâm). Tất cả các cách này đều hữu ích trong những miền tác vụ cụ thể. Nhưng chúng cũng đồng thời kéo theo bộ máy điều phối tường minh cồng kềnh — quy tắc bỏ phiếu, logic rẽ nhánh và cắt tỉa, các giao thức truyền thông điệp giữa các tác tử.

Kỹ thuật suy luận Hogwild! chọn một hướng tiếp cận khác biệt. $N$ worker cùng chia sẻ một bộ nhớ đệm KV duy nhất. Mỗi worker nhìn thấy ngay lập tức các token do mọi worker khác sinh ra, như thể đó chính là ngữ cảnh tự thân của nó. Các worker — mà không cần qua bất kỳ đợt huấn luyện hay tinh chỉnh nào — tự tìm ra cách phân chia khối lượng công việc. Các mô hình suy luận hiện đại (QwQ, DeepSeek-R1, chế độ suy luận của họ Claude) có thể đọc bộ đệm dùng chung và tự đưa ra nhận định kiểu như: "Tôi thấy worker 2 đã giải quyết xong trường hợp cơ sở, vậy tôi sẽ tập trung xử lý bước quy nạp."

Mức tăng tốc phụ thuộc chặt chẽ vào bản chất khối lượng công việc và vẫn đang trong giai đoạn thử nghiệm tính đến tháng 4/2026. Tuy nhiên, ý tưởng này rất đáng để tìm hiểu vì nó mở ra một trục song song hóa suy luận hoàn toàn mới.

## Khái niệm

### Kiến trúc thiết lập

Khởi tạo $N$ tiến trình worker cùng chạy một mô hình LLM giống nhau. Thay vì duy trì các bộ nhớ đệm KV riêng cho từng worker, hệ thống duy trì MỘT bộ nhớ đệm dùng chung duy nhất. Khi worker $i$ sinh ra token $t_j$, token này được ghi ngay vào bộ đệm dùng chung ở vị trí tiếp theo. Khi worker $k$ thực hiện bước tiếp theo của nó, nó đọc trạng thái hiện tại của bộ đệm (bao gồm toàn bộ những gì cả $N$ worker đã sinh ra tính đến thời điểm đó).

Tại thời điểm mỗi bước tính, các worker chạy đua để ghi token. Không có chỉ số vị trí riêng cho từng worker — bộ đệm là một chuỗi tuần tự duy nhất liên tục mở rộng. Thứ tự được xác định bởi thời điểm thao tác ghi được hoàn tất.

### Tại sao sự phối hợp lại tự hình thành

Các worker chia sẻ chung một câu nhắc (prompt). Thường là một câu đại loại như: "Bạn là một trong $N$ thực thể đang làm việc cùng nhau trên bài toán này. Mỗi thực thể đều đọc bộ nhớ dùng chung và có thể thấy những gì các thực thể khác đã viết. Hãy tránh làm các công việc trùng lặp." Câu nhắc này kết hợp với bộ nhớ đệm dùng chung là đủ. Các mô hình suy luận đọc bộ đệm, nhận biết những phần nào của bài toán đã được thử sức, và (thường xuyên nhưng không phải luôn luôn) chủ động rẽ hướng sang các phần chưa được khai phá.

Bài báo Hogwild! (Rodionov et al., 2025) ghi nhận những quan sát thực nghiệm như:

- Các worker lập kế hoạch và truyền đạt kế hoạch tới các worker khác thông qua bộ đệm.
- Các worker phát hiện lỗi sai trong lập luận của worker khác và chỉ rõ lỗi đó.
- Các worker tự điều chỉnh khi một kế hoạch bị bế tắc và đề xuất phương án thay thế.
- Khi được nhắc nhở kiểm tra sự dư thừa, các worker nhận diện được sự trùng lặp và chuyển hướng giải quyết.

Toàn bộ những điều này không đòi hỏi bất kỳ sự tinh chỉnh nào. Hành vi tự hình thành này bắt nguồn từ chính năng lực suy luận mà mô hình vốn dĩ đã sở hữu.

### Nguồn gốc tên gọi

Tên gọi của bài báo bắt nguồn từ thuật toán Hogwild! SGD (Recht et al., 2011), một thuật toán tối ưu hóa cập nhật bất đồng bộ. Phép tương tự ở đây là: các worker bất đồng bộ trong SGD đều ghi vào một vector tham số dùng chung; các worker trong Hogwild! Inference đều ghi vào một bộ nhớ đệm KV dùng chung. Cả hai đều dựa vào sự hội tụ thực nghiệm thay vì các đảm bảo đồng bộ hóa cứng nhắc.

### RoPE giúp kỹ thuật này trở nên khả thi

Nhúng vị trí xoay (Rotary Position Embeddings - RoPE, Su et al. 2021) mã hóa thông tin vị trí thông qua các phép xoay trong vector Q và K. Bởi vì điểm chú ý giữa một query đã xoay và một key đã xoay chỉ phụ thuộc vào khoảng cách tương đối giữa chúng, nên toàn bộ một khối dữ liệu trong bộ nhớ đệm có thể chuyển dịch sang một độ lệch vị trí (offset) mới mà không cần tính toán lại: chỉ cần xoay query thay vì phải xoay lại từng key đã lưu trong đệm. Kỹ thuật suy luận Hogwild! thực hiện điều này một lần cho mỗi khối đệm: đối với query ở vị trí $i_q$ và một khối bắt đầu tại $i_k$ theo góc nhìn của worker đó, query chỉ cần xoay theo khoảng cách tương đối $i_q - i_k$. Nhờ vậy, mỗi worker có thể nhìn thấy các khối đệm dùng chung theo thứ tự riêng của nó. Mẹo này hoạt động với sự dịch chuyển đồng nhất của một khối liên tục. Việc di chuyển từng token đơn lẻ đến các vị trí khác nhau sẽ làm thay đổi phép xoay của chúng và vẫn đòi hỏi phải tính toán lại.

Trong một mô hình sử dụng vị trí học được (learned-position) hoặc vị trí tuyệt đối, Hogwild! sẽ buộc phải vô hiệu hóa và tính toán lại bộ đệm sau mỗi lượt ghi đồng thời. RoPE giúp bộ nhớ đệm giữ được sự ổn định.

### Phép tính thời gian thực

Gọi $T_{serial}$ là thời gian một worker giải quyết bài toán một mình. Gọi $p$ là tỷ lệ công việc có thể song song hóa ở cấp độ tác vụ. Gọi $c$ là chi phí phối hợp phụ trội trên mỗi bước (đọc bộ đệm mở rộng, quyết định nội dung cần viết tiếp).

Thời gian chạy đơn worker: $T_{serial}$.
Thời gian chạy $N$ worker Hogwild! khi chi phí phối hợp bằng 0: $T_{serial} \times ((1 - p) + p / N)$. Đây là định luật Amdahl kinh điển.
Khi tính cả chi phí phối hợp: $T_{serial} \times ((1 - p) + p / N) + c \times steps\_per\_worker$.

Để một worker đạt hiệu quả sản xuất, $c$ phải nhỏ so với thời gian giải mã mỗi bước. Trên các mô hình suy luận sinh ra hơn 5.000 token, các worker hoàn toàn có thể chi trả hàng trăm token cho chi phí điều phối mà vẫn đạt được mức tăng tốc tổng thể. Trên các tác vụ trò chuyện ngắn, chi phí điều phối sẽ chiếm ưu thế hoàn toàn và Hogwild! sẽ chạy chậm hơn cả tuần tự.

### Ví dụ cụ thể

Bài toán suy luận: chuỗi tư duy dài 10.000 token. Giả sử bài toán có tỷ lệ nội dung có thể song song hóa là $p = 0,7$ (các chiến lược chứng minh khác nhau, phân tích các trường hợp khác nhau) và $c = 200$ token chi phí điều phối trên mỗi worker. Với $N = 4$ worker:

- Thời gian tuần tự: 10.000 bước giải mã.
- Thời gian Hogwild!: $10000 \times (0,3 + 0,7 / 4) + 200 \times 4 = 10000 \times 0,475 + 800 = 5550$ bước giải mã.
- Tốc độ tăng tốc: $10000 / 5550 = 1,8$ lần.

Con số này ở mức khiêm tốn. Nhưng trên các bài toán suy luận dài hơn (50.000 token), chi phí điều phối được khấu hao dần và mức tăng tốc có thể chạm ngưỡng 2,5 đến 3 lần. Hogwild! chính là phiên bản tương đương trong suy luận của kỹ thuật song song hóa cấp luồng (thread-level parallelism) trong một ngôn ngữ lập trình cho phép viết mã đa luồng tự nhiên.

### Khi nào nên dùng Hogwild!

- Các bài toán suy luận dài (hàng nghìn token), nơi tác vụ có thể phân chia thành các mục tiêu con độc lập.
- Các mô hình suy luận đã được huấn luyện để tư duy từng bước. Các mô hình không chuyên suy luận không có khả năng tự phối hợp tốt.
- Triển khai trên một node đơn lẻ có đủ VRAM để chứa bộ nhớ đệm dùng chung cùng với $N$ tiến trình worker. Bộ đệm được dùng chung, nhưng mỗi worker vẫn có bộ nhớ kích hoạt riêng của nó.

### Khi nào không nên dùng

- Tác vụ đối thoại tương tác ngắn. Chi phí điều phối sẽ lấn át hoàn toàn lợi ích.
- Các tác vụ không thể song song hóa (chứng minh toán tuyến tính đơn nhất, biên dịch đơn luồng). $N=1$ là mức tối đa.
- Các mô hình không có năng lực suy luận. Sẽ không xuất hiện hành vi phối hợp tự phát.
- Triển khai trên nhiều node. Bộ nhớ đệm dùng chung đòi hỏi sự đồng bộ cực nhanh giữa các worker. Trong cùng một node thì tốt; xuyên node sẽ là thảm họa về độ trễ.

### Hiện trạng thử nghiệm

Tính đến tháng 4/2026, Hogwild! là một phương pháp nghiên cứu đi kèm bản cài đặt mã nguồn mở trên PyTorch. Chưa có ứng dụng nào trong môi trường sản xuất. Có ba rào cản lớn:

1. Quản lý bộ nhớ đệm KV dùng chung giữa các tiến trình chạy đồng thời là một bài toán kỹ thuật phức tạp.
2. Sự phối hợp tự hình thành phụ thuộc nhiều vào từng tác vụ cụ thể; các bộ đánh giá chuẩn (benchmarks) vẫn đang trong quá trình xây dựng.
3. Mức tăng tốc còn khiêm tốn so với những gì giải mã phỏng đoán đã mang lại, và dù cả hai có thể kết hợp với nhau, việc phối hợp kỹ thuật giữa chúng lại tạo thêm một tầng phức tạp mới.

Rất đáng để tìm hiểu. Rất đáng để thử nghiệm. Nhưng chưa phù hợp để đặt cược cho một sản phẩm thương mại.

```figure
continuous-batching
```

## Xây dựng hệ thống

Tệp `code/main.py` cài đặt một trình mô phỏng Hogwild! đơn giản hóa:

- Hai tiến trình worker, mỗi tiến trình là một "LLM" tất định sinh ra một trong các loại token (work-token - làm việc, observe-token - quan sát, coordinate-token - điều phối) với xác suất biết trước.
- Một bộ đệm dùng chung (danh sách các token) mà cả hai worker cùng đọc và ghi.
- Một logic điều phối đơn giản: khi một worker nhận thấy worker kia đã tạo đủ token công việc trong một phân loại, nó sẽ tự động chuyển sang làm việc ở phân loại khác.

Trình mô phỏng chạy trong một ngân sách bước cố định và báo cáo:

- Tổng số work-token tạo ra.
- Tổng thời gian thực (số bước của worker).
- Tốc độ tăng tốc hiệu dụng so với một worker đơn lẻ.
- Dấu vết ghi lại worker nào đã viết token nào.

### Bước 1: Bộ nhớ đệm dùng chung

Một danh sách mà cả hai worker cùng thêm phần tử vào. Sử dụng khóa đơn giản (Python `threading.Lock`) trong triển khai thực tế; ở đây chúng ta mô phỏng bằng một biến đếm.

### Bước 2: Vòng lặp worker

Mỗi worker, tại mỗi bước:

- Đọc trạng thái bộ nhớ đệm dùng chung hiện tại.
- Quyết định loại token cần viết dựa trên những gì đã tồn tại trong đệm.
- Ghi một token vào đệm.

### Bước 3: Heuristic điều phối

Nếu phân loại X đã có $K$ token trong bộ đệm và dự định ban đầu của worker là phân loại X, worker sẽ chuyển sang phân loại Y. Đây là mô hình đại diện đơn giản cho hành vi của mô hình suy luận: "nhận ra phần này đã có người làm, chuyển sang làm việc khác".

### Bước 4: Đo lường tốc độ tăng tốc

Chạy trình mô phỏng với $N=1$ worker và $N=2$ worker với cùng tổng ngân sách bước. Đếm số work-token sinh ra. Cấu hình $N=2$ sẽ tạo ra lượng work-token nhiều hơn khoảng 1,5 đến 1,8 lần nhờ sự phân chia công việc thông qua cơ chế điều phối.

### Bước 5: Thử nghiệm căng thẳng cho cơ chế điều phối

Giảm độ nhạy của heuristic điều phối. Chạy lại. Quan sát thấy rằng khi không có sự phối hợp tốt, $N=2$ sẽ liên tục tạo ra các token trùng lặp dư thừa và tốc độ tăng tốc sẽ tụt xuống dưới 1. Điều này khớp hoàn toàn với quan sát trong bài báo: mẹo này chỉ phát huy tác dụng nếu các worker có đủ năng lực suy luận để tự điều phối lẫn nhau.

## Sử dụng trong thực tế

Việc tích hợp Hogwild! vào môi trường sản xuất tính đến tháng 4/2026 vẫn ở cấp độ nghiên cứu. Bản cài đặt tham chiếu từ Yandex/HSE/IST được xây dựng trên PyTorch và nhắm tới các thiết lập đa tiến trình trên một node đơn lẻ chạy các mô hình DeepSeek-R1 và QwQ.

Lộ trình áp dụng thực tế:

1. Đánh giá hồ sơ khối lượng công việc tác vụ suy luận của bạn. Đo lường tỷ lệ các token mang tính khám phá (nhiều chiến lược, phân tích tình huống, tìm kiếm) so với suy luận tuyến tính.
2. Nếu yếu tố khám phá chiếm ưu thế, hãy chạy thử nghiệm Hogwild! với hai worker. Đo lường mức cải thiện thời gian thực.
3. Nếu mức cải thiện dưới 1,3 lần, bạn đang nằm trong vùng bị chi phí điều phối chi phối. Hãy quay lại dùng worker đơn lẻ.
4. Nếu mức cải thiện trên 1,5 lần, hãy nâng lên $N=4$ và đo lại. Điểm suy giảm hiệu suất biên (diminishing returns) thường rơi vào khoảng $N=4-8$.

Kết hợp với giải mã phỏng đoán: mỗi worker trong Hogwild! có thể độc lập sử dụng giải mã phỏng đoán. Hai mức tăng tốc này nhân lại với nhau (xấp xỉ), đưa mức tăng tốc 3x của giải mã phỏng đoán và 1,8x của Hogwild! lên mức hiệu dụng khoảng 5,4x so với giải mã đơn worker tuần tự thông thường.

## Đóng gói sản phẩm

Bài học này xây dựng `outputs/skill-parallel-inference-router.md`. Khi tiếp nhận hồ sơ khối lượng công việc suy luận (ngân sách token, hồ sơ song song hóa tác vụ, dòng mô hình, mục tiêu triển khai), kỹ năng này sẽ định tuyến giữa các chiến lược: bỏ phiếu, cây tư duy, đa tác tử, Hogwild!, và giải mã phỏng đoán.

## Bài tập

1. Chạy `code/main.py` với cấu hình mặc định. Xác nhận rằng cấu hình Hogwild! với $N=2$ tạo ra nhiều work-token hơn mốc cơ sở $N=1$ trong cùng một khoảng thời gian thực.

2. Giảm sức mạnh của heuristic điều phối (đặt `coordination_weight=0.1`). Chạy lại. Chỉ ra rằng mức tăng tốc bị sụp đổ. Giải thích nguyên nhân: các worker lặp lại nỗ lực của nhau khi chúng không thể phối hợp nhịp nhàng.

3. Tính toán mức tăng tốc Hogwild! kỳ vọng cho một tác vụ suy luận 50k token với $p=0,8, c=500$ và $N=4$ worker. Làm tương tự cho một tác vụ trò chuyện 1k token với $p=0,3, c=200$ và $N=4$. Tại sao một trường hợp mang lại thắng lợi còn trường hợp kia lại là tổn thất?

4. Đọc Mục 4 của bài báo Hogwild! (đánh giá sơ bộ). Xác định hai dạng thức thất bại (failure modes) mà các tác giả đã báo cáo. Mô tả cách một câu nhắc điều phối tốt hơn có thể giảm thiểu từng dạng thức đó.

5. Kết hợp Hogwild! với giải mã phỏng đoán trong mô hình mô phỏng: mỗi worker sử dụng giải mã phỏng đoán 2 token ở bên trong. Báo cáo mức tăng tốc dạng nhân. Vấn đề ghi chép sổ sách nào phát sinh khi cả hai worker đều muốn mở rộng cùng một tiền tố trong bộ đệm dùng chung?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Hogwild! | "Worker song song, đệm dùng chung" | $N$ thực thể của cùng một LLM chạy đồng thời với một bộ nhớ đệm KV dùng chung; sự phối hợp tự hình thành qua tự tạo gợi ý |
| Bộ đệm KV dùng chung (Shared KV cache) | "Phương tiện điều phối" | Một bộ đệm KV mở rộng liên tục duy nhất mà mọi worker cùng đọc và ghi; giúp các worker nhìn thấy token của nhau ngay lập tức |
| Phối hợp tự hình thành (Emergent coordination) | "Không cần huấn luyện" | Các LLM có năng lực suy luận có thể đọc bộ đệm dùng chung và tự phân chia công việc mà không cần tinh chỉnh hay giao thức tường minh |
| Chi phí điều phối phụ trội (c) | "Token dùng để định hướng" | Chi phí trên mỗi worker để đọc bộ đệm mở rộng và quyết định việc cần làm tiếp theo; phải giữ ở mức nhỏ so với tổng thời gian giải mã |
| Tỷ lệ song song hóa (p) | "Phần có thể chạy song song" | Tính song song ở cấp tác vụ: tỷ lệ phần trăm của tổng khối lượng công việc không mang tính tuần tự bản chất |
| RoPE giúp Hogwild! khả thi | "Vị trí xoay bất biến với phép dịch chuyển" | Vì vị trí là các phép xoay, nên việc ghi vào bộ đệm dùng chung không đòi hỏi phải tính toán lại các token trước đó |
| Tập hợp bỏ phiếu (Voting ensemble) | "Chạy N mô hình, chọn đa số" | Cấu trúc liên kết suy luận song song đơn giản nhất; hữu ích cho phân loại, kém hiệu quả cho chuỗi suy luận dài |
| Cây tư duy (Tree of thought) | "Rẽ nhánh và cắt tỉa" | Chiến lược suy luận khám phá nhiều nhánh và cắt tỉa; sử dụng logic điều phối tường minh |
| Framework đa tác tử (Multi-agent framework) | "Phân công nhiệm vụ con" | Mỗi tác tử nhận một vai trò; một bộ điều phối đứng ra dàn xếp; chi phí giao thức rất nặng |

## Đọc thêm

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — bài báo gốc về Hogwild!, đánh giá sơ bộ trên QwQ và DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — công trình Hogwild! gốc, nguồn gốc của tên gọi
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, đặc tính toán học giúp suy luận với bộ nhớ đệm dùng chung trở nên khả thi
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — chiến lược suy luận dạng cây mà Hogwild! nằm ở góc tiếp cận độc lập
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — giải mã phỏng đoán, kỹ thuật song song hóa trong chuỗi có thể kết hợp cùng Hogwild!
- [Bản cài đặt tham chiếu PyTorch của Hogwild!](https://github.com/eqimp/hogwild_llm) — nguồn mã chuẩn duy nhất cho các thí nghiệm trong bài báo
