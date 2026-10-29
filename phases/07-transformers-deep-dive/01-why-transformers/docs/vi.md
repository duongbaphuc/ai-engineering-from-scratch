# Vì sao chọn Transformer — Những hạn chế cố hữu của RNN

> RNN xử lý từng token một theo trình tự. Transformer xử lý tất cả token cùng một lúc. Lựa chọn kiến trúc mang tính bước ngoặt đó đã thay đổi toàn bộ đường cong mở rộng quy mô (scaling curve) trong học sâu sau năm 2017.

**Phân loại:** Học lý thuyết (Learn)  
**Ngôn ngữ:** Python  
**Điều kiện tiên quyết:** Giai đoạn 3 (Nền tảng Học sâu), Giai đoạn 5 · 09 (Mô hình Sequence-to-Sequence), Giai đoạn 5 · 10 (Cơ chế chú ý)  
**Thời lượng:** ~45 phút

## Vấn đề đặt ra

Trước năm 2017, mọi mô hình chuỗi hiện đại nhất (state-of-the-art) trên thế giới — từ ngôn ngữ, dịch máy đến giọng nói — đều là mạng nơ-ron hồi quy (recurrent neural network - RNN). LSTM và GRU đã thống trị các thước đo dịch máy tương đương chuẩn ImageNet trong suốt nửa thập kỷ. Chúng gần như là công cụ khả thi duy nhất mà cộng đồng nghiên cứu có trong tay.

Tuy nhiên, chúng tồn tại ba điểm yếu chí mạng. Việc tính toán tuần tự đồng nghĩa với việc bạn không thể song song hóa (parallelize) theo trục thời gian: token tại bước $t+1$ bắt buộc phải đợi trạng thái ẩn (hidden state) từ token tại bước $t$. Một chuỗi có độ dài 1.024 token đồng nghĩa với 1.024 bước thực thi tuần tự nối tiếp trên một GPU vốn có khả năng xử lý hàng triệu phép tính dấu phẩy động mỗi chu kỳ. Thời gian huấn luyện thực tế (wall-clock time) tăng tuyến tính theo chiều dài chuỗi trên phần cứng sinh ra để chạy tính toán song song.

Hiện tượng biến mất gradient (vanishing gradient) khiến thông tin cách đó 50 token đã bị nén qua 50 tầng phi tuyến tính liên tiếp. Các đơn vị hồi quy có cổng (gated recurrent units như LSTM, GRU) chỉ giảm nhẹ phần nào hiện tượng này chứ không bao giờ triệt tiêu hoàn toàn. Khả năng nắm bắt phụ thuộc tầm xa (long-range dependencies) — chẳng hạn như "quyển sách tôi đọc vào mùa hè năm ngoái trên chuyến bay tới Kyoto là..." — thường xuyên thất bại.

Các trạng thái ẩn có kích thước cố định buộc bộ mã hóa (encoder) phải ép toàn bộ chuỗi nguồn vào một vector duy nhất trước khi bộ giải mã (decoder) có thể nhìn thấy bất kỳ thông tin nào. Bất kể chuỗi nguồn dài 5 token hay 500 token, nút thắt cổ chai (bottleneck) vẫn giữ nguyên một kích thước không đổi.

Bài báo năm 2017 "Attention Is All You Need" đã đề xuất một giải pháp mang tính cách mạng: loại bỏ hoàn toàn cấu trúc hồi quy. Cho phép mỗi vị trí trực tiếp chú ý (attend) đến mọi vị trí khác một cách song song. Huấn luyện toàn bộ chuỗi trong một phép nhân ma trận lớn thay vì 1.024 phép tính tuần tự nối đuôi nhau.

Kết quả là cấu trúc này đã thống trị mọi phương thức dữ liệu (modality) tính đến năm 2026. Từ ngôn ngữ (GPT-5, Claude 4, Llama 4), thị giác máy tính (ViT, DINOv2, SAM 3), âm thanh (Whisper), sinh học (AlphaFold 3) cho tới robot học (RT-2). Vẫn cùng một khối kiến trúc nền tảng, chỉ khác nhau ở dữ liệu đầu vào.

## Khái niệm cốt lõi

![Tính toán tuần tự trong RNN so với cơ chế chú ý song song trong Transformer](../assets/rnn-vs-transformer.svg)

**Tính hồi quy như một nút thắt cổ chai.** Một mạng RNN tính toán $h_t = f(h_{t-1}, x_t)$. Mỗi bước phụ thuộc trực tiếp vào bước trước đó. Bạn không thể tính $h_5$ trước khi tính $h_4$. Trên các GPU hiện đại với hơn 10.000 lõi xử lý song song, điều này gây lãng phí tới 99% tài nguyên phần cứng trên các chuỗi dài.

**Cơ chế chú ý như một phép truyền phát (broadcast).** Cơ chế tự chú ý (self-attention) tính toán `output_i = sum_j(a_ij * v_j)` đồng thời cho mọi cặp $(i, j)$. Toàn bộ ma trận chú ý kích thước $N \times N$ được lấp đầy chỉ trong một phép nhân ma trận theo lô (batched matrix multiplication). Không bước nào phải chờ bước nào. GPU tối ưu hóa hoàn hảo cho mô hình tính toán này.

**Khả năng tăng tốc không chỉ là một hằng số cố định.** Đó là sự khác biệt giữa độ sâu tuần tự (serial depth) cấp $O(N)$ và độ sâu tuần tự cấp $O(1)$. Trong thực tế, mô hình Transformer huấn luyện nhanh hơn 5–10 lần trên mỗi epoch so với RNN trên cùng phần cứng với độ dài chuỗi $N=512$, và khoảng cách này càng mở rộng khi độ dài chuỗi tăng lên cho đến khi bạn chạm phải bức tường bộ nhớ bậc hai $O(N^2)$ của cơ chế chú ý (vấn đề mà FlashAttention đã giải quyết sau này — xem Bài 12).

**Cái giá phải trả của Transformer.** Bộ nhớ phục vụ cơ chế chú ý tăng theo cấp $O(N^2)$. Với ngữ cảnh 2K token, chi phí này hoàn toàn chấp nhận được. Nhưng với ngữ cảnh 128K token, bạn sẽ cần các kỹ thuật như cửa sổ trượt (sliding windows), ngoại suy RoPE, phân mảnh bộ nhớ của FlashAttention (tiling), hoặc các biến thể chú ý tuyến tính (linear attention). RNN có độ phức tạp $O(N)$ cho cả thời gian và bộ nhớ; trong khi đó, Transformer đánh đổi bộ nhớ để tiết kiệm thời gian, sau đó thu hồi lại ưu thế thời gian nhờ khả năng tính toán song song.

**Sự dịch chuyển của thiên kiến quy nạp (inductive bias).** RNN mặc định giả định tính cục bộ và tính kế cận theo thời gian. Transformer không áp đặt giả định nào — mọi cặp vị trí đều là ứng viên tiềm năng cho cơ chế chú ý. Đó là lý do vì sao Transformer cần nhiều dữ liệu hơn để huấn luyện hiệu quả, nhưng lại mở rộng năng lực tốt hơn nhiều khi có đủ dữ liệu. Định luật Chinchilla (2022) đã hình thức hóa điều này: khi được cung cấp đủ lượng token, một mô hình Transformer luôn đánh bại một mô hình RNN có cùng số lượng tham số.

````figure
rnn-vs-parallel
````

## Xây dựng thực tế

Bài học này không xây dựng mạng nơ-ron hoàn chỉnh — chúng ta sẽ mô phỏng nút thắt cổ chai cốt lõi bằng các phép tính số học để bạn trực tiếp cảm nhận khoảng cách hiệu năng ngay trên máy tính cá nhân.

### Bước 1: Đo độ sâu tuần tự

Xem file `code/main.py`. Chúng ta xây dựng hai hàm số. Một hàm mã hóa chuỗi dưới dạng một chuỗi các phép cộng nối tiếp (tuần tự, tương tự RNN). Một hàm mã hóa chuỗi dưới dạng rút gọn song song (parallel reduction, tương tự cơ chế chú ý). Cùng một công thức toán học, nhưng đồ thị phụ thuộc (dependency graph) hoàn toàn khác nhau.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # Không thể song song hóa: h phụ thuộc vào h trước đó
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # Mỗi phần tử x đều độc lập
```

Chúng ta đo thời gian thực thi của cả hai hàm trên các chuỗi có độ dài lên tới 100.000 phần tử. Phiên bản RNN có độ phức tạp $O(N)$ và chạy trên một luồng đơn CPU. Ngay cả trong môi trường Python thuần túy, phép rút gọn kiểu attention vẫn vượt trội ở độ dài $\ge 1.000$ do hàm `sum()` của Python được cài đặt bằng C và lặp không qua lớp chi phí thông dịch (interpreter overhead) ở mỗi bước.

### Bước 2: Đếm số lượng phép tính lý thuyết

Cả hai thuật toán đều thực hiện $N$ phép cộng. Điểm khác biệt nằm ở *độ sâu phụ thuộc* (dependency depth): có bao nhiêu phép toán bắt buộc phải diễn ra tuần tự trước khi bước tiếp theo có thể bắt đầu. Độ sâu của RNN bằng $N$. Độ sâu của Attention bằng $\log(N)$ khi dùng cây rút gọn (tree reduction), hoặc $1$ khi dùng quét song song (parallel scan). Độ sâu phụ thuộc, chứ không phải tổng số phép tính, mới là yếu tố quyết định thời gian thực thi trên GPU.

### Bước 3: Mở rộng thực nghiệm trên chuỗi dài

Chúng ta in ra một bảng so sánh thời gian giúp quan sát rõ ràng khoảng cách của $O(N)$. Trên một chiếc laptop đời 2026, các chuỗi dưới 1.000 phần tử xử lý quá nhanh để có thể đo đạc chính xác. Tuy nhiên, các chuỗi 100.000 phần tử thể hiện rõ ràng hành vi quét tuần tự tuyến tính. Khi ngoại suy điều đó lên một mô hình Transformer 16.384 token tương đương một mạng LSTM 12 tầng, bạn sẽ hiểu rõ vì sao thời gian huấn luyện từng là rào cản ngăn bước ngành học sâu vào năm 2016.

## Ứng dụng thực tế

Khi nào bạn vẫn nên chọn RNN vào năm 2026:

| Tình huống | Lựa chọn kiến trúc |
|------------|-------------------|
| Suy luận dạng luồng (streaming), xử lý từng token một, bộ nhớ cố định | RNN hoặc mô hình không gian trạng thái (Mamba, RWKV) |
| Chuỗi cực dài (>1 triệu token) nơi bộ nhớ attention bùng nổ | Chú ý tuyến tính (Linear attention), Mamba 2, Hyena |
| Thiết bị biên (edge device) không có bộ tăng tốc nhân ma trận | RNN tích chập theo chiều sâu (depthwise-separable) vẫn thắng về FLOPs/watt |
| Mọi trường hợp khác (huấn luyện, suy luận theo lô, ngữ cảnh đến 128K) | Transformer |

Các mô hình không gian trạng thái (state-space models - SSM) như Mamba về bản chất là các mạng RNN được tham số hóa có cấu trúc, mang lại ưu thế của cả hai thế giới: bộ nhớ quét $O(N)$, huấn luyện song song qua cơ chế quét chọn lọc (selective scan). Chúng đạt khoảng 90% chất lượng của Transformer với khả năng mở rộng ngữ cảnh dài tốt hơn. Tính đến năm 2026, phần lớn các phòng nghiên cứu hàng đầu đều huấn luyện các mô hình lai SSM + Transformer (ví dụ: Jamba, Samba) — tính hồi quy không hề biến mất, mà đã trở thành một khối thành phần tích hợp.

## Triển khai thành phẩm

Xem file `outputs/skill-architecture-picker.md`. Kỹ năng này giúp đề xuất kiến trúc phù hợp cho bài toán xử lý chuỗi dựa trên độ dài, thông lượng và giới hạn ngân sách huấn luyện. Nó sẽ luôn từ chối khuyến nghị một mô hình RNN thuần túy cho các lượt huấn luyện vượt quá 1 tỷ token nếu không nêu rõ sự đánh đổi về mặt hiệu năng.

## Bài tập

1. **Dễ.** Sử dụng hàm `rnn_style` từ `code/main.py` và thay thế trạng thái ẩn vô hướng bằng một vector trạng thái ẩn 64 chiều. Đo lại thời gian. Chi phí tuần tự tăng lên bao nhiêu phần trăm theo số chiều của trạng thái ẩn?
2. **Trung bình.** Cài đặt thuật toán tính tổng tiền tố song song (Hillis-Steele parallel prefix-sum scan) bằng Python thuần. Kiểm tra xem nó có cho ra kết quả số học giống hệt phép quét tuần tự trên chuỗi độ dài 1.024 hay không. Đếm độ sâu phụ thuộc.
3. **Khó.** Chuyển đổi phép rút gọn kiểu attention sang PyTorch trên GPU. Đo thời gian thực thi của cả hai phương pháp khi quét độ dài chuỗi từ 64 đến 65.536. Vẽ biểu đồ và giải thích hình dạng của đường cong thu được.

## Thuật ngữ then chốt

| Thuật ngữ | Cách nói thông thường | Ý nghĩa kỹ thuật chính xác |
|-----------|----------------------|---------------------------|
| Tính hồi quy (Recurrence) | "RNN là tuần tự" | Mô hình tính toán trong đó bước $t$ phụ thuộc vào bước $t-1$, bắt buộc thực thi tuần tự dọc theo trục thời gian. |
| Độ sâu tuần tự (Serial depth) | "Đồ thị sâu đến mức nào" | Chuỗi các phép tính phụ thuộc lẫn nhau dài nhất; giới hạn thời gian thực thi tối thiểu ngay cả trên phần cứng vô hạn. |
| Chú ý (Attention) | "Cho phép các token nhìn nhau" | Tổng có trọng số $\sum_j a_{ij} v_j$, trong đó $a_{ij}$ bắt nguồn từ điểm tương đồng giữa hai vị trí $i$ và $j$. |
| Cửa sổ ngữ cảnh (Context window) | "Mô hình nhìn được bao nhiêu" | Số lượng vị trí mà một tầng attention có thể nhận làm đầu vào; chi phí bộ nhớ bậc hai tăng tỉ lệ thuận tại đây. |
| Thiên kiến quy nạp (Inductive bias) | "Giả định tích hợp sẵn trong kiến trúc" | Tiền nghiệm về cấu trúc của dữ liệu; CNN giả định tính bất biến dịch chuyển, RNN giả định tính kế cận thời gian. |
| Mô hình không gian trạng thái (State-space model) | "RNN có toán học hỗ trợ" | Mô hình hồi quy được tham số hóa để huấn luyện song song thông qua các ma trận không gian trạng thái có cấu trúc. |
| Nút thắt cổ chai bậc hai (Quadratic bottleneck) | "Vì sao ngữ cảnh dài lại tốn kém" | Bộ nhớ attention tăng theo cấp $O(N^2)$ theo độ dài chuỗi; FlashAttention chỉ giảm hằng số chứ không thay đổi quy luật tăng trưởng. |

## Tài liệu đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — bài báo khai sinh Transformer và đặt dấu chấm hết cho sự thống trị của RNN trong xử lý ngôn ngữ tự nhiên.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — công trình đặt nền móng cho cơ chế chú ý, ban đầu được ghép nối vào kiến trúc RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — bài báo gốc giới thiệu LSTM.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — câu trả lời hiện đại của kiến trúc hồi quy đối với Transformer.
