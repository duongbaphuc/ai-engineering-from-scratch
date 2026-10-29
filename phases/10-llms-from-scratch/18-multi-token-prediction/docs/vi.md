# Dự đoán đa token (Multi-Token Prediction - MTP)

> Mọi mô hình LLM tự hồi quy từ GPT-2 đến Llama 3 đều được huấn luyện trên một hàm mất mát duy nhất cho mỗi vị trí: dự đoán token tiếp theo. DeepSeek-V3 đã bổ sung thêm hàm mất mát thứ hai cho mỗi vị trí: dự đoán token sau token đó. Lượng tham số bổ sung 14B (trên một mô hình 671B) được chưng cất ngược trở lại vào mô hình chính thông qua dòng chảy gradient, và các đầu MTP đã được huấn luyện này được tái sử dụng khi suy luận làm các mô hình phác thảo giải mã suy đoán với tỷ lệ chấp nhận trên 80%. Thông lượng sinh văn bản tăng $1,8\times$ mà không tốn thêm chi phí. Bài học này sẽ tự tay xây dựng mô-đun MTP tuần tự từ báo cáo kỹ thuật của DeepSeek, tính toán hàm mất mát cùng bố cục tham số đầu ra dùng chung, đồng thời giải thích lý do vì sao MTP tuần tự bảo toàn chuỗi nhân quả trong khi thiết kế MTP song song ban đầu của Gloeckle et al. đã phá vỡ nó.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 10 · 04 (Tiền huấn luyện một mini GPT), Giai đoạn 10 · 15 (Giải mã suy đoán)
**Time:** ~60 phút

## Mục tiêu học tập

- Phát biểu mục tiêu huấn luyện MTP và thiết lập công thức hàm mất mát kết hợp qua các độ sâu dự đoán.
- Giải thích sự khác biệt giữa các đầu MTP song song của Gloeckle et al. (2024) và các mô-đun MTP tuần tự của DeepSeek-V3, đồng thời làm rõ vì sao thiết kế tuần tự lại bảo toàn chuỗi nhân quả.
- Tính toán chi phí tham số và bộ nhớ phát sinh khi bổ sung các mô-đun MTP vào một lượt chạy tiền huấn luyện.
- Tự tay cài đặt một mô-đun MTP từ đầu: tầng nhúng dùng chung, khối transformer theo độ sâu, phép chiếu và đầu ra dùng chung.

## Vấn đề thực tế

Dự đoán token kế tiếp (next-token prediction) là mục tiêu huấn luyện tiêu chuẩn của các mô hình ngôn ngữ lớn. Mỗi trạng thái ẩn chỉ được giám sát để dự đoán chính xác một thứ duy nhất: token ngay sau nó. Đó là một tín hiệu giám sát yếu đến ngạc nhiên. Phần lớn thông tin trong một chuỗi ngôn ngữ kéo dài vượt ra ngoài một token đơn lẻ -- cấu trúc câu, tính mạch lạc ngữ nghĩa, tính chuẩn xác thực tế, và dòng chảy suy luận logic. Mô hình buộc phải học những điều đó bằng cách tích lũy nhiều tín hiệu một-token rời rạc qua hàng nghìn tỷ token.

Ý tưởng MTP đặt ra câu hỏi: điều gì sẽ xảy ra nếu mỗi trạng thái ẩn được giám sát để dự đoán nhiều token tương lai cùng một lúc? Gloeckle et al. (Meta, 2024) đã chứng minh rằng điều này mang lại hiệu quả rõ rệt. Cách triển khai của họ đặt nhiều đầu ra độc lập lên trên mô hình xương sống (backbone), mỗi đầu dự đoán một khoảng cách dịch chuyển (offset) khác nhau. Cách làm này song song và đơn giản, nhưng các đầu này đều nhìn vào cùng một trạng thái ẩn mà không có sự tinh chế phân cấp nào -- và các dự đoán không liên kết nhân quả với nhau, do đó chúng không thể được sử dụng làm mô hình phác thảo cho giải mã suy đoán.

DeepSeek-V3 (tháng 12 năm 2024) đã tái thiết kế MTP thành các mô-đun tuần tự (sequential modules) nhằm bảo toàn chuỗi nhân quả ở từng độ sâu dự đoán. Mô hình dự đoán $t+1$ từ $h_i^{(0)}$, sau đó dự đoán $t+2$ từ một trạng thái ẩn mới $h_i^{(1)}$ vốn kết hợp $h_i^{(0)}$ với vector nhúng $E(t+1)$, và tiếp tục như vậy. Mỗi độ sâu là một khối transformer nhỏ độc lập. Tầng nhúng dùng chung và đầu ra dùng chung giúp mức tăng thêm về tham số luôn ở mức khiêm tốn. Ở quy mô của DeepSeek-V3, chỉ có 14B tham số phụ trợ trên các mô-đun MTP so với 671B trọng số của mô hình chính. Mức chi phí phát sinh chỉ 2% đó vừa mang lại tín hiệu huấn luyện dày đặc hơn, VỪA cung cấp sẵn một mô hình phác thảo giải mã suy đoán hoàn hảo trong thời gian suy luận.

Bài học này xây dựng một mô-đun MTP đơn lẻ và hàm mất mát ở độ sâu $D$ từ đầu. Nền tảng toán học rất tinh gọn. Mã nguồn hoàn chỉnh chỉ trong 150 dòng.

## Khái niệm cốt lõi

### Công thức MTP tuần tự

DeepSeek-V3 bổ sung $D$ mô-đun MTP lên trên mô hình chính. Mô hình chính dự đoán $t_{i+1}$. Mỗi mô-đun $k$ (với $k = 1..D$) sẽ dự đoán token ở vị trí xa hơn $k$ bước, tức $t_{i+k+1}$, dựa trên tiền tố kéo dài tới vị trí $i$.

Mô-đun $k$ bao gồm:

- Một khối transformer $T_k$ với cơ chế chú ý và mạng MLP riêng của nó.
- Một ma trận chiếu $M_k$ kết hợp trạng thái ẩn của độ sâu trước đó với vector nhúng của token nhãn thực (ground-truth) ở độ sâu tiếp theo.
- Tầng nhúng dùng chung $E$ (chung với mô hình chính).
- Đầu ra tuyến tính dùng chung $\text{Out}$ (chung với mô hình chính).

Trong quá trình huấn luyện, với một tiền tố tới vị trí $i$, trạng thái ẩn theo từng độ sâu được tính như sau:

```math
\begin{aligned}
h_i^{(0)} &= \text{xương sống mô hình chính tại vị trí } i \\
h_i^{(k)} &= T_k\Big( M_k \cdot \mathop{\mathrm{concat}}\big(\mathop{\mathrm{RMSNorm}}(h_i^{(k-1)}), \mathop{\mathrm{RMSNorm}}(E(t_{i+k}))\big) \Big) \quad \text{với } k \ge 1
\end{aligned}
```

Dự đoán ở từng độ sâu là:

```math
\text{logits}_{i+k+1} = \mathop{\mathrm{Out}}(h_i^{(k)}) \quad \text{với } k = 1, \ldots, D
```

Hàm mất mát ở từng độ sâu là cross-entropy so với nhãn thực $t_{i+k+1}$:

```math
L_k = \mathop{\mathrm{CE}}(\text{logits}_{i+k+1}, t_{i+k+1})
```

Hàm mất mát kết hợp qua các độ sâu:

```math
L_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_k
```

Trong đó $\lambda$ là hệ số trọng số nhỏ -- DeepSeek-V3 dùng 0,3 trong 10% thời lượng huấn luyện đầu tiên và 0,1 cho phần còn lại. Tổng hàm mất mát huấn luyện là $L_{\text{main}} + L_{\text{MTP}}$.

### Vì sao phải tuần tự mà không phải song song?

Thiết kế MTP song song ban đầu của Gloeckle có $D$ đầu ra, mỗi đầu được áp trực tiếp vào $h_i^{(0)}$. Mỗi đầu dự đoán $t_{i+k}$ từ cùng một trạng thái ẩn của mô hình xương sống. Cách này huấn luyện tốt, nhưng các dự đoán không có sự ràng buộc điều kiện lẫn nhau. Bạn không thể lấy đầu ra của `head_1` để hỗ trợ cho `head_2` -- các đầu được kích hoạt song song độc lập.

Thiết kế tuần tự của DeepSeek-V3 xây dựng $h_i^{(k)}$ từ $h_i^{(k-1)}$ cộng với vector nhúng của token tiếp theo thực tế $E(t_{i+k})$. Điều này bảo toàn chuỗi nhân quả: để dự đoán $t_{i+k+1}$, mô-đun ở độ sâu $k$ nhìn thấy những gì xuất hiện tại $t_{i+k}$. Về mặt cấu trúc, điều này giống hệt cách một bộ giải mã tự hồi quy tiêu thụ chính đầu ra của nó, giúp các mô-đun MTP có thể dùng trực tiếp làm mô hình phác thảo giải mã suy đoán.

Khi suy luận: nạp $h_i^{(k-1)}$ và token phác thảo $t_{i+k}$ vào mô-đun $k$, thu được dự đoán cho $t_{i+k+1}$. Lặp lại quy trình. Đó chính xác là cơ chế phác thảo kiểu EAGLE, sử dụng mô-đun MTP đã được huấn luyện làm mạng phác thảo. DeepSeek-V3 báo cáo tỷ lệ chấp nhận trên 80% đối với mô-đun MTP đầu tiên và đạt mức tăng tốc khoảng $1,8\times$.

### Hạch toán tham số

Với một mô hình có kích thước ẩn $h$ và bộ từ vựng $V$:

- Mô hình chính: hàng tỷ tham số, cộng một đầu ra kích thước $V \times h$.
- Đầu ra dùng chung: tái sử dụng đầu ra của mô hình chính. Không tốn thêm tham số.
- Tầng nhúng dùng chung: tái sử dụng bảng nhúng của mô hình chính. Không tốn thêm tham số.
- Cho mỗi mô-đun MTP:
  - Phép chiếu $M_k$: $(2h) \times h = 2h^2$.
  - Khối transformer $T_k$: chú ý ($4h^2$ cho MHA) cộng với MLP (khoảng $8h^2$ cho SwiGLU với tỷ lệ 8/3). Khoảng $12h^2$ cho mỗi khối.

Tổng cộng thêm cho mỗi mô-đun: $\sim 14h^2$. Với $h = 7168$ của DeepSeek-V3, với $D = 1$ mô-đun: trên lý thuyết là $\sim 14 \times 7168^2 \approx 720\text{M}$ tham số. DeepSeek-V3 báo cáo 14B -- sự chênh lệch này chủ yếu do các tầng chuyên gia trong khối MTP cũng sử dụng kiến trúc MoE.

### Lợi ích đối với giải mã suy đoán

Trong quá trình tiền huấn luyện, các mô-đun MTP làm chậm tốc độ huấn luyện khoảng 10% (do tăng tính toán truyền xuôi và thêm hàm mất mát). Nhưng lợi ích thu lại là gấp đôi:

1. Tín hiệu huấn luyện dày đặc hơn. Mỗi trạng thái ẩn nhận $D+1$ mục tiêu giám sát. Hiệu quả đo đạc trên MMLU, GSM8K, MATH, HumanEval: cải thiện ổn định vài điểm phần trăm trong các thử nghiệm cắt bỏ của DeepSeek-V3.

2. Mô hình phác thảo giải mã suy đoán miễn phí khi suy luận. Mô-đun MTP vốn đã được huấn luyện để dự đoán các token tiếp theo. Khi tái sử dụng làm mạng phác thảo, nó mang lại tỷ lệ chấp nhận trên 80%. Ở mức đó, giải mã suy đoán với $N=3$ hoặc $N=5$ mang lại thông lượng gấp $1,8\times$. Chi phí 10% lúc huấn luyện được hoàn vốn ngay từ lần đầu tiên bạn chạy suy luận.

### Mối liên hệ với EAGLE

EAGLE huấn luyện một mô hình phác thảo nhỏ RIÊNG BIỆT sau giai đoạn tiền huấn luyện. MTP tích hợp mô hình phác thảo vào ngay trong quá trình tiền huấn luyện. Cả hai phương pháp hội tụ về tỷ lệ chấp nhận tương đương nhau nhưng thông qua các đường ống khác nhau:

| Tiêu chí | EAGLE-3 | MTP (DeepSeek-V3) |
|---|---|---|
| Thời điểm huấn luyện | Sau tiền huấn luyện | Trong tiền huấn luyện |
| Tương thích ngược với trọng số cũ | Có | Không (cần huấn luyện lại) |
| Tham số phác thảo | 1-2 lớp transformer | 1 khối transformer + phép chiếu |
| Tỷ lệ chấp nhận | 0,88-0,92 | 0,80+ ở độ sâu 1 |
| Lợi ích ngoài tăng tốc | Chỉ giải mã suy đoán | Tín hiệu huấn luyện đậm đặc hơn + tăng tốc |

````figure
multi-token-predict
````

## Xây dựng thực tế (Build It)

Tệp `code/main.py` xây dựng một mô-đun MTP đơn lẻ từ đầu đến cuối: tầng nhúng dùng chung, phép chiếu, khối transformer, đầu ra dùng chung. Sau đó, nó tính toán hàm mất mát cross-entropy theo từng độ sâu trên một chuỗi nhân tạo ngắn và in ra số lượng tham số theo từng thành phần. Một bộ từ vựng thử nghiệm gồm 32 token giúp các con số hiển thị trực quan, dễ theo dõi.

### Bước 1: Bảng nhúng dùng chung

Một bảng `vocab_size x hidden` duy nhất được sử dụng bởi cả mô hình chính VÀ mọi mô-đun MTP ở mọi độ sâu. Không phải là bản sao thứ hai -- mà theo nghĩa đen là cùng một tensor trọng số.

### Bước 2: Phép kết hợp theo từng độ sâu

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

Mô hình DeepSeek-V3 thực tế nối hai vector đã qua RMSNorm thành vector kích thước `[2h]` và chiếu bằng ma trận kích thước $h \times 2h$. Bản thử nghiệm sử dụng phép cộng vector để giữ cho mã nguồn thư viện chuẩn ngắn gọn.

### Bước 3: Khối Transformer ở độ sâu k

Gồm tự chú ý và MLP. Trong bản thử nghiệm, một khối chú ý tuyến tính một lớp và một mạng MLP SwiGLU giúp thể hiện rõ cấu trúc mà không cần phụ thuộc vào thư viện numpy.

### Bước 4: Đầu ra dùng chung

Tái sử dụng phép chiếu đầu ra của mô hình chính để sinh logits trên toàn bộ từ vựng.

### Bước 5: Hàm mất mát theo từng độ sâu

Tính hàm mất mát cross-entropy của softmax(logits) so với token nhãn thực ở vị trí lệch $k+1$. Tổng hợp qua các độ sâu với hệ số co giãn $\lambda / D$.

### Bước 6: Hạch toán tham số

In ra tổng số tham số, số lượng tham số dùng chung (tầng nhúng, đầu ra), và số tham số bổ sung của từng mô-đun. Chỉ ra tỷ lệ giữa phần MTP bổ sung so với kích thước mô hình chính.

## Sử dụng thực tế (Use It)

MTP được tích hợp sẵn trong DeepSeek-V3 (tháng 12 năm 2024) và dòng mô hình DeepSeek-R1. Trong thời gian suy luận:

- Ngăn xếp phục vụ của chính DeepSeek sử dụng các mô-đun MTP làm bộ giải mã suy đoán ngay khi xuất xưởng.
- vLLM và SGLang đã có đường dẫn tích hợp cho DeepSeek-V3 MTP tính đến tháng 4 năm 2026.
- Hướng dẫn SGLang ROCm của AMD minh họa cấu hình giải mã suy đoán MTP cụ thể với mức tăng tốc đo được $1,8\times$ trên điểm kiểm tra V3.

Khi nào nên dùng MTP trong một lượt tiền huấn luyện mới:

- Bạn làm chủ toàn bộ đường ống tiền huấn luyện và muốn tận dụng tín hiệu giám sát dày đặc hơn.
- Bạn biết chắc mình sẽ phục vụ mô hình ở quy mô lớn và muốn có giải mã suy đoán miễn phí.
- Kích thước ẩn của bạn tối thiểu là 4096. Ở quy mô 1B, chi phí phụ trội sẽ lấn át lợi ích đạt được.

Khi nào không nên sử dụng:

- Tinh chỉnh một mô hình dày đặc đã tiền huấn luyện sẵn. Mô-đun MTP chưa hề được huấn luyện.
- Các mô hình nghiên cứu nơi bạn cần một mốc đối chứng sạch sẽ để so sánh. MTP làm thay đổi kiến trúc cơ bản.

## Xuất bản tạo tác (Ship It)

Bài học này tạo ra `outputs/skill-mtp-planner.md`. Dựa trên bản đặc tả tiền huấn luyện (kích thước mô hình, dữ liệu, năng lực tính toán), nó sẽ cung cấp một kế hoạch tích hợp MTP: số lượng độ sâu $D$, lịch trình tham số $\lambda$, mức tăng bộ nhớ, và cấu hình liên kết giải mã suy đoán lúc suy luận.

## Bài tập thực hành

1. Chạy `code/main.py`. Chứng minh rằng hàm mất mát theo từng độ sâu giảm đơn điệu khi tín hiệu nhân tạo được tăng cường. Sửa đổi dữ liệu nhân tạo để sử dụng một quy luật cố định và xác nhận hàm mất mát ở cả độ sâu 1 và độ sâu 2 đều hội tụ.

2. Tính toán chi phí tham số bổ sung cho một mô hình 70B dày đặc (chiều ẩn 8192, 80 lớp) với $D=1$ mô-đun MTP. So sánh với con số 14B được báo cáo của DeepSeek-V3. Giải thích vì sao con số của DeepSeek lại cao hơn: khối transformer MTP kế thừa cấu trúc MoE tương tự, làm phình to số tham số trên mỗi mô-đun.

3. Triển khai $D=2$ trong bản thử nghiệm: thêm mô-đun MTP thứ hai nhận $h^{(1)}$ và dự đoán $t_{i+3}$. Xác nhận rằng hàm mất mát kết hợp và việc hạch toán tham số khớp với các phương trình 19-21 trong bài báo của DeepSeek.

4. Chuyển bản thử nghiệm sang MTP song song (kiểu Gloeckle): thêm $D$ đầu ra lên trên trạng thái ẩn chính, mỗi đầu dự đoán một khoảng lệch khác nhau. Đo lường sự khác biệt về hàm mất mát trên từng độ sâu so với phiên bản tuần tự trên cùng một tín hiệu nhân tạo. Phiên bản tuần tự sẽ cho hàm mất mát ở độ sâu $k$ thấp hơn với $k > 1$ vì nó có điều kiện dựa trên các dự đoán trung gian.

5. Sử dụng mô-đun MTP đã huấn luyện làm mô hình phác thảo kiểu EAGLE: gọi mô-đun $k$ để đề xuất $t_{i+k+1}$ khi suy luận. Đo lường tỷ lệ chấp nhận của các token phác thảo này so với dự đoán của mô hình chính trên một chuỗi kiểm tra độc lập. Nếu bạn đạt trên 50% trên bản thử nghiệm, bạn đã tái hiện thành công đặc tính thực nghiệm của việc dùng MTP làm mô hình phác thảo.

## Thuật ngữ then chốt

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa thực tế |
|---|---|---|
| Mô-đun MTP | "Khối hàm mất mát bổ sung" | Một khối transformer nhỏ kèm theo phép chiếu có nhiệm vụ dự đoán token đi trước mô hình chính $k$ vị trí |
| Độ sâu dự đoán (Prediction depth) | "Khoảng lệch nào" | Số nguyên $k$ sao cho mô-đun $k$ dự đoán $t_{i+k+1}$ từ tiền tố tới vị trí $i$ (mô hình chính có độ sâu 0 và dự đoán $t_{i+1}$) |
| MTP song song (Parallel MTP) | "Kiểu Gloeckle" | $D$ đầu ra độc lập đặt trên cùng một trạng thái ẩn của mô hình xương sống, không có liên kết điều kiện |
| MTP tuần tự (Sequential MTP) | "Kiểu DeepSeek-V3" | Mỗi mô-đun nhận điều kiện từ trạng thái ẩn của độ sâu trước cộng với vector nhúng của token kế tiếp; bảo toàn chuỗi nhân quả |
| Đầu ra dùng chung (Shared output head) | "Tái sử dụng đầu LM chính" | Các mô-đun MTP gọi trực tiếp đầu LM head của mô hình chính chứ không dùng phép chiếu đầu ra riêng |
| Tầng nhúng dùng chung (Shared embedding) | "Tái sử dụng bảng từ vựng chính" | Cùng một bảng nhúng từ vựng được sử dụng ở mọi nơi; không nhân bản tham số |
| Ma trận chiếu $M_k$ (Projection matrix) | "Gộp trạng thái ẩn + token kế tiếp" | Một tầng tuyến tính kích thước $h \times 2h$ gộp trạng thái ẩn trước đó và vector nhúng token đích vào đầu vào của độ sâu tiếp theo |
| Hàm mất mát kết hợp $L_{\text{MTP}}$ (Joint loss) | "Trung bình các hàm mất mát phụ" | Trung bình cộng của các hàm mất mát cross-entropy theo từng độ sâu, được co giãn bởi hệ số $\lambda$ |
| Tỷ lệ chấp nhận ở độ sâu 1 | "Phác thảo MTP đúng bao nhiêu" | Tỷ lệ mà dự đoán top-1 của mô-đun MTP $D=1$ trùng khớp với dự đoán top-1 của mô hình chính; đạt trên 80% trên DeepSeek-V3 |
| Trọng số Lambda | "Mức độ quan trọng của hàm mất mát phụ" | Hệ số co giãn theo từng độ sâu; 0,3 khi bắt đầu huấn luyện, 0,1 ở giai đoạn sau trên DeepSeek-V3 |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — mô tả đầy đủ về MTP tuần tự (Mục 2.2), bao gồm các phương trình hàm mất mát kết hợp và mức tăng tốc 1,8 lần khi suy luận
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — mốc cơ sở MTP song song mà thiết kế của DeepSeek cải tiến
- [Thẻ mô hình DeepSeek-V3 trên Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — tổng 685B tham số (671B chính + 14B MTP), ghi chú triển khai
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — khung làm việc giải mã suy đoán mà MTP tích hợp vào
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — kiến trúc phác thảo năm 2025 của EAGLE, đối trọng cạnh tranh với MTP
