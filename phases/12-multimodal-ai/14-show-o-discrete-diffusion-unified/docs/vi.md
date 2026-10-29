# Show-o và Các Mô hình Hợp nhất Khuếch tán Rời rạc

> Transfusion kết hợp cả biểu diễn liên tục và rời rạc. Show-o (Xie et al., tháng 8 năm 2024) lại đi theo hướng ngược lại: các token văn bản sử dụng hàm dự đoán token tiếp theo có tính nhân quả (causal next-token prediction), trong khi các token hình ảnh sử dụng cơ chế khuếch tán rời rạc có che mờ (masked discrete diffusion) theo phong cách của MaskGIT. Cả hai cùng nằm bên trong một transformer duy nhất với một ma trận mặt nạ chú ý lai (hybrid attention mask). Kết quả mang lại một mô hình hợp nhất các tác vụ VQA (trả lời câu hỏi thị giác), text-to-image (T2I - chuyển văn bản thành ảnh), inpainting (vẽ bù/phục hồi vùng ảnh) và sinh nội dung đa phương thức kết hợp trên cùng một mạng xương sống (backbone), một bộ token hóa (tokenizer) cho mỗi phương thức, và một dạng thức hàm mất mát duy nhất (dự đoán token tiếp theo được mở rộng thành dự đoán token bị che). Bài học này sẽ phân tích chi tiết thiết kế của Show-o — lý do vì sao khuếch tán rời rạc có che mờ lại là một bộ sinh hình ảnh song song, ít bước — và so sánh với Transfusion và Emu3.

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ lấy mẫu khuếch tán rời rạc có che mờ)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 13 (Transfusion)  
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích cơ chế khuếch tán rời rạc có che mờ: lịch trình che các token đồng đều rồi yêu cầu transformer khôi phục lại chúng.
- So sánh quá trình giải mã hình ảnh song song (Show-o, MaskGIT) với giải mã hình ảnh tự hồi quy (Chameleon, Emu3) về tốc độ và chất lượng.
- Liệt kê bốn tác vụ mà Show-o xử lý trong một checkpoint duy nhất: sinh văn bản, VQA, T2I, inpainting hình ảnh.
- Lựa chọn một lịch trình che (cosine, tuyến tính, cắt cụt) và lập luận về ảnh hưởng của nó đến chất lượng mẫu sinh.

## Vấn đề

Quy trình huấn luyện hai hàm mất mát của Transfusion hoạt động hiệu quả nhưng lại có động lực học phức tạp hơn — hàm mất mát khuếch tán liên tục nằm trên một thang đo số học khác hẳn so với hàm mất mát NTP rời rạc. Cân bằng trọng số hàm mất mát đòi hỏi quá trình tìm kiếm siêu tham số (hyperparameter search) tốn kém. Kiến trúc này mang lại hiệu quả nhưng tương đối phức tạp.

Câu trả lời của Show-o: giữ cho cả hai phương thức ở dạng rời rạc (tương tự như Chameleon), nhưng sinh hình ảnh song song thông qua khuếch tán rời rạc có che mờ thay vì sinh tuần tự. Mục tiêu huấn luyện trở thành một bài toán dự đoán token bị che duy nhất có khả năng tổng quát hóa cơ chế dự đoán token tiếp theo một cách tự nhiên.

## Khái niệm

### Khuếch tán rời rạc có che mờ (Masked discrete diffusion - MaskGIT)

Kỹ thuật MaskGIT nguyên bản của Chang et al. (2022) vô cùng thanh thoát. Bắt đầu từ một hình ảnh bị che hoàn toàn (mọi token đều mang ID đặc biệt `<MASK>`). Ở mỗi bước, mô hình dự đoán đồng thời tất cả các token bị che song song, sau đó giữ lại top-K dự đoán có độ tin cậy cao nhất và che lại phần còn lại. Sau khoảng 8-16 vòng lặp, toàn bộ các token đều được điền đầy. Lịch trình quy định số lượng token được mở che (unmask) sau mỗi bước được tinh chỉnh kỹ lưỡng — trong đó lịch trình hàm cosine (cosine schedule) mang lại hiệu quả rất tốt.

Quá trình huấn luyện rất đơn giản: lấy mẫu tỷ lệ che (masking ratio) phân phối đều từ [0, 1], áp dụng lên các token VQ của hình ảnh, và huấn luyện transformer khôi phục lại các token bị che đó. Đây chính xác là những gì BERT đã làm cho văn bản, nhưng được mở rộng quy mô cho bài toán sinh hình ảnh.

### Show-o: một transformer, mặt nạ chú ý lai (Show-o: one transformer, hybrid mask)

Show-o đưa MaskGIT vào bên trong một transformer mô hình ngôn ngữ nhân quả (causal language model). Mặt nạ chú ý bao gồm:

- Token văn bản: nhân quả (tương tự LLM tiêu chuẩn).
- Token hình ảnh: hai chiều hoàn toàn (full bidirectional) bên trong khối hình ảnh (nhờ đó các token bị che có thể nhìn thấy mọi token hình ảnh khác trong quá trình dự đoán).
- Từ văn bản sang hình ảnh (T2I): văn bản chú ý đến các hình ảnh đứng trước, hình ảnh chú ý đến văn bản đứng trước.

Quá trình huấn luyện luân phiên giữa:
1. Huấn luyện NTP tiêu chuẩn trên các chuỗi văn bản.
2. Mẫu dữ liệu T2I: văn bản → hình ảnh với các token hình ảnh bị che mờ, sử dụng hàm mất mát dự đoán token bị che.
3. Mẫu dữ liệu VQA: hình ảnh → văn bản với các token văn bản bị che (thực chất chính là NTP).

Hàm mất mát hợp nhất là entropy chéo trên các token `<MASK>`, bao quát cả NTP văn bản (khi chỉ có token cuối cùng bị "che") lẫn khuếch tán che mờ hình ảnh (khi một tập hợp con ngẫu nhiên bị che).

### Lấy mẫu song song (Parallel sampling)

Show-o sinh một hình ảnh trong khoảng ~16 bước thay vì ~1000 bước (tự hồi quy từng token) hoặc ~20 bước (khuếch tán liên tục). Ở mỗi bước, mô hình dự đoán toàn bộ các token bị che song song; cam kết cố định top-K token có độ tin cậy cao nhất; và lặp lại quy trình.

So sánh:
- Chameleon / Emu3 (tự hồi quy trên từng token): $N_{\text{tokens}}$ lượt truyền xuôi, thường từ 1024-4096 lượt cho mỗi ảnh.
- Transfusion (khuếch tán liên tục): ~20 bước, mỗi bước là một lượt truyền xuôi đầy đủ qua transformer.
- Show-o (khuếch tán rời rạc có che mờ): ~16 bước, mỗi bước là một lượt truyền xuôi đầy đủ qua transformer.

Show-o chạy nhanh hơn Chameleon trên các mô hình có quy mô tương đương, và gần như tương đương số bước của Transfusion với chi phí cho mỗi bước thấp hơn (logits trên từ vựng rời rạc so với hàm mất mát MSE liên tục).

### Bốn tác vụ trong một checkpoint (Tasks in one checkpoint)

Show-o hỗ trợ bốn tác vụ khi suy luận, được lựa chọn thông qua định dạng prompt:

- Sinh văn bản (Text generation): đầu ra văn bản tự hồi quy tiêu chuẩn.
- VQA: đầu vào là hình ảnh, đầu ra là văn bản.
- T2I: đầu vào là văn bản, đầu ra là hình ảnh thông qua khuếch tán rời rạc có che mờ.
- Inpainting (vẽ bù/phục hồi vùng ảnh): đầu vào là hình ảnh bị che một số token, mô hình điền đầy phần còn thiếu.

Khả năng inpainting xuất hiện một cách hoàn toàn tự nhiên nhờ mục tiêu huấn luyện dự đoán che mờ. Chỉ cần che một vùng trên lưới token VQ, cung cấp phần còn lại cùng với câu lệnh văn bản, và yêu cầu mô hình dự đoán các token bị che.

### Lịch trình che mờ (Masking schedule)

Lịch trình quy định số lượng token được mở che qua từng bước quyết định trực tiếp chất lượng ảnh sinh ra. Show-o khuyến nghị sử dụng lịch trình cosine:

```math
\text{mask\_ratio}(t) = \cos\left(\frac{\pi t}{2T}\right) \qquad t = 0, \ldots, T
```

Ở bước 0, toàn bộ token bị che (tỷ lệ 1.0). Ở bước $T$, không còn token nào bị che. Hàm cosine tập trung mật độ vào các tỷ lệ ở khoảng giữa, nơi quá trình dự đoán mang lại nhiều thông tin nhất. Lịch trình tuyến tính (linear schedule) cũng hoạt động được nhưng đạt ngưỡng bão hòa chất lượng nhanh hơn.

### Show-o2

Show-o2 (công trình tiếp nối năm 2025, arXiv 2506.15564) mở rộng quy mô của Show-o: sử dụng LLM nền tảng lớn hơn, bộ token hóa tốt hơn, và cải tiến lịch trình mặt nạ. Kiến trúc vẫn giữ nguyên cấu trúc cốt lõi.

### Vị thế của Show-o trong toàn cảnh kiến trúc (Where Show-o sits)

Trong phân loại năm 2026:

- Token rời rạc + NTP: Chameleon, Emu3. Đơn giản nhưng tốc độ suy luận chậm.
- Token rời rạc + khuếch tán có che mờ: Show-o, MaskGIT, LlamaGen, Muse. Lấy mẫu song song, chất lượng vẫn chịu giới hạn từ bộ token hóa.
- Biểu diễn liên tục + khuếch tán: Transfusion, MMDiT, DiT. Chất lượng cao nhất, huấn luyện phức tạp hơn.
- Biểu diễn liên tục + flow matching trong một VLM: JanusFlow, InternVL-U. Dòng mô hình mới nhất.

Lựa chọn theo bài toán: Hãy chọn Show-o khi bạn cần tích hợp T2I + inpainting + VQA trong một mô hình nguồn mở duy nhất với tốc độ chấp nhận được; chọn Transfusion khi chất lượng hình ảnh là ưu tiên tối thượng và bạn có đủ nguồn lực để kiểm soát hệ thống hai hàm mất mát.

````figure
masked-diffusion-unmask
````

## Thực hành (Use It)

`code/main.py` mô phỏng quá trình lấy mẫu của Show-o:

- Một lưới đồ chơi gồm 16 token VQ.
- Một "transformer" giả lập dự đoán logits dựa trên câu lệnh prompt và các token hiện đang được mở che.
- Quá trình lấy mẫu che mờ song song qua 8 bước với lịch trình cosine.
- In ra các trạng thái trung gian (sự phát triển của cấu trúc mặt nạ) và chuỗi token cuối cùng.

Hãy chạy mã và quan sát cách mặt nạ tan biến dần theo từng bước.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-unified-gen-model-picker.md`. Với một bài toán sản phẩm yêu cầu cả tác vụ hiểu (VQA, chú thích ảnh) lẫn tác vụ sinh (T2I, inpainting) dưới ràng buộc mô hình trọng số mở, kỹ năng này sẽ lựa chọn giữa họ Show-o, họ Transfusion/MMDiT, và họ Emu3 / Chameleon kèm theo các phân tích đánh đổi cụ thể.

## Bài tập (Exercises)

1. Cơ chế khuếch tán rời rạc có che mờ lấy mẫu trong khoảng ~16 bước. Vì sao không phải là 1 bước duy nhất? Điều gì sẽ xảy ra nếu bạn mở che tất cả mọi token ngay tại bước 0?

2. Khả năng inpainting được cung cấp miễn phí nhờ cơ chế khuếch tán có che mờ. Hãy đề xuất một trường hợp ứng dụng thực tế (thực tế hoặc giả định) mà tính năng inpainting của Show-o vượt trội hơn một mô hình chuyên dụng.

3. Lịch trình cosine so với lịch trình tuyến tính: tính toán số lượng token được mở che qua từng bước với $T=8$. Lịch trình nào mang lại sự cân bằng tốt hơn?

4. Một hình ảnh Show-o kích thước 512x512 bao gồm 1024 token. Với từ vựng $K=16384$, mô hình xuất ra $1024 \cdot \log_2(16384) = 14{,}336$ bit (~1.75 KiB) dữ liệu. Stable Diffusion xuất ra 512*512*24 bit = 6.291.456 bit (~768 KiB) pixel thô. Tỷ lệ nén ở đây là bao nhiêu và nó mang lại sự đánh đổi chất lượng ra sao?

5. Đọc bài báo LlamaGen (arXiv:2406.06525). Mô hình hình ảnh tự hồi quy có điều kiện theo lớp (class-conditional) của LlamaGen khác biệt như thế nào so với phương pháp che mờ của Show-o?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Masked discrete diffusion (Khuếch tán rời rạc có che mờ) | "Kiểu MaskGIT" | Huấn luyện để dự đoán các token bị che; khi suy luận, mở che lặp lại các dự đoán có độ tin cậy cao nhất |
| Cosine schedule (Lịch trình cosine) | "Lịch trình mở che" | Sự suy giảm của tỷ lệ che qua các bước suy luận; tập trung mức tăng độ tin cậy ở khoảng giữa |
| Parallel decoding (Giải mã song song) | "Dự đoán mọi token cùng lúc" | Mỗi bước dự đoán toàn bộ chuỗi token bị che trong một lượt truyền xuôi, sau đó cố định top-K |
| Hybrid attention (Chú ý lai) | "Nhân quả + hai chiều" | Mặt nạ có tính nhân quả trên các token văn bản và hai chiều bên trong các khối hình ảnh |
| Inpainting (Vẽ bù/phục hồi vùng ảnh) | "Sinh ảnh điền khuyết" | Điều kiện hóa trên một ảnh có một số token bị che, dự đoán các token còn thiếu; thừa hưởng tự nhiên từ mục tiêu huấn luyện |
| Commitment rate (Tỷ lệ cam kết) | "Top-K mỗi bước" | Số lượng token được xác nhận là "hoàn tất" sau mỗi vòng lặp; kiểm soát sự đánh đổi giữa thời gian suy luận và chất lượng |

## Tài liệu tham khảo (Further Reading)

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
