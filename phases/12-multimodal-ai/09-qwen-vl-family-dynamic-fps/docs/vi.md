# Họ mô hình Qwen-VL và Video với FPS động

> Họ mô hình Qwen-VL — gồm Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025) và Qwen3-VL (2025) — là dòng mô hình ngôn ngữ - thị giác (Vision-Language Model - VLM) trọng số mở có tầm ảnh hưởng lớn nhất vào năm 2026. Mỗi thế hệ đều đưa ra một quyết định kiến trúc mang tính bước ngoặt mà phần còn lại của hệ sinh thái mã nguồn mở sao chép lại chỉ trong vòng mười hai tháng: độ phân giải động tự nhiên thông qua M-RoPE, kỹ thuật lấy mẫu tốc độ khung hình động (dynamic-FPS) với sự căn chỉnh thời gian tuyệt đối (absolute time alignment), cơ chế chú ý theo cửa sổ (window attention) trong ViT, và các định dạng đầu ra có cấu trúc phục vụ agent. Đến thời Qwen3-VL, công thức này đã đạt trạng thái ổn định: bộ mã hóa 2D-RoPE-ViT hỗ trợ tỷ lệ khung hình tự nhiên, bộ chiếu MLP ánh xạ vào mô hình ngôn ngữ nền tảng Qwen3 quy mô lớn, cùng các giai đoạn huấn luyện coi khả năng nhận dạng ký tự quang học (OCR), định vị tọa độ (grounding) và hành vi của agent là các mục tiêu hạng nhất. Bài học này điểm lại quá trình tiến hóa của họ mô hình theo trình tự thời gian để bạn hiểu thấu đáo lý do đằng sau từng tham số thiết kế.

**Type:** Học (Learn)
**Languages:** Python (thư viện chuẩn, bộ mã hóa M-RoPE + bộ lấy mẫu FPS động)
**Prerequisites:** Giai đoạn 12 · 06 (chia mảnh và đóng gói - patch-n'-pack)
**Time:** ~120 phút

## Mục tiêu học tập

- Tính toán các góc quay ba trục của M-RoPE (thời gian, chiều cao, chiều rộng) và giải thích tại sao cần thiết phải có cả ba trục này.
- Lựa chọn chiến lược lấy mẫu FPS động cho một video và phân tích sự đánh đổi giữa số lượng token trên giây với độ chính xác khi phát hiện sự kiện.
- Kể tên bốn thế hệ nâng cấp của dòng Qwen-VL theo đúng thứ tự và những đột phá mà mỗi thế hệ mang lại.
- Thiết lập định dạng đầu ra JSON cho agent theo phong cách Qwen2.5-VL và bóc tách các lệnh gọi công cụ (tool call) có cấu trúc từ phản hồi của VLM.

## Vấn đề cần giải quyết

Qwen-VL ra mắt vào tháng 8 năm 2023 như một lời đáp trả trực diện đối với LLaVA-1.5 và BLIP-2. Khoảng cách kỹ thuật mà đội ngũ Qwen nhắm tới giải quyết gồm ba khía cạnh: độ phân giải, video và đầu ra có cấu trúc.

- **Độ phân giải (Resolution):** LLaVA-1.5 chỉ hoạt động ở độ phân giải 336x336. Kích thước này đủ tốt cho ảnh chụp thông thường, nhưng hoàn toàn bất lực trước một hóa đơn tiếng Trung hoặc ảnh chụp bảng tính chi chít dữ liệu. Đột phá đầu tiên của Qwen-VL là nâng lên 448x448 và hỗ trợ đầu ra hộp bao định vị (bounding box grounding), cho phép mô hình chỉ ra vị trí chính xác của từng đối tượng.
- **Video:** Video-LLaMA xếp chồng các bộ mã hóa khung hình riêng lẻ rồi nạp vào LLM. Cách làm này hiệu quả với các clip ngắn, nhưng thất bại đối với các video dài nhiều phút vốn đòi hỏi trục thời gian phải đóng vai trò là một tín hiệu liên tục. Đội ngũ Qwen muốn một bộ mã hóa duy nhất có khả năng hiểu trực tiếp dòng thời gian.
- **Đầu ra có cấu trúc (Structured output):** LLaVA chỉ xuất ra văn bản tự do. Một tác nhân tự hành (agent) lại cần định dạng JSON rõ ràng. Qwen-VL được huấn luyện trực tiếp trên các định dạng đầu ra JSON chuẩn xác, bao gồm cả tọa độ hộp bao dưới dạng văn bản.

Mỗi thế hệ tiếp theo của Qwen-VL đều mở rộng mạnh mẽ trên một trong ba trục then chốt này.

## Khái niệm cốt lõi

### Qwen-VL (Tháng 8 năm 2023)

Thế hệ đầu tiên sử dụng OpenCLIP ViT-bigG/14 làm bộ mã hóa (2,5 tỷ tham số), kết hợp Q-Former tương thích với kiến trúc LLaMA (1 bước với 256 truy vấn) cùng mô hình nền tảng Qwen-7B. Những đóng góp nổi bật:

- Độ phân giải 448x448 (tiên tiến nhất trong số các VLM mã nguồn mở tại thời điểm đó).
- Khả năng định vị tọa độ (Grounding): được huấn luyện trên các cặp ảnh - văn bản kèm theo các token tọa độ tường minh: `"The cat is at <box>(112, 204), (280, 344)</box>"`.
- Huấn luyện song ngữ Trung - Anh ngay từ ngày đầu tiên.

Kết quả benchmark thời điểm đó: cạnh tranh sòng phẳng với GPT-4V ở tiếng Anh và áp đảo hoàn toàn ở tiếng Trung. Năng lực định vị không gian được giám sát chặt chẽ là điểm nhấn lớn nhất của thế hệ này.

### Qwen2-VL (Tháng 9 năm 2024) — M-RoPE và độ phân giải tự nhiên

Qwen2-VL đã thay thế chồng kiến trúc "độ phân giải cố định + Q-Former" bằng một bộ mã hóa ViT hỗ trợ độ phân giải động tự nhiên (native dynamic resolution). Các thay đổi cốt lõi:

- **Độ phân giải động tự nhiên:** ViT chấp nhận bất kỳ kích thước $H \times W$ nào chia hết cho 28 (patch kích thước 14 kết hợp gộp không gian 2x). Một ảnh ở kích thước $1120 \times 672$ (tương ứng $40 \times 24$ patch sau khi gộp) sẽ sinh ra đúng 960 token thị giác. Không cần co giãn méo hình (resize), không cần chia mảnh (tiling), không cần ảnh thu nhỏ (thumbnail).
- **M-RoPE (Multimodal RoPE):** Mỗi token mang một vị trí không-thời gian 3D $(t, h, w)$ thay vì 1D. Đối với ảnh đơn $t=0$, đối với video $t = \text{frame\_index}$. M-RoPE xoay các vector truy vấn/khóa (query/key) theo các dải tần số riêng biệt cho từng trục. Hoàn toàn loại bỏ bảng nhúng vị trí cố định (positional embedding table).
- **Bộ chiếu MLP (MLP projector):** Loại bỏ Q-Former; sử dụng một MLP 2 lớp trực tiếp trên các token patch đã gộp.
- **Xử lý video với FPS động:** Video được lấy mẫu mặc định ở mức 1-2 FPS, nhưng mô hình chấp nhận số lượng khung hình tùy ý.

Kết quả: Qwen2-VL-7B đạt điểm số ngang ngửa GPT-4o trên nhiều benchmark đa phương thức và đánh bại GPT-4o trên DocVQA (94,5 so với 88,4). Bước chuyển đổi kiến trúc này là một quyết định thành công mang tính bản lề.

### Qwen2.5-VL (Tháng 2 năm 2025) — FPS động + thời gian tuyệt đối

Bước chuyển mình lớn nhất của Qwen2.5-VL tập trung vào video. Khái niệm FPS động không đơn thuần là "lấy mẫu nhiều khung hình hơn khi cần thiết". Bài báo đã hình thức hóa:

- **Thời gian tuyệt đối trong ID vị trí (Absolute time):** Thay vì sử dụng chỉ số khung hình rời rạc (khung 0, 1, 2...), trục thời gian của M-RoPE bám sát dấu thời gian thực tế tính bằng giây, do đó khoảng cách giữa hai ID vị trí của hai khung hình phản ánh chính xác số giây trôi qua giữa chúng. Ví dụ: "Tại giây 0:04, con mèo nhảy lên". Không cần thêm bất kỳ token văn bản nào để mô tả thời gian, do đó việc mã hóa không làm tăng chiều dài chuỗi.
- **Tốc độ khung hình động (Dynamic FPS):** Lấy mẫu ở mức 1 FPS cho các cảnh chuyển động chậm, và 4+ FPS cho các cảnh hành động nhanh. Người dùng hoặc bộ huấn luyện tự do quyết định; M-RoPE tự động thích ứng.
- **Cơ chế chú ý theo cửa sổ trong ViT (Window attention):** Cơ chế tự chú ý không gian được giới hạn trong các cửa sổ cục bộ để tối ưu thông lượng tính toán; chỉ bổ sung chú ý toàn cục (global attention) sau mỗi vài lớp.
- **Định dạng đầu ra JSON tường minh:** Được huấn luyện chuyên biệt trên dữ liệu gọi công cụ: `{"tool": "click", "coords": [380, 220]}`. Sẵn sàng hoạt động như một tác nhân tự hành ngay khi tải về.
- **Cơ chế co giãn MRoPE-v2:** Tọa độ vị trí được co giãn theo kích thước đầu vào tối đa, giúp video dài 10 phút không bị tràn dải tần số biểu diễn.

Kết quả benchmark: Qwen2.5-VL-72B vượt qua GPT-4o trên hầu hết các benchmark video, bắt kịp Gemini 2.0 trên tài liệu và lập đỉnh kỷ lục mới trong giới mã nguồn mở về khả năng định vị trên giao diện người dùng GUI (ScreenSpot: độ chính xác 84% so với 38% của GPT-4o).

### Qwen3-VL (Tháng 11 năm 2025)

Qwen3-VL là bản nâng cấp có tính chất củng cố và tối ưu hơn là phát minh lại từ đầu: sử dụng xương sống LLM lớn hơn (Qwen3-72B), mở rộng dữ liệu huấn luyện, nâng cao độ chính xác OCR và tăng cường khả năng suy luận thông qua "chế độ suy nghĩ" (thinking mode) của Qwen3. Kiến trúc ViT và cơ chế M-RoPE được giữ nguyên vẹn. Bài báo tập trung vào cải tiến dữ liệu và quy trình huấn luyện thay vì thay đổi các thành phần nguyên thủy.

Bài học rút ra từ dòng họ này: đến năm 2025, kiến trúc Qwen-VL đã chạm tới trạng thái ổn định tối ưu. Các thế hệ tiếp theo chủ yếu mở rộng quy mô tính toán và dữ liệu thay vì thay đổi cấu trúc nền tảng.

### Cơ sở toán học của M-RoPE

Phép nhúng vị trí quay cổ điển (Classical RoPE) xoay một vector truy vấn $q$ có số chiều $d$ tại vị trí $m$ theo từng cặp tọa độ:

```math
\begin{aligned}
q_{\text{rot}}[2i] &= q[2i] \cos(m \theta_i) - q[2i+1] \sin(m \theta_i) \\
q_{\text{rot}}[2i+1] &= q[2i] \sin(m \theta_i) + q[2i+1] \cos(m \theta_i) \\
\theta_i &= 10000^{-2i/d}
\end{aligned}
```

M-RoPE chia không gian chiều ẩn (hidden dimension) thành ba dải tần số độc lập. Giả sử $d = 96$, ta phân bổ 32 chiều cho thời gian ($t$), 32 chiều cho chiều cao ($h$), và 32 chiều cho chiều rộng ($w$). Mỗi dải được quay theo tọa độ của chính trục đó. Một patch tại vị trí $(t=5, h=10, w=20)$ sẽ lần lượt nhận các phép quay $R_t(5)$, $R_h(10)$, $R_w(20)$ áp dụng lên ba dải tương ứng.

Các token văn bản sử dụng $t = \text{text\_index}, h = 0, w = 0$ (hoặc một quy ước chuẩn hóa tương đương) để duy trì tính tương thích. Các khung hình video sử dụng $t = \text{frame\_time}, h = \text{row}, w = \text{col}$. Ảnh đơn lẻ sử dụng $t = 0$.

Lợi ích vượt trội: một cơ chế nhúng vị trí duy nhất xử lý đồng thời văn bản, hình ảnh và video mà không cần rẽ nhánh logic phức tạp hay phải duy trì các bảng nhúng vị trí khác nhau.

### Logic lấy mẫu FPS động

Cho một video có thời lượng $T$ giây và ngân sách token mục tiêu $B$:

1. Tính tốc độ khung hình tối đa có thể đáp ứng: `fps_max = B / (T * tokens_per_frame)`.
2. Chọn một FPS mục tiêu từ tập hợp $\{1, 2, 4, 8\}$ thỏa mãn điều kiện `fps <= fps_max`.
3. Nếu video có mức độ chuyển động cao (đo bằng luồng quang học - optical flow hoặc do người dùng chỉ định), chọn FPS cao hơn. Nếu chuyển động chậm, chọn FPS thấp hơn.
4. Lấy mẫu đều đặn theo mức FPS đã chọn; gán ID thời gian trong M-RoPE cho từng khung hình dựa trên dấu thời gian thực tế tính bằng giây.

Qwen2.5-VL học logic này một cách ngầm định trong quá trình huấn luyện; khi suy luận, người dùng có thể can thiệp trực tiếp thông qua tham số `fps`. Một chuỗi hành động 60 giây ở mức 4 FPS với 81 token mỗi khung hình sẽ tiêu tốn: $60 \times 4 \times 81 = 19.440$ token, hoàn toàn vừa vặn trong cửa sổ ngữ cảnh 32k.

### Đầu ra có cấu trúc cho agent

Quá trình huấn luyện agent của Qwen2.5-VL nhắm trực tiếp vào các lệnh gọi công cụ có cấu trúc:

```json
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Việc phân tích cú pháp hoàn toàn tất định (deterministic): chỉ cần áp dụng `JSON.parse` trực tiếp lên chuỗi văn bản do mô hình sinh ra. Hãy so sánh điều này với định dạng văn bản tự do như "click at (1024, 512)" vốn đòi hỏi biểu thức chính quy (regex) phức tạp và tiềm ẩn nhiều lỗi nhập nhằng. Bước chuyển biến này chính là nguyên nhân đưa điểm số ScreenSpot của Qwen2.5-VL nhảy vọt từ 55% của Qwen2-VL lên 84%.

```figure
mm-mrope-axes
```

## Ứng dụng thực tế

Tệp mã nguồn `code/main.py` cài đặt:

- Cơ chế tính toán vị trí M-RoPE cho một chuỗi đóng gói hỗn hợp gồm văn bản, patch hình ảnh và các khung hình video.
- Bộ lấy mẫu FPS động: dựa trên bộ ba tham số (thời lượng, ngân sách, mức độ chuyển động), tính toán FPS tối ưu và xuất ra các mốc thời gian khung hình.
- Bộ bóc tách đầu ra JSON mô phỏng theo Qwen2.5-VL xử lý phản hồi gọi công cụ kèm theo các trường tọa độ.

Hãy chạy thử chương trình để cảm nhận rõ sự khác biệt về số lượng token tiêu thụ khi chuyển đổi từ phương pháp FPS cố định sang FPS động trên một video dài 5 phút.

## Đóng gói triển khai

Bài học này tạo ra tệp artifact `outputs/skill-qwen-vl-pipeline-designer.md`. Khi nhận đầu vào là một tác vụ video cụ thể (giám sát an ninh, agent thao tác phần mềm, nhận dạng hành động, hỗ trợ người khiếm thị), kỹ năng này sẽ sinh ra cấu hình Qwen2.5-VL tối ưu (ngân sách khung hình, chiến lược FPS, cờ chú ý theo cửa sổ, chế độ đầu ra agent) kèm theo ước lượng độ trễ tương ứng. Hãy sử dụng kỹ năng này bất cứ khi nào bạn triển khai các mô hình thuộc dòng Qwen-VL cho các sản phẩm video thực tế.

## Bài tập

1. Hãy tính toán phép quay M-RoPE cho một patch nằm tại vị trí $(t=3, h=5, w=7)$ với số chiều ẩn là 48 (16 chiều cho mỗi dải, hệ số cơ sở theta bằng 10000). Trình bày góc quay chi tiết cho ba cặp số đầu tiên trong mỗi dải.

2. Một đoạn băng ghi hình từ camera an ninh dài 10 phút được lấy mẫu ở mức 1 FPS sẽ tạo ra bao nhiêu khung hình? Ở độ phân giải 384 với hệ số gộp 3x, tổng số token thị giác là bao nhiêu? Cửa sổ ngữ cảnh 32k mặc định của Qwen2.5-VL có chứa hết đoạn video này không?

3. Lựa chọn mức FPS hợp lý cho: một pha bóng bền quần vợt 30 giây, một video hướng dẫn nấu ăn 30 giây, và một video quay thao tác màn hình agent UI 30 giây. Giải thích căn cứ lựa chọn dựa trên logic lấy mẫu FPS động.

4. Qwen2.5-VL loại bỏ hoàn toàn module Q-Former. Tại sao một mạng MLP đơn giản lại hoạt động rất tốt vào năm 2025 nhưng lại thất bại vào năm 2023? (Gợi ý: quy mô dữ liệu và chất lượng của bộ mã hóa thị giác).

5. Bóc tách ba chuỗi phản hồi gọi công cụ JSON từ Qwen2.5-VL vào các dictionary trong Python. Những lỗi cú pháp nào thường gặp khiến việc parse JSON thất bại và tài liệu cookbook của Qwen khuyến nghị chiến lược khắc phục nào?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| M-RoPE | "RoPE đa phương thức" | Cơ chế nhúng vị trí quay 3D phân bổ các dải tần số theo trục thời gian, chiều cao và chiều rộng trong chiều ẩn |
| FPS động (Dynamic FPS) | "Lấy mẫu thông minh" | Tốc độ lấy mẫu khung hình được lựa chọn linh hoạt theo từng video dựa trên độ chuyển động, thời lượng và ngân sách token |
| Mã hóa thời gian tuyệt đối (Absolute time encoding) | "M-RoPE căn chỉnh theo thời gian" | Các ID vị trí thời gian được gán theo mốc giây thực tế của khung hình, giúp mô hình nhận biết số giây thực mà không tốn thêm token phụ trợ |
| Chú ý theo cửa sổ (Window attention) | "Chú ý cục bộ" | Phép tự chú ý không gian chỉ giới hạn trong các vùng khối nhỏ để tăng tốc độ xử lý; xen kẽ các lớp chú ý toàn cục |
| Đầu ra agent có cấu trúc (Structured agent output) | "Chế độ JSON" | Dữ liệu huấn luyện định hướng giúp VLM xuất ra cú pháp JSON hợp lệ chứa tên công cụ và tọa độ thao tác |
| min_pixels / max_pixels | "Biên độ phân giải" | Các tham số điều khiển của Qwen2.5-VL trên từng truy vấn nhằm giới hạn tổng số pixel và khống chế lượng token sinh ra |
| Định vị tọa độ (Grounding) | "Chỉ điểm đối tượng" | Hành vi xuất ra tọa độ hộp bao dưới dạng các token văn bản; là tính năng cốt lõi từ phiên bản Qwen-VL đầu tiên |

## Tài liệu tham khảo thêm

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
