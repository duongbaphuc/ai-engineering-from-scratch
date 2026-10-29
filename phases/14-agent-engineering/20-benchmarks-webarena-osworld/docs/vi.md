# Điểm chuẩn đánh giá: WebArena và OSWorld

> WebArena kiểm thử năng lực của tác tử web (web agents) trên bốn ứng dụng tự lưu trữ (self-hosted). OSWorld kiểm thử năng lực của tác tử máy tính để bàn (desktop agents) trên các hệ điều hành Ubuntu, Windows, macOS. Tại thời điểm công bố (2023–2024), cả hai bộ chuẩn đều chỉ ra khoảng cách rất lớn giữa các tác tử hàng đầu và con người. Khoảng cách này đang dần thu hẹp; tuy nhiên các chế độ lỗi cốt lõi vẫn không thay đổi.

**Type:** Learn  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)  
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả bốn ứng dụng tự lưu trữ của WebArena và giải thích tầm quan trọng của cơ chế đánh giá dựa trên thực thi (execution-based evaluation).
- Giải thích lý do OSWorld sử dụng ảnh chụp màn hình hệ điều hành thực tế thay vì các giao diện lập trình trợ năng (accessibility APIs).
- Nêu tên hai chế độ lỗi (failure modes) chính của OSWorld: định vị giao diện đồ họa (GUI grounding) và tri thức vận hành thao tác (operational knowledge).
- Tóm tắt những giá trị gia tăng mà OSWorld-G và OSWorld-Human đóng góp cho bộ điểm chuẩn cơ sở.

## Vấn đề đặt ra

Các tác tử tổng quát có thể gọi công cụ. Nhưng liệu chúng có thể điều khiển một trình duyệt web trải qua 20 lần nhấp chuột liên tiếp để hoàn tất quy trình thanh toán mua sắm trực tuyến? Liệu chúng có thể cấu hình một máy chủ Linux chỉ bằng bàn phím và chuột? Đó chính là những bài toán mà WebArena và OSWorld tìm kiếm câu trả lời.

## Khái niệm cốt lõi

### WebArena (Zhou et al., ICLR 2024)

- Bao gồm 812 tác vụ với chân trời tương tác dài (long-horizon tasks) trên bốn ứng dụng web tự lưu trữ: trang thương mại điện tử, diễn đàn thảo luận, công cụ quản lý mã nguồn tương tự GitLab, và một hệ quản trị nội dung (CMS) doanh nghiệp.
- Đi kèm các tiện ích hỗ trợ: bản đồ, máy tính bỏ túi, sổ ghi chú tạm (scratchpad).
- Đánh giá hoàn toàn dựa trên thực thi thông qua các giao diện lập trình gym (gym APIs) — đơn hàng đã được tạo thành công chưa, vấn đề (issue) đã được đóng lại chưa, trang CMS đã được cập nhật nội dung chưa?
- Tại thời điểm ra mắt: tác tử GPT-4 tốt nhất chỉ đạt tỷ lệ thành công 14,41% so với 78,24% của con người.

Cách tiếp cận tự lưu trữ đóng vai trò tối quan trọng — bộ điểm chuẩn không gặp hiện tượng bất định (flaky) do các ứng dụng mục tiêu được cố định phiên bản và có khả năng tái lập hoàn toàn.

### Các biến thể mở rộng

- **VisualWebArena** — các tác vụ gắn liền với thị giác nơi thành công phụ thuộc vào việc diễn giải chính xác hình ảnh (ảnh chụp màn hình được xem như quan sát hạng nhất - first-class observations).
- **TheAgentCompany** (Tháng 12/2024) — bổ sung thêm môi trường dòng lệnh (terminal) và tác vụ lập trình; phản ánh sát thực tế môi trường làm việc từ xa của kỹ sư phần mềm.

### OSWorld (Xie et al., NeurIPS 2024)

- Bao gồm 369 tác vụ máy tính thực tế trên các hệ điều hành Ubuntu, Windows, macOS.
- Điều khiển tự do bàn phím và chuột trên các ứng dụng thực tế.
- Ảnh chụp màn hình độ phân giải 1920×1080 đóng vai trò là quan sát đầu vào duy nhất.
- Tại thời điểm ra mắt: mô hình tốt nhất chỉ đạt 12,24% so với 72,36% của người dùng thực tế.

### Các chế độ lỗi cốt lõi

1. **Định vị giao diện người dùng (GUI grounding):** Ánh xạ từ điểm ảnh (pixel) sang phần tử điều khiển. Các mô hình gặp nhiều khó khăn trong việc định vị chính xác và ổn định các phần tử giao diện trên độ phân giải 1920×1080.
2. **Tri thức vận hành thao tác (Operational knowledge):** Tùy chọn cài đặt nằm ở menu nào, sử dụng phím tắt gì, bảng cấu hình ưu tiên nằm ở đâu. Đây là phần tri thức phân phối dài (knowledge tail) mà con người tích lũy qua nhiều năm sử dụng máy tính.

### Các nghiên cứu nối tiếp

- **OSWorld-G** — bộ dữ liệu định vị gồm 564 mẫu kèm tập dữ liệu huấn luyện Jedi. Tách biệt hoàn toàn bài toán định vị (grounding) khỏi bài toán lập kế hoạch (planning) để đo lường độc lập từng năng lực.
- **OSWorld-Human** — tập hợp các quỹ đạo hành động mẫu chuẩn (gold action trajectories) do chuyên gia tuyển chọn thủ công. Kết quả chỉ ra rằng các tác tử hàng đầu sử dụng số bước thao tác nhiều gấp 1,4 đến 2,7 lần so với cần thiết (khoảng cách hiệu quả quỹ đạo - trajectory-efficiency gap).

### Ý nghĩa thực tiễn

Tính năng sử dụng máy tính của Claude (Claude computer use), OpenAI CUA, và Gemini 2.5 Computer Use (Bài 21) đều được huấn luyện trên các khối lượng công việc được định hình bởi WebArena và OSWorld. Các bộ điểm chuẩn chính là mục tiêu; và các mô hình thương mại đưa ra sản xuất là câu trả lời được triển khai.

### Sai lầm trong đánh giá điểm chuẩn

- **Chỉ đánh giá dựa trên ảnh chụp màn hình:** OSWorld hoạt động hoàn toàn dựa trên ảnh chụp màn hình; việc đánh giá một tác tử sử dụng cây DOM hoặc API trợ năng trên OSWorld sẽ bỏ qua thách thức định vị cốt lõi.
- **Bỏ qua độ dài quỹ đạo thao tác:** Nếu chỉ tính tỷ lệ thành công mà không quan tâm đến độ dài hành động, bạn sẽ bỏ sót sự kém hiệu quả từ 1,4 đến 2,7 lần số bước mà OSWorld-Human đã vạch ra.
- **Để ứng dụng tự lưu trữ bị trôi phiên bản (Stale self-hosted apps):** Các ứng dụng của WebArena gắn chặt với các phiên bản cụ thể; việc tự ý nâng cấp phiên bản mà không thẩm định lại sẽ phá vỡ tính tương thích khi so sánh.

```figure
ae-agent-human-gap
```

## Thực hành Xây dựng (Build It)

Tệp `code/main.py` hiện thực một khung kiểm thử tác tử web mô phỏng:

- Một máy trạng thái (state machine) tối giản của "ứng dụng mua sắm": `list_items`, `add_to_cart`, `checkout`.
- Quỹ đạo mẫu chuẩn (gold trajectories) cho 3 tác vụ.
- Một tác tử hoạt động theo kịch bản thực hiện từng tác vụ.
- Bộ đánh giá dựa trên thực thi (kiểm tra trạng thái) và chỉ số hiệu quả quỹ đạo (số bước thực tế so với quỹ đạo chuẩn).

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả: tỷ lệ thành công theo từng tác vụ và chỉ số hiệu quả quỹ đạo, mô phỏng sát phương pháp luận của OSWorld-Human.

## Ứng dụng Thực tế (Use It)

- **WebArena Verified:** Triển khai tự lưu trữ trên một cụm máy chủ nội bộ để phục vụ quy trình đánh giá liên tục (continuous evaluation).
- **OSWorld:** Vận hành trên một đội máy ảo (VM fleet) để đánh giá tác tử máy tính để bàn.
- **Tác tử sử dụng máy tính (Computer-use agents):** (Xem Bài 21) Claude, OpenAI CUA, Gemini — đều được tôi luyện trên các luồng công việc này.
- **Luồng nghiệp vụ sản phẩm nội bộ:** Ghi lại quỹ đạo thao tác chuẩn cho 20 tác vụ quan trọng nhất của bạn; chạy tác tử đánh giá định kỳ hàng tuần.

## Xuất xưởng Sản phẩm (Ship It)

Tệp `outputs/skill-web-desktop-harness.md` cung cấp bộ khung kỹ năng giúp xây dựng khung kiểm thử tác tử web/desktop với cơ chế đánh giá dựa trên thực thi và đo lường hiệu quả quỹ đạo hành động.

## Bài tập

1. Mở rộng khung kiểm thử mô phỏng với ứng dụng thứ hai (một diễn đàn thảo luận). Viết 3 tác vụ kèm quỹ đạo thao tác chuẩn tương ứng.
2. Bổ sung báo cáo hiệu quả quỹ đạo cho từng tác vụ. Trên hệ thống mô phỏng của bạn, tác tử tốn số bước gấp 1 lần, 2 lần hay 3 lần so với quỹ đạo chuẩn?
3. Triển khai một công cụ "gây nhiễu" (distractor tool) — công cụ mà quỹ đạo chuẩn không bao giờ sử dụng. Tác tử dạng kịch bản có bị dụ dỗ gọi công cụ này không?
4. Đọc tài liệu nghiên cứu OSWorld-G. Bạn sẽ phân tách các lỗi định vị (grounding failures) khỏi lỗi lập kế hoạch (planning failures) như thế nào trong hệ thống đánh giá của riêng bạn?
5. Đọc tệp README về ứng dụng của WebArena. Những thành phần nào sẽ bị hỏng hóc nếu bạn nâng cấp một trong các phiên bản ứng dụng được ghim sẵn?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| WebArena | "Điểm chuẩn tác tử web" | 812 tác vụ trên 4 ứng dụng tự lưu trữ; đánh giá theo kiểu môi trường gym |
| VisualWebArena | "WebArena thị giác" | WebArena gắn liền thị giác; ảnh chụp màn hình là quan sát đầu vào |
| OSWorld | "Điểm chuẩn tác tử desktop" | 369 tác vụ trên các hệ điều hành thực tế Ubuntu/Windows/macOS |
| GUI grounding | "Ánh xạ pixel sang phần tử" | Năng lực của mô hình trong việc định vị phần tử giao diện trên độ phân giải 1920x1080 |
| Tri thức vận hành | "Kinh nghiệm dùng hệ điều hành" | Menu nào, phím tắt nào, bảng tùy chọn cấu hình nào |
| OSWorld-G | "Bộ kiểm thử định vị" | 564 mẫu chuyên biệt cho định vị kèm tập dữ liệu huấn luyện |
| OSWorld-Human | "Quỹ đạo mẫu chuẩn" | Chuỗi hành động chuyên gia thu thập thủ công để đo lường hiệu quả |
| Hiệu quả quỹ đạo | "Số bước vượt chuẩn" | Tổng số bước tác tử thực hiện chia cho số bước tối thiểu của con người |

## Đọc thêm

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — bộ điểm chuẩn web 4 ứng dụng
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — bộ điểm chuẩn máy tính để bàn đa hệ điều hành
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — năng lực sử dụng máy tính của Claude
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — các thông số thực nghiệm trên OSWorld và WebArena
