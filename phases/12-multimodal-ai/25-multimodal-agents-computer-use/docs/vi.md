# Tác tử đa thể thức và khả năng sử dụng máy tính (Multimodal Agents and Computer-Use)

> Sản phẩm công nghệ tiên phong của năm 2026 là một tác tử đa thể thức (multimodal agent) có khả năng đọc ảnh chụp màn hình, nhấp chuột vào các nút bấm, điều hướng giao diện web, điền biểu mẫu và hoàn tất các quy trình công việc phức tạp từ đầu đến cuối. SeeClick và CogAgent (2024) đã chứng minh tính khả thi của cơ chế định vị giao diện đồ họa (GUI grounding). Ferret-UI mở rộng năng lực sang giao diện di động. ChartAgent giới thiệu kỹ thuật sử dụng công cụ thị giác cho các biểu đồ. VisualWebArena và AgentVista (2026) là những bộ điểm chuẩn đo lường tiêu chuẩn mà các mô hình biên giới đang theo đuổi — nơi ngay cả Gemini 3 Pro và Claude Opus 4.7 cũng chỉ đạt khoảng 30% trên các tác vụ khó của AgentVista. Bài học lớn (Capstone) này tổng hòa mọi mạch kiến thức của Phase 12: khả năng tri giác (VLM độ phân giải cao), suy luận logic (LLM tích hợp gọi công cụ), định vị chính xác (sinh tọa độ màn hình), bộ nhớ dài hạn và phương pháp đánh giá toàn diện.

**Type:** Capstone
**Languages:** Python (stdlib, lược đồ hành động + khung sườn vòng lặp tác tử)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 phút

## Mục tiêu học tập

- Thiết kế vòng lặp tác tử đa thể thức hoàn chỉnh: tri giác (perceive) $\to$ suy luận (reason) $\to$ hành động (act) $\to$ quan sát (observe) $\to$ lặp lại (repeat).
- Xây dựng lược đồ đầu ra định vị GUI có cấu trúc (tọa độ nhấp chuột, gõ văn bản, cuộn trang, kéo thả) để VLM phát ra dưới dạng JSON chuẩn.
- So sánh các kiến trúc tác tử: chỉ dựa trên ảnh chụp màn hình (screenshot-only), dựa trên cây trợ năng (accessibility-tree), và tác tử lai kết hợp (hybrid).
- Thiết lập quy trình đánh giá điểm chuẩn tác tử đa thể thức trên một tập con thu nhỏ của VisualWebArena.

## Vấn đề đặt ra

Xét một quy trình đặt vé trên trang web thương mại: "Hãy tìm cho tôi một chuyến bay tới Tokyo vào ngày 15 tháng 4, ghế cạnh lối đi với giá dưới $800, rồi tiến hành đặt vé."

Một tác tử đa thể thức phải thực hiện tuần tự các bước:

1. Chụp ảnh màn hình (screenshot) hiện tại của trình duyệt.
2. Phân tích ảnh chụp màn hình + địa chỉ URL + mục tiêu người dùng để lập kế hoạch hành động.
3. Sinh ra một hành động có cấu trúc: nhấp chuột (tại tọa độ $x, y$), gõ chữ "Tokyo" (vào phần tử $E$), cuộn trang xuống dưới, chọn ô tùy chọn (radio button).
4. Thực thi hành động đó trực tiếp lên trình duyệt.
5. Quan sát trạng thái mới (chụp ảnh màn hình tiếp theo).
6. Lặp lại chu trình cho đến khi hoàn thành toàn bộ tác vụ.

Mỗi bước đều là một lượt gọi mô hình VLM đa thể thức. Đầu ra của VLM bắt buộc phải là chuỗi JSON hợp lệ để có thể phân tích cú pháp tự động. Sai số có xu hướng tích tụ dần qua từng bước, do đó năng lực tự phục hồi sau lỗi (error recovery) đóng vai trò quyết định sự thành bại.

## Khái niệm cốt lõi

### Định vị GUI — Khối nguyên mẫu nền tảng

Định vị GUI (GUI grounding) được định nghĩa là: tiếp nhận một ảnh chụp màn hình và một câu chỉ dẫn bằng ngôn ngữ tự nhiên, sau đó sinh ra tọa độ $(x, y)$ chuẩn xác để nhấp chuột (hoặc thực hiện thao tác tương ứng).

- **SeeClick** (arXiv:2401.10935): Công trình nguồn mở quy mô lớn đầu tiên. Tinh chỉnh VLM trên dữ liệu GUI tổng hợp lẫn thực tế, xuất trực tiếp tọa độ dưới dạng các token văn bản thông thường.
- **CogAgent** (arXiv:2312.08914): Bổ sung bộ mã hóa độ phân giải cao 1120x1120 chuyên dụng cho các giao diện web dày đặc phần tử. Đạt độ chính xác ~84% trên tác vụ điều hướng web.
- **Ferret-UI** (arXiv:2404.05719): Tập trung chuyên sâu vào giao diện ứng dụng di động, tích hợp chặt chẽ với dữ liệu cây trợ năng của iOS.

Định dạng đầu ra tiêu chuẩn thường là cấu trúc JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

Trường `element_desc` (mô tả phần tử) giúp hệ thống tự phục hồi: nếu tọa độ $(x, y)$ bị lệch do màn hình cuộn hoặc co giãn, gợi ý ngữ nghĩa này sẽ giúp tác tử tái định vị phần tử mục tiêu.

### Lược đồ hành động (Action Schemas)

Một lược đồ hành động tiêu chuẩn thường bao gồm từ 6 đến 10 loại thao tác cơ bản:

- `click`: (tọa độ x, y)
- `type`: (chuỗi văn bản, x?, y?)
- `scroll`: (hướng cuộn, khoảng cách cuộn)
- `drag`: (x0, y0, x1, y1)
- `select`: (chỉ số tùy chọn option_index)
- `hover`: (tọa độ x, y)
- `navigate`: (địa chỉ url)
- `wait`: (thời gian chờ ms)
- `done`: (trạng thái thành công success, giải thích explanation)

Tác tử phát ra một hành động tại mỗi bước. Module điều khiển trình duyệt (browser wrapper) sẽ thực thi hành động đó và trả về trạng thái màn hình mới.

### So sánh giữa phương pháp thuần ảnh chụp màn hình và cây trợ năng

Có hai chế độ tiếp nhận đầu vào chính:

- **Thuần ảnh chụp màn hình (Screenshot-only):** Chỉ nhận ảnh thô, hoàn toàn không có thông tin cấu trúc mã nguồn. Tính tổng quát cao nhất; có thể áp dụng cho mọi ứng dụng trên mọi hệ điều hành.
- **Cây trợ năng (Accessibility tree):** Tiếp nhận thông tin cấu trúc phân cấp từ DOM của trình duyệt hoặc API trợ năng của hệ điều hành (Accessibility API). Cực kỳ đáng tin cậy để định vị tọa độ; áp dụng tốt ở những nơi cây cấu trúc có sẵn.
- **Mô hình lai kết hợp (Hybrid):** Sử dụng cả hai. Dùng cây trợ năng làm mỏ neo đáng tin cậy để thực thi các hành động nguyên tử và dùng ảnh chụp màn hình để nắm bắt ngữ cảnh ngữ nghĩa trực quan.

Các tác tử trong môi trường sản xuất luôn ưu tiên giải pháp lai khi có thể. Tự động hóa trình duyệt web (Selenium / Playwright + Accessibility API) luôn trích xuất được cây cấu trúc; trong khi các ứng dụng máy tính để bàn (desktop apps) đôi khi chỉ có thể tiếp cận qua ảnh chụp màn hình.

### Cơ chế quản lý bộ nhớ chân trời dài (Long-Horizon Memory)

Một quy trình làm việc gồm 20 bước sẽ tạo ra 20 bức ảnh chụp màn hình dung lượng lớn. Cửa sổ ngữ cảnh của VLM sẽ nhanh chóng bị lấp đầy. Ba chiến lược nén bộ nhớ phổ biến:

- **Chuỗi tóm tắt định kỳ (Summary-chain):** Cứ sau mỗi 5 bước thao tác, mô hình tóm tắt lại toàn bộ những gì đã diễn ra thành văn bản ngắn và loại bỏ ảnh chụp màn hình cũ khỏi ngữ cảnh.
- **Bỏ khung hình ngắt quãng (Skip-frame):** Chỉ giữ lại ảnh chụp đầu tiên, ảnh chụp gần nhất và cách 3 khung hình mới giữ lại một ảnh.
- **Nhật ký hành động bằng văn bản (Tool-recorded log):** Ghi nhận một bản nhật ký bằng chữ mô tả các thao tác đã thực hiện; không nạp lại các ảnh chụp màn hình cũ vào lượt suy luận sau.

API sử dụng máy tính (computer-use API) của Claude áp dụng mẫu hình nhật ký hành động bằng văn bản này: đơn giản hơn, tiết kiệm token và hoạt động ổn định hơn.

### Sử dụng công cụ thị giác (Visual Tool Use)

ChartAgent (arXiv:2510.04514) giới thiệu khả năng sử dụng công cụ thị giác trong việc phân tích biểu đồ: cắt ảnh (crop), phóng to (zoom), gọi module OCR chuyên sâu, hoặc gọi các mô hình phát hiện vật thể ngoại vi. Tác tử có thể phát ra một lệnh gọi công cụ: `"cắt vùng ảnh (100, 200, 300, 400) rồi chạy OCR"`. Công cụ trả về chuỗi văn bản trích xuất; và VLM tiếp tục chu trình suy luận logic.

Mô hình này mở rộng tự nhiên sang kỹ thuật kích hoạt theo dấu (set-of-mark prompting), khoanh vùng chú thích và tích hợp các công cụ chuyên dụng — tất cả đều tuân theo cùng một cơ chế "phát lệnh gọi công cụ $\to$ nhận phản hồi cấu trúc".

### Các bộ điểm chuẩn đo lường tiêu chuẩn năm 2026

- **ScreenSpot-Pro:** Đánh giá khả năng định vị GUI trên khoảng 1.000 ảnh chụp màn hình web. Mô hình mở SOTA Qwen2.5-VL-72B đạt ~85%. Các mô hình biên giới đạt ~90%.
- **VisualWebArena:** Đánh giá toàn trình các tác vụ web phức tạp (mua sắm trực tuyến, diễn đàn thảo luận, trang rao vặt). Mô hình mở SOTA đạt ~20%. Gemini 3 Pro đạt ~27%.
- **AgentVista** (arXiv:2602.23166): Thước đo thách thức nhất năm 2026. Bao gồm các quy trình thực tế trải rộng trên 12 lĩnh vực nghiệp vụ. Các mô hình biên giới chỉ đạt từ 27-40%; trong khi các mô hình nguồn mở chỉ đạt từ 10-20%.
- **WebArena / WebShop:** Các bộ điểm chuẩn thời kỳ trước; hiện đã bị bão hòa bởi các mô hình thế hệ mới.

### Những rào cản kỹ thuật cốt lõi

Các nút thắt cổ chai hạn chế hiệu năng của tác tử:

1. **Định vị thị giác ở quy mô nhỏ (Fine-scale grounding):** Yêu cầu "nhấp vào dấu X nhỏ để đóng cửa sổ" thường thất bại trên màn hình độ phân giải điện thoại di động.
2. **Lập kế hoạch dài hạn (Long-horizon planning):** Sau khoảng 10 bước hành động liên tiếp, tác tử thường bị chệch hướng khỏi mục tiêu ban đầu.
3. **Khả năng tự phục hồi sau lỗi:** Khi thao tác nhấp chuột không thành công (ấn nhầm nút), khả năng tự phát hiện lỗi và điều chỉnh lại hành động rất hiếm khi xuất hiện trong dữ liệu huấn luyện.
4. **Ngữ cảnh liên trang:** Việc chuyển đổi qua lại giữa các tab trình duyệt hoặc các biểu mẫu dài nhiều trang thường làm thất thoát trạng thái ghi nhớ.

Các hướng nghiên cứu mũi nhọn hiện nay: hoàn thiện kiến trúc bộ nhớ, cơ chế tái lập kế hoạch tường minh (explicit replanning), và module xác thực đa thể thức (so khớp ảnh màn hình để kiểm tra hành động có thành công hay không).

### Bài tập lớn tổng hợp (Capstone Build-It)

Nhiệm vụ thực hành của bài học lớn: xây dựng một tác tử sử dụng máy tính có năng lực:

1. Đọc mã HTML + ảnh chụp màn hình của một trang web đặt chỗ giả lập.
2. Lập kế hoạch chuỗi thao tác đa bước: tìm kiếm $\to$ lựa chọn kết quả $\to$ điền biểu mẫu $\to$ gửi thông tin.
3. Sinh các lệnh JSON tuân thủ chuẩn xác lược đồ hành động.
4. Tự đánh giá trên một bộ kiểm thử chuẩn cố định gồm 10 tác vụ.

Bài học cung cấp sẵn khung sườn mã nguồn hoàn chỉnh, sẵn sàng kết nối và mở rộng với các trình duyệt thực tế.

```figure
mm-agent-loop
```

## Sử dụng trong thực tế

Tệp `code/main.py` là khung sườn bài tập lớn hoàn chỉnh:

- Định nghĩa lược đồ hành động bằng JSON (10 loại hành động: 9 hành động cơ bản nêu trên cộng thêm công cụ `screenshot_region` phục vụ cắt và phóng to ảnh).
- Đối tượng giả lập trạng thái trình duyệt dưới dạng từ điển (dict).
- Khung sườn vòng lặp tác tử: tiếp nhận trạng thái $\to$ sinh hành động $\to$ thực thi $\to$ lặp lại chu trình.
- Bộ đo chuẩn mini gồm 10 tác vụ (trên các trang HTML tổng hợp) để đo lường tỷ lệ hoàn thành tác vụ toàn trình.
- Điểm móc xử lý phục hồi lỗi (error-recovery hook) khi một hành động thực thi thất bại.

## Triển khai sản phẩm

Bài học này xây dựng `outputs/skill-multimodal-agent-designer.md`. Dựa trên đặc tả sản phẩm tác tử máy tính (miền ứng dụng, tập hành động cho phép, mục tiêu đánh giá), kỹ năng này tự động thiết kế toàn bộ vòng lặp tác tử, chiến lược quản lý bộ nhớ, phương thức định vị giao diện và dự phóng điểm số benchmark kỳ vọng.

## Bài tập thực hành

1. Lược đồ trong mã nguồn bổ sung thêm công cụ `screenshot_region` (cắt ảnh + phóng to) bên cạnh 9 hành động cốt lõi. Những tác vụ giao diện cụ thể nào sẽ được hưởng lợi trực tiếp từ công cụ này?

2. Đọc bài báo nghiên cứu AgentVista (arXiv:2602.23166). Hãy mô tả phân nhóm tác vụ khó khăn nhất và giải thích nguyên nhân cốt lõi khiến các mô hình biên giới hiện nay vẫn thất bại.

3. Kỹ thuật nén bộ nhớ chân trời dài: hãy thiết kế một cấu trúc chuỗi tóm tắt duy trì tối đa 4 ảnh chụp màn hình trong ngữ cảnh hoạt động, trong khi lưu trữ toàn bộ lịch sử thao tác dưới dạng nhật ký văn bản.

4. Xây dựng một cơ chế xử lý phục hồi lỗi (error-recovery hook): khi một hành động thất bại (ví dụ không tìm thấy nút bấm tại tọa độ đã định), tác tử cần thực hiện các bước xử lý tiếp theo như thế nào?

5. So sánh hiệu năng giữa Claude 4.7 hoạt động ở chế độ thuần ảnh chụp màn hình và Qwen2.5-VL hoạt động ở chế độ lai (ảnh chụp màn hình + cây trợ năng) trên 10 tác vụ web thực tế. Mô hình nào chiến thắng ở những dạng bài toán nào?

## Các thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Định vị GUI (GUI grounding) | "Tọa độ nhấp chuột" | Mô hình xuất tọa độ (x, y) trên ảnh chụp màn hình ứng với mục tiêu của câu lệnh |
| Lược đồ hành động (Action schema) | "Định nghĩa công cụ" | Đặc tả JSON mô tả các hành động hợp lệ (nhấp chuột, gõ phím, cuộn trang, kéo thả) |
| Cây trợ năng (Accessibility tree) | "DOM cấu trúc" | Cấu trúc phân cấp giao diện có thể đọc bằng máy trích xuất từ API trình duyệt hoặc hệ điều hành |
| Tác tử lai (Hybrid agent) | "Ảnh chụp + Cây trợ năng" | Kết hợp đồng thời cả hình ảnh trực quan và dữ liệu cấu trúc; ổn định hơn từng phương pháp đơn lẻ |
| Công cụ thị giác (Visual tool use) | "Phóng to / Cắt ảnh / Phát hiện" | Tác tử chủ động gọi các công cụ thị giác ngoại vi (OCR, phát hiện vật thể) giữa các bước lập kế hoạch |
| Chuỗi tóm tắt (Summary-chain) | "Nén bộ nhớ" | Tóm tắt văn bản định kỳ thay thế dần lịch sử ảnh chụp màn hình dung lượng lớn |
| VisualWebArena | "Chuẩn đo web toàn trình" | Bộ dữ liệu đo chuẩn năm 2024 dành cho các tác vụ tương tác web từ đầu đến cuối |
| AgentVista | "Chuẩn đo khó 2026" | Bộ kiểm thử gồm 12 lĩnh vực nghiệp vụ thực tế; ngay cả Gemini 3 Pro cũng chỉ đạt ~30% |

## Tài liệu đọc thêm

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
