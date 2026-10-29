# Nghiên cứu Căn chỉnh Tự động (Anthropic AAR - Automated Alignment Research)

> Anthropic đã triển khai các nhóm Nhà nghiên cứu Căn chỉnh Tự trị (Autonomous Alignment Researchers - AAR) sử dụng Claude Opus 4.6 hoạt động song song trong các hộp cát độc lập, điều phối thông qua một diễn đàn chung có nhật ký lưu trữ bên ngoài mọi hộp cát (nhờ đó các tác tử không thể tự xóa hồ sơ của chính mình). Trên bài toán huấn luyện từ yếu sang mạnh (weak-to-strong training), các AAR đã vượt trội hơn các nhà nghiên cứu con người. Bản tóm tắt của chính Anthropic nhấn mạnh rằng các quy trình làm việc được định sẵn (prescribed workflows) thường hạn chế tính linh hoạt của AAR và làm giảm hiệu năng. Tự động hóa nghiên cứu căn chỉnh làm nén chặt dòng thời gian, và điều này diễn ra ngay trên chính những rủi ro mất căn chỉnh mà Chính sách Mở rộng có Trách nhiệm (RSP) được thiết kế để phát hiện.

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 phút

## Vấn đề

Nghiên cứu căn chỉnh (alignment research) tiêu tốn rất nhiều thời gian của các nhà nghiên cứu con người. Các bài toán như giám sát có khả năng mở rộng (scalable oversight), đặc tả hàm phần thưởng (reward specification), hay huấn luyện từ yếu sang mạnh (weak-to-strong training) đòi hỏi các thực nghiệm kéo dài nhiều tuần cho mỗi vòng lặp. Khi năng lực mô hình biên giới ngày càng tăng tốc, khối lượng công việc căn chỉnh tăng nhanh hơn nhiều so với nguồn cung nhân lực nghiên cứu đủ tiêu chuẩn.

Nghiên cứu Căn chỉnh Tự động (Automated Alignment Research - AAR) đặt câu hỏi: liệu chính những mô hình biên giới có năng lực đang vượt xa khả năng căn chỉnh có thể đóng góp vào việc thu hẹp khoảng cách này hay không? Báo cáo năm 2026 của Anthropic về nghiên cứu huấn luyện từ yếu sang mạnh do AAR vận hành (alignment.anthropic.com/2026/automated-w2s-researcher/) là một trong những kết quả công khai đầu tiên từ một hệ thống thuộc lớp này được triển khai trong thực tế.

Kết quả thu được mang tính tích cực thực sự: các AAR đã giải quyết một bài toán nghiên cứu tốt hơn cả mức cơ sở của con người. Tuy nhiên, kết quả này cũng tập trung lại mọi mối lo ngại mà phase học này đã xây dựng. Nếu nghiên cứu căn chỉnh có thể tự động hóa, thì những phần làm tổn hại rào chắn an toàn cũng có thể tự động hóa. Các ngưỡng năng lực Nghiên cứu & Phát triển AI (AI R&D) trong Chính sách Mở rộng có Trách nhiệm (RSP) được xây dựng với sự cân nhắc kỹ lưỡng đối với vòng lặp này.

## Khái niệm

### Kiến trúc (như được mô tả công khai)

- **Các tác tử song song.** Nhiều phiên bản Claude Opus 4.6 chạy dưới dạng các AAR, mỗi phiên bản nằm trong một hộp cát độc lập. Hộp cát giới hạn bề mặt hệ thống tệp, mạng và các tiến trình.
- **Diễn đàn dùng chung.** Các phát hiện được công bố lên một diễn đàn dùng chung có phần phụ trợ lưu trữ nằm hoàn toàn bên ngoài mọi hộp cát. Các tác tử có thể đọc diễn đàn; các tác tử không thể xóa hoặc chỉnh sửa các bản ghi trên diễn đàn từ bên trong hộp cát của chúng.
- **Phân bổ tác vụ.** Một hàng đợi tác vụ do con người thiết kế phân công các bài toán nghiên cứu cho các AAR. Các AAR đề xuất phương án phân rã bài toán, thực thi thí nghiệm và viết báo cáo phát hiện.
- **Trọng tâm huấn luyện từ yếu sang mạnh.** Nghiên cứu được công bố giải quyết bài toán huấn luyện từ yếu sang mạnh — sử dụng một mô hình giám sát yếu hơn để huấn luyện một mô hình học trò mạnh hơn. Các AAR đã thể hiện vượt trội hơn các nhà nghiên cứu con người trong tác vụ này.

### Lý do nhật ký bên ngoài hộp cát mang tính quyết định

Đặc tính toàn vẹn của nhật ký (log-integrity) là ranh giới giữa một thực nghiệm được kiểm soát và một thực nghiệm có nguy cơ bị thao túng. Nếu các AAR có thể chỉnh sửa nhật ký của chính mình, chúng có thể che đậy các thực nghiệm thất bại hoặc đáng ngờ. Anthropic nhấn mạnh trực tiếp điều này: đặc tính này là điều kiện cần để tin tưởng vào kết quả nghiên cứu, và việc xác minh nó là không hề tầm thường.

Về mặt hạ tầng, cơ chế tương đương là nhật ký chỉ-ghi-nối (append-only logs) với khả năng ghi xuyên suốt (write-through) tới bộ lưu trữ bên ngoài (ví dụ: S3 object-lock, nhật ký bất biến của Datadog). Tác tử không có chứng chỉ để sửa đổi các bản ghi trong quá khứ. Đây cũng chính là mô hình mà các nhóm bảo mật doanh nghiệp năm 2026 sử dụng để xây dựng dấu vết kiểm toán chống mã độc tống tiền (ransomware).

### Sự đánh đổi của quy trình làm việc định sẵn

Báo cáo của Anthropic ghi nhận rằng các quy trình làm việc được định sẵn (prescribed workflows) thường hạn chế tính linh hoạt của AAR và làm giảm sút hiệu năng. Nói cách khác: nếu nhà thiết kế con người viết kế hoạch nghiên cứu từng bước một cách cứng nhắc, các AAR thể hiện kém hơn so với khi được giao các mục tiêu cấp cao và quyền tự do tự phân rã nhiệm vụ. Điều này nhất quán với các kết quả trên khắp lĩnh vực nghiên cứu tác tử (LangGraph, Microsoft Agent Framework) — các khung giàn bị quy định quá mức sẽ kìm hãm các tác tử có năng lực cao.

Hệ quả an toàn ở đây rất tinh tế. Sự linh hoạt giúp cải thiện hiệu năng của AAR cũng chính là sự linh hoạt cho phép phân rã ngoài ý muốn, sử dụng công cụ lạ lẫm và trôi dạt mục tiêu (goal-drift). Sự đánh đổi này là một quyết định trong không gian thiết kế: mức độ linh hoạt nào mang lại đủ năng lực để vượt qua con người, mà không trao cho tác tử khoảng trống để khái quát hóa mục tiêu lệch khỏi phân phối ban đầu?

### Rủi ro nén thời gian (Compression risk)

RSP v3.0 (Bài 19) giới thiệu một ngưỡng năng lực AI R&D: khả năng tự động hóa hoàn toàn đường ống Nghiên cứu & Phát triển AI ở mức chi phí cạnh tranh so với sự kết hợp giữa con người và công cụ AI. Khuôn khổ FSF v3 của DeepMind cũng bao gồm một cấp độ tự trị ML R&D tương đương. Cả hai khuôn khổ đều coi ngưỡng này là điều kiện kích hoạt các biện pháp kiểm soát nâng cao.

AAR chỉ còn thiếu một bước nữa là chạm tới ngưỡng này: nó tự động hóa một phần của đường ống (nghiên cứu căn chỉnh trên các tác vụ cụ thể, có phạm vi rõ ràng) nhưng chưa phải là toàn bộ vòng lặp phát triển năng lực đầu-cuối. Câu hỏi về mặt thời gian là khoảng cách đó sẽ thu hẹp nhanh đến mức nào.

Thời gian bị nén lại chính là mối lo ngại về các lỗi cộng dồn (compounding-failure). Nếu nghiên cứu căn chỉnh và nghiên cứu năng lực cùng tăng trưởng với tốc độ tương đương, bề mặt rủi ro mất căn chỉnh sẽ mở rộng nhanh ít nhất bằng năng lực. Nếu năng lực tăng trưởng nhanh hơn (như xu hướng lịch sử), khoảng cách sẽ ngày càng doãng rộng. Đây là lập luận cho thấy AAR là một điều tốt có điều kiện: mỗi kết quả căn chỉnh bổ sung chỉ thu hẹp khoảng cách khi và chỉ khi quy trình nghiên cứu là đáng tin cậy.

### Những gì AAR không thay thế

Các nhà nghiên cứu con người vẫn là bên thiết lập hàng đợi tác vụ, rà soát kết quả và nắm giữ quyền hạn hiến pháp cao nhất. Các AAR chỉ tăng tốc phần giữa của đường ống, không phải hai đầu mút. Các kết quả công bố của Anthropic bao gồm cả đóng góp của AAR và phán đoán của các nhà nghiên cứu con người về việc nên công bố điều gì, rút lại điều gì và tinh chỉnh điều gì.

Điều này khớp hoàn toàn với mô hình đề xuất-rồi-cam kết (propose-then-commit) từ Bài 15 được áp dụng cho chính công việc nghiên cứu: AAR đề xuất; con người phê duyệt và cam kết.

```figure
aar-forum
```

## Sử dụng

`code/main.py` mô phỏng một diễn đàn nghiên cứu đa tác tử AAR. Ba "tác tử nghiên cứu" độc lập chạy song song, đăng các phát hiện lên một diễn đàn chỉ-ghi-nối đi kèm các thử nghiệm can thiệp giả mạo tùy chọn.

Hãy quan sát:

- Cách việc ghi nhật ký bên ngoài hộp cát bắt được các nỗ lực giả mạo nhật ký.
- Cách phân bổ tác vụ với khả năng tự do phân rã vượt trội hơn quy trình làm việc cố định trên các bài toán khó.
- Một AAR bị tổn hại duy nhất có thể làm vấy bẩn tín hiệu của toàn bộ diễn đàn ra sao nếu nhật ký không có cơ chế phát hiện giả mạo (tamper-evident).

## Đóng gói

`outputs/skill-aar-deployment-review.md` là danh mục kiểm tra tiền triển khai cho một đường ống nghiên cứu căn chỉnh tự động.

## Bài tập

1. Chạy `code/main.py`. So sánh thiết lập "quy trình cố định" (fixed-workflow) với "tự do phân rã" (free-decomposition). Liệu tự do phân rã có luôn giành chiến thắng, hay có những lớp bài toán mà quy trình cố định có tính cạnh tranh tương đương?

2. Sửa đổi trình mô phỏng để một tác tử tìm cách giả mạo nhật ký. Xác nhận rằng nhật ký chỉ-ghi-nối phát hiện được hành vi này. Viết một đoạn văn mô tả chính xác dấu hiệu nhận diện trong nhật ký trông như thế nào.

3. Đọc báo cáo AAR từ yếu sang mạnh của Anthropic. Xác định tác vụ con cụ thể mà các AAR đã đánh bại các nhà nghiên cứu con người. Điều gì khiến tác vụ đó dễ dàng được tự động hóa?

4. Thiết kế chính sách phân bổ hàng đợi tác vụ cân bằng giữa tính linh hoạt của AAR (kết quả tốt hơn) với các ràng buộc quy trình định sẵn (dễ kiểm toán hơn). Mô tả cách bạn sẽ tiến hành thử nghiệm A/B cho hai phương án này.

5. Đọc ngưỡng AI R&D-4 của RSP v3.0. Trong một đoạn văn, mô tả những yếu tố nào mà bạn cho rằng sẽ vượt qua ngưỡng đó mà hiện tại AAR vẫn chưa đạt được.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| AAR | "Nhà nghiên cứu căn chỉnh tự động" | Phiên bản Claude Opus 4.6 vận hành tự trị trên các bài toán căn chỉnh |
| Huấn luyện từ yếu sang mạnh (Weak-to-strong training) | "Huấn luyện mô hình mạnh bằng bộ giám sát yếu" | Điểm chuẩn giám sát có thể mở rộng kinh điển mà AAR đã vượt qua con người |
| Diễn đàn dùng chung (Shared forum) | "Nơi các tác tử công bố phát hiện" | Bộ lưu trữ chỉ-ghi-nối, nằm hoàn toàn bên ngoài hộp cát |
| Nhật ký bên ngoài hộp cát (Out-of-sandbox log) | "Tác tử không thể sửa hồ sơ của chính mình" | Cơ chế ghi xuyên suốt chống giả mạo tới bộ lưu trữ bên ngoài |
| Quy trình làm việc định sẵn (Prescribed workflow) | "Kế hoạch từng bước từ nhà thiết kế con người" | Ràng buộc AAR; thường làm giảm hiệu năng so với việc tự do phân rã |
| Tự do phân rã (Free decomposition) | "Tác tử tự quyết định cách chia nhỏ tác vụ" | Năng lực cao hơn nhưng khó kiểm toán hơn |
| Ngưỡng AI R&D (AI R&D threshold) | "Cấp độ năng lực theo RSP/FSF" | Tự động hóa hoàn toàn đường ống R&D ở mức chi phí cạnh tranh |
| Nén thời gian (Compressed timeline) | "Cuộc đua giữa căn chỉnh và năng lực" | Nếu năng lực tăng nhanh hơn căn chỉnh, rủi ro mất căn chỉnh sẽ ngày càng lớn |

## Đọc thêm

- [Anthropic — Nhà nghiên cứu từ yếu sang mạnh tự động (Automated Weak-to-Strong Researcher)](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — nguồn tài liệu chính.
- [Chính sách Mở rộng có Trách nhiệm của Anthropic v3.0 (Anthropic Responsible Scaling Policy v3.0)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — định nghĩa ngưỡng AI R&D.
- [Anthropic — Đo lường quyền tự trị của tác tử AI (Measuring AI agent autonomy)](https://www.anthropic.com/research/measuring-agent-autonomy) — bối cảnh rộng hơn về quyền tự trị của tác tử.
- [Khuôn khổ An toàn Biên giới v3 của DeepMind (DeepMind Frontier Safety Framework v3)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — các cấp độ tự trị ML R&D song hành với RSP.
- [Burns et al. (2023). Khái quát hóa từ yếu sang mạnh (Weak-to-Strong Generalization - OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — bài toán nền tảng mà các AAR giải quyết.
