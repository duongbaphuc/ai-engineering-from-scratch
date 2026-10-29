# Khối Bộ nhớ và Tính toán trong Thời gian Ngủ

> Các khối bộ nhớ chức năng rời rạc mà mô hình có thể trực tiếp chỉnh sửa, kết hợp với một tác tử hoạt động trong thời gian ngủ (sleep-time agent) giúp hợp nhất bộ nhớ bất đồng bộ khi tác tử chính đang rảnh rỗi. Hai ý tưởng này chính là chìa khóa để mở rộng quy mô bộ nhớ vượt qua giới hạn của một cuộc trò chuyện đơn lẻ.

**Loại bài học:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn - stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 07 (MemGPT)  
**Thời lượng:** ~75 phút

## Mục tiêu Học tập

- Nêu tên ba tầng bộ nhớ mà Letta sử dụng (cốt lõi - core, gợi nhớ - recall, lưu trữ - archival) và vai trò của từng tầng.
- Giải thích mô hình khối bộ nhớ (memory-block pattern): khối Con người (Human block), khối Nhân cách (Persona block), và các khối do người dùng định nghĩa dưới dạng các đối tượng có định kiểu hạng nhất (first-class typed objects).
- Mô tả tính toán trong thời gian ngủ (sleep-time compute) là gì, tại sao nó nằm ngoài đường truyền tới hạn (critical path), và tại sao nó có thể sử dụng một mô hình mạnh hơn tác tử chính.
- Hiện thực hóa vòng lặp hai tác tử theo kịch bản mẫu, trong đó tác tử chính phục vụ phản hồi và tác tử thời gian ngủ hợp nhất các khối bộ nhớ giữa các lượt đối thoại.

## Vấn đề

MemGPT (Bài học 07) đã giải quyết luồng điều khiển bộ nhớ ảo. Tuy nhiên, ba vấn đề lớn trong môi trường sản xuất đã nảy sinh:

1. **Độ trễ (Latency).** Mọi thao tác bộ nhớ đều nằm trên đường truyền tới hạn. Nếu tác tử phải cắt tỉa, tóm tắt hoặc đối soát dữ liệu trong lúc người dùng đang chờ đợi, độ trễ đuôi (tail latency) sẽ tăng vọt.
2. **Thối rữa bộ nhớ (Memory rot).** Các thao tác ghi tích lũy liên tục. Những dữ kiện mâu thuẫn vẫn tồn tại. Việc truy xuất bị chìm ngập trong nội dung lỗi thời.
3. **Mất cấu trúc (Structure loss).** Một kho lưu trữ phẳng không thể biểu đạt rõ ràng rằng: "khối Con người luôn phải có mặt trong prompt; khối Nhân cách luôn phải có mặt trong prompt; khối Tác vụ sẽ được hoán đổi theo từng phiên."

Letta (letta.com) là tên nền tảng mà dự án MemGPT ban đầu đã áp dụng vào năm 2024 — mô hình thiết kế của bài báo vẫn giữ tên MemGPT — và bản tái thiết kế Letta V1 năm 2026 là một bước tiến triển riêng biệt sau đó. Các khối bộ nhớ giúp cấu trúc trở nên tường minh; tính toán trong thời gian ngủ chuyển việc hợp nhất dữ liệu ra khỏi đường truyền tới hạn.

## Khái niệm

### Ba tầng lưu trữ (Three tiers)

| Tầng | Phạm vi | Nơi lưu trữ | Được ghi bởi |
|---|---|---|---|
| Cốt lõi (Core) | Luôn hiển thị | Bên trong prompt chính | Lệnh gọi công cụ của tác tử + Tác vụ ghi đè thời gian ngủ |
| Gợi nhớ (Recall) | Lịch sử hội thoại | Có thể truy xuất | Tự động ghi nhật ký theo từng lượt đối thoại |
| Lưu trữ (Archival) | Dữ kiện tùy ý | Vector + Khóa-Giá trị (KV) + Đồ thị | Lệnh gọi công cụ của tác tử + Nạp dữ liệu thời gian ngủ |

Tầng Cốt lõi (Core) tương ứng với phần core của MemGPT. Tầng Gợi nhớ (Recall) là bộ đệm hội thoại cùng phần đuôi đã bị loại bỏ khỏi cửa sổ. Tầng Lưu trữ (Archival) là kho lưu trữ ngoài. Sự phân chia này giúp khắc phục tình trạng quá tải chức năng của mô hình hai tầng trong MemGPT.

### Các khối bộ nhớ (Memory blocks)

Một khối bộ nhớ là một phần có định kiểu, bền vững và có thể chỉnh sửa được thuộc tầng cốt lõi. Bài báo MemGPT gốc đã định nghĩa hai khối:

- **Khối Con người (Human block)** — các dữ kiện về người dùng (tên, vai trò, sở thích, mục tiêu).
- **Khối Nhân cách (Persona block)** — khái niệm tự nhận thức của tác tử (bản sắc, giọng văn, các ràng buộc).

Letta tổng quát hóa thành các khối tùy ý do người dùng định nghĩa: khối `Task` dành cho mục tiêu hiện tại, khối `Project` dành cho các dữ kiện về cơ sở mã nguồn, khối `Safety` dành cho các ràng buộc an toàn nghiêm ngặt. Mỗi khối bao gồm `id`, `label` (nhãn), `value` (giá trị), `limit` (giới hạn ký tự), `description` (mô tả để mô hình biết khi nào cần chỉnh sửa nó).

Các khối có thể chỉnh sửa thông qua bề mặt công cụ:

- `block_append(label, text)` — nối tiếp nội dung vào khối.
- `block_replace(label, old, new)` — thay thế chuỗi ký tự trong khối.
- `block_read(label)` — đọc nội dung khối.
- `block_summarize(label)` — cô đọng một khối khi sắp chạm ngưỡng giới hạn.

### Tính toán trong thời gian ngủ (Sleep-time compute)

Phần bổ sung năm 2025 của Letta: chạy một tác tử thứ hai ở chế độ nền, tách biệt hoàn toàn khỏi đường truyền tới hạn. Các tác tử thời gian ngủ xử lý các bản ghi hội thoại và ngữ cảnh cơ sở mã, ghi `learned_context` (ngữ cảnh đã học) vào các khối dùng chung, đồng thời hợp nhất hoặc vô hiệu hóa các bản ghi lưu trữ lỗi thời.

Các đặc tính ưu việt thu được:

- **Không tốn chi phí độ trễ.** Phản hồi của tác tử chính không phải chờ đợi các thao tác bộ nhớ.
- **Cho phép dùng mô hình mạnh hơn.** Tác tử thời gian ngủ có thể là một mô hình đắt tiền hơn, chạy chậm hơn vì nó không bị ràng buộc bởi độ trễ tương tác trực tiếp.
- **Cửa sổ hợp nhất tự nhiên.** Khử trùng lặp, tóm tắt, vô hiệu hóa các dữ kiện mâu thuẫn vào thời điểm người dùng không phải chờ đợi.

Cơ chế này mô phỏng rất khớp với cách con người hoạt động: bạn thực hiện công việc, bạn ngủ một giấc, và bộ nhớ dài hạn sẽ tự lắng đọng và sắp xếp lại qua đêm.

### Suy luận bản địa (Native reasoning)

Letta V1 (`letta_v1_agent`, 2026) loại bỏ mô hình `send_message`/nhịp tim (heartbeat) và các token `Thought:` dạng nội tuyến, chuyển sang sử dụng cơ chế suy luận bản địa. API Responses (OpenAI) và Messages API tích hợp tư duy mở rộng (extended thinking của Anthropic) phát ra luồng suy luận trên một kênh riêng biệt, được truyền qua các lượt đối thoại (được mã hóa giữa các nhà cung cấp trong môi trường sản xuất). Vòng lặp điều khiển vẫn là ReAct. Tuy nhiên, vết tư duy lúc này mang tính cấu trúc thay vì được định dạng bên trong prompt.

### Những điểm mô hình này dễ gặp sự cố

- **Phình to khối bộ nhớ (Block bloat).** Việc gọi `block_append` vô tội vạ sẽ nhanh chóng chạm ngưỡng giới hạn. Hãy thiết lập một bộ tóm tắt khối trước khi thao tác ghi đẩy kích thước vượt quá trần cho phép.
- **Trôi dạt âm thầm (Silent drift).** Tác tử thời gian ngủ viết lại một khối nhưng tác tử chính không hề nhận biết được sự thay đổi. Hãy đánh phiên bản (versioning) cho các khối và hiển thị các bản so sánh thay đổi (diff) trong vết thực thi.
- **Hợp nhất bị đầu độc (Poisoned consolidation).** Tác tử thời gian ngủ xử lý các nội dung mà kẻ tấn công có thể tiếp cận rồi ghi vào phần cốt lõi. Các nguyên lý phòng vệ trong Bài học 27 cũng áp dụng cho cả bề mặt tác tử thời gian ngủ.

```figure
memory-blocks
```

## Xây dựng Thực tế

`code/main.py` hiện thực hóa:

- `Block` — id, label, value, limit, description.
- `BlockStore` — các thao tác CRUD + hàm trợ giúp `near_limit(label)`.
- Hai tác tử theo kịch bản — `PrimaryAgent` phục vụ lượt đối thoại, `SleepTimeAgent` hợp nhất dữ liệu giữa các lượt.
- Vết thực thi minh họa cuộc hội thoại 3 lượt với các thao tác ghi khối, cùng một lượt xử lý thời gian ngủ giúp tóm tắt khối và vô hiệu hóa dữ kiện cũ.

Chạy chương trình:

```bash
python3 code/main.py
```

Bản ghi cho thấy sự phân tách rõ rệt: các lượt của tác tử chính diễn ra nhanh chóng và tạo ra các thao tác ghi thô; lượt ngủ sau đó tiến hành cô đọng và dọn dẹp dữ liệu.

## Ứng dụng Thực tế

- **Letta** (letta.com) dành cho việc triển khai tham chiếu. Hỗ trợ tự lưu trữ (self-host) hoặc đám mây quản lý (managed cloud).
- **Các kỹ năng trong Claude Agent SDK** hoạt động như tri thức có dạng khối — một kỹ năng là một khối hướng dẫn được đặt tên, đánh phiên bản và có thể truy xuất mà tác tử tải về theo yêu cầu.
- **Các hệ thống tùy chỉnh** dành cho các đội ngũ muốn kiểm soát sâu hạ tầng lưu trữ. Hãy tuân thủ giao ước API của Letta để có thể dễ dàng di chuyển hệ thống sau này.

## Đóng gói Sản phẩm

`outputs/skill-memory-blocks.md` tạo ra một hệ thống khối theo cấu trúc của Letta tích hợp các điểm nối (hooks) thời gian ngủ cho bất kỳ môi trường thực thi nào, bao gồm các quy tắc an toàn và liên kết trích dẫn nguồn.

## Bài tập

1. Thêm công cụ `block_summarize` thay thế giá trị của khối bằng bản tóm tắt do mô hình sinh ra khi `near_limit` trả về true. Ngưỡng kích hoạt nào giúp giảm thiểu cả số lần gọi tóm tắt lẫn sự cố tràn giới hạn khối?
2. Hiện thực hóa cơ chế khử trùng lặp trong thời gian ngủ trên kho lưu trữ: hai bản ghi có độ trùng lặp token >90% sẽ được gộp làm một. Chỉ thực hiện điều này trong lượt ngủ, tuyệt đối không chạy trên đường truyền tới hạn.
3. Đánh phiên bản cho các khối bộ nhớ. Với mỗi thao tác ghi, hãy ghi lại giá trị cũ và một bản so sánh thay đổi (diff). Mở công cụ `block_history(label)` để người vận hành có thể gỡ lỗi "tại sao tác tử lại quên mất thông tin X".
4. Đối xử với các tác tử thời gian ngủ như các tác nhân ghi không đáng tin cậy. Khi chúng can thiệp vào khối Persona hoặc Safety, yêu cầu phải có một tác tử thứ hai xét duyệt trước khi xác nhận lưu (commit).
5. Chuyển đổi mã ví dụ để sử dụng Letta API (`letta_v1_agent`). Những điểm gì thay đổi trong cấu trúc schema của khối, và suy luận bản địa làm thay đổi hình dạng của vết thực thi như thế nào?

## Thuật ngữ Then chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Khối bộ nhớ (Memory block) | "Phần prompt có thể sửa" | Phân đoạn có định kiểu, bền vững, mô hình LLM có thể chỉnh sửa của bộ nhớ cốt lõi |
| Khối Con người (Human block) | "Bộ nhớ về người dùng" | Các dữ kiện về người dùng, được ghim cố định trong tầng cốt lõi |
| Khối Nhân cách (Persona block) | "Bản sắc của tác tử" | Khái niệm tự nhận thức, giọng điệu, các ràng buộc, được ghim trong tầng cốt lõi |
| Tính toán thời gian ngủ (Sleep-time compute) | "Tác vụ bộ nhớ bất đồng bộ" | Tác tử thứ hai thực hiện hợp nhất dữ liệu ngoài đường truyền tới hạn |
| Cốt lõi / Gợi nhớ / Lưu trữ (Core / Recall / Archival) | "Các tầng lưu trữ" | Phân chia bộ nhớ ba lớp: luôn hiển thị / lịch sử hội thoại / kho ngoài |
| Giới hạn khối (Block limit) | "Ngưỡng trần" | Giới hạn ký tự trên mỗi khối; kích hoạt hành vi tóm tắt khi sắp chạm ngưỡng |
| Suy luận bản địa (Native reasoning) | "Kênh tư duy" | Đầu ra suy luận ở cấp độ nhà cung cấp mô hình, không phải dạng `Thought:` ở cấp độ prompt |
| Ngữ cảnh đã học (Learned context) | "Đầu ra sau giấc ngủ" | Các dữ kiện mà tác tử thời gian ngủ ghi vào các khối dùng chung |

## Tài liệu Tham khảo

- [Letta, Blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — Mô hình thiết kế khối bộ nhớ
- [Letta, Blog Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — Hợp nhất bất đồng bộ
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — Tái cấu trúc vòng lặp với suy luận bản địa
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — Nguồn gốc khởi xướng
