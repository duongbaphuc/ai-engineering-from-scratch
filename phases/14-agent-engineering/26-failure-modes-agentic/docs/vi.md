# Các Dạng Thất bại: Tại sao Tác tử bị Đổ vỡ (Failure Modes: Why Agents Break)

> MASFT (Berkeley, 2025) lập danh mục 14 dạng lỗi đa tác tử trong 3 danh mục lớn. Bản phân loại của Microsoft ghi nhận cách thức các lỗi AI hiện hữu bị khuếch đại trong môi trường mang tính tác tử. Dữ liệu thực tế từ ngành công nghiệp quy tụ về 5 dạng lỗi lặp lại phổ biến: hành động ảo giác, bành trướng phạm vi, lỗi dây chuyền, mất ngữ cảnh và lạm dụng công cụ.

**Loại bài học:** Tìm hiểu + Xây dựng (Learn + Build)  
**Ngôn ngữ:** Python (stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 05 (Tự tinh chỉnh và CRITIC - Self-Refine and CRITIC), Phase 14 · 24 (Khả năng quan sát - Observability)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Nêu tên ba nhóm phân loại lỗi của MASFT và ít nhất bốn dạng lỗi cụ thể trong từng nhóm.
- Giải thích tại sao hoạt động mang tính tác tử lại khuếch đại các dạng lỗi AI sẵn có (sự thiên vị, ảo giác).
- Mô tả năm dạng lỗi lặp lại phổ biến trong ngành và các biện pháp giảm thiểu tương ứng.
- Cài đặt một bộ phát hiện bằng thư viện chuẩn (stdlib) có khả năng gắn nhãn dấu vết thực thi (trace) của tác tử theo các dạng lỗi.

## Vấn đề Đặt ra

Các nhóm kỹ thuật thường triển khai những tác tử hoạt động ổn định trên 90% các dấu vết thực thi. Tuy nhiên, 10% thất bại còn lại không phải là nhiễu ngẫu nhiên — chúng rơi vào một số lượng nhỏ các danh mục lỗi mang tính lặp lại. Một khi bạn có thể định danh chính xác chúng, bạn có thể thiết lập giám sát và khắc phục triệt để.

## Khái niệm

### MASFT (Berkeley, arXiv:2503.13657)

Bản Phân loại Lỗi Hệ thống Đa Tác tử (Multi-Agent System Failure Taxonomy - MASFT). 14 dạng lỗi được phân cụm vào 3 danh mục. Hệ số đồng thuận giữa những người gán nhãn Cohen's Kappa đạt 0.88 — chứng minh các danh mục có thể phân biệt một cách đáng tin cậy.

Luận điểm trung tâm: các lỗi này là khiếm khuyết thiết kế căn bản trong kiến trúc hệ thống đa tác tử, không đơn thuần là giới hạn của mô hình ngôn ngữ lớn (LLM) có thể tự sửa được chỉ bằng các mô hình nền tảng tốt hơn.

### Bản phân loại các Dạng Thất bại trong Hệ thống AI Tác tử của Microsoft

- Các lỗi AI hiện có (thiên vị - bias, ảo giác - hallucination, rò rỉ dữ liệu - data leakage) bị khuếch đại mạnh mẽ trong môi trường tác tử.
- Các lỗi mới phát sinh từ tính tự trị (autonomy): hành động ngoài ý muốn ở quy mô lớn, lạm dụng công cụ (tool misuse), chệch hướng mục tiêu (mission drift).
- Báo cáo kỹ thuật này đóng vai trò như một sổ đăng ký rủi ro (risk register) cho các sản phẩm sử dụng tác tử.

### Đặc trưng hóa các Khiếm khuyết trong AI Tác tử (arXiv:2603.06847)

- Các lỗi phát sinh từ quá trình điều phối (orchestration), sự tiến hóa của trạng thái nội bộ và tương tác với môi trường bên ngoài.
- Không chỉ đơn thuần là "mã nguồn tồi" hay "đầu ra mô hình kém".

### Khảo sát Hiện tượng Ảo giác của Tác tử LLM (arXiv:2509.18970)

Hai biểu hiện chính:

1. **Lệch khỏi việc tuân thủ chỉ dẫn (Instruction-following Deviation)** — tác tử không tuân theo chỉ dẫn hệ thống (system prompt).
2. **Lạm dụng ngữ cảnh tầm dài (Long-range Contextual Misuse)** — tác tử quên hoặc áp dụng sai lệch ngữ cảnh từ các lượt hội thoại trước đó.

Các lỗi thuộc cấp độ tiểu ý định (sub-intention errors): Bỏ sót (Omission - bỏ quên một bước), Dư thừa (Redundancy - lặp lại một bước), Xáo trộn (Disorder - thực hiện các bước sai thứ tự).

### Năm dạng lỗi lặp lại phổ biến trong ngành

Các phân tích thực địa từ Arize, Galileo, NimbleBrain giai đoạn 2024-2026 hội tụ về:

1. **Hành động ảo giác (Hallucinated actions):** Tác tử gọi một công cụ không hề tồn tại hoặc bịa đặt đối số đầu vào.
2. **Bành trướng phạm vi (Scope creep):** Tác tử tự ý mở rộng nhiệm vụ vượt quá yêu cầu của người dùng (tự tạo thêm PR, gửi thêm email ngoài phạm vi).
3. **Lỗi dây chuyền (Cascading errors):** Một lệnh gọi sai lầm kích hoạt chuỗi tác động domino ở các bước tiếp theo. Ví dụ: ảo giác về một mã sản phẩm (SKU) ma dẫn tới 4 cuộc gọi API dây chuyền — gây ra sự cố đa hệ thống.
4. **Mất ngữ cảnh (Context loss):** Các nhiệm vụ dài hơi (long-horizon tasks) làm tác tử quên mất các ràng buộc được đưa ra ở những lượt đầu tiên.
5. **Lạm dụng công cụ (Tool misuse):** Gọi đúng công cụ nhưng truyền sai đối số, hoặc gọi hoàn toàn sai công cụ cần dùng.

Lỗi dây chuyền là "sát thủ" nguy hiểm nhất. Các tác tử không thể phân biệt giữa "tôi đã thất bại" và "nhiệm vụ này là bất khả thi", và thường có xu hướng bịa đặt một thông điệp thành công khi gặp lỗi 400 để cố gắng hoàn tất vòng lặp.

### Biện pháp giảm thiểu: Cổng kiểm soát tại từng bước

Thiết lập các cổng xác minh tự động (automated verification gates) tại từng bước trong chuỗi suy luận, kiểm tra tính xác thực đối chiếu với trạng thái thực tế của môi trường. Cụ thể:

- Bộ phân loại an toàn theo từng bước (Bài 21).
- Xác thực đối số khi gọi công cụ (Bài 06).
- Kiểm tra chéo nội dung truy xuất với các sự thật đã biết (Bài 05, CRITIC).
- Phát hiện ảo giác thành công (success hallucination) bằng cách thăm dò lại trạng thái thực (tệp đó đã thực sự được tạo chưa?).

### Những cạm bẫy khi giám sát lỗi

- **Chỉ gắn nhãn các sự cố sập chương trình (crash):** Đa phần các lỗi của tác tử vẫn tạo ra đầu ra có vẻ hợp lệ. Cần có các kiểm tra ở cấp độ nội dung.
- **Không có mốc chuẩn (baseline):** Việc phát hiện trôi dạt (drift detection) cần có trạng thái tốt gần nhất (last-known-good); nếu thiếu mốc này bạn không thể khẳng định "tình trạng đang tồi tệ đi".
- **Cảnh báo quá mức (Over-alerting):** Mọi lỗi nhỏ đều kích hoạt thông báo khẩn. Cần gom cụm và giới hạn tần suất cảnh báo (rate-limit).

```figure
failure-cascade
```

## Tự Xây dựng (Build It)

`code/main.py` cài đặt một bộ gắn nhãn dạng lỗi bằng thư viện chuẩn (stdlib):

- Một tập dữ liệu dấu vết thực thi tổng hợp bao gồm cả năm dạng lỗi.
- Các hàm phát hiện theo từng dạng lỗi (nhận diện mẫu đặc trưng trên lệnh gọi công cụ, đầu ra, hành động lặp lại).
- Một bộ gắn nhãn phân loại từng dấu vết và xuất báo cáo phân bổ dạng lỗi.

Chạy thử mã nguồn:

```bash
python3 code/main.py
```

Kết quả: nhãn chi tiết cho từng dấu vết + phân bổ tổng hợp, mô phỏng lại một cách tinh gọn những gì tính năng phân cụm dấu vết thực thi của Phoenix mang lại.

## Ứng dụng Thực tế (Use It)

- **Phoenix** cho việc phân cụm trôi dạt trên môi trường sản xuất (Bài 24).
- **Langfuse** phục vụ việc phát lại phiên tương tác (session replay) + gắn nhãn chú thích.
- **Tùy chỉnh (Custom)** cho các mẫu đặc trưng mang tính đặc thù miền mà nền tảng giám sát thông thường không thể nhận biết.

## Chuyển giao Đóng gói (Ship It)

`outputs/skill-failure-detector.md` tạo ra các bộ phát hiện dạng lỗi được tinh chỉnh riêng cho miền nghiệp vụ của bạn, được kết nối trực tiếp vào kho lưu trữ dấu vết thực thi.

## Bài tập Thực hành

1. Thêm một bộ phát hiện cho lỗi "ảo giác thành công" (success hallucination): tác tử báo cáo thành công nhưng trạng thái mục tiêu không hề thay đổi.
2. Gắn nhãn 100 dấu vết thực thi thực tế từ một sản phẩm bạn đã xây dựng. Dạng lỗi nào chiếm ưu thế? Chi phí để khắc phục nó là bao nhiêu?
3. Cài đặt chỉ số "bán kính dây chuyền" (cascade radius): nếu phát sinh lỗi tại bước N, lỗi đó sẽ ảnh hưởng đến bao nhiêu bước kế tiếp phía sau?
4. Đọc 14 dạng lỗi của MASFT. Chọn ra ba dạng áp dụng trực tiếp cho sản phẩm của bạn và viết bộ phát hiện cho chúng.
5. Đưa một bộ phát hiện vào quy trình CI: đánh trượt bản dựng (fail build) nếu có >=5% số dấu vết bị gắn nhãn lỗi đó.

## Thuật ngữ Then chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|---|---|---|
| MASFT | "Phân loại lỗi đa tác tử" | Hệ thống phân loại 14 dạng lỗi của Berkeley |
| Lỗi dây chuyền (Cascading error) | "Lỗi gợn sóng" | Một sai sót ban đầu lan truyền qua N bước tiếp theo |
| Mất ngữ cảnh (Context loss) | "Quên mất ràng buộc" | Lượt tương tác dài hạn làm rơi rụng các thông tin ở lượt đầu |
| Lạm dụng công cụ (Tool misuse) | "Sai công cụ / sai đối số" | Lệnh gọi hợp lệ nhưng truyền sai ngữ cảnh hoặc sai tham số |
| Ảo giác thành công (Success hallucination) | "Báo cáo hoàn thành giả" | Tác tử tuyên bố thành công khi gặp lỗi 400; trạng thái giữ nguyên |
| Bành trướng phạm vi (Scope creep) | "Làm quá quyền hạn" | Tác tử thực hiện nhiều hơn những gì được yêu cầu |
| Lệch tuân thủ chỉ dẫn (Instruction-following deviation) | "Không nghe lời" | Phớt lờ chỉ dẫn hệ thống hoặc ràng buộc của người dùng |
| Lỗi tiểu ý định (Sub-intention errors) | "Lỗi kế hoạch" | Bỏ sót, lặp lại dư thừa hoặc đảo lộn thứ tự khi thực thi kế hoạch |

## Đọc thêm

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 dạng lỗi, 3 nhóm phân loại
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — sổ đăng ký rủi ro hệ thống
- [Arize Phoenix](https://docs.arize.com/phoenix) — phân cụm trôi dạt trong thực tế
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — thời điểm các mẫu đơn giản hóa giúp tránh hoàn toàn các dạng lỗi
