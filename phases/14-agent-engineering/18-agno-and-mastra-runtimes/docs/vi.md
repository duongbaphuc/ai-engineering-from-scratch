# Môi trường Thực thi Tác tử Môi trường Sản xuất — Khởi tạo Nhanh và Quy trình Định kiểu

> Một môi trường thực thi tác tử (production agent runtime) tối ưu hóa chính xác những gì mà các framework tạo mẫu nhanh thường bỏ qua: chi phí khởi tạo, các bề mặt quy trình được định kiểu, và backend sẵn sàng phục vụ. Bộ đôi tiêu biểu năm 2026: Agno (Python) hướng tới khả năng khởi tạo tác tử ở mức micro-giây cùng các backend FastAPI phi trạng thái. Mastra cung cấp tác tử, công cụ, quy trình, bộ định tuyến mô hình hợp nhất và lưu trữ hỗn hợp trên nền tảng Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Mục tiêu học tập

- Xác định các mục tiêu hiệu năng của Agno và thời điểm chúng thực sự có ý nghĩa.
- Kể tên ba nguyên thủy cốt lõi của Mastra — Agents (Tác tử), Tools (Công cụ), Workflows (Quy trình) — cùng các bộ điều hợp máy chủ (server adapters) được hỗ trợ.
- Giải thích lý do tại sao backend FastAPI phi trạng thái theo phạm vi phiên (stateless session-scoped FastAPI backend) lại là lộ trình production được khuyến nghị của Agno.
- Lựa chọn giữa Agno và Mastra cho một ngăn xếp công nghệ cụ thể (ưu tiên Python so với ưu tiên TypeScript).

## Vấn đề

LangGraph, AutoGen, CrewAI là những framework khá nặng nề. Các đội ngũ chỉ muốn "vòng lặp tác tử thuần túy, tốc độ cao, nằm ngay trong runtime của tôi" thường tìm đến Agno (Python) hoặc Mastra (TypeScript). Cả hai đều chấp nhận đánh đổi một số nguyên thủy do framework kiểm soát để đổi lấy tốc độ thô và sự tương thích chặt chẽ hơn với ngăn xếp công nghệ xung quanh.

## Khái niệm

### Agno

- Môi trường thực thi Python, trước đây có tên là Phi-data.
- Triết lý: "Không đồ thị, không chuỗi phức tạp hay các mẫu hình rườm rà — chỉ là Python thuần túy."
- Các mục tiêu hiệu năng theo công bố trong tài liệu: thời gian khởi tạo tác tử ~2μs, bộ nhớ ~3.75 KiB cho mỗi tác tử, hỗ trợ ~23 nhà cung cấp mô hình.
- Lộ trình triển khai production: backend FastAPI phi trạng thái theo phạm vi phiên. Mỗi yêu cầu HTTP sẽ khởi tạo một tác tử mới tinh; trạng thái phiên được lưu trữ trong cơ sở dữ liệu.
- Đa phương thức tự nhiên (văn bản, hình ảnh, âm thanh, video, tệp tin) và Agentic RAG tích hợp sẵn.

Các mục tiêu về tốc độ sẽ phát huy giá trị lớn khi bạn có hàng nghìn tác tử tồn tại ngắn hạn mỗi giây (hàng đợi hội thoại tập trung, các đường ống đánh giá mô hình). Chúng ít ý nghĩa hơn khi một tác tử duy nhất chạy liên tục trong 10 phút.

### Mastra

- Nền tảng TypeScript, được xây dựng trên Vercel AI SDK.
- Ba nguyên thủy cốt lõi: **Agents** (Tác tử), **Tools** (Công cụ được định kiểu bằng Zod), **Workflows** (Quy trình làm việc).
- Bộ định tuyến mô hình hợp nhất (Unified Model Router) — hơn 3.300 mô hình xuyên suốt 94 nhà cung cấp (tháng 3/2026).
- Lưu trữ hỗn hợp (Composite storage): bộ nhớ, quy trình và khả năng quan sát được lưu trữ trên các backend riêng biệt; ClickHouse được khuyến nghị cho việc giám sát quan sát ở quy mô lớn.
- Giấy phép Apache 2.0 với các thư mục `ee/` thuộc giấy phép thương mại mở mã nguồn (source-available enterprise license).
- Các bộ chuyển đổi máy chủ (server adapters) cho Express, Hono, Fastify, Koa; tích hợp hạng nhất với Next.js và Astro.
- Đi kèm giao diện Mastra Studio (localhost:4111) để gỡ lỗi và kiểm tra tác tử.
- Hơn 22.000 sao trên GitHub, hơn 300.000 lượt tải npm hàng tuần tại thời điểm ra mắt bản 1.0 (tháng 1/2026).

### Định vị

Cả hai đều không cố gắng trở thành một LangGraph thứ hai. Chúng cạnh tranh dựa trên:

- **Sự tương thích ngôn ngữ.** Agno dành cho các đội ngũ ưu tiên Python; Mastra dành cho các đội ngũ ưu tiên TypeScript.
- **Tính công thái học của môi trường thực thi (Runtime ergonomics).** Agno = chi phí phụ trợ gần như bằng không; Mastra = tích hợp sâu vào hệ sinh thái Vercel.
- **Khả năng quan sát (Observability).** Cả hai đều tích hợp tốt với Langfuse/Phoenix/Opik (Bài 24), nhưng Mastra Studio là công cụ chính chủ tích hợp sẵn.

### Khi nào nên chọn từng nền tảng

- **Agno** — Backend Python, nhiều tác tử tồn tại ngắn hạn, yêu cầu khắt khe về hiệu năng, hệ thống sử dụng FastAPI.
- **Mastra** — Backend TypeScript, triển khai trên Next.js / Vercel, cần định tuyến mô hình hợp nhất đa nhà cung cấp, công cụ định kiểu chặt chẽ bằng Zod.
- **LangGraph** (Bài 13) — khi trạng thái bền vững và khả năng suy luận logic dựa trên đồ thị tường minh quan trọng hơn tốc độ thô.
- **OpenAI / Claude Agent SDK** — khi bạn muốn hình thái sản phẩm tiêu chuẩn từ chính nhà cung cấp mô hình (Bài 16–17).

### Những cạm bẫy thường gặp

- **Tối ưu hiệu năng vì hình thức (Perf-for-perf's-sake).** Chọn Agno chỉ vì con số "2μs" nghe hấp dẫn, trong khi khối lượng công việc thực tế chỉ là một lệnh gọi tác tử chậm chạp trên mỗi yêu cầu. Khi đó, chi phí khung điều hành không phải là điểm nghẽn cổ chai.
- **Bị khóa chặt vào hệ sinh thái (Ecosystem lock-in).** Khả năng tích hợp mang phong cách Vercel của Mastra là điểm cộng lớn trên Vercel, nhưng lại là điểm trừ khi triển khai ở nơi khác.
- **Nhầm lẫn về giấy phép doanh nghiệp.** Các thư mục `ee/` của Mastra là dạng mã nguồn mở có điều kiện (source-available), không phải Apache 2.0. Hãy đọc kỹ giấy phép nếu bạn có ý định fork dự án.

```figure
wb-runtime-spawn
```

## Xây dựng

Bài học này chủ yếu mang tính so sánh — không có một đoạn mã đơn lẻ nào có thể phản ánh đầy đủ cả hai framework. Hãy xem `code/main.py` để quan sát một bản minh họa đối chiếu: quy trình tối giản gồm "chạy tác tử, truyền luồng đầu ra, lưu bền vững phiên" được triển khai hai lần (một lần theo hình thái Agno, một lần theo hình thái Mastra).

Chạy thử:

```bash
python3 code/main.py
```

Hai dấu vết có cấu trúc khác biệt nhưng tương đương về mặt chức năng.

## Sử dụng

- **Agno** — Backend Python cần tốc độ tối đa và phong cách FastAPI.
- **Mastra** — Backend TypeScript kết nối nhiều nhà cung cấp và các nguyên thủy quy trình làm việc.
- Cả hai đều cung cấp sẵn các hook quan sát chính chủ và tích hợp mượt mà với Langfuse.

## Chuyển giao

`outputs/skill-runtime-picker.md` hỗ trợ chọn Agno, Mastra, LangGraph hoặc SDK nhà cung cấp dựa trên ngăn xếp kỹ thuật, ngân sách độ trễ và mô hình vận hành.

## Bài tập

1. Đọc tài liệu Agno. Chuyển đổi vòng lặp ReAct thư viện chuẩn (Bài 01) sang Agno. Những phần nào đã biến mất? Những phần nào được giữ lại?
2. Đọc tài liệu Mastra. Chuyển đổi cùng vòng lặp đó sang Mastra. Việc định kiểu công cụ thay đổi thế nào (Zod so với không định kiểu)?
3. Đo điểm chuẩn (Benchmark): đo lường độ trễ khởi tạo tác tử trên ngăn xếp của bạn. Mục tiêu 2μs của Agno có thực sự tạo ra khác biệt cho khối lượng công việc của bạn không?
4. Thiết kế kế hoạch di chuyển: nếu bạn đang chạy CrewAI bằng Python, điều gì sẽ bị phá vỡ nếu chuyển sang Agno?
5. Đọc điều khoản giấy phép `ee/` của Mastra. Những hạn chế nào sẽ ảnh hưởng nếu tạo một bản fork mã nguồn mở?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Agno | "Tác tử Python siêu tốc" | Môi trường thực thi tác tử phi trạng thái theo phạm vi phiên |
| Mastra | "Tác tử TypeScript trên Vercel AI SDK" | Agents + Tools + Workflows + Bộ định tuyến mô hình |
| Bộ định tuyến mô hình hợp nhất (Unified Model Router) | "Truy cập đa nhà cung cấp" | Một client duy nhất cho hơn 3.300 mô hình qua 94 nhà cung cấp |
| Lưu trữ hỗn hợp (Composite storage) | "Nhiều backend lưu trữ" | Bộ nhớ / quy trình / khả năng quan sát lưu vào các kho riêng biệt |
| Mastra Studio | "Trình gỡ lỗi cục bộ" | Giao diện localhost:4111 để quan sát nội bộ tác tử |
| Mã nguồn mở có điều kiện (Source-available) | "Không hoàn toàn mã nguồn mở (OSS)" | Giấy phép cho phép đọc mã nguồn nhưng hạn chế sử dụng thương mại |

## Tài liệu đọc thêm

- [Tài liệu Agno Agent Framework](https://www.agno.com/agent-framework) — mục tiêu hiệu năng, tích hợp FastAPI
- [Tài liệu Mastra](https://mastra.ai/docs) — các nguyên thủy, bộ điều hợp máy chủ, Model Router
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — giải pháp thay thế dựa trên đồ thị trạng thái
- [Comet Opik](https://www.comet.com/site/products/opik/) — so sánh khả năng quan sát được viện dẫn trong các tích hợp của Mastra
