# Sử dụng công cụ và Gọi hàm

> Toolformer (Schick và cộng sự, 2023) đã mở đầu cho việc tự động gán nhãn công cụ theo phương pháp học tự giám sát. Bảng xếp hạng gọi hàm Berkeley V4 (Patil và cộng sự, 2025) thiết lập tiêu chuẩn năm 2026: 40% tác tử, 30% đa lượt, 10% trực tiếp, 10% phi trực tiếp, 10% ảo giác. Gọi hàm đơn lượt đã được giải quyết trọn vẹn. Bộ nhớ, việc ra quyết định động và các chuỗi công cụ dài hạn thì vẫn chưa.

**Loại bài học:** Xây dựng (Build)
**Ngôn ngữ:** Python (stdlib)
**Điều kiện tiên quyết:** Phase 14 · 01 (Vòng lặp tác tử), Phase 13 · 01 (Đi sâu vào gọi hàm)
**Thời gian:** ~60 phút

## Mục tiêu học tập

- Giải thích tín hiệu huấn luyện tự giám sát của Toolformer: chỉ giữ lại các chú thích công cụ khi việc thực thi giúp giảm hàm mất mát của token tiếp theo (next-token loss).
- Nêu tên năm danh mục đánh giá của BFCL V4 và ý nghĩa đo lường của từng danh mục.
- Triển khai một sổ đăng ký công cụ (tool registry) bằng thư viện chuẩn (stdlib) với khả năng xác thực lược đồ, ép kiểu đối số và đóng hộp cát khi thực thi (execution sandboxing).
- Chẩn đoán ba bài toán mở của năm 2026: liên kết chuỗi công cụ dài hạn, ra quyết định động và quản lý bộ nhớ.

## Vấn đề thực tế

Giai đoạn đầu của việc sử dụng công cụ chỉ đặt câu hỏi: liệu mô hình có thể dự đoán chính xác một lệnh gọi hàm hay không? Việc sử dụng công cụ hiện đại lại đặt ra bài toán: liệu mô hình có thể kết nối chuỗi công cụ qua 40 bước, có bộ nhớ, trong điều kiện quan sát từng phần (partial observability), phục hồi sau các lỗi công cụ, mà không bị ảo giác sinh ra những công cụ không hề tồn tại?

Toolformer đã thiết lập nền tảng cơ sở: các mô hình có thể tự học thời điểm cần gọi công cụ thông qua tự giám sát. BFCL V4 định hình mục tiêu đánh giá cho năm 2026. Khoảng cách giữa hai cột mốc này chính là không gian mà các tác tử trong môi trường sản xuất đang vận hành.

## Khái niệm cốt lõi

### Toolformer (Schick và cộng sự, NeurIPS 2023)

Ý tưởng cốt lõi: để mô hình tự gán nhãn tập ngữ liệu tiền huấn luyện của chính nó bằng các lệnh gọi API ứng viên. Với mỗi ứng viên, hãy thực thi lệnh gọi đó. Chỉ giữ lại chú thích nếu việc đưa kết quả công cụ vào ngữ cảnh giúp giảm tổn thất (loss) khi dự đoán token tiếp theo. Sau đó, tinh chỉnh mô hình trên tập ngữ liệu đã được lọc sạch này.

Các công cụ được thử nghiệm: máy tính, hệ thống hỏi đáp, công cụ tìm kiếm, dịch máy, lịch. Tín hiệu tự giám sát hoàn toàn dựa trên việc liệu công cụ có giúp dự đoán văn bản tốt hơn hay không — không cần bất kỳ nhãn gán thủ công nào của con người.

Kết quả về quy mô: khả năng dùng công cụ xuất hiện theo quy mô mô hình (emerges at scale). Các mô hình nhỏ hơn bị suy giảm hiệu năng do các chú thích công cụ; các mô hình lớn hơn lại hưởng lợi rõ rệt. Đây là lý do tại sao các mô hình tiên tiến năm 2026 đều được tích hợp sẵn khả năng dùng công cụ rất mạnh mẽ, trong khi hầu hết các mô hình 7B cần phải được tinh chỉnh tường minh cho việc dùng công cụ thì mới có thể hoạt động đáng tin cậy.

### Bảng xếp hạng gọi hàm Berkeley V4 (Patil và cộng sự, ICML 2025)

BFCL (Berkeley Function Calling Leaderboard) là tiêu chuẩn đánh giá thực tế của năm 2026. Cơ cấu của V4:

- **Tính tác tử (Agentic - 40%)** — các quỹ đạo tác tử hoàn chỉnh: bộ nhớ, đa lượt, ra quyết định động.
- **Đa lượt (Multi-Turn - 30%)** — các cuộc hội thoại tương tác với chuỗi gọi công cụ.
- **Trực tiếp (Live - 10%)** — các prompt thực tế do người dùng gửi lên (phân phối dữ liệu phức tạp hơn).
- **Phi trực tiếp (Non-Live - 10%)** — các trường hợp kiểm thử tổng hợp.
- **Ảo giác (Hallucination - 10%)** — phát hiện thời điểm không nên gọi bất kỳ công cụ nào.

Phiên bản V3 đã giới thiệu cơ chế đánh giá dựa trên trạng thái (state-based evaluation): sau một chuỗi công cụ, kiểm tra trạng thái thực tế của API (ví dụ: "tệp đã thực sự được tạo chưa?") thay vì chỉ so khớp cây cú pháp trừu tượng (AST) của các lệnh gọi công cụ. Phiên bản V4 bổ sung thêm tìm kiếm web, bộ nhớ và các danh mục đánh giá độ nhạy cảm về định dạng.

Phát hiện then chốt năm 2026: gọi hàm đơn lượt gần như đã được giải quyết trọn vẹn. Các lỗi chủ yếu tập trung vào bộ nhớ (mang theo ngữ cảnh qua nhiều lượt), ra quyết định động (chọn công cụ tiếp theo dựa trên kết quả trước đó), chuỗi tác vụ dài hạn (bị trôi dạt ngữ cảnh sau 20+ bước), và phát hiện ảo giác (từ chối gọi khi không có công cụ nào phù hợp).

### Lược đồ công cụ (Tool schema)

Mỗi nhà cung cấp đều có một lược đồ riêng. Chúng khác nhau về chi tiết nhưng chia sẻ chung một hình thái cấu trúc:

```text
name: string
description: string (chức năng, thời điểm sử dụng)
input_schema: JSON Schema (các thuộc tính, bắt buộc, kiểu dữ liệu, enums)
```

Anthropic sử dụng `input_schema` trực tiếp. OpenAI sử dụng `function.parameters`. Cả hai đều chấp nhận JSON Schema. Phần mô tả (description) đóng vai trò sống còn — mô hình đọc phần này để chọn công cụ phù hợp. Mô tả công cụ tồi là nguyên nhân gốc rễ số 1 dẫn đến việc chọn sai công cụ.

### Xác thực đối số (Argument validation)

Không bao giờ tin tưởng tuyệt đối vào lệnh gọi công cụ. Hãy luôn xác thực:

1. **Ép kiểu dữ liệu (Type coercion).** Mô hình có thể trả về một chuỗi `"5"` trong khi lược đồ yêu cầu kiểu số nguyên (int). Hãy ép kiểu nếu rõ ràng; từ chối nếu mơ hồ.
2. **Xác thực Enum.** Nếu lược đồ quy định `status in {"open", "closed"}` mà mô hình phát ra `"in_progress"`, hãy từ chối kèm theo thông báo lỗi mang tính mô tả chi tiết.
3. **Các trường bắt buộc (Required fields).** Thiếu trường bắt buộc -> lập tức phản hồi một quan sát lỗi cho mô hình, chứ không làm sập ứng dụng.
4. **Xác thực định dạng.** Ngày tháng, email, URL — xác thực bằng các bộ phân tích cú pháp cụ thể, không chỉ dựa vào biểu thức chính quy (regex).

Mọi thất bại xác thực cần trả về một quan sát có cấu trúc để mô hình có thể thử lại với cấu trúc dữ liệu chính xác.

### Lệnh gọi công cụ song song (Parallel tool calls)

Các nhà cung cấp hiện đại hỗ trợ gọi nhiều công cụ song song trong một lượt trợ lý. Vòng lặp:

1. Mô hình phát ra 3 lệnh gọi công cụ với các `tool_use_id` riêng biệt.
2. Runtime thực thi chúng (song song nếu các tác vụ độc lập).
3. Mỗi kết quả được trả về dưới dạng khối `tool_result` tương ứng dựa theo `tool_use_id`.

Quy tắc kỹ nghệ: coi các mã định danh tương quan (correlation IDs) là mắt xích chịu tải cốt lõi. Tráo đổi chúng sẽ dẫn đến việc định tuyến nhầm kết quả công cụ này sang lệnh gọi công cụ khác.

### Đóng hộp cát (Sandboxing)

Việc thực thi công cụ chính là ranh giới hộp cát an toàn. Xem chi tiết tại Phase 13 · 26. Tóm tắt ngắn gọn: mọi công cụ cần chỉ định rõ bề mặt đọc/ghi, quyền truy cập mạng, thời gian chờ (timeout) và giới hạn bộ nhớ. Hàm chung chung như `run_shell(cmd)` là một dấu hiệu cảnh báo nguy hiểm; một hàm cụ thể như `git_status()` sẽ an toàn hơn nhiều.

```figure
tool-routing
```

## Xây dựng thực tế

Tệp `code/main.py` triển khai một sổ đăng ký công cụ chuẩn môi trường sản xuất:

- Bộ xác thực tập con JSON Schema (chỉ sử dụng thư viện chuẩn stdlib).
- Đăng ký công cụ với phần mô tả, lược đồ đầu vào, giới hạn thời gian và hàm thực thi.
- Ép kiểu đối số và xác thực enum.
- Điều phối công cụ song song với mã định danh liên kết.
- Các quan sát lỗi dưới dạng chuỗi có cấu trúc.

Chạy thử:

```bash
python3 code/main.py
```

Bản ghi vết thực thi hiển thị một tác tử thu nhỏ gọi ba công cụ trong một lượt duy nhất, trong đó có một lệnh gọi cố tình sai định dạng và bị từ chối kèm thông báo lỗi chi tiết giúp mô hình có thể tự khắc phục.

## Sử dụng trong thực tế

Mỗi nhà cung cấp đều có lược đồ công cụ riêng — Anthropic, OpenAI, Gemini, Bedrock. Hãy sử dụng một tầng dịch chuyển đổi (OpenAI Agents SDK, Vercel AI SDK, bộ điều hợp công cụ của LangChain) nếu bạn cần hỗ trợ đa nhà cung cấp. BFCL là bài kiểm chuẩn tham chiếu — hãy chạy nó để đánh giá tác tử của bạn trước khi đưa vào sản xuất nếu việc dùng công cụ là tính năng trung tâm của sản phẩm.

## Triển khai thành phẩm

Tệp `outputs/skill-tool-registry.md` tạo ra danh mục công cụ, lược đồ và sổ đăng ký cho một miền tác vụ nhất định. Bao gồm các bài kiểm tra chất lượng mô tả (mô tả của từng công cụ có nêu rõ cho mô hình biết khi nào nên dùng hay không?).

## Bài tập rèn luyện

1. Thêm một công cụ "no-op" (không làm gì) cho phép mô hình từ chối sử dụng bất kỳ công cụ nào một cách tường minh. Đo lường hiệu quả trên bài kiểm thử ảo giác tương tự BFCL.
2. Triển khai ép kiểu đối số cho chuỗi thành số nguyên và chuỗi thành số thực. Đến giới hạn nào thì việc ép kiểu bắt đầu che giấu các lỗi nghiêm trọng thực tế?
3. Thêm thời gian chờ (timeout) cho từng công cụ và một bộ ngắt mạch (circuit breaker - từ chối công cụ trong 60 giây sau 3 lần thất bại liên tiếp). Điều này thay đổi cách mô hình phục hồi sau sự cố như thế nào?
4. Đọc mô tả BFCL V4. Chọn một danh mục (ví dụ: "đa lượt") và chạy 10 prompt mẫu qua tác tử của bạn. Báo cáo tỷ lệ vượt qua.
5. Chuyển đổi bộ xác thực stdlib sang Pydantic hoặc Zod. Pydantic/Zod đã phát hiện được những điểm thiếu sót nào mà bản mô phỏng đơn giản bỏ qua?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất thực sự |
|-----------|----------------------|------------------|
| Gọi hàm (Function calling) | "Sử dụng công cụ" | Gọi công cụ bằng đầu ra có cấu trúc với lược đồ đã được xác thực |
| Toolformer | "Tự động gán nhãn công cụ" | Schick 2023 — giữ lại các lệnh gọi công cụ có kết quả giúp giảm hàm mất mát token tiếp theo |
| BFCL | "Bảng xếp hạng gọi hàm Berkeley" | Bộ đo chuẩn năm 2026: 40% tác tử, 30% đa lượt, 10% trực tiếp, 10% phi trực tiếp, 10% ảo giác |
| Lược đồ công cụ (Tool schema) | "Chữ ký hàm cho mô hình" | name, description, JSON Schema của các đối số |
| tool_use_id | "Mã định danh liên kết" | Khóa gắn kết một lệnh gọi công cụ với kết quả của nó; bắt buộc đối với việc điều phối song song |
| Phát hiện ảo giác (Hallucination detection) | "Biết khi nào không nên gọi" | Danh mục V4: từ chối gọi khi không có công cụ nào phù hợp |
| Ép kiểu đối số (Argument coercion) | "Sửa chuỗi thành số nguyên" | Các hiệu chỉnh thu hẹp cho sự sai lệch lược đồ có thể đoán trước; từ chối nếu có sự mơ hồ |
| Đóng hộp cát (Sandboxing) | "Ranh giới thực thi công cụ" | Giới hạn bề mặt đọc/ghi, quyền mạng, thời gian chờ, bộ nhớ tối đa cho từng công cụ |

## Tài liệu đọc thêm

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — gán nhãn công cụ tự giám sát
- [Bảng xếp hạng gọi hàm Berkeley (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — chuẩn đánh giá năm 2026
- [Anthropic, Tài liệu sử dụng công cụ](https://platform.claude.com/docs/en/agent-sdk/overview) — lược đồ công cụ sản xuất trong Claude Agent SDK
- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — loại công cụ hàm và Guardrails
