# Khung Điều hành dưới dạng Thư viện — Tác tử Phụ và Kho lưu Phiên

> Một khung điều hành (harness) bạn có thể import trực tiếp: tích hợp sẵn các công cụ, tác tử phụ (subagents) giúp cô lập ngữ cảnh, các hook vòng đời, truyền lan ngữ cảnh truy vết theo chuẩn W3C, lưu phiên bền vững. Claude Agent SDK là ví dụ tham chiếu chuẩn — phiên bản thư viện của khung điều hành Claude Code — và Claude Managed Agents là giải pháp lưu trữ đám mây thay thế cho các tác vụ bất đồng bộ chạy dài.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## Mục tiêu học tập

- Giải thích sự khác biệt giữa Anthropic Client SDK (API thô) và Claude Agent SDK (cấu trúc khung điều hành).
- Mô tả các tác tử phụ (subagents) — khả năng song song hóa (parallelization) và cô lập ngữ cảnh (context isolation) — cùng thời điểm nên áp dụng.
- Nêu tên các phương thức giao tiếp của kho lưu phiên (session store) trong Python SDK (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) và vai trò của cờ `--session-mirror`.
- Triển khai một khung điều hành bằng thư viện chuẩn (stdlib) với các công cụ tích hợp sẵn, khởi tạo tác tử phụ với ngữ cảnh cô lập, các hook vòng đời và kho lưu phiên.

## Vấn đề

Một API LLM thô chỉ mang lại cho bạn một lượt khứ hồi (round-trip) yêu cầu-phản hồi đơn lẻ. Một tác tử môi trường sản xuất đòi hỏi việc thực thi công cụ, kết nối các máy chủ MCP (Model Context Protocol), kích hoạt hook vòng đời, phân nhánh tác tử phụ, lưu trữ phiên bền vững, truyền lan truy vết. Claude Agent SDK đóng gói cấu trúc này dưới dạng một thư viện — chính là khung điều hành mà Claude Code sử dụng, được mở ra cho các tác tử tùy biến.

## Khái niệm

### Client SDK so với Agent SDK

- **Client SDK (`anthropic`).** Messages API thô. Bạn tự quản lý vòng lặp, công cụ và trạng thái.
- **Agent SDK (`claude-agent-sdk`).** Tích hợp sẵn cơ chế thực thi công cụ, kết nối MCP, các hook, phân nhánh tác tử phụ và kho lưu phiên. Toàn bộ vòng lặp của Claude Code được cung cấp như một thư viện.

### Các công cụ tích hợp sẵn

SDK cung cấp hơn 10 công cụ có sẵn: đọc/ghi tệp, lệnh shell, grep, glob, lấy dữ liệu web và nhiều công cụ khác. Các công cụ tùy chỉnh được đăng ký thông qua giao diện lược đồ công cụ chuẩn (tool-schema interface).

### Tác tử phụ

Hai mục đích chính được Anthropic ghi nhận:

1. **Song song hóa (Parallelization).** Chạy các tác vụ độc lập một cách đồng thời. "Tìm tệp kiểm thử cho từng module trong số 20 module này" là 20 tác vụ tác tử phụ chạy song song.
2. **Cô lập ngữ cảnh (Context isolation).** Các tác tử phụ hoạt động trong cửa sổ ngữ cảnh riêng biệt của chúng; chỉ có kết quả cuối cùng được trả về cho tác tử điều phối (orchestrator). Ngân sách ngữ cảnh của tác tử điều phối nhờ đó được bảo toàn.

Các bổ sung gần đây trong Python SDK: `list_subagents()`, `get_subagent_messages()` để đọc bản ghi lời thoại (transcripts) của tác tử phụ.

### Kho lưu phiên

Bảo đảm tính tương thích giao thức với TypeScript:

- `append(session_id, message)` — thêm một lượt hội thoại.
- `load(session_id)` — khôi phục cuộc hội thoại.
- `list_sessions()` — liệt kê danh sách các phiên.
- `delete(session_id)` — xóa phiên với cơ chế xóa xếp tầng (cascade) tới các phiên của tác tử phụ.
- `list_subkeys(session_id)` — liệt kê các khóa của tác tử phụ.

Cờ CLI `--session-mirror` sao chép phản chiếu bản ghi hội thoại sang một tệp bên ngoài ngay khi đang truyền luồng (streaming), phục vụ cho việc gỡ lỗi.

### Các Hook

Các hook vòng đời mà bạn có thể đăng ký:

- `PreToolUse`, `PostToolUse` — kiểm duyệt cổng hoặc kiểm toán các lệnh gọi công cụ.
- `SessionStart`, `SessionEnd` — thiết lập khởi tạo và dọn dẹp kết thúc phiên.
- `UserPromptSubmit` — can thiệp vào đầu vào của người dùng trước khi mô hình nhìn thấy.
- `PreCompact` — kích hoạt trước khi nén gọn ngữ cảnh (context compaction).
- `Stop` — dọn dẹp tài nguyên khi tác tử thoát.
- `Notification` — các cảnh báo kênh phụ (side-channel).

Hooks là cách mà pro-workflow (tài liệu tham chiếu chương trình Phase 14) và các hệ thống tương tự áp dụng để bổ sung các hành vi cắt ngang (cross-cutting behaviors).

### Ngữ cảnh truy vết W3C

Các span OpenTelemetry đang hoạt động trên bên gọi sẽ được truyền lan vào tiến trình con của CLI thông qua các tiêu đề ngữ cảnh truy vết W3C. Toàn bộ dấu vết đa tiến trình sẽ hiển thị dưới dạng một dấu vết thống nhất duy nhất trên hệ thống backend của bạn.

### Claude Managed Agents

Giải pháp lưu trữ đám mây được quản lý (tiêu đề beta `managed-agents-2026-04-01`). Dành cho các công việc bất đồng bộ chạy dài, tích hợp sẵn bộ nhớ đệm prompt (prompt caching) và nén ngữ cảnh tự động. Bạn đánh đổi quyền kiểm soát sâu để lấy hạ tầng được quản lý toàn diện.

### Những cạm bẫy thường gặp

- **Phân nhánh quá nhiều tác tử phụ (Subagent over-spawn).** Khởi tạo 100 tác tử phụ cho 100 tác vụ vụn vặt. Chi phí quản lý (overhead) sẽ chiếm ưu thế hoàn toàn. Thay vào đó hãy xử lý theo lô (batch).
- **Lạm dụng hook (Hook creep).** Mỗi nhóm phát triển đều thêm hook riêng; thời gian khởi động hệ thống tăng vọt. Hãy đánh giá lại các hook định kỳ hàng quý.
- **Phình to phiên lưu trữ (Session bloat).** Số lượng phiên tích lũy dần; kích thước phình to. Hãy sử dụng `list_sessions` kết hợp chính sách hết hạn (expiry policy).

```figure
ae-subagent-isolation
```

## Xây dựng

`code/main.py` triển khai hình thái SDK chỉ dùng thư viện chuẩn (stdlib):

- `Tool`, `ToolRegistry` với các công cụ tích hợp sẵn `read_file`, `write_file`, `list_dir`.
- `Subagent` — ngữ cảnh riêng tư, lượt chạy cô lập, trả về kết quả tổng hợp.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Bản demo: tác tử chính phân nhánh 3 tác tử phụ chạy song song (mỗi tác tử đều được cô lập), tổng hợp kết quả và lưu phiên bền vững.

Chạy thử:

```bash
python3 code/main.py
```

Dấu vết thực thi minh họa sự cô lập ngữ cảnh của tác tử phụ (kích thước ngữ cảnh của bộ điều phối luôn giữ trong giới hạn), việc thực thi các hook, và khả năng lưu phiên bền vững.

## Sử dụng

- **Claude Agent SDK** cho các sản phẩm ưu tiên Claude mong muốn cấu trúc khung điều hành tương tự Claude Code.
- **Claude Managed Agents** cho các công việc bất đồng bộ chạy dài trên hạ tầng đám mây.
- **OpenAI Agents SDK** (Bài 16) cho đối trọng tương đương bên hệ sinh thái OpenAI.
- **LangGraph + công cụ tùy biến** nếu bạn muốn một máy trạng thái có cấu trúc đồ thị.

## Chuyển giao

`outputs/skill-claude-agent-scaffold.md` tạo khung ứng dụng Claude Agent SDK với tác tử phụ, các hook, kho lưu phiên, gắn kết máy chủ MCP và truyền lan dấu vết W3C.

## Bài tập

1. Thêm bộ khởi tạo tác tử phụ gom lô 20 nhiệm vụ thành các nhóm gồm 5 tác tử phụ chạy song song. Đo lường kích thước ngữ cảnh của bộ điều phối so với việc chạy từng tác tử một.
2. Triển khai một hook `PreToolUse` giới hạn tần suất (rate-limit) các lệnh gọi `write_file` (5 lần mỗi phút trên mỗi phiên). Ghi nhận hành vi trong dấu vết.
3. Nối `list_subkeys` để hiển thị cây phân cấp tác tử phụ. Cấu trúc sẽ trông như thế nào khi lồng nhau ở mức độ sâu?
4. Chuyển đổi mã đồ chơi sang gói Python `claude-agent-sdk` thực tế. Quy trình đăng ký công cụ có điểm gì thay đổi?
5. Đọc tài liệu Claude Managed Agents. Trong trường hợp nào bạn sẽ chuyển đổi từ tự lưu trữ (self-hosted) sang dịch vụ được quản lý (managed)?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Agent SDK | "Claude Code dưới dạng thư viện" | Cấu trúc khung điều hành: công cụ, MCP, hooks, tác tử phụ, kho lưu phiên |
| Tác tử phụ (Subagent) | "Tác tử con" | Ngữ cảnh tách biệt, ngân sách riêng; kết quả được đưa ngược lên trên |
| Kho lưu phiên (Session store) | "Cơ sở dữ liệu hội thoại" | Lưu bền vững, tải, liệt kê, xóa các lượt hội thoại với cơ chế xóa xếp tầng |
| Hook | "Hàm gọi lại vòng đời (Lifecycle callback)" | Can thiệp trước/sau khi gọi công cụ, phiên, gửi prompt, nén, dừng |
| Ngữ cảnh truy vết W3C (W3C trace context) | "Truy vết xuyên tiến trình" | Span cha được truyền lan vào tiến trình con của CLI |
| Managed Agents | "Khung điều hành đám mây" | Dịch vụ do Anthropic vận hành cho các tác vụ bất đồng bộ chạy dài |
| `--session-mirror` | "Phản chiếu bản ghi" | Ghi các lượt hội thoại ra tệp ngoài theo thời gian thực khi truyền luồng |
| Máy chủ MCP (MCP server) | "Bề mặt công cụ" | Nguồn tài nguyên/công cụ bên ngoài được gắn vào tác tử |

## Tài liệu đọc thêm

- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — hình thái thư viện của Claude Code
- [Anthropic, Xây dựng tác tử với Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — các mẫu hình môi trường sản xuất
- [Tổng quan về Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — giải pháp thay thế dạng dịch vụ lưu trữ
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — đối trọng tương đương
