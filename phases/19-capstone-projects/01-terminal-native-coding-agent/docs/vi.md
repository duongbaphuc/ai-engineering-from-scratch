# Đồ án Tốt nghiệp 01 — Tác tử Lập trình Dòng lệnh (Terminal-Native Coding Agent)

> Đến năm 2026, kiến trúc định hình của một tác tử lập trình (coding agent) đã được chuẩn hóa: Một khung điều khiển giao diện dòng lệnh (TUI harness), một bản kế hoạch có lưu trạng thái (stateful plan), một bề mặt công cụ trong môi trường hộp cát (sandboxed tool surface), cùng một vòng lặp liên tục lập kế hoạch, hành động, quan sát và phục hồi lỗi. Claude Code, Cursor 3 và OpenCode nhìn từ xa đều vận hành theo cùng một mô hình. Đồ án tốt nghiệp này yêu cầu bạn xây dựng hoàn chỉnh từ đầu đến cuối một tác tử như vậy — nhận yêu cầu từ CLI và xuất ra một pull request hoàn chỉnh — đồng thời đánh giá hiệu năng so với mini-swe-agent và Live-SWE-agent trên bộ tiêu chuẩn SWE-bench Pro. Bạn sẽ hiểu vì sao phần phức tạp nhất không nằm ở lượt gọi mô hình ngôn ngữ, mà nằm ở vòng lặp công cụ, môi trường hộp cát và trần chi phí cho một phiên chạy 50 lượt.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (Kỹ thuật LLM), Phase 13 (Công cụ và giao thức), Phase 14 (Tác tử), Phase 15 (Hệ thống tự trị), Phase 17 (Hạ tầng)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** ~35 giờ

## Bài toán (Problem)

Các tác tử lập trình đã trở thành hạng mục ứng dụng AI thống trị vào năm 2026. Claude Code (Anthropic), Cursor 3 với Composer 2 và Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k stars), Factory Droids và Google Jules đều triển khai các biến thể của cùng một kiến trúc cốt lõi: khung điều khiển terminal, bề mặt công cụ được phân quyền, môi trường hộp cát cách ly và vòng lặp lập kế hoạch - hành động - quan sát (plan-act-observe loop) xoay quanh một mô hình biên (frontier model). Ranh giới mô hình đỉnh cao rất hẹp — Live-SWE-agent đạt 79,2% trên SWE-bench Verified với Opus 4.5 — nhưng kỹ nghệ công trình (engineering craft) để hiện thực hóa thì vô cùng rộng lớn. Phần lớn các trường hợp thất bại không đến từ sai sót của mô hình, mà bắt nguồn từ sự bất ổn định của vòng lặp công cụ (tool-loop instability), ngộ độc ngữ cảnh (context poisoning), chi phí token tăng mất kiểm soát và các thao tác phá hủy trên hệ thống tệp.

Bạn không thể suy luận thấu đáo về các tác tử này nếu chỉ nhìn từ bên ngoài. Bạn phải tự tay dựng nên một tác tử, chứng kiến vòng lặp đổ vỡ ở lượt thứ 47 khi ripgrep trả về 8MB kết quả khớp, rồi tái thiết kế lại tầng cắt tỉa dữ liệu (truncation layer). Đó chính là trọng tâm của đồ án tốt nghiệp này.

## Khái niệm (Concept)

Khung điều khiển (harness) sở hữu bốn bề mặt chức năng chính:
- **Lập kế hoạch (Plan):** Duy trì đối tượng trạng thái theo khuôn mẫu TodoWrite mà mô hình viết lại hoàn toàn sau mỗi lượt tương tác.
- **Hành động (Act):** Điều phối và thực thi các lệnh gọi công cụ (đọc, sửa, chạy lệnh, tìm kiếm, thao tác git).
- **Quan sát (Observe):** Thu thập luồng đầu ra stdout / stderr / mã thoát (exit codes), cắt tỉa dữ liệu và truyền bản tóm tắt ngược lại cho mô hình.
- **Phục hồi (Recover):** Xử lý ngoại lệ công cụ mà không làm nổ tung cửa sổ ngữ cảnh (context window) hoặc rơi vào vòng lặp vô tận.

Hình thái năm 2026 bổ sung thêm một thành phần cốt yếu: **các điểm can thiệp (hooks)**. Bao gồm `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, và `PreCompact` — những điểm mở rộng có thể cấu hình, nơi người vận hành tiêm các chính sách kiểm soát, dữ liệu đo lường từ xa (telemetry) và các rào chắn an toàn (guardrails).

Môi trường hộp cát (sandbox) sử dụng E2B hoặc Daytona. Mỗi tác vụ được kích hoạt trong một devcontainer mới với một cây làm việc git (git worktree) được gắn ở chế độ đọc-ghi (read-write). Khung điều khiển tuyệt đối không chạm vào hệ thống tệp của máy chủ (host). Cây làm việc sẽ bị hủy bỏ sau khi tác vụ hoàn thành hoặc thất bại. Việc kiểm soát chi phí được siết chặt tại ba tầng: trần token trên từng lượt (per-turn token ceiling), ngân sách chi tiêu tối đa cho mỗi phiên (per-session dollar budget), và giới hạn lượt tương tác cứng (hard turn limit, thông thường là 50 lượt). Tầng khả năng quan sát (observability layer) sử dụng các span của OpenTelemetry tuân thủ quy ước ngữ nghĩa GenAI (GenAI semantic conventions), gửi dữ liệu về một phiên bản Langfuse tự lưu trữ (self-hosted).

## Kiến trúc (Architecture)

```text
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
            vòng lặp plan / act / observe  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                                (qua OpenRouter, độc lập mô hình)
                  v
            bộ điều phối công cụ (MCP StreamableHTTP client)
                  |
      +-----------+------------+----------+
      v           v            v          v
   read/edit   ripgrep    tree-sitter   git/run
      |           |            |          |
      +-----------+------------+----------+
                  |
                  v
            E2B / Daytona sandbox (cách ly git worktree)
                  |
                  v
            hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
            OpenTelemetry -> Langfuse (spans, tokens, chi phí $)
                  |
                  v
            Tạo PR tự động qua GitHub App
```

## Công nghệ sử dụng (Stack)

- Runtime điều khiển: Bun 1.2 + Ink 5 (React trong terminal)
- Truy cập mô hình: OpenRouter unified API hỗ trợ Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (cho các tác vụ khó nhất)
- Giao thức truyền tải công cụ: Giao thức Ngữ cảnh Mô hình (Model Context Protocol - MCP) StreamableHTTP (bản sửa đổi năm 2026)
- Hộp cát (Sandbox): Hộp cát E2B (JS SDK) hoặc devcontainer của Daytona
- Tìm kiếm mã nguồn: Tiến trình con ripgrep, trình phân tích cú pháp tree-sitter biên dịch sẵn cho 17 ngôn ngữ
- Cách ly (Isolation): `git worktree add` trên từng tác vụ, dọn dẹp khi thành công hoặc thất bại
- Khung đánh giá (Eval harness): SWE-bench Pro (tập con verified) + Terminal-Bench 2.0 + tập kiểm thử độc lập 30 tác vụ của riêng bạn
- Khả năng quan sát (Observability): OpenTelemetry SDK với quy ước ngữ nghĩa `gen_ai.*` → Langfuse tự lưu trữ
- Đăng tải PR: GitHub App với token phân quyền chi tiết, giới hạn phạm vi chỉ trong kho mã nguồn mục tiêu

```figure
ce-agent-loop
```

## Các bước triển khai (Build It)

1. **Giao diện TUI và vòng lặp lệnh.** Khởi tạo dự án Bun sử dụng Ink. Nhận lệnh cú pháp `agent run <repo> "<task>"`. Xuất ra màn hình chế độ xem chia khung: khung kế hoạch (trên cùng), luồng gọi công cụ trực tiếp (ở giữa), và ngân sách token (dưới cùng). Thêm cơ chế hủy bỏ an toàn khi nhấn Ctrl-C để kích hoạt hook `SessionEnd` trước khi thoát.

2. **Trạng thái kế hoạch (Plan state).** Định nghĩa lược đồ TodoWrite có kiểu dữ liệu chặt chẽ (các mục pending / in_progress / done kèm ghi chú). Mô hình sẽ ghi đè toàn bộ trạng thái sau mỗi lượt dưới dạng một lệnh gọi công cụ — không cho phép mô hình chỉnh sửa trạng thái cục bộ gia tăng. Lưu kế hoạch vào `.agent/state.json` để có thể tiếp tục thực thi sau sự cố sập ứng dụng.

3. **Bề mặt công cụ (Tool surface).** Khai báo sáu công cụ cốt lõi: `read_file`, `edit_file` (kèm xem trước diff), `ripgrep`, `tree_sitter_symbols`, `run_shell` (có cấu hình thời gian chờ timeout), `git` (status / diff / commit / push). Đóng gói dịch vụ qua MCP StreamableHTTP giúp khung điều khiển hoàn toàn độc lập với tầng truyền tải. Mọi công cụ phải trả về đầu ra đã được cắt tỉa (giới hạn tối đa 4k token cho mỗi lượt gọi).

4. **Bao bọc môi trường hộp cát (Sandbox wrapping).** Mỗi tác vụ khởi tạo một hộp cát E2B độc lập. Chạy `git worktree add -b agent/$TASK_ID` trên một nhánh làm việc mới. Toàn bộ các lệnh gọi công cụ đều thực thi bên trong hộp cát. Hệ thống tệp của máy chủ hoàn toàn bất khả xâm phạm.

5. **Các điểm can thiệp (Hooks).** Hiện thực hóa toàn bộ tám loại hook định chuẩn năm 2026. Thiết lập ít nhất bốn hook tùy biến: (a) `PreToolUse` rào chắn lệnh phá hoại, chặn đứng các thao tác như `rm -rf` ngoài vùng worktree, (b) `PostToolUse` hạch toán lượng token tiêu thụ, (c) `SessionStart` khởi tạo ngân sách, (d) `Stop` lưu trữ gói vết thực thi (trace bundle) hoàn chỉnh.

6. **Vòng lặp đánh giá (Eval loop).** Tạo bản sao tập con 30 issue từ SWE-bench Pro Python. Chạy khung điều khiển của bạn trên từng tác vụ. So sánh với mini-swe-agent (baseline tối giản) về tỷ lệ pass@1, số lượt tương tác trên mỗi tác vụ (turns-per-task), và chi phí tài chính trên mỗi tác vụ ($-per-task). Ghi nhận toàn bộ kết quả vào `eval/results.jsonl`.

7. **Kiểm soát chi phí (Cost control).** Thiết lập ngưỡng chặn cứng: tối đa 50 lượt, 200k token ngữ cảnh, và $5 cho mỗi tác vụ. Điểm can thiệp `PreCompact` sẽ tóm tắt các lượt tương tác cũ thành một khối trạng thái tiền đề (prior-state block) tại mốc 150k token, giải phóng dung lượng cho các quan sát mới mà không làm mất dấu kế hoạch tổng thể.

8. **Đăng tải PR (PR posting).** Khi giải quyết thành công, bước cuối cùng là `git push` kèm theo một lệnh gọi GitHub API để tạo PR, trong đó phần thân chứa toàn bộ kế hoạch thực hiện và bản tóm tắt diff.

## Ứng dụng thực tế (Use It)

```bash
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Triển khai thành phẩm (Ship It)

Kỹ năng thành phẩm sẵn sàng sử dụng được lưu trữ tại `outputs/skill-terminal-coding-agent.md`. Khi nhận vào một đường dẫn kho mã nguồn và mô tả tác vụ, nó sẽ thực thi toàn bộ vòng lặp plan-act-observe trong môi trường hộp cát và trả về URL của pull request cùng gói vết thực thi. Bảng tiêu chí đánh giá cho đồ án này:

| Trọng số | Tiêu chí | Phương thức đo lường |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 so với baseline | Khung điều khiển của bạn so với mini-swe-agent trên 30 tác vụ Python tương đương |
| 20 | Độ rõ ràng kiến trúc | Sự tách bạch rõ ràng giữa plan/act/observe, bề mặt hook, lược đồ công cụ — đánh giá đối chiếu với cấu trúc của Live-SWE-agent |
| 20 | Tính an toàn | Các bài kiểm thử thoát khỏi hộp cát, hộp thoại yêu cầu cấp quyền, rào chắn lệnh phá hoại vượt qua kiểm thử red-team |
| 20 | Khả năng quan sát | Độ toàn vẹn của dữ liệu vết thực thi (100% lệnh gọi công cụ được ghi nhận span), hạch toán token chi tiết theo từng lượt |
| 15 | Trải nghiệm lập trình viên (UX) | Thời gian khởi động nguội < 2s, khôi phục kế hoạch khi gặp sự cố sập, hủy thao tác sạch sẽ khi nhấn Ctrl-C giữa chừng |
| **100** | | |

## Bài tập mở rộng (Exercises)

1. Hoán đổi mô hình nền tảng từ Claude Sonnet 4.7 sang Qwen3-Coder-30B được phục vụ qua vLLM. So sánh pass@1 và chi phí $-per-task. Phân tích chi tiết những điểm mà mô hình nguồn mở hoạt động kém hiệu quả hơn.

2. Bổ sung một tác tử phụ chịu trách nhiệm phản biện (`reviewer`), có nhiệm vụ phân tích mã diff trước khi gửi PR và có quyền yêu cầu vòng lặp sửa đổi. Đo lường xem các đánh giá dương tính giả (false-positive reviews) có làm giảm tỷ lệ thành công trên SWE-bench xuống thấp hơn mức cơ sở của hệ thống tác tử đơn hay không (gợi ý: thường là có).

3. Kiểm thử sức chịu tải và bảo mật hộp cát: Viết một tác vụ cố gắng thực thi `curl` đến một URL bên ngoài và một tác vụ cố gắng ghi tệp ra ngoài vùng worktree. Xác minh cả hai hành vi đều bị chặn đứng bởi hook `PreToolUse`. Ghi nhật ký các lần vi phạm.

4. Hiện thực hóa cơ chế tóm tắt `PreCompact` bằng một mô hình nhỏ hơn (Haiku 4.5). Đo lường mức độ hao hụt độ chuẩn xác của kế hoạch khi nén ngữ cảnh xuống 3 lần.

5. Hoán đổi cơ chế truyền tải MCP StreamableHTTP sang stdio. Đo kiểm độ trễ khởi động nguội và độ trễ trên từng lệnh gọi. Lựa chọn giải pháp tối ưu cho nhu cầu chỉ chạy cục bộ.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Harness (Khung điều khiển) | "Vòng lặp tác tử" | Phần mã bao quanh mô hình có nhiệm vụ điều phối công cụ, duy trì trạng thái kế hoạch và thực thi ngân sách |
| Hook (Điểm can thiệp) | "Bộ lắng nghe sự kiện tác tử" | Đoạn mã do người dùng viết, được khung điều khiển kích hoạt tại một trong tám sự kiện vòng đời |
| Worktree (Cây làm việc) | "Hộp cát Git" | Một thư mục kiểm xuất git liên kết tại một đường dẫn riêng biệt; có thể xóa bỏ mà không làm ảnh hưởng bản sao chính |
| TodoWrite | "Trạng thái kế hoạch" | Danh sách định kiểu gồm các mục pending/in-progress/done mà mô hình viết lại toàn bộ sau mỗi lượt |
| StreamableHTTP | "Giao thức truyền tải MCP" | Bản sửa đổi MCP 2026: kết nối HTTP tồn tại lâu dài với khả năng truyền phát hai chiều; thay thế cho SSE |
| Token ceiling (Trần token) | "Ngân sách ngữ cảnh" | Giới hạn token đầu vào + đầu ra trên mỗi lượt hoặc mỗi phiên; kích hoạt cơ chế nén ngữ cảnh hoặc dừng phiên |
| pass@1 | "Tỷ lệ vượt qua lần đầu" | Tỷ lệ phần trăm các tác vụ SWE-bench được giải quyết thành công ngay trong lượt chạy đầu tiên mà không cần thử lại hay xem trước tập kiểm thử |

## Tài liệu đọc thêm (Further Reading)

- [Tài liệu Claude Code](https://docs.anthropic.com/en/docs/claude-code) — Khung điều khiển tham chiếu chuẩn mực từ Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — Ghi chú phát hành tính năng Agent Tabs và Composer 2
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — Hệ quy chiếu tối giản dùng để so sánh khung điều khiển trên SWE-bench
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — Đạt 79,2% trên SWE-bench Verified với Opus 4.5
- [OpenCode](https://opencode.ai) — Khung điều khiển mã nguồn mở với 112k stars
- [Bảng xếp hạng SWE-bench Pro](https://www.swebench.com) — Tiêu chuẩn đánh giá mà đồ án tốt nghiệp này hướng tới
- [Lộ trình Model Context Protocol 2026](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, siêu dữ liệu về năng lực công cụ
- [Quy ước ngữ nghĩa GenAI trong OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Lược đồ span cho các lệnh gọi công cụ và mức tiêu thụ token
