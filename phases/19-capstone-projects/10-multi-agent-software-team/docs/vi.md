# Đồ án Tốt nghiệp 10 — Đội ngũ Kỹ thuật Phần mềm Đa Tác tử

> Cấu trúc của một đội ngũ kỹ thuật đa tác tử (multi-agent engineering team) vào năm 2026 đã hội tụ: một kiến trúc sư lập kế hoạch, N lập trình viên làm việc song song trên các cây làm việc (parallel worktrees), một người phản biện kiểm duyệt (reviewer gates), và một người kiểm thử tiến hành xác minh. Kiến trúc nhà máy của SWE-AF, kỹ thuật nhắc dựa trên vai trò của MetaGPT, đồ thị tác tử định kiểu (typed actor graph) của AutoGen 0.4, Devin của Cognition, và Droids của Factory đều độc lập tìm đến khuôn mẫu này. Các worktree song song chuyển đổi thời gian thực tế (wall-clock) thành thông lượng. Trạng thái chia sẻ và các giao thức chuyển giao công việc (handoff protocols) trở thành bề mặt dễ phát sinh lỗi nhất. Đồ án tốt nghiệp này yêu cầu xây dựng đội ngũ tác tử này, đánh giá trên bộ chuẩn SWE-bench Pro, và công bố báo cáo phân tích các khâu chuyển giao bị đổ vỡ cùng tần suất xảy ra.

**Type:** Capstone  
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)  
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)  
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17  
**Time:** ~40 giờ  

## Vấn đề đặt ra

Các khung khai thác lập trình đơn tác tử (single-agent coding harnesses) chạm trần giới hạn khi đối mặt với các tác vụ lớn. Không phải vì từng tác tử riêng lẻ yếu kém, mà vì một ngữ cảnh 200k token không thể cùng lúc chứa đựng bản kế hoạch kiến trúc, bốn lát cắt mã nguồn song song, phản hồi của người phản biện, và kết quả đầu ra của bộ kiểm thử. Các nhà máy đa tác tử (multi-agent factories) chia tách bài toán: kiến trúc sư (architect) nắm giữ bản kế hoạch, các lập trình viên (coders) chịu trách nhiệm hiện thực hóa trong các nhánh git worktree song song, người phản biện (reviewer) đóng vai trò cổng kiểm soát, và người kiểm thử (tester) xác minh tính đúng đắn. Kiến trúc "nhà máy" của SWE-AF, vai trò của MetaGPT, đồ thị tác tử định kiểu của AutoGen — cả ba cách tiếp cận đều mô tả chung một cấu trúc này.

Bề mặt phát sinh lỗi lớn nhất nằm ở khâu chuyển giao (handoff). Kiến trúc sư có thể vạch ra kế hoạch mà lập trình viên không thể triển khai. Các lập trình viên tạo ra các bản diff xung đột nhau. Người phản biện phê duyệt nhầm một bản sửa lỗi mang tính ảo giác. Người kiểm thử chạy đua trong khi lập trình viên vẫn chưa viết xong mã. Bạn sẽ xây dựng một đội ngũ tác tử như vậy, vận hành trên 50 vấn đề (issues) của SWE-bench Pro, theo dõi từng khâu chuyển giao, và công bố báo cáo phân tích nguyên nhân gốc rễ (post-mortem).

## Khái niệm cốt lõi

Các vai trò được định nghĩa như các tác tử định kiểu (typed agents):
- **Kiến trúc sư (Architect - Claude Opus 4.7):** Đọc issue, lập kế hoạch chi tiết, và phân rã thành các nhiệm vụ con (subtasks) với giao diện tương tác minh bạch.
- **Lập trình viên (Coders - Claude Sonnet 4.7, N phiên bản song song):** Mỗi phiên bản hoạt động trong một nhánh `git worktree` riêng biệt kèm hộp cát Daytona, hiện thực hóa các nhiệm vụ con một cách độc lập.
- **Người phản biện (Reviewer - GPT-5.4):** Đọc bản diff đã hợp nhất và quyết định phê duyệt hoặc đưa ra các yêu cầu chỉnh sửa cụ thể.
- **Người kiểm thử (Tester - Gemini 2.5 Pro):** Chạy bộ kiểm thử trong môi trường cô lập và báo cáo trạng thái đạt/không đạt kèm các tạo tác (artifacts).

Giao tiếp diễn ra thông qua một bảng nhiệm vụ dùng chung (task board trên tệp hoặc Redis). Mỗi vai trò chỉ tiếp nhận các nhiệm vụ mà nó có thẩm quyền xử lý. Các khâu chuyển giao sử dụng các thông điệp định kiểu theo giao thức A2A (Agent-to-Agent protocol). Các vấn đề điều phối cần giải quyết bao gồm: giải quyết xung đột hợp nhất (vai trò điều phối viên hợp nhất hoặc hợp nhất ba chiều - three-way merge tự động), đồng bộ hóa trạng thái chia sẻ (bản kế hoạch bị đóng băng một khi các lập trình viên bắt đầu; tái lập kế hoạch là các sự kiện riêng biệt), và cổng kiểm soát phản biện (reviewer không thể phê duyệt các thay đổi do chính mình tạo ra hoặc đề xuất).

Độ khuếch đại token (token amplification) là chi phí tiềm ẩn. Mỗi ranh giới giữa các vai trò đều phát sinh thêm các lời nhắc tóm tắt và ngữ cảnh chuyển giao. Một phiên chạy 40 lượt của đơn tác tử có thể biến thành 160 lượt tổng cộng phân bổ qua bốn vai trò. Barem chấm điểm đặc biệt cân nhắc hiệu quả sử dụng token so với đường cơ sở đơn tác tử, bởi câu hỏi then chốt không phải là "hệ thống đa tác tử có chạy được không" mà là "nó có mang lại hiệu quả vượt trội trên mỗi đơn vị chi phí (dollar) hay không."

## Kiến trúc hệ thống

```text
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Ngăn xếp công nghệ

- **Điều phối (Orchestration):** LangGraph với trạng thái chia sẻ + các đồ thị con (sub-graphs) cho từng tác tử
- **Giao thức nhắn tin:** Giao thức A2A (Google 2025) cho các thông điệp định kiểu giữa các agent
- **Mô hình nền tảng:** Opus 4.7 (kiến trúc sư), Sonnet 4.7 (lập trình viên), GPT-5.4 (phản biện), Gemini 2.5 Pro (kiểm thử)
- **Cô lập nhánh làm việc:** `git worktree add` cho từng lập trình viên + hộp cát Daytona
- **Điều phối viên hợp nhất:** Bộ hợp nhất ba chiều tùy biến + giải quyết xung đột bằng LLM
- **Đánh giá (Eval):** SWE-bench Pro (50 issues), kịch bản SWE-AF, HumanEval++ cho kiểm thử đơn vị
- **Khả năng quan sát (Observability):** Langfuse gắn thẻ span theo vai trò, thống kê hạch toán token cho từng tác tử
- **Triển khai:** Kubernetes với mỗi vai trò là một Deployment độc lập + HPA dựa trên backlog hàng đợi

```figure
ce-team-handoff
```

## Từng bước xây dựng

1. **Bảng nhiệm vụ (Task board).** Sử dụng tệp JSONL với các loại thông điệp định kiểu: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. Các agent đăng ký theo dõi (subscribe) theo các thẻ tương ứng.

2. **Kiến trúc sư (Architect).** Đọc issue trên GitHub, chạy Opus 4.7 với mẫu kế hoạch bắt buộc định nghĩa rõ giao diện của các nhiệm vụ con (các tệp chỉnh sửa, hàm công khai, ảnh hưởng kiểm thử). Phát ra một `plan_request` chứa đồ thị có hướng không chu trình (DAG) các nhiệm vụ con.

3. **Lập trình viên (Coders).** N worker chạy song song, mỗi worker nhận một nhiệm vụ con từ bảng nhiệm vụ. Mỗi worker tạo một nhánh làm việc mới bằng `git worktree add` cùng một hộp cát Daytona riêng. Triển khai nhiệm vụ con. Phát ra thông điệp `diff_ready` kèm bản vá (patch) và biến thiên kiểm thử.

4. **Điều phối viên hợp nhất (Merge coordinator).** Khi tất cả lập trình viên hoàn tất, tiến hành hợp nhất ba chiều N nhánh vào nhánh staging. Chỉ kích hoạt giải quyết xung đột qua LLM khi có sự chồng chéo ở cấp độ tệp tin.

5. **Người phản biện (Reviewer).** GPT-5.4 đọc bản diff hợp nhất. Nghiêm cấm phê duyệt bản diff do chính mình viết. Phát ra `approved` (không thao tác thêm) hoặc `review_feedback` kèm các yêu cầu thay đổi cụ thể chuyển ngược về lập trình viên liên quan.

6. **Người kiểm thử (Tester).** Gemini 2.5 Pro chạy bộ kiểm thử trong một hộp cát sạch. Thu thập tạo tác (artifacts). Phát ra `test_passed` hoặc `test_failed` kèm dấu vết ngăn xếp (stacktraces). Các bài test thất bại được định tuyến quay lại lập trình viên phụ trách nhiệm vụ con gây ra lỗi.

7. **Hạch toán chuyển giao.** Mọi thông điệp đi qua ranh giới vai trò đều được tạo span trong Langfuse với kích thước payload và mô hình sử dụng. Tính toán hệ số khuếch đại token cho từng subtask: `(coder_tokens + reviewer_tokens + tester_tokens + architect_share) / coder_tokens`.

8. **Đánh giá thực nghiệm.** Chạy thử nghiệm trên 50 vấn đề của SWE-bench Pro. So sánh tỷ lệ pass@1 và chi phí $/vấn đề giải quyết được so với đường cơ sở đơn tác tử (một mô hình Sonnet 4.7 trong một worktree duy nhất).

9. **Báo cáo phân tích sau sự cố (Post-mortem).** Đối với mỗi vấn đề thất bại, xác định chính xác khâu chuyển giao nào bị đổ vỡ (kế hoạch quá mơ hồ, xung đột hợp nhất, phản biện duyệt nhầm lỗi ảo giác, kiểm thử chập chờn). Xây dựng biểu đồ tần suất lỗi chuyển giao.

## Vận hành thực tế

```console
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Bàn giao sản phẩm

Tệp `outputs/skill-multi-agent-team.md` mô tả sản phẩm bàn giao. Nhận URL của một issue và mức độ xử lý song song, đội ngũ tác tử sẽ tạo ra một Pull Request sẵn sàng để merge kèm báo cáo hạch toán chi tiết token theo từng vai trò.

| Trọng số | Tiêu chí | Phương pháp đo lường |
|:-:|---|---|
| 25 | Tỷ lệ pass@1 trên SWE-bench Pro | Tỷ lệ pass@1 trên tập con 50 issue đối chuẩn |
| 20 | Tăng tốc độ nhờ song song hóa | Thời gian thực thi (wall-clock) so với đường cơ sở đơn tác tử |
| 20 | Chất lượng phản biện | Tỷ lệ phê duyệt sai (false-approval) trên các mẫu cấy lỗi chủ động |
| 20 | Hiệu quả sử dụng token | Tổng số token tiêu thụ trên mỗi issue giải quyết thành công so với đơn tác tử |
| 15 | Kỹ thuật điều phối | Khả năng giải quyết xung đột hợp nhất, biểu đồ tần suất lỗi chuyển giao |
| **100** | | |

## Bài tập mở rộng

1. Cấy một lỗi hiển nhiên vào bản diff giữa chừng (ví dụ: thêm lệnh `return None` trước khối logic chính). Đo tỷ lệ phê duyệt sai của reviewer. Tinh chỉnh lời nhắc reviewer cho đến khi tỷ lệ phê duyệt sai giảm xuống dưới 5%.

2. Thu hẹp quy mô xuống còn hai lập trình viên (architect + 2 coders + reviewer + tester, mỗi coder chạy 2 subtask tuần tự). So sánh thời gian thực thi và tỷ lệ vượt qua.

3. Thay thế điều phối viên hợp nhất bằng ràng buộc đơn-người-ghi (single-writer constraint: các subtask chỉ được phép chạm vào các tập tệp rời nhau). Đo lường gánh nặng lập kế hoạch đè nặng lên architect.

4. Hoán đổi reviewer từ GPT-5.4 sang Claude Opus 4.7. Đo lường tỷ lệ phê duyệt sai và mức biến động chi phí token.

5. Bổ sung vai trò thứ năm: người viết tài liệu (documenter - Haiku 4.5). Sau khi duyệt mã, tác tử này tạo một mục changelog hoàn chỉnh. Đo lường xem chất lượng tài liệu có xứng đáng với lượng token bổ sung tiêu tốn hay không.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Parallel worktree | "Nhánh làm việc cô lập" | Lệnh `git worktree add` tạo một cây thư mục làm việc độc lập cho mỗi coder |
| Task board | "Bus thông điệp chia sẻ" | Kho lưu trữ thông điệp định kiểu dạng tệp hoặc Redis mà các agent đăng ký theo dõi |
| Handoff | "Ranh giới vai trò" | Bất kỳ thông điệp nào chuyển từ ngữ cảnh của vai trò này sang vai trò khác |
| Token amplification | "Chi phí phụ trội đa tác tử" | Tỷ số giữa tổng token của tất cả vai trò / token của đơn tác tử cho cùng một tác vụ |
| A2A protocol | "Giao thức Agent-sang-Agent" | Đặc tả kỹ thuật năm 2025 của Google dành cho việc truyền nhận thông điệp định kiểu giữa các agent |
| Merge coordinator | "Bộ tích hợp" | Thành phần thực thi hợp nhất ba chiều và làm trung gian hòa giải xung đột mã |
| False approval | "Ảo giác của người phản biện" | Tình huống reviewer phê duyệt bản diff dù mã nguồn chứa lỗi đã biết |

## Tài liệu đọc thêm

- [Kiến trúc nhà máy SWE-AF](https://github.com/Agent-Field/SWE-AF) — mô hình nhà máy đa tác tử tham chiếu của năm 2026
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — khung đa tác tử dựa trên phân định vai trò
- [AutoGen v0.4](https://github.com/microsoft/autogen) — khung kiến trúc tác tử định kiểu từ Microsoft
- [Cognition AI (Devin)](https://cognition.ai) — sản phẩm thương mại tham chiếu
- [Factory Droids](https://www.factory.ai) — giải pháp thương mại tham chiếu thay thế
- [Giao thức Google A2A](https://a2a-protocol.org/latest/) — đặc tả giao tiếp giữa các tác tử
- [Tài liệu git worktree](https://git-scm.com/docs/git-worktree) — cơ chế cách ly nhánh làm việc
- [SWE-bench Pro](https://www.swebench.com) — bộ chuẩn đánh giá mục tiêu
