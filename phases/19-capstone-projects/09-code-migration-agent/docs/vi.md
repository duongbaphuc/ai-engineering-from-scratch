# Đồ án Tốt nghiệp 09 — Agent Nâng cấp và Di chuyển Mã nguồn (Nâng cấp Ngôn ngữ / Môi trường Thực thi cấp Kho mã)

> MigrationBench của Amazon (chuyển đổi từ Java 8 lên 17) và công cụ di chuyển Py2 sang Py3 của Google App Engine đã thiết lập tiêu chuẩn của năm 2026. Nền tảng OpenRewrite của Moderne thực hiện viết lại cây cú pháp trừu tượng (Abstract Syntax Tree - AST) mang tính tất định ở quy mô lớn. Grit giải quyết bài toán tương tự bằng ngôn ngữ đặc thù miền (DSL) theo phong cách codemod. Mô hình sản xuất kết hợp cả hai: một tầng nền tảng tất định (deterministic substrate) cho các phép viết lại mã an toàn cộng với một tầng tác tử (agent layer) cho các trường hợp phức tạp mơ hồ, một môi trường hộp cát (sandbox) cho các bản build trên từng nhánh, và một bộ kiểm thử tự động đảm bảo toàn bộ bài test đều xanh trước khi mở Pull Request (PR). Đồ án tốt nghiệp này yêu cầu bạn di chuyển thành công 50 kho mã (repo) thực tế và công bố tỷ lệ thành công kèm hệ thống phân loại nguyên nhân thất bại (failure taxonomy).

**Type:** Capstone  
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)  
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)  
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17  
**Time:** ~30 giờ  

## Vấn đề đặt ra

Di chuyển mã nguồn quy mô lớn (large-scale code migration) là một trong những ứng dụng sản xuất rõ ràng và hiệu quả nhất của các tác tử lập trình (coding agents) vào năm 2026. Chân lý chuẩn (ground truth) rất rành mạch (bộ kiểm thử có vượt qua sau khi di chuyển hay không?), giá trị mang lại là thực tế (việc di chuyển cả một hệ thống dịch vụ Java 8 là dự án tiêu tốn hàng chục nhân sự), và các bộ chuẩn đánh giá đều công khai (tập con 50 repo của MigrationBench). OpenRewrite của Moderne xử lý phần tất định. Tầng agent sẽ xử lý mọi khía cạnh mà các công thức (recipes) của OpenRewrite không thể bao quát: viết lại mã trong các ngữ cảnh mơ hồ, độ lệch hệ thống build, cú pháp thuộc phần đuôi dài (long-tail syntax), và sự đổ vỡ phụ thuộc bắc cầu (transitive dependencies).

Bạn sẽ xây dựng một agent tiếp nhận một kho mã Java 8 (hoặc Python 2) và tạo ra một nhánh mã nguồn đã di chuyển thành công với quy trình CI hoàn toàn xanh. Bạn sẽ đo lường tỷ lệ vượt qua, mức độ bảo toàn độ bao phủ kiểm thử (test-coverage preservation), chi phí trên mỗi kho mã, và xây dựng một hệ thống phân loại thất bại. Việc đặt song song kết quả với đường cơ sở chỉ dùng công cụ tất định sẽ chỉ ra giá trị thực sự của agent nằm ở đâu.

## Khái niệm cốt lõi

Quy trình gồm hai tầng kiến trúc:
- **Tầng nền tảng tất định (Deterministic substrate):** OpenRewrite cho Java, libcst cho Python thực hiện phần lớn các phép viết lại cơ học một cách an toàn: import, chữ ký phương thức (method signatures), chỉnh sửa an toàn giá trị null (null-safety), cấu trúc try-with-resources, thay thế các API đã lỗi thời (deprecated API). Tầng này xử lý nhanh và tạo ra các bản diff kiểm toán được rõ ràng.
- **Tầng tác tử (Agent layer):** OpenAI Agents SDK hoặc LangGraph chạy trên Claude Opus 4.7 và GPT-5.4-Codex xử lý những tình huống mà các công thức tĩnh không giải quyết được: nâng cấp tệp cấu hình build (Maven/Gradle/pyproject), xung đột phụ thuộc bắc cầu, kiểm thử chập chờn (test flakes), và các chú thích tùy biến (custom annotations).

Mỗi kho mã được cấp một môi trường hộp cát Daytona (Daytona sandbox) đã cài đặt sẵn môi trường thực thi đích. Agent sẽ lặp theo chu trình: chạy build, phân loại lỗi, áp dụng bản sửa lỗi, chạy lại. Các giới hạn cứng (hard limits): 30 phút cho mỗi repo, kinh phí tối đa $8 mỗi repo, 20 lượt agent (turns). Nếu toàn bộ bài kiểm thử đều đạt và độ bao phủ kiểm thử giảm không quá 2 điểm phần trăm, nhánh đó sẽ mở một PR. Nếu không, kho mã sẽ được phân loại vào một nhóm thất bại kèm bằng chứng chi tiết.

Hệ thống phân loại thất bại (failure taxonomy) chính là sản phẩm bàn giao cốt lõi. Qua 50 repo, điều gì đã gây ra lỗi? Phụ thuộc bắc cầu? Chú thích tùy biến? Phiên bản công cụ build? Kiểm thử chập chờn không liên quan đến việc di chuyển? Mỗi nhóm lỗi đều có số lượng thống kê và một bản diff điển hình làm mẫu, giúp các tác giả công thức sau này có thể tập trung giải quyết top 3 nguyên nhân hàng đầu.

## Kiến trúc hệ thống

```text
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Ngăn xếp công nghệ

- **Tầng tất định (Deterministic substrate):** OpenRewrite (Java) hoặc libcst (Python)
- **Tác tử (Agent):** OpenAI Agents SDK hoặc LangGraph trên nền Claude Opus 4.7 + GPT-5.4-Codex
- **Hộp cát (Sandbox):** Devcontainer của Daytona trên từng nhánh, cài sẵn môi trường thực thi đích (Java 17 / Python 3.12)
- **Hệ thống Build:** Maven, Gradle, uv (Python)
- **Bộ chuẩn (Benchmarks):** Tập con 50 repo của Amazon MigrationBench (Java 8 lên 17), các repo Py2 sang Py3 của Google App Engine
- **Bộ kiểm thử tự động:** Trình chạy song song, đo độ bao phủ qua Jacoco (Java) hoặc coverage.py (Python)
- **Khả năng quan sát (Observability):** Langfuse + gói truy vết (trace bundle) cho mỗi repo kèm từng khối diff
- **Bảng điều khiển (Dashboard):** Bảng điều khiển phân loại thất bại với số lượng từng nhóm và diff mẫu trực quan

```figure
ce-migration-funnel
```

## Từng bước xây dựng

1. **Lượt chạy công thức tĩnh.** Chạy các công thức của OpenRewrite (Java) hoặc libcst (Python) trước tiên. Xử lý 70-80% các trường hợp di chuyển mang tính cơ học thuần túy. Commit với thông điệp "recipe".

2. **Thử nghiệm Build.** Thiết lập hộp cát Daytona: cài đặt runtime đích, kích hoạt build. Nếu thành công (xanh), chuyển thẳng sang bước chạy kiểm thử. Nếu thất bại (đỏ), chuyển giao quyền xử lý cho agent.

3. **Vòng lặp tác tử (Agent loop).** Cấu hình LangGraph với các công cụ: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. Agent phân loại nguyên nhân lỗi (phụ thuộc, cú pháp, bài kiểm thử, công cụ build) và áp dụng bản sửa lỗi có mục tiêu. Sau đó kích hoạt chạy lại.

4. **Kiểm soát giới hạn ngân sách.** Giới hạn 30 phút thời gian thực cho mỗi repo, $8 chi phí API, 20 lượt lặp của agent. Bất kỳ vi phạm nào cũng lập tức dừng tiến trình và xếp vào nhóm "budget_exhausted" cùng bản diff hiện tại.

5. **Cổng kiểm tra bài test + độ bao phủ.** Sau khi quá trình build thành công, chạy toàn bộ bộ kiểm thử. So sánh độ bao phủ với repo ban đầu. Nếu độ bao phủ giảm quá 2 điểm phần trăm, xếp vào nhóm "coverage_regression".

6. **Mở Pull Request.** Khi thành công, đẩy nhánh lên kho từ xa, mở PR kèm nội dung diff và bản tóm tắt ghi rõ công thức nào đã áp dụng cùng các commit do agent tự sinh.

7. **Phân loại thất bại.** Đối với mỗi repo không thành công, gắn nhãn phân loại: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. Xây dựng bảng điều khiển trực quan hóa.

8. **Vận hành trên 50 kho mã.** Thực thi trên tập con của MigrationBench. Báo cáo tỷ lệ thành công theo từng nhóm, chi phí trung bình trên mỗi repo, mức độ bảo toàn độ bao phủ kiểm thử, và biểu đồ so sánh với đường cơ sở chỉ dùng công cụ tất định.

## Vận hành thực tế

```console
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Bàn giao sản phẩm

Tệp `outputs/skill-migration-agent.md` mô tả sản phẩm bàn giao. Nhận đầu vào là một repo, hệ thống sẽ thực thi các công thức tất định rồi đến vòng lặp tác tử để tạo ra một nhánh di chuyển thành công với CI xanh, hoặc phân loại repo đó vào một nhóm lỗi cụ thể trong taxonomy.

| Trọng số | Tiêu chí | Phương pháp đo lường |
|:-:|---|---|
| 25 | Tỷ lệ vượt qua MigrationBench | Tỷ lệ pass@1 trên tập con 50 repo |
| 20 | Bảo toàn độ bao phủ kiểm thử | Mức biến thiên độ bao phủ trung bình so với repo gốc |
| 20 | Chi phí trên mỗi repo di chuyển thành công | $/repo trên các lượt chạy đạt |
| 20 | Tích hợp giữa agent và công cụ tất định | Tỷ lệ các bản sửa lỗi do OpenRewrite xử lý so với do agent tự viết |
| 15 | Báo cáo phân tích nguyên nhân thất bại | Tính hoàn thiện của hệ thống phân loại kèm các diff mẫu |
| **100** | | |

## Bài tập mở rộng

1. Chạy quy trình di chuyển chỉ với OpenRewrite (không dùng agent). So sánh tỷ lệ thành công với toàn bộ quy trình hoàn chỉnh. Xác định rõ những trường hợp mà sự can thiệp của agent tạo nên sự khác biệt.

2. Cài đặt thêm bước kiểm tra chuẩn phong cách mã ("lint-clean"): sau khi di chuyển, chạy công cụ linter (spotless cho Java, ruff cho Python). Đánh dấu thất bại nếu xuất hiện lỗi linter mới. Đo lường tỷ lệ các trường hợp bảo toàn độ bao phủ nhưng vi phạm quy chuẩn mã nguồn.

3. Xây dựng bộ tối ưu hóa diff tối thiểu ("minimal-diff optimizer"): sau khi nhánh của agent vượt qua kiểm thử, chạy một lượt tinh lọc để loại bỏ các thay đổi không cần thiết. Báo cáo tỷ lệ cắt giảm kích thước diff.

4. Mở rộng cho kịch bản di chuyển thứ ba: nâng cấp từ Node 18 lên Node 22. Tái sử dụng lớp bao bọc hộp cát; thay thế tầng công thức bằng một codemod tùy biến.

5. Đo lường thời gian cho đến lần build xanh đầu tiên (Time-To-First-Green-Build - TTFGB) như một chỉ số trải nghiệm người dùng (UX metric). Đặt mục tiêu: trung vị (p50) dưới 10 phút.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Deterministic substrate | "Công cụ công thức" | OpenRewrite / libcst: viết lại AST dạng khai báo với cam kết bảo đảm an toàn |
| Codemod | "Chương trình biến đổi mã" | Một quy tắc viết lại làm thay đổi mã nguồn một cách cơ học |
| Build drift | "Lệch phiên bản công cụ" | Các thay đổi hành vi tinh vi của Maven / Gradle / uv giữa các phiên bản chính |
| Failure class | "Nhóm phân loại lỗi" | Nhãn lý do một repo không thể di chuyển: phụ thuộc, cú pháp, kiểm thử, công cụ build, cạn ngân sách |
| Coverage delta | "Độ bảo toàn kiểm thử" | Mức chênh lệch phần trăm độ bao phủ kiểm thử giữa repo gốc và nhánh đã di chuyển |
| Agent turn | "Vòng gọi công cụ" | Một chu trình lập kế hoạch -> hành động -> quan sát trong vòng lặp của agent |
| Budget exhaustion | "Chạm trần giới hạn" | Kho mã tiêu tốn hết hạn mức 30 phút / $8 / 20 lượt mà vẫn chưa vượt qua bài test |

## Tài liệu đọc thêm

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — bộ chuẩn đánh giá chuẩn mực năm 2026
- [Nền tảng Moderne.io OpenRewrite](https://www.moderne.io) — tài liệu tham chiếu về tầng nền tảng tất định
- [Tài liệu OpenRewrite](https://docs.openrewrite.org) — hướng dẫn viết công thức biến đổi mã
- [Grit.io](https://www.grit.io) — DSL thay thế cho codemod
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — tài liệu tham chiếu cho Agents SDK
- [Bộ di chuyển Py2 sang Py3 của Google App Engine](https://cloud.google.com/appengine) — bài chuẩn di chuyển thay thế
- [libcst](https://github.com/Instagram/LibCST) — tầng nền tảng tất định cho Python
- [Hộp cát Daytona](https://daytona.io) — tài liệu tham chiếu môi trường hộp cát trên từng nhánh
