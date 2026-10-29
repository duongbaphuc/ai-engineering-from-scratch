# Toàn cảnh các tác tử lập trình tự chủ (2026)

> SWE-bench Verified đã nhảy vọt từ 4% lên 80.9% chỉ trong chưa đầy ba năm. Cùng một mô hình Claude Sonnet 4.5 đạt 43.2% trên SWE-agent v1 nhưng lại đạt tới 59.8% trên khung giàn tự chủ của Cline — phần khung giàn (scaffolding) bao quanh mô hình giờ đây quan trọng không kém gì bản thân mô hình. OpenHands (tiền thân là OpenDevin) là nền tảng mã nguồn mở giấy phép MIT sôi động nhất và vòng lặp CodeAct của nó thực thi trực tiếp các hành động bằng mã Python trong môi trường cô lập (sandbox) thay vì các lệnh gọi công cụ định dạng JSON (JSON tool calls). Những con số giật tít truyền thông đang che giấu một vấn đề phương pháp luận nghiêm trọng: 161 trong số 500 bài toán của SWE-bench Verified chỉ đòi hỏi thay đổi từ 1–2 dòng mã, trong khi SWE-bench Pro (các tác vụ yêu cầu thay đổi từ 10 dòng mã trở lên) vẫn chỉ dừng lại ở mức 23–59% đối với cùng các mô hình ở ranh giới tiên tiến (frontier models).

**Loại bài học:** Tìm hiểu (Learn)
**Ngôn ngữ:** Python (stdlib, so sánh giữa CodeAct và lệnh gọi công cụ JSON)
**Điều kiện tiên quyết:** Phase 14 · 07 (Sử dụng công cụ), Phase 15 · 01 (Tác tử chân trời dài hạn)
**Thời gian:** ~45 phút

## Vấn đề thực tế

"Tác tử lập trình nào tốt nhất?" là một câu hỏi sai lầm. Câu hỏi đúng phải là: trên một phân phối tác vụ phản ánh công việc thực tế của tôi, với khung giàn tôi sẽ vận hành trên môi trường sản xuất, mức độ tin cậy từ đầu đến cuối (end-to-end reliability) mà tôi nhận được là bao nhiêu?

Trong giai đoạn từ năm 2022 đến 2026, giới công nghệ nhận ra rằng khung giàn (scaffolding) — tầng truy xuất thông tin, bộ lập kế hoạch, môi trường cô lập sandbox, vòng lặp chỉnh sửa-xác minh, định dạng phản hồi — là thành phần chịu lực cốt lõi. Claude Sonnet 4.5 chạy trên SWE-agent v1 đạt 43.2% trên SWE-bench Verified; nhưng chính mô hình đó bên trong khung giàn tự chủ của Cline lại đạt 59.8%. Chênh lệch 16.6 điểm phần trăm tuyệt đối với cùng một tập trọng số mô hình. Mô hình nền tảng chỉ là một linh kiện; chính vòng lặp mới là sản phẩm hoàn chỉnh.

Vấn đề đi kèm là sự bão hòa của các bộ tiêu chuẩn đánh giá (benchmarks) đang che giấu các hiện tượng suy thoái. SWE-bench Verified đã tiến gần đến mức bão hòa, và phần đuôi các bài toán dễ (161 trên 500 bài toán chỉ yêu cầu ≤2 dòng sửa đổi) đã kéo điểm số hàng đầu lên cao. Chất lượng thực tế được đo lường chính xác hơn nhiều trên các phân phối như SWE-bench Pro (yêu cầu thay đổi từ 10 dòng mã trở lên), nơi mà các hệ thống dẫn đầu hiện nay vẫn chỉ đạt từ 23–59%.

## Khái niệm cốt lõi

### Tóm lược về SWE-bench

SWE-bench (Jimenez và cộng sự) thu thập các sự cố (issues) thực tế trên GitHub kèm theo các bản vá chuẩn (ground-truth patches) và yêu cầu tác tử tạo ra một bản vá giúp toàn bộ bộ kiểm thử (test suite) vượt qua thành công. SWE-bench Verified (OpenAI, 2024) là một tập con gồm 500 tác vụ được con người thẩm định kỹ lưỡng nhằm loại bỏ các bài toán mơ hồ hoặc bị lỗi. SWE-bench Pro là bước kế nhiệm khó khăn hơn — bao gồm các tác vụ đòi hỏi chỉnh sửa từ 10 dòng mã trở lên, nơi các tác tử tiên tiến hiện nay chỉ đạt từ 23–59%.

### Đường cong tăng trưởng từ 2022 đến 2026 thực sự chỉ ra điều gì

- **2022**: Các mô hình nghiên cứu chỉ đạt ~4% trên SWE-bench thô ban đầu.
- **2024**: GPT-4 kết hợp khung giàn kiểu Devin đạt ~14%; SWE-agent đạt ~12%.
- **2025**: Claude 3.5/3.7 Sonnet bên trong Aider và SWE-agent vươn lên mốc 40–55%.
- **2026**: Claude Sonnet 4.5 cùng các đối thủ cạnh tranh ở ranh giới tiên tiến đạt 70–80%+ trên SWE-bench Verified. Bảng xếp hạng của Epoch AI theo dõi sát sao dữ liệu này theo thời gian thực.

Độ dốc tăng trưởng đến từ ba nguồn tích hợp: mô hình nền tảng tốt hơn, khung giàn ưu việt hơn (CodeAct, phản tư/reflection, các vòng lặp kiểm chứng), và các bộ benchmark tốt hơn (Verified giúp loại bỏ nhiễu).

### CodeAct so với các lệnh gọi công cụ JSON (JSON tool calls)

OpenHands (All-Hands-AI, arXiv:2407.16741, tiền thân là OpenDevin) đã chọn một canh bạc kiến trúc cụ thể: thay vì mô hình phát ra các lệnh gọi công cụ JSON mà máy chủ phải giải mã rồi thực thi, mô hình sẽ phát trực tiếp mã Python và một nhân hạt nhân kiểu Jupyter (Jupyter-style kernel) sẽ thực thi đoạn mã đó trong sandbox. Tác tử có thể lặp qua các tệp, liên kết chuỗi các công cụ và tự bắt các ngoại lệ của chính nó ngay trong một hành động duy nhất.

Sự đánh đổi:

- **Lệnh gọi công cụ JSON**: Mỗi hành động là một lượt (turn); dễ kiểm toán; tính liên kết (compositionality) hạn chế; an toàn theo mặc định vì mỗi lệnh gọi đều đi qua một bộ xác thực tường minh.
- **CodeAct**: Một hành động có thể là cả một chương trình hoàn chỉnh; có tính liên kết cao; đòi hỏi môi trường sandbox được gia cố nghiêm ngặt (OpenHands sử dụng cơ chế cô lập Docker); các phương thức thất bại bao gồm bất kỳ điều gì mà runtime của sandbox cho phép.

Cả hai kiến trúc đều đang vận hành trong môi trường sản xuất thực tế. CodeAct chiếm ưu thế áp đảo trong các nền tảng mã nguồn mở (OpenHands, smolagents). Trong khi đó, các lệnh gọi công cụ JSON vẫn chiếm ưu thế trong các dịch vụ do nhà cung cấp quản lý (Anthropic Managed Agents, OpenAI Assistants), nơi nhà cung cấp cần toàn quyền kiểm soát trình thực thi.

### Các khung giàn trong bức tranh toàn cảnh năm 2026

| Khung giàn | Giấy phép | Mô hình thực thi | Đặc tính nổi bật |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct trong Docker | Nền tảng mở sôi động nhất; có thể phát lại luồng sự kiện (event-stream replayable) |
| SWE-agent | MIT | Giao diện Tác tử - Máy tính (ACI) | Khung giàn SWE-bench đầu-cuối đầu tiên |
| Aider | Apache-2 | Sửa đổi qua diff trong kho mã cục bộ | Khung giàn tối giản, độ ổn định hồi quy rất cao |
| Cline | Apache-2 | Tác tử VS Code với chính sách công cụ | Khung giàn mở đạt điểm cao nhất trên Sonnet 4.5 |
| Devin (Cognition) | Độc quyền (Proprietary) | Máy ảo được quản lý (Managed VM) + bộ lập kế hoạch | Dòng sản phẩm "kỹ sư phần mềm AI" thương mại đầu tiên |
| Claude Code | Độc quyền (Proprietary) | Các chế độ cấp quyền + quy trình làm việc | Bài 10 sẽ đi sâu chi tiết vào vòng lặp tác tử này |

### Tại sao khung giàn lại mang tính quyết định

Một chu trình chạy của tác tử lập trình là một quỹ đạo có chân trời dài hạn (long-horizon trajectory - Bài 1). Độ tin cậy sẽ nhân lũy kế qua từng bước. Ba thành phần nơi khung giàn đem lại điểm số vượt trội:

1. **Truy xuất thông tin (Retrieval)**: Tìm đúng các tệp cần đọc là nút thắt cổ chai âm thầm. ACI của SWE-agent, chỉ mục tệp của OpenHands và bản đồ repo (repo-map) của Aider đều tập trung giải quyết bài toán này.
2. **Vòng lặp kiểm chứng (Verifier loop)**: Việc chạy các bài kiểm thử, đọc vết ngăn xếp (stack traces) và thử lại đem lại sự chênh lệch hơn 10 điểm phần trăm trên SWE-bench.
3. **Khoanh vùng thất bại (Failure containment)**: Một sandbox có khả năng hoàn tác (rollback) khi gặp lỗi sẽ ngăn chặn thiệt hại dây chuyền. Cùng một mô hình nhưng khi có và không có vòng lặp kiểm chứng trông như hai sản phẩm hoàn toàn khác biệt.

### Sự bão hòa của benchmark và phân phối thực tế

Các tác giả OpenHands và Epoch AI đều cảnh báo rằng SWE-bench Verified tồn tại một phần đuôi các bài toán dễ: 161 trong số 500 tác vụ chỉ cần sửa 1–2 dòng mã. Điểm số cao một phần được thúc đẩy bởi phần đuôi này. SWE-bench Pro giới hạn bài toán ở các sửa đổi từ 10 dòng mã trở lên và trả về điểm số chỉ trong khoảng 23–59% ngay cả đối với các hệ thống ranh giới tiên tiến. Phân phối thực tế trong dự án của bạn gần như chắc chắn sẽ giống với Pro hơn là Verified.

Hàm ý đối với việc lựa chọn tác tử: hãy chạy một tập con dạng Pro trích xuất từ chính danh sách lỗi tồn đọng (bug backlog) của bạn. Điểm số thực sự có ý nghĩa là điểm số trên các tác vụ phản ánh chính xác những gì bạn phát hành trong thực tế.

```figure
a5-scaffold-delta
```

## Sử dụng trong thực tế

Tệp `code/main.py` so sánh hai khung giàn tác tử mẫu trên một phân phối tác vụ nhỏ cố định:

1. Khung giàn **gọi công cụ JSON** thực hiện một hành động trên mỗi lượt.
2. Khung giàn **CodeAct** có khả năng phát ra một đoạn mã Python ngắn trên mỗi hành động.

Cả hai đều sử dụng một "mô hình" giả lập (stub model với các quy tắc tất định) nhằm cô lập cấu trúc khung giàn khỏi chất lượng mô hình. Kết quả đầu ra chứng minh rằng khung giàn CodeAct giải quyết được nhiều tác vụ hơn với số lượt ít hơn, nhưng phải đánh đổi bằng bán kính tác động (blast radius) tiềm ẩn lớn hơn trên mỗi hành động.

## Đóng gói sản phẩm

Tệp `outputs/skill-scaffold-audit.md` hướng dẫn bạn kiểm toán một khung giàn tác tử lập trình được đề xuất trước khi ứng dụng vào thực tế: chất lượng truy xuất, sự hiện diện của bộ kiểm chứng, mức độ cô lập của sandbox, và mức độ phù hợp giữa benchmark với phân phối công việc thực tế.

## Bài tập

1. Chạy `code/main.py`. Mỗi khung giàn mất bao nhiêu lượt trên cùng một tập tác vụ? Bán kính tác động trên mỗi hành động của từng khung giàn là bao nhiêu?

2. Đọc bài báo về OpenHands (arXiv:2407.16741). Bài báo lập luận rằng CodeAct vượt trội hơn các lệnh gọi công cụ JSON trên các tác vụ phức tạp. Hãy xác định một phương thức thất bại mà bài báo thừa nhận và viết một câu nêu rõ khi nào phương thức này sẽ chiếm ưu thế trong môi trường sản xuất.

3. Chọn một tác vụ từ danh sách lỗi tồn đọng của bạn đòi hỏi chỉnh sửa từ 10 dòng mã trở lên trên hai tệp khác nhau. Hãy ước tính xác suất thành công từ đầu đến cuối cho một mô hình tiên tiến theo (a) lệnh gọi công cụ JSON và (b) CodeAct. Biện minh cho sự chênh lệch này.

4. SWE-bench Verified có 161 tác vụ đơn tệp từ 1–2 dòng mã. Hãy xây dựng một công thức tính điểm loại trừ các tác vụ này. Bảng xếp hạng sẽ xáo trộn như thế nào?

5. Đọc bài viết "Introducing SWE-bench Verified" của OpenAI. Giải thích phương pháp luận cụ thể được sử dụng để loại bỏ các tác vụ mơ hồ, và nêu tên một nhóm tác vụ mà quy trình tuyển chọn có thể bỏ sót.

## Thuật ngữ chính

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| SWE-bench | "Benchmark lập trình" | Các sự cố GitHub thực tế kèm bản vá chuẩn và bộ kiểm thử hoàn chỉnh |
| SWE-bench Verified | "Tập con đã làm sạch" | 500 tác vụ được con người thẩm định, vẫn tồn tại phần đuôi bài toán dễ |
| SWE-bench Pro | "Tập con khó hơn" | Các bài toán đòi hỏi sửa từ 10 dòng mã trở lên; các mô hình hàng đầu chỉ đạt 23–59% |
| CodeAct | "Mã là hành động" | Tác tử phát sinh mã Python; nhân kiểu Jupyter thực thi trong sandbox |
| JSON tool call | "Gọi hàm (function calling)" | Mỗi hành động là một gói tin JSON có cấu trúc được xác thực trước khi thực thi |
| Scaffold | "Khung giàn tác tử" | Truy xuất + lập kế hoạch + thực thi + vòng lặp kiểm chứng bao quanh mô hình nền tảng |
| ACI (Agent-Computer Interface) | "Định dạng của SWE-agent" | Tập lệnh được thiết kế tối ưu cho LLM, không phải cho shell của người dùng |
| Verifier loop | "Kiểm thử và thử lại" | Chạy kiểm thử, đọc đầu ra, sửa đổi bản vá; mức tăng độ tin cậy phi mô hình lớn nhất |

## Đọc thêm

- [Jimenez và cộng sự — SWE-bench](https://www.swebench.com/) — bộ benchmark gốc và phương pháp luận.
- [OpenAI — Giới thiệu SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — quy trình xây dựng tập con thẩm định.
- [Wang và cộng sự — OpenHands: Nền tảng mở cho kỹ sư phần mềm AI](https://arxiv.org/abs/2407.16741) — kiến trúc CodeAct và thiết kế luồng sự kiện.
- [Epoch AI — Bảng xếp hạng SWE-bench](https://epoch.ai/benchmarks) — các điểm số được cập nhật trực tiếp.
- [Anthropic — Đo lường quyền tự chủ của tác tử](https://www.anthropic.com/research/measuring-agent-autonomy) — khuôn khổ đánh giá độ tin cậy của tác tử lập trình chân trời dài hạn.
