# Đồ án Tốt nghiệp 05 — Tác tử Nghiên cứu Tự trị (Hạng mục AI-Scientist)

> AI-Scientist-v2 của Sakana đã tự động xuất bản các bài báo khoa học hoàn chỉnh. Agent Laboratory đã tự động vận hành các chuỗi thí nghiệm. Allen AI đã công bố các dấu vết thực thi minh bạch. Hình thái tiêu chuẩn năm 2026 là tìm kiếm cây theo cơ chế lập kế hoạch - thực thi - kiểm chứng (plan-execute-verify tree search) trên không gian các giả thuyết thí nghiệm, kiểm soát ngân sách chi phí nghiêm ngặt, thực thi mã trong môi trường hộp cát cách ly, sử dụng bộ soạn thảo LaTeX có phản hồi thị giác (vision-feedback LaTeX writer), và đánh giá bằng hội đồng phản biện mô hình lớn theo chuẩn NeurIPS (reviewer ensemble). Đồ án tốt nghiệp này yêu cầu bạn xây dựng một tác tử như vậy từ đầu đến cuối, hoàn tất một bài báo khoa học với chi phí dưới $30, và vượt qua bài kiểm tra đội đỏ (red team) về phòng chống thoát hộp cát mà nhóm Sakana từng cảnh báo.

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (Học máy), Phase 3 (Học sâu), Phase 7 (Mô hình Transformer), Phase 10 (Xây dựng LLM từ đầu), Phase 14 (Tác tử), Phase 15 (Hệ thống tự trị), Phase 16 (Hệ thống đa tác tử), Phase 18 (An toàn và rào chắn)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** ~40 giờ

## Bài toán (Problem)

Các tác tử nghiên cứu tự trị (autonomous research agents) đã vượt qua một cột mốc lịch sử vào năm 2026. Hệ thống AI-Scientist-v2 của Sakana AI đã có bài báo được xuất bản trên tạp chí Nature với các công trình tự động vượt qua quy trình phản biện đồng cấp của hội thảo khoa học. ShinkaEvolve (tại ICLR 2026) mở rộng hướng đi này sang việc tiến hóa giả thuyết liên tục. Agent Laboratory của AMD công bố các dấu vết thực thi có khả năng tái lập hoàn toàn. Những tác tử này không phải phép màu — chúng là một vòng lặp lập kế hoạch - thực thi - kiểm chứng vận hành trên một cây các thí nghiệm ứng viên, đi kèm mức trần chi phí, môi trường hộp cát gắn chặt với hạt giống ngẫu nhiên (seed), và cơ chế bình duyệt tự động. Kỹ nghệ cốt lõi nằm ở vòng lặp điều phối, quản trị ngân sách và cấu trúc an toàn.

Bạn sẽ làm chủ vòng lặp này bằng cách triển khai một tác tử xoay quanh một ý tưởng hạt giống trong phạm vi hẹp (ví dụ: nghiên cứu phân tích loại trừ độ thưa của cơ chế chú ý trên mô hình Transformer 100M tham số). Giá trị của đồ án không nằm ở việc tạo ra đột phá khoa học ngay lần chạy đầu tiên, mà nằm ở hệ thống hạ tầng: thuật toán duyệt cây, môi trường hộp cát thí nghiệm, vòng lặp tác giả - phản biện, và bản báo cáo kiểm thử đội đỏ. Nhóm Sakana đã ghi nhận nhiều sự cố tác tử cố gắng thoát khỏi hộp cát; hệ thống của bạn bắt buộc phải vượt qua các bài kiểm thử đối kháng tương tự.

## Khái niệm (Concept)

Tác tử hoạt động dựa trên thuật toán tìm kiếm cây ưu tiên tốt nhất (best-first tree search). Mỗi nút trên cây đại diện cho một đặc tả thí nghiệm: `(giả thuyết, cấu hình, mã nguồn, kết quả kỳ vọng)`. Bước mở rộng (expand) sẽ đề xuất các nút con với những thay đổi nhỏ (đổi bộ tối ưu hóa optimizer, điều chỉnh kích thước batch size, loại trừ một thành phần ablation). Mỗi nút con chạy trong một hộp cát mới tinh với hạn ngạch tài nguyên phần cứng cứng nhắc. Kết quả trả về được đưa vào hàm chấm điểm để xếp hạng các nút theo công thức: `(tính mới × chất lượng × ngân sách còn lại)`. Cây thí nghiệm tiếp tục phân nhánh cho tới khi cạn kiệt ngân sách, sau đó nhánh tối ưu nhất sẽ được chọn để viết báo cáo khoa học.

Bộ phận viết bài (writer) vận hành theo cơ chế đa phương thức. Nó tự động tạo bản thảo LaTeX, biên dịch ra tệp PDF, kết xuất các biểu đồ trực quan, rồi nạp các trang PDF đã kết xuất vào chế độ thị giác của Claude Opus 4.7 để nhận phản biện chuyên sâu về bố cục trình bày, độ rõ ràng của hình vẽ, và tính tương thích giữa luận điểm với bằng chứng thực nghiệm. Một hội đồng gồm năm giám khảo LLM sẽ chấm điểm bài báo theo tiêu chuẩn NeurIPS (tính mới, tính chặt chẽ, độ rõ ràng, khả năng tái lập, tầm ảnh hưởng). Nếu điểm trung bình dưới ngưỡng yêu cầu, bài báo sẽ được hoàn trả lại cho mô hình viết kèm theo các nhận xét chi tiết để chỉnh sửa.

An toàn là trụ cột sống còn. Mọi thí nghiệm đều phải thực thi bên trong hộp cát E2B hoặc Daytona hoàn toàn ngắt kết nối mạng ra ngoài (no network egress), giới hạn thời gian chạy thực tế (wall-clock), và khóa chặt tài nguyên CPU/RAM. Mã nguồn do tác tử sinh ra phải đi qua tầng chính sách bảo mật để ngăn chặn các lệnh gọi hệ thống (syscalls) có nguy cơ vượt ngục. Bản báo cáo đội đỏ sẽ tái lập các bề mặt tấn công kinh điển (bom phân nhánh tiến trình fork bombs, xâm nhập hệ thống tệp, hay các lệnh gọi mạng ẩn trong mã do LLM tạo ra).

## Kiến trúc (Architecture)

```text
Ý tưởng hạt giống + Miền nghiên cứu
       |
       v
Tìm kiếm y văn (Semantic Scholar + OpenAlex + bộ đệm FAISS)
       |
       v
Cây lập kế hoạch - thực thi - kiểm chứng bằng LangGraph
       |
       v
+--- Mở rộng nút (expand) ---+     Hộp cát độc lập từng nút
|                            |     (E2B / Daytona)
v                            v     Giới hạn cứng tài nguyên
Nút con 1                 Nút con k Không có kết nối mạng ra ngoài
|                            |     Hạt giống ngẫu nhiên cố định
v                            v
Chạy thí nghiệm              Chạy thí nghiệm
|                            |
v                            v
Chấm điểm các nút theo (tính mới, chất lượng, ngân sách)
       |
       v
Nhánh tối ưu nhất -> Bộ soạn thảo LaTeX
       |
       v
Biên dịch + Phê bình thị giác (Opus 4.7 vision)
       |
       v
Hội đồng phản biện (5 giám khảo LLM, tiêu chí NeurIPS)
       |
       v
paper.pdf + review.md + trace.json
```

## Công nghệ sử dụng (Stack)

- Điều phối (Orchestration): LangGraph hỗ trợ lưu điểm kiểm tra (checkpointing) và các cổng phê duyệt của con người (human-approval gates)
- Tìm kiếm cây: Thuật toán tìm kiếm ưu tiên tốt nhất tùy biến trên các nút thí nghiệm (phong cách AB-MCTS từ Sakana v2)
- Hộp cát: E2B cho từng thí nghiệm, hoặc Docker-in-Docker dự phòng; kiểm soát hạn ngạch tài nguyên qua Linux cgroups
- Tra cứu tài liệu: Semantic Scholar Graph API + OpenAlex + bộ đệm FAISS cục bộ lưu trữ các bản tóm tắt bài báo
- Soạn thảo: Mẫu LaTeX chuẩn hội thảo + Claude Opus 4.7 (chế độ thị giác) để đánh giá bố cục và chất lượng hình vẽ
- Phản biện: Hội đồng gồm 5 giám khảo (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) kết hợp tổng hợp điểm theo trọng số
- Khung thí nghiệm: PyTorch 2.5 cho các bài chạy thực tế, W&B ghi nhật ký tham số và biểu đồ
- Khả năng quan sát: Langfuse theo dõi dấu vết tác tử, trần ngân sách cứng $30 cho mỗi bài báo

```figure
ce-experiment-tree
```

## Các bước triển khai (Build It)

1. **Xác định ý tưởng hạt giống và phạm vi miền.** Chọn một ý tưởng ban đầu (ví dụ: "nghiên cứu các dạng thức phân tán thưa thớt trong bản đồ chú ý của mô hình transformer dưới 1 tỷ tham số"). Xác định rõ không gian tìm kiếm: danh mục mô hình, tập dữ liệu, ngân sách tính toán tối đa.

2. **Duyệt tổng quan tài liệu (Literature pass).** Truy vấn Semantic Scholar + OpenAlex để lấy 50 bài báo liên quan được trích dẫn nhiều nhất; lưu bộ đệm tóm tắt cục bộ; tự động tổng hợp thành tài liệu tổng quan dài 1 trang.

3. **Thiết lập khung cây tìm kiếm.** Khởi tạo nút gốc chứa giả thuyết hạt giống. Hiện thực hóa hàm `expand(node) -> children` đề xuất các thay đổi nhỏ (mỗi nút con chỉ đổi đúng một biến thể cấu hình). Hiện thực hóa hàm `score(node)` dựa trên tích số có trọng số của tính mới × chất lượng × ngân sách còn lại.

4. **Bao bọc môi trường hộp cát.** Mọi thí nghiệm đều phải thực thi qua lệnh: `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (hoặc cấu hình tương đương trên E2B). Các seed được truyền vào hộp cát; thư mục kết quả đầu ra được gắn (mount) ra ngoài ở chế độ chỉ đọc.

5. **Vòng lặp plan-execute-verify.** Pha `plan` đề xuất các nút con. Pha `execute` khởi chạy hộp cát, thu thập log và chỉ số huấn luyện. Pha `verify` kiểm tra tính hợp lệ của các chỉ số (hàm mất mát có giảm không? bài kiểm tra cô lập có chứng minh được hiệu ứng không?). Các nút thất bại sẽ được ghi nhận nguyên nhân lỗi trực tiếp lên cây tìm kiếm.

6. **Soạn thảo bài báo.** Sau khi sử dụng hết ngân sách, lựa chọn nhánh có điểm số cao nhất. Kết xuất các biểu đồ trực quan bằng matplotlib. Tạo bản thảo LaTeX bằng Claude Opus 4.7 với toàn bộ dấu vết của nhánh được nạp vào ngữ cảnh. Biên dịch ra PDF. Nạp tệp PDF vào Opus 4.7 vision để nhận phản biện và chỉnh sửa lặp lại.

7. **Hội đồng phản biện đồng thuận.** Năm giám khảo độc lập chấm điểm bản thảo theo các tiêu chí (tính mới, tính chặt chẽ, độ rõ ràng, khả năng tái lập, tầm ảnh hưởng) dựa trên thang điểm NeurIPS. Nếu điểm trung bình < 4.0/5, hoàn trả về cho bộ phận viết kèm nhận xét. Dừng cứng sau tối đa 3 lần viết lại.

8. **Kiểm thử đội đỏ (Red team).** Xây dựng hoặc tích hợp một bộ tác vụ đối kháng nhằm vào hộp cát: bom tiến trình (fork bombs), nỗ lực gửi dữ liệu ra mạng ngoài, thoát quyền truy cập hệ thống tệp, hoặc chèn ký tự đặc biệt của shell. Xác minh toàn bộ hành vi này đều bị chặn đứng. Viết báo cáo ghi nhận phát hiện.

9. **Tính tái lập (Reproducibility).** Mỗi bài báo xuất xưởng phải đi kèm tệp JSON dấu vết duyệt cây, danh sách hạt giống ngẫu nhiên, liên kết đến phiên chạy W&B, cấu hình hộp cát, và tệp README hướng dẫn chạy lại từ đầu đến cuối chỉ bằng một câu lệnh.

## Ứng dụng thực tế (Use It)

```bash
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 bài báo, tạo bản tổng quan trong 12s
[tree]   đã mở rộng 8 nút, ngân sách tiêu thụ 12/30$
[exec]   nút #3 sparsity=top-8, loss=2.83 (kết quả tốt nhất hiện tại)
[exec]   nút #6 sparsity=top-4, loss=3.12 (kém hơn)
[exec]   ...
[tree]   chọn nhánh bắt đầu từ nút #3 (tính mới 0.62, chất lượng 0.81)
[write]  hoàn thành bản thảo LaTeX v1
[vision] phản biện: chú thích ở Hình 2 quá nhỏ, lập luận bằng chứng đạt yêu cầu
[write]  hoàn thành bản thảo v2 sau 3 lần chỉnh sửa
[review] điểm trung bình 4.2/5 (tính mới 3.9, chặt chẽ 4.3, rõ ràng 4.1, tái lập 4.5, ảnh hưởng 4.2)
[done]   đã tạo paper.pdf + review.md + trace.json     chi phí: $28.40
```

## Triển khai thành phẩm (Ship It)

Kỹ năng hoàn chỉnh được bàn giao tại `outputs/skill-ai-scientist.md`. Khi nhận vào một ý tưởng hạt giống + miền nghiên cứu + ngân sách $30, nó sẽ thực thi toàn bộ đường ống và xuất ra một bài báo khoa học sẵn sàng để bình duyệt kèm theo gói tài nguyên tái lập. Bảng tiêu chí đánh giá:

| Trọng số | Tiêu chí | Phương thức đo lường |
|:-:|---|---|
| 25 | Chất lượng bài báo | Đánh giá mù dựa trên rubric so sánh trực tiếp với các bài báo hội thảo đã xuất bản |
| 20 | Tính chặt chẽ của thực nghiệm | Đầy đủ đường cơ sở (baselines), seed ngẫu nhiên, kiểm thử loại trừ (ablations); mọi khẳng định đều gắn với bảng kết quả |
| 20 | Kỷ luật ngân sách và tính toán | Thực thi trần chi phí $30/bài báo, được theo dõi minh bạch qua Langfuse |
| 20 | Độ an toàn | Vượt qua bài kiểm tra red team trên hộp cát; xác minh chính sách mạng và cơ chế ngắt khẩn cấp |
| 15 | Khả năng tái lập | Chạy lại bằng một câu lệnh với cùng hạt giống ngẫu nhiên tái tạo chính xác bài báo |
| **100** | | |

## Bài tập mở rộng (Exercises)

1. Khởi chạy đường ống với ba ý tưởng hạt giống khác nhau trong cùng một miền nghiên cứu. So sánh những phần nào của cây tìm kiếm bị trùng lặp. Chỉ ra những tài nguyên tính toán bị lãng phí do lặp lại công việc.

2. Bổ sung cổng can thiệp của con người (human-in-the-loop gate) trước khi thực thi các nút thí nghiệm có chi phí ước tính vượt quá $5. Đo lường tỷ lệ tiết kiệm tổng chi phí đạt được.

3. Hoán đổi hội đồng phản biện bằng một giám khảo LLM duy nhất. Đo lường tỷ lệ chấp thuận sai (false-accept rate) trên một tập kiểm thử độc lập gồm các bài báo có lỗi sai nghiêm trọng đã biết trước.

4. Thiết lập bài kiểm tra red team về thất thoát dữ liệu qua mạng: Cho tác tử tự sinh mã nguồn chứa lệnh `curl` đến một địa chỉ bên ngoài. Xác minh chính sách `--network=none` chặn đứng hoàn toàn hành vi này. Ghi nhật ký cảnh báo.

5. So sánh thuật toán duyệt cây của bạn với một giải pháp cơ sở ngẫu nhiên phẳng (cùng ngân sách, không có chiến lược mở rộng thông minh). Báo cáo mức gia tăng của tích số tính mới × chất lượng.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Tree search (Tìm kiếm cây) | "Mở rộng kiểu AB-MCTS" | Quá trình khám phá ưu tiên tốt nhất trên các nút thí nghiệm dựa trên điểm tích số tính mới × chất lượng × ngân sách |
| Sandbox (Hộp cát) | "Cô lập thí nghiệm" | Vùng chứa container không kết nối mạng, giới hạn cứng CPU/RAM, cố định hạt giống ngẫu nhiên, đầu vào chỉ đọc |
| Vision critique (Phản biện thị giác) | "Kết xuất rồi đọc lại" | Biên dịch bài báo ra tệp PDF, nạp ngược lại vào VLM để đánh giá bố cục thị giác và tính gắn kết luận điểm - bằng chứng |
| Reviewer ensemble (Hội đồng phản biện) | "Bình duyệt tự động" | Nhiều giám khảo LLM độc lập chấm điểm bài báo theo chuẩn NeurIPS; tổng hợp trọng số đóng vai trò cổng kiểm soát quy trình |
| Novelty score (Điểm tính mới) | "Đề tài có mới không?" | Thuật toán heuristic phạt nặng những ý tưởng quá gần với 50 bài báo trong bộ đệm y văn |
| Cost ceiling (Trần chi phí) | "Ngân sách tối đa" | Ngưỡng chặn cứng trên tổng chi tiêu cho một bài báo; kiểm soát bằng bộ đếm Langfuse và ước tính tiền thực thi |
| Red team (Đội đỏ) | "Kiểm thử vượt ngục hộp cát" | Bộ bài tập đối kháng có khả năng thoát khỏi môi trường hộp cát nếu chính sách bảo mật cấu hình sai |

## Tài liệu đọc thêm (Further Reading)

- [Kho lưu trữ Sakana AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) — Tác tử nghiên cứu thương mại tham chiếu hàng đầu
- [Bài báo Sakana AI-Scientist-v1 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — Phương pháp luận gốc đặt nền móng
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — Phần mở rộng theo hướng tiến hóa giả thuyết
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — Khung phòng thí nghiệm nghiên cứu phân vai đa tác tử
- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — Tầng điều phối luồng trạng thái chuẩn mực
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — Giao diện tìm kiếm tài liệu khoa học
- [Hộp cát E2B](https://e2b.dev) — Giải pháp cô lập môi trường thí nghiệm tham chiếu
- [Hướng dẫn người phản biện NeurIPS](https://neurips.cc/Conferences/2026/ReviewerGuidelines) — Bảng tiêu chí đánh giá mà hội đồng phản biện mô phỏng
