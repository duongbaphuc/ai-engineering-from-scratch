# Chuyên môn hóa vai trò — Người lập kế hoạch, Phê bình, Thực thi, Kiểm minh (Role Specialization — Planner, Critic, Executor, Verifier)

> Mô hình phân rã đa tác tử phổ biến nhất năm 2026: một tác tử lập kế hoạch, một tác tử thực thi, một tác tử phê bình hoặc kiểm minh. MetaGPT (arXiv:2308.00352) chuẩn hóa điều này thành các quy trình vận hành tiêu chuẩn (SOP - Standard Operating Procedure) được mã hóa vào các prompt vai trò — Giám đốc sản phẩm (Product Manager), Kiến trúc sư (Architect), Quản trị dự án (Project Manager), Kỹ sư (Engineer), Kỹ sư QA (QA Engineer) — tuân theo nguyên lý `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) liên kết nhà thiết kế, lập trình viên, người thẩm định, kiểm thử viên thông qua một "chuỗi trò chuyện" (chat chain) cùng cơ chế "khử ảo giác qua giao tiếp" (communicative dehallucination - các tác tử chủ động yêu cầu làm rõ chi tiết còn thiếu). Bộ kiểm minh (verifier) giữ vai trò chịu lực cốt lõi: Cemri và cộng sự (MAST, arXiv:2503.13657) chỉ ra rằng mọi lỗi hỏng đa tác tử đều có thể truy nguyên từ khâu kiểm minh bị thiếu sót hoặc hỏng hóc. PwC báo cáo mức tăng độ chính xác gấp 7 lần (10% → 70%) nhờ bổ sung các vòng lặp kiểm thực có cấu trúc trong CrewAI.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 minutes

## Vấn đề

Các hệ thống đa tác tử chung chung thường tạo ra kết quả mờ nhạt, tầm thường. Ba lập trình viên trong một nhóm chat sẽ viết ra ba biến thể của cùng một đoạn mã chất lượng trung bình. Bạn có thể thêm nhiều tác tử hơn, tăng thêm số vòng trao đổi, nhưng vẫn không thể vượt qua được ngưỡng chất lượng mong muốn.

Giải pháp không nằm ở việc bổ sung thêm số lượng tác tử — mà là ở việc sử dụng các tác tử *khác biệt*. Hãy phân bổ các vai trò chuyên biệt. Trao cho người phản biện (critic) những công cụ mà người lập kế hoạch (planner) không có. Trang bị cho bộ kiểm minh (verifier) một bộ kiểm thử khách quan (objective test suite). Khi đó, hệ thống sẽ xuất hiện sự bất đồng mang tính xây dựng nội bộ đi kèm cơ chế hiệu chỉnh có căn cứ xác thực, thay vì chỉ là những dự đoán song song vô thưởng vô phạt.

## Khái niệm

### Bốn vai trò kinh điển

**Người lập kế hoạch (Planner).** Tiếp nhận mục tiêu, tạo ra danh sách các bước hoặc một bản đặc tả yêu cầu (spec). Công cụ: truy xuất tri thức, tài liệu. Đầu ra: kế hoạch có cấu trúc.

**Người thực thi (Executor).** Đọc từng bước trong kế hoạch tại một thời điểm, tạo ra sản phẩm (artifact). Công cụ: các công cụ làm việc thực tế (trình biên dịch mã, shell, API client). Đầu ra: sản phẩm/hiện vật cụ thể.

**Người phê bình (Critic).** Đối chiếu đầu ra của người thực thi với ý định ban đầu của người lập kế hoạch. Công cụ: quyền truy cập chỉ đọc vào sản phẩm, công cụ phân tích tĩnh (static analysis). Đầu ra: chấp thuận/từ chối kèm lý do chi tiết.

**Bộ kiểm minh (Verifier).** Tiếp nhận sản phẩm và thực thi một bước kiểm tra mang tính tất định (deterministic check). Công cụ: trình chạy kiểm thử (test runner), trình kiểm tra kiểu dữ liệu (type checker), trình xác thực lược đồ (schema validator). Đầu ra: đạt/không đạt (pass/fail) kèm bằng chứng rõ ràng.

Người phê bình mang tính chủ quan, thể hiện quan điểm cá nhân và thường dựa trên LLM. Bộ kiểm minh mang tính khách quan, tất định và thường dựa trên mã nguồn thực thi. Chúng không phải là cùng một vai trò.

### Mô hình SOP của MetaGPT

MetaGPT (arXiv:2308.00352) mã hóa các quy trình vận hành tiêu chuẩn của ngành công nghệ phần mềm thành các prompt định hình vai trò:

- **Giám đốc sản phẩm (Product Manager)** viết bản tài liệu yêu cầu sản phẩm (PRD).
- **Kiến trúc sư (Architect)** thiết kế kiến trúc hệ thống.
- **Quản trị dự án (Project Manager)** phân rã và điều phối các nhiệm vụ.
- **Kỹ sư (Engineer)** hiện thực hóa mã nguồn.
- **Kỹ sư QA (QA Engineer)** thực thi các bài kiểm thử.

Mỗi vai trò đều có lược đồ dữ liệu đầu vào/đầu ra nghiêm ngặt. Prompt vai trò quy định rõ vai trò đó *là ai* và *phải tạo ra thứ gì*. Công thức `Code = SOP(Team)` — các SOP mang tính tất định sẽ biến một nhóm LLM thành một đường ống xử lý (pipeline) có thể dự đoán được.

### Cơ chế khử ảo giác qua giao tiếp của ChatDev

ChatDev bổ sung một cơ chế then chốt: khi người thực thi cần một chi tiết cụ thể không có trong kế hoạch, nó sẽ chủ động hỏi người thiết kế trước khi tiếp tục. Điều này ngăn chặn lỗi kinh điển của LLM: tự bịa đặt chi tiết một cách hợp lý bề ngoài.

Cách triển khai: prompt vai trò được bổ sung chỉ dẫn "khi bạn cần thông tin cụ thể chưa được cung cấp, hãy đích danh hỏi vai trò liên quan trước khi tạo ra đầu ra".

### Tại sao bộ kiểm minh lại quan trọng nhất?

Cemri và cộng sự (MAST) đã truy vết 1642 trường hợp lỗi thực thi đa tác tử. Có tới 21.3% xuất phát từ lỗ hổng kiểm minh (verification gaps) — hệ thống trả về câu trả lời mà chưa có bất kỳ ai kiểm tra lại. 79% còn lại thường bắt nguồn từ nguyên nhân "có một bước kiểm tra nhưng bước đó bị lỗi trong im lặng hoặc chưa từng được chạy". Kiểm minh là vai trò chịu lực của toàn bộ hệ thống.

PwC báo cáo (trong các đợt triển khai CrewAI năm 2025) rằng việc bổ sung một vòng lặp kiểm thực có cấu trúc đã nâng độ chính xác từ 10% lên 70%. Mức tăng 7 lần chỉ từ một vai trò duy nhất.

### So sánh Người phê bình và Bộ kiểm minh

- Người phê bình (Critic) là một LLM đánh giá chất lượng sản phẩm. Mang tính chủ quan. Có thể bị đánh lừa bởi văn phong nghe có vẻ xuôi tai và hợp lý.
- Bộ kiểm minh (Verifier) là một chương trình tất định thực thi trực tiếp trên sản phẩm. Mang tính khách quan. Cung cấp kết quả đạt/không đạt kèm bằng chứng thực nghiệm.

Hãy sử dụng cả hai. Người phê bình phát hiện các vấn đề về gu thiết kế mà bộ kiểm minh không thể diễn đạt thành luật. Bộ kiểm minh bắt được các lỗi mà người phê bình không thể thấy vì chúng chỉ bộc lộ khi chạy trong thực tế (runtime).

### Mẫu chống hình mẫu: Toàn-LLM (The All-LLM anti-pattern)

Tất cả các vai trò trong hệ thống của bạn đều là LLM và đầu ra của mỗi vai trò luôn là "trông có vẻ ổn đối với tôi" (looks good to me). Đây là chế độ lỗi kinh điển được chỉ ra trong nghiên cứu MAST. Hãy bổ sung ít nhất một bộ kiểm minh có kết quả đạt/không đạt được quyết định bởi mã nguồn, chứ không phải bởi một LLM.

### Ánh xạ vào các framework thực tế

- **CrewAI** — `Agent(role, goal, backstory)` là giao diện chuyên môn hóa vai trò mẫu mực.
- **LangGraph** — các nút có thể mang prompt chuyên biệt; các cạnh thực thi đường ống tuần tự.
- **AutoGen** — các `ConversableAgent` đặc thù theo vai trò với tên một từ trong một `GroupChat`.
- **OpenAI Agents SDK** — các công cụ bàn giao (handoff tools) giữa các Agent chuyên môn hóa theo vai trò.

```figure
swarm-roles
```

## Xây dựng thực tế

`code/main.py` triển khai một đường ống 4 vai trò để xây dựng một hàm Python đơn giản:

- **Planner** tạo bản đặc tả yêu cầu.
- **Executor** sinh chuỗi mã nguồn.
- **Critic** (mô phỏng bằng LLM) gắn cờ các vấn đề rõ ràng.
- **Verifier** chạy mã đã sinh trong một môi trường cô lập (`exec`) đối chiếu với trường hợp kiểm thử (test case).

Bản minh họa chạy hai lần: một lần khi người thực thi sinh mã chính xác (cả critic và verifier đều thông qua), một lần khi người thực thi sinh mã sai lệch đặc tả (critic bỏ lọt lỗi vì đoạn mã trông rất hợp lý, nhưng verifier bắt được vì bài kiểm thử thất bại).

Chạy chương trình:

```bash
python3 code/main.py
```

## Ứng dụng thực tế

`outputs/skill-role-designer.md` nhận đầu vào là một tác vụ và tạo ra danh sách vai trò (3-5 vai trò), lược đồ đầu vào/đầu ra cho từng vai trò và bài kiểm tra kiểm minh. Sử dụng công cụ này trước khi kết nối các tác tử vào một framework cụ thể.

## Đưa vào sản xuất

Danh sách kiểm tra khi triển khai:

- **Ít nhất một bộ kiểm minh tất định.** Tuyệt đối không để toàn bộ hệ thống đều là LLM.
- **Lược đồ I/O rõ ràng cho từng vai trò.** Người lập kế hoạch trả về bản đặc tả, không phải văn bản tự do; người thực thi tiếp nhận lược đồ đó.
- **Khử ảo giác qua giao tiếp.** Người thực thi bắt buộc phải hỏi lại người lập kế hoạch khi thiếu thông tin; tuyệt đối không tự bịa đặt.
- **Thứ tự thực thi giữa Phê bình và Kiểm minh.** Chạy người phê bình trước (chi phí thấp, phát hiện các lỗi thiết kế), chạy bộ kiểm minh sau (chi phí cao hơn, bắt lỗi thực thi).
- **Ngân sách vòng lặp (loop budget).** Tối đa 2 vòng sửa đổi giữa người phê bình và người thực thi trước khi chuyển tiếp lên con người can thiệp.

## Bài tập

1. Chạy `code/main.py` và quan sát cách bộ kiểm minh bắt được lỗi mà người phê bình đã bỏ lọt. Bổ sung một bước kiểm tra phân tích tĩnh (đếm số lần xuất hiện của từ khóa `return`) như một bộ kiểm minh bổ sung. Bước kiểm tra này bắt được điều gì mà kiểm thử runtime bỏ sót?
2. Bổ sung vai trò thứ 5: "nhà phân tích yêu cầu" (requirements analyst) chuyển đổi mong muốn của người dùng thành bản đặc tả sẵn sàng cho người lập kế hoạch. Những yêu cầu khử ảo giác qua giao tiếp nào cần được chuyển ngược lên vai trò này?
3. Đọc Mục 3 của bài báo MetaGPT ("Agents"). Liệt kê lược đồ đầu vào/đầu ra của từng vai trò trong 5 vai trò của MetaGPT.
4. Xem sơ đồ chuỗi trò chuyện của ChatDev (arXiv:2307.07924, Hình 3). Xác định vị trí mà cơ chế khử ảo giác qua giao tiếp giúp phá vỡ một vòng lặp mà nếu không có nó sẽ trở thành vô tận.
5. Mức tăng độ chính xác gấp 7 lần của PwC đến từ các vòng lặp kiểm minh. Hãy giả định ba tác vụ mà việc bổ sung bộ kiểm minh sẽ không mang lại hiệu quả — nơi việc kiểm tra tính đúng đắn mang tính tất định là bất khả thi hoặc có chi phí đắt đỏ đến mức không thể thực hiện.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật |
|---|---|---|
| Chuyên môn hóa vai trò (Role specialization) | "Tác tử khác nhau làm việc khác nhau" | Các system prompt riêng biệt được tinh chỉnh cho các vai trò lập kế hoạch, thực thi, phê bình, kiểm minh. |
| Mô hình SOP (SOP pattern) | "Quy trình vận hành tiêu chuẩn được mã hóa" | Cách tiếp cận của MetaGPT: các lược đồ I/O nghiêm ngặt cho từng vai trò biến một nhóm thành một đường ống xử lý. |
| Khử ảo giác qua giao tiếp (Communicative dehallucination) | "Hỏi trước khi bịa" | Cơ chế của ChatDev: người thực thi chủ động hỏi người lập kế hoạch khi thiếu chi tiết thay vì tự nghĩ ra. |
| Người phê bình (Critic) | "Người đánh giá LLM" | Người đánh giá mang tính chủ quan, thể hiện quan điểm. Bắt các lỗi về gu/phong cách. Có thể bị đánh lừa bởi văn phong hợp lý. |
| Bộ kiểm minh (Verifier) | "Kiểm tra tất định" | Kết quả đạt/không đạt dựa trên mã nguồn: trình chạy test, trình kiểm tra kiểu, trình xác thực lược đồ. Không thể bị đánh lừa. |
| Lỗ hổng kiểm minh (Verification gap) | "Không ai kiểm tra lại" | Chiếm 21.3% các lỗi trong nghiên cứu MAST. Kết quả được bàn giao mà thiếu bước kiểm tra lẽ ra đã bắt được lỗi. |
| Vòng lặp sửa đổi (Revision loop) | "Phê bình gửi trả lại" | Việc người phê bình từ chối sẽ kích hoạt người thực thi chạy lại kèm thông tin phản hồi. Cần giới hạn ngân sách vòng lặp. |
| Chống hình mẫu Toàn-LLM (All-LLM anti-pattern) | "Trông có vẻ ổn đối với tôi" | Mọi vai trò đều là LLM, không có bước kiểm tra tất định nào. Nguyên nhân gây lỗi kinh điển trong MAST. |

## Tài liệu đọc thêm

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — bài báo gốc về SOP dưới dạng prompt vai trò
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chuỗi trò chuyện + khử ảo giác qua giao tiếp
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — bảng phân loại MAST; lỗ hổng kiểm minh chiếm 21.3% nguyên nhân lỗi
- [Tài liệu CrewAI — Vai trò của tác tử](https://docs.crewai.com/en/introduction) — giao diện thiết lập vai trò trong môi trường sản xuất
