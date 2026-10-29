# Tác tử Tạo sinh và Mô phỏng Hành vi Tự phát sinh (Generative Agents and Emergent Simulation)

> Nghiên cứu đột phá của Park et al. (2023, UIST '23, arXiv:2304.03442) đã thiết lập thị trấn giả lập **Smallville** gồm 25 tác tử vận hành trên kiến trúc ba thành phần cốt lõi: **luồng ký ức** (memory stream - nhật ký ngôn ngữ tự nhiên), **suy tưởng** (reflection - các tổng hợp bậc cao do tác tử tự sinh ra về luồng ký ức của chính mình), và **kế hoạch** (plan - hành vi ở cấp độ ngày, sau đó phân rã thành các kế hoạch con). Kết quả mang tính bước ngoặt là sự tự phát sinh của bữa tiệc Ngày Valentine: chỉ một tác tử duy nhất được gieo mầm ý định "muốn tổ chức một bữa tiệc Valentine", không hề có bất kỳ kịch bản định sẵn nào khác, lời mời đã tự động lan truyền qua mạng lưới cư dân, các cuộc hẹn được sắp xếp nhịp nhàng, và bữa tiệc đã thực sự diễn ra — từ 24 tác tử hoàn toàn không biết gì về bữa tiệc lúc ban đầu. Các thử nghiệm triệt tiêu (ablation studies) chứng minh rằng cả ba thành phần đều bắt buộc phải có để duy trì độ chân thực (believability). Những dạng lỗi được ghi nhận chủ yếu là lỗi chuẩn mực không gian (bước vào các cửa hàng đã đóng cửa, cùng dùng chung nhà vệ sinh đơn). Đây chính là kiến trúc tham chiếu chuẩn mực cho việc mô phỏng tác tử và đánh giá xã hội đa tác tử vào năm 2026.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 phút

## Vấn đề

Đa số các hệ đa tác tử hiện nay là những đội ngũ được lập trình theo kịch bản chặt chẽ: người lập kế hoạch lên phương án, lập trình viên viết mã, người đánh giá duyệt mã. Mô hình này vận hành xuất sắc cho các tác vụ được phân định ranh giới rõ ràng. Tuy nhiên, nó hoàn toàn bất lực trong việc tái hiện những hành vi tự phát sinh (emergent behaviors), không theo kịch bản vốn chỉ xuất hiện khi các tác tử sở hữu ký ức dài hạn, mục tiêu ưu tiên và một thế giới mở. Các nghiên cứu khoa học, mô phỏng xã hội học, và đặc biệt là AI trong trò chơi điện tử đang rất cần dạng thức thứ hai này.

Kiến trúc Smallville chính là thước đo chuẩn mực cho bài toán này. Trước cột mốc Park (2023), các mô phỏng tác tử tốt nhất chỉ dừng lại ở mức tuân theo các kịch bản nông cạn; sau nghiên cứu này, mô hình ba thành phần trở thành lựa chọn mặc định cho các tác tử tạo sinh trong thế giới mở. Nếu bạn xây dựng một hệ mô phỏng tác tử vào năm 2026, bạn hoặc là đang áp dụng ba thành phần của Smallville, hoặc là phải giải trình tường minh lý do vì sao không sử dụng chúng.

## Khái niệm

### Ba thành phần cốt lõi

**Luồng ký ức (Memory stream).** Một nhật ký chỉ ghi thêm (append-only log) lưu giữ các quan sát (observations), hành động (actions), suy tưởng (reflections), và kế hoạch (plans). Mỗi mục nhập mang một dấu thời gian, một loại sự kiện, một mô tả bằng ngôn ngữ tự nhiên, cùng các siêu dữ liệu phái sinh: **tính gần đây (recency)**, **độ quan trọng (importance)** (được tác tử tự chấm điểm từ 1–10), và **độ phù hợp (relevance)** (độ tương đồng cosine đối với truy vấn hiện tại).

```text
[2026-02-14 09:12:03] observation: Isabella Rodriguez hỏi tôi có thích nhạc jazz không
[2026-02-14 09:14:22] reflection:   Tôi rất thích những cuộc trò chuyện sâu sắc về âm nhạc
[2026-02-14 10:05:00] plan:         Tham dự bữa tiệc Valentine của Isabella tối nay
```

Thuật toán truy xuất ký ức kết hợp cả ba trọng số: $\text{score} = w_{\text{recency}} \cdot e^{-\text{decay} \cdot \text{age}} + w_{\text{importance}} \cdot \text{importance} + w_{\text{relevance}} \cdot \cos(\theta)$. Top-$k$ mục nhập có điểm số cao nhất sẽ được đưa vào prompt ngữ cảnh hiện tại.

**Suy tưởng (Reflection).** Định kỳ (sau mỗi $N$ ký ức hoặc khi xuất hiện các sự kiện quan trọng), tác tử sẽ sinh ra các tổng hợp bậc cao từ các ký ức gần nhất. Các mục suy tưởng này được nạp ngược trở lại vào luồng ký ức và có thể được truy xuất như bất kỳ ký ức thông thường nào khác. Đây chính là cơ chế giúp tác tử hình thành "sự thấu hiểu" — tương đương với các niềm tin dài hạn (long-term beliefs) trong nhận thức.

**Kế hoạch (Plan).** Phân rã từ trên xuống dưới (top-down decomposition). Đầu tiên là kế hoạch tổng quan cấp độ ngày ("đi làm, ăn tối với Klaus"). Sau đó là kế hoạch chi tiết từng giờ. Cuối cùng là các hành động cụ thể từng bước. Kế hoạch có thể điều chỉnh linh hoạt: khi một quan sát mới mâu thuẫn với kế hoạch đã định, tác tử sẽ tự động lập lại kế hoạch cho phân đoạn bị ảnh hưởng.

### Vì sao cả ba thành phần đều sống còn (Ablation)

Park et al. đã thực hiện các thử nghiệm triệt tiêu bằng cách lần lượt lược bỏ từng thành phần quan sát, suy tưởng và kế hoạch. Mỗi sự thiếu vắng đều làm sụt giảm nghiêm trọng độ chân thực:

- Thiếu **quan sát (observation)**: tác tử bị mất ngữ cảnh thực tế và hành động dựa trên những niềm tin đã lỗi thời.
- Thiếu **suy tưởng (reflection)**: tác tử không thể hình thành các niềm tin bậc cao; mọi tương tác dừng lại ở mức nông cạn, máy móc.
- Thiếu **kế hoạch (plan)**: hành vi bị biến thành phản ứng ngẫu nhiên tức thời; các mục tiêu dài hạn bị phân rã hoàn toàn.

Điểm số đánh giá độ chân thực từ người thẩm định đạt mức cao nhất khi hội tụ đủ cả ba thành phần; việc lược bỏ bất kỳ yếu tố nào đều dẫn tới sự suy giảm rõ rệt về mặt định lượng.

### Hiện tượng tự phát sinh bữa tiệc Valentine

Một tác tử duy nhất, Isabella Rodriguez, được gieo một mục tiêu ban đầu: "muốn tổ chức một bữa tiệc Ngày Valentine tại Hobbs Cafe vào lúc 5 giờ chiều ngày 14 tháng 2". 24 tác tử còn lại hoàn toàn không được cài đặt thông tin này. Qua các ngày mô phỏng:

1. Kế hoạch của Isabella tự động sinh ra hành động mời mọi người xung quanh.
2. Mỗi lời mời trở thành một quan sát mới trong luồng ký ức của người hàng xóm.
3. Suy tưởng của người hàng xóm hình thành niềm tin: "Isabella đang tổ chức một bữa tiệc".
4. Kế hoạch của người hàng xóm tự động cập nhật: "tham dự bữa tiệc vào ngày 14 tháng 2".
5. Các hàng xóm tiếp tục truyền tai nhau lời mời này. Lời mời lan truyền khắp thị trấn mà không cần bất kỳ bộ điều phối trung tâm nào.
6. Đến đúng 5 giờ chiều ngày 14 tháng 2, nhiều tác tử cùng hội tụ tại Hobbs Cafe.

Đây chính là sự tự phát sinh (emergence) theo đúng định nghĩa khoa học: hành vi ở cấp độ toàn hệ thống (một bữa tiệc cộng đồng) đã tự hình thành từ các tương tác cục bộ (lời mời song phương + lập kế hoạch cá nhân) mà không hề có sự can thiệp của một chỉ huy trung ương.

### Các dạng lỗi thường gặp đã được ghi nhận

Nghiên cứu của Park et al. đã chỉ rõ những điểm yếu điển hình:

- **Lỗi chuẩn mực không gian (Spatial norm errors).** Tác tử tự nhiên bước vào các cửa hàng đã đóng cửa. Tác tử cố tình dùng chung một nhà vệ sinh đơn đang có người. Tác tử ngồi ăn ở những căn phòng không dành cho việc ăn uống. Mô hình ngôn ngữ không thể tự suy luận ra các chuẩn mực vật lý - xã hội nếu chỉ nhìn vào môi trường hình học thuần túy.
- **Tràn bộ nhớ (Memory overflow).** Trong các phiên mô phỏng dài, chi phí truy xuất ký ức phình to nhanh chóng. Giải pháp thực tế: định kỳ nén bộ nhớ (tóm tắt và cắt tỉa - summarize-and-prune) và áp dụng cơ chế phân rã (decay) cho các ký ức có độ quan trọng thấp.
- **Ảo giác trong suy tưởng (Reflection hallucination).** Các suy tưởng có thể tự thêu dệt nên những mối quan hệ hoàn toàn không có thực trong luồng ký ức. Biện pháp giảm thiểu: đưa mã định danh của các ký ức nguồn vào prompt suy tưởng và xác thực chéo tại thời điểm truy xuất.

Đây là những họ lỗi mang tính sản xuất: bất kỳ hệ thống mô phỏng tác tử nào năm 2026 cũng đều phải đối mặt với chúng.

### Các nguyên tắc triển khai kiến trúc ba thành phần

1. **Ký ức là chỉ ghi thêm (append-only).** Tuyệt đối không chỉnh sửa đè lên một mục ký ức cũ. Mọi đính chính đều phải là mục nhập mới.
2. **Chấm điểm độ quan trọng với chi phí thấp.** Gọi LLM để chấm điểm độ quan trọng từ 1–10 ngay tại thời điểm ghi ký ức. Lưu bộ nhớ đệm (cache) điểm số này.
3. **Truy xuất theo thứ hạng, không dùng bộ lọc cứng.** Chọn top-$k$ theo tổng điểm kết hợp; tránh dùng các bộ lọc cứng (hard filters) vì dễ làm mất ngữ cảnh quý giá.
4. **Suy tưởng chạy theo chu kỳ.** Kích hoạt tiến trình suy tưởng khi tổng điểm quan trọng của các ký ức chưa xử lý vượt qua một ngưỡng nhất định (ví dụ: 150 điểm).
5. **Kế hoạch có thể tái cấu trúc.** Khi một quan sát mới làm đảo lộn kế hoạch, chỉ tái sinh phân đoạn kế hoạch bị ảnh hưởng, không lập lại toàn bộ từ đầu.

### Tác tử tạo sinh ngoài phạm vi Smallville

Các nghiên cứu tiếp nối trong giai đoạn 2024–2026 đã mở rộng kiến trúc này sang nhiều lĩnh vực:

- **Mô phỏng xã hội đa tác tử cho nghiên cứu chính sách / thị trường.** Các cộng đồng kiểu Smallville mô phỏng phản ứng của người dùng trước các tính năng sản phẩm mới. Tốc độ nhanh hơn nhiều so với thử nghiệm A/B; tuy nhiên độ chính xác vẫn cần kiểm chứng cẩn trọng.
- **NPC thông minh trong trò chơi (Game AI).** Các tựa game nhập vai RPG tích hợp tác tử Smallville giúp cốt truyện tự phát sinh một cách sống động thay vì các chuỗi nhiệm vụ lập trình sẵn.
- **Bộ đo chuẩn đánh giá tác tử tạo sinh.** Thay vì đo độ chính xác tác vụ đơn thuần, thước đo chuyển dịch sang độ chân thực (believability) và tính mạch lạc (coherence) của hành vi qua các phiên vận hành dài hạn.

Kiến trúc Smallville là bản thiết kế tham chiếu. Các phần mở rộng có thể hoán đổi các khối thành phần (dùng cơ sở dữ liệu vector cho bộ nhớ, suy tưởng tăng cường truy xuất, kế hoạch kết hợp biểu tượng - neurosymbolic plan) nhưng vẫn luôn giữ nguyên cấu trúc ba chân kiềng cốt lõi này.

### Tầm quan trọng đối với kỹ nghệ đa tác tử

Smallville là bằng chứng thực nghiệm cho thấy hành vi tự phát sinh trong hệ đa tác tử hoàn toàn có thể đạt được với chi phí hợp lý nếu các khối cấu trúc được thiết kế đúng đắn. Kiến trúc này hiện đã được tái lập thành công trên các mô hình mã nguồn mở (các LLM nhỏ hơn suy giảm độ chân thực một cách từ từ, không bị sụp đổ đột ngột). Bất kỳ hệ thống sản xuất nào hướng tới **hành vi xã hội tự phát sinh** đều áp dụng cấu trúc này. Ngược lại, hệ thống nào đòi hỏi **thực thi tác vụ nghiêm ngặt** sẽ sử dụng các mô hình giám sát viên / chuyên môn hóa vai trò / các khối nguyên thủy đã học ở phần trước của phase.

````figure
a5-memory-reflection
````

## Xây dựng thực tế

Tệp `code/main.py` triển khai đầy đủ ba thành phần bằng Python tiêu chuẩn (stdlib) với các chính sách tác tử theo kịch bản (không cần phụ thuộc vào LLM thực tế). Bản demo tái hiện lại sự tự phát sinh bữa tiệc Valentine ở quy mô thu nhỏ:

- `MemoryStream` — nhật ký chỉ ghi thêm với thuật toán truy xuất kết hợp độ gần đây / độ quan trọng / độ liên quan.
- `reflect(stream)` — tiến trình suy tưởng theo kịch bản trên các ký ức quan trọng gần nhất.
- `plan(agent_state)` — kế hoạch cấp ngày và cấp giờ dựa trên các niềm tin hiện tại.
- Kịch bản: 5 tác tử. Tác tử 1 bắt đầu với mục tiêu "tổ chức tiệc lúc 5 giờ chiều". Qua các bước mô phỏng (ticks), lời mời lan tỏa và các tác tử cùng hội tụ.

Chạy chương trình:

```bash
python3 code/main.py
```

Đầu ra kỳ vọng: dấu vết mô phỏng chi tiết từng bước. Đến bước cuối cùng, ít nhất 3 trong số 5 tác tử xuất hiện kế hoạch dự tiệc trong lịch trình và cùng tập hợp tại địa điểm diễn ra bữa tiệc. Chỉ một hạt mầm ban đầu đã tạo nên sự phối hợp đồng bộ mà không cần bất kỳ bộ điều phối trung tâm nào.

## Sử dụng trong thực tế

Tệp `outputs/skill-simulation-designer.md` là kỹ năng thiết kế một hệ thống mô phỏng tác tử tạo sinh: xác định số lượng tác tử, lược đồ ký ức, tần suất suy tưởng, tầm nhìn kế hoạch, và các chỉ số đo lường độ chân thực.

## Đưa vào sản xuất

Các quy tắc triển khai mô phỏng trong môi trường sản xuất:

- **Bộ nhớ chính là cơ sở dữ liệu.** Hãy chọn một giải pháp lưu trữ chuyên dụng (Vector DB, PostgreSQL) khi triển khai quy mô lớn. Danh sách trong bộ nhớ RAM của stdlib chỉ phù hợp cho nguyên mẫu thử nghiệm.
- **Ghi vết tiến trình truy xuất.** Đối với mỗi hành động, hãy ghi nhật ký top-$k$ ký ức đã dẫn dắt tới quyết định đó. Đây là chìa khóa vàng để gỡ lỗi hệ thống.
- **Quản lý ngân sách token cho từng tác tử.** Mỗi chu kỳ truy xuất + suy tưởng + lập kế hoạch tiêu tốn $O(k)$ lượt gọi LLM. Công thức: $N \text{ tác tử} \times T \text{ chu kỳ} \times \text{lượt gọi/chu kỳ}$ có thể nhanh chóng thổi bay ngân sách dự án.
- **Định kỳ nén bộ nhớ.** Tóm tắt và cắt tỉa các mục nhập có độ quan trọng thấp. Chính sách lưu giữ dữ liệu là một quyết định kiến trúc then chốt, không phải chi tiết phụ.
- **Chủ động phát hiện vi phạm chuẩn mực không gian / xã hội.** Kiến trúc không thể tự học được các chuẩn mực này nếu không được cài đặt các bộ kiểm tra tường minh.

## Bài tập thực hành

1. Chạy `code/main.py`. Xác nhận có ít nhất 3 tác tử cùng hội tụ tại bữa tiệc. Tăng số lượng tác tử lên 10 — hiện tượng tự phát sinh có còn diễn ra thành công không?
2. Thử loại bỏ bước suy tưởng (reflection). Hành vi của các tác tử thay đổi ra sao? Đối chiếu hiện tượng này với kết quả thử nghiệm triệt tiêu trong bài báo Park 2023.
3. Thiết lập thêm một mục tiêu cạnh tranh ("Klaus muốn tổ chức buổi thuyết trình nghiên cứu cũng vào lúc 5 giờ chiều"). Các tác tử sẽ chia rẽ thành hai nhóm hay một mục tiêu sẽ áp đảo hoàn toàn? Yếu tố nào quyết định kết quả đó?
4. Thêm ràng buộc không gian: quán cà phê Hobbs Cafe chỉ chứa tối đa 4 tác tử. Hệ thống mô phỏng có xử lý tình trạng quá tải một cách mượt mà không, hay nó sẽ rơi vào bế tắc như trường hợp "nhà vệ sinh đơn"?
5. Đọc Mục 6 (thực nghiệm hành vi tự phát sinh) trong bài báo của Park et al. (arXiv:2304.03442). Xác định một hành vi không thể tái hiện trong mô hình thu nhỏ của bạn. Bạn cần nâng cấp thành phần nào của kiến trúc để hiện thực hóa hành vi đó?

## Thuật ngữ then chốt

| Thuật ngữ | Cách hiểu thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Luồng ký ức (Memory stream) | "Nhật ký của tác tử" | Nhật ký chỉ ghi thêm ghi lại quan sát, hành động, suy tưởng, và kế hoạch. |
| Tính gần đây (Recency) | "Ký ức mới đến mức nào" | Điểm số phân rã theo hàm mũ dựa trên độ tuổi của ký ức. |
| Độ quan trọng (Importance) | "Tác tử bận tâm nhiều hay ít" | Điểm số tự chấm từ 1–10 tại thời điểm ghi ký ức. Được lưu trong bộ nhớ đệm. |
| Độ phù hợp (Relevance) | "Liên quan thế nào tới truy vấn" | Độ tương đồng cosine giữa vector nhúng của ký ức và truy vấn hiện tại. |
| Suy tưởng (Reflection) | "Niềm tin nhận thức bậc cao" | Bản tổng hợp trừu tượng được sinh ra từ các ký ức gần đây, được nạp lại như một ký ức mới. |
| Kế hoạch (Plan) | "Phân rã ngày/giờ/hành động" | Cây kế hoạch từ trên xuống dưới. Tự động tái lập khi quan sát thực tế mâu thuẫn. |
| Smallville | "Môi trường giả lập của Park 2023" | Môi trường mô phỏng 25 tác tử tạo sinh nơi bữa tiệc Valentine tự phát sinh. |
| Độ chân thực (Believability) | "Chỉ số chất lượng" | Điểm số do người đánh giá chấm cho mức độ hợp lý, tự nhiên giống con người của tác tử. |

## Tài liệu tham khảo

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Bài báo gốc thiết lập kiến trúc tham chiếu.
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — Kỷ yếu xuất bản chính thức.
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — Kho mã nguồn Python tham chiếu.
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — Tiền đề lịch sử cho các tác tử có cấu trúc ký ức.
