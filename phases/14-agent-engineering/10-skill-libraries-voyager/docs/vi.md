# Thư viện Kỹ năng và Học tập Suốt đời (Voyager)

> Voyager (Wang et al., TMLR 2024) xem mã nguồn có thể thực thi (executable code) như một kỹ năng (skill). Các kỹ năng được đặt tên, có thể truy xuất, có thể hợp thành (composable), và được tinh chỉnh liên tục dựa trên phản hồi từ môi trường. Đây là kiến trúc tham chiếu tiêu chuẩn cho các kỹ năng trong Claude Agent SDK, hệ sinh thái skillkit, và mô hình thư viện kỹ năng (skill-library pattern) của năm 2026.

**Loại bài học:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn - stdlib)  
**Yêu cầu tiên quyết:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Khối Bộ nhớ Letta - Letta Blocks)  
**Thời lượng:** ~75 phút

## Mục tiêu Học tập

- Nêu tên ba thành phần cốt lõi của Voyager — chương trình học tự động (automatic curriculum), thư viện kỹ năng (skill library), cơ chế nhắc lặp (iterative prompting) — và vai trò của từng thành phần.
- Giải thích lý do Voyager thiết lập không gian hành động (action space) là mã nguồn thay vì các lệnh nguyên thủy (primitive commands).
- Hiện thực hóa một thư viện kỹ năng bằng thư viện chuẩn Python với các tính năng đăng ký, truy xuất, hợp thành, và tinh chỉnh dựa trên phản hồi lỗi.
- Ánh xạ mô hình thiết kế của Voyager sang các kỹ năng trong Claude Agent SDK năm 2026 và hệ sinh thái skillkit.

## Vấn đề

Những tác tử phải xây dựng lại toàn bộ năng lực từ đầu trong mỗi phiên làm việc thường mắc phải ba sai lầm nghiêm trọng:

1. **Lãng phí token.** Mỗi tác vụ đều phải khơi gợi lại toàn bộ quy trình suy luận tương tự nhau.
2. **Đánh mất tiến trình tích lũy.** Sự điều chỉnh học được trong phiên A không thể chuyển giao sang phiên B.
3. **Thất bại khi hợp thành các chuỗi tác vụ dài hạn.** Các tác vụ phức tạp đòi hỏi cấu trúc phân cấp năng lực; các lượt nhắc một lần (one-shot prompts) không thể nào biểu đạt trọn vẹn được.

Lời giải của Voyager: coi mỗi năng lực tái sử dụng như một đoạn mã nguồn có định danh được lưu trữ trong một thư viện, có thể truy xuất theo độ tương đồng, có thể hợp thành với các kỹ năng khác, và được tinh chỉnh thông qua phản hồi thực thi.

## Khái niệm

### Ba thành phần cốt lõi

Voyager (arXiv:2305.16291) cấu trúc một tác tử xoay quanh:

1. **Chương trình học tự động (Automatic curriculum).** Một bộ đề xuất dựa trên tính tò mò sẽ lựa chọn tác vụ tiếp theo dựa trên tập kỹ năng hiện tại của tác tử và trạng thái môi trường. Quá trình khám phá diễn ra theo phương pháp từ dưới lên (bottom-up).
2. **Thư viện kỹ năng (Skill library).** Mỗi kỹ năng là mã nguồn có thể thực thi. Kỹ năng mới được thêm vào thư viện khi một tác vụ được hoàn thành thành công. Các kỹ năng được truy xuất dựa trên độ tương đồng giữa câu truy vấn và mô tả kỹ năng.
3. **Cơ chế gợi ý lặp (Iterative prompting mechanism).** Khi gặp thất bại, tác tử nhận các lỗi thực thi, phản hồi từ môi trường và kết quả tự kiểm chứng (self-verification), sau đó tiến hành tinh chỉnh lại kỹ năng.

Đánh giá thực nghiệm trong thế giới Minecraft (Wang et al., 2024): đạt nhiều hơn 3.3 lần các vật phẩm độc nhất, chế tạo công cụ bằng đá nhanh hơn 8.5 lần, công cụ bằng sắt nhanh hơn 6.4 lần, và quãng đường di chuyển trên bản đồ dài hơn 2.3 lần so với các đường cơ sở. Dù các chỉ số này gắn liền với Minecraft, mô hình thiết kế đằng sau nó có thể áp dụng rộng rãi cho mọi lĩnh vực.

### Không gian hành động = Mã nguồn

Phần lớn các tác tử chỉ phát ra các lệnh thô sơ nguyên thủy. Voyager thì khác, nó sinh ra các hàm JavaScript. Một kỹ năng có dạng:

```javascript
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Nó được cấu thành từ các kỹ năng con. Được lưu trữ theo khóa mô tả và vector nhúng. Được truy xuất như một chương trình hoàn chỉnh chứ không đơn thuần là một đoạn prompt.

Đây chính là tiền thân của kỹ năng trong Claude Agent SDK năm 2026: một đoạn mã nguồn có định danh, có thể truy xuất kèm theo hướng dẫn thực thi mà tác tử sẽ nạp vào theo yêu cầu.

### Truy xuất kỹ năng (Skill retrieval)

Khi nhận tác vụ mới: "chế tạo một chiếc cuốc kim cương (diamond pickaxe)". Tác tử sẽ:

1. Tạo vector nhúng cho phần mô tả tác vụ.
2. Truy vấn thư viện kỹ năng để lấy ra top-k kỹ năng tương đồng nhất.
3. Thu hồi các hàm như `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable`, v.v.
4. Hợp thành kỹ năng mới từ các thao tác cơ sở vừa lấy được kết hợp logic mới.

Đây chính là mô hình mà tài nguyên MCP (Phase 13) và các kỹ năng của Agent SDK đang áp dụng: truy xuất trên một bề mặt tri thức/mã nguồn, được giới hạn trong phạm vi tác vụ hiện tại.

### Tinh chỉnh lặp (Iterative refinement)

Vòng lặp phản hồi của Voyager:

1. Tác tử viết mã một kỹ năng.
2. Kỹ năng được thực thi đối chiếu với môi trường.
3. Môi trường trả về một trong ba tín hiệu: `success` (thành công), `error` (kèm vết ngăn xếp stack trace), hoặc `self-verification failure` (tự kiểm chứng thất bại).
4. Tác tử viết lại kỹ năng bằng cách đưa tín hiệu phản hồi vào ngữ cảnh.
5. Lặp lại quy trình cho đến khi thành công hoặc chạm giới hạn số vòng cho phép.

Đây chính là kỹ thuật Tự Tinh chỉnh (Self-Refine - Bài học 05) được áp dụng vào tác vụ sinh mã đi kèm kiểm chứng gắn liền với môi trường thực tế. CRITIC (Bài học 05) cũng sử dụng mô hình tương tự nhưng lấy công cụ bên ngoài làm bộ kiểm chứng.

### Chương trình học và Khám phá

Mô-đun chương trình học của Voyager đề xuất các tác vụ kiểu như "xây dựng một nơi trú ẩn gần bờ hồ" dựa trên những gì tác tử đã làm được và những gì nó chưa từng thử sức. Bộ đề xuất kết hợp trạng thái môi trường và kho kỹ năng hiện có để chọn ra tác vụ nằm ngay trên ngưỡng năng lực hiện tại — điểm cân bằng lý tưởng cho việc khám phá học hỏi (exploration sweet spot).

Đối với các tác tử sản xuất, điều này chuyển thành toán tử "còn thiếu gì": dựa trên thư viện kỹ năng hiện có và miền nghiệp vụ, đâu là những kỹ năng chúng ta chưa bao quát tới? Các đội ngũ thường hiện thực hóa điều này theo phương thức thủ công thông qua các phiên đánh giá chương trình học.

### Những điểm mô hình này dễ gặp sự cố

- **Thối rữa thư viện kỹ năng (Skill library rot).** Một kỹ năng bị thêm vào 10 lần với các mô tả hơi khác nhau. Giải pháp: bổ sung cơ chế khử trùng lặp tại thời điểm ghi; việc truy xuất chỉ trả về một bản chuẩn hóa duy nhất.
- **Trôi dạt kỹ năng hợp thành (Composed-skill drift).** Kỹ năng cha phụ thuộc vào một kỹ năng con vừa được tinh chỉnh. Giải pháp: đánh phiên bản cho kỹ năng; kỹ năng cha gắn chặt với v1 sẽ không tự ý nhảy lên v3 một cách khó kiểm soát.
- **Chất lượng truy xuất suy giảm.** Truy xuất vector dựa trên mô tả kỹ năng sẽ kém hiệu quả khi quy mô thư viện vượt quá vài trăm mục. Hãy bổ sung thêm các bộ lọc thẻ (tags) và các ràng buộc cứng ("chỉ lấy kỹ năng có `category=tooling`").

```figure
voyager-skills
```

## Xây dựng Thực tế

`code/main.py` hiện thực hóa một thư viện kỹ năng bằng thư viện chuẩn Python:

- `Skill` — name, description, code (dưới dạng chuỗi), version, tags, dependencies.
- `SkillLibrary` — register (đăng ký), search (tìm kiếm dựa trên độ trùng token), compose (hợp thành với sắp xếp topo các phụ thuộc), và refine (nâng phiên bản khi cập nhật).
- Một tác tử theo kịch bản đăng ký ba kỹ năng cơ sở, hợp thành kỹ năng thứ tư, gặp lỗi thực thi, và tự động tinh chỉnh.

Chạy chương trình:

```bash
python3 code/main.py
```

Vết thực thi cho thấy các thao tác ghi thư viện, truy xuất, hợp thành, một lần thực thi thất bại, và một lượt tinh chỉnh lên phiên bản v2 — thể hiện trọn vẹn vòng lặp của Voyager từ đầu đến cuối.

## Ứng dụng Thực tế

- **Các kỹ năng trong Claude Agent SDK** (Anthropic) — chuẩn tham chiếu năm 2026: mỗi kỹ năng bao gồm mô tả, mã nguồn và hướng dẫn thực thi; được tải theo yêu cầu trong phiên làm việc của tác tử.
- **skillkit** (npm: skillkit) — quản lý kỹ năng chéo tác tử dành cho hơn 32 tác tử lập trình AI khác nhau.
- **Thư viện kỹ năng tùy chỉnh** — chuyên biệt theo miền (kỹ năng SQL cho tác tử dữ liệu, kỹ năng Terraform cho tác tử hạ tầng). Mô hình của Voyager có thể tinh gọn tùy biến rất linh hoạt.
- **`tools` trong OpenAI Agents SDK** — ở mức độ đơn giản hơn; mỗi công cụ đóng vai trò như một kỹ năng tinh gọn.

## Đóng gói Sản phẩm

`outputs/skill-skill-library.md` tạo ra một thư viện kỹ năng theo cấu trúc Voyager tích hợp sẵn các cơ chế đăng ký, truy xuất, đánh phiên bản, và tinh chỉnh cho bất kỳ môi trường thực thi đích nào.

## Bài tập

1. Thêm bộ phát hiện chu trình phụ thuộc (dependency-cycle detector) vào phương thức `compose()`. Điều gì xảy ra khi kỹ năng A phụ thuộc vào B và B lại phụ thuộc vào A? Nên báo lỗi nghiêm trọng (error) hay cảnh báo (warning)?
2. Hiện thực hóa việc ghim phiên bản (version pinning) cho từng kỹ năng. Khi một kỹ năng cha kết hợp kỹ năng con `crafting@1`, việc tinh chỉnh nâng cấp lên `crafting@2` không được phép âm thầm thay đổi hành vi của kỹ năng cha.
3. Thay thế cơ chế truy xuất trùng lặp token bằng các vector nhúng sentence-transformers (hoặc một bản triển khai BM25 bằng thư viện chuẩn). Đo lường retrieval@5 trên một thư viện giả lập gồm 50 kỹ năng.
4. Bổ sung một tác tử "chương trình học" (curriculum agent): dựa trên thư viện hiện tại và mô tả miền nghiệp vụ, hãy đề xuất 5 kỹ năng còn thiếu. Thiết lập gọi tác tử này theo định kỳ hàng tuần.
5. Đọc tài liệu về kỹ năng trong Claude Agent SDK của Anthropic. Chuyển đổi thư viện đồ chơi sang cấu trúc schema của SDK. Khả năng phát hiện kỹ năng (discoverability) thay đổi như thế nào?

## Thuật ngữ Then chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Kỹ năng (Skill) | "Năng lực tái sử dụng" | Đoạn mã nguồn có định danh kèm mô tả, có thể truy xuất theo độ tương đồng |
| Thư viện kỹ năng (Skill library) | "Bộ nhớ cách thức thực hiện của tác tử" | Kho lưu trữ kỹ năng bền vững, có thể tìm kiếm và hợp thành |
| Chương trình học (Curriculum) | "Bộ đề xuất tác vụ" | Bộ sinh mục tiêu từ dưới lên dựa trên khoảng trống năng lực hiện tại |
| Hợp thành (Composition) | "Đồ thị DAG kỹ năng" | Kỹ năng gọi các kỹ năng khác; được sắp xếp topo khi thực thi |
| Tinh chỉnh lặp (Iterative refinement) | "Vòng lặp tự sửa sai" | Phản hồi môi trường + lỗi + tự kiểm chứng được đưa trở lại phiên bản tiếp theo |
| Không gian hành động là mã nguồn (Action-space-as-code) | "Hành động theo chương trình" | Phát ra các hàm thay vì các lệnh thô sơ, phục vụ các hành vi kéo dài theo thời gian |
| Khử trùng lặp khi ghi (Dedup on write) | "Thu gọn kỹ năng" | Gộp các mô tả gần như trùng lặp thành một kỹ năng chuẩn tắc duy nhất |

## Tài liệu Tham khảo

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — Bài báo gốc về thư viện kỹ năng
- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — Kỹ năng dưới dạng sản phẩm thương mại hóa năm 2026
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — Kỹ năng và tác tử phụ trong thực tế
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Vòng lặp tinh chỉnh nền tảng đằng sau Voyager
