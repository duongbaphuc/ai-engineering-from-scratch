# Đàm phán và Thương lượng (Negotiation and Bargaining)

> Các tác tử đàm phán về tài nguyên, giá cả, phân bổ nhiệm vụ và các điều khoản hợp tác. Bộ đo chuẩn năm 2026 đã chứng minh rõ: NegotiationArena (arXiv:2402.05863) chỉ ra rằng LLM có thể cải thiện ~20% lợi ích thu được (payoff) thông qua việc thao túng nhân cách (persona manipulation, ví dụ "tỏ ra tuyệt vọng"); nghiên cứu "Measuring Bargaining Abilities" (arXiv:2402.15813) cho thấy vai trò người mua khó hơn người bán và việc tăng kích thước mô hình không giải quyết được vấn đề — mô hình phân tách **OG-Narrator** của họ (bộ sinh chào giá tất định + người diễn giải LLM) đã nâng tỷ lệ chốt thỏa thuận từ 26.67% lên 88.88%; Cuộc thi Đàm phán Tự động Quy mô lớn (arXiv:2503.06416) qua ~180.000 phiên đàm phán phát hiện ra rằng các tác tử **che giấu chuỗi suy luận** (chain-of-thought-concealing) giành chiến thắng bằng cách giấu kín quá trình tính toán nội bộ trước đối tác; nghiên cứu của Bhattacharya et al. (2025) trên các chỉ số Dự án Đàm phán Harvard đã xếp hạng Llama-3 hiệu quả nhất, Claude-3 quyết liệt nhất, và GPT-4 công bằng nhất. Bài học này sẽ tự tay triển khai Giao thức Mạng Hợp đồng (Contract Net Protocol - di sản FIPA từ Bài học 02), kết nối người mua/người bán theo phong cách LLM, vận hành kiến trúc phân tách OG-Narrator, và đo lường sự biến đổi của tỷ lệ chốt đơn theo từng quyết định cấu trúc.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 phút

## Vấn đề

Hai tác tử cần thống nhất về một mức giá. Nếu để chúng tự do trao đổi bằng các prompt ngôn ngữ thuần túy, các LLM trong giai đoạn 2024–2026 chốt thỏa thuận với tỷ lệ thấp đến ngạc nhiên (~27% trong các phiên thương lượng có tham số ngặt nghèo trong nghiên cứu arXiv:2402.15813). Việc tăng quy mô tham số không giải quyết được vấn đề: GPT-4 không hề vượt trội về mặt cấu trúc đàm phán so với GPT-3.5; nó chỉ diễn đạt *ngôn từ* đàm phán trôi chảy và thuyết phục hơn.

Nguyên nhân cốt rễ là các LLM đang bị nhập nhằng giữa hai nhiệm vụ khác nhau — quyết định con số chào giá và diễn giải lời lẽ cho con số đó. Mô hình OG-Narrator đã tách biệt rạch ròi hai khâu này: một bộ sinh chào giá (offer generator) tất định tính toán các bước đi bằng con số; LLM chỉ chịu trách nhiệm diễn giải thành lời. Kết quả là tỷ lệ chốt thỏa thuận tăng vọt lên ~89%.

Điều này phản ánh một phát hiện kinh điển trong lĩnh vực đa tác tử: việc tách rời cơ chế (mechanism) khỏi tầng giao tiếp (communication layer) luôn mang lại thắng lợi. Giao thức Mạng Hợp đồng (Contract Net Protocol - FIPA, 1996; Smith, 1980) là cơ chế thị trường tác vụ tham chiếu chuẩn mực. Đưa một LLM vào vị trí người diễn giải, bạn sẽ sở hữu một thị trường tác vụ hiện đại vận hành bằng sức mạnh của LLM.

## Khái niệm

### Giao thức Mạng Hợp đồng tóm lược

Giao thức Mạng Hợp đồng (Contract Net Protocol) của Reid Smith (1980): một **bên quản lý (manager)** phát thanh một **lời mời đề xuất (call for proposals - cfp)**; các **bên dự thầu (bidders)** phản hồi bằng các thông điệp **đề xuất (propose)** chứa giá chào thầu; bên quản lý chọn bên chiến thắng và gửi thông điệp **chấp thuận đề xuất (accept-proposal)** cho người thắng cùng thông điệp **từ chối đề xuất (reject-proposal)** cho những bên thua. Bên chiến thắng tiến hành thực thi công việc. Một thông điệp tùy chọn khác: **từ chối tham gia (refuse)** (bên dự thầu từ chối nộp đề xuất). Chuẩn FIPA đã quy chuẩn hóa quy trình này thành giao thức tương tác `fipa-contract-net`.

### Vì sao kiến trúc OG-Narrator giành chiến thắng

Nghiên cứu "Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) quan sát thấy:

- Các LLM thường xuyên vi phạm các quy tắc đàm phán cơ bản (đưa ra mức giá vô lý, phớt lờ vùng thỏa thuận khả thi - ZOPA của đối phương).
- Chúng thiết lập mỏ neo giá (anchoring) rất kém (chấp nhận ngay những mức giá mở đầu bất lợi; đưa ra giá phản đối mang tính hình thức thay vì có tính toán chiến lược).
- Việc tăng kích thước mô hình đơn thuần không giải quyết được: các mô hình lớn hơn chỉ tạo ra câu từ nghe có vẻ hợp lý hơn nhưng vẫn mắc cùng lỗi chiến lược.

Sự phân tách kiến trúc của OG-Narrator:

```mermaid
flowchart LR
    State["Trạng thái đàm phán<br/>(state)"] --> OG["Bộ sinh chào giá (Offer Generator)<br/>(tính toán tất định)"]
    OG -->|giá (price)| Narrator["Người kể chuyện LLM (LLM Narrator)<br/>(viết diễn ngôn tự nhiên kiểu con người)"]
    Narrator --> Msg["Thông điệp gửi đi<br/>(message)"]
```

Bộ sinh chào giá vận hành dựa trên một chiến lược đàm phán cổ điển: mô hình thương lượng Rubinstein, chiến lược Zeuthen, hoặc thuật toán ăn miếng trả miếng (tit-for-tat) đơn giản theo giá. LLM đảm nhiệm phần diễn giải. Thông điệp gửi đi chứa mức giá tất định kèm theo lời lẽ tự nhiên mang tính thuyết phục.

Tỷ lệ thành công tăng vọt bởi vì:
- Các mức giá luôn nằm trong vùng đàm phán hợp lý.
- Mỏ neo giá mang tính chiến lược, không bị cảm xúc chi phối.
- LLM được giao đúng sở trường mạnh nhất của nó: viết văn bản.

### Những phát hiện từ NegotiationArena

Bài báo arXiv:2402.05863 cung cấp bộ đo chuẩn kinh điển với các kết quả nổi bật:

- LLM có thể gia tăng lợi ích thu được ~20% bằng cách sử dụng nhân cách nhân tạo ("Tôi đang rất cần thanh lý món đồ này trước thứ Sáu") — thao túng nhân cách là một chiến thuật có thật và hiệu quả.
- Các tác tử hợp tác/công bằng rất dễ bị khai thác bởi các tác tử đối kháng; để tự vệ, tác tử bắt buộc phải chủ động lập thế phòng thủ đối trọng (counter-posturing).
- Các cặp đàm phán đối xứng hội tụ về những kết quả bất bình đẳng trong khoảng 40% các kịch bản kiểm thử.

Điều này không có nghĩa là "LLM là những nhà đàm phán tồi". Đúng hơn là "LLM đàm phán quá giống con người, bao gồm cả những điểm yếu dễ bị đối phương khai thác".

### Che giấu chuỗi suy luận (Chain-of-Thought Concealment)

Cuộc thi Đàm phán Tự động Quy mô lớn (arXiv:2503.06416) qua ~180.000 cuộc đàm phán trên nhiều chiến lược LLM cho thấy: các tác tử chiến thắng luôn giấu kín suy luận nội bộ trước đối thủ:

- Nếu một tác tử in ra dòng chữ "Tôi chỉ có thể nhượng bộ tới $75; giá bảo lưu (reservation price) của tôi là $70" vào một bảng nháp hiển thị công khai, đối thủ sẽ đọc được ngay lập tức.
- Các tác tử thắng cuộc tính toán chiến lược hoàn toàn bí mật; kênh xuất ra chỉ chứa mức giá chào và lượng ngôn từ tối thiểu cần thiết.

Đây là sự kế thừa trong năm 2026 của lý thuyết trò chơi cổ điển (nghiên cứu của Aumann năm 1976 về tính duy lý và thông tin): việc để lộ định giá riêng tư sẽ làm tổn hại trực tiếp đến lợi ích thu được. Các LLM không tự nhận thức được điều này và thường vô tư viết giá bảo lưu vào chuỗi suy luận vốn bị phơi bày cho đối phương.

Bài học kỹ thuật sống còn: tách biệt hoàn toàn ngữ cảnh nháp nội bộ (private scratchpad) khỏi ngữ cảnh thông điệp công khai (public message). Đây là yêu cầu bắt buộc, không phải tùy chọn.

### Bảng xếp hạng mô hình của Bhattacharya et al. (2025)

Dựa trên các chỉ số của Dự án Đàm phán Harvard (đàm phán theo nguyên tắc, tôn trọng BATNA, tương hỗ lợi ích):

- **Llama-3** đạt hiệu quả cao nhất trong việc chốt giao dịch (tỷ lệ thành công + lợi ích ròng).
- **Claude-3** là nhà đàm phán quyết liệt nhất (đặt mỏ neo giá cao, nhượng bộ rất muộn).
- **GPT-4** công bằng nhất (phương sai lợi ích thấp nhất giữa các cặp đấu).

Đây là ảnh chụp thực tế năm 2025. Vấn đề không phải là mô hình nào đứng đầu vào thời điểm hiện tại — mà là mỗi mô hình nền tảng đều có một phong cách đàm phán đặc thù bền vững. Các tập hợp không đồng nhất (Bài học 15) khai thác chính sự khác biệt này như một nguồn đa dạng hóa quý giá.

### Phân bổ nhiệm vụ qua Contract Net kết hợp LLM

Cách tái ứng dụng hiện đại của Contract Net trong các hệ đa tác tử LLM:

1. Tác tử quản lý phân rã một nhiệm vụ phức tạp thành các đơn vị công việc nhỏ hơn.
2. Phát thanh thông điệp `cfp` kèm mô tả nhiệm vụ tới các tác tử công nhân.
3. Mỗi công nhân phản hồi một báo giá: `(price, eta, confidence)`, trong đó chi phí có thể là token, đơn vị tính toán, hoặc tiền mặt.
4. Bên quản lý lựa chọn các đề xuất thắng thầu (đơn lẻ hoặc nhiều bên, tùy cấu trúc nhiệm vụ) và trao thầu.
5. Các công nhân bị từ chối được giải phóng để tham gia đấu thầu các nhiệm vụ khác.

Mô hình này mở rộng quy mô vượt xa 100 tác tử vì cơ chế điều phối dựa trên mô hình phát thanh - phản hồi (broadcast-and-respond), thay vì đối thoại trò chuyện đồng bộ. Nó được ứng dụng rộng rãi trong mô hình điều phối của Microsoft Agent Framework và một số triển khai LangGraph.

### Đàm phán tương tác đa bên liên quan (LLM-Stakeholders Interactive Negotiation)

Nghiên cứu tại NeurIPS 2024 giới thiệu các trò chơi tính điểm đa bên với **điểm số bí mật** và **ngưỡng chấp thuận tối thiểu**. Mỗi bên liên quan có một hàm hữu dụng riêng tư; LLM phải tự suy đoán các hàm này thông qua các thông điệp trao đổi. Đây là bước tổng quát hóa từ thương lượng song phương lên bài toán hình thành liên minh đa bên ($N$ bên). Rất quan trọng cho các thị trường tác vụ trong sản xuất với năng lực công nhân không đồng nhất.

### Nguyên tắc Vàng: Diễn giải so với Cơ chế (Narration-vs-Mechanism)

Xuyên suốt tất cả các bộ đo chuẩn đàm phán giai đoạn 2024–2026, quy tắc kỹ thuật bất biến là:

> Hãy để LLM đóng vai trò người kể chuyện. Tuyệt đối không để LLM tự tính toán giá chào.

Nếu giá chào là một con số (giá cả, thời gian hoàn thành ETA, số lượng), hãy tính toán nó một cách tất định từ trạng thái đàm phán và giao cho LLM xây dựng văn cảnh diễn đạt. Nếu giá chào là một cấu trúc đề xuất (phân rã nhiệm vụ, phân công vai trò), hãy để LLM phác thảo, nhưng phải thẩm định lại qua lược đồ dữ liệu và kiểm tra các ràng buộc trước khi gửi đi.

````figure
a5-og-narrator
````

## Xây dựng thực tế

Tệp `code/main.py` triển khai:

- `ContractNetManager`, `ContractNetTask`, `Bid` — bên quản lý + các bên dự thầu, phát thanh cfp, thu thập đề xuất, trao hợp đồng.
- `og_narrator_bargain(state, rng)` — bên mua theo mô hình OG-Narrator: chiến lược nhượng bộ tất định kiểu Zeuthen hướng dần về điểm giữa.
- `seller_response(state, rng)` — chính sách phản hồi giá chào tất định của bên bán (đóng vai trò chân lý cấu trúc cho cả hai bên).
- `naive_llm_bargain(state, rng)` — mô phỏng bên mua dùng LLM thuần túy: chọn giá với phương sai cao, thường xuyên rơi ra ngoài vùng ZOPA.
- Đo lường thực nghiệm: tỷ lệ chốt giao dịch qua 1.000 lượt thử nghiệm với các mức giá bảo lưu được lấy mẫu ngẫu nhiên mới mỗi lượt.

Chạy chương trình:

```bash
python3 code/main.py
```

Đầu ra kỳ vọng: tỷ lệ chốt đơn của LLM ngây thơ đạt ~65–75%; tỷ lệ chốt đơn của OG-Narrator đạt ~85–95%; khoảng cách 15–25 điểm phần trăm này minh chứng cho lợi thế cấu trúc của việc tách rời khâu sinh giá chào khỏi khâu diễn giải. Kèm theo một ví dụ phân bổ nhiệm vụ thị trường Contract Net với ba bên dự thầu và một nhiệm vụ.

## Sử dụng trong thực tế

Tệp `outputs/skill-bargainer-designer.md` là kỹ năng thiết kế giao thức đàm phán: xác định ai sinh giá chào (tất định hay LLM), ai diễn giải ngôn từ, cách ly bảng nháp nội bộ khỏi thông điệp công khai, và cách thức giám sát tỷ lệ chốt giao dịch liên tục.

## Đưa vào sản xuất

Danh mục kiểm tra trước khi đưa hệ thống đàm phán vào sản xuất:

- **Cô lập bảng nháp nội bộ.** Trạng thái riêng tư tuyệt đối không được phép lọt vào ngữ cảnh của đối phương. Đây là yêu cầu bất khả thương lượng.
- **Sinh giá chào tất định.** Giá cả, số lượng, thời gian ETA: tính toán bằng thuật toán, không dựa vào prompt.
- **Thẩm định mọi đề xuất gửi đến** dựa trên lược đồ dữ liệu chuẩn. Loại bỏ ngay các đề xuất nằm ngoài vùng ZOPA ngay tại ranh giới giao thức.
- **Giới hạn số vòng đàm phán.** Tối đa từ 3–5 vòng; chuyển giao cho hòa giải viên (mediator) xử lý nếu rơi vào bế tắc.
- **Đo lường tỷ lệ chốt đơn và phương sai lợi ích liên tục.** Tỷ lệ chốt đơn sụt giảm là triệu chứng báo hiệu sự trôi dạt của prompt hoặc một cuộc tấn công từ phía đối tác.
- **Ghi nhật ký mọi đề xuất bị từ chối** kèm lý do tất định. Đối với người quản lý Contract Net, các bên thua thầu cần hiểu rõ nguyên nhân thất bại.

## Bài tập thực hành

1. Chạy `code/main.py`. Xác nhận OG-Narrator đánh bại LLM ngây thơ về tỷ lệ chốt đơn. Khoảng cách chênh lệch cụ thể là bao nhiêu?
2. Triển khai kỹ thuật **cải thiện lợi ích dựa trên nhân cách** (arXiv:2402.05863) — bên mua áp dụng nhân cách "đang rất cần mua gấp trong tuần này" chỉ trong phần lời lẽ diễn giải, giữ nguyên bộ sinh giá chào. Tỷ lệ chốt đơn hoặc lợi ích thu được có biến động không?
3. Triển khai cơ chế **che giấu chuỗi suy luận**: duy trì một chuỗi bảng nháp nội bộ không được truyền sang đối phương. Điều gì sẽ xảy ra nếu bạn vô tình làm rò rỉ nó (mô phỏng bằng cách tráo đổi kênh truyền)?
4. Mở rộng Contract Net thành phiên đấu giá $N$ bên dự thầu kèm giá sàn (reserve price). Khi tất cả các mức chào giá đều vượt quá giá sàn, bên quản lý sẽ quyết định giữa mức giá thấp nhất và chất lượng cao nhất như thế nào? Bạn chọn quy tắc trao thầu nào và vì sao?
5. Đọc nghiên cứu của Bhattacharya et al. (2025) về các chỉ số Đàm phán Harvard. Triển khai hai nhà đàm phán với hai phong cách khác nhau (quyết liệt đối đầu với công bằng hợp tác). Đo lường phương sai lợi ích dưới các cặp ghép đối xứng và bất đối xứng.

## Thuật ngữ then chốt

| Thuật ngữ | Cách hiểu thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Contract Net | "Thị trường nhiệm vụ" | Smith 1980, FIPA 1996. Giao thức cfp + propose + accept/reject. Mô hình thị trường tác vụ tham chiếu chuẩn. |
| ZOPA | "Vùng có thể đạt thỏa thuận" | Khoảng giao thoa giữa mức giá tối đa người mua chấp nhận và mức giá tối thiểu người bán đưa ra. |
| BATNA | "Phương án thay thế tốt nhất" | Giải pháp dự phòng nếu cuộc đàm phán thất bại. Thiết lập nên mức giá bảo lưu của bạn. |
| OG-Narrator | "Bộ sinh giá + người kể chuyện" | Phân tách cấu trúc: tính toán giá chào tất định, LLM chỉ đảm nhận diễn giải thành lời. |
| Chiến lược Zeuthen (Zeuthen strategy) | "Nhượng bộ tối thiểu hóa rủi ro" | Bộ sinh giá chào cổ điển tự động nhượng bộ dựa trên giới hạn rủi ro xung đột. |
| Đàm phán Rubinstein | "Cân bằng chào giá luân phiên" | Mô hình lý thuyết trò chơi dành cho đàm phán vô hạn vòng có yếu tố chiết khấu thời gian. |
| Che giấu CoT (CoT concealment) | "Giấu kín suy luận" | Bên chiến thắng trong arXiv:2503.06416 lưu giữ suy luận ở bảng nháp kín; kênh công khai chỉ hiển thị giá chào. |
| Thao túng nhân cách (Persona manipulation) | "Tạo lập vị thế cảm xúc" | arXiv:2402.05863: tăng ~20% lợi ích thu được nhờ đóng vai nhân cách cấp bách/tuyệt vọng. |

## Tài liệu tham khảo

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — Bộ đo chuẩn đàm phán; các phát hiện về thao túng nhân cách và khai thác đối thủ.
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — Mô hình OG-Narrator và kết quả người mua gặp nhiều khó khăn hơn người bán.
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180k cuộc đàm phán; chiến lược che giấu chuỗi suy luận giành ưu thế tuyệt đối.
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — Trò chơi tính điểm đa bên với các hàm hữu dụng bí mật.
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — Cơ chế kinh điển, IEEE Transactions on Computers.
