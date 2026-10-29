# Nền kinh tế Tác tử, Cơ chế Khuyến khích bằng Token và Danh tiếng (Agent Economies, Token Incentives, Reputation)

> Các tác tử tự trị có tầm chân trời dài (đường cong làm việc từ 1 đến 8 giờ theo đánh giá của METR) đòi hỏi phải có năng lực kinh tế tự chủ (economic agency). **Ngăn xếp 5 tầng (5-layer stack)** mới nổi bao gồm: **DePIN** (hạ tầng tính toán vật lý) $\rightarrow$ **Danh tính (Identity)** (W3C DID + vốn danh tiếng) $\rightarrow$ **Nhận thức (Cognition)** (RAG + MCP) $\rightarrow$ **Thanh toán (Settlement)** (trừu tượng hóa tài khoản - account abstraction) $\rightarrow$ **Quản trị (Governance)** (các tổ chức tự trị phi tập trung điều hành bằng tác tử - Agentic DAO). Các mạng lưới khuyến khích tác tử trong môi trường sản xuất gồm có **Bittensor** (các mạng con subnet TAO trao thưởng cho các mô hình phục vụ tác vụ chuyên biệt), **Fetch.ai / ASI Alliance** (LLM ASI-1 Mini + token FET), và **Gonka** (cơ chế bằng chứng công việc PoW dựa trên transformer giúp tái phân bổ năng lực tính toán cho các tác vụ AI hữu ích). Trong giới học thuật: nghiên cứu LaMAS phi tập trung tại AAMAS 2025 sử dụng **phân bổ đóng góp theo giá trị Shapley (Shapley-value credit attribution)** để phân chia phần thưởng công bằng cho các tác tử tham gia; nhóm Google Research với bài viết "Thiết kế cơ chế cho các mô hình ngôn ngữ lớn" đề xuất **các phiên đấu giá token** với cơ chế thanh toán giá thứ hai (second-price payment) dưới phép tổng hợp đơn điệu. Bài học này xây dựng một thị trường tác tử tối giản, áp dụng phân bổ đóng góp theo giá trị Shapley vào đường ống đa tác tử và vận hành một phiên đấu giá token giá thứ hai để các cỗ máy lý thuyết trò chơi được hiện thực hóa một cách cụ thể.

**Type:** Learn  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)  
**Time:** ~75 phút  

## Vấn đề (Problem)

Các hệ thống đa tác tử trở nên phức tạp khi các tác tử cùng nhau phối hợp tạo ra giá trị chung nhưng lại cần được trao thưởng một cách riêng lẻ. Các cơ chế cổ điển — chia đều bình quân, hoặc tác tử đóng góp cuối cùng gom toàn bộ phần thưởng — đều thể hiện sự bất công hoặc dễ bị gian lận (gameable). Cơ chế trao thưởng dựa trên liên minh thông qua giá trị Shapley đảm bảo tính công bằng ngay từ bản chất lý thuyết nhưng lại đòi hỏi chi phí tính toán cực lớn. Các công trình nghiên cứu giai đoạn 2025-2026 đã đưa ra các phương pháp xấp xỉ hữu ích: lấy mẫu Shapley (Shapley sampling), đấu giá tổng hợp đơn điệu, và danh tiếng trên chuỗi (on-chain reputation) được tích lũy từ các đóng góp đã qua xác thực.

Vượt ra ngoài bài toán phân bổ công trạng, lĩnh vực này đã chuyển dịch sang các tác tử kinh tế thực thụ: Bittensor TAO trao thưởng cho năng lực tính toán đào coin để tinh chỉnh các mô hình chuyên biệt cho từng mạng con (subnet), Fetch.ai/ASI thưởng token FET cho việc sử dụng LLM ASI-1 Mini, và Gonka phân bổ lại bằng chứng công việc của transformer cho các tác vụ AI thực tế. Các tác tử có khả năng tự chủ giao dịch tài chính đã tồn tại ngay hôm nay; vấn đề đặt ra là làm thế nào để đồng chỉnh các cơ chế khuyến khích (align incentives).

Bài học này tiếp cận nền kinh tế tác tử như một nhóm vấn đề cụ thể — phân bổ công trạng, thiết kế cơ chế và hệ thống danh tiếng — đồng thời hiện thực hóa từng khái niệm với lượng toán học tối giản để giúp bạn nắm vững cốt lõi vấn đề.

## Khái niệm (Concept)

### Ngăn xếp 5 tầng của nền kinh tế tác tử

1. **DePIN (Hạ tầng tính toán vật lý phi tập trung):** Cơ sở hạ tầng phi tập trung cho thuê GPU, lưu trữ và băng thông. Điển hình như các subnet của Bittensor, Render Network, Akash. Tầng này không chỉ dành riêng cho tác tử; tác tử sử dụng nó như tài nguyên nền tảng.
2. **Danh tính (Identity):** Định danh phi tập trung theo chuẩn W3C (Decentralized Identifiers - DID) cấp cho mỗi tác tử một mã định danh bền vững, độc lập với bất kỳ nền tảng nào. Điểm số danh tiếng được gắn liền vào DID này. Giao thức Mạng Tác tử (ANP - Agent Network Protocol) sử dụng DID làm tầng khám phá dịch vụ.
3. **Nhận thức (Cognition):** Vòng lặp lập luận của tác tử: LLM + RAG + MCP. Đây là phần cốt lõi được xây dựng trong các Phase học tập khác.
4. **Thanh toán (Settlement):** Trừu tượng hóa tài khoản (Account Abstraction - chuẩn ERC-4337) cho phép các tác tử tự chi trả phí giao dịch (gas) từ số dư tài khoản của chính mình mà không nhất thiết phải nắm giữ ETH. Tác tử có thể trả tiền mua dịch vụ, thanh toán cho nhau hoặc thuê tài nguyên tính toán.
5. **Quản trị (Governance):** Các DAO do tác tử tham gia (Agentic DAO): các cấu trúc quản trị nơi cả con người *và* tác tử cùng tham gia biểu quyết thay đổi giao thức, với quyền lực biểu quyết gắn liền với điểm số danh tiếng.

Không phải hệ thống thực tế nào cũng triển khai đủ cả năm tầng. Bittensor sử dụng tầng 1, 2, một phần tầng 3, một phần tầng 4, và chưa dùng tầng 5. Các tác tử của OpenAI không sử dụng tầng nào ngoài tầng 3. Ngăn xếp này là một bản đồ tham chiếu chứ không phải yêu cầu mang tính bắt buộc.

### Bittensor, Fetch.ai, Gonka — Những hệ thống đang vận hành

**Bittensor (TAO):** Các mạng con (subnets) đại diện cho các tác vụ chuyên biệt (mô hình hóa ngôn ngữ, sinh hình ảnh, dự báo). Các thợ đào (miners) gửi đầu ra của mô hình. Các bên xác thực (validators) chấm điểm xếp hạng; cơ chế cho điểm theo trọng số cổ phần (stake-weighted scoring) sẽ phân phối phần thưởng token TAO. Mỗi subnet sở hữu bộ tiêu chuẩn đánh giá riêng. Bài học kinh tế rút ra: hãy trả tiền cho chất lượng đầu ra của tác vụ cụ thể, không trả tiền cho lượng tài nguyên tính toán đã tiêu hao.

**Fetch.ai / ASI Alliance:** LLM ASI-1 Mini vận hành trên mạng lưới của Fetch.ai; người dùng thanh toán token FET để thực hiện suy luận. Mô hình tác tử ngang hàng (agents-as-peers) ở đây rõ nét hơn: một tác tử trên Fetch có thể gọi một tác tử khác để giải quyết một tác vụ và thanh toán chi phí bằng FET.

**Gonka:** Bằng chứng công việc trên kiến trúc transformer (Transformer proof-of-work): "công việc" ở đây là các lượt lan truyền xuôi (forward passes) của một transformer. Thợ đào kiếm thu nhập bằng cách chạy các tác vụ suy luận đã biết trước kết quả chính xác (lấy từ dữ liệu huấn luyện). Đây là cơ chế PoW tạo ra giá trị tài nguyên thực tế thay vì PoW băm hàm vô nghĩa.

Cả ba hệ thống đều đạt cấp độ sản xuất vào tháng 4 năm 2026. Cách phân phối phần thưởng của chúng có sự khác biệt: Bittensor thưởng theo chất lượng so với các validator của subnet; Fetch thưởng theo độ hữu dụng do người dùng chi trả đánh giá; Gonka thưởng cho công việc suy luận có thể xác minh được.

### Phân bổ công trạng theo giá trị Shapley (Shapley-Value Credit Attribution)

Ba tác tử cùng hợp tác thực hiện một tác vụ. Đầu ra đạt điểm chất lượng 0,8. Ai đã đóng góp bao nhiêu phần trong kết quả đó?

Giá trị Shapley: là cơ chế phân bổ công trạng duy nhất thỏa mãn đồng thời bốn tiên đề toán học (tính hiệu quả - efficiency, tính đối xứng - symmetry, tính tuyến tính - linearity, và phần tử rỗng - null player). Đối với tác tử $i$:

```text
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

Trong đó $S_{i\_O}$ là tập hợp các tác tử xuất hiện trước $i$ trong một thứ tự sắp xếp $O$. Trong thực tế: liệt kê toàn bộ các hoán vị, ghi nhận đóng góp biên (marginal contribution) của từng tác tử trong mỗi hoán vị, rồi tính trung bình cộng.

Với $N = 3$ tác tử, có 6 hoán vị. Với $N = 10$, có tới 3,6 triệu hoán vị — do đó trong triển khai thực tế, bạn sẽ lấy mẫu các hoán vị thay vì liệt kê vét cạn.

### Đấu giá giá thứ hai cho bài toán tổng hợp (Second-Price Auction)

Google Research ("Thiết kế cơ chế cho các mô hình ngôn ngữ lớn") đề xuất các phiên đấu giá token giá thứ hai (second-price token auctions) để tổng hợp các kết quả đầu ra của LLM. Thiết lập: $N$ tác tử, mỗi tác tử đề xuất một đoạn văn bản hoàn thành; mỗi tác tử có một giá trị riêng (private value) kỳ vọng khi được lựa chọn. Người tổ chức đấu giá sẽ chọn đề xuất có giá trị cao nhất và trả cho tác tử đó mức giá bằng với **giá trị cao thứ hai**. Dưới điều kiện tổng hợp đơn điệu (giá trị phụ thuộc vào đề xuất nào được chọn, chứ không phụ thuộc vào có bao nhiêu đề xuất được chào thầu), cơ chế này mang tính trung thực (truthful) — các tác tử có chiến lược tối ưu là đấu thầu đúng giá trị thực của mình.

Ý nghĩa đối với hệ thống LLM: Bạn có thể thuê ngoài các tác vụ hoàn thành văn bản cho nhiều tác tử với các mức định giá khác nhau; cơ chế đấu giá sẽ chọn ra tác tử tốt nhất, chi trả công bằng, và các tác tử không có động cơ báo cáo sai lệch.

### Vốn danh tiếng (Reputation Capital)

Điểm danh tiếng gắn với DID được tích lũy từ các đóng góp đã được xác minh thành công. Quy tắc cập nhật đơn giản:

```math
\text{rep}(i, t+1) = \alpha \cdot \text{rep}(i, t) + (1 - \alpha) \cdot \text{contribution\_quality}(i, t)
```

Với hệ số suy giảm $\alpha$ gần bằng 1. Điểm danh tiếng có các đặc điểm:

- Rất rẻ khi đọc để đưa ra quyết định định tuyến ("giao các tác vụ khó cho các tác tử có danh tiếng cao").
- Rất đắt để làm giả (phải tích lũy qua thời gian dài, gắn chặt với định danh DID).
- Có thể bị phạt cắt giảm (slashing): những đóng góp không vượt qua khâu kiểm tra xác minh sẽ bị trừ điểm danh tiếng nặng nề.

### Mô hình LaMAS phi tập trung tại AAMAS 2025

Đề xuất LaMAS (AAMAS 2025) kết hợp ba thành tố: danh tính DID, phân bổ công trạng theo giá trị Shapley, và cơ chế đấu giá tinh gọn. Luận điểm trọng tâm: việc phi tập trung hóa bước phân bổ công trạng giúp hệ thống có thể kiểm toán minh bạch và miễn nhiễm trước các hành vi thao túng điểm tập trung (single-point manipulation).

### Những kịch bản nền kinh tế tác tử sụp đổ

- **Thao túng nguồn cấp dữ liệu giá (Price oracle manipulation):** Nếu hàm đánh giá công trạng có kẽ hở để khai thác, các tác tử chắc chắn sẽ khai thác nó. Mọi cơ chế đều cần bài kiểm thử đối kháng.
- **Tấn công Sybil:** Một thực thể khởi tạo hàng loạt $N$ tác tử giả mạo để thổi phồng mức đóng góp của chính mình. DID làm chậm nhưng không thể ngăn chặn triệt để; chi phí làm giả danh tiếng cao chính là giải pháp giảm thiểu.
- **Chi phí xác minh (Verification cost):** Phân bổ công trạng chỉ công bằng khi bộ xác minh hoạt động chuẩn xác. Nếu xác minh quá rẻ (dùng LLM nhỏ), nó dễ bị qua mặt; nếu quá đắt (dùng hội đồng thẩm định con người), hệ thống sẽ không thể mở rộng quy mô.
- **Rào cản pháp lý (Regulatory overhang):** Các nền kinh tế tác tử giao thoa trực tiếp với các quy định pháp luật tài chính. Tính đến năm 2026, Bittensor, Fetch và Gonka đều hoạt động trong các vùng xám pháp lý tại một số khu vực tài phán.

### Khi nào nền kinh tế tác tử thực sự phát huy giá trị

- **Các mạng lưới mở với nhiều đơn vị vận hành không đồng nhất:** Không có bất kỳ một đội ngũ duy nhất nào kiểm soát toàn bộ các tác tử.
- **Các đầu ra có thể kiểm chứng được (Verifiable outputs):** Nếu không thể kiểm chứng độc lập, phân bổ công trạng chỉ là phỏng đoán vô căn cứ.
- **Quy trình làm việc có tầm chân trời dài:** Các tác vụ chạy một lần (one-shot) không mang lại lợi ích gì từ việc tích lũy danh tiếng.
- **Thanh toán bằng token được pháp luật công nhận** tại khu vực tài phán của bạn.

Trong các hệ thống đóng của doanh nghiệp, các cơ chế kinh tế thường nhường chỗ cho các phương thức phân bổ tài nguyên đơn giản hơn (người quản lý giao việc, chỉ số đánh giá nội bộ). Các tài liệu học thuật về kinh tế tác tử chủ yếu phát huy hiệu quả trong các mạng lưới mở.

````figure
swarm-auction
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa:

- `shapley(value_fn, agents)` — Tính toán chính xác giá trị Shapley bằng phương pháp liệt kê cho số lượng tác tử $N$ nhỏ.
- `second_price_auction(bids)` — Cơ chế đấu giá trung thực; tác tử thắng cuộc chỉ trả mức giá cao thứ hai.
- `Reputation` — Hệ thống danh tiếng gắn với DID tích hợp phân rã mũ và cơ chế phạt (slashing).
- Bản thử nghiệm 1 (Demo 1): Ba tác tử hợp tác, giá trị Shapley phân bổ chính xác công trạng của từng bên.
- Bản thử nghiệm 2 (Demo 2): Năm tác tử đấu thầu giành quyền thực hiện tác vụ; đấu giá giá thứ hai chọn ra người thắng và số tiền chi trả.
- Bản thử nghiệm 3 (Demo 3): 100 vòng giao tác vụ cho các tác tử có mức danh tiếng khác nhau; định tuyến dựa trên trọng số danh tiếng đạt chất lượng vượt trội so với định tuyến ngẫu nhiên.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả kỳ vọng: Các giá trị Shapley cho từng tác tử; kết quả đấu giá thể hiện trạng thái cân bằng đấu thầu trung thực; định tuyến theo trọng số danh tiếng mang lại mức tăng chất lượng 10-20% so với ngẫu nhiên sau giai đoạn khởi động (warmup).

## Ứng dụng thực tế (Use It)

Tài liệu `outputs/skill-economy-designer.md` hướng dẫn thiết kế một nền kinh tế tác tử tối giản: lựa chọn tầng danh tính, cơ chế phân bổ công trạng, cơ chế thanh toán và quy tắc cập nhật danh tiếng.

## Triển khai thực tế (Ship It)

Khi vận hành một nền kinh tế tác tử trong năm 2026:

- **Bắt đầu từ danh tiếng, chưa vội dùng token:** Danh tiếng rất rẻ để triển khai và bản thân nó đã mang lại giá trị to lớn; token mang lại nhiều phức tạp về kinh tế và pháp lý.
- **Kiểm chứng trước khi trao thưởng:** Tuyệt đối không phân bổ phần thưởng nếu thiếu bước xác minh độc lập. Việc dựa vào chất lượng tự báo cáo sẽ dẫn đến các chiêu trò gian lận Sybil.
- **Lấy mẫu Shapley thay vì tính toán vét cạn:** Lấy mẫu 100-1000 lượt hoán vị ngẫu nhiên; việc liệt kê chính xác tuyệt đối không thể mở rộng quy mô.
- **Giới hạn hệ số suy giảm và đặt sàn danh tiếng:** Phân rã quá nhanh sẽ xóa sạch công lao của những tác tử chân chính; phân rã quá chậm lại duy trì danh tiếng lỗi thời cho các tác tử đã sa sút.
- **Kiểm toán cơ chế theo phương pháp đối kháng:** Chạy các kịch bản diễn tập đội đỏ (red-team) trước khi mở mạng lưới ra công chúng. Mọi cơ chế kinh tế đều hàm chứa lý thuyết trò chơi; nhiệm vụ của bạn là tìm ra lỗ hổng trước khi kẻ tấn công phát hiện.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Xác minh rằng tổng các giá trị Shapley bằng đúng tổng giá trị của toàn liên minh (tiên đề tính hiệu quả). Hãy thay đổi hàm giá trị; các phân bổ Shapley có biến đổi theo chiều hướng kỳ vọng không?
2. Hiện thực hóa kỹ thuật *lấy mẫu* Shapley (phương pháp Monte Carlo trên $K$ lượt hoán vị). Số lượng mẫu $K$ tác động như thế nào đến độ chính xác xấp xỉ? So sánh với kết quả tính chính xác khi $N = 4$.
3. Hiện thực hóa bước thành lập liên minh trước khi đấu giá: các tác tử có thể hợp nhất thành nhóm và đấu thầu chung. Những liên minh nào sẽ được hình thành? Kết quả thu được có tối ưu hơn theo chuẩn Pareto (Pareto-better) so với việc đấu thầu đơn lẻ không?
4. Đọc bài viết thiết kế cơ chế của Google Research. Hãy chỉ ra một giả định mà nếu bị vi phạm, tính trung thực (truthfulness) của phiên đấu giá sẽ sụp đổ. Điểm thất bại đó biểu hiện như thế nào trong bối cảnh hệ thống LLM?
5. Đọc bài báo LaMAS phi tập trung tại AAMAS 2025. Hiện thực hóa bước tính Shapley của họ trên 10 tác tử cho một tác vụ giả lập. Việc tính toán chính xác mất bao lâu? Phép lấy mẫu với 100 lần rút tiến gần đến kết quả chính xác ở mức độ nào?

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| DePIN | "Hạ tầng vật lý phi tập trung" | Năng lực tính toán/lưu trữ/băng thông được khuyến khích bằng token. Bittensor, Akash, Render. |
| DID | "Định danh phi tập trung" | Đặc tả chuẩn W3C cho các định danh khả chuyển. Danh tiếng tác tử gắn liền với DID, không phụ thuộc vào một nền tảng. |
| ERC-4337 | "Trừu tượng hóa tài khoản" | Các tài khoản hợp đồng thông minh có thể tài trợ phí gas, cho phép tác tử tự thanh toán. |
| Giá trị Shapley (Shapley value) | "Phân bổ công trạng công bằng" | Phương pháp phân bổ duy nhất thỏa mãn đồng thời tính hiệu quả, đối xứng, tuyến tính và phần tử rỗng. |
| Đấu giá giá thứ hai (Second-price auction) | "Đấu giá Vickrey" | Cơ chế trung thực: người trả giá cao nhất thắng cuộc nhưng chỉ trả mức giá cao thứ hai. Tương thích với phép tổng hợp đơn điệu. |
| Vốn danh tiếng (Reputation capital) | "Điểm chất lượng tích lũy" | Điểm số gắn với DID từ các đóng góp đã xác minh; suy giảm dần theo thời gian. |
| Agentic DAO | "Tác tử và con người cùng quản trị" | Mô hình DAO coi tác tử như những thực thể bỏ phiếu hạng nhất, quyền biểu quyết gắn với điểm danh tiếng. |
| TAO / FET / GPU credits | "Các đơn vị token kinh tế" | Token của Bittensor (TAO), Fetch.ai (FET), và các đồng tiền thanh toán hạ tầng DePIN. |

## Tài liệu tham khảo (Further Reading)

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — Khảo sát năm 2026 về ngăn xếp 5 tầng của nền kinh tế tác tử.
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — Đấu giá token kết hợp tổng hợp đơn điệu.
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Phân bổ công trạng dựa trên giá trị Shapley.
- [Bittensor TAO documentation](https://docs.bittensor.com/) — Cấu trúc subnet và phân phối phần thưởng.
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — Mô hình LLM ASI-1 Mini và hệ sinh thái token FET.
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — Đặc tả nền tảng định danh phi tập trung.
