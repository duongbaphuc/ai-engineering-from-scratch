# Lý thuyết Tâm trí và Sự Phối hợp Tự phát sinh (Theory of Mind and Emergent Coordination)

> Nghiên cứu của Li et al. (arXiv:2310.10701) chỉ ra rằng các tác tử LLM trong trò chơi văn bản hợp tác có thể bộc lộ **Lý thuyết Tâm trí (Theory of Mind - ToM) bậc cao tự phát sinh** — tức khả năng suy luận về việc một tác tử khác đang tin vào điều gì về nhận thức của một tác tử thứ ba — nhưng lại dễ thất bại trong việc lập kế hoạch dài hạn do giới hạn quản lý ngữ cảnh và ảo giác. Nghiên cứu của Riedl (arXiv:2510.05174) khi đo lường mức độ hiệp đồng bậc cao trên một quần thể tác tử đã phát hiện ra rằng: **chỉ khi** có điều kiện prompt định hướng ToM, hệ thống mới thực sự sản sinh ra sự phân hóa gắn liền với danh tính (identity-linked differentiation) và tính bổ trợ hướng mục tiêu (goal-directed complementarity); các LLM năng lực thấp hơn chỉ bộc lộ sự tự phát sinh giả tạo. Nói cách khác, sự phối hợp tự phát sinh là một hiện tượng phụ thuộc chặt chẽ vào prompt và mô hình nền tảng, chứ không phải một món quà miễn phí. Bài học này sẽ tự tay xây dựng một tác tử có nhận thức ToM tối giản, vận hành một tác vụ hợp tác có và không có prompt ToM, đồng thời đo lường mức độ chênh lệch phối hợp dựa trên quy thức kiểm định của Riedl (2025).

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 phút

## Vấn đề

Sự phối hợp trong các hệ đa tác tử thoạt nhìn thường mang vẻ kỳ diệu: các tác tử tự phân chia công việc, dự đoán trước hành vi của nhau, và tránh né sự trùng lặp. Tuy nhiên, phần lớn hiện tượng "tự phát sinh" này thực chất chỉ là một sản phẩm nhân tạo của kỹ thuật tạo prompt — một kỹ sư đã ngầm dặn dò các tác tử phải "hãy phối hợp cùng nhau". Bỏ prompt đó đi, sự phối hợp lập tức biến mất.

Phát hiện của Riedl vào năm 2025 mang tính nghiêm ngặt hơn: dưới các điều kiện kiểm soát chặt chẽ, sự phối hợp chỉ thực sự tự phát sinh khi các tác tử được hướng dẫn suy luận về **tâm trí của các tác tử khác** (Lý thuyết Tâm trí - ToM). Nếu thiếu prompt ToM, ngay cả các mô hình mạnh mẽ nhất cũng chỉ tạo ra các mẫu hình phối hợp không vượt qua được các phép kiểm định thống kê. Điều này tác động trực tiếp đến môi trường sản xuất: nhiều đội ngũ đang đưa vào vận hành những tính năng "hợp tác đa tác tử" thực chất chỉ phụ thuộc mong manh vào prompt và rất dễ đổ vỡ.

Bài học này tiếp cận ToM như một năng lực kỹ thuật cụ thể (khả năng suy luận về các niềm tin đối với niềm tin), tự tay xây dựng một tác tử có nhận thức ToM tối giản, và đo lường sự khác biệt rành mạch giữa phối hợp thực chất và việc ngụy trang bằng câu từ trong prompt.

## Khái niệm

### Lý thuyết Tâm trí (ToM) là gì?

Trong tâm lý học phát triển: một đứa trẻ 3 tuổi nghĩ rằng thế giới nội tâm của mọi người xung quanh hoàn toàn trùng khớp với mình. Lên 5 tuổi, đứa trẻ bắt đầu hiểu rằng người khác có những niềm tin và góc nhìn khác biệt. Đến 7 tuổi, trẻ có thể suy luận về niềm tin đối với niềm tin ("cô ấy nghĩ rằng tôi nghĩ quả bóng đang nằm dưới chiếc cốc"). Đó lần lượt là ToM bậc không, bậc một và bậc hai.

Đối với các tác tử LLM, các cấp độ ToM được ánh xạ tương ứng:

- **Bậc không (Zeroth-order):** Không xây dựng mô hình về người khác. Tác tử chỉ hành động dựa trên những quan sát trực tiếp của bản thân.
- **Bậc một (First-order):** Tác tử xây dựng mô hình về niềm tin của từng tác tử khác. "Alice tin rằng X là đúng."
- **Bậc hai (Second-order):** Tác tử mô hình hóa các niềm tin đệ quy. "Alice tin rằng Bob tin rằng X là đúng."

Li et al. (2023) phát hiện ra rằng ToM bậc một và bậc hai có thể tự phát sinh trong các tác tử LLM khi tham gia trò chơi hợp tác, nhưng sẽ suy giảm nhanh chóng khi phiên tương tác kéo dài và các kênh giao tiếp không ổn định.

### Bài kiểm tra Sally-Anne tóm lược

Bài kiểm tra niềm tin sai lệch kinh điển năm 1985: Sally đặt một viên bi vào giỏ A, rồi rời phòng. Anne chuyển viên bi sang giỏ B. Hỏi khi quay trở lại phòng, Sally sẽ tìm viên bi ở đâu? Một đứa trẻ đã phát triển ToM bậc một sẽ trả lời là giỏ A (bởi niềm tin của Sally khác biệt với thực tế khách quan). Một đứa trẻ chưa có ToM sẽ trả lời là giỏ B.

Các LLM thuộc kỷ nguyên GPT-4 vượt qua các bài kiểm tra kiểu Sally-Anne một cách dễ dàng khi câu hỏi được đặt ra trực diện, ngắn gọn. Nhưng chúng sẽ thất bại khi câu chuyện trở nên dài dòng, bối cảnh thay đổi nhiều lần, hoặc câu hỏi được diễn đạt gián tiếp. Đó chính là bức tranh thực tế của ToM trong các LLM hiện nay.

### Khung đo lường mức độ phối hợp của Riedl

Riedl (arXiv:2510.05174) đã thiết lập một bài kiểm tra quy mô quần thể: $N$ tác tử, một mục tiêu hợp tác chung, với các điều kiện prompt khác nhau. Ba chỉ số được đo lường:

1. **Sự phân hóa gắn liền với danh tính (Identity-linked differentiation).** Các tác tử có phát triển được sự phân định vai trò ổn định và nhất quán theo thời gian hay không?
2. **Tính bổ trợ hướng mục tiêu (Goal-directed complementarity).** Hành động của các tác tử có mang tính bổ trợ cho nhau (đảm nhận các nhiệm vụ con khác nhau) thay vì dẫm chân lên nhau không?
3. **Hiệp đồng tương hỗ bậc cao (Higher-order synergy).** Một thước đo thống kê chứng minh tập thể đạt được kết quả vượt trội hơn bất kỳ tập hợp con nào cộng lại.

Kết quả thực nghiệm: chỉ dưới điều kiện prompt định hướng ToM, cả ba chỉ số trên mới tạo ra tín hiệu vượt trội rõ ràng so với mốc cơ sở. Nếu thiếu prompt ToM, các chỉ số chỉ dao động quanh mức ngẫu nhiên đối với các mô hình có năng lực trung bình. Các mô hình cỡ lớn bộc lộ một phần khả năng phối hợp dù không có prompt ToM tường minh, nhưng hiệu ứng thu được yếu hơn đáng kể so với khi được hướng dẫn rõ ràng.

### Ảo giác phối hợp (The Coordination Illusion)

Nếu thiếu vắng các phép kiểm định thống kê nghiêm ngặt, hiện tượng "phối hợp tự phát sinh" trong các bản demo thường chỉ là kết quả của:

- Kỹ thuật prompt cài cắm sẵn hành vi hợp tác (system prompt yêu cầu "hãy làm việc cùng nhau").
- Thiên kiến xác nhận của người quan sát (chúng ta chỉ nhìn thấy những mẫu hình mà chúng ta kỳ vọng).
- Việc chủ động chọn lọc những lượt chạy thành công sau thực nghiệm.

Bất kỳ hệ thống sản xuất nào quảng bá tính năng "phối hợp tự phát sinh" mà không đưa ra được các tín hiệu đo lường định lượng đều nên được xem là chiêu bài tiếp thị. Hãy đo lường trước khi đưa ra kết luận.

### Cấu trúc tác tử nhận thức ToM tối giản

Quy trình cấu trúc:

```text
Trạng thái tác tử:
  own_beliefs:    {các dữ kiện mà tác tử tin là đúng}
  other_models:   {other_agent_id -> {các niềm tin mà tác tử gán cho họ}}
  actions_last_N: [lịch sử hành động của các tác tử khác]

Cập nhật từ quan sát:
  - cập nhật own_beliefs từ quan sát trực tiếp của bản thân
  - cập nhật other_models[agent_id] dựa trên hành động của họ + niềm tin trước đó

Lựa chọn hành động:
  - liệt kê các hành động ứng viên
  - với mỗi hành động, dự đoán xem từng tác tử khác sẽ làm gì tiếp theo dựa trên mô hình niềm tin của họ
  - chọn hành động tối đa hóa kết quả chung dưới các dự đoán đó
```

Thuộc tính `other_models` chính là trạng thái ToM nội tại. ToM bậc một chỉ duy trì một cấp độ duy nhất. ToM bậc hai mở rộng thêm `other_models[i][other_models_of_j]` — tức những gì tôi nghĩ rằng tác tử $i$ đang nghĩ về niềm tin của tác tử $j$.

### Vì sao phiên tương tác dài hạn làm suy thoái ToM

Nghiên cứu của Li et al. chỉ ra: giới hạn của cửa sổ ngữ cảnh khiến các tác tử quên mất niềm tin nào thuộc về ai. Hiện tượng ảo giác tiêm các niềm tin sai lệch vào mô hình nhận thức về người khác. Cả hai yếu tố này sinh ra các lỗi dạng "tôi tưởng anh ấy nghĩ rằng X", và các sai lệch này sẽ tích lũy theo cấp số nhân qua thời gian.

Các biện pháp giảm thiểu được kiểm chứng thực tế:

- **Biểu diễn tường minh trạng thái ToM trong prompt.** Định dạng có cấu trúc: `{agent_id: belief_list}`. Ép buộc khâu truy xuất phải bảo toàn tính ràng buộc giữa danh tính và niềm tin.
- **Rút ngắn chuỗi lập luận.** Giảm bớt số lượt cập nhật ToM trong mỗi lượt trao đổi để hạn chế sự tích tụ của ảo giác.
- **Kho lưu trữ ToM ngoại vi.** Duy trì mô hình tâm trí bên ngoài ngữ cảnh LLM; chỉ tiêm các phần thực sự liên quan vào prompt của từng lượt.

### Những trường hợp ToM thất bại trong môi trường sản xuất

- **Bối cảnh đối kháng (Adversarial settings).** Những tác tử có ToM quá rõ ràng lại rất dễ bị thao túng (đối phương có thể mô phỏng lại cách tác tử nghĩ về họ để giăng bẫy khai thác).
- **Đội ngũ không đồng nhất (Heterogeneous teams).** Khi các mô hình nền tảng quá khác biệt, mô hình ToM được xây dựng cho đối thủ này không thể khái quát hóa cho đối thủ khác.
- **Các tác vụ phụ thuộc vào chân lý thực tế.** ToM tập trung vào việc mô hình hóa niềm tin chủ quan; nếu tính đúng đắn của bài toán chỉ phụ thuộc vào các dữ kiện khách quan, ToM có thể trở thành một sự xao nhãng gây lãng phí tài nguyên.

### Các tín hiệu phối hợp thực chất có thể đo lường

Ba tín hiệu thực tế chứng minh sự phối hợp của một đội ngũ là có thật chứ không phải do câu từ trong prompt ngụy tạo:

1. **Tính bổ trợ qua thời gian.** Xuyên suốt một nhiệm vụ nhiều lượt, hành động của các tác tử có bao phủ các tập nhiệm vụ con không giao nhau hay không?
2. **Khả năng dự đoán đón đầu.** Hành động của tác tử A ở lượt $T+1$ có dựa trên một dự đoán chính xác về hành động của tác tử B ở lượt $T+2$ hay không?
3. **Khả năng tự hiệu chỉnh.** Khi tác tử A hiểu sai niềm tin của tác tử B ở lượt $T$, tác tử A có tự điều chỉnh lại nhận thức trước lượt $T+2$ hay không?

Những chỉ số này hoàn toàn có thể đo lường định lượng dựa trên nhật ký hệ thống đa tác tử. Chúng chính là minh chứng cụ thể cho năng lực phối hợp thực chất.

````figure
sw-theory-of-mind
````

## Xây dựng thực tế

Tệp `code/main.py` triển khai:

- `ToMAgent` — theo dõi niềm tin của bản thân và các mô hình niềm tin của từng tác tử khác.
- Một tác vụ hợp tác: ba tác tử phải thu thập ba đồng xu từ ba chiếc hộp; mỗi hộp chỉ chứa được đúng một đồng xu. Các tác tử không được phép trò chuyện trực tiếp; chúng buộc phải suy đoán ý định thông qua hành động của nhau.
- Hai cấu hình thử nghiệm: `zeroth_order` (không có ToM) và `first_order` (ToM với mô hình niềm tin một cấp độ).
- Đo lường qua 200 lượt thử nghiệm ngẫu nhiên: tỷ lệ hoàn thành, tỷ lệ trùng lặp nỗ lực (hai tác tử cùng nhắm vào một hộp), và số lượt trung bình để hoàn thành.

Chạy chương trình:

```bash
python3 code/main.py
```

Đầu ra kỳ vọng: các tác tử bậc không dẫm chân lên nhau với tỷ lệ ~35% và chỉ hoàn thành ~60% các lượt thử nghiệm trong 10 lượt. Ngược lại, các tác tử ToM bậc một chỉ trùng lặp ở mức ~5% và đạt tỷ lệ hoàn thành ~95%. Mức chênh lệch rõ rệt này chính là hiệu ứng phối hợp có thể đo lường định lượng được.

## Sử dụng trong thực tế

Tệp `outputs/skill-tom-auditor.md` là kỹ năng kiểm toán tuyên bố "phối hợp tự phát sinh" của một hệ đa tác tử: kiểm tra xem có hiện tượng ngụy tạo bằng prompt hay không, kiểm tra ý nghĩa thống kê so với nhóm đối chứng, và đo lường độ bổ trợ thực tế của các hành động.

## Đưa vào sản xuất

Danh mục kiểm tra trước khi công bố năng lực phối hợp:

- **Thiết lập nhóm đối chứng (Control condition).** Xây dựng một phiên bản hệ thống loại bỏ hoàn toàn prompt yêu cầu phối hợp. Tiến hành đo lường cả hai phiên bản.
- **Kiểm định ý nghĩa thống kê.** Mức chênh lệch giữa hệ thống thực tế và nhóm đối chứng có đạt ngưỡng ý nghĩa $p < 0.05$ trên chỉ số đo lường hay không?
- **Đo lường tính bổ trợ.** Đánh giá mức độ tách biệt của các hành động theo thời gian, không chỉ dựa vào tỷ lệ thành công cuối cùng.
- **Nhật ký các ca thất bại.** Khi các tác tử phối hợp bất thành, trạng thái ToM nội tại của chúng đang hiển thị điều gì?
- **Công bố giới hạn năng lực mô hình.** Nếu hiệu ứng phối hợp biến mất trên các mô hình ngôn ngữ nhỏ hơn, hãy công bố rõ ràng giới hạn đó.

## Bài tập thực hành

1. Chạy `code/main.py`. Xác nhận ToM bậc một giúp giảm tỷ lệ trùng lặp khoảng ~7 lần. Khoảng cách chênh lệch này có được duy trì khi bạn mở rộng lên 5 tác tử và 5 chiếc hộp không?
2. Triển khai ToM bậc hai (tác tử A mô hình hóa những gì tác tử B nghĩ về tác tử C). Nó có mang lại bước nhảy vọt so với bậc một không? Trên những lớp tác vụ nào?
3. Tiêm **ảo giác** vào trạng thái ToM: lật ngược ngẫu nhiên một niềm tin sau mỗi lượt. Sự suy giảm hiệu năng của ToM bậc một diễn ra ở mức độ nào?
4. Đọc tài liệu của Li et al. (arXiv:2310.10701). Tái lập hiện tượng "suy thoái trong phiên tương tác dài hạn": khi số lượt tăng từ 10 lên 30, hiệu năng của ToM bậc một biến đổi ra sao?
5. Đọc nghiên cứu của Riedl 2025 (arXiv:2510.05174). Triển khai chỉ số thống kê hiệp đồng bậc cao trên nhật ký mô phỏng của bạn. Hiệu ứng này có còn tồn tại nếu loại bỏ điều kiện prompt ToM không?

## Thuật ngữ then chốt

| Thuật ngữ | Cách hiểu thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Lý thuyết Tâm trí (Theory of Mind - ToM) | "Hiểu tâm trí người khác" | Năng lực mô hình hóa các niềm tin của tác tử khác. Phân cấp theo bậc (0, 1, 2+). |
| Bài kiểm tra Sally-Anne | "Thử nghiệm niềm tin sai lệch" | Phương pháp tâm lý học phát triển 1985; LLM vượt qua các bản đơn giản, thất bại ở bản phức tạp. |
| ToM bậc một (First-order ToM) | "A tin rằng X" | Khả năng mô hình hóa niềm tin của một tác tử khác đối với các sự kiện thực tế. |
| ToM bậc hai (Second-order ToM) | "A tin rằng B tin rằng X" | Khả năng mô hình hóa niềm tin đệ quy sâu thêm một cấp độ. |
| Phân hóa gắn liền danh tính | "Vai trò ổn định qua thời gian" | Chỉ số của Riedl: các vai trò được duy trì bền vững, không phải phân bổ ngẫu nhiên. |
| Bổ trợ hướng mục tiêu | "Các hành động không trùng lặp" | Các tác tử chủ động nhắm tới các nhiệm vụ con khác nhau để tối ưu nguồn lực. |
| Hiệp đồng bậc cao | "Tập thể vượt trội từng phần" | Thước đo thống kê của Riedl chứng minh năng lực phối hợp thực chất của nhóm. |
| Ảo giác phối hợp | "Trông có vẻ ăn ý" | Hiện tượng hệ thống chỉ có vẻ ngoài phối hợp nhờ prompt dặn dò chứ không có tín hiệu đo lường thực chất. |

## Tài liệu tham khảo

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — Sự tự phát sinh của ToM trong trò chơi hợp tác; các dạng lỗi trong phiên tương tác dài hạn.
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — Đo lường trên quy mô quần thể; chứng minh prompt ToM là điều kiện sống còn.
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://doi.org/10.1017/S0140525X00076512) — Nguồn gốc ra đời của khái niệm ToM năm 1978.
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://doi.org/10.1016/0010-0277(85)90022-8) — Bài báo kinh điển về thử nghiệm Sally-Anne (1985).
