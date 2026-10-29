# Tối ưu hóa Đàn cho LLM: PSO và ACO (Swarm Optimization for LLMs)

> Tối ưu hóa phỏng sinh học (bio-inspired optimization) đang tạo nên sự trở lại mạnh mẽ trong kỷ nguyên LLM. **LMPSO** (arXiv:2504.09247) sử dụng thuật toán tối ưu hóa bầy đàn (PSO - Particle Swarm Optimization) trong đó vận tốc của mỗi hạt là một câu lệnh gợi ý (prompt) và mô hình ngôn ngữ lớn (LLM - Large Language Model) sinh ra ứng viên tiếp theo; hoạt động rất hiệu quả trên các đầu ra chuỗi có cấu trúc (biểu thức toán học, chương trình mã nguồn). **Model Swarms** (arXiv:2410.11163) xem mỗi chuyên gia LLM như một hạt PSO trên đa tạp trọng số mô hình (model-weight manifold) và ghi nhận **mức tăng trung bình 13,3%** so với 12 mô hình cơ sở trên 9 tập dữ liệu chỉ với 200 mẫu thử. **SwarmPrompt** (ICAART 2025) lai ghép PSO và thuật toán bầy sói xám (Grey Wolf Optimizer) để tối ưu hóa prompt. **AMRO-S** (arXiv:2603.12933) lấy cảm hứng từ thuật toán bầy kiến (ACO - Ant Colony Optimization) với các pheromone chuyên biệt để định tuyến tác tử LLM — mang lại **tốc độ tăng tốc 4,7 lần**, bằng chứng định tuyến có thể diễn giải được (interpretable routing evidence), cùng cơ chế cập nhật bất đồng bộ có cổng kiểm soát chất lượng (quality-gated asynchronous update) giúp tách rời suy luận khỏi quá trình học. Bài học này hiện thực hóa PSO trên không gian tham số prompt và ACO trên bài toán định tuyến tác tử, phân tích lý do tại sao các thuật toán cổ điển này lại phù hợp với kỷ nguyên LLM, và khi nào thì không nên dùng.

**Type:** Learn + Build  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)  
**Time:** ~75 phút  

## Vấn đề (Problem)

Bạn có một prompt đạt điểm số 62% trong bài đánh giá tác vụ (task eval). Bạn muốn cải thiện nó. Cách làm ngây thơ là tinh chỉnh thủ công không dùng đạo hàm (gradient-free manual tweaking), nhưng phương pháp này mở rộng quy mô rất kém. Học tăng cường (reinforcement learning) đòi hỏi tín hiệu phần thưởng (reward signals) và đủ số lượt thử nghiệm (rollouts) để huấn luyện. Lan truyền ngược (backpropagation) qua prompt là điều bất khả thi — bản thân prompt là một chuỗi ký tự rời rạc, không phải là một tham số khả vi (differentiable parameter).

Các giải thuật tối ưu hóa phỏng sinh học cổ điển — PSO cho không gian tìm kiếm liên tục, ACO cho việc chọn lựa đường đi — được thiết kế chính xác cho cơ chế hoạt động này: không cần đạo hàm (gradient-free), dựa trên quần thể (population-based), và chi phí trên mỗi lượt đánh giá rất rẻ. Khi kết hợp chúng với LLM cho bước tìm kiếm không đạo hàm, bạn sẽ nhận được một bộ tối ưu hóa (optimizer) hiệu quả và thực tế đến bất ngờ.

Các mẫu hình tương tự cũng áp dụng cho việc *định tuyến* tác tử (agent routing) trong các hệ đa tác tử (multi-agent systems). Một vệt pheromone kiểu ACO ghi nhận tác tử nào hoạt động tốt nhất trên từng loại tác vụ cụ thể, cho phép bộ định tuyến khai thác vệt dấu vết này, đồng thời làm bay hơi pheromone theo thời gian để các tuyến đường mới có thể được tái khám phá.

## Khái niệm (Concept)

### Nhắc lại về PSO (Kennedy & Eberhart 1995)

Tối ưu hóa bầy đàn (PSO - Particle Swarm Optimization): một quần thể gồm các hạt di chuyển trong một không gian tìm kiếm liên tục. Mỗi hạt có vị trí $x_i$ và vận tốc $v_i$. Tại mỗi bước lặp (iteration):

```text
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

Trong đó:
- `p_best` là vị trí tốt nhất của bản thân hạt đó từng đạt được (personal best).
- `g_best` là vị trí tốt nhất của toàn bộ bầy đàn (global best).
- `w, c1, c2` lần lượt là trọng số quán tính (inertia weight), trọng số nhận thức cá nhân (cognitive weight), và trọng số xã hội (social weight).
- `r1, r2` là các hệ số ngẫu nhiên phân bố đều trong khoảng $[0, 1]$.

### PSO trên đầu ra LLM — LMPSO

Nghiên cứu arXiv:2504.09247 điều chỉnh PSO dành riêng cho các đầu ra có cấu trúc do LLM sinh ra (biểu thức toán học, chương trình mã nguồn). Mỗi hạt là một ứng viên đầu ra. Vận tốc ở đây là một *prompt* mô tả cách sửa đổi đầu ra hiện tại theo hướng tiến về phía cá nhân tốt nhất (`p_best`) hoặc toàn cục tốt nhất (`g_best`). LLM sẽ tạo ra đầu ra mới dựa trên prompt vận tốc này. Yếu tố "quán tính" của vận tốc chính là một prompt dạng "hãy thực hiện những thay đổi gia tăng nhỏ".

Phương pháp này hoạt động hiệu quả khi:
- Đầu ra có cấu trúc rõ ràng (có thể phân tích cú pháp (parseable), có thể đánh giá (evaluable)).
- Độ thích nghi (fitness) được tính tự động (chạy bộ kiểm thử, tính toán giá trị số học).
- Quy mô quần thể nhỏ (~10-30 hạt) để tổng số lượt gọi LLM nằm trong tầm kiểm soát.

Phương pháp này không phù hợp khi việc tính fitness cần con người đánh giá — chi phí trên mỗi vòng lặp khi đó sẽ trở nên quá đắt đỏ.

### Bầy mô hình (Model Swarms)

Nghiên cứu arXiv:2410.11163 đưa PSO vượt ra khỏi tầng đầu ra và tiến thẳng vào tầng *mô hình* (model layer). Mỗi "hạt" là một mô hình LLM chuyên gia (thể hiện qua các tham số). Cả bầy sẽ dịch chuyển các tham số về phía trạng thái tập thể tốt nhất thông qua một bản cập nhật không cần đạo hàm. Kết quả báo cáo: đạt mức tăng trung bình 13,3% so với 12 mô hình cơ sở trên 9 tập dữ liệu, chỉ với 200 mẫu dữ liệu cho mỗi bước lặp.

Điểm mấu chốt là các mô hình LLM chuyên gia vốn dĩ đã nằm gần nhau trên một đa tạp tham số dùng chung (trọng số adapter, độ lệch delta của LoRA). Chạy thuật toán PSO trên không gian con số chiều thấp này vừa rẻ vừa mang lại hiệu quả cao.

### Nhắc lại về ACO (Dorigo 1992)

Thuật toán tối ưu hóa bầy kiến (ACO - Ant Colony Optimization): các cá thể kiến di chuyển trên một đồ thị; mỗi cạnh/đường đi đều có một vệt pheromone. Xác suất di chuyển của kiến được tính dựa trên độ đậm đặc của pheromone. Những con kiến hoàn thành nhiệm vụ sẽ để lại lượng pheromone tỷ lệ thuận với chất lượng của lời giải. Pheromone sẽ bay hơi (suy giảm) dần theo thời gian.

### AMRO-S — ACO dành cho định tuyến tác tử

Nghiên cứu arXiv:2603.12933 ứng dụng ACO vào bài toán định tuyến đa tác tử. Mỗi loại tác vụ (task-type) là một "điểm đến"; mỗi tác tử là một tuyến đường khả dĩ. Pheromone củng cố các tuyến đường mang lại đầu ra chất lượng tốt. Những đóng góp then chốt bao gồm:

- **Bằng chứng định tuyến có thể diễn giải được (Interpretable routing evidence):** Nồng độ pheromone là một tín hiệu trực quan mà con người có thể đọc và hiểu được.
- **Cập nhật bất đồng bộ có cổng kiểm soát chất lượng (Quality-gated asynchronous update):** Pheromone chỉ được cập nhật sau khi vượt qua các bước kiểm tra chất lượng, giúp tách biệt hoàn toàn giữa suy luận và quá trình học.
- **Tăng tốc 4,7 lần** trên bộ đo chuẩn định tuyến đa tác tử.

Cổng kiểm soát chất lượng (quality gate) đóng vai trò sống còn: nếu thiếu nó, các tác tử phản hồi nhanh nhưng sai lệch sẽ tích lũy pheromone giả tạo, khiến toàn bộ hệ thống bị khóa chặt vào các tuyến đường kém chất lượng.

### Khi nào nên sử dụng PSO / ACO cho LLM

**Nên dùng PSO khi:**
- Không gian tìm kiếm là liên tục hoặc ánh xạ được sang các tham số liên tục (nhúng prompt (prompt embeddings), trọng số LoRA, các siêu tham số sinh văn bản dạng số).
- Việc đánh giá độ thích nghi (fitness) diễn ra tự động và tốn ít chi phí.
- Quần thể có thể duy trì ở quy mô nhỏ (10-30 hạt).

**Nên dùng ACO khi:**
- Bạn gặp bài toán định tuyến hoặc lựa chọn đường đi giữa các tác tử.
- Các quyết định mang tính củng cố tích lũy theo thời gian (các loại tác vụ tương tự liên tục quay trở lại).
- Bạn cần các bằng chứng minh bạch, dễ giải thích cho từng quyết định định tuyến.

**Không nên dùng cả hai khi:**
- Đánh giá fitness đòi hỏi con người tham gia (chi phí quá cao cho mỗi bước lặp).
- Không gian tìm kiếm là rời rạc và tổ hợp theo cách mà PSO không thể bao quát (hãy chuyển sang dùng giải thuật di truyền (genetic algorithms)).
- Các quyết định thời gian thực đòi hỏi độ trễ cực kỳ khắt khe (PSO/ACO hội tụ tương đối chậm so với các luật khám phá (heuristics) xử lý một lượt).

### Tại sao tối ưu hóa phỏng sinh học vẫn chiếm ưu thế

Các phương pháp dựa trên đạo hàm đòi hỏi tín hiệu khả vi. Đầu ra của LLM và các quyết định định tuyến không thể lấy đạo hàm một cách trực tiếp. Các phương pháp giả đạo hàm (pseudo-gradient) như bộ định tuyến học tăng cường hay bộ tinh chỉnh prompt kiểu DPO có hoạt động, nhưng đòi hỏi quá trình huấn luyện vô cùng tốn kém.

Trong khi đó, PSO và ACO chỉ cần một hàm *đánh giá* (evaluator function). Miễn là bạn có thể chấm điểm một ứng viên đầu ra hoặc một quyết định định tuyến, bạn hoàn toàn có thể tối ưu hóa trên không gian đó. Điều này giúp hạ thấp đáng kể rào cản ứng dụng vào thực tế.

### Các giới hạn thực tế

- **Ngân sách quần thể (Population budget):** $N$ hạt $\times$ $T$ vòng lặp $\times$ chi phí mỗi lần đánh giá. Với chi phí đánh giá LLM khoảng ~0,02 USD / lượt gọi, một bầy PSO gồm 20 hạt chạy 50 vòng lặp sẽ tốn ~20 USD. Cần lập kế hoạch ngân sách hợp lý.
- **Khám phá so với Khai thác (Exploration vs Exploitation):** Tốc độ bay hơi của pheromone và trọng số quán tính trong PSO luôn là sự đánh đổi; bay hơi quá nhanh khiến bầy đàn quên mất các giải pháp tốt; bay hơi quá chậm lại khiến hệ thống mắc kẹt trong các điểm cực tiểu cục bộ (local optima) ban đầu.
- **Trôi dạt thảm họa (Catastrophic drift):** Cả hai thuật toán đều có nguy cơ hội tụ rồi phân kỳ nếu bề mặt thích nghi (fitness landscape) dịch chuyển (do phân phối dữ liệu mới). Cần liên tục giám sát tính ổn định của giá trị fitness tốt nhất.

````figure
swarm-stigmergy
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa:

- `LMPSO` — Thuật toán PSO trên không gian tham số prompt dạng số (nhiệt độ `temperature`, trọng số `top_k`). Bước "sinh văn bản của LLM" cho từng hạt được mô phỏng bằng một hàm fitness theo kịch bản. Chạy thuật toán trong 30 vòng lặp và hiển thị sự hội tụ của `g_best`.
- `AMRO_S` — Định tuyến lấy cảm hứng từ ACO. Hệ thống gồm 3 tác tử, 4 loại tác vụ, một ma trận pheromone và 100 tác vụ được định tuyến. Chương trình sẽ in ra phân phối (loại tác vụ $\rightarrow$ lựa chọn tác tử) theo thời gian để minh họa sự hình thành của vệt pheromone.
- So sánh đối chiếu: Định tuyến ngẫu nhiên (random routing) so với định tuyến ACO trên cùng một luồng tác vụ. Đo đạc chất lượng và độ trễ.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả kỳ vọng:
- LMPSO: Điểm fitness của `g_best` cải thiện từ mức ngẫu nhiên đến gần tối ưu sau 30 vòng lặp.
- AMRO-S: Bảng pheromone ổn định dần trên đúng tác tử tối ưu cho từng loại tác vụ; định tuyến ACO vượt trội hơn định tuyến ngẫu nhiên khoảng 30-40% về mặt chất lượng, đồng thời giảm độ trễ (nhờ ít phải thử lại).

## Ứng dụng thực tế (Use It)

Tài liệu `outputs/skill-swarm-optimizer.md` hỗ trợ bạn lựa chọn giữa PSO, ACO, thuật toán di truyền và các bộ tối ưu dựa trên đạo hàm cho các bài toán tối ưu hóa tác tử và LLM.

## Triển khai thực tế (Ship It)

- **Bắt đầu từ quy mô nhỏ:** 10-20 hạt, 20-50 vòng lặp. Chỉ mở rộng quy mô nếu đường cong hội tụ thể hiện mức tăng trưởng rõ ràng.
- **Ghi nhật ký (log) pheromone hoặc `g_best` ở mỗi bước lặp:** Gỡ lỗi các bộ tối ưu hóa đàn mà thiếu nhật ký vết là một cực hình.
- **Kiểm soát cập nhật bằng cổng chất lượng:** Đặc biệt quan trọng với định tuyến ACO: các tác tử phản hồi nhanh nhưng sai tuyệt đối không được phép tích lũy pheromone.
- **Đặt lại tốc độ bay hơi khi có dịch chuyển phân phối dữ liệu:** Khi phân phối đánh giá thay đổi, pheromone cũ sẽ bị lỗi thời; hãy đặt lại hoặc tạm thời nhân đôi tốc độ bay hơi pheromone.
- **Đặt trần chi phí cho mỗi vòng lặp:** Luôn đo đạc chỉ số chi phí trên từng vòng lặp. Một giải thuật PSO tiêu tốn 500 USD / vòng lặp mà chỉ cải thiện 0,5% hiệu năng thì không thể đưa vào môi trường sản xuất.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Quan sát sự hội tụ của LMPSO. Thử nghiệm thay đổi kích thước quần thể: 5, 10, 20, 50. Tại kích thước nào thì thời gian hội tụ bắt đầu bão hòa?
2. Thiết kế một thí nghiệm "trôi dạt thảm họa": sau vòng lặp thứ 30, hãy thay đổi hàm fitness. PSO thích ứng nhanh đến mức nào? Việc đặt lại `p_best` có giúp ích không?
3. Bổ sung cổng chất lượng cho AMRO-S: chỉ gửi thêm pheromone cho các lần chạy đạt điểm đánh giá $> 0,7$. Sự hội tụ thay đổi như thế nào so với phiên bản không có cổng kiểm soát?
4. Đọc bài báo LMPSO (arXiv:2504.09247). Hãy ánh xạ khái niệm "vận tốc dưới dạng prompt" trong bài báo ngược về vận tốc số học của bạn. Điều gì bị mất đi trong mô phỏng và điều gì được bảo toàn?
5. Đọc bài báo AMRO-S (arXiv:2603.12933). Hiện thực hóa "đường dẫn suy luận nhanh" (inference fast-path) được tách rời với quá trình cập nhật pheromone bất đồng bộ. Điều này tác động thế nào đến độ trễ hệ thống dưới tải liên tục?

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| PSO | "Tối ưu hóa bầy đàn" | Kennedy-Eberhart 1995. Thuật toán tối ưu hóa không dùng đạo hàm dựa trên quần thể. |
| ACO | "Tối ưu hóa bầy kiến" | Dorigo 1992. Tối ưu hóa đường đi/tuyến đường dựa trên vết pheromone. |
| LMPSO | "PSO kết hợp sinh văn bản bằng LLM" | arXiv:2504.09247. Vận tốc là một prompt; LLM tạo ra các ứng viên tiếp theo. |
| Model Swarms | "PSO trên trọng số chuyên gia" | arXiv:2410.11163. Cập nhật không dùng đạo hàm trên không gian con tham số mô hình. |
| AMRO-S | "ACO cho định tuyến tác tử" | arXiv:2603.12933. Ma trận pheromone trên cặp (loại tác vụ $\times$ tác tử). |
| `p_best` / `g_best` | "Kỷ lục cá nhân / Kỷ lục toàn bầy" | Lời giải tốt nhất từng đạt được của một hạt và của toàn bộ bầy đàn tính đến thời điểm hiện tại. |
| Pheromone | "Bộ nhớ định tuyến" | Cường độ lưu vết trên một cạnh; bay hơi theo thời gian; được bồi tụ khi đạt chất lượng tốt. |
| Cập nhật có cổng chất lượng (Quality-gated update) | "Chỉ học từ các lần chạy tốt" | Việc tích lũy pheromone được điều kiện hóa dựa trên kết quả kiểm tra chất lượng. |
| Trôi dạt thảm họa (Catastrophic drift) | "Dịch chuyển phân phối dữ liệu" | Bề mặt fitness thay đổi; các giá trị `p_best` và pheromone cũ trở nên lỗi thời. |

## Tài liệu tham khảo (Further Reading)

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — Bài báo đặt nền móng về PSO năm 1995.
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — Các nền tảng lý thuyết ACO năm 1992.
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — Ứng dụng PSO cho các đầu ra LLM có cấu trúc.
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO trên không gian con trọng số mô hình.
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — Định tuyến đa tác tử dựa trên pheromone có cổng chất lượng.
