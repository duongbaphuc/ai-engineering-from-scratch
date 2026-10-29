# MARL — MADDPG, QMIX, MAPPO

> Di sản học tăng cường (reinforcement learning) trong bài toán phối hợp đa tác tử vẫn tiếp tục định hình các hệ thống tác tử LLM trong năm 2026. **MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275) đã giới thiệu mô hình Huấn luyện Tập trung, Thực thi Phi tập trung (CTDE - Centralized Training, Decentralized Execution): mỗi mạng đánh giá (critic) quan sát toàn bộ trạng thái và hành động của tất cả tác tử trong quá trình huấn luyện; nhưng khi triển khai chạy thử nghiệm, chỉ có các mạng hành động cục bộ (local actors) được vận hành. Phương pháp này hoạt động hiệu quả cho cả ba môi trường: hợp tác, cạnh tranh và hỗn hợp. **QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485) là phương pháp phân rã hàm giá trị (value-decomposition) với một mạng trộn đơn điệu (monotonic mixing network); giá trị $Q$ của từng tác tử được kết hợp thành giá trị $Q$ chung toàn cục sao cho phép toán `argmax` phân phối một cách hoàn hảo — thống trị bảng xếp hạng StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955) là thuật toán PPO kết hợp với hàm giá trị tập trung; "hiệu quả đến kinh ngạc" trên particle-world, SMAC, Google Research Football, Hanabi chỉ với rất ít tinh chỉnh siêu tham số. Những nền tảng này đóng vai trò then chốt trong việc huấn luyện chính sách cho các nhóm tác tử cần hành động độc lập. MAPPO chính là **mô hình cơ sở mặc định cho MARL hợp tác năm 2026**. Bài học này xây dựng từng thuật toán từ một môi trường lưới đồ chơi (grid-world toy) đơn giản và khắc sâu ba tư tưởng cốt lõi này vào phản xạ tư duy trước khi bạn bắt tay vào huấn luyện hệ tác tử LLM.

**Type:** Learn  
**Languages:** Python (stdlib, cài đặt tinh gọn không dùng NumPy)  
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)  
**Time:** ~90 phút  

## Vấn đề (Problem)

Các hệ thống tác tử LLM ngày càng hướng tới việc huấn luyện các chính sách (policies) phục vụ sự phối hợp giữa các tác tử: khi nào nên trì hoãn, khi nào nên hành động, và nên kích hoạt tác tử đồng cấp nào. Kho tàng tài liệu hướng dẫn cách huấn luyện các chính sách như vậy chính là Học tăng cường đa tác tử (MARL - Multi-Agent Reinforcement Learning), một lĩnh vực xuất hiện trước làn sóng LLM và sở hữu một tập hợp các thuật toán thống trị kinh điển.

Việc đọc các bài báo nghiên cứu MARL mà thiếu đi vốn từ vựng mẫu hình (pattern vocabulary) sẽ rất chật vật. Các khái niệm như huấn luyện tập trung với thực thi phi tập trung (CTDE), phân rã giá trị (value decomposition), và mạng đánh giá tập trung (centralized critics) không phải là những thuật ngữ thời thượng sáo rỗng — chúng là những câu trả lời cụ thể cho các vấn đề kỹ thuật hóc búa:

- **Học tăng cường độc lập (Independent RL):** Mỗi tác tử tự học một mình, khiến môi trường dưới góc nhìn của từng tác tử luôn bất định, không dừng (non-stationary). Đây là một điểm yếu nghiêm trọng.
- **Học tăng cường tập trung (Centralized RL):** Một tác tử duy nhất kiểm soát tất cả, điều này không thể mở rộng quy mô và vi phạm các ràng buộc thực thi độc lập.
- **CTDE (Centralized Training, Decentralized Execution):** Kết hợp ưu điểm của cả hai: huấn luyện bằng thông tin toàn cục, nhưng triển khai thực thi hoàn toàn bằng các chính sách cục bộ.

## Khái niệm (Concept)

### Ba môi trường kinh điển được sử dụng trong các bài báo

- **Particle World (Môi trường hạt đa tác tử):** Mô phỏng vật lý 2D đơn giản với các tác vụ hợp tác/cạnh tranh. Đây là môi trường thử nghiệm gốc của MADDPG.
- **StarCraft Multi-Agent Challenge (SMAC):** Bài toán vi điều khiển (micro-management) hợp tác, quan sát một phần (partial observation). Môi trường thử nghiệm của QMIX. Không gian hành động rời rạc, không gian trạng thái liên tục.
- **Google Research Football, Hanabi, MPE:** Các bài toán kiểm chuẩn tiêu chuẩn của MAPPO.

Mỗi môi trường sở hữu các dạng hành động và không gian quan sát khác nhau. Tùy thuộc vào đặc tính đó mà chúng ta lựa chọn thuật toán phù hợp.

### MADDPG (2017) — Mẫu hình CTDE

Mỗi tác tử $i$ sở hữu một mạng hành động (actor) $\mu_i(o_i)$ ánh xạ từ quan sát riêng của chính nó $o_i$ thành hành động. Mỗi tác tử cũng sở hữu một mạng đánh giá (critic) $Q_i(x, a_1, \ldots, a_n)$ quan sát được toàn bộ trạng thái $x$ và tất cả hành động của toàn bộ các tác tử trong giai đoạn huấn luyện. Mạng hành động được cập nhật bằng phương pháp đạo hàm chính sách (policy gradient) dựa trên đánh giá của mạng critic.

```text
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Lý do chọn CTDE: Trong thời gian huấn luyện, chúng ta nắm được hành động của tất cả mọi người; chúng ta tận dụng điều đó để triệt tiêu phương sai (variance) trong từng mạng critic. Khi đưa vào vận hành thực tế (deploy time), mỗi tác tử chỉ nhìn thấy $o_i$ của riêng mình và gọi $\mu_i(o_i)$.

Điểm thất bại tiềm ẩn (Failure mode): Kích thước mạng critic phình to theo số lượng $N$ tác tử (vì đầu vào chứa toàn bộ các hành động). Thuật toán không thể mở rộng vượt quá ~10 tác tử nếu không sử dụng các kỹ thuật xấp xỉ.

### QMIX (2018) — Phân rã giá trị (Value Decomposition)

Chỉ áp dụng cho môi trường hợp tác thuần túy. Giá trị hành động chung $Q_{\text{tot}}$ là một hàm trộn đơn điệu (monotonic mixing function) từ các giá trị $Q$ của từng tác tử:

```math
Q_{\text{tot}}(\tau, a) = f\big(Q_1(\tau_1, a_1), \ldots, Q_n(\tau_n, a_n)\big), \quad \frac{\partial f}{\partial Q_i} \ge 0
```

Tính chất đơn điệu đảm bảo rằng phép toán $\arg\max_a Q_{\text{tot}}$ có thể được tính toán bằng cách để mỗi tác tử tự chọn $\arg\max_{a_i} Q_i$ một cách độc lập. Đây **chính xác là thuộc tính thực thi phi tập trung** mà bạn cần. Trong quá trình huấn luyện, một mạng trộn (mixing network) sẽ tổng hợp $Q_{\text{tot}}$ từ các giá trị $Q$ của từng tác tử.

Lý do QMIX thống trị trên SMAC: Tác vụ vi điều khiển hợp tác trong StarCraft có các tác tử đồng nhất (homogeneous agents), quan sát cục bộ và nhận phần thưởng toàn cục — hoàn toàn tương thích với cơ chế phân rã giá trị.

Điểm thất bại tiềm ẩn: Ràng buộc đơn điệu mang tính hạn chế cao; một số tác vụ có cấu trúc phần thưởng không thể phân rã đơn điệu (ví dụ: một tác tử phải hy sinh thân mình vì lợi ích chung của toàn đội). Các biến thể mở rộng (QTRAN, QPLEX) ra đời để nới lỏng ràng buộc này.

### MAPPO (2022) — Lựa chọn mặc định từng bị bỏ quên

Multi-Agent PPO: Thuật toán PPO kết hợp với hàm giá trị tập trung. Mỗi tác tử có chính sách riêng; tất cả tác tử dùng chung (hoặc sở hữu riêng) các hàm giá trị có khả năng quan sát toàn bộ trạng thái toàn cục. Yu et al. 2022 đã so sánh MAPPO với MADDPG, QMIX và các bản mở rộng của chúng trên năm bộ chuẩn và phát hiện ra rằng:

- MAPPO đạt kết quả ngang bằng hoặc vượt trội hơn các phương pháp MARL ngoài chính sách (off-policy) trên particle-world, SMAC, Google Research Football, Hanabi và MPE.
- Đòi hỏi rất ít công sức tinh chỉnh siêu tham số.
- Huấn luyện ổn định; có tính tái lập cao trên nhiều hạt giống ngẫu nhiên (seeds).

Cộng đồng học máy từng đánh giá thấp phương pháp MARL trên chính sách (on-policy) cho đến khi bài báo này xuất hiện. Vào năm 2026, MAPPO là mô hình cơ sở mặc định cho các bài toán MARL hợp tác; bất kỳ phương pháp mới nào cũng phải vượt qua được nó.

### Tại sao kỹ sư tác tử LLM cần quan tâm đến điều này

Ba ứng dụng trực tiếp:

1. **Huấn luyện bộ định tuyến (Router training):** Một siêu tác tử (meta-agent) lựa chọn tác tử con nào sẽ đảm nhận tác vụ. Đây chính là một bài toán MARL với $N$ tác tử con phi tập trung và một bộ định tuyến tập trung. MAPPO hoàn toàn phù hợp.
2. **Sự hình thành vai trò tự phát (Role emergence):** Trong các mô phỏng tác tử sinh (generative-agent simulations), việc huấn luyện các tác tử tự thích ứng với các vai trò bổ trợ lẫn nhau theo thời gian là một bài toán MARL ẩn giấu. Cơ chế phân rã giá trị kiểu QMIX bắt buộc tính bổ trợ phải hình thành ngay từ cấu trúc thiết kế.
3. **Sử dụng công cụ đa tác tử (Multi-agent tool use):** Khi các tác tử dùng chung công cụ và cạnh tranh nhau về ngân sách token/thời gian, việc huấn luyện chúng thông qua CTDE sẽ tạo ra các chính sách cục bộ sẵn sàng triển khai mà vẫn tuân thủ nghiêm ngặt các ràng buộc tài nguyên.

Lưu ý thực tế: Vào năm 2026, đa số các hệ thống tác tử LLM trong môi trường sản xuất định hình chính sách của mình thông qua kỹ thuật prompt thay vì huấn luyện trọng số. MARL chỉ thực sự phát huy giá trị khi bạn hội tụ đủ: (a) lượng lớn dữ liệu tương tác, (b) tín hiệu phần thưởng rõ ràng, và (c) sẵn sàng đầu tư hạ tầng huấn luyện bài bản.

### CTDE như một mẫu hình thiết kế vượt ra ngoài phạm vi RL

Ngay cả khi không huấn luyện mô hình, CTDE vẫn là một mẫu hình kiến trúc cực kỳ hữu ích:

- Trong giai đoạn *thiết kế*, giả định toàn bộ hệ thống có tầm nhìn bao quát (full team visibility).
- Tại *thời điểm chạy (runtime)*, bắt buộc thực thi phi tập trung: mỗi tác tử chỉ nhìn thấy $o_i$ của chính nó.

Mẫu hình này buộc bạn phải làm tường minh trạng thái của từng tác tử và cân nhắc thấu đáo về tính quan sát một phần ngay từ đầu. Nhiều hệ thống đa tác tử thực tế âm thầm giả định trạng thái dùng chung ở khắp mọi nơi — tính kỷ luật của CTDE sẽ ngăn chặn sai lầm đó.

### Vấn đề môi trường bất định (Non-stationarity)

Khi nhiều tác tử cùng học tập đồng thời, môi trường của mỗi tác tử (vốn bao gồm cả chính sách của các tác tử khác) liên tục biến đổi (không dừng - non-stationary). Các định lý chứng minh toán học của RL đơn tác tử cổ điển đều bị sụp đổ. Các thuật toán MARL trong bài học này đều giải quyết bài toán đó:

- MADDPG: Mạng critic toàn cục nhìn thấy toàn bộ hành động, do đó ước lượng giá trị của nó luôn ổn định (stationary).
- QMIX: Cơ chế phân rã giá trị chuyển dịch quá trình học sang không gian $Q$-chung, nơi tính tối ưu được định nghĩa rõ ràng.
- MAPPO: Hàm giá trị tập trung giúp triệt tiêu phương sai phát sinh từ sự biến đổi chính sách của các tác tử khác.

Trong các hệ tác tử LLM, tính bất định này bộc lộ dưới dạng: "Tháng trước tác tử của tôi chạy rất ngon lành, giờ đây tác tử phía trên (upstream) vừa thay đổi một chút là tác tử của tôi lập tức hành xử sai lệch." Huấn luyện MARL theo nguyên lý CTDE là giải pháp căn cơ; việc sửa prompt ở tầng bề mặt tuy nhanh hơn nhưng kém bền vững.

### Nội dung bài học này KHÔNG đề cập

Việc huấn luyện các mạng nơ-ron thực tế là chủ đề của Phase 09. Bài học này xây dựng các phiên bản chính sách theo kịch bản (scripted-policy) nhằm minh họa các mẫu hình CTDE, phân rã giá trị và hàm giá trị tập trung mà không cần cập nhật đạo hàm. Mục tiêu là giúp bạn nắm vững bản chất cốt lõi của các mẫu hình trước khi tiếp cận các thư viện MARL hoàn chỉnh (PyMARL, MARLlib, RLlib multi-agent).

````figure
sw-ctde
````

## Tự tay xây dựng (Build It)

Tệp `code/main.py` hiện thực hóa ba minh họa mẫu hình trên một môi trường lưới đồ chơi hợp tác 2 tác tử:

- Môi trường: 2 tác tử trên lưới $4 \times 4$, một viên thức ăn phần thưởng (reward pellet). Phần thưởng = 1 nếu bất kỳ tác tử nào chạm được viên thức ăn; tác vụ kết thúc.
- `IndependentAgents` — Mỗi tác tử xem các tác tử khác như một phần của môi trường xung quanh. Đây là mô hình cơ sở (baseline).
- `MADDPGStyle` — Mạng critic tập trung tính toán giá trị chung; các chính sách của actor cập nhật dựa trên giá trị đó. Cải tiến chính sách theo kịch bản.
- `QMIXStyle` — Phân rã giá trị với mạng trộn đơn điệu.
- `MAPPOStyle` — Hàm giá trị tập trung; các chính sách cập nhật dựa trên đường cơ sở dùng chung.

Cả bốn biến thể đều chạy qua cùng một tập hợp các tập thử nghiệm (episodes) và báo cáo số bước trung bình để đạt mục tiêu. Các biến thể CTDE nhanh chóng hội tụ về các đường đi ngắn hơn nhiều so với mô hình cơ sở độc lập.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả kỳ vọng: Các tác tử độc lập mất trung bình ~6 bước; các biến thể CTDE hội tụ về ~3,5 bước (đường đi tối ưu trên lưới $4 \times 4$ là 3 bước). Sự khác biệt về mặt kiến trúc mẫu hình được bộc lộ rõ rệt ngay cả với các chính sách dựa trên kịch bản.

## Ứng dụng thực tế (Use It)

Tài liệu `outputs/skill-marl-picker.md` là một kỹ năng (skill) giúp bạn chọn đúng thuật toán MARL cho một tác vụ đa tác tử cụ thể: hợp tác hay cạnh tranh, đồng nhất hay không đồng nhất, loại không gian hành động, quy mô, và dạng tín hiệu phần thưởng.

## Triển khai thực tế (Ship It)

Việc áp dụng MARL vào môi trường sản xuất thực tế là tương đối hiếm. Khi bạn quyết định sử dụng:

- **Bắt đầu với MAPPO:** Bài báo năm 2022 đã chứng minh đây là mô hình cơ sở vững chắc nhất; việc tái lập nó trước tiên sẽ giúp bạn tiết kiệm hàng tuần theo đuổi các phương pháp phức tạp hơn.
- **Ghi log đầy đủ luồng quan sát và hành động của từng tác tử:** Việc gỡ lỗi MARL mà thiếu dấu vết truy vết của từng tác tử là bất khả thi.
- **Tách bạch hoàn toàn mã huấn luyện và mã thực thi:** CTDE là một kỷ luật thiết kế; hãy đảm bảo luồng thực thi trong môi trường chạy chỉ được phép nhìn thấy $o_i$.
- **Cảnh báo về định hình phần thưởng (Reward shaping warning):** MARL cực kỳ nhạy cảm với việc thiết kế hàm phần thưởng. Một lỗi logic nhỏ trong việc phối hợp có thể khiến các tác tử học cách bẫy và khai thác lỗ hổng đó (reward hacking). Hãy luôn chạy các bài kiểm thử đối kháng (adversarial tests).
- **Đối với các tác tử LLM**, hãy ưu tiên cân nhắc các chính sách ở tầng prompt trước. Chỉ đầu tư vào huấn luyện MARL khi bạn đã có sẵn dữ liệu tương tác phong phú, tín hiệu phần thưởng rõ ràng và hạ tầng tính toán tương ứng.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Đo đạc khoảng cách về số bước tới mục tiêu giữa các tác tử độc lập và tác tử kiểu MAPPO. Khoảng cách này mở rộng hay thu hẹp lại khi chuyển sang lưới $6 \times 6$?
2. Hiện thực hóa một biến thể cạnh tranh: hai tác tử, một viên thức ăn, chỉ tác tử nào đến trước mới nhận được phần thưởng. Mẫu hình nào xử lý sự cạnh tranh một cách mạch lạc? Lịch sử chứng minh MADDPG là ứng viên hàng đầu.
3. Đọc Mục 3 bài báo MADDPG (arXiv:1706.02275). Tự viết lại quy tắc cập nhật mạng critic một cách chính xác dưới dạng mã giả theo ngôn từ của bạn.
4. Đọc bài báo MAPPO (arXiv:2103.01955). Tại sao các tác giả lại lập luận rằng việc kết hợp hàm giá trị tập trung với PPO có thể đánh bại các phương pháp MARL ngoài chính sách trên các bộ chuẩn của họ? Hãy liệt kê ba luận điểm đanh thép nhất.
5. Áp dụng CTDE như một mẫu hình thiết kế cho một hệ thống tác tử LLM giả định (ví dụ: tác tử nghiên cứu + tác tử tóm tắt + tác tử lập trình). Thông tin chung nào có thể biết được ở giai đoạn thiết kế mà không thể truy cập được tại thời điểm chạy?

## Các thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Điều mọi người hay nói | Ý nghĩa bản chất thực tế |
|---|---|---|
| MARL | "Học tăng cường đa tác tử" | Reinforcement learning dành cho các hệ thống có nhiều tác tử tương tác. |
| CTDE | "Huấn luyện tập trung, thực thi phi tập trung" | Huấn luyện bằng thông tin toàn cục; vận hành thực tế bằng các chính sách cục bộ. |
| MADDPG | "DDPG đa tác tử" | Kiến trúc CTDE với mạng critic của từng tác tử quan sát toàn bộ trạng thái và hành động. |
| QMIX | "Phân rã hàm giá trị" | Kết hợp đơn điệu các hàm $Q$ thành phần. Dành cho môi trường hợp tác. |
| MAPPO | "PPO đa tác tử" | PPO với hàm giá trị tập trung. Mô hình cơ sở mặc định của năm 2026. |
| Phân rã giá trị (Value decomposition) | "Tổng các hàm $Q$ riêng lẻ" | Biểu diễn hàm $Q$ toàn cục dưới dạng một hàm đơn điệu của các hàm $Q$ thành phần. |
| Tính không dừng (Non-stationarity) | "Mục tiêu di động" | Môi trường của mỗi tác tử liên tục biến đổi khi các tác tử khác cùng học. Vấn đề cốt lõi của MARL. |
| Trên chính sách / Ngoài chính sách (On-policy / off-policy) | "Học từ hiện tại / Học từ bộ nhớ đệm" | PPO thuộc loại trên chính sách (MAPPO); DDPG và Q-learning thuộc loại ngoài chính sách. |
| SMAC | "StarCraft Multi-Agent Challenge" | Bộ chuẩn kiểm nghiệm vi điều khiển hợp tác; cái nôi phát triển của QMIX. |

## Tài liệu tham khảo (Further Reading)

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — Bài báo gốc về MADDPG; NeurIPS 2017.
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — Bài báo gốc về QMIX; ICML 2018.
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — Bài báo MAPPO; NeurIPS 2022.
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — Bài viết phân tích trực quan về kết quả nghiên cứu của MAPPO.
- [SMAC repository](https://github.com/oxwhirl/smac) — Mã nguồn StarCraft Multi-Agent Challenge.
