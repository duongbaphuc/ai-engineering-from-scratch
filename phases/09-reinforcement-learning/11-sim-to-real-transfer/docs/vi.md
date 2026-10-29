# Chuyển giao từ Mô phỏng sang Thực tế (Sim-to-Real Transfer)

> Một chính sách được huấn luyện trong môi trường mô phỏng nhưng thất bại khi chạy trên phần cứng thực tế là chính sách đã học vẹt môi trường mô phỏng. Ngẫu nhiên hóa miền (domain randomization), thích ứng miền (domain adaptation) và nhận dạng hệ thống (system identification) là ba công cụ cốt lõi giúp các bộ điều khiển học được vượt qua khoảng cách thực tế (reality gap).

**Loại:** Học lý thuyết (Learn)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Thời lượng:** ~45 phút

## Vấn đề đặt ra (The Problem)

Huấn luyện trực tiếp trên một robot thực tế là quá trình chậm chạp, nguy hiểm và vô cùng đắt đỏ. Một robot hai chân cần hàng triệu tập huấn luyện chỉ để học cách đi; nhưng một robot thực chỉ cần ngã một lần duy nhất là phần cứng có thể gãy hỏng hoàn toàn. Môi trường mô phỏng (simulation) mang lại cho bạn khả năng tái thiết lập (resets) vô hạn, tính tái lập xác định, chạy song song hàng nghìn môi trường và hoàn toàn không gây tổn hại vật lý.

Thế nhưng, các bộ mô phỏng luôn có sai số so với thực tế. Các ổ bi có ma sát lớn hơn so với mô hình trong MuJoCo. Ống kính camera thực tế có độ méo quang học mà bộ mô phỏng không tính đến. Động cơ thực có độ trễ, độ rơ (backlash) và hiện tượng bão hòa mô-men xoắn mà 99% các mô hình mô phỏng bỏ qua. Gió, bụi và điều kiện chiếu sáng biến đổi liên tục sẽ phá hỏng một chính sách vốn chỉ được huấn luyện trên môi trường đồ họa vô trùng. **Khoảng cách thực tế** (reality gap) — tức sự sai lệch mang tính hệ thống giữa phân phối trong mô phỏng và phân phối ngoài đời thực — là bài toán trung tâm của việc ứng dụng RL vào robot thực tế.

Bạn cần một chính sách có khả năng *bền bỉ trước sự dịch chuyển phân phối (distribution shift) từ mô phỏng sang thực tế*. Có ba hướng tiếp cận kinh điển: ngẫu nhiên hóa bộ mô phỏng (domain randomization), thích ứng chính sách với một lượng nhỏ dữ liệu thực (domain adaptation / fine-tuning), hoặc đo đạc các tham số của hệ thống thực và cấu hình lại bộ mô phỏng cho khớp (system identification). Đến năm 2026, công thức thống trị là kết hợp cả ba phương pháp này với sức mạnh mô phỏng song song quy mô lớn trên GPU (Isaac Sim, Isaac Lab, MuJoCo MJX).

## Khái niệm (The Concept)

![Ba cơ chế sim-to-real: ngẫu nhiên hóa miền, thích ứng miền, nhận dạng hệ thống](../assets/sim-to-real.svg)

**Ngẫu nhiên hóa Miền (Domain Randomization - DR).** Tobin cùng các cộng sự 2017, Peng cùng các cộng sự 2018. Trong quá trình huấn luyện, hãy ngẫu nhiên hóa mọi tham số mô phỏng có khả năng sai khác trên robot thực: khối lượng, hệ số ma sát, hệ số độ lợi PD của động cơ, độ nhiễu cảm biến, vị trí camera, ánh sáng, vân bề mặt (textures), và mô hình tiếp xúc va chạm. Chính sách sẽ học một phân phối có điều kiện để nhận biết "hôm nay nó đang ở môi trường mô phỏng nào" và khái quát hóa trên toàn bộ dải biến thiên đó. Nếu robot thực tế rơi vào đúng dải bao phủ đã huấn luyện, chính sách sẽ hoạt động trơn tru.

- **Ưu điểm:** không cần bất kỳ dữ liệu thực tế nào. Một công thức có thể áp dụng cho nhiều robot.
- **Nhược điểm:** ngẫu nhiên hóa quá mức sẽ tạo ra một chính sách "vạn năng" nhưng cực kỳ rụt rè và thận trọng quá mức. Quá nhiều nhiễu tương đương với việc áp dụng điều chuẩn (regularization) quá mạnh.

**Nhận dạng Hệ thống (System Identification - SI).** Khớp các tham số của bộ mô phỏng với dữ liệu thu thập từ thế giới thực trước khi huấn luyện. Nếu bạn có thể đo đạc chính xác ma sát tại khớp tay của robot thực, hãy đưa trực tiếp giá trị đó vào bộ mô phỏng. Sau đó, huấn luyện chính sách trên đúng các giá trị này. Đòi hỏi phải tiếp cận phần cứng thực tế, nhưng trực tiếp thu hẹp khoảng cách thực tế.

- **Ưu điểm:** mục tiêu huấn luyện chính xác, ít nhiễu.
- **Nhược điểm:** sai số mô hình còn sót lại (residual model error) sẽ bị chính sách bỏ qua; các hiệu ứng vật lý nhỏ chưa được nhận dạng (như vùng chết - deadband của động cơ) vẫn có thể khiến việc triển khai thất bại.

**Thích ứng Miền (Domain Adaptation).** Huấn luyện trước trong mô phỏng, sau đó tinh chỉnh với một lượng nhỏ dữ liệu từ thế giới thực. Gồm hai biến thể chính:

- **Real2Sim2Real:** học một mô hình mô phỏng phần dư $f(s, a, z) - f_{\text{sim}}(s, a)$ dựa trên các quỹ đạo chạy thực tế, sau đó huấn luyện tiếp trong môi trường mô phỏng đã được hiệu chỉnh. Thu hẹp khoảng cách mà không tốn nhiều dữ liệu thực.
- **Thích ứng quan sát (Observation adaptation):** huấn luyện một chính sách ánh xạ quan sát thực $\to$ quan sát giống mô phỏng thông qua một bộ trích xuất đặc trưng học được (ví dụ: mạng GAN chuyển đổi ảnh-sang-ảnh). Bộ điều khiển cốt lõi vẫn giữ nguyên trong miền mô phỏng.

**Học đặc quyền / Thầy - Trò (Privileged learning / Teacher-Student).** Miki cùng các cộng sự 2022 (robot bốn chân ANYmal). Huấn luyện một mạng *thầy* (teacher) trong môi trường mô phỏng với quyền truy cập đầy đủ vào các thông tin đặc quyền (thông tin chân thực về ma sát, độ cao địa hình, độ trôi của cảm biến quán tính IMU). Sau đó, chưng cất (distill) sang một mạng *trò* (student) vốn chỉ nhìn thấy các quan sát từ cảm biến thực tế. Mạng học trò học cách suy diễn các đặc trưng đặc quyền này từ lịch sử quan sát, mang lại sự bền bỉ trên nhiều thông số vật lý khác nhau.

**Mô phỏng song song quy mô lớn (Massively parallel simulation).** Giai đoạn 2024–2026. Isaac Lab, MuJoCo MJX, Brax đều có khả năng chạy hàng nghìn robot song song trên một GPU duy nhất. Thuật toán PPO với 4.096 robot hình người song song có thể thu thập lượng kinh nghiệm tương đương nhiều năm chỉ trong vài giờ. "Khoảng cách thực tế" bị thu hẹp đáng kể khi phân phối huấn luyện được mở rộng; DR trở nên gần như miễn phí khi mỗi môi trường trong số 4.096 môi trường đó mang một tập tham số ngẫu nhiên khác nhau.

**Công thức triển khai thực tế năm 2026 (ví dụ về robot bốn chân tập đi):**

1. Chạy mô phỏng song song quy mô lớn với ngẫu nhiên hóa miền về trọng lực, ma sát, hệ số khuếch đại động cơ, tải trọng mang thêm.
2. Huấn luyện chính sách mạng thầy với thông tin đặc quyền (bản đồ địa hình, vận tốc thân thực sự).
3. Chưng cất chính sách mạng trò từ mạng thầy chỉ sử dụng cảm nhận bản thể (proprioception - các bộ mã hóa góc quay khớp chân).
4. (Tùy chọn) Thích ứng quan sát qua autoencoder trên tín hiệu IMU thực.
5. Triển khai. Chuyển giao tức thì không cần mẫu (zero-shot) trên hơn 10 môi trường khác nhau. Nếu thất bại, tinh chỉnh vài phút trên thế giới thực bằng thuật toán PPO có ràng buộc an toàn.

```figure
f3-reality-gap
```

## Thực hành Xây dựng (Build It)

Mã nguồn bài học này minh họa một ví dụ nhỏ về ngẫu nhiên hóa miền trên môi trường GridWorld với các bước chuyển trạng thái *có nhiễu*. Chúng ta huấn luyện một chính sách trải nghiệm các xác suất trơn trượt (slip) ngẫu nhiên trong "mô phỏng" và đánh giá trên môi trường "thực tế" với mức độ trơn trượt mà nó chưa từng thấy trong quá trình huấn luyện. Cấu trúc này ánh xạ trực tiếp sang bài toán chuyển giao từ MuJoCo sang phần cứng thực tế.

### Bước 1: Bộ mô phỏng có thể tham số hóa

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` là một tham số mà bộ mô phỏng cung cấp. Trong ngành robot thực tế, đây có thể là ma sát, khối lượng, độ lợi động cơ — bất kỳ yếu tố nào có sự sai khác giữa mô phỏng và đời thực.

### Bước 2: Huấn luyện với ngẫu nhiên hóa miền (DR)

Khi bắt đầu mỗi tập (episode), lấy mẫu `slip ~ Uniform[0.0, 0.4]`. Huấn luyện PPO / Q-learning / bất kỳ thuật toán nào. Thực hiện qua nhiều tập.

### Bước 3: Đánh giá zero-shot trên các độ trượt "thực tế"

Đánh giá trên tập giá trị `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Bốn giá trị đầu tiên nằm trong khoảng đã huấn luyện; `0.5` và `0.7` nằm ngoài khoảng (out-of-distribution). Một chính sách được huấn luyện bằng DR sẽ duy trì hiệu năng gần tối ưu trong vùng bao phủ và suy giảm nhẹ nhàng ngoài vùng bao phủ. Ngược lại, một chính sách chỉ học trên một độ trượt cố định sẽ trở nên cực kỳ mong manh khi gặp độ trượt khác.

### Bước 4: So sánh với huấn luyện hạn hẹp

Huấn luyện một chính sách thứ hai chỉ với giá trị cố định `slip = 0.0`. Đánh giá trên cùng dải quét `slip`. Bạn sẽ thấy hiệu năng rơi tự do thảm hại ngay khi độ trượt thực tế > 0.

## Bẫy thường gặp (Pitfalls)

- **Ngẫu nhiên hóa quá đà.** Nếu huấn luyện trên `slip ∈ [0, 0.9]`, chính sách sẽ sợ rủi ro đến mức không bao giờ dám thử con đường tối ưu. Hãy khớp với phân phối *kỳ vọng* ngoài đời thực, thay vì giả định "bất kỳ điều gì cũng có thể xảy ra".
- **Ngẫu nhiên hóa quá ít.** Huấn luyện trên một dải tham số quá hẹp sẽ khiến chính sách mất hoàn toàn khả năng khái quát hóa. Hãy dùng giáo trình thích ứng (Automatic Domain Randomization - ADR) để mở rộng dần phân phối khi năng lực chính sách tăng lên.
- **Xác định sai không gian tham số.** Ngẫu nhiên hóa nhầm yếu tố (ví dụ: đổi màu camera trong khi lỗi thực sự nằm ở độ trễ động cơ) sẽ khiến DR vô tác dụng. Hãy đo kiểm kỹ robot thực tế trước.
- **Rò rỉ thông tin đặc quyền (Privileged info leakage).** Một mạng thầy dựa vào trạng thái toàn cục để đưa ra hành động, chứ không chỉ dùng quan sát, có thể tạo ra một bài toán bất khả thi khiến mạng trò không thể bắt kịp. Hãy đảm bảo chính sách của thầy hoàn toàn có thể hiện thực hóa được bởi trò chỉ dựa trên lịch sử quan sát.
- **Thất bại khi chuyển giao giữa các bộ mô phỏng (Sim-to-sim transfer failure).** Nếu chính sách của bạn không chịu nổi một biến thể mô phỏng khó hơn, nó chắc chắn sẽ thất bại ngoài đời thực. Luôn kiểm tra trên một biến thể mô phỏng giữ lại độc lập trước khi nạp vào phần cứng.
- **Thiếu lớp bọc an toàn thực tế (Safety envelope).** Một chính sách chạy tốt trong mô phỏng và "có vẻ chạy được trên đời thực" nhưng thiếu tấm chắn an toàn cấp thấp vẫn có thể phá hủy phần cứng bất cứ lúc nào. Luôn cài đặt giới hạn tốc độ, giới hạn mô-men xoắn, giới hạn góc khớp trong một bộ điều khiển phi học máy độc lập.

## Ứng dụng thực tế (Use It)

Ngăn xếp công nghệ sim-to-real năm 2026:

| Lĩnh vực | Ngăn xếp công nghệ |
|----------|--------------------|
| Chân chuyển động (ANYmal, Spot, robot hình người) | Isaac Lab + DR + mạng thầy/trò đặc quyền |
| Thao tác khéo léo (bàn tay nhiều ngón, gắp-đặt) | Isaac Lab + DR + DR-GAN cho thị giác |
| Xe tự hành | CARLA / NVIDIA DRIVE Sim + DR + tinh chỉnh trên xe thực |
| Đua drone | RotorS / Flightmare + DR + thích ứng trực tuyến |
| Thao tác xoay vật thể trong lòng bàn tay | OpenAI Dactyl (DR ở quy mô chưa từng có) |
| Cánh tay robot công nghiệp | MuJoCo-Warp + SI + tinh chỉnh nhỏ trên robot thực |

Đối với bài toán điều khiển ở mọi quy mô, quy trình làm việc luôn nhất quán: khớp bộ mô phỏng sát nhất có thể, ngẫu nhiên hóa những gì không thể khớp, huấn luyện các mô hình chính sách quy mô lớn, chưng cất sang mô hình nhẹ hơn và triển khai kèm một lá chắn an toàn nghiêm ngặt.

## Đóng gói Sản phẩm (Ship It)

Lưu thành `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Bài tập thực hành (Exercises)

1. **Dễ.** Huấn luyện một tác tử Q-learning trên GridWorld có độ trượt cố định (slip=0.0). Đánh giá trên dải $\text{slip} \in \lbrace 0.0, 0.1, 0.3, 0.5 \rbrace$. Vẽ biểu đồ phần thưởng tích lũy theo độ trượt.
2. **Trung bình.** Huấn luyện tác tử Q-learning với DR bằng cách lấy mẫu `slip ~ Uniform[0, 0.3]`. Đánh giá trên cùng dải thử nghiệm trên. Kỹ thuật DR mang lại lợi ích bao nhiêu tại điểm slip=0.5 (nằm ngoài phân phối)?
3. **Khó.** Cài đặt một giáo trình thích ứng: bắt đầu với slip=0.0, mở rộng dải DR mỗi khi chính sách đạt 90% mức tối ưu. Đo tổng số bước môi trường cần thiết để đạt năng lực zero-shot tại slip=0.3 so với đường cơ sở DR dải cố định.

## Thuật ngữ Then chốt (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|-----------|----------------------|-----------------|
| Reality gap | "Khoảng cách giữa sim và thực" | Sự dịch chuyển phân phối giữa vật lý/cảm biến khi huấn luyện và khi triển khai. |
| Domain randomization (DR) | "Huấn luyện trên nhiều sim ngẫu nhiên" | Ngẫu nhiên hóa các tham số mô phỏng trong lúc huấn luyện để chính sách biết khái quát hóa. |
| System identification (SI) | "Đo thực tế rồi chỉnh sim" | Ước lượng các tham số vật lý thực; cấu hình mô phỏng cho khớp tương ứng. |
| Domain adaptation | "Tinh chỉnh trên dữ liệu thực" | Quá trình tinh chỉnh nhỏ trên dữ liệu thực tế sau khi đã huấn luyện trong mô phỏng. |
| Privileged info | "Thông tin chân thực cho thầy" | Dữ liệu chỉ có trong môi trường mô phỏng; học trò phải tự suy diễn từ lịch sử quan sát. |
| Teacher/student | "Chưng cất đặc quyền -> quan sát" | Thầy học bằng dữ liệu đặc quyền; trò học cách bắt chước mà không cần dữ liệu đó. |
| ADR | "Ngẫu nhiên hóa miền tự động" | Giáo trình tự động mở rộng dải tham số DR khi chính sách ngày một tốt hơn. |
| Real2Sim | "Kéo sim về gần thực bằng dữ liệu" | Học mô hình phần dư để làm cho mô phỏng phản ánh chính xác các lượt chạy thực tế. |

## Tài liệu đọc thêm (Further Reading)

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — Bài báo DR nguyên bản (thị giác máy tính cho robotics).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR cho động lực học, chuyển động robot bốn chân.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dự án Dactyl, ADR ở quy mô lớn.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — Mô hình Thầy - Trò cho robot ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — Bộ mô phỏng vật lý song song quy mô lớn trên GPU mở đường cho kỷ nguyên 2025–2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — Phương pháp giáo trình ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Khung kiến trúc Dyna (dùng mô hình để lập kế hoạch + sinh quỹ đạo rollout) làm nền tảng cho các quy trình sim-to-real hiện đại.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — Bản phân loại hệ thống các phương pháp sim-to-real kèm kết quả benchmark.
