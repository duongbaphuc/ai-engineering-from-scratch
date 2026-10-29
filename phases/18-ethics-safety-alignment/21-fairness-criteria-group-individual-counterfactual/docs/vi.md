# Các Tiêu chí Công bằng — Nhóm, Cá nhân, Phản thực tế (Fairness Criteria — Group, Individual, Counterfactual)

> Ba họ tiêu chí định hình toàn bộ y văn về tính công bằng. Công bằng nhóm (group fairness): tính ngang bằng nhân khẩu học (demographic parity), tỷ lệ cược cân bằng (equalized odds), tính công bằng về độ chính xác sử dụng có điều kiện (conditional use accuracy equality) — các tỷ lệ ngang bằng nhau giữa các nhóm được bảo vệ xét trên mức trung bình. Công bằng cá nhân (individual fairness - Dwork et al. 2012): các cá nhân tương đồng phải nhận được các quyết định tương đồng; ràng buộc bởi điều kiện Lipschitz trên ánh xạ quyết định. Công bằng phản thực tế (counterfactual fairness - Kusner et al. 2017): một quyết định là công bằng đối với một cá nhân nếu nó giữ nguyên không đổi khi các thuộc tính nhạy cảm của cá nhân đó bị thay đổi theo giả định phản thực tế. Kết quả lý thuyết năm 2024 (NeurIPS 2024): tồn tại một sự đánh đổi cố hữu giữa tính công bằng phản thực tế (CF) và độ chính xác dự đoán; một phương pháp bất khả tri mô hình (model-agnostic) có thể biến đổi một bộ dự đoán tối ưu-nhưng-bất-công thành một bộ dự đoán CF với mức tổn thất độ chính xác có giới hạn chặn. Phản thực tế lần ngược (Backtracking counterfactuals - arXiv:2401.13935, Tháng 1/2024): một mô hình tiếp cận mới giúp tránh đòi hỏi phải can thiệp trực tiếp lên các thuộc tính được pháp luật bảo vệ. Sự hòa giải triết học (ICLR Blogposts 2024): khi có đồ thị nhân quả rõ ràng, việc thỏa mãn một số phép đo công bằng nhóm nhất định sẽ kéo theo tính công bằng phản thực tế.

**Loại hình:** Học lý thuyết (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, so sánh ba tiêu chí)  
**Kiến thức tiên quyết:** Phase 18 · 20 (thiên kiến - bias), Phase 02 (học máy cổ điển - classical ML)  
**Thời lượng:** ~60 phút  

## Mục tiêu học tập

- Phát biểu ba tiêu chí công bằng nhóm (tính ngang bằng nhân khẩu học, tỷ lệ cược cân bằng, tính công bằng về độ chính xác sử dụng có điều kiện) và một định lý bất khả thi.
- Mô tả tính công bằng cá nhân thông qua công thức Lipschitz của Dwork et al. 2012.
- Trình bày tính công bằng phản thực tế và sự phụ thuộc của nó vào đồ thị nhân quả.
- Giải thích phương pháp phản thực tế lần ngược và lý do nó giải quyết được bài toán can thiệp lên thuộc tính được bảo vệ.

## Vấn đề đặt ra

Bài 20 tập trung vào việc đo lường thiên kiến. Bài 21 bàn về việc định nghĩa tiêu chuẩn công bằng mà các phép đo đó cần phục vụ. Ba họ tiêu chí đưa ra các tiêu chuẩn có cấu trúc hoàn toàn khác nhau — một mô hình có thể công bằng theo nhóm nhưng bất công theo cá nhân, hoặc công bằng theo phản thực tế nhưng bất công theo nhóm. Lựa chọn tiêu chuẩn nào là một quyết định mang tính chính sách; không có tiêu chuẩn nào tối ưu một cách phổ quát.

## Khái niệm cốt lõi

### Công bằng nhóm (Group Fairness)

- **Tính ngang bằng nhân khẩu học (Demographic parity):** $P(Y=1 | A=a) = P(Y=1 | A=a')$ với mọi nhóm. Tỷ lệ được chấp thuận ngang nhau giữa các nhóm thuộc tính nhạy cảm $A$.
- **Tỷ lệ cược cân bằng (Equalized odds):** $P(Y=1 | Y^*=y, A=a) = P(Y=1 | Y^*=y, A=a')$. Tỷ lệ dương tính thật (TPR - True Positive Rate) và tỷ lệ dương tính giả (FPR - False Positive Rate) ngang nhau giữa các nhóm.
- **Tính công bằng về độ chính xác sử dụng có điều kiện (Conditional use accuracy equality):** $P(Y^*=y | Y=y, A=a) = P(Y^*=y | Y=y, A=a')$. Giá trị dự đoán dương (PPV) và giá trị dự đoán âm (NPV) ngang nhau giữa các nhóm.

**Định lý bất khả thi (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017):** Ba tiêu chí này không thể được thỏa mãn đồng thời khi tỷ lệ cơ sở (base rates - $P(Y^*=1|A)$) giữa các nhóm là không đồng đều, trừ phi bộ dự đoán đạt độ chính xác hoàn hảo.

### Công bằng cá nhân (Individual Fairness)

Dwork et al. 2012. Một ánh xạ quyết định $f$ được coi là công bằng với từng cá nhân đối với một hàm khoảng cách tương đồng $d$ đặc thù cho tác vụ nếu:
$$|f(x) - f(x')| \le L \cdot d(x, x')$$
với một hằng số Lipschitz $L$ nào đó. Nghĩa là: các cá nhân có đặc điểm tương đồng phải nhận được các quyết định tương đồng.

Phương pháp này đòi hỏi phải định nghĩa được hàm khoảng cách $d$. Đây là một câu hỏi thuộc về chính sách và chuẩn mực xã hội, không phải là một bài toán thuần thống kê.

### Công bằng phản thực tế (Counterfactual Fairness)

Kusner et al. 2017. Một quyết định được coi là công bằng phản thực tế đối với một cá nhân $i$ nếu, dưới một mô hình nhân quả của quần thể, quyết định đó không thay đổi khi các thuộc tính nhạy cảm của $i$ bị biến đổi một cách giả định phản thực tế (counterfactually altered).

Phương pháp này đòi hỏi một đồ thị có hướng phi chu trình nhân quả (causal DAG). Đồ thị DAG này là một giả định mô hình hóa. Do đó, tính công bằng phản thực tế chỉ có giá trị khi cấu trúc DAG được biện minh một cách chặt chẽ.

### Đánh đổi giữa CF và độ chính xác (The CF-vs-accuracy Trade-off)

Nghiên cứu lý thuyết tại NeurIPS 2024: tồn tại một sự đánh đổi cố hữu giữa tính công bằng phản thực tế và độ chính xác dự đoán. Một phương pháp bất khả tri mô hình (model-agnostic) có thể chuyển đổi một bộ dự đoán tối ưu-nhưng-không-công-bằng thành một bộ dự đoán đạt chuẩn CF, với chi phí tổn thất độ chính xác bị chặn trên. Mức tổn thất độ chính xác này phụ thuộc trực tiếp vào độ lớn của hệ số thuộc tính nhạy cảm trong bộ dự đoán tối ưu bất công ban đầu.

### Phản thực tế lần ngược (Backtracking Counterfactuals)

arXiv:2401.13935 (Tháng 1/2024). Phương pháp phản thực tế truyền thống đòi hỏi việc can thiệp trực tiếp lên thuộc tính nhạy cảm — "liệu quyết định có thay đổi nếu người này mang một giới tính khác". Về mặt pháp lý, điều này gây tranh cãi: luật chống phân biệt đối xử không cho phép can thiệp trực tiếp lên các thuộc tính được bảo vệ trong quy trình phân loại.

Phản thực tế lần ngược đảo ngược hướng tiếp cận: thay vì can thiệp vào thuộc tính, nó truy vấn xem tổ hợp các đặc trưng thực tế nào của cá nhân đó có thể tạo ra kết quả phản thực tế mong muốn. Điều này giúp né tránh được rào cản pháp lý nói trên.

### Sự hòa giải triết học (Philosophical Reconciliation)

ICLR Blogposts 2024. Khi có sẵn một đồ thị nhân quả, việc thỏa mãn một số thước đo công bằng nhóm nhất định sẽ trực tiếp kéo theo tính công bằng phản thực tế. Ba họ tiêu chí này không hoàn toàn trực giao; chúng là các lát cắt phản chiếu khác nhau của cùng một cấu trúc nhân quả nền tảng.

Phát hiện này không xóa bỏ các định lý bất khả thi (tỷ lệ cơ sở bất bình đẳng vẫn ngăn cản việc thỏa mãn đồng thời các tiêu chí công bằng nhóm). Nhưng nó chứng minh rằng sự mâu thuẫn bề ngoài giữa "công bằng nhóm" và "công bằng cá nhân / phản thực tế" phần nào là hệ quả của việc không minh bạch hóa mô hình nhân quả.

### Vị trí trong Phase 18

Bài 20 là đo lường thiên kiến. Bài 21 là định nghĩa tính công bằng. Bài 22 là quyền riêng tư (quyền riêng tư vi sai - differential privacy). Bài 23 là thủy vân số (watermarking). Đây là các bài học liên quan đến phân bổ quyền lợi và an toàn cho người dùng, bổ trợ cho chuỗi bài về hành vi lừa dối từ Bài 7 đến Bài 11.

```figure
an-fairness-trilemma
```

## Ứng dụng thực tế (Use It)

Tệp `code/main.py` xây dựng một tập dữ liệu phân loại nhị phân đồ chơi có chứa thuộc tính nhạy cảm và tỷ lệ cơ sở không đều. Bạn sẽ tính toán tính ngang bằng nhân khẩu học, tỷ lệ cược cân bằng, và tính công bằng về độ chính xác sử dụng có điều kiện trên một bộ phân loại đơn giản. Quan sát sự xung đột giữa ba chỉ số này. Áp dụng kỹ thuật gán lại trọng số (re-weighting) để đạt được tính ngang bằng nhân khẩu học và quan sát cái giá phải trả trên hai thước đo còn lại.

## Triển khai thành phẩm (Ship It)

Bài học này xây dựng tài liệu `outputs/skill-fairness-criterion.md`. Khi tiếp nhận một chính sách hoặc một tuyên bố về tính công bằng, tài liệu này giúp xác định tiêu chí nào đang được tuyên bố, liệu mô hình có thể thỏa mãn các tiêu chí còn lại dưới tỷ lệ cơ sở bất bình đẳng hay không, và tuyên bố đó phụ thuộc vào DAG nhân quả nào.

## Bài tập

1. Chạy `code/main.py`. Báo cáo ba chỉ số công bằng nhóm trên dữ liệu mặc định. Áp dụng kỹ thuật gán trọng số nhắm vào tính ngang bằng nhân khẩu học và báo cáo lại các kết quả.

2. Cài đặt chỉ số công bằng cá nhân theo Dwork et al. 2012 bằng cách sử dụng khoảng cách $L_2$ trên các đặc trưng không nhạy cảm. Báo cáo số lượng cặp cá nhân vi phạm điều kiện Lipschitz với hằng số $L=1$.

3. Đọc bài báo Kusner et al. 2017. Xây dựng một DAG nhân quả hai đặc trưng đơn giản cho bài toán chấm điểm hồ sơ và xác định điều kiện công bằng phản thực tế mà nó hàm ý.

4. Bài báo phản thực tế lần ngược năm 2024 tránh can thiệp lên các thuộc tính được bảo vệ. Hãy mô tả một tình huống mà điều này có ý nghĩa quyết định đối với việc tuân thủ pháp luật.

5. Luận điểm hòa giải tại ICLR 2024 cho rằng công bằng nhóm và công bằng phản thực tế là các khía cạnh của cùng một cấu trúc. Hãy chọn hai trong ba tiêu chí trong `code/main.py` và nêu giả định nhân quả giúp biến chúng thành tương đương.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Demographic parity (Ngang bằng nhân khẩu học) | "Tỷ lệ chấp thuận đều nhau" | $P(Y=1 \| A=a)$ ngang bằng nhau giữa mọi nhóm |
| Equalized odds (Tỷ lệ cược cân bằng) | "TPR/FPR đều nhau" | Tỷ lệ dương tính thật và dương tính giả ngang bằng nhau giữa các nhóm |
| Conditional use accuracy (Độ chính xác sử dụng có điều kiện) | "PPV/NPV đều nhau" | Các giá trị dự đoán dương/âm ngang bằng nhau giữa các nhóm |
| Individual fairness (Công bằng cá nhân) | "Điều kiện Lipschitz" | Các cá nhân tương đồng phải nhận được quyết định tương đồng |
| Counterfactual fairness (Công bằng phản thực tế) | "Bất biến khi đổi thuộc tính" | Quyết định không đổi dưới sự can thiệp phản thực tế lên thuộc tính nhạy cảm |
| Backtracking counterfactual (Phản thực tế lần ngược) | "Giải thích qua dữ liệu thực" | Phản thực tế được suy luận ngược từ kết quả thay vì can thiệp xuôi từ thuộc tính |
| Impossibility theorem (Định lý bất khả thi) | "Ba tiêu chí xung đột" | Chouldechova / KMR 2017: Các tiêu chí nhóm triệt tiêu lẫn nhau khi tỷ lệ cơ sở khác biệt |

## Tài liệu đọc thêm

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — Công trình đặt nền móng cho công bằng cá nhân.
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — Khung lý thuyết công bằng phản thực tế dựa trên mô hình nhân quả.
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — Chứng minh định lý bất khả thi trong công bằng thuật toán.
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — Phương pháp tiếp cận mới né tránh can thiệp thuộc tính bảo vệ.
