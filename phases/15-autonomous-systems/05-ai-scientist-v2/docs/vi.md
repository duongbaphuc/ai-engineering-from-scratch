# AI Scientist v2 — Nghiên cứu Tự trị Cấp độ Hội thảo (Workshop-Level Autonomous Research)

> AI Scientist v2 của Sakana (Yamada et al., arXiv:2504.08066) vận hành toàn bộ vòng lặp nghiên cứu: giả thuyết, mã nguồn, thực nghiệm, biểu đồ, bài viết, nộp bài. Đây là hệ thống đầu tiên có một bài báo do nó tự sinh ra vượt qua bình duyệt đồng cấp (peer review) tại một hội thảo chuyên đề (workshop) của ICLR 2025. Đánh giá độc lập (Beel et al.) phát hiện 42% các thực nghiệm thất bại do lỗi lập trình và khâu duyệt tài liệu tham khảo thường xuyên dán nhãn nhầm các khái niệm đã có thành cái mới. Bản thân tài liệu của Sakana cũng cảnh báo rằng cơ sở mã thực thi mã do LLM viết và khuyến nghị cô lập bằng Docker. Cả hai mặt của bức tranh đó chính là trọng tâm của vấn đề.

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 phút

## Vấn đề

Nghiên cứu là một tác vụ mở không giới hạn (open-ended task). Không giống như việc tìm kiếm thuật toán của AlphaEvolve hay sự tự sửa đổi bị giới hạn bởi điểm chuẩn của DGM, một kết quả nghiên cứu không có tiêu chí đúng đắn có thể kiểm tra tự động bằng máy. Một bài báo được đánh giá bởi các chuyên gia bình duyệt, không phải bằng các bài kiểm thử đơn vị. Điều đó khiến vòng lặp khó khép kín hơn — nhưng lại giá trị hơn nhiều nếu làm được, bởi vì nghiên cứu chính là nơi khởi nguồn của sự tiến bộ theo cấp số nhân.

AI Scientist v1 (Sakana, 2024) đã khép kín vòng lặp bằng cách bắt đầu từ các bản mẫu (templates) do con người biên soạn. LLM điền các thực nghiệm vào bên trong một khung giàn cố định. AI Scientist v2 (Yamada et al., 2025) loại bỏ yêu cầu về bản mẫu bằng cách sử dụng tìm kiếm dạng cây của tác tử (agentic tree search) kết hợp với vòng lặp phê bình bằng mô hình thị giác-ngôn ngữ (vision-language model - VLM). Hệ thống tự sinh ý tưởng, triển khai thực nghiệm, tạo biểu đồ hình vẽ, viết bài báo và lặp lại dựa trên phản hồi của người bình duyệt.

Phán quyết từ bình duyệt đồng cấp: một bài báo do v2 tạo ra đã được chấp nhận tại một hội thảo ICLR 2025 (có công khai nguồn gốc). Phán quyết từ đánh giá độc lập: hệ thống còn lâu mới đạt được độ tin cậy. Cả hai nhận định đều đúng.

## Khái niệm

### Kiến trúc

1. **Sinh ý tưởng (Idea generation).** LLM đề xuất các ý tưởng nghiên cứu dựa trên một chủ đề và các tài liệu trước đó. Phiên bản v1 sử dụng bản mẫu; v2 sử dụng tìm kiếm tác tử trên không gian các giả thuyết.
2. **Kiểm tra tính mới (Novelty check).** Bước truy xuất tài liệu kiểm tra xem ý tưởng đã từng được công bố hay chưa. Đây là bước mà đánh giá của Beel et al. phát hiện việc dán nhãn sai — các phương pháp đã có từ lâu thường xuyên bị phân loại nhầm thành ý tưởng mới.
3. **Kế hoạch thực nghiệm (Experiment plan).** Tác tử phác thảo một giao thức thực nghiệm và viết mã nguồn.
4. **Thực thi (Execution).** Mã chạy bên trong hộp cát (sandbox). Các lỗi thất bại được đưa trở lại vòng lặp thử lại. Theo đo lường của Beel et al., 42% các thực nghiệm thất bại do lỗi lập trình ở giai đoạn này.
5. **Tạo biểu đồ (Figure generation).** Một mô hình thị giác-ngôn ngữ đọc các biểu đồ được tạo ra và viết lại mã vẽ để tăng độ trực quan rõ ràng. Đây là cải tiến kỹ thuật then chốt của v2.
6. **Viết bài báo (Writeup).** LLM phác thảo bài báo, lặp lại chỉnh sửa với một bộ bình duyệt nội bộ.
7. **Tùy chọn: nộp bài (Submission).** Bài báo được nộp tới một hội thảo hoặc hội nghị.

### Ý nghĩa của kết quả được chấp nhận tại hội thảo

Một bài báo do v2 tạo ra đã vượt qua khâu bình duyệt đồng cấp tại một hội thảo ICLR 2025. Các tác giả đã công khai nguồn gốc bài báo với ban tổ chức chương trình. Việc được chấp nhận là một dữ liệu thực nghiệm tham khảo; nó không phải là giấy phép để khẳng định hệ thống "biết làm nghiên cứu".

Bối cảnh quan trọng: các bài báo hội thảo chuyên đề (workshop) có tiêu chuẩn thẩm định thấp hơn so với các bài báo của hội nghị chính thức. Quá trình bình duyệt đồng cấp luôn có độ nhiễu nhất định; một tỷ lệ nhỏ các bài nộp ngẫu nhiên vẫn có thể được chấp nhận. Một thành công đơn lẻ là bằng chứng khái niệm (proof of concept), không phải là tuyên bố về độ tin cậy. Bài báo công bố trên Nature năm 2026 ghi lại toàn bộ vòng lặp đầu-cuối và được đồng tác giả bởi chính các nhà nghiên cứu con người; nó không đồng nghĩa với việc "hệ thống đã tự viết một bài báo Nature".

### Những gì đánh giá độc lập phát hiện

Beel et al. (arXiv:2502.14297) đã thực hiện một đánh giá độc lập từ bên ngoài. Các phát hiện nổi bật gồm:

- **Thất bại thực nghiệm.** 42% các thực nghiệm thất bại do lỗi lập trình (lỗi import, sai lệch kích thước tensor, biến chưa được định nghĩa). Vòng lặp thử lại chỉ bắt được một phần, không phải tất cả.
- **Dán nhãn sai tính mới.** Khâu truy xuất tài liệu thường xuyên gắn cờ các khái niệm đã có từ trước là phát kiến mới. Đây là hiện tượng tương đương với ảo giác trong bối cảnh nghiên cứu khoa học.
- **Khoảng cách chất lượng trình bày.** Khâu phê bình biểu đồ bằng mô hình thị giác-ngôn ngữ đã tạo ra các hình ảnh đạt chuẩn xuất bản, che đậy các điểm yếu thực nghiệm bên dưới.

Phát hiện cuối cùng là điểm then chốt nhất đối với phase học này. Một hệ thống tạo ra đầu ra có vẻ ngoài thuyết phục mà không thực hiện nghiên cứu thuyết phục sẽ nguy hiểm hơn, chứ không hề an toàn hơn một hệ thống thất bại rõ ràng. Việc đánh giá bắt buộc phải soi chiếu sâu vào các luận điểm cốt lõi, không thể dừng lại ở vẻ bóng bẩy của các biểu đồ.

### Mối lo ngại về việc thoát hộp cát (Sandbox escape)

Tài liệu README trong kho lưu trữ của Sakana đưa ra cảnh báo:

> Do bản chất của phần mềm này là thực thi mã do LLM sinh ra, chúng tôi không thể đảm bảo an toàn. Có những rủi ro về các gói phần mềm nguy hiểm, truy cập web không kiểm soát và việc sinh ra các tiến trình ngoài ý muốn. Hãy tự chịu rủi ro khi sử dụng và cân nhắc cô lập bằng Docker.

Đây là hình thái vận hành thực tế của quyền tự trị trong một lĩnh vực chưa được xác minh. LLM viết mã; mã được thực thi; mã có thể làm bất cứ điều gì mà tiến trình được cấp quyền. Nếu không có một hộp cát giới hạn nghiêm ngặt các hành động trên hệ thống tệp, mạng và tiến trình, bất kỳ tác tử nghiên cứu tự định hướng nào cũng có thể trích xuất rò rỉ dữ liệu, đốt sạch tài nguyên tính toán hoặc tự viết lại chính nó.

Câu chuyện hộp cát của AlphaEvolve đơn giản hơn vì bộ đánh giá của nó rất chặt chẽ. Vòng lặp của AI Scientist v2 chạy mã tùy ý với các mục tiêu mở. Đó là lý do tại sao nó cần sự cô lập mạnh mẽ hơn (tối thiểu là Docker; ưu tiên seccomp / gVisor) và sự rà soát thủ công của con người đối với mọi bài nộp trước khi nó rời khỏi hệ thống.

### Vị trí của v2 trong ngăn xếp công nghệ biên giới

| Hệ thống | Mục tiêu | Loại đầu ra | Bộ đánh giá | Lỗi đã biết |
|---|---|---|---|---|
| AlphaEvolve | thuật toán | mã nguồn | kiểm thử đơn vị + điểm chuẩn | bị giới hạn bởi độ nghiêm ngặt của bộ đánh giá |
| DGM | khung giàn tác tử | mã nguồn | SWE-bench | gian lận phần thưởng (reward hacking) |
| AI Scientist v2 | bài báo nghiên cứu | văn bản + mã + hình ảnh | bình duyệt đồng cấp (yếu) | lỗi thực nghiệm, dán nhãn sai, bóng bẩy che đậy điểm yếu |

v2 sở hữu bộ đánh giá tự động yếu nhất trong cả ba, bề mặt đầu ra rộng nhất và con đường ngắn nhất dẫn tới các sản phẩm công khai. Các biện pháp kiểm soát vận hành (hộp cát, rà soát, công khai minh bạch) đang phải gánh vác phần lớn nhiệm vụ an toàn.

```figure
mx-research-loop
```

## Sử dụng

`code/main.py` mô phỏng vòng lặp v2 dưới dạng một máy trạng thái (state machine): ý tưởng → kiểm tra tính mới → thực nghiệm → biểu đồ → bài viết → bình duyệt → chấp nhận-hoặc-lặp lại. Mỗi trạng thái có một xác suất lỗi có thể cấu hình được lấy từ các phát hiện của Beel et al. Chạy trình mô phỏng qua N vòng lặp và thống kê:

- Có bao nhiêu ý tưởng đến được giai đoạn nộp bài.
- Có bao nhiêu bài nộp chứa khiếm khuyết thực nghiệm nghiêm trọng bị che giấu bởi bài viết bóng bẩy.
- Cách ngân sách thử lại đánh đổi giữa chất lượng và sản lượng bài báo.

## Đóng gói

`outputs/skill-ai-scientist-sandbox-review.md` là danh mục kiểm tra rà soát hai cổng (two-gate review checklist) cho bất kỳ kết quả nào do tác tử nghiên cứu tạo ra trước khi rời khỏi hộp cát.

## Bài tập

1. Chạy `code/main.py` với các tham số mặc định. Tỷ lệ lượt chạy tạo ra một bài báo "sạch" là bao nhiêu? Tỷ lệ tạo ra một bài báo có khiếm khuyết thực nghiệm bị che lấp bởi khâu phê bình biểu đồ là bao nhiêu?

2. Các tham số mặc định đã sử dụng các con số 42% / 25% của Beel et al. Hãy chạy lại với cờ `--experiment-failure 0.20 --novelty-mislabel 0.10` và sau đó với cờ `--experiment-failure 0.60 --novelty-mislabel 0.40`. Tỷ lệ bài báo được trau chuốt nhưng có lỗi dịch chuyển như thế nào giữa hai lượt chạy?

3. Đọc tệp README của kho lưu trữ Sakana AI Scientist v2 về các yêu cầu hộp cát. Nêu hai biện pháp hạn chế bổ sung (ngoài Docker) mà bạn sẽ áp dụng cho một lượt chạy tự trị kéo dài nhiều ngày.

4. Đọc Mục 4 trong bài báo của Beel et al. về khoảng cách chất lượng trình bày. Thiết kế thêm một bộ đánh giá bổ sung có khả năng bắt được các bài báo trông đẹp mắt nhưng có khiếm khuyết về mặt thực nghiệm.

5. Đề xuất một giao thức rà soát có con người can thiệp cho các đầu ra của tác tử nghiên cứu có khả năng mở rộng tốt hơn phương án "một tiến sĩ đọc từng bài báo". Xác định điểm nghẽn và thiết kế quy trình giải quyết điểm nghẽn đó.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|---|---|---|
| AI Scientist v1 | "Tác tử nghiên cứu dùng bản mẫu của Sakana" | Điền các thực nghiệm vào một khung giàn cố định |
| AI Scientist v2 | "Tác tử nghiên cứu không cần bản mẫu" | Tìm kiếm cây tác tử kết hợp phê bình biểu đồ bằng VLM |
| Tìm kiếm cây tác tử (Agentic tree search) | "Tác tử nghiên cứu phân nhánh" | Mở rộng nhiều kế hoạch thực nghiệm song song; cắt tỉa bằng bộ phê bình nội bộ |
| Phê bình thị giác-ngôn ngữ (Vision-language critique) | "VLM trau chuốt biểu đồ" | Mô hình đa phương thức đọc biểu đồ và viết lại mã vẽ để tăng độ rõ ràng |
| Truy xuất tài liệu (Literature retrieval) | "Kiểm tra tính mới" | Tìm kiếm các công trình trước đó để xác nhận tính mới của ý tưởng — đã được ghi nhận hay dán nhãn sai |
| Trau chuốt che đậy (Polish masking) | "Bài báo đẹp, nghiên cứu hỏng" | Chất lượng trình bày vượt xa chất lượng thực nghiệm; che giấu các điểm yếu cốt lõi |
| Thoát hộp cát (Sandbox escape) | "Mã LLM phá rào thoát ra" | Mã do tác tử thực thi làm những việc mà người thiết kế vòng lặp không dự tính |

## Đọc thêm

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — bài báo khoa học.
- [Blog của Sakana về ấn phẩm Nature 2026](https://sakana.ai/ai-scientist-nature/) — tóm tắt kèm bối cảnh bình duyệt đồng cấp.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — các con số đánh giá độc lập từ bên ngoài.
- [Bài báo AI Scientist v1 của Sakana](https://arxiv.org/abs/2408.06292) — phiên bản tiền nhiệm dựa trên bản mẫu.
- [Anthropic — Đo lường quyền tự trị của tác tử AI (Measuring AI agent autonomy)](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn rộng hơn về các tác tử nghiên cứu mở.
