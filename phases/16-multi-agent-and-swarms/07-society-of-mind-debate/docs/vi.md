# Xã hội Tâm trí và Tranh luận Đa tác tử (Society of Mind and Multi-Agent Debate)

> Giả thuyết năm 1986 của Minsky — trí tuệ là một xã hội gồm các chuyên gia chuyên biệt — được tái khám phá sau mỗi thập kỷ. Năm 2023, Du và cộng sự đã biến nó thành một thuật toán cụ thể: nhiều thực thể (instance) LLM đưa ra câu trả lời đề xuất, đọc câu trả lời của nhau, phê bình và cập nhật. Qua N vòng, chúng hội tụ về một sự đồng thuận vượt trội hơn chuỗi suy nghĩ không mẫu (zero-shot Chain of Thought - CoT) và phản tư (reflection) trên sáu tác vụ suy luận và kiểm chứng dữ kiện thực tế. Hai phát hiện then chốt: cả **nhiều tác tử** lẫn **nhiều vòng** đều đóng góp độc lập. Xã hội tác tử đánh bại màn độc thoại đơn tác tử; quá trình trao đổi nhiều vòng đánh bại cơ chế bỏ phiếu một lần (one-shot voting).

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Vấn đề

Tính tự nhất quán (self-consistency) — lấy mẫu từ một mô hình duy nhất nhiều lần và chọn câu trả lời theo đa số — là phương pháp cải thiện khả năng suy luận ít tốn kém nhất mà bạn có thể áp dụng ngay. Phương pháp này hiệu quả, nhưng nhanh chóng chạm ngưỡng bão hòa. Bạn có thể tăng gấp đôi số mẫu mà không thấy thêm bước nhảy vọt nào đáng kể.

Tranh luận đa tác tử (multi-agent debate) phá vỡ ngưỡng bão hòa đó. Thay vì lấy N mẫu độc lập từ cùng một mô hình, N tác tử đọc chuỗi suy luận của nhau và tự hiệu chỉnh. Tương quan sai số giữa các mẫu giảm xuống (chúng không còn độc lập và cùng phân phối - i.i.d.), và điểm hội tụ thường chính xác ngay cả ở những câu hỏi mà việc bỏ phiếu i.i.d. sai lệch một cách đầy tự tin.

## Khái niệm

### Thuật toán của Du và cộng sự (2023)

Từ công trình arXiv:2305.14325 (ICML 2024):

1. Mỗi tác tử trong số N tác tử tạo ra một câu trả lời ban đầu cho câu hỏi.
2. Với mỗi vòng r = 2..R: mỗi tác tử được hiển thị câu trả lời ở vòng r-1 của các tác tử khác và nhận yêu cầu "xem xét các câu trả lời này, hãy đưa ra câu trả lời cập nhật của bạn".
3. Sau R vòng, tiến hành bỏ phiếu theo đa số (majority vote) trên các câu trả lời cuối cùng.

Nghiên cứu thử nghiệm trên các bộ dữ liệu MMLU, GSM8K, tiểu sử, MATH và các chuẩn đánh giá tính xác thực dữ kiện (factuality benchmarks). Tranh luận liên tục vượt trội so với CoT và tự phản tư (self-reflection).

### Hai nút điều chỉnh độc lập

Các thử nghiệm phân rã thành phần (ablations) từ cùng công trình:

- **Chỉ tăng số lượng tác tử** (1 vòng, bỏ phiếu đa số từ N tác tử) đánh bại đơn tác tử trên hầu hết các tác vụ, nhưng nhanh chóng đi vào bình nguyên (plateau).
- **Chỉ tăng số vòng** (1 tác tử tự xem lại suy luận trước đó của chính mình) hầu như không đem lại lợi ích — đây là điểm yếu đã biết của phương pháp phản tư.
- **Kết hợp cả hai** tạo ra những bước nhảy vọt lớn. Quá trình trao đổi qua lại nhiều vòng giữa nhiều tác tử chính là động lực thúc đẩy sự cải thiện vượt bậc.

### Tại sao phương pháp này hiệu quả?

Hai cơ chế cốt lõi:

1. **Tiếp xúc với sự bất đồng (exposure to disagreement).** Khi một tác tử quan sát chuỗi suy luận của một tác tử khác với kết luận khác biệt, nó buộc phải bảo vệ lập luận của mình hoặc cập nhật lại. Dù theo hướng nào, ngữ cảnh cho vòng r+1 cũng phong phú hơn vòng r.
2. **Giảm thiểu sai số tương quan (correlated error reduction).** Trong cơ chế tự nhất quán, tất cả các mẫu đều xuất phát từ cùng một mô hình, do đó các sai số có tương quan mật thiết — dẫn đến việc bạn lấy trung bình và chọn phải một câu trả lời sai đầy tự tin. Các mô hình khác nhau hoặc các seed ngẫu nhiên khác nhau giúp triệt tiêu tương quan (decorrelate). Các *quan điểm được tranh luận* khác nhau giúp triệt tiêu tương quan sâu sắc hơn nữa.

### Tranh luận dị thể (Heterogeneous debate)

A-HMAD và các nghiên cứu tiếp nối sử dụng *các mô hình nền tảng khác nhau* (different base models) cho các tác tử khác nhau. Việc để Llama + Claude + GPT cùng tranh luận giúp giảm thiểu nguy cơ suy sụp do độc canh (monoculture collapse - Bài học 26) vì các sai số tương quan của một dòng mô hình không bị lặp lại ở các dòng mô hình khác.

Điểm bất lợi: một mô hình yếu kém tham gia vào cuộc tranh luận có thể kéo sự đồng thuận chung về phía câu trả lời sai của nó (tham khảo "Should we be going MAD?", arXiv:2311.17371).

### NLSOM — Mở rộng quy mô lên 129 tác tử

Zhuge và cộng sự ("Mindstorms in Natural Language-Based Societies of Mind", arXiv:2305.17066) đã mở rộng ý tưởng này lên các xã hội gồm 129 thành viên. Kết quả: sự chuyên môn hóa và khả năng tự tổ chức xuất hiện khi mở rộng quy mô, và hệ thống vượt trội hơn đơn tác tử trên các tác vụ như trả lời câu hỏi trực quan (visual question answering).

### Các dạng thức thất bại (Failure modes)

- **Thác nịnh bợ (Sycophancy cascade).** Tất cả các tác tử đều nhượng bộ tác tử nghe có vẻ tự tin nhất. Cuộc tranh luận sụp đổ và chịu sự chi phối của tiếng nói lớn nhất. Giải pháp: điều phối prompt cho các vai trò đối nghịch ("một tác tử bắt buộc phải đưa ra lập luận phản biện").
- **Lệch chủ đề (Topic drift).** Các cuộc tranh luận kéo dài qua nhiều vòng có xu hướng trôi xa khỏi câu hỏi ban đầu. Giải pháp giảm thiểu: đưa lại câu hỏi gốc vào ngữ cảnh ở mỗi vòng.
- **Bùng nổ chi phí tính toán (Compute blowup).** N tác tử × R vòng = N·R lượt gọi LLM, mỗi lượt có ngữ cảnh ngày càng phình to. Một cuộc tranh luận 5 tác tử, 5 vòng tiêu tốn 25 lượt gọi với cửa sổ ngữ cảnh mở rộng liên tục. Chi phí cho mỗi câu hỏi có thể vượt quá 10 lần một lượt gọi CoT đơn lẻ.

```figure
multi-agent-debate
```

## Xây dựng thực tế

`code/main.py` thực thi một phiên tranh luận 3 tác tử × 3 vòng trên một bài toán toán học, trong đó mỗi tác tử bắt đầu với một câu trả lời khác nhau (có thể sai). Các tác tử được lập trình sẵn theo kịch bản — mỗi tác tử "cập nhật" bằng cách lấy trung bình câu trả lời của các tác tử lân cận được gán trọng số theo độ tự tin định sẵn. Quá trình hội tụ được ghi nhận rõ ràng qua nhật ký từng vòng.

Bản minh họa làm nổi bật hai hiệu ứng then chốt:

- Một vòng trao đổi duy nhất đã đưa các tác tử tiến gần hơn đáng kể đến câu trả lời chính xác.
- Các vòng bổ sung sau vòng thứ 2 cho thấy hiệu suất giảm dần (diminishing returns - khớp với hiện tượng bình nguyên trong nghiên cứu của Du và cộng sự).

Chạy chương trình:

```bash
python3 code/main.py
```

## Ứng dụng thực tế

`outputs/skill-debate-configurator.md` cấu hình một phiên tranh luận cho một tác vụ mới: số lượng tác tử, số vòng, tính dị thể (dùng chung một mô hình hay kết hợp nhiều mô hình), phân bổ vai trò (đối xứng hay có một tác tử đối kháng). Tệp này cũng ước tính chi phí token trước khi vận hành.

## Đưa vào sản xuất

Nếu bạn triển khai tranh luận trong môi trường thực tế (production):

- **Giới hạn số vòng tối đa là 3.** Du và cộng sự đã chứng minh rằng 3 vòng thu nhận được hầu hết lợi ích. Vượt quá số vòng này chỉ làm tăng chi phí mà không cải thiện chất lượng.
- **Giới hạn số tác tử tối đa là 5.** Vượt quá 5, hiện tượng phình to ngữ cảnh và chi phí tính toán sẽ lấn át hoàn toàn lợi ích mang lại.
- **Mặc định sử dụng mô hình dị thể.** Đảm bảo có ít nhất hai dòng mô hình nền tảng khác nhau trong tập hợp tác tử.
- **Bố trí vị trí đối nghịch (adversarial slot).** Một tác tử được chỉ định luôn phản biện bất kể tình huống nào. Điều này phá vỡ hiện tượng thác nịnh bợ.
- **Ghi nhật ký chi tiết từng vòng.** Các hệ thống tranh luận ẩn đi các vòng trung gian sẽ không thể gỡ lỗi (debug) hay kiểm toán (audit).

## Bài tập

1. Chạy `code/main.py`, sau đó đặt số vòng là 5 và quan sát hiệu suất giảm dần. Tại vòng nào thì khả năng hội tụ bổ sung chấm dứt?
2. Thêm một tác tử thứ tư với vai trò đối nghịch: luôn bất đồng với ý kiến đa số hiện tại. Việc này phá vỡ hay cải thiện khả năng hội tụ?
3. Vẽ biểu đồ (hoặc in ra màn hình) điểm số đồng thuận qua từng vòng (tỷ lệ tác tử chọn câu trả lời theo đa số). Khi nào điểm số đạt 1.0 và con số đó có đồng nghĩa với "chính xác" hay không?
4. Đọc Mục 4 về các thử nghiệm phân rã của Du và cộng sự. Tái hiện kết quả "chỉ tác tử" so với "chỉ số vòng" so với "kết hợp cả hai" bằng mã nguồn này.
5. Đọc bài báo "Should we be going MAD?" (arXiv:2311.17371) và liệt kê hai biến thể tranh luận ngoài cơ chế vòng tròn (round-robin) — ví dụ: có giám khảo điều phối (judge-led), chuỗi tranh luận (chain-of-debate), hoặc tranh luận đối nghịch.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Bản chất kỹ thuật |
|---|---|---|
| Xã hội Tâm trí (Society of Mind) | "Ý tưởng của Minsky" | Trí tuệ hình thành từ các chuyên gia tương tác với nhau; khung khái niệm 1986 nay được hiện thực hóa qua tranh luận LLM. |
| Tranh luận đa tác tử (Multi-agent debate) | "Các tác tử tranh cãi" | N tác tử đề xuất, phản biện lẫn nhau, hiệu chỉnh qua R vòng, rồi bỏ phiếu theo đa số. |
| Đồng thuận (Consensus) | "Chúng đồng ý với nhau" | Không phải chân lý nhận thức luận — chỉ là tỷ lệ tác tử chọn câu trả lời theo đa số. Vẫn có thể sai lệch một cách đầy tự tin. |
| Vòng tranh luận (Rounds) | "Các bước trao đổi" | Một vòng = mỗi tác tử đọc câu trả lời của các tác tử khác và cập nhật một lần. |
| Tranh luận dị thể (Heterogeneous debate) | "Trộn các dòng mô hình" | Sử dụng các mô hình nền tảng khác nhau để triệt tiêu tương quan sai số. |
| Thác nịnh bợ (Sycophancy cascade) | "Tất cả đồng ý với bên nói to nhất" | Lỗi suy thoái trong tranh luận khi các tác tử nhượng bộ tác tử tự tin nhất mà không màng tới tính đúng đắn. |
| NLSOM | "Xã hội 129 tác tử" | Xã hội tâm trí dựa trên ngôn ngữ tự nhiên (Natural-language society of mind); phiên bản mở rộng quy mô của Zhuge và cộng sự. |
| Sai số tương quan (Correlated error) | "Cùng mô hình, cùng lỗi" | Lý do khiến cơ chế tự nhất quán bão hòa; tranh luận qua các quan điểm khác nhau giúp triệt tiêu tương quan. |

## Tài liệu đọc thêm

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — bài báo gốc tham chiếu, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — NLSOM 129 tác tử
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — đánh giá các biến thể tranh luận
- [Trang dự án Debate](https://composable-models.github.io/llm_debate/) — mã nguồn, bản minh họa và chi tiết phân rã thành phần của Du và cộng sự
