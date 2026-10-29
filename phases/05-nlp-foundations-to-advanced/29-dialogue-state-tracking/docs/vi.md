# Theo dõi trạng thái hội thoại (Dialogue State Tracking)

> "Tôi muốn tìm nhà hàng giá rẻ ở phía bắc... à thực ra đổi thành giá vừa phải nhé... và đồ ăn Ý." Ba lượt đối thoại, ba lần cập nhật trạng thái liên tiếp. DST giữ cho từ điển khe-giá trị (slot-value) luôn đồng bộ chính xác để hệ thống đặt chỗ thực thi chuẩn xác.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 17 (Hệ thống Chatbot), Giai đoạn 5 · 20 (Đầu ra cấu trúc)
**Time:** ~75 phút

## Vấn đề

Trong một hệ thống hội thoại hướng tác vụ (task-oriented dialogue system), mục tiêu của người dùng được mã hóa dưới dạng một tập hợp các cặp khe-giá trị (slot-value pairs): `{cuisine: italian, area: north, price: moderate}`. Mỗi lượt nói của người dùng có thể thêm mới, sửa đổi hoặc xóa bỏ một khe. Hệ thống bắt buộc phải đọc hiểu toàn bộ lịch sử cuộc trò chuyện và xuất ra trạng thái hiện tại một cách tuyệt đối chính xác.

Chỉ cần gán sai một khe duy nhất, hệ thống sẽ đặt nhầm nhà hàng, lên lịch sai chuyến bay, hoặc trừ tiền sai thẻ tín dụng. Kỹ thuật theo dõi trạng thái hội thoại (Dialogue State Tracking - DST) chính là bản lề khớp nối giữa những gì người dùng vừa thốt ra và những gì hệ thống backend sẽ thực thi.

Lý do DST vẫn đóng vai trò sống còn trong năm 2026 bất chấp sự bùng nổ của LLM:

- Các nghiệp vụ đòi hỏi tuân thủ nghiêm ngặt (ngân hàng, y tế, đặt vé máy bay) bắt buộc phải trích xuất các giá trị khe mang tính tất định (deterministic), không chấp nhận văn bản sinh tự do ngẫu hứng.
- Các AI Agent sử dụng công cụ (Tool-use agents) bắt buộc phải giải mã chuẩn xác các tham số khe trước khi thực hiện gọi hàm API.
- Việc xử lý các câu đính chính đa lượt thoại phức tạp hơn nhiều so với bề ngoài: "à không, đổi thành thứ Năm nhé."

Quy trình kỹ thuật hiện đại: Kết hợp các nguyên lý DST kinh điển + Mô hình trích xuất LLM + Rào chắn đầu ra có cấu trúc (structured-output guardrails).

## Khái niệm

![DST: lịch sử hội thoại → trạng thái khe-giá trị](../assets/dst.svg)

**Cấu trúc tác vụ.** Một lược đồ (schema) định nghĩa các miền nghiệp vụ (domain: nhà hàng, khách sạn, taxi) và các khe tương ứng (slots: loại ẩm thực, khu vực, mức giá, số người). Mỗi khe có thể rỗng, nhận giá trị từ một tập đóng cố định (price: {cheap, moderate, expensive}), hoặc nhận giá trị tự do (name: "The Copper Kettle").

**Hai phương thức định hình bài toán DST.**

- **Phân loại (Classification):** Với mỗi cặp (khe, giá trị ứng viên), mô hình dự đoán có/không. Rất hiệu quả cho các khe có tập từ vựng đóng. Tiêu chuẩn chủ đạo trước năm 2020.
- **Mô hình sinh (Generation):** Dựa vào lịch sử hội thoại, mô hình sinh trực tiếp giá trị của khe dưới dạng văn bản tự do. Hoạt động trơn tru với các khe từ vựng mở. Đây là tiêu chuẩn hiện đại mặc định.

**Chỉ số đánh giá.** Độ chính xác mục tiêu chung (Joint Goal Accuracy - JGA) — tỷ lệ các lượt thoại mà *mọi* khe đều được dự đoán hoàn toàn chính xác. Đây là chỉ số "được ăn cả ngã về không" (all-or-nothing). Điểm số cao nhất trên bảng xếp hạng MultiWOZ 2.4 đầu năm 2026 đạt khoảng ~83%.

**Các kiến trúc thuật toán.**

1. **Dựa trên tập luật (Regex trích xuất khe + từ khóa):** Baseline cực kỳ mạnh cho các miền nghiệp vụ hẹp. Dễ dàng gỡ lỗi và kiểm toán.
2. **TripPy / BERT-DST:** Cơ chế sinh dựa trên thao tác sao chép (copy-based) kết hợp bộ mã hóa BERT. Tiêu chuẩn vàng thời kỳ tiền LLM.
3. **LDST (LLaMA + LoRA):** LLM được tinh chỉnh theo chỉ dẫn kết hợp prompt định hướng miền và khe. Đạt chất lượng tương đương ChatGPT trên MultiWOZ 2.4.
4. **DST không cần bản thể luận (Ontology-free 2024–2026):** Bỏ qua lược đồ cố định; mô hình tự động sinh trực tiếp tên khe và giá trị tương ứng. Xử lý tốt các miền mở.
5. **Prompt kết hợp đầu ra cấu trúc (2024–2026):** LLM kết hợp với Pydantic schema + giải mã có ràng buộc. Chỉ với 5 dòng mã nguồn, sẵn sàng triển khai môi trường sản xuất.

### Các dạng lỗi kinh điển

- **Đồng tham chiếu xuyên lượt thoại:** "Hãy chọn phương án đầu tiên đi." Cần giải quyết phương án đầu tiên cụ thể là phương án nào.
- **Ghi đè so với Bổ sung (Overwrite vs append):** Người dùng nói "thêm món Ý." Hệ thống nên thay thế hoàn toàn loại ẩm thực hay bổ sung thêm?
- **Xác nhận ngầm (Implicit confirmations):** "Ừ được đấy" — câu nói này đã đồng ý đặt chỗ chưa?
- **Đính chính thông tin:** "À thực ra đổi thành 7 giờ tối." Cần cập nhật trường thời gian mà không được xóa các khe khác đã điền.
- **Đồng tham chiếu tới phát ngôn trước của hệ thống:** "Đúng, cái đó đấy." "Cái đó" là cái nào trong câu bot vừa liệt kê?

```figure
n5-slot-tracker
```

## Thực hành xây dựng

### Bước 1: Bộ trích xuất khe dựa trên luật

Xem chi tiết tại `code/main.py`. Biểu thức chính quy (Regex) kết hợp từ điển đồng nghĩa bao phủ được 70% các câu thoại chuẩn tắc trong các miền hẹp:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Dễ gãy nếu người dùng dùng từ vựng ngoài danh mục. Tuy nhiên, hoạt động hoàn hảo cho các bước xác nhận khe tất định.

### Bước 2: Vòng lặp cập nhật trạng thái

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Ba bất biến bắt buộc phải tuân thủ:

- Tuyệt đối không xóa một khe mà người dùng không hề nhắc tới.
- Lời phủ định tường minh ("thôi không cần chọn món nữa") bắt buộc phải xóa giá trị khe về `None`.
- Lời đính chính của người dùng ("thực ra là...") bắt buộc phải ghi đè giá trị cũ, không được nối đuôi.

### Bước 3: DST điều khiển bởi LLM với đầu ra cấu trúc

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Sự kết hợp giữa `instructor` và `Pydantic` đảm bảo trả về một đối tượng trạng thái hợp lệ 100%. Không lo lỗi regex, không lo lệch schema, không lo sinh ra các khe bị ảo giác.

### Bước 4: Đánh giá bằng Joint Goal Accuracy (JGA)

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Chuẩn đánh giá: Tỷ lệ phần trăm các lượt thoại mà hệ thống đoán ĐÚNG TẤT CẢ các khe là bao nhiêu? Trên MultiWOZ 2.4, các hệ thống hàng đầu năm 2026 đạt 80-83%. Hệ thống nội bộ của bạn trên miền hẹp phải vượt qua con số này, nếu không giải pháp dùng LLM đơn thuần sẽ đánh bại bạn.

### Bước 5: Xử lý đính chính thông tin

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Khi phát hiện có dấu hiệu đính chính, hãy ghi đè khe vừa cập nhật thay vì cộng dồn. Rất khó xử lý hoàn hảo bằng luật nếu không có LLM hỗ trợ. Mô hình chuẩn năm 2026: Luôn cho LLM sinh lại toàn bộ trạng thái hoàn chỉnh từ lịch sử thay vì cập nhật từng phần (incremental) — cơ chế này tự động xử lý các câu đính chính một cách mượt mà.

## Cạm bẫy

- **Chi phí sinh lại toàn bộ lịch sử.** Việc để LLM tái sinh lại toàn bộ trạng thái ở mỗi lượt thoại tốn chi phí $O(n^2)$ token theo thời gian. Cần giới hạn độ dài lịch sử hoặc tóm tắt các lượt thoại cũ.
- **Trôi dạt lược đồ (Schema drift).** Việc bổ sung các khe mới sau này sẽ làm mất tính tương thích của dữ liệu huấn luyện cũ. Hãy luôn đánh số phiên bản cho schema.
- **Phân biệt hoa thường (Case sensitivity).** "Italian" so với "italian" hay "ITALIAN" — bắt buộc phải chuẩn hóa về một dạng duy nhất ở mọi nơi.
- **Kế thừa ngầm định (Implicit inheritance).** Nếu người dùng đã nói "bàn cho 4 người", thì lượt thoại sau khi họ đổi giờ ăn không được phép xóa số lượng người. Luôn truyền đầy đủ ngữ cảnh lịch sử.
- **Khe tự do so với Tập đóng.** Tên người, thời gian và địa chỉ cần khe tự do; loại món ăn và khu vực thuộc tập đóng. Cần kết hợp linh hoạt cả hai loại trong schema.

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống | Phương pháp khuyến nghị |
|------------|-------------------------|
| Miền nghiệp vụ rất hẹp (1-2 ý định) | Dựa trên tập luật + regex |
| Miền rộng, có sẵn tập dữ liệu gán nhãn | LDST (LLaMA + LoRA trên dữ liệu dạng MultiWOZ) |
| Miền rộng, không có nhãn, cần lên production ngay | LLM + Instructor + Pydantic schema |
| Hệ thống thoại / Giọng nói | ASR + bộ chuẩn hóa văn bản + LLM-DST |
| Luồng đặt chỗ đa miền phức tạp | LLM dẫn dắt bởi schema với các model Pydantic cho từng miền |
| Yêu cầu tuân thủ khắt khe | Dựa trên luật làm bộ lọc chính, LLM dự phòng kèm luồng người dùng xác nhận lại |

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Bài tập

1. **Dễ.** Xây dựng bộ theo dõi trạng thái dựa trên luật trong `code/main.py` cho 3 khe (ẩm thực, khu vực, mức giá). Thử nghiệm trên 10 cuộc hội thoại tự biên soạn. Đo lường chỉ số JGA.
2. **Trung bình.** Chạy lại chính tập dữ liệu đó với Instructor + Pydantic + một LLM nhỏ gọn. So sánh điểm JGA. Khảo sát kỹ các lượt thoại khó nhất.
3. **Khó.** Kết hợp cả hai phương pháp và thiết lập bộ định tuyến: ưu tiên bộ trích xuất dựa trên luật, chỉ kích hoạt LLM khi bộ luật trích xuất dưới 2 khe với độ tin cậy thấp. Đo lường chỉ số JGA tổng thể và chi phí suy luận trung bình trên mỗi lượt thoại.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| DST | Theo dõi trạng thái hội thoại | Duy trì và đồng bộ từ điển khe-giá trị xuyên suốt các lượt đối thoại. |
| Khe (Slot) | Đơn vị ý định của người dùng | Tham số có tên mà hệ thống backend yêu cầu (loại món, ngày giờ). |
| Miền (Domain) | Lĩnh vực tác vụ | Nhà hàng, khách sạn, taxi — nhóm các khe liên quan. |
| JGA | Độ chính xác mục tiêu chung | Tỷ lệ các lượt thoại mà mọi khe đều được điền chính xác tuyệt đối. |
| MultiWOZ | Bộ đo chuẩn quốc tế | Tập dữ liệu hội thoại Wizard-of-Oz đa miền; chuẩn mực đánh giá DST. |
| DST không bản thể luận (Ontology-free) | Không cần schema cố định | Tự do sinh tên khe và giá trị trực tiếp từ ngữ cảnh. |
| Đính chính (Correction) | "À thực ra..." | Lượt thoại đính chính hoặc ghi đè một giá trị khe đã điền trước đó. |

## Đọc thêm

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — bài báo công bố tập dữ liệu chuẩn mực MultiWOZ.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — tinh chỉnh chỉ dẫn LLaMA + LoRA cho bài toán DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — kiến trúc sao chép ba tầng cho DST.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — hội thoại hướng tác vụ học không giám sát.
- [Bảng xếp hạng MultiWOZ](https://github.com/budzianowski/multiwoz) — bảng xếp hạng kết quả DST chính thức.

