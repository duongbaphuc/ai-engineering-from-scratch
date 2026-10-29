# Chatbot — Từ Luật Cố định đến Mạng Nơ-ron và Tác tử LLM (Chatbots — Rule-Based to Neural to LLM Agents)

> ELIZA phản hồi bằng so khớp mẫu hình. DialogFlow ánh xạ ý định người dùng. GPT trả lời từ trọng số tham số. Claude kích hoạt công cụ và kiểm thực kết quả. Mỗi kỷ nguyên ra đời để giải quyết thất bại tồi tệ nhất của kỷ nguyên phía trước.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Hệ thống Hỏi Đáp), Phase 5 · 14 (Truy xuất Thông tin)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Người dùng nói: "Tôi muốn đổi chuyến bay của mình." Hệ thống phải xác định họ muốn gì, thông tin nào còn thiếu, làm thế nào để thu thập thông tin đó và làm thế nào để hoàn tất hành động đổi vé. Sau đó người dùng đột ngột đổi ý: "khoan đã, nếu tôi hủy vé luôn thì sao?" và hệ thống phải nhớ lại toàn bộ ngữ cảnh, chuyển đổi tác vụ nhịp nhàng mà vẫn bảo toàn trạng thái phiên làm việc.

Hội thoại là bài toán cực kỳ khó đối với một hệ thống máy học (ML). Đầu vào hoàn toàn mở. Đầu ra phải mạch lạc qua hàng chục lượt tương tác (multi-turn). Hệ thống thường phải tác động trực tiếp vào thế giới thực (đổi vé máy bay, trừ tiền thẻ tín dụng). Bất kỳ bước đi sai lầm nào cũng hiển thị rõ mồn một trước mắt người dùng.

Các kiến trúc chatbot đã trải qua bốn mô hình chuyển đổi lớn (paradigms), mỗi mô hình ra đời vì mô hình trước đó bộc lộ những điểm gãy đổ quá rõ ràng. Bài học này điểm lại từng mô hình theo trình tự thời gian. Bức tranh sản xuất năm 2026 là sự kết hợp lai giữa hai mô hình cuối cùng.

## Khái niệm (The Concept)

![Tiến hóa của Chatbot: dựa trên luật → truy xuất → mạng nơ-ron → tác tử LLM](../assets/chatbot.svg)

### Nửa thế kỷ kịch bản luật cố định (1950 - 2001)

Mô hình đầu tiên không chỉ kéo dài năm năm. Nó đã tồn tại suốt năm mươi năm. Hiểu được lịch sử này rất quan trọng vì mọi hệ thống trong giai đoạn đó đều chung một cỗ máy cốt lõi — khớp đầu vào, phát ra câu trả lời đóng hộp có sẵn, cập nhật một chút biến trạng thái — và năm mươi năm liên tục bổ sung luật vào cỗ máy đó không bao giờ tạo ra được trí thông minh tổng quát. Ngưỡng trần đó chính là lý do các mô hình từ thứ hai đến thứ tư ra đời.

**1950.** Alan Turing gạt bỏ câu hỏi triết học "máy móc có thể suy nghĩ không?" bằng một phép thử thực nghiệm thay thế: Nếu người thẩm vấn không thể phân biệt được cỗ máy với con người qua đường truyền điện báo teletype, câu hỏi triết học trở nên vô nghĩa. Hội thoại trở thành bài đo chuẩn mực (benchmark) của lĩnh vực này từ trước khi lĩnh vực này có một cái tên chính thức.

**1956.** Tên gọi chính thức xuất hiện — hội thảo mùa hè tại Dartmouth đặt ra thuật ngữ "trí tuệ nhân tạo" (artificial intelligence) dựa trên phỏng đoán rằng mọi đặc trưng của trí thông minh "về nguyên tắc đều có thể được mô tả chính xác đến mức một cỗ máy có thể được chế tạo để mô phỏng nó." Đề xuất này tự tin dự trù thời gian hai tháng để đạt được bước tiến đáng kể.

**1966.** ELIZA phát hành mẹo phản chiếu (reflection trick) mà bạn sẽ tự tay xây dựng trong Bước 1: các luật phân rã (decomposition rules) trích xuất các mảnh câu từ đầu vào, các luật tái ghép nối (reassembly rules) biến chúng thành câu hỏi phản hồi lại. Chỉ khoảng 200 mẫu hình tổng cộng, không có trạng thái bên trong, không có chút thấu hiểu ngữ nghĩa nào — vậy mà người dùng vẫn trút bầu tâm sự với nó như một bác sĩ tâm lý thực thụ. Tác giả Joseph Weizenbaum đã dành phần đời còn lại trong sự bàng hoàng về việc người ta có thể dễ dàng bị đánh lừa bởi một cỗ máy thô sơ đến mức nào.

**1972.** PARRY, được phát triển tại Đại học Stanford để mô phỏng chứng hoang tưởng (paranoia), bổ sung mảnh ghép mà ELIZA còn thiếu: trạng thái nội tại (internal state). Các biến số đại diện cho mức độ sợ hãi, giận dữ và ngờ vực được cập nhật sau mỗi lượt đối thoại và quyết định kịch bản nào sẽ được kích hoạt tiếp theo, khiến cho cùng một câu đầu vào sẽ nhận được những phản hồi rất khác nhau tùy theo diễn biến cuộc trò chuyện trước đó. Trong bài kiểm thử mù hồ sơ đối thoại, các bác sĩ tâm thần chỉ phân biệt được PARRY với bệnh nhân thật ở mức ngẫu nhiên (50%). Đây chính là tổ tiên trực tiếp của kỹ thuật định hình nhân cách (persona conditioning) — một system prompt thời kỳ đầu được cài đặt bằng ba biến số thực (floats). Cùng năm đó, hai con bot được kết nối đối thoại trực tiếp với nhau qua mạng ARPANET: một kịch bản bác sĩ tâm lý phỏng vấn một cỗ máy trạng thái hoang tưởng, cuộc trò chuyện bot-với-bot đầu tiên trong lịch sử mạng máy tính.

**1995.** ALICE mở rộng công thức của ELIZA bằng AIML, một ngôn ngữ XML dành cho các cặp mẫu hình - khuôn mẫu (pattern-template pairs). Khoảng 40.000 danh mục viết tay, ba lần giành giải Loebner Prize. Nó đã chứng minh quy luật tỷ lệ khắc nghiệt của các hệ thống dựa trên luật: càng nhiều luật chỉ mang lại độ bao phủ, không bao giờ đem lại tính tổng quát hóa. Mỗi luật bổ sung là một gánh nặng kỹ thuật mà con người phải bảo trì.

**2001.** SmarterChild đưa công thức này tiếp cận 30 triệu người dùng ứng dụng nhắn tin nhanh (AOL, MSN Messenger) và bổ sung việc tra cứu dữ liệu backend — thời tiết, chứng khoán, lịch chiếu phim — được ghép trực tiếp vào các mẫu câu. Nhìn kỹ, đó chính là cơ chế gọi công cụ (tool calling) khoác bộ trang phục của năm 2001: phân tích ý định, gọi một dịch vụ bên ngoài, và kết xuất kết quả vào câu trả lời.

Năm mươi năm, một cơ chế duy nhất, số lượng luật ngày một phình to. Mô hình này kết thúc không phải vì ai đó chứng minh nó sai, mà vì chi phí bảo trì của các máy trạng thái viết tay tăng tuyến tính theo độ bao phủ, trong khi kỳ vọng của người dùng thì tăng vọt theo bất kỳ điều kỳ diệu nào họ vừa thấy tuần trước.

```figure
chatbot-lineage
```

**Dựa trên luật (ELIZA, AIML, DialogFlow).** Các mẫu hình được viết tay để so khớp đầu vào của người dùng và sinh ra phản hồi. Bộ phân loại ý định (intent classifiers) điều hướng sang các luồng được định nghĩa trước. Cỗ máy trạng thái điền thông tin vào ô trống (slot-filling state machines) thu thập các thông tin bắt buộc còn thiếu. Hoạt động xuất sắc trong phạm vi hẹp được thiết kế, nhưng gãy đổ ngay lập tức khi ra ngoài phạm vi đó. Vẫn tiếp tục được triển khai trong các miền tối quan trọng về an toàn (xác thực ngân hàng, đặt vé máy bay) nơi hiện tượng ảo giác là điều tuyệt đối không được phép xảy ra.

**Dựa trên truy xuất (Retrieval-based).** Hệ thống dạng hỏi đáp FAQ: Mã hóa mọi cặp (câu hỏi, câu trả lời). Khi chạy, mã hóa câu nói của người dùng và truy xuất câu trả lời lưu sẵn gần nhất trong không gian vector. Giống như tính năng "các bài viết tương tự" của Zendesk. Xử lý diễn đạt tương đương tốt hơn luật viết tay. Không tự tạo sinh văn bản nên loại bỏ hoàn toàn nguy cơ ảo giác.

**Mạng nơ-ron tạo sinh (Neural seq2seq).** Mô hình encoder-decoder được huấn luyện trên nhật ký hội thoại. Sinh câu trả lời hoàn toàn mới từ đầu. Câu chữ rất trôi chảy nhưng dễ đưa ra các phản hồi chung chung vô thưởng vô phạt ("Tôi không biết") và trôi dạt dữ kiện. Không giữ vững được chủ đề hội thoại một cách ổn định. Đây là nguyên nhân khiến toàn bộ chatbot của Google, Facebook và Microsoft trong giai đoạn 2016-2019 đều gây thất vọng lớn.

**Tác tử LLM (LLM Agents).** Một mô hình ngôn ngữ lớn được đặt trong một vòng lặp có khả năng lập kế hoạch (plan), kích hoạt công cụ (call tools), và kiểm thực kết quả (verify). Không phải là một chatbot với một prompt dài dằng dặc, mà là một vòng lặp tác tử thực thụ: lập kế hoạch → gọi công cụ → quan sát kết quả → quyết định bước tiếp theo. Việc neo dữ kiện qua truy xuất (RAG) ngăn chặn ảo giác. Các lệnh gọi công cụ giúp tác tử thực sự thực hiện được các hành động trong thế giới thực. Đây là kiến trúc chuẩn mực của năm 2026.

Bốn mô hình này không loại trừ lẫn nhau theo trình tự tuyến tính. Một chatbot sản xuất năm 2026 sẽ điều hướng linh hoạt qua cả bốn: Dựa trên luật cho bước xác thực danh tính và các hành động nguy hiểm; Dựa trên truy xuất cho các câu hỏi thường gặp FAQ chuẩn mực; Mạng nơ-ron tạo sinh để diễn đạt câu từ tự nhiên; và Tác tử LLM cho các câu hỏi mở, đa bước và mơ hồ.

## Xây dựng thực tế (Build It)

### Bước 1: So khớp mẫu hình dựa trên luật

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA chỉ trong 20 dòng mã: Mẹo phản chiếu ("I feel sad" → "Why do you feel sad") là màn trình diễn bác sĩ tâm lý kinh điển của Weizenbaum năm 1966. Vẫn mang tính giáo khoa sâu sắc.

### Bước 2: Dựa trên truy xuất (FAQ)

Đoạn mã minh họa này sử dụng thư viện `sentence-transformers`. Trong mã nguồn chạy được `code/main.py` của bài học, thư viện chuẩn với độ tương đồng Jaccard được sử dụng để bài học chạy độc lập mà không cần phụ thuộc bên ngoài:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Từ chối dựa trên ngưỡng (Threshold-based refusal) là quyết định thiết kế then chốt: Nếu kết quả khớp nhất không đủ gần, trả về `None` để hệ thống chuyển tiếp sang cấp xử lý cao hơn.

### Bước 3: Mạng nơ-ron tạo sinh (Mô hình cơ sở)

Sử dụng một mô hình encoder-decoder nhỏ được điều chỉnh theo chỉ dẫn (FLAN-T5) hoặc một mô hình hội thoại fine-tune. Không thể dùng độc lập trong sản xuất năm 2026 (do nguy cơ mâu thuẫn dữ kiện, trôi dạt lạc đề, thông tin sai sự thật), nhưng vẫn được cắm bên trong các hệ thống lai để trau chuốt câu từ:

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Bước 4: Vòng lặp tác tử LLM (Agent Loop)

Cấu trúc sản xuất năm 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Ba yếu tố cốt lõi: Công cụ là các hàm Python mà LLM có thể gọi. Vòng lặp kết thúc khi LLM trả về câu trả lời cuối cùng thay vì một lệnh gọi công cụ. Ngân sách số bước (step budget) ngăn chặn vòng lặp vô tận khi gặp các tác vụ mơ hồ.

Trong hệ thống sản xuất thực tế, bạn sẽ bổ sung thêm: Neo dữ kiện trước bằng truy xuất (RAG), lan can bảo vệ (guardrails chặn hành động phá hủy khi chưa có xác nhận từ người dùng), khả năng quan sát (observability ghi nhật ký từng bước), và kiểm định tự động (evaluations đảm bảo hành vi của tác tử luôn đúng đặc tả kỹ thuật).

### Bước 5: Điều hướng lai (Hybrid Routing)

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Quy tắc thiết kế: Sử dụng luật tất định cho các hành động có tính phá hủy hoặc thay đổi trạng thái, sử dụng truy xuất cho các câu hỏi FAQ chuẩn, và sử dụng tác tử LLM cho tất cả các tình huống còn lại. Đây là chuẩn mực vận hành của các hệ thống chăm sóc khách hàng năm 2026.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Trường hợp sử dụng | Kiến trúc tối ưu |
|-------------------|-------------------|
| Đặt chỗ, thanh toán, xác thực danh tính | Máy trạng thái dựa trên luật + điền thông tin (slot filling) |
| Câu hỏi thường gặp chăm sóc khách hàng | Truy xuất trên ngân hàng câu trả lời đã kiểm duyệt |
| Trợ giúp giải đáp thắc mắc mở | Tác tử LLM kết hợp RAG + gọi công cụ (tool calls) |
| Công cụ nội bộ / Trợ lý lập trình IDE | Tác tử LLM với các công cụ (tìm kiếm, đọc file, ghi file) |
| Chatbot bạn đồng hành / nhân vật giả lập | LLM tinh chỉnh với system prompt nhân cách + truy xuất kiến thức nền |

Luôn luôn sử dụng kiến trúc điều hướng lai trong môi trường sản xuất. Không một kiến trúc đơn lẻ nào có thể xử lý hoàn hảo mọi loại yêu cầu. Bản thân tầng điều hướng thường là một bộ phân loại ý định (intent classifier) nhỏ gọn và cực nhanh.

## Các phương thức lỗi vẫn âm thầm lọt vào sản xuất

- **Bịa đặt tự tin (Confident fabrication).** Tác tử LLM cả quyết rằng nó đã hoàn thành một hành động trong khi thực tế nó chưa hề làm. Giải pháp: kiểm thực kết quả thực tế, ghi nhật ký mọi lệnh gọi công cụ, tuyệt đối không cho phép LLM tuyên bố đã thực hiện hành động nếu không có kết quả trả về thành công từ công cụ đó.
- **Tấn công tiêm nhiễm Prompt (Prompt injection).** Người dùng chèn đoạn văn bản độc hại nhằm ghi đè system prompt của hệ thống. Được xếp hạng LLM01 trong OWASP Top 10 cho ứng dụng LLM năm 2025. Có hai dạng: tiêm nhiễm trực tiếp (dán thẳng vào khung chat) và tiêm nhiễm gián tiếp (ẩn giấu bên trong tài liệu, email hoặc kết quả trả về của công cụ mà tác tử đọc).

  Tỷ lệ tấn công thành công thay đổi theo từng tình huống: Dao động từ ~0.5-8.5% trên các mô hình hàng đầu trong các bài đo công cụ và lập trình tổng quát. Nhưng trong các kịch bản nguy cơ cao (tấn công thích ứng nhằm vào AI coding agent, hoặc kiến trúc điều phối lỏng lẻo), tỷ lệ này có thể lên tới ~84%. Các lỗ hổng CVE thực tế trong sản xuất bao gồm EchoLeak (CVE-2025-32711, điểm CVSS 9.3) — lỗ hổng trích xuất dữ liệu zero-click trong Microsoft 365 Copilot bị kích hoạt bởi một email do kẻ tấn công kiểm soát.

  Giải pháp phòng vệ: Luôn coi đầu vào của người dùng là không đáng tin cậy trong suốt vòng lặp; khử khuẩn dữ liệu trước khi gọi công cụ; cách ly kết quả trả về của công cụ khỏi prompt chính; áp dụng mẫu Plan-Verify-Execute (PVE) nơi tác tử phải lập kế hoạch trước, sau đó kiểm thực từng hành động so với kế hoạch đó trước khi thực thi (ngăn chặn kết quả công cụ tiêm nhiễm các hành động mới chưa được lên kế hoạch); bắt buộc người dùng xác nhận thủ công cho các hành động có tính phá hủy; áp dụng nguyên tắc đặc quyền tối thiểu (least-privilege) cho phạm vi quyền hạn của công cụ.

  Không một kỹ thuật prompt engineering nào có thể triệt tiêu hoàn toàn rủi ro này. Bắt buộc phải có các lớp phòng thủ thực thi bên ngoài (LLM Guard, kiểm thực danh sách trắng allowlist, phát hiện dị thường ngữ nghĩa).
- **Trôi dạt phạm vi (Scope creep).** Tác tử đi chệch khỏi mục tiêu ban đầu vì một công cụ trả về thông tin liên quan ngoài lề. Giải pháp: thu hẹp giao ước công cụ (tool contracts); giữ system prompt tập trung cao độ; bổ sung chỉ số đo lường tỷ lệ lệch mục tiêu (off-task rate).
- **Vòng lặp vô tận (Infinite loops).** Tác tử liên tục gọi đi gọi lại cùng một công cụ. Giải pháp: ngân sách số bước tối đa, cơ chế loại bỏ trùng lặp lệnh gọi công cụ, sử dụng một LLM giám khảo đánh giá "chúng ta có đang đạt được tiến triển hay không".
- **Cạn kiệt cửa sổ ngữ cảnh (Context window exhaustion).** Hội thoại kéo dài đẩy các lượt tương tác ban đầu ra khỏi bộ nhớ ngữ cảnh. Giải pháp: tóm tắt các lượt tương tác cũ, truy xuất các lượt đối thoại liên quan trong quá khứ bằng độ tương đồng, hoặc chuyển sang dùng mô hình có cửa sổ ngữ cảnh dài.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Thiết kế ngăn xếp chatbot cho một trường hợp sử dụng cụ thể.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Dựa trên bối cảnh sản phẩm (nhu cầu người dùng, ràng buộc tuân thủ pháp lý, các công cụ sẵn có, khối lượng dữ liệu), bạn đưa ra:

1. Kiến trúc: Dựa trên luật, truy xuất, mạng nơ-ron tạo sinh, tác tử LLM, hoặc hybrid lai ghép (chỉ rõ luồng nào đi vào đâu).
2. Lựa chọn LLM nếu có: Đặt tên họ mô hình cụ thể (Claude, GPT-4, Llama-3.1, Mixtral). Khớp với chất lượng sử dụng công cụ và chi phí tài chính.
3. Chiến lược neo dữ kiện (Grounding): Các nguồn RAG, phương pháp truy xuất (xem bài học 14), giao ước công cụ.
4. Kế hoạch đánh giá: Tỷ lệ hoàn thành tác vụ (task success rate), độ chuẩn xác của lệnh gọi công cụ, tỷ lệ lệch mục tiêu (off-task rate), tỷ lệ ảo giác trên tập hội thoại kiểm thử riêng biệt.

Từ chối đề xuất tác tử thuần LLM cho bất kỳ hành động phá hủy hoặc thay đổi trạng thái nào (thanh toán, xóa tài khoản, sửa đổi dữ liệu) mà thiếu luồng xác nhận có cấu trúc từ người dùng. Từ chối bỏ qua bước kiểm toán bảo mật tấn công prompt injection nếu tác tử có quyền ghi (write access) vào bất kỳ hệ thống nào.
```

## Bài tập (Exercises)

1. **Dễ.** Cài đặt bộ phản hồi dựa trên luật ở trên với 10 mẫu hình cho bot đặt đồ uống quán cà phê. Kiểm thử các trường hợp biên: gọi trùng món, sửa đổi thành phần, hủy món, ý định không rõ ràng.
2. **Trung bình.** Xây dựng cơ chế lai kết hợp FAQ + dự phòng bằng LLM: 50 mục FAQ soạn sẵn cho sản phẩm SaaS, kết hợp LLM dự phòng kèm truy xuất tài liệu trang hướng dẫn. Đo lường tỷ lệ từ chối và độ chính xác trên 100 câu hỏi hỗ trợ thực tế.
3. **Khó.** Cài đặt vòng lặp tác tử ở trên với ba công cụ (tìm kiếm search, đọc dữ liệu người dùng read-user-data, gửi email send-email). Chạy đánh giá trên 50 kịch bản kiểm thử bao gồm cả các nỗ lực tấn công prompt injection. Báo cáo tỷ lệ lệch mục tiêu, tỷ lệ tác vụ thất bại và các trường hợp tiêm nhiễm thành công (nếu có).

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Ý định (Intent) | Điều người dùng muốn | Nhãn danh mục phân loại (book_flight, reset_password). Được điều hướng tới hàm xử lý tương ứng. |
| Ô thông tin (Slot) | Một mảnh dữ liệu | Tham số mà bot cần thu thập (ngày bay, điểm đến). Điền ô (Slot filling) là chuỗi các câu hỏi để thu thập đủ tham số. |
| RAG | Truy xuất + tạo sinh | Truy xuất tài liệu liên quan, sau đó neo câu trả lời của LLM vào ngữ cảnh đó. |
| Gọi công cụ (Tool call) | Kích hoạt hàm | LLM xuất ra một lệnh có cấu trúc gồm tên hàm + tham số. Môi trường thực thi sẽ chạy hàm và trả kết quả lại cho LLM. |
| Vòng lặp tác tử (Agent loop) | Lập kế hoạch, hành động, kiểm thực | Bộ điều khiển chạy xen kẽ các lệnh gọi LLM và các lệnh gọi công cụ cho đến khi tác vụ hoàn tất. |
| Prompt injection | Tấn công prompt | Dữ liệu đầu vào độc hại tìm cách ghi đè hoặc vô hiệu hóa system prompt của hệ thống. |

## Đọc thêm (Further Reading)

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — bài báo kinh điển biến hội thoại thành thước đo chuẩn mực của trí tuệ máy móc.
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — bài báo gốc khai sinh chatbot dựa trên luật.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — kiến trúc biến trạng thái cảm xúc của PARRY, chatbot có trạng thái đầu tiên.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — công trình của Google về chatbot nơ-ron ngay trước khi kỷ nguyên tác tử bùng nổ.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — bài báo đặt nền móng cho mô hình vòng lặp tác tử ReAct.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — cẩm nang thực chiến năm 2024 về xây dựng tác tử hiệu quả.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — bài báo tiên phong về tấn công tiêm nhiễm prompt gián tiếp.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — bảng xếp hạng đưa prompt injection lên vị trí rủi ro an ninh số một.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — các phòng tuyến kiến trúc điều phối bao gồm mẫu PVE và luồng xác nhận từ người dùng.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — trường hợp lỗ hổng CVE zero-click kinh điển minh họa lý do tác tử có quyền ghi bắt buộc phải có các lớp phòng thủ thời gian chạy.

