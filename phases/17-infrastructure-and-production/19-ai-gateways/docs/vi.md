# Cổng AI — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Một cổng kết nối (gateway) nằm giữa các ứng dụng của bạn và các nhà cung cấp mô hình (model providers). Các tính năng cốt lõi bao gồm định tuyến nhà cung cấp (provider routing), dự phòng chuyển đổi khi lỗi (fallback), thử lại (retries), giới hạn tốc độ (rate limiting), tham chiếu bí mật (secret references), khả năng quan sát (observability), và hàng rào bảo vệ (guardrails). Bức tranh thị trường năm 2026: **LiteLLM** là mã nguồn mở cấp phép MIT hỗ trợ hơn 100 nhà cung cấp, tương thích chuẩn OpenAI, nhưng bắt đầu quá tải quanh ngưỡng ~2000 RPS (chiếm 8 GB bộ nhớ, lỗi dây chuyền trong các bài kiểm chuẩn benchmark đã công bố); phù hợp nhất cho hệ sinh thái Python, lưu lượng <500 RPS, phát triển/tạo mẫu thử nghiệm (prototyping). **Portkey** được định vị ở tầng điều khiển (control plane) (hàng rào bảo vệ, làm sạch dữ liệu cá nhân PII redaction, phát hiện tấn công bẻ khóa jailbreak detection, lưu vết kiểm toán audit trails), chuyển sang mã nguồn mở Apache 2.0 vào tháng 3 năm 2026, độ trễ bổ sung (latency overhead) từ 20-40 ms, gói production giá $49/tháng. **Kong AI Gateway** được xây dựng trên nền tảng Kong Gateway — bài kiểm chuẩn nội bộ của Kong trên cùng cấu hình 12 CPU: nhanh hơn Portkey 228%, nhanh hơn LiteLLM 859%; định giá $100/mô hình/tháng (tối đa 5 mô hình trên gói Plus); phù hợp cho doanh nghiệp lớn nếu bạn đã vận hành Kong. **Bifrost** (Maxim AI) — tự động thử lại với thuật toán lùi thời gian tùy biến (configurable backoff), dự phòng sang Anthropic khi OpenAI gặp lỗi 429. **Cloudflare / Vercel AI Gateways** — dịch vụ được quản lý (managed), không tốn công vận hành (zero-ops), hỗ trợ thử lại cơ bản. Ràng buộc về vị trí lưu trữ dữ liệu (data residency) thúc đẩy quyết định tự lưu trữ (self-host); Portkey và Kong nằm ở khoảng giữa với phiên bản mã nguồn mở kết hợp tùy chọn dịch vụ được quản lý.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Learning Objectives

- Liệt kê bảy tính năng cốt lõi của một cổng kết nối AI (định tuyến, dự phòng khi lỗi, thử lại, giới hạn tốc độ, bí mật bảo mật, khả năng quan sát, hàng rào bảo vệ).
- Ánh xạ bốn cổng AI tiêu biểu của năm 2026 (LiteLLM, Portkey, Kong AI, Bifrost) tới giới hạn quy mô tải (scale ceilings) và kịch bản sử dụng phù hợp.
- Trích dẫn số liệu kiểm chuẩn của Kong (nhanh hơn 228% so với Portkey, 859% so với LiteLLM) và giải thích lý do số liệu này có ý nghĩa quan trọng với lưu lượng >500 RPS.
- Đưa ra quyết định lựa chọn tự lưu trữ (self-hosted) hay dùng dịch vụ được quản lý (managed) dựa trên yêu cầu cư trú dữ liệu (data residency) và ngân sách vận hành (ops budget).

## The Problem

Sản phẩm của bạn gửi yêu cầu tới OpenAI, Anthropic, và một mô hình Llama tự triển khai (self-hosted). Mỗi nhà cung cấp sở hữu một bộ SDK, mô hình xử lý lỗi, cơ chế giới hạn tốc độ (rate limit), và lược đồ xác thực (auth scheme) hoàn toàn khác nhau. Bạn cần cơ chế chuyển đổi dự phòng (failover) (nếu OpenAI trả về lỗi 429, tự động thử tiếp với Anthropic), một kho lưu trữ thông tin xác thực tập trung, khả năng quan sát hợp nhất, và giới hạn tốc độ theo từng khách thuê (per-tenant rate limits).

Tái phát minh những thành phần này ở tầng ứng dụng sẽ làm liên kết chặt chẽ (tightly couple) mọi dịch vụ với từng nhà cung cấp. Một tầng cổng kết nối (gateway layer) gom tất cả vào một tiến trình duy nhất với một API chuẩn (thường tương thích với OpenAI API), chịu trách nhiệm rẽ nhánh phân phối (fan out) yêu cầu đến các nhà cung cấp tương ứng.

## The Concept

### Bảy tính năng cốt lõi

1. **Định tuyến nhà cung cấp (Provider routing)** — Đặt OpenAI, Anthropic, Gemini, mô hình tự lưu trữ,... phía sau một giao diện API duy nhất.
2. **Dự phòng chuyển đổi (Fallback)** — Tự động thử lại ở nhà cung cấp khác khi gặp lỗi 429 (Too Many Requests), 5xx (lỗi máy chủ), hoặc suy giảm chất lượng đầu ra.
3. **Thử lại (Retries)** — Chiến lược lùi theo cấp số nhân (exponential backoff) với giới hạn số lần thử.
4. **Giới hạn tốc độ (Rate limits)** — Phân tầng theo từng khách thuê (tenant), khóa API (key), hoặc mô hình (model).
5. **Tham chiếu bí mật (Secret references)** — Nạp thông tin xác thực từ kho bảo mật (vault) trong thời gian chạy (runtime), tuyệt đối không để lộ trong mã nguồn ứng dụng.
6. **Khả năng quan sát (Observability)** — Tích hợp OpenTelemetry (OTel) + các thuộc tính GenAI (Phase 17 · 13) + phân bổ chi phí (cost attribution).
7. **Hàng rào bảo vệ (Guardrails)** — Làm sạch dữ liệu cá nhân (PII redaction), phát hiện tấn công bẻ khóa (jailbreak detection), bộ lọc chủ đề được phép (allowed-topics filters).

### LiteLLM — Mã nguồn mở MIT, Python

- Hỗ trợ hơn 100 nhà cung cấp, tương thích định dạng OpenAI, cấu hình bộ định tuyến linh hoạt, dự phòng lỗi, khả năng quan sát cơ bản.
- Bắt đầu suy sụp quanh ngưỡng 2000 RPS theo bài kiểm chuẩn của Kong; dấu chân bộ nhớ (memory footprint) lên tới 8 GB, phát sinh lỗi dây chuyền dưới tải liên tục.
- Phù hợp nhất: Ứng dụng Python, lưu lượng <500 RPS, cổng kết nối cho môi trường phát triển/staging, định tuyến mang tính thử nghiệm.
- Chi phí: $0 cho bản mã nguồn mở (OSS); có gói đám mây miễn phí giới hạn.

### Portkey — Định vị ở tầng điều khiển (Control Plane)

- Mã nguồn mở Apache 2.0 từ tháng 3 năm 2026. Tích hợp sẵn hàng rào bảo vệ, làm sạch PII, phát hiện jailbreak, lưu vết kiểm toán (audit trails).
- Độ trễ bổ sung từ 20-40 ms cho mỗi yêu cầu.
- Gói production giá $49/tháng kèm cam kết lưu trữ nhật ký (retention) và thỏa thuận mức dịch vụ (SLA).
- Phù hợp nhất: Các ngành chịu sự kiểm soát nghiêm ngặt cần gói giải pháp tích hợp sẵn hàng rào bảo vệ và khả năng quan sát.

### Kong AI Gateway — Bước đi hướng tới quy mô lớn (Scale Play)

- Được xây dựng trực tiếp trên nền Kong Gateway (sản phẩm API gateway lâu năm, trưởng thành bằng Lua + OpenResty).
- Kết quả kiểm chuẩn nội bộ của Kong trên hệ thống tương đương 12 CPU: nhanh hơn 228% so với Portkey, nhanh hơn 859% so với LiteLLM.
- Định giá: $100/mô hình/tháng, tối đa 5 mô hình trên gói Plus.
- Phù hợp nhất: Đã triển khai sẵn hệ sinh thái Kong; lưu lượng >1000 RPS; có ngân sách mua bản quyền doanh nghiệp.

### Bifrost (Maxim AI)

- Tự động thử lại với chiến lược lùi thời gian cấu hình được.
- Chuyển đổi dự phòng sang Anthropic khi OpenAI trả về lỗi 429 là kịch bản mẫu điển hình.
- Tân binh mới gia nhập thị trường; định hướng thương mại.

### Cloudflare AI Gateway / Vercel AI Gateway

- Dịch vụ được quản lý hoàn toàn, không tốn công vận hành (zero-ops). Tự động thử lại và quan sát cơ bản.
- Phù hợp nhất: Ứng dụng JavaScript phục vụ ở biên (edge-serving) triển khai trên Cloudflare hoặc Vercel.
- Hạn chế hơn so với Kong/Portkey về hàng rào bảo vệ chuyên sâu và cơ chế giới hạn tốc độ phức tạp.

### Tự lưu trữ (Self-hosted) so với Dịch vụ được quản lý (Managed)

Yêu cầu về vị trí lưu trữ dữ liệu (data residency) là yếu tố mang tính quyết định. Lĩnh vực y tế và tài chính mặc định lựa chọn tự lưu trữ (LiteLLM hoặc Portkey OSS hoặc Kong). Sản phẩm tiêu dùng đại chúng thường ưu tiên dịch vụ được quản lý (Cloudflare AI Gateway) hoặc phân tầng trung cấp (Portkey managed). Mô hình lai (hybrid): tự lưu trữ cho các khách thuê thuộc diện kiểm soát pháp lý, dùng dịch vụ quản lý cho các nhóm còn lại.

### Ngân sách độ trễ (Latency Budget)

- LiteLLM: Độ trễ bổ sung điển hình từ 5-15 ms.
- Portkey: Độ trễ bổ sung từ 20-40 ms.
- Kong: Độ trễ bổ sung từ 3-8 ms.
- Cloudflare/Vercel: Độ trễ bổ sung từ 1-3 ms (lợi thế tính toán tại biên edge).

Độ trễ của cổng kết nối cộng trực tiếp vào thời gian nhận token đầu tiên (TTFT - Time To First Token). Nếu SLA yêu cầu TTFT P99 < 100 ms, hãy chọn Kong hoặc Cloudflare. Với yêu cầu P99 < 500 ms, bất kỳ lựa chọn nào cũng đáp ứng được.

### Ngữ nghĩa giới hạn tốc độ (Rate-limit Semantics)

Thuật toán thùng token đơn giản (token-bucket) hoạt động tốt ở quy mô vừa phải. Hệ thống đa khách thuê (multi-tenant) phức tạp đòi hỏi cơ chế cửa sổ trượt (sliding-window) + cho phép bùng nổ lưu lượng tạm thời (burst allowance) + phân cấp theo gói dịch vụ từng khách thuê. LiteLLM cung cấp token-bucket; Kong hỗ trợ sliding-window; Portkey cung cấp phân cấp theo tầng (tiered).

### Kết hợp Cổng kết nối + Khả năng quan sát + Định tuyến

Phase 17 · 13 (khả năng quan sát) + 16 (định tuyến mô hình) + 19 (cổng kết nối AI) cùng thuộc một lớp kiến trúc hạ tầng trong môi trường sản xuất. Bạn có thể chọn một công cụ bao quát cả ba hoặc kết nối chúng với nhau: hầu hết các hệ thống triển khai năm 2026 kết hợp Helicone (khả năng quan sát) hoặc Portkey (hàng rào bảo vệ) cùng với Kong (tối ưu quy mô hiệu năng) để phân tách vai trò rõ ràng.

### Những con số cần ghi nhớ

- LiteLLM: Bắt đầu sụp đổ ở ngưỡng ~2000 RPS, ngốn 8 GB bộ nhớ.
- Portkey: Độ trễ bổ sung 20-40 ms; chuyển sang mã nguồn mở Apache 2.0 từ tháng 3 năm 2026.
- Kong: Nhanh hơn 228% so với Portkey, nhanh hơn 859% so với LiteLLM.
- Chi phí Kong: $100/mô hình/tháng, tối đa 5 mô hình trên gói Plus.
- Cloudflare/Vercel: Độ trễ bổ sung 1-3 ms tại mạng biên.

```figure
mx-gateway-fallback
```

## Use It

Tệp `code/main.py` mô phỏng quy trình định tuyến qua cổng kết nối với tính năng chuyển đổi dự phòng qua 3 nhà cung cấp dưới các tình huống tiêm lỗi 429/5xx. Mô phỏng báo cáo chi tiết về độ trễ, tỷ lệ thử lại (retry rate), và tỷ lệ chuyển đổi dự phòng thành công (fallback hit rate).

## Ship It

Bài học này xây dựng tệp `outputs/skill-gateway-picker.md`. Dựa trên quy mô, tư thế vận hành (ops posture), tuân thủ quy định pháp lý, và ngân sách độ trễ, kỹ năng này sẽ lựa chọn cổng kết nối tối ưu nhất.

## Exercises

1. Chạy `code/main.py`. Cấu hình chuỗi dự phòng từ OpenAI → Anthropic → mô hình tự lưu trữ. Tỷ lệ đáp ứng thành công kỳ vọng (hit rate) là bao nhiêu khi tỷ lệ lỗi của nhà cung cấp là 5%?
2. SLA của bạn yêu cầu TTFT P99 < 320 ms và riêng bản thân mô hình đã đo được 300 ms ở P99. Những cổng kết nối nào đảm bảo nằm trong ngân sách cho phép?
3. Một khách hàng ngành y tế yêu cầu bắt buộc tự lưu trữ (self-hosted) + làm sạch dữ liệu cá nhân PII + lưu vết kiểm toán (audit). Bạn sẽ chọn Portkey OSS hay Kong?
4. So sánh LiteLLM và Kong: ở ngưỡng trần RPS nào thì một đội ngũ kỹ thuật bắt buộc phải di chuyển hệ thống?
5. Thiết kế chính sách giới hạn tốc độ cho nền tảng SaaS đa khách thuê: gói miễn phí (free tier), gói dùng thử (trial tier), gói trả phí (paid tier). Nên chọn thuật toán thùng token (token-bucket) hay cửa sổ trượt (sliding-window)?

## Key Terms

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| Cổng kết nối (Gateway) | "Bộ môi giới API" | Tiến trình trung gian nằm giữa các ứng dụng và nhà cung cấp mô hình |
| LiteLLM | "Bản mã nguồn mở MIT" | Thư viện OSS Python, hơn 100 nhà cung cấp, quá tải tại 2K RPS |
| Portkey | "Cổng kết nối kèm hàng rào bảo vệ" | Tầng điều khiển + khả năng quan sát, cấp phép Apache 2.0 |
| Kong AI Gateway | "Giải pháp hiệu năng cao quy mô lớn" | Xây dựng trên Kong Gateway, dẫn đầu bài kiểm chuẩn hiệu năng |
| Bifrost | "Cổng kết nối của Maxim" | Công thức cấu hình thử lại + dự phòng sang Anthropic |
| Cloudflare AI Gateway | "Dịch vụ quản lý tại biên" | Cổng kết nối được quản lý triển khai tại mạng biên, zero-ops |
| Làm sạch PII (PII redaction) | "Tẩy rửa dữ liệu" | Che giấu dữ liệu bằng biểu thức chính quy (Regex) + nhận diện thực thể (NER) trước khi gửi tới mô hình |
| Phát hiện tấn công bẻ khóa (Jailbreak detection) | "Bảo vệ chống tiêm prompt" | Bộ phân loại kiểm duyệt trên dữ liệu đầu vào của người dùng |
| Vết kiểm toán (Audit trail) | "Nhật ký tuân thủ" | Bản ghi bất biến (immutable record) cho mọi lượt gọi LLM |
| Thùng token (Token-bucket) | "Giới hạn tốc độ đơn giản" | Bộ giới hạn tốc độ dựa trên cơ chế nạp đầy định kỳ |
| Cửa sổ trượt (Sliding-window) | "Giới hạn tốc độ chính xác" | Bộ giới hạn tốc độ theo cửa sổ thời gian; đảm bảo độ công bằng cao hơn |

## Further Reading

- [Bài kiểm chuẩn Kong AI Gateway](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — So sánh các cổng AI năm 2026](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top các công cụ cổng LLM năm 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [Kho mã nguồn LiteLLM trên GitHub](https://github.com/BerriAI/litellm)
- [Kho mã nguồn Portkey trên GitHub](https://github.com/Portkey-AI/gateway)
- [Tài liệu hướng dẫn Kong AI Gateway](https://docs.konghq.com/gateway/latest/ai-gateway/)
