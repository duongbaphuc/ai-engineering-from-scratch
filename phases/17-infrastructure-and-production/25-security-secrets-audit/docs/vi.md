# Bảo mật — Bí mật, Xoay vòng Khóa API, Nhật ký Kiểm toán, Rào chắn Bảo vệ

> Loại bỏ tình trạng rò rỉ và phân tán bí mật tràn lan (secret sprawl) thông qua các kho lưu trữ tập trung (centralized vaults như HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Tuyệt đối không lưu trữ thông tin xác thực trong các tệp cấu hình, tệp môi trường trong hệ thống quản lý phiên bản (VCS), hoặc bảng tính. Sử dụng vai trò IAM (IAM roles) thay vì các khóa tĩnh (static keys); áp dụng OIDC cho quy trình CI/CD. Mô hình cổng AI (AI-gateway pattern) là chuẩn mực sản xuất năm 2026: các ứng dụng → cổng AI → nhà cung cấp mô hình, trong đó cổng AI sẽ tự động trích xuất thông tin xác thực từ vault trong thời gian thực thi (runtime). Khi xoay vòng khóa (rotation) trên vault, toàn bộ ứng dụng sẽ nhận diện khóa mới chỉ trong vài phút — không cần triển khai lại (redeploy), không cần nhắn tin trên Slack hỏi "ai đang giữ khóa mới". Chính sách xoay vòng định kỳ ≤90 ngày; quét mã nguồn bằng TruffleHog / GitGuardian / Gitleaks trên từng lượt commit. Áp dụng mô hình Không tin cậy (Zero-trust): xác thực đa yếu tố (MFA), đăng nhập một lần (SSO), kiểm soát truy cập dựa trên vai trò/thuộc tính (RBAC/ABAC), mã token ngắn hạn, kiểm định tình trạng thiết bị (device posture). Tính năng làm sạch thông tin nhận dạng cá nhân (PII scrubbing) sử dụng nhận diện thực thể (NER) để che giấu dữ liệu y tế (PHI) / dữ liệu cá nhân (PII) trước khi chuyển tiếp; cơ chế token hóa nhất quán (consistent tokenization theo cách tiếp cận Mesh) ánh xạ các giá trị nhạy cảm tới các giá trị giữ chỗ (placeholders) cố định nhằm giúp LLM bảo toàn ngữ nghĩa mã lệnh và mối quan hệ thực thể. Kiểm soát luồng mạng ra (network egress): đặt các dịch vụ LLM trong một mạng con (subnet) VPC/VNet chuyên dụng với danh sách trắng (whitelist) chỉ cho phép kết nối đến `api.openai.com`, `api.anthropic.com`, v.v.; chặn đứng mọi kết nối outbound khác. Động lực sự cố năm 2026: cuộc tấn công chuỗi cung ứng (supply-chain attack) nhằm vào Vercel thông qua việc xâm phạm thông tin xác thực CI/CD đã làm lộ biến môi trường trên hàng nghìn triển khai khách hàng.

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 phút

## Mục tiêu học tập

- Liệt kê bốn phản mẫu (anti-patterns) trong quản lý thông tin bí mật (tệp cấu hình trong VCS, biến môi trường hardcoded, bảng tính, khóa tĩnh) và chỉ ra các giải pháp thay thế chuẩn mực.
- Giải thích mô hình cổng AI trích xuất thông tin bí mật từ kho lưu trữ (AI-gateway-pulls-from-vault) với vai trò là tiêu chuẩn sản xuất năm 2026.
- Cài đặt một bộ làm sạch PII (PII scrubber) với cơ chế token hóa nhất quán (cùng một giá trị → cùng một placeholder) để bảo toàn ngữ nghĩa văn bản.
- Phân tích sự cố chuỗi cung ứng Vercel năm 2026 và các bài học rút ra về vệ sinh an toàn thông tin xác thực trong CI/CD.

## Vấn đề thực tế

Một thực tập sinh vô tình commit tệp `.env` chứa các khóa API lên kho lưu trữ. Họ phát hiện và xóa nó ngay sau đó. Tuy nhiên, các khóa này đã nằm trong lịch sử git — công cụ quét GitGuardian phát hiện sự cố, và quy trình xoay vòng khóa của bạn lúc này biến thành: "Nhắn tin thông báo toàn đội trên Slack, cập nhật 40 tệp cấu hình, triển khai lại tất cả dịch vụ." Tám giờ sau, một nửa dịch vụ của bạn đã chạy bản mới, trong khi nửa còn lại vẫn đang chờ khung giờ bảo trì để cập nhật.

Ở một tình huống khác, câu lệnh gợi ý (prompt) của người dùng chứa nội dung: "Số SSN của tôi là 123-45-6789." Prompt này được gửi thẳng tới OpenAI. Dù tổ chức của bạn đã ký Thỏa thuận Đối tác Kinh doanh (BAA), chính sách bảo mật nội bộ vẫn bắt buộc phải che giấu thông tin PII trước khi chuyển tiếp ra bên ngoài. Bạn đã không thực hiện điều đó.

Trong một kịch bản khác, pod chạy mô hình LLM trên cụm EKS của bạn có quyền truy cập tùy ý ra mạng internet công cộng. Kẻ tấn công thực hiện trích xuất dữ liệu trái phép (exfiltration) thông qua tra cứu DNS tới tên miền do chúng kiểm soát. Không có cơ chế nào ngăn chặn hành vi này.

Bảo mật cho các dịch vụ LLM bắt buộc phải giải quyết triệt để cả ba véc-tơ tấn công này và có khả năng chứng minh tính tuân thủ sau đó: Thông tin xác thực được quản lý qua kho lưu trữ (Vault). Làm sạch dữ liệu PII. Lọc lưu lượng mạng chiều ra (egress filtering). Và nhật ký kiểm toán (audit logs) đóng vai trò là chuỗi bằng chứng xác thực.

## Khái niệm then chốt

### Kho lưu trữ tập trung + Trích xuất qua IAM Role

**Vault (Kho lưu trữ bí mật)**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Điểm chân lý duy nhất (single source of truth).

**IAM Role (Vai trò IAM)**: Ứng dụng hoặc cổng AI xác thực thông qua danh tính IAM của chính nó, không dựa vào khóa tĩnh. Vault sẽ trả về bí mật chỉ có hiệu lực tương ứng với thời hạn sống của token.

**Mô hình cổng AI (The AI-gateway pattern)**: Cổng AI trích xuất `OPENAI_API_KEY` từ vault tại thời điểm tiếp nhận yêu cầu (request time). Khi xoay vòng khóa trong vault, yêu cầu tiếp theo sẽ lập tức nhận được khóa mới. Không cần triển khai lại hệ thống.

### Chính sách xoay vòng khóa ≤ 90 ngày

Áp dụng cho mọi khóa API, root token của vault, và thông tin xác thực CI/CD. Tự động hóa quá trình xoay vòng khóa ở bất kỳ nơi nào có thể. Các quy trình xoay vòng thủ công phải được ghi nhật ký và giám sát chặt chẽ.

### Quét mã bí mật (Secret Scanning)

- **TruffleHog** — phân tích biểu thức chính quy (regex) kết hợp tính toán độ hỗn loạn thông tin (entropy) trên từng commit.
- **GitGuardian** — giải pháp thương mại với độ chính xác cao.
- **Gitleaks** — công cụ nguồn mở, dễ dàng nhúng vào đường ống CI.

Chạy quét trên mọi lượt commit. Chặn lệnh kéo mã (Pull Request) nếu phát hiện thông tin bí mật mới bị lộ.

### Tư thế Không tin cậy (Zero-trust Posture)

- Bắt buộc kích hoạt MFA trên toàn bộ tài khoản.
- Đăng nhập một lần (SSO) qua SAML/OIDC.
- Kiểm soát truy cập dựa trên vai trò (RBAC) hoặc thuộc tính (ABAC) để phân quyền chi tiết.
- Sử dụng token ngắn hạn (hiệu lực tính bằng giờ, không phải bằng ngày).
- Đánh giá tình trạng an ninh thiết bị (device posture) — chỉ cho phép các thiết bị doanh nghiệp đã mã hóa toàn bộ ổ đĩa truy cập.

### Làm sạch PII / PHI (PII / PHI Scrubbing)

Trước khi prompt rời khỏi hạ tầng nội bộ của bạn:

1. Nhận diện thực thể có tên (Named Entity Recognition - spaCy NER, Presidio hoặc giải pháp thương mại).
2. Che giấu các thực thể khớp mẫu: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. Token hóa nhất quán (cách tiếp cận Mesh): cùng một giá trị gốc sẽ luôn được ánh xạ sang cùng một placeholder, giúp LLM duy trì được mối liên kết ngữ nghĩa giữa các câu.
4. Tùy chọn ánh xạ ngược lại (reverse mapping) giá trị thực tế khi nhận phản hồi từ LLM.

Các bộ lọc regex tĩnh giúp bắt các mẫu định dạng cơ bản; mô hình NER phát hiện các thực thể phức tạp hơn. Nên kết hợp cả hai.

### Rào chắn bảo vệ đầu vào + đầu ra (Input + Output Guardrails)

Đầu vào: Chặn đứng các kỹ thuật vượt rào (jailbreak) đã biết, các chủ đề cấm; giới hạn tần suất gọi (rate-limit) theo từng người dùng.

Đầu ra: Sử dụng regex để làm sạch các bí mật bị lộ (mẫu định dạng khóa API, địa chỉ email trong ngữ cảnh từ chối phản hồi), kết hợp bộ phân loại (classifier) để phát hiện vi phạm chính sách nội dung.

### Danh sách trắng kết nối ra ngoài (Network Egress Whitelist)

Đặt các dịch vụ LLM trong một mạng con chuyên dụng:
- Danh sách trắng (Whitelist): `api.openai.com`, `api.anthropic.com`, các endpoint cơ sở dữ liệu véc-tơ, các endpoint của vault.
- Mọi kết nối khác: hủy bỏ (drop).
- Phân giải tên miền (DNS) qua bộ phân giải chỉ chấp nhận danh sách cho phép (ngăn chặn hành vi đánh cắp dữ liệu qua kỹ thuật DNS tunneling).

### Nhật ký kiểm toán (Audit Log)

Bản ghi bất biến (immutable log) cho mỗi lệnh gọi LLM bao gồm:
- Dấu thời gian (Timestamp).
- Định danh người dùng / đối tượng thuê (User / tenant).
- Giá trị băm của prompt (prompt hash — không lưu prompt thô vì lý do quyền riêng tư).
- Mô hình + phiên bản.
- Số lượng token tiêu thụ.
- Chi phí phát sinh.
- Giá trị băm của phản hồi (response hash).
- Mọi sự kiện kích hoạt rào chắn bảo vệ (guardrail trips).

Lưu trữ theo đúng yêu cầu quy định pháp lý (SOC 2 yêu cầu 1 năm, HIPAA yêu cầu 6 năm).

### Sự cố Vercel năm 2026

Cuộc tấn công chuỗi cung ứng: Thông tin xác thực CI/CD bị xâm phạm đã làm rò rỉ biến môi trường của hàng nghìn đợt triển khai khách hàng. Bài học rút ra: Thông tin xác thực CI/CD có giá trị tương đương với môi trường sản xuất (production). Phải lưu trữ trong vault, cấp quyền tối thiểu nghiêm ngặt, và xoay vòng liên tục.

### Các con số cần ghi nhớ

- Chính sách xoay vòng khóa: ≤ 90 ngày.
- Quét trên mọi commit: TruffleHog / GitGuardian / Gitleaks.
- Sự cố Vercel 2026: Thông tin xác thực CI/CD bị xâm phạm → hàng nghìn biến môi trường khách hàng bị rò rỉ.
- Thời gian lưu trữ nhật ký kiểm toán: SOC 2 = 1 năm, HIPAA = 6 năm.

```figure
i4-vault-rotation
```

## Thực hành (Use It)

Tệp `code/main.py` cài đặt một bộ làm sạch PII đơn giản với kỹ thuật token hóa nhất quán và hệ thống ghi nhật ký kiểm toán chỉ cho phép ghi nối tiếp (append-only audit log).

## Triển khai thực tế (Ship It)

Bài học này cung cấp sản phẩm mẫu `outputs/skill-llm-security-plan.md`. Dựa trên phạm vi pháp lý và hiện trạng hệ thống, tài liệu này lập kế hoạch chi tiết cho việc chuyển đổi sang vault, tích hợp bộ làm sạch dữ liệu, kiểm soát luồng mạng ra và thiết lập nhật ký kiểm toán.

## Bài tập

1. Chạy `code/main.py`. Gửi hai prompt cùng tham chiếu đến một số SSN giống nhau. Xác nhận rằng cả hai đều nhận được cùng một chuỗi placeholder.
2. Thiết kế chính sách kiểm soát luồng mạng ra (network egress policy) cho cụm triển khai vLLM trên EKS cần gọi OpenAI + Anthropic + Weaviate.
3. Bạn phát hiện một khóa bí mật nằm trong lịch sử git từ 2 năm trước. Phản ứng chuẩn xác là gì — xoay vòng khóa, làm sạch lịch sử git, hay thực hiện cả hai? Đưa ra lập luận bảo vệ quan điểm.
4. Nhật ký kiểm toán của bạn tăng trưởng 10 GB/ngày. Hãy thiết kế các tầng lưu trữ theo vòng đời (hot 30 ngày, warm 12 tháng, cold 6 năm).
5. Phân tích xem kỹ thuật đảo ngược token (thay thế lại giá trị thực vào câu trả lời của LLM) có xứng đáng với độ phức tạp triển khai tăng thêm so với việc giữ nguyên các placeholder hiển thị cho người dùng hay không.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Vault | "kho lưu trữ bí mật" | Dịch vụ quản lý thông tin xác thực tập trung |
| IAM role | "xác thực dựa trên danh tính" | Vai trò được ứng dụng đảm nhận; cấp thông tin xác thực ngắn hạn |
| OIDC for CI/CD | "token do cloud phát hành" | Không dùng khóa tĩnh trong CI — định danh thông qua OIDC |
| TruffleHog / GitGuardian / Gitleaks | "công cụ quét mã bí mật" | Phát hiện thông tin bí mật tại thời điểm commit |
| RBAC / ABAC | "kiểm soát truy cập" | Dựa trên vai trò (Role-based) so với Dựa trên thuộc tính (Attribute-based) |
| PII scrubbing | "che giấu dữ liệu" | Loại bỏ hoặc mã hóa token các thực thể nhạy cảm |
| Consistent tokenization | "placeholder cố định" | Cùng một giá trị gốc → luôn ra cùng một token |
| Mesh approach | "token hóa kiểu Mesh" | Mẫu token hóa bảo toàn ngữ nghĩa |
| Egress whitelist | "danh sách trắng chiều ra" | Chỉ cho phép kết nối tới các tên miền được phê duyệt |
| Audit log | "lịch sử bất biến" | Bản ghi chỉ thêm (append-only) phục vụ tuân thủ quy định |

## Tài liệu tham khảo

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — Phát hiện và ẩn danh hóa dữ liệu PII.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
