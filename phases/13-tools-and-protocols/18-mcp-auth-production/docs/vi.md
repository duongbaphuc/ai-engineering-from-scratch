# Xác thực & Ủy quyền MCP trong môi trường sản xuất: Đăng ký và Token ràng buộc Bên phát hành

> Bài học 16 đã xây dựng máy trạng thái OAuth 2.1. Bài học này làm vững chắc các ranh giới vận hành thực tế cho chuẩn MCP 2026-07-28: ưu tiên Tài liệu Siêu dữ liệu Client ID (Client ID Metadata Documents - CIMD), đăng ký client động (Dynamic Client Registration - DCR) đã lỗi thời chỉ dùng cho mục đích tương thích, xác thực bên phát hành trong phản hồi ủy quyền, thông tin client có khóa theo bên phát hành, làm mới tập khóa JWKS, và token được ghim đối tượng thụ hưởng (audience) trên từng yêu cầu phi trạng thái.
>
> **Lưu ý đặc tả (2026-07-28):** Đăng ký Client Động bị coi là lỗi thời để nhường chỗ cho Tài liệu Siêu dữ liệu Client ID. DCR chỉ còn đóng vai trò là cơ chế tương thích kế thừa. Khi sử dụng DCR, client bắt buộc phải khai báo chính xác `application_type`. Client phải xác thực giá trị `iss` theo RFC 9207 khi xuất hiện và tuyệt đối không bao giờ tái sử dụng thông tin xác thực xuyên suốt các bên phát hành của máy chủ ủy quyền.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python (thư viện chuẩn - stdlib)
**Yêu cầu tiên quyết:** Phase 13 · 16 (máy trạng thái OAuth 2.1), Phase 13 · 17 (cổng kết nối)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Khám phá máy chủ ủy quyền (authorization server) thông qua siêu dữ liệu RFC 8414 và xác minh hợp đồng kỹ thuật.
- Đăng ký thông qua Tài liệu Siêu dữ liệu Client ID (CIMD) và cô lập cơ chế DCR đã lỗi thời như một phương án dự phòng.
- Xác thực tham số `iss` theo RFC 9207, gắn khóa đăng ký theo bên phát hành của máy chủ ủy quyền, và gắn khóa token ràng buộc tài nguyên theo cặp (bên phát hành, tài nguyên).
- Lưu đệm và làm mới các khóa JWKS theo lịch trình để việc xác minh chữ ký sống sót qua các đợt xoay vòng khóa (key roll-over).
- Ghim token vào một tài nguyên MCP duy nhất bằng các chỉ báo tài nguyên RFC 8707 và từ chối hành vi tái sử dụng kiểu đại diện bị nhầm lẫn (confused-deputy reuse).
- Lựa chọn giữa xác thực JWT hoặc nội soi token (token introspection), xác định tính tươi mới của việc thu hồi (revocation freshness), và chuyển sang trạng thái an toàn khi các phụ thuộc danh tính không khả dụng.
- Phân tách rạch ròi giữa máy chủ ủy quyền, máy chủ tài nguyên và client để mỗi bên chỉ thực thi các bước kiểm tra thuộc quyền hạn của mình.
- Kiểm toán máy chủ ủy quyền dựa trên bảng kiểm tra triển khai thực tế và từ chối quy trình đăng ký hoặc tái sử dụng token không an toàn.

## Vấn đề đặt ra

Trình mô phỏng ở Bài học 16 vận hành OAuth 2.1 hoàn toàn trong bộ nhớ. Môi trường sản xuất thực tế tồn tại ba lỗ hổng vận hành mà một bộ mô phỏng trong bộ nhớ không thể thấy được.

Lỗ hổng đầu tiên là khâu đăng ký và cô lập thông tin xác thực. Một tổ chức thực tế có thể vận hành hàng trăm máy chủ MCP và hàng nghìn client MCP. Bản sửa đổi 2026-07-28 ưu tiên **Tài liệu Siêu dữ liệu Client ID (CIMD)**: client sử dụng một URL HTTPS có đường dẫn do chính mình kiểm soát làm định danh, và máy chủ ủy quyền sẽ chủ động kéo siêu dữ liệu về. Đăng ký động theo RFC 7591 chỉ còn tồn tại như một lộ trình tương thích kế thừa đã bị bãi bỏ. Khi DCR là không thể tránh khỏi, yêu cầu bắt buộc phải khai báo đúng `application_type`. Client lưu trữ các bản ghi đăng ký theo bên phát hành của máy chủ ủy quyền và lưu trữ access token theo cặp `(bên phát hành, tài nguyên)`. Một bên phát hành thay đổi đồng nghĩa với một lần đăng ký mới, và một tài nguyên khác đòi hỏi một token được ràng buộc đối tượng thụ hưởng riêng biệt.

Lỗ hổng thứ hai là xoay vòng khóa (key rotation). Xác thực JWT phụ thuộc vào các khóa ký của máy chủ ủy quyền, được xuất bản dưới dạng Tập khóa Web JSON (JWKS). Máy chủ ủy quyền xoay vòng các khóa này theo lịch trình (thường là hàng giờ, đôi khi nhanh hơn trong các đợt xử lý sự cố). Một máy chủ MCP chỉ nạp JWKS một lần duy nhất lúc khởi động sẽ xác thực rất tốt cho đến khi chạm cửa sổ xoay vòng khóa — sau đó mọi yêu cầu đều thất bại cho đến khi khởi động lại tiến trình. Môi trường sản xuất kết nối JWKS như một giá trị được lưu đệm kèm tác vụ làm mới định kỳ ghi đè lên bộ đệm trước khi các khóa cũ hết hạn, cộng với cơ chế nạp dự phòng khi bị trượt bộ đệm (cache miss) trong trường hợp xuất hiện một token được ký bởi khóa mới hơn khóa trong bộ nhớ đệm.

Lỗ hổng thứ ba là ràng buộc đối tượng thụ hưởng (audience binding). Bài học 16 đã giới thiệu các chỉ báo tài nguyên RFC 8707. Trong môi trường sản xuất, chỉ báo đó trở thành một bước kiểm tra xác nhận (claim check) bắt buộc trên mọi yêu cầu. Máy chủ MCP so sánh `token.aud` với URL tài nguyên chuẩn tắc của chính nó và từ chối các trường hợp không khớp bằng mã lỗi HTTP 401. Đây là biện pháp phòng thủ duy nhất chống lại việc một máy chủ MCP thượng nguồn (hoặc một client độc hại nắm giữ token dành cho một máy chủ khác) phát lại token đó để tấn công một máy chủ khác trong cùng lưới tin cậy (trust mesh).

Bài học này ánh xạ từng lỗ hổng vào một thành phần cụ thể trên bề mặt giao thức. Tài liệu siêu dữ liệu là một điểm cuối HTTP. Làm mới bộ đệm JWKS là một tác vụ theo lịch cộng với một kho lưu trữ khóa-giá trị (key-value cache). Xác thực JWT là một quy trình thường lệ mà máy chủ tài nguyên thực thi trước khi điều phối bất kỳ công cụ nào. Hãy giữ ba vai trò hoàn toàn tách biệt và mỗi bên chỉ thực thi những kiểm tra thuộc thẩm quyền của mình: máy chủ ủy quyền phát hành và xoay vòng khóa, máy chủ tài nguyên lưu đệm và xác thực, client khám phá và đăng ký.

## Phạm vi: Thực thi trong sản xuất sau Bài học 16

[Bài học 16: Bảo mật MCP với OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/vi.md) đảm nhiệm máy trạng thái mã ủy quyền, PKCE, khám phá tài nguyên được bảo vệ, chỉ báo tài nguyên, và các quyết định về phạm vi quyền. Bài học này không định nghĩa thêm một luồng OAuth thứ hai. Nó bắt đầu sau khi các hợp đồng đó đã hiện hữu và đặt câu hỏi: làm thế nào một máy chủ tài nguyên đã triển khai tiếp tục thực thi các quy tắc đó trong suốt quá trình xoay vòng khóa, xác thực token mờ đục, thu hồi token, sự cố phụ thuộc, triển khai phát hành (rollout), và xử lý sự cố an ninh.

Ranh giới sản xuất hẹp hơn và mang tính vận hành trực tiếp hơn:

- Lộ trình JWT xác minh bên phát hành đã ghim, thuật toán, khóa ký, đối tượng thụ hưởng, các trường thời gian, và phạm vi quyền trên mọi yêu cầu trong khi vẫn làm mới JWKS an toàn.
- Lộ trình token mờ đục (opaque-token path) gọi điểm cuối nội soi đã xác thực của bên phát hành và xác thực trạng thái hoạt động trả về, đối tượng thụ hưởng hoặc tài nguyên, thời hạn hết hạn, chủ thể, và các phạm vi quyền.
- Chính sách thu hồi xác định mức độ nhanh chóng mà một thông tin xác thực phải ngừng hoạt động và bộ nhớ đệm nào có thể làm chậm trễ thực tế đó.
- Chính sách xử lý sự cố quyết định điều gì xảy ra khi hạ tầng khám phá, JWKS, nội soi hoặc thu hồi không khả dụng.
- Bằng chứng kiểm toán ghi nhận siêu dữ liệu bên phát hành, tập khóa hoặc phản hồi nội soi, các xác nhận của token, phiên bản chính sách, và lý do từ chối mà không lưu trữ trực tiếp chuỗi token thô.

Sự phân định này giúp các bài học giữ được tính lắp ghép mô-đun. Bài học 16 chứng minh luồng hoạt động. Bài học 18 chứng minh rằng một token vẫn duy trì độ tin cậy, hoặc bị từ chối, sau khi nó tiến vào một lộ trình yêu cầu MCP thực tế.

## Khái niệm cốt lõi

### RFC 8414 — Siêu dữ liệu Máy chủ Ủy quyền OAuth

Một tài liệu tại `/.well-known/oauth-authorization-server` mô tả mọi thứ mà client cần:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

Một client khi được cung cấp URL tài nguyên MCP sẽ tiến hành khám phá theo chuỗi: `oauth-protected-resource` từ RFC 9728 (tài liệu của máy chủ tài nguyên) chỉ ra bên phát hành, sau đó `oauth-authorization-server` (RFC này) cung cấp tên của từng điểm cuối. Client tuyệt đối không mã hóa cứng (hard-code) một URL ủy quyền nào.

Đối với một định danh tài nguyên có đường dẫn, hãy chèn phân đoạn well-known trước đường dẫn đó. Ví dụ, `https://mcp.example.com/team/server` phân giải siêu dữ liệu tài nguyên được bảo vệ tại `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`. Việc nối thêm `/.well-known/...` vào sau đường dẫn tài nguyên là hoàn toàn sai.

Bản hợp đồng bạn cần xác minh trước khi tin cậy một IdP cho MCP:

- `code_challenge_methods_supported` bao gồm `S256` (PKCE theo RFC 7636). Đặc tả nêu rõ: nếu trường này **vắng mặt**, máy chủ ủy quyền không hỗ trợ PKCE và client **BẮT BUỘC** phải từ chối tiếp tục.
- `grant_types_supported` bao gồm `authorization_code` và từ chối `password` cùng `implicit`.
- Ít nhất một lộ trình đăng ký khả dụng: `client_id_metadata_document_supported: true` (CIMD, ưu tiên), một client đã đăng ký trước, hoặc `registration_endpoint` (tương thích kế thừa RFC 7591 đã bãi bỏ).
- Nếu `authorization_response_iss_parameter_supported` là true, client bắt buộc phải yêu cầu giá trị `iss` theo RFC 9207 trả về và so sánh chính xác với bên phát hành đã ghi nhận trước khi chuyển hướng.
- `response_types_supported` phải chính xác là `["code"]` đối với OAuth 2.1.

Nếu thiếu `S256`, máy chủ MCP kiên quyết từ chối triển khai kết nối với IdP này — không có chế độ suy giảm (degraded mode) nào dành cho PKCE. Nếu *cả hai* lộ trình đăng ký đều không được công bố và bạn không có sẵn `client_id` đăng ký trước, bạn cũng không thể đăng ký; lỗi nằm ở bản kê khai triển khai (deployment manifest), chứ không phải do mã nguồn.

### RFC 9728 (tóm lược) — Siêu dữ liệu Tài nguyên được Bảo vệ

Bài học 16 đã đề cập đến RFC 9728. Điểm khác biệt trong môi trường sản xuất: tài liệu này là nơi duy nhất mà client tìm kiếm để xác định danh sách các máy chủ ủy quyền được tin cậy bởi *chính* máy chủ MCP này. Một máy chủ MCP đơn lẻ có thể chấp nhận token từ nhiều IdP (một cho nhân viên, một cho đối tác). RFC 9728 khai báo tập hợp đó; RFC 8414 tài liệu hóa những gì từng IdP hỗ trợ.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Tài liệu Siêu dữ liệu Client ID (Mặc định được khuyến nghị)

CIMD đảo ngược quá trình đăng ký từ cơ chế *đẩy* (push) sang cơ chế *kéo* (pull). Thay vì yêu cầu máy chủ ủy quyền đúc ra một `client_id`, client sử dụng một URL HTTPS do chính mình kiểm soát **như là** `client_id` của nó. URL này trỏ tới một tài liệu siêu dữ liệu JSON; máy chủ ủy quyền sẽ nạp nó theo nhu cầu trong suốt luồng OAuth. Niềm tin bắt nguồn từ DNS: nếu bên vận hành máy chủ tin cậy `app.example.com`, họ sẽ tin cậy client được phục vụ từ `https://app.example.com/client.json`. Không còn vòng trao đổi đăng ký, không làm cạn kiệt không gian tên `client_id`, không cần đồng bộ trạng thái trên từng máy chủ.

Tài liệu siêu dữ liệu do client lưu trữ:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

Giá trị `client_id` trong tài liệu **BẮT BUỘC** phải bằng URL nơi nó được phục vụ (máy chủ ủy quyền sẽ kiểm tra điều này; các sai lệch sẽ bị từ chối). Máy chủ ủy quyền công bố sự hỗ trợ bằng cờ `client_id_metadata_document_supported: true` trong siêu dữ liệu RFC 8414 của nó.

Đối với hợp đồng CIMD hiện hành, các trường `client_id`, `client_name`, và một mảng `redirect_uris` không rỗng là bắt buộc. Mã định danh client là một URL HTTPS tuyệt đối có đường dẫn. Trường `application_type` có thể được đưa vào, nhưng không phải là trường bắt buộc của CIMD. Đừng sao chép yêu cầu bắt buộc `application_type` của DCR sang lộ trình CIMD ưu tiên.

Hai sự thật an ninh được đặc tả chỉ rõ:

- **SSRF.** Máy chủ ủy quyền nạp một URL do người khác cung cấp. Nó bắt buộc phải phòng vệ chống tấn công giả mạo yêu cầu từ phía máy chủ (không nạp các điểm cuối nội bộ hoặc quản trị).
- **Mạo danh localhost.** Chỉ riêng CIMD không thể ngăn chặn một kẻ tấn công cục bộ tuyên bố quyền sở hữu URL siêu dữ liệu của một client hợp pháp và liên kết với bất kỳ chuyển hướng `localhost` nào. Máy chủ ủy quyền **BẮT BUỘC** phải hiển thị rõ ràng tên máy chủ của URI chuyển hướng trong lúc xin sự đồng thuận và **NÊN** cảnh báo đối với các chuyển hướng chỉ có `localhost`.

Vì CIMD không cần trạng thái phía máy chủ, không cần phải duy trì một cơ quan đăng ký (registrar) như cách DCR yêu cầu. Phía client ở trạng thái chỉ đọc: phục vụ tài liệu siêu dữ liệu của bạn từ một điểm cuối HTTPS tĩnh và để máy chủ ủy quyền tự kéo về.

Nếu bên vận hành máy chủ ủy quyền đã cấp sẵn một định danh client, hãy sử dụng bản đăng ký thuộc phạm vi bên phát hành đó trước khi thử đăng ký tự động. Nếu không, hãy ưu tiên CIMD. Chỉ sử dụng DCR đã lỗi thời khi bên phát hành không thể dùng cả việc đăng ký trước lẫn CIMD.

### RFC 7591: Đăng ký tương thích kế thừa đã bị bãi bỏ

DCR đã bị coi là lỗi thời trong bản sửa đổi 2026-07-28. Chỉ giữ lại nó cho các máy chủ ủy quyền không thể tiêu thụ CIMD và khi việc đăng ký trước là bất khả thi. Một client tương thích sẽ gửi yêu cầu POST:

```json
{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Máy chủ phản hồi với `client_id` và một `registration_access_token` để cập nhật sau này:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

Trường `application_type` không phải là để trang trí. Một client desktop chạy loopback khai báo `native`; một client lưu trữ trên máy chủ khai báo `web` và sử dụng các URI chuyển hướng HTTPS. Giá trị mặc định `token_endpoint_auth_method: none` là phù hợp cho một client native công khai. Nó chỉ nhận về một `client_id`, với PKCE cung cấp bằng chứng sở hữu (proof-of-possession).

Ba cạm bẫy trong môi trường sản xuất:

- Điểm cuối đăng ký bắt buộc phải giới hạn tốc độ theo IP nguồn. Nếu không có điều đó, một kẻ xấu có thể viết mã kịch bản tạo ra hàng triệu lượt đăng ký giả và làm cạn kiệt không gian tên `client_id`. Hãy chạy kiểm tra giới hạn tốc độ trước khi cơ quan đăng ký xử lý yêu cầu.
- `software_statement` (một JWT có chữ ký bảo chứng cho client) là bắt buộc ở một số IdP doanh nghiệp. Trình mô phỏng của bài học bỏ qua nó; môi trường sản xuất cần gắn một bước xác minh từ chối các lượt đăng ký không có chữ ký từ bất kỳ thứ gì ngoài các URI chuyển hướng localhost.
- `registration_access_token` bắt buộc phải được lưu trữ dưới dạng băm, không lưu dạng văn bản thô. Kẻ tấn công đánh cắp token này có thể viết lại toàn bộ các URI chuyển hướng của client.

### RFC 8707 (tóm lược) — Chỉ báo Tài nguyên

Bài học 16 đã thiết lập cấu trúc này. Quy tắc sản xuất: mọi yêu cầu lấy token đều bao gồm `resource=<canonical-mcp-url>`, và máy chủ MCP xác minh `token.aud` khớp với URL tài nguyên của chính nó trên mỗi lệnh gọi. URI chuẩn tắc là định danh *cụ thể nhất* dành cho máy chủ: nó sử dụng giao thức và tên máy chủ chữ thường, không có đoạn phân mảnh (fragment), và theo thông lệ không có dấu gạch chéo cuối cùng. Thành phần đường dẫn **không** bị lược bỏ theo quy tắc — đặc tả giữ lại nó khi cần thiết để định danh một máy chủ MCP riêng lẻ. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, và `https://mcp.example.com/server/mcp` đều là các URI chuẩn tắc hợp lệ. Hãy chọn một URI cho mỗi máy chủ và ghim `aud` chính xác vào giá trị đó.

### RFC 7636 (tóm lược) — PKCE

PKCE là bắt buộc trong OAuth 2.1. Luồng mã ủy quyền của bài học luôn mang `code_challenge` và `code_verifier`. Máy chủ từ chối bất kỳ yêu cầu lấy token nào không có mã xác minh hoặc có mã xác minh không băm khớp với thử thách đã lưu.

### Hồ sơ ủy quyền MCP 2026-07-28

Bản sửa đổi MCP hiện tại giữ nguyên ranh giới máy chủ tài nguyên OAuth trong khi làm cho tầng giao vận MCP trở nên phi trạng thái. Không có phiên giao thức nào để lưu đệm quyết định danh tính. Tầng ủy quyền do đó phải xác thực độc lập từng yêu cầu:

- Triển khai siêu dữ liệu tài nguyên được bảo vệ theo RFC 9728, và cung cấp vị trí của nó thông qua tiêu đề `WWW-Authenticate: Bearer resource_metadata="..."` trên mã lỗi 401 **hoặc** URI well-known `/.well-known/oauth-protected-resource` (SEP-985 đã biến tiêu đề này thành tùy chọn với cơ chế dự phòng qua well-known). Trường `authorization_servers` của siêu dữ liệu **BẮT BUỘC** phải nêu tên ít nhất một máy chủ.
- Chỉ chấp nhận token qua `Authorization: Bearer ...` trên **mọi** yêu cầu — tuyệt đối không nhận qua chuỗi truy vấn (query string), không bao giờ chỉ xác thực một lần lúc bắt đầu phiên.
- Xác thực `aud`, `iss`, `exp`, và các phạm vi quyền bắt buộc trên từng yêu cầu. Máy chủ **BẮT BUỘC** phải xác thực rằng token được phát hành riêng cho nó (đối tượng thụ hưởng); một trường `aud` bị thiếu hoặc không khớp sẽ bị từ chối, tuyệt đối không được coi là ký tự đại diện (wildcard).
- Khi trả về 401/403, trả về `WWW-Authenticate: Bearer` mang `error=...`, tham số `resource_metadata="<PRM-URL>"` (URL của tài nguyên siêu dữ liệu, *không phải* tài nguyên thô), và `scope="..."` khi gặp lỗi `insufficient_scope` (403). Lưu ý: tham số là `resource_metadata`, một con trỏ khám phá — không có tham số `resource` trong lời thách thức.
- Việc khám phá máy chủ ủy quyền chấp nhận **hoặc** siêu dữ liệu OAuth RFC 8414 **hoặc** OpenID Connect Discovery 1.0; client phải thử cả hai hậu tố well-known theo thứ tự ưu tiên.
- Client (không phải máy chủ) có trách nhiệm phòng thủ chống lại các cuộc **tấn công tráo đổi (mix-up attacks)**: nó ghi lại `issuer` kỳ vọng trước khi chuyển hướng và xác thực giá trị `iss` trả về trong phản hồi ủy quyền thực tế (RFC 9207) trước khi đổi mã lấy token. Chỉ riêng PKCE không thể ngăn chặn tấn công tráo đổi, vì client sẽ trao `code_verifier` của nó cho bất kỳ điểm cuối token nào mà nó bị dẫn dụ tới.
- Một thông tin xác thực client thuộc về một bên phát hành máy chủ ủy quyền duy nhất. Nếu việc khám phá phân giải ra một bên phát hành khác, client phải đăng ký lại thay vì xuất trình `client_id`, token đăng ký, hoặc access token cũ.
- CIMD là cơ chế đăng ký được ưu tiên. DCR đã bị coi là lỗi thời; một yêu cầu DCR tương thích vẫn bắt buộc phải khai báo đúng `application_type`.

Bản thảo OAuth 2.1 là nền tảng; các chuẩn RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD là bề mặt giao tiếp; đặc tả MCP là hồ sơ quy định.

### Bảng kiểm tra năng lực triển khai

Bảng tính năng của các nhà cung cấp thường nhanh chóng trở nên lỗi thời. Hãy trực tiếp kiểm tra siêu dữ liệu được trả về bởi máy chủ ủy quyền mà bạn thực sự sẽ triển khai. Ranh giới kiểm tra mang tính máy móc:

| Kiểm tra | Quyết định bắt buộc |
|---|---|
| Bên phát hành khám phá được | Chính xác bên phát hành HTTPS theo kỳ vọng của chính sách |
| PKCE | Đã công bố `S256`; nếu không có thì dừng lại ngay |
| Đăng ký | Ưu tiên CIMD, chấp nhận đăng ký trước, DCR chỉ là tương thích lỗi thời |
| Phản hồi ủy quyền | Xác thực giá trị `iss` theo RFC 9207 khi xuất hiện hoặc được công bố |
| Ràng buộc tài nguyên | Yêu cầu token mang `resource`; máy chủ tài nguyên đòi hỏi `aud` trùng khớp |
| Lưu trữ thông tin xác thực | Gắn khóa client ID và thông tin đăng ký theo bên phát hành; gắn khóa access token theo cặp (bên phát hành, tài nguyên) |
| Tương thích DCR | Khai báo `native` hoặc `web`; từ chối các URI chuyển hướng không khớp với kiểu ứng dụng đã khai báo |

Không suy diễn sự hỗ trợ từ tên sản phẩm hay gói giá dịch vụ. Hãy ghi lại tài liệu đã khám phá vào bằng chứng triển khai và từ chối nếu thiếu bất kỳ trường bắt buộc nào.

### Mô hình làm mới JWKS (Xoay vòng tại AS, Làm mới tại Resource Server)

Cần phân định rạch ròi hai động từ, vì việc nhầm lẫn giữa chúng là một lỗi thực tế thường gặp trong sản xuất:

- **Xoay vòng (Rotate)** là việc mà *máy chủ ủy quyền* thực hiện: đúc ra một khóa ký mới, xuất bản nó trong JWKS, và thu hồi khóa cũ sau đó. Máy chủ tài nguyên hoàn toàn không tham gia vào việc này và không thể làm được — nó không nắm giữ khóa riêng của IdP.
- **Làm mới (Refresh)** là việc mà *máy chủ tài nguyên* thực hiện: gửi lại yêu cầu `GET` để tải JWKS đã xuất bản vào bộ nhớ đệm của mình. Đó là hành động JWKS duy nhất mà một máy chủ tài nguyên từng thực hiện.

Chế độ lỗi trong sản xuất là một bộ nhớ đệm bị cũ kỹ (stale cache). Hãy giải quyết bằng một tác vụ làm mới theo lịch cộng với một kho lưu trữ khóa-giá trị. Máy chủ tài nguyên chạy một tác vụ (cron, bộ đếm thời gian) định kỳ nạp `<issuer>/.well-known/jwks.json` và ghi đè `cache[issuer] = {keys, fetched_at}`. Bộ xác thực đọc từ bộ đệm đó. Một token có `kid` bị thiếu trong bộ đệm sẽ kích hoạt **duy nhất một lần** làm mới đồng bộ như một phương án dự phòng, sau đó kiểm tra lại. Cơ chế này xử lý cùng lúc hai trường hợp: đợt làm mới theo lịch, và các cửa sổ gối đầu khóa khi một token được ký bởi một khóa hoàn toàn mới xuất hiện trước đợt làm mới định kỳ tiếp theo.

Hành động dự phòng **bắt buộc phải là nạp lại (re-fetch), tuyệt đối không bao giờ là xoay vòng (rotate)**. Nếu bạn nối lộ trình trượt bộ đệm vào một hành động xoay vòng và đúc khóa mới, hai lỗi sẽ phát sinh: (1) đúc một khóa mới sẽ sinh ra một `kid` *vẫn* không khớp với token, do đó việc tra cứu vẫn thất bại; và (2) kẻ tấn công rải các token với các giá trị `kid` ngẫu nhiên sẽ ép hệ thống tạo ra vô hạn các khóa mới — một cuộc tấn công từ chối dịch vụ tự gây ra (self-inflicted DoS). Một thao tác nạp lại có tính lũy suy, do đó một `kid` giả mạo chỉ làm tốn tối đa một lần nạp vô ích.

Cấu trúc bộ đệm:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

Trạng thái ổn định thường duy trì hai khóa cùng lúc. Các máy chủ ủy quyền xoay vòng bằng cách giới thiệu khóa tiếp theo (`k_2026_04`) trước khi thu hồi khóa trước đó (`k_2026_03`), nhờ đó các token được phát hành dưới khóa cũ vẫn hợp lệ cho đến khi chúng hết hạn. Bộ nhớ đệm lưu trữ phép hợp của các khóa; bộ xác thực chọn khóa theo `kid`.

### Quy trình xác thực

Máy chủ MCP chạy quy trình xác thực trước khi điều phối bất kỳ công cụ nào. Cấu trúc mà `code/main.py` sử dụng:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

Hàm `validate` giải mã JWT, phân giải khóa ký từ bộ đệm JWKS (làm mới một lần khi bị trượt bộ đệm), xác minh chữ ký, sau đó kiểm tra `iss` đối chiếu với danh sách cho phép, `aud` đối chiếu với tài nguyên chuẩn tắc của máy chủ này, kiểm tra `exp`, và phạm vi quyền bắt buộc — trả về lời thách thức `WWW-Authenticate` ngay khi gặp lỗi đầu tiên. Việc duy trì thành một quy trình duy nhất trên máy chủ tài nguyên đảm bảo rằng mọi điểm vào (mọi lệnh gọi công cụ, mọi giao vận) đều đi qua các kiểm tra giống nhau; không có đường dẫn nào tiếp cận được công cụ mà không qua xác thực trước.

### Token mờ đục sử dụng cơ chế nội soi, không đoán mò

Không phải mọi access token đều là JWT. Nếu bên phát hành xuất bản token mờ đục (opaque token), máy chủ tài nguyên không thể tự giải mã nó thành các xác nhận đáng tin cậy. Nó gửi token tới điểm cuối nội soi RFC 7662 của bên phát hành qua một kênh phụ đã xác thực và yêu cầu `active: true`, ngữ cảnh bên phát hành mong đợi, đối tượng thụ hưởng hoặc tài nguyên MCP chính xác, các xác nhận thời gian chưa hết hạn, và các phạm vi quyền theo đòi hỏi của công cụ cụ thể.

Lưu đệm kết quả nội soi theo bên phát hành, mã băm một chiều của token, và tài nguyên MCP. Tuyệt đối không dùng token thô làm nhãn ghi log hoặc nhãn lưu đệm. Giới hạn mục lưu đệm tích cực theo giá trị sớm nhất trong số: thời điểm hết hạn của token, hướng dẫn lưu đệm của bên phát hành, và mục tiêu tươi mới thu hồi của hệ thống triển khai. Giữ thời gian lưu đệm tiêu cực đủ ngắn để một token mới phát hành không bị giữ ở trạng thái không hoạt động giả tạo. Kết quả cho một tài nguyên không thể dùng để ủy quyền cho một tài nguyên khác ngay cả khi chuỗi token mờ đục hoàn toàn giống nhau.

Không lựa chọn chế độ xác thực dựa trên nội dung token do kẻ tấn công kiểm soát. Hãy ghim chặt hành vi JWT hay nội soi theo siêu dữ liệu bên phát hành đã xác thực và cấu hình triển khai. Trên lộ trình JWT, hãy ghim các thuật toán được chấp nhận và `jwks_uri` đáng tin cậy; tuyệt đối không lần theo một URL khóa hay thuật toán chỉ được chỉ định bởi tiêu đề của token.

### Thu hồi là một bản hợp đồng về tính tươi mới

Chuẩn RFC 7009 cho phép client yêu cầu máy chủ ủy quyền thu hồi một token. Yêu cầu đó không tự động xóa sạch các bản sao đã được lưu đệm bởi mọi máy chủ tài nguyên. Hãy xác định độ trễ thu hồi tối đa có thể chấp nhận được và buộc mọi bộ nhớ đệm phải tuân thủ điều đó.

Các hệ thống triển khai token mờ đục có thể đạt được khả năng thu hồi chặt chẽ hơn bằng cách nội soi trên từng lệnh gọi rủi ro cao hoặc sử dụng bộ nhớ đệm tích cực ngắn hạn. Các hệ thống triển khai JWT tự chứa thường kết hợp vòng đời access token ngắn với việc thu hồi refresh token, thu hồi toàn bộ khóa cho các sự cố diện rộng của bên phát hành, và một danh sách từ chối (denylist) khẩn cấp tại chỗ cho chủ thể, phiên hoặc mã token. Một JWT có chữ ký duy trì tính hợp lệ về mặt mật mã học cho đến khi hết hạn trừ khi máy chủ tài nguyên có bằng chứng thu hồi bên ngoài cập nhật.

Đăng xuất, vô hiệu hóa tài khoản, rút lại sự đồng thuận, và phản ứng sự cố là các tác nhân kích hoạt khác nhau nhưng phải hội tụ về một cam kết có thể đo lường: sau tối đa khoảng thời gian thu hồi đã công bố, mọi bản sao đều phải từ chối thông tin xác thực. Hãy kiểm thử cam kết đó thông qua bộ cân bằng tải, chứ không chỉ trên một tiến trình đơn lẻ.

### Sự cố phụ thuộc đòi hỏi một quyết định được tuyên bố trước

Tuyệt đối không ứng biến chính sách sẵn sàng (availability policy) bên trong một khối xử lý ngoại lệ (exception handler).

| Sự cố | Hành vi sản xuất an toàn |
|---|---|
| Làm mới JWKS định kỳ thất bại, `kid` đã biết vẫn nằm trong bộ đệm có giới hạn còn hiệu lực | Chỉ tiếp tục trong cửa sổ cho phép dùng dữ liệu cũ khi có lỗi và phát ra cảnh báo suy giảm sức khỏe hệ thống |
| Token có `kid` chưa biết và lần làm mới duy nhất được phép bị thất bại | Từ chối ngay; tuyệt đối không chấp nhận một chữ ký không thể xác minh |
| Điểm cuối nội soi không khả dụng | Chặn hoàn toàn (fail closed) đối với các lệnh gọi được bảo vệ; không biến sự cố mạng thành `active: true` |
| Siêu dữ liệu tài nguyên được bảo vệ hoặc bên phát hành thay đổi bất thường | Dừng việc đăng ký mới và xin cấp token mới; chỉ giữ lại cấu hình đã ghim rõ ràng và chưa hết hạn trong một chính sách sự cố có giới hạn |
| Điểm cuối thu hồi không khả dụng | Báo cáo việc đăng xuất hoặc thu hồi là chưa hoàn tất, giữ thông tin xác thực tại chỗ ở trạng thái không thể dùng nếu có thể, và không tuyên bố thu hồi toàn cục thành công |
| Nguồn đồng hồ hoặc kiểu xác nhận không hợp lệ | Từ chối thay vì nới rộng dung sai độ lệch thời gian (skew) cho đến khi token vượt qua |

Phân loại các sự cố tách biệt khỏi các thông tin xác thực không hợp lệ. Một sự cố ngừng hoạt động của thành phần phụ thuộc là lỗi vận hành đi kèm chính sách thử lại và theo dõi sức khỏe. Một chữ ký, bên phát hành, đối tượng thụ hưởng, thời hạn hết hạn, hoặc phạm vi quyền bị sai là sự từ chối ủy quyền. Cả hai trường hợp đều không được chạm tới bộ xử lý công cụ, và không trường hợp nào được phép làm rò rỉ nội dung token vào bằng chứng kiểm toán.

### Phân tích kịch bản phát lại đối tượng thụ hưởng

Máy chủ A (`notes.example.com`) và Máy chủ B (`tasks.example.com`) cùng đăng ký với một máy chủ ủy quyền. Máy chủ A bị xâm nhập. Kẻ tấn công lấy token ghi chú của người dùng và phát lại nó để gọi Máy chủ B.

Bộ xác thực của Máy chủ B:

1. Giải mã JWT, nạp JWKS theo `kid`, xác minh chữ ký.
2. Kiểm tra `iss` đối chiếu với `authorization_servers` trong siêu dữ liệu tài nguyên được bảo vệ của nó. (Vượt qua — cùng IdP.)
3. Kiểm tra `aud == "https://tasks.example.com"`. (Thất bại — trường `aud` của token là `https://notes.example.com`.)
4. Trả về mã 401 kèm `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`.

Xác nhận đối tượng thụ hưởng (audience claim) là tuyến phòng thủ duy nhất chống lại cuộc tấn công này ở tầng giao thức. Việc bỏ qua nó vì lý do hiệu năng là lỗi sản xuất phổ biến nhất; bộ xác thực bắt buộc phải chạy trên mọi yêu cầu, chứ không chỉ riêng lúc bắt đầu phiên. Đặc tả gọi đây là **hạn chế đặc quyền của access token (access-token privilege restriction)**: máy chủ MCP `BẮT BUỘC` phải từ chối bất kỳ token nào không ghi tên nó trong đối tượng thụ hưởng.

> **Lưu ý thuật ngữ.** Đặc tả giữ thuật ngữ *đại diện bị nhầm lẫn (confused deputy)* cho một bài toán liên quan nhưng khác biệt: một máy chủ MCP đóng vai trò là một **proxy** OAuth tới một API bên thứ ba, sử dụng một client ID tĩnh, chuyển tiếp một token mà không xin sự đồng thuận của người dùng theo từng client. Ràng buộc đối tượng thụ hưởng giải quyết bài toán phát lại ở trên; biện pháp xử lý confused deputy là sự đồng thuận theo từng client **cộng với việc** không bao giờ chuyển tiếp token đến tới các API thượng nguồn (máy chủ MCP `BẮT BUỘC` phải lấy token thượng nguồn riêng biệt của chính nó).

### Tấn công tráo đổi (Một cơ chế phòng thủ phía client mà server không thể cung cấp)

Một client giao tiếp với nhiều máy chủ ủy quyền trong suốt vòng đời của nó. Một AS độc hại có thể cố gắng làm cho client đổi mã ủy quyền của một AS trung thực tại điểm cuối của kẻ tấn công. Ràng buộc đối tượng thụ hưởng không giúp ích gì ở đây — cuộc tấn công xảy ra trước khi bất kỳ token nào tồn tại. Cơ chế phòng thủ nằm ở phía client (RFC 9207):

1. Trước khi chuyển hướng, client ghi lại `issuer` kỳ vọng từ siêu dữ liệu AS đã được xác thực.
2. Trong phản hồi ủy quyền, client so sánh tham số `iss` trả về với bên phát hành đã ghi nhận (so sánh chuỗi đơn giản, không chuẩn hóa) trước khi gửi mã đi bất kỳ đâu.
3. Không khớp (hoặc thiếu `iss` khi AS đã công bố `authorization_response_iss_parameter_supported`) → từ chối ngay, và thậm chí không hiển thị các trường `error`.

Chỉ riêng PKCE không ngăn chặn được tấn công tráo đổi, vì client sẽ trao `code_verifier` của mình cho bất kỳ điểm cuối token nào mà nó bị điều hướng tới. Đây là lý do tại sao đặc tả yêu cầu ghi nhận bên phát hành trên từng yêu cầu cùng với mã xác minh PKCE và `state`.

### Các chế độ lỗi thường gặp

- **JWKS bị cũ.** Bộ xác thực từ chối các token hợp lệ sau khi AS xoay vòng khóa. Biện pháp xử lý là mô hình cron-refresh + cache-miss-refetch ở trên. Tuyệt đối không lưu đệm JWKS mà không có tác vụ làm mới.
- **Xoay vòng như một giải pháp dự phòng.** Việc nối lộ trình trượt bộ đệm vào thao tác xoay-và-đúc khóa thay vì nạp lại là một lỗi nghiêm trọng: nó không bao giờ sinh ra `kid` đang thiếu, và nó biến các giá trị `kid` do kẻ tấn công kiểm soát thành một cuộc tấn công DoS tạo khóa. Giải pháp dự phòng bắt buộc phải là thao tác nạp lại có tính lũy suy `refresh-jwks`.
- **Thiếu xác nhận `aud`.** Một số IdP mặc định bỏ qua `aud` trừ khi `resource` xuất hiện trong yêu cầu token. Bộ xác thực bắt buộc phải từ chối các token thiếu `aud`, không được coi sự vắng mặt đó là ký tự đại diện.
- **Bị tấn công tráo đổi do thiếu kiểm tra `iss`.** Một client không xác thực tham số phản hồi ủy quyền `iss` theo RFC 9207 so với bên phát hành đã ghi nhận có thể bị dẫn dụ đổi mã của một AS trung thực tại điểm cuối của kẻ tấn công. Đây là lỗi phía client; máy chủ tài nguyên không thể bù đắp được.
- **Xung đột nâng cấp phạm vi quyền (Scope upgrade race).** Hai luồng step-up đồng thời cho cùng một người dùng đều có thể thành công và sinh ra hai access token với các phạm vi quyền khác nhau. Bộ xác thực bắt buộc phải dùng token được xuất trình trong yêu cầu, chứ không tra cứu "phạm vi quyền hiện tại của người dùng" — điều đó tạo ra lỗ hổng TOCTOU (Time-of-Check to Time-of-Use).
- **Trộm cắp token đăng ký.** Một `registration_access_token` bị lộ cho phép kẻ tấn công viết lại các URI chuyển hướng. Hãy băm các token này khi lưu trữ; yêu cầu client xuất trình văn bản thô trên mọi lần cập nhật; xoay vòng ngay khi có nghi ngờ.
- **Không ghim `iss`.** Một bộ xác thực chấp nhận bất kỳ `iss` nào cho phép kẻ tấn công dựng máy chủ ủy quyền riêng, đăng ký một client cho đối tượng thụ hưởng mục tiêu, và tự phát hành token. Danh sách `authorization_servers` trong siêu dữ liệu tài nguyên được bảo vệ chính là danh sách cho phép; hãy bắt buộc thực thi nó.
- **Xung đột bộ nhớ đệm thông tin xác thực hoặc token.** Một client chỉ gắn khóa đăng ký theo tài nguyên có thể xuất trình danh tính của một máy chủ ủy quyền này cho một máy chủ khác. Một client chỉ gắn khóa access token theo bên phát hành có thể phát lại token sai đối tượng thụ hưởng. Hãy gắn khóa đăng ký theo bên phát hành đã xác thực, gắn khóa access token theo cặp `(bên phát hành, tài nguyên)`, và đăng ký lại bất cứ khi nào bên phát hành thay đổi.

````figure
t3-jwks-rotate
````

## Thử nghiệm thực tế

Tệp `code/main.py` thực hiện toàn bộ luồng sản xuất bằng thư viện chuẩn Python (stdlib) và ba vai trò: `AuthorizationServer`, `ResourceServer`, và `Client`.

Từ thư mục gốc của kho lưu trữ, chạy lệnh:

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Lệnh đầu tiên in ra bản ghi quá trình đăng ký ràng buộc bên phát hành và xác thực token. Lệnh thứ hai báo cáo mười tám bài kiểm tra đều vượt qua. Cả hai lệnh đều không mở cổng mạng lắng nghe và không ghi các thông tin xác thực ra đĩa.

1. Máy chủ ủy quyền xuất bản siêu dữ liệu RFC 8414 tại `/.well-known/oauth-authorization-server`.
2. Client MCP gọi điểm cuối siêu dữ liệu và kiểm tra các tùy chọn đăng ký (`client_id_metadata_document_supported` cho CIMD, `registration_endpoint` cho DCR) và khả năng hỗ trợ PKCE `S256`.
3. Client kiểm tra bản đăng ký trước theo phạm vi bên phát hành, nếu không sẽ đăng ký bằng Tài liệu Siêu dữ liệu Client ID chạy qua HTTPS của nó. DCR đã lỗi thời vẫn là một phương thức tương thích có thể kiểm thử riêng biệt.
4. Client ghi lại bên phát hành đã xác thực, tạo thử thách S256, nhận mã ủy quyền dùng một lần cùng tham số `iss`, xác thực bên phát hành trả về đó, và đổi mã lấy token với mã xác minh ban đầu và chỉ báo `resource` theo RFC 8707.
5. Client MCP gọi một công cụ trên máy chủ MCP với tiêu đề `Authorization: Bearer ...`.
6. Máy chủ MCP chạy hàm `validate`, phân giải khóa ký từ bộ nhớ đệm JWKS.
7. IdP xoay vòng một khóa; tác vụ làm mới theo lịch kéo lại JWKS vào bộ đệm.
8. Lệnh gọi tiếp theo xác thực thành công với các khóa đã được làm mới mà không cần khởi động lại, và token trước đó vẫn xác thực được trong suốt cửa sổ gối đầu.
9. Một nỗ lực phát lại đối tượng thụ hưởng nhắm vào một tài nguyên MCP khác sẽ nhận mã lỗi 401 với thông báo `audience mismatch` và con trỏ `resource_metadata`.

JWT ở đây sử dụng thuật toán HS256 với một khóa bí mật dùng chung (nhờ đó bài học chạy hoàn toàn trên stdlib). Môi trường sản xuất sử dụng RS256 hoặc EdDSA với mô hình JWKS ở trên; logic xác thực hoàn toàn tương tự. Vì IdP và máy chủ tài nguyên cùng nằm trong một tiến trình, hàm `refresh_jwks` đọc trực tiếp danh sách khóa của máy chủ ủy quyền; trên đường truyền mạng thực tế, đây là một lệnh HTTP `GET` tới `jwks_uri`.

## Đóng gói sản phẩm

Bài học này đóng gói tệp `outputs/skill-mcp-auth.md`. Dựa trên cấu hình máy chủ MCP và tập năng lực của IdP, kỹ năng này sẽ phát ra bề mặt xác thực và ủy quyền cần thiết lập — siêu dữ liệu tài nguyên được bảo vệ, lộ trình đăng ký cần dùng (CIMD, đăng ký trước, hoặc dự phòng qua DCR), lịch trình làm mới JWKS, ánh xạ phạm vi quyền, và các quy tắc từ chối áp dụng khi IdP không hỗ trợ đầy đủ hồ sơ RFC.

## Bài tập mở rộng

1. Chạy `code/main.py`. Lần theo luồng thực thi. Lưu ý cách IdP xoay vòng khóa ở bước 6 của kết quả script (bước 7 của luồng ở trên), tác vụ `refresh_jwks` theo lịch kéo lại tập khóa đã xuất bản, và cả token cũ (cửa sổ gối đầu) lẫn token mới đều xác thực thành công mà không cần khởi động lại tiến trình.
2. Thêm một IdP mới vào danh sách `authorization_servers` của siêu dữ liệu tài nguyên được bảo vệ. Phát hành một token được ký bởi IdP mới và xác nhận bộ xác thực chấp nhận nó. Phát hành một token được ký bởi một IdP không có trong danh sách và xác nhận bộ xác thực từ chối với `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.
3. Bổ sung kiểm tra giới hạn tốc độ vào `register_client` chạy trước khi cơ quan đăng ký chấp nhận yêu cầu. Sử dụng thuật toán thùng chứa token (token-bucket) cho mỗi IP nguồn lưu trong một dict nhỏ theo IP.
4. Đọc RFC 7591 và xác định hai trường mà bộ xử lý `/register` của bài học chưa xác thực. Bổ sung việc xác thực đó. (Gợi ý: `software_statement` và lược đồ URI của `redirect_uris`.)
5. Thêm một máy chủ ủy quyền thứ hai. Xác nhận rằng client lưu trữ một bản đăng ký riêng biệt có khóa theo bên phát hành và từ chối tái sử dụng token hoặc `client_id` của bên phát hành thứ nhất.
6. Chứng minh giải pháp chống tấn công DoS. Gửi tới bộ xác thực một token với `kid` ngẫu nhiên và xác nhận rằng `refresh_jwks` chỉ chạy tối đa một lần và số lượng khóa của máy chủ ủy quyền không bị tăng lên. Sau đó cố tình sửa lại phương án dự phòng thành xoay-và-đúc khóa và quan sát số lượng khóa tăng vọt theo từng token giả mạo — khôi phục lại cơ chế nạp lại sau đó.
7. Thử nghiệm DCR đã lỗi thời với cả client kiểu `native` và `web`. Xác nhận rằng một web client có URI chuyển hướng HTTP và một native client không có chuyển hướng loopback chính xác đều bị từ chối.

## Các thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| ASM | "Tài liệu siêu dữ liệu OAuth" | Tệp JSON RFC 8414 `/.well-known/oauth-authorization-server` |
| CIMD | "URL siêu dữ liệu client" | Tài liệu Siêu dữ liệu Client ID: một URL HTTPS dùng làm `client_id`; AS kéo JSON về. Cơ chế đăng ký ưu tiên trong MCP 2026-07-28 |
| DCR | "Đăng ký client tự phục vụ" | Thao tác `POST /register` theo RFC 7591; đã bị bãi bỏ trong MCP hiện tại và chỉ giữ lại để tương thích |
| JWKS | "Các khóa công khai để xác thực JWT" | Tập khóa Web JSON, nạp từ `jwks_uri`, đánh chỉ mục theo `kid` |
| Xoay vòng vs Làm mới | "Cập nhật khóa" | *Xoay vòng (Rotate)* = AS đúc/thu hồi khóa ký; *Làm mới (Refresh)* = máy chủ tài nguyên nạp lại tập khóa đã xuất bản. Máy chủ tài nguyên chỉ thực hiện làm mới |
| Chỉ báo tài nguyên (Resource indicator) | "Tham số đối tượng thụ hưởng" | Tham số `resource` RFC 8707 ghim token vào một máy chủ duy nhất |
| Xác nhận `aud` | "Đối tượng thụ hưởng (Audience)" | Trường xác nhận JWT mà bộ xác thực so sánh với URL tài nguyên chuẩn tắc |
| Phát lại đối tượng thụ hưởng | "Phát lại token" | Token cấp cho Máy chủ A bị xuất trình cho Máy chủ B; phòng thủ bằng xác thực đối tượng thụ hưởng (hạn chế đặc quyền access token) |
| Đại diện bị nhầm lẫn (Confused deputy) | "Lạm dụng token qua proxy" | Một proxy MCP với client ID tĩnh chuyển tiếp token mà không có sự đồng thuận theo từng client; khác với phát lại đối tượng thụ hưởng |
| Tấn công tráo đổi (Mix-up attack) | "Sai điểm cuối token" | Client bị điều hướng đổi mã của một AS trung thực tại điểm cuối của kẻ tấn công; phòng thủ phía client qua `iss` theo RFC 9207 |
| Danh sách cho phép `iss` | "Các máy chủ ủy quyền tin cậy" | Tập hợp được nêu tên trong trường `authorization_servers` của siêu dữ liệu tài nguyên được bảo vệ |
| `resource_metadata` | "Nơi tìm tài liệu PRM" | Tham số `WWW-Authenticate` chỉ định URL siêu dữ liệu RFC 9728 khi gặp lỗi 401/403 |
| Client công khai (Public client) | "Client native hoặc trình duyệt" | Client OAuth không có `client_secret`; PKCE bù đắp cho điều này |
| `WWW-Authenticate` | "Tiêu đề phản hồi 401/403" | Mang các chỉ thị `Bearer error=...` giúp client thực hiện quy trình khôi phục |

## Đọc thêm

- [Đặc tả ủy quyền MCP (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - hồ sơ ủy quyền MCP hiện hành
- [Nhật ký thay đổi MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - các thay đổi về CIMD, xác thực bên phát hành, bãi bỏ DCR, và thông tin xác thực gắn khóa bên phát hành
- [Tài liệu Siêu dữ liệu Client ID OAuth (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — hợp đồng khám phá
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR (lộ trình dự phòng)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — bằng chứng sở hữu dành cho client công khai
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — ghim đối tượng thụ hưởng
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — khám phá máy chủ tài nguyên
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — tham số `iss` phòng thủ chống tấn công tráo đổi
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
