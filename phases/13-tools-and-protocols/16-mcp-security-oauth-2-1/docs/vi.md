# Ủy quyền MCP: CIMD, Ràng buộc bên phát hành, PKCE và Nâng cấp phạm vi

> Một yêu cầu MCP từ xa mang tính phi trạng thái, nhưng việc ủy quyền của nó không hề ẩn danh. Hãy ràng buộc mọi thông tin xác thực vào bên phát hành đã tạo ra nó, và ràng buộc mọi token vào chính tài nguyên sẽ tiếp nhận nó.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python
**Yêu cầu tiên quyết:** Phase 13 · 09 (giao vận), Phase 13 · 15 (bảo mật)
**Thời lượng:** ~90 phút

## Mục tiêu học tập

- Khám phá các máy chủ ủy quyền (authorization servers) thông qua siêu dữ liệu tài nguyên được bảo vệ (protected-resource metadata).
- Ưu tiên Tài liệu Siêu dữ liệu Client ID (Client ID Metadata Documents - CIMD) thay vì Đăng ký Client Động (Dynamic Client Registration - DCR) vốn đã bị coi là lỗi thời.
- Khai báo đúng `application_type` khi bắt buộc phải sử dụng lộ trình tương thích DCR.
- Xác thực tham số `iss` trong phản hồi ủy quyền và cô lập thông tin xác thực theo từng bên phát hành (issuer).
- Sử dụng PKCE, chỉ báo tài nguyên (resource indicators), xác thực đối tượng thụ hưởng (audience), và các phạm vi quyền hạn tăng dần (incremental scopes).
- Gửi các yêu cầu MCP chuẩn 2026-07-28 đã được ủy quyền mà không cần đến các phiên giao thức.

## Vấn đề đặt ra

Một máy chủ MCP từ xa có thể đọc các bản ghi riêng tư, ghi dữ liệu vào các hệ thống bên ngoài, hoặc kích hoạt các tác vụ tốn kém. Xác thực (authentication) cho biết ai đang xuất trình thông tin xác thực. Nhưng ủy quyền (authorization) còn phải trả lời các câu hỏi:

- Máy chủ ủy quyền nào đã phát hành thông tin xác thực đó?
- Token này được cấp cho tài nguyên MCP cụ thể nào?
- Client và URI chuyển hướng (redirect URI) nào đã hoàn tất luồng xác thực?
- Người dùng đã phê duyệt những thao tác cụ thể nào?
- Yêu cầu chính xác này có còn phù hợp với sự phê duyệt đó hay không?

Hồ sơ ủy quyền chuẩn 2026-07-28 siết chặt việc đăng ký client và xử lý bên phát hành. Nó ưu tiên các Tài liệu Siêu dữ liệu Client ID, bãi bỏ Đăng ký Client Động, yêu cầu khai báo đúng `application_type` trên DCR, xác thực phản hồi bên phát hành theo RFC 9207, và nghiêm cấm việc tái sử dụng thông tin xác thực giữa các bên phát hành khác nhau.

Các quy tắc này bổ trợ cho tầng lõi phi trạng thái. Chúng tuyệt đối không khôi phục lại quy trình bắt tay giao thức lõi hay tiêu đề `Mcp-Session-Id`.

## Khái niệm cốt lõi

### Nắm vững ba vai trò

- **MCP client:** gửi các yêu cầu thay mặt cho chủ sở hữu tài nguyên (resource owner).
- **MCP resource server (Máy chủ tài nguyên MCP):** tiếp nhận access token và phục vụ điểm cuối MCP.
- **Authorization server (Máy chủ ủy quyền):** xác thực chủ sở hữu tài nguyên, thu thập sự đồng thuận và phát hành token.

Máy chủ tài nguyên và máy chủ ủy quyền có thể được vận hành cùng nhau, nhưng phải phân định rạch ròi các định danh và trách nhiệm xác thực của chúng.

### Ủy quyền áp dụng cho tầng HTTP

Đặc tả ủy quyền MCP áp dụng cho các tầng giao vận dựa trên HTTP. Một máy chủ chạy qua giao thức stdio cục bộ hoạt động dưới ranh giới tin cậy của tiến trình và hệ điều hành. Tuyệt đối không thêm luồng OAuth trình duyệt giả tạo vào stdio chỉ vì mục đích đối xứng cấu trúc.

Đối với giao vận Streamable HTTP từ xa, hãy gửi token người mang (bearer token) trong tiêu đề `Authorization` ở mỗi yêu cầu. Tuyệt đối không đặt token vào URL.

### Bắt đầu với siêu dữ liệu tài nguyên được bảo vệ

Máy chủ tài nguyên công bố tài liệu siêu dữ liệu theo chuẩn RFC 9728:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

Client bắt đầu từ URL tài nguyên MCP, nạp tài liệu này, chọn một máy chủ ủy quyền được công bố, và sau đó nạp siêu dữ liệu OAuth hoặc OpenID Connect của máy chủ đó.

Hãy bảo toàn đường dẫn tài nguyên khi xây dựng URL well-known theo RFC 9728. Đối với tài nguyên `https://notes.example.com/mcp`, bài học này sử dụng `https://notes.example.com/.well-known/oauth-protected-resource/mcp`. Việc bỏ rơi phần đuôi `/mcp` có thể dẫn đến việc chọn nhầm siêu dữ liệu của một tài nguyên được bảo vệ khác trên cùng một nguồn gốc (origin).

Không đoán máy chủ ủy quyền từ tên máy chủ (hostname). Không tuân theo một bên phát hành được khám phá từ một thân thông báo lỗi chưa được xác thực. Hãy duy trì một chính sách rõ ràng về những bên phát hành mà client sẵn sàng tin cậy.

### Xác minh siêu dữ liệu của máy chủ ủy quyền

Siêu dữ liệu phải phơi bày các điểm cuối và các cơ chế kiểm soát được hỗ trợ:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

Bắt buộc phải yêu cầu S256 cho PKCE. Ghi lại chính xác chuỗi định danh bên phát hành (issuer). Giá trị chính xác đó sẽ trở thành khóa dùng cho việc đăng ký và lưu trữ token.

### Tuân thủ thứ tự ưu tiên đăng ký

Sử dụng thông tin client đã được đăng ký trước (pre-registered) khi client đã có mối quan hệ tường minh với bên phát hành được chọn. Nếu không, hãy ưu tiên các Tài liệu Siêu dữ liệu Client ID khi máy chủ ủy quyền công bố hỗ trợ. Chỉ sử dụng DCR như một giải pháp dự phòng tương thích đã bị coi là lỗi thời, sau đó mới nhắc người dùng nhập thông tin client nếu không có cơ chế nào khả dụng.

### Ưu tiên Tài liệu Siêu dữ liệu Client ID (CIMD)

Một Tài liệu Siêu dữ liệu Client ID cung cấp cho máy chủ ủy quyền một URL HTTPS vừa là định danh của client, vừa là vị trí lưu trữ siêu dữ liệu của nó:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

Máy chủ ủy quyền nạp và xác thực tài liệu này. `client_id` bắt buộc phải là một URL HTTPS có đường dẫn, và giá trị bên trong tài liệu phải trùng khớp hoàn toàn với URL đó. Các trường bắt buộc trong tài liệu là `client_id`, `client_name`, và `redirect_uris`. Trường `application_type` xuất hiện trong ví dụ này nhưng không phải là yêu cầu bắt buộc của CIMD; trường này hiện bắt buộc áp dụng riêng cho lộ trình DCR.

Hãy xử lý thao tác nạp tài liệu này như một tác vụ nhạy cảm với tấn công SSRF (Server-Side Request Forgery). Phân giải và xác thực đích đến, từ chối các địa chỉ loopback, mạng riêng tư, link-local, và các địa chỉ bị cấm khác; kiểm tra lại sau khi chuyển hướng và khi có thay đổi DNS; giới hạn số lần chuyển hướng, số byte và thời gian chờ; yêu cầu kiểu dữ liệu JSON; và chỉ lưu đệm theo đúng các tiêu đề kiểm soát bộ nhớ đệm HTTP đã được xác thực. Xử lý `client_name` và các trường hiển thị khác như văn bản không đáng tin cậy.

CIMD giúp loại bỏ nhu cầu phải phát hành một định danh động mới cho mỗi lần tiếp xúc đầu tiên. Nó không loại bỏ việc xác thực URI chuyển hướng, chính sách bên phát hành, hay sự đồng thuận của người dùng.

### DCR là một lộ trình tương thích kế thừa

Đăng ký Client Động (DCR) vẫn khả dụng đối với các máy chủ ủy quyền cũ hơn, nhưng đã bị bãi bỏ đối với các bản triển khai MCP mới.

Khi sử dụng DCR, bắt buộc phải khai báo `application_type`:

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- Các ứng dụng máy tính để bàn (desktop), di động, giao diện dòng lệnh (CLI), và client loopback sử dụng `native`.
- Các ứng dụng trình duyệt được lưu trữ từ xa sử dụng `web` và các URI chuyển hướng HTTPS từ xa.

Việc bỏ sót trường này có thể khiến hệ thống OpenID Connect mặc định áp dụng kiểu `web` và làm cho lệnh chuyển hướng loopback hợp lệ bị từ chối thất bại.

Giữ mã DCR phía sau một quyết định chuyển tiếp (fallback) tường minh. Không âm thầm chuyển sang DCR sau một lỗi xác thực CIMD bất kỳ, vì điều đó có thể biến một lỗi bảo mật thành một lộ trình đăng ký yếu hơn.

### Ràng buộc thông tin xác thực vào bên phát hành

Lưu trữ tài liệu đăng ký do bên phát hành tạo ra dưới khóa là chính bên phát hành đó:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

Nếu việc khám phá tài nguyên được bảo vệ chuyển từ `https://auth-one.example` sang `https://auth-two.example`, hãy đánh giá lại độ tin cậy. Tuyệt đối không gửi client secret, DCR client id, token truy cập đăng ký, refresh token hoặc access token của bên phát hành thứ nhất cho bên thứ hai. Các client đăng ký trước và client DCR bắt buộc phải sử dụng các thông tin xác thực được phát hành riêng cho bên phát hành mới.

URL client id của CIMD có điểm khác biệt vì đây là một URL HTTPS tự lưu trữ, không phải thông tin xác thực do máy chủ ủy quyền sinh ra. Cùng một URL CIMD có thể mang tính khả chuyển: một bên phát hành tin cậy mới có thể nạp và xác thực tài liệu đó mà không cần đăng ký lại qua DCR. Tuy nhiên, các phản hồi ủy quyền và token vẫn phải được xác thực và lưu trữ riêng biệt theo bên phát hành mới.

### Mã ủy quyền kết hợp PKCE

Luồng tương tác diễn ra như sau:

1. Sinh ra một `code_verifier` có độ hỗn loạn (entropy) cao.
2. Tính toán `code_challenge` theo thuật toán S256.
3. Gửi yêu cầu ủy quyền kèm chính xác các trường `client_id`, `redirect_uri`, `scope`, `code_challenge`, và `resource`.
4. Nhận phản hồi ủy quyền chứa `code` và tham số `iss` (nếu có).
5. Xác thực `iss` so với bên phát hành chính xác đã ghi nhận trước khi sử dụng bất kỳ trường nào trong phản hồi.
6. Đổi mã lấy token bằng `code_verifier`, cùng URI chuyển hướng và cùng `resource`.
7. Lưu trữ token thu được dưới khóa `(issuer, resource)`.

Tham số `resource` từ RFC 8707 xuất hiện trong cả yêu cầu ủy quyền lẫn yêu cầu lấy token. Nó định danh URI chuẩn tắc của máy chủ MCP.

### Xác thực tham số `iss` chính xác tuyệt đối

Chuẩn RFC 9207 ngăn chặn việc phản hồi ủy quyền từ một bên phát hành bị nhầm lẫn với phản hồi từ một bên phát hành khác.

Khi `iss` xuất hiện, hãy so sánh nó với bên phát hành đã ghi nhận mà không chuyển đổi chữ hoa chữ thường, không thay đổi dấu gạch chéo cuối, không loại bỏ cổng mặc định, và không chuẩn hóa mã hóa phần trăm. Nếu không khớp, không sử dụng mã ủy quyền và thậm chí không hiển thị các thông tin chi tiết về lỗi do kẻ tấn công kiểm soát từ phản hồi đó.

Một máy chủ ủy quyền có trả về `iss` sẽ công bố `authorization_response_iss_parameter_supported: true`. Các client hiện đại vẫn sẽ xác thực `iss` khi nó hiện diện ngay cả khi máy chủ không công bố cờ này.

### Xác thực đối tượng thụ hưởng tại máy chủ MCP

Máy chủ tài nguyên chỉ chấp nhận các token được phát hành cho chính bản thân nó:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

Các token không hợp lệ, hết hạn, sai bên phát hành, hoặc sai đối tượng thụ hưởng đều phải nhận mã lỗi 401. Máy chủ MCP tuyệt đối không chấp nhận hoặc chuyển tiếp một token vốn được cấp cho một dịch vụ khác.

### Yêu cầu phạm vi quyền hạn tối thiểu hiện thời

Bắt đầu với phạm vi quyền hạn vừa đủ cho nhu cầu hiện tại. Nếu một công cụ sau đó đòi hỏi nhiều quyền hơn, máy chủ sẽ trả về lỗi 403 kèm theo thách thức phạm vi quyền có thẩm quyền (scope challenge):

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

Client giải thích quyền hạn mới cho người dùng, xin sự đồng thuận, thực hiện một luồng ủy quyền mới với tập hợp phạm vi quyền đã kết hợp, và thử lại yêu cầu MCP với một id JSON-RPC mới.

Đừng giả định rằng phạm vi quyền bị thách thức luôn là tập con của `scopes_supported`. Thách thức trả về là thẩm quyền tối cao đối với thao tác hiện tại.

### Ủy quyền và đường truyền MCP phi trạng thái

Một lệnh gọi công cụ đã được ủy quyền vẫn mang phong bì yêu cầu hiện hành hoàn chỉnh:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Token ủy quyền cho chủ thể. Siêu dữ liệu yêu cầu đàm phán hành vi giao thức. Cả hai không thể thay thế cho nhau.

Xác thực đường truyền theo một thứ tự cố định: kiểu JSON-RPC và kiểu siêu dữ liệu, sự trùng khớp giữa tiêu đề và phần thân, sau đó là sự hỗ trợ của giao thức. Bất kỳ sự sai lệch nào về tiêu đề định tuyến hoặc tiêu đề phiên bản đều trả về HTTP 400 với `-32020`. Nếu tiêu đề và thân thống nhất về một phiên bản không được hỗ trợ, trả về HTTP 400 với `-32022` và `data` chính xác là `{"supported":["2026-07-28"],"requested":"<actual>"}`. Phương thức không xác định trả về HTTP 404 với `-32601`.

Mọi lỗi yêu cầu, bao gồm lỗi 401 token không hợp lệ và lỗi 403 không đủ phạm vi quyền, đều là phong bì lỗi JSON-RPC mang đúng `id` của yêu cầu ban đầu. Thông tin phục hồi có cấu trúc thuộc về trường `data` tùy chọn của lỗi; `WWW-Authenticate` vẫn là tiêu đề phản hồi HTTP. Một thông báo không có `id`, do đó nó không nhận được thân JSON-RPC. Một thông báo HTTP được chấp nhận sẽ trả về 202 với thân rỗng.

Máy chủ triển khai `server/discover` và công bố công cụ, do đó nó cũng phải triển khai phương thức bắt buộc `tools/list`. Các bộ mô tả công cụ có tên ổn định, phần mô tả và giá trị `inputSchema` có gốc là đối tượng. Danh sách này mang tính tất định và trả về `resultType`, siêu dữ liệu định danh máy chủ, thời gian `ttlMs` có giới hạn và `cacheScope`. Việc khám phá và danh sách công cụ không phụ thuộc người dùng có thể hiển thị trước khi ủy quyền. Hãy áp dụng chính sách thông thường và lưu đệm riêng tư (private caching) nếu có bất kỳ yếu tố nào thay đổi theo chủ thể.

### Không chuyển tiếp token (No Token Passthrough)

Một máy chủ MCP tuyệt đối không được chuyển tiếp access token MCP của client tới một API hạ nguồn. Hãy lấy một token hạ nguồn riêng biệt với đối tượng thụ hưởng phù hợp hoặc sử dụng thiết kế trao đổi token (token-exchange) rõ ràng. Việc xác thực đối tượng thụ hưởng chỉ phát huy tác dụng khi các dịch vụ kiên quyết từ chối các token được đúc cho đơn vị khác.

### Refresh token

Refresh token là tùy chọn. Khi được phát hành, hãy lưu trữ chúng bảo mật và gắn khóa theo bên phát hành và tài nguyên. Đừng mặc định rằng chúng luôn tồn tại. Hãy xoay vòng (rotate) chúng khi máy chủ ủy quyền hỗ trợ xoay vòng và phát hiện kịp thời việc tái sử dụng các giá trị đã bị vô hiệu hóa.

````figure
t3-scope-stepup
````

## Xây dựng hệ thống

Tệp `code/main.py` là một trình mô phỏng giao thức và ủy quyền chạy nội tiến trình. Nó triển khai quy trình khám phá tài nguyên được bảo vệ, siêu dữ liệu máy chủ ủy quyền, đăng ký CIMD, chuyển tiếp DCR theo phiên bản, kiểm tra kiểu ứng dụng, PKCE, xác thực bên phát hành, token ràng buộc tài nguyên, nâng cấp phạm vi quyền (scope step-up), `server/discover`, `tools/list`, và một yêu cầu công cụ phi trạng thái.

Mô hình tiếp nhận các phần thân yêu cầu và tiêu đề định tuyến đã được phân tích cú pháp. Nó không phải là một bộ điều hợp HTTP hoàn chỉnh và không trực tiếp phân tích `Content-Type` hay `Accept`. Hãy kết nối nó với bộ điều hợp Streamable HTTP của Bài học 09, vốn yêu cầu `Content-Type: application/json` và giá trị `Accept` chứa cả `application/json` lẫn `text/event-stream`.

Chạy chương trình:

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Kết quả đầu ra hiển thị lần lượt: bước khám phá đầu tiên, đăng ký CIMD, một thao tác đọc bình thường, hai lần nâng cấp phạm vi quyền riêng biệt, và việc lưu trữ thông tin xác thực theo khóa bên phát hành.

## Thử nghiệm thực tế

Ánh xạ các đối tượng trong trình mô phỏng vào các thành phần thực tế:

- `ResourceServer.protected_resource_metadata` trở thành điểm cuối RFC 9728.
- `AuthorizationServer.metadata` trở thành điểm phát hiện RFC 8414 hoặc OpenID Connect.
- `Client.enroll` trở thành cơ chế phân giải CIMD kèm nhánh tương thích DCR tường minh.
- Thông tin client do bên phát hành tạo ra và bảng `tokens_by_issuer_resource` trở thành các bản ghi được mã hóa. URL CIMD có thể duy trì tính khả chuyển trong khi các kết quả ủy quyền của nó vẫn bị ràng buộc theo bên phát hành.
- `ResourceServer.handle` trở thành phần mềm trung gian (middleware) xác thực các tiêu đề MCP hiện tại, token và phạm vi công cụ trước khi điều phối, đồng thời giữ mọi lỗi yêu cầu bên trong phong bì JSON-RPC tương ứng.

## Đóng gói sản phẩm

Bài học này đóng gói tệp `outputs/skill-oauth-scope-planner.md`. Tài liệu này giúp thiết kế thứ tự ưu tiên đăng ký, lưu trữ thông tin xác thực gắn với bên phát hành, kiểu ứng dụng, PKCE, chỉ báo tài nguyên, các thách thức phạm vi quyền, và ranh giới yêu cầu phi trạng thái hiện tại.

## Bài tập mở rộng

1. Bổ sung cơ chế xoay vòng refresh-token và từ chối việc tái sử dụng refresh token trước đó.
2. Thêm một danh sách cho phép bên phát hành (issuer allowlist). Khi đổi bên phát hành, chỉ tái sử dụng URL CIMD khả chuyển; từ chối tất cả các thông tin xác thực và token do bên phát hành trước đó cấp.
3. Thêm thời gian hết hạn cho mã ủy quyền và xác nhận rằng việc đổi mã muộn sẽ thất bại.
4. Xây dựng một biến thể client nền web với lệnh chuyển hướng HTTPS từ xa và so sánh siêu dữ liệu DCR của nó với client dạng native.
5. Thêm một tài nguyên thứ hai dưới cùng một bên phát hành. Xác nhận rằng access token của nó không thể sử dụng được tại tài nguyên thứ nhất.

## Các thuật ngữ then chốt

| Thuật ngữ | Ý nghĩa |
|---|---|
| Siêu dữ liệu tài nguyên được bảo vệ (Protected-resource metadata) | Tài liệu RFC 9728 định danh tài nguyên và các máy chủ ủy quyền |
| CIMD | Tài liệu siêu dữ liệu HTTPS có URL đóng vai trò là định danh client OAuth |
| DCR | Cơ chế đăng ký client động đã bị bãi bỏ, được giữ lại để tương thích |
| `application_type` | `native` hoặc `web`, dùng để xác thực các quy tắc URI chuyển hướng |
| PKCE | Khóa xác thực và mã thử thách S256 giúp bảo vệ mã ủy quyền bị chặn bắt |
| `iss` | Định danh bên phát hành phản hồi ủy quyền theo chuẩn RFC 9207 |
| Chỉ báo tài nguyên (Resource indicator) | Tham số RFC 8707 ràng buộc yêu cầu cấp token vào một tài nguyên MCP |
| Đối tượng thụ hưởng (Audience) | Tài nguyên cụ thể mà một token có hiệu lực |
| Nâng cấp phạm vi (Step-up) | Sự đồng thuận mới và việc phát hành token cho một phạm vi bổ sung của thao tác hiện tại |
| Thông tin xác thực ràng buộc bên phát hành (Issuer-bound credentials) | Các bản ghi đăng ký và token được cô lập theo chính xác bên phát hành của máy chủ ủy quyền |

## Đọc thêm

- [Đặc tả ủy quyền MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [Bản thảo OAuth Client ID Metadata Document](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
