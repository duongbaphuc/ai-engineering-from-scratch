# Dựng Lại và Chứng Minh Tính Tương Đương: Kiểm Thử Sai Biệt, So Sánh Byte và Vòng Lặp Bằng Chứng

> Việc vượt qua các bài kiểm thử của bạn chỉ chứng minh sự đồng thuận trên chính các bài kiểm thử đó; do vậy, tính tương đương đòi hỏi một lưới thử nghiệm chạm tới mọi ranh giới hiểm hóc cùng các byte nhị phân trùng khớp sau khi đã che đi trường tái định vị.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 34 (lifting to pseudocode), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~110 phút

## Mục tiêu Học tập

- Viết một bản dựng lại (rebuild) từ danh sách lệnh mã máy và nêu rõ bản ghi bằng chứng gốc mà nó bắt nguồn.
- Xây dựng một lưới dữ liệu đầu vào với mỗi lớp tương ứng với một ranh giới hiểm hóc: lính canh kiểm tra số 0 (zero guard), các giá trị âm, phép chia cắt cụt (truncating division), tràn số 32-bit (32-bit overflow), và bẫy phép chia (divide trap).
- Thực thi đồng thời cả hàm gốc lẫn hàm dựng lại trong cùng một chương trình và ghi lại kết quả của từng ô kiểm thử, bao gồm cả các trường hợp kích hoạt bẫy tín hiệu.
- Biên dịch lại bản dựng lại với cờ biên dịch của bản gốc và so sánh trực tiếp các byte của hàm sau khi đã che các trường tái định vị (relocation masking).
- Đọc các hằng số mà một bản ghi tái định vị trỏ tới và so sánh chúng, đảm bảo một hằng số sai lệch không thể vượt qua bài kiểm tra.
- Điều khiển máy chủ MCP của rea từ một client stdio tinh gọn và ghi lại từng bước kèm mã định danh bằng chứng của nó.
- Phát hành một bản ghi chứng minh theo nguyên tắc an toàn đóng (fail-closed proof record) có mã định danh là bản băm nội dung của các khẳng định và liên kết bằng chứng.

## Vấn đề Cần Giải Quyết

Bạn yêu cầu rea cung cấp hồ sơ hàm (function dossier) của `percent_change` trong tệp fixture `ledger-x86_64-O1.o`. Ghidra 12.1.4 trả về mã giả, và AI agent đã viết lại đoạn mã C sau từ kết quả đó:

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (int)(((long)after - before) * 100 / before);
}
```

Bốn ca kiểm thử ban đầu đều vượt qua: `(100, 150)` ra 50, `(200, 100)` ra -50, `(0, 5)` ra 0, và `(50, 100)` ra 100. Agent cũng dựng lại hàm `grade` dưới dạng bốn câu lệnh `if` kiểm tra các ngưỡng 90, 80, 70, và 60. Sáu bài kiểm thử tiếp theo tiếp tục vượt qua. Báo cáo tuyên bố trạng thái: "verified" (đã xác minh).

Tuy nhiên, báo cáo đó hoàn toàn bỏ ngỏ các câu hỏi trọng yếu:

- Hàm gốc trả về giá trị gì khi biểu thức `(after - before) * 100` bị tràn số và không vừa vặn trong 32 bit?
- Hàm gốc trả về kết quả gì đối với một điểm số là 110?
- Bản dựng lại có biên dịch ra đúng 16 byte mã máy y hệt như bản gốc hay không?
- Khi hai hàm đã biên dịch chứa các byte mã máy giống nhau, liệu các bản ghi tái định vị của chúng có trỏ tới cùng các giá trị hằng số hay không?
- Bản dựng lại bắt nguồn từ bản ghi bằng chứng cụ thể nào, và agent đã đọc danh sách mã máy hay đọc mã giả của trình dịch ngược?
- Bản ghi chứng minh sẽ nhận trạng thái gì trên một máy tính không cài đặt sẵn trình biên dịch C?

Mọi đối tượng mục tiêu trong bài học này đều là các chương trình do chính chúng ta viết và biên dịch. Phạm vi phân tích và các quy tắc pháp lý từ bài học 32 được áp dụng nguyên vẹn.

````figure re-36-prove-equal
````

## Dữ Liệu Bằng Chứng Thực Tế

Vòng lặp agent trong `code/main.py` đã thực hiện 6 lời gọi tới rea 6.5.0 qua giao tiếp luồng vào/ra chuẩn (stdio). Phiên chạy đã ghi nhận được lưu trữ tại `code/fixtures/runs/agent-run.json`. Máy chủ phản hồi lệnh `initialize` với phiên bản giao thức `2025-11-25`. Ba lời gọi đã trả về các bản ghi Evidence chính thức:

| Bước | Công cụ gọi | Thời gian | Mã định danh bằng chứng (Evidence id) |
|---|---|---|---|
| 2 | `analyze_function percent_change` | 6,31 s | `ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd` |
| 3 | `analyze_function grade` | 0,02 s | `ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1` |
| 4 | `read_bytes 0x1000b4, 20` | 0,00 s | `ev_c26146cfedf4c1ee146d8ff00cfa9a5d3f5335bdd9c97ccf4e1af5f6ea3ed6bc` |

Lời gọi đầu tiên chịu chi phí thời gian cho việc nhập Ghidra và phân tích tự động. Mỗi bản ghi đều mang thuộc tính `confidence: "observed"`, `authority: "shipped-artifact"`, và mã SHA-256 của tệp đối tượng là `9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f`. Ghidra nạp tệp đối tượng có thể tái định vị với phân đoạn `.text` tại `0x100000`, vì vậy `percent_change` nằm tại `0x100000` và `grade` nằm tại `0x100010`.

Đoạn mã giả của hàm `percent_change` chính là nơi kiểu `long` xuất hiện:

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{
  long lVar1;
  undefined1 auVar3 [16];
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  ...
}
```

Trình dịch ngược thấy `cltd` và `idivl` ghi vào cả `edx` lẫn `eax` nên đã mô hình hóa một giá trị trả về 16-byte đi cùng phép chia 64-bit. Nhưng danh sách lệnh mã máy trong cùng bản ghi đó lại thể hiện điều đơn giản hơn nhiều:

```text
0x100000: TEST EDI, EDI
0x100002: JZ 0x0010000d
0x100004: SUB ESI, EDI
0x100006: IMUL EAX, ESI, 0x64
0x100009: CDQ
0x10000a: IDIV EDI
0x10000c: RET
0x10000d: XOR EAX, EAX
0x10000f: RET
```

Mọi thao tác thực chất đều có độ rộng 32-bit: Lệnh `IMUL EAX, ESI, 0x64` nhân ở độ rộng 32-bit và giữ lại nửa thấp. Lệnh `CDQ` mở rộng dấu `eax` sang `edx`, và lệnh `IDIV EDI` chia cặp 64-bit đó cho một số chia 32-bit. Một bản dựng lại bám sát danh sách lệnh mã máy sẽ là biểu thức `100 * (after - before) / before` hoàn toàn trong phạm vi kiểu `int`.

Bản ghi của hàm `grade` cũng có cấu trúc tương tự: Mã giả viện dẫn một bảng dữ liệu, `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`, cùng một biểu thức mặc định khó hiểu xây dựng từ hằng số `0x66666667`. Danh sách lệnh mã máy chỉ rõ phép chia cho 10 thông qua chuỗi: `IMUL RAX, RAX, 0x66666667`, `SAR RCX, 0x22`, `SHR RAX, 0x3f`, `ADD ECX, EAX`. Tiếp theo, các lệnh `ADD ECX, -0x6`, `CMP ECX, 0x4`, và `JA` giữ lại các hàng chục từ 6 đến 10. Lệnh `XOR EAX, EAX` trước phép so sánh thiết lập giá trị mặc định bằng 0. Lệnh `LEA RCX, [0x1000b4]` và `MOV EAX, dword ptr [RCX + RAX*0x4]` đọc một bảng gồm 5 mục (mỗi mục 4-byte).

Bước 4 đã đọc chính xác 20 byte đó: `01000000 02000000 03000000 04000000 04000000`. Bảng chứa dãy số `1, 2, 3, 4, 4` cho các hàng chục 6, 7, 8, 9, 10. Đó chính là câu lệnh `switch` trong `ledger.c`, với `case 10` và `case 9` cùng chia sẻ giá trị trả về là 4.

Cả hai chứng minh trong các trang bằng chứng chính thức của rea đều bắt đầu từ danh sách mã máy và từ việc gọi `read_bytes` để đọc các hằng số. Mã giả đầu tiên của dự án DX-Ball từng là `FUN_00406400(void)` không có đầu vào ngăn xếp, và chính khung nhìn mã lệnh đã đính chính điều đó. Hãy luôn đọc danh sách mã máy trước tiên! Hãy trích dẫn mã định danh bản ghi, sau đó mới tiến hành viết mã C.

## Viết Các Ứng Viên Dựng Lại

Thư mục `code/fixtures/candidates/` lưu trữ 8 bản dựng lại, mỗi hàm nằm trong một tệp riêng biệt. Bốn bản dựng bám sát danh sách lệnh hoặc biên dịch ra cùng mã máy; bốn bản dựng còn lại nhìn có vẻ hợp lý nhưng bị sai lệch ở những điểm hiểm hóc:

| Tệp mã nguồn | Nguồn gốc | Bản chất logic |
|---|---|---|
| `percent_change_mul_first.c` | `100 * (after - before) / before` | Chuyển dịch danh sách mã máy sang C |
| `percent_change_split.c` | `(after * 100 - before * 100) / before` | Cú pháp khác nhưng cùng byte sau khi clang tối ưu |
| `percent_change_wide.c` | `(int)(((long)after - before) * 100 / before)` | Dùng kiểu `long` của mã giả, chia trước khi tràn |
| `percent_change_div_first.c` | `(after - before) / before * 100` | Chia trước rồi mới nhân, làm sai lệch quy tắc làm tròn |
| `grade_switch.c` | `switch (score / 10)` kèm 5 nhánh case | Khôi phục mã nguồn gốc |
| `grade_table.c` | `unsigned slot = score / 10 - 6; slot < 5 ? points[slot] : 0` | Chuyển dịch danh sách mã máy sang C |
| `grade_thresholds.c` | `if (score >= 90) return 4; ...` | Thiếu cận trên cho dải điểm cao nhất |
| `grade_case10_wrong.c` | `case 10: return 3;` | Sai lệch một giá trị trong bảng hằng số |

Mỗi tệp được biên dịch bằng lệnh quy chuẩn của fixture: `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c`. Các tệp đối tượng đã biên dịch được cam kết vào Git bên cạnh mã nguồn để các bài kiểm thử có thể chạy trên môi trường không có sẵn clang.

## Trụ Cột Thứ Nhất: Thực Thi và So Sánh Sai Biệt

Định nghĩa năm 1998 của McKeeman về kiểm thử sai biệt (differential testing) đòi hỏi: "hai hoặc nhiều hệ thống tương đương nhau" được cấp "các ca kiểm thử được sinh tự động bằng máy", và xem bất kỳ kết quả sai khác, tình trạng treo hoặc sự cố sụp đổ nào là một ứng viên lỗi (bug candidate). Ông đã chỉ ra điểm hóc búa nhất: "Vấn đề xấu xí nhất trong kiểm thử là đánh giá kết quả của một bài kiểm tra." Ở đây, hàm gốc đóng vai trò là chân lý chuẩn (oracle). Chúng ta không cần tự tính toán trước câu trả lời đúng; chúng ta chỉ cần hàm gốc và bản dựng lại không bất đồng ở bất kỳ điểm nào trên một lưới dữ liệu chạm tới mọi ranh giới hiểm hóc.

### Thiết kế lưới dữ liệu đầu vào

Hàm `input_grid()` trong `code/main.py` tạo ra 92 ô thử nghiệm. Đối với `percent_change`, nó kết hợp chéo từng cặp từ 10 giá trị `before` và 7 giá trị `after`:

```text
before: 0, 1, -1, 3, -3, 7, 100, -100, 2147483647, -2147483648
after:  0, 5, -5, 105, 536870911, 2147483647, -2147483648
```

Đối với `grade`, nó chọn 22 điểm số: cả hai phía của từng biên điểm số, các số âm, 0, 110, và hai giới hạn cực trị của kiểu `int`. Hàm `classify()` gắn thẻ cho từng ô: Tràn số có 32 ô, số âm có 17 ô, giá trị biên có 14 ô, thông thường có 12 ô, và lính canh số 0 có 7 ô. Cắt cụt số âm có 5 ô, làm tròn có 2 ô, vượt quá giới hạn switch có 2 ô, và bẫy ngoại lệ có 1 ô.

Mỗi lớp thử nghiệm tồn tại vì một lý do cụ thể: Lính canh số 0 kiểm tra nhánh `TEST EDI, EDI`. Các ô số âm kiểm tra phép chia cắt cụt: C và lệnh `idiv` đều làm tròn về phía số 0, do đó `-9 / 10` bằng 0 và `-10 / 10` bằng -1. Các ô làm tròn kiểm tra việc phép nhân phải diễn ra trước phép chia: `(105 - 100) * 100 / 100` ra 5 chứ không phải 0. Các ô tràn số làm cho `(after - before) * 100` vượt quá giới hạn 32-bit. Ô bẫy ngoại lệ duy nhất làm cho tích sau khi quay vòng (wrapped product) bằng đúng `INT_MIN` đi cùng `before` bằng -1.

Yang, Chen, Eide, và Regehr đã xây dựng công cụ Csmith để sinh các chương trình không chứa hành vi bất định (undefined behavior). Một sự bất đồng giữa các chân lý chuẩn trên hành vi bất định chỉ là nhiễu và không bao giờ cấu thành một báo cáo lỗi. Tràn số nguyên có dấu là hành vi bất định trong chuẩn ngôn ngữ C. Tuy nhiên, chân lý chuẩn của chúng ta là một tệp đối tượng nhị phân đã biên dịch với một hành vi phần cứng hoàn toàn cố định, do đó các ô tràn số hoàn toàn nằm trong phạm vi đánh giá. Bài học ở đây là phải phân định rõ ô thử nghiệm nào dựa trên chuẩn ngôn ngữ và ô nào dựa trên các byte máy đã biên dịch.

### Trình điều khiển và bẫy ngoại lệ tín hiệu

Tệp `code/driver.c` đọc từng ô thử nghiệm từ stdin và in ra các đầu vào, kết quả của bản gốc, và kết quả của bản dựng lại. Hàm `build_harness()` biên dịch ứng viên với các định nghĩa cờ `-Dpercent_change=cand_percent_change -Dgrade=cand_grade`, giúp liên kết cả hai phiên bản vào cùng một chương trình thực thi dưới các tên gọi khác nhau. Trên máy Linux x86-64, bản gốc chính là tệp đối tượng fixture thực tế. Trên các hệ điều hành khác, nó là tệp `ledger.c` được biên dịch trực tiếp, và báo cáo sẽ ghi nhận `original_kind: fixture-source`.

Trình điều khiển thiết lập một bộ xử lý tín hiệu `SIGFPE` bằng cặp hàm `sigsetjmp` và `siglongjmp`. Đối với cặp đầu vào `before = -1` và `after = 536870911`, hiệu số là 536870912 và tích 32-bit bị quay vòng thành -2147483648. Trên kiến trúc x86-64, lệnh `idiv` với số bị chia đó và số chia là -1 sẽ kích hoạt ngoại lệ lỗi chia (Divide Error Exception) vì thương số vượt quá giới hạn 32-bit (Intel SDM Tập 2, IDIV). Bộ xử lý tín hiệu sẽ bắt sự kiện này và ghi nhận giá trị chuỗi là `trap`.

Trên kiến trúc AArch64 (ARM64), lệnh `sdiv` không kích hoạt ngoại lệ bẫy tín hiệu mà trả về thẳng giá trị -2147483648 cho cùng các toán hạng đó. Lần chạy thực tế trên máy Mac ARM64 đã ghi nhận chính xác giá trị đó. Chân lý chuẩn phụ thuộc vào kiến trúc phần cứng, và bản báo cáo luôn nêu rõ kiến trúc nào đã thực thi.

### Kết quả kiểm thử sai biệt

Hàm `prove_behaviour(PLAUSIBLE)` đánh giá cặp ứng viên dùng kiểu `long` và dùng ngưỡng `if`. Toàn bộ 10 ca kiểm thử ngây thơ ban đầu đều vượt qua. Nhưng lưới dữ liệu toàn diện đã đánh trượt ở 28 ô thử nghiệm:

| Hàm | Đầu vào | Lớp kiểm thử | Kết quả gốc | Kết quả dựng lại |
|---|---|---|---|---|
| `percent_change` | 3, 2147483647 | tràn số | -133 | -1431655899 |
| `percent_change` | 100, 2147483647 | tràn số | -101 | 2147483547 |
| `percent_change` | 2147483647, 5 | tràn số | 0 | -99 |
| `grade` | 110 | vượt quá switch | 0 | 4 |
| `grade` | 2147483647 | vượt quá switch | 0 | 4 |

Toàn bộ 26 thất bại của `percent_change` đều rơi vào các ô tràn số với `|before| > 1`. Khi `before` bằng 1 hoặc -1, thương số 64-bit khi cắt cụt về 32-bit lại vô tình trùng với giá trị bị quay vòng, nên các ô đó vượt qua một cách tình cờ. Bản dựng lại của `grade` dùng ngưỡng bị trượt vì câu lệnh switch gốc không có nhánh nào cho hàng chục từ 11 trở lên, mà `110 / 10` bằng 11.

Hàm `prove_behaviour(EQUAL)` đánh giá `percent_change_mul_first.c` và `grade_switch.c`. Toàn bộ 92 ô thử nghiệm đều đồng thuận tuyệt đối. Ứng viên `percent_change_div_first.c` bị trượt ngay ở 2 trong 4 ca kiểm thử ngây thơ, đây chính là trường hợp hiếm hoi mà bài kiểm thử viết tay phát hiện được.

Hàm `oracle_disagreements()` kiểm tra một khía cạnh thứ hai: Các hàm `percent_change_model()` và `grade_model()` là các mô hình Python thuần túy mô phỏng danh sách mã máy (với `wrap32()` cho phép nhân và `idiv32()` cho quy tắc kiến trúc). Bản báo cáo so sánh mô hình này với bản gốc quan sát được trên từng ô. Không có bất kỳ sự bất đồng nào, chứng minh mô hình và mã nhị phân trùng khớp 100%. Mô hình là bằng chứng dẫn xuất (`derived`) còn lần chạy thực tế là bằng chứng quan sát (`observed`); báo cáo luôn lưu trữ chúng ở hai trường độc lập.

## Trụ Cột Thứ Hai: Biên Dịch Lại và So Sánh Byte

Đồng thuận hành vi trên lưới thử nghiệm mới chỉ là một trụ cột. Trụ cột thứ hai trả lời câu hỏi: Liệu bản dựng lại, khi được biên dịch bằng chính trình biên dịch và các cờ tối ưu hóa của bản gốc, có tạo ra cùng các byte mã máy hay không? Dự án DX-Ball từng báo cáo "63 matching bytes" cho hàm phụ trợ của nó. Dự án TH04 báo cáo khớp 2.139 byte. Mỗi dự án đều sử dụng trình biên dịch thuộc đúng thời kỳ của nó. Bản thân rea không thể thực hiện điều này; công cụ `compare_functions` của nó chỉ so sánh dòng văn bản và thừa nhận hạn chế: "Assembly is opaque provider text; relocation normalization is unavailable."

### Đọc tệp đối tượng nhị phân

Lớp `ElfObject` trong `code/main.py` đọc tệp đối tượng có thể tái định vị ELF64 bằng `struct.unpack_from`. Nó kiểm tra magic, class 64-bit, dữ liệu little-endian, `e_machine` 62, và `e_type` 1. Sau đó nó duyệt qua các tiêu đề phân đoạn, bảng chuỗi, `.symtab`, và từng phân đoạn `.rela`. Trong fixture gốc, `percent_change` bắt đầu tại độ lệch 0 của phân đoạn `.text` với kích thước 16 byte, và `grade` tại `0x10` với kích thước 46 byte. Hàm `total` tại `0x40` (31 byte) và `grade_label` tại `0x60` (63 byte).

Mười sáu byte của `percent_change` là `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`. Hàm này hoàn toàn không có bản ghi tái định vị nào. Hàm `grade` có đúng một bản ghi: nằm tại độ lệch `0x26` bên trong hàm, kiểu `R_X86_64_PC32`, ký hiệu `.rodata`, số hạng cộng `0x10`. Nó thuộc về lệnh `leaq` 7-byte tại độ lệch `0x23` (với 4 byte cuối là trường độ dời).

### Che trường tái định vị, sau đó so sánh bản ghi

Trong một tệp đối tượng RELA, số hạng cộng nằm trực tiếp trong bản ghi tái định vị, và 4 byte trường độ dời trong mã máy bằng 0. Ứng viên `grade_switch.o` cũng mang giá trị 0 tại đó. Phép so sánh byte thô thông thường sẽ vượt qua ngay cả khi hai bản ghi tái định vị trỏ tới hai vị trí hoàn toàn khác nhau! Hàm `mask_relocations()` chủ động ghi đè các byte 0 vào trường này ở cả hai phía để đảm bảo thuật toán hoạt động nhất quán cho cả tệp REL lẫn tệp Mach-O (nơi trường độ dời chứa sẵn số hạng cộng).

Sau đó, hàm `prove_bytes()` so sánh các bản ghi tái định vị theo độ lệch, kiểu và phân đoạn mục tiêu. Bản gốc ghi `.rodata+0x10` trong khi ứng viên ghi `.rodata-0x4`. Số hạng cộng khác nhau vì tệp đối tượng ứng viên không có bảng `labels.rel` nằm phía trước bảng switch. Số hạng cộng khác nhau chỉ phản ánh độ lệch khác nhau trong phân đoạn dữ liệu, chứ không đồng nghĩa với hàm bị sai khác logic.

### Đọc các hằng số mà bản ghi tái định vị trỏ tới

Đối với trường tương đối theo RIP nằm ở cuối lệnh, bộ xử lý cộng thêm địa chỉ của lệnh tiếp theo, do đó đích đến thực tế là: ký hiệu + số hạng cộng + 4. Trong bản gốc, vị trí đó là `.rodata+0x14`, còn trong ứng viên là `.rodata+0x0`. Hàm `read_table()` đọc 5 mục 4-byte tại các vị trí đó: Cả hai đều đọc ra dãy số `1, 2, 3, 4, 4`.

Số lượng 5 mục xuất phát từ danh sách mã máy: Lệnh `CMP ECX, 0x4` đi cùng `JA` chấp nhận các chỉ số từ 0 đến 4, và `[RCX + RAX*0x4]` đọc 4 byte cho mỗi chỉ số. Từ điển `CLAIMS` trong `code/main.py` ghi nhận cấu trúc này cho `grade` và ghi nhận 0 tái định vị cho `percent_change`.

Bước kiểm tra này được thiết lập để ngăn chặn một lỗi kinh điển từng được ghi nhận trong dự án TH08: Bộ so sánh cũ của họ "đã phát lại địa chỉ mục tiêu vào trường tái định vị của lệnh, rồi so sánh các byte lệnh mà không hề kiểm tra dữ liệu hằng số thực tế tại địa chỉ đó. Do đó hằng số 0.0 trong mã nguồn có thể hiển thị khớp hoàn hảo với hằng số 128.0 của mục tiêu!". Việc khắc phục đã đòi hỏi kiểm tra lại 1.548 tham chiếu hằng số thực tế và sửa chữa thêm 12 lỗi khác. Tệp `grade_case10_wrong.c` chính là phiên bản mô phỏng lỗi đó trong bài học của chúng ta: Toàn bộ 46 byte mã máy của nó khớp hoàn hảo với bản gốc, bản ghi tái định vị có cùng cấu trúc, nhưng bảng dữ liệu của nó là `1, 2, 3, 4, 3`. Chỉ có phép kiểm tra hằng số tham chiếu mới vạch trần được lỗi sai này!

### Kết quả so sánh byte

| Ứng viên | Kích thước | Byte sau khi che | Bản ghi tái định vị | Bảng hằng số | Trạng thái |
|---|---|---|---|---|---|
| `percent_change_mul_first` | 16 so với 16 | Trùng khớp | Không có | Không có | **pass** |
| `percent_change_split` | 16 so với 16 | Trùng khớp | Không có | Không có | **pass** |
| `percent_change_wide` | 16 so với 26 | Sai kích thước | Không có | Không có | **fail** |
| `percent_change_div_first` | 16 so với 18 | Sai kích thước | Không có | Không có | **fail** |
| `grade_switch` | 46 so với 46 | Trùng khớp | Cùng cấu trúc | `1,2,3,4,4` cả hai | **pass** |
| `grade_table` | 46 so với 46 | Trùng khớp | Cùng cấu trúc | `1,2,3,4,4` cả hai | **pass** |
| `grade_thresholds` | 46 so với 39 | Sai kích thước | 1 so với 0 | Không có | **fail** |
| `grade_case10_wrong` | 46 so với 46 | Trùng khớp | Cùng cấu trúc | `1,2,3,4,4` vs `1,2,3,4,3` | **fail** |

Hai kết quả đáng chú ý: Tệp `percent_change_split.c` có văn bản mã nguồn khác nhưng byte nhị phân lại trùng khớp tuyệt đối, vì clang xem việc tràn số nguyên có dấu là bất định nên đã tối ưu `after * 100 - before * 100` thành một phép nhân duy nhất. Tệp `grade_table.c` cũng khớp chuẩn xác từng byte, chứng tỏ trình biên dịch đi tới cùng một bảng dữ liệu từ cả cú pháp chỉ số mảng lẫn câu lệnh `switch`. Một sự trùng khớp byte chứng minh rằng trình biên dịch cụ thể này, với các cờ này, đã sinh ra cùng mã máy từ ứng viên; nó không chứng minh ứng viên là mã nguồn gốc ban đầu từng dòng chữ.

Kích thước hàm chỉ là thông tin chẩn đoán chứ không phải tiêu chí nghiệm thu: Bản dựng lại `wide` dài 26 byte vì các lệnh `movslq`, `imulq`, `cqto`, và `idivq` đã thay thế các lệnh 32-bit. Danh sách mã máy đã cho chúng ta biết độ rộng này trước khi thực hiện bất kỳ lệnh biên dịch nào.

## Vòng Lặp Agent Qua Giao Thức rea

Sáu mươi lăm công cụ MCP của rea không có lệnh CLI tương ứng, và `verify_reconstruction` là một trong số đó. Lớp `McpStdioClient` trong `code/main.py` là một client tối giản để kết nối với chúng: Nó khởi chạy `npx -y rea-agents@6.5.0 mcp`, ghi từng yêu cầu JSON-RPC ra stdin và đọc từng đối tượng JSON từ stdout trên một luồng riêng biệt. Nó gửi lệnh `initialize` với phiên bản giao thức `2025-11-25`, tiếp theo là `notifications/initialized`, rồi đến `tools/call`. Thời hạn chờ mặc định cho mỗi lời gọi là 330 giây (bằng đúng khoảng thời gian rea cho phép Ghidra khởi động nguội).

Hàm `agent_loop()` thực thi 6 lời gọi và ghi lại từng bước:

```json
{
  "step": 2,
  "tool": "analyze_function",
  "arguments": {"procedure": "percent_change"},
  "seconds": 6.31,
  "is_error": false,
  "evidence_id": "ev_e88b10dc0c6400e51bb6f7fdc3eaea8d2c97cc02463f0db3e6afeefc273197cd",
  "confidence": "observed",
  "authority": "shipped-artifact",
  "limitations": ["Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals.", "..."],
  "result_sha256": "..."
}
```

Các bản ghi Evidence hoàn chỉnh được lưu trữ theo mã định danh của chúng trong cùng một tệp. Mã giả và danh sách mã máy mà bản dựng lại bắt nguồn là một phần của fixture. Hàm `table_address()` tìm toán hạng của lệnh `LEA` trong danh sách mã máy của `grade` và chuyển nó cho `read_bytes`. Các thao tác `open_binary`, `get_evidence_bundle`, và `close_binary` không trả về bản ghi Evidence nên được ghi nhận với mã định danh null.

Lệnh `run_agent(live=True)` điều khiển rea khi `npx` có trên PATH và biến `GHIDRA_INSTALL_DIR` trỏ tới một thư mục hợp lệ. Nếu không, nó sẽ tự động phát lại fixture có sẵn và thông báo ra stderr. Bộ kiểm thử thậm chí còn chặt chẽ hơn: Tệp `code/tests/fake_rea_server.py` là một máy chủ giả lập phản hồi từng lệnh `tools/call` bằng bản ghi đã lưu trữ cho đúng công cụ và tham số đó. Client và vòng lặp agent chạy trơn tru trên mọi lần chạy kiểm thử mà không cần cài đặt rea, Ghidra hay Java.

### Điều Bộ Xác Minh Của rea Có Thể Và Không Thể Nói

Công cụ `verify_reconstruction` của rea nhận các khẳng định trỏ tới Evidence so sánh và trả về `pass`, `fail`, hoặc `unknown` cho từng khẳng định. Giới hạn cố định của nó nêu rõ: "Pass means every declared claim passed; it does not establish global implementation equivalence." (Vượt qua nghĩa là mọi khẳng định đã khai báo đều đạt; nó không xác lập tính tương đương triển khai toàn cục). Một khẳng định về hàm chỉ so sánh văn bản công cụ từ `compare_functions`. Một khẳng định hành vi chỉ so sánh hai bản chụp tiến trình từ `compare_process_captures`. Không có thành phần nào bên trong rea tự biên dịch một hàm dựng lại hoặc chạy nó trên lưới dữ liệu của bạn. Hai trụ cột trong bài học này chính là phần mà rea để bạn tự thực hiện.

Sổ cái nghĩa vụ của rea kỳ vọng một bản ghi chứng minh mang thuộc tính `predicate_type: "rea.reconstruction-proof"`, `operation: "verify_reconstruction_obligations"`, `confidence: "observed"`, và `normalized_result.passed === true`. Không có công cụ công khai nào của rea phát ra bản ghi này. Hàm `build_proof()` trong `code/main.py` sẽ phát hành nó:

```json
{
  "predicate_type": "rea.reconstruction-proof",
  "operation": "verify_reconstruction_obligations",
  "confidence": "observed",
  "authority": "controlled-replay",
  "normalized_result": {
    "passed": true,
    "status": "pass",
    "claims": [
      {"claim_id": "percent_change.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.behaviour", "kind": "behavioral", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "percent_change.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_e88b10dc..."]},
      {"claim_id": "grade.bytes", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]},
      {"claim_id": "grade.referenced-constants", "kind": "structural-function", "status": "pass", "evidence_ids": ["ev_67b6d3f9..."]}
    ]
  },
  "evidence_links": ["ev_e88b10dc...", "ev_67b6d3f9..."],
  "evidence_id": "ev_..."
}
```

Mã định danh là `ev_` cộng với SHA-256 của bản ghi (không tính trường id) trên định dạng JSON chuẩn hóa. Hàm `aggregate()` áp dụng nguyên tắc an toàn đóng (fail-closed): Trạng thái `fail` chiếm ưu tiên cao nhất, tiếp theo là `unknown`, và một danh sách khẳng định rỗng sẽ bị coi là `unknown`. Trên một máy tính không có trình biên dịch C, hai khẳng định hành vi sẽ nhận trạng thái `unknown`, do đó bản ghi chứng minh sẽ mang trạng thái `unknown` ngay cả khi cả hai phép so sánh byte đều đạt. Thẩm quyền là `controlled-replay`, vì chứng minh xuất phát từ chính harness của chúng ta chứ không chỉ từ tệp đối tượng đơn lẻ.

## Giới Nghiên Cứu Nói Gì Về Việc Chứng Minh Tương Đương

Liu, Raff, và Micinski năm 2026 đã đo lường 8 hệ thống dịch ngược trên 9 cấu hình khác nhau: Các ứng viên vượt qua toàn bộ các bài kiểm thử của benchmark vẫn phân kỳ hành vi khi bị kiểm thử fuzzing (tỷ lệ 4,9% tổng thể và lên tới 13% ở một hệ thống). Trên 300 hàm thư viện GitHub và 287 hàm liên quan tới lỗ hổng CVE, mô hình tinh chỉnh mạnh nhất đã nâng tỷ lệ biên dịch của Ghidra từ 75% lên 90%, nhưng tỷ lệ khớp đúng hành vi lại giảm từ 74% xuống 62%! Nguyên nhân được xác định là do các kiểu dữ liệu, trường và khối bảo vệ do mô hình tự bịa đặt đã thay thế các yếu tố chưa biết thực tế.

Tệp `percent_change_wide.c` của chúng ta chính là một ví dụ điển hình về kiểu dữ liệu tự bịa đặt đó: Mã giả đưa ra kiểu `long`, bản dựng lại giữ nguyên nó, kết quả là biên dịch thành công, vượt qua 10 bài kiểm thử ngây thơ, nhưng thất bại ở 26 ô tràn số thực tế! Bản dựng lại của `grade` dùng ngưỡng cũng là một khối bảo vệ tự biên tự diễn. Cả hai đều sẽ được tính là "tái thực thi thành công" theo thang đo của LLM4Decompile.

DecompileBench đo lường 23.400 hàm từ 130 chương trình thực tế và kết luận: Các phương pháp tiếp cận bằng LLM có độ chính xác chức năng thấp hơn tới 52,2% so với các trình dịch ngược công nghiệp, dù điểm số về độ dễ đọc cao hơn.

Ba phương pháp chứng minh mà giới học thuật áp dụng tương ứng với ba trụ cột bạn có thể xây dựng: Phoenix và DREAM biên dịch lại đầu ra và chạy bộ kiểm thử của chương trình gốc; McKeeman và Csmith chạy các đầu vào được sinh tự động đối chiếu với bản gốc trong phạm vi hành vi xác định; Ramos và Engler kiểm tra hai hàm C bằng thực thi biểu trưng có giới hạn (bounded symbolic execution), chứng minh tính tương đương trên toàn bộ các nhánh rẽ được khám phá. Bài học này xây dựng hai phương pháp đầu tiên. Phương pháp thứ ba chính là bài tập thực hành trong sổ cái bằng chứng của bài học 35 khi lưới dữ liệu quá nhỏ để có thể tin cậy hoàn toàn.

## Xây Dựng Mã Nguồn (Build It)

Chạy chương trình kiểm thử thuần thư viện chuẩn:

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
```

Khi có clang trên máy, bản demo sẽ biên dịch cả hai cặp ứng viên và chạy 92 ô thử nghiệm cho mỗi cặp. Nó biên dịch lại 8 ứng viên, phát lại phiên chạy của agent, và phát hành bản ghi chứng minh. Nếu không có clang, nó sẽ đọc các tệp đối tượng đã cam kết và báo cáo các khẳng định hành vi là `unknown`.

Đầu ra kỳ vọng trên máy Mac ARM64:

```json
{
  "host": "aarch64 Darwin",
  "originalKind": "fixture-source",
  "gridCells": 92,
  "plausible": {"naivePassed": true, "gridMismatches": 28, "oracleDisagreements": 0},
  "equal": {"naivePassed": true, "gridMismatches": 0, "oracleDisagreements": 0},
  "legTwo": {
    "percent_change_mul_first": {"status": "pass", "reasons": []},
    "percent_change_wide": {"status": "fail", "reasons": ["size 16 vs 26 bytes"]},
    "grade_case10_wrong": {"status": "fail", "reasons": ["referenced constants differ: [1, 2, 3, 4, 4] vs [1, 2, 3, 4, 3]"]}
  },
  "agentRun": {"mode": "replayed fixture", "server": {"name": "rea", "version": "6.5.0", "protocolVersion": "2025-11-25"}},
  "proof": {"status": "pass", "evidence_id": "ev_..."}
}
```

Hãy đọc `code/main.py` theo trình tự sau:

1. `wrap32()`, `c_div()`, `idiv32()`, và hai hàm `*_model()` lưu giữ ngữ nghĩa số nguyên của danh sách mã máy.
2. `input_grid()` và `classify()` xây dựng và gắn thẻ cho 92 ô thử nghiệm.
3. `build_harness()` và `execute()` biên dịch một chương trình duy nhất từ bản gốc, các ứng viên đổi tên, và `driver.c`, sau đó thực thi các ô.
4. `prove_behaviour()` trả về báo cáo trụ cột thứ nhất kèm danh sách bất đồng và sai lệch mô hình.
5. `ElfObject` đọc tiêu đề, ký hiệu và bản ghi `.rela`. Các hàm `mask_relocations()` và `read_table()` chuẩn bị dữ liệu so sánh.
6. `prove_bytes()` trả về báo cáo trụ cột thứ hai kèm kích thước, dải địa chỉ đã che, cấu trúc tái định vị và bảng dữ liệu.
7. `McpStdioClient` và `agent_loop()` điều khiển rea và ghi lại từng bước.
8. `build_proof()` và `aggregate()` phát hành bản ghi chứng minh an toàn đóng.

## Sử Dụng Mã Nguồn (Use It)

Để chứng minh một hàm mới là tương đương, hãy lặp lại các bước sau:

1. Gọi `analyze_function` cho hàm đó và ghi lại mã định danh bằng chứng. Hãy luôn đọc danh sách mã máy trước khi đọc mã giả.
2. Viết bản dựng lại ứng viên dựa trên độ rộng thao tác của danh sách mã máy. Giữ nguyên tên hàm gốc.
3. Liệt kê mọi phép nhân, cộng, trừ, và chia trong mã máy. Bổ sung các đầu vào làm quay vòng (wrap) từng phép toán ở độ rộng tương ứng.
4. Bổ sung cả hai phía của mọi hằng số so sánh trong danh sách mã máy (đối với `grade` là 6 và 4 sau phép trừ).
5. Bổ sung mọi số chia có thể kích hoạt ngoại lệ bẫy trên kiến trúc mục tiêu, và ghi nhận rõ kiến trúc nào đã chạy lưới thử nghiệm.
6. Chạy cả hai trụ cột kiểm chứng và đọc kỹ các nguyên nhân trước khi nhìn vào trạng thái tổng quát.
7. Lưu giữ bản ghi chứng minh bên cạnh ứng viên. Trường `evidence_links` của nó cho người đọc biết khẳng định dựa trên những bản ghi nào.

Hãy chạy lại cả hai trụ cột sau bất kỳ lần cập nhật trình biên dịch nào: Sự trùng khớp byte là một khẳng định gắn liền với một trình biên dịch cụ thể đi cùng một tập hợp cờ tối ưu hóa nhất định.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: Mô hình số học và lần chạy thực tế

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal/code
python3 -q
```

Chạy:

```python
from main import *
percent_change_model(3, INT_MAX)
percent_change_model(-1, 536870911, "x86_64")
percent_change_model(-1, 536870911, "aarch64")
report = prove_behaviour(PLAUSIBLE)
report.naive_passed, len(report.mismatches), report.mismatches[0]
```

Lời gọi đầu tiên cho ra -133 (tích bị quay vòng chia cho 3). Lời gọi thứ hai cho ra `'trap'` và lời gọi thứ ba cho ra -2147483648. Báo cáo chỉ ra 10 ca kiểm thử ngây thơ đều đạt nhưng 28 ô lưới đã thất bại.

### Lab B: Đọc tệp đối tượng nhị phân

```python
obj = ElfObject(ORIGINAL_OBJECT)
grade = obj.function("grade")
grade.size, grade.data[0x26:0x2a].hex(), grade.relocations
read_table(obj, grade.relocations[0], 5, 4)
```

Bốn byte của trường độ dời là `00000000`. Bản ghi tái định vị chỉ định `.rodata` với số hạng cộng `0x10`, và bảng dữ liệu đọc ra dãy số `[1, 2, 3, 4, 4]`.

### Lab C: Làm sai lệch một hằng số

```python
report = prove_bytes("grade", CANDIDATES / "grade_case10_wrong.o")
report.masked_equal, report.relocations_match, report.tables_equal, report.reasons()
```

Các byte sau khi che trùng khớp, các bản ghi tái định vị trùng khớp, nhưng các bảng hằng số thì không! Bây giờ hãy sửa tệp `grade_case10_wrong.c` để `case 10` trả về 4 như cũ, biên dịch lại bằng lệnh fixture, và quan sát trạng thái chuyển thành `pass`.

### Lab D: Điều khiển rea trực tiếp

Khi đã cài đặt rea 6.5.0 và Ghidra 12.1.4:

```bash
GHIDRA_INSTALL_DIR=/opt/homebrew/Cellar/ghidra/12.1.4/libexec \
JAVA_HOME=/opt/homebrew/opt/openjdk@21 \
python3 code/main.py --live
```

Lệnh `analyze_function` đầu tiên mất vài giây trong khi Ghidra nhập đối tượng. So sánh các mã định danh bằng chứng mới với phiên chạy đã cam kết: Cùng các byte nhị phân dưới cùng một cấu hình phân tích sẽ luôn sinh ra cùng các mã định danh, vì mã định danh là bản băm nội dung không phụ thuộc vào đường dẫn tệp.

## Bài Tập Thực Hành (Practice Lab)

Mở rộng cả hai trụ cột kiểm chứng cho các hàm `total` và `grade_label` từ cùng tệp đối tượng.

Các yêu cầu:

- Đối với hàm `total`, đầu vào là một mảng và biến đếm số phần tử. Xây dựng các ô thử nghiệm với biến đếm bằng 0, biến đếm âm, mảng một phần tử, và các tổng số bị tràn ở giới hạn 32-bit.
- Đối với `grade_label`, hàm này gọi nội tuyến `grade` và đọc mảng `labels.rel` (một bảng gồm 5 độ lệch chuỗi tương đối có tái định vị riêng trong `.rela.rodata`).
- Thêm một mục `CLAIMS` cho `grade_label` với 2 bản ghi tái định vị và một bảng gồm 5 mục 4-byte, đồng thời so sánh các chuỗi ký tự mà các mục đó trỏ tới.
- Ghi nhận những ô thử nghiệm nào mà mô hình và lần chạy thực tế bất đồng (nếu có) và giải thích từng trường hợp dựa trên danh sách mã máy.
- Bổ sung một khẳng định cho mỗi trụ cột cho mỗi hàm vào bản ghi chứng minh và duy trì nguyên tắc an toàn đóng.

Thành công đồng nghĩa với việc bản ghi chứng minh liệt kê đầy đủ cả 4 hàm, mỗi khẳng định đều mang mã định danh bằng chứng, và hàm `grade_label` bị đánh trượt ngay lập tức nếu chỉ có một ký tự nhãn bị sai lệch.

## Bài Tập Đánh Giá (Graded Lab)

Năm câu hỏi về các fixture của bài học này. Hãy tìm câu trả lời bằng chính mã nguồn bạn đã xây dựng, sau đó ghi kết quả vào `code/lab_answers.json`. Ghi số dưới dạng số nguyên hoặc chuỗi thập lục phân như `"0x40"`, và danh sách dưới dạng mảng JSON.

1. `t1`: Kết quả của `percent_change_model(3, 105)` trên kiến trúc x86_64.
2. `t2`: Kết quả của `percent_change_model(0, 5)` trên kiến trúc x86_64.
3. `t3`: Kết quả của `grade_model(100)`.
4. `t4`: Số lượng ô thử nghiệm có trong `input_grid()`.
5. `t5`: Số lượng ô thử nghiệm trên lưới mà mô hình x86_64 kích hoạt bẫy ngoại lệ (`trap`).

Kiểm tra câu trả lời của bạn:

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/lab.py
```

Trình kiểm tra sẽ tính toán các giá trị kỳ vọng từ `code/main.py` trên các fixture đã cam kết. Nó in ra `PASS`, `FAIL` hoặc `TODO` cho từng nhiệm vụ cùng điểm số, và chỉ thoát với mã 0 khi cả 5 bài đều đạt. Lệnh `python3 code/lab.py my-answers.json` cho phép kiểm tra một tệp câu trả lời khác. Bài kiểm thử `code/tests/test_lab.py` chứng minh trình chấm chấp nhận câu trả lời đúng và từ chối câu trả lời sai.

### Trực Quan Hóa Hoạt Động

Cỗ máy tương tác [Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) kết thúc ở bước xuất mã giả và xếp nó vào khay suy luận: giao diện web không chứng minh được tính đúng đắn của nó. Bài học này chính là bước chuyển hóa khẳng định đó thành một kết quả đã được kiểm chứng toàn diện.

## Sản Phẩm Đóng Gói (Output Skill)

Bài học này tạo ra sản phẩm `outputs/skill-rebuild-and-prove-equal.md`. Hãy sử dụng nó để chuyển đổi bằng chứng hàm và một ứng viên dựng lại thành một chứng minh hai trụ cột và một bản ghi chứng minh chuẩn mực. Kỹ năng đòi hỏi việc đọc danh sách mã máy trước mã giả, một lưới dữ liệu chạm tới mọi ranh giới hiểm hóc, che trường tái định vị kết hợp kiểm tra hằng số tham chiếu, và một cơ chế tổng hợp an toàn đóng.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử hoàn chỉnh:

```bash
cd phases/13-tools-and-protocols/36-rebuild-and-prove-equal
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Toàn bộ 35 bài kiểm tra chứng minh:

- `c_div()` làm tròn về phía số 0 và `wrap32()` tuân thủ số học bù hai.
- Mô hình trả về -133 cho `(3, INT_MAX)`, trả về `'trap'` cho ô bẫy trên x86-64, và trả về -2147483648 trên AArch64.
- Lưới thử nghiệm có đúng 92 ô bao quát toàn bộ các lớp ranh giới hiểm hóc, trong khi các ca kiểm thử ngây thơ né tránh mọi lớp khó.
- Bộ đọc ELF tìm thấy 4 hàm tại các độ lệch và kích thước đã ghi nhận.
- `percent_change` chiếm 16 byte không có tái định vị và `grade` có một bản ghi `R_X86_64_PC32` tại `0x26`.
- Các byte trường độ dời bằng 0 và việc che trường không làm thay đổi nội dung trong tệp RELA này.
- Toàn bộ 8 ứng viên đã cam kết đều đạt trạng thái kỳ vọng với các nguyên nhân đã ghi nhận.
- Tệp `grade_case10_wrong` vượt qua phép so sánh byte sau khi che và chỉ bị trượt ở phép kiểm tra hằng số.
- Các lần chạy trụ cột thứ nhất đã ghi nhận chỉ ra 10 ca ngây thơ đạt, 28 ô lưới sai lệch, và không có sự bất đồng mô hình nào.
- Khi có trình biên dịch, cả hai cặp ứng viên đều biên dịch lại và chạy lại với kết quả y hệt.
- Phiên chạy agent nhắm đúng tệp đối tượng theo mã SHA-256, tuần tự, kèm mã `ev_` ở mọi bước Evidence.
- Bản ghi `read_bytes` khớp hoàn toàn với bảng dữ liệu mà bộ đọc ELF tìm thấy.
- Client điều khiển máy chủ giả lập và ghi lại cùng các mã định danh bằng chứng như phiên chạy thực tế.
- Kết quả lỗi từ máy chủ được ghi nhận đầy đủ vào bản ghi phiên chạy.
- Hàm `aggregate()` tuân thủ nguyên tắc an toàn đóng và mã định danh chứng minh được tính toán lại từ nội dung.

## Các Chế Độ Thất Bại Trong Môi Trường Thực Tế

| Thất bại | Báo cáo yếu kém tuyên bố | Điều khung kiểm thử bắt buộc phải chứng minh |
|---|---|---|
| Bản dựng lại giữ kiểu `long` của mã giả | "vượt qua 10 bài kiểm thử" | 26 ô tràn số cho kết quả khác biệt so với bản gốc |
| Thay thế switch bằng các ngưỡng `if` | "mọi dải điểm số đều đạt" | Điểm từ 110 trở lên trả về 0 trong bản gốc nhưng trả về 4 ở bản dựng |
| Chia trước rồi mới nhân | "công thức tương đương về mặt đại số" | Các ô làm tròn bị sai lệch, các ca ngây thơ bị trượt |
| Bỏ qua ô thử nghiệm bẫy ngoại lệ | "không có sự cố sụp đổ nào" | x86-64 kích hoạt `#DE`, ARM64 trả về `INT_MIN`, báo cáo nêu rõ kiến trúc |
| So sánh byte không che tái định vị | "các byte trùng khớp" | Các bản ghi tái định vị thực chất trỏ vào các độ lệch khác nhau |
| Che tái định vị nhưng không đọc hằng số | "các byte trùng khớp sau khi che" | Bảng `1,2,3,4,3` khác biệt hoàn toàn so với `1,2,3,4,4` |
| Coi kích thước là tiêu chuẩn nghiệm thu | "cùng độ dài byte" | Các byte bên trong cùng độ dài đó bị sai lệch |
| Dùng sai trình biên dịch hoặc cờ tối ưu | "byte khác nhau, bản dựng sai" | Cùng mã nguồn sinh ra byte khác nhau, do đó phép so sánh vô giá trị |
| Coi mô hình toán học là quan sát | "mô hình hoàn toàn đồng thuận" | Lần chạy là quan sát, mô hình là dẫn xuất, các bất đồng phải được liệt kê |
| Khẳng định không có mã bằng chứng | "đã xác minh (verified)" | Mọi khẳng định bắt buộc phải liên kết mã `ev_` của hàm mà nó bao phủ |
| Thiếu trình biên dịch trên máy CI | "đã vượt qua" | Khẳng định hành vi phải là `unknown` và kết quả tổng hợp phải là `unknown` |
| Coi mã giả là mã nguồn nguyên bản | "trình dịch ngược nói là kiểu long" | Danh sách mã máy chứng minh `IMUL EAX`, `CDQ`, `IDIV EDI` đều là 32-bit |

## Quy Tắc Vận Hành Cốt Lõi

Hãy viết bản dựng lại dựa trên danh sách mã máy và nêu rõ bản ghi bằng chứng mà nó bắt nguồn. Hãy thực thi nó đối chiếu với bản gốc trên một lưới dữ liệu làm quay vòng mọi phép toán trung gian và vượt qua mọi hằng số so sánh. Hãy biên dịch lại nó bằng các cờ tối ưu hóa của bản gốc, che các trường tái định vị, so sánh các bản ghi, và đọc các hằng số mà chúng trỏ tới. Ghi nhận từng khẳng định kèm theo mã định danh bằng chứng của nó, và để một khẳng định chưa được xác minh ở nguyên trạng thái chưa biết (unknown).

## Đọc Thêm

- [McKeeman, Differential Testing for Software, Digital Technical Journal 10(1), 1998](https://www.cs.tufts.edu/comp/150FP/archive/bill-mckeeman/DifferentailTesting.pdf)
- [Yang, Chen, Eide, Regehr, Finding and Understanding Bugs in C Compilers, PLDI 2011](https://doi.org/10.1145/1993498.1993532)
- [Ramos, Engler, Practical, Low-Effort Equivalence Verification of Real Code, CAV 2011](https://doi.org/10.1007/978-3-642-22110-1_55)
- [Schwartz, Lee, Woo, Brumley, Native x86 Decompilation Using Semantics-Preserving Structural Analysis, USENIX Security 2013](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370, 2026](https://arxiv.org/abs/2609.05370)
- [Gao et al., DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Tan et al., LLM4Decompile, arXiv 2403.05286, 2024](https://arxiv.org/abs/2403.05286)
- [System V gABI, Chapter 4, Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html)
- [Kho lưu trữ x86-64 psABI, các kiểu tái định vị](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2, IDIV](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm Architecture Reference Manual for A-profile, DDI 0487, SDIV](https://developer.arm.com/documentation/ddi0487/latest)
- [Hợp đồng MCP của rea 6.5.0](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/mcp-contracts.md)
- [Sổ cái nghĩa vụ tái cấu trúc của rea 6.5.0](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/reconstruction-obligation-ledgers.md)
