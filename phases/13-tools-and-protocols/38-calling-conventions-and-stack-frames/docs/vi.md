# Quy ước gọi hàm và Khung ngăn xếp: System V, Windows x64 và AArch64

> Một bản dịch ngược hiển thị các thanh ghi mà một hàm đọc. Nhưng chỉ có ABI mới quy định ý nghĩa của những thanh ghi đó, và ABI được lựa chọn bởi tiêu đề đối tượng, vì vậy bài học này sẽ đọc tiêu đề trước tiên rồi kiểm tra từng thanh ghi dựa trên quy tắc đã đặt nó vào đó.

**Type:** Build
**Languages:** Python, C, mã assembly x86-64 và AArch64
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~140 phút

## Learning Objectives

- Xác định quy ước gọi hàm (calling convention) từ tiêu đề đối tượng (object header): ELF x86-64 tuân theo System V psABI, COFF AMD64 tuân theo Microsoft x64, ELF AArch64 tuân theo AAPCS64, Mach-O arm64 tuân theo biến thể của Apple.
- Giải mã đoạn mở đầu (prologue) của một hàm theo từng byte cho từng ABI và xác định từng thanh ghi được lưu (saved register), địa chỉ trả về (return address), và con trỏ khung (frame pointer) tại độ lệch (offset) tương ứng của nó so với con trỏ ngăn xếp lúc nhập hàm (entry stack pointer).
- Suy luận các thanh ghi tham số của một hàm từ những thanh ghi mà hàm đọc trước khi ghi đè, và các vị trí đối số ngăn xếp (stack argument slots) từ các thao tác nạp phía trên con trỏ ngăn xếp lúc nhập hàm.
- Phân biệt các chuỗi thanh ghi độc lập của System V với các vị trí tham số theo thứ tự của Microsoft x64, và phân biệt cả hai với các bộ đếm NGRN và NSRN của AAPCS64.
- Đối chiếu từng suy luận với chữ ký hàm trong mã nguồn (source signature), quy tắc trong tài liệu đặc tả, và dữ liệu tháo gỡ (unwind data) của chính trình biên dịch trong `.pdata`, `.xdata`, và `.eh_frame`.
- So sánh cách Ghidra, IDA, Binary Ninja, radare2, rizin, và angr đặt tên cũng như khôi phục cùng các dữ kiện đó, đồng thời ghi nhận những điểm mà kết quả xuất ra của chúng bất đồng với mã nguồn.

## Vấn đề thực tế

Bạn biên dịch `convention.c` cho Linux x86-64 ở mức tối ưu hóa `-O2` và yêu cầu radare2 6.2.4 phân tích các đối số của hàm `eight_args`. Công cụ phản hồi với bảy đối số: `rdi`, `rsi`, `rdx`, `r8`, `r9`, `rsp+0x8`, và `rsp+0x10`. Trong khi đó, mã nguồn C có tám tham số. Các byte thực thi đọc `%rcx` đúng một lần trong câu lệnh `leaq (%rdx,%rcx), %rax`, nhưng cơ chế khôi phục biến của công cụ đã không tính một thanh ghi chỉ xuất hiện bên trong một biểu thức địa chỉ.

Sau đó, bạn mở bản dựng Windows của cùng tệp đó trong rizin 0.9.1. Nó báo cáo quy ước là `amd64` và tìm thấy `r8`, `rcx`, `rdx`, cùng ba vị trí ngăn xếp (stack slots). Tệp đối tượng có định dạng COFF, do đó quy ước chuẩn phải là Microsoft x64, đối số thanh ghi thứ tư là `r9`, và phải có bốn vị trí ngăn xếp. Hai công cụ khác nhau đã bỏ sót các thanh ghi khác nhau từ cùng một mã nguồn.

Một danh sách mã máy (listing) là một chuỗi các chỉ lệnh. Quy ước gọi hàm là một giao kèo (contract) tồn tại bên ngoài danh sách mã máy đó: quy định thanh ghi nào mang đối số, thanh ghi nào hàm được gọi (callee) phải bảo toàn, địa chỉ trả về và các đối số ngăn xếp nằm ở đâu, cũng như bên gọi (caller) phải chừa lại khoảng trống nào phía trên con trỏ ngăn xếp. Bốn tài liệu sau đây định nghĩa các giao kèo trọng yếu: System V AMD64 psABI (Phiên bản 1.0, bản dựng ngày 12 tháng 3 năm 2025) các mục 3.2.1 đến 3.2.3; các trang tài liệu của Microsoft về quy ước gọi x64, cách dùng ngăn xếp, đoạn mở đầu và đoạn kết thúc (từ 2018-12-17 đến 2025-03-19); tài liệu Arm AAPCS64 phát hành 2025Q4 (23 tháng 1 năm 2026) các mục 6.1 đến 6.9; và các ghi chú nền tảng arm64 của Apple. Bài học này sẽ đọc các quy tắc đó, áp dụng chúng vào một tệp C được biên dịch theo ba cách khác nhau, và xây dựng một bộ phân tích chuyển đổi danh sách mã máy thành danh sách đã kiểm chứng gồm các vị trí tham số, thanh ghi được lưu và độ lệch khung ngăn xếp.

```figure
re-38-three-frames
```

## Một tiêu đề duy nhất chọn bộ quy tắc

Tệp `convention.c` chứa mười hai hàm: `leaf_add`, `eight_args`, `mixed`, `by_value`, `big_ret`, `sum_varargs`, `keeps`, và năm hàm gọi các hàm `extern` có cùng cấu trúc. Các cấu trúc dữ liệu (`struct`) sử dụng kiểu `long long`, bởi vì `long` có kích thước 4 byte trên mục tiêu MSVC nhưng lại là 8 byte trên các mục tiêu khác, và một cấu trúc thay đổi kích thước sẽ làm thay đổi quy ước truyền dữ liệu. Trình biên dịch Apple clang 17.0.0 đã biên dịch tệp này bảy lần từ máy thử nghiệm: `-target x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, và `aarch64-unknown-linux-gnu` ở các mức `-O0` và `-O2`, cùng với `arm64-apple-macos` ở mức `-O2`. Vì không có sẵn các tệp tiêu đề của Linux hay Windows, tệp mã nguồn tự khai báo những gì nó cần và chỉ bao hàm `stdarg.h` cùng `stddef.h` của chính clang.

Hàm `describe_object()` trong `code/main.py` đọc mã định danh 4 byte (magic number) cùng một trường thông tin kiến trúc máy để xác định giao kèo:

| Tiêu đề | Trường dữ liệu | Giá trị | Bộ quy tắc |
|---|---|---|---|
| `7f 45 4c 46` ELF | `e_machine` tại độ lệch 0x12 | 62 | System V AMD64 psABI |
| `64 86` COFF | `Machine` tại độ lệch 0 | 0x8664 | Microsoft x64 |
| `7f 45 4c 46` ELF | `e_machine` | 183 | AAPCS64 |
| `cf fa ed fe` Mach-O 64 | `cputype` tại độ lệch 4 | 0x0100000c | AAPCS64 với các biến thể của Apple |

Ba giao kèo này có sự khác biệt ở hầu hết các hàng trong bảng đối chiếu dưới đây. Mọi giá trị đều được trích dẫn trực tiếp từ tài liệu tương ứng và được xác minh qua các bài kiểm thử tự động.

| Quy tắc | System V psABI | Microsoft x64 | AAPCS64 |
|---|---|---|---|
| Đối số số nguyên (Integer arguments) | `%rdi %rsi %rdx %rcx %r8 %r9`, chọn thanh ghi trống tiếp theo | `RCX RDX R8 R9`, theo vị trí cố định | `x0` đến `x7`, theo bộ đếm NGRN |
| Đối số số thực (Floating-point arguments) | `%xmm0` đến `%xmm7`, chọn thanh ghi trống tiếp theo | `XMM0` đến `XMM3`, theo vị trí cố định | `v0` đến `v7`, theo bộ đếm NSRN |
| Kết quả số nguyên | `%rax`, sau đó là `%rdx` | `RAX` | `x0`, sau đó là `x1` |
| Kết quả số thực | `%xmm0`, sau đó là `%xmm1` | `XMM0` | `v0` |
| Kết quả quá lớn cho thanh ghi | Địa chỉ truyền qua `%rdi` "như thể là đối số đầu tiên", trả về trong `%rax` | Con trỏ truyền trong `RCX`, các đối số khác dịch sang phải, trả về trong `RAX` | Địa chỉ truyền trong `x8`, giữ nguyên `x0` đến `x7` |
| Thanh ghi bên được gọi phải lưu (Callee-saved) | `%rbp %rbx %r12` đến `%r15` | `RBX RBP RDI RSI RSP R12` đến `R15`, `XMM6` đến `XMM15` | `r19` đến `r29`, `SP`, 64 bit thấp của `v8` đến `v15` |
| Đối số ngăn xếp đầu tiên lúc nhập hàm | `8(%rsp)` | `40(%rsp)`, nằm phía trên vùng bóng (shadow store) 32 byte | `0(sp)` |
| Vùng nhớ phía dưới con trỏ ngăn xếp | Vùng đỏ (red zone) 128 byte | "Được coi là khả biến (volatile)" | Không có vùng đỏ (Apple: có 128 byte) |
| Căn chỉnh ngăn xếp tại điểm gọi | `(%rsp + 8)` là bội số của 16 lúc nhập hàm | Căn chỉnh bội số 16 bên ngoài prologue và epilogue | `SP mod 16 = 0` |
| Đánh dấu hàm nhận số đối số biến thiên (Variadic marker) | `%al` chứa cận trên số thanh ghi vector được dùng | Giá trị số thực được nhân bản vào thanh ghi số nguyên tương ứng | Không đánh dấu trên thanh ghi; Apple dùng các slot 8 byte trên ngăn xếp |

## System V: Hai chuỗi thanh ghi độc lập và một vùng đỏ

Mục 3.2.3 của psABI phân loại từng khối 8 byte (eightbyte) của mỗi đối số thành INTEGER, SSE, hoặc MEMORY, sau đó gán các thanh ghi từ hai danh sách hoàn toàn không phụ thuộc lẫn nhau. Lớp "INTEGER" sẽ lấy "thanh ghi khả dụng tiếp theo trong chuỗi %rdi, %rsi, %rdx, %rcx, %r8 và %r9". Lớp SSE sẽ lấy từ `%xmm0` đến `%xmm7`. "Nếu không còn thanh ghi nào khả dụng cho bất kỳ khối 8 byte nào của một đối số, toàn bộ đối số đó sẽ được truyền trên ngăn xếp", và các đối số bộ nhớ "được đẩy vào ngăn xếp theo thứ tự đảo ngược (từ phải sang trái)".

Hàm `eight_args(a, b, c, d, e, f, g, h)` biên dịch ở `-O2` có kích thước 27 byte:

```text
      10: 48 01 fe                     	addq	%rdi, %rsi
      13: 48 8d 04 0a                  	leaq	(%rdx,%rcx), %rax
      17: 48 01 f0                     	addq	%rsi, %rax
      1a: 4c 01 c0                     	addq	%r8, %rax
      1d: 4c 01 c8                     	addq	%r9, %rax
      20: 48 03 44 24 08               	addq	0x8(%rsp), %rax
      25: 48 03 44 24 10               	addq	0x10(%rsp), %rax
      2a: c3                           	retq
```

Sáu thanh ghi, sau đó là hai lệnh nạp tại `8(%rsp)` và `16(%rsp)`. Hình 3.3 của psABI đặt "khối 8 byte đối số bộ nhớ 0" tại `16(%rbp)` khi `%rbp` đóng vai trò là gốc khung (frame base), tương ứng với `8(%rsp)` lúc nhập hàm trước bất kỳ thao tác đẩy ngăn xếp (push) nào. `g` nằm tại độ lệch 8, `h` nằm tại 16.

Hàm `mixed(int i, double d, long long l, float f)` minh họa hai danh sách độc lập này. Các số nguyên chiếm `%edi` và `%rsi`, là thanh ghi INTEGER thứ nhất và thứ hai. Số thực kiểu double và float chiếm `%xmm0` và `%xmm1`, là thanh ghi SSE thứ nhất và thứ hai. Chỉ lệnh đầu tiên là `cvtsi2sd %edi, %xmm2`, và `%xmm2` ở đây là một thanh ghi tạm thời (scratch register) mà hàm ghi vào trước khi đọc. Một bộ phân tích coi đích đến của `cvtsi2sd` như một thao tác đọc sẽ tạo ra một đối số số thực thứ ba ảo. Đó chính xác là điều radare2 đã làm và báo cáo `mixed` có dạng `(double arg1, double arg2, double arg3)` trong `xmm0`, `xmm1`, `xmm2`.

Hàm `by_value(struct pair p)` truyền một cấu trúc 16 byte dưới dạng hai khối 8 byte INTEGER trong `%rdi` và `%rsi`, đồng thời trả về kết quả theo cách tương tự: `leaq 0x1(%rdi), %rax; leaq -0x1(%rsi), %rdx; retq`. Hàm `big_ret(long long seed)` trả về 32 byte, thuộc lớp MEMORY, và psABI quy định "bên gọi cung cấp không gian bộ nhớ cho giá trị trả về và truyền địa chỉ của vùng nhớ này trong %rdi như thể nó là đối số đầu tiên". Biến `seed` bị đẩy sang `%rsi`, và hàm bắt đầu bằng `movq %rdi, %rax` bởi vì "Khi trở về, %rax sẽ chứa địa chỉ đã được bên gọi truyền vào trong %rdi".

Vùng đỏ (red zone) là dấu hiệu nhận diện đặc thù khác của System V. psABI 3.2.2 nêu rõ: "Khu vực 128 byte nằm ngoài vị trí được trỏ bởi %rsp được coi là vùng dành riêng và không được phép bị sửa đổi bởi các trình xử lý tín hiệu (signal) hoặc ngắt (interrupt)." Ở mức `-O0`, `eight_args` có dạng `pushq %rbp; movq %rsp, %rbp` sau đó là sáu thao tác lưu vào `-0x8(%rbp)` đến `-0x30(%rbp)` mà không hề có lệnh `subq`. Các lệnh lưu này nằm sâu 48 byte dưới con trỏ ngăn xếp. Bản dựng `-O2` của `sum_varargs` cấp phát 88 byte bằng `subq $0x58, %rsp` rồi ghi vùng lưu thanh ghi từ `-0x58(%rsp)` xuống `-0x80(%rsp)`: đúng 128 byte phía dưới con trỏ ngăn xếp, trọn vẹn vùng đỏ. Nếu biên dịch cùng tệp đó với tùy chọn `-mno-red-zone`, hàm `leaf_add` sẽ phải thêm một lệnh `pushq %rax` và một lệnh `addq $0x8, %rsp` để chứa 8 byte biến cục bộ phía trên con trỏ ngăn xếp.

Các lời gọi hàm biến thiên đối số bổ sung thêm một thanh ghi: "%al được sử dụng làm đối số ẩn để chỉ định số lượng thanh ghi vector được dùng", đóng vai trò cận trên từ 0 đến 8. Hàm `call_varargs` thiết lập nó bằng `xorl %eax, %eax` ở `-O2` và `movb $0x0, %al` ở `-O0`. Hàm `sum_varargs` đọc nó đầu tiên: `testb %al, %al; je` sẽ bỏ qua tám thao tác lưu `movaps %xmmN` khi không có thanh ghi vector nào được sử dụng.

## Microsoft x64: Các vị trí theo thứ tự, vùng bóng và không có vùng đỏ

Quy tắc của Microsoft dựa trên vị trí đối số (positional). "Các đối số số nguyên được truyền trong các thanh ghi RCX, RDX, R8, và R9. Các đối số số thực dấu phẩy động được truyền trong XMM0L, XMM1L, XMM2L, và XMM3L." và "Bất kỳ đối số dấu phẩy động độ chính xác đơn hay kép nào trong bốn tham số đầu tiên đều được truyền trong XMM0 - XMM3, tùy thuộc vào vị trí của nó." Cùng hàm `mixed` đó sẽ nhận các tham số lần lượt là `%ecx`, `%xmm1`, `%r8`, `%xmm3`: vị trí 0 là số nguyên nên dùng `RCX`; vị trí 1 là double nên dùng `XMM1`; vị trí 3 là float nên dùng `XMM3`. Các thanh ghi `XMM0` và `XMM2` hoàn toàn bị bỏ trống.

Bên gọi cũng bắt buộc phải chừa chỗ. "Bên gọi luôn phải cấp phát đủ không gian để lưu bốn tham số thanh ghi, ngay cả khi bên được gọi không sử dụng nhiều tham số như vậy." Tài liệu gọi đây là vùng bóng (shadow store) hoặc địa chỉ nhà (home addresses), và bốn vị trí này "về bản chất thuộc quyền sở hữu của hàm được gọi". Lúc nhập hàm, `0(%rsp)` là địa chỉ trả về, `8(%rsp)` đến `39(%rsp)` là bốn vị trí nhà, và đối số ngăn xếp đầu tiên nằm tại `40(%rsp)`. Hàm `eight_args` ở mức `-O2` đọc `0x28(%rsp)`, `0x30(%rsp)`, `0x38(%rsp)`, `0x40(%rsp)`: tương ứng 40, 48, 56, 64 byte.

Phía bên gọi thể hiện rõ các phép tính số học này. Hàm `call8` ở `-O2` thực hiện `subq $0x48, %rsp` sau đó lưu các giá trị 5, 6, 7, 8 tại `0x20(%rsp)` đến `0x38(%rsp)` trước khi thực hiện `movl $1, %ecx` và các lệnh tương ứng. 72 byte ở đây bao gồm 32 byte cho vùng bóng, 32 byte cho bốn đối số trên ngăn xếp, và 8 byte để địa chỉ trả về nằm đúng ranh giới căn chỉnh 16 byte. Hàm `call_varargs` chỉ có các đối số thanh ghi nhưng vẫn phải cấp phát `subq $0x28, %rsp`: 32 byte cho vùng bóng và 8 byte để căn chỉnh. Hàm `keeps` đẩy ba thanh ghi không khả biến (nonvolatile registers) lên ngăn xếp (24 byte) và sau đó chạy `subq $0x20, %rsp`, để hàm `ext` tìm thấy vùng bóng 32 byte của nó ngay phía dưới địa chỉ trả về.

Cấu trúc dữ liệu tuân theo quy tắc riêng: "Bất kỳ đối số nào không vừa trong 8 byte, hoặc không có kích thước 1, 2, 4, hay 8 byte, bắt buộc phải được truyền qua tham chiếu (con trỏ)." và "Các đối số 16 byte được truyền qua tham chiếu." Do đó, `by_value` nhận một con trỏ trỏ tới bản sao trong `%rdx`, sau một con trỏ kết quả ẩn trong `%rcx`: "bên gọi phải cấp phát bộ nhớ cho giá trị trả về và truyền một con trỏ tới vùng nhớ đó như là đối số đầu tiên. Các đối số còn lại sau đó được dịch chuyển một vị trí sang phải." Hàm `call_pair` khởi tạo bản sao tại `0x20(%rsp)` bằng lệnh `movaps`, truyền `leaq 0x20(%rsp), %rdx`, truyền bộ đệm kết quả bằng `leaq 0x38(%rsp), %rcx`, và đọc lại trường `.b` từ `0x40(%rsp)`.

Hoàn toàn không có vùng đỏ trong Microsoft x64: "Tất cả vùng nhớ nằm ngoài địa chỉ hiện tại của RSP đều được coi là khả biến: Hệ điều hành, hoặc trình gỡ lỗi, có thể ghi đè vùng nhớ này trong phiên gỡ lỗi người dùng hoặc trong một trình xử lý ngắt." Hàm lá `leaf_add` ở mức `-O0` cần 8 byte cho các biến cục bộ và lấy chúng bằng cách chạy `pushq %rax` khi nhập hàm và `popq %rcx` trước lệnh `retq`, sử dụng hai chỉ lệnh 1 byte thay vì dùng cặp lệnh `subq` và `addq`. Trang tài liệu về mức dùng ngăn xếp định nghĩa hàm lá có thể làm điều này: "Một hàm lá là hàm không yêu cầu mục trong bảng hàm. Nó không được phép thực hiện thay đổi trên bất kỳ thanh ghi không khả biến nào, bao gồm cả RSP".

Hàm nhận số lượng đối số biến thiên bên được gọi sẽ đưa thanh ghi về vùng nhà: "Đối với các hàm vararg hoặc hàm không có nguyên mẫu (unprototyped), bất kỳ giá trị dấu phẩy động nào đều phải được nhân bản vào thanh ghi đa dụng tương ứng." Hàm `sum_varargs` ở `-O2` lưu `%rdx`, `%r8`, `%r9` vào `0x18(%rsp)`, `0x20(%rsp)`, `0x28(%rsp)` sau một lệnh `pushq %rax`, tương ứng với độ lệch 16, 24, 32 tính từ lúc nhập hàm: chính là các vị trí nhà của tham số thanh ghi thứ hai, thứ ba và thứ tư. Cấu trúc `va_list` của nó là `leaq 0x18(%rsp), %rax`, trỏ đến vị trí nhà của `%rdx`, đối số không tên đầu tiên.

## AAPCS64, và những thay đổi từ phía Apple

AAPCS64 mục 6.8.2 duyệt qua một danh sách quy tắc với hai bộ đếm. C.9: một đối số số nguyên "được sao chép vào các bit có trọng số thấp nhất trong x[NGRN]" và bộ đếm NGRN tăng lên. C.1 đối với các đối số số thực cũng làm điều tương tự với bộ đếm NSRN. C.12: một kiểu hỗn hợp có kích thước tối đa `8 - NGRN` từ kép (double-words) được "sao chép vào các thanh ghi đa dụng liên tiếp". C.13 quy định "NGRN được đặt thành 8" một khi có một đối số bị đẩy xuống ngăn xếp, do đó không còn đối số nào phía sau được đưa trở lại thanh ghi nữa. Hàm `eight_args` sử dụng trọn vẹn `x0` đến `x7` và không bao giờ chạm tới ngăn xếp. Hàm `mixed` dùng `w0`, `d0`, `x1`, `s1`. Hàm `by_value` nhận `x0` và `x1`, đồng thời trả về kết quả qua chính `x0` và `x1`.

Mục 6.9 quy định một kết quả nếu có thể truyền qua thanh ghi thì "được trả về trong cùng các thanh ghi đó", nếu không thì "Địa chỉ của khối bộ nhớ phải được truyền như một đối số bổ sung cho hàm trong thanh ghi x8". Hàm `big_ret` thực hiện `str x0, [x8]`, `stur q0, [x8, #0x8]`, `str x9, [x8, #0x18]` và không bao giờ ghi vào `x0`. Hàm `call_big` gán `mov x8, sp`, và biến `seed` giữ nguyên trong `x0`, bởi vì mục 6.9 bổ sung thêm "không có yêu cầu nào bắt buộc bên được gọi phải bảo toàn giá trị được lưu trong x8" và `x8` không nằm trong danh sách các thanh ghi đối số.

Mục 6.1.1: "Một lời gọi chương trình con phải bảo toàn nội dung của các thanh ghi r19-r29 và SP." Mục 6.4.5.2: tại giao diện công khai "SP mod 16 = 0". Ngăn xếp thuộc loại giảm dần hoàn toàn (full-descending) và "Không có luồng nào được phép truy cập (để đọc hoặc ghi) vùng không hoạt động" phía dưới nó, vì vậy bản dựng Linux `-O0` của `leaf_add` sinh ra lệnh `sub sp, sp, #0x10` cho 8 byte biến cục bộ trong khi System V tận dụng vùng đỏ.

Mục 6.4.6 định nghĩa bản ghi khung (frame record): "một bản ghi khung gồm hai giá trị 64-bit trên ngăn xếp ... Từ kép có địa chỉ thấp hơn sẽ trỏ tới bản ghi khung trước đó và từ kép có địa chỉ cao hơn sẽ chứa giá trị được truyền trong LR khi nhập hàm hiện tại." và "Vị trí của bản ghi khung bên trong một khung ngăn xếp là không cố định." Trình biên dịch clang thể hiện cả hai vị trí ngay trong tệp này. Hàm `keeps` ở `-O2` trên Linux đặt bản ghi ở đáy ngăn xếp với `stp x29, x30, [sp, #-0x30]!; mov x29, sp`. Cùng hàm đó ở `-O0` lại đặt bản ghi ở đỉnh với `sub sp, sp, #0x20; stp x29, x30, [sp, #0x10]; add x29, sp, #0x10`.

Các ghi chú arm64 của Apple giữ nguyên quy tắc về thanh ghi và căn chỉnh nhưng thay đổi khung ngăn xếp, vùng đỏ và các lời gọi hàm biến thiên đối số. "Thanh ghi con trỏ khung (x29) luôn phải trỏ tới một bản ghi khung hợp lệ. Một số hàm, chẳng hạn như hàm lá hoặc lời gọi đuôi (tail call), có thể chọn không tạo mục trong danh sách này." Bản dựng của Apple cho `keeps` lưu `x22, x21, x20, x19` trước rồi đặt bản ghi ở đỉnh: `stp x29, x30, [sp, #0x20]; add x29, sp, #0x20`. "Nền tảng này dành riêng thanh ghi x18. Tuyệt đối không sử dụng thanh ghi này." "Vùng đỏ ARM64 gồm 128 byte ngay phía dưới con trỏ ngăn xếp", tuy nhiên clang vẫn phát sinh lệnh `sub sp, sp, #0x10` cho hàm lá ở `-O0`, nghĩa là trình biên dịch mặc định không dùng tới nó. Các đối số biến thiên được gán "vào số lượng thích hợp các vị trí ngăn xếp 8 byte" và "kiểu va_list là một bí danh của char*". Hàm `call_varargs` trên Apple lưu 10, 20, 30 vào `[sp]`, `[sp, #8]`, `[sp, #0x10]` và chỉ truyền `w0`; bản dựng Linux thì truyền `w1`, `w2`, `w3` và thực hiện gọi đuôi với `b extva`. Hàm `sum_varargs` trên Apple đọc `add x8, sp, #0x10`, chính là con trỏ ngăn xếp của bên gọi, để làm `va_list`; trong khi `sum_varargs` trên Linux lưu `x1` đến `x7` cùng `q0` đến `q7` vào một khung ngăn xếp 224 byte và khởi tạo các trường cấu trúc `__gr_offs` thành -56 và `__vr_offs` thành -128 bằng một cặp lệnh `mov x8, #-0x38; movk x8, #0xff80, lsl #32`.

Ký hiệu (symbol) cũng có sự khác biệt. Mach-O thêm tiền tố `_` vào mọi ký hiệu C, do đó danh sách mã máy hiển thị `_keeps` và hàm đầu tiên xuất hiện dưới nhãn riêng tư `ltmp0`. Bộ phân tích đổi tên các nhãn này thông qua bảng ký hiệu.

## Giải mã thủ công một đoạn mở đầu qua ba trường hợp

Hàm `keeps(int n)` gọi hàm `ext(i)` bên trong một vòng lặp và phải duy trì giá trị của `n`, `i`, cùng `acc` qua lời gọi hàm đó, vì vậy mọi ABI đều bắt buộc phải lưu các thanh ghi. Hãy đọc những byte đầu tiên của bản dựng `-O2` cho từng mục tiêu.

System V, 4 byte:

1. `55` là `push r64` với thanh ghi nằm ở 3 bit thấp: `50 + 5`, tức `%rbp`. Độ sâu ngăn xếp là 8, `%rbp` được lưu tại độ lệch nhập hàm là -8.
2. `41 56`: `41` là tiền tố REX có bit `B` được bật, `56` là `50 + 6`. Bit REX.B mở rộng 6 thành 14: `pushq %r14` tại độ lệch -16.
3. `53` là `50 + 3`: `pushq %rbx` tại độ lệch -24.

Ba lệnh push tốn 24 byte. Địa chỉ trả về thêm 8 byte nữa, và 32 là bội số của 16, do đó lệnh `callq ext` bên trong vòng lặp đã được căn chỉnh chuẩn mà không cần lệnh `subq`. Không có lệnh `movq %rsp, %rbp`, nghĩa là bản dựng này không dùng con trỏ khung. Thân hàm dùng `%rbp` như một thanh ghi được bên gọi bảo toàn để chứa biến tích lũy `acc` (`xorl %ebp, %ebp`, `addl %eax, %ebp`). Đoạn kết thúc (epilogue) là hình ảnh phản chiếu đối xứng: `5b 41 5e 5d c3`.

Microsoft x64, 7 byte:

1. `56` là `pushq %rsi` tại -8. `57` là `pushq %rdi` tại -16. `53` là `pushq %rbx` tại -24. Cả ba thanh ghi đều là loại không khả biến ở đây, trong khi `RDI` và `RSI` vốn là thanh ghi truyền đối số trong System V.
2. `48 83 ec 20`: tiền tố REX.W, mã opcode `83 /5` với giá trị tức thời 8-bit, byte ModRM `ec` là `11 101 100`: `subq $0x20, %rsp`. Độ sâu ngăn xếp đạt 56.

56 cộng thêm 8 byte của địa chỉ trả về là 64, đảm bảo căn chỉnh 16 byte. 32 byte cấp phát thêm chính là vùng bóng dành riêng cho `ext`. Tài liệu về epilogue chỉ cho phép "một lệnh add RSP,constant hoặc lea RSP,constant[FPReg], theo sau bởi chuỗi gồm không hoặc nhiều lệnh pop thanh ghi 8 byte và một lệnh return hoặc jmp", và các byte thực thi là `48 83 c4 20 5b 5f 5e c3`.

AArch64, 16 byte trong bốn từ máy (words), định dạng little-endian:

1. `a9bd7bfd` là lệnh `stp` với chế độ tiền chỉ mục (pre-index). Các bit 31:30 là `10` chọn thanh ghi 64-bit, các bit 25:23 là `011` chọn tiền chỉ mục, bit 22 là 0 thể hiện phép lưu (store). imm7 tại các bit 21:15 là `1111010`, tương đương -6, nhân với tỷ lệ 8 thành -48. Rt2 tại 14:10 là 30, Rn tại 9:5 là 31 (`sp`), Rt tại 4:0 là 29. Lệnh `stp x29, x30, [sp, #-0x30]!`: con trỏ ngăn xếp giảm đi 48, `x29` được lưu tại -48 và `x30` tại -40.
2. `f9000bf5` là lệnh `str` với độ lệch không dấu 12-bit nhân tỷ lệ 8: imm12 tại 21:10 là 2, Rn là `sp`, Rt là 21. Lệnh `str x21, [sp, #0x10]`, độ lệch nhập hàm là -32.
3. `910003fd` là lệnh `add` tức thời: imm12 là 0, Rn là `sp`, Rd là 29. Lệnh `mov x29, sp`: con trỏ khung trỏ vào bản ghi tại -48.
4. `a9024ff4` là lệnh `stp` với độ lệch có dấu: imm7 là 4, tỷ lệ thành 32. Rt2 là 19, Rt là 20. Lệnh `stp x20, x19, [sp, #0x20]`: `x20` tại -16, `x19` tại -8.

Khung ngăn xếp rộng 48 byte: `x29` tại 0, `x30` tại 8, `x21` tại 16, 8 byte đệm tại 24, `x20` tại 32, `x19` tại 40. Từ máy có địa chỉ thấp nhất của bản ghi trỏ tới bản ghi trước và từ máy kế tiếp chứa LR, đúng theo yêu cầu của mục 6.4.6. Lệnh `cmp w0, #0x1` nằm xen giữa lệnh lưu thứ nhất và thứ hai; bộ lập lịch chỉ lệnh đã di chuyển nó vào đoạn mở đầu, và bộ phân tích chấp nhận lệnh so sánh hoặc lệnh số học nằm xen giữa các thao tác ngăn xếp.

## Trình biên dịch đã ghi lại cùng dữ kiện hai lần

Phân tích đoạn mở đầu chỉ là một hướng suy luận. Trình biên dịch còn phát sinh một mô tả thứ hai phục vụ cho cơ chế tháo gỡ ngăn xếp (unwinder), và hai nguồn dữ liệu này buộc phải khớp nhau.

Trên Windows, mô tả thứ hai này nằm trong `.pdata` và `.xdata`. Biên dịch đối tượng `-O2` mà không dùng cờ `-fno-asynchronous-unwind-tables`, lệnh `llvm-objdump -h` sẽ hiển thị mục `.xdata` rộng 0x34 byte và `.pdata` rộng 0x48 byte. 0x48 tức 72 byte, tương ứng sáu mục `RUNTIME_FUNCTION` mỗi mục 12 byte cùng ba bảng tái định vị `IMAGE_REL_AMD64_ADDR32NB`. Sáu hàm đó là `sum_varargs`, `keeps`, `call8`, `call_varargs`, `call_pair`, và `call_big`. Sáu hàm còn lại là các hàm lá, và tài liệu xử lý ngoại lệ quy định: "Nếu tìm kiếm không thấy mục nào trong bảng hàm, mã được coi là thuộc về một hàm lá, và RSP trực tiếp trỏ tới con trỏ trả về."

Các byte `.xdata` của `keeps` là `01 07 04 00 07 32 03 30 02 70 01 60`. Cấu trúc `UNWIND_INFO` bắt đầu với Phiên bản trong 3 bit và Cờ trong 5 bit: phiên bản 1, không có cờ. SizeOfProlog là 7, CountOfCodes là 4, FrameRegister và FrameOffset là 0. Mỗi `UNWIND_CODE` gồm một độ lệch trong prologue và một byte chỉ định mã thao tác cùng thông tin. `07 32`: tại độ lệch 7, thao tác 2 là `UWOP_ALLOC_SMALL`, info 3 nghĩa là `3 * 8 + 8 = 32` byte. `03 30`: thao tác 0 là `UWOP_PUSH_NONVOL`, thanh ghi 3 là `rbx`. `02 70`: `rdi`. `01 60`: `rsi`. Các mã này được lưu theo thứ tự thực thi đảo ngược, và chúng chỉ rõ ba thao tác đẩy thanh ghi cùng một thao tác cấp phát 32 byte y hệt như các byte mã máy `56 57 53 48 83 ec 20`. Mục của `sum_varargs` là `01 01 01 00 01 02 00 00`: một prologue 1 byte với một mã thao tác, `UWOP_ALLOC_SMALL` kích thước 8 byte. Đó là cách mà unwinder nhìn nhận lệnh `pushq %rax`: một thao tác cấp phát bộ nhớ, vì `%rax` là thanh ghi khả biến và không có dữ liệu nào cần được bảo toàn ở đây.

Trên Linux, mô tả thứ hai nằm trong `.eh_frame`, được định nghĩa bởi psABI mục 4.2.4 và mã hóa theo DWARF 5 mục 6.4. Chạy `llvm-dwarfdump --eh-frame` trên tệp đối tượng tái định vị sẽ in ra một bản ghi FDE cho mỗi hàm và giải quyết các vị trí tái định vị `R_X86_64_PC32`. Với hàm `keeps`:

```text
  DW_CFA_def_cfa_offset: +16
  DW_CFA_def_cfa_offset: +24
  DW_CFA_def_cfa_offset: +32
  DW_CFA_offset: RBX -32
  DW_CFA_offset: R14 -24
  DW_CFA_offset: RBP -16
```

CFA (Canonical Frame Address) là con trỏ ngăn xếp trước khi diễn ra lời gọi hàm, cao hơn 8 byte so với con trỏ ngăn xếp lúc nhập hàm, vì vậy `RBP -16` tương ứng độ lệch nhập hàm là -8, `R14 -24` là -16, `RBX -32` là -24, và độ lệch CFA sâu nhất 32 byte chính là khung 24 byte cộng với địa chỉ trả về. Bản ghi FDE của AArch64 cho `keeps` ghi `DW_CFA_def_cfa: W29 +48` cùng `W29 -48`, `W30 -40`, `W21 -32`, `W20 -16`, `W19 -8`. Ở đây CFA chính là con trỏ ngăn xếp lúc nhập hàm, do đó các độ lệch chính là độ lệch khung ngăn xếp mà không cần chuyển đổi, và CFA được tính toán từ `x29` trong suốt thời gian bản ghi khung còn tồn tại. Hàm `eh_frame_cross_check()` thực hiện việc đối chiếu này cho toàn bộ 12 mục FDE của cả hai tệp đối tượng. Có một trường hợp cần lưu ý: hàm `call8` theo System V đẩy hai đối số xuất phát ở giữa thân hàm, vì vậy độ lệch CFA sâu nhất của nó là 32 trong khi khung mở đầu chỉ có 8 byte, và phép kiểm tra sẽ so sánh con trỏ ngăn xếp sâu nhất mà thân hàm đạt tới là 24, cộng thêm 8.

## Cách các công cụ đặt tên cho quy ước này

Mỗi công cụ phân tích nhị phân đều có cấu trúc dữ liệu riêng đại diện cho quy ước gọi hàm cùng một lượt phân tích (pass) để gán quy ước cho một hàm. Tên gọi và các lỗ hổng nhận diện của chúng khác nhau. Các công cụ mã nguồn đóng được mô tả theo tài liệu chính thức; các công cụ mã nguồn mở được chạy trực tiếp trên các tệp mẫu và kết quả đầu ra được lưu lại.

**Ghidra 12.1.4 (Apache-2.0).** Các quy tắc nằm trong các tệp đặc tả trình biên dịch `.cspec`, bên cạnh ngôn ngữ SLEIGH. Tệp `x86-64-gcc.cspec` khai báo một `<default_proto>` có tên `__stdcall`, "Bắt nguồn từ System V Application Binary Interface AMD64 Architecture Processor Supplement tháng 4 năm 2016", với các danh sách `<pentry>` cho `XMM0_Qa` đến `XMM7_Qa` và `RDI, RSI, RDX, RCX, R8, R9`, một mục ngăn xếp tại độ lệch 8, quy tắc `join_dual_class` cho các kiểu dữ liệu tối đa 16 byte, các thanh ghi đầu ra `XMM0_Qa, XMM1_Qa, RAX, RDX`, và `hidden_return` cho các trường hợp còn lại. Tệp `x86-64-win.cspec` khai báo `__fastcall` với bốn phần tử `<group>` ghép cặp `XMM0_Qa` với `RCX`, `XMM1_Qa` với `RDX`, v.v. (đúng theo quy tắc vị trí), một mục ngăn xếp tại độ lệch 40, cờ `convert_to_ptr` cho các cấu trúc có kích thước 3, 5, 6, hoặc 7 byte cùng chú thích "chỉ các struct kích thước 8, 16, 32, 64 bit mới có thể đóng gói vào thanh ghi", danh sách `unaffected` gồm `RBX RBP RDI RSI RSP R12` đến `R15` và `XMM6` đến `XMM15`, cùng `localrange` từ 8 đến 39 trên ngăn xếp (chính là vùng bóng). Tệp `AARCH64.cspec` khai báo `__cdecl` có mục đầu tiên là `x8` với `storage="hiddenret"`, sau đó là `q0` đến `q7` và `x0` đến `x7`, cờ `convert_to_ptr` cho struct từ 17 byte trở lên, và `killedbycall` chứa `x8` với chú thích "thanh ghi vị trí kết quả gián tiếp, không được phản ánh trong danh sách pentry". Giai đoạn cuối của decompiler có tên "Establish Function's Prototype"; các analyzer là "Call Convention ID" và "Decompiler Parameter ID"; tùy chọn "Prototype Evaluation" chọn mô hình mặc định.

**IDA Pro và IDA Free (Hex-Rays, thương mại; phiên bản 9.4, bản dựng 9.4.260714).** Kiểu của hàm là một khai báo C được thiết lập thông qua SetType, và "IDA sẽ sử dụng thông tin kiểu để chú thích bản dịch ngược với thông tin về các đối số hàm." Các vị trí không được quy định bởi bất kỳ quy ước chuẩn nào sẽ được viết theo quy ước riêng của IDA: `int __usercall func@<ebx>(int x, int y@<esi>)` đặt kết quả trong `EBX` và `y` trong `ESI`, còn `@<^12.4>` biểu thị một vị trí ngăn xếp theo độ lệch và kích thước. Từ khóa `__userpurge` tương tự nhưng bên được gọi chịu trách nhiệm dọn dẹp ngăn xếp. Tài liệu cũng liệt kê `__spoils` cho tập thanh ghi bị ghi đè, `__hidden`, `__return_ptr`, và `__struct_ptr` cho các đối số ẩn mà bài học này khôi phục thủ công. Hành động "Decompile as call" cho phép nhập nguyên mẫu `__usercall` cho một chỉ lệnh cụ thể. Bản IDA Free chỉ hỗ trợ x86 và x86-64.

**Binary Ninja 6.0 "Krypton" (Vector 35, thương mại, có bản Free dịch ngược x86, x86_64, ARMv7, và AArch64).** Một đối tượng `Platform` "chứa tất cả thông tin liên quan đến môi trường thực thi của tệp nhị phân, chủ yếu là các quy ước gọi hàm được sử dụng, hệ điều hành và kiến trúc", danh sách bao gồm `linux-x86_64`, `windows-x86_64`, `linux-aarch64`, và `mac-aarch64`, mỗi môi trường có một `default_calling_convention`. Một lớp `CallingConvention` sở hữu `int_arg_regs`, `float_arg_regs`, `callee_saved_regs`, `caller_saved_regs`, `int_return_reg`, `high_int_return_reg`, `float_return_reg`, cùng hai cờ mã hóa trực tiếp những khác biệt trong bài học này: `arg_regs_share_index` ("Khi bật True, đối số thứ N sẽ tiêu thụ vị trí thứ N của cả danh sách thanh ghi số nguyên lẫn số thực bất kể kiểu dữ liệu của nó"), và `stack_reserved_for_arg_regs` ("vùng bóng/vùng nhà được sử dụng bởi quy ước Windows x64"). Các trường `eligible_for_heuristics`, `required_arg_regs`, và `required_clobbered_regs` điều khiển cơ chế nhận diện quy ước bằng heuristic. Các kiểu nền tảng là các tệp tiêu đề C trong `types/platform/`.

**radare2 6.2.4 (LGPL-3.0, chạy trực tiếp).** Lệnh `afcl` liệt kê các quy ước mà kiến trúc hỗ trợ: `amd64 amd64syscall dlang golang golang_abi0 ms p9 reg swift` cho x86-64 và `arm64 dlang golang p9 reg swift` cho AArch64. Lệnh `afc` báo cáo quy ước của hàm hiện tại, `afcr` in `cc.amd64.arg0: rdi` đến `arg5: r9` và `ret0: rax`, các hồ sơ cấu hình là tệp SDB dạng `cc-ARCH-BITS.sdb`. radare2 đã chọn đúng `amd64` cho tệp ELF và `ms` cho tệp COFF. Lệnh `afv` liệt kê đối số và biến, còn `afs` in chữ ký hàm. Trên tệp Linux `-O2`, `afv` tìm thấy bảy đối số cho `eight_args` và bỏ sót `rcx`; trên tệp Windows nó tìm thấy `rcx, rdx, r8` cùng ba vị trí ngăn xếp và bỏ sót `r9`. Cả hai thanh ghi bị thiếu chỉ xuất hiện bên trong lệnh `lea`. Với hàm `mixed`, nó báo cáo `double arg1 @ xmm0, arg2 @ xmm1, arg3 @ xmm2` trên Linux và không tìm thấy đối số nào trên Windows. Trên AArch64 nó nhận diện đúng `x0` đến `x7` cho `eight_args`, và với `big_ret` nó liệt kê `x0` cùng `double arg2 @ d1` trong khi hàm thực tế đọc `x8` và nạp một hằng số vào `q1`. Mọi chữ ký `afs` đều bắt đầu bằng `void` vì kiểu trả về không được suy luận. Ở mức `-O0`, nơi mọi đối số đều được lưu vào khung ngăn xếp, `afv` tìm thấy đủ tám đối số của `eight_args` với cặp ngăn xếp `rbp+0x10` và `rbp+0x18`.

**rizin 0.9.1 (LGPL-3.0, chạy trực tiếp) và Cutter 2.5.0.** Lệnh `afcl` liệt kê `amd64 amd64syscall ms reg swift`. Lệnh `afc` trả về `amd64` cho đối tượng COFF, cho thấy cấu hình Microsoft có tồn tại nhưng không được tự động chọn từ định dạng tệp. Lệnh `afi` liệt kê bảy đối số cho `eight_args` trên Linux (vẫn thiếu `rcx`), và trên Windows tìm thấy `r8, rcx, rdx` với ba vị trí ngăn xếp, thiếu `r9` và thiếu vị trí tại `0x40`. Lệnh `afv` không tồn tại, thay vào đó là `afvl`. Cutter là giao diện đồ họa phía trên rizin.

**angr 10.0.3 (BSD-2, chạy trực tiếp).** Phân tích `CFGFast` theo sau bởi `CompleteCallingConventions(recover_variables=True)` sẽ gán `function.calling_convention` thành `SimCCSystemVAMD64`, `SimCCMicrosoftAMD64`, hoặc `SimCCAArch64` dựa vào định dạng nhị phân, và `function.prototype` thành chữ ký hàm. Nó khôi phục đủ tám đối số của `eight_args` trên cả ba mục tiêu, phát hiện con trỏ ẩn của `big_ret` trên x86-64 dưới dạng `(BOT*, long long) -> BOT*`, cấu trúc truyền qua tham chiếu của Windows `by_value` thành `(BOT*, BOT*) -> BOT*`, và hàm vararg System V `sum_varargs` thành `(long long, ...)`. Lỗi của angr khác với radare2: `mixed` bị nhận diện thành `(long long, long long, double, float)` trên System V và AArch64 (đúng thanh ghi nhưng sai thứ tự, do danh sách số nguyên được in trước số thực); hàm AArch64 `big_ret` bị hiểu thành `(long long) -> double` (bỏ sót quy ước `x8` và coi thao tác lưu vector là giá trị trả về); và hàm AArch64 `sum_varargs` bị nhận diện có tới 16 tham số (trọn vẹn vùng lưu thanh ghi).

**Các công cụ khác.** Hopper 6.6.0 là trình dịch ngược thương mại cho macOS và Linux. lldb 1703 và gdb hiển thị khung ngăn xếp động với `frame info` và `register read`, chính là góc nhìn động của bài học 42 về cùng các độ lệch này. Compiler Explorer hiển thị ba danh sách mã song song từ mã nguồn, là cách nhanh nhất để xác minh quy tắc trước khi tìm kiếm chúng trong các byte thực thi.

| Khái niệm | Ghidra | IDA | Binary Ninja | radare2 / rizin | angr | Bài học này |
|---|---|---|---|---|---|---|
| Đối tượng quy ước | `prototype` trong `.cspec` | từ khóa quy ước gọi hàm trong kiểu hàm | `CallingConvention` trên `Platform` | tên trong `afc`, `cc-ARCH-BITS.sdb` | lớp con `SimCC` | `expected_locations(abi, name)` |
| Mặc định theo mục tiêu | `<default_proto>` theo cspec | cấu hình trình biên dịch | `Platform.default_calling_convention` | `afc=`, `anal.cc` | theo định dạng tệp | `describe_object()` |
| Thanh ghi đối số | danh sách `<pentry>`, `<group>` cho vị trí | `@<reg>` trong `__usercall` | `int_arg_regs`, `float_arg_regs`, `arg_regs_share_index` | `cc.NAME.argN` | `ARG_REGS`, `FP_ARG_REGS` | `INT_ARG_REGS`, `FLOAT_ARG_REGS` |
| Vùng bóng (Shadow store) | `localrange` ngăn xếp 8 đến 39 | ngầm định | `stack_reserved_for_arg_regs` | `cc.ms.argn=stack` | `STACKARG_SP_DIFF` | `STACK_ARG_BASE[WIN64] = 40` |
| Con trỏ kết quả ẩn | `hidden_return`, `storage="hiddenret"` | `__return_ptr`, `__hidden` | kiểu trả về quá lớn cho `int_return_reg` | không mô hình hóa | đối số đầu `BOT*` | `HIDDEN_RETURN_REG` |
| Thanh ghi bảo toàn (Callee-saved) | `<unaffected>` | `__spoils` liệt kê ngược lại | `callee_saved_regs` | tập hồ sơ | `CALLEE_CLEANUP` và kiến trúc | `CALLEE_SAVED` |
| Gán quy ước cho hàm | analyzer "Call Convention ID", "Establish Function's Prototype" | người dùng hoặc thư viện kiểu | nhận diện heuristic với `required_arg_regs` | `afc`, `afva`, `aaft` | `CallingConventionAnalysis` | đọc trước khi ghi kết hợp bộ quy tắc |

## Xây dựng hệ thống

Chạy chương trình minh họa:

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
```

Đầu ra là một báo cáo JSON hoàn chỉnh. Đối với từng tệp trong số bảy đối tượng, báo cáo nêu rõ định dạng, kiến trúc máy, và ABI; đối với từng hàm trong số mười hai hàm, nó cung cấp các byte đoạn mở đầu, các thanh ghi được lưu kèm độ lệch nhập hàm, kích thước khung ngăn xếp, con trỏ khung, các thanh ghi tham số, độ lệch đối số ngăn xếp, các vùng tràn (spills), thanh ghi trả về hoặc con trỏ ẩn, độ sâu tại điểm gọi, và các thanh ghi cùng thao tác lưu bên ngoài của caller. Mỗi hàm kết thúc bằng các phép kiểm tra: quy tắc, giá trị kỳ vọng từ chữ ký hàm, giá trị quan sát từ mã máy, và trạng thái đồng thuận. Báo cáo kết thúc bằng hai đối chiếu tháo gỡ ngăn xếp và các kết quả thu thập từ công cụ ngoài. Trên 84 lượt phân tích hàm, 489 phép kiểm tra được thực thi và toàn bộ 489 phép kiểm tra đều nhất quán tuyệt đối.

Cấu trúc định dạng kỳ vọng:

```json
{
  "objects": {
    "convention-x86_64-windows-O2": {
      "format": "COFF", "machine": "AMD64", "abi": "Microsoft x64", "checks": "73/73 agree",
      "functions": {
        "keeps": {
          "prologue": ["56  pushq %rsi", "57  pushq %rdi", "53  pushq %rbx", "48 83 ec 20  subq $0x20, %rsp"],
          "frameBytes": 56,
          "savedRegisters": [{"register": "rsi", "entryOffset": -8}, {"register": "rdi", "entryOffset": -16}, {"register": "rbx", "entryOffset": -24}],
          "parameterRegisters": ["rcx"],
          "returnRegisters": ["rax"],
          "calls": [{"callee": "ext", "depth": 56}],
          "checks": [{"rule": "Microsoft x64: the caller must always allocate ...", "expected": "at least 32 bytes ...", "observed": [56], "agree": true}, "..."]
        }
      }
    }
  },
  "sehCrossCheck": ["..."],
  "ehFrameCrossCheck": {"...": "..."}
}
```

Hãy đọc mã nguồn trong `code/main.py` theo trình tự sau:

1. `describe_object()` đọc tiêu đề ELF, COFF, hoặc Mach-O cùng bảng ký hiệu, sau đó trả về ABI và danh sách hàm.
2. `parse_listing()` phân tách văn bản từ `llvm-objdump -d -r` thành từng hàm, ghi nhận đích đến tái định vị của mỗi lệnh gọi và nhảy, đồng thời loại bỏ chú thích theo cú pháp phù hợp: `#` cho AT&T, `//` và `;` cho AArch64 (nơi `#` được dùng cho giá trị tức thời).
3. `effect_x86()` và `effect_arm()` chuyển đổi một chỉ lệnh thành các tập: đọc, đọc địa chỉ, ghi, nạp (load), lưu (store), và độ biến thiên con trỏ ngăn xếp. Toán hạng cuối cùng trong cú pháp AT&T là đích đến; toán hạng đầu tiên trong AArch64 là đích đến ngoại trừ các lệnh lưu, so sánh, và rẽ nhánh. Lệnh `lea` đọc một địa chỉ và không nạp gì cả. Đích đến của `mov`, `cvtsi2sd`, `setcc`, và mẫu gán 0 `xor r, r` chỉ là thao tác ghi đơn thuần. Lệnh gọi hàm ghi đè toàn bộ tập thanh ghi khả biến. Một lệnh lưu tiền chỉ mục (pre-indexed store) sẽ thay đổi `sp` trước khi truy cập bộ nhớ và một lệnh nạp hậu chỉ mục (post-indexed load) sẽ thay đổi `sp` sau khi truy cập.
4. `analyze_function()` duyệt qua thân hàm một lượt cho đến lệnh `ret` hoặc lệnh nhảy đuôi đầu tiên. Nó theo dõi độ sâu của con trỏ ngăn xếp so với giá trị lúc nhập hàm, sao cho một toán hạng bộ nhớ `disp(%rsp)` sẽ nằm tại độ lệch nhập hàm là `disp - depth` và `disp(%rbp)` sẽ nằm tại `frame_pointer + disp`. Nó ghi nhận các thanh ghi thuộc lớp đối số được đọc trước khi ghi, các thao tác nạp tại hoặc phía trên độ lệch đối số ngăn xếp đầu tiên của ABI, thao tác lưu thanh ghi đối số vào khung, các truy cập phía dưới con trỏ ngăn xếp hiện tại, và độ sâu ngăn xếp tại mỗi lời gọi.
5. `_return_registers()` giữ lại một thanh ghi kết quả khi truy cập cuối cùng vào nó trước khi thoát hàm là một thao tác ghi. Thao tác đọc nó như một toán hạng dữ liệu sẽ loại bỏ ứng viên này; tuy nhiên đọc nó như một thanh ghi cơ sở địa chỉ thì không loại bỏ, điều này giúp giữ lại `rax` trong hàm `big_ret` trên Microsoft x64, nơi `movq %rcx, %rax` được theo sau bởi các lệnh lưu thông qua `(%rax)`.
6. `expected_locations()` áp dụng ba bộ quy tắc vào chữ ký C: hai danh sách System V với các lớp 8-byte, các vị trí theo thứ tự của Microsoft với cấu trúc truyền qua tham chiếu, và các bộ đếm AAPCS64 với `x8` cho kết quả lớn hơn 16 byte.
7. `run_checks()` so sánh hai bên và bổ sung các quy tắc khung ngăn xếp: thanh ghi callee-saved bị ghi bắt buộc phải được lưu trước đó, `(rsp + 8) % 16 == 0` hoặc `sp % 16 == 0` tại mỗi lời gọi, 32 byte vùng bóng tại mỗi lời gọi trên Microsoft, không được truy cập phía dưới con trỏ ngăn xếp trên Microsoft và AAPCS64 và tối đa 128 byte trên System V, cùng một đoạn kết thúc đối xứng với đoạn mở đầu sao cho con trỏ ngăn xếp quay trở lại giá trị nhập hàm ban đầu.
8. `seh_cross_check()` và `eh_frame_cross_check()` đọc dữ liệu tháo gỡ ngăn xếp từ các đối tượng biên dịch không dùng `-fno-asynchronous-unwind-tables` và đối chiếu chúng với kết quả phân tích đoạn mở đầu.

Có thêm ba lệnh thực thi hỗ trợ: `python3 code/main.py analyze convention-x86_64-windows-O2 keeps` in phân tích một hàm duy nhất. `python3 code/main.py object FILE` in phán quyết từ tiêu đề. `python3 code/main.py seh` in cấu trúc `UNWIND_INFO` đã giải mã của mọi hàm sở hữu bảng này.

## Sử dụng hệ thống

Sử dụng bộ phân tích theo thứ tự sau trên một tệp đối tượng nhị phân mới mà bạn biên dịch:

1. Chạy `python3 code/main.py object FILE.o` và ghi lại ABI mà nó xác định. Nếu tiêu đề tệp và lệnh biên dịch của bạn không khớp nhau, hãy dừng lại; chắc chắn một trong hai bên đã sai.
2. Thu xuất bản dịch ngược bằng `llvm-objdump -d -r FILE.o` đặt cạnh tệp đối tượng và chạy phân tích. Đọc các byte mở đầu và các thanh ghi được lưu trước tiên, rồi so sánh kích thước khung với các thao tác `push` cùng lệnh `sub`.
3. Đọc các thanh ghi tham số và các độ lệch ngăn xếp đối chiếu với chữ ký hàm trong mã nguồn. Một thanh ghi không được giải thích bởi chữ ký hàm có thể là một con trỏ ẩn, một dấu hiệu vararg, hoặc một thanh ghi tạm thời mà bộ phân tích nhận diện nhầm thành đối số. Trường hợp cuối là lỗi trong bảng hiệu ứng lệnh (effect table). Hãy sửa nó bằng một quy tắc tổng quát cho chỉ lệnh đó.
4. Đọc các phép kiểm tra không khớp (nếu có). Mỗi phép kiểm tra đều nêu rõ quy tắc mà nó dựa vào, vì vậy điều đầu tiên cần xem xét là quy tắc, tiếp theo là các byte mã máy, và cuối cùng là bộ phân tích.
5. Khi đối tượng có dữ liệu tháo gỡ (unwind data), hãy chạy kiểm tra chéo. Hai phép suy luận độc lập mà đồng thuận với nhau là bằng chứng xác thực mạnh mẽ nhất mà bài học này tạo ra.
6. Khi có cài đặt công cụ phân tích khác, hãy thu xuất kết quả của nó bằng lệnh chính xác và so sánh. Khi công cụ đó đồng thuận, bạn có ba nguồn xác thực độc lập. Khi nó bất đồng, bạn đã định vị được một thanh ghi cụ thể và một chỉ lệnh cụ thể cần kiểm tra sâu hơn.

## Phòng thí nghiệm tương tác

### Lab A: Một chỉ lệnh, ba hiệu ứng

Từ thư mục `code`, khởi động môi trường Python:

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames/code
python3 -q
```

Thực thi:

```python
from main import *
effect(Instruction(0, "48 8d 04 0a", "leaq", "(%rdx,%rcx), %rax"), SYSV)
effect(Instruction(0, "f2 0f 2a d7", "cvtsi2sd", "%edi, %xmm2"), SYSV)
effect(Instruction(0, "a9bd7bfd", "stp", "x29, x30, [sp, #-0x30]!"), AAPCS64)
```

Ghi lại các thao tác đọc, đọc địa chỉ, ghi, và độ lệch ngăn xếp mà bạn dự đoán trước mỗi lần gọi. Sau đó đổi `[sp, #-0x30]!` thành `[sp], #0x30` với lệnh `ldp` và quan sát độ biến thiên dịch chuyển từ `sp_delta` sang `post_sp_delta`.

### Lab B: Khung ngăn xếp của hàm keeps

```python
info, listing = load_fixture("convention-aarch64-linux-O2")
keeps = analyze_function("keeps", listing["keeps"], info.abi)
keeps.saved
keeps.frame_pointer
[c.to_dict() for c in frame_cells(keeps)]
```

So sánh các độ lệch với phép giải mã thủ công phía trên. Tiếp theo tải `convention-arm64-macos-O2` và chạy lại ba dòng lệnh này; bản ghi khung chuyển lên -16 và bốn thanh ghi nằm phía dưới nó. Sau đó tải `convention-x86_64-linux-O2-fp` và xem trường `saved` của `keeps`: con trỏ khung xuất hiện trở lại, cùng với một lệnh `pushq %rax` mà mục đích duy nhất là làm cho `(rsp + 8)` trở thành bội số của 16 tại điểm gọi hàm.

### Lab C: Cơ chế đọc trước khi ghi trên hàm mixed

```python
info, listing = load_fixture("convention-x86_64-linux-O2")
mixed = analyze_function("mixed", listing["mixed"], info.abi)
mixed.param_registers
mixed.return_registers
expected_locations(SYSV, "mixed")["parameters"]
expected_locations(WIN64, "mixed")["parameters"]
```

Danh sách của System V là `rdi, xmm0, rsi, xmm1` và danh sách của Microsoft là `rcx, xmm1, r8, xmm3`. Bây giờ nạp tệp Windows và chạy lại ba dòng đầu. Tiếp theo mở tệp ghi nhận kết quả radare2 cho đối tượng Linux và tìm mục `[afv mixed]`: xuất hiện ba số thực double. Hãy tìm chỉ lệnh giải thích tại sao có số thực thứ ba đó.

### Lab D: Ý kiến độc lập thứ hai từ unwinder

```python
for row in seh_cross_check():
    print(row["function"], row["sizeOfProlog"], row["unwindPushes"], row["unwindAllocBytes"], row["agree"])
for row in eh_frame_cross_check("convention-x86_64-linux-O2-eh"):
    print(row["function"], row["cfiSaved"], row["cfiMaxOffset"], row["deepestStack"], row["agree"])
```

Tìm lệnh `pushq %rax` của hàm `sum_varargs` trong bảng SEH và giải thích tại sao mã tháo gỡ ghi nhận nó là một thao tác cấp phát (allocation). Sau đó tìm `call8` trong bảng `.eh_frame` và giải thích con số 32.

## Bài tập thực hành

Bổ sung mục tiêu thứ tư và bộ quy tắc thứ tư vào hệ thống.

Yêu cầu thực hiện:

- Biên dịch `convention.c` với tùy chọn `-target x86_64-apple-macos -O2` và trích xuất danh sách mã máy. Mở rộng `describe_object()` để đọc giá trị Mach-O `cputype` là 0x01000007 và ánh xạ nó vào System V psABI (chuẩn mà Apple tuân theo trên x86-64), xử lý thêm tiền tố ký hiệu `_`.
- Xác minh rằng mọi hàm đều thỏa mãn các phép kiểm tra của System V. Ghi nhận một điểm khác biệt duy nhất bạn tìm thấy trong các đoạn mở đầu: Darwin luôn duy trì con trỏ khung ở mọi mức tối ưu hóa.
- Bổ sung một bản dựng với cờ `-fstack-protector-strong` cho đối tượng Linux x86-64. Mở rộng bộ quét mở đầu sao cho thao tác nạp canary từ `%fs:0x28` và thao tác lưu phía dưới các biến cục bộ được ghi nhận là một ô `canary`, và lệnh so sánh trước `ret` được ghi nhận là phép kiểm tra bảo vệ đoạn kết thúc.
- Viết một bài kiểm thử xác nhận thất bại nếu thao tác lưu canary bị tính nhầm thành một đối số tràn ra ngăn xếp.
- Vẽ sơ đồ khung ngăn xếp của đoạn mã theo mẫu `with_canary` thủ công trước khi chạy bộ phân tích, và lưu lại bản vẽ cạnh bài kiểm thử.

Thành công đạt được khi `python3 code/main.py` báo cáo đối tượng mới với mọi phép kiểm tra đều khớp, và các ô khung ngăn xếp của hàm được bảo vệ hiển thị vị trí canary nằm giữa các biến cục bộ và các thanh ghi được bảo toàn.

## Thành phẩm bài học

Bài học này bao gồm tệp `outputs/skill-calling-conventions-and-stack-frames.md`. Sử dụng nó để biến một tệp đối tượng bạn đã biên dịch, danh sách dịch ngược mã máy, và các chữ ký hàm C thành các vị trí tham số, bố cục khung ngăn xếp, thanh ghi trả về, cùng phép kiểm tra từng suy luận đối chiếu với quy tắc chuẩn và dữ liệu tháo gỡ của trình biên dịch. Kỹ năng này bắt buộc phải lấy phán quyết ABI từ tiêu đề trước tiên, cấm đọc đích đến của `cvtsi2sd` hay `xor r, r` như một thao tác đọc, và dừng lại ngay tại điểm bất đồng đầu tiên.

## Kiểm chứng thực tế

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/38-calling-conventions-and-stack-frames
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình kiểm chứng cần chứng minh:

- Tiêu đề ELF, COFF, và Mach-O của bảy đối tượng xác định đúng System V, Microsoft x64, AAPCS64, và Apple arm64
- Bảng ký hiệu của mỗi đối tượng liệt kê đủ mười hai hàm, với tiền tố `_` của Mach-O và nhãn `ltmp0` được giải quyết chuẩn xác
- Bộ phân tích danh sách mã máy giữ nguyên `ext` làm đích đến của lệnh gọi và duy trì các giá trị tức thời AArch64 như `#-0x30` nguyên vẹn
- Lệnh `leaq (%rdx,%rcx), %rax` đọc hai thanh ghi địa chỉ và không nạp gì; lệnh `xorl %eax, %eax` và `cvtsi2sd` chỉ là thao tác ghi; lệnh `stp` tiền chỉ mục di chuyển `sp` trước thao tác lưu và `ldp` hậu chỉ mục di chuyển `sp` sau thao tác nạp
- Hàm `keeps` ở `-O2` có các byte `55 41 56 53` trên System V với khung 24 byte không có con trỏ khung; có dạng `56 57 53 48 83 ec 20` trên Microsoft x64 với khung 56 byte; và có `a9bd7bfd` đầu tiên trên AArch64 với bản ghi tại -48
- Bản dựng Apple đặt bản ghi tại -16 phía trên bốn thanh ghi được lưu
- Mọi hàm đều có con trỏ khung ở mức `-O0` trên Linux và không có ở `-O2`, cờ `-fno-omit-frame-pointer` sẽ khôi phục con trỏ khung cùng lệnh đẩy ngăn xếp căn chỉnh
- Hàm lá ở `-O0` dùng 8 byte vùng đỏ trên System V, dùng `pushq %rax` trên Microsoft x64, và dùng `sub sp, sp, #0x10` trên AAPCS64; cờ `-mno-red-zone` loại bỏ việc dùng vùng đỏ
- Mọi đoạn kết thúc đều phản chiếu đoạn mở đầu và con trỏ ngăn xếp quay về đúng giá trị lúc nhập hàm
- Hàm `eight_args` đọc sáu thanh ghi cùng các vị trí 8 và 16; bốn thanh ghi cùng các vị trí 40 đến 64; hoặc `x0` đến `x7`, ở cả hai mức tối ưu hóa
- Hàm `mixed` dùng `rdi, xmm0, rsi, xmm1` trên System V và `rcx, xmm1, r8, xmm3` trên Microsoft x64, với `xmm0` là thanh ghi kết quả trên cả hai
- Cấu trúc 16 byte truyền trong hai thanh ghi trên System V và AAPCS64, và truyền qua tham chiếu phía sau con trỏ ẩn trên Microsoft x64
- Cấu trúc 32 byte trả về thông qua `rdi`, `rcx`, hoặc `x8`
- Bên được gọi trong hàm vararg đọc `%al`, đưa `rdx, r8, r9` về các vị trí nhà 16, 24, 32, lưu `x1` đến `x7` cùng `q0` đến `q7`, hoặc đọc con trỏ ngăn xếp của bên gọi trên Apple
- Bên gọi thiết lập đúng các thanh ghi mong đợi, lưu 7 và 8 tại 0 và 8 hoặc 5 đến 8 tại 32 đến 56, xóa `%al` về 0 cho lời gọi vararg, và truyền địa chỉ bộ đệm kết quả trong `rdi`, `rcx`, hoặc `x8`
- Toàn bộ 489 phép kiểm tra đều khớp với các quy tắc ABI
- Sáu mục `UNWIND_INFO` khớp với phân tích đoạn mở đầu và các hàm lá không có mục nào
- 24 mục FDE khớp với các độ lệch thanh ghi được lưu và con trỏ ngăn xếp sâu nhất
- Dữ liệu chụp từ các công cụ ghi nhận chính xác `amd64` và `ms` của radare2, `amd64` của rizin trên đối tượng COFF, thanh ghi `rcx` bị thiếu, ba số thực double, và các nguyên mẫu hàm của angr
- Dữ liệu hình ảnh nhúng trong mô-đun trang web hoàn toàn khớp với kết quả xuất ra của bộ phân tích

## Các kịch bản lỗi trong môi trường thực tế

| Tình huống lỗi | Báo cáo từ công cụ yếu | Điều phân tích chuẩn phải chứng minh |
|---|---|---|
| Đối tượng COFF bị đọc theo System V | Báo `rdi` và `rsi` là các đối số mà hàm không bao giờ đọc | Tiêu đề là 0x8664; quy ước bắt buộc là Microsoft x64 |
| Thanh ghi chỉ được đọc bên trong lệnh `lea` | Báo bảy đối số cho một hàm có tám đối số | Thanh ghi cơ sở và thanh ghi chỉ mục đều là các thao tác đọc |
| Đích đến của lệnh `cvtsi2sd` bị tính là thao tác đọc | Báo một đối số double thứ ba ảo trong `xmm2` | Lần truy cập đầu vào `xmm2` là thao tác ghi; `mixed` chỉ có hai đối số SSE |
| Các vị trí theo thứ tự bị đọc nhầm thành chuỗi độc lập | Gán `xmm0` là đối số double trên Windows | Vị trí 1 là `XMM1`, vị trí 3 là `XMM3` |
| Lệnh `push rbp` bị nhầm thành con trỏ khung | Báo có con trỏ khung trong hàm `-O2` vốn dùng `rbp` làm biến tích lũy | Chỉ lệnh `mov %rsp, %rbp` nằm ngay sau push mới tạo ra con trỏ khung |
| Vùng đỏ được đo từ con trỏ ngăn xếp lúc nhập hàm | Báo 216 byte "phía dưới con trỏ ngăn xếp" | Thực chất là 128 byte phía dưới con trỏ ngăn xếp hiện tại sau lệnh `sub` 88 byte |
| Vùng bóng bị xử lý nhầm thành biến cục bộ | Đặt các đối số ngăn xếp tại 8 đến 32 | Các đối số ngăn xếp của Microsoft bắt đầu tại 40 |
| Lệnh `lea` bị xử lý nhầm thành thao tác nạp bộ nhớ | Báo một đối số ngăn xếp tại `0x60(%rsp)` trong hàm vararg | Lệnh `lea` tính toán `overflow_arg_area`; nó không nạp dữ liệu |
| Con trỏ ẩn bị đọc như một đối số bình thường | Suy luận hàm dạng `big_ret(void *, long long)` | Con trỏ được lưu gián tiếp qua và sao chép vào `rax`, `rax`, hoặc giữ nguyên trong `x8` |
| Giá trị CFA sâu nhất đem so sánh với khung mở đầu | Báo lệch trên một hàm gọi có đẩy các đối số xuất phát ra ngăn xếp | Phải so sánh với con trỏ ngăn xếp sâu nhất mà thân hàm đạt tới |
| Quy ước bị phán đoán thuần túy từ tập lệnh ISA | Gán `amd64` cho một tệp đối tượng COFF | Định dạng tệp chọn bộ quy tắc, sau đó ISA mới chọn các thanh ghi cụ thể |
| Tin tưởng tuyệt đối vào nguyên mẫu từ một công cụ đơn lẻ | Gặp lỗi ba số thực double, hoặc mười sáu tham số vararg | Bắt buộc phải có sự đồng thuận từ chữ ký mã nguồn hoặc một phép suy luận độc lập thứ hai |

## Quy tắc vận hành

Hãy đọc tiêu đề tệp trước chỉ lệnh đầu tiên, và gọi tên bộ quy tắc áp dụng. Đặt từng thanh ghi được lưu và từng đối số tại một độ lệch tính từ con trỏ ngăn xếp lúc nhập hàm, rồi kiểm tra nó dựa trên quy tắc đã định vị nó tại đó và đối chiếu với dữ liệu tháo gỡ ngăn xếp mà trình biên dịch đã tạo ra cho cùng khung đó. Hãy xem nguyên mẫu hàm mà một công cụ tự động đưa ra như một giả thuyết cho đến khi các byte thực thi hoặc mã nguồn xác nhận nó.

## Tài liệu đọc thêm

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Version 1.0 (sections 3.2.1 to 3.2.3, 4.2.4)](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Microsoft x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)
- [Microsoft x64 stack usage](https://learn.microsoft.com/en-us/cpp/build/stack-usage)
- [Microsoft x64 prolog and epilog](https://learn.microsoft.com/en-us/cpp/build/prolog-and-epilog)
- [Microsoft x64 exception handling: RUNTIME_FUNCTION, UNWIND_INFO, UNWIND_CODE](https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64)
- [Arm Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), release 2025Q4](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Apple: Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [DWARF Debugging Information Format Version 5, section 6.4 Call Frame Information](https://dwarfstd.org/doc/DWARF5.pdf)
- [Clang driver frame-pointer defaults, `useFramePointerForTargetByDefault` in `CommonArgs.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/CommonArgs.cpp)
- [Clang command line reference: `-fomit-frame-pointer`, `-momit-leaf-frame-pointer`](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Fedora 38 change: frame pointers by default](https://fedoraproject.org/wiki/Changes/fno-omit-frame-pointer)
- [Ubuntu 24.04: performance engineering with frame pointers by default](https://ubuntu.com/blog/ubuntu-performance-engineering-with-frame-pointers-by-default)
- [Ghidra compiler specification: `x86-64-gcc.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-gcc.cspec)
- [Ghidra compiler specification: `x86-64-win.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/x86/data/languages/x86-64-win.cspec)
- [Ghidra compiler specification: `AARCH64.cspec`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/languages/AARCH64.cspec)
- [Hex-Rays: SetType and the `__usercall` convention](https://docs.hex-rays.com/ida-actions/settype)
- [Hex-Rays: Decompile as call](https://docs.hex-rays.com/ida-actions/hx_userdefinedcall)
- [Binary Ninja API 6.0: `callingconvention` module](https://api.binary.ninja/binaryninja.callingconvention-module.html)
- [Binary Ninja API 6.0: `platform` module](https://api.binary.ninja/binaryninja.platform-module.html)
- [Binary Ninja user documentation: platform types](https://docs.binary.ninja/guide/types/platformtypes.html)
- [The Official Radare2 Book: calling conventions](https://book.rada.re/analysis/calling_conventions.html)
- [The Official Radare2 Book: code analysis](https://book.rada.re/analysis/code_analysis.html)
- [angr documentation: decompiler analysis passes, including calling convention recovery](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [rizin 0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
