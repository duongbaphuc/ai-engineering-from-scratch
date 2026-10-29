# Khôi phục Dữ liệu và Kiểu dữ liệu: Cấu trúc, Mảng, Đối tượng C++ và DWARF

> Một trình dịch ngược sẽ in `a->balance` khi nó có bảng kiểu dữ liệu và in `*(int *)(param_1 + 4)` khi không có. Bài học này xây dựng bảng kiểu từ các thao tác truy cập, đọc bảng kiểu thực tế từ DWARF 5, và chỉ ra những điểm hai bên đồng thuận.

**Type:** Build
**Languages:** Python, C, C++, mã assembly x86-64
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~140 phút

## Learning Objectives

- Đọc thao tác truy cập thành viên cấu trúc (struct member access) dưới dạng thanh ghi cơ sở cộng độ dời (displacement), và truy cập mảng dưới dạng cơ sở cộng chỉ số nhân bước nhảy (stride), từ các quy tắc định địa chỉ trong tài liệu Intel SDM.
- Áp dụng các quy tắc căn chỉnh (alignment rules) của System V bằng tay cho `struct account` và giải thích từng byte đệm (padding).
- Xây dựng một lượt khôi phục kiểu dữ liệu (type recovery pass) biến các truy cập trong danh sách mã máy thành độ lệch trường, độ rộng, kiểu dữ liệu, bước nhảy và kích thước tổng thể, đi kèm bằng chứng xác thực cho từng mục.
- Viết bộ đọc DWARF 5 cho các phân đoạn `.debug_info`, `.debug_abbrev`, `.debug_str_offsets`, và `.debug_addr` của một tệp đối tượng tái định vị, đối chiếu kiểm tra với `llvm-dwarfdump`.
- So sánh bố cục bộ nhớ được khôi phục với bố cục DWARF và phân loại mọi khác biệt thành: thiếu bằng chứng, truy cập bị gộp, hoặc kiểu dữ liệu không thể chứng minh.
- Đọc bảng hàm ảo (vtable) Itanium C++, thông tin kiểu thời gian chạy (typeinfo / RTTI), và các tên bị biến đổi (mangled names) từ các mục tái định vị, sau đó phân giải một lời gọi ảo về một ô nhớ (slot) cụ thể.

## Vấn đề thực tế

Bạn biên dịch `account.c` không kèm cờ `-g` và hỏi một AI agent xem hàm `bump` thực hiện công việc gì. Agent đưa tệp đối tượng vào Ghidra 12.1.4 và nhận lại kết quả:

```c
void bump(long param_1,int param_2)
{
  *(int *)(param_1 + 4) = param_2 + *(int *)(param_1 + 4);
  return;
}
```

Cùng mã nguồn đó khi biên dịch với cờ `-g` cho ra kết quả sau từ chính trình dịch ngược đó:

```c
int bump(account *a,int by)
{
  a->balance = by + a->balance;
  return a->balance;
}
```

Các chỉ lệnh mã máy hoàn toàn giống nhau ở cả hai tệp đối tượng. Sự khác biệt duy nhất là bảng kiểu dữ liệu (type table). Khi có bảng kiểu, trình dịch ngược biết rằng tham số đầu tiên trỏ tới một cấu trúc rộng 32 byte mà trường tại độ lệch 4 là một số nguyên `int` 4 byte có tên `balance`. Khi không có bảng kiểu, trình dịch ngược chỉ biết rằng có một giá trị 8 byte được truyền qua `%rdi` và 4 byte tại vị trí `+4` được đọc và ghi lại.

Mọi chi tiết trong bảng kiểu đó đều có nguồn gốc rõ ràng. Độ lệch 4 xuất phát từ byte độ dời của lệnh `addl 0x4(%rax), %ecx`. Độ rộng 4 byte xuất phát từ hậu tố `l`. Tên biến xuất phát từ mục `DW_TAG_member` trong `.debug_info`. Bài học này xây dựng cả hai nửa: một lượt phân tích khôi phục những gì các chỉ lệnh chứng minh được, và một bộ đọc giải mã những gì trình biên dịch đã ghi lại. Khoảng trống giữa hai nửa đó chính là cái giá phải trả của tên kiểu dữ liệu.

```figure
re-41-struct-recovery
```

## Truy cập thành viên là một phép cộng độ dời

Tài liệu Intel SDM Tập 1, mục 3.7.5, định nghĩa địa chỉ hiệu dụng là `Offset = Base + (Index * Scale) + Displacement`, với tỷ lệ scale là 1, 2, 4, hoặc 8 và độ dời displacement là 8, 16, hoặc 32 bit. Mục 3.7.5.1 bổ sung dạng chế độ 64-bit `RIP + Displacement`. Thao tác truy cập thành viên cấu trúc sử dụng dạng đầu tiên không có chỉ số index: thanh ghi cơ sở chứa địa chỉ của đối tượng và độ dời chính là độ lệch của thành viên. Truy cập mảng bổ sung thêm thành phần chỉ số, và tỷ lệ scale hoặc một phép nhân riêng biệt sẽ mang kích thước của phần tử.

Xem xét hàm `bump` ở `-O0`. Danh sách mã máy từ `llvm-objdump -d -r` như sau:

```text
       4: 48 89 7d f8                  	movq	%rdi, -0x8(%rbp)
       e: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
      12: 03 48 04                     	addl	0x4(%rax), %ecx
      15: 89 48 04                     	movl	%ecx, 0x4(%rax)
```

Giải mã chỉ lệnh `03 48 04` theo các quy tắc từ bài học 33: `03` là `ADD r32, r/m32`. Byte ModRM `48` là `01 001 000`: `mod = 01` biểu thị toán hạng bộ nhớ với độ dời 8-bit, `reg = 001` là `%ecx`, `r/m = 000` là `%rax`. Byte độ dời là `04`. Chỉ lệnh đọc 4 byte tại `%rax + 4` và cộng chúng vào `%ecx`. Hậu tố `l` mà `llvm-objdump` in ra thể hiện kích thước toán hạng, do đó trường này rộng 4 byte. Giá trị trong `%rax` đến từ vị trí khung ngăn xếp tại `-0x8(%rbp)`, và vị trí đó nhận `%rdi`, đối số số nguyên đầu tiên theo quy ước System V. Ba dữ kiện được rút ra: tham số đầu tiên là một con trỏ, đối tượng mà nó trỏ tới có một trường 4 byte tại độ lệch 4, và hàm `bump` đọc và ghi vào trường đó.

Không có thông tin nào trong các byte này nói lên kiểu `int`, tên `balance`, hay `struct account`. Cũng không có thông tin nào cho biết đối tượng rộng bao nhiêu. Toàn bộ 28 byte còn lại của cấu trúc hoàn toàn vô hình đối với hàm `bump`.

## Các quy tắc bố cục của System V

Tài liệu x86-64 psABI, mục 3.1.2, đưa ra ba quy tắc cho các kiểu dữ liệu hỗn hợp (aggregates): "Cấu trúc và liên hiệp (unions) nhận sự căn chỉnh của thành phần có yêu cầu căn chỉnh nghiêm ngặt nhất. Mỗi thành viên được gán vào độ lệch khả dụng thấp nhất có sự căn chỉnh phù hợp. Kích thước của bất kỳ đối tượng nào luôn là bội số của mức căn chỉnh của đối tượng đó." Bảng 3.1 của tài liệu này cung cấp kích thước và mức căn chỉnh vô hướng: `char` là 1 và 1, `short` là 2 và 2, `int` là 4 và 4, `double` là 8 và 8, mọi con trỏ là 8 và 8.

Áp dụng các quy tắc này vào cấu trúc trong `account.c`:

```c
struct account {
    char kind;
    int balance;
    short code;
    double rate;
    const char *label;
};
```

| Thành viên | Kích thước | Căn chỉnh | Độ lệch thấp nhất phù hợp | Byte đệm phía trước |
|---|---|---|---|---|
| `kind` | 1 | 1 | 0 | 0 |
| `balance` | 4 | 4 | 4 | 3 (tại byte 1 đến 3) |
| `code` | 2 | 2 | 8 | 0 |
| `rate` | 8 | 8 | 16 | 6 (tại byte 10 đến 15) |
| `label` | 8 | 8 | 24 | 0 |

Thành viên cuối cùng kết thúc tại byte 32. Mức căn chỉnh nghiêm ngặt nhất là 8, và 32 là bội số của 8, do đó kích thước tổng thể là 32 và không có phần đệm ở đuôi. 9 byte trong số 32 byte là phần đệm, và psABI quy định "Nội dung của bất kỳ byte đệm nào là không xác định".

Thứ tự khai báo thành viên quyết định kích thước cấu trúc. Cùng năm thành viên đó nếu xếp theo thứ tự `double, con trỏ, int, short, char` sẽ lấp đầy các độ lệch 0, 8, 16, 20, 22 và kết thúc tại 23, làm tròn lên thành 24 byte. Hàm `sysv_layout()` trong `code/main.py` áp dụng ba quy tắc này để tính toán. Mô-đun `ctypes` của Python cũng cho ra kết quả tương tự: kích thước 32, căn chỉnh 8, các độ lệch `0, 4, 8, 16, 24`.

Bản dựng arm64 cũng cho kết quả tương tự vì chuẩn AAPCS64 mục 5.10.1 giữ nguyên các quy tắc kích thước và căn chỉnh này.

## Đọc phần còn lại của cấu trúc bằng tay

Hàm `bump` chỉ thấy một trường. Bốn hàm còn lại trong `account.c` truy cập các trường khác:

Hàm `sum_balances` đánh chỉ mục một mảng cấu trúc:

```text
       51: 48 8b 45 f8                  	movq	-0x8(%rbp), %rax
       55: 48 63 4d ec                  	movslq	-0x14(%rbp), %rcx
       59: 48 c1 e1 05                  	shlq	$0x5, %rcx
       5d: 48 01 c8                     	addq	%rcx, %rax
       60: 8b 40 04                     	movl	0x4(%rax), %eax
```

`%rax` chứa `list`, `%rcx` chứa `i` được mở rộng dấu lên 64-bit, và lệnh `shlq $0x5` nhân giá trị đó với 32 ($2^5$). Sau lệnh `addq`, `%rax` trở thành `list + 32 * i`, và lệnh nạp đọc 4 byte tại độ lệch 4 của phần tử thứ `i`. Số dịch bit này là vị trí duy nhất trong tệp đối tượng hiển thị kích thước của cấu trúc dưới dạng một con số cụ thể.

Hàm `tag` đọc hai trường kích thước hẹp:

```text
       8c: 0f be 00                     	movsbl	(%rax), %eax
       98: 0f bf 48 08                  	movswl	0x8(%rax), %ecx
```

Lệnh `movsbl` nạp một byte tại độ lệch 0 và mở rộng dấu, cho thấy trường tại 0 là số nguyên có dấu 1 byte. Lệnh `movswl` nạp hai byte tại độ lệch 8 có mở rộng dấu, chứng minh trường tại 8 là số nguyên có dấu 2 byte.

Hàm `projected` đọc số thực:

```text
       cc: f2 0f 2a 40 04               	cvtsi2sdl	0x4(%rax), %xmm0
       d5: f2 0f 59 40 10               	mulsd	0x10(%rax), %xmm0
```

Lệnh `cvtsi2sdl` chuyển đổi số nguyên có dấu 32-bit từ bộ nhớ sang số thực double, xác nhận độ lệch 4 là số nguyên 4 byte có dấu. Lệnh `mulsd` nhân với số thực double từ bộ nhớ, chứng minh độ lệch 16 chứa giá trị dấu phẩy động 8 byte.

Hàm `label_length` đọc một con trỏ rồi giải tham chiếu nó:

```text
       f3: 48 8b 40 18                  	movq	0x18(%rax), %rax
       f7: 48 63 4d f4                  	movslq	-0xc(%rbp), %rcx
       fb: 80 3c 08 00                  	cmpb	$0x0, (%rax,%rcx)
```

Lệnh `movq` đọc 8 byte tại độ lệch 24. Đứng riêng, nó có thể là số nguyên 8 byte hoặc con trỏ. Lệnh `cmpb` phía sau dùng giá trị vừa nạp làm thanh ghi cơ sở cho một toán hạng bộ nhớ. Một giá trị được dùng làm địa chỉ thì chính là con trỏ. Lệnh `cmpb` đọc 1 byte tại chỉ số `%rcx`, và ở `-O1` con trỏ tăng dần với `leaq 0x1(%rcx), %rcx`, chứng minh các phần tử được trỏ tới cách nhau 1 byte.

Tổng hợp năm hàm, cấu trúc là `{ i8 tại 0, i32 tại 4, i16 tại 8, f64 tại 16, pointer tại 24 }` với bước nhảy stride là 32. Các khoảng trống tại byte 1 đến 3 và 10 đến 15 khớp hoàn hảo với phần đệm mà psABI dự đoán.

## Lượt phân tích khôi phục kiểu

Hàm `trace_function()` trong `code/main.py` duyệt qua các chỉ lệnh của một hàm theo thứ tự địa chỉ và duy trì trạng thái trừu tượng nhỏ. Một ngăn chứa (holder) là một thanh ghi 64-bit hoặc một ô ngăn xếp như `-0x8(%rbp)`. Mỗi ngăn chứa mang một trong ba trạng thái: `Pointer` (gốc, độ lệch hằng số, tập các bước nhảy stride), `Index` (số nguyên mang tỷ lệ scale), hoặc không xác định.

Trạng thái khởi tạo với sáu thanh ghi đối số System V chứa `Pointer("param0", 0)` đến `Pointer("param5", 0)`. Các quy tắc chuyển đổi trạng thái:

- Lệnh `movq` giữa thanh ghi 64-bit và ô ngăn xếp sao chép giá trị. Ghi 32-bit hoặc hẹp hơn vào thanh ghi biến nó thành `Index` với scale 1.
- Lệnh `leaq` với cơ sở con trỏ tạo ra con trỏ có cộng thêm độ dời; với thanh ghi chỉ số index nó cộng `scale * index_scale` vào tập stride. Lệnh `leaq` dựa trên `%rbp` hoặc `%rsp` định danh một đối tượng trên khung ngăn xếp.
- Lệnh `shlq $n` trên `Index` nhân scale với $2^n$, và `imulq $k` nhân với $k$.
- Lệnh `addq` giữa con trỏ và index giữ nguyên con trỏ và thêm scale của index vào tập stride. Lệnh `addq $k` vào con trỏ làm dịch chuyển độ lệch của nó.
- Lời gọi hàm làm mất giá trị của các thanh ghi caller-saved; các thanh ghi đối số chứa con trỏ được ghi nhận là một điểm gọi hàm (call site).
- Mọi toán hạng bộ nhớ có thanh ghi cơ sở là con trỏ đều trở thành một bản ghi `Access`.

Độ rộng và phân loại kiểu lấy từ mã gợi nhớ (mnemonic): `b, w, l, q` thành 1, 2, 4, 8 byte; `movsbl` là số nguyên có dấu; `sd, ss` là số thực SSE; `cvtsi2sdl` là số nguyên chuyển thành số thực; và các lệnh giải tham chiếu con trỏ xác nhận trường nguồn là một con trỏ thực thụ.

```python
if memory.base in FRAME_REGISTERS:
    return Pointer(f"frame({memory.base}{format_signed(memory.displacement) if memory.displacement else ''})", 0)
base = env.get(memory.base)
if not isinstance(base, Pointer):
    return Unknown
strides = set(base.strides)
if memory.index is not None:
    index_value = env.get(memory.index)
    strides.add((index_value.scale if isinstance(index_value, Index) else 1) * memory.scale)
return Pointer(base.root, base.offset + memory.displacement, frozenset(strides), base.origin)
```

### Bốn nguồn bằng chứng xác định bước nhảy (Stride)

Hàm `infer_layout()` hợp nhất các truy cập từ mọi hàm nhân chứng và thu thập bước nhảy từ bốn nguồn:

| Loại bằng chứng | Nguồn gốc xuất phát | Bản dựng minh họa |
|---|---|---|
| `scale` | Một `Index` có scale > 1 cộng vào con trỏ, từ `shlq $0x5` hoặc `imulq` | `-O0`, lệnh `shlq $0x5, %rcx` tại `0x59` |
| `loop-step` | Phép tăng 64-bit trên thanh ghi cơ sở/chỉ số trong vòng lặp chứa truy cập | `-O1`, lệnh `addq $0x20, %rdx` tại `0x24` |
| `offset-delta` | Các truy cập cùng độ rộng qua một thanh ghi cơ sở trong vòng lặp, cách nhau khoảng cố định | `-O2`, tám lệnh nạp cách nhau 32 byte |
| `call-site` | Các độ lệch mà một đối tượng ngăn xếp được truyền cho các hàm nhân chứng | `-O0`, `main` truyền `rows`, `rows+0x20`, `rows+0x40` |

Bước nhảy cuối cùng là ước số chung lớn nhất (GCD) của toàn bộ các giá trị tìm thấy. Kích thước cấu trúc chính là bước nhảy khi bước nhảy này đủ lớn để chứa trường cuối cùng.

### Vòng lặp vector hóa ở mức -O2

Ở `-O2`, clang vector hóa `sum_balances` và trải rộng vòng lặp để đọc tám phần tử mỗi lượt:

```text
      3b: 48 8d 87 e4 00 00 00         	leaq	0xe4(%rdi), %rax
      50: 66 0f 6e 50 80               	movd	-0x80(%rax), %xmm2
      55: 66 0f 6e 98 60 ff ff ff      	movd	-0xa0(%rax), %xmm3
      61: 66 0f 6e 90 40 ff ff ff      	movd	-0xc0(%rax), %xmm2
      69: 66 0f 6e a0 20 ff ff ff      	movd	-0xe0(%rax), %xmm4
      7d: 66 0f 6e 10                  	movd	(%rax), %xmm2
      81: 66 0f 6e 58 e0               	movd	-0x20(%rax), %xmm3
      8a: 66 0f 6e 50 c0               	movd	-0x40(%rax), %xmm2
      8f: 66 0f 6e 60 a0               	movd	-0x60(%rax), %xmm4
      a0: 48 05 00 01 00 00            	addq	$0x100, %rax
      aa: 75 a4                        	jne	0x50 <sum_balances+0x40>
```

Tám độ dời đặt các thao tác nạp tại các độ lệch `4, 36, 68, 100, 132, 164, 196, 228` tính từ `list`. Bước tăng là `0x100` (256 byte). Một bộ phân tích ngây thơ sẽ báo cáo tám trường và cấu trúc rộng ít nhất 232 byte.

Hàm `fold_unrolled()` gập các truy cập này lại: nếu các độ lệch cách nhau `d = 32`, bước tăng vòng lặp là `k * d = 8 * 32 = 256` và có đúng `k = 8` độ lệch phân biệt, thì đây là một vòng lặp được trải rộng 8 lần. Toàn bộ tám lệnh nạp được gập về độ lệch `(offset - lowest) mod d = 4`. Nhờ đó, bộ phân tích báo cáo chính xác bước nhảy 32 và một trường duy nhất tại độ lệch 4.

## Đọc DWARF 5 trực tiếp từ các byte

Bản dựng `-g` chứa các phân đoạn gỡ lỗi. Tiêu đề đơn vị biên dịch (compilation unit header) trong DWARF 5 gồm `unit_length`, `version` (5), `unit_type` (`DW_UT_compile` = 1), `address_size` (8), và `debug_abbrev_offset`.

Các mục thông tin gỡ lỗi (DIE) sử dụng mã viết tắt trỏ vào bảng khai báo trong phân đoạn `.debug_abbrev`. Phân đoạn này định nghĩa thẻ (tag), cờ con cái (children flag), và danh sách thuộc tính kèm định dạng mã hóa (form).

Giải mã mục thành viên `balance` tại offset `0x1a3`: mã viết tắt 13 (`DW_TAG_member`), `DW_AT_name` dùng `strx1` trỏ vào chỉ số 16 của bảng chuỗi, `DW_AT_type` dùng `ref4` trỏ tới kiểu `int`, và `DW_AT_data_member_location` dùng `data1` mang giá trị `04`. Giá trị 4 byte này khớp hoàn hảo với byte độ dời trong mã máy.

Trong tệp đối tượng tái định vị, các bảng `.debug_str_offsets` và `.debug_addr` chứa giá trị 0 ban đầu và được cập nhật thông qua các bản ghi tái định vị `.rela.*`. Bộ đọc áp dụng công thức tái định vị trước khi phân tích.

Hàm `type_name()` lần theo các tham chiếu `DW_AT_type` để tái tạo chuỗi tên kiểu hoàn chỉnh, chẳng hạn biến đổi một chuỗi DIE thành `const char *`.

## So sánh hai bố cục

Hàm `compare_layouts()` đối chiếu các thành viên DWARF với các trường được suy luận từ mã máy:

| Bản dựng | Số truy cập qua tham số 1 | Số trường | Bằng chứng bước nhảy | Thành viên khớp | Kích thước |
|---|---|---|---|---|---|
| `-O0` | 11 | 5 | `scale 32`, `call-site 32`, `call-site 64` | 5/5 | 32, khớp |
| `-O1` | 9 | 5 | `loop-step 32` | 5/5 | 32, khớp |
| `-O2` | 17 | 5 (sau khi gập) | `offset-delta 32`, `unrolled 32`, `loop-step 256`, v.v. | 5/5 | 32, khớp |

Các khoảng trống suy luận tại byte 1-3 và 10-15 ở mọi bản dựng khớp hoàn toàn với các byte đệm trong DWARF. Mọi trường suy luận đều nằm tại độ lệch là bội số của kích thước chính nó (thỏa mãn căn chỉnh tự nhiên).

Những yếu tố không thể khôi phục từ chỉ lệnh mã máy: tên thành viên, từ khóa hạn định `const`, và kiểu cụ thể của đối tượng mà con trỏ trỏ tới (chỉ biết các phần tử cách nhau 1 byte, có thể là `char`, `unsigned char` hoặc `bool`).

## Đối tượng C++ và Bảng hàm ảo

Tệp `shapes.cpp` định nghĩa lớp cơ sở `Shape` có hàm hủy ảo và hàm thuần ảo `area()`, cùng lớp dẫn xuất `Rect`. Chuẩn Itanium C++ ABI chi phối cách bố trí của chúng.

### Tên biến đổi (Mangled names)

Tên biến đổi bắt đầu bằng tiền tố `_Z`. Ký hiệu `_ZNK4Rect4areaEv` giải mã thành: `_Z`, `N` (tên lồng nhau), `K` (thành viên const), `4Rect` và `4area` (tên kèm độ dài), `E` (kết thúc lồng), `v` (danh sách tham số rỗng rỗng) -> `Rect::area() const`. Bộ giải mã `Demangler` tự viết giải mã chính xác 19 tên ký hiệu trong tệp.

### Bảng hàm ảo (vtable) từ các mục tái định vị

Bảng vtable gồm: độ lệch tới đỉnh (offset-to-top), con trỏ typeinfo, và danh sách các con trỏ hàm ảo. Điểm địa chỉ (address point) mà con trỏ vptr trỏ vào nằm tại vị trí của con trỏ hàm đầu tiên (sau offset-to-top và typeinfo, tức độ lệch +16).

Biểu tượng `_ZTV4Rect` rộng 48 byte (sáu ô 8-byte). Các ô chức năng được điền thông qua phân đoạn tái định vị: ô 1 là typeinfo `_ZTI4Rect`, ô 2 là hàm hủy hoàn chỉnh, ô 3 là hàm hủy xóa, ô 4 là `area()`, và ô 5 là `name()`.

### Lời gọi hàm ảo

Hàm `describe` ở `-O0` thực hiện lời gọi ảo:

```text
       f: 48 8b 7d f8                  	movq	-0x8(%rbp), %rdi
      13: 48 8b 07                     	movq	(%rdi), %rax
      16: ff 50 10                     	callq	*0x10(%rax)
```

Lệnh `movq (%rdi), %rax` nạp con trỏ vptr nằm ở 8 byte đầu của đối tượng. Lệnh `callq *0x10(%rax)` gọi qua độ lệch `0x10` tính từ vptr. `0x10 / 8 = 2`, tương ứng với ô số 2 (slot 2) trong bảng hàm ảo. Phân giải slot 2 cho ra hai ứng viên: `Rect::area() const` trong vtable của `Rect` hoặc `__cxa_pure_virtual` trong vtable của `Shape`. DWARF xác nhận chính xác ô số 2 qua thuộc tính `DW_AT_vtable_elem_location (DW_OP_constu 0x2)`.

Ở mức `-O1`, do đối tượng `Rect` được cấp phát cục bộ trong `main`, trình biên dịch đã tối ưu hóa trực tiếp các lời gọi tại thời điểm biên dịch và nhúng nội tuyến chúng, dẫn tới việc bảng vtable hoàn toàn không được phát sinh trong tệp đối tượng.

## Cách các công cụ hiển thị kiểu dữ liệu khôi phục

Các công cụ phân tích nhị phân (Ghidra, IDA, Binary Ninja, radare2, rizin) đều có cơ chế quản lý kiểu dữ liệu riêng. Khi có DWARF, Ghidra nhập trực tiếp cấu trúc `/DWARF/account.c/account size=32 align=8` và hiển thị mã dịch ngược tự nhiên `a->balance`. Khi không có DWARF, Ghidra hiển thị các phép truy cập bộ nhớ thô `*(int *)(param_1 + 4)` và gán kiểu `long param_1`.

Bằng cách sử dụng bộ phân tích trong bài học này, bạn có thể trích xuất chính xác cấu trúc trường và nạp ngược lại vào bộ quản lý kiểu dữ liệu của các công cụ dịch ngược để tái tạo mã nguồn rõ ràng.

## Nghiên cứu suy luận kiểu dữ liệu

Bài toán khôi phục kiểu dữ liệu từ mã máy là trọng tâm của nhiều nghiên cứu học thuật:
- **TIE (NDSS 2011)**: Khởi xướng việc sinh ràng buộc và giải ràng buộc kiểu trên biểu diễn trung gian.
- **Retypd (PLDI 2016)**: Bổ sung đa hình, kiểu đệ quy và khôi phục định danh `const`.
- **OSPREY (IEEE S&P 2021)**: Phân tích xác suất trên các biến bộ nhớ.
- **DIRTY (USENIX Security 2022) & ReSym (CCS 2024)**: Sử dụng các mô hình học sâu và mô hình ngôn ngữ lớn để dự đoán tên và kiểu cấu trúc.
- **RecStruct (USENIX Security 2026)**: Khôi phục cấu trúc lồng nhau trực tiếp không cần decompiler qua thực thi ký hiệu không phụ thuộc luồng.

## Xây dựng hệ thống

Chạy chương trình minh họa:

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
```

Đầu ra là một báo cáo JSON gồm năm phần: `dwarf` chứa thông tin giải mã DWARF 5; `oracles` chứa so sánh giữa các nguồn chuẩn (psABI, ctypes, DWARF); `builds` chứa các truy cập và bố cục suy luận cho ba mức tối ưu; `cxx` chứa thông tin giải mã C++ (vtable, typeinfo, virtual call); và `tools` ghi nhận dữ liệu từ các công cụ khác.

Cấu trúc định dạng kỳ vọng:

```json
{
  "dwarf": {"unit": {"unitLength": 489, "version": 5, "unitType": "DW_UT_compile", "addressSize": 8, "abbrevOffset": 0}, "...": "..."},
  "oracles": {"agree": true, "psabiRules": {"size": 32, "align": 8, "...": "..."}, "...": "..."},
  "builds": {
    "O0": {"layout": {"stride": 32, "size": 32, "gaps": [1, 2, 3, 10, 11, 12, 13, 14, 15], "...": "..."}, "comparison": {"matches": 5, "members": 5, "...": "..."}},
    "O1": {"...": "..."},
    "O2": {"...": "..."}
  },
  "cxx": {"virtualCalls": [{"slot": 2, "candidates": {"Shape": "__cxa_pure_virtual", "Rect": "_ZNK4Rect4areaEv"}, "...": "..."}], "...": "..."}
}
```

Hãy đọc mã nguồn trong `code/main.py` theo trình tự sau:

1. `ElfObject` đọc cấu trúc tệp ELF và tái định vị các phân đoạn gỡ lỗi.
2. `parse_abbreviations()` và `DwarfInfo._parse_unit()` giải mã các mục DWARF và giải quyết bảng chuỗi/địa chỉ.
3. `DwarfInfo.struct_layouts()` và `functions()` trích xuất thông tin cấu trúc và hàm gỡ lỗi.
4. `parse_listing()` và `access_width()` phân tích chỉ lệnh và độ rộng truy cập từ mã máy.
5. `trace_function()`, `fold_unrolled()`, và `infer_layout()` triển khai máy trạng thái trừu tượng và suy luận bố cục cấu trúc.
6. `compare_layouts()` đối chiếu kết quả suy luận với DWARF.
7. `Demangler`, `read_vtable()`, và `find_virtual_calls()` phân tích các thành phần hướng đối tượng C++.

## Sử dụng hệ thống

Quy trình sử dụng trên một tệp đối tượng mới:

1. Trích xuất danh sách dịch ngược bằng `llvm-objdump -d -r`.
2. Chạy `python3 code/main.py infer LISTING.txt`. Xem xét dòng nhân chứng (witnesses) để xác định các hàm truy cập vào tham số.
3. Xem xét từng trường và số lượt truy cập. Một trường chỉ có một lượt truy cập cần được đánh dấu là suy luận sơ bộ.
4. Kiểm tra bằng chứng bước nhảy stride. Tìm kiếm dấu hiệu vòng lặp trải rộng và kiểm tra thao tác gập unroll.
5. Nếu có bản dựng `-g`, chạy đối chiếu so sánh để định vị các thành phần không có bằng chứng mã máy hoặc các truy cập bất thường.
6. Nhập bố cục đã khôi phục vào trình dịch ngược để tái cấu trúc mã giả.

## Phòng thí nghiệm tương tác

### Lab A: Giải mã mục thành viên DWARF

Khởi động Python trong thư mục `code`:

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types/code
python3 -q
```

Thực thi:

```python
from main import *
elf = load_object("account-x86_64-O0-g.o")
dwarf = DwarfInfo(elf)
dwarf.header()
elf.section(".debug_info").data[0x1a3:0x1ac].hex(" ")
dwarf.dies[0x1a3].attributes
dwarf.type_name(dwarf.type_of(dwarf.dies[0x1a3]))
```

Giải mã chín byte trên giấy theo cấu trúc viết tắt 13 trước khi xem kết quả từ từ điển.

### Lab B: Lần vết một hàm

```python
listing = load_listing("account-x86_64-O0-g.objdump.txt")
trace = trace_function("label_length", listing["label_length"])
[(hex(a.address), a.root, a.offset, a.width, a.kind) for a in trace.accesses]
trace.pointer_uses
trace_function("sum_balances", listing["sum_balances"]).loops
```

Xác định truy cập nào chứng minh độ lệch 24 là một con trỏ. Chạy lại trên bản dựng `-O1` để tìm lệnh tăng con trỏ `leaq 0x1(%rcx), %rcx`.

### Lab C: Gập vòng lặp vector hóa

```python
o2 = load_listing("account-x86_64-O2-g.objdump.txt")
trace = trace_function("sum_balances", o2["sum_balances"])
sorted({a.offset for a in trace.accesses})
folded, evidence = fold_unrolled(trace, trace.accesses)
sorted({a.offset for a in folded}), [(e.kind, e.value, e.text) for e in evidence]
```

Quan sát cách 8 độ lệch phân tán được gập chính xác về độ lệch 4.

### Lab D: Bảng hàm ảo và Tên biến đổi

```python
shapes = load_object("shapes-x86_64-O0-g.o")
read_vtable(shapes, "_ZTV4Rect")
read_typeinfo(shapes, "_ZTI4Rect")
demangle("_ZNK4Rect4areaEv"), demangle("_Z8describeRK5Shapei")
find_virtual_calls(load_listing("shapes-x86_64-O0-g.objdump.txt"), [read_vtable(shapes, "_ZTV5Shape"), read_vtable(shapes, "_ZTV4Rect")])
```

Tính toán ô nhớ từ lệnh `callq *0x10(%rax)` và đối chiếu với thông tin DWARF.

## Bài tập thực hành

Khôi phục cấu trúc ngăn xếp trong hàm `main` ở `-O1`.

Yêu cầu thực hiện:

- Mở rộng `trace_function()` để hỗ trợ gốc con trỏ xuất phát từ một vùng ngăn xếp, bắt đầu từ các lệnh lưu `movaps` khởi tạo mảng `rows` 96 byte.
- Gán các truy cập nội tuyến `0x24(%rsp)`, `0x40(%rsp)`, `0x48(%rsp)`, `0x30(%rsp)`, và `0x38(%rsp)` vào gốc ngăn xếp này và tính độ lệch của chúng theo modulo bước nhảy.
- Dẫn xuất bước nhảy 32 từ vòng lặp tại `0x120` đến `0x12c`, nơi `%r14` bắt đầu từ 4 và tăng mỗi lần `0x20` cho đến `0x64`.
- So sánh kết quả với DWARF để đạt được năm trường khớp.
- Viết kiểm thử tự động xác nhận lỗi nếu gốc ngăn xếp bị gán nhầm thành tham số hàm.

Thành công đạt được khi `python3 code/main.py infer` trên danh sách `-O1` nhận diện `main` là nhân chứng với gốc ngăn xếp và đối chiếu khớp cả năm thành viên từ riêng hàm `main`.

## Thành phẩm bài học

Bài học này bao gồm tệp `outputs/skill-recovering-data-and-types.md`. Sử dụng nó để biến một danh sách dịch ngược và bản dựng gỡ lỗi thành bố cục cấu trúc kèm bằng chứng cho từng trường, bước nhảy kèm nguồn gốc, so sánh chi tiết với DWARF, và danh sách các yếu tố không thể xác định từ mã máy.

## Kiểm chứng thực tế

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/41-recovering-data-and-types
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình kiểm chứng cần chứng minh:

- Tiêu đề đơn vị DWARF, 14 mục viết tắt, và năm độ lệch thành viên khớp với `llvm-dwarfdump`
- Tên thành viên phân giải chính xác qua bảng `.debug_str_offsets` đã tái định vị
- Sáu hàm gỡ lỗi mang đúng phạm vi địa chỉ và vị trí khung ngăn xếp
- Quy tắc psABI, `ctypes`, DWARF, và bản chạy thực tế đều cho ra kích thước 32 và các độ lệch `0, 4, 8, 16, 24`
- Bộ phân tích danh sách mã máy giải mã đúng toán hạng bộ nhớ, độ rộng và dấu
- Bước nhảy stride được xác định đúng qua `shlq` ở `-O0`, `addq` ở `-O1`, và phép gập unroll ở `-O2`
- Thao tác giải tham chiếu trong `label_length` chứng minh độ lệch 24 là con trỏ
- Toàn bộ ba bản dựng đều khớp 5/5 thành viên cấu trúc với kích thước 32 và khoảng trống trùng khớp phần đệm DWARF
- Bộ giải mã tên C++ hoàn trả chính xác 19 chuỗi biểu tượng
- Lời gọi ảo tại `0x16` phân giải chính xác vào ô số 2 (slot 2), khớp với vtable và DWARF

## Các kịch bản lỗi trong môi trường thực tế

| Tình huống lỗi | Báo cáo từ công cụ yếu | Điều phân tích chuẩn phải chứng minh |
|---|---|---|
| Tám lệnh nạp trong vòng lặp trải rộng | Báo cấu trúc có tám trường từ 4 đến 228 | Vòng lặp bước nhảy 256 trải rộng 8 lần độ lệch 32; chỉ có một trường tại độ lệch 4 |
| Đọc bước nhảy từ một lệnh tăng duy nhất | Báo kích thước cấu trúc là 256 từ `addq $0x100` | Bước nhảy là ước số chung lớn nhất của mọi nguồn, kết hợp với phần đuôi vô hướng `addq $0x20` |
| Trường 8-byte bị coi mặc định là con trỏ | Gán `void *` tại độ lệch 24 | Phải có truy cập giải tham chiếu sau đó làm cơ sở địa chỉ, nếu không phải giữ là `int64` |
| Con trỏ bị nhầm thành số nguyên | Gán `long` tại độ lệch 24 | Thao tác giải tham chiếu tại `0xfb` là bằng chứng xác thực con trỏ |
| So sánh giá trị hẹp | Gán `char` có dấu từ lệnh `cmpb` | Lệnh `cmpb` không mang dấu; chỉ có `movsbl` hoặc `movzbl` mới xác định dấu |
| Mặc định năm hàm dùng chung một cấu trúc | Gộp cấu trúc khi chưa có bằng chứng alias | Phải chứng minh qua điểm gọi hàm trong `main` truyền cùng một đối tượng ngăn xếp |
| Đọc chuỗi DWARF trực tiếp theo chỉ số | Tên biến bị rác trong tệp tái định vị | Phải tính qua `str_offsets_base` cộng mục bảng `.debug_str_offsets` đã tái định vị |
| Thiếu bảng hàm ảo ở mức `-O1` | Báo lớp không có hàm ảo | Do trình biên dịch đã tối ưu nhúng nội tuyến lời gọi; bản `-O0` chứa đầy đủ vtable |
| Đọc sai gốc bảng hàm ảo | Bị lệch 2 ô trong vtable | Điểm địa chỉ nằm sau offset-to-top và typeinfo (cách đầu biểu tượng 16 byte) |

## Quy tắc vận hành

Hãy khôi phục từng trường từ các thao tác truy cập và bước nhảy từ các vòng lặp, đồng thời gắn kèm chỉ lệnh làm bằng chứng cho từng suy luận. Luôn gập các vòng lặp được trải rộng trước khi tin tưởng vào một độ lệch vượt quá bước nhảy cơ sở. Đọc DWARF thông qua bảng viết tắt và các bảng đã được tái định vị, tuyệt đối không đọc chỉ số thô vào phân đoạn chuỗi. So sánh từng trường và ghi nhận rõ ràng mọi thành viên thiếu bằng chứng, mọi con trỏ chưa được chứng minh, và mọi tên gọi dưới dạng chưa xác định. Trong đối tượng C++, đọc vtable từ các mục tái định vị và xác định ô nhớ từ lời gọi hàm, sau đó dùng DWARF để tái kiểm chứng.

## Tài liệu đọc thêm

- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, section 3.1.2: Data Representation](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64), section 5.10: Composite Types](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1, section 3.7.5: Specifying an Offset](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [DWARF Debugging Information Format Version 5, sections 2.1, 5.7.6, 7.5.1.1, 7.5.3, 7.5.6](https://dwarfstd.org/doc/DWARF5.pdf)
- [Itanium C++ ABI, sections 2.5 (Virtual Table Layout), 2.9 (Run-Time Type Information), 5.1 (External Names)](https://itanium-cxx-abi.github.io/cxx-abi/abi.html)
- [Microsoft Learn: Decorated names](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names?view=msvc-170) and [Name Decoration](https://learn.microsoft.com/en-us/cpp/error-messages/tool-errors/name-decoration?view=msvc-170)
- [llvm-dwarfdump command guide](https://llvm.org/docs/CommandGuide/llvm-dwarfdump.html) and [llvm-cxxfilt command guide](https://llvm.org/docs/CommandGuide/llvm-cxxfilt.html)
- [Python ctypes: Structure layout, sizeof, alignment](https://docs.python.org/3/library/ctypes.html)
- [Ghidra Data Type Manager](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Base/src/main/help/help/topics/DataTypeManagerPlugin/data_type_manager_description.htm), [Decompiler Concepts](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerConcepts.html), [Decompiler Options](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/main/help/help/topics/DecompilePlugin/DecompilerOptions.html), and [Ghidra 12.1.4 release](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.1.4_build)
- [IDA 9.1 user guide: Types](https://docs.hex-rays.com/9.1/user-guide/types), [IDA 9.4 release post](https://hex-rays.com/blog/ida-9.4-release-a-new-dyld-shared-cache-swift-analysis-new-teams-add-on-and-more-), and [IDA Free](https://hex-rays.com/ida-free)
- [Binary Ninja user guide: Types](https://docs.binary.ninja/guide/types/) and [Binary Ninja 6.0 Krypton](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [radare2 6.2.4 release](https://github.com/radareorg/radare2/releases/tag/6.2.4) and [rizin v0.9.1 release](https://github.com/rizinorg/rizin/releases/tag/v0.9.1)
- [GDB manual: Examining the Symbol Table, ptype /o](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Symbols.html)
- [pyelftools on PyPI](https://pypi.org/project/pyelftools/)
- [Lee, Avgerinos, Brumley, "TIE: Principled Reverse Engineering of Types in Binary Programs" (NDSS 2011)](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, "Polymorphic Type Inference for Machine Code" (PLDI 2016)](https://arxiv.org/abs/1603.05495) and [the retypd repository](https://github.com/GrammaTech/retypd)
- [Zhang et al., "OSPREY: Recovery of Variable and Data Structure via Probabilistic Analysis for Stripped Binary" (IEEE S&P 2021)](https://ieeexplore.ieee.org/document/9519451/)
- [Chen et al., "Augmenting Decompiler Output with Learned Variable Names and Types" (USENIX Security 2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/chen-qibin)
- [Xie et al., "ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries" (CCS 2024)](https://dl.acm.org/doi/10.1145/3658644.3670340)
- [Chen et al., "RecStruct" (USENIX Security 2026)](https://www.usenix.org/conference/usenixsecurity26/presentation/chen-yuxin)
