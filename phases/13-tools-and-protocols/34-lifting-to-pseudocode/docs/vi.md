# Nâng Cấp Lên Mã Giả: Trung Gian IR, Dạng SSA và Tái Cấu Trúc Mã Từ Đầu

> Đầu ra của trình dịch ngược mã thực chất là một khẳng định với quá trình suy dẫn bị ẩn giấu. Khi bạn tự tay xây dựng quá trình suy dẫn đó, từng dòng mã trong đầu ra sẽ trở thành điều mà bạn hoàn toàn có thể kiểm chứng.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow)
**Time:** ~150 phút

## Mục tiêu Học tập

- Nâng cấp (lifting) các lệnh x86-64 đã giải mã lên một biểu diễn trung gian (IR - Intermediate Representation) tinh gọn với các cờ, lệnh nạp (loads), lưu (stores), rẽ nhánh, gọi hàm và trả về tường minh.
- Chuyển đổi các ô nhớ khung ngăn xếp (frame slots) thành các biến cục bộ và các thanh ghi đối số theo chuẩn System V thành các tham số hàm.
- Xây dựng dạng gán tĩnh đơn tối thiểu (minimal SSA - Static Single Assignment) với các hàm phi ($\phi$) trên các ranh giới thống trị (dominance frontiers), sau đó thực hiện thay thế biểu thức, rút gọn và loại bỏ mã chết.
- Khôi phục các cấu trúc `if`, `if-else`, `while`, và `do-while` từ đồ thị luồng điều khiển bằng các nút hậu thống trị (post-dominators) và các cạnh ngược (back edges).
- Giải thích bản chất phép nhân với hằng số `0x66666667` tương đương với phép chia cho 10 và chứng minh tính tương đương đại số này.
- Đọc từng dòng mã giả do Ghidra sinh ra cho cùng các hàm và gọi tên chính xác lệnh thực thi, hoặc sự suy đoán, đằng sau mỗi điểm sai khác.

## Vấn đề Cần Giải Quyết

Bạn yêu cầu rea trích xuất mã giả cho hàm `percent_change` trong tệp `ledger-x86_64-O1.o`. Trình dịch ngược Ghidra 12.1.4 trả về kết quả sau khoảng 10 giây:

```c
undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)

{
  long lVar1;
  undefined1 auVar2 [16];
  undefined1 auVar3 [16];
  
  if (param_1 != 0) {
    lVar1 = (long)((param_2 - param_1) * 100);
    auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
    auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
    return auVar3;
  }
  auVar2._8_8_ = 0;
  auVar2._0_8_ = param_3;
  return auVar2 << 0x40;
}
```

Hàm nguồn trong tệp C gốc chỉ dài 4 dòng. Nó nhận hai tham số kiểu `int` và trả về một giá trị kiểu `int`. Thế nhưng Ghidra lại báo cáo một kiểu trả về rộng tới 16 byte cùng một tham số thứ ba hoàn toàn không tồn tại trong mã nguồn gốc! Toàn bộ 16 byte mã máy là hoàn toàn chính xác, và Ghidra cũng đã giải mã từng byte một cách chuẩn xác. Chữ ký hàm ở đây thực chất là một sự suy đoán, và không có dấu hiệu nào trong văn bản cảnh báo cho người đọc biết điều đó.

rea trả về đoạn văn bản này dưới dạng kết quả của `procedure_pseudo_code` kèm theo một lưu ý giới hạn: "Pseudocode is Ghidra decompiler output, not original source and not text-equivalent to Hopper output" (Mã giả là đầu ra của trình dịch ngược Ghidra, không phải mã nguồn gốc và không tương đương văn bản với đầu ra của Hopper). Công cụ `compare_functions` so sánh mã giả dưới dạng các dòng văn bản đơn thuần. Bản ghi thuộc tính của chính nó thừa nhận "Assembly is opaque provider text" (Mã máy là văn bản mờ đục từ nhà cung cấp) và việc chuẩn hóa tái định vị là không khả thi. rea không thể nhìn thấu bên trong văn bản, và một AI agent nếu chỉ đọc lướt văn bản đó cũng sẽ hoàn toàn mù tịt.

Bản ghi sự cố DX-Ball của rea cho thấy cái giá phải trả. Kết quả dịch ngược đầu tiên trả về `FUN_00406400(void)` gọi hàm `__ftol()`. Khung nhìn mã lệnh lại chỉ ra một đầu vào nằm trên ngăn xếp và một biểu thức toán học x87. Agent chỉ có thể tiếp tục phân tích bởi vì hai khung nhìn này mâu thuẫn trực tiếp với nhau.

Bài học này sẽ tự tay xây dựng toàn bộ quy trình suy dẫn mà trình dịch ngược đã giấu kín. Bạn sẽ bắt đầu từ 56 lệnh máy mà bài học 33 đã giải mã, nâng cấp chúng lên mã trung gian IR, đổi tên biến theo dạng SSA, tối ưu hóa rút gọn, và tái cấu trúc điều khiển. Sau đó, bạn sẽ đặt mã giả do chính bạn tạo ra cạnh mã giả của Ghidra cho các hàm `percent_change`, `grade`, và `total`, rồi giải trình rõ ràng cho từng điểm khác biệt. Các bài kiểm thử sẽ biên dịch đầu ra của bạn bằng clang và so sánh kết quả thực thi đối chiếu với mã nguồn gốc trên một lưới dữ liệu đầu vào toàn diện.

````figure lifting-stages-percent-change
````

## Fixture và Chân Lý Chuẩn (The Fixture and the Oracle)

Tệp `code/fixtures/ledger.c` là fixture được chia sẻ xuyên suốt phần này của khóa học:

```c
int percent_change(int before, int after) {
    if (before == 0) return 0;
    return (after - before) * 100 / before;
}

int grade(int score) {
    switch (score / 10) {
    case 10:
    case 9: return 4;
    case 8: return 3;
    case 7: return 2;
    case 6: return 1;
    default: return 0;
    }
}

int total(const int *values, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) sum += values[i];
    return sum;
}
```

Hàm `grade_label` cũng nằm trong tệp và trong tệp đối tượng, và bản demo cũng sẽ nâng cấp nó. Ba hàm ở trên là các hàm được kiểm thử trọng tâm trong bài học này.

Tệp đối tượng được biên dịch bằng lệnh `clang -target x86_64-unknown-linux-gnu -O1 -fno-asynchronous-unwind-tables -c ledger.c`. Tệp đối tượng biên dịch ở mức `-O0` cũng được đặt bên cạnh để phục vụ phần phân tích các ô nhớ ngăn xếp. Tệp `code/fixtures/README.md` liệt kê cả hai lệnh biên dịch, mã SHA-256 của từng tệp đối tượng, và lệnh rea đã dùng để tạo ra các fixture Ghidra. Quá trình biên dịch có khả năng tái tạo chuẩn xác từng byte trên chuỗi công cụ đã ghim.

Chân lý chuẩn (oracle) cho bộ giải mã là lệnh `llvm-objdump -d -r`. Đầu ra văn bản của nó cho cả hai tệp đối tượng được cam kết dưới dạng `ledger-x86_64-O1.objdump.txt` và `ledger-x86_64-O0.objdump.txt`. Một bài kiểm thử sẽ phân tích các tệp này và so sánh từng lệnh theo địa chỉ, byte nhị phân và cú pháp AT&T. Phép kiểm tra này bao trùm toàn bộ 56 lệnh ở mức `-O1` và mọi lệnh ở mức `-O0`. Khi `llvm-objdump` được cài đặt tại đường dẫn đã ghim, bài kiểm thử thứ hai sẽ chạy trực tiếp công cụ này và so sánh lại một lần nữa.

Dưới đây là hàm `percent_change` do oracle in ra:

```text
0000000000000000 <percent_change>:
       0: 85 ff                        	testl	%edi, %edi
       2: 74 09                        	je	0xd <percent_change+0xd>
       4: 29 fe                        	subl	%edi, %esi
       6: 6b c6 64                     	imull	$0x64, %esi, %eax
       9: 99                           	cltd
       a: f7 ff                        	idivl	%edi
       c: c3                           	retq
       d: 31 c0                        	xorl	%eax, %eax
       f: c3                           	retq
```

## Giai Đoạn 1: Từ Byte Nhị Phân Lên Lệnh Máy

Bài học 33 đã xây dựng bộ giải mã hoàn chỉnh. Bài học này duy trì một bản sao thu nhỏ độc lập để mã nguồn có thể chạy tự chủ. Bản sao này bao quát chính xác các opcode mà clang đã phát ra cho `ledger.c` ở các mức `-O0` và `-O1`. Bất kỳ opcode nào khác sẽ kích hoạt ngoại lệ `DecodeError` kèm địa chỉ và byte opcode lạ.

Bộ giải mã tuân thủ thứ tự byte theo Intel SDM Tập 2, Mục 2.1: Các tiền tố đứng đầu, tiếp theo là opcode, byte ModR/M, byte SIB, trường độ dời (displacement), và cuối cùng là giá trị tức thời (immediate). Mục 2.2.1 bổ sung tiền tố REX (một byte từ `0x40` đến `0x4F`) nằm ngay trước opcode. REX.W chọn toán hạng 64-bit, REX.R mở rộng trường `reg`, REX.X mở rộng trường index của SIB, và REX.B mở rộng trường `r/m` hoặc base của SIB. Mục 2.1.3 định nghĩa ba trường ModR/M (`mod`, `reg`, `r/m`) và ba trường SIB (`scale`, `index`, `base`). Trong chế độ 64-bit, `mod = 00` đi cùng `r/m = 101` biểu thị độ dời 32-bit tương đối so với lệnh tiếp theo (Mục 2.2.1.6).

| Các byte Opcode | Định dạng | Ví dụ trong fixture |
|---|---|---|
| `01 29 31 39 85 89` | `op Ev, Gv` | `29 fe` `subl %edi, %esi` |
| `03 2b 3b 8b 8d` | `op Gv, Ev` | `03 04 97` `addl (%rdi,%rdx,4), %eax` |
| `63` có REX.W | `movsxd Gv, Ev` | `48 63 c7` `movslq %edi, %rax` |
| `69` và `6b` | `imul Gv, Ev, imm` | `48 69 c0 67 66 66 66` |
| `83 /0 /5 /7` | `add sub cmp Ev, imm8` | `83 c1 fa` `addl $-0x6, %ecx` |
| `c1 /5 /7` | `shr sar Ev, imm8` | `48 c1 f9 22` `sarq $0x22, %rcx` |
| `f7 /7` | `idiv Ev` | `f7 ff` `idivl %edi` |
| `ff /0 /4` | `inc Ev`, `jmp Ev` | `48 ff c2` `incq %rdx` |
| `70` đến `7f`, `eb`, `e8` | `jcc rel8`, `jmp rel8`, `call rel32` | `74 09` `je 0xd` |
| `99`, `c3`, `90`, `0f 1f` | `cltd`, `retq`, `nop` | `66 0f 1f 44 00 00` `nopw (%rax,%rax)` |

Lệnh `leaq (%rip), %rcx` tại vị trí `grade+0x23` dài 7 byte với độ dời bằng 0. Tệp đối tượng mang bản ghi tái định vị `R_X86_64_PC32 .rodata+0x10` tại độ lệch `0x36` (vị trí trường độ dời). Bộ nâng cấp mã sẽ giải quyết tham chiếu này ở Giai đoạn 3.

## Giai Đoạn 2: Khối Cơ Bản và Nút Thống Trị

Khối cơ bản (basic block) là một chuỗi các câu lệnh có duy nhất một điểm nhập ở đầu và duy nhất một điểm thoát ở cuối. Đó là định nghĩa chuẩn tắc từ Allen (1970) mà mọi công trình nghiên cứu về trình dịch ngược đều tái sử dụng. Hàm `build_blocks()` thu thập các leader: điểm nhập của hàm, mọi đích nhảy nằm bên trong hàm, và mọi địa chỉ nằm ngay sau một lệnh nhảy hoặc lệnh trở về. Mỗi khối nhận danh sách các khối kế vị từ lệnh cuối cùng của nó: Lệnh nhảy có điều kiện có hai khối kế vị, lệnh `jmp` có một, lệnh `ret` không có khối kế vị nào, và một khối kết thúc bằng lệnh bình thường sẽ đi thẳng (fallthrough) sang leader tiếp theo.

Hàm `total` ở mức `-O1` có 5 khối cơ bản:

| Khối | Các lệnh máy | Các khối kế vị |
|---|---|---|
| `0x40` | `testl %esi, %esi` sau đó `jle 0x5c` | `0x5c`, `0x44` |
| `0x44` | `movl %esi, %ecx`, `xorl %edx, %edx`, `xorl %eax, %eax`, `nopw` | `0x50` |
| `0x50` | `addl (%rdi,%rdx,4), %eax`, `incq %rdx`, `cmpq %rdx, %rcx`, `jne 0x50` | `0x50`, `0x5b` |
| `0x5b` | `retq` | thoát hàm |
| `0x5c` | `xorl %eax, %eax` sau đó `retq` | thoát hàm |

Hàm `immediate_dominators()` triển khai thuật toán lặp của Cooper, Harvey và Kennedy (2001). Thuật toán duyệt qua các khối theo thứ tự hậu duyệt đảo ngược (reverse postorder) và giao các nút thống trị của các khối tiền nhiệm đã xử lý cho đến khi trạng thái hội tụ. Trên các đồ thị kích thước nhỏ này, nó kết thúc chỉ sau hai lượt lặp. Đối với hàm `total`, nó xác định rằng `0x44` thống trị `0x50`, và `0x50` tự thống trị chính nó qua cạnh ngược (back edge), đây chính là dấu hiệu để Giai đoạn 8 tìm ra vòng lặp.

Cùng hàm này được chạy thêm hai lần nữa: `dominance_frontiers()` sử dụng nó để đặt các hàm phi trong SSA ở Giai đoạn 5; và `post_dominators()` chạy nó trên đồ thị đảo ngược với một nút thoát ảo để xác định các điểm hội tụ rẽ nhánh ở Giai đoạn 8.

## Giai Đoạn 3: Nâng Cấp Lên Mã Trung Gian IR Tinh Gọn

Hệ thống biểu diễn trung gian (IR) gồm 6 loại câu lệnh: `Assign` (gán), `Store` (lưu bộ nhớ), `Branch` (rẽ nhánh), `Jump` (nhảy tuyệt đối), `IndirectJump` (nhảy gián tiếp), và `Return` (trả về). Các biểu thức gồm có hằng số, biến, phép toán nhị phân có độ rộng bit, ép kiểu (`sext`, `zext`, `trunc`), và phép nạp `Load`. Năm cấu trúc phụ trợ khác chỉ tồn tại tạm thời để được rút gọn ở các bước sau: `Flags`, `Cond`, `Concat`, `SignFill`, và `Symbolic`.

Các thanh ghi được coi là các biến được đặt tên theo họ 64-bit của chúng. Phép ghi vào `%eax` là một lệnh `Assign` vào biến `rax` với độ rộng 32-bit; và phép đọc từ `%eax` là phép đọc biến `rax` tại độ rộng 32-bit. Trong chế độ 64-bit, việc ghi 32-bit sẽ tự động mở rộng zero toàn bộ nửa trên của thanh ghi, do đó phép đọc 64-bit sau đó đối với cùng giá trị sẽ là một phép `zext`. Bộ nâng cấp chưa biết định nghĩa nào vươn tới được điểm đọc, vì vậy nó ghi lại độ rộng đọc và để SSA chèn phép ép kiểu ở Giai đoạn 5.

| Lệnh máy | Các câu lệnh IR tương ứng |
|---|---|
| `subl %edi, %esi` | `flags = cmp(esi, edi)` tiếp theo `esi = esi - edi` |
| `testl %edi, %edi` | `flags = test(edi, edi)` |
| `je 0xd` | `if je(flags) goto 0xd else goto 0x4` |
| `xorl %eax, %eax` | `eax = 0` |
| `cltd` | `edx = signfill(eax)` |
| `idivl %edi` | `dividend = edx:eax`, `eax = dividend / edi`, `edx = dividend % edi` |
| `movslq %edi, %rax` | `rax = sext64(edi)` |
| `sarq $0x22, %rcx` | `rcx = rcx >>s 34` |
| `shrq $0x3f, %rax` | `rax = rax >>u 63` |
| `addl (%rdi,%rdx,4), %eax` | `flags = add(eax, load32(rdi + rdx * 4))` rồi `eax = eax + load32(rdi + rdx * 4)` |
| `leaq (%rip), %rcx` kèm tái định vị | `rcx = &.rodata+0x14` |
| `pushq %rbp`, `movq %rsp, %rbp`, `subq $0x10, %rsp`, `popq %rbp` | không sinh mã (quản lý khung ngăn xếp) |
| `retq` | `return rax` |

### Cờ điều kiện được biểu diễn dưới dạng biến

Mọi lệnh thay đổi mã điều kiện (condition codes) đều ghi vào một biến duy nhất có tên là `flags`, và mọi lệnh nhảy có điều kiện đều đọc biến này. `cmp a, b` ghi `cmp(a, b)`, `test a, b` ghi `test(a, b)`, và `sub` ghi cùng dạng `cmp` như một tác dụng phụ. Lệnh `jcc` trở thành `Branch(Cond(cc, flags), target, fallthrough)`. Mã điều kiện sẽ quyết định toán tử so sánh khi chúng kết hợp với nhau ở Giai đoạn 6:

| Mã | Sau `cmp(a, b)` | Sau `test(a, a)` |
|---|---|---|
| `je` | `a == b` | `a == 0` |
| `jne` | `a != b` | `a != 0` |
| `jl`, `jge`, `jle`, `jg` | so sánh có dấu `<`, `>=`, `<=`, `>` | so sánh có dấu với 0 |
| `jb`, `jae`, `jbe`, `ja` | so sánh không dấu `<`, `>=`, `<=`, `>` | so sánh không dấu với 0 |

Lệnh `xorl %eax, %eax` được nâng cấp thẳng thành `eax = 0`. Các cờ mà nó thiết lập được ghi dưới dạng cờ kiểu `add` mà không có lệnh nhảy nào trong fixture đọc tới. Mẫu câu lệnh `xor` này cực kỳ quan trọng trong hàm `grade`, nơi Ghidra đã phân tích sai lầm.

### cltd và idiv

Lệnh `cltd` điền bit dấu của `%eax` vào toàn bộ thanh ghi `%edx`. Lệnh `idivl %edi` chia cặp 64-bit `%edx:%eax` cho `%edi`, lưu thương số vào `%eax` và số dư vào `%edx`. Bộ nâng cấp ghi cặp thanh ghi này vào một biến tạm có tên là `dividend` trước. Câu lệnh tính số dư sau đó sẽ đọc giá trị cũ của `%eax` chứ không phải thương số mới vừa tính. Trong SSA, nửa trên xuất hiện dưới dạng `signfill` của nửa dưới. Bộ rút gọn biểu thức sẽ rút gọn cặp này về lại nửa dưới duy nhất, và phép chia trở thành phép toán 32-bit `/` bình thường.

### Các ô nhớ ngăn xếp ở mức -O0

Ở mức `-O0`, clang lưu mọi biến cục bộ trong khung ngăn xếp (stack frame). Hàm `percent_change` lưu `%edi` vào `-0x8(%rbp)` và `%esi` vào `-0xc(%rbp)`, sau đó đọc lại chúng cho mỗi lần sử dụng. Bộ nâng cấp theo dõi chuỗi lệnh mở đầu `pushq %rbp` theo sau bởi `movq %rsp, %rbp`. Sau đoạn mở đầu (prologue) đó, bất kỳ toán hạng bộ nhớ nào có thanh ghi cơ sở là `%rbp`, không có thanh ghi chỉ số, và có độ dời sẽ được biến đổi thành biến có tên `local_<hex>`. Ví dụ: `-0x8(%rbp)` thành `local_8`, `-0xc(%rbp)` thành `local_c`.

Bản thân các lệnh thiết lập khung ngăn xếp không sinh ra câu lệnh IR nào, và các lệnh `subq`, `addq` tác động lên `%rsp` cũng vậy. Mọi toán hạng bộ nhớ khác vẫn được giữ nguyên dưới dạng `Load` hoặc `Store`.

### Tái định vị (Relocations)

Lệnh `leaq (%rip), %rcx` trong hàm `grade` có độ dời bằng 0 trong tệp nhị phân. Bản ghi tái định vị tại độ lệch `0x36` ghi `R_X86_64_PC32 .rodata+0x10`. Trình liên kết sẽ ghi giá trị `S + A - P` vào trường này. Trong đó `S` là địa chỉ của `.rodata`, `A` là số hạng cộng `0x10`, và `P` là địa chỉ của trường đang xét. Lệnh máy sau đó tính toán: `next_ip + disp = (P + 4) + S + A - P = S + A + 4`.

Bộ nâng cấp áp dụng chính phép toán số học đó mà không cần đến trình liên kết. Độ lệch đích là số hạng cộng cộng với khoảng cách từ trường độ dời đến cuối lệnh: `0x10 + 4 = 0x14`. Mã IR ghi nhận `rcx = &.rodata+0x14`, và bảng dữ liệu tại vị trí đó chứa 5 giá trị số nguyên `int`: `1, 2, 3, 4, 4`.

## Giai Đoạn 4: Trích Xuất Tham Số Từ Quy Ước Gọi Hàm

Tài liệu System V AMD64 psABI, Mục 3.2.3, quy định việc truyền các đối số thuộc lớp INTEGER lần lượt qua các thanh ghi: `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9`. Giá trị trả về nằm trong `%rax`, và nếu cần sẽ dùng thêm `%rdx`. Mục 3.2.1 quy định rằng các thanh ghi `%rbp`, `%rbx`, và `%r12` đến `%r15` thuộc về bên gọi (caller-saved) và bắt buộc phải được bảo toàn nguyên vẹn; toàn bộ các thanh ghi còn lại thuộc quyền sở hữu của bên bị gọi.

Hàm `collect_params()` áp dụng một quy tắc duy nhất: một thanh ghi đối số được đọc trước bất kỳ phép ghi nào trong hàm thì đó chính là một tham số. Thanh ghi cao nhất thỏa mãn điều kiện này sẽ chốt số lượng tham số, vì vậy một hàm đọc `%rsi` mà không đọc `%rdi` thì vẫn có 2 tham số. Hàm `percent_change` đọc `%edi` và `%esi`, do đó nó có `param_1` và `param_2`. Hàm `grade` chỉ đọc `%edi`. Hàm `total` đọc `%rdi` dưới dạng con trỏ cơ sở 64-bit và `%esi` dưới dạng biến đếm 32-bit.

Bây giờ, tham số `param_3` bí ẩn của Ghidra đã có lời giải thích thỏa đáng: Lệnh `idivl` ghi vào `%rdx`, và psABI cho phép trả về một giá trị INTEGER rộng 128-bit qua cặp `%rax:%rdx`. Trong một tệp đối tượng riêng lẻ, không có hàm gọi nào để xác nhận độ rộng trả về là bao nhiêu. Ghidra đã võ đoán giữ cả hai thanh ghi làm giá trị trả về.

Tại nhánh rẽ trả về 0, lệnh `xorl %eax, %eax` chỉ ghi vào `%eax`. Ghidra thấy thanh ghi `%rdx` được đọc tại điểm thoát mà không được gán trên nhánh này, nên nó suy luận rằng `%rdx` phải là một đầu vào của hàm, và đầu vào đó biến thành `param_3`. Các lệnh mã máy hoàn toàn đúng, nhưng chữ ký hàm của Ghidra chỉ là một sự suy đoán vội vã đối với những gì mã máy bỏ ngỏ.

## Giai Đoạn 5: Dạng SSA - Một Tên Biến Cho Mỗi Phép Gán

Sau khi nâng cấp, hàm `grade` ghi vào biến `rax` 4 lần và `rcx` 3 lần. Cytron, Ferrante, Rosen, Wegman, và Zadeck (1991) đã định nghĩa dạng gán tĩnh đơn (SSA - Static Single Assignment): Mỗi biến chỉ được gán giá trị đúng một lần duy nhất, và các hàm phi ($\phi$) tại các điểm giao nhau (join points) sẽ lựa chọn giữa các định nghĩa vươn tới được chúng. Thuật toán gồm hai bước: Mục 5.1 đặt các hàm phi dựa trên các ranh giới thống trị (dominance frontiers), và Mục 5.2 đổi tên từng định nghĩa và lần sử dụng thông qua một lượt duyệt cây thống trị. Các hàm `insert_phis()` và `rename()` được xây dựng bám sát hai mục này.

Danh sách SSA của hàm `total` vạch rõ kết quả:

```text
block 0x44 -> 0x50
  ecx_1 = esi_0
  edx_1 = 0
  eax_1 = 0
block 0x50 -> 0x50, 0x5b
  eax_2 = phi(0x44: eax_1, 0x50: eax_3)
  rdx_2 = phi(0x44: edx_1, 0x50: rdx_3)
  flags_2 = phi(0x44: flags_1, 0x50: flags_5)
  flags_3 = add(eax_2, load32(rdi_0 + rdx_2 * 4))
  eax_3 = eax_2 + load32(rdi_0 + rdx_2 * 4)
  flags_4 = inc(rdx_2, 1)
  rdx_3 = rdx_2 + 1
  flags_5 = cmp(zext64(ecx_1), rdx_3)
  if jne(flags_5) goto 0x50 else goto 0x5b
```

Phiên bản 0 của một thanh ghi biểu thị giá trị của nó tại thời điểm nhập hàm: `esi_0` và `rdi_0` chính là các tham số. Đầu vòng lặp `0x50` có hai khối tiền nhiệm (khối khởi tạo và chính nó), do đó mọi biến có định nghĩa trong vòng lặp đều nhận một hàm phi tại đây. Hàm phi `flags_2` thực chất đã "chết ngay khi sinh ra": Thuật toán SSA tối thiểu đặt nó ở đó vì biến `flags` có định nghĩa ở cả hai khối tiền nhiệm, nhưng Giai đoạn 6 sẽ xóa bỏ nó vì không có câu lệnh nào đọc tới nó.

Bước đổi tên cũng giải quyết triệt để độ rộng bit: `edx_1 = 0` là định nghĩa 32-bit nhưng `rdx_3 = rdx_2 + 1` lại là 64-bit, vì vậy hàm phi `rdx_2` sẽ lấy độ rộng lớn hơn (64-bit). Phép đọc `%rcx` trong lệnh `cmpq %rdx, %rcx` là 64-bit trong khi định nghĩa của nó `ecx_1 = esi_0` là 32-bit, do đó điểm sử dụng biến thành `zext64(ecx_1)`. Mọi phép ép kiểu trong mã giả cuối cùng đều bắt nguồn từ quy tắc nhất quán này.

Phần thoát khỏi SSA được thực hiện tinh gọn: Hàm `coalesce_webs()` gộp từng hàm phi cùng các đối số của nó thành một biến duy nhất (gọi là một mạng biến - web) và gán cho nó một tên đại diện. Cách này hoạt động hoàn hảo cho các cấu trúc mã mà trình biên dịch phát ra cho các hàm trong bài học. Nó có thể gặp khó khăn với các phép sao chép song song (parallel copies) khi hai hàm phi tráo đổi giá trị cho nhau, điều mà bài học này chưa cần xử lý. Mục 7 trong bài báo của Cytron mô tả thuật toán tổng quát thông qua việc tô màu đồ thị (graph coloring).

## Giai Đoạn 6: Rút Gọn, Thay Thế Tiến và Loại Bỏ Mã Chết

Hàm `optimize()` lặp lại ba lượt quét cho đến khi không còn thay đổi nào. Bộ rút gọn viết lại các biểu thức từ dưới lên theo các quy tắc:

| Trước rút gọn | Sau rút gọn |
|---|---|
| `Cond(je, cmp(a, b))` | `a == b`, tương tự cho 9 mã điều kiện khác |
| `Cond(jle, test(a, a))` | `a <= 0` |
| `x + (-6)` | `x - 6` |
| `trunc32(zext64(x))` với `x` là 32-bit | `x` |
| `trunc32(a + b)` trên toán hạng 64-bit | `trunc32(a) + trunc32(b)` |
| `(signfill(x): x) / d` | `x / d` ở độ rộng 32-bit |
| `load32(base + idx * 4)` | `base[idx]` |
| `trunc32(p >>s 34) + trunc32(p >>u 63)` với `p = sext64(x) * 0x66666667` | `x / 10` |

Thay thế tiến (Forward substitution) di chuyển một định nghĩa trực tiếp vào điểm sử dụng duy nhất của nó. Quy tắc này đòi hỏi ba điều kiện: Điểm sử dụng nằm trong cùng một khối; điểm sử dụng là một câu lệnh chứ không phải hàm phi; và nếu định nghĩa chứa một lệnh `Load`, thì không có lệnh `Store` hay `Call` nào nằm xen giữa định nghĩa và điểm sử dụng. Hằng số hoặc địa chỉ phân đoạn có thể được thay thế ở bất kỳ đâu trừ các hàm phi, vì đối số của hàm phi phải là biến để phục vụ việc tạo mạng biến (web).

Loại bỏ mã chết (Dead code elimination) xóa bỏ bất kỳ phép gán hoặc hàm phi nào mà biến đích không có nơi sử dụng và biểu thức không chứa lời gọi hàm. Thứ tự các lượt tối ưu hóa là yếu tố then chốt: Trong hàm `percent_change`, phép tính số dư `edx_2 = dividend_1 % edi_0` là mã chết nên bị xóa trước. Điều này khiến `edx_1 = signfill(eax_1)` chỉ còn một nơi dùng duy nhất, nên nó được thay thế tiến vào `dividend_1 = edx_1:eax_1`.

Quy tắc cặp bit dấu sau đó rút gọn `signfill(eax_1):eax_1` thành `eax_1`, và phép chia biến thành `eax_1 / edi_0`. Bây giờ, `eax_1 = esi_1 * 100` chỉ còn một nơi dùng, và `esi_1 = esi_0 - edi_0` cũng chỉ còn một nơi dùng, do đó cả hai được đẩy thẳng vào câu lệnh `return`. Các định nghĩa biến `flags` đã được thay thế vào điều kiện rẽ nhánh và biến mất hoàn toàn. Bốn lệnh máy phức tạp cô đọng lại thành đúng một dòng mã thanh thoát:

```c
return (param_2 - param_1) * 100 / param_1;
```

### Phép nhân với hằng số ma thuật 0x66666667

Hàm `grade` chia điểm số cho 10, nhưng clang hoàn toàn không phát ra lệnh `idiv`. Nó phát ra lệnh `imulq $0x66666667, %rax, %rax`, sau đó thực hiện `sarq $0x22, %rcx` trên một bản sao, tiếp tục `shrq $0x3f, %rax`, rồi cộng hai kết quả lại với nhau. Granlund và Montgomery, trong bài báo "Division by Invariant Integers using Multiplication" (PLDI 1994), đã mô tả kỹ thuật tối ưu hóa này: Trình biên dịch chọn một thừa số nhân `m` và số bit dịch `s`. Với mọi giá trị `x` trong miền xác định, biểu thức `(x * m) >> s` phải tương đương tuyệt đối với `x / d`.

Ở đây, $m = \text{0x66666667} = 1717986919$ và $s = 34$. Phép toán số học hoàn toàn khớp: $2^{34} / 10 = 1717986918.4$, và làm tròn lên (ceiling) là $1717986919$. Sai số là $m \times 10 - 2^{34} = 6$. Để đẳng thức luôn đúng trên mọi số nguyên 32-bit $x$, tích của sai số với $2^{31}$ phải nhỏ hơn $2^{34}$, tức sai số không được vượt quá 8. Sáu nhỏ hơn tám, do đó $(x \times m) \gg 34$ chính xác bằng $\lfloor x / 10 \rfloor$ với mọi số nguyên không âm $x$.

Đối với $x$ âm, phép làm tròn xuống (floor) làm tròn ra xa số 0 trong khi ngôn ngữ C lại làm tròn về phía số 0. Lệnh `shrq $0x3f` trích xuất bit dấu của tích, bit này bằng 1 chính xác khi $x$ là số âm. Việc cộng thêm bit này vào sẽ hiệu chỉnh việc làm tròn số âm về đúng chuẩn của C.

Hàm `magic_divisor(m, s)` kiểm tra cặp tham số ứng viên: Nó tính toán số chia `d` dưới dạng thương nguyên của $2^s$ chia cho $m$ và thử nghiệm với cả `d` lẫn `d + 1`. Nó chỉ chấp nhận `d` khi $m$ đúng bằng trần của $2^s / d$ và biên sai số thỏa mãn. Cùng thừa số này với $s = 33$ sẽ cho $d = 5$, điều này hoàn toàn chính xác vì `0x66666667` cũng là trần của $2^{33} / 5$. Một bài kiểm thử sẽ xác nhận đẳng thức này đối chiếu với phép chia chuẩn C `c_div(x, 10)` trên 2.406 giá trị kiểm thử bao gồm cả hai đầu cực trị của kiểu `int`.

Khi bật tính năng nhận diện quy chuẩn, mã giả của hàm `grade` hiển thị: `ecx_1 = param_1 / 10 - 6;`. Khi tắt tính năng này, bản demo sẽ in ra dạng nguyên bản chân thực:

```c
rax_1 = (long)param_1 * 0x66666667;
ecx_1 = (int)(rax_1 >> 34) + (int)((unsigned long)rax_1 >> 63) - 6;
```

Cả hai dạng đều biên dịch thành công và tính toán ra cùng kết quả. Hãy giữ lại công tắc bật/tắt này: Khi bạn đọc mã máy từ một trình biên dịch lạ mà bạn chưa từng nghiên cứu sâu, dạng thô nguyên bản chính là dạng chân thực và đáng tin cậy nhất.

## Giai Đoạn 7: Suy Luận Kiểu Dữ Liệu Từ Ngữ Cảnh Sử Dụng

Mã máy chỉ có các độ rộng bit chứ không hề có kiểu dữ liệu. Hàm `infer_types()` biến ba nguồn bằng chứng thành các kiểu dữ liệu C: Một mạng biến có các định nghĩa rộng 32-bit sẽ mang kiểu `int`, và rộng 64-bit sẽ là `long`. Một mạng biến đóng vai trò là con trỏ cơ sở của phép truy cập phần tử `base[idx]` với các phần tử rộng 4-byte sẽ được suy luận là `int *`. Kiểu con trỏ này sẽ lan truyền qua các phép sao chép và các hàm phi, vì vậy phép gán `local_8 = param_1` ở mức `-O0` biến cả hai thành `int *`. Kiểu trả về của hàm chính là độ rộng của giá trị xuất hiện trong các câu lệnh `return`.

Kết quả thu được cho hàm `total` là `int total(int *param_1, int param_2)`. Ghidra viết `int total(long param_1,uint param_2)`. Cả hai cách viết đều nhất quán với các lệnh mã máy: Bộ nâng cấp của chúng ta sử dụng phép nạp có nhân hệ số làm bằng chứng con trỏ; còn Ghidra giữ nguyên độ rộng thanh ghi và viết phép truy cập dưới dạng biểu thức con trỏ thô `*(int *)(param_1 + uVar2 * 4)`. Cả hai hệ thống đều không thể phát hiện từ khóa `const`, và cũng không thể phân biệt giữa `int *` và `unsigned *`.

Lee, Avgerinos, và Brumley (TIE, NDSS 2011) mô hình hóa việc khôi phục kiểu dữ liệu dưới dạng bài toán thỏa mãn ràng buộc trên một lưới đại số (lattice), mang lại cận trên và cận dưới thay vì một phỏng đoán duy nhất. Noonan, Loginov, và Cok (Retypd, PLDI 2016) bổ sung thêm tính đa hình và đệ quy, đạt độ bao phủ 98% đối với các chú thích con trỏ `const` dựa trên các mẫu mà trình biên dịch để lại. Bài học này áp dụng hai quy tắc trực quan và hiệu quả nhất từ các nghiên cứu đó, đồng thời gọi tên chính xác những gì mà chúng không thể nhìn thấy.

Tính toán không dấu được giữ lại trong các biểu thức thay vì gán vào kiểu của biến. Lệnh `ja` sau `cmpl $0x4, %ecx` được in ra thành `(unsigned int)ecx_1 > 4`, và `shrq` được in thành `(unsigned long)rax_1 >> 63`. Các biến vẫn được giữ ở dạng có dấu để người đọc vừa thấy được giá trị thanh ghi tổng quát, vừa thấy rõ vị trí cụ thể mà lệnh máy đối xử với nó như một số không dấu.

## Giai Đoạn 8: Tái Cấu Trúc Điều Khiển Từ Đồ Thị (Structuring)

Cifuentes (1994), Chương 6, so khớp các vùng đồ thị với các cấu trúc hình học của ngôn ngữ bậc cao: Vòng lặp được xử lý trước (6.6.1), tiếp theo là rẽ nhánh hai chiều (6.6.2), rồi đến rẽ nhánh đa chiều (6.6.3). Yakdan, Eschweiler, Gerhards-Padilla, và Smith (NDSS 2015) lại chọn một hướng đi khác: Họ suy ra điều kiện logic mà theo đó mỗi nút được thực thi và xây dựng cấu trúc từ các điều kiện đó mà không dựa vào các mẫu hình cố định. Động lực của họ đến từ thực tế: Trình dịch ngược Hex-Rays trên một mẫu thử nghiệm tạo ra tới 1.571 lệnh `goto` trong 49.514 dòng mã (trung bình 1 lệnh goto trên 32 dòng), trong khi thuật toán của họ không tạo ra bất kỳ lệnh `goto` nào trên 1.530 hàm của bộ công cụ coreutils.

Hàm `structure()` trong bài học là một biến thể tinh gọn thuộc trường phái thứ nhất. Nó cần hai dữ kiện từ Giai đoạn 2: Nút hậu thống trị trực tiếp (immediate post-dominator) của một khối kết thúc bằng `Branch` chính là điểm hội tụ (join) của cấu trúc điều kiện; và một cạnh ngược là một cạnh dẫn tới một khối thống trị điểm xuất phát của nó. Vòng lặp tự nhiên (natural loop) là tập hợp các khối có thể đi tới điểm xuất phát của cạnh ngược mà không cần đi qua đầu vòng lặp.

Đối với một lệnh rẽ nhánh `Branch` có điểm hội tụ là `j`:

- Nếu `j` chính là đích nhảy, phát ra: `if (!cond) { khối đi thẳng }` và tiếp tục tại `j`.
- Nếu `j` chính là khối đi thẳng, phát ra: `if (cond) { khối đích nhảy }` và tiếp tục tại `j`.
- Nếu `j` là điểm thoát của hàm, cả hai nhánh đều trả về. Phát ra nhánh ngắn hơn dưới dạng `if (cond) {... }` và đặt nhánh còn lại ngay sau đó mà không cần từ khóa `else`. Đây chính là cấu trúc trả về sớm (early return) trong `percent_change` và `total`.
- Trong các trường hợp còn lại, phát ra: `if (cond) { đích } else { đi thẳng }` và tiếp tục tại `j`. Đây là cấu trúc của `percent_change` ở mức `-O0`, nơi cả hai nhánh đều gán giá trị cho `local_4` rồi cùng đi tới đoạn kết hàm (epilogue).

Đối với đầu vòng lặp (loop header):

- Nếu đầu vòng lặp kết thúc bằng một `Branch` với một khối kế vị nằm ngoài vòng lặp, phát ra: `while (cond) { thân vòng lặp }`. Hàm `total` ở mức `-O0` có cấu trúc này: đầu vòng lặp so sánh `local_14` với `local_c` và thoát vòng lặp bằng lệnh `jge`.
- Nếu đầu vòng lặp cũng chính là khối chứa cạnh ngược, hoặc chốt thoát nằm ở cuối, phát ra: `do { thân vòng lặp } while (cond);`. Hàm `total` ở mức `-O1` có cấu trúc này: khối `0x50` cộng giá trị, tăng biến đếm, so sánh và tự nhảy về chính nó.

Lệnh `IndirectJump` kết thúc một khối mà không có khối kế vị xác định nào. Hàm `grade` ở mức `-O0` sử dụng một bảng nhảy thực sự (`jmpq *%rax` qua 5 độ lệch tương đối tại `.rodata+0x0`), và bài học 33 đã giải quyết các đích đến này. Ở bài học này, bộ nâng cấp mã sẽ in ra một ghi chú cảnh báo: Bốn khối tại `0x76`, `0x7f`, `0x88`, và `0x91` chỉ có thể tiếp cận được qua bước nhảy gián tiếp và chưa được nâng cấp. Bài tập thực hành sẽ bổ sung cấu trúc `switch` hoàn chỉnh.

### Trình biên dịch đã biến đổi câu lệnh switch thế nào ở mức -O1

Ở mức `-O1`, clang không hề duy trì một bảng nhảy cho hàm `grade`. Nó tính toán `score / 10 - 6` và so sánh kết quả với số 4 dưới dạng số không dấu. Sau đó, nó nạp giá trị trả về trực tiếp từ một bảng gồm 5 phần tử kiểu `int` tại `.rodata+0x14`. Phép so sánh không dấu là một lệnh duy nhất nhưng bao trùm cả hai trường hợp: `score / 10 < 6` và `score / 10 > 10` (vì một hiệu số âm khi coi là số không dấu sẽ trở thành một số dương khổng lồ, do đó cũng trượt phép kiểm tra này).

Bảng chứa dãy số `1, 2, 3, 4, 4` tương ứng với các thương số từ 6 đến 10. Cấu trúc `switch` với 5 câu lệnh `return` đã được thu gọn hoàn toàn thành một câu lệnh `if` và một phép đọc mảng tĩnh. Bộ nâng cấp phát ra chính xác cấu trúc đó, và đặt tên cho bảng là `lookup_14` theo độ lệch phân đoạn của nó.

Độ dài của bảng được xác định theo phương pháp phỏng đoán thực nghiệm (heuristic): Hàm `table_values()` đọc từ điểm bắt đầu của bảng tới độ lệch `.rodata` tiếp theo mà bất kỳ bản ghi tái định vị nào trong `.text` tham chiếu tới, hoặc tới cuối phân đoạn. Trong tệp đối tượng này, độ lệch tham chiếu tiếp theo là `0x28` (bảng của `grade_label`), do đó `lookup_14` có kích thước là `(0x28 - 0x14) / 4 = 5` phần tử. Phép so sánh không dấu với 4 hoàn toàn khớp với kích thước này. Bản thân Ghidra cũng không biết độ dài thực tế của bảng: Nó viết `*(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4)`, tức một địa chỉ byte cộng với độ dời, và để mặc cho người đọc tự tra cứu 5 giá trị đó.

## Đặt Cạnh Nhau Đối Chiếu Với Ghidra

Bản demo in ra từng cặp hàm đối chiếu. Văn bản của Ghidra được trích xuất từ `code/fixtures/ghidra/*.json`, là các bản ghi Evidence mà rea trả về ngày 2026-10-14 bằng câu lệnh `npx -y rea-agents@6.5.0 decompile ledger-x86_64-O1.o <function> --provider ghidra --format json`. Mỗi bản ghi đều mang thuộc tính `confidence: observed` và `authority: shipped-artifact`. Bản thân văn bản là thứ quan sát được; nhưng các khẳng định nằm bên trong văn bản lại là kết quả suy diễn, và các bản ghi không hề chỉ rõ khẳng định nào là suy diễn.

### percent_change

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int percent_change(int param_1, int param_2)   | undefined1  [16] percent_change(int param_1,int param_2,ulong param_3)
{                                              | {
  if (param_1 == 0) {                          |   long lVar1;
    return 0;                                  |   undefined1 auVar2 [16];
  }                                            |   undefined1 auVar3 [16];
  return (param_2 - param_1) * 100 / param_1;  |   if (param_1 != 0) {
}                                              |     lVar1 = (long)((param_2 - param_1) * 100);
                                               |     auVar3._0_8_ = lVar1 / (long)param_1 & 0xffffffff;
                                               |     auVar3._8_8_ = lVar1 % (long)param_1 & 0xffffffff;
                                               |     return auVar3;
                                               |   }
                                               |   auVar2._8_8_ = 0;
                                               |   auVar2._0_8_ = param_3;
                                               |   return auVar2 << 0x40;
                                               | }
```

| Điểm sai khác | Lệnh máy thực tế hay suy đoán |
|---|---|
| Kiểu trả về `undefined1 [16]` so với `int` | **Suy đoán**. `idivl` ghi vào `%rax` và `%rdx`, và không có hàm gọi nào ấn định độ rộng trả về, nên Ghidra giữ cả hai thanh ghi. |
| Tham số `param_3` | **Suy đoán** kéo theo từ phỏng đoán trên: Tại nhánh trả về 0 chỉ có `%eax` được ghi, nên `%rdx` bị hiểu nhầm là một đầu vào. |
| `lVar1 / (long)param_1 & 0xffffffff` so với `/ param_1` | **Lệnh máy**. `cltd` kết hợp `idivl` là phép chia 64-bit của một số 32-bit mở rộng dấu, cắt cụt về 32-bit. Cả hai văn bản đều phản ánh điều này; bộ nâng cấp của chúng ta đã rút gọn cặp này bằng quy tắc `signfill`. |
| Giữ lại phép chia lấy dư `lVar1 % ...` | Hệ quả từ kiểu trả về 16-byte: Số dư trong `%rdx` là mã chết trong phân tích của chúng ta nhưng lại còn sống trong phân tích của Ghidra. |
| `if (param_1 != 0)` với nhánh 0 nằm sau | Thứ tự lệnh máy: `je 0xd` nhảy tới khối trả về 0. Ghidra đặt nhánh tính toán lên trước, chúng ta đặt nhánh ngắn hơn lên trước; cả hai đều đúng ngữ nghĩa. |
| `return auVar2 << 0x40` | **Suy đoán**. Một giá trị 16-byte với `param_3` ở nửa dưới, dịch trái sang nửa trên, là cách trình dịch ngược biểu diễn `%rax = 0` với `%rdx` chưa xác định. |

Phiên bản sửa lỗi trong `code/fixtures/ghidra/ledger-x86_64-O1.ghidra-repaired.c` chỉ chỉnh sửa duy nhất hàm này, và bài kiểm thử sẽ so sánh nó với mã nguồn gốc trên 81 cặp dữ liệu đầu vào.

### grade

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
static const int lookup_14[5] = {1, 2, 3, 4, 4}; | uint grade(int param_1)
int grade(int param_1)                         | {
{                                              |   uint uVar1;
  int eax_1;                                   |   uint uVar2;
  int ecx_1;                                   |   
                                               |   uVar2 = param_1 / 10 - 6;
  ecx_1 = param_1 / 10 - 6;                    |   uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  eax_1 = 0;                                   |   if (uVar2 < 5) {
  if ((unsigned int)ecx_1 <= 4) {              |     uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
  }                                            |   return uVar1;
  return eax_1;                                | }
}                                              |
```

| Điểm sai khác | Lệnh máy thực tế hay suy đoán |
|---|---|
| Cả hai đều có `param_1 / 10 - 6` | **Lệnh máy**. Cả hai đều nhận diện thành công mẫu hằng số ma thuật `0x66666667`. |
| `uVar1 = -(param_1 >> 0x1f) ^ ...` so với `eax_1 = 0` | Lệnh `xorl %eax, %eax` tại `grade+0x1a`. Thanh ghi này thực chất bằng 0! Ghidra biểu diễn phép `xor` này thành bit dấu của `param_1` XOR với bit dấu của tích (hai cách viết khác nhau của cùng một bit) nhưng lại không thể triệt tiêu chúng. Giá trị luôn luôn bằng 0 cho mọi số `int`, và bài kiểm thử chứng minh điều đó trên 141 điểm số. Một người đọc tin vào văn bản Ghidra sẽ lầm tưởng rằng giá trị mặc định phụ thuộc vào điểm số đầu vào! |
| `uVar2 < 5` so với `(unsigned int)ecx_1 <= 4` | Lệnh `cmpl $0x4, %ecx` rồi `ja`. Cùng một điều kiện logic: Ghidra định kiểu `uVar2` là `uint`, chúng ta ép kiểu ngay tại điểm sử dụng. |
| `*(uint *)(&DAT_001000b4 + ...)` so với `lookup_14[...]` | Lệnh máy cộng tái định vị: Ghidra nạp tệp đối tượng với `.rodata` tại `0x001000a0`, do đó `0x001000b4` là `.rodata + 0x14`. Nó áp dụng tái định vị và giữ lại địa chỉ byte thô; chúng ta đặt tên cho bảng và đọc trực tiếp 5 giá trị của nó. |
| Kiểu trả về `uint` so với `int` | **Suy đoán** ở cả hai phía: Lệnh máy chỉ cho biết độ rộng 32-bit chứ không phân biệt có dấu hay không dấu. |

### total

```text
lifter                                         | ghidra
-----------------------------------------------+-----------------------------------------------
int total(int *param_1, int param_2)           | int total(long param_1,uint param_2)
{                                              | {
  int eax_1;                                   |   int iVar1;
  int ecx_1;                                   |   ulong uVar2;
  long rdx_1;                                  |   
                                               |   if (0 < (int)param_2) {
  if (param_2 <= 0) {                          |     uVar2 = 0;
    return 0;                                  |     iVar1 = 0;
  }                                            |     do {
  ecx_1 = param_2;                             |       iVar1 = iVar1 + *(int *)(param_1 + uVar2 * 4);
  rdx_1 = 0;                                   |       uVar2 = uVar2 + 1;
  eax_1 = 0;                                   |     } while (param_2 != uVar2);
  do {                                         |     return iVar1;
    eax_1 = eax_1 + param_1[rdx_1];            |   }
    rdx_1 = rdx_1 + 1;                         |   return 0;
  } while ((unsigned long)(unsigned int)ecx_1 != rdx_1); | }
  return eax_1;                                |
}                                              |
```

| Điểm sai khác | Lệnh máy thực tế hay suy đoán |
|---|---|
| `int *param_1` so với `long param_1` | Suy luận từ phía chúng ta dựa trên `addl (%rdi,%rdx,4), %eax`. Ghidra giữ nguyên độ rộng thanh ghi. |
| `uint param_2` và `(int)param_2` so với `int param_2` | Ghidra chọn kiểu không dấu vì `cmpq %rdx, %rcx` so sánh biến đếm đã mở rộng zero. Chúng ta giữ kiểu `int` và ép kiểu tại phép so sánh. |
| Cả hai đều dùng `do { } while` | Lệnh `jne 0x50` từ khối nhảy về chính nó: Trình biên dịch đã xoay vòng lặp (loop rotation). Tệp đối tượng `-O0` giữ nguyên cấu trúc `while`. |
| Giữ lại `ecx_1 = param_2` so với `param_2 != uVar2` | Quy tắc thay thế tiến của chúng ta chỉ áp dụng trong nội bộ một khối: Phép gán `movl %esi, %ecx` nằm ở khối khởi tạo còn phép so sánh nằm trong vòng lặp, nên biến được giữ lại. Ghidra đã lan truyền biến này qua ranh giới khối. |
| Biểu thức `(unsigned long)(unsigned int)ecx_1` | **Lệnh máy**. Lệnh `movl %esi, %ecx` mở rộng zero, và phép so sánh 64-bit đọc toàn bộ thanh ghi. |

Bài kiểm tra hành vi biên dịch văn bản Ghidra cho `grade` và `total` mà không cần chỉnh sửa gì. Nó bổ sung hai dòng `typedef` và định nghĩa mảng `DAT_001000b4` từ 20 byte tại `.rodata+0x14`. Cả hai hàm đều chạy khớp hoàn toàn với mã nguồn gốc trên lưới dữ liệu thử nghiệm.

## Giới Nghiên Cứu Nói Gì Về Việc Đọc Mã Giả

Công trình LLM4Decompile (EMNLP 2024) đo lường Ghidra đơn lẻ chỉ đạt tỷ lệ 20,12% khả năng tái thực thi thành công (re-executability) trên tập dữ liệu HumanEval-Decompile (tức chỉ khoảng 1 trong 5 hàm biên dịch lại được và vượt qua các phép kiểm tra của benchmark). Việc tinh chỉnh đầu ra của Ghidra bằng mô hình 6.7B của họ đã nâng tỷ lệ này lên 52,74%. Tuy nhiên, DecompileBench (Findings of ACL 2025) khi đánh giá 6 trình dịch ngược công nghiệp và 6 phương pháp tiếp cận bằng LLM trên 23.400 hàm từ 130 chương trình đã chỉ ra một sự thật đáng kinh ngạc: Các đầu ra của LLM trông "dễ đọc" hơn nhiều nhưng lại có độ chính xác về chức năng thấp hơn tới 52,2%!

Liu, Raff, và Micinski (arXiv 2609.05370, tháng 9 năm 2026) đã giải thích cội rễ của hố sâu ngăn cách này. Họ thử nghiệm trên 300 hàm thư viện GitHub và 287 hàm liên quan đến các lỗ hổng CVE thực tế: Mô hình tinh chỉnh mạnh nhất đã nâng tỷ lệ biên dịch thành công của Ghidra từ 75% lên 90%, nhưng tỷ lệ khớp đúng hành vi chức năng lại sụt giảm nghiêm trọng từ 74% xuống còn 62%! Các ứng viên vượt qua toàn bộ các bài kiểm thử tĩnh trong benchmark vẫn phân kỳ hành vi khi bị kiểm thử fuzzing (điều này xảy ra ở 4,9% trường hợp trên 8 hệ thống, và lên tới 13% ở một hệ thống cụ thể). Đáng báo động nhất, có tới 10% các lỗ hổng bảo mật đã được công bố bị biến mất hoàn toàn khỏi đoạn mã đã qua tinh chỉnh của LLM!

Nguyên nhân mà họ vạch ra vô cùng đơn giản: Các trường dữ liệu, kiểu dữ liệu, hàm bị gọi và các khối bảo vệ do LLM "tự sáng tác" đã âm thầm thay thế các yếu tố chưa biết rõ ràng mà trình dịch ngược ban đầu để lại trong văn bản.

Đó chính là lý do tại sao bộ nâng cấp mã của chúng ta chọn in ra `(unsigned long)(unsigned int)ecx_1` thay vì một cái tên trau chuốt giả tạo. Nó in ra một ghi chú cảnh báo thay vì tự vẽ ra một câu lệnh `switch` cho một bước nhảy chưa giải quyết, và dùng tên `lookup_14` thay vì một nhãn mỹ miều do mô hình tự bịa ra. Mọi dòng mã đều có một lộ trình suy dẫn minh bạch mà bạn có thể trực tiếp kiểm tra trong `code/main.py`. Kiểu `undefined1 [16]` của Ghidra tuy xấu xí nhưng trung thực. Một chữ ký bóng bẩy `int percent_change(int, int)` từ một mô hình tinh chỉnh có thể đúng một cách tình cờ trong trường hợp này, nhưng cùng mô hình đó sẽ tạo ra một chữ ký hoàn toàn sai lệch và tai hại ở hàm tiếp theo.

## Xây Dựng Mã Nguồn (Build It)

Chạy chương trình demo:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
```

Bản demo giải mã toàn bộ 4 hàm của tệp đối tượng `-O1` và đối chiếu từng lệnh với văn bản `llvm-objdump` đã cam kết. Đối với `percent_change`, `grade`, và `total`, nó in ra mã IR trước SSA, danh sách SSA với các hàm phi, và mã giả đặt cạnh văn bản Ghidra. Sau đó, nó in hàm `grade` khi tắt nhận diện phép chia, các phiên bản `-O0` của `percent_change`, `total`, và `grade_label`, cùng một bản tóm tắt:

```json
{
  "functions": ["grade", "grade_label", "percent_change", "total"],
  "instructions": 56,
  "blocks": {"percent_change": 3, "grade": 3, "total": 5, "grade_label": 4},
  "phis": {"percent_change": 0, "grade": 1, "total": 2, "grade_label": 1},
  "objdumpDifferences": 0
}
```

Hãy đọc `code/main.py` theo trình tự sau:

1. `read_object()` đọc các tiêu đề phân đoạn, bảng ký hiệu, và các phân đoạn `.rela` của một tệp đối tượng có thể tái định vị ELF64.
2. `decode_one()` và `render_att()` giải mã một lệnh máy và in ra theo đúng chuẩn `llvm-objdump`.
3. `build_blocks()`, `immediate_dominators()`, và `dominance_frontiers()` xây dựng các cấu trúc đồ thị luồng điều khiển.
4. `lift_insn()` biến đổi một lệnh máy thành các câu lệnh IR và giải quyết các tham chiếu tái định vị.
5. `insert_phis()` và `rename()` xây dựng dạng SSA và chèn các phép ép kiểu độ rộng.
6. `make_simplifier()`, `substitute_once()`, và `remove_dead_code()` thực thi các lượt tối ưu hóa bên trong `optimize()`.
7. `collect_params()`, `coalesce_webs()`, và `infer_types()` định danh và suy luận kiểu cho các mạng biến.
8. `structure()` kết hợp với `post_dominators()` và `natural_loops()` phát ra thân hàm C hoàn chỉnh.
9. `lift_function()` liên kết toàn bộ các giai đoạn lại với nhau và `lift_all()` nâng cấp toàn bộ tệp đối tượng kèm chữ ký hàm.

## Sử Dụng Mã Nguồn (Use It)

Để nâng cấp một hàm từ tệp đối tượng của riêng bạn, hãy biên dịch nó bằng `clang -target x86_64-unknown-linux-gnu -c` và gọi `lift_all(read_object(path))`. Khi bộ giải mã ném ra ngoại lệ `DecodeError`, hãy đọc byte opcode trong thông báo lỗi và tra cứu nó trong bản đồ opcode của tài liệu SDM, sau đó bổ sung một mục vào bảng ở đầu tệp. Khi bộ nâng cấp gặp một tên lệnh chưa hỗ trợ, hãy thêm một quy tắc vào `lift_insn()`. Khi bộ tái cấu trúc phát ra lệnh `goto`, hãy in đồ thị bằng `IrPrinter` và xác định quy tắc cấu trúc nào còn thiếu. Mỗi bổ sung đều rất nhỏ và đều có thể được kiểm chứng đối chiếu với `llvm-objdump` hoặc clang thông qua các bài kiểm thử sẵn có.

Khi đã cài đặt Ghidra, hãy chạy rea trên cùng hàm đó và đặt hai đoạn mã giả nằm cạnh nhau. Hãy điền đầy đủ bảng sai khác trước khi bạn vội tin tưởng vào bất kỳ đoạn mã nào. Một điểm sai khác có một lệnh máy bảo chứng phía sau là một sự lựa chọn về cách trình bày. Một điểm sai khác không có lệnh máy nào bảo chứng chính là một sự phỏng đoán, và sự phỏng đoán đó có thể nằm ở bất kỳ phía nào.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: Một hàm xuyên suốt 5 giai đoạn

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode/code
python3 -q
```

Chạy:

```python
from main import *
obj = read_object(FIXTURES / "ledger-x86_64-O1.o")
lifted = lift_function(obj, "percent_change")
for insn in lifted.decoded: print(f"0x{insn.address:x}: {insn.raw.hex(' ')}  {insn.att()}")
print(lifted.ir_text)
print(lifted.ssa_text)
print(lifted.pseudocode)
```

Hãy lần theo thanh ghi `%esi` bắt đầu từ lệnh `subl %edi, %esi` đến câu lệnh IR `esi_1 = esi_0 - edi_0` rồi đến biểu thức `(param_2 - param_1)`. Byte `fe` chính là ModR/M của lệnh `subl`, với `mod = 11`, `reg = edi`, và `r/m = esi`.

### Lab B: Phép chia bằng hằng số ma thuật

```python
print(lift_function(obj, "grade", recognize_division=False).pseudocode)
print(magic_divisor(0x66666667, 34), magic_divisor(0x66666667, 33), magic_divisor(0x66666667, 31))
```

Dòng đầu tiên in ra phép nhân và dịch bit nguyên bản. Dòng thứ hai in ra `10 5 None`. Hãy tự tính toán xem tại sao cùng thừa số đó lại chia cho 5 khi giảm 1 bit dịch, và tại sao số bit dịch 31 lại trượt điều kiện biên sai số.

### Lab C: Một tệp đối tượng mới làm dừng bộ giải mã

Biên dịch fixture ở mức `-O2` và nâng cấp nó:

```bash
clang -target x86_64-unknown-linux-gnu -O2 -fno-asynchronous-unwind-tables -c fixtures/ledger.c -o "$TMPDIR/ledger-O2.o"
python3 -c 'from main import *; from pathlib import Path; import os; lift_all(read_object(Path(os.environ["TMPDIR"]) / "ledger-O2.o"))'
```

Bộ giải mã dừng lại tại `0x56` với thông báo `opcode 81 is outside the fixture subset`. Opcode `81` là dạng nhóm 1 với giá trị tức thời 32-bit. Lệnh máy là `andl $0x7ffffff8, %edx`, được clang sử dụng trong vòng lặp vector hóa của hàm `total`. Bổ sung nó bên cạnh `0x83` trong `decode_one()`, chạy lại, và quan sát xem opcode nào sẽ dừng bạn tiếp theo. Tập con lệnh được mở rộng từng bước một cách có kiểm soát và kiểm chứng đối chiếu với `llvm-objdump`.

### Lab D: Giá trị mặc định của Ghidra trong grade

```python
print(ghidra_pseudocode("grade"))
for score in (-25, 0, 59, 60, 100, 115):
    sign = -(score >> 31)
    product_sign = ((score * 0x66666667) & ((1 << 64) - 1)) >> 63
    print(score, sign ^ product_sign)
```

Mọi hàng đều in ra số 0. Biểu thức `uVar1` của Ghidra tuy đúng về mặt toán học nhưng hoàn toàn không thể đọc hiểu, trong khi lệnh máy `xorl %eax, %eax` diễn đạt điều đó chỉ vỏn vẹn trong 2 byte!

## Bài Tập Thực Hành (Practice Lab)

Bổ sung khả năng khôi phục câu lệnh `switch` cho hàm `grade` ở mức `-O0`.

Các yêu cầu:

- Đọc 5 bản ghi tái định vị trong phân đoạn `.rela.rodata` của `ledger-x86_64-O0.o` (mỗi bản ghi là `R_X86_64_PC32 .text+offset`), và tính toán 5 đích nhảy tương ứng từ bảng tại `.rodata+0x0`.
- Thêm một điểm kết thúc khối lệnh `Switch` vào IR với biểu thức chỉ số, danh sách các đích đến, và nhánh mặc định lấy từ lệnh `ja` đứng trước lệnh nạp bảng.
- Gán cho khối switch một khối kế vị cho mỗi đích nhảy để các phép tính nút thống trị, nút hậu thống trị và SSA tiếp tục hoạt động chính xác.
- Phát ra cấu trúc: `switch (local_10) { case 0:... }` với một nhãn `case` cho từng đích và phần kết hàm chung sau điểm hội tụ.
- Giữ lại ghi chú cảnh báo cho bất kỳ bước nhảy gián tiếp nào mà bạn không thể xác định được biên giới hạn của bảng.
- Biên dịch kết quả bằng clang và so sánh đối chiếu với mã nguồn gốc trên lưới điểm số trong các bài kiểm thử.

Thành công đồng nghĩa với việc hàm `grade` ở mức `-O0` và `-O1` sinh ra cùng kết quả thực thi trên lưới thử nghiệm, với một cấu trúc `switch` hoàn chỉnh và không có bất kỳ lệnh `goto` nào.

## Bài Tập Đánh Giá (Graded Lab)

Năm câu hỏi về các fixture của bài học này. Hãy tìm câu trả lời bằng chính mã nguồn bạn đã xây dựng, sau đó ghi kết quả vào `code/lab_answers.json`. Ghi số dưới dạng số nguyên hoặc chuỗi thập lục phân như `"0x40"`, và danh sách dưới dạng mảng JSON.

1. `t1`: Danh sách các tham số mà quy ước gọi hàm gán cho hàm `percent_change` ở mức -O1.
2. `t2`: Số lượng hàm phi trong dạng SSA của hàm `total` ở mức -O1.
3. `t3`: Số chia mà hàm `magic_divisor` tìm thấy cho thừa số `0x66666667` với số bit dịch 34.
4. `t4`: Câu lệnh cuối cùng trong mã giả của hàm `percent_change` ở mức -O1, dưới dạng một dòng văn bản.
5. `t5`: Kiểu dữ liệu C khôi phục được của tham số `param_1` trong hàm `total` ở mức -O1.

Kiểm tra câu trả lời của bạn:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/lab.py
```

Trình kiểm tra sẽ tính toán các giá trị kỳ vọng từ `code/main.py` trên các fixture đã cam kết. Nó in ra `PASS`, `FAIL` hoặc `TODO` cho từng nhiệm vụ cùng điểm số, và chỉ thoát với mã 0 khi cả 5 bài đều đạt. Lệnh `python3 code/lab.py my-answers.json` cho phép kiểm tra một tệp câu trả lời khác. Bài kiểm thử `code/tests/test_lab.py` chứng minh trình chấm chấp nhận câu trả lời đúng và từ chối câu trả lời sai.

### Trực Quan Hóa Hoạt Động

Cỗ máy tương tác [Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?opt=O0) chạy bài học này dưới dạng máy ép nâng cấp (lift press), các ghim định kiểu (type pins), dưỡng gá cấu trúc (structuring jig) và máy in mã giả, với đầy đủ các dạng IR, SSA và mã giả mà `code/main.py` in ra. Đường dẫn này mở tệp đối tượng `-O0`: ba ô nhớ ngăn xếp biến thành các biến cục bộ và dưỡng gá kẹp chặt cả hai hàm cho cấu trúc if/else.

## Sản Phẩm Đóng Gói (Lesson Artifact)

Bài học này tạo ra sản phẩm `outputs/skill-lifting-to-pseudocode.md`. Hãy sử dụng nó để nâng cấp một hàm với lộ trình suy dẫn minh bạch cho từng dòng mã và điền đầy đủ bảng sai khác đối chiếu với văn bản của trình dịch ngược. Sản phẩm yêu cầu danh sách mã máy tham chiếu, các bản ghi tái định vị, quy ước gọi hàm, kiểm tra biên dịch, so sánh hành vi thực thi, và một phán quyết rõ ràng cho từng dòng mã có sự khác biệt.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/34-lifting-to-pseudocode
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Bộ kiểm thử gồm 25 bài kiểm tra, chứng minh:

- Bộ giải mã đồng thuận với văn bản `llvm-objdump` đã cam kết trên mọi lệnh của cả hai tệp đối tượng, và với cả `llvm-objdump` thực tế khi được cài đặt.
- REX, ModR/M, SIB, và độ lệch trường của lệnh `leaq (%rip), %rcx` được giải mã chính xác như tài liệu SDM mô tả.
- Một opcode nằm ngoài tập con sẽ kích hoạt `DecodeError` thay vì suy đoán bừa bãi.
- Bộ đọc ELF liệt kê chuẩn xác 4 hàm và 3 bản ghi tái định vị trong phân đoạn `.text`.
- Hàm `total` có 5 khối, `0x44` thống trị `0x50`, và `0x50` là đầu vòng lặp với chính nó làm thân vòng lặp.
- Các thanh ghi `%rdi` và `%rsi` trở thành `param_1` và `param_2`, và `param_1` trong `total` mang kiểu `int *`.
- Lệnh `test` và `cmp` đi cùng `je`, `jle`, `ja`, `jge` trở thành `==`, `<=`, so sánh không dấu `>`, và `<`.
- Dạng SSA đặt 2 hàm phi còn sống tại đầu vòng lặp của `total` và không có hàm phi nào trong `percent_change`.
- Mã giả của `percent_change`, `grade`, và `total` khớp hoàn toàn với văn bản kỳ vọng.
- Hàm `grade` đọc `param_1 / 10 - 6` khi bật nhận diện và đọc phép nhân thô khi tắt nhận diện.
- `0x66666667` với dịch 34 thực hiện chia cho 10, với dịch 33 thực hiện chia cho 5, và đẳng thức luôn đúng trên lưới thử nghiệm.
- Các ô nhớ ngăn xếp ở mức `-O0` trở thành `local_4`, `local_8`, `local_c`, và con trỏ khung ngăn xếp không bao giờ xuất hiện trong mã giả.
- Bảng dữ liệu tại `.rodata+0x14` đọc ra dãy số `1, 2, 3, 4, 4` thông qua bản ghi tái định vị.
- Bước nhảy gián tiếp trong `grade` ở mức `-O0` được báo cáo rõ ràng kèm 4 khối chưa được nâng cấp.
- Các fixture Ghidra chứa đầy đủ kiểu trả về 16-byte, tham số `param_3`, phép `xor` chưa rút gọn, và `long param_1`.
- Sáu đơn vị mã phát ra biên dịch thành công với lệnh `clang -std=c11 -Wall -Werror` cho mục tiêu `x86_64-unknown-linux-gnu`.
- Các hàm được nâng cấp, văn bản Ghidra đã sửa lỗi, và mã nguồn gốc in ra cùng 232 dòng kết quả trên lưới dữ liệu thử nghiệm.

## Các Chế Độ Thất Bại Trong Môi Trường Thực Tế

| Thất bại | Công cụ yếu kém báo cáo | Điều quy trình suy dẫn phải chỉ rõ |
|---|---|---|
| Trình dịch ngược nới rộng kiểu trả về | "trả về một struct 16-byte" | Lệnh `idivl` ghi vào `%rdx`, và không có bên gọi nào ấn định độ rộng trả về |
| Trình dịch ngược tự thêm tham số | "nhận ba đối số đầu vào" | Thanh ghi chỉ được đọc trên một nhánh duy nhất vì độ rộng trả về bị đoán mò |
| Mẫu tối ưu hóa không được nhận diện | "nhân với một hằng số ma thuật lạ" | Thừa số nhân là trần của $2^s / d$ và biên sai số hoàn toàn thỏa mãn |
| Mẫu được nhận diện nhưng mất phép `xor` | "giá trị mặc định phụ thuộc đầu vào" | `xorl %eax, %eax` ghi số 0, hai cách biểu diễn của cùng một bit triệt tiêu nhau |
| Con trỏ bị nhận diện nhầm thành số nguyên | "nhận một số kiểu long" | Phép nạp 4-byte có tỷ lệ từ thanh ghi là bằng chứng rõ ràng của con trỏ |
| Không xác định được độ dài bảng | "đọc tràn vượt qua cuối bảng" | Độ lệch tham chiếu tiếp theo hoặc điểm cuối phân đoạn giới hạn nó, và phép kiểm tra biên đồng thuận |
| Đọc sai dạng thức vòng lặp | "vòng lặp while" | Đầu vòng lặp không có lối thoát, chốt nhảy ngược về đầu, do đó nó là `do while` |
| Bước nhảy gián tiếp bị phỏng đoán | "câu lệnh switch với 5 nhánh" | Các đích đến xuất phát từ `.rela.rodata`, hoặc bước nhảy phải được báo cáo là chưa giải quyết |
| Thay thế tiến vượt qua lệnh ghi bộ nhớ | "gộp thành một biểu thức duy nhất" | Một giá trị nạp bị dời qua một lệnh lưu bộ nhớ hoặc lời gọi hàm sẽ làm sai lệch ngữ nghĩa |
| Mô hình LLM tự tiện chỉnh sửa văn bản | "biên dịch được và nhìn rất đẹp" | Tỷ lệ build tăng nhưng tỷ lệ đúng hành vi giảm, các khối bảo vệ bịa đặt thay thế các yếu tố chưa biết |
| Bộ giải mã phỏng đoán một opcode lạ | "danh sách dịch ngược trông ổn" | Ném ngoại lệ `DecodeError` kèm địa chỉ và byte, sau đó bổ sung mục bảng và kiểm tra với `llvm-objdump` |

## Quy Tắc Vận Hành Cốt Lõi

Hãy đối xử với mã giả như một quy trình suy dẫn mà bạn chưa từng được tận mắt chứng kiến. Hãy tái tạo lại nó từ các lệnh mã máy bằng một bộ nâng cấp mà bạn có thể đọc hiểu từng dòng. Đặt hai đoạn văn bản nằm cạnh nhau, và chỉ rõ lệnh máy thực tế hoặc sự suy đoán đằng sau mỗi dòng có sự khác biệt. Hãy biên dịch cả hai và chạy đối chiếu cả hai với mã nguồn gốc khi bạn có nó trong tay. Một dòng mã giả không có lệnh máy bảo chứng phía sau thì mãi mãi chỉ là một sự suy đoán, bất kể nó trông có vẻ sạch sẽ và thuận mắt đến đâu.

## Đọc Thêm

- [Intel SDM Tập 2, Chương 2, Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [System V AMD64 psABI, Mục 3.2, Function Calling Sequence](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Cytron et al., Efficiently Computing Static Single Assignment Form and the Control Dependence Graph, TOPLAS 1991](https://doi.org/10.1145/115372.115320)
- [Cooper, Harvey, Kennedy, A Simple, Fast Dominance Algorithm, 2001](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Cifuentes, Reverse Compilation Techniques, PhD thesis, QUT 1994](https://eprints.qut.edu.au/36820/)
- [Yakdan et al., No More Gotos, NDSS 2015](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Granlund, Montgomery, Division by Invariant Integers using Multiplication, PLDI 1994](https://doi.org/10.1145/178243.178249)
- [Lee, Avgerinos, Brumley, TIE, NDSS 2011](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)
- [Noonan, Loginov, Cok, Polymorphic Type Inference for Machine Code, PLDI 2016](https://arxiv.org/abs/1603.05495)
- [LLM4Decompile, EMNLP 2024](https://arxiv.org/abs/2403.05286)
- [DecompileBench, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, When LLM Decompilers Recompile More and Preserve Less, 2026](https://arxiv.org/abs/2609.05370)
- [Ghi chú về trình cung cấp Ghidra của rea 6.5.0](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
- Mã nguồn rea 6.5.0 tại tag `3b79af2` mà bài học này mô tả: `src/domain/functionComparisonDimensions.ts`, `website/evidence/dx-ball-sound-pan.md`.
