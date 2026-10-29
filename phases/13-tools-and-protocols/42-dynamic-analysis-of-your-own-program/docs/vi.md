# Phân tích động chương trình của chính bạn: Trình gỡ lỗi, Vết thực thi và Xác thực các nhận định tĩnh

> Một lần chạm điểm dừng là một quan sát thực tế về một lần chạy của một bản dựng cụ thể. Bài học này xây dựng khung kiểm thử biến những quan sát đó thành các phán quyết xác nhận, bác bỏ, hoặc không quan sát được đối với các nhận định mà các bài học phân tích tĩnh đã đưa ra.

**Type:** Build
**Languages:** Python, C, mã assembly arm64, ngôn ngữ lệnh lldb
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 35 (evidence, confidence, and unknowns)
**Time:** ~130 phút

## Learning Objectives

- Giải thích những gì một trình gỡ lỗi đòi hỏi từ hệ điều hành trên Linux (`ptrace`) và trên macOS (các cổng tác vụ Mach task ports và exception ports), và lý do vì sao một shell nằm trong sandbox không thể khởi chạy tiến trình bị gỡ lỗi.
- Thiết lập các điểm dừng phần mềm (software breakpoints) theo tên và theo địa chỉ trong danh sách mã máy, thiết lập điểm theo dõi phần cứng (hardware watchpoint), và đọc các giới hạn vật lý do bộ xử lý áp đặt lên cả hai.
- Điều khiển lldb ở chế độ hàng loạt (batch mode) từ Python bằng một móc dừng (stop hook) nhằm ghi lại các thanh ghi, bộ nhớ và các biến trong khung tại mỗi lần dừng, rồi phân tích bản ghi xuất ra để tái hiện dòng thời gian thực thi.
- Kiểm tra các nhận định tĩnh từ các bài học 32, 33, 34, và 41 đối chiếu với dòng thời gian đó: các thanh ghi đối số, độ lệch trường cấu trúc, nhánh rẽ nào được kích hoạt, và các lời gọi hàm nào thực sự diễn ra.
- Chạy cùng kế hoạch trên bản dựng `-O2`, ghi nhận các điểm dừng được phân giải thành công nhưng không bao giờ chạm tới, và giải thích rõ những gì điều đó chứng minh được và không chứng minh được.
- Lưu trữ và nạp lại một tệp kết xuất lõi (core file), đồng thời chỉ rõ những thành phần mà tệp kết xuất kiểu ngăn xếp (stack-style core) bỏ sót.
- Ánh xạ tập lệnh giữa gdb và lldb, và định vị `strace`, `ltrace`, `bpftrace`, `dtrace`, cùng `rr` trên đúng hệ điều hành kèm theo phiên bản và giấy phép tương ứng.
- Nêu rõ các quy tắc ký mã (code-signing) trên macOS cho phép lldb gỡ lỗi một tệp nhị phân do bạn tự biên dịch, cùng hai điều kiện khiến nó từ chối.

## Vấn đề thực tế

Bài học 41 đã đọc cấu trúc `struct account` từ DWARF và chỉ ra rằng trường `balance` nằm tại độ lệch 4. Bài học 33 đã dựng đồ thị luồng điều khiển của `classify` và khẳng định nhánh rẽ đầu tiên nhảy tới khối `return -1` khi số dư mang giá trị âm. Bài học 32 khẳng định hai đối số đầu tiên được truyền qua `x0` và `x1`. Mọi câu khẳng định đó đều được dẫn xuất từ các byte nhị phân tĩnh. Chưa có khẳng định nào trong số đó được quan sát thực tế trong lúc chạy.

Sau đó, một AI agent đọc cùng chương trình đó và báo cáo: "hàm main gọi `bump`, `classify`, và `sum_balances`". Bảng ký hiệu liệt kê đầy đủ cả ba hàm, vì vậy câu nói đó nghe có vẻ rất an toàn. Bạn biên dịch chương trình ở mức `-O2`, đặt điểm dừng trên từng hàm, và chạy nó. Tiến trình in ra bốn dòng kết quả rồi thoát. Toàn bộ các điểm dừng đều báo cáo `hit count = 0` (số lần chạm bằng 0).

Agent không hề nói sai về mã nguồn. Nhưng nó đã sai hoàn toàn về bản dựng cụ thể mà nó được yêu cầu phân tích. Trình biên dịch đã nhúng nội tuyến (inlined) cả ba lời gọi hàm và tính toán sẵn hàm `sum_balances(rows, 3)` thành hằng số 315 ngay lúc biên dịch. Cách duy nhất để biết chắc chắn điều này trên một tệp nhị phân cụ thể tại một máy cụ thể là chạy nó dưới sự giám sát của trình gỡ lỗi và trực tiếp quan sát.

Bài học này xây dựng một khung kiểm thử thực hiện việc quan sát đó. Nó biên dịch `account.c`, đọc danh sách mã máy, lập kế hoạch cho một phiên lldb, thực thi phiên gỡ lỗi, và chấm điểm từng nhận định tĩnh dựa trên những gì các điểm dừng hiển thị. Kết quả chấm điểm có ba phán quyết: `confirmed` (xác nhận - điểm dừng chứng minh nhận định đúng trong lần chạy này), `refuted` (bác bỏ - điểm dừng chứng minh nhận định sai), và `not-observed` (không quan sát được - không có điểm dừng nào kiểm chứng được nó, kèm lý do rõ ràng). Bài học 35 sau đó sẽ lưu trữ từng phán quyết dưới dạng bản ghi `observed` với thẩm quyền `controlled-replay`.

```figure
re-42-run-timeline
```

## Trình gỡ lỗi là ba dịch vụ hệ thống và một bảng theo dõi

Một trình gỡ lỗi thực hiện bốn thao tác trên một tiến trình mà nó không sở hữu mã nguồn: dừng tiến trình, đọc/ghi thanh ghi, đọc/ghi bộ nhớ, và nhận thông báo khi tiến trình dừng lại tiếp theo. Hạt nhân hệ điều hành cung cấp mỗi thao tác này như một dịch vụ, và các dịch vụ này khác nhau giữa các hệ điều hành.

Trên Linux, dịch vụ này là lời gọi hệ thống `ptrace(2)`. Tài liệu mô tả nó "cung cấp phương tiện để một tiến trình (tracer) quan sát và kiểm soát việc thực thi của một tiến trình khác (tracee)". Thao tác `PTRACE_TRACEME` cho phép tiến trình con tuyên bố cha nó sẽ theo dõi nó, `PTRACE_ATTACH` gắn vào tiến trình đang chạy, `PTRACE_PEEKDATA` đọc một từ máy trong bộ nhớ của tiến trình con, `PTRACE_GETREGS` đọc các thanh ghi, và `PTRACE_SINGLESTEP` cho chạy từng chỉ lệnh đơn lẻ. Mô-đun bảo mật Yama giới hạn thao tác gắn: tệp `/proc/sys/kernel/yama/ptrace_scope` ở giá trị mặc định 1 quy định "tiến trình đích bắt buộc phải là hậu duệ của bên gọi", mức 2 chỉ cho phép quản trị viên, và mức 3 cấm toàn bộ.

Trên macOS, `ptrace` là một giao diện rất hạn chế. Tài liệu liệt kê `PT_TRACE_ME`, `PT_ATTACHEXC`, `PT_STEP`, `PT_CONTINUE`, `PT_KILL`, và nêu rõ với `PT_ATTACHEXC` thì "các tín hiệu từ con sẽ chuyển tới cha dưới dạng các ngoại lệ Mach (Mach exceptions)". Bộ nhớ và thanh ghi hoàn toàn không đi qua `ptrace`. Công cụ `debugserver` của lldb gọi `task_for_pid()` để lấy cổng tác vụ Mach (Mach task port) của tiến trình, sau đó gọi `task_set_exception_ports()` để các điểm dừng được gửi về dưới dạng thông điệp ngoại lệ. Daemon hệ thống `taskgated` sẽ quyết định quyền hạn gọi `task_for_pid`.

Bạn có thể thấy rõ ràng cái giá của thiết kế này: Một shell chạy bên trong một Seatbelt sandbox sẽ bị chặn hoàn toàn. Cùng một kế hoạch gỡ lỗi tạo ra 11 điểm dừng bên ngoài sandbox sẽ gặp lỗi sau khi chạy bên trong:

```text
(lldb) run
error: process exited with status -1 (attach failed ((os/kern) invalid address))
```

Tiến trình đã được tạo ra nhưng trình gỡ lỗi không có quyền gắn vào nó. Lỗi này là do quyền hạn môi trường của trình gỡ lỗi, không phải lỗi của chương trình. Khung kiểm thử ghi nhận nó là một lỗi khởi chạy, với 0 điểm dừng và không có phán quyết nào được đưa ra.

Bảng theo dõi là bảng điểm dừng (breakpoint table). Trình gỡ lỗi ghi nhớ byte chỉ lệnh gốc mà nó đã ghi đè tại mỗi địa chỉ, để nó có thể khôi phục lại khi bạn tiếp tục chạy (`continue`), đồng thời đếm số lần chạm điểm dừng. Lệnh `breakpoint list` in ra bảng này, và khung kiểm thử đọc nó ở cuối mỗi lần chạy.

## Điểm dừng: Thay thế một byte hoặc một từ máy

Một điểm dừng phần mềm (software breakpoint) là một chỉ lệnh khiến bộ xử lý phát sinh bẫy (trap), được ghi đè lên các byte đầu tiên của chỉ lệnh mà bạn muốn dừng lại. Trên x86-64, đó là chỉ lệnh `INT3`, opcode `CC`, dài 1 byte. Trên arm64, đó là `BRK #imm16`, chỉ lệnh 4 byte phát sinh ngoại lệ ghi nhận trong thanh ghi `ESR_ELx` với mã `EC = 0x3C`. Địa chỉ trả về ưu tiên chính là địa chỉ của chính chỉ lệnh `BRK` đó, đó là lý do điểm dừng báo cáo địa chỉ của chỉ lệnh bị thay thế chứ không phải địa chỉ của lệnh tiếp theo.

lldb thiết lập điểm dừng theo tên, theo dòng mã nguồn, hoặc theo địa chỉ. Khi đặt theo tên, nó tự động bỏ qua đoạn mở đầu (prologue):

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O0`bump + 12 at account.c:13:19, address = 0x00000001000004bc
```

Vị trí `bump + 12` là chỉ lệnh thứ tư. Ba lệnh trước đó là `sub sp, sp, #0x10`, `str x0, [sp, #0x8]`, và `str w1, [sp, #0x4]`, là đoạn thiết lập khung ngăn xếp mà bảng dòng DWARF gán cho dòng mở đầu hàm. Tại điểm dừng này, `x0` và `x1` vẫn chứa các đối số gốc vì prologue chỉ sao chép chúng ra ngăn xếp.

Khi đặt theo địa chỉ, lldb cần biết địa chỉ đó thuộc về mô-đun nào. Một lệnh trần `breakpoint set -a 0x100000504` trước khi tiến trình tồn tại chỉ là một địa chỉ nạp không có phân đoạn phía sau, và nó sẽ ở trạng thái `unresolved` suốt quá trình chạy với `hit count = 0`. Cú pháp chuẩn mà khung kiểm thử sử dụng chỉ định rõ tên mô-đun:

```text
(lldb) breakpoint set -a 0x100000504 -s account-O0
Breakpoint 4: address = 0x0000000100000504
```

Nhờ cờ `-s`, con số được hiểu là địa chỉ tệp bên trong `account-O0`, lldb tự động cộng thêm độ dời nạp (load slide), và sau khi chạy xong điểm dừng báo cáo `resolved, hit count = 2`. Danh sách mã máy từ bài học 33 cung cấp các địa chỉ tệp, vì vậy đây là hình thức kết nối trực tiếp bản dịch ngược tĩnh với một điểm dừng động.

### Giải mã nhánh rẽ bằng tay

Nhánh rẽ có điều kiện đầu tiên trong `classify` ở `-O0` là từ máy `36f800a8` tại `0x1000004f0`, được `llvm-objdump` hiển thị là `tbz w8, #0x1f, 0x100000504`. Giải mã chỉ lệnh từ cấu trúc mã hóa Arm A64:

1. Bit 31 là `b5` (bit cao của số bit kiểm tra), giá trị là 0.
2. Các bit 30 đến 25 là `011011` (nhóm test-and-branch). Bit 24 là 0, chọn `TBZ` (Test and Branch if Zero) thay vì `TBNZ`.
3. Các bit 23 đến 19 là `b40` (năm bit thấp): `11111` = 31. Kết hợp với `b5 = 0`, bit được kiểm tra là bit 31, tức bit dấu của một số 32-bit.
4. Các bit 18 đến 5 là `imm14`: `00000000000101` = 5. Độ dời nhánh là `imm14 * 4` = 20 = `0x14`, do đó đích đến là `0x1000004f0 + 0x14 = 0x100000504`.
5. Các bit 4 đến 0 là thanh ghi `Rt`: `01000` = 8, tức `w8`.

Thanh ghi `w8` được nạp trước đó một lệnh bằng `ldr w8, [x8, #0x4]`, chính là `a->balance` tại độ lệch 4. Lệnh `TBZ` nhảy nhánh khi bit được kiểm tra bằng 0 (tức bit dấu không bật, `balance >= 0`). Mã nguồn gốc viết `if (a->balance < 0) return -1;`, do đó trình biên dịch đã đảo ngược điều kiện: nhánh rẽ được kích hoạt sẽ bỏ qua khối `return -1`, còn nhánh rơi thẳng (fall-through) tại `0x1000004f4` sẽ bước vào khối đó. Nhận định tĩnh mà khung kiểm thử mang theo là một nhận định cạnh kèm vị từ: nhánh rẽ kích hoạt khi `a->balance >= 0`, rơi thẳng trong trường hợp ngược lại.

## Điểm theo dõi phần cứng (Watchpoints) rất ít và dựa trên phần cứng

Một điểm theo dõi (watchpoint) sẽ dừng tiến trình khi một vùng bộ nhớ tại địa chỉ bị truy cập. Nó không thể là một chỉ lệnh bị thay thế, vì bất kỳ chỉ lệnh nào trong chương trình cũng có thể thực hiện thao tác truy cập đó. Bộ xử lý cung cấp một số lượng nhỏ các bộ so sánh địa chỉ phần cứng. Trên x86-64, đó là các thanh ghi gỡ lỗi `DR0` đến `DR3` (tối đa 4 điểm dừng phần cứng), và `DR7` để kích hoạt điều kiện (thực thi, ghi, hoặc đọc-ghi) cùng kích thước (1, 2, 4, 8 byte). Trên arm64, đó là các cặp thanh ghi `DBGWVR<n>_EL1` và `DBGWCR<n>_EL1` (hỗ trợ từ 2 đến 64 điểm theo dõi tùy vi kiến trúc).

lldb báo cáo số lượng hỗ trợ trên máy hiện tại:

```text
(lldb) watchpoint set variable a->balance
Watchpoint created: Watchpoint 1: addr = 0x16fdfe00c size = 4 state = enabled type = m
(lldb) watchpoint list
Number of supported hardware watchpoints: 4
```

Đúng 4 điểm theo dõi phần cứng. Địa chỉ được theo dõi `0x16fdfe00c` chính là `x0 + 4`: con trỏ cấu trúc tại điểm dừng `bump` cộng với độ lệch DWARF của `balance`. `size = 4` là kiểu `int`. `type = m` nghĩa là theo dõi thao tác sửa đổi (modify). Khi hết ô nhớ phần cứng, lldb sẽ từ chối thiết lập thêm.

Điểm theo dõi dữ liệu báo cáo sau khi thao tác truy cập đã hoàn tất. Bộ nhớ đã bị ghi đè trước khi ngoại lệ gỡ lỗi được phát sinh, và lldb duy trì bộ nhớ đệm riêng để in ra cả giá trị cũ lẫn giá trị mới:

```text
Watchpoint 1 hit:
old value: 250
new value: 255
* thread #1, queue = 'com.apple.main-thread', stop reason = watchpoint 1
    frame #0: 0x00000001000004d0 account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:14:12
```

Điểm dừng nằm tại `0x1000004d0`, ngay sau chỉ lệnh ghi `str w8, [x9, #0x4]`.

## Bước lệnh, Móc dừng và Chế độ hàng loạt

Ngữ pháp lệnh của lldb có dạng `<danh từ> <động từ> [-tùy chọn] [đối số]`. Các động từ bước lệnh gồm `thread step-in` (`s`), `thread step-over` (`n`), `thread step-inst` (`si`), `thread step-out` (`finish`). Khung kiểm thử trong bài học này không bước lệnh thủ công mà dừng tại các địa chỉ đã lên kế hoạch và để móc dừng (stop hook) tự động thu thập dữ liệu:

```text
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
```

Các lệnh này tự động chạy theo thứ tự tại mọi lần dừng. Lệnh `register read` in các thanh ghi đối số và con trỏ lệnh. Lệnh `memory read` đọc 24 từ máy 4-byte dạng hex từ địa chỉ trong `x0` (tương ứng 96 byte, ba phần tử của `struct account`). Lệnh `frame variable -P 1` in các đối số và biến cục bộ, mở rộng con trỏ một cấp để hiển thị các trường của cấu trúc.

Chế độ hàng loạt được chạy qua `lldb -b`. Cấu hình quan trọng cần thiết lập là: `settings set interpreter.stop-command-source-on-error false`. Nếu không có nó, lệnh `continue` đầu tiên sau khi tiến trình đã thoát sẽ làm gián đoạn toàn bộ tệp kịch bản và lệnh `breakpoint list` ở cuối sẽ không bao giờ được chạy. Khi tắt cờ này, lldb chỉ in cảnh báo và tiếp tục thực thi các lệnh còn lại.

Các điểm dừng có cờ `--auto-continue` sẽ không kích hoạt stop hook và không báo cáo điểm dừng. Vì vậy, kế hoạch gỡ lỗi sử dụng các điểm dừng bình thường kết hợp với một hạn ngạch các lệnh `continue` liên tiếp.

## Kế hoạch xuất phát từ danh sách mã máy

Hàm `Plan.commands()` xây dựng tệp lệnh từ bản dịch ngược `llvm-objdump -d`. Đối với bản dựng `account-O0`:

```text
breakpoint set -n bump
breakpoint set -n classify
breakpoint set -n sum_balances
breakpoint set -a 0x100000504 -s account-O0
breakpoint set -a 0x1000004f4 -s account-O0
breakpoint set -a 0x100000524 -s account-O0
breakpoint set -a 0x100000514 -s account-O0
target stop-hook add -o "register read x0 x1 pc" -o "memory read -s4 -fx -c24 $x0" -o "frame variable -P 1"
run
watchpoint set variable a->balance
watchpoint list
continue
...
breakpoint list
watchpoint list -v
```

Ba điểm dừng theo tên là điểm nhập hàm. Bốn điểm dừng theo địa chỉ là đích đến và nhánh rơi thẳng của từng nhánh rẽ có điều kiện trong `classify`. Điểm theo dõi được đặt tại lần dừng đầu tiên trong `bump`. Mười sáu lệnh `continue` theo sau đảm bảo tiến trình chạy hết vòng đời.

Các nhận định tĩnh được xây dựng bởi `claims_for()`:

| Mã định danh | Loại nhận định | Nội dung nhận định | Nguồn gốc |
|---|---|---|---|
| `arg-registers` | argument-register | Tại mỗi điểm nhập hàm, các đối số đầu tiên nằm trong `x0` và `x1` | Bài 32 và chuẩn AAPCS64 |
| `field-offsets` | field-offset | Cấu trúc `account` rộng 32 byte: `kind` tại 0, `balance` tại 4, `code` tại 8, `rate` tại 16, `label` tại 24 | Bài 41 và dữ liệu DWARF |
| `branch-1` | branch-taken | `tbz` tại `0x1000004f0` nhảy tới `0x100000504` khi `balance >= 0`, ngược lại rơi xuống `0x1000004f4` | Bài 33 và bài 34 |
| `branch-2` | branch-taken | `b.le` tại `0x100000510` nhảy tới `0x100000524` khi `balance <= 200`, ngược lại rơi xuống `0x100000514` | Bài 33 và bài 34 |
| `calls` | call-edge | `main` gọi `bump` 1 lần, `classify` 3 lần, `sum_balances` 1 lần | Mã nguồn và bảng ký hiệu |
| `store-once` | store | `bump` ghi vào `a->balance` đúng một lần với giá trị bằng giá trị cũ cộng `by` | Bài 34 |
| `stride` | struct-stride | `list[i].balance` nằm tại `list + 32*i + 4` | Bài 41 và quy tắc System V |
| `output` | behaviour | Chương trình in ra `class 0 1 -1` và `sum 315` | Logic số học của mã nguồn |

## Lần chạy ở mức -O0: Mười một lần dừng

Lần dừng đầu tiên là điểm dừng tại hàm `bump`. Stop hook in ra dữ liệu:

```text
      x0 = 0x000000016fdfe008
      x1 = 0x0000000000000005
      pc = 0x00000001000004bc  account-O0`bump + 12 at account.c:13:19
0x16fdfe008: 0x00000073 0x000000fa 0x00000009 0x00000000
0x16fdfe018: 0x00000000 0x3ff40000 0x00000719 0x00000001
0x16fdfe028: 0x0000006c 0xffffffd8 0x00000003 0x00000000
0x16fdfe038: 0x00000000 0x40000000 0x00000721 0x00000001
(account *) a = 0x000000016fdfe008 {
  kind = 's'
  balance = 250
  code = 9
  rate = 1.25
  label = 0x0000000100000719 "savings"
}
(int) by = 5
```

Đối chiếu tám từ máy của hàng đầu tiên với độ lệch DWARF: Từ máy 0 tại offset 0 là `0x73` (`'s'`, trường `kind`), theo sau bởi ba byte đệm. Từ máy 1 tại offset 4 là `0xfa` = 250 (`balance`). Từ máy 2 tại offset 8 là `0x9` (`code`, kiểu `short`). Từ máy 3 tại offset 12 là byte đệm căn chỉnh cho kiểu `double`. Từ máy 4 và 5 tại offset 16 là biểu diễn IEEE-754 của số thực 1.25. Từ máy 6 và 7 tại offset 24 là con trỏ `label` trỏ vào phân đoạn `__TEXT`. Tám từ máy tiếp theo là hàng thứ hai (`loan`), nằm cách đúng 32 byte. Mọi con số đều khớp với `frame variable`, do đó nhận định `field-offsets` được ghi nhận là `confirmed`.

Giá trị `x0` bằng `a` và `x1` bằng `by`, do đó nhận định `arg-registers` được xác nhận tại điểm nhập này.

Lần dừng thứ hai là điểm theo dõi (watchpoint): giá trị cũ 250, giá trị mới 255, `by` = 5. Chỉ có duy nhất một lần chạm trong toàn bộ lần chạy, do đó `store-once` được xác nhận.

Tiếp theo là ba lần kích hoạt hàm `classify`, mỗi lần gồm một điểm dừng nhập hàm và các điểm dừng tại nhánh rẽ:

| Lần dừng | Địa chỉ | Lý do dừng | Nhận định được kiểm chứng |
|---|---|---|---|
| 2 | `0x1000004e8` | Điểm dừng 2.1 | Nhập hàm, `a->balance` = 100, `x0` = `a` |
| 3 | `0x100000504` | Điểm dừng 4.1 | Nhánh `tbz` kích hoạt (100 >= 0), xác nhận `branch-1` |
| 4 | `0x100000524` | Điểm dừng 6.1 | Nhánh `b.le` kích hoạt (100 <= 200), xác nhận `branch-2` |
| 5 | `0x1000004e8` | Điểm dừng 2.1 | Nhập hàm, `a->balance` = 255 |
| 6 | `0x100000504` | Điểm dừng 4.1 | Nhánh `tbz` kích hoạt (255 >= 0) |
| 7 | `0x100000514` | Điểm dừng 7.1 | Nhánh `b.le` rơi thẳng (255 > 200) |
| 8 | `0x1000004e8` | Điểm dừng 2.1 | Nhập hàm, `a->balance` = -40 |
| 9 | `0x1000004f4` | Điểm dừng 5.1 | Nhánh `tbz` rơi thẳng (-40 < 0) |
| 10 | `0x100000544` | Điểm dừng 3.1 | Nhập hàm `sum_balances`, `x1` = 3 |

Tại lần dừng 10, hook đọc 24 từ máy từ mảng `list`. Các giá trị tại độ lệch 4, 36, và 68 lần lượt là 100, 255, và -40, có tổng bằng 315, khớp với dòng kết quả in ra màn hình. Toàn bộ 8 nhận định đều kết thúc với phán quyết `confirmed`.

## Lần chạy ở mức -O2: Không có lần dừng nào

Cùng mã nguồn đó ở `-O2` tạo ra một chương trình hoàn toàn khác: thân hàm `main` chỉ có bốn lệnh gọi hàm `printf`. Các kết quả tính toán đều bị thay thế bằng các hằng số nạp trực tiếp vào thanh ghi. Bảng ký hiệu vẫn giữ tên ba hàm vì chúng là hàm xuất khẩu bên ngoài, nhưng hàm `main` không hề gọi chúng.

Kế hoạch cho bản dựng này chỉ có ba điểm dừng theo tên và không có điểm dừng nhánh rẽ (vì hàm `classify` ở `-O2` được tối ưu hóa thành chuỗi chỉ lệnh không rẽ nhánh: `ldr`, `cmp`, `cset`, `cmp`, `csinv`, `ret`).

Kết quả thực thi:

```text
(lldb) breakpoint set -n bump
Breakpoint 1: where = account-O2`bump at account.c:13:16, address = 0x0000000100000460
(lldb) run
sizeof 32 align 8
offsets 0 4 8 16 24
class 0 1 -1
sum 315
Process 48444 exited with status = 0 (0x00000000)
(lldb) watchpoint set variable a->balance
error: Command requires a process which is currently stopped.
(lldb) breakpoint list
1: name = 'bump', locations = 1, resolved = 1, hit count = 0
```

Các điểm dừng đều được phân giải thành công nhưng số lần chạm bằng 0. Các phán quyết:

| Nhận định | Phán quyết | Bản ghi thực tế |
|---|---|---|
| `calls` | refuted (bác bỏ) | Sau một lần chạy hoàn chỉnh thoát với mã 0, số lần chạm của bump, classify, sum_balances đều là 0 |
| `output` | confirmed (xác nhận) | Bốn dòng kết quả in ra đúng và thoát với mã 0 |
| `arg-registers` | not-observed | Không có lần dừng nào kiểm chứng được, hit count = 0 |
| `field-offsets` | not-observed | Không có lần dừng nào kiểm chứng được |
| `store-once` | not-observed | Không dừng trong bump nên không thể đặt watchpoint |
| `stride` | not-observed | Không dừng trong sum_balances |

Phán quyết `refuted` khẳng định rằng trong bản dựng này, `main` không hề gọi ba hàm đó. Điều đó không có nghĩa là ba hàm đó là mã chết hoàn toàn: chúng vẫn được xuất khẩu để các đối tượng khác có thể liên kết gọi tới. Bốn nhận định còn lại mang trạng thái `not-observed` (chưa quan sát được), giữ nguyên trạng thái chưa xác định trong sổ cái bằng chứng cho đến khi có một lần chạy khác kiểm chứng được chúng.

## Tệp kết xuất lõi (Core Dumps)

Một tệp kết xuất lõi (core file) ghi lại toàn bộ bộ nhớ và thanh ghi của tiến trình xuống đĩa để trình gỡ lỗi có thể phân tích sau khi tiến trình đã dừng. Trên macOS, lệnh lldb sau tạo tệp kết xuất theo yêu cầu:

```text
(lldb) process save-core --style stack account-O0-stack.core
Saving 49152 bytes of data for memory region at 0x16fdf4000
```

Tệp tạo ra rộng 53.248 byte. Khi nạp lại tệp kết xuất này cùng với tệp nhị phân gốc:

```text
(lldb) target create "./account-O0" --core "account-O0-stack.core"
(lldb) bt
* thread #1, stop reason = ESR_EC_BRK_AARCH64 (fault address: 0x0)
  * frame #0: 0x00000001000004bc account-O0`bump(a=0x000000016fdfe008, by=5) at account.c:13:19
    frame #1: 0x0000000100000640 account-O0`main at account.c:40:5
(lldb) frame variable *a
(account) *a = (kind = 's', balance = 250, code = 9, rate = 1.25, label = "")
(lldb) frame variable a->label
(const char *) a->label = 0x0000000100000719 ""
```

Dấu vết ngăn xếp (backtrace) và các trường số nguyên nguyên vẹn vì chúng nằm trên ngăn xếp. Tuy nhiên trường `label` hiển thị chuỗi rỗng: con trỏ `0x100000719` hoàn toàn chính xác, nhưng chuỗi ký tự thực tế nằm trong phân đoạn `__TEXT` của tệp thực thi, vốn không được bao gồm trong tệp kết xuất chỉ lưu vùng ngăn xếp (`--style stack`). Kiểu kết xuất toàn bộ (`--style full`) sẽ lưu toàn bộ bộ nhớ nhưng tốn tới 2.1 GB do ánh xạ bộ nhớ đệm chia sẻ của hệ thống.

## So sánh gdb và lldb song song

Hai trình gỡ lỗi thực hiện cùng công việc nhưng với cú pháp khác nhau:

| Tác vụ | gdb | lldb | Tên viết tắt lldb |
|---|---|---|---|
| Đặt điểm dừng tại một hàm | `break bump` | `breakpoint set -n bump` | `b bump` |
| Đặt điểm dừng tại địa chỉ trong mô-đun | `break *0x1000004f0` | `breakpoint set -a 0x1000004f0 -s account-O0` | |
| Chạy lệnh tự động tại mỗi điểm dừng | `define hook-stop` ... `end` | `target stop-hook add -o '...'` | |
| Liệt kê điểm dừng kèm số lần chạm | `info breakpoints` | `breakpoint list` | `br l` |
| Theo dõi ghi biến | `watch a->balance` | `watchpoint set variable a->balance` | `wa s v a->balance` |
| Liệt kê điểm theo dõi | `info watchpoints` | `watchpoint list -v` | `wa l -v` |
| Đọc thanh ghi | `p $x0` | `register read x0` | |
| Đọc bộ nhớ theo từ máy | `x/8xw 0x16fdfe008` | `memory read -s4 -fx -c8 0x16fdfe008` | `x -s4 -fx -c8 0x16fdfe008` |
| Hiển thị đối số hàm | `info args` | `frame variable --no-locals` | |
| Bước từng chỉ lệnh đơn | `stepi` | `thread step-inst` | `si` |
| Chạy đến khi hàm kết thúc | `finish` | `thread step-out` | `finish` |
| Lưu tệp kết xuất lõi | `gcore core` | `process save-core --style stack core` | |
| Mở tệp kết xuất lõi | `gdb account-O0 core` | `lldb -c core account-O0` | |

Ưu tiên sử dụng `frame variable` thay vì `p` (print) trong các bộ kiểm thử tự động vì nó đọc trực tiếp các vị trí DWARF và không bao giờ thực thi mã bên trong tiến trình đích.

## Truy vết hệ thống: Những gì chương trình yêu cầu hạt nhân

| Công cụ | Hệ điều hành | Phiên bản | Giấy phép | Nội dung ghi lại |
|---|---|---|---|---|
| `strace` | Linux | 7.2 | LGPL-2.1-or-later | Các lời gọi hệ thống (system calls) kèm đối số và kết quả trả về |
| `ltrace` | Linux | 0.7.3 | GPL-2.0 | Các lời gọi hàm thư viện động |
| `bpftrace` | Linux | v0.27.0 | Apache-2.0 | Các đầu dò hạt nhân và người dùng qua eBPF |
| `dtrace` / `dtruss` | macOS | `Sun D 1.19` | CDDL | Các đầu dò DTrace; `dtruss` in các lời gọi hệ thống |

Truy vết trả lời câu hỏi: "lần chạy này có phát sinh lệnh ghi cho từng dòng `printf` không", trong khi trình gỡ lỗi trả lời: "thanh ghi `x1` có mang giá trị 5 tại hàm `bump` không".

## Ghi lại và Phát lại (Record and Replay)

Công cụ `rr` (chỉ có trên Linux) ghi lại toàn bộ lần chạy một lần duy nhất và cho phép phát lại tất định dưới gdb, hỗ trợ thực thi ngược chiều (reverse execution): đặt điểm theo dõi và chạy ngược lại bằng `reverse-cont` để tìm chỉ lệnh đã ghi dữ liệu. Nó đòi hỏi bộ đếm hiệu năng phần cứng trên CPU Intel/AMD hoặc Neoverse.

Trên macOS, bản ghi xuất ra từ khung kiểm thử chính là bản ghi kiểm chứng: cùng kế hoạch trên cùng tệp nhị phân sẽ tạo ra cùng chuỗi điểm dừng với cùng dữ liệu trạng thái.

## Chỉ gỡ lỗi các bản dựng do chính bạn tạo ra trên macOS

Trên Apple silicon, trình liên kết tự động áp dụng cơ chế ký mã ad-hoc (`-adhoc_codesign`). Lệnh `codesign -dvvv account-O0` hiển thị các cờ `0x20002(adhoc,linker-signed)`. Tệp nhị phân không cần bất kỳ quyền đặc biệt (entitlements) nào để lldb có thể gỡ lỗi nó.

Hai quyền (entitlements) kiểm soát việc gỡ lỗi:
- `com.apple.security.cs.debugger`: Dành cho chính trình gỡ lỗi (cho phép gắn vào các tiến trình khác).
- `com.apple.security.get-task-allow`: Dành cho tiến trình đích. Bản dựng debug của Xcode tự động gắn cờ này, nhưng bản dựng xuất khẩu thì bị gỡ bỏ. Nếu một tiến trình bật cơ chế bảo vệ Hardened Runtime mà không có quyền này, hệ thống sẽ từ chối cung cấp cổng tác vụ. Một bản dựng bằng `clang` thông thường không bật hardened runtime nên không gặp phải hạn chế này.

## Xây dựng hệ thống

Chạy chương trình minh họa:

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
```

Đầu ra là một báo cáo JSON phân tích các bản ghi thực thi: đối với từng bản dựng, nó liệt kê các điểm dừng kèm số lần chạm, số điểm theo dõi phần cứng, dữ liệu chi tiết của từng lần dừng, trạng thái thoát, các nhận định kèm phán quyết xác nhận/bác bỏ, và kiểm tra vòng lặp tệp kết xuất lõi.

Cấu trúc định dạng kỳ vọng:

```json
{
  "program": "account.c",
  "variants": {
    "O0": {
      "binary": "account-O0",
      "breakpoints": {"1": {"spec": "name=bump", "address": "0x1000004bc", "resolved": true, "hits": 1}, "...": "..."},
      "hardwareWatchpoints": 4,
      "watchpoints": {"1": {"address": "0x16fdfe00c", "size": 4, "type": "m", "hits": [[250, 255]], "hitCount": 1}},
      "stops": [{"index": 0, "pc": "0x1000004bc", "function": "bump", "kind": "entry", "registers": {"x0": "0x16fdfe008", "x1": "0x5"}, "...": "..."}],
      "exitStatus": 0,
      "claims": [{"id": "arg-registers", "verdict": "confirmed", "confirmed": 5, "refuted": 0, "not-observed": 0, "...": "..."}],
      "evidence": [{"claim": "arg-registers", "verdict": "confirmed", "stop": 0, "observation": "bump: x0 = 0x16fdfe008 and a = 0x000000016fdfe008, x1 = 0x5 and by = 5"}]
    },
    "O2": {"stops": [], "claims": [{"id": "calls", "verdict": "refuted", "...": "..."}], "...": "..."}
  },
  "core": {"savedAt": {"function": "bump", "pc": "0x1000004bc"}, "labelString": "", "...": "..."},
  "types": {"python": "3.9.6", "size": 32, "...": "..."},
  "sandbox": {"stops": 0, "errors": ["process exited with status -1 (attach failed ((os/kern) invalid address))"]},
  "absentTools": ["bpftrace", "gdb", "llvm-objdump", "ltrace", "rr", "strace"]
}
```

Hãy đọc mã nguồn trong `code/main.py` theo trình tự sau:

1. `Listing` phân tích mã máy thành các hàm và chỉ lệnh; `conditional_branches()` tìm kiếm các nhánh rẽ điều kiện.
2. `struct_layout()` đọc thông tin cấu trúc từ dữ liệu DWARF.
3. `lift_condition()` chuyển đổi nhánh rẽ có điều kiện thành biểu thức vị từ.
4. `Plan.commands()` tạo kịch bản lệnh cho lldb từ danh sách mã máy.
5. `parse_transcript()` phân tích kết quả xuất ra thành các bản ghi `Stop`.
6. `claims_for()` định nghĩa tám nhận định tĩnh, và `evaluate()` chấm điểm từng nhận định qua các hàm `_check_*`.
7. `rollup()` tổng hợp các bằng chứng thành một phán quyết duy nhất cho mỗi nhận định (trong đó bác bỏ được ưu tiên hơn xác nhận).

## Sử dụng hệ thống

Quy trình sử dụng trên một chương trình của chính bạn:

1. Biên dịch với cờ `-g` ở mức tối ưu hóa bạn dự kiến phát hành. Giữ tệp nhị phân, thông tin DWARF và bản dịch ngược cạnh nhau.
2. Viết các nhận định tĩnh trước khi chạy: tên hàm, thanh ghi, địa chỉ, giá trị kỳ vọng, và nguồn gốc tài liệu. Tuyệt đối không viết nhận định sau khi đã thấy kết quả dừng.
3. Sinh kế hoạch gỡ lỗi tự động từ danh sách mã máy và kiểm tra lại các địa chỉ điểm dừng.
4. Chạy chế độ hàng loạt một lần duy nhất và lưu lại toàn bộ bản ghi xuất ra. Bản ghi này chính là bằng chứng xác thực.
5. Đánh giá phán quyết: xem xét các mục `not-observed` và lý do của chúng trước khi xem các mục được xác nhận.
6. Chạy trên bản dựng tối ưu hóa với cùng các nhận định để phát hiện các vị trí bị tối ưu nhúng nội tuyến.
7. Chuyển các phán quyết xác nhận và bác bỏ vào sổ cái bằng chứng dưới dạng `observed` với thẩm quyền `controlled-replay`.

## Phòng thí nghiệm tương tác

### Lab A: Đọc dữ liệu điểm dừng bằng tay

Khởi động Python từ thư mục `code`:

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program/code
python3 -q
```

Thực thi:

```python
from main import *
t = load_transcript("session-O0.lldb.txt")
s = t.stops[0]
hex(s.pc), s.function, s.reason, s.registers
[hex(w) for w in s.memory[:8]]
s.struct_pointer().fields
```

Kiểm tra từ máy chứa giá trị 250 và các từ máy chứa số thực 1.25. Sau đó xem xét `t.stops[1].watch_hit` để thấy giá trị thay đổi từ 250 lên 255 ngay sau lệnh lưu.

### Lab B: Nhánh rẽ và Điểm đến kế tiếp

```python
plan = make_plan("O0")
plan.branches()
classify = plan.listing.function("classify")
[lift_condition(classify, b, "a->balance")[0] for b in plan.branches()]
[(hex(s.pc), s.struct_pointer().field_int("balance")) for s in t.stops_in("classify")]
```

Đối chiếu từng điểm dừng tại nhánh rẽ với giá trị số dư tương ứng để xác nhận xem vị từ có dự đoán chính xác khối kế tiếp hay không.

### Lab C: Bản dựng không bao giờ dừng

```python
o2 = load_transcript("session-O2.lldb.txt")
len(o2.stops), o2.exit_status, o2.hits()
o2.errors[:3]
make_plan("O2").branches()
rollup(evaluate(make_plan("O2"), o2, struct_layout(open("fixtures/account-O2.dwarfdump.txt").read(), "account")))
```

Quan sát cách mà phán quyết `refuted` được gán cho lời gọi hàm bị nhúng nội tuyến và trạng thái `not-observed` cho các nhận định không thể kiểm chứng.

### Lab D: Cố ý phá vỡ bằng chứng để kiểm tra bộ chấm điểm

```python
text = open("fixtures/session-O0.lldb.txt").read().replace("x1 = 0x0000000000000005", "x1 = 0x0000000000000006", 1)
broken = parse_transcript(text)
[e.to_dict() for e in evaluate(plan, broken, struct_layout(open("fixtures/account-O0.dwarfdump.txt").read(), "account")) if e.claim == "arg-registers"][:2]
```

Một giá trị thanh ghi bị sửa đổi ngay lập tức biến phán quyết của `arg-registers` thành `refuted`. Một bộ chấm điểm chuẩn mực bắt buộc phải biết báo cáo thất bại khi bằng chứng sai lệch.

## Bài tập thực hành

Mở rộng khung kiểm thử với nhận định về vòng lặp trong `sum_balances`.

Yêu cầu thực hiện:

- Tìm cạnh quay lui của vòng lặp trong `sum_balances` ở danh sách `-O0` bằng `conditional_branches()` và thêm điểm dừng tại đầu vòng lặp (loop header).
- Bổ sung nhận định rằng đầu vòng lặp thực thi `count + 1` lần với `count = 3`.
- Mở rộng stop hook để đọc ô ngăn xếp của biến đếm vòng lặp tại mỗi lần dừng, và viết hàm `_check_loop()` ghi lại chuỗi giá trị của biến đếm.
- Thu xuất lại bản ghi bằng `python3 main.py capture O0` và chạy lại bộ kiểm thử.
- Viết kiểm thử xác nhận rằng nếu chỉnh sửa một giá trị biến đếm trong bản ghi thì nhận định sẽ bị chuyển thành `refuted`.

Thành công đạt được khi báo cáo hiển thị tuần tự các lần dừng đầu vòng lặp kèm giá trị biến đếm chính xác, và bản dựng `-O2` giữ nhận định mới ở trạng thái `not-observed` do vòng lặp đã bị vector hóa hoặc tối ưu hóa hoàn toàn.

## Thành phẩm bài học

Bài học này bao gồm tệp `outputs/skill-dynamic-analysis-of-your-own-program.md`. Sử dụng nó để biến một chương trình bạn đã biên dịch, danh sách dịch ngược và danh sách nhận định tĩnh thành một bản ghi gỡ lỗi hoàn chỉnh kèm phán quyết xác thực cho từng nhận định, có liên kết tới điểm dừng cụ thể và giải trình rõ ràng cho mọi mục chưa quan sát được.

## Kiểm chứng thực tế

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/42-dynamic-analysis-of-your-own-program
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình kiểm chứng cần chứng minh:

- Bộ phân tích danh sách mã máy nhận diện đủ bốn hàm và hai nhánh rẽ có điều kiện của `classify` ở `-O0` kèm đích đến và nhánh rơi thẳng
- Biểu thức nâng điều kiện của `tbz w8, #0x1f` là `a->balance >= 0` và của `subs` cộng `b.le` là `a->balance <= 200`
- Hàm `main` ở `-O0` gọi `bump`, `classify` 3 lần, và `sum_balances`, trong khi ở `-O2` chỉ gọi duy nhất `printf`
- Cấu trúc DWARF giải mã đúng cấu trúc 32 byte với các thành viên tại 0, 4, 8, 16, 24
- Bản ghi `-O0` ghi nhận đúng 11 lần dừng trong các hàm mong đợi, mã thoát 0, và bốn dòng đầu ra
- Dữ liệu hook gắn chính xác vào lần dừng tiếp theo, với `x1 = 5`, bộ nhớ bắt đầu tại `x0`, và chuỗi nhãn
- Bảng điểm dừng mang đúng trạng thái phân giải và số lần chạm: 1, 3, 1, 2, 1, 1, 1
- Điểm theo dõi ghi nhận đúng 4 byte tại `x0 + 4`, chạm 1 lần với giá trị cũ 250 và mới 255
- Bản ghi `-O2` ghi nhận 0 lần dừng, ba điểm dừng phân giải với 0 lần chạm, và không có điểm theo dõi
- Lần chạy trong sandbox thất bại trước mọi điểm dừng với lỗi attach hợp lệ
- Tệp kết xuất lõi giữ nguyên dấu vết ngăn xếp và `balance = 250`, mất chuỗi ký tự phía sau con trỏ `label`
- Toàn bộ các nhận định đều được xác nhận ở `-O0` và nhận định gọi hàm bị bác bỏ ở `-O2`
- Dữ liệu hình ảnh tương tác phản ánh trung thực kết quả của cả hai phiên chạy

## Các kịch bản lỗi trong môi trường thực tế

| Tình huống lỗi | Báo cáo từ công cụ yếu | Điều phân tích chuẩn phải chứng minh |
|---|---|---|
| Điểm dừng phân giải nhưng 0 lần chạm | "Hàm có chạy" hoặc "Hàm là mã chết" | Chạy hết tiến trình, kiểm tra mã thoát, và chỉ gán `refuted` riêng cho cạnh gọi hàm |
| Đọc thanh ghi tại sai lần dừng | "`x1` = 5 bên trong `classify`" | Chỉ so sánh các đối số được khai báo tại điểm dừng nhập hàm |
| Đọc cấu trúc không kèm độ lệch | In một cột các từ máy hex | Từng từ máy phải ánh xạ vào `DW_AT_data_member_location` và đối chiếu với `frame variable` |
| Đánh giá nhánh rẽ chỉ từ mã gợi nhớ | "`tbz` nghĩa là kích hoạt khi âm" | Phải căn cứ vào vị từ được nâng từ mã hóa và điểm dừng tại khối kế tiếp |
| Đặt watchpoint trên biến nằm trong thanh ghi | "Không có thao tác ghi nào diễn ra" | Watchpoint phải đặt trên địa chỉ bộ nhớ; nếu biến không nằm trong bộ nhớ phải gán `not-observed` |
| Dùng điểm dừng tự động chạy tiếp (auto-continue) | Bản ghi rỗng không có điểm dừng | Phải dùng điểm dừng thông thường, stop hook, và hạn ngạch lệnh continue |
| Điểm dừng địa chỉ không có tên mô-đun | `unresolved, hit count = 0` | Phải dùng cờ `-s MODULE` để địa chỉ danh sách mã máy phân giải thành địa chỉ nạp |
| Lỗi lệnh đầu tiên làm dừng toàn bộ batch | Không chạy được `breakpoint list` ở cuối | Phải cấu hình `interpreter.stop-command-source-on-error false` |
| Tệp kết xuất lõi ngăn xếp bị đọc như bộ nhớ đầy đủ | Báo cáo `label = ""` là giá trị của chương trình | Phải phân biệt kiểu kết xuất lõi, con trỏ và phân đoạn mà nó trỏ tới |
| Môi trường shell bị sandbox | "lldb bị hỏng" | Báo cáo lỗi attach như một dữ kiện về môi trường thực thi |

## Quy tắc vận hành

Hãy lập kế hoạch các điểm dừng từ danh sách mã máy tĩnh, chạy một lần duy nhất ở chế độ hàng loạt, lưu giữ bản ghi xuất ra nguyên vẹn, và chấm điểm từng nhận định tĩnh tại mọi điểm dừng có khả năng kiểm chứng nó. Chỉ đưa ra phán quyết xác nhận hoặc bác bỏ từ dữ liệu điểm dừng thực tế, ghi nhận trạng thái chưa quan sát được kèm số lần chạm và lý do cụ thể, và tuyệt đối không bao giờ coi một điểm dừng đã được phân giải là bằng chứng thay thế cho một lần chạy thực tế. Chỉ thực hiện gỡ lỗi trên các bản dựng do chính bạn tạo ra.

## Tài liệu đọc thêm

- [LLDB Tutorial: command grammar, breakpoints, watchpoints, stop hooks](https://lldb.llvm.org/use/tutorial.html)
- [lldb(1) manual page: --batch, --one-line, --source, --core](https://lldb.llvm.org/man/lldb.html)
- [GDB to LLDB Command Map](https://lldb.llvm.org/use/map.html)
- [LLDB Python Reference: SBTarget, SBFrame, SBValue, and the embedded interpreter](https://lldb.llvm.org/use/python-reference.html)
- [LLDB Caveats: the Python build lldb links against](https://lldb.llvm.org/resources/caveats.html)
- [LLVM 23.1.3 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.3)
- [GDB 18.1 release announcement, 2026-09-25](https://sourceware.org/pipermail/gdb-announce/2026/000150.html)
- [GDB manual: Process Record and Replay](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Process-Record-and-Replay.html)
- [GDB manual: Debugging Optimized Code](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Optimized-Code.html)
- [GDB wiki: Permissions on Darwin, code signing a debugger](https://sourceware.org/gdb/wiki/PermissionsDarwin)
- [gdb-patches: Initial AArch64-Darwin support series, October 2025](https://sourceware.org/pipermail/gdb-patches/2025-October/222196.html)
- [Homebrew gdb formula pull request adding arm64 bottles](https://github.com/Homebrew/homebrew-core/pull/209753)
- [ptrace(2), Linux manual page](https://man7.org/linux/man-pages/man2/ptrace.2.html)
- [lldb debugserver MachTask.mm: task_for_pid and task_set_exception_ports](https://github.com/llvm/llvm-project/blob/main/lldb/tools/debugserver/source/MacOSX/MachTask.mm)
- [Intel SDM Volume 2: INT n, INTO, INT3, INT1](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Intel SDM Volume 3: debug registers DR0 to DR7 (Debug, Branch Profile, TSC, and Resource Monitoring Features)](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Arm A64 ISA: BRK, Breakpoint instruction](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/BRK--Breakpoint-instruction-)
- [Arm A64 ISA: TBZ, Test bit and branch if zero](https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/TBZ--Test-bit-and-branch-if-zero-)
- [Arm Architecture Reference Manual, D2.9 Watchpoint exceptions](https://developer.arm.com/documentation/ddi0487/latest/)
- [Apple ld(1): -adhoc_codesign](https://keith.github.io/xcode-man-pages/ld.1.html)
- [Apple ld64 LinkEdit.hpp: CS_LINKER_SIGNED](https://github.com/apple-oss-distributions/ld64/blob/59a99ab60399c5e6c49e6945a9e1049c42b71135/src/ld/LinkEdit.hpp)
- [Apple: Debugging Tool Entitlement, com.apple.security.cs.debugger](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.debugger)
- [Apple: Hardened Runtime](https://developer.apple.com/documentation/security/hardened-runtime)
- [Apple: Resolving common notarization issues, the get-task-allow entitlement](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)
- [Apple: Disabling and Enabling System Integrity Protection](https://developer.apple.com/documentation/security/disabling-and-enabling-system-integrity-protection)
- [Apple Developer Forums: debugserver may not get the taskport of a hardened process](https://developer.apple.com/forums/thread/127544)
- [Apple DevToolsSecurity(8)](https://keith.github.io/xcode-man-pages/DevToolsSecurity.8.html)
- [Apple core(5)](https://keith.github.io/xcode-man-pages/core.5.html)
- [Apple Technical Note TN2124: Mac OS X Debugging Magic, core dumps](https://developer.apple.com/library/archive/technotes/tn2124/_index.html)
- [lldb `process save-core` help text](https://github.com/llvm/llvm-project/pull/143126)
- [core(5), Linux manual page](https://man7.org/linux/man-pages/man5/core.5.html)
- [coredumpctl(1)](https://man7.org/linux/man-pages/man1/coredumpctl.1.html)
- [Apple dtrace(1)](https://keith.github.io/xcode-man-pages/dtrace.1.html)
- [Apple dtruss(1m)](https://keith.github.io/xcode-man-pages/dtruss.1m.html)
- [strace releases](https://github.com/strace/strace/releases)
- [ltrace](https://ltrace.org/)
- [bpftrace CHANGELOG](https://github.com/bpftrace/bpftrace/blob/master/CHANGELOG.md)
- [rr project](https://rr-project.org/)
- [rr 5.9.0 release](https://github.com/rr-debugger/rr/releases/tag/5.9.0)
- [rr wiki: Building And Installing, hardware and VM requirements](https://github.com/rr-debugger/rr/wiki/Building-And-Installing)
- [Clang UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [ISO/IEC 9899:2024 working draft N3220, clause 3.5 undefined behavior](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3220.pdf)
- [DWARF Debugging Information Format Version 5](https://dwarfstd.org/doc/DWARF5.pdf)
- [Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64)](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI)
