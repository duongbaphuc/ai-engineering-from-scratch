# Quán ngữ trình biên dịch: Tối ưu hóa tác động gì đến mã nguồn và cách dịch ngược lại

> Một trình biên dịch tối ưu hóa thay thế các cấu trúc mã nguồn bằng các khuôn mẫu chỉ lệnh có chi phí thấp hơn. Mỗi khuôn mẫu đều có một phép dẫn xuất toán học, và bài học này sẽ xây dựng bộ nhận diện thực hiện phép dẫn xuất ngược đó và đối chiếu kiểm chứng với mã nguồn.

**Type:** Build
**Languages:** Python, C, mã assembly x86-64, mã assembly AArch64
**Prerequisites:** Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode), Phase 13 · 38 (calling conventions)
**Time:** ~150 phút

## Learning Objectives

- Dẫn xuất số ma thuật (magic number) và số dịch bit cho phép chia cho một hằng số bất kỳ bằng chặn Granlund-Montgomery, và khôi phục số chia từ các hằng số do clang sinh ra.
- Xác định năm phương thức mà LLVM dùng để hạ cấp (lowering) một lệnh `switch` và đọc số lượng mục, vị trí bảng cùng các phần tử trực tiếp từ tệp đối tượng.
- Đọc một vòng lặp vector hóa (vectorized loop) dưới dạng các làn (lanes), các biến tích lũy (accumulators), số phần tử trên mỗi bước lặp, mặt nạ đếm số vòng lặp (trip-count mask), phép rút gọn (reduction) và vòng lặp xử lý phần dư (remainder loop).
- Nhận diện lời gọi đuôi (tail calls), các hàm được gọi được nhúng nội tuyến (inlined callees), các lựa chọn không rẽ nhánh (branchless selects), và các vòng lặp được thay thế bằng `memset` hoặc `memcpy`.
- Báo cáo từng quán ngữ (idiom) kèm theo các chỉ lệnh làm bằng chứng và dạng mã nguồn đã giải mã của nó, trên cả x86-64 lẫn AArch64, từ mức tối ưu hóa `-O0` đến `-O3`.
- So sánh kết quả với những gì Ghidra, angr, radare2, rizin, và lldb hiển thị cho cùng các byte thực thi đó.

## Vấn đề thực tế

Bạn biên dịch `idioms.c` ở mức tối ưu hóa `-O2` và hỏi một AI agent xem hàm `div7u` thực hiện công việc gì. Agent đọc chỉ lệnh `imulq $0x24924925, %rax, %rax` và trả lời rằng hàm này nhân với 613566757, dịch bit, rồi cộng. Từng từ đều đúng về mặt cú pháp. Tuy nhiên bản chất hàm đó tính `x / 7`.

Cùng tệp đối tượng đó chứa một lệnh `switch` với bảy trường hợp nhưng không có bất kỳ lệnh nhảy nào, một `switch` với năm trường hợp là một bảng nhảy (jump table), một `switch` với hai trường hợp được chuyển thành `sete` và `cmovne`, cùng một vòng lặp trên `n` số nguyên xử lý tám số nguyên mỗi lần lặp trong hai thanh ghi `xmm` rồi hoàn tất phần còn lại trong một vòng lặp thứ hai. Ở mức `-O0`, cùng mã nguồn đó sinh ra lệnh `divl`, bảng nhảy bảy mục, và một vòng lặp vô hướng (scalar loop). Ở mức `-O1`, câu lệnh switch bảy trường hợp trở thành một bảng các giá trị và vòng lặp vector vẫn chưa xuất hiện.

Các khuôn mẫu này được gọi là quán ngữ trình biên dịch (compiler idioms). Trình biên dịch phát sinh từng khuôn mẫu dựa trên một quy tắc cụ thể trong mã nguồn của nó, và mỗi khuôn mẫu đều có thể được giải mã ngược trở lại cấu trúc mã nguồn ban đầu. Quá trình giải mã này hoàn toàn mang tính số học, và số học thì luôn kiểm chứng được. Bài học 34 đã chứng minh một trường hợp, phép nhân với `0x66666667`. Bài học này sẽ xây dựng bộ nhận diện cho toàn bộ các quán ngữ đó, chạy trên 18 bản dựng từ mã nguồn của chúng ta, và xác minh toàn bộ 373 báo cáo đối chiếu với mã nguồn C.

```figure
re-39-idioms-by-level
```

## Một quán ngữ là một khuôn mẫu đi kèm phép dẫn xuất toán học

Một quán ngữ là một chuỗi chỉ lệnh mà trình biên dịch phát sinh cho một cấu trúc mã nguồn nhất định. Ba yếu tố giúp nhận diện nó: nó có khuôn mẫu cố định, các toán hạng mang các hằng số mà phép dẫn xuất dự đoán trước, và hình thức này khác với những gì lập trình viên tự viết tay.

Bộ nhận diện trong `code/main.py` hoạt động trên văn bản xuất ra từ `llvm-objdump -d -r`, vốn được bài học 33 xác lập như nguồn chuẩn xác định ranh giới chỉ lệnh. Nó phân tích từng hàm thành các chỉ lệnh bao gồm địa chỉ, các byte mã máy, mã gợi nhớ (mnemonic), các toán hạng, và dòng tái định vị (relocation) đi kèm. Nó đọc tệp đối tượng chỉ vì hai mục đích: các byte của phân đoạn `.rodata`, nơi lưu trữ các bảng giá trị, và các mục `.rela.rodata`, nơi lưu đích đến của bảng nhảy trước khi liên kết.

Mỗi phát hiện là một bản ghi `Idiom` chứa trường `kind`, dạng `source` đã giải mã, danh sách địa chỉ `evidence`, từ điển `detail`, và cờ `verified`. Cờ này được đối chiếu từ chính mã nguồn C. Hàm `parse_source()` đọc `idioms.c`, tìm từng hàm, và trích xuất các số chia sau `/` và `%`, các số nhân sau `*`, các trường hợp `switch` cùng giá trị của chúng, các toán tử điều kiện `? :`, các vòng lặp `for`, và các hàm được gọi. Một quán ngữ chia được coi là đã kiểm chứng (verified) khi số chia của nó nằm trong danh sách số chia của hàm. Một bảng tra cứu được xác nhận khi các mục của nó khớp với các giá trị trường hợp trong switch, tối đa tới một hệ số nhân hằng số. Một lời gọi đuôi được xác nhận khi hàm được gọi thực sự xuất hiện trong mã nguồn. Báo cáo demo kết thúc bằng danh sách `unverified`, và bộ kiểm thử đảm bảo danh sách này rỗng trên toàn bộ 18 bản dựng.

Quá trình giải mã cần tên biến. Bộ nhận diện chạy một bộ đánh giá ký hiệu (symbolic evaluator) nhỏ trên các khối cơ bản (basic blocks) của từng hàm theo thứ tự danh sách mã máy, với tên tham số lấy từ mã nguồn đặt vào các thanh ghi đối số theo quy ước System V và AAPCS64 từ bài học 38. Các vị trí ngăn xếp được theo dõi bằng chuỗi toán hạng của chúng để mã ở `-O0` đọc xuyên qua các vị trí tràn (spills). Tại mỗi điểm nhập khối, trạng thái của các khối tiền nhiệm được hợp nhất, và khi có sự khác biệt, tên biến được gán từ một quán ngữ trước đó sẽ được ưu tiên hơn một hằng số. Đó là cách mà `cmovll %edx, %eax` trong `digest` được đọc ngược lại thành `sum % 10 < 5 ? sum % 10 : 5`: quán ngữ vòng lặp đặt tên biến tích lũy là `sum`, quán ngữ chia đặt tên thương số là `sum / 10`, và quán ngữ modulo đặt tên kết quả là `sum % 10`. Mã nguồn gốc viết `rem > 5 ? 5 : rem`. Cả hai biểu thức đều cho ra giá trị nhỏ hơn giữa `rem` và 5, và chúng đồng thuận tại `rem == 5`.

## Phép chia cho một hằng số

Granlund và Montgomery trong bài báo "Division by Invariant Integers using Multiplication" (PLDI 1994) đã thay thế phép tính `n / d` bằng một phép nhân và một phép dịch bit. Chọn một độ dịch `s` và hệ số nhân `m = ceil(2^s / d)`. Khi đó `floor(n * m / 2^s)` bằng `floor(n / d)` cho mọi giá trị `n` trong phạm vi khi sai số `e = m * d - 2^s` đủ nhỏ. Đối với số nguyên không dấu 32-bit `n`, chặn sai số là `e <= 2^(s - 32)`. Đối với số có dấu `n` với `|n| < 2^31`, chặn sai số là `e <= 2^(s - 31)`, và một phép hiệu chỉnh dấu sẽ cộng thêm 1 khi `n` âm, vì phép lấy sàn (floor) làm tròn sai hướng đối với các thương số âm.

Hàm `magic_numbers(d, bits, signed)` trong `main.py` tìm kiếm giá trị `s` tăng dần từ 32 và trả về cặp giá trị đầu tiên thỏa mãn chặn sai số. Phép tìm kiếm đó tái tạo chính xác từng hằng số mà clang đã phát sinh trong các tệp mẫu.

### Tính toán thủ công phép chia số không dấu cho 10

`2^35 / 10 = 3435973836.8`. Giá trị trần là `3435973837 = 0xCCCCCCCD`. Sai số là `0xCCCCCCCD * 10 - 2^35 = 2`, và chặn tại `s = 35` là `2^3 = 8`. Tại `s = 34` sai số sẽ là 6 so với chặn 4, do đó 35 là độ dịch nhỏ nhất khả dĩ. Thử nghiệm với `n = 123`: `123 * 3435973837 = 422624781951`, và `422624781951 >> 35 = 12`.

clang phát sinh chính xác cấu trúc này cho hàm `div10u` ở `-O2` trên x86-64:

```text
       0: 89 f9                        	movl	%edi, %ecx
       2: b8 cd cc cc cc               	movl	$0xcccccccd, %eax
       7: 48 0f af c1                  	imulq	%rcx, %rax
       b: 48 c1 e8 23                  	shrq	$0x23, %rax
```

`0x23` chính là 35. Lệnh `movl %edi, %ecx` mở rộng số 0 (zero-extend) đối số 32-bit vào thanh ghi 64-bit, đặc trưng của dạng không dấu. Bộ nhận diện đọc hệ số nhân từ lệnh `movl $imm` cấp dữ liệu cho lệnh `imulq` hai toán hạng, đọc độ dịch bit, và gọi `divisor_from_magic(0xCCCCCCCD, 35)`. Hàm đó thử nghiệm `2^35 // m = 10` và 11, chấp nhận 10 vì `magic_numbers(10)` cho ra cùng hệ số nhân và độ dịch bit đó.

Trên AArch64, cùng hàm đó có dạng `mov w8, #0xcccd; movk w8, #0xcccc, lsl #16; umull x8, w0, w8; lsr x0, x8, #35`. Hằng số được nạp qua hai nửa 16-bit, phép nhân là `umull`, và độ dịch bit được ghi bằng hệ thập phân. Bộ nhận diện ghép hai nửa lại và báo cáo cùng số chia.

### Tính toán thủ công phép chia không dấu cho 7, đi kèm phép hiệu chỉnh cộng

`2^35 / 7 = 4908534052.57`. Giá trị trần là `4908534053 = 0x124924925`, cần tới 33 bit. Sai số là 3 so với chặn 8, do đó độ dịch là 35, và không có độ dịch nào nhỏ hơn thỏa mãn. Phép nhân 32-bit không thể chứa vừa hệ số nhân này, vì vậy trình biên dịch nhân với 32 bit thấp `0x24924925` rồi cộng số bị chia trở lại theo một chuỗi cố định:

```text
      32: 48 69 c0 25 49 92 24         	imulq	$0x24924925, %rax, %rax
      39: 48 c1 e8 20                  	shrq	$0x20, %rax
      3d: 29 c7                        	subl	%eax, %edi
      3f: d1 ef                        	shrl	%edi
      41: 01 f8                        	addl	%edi, %eax
      43: c1 e8 02                     	shrl	$0x2, %eax
```

Hãy đọc chuỗi này là `t = (n * 0x24924925) >> 32`, sau đó `q = (((n - t) >> 1) + t) >> 2`. Tổng độ dịch bit là `32 + 1 + 2 = 35`. Với `n = 100`: `t = 14`, `(100 - 14) >> 1 = 43`, `43 + 14 = 57`, `57 >> 2 = 14`. Bộ nhận diện phát hiện đuôi chỉ lệnh `sub`, `shr`, `add`, `shr` sau phép dịch 32 bit, cộng `2^32` trở lại hệ số nhân, và khôi phục được số 7. Hàm `evaluate_division()` chạy chuỗi chỉ lệnh chính xác này và bài kiểm thử xác minh nó khớp với `n // 7` trên các giá trị biên 32-bit. Dạng AArch64 là `sub w9, w0, w8; add w8, w8, w9, lsr #1; lsr w0, w8, #2`, ngắn hơn một lệnh vì phép cộng kèm dịch bit được gom trong một chỉ lệnh duy nhất.

### Tính toán thủ công phép chia có dấu cho 3

`2^32 / 3 = 1431655765.33`, giá trị trần là `1431655766 = 0x55555556`, sai số 2, và chặn có dấu tại `s = 32` là `2^1 = 2`. Chặn vừa khít không dư, vì vậy độ dịch là 32 và clang sử dụng `shrq $0x20`. Với `n = -7`: tích số là `-10021590362`, dịch số học 32 bit cho ra `-3`, bit dấu được bật nên phép hiệu chỉnh cộng thêm 1 và kết quả là `-2`, khớp với `-7 / 3` trong chuẩn C. Cả hai tệp mẫu có dấu `div10s` và `div3s` đều mang phép hiệu chỉnh dấu này dưới dạng `shrq $0x3f` trên một bản sao của tích số theo sau bởi lệnh `addl`.

### Bản thân hằng số không nói lên số chia

Một mình hệ số nhân không định danh được số chia. Giá trị `0x66666667` với độ dịch 34 tương ứng số chia 10. Nhưng cùng hệ số nhân đó với độ dịch 33 lại là số chia 5, như bài học 34 đã chỉ ra. Vì vậy, bộ nhận diện luôn ghép đôi hệ số nhân với tổng độ dịch, và trường `clangMatchesAlgorithm` ghi nhận cặp giá trị này có đúng là kết quả do thuật toán tìm kiếm sinh ra hay không. Lũy thừa của 2 không bao giờ dùng phép nhân. Hàm `magic_numbers()` từ chối chúng, vì trình biên dịch sẽ phát sinh phép dịch bit kèm độ lệch dấu thay thế, và hàm `mod8s` ở phần tiếp theo minh họa khuôn mẫu đó.

### Quán ngữ thuộc về bộ chọn chỉ lệnh

Ở mức `-O0`, hàm `div10u` có dạng `movl $0xa, %ecx; xorl %edx, %edx; divl %ecx`. Hàm `div10s` có dạng `cltd; idivl %ecx`. Trong cùng tệp đối tượng đó, hàm `digest` chia tổng vòng lặp cho 10 bằng lệnh nhân `imulq $0x66666667`. Một cờ biên dịch giải thích sự khác biệt này: tệp mẫu `idioms-x86_64-nofastisel-O0.o` được dựng với tùy chọn `-O0 -mllvm -fast-isel=false`, và trong đó mọi phép chia đều biến thành phép nhân, bao gồm cả `div10u` với chuỗi `movl $0xcccccccd, %ecx; imulq %rcx, %rax; shrq $0x23`. Lệnh chia phần cứng là lựa chọn của bộ chọn chỉ lệnh nhanh (FastISel) ở `-O0`. Phép nhân là kết quả hạ cấp của SelectionDAG, do các lớp `UnsignedDivisionByConstantInfo` và `SignedDivisionByConstantInfo` của LLVM sinh ra. Khối mã trong `digest` chứa phép chia cũng đồng thời chứa lệnh `switch`, và khối đó được chuyển sang xử lý bởi SelectionDAG. Bộ nhận diện báo cáo cả hai dạng: `hardware idivl` và `magic multiply`.

## Phép chia lấy dư đi theo sau thương số

Biểu thức `x % d` tương đương `x - (x / d) * d`. Trình biên dịch tính thương số bằng quán ngữ nêu trên, nhân ngược lại, rồi trừ đi. Trên x86-64, phép nhân ngược lại với 10 có dạng `addl %ecx, %ecx; leal (%rcx,%rcx,4), %ecx`, tương ứng `2 * q * 5`. Trên AArch64, nó chỉ tốn đúng một chỉ lệnh: `msub w0, w8, w9, w0`, tính toán `w0 - w8 * w9` với `w9 = 10`. Bộ nhận diện lần theo thanh ghi thương số qua các lệnh sao chép, theo dõi hệ số nhân tích lũy qua `add`, `lea`, `shl`, và `imul`, rồi chấp nhận lệnh `sub` khi hệ số bằng đúng số chia.

Khi cùng thương số đó cấp dữ liệu cho một biểu thức khác, cả hai quán ngữ đều được giữ lại trong báo cáo. Ở `-O1`, trình biên dịch tính `sum / 10` một lần trong `digest` và dùng nó cho cả chỉ mục của `switch` lẫn phép lấy dư. Trên x86-64 nó thậm chí kết hợp thương số hai lần: `leal (%rax,%rsi), %edi` cho phép chia dư và `addl %esi, %eax` cho switch. Bộ nhận diện báo cáo một phép chia và một phép chia dư, đồng thời đánh dấu thương số là dùng chung (shared). Một báo cáo làm mất phép chia ở đây sẽ khiến chỉ mục của `switch` trở nên không thể đọc hiểu được.

Phép chia dư có dấu cho lũy thừa của 2 có hình thái riêng biệt, vì thương số không bao giờ được tính ra:

```text
      92: 8d 48 07                     	leal	0x7(%rax), %ecx
      95: 85 ff                        	testl	%edi, %edi
      97: 0f 49 cf                     	cmovnsl	%edi, %ecx
      9a: 83 e1 f8                     	andl	$-0x8, %ecx
      9d: 29 c8                        	subl	%ecx, %eax
```

Đối với giá trị `x` âm, mã sẽ cộng độ lệch 7 trước khi tạo mặt nạ bit, sao cho phép cắt ngắn làm tròn về phía 0 theo đúng chuẩn C. Giá trị được tạo mặt nạ là `x` được làm tròn thành bội số của 8, và phép trừ cho ra kết quả `x % 8`. AArch64 thực hiện việc này bằng `negs w8, w0; and w9, w0, #0x7; and w8, w8, #0x7; csneg w0, w9, w8, mi`: tạo mặt nạ cho cả `x` lẫn `-x`, sau đó chọn và đảo dấu theo bit dấu. Ghidra đặt tên cho quy tắc này là `RuleSignMod2nOpt`.

Ở `-O0`, AArch64 viết phép chia dư thành `sdiv w9, w8, w10; mul w9, w9, w10; subs w0, w8, w9`. Bộ nhận diện cũng chấp nhận dạng này và báo cáo `hardware sdiv + mul + subs`.

## Năm phương thức hạ cấp câu lệnh switch

LLVM quyết định hình thái của một lệnh `switch` ở hai vị trí. Lượt phân tích `SimplifyCFG::simplifySwitchLookup` chạy từ `-O1` trở lên và chuyển một switch có các case đều tạo ra hằng số thành một bảng các giá trị, khi có ít nhất ba case và kiến trúc mục tiêu hỗ trợ tạo bảng tra cứu. Các kiểu bảng của nó gồm `SingleValueKind`, `LinearMapKind`, `BitMapKind`, và `ArrayKind`. Pipeline của `-O0` không bao giờ chạy lượt phân tích này. Các lệnh switch còn lại sẽ chuyển đến `SwitchLowering::findJumpTables` trong khâu chọn chỉ lệnh, nơi sẽ dựng bảng nhảy nếu cụm case có ít nhất `min-jump-table-entries` mục (mặc định là 4) và đủ dày đặc: `NumCases * 100 >= Range * MinDensity` với mật độ `jump-table-density` là 10 cho hàm thông thường và 40 khi dùng `-Os`. Những gì còn sót lại sẽ trở thành một cây tìm kiếm nhị phân trên các hằng số case, với hàm `splitWorkItem` chọn điểm chia (pivot). Các tệp mẫu chứa mỗi hàm tương ứng với một kết quả hạ cấp.

| Hàm | Nguồn | `-O0` x86-64 | `-O2` x86-64 | `-O2` AArch64 |
|---|---|---|---|---|
| `linear_switch` | case 0 đến 6 trả về 10 đến 70 | Bảng nhảy, 7 mục | Ánh xạ tuyến tính `10*k + 10`, mặc định -1 | `madd` và `csinv` |
| `table_switch` | case 0 đến 6 trả về 10 đến 60 rồi 71 | Bảng nhảy, 7 mục | Bảng tra cứu tại `.rodata+0x28` | Bảng tra cứu tại `.rodata+0x0` |
| `call_switch` | case 0 đến 4 gọi một trong năm con trỏ hàm | Bảng nhảy, 5 mục | Bảng nhảy, 5 mục | Cây so sánh, điểm chia 1 |
| `sparse_switch` | case 1, 100, 10000, 1000000, 100000000 | Chuỗi so sánh tuần tự | Cây so sánh, điểm chia 9999 | Cây so sánh, điểm chia 9999 |
| `two_cases` | case 3 và 4 trả về 7 và 9 | Chuỗi so sánh tuần tự | Chuỗi chọn `k != 3 ? (k == 4)*9 : 7` | Hai lệnh `csel` |

### Bảng tra cứu, đọc trực tiếp từ tệp đối tượng

Hàm `table_switch` ở mức `-O2`:

```text
      d0: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      d5: 83 ff 06                     	cmpl	$0x6, %edi
      d8: 77 0c                        	ja	0xe6 <table_switch+0x16>
      da: 89 f8                        	movl	%edi, %eax
      dc: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		00000000000000df:  R_X86_64_PC32	.rodata+0x24
      e3: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
      e6: c3                           	retq
```

Phép kiểm tra biên `cmpl $6; ja` tương ứng với 7 mục hợp lệ. Phần bù tái định vị (relocation addend) `0x24` cộng với 4 byte của trường tương đối theo RIP đặt bảng tại `.rodata+0x28`. Các byte tại đó là `0a 00 00 00 14 00 00 00 ...`, vì vậy các mục là `[10, 20, 30, 40, 50, 60, 71]`, khớp hoàn toàn với các giá trị case trong mã nguồn. Trình biên dịch hợp ngữ đặt tên cho bảng này là `.Lswitch.table.table_switch`. Trong tệp đối tượng chỉ còn lại phép tái định vị và các byte dữ liệu. Không hề có lệnh nhảy nào, vì vậy đồ thị luồng điều khiển chỉ có ba khối cơ bản.

Dạng AArch64 ghép địa chỉ bảng bằng `adrp x8` và `add x8, x8, #0x0`, mỗi lệnh mang một vị trí tái định vị tới `.rodata`, rồi nạp bằng `ldr w0, [x8, w0, uxtw #2]`.

### Ánh xạ tuyến tính, hoàn toàn không cần bảng

Hàm `linear_switch` trả về `10 * (k + 1)` cho `k` từ 0 đến 6. `simplifySwitchLookup` phát hiện mối quan hệ affine và phát sinh phép tính số học:

```text
      b0: 83 ff 07                     	cmpl	$0x7, %edi
      b3: 8d 04 bf                     	leal	(%rdi,%rdi,4), %eax
      b6: 8d 4c 00 0a                  	leal	0xa(%rax,%rax), %ecx
      ba: b8 ff ff ff ff               	movl	$0xffffffff, %eax
      bf: 0f 42 c1                     	cmovbl	%ecx, %eax
```

Bộ đánh giá đọc `leal (%rdi,%rdi,4)` là `5*k`, `leal 0xa(%rax,%rax)` là `10*k + 10`, và `cmovb` sau `cmpl $7` là `k < 7 ? 10*k + 10 : -1`. Sau đó bộ nhận diện kiểm tra công thức với từng case mã nguồn và xác nhận quán ngữ. AArch64 đạt được biểu thức tương đương bằng lệnh `madd w8, w0, w8, w9` và `csinv w0, w8, wzr, lo`.

### Bảng nhảy, đọc từ các mục tái định vị

Hàm `call_switch` ở mức `-O2` có dạng: `cmpl $4; ja`, một lệnh `leaq` trỏ tới `.rodata-0x4`, một lệnh `movslq` nạp mục 4 byte, `addq`, và `jmpq *%rax`. Năm mục là các phép tái định vị trong `.rela.rodata`, và công thức `target = addend - 4k` cho ra `0x10c, 0x107, 0x10e, 0x110, 0x109`, năm khối phân biệt mà mỗi khối đều thực hiện lệnh nhảy đuôi qua con trỏ hàm. Tệp mẫu `dispatch` mở rộng điều này lên 16 trường hợp: 16 mục tái định vị trên x86-64, và trên AArch64 là một bảng nén gồm 16 byte đơn đọc qua `ldrb w11, [x9, x8]`, nhân tỷ lệ với 4 bằng `add x10, x10, x11, lsl #2`, dựa trên `adr` của khối case đầu tiên.

### Cùng một lệnh switch ở bốn mức tối ưu hóa

Hàm `digest` chứa một switch trên `sum / 10` với các case từ 6 đến 10 có giá trị 1, 2, 4, 8, 8, sau đó nhân kết quả với 16. Trên x86-64, nó là một bảng nhảy ở `-O0`, một bảng tra cứu ở `-O1` với các mục `[16, 32, 64, 128, 128]`, và trở lại thành bảng nhảy ở `-O2` cùng `-O3` với bốn khối lưu các giá trị `0x10`, `0x20`, `0x40`, `0x80`. Từ `-O1` trở lên, phép nhân với 16 đã được gộp trực tiếp vào bên trong các case. Ở `-O0`, các case lưu 1, 2, 4, và 8, còn phép nhân nằm ở phía sau switch. Đối với bảng tra cứu ở `-O1`, bộ nhận diện so sánh các mục với giá trị mã nguồn, nhận ra tỷ lệ không đổi, và báo cáo `foldedFactor: 16`. Trên AArch64, dạng `-O2` là một cây so sánh với điểm chia tại 8, các phép kiểm tra bằng cho 6, 7, 8, và một phép kiểm tra khoảng `sub w9, w9, #9; cmp w9, #2; b.hs` bao trọn cả hai trường hợp 9 và 10 chỉ trong hai chỉ lệnh. Bộ nhận diện báo cáo `rangeTests: [[9, 10]]`.

Hai cờ làm dịch chuyển các ranh giới này. Tùy chọn `-fno-jump-tables` biến `table_switch`, `call_switch`, và switch trong `digest` thành các cây so sánh trên x86-64, với điểm chia 2 và 4 cho hàm bảy trường hợp. Trên AArch64, `-mllvm -min-jump-table-entries=4` giữ `call_switch` ở dạng cây so sánh, cho thấy trên kiến trúc này số lượng mục không phải là yếu tố quyết định duy nhất cho một switch năm case gồm các lời gọi gián tiếp.

## Vòng lặp: Vector hóa, Trải rộng, hoặc Bị thay thế

Bộ vector hóa vòng lặp của LLVM được bật mặc định ở `-O2`, `-O3`, và `-Os`, đồng thời tắt ở `-O0` và `-O1`. Hàm `sum_n` ở `-O2` thể hiện trọn vẹn khuôn mẫu:

```text
     1f6: 83 fe 07                     	cmpl	$0x7, %esi
     1f9: 77 09                        	ja	0x204 <sum_n+0x14>
     206: 81 e2 f8 ff ff 7f            	andl	$0x7ffffff8, %edx
     230: f3 0f 6f 14 37               	movdqu	(%rdi,%rsi), %xmm2
     235: 66 0f fe c2                  	paddd	%xmm2, %xmm0
     239: f3 0f 6f 54 37 10            	movdqu	0x10(%rdi,%rsi), %xmm2
     23f: 66 0f fe ca                  	paddd	%xmm2, %xmm1
     243: 48 83 c6 20                  	addq	$0x20, %rsi
     24a: 75 e4                        	jne	0x230 <sum_n+0x40>
     24c: 66 0f fe c8                  	paddd	%xmm0, %xmm1
     250: 66 0f 70 c1 ee               	pshufd	$0xee, %xmm1, %xmm0
     262: 66 0f 7e c8                  	movd	%xmm1, %eax
     270: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
     279: 75 f5                        	jne	0x270 <sum_n+0x80>
```

Khối bảo vệ `cmpl $7; ja` chuyển các số lần lặp dưới 8 sang vòng lặp vô hướng. Mặt nạ `andl $0x7ffffff8` làm tròn số lần lặp xuống bội số của 8. Thân vòng lặp nạp 32 byte mỗi lần lặp vào hai thanh ghi tích lũy `xmm`, mỗi thanh ghi gồm bốn làn 32-bit, do đó hệ số vector hóa là 4 và hệ số đan xen (interleave) là 2. Sau vòng lặp, các lệnh `paddd`, hai lệnh `pshufd`, hai lệnh `paddd` nữa, và lệnh `movd` rút gọn tám làn thành một số nguyên duy nhất. Vòng lặp tại `0x270` là phần dư vô hướng cho 0 đến 7 phần tử cuối cùng. Bộ nhận diện tìm thấy vòng lặp từ cạnh quay lui (back edge), nhận diện các thanh ghi SIMD, đọc số làn từ `paddd`, đếm các bộ tích lũy phân biệt, chia bước nhảy `0x20` cho 4, và báo cáo `4 lanes x 2 accumulators = 8 elements per iteration`. AArch64 thực hiện điều tương tự với `ldp q2, q3`, hai lệnh `add v.4s`, `addv s0, v0.4s`, và `fmov w8, s0`.

Số lần lặp cố định sẽ loại bỏ hoàn toàn vòng lặp. Hàm `sum4` chỉ gồm một lệnh `movdqu`, phép rút gọn và `movd`, không có cạnh quay lui. Hàm `poly8` không thể vector hóa vì mỗi bước lặp phụ thuộc vào giá trị tích lũy trước đó, do đó trình biên dịch trải rộng (unroll) tám lần: tám lệnh nạp tại các độ lệch từ 0 đến `0x1c` từ `%rdi`, mỗi lệnh kèm theo một phép dịch 5 bit và một phép trừ để tính `acc * 31`.

Tắt bộ vector hóa bằng `-fno-vectorize -fno-slp-vectorize`, hàm `sum_n` sẽ trở thành một vòng lặp vô hướng được trải rộng 8 lần, với tám lệnh `addl k(%rdi,%rdx,4), %eax` và `addq $8`, cộng với một vòng lặp phần dư.

### Vòng lặp biến thành lời gọi thư viện

Lượt phân tích `LoopIdiomRecognize` chuyển một vòng lặp gán 0 vào từng byte thành lời gọi `memset`, từ mức `-O1` trở lên. Hàm `fill` ở `-O2` có dạng `testl %esi, %esi; jle; movl %esi, %edx; xorl %esi, %esi; jmp memset@PLT`. Lời gọi là một lệnh nhảy đuôi mang tái định vị `R_X86_64_PLT32 memset-0x4`, và lệnh `retq` của hàm chỉ nằm trên nhánh rỗng. Bộ nhận diện báo cáo `store loop recognized as memset(...)` và xác minh nó đối chiếu với vòng lặp trong mã nguồn C.

Hàm `copy16` gọi `memcpy` với kích thước hằng số 16, và backend x86 mở rộng nó nội tuyến thành `movups (%rsi), %xmm0; movups %xmm0, (%rdi)`. Ở `-O0`, cùng lời gọi đó là hai cặp `movq` 8-byte. Dùng cờ `-fno-builtin`, lời gọi giữ nguyên là lời gọi hàm `jmp memcpy@PLT`, và `fill` giữ lại vòng lặp của nó, sau đó bộ vector hóa biến nó thành vòng lặp gồm hai lệnh lưu 16-byte `movups` mỗi lần lặp.

## Lời gọi hàm biến mất hoặc trở thành lệnh nhảy

Hàm `tail_caller` trả về `tail_target(x + 1)`. Ở `-O0`, nó có dạng `addl $0x1, %edi; callq tail_target@PLT; addq $0x10, %rsp; popq %rbp; retq`. Ở `-O2`, nó chỉ còn `incl %edi; jmp tail_target@PLT`. Khung ngăn xếp của bên gọi biến mất và bên được gọi sẽ trả thẳng về cho bên đã gọi hàm gọi ban đầu. Trên AArch64, lệnh nhảy là `b tail_target` với `R_AARCH64_JUMP26`. Bộ nhận diện coi một lệnh `jmp` hoặc `b` mang tái định vị của lời gọi là một lời gọi đuôi (tail call).

Hàm `twice_sum` gọi `leaf_add`. Ở `-O0`, lời gọi tồn tại rõ ràng: `callq leaf_add@PLT`. Từ `-O1`, thân của `leaf_add` là `leal (%rdi,%rsi), %eax` xuất hiện trực tiếp bên trong `twice_sum` và không còn lời gọi nào nữa. Bộ nhận diện đối chiếu với mã nguồn để biết `twice_sum` gọi `leaf_add`, khi không thấy lời gọi hay lệnh nhảy đuôi, nó sẽ tìm kiếm hình thái thân hàm của callee bên trong caller: khớp mã lệnh và toán hạng nguồn sau khi đã loại bỏ toán hạng đích.

Hàm `dispatch` kết hợp cả hai: một bảng nhảy 16 mục mà mỗi case đều là `movl %esi, %edi; jmp fN@PLT`. Bộ nhận diện báo cáo một switch và 16 lời gọi đuôi.

## Rẽ nhánh biến thành di chuyển dữ liệu

Hàm `clamp_max` có dạng `movl %esi, %eax; cmpl %esi, %edi; cmovgl %edi, %eax`. Bộ đánh giá đặt tên `%edi` là `a` và `%esi` là `b`, ghi lại lệnh so sánh, và đọc `cmovg` thành `a > b ? a : b`, khớp từng ký tự với mã nguồn. Hàm `abs_int` gồm `movl %edi, %eax; negl %eax; cmovsl %edi, %eax` và được đọc thành `-x < 0 ? x : -x`; AArch64 viết thành `cmp w0, #0; cneg w0, w0, mi` và đọc thành `x < 0 ? -x : x`. Hàm `is_pos` có dạng `xorl %eax, %eax; testl %edi, %edi; setg %al`, đọc thành `(x > 0)`. Đây chính là chuyển đổi điều kiện if (if-conversion) và chọn chỉ lệnh `cmov`.

Ở `-O0`, không có cấu trúc nào trong số này tồn tại. Hàm `clamp_max` là một lệnh so sánh, một lệnh `jle`, hai lệnh lưu vào cùng một vị trí ngăn xếp, và một lệnh nạp.

Hàm `mul9` là một lệnh duy nhất `leal (%rdi,%rdi,8), %eax`, và bộ nhận diện báo cáo `strength reduction` với `x * 9`. Nó chỉ làm vậy khi mã nguồn thực sự có hệ số nhân đó; một lệnh `lea` dùng để tính tỷ lệ chỉ mục bảng không phải là một phép nhân giảm cường độ.

## Hai cơ chế tăng cường bảo mật trông giống quán ngữ

Bộ bảo vệ ngăn xếp (stack protector) là tính năng của trình biên dịch, và nó để lại hình thái mà nếu không cẩn thận một agent sẽ đọc nhầm thành logic chương trình. Tệp `canary.c` biên dịch với `-fstack-protector-strong` cho x86-64 Linux sinh ra:

```text
       7: 64 48 8b 04 25 28 00 00 00   	movq	%fs:0x28, %rax
      10: 48 89 44 24 40               	movq	%rax, 0x40(%rsp)
      26: 64 48 8b 0c 25 28 00 00 00   	movq	%fs:0x28, %rcx
      2f: 48 3b 4c 24 40               	cmpq	0x40(%rsp), %rcx
      34: 75 05                        	jne	0x3b <with_canary+0x3b>
      3b: e8 00 00 00 00               	callq	0x40 <plain_leaf>
		000000000000003c:  R_X86_64_PLT32	__stack_chk_fail-0x4
```

Đoạn `%fs:0x28` là `stack_guard` trong cấu trúc `tcbhead_t` của glibc tại độ lệch byte 40. Đoạn mở đầu sao chép giá trị bảo vệ vào vị trí phía trên bộ đệm, đoạn kết thúc so sánh lại, và nhánh thất bại sẽ gọi `__stack_chk_fail`. Trên AArch64 Linux, giá trị bảo vệ lấy từ biến toàn cục `__stack_chk_guard` thông qua bảng GOT với `adrp` và `ldr`, nhánh thất bại là `bl __stack_chk_fail`. Bộ nhận diện báo cáo nguồn bảo vệ, vị trí ô nhớ, lệnh so sánh, và lời gọi thất bại. Hàm `plain_leaf` không có mảng nên không có canary ở mức bảo vệ `strong`.

Tùy chọn `-fcf-protection=full` đặt lệnh `endbr64` (các byte `f3 0f 1e fa`) làm chỉ lệnh đầu tiên của mọi hàm. Đây là điểm đáp (landing pad) cho các nhánh gián tiếp trên CPU có tính năng cưỡng chế luồng điều khiển (CET), và nó hoạt động như lệnh `nop` trên các máy khác. Bộ nhận diện ghi nhận nó là `cet landing pad`, giúp hàm không bị nhận diện nhầm là khối dữ liệu.

## Các trình dịch ngược gập các cấu trúc này như thế nào

Mọi trình dịch ngược đều chạy một số phép dẫn xuất này dưới dạng các quy tắc viết lại (rewrite rules). Decompiler của Ghidra đặt tên chúng trong `ruleaction.cc`: `RuleDivOpt` ("Chuyển đổi các dạng INT_MULT và shift thành INT_DIV hoặc INT_SDIV"), `RuleModOpt` ("Đơn giản hóa biểu thức tối ưu hóa INT_REM và INT_SREM"), `RuleDivTermAdd` xử lý phần hiệu chỉnh cộng, `RuleSignDiv2`, và `RulePositiveDiv`.

Dữ liệu đầu vào của các quy tắc đó có thể quan sát trực tiếp. Thư viện pypcode 4.0.1 nâng mã bằng đặc tả SLEIGH tương tự Ghidra, và tệp `code/fixtures/idioms-x86_64-O2.lifts.json` chứa P-code thô cho hàm `div10s`: 112 mục cho bảy chỉ lệnh. Quy tắc `RuleDivOpt` khớp phép nhân và hai phép dịch bit trong luồng này, và việc loại bỏ mã chết sẽ xóa các cờ mà không ai đọc tới.

angr 10.0.3 chạy trên tệp đối tượng `-O2` của chúng ta, và kết quả dịch ngược 20 hàm nằm trong `idioms-x86_64-O2.angr.json`. Nó gập thành công dạng không dấu: `div10u` thành `return a0 / 10;`. Tuy nhiên nó không gập dạng có dấu: `div10s` trở thành `v1 = a0 * 1717986919; return ((long long)(v1) >> 0x22) + (v1 >> 63);` và `mod10s` thành `a0 - (...) * 2 * 5`. Với `div7u` nó in biểu thức số học phức tạp do các thành phần hiệu chỉnh cộng sót lại. Nó gập `mul9` thành `a0 * 9`, `linear_switch` thành `(a0 < 7 ? a0 * 10 + 10 : 0xffffffff)`, `clamp_max` thành `((int)a1 < (int)a0 ? a0 : a1)`, `is_pos` thành `a0 > 0`, và `fill` thành `return memset(a0, 0, a1);`. Nó in `table_switch` dưới dạng nạp ô nhớ `*(int *)(4 * a0 + 4195400)` mà không kèm giá trị, và `sum_n` đi kèm các hàm nội tại `_mm_add_epi32`.

radare2 6.2.4 và rizin 0.9.1 khi chạy trên cùng đối tượng: lệnh `pdc` của radare2 in `div10s` thành mã giả mức thanh ghi `rax = rax * 0x66666667; rcx >>>= 0x3f; rax >>= 0x22; eax += ecx` mà không hề gập lại, và `table_switch` thành `eax = dword [rcx + rax*4]`. rizin in RzIL của `imul` dưới dạng tích số 128-bit. lldb và objdump in cùng bảy chỉ lệnh vì cùng dùng chung bộ phân tích của LLVM. Capstone 5.0.7 chỉ là bộ giải mã chỉ lệnh, và một quán ngữ không bao giờ xuất hiện trong đầu ra của bộ giải mã.

Compiler Explorer là công cụ trực quan theo chiều xuôi: dán `idioms.c`, chọn trình biên dịch và cờ tương ứng để quan sát trực tiếp các ngưỡng chuyển đổi và vector hóa.

## Xây dựng hệ thống

Chạy chương trình minh họa:

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
```

Đầu ra là một báo cáo JSON hoàn chỉnh. Đối với từng bản dựng trong số 18 bản dựng, báo cáo liệt kê mọi hàm và các quán ngữ của nó, kèm theo loại (kind), dạng mã nguồn, địa chỉ bằng chứng, kết quả kiểm chứng, và các trường chi tiết. Báo cáo kết thúc bằng ma trận phân loại theo bản dựng, phép so sánh `-O2` với `-O3` cho cả hai kiến trúc, và danh sách `unverified`.

Cấu trúc định dạng kỳ vọng:

```json
{
  "builds": {
    "idioms-x86_64-O2": {
      "arch": "x86_64", "level": "O2", "flags": "x86_64",
      "functions": {
        "div7u": [{"kind": "division", "source": "x / 7", "evidence": ["0x30", "0x32", "0x39", "0x3d", "0x3f", "0x41", "0x43"], "verified": true, "form": "magic multiply with add fix-up", "magic": "0x24924925", "effectiveMagic": "0x124924925", "shift": 35, "...": "..."}],
        "table_switch": [{"kind": "switch", "source": "switch (k) as lookup table", "table": ".rodata+0x28", "entries": 7, "values": [10, 20, 30, 40, 50, 60, 71], "...": "..."}],
        "sum_n": [{"kind": "vector loop", "lanes": 4, "accumulators": 2, "elementsPerIteration": 8, "...": "..."}]
      }
    }
  },
  "kindMatrix": {"idioms-x86_64-O2": {"division": 5, "modulo": 3, "switch": 6, "vector loop": 3, "...": "..."}},
  "sameCodeAtO2AndO3": {"x86_64": {"identical": true}, "arm64": {"identical": false, "differences": ["..."]}},
  "unverified": []
}
```

Hãy đọc mã nguồn trong `code/main.py` theo trình tự sau:

1. `parse_listing()` chuyển văn bản từ `llvm-objdump -d -r` thành các hàm, chỉ lệnh và tái định vị đi kèm cho cả hai kiến trúc.
2. `ElfFile` đọc các phân đoạn, ký hiệu, và các mục `.rela.*`, còn `words()` đọc các mục trong bảng.
3. `magic_numbers()`, `divisor_from_magic()`, và `evaluate_division()` triển khai phép dẫn xuất Granlund-Montgomery theo chiều xuôi, chiều ngược, và dưới dạng chuỗi chỉ lệnh thực thi.
4. `parse_source()` trích xuất các cấu trúc phục vụ xác minh từ tệp C.
5. `Evaluator` thực hiện lượt đánh giá ký hiệu theo từng khối cơ bản để gán tên cho các thanh ghi và ô nhớ ngăn xếp.
6. `find_divisions()` và `find_modulos()` nhận diện các dạng nhân số ma thuật, dịch bit, hiệu chỉnh, nhân ngược, và lũy thừa của 2.
7. `find_switches()` phân loại các bảng nhảy, bảng tra cứu, ánh xạ tuyến tính, chuỗi select, cây so sánh, và chuỗi so sánh tuần tự.
8. `find_vector_loops()`, `find_unrolled_loops()`, `find_tail_calls()`, `find_inlined_calls()`, `find_selects()`, `find_memory_builtins()`, `find_stack_protector()`, `find_cet()`, và `find_strength_reduction()` xử lý các trường hợp còn lại.
9. `analyze_function()` chạy các bộ tìm kiếm theo thứ tự, nạp tên kết quả của từng quán ngữ vào lượt đánh giá tiếp theo.

Có thêm ba lệnh thực thi hỗ trợ: `python3 code/main.py scan FILE.objdump.txt [FUNCTION]` quét một bản dịch ngược cụ thể. `python3 code/main.py magic DIVISOR [signed]` in số nhân, độ dịch, sai số và kiểm chứng trên giá trị biên. `python3 code/main.py figure-data` in dữ liệu nhúng vào hình minh họa.

## Sử dụng hệ thống

Sử dụng bộ nhận diện theo thứ tự sau trên một tệp đối tượng mới mà bạn biên dịch:

1. Thu xuất bản dịch ngược bằng `llvm-objdump -d -r` từ tệp đối tượng và lưu lại lệnh biên dịch chính xác bên cạnh. Bộ nhận diện tự đọc kiến trúc từ tiêu đề của bản dịch ngược.
2. Chạy lệnh `scan` trên danh sách mã và xem xét danh sách `unverified` đầu tiên. Mỗi mục chưa kiểm chứng hoặc là một cấu trúc mà bộ phân tích mã nguồn bỏ sót, hoặc là một khuôn mẫu mà bộ nhận diện đọc sai.
3. Với mỗi phép chia, đọc trường `algorithm`. Khi `clangMatchesAlgorithm` là true, trình biên dịch đã chọn độ dịch hợp lệ nhỏ nhất. Khi nó là false, trình biên dịch có lý do riêng; hãy ghi lại các hằng số đó.
4. Với mỗi lệnh switch, so sánh phương thức `lowering` qua các mức tối ưu hóa bạn đã dựng để nhận biết các ngưỡng chuyển đổi.
5. Với mỗi vòng lặp, đọc `elementsPerIteration` và `scalarRemainderLoop`, rồi xác nhận hai cạnh quay lui mà bài học 33 báo cáo thực chất chỉ là một vòng lặp mã nguồn duy nhất.
6. Chạy một trình dịch ngược trên cùng tệp đối tượng, trích dẫn văn bản của nó bên cạnh mỗi quán ngữ, và ghi nhận những cấu trúc nào nó đã gập được.

## Phòng thí nghiệm tương tác

### Lab A: Dẫn xuất số ma thuật rồi tìm kiếm trong mã máy

Từ thư mục `code`, khởi động môi trường Python:

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization/code
python3 -q
```

Thực thi:

```python
from main import *
magic_numbers(10), magic_numbers(10, signed=True)
magic_numbers(7), magic_numbers(7, signed=True)
divisor_from_magic(0x66666667, 34, signed=True), divisor_from_magic(0x66666667, 33, signed=True)
```

Tính toán `ceil(2^35 / 10)` trên giấy trước khi gọi hàm. Sau đó thêm một hàm `unsigned div13(unsigned x) { return x / 13; }` vào bản sao của `idioms.c`, biên dịch ở `-O2`, và đối chiếu hằng số trong danh sách mã máy với `magic_numbers(13)`.

### Lab B: Bảng tra cứu và Cây so sánh

```python
listing = load_listing("idioms-x86_64-O2.objdump.txt")
elf = load_elf("idioms-x86_64-O2.o")
sources = load_source("idioms.c")
[i.to_dict() for i in analyze_function(listing, "table_switch", elf, sources)]
[i.to_dict() for i in analyze_function(listing, "sparse_switch", elf, sources)]
elf.words(".rodata", 0x28, 7)
```

Đổi giá trị `71` trở lại thành `70` trong bản sao của `table_switch`, biên dịch lại ở `-O2`, và quan sát bảng tra cứu biến thành một ánh xạ tuyến tính. Sau đó đổi một giá trị case thành 1000 và xem nó vẫn giữ nguyên là bảng `ArrayKind`.

### Lab C: Vòng lặp qua ba thiết lập biên dịch

```python
for name in ("idioms-x86_64-O1", "idioms-x86_64-O2", "idioms-x86_64-novec-O2"):
    print(name, [i.source for i in analyze_function(load_listing(name + ".objdump.txt"), "sum_n", load_elf(name + ".o"), sources)])
```

Vẽ đồ thị luồng điều khiển của `sum_n` ở `-O2` bằng tay, đánh dấu khối bảo vệ, mặt nạ bit, thân vector, phép rút gọn, và vòng lặp phần dư, rồi so sánh với các trường trong `detail`.

### Lab D: Thương số dùng chung

```python
for name in ("idioms-x86_64-O1", "idioms-arm64-O1"):
    print([(i.kind, i.source, i.detail.get("sharedQuotient")) for i in analyze_function(load_listing(name + ".objdump.txt"), "digest", load_elf(name + ".o"), sources)])
```

Đọc danh sách mã `-O1` của `digest` và tìm hai chỉ lệnh kết hợp thương số với phép hiệu chỉnh dấu trên x86-64. Sau đó giải thích vì sao bản dựng AArch64 chỉ cần một lệnh duy nhất.

## Bài tập thực hành

Bổ sung các mẹo thao tác bit (bit tricks) vào bộ mẫu và mở rộng bộ nhận diện.

Yêu cầu thực hiện:

- Viết các hàm `popcount`, `ctz`, và `rotl` bằng các hàm nội tại của GCC và phép quay vòng dịch-hoặc (shift-or rotate), biên dịch ở `-O2` có và không có `-mpopcnt -mbmi`, lưu lại các tệp đối tượng cùng danh sách mã máy.
- Nhận diện chuỗi đếm số bit 1 gồm `0x55555555`, `0x33333333`, `0x0f0f0f0f`, `0x01010101` do `TargetLowering::expandCTPOP` phát sinh, và báo cáo nó là `popcount` với bốn hằng số làm bằng chứng.
- Nhận diện mã hóa `rep bsfl` thành `tzcnt` hoặc `bsf`, và `roll %cl` thành thao tác quay bit, giải mã cả hai về hàm nội tại tương ứng.
- Thêm các loại mới vào quy trình xác thực của `parse_source()`, sao cho quán ngữ `popcount` chỉ được xác nhận khi mã nguồn thực sự gọi `__builtin_popcount`.
- Viết bài kiểm thử cho từng quán ngữ để đọc các hằng số từ danh sách mã máy.

Thành công đạt được khi `python3 code/main.py` báo cáo ba loại mới với `verified: true` ở cả hai bản dựng, và bộ kiểm thử vẫn báo cáo danh sách `unverified` hoàn toàn rỗng.

## Thành phẩm bài học

Bài học này cung cấp tệp `outputs/skill-compiler-idioms-and-optimization.md`. Sử dụng nó để biến một danh sách mã máy, một tệp đối tượng và mã nguồn thành một báo cáo quán ngữ đã kiểm chứng kèm so sánh với decompiler. Kỹ năng này đòi hỏi địa chỉ bằng chứng cho mọi nhận định, phép dẫn xuất số ma thuật cho mọi phép chia, loại hạ cấp và nguồn dữ liệu bảng cho mọi lệnh switch, cùng một dòng trích dẫn từ decompiler cho mỗi quán ngữ.

## Kiểm chứng thực tế

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/39-compiler-idioms-and-optimization
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình kiểm chứng cần chứng minh:

- Bộ phân tích danh sách mã máy đọc đủ 25 hàm từ tệp đối tượng `-O2`, giữ nguyên các byte, và gán đúng `R_X86_64_PC32 .rodata+0x24` vào lệnh `leaq` trong `table_switch`
- Bộ đọc ELF trả về `[10, 20, 30, 40, 50, 60, 71]` từ `.rodata+0x28` và mười mục tái định vị trong `.rela.rodata`
- `magic_numbers` trả về `0xcccccccd` với độ dịch 35, `0x66666667` với 34, `0x124924925` với 35 kèm dạng cộng hiệu chỉnh, `0x55555556` với 32, và `0xaaaaaaab` với 33
- Đẳng thức số học thỏa mãn trên các giá trị biên 32-bit cho mọi số chia từ 2 đến 64 không phải là lũy thừa của 2, cả có dấu và không dấu
- `divisor_from_magic` khôi phục chính xác 10, 5, 7, và 3 từ các cặp tương ứng và từ chối cặp sai
- Chuỗi hiệu chỉnh cộng tính toán chính xác giá trị `n // 7`
- Bốn phép chia x86-64 báo cáo đúng hệ số nhân, độ dịch, số chia, và hình thái; bốn phép chia AArch64 giải mã đúng `mov` và `movk`
- Bản dựng `-O0` dùng `divl` và `idivl`, `-O0 -fast-isel=false` dùng phép nhân, và `digest` ở `-O0` kết hợp cả hai
- Phép chia dư được nhận diện chính xác ở các dạng nhân ngược, `msub`, lũy thừa của 2, `sdiv` cộng `mul` cộng `subs`, và `idivl` phần cứng
- Thương số dùng chung giữ lại cả phép chia và phép chia dư ở `-O1`
- Năm phương thức hạ cấp switch được phân loại chuẩn xác ở `-O2` trên cả hai kiến trúc, kèm công thức, giá trị, đích đến, điểm chia, và biểu thức select
- Switch ở `-O0` là bảng nhảy trên x86-64 và chuỗi so sánh trên AArch64
- Switch trong `digest` chuyển từ bảng nhảy sang bảng tra cứu với hệ số gập 16, rồi trở lại bảng nhảy với bốn đích phân biệt trên x86-64, và là cây so sánh với phép kiểm tra khoảng 9 đến 10 trên AArch64
- Tùy chọn `-fno-jump-tables` tạo ra các cây so sánh, và hàm `dispatch` 16 trường hợp đọc đích đến từ tái định vị trên x86-64 và từ bảng byte nén trên AArch64
- Vòng lặp vector báo cáo 4 làn, 2 bộ tích lũy, 8 phần tử mỗi lần lặp, mặt nạ bit, khối bảo vệ, phép rút gọn, và vòng lặp phần dư
- Vòng lặp có số lần lặp cố định được vector hóa hoặc trải rộng hoàn toàn, và `-fno-vectorize` trải rộng 8 lần
- Bản dựng `-O1` không có vector hóa, `-O2` và `-O3` giống hệt nhau trên x86-64, và trên AArch64 chúng chỉ khác nhau ở `call_switch` (nơi `-O3` có 17 chỉ lệnh và `-O2` có 18)
- Tùy chọn `-fno-builtin` biến vòng lặp gán thành vòng lặp lưu dữ liệu vector hóa và giữ `memcpy` dưới dạng lời gọi hàm
- Lời gọi đuôi, thay thế `memset`, phép sao chép 16 byte nội tuyến, và phép sao chép hai lệnh chuyển ở `-O0` đều được nhận diện
- Hàm nhúng nội tuyến được tìm thấy từ `-O1` dựa trên hình thái thân hàm
- Các lệnh select giải mã thành `a > b ? a : b`, `-x < 0 ? x : -x`, `(x > 0)`, và `sum % 10 < 5 ? sum % 10 : 5`
- Cơ chế bảo vệ ngăn xếp báo cáo đúng `%fs:0x28`, vị trí ô nhớ và `__stack_chk_fail`; lệnh `endbr64` được ghi nhận trên mọi hàm của bản dựng CET
- Bộ phân tích mã nguồn đọc đúng các tham số con trỏ hàm, các case của switch, các lời gọi và tên biến tích lũy
- Toàn bộ các quán ngữ trong mọi bản dựng đều được kiểm chứng và ma trận phân loại khớp hoàn toàn
- Dữ liệu hình ảnh tương tác nhúng đúng kết quả phân tích

## Các kịch bản lỗi trong môi trường thực tế

| Tình huống lỗi | Báo cáo từ công cụ yếu | Điều phân tích chuẩn phải chứng minh |
|---|---|---|
| Phép nhân với một hằng số lẻ lớn | "Nhân với 1717986919" | Hằng số là `ceil(2^s / d)` cho một số chia `d`, sai số thỏa mãn chặn, và độ dịch là nhỏ nhất |
| Hiệu chỉnh cộng sau phép dịch 32 bit | Năm chỉ lệnh số học | Hệ số nhân rộng 33 bit và phần đuôi là dạng cộng của cùng thuật toán |
| Cùng hệ số nhân nhưng khác độ dịch bit | Gán cùng một số chia | Độ dịch quyết định số chia: độ dịch 34 là 10, độ dịch 33 là 5 |
| Chia dư cho lũy thừa 2 có dấu | "Cộng 7, tạo mặt nạ, trừ" | Độ lệch làm tròn về phía 0 và kết quả là `x % 8` |
| Lệnh nạp có chỉ mục từ `.rodata` | "Một bảng nhảy" | Một bảng tra cứu các giá trị, với các phần tử đọc từ byte tại addend + 4 |
| Phép tính affine đi kèm lệnh `cmov` | "Tính 10k + 10" | Một lệnh switch được hạ cấp thành ánh xạ tuyến tính, đối chiếu khớp mọi case |
| Chuỗi so sánh có phân chia thứ tự | "Một chuỗi các lệnh if" | Một cây tìm kiếm nhị phân trên các hằng số case với các điểm chia xác định |
| Nhánh nhảy lùi về một khối mặc định | "Một vòng lặp" | Đích đến thoát khỏi khu vực bằng lệnh nhảy không điều kiện; đó là phần đuôi dùng chung |
| Hai cạnh quay lui trong một hàm | "Hai vòng lặp" | Thực chất là một vòng lặp mã nguồn: thân vector và phần dư vô hướng |
| Lệnh `jmp memset@PLT` không có `ret` | "Hàm không có lệnh trả về" | Lời gọi đuôi trả về qua callee, thay thế cho một vòng lặp gán bộ nhớ |
| Callee vắng mặt trong đồ thị lời gọi | "Không bao giờ được gọi" | Đã nhúng nội tuyến: hình thái thân của callee xuất hiện bên trong caller |
| Lệnh nạp và so sánh `%fs:0x28` | "Đọc dữ liệu cục bộ luồng (TLS)" | Giá trị canary bảo vệ ngăn xếp, với nhánh lỗi trỏ tới `__stack_chk_fail` |
| Bốn byte lạ ở đầu hàm | "Dữ liệu đặt trước hàm" | Lệnh `endbr64`, điểm đáp CET của kiến trúc |
| Decompiler hiển thị phép toán thô | "Công cụ bị sai" | Công cụ đã không áp dụng quy tắc gập; phép dẫn xuất cung cấp dạng gốc và ghi nhận khoảng cách |

## Quy tắc vận hành

Hãy so khớp hình thái, thực hiện phép dẫn xuất toán học theo chiều ngược lại, và ghi lại các hằng số chứng minh điều đó. Báo cáo mọi quán ngữ đi kèm địa chỉ bằng chứng và dạng mã nguồn đã giải mã của nó, kiểm chứng từng mục đối chiếu với mã nguồn bạn đã biên dịch, và trích dẫn những gì từng trình dịch ngược đã hiển thị cho cùng các byte đó.

## Tài liệu đọc thêm

- [Granlund, Montgomery, "Division by Invariant Integers using Multiplication" (PLDI 1994)](https://gmplib.org/~tege/divcnst-pldi94.pdf)
- [Warren, Hacker's Delight, 2nd edition, Chapter 10: Integer Division By Constants](https://www.informit.com/store/hackers-delight-9780321842688)
- [Lemire, Kaser, Kurz, "Faster Remainder by Direct Computation" (2019)](https://arxiv.org/abs/1902.01961)
- [libdivide, header-only division by runtime constants](https://github.com/ridiculousfish/libdivide)
- [LLVM `DivisionByConstantInfo.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/DivisionByConstantInfo.cpp)
- [LLVM `SwitchLoweringUtils.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SwitchLoweringUtils.cpp) and [`TargetLoweringBase.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/TargetLoweringBase.cpp)
- [LLVM `SimplifyCFG.cpp`, `simplifySwitchLookup`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Utils/SimplifyCFG.cpp)
- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html) and [LLVM 3.4 release notes](https://releases.llvm.org/3.4/docs/ReleaseNotes.html)
- [LLVM `LoopIdiomRecognize.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/Scalar/LoopIdiomRecognize.cpp)
- [LLVM `X86CmovConversion.cpp`](https://github.com/llvm/llvm-project/blob/release/20.x/llvm/lib/Target/X86/X86CmovConversion.cpp)
- [LLVM `StackProtector.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/StackProtector.cpp) and [`X86ISelLoweringCall.cpp`, `getIRStackGuard`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86ISelLoweringCall.cpp)
- [glibc `sysdeps/x86_64/nptl/tls.h`, `tcbhead_t`](https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/x86_64/nptl/tls.h)
- [GCC `tree-switch-conversion.cc`](https://github.com/gcc-mirror/gcc/blob/master/gcc/tree-switch-conversion.cc), [GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), and [GCC 12 release notes](https://gcc.gnu.org/gcc-12/changes.html)
- [Ghidra decompiler `ruleaction.cc`](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/decompile/cpp/ruleaction.cc)
- [angr decompiler documentation](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [radare2 book: decompilers](https://book.rada.re/arch/decompile.html)
- [Compiler Explorer API](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/API.md)
- [Godbolt, "Optimizations in C++ Compilers: A practical journey", ACM Queue 17(5), 2019](https://queue.acm.org/detail.cfm?id=3372264)
- [Microsoft, `/GS` (Buffer Security Check)](https://learn.microsoft.com/en-us/cpp/build/reference/gs-buffer-security-check)
