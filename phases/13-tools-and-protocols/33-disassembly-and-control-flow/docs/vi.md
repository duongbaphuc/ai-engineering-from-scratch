# Dịch Ngược Mã Máy và Luồng Điều Khiển: Từ Byte Nhị Phân Đến Đồ Thị Có Thể Kiểm Chứng

> Một danh sách dịch ngược mã máy thực chất là một khẳng định về việc những byte nào là lệnh thực thi, và bài học này sẽ xây dựng bộ giải mã cùng đồ thị để kiểm chứng khẳng định đó.

**Type:** Build
**Languages:** Python, C, mã hợp ngữ x86-64 (assembly)
**Prerequisites:** Phase 13 · 32 (ELF from scratch)
**Time:** ~150 phút

## Mục tiêu Học tập

- Giải mã tập con lệnh x86-64 trong các fixture của bài toán ledger từ các quy tắc trong Intel SDM Tập 2, Chương 2.
- Kiểm chứng bộ giải mã đối chiếu với `llvm-objdump -d` trên từng lệnh của mọi bản dựng fixture.
- Chạy phương pháp quét tuyến tính (linear sweep) và duyệt đệ quy (recursive descent) trên cùng các byte nhị phân và giải thích chính xác điểm sai của từng phương pháp.
- Cắt các chuỗi lệnh thành các khối cơ bản (basic blocks) và xây dựng đồ thị luồng điều khiển (control-flow graph - CFG) với các cạnh được định kiểu.
- Tính toán các nút thống trị trực tiếp (immediate dominators) bằng thuật toán lặp của Cooper, Harvey và Kennedy, đồng thời tìm kiếm các vòng lặp từ các cạnh ngược (back edges).
- Khôi phục các đích đến của một bước nhảy gián tiếp (indirect jump) từ các mục tái định vị trỏ tới bảng của nó.

## Vấn đề Cần Giải Quyết

Bạn đã biên dịch `ledger.c` ở mức tối ưu hóa `-O0` và hỏi một AI agent xem hàm `grade` thực hiện công việc gì. Agent gọi một công cụ dịch ngược (disassembler), đọc danh sách lệnh thu được và trả lời rằng hàm này kết thúc bằng lệnh `jmpq *%rax` mà không có đích đến xác định nào. Sau đó, nó tạo ra mã giả chỉ với một dòng duy nhất: `return 0`.

Khi bạn tự mở cùng tệp đối tượng đó bằng `llvm-objdump -d`, bạn thấy có tới 4 khối lệnh nằm ngay sau bước nhảy đó, thực hiện lưu trữ các giá trị 4, 3, 2 và 1. Bản danh sách dịch ngược thực ra đã chứa đầy đủ các khối này từ đầu. Tuy nhiên, không có thông tin nào trong danh sách chỉ ra cách thức luồng thực thi tiếp cận được chúng, vì vậy công cụ đã coi chúng là mã chết (dead code).

Một danh sách mã máy chỉ trả lời được một câu hỏi duy nhất: những byte nào tạo nên lệnh nào. Nó không thể cho biết những lệnh nào sẽ thực sự chạy, theo thứ tự nào, hay bắt nguồn từ đâu. Để trả lời các câu hỏi đó, bạn cần thêm ba cấu trúc nền tảng: các byte mà luồng điều khiển có thể vươn tới (reachable bytes), các khối cơ bản (basic blocks), và các cạnh liên kết giữa chúng.

Bộ công cụ rea 6.5.0 không có bộ dịch ngược mã máy tích hợp sẵn. Để phân tích sâu, tệp README của nó yêu cầu bạn phải kết nối với Ghidra, Hopper hoặc IDA. Đối với việc kiểm tra ngoại tuyến, rea có thể gọi GNU objdump hoặc Rizin được cài đặt trên hệ thống. Đồ thị lời gọi của nó sẽ dừng lại trước các đích chưa được giải quyết: "Callers/callees contain only resolved functions.... Targetless computed flow remains unknown." (Các hàm gọi/bị gọi chỉ chứa các hàm đã được giải quyết.... Luồng tính toán không có đích vẫn ở trạng thái chưa xác định). Bài học này sẽ tự tay xây dựng thành phần sản sinh ra các dữ kiện đó, giúp bạn có đủ năng lực đánh giá khi một công cụ báo cáo chúng.

````figure re-33-sweep-vs-descent
````

## Định Dạng Lệnh: Mỗi Trường Có Một Quy Tắc Riêng

Tài liệu Intel SDM Tập 2, Hình 2-1 trình bày thứ tự byte của mọi lệnh máy. Đầu tiên là các tiền tố tùy chọn (optional prefixes), tiếp theo là 1 đến 3 byte mã thao tác (opcode), sau đó là một byte ModRM tùy chọn và một byte SIB tùy chọn. Tiếp theo là trường độ dời (displacement) rộng 0, 1, 2 hoặc 4 byte, và cuối cùng là giá trị tức thời (immediate) rộng 0, 1, 2 hoặc 4 byte. Toàn bộ một lệnh có độ dài tối đa là 15 byte (SDM 2.3.11). Tài liệu AMD APM Tập 3, Chương 1, Mục 1.1 đến 1.7 cũng mô tả cùng một định dạng này với dung lượng ngắn gọn hơn và là nguồn đối chiếu chéo rất tốt.

Mỗi trường dữ liệu sẽ quyết định ý nghĩa của trường tiếp theo. Do đó, một bộ giải mã thực chất là một chuỗi các quyết định nhỏ, và mỗi quyết định đều có một câu quy chuẩn trong tài liệu hướng dẫn bảo chứng phía sau.

### Tiền tố được chia thành 4 nhóm

SDM 2.1.1 liệt kê các tiền tố kế thừa (legacy prefixes). Nhóm 1 chứa `F0` (khóa bus - lock) và các tiền tố lặp `F2`, `F3` (repeat prefixes). Nhóm 2 chứa 6 tiền tố ghi đè phân đoạn (`2E` là phân đoạn mã `%cs`) và các gợi ý rẽ nhánh (branch hints). Nhóm 3 là `66`, tiền tố ghi đè kích thước toán hạng (operand-size override). Nhóm 4 là `67`, tiền tố ghi đè kích thước địa chỉ (address-size override).

Tài liệu quy định rằng mỗi nhóm chỉ được đóng góp một tiền tố duy nhất. Tuy nhiên, các trình biên dịch thực tế lại phát ra nhiều hơn. Vùng đệm căn chỉnh trong `ledger-x86_64-O0.o` tại độ lệch `0xf2` chứa chuỗi byte `66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00`, tức là 5 byte `66` liên tiếp. `llvm-objdump` in chuỗi đó ra thành `nopw %cs:(%rax,%rax)` và bộ giải mã bạn xây dựng cũng sẽ làm chính xác như vậy. Bộ giải mã thu thập mọi byte tiền tố nhìn thấy và sẽ phân loại xem tiền tố nào thực sự có hiệu lực sau.

Byte `66` còn có một nhiệm vụ thứ hai. Đối với các lệnh SSE, nó là một tiền tố bắt buộc (mandatory prefix) dùng để chọn mã thao tác opcode, vì vậy `66 0f ef c0` là lệnh `pxor %xmm0, %xmm0` chứ không phải là một phép toán 16-bit nào cả. Tương tự, `F3 0f 6f` là `movdqu`. Bộ giải mã trước hết sẽ tra cứu cặp tiền tố bắt buộc và byte opcode thứ hai. Chỉ khi cặp này không tồn tại trong bảng ánh xạ thì byte `66` mới được hiểu là tiền tố chọn toán hạng 16-bit.

### Tiền tố REX bổ sung 4 bit và phải đứng ngay sát opcode

SDM 2.2.1 định nghĩa REX là một byte nằm trong khoảng từ `40` đến `4F` có dạng nhị phân `0100WRXB`. Bit `W` chọn toán hạng 64-bit. Bit `R` mở rộng trường reg của ModRM. Bit `X` mở rộng trường index của SIB. Bit `B` mở rộng trường r/m của ModRM, trường base của SIB, hoặc thanh ghi nằm trong chính byte opcode (Bảng 2-4). REX bắt buộc phải nằm ngay trước opcode hoặc các byte thoát (escape bytes); nếu một byte REX bị theo sau bởi một tiền tố kế thừa thì nó sẽ bị bỏ qua.

Trong chuỗi lệnh `48 89 e5`, byte `48` là REX với bit `W` được bật. Opcode `89` là `MOV r/m, r`, và ModRM `e5` chọn chuyển thanh ghi `%rsp` vào `%rbp`. Nếu bỏ byte REX đi, cùng các byte đó sẽ trở thành `movl %esp, %ebp`.

REX cũng làm thay đổi tên của các thanh ghi byte (8-bit). Khi có bất kỳ byte REX nào xuất hiện, các mã hóa từ 4 đến 7 sẽ chọn `%spl`, `%bpl`, `%sil`, và `%dil` thay vì `%ah`, `%ch`, `%dh`, và `%bh`. Bộ giải mã duy trì hai bảng ánh xạ thanh ghi riêng biệt chính vì lý do này.

### Mã thao tác Opcode nằm trong hai bảng ánh xạ

Trường hợp phổ biến là opcode dài 1 byte. Byte `0F` đóng vai trò là byte thoát (escape) chuyển sang bảng ánh xạ 2-byte, và các chuỗi `0F 38` hoặc `0F 3A` chuyển sang các bảng 3-byte mà các fixture của chúng ta không sử dụng tới (SDM 2.1.2). Một số opcode sử dụng chính trường reg của ModRM làm 3 bit opcode bổ sung, được ký hiệu là `/digit` trong tài liệu. `83 /7` là lệnh `cmp` với giá trị tức thời 8-bit, `83 /5` là lệnh `sub`, và `FF /4` là lệnh nhảy gián tiếp `jmp`.

Bộ giải mã trong `code/main.py` bao quát toàn bộ các opcode xuất hiện trong 4 tệp đối tượng fixture cùng một số lệnh lân cận. Danh sách này bao gồm 8 dạng toán học số học, `mov`, `lea`, `movslq`, `movzbl`, `imul`, các lệnh dịch bit, `test`, và `idiv`. Nó tiếp tục với `push`, `pop`, `call`, `ret`, 16 lệnh nhảy có điều kiện, lệnh `nop` đa byte, và 5 opcode SSE2. Bất kỳ opcode nào khác sẽ kích hoạt ngoại lệ `DecodeError` kèm theo địa chỉ và nguyên nhân cụ thể. Một bộ giải mã phỏng đoán bừa bãi nguy hiểm hơn nhiều so với một bộ giải mã biết dừng lại đúng lúc.

### ModRM và SIB lựa chọn các toán hạng

SDM 2.1.3 phân chia byte ModRM thành `mod` (bit 7:6), `reg` (bit 5:3), và `r/m` (bit 2:0). Khi `mod = 11`, r/m đóng vai trò là một thanh ghi. Các giá trị `mod = 00`, `01`, và `10` biểu thị truy cập bộ nhớ tương ứng với không có độ dời, độ dời 8-bit, hoặc độ dời 32-bit. Khi `r/m = 100`, một byte SIB bắt buộc phải đi theo sau. Byte SIB được chia thành `scale` (bit 7:6), `index` (bit 5:3), và `base` (bit 2:0); giá trị index bằng `100` nghĩa là không sử dụng thanh ghi chỉ số.

Hai hàng đặc biệt trong Bảng 2-2 có ý nghĩa sống còn trong chế độ 64-bit: Khi `mod = 00` đi cùng `r/m = 101`, nó không biểu thị `(%rbp)` mà là một độ dời 32-bit tương đối so với lệnh tiếp theo (RIP-relative addressing, SDM 2.2.1.6). Khi `mod = 00` đi cùng trường base của SIB là `101`, nó biểu thị một độ dời 32-bit tuyệt đối không có thanh ghi cơ sở.

Hãy xem xét chuỗi `8d 44 08 fa` từ hàm `grade` ở mức `-O0`. Opcode `8D` là lệnh `lea`. ModRM `44` có dạng nhị phân `01 000 100`: tức độ dời 8-bit, trường reg là `%eax`, và theo sau bởi byte SIB. SIB `08` có dạng `00 001 000`: hệ số nhân scale là 1, index là `%rcx`, base là `%rax`. Byte độ dời `fa` mang giá trị -6. Cú pháp AT&T tương ứng là `leal -0x6(%rax,%rcx), %eax`, thực hiện tính toán `score / 10 - 6`, chính là chỉ số cho câu lệnh switch.

### Độ dời là nơi trình biên tập liên kết ghi đè

Chuỗi `48 8d 0d 00 00 00 00` là lệnh `leaq (%rip), %rcx`. Bốn byte 0 chính là trường độ dời của toán hạng tương đối theo con trỏ lệnh RIP. Trong một tệp đối tượng có thể tái định vị, chúng chỉ là những vị trí giữ chỗ tạm thời. Mục tương ứng trong `.rela.text` tại độ lệch `0x69` ghi nhận `R_X86_64_PC32 .rodata-0x4`. psABI (Bảng 4.9) định nghĩa `R_X86_64_PC32` là `S + A - P`: địa chỉ ký hiệu cộng với số hạng cộng trừ đi địa chỉ của chính trường đang được vá.

Bộ xử lý sẽ cộng trường này vào địa chỉ của lệnh kế tiếp, do đó đích đến cuối cùng là `S + A + (end - P)`. Đối với lệnh `lea` này, trường độ dời nằm ở 4 byte cuối cùng của lệnh, vì vậy `end - P` bằng 4 và đích đến là `.rodata + (-4) + 4 = .rodata + 0x0`. Số hạng cộng -4 đã bù trừ trước cho 4 byte này. Mọi toán hạng tương đối theo RIP trong các fixture đều mang giá trị -4 ẩn trong số hạng cộng của nó, vì vậy `.rodata+0x10` trên lệnh `lea` của bản dựng `-O1` thực chất trỏ tới `.rodata+0x14`.

Bộ giải mã ghi lại độ lệch của trường độ dời bên trong từng lệnh. Hàm `resolve_reference()` tra cứu độ lệch đó trong bảng tái định vị và báo cáo vị trí mà toán hạng sẽ trỏ tới sau khi liên kết. Hàm duy nhất đó chính là thứ biến dòng lệnh vô hồn `leaq (%rip), %rcx` thành thông tin ngữ nghĩa: "bảng dữ liệu tại `.rodata+0x0`".

### Giá trị tức thời tối đa 32-bit, trừ một ngoại lệ duy nhất

SDM 2.2.1.5 quy định rằng các giá trị tức thời (immediates) giữ kích thước tối đa 32-bit trong chế độ 64-bit và được mở rộng dấu (sign-extended). Ngoại lệ duy nhất là lệnh `MOV r64, imm64` (`B8+r` có `REX.W`), mà cú pháp AT&T gọi là `movabsq`. Bộ giải mã đọc các giá trị tức thời 8-bit dưới dạng số có dấu cho các phép toán số học (`83 c1 fa` là `addl $-0x6, %ecx`) và dưới dạng số không dấu cho số lượng dịch bit hoặc bộ chọn xáo trộn (`pshufd $0xee`). Sự phân biệt này hoàn toàn trùng khớp với cách hiển thị của `llvm-objdump`.

Các lệnh rẽ nhánh gần (near branches), `push`, và `pop` mặc định sử dụng toán hạng 64-bit và bỏ qua tiền tố `66` (SDM 2.2.1.7). Đó là lý do tại sao bộ giải mã đặt tên cho chúng là `pushq`, `popq`, `callq`, và `retq` mà không cần đọc tiền tố ghi đè kích thước toán hạng.

## Tự Giải Mã Thủ Công Một Lệnh

Trước khi tin tưởng vào mã nguồn, hãy tự giải mã chuỗi `48 69 c0 67 66 66 66` từ hàm `grade` ra giấy:

1. `48` nằm trong khoảng `40` đến `4F`, nên nó là tiền tố REX với `W = 1`. Kích thước toán hạng là 64-bit.
2. `69` là mã thao tác `IMUL r, r/m, imm32`.
3. `c0` là ModRM dạng nhị phân `11 000 000`: dạng thanh ghi (`mod = 11`), trường reg là `%rax`, trường r/m là `%rax`.
4. Bốn byte tiếp theo là giá trị tức thời, đọc theo thứ tự little-endian: `0x66666667`.
5. Thứ tự cú pháp AT&T là giá trị tức thời, nguồn, đích: `imulq $0x66666667, %rax, %rax`.

Chuỗi này chiếm đúng 7 byte, và `llvm-objdump` in ra văn bản y hệt. Hằng số này chính là nghịch đảo số học cho phép clang thực hiện phép chia cho 10 bằng một phép nhân kết hợp với lệnh `sarq $0x22` ngay sau đó (dịch phải 34 bit). Bạn sẽ gặp lại cặp lệnh này trong bài học 34, nơi bộ nâng cấp mã sẽ biến nó trở lại thành biểu thức `score / 10`.

Bộ giải mã thực hiện 5 bước này lần lượt trong `Decoder.decode()`, `_State.modrm()`, và `_State.finish()`. Hãy đọc mã theo trình tự đó và đối chiếu từng nhánh điều kiện với mục tài liệu SDM tương ứng trong docstring.

## Quét Tuyến Tính (Linear Sweep)

Schwarz, Debray, và Andrews mô tả phương pháp quét tuyến tính trong Mục 3.1 của bài báo "Disassembly of Executable Code Revisited" (WCRE 2002). Bắt đầu tại byte đầu tiên của phân đoạn mã máy, giải mã một lệnh, tiếp tục tại byte ngay sau nó, và lặp lại cho đến khi chạm tới cuối phân đoạn. `objdump` hoạt động theo nguyên lý này, và `llvm-objdump` cũng vậy.

```python
def linear_sweep(code: bytes, start: int = 0, end: int | None = None) -> SweepResult:
    position = start
    while position < end:
        try:
            instruction = decode(code[:end], position)
        except DecodeError:
            undecodable.append((position, code[position]))
            position += 1
            continue
        instructions.append(instruction)
        position = instruction.end
```

Phương pháp này có một điểm yếu chí mạng, và Hình 1 trong bài báo minh họa rõ điều đó: Các byte thực chất là dữ liệu (data) lại bị giải mã nhầm thành các lệnh thực thi, khiến bộ giải mã sau đó bị lệch khỏi ranh giới lệnh thực sự trong một khoảng thời gian. Trên kiến trúc x86, bộ giải mã thường tự đồng bộ lại (re-synchronises) chỉ sau vài byte, và chính điều này làm cho lỗi trở nên cực kỳ khó phát hiện. Việc phát hiện chỉ xảy ra khi gặp phải một opcode thực sự không hợp lệ, và kết luận của bài báo là các thuật toán này "thất bại trong im lặng" (fail silently).

Dù vậy, phương pháp quét tuyến tính cũng chính là chân lý chuẩn (oracle) cho bài học này. Bộ giải mã của bạn bắt buộc phải khớp hoàn toàn với `llvm-objdump -d` trên từng lệnh của mọi fixture, bởi bản thân `llvm-objdump` cũng là một bộ quét tuyến tính. Văn bản thu thập được lưu trữ trong `code/fixtures/*.objdump.txt`. Bài kiểm thử phân tích từng dòng thành địa chỉ, chuỗi byte và văn bản lệnh, loại bỏ các chú thích sau dấu `#`, và so sánh đối chiếu. Bốn tệp đối tượng fixture chứa lần lượt 86, 56, 84, và 11 lệnh (tổng cộng 237 lệnh), và bộ giải mã của chúng ta khớp chính xác 100%.

## Duyệt Đệ Quy (Recursive Descent)

Mục 3.2 của cùng bài báo gọi phương pháp thứ hai là duyệt đệ quy (recursive traversal). Bắt đầu tại một điểm nhập đã biết và giải mã tuần tự cho đến khi gặp một lệnh chuyển giao quyền điều khiển (control transfer). Đối với một lệnh nhảy trực tiếp (direct jump), tiếp tục giải mã tại đích đến. Đối với một lệnh rẽ nhánh có điều kiện, tiếp tục giải mã tại cả đích rẽ nhánh lẫn lệnh kế tiếp ngay sau nó; và đối với một lệnh gọi hàm (`call`), tiếp tục giải mã ở lệnh kế tiếp, vì hàm được gọi sẽ trả về quyền điều khiển. Đối với một lệnh trở về (`ret`), dừng nhánh đó lại. Thuật toán duy trì một tập hợp các địa chỉ đã thăm để đảm bảo quá trình duyệt kết thúc.

```python
if instruction.flow == "branch":
    worklist.append(instruction.target)
    address = instruction.end
elif instruction.flow == "indirect-jump":
    table = resolver(instruction, instructions) if resolver else None
    if table is None:
        unresolved.append(instruction)
        break
```

Chế độ thất bại của phương pháp này hoàn toàn trái ngược. Một bước nhảy gián tiếp (indirect jump) không chứa đích đến trong các byte của nó. Quá trình duyệt buộc phải dừng lại tại đó, và mọi khối mã chỉ có thể tiếp cận được qua bước nhảy đó sẽ hoàn toàn không được biết tới. Hình 2 của bài báo chỉ ra một nguy cơ thứ hai: một tập hợp đích đến suy ra từ tái định vị lại trỏ vào ngay giữa một lệnh. Trình duyệt đệ quy trong `main.py` ghi nhận trường hợp này dưới dạng xung đột (conflict) và dừng nhánh đó lại, để sự chồng lấn trở nên minh bạch thay vì âm thầm bị giải mã hai lần.

Andriesse và các cộng sự đã đánh giá 9 bộ dịch ngược trên 981 tệp nhị phân do trình biên dịch tạo ra (USENIX Security 2016). Dữ liệu nhúng nội tuyến (inline data) và mã chồng lấn hiếm gặp trong đầu ra của trình biên dịch thực tế hơn nhiều so với giả định của giới học thuật, và bài toán hóc búa thực sự là khôi phục ranh giới hàm. Hãy ghi nhớ cả hai phát hiện này: Bẫy dữ liệu trong phần tiếp theo là hiếm gặp, nhưng bước nhảy gián tiếp trong hàm `grade` lại cực kỳ phổ biến.

## Một Fixture Khiến Quét Tuyến Tính Bị Sai và Duyệt Đệ Quy Đúng

Tệp `code/fixtures/sweep-trap.s` chứa 24 byte hợp ngữ được viết tay với một bảng 4-byte nằm ngay trong thân hàm. Hàm lập chỉ mục vào bảng bằng lệnh `movzbl (%rcx,%rax), %eax` rồi nhảy vượt qua nó:

```asm
lookup:
        movslq  %edi, %rax
        leaq    .Ltable(%rip), %rcx
        movzbl  (%rcx,%rax), %eax
        jmp     .Ldone
.Ltable:
        .byte   0x31, 0xc0, 0x90, 0x48
.Ldone:
        addl    $1, %eax
        retq
```

Biên dịch nó bằng lệnh `clang -target x86_64-unknown-linux-gnu -c sweep-trap.s`. Bốn byte của bảng được cố tình lựa chọn để chúng có thể giải mã thành các lệnh hợp lệ: `31 c0` là `xorl %eax, %eax`. `90` là `nop`. `48` là một tiền tố REX, vì vậy phương pháp quét tuyến tính sẽ gộp nó với 3 byte của lệnh `addl $1, %eax` thực sự phía sau và in ra thành `addq $0x1, %rax`. Đầu ra thu thập được từ `llvm-objdump` phản ánh chính xác sai lầm đó:

```text
       e: eb 04                        	jmp	0x14 <lookup+0x14>
      10: 31 c0                        	xorl	%eax, %eax
      12: 90                           	nop
      13: 48 83 c0 01                  	addq	$0x1, %rax
      17: c3                           	retq
```

Nhìn bề ngoài, danh sách này không có vẻ gì là sai hỏng. Từng byte đều được giải mã, mọi tên lệnh đều chuẩn tắc, và lệnh `retq` vẫn xuất hiện. Tuy nhiên, danh sách này sai nghiêm trọng ở ba điểm: Lệnh nhảy tại `0xe` có đích đến là `0x14`, nhưng không có lệnh nào trong danh sách bắt đầu tại `0x14`. Lệnh thực sự chạy tại `0x14` là lệnh 32-bit `addl`, nhưng danh sách lại hiển thị một lệnh 64-bit `addq` bắt đầu sớm hơn 1 byte. Cuối cùng, 4 byte dữ liệu của bảng lại bị biến thành mã thực thi.

Phương pháp duyệt đệ quy bắt đầu từ ký hiệu `lookup` sẽ giải mã đúng 6 lệnh, đi theo bước nhảy tới `0x14`, và báo cáo chính xác `addl $0x1, %eax`. Nó không nhận vơ bất kỳ byte nào trong 4 byte dữ liệu hay 8 byte đệm căn chỉnh trước hàm `plus_one`. Hàm `trap_report()` trong `main.py` in ra bảng đối chiếu:

```json
"linearSweep": {
  "instructions": 9,
  "atJumpTarget": {"starts": "0x13", "text": "addq\t$0x1, %rax", "startsAtTarget": false},
  "claimedDataBytes": 4,
  "claimedPaddingBytes": 8
},
"recursiveDescent": {
  "instructions": 6,
  "atJumpTarget": "addl\t$0x1, %eax",
  "claimedDataBytes": 0,
  "claimedPaddingBytes": 0
}
```

Bài kiểm thử `test_sweep_and_objdump_share_the_same_mistake` khẳng định rằng `llvm-objdump` đã in ra `addq $0x1, %rax` tại vị trí `0x13`. Chân lý chuẩn của chúng ta là một bộ quét tuyến tính và nó kế thừa toàn bộ điểm mù này. Sự đồng thuận với chân lý chuẩn kiểm tra tính đúng đắn của bộ giải mã, trong khi phương pháp duyệt đệ quy kiểm tra xem những lệnh nào thực sự được chạy.

## Các Khối Cơ Bản (Basic Blocks)

Allen định nghĩa khối cơ bản trong bài báo "Control Flow Analysis" (1970): Đó là một chuỗi các lệnh tuần tự chỉ được nhập tại lệnh đầu tiên và chỉ được thoát tại lệnh cuối cùng. Yakdan và các cộng sự diễn đạt lại định nghĩa này cho quá trình dịch ngược mã trong Mục II.A.2 của bài báo "No More Gotos" (NDSS 2015), với nút tiêu đề (header node) là điểm nhập của hàm.

Quá trình xây dựng là một lượt quét hai bước trên các lệnh đã được giải mã, được sắp xếp theo địa chỉ. Lượt đầu tiên thu thập các điểm khởi đầu khối (leaders). Một leader có thể là: điểm nhập hàm, đích đến của một nhánh rẽ hoặc bước nhảy, đích đến của một bảng nhảy, hoặc lệnh nằm ngay sau bất kỳ lệnh chuyển giao quyền điều khiển nào. Lượt thứ hai bắt đầu một khối mới tại mỗi leader và tại mỗi khoảng gián đoạn trong chuỗi địa chỉ.

```python
for instruction in self.instructions:
    if instruction.address in leaders or current is None or current.end != instruction.address:
        current = Block(instruction.address, [])
        self.blocks[instruction.address] = current
    current.instructions.append(instruction)
```

Các cạnh liên kết được suy ra từ lệnh cuối cùng của mỗi khối: Một lệnh `ret` không có khối kế tiếp. Một lệnh `jmp` có đúng một khối kế tiếp, được gắn nhãn `jump`. Một lệnh rẽ nhánh có điều kiện có hai khối kế tiếp, được gắn nhãn `taken` (nhánh được chọn) và `fallthrough` (nhánh đi thẳng). Một lệnh thông thường nằm ở cuối khối sẽ đi thẳng (`fallthrough`) sang khối tiếp theo, điều này xảy ra khi lệnh kế tiếp là một đích rẽ nhánh từ nơi khác. Một bước nhảy gián tiếp sẽ có các đích đến từ bảng của nó (gắn nhãn `table`), hoặc không có đích nào, và khối đó sẽ được ghi nhận vào danh sách `unresolved`.

Hai dữ kiện quan trọng khác cũng được rút ra từ cùng lượt quét này: Một khối mà điểm nhập không thể vươn tới được xem là không thể tiếp cận (unreachable) — phép quét tuyến tính trên hàm `grade` ở mức `-O0` sinh ra tới 5 khối như vậy. Một khối có địa chỉ bắt đầu rơi vào giữa một lệnh của danh sách phương thức khác chính là trường hợp chồng lấn từ chiếc bẫy mã máy.

## Đồ Thị Luồng Điều Khiển của Hàm total

Hàm `total` ở mức `-O1` có dung lượng 31 byte gồm 13 lệnh:

```text
       40: 85 f6                        	testl	%esi, %esi
       42: 7e 18                        	jle	0x5c <total+0x1c>
       44: 89 f1                        	movl	%esi, %ecx
       46: 31 d2                        	xorl	%edx, %edx
       48: 31 c0                        	xorl	%eax, %eax
       4a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
       50: 03 04 97                     	addl	(%rdi,%rdx,4), %eax
       53: 48 ff c2                     	incq	%rdx
       56: 48 39 d1                     	cmpq	%rdx, %rcx
       59: 75 f5                        	jne	0x50 <total+0x10>
       5b: c3                           	retq
       5c: 31 c0                        	xorl	%eax, %eax
       5e: c3                           	retq
```

Các leader được xác định là: `0x40` (điểm nhập), `0x44` (sau lệnh `jle`), `0x50` (đích của lệnh `jne`), `0x5b` (sau lệnh `jne`), và `0x5c` (đích của lệnh `jle`). Năm khối cơ bản, năm cạnh:

| Từ | Đến | Loại cạnh | Nguyên nhân |
|---|---|---|---|
| `0x40` | `0x44` | fallthrough | `jle` không rẽ nhánh, biến đếm là số dương |
| `0x40` | `0x5c` | taken | biến đếm bằng 0 hoặc âm, trả về 0 |
| `0x44` | `0x50` | fallthrough | lệnh `nopw` chỉ là đệm căn chỉnh trong khối, luồng đi xuyên qua nó |
| `0x50` | `0x50` | taken | lệnh `jne 0x50`, chính là vòng lặp |
| `0x50` | `0x5b` | fallthrough | thoát vòng lặp, trả về tổng tính được |

Lệnh `nopw` 6-byte tại `0x4a` là vùng đệm để đầu vòng lặp tại `0x50` nằm trên ranh giới căn chỉnh 16-byte. Nó có thể tiếp cận được, do đó nó thuộc về khối `0x44`. Lệnh `nop` 1-byte tại `0x5f` nằm sau lệnh `retq` cuối cùng cũng là phần đệm, nhưng không có luồng nào chạm tới nó. Phép quét tuyến tính tạo ra khối thứ sáu cho nó, và hàm `unreachable()` sẽ phát hiện ra nó.

Ở mức tối ưu hóa `-O2`, cùng hàm này phình to lên 140 byte với hai vòng lặp riêng biệt. Thân vòng lặp SSE2 tại `0x80` cộng 8 giá trị mỗi lượt bằng lệnh `paddd`, và phần dư vô hướng chạy tại `0xc0`. Đồ thị CFG có hai cạnh ngược: `0x80 -> 0x80` và `0xc0 -> 0xc0`, và bộ kiểm thử kiểm tra cả hai.

## Nút Thống Trị Bằng Thuật Toán Lặp (Dominators)

Một khối `d` thống trị (dominates) một khối `b` khi mọi đường đi từ điểm nhập đến `b` đều bắt buộc phải đi qua `d`. Nút thống trị trực tiếp (immediate dominator - idom) của `b` là nút thống trị của `b` nằm gần `b` nhất. Khái niệm này bắt nguồn từ Prosser (1959) qua Allen (1970), và nó là cấu trúc cốt lõi mà bài học 34 cần để tái tạo chính xác các cấu trúc `if` và `while`.

Cooper, Harvey, và Kennedy đã viết bài báo "A Simple, Fast Dominance Algorithm" (2001). Họ lập luận rằng việc triển khai bài toán luồng dữ liệu theo thứ tự hậu duyệt đảo ngược (reverse postorder) với biểu diễn mảng gọn nhẹ sẽ đánh bại thuật toán Lengauer-Tarjan (1979) trên các đồ thị có dưới khoảng một nghìn nút. Mục đích của bài báo là chứng minh các thuật toán đơn giản có thể chiến thắng trong thực tế và là công cụ sư phạm tuyệt vời. Đồ thị CFG của một hàm hầu như không bao giờ vượt quá một nghìn nút.

```python
def intersect(first: int, second: int) -> int:
    while first != second:
        while number[first] > number[second]:
            first = idom[first]
        while number[second] > number[first]:
            second = idom[second]
    return first

changed = True
while changed:
    changed = False
    for start in order[1:]:
        processed = [p for p in self.predecessors(start) if p in idom]
        candidate = processed[0]
        for other in processed[1:]:
            candidate = intersect(other, candidate)
        if idom.get(start) != candidate:
            idom[start] = candidate
            changed = True
```

Mảng `number` lưu trữ vị trí thứ tự hậu duyệt đảo ngược của từng khối, với điểm nhập tại vị trí 0. Hàm `intersect` di chuyển hai con trỏ ngược lên cây thống trị cho đến khi chúng gặp nhau. Bài báo gốc viết phép so sánh bằng số thứ tự postorder và dấu `<`. Mã nguồn này sử dụng số thứ tự reverse postorder, do đó phép so sánh đổi thành `>`. Mọi chi tiết khác đều tuân thủ chính xác mã giả của bài báo.

Đối với hàm `total` ở mức `-O1`, kết quả thu được là `idom(0x44) = 0x40`, `idom(0x50) = 0x44`, `idom(0x5b) = 0x50`, và `idom(0x5c) = 0x40`. Cạnh `0x50 -> 0x50` là một cạnh ngược (back edge) vì đích đến của nó thống trị điểm xuất phát của nó, và một cạnh ngược đánh dấu sự hiện diện của một vòng lặp (Allen 1970). Hàm `back_edges()` trả về cạnh này. Bài kiểm thử `test_dominators_on_a_hand_built_diamond_with_a_loop` chạy cùng thuật toán trên một chuỗi 16 byte có cấu trúc phân nhánh kim cương (diamond join), giúp hàm `intersect` kiểm tra trường hợp có hai nút tiền nhiệm.

## Bảng Nhảy trong Hàm grade ở Mức -O0

Ở mức `-O0`, clang biên dịch câu lệnh `switch` trong `grade` thành mẫu bảng nhảy độc lập vị trí:

```text
       5d: 83 e8 04                     	subl	$0x4, %eax
       60: 77 38                        	ja	0x9a <grade+0x6a>
       62: 48 8b 45 f0                  	movq	-0x10(%rbp), %rax
       66: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000069:  R_X86_64_PC32	.rodata-0x4
       6d: 48 63 04 81                  	movslq	(%rcx,%rax,4), %rax
       71: 48 01 c8                     	addq	%rcx, %rax
       74: ff e0                        	jmpq	*%rax
```

Hãy đọc đoạn mã trên thành 4 dữ kiện: Lệnh `subl $0x4` theo sau bởi `ja` là phép kiểm tra biên (bounds check): các chỉ số lớn hơn 4 sẽ nhảy thẳng về nhánh mặc định (default case) tại `0x9a`, do đó bảng có đúng 5 mục. Lệnh `leaq` cùng thông tin tái định vị của nó đưa địa chỉ bảng vào thanh ghi `%rcx`, và mục trước đã chỉ ra địa chỉ này là `.rodata+0x0`. Lệnh `movslq` nạp một mục 32-bit có dấu được nhân với tỷ lệ 4. Lệnh `addq` cộng địa chỉ cơ sở của bảng vào, vì vậy mỗi mục trong bảng chính là khoảng cách từ bảng đến khối lệnh xử lý nhánh case tương ứng.

Trong tệp đối tượng nhị phân, mọi byte của bảng này đều bằng 0. Các đích đến thực sự chỉ tồn tại dưới dạng các bản ghi tái định vị trong phân đoạn `.rela.rodata`:

```text
RELOCATION RECORDS FOR [.rodata]:
OFFSET           TYPE                     VALUE
0000000000000000 R_X86_64_PC32            .text+0x91
0000000000000004 R_X86_64_PC32            .text+0x8c
0000000000000008 R_X86_64_PC32            .text+0x87
000000000000000c R_X86_64_PC32            .text+0x82
0000000000000010 R_X86_64_PC32            .text+0x86
```

Áp dụng công thức `S + A - P` cho mục thứ `k` tại độ lệch `4k`. Giá trị được lưu trữ là `.text + A - (.rodata + 4k)`, và mã máy lại cộng `.rodata + 0` vào giá trị đó, vì vậy đích đến thực tế là `.text + A - 4k`. Mục 0 là `0x91 - 0 = 0x91`, mục 1 là `0x8c - 4 = 0x88`, và mục 2 là `0x87 - 8 = 0x7f`. Các mục 3 và 4 là `0x82 - 12 = 0x76` và `0x86 - 16 = 0x76`. Các trường hợp 6, 7, 8, 9 và 10 của biểu thức `score / 10` sẽ nhảy tới các khối lệnh lưu trữ lần lượt 1, 2, 3, 4, và 4. Điều này khớp chính xác từng dòng với mã nguồn C gốc, bao gồm cả việc dùng chung một nhánh cho điểm 9 và 10.

Hàm `recover_table()` thực hiện phép tính số học này. Hàm `jump_table_resolver()` nhận diện mẫu câu lệnh này bằng cách quét ngược từ lệnh `jmpq`. Nó tìm lệnh `addq` tạo ra thanh ghi đích, lệnh `movslq` nạp dữ liệu, lệnh `leaq` thiết lập địa chỉ cơ sở, và lệnh `ja` kiểm tra biên. Sau đó, quá trình duyệt đệ quy đẩy 4 đích đến riêng biệt này vào hàng đợi và tiếp tục phân tích. Một tệp thực thi đã liên kết sẽ chứa các giá trị cuối cùng ngay trong bảng và không còn thông tin tái định vị. Khi đó, cùng hàm này sẽ đọc trực tiếp các mục 32-bit có dấu và cộng địa chỉ bảng vào, và trường `evidence` sẽ ghi nhận nguồn dữ liệu nào đã sinh ra các đích đến đó.

Các thông số thống kê cho hàm `grade` ở mức `-O0`, trích xuất từ lệnh `python3 code/main.py cfg code/fixtures/ledger-x86_64-O0.o grade`:

| Phương pháp | Lệnh | Khối | Cạnh | Byte | Điểm sai |
|---|---|---|---|---|---|
| Quét tuyến tính | 34 | 9 | 7 | 118 mã + 10 đệm | 5 khối không tiếp cận được, không có cạnh vào 4 khối case, vùng đệm bị giải mã thành `nopw` |
| Duyệt đệ quy thuần | 25 | 3 | 2 | 82 mã | 36 byte mã máy không bao giờ chạm tới, `jmpq *%rax` tại `0x74` bị bỏ ngỏ |
| Duyệt đệ quy có bảng | 33 | 8 | 11 | 118 mã | Hoàn hảo không có lỗi |

Khi có bảng nhảy, cây thống trị được tái tạo hoàn chỉnh: `idom` của từng khối case là `0x62` (khối chứa lệnh nhảy). `idom` của khối `0xa1` là điểm nhập `0x30`, bởi vì nhánh mặc định bắt nguồn từ `0x9a` có thể đi tới `0xa1` mà không cần đi qua `0x62`.

## Trình Tối Ưu Hóa Đã Biến Đổi grade Thế Nào ở Mức -O1

Bạn sẽ thường nghe mô tả rằng hàm `grade` là một câu lệnh switch được biên dịch thành một bảng nhảy. Tuy nhiên, các byte nhị phân thực tế lại kể một câu chuyện cụ thể hơn nhiều: Ở mức `-O1` và `-O2`, hoàn toàn không có bất kỳ bước nhảy gián tiếp nào trong hàm `grade`:

```text
       2c: 83 f9 04                     	cmpl	$0x4, %ecx
       2f: 77 0c                        	ja	0x3d <grade+0x2d>
       31: 89 c8                        	movl	%ecx, %eax
       33: 48 8d 0d 00 00 00 00         	leaq	(%rip), %rcx
		0000000000000036:  R_X86_64_PC32	.rodata+0x10
       3a: 8b 04 81                     	movl	(%rcx,%rax,4), %eax
       3d: c3                           	retq
```

Số hạng cộng tái định vị `0x10` cộng với 4 byte bù trừ thông thường định vị bảng dữ liệu tại `.rodata+0x14`. Lệnh nạp là một lệnh `movl` thông thường, và các byte tại đó là `01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00 04 00 00 00`. Trình biên dịch đã thay thế toàn bộ 5 khối lệnh rẽ nhánh bằng một bảng chứa 5 giá trị trả về trực tiếp! Vẫn có bản ghi tái định vị, vẫn có bảng dữ liệu, và vẫn có phép kiểm tra biên. Nhưng luồng điều khiển giờ chỉ còn 3 khối và 3 cạnh với một lệnh rẽ nhánh có điều kiện duy nhất. Hàm `find_value_tables()` báo cáo cấu trúc này dưới dạng một `value-table` với các mục `[1, 2, 3, 4, 4]` và bằng chứng `evidence: "bytes"`.

Hàm `grade_label` ở mức `-O2` thậm chí còn tiến xa hơn một bước: Mảng `labels` chứa 5 con trỏ chuỗi đã được biến đổi thành một bảng chứa 5 độ lệch 32-bit tại `.rodata+0x0`. Mỗi mục mang một bản ghi tái định vị `R_X86_64_PC32` trỏ tới phân đoạn chứa chuỗi, và hàm sẽ cộng địa chỉ bảng vào mục mà nó nạp lên. Hàm `recover_table()` đọc 5 bản ghi tái định vị này và trả về các độ lệch `0, 2, 4, 6, 8` bên trong phân đoạn `.rodata.str1.1` (nơi lưu các chuỗi `F`, `D`, `C`, `B`, và `A`). Cùng một phép toán số học khôi phục bảng nhảy cũng có thể khôi phục bảng dữ liệu tương đối, bởi vì cả hai đều là dạng `S + A - P` tại một độ lệch đã biết.

Bài học rút ra từ hai bản dựng này là: Hãy gọi tên chính xác những gì các byte thể hiện. Một bảng nhảy sẽ làm biến đổi đồ thị luồng điều khiển. Một bảng giá trị chỉ làm thay đổi một lệnh nạp bộ nhớ. Việc gộp chung cả hai khái niệm thành "câu lệnh switch" sẽ xóa nhòa sự khác biệt cốt tử mà bài học 34 cần phải tái cấu trúc.

## Những Gì rea và Ghidra Cung Cấp Cho Bạn

Bộ công cụ rea 6.5.0 kết hợp với Ghidra 12.1.4 trả về văn bản danh sách mã máy của một hàm từ `procedure_assembly` dưới dạng một chuỗi văn bản duy nhất. Công cụ `inspect_native_instruction` trả về các byte, độ dài, văn bản giải mã, các token toán hạng, các tham chiếu có kiểu và luồng điều khiển của một lệnh, với việc phân loại lệnh trả về dựa trên mã trung gian p-code. Công cụ `xrefs` trả về các địa chỉ nguồn duy nhất đã được sắp xếp từ phương thức `ReferenceManager.getReferencesTo` của Ghidra. `analyze_function` cung cấp đồ thị CFG và trường `body` của hàm. `procedure_info` cung cấp thân hàm dưới dạng các dải địa chỉ bao gồm và đưa ra cảnh báo: "Gaps and shared tails must not be treated as owned bytes" (Các khoảng trống và phần đuôi chia sẻ không được xem là các byte thuộc quyền sở hữu của hàm).

Mỗi dữ kiện nói trên thực chất là kết quả duyệt đệ quy của Ghidra kết hợp với các bộ phân tích của nó đã được tuần tự hóa. Hồ sơ phân tích của rea ghim chặt các phiên bản hợp đồng cho `instruction_flow_evidence`, `function_references`, và `jump_table_evidence`. Do đó, một kết quả luôn nêu rõ các quy tắc đã sinh ra nó. Điều duy nhất mà các công cụ này không thể làm là hiển thị các bước tính toán chi tiết của chúng. Khi Ghidra không thể tìm thấy bảng nhảy, luồng điều khiển sẽ bị đánh dấu là "unknown", và các khối lệnh nhánh case sẽ hoàn toàn biến mất khỏi danh sách bên gọi, bên bị gọi và đồ thị CFG.

Giờ đây bạn đã nắm giữ toàn bộ các bước tính toán đó trong tay. Chạy lệnh `python3 code/main.py cfg` trên cùng tệp đối tượng và so sánh các điểm bắt đầu khối, số lượng cạnh và bảng nhảy khôi phục được với những gì nhà cung cấp trả về. Khi chúng trùng khớp, bạn có trong tay hai nguồn chứng minh độc lập. Khi chúng khác biệt, bạn có một địa chỉ cụ thể để điều tra, và đó chính là những gì bài học 35 sẽ ghi nhận làm một yếu tố chưa biết (unknown).

Một lưu ý về mặt kỹ thuật lập trình: Thư viện Capstone 6.0.0-Alpha10 (2026-07-21) đã vá các lỗ hổng đọc/ghi ngoài vùng nhớ theo mã CVE-2026-55893 và CVE-2026-55894, và bản 5.0.9 (2026-05-28) cũng đã đưa các bản sửa lỗi CVE này về bản cũ. Một bộ giải mã luôn phải đọc các byte mà nó không tự tạo ra. Giới hạn 15 byte, kiểm tra cắt cụt trong `_Cursor.byte()`, và việc kiên quyết từ chối phỏng đoán một opcode lạ chính là những rào chắn bảo vệ `main.py` trước các dữ liệu đầu vào không hợp lệ.

## Xây Dựng Mã Nguồn (Build It)

Chạy chương trình demo:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
```

Đầu ra là một báo cáo JSON hoàn chỉnh. Đối với từng tệp đối tượng fixture, nó cung cấp tỷ lệ đồng thuận với chân lý chuẩn. Đối với từng hàm, nó cung cấp số lượng lệnh, khối và cạnh theo cả ba phương pháp phân tích. Nó cũng liệt kê các khối không thể tiếp cận, các bước nhảy gián tiếp chưa giải quyết, các cạnh ngược, các nút thống trị trực tiếp và mọi bảng nhảy tìm thấy. Báo cáo kết thúc bằng phần đối chiếu bẫy mã máy.

Cấu trúc đầu ra kỳ vọng:

```json
{
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "oracle": "86/86 instructions agree with llvm-objdump",
      "functions": {
        "grade": {
          "range": "0x30-0xa6",
          "paddingBytes": 10,
          "linearSweep": {"instructions": 34, "blocks": 9, "edges": 7, "unreachableBlocks": ["0x76", "0x7f", "0x88", "0x91", "0xa6"], "...": "..."},
          "recursiveDescent": {"instructions": 25, "blocks": 3, "edges": 2, "unresolvedIndirectJumps": ["0x74"], "...": "..."},
          "recursiveDescentWithTables": {"instructions": 33, "blocks": 8, "edges": 11, "...": "..."},
          "tables": [{"kind": "jump-table", "table": ".rodata+0x0", "count": 5, "entries": ["0x91", "0x88", "0x7f", "0x76", "0x76"], "evidence": "relocation", "...": "..."}]
        }
      }
    }
  },
  "sweepTrap": {"...": "..."}
}
```

Hãy đọc `code/main.py` theo trình tự sau:

1. `ElfObject` đọc các tiêu đề phân đoạn, ký hiệu và các mục `.rela.*`, còn `symbolize()` đặt tên cho địa chỉ tương tự như `llvm-objdump`.
2. `Decoder.decode()` đọc các tiền tố và REX, sau đó phân phối xử lý theo bảng opcode 1-byte hoặc 2-byte.
3. `_State.modrm()` giải mã ModRM, SIB và trường độ dời, đồng thời ghi lại độ lệch của trường độ dời.
4. `resolve_reference()` áp dụng công thức `S + A - P` cho thông tin tái định vị trên trường độ dời hoặc giá trị tức thời.
5. `linear_sweep()` và `recursive_descent()` là hai phương pháp dịch ngược, mỗi phương pháp đều ghi lại các thất bại đặc thù của nó.
6. `recover_table()` và `jump_table_resolver()` đọc bảng dữ liệu thông qua các mục tái định vị hoặc qua các byte nhị phân trực tiếp.
7. `ControlFlowGraph` xây dựng các leader, các khối và các cạnh có kiểu, còn `dominators()` triển khai thuật toán lặp tìm nút thống trị.
8. `analyze_function()` thực thi mọi phương pháp trên một hàm và `compare_with_oracle()` kiểm tra bộ giải mã đối chiếu với chân lý chuẩn.

Bài học còn cung cấp hai câu lệnh phụ trợ: `python3 code/main.py disasm TỆP.o` in ra danh sách mã máy kèm theo đích tái định vị sau mỗi lệnh tương ứng; và `python3 code/main.py cfg TỆP.o TÊN_HÀM` in ra báo cáo phân tích chi tiết của một hàm cụ thể.

## Sử Dụng Mã Nguồn (Use It)

Hãy áp dụng ba phương pháp theo trình tự sau khi phân tích một tệp đối tượng mới:

1. Chạy phép đối chiếu với chân lý chuẩn trước tiên: Một bộ giải mã mà bất đồng với `llvm-objdump` trên một lệnh thông thường thì chắc chắn có lỗi bên trong, và mọi cấu trúc phân tích phía sau đều sẽ kế thừa sai sót này.
2. Chạy phương pháp duyệt đệ quy bắt đầu từ từng ký hiệu hàm và đọc danh sách `unresolved`. Mỗi mục trong danh sách là một bước nhảy hoặc lời gọi gián tiếp mà bạn chưa xác định được đích đến.
3. Đối với mỗi bước nhảy chưa giải quyết, hãy tìm kiếm mẫu câu lệnh bảng nhảy và thông tin tái định vị trỏ tới nó. Nếu mẫu không khớp, hãy đọc các byte lệnh trước bước nhảy và ghi lại những gì bạn thấy. Ghi chú đó chính là yếu tố chưa biết đầu tiên trong bài học 35.
4. So sánh phương pháp quét tuyến tính và duyệt đệ quy từng byte một. Các byte chỉ được phương pháp quét tuyến tính nhận diện thường là phần đệm hoặc dữ liệu nhúng. Các byte mà cả hai phương pháp đều không chạm tới là mã chết hoặc thuộc về một hàm mà bạn chưa tìm thấy.
5. Xây dựng đồ thị CFG từ phương pháp duyệt đệ quy có khôi phục bảng nhảy và đọc các cạnh ngược. Mỗi cạnh ngược đại diện cho một vòng lặp mà bài học 34 sẽ chuyển hóa thành `while` hoặc `for`.

Lưu trữ các tệp văn bản chuẩn của oracle cùng với các tệp đối tượng. Khi bạn nâng cấp lên một phiên bản trình biên dịch mới, hãy thu thập lại chúng, bởi một danh sách mã máy thay đổi đồng nghĩa với việc fixture kiểm thử đã thay đổi.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: Giải mã trên giấy rồi kiểm tra lại

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow/code
python3 -q
```

Chạy:

```python
from main import *
decode(bytes.fromhex("48 8d 0d 00 00 00 00"), 0).text
decode(bytes.fromhex("8d 44 08 fa"), 0).text
decode(bytes.fromhex("66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00"), 0).length
```

Hãy viết ra kết quả bạn kỳ vọng trước mỗi lời gọi. Sau đó thử bỏ byte `48` khỏi chuỗi đầu tiên và quan sát kích thước toán hạng thay đổi. Tiếp theo thử chạy `decode(bytes.fromhex("0f 0a"), 0)` và đọc thông báo lỗi `DecodeError`.

### Lab B: Bản ghi tái định vị trên lệnh lea

```python
elf = load_fixture("ledger-x86_64-O0.o")
code = elf.section(".text").data
lea = decode(code, 0x66)
lea.displacement_offset
resolve_reference(elf, lea)
```

Tham chiếu chỉ ra đích đến là `.rodata+0x0`. Đổi địa chỉ thành `0x116` trong hàm `grade_label` và đọc tham chiếu trỏ vào phân đoạn `.data`. Sau đó giải mã lệnh `callq` tại `0x10e` và giải quyết tham chiếu của nó. Trường tức thời mang `R_X86_64_PLT32 grade-0x4`, và đích đến được giải quyết là `0x30`, chính là điểm bắt đầu của hàm `grade`.

### Lab C: Quan sát hai phương pháp bất đồng quan điểm

```python
trap = load_fixture("sweep-trap.o")
text = trap.section(".text").data
[i.text for i in linear_sweep(text, 0, 0x20).instructions]
[i.text for i in recursive_descent(text, [0], 0x20).sorted_instructions()]
```

Đếm số lượng lệnh trong từng danh sách và tìm lệnh bắt đầu tại `0x13`. Sau đó đổi byte cuối cùng của bảng trong `sweep-trap.s` từ `0x48` thành `0xe8`, biên dịch lại và chạy lại cả hai phương pháp. Lúc này phương pháp quét tuyến tính sẽ nuốt chửng cả lệnh `addl` và `retq` thành một lệnh `callq` có đích đến nằm ngoài phân đoạn mã. Hàm sẽ trông như thể không bao giờ trả về!

### Lab D: Tính toán nút thống trị thủ công

```python
analysis = analyze_function(load_fixture("ledger-x86_64-O1.o"), "total")
analysis.cfg.edges
analysis.cfg.reverse_postorder()
analysis.cfg.dominators()
analysis.cfg.back_edges()
```

Vẽ 5 khối cơ bản ra giấy và tự tính toán các nút thống trị trước khi xem kết quả. Sau đó chạy cùng 4 dòng mã trên hàm `grade` ở mức `-O0` và giải thích lý do tại sao `idom(0xa1)` lại là `0x30` chứ không phải là `0x62`.

## Bài Tập Thực Hành (Practice Lab)

Bổ sung một điểm gọi hàm gián tiếp vào các fixture và mở rộng quy trình phân tích.

Các yêu cầu:

- Viết một hàm C thực hiện gọi hàm thông qua một con trỏ hàm được lựa chọn từ một mảng nhỏ. Biên dịch hàm này ở các mức `-O0` và `-O1`, đồng thời cam kết các tệp đối tượng cùng văn bản `llvm-objdump` tương ứng vào Git.
- Đảm bảo bộ giải mã đồng thuận hoàn toàn với chân lý chuẩn trên cả hai tệp đối tượng. Chỉ bổ sung opcode khi có một lệnh thực tế trong fixture yêu cầu.
- Mở rộng hàm `recursive_descent()` để một lời gọi gián tiếp sẽ ghi nhận vị trí gọi vào danh sách `calls` với giá trị đích là `None` và tiếp tục phân tích lệnh ngay sau nó.
- Viết một bộ giải quyết (resolver) đọc mảng con trỏ hàm thông qua các bản ghi tái định vị và báo cáo danh sách các hàm có thể được gọi.
- Bổ sung tập hợp các hàm bị gọi vào báo cáo hàm dưới dạng `indirectCalls` kèm theo loại bằng chứng cho từng đích đến.
- Viết một bài kiểm thử bị đánh trượt khi bộ giải quyết trả về một đích đến không phải là một ký hiệu hàm.

Thành công đồng nghĩa với việc lệnh `python3 code/main.py cfg` nêu rõ mọi hàm có thể bị gọi kèm theo bản ghi tái định vị chứng minh nó, và quá trình đối chiếu với chân lý chuẩn vẫn vượt qua trên mọi tệp đối tượng.

## Bài Tập Đánh Giá (Graded Lab)

Năm câu hỏi về các fixture của bài học này. Hãy tìm câu trả lời bằng chính mã nguồn bạn đã xây dựng, sau đó ghi kết quả vào `code/lab_answers.json`. Ghi số dưới dạng số nguyên hoặc chuỗi thập lục phân như `"0x40"`, và danh sách dưới dạng mảng JSON.

1. `t1`: Số lượng lệnh mà phương pháp duyệt đệ quy giải mã được trong hàm `grade` ở mức -O1.
2. `t2`: Số lượng khối cơ bản trong hàm `total` ở mức -O1.
3. `t3`: Địa chỉ đầu vòng lặp (loop header) trong hàm `total` ở mức -O1, dưới dạng chuỗi hex.
4. `t4`: Trong tệp `sweep-trap.o`, địa chỉ bắt đầu của lệnh quét tuyến tính bao trùm vị trí `0x14`, dưới dạng chuỗi hex.
5. `t5`: Số lượng mục trong bảng nhảy của hàm `grade` ở mức -O0.

Kiểm tra câu trả lời của bạn:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/lab.py
```

Trình kiểm tra sẽ tính toán các giá trị kỳ vọng từ `code/main.py` trên các fixture đã cam kết. Nó in ra `PASS`, `FAIL` hoặc `TODO` cho từng nhiệm vụ cùng điểm số, và chỉ thoát với mã 0 khi cả 5 bài đều đạt. Lệnh `python3 code/lab.py my-answers.json` cho phép kiểm tra một tệp câu trả lời khác. Bài kiểm thử `code/tests/test_lab.py` chứng minh trình chấm chấp nhận câu trả lời đúng và từ chối câu trả lời sai.

### Trực Quan Hóa Hoạt Động

Cỗ máy tương tác [Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?program=trap&sweep=linear) chạy bài học này dưới dạng khoang giải mã (decode bed) và bảng khối (block board). Đường dẫn này mở tệp `sweep-trap.o` với phương pháp quét tuyến tính: khoang hiển thị 4 byte dữ liệu bị giải mã thành mã máy, và một chiếc ghim đánh dấu đích nhảy tại `0x14` nằm bên trong lệnh `addq` tại `0x13`. Hãy chuyển bộ giải mã sang chế độ duyệt đệ quy để quan sát lệnh nhảy bỏ qua vùng dữ liệu một cách chuẩn xác.

## Sản Phẩm Đóng Gói (Lesson Artifact)

Bài học này bao gồm sản phẩm `outputs/skill-disassembly-and-control-flow.md`. Hãy sử dụng nó để chuyển đổi một tệp đối tượng và tên hàm thành một danh sách mã máy đã kiểm chứng, các khối cơ bản, các cạnh có kiểu, cây thống trị và mọi bước chuyển giao quyền điều khiển chưa được giải quyết. Kỹ năng này đòi hỏi sự đối chiếu với chân lý chuẩn, cả hai phương pháp dịch ngược, thống kê byte minh bạch, bằng chứng tái định vị cho mọi bảng khôi phục được, và bản kê khai rõ ràng những yếu tố nào vẫn còn chưa biết.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/33-disassembly-and-control-flow
python3 code/main.py
python3 code/tests/test_main.py
```

Quá trình xác minh cần chứng minh được:

- Bộ giải mã đồng thuận với `llvm-objdump -d` trên toàn bộ 237 lệnh của 4 tệp đối tượng fixture.
- Tiền tố REX.W chọn toán hạng 64-bit và REX.R cùng REX.B mở rộng các trường thanh ghi chính xác.
- ModRM và SIB kết xuất chính xác toán hạng bộ nhớ theo cú pháp AT&T, bao gồm hệ số nhân scale, index và độ dời có dấu.
- Độ dời tương đối theo RIP được ghi nhận chính xác cùng độ lệch trường bên trong lệnh.
- Các tiền tố kế thừa lặp lại và tiền tố ghi đè phân đoạn giải mã tương tự như chân lý chuẩn.
- Tiền tố bắt buộc `66` chọn đúng mã thao tác SSE và giữ nguyên các thanh ghi ở kích thước 32-bit.
- Giá trị tức thời 8-bit được mở rộng dấu cho phép toán số học và giữ nguyên không dấu cho phép dịch bit.
- Một lệnh dài hơn 15 byte, một lệnh bị cắt cụt và một opcode không xác định đều kích hoạt ngoại lệ `DecodeError`.
- `R_X86_64_PC32` trên lệnh `lea` giải quyết chuẩn xác tới bảng dữ liệu và `R_X86_64_PLT32` trên lệnh `call` giải quyết chuẩn xác tới hàm bị gọi.
- Phương pháp quét tuyến tính giải mã bảng bẫy thành mã máy và không có lệnh nào bắt đầu tại đích nhảy.
- Phương pháp duyệt đệ quy không bao giờ nhận vơ bảng dữ liệu hay phần đệm và đọc chính xác `addl $0x1, %eax` tại đích đến.
- `llvm-objdump` mắc cùng sai lầm như phương pháp quét tuyến tính trên tệp bẫy mã máy.
- Một điểm nhập rơi vào giữa một lệnh được ghi nhận là một xung đột (conflict).
- Một bước nhảy gián tiếp chưa giải quyết sẽ dừng nhánh duyệt đệ quy và được liệt kê rõ ràng.
- Hàm `total` ở mức `-O1` có 5 khối, 5 cạnh có kiểu, 1 cạnh ngược và các nút thống trị trực tiếp kỳ vọng.
- Một cấu trúc phân nhánh kim cương tự dựng kèm vòng lặp sinh ra các nút thống trị đúng theo định nghĩa thuật toán.
- Hàm `total` ở mức `-O2` có hai vòng lặp riêng biệt.
- Hàm `grade` ở mức `-O0` khôi phục được một bảng nhảy 5 mục từ `.rela.rodata` với các đích đến `0x91, 0x88, 0x7f, 0x76, 0x76`.
- Khi không có bảng nhảy, hàm `grade` ở mức `-O0` chỉ có 3 khối và 1 bước nhảy chưa giải quyết.
- Hàm `grade` ở mức `-O1` và `-O2` là một bảng giá trị `[1, 2, 3, 4, 4]` không có bước nhảy gián tiếp nào.
- Hàm `grade_label` ở mức `-O2` đọc một bảng chuỗi tương đối thông qua 5 bản ghi tái định vị.
- Mô hình hình vẽ tương tác nhúng chính xác dữ liệu mà quy trình phân tích tạo ra.

## Các Chế Độ Thất Bại Trong Môi Trường Thực Tế

| Thất bại | Công cụ yếu kém báo cáo | Điều quy trình phân tích phải chứng minh |
|---|---|---|
| Dữ liệu nằm trong phân đoạn `.text` | Danh sách lệnh nhìn có vẻ hợp lý với mọi byte đều được giải mã | Duyệt đệ quy không nhận vơ byte dữ liệu và đích nhảy bắt đầu một lệnh chuẩn |
| Phần đệm sau lệnh return cuối cùng | Một lệnh `nopw` nằm ở cuối hàm | Các byte đó hoàn toàn không thể tiếp cận được từ điểm nhập |
| Bước nhảy gián tiếp qua một bảng | Lệnh "jmp rax" và 4 khối mã chết | Địa chỉ bảng từ tái định vị `lea`, số lượng mục từ kiểm tra biên, đích đến từ `.rela.rodata` |
| Câu lệnh switch biến thành bảng giá trị | "một bảng nhảy có tái định vị" | Một lệnh nạp `movl` từ `.rodata+0x14`, 3 khối, không có chuyển giao gián tiếp |
| Lệnh gọi hàm có độ dời bằng 0 | `callq 0x113`, chính là lệnh tiếp theo | `R_X86_64_PLT32 grade-0x4` được giải quyết chuẩn xác về `0x30` |
| Bộ giải mã phỏng đoán một opcode lạ | Một tên lệnh nhìn có vẻ đúng | Ngoại lệ `DecodeError` kèm địa chỉ, và phương pháp quét tuyến tính ghi lại byte đó |
| Lệnh dài quá 15 byte | Một "lệnh" dài ngoằng bất thường | Con trỏ ném ngoại lệ trước khi cố đọc byte thứ 16 |
| Nhánh rẽ đâm vào giữa một lệnh | Hai danh sách mã máy chồng lấn lên nhau | Một bản ghi xung đột ghi nhận cả hai địa chỉ |
| Vòng lặp bị báo cáo là mã tuần tự thẳng | Không có cấu trúc `while` trong mã giả | Một cạnh ngược có đích đến thống trị điểm xuất phát của nó |
| Chân lý chuẩn thu thập từ trình biên dịch khác | Các bài kiểm thử vượt qua trên văn bản cũ | Tệp README của fixture nêu rõ trình biên dịch và câu lệnh chính xác |

## Quy Tắc Vận Hành Cốt Lõi

Hãy giải mã đối chiếu với một chân lý chuẩn, duyệt luồng điều khiển bắt đầu từ các điểm nhập đã biết, và giải trình minh bạch từng byte nhị phân dưới dạng: đã nhận diện, phần đệm, dữ liệu, hoặc không thể tiếp cận. Hãy khôi phục từng đích đến gián tiếp từ một bản ghi tái định vị hoặc một bảng dữ liệu mà bạn có thể chỉ rõ vị trí. Ghi nhận toàn bộ phần còn lại là chưa xác định (unknown) kèm theo địa chỉ cụ thể.

## Đọc Thêm

- [Intel 64 and IA-32 Architectures Software Developer's Manual, Tập 2, Chương 2: Instruction Format](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [AMD64 Architecture Programmer's Manual Tập 3, Chương 1: Instruction Encoding](https://docs.amd.com/v/u/en-US/24594_3.38_APM_Vol3_PUB)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Bảng 4.9: Relocation Types](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Schwarz, Debray, Andrews, "Disassembly of Executable Code Revisited" (WCRE 2002)](https://www2.cs.arizona.edu/~debray/Publications/disasm.pdf)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Allen, "Control Flow Analysis" (1970)](https://doi.org/10.1145/800028.808479)
- [Cooper, Harvey, Kennedy, "A Simple, Fast Dominance Algorithm" (2001)](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf)
- [Lengauer, Tarjan, "A fast algorithm for finding dominators in a flowgraph" (1979)](https://doi.org/10.1145/357062.357071)
- [Ghi chú phát hành Capstone 6.0.0-Alpha10](https://github.com/capstone-engine/capstone/releases/tag/6.0.0-Alpha10)
- Mã nguồn rea 6.5.0 tại tag `3b79af2` mà bài học này mô tả: `README.md`, `docs/ghidra-provider.md`, `docs/ghidra-dos.md`, `docs/reverse-engineering-tools.md`, `bridge/ghidra/ReaGhidraBridge.java`, `src/contracts/officialToolContracts.ts`, `src/ghidra/GhidraAnalysisProfile.ts`.
