# Liên kết, Nạp và Tệp nhị phân bị tước biểu tượng: PLT, GOT, Nhập khẩu, Hiệu chỉnh địa chỉ và Nhận diện hàm

> Một chương trình sau khi liên kết sẽ gọi những hàm mà bản thân nó không chứa. Bài học này đọc các bảng mà trình biên tập liên kết tạo ra cho các lời gọi đó trên cả ba định dạng, lần theo một lời gọi tới ô nhớ mà bộ nạp điền vào, và tìm kiếm các hàm trong một bản dựng sau khi tên gọi của chúng đã bị xóa sạch.

**Type:** Build
**Languages:** Python, C, mã assembly x86-64 và arm64
**Prerequisites:** Phase 13 · 32 (ELF from scratch), 33 (disassembly and control flow), 38 (calling conventions)
**Time:** ~160 phút

## Learning Objectives

- Quyết định từ một bản ghi tái định vị (relocation record) và các bảng ký hiệu xem trình biên tập liên kết (link editor) sẽ xử lý nó như thế nào: một khoảng cách cố định, một mục PLT đi kèm ô GOT, hay một phép tái định vị động.
- Đọc phân đoạn động (dynamic section), PLT, và GOT của một tệp thực thi ELF độc lập vị trí (position-independent executable - PIE) và lần theo một lời gọi hàm xuyên qua chúng bằng các công thức psABI.
- Đọc một tệp đối tượng COFF (tiêu đề tệp 20 byte, các mục phân đoạn 40 byte, ký hiệu 18 byte, bản ghi tái định vị 10 byte) cùng các thư mục nhập khẩu (import), xuất khẩu (export), và tái định vị cơ sở (base relocation) của một ảnh PE32+.
- Giải mã từng từ máy (word) trong chuỗi hiệu chỉnh liên kết (chained fixup chain) của Mach-O và đối chiếu từng từ với kết quả từ `dyld_info`.
- Nêu rõ những gì lệnh `strip` loại bỏ và những gì còn tồn tại trên từng định dạng, đồng thời chạy chương trình đã bị tước biểu tượng để chứng minh nó vẫn hoạt động bình thường.
- Khôi phục các điểm bắt đầu hàm của một bản dựng arm64 bị tước biểu tượng bằng `LC_FUNCTION_STARTS`, quét đoạn mở đầu (prologue scans), và các đích đến của lệnh gọi hàm, chấm điểm từng phương pháp đối chiếu với bảng ký hiệu của bản dựng song sinh chưa tước biểu tượng, và đặt tên cho các hàm bằng chữ ký dạng FLIRT.

## Vấn đề thực tế

Bạn liên kết `main.c` với `libledger.dylib` và hỏi một AI agent xem hàm `main` gọi hàm nào tại địa chỉ `0x1000005a0`. Agent dịch ngược chỉ lệnh và đọc được `bl 0x100000694`. Địa chỉ đó chứa mười hai byte: `adrp x16`, `ldr x16, [x16, #0x38]`, `br x16`. Agent lần theo thao tác nạp tới `0x100004038`, đọc tám byte `00 00 00 00 00 00 10 80`, và báo cáo rằng đây là một lời gọi thông qua con trỏ bị lỗi trỏ vào không gian nhân (kernel space).

Sau đó, bạn tước biểu tượng (strip) bản dựng tĩnh của cùng chương trình đó và hỏi nó có bao nhiêu hàm. Agent chạy `nm`, chỉ thấy một tên duy nhất là `__mh_execute_header`, và trả lời là có một hàm.

Cả hai câu trả lời đều đọc tệp như thể nó là tiến trình đang chạy. Từ máy tại `0x100004038` thực chất là một mắt xích trong chuỗi hiệu chỉnh liên kết (chained fixup). Bit 63 báo hiệu thao tác ràng buộc (bind), mười hai bit tiếp theo chỉ định vị trí của hiệu chỉnh tiếp theo, và các bit thấp cho biết cần liên kết với mục nhập khẩu nào. Trình liên kết động dyld sẽ thay thế từ máy này bằng địa chỉ của `_puts` trước khi `main` bắt đầu chạy. Tệp bị tước biểu tượng đã mất bảng ký hiệu nhưng vẫn giữ lại lệnh tải `LC_FUNCTION_STARTS`, nơi liệt kê toàn bộ năm điểm bắt đầu hàm chỉ trong chín byte.

Bài học này xây dựng năm bộ đọc bằng thư viện chuẩn của Python: Bộ đọc COFF cho các tệp đối tượng và bộ đọc PE32+ cho các thư mục import, export, và base relocation. Bộ đọc Mach-O cho các lệnh nạp (load commands), các đoạn mã đệm (stubs), bảng ký hiệu gián tiếp (indirect symbol table), chuỗi hiệu chỉnh liên kết, và cây tiền tố xuất khẩu (exports trie), đối chiếu chuẩn xác từng từ với `dyld_info`. Bộ đọc ELF cho phân đoạn động, PLT, và GOT, cộng với một bộ lập kế hoạch dự đoán liên kết PIE cho các đối tượng. Một bộ tìm kiếm hàm cho bản dựng bị tước biểu tượng với bốn phương pháp kèm điểm số đánh giá. Và một bộ tạo chữ ký theo khuôn mẫu FLIRT với các byte tái định vị được thay bằng ký tự đại diện (wildcard) cùng mã CRC16 của phần đuôi.

```figure
re-40-call-through-three-formats
```

## Trình biên tập liên kết làm gì với một bản ghi tái định vị

Bài học 32 đã đọc các bản ghi tái định vị và bài học 33 đã áp dụng công thức `S + A - P` cho lệnh `lea`. Một bản ghi xác định một vị trí `P`, một kiểu, một ký hiệu `S`, và một phần bù `A`. Bảng 4.9 của x86-64 psABI cung cấp một công thức cho mỗi loại, trong đó `L` là vị trí của mục PLT tương ứng với ký hiệu, `G` là độ lệch GOT của nó, `GOT` là địa chỉ của GOT, và `B` là địa chỉ nạp cơ sở của ảnh nhị phân. Ba kịch bản sau bao quát toàn bộ các đối tượng của chúng ta:

Trình biên tập liên kết ghi trực tiếp giá trị vào trường dữ liệu và bản ghi biến mất: Trường hợp `R_X86_64_PC32` đối chiếu với một ký hiệu được định nghĩa ngay trong các tệp đầu vào. Khoảng cách giữa hai byte trong cùng một ảnh nhị phân không bao giờ thay đổi, bất kể địa chỉ nạp là bao nhiêu.

Trình biên tập liên kết tạo một mục PLT và một ô nhớ GOT: Trường hợp `R_X86_64_PLT32` đối chiếu với một hàm chưa được định nghĩa. Lời gọi hàm nhận giá trị `L + A - P`, tức khoảng cách tới mục PLT. Mục PLT sẽ nhảy thông qua ô GOT tương ứng. Một bản ghi `R_X86_64_JUMP_SLOT` trong `.rela.plt` sẽ chỉ dẫn cho trình liên kết động điền địa chỉ vào ô GOT đó.

Trình biên tập liên kết giữ lại một phép tái định vị cho bộ nạp: `R_X86_64_64` là địa chỉ tuyệt đối 8 byte, và một tệp thực thi độc lập vị trí không thể biết địa chỉ của chính nó cho đến khi được nạp vào bộ nhớ. Bản ghi này trở thành `R_X86_64_RELATIVE` với công thức `B + A` nằm trong phân đoạn `.rela.dyn`.

Tệp `main-x86_64-O1.o` có 18 bản ghi: 14 trong `.text`, 2 trong `.data.rel.ro`, và 2 trong `.rodata`. Hàm `plan_dynamic_link()` trong `code/main.py` áp dụng ba quy tắc này cho chúng cùng với `ledger-x86_64-O1.o`:

| Bản ghi | Số lượng | Quyết định | Lý do |
|---|---|---|---|
| `R_X86_64_PC32` tới `.rodata`, `.L.str`, `fns`, bảng trong `.rodata` | 16 | Ghi tại chỗ | Ký hiệu đã được định nghĩa trong các đầu vào |
| `R_X86_64_PLT32` tới `percent_change`, `grade`, `total`, `grade_label` | 4 | Gọi trực tiếp, không qua PLT | Được định nghĩa trong `ledger-x86_64-O1.o`, do đó linker rút gọn PLT32 thành PC32 |
| `R_X86_64_PLT32` tới `puts` hai lần và `printf` hai lần | 4 | Tạo mục PLT và ô GOT | Ký hiệu chưa định nghĩa, được cung cấp bởi thư viện chia sẻ |
| `R_X86_64_64` tới `grade` trong mảng `fns[]` | 2 | `R_X86_64_RELATIVE` | Một trường tuyệt đối bên trong tệp PIE |

Hai hàm nhập khẩu tạo ra phân đoạn `.plt` rộng 48 byte (PLT0 cộng với hai mục, mỗi mục 16 byte), phân đoạn `.got.plt` rộng 40 byte (ba ô dành riêng cộng với hai ô nhập khẩu), và `.rela.plt` rộng 48 byte. Hai bản ghi tương đối tạo ra phân đoạn `.rela.dyn` rộng 48 byte. Các con số này xuất phát trực tiếp từ việc áp dụng đặc tả chuẩn vào các bản ghi của chúng ta. Tệp dựng thủ công `tiny-dyn-x86_64` triển khai phần PLT của kế hoạch này cho cùng hai hàm nhập khẩu, và kích thước của nó khớp chính xác với dự toán.

Trình biên dịch luôn ghi `PLT32` trên mọi lệnh gọi hàm, dù hàm đó đã được định nghĩa hay chưa, kể từ binutils 2.31 và LLVM 7.0.0. Quyết định cuối cùng thuộc về trình biên tập liên kết. Do đó, một mã `PLT32` trong tệp đối tượng chỉ thuần túy có nghĩa là "một lời gọi hàm", chứ không khẳng định mục PLT có thực sự được tạo ra hay không.

## Phân đoạn động, PLT và GOT

Chương 5 của tài liệu gABI quy định rằng một tệp thực thi tham gia vào quá trình liên kết động bắt buộc phải có một tiêu đề chương trình `PT_INTERP` nằm trước bất kỳ phân đoạn nạp nào, và một `PT_DYNAMIC` bao trùm phân đoạn `.dynamic`. Phân đoạn này là một mảng các cặp `Elf64_Dyn` gồm thẻ loại có dấu và một giá trị, kết thúc bằng `DT_NULL`.

Tệp `tiny-dyn-x86_64` có kích thước 5.184 byte do `build_tiny_dyn_elf.py` tạo ra từ chuẩn gABI và psABI. Phân đoạn động của nó chứa mười hai cặp tại địa chỉ `0x1000`:

| Thẻ | Mã số | Giá trị | Ý nghĩa |
|---|---|---|---|
| `DT_NEEDED` | 1 | `libc.so.6` | Độ lệch bảng chuỗi `0xb`, thư viện cần nạp |
| `DT_HASH` | 4 | `0x1a0` | Bảng băm SysV |
| `DT_STRTAB` | 5 | `0x188` | `.dynstr` |
| `DT_SYMTAB` | 6 | `0x140` | `.dynsym`, 24 byte mỗi mục |
| `DT_STRSZ` | 10 | `0x15` | Bảng chuỗi dài 21 byte |
| `DT_SYMENT` | 11 | `0x18` | Kích thước mỗi mục ký hiệu là 24 byte |
| `DT_PLTGOT` | 3 | `0x10c0` | Phân đoạn `.got.plt` |
| `DT_PLTRELSZ` | 2 | `0x30` | Hai bản ghi, mỗi bản ghi 24 byte |
| `DT_PLTREL` | 20 | 7 | Các bản ghi có kiểu `DT_RELA` |
| `DT_JMPREL` | 23 | `0x1b8` | Phân đoạn `.rela.plt` |
| `DT_FLAGS_1` | `0x6ffffffb` | `0x08000000` | Cờ `DF_1_PIE` của GNU |
| `DT_NULL` | 0 | 0 | Đánh dấu kết thúc |

Tài liệu gABI giải thích lý do `DT_JMPREL` được tách riêng: nó "chứa địa chỉ của các mục tái định vị chỉ liên kết riêng với bảng liên kết thủ tục. Việc tách riêng các mục này cho phép trình liên kết động bỏ qua chúng trong quá trình khởi tạo tiến trình, nếu cơ chế liên kết lười (lazy binding) được kích hoạt." Các cờ `DF_BIND_NOW` hoặc biến môi trường `LD_BIND_NOW` sẽ tắt liên kết lười và buộc trình liên kết động xử lý chúng ngay khi khởi động.

### PLT0 và mục liên kết cho hàm puts

Mục 5.2 của psABI mô tả hai cấu trúc trong Hình 5.2. Dưới đây là các byte thực thi của tệp mẫu:

```text
      1f0: ff 35 d2 0e 00 00    pushq  0xed2(%rip)      # 0x10c8 = GOT+8
      1f6: ff 25 d4 0e 00 00    jmpq   *0xed4(%rip)     # 0x10d0 = GOT+16
      1fc: 0f 1f 40 00          nopl   (%rax)

      200: ff 25 d2 0e 00 00    jmpq   *0xed2(%rip)     # 0x10d8 = GOT+24, ô nhớ của puts
      206: 68 00 00 00 00       pushq  $0x0             # chỉ mục 0 trong DT_JMPREL
      20b: e9 e0 ff ff ff       jmp    0x1f0            # Nhảy về PLT0
```

Tính toán độ dịch: Độ dời tương đối theo RIP được tính so với chỉ lệnh kế tiếp. `0x1f6 + 0xed2 = 0x10c8`, chính là `.got.plt + 8`. `0x1fc + 0xed4 = 0x10d0`, chính là `.got.plt + 16`. `0x206 + 0xed2 = 0x10d8`, ô nhớ thứ tư. `0x210 - 0x20 = 0x1f0`, chính là PLT0. Mỗi mục rộng 16 byte, do đó mục cho `exit` bắt đầu tại `0x210`, đẩy `$0x1` và đọc `0x10e0`.

Bảng GOT dành riêng ba ô đầu tiên: ô 0 chứa địa chỉ của `_DYNAMIC`, còn ô 1 và 2 cũng được dành riêng. Phân đoạn `.got.plt` của chúng ta chứa các giá trị `0x1000, 0, 0, 0x206, 0x216`. glibc điền ô 1 bằng cấu trúc `struct link_map` và ô 2 bằng hàm giải quyết địa chỉ `_dl_runtime_resolve` khi nạp tệp. Ô 3 trong tệp chứa `0x206`, trỏ thẳng tới lệnh `pushq` nằm ngay sau lệnh nhảy bên trong `puts@plt`.

Khi hàm được gọi lần đầu: lệnh `call 0x200` chạy, `jmpq *0x10d8` nhảy tới `pushq $0`, sau đó `jmp 0x1f0` đẩy giá trị GOT+8 và nhảy qua GOT+16 vào hàm giải quyết của bộ nạp. Bộ nạp đọc bản ghi 0 trong `DT_JMPREL`, tra cứu `puts` trong `libc.so.6`, ghi địa chỉ thực vào `0x10d8`, rồi nhảy tới hàm thực tế. Ở lần gọi thứ hai, `jmpq *0x10d8` sẽ nhảy thẳng tới hàm `puts` mà không qua bộ giải quyết nữa.

### Cơ chế RELRO và các phiên bản biểu tượng

Cơ chế RELRO (Relocation Read-Only) chuyển các phân đoạn chỉ được ghi trong giai đoạn tái định vị lên đầu phân đoạn có quyền ghi, để sau khi giải quyết xong, trình liên kết động có thể gọi `mprotect` đặt chúng thành chỉ đọc (read-only). Với liên kết lười, các ô nhớ GOT vẫn phải giữ quyền ghi (partial RELRO). Tùy chọn `-z now` bật full RELRO, đưa toàn bộ bảng GOT vào vùng chỉ đọc vì mọi biểu tượng đều được phân giải ngay từ đầu.

Tùy chọn `-fno-plt` trên trình biên dịch sẽ sinh ra các lời gọi gián tiếp trực tiếp qua GOT `jmpq *name@GOTPCREL(%rip)`, triệt tiêu hoàn toàn liên kết lười và loại bỏ sự cần thiết của PLT.

### .dynsym, .symtab, và lệnh strip

Chuẩn gABI quy định `.dynsym` và `.dynstr` là các phân đoạn được cấp phát bộ nhớ khi chạy (allocated sections), còn `.symtab` và `.strtab` là các phân đoạn không được nạp. Trình liên kết động chỉ đọc cặp phân đoạn đầu tiên và không bao giờ đọc cặp thứ hai.

Lệnh GNU `strip --strip-all` sẽ xóa `.symtab`, `.strtab`, `.debug_*`, và `.comment`. Mọi phân đoạn được cấp phát phục vụ thực thi đều được giữ nguyên vẹn: `.interp`, `.dynsym`, `.dynstr`, `.plt`, `.got`, `.got.plt`, v.v. Chương trình vẫn chạy hoàn toàn bình thường sau khi bị tước bỏ biểu tượng.

## Tệp đối tượng COFF

Trình biên dịch clang trên kiến trúc Windows x64 sinh ra tệp đối tượng COFF với cấu trúc: tiêu đề tệp 20 byte, không có tiêu đề tùy chọn (optional header), bảng phân đoạn gồm các mục 40 byte, bảng ký hiệu gồm các bản ghi 18 byte, và các bản ghi tái định vị 10 byte cho mỗi phân đoạn.

Lớp `CoffObject` trong `code/main.py` đọc toàn bộ bốn thành phần này. Tiêu đề chỉ định kiến trúc `0x8664`, 10 phân đoạn, 37 bản ghi ký hiệu. Các tên phân đoạn bắt đầu bằng `/` như `/16` biểu thị độ lệch dạng số thập phân trỏ vào bảng chuỗi, giải mã thành `.llvm_addrsig`.

Các bản ghi ký hiệu COFF chứa các bản ghi phụ trợ (auxiliary records). Một ký hiệu phân đoạn mang một bản ghi phụ chứa chiều dài, số lượng tái định vị, mã kiểm tra checksum, và lựa chọn COMDAT. Các bản ghi tái định vị gồm `IMAGE_REL_AMD64_REL32` trong `.text` (tương đương `R_X86_64_PC32` nhưng phần bù -4 được tính ngầm trong trường dữ liệu) và `IMAGE_REL_AMD64_ADDR64` trong `.rdata` (địa chỉ tuyệt đối 8 byte, sẽ trở thành tái định vị cơ sở trong ảnh thực thi).

| Đặc điểm | ELF `Elf64_Rela` | Tái định vị COFF | Mach-O `relocation_info` |
|---|---|---|---|
| Kích thước bản ghi | 24 byte | 10 byte | 8 byte |
| Các trường | offset, type và symbol trong `r_info`, addend | offset, symbol index, type | address, từ 32-bit gồm symbolnum, pcrel, length, extern, type |
| Phần bù (Addend) | Rõ ràng (explicit) | Nằm trong trường dữ liệu | Nằm trong trường dữ liệu hoặc bản ghi phụ |
| Lời gọi hàm `puts` | `R_X86_64_PLT32 puts-0x4` | `IMAGE_REL_AMD64_REL32 puts` | `ARM64_RELOC_BRANCH26 _puts` |

## Ảnh thực thi PE: Nhập khẩu, Xuất khẩu và Tái định vị cơ sở

Tệp `tiny-x86_64.exe` có cấu trúc PE chuẩn: Đoạn mồi MS-DOS bắt đầu bằng `MZ` với `e_lfanew` tại `0x3c` trỏ tới chữ ký `PE\0\0`. Tiêu đề tệp COFF theo sau bởi tiêu đề tùy chọn định dạng PE32+ (Magic `0x20b`). Trường `AddressOfEntryPoint` trỏ tới RVA `0x1000`, `ImageBase` là `0x140000000`. Tiếp theo là 16 thư mục dữ liệu (data directories).

Địa chỉ trong một ảnh PE là các RVA (Relative Virtual Addresses - Địa chỉ ảo tương đối tính từ ImageBase). Bảng phân đoạn ánh xạ các RVA này sang độ lệch tệp.

### Thư mục nhập khẩu (Import Directory)

Tại RVA `0x2000` là mảng các bộ mô tả nhập khẩu 20 byte. Bảng tra cứu nhập khẩu (Import Lookup Table) chứa các mục trỏ tới các cấu trúc gợi ý/tên (hint/name entries) gồm chỉ số hint và chuỗi tên ASCII. Bảng địa chỉ nhập khẩu (Import Address Table - IAT) có nội dung ban đầu giống hệt bảng tra cứu.

Khi nạp chương trình, bộ nạp Windows sẽ ghi đè các ô trong IAT bằng địa chỉ 64-bit thực tế của các hàm được nhập khẩu. Hoàn toàn không có liên kết lười hay stub trung gian; mã máy thực hiện lời gọi gián tiếp thẳng tới ô nhớ IAT:

```text
    100b: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # Ô nhớ IAT của puts tại RVA 0x2040
    1013: ff 15 2f 10 00 00    call qword ptr [rip + 0x102f]   # Ô nhớ IAT của exit tại RVA 0x2048
```

Chỉ lệnh `FF /2` là lời gọi gần gián tiếp (indirect near call).

### Thư mục xuất khẩu và Tái định vị cơ sở

Thư mục xuất khẩu định nghĩa bảng địa chỉ hàm, bảng con trỏ tên hàm và bảng số thứ tự (ordinal table) ánh xạ tên tới địa chỉ.

Phân đoạn `.reloc` chứa các khối tái định vị cơ sở dạng trang bộ nhớ (page RVA) kèm danh sách các mục 16-bit. Nếu bộ nạp nạp ảnh nhị phân tại địa chỉ khác `ImageBase`, nó sẽ cộng khoảng chênh lệch vào các địa chỉ được chỉ định trong bảng này. Tùy chọn `/DYNAMICBASE` được bật mặc định trong các trình liên kết hiện đại để hỗ trợ ASLR.

Một ảnh PE không dùng bảng ký hiệu COFF để lưu tên hàm; thông tin gỡ lỗi được lưu trong tệp PDB riêng.

## Ảnh thực thi Mach-O: Lệnh nạp, Stubs và Bảng ký hiệu gián tiếp

Tệp `ledger_main` là tệp thực thi arm64 được liên kết. Lệnh `otool -h` hiển thị các cờ `MH_NOUNDEFS`, `MH_DYLDLINK`, `MH_TWOLEVEL`, và `MH_PIE`. Các lệnh nạp bao gồm bốn phân đoạn `LC_SEGMENT_64` (`__PAGEZERO`, `__TEXT`, `__DATA_CONST`, `__LINKEDIT`), các lệnh nạp thư viện `LC_LOAD_DYLIB`, cùng các lệnh quản lý fixup và xuất khẩu.

Phân đoạn `__stubs` chứa các đoạn mã đệm 12 byte. Phân đoạn `__DATA_CONST` chứa `__got` gồm các ô nhớ con trỏ không liên kết lười (non-lazy symbol pointers).

### Cấu trúc một stub trên arm64

Stub cho `_puts` tại `0x100000694`:

```text
    30 00 00 90    adrp x16, 0x100004000
    10 1e 40 f9    ldr  x16, [x16, #0x38]
    00 02 1f d6    br   x16
```

Lệnh `adrp` tính toán địa chỉ trang cơ sở của vùng dữ liệu `0x100004000`. Lệnh `ldr` nạp con trỏ từ độ lệch `0x38` (tức `0x100004038`). Lệnh `br x16` nhảy thẳng tới địa chỉ vừa nạp.

Tên hàm tương ứng với ô nhớ được xác định thông qua bảng ký hiệu gián tiếp (indirect symbol table) trong `LC_DYSYMTAB`.

### Không gian tên hai cấp (Two-level namespace)

Lệnh `nm -m` hiển thị cội nguồn của từng biểu tượng, ví dụ `_puts (from libSystem)`. Thông tin thư viện cung cấp được mã hóa trực tiếp trong trường `n_desc` của bản ghi ký hiệu. Trình liên kết ghi nhận chính xác thư viện chứa biểu tượng, tránh xung đột tên giữa các thư viện khác nhau.

## Chuỗi hiệu chỉnh liên kết (Chained Fixups) theo từng byte

Lệnh `LC_DYLD_CHAINED_FIXUPS` trỏ tới cấu trúc định nghĩa trong `fixup-chains.h`. Tiêu đề xác định vị trí của bảng bắt đầu, bảng nhập khẩu, và bảng ký hiệu.

Mỗi từ máy 64-bit trong vùng dữ liệu vừa đóng vai trò là dữ liệu fixup, vừa chứa con trỏ trỏ tới mắt xích tiếp theo trong chuỗi:

```text
  dyld_chained_ptr_64_bind    ordinal:24  addend:8  reserved:19  next:12  bind:1 = 1
  dyld_chained_ptr_64_rebase  target:36   high8:8   reserved:7   next:12  bind:1 = 0
```

Từ máy đầu tiên tại `0x4000` có bit 63 bằng 1 (thao tác bind). Trường `next` chỉ định bước nhảy tới ô tiếp theo (đơn vị 4 byte). Khi bit 63 bằng 0, đó là thao tác rebase địa chỉ nội bộ. Mắt xích cuối cùng có trường `next` bằng 0.

Trình liên kết động dyld duyệt qua chuỗi này một lần duy nhất lúc nạp chương trình và điền địa chỉ thực tế của hàm vào ô nhớ. Trong định dạng chained fixups hiện đại, không còn cơ chế liên kết lười; toàn bộ các ô nhớ đều được liên kết trước khi hàm `main` thực thi.

### Cây tiền tố xuất khẩu (Exports Trie)

Lệnh `LC_DYLD_EXPORTS_TRIE` mã hóa các biểu tượng xuất khẩu dưới dạng một cây tiền tố (prefix tree / trie). Mỗi nút chứa kích thước, cờ thuộc tính, độ lệch địa chỉ của biểu tượng, và danh sách các nhánh con tương ứng với các ký tự tiếp theo. Bộ nạp tra cứu tên hàm xuất khẩu thông qua cây này với tốc độ rất cao.

## Những gì còn lại sau khi tước biểu tượng (strip)

Khi chạy `strip` trên một tệp thực thi động Mach-O, các biểu tượng cục bộ và tên hàm định nghĩa trong tệp bị loại bỏ, nhưng các biểu tượng nhập khẩu chưa định nghĩa bắt buộc phải được giữ lại để dyld có thể liên kết. Lệnh tải `LC_FUNCTION_STARTS` vẫn được bảo toàn nguyên vẹn.

Trên một thư viện động `libledger.dylib`, việc tước biểu tượng có thể xóa sạch `LC_SYMTAB`, nhưng cây tiền tố `LC_DYLD_EXPORTS_TRIE` vẫn tồn tại hoàn chỉnh vì dyld và trình biên tập liên kết tìm kiếm hàm thông qua trie chứ không dùng `LC_SYMTAB`.

| Thành phần | ELF sau `--strip-all` | PE Image | Mach-O sau `strip` |
|---|---|---|---|
| Bảng ký hiệu tĩnh | Bị xóa | Không có trong ảnh thực thi | Tên cục bộ bị xóa, tên nhập khẩu còn lại |
| Tên phục vụ bộ nạp | `.dynsym` và `.dynstr` giữ nguyên | Thư mục Import giữ nguyên | Giữ lại trong imports trie và bảng chained imports |
| Tên xuất khẩu | `.dynsym` giữ nguyên | Thư mục Export giữ nguyên | Cây tiền tố exports trie giữ nguyên |
| Điểm bắt đầu hàm | Bản ghi FDE trong `.eh_frame` | `.pdata` cho các hàm không phải hàm lá | `LC_FUNCTION_STARTS` giữ nguyên |
| Thông tin gỡ lỗi | Các phân đoạn `.debug_*` bị xóa | Nằm trong PDB riêng | Nằm trong dSYM riêng |
| Tiêu đề phân đoạn | Giữ nguyên | Giữ nguyên | Giữ nguyên |

## Nhận diện hàm khi không có bảng biểu tượng

Khi một tệp thực thi tĩnh bị tước sạch biểu tượng, lớp `FunctionFinder` áp dụng bốn phương pháp để xác định ranh giới các hàm:

### LC_FUNCTION_STARTS

Lệnh tải này trỏ tới một chuỗi byte mã hóa uleb128 đại diện cho các độ biến thiên địa chỉ giữa các hàm liên tiếp trong phân đoạn mã. Phương pháp này đạt độ chính xác 100% (5/5 hàm) mà không có bất kỳ kết quả dương tính giả nào trên cả hai mức tối ưu `-O0` và `-O2`. Bảng này được linker tạo ra phục vụ crash reporter và unwinder, và `strip` không bao giờ can thiệp vào nó.

### Quét đoạn mở đầu (Prologue scans)

Tìm kiếm các mẫu chỉ lệnh thiết lập khung ngăn xếp như `sub sp, sp, #imm`. Phương pháp này tìm được 5/5 hàm ở `-O0` (nơi mọi hàm đều tạo khung), nhưng chỉ tìm được 1/5 hàm ở `-O2`, bởi vì các hàm còn lại là hàm lá (leaf functions) không hề cấp phát ngăn xếp.

### Đích đến của lời gọi hàm (Call targets)

Mọi chỉ lệnh `bl` trong phân đoạn mã đều trỏ tới điểm bắt đầu của một hàm hợp lệ. Thu thập toàn bộ đích đến của các lệnh `bl`, loại trừ các điểm đến trong `__stubs`, và bổ sung thêm địa chỉ nhập hàm từ `LC_MAIN` sẽ khôi phục được trọn vẹn 5/5 hàm ở cả hai mức tối ưu hóa.

## Chữ ký hàm dạng FLIRT

Hex-Rays mô tả công nghệ FLIRT: mẫu nhận diện là 32 byte đầu tiên của hàm với các byte biến thiên (variant bytes) được đánh dấu ký tự đại diện. Các byte biến thiên này xuất phát từ các vị trí tái định vị và tham chiếu ngoài. Phần còn lại của hàm từ byte 33 trở đi đến vị trí biến thiên kế tiếp được tính mã băm CRC16.

Hàm `signatures_from_object()` tạo chữ ký từ các tệp đối tượng. Các byte nằm trong phạm vi của bản ghi tái định vị trở thành ký tự đại diện. Mỗi dòng chữ ký `.pat` chứa mẫu byte, chiều dài kiểm tra CRC, giá trị CRC16, chiều dài hàm và tên hàm:

```text
A00000342800004B890C8052087D091B000DC01AC0035FD6 00 0000 0018 :0000 _percent_change
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 003C :0000 _grade ^0000 l_switch.table.grade ^0001 l_switch.table.grade
E8CC8C52C8CCAC72087C289B09FD7FD308FD62930801090B081900511F110071 04 9AAF 0048 :0000 _grade_label ^0000 l_switch.table.grade_label ^0001 l_switch.table.grade_label ^0002 _labels ^0003 _labels
```

Hàm `_percent_change` dài 24 byte nên toàn bộ thân hàm nằm gọn trong mẫu và độ dài CRC là 0. Hàm `_grade` dài 60 byte, có tái định vị tại byte 36, do đó CRC bao phủ 4 byte. Đa thức CRC16 chuẩn là `0x8408`.

Ở mức `-O2`, hàm `grade_label` nhúng nội tuyến `grade`, dẫn tới việc hai hàm này trùng khớp 36 byte đầu tiên cùng 4 byte CRC (hiện tượng xung đột chữ ký). Bộ nhận diện giải quyết xung đột bằng các byte ở phần đuôi: tại byte 44, `_grade` mang giá trị `20` trong khi `_grade_label` mang giá trị `28` (khác biệt giữa nạp thanh ghi 32-bit và 64-bit), đồng thời chiều dài tổng thể là 60 byte so với 72 byte. Nhờ đó, toàn bộ 5/5 hàm đều được gán đúng tên.

## Xây dựng hệ thống

Chạy chương trình minh họa:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
```

Đầu ra là một báo cáo JSON gồm năm phần: `elf` chứa thông tin tái định vị, kế hoạch liên kết, và phân tích tệp `tiny-dyn-x86_64`; `coff` chứa phân đoạn, tái định vị và biểu tượng của tệp COFF; `pe` chứa các thư mục, nhập khẩu, xuất khẩu và lời gọi IAT của `tiny-x86_64.exe`; `macho` chứa chi tiết về `ledger_main`, chuỗi hiệu chỉnh liên kết, các hàng fixup, exports trie, và báo cáo tước biểu tượng; `identification` chứa kết quả đánh giá bốn phương pháp tìm hàm và khớp chữ ký FLIRT.

Cấu trúc định dạng kỳ vọng:

```json
{
  "elf": {"linkPlan": {"pltSymbols": ["puts", "printf"], "sizes": {".plt": 48, ".got.plt": 40, ".rela.plt": 48, ".rela.dyn": 48}, "...": "..."}, "tiny-dyn-x86_64": {"...": "..."}},
  "coff": {"main-x86_64.obj": {"...": "..."}},
  "pe": {"imports": [{"dll": "msvcrt.dll", "iat": "0x2040", "...": "..."}], "...": "..."},
  "macho": {"ledger_main": {"flags": "0x200085", "fixupRows": ["__DATA_CONST __got 0x100004000 bind libledger/_grade", "..."], "...": "..."}, "strip": {"...": "..."}},
  "identification": {"O2": {"methods": {"functionStarts": {"recall": "5/5"}, "prologueSub": {"recall": "1/5"}, "...": "..."}, "correctNames": 5}}
}
```

Hãy đọc mã nguồn trong `code/main.py` theo trình tự sau:

1. `ElfFile` đọc tiêu đề, phân đoạn, bảng biểu tượng, tái định vị, và phân tích phân đoạn động, PLT, GOT của ảnh nhị phân.
2. `plan_dynamic_link()` áp dụng bảng psABI để dự toán kích thước các phân đoạn liên kết động.
3. `CoffObject` đọc tiêu đề tệp COFF, bảng phân đoạn, biểu tượng 18-byte kèm bản ghi phụ, và tái định vị 10-byte.
4. `PeImage` đọc cấu trúc ảnh PE, giải mã thư mục nhập khẩu, xuất khẩu, tái định vị cơ sở, và giải mã các lệnh gọi `ff 15`.
5. `MachoFile` đọc các lệnh nạp, phân đoạn, bảng ký hiệu gián tiếp, stubs, chuỗi hiệu chỉnh liên kết, và cây tiền tố xuất khẩu.
6. `FunctionFinder` thực thi bốn chiến lược tìm hàm và chấm điểm độ chính xác.
7. `signatures_from_object()` và `match_signatures()` xây dựng và áp dụng chữ ký nhận diện kiểu FLIRT.

## Sử dụng hệ thống

Sử dụng bộ công cụ theo quy trình sau trên một chương trình bạn xây dựng:

1. Đọc tiêu đề tệp để xác định chính xác định dạng: tệp đối tượng tái định vị, tệp thực thi, hay thư viện chia sẻ.
2. Liệt kê các hàm nhập khẩu cùng nguồn cung cấp: `DT_NEEDED` trong ELF, thư mục Import trong PE, hoặc bảng chained imports trong Mach-O.
3. Lần theo một lời gọi hàm ngoại vi từ điểm gọi tới ô nhớ mà bộ nạp sẽ điền địa chỉ. Ghi lại các byte mã lệnh, stub, địa chỉ ô nhớ GOT/IAT, và giải mã dữ liệu hiệu chỉnh ban đầu trước khi bộ nạp can thiệp.
4. So sánh bảng ký hiệu với bảng xuất khẩu và bảng bắt đầu hàm. Những tên gọi mà bộ nạp cần sẽ tồn tại sau khi chạy `strip`, còn tên dành riêng cho trình gỡ lỗi sẽ biến mất.
5. Trên tệp đã bị tước biểu tượng, ưu tiên khai thác các bảng do chính linker tạo ra (`LC_FUNCTION_STARTS`, `.pdata`, `.eh_frame`) trước khi dùng các phương pháp quét mẫu byte, và dùng đích đến của các lệnh gọi hàm để bù đắp thiếu sót.
6. Xây dựng chữ ký từ các tệp đối tượng gốc và áp dụng nhận diện. Ghi nhận mọi điểm xung đột và phân giải bằng các byte khác biệt ở phần đuôi hàm.

## Phòng thí nghiệm tương tác

### Lab A: Phân tích PLT thủ công

Khởi động môi trường Python từ thư mục `code`:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code
python3 -q
```

Thực thi:

```python
from main import *
elf = load_elf("tiny-dyn-x86_64")
[(e.tag_name, e.text) for e in elf.dynamic_entries()]
elf.plt0()
[(e.symbol, hex(e.address), hex(e.slot), hex(e.slot_initial)) for e in elf.plt_entries()]
[hex(w) for w in elf.got_plt_words()]
```

Tính toán kết quả từ các byte mã máy trước khi gọi hàm. Sau đó thay đổi mảng `IMPORTS` trong `build_tiny_dyn_elf.py` thành ba tên hàm, tạo lại tệp, và dự đoán kích thước mới của `.plt`, `.got.plt`, và `.rela.plt`.

### Lab B: Giải mã chuỗi hiệu chỉnh liên kết

```python
exe = load_macho("ledger_main")
stub = next(s for s in exe.stubs() if s.symbol == "_puts")
stub.lines()
raw = int.from_bytes(exe.read(stub.slot, 8), "little")
hex(raw), raw >> 63, (raw >> 51) & 0xfff, raw & 0xffffff
[(i.index, i.name, i.library) for i in exe.chained_fixups()["imports"]]
```

Giải mã từ máy bind trên giấy: bit bind, trường `next` tính theo đơn vị 4 byte, và số thứ tự ordinal. Đọc từ máy cách đó 8 byte để xác nhận mắt xích tiếp theo trong chuỗi.

### Lab C: Tước biểu tượng và Liên kết lại

Trong thư mục mẫu, liên kết một chương trình mới với thư viện đã bị tước sạch biểu tượng:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries/code/fixtures
clang -target arm64-apple-macos -O1 main.c -L. -lledger.stripped -Wl,-rpath,@executable_path -o /tmp/relinked
nm /tmp/relinked | grep ' U '
dyld_info -exports libledger.stripped.dylib
```

Quá trình liên kết vẫn thành công rực rỡ dù `LC_SYMTAB` của thư viện rỗng hoàn toàn, bởi vì linker đọc trực tiếp cây tiền tố exports trie.

### Lab D: Xử lý xung đột chữ ký hàm

```python
from main import *
stripped = load_macho("static-arm64-O2.stripped")
sigs = signatures_from_object(load_macho("ledger-arm64-O2.o"))
[s.pat_line() for s in sigs]
starts = FunctionFinder(stripped).function_starts()
match_signatures(sigs, stripped, starts)
```

Đọc hai bản ghi xung đột. Tạo chữ ký từ `ledger-arm64-O0.o`, đối chiếu với `static-arm64-O0.stripped`, và giải thích từ các dòng `.pat` vì sao ở `-O0` không hề xảy ra xung đột.

## Bài tập thực hành

Giải mã các bảng liên kết kiểu cũ (legacy) và bổ sung bộ tìm hàm cho PE.

Yêu cầu thực hiện:

- Giải mã các luồng opcode rebase, bind, và lazy bind trong `LC_DYLD_INFO_ONLY` của `ledger_main_nochain` thành các hàng chứa phân đoạn, địa chỉ, loại, và đích đến, đảm bảo khớp với 21 hàng trong đầu ra của `dyld_info -fixups`.
- Thêm tiêu đề chương trình `PT_GNU_RELRO` và cờ `DT_FLAGS` với `DF_BIND_NOW` vào `build_tiny_dyn_elf.py`, đưa `.got.plt` vào phạm vi bảo vệ, và kiểm tra bằng `llvm-objdump -p`.
- Viết bộ đọc `.pdata` cho ảnh PE32+ và bộ tìm hàm sử dụng địa chỉ bắt đầu trong `RUNTIME_FUNCTION`. Biên dịch một đối tượng không có `-fno-asynchronous-unwind-tables`, đọc các tái định vị `.pdata`, và xác định những hàm lá nào bị bảng này bỏ qua.
- Viết kiểm thử tự động so sánh kết quả với công cụ hệ thống.

Thành công đạt được khi `python3 code/main.py macho ledger_main_nochain` in đủ 21 hàng fixup, tệp tiny ELF mang tiêu đề `PT_GNU_RELRO`, và bộ tìm hàm `.pdata` chỉ ra chính xác các hàm hợp lệ cùng các hàm lá bị bỏ sót.

## Thành phẩm bài học

Bài học này bao gồm tệp `outputs/skill-linking-loading-and-stripped-binaries.md`. Sử dụng nó để chuyển một chương trình đã biên dịch thành danh sách hàm nhập khẩu kèm nhà cung cấp, lần vết lời gọi hàm tới ô nhớ nạp, báo cáo những gì còn lại sau khi tước biểu tượng, và khôi phục có chấm điểm tên hàm cùng vị trí bắt đầu trong bản dựng không có ký hiệu.

## Kiểm chứng thực tế

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/40-linking-loading-and-stripped-binaries
python3 code/main.py
python3 code/tests/test_main.py
```

Quá trình kiểm chứng cần chứng minh:

- Phân đoạn động của `tiny-dyn-x86_64` khớp hoàn toàn với `llvm-objdump -p` theo từng thẻ
- Hai bản ghi `R_X86_64_JUMP_SLOT` khớp với `llvm-objdump -R` cho `puts` và `exit`
- PLT0 đẩy GOT+8 và nhảy qua GOT+16; mỗi mục PLT nạp ô nhớ tương ứng và đẩy chỉ mục của nó
- Kế hoạch liên kết xác định đúng hai ký hiệu PLT, bốn lời gọi trực tiếp, và hai tái định vị tương đối
- Bảng phân đoạn COFF và 37 bản ghi ký hiệu kèm bản ghi phụ khớp với `llvm-objdump`
- Tiêu đề PE, cờ thuộc tính, thư mục Import/Export, và IAT khớp với `pefile`
- Lệnh gọi `ff 15` tại RVA `0x100b` trỏ chính xác tới ô nhớ IAT của `puts`
- 19 lệnh nạp của Mach-O và hai thư viện phụ thuộc khớp với `otool`
- Bảy stubs giải mã đúng vào tám ô nhớ `__got` và khớp với bảng ký hiệu gián tiếp
- Duyệt chuỗi hiệu chỉnh liên kết tái tạo trọn vẹn 12 từ máy gốc từ `dyld_info`
- Cây tiền tố exports trie khớp với `dyld_info -exports` trên cả thư viện gốc và bản đã tước biểu tượng
- Lệnh `strip` loại bỏ `_main`, `_fns`, `_names`, giữ lại chín tên và một mục stab, chương trình vẫn chạy chuẩn xác
- `LC_FUNCTION_STARTS` khôi phục trọn vẹn 5/5 hàm ở cả hai mức tối ưu
- Quét lệnh `sub sp` tìm được 5/5 hàm ở `-O0` và 1/5 hàm ở `-O2`
- Đích đến của lệnh `bl` cộng với điểm nhập `LC_MAIN` khôi phục đủ 5/5 hàm
- Chữ ký nhận diện đặt tên chính xác cho toàn bộ năm hàm và phân giải thành công xung đột của `grade`

## Các kịch bản lỗi trong môi trường thực tế

| Tình huống lỗi | Báo cáo từ công cụ yếu | Điều phân tích chuẩn phải chứng minh |
|---|---|---|
| Lệnh `bl` hoặc `call` trỏ vào stub | "Gọi một hàm không tên tại 0x694" | Các byte của stub, ô nhớ mà nó nạp, và tên import gắn với ô nhớ đó |
| Đọc từ máy GOT trực tiếp từ tệp | "Con trỏ bị lỗi 0x8010000000000007" | Bit 63 là bind, `next` là 2, ordinal 7 là `_puts` từ thư viện 2 |
| Thấy `PLT32` trong tệp đối tượng | "Mọi lời gọi đều đi qua PLT" | Các ký hiệu đã định nghĩa được rút gọn thành khoảng cách trực tiếp, chỉ ký hiệu ngoại vi mới qua PLT |
| Nhầm lẫn giữa partial và full RELRO | "Toàn bộ GOT là chỉ đọc" | Các ô `.got.plt` sau ba ô đầu tiên vẫn có quyền ghi nếu không bật `-z now` |
| So sánh COFF `REL32` với ELF `PC32` | "Bản dựng Windows có tái định vị khác" | Cùng độ lệch, giá trị `-4` nằm trong addend của ELF và nằm trong trường dữ liệu của COFF |
| Đọc IAT trước khi chương trình chạy | "Mục nhập khẩu trỏ vào chuỗi ký tự" | IAT có giá trị bằng bảng tra cứu cho đến khi bộ nạp điền địa chỉ thực |
| Thư viện dYLyb bị tước biểu tượng | "Không có biểu tượng, không xuất khẩu gì" | Cây tiền tố exports trie và `LC_FUNCTION_STARTS` vẫn liệt kê đầy đủ các hàm |
| Chạy `nm` trên tệp thực thi bị strip | "Chương trình chỉ có một hàm" | Bảng `LC_FUNCTION_STARTS` chứa đủ năm hàm chỉ trong chín byte |
| Quét prologue ở mức `-O2` | "Chỉ tìm thấy một hàm" | Bốn hàm lá không tạo khung ngăn xếp; phải dùng bảng starts và đích đến lệnh gọi |
| Hai hàm dùng chung 36 byte đầu | "grade_label chính là grade" | Byte đuôi tại vị trí 44 và chiều dài 60 so với 72 giúp phân biệt rõ ràng |

## Quy tắc vận hành

Hãy lần theo từng lời gọi hàm ngoại vi từ điểm gọi mã máy tới ô nhớ mà bộ nạp sẽ điền địa chỉ thực tế, và giải mã từ máy tại ô nhớ đó theo đúng định dạng tệp trước khi kết luận nó là một con trỏ rác. Chỉ thực hiện tước biểu tượng sau khi đã ghi nhận những thành phần mà bộ nạp bắt buộc phải sử dụng. Khôi phục điểm bắt đầu hàm từ các bảng cấu trúc do linker tạo ra trước khi quét các mẫu byte, chấm điểm từng phương pháp đối chiếu với bản dựng có biểu tượng, và ghi lại chi tiết các trường hợp xung đột chữ ký kèm byte giải quyết.

## Tài liệu đọc thêm

- [System V gABI, chapter 5: Program Loading and Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement: section 5.2 and Table 4.9](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Drepper, "How To Write Shared Libraries", version 4.1.2](https://akkadia.org/drepper/dsohowto.pdf)
- [Drepper, "ELF Symbol Versioning"](https://akkadia.org/drepper/symbol-versioning)
- [Jelinek, "Little hardening DSOs/executables against exploits", binutils list, 2004-01-06](https://www.sourceware.org/ml/binutils/2004-01/msg00070.html)
- [Taylor, "Linker relro", 2008](https://www.airs.com/blog/archives/189)
- [binutils commit bd7ab16b: PLT32 for calls, 2018-02-13](https://gnu.googlesource.com/binutils-gdb/+/bd7ab16b4537788ad53521c45469a1bdae84ad4a) and [LLVM D43383](https://reviews.llvm.org/D43383)
- [GNU Binutils: strip](https://sourceware.org/binutils/docs/binutils/strip.html) and [llvm-objcopy](https://llvm.org/docs/CommandGuide/llvm-objcopy.html)
- [Microsoft, PE Format](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [CALL instruction reference, FF /2](https://www.felixcloutier.com/x86/call)
- [Apple, mach-o/loader.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h) and [mach-o/nlist.h](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h)
- [Apple dyld, mach-o/fixup-chains.h](https://github.com/apple-oss-distributions/dyld/blob/main/include/mach-o/fixup-chains.h)
- [Apple ld64, stub_arm64.hpp](https://github.com/apple-oss-distributions/ld64/blob/main/src/ld/passes/stubs/stub_arm64.hpp)
- [Xcode 13 release notes: chained fixups](https://developer.apple.com/documentation/xcode-release-notes/xcode-13-release-notes) and [WWDC22 session 110362, "Link fast: Improve build and launch times"](https://developer.apple.com/videos/play/wwdc2022/110362/)
- [strip(1)](https://keith.github.io/xcode-man-pages/strip.1.html), [ld(1)](https://keith.github.io/xcode-man-pages/ld.1.html), and [dyld(1)](https://keith.github.io/xcode-man-pages/dyld.1.html)
- [Hex-Rays, IDA F.L.I.R.T. Technology: In-Depth](https://docs.hex-rays.com/core/flirt/concepts/ida-f.l.i.r.t.-technology-in-depth) and [Lumina overview](https://docs.hex-rays.com/add-ons/lumina/overview)
- [Ghidra Function ID help](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/FunctionID/src/main/help/help/topics/FunctionID/FunctionID.html) and [AARCH64 function start patterns](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Processors/AARCH64/data/patterns/AARCH64_LE_patterns.xml)
- [Vector 35, WARP](https://github.com/Vector35/warp) and [Binary Ninja WARP guide](https://docs.binary.ninja/guide/warp.html)
- [BinDiff manual](https://www.zynamics.com/bindiff/manual/) and [BinDiff source](https://github.com/google/bindiff)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [rizin pat.c, the .pat parser](https://github.com/rizinorg/rizin/blob/dev/librz/sign/pat.c), [radare2 flirt.c](https://github.com/radareorg/radare2/blob/master/libr/anal/flirt.c), and [radare2 book: Zignatures](https://book.rada.re/signatures/zignatures.html)
- [Andriesse et al., "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse) and [Andriesse, Slowinska, Bos, "Compiler-Agnostic Function Detection in Binaries" (EuroS&P 2017)](https://danielandriesse.com/papers/eurosp-2017.pdf)
- [Bao et al., ByteWeight (USENIX Security 2014)](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/bao) and [Shin, Song, Moazzezi, "Recognizing Functions in Binaries with Neural Networks" (USENIX Security 2015)](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/shin)
- [LIEF changelog](https://lief.re/doc/latest/changelog.html), [pyelftools](https://github.com/eliben/pyelftools), and [pefile](https://github.com/erocarrera/pefile)
