# Giải phẫu Chương trình: Bộ Đọc ELF Tự Xây dựng từ Đầu

> Một chương trình đã biên dịch thực chất là một tập hợp các bảng trỏ lẫn nhau. Một bộ đọc do chính bạn viết sẽ cung cấp đầu ra mà bạn có thể tự mình kiểm tra chuẩn xác đến từng byte.

**Type:** Build
**Languages:** Python (stdlib), C (chỉ dùng làm fixture)
**Prerequisites:** Phase 0 · 01 (Dev Environment) với clang đã cài đặt, Python 3.10 trở lên, số học thập lục phân (hexadecimal arithmetic)
**Time:** ~120 phút

## Mục tiêu Học tập

- Nắm vững quy tắc ủy quyền cho mọi đối tượng mục tiêu trong học phần này cùng bốn giới hạn của Điều luật 17 U.S.C. 1201(f).
- Đọc tiêu đề ELF (ELF header), bảng tiêu đề phân đoạn (section header table) và cả hai bảng chuỗi (string tables) bằng `struct.unpack`.
- Giải mã các mục `.symtab` và các bản ghi `.rela` bằng các định danh quy chuẩn từ gABI và x86-64 psABI.
- Trích xuất chính xác các byte của một hàm được chỉ định theo tên và ánh xạ từng độ lệch tệp (file offset) tới cấu trúc sở hữu nó.
- Đọc các tiêu đề chương trình (program headers) và ánh xạ một địa chỉ ảo (virtual address) tới độ lệch tệp thông qua `PT_LOAD`.
- Đọc các lệnh nạp (load commands) của một đối tượng Mach-O và gọi tên khái niệm tương ứng trong định dạng ELF.

## Trước Khi Bắt Đầu: Quyền Ủy Quyền và Khung Pháp Lý

Mọi đối tượng mục tiêu trong các bài học từ 32 đến 36 đều là các chương trình do chính bạn viết và biên dịch. Mã nguồn là `code/fixtures/ledger.c`, bao gồm bốn hàm C ngắn gọn. Bạn sở hữu hoàn toàn mã nguồn, câu lệnh biên dịch và các tệp đối tượng được tạo ra. Không có bất kỳ giấy phép nào cần phải đọc lén, không có cơ chế bảo vệ nào cần vượt qua, và không có bên thứ ba nào bị truy cập mã nguồn. Học phần này tuyệt đối không bao giờ đề cập đến các cơ chế bảo vệ phần mềm, kiểm tra bản quyền hay các chương trình mà bạn không sở hữu.

Quy tắc này tồn tại vì pháp luật xem việc dịch ngược mã máy (disassembly) tương đương với hành vi sao chép (copying). Trong án lệ *Sega Enterprises Ltd. v. Accolade, Inc.*, 977 F.2d 1510 (9th Cir. 1992), tòa án đã phán quyết rằng các bản sao trung gian được tạo ra trong quá trình dịch ngược mã là các bản sao thuộc phạm vi điều chỉnh của Đạo luật Bản quyền (Copyright Act). Sau đó, tòa tuyên bố rằng "khi dịch ngược mã là con đường duy nhất để tiếp cận các ý tưởng và yếu tố chức năng được thể hiện trong một chương trình máy tính có bản quyền, và khi có lý do chính đáng để tìm kiếm quyền tiếp cận đó, thì việc dịch ngược mã cấu thành hành vi sử dụng hợp lý (fair use) đối với tác phẩm có bản quyền theo quy định của pháp luật."

Án lệ thứ hai là *Sony Computer Entertainment, Inc. v. Connectix Corp.*, 203 F.3d 596 (9th Cir. 2000). Connectix đã sao chép BIOS của PlayStation trong quá trình xây dựng một trình giả lập, và tòa án phán quyết các bản sao đó là sử dụng hợp lý. Cả hai vụ kiện đều bảo vệ quyền nghiên cứu, nhưng không bảo vệ mã bị sao chép trong sản phẩm thương mại cuối cùng.

Hai đạo luật sau đây thiết lập các ranh giới pháp lý. Điều 17 U.S.C. 1201(f) cho phép một người "đã có được quyền sử dụng bản sao của chương trình máy tính một cách hợp pháp" được quyền vượt qua biện pháp kiểm soát truy cập "chỉ nhằm mục đích duy nhất là xác định và phân tích các yếu tố của chương trình cần thiết để đạt được khả năng tương tác liên thông (interoperability) của một chương trình máy tính được tạo ra độc lập với các chương trình khác". Bốn giới hạn này được xếp chồng lên nhau: bản sao hợp pháp, mục đích tương tác liên thông, các yếu tố chưa từng "có sẵn một cách dễ dàng", và việc phân tích không cấu thành hành vi xâm phạm bản quyền khác. Điều 1201(f) chỉ là một ngoại lệ đối với quy tắc chống phá vỡ rào cản kỹ thuật (anti-circumvention rule); câu hỏi về quyền sao chép vẫn phải dựa trên nguyên tắc sử dụng hợp lý (fair use).

Tại Liên minh Châu Âu, Chỉ thị 2009/24/EC Điều 5(3) điều chỉnh quyền nghiên cứu. Người có quyền sử dụng một bản sao có thể "quan sát, nghiên cứu hoặc thử nghiệm hoạt động của chương trình nhằm xác định các ý tưởng và nguyên lý làm nền tảng cho bất kỳ yếu tố nào của chương trình". Quyền này được áp dụng trong khi nạp, hiển thị, chạy, truyền tải hoặc lưu trữ chương trình.

Điều 6 cho phép dịch ngược mã (decompilation) mà không cần sự ủy quyền của chủ sở hữu quyền khi việc đó là "không thể thiếu để có được thông tin cần thiết nhằm đạt được khả năng tương tác liên thông của một chương trình máy tính được tạo ra độc lập". Các điều kiện này tương đồng với 1201(f): hành động do người được cấp phép thực hiện, thông tin chưa có sẵn từ trước, và hành vi chỉ giới hạn trong các phần thực sự cần thiết. Điều 6(2) nghiêm cấm việc sử dụng kết quả cho bất kỳ mục đích nào khác hoặc dùng để phát triển một chương trình có biểu hiện tương tự đáng kể. Điều 8 tuyên bố vô hiệu đối với bất kỳ điều khoản hợp đồng nào trái với Điều 6 hoặc Điều 5(3).

| Nguồn | Quyền được phép | Điều kiện tiên quyết |
|---|---|---|
| 17 U.S.C. 1201(f) | Vượt qua kiểm soát truy cập để phân tích chương trình | Bản sao hợp pháp, mục đích tương tác, thông tin chưa có sẵn, không xâm phạm quyền |
| Chỉ thị 2009/24/EC Điều 5(3) | Quan sát, nghiên cứu và thử nghiệm chương trình đang chạy | Có quyền sử dụng bản sao, hành vi diễn ra trong quá trình nạp và chạy bình thường |
| Chỉ thị 2009/24/EC Điều 6 | Dịch ngược mã (Decompilation) | Là bên được cấp phép, mục đích tương tác, thông tin chưa có sẵn, chỉ giới hạn ở các phần cần thiết |
| *Sega v. Accolade* | Dịch ngược mã máy là sử dụng hợp lý (Fair use) | Là cách duy nhất để tiếp cận ý tưởng và yếu tố chức năng, có lý do chính đáng |
| *Sony v. Connectix* | Bản sao trung gian trong kỹ nghệ đảo ngược | Sản phẩm cuối cùng không xâm phạm bản quyền |

Những ghi chú này được trích xuất từ các đạo luật và phán quyết tư pháp, và không cấu thành lời khuyên pháp lý. Khi một mục tiêu phân tích trong tương lai không thuộc quyền sở hữu của bạn, hãy xin văn bản ủy quyền chính thức trước khi mở tệp và lưu văn bản ủy quyền đó cùng với bằng chứng phân tích của bạn. Mục 2.1 của cẩm nang Reverse Engineering 101 cung cấp đầy đủ các trích dẫn và các vụ án của Tòa án Công lý Liên minh Châu Âu (CJEU).

## Vấn đề Cần Giải Quyết

Bạn đã biên dịch `ledger.c` và chạy `llvm-objdump -t` trên tệp đối tượng. Đầu ra liệt kê hàm `percent_change` tại giá trị `0000000000000000` với kích thước `0000000000000010`. Vậy chính xác 16 byte nào trong tệp có dung lượng 1.768 byte này cấu thành nên hàm đó? Danh sách cũng hiển thị `R_X86_64_PC32 .rodata+0x10` tại độ lệch `0x36`, và 4 byte tại vị trí đó trong tệp hiện đang bằng 0. Chúng sẽ trở thành giá trị gì, và tại sao số hạng cộng (addend) lại là `0x10` trong khi bảng dữ liệu thực tế bắt đầu tại `0x14`?

Bộ công cụ `rea` trả lời các câu hỏi về bố cục thông qua công cụ `inspect_binary_layout`, nhưng công cụ đó chỉ tồn tại trên môi trường Linux x64 với pwntools 4.15.0 đứng sau biến `REA_PWNTOOLS_PYTHON`. Trên máy Mac này, nó hoàn toàn vắng bóng. Công cụ `address_to_file_offset` lại đòi hỏi một phiên Ghidra đang chạy và một bản sao đầy đủ các byte gốc của tệp với mã băm SHA-256 trùng khớp. `inspect_artifact` chỉ liệt kê nội dung chứa bên trong thùng chứa theo đường dẫn và mã băm. Nó không bao giờ tự mở các bảng dữ liệu bên trong một tệp đối tượng. Vì vậy, bạn sẽ tự tay viết bộ đọc nhị phân này trong khoảng 950 dòng mã Python bằng `struct.unpack` mà không phụ thuộc vào thư viện bên ngoài nào. Các bài kiểm thử sẽ so sánh từng bảng với văn bản `llvm-objdump` đã được thu thập sẵn vào các fixture.

````figure elf-byte-map
````

## Tệp Nhị Phân Là Các Bảng Dữ Liệu Trỏ Lẫn Nhau

Tài liệu System V gABI, chương 4, Hình 4-1, mô tả hai khung nhìn (views) của cùng một tệp nhị phân. Khung nhìn liên kết (linking view) gồm có một tiêu đề ELF, một bảng tiêu đề chương trình tùy chọn (optional program header table), các phân đoạn nội dung (sections), và một bảng tiêu đề phân đoạn bắt buộc (section header table). Khung nhìn thực thi (execution view) gồm có một tiêu đề ELF, một bảng tiêu đề chương trình bắt buộc, các phân đoạn bộ nhớ (segments), và một bảng tiêu đề phân đoạn tùy chọn. Tài liệu gABI bổ sung một câu mang tính nguyên lý cho toàn bộ bộ đọc của chúng ta: "Chỉ duy nhất tiêu đề ELF là có vị trí cố định trong tệp". Mọi cấu trúc khác đều được tìm thấy thông qua một độ lệch được lưu trữ bên trong một cấu trúc khác.

gABI cũng cố định các kiểu dữ liệu vô hướng và khẳng định rằng "ELF không sử dụng các trường bit (bit-fields)". `Elf64_Addr` và `Elf64_Off` chiếm 8 byte, `Elf64_Half` chiếm 2 byte, `Elf64_Word` chiếm 4 byte, và `Elf64_Xword` chiếm 8 byte. Tài liệu bổ trợ x86-64 psABI, mục 4.1.1, yêu cầu `ELFCLASS64`, `ELFDATA2LSB`, `e_machine` mang giá trị `EM_X86_64` (62), và `e_flags` bằng 0. Thứ tự byte thấp trước (little-endian) với các độ rộng cố định đồng nghĩa với việc hàm `struct.unpack` với tiền tố `<` có thể giải mã từng bản ghi chỉ trong một lời gọi duy nhất.

64 byte đầu tiên của `ledger-x86_64-O1.o`:

```text
00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............
00000010: 0100 3e00 0100 0000 0000 0000 0000 0000  ..>.............
00000020: 0000 0000 0000 0000 2804 0000 0000 0000  ........(.......
00000030: 0000 0000 4000 0000 0000 4000 0b00 0100  ....@.....@.....
```

Hãy đối chiếu các byte này với trang "ELF Header" của gABI. Trường `e_ident` gồm 16 byte đầu tiên: `7f 45 4c 46` (chuỗi magic), class 2 (64-bit), data 1 (little-endian), phiên bản 1, OS ABI 0 (System V), và các byte đệm.

Tiếp theo, `e_type` mang giá trị `0x0001` chính là `ET_REL`, một tệp đối tượng có thể tái định vị (relocatable object). `e_machine` `0x003e` là 62, tức `EM_X86_64`. `e_entry` bằng 0 vì tệp đối tượng chưa có điểm nhập (entry point). `e_phoff` bằng 0 vì tệp đối tượng không có các tiêu đề chương trình (program headers). `e_shoff` là `0x428`, độ lệch tệp dẫn tới bảng tiêu đề phân đoạn. Hàng cuối cùng chứa `e_ehsize` `0x40` (64 byte), `e_phentsize` 0, `e_phnum` 0, `e_shentsize` `0x40` (64 byte), `e_shnum` `0x0b` (11 tiêu đề phân đoạn), và `e_shstrndx` `0x0001`.

Bộ đọc chuyển hàng byte đó thành một lời gọi duy nhất:

```python
fields = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
```

Chuỗi định dạng đọc thẳng theo sơ đồ gABI từ trái qua phải: hai halfword, một word, ba giá trị 8-byte, một word, và sáu halfword. Bộ đọc sẽ từ chối tệp nếu magic, class hoặc thứ tự byte khác biệt, và nó cũng từ chối các giá trị `e_shentsize` hoặc `e_phentsize` không phải là 64 và 56. Một độ rộng sai lệch sẽ làm hỏng mọi phép tính độ lệch phía sau, vì vậy kiểm tra này được đặt ngay trên cùng.

## Tiêu Đề Phân Đoạn và Hai Bảng Chuỗi

Mỗi tiêu đề phân đoạn là một cấu trúc `Elf64_Shdr` rộng 64 byte (gABI Hình 4-8): `sh_name`, `sh_type`, `sh_flags`, `sh_addr`, `sh_offset`, `sh_size`, `sh_link`, `sh_info`, `sh_addralign`, và `sh_entsize`. Chuỗi định dạng giải mã là `<IIQQQQIIQQ`. Trường `sh_name` là một độ lệch tính vào phân đoạn có chỉ số là `e_shstrndx`, và chuỗi tên kết thúc tại byte 0 đầu tiên. Trong tệp đối tượng này, `e_shstrndx` là 1, và clang đã đặt tên cho phân đoạn đó là `.strtab`, do đó một bảng chuỗi duy nhất phục vụ cho cả tên phân đoạn lẫn tên ký hiệu.

Bộ đọc in ra 11 tiêu đề của tệp đối tượng biên dịch với -O1:

```text
Idx Name             Type         Flags   Offset     Size       Addr Link Info Align
  0                  NULL                    0x0      0x0        0x0    0    0     0
  1 .strtab          STRTAB                0x368     0xbe        0x0    0    0     1
  2 .text            PROGBITS     AX        0x40     0x9f        0x0    0    0    16
  3 .rela.text       RELA         I        0x2a8     0x48        0x0   10    2     8
  4 .rodata          PROGBITS     A         0xe0     0x50        0x0    0    0     8
  5 .rela.rodata     RELA         I        0x2f0     0x78        0x0   10    4     8
  6 .rodata.str1.1   PROGBITS     AMS      0x130      0xa        0x0    0    0     1
  7 .comment         PROGBITS     MS       0x13a     0x2f        0x0    0    0     1
  8 .note.GNU-stack  PROGBITS              0x169      0x0        0x0    0    0     1
  9 .llvm_addrsig    LLVM_ADDRSIG E        0x368      0x0        0x0   10    0     1
 10 .symtab          SYMTAB                0x170    0x138        0x0    1    9     8
```

Chỉ số 0 là tiêu đề null được bảo lưu. `.text` chứa 159 byte mã máy tại độ lệch tệp `0x40`, ngay sau tiêu đề ELF, với các cờ `A` (được cấp phát trong bộ nhớ - allocated) và `X` (có thể thực thi - executable). `.rodata` chứa 80 byte dữ liệu chỉ đọc. `.rodata.str1.1` chứa 5 chuỗi nhãn `F\0D\0C\0B\0A\0`, gồm 10 byte, với các cờ `M` (có thể gộp - mergeable) và `S` (chuỗi ký tự - strings).

`.comment` chứa chuỗi định danh trình biên dịch dài 47 byte: `\0Apple clang version 17.0.0 (clang-1700.6.3.2)\0`. `.llvm_addrsig` có kiểu `0x6fff4c03`, nằm trong dải dành riêng cho hệ điều hành `SHT_LOOS` đến `SHT_HIOS` mà gABI bảo lưu, và tệp `ELF.h` của LLVM đặt tên cho nó. Cờ `E` của nó là `SHF_EXCLUDE`, báo hiệu trình liên kết (linker) loại bỏ nó khi tạo tệp thực thi.

Các cột `sh_link` và `sh_info` là các con trỏ liên kết giữa các bảng với nhau. `.rela.text` có `sh_info` bằng 2 và `sh_link` bằng 10: nó vá mã cho phân đoạn 2 (`.text`) và lấy các ký hiệu tham chiếu từ phân đoạn 10 (`.symtab`). Phân đoạn `.symtab` có `sh_link` bằng 1, tức bảng chuỗi chứa tên các ký hiệu, và `sh_info` bằng 9, là chỉ số của ký hiệu phi cục bộ đầu tiên. Trường `sh_addr` bằng 0 ở mọi hàng vì tệp đối tượng chưa được gán địa chỉ bộ nhớ; trình liên kết sẽ thực hiện việc gán này sau.

Hai quy tắc của gABI cho phép bộ đọc vẽ nên một bản đồ hoàn chỉnh của tệp: Các phân đoạn "nằm liên tục và không chồng lấn lên nhau", và một tệp "có thể chứa các vùng không gian không hoạt động (inactive space)". Bộ đọc sắp xếp mọi cấu trúc theo độ lệch và điền đầy các khoảng trống:

```text
     0x0 +64    header   ELF header
    0x40 +159   section  .text
    0xdf +1     gap      inactive space
    0xe0 +80    section  .rodata
   0x130 +10    section  .rodata.str1.1
   0x13a +47    section  .comment
   0x169 +7     gap      inactive space
   0x170 +312   section  .symtab
   0x2a8 +72    section  .rela.text
   0x2f0 +120   section  .rela.rodata
   0x368 +190   section  .strtab
   0x426 +2     gap      inactive space
   0x428 +704   table    section header table
```

Ba khoảng trống có tổng dung lượng 10 byte, và mỗi khoảng trống đều phục vụ cho việc căn chỉnh bộ nhớ (alignment). Phân đoạn `.rodata` có `sh_addralign` là 8, nên nó bắt đầu tại `0xe0` thay vì `0xdf`. `.symtab` có căn chỉnh 8, nên 7 byte nằm không sử dụng sau `.comment`. Bản thân bảng tiêu đề phân đoạn cũng được căn chỉnh theo bội số của 8.

Các phân đoạn có kiểu `SHT_NOBITS` hoàn toàn không chiếm bất kỳ byte nào trong tệp nhị phân, vì vậy bản đồ sẽ bỏ qua chúng. Không có phần tử nào trong tệp đối tượng này thuộc kiểu đó. Tuy nhiên, một phân đoạn `.bss` trong một chương trình đã liên kết thì có kiểu này, và một bộ đọc nếu mù quáng tin vào `sh_size` ở đó sẽ đọc vượt quá phần cuối của tệp.

## Ký Hiệu: Ràng Buộc, Kiểu, Phân Đoạn, Giá Trị, Kích Thước

Một cấu trúc `Elf64_Sym` chiếm 24 byte (gABI Hình 4-18): `st_name`, `st_info`, `st_other`, `st_shndx`, `st_value`, `st_size`, định dạng `<IBBHQQ`. Trường `st_info` đóng gói hai trường: thuộc tính liên kết (binding) ở 4 bit cao và kiểu (type) ở 4 bit thấp. gABI đặt tên cho các ràng buộc là `STB_LOCAL` 0, `STB_GLOBAL` 1, `STB_WEAK` 2, và các kiểu là `STT_NOTYPE` 0, `STT_OBJECT` 1, `STT_FUNC` 2, `STT_SECTION` 3, `STT_FILE` 4. Hai bit thấp của `st_other` chứa thuộc tính hiển thị (visibility), ở đây luôn là `STV_DEFAULT` 0.

```text
Idx            Value   Size Bind   Type    Section          Name
  0              0x0      0 LOCAL  NOTYPE  *UND*
  1              0x0      0 LOCAL  FILE    *ABS*            ledger.c
  2              0x0     20 LOCAL  OBJECT  .rodata          labels.rel
  3              0x0      0 LOCAL  SECTION .rodata          .rodata
  4              0x2      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.1
  5              0x4      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.2
  6              0x6      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.3
  7              0x8      2 LOCAL  OBJECT  .rodata.str1.1   .L.str.4
  8              0x0      0 LOCAL  SECTION .rodata.str1.1   .rodata.str1.1
  9              0x0     16 GLOBAL FUNC    .text            percent_change
 10             0x10     46 GLOBAL FUNC    .text            grade
 11             0x40     31 GLOBAL FUNC    .text            total
 12             0x60     63 GLOBAL FUNC    .text            grade_label
```

Chỉ số 0 là ký hiệu chưa định nghĩa (undefined symbol) được bảo lưu. Trường `st_shndx` chỉ định phân đoạn định nghĩa ký hiệu, với ba giá trị bảo lưu đặc biệt: `SHN_UNDEF` 0, `SHN_ABS` `0xfff1` dành cho tên tệp nguồn, và `SHN_COMMON` `0xfff2`. Các ký hiệu phân đoạn (section symbols) có trường tên rỗng, vì vậy `llvm-objdump` sẽ in tên của chính phân đoạn đó cho chúng, và bộ đọc của chúng ta cũng làm tương tự. Quy tắc gABI quy định "mọi ký hiệu có ràng buộc STB_LOCAL phải đứng trước các ký hiệu weak và global", điều này được thể hiện rõ qua 9 hàng đầu tiên, và `.symtab` ghi nhận số 9 trong `sh_info` chính là vì lý do đó.

Trong một tệp đối tượng có thể tái định vị, `st_value` là một độ lệch bên trong phân đoạn chứa ký hiệu, bởi vì `sh_addr` bằng 0. Do đó, độ lệch tệp của một hàm là `sh_offset + st_value - sh_addr`, và độ dài của nó chính là `st_size`:

```python
section = elf.sections[symbol.st_shndx]
file_offset = section.sh_offset + (symbol.st_value - section.sh_addr)
code = elf.data[file_offset : file_offset + symbol.st_size]
```

Đối với hàm `percent_change`, giá trị đó là `0x40 + 0`, và 16 byte tương ứng là `85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3`. Kích thước cũng quan trọng không kém gì điểm bắt đầu. Hàm `grade` kết thúc tại độ lệch `0x3e` của `.text`. Hai byte `66 90` tiếp theo là lệnh `nop` mà trình biên dịch chèn vào để hàm `total` bắt đầu tại `0x40`, một ranh giới căn chỉnh 16 byte. Hai byte đó không thuộc về bất kỳ ký hiệu nào. Bài học 33 sẽ sử dụng `st_size` làm điều kiện dừng cho bộ giải mã assembly, và bài học 36 sẽ che (mask) hai byte này trước khi so sánh bản dựng lại.

## Tái Định Vị: Ý Nghĩa của Các Byte Bằng Không

Một cấu trúc `Elf64_Rela` chiếm 24 byte (gABI Hình 4-23): `r_offset`, `r_info`, và một số hạng cộng có dấu `r_addend`, định dạng `<QQq`. gABI tách `r_info` bằng hai macro: `ELF64_R_SYM(i)` là `i >> 32` và `ELF64_R_TYPE(i)` là `i & 0xffffffff`. Trong tệp đối tượng, `r_offset` là độ lệch bên trong phân đoạn được chỉ định bởi trường `sh_info` của bảng tái định vị. Tên kiểu và công thức tính toán xuất phát từ x86-64 psABI, Bảng 4.9. Trong đó: `S` là giá trị ký hiệu, `A` là số hạng cộng, `P` là vị trí được vá (place being patched), và `L` là mục PLT. `R_X86_64_64` được tính bằng `S + A`, `R_X86_64_PC32` tính bằng `S + A - P`, và `R_X86_64_PLT32` tính bằng `L + A - P`.

Tệp đối tượng biên dịch với -O1 có 8 bản ghi tái định vị:

```text
  .rela.text       0x36 R_X86_64_PC32          .rodata+0x10
  .rela.text       0x84 R_X86_64_PC32          .rodata+0x24
  .rela.text       0x93 R_X86_64_PC32          .rodata-0x4
  .rela.rodata      0x0 R_X86_64_PC32          .rodata.str1.1
  .rela.rodata      0x4 R_X86_64_PC32          .L.str.1+0x4
  .rela.rodata      0x8 R_X86_64_PC32          .L.str.2+0x8
  .rela.rodata      0xc R_X86_64_PC32          .L.str.3+0xc
  .rela.rodata     0x10 R_X86_64_PC32          .L.str.4+0x10
```

Hãy phân tích bản ghi đầu tiên. Độ lệch `0x36` nằm bên trong hàm `grade` (trải dài từ `0x10` đến `0x3e`). Lệnh tại `0x33` là `48 8d 0d 00 00 00 00`, một lệnh `lea` nạp địa chỉ tương đối theo con trỏ lệnh RIP, và độ dời (displacement) 4 byte của nó nằm tại `0x36`. Bộ xử lý sẽ cộng độ dời này vào địa chỉ của lệnh tiếp theo, tức `0x3a`.

Trình biên dịch muốn kết quả trỏ tới `.rodata + 0x14`, nơi bắt đầu một bảng gồm 5 giá trị 32-bit: `1, 2, 3, 4, 4`. Do đó, giá trị được lưu trữ phải là `(.rodata + 0x14) - 0x3a`. Công thức `S + A - P` với `P = 0x36` sẽ trở thành `.rodata + A - 0x36`. Cho hai biểu thức này bằng nhau, ta được `A = 0x14 - 4 = 0x10`. Số hạng cộng `A` mang độ lệch mục tiêu trừ đi 4 byte nằm giữa độ dời và điểm kết thúc của lệnh.

Bảng dữ liệu đó chính là cách clang biên dịch cấu trúc `switch` trong hàm `grade` ở mức tối ưu hóa -O1. Mã nguồn tính toán `score / 10` bằng phép nhân với hằng số `0x66666667` và hai phép dịch bit, trừ đi 6, so sánh với 4, và lập chỉ mục vào bảng. Tệp đối tượng ELF biên dịch với -O1 không giữ lại ký hiệu nào cho bảng này. Tham chiếu chỉ đơn thuần là ký hiệu phân đoạn `.rodata` cộng với một số hạng cộng, và chỉ bản dựng arm64 của cùng mã nguồn mới đặt tên cho nó là `l_switch.table.grade`. Bản ghi thứ hai, tại `0x84` trong `grade_label`, trỏ tới một bảng thứ hai gồm các giá trị 8-byte tại `.rodata + 0x28`. Bản ghi thứ ba, tại `0x93`, trỏ tới `.rodata + 0x0`, điểm bắt đầu của `labels.rel`.

Năm bản ghi `.rela.rodata` điền dữ liệu cho mảng `labels.rel`. Mỗi mục nằm tại độ lệch `4i` và nhận số hạng cộng `4i`, vì vậy `S + A - P` rút gọn thành địa chỉ của chuỗi trừ đi địa chỉ cơ sở của bảng. Hàm `grade_label` đọc một mục bằng lệnh `movslq`, cộng lại địa chỉ cơ sở của bảng, và thu được con trỏ chuỗi hoàn chỉnh. Bản ghi đầu tiên chỉ định ký hiệu phân đoạn `.rodata.str1.1` với số hạng cộng 0 vì chuỗi `F` nằm ở byte đầu tiên của phân đoạn đó. Các bản ghi còn lại chỉ định `.L.str.1` đến `.L.str.4`, là các nhãn cục bộ mà trình biên dịch tạo ra cho các chuỗi `D`, `C`, `B` và `A`.

Cùng một mã nguồn C nhưng khi biên dịch ở ba mức tối ưu hóa khác nhau sẽ sinh ra ba cấu trúc bảng hoàn toàn khác nhau. Bộ đọc vạch rõ sự khác biệt này mà chưa cần tới một bộ giải mã assembly:

| Bản dựng | Vị trí của `labels` | Cách `grade` lựa chọn | Cách `grade_label` gọi `grade` |
|---|---|---|---|
| -O0 | `.data`, 5 con trỏ 8-byte, 5 bản ghi `R_X86_64_64` | Một bảng nhảy (jump table) trong `.rodata`: 5 bản ghi `R_X86_64_PC32` trỏ ngược vào `.text` | Một lời gọi với `R_X86_64_PLT32 grade-0x4` |
| -O1 | `.rodata` dưới dạng `labels.rel`, 5 mục tương đối 32-bit | Một bảng giá trị tại `.rodata+0x14`, 1 bản ghi `R_X86_64_PC32` | Được nội tuyến hóa (inlined), không có lệnh gọi hàm |
| -O2 | Tương tự như -O1 | Tương tự như -O1 | Được nội tuyến hóa, hàm `total` phình to lên 140 byte |

Ở mức -O0, cấu trúc `switch` là một bảng nhảy thực thụ với 5 mục trỏ ngược vào `.text` tại `+0x91`, `+0x8c`, `+0x87`, `+0x82` và `+0x86`. Lời gọi hàm `grade` mang số hạng cộng `-0x4` cùng nguyên lý như lệnh `lea`. Ở mức -O1, bộ tối ưu hóa nhận thấy mọi nhánh `case` đều trả về một hằng số, nên nó đã thay thế toàn bộ các bước nhảy bằng một bảng kết quả tĩnh. Bài học 33 sẽ giải mã chi tiết cả hai dạng cấu trúc này.

## Tiêu Đề Chương Trình: Khung Nhìn Của Bộ Nạp (The Loader's View)

Một tệp đối tượng có thể tái định vị không có tiêu đề chương trình, do đó `e_phnum` bằng 0. Một tệp thực thi (executable) hoặc một thư viện chia sẻ (shared object) bắt buộc phải có chúng, và bộ nạp (loader) của hệ điều hành chỉ đọc các tiêu đề này mà bỏ qua mọi thứ khác. Một cấu trúc `Elf64_Phdr` chiếm 56 byte (gABI Hình 5-1): `p_type`, `p_flags`, `p_offset`, `p_vaddr`, `p_paddr`, `p_filesz`, `p_memsz`, `p_align`, định dạng `<IIQQQQQQ`. gABI quy định ba quy tắc cho các mục `PT_LOAD`: Chúng xuất hiện theo thứ tự `p_vaddr` tăng dần; trường `p_memsz` lớn hơn `p_filesz` đồng nghĩa với việc phần dôi ra được điền bằng các byte 0; và `p_vaddr` phải đồng dư với `p_offset` theo modulo `p_align`.

Mọi mục tiêu trong học phần này đều là chương trình do chính bạn tạo ra, và không có trình liên kết nào trên chiếc máy Mac này tạo ra được một tệp thực thi Linux từ `ledger.o`. Do đó, fixture `tiny-x86_64` được tạo ra từ kịch bản `code/fixtures/build_tiny_elf.py`, ghi trực tiếp 736 byte bằng `struct.pack`. Tệp này chứa một tiêu đề ELF, hai tiêu đề chương trình, 36 byte của hàm `_start`, 3 byte của phân đoạn `.rodata`, một bảng ký hiệu và 6 tiêu đề phân đoạn. Hàm `_start` ghi chuỗi `ok\n` ra bộ mô tả tệp 1 (stdout) và thoát với mã trạng thái 42. Trên Linux x86-64, tệp này thực thi hoàn hảo. Ở đây, nó đóng vai trò là fixture phục vụ kiểm thử nhánh tiêu đề chương trình.

```text
Program headers:
  PT_LOAD          r-x offset=0x0 vaddr=0x400000 filesz=0x2e0 memsz=0x2e0 align=0x1000
  PT_GNU_STACK     rw- offset=0x0 vaddr=0x0 filesz=0x0 memsz=0x0 align=0x10

entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

Một mục `PT_LOAD` duy nhất ánh xạ toàn bộ tệp 736 byte vào địa chỉ bộ nhớ `0x400000`. Bộ đọc ánh xạ một địa chỉ ảo sang độ lệch tệp bằng cách tìm phân đoạn `PT_LOAD` chứa địa chỉ đó và tính toán: `p_offset + (address - p_vaddr)`. Đối với điểm nhập (entry point), phép tính là `0 + (0x4000b0 - 0x400000) = 0xb0`, chính là điểm bắt đầu của `.text`.

Đây chính là phép toán nền tảng đằng sau công cụ `address_to_file_offset` của `rea`. Đối với một mục tiêu Ghidra, công cụ đó trả về `{address, file_offset}`. Nhà cung cấp Hopper cũng trả về `provider_file_offset` và `image_base_file_offset`. Công cụ yêu cầu một bản sao gốc hoàn chỉnh của tệp và mã SHA-256 của bản sao đó phải khớp với mục tiêu. Tài liệu của `inspect_binary_layout` trong `rea` nhấn mạnh một nguyên tắc cốt lõi: "Địa chỉ liên kết không phải là địa chỉ thực thi thời gian chạy". Một tệp thực thi độc lập vị trí (PIE - position-independent executable) sẽ được nạp tại một địa chỉ cơ sở do kernel lựa chọn ngẫu nhiên, và độ lệch tệp là tọa độ ổn định duy nhất.

Tiêu đề chương trình cũng giải thích lý do tại sao một tệp thực thi đã bị tước bỏ thông tin ký hiệu (stripped executable) vẫn chạy bình thường. Trình liên kết có thể loại bỏ hoàn toàn bảng tiêu đề phân đoạn, khi đó `e_shoff` trở về 0. Một công cụ chỉ biết duyệt các phân đoạn sẽ không nhìn thấy gì cả. Tuy nhiên, bộ nạp của hệ điều hành chưa từng quan tâm tới bảng tiêu đề phân đoạn.

## Các Khái Niệm Tương Đương Trong Định Dạng Mach-O

Apple không xuất bản tài liệu đặc tả bằng văn bản cho định dạng Mach-O. Các tệp tiêu đề C trong mã nguồn của nhân XNU chính là bản đặc tả chính thức: `EXTERNAL_HEADERS/mach-o/loader.h`, `nlist.h`, và `reloc.h`. Fixture `ledger-arm64-O1.o` chính là cùng một mã nguồn `ledger.c` được biên dịch bằng lệnh `clang -target arm64-apple-macos -O1 -fno-asynchronous-unwind-tables -c`.

Một cấu trúc `mach_header_64` chiếm 32 byte: `magic`, `cputype`, `cpusubtype`, `filetype`, `ncmds`, `sizeofcmds`, `flags`, `reserved`. Chuỗi magic là `0xfeedfacf` (`MH_MAGIC_64`). `cputype` là `0x0100000c` (`CPU_TYPE_ARM64`). `filetype` mang giá trị 1 (`MH_OBJECT`). `ncmds` là 5 và `sizeofcmds` là 616: các lệnh nạp chiếm 616 byte ngay sau tiêu đề, và mỗi lệnh bắt đầu bằng `{cmd, cmdsize}`. Cờ `0x2000` là `MH_SUBSECTIONS_VIA_SYMBOLS`.

```text
Load commands:
  0 LC_SEGMENT_64                cmdsize=472 at 0x20
  1 LC_BUILD_VERSION             cmdsize=24 at 0x1f8
  2 LC_LINKER_OPTIMIZATION_HINT  cmdsize=16 at 0x210
  3 LC_SYMTAB                    cmdsize=24 at 0x220
  4 LC_DYSYMTAB                  cmdsize=80 at 0x238
```

Lệnh `LC_SEGMENT_64` (0x19) là câu trả lời của Mach-O cho cả hai bảng dữ liệu của ELF. Cấu trúc `segment_command_64` của nó chứa `segname`, `vmaddr`, `vmsize`, `fileoff`, `filesize`, `maxprot`, `initprot`, `nsects`, và `flags`. Sau đó, `nsects` bản ghi thuộc kiểu `section_64` (mỗi bản ghi rộng 80 byte) tiếp nối ngay trong cùng một lệnh. Chú thích trong `loader.h` về kiểu `MH_OBJECT` nêu rõ các phân đoạn nằm "trong một segment không tên và không có đệm phân đoạn (segment padding)". Fixture phản ánh chính xác điều đó: một segment có tên rỗng, `fileoff` bằng 648, `filesize` bằng 456, và chứa 5 section.

```text
  Idx Section                    Addr   Size Offset Reloff Nreloc Type
    1 __TEXT,__text               0x0   0xd0    648   1104      6 S_REGULAR
    2 __DATA,__const             0xd0   0x28    856   1152      5 S_REGULAR
    3 __TEXT,__cstring           0xf8    0xa    896      0      0 S_CSTRING_LITERALS
    4 __TEXT,__const            0x108   0x40    912      0      0 S_REGULAR
    5 __LD,__compact_unwind     0x148   0x80    976   1192      4 S_REGULAR
```

Mỗi section định danh segment cha của mình trong trường `segname`, vì vậy hai section cùng có tên `__const` vẫn phân biệt rạch ròi vì một section thuộc `__DATA,__const` và section kia thuộc `__TEXT,__const`. Thông tin tái định vị được gắn trên từng section thông qua `reloff` và `nreloc`, trong khi ELF sử dụng một phân đoạn `.rela.X` riêng biệt cho mỗi mục tiêu.

Mỗi bản ghi `relocation_info` chiếm 8 byte. Trường `r_address` có dấu đứng đầu tiên. Một trường 32-bit tiếp theo được đóng gói thành `r_symbolnum` (24 bit), `r_pcrel` (1 bit), `r_length` (2 bit), `r_extern` (1 bit), và `r_type` (4 bit). Sáu bản ghi của `__text` là ba cặp `ARM64_RELOC_PAGE21` và `ARM64_RELOC_PAGEOFF12`, tương ứng với một cặp cho mỗi tham chiếu bảng. Cặp lệnh arm64 `adrp` cộng với `add` thực hiện chức năng tương đương một lệnh `lea` trên x86-64, do đó mỗi tham chiếu cần hai bản ghi tái định vị. Năm bản ghi của `__DATA,__const` là `ARM64_RELOC_UNSIGNED` với `r_length` bằng 3, nghĩa là mỗi bản ghi vá một con trỏ tuyệt đối 8-byte.

Ở mức -O1, bản dựng arm64 giữ mảng `labels` dưới dạng các con trỏ thông thường trong khi bản dựng x86-64 tạo ra một bảng dữ liệu tương đối.

Lệnh `LC_SYMTAB` (0x2) lưu trữ `symoff` bằng 1240, `nsyms` bằng 17, `stroff` bằng 1512, `strsize` bằng 176; và 1512 cộng với 176 bằng 1688, chính là kích thước tệp. Mỗi mục `nlist_64` chiếm 16 byte: `n_strx`, `n_type`, `n_sect`, `n_desc`, `n_value`. `n_type` là một trường bit. Khi `N_TYPE` (mặt nạ `0x0e`) bằng `N_SECT` (`0xe`), điều đó có nghĩa là ký hiệu được định nghĩa trong section có số thứ tự (bắt đầu từ 1) là `n_sect`. Cờ `N_EXT` (`0x01`) đánh dấu ký hiệu ngoại bộ (external symbol).

Cấu trúc `nlist_64` hoàn toàn không có trường kích thước (size). Để lấy được 24 byte của hàm `_percent_change`, bộ đọc sẽ tìm địa chỉ của ký hiệu tiếp theo trong cùng section, ở đây là `_grade` tại `0x18`. Các ký hiệu C mang tiền tố `_` trên kiến trúc này, vì vậy hàm có tên là `_percent_change`.

Lệnh `LC_DYSYMTAB` (0xb) ghi nhận cùng quy tắc ưu tiên ký hiệu cục bộ như trường `sh_info` của ELF: `ilocalsym` 0, `nlocalsym` 13, `iextdefsym` 13, `nextdefsym` 4. Lệnh `LC_BUILD_VERSION` (0x32) ghi lại `platform` 1 (macOS), `minos` 26.0, và `sdk` 26.2 dưới dạng số nguyên đóng gói `XXXX.YY.ZZ`. Lệnh này chính là lý do tại sao fixture arm64 được cam kết vào Git dưới dạng tệp byte nhị phân tĩnh. Việc biên dịch lại trên một máy Mac khác với SDK khác sẽ sinh ra một tệp có nội dung khác.

Phương thức `compare_formats` của bộ đọc đặt hai tệp nằm cạnh nhau để đối chiếu:

| Khái niệm | ELF, `ledger-x86_64-O1.o` | Mach-O, `ledger-arm64-O1.o` |
|---|---|---|
| Định danh tệp | `e_ident` `7f 45 4c 46`, class 64, data LSB | `magic` `0xfeedfacf` |
| Kiến trúc máy | `e_machine` 62, `EM_X86_64` | `cputype` `0x100000c`, `CPU_TYPE_ARM64` |
| Kiểu tệp | `e_type` 1, `ET_REL` | `filetype` 1, `MH_OBJECT` |
| Vị trí các bảng | `e_shoff` `0x428`, 11 tiêu đề phân đoạn | 5 lệnh nạp, 616 byte ngay sau tiêu đề |
| Đoạn mã máy | `.text` tại `0x40`, kích thước `0x9f`, cờ `AX` | `__TEXT,__text` tại 648, kích thước `0xd0`, trong segment không tên |
| Tên ký hiệu | `.symtab` (phân đoạn 10), tên trong `.strtab` qua `sh_link` 1 | `LC_SYMTAB` `symoff` 1240, `nsyms` 17, tên tại `stroff` 1512 |
| Kích thước ký hiệu | `st_size`: `percent_change` chiếm 16 byte | Không có trường này: lấy ký hiệu kế tiếp trừ đi, được 24 byte |
| Tái định vị | `.rela.text`, 3 bản ghi, `R_X86_64_PC32` | `__text` `reloff` 1104, 6 bản ghi, `PAGE21` và `PAGEOFF12` |
| Byte của hàm | `percent_change`: 16 byte tại `0x40` | `_percent_change`: 24 byte tại `0x288` |

Công cụ `rea` phân tích Mach-O thông qua `inspect_macho`, công cụ này thực thi `otool -h -l`, `nm -gjU`, `dyld_info -imports -exports`, và `dwarfdump --uuid` thông qua lệnh `xcrun`. Bộ đọc tự xây dựng của bạn không cần bất kỳ công cụ phụ trợ nào trong số đó, và nó đọc chính xác các giá trị `symoff` và `stroff` mà lệnh `otool -l` in ra. Công cụ `open_binary` của `rea` tự phát hiện các định dạng ELF, MZ, PE và Mach-O đơn lẻ hoặc đa kiến trúc (universal) từ các byte magic. Nó cũng tự đọc trường `e_machine` tại độ lệch 18. Đó chính là 20 byte đầu tiên trong những gì bạn đã xây dựng ở đây. Đối với một tệp không có tiêu đề, bạn cần cung cấp `format: raw-binary`, kiến trúc và địa chỉ điểm nhập. Trình cung cấp Ghidra sẽ từ chối một tệp nhị phân đa kiến trúc và yêu cầu trích xuất một lát cắt đơn kiến trúc. Bộ đọc của bạn cũng làm tương tự khi gặp chuỗi magic `0xcafebabe`.

## Xây Dựng Mã Nguồn (Build It)

Chạy bộ đọc thuần thư viện chuẩn:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
```

Bản demo đọc tệp `ledger-x86_64-O1.o` và in ra tiêu đề, các phân đoạn, ký hiệu, tái định vị và bản đồ độ lệch tệp. Sau đó, nó in ra chuỗi byte của 4 hàm, đọc tệp `ledger-arm64-O1.o` và in bảng so sánh. Cuối cùng, nó in ra các tiêu đề chương trình của tệp `tiny-x86_64`. Các dòng đầu ra kỳ vọng:

```text
percent_change: 16 bytes at file offset 0x40: 85 ff 74 09 29 fe 6b c6 64 99 f7 ff c3 31 c0 c3
grade: 46 bytes at file offset 0x50: 48 63 c7 48 69 c0 67 66 66 66 ...
entry 0x4000b0 is file offset 0xb0 through PT_LOAD
```

Truyền một đường dẫn để đọc một tệp cụ thể: `python3 code/main.py code/fixtures/ledger-x86_64-O0.o`. Bộ đọc tự chọn giải mã theo định dạng ELF hoặc Mach-O dựa trên 4 byte đầu tiên.

Hãy đọc `code/main.py` theo trình tự sau:

1. `ElfFile._read_header()` kiểm tra magic, class, thứ tự byte và kích thước mục, sau đó giải mã tiêu đề ELF.
2. `ElfFile._read_sections()` và `read_cstring()` đọc bảng tiêu đề phân đoạn và giải quyết tên thông qua `e_shstrndx`.
3. `ElfFile._read_symbols()` tách trường `st_info`, giải quyết tên qua `sh_link`, và giữ cho các chỉ số phân đoạn bảo lưu hiển thị rõ ràng.
4. `ElfFile._read_relocations()` duyệt qua từng phân đoạn `SHT_RELA`, tách `r_info`, và gọi tên các kiểu từ Bảng 4.9.
5. `ElfFile.function_span()` và `function_bytes()` chuyển đổi một ký hiệu thành dải tọa độ tệp và trích xuất các byte bên trong.
6. `ElfFile.file_offset_map()` sắp xếp mọi cấu trúc theo độ lệch và lấp đầy các vùng trống không hoạt động.
7. `ElfFile._read_program_headers()` và `address_to_file_offset()` xử lý các tệp thực thi.
8. `MachOFile` đọc tiêu đề, các lệnh nạp, `LC_SEGMENT_64` cùng các section bên trong, `LC_SYMTAB`, `LC_BUILD_VERSION`, và các bản ghi tái định vị trên từng section.
9. `compare_formats()` đặt tên cùng một ý niệm trên cả hai định dạng với các giá trị đọc được từ mỗi tệp.

## Sử Dụng Mã Nguồn (Use It)

Bạn có thể nhập bộ đọc vào các bài học sau bằng câu lệnh `from main import ElfFile` sau khi thêm thư mục `code` của bài học này vào `sys.path`. Bài học 33 sẽ yêu cầu hàm `function_bytes` và `relocations_for(".text")`. Bài học 35 ghi lại đầu ra của nó làm bằng chứng quan sát thực nghiệm. Bài học 36 yêu cầu nó cung cấp các độ lệch tái định vị để che đi trước khi so sánh byte.

Biên dịch mã nguồn C của riêng bạn thành một tệp đối tượng bằng cùng câu lệnh `clang -target x86_64-unknown-linux-gnu -c` và chạy bộ đọc trên tệp đó. Chỉ bổ sung tên định danh tái định vị khi Bảng 4.9 của x86-64 psABI có liệt kê nó. Chỉ bổ sung kiểu phân đoạn khi gABI hoặc chuỗi công cụ sản xuất có tài liệu quy chuẩn. Khi bộ đọc gặp một giá trị chưa có tên định nghĩa, nó sẽ in số đó dưới dạng thập lục phân thay vì suy đoán bừa bãi.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: Khám phá tiêu đề thủ công

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch/code
python3 -q
```

Chạy:

```python
import struct
from main import ElfFile, FIXTURES
data = (FIXTURES / "ledger-x86_64-O1.o").read_bytes()
data[:4], data[4], data[5]
struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
elf = ElfFile(data)
elf.header.e_shoff, elf.header.e_shnum, elf.sections[elf.header.e_shstrndx].name
```

Bộ tuple hiển thị `e_type` là 1, `e_machine` là 62, và `e_shoff` là 1064 (chính là `0x428`). Dòng cuối cùng chứng minh rằng phân đoạn 1, `.strtab`, chính là nơi lưu giữ tên của các phân đoạn.

### Lab B: Bảng giá trị phía sau lệnh switch

```python
rodata = elf.section_bytes(elf.section(".rodata"))
[int.from_bytes(rodata[i : i + 4], "little") for i in range(0x14, 0x28, 4)]
[int.from_bytes(rodata[i : i + 8], "little") for i in range(0x28, 0x50, 8)]
[r for r in elf.relocations_for(".text") if r.symbol == ".rodata"]
```

Cả hai bảng đều đọc ra dãy số `1, 2, 3, 4, 4`. Đối chiếu lại với `ledger.c`: Lời gọi `grade(65)` tính toán `65 / 10 - 6 = 0` và đọc mục 0 (giá trị 1, tương ứng với `case 6`). Lời gọi `grade(100)` đọc mục 4 (giá trị 4, là giá trị chung cho cả `case 10` và `case 9`).

### Lab C: Ba bản dựng từ một mã nguồn duy nhất

```python
for level in ("O0", "O1", "O2"):
    obj = ElfFile.from_path(FIXTURES / f"ledger-x86_64-{level}.o")
    print(level, [s.name for s in obj.sections if s.name in (".data", ".rodata")])
    print(level, sorted({r.type_name for r in obj.relocations}))
    print(level, {s.name: s.st_size for s in obj.symbols if s.type == "FUNC"})
```

Tệp đối tượng -O0 có phân đoạn `.data` và ba kiểu tái định vị. Các tệp đối tượng -O1 và -O2 không có `.data` và chỉ có một kiểu tái định vị. Hàm `total` dài 66 byte ở -O0, 31 byte ở -O1, và lên tới 140 byte ở -O2.

### Lab D: Cùng một hàm trên định dạng Mach-O

```python
from main import MachOFile
macho = MachOFile.from_path(FIXTURES / "ledger-arm64-O1.o")
[(c.name, c.cmdsize) for c in macho.load_commands]
macho.function_span("_percent_change")
macho.function_bytes("_percent_change")[:4].hex()
```

Phạm vi hàm là 24 byte tại độ lệch tệp 648. Bốn byte đầu tiên là `a0000034`, là dạng little-endian của lệnh `0x340000a0`, mà lệnh `llvm-objdump -d` dịch ra thành `cbz w0, 0x14`. Đó chính là phép kiểm tra lính canh `before == 0`, phép thử tương đương với chuỗi lệnh `85 ff 74 09` trên bản dựng x86-64.

## Bài Tập Thực Hành (Practice Lab)

Mở rộng bộ đọc cho một chương trình đã được liên kết hoàn chỉnh. Trên một máy Linux x86-64, biên dịch các lệnh `clang -O1 -shared -fPIC code/fixtures/ledger.c -o libledger.so` và `clang -O1 code/fixtures/ledger.c -o ledger_main` với một hàm `main` tự thêm vào, sau đó chạy bộ đọc trên cả hai tệp.

Các yêu cầu:

- Đọc phân đoạn `.dynsym` bằng cùng cấu trúc `Elf64_Sym`, vì `SHT_DYNSYM` mang giá trị 11 và định dạng bản ghi không thay đổi.
- Đọc phân đoạn `.dynamic` dưới dạng các bản ghi `Elf64_Dyn` gồm `d_tag` và `d_val`, và gọi đúng tên `DT_NEEDED`, `DT_STRTAB`, `DT_SYMTAB`, `DT_RELA`, và `DT_JMPREL` từ trang "Dynamic Section" của gABI.
- Liệt kê các ký hiệu chưa định nghĩa có `st_shndx` là `SHN_UNDEF` làm danh sách các hàm nhập khẩu (imports), và các bản ghi `R_X86_64_JUMP_SLOT` trong `.rela.plt` làm các điểm gọi tiếp cận chúng.
- Ánh xạ từng phân đoạn `PT_LOAD` tới các section mà nó chứa, tuân theo quy tắc gABI rằng "một phân đoạn của tệp đối tượng chứa một hoặc nhiều section".
- Chứng minh bằng hàm `address_to_file_offset` rằng các byte của mỗi hàm được định nghĩa trong thư viện chia sẻ trùng khớp hoàn toàn với các byte mà `function_bytes` trả về.
- Thu thập văn bản đầu ra của `llvm-objdump -h -t -r -p` và `llvm-objdump -T` cho cả hai tệp vào thư mục `code/fixtures/` và viết bài kiểm thử đối chiếu bộ đọc với chúng.

Thành công đồng nghĩa với việc danh sách nhập khẩu, bản đồ phân đoạn và mọi phạm vi hàm đều khớp chính xác với đầu ra của công cụ chuẩn, mà không cần chạy bất kỳ công cụ ngoài nào trong quá trình kiểm thử.

## Bài Tập Đánh Giá (Graded Lab)

Năm câu hỏi về các fixture của bài học này. Hãy tìm câu trả lời bằng chính mã nguồn bạn đã xây dựng, sau đó ghi kết quả vào `code/lab_answers.json`. Ghi số dưới dạng số nguyên hoặc chuỗi thập lục phân như `"0x40"`, và danh sách dưới dạng mảng JSON.

1. `t1`: Giá trị `e_shoff` của `ledger-x86_64-O1.o`, dưới dạng số.
2. `t2`: Giá trị `st_size` của `percent_change` trong `ledger-x86_64-O1.o`.
3. `t3`: Độ lệch tệp của phân đoạn `.text` trong `ledger-x86_64-O1.o`, dưới dạng chuỗi hex.
4. `t4`: Tên các kiểu tái định vị trong `ledger-x86_64-O0.o`, đã sắp xếp, dưới dạng danh sách chuỗi.
5. `t5`: Năm giá trị số nguyên tại `.rodata+0x14` trong `ledger-x86_64-O1.o`, dưới dạng danh sách số nguyên.

Kiểm tra câu trả lời của bạn:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/lab.py
```

Trình kiểm tra sẽ tính toán các giá trị kỳ vọng từ `code/main.py` trên các fixture đã cam kết. Nó in ra `PASS`, `FAIL` hoặc `TODO` cho từng nhiệm vụ cùng điểm số, và chỉ thoát với mã 0 khi cả 5 bài đều đạt. Lệnh `python3 code/lab.py my-answers.json` cho phép kiểm tra một tệp câu trả lời khác. Bài kiểm thử `code/tests/test_lab.py` chứng minh trình chấm chấp nhận câu trả lời đúng và từ chối câu trả lời sai.

### Trực Quan Hóa Hoạt Động

Cỗ máy tương tác [Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler) chạy bài học này như hai trạm dừng đầu tiên của nó. Cuộn băng đọc tiêu đề của tệp đối tượng chứa `percent_change`, và khoang xử lý mở `.text` và đọc ký hiệu. Hãy tắt tính năng Symbols để quan sát những dữ kiện nào thực sự do bảng ký hiệu mang lại.

## Sản Phẩm Đóng Gói (Lesson Artifact)

Bài học này tạo ra sản phẩm `outputs/skill-program-anatomy-elf-from-scratch.md`. Hãy sử dụng nó với Claude, Cursor, Codex, OpenClaw, Hermes hoặc bất kỳ agent nào có khả năng đọc tệp SKILL.md. Kỹ năng này biến một tệp đối tượng hoặc tệp thực thi do người dùng tự biên dịch thành một báo cáo bố cục chi tiết. Báo cáo cung cấp dải tọa độ tệp của từng cấu trúc, ký hiệu và bản ghi tái định vị, được đối chiếu chuẩn xác với đầu ra của công cụ tiêu chuẩn.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/32-program-anatomy-elf-from-scratch
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Bộ kiểm thử chứng minh:

- Mọi chỉ số phân đoạn, tên, kích thước, địa chỉ và phân loại đều khớp hoàn toàn với văn bản `llvm-objdump -h` thu thập từ 3 tệp đối tượng x86-64 và tệp thực thi siêu nhỏ.
- Mọi giá trị ký hiệu, ràng buộc, kiểu, phân đoạn, kích thước và tên đều khớp với văn bản `llvm-objdump -t`.
- Mọi độ lệch tái định vị, tên kiểu, ký hiệu và số hạng cộng đều khớp với văn bản `llvm-objdump -r`, bao gồm cả `R_X86_64_PLT32` và `R_X86_64_64` ở mức -O0.
- Cả hai tiêu đề chương trình của `tiny-x86_64` khớp với văn bản `llvm-objdump -p`, và các tệp đối tượng xác nhận không có tiêu đề này.
- Hàm `percent_change` chiếm đúng 16 byte tại độ lệch tệp `0x40` với các byte kỳ vọng, và mọi hàm đều nằm gọn bên trong `.text`.
- Bản đồ độ lệch tệp bao quát chính xác từng byte của mỗi fixture một lần duy nhất, với 3 khoảng trống căn chỉnh của tệp -O1 nằm đúng vị trí.
- Điểm nhập của `tiny-x86_64` ánh xạ chuẩn xác tới độ lệch tệp `0xb0` thông qua `PT_LOAD`, và tệp đối tượng từ chối việc ánh xạ địa chỉ này.
- Tiêu đề Mach-O, mọi lệnh nạp, segment, 5 section, `LC_SYMTAB` và `LC_BUILD_VERSION` đều khớp với văn bản `--private-headers`.
- Các ký hiệu Mach-O và 15 bản ghi tái định vị khớp với văn bản `llvm-objdump`, và hàm `_percent_change` chiếm đúng 24 byte tại độ lệch 648.
- Chuỗi magic sai, class 32-bit, tệp big-endian, bảng bị cắt cụt, tệp nhị phân đa kiến trúc và tệp mã nguồn C đều kích hoạt ngoại lệ `FormatError`.
- Kịch bản `build_tiny_elf.py` tái tạo chuẩn xác từng byte của tệp `tiny-x86_64` đã cam kết.
- Mô-đun hình vẽ `site/figures-re-32.js` mang chính xác các byte của tệp `ledger-x86_64-O1.o`.

## Các Chế Độ Thất Bại Trong Môi Trường Thực Tế

| Thất bại | Bài đọc yếu kém báo cáo | Điều bộ đọc phải chứng minh |
|---|---|---|
| Dùng `st_value` làm độ lệch tệp | "`percent_change` nằm tại vị trí 0" | Độ lệch chuẩn xác là `sh_offset + st_value - sh_addr`, ở đây là `0x40` |
| Đếm byte đệm căn chỉnh vào mã hàm | "`grade` dài 48 byte" | `st_size` bằng 46 và 2 byte `66 90` phía sau không thuộc về ký hiệu nào |
| Đọc tên phân đoạn từ sai bảng chuỗi | Tên rác vô nghĩa hoặc ném ngoại lệ | Trường `e_shstrndx` chỉ định chính xác bảng, ở đây là phân đoạn 1 |
| Tin tưởng `sh_size` đối với `SHT_NOBITS` | Đọc tràn vượt qua cuối tệp | `.bss` không chiếm byte nào trên đĩa và bản đồ phải bỏ qua nó |
| Đọc các byte 0 làm toán hạng thực tế | "`lea 0(%rip)`" | Bản ghi `.rela.text` tại `0x36` chỉ định đích đến là `.rodata+0x10` |
| Hiểu nhầm số hạng cộng là độ lệch đích | "bảng bắt đầu tại `0x10`" | Công thức `S + A - P` với điểm kết thúc lệnh chứng minh đích là `0x14` |
| Cố duyệt section trên tệp đã strip | "tệp không chứa mã máy" | `PT_LOAD` ánh xạ mọi địa chỉ mà bộ nạp thực sự cần |
| Nhầm lẫn địa chỉ liên kết là địa chỉ chạy | Độ lệch sai lệch khi địa chỉ cơ sở bị đổi | Độ lệch tệp từ `PT_LOAD` là tọa độ bất biến duy nhất |
| Tra cứu section Mach-O chỉ bằng `sectname` | Hai section `__const` bị gộp làm một | Định danh duy nhất bắt buộc phải là cặp `segname,sectname` |
| Kỳ vọng trường kích thước trong `nlist_64` | Ngoại lệ hoặc đọc ra 0 byte | Điểm kết thúc hàm được xác định bởi ký hiệu kế tiếp trong cùng section |
| Đọc tệp đa kiến trúc như Mach-O đơn | "magic `0xcafebabe` không xác định" | Từ chối tệp và yêu cầu trích xuất một lát cắt đơn kiến trúc |
| Đối tượng arm64 biên dịch lại trên Mac khác | "fixture kiểm thử bị biến đổi" | `LC_BUILD_VERSION` ghi lại SDK cụ thể, do đó các byte đã cam kết là chân lý |

## Quy Tắc Vận Hành Cốt Lõi

Hãy đọc tiêu đề mà bạn có thể kiểm chứng, bám sát các độ lệch mà nó lưu trữ, và định danh từng trường dựa trên tài liệu đặc tả chuẩn quy định ra nó. Hãy lưu trữ đầu ra công cụ chuẩn mà bạn dùng để đối chiếu dưới dạng fixture tĩnh đã cam kết, để quá trình kiểm tra có thể chạy độc lập mà không cần cài đặt các công cụ đó. Khi một giá trị chưa có tên chính thức trong tài liệu đặc tả, hãy in ra giá trị số thập lục phân. Một bộ đọc mà bạn có thể tự mình kiểm tra tới từng byte chính là nền móng vững chắc để mọi bài học tiếp theo xây dựng lên.

## Đọc Thêm

- [System V gABI, chương 4: Object Files](https://www.sco.com/developers/gabi/latest/ch4.intro.html) cùng các trang [ELF Header](https://www.sco.com/developers/gabi/latest/ch4.eheader.html), [Sections](https://www.sco.com/developers/gabi/latest/ch4.sheader.html), [Symbol Table](https://www.sco.com/developers/gabi/latest/ch4.symtab.html), và [Relocation](https://www.sco.com/developers/gabi/latest/ch4.reloc.html)
- [System V gABI, chương 5: Program Header](https://www.sco.com/developers/gabi/latest/ch5.pheader.html) và [Dynamic Linking](https://www.sco.com/developers/gabi/latest/ch5.dynamic.html)
- [System V ABI, AMD64 Architecture Processor Supplement](https://gitlab.com/x86-psABIs/x86-64-ABI), Bảng 4.9 Relocation Types
- [Apple XNU `loader.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h), [`nlist.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/nlist.h), và [`reloc.h`](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/reloc.h)
- [17 U.S.C. 1201](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title17-section1201&num=0&edition=prelim) và [Chỉ thị 2009/24/EC](https://eur-lex.europa.eu/eli/dir/2009/24/oj/eng)
- [Sega v. Accolade, 977 F.2d 1510](https://law.resource.org/pub/us/case/reporter/F2/977/977.F2d.1510.92-15655.html) và [Sony v. Connectix, 203 F.3d 596](https://law.resource.org/pub/us/case/reporter/F3/203/203.F3d.596.99-15852.html)
- [rea 6.5.0 `docs/binary-diagnostics.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/binary-diagnostics.md) và [`docs/ghidra-provider.md`](https://github.com/morluto/rea/blob/rea-agents-6.5.0/docs/ghidra-provider.md)
