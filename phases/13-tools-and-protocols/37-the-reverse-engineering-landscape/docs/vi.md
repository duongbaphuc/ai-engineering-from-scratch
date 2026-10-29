# Bối Cảnh Kỹ Nghệ Dịch Ngược: Lịch Sử, Các Họ Công Cụ, và Cách Một Trình Dịch Ngược Được Xây Dựng

> Năm công cụ cùng đọc một tệp đối tượng và đưa ra bốn địa chỉ, ba số lượng khối cơ bản, cùng hai câu trả lời cho một câu lệnh switch, và bài học này sẽ xây dựng bộ khung kiểm thử để chỉ rõ bản chất của từng công cụ.

**Type:** Build
**Languages:** Python, C, x86-64 assembly, Java (one Ghidra script)
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~150 phút

## Mục Tiêu Học Tập

- Định vị các công cụ chính của lĩnh vực trên dòng thời gian có ghi rõ năm tháng từ 1973 đến 2026 và gọi tên họ công cụ mà mỗi phần mềm thuộc về.
- Nêu tên tám giai đoạn của một trình dịch ngược (decompiler) cùng thuật ngữ mà Ghidra, Binary Ninja, Hex-Rays, radare2, rizin và angr sử dụng cho từng giai đoạn.
- Nâng (lift) một lệnh sang ESIL, RzIL, P-code, và VEX, đồng thời giải thích IR nào ghi cờ hiệu một cách tức thời (eagerly) và IR nào lưu trữ mẩu đệm tính toán (thunk).
- Quy đổi địa chỉ của năm công cụ về một độ lệch duy nhất trong phân đoạn `.text` và chứng minh phép quy đổi này trên một hàm cụ thể.
- Xây dựng một bộ khung so sánh (comparison harness) đọc kết quả thu thập từ các công cụ và báo cáo mức độ đồng thuận về điểm bắt đầu lệnh (instruction start), khối cơ bản (basic block), cạnh đồ thị (edge), và điểm bắt đầu hàm (function start).
- Chỉ rõ, từ bằng chứng thu thập được, nơi mỗi công cụ tự ý sinh thêm mã (invent code), bỏ sót mã (miss code), hoặc dừng lại trước một lệnh nhảy gián tiếp (indirect jump).

## Vấn Đề Cần Giải Quyết

Bạn yêu cầu một AI agent tóm tắt những gì ba công cụ kỹ nghệ dịch ngược (reverse engineering tools) báo cáo về hàm `grade` trong tệp `ledger-x86_64-O1.o`. Agent chạy từng công cụ, đọc ba bản niêm yết mã lệnh (assembly listing), và báo cáo rằng hàm này nằm tại `0x08000050` ở một công cụ, `0x400010` ở công cụ thứ hai, và `0x100010` ở công cụ thứ ba. Nó vội vã kết luận rằng các công cụ đang bất đồng về vị trí thực tế của hàm.

Trên thực tế, các công cụ hoàn toàn đồng thuận. Tệp đối tượng có thể tái định vị (relocatable object) không có địa chỉ nạp cố định sẵn trong tiêu đề, do đó mỗi công cụ tự chọn một địa chỉ cơ sở riêng. Nếu trừ đi địa chỉ cơ sở mà mỗi công cụ gán cho phân đoạn `.text`, mọi kết quả đều trả về chính xác độ lệch `0x10`. Agent đã so sánh các địa chỉ thô vì không có quy tắc nào cho nó biết ý nghĩa địa chỉ trong từng công cụ.

Sau đó, chính agent này đọc hàm `grade` trong bản dựng không tối ưu `-O0` và báo cáo 3, 8, 8, và 9 khối cơ bản từ bốn công cụ. Cả bốn con số này đều có thật trong kết quả phân tích: Một công cụ dừng lại ngay tại lệnh `jmp rax`. Hai công cụ đọc bảng nhảy (jump table) thông qua các bản ghi tái định vị của nó. Một công cụ tách khối tại lệnh nhảy để đặt một nhãn điều khiển. Một số lượng khối cơ bản mà không đi kèm nguyên nhân đằng sau chỉ là một con số vô nghĩa không có giá trị khẳng định.

Bài học này chính là tấm bản đồ bạn cần nắm vững trước khi đặt niềm tin vào bất kỳ con số nào. Các bài học từ 32 đến 36 đã tự tay xây dựng một bộ giải mã (decoder), một bộ nâng mã (lifter), một sổ cái bằng chứng (evidence record), và một quy trình dựng lại chứng minh tương đương. Nửa sau của lộ trình sẽ đặt những gì chúng ta tự làm bên cạnh các công cụ tiêu chuẩn mà thế giới đang sử dụng. Để làm được điều đó, bạn cần hiểu rõ lịch sử phát triển của lĩnh vực, các họ công cụ cấu thành, quy trình đường ống (pipeline) mà mọi trình dịch ngược đều vận hành, và một bộ khung kiểm thử có khả năng so sánh phát hiện của các công cụ trên những tệp đối tượng do chính bạn biên dịch.

```figure
re-37-field-map
```

## Năm Mươi Ba Năm Qua Các Dấu Mốc Lịch Sử

Lĩnh vực này lâu đời hơn phần lớn các công cụ đang hiện hành. Mỗi mốc thời gian dưới đây đều có tài liệu nguồn gốc được trích dẫn trong phần Đọc Thêm.

### 1973 đến 1991: các trình gỡ lỗi xuất hiện trước, sau đó đến các trình phân dịch

Những luận án đầu tiên về dịch ngược xuất hiện từ năm 1973. Hollander tại Đại học Stanford và Housel tại Đại học Purdue đã công bố các công trình này, và ngày nay cả hai đều được biết đến qua các tài liệu trích dẫn kinh điển. Những công cụ thực tế của thời kỳ này chủ yếu là các trình gỡ lỗi (debuggers). Maranzano và Bourne đã viết tài liệu hướng dẫn sử dụng `adb` tại Bell Labs vào năm 1977, và trình gỡ lỗi này được phát hành chính thức cùng UNIX V7 vào năm 1979. Công cụ `DEBUG` của Tim Paterson, đi kèm một bộ phân dịch (disassembler) cho vi xử lý 8086, được tích hợp trong PC-DOS 1.00 vào tháng 8 năm 1981. GDB của Richard Stallman bắt đầu được phân phối vào năm 1986. SoftICE đã đưa một trình gỡ lỗi nhân (kernel debugger) trực tiếp vào môi trường DOS vào năm 1987.

Các trình phân dịch xử lý theo lô (batch disassemblers) xuất hiện tiếp theo. Sourcer, do V Communications phát hành năm 1988, thực hiện nhiều lượt quét qua chương trình DOS và xuất ra một bản niêm yết mã kèm chú thích chi tiết. Dự án GNU binutils bắt đầu từ các bản chụp chưa đánh số vào năm 1988, và lệnh `objdump` mang bản quyền từ năm 1990. Ilfak Guilfanov đã viết dòng mã đầu tiên của IDA vào tháng 1 năm 1991 và phát hành phiên bản IDA 0.1 dưới dạng shareware vào ngày 20 tháng 5 năm 1991.

### 1991 đến 2009: các trình phân dịch tương tác và những trình dịch ngược đầu tiên

Cristina Cifuentes hoàn thành luận án tiến sĩ "Reverse Compilation Techniques" tại Đại học Công nghệ Queensland (QUT) vào tháng 7 năm 1994. Trình dịch ngược `dcc` của bà đã dịch thành công các chương trình DOS i80286 sang ngôn ngữ C, và bài báo năm 1995 của bà cùng Gough đã xác lập các giai đoạn phân tích mà mọi trình dịch ngược sau này đều kế thừa. Tiếp sau đó là REC vào năm 1997 và Boomerang vào năm 2002 — trình dịch ngược mã nguồn mở đầu tiên được phát hành theo giấy phép BSD.

IDA trở thành sản phẩm thương mại tại DataRescue vào năm 1996, bổ sung bộ SDK phát triển plugin vào năm 1999, và chuyển giao cho Hex-Rays (thành lập năm 2005). Trình dịch ngược Hex-Rays Decompiler 1.0 chính thức phát hành ngày 17 tháng 9 năm 2007. Cùng năm đó, Mike Van Emmerik bảo vệ luận án "Static Single Assignment for Decompilation", đặt dạng gán đơn tĩnh (Static Single Assignment - SSA) vào vị trí trung tâm của việc phân tích luồng dữ liệu trong trình dịch ngược. Bài báo "Structural Comparison of Executable Objects" của Halvar Flake (2004) đã khai sinh họ công cụ so sánh sai biệt cấu trúc (diffing), và BinDiff chính thức được thương mại hóa vào tháng 8 năm 2004.

radare khởi đầu vào tháng 2 năm 2006 dưới dạng một trình chỉnh sửa mã hex hỗ trợ điều tra kỹ thuật số với độ lệch 64-bit. radare2, bản viết lại theo kiến trúc module hóa theo giấy phép LGPL-3.0, phát hành phiên bản đầu tiên vào ngày 7 tháng 3 năm 2009.

### 2009 đến 2019: các nền tảng phân tích và biểu diễn trung gian

LLDB được công bố là một tiểu dự án của LLVM vào ngày 8 tháng 6 năm 2010. BAP (Binary Analysis Platform) cùng ngôn ngữ biểu diễn trung gian BIL xuất hiện tại hội nghị CAV 2011. Google mua lại zynamics (nhà phát triển BinDiff) vào ngày 1 tháng 3 năm 2011. Hopper xuất hiện trên Mac App Store vào cuối năm 2011. Matt Godbolt lập ra Compiler Explorer vào năm 2012 để các lập trình viên C++ có thể quan sát trực tiếp mã máy mà trình biên dịch tạo ra từ mã nguồn.

angr đón nhận dòng mã đầu tiên vào tháng 8 năm 2013. Bài báo về Phoenix (Schwartz, Lee, Woo, Brumley, USENIX Security 2013) đã chứng minh kỹ thuật cấu trúc hóa bảo toàn ngữ nghĩa chỉ sinh ra 40 lệnh `goto` trên tổng số 8.676 hàm được thử nghiệm. DREAM (Yakdan và cộng sự, NDSS 2015) tiếp tục loại bỏ hoàn toàn các câu lệnh `goto` bằng phương pháp cấu trúc hóa không phụ thuộc mẫu (pattern-independent structuring). Công ty Vector 35 được thành lập vào ngày 14 tháng 1 năm 2015, và Binary Ninja chính thức ra mắt vào ngày 23 tháng 7 năm 2016 với ngôn ngữ trung gian Low Level IL cùng giao diện lập trình Python. angr được phát hành công khai vào năm 2015 và giành vị trí thứ ba tại cuộc thi DARPA Cyber Grand Challenge vào tháng 8 năm 2016. BinDiff chuyển sang phát hành miễn phí vào tháng 3 năm 2016. Cutter 1.0 cung cấp giao diện đồ họa Qt cho radare2 vào tháng 12 năm 2017. Năm 2018, Guilfanov trình bày bài giảng "Decompiler internals: microcode" tại hội nghị Recon, hé lộ công khai lần đầu tiên về biểu diễn trung gian microcode của Hex-Rays.

### 2019 đến 2026: các trình dịch ngược mở và bước ngoặt nghiên cứu

Ghidra 9.0 được Cơ quan An ninh Quốc gia Hoa Kỳ (NSA) phát hành công khai tại Hội nghị RSA vào ngày 5 tháng 3 năm 2019 theo giấy phép Apache-2.0, và toàn bộ mã nguồn được đưa lên GitHub một tháng sau đó. angr 8.19.2.4 bổ sung trình dịch ngược sơ khởi vào cùng năm. Binary Ninja 2.0 ra mắt bộ dịch ngược High Level IL vào ngày 20 tháng 5 năm 2020. Rizin được công bố là nhánh tách rời (fork) từ radare2 vào ngày 8 tháng 12 năm 2020, và đội ngũ phát triển Cutter chuyển sang sử dụng Rizin. IDA Freeware 7.6 bổ sung trình dịch ngược đám mây cho kiến trúc x64 vào tháng 5 năm 2021.

BinDiff 8 được mở mã nguồn hoàn toàn theo giấy phép Apache-2.0 vào ngày 25 tháng 9 năm 2023. SAILR (Basque và cộng sự, USENIX Security 2024) trở thành thuật toán cấu trúc hóa mặc định của angr và lập luận rằng chỉ nên loại bỏ các câu lệnh `goto` giả tạo (spurious gotos), bởi mã nguồn nhân Linux 6.1 vốn dĩ đã chứa sẵn 3.754 câu lệnh `goto` có chủ đích. Binary Ninja 4.0 bổ sung phiên bản Free vào tháng 2 năm 2024, và IDA 9.0 ra mắt `idalib` cho việc phân tích tự động không cần giao diện đồ họa. Các công trình nghiên cứu về trình dịch ngược học máy (learned decompilers) phát triển mạnh mẽ từ SLaDe (2023), LLM4Decompile (2024), DecompileBench (2025) cho tới hai nghiên cứu năm 2026 về độ dễ đọc có sự trợ giúp của AI agent và về hiện tượng các mô hình tái biên dịch được nhiều hơn nhưng lại bảo toàn ngữ nghĩa ít hơn.

Các phiên bản được vận hành trong bài học này là các bản phát hành chuẩn vào ngày 14 tháng 10 năm 2026: Ghidra 12.1.4 (21 tháng 9 năm 2026), radare2 6.2.4 (5 tháng 10 năm 2026), rizin 0.9.1 (29 tháng 6 năm 2026), và angr 10.0.3 (10 tháng 10 năm 2026). Binary Ninja 6.0 "Krypton" (3 tháng 9 năm 2026) và IDA 9.4 (13 tháng 7 năm 2026) là các công cụ thương mại nguồn đóng hiện hành. Bài học này không trực tiếp thực thi hai công cụ đó mà trích xuất đặc tính từ tài liệu kỹ thuật của chúng.

## Các Họ Công Cụ

Một công cụ thuộc về một họ nhất định dựa trên sản phẩm đầu ra mà nó tạo ra. Các họ này thường giao thoa bên trong cùng một chương trình phần mềm, và sự giao thoa đó là nguồn gốc của hầu hết mọi sự nhầm lẫn.

### Bộ phân dịch (Disassemblers)

Phương pháp quét tuyến tính (linear sweep) giải mã tuần tự từ byte đầu tiên của một phân đoạn mã cho đến byte cuối cùng. Lệnh `objdump -d` của GNU và `llvm-objdump -d` hoạt động theo nguyên lý này, và không công cụ nào trong số chúng xây dựng đồ thị luồng điều khiển (control-flow graph - CFG). GNU `objdump` coi các ký hiệu (symbols) là ranh giới lệnh và quét qua các khoảng giữa chúng. binutils 2.47 (26 tháng 7 năm 2026) mang giấy phép GPL-3.0-or-later. LLVM 23.1.3 sử dụng giấy phép Apache-2.0 kèm ngoại lệ LLVM. Bộ công cụ dòng lệnh của Apple (Command Line Tools) phân phối `llvm-objdump` dưới nhãn "Apple LLVM version 17.0.0", và đó chính là bản sao mà bài học này thu thập dữ liệu.

Một trình phân dịch tương tác (interactive disassembler) thực hiện duyệt đệ quy (recursive descent) từ các điểm vào (entry points) đã biết, bổ sung các phương pháp phỏng đoán để dò tìm điểm bắt đầu của hàm, và lưu trữ dữ liệu trong một cơ sở dữ liệu phân tích. IDA Pro (với các module vi xử lý, hàng đợi tự động phân tích, và chữ ký nhận diện hàm FLIRT), Ghidra (lớp `Disassembler` kết hợp các bộ phân tích Auto Analysis), Binary Ninja (duyệt đệ quy, sau đó quét tuyến tính qua các vùng chưa xác định), Hopper, radare2 và rizin (`aa`, `aaa`, `af`, với Cutter và iaito làm giao diện đồ họa) đều thuộc họ này. Andriesse và cộng sự (USENIX Security 2016) đã tiến hành đo lường 9 công cụ trên các tệp nhị phân do trình biên dịch sinh ra: Độ chính xác phân dịch lệnh đạt gần 100%, nhưng việc xác định điểm bắt đầu của hàm lại là nguyên thủy hóc búa nhất với tỷ lệ bỏ sót âm tính giả (false negative) lên tới 20% hoặc hơn.

### Bộ dịch ngược (Decompilers)

Các trình dịch ngược thương mại bao gồm: Hex-Rays Decompiler (hỗ trợ 12 họ kiến trúc từ x86 đến RISC-V), Binary Ninja HLIL với chế độ hiển thị Pseudo C, và khung nhìn mã giả của Hopper. Các trình dịch ngược mở gồm: Ghidra decompiler — một động cơ C++ theo giấy phép Apache-2.0 được r2ghidra và rz-ghidra tái sử dụng dưới dạng plugin và được pypcode sử dụng làm bộ nâng mã; trình dịch ngược angr dựa trên AIL với bộ cấu trúc hóa SAILR theo giấy phép BSD-2-Clause; r2dec và jsdec viết bằng JavaScript; cùng Boomerang và `dcc` đóng vai trò lịch sử. radare2 cũng tích hợp sẵn lệnh `pdc` ("pseudo disassembler output in C-like syntax"). rizin không có trình dịch ngược tích hợp: Lệnh `pdc` không tồn tại trong rizin 0.9.1, và lệnh `pdg` đòi hỏi phải cài đặt plugin rz-ghidra.

### Bộ gỡ lỗi (Debuggers)

GDB (1986) cùng các giao diện mở rộng GEF và pwndbg, LLDB (2010), WinDbg với tính năng Time Travel Debugging (2017), OllyDbg (dừng phát triển từ 2014), x64dbg (GPL-3.0, 2014), Ghidra Debugger với các cầu nối Python, Binary Ninja debugger, cùng các trình gỡ lỗi tích hợp sẵn trong radare2 và rizin. Bài học 42 sẽ sử dụng hai trong số các công cụ này trên chính các bản dựng của bạn.

### Các khung phân tích nhị phân (Binary analysis frameworks)

angr (ngôn ngữ trung gian VEX IR, thực thi biểu trưng, thuật toán `CFGFast` và `CFGEmulated`, trình dịch ngược), BAP (ngôn ngữ BIL, viết bằng OCaml), Dyninst, Jakstab, và ByteWeight là các khung phân tích tiêu biểu được Andriesse liệt kê. Ghidra có thể dùng như một thư viện thông qua PyGhidra, Binary Ninja thông qua gói `import binaryninja`, và IDA thông qua `idalib`.

### Thư viện định dạng tệp và động cơ phân dịch (Format and engine libraries)

LIEF 1.0.0 (12 tháng 7 năm 2026, Apache-2.0) phân tích các định dạng ELF, PE, Mach-O, COFF và Android. CLE ("CLE Loads Everything") là bộ nạp của angr. Capstone 5.0.9 (28 tháng 5 năm 2026, BSD-3-Clause) đóng vai trò giải mã; Keystone 0.9.2 (GPL-2.0) làm nhiệm vụ hợp dịch; và Unicorn 2.1.4 (GPL-2.0) mô phỏng thực thi CPU. radare2 6.2.4 đóng gói sẵn Capstone 5, trong khi rizin 0.9.1 tích hợp trực tiếp Capstone, Zydis, và tree-sitter.

### Công cụ so sánh sai biệt và bộ hiển thị phía trình biên dịch (Diffing tools and compiler-side viewers)

BinDiff 8 (Apache-2.0) cùng plugin BinExport 12 dành cho IDA Pro, Binary Ninja, và Ghidra; Diaphora 3.4.3 (11 tháng 10 năm 2026, AGPL-3.0) dưới dạng plugin cho IDA; cùng các công cụ dòng lệnh `radiff2` và `rz-diff`. Compiler Explorer (BSD-2-Clause) hiển thị mã assembly mà trình biên dịch tạo ra từ một đoạn mã nguồn — đây chính là chiều xuôi của mọi thao tác phân tích mà danh sách trên thực hiện. Bài học 44 sẽ đi sâu vào họ công cụ so sánh sai biệt này.

### Tổng hợp phiên bản, giấy phép và các công cụ được thực thi

| Công cụ | Phiên bản (14/10/2026) | Giấy phép | Phương thức gọi headless | Thực thi trong bài |
|---|---|---|---|---|
| Ghidra | 12.1.4 | Apache-2.0 | `analyzeHeadless`, `pyghidra.start()` | có, trên cả 4 đối tượng |
| radare2 | 6.2.4 | LGPL-3.0 | `r2 -qc '<cmds>' file`, `rabin2` | có |
| rizin | 0.9.1 | LGPL-3.0 | `rizin -qc '<cmds>' file`, `rz-bin` | có |
| angr | 10.0.3 | BSD-2-Clause | `angr.Project` trong Python | có |
| llvm-objdump | Apple LLVM 17.0.0 | Apache-2.0 kèm ngoại lệ LLVM | `llvm-objdump -d -r` | có |
| Binary Ninja | 6.0 "Krypton" | Thương mại; bản Free cho phi thương mại | `binaryninja.load()` (bản Commercial / Ultimate / headless) | không, chỉ dựa vào tài liệu |
| IDA Pro | 9.4 | Thuê bao thương mại; IDA Free cho x86/x64 phi thương mại | `idat`, `ida -A -S`, `idalib` | không, chỉ dựa vào tài liệu |
| Hopper | 6.6.0 | Thương mại, bản quyền vĩnh viễn | không có tài liệu công khai | không, chỉ dựa vào tài liệu |
| Cutter | 2.5.0 | GPL-3.0 | Giao diện GUI trên nền rizin | không |

Phiên bản Free của Binary Ninja chỉ dịch ngược được x86, x86-64, ARMv7, và AArch64, đồng thời không cung cấp API. IDA Free không hỗ trợ IDAPython, không có SDK, không có `idalib`, và trình dịch ngược của nó chạy trên máy chủ đám mây. Mọi nhận định về hai công cụ này trong bài học đều bắt nguồn trực tiếp từ tài liệu kỹ thuật và ghi chú phát hành chính thức của nhà phát triển, và sơ đồ minh họa đánh dấu rõ chúng không được chạy thực tế.

## Quy Trình Pipeline Của Trình Dịch Ngược Qua Từng Giai Đoạn

Cifuentes đã đặt tên cho các giai đoạn phân tích từ năm 1994: phân tích cú pháp (syntax analysis), phân tích ngữ nghĩa (semantic analysis), sinh mã trung gian (intermediate code generation), tạo đồ thị luồng điều khiển (control-flow graph generation), phân tích luồng dữ liệu (data-flow analysis), phân tích luồng điều khiển (control-flow analysis), và phát sinh mã nguồn (code generation). Bà chia chúng thành ba khối: khối đầu (front end) phụ thuộc máy, khối giữa (middle) độc lập với cả máy lẫn ngôn ngữ, và khối cuối (back end) phụ thuộc ngôn ngữ đích. Mọi trình dịch ngược kể từ đó đều giữ nguyên cấu trúc khung này và chỉ thay đổi tên gọi các bộ phận cấu thành. Bảng dưới đây thể hiện tám giai đoạn mà bài học này quy chuẩn cùng thuật ngữ tương ứng trong từng công cụ:

| Giai đoạn | Ghidra | Binary Ninja | Hex-Rays / IDA | radare2 | rizin | angr |
|---|---|---|---|---|---|---|
| Bộ nạp (Loader) | `ElfLoader`, `MachoLoader`, `PeLoader`; `.ldefs` chọn ngôn ngữ | `BinaryView`, `BinaryViewType` | module nạp (`LDSC`) | RBin (`rabin2`) | RzBin (`rz-bin`) | CLE (`cle.Loader`, `main_object`) |
| Phân dịch (Disassembly) | `Disassembler`; SLEIGH `.slaspec` biên dịch ra `.sla`; các bộ phân tích "Disassemble Entry Points", "Function Start Search" | plugin kiến trúc; Duyệt đệ quy (Recursive Descent), sau đó quét tuyến tính | module vi xử lý (`ana.cpp`, `emu.cpp`, `out.cpp`); hàng đợi tự động; FLIRT | plugin RArch xuất sang `RAnalOp`; `aa`, `aaa`, `af` | plugin RzAnalysis; `aa`, `aaa`, `af` | Khối lệnh Capstone; `CFGFast` tìm điểm bắt đầu hàm |
| Khôi phục CFG (CFG recovery) | "Generate Basic Blocks and the CFG"; `BlockGraph`; "Decompiler Switch Analysis" | `Function`, `BasicBlock`; giai đoạn "Control Flow Graph (CFG) Analysis" | `mbl_graph_t`, sẵn sàng tại mức `MMAT_LOCOPT` | `afb`, `agf`, `aab` | `afb`, `agf`, `aab` (Nucleus) | `CFGFast`, `CFGEmulated`; `kb.functions`; IRSB |
| Nâng mã (Lifting) | "Generate Raw P-code"; Các Varnode trong không gian `ram`, `register`, `const`, `unique` | Lifted IL, LLIL, Mapped MLIL, MLIL | microcode (`minsn_t`, `mop_t`); mức `MMAT_GENERATED` | ESIL (`asm.esil`, `ae`) | RzIL (`aoi`, `aez`) | VEX (`pyvex.lift`, IRSB), sau đó là AIL |
| Dạng SSA (SSA) | `Heritage`, `ActionHeritage`; toán tử `MULTIEQUAL`, `INDIRECT` | LLIL SSA, MLIL SSA, HLIL SSA; `HLIL_VAR_PHI` | không đặt tên SSA riêng; luồng dữ liệu use-def tại `MMAT_GLBOPT1` đến `3` | không có; ESIL được thiết kế cho mô phỏng | không có; RzIL được thiết kế cho mô phỏng | `_stage_transform_to_ssa_level0` và `level1`; `SeqNodeDephication` |
| Luồng dữ liệu và kiểu (Data flow & types) | `ActionDeadCode`, `ActionInferTypes`, tập quy tắc `Rule`; nguyên mẫu hàm từ `.cspec` | Lan truyền hằng số trong MLIL, biến và kiểu; `analyzeReturns` | "Transform microcode"; `MMAT_CALLS`, `MMAT_GLBOPT`, `MMAT_LVARS`; thư viện TIL | `afv`, `afc`, `aaft`; mô phỏng bằng ESIL | `afv`, `afc`, `aft` | `CallingConventionAnalysis`, `VariableRecoveryFast`, `Typehoon` |
| Cấu trúc hóa (Structuring) | `CollapseStructure`; `BlockIf`, `BlockWhileDo`, `BlockSwitch`, `BlockGoto` | `HLIL_IF`, `HLIL_WHILE`, `HLIL_SWITCH`; cây AST tại `root` | "Generate ctree"; phân tích cấu trúc; các vùng goto | `pdc`; `pdd` (r2dec); `pdg` (r2ghidra) | `pdg` (rz-ghidra); jsdec | `RegionIdentifier`; `RecursiveStructurer` cùng `SAILRStructurer` |
| Phát sinh mã (Emission) | "Emit Final C Tokens"; `PrintC`, `ActionNameVars`, `ActionSetCasts` | Kết xuất HLIL; Pseudo C, Pseudo Rust, Pseudo Python | "Beautify ctree; Print ctree"; cấu trúc `cfunc_t` | văn bản xuất ra từ `pdc`, `pdd`, `pdg` | văn bản xuất ra từ `pdg` | `CStructuredCodeGenerator`; thuộc tính `codegen.text` |

### Bộ nạp (Loader)

Bộ nạp phân tích cấu trúc vỏ bọc (container), ánh xạ các phân đoạn (sections/segments) vào bộ nhớ, đồng thời đọc bảng ký hiệu và các bản ghi tái định vị (relocations). Trong một tệp đối tượng có thể tái định vị, chưa có thực thể nào sở hữu địa chỉ bộ nhớ cố định, vì vậy bộ nạp phải tự chọn một địa chỉ cơ sở. radare2 và rizin ánh xạ tệp tại `0x08000000`, dẫn đến phân đoạn `.text` nằm tại `0x08000040` vì phân đoạn này bắt đầu tại độ lệch tệp `0x40`. CLE đặt `.text` tại `0x400000`. `ElfLoader` của Ghidra đặt nó tại `0x100000`. Ngược lại, `llvm-objdump` chỉ in các độ lệch phân đoạn và không áp dụng bất kỳ địa chỉ cơ sở nào.

Các bộ nạp cũng xử lý các bản ghi tái định vị theo cách rất khác nhau. Ghidra xử lý tái định vị ngay khi nhập tệp, và CLE áp dụng chúng trực tiếp vào bộ nhớ nạp. radare2 in một cảnh báo khi mở tệp: các bản ghi tái định vị chưa được áp dụng, và thiết lập `bin.relocs.apply=true` sẽ thay đổi điều đó. Cách bộ nạp xử lý tái định vị quyết định trực tiếp những gì các giai đoạn sau có thể nhìn thấy, và bạn sẽ thấy rõ tác động của nó trong bảng nhảy của bản dựng `-O0` dưới đây.

### Phân dịch (Disassembly)

Bài học 33 đã tự xây dựng cả hai phương pháp phân dịch. Mọi công cụ trong bảng trên (ngoại trừ `llvm-objdump`) đều thực hiện duyệt đệ quy và bổ sung thêm thuật toán tìm kiếm điểm bắt đầu hàm. Tài liệu của Binary Ninja liệt kê thứ tự các giai đoạn: Duyệt đệ quy (Recursive Descent), Phân tích đích gọi hàm (Call Target Analysis), Phân tích đồ thị CFG dưới dạng "phân dịch tuyến tính các vùng chưa xác định", và Phân tích lời gọi đuôi (Tail Call Analysis). Ghidra chạy các bộ phân tích theo mức độ ưu tiên, trong đó "Disassemble Entry Points" và "Function Start Search" diễn ra rất sớm. IDA duy trì các hàng đợi tự động phân tích theo trọng số ưu tiên, kết hợp chữ ký FLIRT cho mã thư viện. Thuật toán `CFGFast` của angr ghi nhận bốn nguồn xác định điểm bắt đầu hàm: các ký hiệu, quét mẫu mở đầu hàm (prologue scan), các đích của lệnh gọi hàm, và cuối cùng là quét toàn bộ các byte còn lại. Chính nguồn tìm kiếm cuối cùng này là nơi angr tự ý sinh ra các hàm phụ tại những vùng đệm căn chỉnh (padding).

### Khôi phục CFG (CFG recovery)

Mỗi công cụ phân cắt các khối cơ bản tương tự như cách chúng ta đã làm trong bài học 33, với một điểm khác biệt then chốt bạn sẽ đo lường được: angr ngắt một khối cơ bản ngay tại lệnh gọi hàm `call`, bởi cấu trúc IRSB của nó là một khối đơn điểm vào nhưng có thể đa điểm ra, và lệnh `call` được coi là một điểm ra. Ngược lại, Ghidra, radare2, và rizin xem lệnh `call` là một lệnh tuần tự và tiếp tục khối lệnh đi qua nó. Trên hàm `grade_label` ở bản dựng `-O0` (vốn có một lệnh gọi tới `grade`), angr báo cáo 2 khối cơ bản trong khi ba công cụ còn lại báo cáo 1 khối. Các lệnh máy bên trong hoàn toàn giống hệt nhau.

### Nâng mã (Lifting)

Binary Ninja định nghĩa việc nâng mã là "chuyển dịch từ một ngôn ngữ cấp thấp hơn sang một ngôn ngữ trung gian được coi là cấp cao hơn". Các dạng IR được thiết kế cho các mục đích hoàn toàn khác biệt. P-code và VEX được xây dựng phục vụ phân tích tĩnh và dịch ngược mã nguồn. Ngược lại, ESIL và RzIL được tạo ra cho mục đích mô phỏng thực thi (emulation), và không dạng nào trong hai dạng này sở hữu giai đoạn biến đổi SSA. Microcode của Hex-Rays bắt đầu như một tập lệnh dạng RISC với 72 mã thao tác (opcodes) và đi qua các mức trưởng thành từ `MMAT_GENERATED` đến `MMAT_LVARS`. Binary Ninja nâng mã lên Lifted IL, sau đó gộp các cờ điều kiện vào các biểu thức trong LLIL, và MLIL tiếp tục "chuyển dịch các thanh ghi và truy cập bộ nhớ thành các biến định danh". Phần tiếp theo sẽ đặt bốn dạng IR này cạnh nhau trên cùng một lệnh máy.

### Dạng SSA (SSA)

Luận án của Van Emmerik đã tổng kết những lợi ích to lớn mà SSA mang lại cho một trình dịch ngược: lan truyền biểu thức (expression propagation), nhận diện tham số và giá trị trả về mà không cần giả định trước về ABI, phân tích tính bảo toàn thanh ghi, suy luận kiểu dữ liệu dưới dạng bài toán luồng dữ liệu thưa, cùng khả năng phân tích các lệnh nhảy và gọi hàm gián tiếp. Ghidra gọi quá trình xây dựng SSA của mình là `Heritage` và bổ sung hai toán tử đặc thù chỉ tồn tại trong đồ thị: `MULTIEQUAL` (đóng vai trò hàm phi $\phi$) và `INDIRECT` (mô hình hóa các tác dụng phụ lên bộ nhớ). Binary Ninja cung cấp dạng biểu diễn SSA ở mọi cấp độ IL. angr triển khai hai giai đoạn SSA bên trong module `Clinic`. Hex-Rays không dùng tên gọi SSA riêng biệt mà thực hiện phân tích luồng dữ liệu use-def trong các lượt tối ưu hóa toàn cục. ESIL và RzIL hoàn toàn không có dạng SSA.

### Luồng dữ liệu và kiểu (Data flow and types)

Vòng lặp đơn giản hóa chính của Ghidra thực thi các bước "Eliminate Dead Code", "Propagate Local Types", và "Perform Term Rewriting" cho đến khi trạng thái đồ thị hội tụ không đổi, trong đó việc viết lại biểu thức được thực hiện bởi một nhóm các đối tượng quy tắc `Rule`. Hex-Rays biến đổi microcode, nhận diện các đối số của lời gọi hàm tại `MMAT_CALLS`, và cấp phát các biến cục bộ tại `MMAT_LVARS`. angr chạy các lượt `CallingConventionAnalysis`, `VariableRecoveryFast`, và `Typehoon` — một cài đặt của thuật toán suy luận kiểu đa hình từ Noonan, Loginov, và Cok. Bài học 41 sẽ dành trọn vẹn nội dung cho giai đoạn này.

### Cấu trúc hóa (Structuring)

Lớp `CollapseStructure` của Ghidra tìm kiếm các đồ thị con khớp với các mẫu cấu trúc điều khiển chuẩn, thu gọn chúng thành một nút duy nhất, và khi gặp bế tắc, nó chủ động loại bỏ một cạnh và đánh dấu cạnh đó là phi cấu trúc (unstructured). Hex-Rays rút gọn đồ thị CFG thành các cấu trúc `if`, `while`, `do`, và `switch`, đồng thời giữ lại phần còn lại dưới dạng các vùng goto. angr nhận diện các vùng đơn điểm vào - đơn điểm ra (single-entry, single-exit regions) và cấu trúc hóa chúng bằng `SAILRStructurer` theo mặc định, với `PhoenixStructurer` và `DreamStructurer` làm các phương án thay thế. Bài học 34 đã tự tay lập trình một bộ cấu trúc hóa tương tự.

### Phát sinh mã (Emission)

Ghidra phát sinh các thẻ cú pháp C thông qua lớp `PrintC` kết hợp bộ định dạng Oppen sau khi đã hoàn thành `ActionNameVars` và `ActionSetCasts`. Hex-Rays làm đẹp và in cây cú pháp ctree. Binary Ninja kết xuất HLIL thành Pseudo C, Pseudo Rust, hoặc Pseudo Python. angr trả về chuỗi văn bản tại `codegen.text`. radare2 in ra văn bản từ các lệnh `pdc`, `pdd`, hoặc `pdg`.

## Một Lệnh Máy Qua Bốn Dạng IR

Xét chuỗi byte `48 83 ec 10`, tương ứng với lệnh `subq $0x10, %rsp`. Hãy giải mã nó trên giấy theo các quy tắc từ bài học 33:

1. `48` là tiền tố REX với bit `W = 1`. Toán hạng có độ rộng 64 bit.
2. `83` là mã thao tác số học nhóm 1 (group-1 arithmetic opcode) đi kèm toán hạng tức thời 8-bit. Trường `reg` trong byte ModRM sẽ quyết định thao tác cụ thể.
3. `ec` là ModRM `11 101 100`: `mod = 11` chỉ định dạng thanh ghi; `reg = 101` tương ứng `/5`, chính là phép trừ `sub`; `r/m = 100` chỉ định thanh ghi `%rsp`.
4. `10` là giá trị tức thời, 16 trong hệ thập phân.

Bốn bộ nâng mã đã xử lý 4 byte đó. Thư mục fixture lưu trữ chính xác các lệnh gọi và toàn bộ chuỗi văn bản kết quả.

Dạng ESIL từ radare2 6.2.4 là một chuỗi duy nhất gồm 30 phần tử phân tách bởi dấu phẩy:

```text
16,0x8000000000000000,-,!,16,rsp,-=,63,$o,^,of,:=,63,$s,sf,:=,$z,zf,:=,$p,pf,:=,64,$b,cf,:=,4,$b,af,:=
```

Đây là một máy ngăn xếp (stack machine). Chuỗi `16,rsp,-=` thực hiện trừ trực tiếp tại chỗ. Mọi cờ điều kiện sau đó được ghi thông qua các toán tử nội bộ: `$o` cho cờ tràn (overflow), `$s` cho cờ dấu (sign), `$z` cho cờ zero, `$p` cho cờ chẵn lẻ (parity), `$b` cho cờ mượn (borrow).

Dạng RzIL từ rizin 0.9.1 là một biểu thức dạng S-expression với 14 hiệu ứng gán `set`:

```text
(seq (set op1 (var rsp)) (set op2 (bv 64 0x10)) (set sub (- (var op1) (var op2))) (set rsp (var sub))
     (set _result (var sub)) (set pf (! (lsb (let _val (cast 8 false (var _result)) ...))))
     (set zf (is_zero (var _result))) (set sf (msb (var _result))) ... (set cf (|| ...)) (set of (|| ...)) (set af (|| ...)))
```

Mỗi cờ điều kiện là một biểu thức logic boolean trên các vectơ bit (bitvectors). Riêng cờ chẵn lẻ `pf` là một cấu trúc `let` lồng nhau tới bốn tầng.

Dạng P-code từ đặc tả SLEIGH `x86:LE:64:default` thông qua thư viện pypcode 4.0.1 gồm 9 phép toán sau thẻ `IMARK`:

```text
CF = RSP < 0x10
OF = sborrow(RSP, 0x10)
RSP = RSP - 0x10
SF = RSP s< 0x0
ZF = RSP == 0x0
unique[58300:8] = RSP & 0xff
unique[58400:1] = popcount(unique[58300:8])
unique[58500:1] = unique[58400:1] & 0x1
PF = unique[58500:1] == 0x0
```

Các phép chuyển dịch thanh ghi sử dụng các biến tạm trong không gian `unique`. Các cờ được tính toán một cách tức thời (eagerly) ngay tại lệnh: `CF` và `OF` được tính trước khi phép trừ ghi đè lên thanh ghi `RSP`.

Dạng VEX từ pyvex 10.0.3 gồm 7 câu lệnh sau thẻ `IMark`:

```text
t2 = GET:I64(offset=48)
t0 = Sub64(t2,0x0000000000000010)
PUT(offset=144) = 0x0000000000000008
PUT(offset=152) = t2
PUT(offset=160) = 0x0000000000000010
PUT(offset=168) = 0x0000000000000000
PUT(offset=48) = t0
```

Không hề có bất kỳ cờ điều kiện nào xuất hiện! VEX định danh các thanh ghi bằng độ lệch của chúng trong cấu trúc trạng thái máy khách (guest state), và độ lệch 48 tương ứng với `rsp`. Các độ lệch từ 144 đến 168 là `cc_op`, `cc_dep1`, `cc_dep2`, và `cc_ndep`. Bộ nâng mã chỉ đơn thuần lưu trữ mã thao tác 8 (tương ứng với phép trừ 64-bit) cùng hai toán hạng đầu vào. Một lệnh rẽ nhánh sau này nếu cần đọc cờ sẽ tự động tính giá trị cờ từ chính mẩu đệm (thunk) được lưu trữ đó. Đó là lý do tại sao thuộc tính `pcode.writesFlags` là true còn `vex.computesFlags` là false trong báo cáo kiểm thử, và cũng là lý do VEX có số lượng lệnh ngắn nhất trong cả bốn dạng.

Cùng bốn bộ nâng mã đó khi xử lý lệnh `48 63 c7` (`movslq %edi, %rax` — lệnh đầu tiên của `grade` ở bản dựng `-O1`) cho ra 5 token ESIL (`32,edi,~,rax,=`), 1 lệnh gán `set` trong RzIL, 1 phép toán P-code (`RAX = sext(EDI)`), và 4 câu lệnh VEX. Không bộ nâng mã nào ghi cờ, bởi bản thân lệnh máy này không làm thay đổi cờ trạng thái. Bản niêm yết của Ghidra cho lệnh này khi thu thập qua script headless in ra phép toán: `(register, 0x0, 8) INT_SEXT (register, 0x38, 4)`: độ lệch thanh ghi 0 là `RAX`, độ lệch `0x38` là `RDI`, và kích thước tương ứng là 8 và 4 byte.

Lệnh `83 f9 04` (`cmpl $0x4, %ecx` — lệnh kiểm tra biên trong hàm `grade`) sinh ra 30 token ESIL, 13 hiệu ứng `set` trong RzIL, 10 phép toán P-code, và 7 câu lệnh VEX với `cc_op` bằng 7 (mã định danh cho phép trừ 32-bit). Lệnh này hoàn toàn không ghi đè giá trị lên `%ecx`. Hai độ rộng phép trừ bạn sẽ gặp trong VEX là 7 cho 32-bit và 8 cho 64-bit, và bộ chuyển dịch trong bài học này sẽ lưu giữ bảng tra cứu đó.

## Bốn Địa Chỉ Cơ Sở Cho Một Hàm

Trước khi bộ khung so sánh có thể đối chiếu bất kỳ dữ liệu nào, nó phải thống nhất được vị trí của hàm trong không gian bộ nhớ. Hãy làm phép tính số học này một lần trên giấy đối với hàm `grade` trong tệp `ledger-x86_64-O1.o`:

| Công cụ | Địa chỉ của `.text` | Địa chỉ của `grade` | Độ lệch thực tế |
|---|---|---|---|
| `llvm-objdump` | `0x0` | `0x10` | `0x10 - 0x0 = 0x10` |
| radare2, rizin | `0x08000040` | `0x08000050` | `0x08000050 - 0x08000040 = 0x10` |
| angr | `0x400000` | `0x400010` | `0x400010 - 0x400000 = 0x10` |
| Ghidra | `0x100000` | `0x100010` | `0x100010 - 0x100000 = 0x10` |

Bảng ký hiệu gốc xác định rõ `grade` nằm tại `.text + 0x10` với kích thước 46 byte. Mọi công cụ đều hoàn toàn nhất quán khi địa chỉ cơ sở được loại bỏ. Bộ khung kiểm thử trích xuất địa chỉ cơ sở từ dữ liệu thu thập của từng công cụ: lệnh `iSj` trong radare2 và rizin, mục phân đoạn `.text` trong dữ liệu của angr, khối bộ nhớ `.text` trong dữ liệu của Ghidra, và mặc định bằng 0 cho phương pháp quét tuyến tính. Bài kiểm thử `test_every_tool_places_grade_at_offset_0x10_from_its_own_base` kiểm chứng quy tắc quy đổi này.

## Nơi Các Công Cụ Đồng Thuận và Nơi Chúng Phân Kỳ

Bộ khung kiểm thử trong `code/main.py` đọc dữ liệu thu thập, quy đổi mọi địa chỉ về độ lệch chuẩn, và thực hiện so sánh đối chiếu. Dưới đây là các số liệu ghi nhận được, và bộ kiểm thử tự động kiểm chứng từng khẳng định này.

### Trường hợp đồng thuận: ledger ở mức tối ưu -O1

Tệp đối tượng chứa bốn hàm, 56 lệnh máy trong lần quét tuyến tính (trong đó 54 lệnh nằm trong phạm vi các ký hiệu hàm và 2 lệnh nằm ở vùng đệm căn chỉnh). Mọi công cụ đều báo cáo chính xác các điểm bắt đầu lệnh trong từng hàm: 9, 14, 13, và 18 lệnh. Bốn bộ dựng đồ thị báo cáo lần lượt 3, 3, 5, và 4 khối cơ bản cùng 2, 3, 5, và 4 cạnh điều khiển; tập hợp các cạnh là hoàn toàn trùng khớp từng cặp một giữa radare2, rizin, angr, và Ghidra. Năm khối cơ bản và năm cạnh của hàm `total` chính là cấu trúc mà chúng ta đã vẽ tay trong bài học 33.

Điểm khác biệt duy nhất nằm ở vùng đệm căn chỉnh: 3 byte nằm giữa các hàm (`66 90` sau `grade`, `90` sau `total`) được giải mã bởi bộ quét tuyến tính (vốn quét qua mọi byte mà không có điều kiện dừng), và bởi angr — công cụ đã tự động tạo thêm hai hàm `sub_40003e` và `sub_40005f` mà không hề có ký hiệu tương ứng. radare2, rizin, và Ghidra hoàn toàn không nhận diện các byte đệm này thành hàm.

### Trường hợp vector hóa: ledger ở mức tối ưu -O2

Hàm `total` gồm 41 lệnh máy phân bố trong 10 khối cơ bản với 13 cạnh điều khiển, đồng nhất trên cả bốn công cụ dựng đồ thị. Vòng lặp vector hóa SSE2 và vòng lặp xử lý phần dư vô hướng (scalar remainder loop) từ bài học 33 được phân cắt hoàn toàn giống nhau ở mọi nơi. angr một lần nữa tự ý tạo thêm hai hàm phụ tại các vùng đệm căn chỉnh.

### Bảng nhảy: hàm grade ở mức -O0

Tại đây, các công cụ bắt đầu phân kỳ rõ rệt, và sự phân kỳ này chính là bài học đắt giá:

| Công cụ | Khối | Cạnh | Bản chất ghi nhận từ dữ liệu |
|---|---|---|---|
| radare2 | 3 | 2 | Khối tại `0x62` không có khối kế tiếp; các byte từ `0x76` đến `0x9a` không bao giờ được chạm tới; thiếu 8 điểm bắt đầu lệnh |
| rizin | 8 | 11 | Khối tại `0x62` mang thuộc tính `switch_op` với 5 trường hợp và 4 đích đến phân biệt |
| angr | 8 | 7 | Các khối case tồn tại, nhưng bước chuyển tại `0x62` dẫn tới một nút đích không thể giải quyết (unresolvable target node) |
| Ghidra | 9 | 12 | Nhãn `switchD` tại `0x74` làm tách khối; bốn luồng nhảy tính toán `COMPUTED_JUMP` dẫn tới `0x76`, `0x7f`, `0x88`, `0x91` |

Cả bốn công cụ đều giải mã cùng 33 lệnh máy khi chúng tiếp cận được mã lệnh. Tuy nhiên, radare2 dừng lại ngay tại lệnh `jmp rax` và bỏ sót 36 byte không được nhận diện, và bộ khung kiểm thử gắn nhãn vùng này là mã bị bỏ sót (missed code). rizin và Ghidra tìm thấy chính xác bốn đích đến phân biệt mà bài học 33 đã đọc từ năm bản ghi tái định vị trong phân đoạn `.rela.rodata`. angr liệt kê đầy đủ các khối nhánh case bên trong hàm `grade` nhưng lại không nối cạnh nào từ khối lệnh nhảy tới chúng; dữ liệu thu thập của nó ghi nhận bước chuyển tại `0x62` dẫn tới một nút đích không thể giải quyết, và trình dịch ngược của nó in ra một câu lệnh `goto` tính toán. Ghidra có nhiều hơn rizin 1 khối và 1 cạnh cho cùng một luồng điều khiển logic, bởi bộ phân tích switch của nó đã gắn thêm một nhãn vào lệnh `jmp rax`, và mô hình khối của Ghidra luôn bắt đầu một khối mới tại vị trí xuất hiện nhãn.

Đồ thị trong bài học 33 có 8 khối và 11 cạnh, và rizin trùng khớp hoàn toàn với cấu trúc đó. Điểm cốt lõi là nhận ra rằng các con số 9, 11, 7, và 2 không phải là bốn phép đo khác nhau về lượng mã máy được tìm thấy. Chúng là bốn câu trả lời khác nhau cho câu hỏi: Công cụ đã lần theo lệnh nhảy gián tiếp đi xa tới đâu và nó lựa chọn phân cắt ranh giới khối tại vị trí nào!

### Dữ liệu nằm xen trong mã lệnh: bẫy quét tuyến tính (sweep-trap)

Hàm `lookup` có độ dài 24 byte với 4 byte của bảng dữ liệu `31 c0 90 48` nằm tại độ lệch `0x10`. Bộ quét tuyến tính báo cáo 8 điểm bắt đầu lệnh trong phạm vi đó. Bộ khung so sánh chỉ ra các sai lệch của nó so với số đông: thừa các điểm bắt đầu tại `0x10`, `0x12`, và `0x13`, đồng thời thiếu mất điểm bắt đầu tại `0x14`. Đó chính là lệnh `addq $0x1, %rax` từ bài học 33 bị đọc sớm một byte! radare2, rizin, và Ghidra báo cáo 6 điểm bắt đầu lệnh, 2 khối cơ bản, 1 cạnh, và để trống đoạn từ `0x10` đến `0x14`. angr báo cáo cùng 2 khối và 1 cạnh cho `lookup`, nhưng sau đó lại tự ý sinh thêm hàm `sub_400010` ngay trên các byte của bảng dữ liệu: 2 lệnh trong một khối 4-byte kết thúc tại `0x14` (nơi khối thứ hai của `lookup` bắt đầu), dẫn đến byte REX tại `0x13` không thuộc về bất kỳ lệnh nào. Kết luận cuối cùng của bộ khung so sánh cho tệp này là: không có bộ dựng đồ thị nào giải mã từ `0x10` đến `0x14`, vì vậy các byte đó nằm bên trong một hàm và bản chất của chúng là dữ liệu thuần túy.

### Thống kê mã tự sinh và mã bị bỏ sót theo từng công cụ

Trên cả bốn tệp đối tượng, bộ khung kiểm thử ghi nhận 9 hàm do angr tự ý sinh ra mà không có ký hiệu đi kèm — tất cả đều nằm tại các khoảng đệm căn chỉnh hoặc tại bảng dữ liệu bẫy. Nó ghi nhận một vùng duy nhất bị radare2 bỏ sót: 36 byte của các nhánh case trong hàm `grade` ở bản dựng `-O0`. Nó ghi nhận 9 vùng mà bộ quét tuyến tính giải mã nhưng không có bộ dựng đồ thị nào coi là mã lệnh: tám khoảng đệm căn chỉnh cộng với bảng dữ liệu bẫy. Tuyệt đối không có hàm thực sự nào bị bỏ sót ở bất kỳ công cụ nào.

## Ba Trình Dịch Ngược Trên Hàm grade

Ba trong số các công cụ được thu thập xuất ra văn bản tựa C cho hàm `grade` ở bản dựng `-O1`. Mã nguồn gốc là một câu lệnh `switch (score / 10)` gồm năm nhánh case, được biên dịch thành một phép kiểm tra biên kết hợp một bảng tra cứu giá trị trả về.

Ghidra 12.1.4:

```c
uint grade(int param_1)
{
  uint uVar1;
  uint uVar2;

  uVar2 = param_1 / 10 - 6;
  uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);
  if (uVar2 < 5) {
    uVar1 = *(uint *)(&DAT_001000b4 + (ulong)uVar2 * 4);
  }
  return uVar1;
}
```

angr 10.0.3:

```c
extern unsigned int g_4000b4[4];

unsigned int grade(int a0)
{
    unsigned int v1;  // ecx
    unsigned int v2;  // eax

    v1 = (unsigned int)((long long)(a0 * 1717986919) >> 0x22) + (unsigned int)(a0 * 1717986919 >> 63) - 6;
    v2 = 0;
    if (v1 <= 4)
        v2 = g_4000b4[v1];
    return v2;
}
```

radare2 6.2.4, lệnh `pdc`:

```c
void sym.grade (int64_t arg1) {
    rax = edi
    rax = rax * 0x66666667
    rcx = rax
    rcx >>= 0x22
    rax >>>= 0x3f
    ecx += eax
    ecx += 0xfffffffa
    eax = 0
    v = ecx - 4
    if (((unsigned) v) <= 0) {
        eax = ecx
        rcx = rip
        eax = dword [rcx + rax*4]
    }
    return
}
```

Hãy đối chiếu ba đoạn mã trên với các giai đoạn của đường ống dịch ngược: Lượt viết lại biểu thức của Ghidra đã gộp phép nhân với `0x66666667` và phép dịch phải 34 bit quay trở lại biểu thức quen thuộc `param_1 / 10` (chính là thành ngữ tối ưu từ bài học 34). angr nâng cùng các byte đó nhưng giữ nguyên phép tính số học dưới dạng `a0 * 1717986919 >> 0x22`. Lệnh `pdc` của radare2 thực chất là mô phỏng ESIL kết hợp với cú pháp giả lập: giữ nguyên tên thanh ghi, không có biến, không có kiểu dữ liệu, và phép kiểm tra biên được in thành `<= 0` sau một phép trừ không dấu. Ghidra gán kiểu cho việc nạp bảng qua nhãn `DAT_001000b4`; angr khai báo mảng `g_4000b4[4]` — chỉ có bốn phần tử trong khi mã nguồn gốc có năm. Cả Ghidra lẫn angr đều không khôi phục lại cấu trúc liệt kê các nhánh case, bởi ở mức tối ưu `-O1`, cấu trúc lệnh switch đã hoàn toàn biến mất khỏi các byte mã máy. Ở bản dựng `-O0` (nơi bảng nhảy thực sự tồn tại), Ghidra in ra `switch(param_1 / 10)` với các case từ 6 đến 10 kèm nhánh default, trong khi angr in ra một câu lệnh `goto` tính toán.

rizin không có bộ dịch ngược riêng, và dữ liệu thu thập ghi nhận lệnh `pdc` không tồn tại ở công cụ này. Dạng HLIL của Binary Ninja và cấu trúc ctree của Hex-Rays là hai đầu ra mà bài học này không thể thu thập tự động do rào cản bản quyền đóng.

## Xây Dựng Mã Nguồn (Build It)

Chạy chương trình minh họa:

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
```

Đầu ra là một báo cáo JSON duy nhất. Nó mở đầu bằng các phiên bản công cụ được ghi nhận tại thời điểm thu thập dữ liệu và danh sách các công cụ hiện diện trên máy cục bộ của bạn (có thể hoàn toàn vắng mặt). Tiếp theo, đối với mỗi tệp trong bốn tệp đối tượng, báo cáo cung cấp phạm vi ký hiệu hàm, các khoảng đệm căn chỉnh, địa chỉ cơ sở mà từng công cụ sử dụng, các điểm bắt đầu hàm mà từng công cụ tìm thấy hoặc tự ý sinh ra, và đối với từng hàm: sự đồng thuận về điểm bắt đầu lệnh, số lượng khối cơ bản, số lượng cạnh, cùng danh sách các cạnh mà từng công cụ bị thiếu. Báo cáo kết thúc bằng bảng kiểm kê byte của từng công cụ, các ghi chú kỹ thuật riêng của từng công cụ, danh sách các phát hiện được diễn đạt bằng câu văn rõ ràng, và phần phân tích IR: đầu ra của bộ chuyển dịch cho ba lệnh máy đặt cạnh các số đo thực tế thu được từ các bản nâng mã.

Cấu trúc đầu ra kỳ vọng:

```json
{
  "captureTools": {"llvm-objdump": "Apple LLVM version 17.0.0", "radare2": "radare2 6.2.4 ...", "rizin": "rizin 0.9.1 ...", "angr": "10.0.3", "ghidra": "12.1.4"},
  "localTools": {"radare2": "absent", "...": "..."},
  "fixtures": {
    "ledger-x86_64-O0.o": {
      "bases": {"llvm-objdump": "0x0", "radare2": "0x8000040", "angr": "0x400000", "ghidra": "0x100000", "...": "..."},
      "functions": {"grade": {"blocks": {"radare2": 3, "rizin": 8, "angr": 8, "ghidra": 9}, "edges": {"radare2": 2, "rizin": 11, "angr": 7, "ghidra": 12}, "...": "..."}},
      "findings": ["radare2 never reaches bytes 0x76-0x9a inside grade that the other graph builders decode", "..."]
    }
  },
  "ir": {"translator": [{"text": "subq $0x10, %rsp", "flags": {"pcodeLike": "eager", "vexLike": "thunk"}, "...": "..."}], "observed": {"...": "..."}}
}
```

Hãy đọc `code/main.py` theo trình tự sau:

1. `read_text_symbols()` đọc tiêu đề phân đoạn và các ký hiệu `STT_FUNC` của phân đoạn `.text`. Bảng ký hiệu là chân lý chuẩn duy nhất mà bộ khung kiểm thử sở hữu, và hàm `TextSection.gaps()` suy diễn các khoảng đệm từ bảng này.
2. Các hàm `parse_objdump()`, `parse_r2_family()`, `parse_angr()`, và `parse_ghidra()` chuyển đổi dữ liệu thu thập của từng công cụ thành một đối tượng `ToolView`: gồm địa chỉ cơ sở, ánh xạ từ độ lệch sang độ dài lệnh, và một đối tượng `FunctionView` cho mỗi hàm chứa các khối, các cạnh và văn bản dịch ngược (nếu có). Mỗi bộ phân tích chỉ hiểu cú pháp riêng của một công cụ nhất định.
3. `majority()` và `ranges()` là hai hàm bổ trợ được sử dụng xuyên suốt mọi phép so sánh.
4. `function_report()` so sánh điểm bắt đầu lệnh đối chiếu với số đông của các bộ dựng đồ thị, sau đó so sánh số lượng khối, tập hợp các cạnh, và liệt kê các cạnh mà từng công cụ bị thiếu.
5. `byte_report()` kiểm kê từng byte dưới dạng: mã lệnh trong ký hiệu, khoảng đệm, chưa được nhận diện, tự ý sinh thêm, hoặc bị bỏ sót; đồng thời chỉ rõ các byte bên trong hàm mà không có bộ dựng đồ thị nào giải mã.
6. `findings()` chuyển đổi các sai biệt thành các câu văn kết luận, và hàm `compare()` tổng hợp thành báo cáo hoàn chỉnh cho từng tệp đối tượng.
7. `decode_small()`, `to_pcode_like()`, và `to_vex_like()` cấu thành bộ chuyển dịch: hỗ trợ ba dạng mã hóa thanh ghi, một theo phong cách tính cờ tức thời (eager) và một theo phong cách mẩu đệm (thunk).
8. `observed_ir()` đo đạc các dữ liệu nâng mã thực tế, và `figure_data()` tập hợp dòng thời gian, bảng giai đoạn và các bằng chứng hiển thị trong sơ đồ.

Ba lệnh bổ trợ hữu ích: `python3 code/main.py compare FIXTURE.o` in báo cáo cho một tệp đối tượng cụ thể; `python3 code/main.py ir 4883ec10` in hai dạng chuyển dịch IR cho một lệnh máy; và `python3 code/main.py figure-data` xuất dữ liệu phục vụ trực quan hóa.

Dữ liệu thu thập được tạo ra một lần bởi tập lệnh `code/fixtures/capture_tools.py` — ghi nhận từng lệnh thực thi và phiên bản công cụ vào tệp `captures.json`. Bộ khung kiểm thử và các bài kiểm tra tuyệt đối không khởi chạy bất kỳ công cụ ngoài nào trong quá trình chạy. Nếu máy bạn đã cài đặt sẵn các công cụ, bạn có thể chạy tập lệnh thu thập để làm mới dữ liệu và đọc tệp README trong thư mục fixtures để biết các yêu cầu môi trường cần thiết.

## Sử Dụng Mã Nguồn (Use It)

Khi phân tích một tệp đối tượng mới bằng bộ khung kiểm thử, hãy tuân thủ trình tự sau:

1. Tự tay biên dịch tệp đối tượng và lưu lại cờ lệnh biên dịch. Đọc bảng ký hiệu của tệp trước khi mở bất kỳ công cụ nào để nắm rõ có bao nhiêu hàm và các khoảng đệm nằm ở đâu.
2. Thu thập dữ liệu từng công cụ một lần ở chế độ headless. Lưu trữ đầu ra thô và câu lệnh chính xác bên cạnh tệp đối tượng. Một bất đồng mà không thể tái tạo lại thì không được coi là một phát hiện hợp lệ.
3. Kiểm tra địa chỉ cơ sở. Chọn một hàm và xác nhận rằng mọi công cụ đều đặt nó tại cùng một độ lệch sau khi đã trừ đi địa chỉ của `.text`. Nếu có sai lệch, hãy dừng lại và kiểm tra xem công cụ đó đã nạp những gì.
4. So sánh điểm bắt đầu lệnh trước tiên. Nếu các bộ dựng đồ thị bất đồng ở bước này, phần còn lại của phép so sánh chỉ xoay quanh việc giải mã lệnh chứ không phản ánh sự khác biệt về năng lực phân tích.
5. So sánh các khối và các cạnh, và luôn đọc kỹ ghi chú riêng của từng công cụ trước khi phán xét số lượng. Một cấu trúc `switch_op`, một luồng `COMPUTED_JUMP`, hoặc một nút đích không thể giải quyết sẽ giải thích cho phần lớn các khác biệt.
6. Kiểm kê byte một cách minh bạch. Vùng đệm mà một công cụ nhận diện thành hàm phản ánh một thuật toán phỏng đoán đang hoạt động. Các byte hàm bị bỏ sót phản ánh một bước duyệt đệ quy bị dừng sớm. Các byte hàm mà không công cụ nào nhận diện chính là dữ liệu thuần túy nằm lẫn trong mã.
7. Ghi rõ công cụ nào chỉ được mô tả dựa trên tài liệu. Thông tin từ tài liệu là hữu ích nhưng thuộc về một cấp độ bằng chứng hoàn toàn khác so với dữ liệu thực nghiệm được thu thập trực tiếp.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: một hàm, bốn địa chỉ cơ sở

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape/code
python3 -q
```

Chạy:

```python
from main import *
views = load_views("ledger-x86_64-O1.o")
{tool: hex(view.base) for tool, view in views.items()}
{tool: hex(view.functions["grade"].start + view.base) for tool, view in views.items()}
```

Ghi lại bốn địa chỉ thô ra giấy trước khi chạy dòng thứ hai và đối chiếu chúng với bảng số liệu trong bài học. Sau đó thay thế `grade` bằng `grade_label` và tự tính toán trên giấy xem nó nằm ở địa chỉ nào trong Ghidra.

### Lab B: bảng nhảy từ bốn góc nhìn

```python
report = compare("ledger-x86_64-O0.o")
report["functions"]["grade"]["blocks"]
report["functions"]["grade"]["edgesMissingPerTool"]["radare2"]
report["notes"]["rizin"]
report["notes"]["ghidra"]
report["bytes"]["radare2"]["misses"]
```

Đếm số lượng cạnh mà radare2 bị thiếu và đối chiếu từng cạnh với một dòng trong bản niêm yết mã `-O0` từ bài học 33. Sau đó đọc ghi chú của Ghidra và tìm nhãn `switchD` trong dữ liệu thu thập của Ghidra.

### Lab C: cờ tức thời đối chiếu với mẩu đệm tính toán

```python
translate("48 83 ec 10")["pcodeLike"]
translate("48 83 ec 10")["vexLike"]
translate("83 f9 04")["vexLike"]
{row["name"]: (row["pcode"]["ops"], row["vex"]["statements"]) for row in observed_ir()["instructions"]}
```

Tìm hằng số `cc_op` trong cả hai kết quả đầu ra và giải thích vì sao một bên là 8 và bên kia là 7. Sau đó thử chạy `translate("48 01 c8")` và quan sát thông báo ngoại lệ `TranslateError`.

### Lab D: bẫy dữ liệu dưới năm công cụ

```python
report = compare("sweep-trap.o")
report["functions"]["lookup"]["instructionStarts"]
report["bytes"]["dataInsideSymbols"]
report["functionStarts"]["angr"]
```

Giải mã chuỗi byte `31 c0 90 48 83 c0 01` trên giấy theo cách quét tuyến tính, sau đó giải mã theo cách duyệt đệ quy sau lệnh `jmp`. Đối chiếu hai bản niêm yết của bạn với các điểm bắt đầu thừa và thiếu mà bộ khung kiểm thử báo cáo.

## Bài Tập Thực Hành (Practice Lab)

Bổ sung dữ liệu thu thập từ một công cụ thứ năm và mở rộng bộ khung kiểm thử.

Các yêu cầu:

- Chọn một công cụ mà bài học này chưa thu thập và bạn có thể chạy được trên máy: GNU `objdump`, LIEF kết hợp Capstone, một phiên làm việc xuất thủ công từ Binary Ninja Free, hoặc Ghidra với một tập hợp bộ phân tích khác. Thu thập đầu ra của nó trên cả bốn tệp đối tượng kèm theo câu lệnh chính xác và phiên bản trong `captures.json`.
- Viết một bộ phân tích cú pháp để chuyển đổi dữ liệu thu thập thành đối tượng `ToolView` mang địa chỉ cơ sở, các lệnh, các hàm, các khối (nếu công cụ hỗ trợ), và các cạnh (nếu công cụ hỗ trợ).
- Thêm công cụ vào danh sách `TOOLS` và (nếu nó xây dựng đồ thị) vào danh sách `DESCENT_TOOLS`. Chạy bản demo và quan sát những thay đổi trong tập hợp số đông (majority sets).
- Viết một bài kiểm thử xác nhận số lượng khối cơ bản của công cụ trên `ledger-x86_64-O1.o` và một bài kiểm thử xác nhận cách nó xử lý bảng dữ liệu bẫy trong `sweep-trap.o`.
- Viết một đoạn văn ngắn cho bài học mô tả, bằng chính thuật ngữ của công cụ đó, cách nó gọi tên bộ nạp, bộ phân dịch, và đồ thị CFG.

Thành công đồng nghĩa với việc lệnh `python3 code/main.py` báo cáo công cụ mới trong mục `captureTools`, mọi hàm trong `ledger-x86_64-O1.o` duy trì cờ `allDescentToolsAgree` bằng true, và phần kết luận cho `sweep-trap.o` nêu rõ công cụ mới đã làm gì với các byte từ `0x10` đến `0x14`.

## Sản Phẩm Đóng Gói (Lesson Artifact)

Bài học này đi kèm sản phẩm `outputs/skill-the-reverse-engineering-landscape.md`. Hãy sử dụng nó để vận hành nhiều công cụ trên một tệp đối tượng do bạn tự xây dựng, quy đổi địa chỉ của chúng về một cơ sở chung, và báo cáo chi tiết theo từng hàm về mức độ đồng thuận trên điểm bắt đầu lệnh, khối cơ bản, cạnh điều khiển, cùng danh sách các byte mà từng công cụ tự ý sinh thêm hoặc bỏ sót. Kỹ năng này đòi hỏi dữ liệu thu thập phải có câu lệnh và phiên bản rõ ràng, bảng ký hiệu phải được đọc trực tiếp từ tệp nhị phân, và phải phân định minh bạch công cụ nào chỉ được trích dẫn từ tài liệu kỹ thuật.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử hoàn chỉnh:

```bash
cd phases/13-tools-and-protocols/37-the-reverse-engineering-landscape
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Toàn bộ các bài kiểm tra chứng minh:

- Bộ đọc ELF tìm thấy bốn ký hiệu hàm trong `ledger-x86_64-O1.o` với các kích thước lần lượt là 16, 46, 31, và 63 byte, cùng hai khoảng đệm căn chỉnh.
- Dữ liệu quét tuyến tính chứa lần lượt 86, 56, 84, và 11 lệnh máy cho bốn tệp đối tượng.
- radare2 và rizin báo cáo cùng các lệnh, khối và cạnh trên `ledger-x86_64-O1.o`, và chỉ có radare2 có văn bản `pdc`.
- angr đánh dấu `sub_40003e` và `sub_40005f` là các hàm không có ký hiệu.
- Các khối của Ghidra cho hàm `total` chính là năm khối từ bài học 33, và luồng gọi hàm không bị coi là cạnh đồ thị CFG.
- Mọi công cụ đều đặt `grade` tại độ lệch `0x10` sau khi loại bỏ địa chỉ cơ sở.
- Mọi công cụ đều đồng thuận trên từng điểm bắt đầu lệnh trong mọi hàm của `ledger-x86_64-O1.o`.
- Bốn bộ dựng đồ thị báo cáo lần lượt 3, 3, 5, 4 khối và 2, 3, 5, 4 cạnh tại tệp đó, với tập hợp các cạnh hoàn toàn trùng khớp.
- Hàm `total` ở mức tối ưu `-O2` có 10 khối và 13 cạnh trên mọi bộ dựng đồ thị.
- Hàm `grade` ở mức `-O0` có lần lượt 3, 8, 8, và 9 khối cùng 2, 11, 7, và 12 cạnh trong radare2, rizin, angr, và Ghidra, với nguyên nhân được giải thích rõ trong ghi chú của từng công cụ.
- Hàm `grade_label` ở mức `-O0` là một khối duy nhất ở mọi nơi, ngoại trừ angr — công cụ ngắt khối ngay tại lệnh gọi hàm.
- angr tự ý sinh thêm một hàm cho mỗi khoảng đệm cộng thêm một hàm trên bảng dữ liệu bẫy, và không có công cụ nào khác tự sinh hoặc bỏ sót hàm.
- Phương pháp quét tuyến tính thu nhận thêm các điểm bắt đầu `0x10`, `0x12`, `0x13` và mất điểm bắt đầu `0x14` trên bẫy dữ liệu, và không bộ dựng đồ thị nào giải mã từ `0x10` đến `0x14`.
- Kết luận chỉ rõ phạm vi bị bỏ sót trong radare2 và mọi hàm tự ý sinh ra trong angr.
- Tệp kê khai lưu trữ đầy đủ phiên bản và câu lệnh cho toàn bộ 21 tệp dữ liệu thu thập.
- Ba công cụ dịch ngược hàm `grade`, và duy nhất Ghidra gộp được thành ngữ phép chia.
- Bộ chuyển dịch ghi các cờ CF, OF, SF, ZF, PF một cách tức thời trong phong cách thứ nhất và lưu mẩu đệm `cc_op` trong phong cách thứ hai; không ghi thanh ghi nào cho `cmp`, và không ghi cờ nào cho `movslq`.
- Bộ chuyển dịch từ chối các dạng thao tác bộ nhớ, các mã thao tác khác, các bit REX thừa, và các độ dài không hợp lệ.
- Các chỉ số `cc_op` bằng 8 và 7 của bộ chuyển dịch trùng khớp hoàn toàn với các chỉ số trong dữ liệu VEX thực tế.
- Dữ liệu nâng mã thực tế cho ra 9, 1, 10 phép toán P-code và 7, 4, 7 câu lệnh VEX cho ba lệnh máy được kiểm tra.
- Dòng thời gian trải dài từ 1973 đến 2026 theo thứ tự thời gian; quy trình đường ống gồm tám giai đoạn và sáu công cụ với thuật ngữ đầy đủ trong từng ô; hai công cụ nguồn đóng được đánh dấu rõ là không chạy thực tế.
- Sơ đồ tương tác nhúng chính xác các dữ liệu mà bộ khung kiểm thử tạo ra.

## Các Chế Độ Thất Bại Trong Môi Trường Thực Tế

| Thất bại | Báo cáo yếu kém tuyên bố | Điều bộ khung kiểm thử bắt buộc phải chứng minh |
|---|---|---|
| Địa chỉ thô từ các công cụ khác nhau | "các công cụ bất đồng về vị trí của grade" | một độ lệch duy nhất sau khi trừ địa chỉ cơ sở, kiểm chứng trên một hàm |
| Số lượng khối là thước đo lượng mã tìm thấy | "Ghidra tìm thấy nhiều mã nhất" | cùng các điểm bắt đầu lệnh, khối thừa được giải thích là do nhãn phân nhánh |
| Lấy một công cụ duy nhất làm chân lý chuẩn | "objdump cho thấy các byte bảng là lệnh máy" | quét tuyến tính không có ngữ cảnh; ba bộ dựng đồ thị để trống các byte đó |
| Các hàm ở vùng đệm được coi là phát hiện mới | "angr tìm thêm được bảy hàm" | tất cả đều nằm trong khoảng đệm hoặc dữ liệu, và không hàm nào có ký hiệu |
| Dừng duyệt đệ quy bị báo cáo là mã chết | "bốn khối case là không thể tiếp cận" | 36 byte bị một công cụ bỏ sót nhưng được ba công cụ khác tìm thấy đầy đủ |
| So sánh số cạnh mà không đọc ghi chú | "rizin và Ghidra bất đồng về hàm grade" | cùng bốn đích nhảy tính toán, Ghidra tách thêm một khối tại vị trí nhãn |
| Phép toán của trình dịch ngược bị coi là chương trình | "hàm này nhân với số 1717986919" | thành ngữ từ bài học 34: một công cụ đã gộp lại còn công cụ kia giữ nguyên |
| Mô tả công cụ đóng như thể đã chạy thực tế | "Binary Ninja báo cáo năm khối" | sơ đồ và bảng số liệu đánh dấu rõ đây chỉ là thông tin từ tài liệu |
| Thu thập dữ liệu mà không ghi lại câu lệnh | "chúng tôi đã chạy radare2" | tệp kê khai ghi rõ phiên bản, câu lệnh, và ngày tháng cho từng tệp |
| IR không có cờ bị coi là nâng mã thiếu sót | "VEX đã làm mất các cờ điều kiện" | mẩu đệm trong `cc_op`, `cc_dep1`, `cc_dep2` được đánh giá khi có lệnh rẽ nhánh |

## Quy Tắc Vận Hành Cốt Lõi

Hãy thu thập dữ liệu từng công cụ một lần kèm theo câu lệnh và phiên bản chính xác, quy đổi mọi địa chỉ về một địa chỉ cơ sở chung, và luôn so sánh các điểm bắt đầu lệnh trước khi so sánh các khối, và so sánh các khối trước khi so sánh các cạnh. Hãy kiểm kê từng byte một cách minh bạch: mã lệnh được nhận diện, khoảng đệm căn chỉnh, dữ liệu thuần túy, hay mã bị bỏ sót. Hãy đọc kỹ ghi chú riêng của công cụ trước khi phán xét các con số thống kê, và luôn nêu rõ công cụ nào bạn chỉ mới tìm hiểu qua tài liệu.

## Đọc Thêm

- [Cifuentes, "Reverse Compilation Techniques", Luận án Tiến sĩ, QUT (1994)](https://eprints.qut.edu.au/36820/)
- [Cifuentes và Gough, "Decompilation of Binary Programs", Software: Practice and Experience (1995)](https://doi.org/10.1002/spe.4380250706)
- [Van Emmerik, "Static Single Assignment for Decompilation", Luận án Tiến sĩ, Đại học Queensland (2007)](https://espace.library.uq.edu.au/view/UQ:158682)
- [Schwartz, Lee, Woo, Brumley, "Native x86 Decompilation Using Semantics-Preserving Structural Analysis and Iterative Control-Flow Structuring" (USENIX Security 2013)](https://www.usenix.org/conference/usenixsecurity13/technical-sessions/presentation/schwartz)
- [Yakdan và cộng sự, "No More Gotos" (NDSS 2015)](https://www.ndss-symposium.org/ndss2015/ndss-2015-programme/no-more-gotos-decompilation-using-pattern-independent-control-flow-structuring-and-semantics/)
- [Basque và cộng sự, "Ahoy SAILR! There is No Need to DREAM of C" (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/basque)
- [Andriesse và cộng sự, "An In-Depth Analysis of Disassembly on Full-Scale x86/x64 Binaries" (USENIX Security 2016)](https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/andriesse)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004)](https://dl.gi.de/items/7f573931-cb51-49ef-a12c-0452a9fb0ea4)
- [Maranzano và Bourne, "A Tutorial Introduction to ADB", Bell Labs (1977)](http://squoze.net/UNIX/v7/files/doc/18_adb.pdf)
- [Stallman, bài giảng tại KTH về việc phát hành GDB (30 tháng 10 năm 1986)](https://www.gnu.org/philosophy/stallman-kth.html)
- [Hex-Rays, "IDA: celebrating 30 years of binary analysis innovation"](https://hex-rays.com/blog/ida-celebrating-30-years-of-binary-analysis-innovation)
- [Guilfanov, "Decompiler internals: microcode", Recon Brussels 2018 (slide thuyết trình)](https://www.recon.cx/2018/brussels/resources/slides/RECON-BRX-2018-Decompiler-internals-microcode.pdf)
- [Hex-Rays, khung nhìn microcode và các mức trưởng thành](https://docs.hex-rays.com/core/decompiler/how-tos/microcode-view)
- [Hex-Rays, ghi chú phát hành IDA 9.4](https://docs.hex-rays.com/release-notes/9_4)
- [Hex-Rays, IDA Free](https://hex-rays.com/ida-free)
- [Các bản phát hành Ghidra (12.1.4)](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Tài liệu tham khảo P-code của Ghidra](https://github.com/NationalSecurityAgency/ghidra/blob/master/GhidraDocs/languages/html/pcoderef.html)
- [Mã nguồn bộ dịch ngược Ghidra, luồng làm việc chính và các lớp Action](https://github.com/NationalSecurityAgency/ghidra/tree/master/Ghidra/Features/Decompiler/src/decompile/cpp)
- [Hướng dẫn sử dụng analyzeHeadless của Ghidra](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/RuntimeScripts/support/analyzeHeadlessREADME.md)
- [Binary Ninja, tổng quan về BNIL](https://docs.binary.ninja/dev/bnil-overview.html)
- [Binary Ninja, "Architecture Agnostic Function Detection in Binaries" (2017)](https://binary.ninja/2017/11/06/architecture-agnostic-function-detection-in-binaries.html)
- [Binary Ninja, xử lý hàng loạt và giấy phép headless](https://docs.binary.ninja/dev/batch.html)
- [Bài viết phát hành Binary Ninja 6.0 "Krypton"](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html)
- [Binary Ninja Free](https://binary.ninja/free/)
- [Sách hướng dẫn radare2, ESIL](https://book.rada.re/emulation/esil.html)
- [Sách hướng dẫn radare2, các trình dịch ngược](https://book.rada.re/arch/decompile.html)
- [Các bản phát hành radare2 (6.2.4)](https://github.com/radareorg/radare2/releases)
- [rizin, ghi chú thiết kế RzIL](https://github.com/rizinorg/rizin/blob/dev/doc/rzil.md)
- [Các bản phát hành rizin (0.9.1)](https://github.com/rizinorg/rizin/releases)
- [Công bố phát hành Rizin (8 tháng 12 năm 2020)](https://rizin.re/posts/announcing-rizin/)
- [Các bản phát hành Cutter (2.5.0)](https://github.com/rizinorg/cutter/releases)
- [Tài liệu angr, khôi phục đồ thị CFG](https://docs.angr.io/en/latest/analyses/cfg.html)
- [Tài liệu angr, trình dịch ngược và các lượt phân tích](https://docs.angr.io/en/latest/analyses/decompiler.html)
- [Tài liệu angr, biểu diễn trung gian (VEX)](https://docs.angr.io/en/latest/advanced-topics/ir.html)
- [angr trên PyPI (10.0.3)](https://pypi.org/project/angr/)
- [pypcode trên PyPI](https://pypi.org/project/pypcode/)
- [Hướng dẫn lệnh llvm-objdump](https://llvm.org/docs/CommandGuide/llvm-objdump.html)
- [Sổ tay hướng dẫn GNU objdump](https://sourceware.org/binutils/docs/binutils/objdump.html)
- [Tài liệu LIEF](https://lief.re/doc/stable/index.html)
- [Các bản phát hành Capstone](https://github.com/capstone-engine/capstone/releases)
- [Các bản phát hành BinDiff (v8, mã nguồn mở)](https://github.com/google/bindiff/releases/tag/v8)
- [Diaphora](https://github.com/joxeankoret/diaphora)
- [Trang tải xuống Hopper Disassembler](https://www.hopperapp.com/download.html)
- [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)
- [LLDB công bố là tiểu dự án của LLVM (2010)](https://blog.llvm.org/2010/06/new-lldb-debugger.html)
- [Trình dịch ngược Boomerang](https://boomerang.sourceforge.net/)
- [Tan và cộng sự, "LLM4Decompile" (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.203/)
- [Gao và cộng sự, "DecompileBench" (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.1194/)
- [Liu, Raff, Micinski, "When LLM Decompilers Recompile More and Preserve Less" (2026)](https://arxiv.org/abs/2609.05370)
