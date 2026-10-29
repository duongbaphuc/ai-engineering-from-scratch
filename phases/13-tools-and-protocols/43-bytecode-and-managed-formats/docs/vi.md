# Vượt ra ngoài mã máy gốc: Bytecode của Python, JVM, .NET và WebAssembly

> Một tệp mã bytecode bản chất là một chương trình kèm theo các ghi chú của trình biên dịch về chính chương trình đó, và bài học này sẽ xây dựng các bộ phân tích (reader) nhằm chỉ ra định dạng vùng chứa nào lưu giữ lại những ghi chú nào.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python, Java, C#, WebAssembly text
**Yêu cầu tiên quyết:** Phase 13 · 32 (ELF từ con số không), 33 (dịch ngược mã assembly và luồng điều khiển)
**Thời gian:** ~150 phút

## Mục tiêu học tập

- Đọc tiêu đề 16 byte của `.pyc` từ PEP 552 và giải tuần tự hóa (unmarshal) đối tượng mã (code object) mà không cần dùng mô-đun `marshal`, áp dụng cho các tệp được ghi bởi 4 phiên bản trình thông dịch khác nhau.
- Phân tích cú pháp tệp lớp (class file) JVM theo JVMS Chương 4: vùng hằng số (constant pool), các trường (fields), các phương thức (methods), thuộc tính `Code` (Code attribute) và các bảng do `javac -g` thêm vào.
- Dịch ngược (disassemble) các mã thao tác (opcodes) JVM trong `Ledger.class` và đối chiếu danh sách với `javap -c` trên 4 bản dựng từ cùng một mã nguồn.
- Đọc từng phân đoạn (section) của một mô-đun WebAssembly, giải mã toàn bộ thân hàm (function body) và kiểm chứng kết quả với `wasm-objdump -d`.
- Duyệt qua siêu dữ liệu (metadata) ECMA-335 của một assembly .NET từ thư mục dữ liệu PE đến các hàng `MethodDef` và mã CIL (Common Intermediate Language) của từng phương thức, được kiểm tra đối chiếu với thư viện phân tích mà ILSpy sử dụng.
- Đo lường đối với một hàm trên 5 vùng chứa khác nhau xem mỗi định dạng lưu giữ những định danh (names), kiểu dữ liệu (types), cấu trúc điều khiển nào và đánh mất những thông tin nào.

## Vấn đề thực tế

Bạn đã biên dịch `ledger.py` và tìm thấy tệp `ledger.cpython-314.pyc` trong thư mục `__pycache__`. Bạn nhờ một AI agent phân tích xem tệp đó chứa những gì. Agent đó đã chạy một trình dịch ngược (decompiler). Công cụ in ra `Unsupported opcode` kèm một dòng cảnh báo `# WARNING: Decompyle incomplete`. Agent liền vội vàng kết luận rằng tệp này đã bị làm rối mã (obfuscated).

Thực tế tệp này hoàn toàn không bị làm rối mã. Hai byte đầu tiên của nó là `2b 0e`, biểu diễn từ 16-bit little-endian có giá trị 3627. Đó chính là số ma thuật mà CPython 3.14 ghi ra, trong khi bảng mã thao tác (opcode table) của trình dịch ngược kia chỉ hỗ trợ đến bản 3.13. Công cụ đã lỗi thời. Agent đã không đọc tiêu đề tệp, do đó không thể phân biệt được giữa lỗi công cụ không tương thích và mã bị làm rối.

Bài học 33 đã chỉ ra những gì mà mã máy gốc (native code) đánh mất: tên tham số, kiểu dữ liệu và cấu trúc `switch` bị biến thành một bảng nhảy (jump table). Các vùng chứa bytecode làm mất ít thông tin hơn, nhưng mỗi loại lại đánh mất những thành phần khác nhau. Một tệp `.pyc` giữ lại mọi tên biến nhưng không giữ kiểu dữ liệu. Một tệp `.class` giữ lại kiểu dữ liệu nhưng chỉ lưu tên biến cục bộ khi bạn yêu cầu rõ ràng. Một mô-đun Wasm bảo toàn cấu trúc điều khiển nhưng lược bỏ mọi định danh trừ khi có một phân đoạn tùy chỉnh (custom section) mang theo chúng. Một assembly .NET giữ tên tham số trong một bảng riêng và đẩy tên biến cục bộ sang một tệp khác. Bạn không thể đánh giá kết quả xuất ra của một trình dịch ngược trên bất kỳ định dạng nào nếu bạn không biết vùng chứa đó vốn dĩ lưu giữ những ghi chú gì ngay từ đầu.

Bài học này xây dựng 4 bộ đọc (readers) bằng thư viện chuẩn của Python. Mỗi bộ đọc sẽ phân tích vùng chứa từ byte đầu tiên, liệt kê một hàm và được kiểm chứng đối chiếu với chính công cụ chuyên dụng của định dạng đó trên các tệp mẫu (fixtures) được biên dịch sẵn. Phần cuối cùng sẽ đặt hàm `grade` từ `ledger` lên bàn cân so sánh song song trên cả 5 vùng chứa.

````figure
re-43-five-containers
````

## Tiêu đề tệp pyc gồm bốn từ 32-bit

PEP 552 định nghĩa tiêu đề `.pyc` gồm bốn từ 32-bit (32-bit words). Từ đầu tiên là số ma thuật (magic number). Từ thứ hai là một trường bit (bit field). Bit 0 bật nghĩa là tệp hoạt động dựa trên mã băm (hash-based). Bit 1 bật biểu thị `check_source`: trình thông dịch sẽ băm mã nguồn lúc import và so sánh. Khi bit 0 tắt, từ thứ ba và thứ tư là mtime (thời gian sửa đổi) và kích thước của tệp nguồn. Khi bit 0 bật, chúng chứa mã băm SipHash 64-bit của mã nguồn.

Số ma thuật nằm trong tệp `Include/internal/pycore_magic_number.h`. Quy tắc từ bản 3.11 là Python 3.n bắt đầu từ giá trị `2900 + 50n`, và mỗi lần thay đổi bytecode trong chu kỳ phát hành sẽ cộng thêm 1. Giá trị lưu trữ là số 16-bit đi liền với ký tự `\r\n`, do đó nếu mở tệp `.pyc` ở chế độ văn bản (text mode) sẽ làm hỏng số ma thuật. Các giá trị chính thức là 3495 cho 3.11, 3531 cho 3.12, 3571 cho 3.13, 3627 cho 3.14 và 3666 cho 3.15. Python 3.15.0 được phát hành vào ngày 2026-10-09 và không được cài đặt trên máy dựng các fixture này, vì vậy số của nó được lấy từ tệp tiêu đề CPython chứ không phải từ tệp nhị phân trên máy.

Thư mục fixtures chứa hai tệp `.pyc` cho mỗi trình thông dịch. Tệp `compile_pyc.py` đã tạo ra chúng bằng lệnh `py_compile.compile(..., dfile="ledger.py")` để tên tệp lưu bên trong là `ledger.py` thay vì đường dẫn tuyệt đối của máy biên dịch. 16 byte đầu tiên của các tệp 3.14:

```text
checked-hash   2b 0e 0d 0a  03 00 00 00  c7 16 c7 fc c5 1d 81 01
timestamp      2b 0e 0d 0a  00 00 00 00  8d 60 cf 6a  f1 01 00 00
```

Hãy đọc từng từ trong tiêu đề của tệp timestamp. `03 00 00 00` hoặc `00 00 00 00` là từ cờ (flags word). `8d 60 cf 6a` là `0x6acf608d` = 1791975565, tức mtime của `ledger.py` khi tệp được biên dịch. `f1 01 00 00` là 497, kích thước tệp nguồn tính theo byte. Tệp checked-hash mang cờ giá trị 3 và 8 byte mã băm thay thế. Cả 4 trình thông dịch đều thống nhất về mọi thứ trong tiêu đề ngoại trừ từ ma thuật:

| Trình thông dịch | Các byte ma thuật | Từ số nguyên | Kích thước tệp (bytes) | Kích thước `grade.co_code` | Số lệnh của `grade` |
|---|---|---|---|---|---|
| 3.11.14 | `a7 0d 0d 0a` | 3495 | 1054 | 80 | 31 |
| 3.12.12 | `cb 0d 0d 0a` | 3531 | 937 | 62 | 26 |
| 3.13.12 | `f3 0d 0d 0a` | 3571 | 943 | 70 | 26 |
| 3.14.6 | `2b 0e 0d 0a` | 3627 | 965 | 96 | 35 |

Hàm `read_pyc_header()` trong `code/main.py` trả về các trường này và suy ra phiên bản từ từ số nguyên. `importlib.util.MAGIC_NUMBER` trả về cùng 4 byte đó cho trình thông dịch đang chạy. Lệnh `python3 code/main.py compare` sẽ biên dịch mã nguồn với mọi phiên bản từ `python3.11` đến `python3.15` tìm thấy trong `PATH`. Với mỗi phiên bản, nó in tiêu đề mà bộ đọc của chúng ta phân tích được bên cạnh các thông số tự báo cáo của chính trình thông dịch đó.

## Marshal ghi đối tượng mã theo từng trường

Byte thứ 16 của mọi tệp fixture là `e3`. Đó chính là `0x80 | ord('c')`: mã kiểu marshal cho một đối tượng mã (code object) với cờ `FLAG_REF` được bật, thông báo cho bộ đọc ghi nhớ đối tượng này để giải quyết các tham chiếu `r` phía sau. Tài liệu của marshal nêu rõ định dạng này cố tình không được chuẩn hóa thành tài liệu chính thức và các đối tượng mã không tương thích giữa các phiên bản. Cả hai nhận định đó đều đúng, nhưng định dạng này vẫn đủ tinh gọn để tự đọc được. Tệp `Python/marshal.c` liệt kê các mã kiểu: `N` cho None, `i` cho số nguyên 32-bit, `s` cho chuỗi byte, `z` và `Z` cho chuỗi ASCII ngắn, `(` và `)` cho tuple, `c` cho đối tượng mã, `r` cho tham chiếu ngược (back reference). Các vùng chứa sẽ dành sẵn khe tham chiếu cho chúng trước các phần tử con, do đó bộ đọc cũng làm tương tự.

Hàm `w_complex_object()` ghi một đối tượng mã theo một thứ tự cố định. Từ 3.11 đến 3.15, thứ tự đó là `argcount`, `posonlyargcount`, `kwonlyargcount`, `stacksize`, `flags` dưới dạng các số nguyên 32-bit, sau đó đến `code`, `consts`, `names`, `localsplusnames`, `localspluskinds`, `filename`, `name`, `qualname` dưới dạng các đối tượng, kế tiếp là `firstlineno`, rồi đến `linetable` và `exceptiontable`. Hãy tự giải mã đối tượng mã cấp mô-đun của tệp 3.14 bằng tay bắt đầu từ byte 17:

```text
00 00 00 00   argcount 0
00 00 00 00   posonlyargcount 0
00 00 00 00   kwonlyargcount 0
01 00 00 00   stacksize 1
00 00 00 00   flags 0
f3 22 00 00 00   kiểu 's' kèm FLAG_REF, theo sau là 34 byte của co_code
80 00  52 06  74 00  52 01  17 00  74 01 ...
```

Các byte mã bytecode gồm 2 byte cho mỗi lệnh: `80 00` là `RESUME 0`, `52 06` là `LOAD_CONST 6`, `74 00` là `STORE_NAME 0` và `17 00` là `MAKE_FUNCTION`. Bản phân tích 3.14 trong fixtures in ra đúng 5 câu lệnh cấp mô-đun này. Sau `code` là các hằng số (constants), nơi chứa 4 đối tượng mã lồng nhau, và sau đó là `names`: `('LABELS', 'percent_change', 'grade', 'total', 'grade_label')`.

Lớp `Unmarshaller` trong `code/main.py` đọc bố cục này cũng như bố cục cũ hơn từng dùng ở 3.9 và 3.10, nơi `nlocals`, `varnames`, `freevars` và `cellvars` là các trường tách biệt. Nó xây dựng một `CodeObject` mang các tên thuộc tính tương tự như đối tượng thực tế của Python. `varnames` lọc `localsplusnames` dựa trên bit `CO_FAST_LOCAL` của `localspluskinds`. Đối với hàm `grade` trong cả 4 tệp, nó đều trả về `('score', 'bucket')`, `firstlineno` bằng 10 và một `linetable` không rỗng. Bài kiểm tra `test_unmarshaller_agrees_with_the_running_interpreter` so sánh từng trường với `marshal.loads()` trên tệp được tạo bởi trình thông dịch đang chạy. Ba tệp còn lại hoàn toàn có thể đọc được bằng mã nguồn của chúng ta nhưng sẽ bị `marshal` của trình thông dịch hiện tại từ chối.

## Bốn trình thông dịch, bốn dạng bytecode cho hàm grade

Tài liệu của mô-đun `dis` luôn có một lời cảnh báo trên mọi phiên bản: bytecode chỉ là một chi tiết triển khai nội bộ của CPython và không có bất kỳ đảm bảo nào rằng nó sẽ giữ nguyên. Các fixture minh họa rõ điều này cho 11 dòng mã Python. Lệnh `python3.1N -m dis ledger.py` đã được ghi lại cho từng trình thông dịch.

Trong Python 3.11, hàm `grade` bắt đầu như sau:

```text
 11           2 LOAD_FAST                0 (score)
              4 LOAD_CONST               1 (10)
              6 BINARY_OP                2 (//)
             10 STORE_FAST               1 (bucket)
 12          12 LOAD_FAST                1 (bucket)
             14 LOAD_CONST               2 (9)
             16 COMPARE_OP               5 (>=)
             22 POP_JUMP_FORWARD_IF_FALSE     2 (to 28)
```

`BINARY_OP` nằm ở độ lệch 6 và lệnh tiếp theo nằm ở 10. Hai byte nằm giữa chúng là một mục bộ đệm nội tuyến (inline cache entry) từ PEP 659 (trình thông dịch thích ứng chuyên biệt hóa - specializing adaptive interpreter). `COMPARE_OP` ở độ lệch 16 được theo sau bởi 22, nghĩa là nó sở hữu 2 mục bộ đệm. Các mục đệm này bị ẩn đi trừ khi bạn truyền `show_caches=True`. Python 3.12 thay thế cặp `LOAD_CONST` cộng `RETURN_VALUE` bằng `RETURN_CONST`, đổi tên lệnh nhảy thành `POP_JUMP_IF_FALSE` và thu nhỏ hàm `grade` từ 80 xuống 62 byte. Bản 3.13 in các nhãn `L1` đến `L4` thay vì độ lệch và bổ sung cho CLI tùy chọn `-C` (bộ đệm) và `-O` (độ lệch). Bản 3.14 loại bỏ `RETURN_CONST`, thêm `LOAD_SMALL_INT` cùng `LOAD_FAST_BORROW`, đồng thời chèn thêm lệnh `NOT_TAKEN` phía sau mỗi lệnh nhảy có điều kiện:

```text
 12         20       LOAD_FAST_BORROW         1 (bucket)
            22       LOAD_SMALL_INT           9
            24       COMPARE_OP             188 (bool(>=))
            26       CACHE                    0 (counter: 0)
            28       POP_JUMP_IF_FALSE        3 (to L1)
            30       CACHE                    0 (counter: 0)
            32       NOT_TAKEN
```

Danh sách hiển thị `-C -O` cho thấy 13 mục `CACHE` bên trong hàm `grade`: 5 mục sau `BINARY_OP`, 1 mục sau mỗi `COMPARE_OP`, 1 mục sau mỗi `POP_JUMP_IF_FALSE`. Tệp fixture thích ứng (adaptive fixture) chứng minh các bộ đệm này dùng để làm gì. Sau 200 lượt gọi, `dis.dis(grade, adaptive=True)` in ra `RESUME_CHECK` và `COMPARE_OP_INT` tại những vị trí mà danh sách thông thường hiển thị `RESUME` và `COMPARE_OP`, trong khi hàm `total` xuất hiện `FOR_ITER_LIST`, `BINARY_OP_ADD_INT` và `JUMP_BACKWARD_NO_JIT`. Các dạng chuyên biệt hóa này nằm trên chính các byte `co_code` đó. Tệp `.pyc` không bao giờ chứa chúng, bởi vì quá trình chuyên biệt hóa diễn ra trực tiếp trong bộ nhớ RAM lúc thực thi.

Đây chính là lý do vì sao các trình dịch ngược luôn có mức giới hạn trần phiên bản (version ceilings). uncompyle6 3.9.3 tuyên bố chấp nhận bytecode từ Python 1.0 đến 3.8. decompyle3 3.9.3 đọc được 3.7 và 3.8. xdis 6.3.0, bộ dịch ngược đa phiên bản đứng sau cả hai công cụ trên, hỗ trợ đến 3.13. pycdc có bảng opcode đến `python_3_13.cpp` trong mã nguồn bản dựng của nó, và khi gặp một opcode lạ sẽ in `Unsupported opcode` rồi đánh dấu kết quả chưa hoàn tất. PyLingual hỗ trợ mọi phiên bản phát hành từ 3.6 và dùng các trình thông dịch từ 3.8 đến 3.15 để tái biên dịch và tự kiểm tra đầu ra của mình. Đối với tệp 3.14, ba công cụ đầu tiên sẽ từ chối hoặc thất bại, chỉ có hai công cụ sau là áp dụng được. Hãy luôn đọc từ ma thuật trước khi chọn công cụ phân tích.

## Tệp Class lưu giữ một từ điển về chính nó

Đặc tả JVM Java SE 25, mục 4.1, đưa ra cấu trúc tệp lớp gồm: `magic`, `minor_version`, `major_version`, `constant_pool_count`, vùng hằng số (constant pool), `access_flags`, `this_class`, `super_class`, các giao diện (interfaces), các trường (fields), các phương thức (methods), các thuộc tính (attributes). Số ma thuật là `0xCAFEBABE`. Máy ảo Java SE 25 chấp nhận phiên bản chính (major version) từ 45 đến 69, và phiên bản phụ (minor version) 65535 đánh dấu tệp sử dụng tính năng xem trước (preview-feature). Mười byte đầu tiên của `Ledger-g.class` là `ca fe ba be 00 00 00 41 00 5e`: major 65 là Java 21, và `0x5e` = 94 là số lượng phần tử trong vùng hằng số. Tệp `Ledger-jdk25.class`, được biên dịch bởi javac 25.0.2 từ cùng một nguồn, bắt đầu bằng `ca fe ba be 00 00 00 45`: major 69.

Vùng hằng số là nơi tệp class lưu trữ cuốn từ điển nội bộ của riêng mình. Mục `#13` trong bản dựng `-g` là một `Methodref` trỏ tới lớp `#8` và cặp tên-kiểu (name-and-type) `#14`. Lần theo các chỉ số này ta được `Ledger.grade:(I)I`, đúng bằng chuỗi mà `javap -v` in ra sau dấu `//`. Bộ mô tả (descriptor) `(I)I` biểu thị một tham số `int` và trả về kết quả kiểu `int`. `([I)I` ở hàm `total` biểu thị tham số là một mảng `int[]`. Mọi phương thức, trường và lớp mà mã chạm tới đều có một mục như vậy, và bộ kiểm chứng (verifier) trong mục 4.10 sẽ từ chối tải bất kỳ tệp nào có các lệnh mâu thuẫn với vùng hằng số. Lớp `ClassFile` trong `code/main.py` đọc toàn bộ 17 thẻ hằng số (constant tags) từ bảng 4.4 và xử lý `Long` hoặc `Double` như hai khe mục (slots). Nó sẽ báo lỗi khi gặp thẻ không xác định hoặc khi còn thừa byte sau thuộc tính cuối cùng.

Các thuộc tính gắn liền với lớp, từng trường, từng phương thức và chính bản thân thuộc tính `Code`. JVMS quy định máy ảo phải bỏ qua các thuộc tính mà nó không nhận biết, đó là lý do tại sao các bảng gỡ lỗi có thể được thêm vào hoặc loại bỏ mà không làm thay đổi phần thân hàm thực thi. Hướng dẫn sử dụng javac nêu rõ tùy chọn `-g` sẽ tạo toàn bộ thông tin gỡ lỗi bao gồm cả các biến cục bộ, mặc định chỉ tạo số dòng và tệp nguồn, còn `-g:none` thì không tạo gì. Ba bản dựng đo lường sự khác biệt đó:

| Bản dựng | Kích thước byte | Số mục constant pool | `LineNumberTable` | `LocalVariableTable` | `SourceFile` | Byte mã của `grade` |
|---|---|---|---|---|---|---|
| `javac -g` | 1832 | 94 | 7 phương thức | 6 phương thức | có | 50 |
| `javac` | 1581 | 82 | 7 phương thức | không có | có | 50 |
| `javac -g:none` | 1404 | 79 | không có | không có | không | 50 |

50 byte mã của hàm `grade` giống hệt nhau trên cả 3 tệp, và bài kiểm tra đã xác nhận điều này. Những gì `-g` thêm vào cho `grade` là một hàng trong `LocalVariableTable`: điểm bắt đầu 0, độ dài 50, khe (slot) 0, tên `score`, bộ mô tả `I`. Nếu không có bảng này, trình dịch ngược chỉ thấy khe 0 có kiểu `int` và bắt buộc phải tự bịa ra một cái tên biến. `StackMapTable` luôn có mặt trong mọi bản dựng vì bộ kiểm chứng yêu cầu nó, và thuộc tính `BootstrapMethods` được giữ lại vì hàm `main` nối chuỗi thông qua lệnh `invokedynamic`.

## Giải mã tableswitch bằng tay

Chương 6 của JVMS định nghĩa mỗi lệnh gồm một byte mã thao tác (opcode byte) và các toán hạng (operands) tương ứng. Các độ lệch nhánh (branch offsets) là các giá trị số nguyên có dấu 16-bit tương đối so với địa chỉ của chính opcode rẽ nhánh đó, khác với x86-64 nơi độ dịch chuyển tương đối so với lệnh kế tiếp. Các lệnh `tableswitch` và `lookupswitch` bắt đầu bằng từ 0 đến 3 byte đệm (padding bytes) để các toán hạng 32-bit của chúng bắt đầu tại một địa chỉ là bội số của 4.

Các byte của hàm `grade` bắt đầu từ pc 4 trong `Ledger-g.class`:

```text
aa                 tableswitch tại pc 4
00 00 00           đệm padding tới pc 8
00 00 00 2c        default: 4 + 44 = 48
00 00 00 06        low 6
00 00 00 0a        high 10
00 00 00 2a        case 6:  4 + 42 = 46
00 00 00 28        case 7:  4 + 40 = 44
00 00 00 26        case 8:  4 + 38 = 42
00 00 00 24        case 9:  4 + 36 = 40
00 00 00 24        case 10: 4 + 36 = 40
```

Ba mươi sáu byte, một lệnh duy nhất, thể hiện toàn bộ khối `switch (score / 10)` với nhánh gộp chung cho cả 9 và 10. Bản dựng mã máy gốc của cùng lệnh `switch` này trong bài học 33 cần một bước kiểm tra biên, một lệnh `lea`, một lệnh `movslq`, một lệnh `add`, một bước nhảy gián tiếp và 5 mục tái định vị (relocations) trong một phân đoạn khác. Ở đây, các mục tiêu nhảy nằm gọn ngay trong lệnh, cùng với thông tin rõ ràng rằng chúng thuộc về cùng một cấu trúc logic.

Hàm `disassemble_jvm()` hỗ trợ toàn bộ 202 mã thao tác đã được định nghĩa cùng độ rộng toán hạng của chúng, tuân thủ đúng quy tắc căn chỉnh đệm và kết xuất toán hạng tương tự như `javap`. Hàm `compare_with_javap()` phân tích cú pháp văn bản đầu ra thu được từ `javap -c -p`, bao gồm cả khối lệnh `tableswitch` nhiều dòng, rồi kiểm tra độ lệch, tên gợi nhớ (mnemonic) và toán hạng cho từng lệnh. Cả 4 bản dựng đều thống nhất trên 108/108 lệnh. Bài kiểm tra `test_branch_offsets_are_relative_to_the_branch_instruction` kiểm tra lệnh `ifne` tại pc 1 nhảy tới 6, `if_icmpge` tại 13 nhảy tới 33 và `goto` tại 30 nhảy quay ngược lại 10.

Công cụ radare2 6.2.4 biên dịch từ mã nguồn cũng đọc được tệp này. Lệnh `rabin2 -I` báo cáo `arch java`, `lang java 21`, `linenum true` và `lsyms true`, đó là kết quả đọc phiên bản và hai bảng gỡ lỗi của nó. Lệnh `pd 14 @ sym.Ledger.grade` in khối `tableswitch` kèm các mũi tên trỏ tới 5 đích nhảy. Nó cũng in thông báo `Unable to parse code attribute: 2` trên tệp này, đây là một giới hạn được công cụ tự nêu ra và là lý do chúng ta chọn `javap` làm chuẩn đối chiếu chân lý (oracle). Các trình dịch ngược cho định dạng này phát triển rất hoàn thiện so với Python. CFR 0.152 có giấy phép MIT. Procyon 0.6.0 và Vineflower 1.12.0 dùng giấy phép Apache-2.0, trong đó Vineflower là bản rẽ nhánh (fork) của Fernflower hỗ trợ Java 21. Bản thân Fernflower nằm trong kho lưu trữ IntelliJ Community. JADX 1.5.6 hỗ trợ định dạng Android DEX, còn JD-GUI có bản phát hành cuối vào năm 2019. Harrand, Soto-Valero, Monperrus và Baudry đã đo lường 8 công cụ dịch ngược vào năm 2020. Công cụ tốt nhất tạo ra đầu ra đúng cú pháp cho 84% các lớp và tương đương về mặt ngữ nghĩa cho 78%. Trình dịch ngược tổng hợp (meta-decompiler) của họ đã phục hồi được 37.6% các lớp mà không có công cụ đơn lẻ nào xử lý nổi. Các kiểu dữ liệu, định danh và ngăn xếp đã được kiểm chứng khiến định dạng này rất dễ dịch ngược. Tuy nhiên, điều đó không có nghĩa là bất kỳ một trình dịch ngược đơn lẻ nào cũng hoàn hảo tuyệt đối.

## Mô-đun Wasm là một danh sách các phân đoạn

Đặc tả cốt lõi định nghĩa một mô-đun gồm phần mở đầu (preamble) `00 61 73 6d 01 00 00 00` (chuỗi ký tự `\0asm` và phiên bản 1), theo sau là các phân đoạn (sections). Mỗi phân đoạn gồm một byte định danh id, kích thước `u32` mã hóa theo chuẩn LEB128 và nội dung tải trọng. Các id từ 1 đến 13 là các phân đoạn đã biết, mỗi phân đoạn xuất hiện tối đa một lần và theo đúng thứ tự. Phân đoạn mang id 0 là phân đoạn tùy chỉnh (custom section) có tên, được phép xuất hiện ở bất kỳ đâu. Tất cả các số nguyên đều dùng chuẩn LEB128, do đó `e4 00` là 100 và `80 08` là 1024. Lệnh `wasm-objdump -h` trên `ledger.wasm` liệt kê 7 phân đoạn, và `WasmModule` tìm thấy đúng 7 phân đoạn đó với độ lệch tải trọng hoàn toàn khớp:

| Phân đoạn | Vùng tải trọng (Payload) | Số lượng | Nội dung lưu trữ cho ledger |
|---|---|---|---|
| Type | 10 đến 26 | 3 | `(i32) -> nil`, `(i32, i32) -> i32`, `(i32) -> i32` |
| Import | 28 đến 39 | 1 | `env.log` với kiểu 0 |
| Function | 41 đến 46 | 4 | chỉ số kiểu cho từng hàm được định nghĩa |
| Memory | 48 đến 51 | 1 | một trang nhớ, không giới hạn tối đa |
| Export | 53 đến 110 | 5 | `memory` và tên của 4 hàm |
| Code | 113 đến 271 | 4 | thân hàm có kích thước 23, 63, 44 và 23 byte |
| Data | 273 đến 290 | 1 | `F\0D\0C\0B\0A\0` tại độ lệch 1024 |

Sự tách biệt giữa phân đoạn Function và phân đoạn Code giúp trình duyệt có thể bắt đầu biên dịch ngay trước khi toàn bộ tệp được tải xuống hoàn tất. Mỗi mục trong phân đoạn Code gồm kích thước, một vector khai báo biến cục bộ và một biểu thức. Đối với hàm `grade`, vector biến cục bộ là `01 7f`: một nhóm gồm một biến `i32`. Đó chính là biến `bucket`, giữ lại kiểu dữ liệu nhưng mất đi tên biến.

Luồng điều khiển (control flow) được tổ chức có cấu trúc theo thiết kế chuẩn. Đặc tả quy định các nhánh rẽ chỉ có thể nhắm tới các cấu trúc bao bọc gồm `block`, `loop` hoặc `if`. Các nhãn được đánh chỉ số theo độ sâu lồng nhau (nesting depth). Lệnh `br_table` thực hiện một bước nhảy gián tiếp thông qua toán hạng tra cứu chỉ số trong danh sách nhãn. Hãy giải mã cấu trúc switch của `grade` tại độ lệch tệp `0xa3`:

```text
0e                  br_table
0b                  11 nhãn theo sau
04 04 04 04 04 04   các bucket từ 0 đến 5: nhãn 4, khối ngoài cùng, trả về 0
03 02 01 00         các bucket 6, 7, 8, 9: nhãn 3, 2, 1, 0
00                  bucket 10: nhãn 0, dùng chung với 9
04                  nhãn mặc định (default) 4
```

Mỗi nhãn biểu thị một độ sâu. Nhãn 0 là khối `block` trong cùng, và mã lệnh sau khối `end` của nó là `i32.const 4` theo sau bởi `br 4` để thoát khỏi khối kết quả. Sáu lệnh `block` trước bảng, 11 nhãn và 5 lệnh `br` sau đó tạo thành toàn bộ cấu trúc `switch`, và bộ xác thực sẽ từ chối bất kỳ bước nhảy nào phá vỡ cấu trúc này. Hàm `disassemble_wasm()` theo dõi độ sâu lồng nhau để danh sách thụt lề chuẩn như `wasm-objdump`, hỗ trợ toàn bộ các opcode đơn byte từ `unreachable` đến `i64.extend32_s`, và khớp hoàn toàn với đầu ra `wasm-objdump -d` trên 77/77 lệnh cho cả hai bản dựng.

Tên định danh là thứ duy nhất bị loại bỏ trong một mô-đun cơ bản. `wasm-objdump` vẫn in được `<grade>` bởi vì phân đoạn Export ánh xạ chuỗi đó tới chỉ số hàm 2, và bộ đọc của chúng ta cũng làm tương tự. Các tên biến cục bộ không tồn tại ở bất kỳ đâu trong tệp 290 byte này. Tùy chọn `wat2wasm --debug-names` sẽ thêm một phân đoạn tùy chỉnh có tên là `name` với các phân đoạn phụ lưu tên mô-đun, tên hàm và tên biến cục bộ. Kích thước tệp khi đó tăng lên 425 byte. Bộ đọc tìm thấy `{0: 'score', 1: 'bucket'}` cho hàm 2 và `{0: 'values', 1: 'count', 2: 'sum', 3: 'i'}` cho hàm 3. Lệnh `wasm-strip` sẽ loại bỏ phân đoạn này đi. Kho lưu trữ `tool-conventions` mô tả thêm hai cơ chế mang thông tin: DWARF có thể được nhúng dưới dạng các phân đoạn tùy chỉnh `.debug_*`. Phân đoạn tùy chỉnh `sourceMappingURL` có thể trỏ tới một bản đồ nguồn (source map) có số cột tương ứng với các độ lệch byte trong tệp nhị phân.

Hệ sinh thái công cụ có sự thay đổi trong năm 2026. wabt 1.0.42 (phát hành 2026-09-15) đã loại bỏ công cụ `wasm-decompile`, và bản Homebrew trên máy này giữ lại chính xác 11 công cụ còn lại. Bộ wasm-tools 1.261.0 cung cấp `print`, `objdump` và `demangle`. Binaryen version_133 cung cấp `wasm-dis`. Công cụ radare2 6.2.4 cũng đọc được mô-đun này. Lệnh `rabin2 -I` hiển thị `arch wasm`. Biểu tượng của nó cho `grade` bắt đầu tại `0x8e`, nơi vector biến cục bộ bắt đầu, trong khi `wasm-objdump` gắn nhãn hàm tại `0x8b`, nơi trường kích thước bắt đầu. Danh sách lệnh của nó sử dụng các tên gợi nhớ cũ hơn gồm `get_local`, `set_local` và `brtable 11 4 4 ...` kèm theo số lượng nhãn được in ra. Các byte dữ liệu là hoàn toàn đồng nhất trên cả 3 công cụ dù cách biểu diễn chữ có khác biệt, do đó một phép so sánh bắt buộc phải ghi rõ công cụ nào đã được sử dụng. Ghidra không có sẵn mô-đun xử lý WebAssembly trong thư mục mặc định; tiện ích mở rộng bên thứ ba `ghidra-wasm-plugin` v2.4.0 hỗ trợ Ghidra 12.0. Bản thân đặc tả kỹ thuật là một tài liệu sống. Trang W3C cho `wasm-core-2` là Bản dự thảo khuyến nghị ứng viên ngày 13 tháng 10 năm 2026 mô tả bản phát hành 3.0, và bản thảo của các biên tập viên cũng mang cùng ngày.

## Siêu dữ liệu ECMA-335 là cơ sở dữ liệu nội tại của chương trình

Một assembly .NET bản chất là một tệp PE (Portable Executable). Tệp `LedgerNet.dll` bắt đầu bằng hai ký tự `MZ`, có 3 phân đoạn mang tên `.text`, `.rsrc` và `.reloc`, đồng thời khai báo kiểu máy `0x14c` vì trình biên dịch tạo ra mã trung gian độc lập với nền tảng phần cứng. Thư mục dữ liệu thứ 14 của tiêu đề tùy chọn (optional header) — tức CLR Runtime Header — trỏ tới địa chỉ RVA `0x2008`. Mục II.25.3.3 của chuẩn ECMA-335 định nghĩa 72 byte tại vị trí đó:

```text
48 00 00 00   Cb 72
02 00 05 00   MajorRuntimeVersion 2, MinorRuntimeVersion 5
78 21 00 00   MetaData RVA 0x2178
84 07 00 00   MetaData size 1924
01 00 00 00   Flags: COMIMAGE_FLAGS_ILONLY
05 00 00 06   EntryPointToken 0x06000005, MethodDef hàng 5, tức hàm Main
```

Gốc siêu dữ liệu (metadata root) tại RVA `0x2178` bắt đầu bằng chữ ký `42 53 4a 42` (đọc là chuỗi `BSJB`), tiếp theo là phiên bản 1.1 và chuỗi `v4.0.30319`. Năm tiêu đề luồng (stream headers) nối tiếp: `#~` tại độ lệch `0x6c`, rồi đến `#Strings`, `#US`, `#GUID` và `#Blob`. Mục II.24.2 quy định mỗi loại luồng chỉ xuất hiện tối đa một lần. Luồng `#~` bắt đầu bằng một byte `HeapSizes` (bằng 0 ở đây, nghĩa là mọi chỉ số trỏ vào chuỗi, GUID và blob đều rộng 2 byte), và một mặt nạ 64-bit `Valid` chỉ định các bảng có mặt. Mười bốn bit được bật:

| Bảng | Id | Số hàng | Kích thước hàng | Ý nghĩa các hàng |
|---|---|---|---|---|
| Module | 0x00 | 1 | 10 | tên mô-đun và GUID của nó |
| TypeRef | 0x01 | 21 | 6 | `System.Object`, `System.Console`, và 19 mục khác |
| TypeDef | 0x02 | 4 | 14 | `<Module>`, `Ledger`, và hai lớp trợ giúp của trình biên dịch |
| Field | 0x04 | 2 | 6 | `Labels` và bộ khởi tạo mảng tĩnh |
| MethodDef | 0x06 | 6 | 14 | 5 phương thức và hàm khởi tạo tĩnh `.cctor` |
| Param | 0x08 | 5 | 6 | `before`, `after`, `score`, `values`, `score` |
| MemberRef | 0x0a | 16 | 6 | 12 hàm dựng thuộc tính và 4 phương thức mà `Main` gọi |
| CustomAttribute | 0x0c | 12 | 6 | các thuộc tính do trình biên dịch và SDK bổ sung |
| StandAloneSig | 0x11 | 3 | 2 | ba chữ ký biến cục bộ |
| Assembly, AssemblyRef, ClassLayout, FieldRva, NestedClass | | 1, 2, 1, 1, 1 | | danh tính, tham chiếu, bố cục bộ nhớ |

Kích thước của một hàng không được lưu trữ sẵn ở bất kỳ đâu. Mục II.22 phân loại từng cột, và II.24.2.6 cung cấp quy tắc tính kích thước. Một chỉ số heap có độ rộng 2 hoặc 4 byte tùy thuộc vào `HeapSizes`. Một chỉ số bảng đơn giản chiếm 2 byte nếu bảng đó có dưới 65536 hàng. Một chỉ số mã hóa (coded index) có thể trỏ vào n bảng với b bit thẻ (tag bits) sẽ chiếm 2 byte chừng nào mọi bảng trong số đó đều có ít hơn 2^(16-b) hàng. Lớp `CliAssembly` tính toán kích thước của 14 hàng từ các quy tắc này và số lượng hàng. Bản trích xuất System.Reflection.Metadata trong thư mục fixtures báo cáo kích thước và độ lệch bảng hoàn toàn trùng khớp, chứng minh công thức số học của chúng ta là chính xác.

Mục III.1.9 định nghĩa cấu trúc thẻ bài định danh (token): byte cao là mã bảng, 3 byte thấp là số thứ tự hàng (bắt đầu từ 1). Giá trị `0x06000002` là hàng thứ 2 của bảng `MethodDef`. Hàng đó có RVA `0x2060`, tên `Grade` và một blob chữ ký `00 01 08 08`: quy ước gọi hàm mặc định, một tham số, kiểu trả về `int32`, tham số kiểu `int32`. Cột `ParamList` của nó trỏ vào hàng 3 của bảng `Param`, mang tên là `score`. Bảng Param chính là lý do vì sao ILSpy có thể in ra chữ ký `Grade(int score)` từ một bản dựng release đã bị lược bỏ biểu tượng gỡ lỗi. Tại RVA `0x2060`, thân phương thức bắt đầu bằng một tiêu đề béo (fat header):

```text
13 30         flags 0x13: fat, khởi tạo locals; kích thước 3 dwords
02 00         MaxStack 2
2d 00 00 00   CodeSize 45
01 00 00 11   LocalVarSigTok 0x11000001
```

Hàng 1 của `StandAloneSig` chứa `07 01 08`: `LOCAL_SIG`, một biến cục bộ, kiểu `int32`. Biến cục bộ này có kiểu nhưng không có tên. ECMA-335 lưu trữ tên biến cục bộ trong định dạng trao đổi gỡ lỗi ở Phần V, và .NET SDK ghi chúng ra tệp PDB. Bản fixture này được biên dịch với tùy chọn `DebugType` là none, nên không có bảng nào và không có tệp nào lưu lại tên `bucket`. Hàm `PercentChange` lại sử dụng tiêu đề rút gọn (tiny header): byte đầu tiên của nó là `3a` tương ứng với `0x0e << 2 | 2`, thân hàm dài 14 byte không có biến cục bộ và kích thước stack mặc định là 8.

Bốn mươi lăm byte mã của hàm `Grade` được giải mã theo bảng opcode ở Phần III. Lệnh `switch` mang opcode `0x45`, theo sau là một số đếm `u32` và bấy nhiêu đích nhảy `int32` tương đối so với lệnh nằm ngay sau bảng switch:

```text
IL_0000: 02           ldarg.0
IL_0001: 1f 0a        ldc.i4.s 10
IL_0003: 5b           div
IL_0004: 0a           stloc.0
IL_0005: 06           ldloc.0
IL_0006: 1c           ldc.i4.6
IL_0007: 59           sub
IL_0008: 45 05 00 00 00 08 00 00 00 06 00 00 00 04 00 00 00 02 00 00 00 02 00 00 00
                      switch (IL_0029, IL_0027, IL_0025, IL_0023, IL_0023)
IL_0021: 2b 08        br.s IL_002b
IL_0023: 1a 2a        ldc.i4.4, ret
```

Trình biên dịch đã trừ bớt 6 trước khi vào `switch`, do đó nhánh 6 là đích nhảy đầu tiên. Hàm `disassemble_cil()` diễn giải các token thông qua các bảng: `ldsfld Labels`, `call Grade`, `call System.Console::WriteLine` và `ldstr ' '` từ vùng heap `#US`. Bản chuẩn đối chiếu cho toàn bộ phần này là `MetaDump`, một chương trình C# ngắn trong fixtures được xây dựng trên `System.Reflection.Metadata`, thư viện nền tảng của ILSpy. Nó in ra số hàng, kích thước và độ lệch của từng bảng, RVA, chữ ký, tên tham số và các byte IL của từng phương thức, cùng 226 trường của `System.Reflection.Emit.OpCodes`. Hàm `compare_with_metadump()` khớp trên cả 246 phép kiểm tra. Bảng thời gian chạy của .NET có 218 opcode cộng thêm 8 mục giữ chỗ cho tiền tố `prefix`. Bảng opcode của bài học này có 219 mục vì nó bao gồm cả lệnh `no.` từ mục III.2.2 mà bảng Emit không có.

Các công cụ cho định dạng này: ILSpy 11.0 (MIT, phát hành 2026-08-12) chạy trên macOS qua bản Avalonia desktop. Lệnh `ilspycmd` 11.1.0.9782 in mã IL bằng cờ `-il` và chọn phương thức theo token bằng `-m 0x06000002`. dnSpyEx 6.6.0 tiếp tục dự án dnSpy chỉ trên Windows. Trình dịch ngược IL `ildasm` được phân phối dưới dạng các gói NuGet thay vì tích hợp sẵn trong SDK. radare2 6.2.4 cũng đọc được siêu dữ liệu: lệnh `rabin2 -I` hiển thị `arch cil` và `class PE32`, còn `is` liệt kê `Ledger.Grade(int:score):int` kèm tên tham số và cả hai kiểu dữ liệu lấy từ bảng Param và bảng chữ ký. Nó chưa tích hợp bộ dịch ngược CIL trong bản dựng này. Bản fixture này được biên dịch với .NET SDK 8.0.205 vì đó là phiên bản cài sẵn trên máy dựng. .NET 10 đã trở thành bản phát hành ổn định từ 2025-11-11 và cũng xuất ra các cấu trúc ECMA-335 tương tự.

## Những gì mỗi định dạng vùng chứa lưu giữ lại

Hãy đặt hàm `grade` trên cả 5 vùng chứa vào một bảng tổng hợp. Mọi hàng dữ liệu đều lấy từ lệnh `python3 code/main.py keeps`, thực thi các bộ đọc trên các tệp fixture:

| Vùng chứa | Kích thước thân hàm (bytes) | Tên hàm | Tên tham số | Kiểu tham số và giá trị trả về | Tên biến cục bộ | Số dòng nguồn | Cấu trúc luồng điều khiển |
|---|---|---|---|---|---|---|---|
| Tệp đối tượng ELF, clang -O0 | 118 | `grade` trong `.symtab` | không có | không có | không có | không có, không có các phân đoạn `.debug_*` | 5 bước nhảy và 1 bước nhảy gián tiếp qua bảng tái định vị |
| CPython 3.14 pyc | 96 | `co_qualname` | `score` | không có | `bucket` | `co_linetable`, dòng đầu tiên là 10 | 4 cặp so sánh và `POP_JUMP_IF_FALSE` |
| Lớp JVM, `javac -g` | 50 | vùng hằng số (constant pool) | `score` trong `LocalVariableTable` | `(I)I` | không có ngoài tham số | `LineNumberTable`, 6 hàng | 1 lệnh `tableswitch` |
| WebAssembly | 63 | phân đoạn Export | chỉ khi có phân đoạn name | `(i32) -> i32` | chỉ khi có phân đoạn name | không có | 6 khối lồng nhau, 1 `br_table`, 4 `br` |
| Assembly .NET | 45 | `MethodDef` | `score` trong `Param` | `static int32(int32)` | không có, kiểu chỉ có trong `StandAloneSig` | không có nếu thiếu PDB | 1 lệnh `switch`, 1 lệnh `br.s` |

Bảng này giải thích nguyên nhân đằng sau kết quả của các trình dịch ngược mà bạn gặp. Bài học 34 đã so sánh chữ ký gốc `int total(const int *, int)` với dạng `int total(long param_1, uint param_2)` mà Ghidra suy luận từ mã máy, bởi vì mã máy gốc không hề lưu giữ bất kỳ cột nào trong số này. Một tệp `.pyc` giữ lại mọi tên gọi nhưng không giữ kiểu, và luồng điều khiển của nó là các bước nhảy thay đổi cấu trúc sau mỗi bản phát hành, do đó các trình dịch ngược Python thường hỏng theo từng phiên bản nhưng lại khôi phục tên biến rất dễ dàng. Một tệp class lưu giữ kiểu dữ liệu và ngăn xếp đã kiểm chứng, giữ tên biến khi bật `-g`, và biểu diễn lệnh `switch` chỉ bằng một lệnh duy nhất. Một mô-đun Wasm bảo toàn cấu trúc logic ngay từ khâu thiết kế, do đó lệnh `wasm2wat` là một phép chuyển đổi khứ hồi không mất mát thông tin (lossless round trip) và thứ duy nhất bị mất là tên gọi. Một assembly .NET lưu giữ mô tả đầy đủ nhất vì các bảng siêu dữ liệu mang tính quy chuẩn bắt buộc, nhưng nó vẫn cần một tệp PDB để biết tên của biến cục bộ.

Bảng này cho bạn biết trình dịch ngược có những nguyên liệu gì để làm việc, và đầu ra nào của nó là sự sao chép một dữ kiện có sẵn trong tệp còn đầu ra nào là một suy đoán. Một tên tham số trong đầu ra của ILSpy là một phép sao chép nguyên trạng. Một tên biến cục bộ trong cùng đầu ra đó là một suy đoán vô căn cứ trừ khi có tệp PDB đi kèm. Bài học 35 sẽ ghi nhận sự khác biệt này dưới khái niệm độ tin cậy (authority) của từng khẳng định phân tích.

## Xây dựng thực tế

Chạy kịch bản minh họa:

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
```

Đầu ra là một báo cáo JSON duy nhất. Với mỗi tệp `.pyc`, nó cung cấp cả hai tiêu đề, kích thước tệp và kích thước hàm `grade`, số lượng lệnh và tên các opcode từ danh sách trích xuất, kèm theo việc trình thông dịch hiện tại có đọc được nó hay không. Với mỗi bản dựng class, nó cung cấp phiên bản, kích thước, số lượng bảng gỡ lỗi và mức độ khớp với `javap`. Với mỗi bản dựng Wasm, nó liệt kê các phân đoạn, kiểu dữ liệu, các hàm, nội dung phân đoạn name và mức độ khớp với `wasm-objdump`. Với assembly .NET, nó cung cấp tiêu đề CLI, các luồng, số hàng của bảng, các phương thức và độ khớp với System.Reflection.Metadata. Báo cáo kết thúc bằng bảng so sánh lưu giữ (keeps table) và danh sách các trình thông dịch tìm thấy trong `PATH`.

Hãy đọc mã nguồn trong `code/main.py` theo thứ tự sau:

1. Lớp `Reader` quản lý con trỏ đọc và các độ rộng số nguyên. `uleb()` và `sleb()` là các bộ đọc số LEB128 mà Wasm và vùng heap blob của .NET yêu cầu.
2. `read_pyc_header()`, `Unmarshaller` và `PycFile` dùng để đọc tệp `.pyc`. Hàm `parse_dis_text()` đọc các danh sách `dis` được trích xuất sẵn để quá trình kiểm tra đa phiên bản không đòi hỏi phải cài đủ 4 trình thông dịch tại thời điểm chạy test.
3. `ClassFile` đọc tệp lớp và `disassemble_jvm()` giải mã bytecode. `parse_javap_code()` và `compare_with_javap()` thực hiện nhiệm vụ kiểm chứng đối chiếu với oracle.
4. `WasmModule` đọc các phân đoạn và `disassemble_wasm()` giải mã biểu thức theo độ sâu lồng nhau. `compare_with_wasm_objdump()` thực hiện kiểm chứng đối chiếu.
5. `CliAssembly` duyệt qua PE, tiêu đề CLI, gốc siêu dữ liệu, các luồng và các bảng. `TABLE_SCHEMAS` và `CODED_INDEXES` là bản biểu diễn dữ liệu của các mục II.22 và II.24.2.6. `disassemble_cil()` giải mã thân hàm và `compare_with_metadump()` kiểm chứng nó.
6. `NativeObject` đọc cấu trúc ELF cơ bản để liệt kê biểu tượng hàm và lấy danh sách của bài học 33 làm văn bản đối chiếu. `what_each_keeps()` và `figure_data()` xây dựng bảng so sánh tổng hợp.

Công cụ hỗ trợ thêm 5 lệnh khác. `python3 code/main.py pyc FILE`, `class FILE`, `wasm FILE` và `dll FILE` in cấu trúc và danh sách lệnh của mọi hàm trong tệp được truyền vào. `python3 code/main.py compare` biên dịch `ledger.py` với mọi trình thông dịch có trong `PATH` và in ra những gì mỗi trình thông dịch đã ghi.

## Cách sử dụng

Sử dụng các bộ đọc theo thứ tự sau trên một vùng chứa bạn vừa biên dịch:

1. Đọc các byte đầu tiên để xác định định dạng. Hai byte theo sau là `0d 0a` là `.pyc`, `ca fe ba be` là tệp class, `00 61 73 6d` là mô-đun Wasm, `4d 5a` với thư mục dữ liệu 14 được thiết lập là assembly .NET.
2. Đọc tiêu đề và ghi lại phiên bản của trình biên dịch tạo ra nó. Đối với `.pyc` đó là từ ma thuật, đối với class là phiên bản major, đối với assembly là chuỗi phiên bản runtime và metadata.
3. Kiểm tra xem trình dịch ngược bạn định sử dụng có hỗ trợ phiên bản đó hay không trước khi thực thi. Ghi lại giới hạn phiên bản tối đa và vị trí tài liệu của công cụ nêu rõ điều đó.
4. Chạy bộ đọc tự viết và công cụ chuẩn của chính định dạng đó trên cùng một tệp rồi so sánh từng dòng. Bất kỳ sự lệch pha nào cũng là lỗi trong bộ đọc hoặc do công cụ thay đổi, và điều này phải được làm rõ trước khi sử dụng kết quả làm bằng chứng.
5. Điền thông tin vào bảng lưu giữ (keeps table) cho hàm mà bạn quan tâm. Đánh dấu từng dữ kiện là được lưu giữ (kept) hay bị mất (lost) và chỉ rõ tên bảng, thuộc tính hoặc phân đoạn lưu trữ nó.
6. Chỉ sau đó bạn mới đọc kết quả dịch ngược. Đánh dấu từng định danh và kiểu dữ liệu trong kết quả là sự sao chép một dữ kiện có sẵn hay là một suy đoán.

Hãy lưu các tệp văn bản chuẩn đối chiếu (oracle text files) cùng với các tệp vùng chứa. Khi phiên bản trình thông dịch, JDK, SDK hoặc wabt thay đổi, hãy biên dịch lại bằng `build_fixtures.sh` và ghi lại dữ liệu đối chiếu, bởi vì danh sách lệnh thay đổi đồng nghĩa với fixture đã thay đổi.

## Thực hành tương tác

### Bài thực hành A: đọc tiêu đề, sau đó đọc đối tượng mã

Từ thư mục `code`, mở môi trường Python:

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats/code
python3 -q
```

Thực thi:

```python
from main import *
pyc = load_pyc("ledger.cpython-311.pyc")
pyc.header.to_dict()
pyc.module.names
pyc.functions()["grade"].varnames
len(pyc.functions()["grade"].code)
```

Ghi lại từ ma thuật trước lệnh gọi đầu tiên. Sau đó tải tệp `ledger.cpython-314.pyc` và so sánh độ dài `code` của cả hai. Tiếp theo, thử gọi `import marshal; marshal.loads(pyc.data[16:])` trên tệp 3.11 trong trình thông dịch của bạn và quan sát thông báo lỗi.

### Bài thực hành B: chuỗi tra cứu vùng hằng số

```python
cls = load_class("Ledger-g.class")
cls.constant(13)
cls.constant(14)
cls.describe_constant(13)
[row["text"] for row in jvm_listing(cls, "grade")]
```

Lần theo mục `#13` tới `#8` và `#14`, sau đó tới các mục `Utf8` bằng tay trước khi bạn gọi hàm `describe_constant`. Sau đó tải `Ledger-gnone.class` và gọi `cls.method("grade").attribute_names()` trên cả hai. Mảng `Code` hoàn toàn giống nhau, nhưng các thuộc tính thì khác nhau.

### Bài thực hành C: độ sâu lồng nhau và các nhãn

```python
module = load_wasm("ledger.wasm")
[(row["depth"], row["text"]) for row in wasm_listing(module, "grade")]
module.function_name(2)
load_wasm("ledger-names.wasm").names["locals"][2]
```

Tìm dòng lệnh `br_table` và đọc độ sâu của nó. Đếm các dòng `end` phía sau và khớp từng dòng với một số nhãn trong bảng. Sau đó mở `ledger.wat` và tìm khối `block $six` mà nhãn 3 trỏ tới.

### Bài thực hành D: từ thẻ bài token đến các byte mã

```python
asm = load_assembly("LedgerNet.dll")
asm.method("Grade")
body = asm.method_body(asm.method("Grade")["RVA"])
body.fat, body.max_stack, body.code_size, hex(body.local_signature_token)
asm.token_text(0x11000001)
[row["text"] for row in cil_listing(asm, "GradeLabel")]
```

Tính toán độ lệch trong tệp của RVA `0x2060` từ tiêu đề phân đoạn `.text` bằng tay và so sánh với `asm.rva_to_offset(0x2060)`. Sau đó tra cứu `asm.row(0x08, 3)` và tìm vị trí mà tên `score` được lưu trữ.

## Bài tập rèn luyện

Bổ sung ngôn ngữ nguồn thứ 5 vào bảng so sánh, hoặc bổ sung dữ kiện thứ 6 vào bảng phân tích.

Yêu cầu:

- Viết một hàm chứa vòng lặp và một biến cục bộ bằng một ngôn ngữ mà bạn có sẵn trình biên dịch, ví dụ như Kotlin biên dịch ra tệp class hoặc Rust biên dịch ra mô-đun Wasm. Lưu tệp xuất ra và danh sách kết quả đối chiếu bằng câu lệnh chính xác trong `build_fixtures.sh`.
- Hoàn thiện bộ đọc tương ứng để nó khớp hoàn toàn với kết quả đối chiếu trên từng lệnh. Chỉ bổ sung opcode khi fixture thực sự yêu cầu.
- Mở rộng hàm `what_each_keeps()` với vùng chứa mới, và bổ sung thêm một dữ kiện cho mọi vùng chứa: liệu thân hàm có chứa bảng xử lý ngoại lệ (exception table) hay vùng `try` hay không.
- Bổ sung vùng chứa mới vào `figure_data()` và tạo lại tệp `site/figures-re-43.js`.
- Viết một bài kiểm tra tự động thất bại khi bộ đọc báo cáo một dữ kiện là được lưu giữ (kept) mà không chỉ rõ tên bảng, thuộc tính hoặc phân đoạn lưu giữ nó.

Hoàn thành khi `python3 code/main.py keeps` hiển thị hàng mới với đầy đủ bằng chứng cho từng ô dữ liệu, và mọi phép so sánh với kết quả đối chiếu đều khớp 100%.

## Sản phẩm bàn giao

Bài học này cung cấp tệp `outputs/skill-bytecode-and-managed-formats.md`. Sử dụng nó để biến một vùng chứa bytecode thành một báo cáo tiêu đề, một danh sách lệnh đã kiểm chứng của một hàm và một bảng dữ kiện đánh dấu rõ từng tên, kiểu và cấu trúc là được lưu giữ hay bị mất. Nó đòi hỏi công cụ chuẩn của chính định dạng đó làm căn cứ đối chiếu, kiểm tra phiên bản trước khi chạy bất kỳ trình dịch ngược nào và nêu rõ vị trí mà mỗi dữ kiện bị mất đáng lẽ đã nằm ở đó.

## Kiểm chứng tính đúng đắn

Chạy kịch bản minh họa và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/43-bytecode-and-managed-formats
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình kiểm chứng cần chứng minh:

- 4 tiêu đề `.pyc` mang các số ma thuật 3495, 3531, 3571 và 3627, cờ giá trị 3 và một mã băm 8 byte; các tệp timestamp mang cờ 0, mtime 1791975565 và kích thước 497
- Mọi từ ma thuật từ 3.11 trở đi đều nằm đúng khoảng giá trị mà quy tắc `2900 + 50n` quy định cho bản phát hành tương ứng
- Bộ giải tuần tự hóa (unmarshaller) trả về cùng các tên biến, hằng số, byte mã, danh sách tên biến và bảng dòng tương tự như `marshal.loads()` trên tệp do trình thông dịch đang chạy tạo ra
- Cả 4 tệp đều giữ lại tên mô-đun, `('score', 'bucket')` cho `grade`, dòng bắt đầu là 10 và tên tệp là `ledger.py`
- `grade.co_code` có kích thước lần lượt là 80, 62, 70 và 96 byte qua 4 phiên bản và danh sách lệnh thu được có 31, 26, 26 và 35 lệnh
- Danh sách 3.11 có lệnh `POP_JUMP_FORWARD_IF_FALSE`, 3.12 và 3.13 có `RETURN_CONST`, bản 3.14 có `LOAD_SMALL_INT`, `LOAD_FAST_BORROW` và `NOT_TAKEN`
- Danh sách 3.14 với `-C -O` cho thấy 13 mục bộ đệm trong `grade`, 5 trong số đó nằm sau `BINARY_OP`, và danh sách thích ứng hiển thị `COMPARE_OP_INT`
- Tiêu đề thiếu `0d 0a`, tiêu đề bị cắt cụt và thân tệp không phải đối tượng mã sẽ phát sinh lỗi `FormatError`
- Danh sách lệnh chỉ hiển thị cho tệp `.pyc` được biên dịch bởi trình thông dịch đang chạy và khớp với đầu ra `dis` đã thu thập
- Các tệp class có phiên bản major 65 hoặc 69, kích thước 1581, 1832, 1404 và 1581 byte, số lượng mục trong constant pool lần lượt là 82, 94, 79 và 82
- Hằng số `#13` phân giải thành `Method Ledger.grade:(I)I` và `#47` phân giải thành `String F`
- Bộ dịch ngược JVM khớp với `javap -c -p` trên 108/108 lệnh trong cả 4 bản dựng
- Khối `tableswitch` tại pc 4 dài 36 byte với low là 6, high là 10, default là 48 và các đích nhảy là 46, 44, 42, 40, 40
- Ba mức gỡ lỗi làm thay đổi số lượng thuộc tính nhưng giữ nguyên 50 byte mã của `grade`
- Các đích nhảy của `ifne`, `if_icmpge` và `goto` đều tương đối so với opcode rẽ nhánh
- Số ma thuật sai, byte thừa ở cuối và opcode không xác định sẽ phát sinh `FormatError`
- Bảy phân đoạn Wasm có độ lệch tải trọng trùng khớp với những gì `wasm-objdump -h` in ra, và bản dựng names bổ sung thêm phân đoạn tùy chỉnh `name` tại vị trí 293 đến 425
- Các kiểu dữ liệu, import `env.log`, 5 export và đoạn dữ liệu tại 1024 giải mã đúng như khai báo trong `ledger.wat`
- Bộ dịch ngược Wasm khớp với `wasm-objdump -d` trên 77/77 lệnh trong cả hai bản dựng
- Hàm `grade` có 6 khối lồng nhau, một lệnh `br_table` với các nhãn `4 4 4 4 4 4 3 2 1 0 0` và default là 4 ở độ sâu 6, cùng một biến cục bộ `i32`
- Tên hàm lấy từ phân đoạn export khi không có phân đoạn name và tên biến cục bộ chỉ có khi có phân đoạn name
- LEB128: `e4 00` là 100, `7f` là -1, `80 08` là 1024, và các kiểu khối hiển thị giống như `wasm-objdump`
- Phần mở đầu sai, phiên bản 2 và opcode không xác định sẽ phát sinh `FormatError`
- Assembly là PE32 với machine `0x14c`, tiêu đề CLI tại RVA `0x2008`, runtime 2.5, cờ `ILONLY` và điểm vào `0x06000005`
- Phiên bản metadata là `v4.0.30319`, có 5 luồng và 14 bảng có số hàng cùng kích thước hàng đúng như bản đối chiếu báo cáo
- Bộ đọc .NET khớp với System.Reflection.Metadata trên 246/246 phép kiểm tra
- Phương thức `Grade` có tham số `score`, chữ ký `static int32(int32)`, tiêu đề béo với `MaxStack` là 2 và 45 byte mã, cùng một biến cục bộ kiểu `int32` không có tên
- Các toán hạng của `switch`, `br.s`, `ldsfld`, `call` và `ldstr` trong CIL phân giải chính xác qua các bảng và các vùng heap
- Bảng opcode CIL bao quát mọi opcode thời gian chạy và chỉ bổ sung thêm lệnh `no.`
- Bảng lưu giữ báo cáo tên tham số có trong 4 vùng chứa và không có trong mã máy gốc, kiểu dữ liệu có trong 3 vùng chứa
- Mọi dòng đối chiếu trong kịch bản demo đều hiển thị n trên n
- Sơ đồ tương tác nhúng đúng dữ liệu mà `figure_data()` tạo ra

## Các chế độ thất bại trong thực tế

| Thất bại | Những gì công cụ yếu báo cáo | Những gì phân tích phải chứng minh |
|---|---|---|
| Trình dịch ngược cũ hơn trình thông dịch | `Unsupported opcode`, kết quả bị đánh dấu chưa hoàn thành, hoặc khẳng định mã bị làm rối | Số ma thuật, phiên bản phát hành tương ứng, và giới hạn phiên bản tối đa được ghi nhận của công cụ |
| Đọc `.pyc` bằng sai phiên bản trình thông dịch | `marshal.loads` phát sinh lỗi hoặc trả về dữ liệu rác | Bộ giải tuần tự hóa của chúng ta đọc tên, hằng số và kích thước từ bất kỳ tệp 3.9 đến 3.15 nào, và lệnh được lấy từ `dis` tương ứng |
| Nhầm lẫn opcode chuyên biệt hóa với nội dung tệp | `COMPARE_OP_INT` xuất hiện trong danh sách lệnh | Tệp `.pyc` chỉ lưu dạng chưa chuyên biệt hóa, việc chuyên biệt hóa chỉ diễn ra trên RAM sau giai đoạn khởi động |
| Báo cáo tên biến cục bộ cho bản dựng javac mặc định | Hiển thị `var1`, `var2` như thể đó là tên trong mã nguồn | `LocalVariableTable` chỉ có mặt khi bật `-g`, 50 byte mã lệnh hoàn toàn giống nhau trong cả hai trường hợp |
| Đích nhảy bị lệch một khoảng bằng độ dài lệnh | Danh sách lệnh JVM có các bước nhảy rơi vào giữa một lệnh | Độ lệch tương đối so với opcode của lệnh rẽ nhánh, được kiểm chứng đối chiếu với `javap` |
| Bỏ qua byte đệm căn chỉnh của switch | Lệnh `tableswitch` có nhánh default sai và các case bị dịch chuyển | Có từ 0 đến 3 byte đệm để đạt địa chỉ pc chia hết cho 4, 36 byte cho 5 nhánh case |
| Nhầm tưởng Wasm bị tước bỏ toàn bộ tên | "không có biểu tượng" (no symbols) | Phân đoạn Exports lưu tên của mọi hàm được xuất ra, biến cục bộ cần phân đoạn `name`, `wasm-strip` chỉ loại bỏ phân đoạn đó |
| Ba công cụ, ba cách viết khác nhau | `get_local` so với `local.get`, hàm bắt đầu tại `0x8b` so với `0x8e` | Cùng một chuỗi byte, và danh sách lệnh phải ghi rõ công cụ đã dùng để trích xuất |
| Đoán mò kích thước hàng của bảng metadata | Mọi hàng nằm sau bảng đầu tiên đều bị đọc sai | Kích thước hàng suy ra từ `HeapSizes`, số lượng hàng và độ rộng chỉ số mã hóa, trùng khớp với bản đối chiếu |
| Kỳ vọng có tên biến cục bộ trong assembly release | `V_0` được hiển thị như tên biến trong mã nguồn | Biến cục bộ có kiểu trong `StandAloneSig` và chỉ có tên trong tệp PDB mà `DebugType` none không tạo ra |
| Bộ đọc tự đoán một opcode hoặc mã kiểu không rõ | Một tên gợi nhớ có vẻ hợp lý | Báo lỗi `FormatError` kèm độ lệch byte, và dừng ngay phép so sánh đối chiếu |
| Bản đối chiếu lấy từ một bản dựng khác | Các bài kiểm tra vượt qua trên dữ liệu văn bản cũ | Kịch bản `build_fixtures.sh` ghi rõ phiên bản của từng công cụ và lưu vào `versions.txt` cùng `sha256.txt` |

## Quy tắc vận hành

Hãy luôn đọc tiêu đề tệp trước tiên và xác định rõ công cụ biên dịch đã tạo ra nó. Đọc vùng chứa bằng một bộ đọc có cơ chế từ chối những gì nó không nhận biết, và đối chiếu danh sách lệnh của nó với chính công cụ chuẩn của định dạng đó trên cùng một tệp. Sau đó, với từng tên gọi và kiểu dữ liệu xuất hiện trong đầu ra của trình dịch ngược, hãy tuyên bố rõ ràng liệu vùng chứa thực sự lưu giữ thông tin đó hay do công cụ tự suy đoán ra.

## Tài liệu đọc thêm

- [PEP 552: Deterministic pycs](https://peps.python.org/pep-0552/)
- [CPython `Include/internal/pycore_magic_number.h`](https://github.com/python/cpython/blob/main/Include/internal/pycore_magic_number.h)
- [Tài liệu mô-đun `marshal` của Python](https://docs.python.org/3/library/marshal.html)
- [CPython `Python/marshal.c`](https://github.com/python/cpython/blob/main/Python/marshal.c)
- [Tài liệu mô-đun `dis` của Python 3.14](https://docs.python.org/3.14/library/dis.html)
- [PEP 659: Specializing Adaptive Interpreter](https://peps.python.org/pep-0659/)
- [PEP 626: Số dòng chính xác cho gỡ lỗi và các công cụ khác](https://peps.python.org/pep-0626/)
- [PEP 790: Kế hoạch phát hành Python 3.15](https://peps.python.org/pep-0790/)
- [uncompyle6 trên PyPI](https://pypi.org/project/uncompyle6/), [decompyle3 trên PyPI](https://pypi.org/project/decompyle3/), [xdis trên PyPI](https://pypi.org/project/xdis/), [pycdc trên GitHub](https://github.com/zrax/pycdc), [PyLingual trên GitHub](https://github.com/syssec-utd/pylingual)
- [Đặc tả máy ảo Java (JVMS), Java SE 25, Chương 4: Định dạng tệp class](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-4.html)
- [Đặc tả máy ảo Java (JVMS), Java SE 25, Chương 6: Tập lệnh máy ảo Java](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-6.html)
- [Hướng dẫn sử dụng javac, Java SE 25](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javac.html) và [Hướng dẫn sử dụng javap](https://docs.oracle.com/en/java/javase/25/docs/specs/man/javap.html)
- [CFR](https://www.benf.org/other/cfr/), [Procyon](https://github.com/mstrobel/procyon), [Vineflower](https://github.com/Vineflower/vineflower), [Fernflower trong intellij-community](https://github.com/JetBrains/intellij-community/tree/master/plugins/java-decompiler/engine), [JADX](https://github.com/skylot/jadx)
- [Harrand, Soto-Valero, Monperrus, Baudry, "Java Decompiler Diversity and its Application to Meta-decompilation" (JSS 2020)](https://arxiv.org/abs/2005.11315)
- [ECMA-335, Common Language Infrastructure, Phiên bản 6 (Tháng 6 năm 2012)](https://ecma-international.org/publications-and-standards/standards/ecma-335/)
- [Định dạng PE, Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [System.Reflection.Metadata.Ecma335.TableIndex](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.metadata.ecma335.tableindex)
- [ILSpy trên GitHub](https://github.com/icsharpcode/ILSpy), [ilspycmd trên NuGet](https://www.nuget.org/packages/ilspycmd), [dnSpyEx trên GitHub](https://github.com/dnSpyEx/dnSpy)
- [Đặc tả WebAssembly Core: Định dạng nhị phân, Mô-đun](https://webassembly.github.io/spec/core/binary/modules.html) và [Giá trị](https://webassembly.github.io/spec/core/binary/values.html)
- [Đặc tả WebAssembly Core: Các phân đoạn tùy chỉnh (phân đoạn name)](https://webassembly.github.io/spec/core/appendix/custom.html)
- [Quy ước công cụ WebAssembly: Gỡ lỗi](https://github.com/WebAssembly/tool-conventions/blob/main/Debugging.md)
- [Ghi chú phát hành wabt 1.0.42](https://github.com/WebAssembly/wabt/releases/tag/1.0.42), [wasm-tools](https://github.com/bytecodealliance/wasm-tools), [Binaryen](https://github.com/WebAssembly/binaryen), [ghidra-wasm-plugin](https://github.com/nneonneo/ghidra-wasm-plugin)
- [radare2 trên GitHub](https://github.com/radareorg/radare2), phiên bản 6.2.4 biên dịch từ mã nguồn cho bài học này
