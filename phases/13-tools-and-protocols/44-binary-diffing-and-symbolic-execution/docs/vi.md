# So khớp nhị phân và Thực thi biểu trưng: Khớp hàm và Giải tìm đầu vào

> Một kết quả khớp giữa hai hàm là một khẳng định do một chữ ký cụ thể đưa ra với một mức chất lượng xác định, và một đầu vào giải được là một mô hình nghiệm của một điều kiện đường đi; bài học này xây dựng cả hai cơ chế trên và kiểm chứng chúng đối chiếu với radiff2, Ghidra, angr và z3.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python, C, mã assembly x86-64, Java (một kịch bản Ghidra)
**Yêu cầu tiên quyết:** Phase 13 · 33 (dịch ngược mã assembly và luồng điều khiển), 34 (nâng cấp mã assembly thành mã giả), 36 (tái biên dịch và chứng minh tính tương đương)
**Thời gian:** ~150 phút

## Mục tiêu học tập

- Xây dựng 5 loại chữ ký hàm (function signatures) từ một tệp đối tượng và danh sách lệnh của nó: byte chính xác (exact bytes), byte che mặt nạ trường tái định vị (masked bytes), chuỗi tên gợi nhớ (mnemonic sequence), hình dạng đồ thị luồng điều khiển (flow-graph shape) và tên biểu tượng (symbol name).
- Vận hành một bộ khớp tham lam theo từng giai đoạn (staged greedy matcher) theo phong cách BinDiff, trong đó một chữ ký chỉ được coi là khớp khi nó là duy nhất ở cả hai phía, đồng thời ghi lại giai đoạn và mức chất lượng đằng sau mỗi cặp khớp.
- Tái hiện trường hợp hai hàm có mã byte giống hệt nhau nhưng lại thất bại trước bộ tương quan byte (byte correlator) trên ảnh bộ nhớ đã nạp (loaded images) và chỉ vượt qua sau khi áp dụng mặt nạ che trường tái định vị, kèm theo công thức số học tái định vị giải thích rõ 4 byte bị biến đổi.
- Tái hiện tỷ lệ tương đồng (similarity ratios) của `radiff2` từ khoảng cách Levenshtein và giải thích vì sao một tỷ lệ chỉ xuất hiện khi phép so sánh sử dụng đúng 82 byte mà radare2 duyệt tới.
- Liệt kê các đường đi thực thi (paths) của một hàm trên một biểu diễn trung gian (IR) nhỏ, duy trì một điều kiện cho mỗi đường đi và giải tìm đầu vào cho từng nút lá (leaf) bằng thuật toán tìm kiếm có giới hạn (bounded search).
- Đo lường hiện tượng bùng nổ đường đi (path explosion) trên một vòng lặp có cận biểu trưng và so sánh số lượng, các lớp nghiệm cùng mô hình nghiệm với angr 10.0.3 và z3 5.1.0.

## Vấn đề thực tế

Bạn đã biên dịch lại `ledger.c` ở mức tối ưu `-O2` và nhờ một AI agent kiểm tra xem có thay đổi gì so với bản dựng `-O1` hay không. Agent đó đã chạy một công cụ so khớp nhị phân (differ) và báo cáo rằng hàm `grade` đã bị sửa đổi, bởi vì các byte của nó khác nhau trong hai chương trình khi nạp vào bộ nhớ. Sau đó, nó đã lãng phí tài nguyên để giải thích một sự thay đổi vốn dĩ không hề tồn tại. Thực tế, 46 byte mã của hàm `grade` hoàn toàn giống hệt nhau trong cả hai tệp đối tượng nhị phân. Công cụ differ đã so sánh các ảnh nhị phân sau khi đã áp dụng tái định vị (relocation), và phân đoạn `.rodata` nằm cách xa hơn `0x70` byte trong chương trình `-O2`, khiến cho một trường độ lệch (displacement) trong `grade` bị tăng thêm đúng `0x70`.

Cũng trong buổi chiều hôm đó, bạn hỏi giá trị điểm số nào sẽ khiến hàm `grade` trả về 4. Agent đọc mã giả và trả lời là "90". Đó chỉ là một câu trả lời đơn lẻ. Hàm thực tế trả về 4 cho mọi điểm số từ 90 đến 109, và câu hỏi vốn không đặt ra giới hạn biên nào, do đó câu trả lời của agent chỉ là một mẫu đơn lẻ trong khi bạn cần một tập hợp nghiệm đầy đủ.

Cả hai thất bại trên đều bắt nguồn từ cùng một lỗ hổng phương pháp luận. Một kết quả khớp và một đầu vào giải được bản chất đều là những khẳng định đi kèm một phương pháp luận cụ thể phía sau. Bài học này xây dựng hai phương pháp đó bằng thư viện chuẩn của Python và vận hành các công cụ chuẩn công nghiệp song song bên cạnh chúng. Khi công cụ và mã nguồn của bạn thống nhất, bạn có trong tay hai cách dẫn xuất độc lập. Khi chúng bất đồng, bạn sẽ có một địa chỉ, một byte cụ thể hoặc một cận biên rõ ràng để đối chiếu và phân tích.

````figure
re-44-matches-and-paths
````

## Chữ ký hàm là một khẳng định đi kèm chất lượng

Flake đã định hình bài toán so khớp nhị phân vào năm 2004 như là việc xây dựng một phép đẳng cấu (isomorphism) giữa các tập hợp hàm của hai tệp thực thi tương tự nhau. Dullien và Rolles đã mở rộng phương pháp này vào năm 2005 xuống cấp độ các khối cơ bản (basic blocks) và lệnh máy. Trang tài liệu khái niệm của BinDiff mô tả chiến lược cốt lõi bắt nguồn từ các công trình này: Tính toán một thuộc tính cho mọi hàm ở cả hai phía. Một giá trị xuất hiện chính xác một lần ở mỗi phía sẽ tạo thành một cặp khớp. Một giá trị xuất hiện nhiều hơn một lần là mơ hồ và sẽ được tinh lọc sâu hơn bằng thuộc tính tiếp theo bên trong lớp tương đương đó. Một giá trị không có đối ứng sẽ được giữ lại ở trạng thái chưa khớp. Sau mỗi lượt duyệt, tập hợp các hàm gọi (callers) và hàm được gọi (callees) của các cặp khớp mới sẽ tạo thành các tập hợp con nhỏ hơn, nơi thuộc tính kế tiếp có khả năng cao là duy nhất, và quy trình lặp lại với thuộc tính tiếp theo.

Mỗi thuật toán so khớp đều mang một mức chất lượng (quality) xác định. Trang tài liệu đánh giá việc so khớp băm byte thô của hàm và băm tên hàm là mức rất tốt (very good), chỉ số MD của đồ thị luồng điều khiển và chữ ký số nguyên tố (mỗi opcode là một số nguyên tố nhỏ, tích phân bố trên toàn hàm) là mức tốt (good), tham chiếu chuỗi là trung bình (medium), còn số lượng vòng lặp hay chuỗi địa chỉ là kém (poor). Độ tin cậy (confidence) của một cặp khớp là chất lượng trung bình của các thuật toán đã tạo ra nó, được cân số học qua hàm sigmoid. Độ tương đồng của hàm (function similarity) được tính xấp xỉ bằng: 25% cạnh đồ thị luồng khớp nhau, 15% khối cơ bản khớp, 10% lệnh máy khớp và 50% độ chênh lệch chỉ số MD đồ thị luồng, nhân với độ tin cậy.

Bài học này giữ lại 5 trong số các thuộc tính đó cùng một quy tắc tối thượng: duy nhất ở cả hai phía thì mới tính là khớp, nếu không thì không khớp. Bài học lược bỏ chỉ số MD (vì tài liệu chỉ nêu tên mà không đưa ra công thức toán học) và dùng một chuỗi bậc đỉnh (degree sequence) rõ ràng để thay thế. Các nhãn chất lượng được giữ nguyên, bởi vì một cặp khớp chỉ đáng tin cậy bằng chính dữ kiện yếu nhất đã tạo ra nó.

## Năm chữ ký từ một bản ghi hàm

Bộ so khớp đọc hai thành phần từ mỗi tệp đối tượng. Tệp ELF cung cấp các byte, kích thước biểu tượng và các bản ghi `.rela.*`. Danh sách lệnh từ `llvm-objdump -d` cung cấp ranh giới lệnh và tên gợi nhớ, và bài học 33 đã chứng minh bộ giải mã của chúng ta hoàn toàn khớp với danh sách này trên từng lệnh. Hàm `function_record()` kiểm tra xem các byte của từng lệnh trong danh sách có bằng đúng các byte của tệp đối tượng tại địa chỉ đó hay không, nếu không sẽ phát sinh lỗi. Một danh sách bị lệch đồng nghĩa với việc fixture đã bị thay đổi.

Xét hàm `grade` trong `ledger-x86_64-O1.o`. Biểu tượng chỉ ra độ lệch là `0x10` và kích thước là 46, vì vậy bản ghi bao trùm từ `0x10` đến `0x3e` và loại bỏ 2 byte `nop` dùng để căn chỉnh hàm kế tiếp.

Chữ ký byte chính xác (exact-bytes signature) là mã băm SHA-256 của toàn bộ 46 byte đó. Chữ ký byte che mặt nạ (masked-bytes signature) sẽ gán giá trị 0 cho mọi trường tái định vị trước khi băm. Ở đây có một trường: bản ghi `R_X86_64_PC32 .rodata+0x10` tại độ lệch `0x36`, rộng 4 byte, là trường độ lệch của lệnh `leaq (%rip), %rcx` tại `0x33`. Trong một tệp đối tượng RELA, 4 byte đó vốn đã chứa số 0, do đó hai chữ ký này hoàn toàn trùng khớp trên mọi hàm của mọi tệp đối tượng tại đây. Mặt nạ này được thiết kế để phục vụ cho phần phân tích tiếp theo.

Chữ ký tên gợi nhớ (mnemonic signature) là mã băm của chuỗi 14 tên lệnh:

```text
movslq imulq movq sarq shrq addl addl xorl cmpl ja movl leaq movl retq
```

Chữ ký hình dạng đồ thị luồng điều khiển (flow-graph shape signature) được trích xuất từ các khối cơ bản và các cạnh có định kiểu từ bài học 33, giới hạn trong phạm vi các khối mà điểm vào có thể duyệt tới. Hàm `grade` ở `-O1` có 3 khối và 3 cạnh. Khối `0x10` kết thúc bằng `ja 0x3d` với một cạnh nhảy và một cạnh rơi tự do (fallthrough), khối `0x31` rơi tự do vào `0x3d`, và `0x3d` thực hiện trả về. Chuỗi bậc đỉnh liệt kê cặp (bậc vào, bậc ra) cho từng khối theo thứ tự đã sắp xếp, do đó chữ ký có dạng `3b 3e [0/2,1/1,2/0]`. Đối với `grade` ở `-O0`, bản ghi phân giải bước nhảy gián tiếp thông qua 5 mục trong `.rela.rodata` đúng như bài học 33 đã làm, tạo ra 8 khối và 11 cạnh với 4 cạnh thuộc bảng nhảy (`table`).

Chữ ký tên (name signature) chính là tên biểu tượng của hàm. Đây là dữ kiện mạnh mẽ nhất khi nó tồn tại, và hoàn toàn vô dụng khi tệp nhị phân bị tước bỏ biểu tượng (stripped).

Thêm một trường dữ liệu được theo dõi để phục vụ trường hợp phân tích bản vá (patch) sau này: Khi một hàm có một mục tái định vị `lea` trỏ vào `.rodata`, một phép kiểm tra biên và một phép nạp chỉ số mảng, hàm `referenced_table()` sẽ đọc bảng dữ liệu theo cách của bài học 36: lệnh `cmpl $0x4` trước lệnh `ja` cho biết bảng có 5 phần tử và lệnh nạp `movl (%rcx,%rax,4), %eax` cho biết mỗi phần tử rộng 4 byte. Với `grade` ở `-O1`, bảng tại `.rodata+0x14` có các giá trị là `1, 2, 3, 4, 4`.

## Bộ so khớp tham lam

Hàm `match_functions()` thực thi các giai đoạn theo thứ tự định sẵn. Tại mỗi giai đoạn, nó nhóm các hàm chưa được khớp ở mỗi phía theo chữ ký của giai đoạn đó, và chấp nhận mọi giá trị xuất hiện chính xác một lần ở cả hai bên. Bản ghi khớp lưu lại giai đoạn, mức chất lượng của giai đoạn, danh sách các chữ ký khác cũng đồng thuận và tỷ lệ tương đồng.

```python
for stage in order:
    for signature, candidates in left_groups.items():
        partners = right_groups.get(signature, [])
        if len(candidates) != 1 or len(partners) != 1:
            continue
        a, b = candidates[0], partners[0]
        agreeing = [s.id for s in STAGES if a.signatures()[s.id] == b.signatures()[s.id]]
        matches.append(Match(a, b, stage, agreeing, radare2_ratio(a.raw, b.raw), data_equal))
```

Thứ tự của bài học chạy 4 giai đoạn cấu trúc trước và tên hàm chạy cuối cùng, nhằm mục đích kiểm tra năng lực của các giai đoạn cấu trúc trước rồi dùng tên để chấm điểm chúng sau. Đối với cặp `-O1` so với `-O2`:

| Hàm | Giai đoạn khớp | Các chữ ký đồng thuận | Tỷ lệ tương đồng |
|---|---|---|---|
| `percent_change` | byte chính xác | toàn bộ 5 chữ ký | 1.000 |
| `grade` | byte chính xác | toàn bộ 5 chữ ký | 1.000 |
| `total` | tên biểu tượng | chỉ có tên | 0.171 |
| `grade_label` | byte chính xác | toàn bộ 5 chữ ký | 1.000 |

Hàm `total` đã tăng từ 31 byte (13 lệnh) lên 140 byte (41 lệnh), bởi vì clang đã vector hóa vòng lặp bằng tập lệnh SSE2 và duy trì một vòng lặp vô hướng (scalar remainder loop) cho phần dư. Hình dạng đồ thị của nó chuyển từ 5 khối (5 cạnh) thành 10 khối (13 cạnh). Không còn bất kỳ đặc điểm cấu trúc nào tồn tại nguyên vẹn, và chỉ có biểu tượng tên là mối liên kết duy nhất giữa hai phiên bản.

Đối với cặp `-O0` so với `-O1`, cùng thứ tự đó lại tạo ra một kết quả khớp sai:

| Hàm ở -O0 | Hàm khớp ở -O1 | Giai đoạn khớp | Tên có khớp không |
|---|---|---|---|
| `percent_change` | `grade_label` | hình dạng đồ thị luồng | không |
| `grade` | `grade` | tên biểu tượng | có |
| `total` | `total` | tên biểu tượng | có |
| `grade_label` | không khớp | | |

Hàm `percent_change` ở `-O0` có cấu trúc hình thoi 4 khối (diamond): một khối vào rẽ nhánh theo điều kiện `before == 0`, hai nhánh thực thi và một khối hội tụ để trả về. Chuỗi bậc đỉnh của nó là `0/2, 1/1, 1/1, 2/0`. Hàm `grade_label` ở `-O1` được inline hàm `grade` và cũng có cấu trúc hình thoi tương tự xung quanh phép kiểm tra biên. Hình dạng này là duy nhất ở cả hai phía nên quy tắc so khớp đã chấp nhận nó. Nhãn chất lượng trung bình (medium) chính là dấu hiệu cảnh báo, và giai đoạn so khớp tên chạy sau đó đã bác bỏ kết quả này. Sự ghép cặp sai này cũng vô tình chiếm dụng luôn hàm `grade_label` ở phía bên phải, khiến cho hai cặp hàm thực sự là `percent_change` và `grade_label` đều rơi vào trạng thái chưa được khớp.

Thứ tự chuẩn của BinDiff sẽ đưa giai đoạn khớp tên lên vị trí thứ hai khi biểu tượng thực tế tồn tại, và biến `BINDIFF_ORDER` trong mã nguồn cũng thực hiện tương tự. Với thứ tự này, toàn bộ 4 hàm ở `-O0` đều khớp chính xác với phiên bản ở `-O1` theo tên, đúng như kết quả mà Ghidra báo cáo dưới đây. Bài học duy trì cả hai thứ tự trong phần demo để lỗi so khớp sai hiển thị rõ ràng. Một đồ thị 4 nút là hình dạng phổ biến của hầu như mọi cấu trúc `if-else` hội tụ trước lệnh return.

## Byte giống hệt nhau, toán hạng bị tái định vị

Ghidra áp dụng các phép tái định vị khi nó nạp (import) một tệp đối tượng. Do đó, bộ tương quan byte của nó so sánh các ảnh nhị phân đã nạp trong bộ nhớ, và bài học 36 đã che mặt nạ các trường tái định vị chính vì lý do này. Hàm `load_image()` tái tạo lại quá trình nạp đó: Nó duyệt qua các phân đoạn được cấp phát theo thứ tự tiêu đề, căn chỉnh từng phân đoạn theo `sh_addralign` và bố trí chúng bắt đầu từ địa chỉ `0x100000`. Sau đó, nó áp dụng từng bản ghi tái định vị: `S + A - P` ghi dưới dạng số nguyên có dấu 32-bit cho `R_X86_64_PC32` và `R_X86_64_PLT32`, còn `S + A` ghi dưới dạng 64-bit cho `R_X86_64_64`.

Bố cục bộ nhớ này khớp hoàn toàn với Ghidra 12.1.4 trên từng điểm bắt đầu phân đoạn ở cả hai chương trình, được kiểm chứng đối chiếu với đầu ra Version Tracking:

| Phân đoạn | Chương trình `-O1` | Chương trình `-O2` |
|---|---|---|
| `.text` | 0x100000, 159 bytes | 0x100000, 271 bytes |
| `.rodata` | 0x1000a0 | 0x100110 |
| `.rodata.str1.1` | 0x1000f0 | 0x100160 |

Phân đoạn `.text` dài hơn 112 byte ở `-O2` (do sự phình to của `total`), do đó `.rodata` bắt đầu cách xa hơn `0x70` byte. Hàm `grade` nằm tại địa chỉ `0x100010` trong cả hai chương trình. Lệnh `leaq` tại độ lệch `0x33` có trường độ lệch tại `0x36` và kết thúc tại `0x3a`. Số hạng cộng thêm của tái định vị (relocation addend) là `0x10` cộng thêm 4 byte thông thường sẽ trỏ bảng dữ liệu tới vị trí `.rodata+0x14`.

Trong ảnh bộ nhớ `-O1`: `0x1000a0 + 0x14 - 0x10003a = 0x7a = 122`. Trong ảnh bộ nhớ `-O2`: `0x100110 + 0x14 - 0x10003a = 0xea = 234`. Các byte của trường này là `7a 00 00 00` trong một ảnh và `ea 00 00 00` trong ảnh kia. Đúng 1 byte trong tổng số 46 byte bị khác biệt, tỷ lệ Levenshtein đạt 0.978, và giai đoạn so khớp byte chính xác bị thất bại. Giai đoạn byte che mặt nạ sẽ xóa trường này về 0 và khớp thành công. Hàm `grade_label` di chuyển cùng khoảng cách `0x70` tương ứng với bảng dữ liệu của nó, do đó các độ lệch của nó vẫn giữ nguyên giá trị bằng nhau và nó vượt qua bài kiểm tra byte chính xác ở cả hai góc nhìn.

Ghidra báo cáo sự phân hóa tương tự. Trên cặp `-O1` so với `-O2`, bộ tương quan Exact Function Bytes Match tìm thấy `percent_change` và `grade_label`, Exact Function Instructions Match tìm thấy thêm `grade`, và Exact Function Mnemonics Match cũng tìm thấy đúng 3 hàm đó. Bộ tương quan lệnh máy thực hiện che mặt nạ các toán hạng, bao quát cả trường độ lệch. Góc nhìn ảnh bộ nhớ của bài học này cho kết quả khớp byte chính xác cho đúng hai hàm mà bộ tương quan byte của Ghidra tìm thấy, và cho kết quả khớp byte che mặt nạ cho đúng ba hàm mà bộ tương quan lệnh của nó tìm thấy. Bộ kiểm thử khẳng định cả hai phép tương đương này.

Công cụ rea 6.5.0 (`compare_functions`) thực hiện so sánh dòng văn bản. Trên khía cạnh assembly, nó gắn kèm giới hạn "không hỗ trợ chuẩn hóa tái định vị (relocation normalization is unavailable)". Bộ nạp và cơ chế mặt nạ trong bài học này cung cấp chính xác sự chuẩn hóa đó.

## Những gì radiff2 thực sự đo lường

radare2 6.2.4 tích hợp công cụ `radiff2`. Với tùy chọn `-AC`, nó phân tích cả hai tệp và in ra một hàng cho mỗi hàm: kích thước và địa chỉ phía trái, nhãn MATCH hoặc UNMATCH, một tỷ lệ số học, cùng kích thước và địa chỉ phía phải. Các hàng kết quả ghi nhận được:

| Cặp so sánh | `percent_change` | `grade` | `total` | `grade_label` |
|---|---|---|---|---|
| `-O0` vs `-O1` | UNMATCH 0.130435 | UNMATCH 0.426829 | UNMATCH 0.106061 | UNMATCH 0.269841 |
| `-O1` vs `-O2` | MATCH 1.000000 | MATCH 1.000000 | UNMATCH 0.171429 | MATCH 1.000000 |
| `-O1` vs `-O1` | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 | MATCH 1.000000 |

Công thức tính tỷ lệ trong mã nguồn radare2 là `1 - distance / max(len_a, len_b)`, trong đó `distance` là khoảng cách Levenshtein giữa hai dấu vân tay hàm (function fingerprints), và một dấu vân tay là chuỗi nối liền các khối cơ bản của hàm theo thứ tự địa chỉ. Ngưỡng khớp là 0.7. Hàm `radare2_ratio()` trong `code/main.py` triển khai đúng công thức này trên các byte thô của hàm và tái hiện chính xác các con số 0.130435, 0.106061, 0.269841 và 0.171429 tới 6 chữ số thập phân.

Tuy nhiên, nó không tái hiện được con số 0.426829 cho `grade` ở `-O0`. Tính trên toàn bộ 118 byte của hàm, tỷ lệ thực tế là 0.296610. radare2 vẫn in `118` ở cột kích thước, nhưng dấu vân tay thực tế của nó lại ngắn hơn. Nếu không áp dụng các phép tái định vị, radare2 không thể phân giải bảng nhảy (jump table), do đó giải thuật phân tích của nó chỉ duyệt tới khối vào, khối kết thúc bằng `jmpq *%rax`, nhánh default và lệnh return: tổng cộng đúng 82 byte, tương đương 82 byte mà giải thuật duyệt đệ quy cơ bản ở bài học 33 đã chạm tới. Khoảng cách Levenshtein giữa 82 byte đó và 46 byte ở `-O1` là 47, và `1 - 47/82 = 0.426829`. Khi truyền cờ `-e bin.relocs.apply=true`, radare2 tìm thấy đủ 118 byte và in ra tỷ lệ 0.288136, đúng bằng tỷ lệ mà bài học tính toán trên các ảnh đã tái định vị, nơi 4 byte độ lệch khác với các trường bằng 0 của tệp đối tượng. Hai công cụ chỉ có thể đồng thuận khi chúng so sánh cùng một chuỗi byte, và cột kích thước không hề nói cho bạn biết những byte nào thực sự đã được đưa vào giải thuật.

Tùy chọn `-O` giới hạn dấu vân tay chỉ gồm các byte opcode và gán 0 cho các toán hạng. Tỷ lệ giữa `-O0` và `-O1` khi đó tăng lên 0.239130, 0.512195, 0.303030 và 0.523810, còn hàm `total` giữa `-O1` và `-O2` tăng lên 0.192857. Các phép đo khoảng cách trên toàn bộ tệp cũng tồn tại: `-s` (thuật toán Myers) báo cáo độ tương đồng 0.956 và khoảng cách 162 giữa hai tệp đối tượng, `-ss` (Levenshtein) báo cáo 0.927 và 137. Những con số này bao gồm cả phần tiêu đề, bảng biểu tượng và các phân đoạn tái định vị, hoàn toàn không phản ánh bản chất của từng hàm riêng lẻ.

## Các bộ tương quan của Ghidra Version Tracking

Công cụ Version Tracking của Ghidra thực hiện so khớp một chương trình nguồn với một chương trình đích thông qua các bộ tương quan (correlators). Các bộ tương quan cấp hàm bao gồm: Exact Function Bytes Match, Exact Function Instructions Match (che mặt nạ toán hạng), Exact Function Mnemonics Match và Duplicate Function Instructions Match (quan hệ 1:N và N:M). Các bộ tương quan dữ liệu gồm: Exact, Duplicate và Similar Data Match. Các bộ tương quan biểu tượng so khớp tên chính xác, trùng lặp hoặc theo độ tương đồng. Các bộ tương quan tham chiếu sử dụng các cặp khớp đã được chấp nhận để tìm kiếm thêm các cặp mới. Các hàm phải được định nghĩa trong chương trình đích, và độ dài hàm tối thiểu mặc định là 10 byte. Tính năng Auto Version Tracking chạy tuần tự: Exact Symbol Name, Exact Data, Exact Function Bytes, Exact Function Instructions, Exact Function Mnemonics, Duplicate Function Instructions, sau đó đến một bộ tương quan tham chiếu và tự động suy ra các cặp khớp gián tiếp. Bộ tương quan BSim được bổ sung từ bản 11.2.

Bài học này đã thu thập kết quả từ các bộ tương quan ở chế độ không giao diện (headless). Kịch bản `capture_ghidra_vt.sh` nạp 3 tệp đối tượng vào một dự án và chạy `ListVersionTrackingMatches.java` — một kịch bản Java ngắn kế thừa `AbstractGhidraVersionTrackingScript`, mở các chương trình nguồn và đích theo đường dẫn, thực thi 6 bộ tương quan không tham chiếu theo đúng thứ tự của Auto Version Tracking, rồi in ra một dòng cho mỗi khối, hàm, bộ tương quan và kết quả khớp:

| Bộ tương quan | Từ `-O0` sang `-O1` | Từ `-O1` sang `-O2` |
|---|---|---|
| Exact Symbol Name Match | 4 hàm | 4 hàm, 5 mục dữ liệu (`labels.rel`, `.L.str.1` đến `.L.str.4`) |
| Exact Data Match | 0 | 0 |
| Exact Function Bytes Match | 0 | 2 (`percent_change`, `grade_label`) |
| Exact Function Instructions Match | 0 | 3 (bổ sung thêm `grade`) |
| Exact Function Mnemonics Match | 0 | 3 |
| Duplicate Function Instructions Match | 0 | 0 |

Mọi hàm đều bị thay đổi giữa `-O0` và `-O1`, do đó chỉ có tên là khớp, trùng khớp với kết quả mà bộ so khớp của bài học đưa ra khi chạy theo thứ tự BinDiff. Mọi cặp khớp đều in độ tương đồng 1.000 và độ tin cậy 10.000. Trang tài liệu trợ giúp mô tả các bộ tương quan chính xác có độ tương đồng 1.0 và độ tin cậy 1.0, do đó thang đo của độ tin cậy được in ra là một câu hỏi mở mà bài học này ghi nhận lại.

Công cụ Program Differences trong menu Tools lại là một cơ chế hoàn toàn khác. Nó so sánh hai chương trình có cùng không gian địa chỉ dựa trên byte, nhãn, đơn vị mã, tham chiếu, chú thích và các hàm tại các địa chỉ bằng nhau. Đó là một phép diff căn chỉnh theo địa chỉ cố định và không thể khớp bất kỳ thứ gì đã bị dịch chuyển vị trí.

## Diaphora và Binary Ninja qua tài liệu kỹ thuật

Diaphora là một plugin cho IDA Pro, phát hành lần đầu năm 2015 và đạt phiên bản 3.4.2 tính đến 2026-08-18. Nó áp dụng giấy phép AGPL-3.0 từ bản 2.0 và yêu cầu IDAPython, do đó nó chạy được trên IDA Home hoặc IDA Pro chứ không chạy được trên IDA Free. Nó xuất mỗi cơ sở dữ liệu sang SQLite và thực thi các luật heuristic dưới dạng các câu truy vấn SQL. Mỗi heuristic có một tên gọi, một phân loại (Best, Partial hoặc Unreliable), một loại tỷ lệ và các cờ tùy chọn như `SAME_CPU` hoặc `SLOW`. Các tên gọi trong mã nguồn hiện tại tương tự như các giai đoạn trong bài học này nhưng đa dạng hơn: băm byte, trùng địa chỉ và tên gợi nhớ, trùng assembly đã làm sạch, tích số nguyên tố của tên gợi nhớ, trùng số nút, cạnh, vòng lặp và các thành phần liên thông mạnh, băm sắp xếp topo, cùng các hàm băm mờ mã giả yêu cầu trình dịch ngược Hex-Rays. Lịch sử tinh chỉnh của nó cho thấy: các heuristic chậm chỉ chạy khi số lượng hàm dưới 2,000, và quy tắc "Same graph" đã bị hạ cấp xuống nhóm không đáng tin cậy (unreliable). Sự hạ cấp đó xuất phát chính từ cấu trúc hình thoi 4 khối mà bài học này đã chỉ ra khi triển khai ở quy mô lớn.

Binary Ninja 6.0 (phát hành 2026-09-03) bổ sung tính năng Binary Similarity như một khung kiến trúc gồm các nhà cung cấp (providers) và bộ phân giải (resolvers). Nó tích hợp sẵn BinDiff làm nhà cung cấp cấu trúc và WARP cho các phép khớp hàm chính xác có khả năng dung thứ các toán hạng có thể tái định vị. Tài liệu hướng dẫn khuyến nghị dùng WARP khi các hàm khớp nhau ngoại trừ các lệnh có thể tái định vị, và dùng BinDiff khi chúng duy trì mối quan hệ cấu trúc sau khi tái biên dịch. Tính năng này chỉ có trên bản Ultimate.

Bản thân BinDiff đã trở thành mã nguồn mở từ 2023-09-25 theo giấy phép Apache-2.0, với phiên bản v8 là bản phát hành mới nhất. Nó yêu cầu một trình dịch ngược và plugin BinExport (hỗ trợ IDA Pro 9.4 trở lên, Binary Ninja và Ghidra). Con đường miễn phí cho người học là kết hợp Ghidra + BinExport + công cụ dòng lệnh BinDiff.

## Cùng mã lệnh, khác hằng số dữ liệu

Bài học 36 đã lưu giữ một bản dựng ứng viên của `grade` có phần tử cuối cùng trong bảng là 3, trong khi bản gốc là 4. Tệp `grade_case10_wrong.o` có cùng 46 byte mã lệnh giống hệt `grade` ở `-O1`. Bộ so khớp ghép đôi chúng ở giai đoạn byte chính xác với toàn bộ các chữ ký đồng thuận. Phép kiểm tra bảng dữ liệu tham chiếu đọc được chuỗi `1, 2, 3, 4, 4` qua một liên kết tái định vị và `1, 2, 3, 4, 3` qua liên kết kia, từ đó báo cáo `dataEqual: false`. Ba hàm còn lại của `-O1` rơi vào trạng thái chưa khớp ở phía trái vì tệp đối tượng ứng viên chỉ chứa duy nhất một hàm.

Đây chính là bản chất của việc phân tích bản vá bảo mật (patch analysis). Một công cụ so khớp hàm nếu chỉ dừng lại ở các byte mã sẽ khẳng định không có gì thay đổi. Bộ tương quan Exact Data Match của Ghidra và cơ chế khớp tham chiếu chuỗi của BinDiff tồn tại chính là để giải quyết bài toán này. Phiên bản của bài học này rất tinh gọn: đối với mỗi cặp đã khớp, so sánh các byte mà các liên kết tái định vị của từng hàm trỏ tới, với số lượng phần tử lấy từ lệnh kiểm tra biên và độ rộng phần tử lấy từ lệnh nạp.

## Thực thi biểu trưng: Đường đi, Điều kiện và Bộ giải

King đã mô tả phương pháp thực thi biểu trưng (symbolic execution) vào năm 1976. Các đầu vào trở thành các biểu trưng (symbols). Mỗi câu lệnh cập nhật một kho lưu trữ biểu trưng (symbolic store). Mỗi nhánh rẽ bổ sung một điều kiện vào đường đi thực thi, và một bộ giải (solver) sẽ quyết định xem đường đi đó có khả thi (feasible) hay không, đồng thời tạo ra một đầu vào cụ thể đi theo đường đi đó. Một hàm chứa các nhánh rẽ sẽ biến thành một cây các đường đi, và một nút lá là một lệnh return đi kèm một điều kiện ràng buộc.

Baldoni và các cộng sự đã khảo sát lĩnh vực này vào năm 2018. Mục 5 trong bài báo của họ chỉ rõ hiện tượng bùng nổ đường đi (path explosion) là chi phí thách thức cốt lõi: bộ thực thi có thể phân nhánh tại mọi điểm rẽ nhánh, do đó số lượng trạng thái có thể tăng theo cấp số nhân theo số lượng nhánh, và một vòng lặp có cận phụ thuộc vào một đầu vào biểu trưng sẽ tạo ra một tập hợp đường đi vô hạn. Các giải pháp giảm thiểu bao gồm: tỉa bớt các đường đi không thể thực hiện bằng cách giải ràng buộc tại từng nhánh rẽ, tóm tắt hàm và vòng lặp (summaries), hợp nhất trạng thái (state merging) và xấp xỉ dưới (under-approximation). Kỹ thuật thực thi concolic kết hợp thực thi cụ thể và biểu trưng cùng lúc, ghi lại các điều kiện dọc theo đường chạy cụ thể, phủ định một điều kiện và yêu cầu bộ giải tìm đầu vào tiếp theo (tiêu biểu như DART, CUTE và SAGE).

KLEE (OSDI 2008) là một máy ảo biểu trưng chạy trên mã bitcode LLVM (đầu vào là mã phát sinh từ `clang -emit-llvm` thay vì tệp nhị phân). Nó cung cấp mô hình môi trường POSIX và thư viện chạy lại (replay library). Trên 89 chương trình GNU coreutils, nó đạt độ bao phủ dòng 84.5% so với 67.7% của các bài kiểm thử từ nhà phát triển và phát hiện 56 lỗi nghiêm trọng. KLEE 3.2 được phát hành ngày 2025-12-23 theo giấy phép NCSA.

angr làm việc trực tiếp trên tệp nhị phân. Lớp `angr.Project` chứa bộ nạp CLE (ánh xạ tệp đối tượng và áp dụng tái định vị), bộ nâng mã (lifter) chuyển đổi một khối cơ bản thành VEX IR (`block.vex`), và một factory tạo trạng thái. Một `SimState` chứa các thanh ghi, bộ nhớ và một bộ giải. Một `SimulationManager` điều phối các ngăn chứa trạng thái (stashes) gồm `active`, `deadended`, `found`, `avoided` và `errored`, được bao bọc bởi phương thức `explore(find=..., avoid=...)`. claripy là giao diện bộ giải bao bọc bên ngoài Z3, và trong angr 10.0 nó là một bản triển khai bằng Rust được tích hợp sẵn. Z3 5.1.0 (phát hành 2026-08-16, MIT) là bộ giải đứng sau angr, Triton và backend tùy chọn của KLEE.

## Bộ thực thi tinh gọn trên biểu diễn trung gian của bài học 34

Bộ thực thi trong `code/main.py` sử dụng một IR nhỏ với các cấu trúc câu lệnh của bài học 34: `Assign`, `Branch`, `Jump`, `Return`, cộng thêm câu lệnh kết thúc `Switch` mà bài tập của bài học 34 đã đề cập. Các biểu thức gồm `Const`, `Var`, `Bin` kèm độ rộng, `Cast`, `Load` từ một bảng cụ thể có tên và `Not`. Bốn hàm fixture được nâng mã thủ công ở cấp độ bài học 34 đạt được sau quá trình đơn giản hóa, nơi phép nhân với `0x66666667` đã được rút gọn thành `score / 10`, và mỗi khối IR đều liệt kê các địa chỉ lệnh máy mà nó sinh ra. Hàm `check_ir_against_listing()` xác nhận rằng các địa chỉ này bao trùm danh sách lệnh của hàm chính xác một lần.

Hàm `grade` ở `-O0` trong biểu diễn IR đó:

```text
0x30: local_8 = score; local_10 = score / 10 - 6; if (local_10 >u 4) goto 0x9a else 0x62
0x62: switch local_10 -> 0x91, 0x88, 0x7f, 0x76, 0x76
0x76: local_4 = 4; goto 0xa1        0x7f: local_4 = 3; goto 0xa1
0x88: local_4 = 2; goto 0xa1        0x91: local_4 = 1; goto 0xa1
0x9a: local_4 = 0                    0xa1: return local_4
```

Các đích nhảy của switch được trích xuất từ `jump_table_targets()`, dùng cùng công thức số học tái định vị như bài học 33.

Hàm `execute()` duy trì một danh sách công việc (worklist) chứa các trạng thái đường đi. Một trạng thái lưu khối hiện tại, vết (trace) các khối đã qua, danh sách điều kiện và một môi trường ánh xạ từng biến sang một biểu thức biểu trưng theo các đầu vào. Lệnh `Assign` thế môi trường vào biểu thức nguồn và gán kết quả. Lệnh `Branch` thế vào điều kiện rẽ nhánh, rồi truy vấn bộ giải hai lần: một lần với điều kiện được thêm vào, một lần với phủ định của nó. Nhánh nào bộ giải không thể thỏa mãn sẽ bị loại bỏ ngay lập tức (eager pruning). Lệnh `Switch` nhóm các mục trong bảng theo đích nhảy và phân nhánh một lần cho mỗi đích riêng biệt kèm theo phép tuyển (disjunction) các điều kiện bằng nhau, tương tự như cách angr xử lý một đích nhảy biểu trưng. Lệnh `Return` thế vào giá trị trả về và đóng nút lá lại. Một vết vượt quá giới hạn số bước sẽ bị đánh dấu là bị cắt cụt (truncated) và dừng lại.

Hãy lần theo đường đi tới `return 4` bằng tay: Khối vào gán `local_10 = score / 10 - 6`. Điều kiện rẽ nhánh `local_10 >u 4` là sai trên đường đi này, vì vậy danh sách điều kiện chứa `!(score / 10 - 6 >u 4)`. Các mục switch 3 và 4 đều trỏ tới `0x76`, do đó nhánh rẽ sang `0x76` bổ sung thêm `score / 10 - 6 == 3 || score / 10 - 6 == 4`. Khối `0x76` gán `local_4 = 4`, nhảy tới `0xa1`, và lệnh return thế thành giá trị `4`. Nút lá là cặp gồm điều kiện đó và giá trị trả về đó.

Bộ giải ở đây là `BoundedSearch`. Nó duy trì một miền giá trị hữu hạn cho mỗi đầu vào, duyệt qua tích Descartes của các miền giá trị và đánh giá từng ràng buộc theo ngữ nghĩa của ngôn ngữ C: quay vòng tràn số 32-bit (wraparound), phép chia cắt cụt phần thập phân, so sánh có dấu và không dấu, phép dịch bit số học hoặc logic. Phép chia cho 0 hoặc chỉ số bảng nằm ngoài phạm vi sẽ làm cho phép gán đó thất bại trước ràng buộc. Với `score` nằm trong khoảng từ -200 đến 1200, 5 đường đi của hàm `grade` tốn 44,460 phép đánh giá và chỉ mất vài mili-giây. Giải từng nút lá để tìm tập hợp đầu vào ta được:

| Nút lá | Giá trị trả về | Điều kiện ràng buộc | Các điểm số nằm trong biên |
|---|---|---|---|
| `0x76` | 4 | `!(q >u 4) && (q == 3 \|\| q == 4)` | 90 đến 109, 20 đầu vào |
| `0x7f` | 3 | `!(q >u 4) && q == 2` | 80 đến 89, 10 đầu vào |
| `0x88` | 2 | `!(q >u 4) && q == 1` | 70 đến 79, 10 đầu vào |
| `0x91` | 1 | `!(q >u 4) && q == 0` | 60 đến 69, 10 đầu vào |
| `0x9a` | 0 | `q >u 4` | 1,351 đầu vào, phần còn lại của miền biên |

Trong đó `q` đại diện cho `score / 10 - 6`. Các lớp nghiệm tương tự cũng được tìm thấy từ hàm `grade` ở `-O1`, nơi IR chỉ có 2 đường đi: đường đi qua bảng trả về `lookup_14[q]` và đường đi mặc định trả về 0, và câu hỏi khi đó trở thành một ràng buộc đặt lên giá trị trả về. Bảng `lookup_14` được đọc trực tiếp từ `.rodata` của tệp đối tượng tại độ lệch `0x14`.

Biên tìm kiếm chính là giới hạn của phương pháp này. Tìm kiếm có giới hạn mang tính đầy đủ (complete) bên trong miền giá trị của nó và hoàn toàn im lặng bên ngoài miền đó. Trong khoảng -200 đến 1200, không có điểm số nào trong đoạn 60 đến 109 trả về 0 và không có điểm số nào từ 110 trở lên trả về giá trị khác 0. z3 chứng minh cả hai mệnh đề này cho toàn bộ 2^32 giá trị điểm số khả dĩ, và phần tiếp theo sẽ chỉ ra vị trí mà hai phương pháp trả về các bằng chứng nghiệm (witnesses) khác nhau.

## Đo lường sự bùng nổ đường đi

Hàm `total` ở `-O1` là vòng lặp của bài học 33: kiểm tra biến `count`, một thân vòng lặp cộng dồn `values[rdx]` và tăng `rdx`, cùng một lệnh `jne` quay lại thân vòng lặp chừng nào `rdx != rcx`. Trong biểu diễn IR, `rcx` là `zext64(count)` và `rdx` bắt đầu từ 0. Mỗi lần đi qua cạnh quay lui (back edge) sẽ rẽ nhánh theo điều kiện `rdx != count` với `rdx` là một hằng số, vì vậy đường đi cho `count = c` sẽ thực thi thân vòng lặp đúng `c` lần rồi thoát ra. Khi giới hạn `count` trong khoảng 0 đến 2, bộ thực thi tìm thấy 3 đường đi. Với 0 đến 4, nó tìm thấy 5, và với 0 đến 8, nó tìm thấy 9 đường đi. Mỗi đường đi trả về tổng tam giác (triangular sum) của số lần lặp: (0, 0), (1, 1), (2, 3), cho đến (8, 36). angr báo cáo 3, 5 và 9 trạng thái kết thúc (deadended) cùng 9 cặp (count, sum) tương tự ở cả 3 mức tối ưu, bao gồm cả `-O2` (vốn có vòng lặp vector yêu cầu count tối thiểu bằng 8 nên không làm thay đổi số lượng đường đi trong khoảng giá trị này).

Nếu không có cận biên giới hạn, bộ thực thi sẽ không bao giờ kết thúc, bởi vì mỗi giá trị số lần lặp lại tạo ra thêm một đường đi khả thi mới. Với mức giới hạn trần 24 khối cho mỗi đường đi và cho `count` chạy tới 64, bộ thực thi trả về 22 đường đi, cắt cụt 2 đường và không bao giờ bắt đầu 41 đường đi còn lại trong tổng số 65 đường đi tồn tại. Chi phí giải nghiệm cũng tăng lên tương ứng: 33 phép đánh giá ở biên 2, 115 ở biên 4, 531 ở biên 8 và 6,900 cho lượt chạy có áp trần.

## angr và z3 trên cùng các câu hỏi

Kịch bản `capture_angr.py` đã thực thi angr 10.0.3 trên cả 3 tệp đối tượng. CLE nạp từng tệp đối tượng có thể tái định vị với backend ELF tại `0x400000`, áp dụng 13 phép tái định vị cho `-O0` và 8 cho các bản còn lại, rồi phân giải 4 biểu tượng. Đối với `grade` với một đối số biểu trưng 64-bit, phương thức `run()` kết thúc với 5 trạng thái deadended ở `-O0` và 2 ở `-O1` cùng `-O2`, trong khoảng thời gian lần lượt là 0.475, 0.408 và 0.318 giây. Thêm điều kiện `eax == v` vào từng trạng thái deadended và yêu cầu bộ giải tìm giá trị cực tiểu và cực đại có dấu cho ra các khoảng 60 đến 69, 70 đến 79, 80 đến 89 và 90 đến 109 cho `v` từ 1 đến 4 ở mọi mức tối ưu, và từ -2147483648 đến 2147483647 cho giá trị 0. Cả hai phép kiểm tra chéo đều cho kết quả vô nghiệm (unsat).

Đối với bài toán tiếp cận một khối cụ thể (reach-a-block), CFGFast được dùng trên `grade` ở `-O0`, liệt kê 8 khối mà bài học 33 đã tìm thấy, sau đó gọi `explore(find=0x400076)` cho khối lưu giá trị 4. Nó tìm thấy 1 trạng thái thỏa mãn (với 3 trạng thái vẫn đang hoạt động) trong 0.47 giây, với giá trị nhỏ nhất là 90, lớn nhất là 109 và các mẫu nghiệm cụ thể là 91, 93, 94, 97 và 101.

Đối với `percent_change`, có 2 trạng thái deadended với các ràng buộc `before[31:0] != 0` và `before[31:0] == 0`. Yêu cầu trạng thái khác 0 thỏa mãn `eax == 50` cho nghiệm `before = 21055744, after = 332315210` ở `-O0`, `40566804, 834296877` ở `-O1`, và `-24484867, 993934694` ở `-O2`. Phép tìm kiếm có giới hạn trong khoảng -16 đến 16 trả về `before = -10, after = -15`. Công cụ z3 khi được hỏi cùng câu hỏi trên danh sách lệnh của bài học 34 trả về `before = -25092599, after = 305712160`. Bộ kiểm thử đánh giá toàn bộ 5 cặp nghiệm này bằng bộ đánh giá IR và mỗi cặp đều cho ra kết quả đúng bằng 50. Một mô hình nghiệm chỉ chứng minh rằng tồn tại một đầu vào thỏa mãn. Nó không phải là nghiệm duy nhất, và ba công cụ hoàn toàn chính xác đã đưa ra ba nghiệm hoàn toàn khác biệt.

Kịch bản `capture_z3.py` mô hình hóa danh sách lệnh `-O1` của `grade` dưới dạng các vector bit (bit-vectors) 32-bit kèm phép nhân, hai phép dịch bit và bảng tra cứu, rồi đặt thêm 4 câu hỏi kiểm tra: Liệu danh sách lệnh có khác với cấu trúc `switch` trong mã nguồn đối với một điểm số 32-bit nào đó không: vô nghiệm (unsat) trong 0.068 giây, chứng minh hàm đã biên dịch tương đương tuyệt đối với mã nguồn trên toàn bộ 2^32 đầu vào. Liệu danh sách lệnh có khác với bản dựng lại bằng các ngưỡng (threshold rebuild) của bài học 36 không: có nghiệm (sat) tại điểm số 706346536, nơi `score / 10` bằng 70634653 và switch rơi vào nhánh default trong khi bản dựng ngưỡng lại trả về 4. Giới hạn sự khác biệt đó trong khoảng 0 đến 109: vô nghiệm (unsat), giải thích vì sao lưới kiểm tra của bài học 36 không thể phát hiện ra sai lệch này. Đối với `percent_change`, điều kiện `before == 3` cho kết quả -133 là có nghiệm (sat) với `after = 1073741823` (ô tràn số của bài học 36), và phép tìm kiếm có giới hạn tìm thấy một nghiệm chứng minh thứ hai ngay trong miền của nó: `after = -1`, nơi `(-1 - 3) * 100 / 3` cắt cụt thành -133 mà không hề xảy ra tràn số.

## Xây dựng thực tế

Chạy kịch bản minh họa:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
```

Đầu ra là một báo cáo JSON duy nhất. Trường `diffs` chứa kết quả khớp tệp đối tượng và ảnh bộ nhớ cho `-O0` so với `-O1` và `-O1` so với `-O2`, cộng thêm cặp `-O0` theo thứ tự BinDiff. Trường `toolComparison` đặt giai đoạn của bài học bên cạnh kết luận và tỷ lệ của `radiff2` cùng các bộ tương quan Ghidra cho từng hàm. Trường `patch` thể hiện trường hợp `grade_case10_wrong.o`. Trường `radare2Ratios` tái hiện các con số của radiff2, `gradeRelocationCase` hiển thị bố cục và hai độ lệch, và `symbolic` chứa các đường đi, lớp nghiệm và số lượng bùng nổ đường đi.

Hãy đọc mã nguồn trong `code/main.py` theo thứ tự sau:

1. `ElfObject` đọc các phân đoạn, biểu tượng và bản ghi `.rela.*`. `parse_listing()` đọc văn bản xuất ra từ `llvm-objdump`.
2. `jump_table_targets()` và `build_cfg()` tạo ra các khối có thể duyệt tới và các cạnh có kiểu. `FunctionRecord.signatures()` xây dựng 5 loại chữ ký.
3. `load_image()` bố trí các phân đoạn được cấp phát và áp dụng các công thức `S + A - P` cùng `S + A`.
4. `match_functions()` là bộ so khớp theo từng giai đoạn. `levenshtein()` và `radare2_ratio()` tính toán tỷ lệ tương đồng.
5. `parse_radiff2()` và `parse_ghidra_vt()` đọc các đầu ra công cụ đã thu thập để phục vụ so sánh.
6. `evaluate()`, `substitute()` và `BoundedSearch` hiện thực hóa ngữ nghĩa biểu thức và bộ giải nghiệm.
7. `execute()` liệt kê các đường đi thực thi. `solve_returns()` truy vấn từng nút lá để lấy tập hợp đầu vào.
8. `grade_ir_o0()`, `grade_ir_o1()`, `percent_change_ir_o1()` và `total_ir_o1()` là các bản nâng mã thủ công đi kèm địa chỉ lệnh máy của chúng.

Công cụ hỗ trợ thêm 3 lệnh khác: `python3 code/main.py diff A.o B.o` so khớp hai tệp đối tượng có sẵn danh sách lệnh kèm theo, và cờ `--image` so khớp ảnh nạp trong bộ nhớ của chúng. Lệnh `python3 code/main.py paths grade O0` in các đường đi và lớp nghiệm của một hàm, `paths total 8` chạy vòng lặp với cận biên. Lệnh `python3 code/main.py figure-data` in dữ liệu mà sơ đồ tương tác nhúng vào.

## Cách sử dụng

Sử dụng bộ so khớp và bộ thực thi biểu trưng theo thứ tự sau trên hai bản dựng của chương trình bạn viết:

1. Trích xuất danh sách lệnh cho từng tệp đối tượng và để `function_record()` kiểm tra tính nhất quán với các byte mã. Khắc phục sự sai lệch trong danh sách lệnh trước khi làm bất kỳ điều gì khác.
2. So khớp các tệp đối tượng trước. Đọc kỹ giai đoạn của từng cặp khớp và chỉ coi các cặp khớp theo hình dạng đồ thị luồng là gợi ý cho đến khi có một chữ ký thứ hai đồng thuận.
3. So khớp các ảnh bộ nhớ đã nạp. Một hàm bị tụt hạng từ khớp byte chính xác xuống khớp byte che mặt nạ đồng nghĩa với việc nó đã bị dịch chuyển tương đối so với dữ liệu của nó, và phép tính số học độ lệch sẽ chỉ ra phân đoạn nào đã di chuyển.
4. Với mỗi cặp đã khớp, hãy so sánh các bảng dữ liệu được tham chiếu. Cùng mã lệnh nhưng khác hằng số dữ liệu bản chất là một hàm đã bị sửa đổi.
5. Chạy giai đoạn khớp tên và so sánh đối chiếu. Khi bản dựng có sẵn biểu tượng, hãy chạy theo thứ tự BinDiff và chỉ dùng kết quả cấu trúc để đánh giá năng lực của các giai đoạn.
6. Khi một công cụ báo cáo một tỷ lệ, hãy tìm hiểu rõ những byte nào đã được đưa vào thuật toán trước khi so sánh với con số của bạn.
7. Đối với hàm mà bạn muốn tìm đầu vào, hãy viết biểu diễn IR với một địa chỉ cho mỗi câu lệnh, chạy `execute()` với cận biên đặt trên mọi biến đếm vòng lặp biểu trưng và giải từng nút lá. Báo cáo cận biên đi kèm kết quả.
8. Chạy cùng câu truy vấn đó qua angr hoặc z3 khi có sẵn và so sánh các biên của lớp nghiệm. Một mẫu nghiệm đơn lẻ chỉ là một mô hình đại diện.

## Thực hành tương tác

### Bài thực hành A: tự tính năm chữ ký bằng tay

Từ thư mục `code`, mở môi trường Python:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution/code
python3 -q
```

Thực thi:

```python
from main import *
grade = {r.name: r for r in fixture_records("ledger-x86_64-O1.o")}["grade"]
grade.raw.hex(" ")
grade.masked_bytes() == grade.raw
grade.mnemonics
grade.cfg_shape()
grade.signatures()
```

Ghi lại 4 byte tại độ lệch `0x26` của hàm trước khi bạn in `grade.raw`. Sau đó chạy các dòng lệnh tương tự trên `grade` từ `ledger-x86_64-O0.o` và đếm các cạnh `table` trong `grade.edges`.

### Bài thực hành B: quan sát phép tái định vị phá vỡ mã băm byte

```python
case = grade_relocation_case()
case["layout"]
case["displacements"], case["rodataShift"]
case["objectBytesEqual"], case["imageBytesEqual"], case["maskedImageBytesEqual"]
```

Tính toán giá trị `0x100110 + 0x14 - 0x10003a` trên giấy trước khi bạn đọc trường `displacements`. Sau đó đổi địa chỉ cơ sở trong `load_image()` thành `0x400000` và xác nhận rằng các độ lệch không hề thay đổi, bởi vì cả hai toán hạng đều di chuyển cùng nhau.

### Bài thực hành C: cặp khớp sai và thứ tự khắc phục

```python
report = diff_fixtures("ledger-x86_64-O1.o", "ledger-x86_64-O2.o")
[(m.left.name, m.right.name, m.stage) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o")
[(m.left.name, m.right.name, m.stage, m.names_agree) for m in report.matches]
report = diff_fixtures("ledger-x86_64-O0.o", "ledger-x86_64-O1.o", stages=BINDIFF_ORDER)
[(m.left.name, m.right.name, m.stage) for m in report.matches]
```

In `records(...)["percent_change"].cfg_shape()` cho `-O0` và `grade_label` cho `-O1` rồi đọc hai chuỗi bậc đỉnh. Sau đó viết một giai đoạn thứ 6 thực hiện băm các tên gợi nhớ đã được sắp xếp (để thứ tự không còn ảnh hưởng) và quan sát xem liệu nó có tách biệt được hai hàm này hay không.

### Bài thực hành D: các đường đi và cận biên

```python
report = grade_paths("O0")
[(p["trace"], p["condition"], p["returns"]) for p in report["paths"]]
report["returns"]["4"]
total_paths(2), total_paths(4), total_paths(8)
total_paths(64, step_limit=24)["truncated"]
```

Đổi `SCORE_DOMAIN` thành `range(0, 110)` và chạy lại `grade_paths("O0")`. Lớp nghiệm cho giá trị 0 thu hẹp lại còn 60 đầu vào và hai phép kiểm tra chéo vẫn giữ nguyên tính đúng đắn. Sau đó đặt miền giá trị thành `range(90, 100)` và quan sát xem những nút lá nào không còn mô hình nghiệm nào thỏa mãn.

## Bài tập rèn luyện

Chạy BinDiff trên các tệp fixture và so sánh nó với 3 kết quả bạn đã có.

Yêu cầu:

- Cài đặt tiện ích mở rộng BinExport trong Ghidra 12.1.4 và xuất 3 chương trình ledger. Ghi lại thẻ phiên bản BinExport và phiên bản Ghidra.
- Cài đặt công cụ dòng lệnh BinDiff 8 và tạo kết quả cho cặp `-O0` so với `-O1` và `-O1` so với `-O2` với tùy chọn chỉ tính độ tương đồng (similarity-only).
- Thu thập độ tương đồng và độ tin cậy của từng hàm vào thư mục `code/fixtures/` với các câu lệnh chính xác trong `build_fixtures.sh`.
- Mở rộng hàm `tool_comparison()` với một cột `bindiff` và thêm một bài kiểm tra so sánh tập hợp khớp của BinDiff trên `-O1` so với `-O2` với tập hợp byte che mặt nạ của bài học.
- Viết một đoạn văn nêu rõ thuật toán mà BinDiff đã sử dụng cho hàm `grade` và liệu nó có so sánh các ảnh nạp trong bộ nhớ hay không.

Hoàn thành khi kịch bản demo in ra 4 cột công cụ cho từng hàm và bài kiểm tra mới tuyên bố một kết quả bằng nhau hoặc một sự khác biệt được giải thích rõ ràng.

## Sản phẩm bàn giao

Bài học này cung cấp tệp `outputs/skill-binary-diffing-and-symbolic-execution.md`. Sử dụng nó để biến hai bản dựng và một hàm thành một báo cáo so khớp theo từng giai đoạn với mức chất lượng được nêu rõ cho từng cặp khớp, một lời giải thích cho mọi hàm chưa khớp hoặc bị thay đổi, cùng một báo cáo đường đi với tập hợp đầu vào đã giải cho từng nút lá đi kèm cận biên đã tạo ra nó.

## Kiểm chứng tính đúng đắn

Chạy kịch bản demo và bộ kiểm thử:

```bash
cd phases/13-tools-and-protocols/44-binary-diffing-and-symbolic-execution
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Quá trình kiểm chứng cần chứng minh:

- Bảng biểu tượng cung cấp 4 hàm với kích thước 46, 118, 66, 39 ở `-O0`, 16, 46, 31, 63 ở `-O1`, và 16, 46, 140, 63 ở `-O2`
- Các bản ghi `.rela.text` của tệp đối tượng `-O1` mô tả đúng như những gì `llvm-objdump -r` in ra
- Mọi lệnh trong danh sách đều nằm trong phạm vi biểu tượng của nó và các byte của nó bằng với tệp đối tượng, một byte bị thay đổi sẽ phát sinh lỗi
- Hàm `total` ở `-O1` có 5 khối, 5 cạnh và 1 vòng lặp tự thân, chuỗi hình dạng của nó mang tính tất định
- Hàm `grade` ở `-O0` khôi phục được 5 đích của bảng nhảy từ `.rela.rodata` và có 8 khối cùng 11 cạnh
- Nếu không có bảng nhảy, `grade` ở `-O0` chỉ duyệt tới 82 byte trong 4 khối
- `percent_change` ở `-O0` và `grade_label` ở `-O1` có cùng hình dạng 4 khối
- Mã băm byte chính xác và byte che mặt nạ trùng khớp trên mọi hàm của một tệp đối tượng có thể tái định vị
- `-O1` và `-O2` chia sẻ mã byte cho 3 hàm, `-O0` và `-O1` không chia sẻ hàm nào
- Bảng tham chiếu của `grade` đọc được `1, 2, 3, 4, 4` và của ứng viên đọc được `1, 2, 3, 4, 3`
- Bộ nạp bố trí `.text`, `.rodata` và `.rodata.str1.1` đúng vị trí Ghidra đã đặt chúng trong cả hai chương trình
- `R_X86_64_PC32` được ghi dưới dạng `S + A - P` và cho ra độ lệch 122, các mục `R_X86_64_64` trong `.data` nhận địa chỉ chuỗi
- Các byte của `grade` giống nhau trong tệp đối tượng và khác nhau trong ảnh nạp do độ lệch `0x70`, và việc che mặt nạ khôi phục lại kết quả khớp
- Cặp `-O1` so với `-O2` khớp 3 hàm theo byte chính xác và `total` theo tên; góc nhìn ảnh bộ nhớ khớp `grade` theo byte che mặt nạ với độ tương đồng 1 - 1/46
- Thứ tự của bài học tạo ra một kết quả khớp sai theo hình dạng đồ thị luồng trên `-O0` so với `-O1`, và thứ tự BinDiff khớp toàn bộ 4 hàm theo tên
- Bản vá ứng viên khớp theo byte nhưng thất bại ở dữ liệu tham chiếu
- Tỷ lệ Levenshtein tái hiện chính xác 0.130435, 0.106061, 0.269841, 0.171429, 0.426829 trên 82 byte, và các tỷ lệ byte tái định vị mà radiff2 in ra khi áp dụng tái định vị
- Tập hợp MATCH của radiff2 bằng với giai đoạn byte chính xác, và các bộ tương quan byte cùng lệnh của Ghidra bằng với các tập hợp byte chính xác và che mặt nạ trong góc nhìn ảnh bộ nhớ
- Ghidra chỉ tìm thấy tên trên cặp `-O0` so với `-O1`, với độ tương đồng 1.0 và độ tin cậy 10.0 trên mọi cặp khớp
- Biểu diễn IR nâng mã thủ công bao trùm mọi lệnh đã liệt kê đúng một lần, và bộ đánh giá tuân thủ ngữ nghĩa C
- Hàm `grade` có 5 đường đi ở `-O0` với các điều kiện switch và 2 đường đi ở `-O1` với cùng các lớp nghiệm
- Các lớp nghiệm có giới hạn 60 đến 69, 70 đến 79, 80 đến 89, 90 đến 109 bằng với kết quả của angr ở cả 3 mức tối ưu và của z3, với 5 và 2 trạng thái deadended cùng khoảng tiếp cận khối 90 đến 109
- Hai phép kiểm tra chéo không có mô hình nghiệm nào trong miền biên, trong angr, hay trong z3, và z3 chứng minh danh sách lệnh tương đương với cấu trúc switch trong mã nguồn
- Mọi mô hình nghiệm của `percent_change` từ phép tìm kiếm có giới hạn, angr và z3 đều tính toán ra đúng giá trị yêu cầu
- Hàm `total` cho ra 3, 5, 9 đường đi với các tổng tam giác tương ứng, giống hệt angr, và việc áp trần số bước trả về 22 đường đi và cắt cụt 2 đường
- Dữ liệu sơ đồ mang cả hai góc nhìn, các số liệu tái định vị, cùng 5 nút lá đi kèm tập hợp đầu vào của chúng, và mô-đun trang web nhúng cùng dữ liệu đó

## Các chế độ thất bại trong thực tế

| Thất bại | Những gì công cụ yếu báo cáo | Những gì phân tích phải chứng minh |
|---|---|---|
| Băm byte trên ảnh bộ nhớ đã nạp | Báo "grade bị sửa đổi" giữa hai bản dựng có mã lệnh giống hệt nhau | Độ lệch trường, hai giá trị độ lệch, và khoảng dịch chuyển phân đoạn; mã băm che mặt nạ khớp thành công |
| Băm che mặt nạ trên tệp đối tượng có thể tái định vị | Khẳng định "che mặt nạ không tạo ra khác biệt gì" | Các trường vốn dĩ đã bằng 0; các bản ghi tái định vị và các hằng số được tham chiếu mang sự thay đổi |
| Chỉ dựa vào hình dạng đồ thị luồng điều khiển | Ghép đôi đầy tự tin hai hàm hoàn toàn không liên quan | Cần một chữ ký thứ hai đồng thuận, hoặc tên hàm mâu thuẫn dẫn tới việc loại bỏ cặp khớp |
| Đặt tên trước cấu trúc trên bản dựng đã tước bỏ biểu tượng | Không khớp được bất kỳ hàm nào | Thực thi các giai đoạn cấu trúc đi kèm chất lượng của chúng, và danh sách chưa khớp |
| So sánh tỷ lệ giữa các công cụ khác nhau | "radiff2 báo 0.43 trong khi bài học báo 0.30" | Những byte nào đã được đưa vào dấu vân tay; 82 byte khi không có tái định vị, 118 byte khi có |
| Cùng mã lệnh, khác hằng số dữ liệu | Báo cáo "khớp chính xác tuyệt đối" | Bảng dữ liệu được đọc qua liên kết tái định vị và lệnh kiểm tra biên |
| Báo cáo một mô hình nghiệm đơn lẻ như là đầu vào duy nhất | Khẳng định "điểm số là 90" | Khoảng lớp nghiệm từ 90 đến 109 kèm số lượng nghiệm, và cận biên mà nó được tìm thấy |
| Tìm kiếm có giới hạn báo cáo vô nghiệm (unsat) | Khẳng định "không có đầu vào nào cho ra -133" | Chỉ vô nghiệm bên trong miền biên; z3 tìm thấy nghiệm 1073741823 bên ngoài miền đó |
| Vòng lặp biểu trưng không có cận biên | Bị treo do vượt quá thời gian (timeout) | Cận biên hoặc mức áp trần, số đường đi trả về, bị cắt cụt và chưa bao giờ bắt đầu |
| Mô tả một công cụ đóng như thể đã được thực thi | Đưa ra các con số mà không có dữ liệu trích xuất | Tệp thu thập kèm câu lệnh và phiên bản, hoặc tuyên bố rõ ràng chỉ dựa trên tài liệu kỹ thuật |

## Quy tắc vận hành

Hãy so khớp trên từng chữ ký một tại mỗi thời điểm, chỉ chấp nhận các giá trị là duy nhất ở cả hai phía, và luôn lưu giữ giai đoạn cùng mức chất lượng đi liền với cặp khớp đó. So khớp cả tệp đối tượng lẫn ảnh nạp trong bộ nhớ và giải thích rõ mọi hàm bị thay đổi giai đoạn giữa hai góc nhìn. Liệt kê các đường đi bằng một bộ giải tại từng nhánh rẽ, đặt cận biên cho mọi vòng lặp biểu trưng, và báo cáo tập hợp đầu vào đi kèm cận biên của nó. Một mô hình nghiệm đơn lẻ không có cận biên chỉ đơn thuần là một mẫu thử đại diện.

## Tài liệu đọc thêm

- [Tài liệu khái niệm BinDiff: chiến lược so khớp, thuật toán và đánh giá chất lượng](https://github.com/google/bindiff/blob/main/docs/concepts.md)
- [Kho lưu trữ BinDiff và bản phát hành mã nguồn mở v8](https://github.com/google/bindiff/releases/tag/v8)
- [README của BinExport: các bộ xuất dữ liệu cho IDA Pro, Binary Ninja và Ghidra](https://github.com/google/binexport)
- [Flake, "Structural Comparison of Executable Objects" (DIMVA 2004) và Dullien, Rolles, "Graph-based comparison of Executable Objects" (SSTIC 2005)](https://github.com/google/bindiff/tree/main/docs/papers)
- [README và mã nguồn luật heuristic của Diaphora](https://github.com/joxeankoret/diaphora)
- [Trang trợ giúp các bộ tương quan Ghidra Version Tracking](https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/VersionTracking/src/main/help/help/topics/VersionTrackingPlugin/VT_Correlators.html)
- [Trang trợ giúp Ghidra Auto Version Tracking](https://www.ghidradocs.com/11.0_PUBLIC/help/VersionTracking/help/topics/VersionTrackingPlugin/VT_AutoVT.html)
- [Bản phát hành Ghidra 12.1.4](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Ghi chú phát hành Binary Ninja 6.0 "Krypton"](https://binary.ninja/2026/09/03/binary-ninja-6.0-krypton.html) và [Hướng dẫn Binary Similarity](https://docs.binary.ninja/guide/similarity.html)
- [radare2 book: radiff2](https://book.rada.re/tools/radiff2/intro.html) và [mã nguồn radare2 6.2.4](https://github.com/radareorg/radare2/releases/tag/6.2.4)
- [King, "Symbolic execution and program testing" (CACM 1976)](https://dl.acm.org/doi/10.1145/360248.360252)
- [Baldoni et al., "A Survey of Symbolic Execution Techniques" (ACM Computing Surveys 2018)](https://dl.acm.org/doi/10.1145/3182657)
- [Cadar, Sen, "Symbolic execution for software testing: three decades later" (CACM 2013)](https://dl.acm.org/doi/10.1145/2408776.2408795)
- [Cadar, Dunbar, Engler, "KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs" (OSDI 2008)](https://www.usenix.org/legacy/events/osdi08/tech/full_papers/cadar/cadar.pdf) và [bản phát hành KLEE 3.2](https://github.com/klee/klee/releases/tag/v3.2)
- [Shoshitaishvili et al., "SoK: (State of) The Art of War: Offensive Techniques in Binary Analysis" (IEEE S&P 2016)](https://oaklandsok.github.io/papers/shoshitaishvili2016.pdf)
- [Tài liệu angr: loading, simulation managers, và pipeline mô phỏng](https://docs.angr.io/en/latest/core-concepts/toplevel.html) và [angr 10.0.3 trên PyPI](https://pypi.org/project/angr/)
- [Bản phát hành Z3 5.1.0](https://github.com/Z3Prover/z3/releases/tag/z3-5.1.0)
- [Thư viện phân tích nhị phân động Triton](https://github.com/JonathanSalwan/Triton)
- [System V Application Binary Interface, AMD64 Architecture Processor Supplement, Bảng 4.9: Các kiểu tái định vị](https://gitlab.com/x86-psABIs/x86-64-ABI)
