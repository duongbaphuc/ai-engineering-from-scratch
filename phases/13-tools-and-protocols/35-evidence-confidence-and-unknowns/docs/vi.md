# Bằng Chứng, Độ Tin Cậy và Các Yếu Tố Chưa Biết Trong Phân Tích Nhị Phân

> Một phát hiện về một chương trình đã biên dịch là một bản ghi có mã băm, mức độ tin cậy, thẩm quyền xác thực và một danh sách những gì nó không thể chỉ ra.

**Type:** Build
**Languages:** Python, C
**Prerequisites:** Phase 13 · 32 (ELF from scratch), Phase 13 · 33 (disassembly and control flow), Phase 13 · 34 (lifting to pseudocode)
**Time:** ~105 phút

## Mục tiêu Học tập

- Xây dựng một bản ghi bằng chứng (evidence record) mô phỏng theo cấu trúc `Evidence` của rea với ba mức độ tin cậy: quan sát được (`observed`), dẫn xuất (`derived`), và suy luận (`inferred`).
- Tính toán mã định danh băm nội dung (content-hash ids) bất biến ngay cả khi tệp được di chuyển sang một thư mục khác.
- Ghi nhận những gì từng công cụ phân tích không thể chỉ ra dưới dạng các giới hạn (limitations) đi liền với bản ghi.
- Kiên quyết từ chối khẳng định một kết quả cho đến khi có một lần thực thi quan sát được của hàm chứng minh điều đó.
- Quản lý các yếu tố chưa biết còn tồn đọng (residual unknowns) dưới dạng các bản ghi có phiên bản được liên kết bằng một chuỗi băm (hash chain).
- Lưu đệm (cache) các câu trả lời của công cụ dưới một khóa định danh chính xác và từ chối việc phát lại dữ liệu đã lỗi thời (stale replay).

## Vấn đề Cần Giải Quyết

Bạn hỏi một AI agent xem hàm `grade` trong tệp `ledger-x86_64-O1.o` trả về giá trị gì khi điểm số đầu vào là 55. Agent gọi công cụ rea, đọc mã giả do Ghidra sinh ra và trả lời là 0. Một người đánh giá hỏi làm thế nào agent biết được điều đó. Agent đưa ra một đoạn văn bản mà Ghidra đã in. Đối với nhánh rẽ mặc định, đoạn văn bản đó ghi: `uVar1 = -(param_1 >> 0x1f) ^ (uint)((ulong)((long)param_1 * 0x66666667) >> 0x3f);`. Nhưng các byte mã máy thực tế tại vị trí đó lại là `xor eax, eax`.

Cả biểu thức toán học phức tạp kia lẫn lệnh mã máy đơn giản đều cho ra kết quả bằng 0 với mọi đầu vào. Nhưng agent không hề biết điều đó. Nó tin vào đoạn văn bản chỉ vì một công cụ đã phát ra nó. Một nghiên cứu năm 2026 đã cho ba AI agent chạy trên 24 bài toán kỹ nghệ đảo ngược ([NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)) và chỉ ra bốn điểm yếu lặp đi lặp lại: thiên kiến huấn luyện (training bias), tin tưởng thái quá vào quan sát bề mặt (over-trust in observations), giới hạn ngữ cảnh (context limits), và sự bảo thủ với kế hoạch ban đầu (plan persistence). Một nghiên cứu khác năm 2026 đã cài cắm văn bản độc hại vào trong các tệp nhị phân và phát hiện ra rằng các mô hình ngôn ngữ đã đề xuất hành động không an toàn được cài cắm trong 35 trên tổng số 40 trường hợp ([arXiv 2607.12507](https://arxiv.org/abs/2607.12507)).

Bộ công cụ rea 6.5.0 cung cấp cho agent một hệ thống thuật ngữ chặt chẽ để giải quyết vấn đề này. Mọi lời gọi công cụ thành công đều sinh ra một bản ghi `Evidence` với một thuộc tính `confidence` mang một trong ba giá trị `observed`, `derived`, hoặc `inferred`; một thuộc tính `authority`; và một danh sách các `limitations`. Kỹ năng đóng gói đi kèm quy định rõ cho agent: "Mọi kết luận đều phải phân biệt rạch ròi giữa quan sát thực nghiệm, suy diễn và các yếu tố chưa biết. Hãy trích dẫn mã Evidence ID, bảo toàn các giới hạn cùng phạm vi bao phủ chưa hoàn tất, và tuyệt đối không bao giờ được ngụ ý rằng phân tích tĩnh đã quan sát quá trình thực thi thời gian chạy.".

Ba điều mà rea để bạn tự xử lý: Các snapshot của nó chỉ lưu đệm các kết quả truy vấn chính xác chứ không lưu đệm logic phân tích. Quy trình xác minh của nó phê duyệt một khẳng định khi một chiều kích so sánh không đổi, và chính chú thích trong mã của nó thừa nhận việc vượt qua kiểm tra "does not establish global implementation equivalence" (không xác lập tính tương đương triển khai toàn cục). Sổ cái nghĩa vụ của nó chỉ tiếp nhận các ứng viên từ JavaScript và các đồ thị được quản lý, vì vậy một hàm mã máy gốc (native function) không bao giờ đạt tới trạng thái `verified` thông qua nó.

Bài học này sẽ tự tay xây dựng mảnh ghép còn thiếu đó hoàn toàn bằng Python thuần túy. Bạn sẽ viết một sổ cái lưu trữ mọi dữ kiện về fixture dưới dạng một bản ghi có mã băm nội dung. Bạn đưa ra các dự đoán về `percent_change` và `grade`, đánh dấu chúng là suy luận (`inferred`), và kiểm tra chúng đối chiếu với các lượt chạy thực tế của một chương trình kiểm thử do chính bạn biên dịch. Bạn nhập hai bản ghi thực tế từ rea cùng snapshot gốc của chúng, và sổ cái của bạn sẽ tự tính toán lại chuẩn xác từng mã băm mà rea đã phát ra.

````figure re-evidence-ledger
````

## Ba Từ Cho Độ Tin Cậy, Năm Từ Cho Thẩm Quyền

Thuộc tính `confidence` (mức độ tin cậy) là một kiểu liệt kê (enum) chỉ có đúng ba giá trị, tuyệt đối không phải là một con số xác suất. Thuộc tính `authority` (thẩm quyền xác thực) có năm giá trị. Khi một bên phát sinh dữ liệu không nêu rõ, rea sẽ mặc định sử dụng `shipped-artifact` đi cùng `observed`.

| Giá trị | Ý nghĩa đối với fixture | Các bản ghi trong bài học này |
|---|---|---|
| `observed` (Quan sát) | Dữ kiện đọc trực tiếp từ byte hoặc từ một lần chạy | Đọc phân đoạn và ký hiệu của bộ đọc ELF, hai bản ghi Ghidra, 17 lần chạy host |
| `derived` (Dẫn xuất) | Dữ kiện được tính toán từ các bản ghi quan sát | Mô hình bảng switch, mô hình phép chia, mọi phép biến đổi yếu tố chưa biết |
| `inferred` (Suy luận) | Một kỳ vọng chưa có quan sát thực nghiệm nào xác nhận | 18 dự đoán hành vi |
| `shipped-artifact` | Tệp nhị phân trên đĩa là nguồn thông tin | Mọi lần đọc từ tệp `ledger-x86_64-O1.o` |
| `controlled-replay` | Một lần thực thi do bạn kiểm soát tạo ra dữ kiện | Các lần chạy mã nhị phân `harness.c` trên máy host |
| `analyst-inference` | Phán đoán chuyên môn của chính bạn | Các dự đoán và biến đổi yếu tố chưa biết |
| `historical-reference` | Tài liệu hoặc cây mã nguồn cũ hơn | Không có trong bài học này |
| `external-service` | Một dịch vụ nằm ngoài tệp và máy host | Không có trong bài học này |

Hai sự nhầm lẫn tai hại nhất làm tiêu tốn nhiều thời gian của kỹ sư: Thứ nhất là coi mã giả như một quan sát về hành vi thời gian chạy; thực chất nó chỉ là một quan sát về những gì Ghidra đã in ra. Thứ hai là coi một danh sách rỗng là bằng chứng của sự vắng mặt; nguyên lý thiết kế của rea nhấn mạnh: "Missing coverage is unknown, not absence." (Thiếu dữ liệu bao phủ nghĩa là chưa biết, không phải là không có). Sổ cái trong `code/main.py` mã hóa nghiêm ngặt cả hai quy tắc này.

## Cấu Trúc Bản Ghi

Một bản ghi có chính xác 16 trường dữ liệu, và rea 6.5.0 sẽ từ chối bất kỳ bản ghi nào thừa hoặc thiếu trường. Mã nguồn `code/main.py` duy trì chính xác các tên trường như rea, do đó một bản ghi từ bên này có thể được xác thực hoàn toàn trên bên kia. Trường thứ 16, `relationship_coverage`, cho biết mức độ hoàn chỉnh của danh sách các hàm gọi hoặc bị gọi; giá trị của nó là `complete` (hoàn tất), `partial` (một phần), hoặc `unknown` (chưa rõ), hoặc `null` khi thao tác không báo cáo các mối quan hệ lời gọi hàm.

| Tên trường | Nội dung lưu trữ |
|---|---|
| `evidence_id` | Chuỗi `ev_` kèm 64 ký tự hex, là mã băm của nội dung |
| `subject` | `name`, `digest.sha256`, `format`, `architecture`, `local_path`, hoặc `null` |
| `provider` | `id`, `name`, `version` hoặc `null` |
| `analysis_profile` | Cấu hình của công cụ và mã băm của cấu hình đó, hoặc `null`, luôn phải hiện diện |
| `predicate_type` | Thẻ phân loại, mặc định là `rea.analysis` |
| `operation`, `parameters` | Tên công cụ và các tham số đầu vào của nó |
| `raw_result`, `normalized_result` | Đầu ra thô của công cụ và kết quả chuẩn hóa quy chuẩn |
| `confidence`, `authority` | Các kiểu liệt kê đã nêu ở trên |
| `environment` | `id`, `platform`, `architecture`, `isolation`, hoặc `null` |
| `limitations` | Các câu văn bản mô tả giới hạn thực tế |
| `relationship_coverage` | Trạng thái `complete`, `partial`, hoặc `unknown` kèm chẩn đoán, hoặc `null` |
| `locations` | Danh sách vị trí: `address`, `address-range`, `artifact-path`, `file-offset`, `file-offset-range` |
| `evidence_links` | Danh sách các mã `ev_` mà bản ghi này dựa vào |

Dưới đây là bản ghi thực tế mà rea đã ghi nhận cho hàm `grade`, được cắt gọt hiển thị các trường quan trọng. Toàn bộ tệp gốc là `code/fixtures/rea-evidence-grade.json` với dung lượng 37.573 byte:

```json
{
  "evidence_id": "ev_67b6d3f992eaf49a2e16de084b6ebfdd9884e8c5c7f75e175b7bb20f4f748ce1",
  "subject": {
    "name": "ledger-x86_64-O1.o",
    "digest": { "sha256": "9a4a8082ba5e285a732552981a6935e0c5255725664f380e37c44ebfb04d803f" },
    "format": "elf",
    "architecture": "x86_64",
    "local_path": "/private/tmp/aiefs-re/phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code/fixtures/ledger-x86_64-O1.o"
  },
  "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
  "analysis_profile": {
    "provider": { "id": "ghidra", "name": "Ghidra", "version": "12.1.4" },
    "parameters": { "jump_table_evidence": "typed-case-default-blocks-v1", "import_mode": "ephemeral-source-immutable" },
    "digest": "9c7f6b24be69294e61da046e4583e3856cb9e40433b149c1bf8bb534d2fe03ef"
  },
  "operation": "analyze_function",
  "parameters": { "procedure": "grade" },
  "confidence": "observed",
  "authority": "shipped-artifact",
  "environment": null,
  "limitations": [
    "Default-space addresses use lowercase 0x-prefixed hexadecimal; other address spaces use <percent-encoded-space>:0x<hex>.",
    "Resolved reference metadata identifies computed, indirect, external, call, jump, and data edges; unresolved targetless flows remain unknown, and function classifications distinguish thunks and externals."
  ],
  "relationship_coverage": {
    "status": "complete",
    "diagnostics": [
      "Only resolved Ghidra call references are returned; unresolved computed or indirect calls remain unknown, while function classifications distinguish thunks and externals."
    ]
  },
  "locations": [],
  "evidence_links": []
}
```

Đối tượng `parameters` thực tế của profile có tới 25 khóa. Danh sách `limitations` thực tế có 10 câu. Hãy giữ nguyên toàn bộ chúng: Giới hạn là một phần cấu thành nội dung, vì vậy việc xóa bỏ một giới hạn sẽ làm thay đổi mã băm và khiến sổ cái từ chối bản ghi.

Hãy chú ý những gì bản ghi không nói: `environment` là `null` vì Ghidra chỉ đọc một tệp tĩnh chứ không chạy chương trình. `locations` rỗng vì thao tác này báo cáo địa chỉ ngay bên trong `normalized_result`. `relationship_coverage` có trạng thái `complete`, nhưng phần chẩn đoán nêu rõ rằng nó chỉ phản ánh các tham chiếu lời gọi đã được Ghidra giải quyết; các lời gọi gián tiếp hay tính toán chưa xác định vẫn ở trạng thái chưa biết.

## Mã Định Danh Là Bản Băm Nội Dung Và Không Chứa Đường Dẫn Tệp

rea tính toán `evidence_id` bằng tiền tố `ev_` cộng với mã băm SHA-256 trên chuỗi JSON chuẩn hóa RFC 8785 của một phép chiếu ngữ nghĩa (semantic projection). Phép chiếu này giữ lại `digest`, `format`, và `architecture` của đối tượng mục tiêu, đồng thời loại bỏ `name` và `local_path`. Nó cũng loại bỏ `relationship_coverage` khi giá trị là `null`, giúp cho bản ghi không có quan hệ lời gọi vẫn giữ nguyên mã định danh như trước khi trường này xuất hiện. Toàn bộ phần còn lại (bao gồm `raw_result` và mọi câu giới hạn) đều được băm.

Quy chuẩn RFC 8785 sắp xếp các khóa của đối tượng theo thứ tự từ điển, loại bỏ khoảng trắng thừa và chuẩn hóa định dạng số và chuỗi. Đối với các bản ghi mà số đều là số nguyên, thư viện chuẩn của Python tạo ra chuỗi byte tương đương:

```python
def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

Mã nguồn `code/main.py` từ chối các số thực (floats) trước khi băm, bởi vì RFC 8785 và Python có thể định dạng một số số thực khác nhau. Hai bản ghi rea và snapshot đều không chứa số thực, do đó công thức này tái tạo chuẩn xác từng mã băm của rea:

| Giá trị gốc trong rea | Tính toán lại bởi `code/main.py` |
|---|---|
| `ev_67b6d3f992eaf49a…` cho hàm `grade` | Trùng khớp tuyệt đối |
| `ev_e88b10dc0c6400e5…` cho hàm `percent_change` | Trùng khớp tuyệt đối |
| Mã băm profile `9c7f6b24be69294e…` trên `{provider, parameters}` | Trùng khớp tuyệt đối |
| `query_26122aebe9d8abae…` cho mục snapshot của `grade` | Trùng khớp tuyệt đối |
| `query_0f95ef38038d3666…` cho mục của `percent_change` | Trùng khớp tuyệt đối |

Tại sao lại loại trừ đường dẫn tệp? Vì cùng một quan sát trên cùng các byte nhị phân phải nhận cùng một mã định danh trên mọi máy tính và trong mọi thư mục. Trường `local_path` trong ví dụ trên chỉ ra một thư mục tạm trên một máy Mac cụ thể; mã định danh không được phép thay đổi khi bạn sao chép fixture sang thư mục `/home/you/re`. Hàm `rebind_subject_path()` cho phép bạn hiển thị một bản ghi dưới một đường dẫn mới; nó sẽ từ chối nếu `sha256`, `format` hoặc `architecture` bị sai khác. rea 6.5.0 cũng làm điều tương tự với `rebindEvidenceToTargetPath` khi phát lại một bản ghi snapshot cho một mục tiêu tại đường dẫn mới.

Hàm `validate_evidence()` tính toán lại mã định danh và từ chối nếu có sự bất đồng bằng câu thông báo của rea: "Evidence semantic identifier does not match its record" (Mã định danh ngữ nghĩa của bằng chứng không khớp với bản ghi của nó). Nó cũng từ chối bản ghi nếu `analysis_profile.provider` khác với `provider`. Bài kiểm thử `test_tampered_id_and_mismatched_profile_are_rejected` sửa đổi một dòng mã giả bên trong bản ghi thực tế của `grade` và chứng kiến sổ cái kiên quyết từ chối nó.

## Tự Quan Sát Đối Tượng Nhị Phân

Trước khi sử dụng bất kỳ công cụ ngoài nào, sổ cái sẽ tự đọc tệp nhị phân. Hàm `observe_object()` phân tích các tiêu đề phân đoạn ELF64, `.symtab`, `.strtab`, và `.rela.text` bằng `struct.unpack_from`. Nó ghi nhận một bản ghi `inspect_object` và một bản ghi `read_function_bytes` cho mỗi ký hiệu hàm. Cả 5 bản ghi này đều mang độ tin cậy `observed` với thẩm quyền `shipped-artifact`, bởi nguồn gốc là tệp thực tế trên đĩa.

| Bản ghi | Nội dung tóm tắt | Mã định danh |
|---|---|---|
| `inspect_object` | 5 phân đoạn kèm độ lệch, kích thước và mã băm; 4 hàm; 3 tái định vị | `ev_645e0a145f28bd5c…` |
| `read_function_bytes percent_change` | 16 byte tại `.text+0x00` | `ev_2572a7d254ad15cb…` |
| `read_function_bytes grade` | 46 byte tại `.text+0x10`, SHA-256 `ffcb75c2a7bb1104…` | `ev_a467c2427bfc0c56…` |
| `read_function_bytes total` | 31 byte tại `.text+0x40` | `ev_b1e845cea2543073…` |
| `read_function_bytes grade_label` | 63 byte tại `.text+0x60` | `ev_c50041be26e6b7f4…` |

Lệnh `llvm-objdump -t` hoàn toàn đồng thuận: hàm `grade` tại `0000000000000010` với kích thước `2e`. Ba bản ghi tái định vị trong `.rela.text` đều là `R_X86_64_PC32` trỏ vào `.rodata` tại các độ lệch `0x36`, `0x84`, và `0x93`, với các số hạng cộng tương ứng là `0x10`, `0x24`, và `-0x4`. Mỗi bản ghi đều mang vị trí kiểu `file-offset-range` và một câu giới hạn: "Section bytes are read from the file on disk. The reader applies no relocation and runs no code." (Các byte phân đoạn được đọc trực tiếp từ tệp trên đĩa. Bộ đọc không áp dụng tái định vị và không thực thi mã máy nào).

Bài học 32 đã tự tay xây dựng toàn bộ bộ đọc này. Ở đây nó được tóm gọn trong 60 dòng mã, bởi trọng tâm bài học này là cách bạn ghi nhận các dữ kiện thu được từ lần đọc đó.

## Dẫn Xuất Bảng Dữ Liệu Và Tuyên Bố Rõ Việc Dẫn Xuất

Toàn bộ 46 byte của hàm `grade` chứa 4 mẫu lệnh mà bài học 33 đã giải mã:

| Chuỗi byte | Lệnh máy | Độ lệch trong `grade` |
|---|---|---|
| `48 69 c0 67 66 66 66` | `imul rax, rax, 0x66666667` | 3 |
| `48 c1 f9 22` | `sar rcx, 34` | 13 |
| `83 c1 fa` | `add ecx, -6` | 23 |
| `83 f9 04 77 0c` | `cmp ecx, 4` tiếp theo `ja` | 28 |

Một phép nhân với `0x66666667` theo sau bởi dịch phải 34 bit là một phép chia số nguyên có dấu cho 10. Nhưng tệp nhị phân không hề chứa số 10. Hàm `derive_grade_model()` ghi nhận dữ kiện đó thành một giới hạn: "The divisor 10 is a derivation from the multiplier 0x66666667 and the shift 34. The file holds no 10." (Số chia 10 là kết quả dẫn xuất từ thừa số nhân 0x66666667 và phép dịch 34 bit. Tệp nhị phân hoàn toàn không chứa số 10).

Lệnh `add ecx, -6` thiết lập trường hợp nhánh đầu tiên là 6. Lệnh `cmp ecx, 4` với bước nhảy không dấu cho phép đúng 5 trường hợp: từ 6 đến 10.

Lệnh `lea` tại `.text+0x33` mang bản ghi tái định vị với số hạng cộng `0x10`. Phép tái định vị PC32 đo từ cuối trường độ dời 4-byte, do đó bảng dữ liệu bắt đầu tại `.rodata+0x14`. Năm từ little-endian tại đó chứa các số: `1, 2, 3, 4, 4`. Câu lệnh switch trong `ledger.c` trả về 1 cho điểm 60-69, 2 cho đầu 7, 3 cho đầu 8, và 4 cho 90-100. Trình biên dịch đã chuyển hóa câu lệnh switch thành một bảng tra cứu giá trị trả về tĩnh.

Bản ghi kết quả `ev_605106bff2d9998f…` có `confidence` là `derived`, và trường `evidence_links` của nó nêu rõ hai bản ghi quan sát mà nó dựa vào. Nếu xóa bản ghi byte khỏi sổ cái, bản ghi dẫn xuất sẽ có một liên kết treo (dangling link). Sơ đồ hình vẽ ở trên minh họa trạng thái đó khi bạn tắt `ev_a467c242…`.

Bây giờ hãy so sánh với Ghidra. Mã giả của nó cho cùng các byte lệnh:

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

Ghidra đã áp dụng tái định vị và đặt tên cho bảng là `DAT_001000b4` (chính là `.rodata+0x14` sau khi nó nạp `.text` tại `0x100000`). Nó khôi phục phép chia cho 10 và 5 nhánh rẽ. Đối với giá trị mặc định, nó in ra một biểu thức luôn bằng 0 nhưng không bao giờ nói là 0. Mã trung gian p-code trong `native_value_flow` đánh dấu các thao tác đó là `is_dead: true`. Bản ghi là một sự quan sát trung thực về Ghidra; nhưng ý nghĩa bạn rút ra từ nó là dẫn xuất (`derived`), và giá trị bạn kỳ vọng từ một lần chạy thực tế là suy luận (`inferred`).

Bản ghi của `percent_change` vạch trần một sai lầm khác: Chữ ký của Ghidra là `undefined1 [16] percent_change(int param_1,int param_2,ulong param_3)`. Mã nguồn gốc là `int percent_change(int before, int after)`. Lệnh `idiv` ghi vào `EDX:EAX`, và nhánh trả về 0 để nguyên `EDX`, nên trình dịch ngược đã suy đoán một giá trị trả về 16-byte cùng tham số thứ ba. Thuộc tính `native_api.return_type.confidence` của chính nó cho hàm `grade` bị đánh giá là `"low"`. Hãy đối xử với chữ ký hàm được khôi phục như một sự suy luận của công cụ, và ghi nhận nó đúng như vậy.

## Các Dự Đoán Là Bản Ghi Suy Luận (Inferred Records)

Một dự đoán là một khẳng định về hành vi mà chưa có quan sát thực nghiệm nào xác nhận. Hàm `claim_suite()` xây dựng 18 khẳng định từ các mô hình dẫn xuất và mã nguồn. Hàm `record_prediction()` lưu từng khẳng định thành một bản ghi với `confidence` là `inferred` và `authority` là `analyst-inference`, được liên kết chặt chẽ với các bản ghi mà nó dựa vào:

| Khẳng định | Giá trị dự đoán | Dựa trên |
|---|---|---|
| `percent_change(12400, 15500)` | 25, từ 310000 / 12400 | mô hình phép chia `ev_076a9167…` |
| `percent_change(0, 5)` | 0, lính canh kiểm tra số 0 | mô hình phép chia |
| `percent_change(100, 50)` | -50 | mô hình phép chia |
| `percent_change(-7, 3)` | -142, vì `idiv` làm tròn về phía số 0 | mô hình phép chia |
| `grade(55)` | 0, chỉ số -1 trượt phép so sánh không dấu | mô hình bảng switch `ev_605106bf…` |
| `grade(60)`, `grade(69)` | 1, 1 | mô hình bảng switch |
| `grade(70)`, `grade(85)` | 2, 3 | mô hình bảng switch |
| `grade(95)`, `grade(100)`, `grade(105)` | 4, 4, 4 | mô hình bảng switch |
| `grade(110)`, `grade(-5)` | 0, 0 | mô hình bảng switch |
| `total(1, 2, 3)`, `total()` | 6, 0 | bản ghi mã nguồn `ev_2f1e6d25…` |
| `grade_label(85)` | `"B"` | bản ghi nguồn và mô hình bảng |
| `grade(score)` cho 90 đến 100 | 4 cho toàn bộ 11 giá trị đầu vào | mô hình bảng switch |

Hàm `predict()` sử dụng ngữ nghĩa số học của C chứ không phải của Python. Hàm `c_div()` làm tròn về phía số 0, và `wrap32()` gói gọn mọi kết quả trung gian về 32 bit. Toán tử `//` của Python làm tròn về phía âm vô cùng, do đó `(3 - -7) * 100 // -7` trong Python ra -143 nhưng chương trình C thực tế lại cho ra -142! Bài học 36 sẽ mở rộng lưới kiểm thử này lớn hơn rất nhiều.

Mỗi dự đoán sẽ mở ra một yếu tố chưa biết tồn đọng (residual unknown). Yếu tố chưa biết này yêu cầu một bằng chứng `observed` với thẩm quyền `controlled-replay` và khuyến nghị một phép thăm dò cụ thể: gọi hàm `execute_function` trên bản dựng host. Phần tiếp theo sẽ sản sinh ra bằng chứng đó. Câu hỏi cho khẳng định điểm số đầu tiên là: "Does grade(95) == 4 hold when the compiled function runs?" (Khẳng định grade(95) == 4 có thực sự đúng khi hàm đã biên dịch chạy hay không?).

## Kiểm Tra Đối Chiếu Với Một Lần Chạy Thực Tế

Tệp `code/fixtures/harness.c` chứa mã nguồn nhúng `ledger.c` và in ra kết quả cho từng lần gọi. Hàm `observe_runs()` biên dịch nó bằng trình biên dịch đầu tiên tìm thấy trên PATH (`clang`, `cc`, hoặc `gcc`) và chạy từng đầu vào trong lưới thử nghiệm một lần. Mỗi lần chạy trở thành một bản ghi bằng chứng:

```json
{
  "operation": "execute_function",
  "parameters": { "function": "grade", "args": [95] },
  "normalized_result": { "result": 4, "stdout": "4\n", "exit_code": 0 },
  "confidence": "observed",
  "authority": "controlled-replay",
  "environment": { "id": "host-process", "platform": "darwin", "architecture": "arm64", "isolation": "process" },
  "limitations": [
    "The run executed a host build of ledger.c, not the x86-64 object. The claim about the object rests on the shared source and on the compiler.",
    "One run observes one input. It proves nothing about other inputs."
  ]
}
```

Đối tượng mục tiêu của bản ghi thực thi là tệp nhị phân harness (một tệp Mach-O arm64 trên máy Mac này) với mã SHA-256 riêng biệt. Nhà cung cấp là `course-harness` kèm dòng thông tin phiên bản của trình biên dịch (ở đây là `Apple clang version 17.0.0 (clang-1700.6.3.2)`). Danh sách `evidence_links` liên kết tới bản ghi mã nguồn `ev_2f1e6d251543d017…`, vì lần chạy và tệp đối tượng chỉ chia sẻ chung tệp `ledger.c` chứ không có gì khác.

Hãy đọc câu giới hạn đầu tiên hai lần: Tệp đối tượng x86-64 chưa từng thực sự chạy trên chiếc máy Mac này. Thứ vừa chạy là một bản dựng thứ hai của cùng mã nguồn nhưng dành cho một kiến trúc khác! Đó vẫn là một quan sát thực nghiệm, và là quan sát mạnh nhất có thể đạt được nếu không dùng trình giả lập, vì vậy khẳng định được phê duyệt kèm theo câu văn bản đó. Một báo cáo bỏ qua câu này đã làm mất đi thông tin cốt tử mà người đánh giá cần biết.

Toàn bộ 17 lần chạy đơn lẻ đều đồng thuận hoàn toàn với các dự đoán. Tệp `code/fixtures/run-evidence.json` lưu giữ 17 bản ghi này cùng bản ghi mã nguồn, được trích xuất bằng cờ `--record` ngày 2026-10-13. Khi môi trường không có sẵn trình biên dịch C, hàm `build_ledger()` sẽ tự động nhập gói dữ liệu này thay vì cố chạy lệnh, và bản báo cáo sẽ nêu rõ điều đó trong trường `runs`.

## Quy Tắc Thăng Cấp Khẳng Định (Promotion Rules)

Một khẳng định chỉ được chuyển từ trạng thái suy luận (`inferred`) sang trạng thái được hỗ trợ (`supported`) thông qua các lần chạy quan sát được. Hàm `check_claim()` tìm kiếm các bản ghi `execute_function` có cùng tên hàm và đối số, so sánh kết quả với dự đoán, và đếm số lượng đầu vào được bao phủ:

| Trạng thái | Điều kiện kích hoạt | Trạng thái trong sổ cái hoàn tất của rea |
|---|---|---|
| `pass` | Mọi đầu vào đều có lần chạy đồng thuận và không có lần chạy nào bất đồng | `pass` |
| `fail` | Có bất kỳ lần chạy nào cho kết quả bất đồng | `fail` |
| `unknown` | Một số đầu vào có kết quả chạy, một số đầu vào khác chưa có | `unknown` |
| `unsupported` | Không có bất kỳ đầu vào nào có kết quả chạy | `unsupported` |

Hàm `promote_claim()` áp dụng quy tắc nghiêm ngặt: Nó ném ngoại lệ `ClaimError` khi bất kỳ mã định danh nào mang mức độ tin cậy `inferred` hoặc `derived`. Nó ném ngoại lệ khi mã định danh trỏ tới một lần chạy của hàm khác hoặc đối số khác. Nó ném ngoại lệ khi danh sách rỗng, hoặc khi các lần chạy bao phủ ít đầu vào hơn phạm vi khẳng định. Cả bản ghi mã giả Ghidra lẫn mô hình bảng đều tồn tại cho `grade(55)`, nhưng không nguồn nào có thể thăng cấp khẳng định này. Tắt `ev_7be90c0c…` trong hình vẽ và quan sát khẳng định tụt xuống `unsupported` dù vẫn còn hai bản ghi khác về `grade`.

Khẳng định dải giá trị (range claim) minh họa rõ trường hợp bao phủ một phần: Chỉ có 2 trong số 11 giá trị đầu vào (95 và 100) xuất hiện trong lưới chạy. Hàm `check_claim()` trả về `unknown` với độ bao phủ `covered` là 2/11, và `promote_claim()` từ chối thăng cấp với lý do: "2 of 11 inputs have observed runs". Sổ cái tuyệt đối không được phép làm tròn hay giả định phần còn lại.

Hàm `completion_ledger()` mô phỏng cấu trúc `EvidenceCompletionLedger` của rea. Mã định danh của nó là `ecl_` cộng với mã băm các bản ghi của nó, và cờ `complete` chỉ bằng `true` khi số lượng `pass` bằng đúng tổng số `total`. Bản demo in ra `ecl_da64933ea13befed…` với `complete: false`, chính là vì khẳng định dải giá trị còn đang dở dang kia.

## Các Yếu Tố Chưa Biết Là Bản Ghi Có Chuỗi Băm

rea quản lý các yếu tố chưa biết tồn đọng dưới dạng các bản ghi có cấu trúc nghiêm ngặt. Mã nguồn `code/main.py` duy trì cùng các trường dữ liệu và các bất biến đó.

Mã định danh là `unk_` cộng với SHA-256 trên `{domain, question, scope_digest}`, với khoảng trắng của câu hỏi được chuẩn hóa thành dấu cách đơn. Hai nhà phân tích đặt cùng một câu hỏi về cùng một tệp sẽ luôn nhận cùng một mã định danh. Trường `scope_digest` chính là mã SHA-256 của tệp đối tượng.

Mỗi thay đổi là một bản sửa đổi (revision) mới: Bản sửa đổi 1 có `previous_revision_digest` là `null`. Mọi bản sửa đổi sau đó đều nêu rõ `revision_digest` của bản sửa đổi ngay trước nó, và mã băm này bao trùm mọi trường ngoại trừ chính nó. Hàm `verify_unknown_chain()` duyệt qua danh sách và tính toán lại từng mã băm: Nó kiểm tra xem mỗi mắt xích có trỏ đúng vào bản sửa đổi trước đó không, và kiểm tra xem bằng chứng biến đổi (mutation evidence) mà mỗi bản sửa đổi viện dẫn có thực sự nằm trong sổ cái hay không.

Bản thân mỗi biến đổi là một bản ghi bằng chứng, tương tự như cách rea làm với nhà cung cấp `rea-unknown-registry`. Bản ghi của chúng ta mang độ tin cậy `derived`, thẩm quyền `analyst-inference`, và không có đối tượng mục tiêu. Giới hạn của nó nêu rõ: "A registry mutation records what the analyst decided. It is not proof of the answer." (Một biến đổi trong sổ đăng ký chỉ ghi nhận những gì chuyên viên phân tích đã quyết định. Nó không phải là bằng chứng của câu trả lời).

Đó là lý do tại sao 81 bản ghi của bản demo bao gồm tới 36 bản ghi biến đổi dẫn xuất: 18 bản ghi mở câu hỏi và 18 bản ghi cập nhật trạng thái.

Ba bất biến bắt buộc phải được thực thi trên mọi bản sửa đổi:

- Chỉ một yếu tố chưa biết ở trạng thái `resolved` (đã giải quyết) mới được phép mang trường `resolution`, và chỉ bản sửa đổi 1 mới có tiền nhiệm là null.
- Một yếu tố chưa biết ở trạng thái `contradicted` (bị mâu thuẫn) bắt buộc phải chỉ ra ít nhất một mã bằng chứng mâu thuẫn, và không có mã nào được phép xuất hiện đồng thời ở cả hai danh sách ủng hộ và mâu thuẫn.
- Một giải pháp ở trạng thái `verified` bắt buộc phải viện dẫn ít nhất một mã bằng chứng có độ tin cậy và thẩm quyền đáp ứng yêu cầu.

Quy tắc thứ ba là nơi sổ cái từ chối thường xuyên nhất: Tài liệu quy trình của rea nêu rõ "Only qualifying observed evidence can verify a resolution." (Chỉ bằng chứng quan sát thực nghiệm đủ điều kiện mới có thể xác minh một giải pháp). Do đó, hàm `resolve_unknown()` kiên quyết từ chối mô hình bảng dẫn xuất, từ chối bản ghi quan sát có thẩm quyền `shipped-artifact`, và chỉ chấp nhận bản ghi thực thi `ev_f4ffbe8f922ec7fe…`. Bản sửa đổi đã giải quyết cho `grade(95)` là bản sửa đổi 2, và `previous_revision_digest` của nó là `ffca840b55a8a188…` (mã băm của bản sửa đổi 1).

Hàm `update_unknown()` nhận tham số `expected_revision` và ném ngoại lệ `revision-conflict` khi bản sửa đổi bị lỗi thời, tương tự như công cụ `update_unknown` của rea. Phản ứng chuẩn mực là đọc lại bản sửa đổi hiện tại và đưa ra quyết định mới.

Một yếu tố chưa biết vẫn còn mở sau khi bản demo kết thúc: `unk_f2eb0d7f84c80a19…`, với câu hỏi: "Does grade(score) == 4 for every score from 90 to 100 hold when the compiled function runs?". Trạng thái của nó là `investigating` (đang điều tra), danh sách bằng chứng ủng hộ chứa 2 lần chạy, và phép thăm dò vẫn nêu rõ 9 đầu vào còn thiếu.

## Bộ Nhớ Đệm Câu Trả Lời Không Phải Là Cơ Sở Dữ Liệu Phân Tích

rea có ba cơ chế trông giống như bộ nhớ đệm: Target snapshot của Ghidra là một bản sao riêng tư của đầu vào, được nhập với các cờ `-readOnly -deleteProject` và tự hủy khi đóng. Bộ đệm trong bộ nhớ `AnalysisSnapshotCache` lưu trữ tới 10.000 mục cho một định danh nhị phân. Tệp snapshot lưu trữ `target`, `binding`, `entries`, `workflow_entries`, và một gói `evidence_bundle`.

Khóa bộ nhớ đệm là `query_` cộng với SHA-256 trên `{target, binding, operation, parameters}`. `target` gồm có `{sha256, kind, format, architecture}`, và một mục tiêu `raw-binary` sẽ có thêm `entry_point`. `binding` bao gồm nhà cung cấp và toàn bộ cấu hình phân tích (kèm theo mã băm). Thay đổi bất kỳ thông số nào trong 25 thông số cấu hình (chẳng hạn nâng `jump_table_evidence` từ `typed-case-default-blocks-v1` lên bản `v2`), mã băm sẽ thay đổi và toàn bộ các khóa đệm đều thay đổi theo!

Tệp `code/fixtures/rea-snapshot.json` là tệp mà rea đã ghi lại trong hai lần chạy, có dung lượng 142.690 byte. Phương thức `AnswerCache.import_snapshot()` tính toán lại từng `query_id` và từ chối tệp nếu có bất kỳ mã nào sai khác. Nó từ chối một snapshot nếu `target` hoặc `binding` khác với cấu hình đang hoạt động với thông báo lỗi của rea: `profile_mismatch`. Nó cũng từ chối nếu `format_version` không phải là `course-answer-cache-1`.

Ba phép đo thực tế trên chiếc máy này: Lần truy vấn đầu tiên cho `grade` mất 8,6 giây (kèm việc nhập Ghidra mới). Lần truy vấn `percent_change` trên cùng snapshot mất 8,5 giây (vì truy vấn khác là một lần trượt cache). Lần truy vấn thứ ba cho `grade` chỉ mất 2,1 giây và trả về bản ghi y hệt với cùng mã `evidence_id`. Quá trình phát lại từ bộ nhớ trong rea bổ sung câu giới hạn: "Loaded from a local REA analysis snapshot; this call did not re-run the provider." (Được nạp từ bản chụp phân tích REA cục bộ; lời gọi này không chạy lại nhà cung cấp). Hàm `lookup()` của chúng ta cũng bổ sung một câu tương tự để người đọc phân biệt rõ việc phát lại dữ liệu đệm với một câu trả lời mới tinh.

Một biến đổi dữ liệu sẽ lập tức vô hiệu hóa bộ nhớ đệm: Trong rea 6.5.0, một thao tác đổi tên hoặc thêm chú thích sẽ xóa sạch bộ đệm snapshot trước khi rea gửi lời gọi tới nhà cung cấp. Một phản hồi thất bại không thể chứng minh rằng việc ghi đã được hoàn nguyên, vì vậy một lần ghi lỗi cũng sẽ dừng việc phát lại cache.

## Tại Sao Một AI Agent Cần Hệ Thống Này?

Một agent điều khiển rea thông qua MCP nhìn thấy bản ghi đầy đủ trong mỗi lời gọi và có thể trích dẫn `evidence_id` trong báo cáo của mình. Các nghiên cứu năm 2026 đã chỉ ra chính xác nơi mà quy trình này sụp đổ: Sự tin tưởng thái quá vào đầu ra công cụ là một trong bốn điểm yếu trong nghiên cứu NDSS BAR. Trong nghiên cứu cài cắm văn bản, việc chỉ hiển thị nội dung nhị phân dưới dạng dữ liệu vẫn để lại 15/40 đề xuất không an toàn; chỉ có kiểm tra nguồn gốc dữ liệu kèm ủy quyền công cụ mới loại bỏ được chúng hoàn toàn. Một nghiên cứu thứ ba theo dõi một agent điều khiển Ghidra qua MCP ([arXiv 2606.06838](https://arxiv.org/abs/2606.06838)): Trên các tệp nhị phân có hàng trăm hàm, chỉ có 10 đến 15% nhận được sự phân tích thực sự trước khi agent bắt đầu tạo ra các hàm giả rỗng (stubs).

Sổ cái bằng chứng giải quyết từng vấn đề này bằng một bản ghi tường minh, và agent chỉ được phép báo cáo những gì mà sổ cái thực sự nắm giữ: Một khẳng định không có lần chạy quan sát được sẽ là `unsupported`. Một khẳng định chỉ có một vài lần chạy sẽ là `unknown`. Một văn bản từ công cụ chỉ là `observed` về chính công cụ đó chứ không phải về hành vi chương trình. Hãy giao cho agent tệp kỹ năng trong thư mục `outputs/` và biến sổ cái thành nơi duy nhất mà nó được phép ghi lại các kết luận. Quy tắc này áp dụng cho Claude, Cursor, Codex, OpenClaw, Hermes hoặc bất kỳ agent nào đọc tệp SKILL.md.

## Xây Dựng Mã Nguồn (Build It)

Chạy chương trình demo thuần thư viện chuẩn:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
```

Bản demo đọc tệp đối tượng, nhập hai bản ghi rea, xác thực snapshot của rea và dẫn xuất hai mô hình. Sau đó, nó ghi nhận 18 dự đoán, mở 18 yếu tố chưa biết, biên dịch và chạy harness kiểm thử, kiểm tra từng khẳng định, và giải quyết 17 yếu tố chưa biết. Dạng đầu ra kỳ vọng:

```json
{
  "records": 81,
  "by_confidence": { "observed": 25, "derived": 38, "inferred": 18 },
  "rea_snapshot": { "entries": 2, "replay_hit": true, "stale_profile_refused": true },
  "grade_table": [1, 2, 3, 4, 4],
  "runs": "live: Apple clang version 17.0.0 (clang-1700.6.3.2)",
  "claims": [ { "claim_id": "grade(95)", "status": "pass", "covered": "1/1" }, { "claim_id": "grade(90..100)", "status": "unknown", "covered": "2/11" } ],
  "completion_ledger_id": "ecl_da64933ea13befed9483f8b00447bb3f4ceb74324c01e2515eff03f3af3ca750",
  "complete": false,
  "unknowns": { "resolved": 17, "investigating": 1 }
}
```

Nếu không có trình biên dịch C, trường `runs` sẽ ghi `imported 17 recorded run records from run-evidence.json` và toàn bộ phần còn lại giữ nguyên.

Hãy đọc `code/main.py` theo trình tự sau:

1. `canonical_json()` và `compute_evidence_id()` chứa công thức băm chuẩn hóa và phép chiếu ngữ nghĩa.
2. `create_evidence()` và `validate_evidence()` xây dựng và kiểm tra tính toàn vẹn của một bản ghi.
3. `EvidenceLedger.record()` và `import_bundle()` lưu trữ các bản ghi và từ chối xung đột.
4. `open_unknown()`, `update_unknown()`, và `resolve_unknown()` duy trì chuỗi bản sửa đổi của các yếu tố chưa biết.
5. `observe_object()` đọc tệp ELF, và `derive_grade_model()` giải mã bảng dữ liệu.
6. `claim_suite()`, `predict()`, và `record_prediction()` ghi nhận tầng suy luận.
7. `observe_runs()` biến mỗi lần thực thi harness thành một bản ghi quan sát thực nghiệm.
8. `check_claim()`, `promote_claim()`, và `completion_ledger()` áp dụng các quy tắc thăng cấp khẳng định.
9. `AnswerCache` lập chỉ mục câu trả lời tương tự snapshot của rea và từ chối các câu trả lời lỗi thời.

## Sử Dụng Mã Nguồn (Use It)

Duy trì một sổ cái duy nhất cho mỗi mục tiêu và cho mỗi cuộc điều tra. Đặt mã SHA-256 của mục tiêu vào mọi trường `subject` và đặt mã băm phạm vi câu hỏi vào mọi yếu tố chưa biết. Khi bạn thay đổi công cụ, hãy đổi trường `provider` và để các mã định danh phân kỳ tự nhiên. Khi bạn biên dịch lại fixture, mã băm đối tượng sẽ thay đổi và toàn bộ các bản ghi cũ sẽ tự động trở thành bằng chứng về một tệp khác trong quá khứ.

Hãy viết các câu giới hạn (limitations) ngay tại thời điểm bạn nhận thức được chúng: Bộ đọc không áp dụng tái định vị phải nêu rõ điều đó; lần chạy trên bản dựng host phải nêu rõ điều đó; mô hình suy luận số chia phải nêu rõ điều đó. Một bản ghi được ghi lại mà thiếu đi các giới hạn sẽ trông có vẻ mạnh mẽ hơn thực tế, và một người đọc sau này sẽ không có cách nào sửa chữa được sự sai lệch đó.

Hãy coi mọi văn bản từ công cụ phân tích là dữ liệu đơn thuần: Các chuỗi ký tự và mã giả trích xuất từ mục tiêu là văn bản do chính mục tiêu kiểm soát. Chúng không bao giờ được trở thành lệnh máy, và không bao giờ được xem là quan sát về hành vi thời gian chạy. rea 6.5.0 đánh dấu các kết quả này bằng cờ `_meta.provenance.untrusted_content: true`. Các chỉ dẫn máy chủ của nó yêu cầu mô hình phải đối xử với nội dung mục tiêu như dữ liệu và báo cáo bất kỳ yêu cầu thực thi, cài đặt, tải xuống hay gửi dữ liệu nào như một cuộc tấn công tiêm lời nhắc (prompt-injection). Dấu hiệu cảnh báo rất hữu ích, nhưng sổ cái của bạn vẫn phải là bức tường kiên cố giữ cho văn bản mục tiêu nằm ngoài tập lệnh thực thi.

## Phòng Thí Nghiệm Tương Tác (Interactive Lab)

### Lab A: Tính toán lại mã định danh rea

Từ thư mục `code`, mở môi trường tương tác Python:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns/code
python3 -q
```

Chạy:

```python
from main import *
record = load_evidence_file(FIXTURES / "rea-evidence-grade.json")
record["evidence_id"] == compute_evidence_id(record)
record["subject"]["local_path"] = "/home/you/re/ledger-x86_64-O1.o"
record["evidence_id"] == compute_evidence_id(record)
record["limitations"].pop()
record["evidence_id"] == compute_evidence_id(record)
```

Hai phép so sánh đầu tiên trả về `True`: Đường dẫn tệp không nằm trong mã băm. Phép so sánh thứ ba trả về `False`: vì câu giới hạn là một phần cấu thành nội dung!

### Lab B: Từ chối thăng cấp khẳng định

```python
ledger, report = build_ledger(live_runs=False)
claim = [c for c in claim_suite(ledger.find(operation="derive_switch_model")[0], ledger.find(operation="derive_arithmetic_model")[0], ledger.find(operation="read_source")[0]) if c.claim_id == "grade(55)"][0]
promote_claim(ledger, claim, [ledger.find(operation="derive_switch_model")[0]["evidence_id"]])
```

Lời gọi ném ngoại lệ `ClaimError` và chỉ rõ mã định danh dẫn xuất. Bây giờ hãy truyền mã định danh của lần chạy thực tế:

```python
run = run_records_for(ledger, "grade", [55])[0]
promote_claim(ledger, claim, [run["evidence_id"]])["status"]
```

Trạng thái chuyển thành `pass`, và mảng `limitations` trả về mang theo câu cảnh báo về bản dựng host.

### Lab C: Từ chối snapshot lỗi thời

```python
import json
snapshot = json.loads(REA_SNAPSHOT_PATH.read_text())
target, binding = rea_snapshot_binding(snapshot)
AnswerCache(target, binding).import_snapshot(snapshot)
stale = dict(binding["analysis_profile"]["parameters"], jump_table_evidence="typed-case-default-blocks-v2")
AnswerCache(target, {"provider": binding["provider"], "analysis_profile": make_profile(binding["provider"], stale)}).import_snapshot(snapshot)
```

Lần nhập đầu tiên trả về 2 mục. Lần nhập thứ hai ném ngoại lệ `CacheError` với lý do `profile_mismatch`. Chỉ cần một chuỗi phiên bản thay đổi, toàn bộ snapshot bị từ chối phát lại ngay lập tức.

### Lab D: Phá vỡ chuỗi băm của yếu tố chưa biết

```python
unknown_id = [u for u, r in ledger.unknowns.items() if r[-1]["status"] == "investigating"][0]
ledger.verify_unknown_chain(unknown_id)
ledger.unknowns[unknown_id][0]["severity"] = "critical"
ledger.verify_unknown_chain(unknown_id)
```

Lời gọi đầu tiên trả về 2. Lời gọi thứ hai ném ngoại lệ `EvidenceIntegrityError`, bởi vì bản sửa đổi 1 không còn khớp với mã băm của chính nó, và bản sửa đổi 2 đang trỏ vào một mã băm tiền nhiệm không còn tồn tại!

## Bài Tập Thực Hành (Practice Lab)

Khép lại yếu tố chưa biết đang mở và bổ sung một nhà cung cấp công cụ thứ hai.

Các yêu cầu:

- Mở rộng hàm `run_grid()` để 9 đầu vào còn thiếu của khẳng định dải giá trị được thực thi, sau đó xác nhận yếu tố chưa biết chuyển sang trạng thái `resolved` ở bản sửa đổi 3.
- Bổ sung `llvm-objdump -d` làm nhà cung cấp thứ hai với đối tượng `provider` riêng biệt và ghi nhận danh sách mã máy của `grade` dưới dạng `observed`.
- Viết một bản ghi `derived` khẳng định rằng cả hai danh sách mã máy (của Ghidra và của llvm-objdump) đều giải mã ra cùng 14 lệnh, và liên kết tới cả hai bản ghi gốc.
- Mở một yếu tố chưa biết cho `grade_label(105)` với mức độ nghiêm trọng `high` (vì chỉ số bảng 4 đọc `labels[4]`), và giải quyết nó bằng một lần chạy thực tế.
- Xuất toàn bộ sổ cái bằng `export_bundle()`, nhập nó vào một sổ cái mới tinh, và chứng minh mọi chuỗi băm đều tiếp tục được xác thực thành công.

Thành công đồng nghĩa với việc bạn có một bản báo cáo mà trong đó từng câu khẳng định đều viện dẫn một mã định danh từ gói dữ liệu và khẳng định `unknown` duy nhất đã biến mất hoàn toàn.

## Bài Tập Đánh Giá (Graded Lab)

Năm câu hỏi về các fixture của bài học này. Hãy tìm câu trả lời bằng chính mã nguồn bạn đã xây dựng, sau đó ghi kết quả vào `code/lab_answers.json`. Ghi số dưới dạng số nguyên hoặc chuỗi thập lục phân như `"0x40"`, và danh sách dưới dạng mảng JSON.

1. `t1`: Mã định danh bằng chứng mà hàm `compute_evidence_id` tính toán cho tệp `rea-evidence-grade.json`.
2. `t2`: Số lượng câu giới hạn (limitations) có trong bản ghi của hàm `grade`.
3. `t3`: Trạng thái `relationship_coverage` của bản ghi hàm `grade`.
4. `t4`: Mức độ tin cậy (`confidence`) và thẩm quyền (`authority`) của bản ghi hàm `grade`, dưới dạng danh sách gồm hai chuỗi.
5. `t5`: Mã băm SHA-256 của đối tượng mục tiêu (`subject`) trong bản ghi hàm `grade`.

Kiểm tra câu trả lời của bạn:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/lab.py
```

Trình kiểm tra sẽ tính toán các giá trị kỳ vọng từ `code/main.py` trên các fixture đã cam kết. Nó in ra `PASS`, `FAIL` hoặc `TODO` cho từng nhiệm vụ cùng điểm số, và chỉ thoát với mã 0 khi cả 5 bài đều đạt. Lệnh `python3 code/lab.py my-answers.json` cho phép kiểm tra một tệp câu trả lời khác. Bài kiểm thử `code/tests/test_lab.py` chứng minh trình chấm chấp nhận câu trả lời đúng và từ chối câu trả lời sai.

### Trực Quan Hóa Hoạt Động

Cỗ máy tương tác [Inside a Decompiler](https://aiengineeringfromscratch.com/inside-a-decompiler?symbols=off) duy trì một sổ cái bằng chứng với bốn khay phân loại: quan sát (`observed`), dẫn xuất (`derived`), suy luận (`inferred`), và chưa biết (`unknown`). Đường dẫn này mở phiên chạy khi tắt bảng ký hiệu, khiến tên hàm và điểm kết thúc hàm trở thành các yếu tố chưa biết, còn điểm bắt đầu được xếp vào khay suy luận.

## Sản Phẩm Đóng Gói (Lesson Artifact)

Bài học này tạo ra sản phẩm `outputs/skill-evidence-ledger.md`. Nó hướng dẫn một AI agent cách ghi nhận từng phát hiện dưới dạng một bản ghi có độ tin cậy, thẩm quyền, các giới hạn và các liên kết nguồn gốc. Nó hướng dẫn agent cách duy trì các yếu tố chưa biết dưới dạng các bản sửa đổi liên kết băm, và cách xác định khi nào một khẳng định đạt trạng thái `pass`, `fail`, `unknown`, hay `unsupported`.

## Xác Minh Tính Đúng Đắn (Verify It)

Chạy bản demo và bộ kiểm thử tất định:

```bash
cd phases/13-tools-and-protocols/35-evidence-confidence-and-unknowns
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Toàn bộ 28 bài kiểm tra chứng minh:

- Cả hai giá trị `evidence_id` của rea, mã băm cấu hình, và cả hai giá trị `query_id` của snapshot đều được tính toán lại chuẩn xác từ nội dung.
- Thay đổi đường dẫn tệp vẫn giữ nguyên mã định danh, nhưng thay đổi một câu giới hạn sẽ làm đổi mã định danh.
- Một bản ghi bị chỉnh sửa nội dung, sai lệch nhà cung cấp cấu hình, chứa số thực float, hoặc mang giá trị enum lạ đều bị từ chối.
- Bản ghi trùng lặp trả về `duplicate`, bản ghi xung đột ném ngoại lệ, và việc nhập gói dữ liệu mang tính nguyên tử (atomic).
- Gói dữ liệu chạy đã ghi nhận được nhập thành công với 18 bản ghi và không có liên kết treo nào.
- Một yếu tố chưa biết mở tại bản sửa đổi 1, từ chối một `expected_revision` lỗi thời, và chỉ được giải quyết bằng bằng chứng quan sát thực nghiệm có thẩm quyền `controlled-replay`.
- Chỉnh sửa một bản sửa đổi trong quá khứ sẽ làm gãy chuỗi băm ngay lập tức.
- Quy trình thăng cấp từ chối bằng chứng suy luận, dẫn xuất, rỗng hoặc không liên quan, và phê duyệt khi có lần chạy đồng thuận.
- Độ bao phủ một phần cho ra trạng thái `unknown` và một lần chạy bất đồng cho ra trạng thái `fail`.
- Bộ đệm khớp trúng trên khóa chính xác và trượt trên các tham số khác.
- Bộ đệm từ chối một phiên bản cấu hình bị nâng cấp, sai mục tiêu, sai phiên bản định dạng, và khóa bị giả mạo.
- Bộ đọc ELF đồng thuận với `llvm-objdump` về độ lệch, kích thước, byte và tái định vị, và bảng dữ liệu đọc ra dãy số `1, 2, 3, 4, 4`.
- Các dự đoán áp dụng phép chia C, và các lần chạy đã ghi nhận đồng thuận với toàn bộ 17 khẳng định đầu vào đơn lẻ.
- Khi có trình biên dịch C trên máy, các lần chạy trực tiếp cũng đồng thuận hoàn toàn.

## Các Chế Độ Thất Bại Trong Môi Trường Thực Tế

| Thất bại | Báo cáo yếu kém tuyên bố | Điều sổ cái bắt buộc phải có |
|---|---|---|
| Mã giả bị hiểu nhầm là hành vi chạy | "grade trả về 0 cho các điểm dưới trung bình" | Một bản ghi `execute_function` cho đầu vào đó, nếu không phải là `unsupported` |
| Chữ ký của trình dịch ngược được tin tưởng | "percent_change trả về 16 byte" | Chữ ký phải được đánh dấu là suy luận của công cụ với độ tin cậy `low` |
| Danh sách rỗng bị coi là sự vắng mặt | "hàm không có hàm nào gọi tới" | Trạng thái `unknown`, hoặc trường phạm vi bao phủ nếu công cụ có hỗ trợ |
| Đường dẫn tệp bị băm vào mã định danh | "bản ghi bị biến đổi" sau khi copy | Phép chiếu ngữ nghĩa loại bỏ `name` và `local_path` |
| Bỏ rơi câu giới hạn khi sao chép | Bản ghi trông có vẻ mạnh hơn nguồn gốc | Mã định danh không khớp và bị từ chối khi nhập |
| Lưới thử nghiệm một phần bị làm tròn | "toàn bộ điểm từ 90 trở lên đều đạt" | Trạng thái `unknown` với độ bao phủ `covered` 2/11 |
| Lần chạy host bị mạo danh là lần chạy của tệp | "tệp đối tượng x86-64 trả về 25" | Câu giới hạn về bản dựng host bắt buộc phải đi liền với khẳng định |
| Yếu tố chưa biết bị đóng bằng một câu văn | "đã giải quyết, xem phần trên" | Một giải pháp `verified` viện dẫn các mã bằng chứng quan sát đủ thẩm quyền |
| Bản sửa đổi cũ bị ghi đè mất dấu | Hai chỉnh sửa của chuyên viên va chạm nhau | Ném lỗi `revision-conflict` và bắt buộc đọc lại trạng thái mới nhất |
| Snapshot cũ bị phát lại sau đợt phát hành | Trả về câu trả lời từ một giao ước đã đổi | Ném lỗi `profile_mismatch` khi nhập snapshot |
| Giữ lại cache sau khi đổi tên hàm | Phát lại kết quả che giấu một lần ghi siêu dữ liệu | Vô hiệu hóa cache trước khi điều phối và hủy bỏ việc xuất dữ liệu lỗi |

## Quy Tắc Vận Hành Cốt Lõi

Hãy ghi nhận những gì bạn đọc được, những gì bạn tính toán được, và những gì bạn kỳ vọng thành ba loại bản ghi riêng biệt với ba từ định danh duy nhất. Hãy khép lại một yếu tố chưa biết bằng một mã ID bằng chứng, tuyệt đối không bao giờ bằng một câu văn trần thuật. Chỉ phát lại một câu trả lời dưới chính xác khóa định danh mà nó đã được lưu trữ, và kiên quyết từ chối phát lại khi chỉ có một chuỗi phiên bản bị thay đổi.

## Đọc Thêm

- [Mã nguồn rea 6.5.0, `src/domain/evidence.ts` và `src/domain/residualUnknown.ts`](https://github.com/morluto/rea)
- [RFC 8785, JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [Bản phát hành Ghidra 12.1.4](https://github.com/NationalSecurityAgency/ghidra/releases)
- [Agent Failure Patterns in CTF Reverse Engineering, NDSS BAR 2026](https://www.ndss-symposium.org/wp-content/uploads/bar2026-58.pdf)
- [When Binaries Talk Back: Representation-Confusion Attacks on LLM-Assisted Reverse Engineering, arXiv 2607.12507](https://arxiv.org/abs/2607.12507)
- [Nghiên cứu về dịch ngược bằng tác tử AI, arXiv 2606.06838](https://arxiv.org/abs/2606.06838)
- [When LLM Decompilers Recompile More and Preserve Less, arXiv 2609.05370](https://arxiv.org/abs/2609.05370)
- Mã nguồn rea 6.5.0 tại tag `3b79af2` mà bài học này mô tả: `src/domain/evidence.ts`, `.agents/skills/reverse-engineer-anything/SKILL.md`, `docs/cli.md`, `src/domain/reconstructionVerification.ts`, `src/application/ReconstructionObligationCandidates.ts`, `docs/tool-design.md`, `src/domain/evidenceCompletionLedger.ts`, `src/domain/residualUnknown.ts`, `src/application/investigation/UnknownEvidence.ts`, `.agents/skills/reverse-engineer-anything/references/evidence-workflows.md`, `src/application/binary/AnalysisSnapshotCache.ts`, `src/domain/analysisSnapshot.ts`, `src/application/binary/BinarySession.ts`, `src/application/binary/BinarySessionRecords.ts`, `docs/mcp-contracts.md`.
