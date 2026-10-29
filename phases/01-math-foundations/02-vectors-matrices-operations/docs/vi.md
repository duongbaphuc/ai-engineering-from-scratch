# Vector, Ma Trận & Các Phép Toán (Vectors, Matrices & Operations)

> Mọi mạng nơ-ron thực chất chỉ là phép nhân ma trận được thêm thắt vài bước.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 phút

## Mục Tiêu Học Tập

- Xây dựng lớp `Matrix` hoàn chỉnh từ đầu với các phép toán theo từng phần tử (element-wise), nhân ma trận, chuyển vị, định thức và ma trận nghịch đảo
- Phân biệt rõ ràng phép nhân theo từng phần tử và phép nhân ma trận, giải thích trường hợp áp dụng của từng phép toán
- Cài đặt một tầng nơ-ron kết nối đầy đủ (dense layer: `relu(W @ x + b)`) chỉ sử dụng lớp `Matrix` tự viết từ đầu
- Giải thích các quy tắc lan truyền kích thước (broadcasting) và cách phép cộng bias hoạt động trong các framework học sâu

## Vấn Đề

Bạn muốn xây dựng một mạng nơ-ron. Khi đọc mã nguồn, bạn bắt gặp dòng này:

```
output = activation(weights @ input + bias)
```

Toán tử `@` đó là phép nhân ma trận. `weights` là một ma trận. `input` là một vector. Nếu bạn không hiểu các phép toán đó làm gì, dòng mã trên trông như một trò ảo thuật. Nhưng nếu bạn hiểu, đó chính là toàn bộ lượt lan truyền xuôi (forward pass) của một tầng nơ-ron được gói gọn trong 3 phép toán.

Mọi hình ảnh mà mô hình của bạn xử lý đều là ma trận giá trị điểm ảnh (pixels). Mọi embedding từ ngữ là một vector. Mọi tầng trong mạng nơ-ron đều là một phép biến đổi ma trận. Bạn không thể xây dựng các hệ thống AI nếu không thành thạo các thao tác ma trận, giống như bạn không thể viết code nếu không hiểu khái niệm biến số.

Bài học này sẽ giúp bạn làm chủ sự thông thạo đó từ con số không.

## Khái Niệm

### Vector: Danh sách các số có thứ tự

Một vector là một danh sách các con số có độ lớn (độ dài) và hướng. Trong AI, vector biểu diễn các điểm dữ liệu, các đặc trưng (features) hoặc các tham số.

```
v = [3, 4]        -- vector 2D
w = [1, 0, -2]    -- vector 3D
```

Vector 2D `[3, 4]` chỉ tới tọa độ (3, 4) trên mặt phẳng. Độ dài (magnitude) của nó là 5 (theo định lý Pythagoras cho tam giác 3-4-5).

### Ma Trận: Lưới các con số

Ma trận là một bảng/lưới 2 chiều gồm các hàng và cột. Một ma trận $m \times n$ có $m$ hàng và $n$ cột.

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{ma trận 2x3 (2 hàng, 3 cột)}
```

Trong mạng nơ-ron, ma trận trọng số biến đổi vector đầu vào thành vector đầu ra. Một tầng có 784 đầu vào và 128 đầu ra sẽ sử dụng ma trận trọng số kích thước 128x784.

### Vì sao kích thước (shape) lại quan trọng

Phép nhân ma trận tuân thủ một quy tắc nghiêm ngặt: `(m x n) @ (n x p) = (m x p)`. Số chiều bên trong (inner dimensions) bắt buộc phải khớp nhau.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Kích thước bên trong: 784 = 784  -- hợp lệ
```

Nếu bạn gặp lỗi không khớp kích thước (shape mismatch error) trong PyTorch, đây chính là nguyên nhân.

### Bảng tóm tắt các phép toán

| Phép toán | Tác dụng | Ứng dụng trong mạng nơ-ron |
|-----------|----------|----------------------------|
| Cộng | Kết hợp theo từng phần tử | Cộng bias vào đầu ra tầng |
| Nhân vô hướng | Nhân tỉ lệ từng phần tử | Tốc độ học (learning rate) * gradients |
| Nhân ma trận | Biến đổi vector | Lan truyền xuôi của tầng (forward pass) |
| Chuyển vị (Transpose) | Đổi hàng thành cột và ngược lại | Lan truyền ngược (backpropagation) |
| Định thức (Determinant) | Giá trị số tóm lược ma trận vuông | Kiểm tra tính khả nghịch |
| Nghịch đảo (Inverse) | Hoàn tác một phép biến đổi | Giải các hệ phương trình tuyến tính |
| Đơn vị (Identity) | Ma trận giữ nguyên giá trị | Khởi tạo, kết nối tắt (residual connections) |

### Nhân theo từng phần tử vs Nhân ma trận

Sự nhầm lẫn này xảy ra liên tục đối với người mới bắt đầu.

Nhân theo từng phần tử (Element-wise): nhân các vị trí tương ứng với nhau. Hai ma trận bắt buộc phải có cùng kích thước.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

Nhân ma trận (Matrix multiplication): tính tích vô hướng của từng hàng với từng cột. Chiều bên trong bắt buộc phải bằng nhau.

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

Phép toán khác nhau, kết quả khác nhau, quy tắc kiểm tra kích thước hoàn toàn khác nhau.

### Kỹ thuật lan truyền kích thước (Broadcasting)

Khi bạn cộng một vector bias vào một ma trận đầu ra, kích thước của chúng thường không khớp nhau. Cơ chế broadcasting sẽ tự động "kéo giãn" mảng nhỏ hơn để khớp với mảng lớn hơn.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting kéo giãn vector dọc theo các hàng:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mọi framework hiện đại đều tự động thực hiện việc này. Hiểu rõ nó giúp bạn không bị bối rối khi thấy kích thước có vẻ lệch nhau nhưng mã lệnh vẫn chạy bình thường.

```figure
vector-projection
```

## Tự Xây Dựng (Build It)

### Bước 1: Lớp Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Bước 2: Lớp Matrix với các phép toán cốt lõi

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Bước 3: Xem các phép toán hoạt động

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Bước 4: Kết nối với mạng nơ-ron

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Đây chính là một tầng dense layer hoàn chỉnh: `output = relu(W @ x + b)`. Mọi tầng kết nối đầy đủ trong bất kỳ mạng nơ-ron nào cũng thực hiện chính xác điều này.

## Sử Dụng Thư Viện (Use It)

NumPy thực hiện tất cả những điều trên với ít dòng code hơn và nhanh hơn nhiều bậc độ lớn:

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (nhân ma trận) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nTầng mạng nơ-ron: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Đầu ra:\n{output}")
```

Toán tử `@` trong Python sẽ gọi hàm đặc biệt `__matmul__`. NumPy cài đặt nó với các chương trình con BLAS tối ưu viết bằng C và Fortran. Cùng một công thức toán học, nhưng nhanh hơn 100 lần.

Broadcasting trong NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động broadcast vector bias 1 chiều qua cả 2 hàng. Đây chính xác là cách phép cộng bias hoạt động trong mọi thư viện deep learning hiện nay.

## Xuất Bản Sản Phẩm (Ship It)

Bài học này cung cấp:
- Prompt hướng dẫn dạy các phép toán ma trận thông qua trực giác hình học tại `outputs/prompt-matrix-operations.md`.
- Lớp `Matrix` xây dựng ở đây là nền tảng trực tiếp cho mini neural network framework mà chúng ta sẽ xây dựng ở Phase 3, Lesson 10.

## Bài Tập Thực Hành

1. **Kiểm chứng nghịch đảo:** Nhân `A @ A.inverse_2x2()` và xác nhận rằng bạn nhận được ma trận đơn vị. Thử nghiệm với 3 ma trận 2x2 khác nhau. Điều gì xảy ra khi định thức bằng 0?
2. **Cài đặt nghịch đảo 3x3:** Mở rộng lớp `Matrix` để tính ma trận nghịch đảo cho ma trận 3x3 bằng phương pháp ma trận phụ hợp (adjugate matrix). Kiểm thử lại với hàm `np.linalg.inv` của NumPy.
3. **Xây dựng mạng hai tầng:** Chỉ sử dụng lớp `Matrix` tự viết (không dùng NumPy), tạo một mạng nơ-ron 2 tầng: input (3 chiều) -> hidden (4 chiều) -> output (2 chiều). Khởi tạo trọng số ngẫu nhiên, chạy một lượt forward pass và xác minh tất cả các kích thước shape đều chính xác.

### Gợi Ý

**Bài 1.** Nhân `A @ A.inverse_2x2()` cho 3 ma trận và so sánh từng phần tử với ma trận đơn vị trong khoảng dung sai `1e-9`. Đối với ma trận có định thức bằng 0, chẳng hạn như [[1, 2], [2, 4]], hàm `inverse_2x2` sẽ ném ra lỗi `ValueError`, và đó là hành vi chính xác mong muốn.

**Bài 2.** Tính ma trận phần bù đại số 3x3 (cofactor matrix), chuyển vị nó để thu được ma trận phụ hợp (adjugate), sau đó chia từng phần tử cho `determinant()`. Kiểm tra lại `A @ A_inv` có ra ma trận đơn vị và khớp với `np.linalg.inv(A)` trong ngưỡng sai số `1e-9` hay không.

**Bài 3.** Viết kích thước của từng ma trận ra giấy trước khi nhân: `W1` là 4x3 và `W2` là 2x4, tương tự trong hàm `demo_neural_network_layer`. Đầu vào kích thước 3x1 phải tạo ra vector ẩn 4x1 và vector đầu ra 2x1.

## Thuật Ngữ Then Chốt

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|-----------|--------------------|-------------------|
| Vector | "Một mũi tên" | Danh sách các số có thứ tự. Trong AI: một điểm trong không gian nhiều chiều. |
| Ma trận | "Một bảng số" | Phép biến đổi tuyến tính. Ánh xạ các vector từ không gian này sang không gian khác. |
| Nhân ma trận | "Cứ nhân các số lại" | Tích vô hướng giữa từng hàng của ma trận thứ nhất với từng cột của ma trận thứ hai. Thứ tự nhân là bắt buộc. |
| Chuyển vị (Transpose) | "Lật ngược nó" | Đổi hàng thành cột và ngược lại. Biến ma trận $m \times n$ thành $n \times m$. Tối quan trọng trong lan truyền ngược. |
| Định thức (Determinant) | "Một con số từ ma trận" | Đo lường mức độ co giãn diện tích (2D) hoặc thể tích (3D). Bằng 0 nghĩa là phép biến đổi đã nén bẹp một chiều không gian. |
| Nghịch đảo (Inverse) | "Hoàn tác ma trận" | Ma trận đảo ngược lại phép biến đổi ban đầu. Chỉ tồn tại khi định thức khác 0. |
| Ma trận đơn vị (Identity) | "Ma trận không đổi" | Tương đương số 1 trong đại số thông thường. Dùng nhiều trong các liên kết tắt (ResNets). |
| Broadcasting | "Sửa kích thước ma thuật" | Kéo giãn mảng nhỏ hơn để khớp với mảng lớn hơn bằng cách lặp lại dọc theo chiều bị thiếu. |
| Element-wise | "Phép nhân thông thường" | Nhân các vị trí tương ứng với nhau. Cả hai mảng phải cùng kích thước hoặc có thể broadcast được. |

## Tài Liệu Đọc Thêm

- [3Blue1Brown: Bản chất của Đại số Tuyến tính](https://www.3blue1brown.com/topics/linear-algebra) - trực giác hình học trực quan xuất sắc cho mọi phép toán
- [NumPy documentation về broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - các quy tắc chính xác mà NumPy tuân theo
- [Stanford CS229 Ôn tập Đại số Tuyến tính](http://cs229.stanford.edu/section/cs229-linalg.pdf) - tài liệu cô đọng dành riêng cho Machine Learning
