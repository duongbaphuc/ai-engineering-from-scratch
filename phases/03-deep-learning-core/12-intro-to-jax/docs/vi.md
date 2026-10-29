# Giới thiệu về JAX (Introduction to JAX)

> PyTorch làm biến đổi (mutate) các tensor. TensorFlow xây dựng các đồ thị tĩnh. Còn JAX biên dịch các hàm thuần túy (pure functions). Lựa chọn sau cùng này sẽ thay đổi hoàn toàn cách bạn tư duy về deep learning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Bài học 01-10, kiến thức cơ bản về NumPy
**Time:** ~90 phút

## Learning Objectives

- Viết mã mạng nơ-ron dựa trên hàm thuần túy sử dụng API lập trình hàm của JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Giải thích sự khác biệt cốt lõi trong thiết kế giữa cơ chế đột biến trực tiếp (eager mutation) của PyTorch và mô hình biên dịch hàm của JAX
- Ứng dụng biên dịch JIT và vector hóa tự động vmap để tăng tốc vòng lặp huấn luyện so với mã Python thông thường
- Huấn luyện một mạng nơ-ron đơn giản trong JAX và đối chiếu cơ chế quản lý trạng thái tường minh với cách tiếp cận hướng đối tượng của PyTorch

## Vấn đề thực tế (The Problem)

Bạn đã biết cách xây dựng mạng nơ-ron trong PyTorch. Bạn định nghĩa một lớp kế thừa `nn.Module`, gọi `.backward()`, rồi bước bộ tối ưu `optimizer.step()`. Mọi thứ hoạt động hoàn hảo. Hàng triệu kỹ sư trên thế giới đang dùng nó.

Thế nhưng PyTorch có một ràng buộc ăn sâu vào bản chất: nó lần vết (trace) các phép toán một cách háo hức (eagerly), từng phép toán một trong môi trường Python. Mỗi phép tính `tensor + tensor` là một lần khởi chạy kernel GPU riêng biệt. Mỗi bước huấn luyện đều thông dịch lại cùng những dòng mã Python đó. Điều này hoạt động tốt cho đến khi bạn cần huấn luyện một mô hình 540 tỷ tham số trải rộng trên 2.048 chip TPU. Khi đó, chi phí thông dịch (overhead) sẽ bóp nghẹt hiệu năng của bạn.

Google DeepMind huấn luyện Gemini trên JAX. Anthropic huấn luyện Claude trên JAX. Đây không phải là các thử nghiệm nhỏ -- đây là những đợt huấn luyện mạng nơ-ron lớn nhất trên hành tinh. Họ chọn JAX vì nó đối xử với toàn bộ vòng lặp huấn luyện như một chương trình máy tính có thể biên dịch toàn vẹn, chứ không phải một chuỗi các lệnh gọi hàm Python rời rạc.

JAX chính là NumPy được trao cho ba siêu năng lực: vi phân tự động (automatic differentiation), biên dịch JIT ra XLA, và vector hóa tự động (automatic vectorization). Bạn chỉ cần viết một hàm xử lý cho duy nhất một mẫu dữ liệu. JAX sẽ cung cấp cho bạn một hàm xử lý cho cả batch, tự động tính đạo hàm, biên dịch ra mã máy tối ưu, và chạy song song trên nhiều thiết bị phần cứng. Tất cả mà không cần thay đổi dù chỉ một dòng mã trong hàm gốc.

## Khái niệm (The Concept)

### Triết lý của JAX

JAX là một framework mang phong cách lập trình hàm (functional framework). Không có lớp (class), không có trạng thái biến đổi được (mutable state), không có phương thức `.backward()`. Thay vào đó:

| PyTorch | JAX |
|---------|-----|
| Lớp `nn.Module` có chứa trạng thái nội tại | Hàm thuần túy: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Thực thi trực tiếp (Eager execution) | Biên dịch JIT thông qua XLA |
| Vòng lặp thủ công `for x in batch:` | Vector hóa tự động `jax.vmap(f)` |
| `DataParallel` / `FSDP` | Song song hóa tự động `jax.pmap(f)` |
| `model.parameters()` biến đổi trực tiếp | Cây mảng bất biến (Immutable pytree of arrays) |

Đây không phải là một sự ưa chuộng phong cách viết mã thông thường. Đây là một ràng buộc mang tính chất của trình biên dịch. Quá trình biên dịch JIT bắt buộc phải làm việc với các hàm thuần túy (pure functions) -- cùng một đầu vào thì luôn luôn trả về cùng một đầu ra, không gây ra bất kỳ tác dụng phụ (side effects) nào. Chính sự ràng buộc khắt khe này đã mở đường cho mức tăng tốc hiệu năng lên tới 100 lần.

### jax.numpy: Giao diện quen thuộc

JAX tái hiện lại hoàn chỉnh API của NumPy nhưng chạy trên các bộ tăng tốc phần cứng (GPU/TPU):

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Cùng tên hàm. Cùng quy tắc phát thanh (broadcasting). Cùng ngữ nghĩa cắt lát chỉ mục (slicing). Nhưng các mảng này nằm trực tiếp trên GPU/TPU, và mọi phép toán đều có thể được theo dõi và tối ưu bởi trình biên dịch.

Một điểm khác biệt chí mạng: các mảng trong JAX là **bất biến (immutable)**. Bạn không thể gán `a[0] = 5`. Thay vào đó, bạn phải viết: `a = a.at[0].set(5)`. Điều này có thể khiến bạn thấy ngượng nghịu trong vài ngày đầu, nhưng sau đó bạn sẽ nhận ra -- tính bất biến chính là nền tảng giúp các phép biến đổi như `grad`, `jit`, và `vmap` có thể kết hợp (compose) mượt mà với nhau.

### jax.grad: Vi phân tự động theo phong cách hàm

PyTorch gắn gradient vào chính các tensor (`.grad`). JAX gắn gradient vào các **hàm số**.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` nhận vào một hàm số và trả về một hàm số mới có nhiệm vụ tính đạo hàm của hàm ban đầu. Không có lệnh gọi `.backward()`. Không có đồ thị tính toán nào được lưu vào các tensor. Đạo hàm chỉ đơn giản là một hàm số bình thường khác mà bạn có thể gọi, kết hợp, hoặc biên dịch JIT.

Khả năng kết hợp này là không giới hạn:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Đạo hàm bậc hai. Đạo hàm bậc ba. Ma trận Jacobian. Ma trận Hessian. Tất cả đều thu được bằng cách lồng ghép `grad`. PyTorch cũng có thể làm điều này (`torch.autograd.functional.hessian`), nhưng nó chỉ là tính năng chắp vá thêm vào. Trong JAX, đó là nền tảng cốt lõi.

Ràng buộc: `grad` chỉ hoạt động trên các hàm thuần túy. Tuyệt đối không đặt câu lệnh `print` bên trong (chúng chỉ chạy một lần duy nhất lúc tạo vết tracing, chứ không chạy khi hàm thực thi thực tế). Không làm biến đổi trạng thái bên ngoài. Không sinh số ngẫu nhiên nếu không quản lý khóa (key) tường minh.

### jit: Biên dịch sang XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Trong lần gọi đầu tiên, JAX sẽ "lần vết" (trace) hàm số -- nó ghi nhận chuỗi các phép toán xảy ra mà không thực sự tính toán giá trị. Sau đó, nó chuyển vết ghi này cho XLA (Accelerated Linear Algebra), trình biên dịch chuyên dụng của Google dành cho TPU và GPU. XLA sẽ hợp nhất (fuse) các phép toán, loại bỏ các bước sao chép bộ nhớ thừa thãi, và sinh ra mã máy tối ưu hóa cao nhất.

Các lần gọi tiếp theo sẽ bỏ qua hoàn toàn tầng thông dịch Python. Mã máy đã biên dịch sẽ chạy trực tiếp trên bộ tăng tốc phần cứng với tốc độ tương đương C++.

Khi nào nên dùng JIT:
- Các bước huấn luyện (cùng một khối tính toán lặp đi lặp lại hàng nghìn lần)
- Suy luận (cùng mô hình, dữ liệu đầu vào khác nhau)
- Bất kỳ hàm nào được gọi nhiều lần với kích thước đầu vào giữ nguyên

Khi nào JIT gây phản tác dụng:
- Các hàm có luồng điều khiển Python phụ thuộc vào giá trị dữ liệu (`if x > 0` trong đó x là một mảng đang được trace)
- Các phép tính chạy một lần (thời gian biên dịch lâu hơn thời gian chạy)
- Quá trình gỡ lỗi (bước tracing che giấu luồng thực thi thực tế)

Ràng buộc về luồng điều khiển là có thật: `jax.lax.cond` thay thế cho `if/else`. `jax.lax.scan` thay thế cho vòng lặp `for`. Đây không phải là tùy chọn -- đây là cái giá phải trả để đổi lấy tốc độ biên dịch.

### vmap: Vector hóa tự động

Bạn chỉ cần viết một hàm xử lý cho một mẫu dữ liệu đơn lẻ:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` sẽ nâng cấp hàm đó để xử lý cả một batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` mang ý nghĩa: không chia batch cho `params` (dùng chung cho mọi mẫu), chia batch theo chiều 0 của `x`. Không cần viết vòng lặp `for`. Không cần reshape mảng. Không cần đau đầu quản lý trục batch. JAX tự động nhận diện chiều dữ liệu và vector hóa toàn bộ khối tính toán.

Đây không phải là cú pháp đường (syntactic sugar). `vmap` tạo ra mã vector hóa hợp nhất chạy nhanh gấp 10-100 lần so với vòng lặp Python. Và nó kết hợp hoàn hảo với `jit` và `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Tính gradient cho từng mẫu riêng biệt (per-example gradients). Chỉ cần duy nhất một dòng mã. Điều này gần như là bất khả thi trong PyTorch nếu không dùng các mẹo phức tạp.

### pmap: Song song hóa dữ liệu trên nhiều thiết bị

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` nhân bản hàm số trên toàn bộ các thiết bị khả dụng (GPU/TPU) và tự động chia nhỏ batch dữ liệu. Bên trong hàm, `jax.lax.pmean` và `jax.lax.psum` sẽ đồng bộ hóa gradient giữa các thiết bị.

Google huấn luyện Gemini trên hàng nghìn chip TPU v5e bằng cách sử dụng `pmap` (và phiên bản kế nhiệm `shard_map`). Mô hình lập trình: viết phiên bản cho thiết bị đơn, bọc ngoài bằng `pmap`, và mọi thứ đã sẵn sàng.

### Pytrees: Cấu trúc dữ liệu phổ quát

JAX thao tác trên các "pytree" -- các cấu trúc lồng nhau tùy ý của list, tuple, dict, và mảng array. Toàn bộ tham số mô hình của bạn chính là một pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Mọi phép biến đổi trong JAX -- `grad`, `jit`, `vmap` -- đều hiểu cách duyệt qua các pytree. `jax.tree.map(f, tree)` áp dụng hàm `f` lên từng nút lá (leaf). Đây là cách mà các bộ tối ưu cập nhật toàn bộ tham số cùng lúc:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Không cần phương thức `.parameters()`. Không cần cơ chế đăng ký tham số phức tạp. Bản thân cấu trúc cây chính là mô hình.

### Phong cách hàm so với Hướng đối tượng

PyTorch lưu trữ trạng thái bên trong các đối tượng:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX sử dụng các hàm thuần túy với trạng thái được truyền vào tường minh:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Các tham số `params` được truyền vào từ bên ngoài. Không có gì được lưu trữ ngầm. Không có trạng thái nào bị đột biến. Điều này làm cho mọi hàm số đều dễ kiểm thử, dễ kết hợp và dễ biên dịch. Nó cũng đồng nghĩa với việc bạn phải tự quản lý tham số -- hoặc nhờ tới sự trợ giúp của các thư viện như Flax hoặc Equinox.

### Hệ sinh thái JAX

JAX cung cấp các khối nguyên thủy. Các thư viện cung cấp trải nghiệm lập trình tiện lợi:

| Thư viện | Vai trò | Phong cách |
|----------|---------|------------|
| **Flax** (Google) | Các lớp mạng nơ-ron | `nn.Module` với trạng thái tường minh |
| **Equinox** (Patrick Kidger) | Các lớp mạng nơ-ron | Dựa trên pytree, rất tự nhiên theo kiểu Python |
| **Optax** (DeepMind) | Bộ tối ưu hóa & lịch biểu LR | Các phép biến đổi gradient có thể ghép chuỗi |
| **Orbax** (Google) | Checkpointing (lưu điểm kiểm tra) | Lưu/phục hồi các pytree |
| **CLU** (Google) | Đo lường metrics & ghi log | Tiện ích cho vòng lặp huấn luyện |

Optax là thư viện bộ tối ưu tiêu chuẩn. Nó tách biệt phép biến đổi gradient (Adam, SGD, cắt gradient) khỏi bước cập nhật tham số, giúp việc kết hợp trở nên cực kỳ linh hoạt:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Khi nào nên dùng JAX so với PyTorch

| Yếu tố | JAX | PyTorch |
|--------|-----|---------|
| Hỗ trợ TPU | Hạng nhất (Google xây dựng cả hai) | Cộng đồng duy trì (torch_xla) |
| Hỗ trợ GPU | Tốt (CUDA thông qua XLA) | Đỉnh cao nhất (CUDA gốc) |
| Gỡ lỗi | Khó (do lần vết tracing + biên dịch) | Dễ dàng (eager, từng dòng lệnh) |
| Hệ sinh thái | Tập trung nghiên cứu (Flax, Equinox) | Khổng lồ (Hugging Face, torchvision, v.v.) |
| Thị trường việc làm | Ngách (Google/DeepMind/Anthropic) | Phổ biến rộng rãi khắp nơi |
| Huấn luyện quy mô cực lớn | Vượt trội (XLA, pmap, mesh) | Tốt (FSDP, DeepSpeed) |
| Tốc độ tạo mẫu | Chậm hơn (chi phí tư duy hàm) | Nhanh hơn (đột biến trạng thái và chạy) |
| Suy luận production | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Ai đang sử dụng | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Câu trả lời thực tế: hãy dùng PyTorch trừ khi bạn có lý do cụ thể để chọn JAX. Các lý do đó là: sở hữu tài nguyên TPU, cần tính gradient cho từng mẫu riêng biệt, huấn luyện đa thiết bị ở quy mô siêu khổng lồ, hoặc làm việc tại Google/DeepMind/Anthropic.

### Số ngẫu nhiên trong JAX

JAX không có trạng thái ngẫu nhiên toàn cục (global random state). Mọi phép toán ngẫu nhiên đều đòi hỏi một PRNG key tường minh:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Ban đầu điều này có thể hơi phiền toái. Nhưng nó đảm bảo tính tái lập (reproducibility) hoàn hảo trên mọi thiết bị và qua mọi lần biên dịch -- một đặc tính mà lệnh `torch.manual_seed` của PyTorch không thể bảo đảm trong môi trường đa GPU.

```figure
batchnorm-effect
```

## Tự tay xây dựng (Build It)

### Bước 1: Thiết lập và Dữ liệu

Chúng ta sẽ huấn luyện một mạng MLP 3 lớp trên tập MNIST sử dụng JAX và Optax: 784 đầu vào, hai lớp ẩn 256 và 128 neuron, 10 lớp đầu ra.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Bước 2: Khởi tạo tham số

Không dùng class. Chỉ là một hàm trả về một pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Khởi tạo He được cài đặt thủ công. Ba PRNG key được tách ra từ một seed gốc. Mỗi trọng số là một mảng bất biến nằm trong một từ điển lồng nhau.

### Bước 3: Lượt lan truyền xuôi (Forward Pass)

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Các hàm thuần túy. Nhận params vào, trả về dự đoán ra. Không có `self`, không lưu trạng thái. Hàm `loss_fn` tính cross-entropy từ đầu -- softmax, log, và lấy trung bình phủ định.

### Bước 4: Bước huấn luyện được biên dịch JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` trả về cả giá trị mất mát lẫn các gradient chỉ trong một lượt duyệt duy nhất. Decorator `@jax.jit` biên dịch cả hai hàm sang XLA. Sau lần gọi đầu tiên, từng bước huấn luyện chạy mà không hề chạm tới trình thông dịch Python.

### Bước 5: Vòng lặp huấn luyện

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epoch. Độ chính xác kiểm thử ~97%. Epoch đầu tiên sẽ chậm hơn (do quá trình biên dịch JIT). Các epoch từ 2 đến 10 chạy cực kỳ nhanh.

Hãy chú ý những gì đã biến mất: không có `.zero_grad()`, không có `.backward()`, không có `.step()`. Toàn bộ bước cập nhật tham số nằm gọn trong một lời gọi hàm kết hợp duy nhất. Gradient được tính, được Adam biến đổi, và được áp dụng vào tham số -- tất cả diễn ra ngay bên trong `train_step`.

## Sử dụng trong thực tế (Use It)

### Flax: Tiêu chuẩn của Google

Flax là thư viện mạng nơ-ron phổ biến nhất cho JAX. Nó mang `nn.Module` quay trở lại, nhưng với cơ chế quản lý trạng thái tường minh:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Cấu trúc tương tự PyTorch, nhưng `params` tách biệt hoàn toàn khỏi đối tượng mô hình. `model.init()` tạo ra params. `model.apply(params, x)` thực hiện lượt tính xuôi. Bản thân đối tượng model không hề chứa trạng thái nào.

### Equinox: Sự thay thế tự nhiên theo phong cách Python

Equinox (do Patrick Kidger phát triển) biểu diễn các mô hình trực tiếp dưới dạng các pytree:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Bản thân mô hình chính là một pytree. Không cần gọi `.apply()`. Các tham số chính là các nút lá của mô hình. Cách tiếp cận này gần gũi nhất với tư duy thiết kế cốt lõi của JAX.

### Optax: Các bộ tối ưu hóa có thể ghép chuỗi

Optax tách rời phép biến đổi gradient khỏi thao tác cập nhật:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Cắt gradient (clipping), khởi động tốc độ học (warmup), suy giảm trọng số (weight decay) -- tất cả đều được kết hợp thành một chuỗi các phép biến đổi. Mỗi phép biến đổi nhận gradient, sửa đổi chúng, và truyền tiếp cho khâu sau. Không có một lớp bộ tối ưu nguyên khối cồng kềnh nào ở đây.

## Sản phẩm bàn giao (Ship It)

**Cài đặt:**

```bash
pip install jax jaxlib optax flax
```

Hỗ trợ GPU:

```bash
pip install jax[cuda12]
```

Dành cho TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Các bẫy hiệu năng cần tránh:**

- Lời gọi JIT đầu tiên luôn chậm (do khâu biên dịch). Cần khởi động làm nóng (warm up) trước khi đo điểm chuẩn benchmark.
- Tránh dùng vòng lặp Python trên mảng JAX bên trong hàm JIT. Hãy dùng `jax.lax.scan` hoặc `jax.lax.fori_loop`.
- `jax.debug.print()` hoạt động được bên trong hàm JIT. Hàm `print()` thông thường của Python thì không.
- Đo kiểm hiệu năng với `jax.profiler` hoặc TensorBoard. Quá trình biên dịch XLA có thể che giấu các điểm nghẽn cổ chai.
- JAX mặc định cấp phát trước 75% bộ nhớ GPU. Thiết lập `XLA_PYTHON_CLIENT_PREALLOCATE=false` để tắt hành vi này.

**Lưu checkpoint:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Bài học này tạo ra:**
- `outputs/prompt-jax-optimizer.md` -- prompt lựa chọn cấu hình bộ tối ưu JAX phù hợp
- `outputs/skill-jax-patterns.md` -- tài liệu kỹ năng bao quát các mẫu thiết kế lập trình hàm trong JAX

## Bài tập rèn luyện (Exercises)

1. Thêm dropout vào mạng MLP. Trong JAX, dropout yêu cầu một PRNG key -- hãy truyền key xuyên suốt lượt tính xuôi và tách (split) nó cho từng lớp dropout. So sánh độ chính xác kiểm thử khi có và không có dropout.
2. Sử dụng `jax.vmap` để tính gradient cho từng mẫu riêng biệt (per-example gradients) trên một batch 32 ảnh MNIST. Tính chuẩn của gradient (gradient norm) cho mỗi mẫu. Mẫu nào có gradient lớn nhất, và tại sao?
3. Thay thế hàm forward thủ công bằng một hàm tổng quát `mlp_forward(params, x)` có thể hoạt động với số lượng lớp bất kỳ. Sử dụng `jax.tree.leaves` để tự động xác định độ sâu của mạng.
4. Đo kiểm hiệu năng bước huấn luyện khi có và không có decorator `@jax.jit`. Đo thời gian chạy 100 bước cho mỗi trường hợp. Tốc độ tăng lên bao nhiêu lần trên phần cứng của bạn? Thời gian biên dịch ở lần gọi đầu tiên chiếm bao lâu?
5. Cài đặt cắt gradient bằng cách ghép chuỗi `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Huấn luyện khi có và không có cắt gradient. Vẽ biểu đồ chuẩn gradient trong suốt quá trình huấn luyện để quan sát hiệu quả.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Người ta thường nói | Bản chất thực sự |
|-----------|--------------------|-------------------|
| XLA | "Thứ giúp JAX chạy nhanh" | Accelerated Linear Algebra -- trình biên dịch hợp nhất các phép toán và sinh ra các kernel GPU/TPU tối ưu từ đồ thị tính toán |
| JIT | "Biên dịch tức thời" | JAX lần vết (trace) hàm số ở lần gọi đầu, biên dịch sang XLA, rồi chạy phiên bản đã biên dịch ở các lần gọi sau |
| Pure function (Hàm thuần túy) | "Không có tác dụng phụ" | Hàm mà đầu ra chỉ phụ thuộc duy nhất vào đầu vào -- không có trạng thái toàn cục, không làm đột biến dữ liệu, không có tính ngẫu nhiên nếu không truyền key tường minh |
| vmap | "Tự động chia batch" | Biến đổi một hàm xử lý một mẫu đơn lẻ thành một hàm xử lý cả một batch dữ liệu mà không cần viết lại mã |
| pmap | "Tự động chạy song song" | Nhân bản một hàm trên nhiều thiết bị phần cứng và tự động phân chia batch đầu vào |
| Pytree | "Từ điển mảng lồng nhau" | Bất kỳ cấu trúc lồng nhau nào của list, tuple, dict, và array mà JAX có thể duyệt qua và thực hiện các phép biến đổi |
| Tracing (Lần vết) | "Ghi nhận khối tính toán" | JAX thực thi hàm số với các giá trị trừu tượng để dựng đồ thị tính toán, mà không tính ra kết quả số học thực sự |
| Functional autodiff | "Đạo hàm của một hàm" | Tính toán đạo hàm bằng cách biến đổi hàm số, chứ không phải bằng cách lưu trữ gradient vào các tensor |
| Optax | "Thư viện tối ưu của JAX" | Thư viện tập hợp các phép biến đổi gradient có thể ghép chuỗi -- Adam, SGD, clipping, scheduling |
| Flax | "nn.Module của JAX" | Thư viện mạng nơ-ron của Google dành cho JAX, cung cấp các trừu tượng hóa lớp mạng trong khi vẫn duy trì trạng thái tường minh |

## Tài liệu đọc thêm (Further Reading)

- JAX documentation: https://jax.readthedocs.io/ -- tài liệu chính thức với các hướng dẫn xuất sắc về grad, jit, và vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- bài báo gốc giải thích triết lý thiết kế
- Flax documentation: https://flax.readthedocs.io/ -- thư viện mạng nơ-ron của Google dành cho JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- giải pháp thay thế tự nhiên theo phong cách Python cho Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- thư viện bộ tối ưu tiêu chuẩn
- "You Don't Know JAX" (Colin Raffel, 2020) -- hướng dẫn thực tế về các bẫy và mẫu thiết kế trong JAX từ một trong các tác giả của T5

