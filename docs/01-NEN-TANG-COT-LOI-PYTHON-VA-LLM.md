# 01. Nền Tảng Cốt Lõi: Python Hiện Đại & Giải Phẫu Mô Hình Ngôn Ngữ Lớn (LLM)

> **Mục tiêu tài liệu:** Trang bị cho bạn 2 nền móng quan trọng nhất trước khi bắt tay vào làm dự án:
> 1. **Kỹ nghệ phần mềm với Python hiện đại** (thứ phân biệt một AI Engineer chuyên nghiệp với một người chỉ biết chạy code trên Notebook).
> 2. **Nguyên lý hoạt động bên trong của LLM** (từ lúc bạn gõ một câu tiếng Việt cho đến khi mô hình sinh ra từng chữ trả lời).

---

## PHẦN A: 4 Trụ Cột Python Bắt Buộc Cho AI Engineer

Trong các hệ thống AI thực tế, LLM chỉ chiếm khoảng 20% lượng code; 80% còn lại là code điều phối dữ liệu, gọi mạng, kiểm tra định dạng và xử lý lỗi. Bạn cần nắm vững 4 khái niệm sau:

### 1. Type Hints (Gợi ý kiểu dữ liệu)
Python là ngôn ngữ định kiểu động (dynamic typing), rất dễ gây lỗi ngầm khi dự án lớn lên. Trong dự án AI chuyên nghiệp, **mọi hàm đều phải khai báo rõ kiểu đầu vào và đầu ra**.

```python
# ❌ Cách viết sơ sài (Không biết hàm nhận vào gì, trả về gì)
def calculate_pe_ratio(price, eps):
    return price / eps

# ✅ Cách viết chuẩn AI Engineer
def calculate_pe_ratio(stock_price: float, earnings_per_share: float) -> float:
    if earnings_per_share <= 0:
        raise ValueError("EPS phải lớn hơn 0 để tính chỉ số P/E.")
    return round(stock_price / earnings_per_share, 2)
```

### 2. Pydantic (Kiểm soát dữ liệu & Hợp đồng giao tiếp với LLM)
LLM bản chất là sinh ra văn bản tự do (free text). Nhưng hệ thống phần mềm lại cần dữ liệu có cấu trúc chặt chẽ (JSON có đúng tên trường, đúng kiểu số/chuỗi). **Pydantic** là thư viện quan trọng số 1 trong AI Engineering vì nó giúp:
*   Tự động kiểm tra (validate) dữ liệu đầu vào/đầu ra.
*   Ép LLM phải trả về đúng cấu trúc JSON mà ta định nghĩa (**Structured Outputs**).

```python
from pydantic import BaseModel, Field

class FinancialMetricExtraction(BaseModel):
    ticker: str = Field(..., description="Mã chứng khoán viết hoa, ví dụ: FPT, VCB")
    year: int = Field(..., ge=2000, le=2030, description="Năm báo cáo tài chính")
    revenue_billion_vnd: float = Field(..., gt=0, description="Doanh thu thuần (đơn vị: tỷ VNĐ)")
    summary: str = Field(..., description="Tóm tắt ngắn gọn nguyên nhân tăng trưởng")
```
*Nếu LLM trả về `year: "hai nghìn hai tư"` hoặc thiếu trường `ticker`, Pydantic sẽ lập tức bắt lỗi ngay tại cửa ngõ để ta yêu cầu LLM sửa lại, không để dữ liệu rác làm sập hệ thống.*

### 3. Lập trình Bất đồng bộ (`AsyncIO` - `async / await`)
**Tại sao AI Engineer bắt buộc phải giỏi Async?**
*   Khi bạn gọi một mô hình LLM hoặc tìm kiếm trên Vector DB, CPU của máy bạn **không hề phải tính toán nặng**, mà nó chỉ **ngồi chờ phản hồi qua mạng (I/O Bound)** mất từ 1 đến 10 giây.
*   Nếu viết code đồng bộ thông thường (`def`), khi có 10 người dùng cùng hỏi, người thứ 10 phải đợi 9 người trước hỏi xong (mất 50 giây!).
*   Nếu dùng bất đồng bộ (`async def` và `await`), trong lúc chờ LLM trả lời cho người thứ 1, chương trình lập tức xử lý yêu cầu của người thứ 2, 3... hoặc vừa tìm kiếm Qdrant vừa truy vấn PostgreSQL **cùng một lúc** (`asyncio.gather`).

```python
import asyncio

async def search_vector_db(query: str) -> list[str]:
    await asyncio.sleep(0.5)  # Giả lập thời gian chờ Qdrant
    return ["Đoạn văn báo cáo tài chính FPT..."]

async def query_sql_db(ticker: str) -> dict[str, float]:
    await asyncio.sleep(0.5)  # Giả lập thời gian chờ PostgreSQL
    return {"pe": 22.5, "roe": 28.1}

async def fetch_all_context(query: str, ticker: str) -> tuple[list[str], dict[str, float]]:
    # Chạy song song cả 2 tác vụ: Tổng thời gian chỉ mất ~0.5s thay vì 1.0s!
    docs, sql_data = await asyncio.gather(
        search_vector_db(query),
        query_sql_db(ticker)
    )
    return docs, sql_data
```

### 4. FastAPI (Xây dựng API cho hệ thống AI)
FastAPI là web framework chuẩn mực cho AI hiện nay vì nó tích hợp sẵn **Pydantic** và **AsyncIO**, tự động sinh tài liệu API (Swagger UI) và hỗ trợ **Streaming** (trả về từng chữ ngay khi LLM vừa sinh ra).

---

## PHẦN B: Giải Phẫu Mô Hình Ngôn Ngữ Lớn (LLM) Dưới Góc Nhìn Kỹ Sư

Để tối ưu hóa hệ thống và trả lời các câu hỏi phỏng vấn chuyên sâu tại VinAI, bạn cần hiểu rõ những gì diễn ra từ lúc một câu văn đi vào LLM cho đến khi chữ tiếp theo được sinh ra.

### 1. Tokenization (Tách từ thành con số)
Máy tính không hiểu chữ `"Doanh thu FPT tăng trưởng"`. Bước đầu tiên, bộ **Tokenizer** (thường dùng thuật toán **BPE - Byte-Pair Encoding**) sẽ cắt câu văn thành các mảnh nhỏ gọi là **Token** và ánh xạ mỗi token thành một mã số nguyên (`Token ID`).

*   Một token có thể là cả một từ, một phần của từ, hoặc một dấu câu.
*   **Lưu ý quan trọng với Tiếng Việt:** Các mô hình cũ (như GPT-2 hay Llama-2) có bộ từ vựng (Vocabulary) rất ít tiếng Việt, nên 1 từ tiếng Việt có dấu có thể bị cắt vụn thành 3–4 tokens (làm tốn tiền API, nhanh đầy bộ nhớ và chạy chậm). Các mô hình hiện đại như **Qwen-2.5**, **Llama-3.1**, **GPT-4o** có bộ từ vựng > 100.000–150.000 tokens, nén tiếng Việt cực tốt.
*   **Quy tắc ước lượng nhanh:** `1.000 tokens` tương đương khoảng `650 - 750 từ tiếng Việt`.

### 2. Bản chất Toán học của LLM: Mô hình Tự hồi quy (Autoregressive Model)
Đừng nghĩ LLM là một "bộ não biết suy nghĩ". Bản chất toán học của mọi LLM hiện nay là **dự đoán phân phối xác suất của Token tiếp theo dựa trên toàn bộ các Token đứng trước nó**:

$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

Ví dụ: Khi đưa vào chuỗi token `"Thủ đô của Việt Nam là"`, mô hình sẽ tính toán một bảng điểm (**Logits**) cho toàn bộ 150.000 token trong từ điển, sau đó qua hàm **Softmax** để đổi thành xác suất %:
*   `"Hà"` -> `96.5%`
*   `"Thành"` -> `2.1%`
*   `"Sài"` -> `0.8%`
*   ...

Sau khi chọn được chữ `"Hà"`, nó nối vào chuỗi cũ thành `"Thủ đô của Việt Nam là Hà"` và lại chạy tiếp vòng lặp để đoán chữ `"Nội"`. Quá trình này lặp lại cho đến khi mô hình sinh ra một token đặc biệt báo hiệu kết thúc (`<|endoftext|>` hoặc `<|im_end|>`).

### 3. Cơ chế Self-Attention và Context Window (Cửa sổ ngữ cảnh)
Trái tim của kiến trúc **Transformer** (kiến trúc tạo nên mọi LLM) là cơ chế **Self-Attention**:
*   Khi đọc một câu như: *"Công ty FPT vừa mua lại một công ty AI tại Nhật Bản vì **họ** muốn mở rộng thị trường"*, cơ chế Self-Attention giúp từ **"họ"** tính toán độ liên kết (trọng số chú ý) với các từ đứng trước để hiểu **"họ"** ở đây chính là **"Công ty FPT"** chứ không phải Nhật Bản.
*   Mỗi token sẽ sinh ra 3 vector:
    *   **Query ($Q$):** Thứ tôi đang tìm kiếm.
    *   **Key ($K$):** Nhãn nhận diện của từng token khác.
    *   **Value ($V$):** Nội dung thông tin của token đó.
*   Công thức Attention nổi tiếng (bạn **bắt buộc phải thuộc lòng** khi thi/phỏng vấn VinAI):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

*(Trong đó $\sqrt{d_k}$ là căn bậc hai số chiều của vector Key, dùng để chia nhỏ tích vô hướng giúp hàm Softmax không bị bão hòa gradient khi số chiều lớn).*

*   **Tại sao không nên nhét tài liệu quá dài vào Context Window?**
    Vì mỗi token đều phải so sánh ($Q \times K^T$) với tất cả các token đứng trước nó, nên chi phí tính toán của Self-Attention tăng theo **bình phương độ dài chuỗi** $\mathcal{O}(N^2)$. Chuỗi dài gấp 10 lần thì lượng tính toán nặng gấp 100 lần! Đó là lý do ta cần **RAG** (chỉ trích xuất đúng đoạn văn liên quan khoảng vài nghìn token đưa vào LLM).

### 4. Hai giai đoạn khi LLM trả lời: Prefill vs Decode & Bộ nhớ KV Cache
Khi bạn gửi một Prompt dài 2.000 tokens (chứa câu hỏi + tài liệu báo cáo tài chính) vào LLM và nhận về câu trả lời 200 tokens, quá trình suy luận (Inference) diễn ra qua 2 pha có bản chất phần cứng hoàn toàn khác nhau:

1.  **Pha 1 - Prefill (Đọc hiểu Prompt đầu vào):**
    *   Mô hình xử lý toàn bộ 2.000 tokens đầu vào **song song cùng một lúc** để tính ra các ma trận Key ($K$) và Value ($V$) của 2.000 tokens đó, rồi sinh ra **token trả lời đầu tiên**.
    *   Thời gian của pha này quyết định chỉ số **TTFT (Time To First Token)** — thời gian từ lúc người dùng bấm gửi đến khi thấy chữ đầu tiên hiện ra trên màn hình.
2.  **Pha 2 - Decode (Sinh từng chữ tiếp theo):**
    *   Mô hình sinh lần lượt token thứ 2, thứ 3... đến thứ 200 (tuần tự từng chữ một).
    *   Để không phải tính lại vector $K$ và $V$ của 2.000 tokens ban đầu ở mỗi bước, người ta lưu toàn bộ ma trận $K, V$ đã tính vào bộ nhớ GPU (VRAM). Vùng nhớ này gọi là **KV Cache**.
    *   Thời gian giữa mỗi chữ sinh ra quyết định chỉ số **ITL (Inter-Token Latency)** — tốc độ chữ chạy nhanh hay chậm trước mắt người dùng.

### 5. Các tham số điều khiển đầu ra của LLM (Sampling Parameters)
Khi gọi LLM trong code, bạn sẽ luôn phải cấu hình các thông số này tùy theo tác vụ:

*   **`temperature` (Nhiệt độ - thường từ `0.0` đến `1.0`):**
    *   Làm phẳng hoặc làm nhọn phân phối xác suất trước khi bốc thăm chọn từ.
    *   **Trong dự án Tài chính / RAG / Text-to-SQL của chúng ta:** Luôn đặt `temperature = 0.0` (hoặc `0.1`). Khi `temperature = 0` (Greedy Decoding), mô hình luôn chọn từ có xác suất cao nhất, đảm bảo tính nhất quán, chính xác, không sáng tác lung tung.
*   **`top_p` (Nucleus Sampling - ví dụ `0.9`):**
    *   Chỉ xét nhóm các từ có tổng xác suất cộng lại đạt 90%, loại bỏ hoàn toàn 10% các từ kỳ quặc ở đuôi phân phối.
*   **`max_tokens`:** Giới hạn số lượng token tối đa mà mô hình được phép sinh ra trong câu trả lời (tránh việc mô hình bị kẹt trong vòng lặp sinh chữ vô tận gây tốn tiền/treo máy).

---

## Tóm tắt Kiến thức Cần Nhớ Từ Bài 01
1.  Luôn viết Python có **Type Hints**, dùng **Pydantic** để kiểm soát dữ liệu JSON từ LLM, và dùng **AsyncIO** để gọi DB/LLM song song.
2.  LLM chỉ là cỗ máy dự đoán **xác suất token tiếp theo** $P(w_t \mid w_{<t})$. Nó không phải là cơ sở dữ liệu lưu trữ sự thật — vì vậy muốn nó trả lời đúng số liệu tài chính, ta phải cung cấp ngữ cảnh chuẩn cho nó qua **RAG** hoặc **SQL Tool** và đặt `temperature = 0.0`.
3.  Hiểu công thức **Self-Attention**, vai trò của **KV Cache**, và hai chỉ số tốc độ **TTFT** (pha Prefill) vs **ITL** (pha Decode).
