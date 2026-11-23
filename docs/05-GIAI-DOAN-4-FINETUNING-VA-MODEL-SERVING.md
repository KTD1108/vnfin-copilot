# 05. Giai Đoạn 4: Tinh Chỉnh Mô Hình (Fine-Tuning SLM) & Tối Ưu Hóa Suy Luận (vLLM Serving)

> **Mục tiêu tài liệu:** Đây là chương **quan trọng bậc nhất để tạo sự khác biệt khi ứng tuyển VinAI**. Phần lớn ứng viên hiện nay chỉ dừng lại ở mức gọi API của OpenAI/Gemini để làm RAG hoặc Agent. Chương này giúp bạn hiểu sâu về kiến trúc bên trong của mô hình Transformer, biết cách tự huấn luyện (Fine-tune) một Mô hình Ngôn ngữ Nhỏ (**SLM** như `Qwen-2.5-3B` hoặc `7B`) trên bài toán Tài chính tiếng Việt, nén mô hình (**Quantization**) và triển khai phục vụ tốc độ cao bằng **vLLM**.

---

## 1. Khi Nào Dùng RAG? Khi Nào Cần Fine-Tuning? (Câu Hỏi Phỏng Vấn Kinh Điển)

Rất nhiều người mới hiểu sai rằng: *"Muốn mô hình biết báo cáo tài chính mới của FPT thì đem báo cáo đó đi Fine-tune mô hình"*. **Đó là cách làm sai!**

Hãy nhớ quy tắc vàng sau:
*   **RAG = Cung cấp Tri thức mới (Knowledge Injection):** Dùng khi dữ liệu thay đổi liên tục (giá cổ phiếu hôm nay, báo cáo tài chính quý mới), cần trích dẫn nguồn chính xác và chống bịa đặt.
*   **Fine-Tuning = Dạy Kỹ năng, Định dạng hoặc Ngôn ngữ chuyên ngành (Behavior & Skill Adaptation):** Dùng khi bạn muốn một mô hình nhỏ (như `Qwen-2.5-3B` miễn phí, chạy cực nhanh trên 1 GPU cá nhân) học được một **kỹ năng chuyên biệt** đạt độ chính xác ngang ngửa mô hình khổng lồ (như GPT-4o):
    *   *Ví dụ 1 trong dự án VnFinCopilot:* Huấn luyện mô hình chuyển đổi câu hỏi tài chính tiếng Việt lắt léo sang câu lệnh PostgreSQL chuẩn xác (**Vietnamese Financial Text-to-SQL**).
    *   *Ví dụ 2:* Huấn luyện mô hình **Query Router & Structured Extractor** (luôn trả về JSON phân tích ý định câu hỏi với độ trễ < 100ms mà không cần tốn tiền gọi API thương mại).

---

## 2. Bản Chất Toán Học Của LoRA & QLoRA (Tại Sao Fine-Tune Được LLM Trên 1 GPU Nhỏ?)

### 2.1. Bài toán bộ nhớ của Full Fine-Tuning
Một mô hình vừa phải như `Qwen-2.5-7B` có **7 tỷ tham số (parameters)**.
*   Nếu lưu ở định dạng số thực 16-bit (`FP16` hoặc `BF16` — mỗi tham số tốn 2 bytes): chỉ riêng việc tải trọng số mô hình lên GPU đã tốn $7 \times 2 = 14\text{ GB VRAM}$.
*   Nhưng khi huấn luyện toàn bộ 7 tỷ tham số (**Full Fine-Tuning**), bạn phải lưu thêm **Đạo hàm (Gradients - 14 GB)** và **Trạng thái bộ tối ưu AdamW (Optimizer States - 28 GB)** cùng với **Activations**. Tổng cộng bạn cần khoảng **60 – 80 GB VRAM** (phải thuê siêu máy chủ A100/H100 đắt đỏ)!

### 2.2. Giải pháp đột phá: LoRA (Low-Rank Adaptation)
Năm 2021, các nhà nghiên cứu phát hiện ra rằng: khi tinh chỉnh một mô hình đã được huấn luyện sẵn (Pre-trained LLM) cho một tác vụ cụ thể, ma trận thay đổi trọng số $\Delta W$ có **"hạng nội tại rất thấp" (Low Intrinsic Rank)** — tức là không cần phải cập nhật cả một ma trận khổng lồ.

Giả sử một tầng tuyến tính trong Transformer có ma trận trọng số gốc là $W_0 \in \mathbb{R}^{d \times d}$ (ví dụ $4096 \times 4096 = 16.777.216$ tham số).
Thay vì cập nhật trực tiếp $W_0$, **LoRA đóng băng hoàn toàn $W_0$** (không tính gradient cho $W_0$) và cộng thêm một nhánh song song gồm tích của **2 ma trận cực nhỏ** $B$ và $A$ có hạng $r$ rất nhỏ (ví dụ $r = 16 \ll 4096$):

$$W_{\text{mới}} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$$

Trong đó:
*   $A \in \mathbb{R}^{r \times d}$ (kích thước $16 \times 4096 = 65.536$ tham số).
*   $B \in \mathbb{R}^{d \times r}$ (kích thước $4096 \times 16 = 65.536$ tham số).
*   Tổng số tham số cần huấn luyện của $A$ và $B$ chỉ là $131.072$ tham số — **giảm hơn 99% số tham số cần huấn luyện** so với ma trận gốc $16,7$ triệu tham số!
*   **Các siêu tham số của LoRA bạn cần nắm rõ:**
    *   **`r` (Rank):** Hạng của ma trận nén (thường chọn `16`, `32` hoặc `64`). `r` càng lớn thì mô hình học được tác vụ càng phức tạp nhưng tốn thêm chút VRAM.
    *   **`lora_alpha` ($\alpha$):** Hệ số khuếch đại mức độ ảnh hưởng của nhánh LoRA (thường đặt $\alpha = 2 \times r$, ví dụ `r = 16` thì `lora_alpha = 32`).
    *   **`target_modules`:** Các ma trận trong khối Transformer được gắn nhánh LoRA (thường gắn vào cả các tầng Attention `q_proj, k_proj, v_proj, o_proj` và các tầng MLP `gate_proj, up_proj, down_proj`).

### 2.3. QLoRA (Quantized LoRA)
**QLoRA** tiến thêm một bước nữa để bạn có thể Fine-tune mô hình 7B ngay trên 1 card GPU phổ thông (như RTX 3060/4060 hoặc Google Colab T4 16GB miễn phí):
1.  Nén ma trận trọng số gốc $W_0$ đã đóng băng xuống định dạng **4-bit NormalFloat (`NF4`)** kèm **Double Quantization** (giúp mô hình 7B chỉ còn chiếm ~4.5 GB VRAM thay vì 14 GB).
2.  Trong khi đó, hai ma trận nhỏ $A$ và $B$ của nhánh LoRA vẫn được huấn luyện ở độ chính xác cao **16-bit (`BF16`)**.
3.  Khi kết hợp với thư viện tối ưu hóa kernel CUDA như **Unsloth**, tốc độ huấn luyện tăng gấp 2 lần và tiết kiệm thêm 60% VRAM!

---

## 3. Hai Giai Đoạn Tinh Chỉnh: SFT vs DPO

Khi huấn luyện mô hình cho dự án:
1.  **SFT (Supervised Fine-Tuning - Tinh chỉnh có giám sát):**
    *   Dạy mô hình bằng các cặp `(Câu hỏi + Schema Database) -> (Câu lệnh SQL chuẩn)`.
    *   Mô hình học cách bắt chước định dạng và cú pháp chuẩn.
2.  **DPO (Direct Preference Optimization - Tối ưu hóa theo sự ưu tiên trực tiếp):**
    *   Sau khi chạy SFT, mô hình đã biết viết SQL nhưng đôi khi vẫn viết câu SQL chưa tối ưu hoặc vi phạm quy tắc nghiệp vụ.
    *   Ta tạo bộ dữ liệu gồm bộ ba: `(Prompt, Câu trả lời Tốt - Chosen, Câu trả lời Tệ/Lỗi - Rejected)`.
    *   Thuật toán **DPO** trực tiếp đẩy xác suất sinh ra câu `Chosen` lên cao và ép xác suất sinh ra câu `Rejected` xuống thấp mà không cần phải huấn luyện một mô hình chấm điểm riêng (Reward Model) phức tạp như RLHF truyền thống.

---

## 4. Lượng Tử Hóa Mô Hình (Quantization) & Triển Khai Suy Luận Tốc Độ Cao Với `vLLM`

Sau khi Fine-tune xong và gộp nhánh LoRA vào mô hình gốc, làm sao đưa mô hình lên server phục vụ hàng chục người dùng cùng lúc với tốc độ cực nhanh?

### 4.1. Bản chất Phần cứng: Tại sao sinh chữ (LLM Decoding) lại bị nghẽn cổ chai ở Băng thông Bộ nhớ (Memory-Bandwidth Bound)?
*(Đây là câu hỏi phân loại ứng viên xuất sắc nhất tại VinAI!)*
*   Như đã học ở Bài 01, ở pha **Decode** (sinh từng chữ một), để sinh ra **1 token duy nhất**, GPU phải đọc toàn bộ hàng tỷ tham số của mô hình từ bộ nhớ VRAM đưa vào các nhân tính toán (Tensor Cores/ALU), thực hiện phép nhân ma trận-vector rồi mới ra được 1 chữ.
*   Các nhân tính toán của GPU hiện đại chạy cực nhanh, nhưng **tốc độ truyền dữ liệu từ VRAM vào nhân tính toán (Memory Bandwidth)** lại không theo kịp! GPU dành phần lớn thời gian chỉ để chờ tải trọng số từ VRAM.
*   👉 **Hệ quả:** Khi bạn nén mô hình từ **16-bit (`FP16`) xuống 4-bit (`AWQ` - Activation-aware Weight Quantization)**, dung lượng trọng số cần đọc từ VRAM giảm đi **4 lần** -> Không chỉ giúp tiết kiệm 75% VRAM mà còn giúp **tốc độ sinh chữ (Tokens/giây) tăng lên gấp 2.5 – 3 lần** vì giảm được nghẽn băng thông bộ nhớ!

### 4.2. Tại sao không dùng `HuggingFace Transformers` để chạy server mà phải dùng `vLLM`?
Nếu bạn dùng hàm `model.generate()` mặc định của HuggingFace để làm API server, hệ thống sẽ cực chậm và nhanh tràn VRAM vì 2 lý do mà **vLLM** đã giải quyết triệt để:

#### a. Thuật toán PagedAttention (Quản lý bộ nhớ KV Cache như Hệ điều hành)
*   Khi một người dùng gửi câu hỏi vào LLM, hệ thống không biết trước câu trả lời sẽ dài 50 tokens hay 2.000 tokens.
*   Các hệ thống cũ phải **cấp phát trước một vùng VRAM liên tục dài tối đa (ví dụ 4.096 tokens)** cho mỗi request để chứa **KV Cache**. Nếu người dùng chỉ hỏi câu ngắn 100 tokens, hơn 95% vùng VRAM đã đặt chỗ đó bị bỏ trống lãng phí (**Internal Fragmentation**), khiến GPU chỉ phục vụ được vài người cùng lúc là báo lỗi `Out of Memory (OOM)`.
*   **PagedAttention của vLLM:** Lấy cảm hứng từ cơ chế **Bộ nhớ ảo phân trang (Virtual Memory & Paging)** trong hệ điều hành Linux:
    *   Chia KV Cache thành các khối nhỏ không cần nằm liên tiếp nhau trong VRAM (gọi là **KV Blocks**, mỗi block chứa 16 hoặc 32 tokens).
    *   Sinh thêm chữ đến đâu mới cấp phát thêm block đến đó! Nhờ vậy, tỷ lệ lãng phí bộ nhớ KV Cache giảm từ **60-80% xuống dưới 4%**, cho phép nhét số lượng request chạy đồng thời (Batch size) lớn gấp 5–10 lần!

#### b. Continuous Batching (Gom nhóm liên tục ở cấp độ từng vòng lặp)
*   **Static Batching (Cách cũ):** Gom 4 câu hỏi vào chạy cùng lúc. Câu số 1 chỉ sinh 20 chữ là xong, nhưng câu số 4 sinh tới 500 chữ. Ba câu đã xong sớm vẫn phải nằm chờ câu số 4 chạy hết 500 chữ thì cả nhóm mới được trả kết quả và nhận câu hỏi mới!
*   **Continuous Batching (vLLM):** Ngay ở giây câu số 1 vừa sinh ra token kết thúc `<|im_end|>`, nó lập tức được trả về cho người dùng và **nhét ngay một câu hỏi mới đang xếp hàng vào chỗ trống đó** ở vòng lặp token tiếp theo mà không cần chờ các câu khác trong batch!

---

## Tóm tắt Checklist Kỹ Thuật Bạn Sẽ Làm Ở Giai Đoạn 4
1.  Tạo bộ dữ liệu **3.000 mẫu Vietnamese Financial Text-to-SQL & Query Routing** (sử dụng kỹ thuật Synthetic Data Generation có kiểm duyệt bằng thực thi SQL thực tế).
2.  Viết script huấn luyện **QLoRA Fine-Tuning** trên mô hình `Qwen-2.5-3B-Instruct` bằng `Unsloth` / `TRL`.
3.  Đánh giá độ chính xác của mô hình `Qwen-2.5-3B` trước và sau khi Fine-tune bằng chỉ số **SQL Execution Accuracy (EX)** (tỷ lệ câu SQL chạy ra đúng bảng kết quả trên PostgreSQL).
4.  Xuất mô hình đã nén **4-bit AWQ** và triển khai bằng **vLLM** thành API chuẩn OpenAI-compatible.
5.  Viết script **Load Testing & Benchmark** đo các chỉ số: `TTFT (ms)`, `ITL (ms/token)`, `Throughput (tokens/s)` khi bắn đồng thời 1, 10, 30 requests.
