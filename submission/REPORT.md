# Lab 21 — Evaluation Report

**Họ tên**: Vũ Bá Anh  **MSSV**: 2A202602893  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4, 14.6 GB khả dụng (fp16, không có bf16)

> Mọi con số dưới đây lấy từ `results/` (`verdict.json`, `baselines_frozen.json`, `runs.csv`,
> `autopsy.json`, `qualitative.json`, `mask_proof.json`, `token_stats.json`, `template_check.json`).

---

## 1. Setup

| | |
|---|---|
| Base model | `unsloth/Qwen3.5-4B` — mặc định của tier T4 |
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định của lab) |
| Train / val | 225 / 25 (seed 42); eval: 50 mẫu target + 15 câu regression, dùng đủ, không đặt `EVAL_LIMIT` |
| `max_length` | 1024 (mặc định của tier); p95 đo được là **98** token *(token_stats.json)*, gợi ý 256 |
| `MASK_MODE` | `assistant-only` |
| LoRA | all-linear của text decoder, 12 loại module, r=16, alpha=32, LR 1e-4, fp16 |
| Epochs / max_steps | 2 epoch = **30 step**, batch 1 × grad-accum 16 = 16 hiệu dụng |

**Lý do chọn.** Tôi chạy cấu hình mặc định của lab (Qwen3.5-4B, corpus mặc định). Lý do là Colab Free
T4 chỉ có 14.6 GB, mà 4B bf16-LoRA đo được chỉ dùng 8.78 GB đỉnh, còn 9B sẽ không vừa. Tôi không đổi
dataset để giữ checksum tập eval và SHA prompt (b) khớp với gatekeeper, nhờ đó phép so sánh là phép đã
được lab kiểm chứng. Cái giá là bài toán này dễ, nên kết quả cho một phép thử khá sạch nhưng không nói
được nhiều về dữ liệu khó.

**`max_length` lệch p95.** Mẫu dài nhất chỉ 101 token (max, p99 = 100) nên đặt 1024 không cắt mẫu nào và
không đổi loss. Phần chi phí là bộ nhớ dự phòng, và peak VRAM 8.78 GB vẫn thoải mái dưới 14.6 GB. Tôi giữ
1024 vì đó là mặc định của tier và không có lý do đo được để đổi; nếu chạy lại tôi sẽ đặt 256.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`: `open_tag_present = true`,
`body_present = true`, verdict "reasoning preserved". Tuy vậy chuỗi huấn luyện của bài này bắt đầu bằng khối
`<think>` **rỗng** (xem `masked_preview` trong mask_proof), nên mô hình không học suy luận nào ở đây.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token trên mẫu minh hoạ; 9014/20951 = 43.0% trên tập train) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị che (`masked_preview`) gồm system, câu hỏi của user và phần mở đầu `<|im_start|>assistant\n<think>\n\n`.
Chế độ đối chứng `everything` cho 94/94 (100%), tức là tính loss cả trên prompt; `assistant-only` thấp hơn
ngưỡng 0.95 nên mục 1.1 không bị trừ.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3542.7 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1060.5 |
| (c) LoRA fine-tune | 0.970 | 0.5444 | 1.000 | 1596.7 |

**(b) có mạnh hơn (a) không?** Có: target 0.000 → 0.765, format 0.000 → 1.000. (a) cho target 0 vì không
sinh được JSON hợp lệ (format = 0), tức là (a) thua vì định dạng chứ không phải vì không hiểu ticket. Tôi
**không sửa** `OPTIMIZED_PROMPT` (gatekeeper xác nhận SHA `719e74d3b6232053` nguyên vẹn), nên (b) là mốc
nguyên bản của lab.

**Về độ nhiễu của số đo.** Tôi đã chạy NB2 và NB5 hai lần với cùng adapter. Lần đầu (output notebook) cho
regression của (c) = 0.6778 (Δ = −0.113); lần chạy lưu trong `results/` cho 0.5444 (Δ = −0.247). Latency
của (c) cũng đo được 1534.8 ms (autopsy) và 1596.7 ms (verdict). Tập regression chỉ có 15 câu nên mỗi lần
sinh văn bản lệch vài câu là điểm đổi nhiều. Báo cáo này dùng số trong `results/`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6259 | **0.97** | 428.7 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5373 | **0.97** | 270.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.00** | 400.6 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 467.7 | 3.86 |

Cả bốn run dùng đúng 30 step. `attn_only` lệch ngân sách tham số (32,456,704 so với 32,464,896) chỉ
0.025%, rất xa ngưỡng 5%. `runs.csv` có hai dòng `correct` (loss 0.6264 và 0.6259) vì NB3 được chạy hai
lần; bảng dùng lần cuối, lần đã sinh ra adapter đem đi đánh giá. Cột "train loss" là loss **trung bình cả
run** (`train_loss`), kéo cao bởi các step đầu (loss 2.16 ở step 5); ở step cuối loss chỉ còn ~0.027.

**4.1 — `attn_only` so với `correct`.** Hai run có cùng ~32.5 M tham số và hoà nhau trên target (0.97 và
0.97, cùng format 1.0). Theo train loss thì `attn_only` "thắng" (0.537 so với 0.626), nên hai thứ tự khác
nhau: xếp theo loss thì `attn_only` đứng nhất, xếp theo target thì chúng hoà. Đây chính là lý do rubric đòi
xếp hạng bằng target. Kết luận hợp lý nhất là trên bài toán dễ này, với số tham số đã khớp, *vị trí gắn
adapter không phải là đòn bẩy*; ngay cả khi chỉ gắn vào q,v với rank cực lớn (283), mô hình vẫn đạt trần.
Tôi không kết luận vị trí "không quan trọng" nói chung: trên 50 mẫu, 0.97 so với 0.97 không phân biệt được
hai run, `attn_only` với r=283 là cấu hình bất thường, và tôi chưa đo regression của `attn_only` nên không
biết nó có quên ít hơn hay không. Một điểm phụ đáng ghi: `attn_only` sinh nhanh hơn rõ (954.6 ms so với
1534.8 ms), và huấn luyện nhanh hơn (270.6 s so với 428.7 s); tôi đoán là do chỉ 2 loại module có nhánh adapter, nhưng chưa kiểm chứng.

**4.2 — `wrong_lr`.** Chỉ đổi LR từ 1e-4 sang 1e-5 (thang full-FT) mà loss trung bình chỉ giảm tới 1.570
(so với 0.626), và kết quả là target = 0.00, format = 0.00: mô hình không học được cách xuất JSON. Cái bẫy
nằm ở chỗ nếu chỉ nhìn đường loss mà không biết LR, tôi dễ kết luận "dữ liệu khó" hoặc "LoRA không học
được bài này", trong khi dữ liệu và kiến trúc y hệt `correct` và chỉ khác một con số. Loss 1.57 sau 30
step là triệu chứng của việc đi quá chậm, không phải của việc bài toán không học được. Latency 5551.5 ms
cũng cao bất thường nhưng tôi chưa mở đầu ra để biết nguyên nhân, nên không khẳng định.

**4.3 — `qlora`.** QLoRA 4-bit hạ VRAM đỉnh từ 8.78 GB xuống 3.86 GB (giảm 4.92 GB, ~56%). Giá phải trả
đo được: target 0.94 so với 0.97 (−0.03, tức là thêm khoảng 1.5 mẫu sai trên 50), train lâu hơn (467.7 s
so với 428.7 s, +9%) và sinh chậm hơn (1872.3 ms so với 1534.8 ms, +22%). Tôi chưa đo regression của
`qlora`. Vì vậy số đo của tôi *ủng hộ một phần* khuyến nghị "đừng dùng QLoRA cho dòng này": tụt chất lượng
là có thật nhưng nhỏ, và 50 mẫu không đủ để khẳng định nó nằm ngoài nhiễu. Trên T4 16 GB, bf16-LoRA đã
vừa (8.78 GB), nên không có lý do gì phải trả giá đó; QLoRA chỉ đáng dùng khi VRAM thật sự không đủ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.247` · `valid_trace_rate = 0.00`

Fine-tune thắng rõ trên đúng bài nó được huấn luyện: target từ 0.765 lên 0.970, format giữ 1.000. Nhưng
cổng hồi quy bị chặn ở nhóm thứ hai: điểm kiến thức phổ thông giảm từ 0.7911 xuống 0.5444, trong khi ngưỡng
chấp nhận là 0.02. Nghĩa là cái giá của +0.205 là mất khoảng một phần tư năng lực chung của mô hình. Nguyên
nhân khả dĩ nhất là quên thảm hoạ: 30 step trên 225 ticket cùng một khuôn (một định dạng JSON, một miền)
đẩy trọng số về phân phối hẹp mà không có dữ liệu phổ thông nào kéo lại. Đây chính là cảnh báo của công
cụ (deck §6.3): trộn 1–5% dữ liệu phổ thông vào tập train. Tôi chưa thử điều đó nên đây vẫn là giả thuyết,
chưa phải kết luận đã kiểm chứng. Ngoài ra regression chỉ có 15 câu, và hai lần chạy cho 0.678 và 0.544
(§3), nên con số −0.247 không chính xác; nhưng cả hai lần đều vượt xa ngưỡng 0.02, nên bản FAILED là
vững. `valid_trace_rate = 0.00` không mang ý nghĩa ở bài này: dữ liệu huấn luyện không có chuỗi suy luận
(khối `<think>` rỗng), nên không có "trace" nào để mất. Phán quyết FAILED là kết quả hợp lệ và tôi không
nới cổng để biến nó thành PASS.

---

## 6. Định tính — gồm cả ca THUA

`qualitative.json` chỉ lưu dự đoán của bản fine-tune (cắt ở ~100 ký tự) và điểm của nó. Nó **không** lưu
dự đoán của (b) theo từng ca, nên các dòng dưới đây so với **nhãn đúng** (lấy từ `data/eval_target.jsonl`),
không so được với (b). Điểm mỗi ca = tỉ lệ trong 4 trường đúng.

| # | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 0 | chuột không dây · Cho tôi trả lại. Gấp. | doi_tra · cao | doi_tra · cao (1.0) | ✅ đúng |
| 4 | đèn bàn LED · Vỡ khi nhận. Gấp. | san_pham_loi · cao | san_pham_loi · cao (1.0) | ✅ đúng |
| 47 | ốp lưng điện thoại · Shipper khô… | van_chuyen · thap | van_chuyen · thap (1.0) | ✅ đúng |
| 3 | bình giữ nhiệt · Chưa thấy tiền. **Khi nào tiện.** | urgency = **thap** | urgency = **trung_binh** (0.75) | ❌ **thua**: sai urgency |
| 12 | áo khoác gió · Bị lỗi. **Khi nào tiện.** | urgency = **thap** | urgency = **trung_binh** (0.75) | ❌ **thua**: sai urgency |
| 5, 39, 41, 46 | các sản phẩm khác · **Khi nào tiện.** | urgency = thap | urgency = trung_binh (0.75) | ❌ thua |

**Mẫu chung của các ca thua.** Cả 6 ca fine-tune sai (index 3, 5, 12, 39, 41, 46) đều sai ở **đúng một
trường, urgency**, và đều cùng hướng thap → trung_binh. Cả 6 ticket đều chứa câu **"Khi nào tiện."**. Trong
tập train, 35/35 ticket chứa câu này đều có nhãn `thap`, và trong tập eval 6/6 cũng vậy; tức là có một
dấu hiệu hoàn toàn nhất quán mà mô hình vẫn chưa nắm được sau 30 step. Điều này khớp với con số: 6 ca ×
0.25 điểm / 50 mẫu = 0.03, ra đúng target 0.97; ba trường còn lại (intent, product, sentiment) đúng ở cả
50 mẫu, và urgency đúng 44/50. Đây là lỗi hệ thống chứ không phải ngẫu nhiên, và nó cho thấy "0.97" che
giấu một kiểu sai duy nhất. Tôi chưa kiểm chứng (b) làm đúng hay sai trên các ca này.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** nên deploy bản fine-tune này nguyên trạng, dù nó thắng (b) trên bài target
(0.970 so với 0.765). Lý do là phán quyết FAILED không phải chuyện nhỏ: nó đánh đổi +0.205 điểm trên bài
của mình lấy −0.247 điểm năng lực chung, và nếu sản phẩm thật có khách hàng hỏi ngoài khuôn ticket thì mô
hình sẽ tệ hơn base. Có hai hướng hợp lý. Nếu hệ thống chỉ phục vụ đúng bài triage (đầu vào luôn là ticket),
thì mất regression có thể chấp nhận được và bản fine-tune thắng rõ; (b) là cách dễ hơn nhưng kém 0.2 điểm.
Nếu cần giữ năng lực chung, tôi sẽ trộn 1–5% dữ liệu phổ thông và chạy lại cổng hồi quy trước khi quyết
định. Đòn bẩy thật sự trong lab này theo bằng chứng của tôi là **learning rate** chứ không phải vị trí hay
rank: `wrong_lr` chỉ khác một con số (1e-5 so với 1e-4) mà target rơi từ 0.97 xuống 0.00, trong khi
`attn_only` (đổi vị trí và rank) và `qlora` (đổi độ chính xác) chỉ lệch 0.00–0.03. Mask quan trọng ở một
tầng khác: nó là điều kiện để mọi số sau đó có nghĩa, và đã được chứng minh (0.4149, không phải 100%).
Ngoài ra thứ tự theo train loss không trùng thứ tự theo target (`attn_only` thấp loss nhất nhưng chỉ hoà
`correct`), nên chấm bằng loss sẽ cho kết luận sai. Giới hạn của tôi: n = 50 (target) và n = 15 (regression)
nên các chênh lệch dưới ~0.03 là trong nhiễu; một seed mỗi run; chưa đo regression cho ba run đối chứng; và
không có dự đoán (b) theo từng ca nên chưa so sánh (b) với fine-tune ở mức mẫu.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Train loss không xếp hạng được các run.** `attn_only` có loss thấp nhất (0.537 so với 0.626 của
   `correct`) nhưng target chỉ hoà (0.97 và 0.97). Loss ở đây còn là trung bình cả run, kéo cao bởi các
   step đầu (2.16 ở step 5, còn ~0.027 ở cuối). Từ giờ tôi chỉ xếp hạng bằng chỉ số đánh giá trên tập
   giữ riêng, và dùng loss để phát hiện run hỏng (như `wrong_lr` ở 1.57) chứ không để chọn run tốt nhất.
2. **Điểm tổng có thể che một kiểu lỗi duy nhất.** Target 0.97 trông gần hoàn hảo, nhưng cả 6 ca sai đều
   là urgency `thap` bị đoán thành `trung_binh`, và đều chứa "Khi nào tiện." (35/35 ví dụ train có câu này
   đều là `thap`). Nên tôi sẽ luôn tách lỗi theo trường và theo cụm từ trước khi tin vào một con số trung
   bình.
3. **Số đo trên tập nhỏ dao động nhiều hơn mức tôi tưởng.** Cùng một adapter, regression cho 0.678 ở lần
   chạy này và 0.544 ở lần chạy khác (15 câu, lệch 0.13), trong khi target giữ nguyên. Cổng hồi quy vẫn
   FAILED ở cả hai lần, nhưng tôi sẽ không báo cáo một con số đơn lẻ như −0.247 mà không kèm khoảng dao
   động, và sẽ dùng tập regression lớn hơn.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn 1–5% dữ liệu phổ thông vào tập train rồi chạy lại cổng
hồi quy, để kiểm chứng giả thuyết quên thảm hoạ (hiện mới là giả thuyết); (2) chạy 3 seed cho `correct` để
đo độ nhiễu thật; (3) lưu dự đoán của (b) theo từng ca để so (b) với fine-tune ở mức mẫu; (4) đo regression
cho `attn_only` và `qlora`, vì tôi mới chỉ đo target của hai run này.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
