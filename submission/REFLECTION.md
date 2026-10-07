# Reflection — Lab 21

**Họ tên**: Vũ Bá Anh  **MSSV**: 2A202602893

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Cả 6 ca fine-tune sai trên 50 mẫu đều sai đúng một chỗ: cùng trường `urgency`, cùng hướng `thap` → `trung_binh`, và cả 6 ticket đều chứa câu "Khi nào tiện." Trong tập train, 35/35 ticket chứa câu đó đều có nhãn `thap`. Tức là mô hình đạt 0.97 nhưng vẫn không bắt được một dấu hiệu hoàn toàn nhất quán. Con số 0.97 trông gần hoàn hảo, nhưng thực chất che một lỗi hệ thống.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi dự đoán. Tôi tưởng phần train tốn nhất, nhưng NB3 chỉ mất khoảng 7 phút. Thời gian đi vào việc chạy lại, vì ô 3 của notebook để mặc định `EVAL_LIMIT=8` (bản rút gọn, không nộp được), nên phải chạy lại bản đầy đủ. Ngoài ra là việc gom kết quả: ipynb không mang đủ file, tôi phải thêm ô đóng gói `results/` và tải zip về, rồi phát hiện NB5 đã chạy hai lần cho hai bộ số khác nhau.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin rằng loss thấp hơn nghĩa là mô hình tốt hơn. `attn_only` có train loss thấp nhất (0.537 so với 0.626 của `correct`) nhưng target chỉ hoà (0.97 và 0.97). Tôi cũng từng nghĩ fine-tune thắng baseline là đủ để kết luận "nên dùng". Ở đây nó thắng target +0.205 nhưng cổng hồi quy FAILED vì regression tụt 0.247, nên điều đó không đủ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude để đọc repo và tóm tắt đề, kiểm tra notebook trước khi chạy (nó chỉ ra `EVAL_LIMIT=8` là mặc định), soạn ô tải kết quả, đối chiếu `results/` với nhãn đúng để tìm ra mẫu lỗi "Khi nào tiện", và viết bản nháp `REPORT.md`. Chỗ nó chưa đủ: `qualitative.json` không lưu dự đoán của baseline (b) theo từng ca, nên so sánh "fine-tune thua (b)" không làm được và report phải so với nhãn đúng. Hai bộ số regression (0.678 và 0.544) chỉ lộ ra khi tôi tải file về và đối chiếu, chứ không nằm trong output notebook.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng một tập đánh giá và đo baseline (b) với prompt tốt nhất có thể trước khi huấn luyện bất kỳ thứ gì, kèm một tập kiểm tra năng lực chung đủ lớn (15 câu là quá ít, kết quả dao động tới 0.13 giữa hai lần chạy). Chỉ khi (b) không đạt yêu cầu mới tính đến fine-tune, và nếu fine-tune thì trộn 1–5% dữ liệu phổ thông ngay từ đầu.
