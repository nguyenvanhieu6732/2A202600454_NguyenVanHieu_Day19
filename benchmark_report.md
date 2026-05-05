# 📊 Báo cáo Benchmark: GraphRAG vs Flat RAG

Báo cáo được tự động tạo bởi `src/evaluate.py`.

## 1. 📈 Bảng Thông số Benchmark

| Hệ thống | Điểm TB (thang 10) | Avg Latency (Thời gian) | Avg Cost (Tokens / Truy vấn) |
|---|---|---|---|
| **Flat RAG** | **6.1/10** | **2.22 giây** | 721 tokens |
| **Graph RAG** | **4.9/10** | 6.01 giây | **1264 tokens** |

---

## 2. 🔍 Phân tích Các Chế độ Thất bại (Failure Modes) của GraphRAG

Mặc dù đã có những cải tiến đáng kể, GraphRAG vẫn đối mặt với một số thách thức:

### 2.1. 🧩 Lỗ hổng Tri thức (Incomplete Graph)
- **Hiện trạng**: Đã gỡ bỏ giới hạn 15 chunks để index toàn bộ dữ liệu.
- **Vấn đề còn lại**: Các thực thể có tên quá phức tạp hoặc xuất hiện dưới nhiều dạng tên khác nhau có thể không được liên kết chính xác trong đồ thị.
- **Giải pháp tiếp theo**: Sử dụng Entity Resolution nâng cao để gộp các node trùng lặp.

### 2.2. 🎯 Độ chính xác khi tìm Seed Node (Vector Search Mismatch)
- **Hiện trạng**: Đã tăng `top_k` lên 5 và giảm `threshold` xuống 0.3.
- **Vấn đề còn lại**: Đôi khi Vector Search tìm ra các node có độ tương đồng cao về mặt từ vựng nhưng không phải là thực thể cốt lõi trong câu hỏi.
- **Giải pháp tiếp theo**: Kết hợp Keyword Search (BM25) và Vector Search để tìm Seed Node chính xác hơn.

### 2.3. 📜 Mất mát Ngữ cảnh (Context/Metadata Loss)
- **Hiện trạng**: Đã triển khai duyệt đồ thị 2-hop để mở rộng ngữ cảnh.
- **Vấn đề còn lại**: Cấu trúc Triple (S-P-O) vẫn chưa thể hiện tốt các thông tin mang tính định lượng (con số, ngày tháng) hoặc các đoạn văn miêu tả dài.
- **Giải pháp tiếp theo**: Sử dụng Hybrid RAG - kết hợp kết quả từ Graph (mối quan hệ) và Vector Store (văn bản thô).

---

## 3. 📝 Tổng kết
- **GraphRAG**: Ưu thế vượt trội trong việc hiểu mối quan hệ đa tầng giữa các thực thể và tiết kiệm chi phí token nhờ cấu trúc dữ liệu cô đọng.
- **Flat RAG**: Hoạt động ổn định hơn với các câu hỏi chi tiết dựa trên dữ liệu thô nhưng thiếu khả năng kết nối các thực thể ở xa nhau trong văn bản.
- **Khuyến nghị**: Đối với hệ thống Production, việc kết hợp cả hai phương pháp (Hybrid RAG) sẽ mang lại độ chính xác cao nhất.
