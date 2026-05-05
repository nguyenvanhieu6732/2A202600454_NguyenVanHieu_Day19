# Báo cáo Benchmark: GraphRAG vs Flat RAG

Báo cáo được tự động tạo bởi `src/evaluate.py`.

## 1. Bảng Thông số Benchmark

| Hệ thống | Accuracy | Avg Latency (Thời gian) | Avg Cost (Tokens / Truy vấn) |
|---|---|---|---|
| **Flat RAG** | **55.0%** | **2.21 giây** | 727 tokens |
| **Graph RAG** | **40.0%** | 3.94 giây | **598 tokens** |

## 2. Phân tích Các Chế độ Thất bại (Failure Modes) của GraphRAG

Phân tích sâu vào các câu trả lời thất bại của GraphRAG, có 3 Failure Modes chính:

### 2.1. Lỗ hổng Tri thức (Incomplete Graph / Missing Nodes)
- **Biểu hiện**: Hệ thống từ chối trả lời do không có thông tin (Ví dụ các câu hỏi về công ty bị thiếu trong đồ thị).
- **Nguyên nhân**: Quá trình index bị giới hạn ở 15 chunks để tiết kiệm thời gian, dẫn đến các thực thể ở cuối đoạn văn bản không được chèn vào Neo4j.
- **Giải pháp**: Bỏ giới hạn chunk và chạy index toàn bộ Corpus.

### 2.2. Nhầm lẫn khi tìm Seed Node (Entity Disambiguation / Vector Search Mismatch)
- **Biểu hiện**: Tìm sai hoặc thiếu seed node do vector embeddings không tương đồng hoàn toàn với keyword.
- **Giải pháp**: Tăng top_k trong vector search hoặc tăng số hop duyệt (BFS) lên 3-hop.

### 2.3. Mất mát Ngữ cảnh (Context/Metadata Loss)
- **Biểu hiện**: Hệ thống không trả lời được các câu hỏi chi tiết "tại sao", "như thế nào".
- **Nguyên nhân**: Cấu trúc Triples (Node-Edge-Node) làm rơi rụng nhiều chi tiết lịch sử, ngữ cảnh thời gian so với đoạn văn bản thô.
- **Giải pháp**: Bổ sung Node/Edge Properties hoặc dùng Hybrid RAG (kết hợp cả Graph và Vector văn bản).

## 3. Tổng kết
- **GraphRAG** tốn ít token truy vấn hơn do prompt ngắn gọn (chỉ chứa các triples).
- **Flat RAG** ổn định và dễ lấy nguyên bản dữ liệu thô.
- Sự kết hợp của cả hai (Hybrid RAG) sẽ là giải pháp tối ưu cho ứng dụng thực tế.
