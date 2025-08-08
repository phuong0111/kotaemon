## 1. Thêm các mô hình AI của bạn

![tab tài nguyên](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/resources-tab.png)

- Công cụ này sử dụng Mô hình Ngôn ngữ Lớn (LLM) để thực hiện các nhiệm vụ khác nhau trong quy trình hỏi đáp.
  Vì vậy, bạn cần cung cấp cho ứng dụng quyền truy cập vào các LLM mà bạn muốn sử dụng.
- Bạn chỉ cần cung cấp ít nhất một mô hình. Tuy nhiên, khuyến nghị bạn nên bao gồm tất cả các LLM mà bạn có quyền truy cập, bạn sẽ có thể chuyển đổi giữa chúng khi sử dụng ứng dụng.

Để thêm mô hình:

1. Điều hướng đến tab `Tài nguyên`.
2. Chọn tab phụ `LLMs`.
3. Chọn tab phụ `Thêm`.
4. Cấu hình mô hình để thêm:
   - Đặt tên cho nó.
   - Chọn nhà cung cấp/nhà phát hành (ví dụ: `ChatOpenAI`).
   - Cung cấp các thông số kỹ thuật.
   - (Tùy chọn) Đặt mô hình làm mặc định.
5. Nhấp `Thêm` để thêm mô hình.
6. Chọn tab phụ `Mô hình nhúng` và lặp lại bước 3 đến 5 để thêm mô hình nhúng.

<details markdown>

<summary>(Tùy chọn) Cấu hình mô hình qua tệp .env</summary>

Thay vào đó, bạn có thể cấu hình các mô hình qua tệp `.env` với thông tin cần thiết để kết nối với các LLM. Tệp này nằm trong thư mục của ứng dụng. Nếu bạn không thấy nó, bạn có thể tạo một tệp.

Hiện tại, các nhà cung cấp sau được hỗ trợ:

### OpenAI

Trong tệp `.env`, đặt biến `OPENAI_API_KEY` với khóa API OpenAI của bạn để cho phép truy cập vào các mô hình của OpenAI. Có các biến khác có thể được sửa đổi, vui lòng chỉnh sửa chúng để phù hợp với trường hợp của bạn. Nếu không, tham số mặc định sẽ hoạt động cho hầu hết mọi người.

```shell
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=<khóa API OpenAI của bạn ở đây>
OPENAI_CHAT_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDINGS_MODEL=text-embedding-ada-002
```

### Azure OpenAI

Đối với các mô hình OpenAI qua nền tảng Azure, bạn cần cung cấp endpoint Azure và khóa API của bạn. Bạn cũng có thể cần cung cấp tên triển khai của mình cho mô hình chat và mô hình nhúng tùy thuộc vào cách bạn thiết lập triển khai Azure.

```shell
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
OPENAI_API_VERSION=2024-02-15-preview # có thể khác với bạn
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo # thay đổi thành tên triển khai của bạn
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002 # thay đổi thành tên triển khai của bạn
```

### Mô hình cục bộ

Ưu điểm:

- Quyền riêng tư. Tài liệu của bạn sẽ được lưu trữ và xử lý cục bộ.
- Sự lựa chọn. Có nhiều LLM đa dạng về kích thước, lĩnh vực, ngôn ngữ để lựa chọn.
- Chi phí. Miễn phí.

Nhược điểm:

- Chất lượng. Mô hình cục bộ nhỏ hơn nhiều và do đó có chất lượng sinh ra thấp hơn các API trả phí.
- Tốc độ. Mô hình cục bộ được triển khai bằng máy của bạn nên tốc độ xử lý bị giới hạn bởi phần cứng của bạn.

#### Tìm và tải xuống LLM

Bạn có thể tìm kiếm và tải xuống LLM để chạy cục bộ từ [Hugging Face Hub](https://huggingface.co/models). Hiện tại, các định dạng mô hình này được hỗ trợ:

- GGUF

Bạn nên chọn một mô hình có kích thước nhỏ hơn bộ nhớ thiết bị của bạn và nên dành khoảng 2 GB. Ví dụ, nếu bạn có tổng cộng 16 GB RAM, trong đó 12 GB có sẵn, thì bạn nên chọn một mô hình chiếm tối đa 10 GB RAM. Mô hình lớn hơn có xu hướng tạo ra kết quả tốt hơn nhưng cũng mất nhiều thời gian xử lý hơn.

Dưới đây là một số khuyến nghị và kích thước của chúng trong bộ nhớ:

- [Qwen1.5-1.8B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q8_0.gguf?download=true):
  khoảng 2 GB

#### Kích hoạt mô hình cục bộ

Để thêm mô hình cục bộ vào bộ mô hình, đặt biến `LOCAL_MODEL` trong tệp `.env` thành đường dẫn của tệp mô hình.

```shell
LOCAL_MODEL=<đường dẫn đầy đủ đến tệp mô hình của bạn>
```

Đây là cách lấy đường dẫn đầy đủ của tệp mô hình:

- Trên Windows 11: nhấp chuột phải vào tệp và chọn `Copy as Path`.
</details>

## 2. Tải lên tài liệu của bạn

![tab chỉ mục tệp](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/file-index-tab.png)

Để thực hiện hỏi đáp trên tài liệu của bạn, trước tiên bạn cần tải chúng lên ứng dụng.
Điều hướng đến tab `Chỉ mục tệp` và bạn sẽ thấy 2 phần:

1. Tải lên tệp:
   - Kéo và thả tệp của bạn vào giao diện người dùng hoặc chọn nó từ hệ thống tệp của bạn.
     Sau đó nhấp `Tải lên và lập chỉ mục`.
   - Ứng dụng sẽ mất một thời gian để xử lý tệp và hiển thị thông báo khi hoàn thành.
2. Danh sách tệp:
   - Phần này hiển thị danh sách các tệp đã được tải lên ứng dụng và cho phép người dùng xóa chúng.

## 3. Trò chuyện với tài liệu của bạn

![tab trò chuyện](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/chat-tab.png)

Bây giờ điều hướng trở lại tab `Trò chuyện`. Tab trò chuyện được chia thành 3 vùng:

1. Bảng cài đặt cuộc trò chuyện
   - Ở đây bạn có thể chọn, tạo, đổi tên và xóa cuộc trò chuyện.
     - Theo mặc định, một cuộc trò chuyện mới được tạo tự động nếu không có cuộc trò chuyện nào được chọn.
   - Bên dưới đó bạn có chỉ mục tệp, nơi bạn có thể chọn tắt, chọn tất cả tệp, hoặc chọn tệp nào để truy xuất tham chiếu.
     - Nếu bạn chọn "Tắt", không có tệp nào sẽ được xem xét làm ngữ cảnh trong cuộc trò chuyện.
     - Nếu bạn chọn "Tìm kiếm tất cả", tất cả các tệp sẽ được xem xét trong cuộc trò chuyện.
     - Nếu bạn chọn "Chọn", một menu thả xuống sẽ xuất hiện để bạn chọn các tệp được xem xét trong cuộc trò chuyện. Nếu không có tệp nào được chọn, thì không có tệp nào sẽ được xem xét trong cuộc trò chuyện.
2. Bảng trò chuyện
   - Đây là nơi bạn có thể trò chuyện với chatbot.
3. Bảng thông tin

![bảng thông tin](https://raw.githubusercontent.com/Cinnamon/kotaemon/develop/docs/images/info-panel-scores.png)

- Thông tin hỗ trợ như bằng chứng được truy xuất và tham chiếu sẽ được hiển thị ở đây.
- Trích dẫn trực tiếp cho câu trả lời được tạo bởi LLM được làm nổi bật.
- Điểm tin cậy của câu trả lời và điểm liên quan của bằng chứng được hiển thị để đánh giá nhanh chất lượng của câu trả lời và nội dung được truy xuất.

- Ý nghĩa của điểm số được hiển thị:
  - **Độ tin cậy của câu trả lời**: mức độ tin cậy của câu trả lời từ mô hình LLM.
  - **Điểm liên quan**: điểm liên quan tổng thể giữa bằng chứng và câu hỏi của người dùng.
  - **Điểm Vectorstore**: điểm liên quan từ tính toán độ tương tự nhúng vector (hiển thị `tìm kiếm toàn văn` nếu được truy xuất từ cơ sở dữ liệu tìm kiếm toàn văn).
  - **Điểm liên quan LLM**: điểm liên quan từ mô hình LLM (đánh giá sự liên quan giữa câu hỏi và bằng chứng bằng prompt cụ thể).
  - **Điểm sắp xếp lại**: điểm liên quan từ [mô hình sắp xếp lại](https://cohere.com/rerank) của Cohere.

Nói chung, chất lượng điểm số là `Điểm liên quan LLM` > `Điểm sắp xếp lại` > `Điểm Vector`.
Theo mặc định, điểm liên quan tổng thể được lấy trực tiếp từ điểm liên quan LLM. Bằng chứng được sắp xếp dựa trên điểm liên quan tổng thể của chúng và liệu chúng có trích dẫn hay không.