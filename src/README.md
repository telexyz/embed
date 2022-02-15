# Thuật ngữ

* `token` là `u16`, MỌI văn bản đầu vào đều được chuyển hóa thành chuỗi `tokens`

* `   tokens =    syllables  +    words  +   terms  +  others`
      65_536 =       17_880  +   28_522  +   19_134
     (66k)          (18k)       (29k)       (19k)

* `other` dùng để đánh dấu những `token` điều khiển như hết câu xuống dòng 
  hoặc chèn để giãn bớt khoảng cách, đánh dấu `OOS / OOV tokens`.

* `syllable` là `token` có value trong khoảng `u15` tương đương 1-1 với dạng text và 
  khả chuyển 1-1 bằng mã nguồn chương trình (rule-based, hard-coded)

* `meaningful tokens` là `syllables` và `pairs`

* `pair` được tạo mới bởi việc ghép cặp từ 2 `meaningful tokens` liền nhau.
  `pair` có 2 loại gồm: 

  - `word` dùng từ điển để so khớp, để đơn giản hóa ta chỉ dùng từ 2 âm tiết 
    nên `word` cũng là `pair`.
    Note: Phải thật chắc chắn thì mới chuyển hóa `syllable pair` thành `word`.
  
  - `term` là `pair` được gộp bằng thống kê tần xuất cùng xuất hiện (xem `word2phrase.c`)


* `bigram` là 2 `meaningful tokens` xuất hiện liền nhau trong văn bản đầu vào


# Modules

* sampling
  - input  `data/corpus.xyz.cdx`
           `data/dict_2syll.xyz.cdx`

  - output `data/vocab.txt`
           `data/inputs_outputs.txt`

* training
  - input  `data/vocab.txt`
           `data/inputs_outputs.txt`

  - output `data/vocab.vec`

* similar
  - input  `data/vocab.vec`
           danh sách các từ
  - output bảng n-best các từ gần nhau ngữ cảnh sử dụng (bao gồm ngữ nghĩa)

* analogy
  - input  `data/vocab.vec`
           1 cặp từ cho trước
  - output 1 cặp từ có nghĩa gần tương đương

Nhiệm vụ vụ thể của từng chương trình như sau:

## `sampling.zig`

Ghép âm tiết thành từ, rồi lấy mẫu training.

* get `syllables` from `.cdx` file

* dùng dict các từ 2 âm tiết để gộp `syllables2words` vì:
  - số lượng từ = 2 âm tiết chiếm khoảng 85% từ điển
  - các từ >= 3 âm tiết sẽ được covered trong `bigram`
  - dùng chung bảng `[u16, u16] => u16` và mảng `u16 => [u16, u16]` với `terming`
  => Những trường hợp gây nhập nhằng như `[s1, s2] [s2, s3]` không gộp !!!
     VD: rời bỏ cuộc sống => rời bỏ_cuộc cuộc_sống => không gộp.

* (optional) remove `stop-words`: remove `của` (in `của tôi`) but keep `của cải`

* gộp `tokens` thành `terms` (xem `terming.zig`)
  - chỉ gộp syllables
  - bỏ vào từ điển 2 âm tiết
  - sau đó dùng từ điển để pair :D

* sub-sampling


## `training.zig`

Huấn luyện mô hình Neural Network một lớp ẩn và ghi hidden_layer 
chính là representation vectors của từng word trong vocab ra file.

training sử dụng cấu trúc dữ liệu và thuật toán huấn luyện trong `model.zig`

### `model.zig`

Cấu trúc dữ liệu Neural Network và các tiện ích liên quan:
* one-hot vector
* hidden layer
* output layer
* training algorithm


## `similar.zig`

Dùng thuật toán đo khoảng cách từ `distance.c`


## `analogy.zig`

Tham khảo `word-analogy.c`