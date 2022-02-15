// Tham khảo `word2vec.c`, `dict2vec.c`, `pword2vec.c`

// Đầu tiên hãy đọc docs/word2vec.md để hiểu cơ bản về SGNS: skip gram negative sampling

const std = @import("std");
const util = @import("util.zig");
const fastmath = @import("fastmath.zig");

// Các tham số dùng để khởi tạo và huấn luyện neural network (nn) một lớp ẩn
pub const params: struct {
    //
    // Các tham số của thuật toán cơ bản
    //
    vec_size: u16 = 256, // số lượng chiều của word vector, để là bội số cùa 8 để tiện cho simd
    epoch: u8 = 8, // số lần lặp huấn luyện, cần đủ lớn để khớp với dữ liệu
    threads_num: u8 = 4, // số threads chạy song song để tăng tốc huấn luyện

    negative: u8 = 10, // số mẫu negative sẽ lấy để huấn luyện

    starting_alpha: f32 = 0.025, // tốc độ học bắt đầu lớn
    min_alpha: f32 = 0.025 * 0.0001, // càng về sau càng giảm dần nhưng ko được quá bé

    // Các tham số dành cho phần mẹo mực hogbatch #nhà_làm
    // Phần này adhoc, cần kiểm tra lại độ tin cậy
    // Hiện tại nếu để `mini_batch` từ 8 trở lên sẽ làm hỏng việc huấn luyện
    mini_batch: u8 = 3, // số mẫu gộp chung để train, ko để quá 4, set = 0 để tắt
} = .{};

// Đầu vào của nn là one-hot vectors, mỗi one-hot đại diện cho 1 từ trong vocab
// one-hot nghĩa là chỉ 1 giá trị x_i tương ứng với index của từ = 1, còn lại là 0
//
// Cấu trúc nn đơn giản gồm 2 lớp, lớp ẩn là ma trận vocab_count x vec_size để
// map một từ trong vocab (one-hot vector) thành embeded vector
pub var hidden_layer: []f32 = undefined;

// Đầu ra là ma trận vec_size x vocab_count để map embeded vector thành
// một one-hot vector (trường hợp lý tưởng) tương ứng với từ được dự đoán
//
// Thực tế đầu ra là vector có vocab_size giá trị trong khoảng 0 -> 1
// tương ứng với khả năng từ này là từ được dự đoán.
pub var output_layer: []f32 = undefined;

// Bằng việc huấn luyện cho nn cho 1 từ đầu vào, dự đoán từ đầu ra (cặp huấn luyện input - output),
// ta cùng làm hai việc một lúc là xây dựng ma trận hidden_layer, và output_layer
// hidden_layer hay còn gọi là embed_layer hay embed_matrix là các vector đại diện của từ
// trong một không gian ít chiều hơn (256 chiều)
// thay vì không gian vào chục ngàn chiều của one-hot vectors.

// Kết quả được quan tâm ở đây là hidden_layer (embed_matrix) và output_layer quẳng đi.
// Một mô hình rất đơn giản nhưng hoạt động hiệu quả.
//
// Note: dùng thuật ngữ từ trong bộ từ vựng cho dễ hiểu chứ thực tế nn không quan tâm
// nó làm việc trên các vector đầu vào tương ứng với 1 type của text corpus.
// Khi làm việc với copus thì hiểu đơn giản nhất text là một chuỗi liên tục của các token
// Các token giống nhau được gọi là 1 type. Token có thể là từ, là ký tự hoặc bất kỳ
// cách phân tách nào phù hợp với bài toán bạn đang xử lý.

pub fn deinit() void {
    util.allocator.free(hidden_layer);
    util.allocator.free(output_layer);
}

pub fn init(vocab_count: usize) !void {
    // Alloc hidden layer and output layer of the network
    hidden_layer = try util.allocator.alloc(f32, vocab_count * params.vec_size);
    output_layer = try util.allocator.alloc(f32, vocab_count * params.vec_size);

    // Randomly initialize the weights for the hidden layer (word vector layer).
    const random = prng.random();
    const vec_size_f32 = @intToFloat(f32, params.vec_size);
    for (hidden_layer) |*elem| elem.* = (random.float(f32) - 0.5) / vec_size_f32;

    // Set all of the weights in the output layer to 0
    std.mem.set(f32, output_layer, 0);
}

// Các hàm tiện ích phục vụ việc huấn luyện
pub inline fn getVector(token: u16, matrix: []f32) []f32 {
    const start: usize = @intCast(usize, token) * params.vec_size;
    return matrix[start .. start + params.vec_size];
}

var prev_percentage: usize = 0;
fn showProgress(actual: f32, total: f32) void {
    const n = @floatToInt(usize, actual);
    const curr_percentage = n * 100 / @floatToInt(usize, total);

    if (curr_percentage == 100 or curr_percentage - prev_percentage >= 3) {
        std.debug.print("\n * {d:3}% progress, {d:4}k pairs trained,  {d:.5} learning rate", .{ curr_percentage, n / 1000, alpha });
        prev_percentage = curr_percentage;
    }
}

// Phần code chính của việc huấn luyện
// Các biến dưới đây khai báo toàn cục để dùng chung cho các train_thread
// Đỡ công truyền qua params, đơn giản vậy thôi.
//
var alpha: f32 = undefined; // tốc độ học đang được sử dụng
var pairs_count_actual: f32 = undefined; // số cặp input-output đã được huấn luyện
var total_training_pairs: f32 = undefined; // tổng số cặp cần được huấn luyện
// việc học kết thúc khi pairs_count_actual == total_training_pairs

// Bộ đệm để tích lũy gradients cho phần hồi quy,
var gradients_buffer: [params.vec_size * params.threads_num]f32 = undefined;
var prng = std.rand.DefaultPrng.init(123);

//
// Bắt đầu việc huấn luyện bằng cách bón các cặp input-output và bảng lấy mẫu sampling_table
// cho thuật toán huấn luyện.
//
// - input-output là để trích ra hidden_vector và output_vector tương ứng với từ đầu vào và
//   từ đầu ra thông qua hàm tiện ích getVector().
//
// - sampling_table là để trích ra params.negative mẫu để huấn luyện tiếp
//
// hàm train() là bước khởi tạo các train_thread và phân bổ dữ liệu đồng đều
// và không trùng lặp cho từng train_thread
//
pub fn train(inputs_outputs: [][2]u16, sampling_table: []u16) !void {
    // Khởi tạo các biến dùng chung cho các train_thread
    pairs_count_actual = 0;
    total_training_pairs = @intToFloat(f32, params.epoch * inputs_outputs.len + 1);
    alpha = params.starting_alpha;

    // tạo mảng threads để lưu dấu các train_thread đang chạy
    var threads_buffer: [params.threads_num]std.Thread = undefined;
    const threads = threads_buffer[0..];

    const pairs_per_thread = (inputs_outputs.len / params.threads_num) + 1;

    for (threads) |*thread, i| {
        // khởi tạo begin và end của đoạn dữ liệu sẽ phân cho train_thread i
        const begin = pairs_per_thread * i;
        var end = begin + pairs_per_thread;
        if (end > inputs_outputs.len) end = inputs_outputs.len;

        // Đoạn dữ liệu riêng của train thread chuẩn bị khởi tạo
        var thread_pairs = inputs_outputs[begin..end];

        // Quậy ngẫu nhiên dữ liệu đầu vào ở từng thread
        const random = prng.random();
        random.shuffle([2]u16, thread_pairs);

        if (params.mini_batch > 0) {
            // Sắp xếp lại dữ liệu sao cho các input giống nhau sẽ đứng liền nhau
            // để khi mini-batching dễ dàng load các pairs có input giống nhau
            std.sort.sort([2]u16, thread_pairs, {}, orderPairByInputToken);
        }

        // Mỗi train_thread được khởi tạo và phân cho một đoạn dữ liệu riêng
        thread.* = try std.Thread.spawn(.{}, trainThread, .{ thread_pairs, sampling_table, i });
    }

    // chờ cho các train_thread cùng nhau kết thúc
    for (threads) |*thread| thread.join();
}
fn orderPairByInputToken(context: void, a: [2]u16, b: [2]u16) bool {
    _ = context;
    return a[0] < b[0];
}

// Đây mới là phần code chính của thuật toán huấn luyện
fn trainThread(inputs_outputs: [][2]u16, sampling_table: []const u16, thread_num: usize) void {
    // training epoch times
    var iter: u8 = 0;

    // lặp lại epoch lần với tập mẫu huấn luyện đầu vào
    while (iter < params.epoch) {
        //
        iter += 1;
        if (thread_num == params.threads_num - 1)
            std.debug.print("\n\nTraining epoc {}: ", .{iter});

        var last_pair_index: usize = 0;
        var last_batch_index: usize = 0;
        var current_input: u16 = inputs_outputs[0][0];
        var pair_index: usize = 0;

        while (pair_index < inputs_outputs.len) : (pair_index += 1) {
            const pair = inputs_outputs[pair_index];
            const delta = pair_index - last_pair_index;
            // Với mỗi 10k mẫu được huấn luyện thì:
            // - show tiến độ huấn luyện
            // - giảm dần tốc độ học
            if (delta > 10_000) {

                // update counters
                last_pair_index = pair_index;
                pairs_count_actual += @intToFloat(f32, delta);

                // The more pairs learnt the slower learning rate
                // Để các tham số hội tụ dần về điểm ổn định
                alpha = params.starting_alpha * (1 - pairs_count_actual / total_training_pairs);
                if (alpha < params.min_alpha) alpha = params.min_alpha;

                // Show info while waiting the model to be trained
                if (thread_num == params.threads_num - 1)
                    showProgress(pairs_count_actual, total_training_pairs);
            }

            // Đoạn này có thể lựa chọn 1/ hoặc 2/
            if (params.mini_batch == 0) {
                // 1/ huấn luyện từng mẫu một như giải thuật `word2vec` nguyên bản:
                // với mỗi mẫu huấn luyện tiến hành skip gram negative sampling training
                // đọc đến đây và đọc tiếp sgnsTraining() là đủ để hiểu giải thuật
                sgnsTraining(pair[0], pair[1], sampling_table, thread_num);
            } else {
                // 2/ mini-batching với input giống nhau
                // phần này áp dụng mẹo mực để giảm khối lượng tính toán nên sẽ rối mắt hơn
                // chọn mini-batching + negative sample sharing để tập trung tính toán
                // cho một phần nhỏ của mạng nơ-ron và tiết kiệm chi phí vận chuyển bộ nhớ
                // Ở đây có 1 sự cải tiến nhẹ so với bản mini-batching của `pword2vec.c` là
                // gộp các pairs có input giống nhau vào cùng 1 batch.
                if (pair_index - last_batch_index > params.mini_batch or
                    pair[0] != current_input)
                {
                    const batch = inputs_outputs[last_batch_index..pair_index];
                    batchTraining(batch, sampling_table, thread_num);
                    // reset batch
                    last_batch_index = pair_index;
                    current_input = pair[0];
                }
            }
        } // Hết inputs_outputs pairs
        if (params.mini_batch != 0) {
            // Train nốt cặp cuối là xong
            const last_pair = inputs_outputs[pair_index - 1];
            sgnsTraining(last_pair[0], last_pair[1], sampling_table, thread_num);
        }
    } // Hết training epoc

}

// thuật toán huấn luyện skip-gram negative-sampling với từng mẫu đầu vào
inline fn sgnsTraining(
    input_token: u16,
    positive_token: u16,
    sampling_table: []const u16,
    thread_num: usize,
) void {
    // Reset gradients array
    const begin = thread_num * params.vec_size;
    const gradients = gradients_buffer[begin .. begin + params.vec_size];
    std.mem.set(f32, gradients, 0);

    var hidden_vec = getVector(input_token, hidden_layer);
    var output_vec = getVector(positive_token, output_layer);

    // update gradients and output vector for positive token
    updateGradientsAndOutputVec(hidden_vec, output_vec, 1, gradients);

    // then update gradients and output vectors for selected negative_tokens
    var k: u8 = 0;
    while (k < params.negative) : (k += 1) {
        // Get a random token from sampling table
        const rand = prng.random().uintAtMost(usize, sampling_table.len - 1);
        const negative_token = sampling_table[rand];

        const valid_negative_token = (negative_token != positive_token);
        if (valid_negative_token) {
            output_vec = getVector(negative_token, output_layer);
            updateGradientsAndOutputVec(hidden_vec, output_vec, 0, gradients);
        }
    }

    // Once the hidden layer gradients for the negative samples plus the
    // positive sample have been accumulated, update the hidden layer weights.
    // Note that we do not average the gradient before applying it.
    for (hidden_vec) |*node, index| node.* += gradients[index];
}

//
inline fn updateGradientsAndOutputVec(
    hidden_vec: []f32,
    output_vec: []f32,
    desired_output: f32,
    gradients: []f32,
) void {
    const model_output = fastmath.sigmoid(fastmath.dotProduct(hidden_vec, output_vec));
    const loss = (desired_output - model_output) * alpha;
    fastmath.vecMulAdd(output_vec, loss, gradients); // output_vec * loss then add to gradients
    fastmath.vecMulAdd(hidden_vec, loss, output_vec); // hidden_vec * loss then add to output_vec
}
//
// Hết phần cơ bản của giải thuật gốc `word2vec.c`
// - - - - - - - - - - - - - - - - - - - - - - - -

// Đến mẹo mực để giảm khối lượng tính toán
// Huấn luyện mini-batching negative sample sharing
const Vector = std.meta.Vector;
fn batchTraining(batch: []const [2]u16, sampling_table: []const u16, thread_num: usize) void {
    //
    var begin = thread_num * params.vec_size;
    const gradients = gradients_buffer[begin .. begin + params.vec_size];

    // hidden_vec dùng chung
    const input_token = batch[0][0];
    var hidden_vec = getVector(input_token, hidden_layer);

    var tokens: [params.negative]u16 = undefined;
    var dot_prods: [params.negative]f32 = undefined;
    // chọn ngẫu nhiên mẫu âm tính từ sampling_table, chừa slot 0 cho mẫu dương tính
    var k: usize = 0;
    while (k < params.negative) : (k += 1) {
        // Get random tokens from sampling table
        const rand = prng.random().uintAtMost(usize, sampling_table.len - 1);
        const token = sampling_table[rand];
        dot_prods[k] = 0; // fastmath.dotProduct(hidden_vec, getVector(token, output_layer));
        tokens[k] = token;
    }

    var i: usize = 0;
    while (i < hidden_vec.len) : (i += 4) {
        const a: Vector(4, f32) = .{
            hidden_vec[i],
            hidden_vec[i + 1],
            hidden_vec[i + 2],
            hidden_vec[i + 3],
        };

        k = 0;
        while (k < params.negative) : (k += 1) {
            const o = getVector(tokens[k], output_layer);
            const b: Vector(4, f32) = .{
                o[i],
                o[i + 1],
                o[i + 2],
                o[i + 3],
            };
            dot_prods[k] += @reduce(.Add, a * b);
        }
    }

    // Reset gradients
    std.mem.set(f32, gradients, 0);

    var positive_token: util.Token = undefined;
    var output_vec: []f32 = undefined;
    var loss: f32 = undefined;
    var c: usize = undefined;

    comptime var bi: usize = 0;
    inline while (bi < params.mini_batch) : (bi += 1) {
        positive_token = batch[bi][1];
        output_vec = getVector(positive_token, output_layer);
        updateGradientsAndOutputVec(hidden_vec, output_vec, 1, gradients);

        c = 0;
        while (c < params.negative) : (c += 1) {
            if (positive_token != tokens[c]) {
                output_vec = getVector(tokens[c], output_layer);
                loss = -fastmath.sigmoid(dot_prods[c]) * alpha;
                fastmath.vecMulAdd(output_vec, loss, gradients);
                fastmath.vecMulAdd(hidden_vec, loss, output_vec);
            }
        }
    } // for batch

    comptime var v: usize = 0;
    inline while (v < params.vec_size) : (v += 1) hidden_vec[v] += gradients[v];
}
