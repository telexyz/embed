// Module tại input và output để train neural network
// Tham khảo phiên bản gốc đã được comment tại `word2vec.c`
// Các bước tiến hành như sau:
//
// 1/ Dùng new_tokens đã được ghép âm tiết thành từ để xây dựng vocab
//    và đếm train_words là total tokens có trong corpus
//
// 2/ Loại bỏ những từ không hợp lệ hoặc count < min_count (default là 5)
//
// 3/ Thống kê input-output training pairs và normalize pair count to 1-100
//
// 4/ Ghi training pairs + normalized counts ra `data/inputs_outputs.txt`
//    Ghi từ điển hợp lệ ra `data/vocab.txt` để tham khao cho biết

const std = @import("std");
const util = @import("util.zig");
const sampling = @import("sampling.zig");

pub const Word = struct {
    token: util.Token,
    count: u24 = 0,
    kept_prob: f16 = -1,
};

pub var vocab: []Word = undefined; // vocab[token] => coresponding word
pub var train_words: usize = 0; // tổng từ có nghĩa có trong corpus

// vocab hợp lệ, word's count >= min_count và word's token là có nghĩa
var valid_vocab = std.ArrayList(Word).init(util.allocator);

// Đếm cặp input-output để lấy mẫu huấn luyện
var inputs_outputs = std.AutoHashMap(util.TokenPair, u16).init(util.allocator);

const Pair = struct {
    input: util.Token,
    output: util.Token,
    count: u16,
};

// Input to train neural network
pub var norm_inputs_outputs = std.ArrayList(Pair).init(util.allocator);

//
pub fn initVocab(tokens: util.TokenArray, min_count: u24) !void {
    // Populate vocab array that map token to it's word,
    // increase word's count and then init discard probs
    train_words = tokens.items.len;
    vocab = try util.allocator.alloc(Word, util.last_new_token + 1);
    for (vocab) |*word, i| word.* = .{ .token = @intCast(util.Token, i) };

    for (tokens.items) |token| {
        if (util.isMeaningfulToken(token)) {
            // std.debug.print("{} ", .{token});
            vocab[token].count += 1;
        } else train_words -= 1;
    }

    const train_words_f64: f64 = @intToFloat(f64, train_words);
    var token: util.Token = 0;
    var valid_count: usize = 0;
    var discardable_count: usize = 0;

    while (token < vocab.len) : (token += 1) {
        if (!util.isMeaningfulToken(token)) continue;

        const count = vocab[token].count;
        if (count < min_count) continue;

        // Công thức tính kept probabilty từ `word2vec.c`:
        // p(w) = (sqrt(f(w) / 0.001) + 1) * (0.001 / f(w))
        const freq_div_sample = @intToFloat(f64, count) / train_words_f64 / sampling.config.SAMPLING_RATE;
        const prob = (@sqrt(freq_div_sample) + 1) / freq_div_sample;

        vocab[token].kept_prob = @floatCast(f16, prob);

        if (prob < 1) {
            discardable_count += 1;
            // if (prng.random().float(f64) < 0.3) {
            //     // chỉ in ra 1/3 số lượng từ hợp lệ
            //     std.debug.print(" (( {d} ", .{count});
            //     util.printSeparateToken(token);
            //     std.debug.print("{d}% )) ", .{@ceil(prob * 100)});
            //     if (discardable_count % 5 == 0) std.debug.print("\n\n", .{});
            // }
        }

        try valid_vocab.append(vocab[token]);
        valid_count += 1;
    }

    std.debug.print("\n\nTotal valid words: {}\nTotal discardable words: {}\n", .{
        valid_count,
        discardable_count,
    });
}

var prng = std.rand.DefaultPrng.init(0x87654321);
fn toBeKept(token: util.Token) bool {
    const kept_prob = vocab[token].kept_prob;
    if (kept_prob < 0) return false; // invalid token
    if (kept_prob > 1) return true;
    const rand = prng.random().float(f64);
    return kept_prob > rand;
}

inline fn countInputOutput(input: util.Token, output: util.Token) !void {
    const gop = try inputs_outputs.getOrPutValue(.{ input, output }, 0);
    gop.value_ptr.* += 1;
}
pub fn makeInputsOutputs(input_tokens: util.TokenArray) !void {
    const tokens = input_tokens.items;
    var sent_begin: usize = 0;
    for (tokens) |token, index| {
        //
        if (!util.isMeaningfulToken(token)) {
            if (token == util.END_SENTENCE_TOKEN) {
                sent_begin = index + 1;
            }
            continue;
        }

        var prev = if (index > 0) index - 1 else 0;

        var min_prev = if (index > sampling.config.SAMPLING_WINDOW)
            index - sampling.config.SAMPLING_WINDOW
        else
            0;

        if (min_prev < sent_begin) min_prev = sent_begin;

        while (prev > min_prev) : (prev -= 1) {
            //
            const prev_token = tokens[prev];
            if (prev_token != token and toBeKept(prev_token) and toBeKept(token)) {
                //
                try countInputOutput(prev_token, token);
                try countInputOutput(token, prev_token);
            }
        }
    }
}

pub fn makeTrainingPair(token_max_count: u24) !void {
    defer inputs_outputs.deinit();

    var it = inputs_outputs.iterator();
    var pairs = try util.allocator.alloc(Pair, inputs_outputs.count());
    defer util.allocator.free(pairs);

    var index: usize = 0;
    while (it.next()) |entry| {
        const count = entry.value_ptr.*;
        // if (count < model.params.min_count) continue;

        pairs[index] = .{
            .input = entry.key_ptr[0],
            .output = entry.key_ptr[1],
            .count = count,
        };
        index += 1;
    }

    std.sort.sort(Pair, pairs, {}, orderPairByCountDesc);
    //
    var uniq_pair_count: usize = 0;
    var pair_count: usize = 0;
    const max_count = pairs[0].count;

    var normalized_count: usize = 0;
    var normalized_uniq_count: usize = 0;

    for (pairs) |pair| {
        //
        var count: usize = pair.count;
        uniq_pair_count += 1;
        pair_count += count;

        if (max_count > token_max_count and token_max_count >= 200) {
            // nếu max_count < token_max_count (ko scale lên)
            // token_max_count < 200 thì k tiến hành làm tròn vì dữ liệu quá bé ko đủ để train
            const tmp = @intToFloat(f64, count * token_max_count) / @intToFloat(f64, max_count);
            count = @floatToInt(usize, @ceil(tmp));
        }

        normalized_count += count;
        normalized_uniq_count += 1;

        try norm_inputs_outputs.append(.{
            .input = pair.input,
            .output = pair.output,
            .count = @intCast(u16, count),
        });
        //
    }

    std.debug.print("\n\nTotal input-output pairs: {d}", .{pair_count});
    std.debug.print("\nTotal unique pairs: {d}", .{uniq_pair_count});
    std.debug.print("\nMax count: {d}\n", .{max_count});

    std.debug.print("\n\nNomalized input-output pairs: {d}", .{normalized_count});
    std.debug.print("\nNomalized unique pairs: {d}\n", .{normalized_uniq_count});
}

pub fn writeInputsOutputs(filename: []const u8) !void {
    var file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var wrt = std.io.bufferedWriter(file.writer());
    var writer = wrt.writer();

    std.sort.sort(Pair, norm_inputs_outputs.items, {}, orderPairByInputToken);

    for (norm_inputs_outputs.items) |pair| {
        try util.writeToken(writer, pair.input);
        try util.writeToken(writer, pair.output);
        try writer.print("{d}\n", .{pair.count});
    }

    try wrt.flush();
}

pub fn writeAllInputsOutputs(filename: []const u8) !void {
    var file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var wrt = std.io.bufferedWriter(file.writer());
    var writer = wrt.writer();

    std.sort.sort(Pair, norm_inputs_outputs.items, {}, orderPairByInputToken);

    for (norm_inputs_outputs.items) |pair| {
        var i: usize = 0;
        while (i < pair.count) : (i += 1) {
            try util.writeToken(writer, pair.input);
            try util.writeToken(writer, pair.output);
            try writer.print("\n", .{});
        }
    }

    try wrt.flush();
}

fn orderPairByCountDesc(context: void, a: Pair, b: Pair) bool {
    _ = context;
    return a.count > b.count;
}
fn orderPairByInputToken(context: void, a: Pair, b: Pair) bool {
    _ = context;
    if (a.input == b.input) return a.count > b.count;
    return a.input < b.input;
}
fn order_word_by_count_desc(context: void, a: Word, b: Word) bool {
    _ = context;
    return a.count > b.count;
}

pub fn writeValidVocab(filename: []const u8) !void {
    std.sort.sort(Word, valid_vocab.items, {}, order_word_by_count_desc);
    //
    var file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var wrt = std.io.bufferedWriter(file.writer());
    var writer = wrt.writer();

    for (valid_vocab.items) |word| {
        try util.writeToken(writer, word.token);
        try writer.print("{d}\n", .{word.count});
    }

    try wrt.flush();
}

pub fn deinit() void {
    util.allocator.free(vocab);
    valid_vocab.deinit();
    // inputs_outputs.deinit();
    norm_inputs_outputs.deinit();
}
