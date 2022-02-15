// Đọc corpus đầu vào dưới dạng mã hóa âm tiết *.xyz.cdx (mảng `tokens`)
//
// Đọc từ điển 2 âm tiết và tạo filter (`word_filter`)
//
// Tự động tạo ra các terms mới từ 2 âm tiết liền nhau bằng cách thống kê corpus
//
// Bổ xung terms mới chưa có trong từ điển vào filter
//
// Dùng filter để nhóm 2 âm tiết liền nhau thành từ,
// bỏ qua trường hợp nhập nhằng và tạo mảng mới `new_tokens`
//
// Bón `new_tokens` cho module `inandout.zig` để tạo ra file data/inputs_outputs.txt
// là đầu vào để huấn luyện neural network. Kết thúc quá trình lấy mẫu.

const std = @import("std");
const util = @import("util.zig");
const inandout = @import("inandout.zig");

pub const config = struct {
    pub const TOKEN_MIN_COUNT: u8 = 3;
    pub const TERM_THRESHOLD: f32 = 220;
    pub const PAIR_MAX_COUNT: u16 = 1000;
    pub const SAMPLING_RATE: f32 = 0.0001;
    pub const SAMPLING_WINDOW: u8 = 5;

    pub fn showDictIndex(i: usize) bool {
        return 0 <= i and i <= 20;
    }

    pub fn showTermIndex(i: usize) bool {
        return 500 <= i and i <= 1800;
    }

    pub fn showTokenIndex(i: usize) bool {
        return 0 <= i and i <= 600;
    }
};

// Input: file dưới định dạng xyz.cdx (ví dụ: `data/corpus.xyz.cdx`)
// Định dạng này sinh ra vởi `./telexify` thuộc `engine` repo
// File này chứa (<token_header><base64_syllable_id>?)+
//
// <token_header> là 1-byte
// * Nếu <token_header> >= SYLLABLE_HEADER_MIN_VALUE thì
// tiếp theo chắc chắn sẽ là <base64_syllable_id>
// * Nếu <token_header> == '\n' thì đó là kết thúc câu
//
// Còn lại thì là token không phải syllable, và không ghi thêm thông tin nào
// vì hiện tại chỉ làm việc với syllables nên quy các non_syllable_tokens thành
// util.NON_SYLLABLE_TOKEN hết!

fn loadTokensFromXyzCdxFile(filename: []const u8) !util.TokenArray {
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    const input_bytes = try file.reader().readAllAlloc(
        util.allocator,
        // 1024 * 1024 * 10, // max 10MB to fit corpus.xyz.cdx
        1024 * 1024 * 650, // max 650MB to fit combined.xyz.cdx
    );
    defer util.allocator.free(input_bytes);

    const n = input_bytes.len;
    var tokens = try util.TokenArray.initCapacity(util.allocator, n / 6);

    // Init tokens from bytes
    var in_blank_zone = true;
    var syll_id: util.Token = undefined;
    var i: usize = 0;
    var sents_count: usize = 0;

    var print_sent = true;
    var rng = std.rand.DefaultPrng.init(5);
    const random = rng.random();

    std.debug.print("\n", .{});
    std.log.info("Số bytes của {s}: {d}", .{ filename, n });

    if (print_sent) {
        std.debug.print("sent[{}]: ", .{sents_count});
    }

    while (i < n) : (i += 1) {
        var char = input_bytes[i];
        // Check if token is syllable
        if (char >= util.SYLLABLE_HEADER_MIN_VALUE) {
            // Decode syllable_id from base64
            char = input_bytes[i + 1];
            syll_id = util.base64_char_to_num[char] << 12;

            char = input_bytes[i + 2];
            syll_id |= util.base64_char_to_num[char] << 6;

            i += 3;
            char = input_bytes[i];
            syll_id |= util.base64_char_to_num[char];

            try tokens.append(syll_id);
            in_blank_zone = false;

            if (print_sent) util.printSeparateToken(syll_id);
            //
        } else if (char == '\n') { // end of sentence
            //
            if (print_sent) {
                std.debug.print("\n", .{});
            }
            //
            sents_count += 1;
            try tokens.append(util.END_SENTENCE_TOKEN);

            print_sent = random.int(u16) < 10;
            if (print_sent) {
                std.debug.print("sent[{}]: ", .{sents_count});
            }
            //
        } else if (!in_blank_zone) {
            if (print_sent) {
                std.debug.print("# ", .{});
            }

            try tokens.append(util.NON_SYLLABLE_TOKEN);
            in_blank_zone = true;
        }
    }

    try tokens.append(util.END_SENTENCE_TOKEN);

    std.log.info("Số lượng sentences: {}", .{sents_count});
    std.log.info("Số lượng tokens: {}", .{tokens.items.len});

    return tokens;
}

// Khởi tạo word_filter từ danh sách từ điển 2 âm tiết
fn initWordFilterFrom(vocab_filename: []const u8) !void {
    std.log.info("Init 2 syllable word filter:", .{});

    const tokens = try loadTokensFromXyzCdxFile(vocab_filename);
    defer tokens.deinit(); // dùng xong giải phóng luôn

    // Mỗi từ gồm <syllable_id_1, syllable_id_2, "\n">
    // nên chia 3 số lượng tokens sẽ được số lượng từ
    const words_number = (tokens.items.len / 3);

    // Thêm word keys
    std.log.info("\n\nThêm words:\n", .{});
    var token_index: usize = undefined;
    var i: usize = 0;
    while (i < words_number) : (i += 1) {
        token_index = i * 3;
        const token1 = tokens.items[token_index];
        const token2 = tokens.items[token_index + 1];
        const key = util.tokenPair(token1, token2);
        word_filter.set(key);
        if (config.showDictIndex(i)) {
            util.printWordWithInfo(i, token1, token2, key);
        }
    }
}

fn addNewTermsToWordFilter(new_terms: util.TokenPairArray) !void {
    // Thêm term keys
    std.log.info("\n\nThêm terms:\n", .{});
    var dup_count: usize = 0;

    for (new_terms.items) |pair, i| {
        const key = util.tokenPair(pair[0], pair[1]);

        if (word_filter.isSet(key)) {
            dup_count += 1;
            continue;
        }

        word_filter.set(key);
        if (config.showDictIndex(i)) util.printWordWithInfo(i, pair[0], pair[1], key);
    }
    std.debug.print("\n\nSố terms trùng với từ điển: {}\n\n", .{dup_count});
}

fn tokens2words(tokens: util.TokenArray) !util.TokenArray {
    var n = tokens.items.len - 1;
    var i: usize = 1;

    var prev_token = util.NON_SYLLABLE_TOKEN;
    var curr_token = util.NON_SYLLABLE_TOKEN;
    var next_token = tokens.items[0];

    var word_count: usize = 0;
    var uniq_word_count: usize = 0;

    var prev_pair_is_word: bool = false;
    var curr_pair_is_word: bool = false;
    var next_pair_is_word: bool = false;

    var curr_key: u64 = undefined;
    var next_key: u64 = undefined;

    var new_tokens = try util.TokenArray.initCapacity(util.allocator, tokens.items.len);

    std.debug.print("\n[ BEGIN TÌM TỪ HAI ÂM TIẾT ]\n\n", .{});
    while (i < n) : (i += 1) {
        prev_token = curr_token;
        curr_token = next_token;
        next_token = tokens.items[i];

        curr_key = next_key;
        next_key = util.tokenPair(curr_token, next_token);

        prev_pair_is_word = curr_pair_is_word;
        curr_pair_is_word = next_pair_is_word;
        next_pair_is_word = word_filter.isSet(next_key);

        const print_ = config.showTokenIndex(i);

        // Lưu mọi cặp từ có thể ghép (thà giết nhầm còn hơn bỏ sót)
        if (curr_pair_is_word and !util.vocabIsFull()) {
            word_count += 1;
            if (!util.pairTokenExist(prev_token, curr_token)) uniq_word_count += 1;
            // Lưu token mới được tạo
            try new_tokens.append(try util.getPairToken(prev_token, curr_token));
            if (print_) util.printPair(prev_token, curr_token);
        } else if (!next_pair_is_word) {
            try new_tokens.append(curr_token);
            if (print_) util.printSeparateToken(curr_token);
        }

        // // Phần code này ngược lại với vế trên (thà bỏ sót còn hơn giết nhầm)
        // const accept_curr_word = curr_pair_is_word and !prev_pair_is_word and
        //     !next_pair_is_word and !util.vocabIsFull();

        // // Phần logic if-else phía dưới được tạo ra từ việc thử sai nên đừng cố hiểu
        // if (accept_curr_word) {
        //     // Xác nhận cách nhóm cặp hiện tại là hợp lệ => Tạo token mới!
        //     word_count += 1;
        //     if (!util.pairTokenExist(prev_token, curr_token)) {
        //         uniq_word_count += 1;
        //     }
        //     // Lưu token mới được tạo
        //     try new_tokens.append(try util.getPairToken(prev_token, curr_token));
        //     if (print_) util.printPair(prev_token, curr_token);
        // } else if (!next_pair_is_word) {
        //     // Lưu curr_token
        //     try new_tokens.append(curr_token);
        //     if (print_) util.printSeparateToken(curr_token);
        // } else if (curr_pair_is_word and next_pair_is_word) {
        //     //
        //     if (!prev_pair_is_word) {
        //         // Lưu prev_token
        //         try new_tokens.append(prev_token);
        //         if (print_) util.printSeparateToken(prev_token);
        //     }
        //     // Lưu curr_token
        //     try new_tokens.append(curr_token);
        //     if (print_) util.printSeparateToken(curr_token);
        // }
    }
    std.debug.print("\n\n[ END TÌM TỪ HAI ÂM TIẾT ]\n\n", .{});

    var new_token_count = tokens.items.len - word_count;
    std.log.info("Total unique words found: {}", .{uniq_word_count});
    std.log.info("Total words found: {}", .{word_count});
    std.log.info("Total tokens: {}", .{new_token_count});
    std.log.info("words / tokens: {}%", .{word_count * 100 / new_token_count});

    std.debug.print("\n\n[ BEGIN RE-CHECK WITH NEW_TOKENS ]\n\n", .{});
    i = 0;
    while (config.showTokenIndex(i)) : (i += 1) {
        util.printSeparateToken(new_tokens.items[i]);
    }
    std.debug.print("\n\n[ END RE-CHECK WITH NEW_TOKENS ]\n\n", .{});

    new_token_count = new_tokens.items.len;
    std.log.info("Total unique words found: {}", .{uniq_word_count});
    std.log.info("Total words found: {}", .{word_count});
    std.log.info("Total tokens: {}", .{new_token_count});
    std.log.info("words / tokens: {}%", .{word_count * 100 / new_token_count});

    return new_tokens;
}

pub fn tokens2terms(tokens: util.TokenArray) !util.TokenPairArray {
    // Ported and modified from word2phrase.c

    const min_count: u32 = config.TOKEN_MIN_COUNT;
    const threshold: f32 = config.TERM_THRESHOLD;
    var train_words: u32 = 0;

    var tokens_counts = std.AutoHashMap(util.Token, util.Count).init(util.allocator);
    defer tokens_counts.deinit();

    var pairs_counts = std.AutoHashMap(util.TokenPair, util.Count).init(util.allocator);
    defer pairs_counts.deinit();

    // đếm tần suất xuất hiện của mọi tokens và pairs
    var prev_token = util.NON_SYLLABLE_TOKEN;
    var curr_token = util.NON_SYLLABLE_TOKEN;

    for (tokens.items) |next_token| {
        prev_token = curr_token;
        curr_token = next_token;
        if (util.isMeaningfulToken(curr_token)) {
            // count curr_token
            train_words += 1;
            const gop1 = try tokens_counts.getOrPutValue(curr_token, 0);
            gop1.value_ptr.* += 1;

            if (util.isMeaningfulToken(prev_token)) {
                // count curr pair
                const pair = .{ prev_token, curr_token };
                const gop2 = try pairs_counts.getOrPutValue(pair, 0);
                gop2.value_ptr.* += 1;
            }
        }
    }

    // Vòng lặp 2: phát hiện pair có thể gộp
    var it = pairs_counts.iterator();
    var new_terms = try util.TokenPairArray.initCapacity(util.allocator, 500);
    var term_count: usize = 0;
    var printed_count: usize = 0;

    while (it.next()) |kv| {
        const ab = kv.key_ptr.*;
        const pab = kv.value_ptr.*;
        if (pab < min_count) continue;

        const a = ab[0];
        if (!util.isMeaningfulToken(a)) continue;
        const pa = tokens_counts.get(a).?;
        // if (pa < min_count) continue;

        const b = ab[1];
        if (!util.isMeaningfulToken(b)) continue;
        const pb = tokens_counts.get(b).?;
        // if (pb < min_count) continue;

        const score = @intToFloat(f32, pab - min_count) / @intToFloat(f32, pa) / @intToFloat(f32, pb) * @intToFloat(f32, train_words);

        if (score > threshold) {
            if (config.showTermIndex(term_count)) {
                util.printPair(a, b);
                std.debug.print("  ", .{});
                printed_count += 1;
                if (printed_count % 12 == 0) std.debug.print("\n\n", .{});
            }
            term_count += 1;
            try new_terms.append(ab);
        }
    }

    std.debug.print("\n\nTotal new terms: {}", .{term_count});
    return new_terms;
}

// Lười truyền dữ liệu qua params nên để global vars ở đây
var word_filter: std.DynamicBitSet = undefined;

fn writeTokensToFile(tokens: util.TokenArray, filename: []const u8) !void {
    var file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var wrt = std.io.bufferedWriter(file.writer());
    var writer = wrt.writer();

    for (tokens.items) |token|
        try switch (token) {
            util.NON_SYLLABLE_TOKEN => {},
            util.END_SENTENCE_TOKEN => try writer.print("\n\n", .{}),
            else => util.writeToken(writer, token),
        };

    try wrt.flush();
}

pub fn doSampling() anyerror!void {
    // Init and deinit util data one time ONLY!
    try util.init();
    defer util.deinit();

    word_filter = try std.DynamicBitSet.initEmpty(util.allocator, util.MAX_SYLLABLE_PAIR);
    defer word_filter.deinit();

    // step-1
    const start_time = std.time.milliTimestamp();

    // const tokens: util.TokenArray = try loadTokensFromXyzCdxFile("/Users/t/repos/data/combined.xyz.cdx");
    const tokens: util.TokenArray = try loadTokensFromXyzCdxFile("data/corpus.xyz.cdx");
    defer tokens.deinit();

    const step1_time = util.showTime(start_time, "step-1: `data/corpus.xyz.cdx` loaded!");

    // step-2
    try initWordFilterFrom("data/dict_2syll.xyz.cdx");

    const step2_time = util.showTime(step1_time, "step-2: init vocab filter Done!");

    // step-3
    const new_terms = try tokens2terms(tokens);
    defer new_terms.deinit();

    try addNewTermsToWordFilter(new_terms);
    const new_tokens = try tokens2words(tokens);
    try writeTokensToFile(new_tokens, "data/corpus.txt");

    const step3_time = util.showTime(step2_time, "step-3: `tokens2terms` Done!");

    // step-4
    std.log.info("new tokens / origin tokens = {}%", .{new_tokens.items.len * 100 / tokens.items.len});

    const step_4 = util.showTime(step3_time, "step-4: `tokens2words` Done!");

    // step-5
    try inandout.initVocab(new_tokens, config.TOKEN_MIN_COUNT);
    defer inandout.deinit();

    try inandout.writeValidVocab("data/vocab.txt");
    try inandout.makeInputsOutputs(new_tokens);

    // Tạo dữ liệu huấn luyện => model.norm_inputs_outputs.items
    try inandout.makeTrainingPair(config.PAIR_MAX_COUNT);
    try inandout.writeInputsOutputs("data/inputs_outputs.txt");
    try inandout.writeAllInputsOutputs("data/ngram_inputs_outputs.txt");

    _ = util.showTime(step_4, "step-5: build vocab pruning and sub-sampling Done!");
    // Finally all done!
    _ = util.showTime(start_time, "FINISHED: Total");
}

pub fn main() anyerror!void {
    try doSampling();
}
