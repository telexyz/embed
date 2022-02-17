const std = @import("std");
const util = @import("util.zig");
const model = @import("model.zig");

// Giữ lại giá trị text để dùng làm key_value của vocab là StringHashMap
var vocab_text: []const u8 = undefined;

// Map word in text format to id and count
const Vocab = std.StringHashMap(struct { id: u16, count: u24 = 0 });
const Id2Str = std.ArrayList([]const u8);

pub var vocab = Vocab.init(util.allocator);
pub var id2str = Id2Str.init(util.allocator);

pub var inputs_outputs: std.ArrayList([2]u16) = undefined;
pub var sampling_table: std.ArrayList(u16) = undefined;

//
pub fn deinit() void {
    vocab.deinit();
    model.deinit();
    inputs_outputs.deinit();
    sampling_table.deinit();
    id2str.deinit();
    util.allocator.free(vocab_text);
}

//
pub fn loadVocab(filename: []const u8) !void {
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    vocab_text = try file.reader().readAllAlloc(
        util.allocator,
        1024 * 1024 * 1, // max 1MB
    );

    var line_it = std.mem.split(u8, vocab_text, "\n");

    while (line_it.next()) |line| {
        if (line.len == 0) break;

        var it = std.mem.tokenize(u8, line, " ");

        const word = it.next().?;
        const count = try std.fmt.parseInt(u24, it.next().?, 10);

        try vocab.put(word, .{
            .id = @intCast(u16, id2str.items.len),
            .count = count,
        });

        try id2str.append(word);
        // std.debug.print("{s}\n", .{model.vocab.get(word)});
    }
}

//
pub fn loadInputsOutputs(filename: []const u8) !void {
    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    const input_bytes = try file.reader().readAllAlloc(
        util.allocator,
        1024 * 1024 * 150, // max 150MB
    );
    defer util.allocator.free(input_bytes);

    var line_it = std.mem.split(u8, input_bytes, "\n");

    inputs_outputs = try std.ArrayList([2]u16).initCapacity(util.allocator, input_bytes.len / 20);

    while (line_it.next()) |line| {
        if (line.len == 0) continue;

        var it = std.mem.tokenize(u8, line, " ");

        const input = vocab.get(it.next().?).?.id;
        const output = vocab.get(it.next().?).?.id;
        const count = try std.fmt.parseInt(u16, it.next().?, 10);

        var i: u16 = 0;
        while (i < count) : (i += 1)
            try inputs_outputs.append(.{ input, output });
    }
}

//
pub fn initSamplingTable() !void {
    std.debug.print("\n\nInit sampling table ...\n", .{});

    sampling_table = try std.ArrayList(u16).initCapacity(util.allocator, vocab.count() * 200);

    var it = vocab.valueIterator();

    while (it.next()) |word| {
        var i: u24 = 0;
        while (i < word.count) : (i += 1)
            try sampling_table.append(word.id);
    }

    std.debug.print("sampling_table size: {d}\n", .{sampling_table.items.len});
}

//
fn writeMatrixToFile(words: [][]const u8, matrix: []f32, filename: []const u8) !void {
    //
    var file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var wrt = std.io.bufferedWriter(file.writer());
    var writer = wrt.writer();

    try writer.print("{d} {d}", .{ words.len, matrix.len / words.len });

    var vec_start: usize = 0;

    for (words) |word| {
        const vec_end = vec_start + model.params.vec_size;
        try writer.print("\n{s}", .{word});
        for (matrix[vec_start..vec_end]) |v| try writer.print(" {d:0.6}", .{v});
        vec_start = vec_end;
    }

    try wrt.flush();
}
pub fn writeWordVectorsToFile(words: [][]const u8, filename: []const u8) !void {
    try writeMatrixToFile(words, model.hidden_layer, filename);
}

//
pub fn doTraining() anyerror!void {
    const start_time = std.time.milliTimestamp();

    //
    // step-1
    try loadVocab("data/vocab.txt"); // => model.vocab
    try loadInputsOutputs("data/inputs_outputs.txt"); // => inputs_outputs
    try initSamplingTable();

    const step1_time = util.showTime(start_time, "step-1: Data loaded!");

    //
    // step-2
    try model.init(vocab.count());

    try model.train(inputs_outputs.items, sampling_table.items);

    // _ = util.showTime(start_time, "debug");

    const step2_time = util.showTime(step1_time, "step-2: Training done!");

    const total_training_pairs = @intToFloat(f32, model.params.epoch * inputs_outputs.items.len);
    const duration = step2_time - start_time;
    const seconds = @intToFloat(f32, duration) / 1000;
    const pairs_per_sec = total_training_pairs / seconds;
    const pairs_per_thread_per_sec = pairs_per_sec / @intToFloat(f32, model.params.threads_num);

    // std.debug.print("{any}\n\n", .{model.params});

    std.debug.print("\n(( word2vec #nhà_làm speed = {d:0.2}k pairs/thread/second ))", .{
        pairs_per_thread_per_sec / 1000,
    });

    //
    // step-3
    try writeWordVectorsToFile(id2str.items, "data/vocab.vec");
    _ = util.showTime(step2_time, "step-3: Write word vectors to file done!");
}

pub fn main() anyerror!void {
    defer deinit();
    try doTraining();
}
