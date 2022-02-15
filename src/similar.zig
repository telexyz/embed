// https://vi.wikipedia.org/wiki/Độ_tương_tự_cosin

const std = @import("std");
const util = @import("util.zig");
const model = @import("model.zig");

const Result = struct {
    token: usize,
    similarity: f32,
};

var results: []Result = undefined;
var matrix: []f32 = undefined;
var text: []const u8 = undefined;

var vocab: std.StringHashMap(usize) = undefined;
var id2str: std.ArrayList([]const u8) = undefined;

var vocab_size: usize = undefined;
var vec_size: u16 = undefined;

pub fn init(filename: []const u8) !void {
    vocab = std.StringHashMap(usize).init(util.allocator);
    id2str = std.ArrayList([]const u8).init(util.allocator);

    var file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    text = try file.reader().readAllAlloc(
        util.allocator,
        1024 * 1024 * 200, // max 100MB
    );

    var line_it = std.mem.split(u8, text, "\n");
    var it = std.mem.tokenize(u8, line_it.next().?, " ");

    vocab_size = try std.fmt.parseInt(usize, it.next().?, 10);
    vec_size = try std.fmt.parseInt(u16, it.next().?, 10);

    results = try util.allocator.alloc(Result, vocab_size);
    matrix = try util.allocator.alloc(f32, vocab_size * vec_size);

    var index: usize = 0;

    while (line_it.next()) |line| {
        if (line.len == 0) break;

        it = std.mem.tokenize(u8, line, " ");

        const word = it.next().?;

        try vocab.put(word, @intCast(usize, id2str.items.len));
        try id2str.append(word);

        // std.debug.print("{d}\n", .{vocab.get(word)});

        while (it.next()) |v| {
            matrix[index] = try std.fmt.parseFloat(f32, v);
            index += 1;
        }
    }
}

fn deinit() void {
    util.allocator.free(results);
    util.allocator.free(matrix);
    util.allocator.free(text);

    vocab.deinit();
    id2str.deinit();
}

fn getWordVector(token: usize) []f32 {
    const begin = token * vec_size;
    return matrix[begin .. begin + vec_size];
}

const spaces = "                          ";
pub fn nBestSimilar(word: []const u8, n_best: u8) ![]Result {
    const token = vocab.get(word).?;
    const token_vec = getWordVector(token);

    var other_token: usize = 0;

    while (other_token < vocab_size) : (other_token += 1) {
        const other_vec = getWordVector(other_token);
        const sim = cosineSimilarity(token_vec, other_vec);

        // std.debug.print("\n{s} {s}: {d:.5}\n", .{
        //     id2str.items[token],
        //     id2str.items[other_token],
        //     sim,
        // });

        results[other_token] = .{
            .token = other_token,
            .similarity = sim,
        };
    }

    std.sort.sort(Result, results, {}, by_similarity_desc);

    var i: usize = 0;
    while (i < n_best) : (i += 1) {
        if (i % 5 == 0) std.debug.print("\n", .{});
        //
        const other_word = id2str.items[results[i].token];
        const tab = spaces[0 .. 15 - try std.unicode.utf8CountCodepoints(other_word)];
        std.debug.print("{d:.3} {s} {s} ", .{ results[i].similarity, other_word, tab });
    }
    std.debug.print("\n\n", .{});

    return results[0..n_best];
}

fn by_similarity_desc(context: void, a: Result, b: Result) bool {
    _ = context;
    return a.similarity > b.similarity;
}

inline fn cosineSimilarity(vec1: []f32, vec2: []f32) f32 {
    var ab: f32 = 0;
    var a: f32 = 0;
    var b: f32 = 0;

    var i: usize = 0;
    while (i < vec1.len) : (i += 1) {
        const x = vec1[i];
        const y = vec2[i];
        ab += x * y;
        a += x * x;
        b += y * y;
    }

    return ab / (@sqrt(a) * @sqrt(b));
}

pub fn main() anyerror!void {
    //
    try init("data/vocab.vec");
    std.debug.print("\n(( word2vec nhà làm ))\n", .{});

    _ = try nBestSimilar("trí_tuệ", 20);

    _ = try nBestSimilar("thực_hành", 20);

    _ = try nBestSimilar("hạnh_phúc", 20);

    deinit();

    try init("data/wordvec.out");
    std.debug.print("\n(( word2vec nguyên bản ))\n", .{});

    _ = try nBestSimilar("trí_tuệ", 20);

    _ = try nBestSimilar("thực_hành", 20);

    _ = try nBestSimilar("hạnh_phúc", 20);

    deinit();

    // try init("data/ngram2vec_sgns.input");
    // std.debug.print("\n(( ngram2vec ))\n", .{});

    // _ = try nBestSimilar("trí_tuệ", 20);

    // _ = try nBestSimilar("thực_hành", 20);

    // _ = try nBestSimilar("hạnh_phúc", 20);

    // deinit();
}
