// https://vi.wikipedia.org/wiki/Độ_tương_tự_cosin

const std = @import("std");
const util = @import("util.zig");
const wordvecs = @import("wordvecs.zig");

const Result = struct {
    token: usize,
    similarity: f32,
};

var results: []Result = undefined;

const spaces = "                          ";
pub fn nBestSimilar(word: []const u8, n_best: u8) ![]Result {
    const token = wordvecs.getToken(word);
    const token_vec = wordvecs.getVector(token);

    var other_token: usize = 0;

    while (other_token < wordvecs.vocab_size) : (other_token += 1) {
        const other_vec = wordvecs.getVector(other_token);
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

    std.sort.sort(Result, results, {}, orderBySimilarityDesc);

    var i: usize = 0;
    while (i < n_best) : (i += 1) {
        if (i % 5 == 0) std.debug.print("\n", .{});
        //
        const other_word = wordvecs.getWord(results[i].token);
        const tab = spaces[0 .. 15 - try std.unicode.utf8CountCodepoints(other_word)];
        std.debug.print("{d:.3} {s} {s} ", .{ results[i].similarity, other_word, tab });
    }
    std.debug.print("\n\n", .{});

    return results[0..n_best];
}

fn orderBySimilarityDesc(context: void, a: Result, b: Result) bool {
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

fn testRun(filename: []const u8) !void {
    try wordvecs.init(filename);
    defer wordvecs.deinit();

    results = try util.allocator.alloc(Result, wordvecs.vocab_size);
    defer util.allocator.free(results);

    _ = try nBestSimilar("trí_tuệ", 20);
    _ = try nBestSimilar("thực_hành", 20);
    _ = try nBestSimilar("hạnh_phúc", 20);
}

pub fn main() anyerror!void {
    std.debug.print("\n(( word2vec nhà làm ))\n", .{});
    try testRun("data/vocab.vec");
    //
    std.debug.print("\n(( word2vec nguyên bản ))\n", .{});
    try testRun("data/wordvec.out");
}
