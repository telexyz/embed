const std = @import("std");
const util = @import("util.zig");
const blas = @cImport({
    @cInclude("cblas.h");
});

pub var matrix: []f32 = undefined;
var text: []const u8 = undefined;

var vocab: std.StringHashMap(usize) = undefined;
var id2str: std.ArrayList([]const u8) = undefined;

pub var vocab_size: usize = undefined;
pub var vec_size: u16 = undefined;

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

pub fn deinit() void {
    util.allocator.free(matrix);
    util.allocator.free(text);

    vocab.deinit();
    id2str.deinit();
}

pub inline fn getVector(token: usize) []f32 {
    const begin = token * vec_size;
    return matrix[begin .. begin + vec_size];
}
pub inline fn getToken(word: []const u8) usize {
    return vocab.get(word).?;
}
pub inline fn getWord(token: usize) []const u8 {
    return id2str.items[token];
}

pub fn main() anyerror!void {
    try init("data/vocab.vec");
    defer deinit();
    binarize();
}

fn binarize() {

}

fn testBlas() void {
    // https://github.com/xianyi/blas/wiki/User-Manual
    var A: [6]f64 = .{ 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    var B: [6]f64 = .{ 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    var C: [9]f64 = .{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

    blas.cblas_dgemm(
        blas.CblasColMajor,
        blas.CblasNoTrans,
        blas.CblasTrans,
        3,
        3,
        2,
        1,
        A[0..],
        3,
        B[0..],
        3,
        2,
        C[0..],
        3,
    );

    var i: usize = 0;
    while (i < 9) : (i += 1) {
        std.debug.print("{d} ", .{C[i]});
    }
    std.debug.print("\nRun code using blas OK!", .{});
}
