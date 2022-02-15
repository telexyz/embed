const std = @import("std");
pub const allocator = std.heap.page_allocator;

pub const Syllable = @import("syllable.zig");

// Dùng để decode syllable_id từ base64
pub const SYLLABLE_HEADER_MIN_VALUE: u8 = 24; // Lấy từ format mặc định của xyz.cdx file
pub var base64_char_to_num = [_]Token{0xff} ** 256;

// Định nghĩa Token, Pair, Count
pub const Token = u16;
pub const TokenArray = std.ArrayList(Token);

pub const TokenPair = [2]Token;
pub const TokenPairArray = std.ArrayList(TokenPair);
pub const Count = u32;

// Vì còn dư 4 syllable_ids ở cuối là syllable_ids không hợp lệ
// Nên lấy chúng để định danh
pub const NON_SYLLABLE_TOKEN: Token = Syllable.MAXX_ID - 1; // token vô nghĩa
pub const END_SENTENCE_TOKEN = NON_SYLLABLE_TOKEN - 1; // và kết thúc câu.

// Paired tokens: Words, Terms ... còn khoảng 40k slots để chứa
pub const PAIR_TOKEN_BEGIN = Syllable.MAXX_ID; // 26250
pub const PAIR_TOKEN_END = 40000; //     40000*40000/(1024*1024*8)=191MB bitset
// const PAIR_TOKEN_END = 65535; //  65535*65535/(1024*1024*8)=512MB bitset
const PAIR_TOKEN_SIZE: usize = PAIR_TOKEN_END - PAIR_TOKEN_BEGIN + 1;

const TOKEN_PAIR_PADDING: u32 = PAIR_TOKEN_END + 1;
pub inline fn tokenPair(t1: Token, t2: Token) u32 {
    return t1 * TOKEN_PAIR_PADDING + t2;
}
pub const MAX_TOKEN_PAIR = tokenPair(PAIR_TOKEN_END, PAIR_TOKEN_END);
pub const MAX_SYLLABLE_PAIR = tokenPair(Syllable.MAXX_ID, Syllable.MAXX_ID);

//
pub var last_new_token = PAIR_TOKEN_BEGIN - 1;

var pairs_to_tokens = std.AutoHashMap(TokenPair, Token).init(allocator);
// Lợi dụng tính chất một pair chỉ gồm 2 tokens (u16) và
// pair_token được định danh liên tục từ Syllable.MAXX_ID
// ta dùng mảng cố định để map tokens_to_pairs
var tokens_to_pairs: []TokenPair = undefined;

pub fn init() !void {
    // init base64_char_to_num
    for (std.base64.standard_alphabet_chars) |c, i| {
        base64_char_to_num[c] = @intCast(Token, i);
    }
    tokens_to_pairs = try allocator.alloc(TokenPair, PAIR_TOKEN_SIZE);
    try Syllable.init();
}
pub fn deinit() void {
    pairs_to_tokens.deinit();
    allocator.free(tokens_to_pairs);
    Syllable.deinit();
}

pub fn vocabIsFull() bool {
    return last_new_token == PAIR_TOKEN_END;
}
pub fn isMeaningfulToken(t: Token) bool {
    return t >= Syllable.MAXX_ID or Syllable.isValid(t);
}
pub fn isPairToken(t: Token) bool {
    return t >= PAIR_TOKEN_BEGIN and t <= PAIR_TOKEN_END;
}

pub fn pairTokenExist(t1: Token, t2: Token) bool {
    return (pairs_to_tokens.contains(.{ t1, t2 }));
}
pub fn getPairToken(t1: Token, t2: Token) !Token {
    const pair = .{ t1, t2 };
    if (pairs_to_tokens.contains(pair)) {
        //
        return pairs_to_tokens.get(pair).?;
    } else {
        //
        last_new_token += 1;
        std.debug.assert(isPairToken(last_new_token));
        try pairs_to_tokens.put(pair, last_new_token);
        tokens_to_pairs[last_new_token - PAIR_TOKEN_BEGIN] = pair;
        return last_new_token;
    }
}
pub fn getTokenPair(t: Token) TokenPair {
    std.debug.assert(isPairToken(t));
    return tokens_to_pairs[t - PAIR_TOKEN_BEGIN];
}

pub fn printSeparateToken(t: Token) void {
    printToken(t, true);
}
fn printToken(t: Token, is_single: bool) void {
    if (isPairToken(t)) {
        const pair = getTokenPair(t);
        printToken(pair[0], false);
        printToken(pair[1], is_single);
    } else switch (t) {
        NON_SYLLABLE_TOKEN => std.debug.print("# ", .{}),
        END_SENTENCE_TOKEN => std.debug.print(".\n\n", .{}),
        else => if (is_single) {
            std.debug.print("{s} ", .{Syllable.id2str(t)});
        } else {
            std.debug.print("{s}_", .{Syllable.id2str(t)});
        },
    }
}
pub fn printPair(t1: Token, t2: Token) void {
    printToken(t1, false);
    printToken(t2, true);
}
pub fn printWordWithInfo(i: usize, t1: Token, t2: Token, key: u64) void {
    std.debug.print("found word[{}]: ", .{i});
    printPair(t1, t2);
    std.debug.print("=> pairing({}, {}) = {}\n", .{
        t1,
        t2,
        key,
    });
}

pub fn showTime(start_time: i64, comptime fmt_str: []const u8) i64 {
    const now = std.time.milliTimestamp();
    const duration = now - start_time;
    const minutes = @intToFloat(f32, duration) / 60000;
    std.debug.print("\n\n((( " ++ fmt_str ++ " Duration {d:.2} minutes )))\n\n", .{minutes});
    return now;
}

pub fn writeToken(writer: anytype, token: Token) !void {
    if (isPairToken(token)) {
        //
        const pair = getTokenPair(token);
        try writer.print("{s}_{s} ", .{ Syllable.id2str(pair[0]), Syllable.id2str(pair[1]) });
    } else {
        //
        try writer.print("{s} ", .{Syllable.id2str(token)});
    }
}
