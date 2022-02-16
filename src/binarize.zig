// a port of `binarize.c`
// /* Copyright (c) 2019-present, All rights reserved.
//  * Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
//  *
//  * This file is part of the "Near-lossless Binarization of Word Embeddings"
//  * software (https://github.com/tca19/near-lossless-binarization).
//  *
//  * This program is free software: you can redistribute it and/or modify
//  * it under the terms of the GNU General Public License as published by
//  * the Free Software Foundation, either version 3 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU General Public License at the root of this repository for
//  * more details.
//  *
//  * You should have received a copy of the GNU General Public License
//  * along with this program. If not, see <http://www.gnu.org/licenses/>.
//  */

const std = @import("std");

const openblas = @cImport({
    @cInclude("cblas.h");
});

pub fn main() anyerror!void {
    //
    // https://github.com/xianyi/OpenBLAS/wiki/User-Manual
    var A: [6]f64 = .{ 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    var B: [6]f64 = .{ 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    var C: [9]f64 = .{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

    openblas.cblas_dgemm(openblas.CblasColMajor, openblas.CblasNoTrans, openblas.CblasTrans, 3, 3, 2, 1, A[0..], 3, B[0..], 3, 2, C[0..], 3);

    var i: usize = 0;
    while (i < 9) : (i += 1) {
        std.debug.print("{d} ", .{C[i]});
    }
    std.debug.print("\nRun code using openblas OK!", .{});
}
