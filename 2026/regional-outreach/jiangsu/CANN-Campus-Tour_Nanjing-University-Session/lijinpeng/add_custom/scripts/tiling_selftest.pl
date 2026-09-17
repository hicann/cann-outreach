#!/usr/bin/perl
# 离线复核 main.cpp 中 ComputeTiling 的切分正确性（逐行等价移植）与边界不变量：
#   1. 覆盖性：堆芯覆盖 [0, totalLength)，不遗漏、不越界
#   2. 对齐性：每核起始元素下标是 16（32Byte）的整数倍
#   3. 边界性：tileLength 在 [TILE_LENGTH_MIN, TILE_LENGTH_MAX] 且 <= blockLength
#   4. 循环性：核内 loopCount >= 1，且最后一个 tile 长度 >= 1
use strict; use warnings;

use constant ALIGN_ELEM        => 16;
use constant TILE_LENGTH_MIN   => 256;
use constant TILE_LENGTH_MAX   => 8192;
use constant TILE_NUM_TARGET   => 8;
use constant MIN_ELEMS_PER_CORE => 1024;

sub align_up16 { my ($v) = @_; return int(($v + ALIGN_ELEM - 1) / ALIGN_ELEM) * ALIGN_ELEM; }

sub compute_tiling {
    my ($total, $core_num) = @_;
    my $usable = int(($total + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    $usable = 1 if $usable == 0;
    my $block_dim = $core_num < $usable ? $core_num : $usable;
    $block_dim = 1 if $block_dim == 0;
    my $block_length = align_up16(int(($total + $block_dim - 1) / $block_dim));
    my $tile_length = align_up16(int($block_length / TILE_NUM_TARGET));
    $tile_length = TILE_LENGTH_MIN if $tile_length < TILE_LENGTH_MIN;
    $tile_length = TILE_LENGTH_MAX if $tile_length > TILE_LENGTH_MAX;
    $tile_length = $block_length if $tile_length > $block_length;
    my $tile_num = int(($block_length + $tile_length - 1) / $tile_length);
    return ($block_dim, $block_length, $tile_length, $tile_num);
}

my $fail = 0;
my @shapes = (
    # [N2, N1] 覆盖：极小、非 16 对齐、刚好对齐、对齐 +1、大 shape、非 2 次幂
    [1,1],[1,3],[1,4],[1,15],[1,16],[1,17],[1,256],[1,1024],[1,2048],[1,10000],
    [3,5],[7,13],[8,2048],[16,4096],[100,1000],[127,127],[1,65536],[2,65536],
    [255,4097],[1024,1024],
);
my @cores = (1,2,8,20,32);

for my $s (@shapes) {
    my ($n2, $n1) = @$s;
    my $total = $n2 * $n1;
    for my $core (@cores) {
        my ($block_dim, $block_length, $tile_length, $tile_num) = compute_tiling($total, $core);

        # 不变量 2/3/4
        if ($block_length % ALIGN_ELEM != 0) { printf("FAIL align blockLength total=%d core=%d\n", $total, $core); $fail++; }
        if ($tile_length % ALIGN_ELEM != 0)  { printf("FAIL align tileLength total=%d core=%d\n", $total, $core); $fail++; }
        if ($tile_length < TILE_LENGTH_MIN && $tile_length != $block_length) {
            printf("FAIL tile min total=%d core=%d tile=%d block=%d\n", $total, $core, $tile_length, $block_length); $fail++;
        }
        if ($tile_length > TILE_LENGTH_MAX) { printf("FAIL tile max total=%d core=%d\n", $total, $core); $fail++; }
        if ($tile_length > $block_length)   { printf("FAIL tile > block total=%d core=%d\n", $total, $core); $fail++; }
        if ($tile_num < 1)                  { printf("FAIL tileNum < 1 total=%d core=%d\n", $total, $core); $fail++; }

        # 不变量 1：堆芯覆盖
        my @covered = (0) x $total;
        my $written_max = 0;
        for my $c (0 .. $block_dim - 1) {
            my $start = $c * $block_length;
            if ($start % ALIGN_ELEM != 0) { printf("FAIL core start align c=%d total=%d\n", $c, $total); $fail++; }
            next if $start >= $total;
            my $remain = $total - $start;
            my $core_length = $remain < $block_length ? $remain : $block_length;
            my $loop = int(($core_length + $tile_length - 1) / $tile_length);
            if ($loop < 1) { printf("FAIL loopCount<1 c=%d total=%d core=%d\n", $c, $total, $core); $fail++; }
            my $sum = 0;
            for my $i (0 .. $loop - 1) {
                my $off = $i * $tile_length;
                my $len = $core_length - $off;
                $len = $tile_length if $len > $tile_length;
                last if $len <= 0;
                # 模拟 CopyOut 写回的范围
                for my $k (0 .. $len - 1) {
                    my $idx = $start + $off + $k;
                    if ($idx >= $total) { printf("FAIL out of range idx=%d total=%d\n", $idx, $total); $fail++; }
                    else { $covered[$idx]++; }
                }
                my $last = $start + $off + $len - 1;
                $written_max = $last if $last > $written_max;
                $sum += $len;
            }
            if ($sum != $core_length) { printf("FAIL core sum c=%d sum=%d expect=%d\n", $c, $sum, $core_length); $fail++; }
        }
        my $miss = 0; my $dup = 0;
        for my $i (0 .. $total - 1) { $miss++ if $covered[$i] == 0; $dup++ if $covered[$i] > 1; }
        if ($miss || $dup) { printf("FAIL coverage total=%d core=%d miss=%d dup=%d\n", $total, $core, $miss, $dup); $fail++; }
    }
}

# 边界条件：非对齐尾块触发 DataCopyPad 的场景必须存在（len != tileLength）
my ($bd, $bl, $tl, $tn) = compute_tiling(2048*8 + 5, 8);
print "sample: total=", 2048*8+5, " blockDim=$bd blockLength=$bl tileLength=$tl tileNum=$tn\n";

print $fail == 0 ? "ALL TILING INVARIANTS PASSED\n" : "$fail INVARIANT CHECK(S) FAILED\n";
exit($fail == 0 ? 0 : 1);
