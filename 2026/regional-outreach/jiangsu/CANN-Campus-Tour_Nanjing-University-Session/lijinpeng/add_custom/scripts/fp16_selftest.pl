#!/usr/bin/perl
# 离线复核 fp16_utils.h 中 float<->fp16 的转换逻辑（与头文件算法逐行等价移植）
use strict; use warnings;

sub float_to_fp16 {
    my ($value) = @_;
    my $bits = unpack("V", pack("f", $value));   # 小端 uint32 位模式
    my $sign = ($bits >> 16) & 0x8000;
    my $exp_field = ($bits >> 23) & 0xFF;
    my $mantissa = $bits & 0x7FFFFF;

    if ($exp_field == 0xFF) {
        return $mantissa == 0 ? ($sign | 0x7C00) : ($sign | 0x7E00);
    }

    my $exponent = $exp_field - 127 + 15;
    return $sign | 0x7C00 if $exponent >= 31;

    if ($exponent <= 0) {
        return $sign if $exponent < -10;
        $mantissa |= 0x800000;
        my $shift = 14 - $exponent;
        my $half_mantissa = $mantissa >> $shift;
        my $remainder = $mantissa & ((1 << $shift) - 1);
        my $halfway = 1 << ($shift - 1);
        $half_mantissa++ if ($remainder > $halfway) || ($remainder == $halfway && ($half_mantissa & 1));
        return $sign | $half_mantissa;
    }

    my $half = $sign | ($exponent << 10) | ($mantissa >> 13);
    my $remainder = $mantissa & 0x1FFF;
    $half++ if ($remainder > 0x1000) || ($remainder == 0x1000 && ($half & 1));
    return $half;
}

sub fp16_to_float {
    my ($value) = @_;
    my $sign = ($value & 0x8000) << 16;
    my $exponent = ($value >> 10) & 0x1F;
    my $mantissa = $value & 0x3FF;
    my $bits;
    if ($exponent == 0) {
        if ($mantissa == 0) { $bits = $sign; }
        else {
            my $shift = 0; my $m = $mantissa;
            while (($m & 0x400) == 0) { $shift++; $m <<= 1; }
            $m &= 0x3FF;
            $bits = $sign | ((113 - $shift) << 23) | ($m << 13);
        }
    } elsif ($exponent == 31) {
        $bits = $sign | 0x7F800000 | ($mantissa << 13);
    } else {
        $bits = $sign | (($exponent + 112) << 23) | ($mantissa << 13);
    }
    return unpack("f", pack("V", $bits & 0xFFFFFFFF));
}

# 期望值：{输入 float, 期望 fp16 位模式}
my @cases = (
    [0.0,        0x0000],
    [-0.0,       0x8000],
    [1.0,        0x3C00],
    [-1.0,       0xBC00],
    [2.0,        0x4000],
    [-2.0,       0xC000],
    [0.5,        0x3800],
    [65504.0,    0x7BFF],   # fp16 最大正规数
    [6.103515625e-05, 0x0400],  # 最小正规数 2^-14
    [5.960464477539063e-08, 0x0001],  # 最小次正规数 2^-24
    [1.0e-08,    0x0000],   # 下溢为 0
    [1.0e+06,    0x7C00],   # 上溢为 Inf
    [0.3333333333333333, 0x3555],
    [0.1,        0x2E66],
    [3.140625,   0x4248],
);

my $fail = 0;
for my $c (@cases) {
    my ($in, $expect) = @$c;
    my $got = float_to_fp16($in);
    my $ok = ($got == $expect);
    $fail++ unless $ok;
    printf("%-24s -> 0x%04X (expect 0x%04X) %s\n", $in, $got, $expect, $ok ? "OK" : "FAIL");
}

# 往返一致性：fp16 -> float -> fp16 必须稳定（排除 NaN/Inf）
for (my $h = 0; $h <= 0xFFFF; $h++) {
    my $exp = ($h >> 10) & 0x1F;
    next if $exp == 0x1F;                       # 跳过 Inf / NaN
    my $back = float_to_fp16(fp16_to_float($h));
    if ($back != $h) {
        printf("ROUNDTRIP FAIL: 0x%04X -> %s -> 0x%04X\n", $h, fp16_to_float($h), $back);
        $fail++;
        last if $fail > 5;
    }
}

# 重点抽查次正规区间的往返
for (my $h = 0x0001; $h <= 0x0400; $h++) {
    my $back = float_to_fp16(fp16_to_float($h));
    if ($back != $h) { printf("SUBNORMAL ROUNDTRIP FAIL: 0x%04X -> 0x%04X\n", $h, $back); $fail++; last if $fail > 5; }
}

print $fail == 0 ? "\nALL FP16 CHECKS PASSED\n" : "\n$fail CHECK(S) FAILED\n";
exit($fail == 0 ? 0 : 1);
