#pragma once

// Window function applied before the FFT (reduces spectral leakage).
enum class AudioWindowType {
    Hann = 0,       // default: good leakage suppression, music-friendly
    Hamming,
    Blackman,
    Rectangular     // no window (useful for comparison / debugging)
};
