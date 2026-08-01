// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Small pure vision helpers shared by the in-tree towers, split out so they can
// be unit-tested without a model or a GPU.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace vla {

// A tower that reads side*side*3 from a view needs the view to be exactly that
// size with real data, else it runs past the buffer.
inline bool view_is_side(const void * data, int w, int h, int64_t side) {
    return data != nullptr && (int64_t) w == side && (int64_t) h == side;
}

// IDEFICS3/SmolVLM pixel-shuffle (space-to-depth), c-innermost channel order.
// src [embed, n_patches] row-major (patch p, channel e) -> dst [embed*s^2, (grid/s)^2].
inline void pixel_shuffle_hf(const float * src, float * dst,
                             int64_t embed, int64_t grid, int64_t s) {
    const int64_t g2 = grid / s, c4 = embed * s * s;
    for (int64_t h2 = 0; h2 < g2; ++h2)
        for (int64_t w2 = 0; w2 < g2; ++w2) {
            const int64_t t = h2 * g2 + w2;
            for (int64_t hs = 0; hs < s; ++hs)
                for (int64_t ws = 0; ws < s; ++ws) {
                    const int64_t p    = (h2 * s + hs) * grid + (w2 * s + ws);
                    const int64_t base = (hs * s + ws) * embed;
                    std::memcpy(dst + t * c4 + base, src + p * embed,
                                (size_t) embed * sizeof(float));
                }
        }
}

// Bilinear upsampling on a CHW buffer stored position-major (pos, channel).
// src[K_in][channels] where K_in = g_in*g_in is a g_in×g_in spatial grid.
// dst[K_out][channels] where K_out = g_out*g_out is a g_out×g_out spatial grid.
// Uses align_corners=false coordinate mapping (PyTorch F.interpolate default).
inline void bilinear_upsample_2d(const float * src, float * dst,
                                  int64_t channels, int64_t g_in, int64_t g_out) {
    const float scale = float(g_in) / float(g_out);
    for (int64_t yo = 0; yo < g_out; ++yo) {
        for (int64_t xo = 0; xo < g_out; ++xo) {
            const float yf = (yo + 0.5f) * scale - 0.5f;
            const float xf = (xo + 0.5f) * scale - 0.5f;
            const int64_t y0 = std::max(int64_t(0), (int64_t) yf);
            const int64_t x0 = std::max(int64_t(0), (int64_t) xf);
            const int64_t y1 = std::min(y0 + 1, g_in - 1);
            const int64_t x1 = std::min(x0 + 1, g_in - 1);
            const float dy = yf - (float) y0;
            const float dx = xf - (float) x0;
            const float * p00 = src + (y0 * g_in + x0) * channels;
            const float * p01 = src + (y0 * g_in + x1) * channels;
            const float * p10 = src + (y1 * g_in + x0) * channels;
            const float * p11 = src + (y1 * g_in + x1) * channels;
            float * p_out = dst + (yo * g_out + xo) * channels;
            for (int64_t c = 0; c < channels; ++c) {
                const float v00 = p00[c], v01 = p01[c], v10 = p10[c], v11 = p11[c];
                const float top  = v00 + (v01 - v00) * dx;
                const float bot  = v10 + (v11 - v10) * dx;
                p_out[c] = top + (bot - top) * dy;
            }
        }
    }
}

}  // namespace vla
