// Small dynamic bitset over terminal indices (docs/optimizations.md O9).
#pragma once
#include <cstdint>
#include <vector>

namespace glcore {

struct TermSet {
    int k = 0;
    std::vector<uint64_t> w;  // ceil(k/64) words

    TermSet() = default;
    explicit TermSet(int k_) : k(k_), w((k_ + 63) / 64, 0) {}
    static TermSet all(int k_) {
        TermSet s(k_);
        for (int i = 0; i < k_; ++i) s.set(i);
        return s;
    }
    void set(int i) { w[i >> 6] |= (uint64_t(1) << (i & 63)); }
    void reset(int i) { w[i >> 6] &= ~(uint64_t(1) << (i & 63)); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    bool any() const { for (auto x : w) if (x) return true; return false; }
    int count() const { int c = 0; for (auto x : w) c += __builtin_popcountll(x); return c; }
    bool is_subset_of(const TermSet& o) const { for (size_t i = 0; i < w.size(); ++i) if (w[i] & ~o.w[i]) return false; return true; }
    TermSet minus(const TermSet& o) const { TermSet r(k); for (size_t i = 0; i < w.size(); ++i) r.w[i] = w[i] & ~o.w[i]; return r; }
    TermSet unite(const TermSet& o) const { TermSet r(k); for (size_t i = 0; i < w.size(); ++i) r.w[i] = w[i] | o.w[i]; return r; }
    bool operator==(const TermSet& o) const { return k == o.k && w == o.w; }
    bool operator!=(const TermSet& o) const { return !(*this == o); }
    std::vector<int> to_list() const { std::vector<int> r; for (int i = 0; i < k; ++i) if (test(i)) r.push_back(i); return r; }
};

}  // namespace glcore
