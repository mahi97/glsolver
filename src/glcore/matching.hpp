// Bipartite matching terminals -> pre-terminals [Lem 7.8], minimal Hall-deficient set [Lem 7.6].
#pragma once
#include <vector>

#include "graph.hpp"

namespace glcore {

// Returns for each terminal in S (default: all live terminals) the matched pre-terminal, or an empty
// vector if no matching saturating S exists. Hopcroft–Karp on the bipartite graph {t} x in-neighbours.
std::vector<int> saturating_matching(const Graph& g, const std::vector<int>& S);
std::vector<int> saturating_matching(const Graph& g);

// Inclusion-minimal nonempty S ⊆ T with no saturating matching (paper's procedure); empty if T saturable.
// Any inclusion-minimal deficient set would satisfy [Lem 7.6]; this one is the set the reference
// (glref.matching.minimal_hall_deficient_set) returns, so core and reference agree exactly.
std::vector<int> minimal_hall_deficient_set(const Graph& g);

}  // namespace glcore
