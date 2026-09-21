// Trace events (docs/paper_notes.md §14). Events are stored as (type, JSON payload string).
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace glcore {

struct Trace {
    bool enabled = false;
    std::vector<std::pair<std::string, std::string>> events;  // (type, json object without the "type" key)
    void record(const std::string& type, const std::string& json_fields) {
        if (enabled) events.emplace_back(type, json_fields);
    }
};

}  // namespace glcore
